import re
import sys


def convert_to_cuda(filepath):
    with open('/home/belisarius/Projects/lbm-miniapp/lbmini/Lbm/OpenMp/Gpu/LbmTube.hpp', 'r') as f:
        content = f.read()

    start_idx = content.find("template<typename Scalar, typename LatticeType>\nLbmTube<Scalar, LatticeType>::LbmTube(")
    content = content[start_idx:]
    content = (
            '#include "Lbm/Cuda/LbmTube.hpp"\n#include "Lbm/Cuda/LatticeD2Q9.hpp"\n#include <iostream>\n\nnamespace lbmini::cuda {\n\n' +
            '__constant__ int kD2Q9Cx[9] = {  0,  1, -1,  0,  0,  1, -1,  1, -1 };\n' +
            '__constant__ int kD2Q9Cy[9] = {  0,  0,  0,  1, -1,  1, -1, -1,  1 };\n\n' +
            content
    )

    content = re.sub(r"omp_set_num_threads\(.*?\);", "", content)
    content = re.sub(r"dev_\s*=\s*omp_get_default_device\(\);", "", content)
    content = re.sub(r"host_\s*=\s*omp_get_initial_device\(\);", "", content)
    
    content = re.sub(r"([a-zA-Z0-9_]+)\s*=\s*static_cast<Scalar\*>\(omp_target_alloc\(([^,]+),\s*dev_\)\);", r"cudaMalloc(&\1, \2);", content)
    content = re.sub(r"if\s*\(!rhoDev_ \|\|.*?\)\s*\{.*?\throw.*?;?\s*\}", "", content, flags=re.DOTALL)
    
    content = re.sub(r"if \(([a-zA-Z0-9_]+)\)\s*omp_target_free\(\1,\s*dev_\);", r"if (\1) cudaFree(\1);", content)
    
    content = re.sub(r"omp_target_memcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*0,\s*0,\s*host_,\s*dev_\);", r"cudaMemcpy(\1, \2, \3, cudaMemcpyDeviceToHost);", content)
    content = re.sub(r"omp_target_memcpy\(([^,]+),\s*([^,]+),\s*([^,]+),\s*0,\s*0,\s*dev_,\s*host_\);", r"cudaMemcpy(\1, \2, \3, cudaMemcpyHostToDevice);", content)

    content = re.sub(r"LBMINI_UNROLL\((\d+)\)", r"#pragma unroll \1", content)
    content = content.replace("} // namespace lbmini::openmp::gpu", "")

    kernels = []
    
    def process_method(method_name):
        nonlocal content
        pattern = r"(template<typename Scalar, typename LatticeType>\s*void\s*LbmTube<Scalar, LatticeType>::" + method_name + r"\(\)\s*\{)(.*?)(^\})"
        m = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if not m: return
        
        body = m.group(2)
        
        loop_pattern = r"#pragma omp target.*?is_device_ptr\(([^)]+)\)\s*for\s*\(int ci = 0; ci < nx; \+\+ci\)\s*\{\s*for\s*\(int cj = 0; cj < ny; \+\+cj\)\s*\{(.*?)^\s*\}\s*^\s*\}"
        loop_m = re.search(loop_pattern, body, re.MULTILINE | re.DOTALL)
        
        if loop_m:
            device_ptrs = [x.strip() for x in loop_m.group(1).split(',')]
            kernel_body = loop_m.group(2)
            pre_pragma = body[:loop_m.start()]
            
            var_defs = re.findall(r"const\s+([a-zA-Z0-9_<>]+)\s+\**([a-zA-Z0-9_]+)\s*=\s*([^;]+);", pre_pragma)
            var_defs += re.findall(r"(?!const)\b([a-zA-Z0-9_<>]+)\s+\**([a-zA-Z0-9_]+)\s*=\s*([^;]+);", pre_pragma)
            var_defs += re.findall(r"const\s+int\s+([a-zA-Z0-9_]+)\s*=\s*([^;]+);", pre_pragma)
            
            kernel_args = []
            kernel_params = []
            
            # First add device pointers
            for ptr in device_ptrs:
                # Find its definition in pre_pragma to see if it's const
                is_const = False
                if f"const Scalar* {ptr}" in pre_pragma or f"const Scalar* {ptr}" in pre_pragma or f"const Scalar* p{ptr[1:]}" in pre_pragma:
                    is_const = True
                
                t = "const Scalar*" if is_const else "Scalar*"
                kernel_args.append(f"{t} {ptr}")
                kernel_params.append(ptr)
            
            # Then add other variables
            added_vars = set(device_ptrs)
            for var in var_defs:
                if len(var) == 3:
                    t, name, val = var
                else:
                    t, name, val = "int", var[0], var[1]
                    
                if name in added_vars: continue
                added_vars.add(name)
                
                t = t.strip()
                if "Scalar" in t and "*" in pre_pragma.split(name)[0][-3:]:
                    continue # Pointer handled above or shouldn't be here
                        
                if t == "int": kernel_args.append(f"int {name}")
                else: kernel_args.append(f"const {t} {name}")
                kernel_params.append(name)
                
            kernel_name = f"{method_name}Kernel"
            
            kernel_code = f"template<typename Scalar>\n__global__ void {kernel_name}({', '.join(kernel_args)}) {{\n"
            kernel_code += "  int ci = blockIdx.x * blockDim.x + threadIdx.x;\n"
            kernel_code += "  int cj = blockIdx.y * blockDim.y + threadIdx.y;\n"
            kernel_code += "  if (ci < nx && cj < ny) {\n"
            kernel_code += kernel_body
            kernel_code += "  }\n}\n"
            
            kernels.append(kernel_code)
            
            launch_code = pre_pragma
            launch_code += f"  dim3 threads(8, 8);\n"
            launch_code += f"  dim3 blocks((nx + threads.x - 1) / threads.x, (ny + threads.y - 1) / threads.y);\n"
            launch_code += f"  {kernel_name}<Scalar><<<blocks, threads>>>({', '.join(kernel_params)});\n"
            
            content = content[:m.start(2)] + launch_code + content[m.end(2):]

    process_method("computeMacroscopic")
    process_method("seedEquilibria")
    process_method("collisionAndEquilibria")
    process_method("streamAndMacroscopic")
    
    content = content.replace("namespace lbmini::cuda {\n\n", "namespace lbmini::cuda {\n\n" + "\n".join(kernels) + "\n\n")
    content += "\n\ntemplate class LbmTube<double, LatticeD2Q9<double>>;\n\n} // namespace lbmini::cuda\n"
    
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    convert_to_cuda(sys.argv[1])

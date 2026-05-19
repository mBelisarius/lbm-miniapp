ARG BASE_IMAGE=nvcr.io/nvidia/nvhpc:26.1-devel-cuda13.1-rockylinux9
FROM ${BASE_IMAGE}

LABEL maintainer="HPC Research Team" \
      description="Lattice Boltzmann Method CFD solver container for cloud HPC clusters (OpenMP, OpenACC, OpenCL, CUDA, Kokkos, RAJA, MPI)" \
      version="1.0"

USER root
WORKDIR /opt/lbm-miniapp

# Build argument to bypass SSL verification if building behind corporate SSL-intercepting proxies/firewalls (default: true)
ARG INSECURE_SSL=true

# Build argument to limit parallel build jobs (default: auto-detected based on available memory and CPU cores)
ARG PARALLEL_JOBS=""

# Build argument selecting the NVIDIA GPU architecture the device backends are
# compiled for. Kokkos and RAJA/BLT accept a single architecture, so this is one
# compute capability, not a list: 80 = A100 (Ampere), 86 = RTX 30xx / A40,
# 89 = RTX 40xx / L40 (Ada), 90 = H100 (Hopper), 100/120 = Blackwell.
# Building for the wrong capability still runs (the driver JITs the embedded
# PTX) but Kokkos warns and performance drops, so override it to match the
# deployment GPU: docker build --build-arg CUDA_ARCHITECTURES=80 ...
ARG CUDA_ARCHITECTURES=89

# 1. Enable EPEL, CRB, and Intel oneAPI repositories for Rocky Linux 9
RUN if [ "$INSECURE_SSL" = "true" ] || [ "$INSECURE_SSL" = "1" ]; then \
        echo "sslverify=0" >> /etc/dnf/dnf.conf; \
        git config --global http.sslVerify false 2>/dev/null || true; \
        DNF_SSL_OPT="--setopt=sslverify=0"; \
    else \
        DNF_SSL_OPT=""; \
    fi && \
    dnf install -y ${DNF_SSL_OPT} epel-release dnf-plugins-core && \
    printf '[oneAPI]\nname=Intel(R) oneAPI repository\nbaseurl=https://yum.repos.intel.com/oneapi\nenabled=1\ngpgcheck=1\nrepo_gpgcheck=1\ngpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB\n' > /etc/yum.repos.d/oneAPI.repo && \
    if [ "$INSECURE_SSL" = "true" ] || [ "$INSECURE_SSL" = "1" ]; then \
        echo "sslverify=0" >> /etc/yum.repos.d/epel.repo 2>/dev/null || true; \
        find /etc/yum.repos.d/ -type f -name "*.repo" -exec sed -i '/^\[/a sslverify=0' {} + 2>/dev/null || true; \
    fi && \
    dnf config-manager --set-enabled crb && \
    dnf update -y ${DNF_SSL_OPT} && \
    dnf install -y ${DNF_SSL_OPT} --allowerasing \
        git \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        libasan \
        libubsan \
        libtsan \
        make \
        ninja-build \
        cmake \
        pkgconf-pkg-config \
        which \
        valgrind \
        gdb \
        tar \
        gzip \
        unzip \
        wget \
        curl \
        opencl-headers \
        ocl-icd \
        ocl-icd-devel \
        intel-oneapi-runtime-opencl \
        intel-oneapi-runtime-dpcpp-sycl-opencl-cpu \
        python3 \
        python3-pip \
        python3-devel && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# 2. Configure OpenCL ICDs for NVIDIA GPUs and CPU (Intel OpenCL CPU Runtime)
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    if [ -f /opt/intel/oneapi/redist/etc/intel64.icd ]; then \
        cp /opt/intel/oneapi/redist/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd; \
    elif [ -f /opt/intel/oneapi/lib/etc/intel64.icd ]; then \
        cp /opt/intel/oneapi/lib/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd; \
    else \
        echo "libintelocl.so" > /etc/OpenCL/vendors/intel64.icd; \
    fi && \
    echo "/opt/intel/oneapi/redist/lib" > /etc/ld.so.conf.d/intel-opencl.conf && \
    echo "/opt/intel/oneapi/lib" >> /etc/ld.so.conf.d/intel-opencl.conf && \
    ldconfig 2>/dev/null || true

# 3. Install Python dependencies for simulation analysis and benchmarking campaigns
RUN if [ "$INSECURE_SSL" = "true" ] || [ "$INSECURE_SSL" = "1" ]; then \
        python3 -m pip install --no-cache-dir --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org && \
        python3 -m pip install --no-cache-dir \
            --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
            numpy \
            scipy \
            matplotlib \
            pyyaml \
            polars; \
    else \
        python3 -m pip install --no-cache-dir --upgrade pip && \
        python3 -m pip install --no-cache-dir \
            numpy \
            scipy \
            matplotlib \
            pyyaml \
            polars; \
    fi

# 4. Configure environment variables for NVHPC, CUDA, OpenMPI/HPC-X, and OpenMP
ENV NVHPC_ROOT=/opt/nvidia/hpc_sdk
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=threads

# 5. Copy project files into container
COPY . /opt/lbm-miniapp

# 6. Pin the NVHPC / HPC-X installation behind stable paths.
#
# The SDK exposes the same release under two directory names (Linux_x86_64/2026
# and Linux_x86_64/26.1) and reaches its MPI through the comm_libs/mpi symlink.
# Hardcoding one name here while the entrypoint discovered another made a binary
# linked against one tree load MCA components from the other, and HPC-X OpenMPI
# then died with `libopen-pal.so: undefined symbol: pmix_framework_names` before
# reaching main(). scripts/nvhpc-env.sh resolves exactly one tree; the symlinks
# below turn that result into paths that ENV (and `apptainer exec`, which never
# runs the entrypoint) can rely on, and ld.so.conf keeps the HPC-X PMIx ahead of
# the Rocky system one.
RUN . /opt/lbm-miniapp/scripts/nvhpc-env.sh && \
    if [ -z "${LBMINI_NVHPC_DIR:-}" ]; then echo "NVIDIA HPC SDK not found under ${NVHPC_ROOT}" >&2; exit 1; fi && \
    ln -sfn "${LBMINI_NVHPC_DIR}" /opt/nvhpc && \
    if [ -n "${LBMINI_MPI_DIR:-}" ]; then \
        ln -sfn "${LBMINI_MPI_DIR}" /opt/nvhpc-mpi; \
    else \
        echo "No MPI installation found in ${LBMINI_NVHPC_DIR}/comm_libs" >&2; exit 1; \
    fi && \
    : > /etc/ld.so.conf.d/000-nvhpc-mpi.conf && \
    for d in "${LBMINI_MPI_DIR}/lib" \
             "$(dirname "${LBMINI_MPI_DIR}")"/pmix/lib \
             "$(dirname "${LBMINI_MPI_DIR}")"/ucx/lib \
             "$(dirname "${LBMINI_MPI_DIR}")"/ucc/lib \
             "$(dirname "${LBMINI_MPI_DIR}")"/hcoll/lib; do \
        if [ -d "${d}" ]; then echo "${d}" >> /etc/ld.so.conf.d/000-nvhpc-mpi.conf; fi; \
    done && \
    ldconfig && \
    printf '. /opt/lbm-miniapp/scripts/nvhpc-env.sh\n' > /etc/profile.d/lbm-miniapp-nvhpc.sh

# Stable, symlink-based toolchain paths (see the step above)
ENV PATH=/opt/nvhpc/compilers/bin:/opt/nvhpc-mpi/bin:/opt/nvhpc/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/nvhpc/compilers/lib:/opt/nvhpc-mpi/lib:/opt/nvhpc/cuda/lib64:/opt/intel/oneapi/redist/lib:/opt/intel/oneapi/lib:${LD_LIBRARY_PATH}
ENV CMAKE_PREFIX_PATH=/opt/nvhpc/cmake
ENV OPAL_PREFIX=/opt/nvhpc-mpi

# 7. Build the LBM mini-app in Release mode with all acceleration backends.
#
# Four trees, deliberately: `build` is the CUDA-enabled Release build used for
# benchmarking and the GPU suite; `build-asan` / `build-tsan` are the sanitized
# host builds; `build-memcheck` is an uninstrumented *CUDA-free* build that the
# Valgrind stage runs. Pointing Valgrind at the CUDA-enabled `build` instead
# dragged the NVIDIA driver through memcheck (33,574 suppressed errors) in a
# stage that must not touch the GPU at all.
#
# `ENABLE_TEST_DEVICE_BACKENDS=ON` in the two trees that may see a device is
# what puts the GPU backends under a memory tool at all: `build` carries the
# CUDA / Kokkos-GPU / RAJA-GPU instantiations of the memory-safety sweep that
# `test gpu` runs under compute-sanitizer, and `build-asan` carries the OpenCL
# GPU one, which compute-sanitizer cannot instrument (it reaches the device
# through the NVIDIA ICD, never through CUDA) and which AddressSanitizer can.
# The GPU-targeted cases skip themselves when no device is visible, so a
# CPU-only container is unaffected - and `test/memory-check.sh` hides the
# devices, keeping the CPU stage CPU-only.
RUN if [ "$INSECURE_SSL" = "true" ] || [ "$INSECURE_SSL" = "1" ]; then \
        git config --global http.sslVerify false; \
        CMAKE_SSL_FLAGS="-DCMAKE_TLS_VERIFY=OFF"; \
    else \
        CMAKE_SSL_FLAGS=""; \
    fi && \
    if [ -n "$PARALLEL_JOBS" ]; then \
        BUILD_JOBS="$PARALLEL_JOBS"; \
    else \
        MEM_JOBS=$(awk '/MemTotal/ {m=int($2/1024/2048); print (m<1?1:m)}' /proc/meminfo 2>/dev/null || echo 2); \
        CPU_JOBS=$(nproc 2>/dev/null || echo 1); \
        if [ "$MEM_JOBS" -lt "$CPU_JOBS" ]; then \
            BUILD_JOBS="$MEM_JOBS"; \
        else \
            BUILD_JOBS="$CPU_JOBS"; \
        fi; \
    fi && \
    cmake -B /opt/lbm-miniapp/build -S /opt/lbm-miniapp \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++ \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DENABLE_CUDA=ON \
        -DENABLE_KOKKOS=ON \
        -DENABLE_KOKKOS_CUDA=ON \
        -DENABLE_RAJA=ON \
        -DENABLE_RAJA_CUDA=ON \
        -DENABLE_TEST_OPENMP_BACKEND=ON \
        -DENABLE_TEST_DEVICE_BACKENDS=ON \
        ${CMAKE_SSL_FLAGS} && \
    cmake --build /opt/lbm-miniapp/build --parallel ${BUILD_JOBS} && \
    cmake -B /opt/lbm-miniapp/build-asan -S /opt/lbm-miniapp \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLBMINI_SANITIZER=address+undefined \
        -DENABLE_CUDA=OFF \
        -DENABLE_TEST_DEVICE_BACKENDS=ON \
        ${CMAKE_SSL_FLAGS} && \
    cmake --build /opt/lbm-miniapp/build-asan --parallel ${BUILD_JOBS} && \
    cmake -B /opt/lbm-miniapp/build-tsan -S /opt/lbm-miniapp \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DLBMINI_SANITIZER=thread \
        -DENABLE_CUDA=OFF \
        -DENABLE_OPENCL=OFF \
        -DENABLE_MPI=OFF \
        -DENABLE_TEST_DEVICE_BACKENDS=OFF \
        ${CMAKE_SSL_FLAGS} && \
    cmake --build /opt/lbm-miniapp/build-tsan --parallel ${BUILD_JOBS} && \
    cmake -B /opt/lbm-miniapp/build-memcheck -S /opt/lbm-miniapp \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DENABLE_CUDA=OFF \
        -DENABLE_TEST_DEVICE_BACKENDS=OFF \
        ${CMAKE_SSL_FLAGS} && \
    cmake --build /opt/lbm-miniapp/build-memcheck --target lbm-miniapp-test --parallel ${BUILD_JOBS} && \
    ln -sf /opt/lbm-miniapp/build/lbm-miniapp-main /usr/local/bin/lbm-miniapp-main && \
    ln -sf /opt/lbm-miniapp/build/lbm-miniapp-test /usr/local/bin/lbm-miniapp-test && \
    chmod +x /opt/lbm-miniapp/entrypoint.sh /opt/lbm-miniapp/test/run-tests.sh /opt/lbm-miniapp/test/memory-check.sh /opt/lbm-miniapp/cmake-build.sh

# 8. Set entrypoint and default command
ENTRYPOINT ["/opt/lbm-miniapp/entrypoint.sh"]
CMD ["lbm-miniapp-main"]

"""Contagem de operações e de tráfego de memória do miniaplicativo.

Extrai os corpos dos kernels diretamente do código-fonte, compila-os com um
tipo escalar instrumentado (roofline-count.cpp) e mede o número de operações
de ponto flutuante por nó e por passo de tempo. O custo de exp e log é
calibrado contando as instruções da glibc sob gdb. Saída: roofline-count.json.

Uso (a partir da raiz do repositório): python thesis/fig/roofline-count.py
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / 'lbmini'
EIGEN = ROOT / 'build' / '_deps' / 'eigen-src'
OUT = HERE / 'roofline-count.json'


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', lambda m: ' ' * len(m.group(0)), text, flags=re.S)
    return re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), text)


def match_brace(text, start):
    """Índice da chave que fecha a chave aberta em text[start]."""
    clean = strip_comments(text)
    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == '{':
            depth += 1
        elif clean[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('chaves desbalanceadas')


def cpu_body(path, name):
    text = path.read_text(encoding='utf-8')
    sig = re.search(r'void LbmTube<Scalar, LatticeType>::%s\(\)\s*\{' % name, text)
    open_i = sig.end() - 1
    return text[open_i + 1:match_brace(text, open_i)]


def cuda_kernel(path, name):
    text = path.read_text(encoding='utf-8')
    kern = re.search(r'__global__ void %s\(' % name, text)
    head = text.rfind('template<', 0, kern.start())
    par = text.index(') {', kern.end())
    open_i = par + 2
    return text[head:match_brace(text, open_i) + 1]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def parse_report(out):
    res = {}
    for line in out.splitlines():
        name, *fields = line.split()
        res[name] = {k: float(v) for k, v in (f.split('=') for f in fields)}
    return res


def calibrate_libm(tmp):
    """FLOP por chamada de exp e log da glibc (FMA = 2), via gdb."""
    exe = tmp / 'libm_probe'
    run(['g++', '-O1', str(HERE / 'roofline-libm.cpp'), '-o', str(exe)])
    out = {}
    for func, args in (('exp', ['0.3', '-1.2', '5.0', '-0.01']), ('log', ['0.3', '1.7', '1e-12', '0.0625'])):
        vals = []
        for a in args:
            o = run(['gdb', '-batch', '-ex', 'break main', '-ex', 'run %s %s' % (func, a),
                     '-ex', 'break %s' % func, '-ex', 'continue',
                     '-x', str(HERE / 'roofline-gdbcount.py'), str(exe)])
            vals.append(int(re.search(r'flop=(\d+)', o).group(1)))
        assert len(set(vals)) == 1, vals
        out[func] = vals[0]
    return out


def executed_flops(tmp):
    """FLOP efetivamente executados por célula no código otimizado (gcc -O3,
    sem vetorização automática), incluindo o interior de exp e log."""
    exe = tmp / 'probe'
    run(['g++', '-std=c++17', '-O3', '-mavx2', '-mfma', '-fno-tree-vectorize', '-fno-tree-slp-vectorize', '-w',
         '-Wl,-z,now', '-I', str(tmp), '-I', str(SRC), '-I', str(EIGEN), str(HERE / 'roofline-count.cpp'),
         '-o', str(exe)])
    out = {}
    for which in ('cpu_collision', 'cpu_stream', 'cpu_streammacro', 'cuda_collision', 'cuda_streammacro'):
        out[which] = {}
        for moving in (0, 1):
            o = run(['gdb', '-batch', '-ex', 'break main', '-ex', 'run probe 1 %d %s' % (moving, which),
                     '-ex', 'break *probe_%s' % which, '-ex', 'continue',
                     '-x', str(HERE / 'roofline-gdbcount.py'), str(exe)])
            m = re.search(r'flop=(\d+) fma=(\d+) arith=(\d+) instr=(\d+)', o)
            out[which]['moving' if moving else 'rest'] = dict(zip(('flop', 'fma', 'arith', 'instr'), map(int, m.groups())))
            print('executado', which, moving, m.group(0), flush=True)
    return out


def main():
    cpu = SRC / 'Lbm' / 'OpenMp' / 'Cpu' / 'LbmTube.hpp'
    cu = SRC / 'Lbm' / 'Cuda' / 'Gpu' / 'LbmTube.cu'
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / 'cpu_collision.inc').write_text(cpu_body(cpu, 'collisionAndEquilibria'))
        (tmp / 'cpu_stream.inc').write_text(cpu_body(cpu, 'stream'))
        (tmp / 'cpu_streammacro.inc').write_text(cpu_body(cpu, 'streamAndMacroscopic'))
        (tmp / 'cuda_collision.inc').write_text(cuda_kernel(cu, 'collisionAndEquilibriaKernel'))
        (tmp / 'cuda_stream.inc').write_text(cuda_kernel(cu, 'streamAndMacroscopicKernel'))
        exe = tmp / 'count'
        common = ['g++', '-std=c++17', '-O1', '-w', '-I', str(tmp), '-I', str(SRC), '-I', str(EIGEN),
                  str(HERE / 'roofline-count.cpp')]
        run(common + ['-DCOUNTING', '-o', str(exe)])
        counts = parse_report(run([str(exe), 'all', '20000']))
        libm = calibrate_libm(tmp)
        executed = executed_flops(tmp)

    result = {'libm_flop': libm, 'source_level': counts, 'executed': executed}
    result['model'] = build_model(counts, libm, executed)
    print(json.dumps(result['model'], indent=2))
    OUT.write_text(json.dumps(result, indent=2))


# Tetos da plataforma-alvo (valores de especificação; não aparecem no texto da tese).
# CPU: 12 núcleos, 2 unidades FMA de 512 bits por núcleo (32 FLOP/ciclo em dupla
# precisão), 4,4 GHz; DDR5-6400 em dois canais (2 x 64 bit x 6400 MT/s).
# GPU: 6144 núcleos FP32 a 2,512 GHz (30,87 TFLOP/s); FP64 a 1/64; GDDR7 de
# 192 bit a 28 Gbit/s por pino.
# GPU hipotética (gpu_1_2): a mesma GPU com FP64 a 1/2 da taxa FP32, razão
# típica dos aceleradores de HPC, mantida a largura de banda.
GPU_FP32_GFLOPS = 6144 * 2 * 2.512
GPU_BW_GBS = 192 / 8 * 28
PLATFORM = {
    'cpu': {'peak_gflops': 12 * 32 * 4.4, 'bw_gbs': 2 * 8 * 6.4},
    'gpu': {'peak_gflops': GPU_FP32_GFLOPS / 64, 'bw_gbs': GPU_BW_GBS},
    'gpu_1_2': {'peak_gflops': GPU_FP32_GFLOPS / 2, 'bw_gbs': GPU_BW_GBS},
}

# Tráfego de DRAM por nó e por passo (bytes), obtido dos acessos dos kernels
# (lbmini/Lbm/OpenMp/Cpu/LbmTube.hpp e lbmini/Lbm/Cuda/Gpu/LbmTube.cu), com
# o domínio muito maior que a cache. Escalares de 8 bytes; índices de 4 bytes.
D = 8
Q = 9


def traffic():
    cpu_coll = {'macro_leitura': 4 * D, 'multiplicadores': 2 * 3 * D, 'f_g': 2 * 2 * Q * D}
    gpu_coll = {'macro_leitura': 4 * D, 'multiplicadores': (2 + 3) * D, 'f_g': 2 * 2 * Q * D}
    cpu_prop = {'f_g_origem': 2 * Q * D, 'indices': 4 * Q * 4, 'pesos': 4 * Q * D, 'f_g_escrita': 2 * Q * D,
                'alocacao_escrita_f_g': 2 * Q * D, 'releitura_f_g': 2 * Q * D, 'macro_escrita': 5 * D,
                'alocacao_escrita_macro': 5 * D}
    gpu_prop = {'f_g_origem': 2 * Q * D, 'f_g_escrita': 2 * Q * D, 'macro_escrita': 5 * D}
    sums = lambda d: sum(d.values())
    return {'cpu': {'colisao': cpu_coll, 'propagacao': cpu_prop,
                    'total': sums(cpu_coll) + sums(cpu_prop)},
            'gpu': {'colisao': gpu_coll, 'propagacao': gpu_prop,
                    'total': sums(gpu_coll) + sums(gpu_prop)}}


def build_model(counts, libm, executed):
    # FLOP: contagem executada (estado em movimento, argumentos genéricos para exp/log).
    ex = lambda k: executed[k]['moving']['flop']
    flop = {
        'cpu': {'colisao': ex('cpu_collision'), 'propagacao': ex('cpu_streammacro')},
        'gpu': {'colisao': ex('cuda_collision'), 'propagacao': ex('cuda_streammacro')},
    }
    ncall = {
        'cpu': {'colisao': (counts['cpu_collision']['exp'], counts['cpu_collision']['log']),
                'propagacao': (counts['cpu_stream+macro']['exp'], counts['cpu_stream+macro']['log'])},
        'gpu': {'colisao': (counts['cuda_collision']['exp'], counts['cuda_collision']['log']),
                'propagacao': (counts['cuda_stream+macro']['exp'], counts['cuda_stream+macro']['log'])},
    }
    tr = traffic()
    out = {'traffic': tr, 'platform': PLATFORM, 'variants': {}}
    for name, key, bytes_ in (('cpu', 'cpu', tr['cpu']['total']), ('gpu', 'gpu', tr['gpu']['total'])):
        stages = {}
        for st in ('colisao', 'propagacao'):
            ne, nl = ncall[key][st]
            total = flop[key][st]
            stages[st] = {'exp': ne, 'log': nl, 'flop': total,
                          'aritmetica': total - libm['exp'] * ne - libm['log'] * nl}
        out['variants'][name] = {'stages': stages, 'flop': sum(s['flop'] for s in stages.values()),
                                 'bytes': bytes_}
    # GPU hipotética com tabelas de propagação: kernel de propagação como o da CPU
    # (sem exp/log), e leitura de índices e pesos, com propagação fundida às macros.
    tab = {'colisao': out['variants']['gpu']['stages']['colisao'], 'propagacao': out['variants']['cpu']['stages']['propagacao']}
    out['variants']['gpu_tabelas'] = {
        'stages': tab,
        'flop': tab['colisao']['flop'] + tab['propagacao']['flop'],
        'bytes': tr['gpu']['total'] + 4 * Q * 4 + 4 * Q * D,
    }
    # Mesmos kernels na GPU hipotética: FLOP e bytes inalterados, muda só o teto.
    for name in ('gpu', 'gpu_tabelas'):
        out['variants'][name + '_1_2'] = dict(out['variants'][name])
    for name, v in out['variants'].items():
        plat = PLATFORM['cpu' if name == 'cpu' else 'gpu_1_2' if name.endswith('_1_2') else 'gpu']
        v['ai'] = v['flop'] / v['bytes']
        v['ridge'] = plat['peak_gflops'] / plat['bw_gbs']
        v['gflops'] = min(plat['peak_gflops'], v['ai'] * plat['bw_gbs'])
        v['regime'] = 'memoria' if v['ai'] < v['ridge'] else 'calculo'
        v['mlups_memoria'] = plat['bw_gbs'] * 1e3 / v['bytes']
        v['mlups_calculo'] = plat['peak_gflops'] * 1e3 / v['flop']
        v['mlups'] = min(v['mlups_memoria'], v['mlups_calculo'])
    return out


if __name__ == '__main__':
    main()

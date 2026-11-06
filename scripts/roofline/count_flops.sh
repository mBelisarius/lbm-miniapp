#!/bin/bash
# Operações de ponto flutuante por célula e passo do núcleo vetorial de CPU, contadas passo a passo no gdb.
#
# Uso (a partir da raiz do repositório, com o executável compilado com AVX2):
#   scripts/roofline/count_flops.sh build/lbm-miniapp-main [ignorar]
# Executa o núcleo com um fluxo em uma malha de 1000 x 10 nós, para no segmento de linha de índice `ignorar`
# (padrão 11, uma linha interior) e soma as operações de ponto flutuante de suas 16 células por bloco.
# Também mede o custo, em operações por via, da exponencial vetorial própria (exp_probe.cpp); o modelo do
# Capítulo 5 substitui esse custo por 20 operações por chamada: N_modelo = total - 8 * exp_via + 8 * 20.
# O resultado alimenta thesis/fig/roofline-final.json.
set -euo pipefail
exe=${1:?executável}
skip=${2:-11}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

python3 - "$root" "$work" <<'PY'
import re
import sys
root, work = sys.argv[1:3]
src = open(f'{root}/scripts/cpu-scaling.py').read()
cfg = re.search(r'CONFIG = """(.*?)"""', src, re.S).group(1)
open(f'{work}/count.yaml', 'w').write(cfg.format(nx=1000, ny=10, tmax=0.02, backend='OpenMP', cores=1, target='CPU'))
PY

sym=$(nm "$exe" | awk '/StepRowILb0E/ && !s {s = $3} END {print s}')
[ -n "$sym" ] || { echo "símbolo StepRow<false> não encontrado em $exe" >&2; exit 1; }
export OMP_NUM_THREADS=1
COUNT_CELLS_ARG=stack gdb -q -batch -ex 'set pagination off' -ex 'set scheduler-locking on' \
  -ex "break $sym" -ex "ignore 1 $skip" -ex "run --config $work/count.yaml --outpath $work/out > /dev/null" \
  -ex "source $here/gdbcount.py" "$exe" 2>&1 | grep -E 'RESULT|rror'

"${CXX:-g++}" -O3 -march=native -ffast-math -std=gnu++20 -w -I"$root/lbmini" "$here/exp_probe.cpp" -o "$work/exp_probe"
unset COUNT_CELLS_ARG
gdb -q -batch -ex 'set pagination off' -ex 'break ExpProbe' -ex 'run' -ex "source $here/gdbcount.py" "$work/exp_probe" 2>&1 |
  grep -E 'RESULT' | sed 's/^RESULT/EXP (por chamada de 4 vias)/'

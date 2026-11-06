"""Executado dentro do gdb (-x): conta as operações de ponto flutuante executadas desde a entrada até o retorno
da função em que o ponto de parada está posicionado.

FMA conta 2 FLOP por via; adição, subtração, multiplicação, divisão e raiz quadrada, 1. Instruções vetoriais
empacotadas contam pelo número de vias (ymm: 4; zmm: 8). Comparações, seleções, máximos e mínimos não contam.
Se a variável de ambiente COUNT_CELLS_ARG for "stack", o número de blocos (oitavo argumento, na pilha) é lido
e a contagem é normalizada por célula (16 células por bloco)."""
import os
import re

import gdb

FMA = re.compile(r'^v?f(n?m(add|sub))\d*[sp]d$')
ARITH = re.compile(r'^v?(add|sub|mul|div|sqrt)[sp]d$')
LONG = gdb.lookup_type('long')

sp0 = int(gdb.parse_and_eval('$sp').cast(LONG))
blocks = None
if os.environ.get('COUNT_CELLS_ARG') == 'stack':
    blocks = int(gdb.parse_and_eval('*(long*)($sp+16)'))
flop = fma = arith = div = instr = 0
fp_instr = 0
while instr < 3000000:
    pc = int(gdb.parse_and_eval('$pc').cast(LONG))
    text = gdb.execute('x/i %d' % pc, to_string=True)
    mnem = re.search(r':\s+(?:bnd\s+|notrack\s+)?(\S+)', text).group(1)
    instr += 1
    lanes = 8 if 'zmm' in text else (4 if 'ymm' in text else (2 if mnem.endswith('pd') else 1))
    if FMA.match(mnem):
        flop += 2 * lanes
        fma += 1
        fp_instr += 1
    elif ARITH.match(mnem):
        flop += lanes
        arith += 1
        fp_instr += 1
        if 'div' in mnem:
            div += lanes
    gdb.execute('stepi', to_string=True)
    if int(gdb.parse_and_eval('$sp').cast(LONG)) > sp0:
        break
out = 'RESULT flop=%d fma=%d arith=%d div_lanes=%d fp_instr=%d instr=%d' % (flop, fma, arith, div, fp_instr, instr)
if blocks:
    cells = blocks * 16
    out += ' blocks=%d cells=%d flop_per_cell=%.1f instr_per_cell=%.1f fp_instr_per_cell=%.1f' % (
        blocks, cells, flop / cells, instr / cells, fp_instr / cells)
print(out)

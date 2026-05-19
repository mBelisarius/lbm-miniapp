"""Executado dentro do gdb (-x): conta as instruções aritméticas de ponto
flutuante executadas desde a entrada até o retorno da função em que o
ponto de parada está posicionado. FMA conta 2 FLOP; add, sub, mul, div e
sqrt contam 1. Instruções vetoriais empacotadas são contadas por instrução
(o caminho rápido da glibc é escalar)."""
import re

import gdb

FMA = re.compile(r'^v?f(n?m(add|sub))\d*[sp]d$')
ARITH = re.compile(r'^v?(add|sub|mul|div|sqrt)[sp]d$')
LONG = gdb.lookup_type('long')

sp0 = int(gdb.parse_and_eval('$sp').cast(LONG))
flop = fma = arith = instr = 0
while instr < 500000:
    pc = int(gdb.parse_and_eval('$pc').cast(LONG))
    text = gdb.execute('x/i %d' % pc, to_string=True)
    mnem = re.search(r':\s+(?:bnd\s+|notrack\s+)?(\S+)', text).group(1)
    instr += 1
    lanes = 8 if 'zmm' in text else (4 if 'ymm' in text else (2 if mnem.endswith('pd') else 1))
    if FMA.match(mnem):
        flop += 2 * lanes
        fma += 1
    elif ARITH.match(mnem):
        flop += lanes
        arith += 1
    gdb.execute('stepi', to_string=True)
    if int(gdb.parse_and_eval('$sp').cast(LONG)) > sp0:
        break
print('RESULT flop=%d fma=%d arith=%d instr=%d' % (flop, fma, arith, instr))

"""Gera a comparação entre as disposições de memória AoS e SoA para a rede D2Q9 (FIGURA 13).

Cada painel mostra três células vizinhas, com as nove funções de distribuição por célula, em
um vetor linear de endereços crescentes. Três fluxos de execução vizinhos (T0, T1, T2), um por
célula, leem a mesma direção (f_1): em AoS os endereços ficam espaçados, em SoA ficam adjacentes.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

W = 8.5
CELLS, Q = 3, 9
BW, BH = 0.3, 0.32
X0 = (W - CELLS * Q * BW) / 2
EDGE = '#333333'
FILL = '#d6e4f0'
HILITE = '#e0893a'
THREAD = '#d9ead3'
FS = 11
DIR = 1


def box_x(k):
    return X0 + k * BW


def panel(ax, y0, layout, caption):
    yb = y0 + 0.8
    hi = []
    for k in range(CELLS * Q):
        if layout == 'aos':
            cell, i = divmod(k, Q)
            lab, on = rf'$f_{i}$', i == DIR
        else:
            i, cell = divmod(k, CELLS)
            lab, on = rf'$c_{cell}$', i == DIR
        if on:
            hi.append(k)
        ax.add_patch(Rectangle((box_x(k), yb), BW, BH, facecolor=HILITE if on else FILL, edgecolor=EDGE,
                               lw=1.3 if on else 1.0, zorder=3))
        ax.text(box_x(k) + BW / 2, yb + BH / 2, lab, ha='center', va='center', fontsize=8.5, zorder=4)

    # chaves dos grupos contíguos
    group = Q if layout == 'aos' else CELLS
    for g in range(CELLS * Q // group):
        xa, xb = box_x(g * group) + 0.03, box_x((g + 1) * group) - 0.03
        ax.plot([xa, xa, xb, xb], [yb - 0.1, yb - 0.17, yb - 0.17, yb - 0.1], color=EDGE, lw=1.1, zorder=3)
        txt = f'célula {g}' if layout == 'aos' else rf'$f_{g}$'
        ax.text((xa + xb) / 2, yb - 0.3, txt, ha='center', va='center', fontsize=FS - 1)

    # fluxos de execução e acessos
    ty = y0 + 2.05
    xs = [box_x(k) + BW / 2 for k in hi]
    for n, x in enumerate(xs):
        ax.add_patch(Circle((x, ty), 0.15, facecolor=THREAD, edgecolor=EDGE, lw=1.2, zorder=4))
        ax.text(x, ty, f'T{n}', ha='center', va='center', fontsize=8.5, zorder=5)
        ax.add_patch(FancyArrowPatch((x, ty - 0.15), (x, yb + BH + 0.02), arrowstyle='-|>', mutation_scale=12,
                                     lw=1.4, color=EDGE, shrinkA=0, shrinkB=0, zorder=4))
    ax.text(xs[-1] + 0.3, ty, 'fluxos de execução\nvizinhos', ha='left', va='center', fontsize=FS - 1, linespacing=1.15)

    ya = y0 + 1.45
    if layout == 'aos':
        ax.add_patch(FancyArrowPatch((xs[0] + 0.05, ya), (xs[1] - 0.05, ya), arrowstyle='<|-|>',
                                     mutation_scale=11, lw=1.2, color=EDGE, shrinkA=0, shrinkB=0, zorder=4))
        ax.text((xs[0] + xs[1]) / 2, ya + 0.16, 'passo de 9 valores', ha='center', va='center',
                fontsize=FS - 1, bbox=dict(facecolor='white', edgecolor='none', pad=1))
        note = 'endereços espaçados:\numa transação por fluxo'
    else:
        note = 'endereços adjacentes:\numa única transação (acesso coalescido)'
    ax.text(xs[-1] + 0.3, ya + 0.1, note, ha='left', va='center', fontsize=FS - 1, linespacing=1.15)

    ax.text(W / 2, y0 + 0.02, caption, ha='center', va='center', fontsize=FS + 1)


def main():
    H = 5.35
    fig = plt.figure(figsize=(W, H), dpi=600, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    panel(ax, 3.1, 'aos', '(a) Vetor de estruturas (AoS)')
    panel(ax, 0.3, 'soa', '(b) Estrutura de vetores (SoA)')
    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

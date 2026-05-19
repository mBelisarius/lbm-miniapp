"""Gera a figura das etapas de colisão e de propagação do LBM em uma rede D2Q9 (FIGURA 8).

Reconstrução em alta resolução, com rótulos em português, da figura equivalente de
Yoo (2022). A geometria é definida em pixels da figura de referência
(lbm-collision-streaming-d2q9.png).
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).with_suffix('.png')

GRAY = '#a0a0a0'
RED = '#ff0000'
FONT = 22
W, H = 740, 430
CELL = 110
TIP = 0.88

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]


def arrow(ax, start, end, color, lw, scale, zorder):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def cell_arrows(ax, center, directions, color, lw, scale, zorder):
    cx, cy = center
    reach = TIP * CELL / 2
    for dx, dy in directions:
        arrow(ax, (cx, cy), (cx + reach * dx, cy - reach * dy), color, lw, scale, zorder)


def panel(ax, origin, highlight):
    ox, oy = origin
    for row in range(3):
        for col in range(3):
            center = (ox + (col + 0.5) * CELL, oy + (row + 0.5) * CELL)
            cell_arrows(ax, center, DIRECTIONS, GRAY, 1.3, 15, 2)
            ax.add_patch(Rectangle((ox + col * CELL, oy + row * CELL), CELL, CELL, fill=False, ec='black', lw=2.4,
                                   zorder=1, joinstyle='miter'))
            offset = (col - 1, 1 - row)
            directions = highlight(offset)
            if directions:
                cell_arrows(ax, center, directions, RED, 3.6, 22, 4)
    ax.add_patch(Rectangle(origin, 3 * CELL, 3 * CELL, fill=False, ec='black', lw=2.4, zorder=3, joinstyle='miter'))


def collision(offset):
    return DIRECTIONS if offset == (0, 0) else []


def streaming(offset):
    return [offset] if offset != (0, 0) else []


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    left, right = (18, 20), (393, 20)
    panel(ax, left, collision)
    panel(ax, right, streaming)
    ax.text(left[0] + 1.5 * CELL, 387, 'Colisão', ha='center', va='center', fontsize=FONT)
    ax.text(right[0] + 1.5 * CELL, 387, 'Propagação', ha='center', va='center', fontsize=FONT)
    fig.savefig(OUT, dpi=300, facecolor='white')
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

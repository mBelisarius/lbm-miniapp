"""Draws the 1D MPI domain decomposition with ghost-cell exchange of the implementation chapter.

Usage: python mpi-domain-decomposition.py
Writes mpi-domain-decomposition.png next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

GHOST_COLOR = '#f3e9d2'
EDGE_COLOR = '#5b8db8'
INNER_COLOR = '#d6e4f0'
LINE = '#333333'
RIGHT_COLOR = '#2e6da4'
LEFT_COLOR = '#c4622d'

N_RANKS = 4
ROWS = 4
# Columns of a subdomain: 2 ghosts, 2 boundary, 2 inner, 2 boundary, 2 ghosts (ghosts_ = 2 in the code).
LAYOUT = ['G', 'G', 'B', 'B', 'I', 'I', 'B', 'B', 'G', 'G']
WIDTH = len(LAYOUT)
GAP = 4.0
COLORS = {'G': GHOST_COLOR, 'B': EDGE_COLOR, 'I': INNER_COLOR}


def x0(rank):
    return rank * (WIDTH + GAP)


def draw_rank(ax, rank):
    for c, kind in enumerate(LAYOUT):
        # The outermost ranks have no neighbor on the tube ends, so no ghost cells are drawn there.
        if (rank == 0 and c < 2) or (rank == N_RANKS - 1 and c >= WIDTH - 2):
            continue
        for r in range(ROWS):
            ax.add_patch(Rectangle((x0(rank) + c, r), 1, 1, fc=COLORS[kind], ec=LINE, lw=0.8,
                                   hatch='////' if kind == 'G' else None, zorder=2))
    left = x0(rank) + (2 if rank == 0 else 0)
    right = x0(rank) + WIDTH - (2 if rank == N_RANKS - 1 else 0)
    ax.add_patch(Rectangle((left, 0), right - left, ROWS, fc='none', ec=LINE, lw=2.0, zorder=4))
    ax.text((left + right) / 2, -2.9, f'Processo {rank}', ha='center', va='center', fontsize=11,
            fontweight='bold')


def arc(ax, p0, p1, color, rad):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
                                 mutation_scale=16, lw=1.8, color=color, zorder=6))


def main():
    plt.rcParams['hatch.linewidth'] = 0.6
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_aspect('equal')
    ax.axis('off')

    for rank in range(N_RANKS):
        draw_rank(ax, rank)

    # Exchange between each pair of neighbors: send boundary columns, receive into ghost columns.
    for rank in range(N_RANKS - 1):
        a, b = x0(rank), x0(rank + 1)
        # Rightward: right boundary cells of `rank` -> left ghost cells of `rank + 1`.
        arc(ax, (a + WIDTH - 3, ROWS + 0.1), (b + 1, ROWS + 0.1), RIGHT_COLOR, -0.35)
        # Leftward: left boundary cells of `rank + 1` -> right ghost cells of `rank`.
        arc(ax, (b + 3, -0.1), (a + WIDTH - 1, -0.1), LEFT_COLOR, -0.35)

    mid = (x0(1) + WIDTH + x0(2)) / 2
    ax.text(mid, ROWS + 3.0, 'Envio de $f$ e $g$ ao vizinho da direita', ha='center', va='center',
            fontsize=10, color=RIGHT_COLOR)
    ax.text(mid, -4.9, 'Envio de $f$ e $g$ ao vizinho da esquerda', ha='center', va='center',
            fontsize=10, color=LEFT_COLOR)

    # Tube ends
    for xe, label in ((x0(0) + 2, 'Parede'), (x0(N_RANKS - 1) + WIDTH - 2, 'Parede')):
        ax.text(xe, ROWS + 0.7, label, ha='center', va='center', fontsize=9, color='#444444')

    # Longitudinal axis
    y_axis = -6.6
    ax.add_patch(FancyArrowPatch((x0(0) + 2, y_axis), (x0(N_RANKS - 1) + WIDTH - 2, y_axis),
                                 arrowstyle='-|>', mutation_scale=16, lw=1.4, color=LINE))
    ax.text(x0(N_RANKS - 1) + WIDTH - 2, y_axis - 0.9, 'Eixo longitudinal $x$', ha='right', va='center',
            fontsize=10)

    # Legend
    items = [(GHOST_COLOR, '////', 'Células fantasma (recebidas)'),
             (EDGE_COLOR, None, 'Células de fronteira (enviadas, calculadas antes da troca)'),
             (INNER_COLOR, None, 'Células internas (calculadas durante a troca)')]
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec=LINE, lw=0.8, hatch=h) for c, h, _ in items]
    ax.legend(handles, [t for _, _, t in items], loc='upper center', bbox_to_anchor=(mid, -8.4),
              ncol=1, frameon=False, fontsize=10, bbox_transform=ax.transData, borderaxespad=0)

    ax.set_xlim(-1, x0(N_RANKS - 1) + WIDTH + 1)
    ax.set_ylim(-13.0, ROWS + 4)

    out = Path(__file__).with_name('mpi-domain-decomposition.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

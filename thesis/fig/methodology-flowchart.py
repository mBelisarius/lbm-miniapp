"""Draws the methodology flowchart of the methodology chapter.

Usage: python methodology-flowchart.py
Writes methodology-flowchart.png next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

EDGE = '#333333'
COLORS = ['#d6e4f0', '#d6e4f0', '#f3e9d2', '#d6e4f0', '#d9ead3']

STEPS = [
    ('1. Seleção do modelo', 'Dupla distribuição com redes deslocadas'),
    ('2. Implementação sequencial de referência', 'Comportamento numérico esperado e base da aceleração'),
    ('3. Verificação e validação', 'Comparação com a solução analítica de Sod'),
    ('4. Reimplementação paralela', 'Uma versão para cada tecnologia de paralelização'),
    ('5. Campanha de medições', 'Desempenho, escalabilidade e complexidade de código'),
]

X, W, H, GAP = 4, 72, 11, 6


def arrow(ax, p0, p1, style='-|>'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18, lw=1.8, color=EDGE,
                                 shrinkA=0, shrinkB=0, zorder=5))


def main():
    n = len(STEPS)
    top = n * H + (n - 1) * GAP
    fig, ax = plt.subplots(figsize=(8, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, top + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    centers = []
    for i, (title, subtitle) in enumerate(STEPS):
        y = top - (i + 1) * H - i * GAP
        ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle='round,pad=0,rounding_size=1.2',
                                    fc=COLORS[i], ec=EDGE, lw=1.4, zorder=2))
        ax.text(X + W / 2, y + H * 0.64, title, ha='center', va='center', fontsize=11.5,
                fontweight='bold', zorder=3)
        ax.text(X + W / 2, y + H * 0.30, subtitle, ha='center', va='center', fontsize=9.5, zorder=3)
        centers.append(y + H / 2)
        if i:
            arrow(ax, (X + W / 2, y + H + GAP), (X + W / 2, y + H))

    # Return arrow: each new version is validated before being measured
    x_right = X + W
    x_loop = x_right + 8
    ax.plot([x_right, x_loop, x_loop], [centers[3], centers[3], centers[2]], color=EDGE, lw=1.8,
            solid_capstyle='butt', zorder=5)
    arrow(ax, (x_loop, centers[2]), (x_right + 0.2, centers[2]))
    ax.text(x_loop + 2.5, (centers[2] + centers[3]) / 2, 'Cada nova versão é\nvalidada antes de ser medida',
            ha='left', va='center', fontsize=9.5, style='italic')

    out = Path(__file__).with_name('methodology-flowchart.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

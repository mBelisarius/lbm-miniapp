"""Gera a figura da rede D2Q9 padrão e da rede deslocada no espaço de velocidades (FIGURA 10).

Reconstrução em alta resolução, com rótulos em português, da figura equivalente de
Rathnayake (2024). Cada painel mostra as velocidades discretas c_i = v_i + U_ref a partir
da origem do espaço de velocidades, com a janela centrada no centro da rede.
Para que a figura seja legível em impressão em preto e branco, a direção 0 (c_0 = U_ref) é
destacada por um ponto com contorno preto em (a) e por uma seta espessa com contorno preto em (b),
além da cor.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

RED = '#ff2a2a'
HALF = 1.38
GRID = '#e4e4e4'

STANDARD = {
    1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1),
    5: (1, 1), 6: (-1, 1), 7: (-1, -1), 8: (1, -1),
}


def arrow(ax, end, color='black', lw=2.1, scale=20, zorder=3, outline=False):
    patch = FancyArrowPatch((0, 0), end, arrowstyle='-|>', mutation_scale=scale, lw=lw, color=color,
                            shrinkA=0, shrinkB=0, zorder=zorder)
    if outline:
        patch.set_path_effects([pe.Stroke(linewidth=lw + 2.6, foreground='black'), pe.Normal()])
    ax.add_patch(patch)


def panel(ax, shift, caption):
    ux, uy = shift
    ax.set_xlim(ux - HALF, ux + HALF)
    ax.set_ylim(uy - HALF, uy + HALF)
    ax.set_aspect('equal')
    ax.axis('off')

    for g in np.arange(-3, 3.01, 0.25):
        ax.plot([g, g], [-3, 3], color=GRID, lw=0.8, zorder=0)
        ax.plot([-3, 3], [g, g], color=GRID, lw=0.8, zorder=0)

    for i, (vx, vy) in STANDARD.items():
        cx, cy = vx + ux, vy + uy
        arrow(ax, (cx, cy))
        norm = np.hypot(cx, cy)
        dx, dy = cx / norm, cy / norm
        ax.text(cx + 0.2 * dx, cy + 0.2 * dy, str(i), ha='center', va='center', fontsize=16, zorder=4)

    if ux or uy:
        arrow(ax, (ux, uy), color=RED, lw=3.2, scale=19, zorder=5, outline=True)
        ax.plot([0], [0], marker='o', ms=7, color='black', zorder=6)
    else:
        ax.plot([0], [0], marker='o', ms=9.5, mfc=RED, mec='black', mew=1.8, zorder=6)
    ax.text(ux + 0.32, uy - 0.17, '0', ha='center', va='center', fontsize=16, zorder=4)

    ax.text(ux, uy - HALF - 0.12, caption, ha='center', va='top', fontsize=16)


def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.6), dpi=600, facecolor='white')
    panel(ax_left, (0.0, 0.0), r'(a) $\mathbf{U}_{ref} = (0{,}0;\ 0{,}0)$')
    panel(ax_right, (0.5, 0.0), r'(b) $\mathbf{U}_{ref} = (0{,}5;\ 0{,}0)$')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08)
    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

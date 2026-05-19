"""Gera a figura da rede em movimento e da rede estacionária (FIGURA 11).

Reconstrução em alta resolução, com a notação do texto, da figura equivalente de
Rathnayake (2024). As coordenadas estão em unidades do espaçamento da malha, com origem no
nó "a" da rede estacionária; os nós da rede em movimento estão deslocados de U_ref em x.
Para que a figura seja legível em impressão em preto e branco, a rede estacionária usa linhas
contínuas e nós circulares, e a rede em movimento, linhas pontilhadas e nós quadrados.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

BLUE = '#1f1fff'
BLACK = '#111111'
SHIFT = 0.375
WALL_X = -0.5
XMIN, XMAX = -1.0, 2.55
YMIN, YMAX = -1.2, 1.2
FS = 18


def main():
    fig, ax = plt.subplots(figsize=(7.4, 5.1), dpi=600, facecolor='white')
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect('equal')
    ax.axis('off')

    for k in range(-3, 4):
        y = 0.5 * k
        ax.plot([WALL_X - 0.49, WALL_X], [y - 0.48, y], color=BLACK, lw=1.6, zorder=1)
    ax.plot([WALL_X, WALL_X], [YMIN, YMAX], color=BLACK, lw=5.5, solid_capstyle='butt', zorder=2)

    rows = (-1, 0, 1)
    for y in rows:
        ax.plot([WALL_X, XMAX], [y, y], color=BLACK, lw=1.6, zorder=2)
    for i in range(3):
        ax.plot([i, i], [YMIN, YMAX], color=BLACK, lw=1.6, zorder=2)
        ax.plot([i + SHIFT, i + SHIFT], [YMIN, YMAX], color=BLACK, lw=3.2, ls=(0, (0.1, 2.0)),
                dash_capstyle='round', zorder=2)
        for y in rows:
            ax.plot(i, y, 'o', ms=9, color=BLACK, zorder=4)
            ax.plot(i + SHIFT, y, 's', ms=9, color=BLUE, mec=BLACK, mew=1.0, zorder=4)

    ax.add_patch(FancyArrowPatch((0.0, 0.5), (SHIFT, 0.5), arrowstyle='<|-|>', mutation_scale=15, lw=1.6,
                                 color=BLACK, shrinkA=0, shrinkB=0, zorder=5))
    ax.text(SHIFT / 2, 0.58, r'$U_{ref}$', ha='center', va='bottom', fontsize=FS - 2)

    ax.text(SHIFT + 0.09, 0.14, r"$f'_{ia}$", color=BLUE, ha='left', va='bottom', fontsize=FS + 2, zorder=5)
    ax.text(1.0 + 0.09, 0.14, r'$f_i^{-}$', color=BLACK, ha='left', va='bottom', fontsize=FS + 2, zorder=5)
    ax.text(1.0 + SHIFT + 0.09, 0.14, r"$f'_{ib}$", color=BLUE, ha='left', va='bottom', fontsize=FS + 2, zorder=5)

    ax.text(-0.04, -0.1, r'$a$', ha='right', va='top', fontsize=FS + 2)
    ax.text(SHIFT + 0.1, -0.1, r"$a'$", ha='left', va='top', fontsize=FS + 2)
    ax.text(1.0 + 0.03, -0.1, r'$b$', ha='left', va='top', fontsize=FS + 2)
    ax.text(1.0 + SHIFT + 0.1, -0.1, r"$b'$", ha='left', va='top', fontsize=FS + 2)

    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

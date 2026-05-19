"""Gera a figura da decomposição do espaço de momentos e da regularização (FIGURA 9).

Diagrama conceitual: a parcela de não equilíbrio f^neq é decomposta em uma componente no
subespaço hidrodinâmico (tensor de tensões) e outra no subespaço dos momentos fantasma;
a regularização preserva apenas a primeira.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

EDGE = '#333333'
HYDRO = '#2f6ea5'
GHOST = '#c4622d'
FADED = '#6e6e6e'

TIP = (1.0, 0.72)
X_END, Y_END = 1.5, 1.05


def arrow(ax, start, end, color, lw=2.0, scale=18, ls='-', zorder=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, lw=lw, color=color,
                                 ls=ls, shrinkA=0, shrinkB=0, zorder=zorder))


def frame(ax, title):
    ax.set_xlim(-0.35, 1.7)
    ax.set_ylim(-0.55, 1.12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.8, -0.47, title, ha='center', va='center', fontsize=14)
    arrow(ax, (0, 0), (X_END, 0), EDGE, lw=1.3, scale=15, zorder=1)
    arrow(ax, (0, 0), (0, Y_END), EDGE, lw=1.3, scale=15, zorder=1)
    ax.text(X_END, -0.1, 'Subespaço\nhidrodinâmico', ha='right', va='top', fontsize=11, color=HYDRO)
    ax.text(0.05, Y_END, 'Subespaço\nfantasma', ha='left', va='center', fontsize=11, color=GHOST)


def before(ax):
    frame(ax, '(a) Antes da regularização')
    ax.plot([TIP[0], TIP[0]], [0, TIP[1]], color=FADED, lw=1.2, ls=(0, (3, 3)), zorder=1)
    ax.plot([0, TIP[0]], [TIP[1], TIP[1]], color=FADED, lw=1.2, ls=(0, (3, 3)), zorder=1)
    arrow(ax, (0, 0), (TIP[0], 0), HYDRO, lw=3.2, scale=20)
    arrow(ax, (0, 0), (0, TIP[1]), GHOST, lw=3.2, scale=20)
    arrow(ax, (0, 0), TIP, 'black', lw=2.4, scale=20, zorder=4)
    ax.text(TIP[0] + 0.05, TIP[1] + 0.02, r'$\mathbf{f}^{\,neq}$', ha='left', va='bottom', fontsize=15)
    ax.text(TIP[0] / 2, 0.06, r'$\mathbf{f}^{\,neq}_{\,hid}$', ha='center', va='bottom', fontsize=14, color=HYDRO)
    ax.text(-0.06, TIP[1] / 2, r'$\mathbf{f}^{\,neq}_{\,fant}$', ha='right', va='center', fontsize=14, color=GHOST)


def after(ax):
    frame(ax, '(b) Após a regularização')
    arrow(ax, (0, 0), TIP, FADED, lw=1.6, scale=16, ls=(0, (4, 3)), zorder=2)
    ax.text(TIP[0] + 0.05, TIP[1] + 0.02, r'$\mathbf{f}^{\,neq}$', ha='left', va='bottom', fontsize=15, color=FADED)
    arrow(ax, (TIP[0], TIP[1] - 0.03), (TIP[0], 0.06), FADED, lw=1.3, scale=14, zorder=2)
    ax.text(TIP[0] + 0.05, TIP[1] / 2, 'projeção', ha='left', va='center', fontsize=11, color='#666666')
    arrow(ax, (0, 0), (TIP[0], 0), HYDRO, lw=3.2, scale=20)
    ax.text(TIP[0] / 2, 0.06, r'$\mathbf{f}^{\,neq,\,reg}$', ha='center', va='bottom', fontsize=14, color=HYDRO)
    ax.plot([0], [0], marker='o', ms=7, mfc='white', mec=GHOST, mew=2.0, zorder=5)
    ax.text(-0.06, 0.0, r'$\mathbf{f}^{\,neq}_{\,fant}=0$', ha='right', va='center', fontsize=14, color=GHOST)


def main():
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.3), dpi=300, facecolor='white')
    before(ax_left)
    after(ax_right)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05)
    fig.savefig(OUT, dpi=300, facecolor='white')
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

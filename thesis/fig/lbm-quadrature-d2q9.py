"""Draws the construction of the D2Q9 lattice from the three-point Gauss-Hermite rule.

Usage: python lbm-quadrature-d2q9.py
Writes lbm-quadrature-d2q9.png next to this script.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

EDGE = '#333333'
SQRT3 = np.sqrt(3.0)
WEIGHT_COLORS = {'4/9': '#4f81a8', '1/9': '#9dbfd9', '1/36': '#dbe7f1'}
GAUSS_COLOR = '#d6e4f0'

plt.rcParams['mathtext.fontset'] = 'dejavusans'


def gaussian(x):
    return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


def one_dimensional(ax):
    x = np.linspace(-3.2, 3.2, 400)
    ax.fill_between(x, gaussian(x), color=GAUSS_COLOR, zorder=1)
    ax.plot(x, gaussian(x), color=EDGE, lw=1.6, zorder=2)
    ax.annotate(r'$\varpi(\xi)$', xy=(1.05, gaussian(1.05)), xytext=(1.6, 0.3), fontsize=13,
                arrowprops=dict(arrowstyle='-', color=EDGE, lw=1.0), ha='left', va='center')

    points = (-SQRT3, 0.0, SQRT3)
    weights = ((1, 6), (2, 3), (1, 6))
    for xi, (num, den) in zip(points, weights):
        ax.plot([xi, xi], [0, gaussian(xi)], color=EDGE, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.plot(xi, 0, 'o', ms=9, mfc='#4f81a8', mec=EDGE, mew=1.2, zorder=4, clip_on=False)
        ax.text(xi, -0.2, rf'$w = \frac{{{num}}}{{{den}}}$', ha='center', va='center', fontsize=13,
                transform=ax.get_xaxis_transform())

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 0.46)
    ax.set_xticks(points)
    ax.set_xticklabels([r'$-\sqrt{3}$', '$0$', r'$+\sqrt{3}$'], fontsize=13)
    ax.tick_params(axis='x', length=4, pad=4)
    ax.set_yticks([])
    for side in ('left', 'right', 'top'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.4)
    ax.text(3.3, 0, r'$\xi$', ha='left', va='center', fontsize=14, clip_on=False)
    ax.set_title('Regra unidimensional de três pontos', fontsize=12, pad=10)


def two_dimensional(ax):
    grid = (-1, 0, 1)
    for i in grid:
        ax.plot([i * SQRT3] * 2, [-SQRT3 * 1.0, SQRT3 * 1.0], color='#999999', lw=0.9, zorder=1)
        ax.plot([-SQRT3 * 1.0, SQRT3 * 1.0], [i * SQRT3] * 2, color='#999999', lw=0.9, zorder=1)

    for i in grid:
        for j in grid:
            n = abs(i) + abs(j)
            weight = {0: '4/9', 1: '1/9', 2: '1/36'}[n]
            x, y = i * SQRT3, j * SQRT3
            ax.plot(x, y, 'o', ms=14, mfc=WEIGHT_COLORS[weight], mec=EDGE, mew=1.3, zorder=3)
            ax.text(x + 0.3, y + 0.32, weight, ha='left', va='bottom', fontsize=12, fontweight='bold',
                    zorder=4)
            ax.text(x + 0.3, y - 0.3, f'$({i},{j})$'.replace('-', '\u2212'), ha='left', va='top',
                    fontsize=10, color='#444444', zorder=4)

    ax.set_xlim(-2.9, 3.4)
    ax.set_ylim(-2.9, 2.9)
    ax.set_aspect('equal')
    ticks = [-SQRT3, 0, SQRT3]
    labels = [r'$-\sqrt{3}$', '$0$', r'$+\sqrt{3}$']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(r'$\xi_x$', fontsize=13, loc='right')
    ax.set_ylabel('')
    ax.text(-0.04, 1.0, r'$\xi_y$', transform=ax.transAxes, ha='right', va='bottom', fontsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.set_title('Rede D2Q9 resultante', fontsize=12, pad=10)


def main():
    fig = plt.figure(figsize=(12, 5.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.42, 1.0], wspace=0.08)
    ax_left = fig.add_subplot(grid[0])
    ax_mid = fig.add_subplot(grid[1])
    ax_right = fig.add_subplot(grid[2])

    one_dimensional(ax_left)
    ax_left.set_box_aspect(0.74)
    two_dimensional(ax_right)

    ax_mid.axis('off')
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.add_patch(FancyArrowPatch((0.12, 0.5), (0.72, 0.5), arrowstyle='-|>', mutation_scale=22, lw=2.0,
                                     color=EDGE))
    ax_mid.text(0.42, 0.58, 'produto\ntensorial', ha='center', va='bottom', fontsize=11)

    out = Path(__file__).with_name('lbm-quadrature-d2q9.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

"""Gera a figura das escalas de descrição de um fluido: microscópica, mesoscópica e macroscópica.

Reconstrução, com rótulos em português, da figura equivalente de Saito, Abe e Koyama (2017): partículas
individuais (dinâmica molecular), funções de distribuição sobre a rede D2Q9 (método de Lattice Boltzmann) e
escoamento contínuo em um canal (equações de Navier-Stokes). A geometria é definida em unidades arbitrárias
do quadro de desenho.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

OUT = Path(__file__).with_suffix('.png')

GRAY = '#9a9a9a'
RED = '#d62728'
FONT = 14
W, H = 740, 250

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]


def arrow(ax, start, end, color='black', lw=1.6, scale=13, zorder=3, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=zorder, connectionstyle=f'arc3,rad={rad}'))


def molecular_panel(ax, x0, y0, size):
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec='black', lw=2.0, zorder=2))
    rng = np.random.default_rng(7)
    centers = [(0.22, 0.78), (0.70, 0.80), (0.48, 0.52), (0.22, 0.28), (0.74, 0.30), (0.50, 0.20)]
    for cx, cy in centers:
        angle = rng.uniform(0, 2 * np.pi)
        px, py = x0 + cx * size, y0 + cy * size
        ax.add_patch(Circle((px, py), 5.5, fc='#8c8c8c', ec='black', lw=1.0, zorder=4))
        length = 34
        arrow(ax, (px, py), (px + length * np.cos(angle), py + length * np.sin(angle)), lw=1.5, scale=11)


def lattice_panel(ax, x0, y0, cell):
    reach = 0.88 * cell / 2
    for row in range(3):
        for col in range(3):
            cx, cy = x0 + (col + 0.5) * cell, y0 + (row + 0.5) * cell
            central = (row, col) == (1, 1)
            for dx, dy in DIRECTIONS:
                arrow(ax, (cx, cy), (cx + reach * dx, cy + reach * dy), RED if central else GRAY,
                      lw=2.6 if central else 1.1, scale=16 if central else 9, zorder=5 if central else 3)
            ax.add_patch(Rectangle((x0 + col * cell, y0 + row * cell), cell, cell, fill=False, ec='#6e6e6e',
                                   lw=1.0, zorder=1))
    ax.add_patch(Rectangle((x0, y0), 3 * cell, 3 * cell, fill=False, ec='black', lw=2.0, zorder=2))


def continuum_panel(ax, x0, y0, size):
    wall = 7
    for y in (y0 - wall, y0 + size + wall):
        ax.plot([x0 - 20, x0 + size + 20], [y, y], color='black', lw=1.4, zorder=2)
        for k in np.arange(x0 - 20, x0 + size + 20, 9):
            ax.plot([k, k + 7], [y, y + (7 if y < y0 else -7)], color='#555555', lw=0.8, zorder=2)
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec='black', lw=2.0, zorder=2))
    for k, level in enumerate(np.linspace(0.12, 0.88, 6)):
        profile = 1 - (2 * level - 1) ** 2
        arrow(ax, (x0 - 14, y0 + level * size), (x0 - 14 + 14 + 38 * profile, y0 + level * size), lw=1.3, scale=10)
    for cy in (0.74, 0.28):
        arrow(ax, (x0 + 0.58 * size, y0 + cy * size), (x0 + 0.92 * size, y0 + (cy + 0.1) * size), lw=1.3, scale=10,
              rad=-0.9)
        arrow(ax, (x0 + 0.78 * size, y0 + (cy - 0.12) * size), (x0 + 0.52 * size, y0 + (cy - 0.04) * size),
              lw=1.3, scale=10, rad=-0.9)


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    size = 130
    top = 70
    mol = (55, top)
    lat_cell = 43
    lat = (298, top + 0.5 * (size - 3 * lat_cell))
    con = (570, top)

    molecular_panel(ax, mol[0], mol[1], size)
    lattice_panel(ax, lat[0], lat[1], lat_cell)
    continuum_panel(ax, con[0], con[1], size)

    # Linhas de ampliação entre os painéis (do trecho central da escala menor à moldura da seguinte).
    zoom = 38
    sx, sy = mol[0] + size / 2, mol[1] + size / 2
    inner = (sx - zoom / 2, sy - zoom / 2, sx + zoom / 2, sy + zoom / 2)
    ax.add_patch(Rectangle((inner[0], inner[1]), zoom, zoom, fill=False, ec='black', lw=1.2, zorder=3))
    ax.plot([inner[2], lat[0]], [inner[1], lat[1]], color='black', lw=1.1, zorder=1)
    ax.plot([inner[2], lat[0]], [inner[3], lat[1] + 3 * lat_cell], color='black', lw=1.1, zorder=1)
    cx0, cy0 = lat[0] + lat_cell, lat[1] + lat_cell
    ax.plot([cx0 + lat_cell, con[0]], [cy0, con[1]], color='black', lw=1.1, zorder=1)
    ax.plot([cx0 + lat_cell, con[0]], [cy0 + lat_cell, con[1] + size], color='black', lw=1.1, zorder=1)

    ax.text(mol[0] + size / 2, top + size + 22, 'Dinâmica molecular', ha='center', va='center', fontsize=FONT)
    ax.text(lat[0] + 1.5 * lat_cell, top + size + 22, 'Lattice Boltzmann', ha='center', va='center',
            fontsize=FONT, style='italic')
    ax.text(con[0] + size / 2, top + size + 22, 'Navier–Stokes', ha='center', va='center', fontsize=FONT)

    base = 40
    arrow(ax, (W / 2, base), (24, base), lw=1.8, scale=16)
    arrow(ax, (W / 2, base), (W - 24, base), lw=1.8, scale=16)
    ax.text(mol[0] + size / 2, base - 24, 'Microscópica', ha='center', va='center', fontsize=FONT)
    ax.text(lat[0] + 1.5 * lat_cell, base - 24, 'Mesoscópica', ha='center', va='center', fontsize=FONT, style='italic')
    ax.text(con[0] + size / 2, base - 24, 'Macroscópica', ha='center', va='center', fontsize=FONT)

    fig.savefig(OUT, dpi=300, facecolor='white')
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

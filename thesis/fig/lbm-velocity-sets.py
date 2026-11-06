"""Gera a figura dos conjuntos de velocidades discretas D2Q9, D3Q19 e D3Q27 (FIGURA 7).

Reconstrução em alta resolução, com rótulos e numeração das direções, baseada nas
figuras equivalentes de Krüger et al. (2017). A geometria é definida em pixels das
figuras de referência (lbm-velocities-d2q9.png e lbm-velocities-d3q19.png). O painel
do D3Q27 reutiliza a projeção do D3Q19 e acrescenta as oito direções diagonais dos
vértices do cubo, com os rótulos posicionados além da ponta de cada seta.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).with_suffix('.png')

AXIAL = '#111111'
DIAG = '#6e6e6e'
FONT = 18
TIP = 0.98
SHIFT_B = 470
SHIFT_C = 1010
W, H = 1010 + 510, 460

D2_ORIGIN = (162, 250)
D2_HALF = 125.5
D3_ORIGIN = (470 + 199, 212.7)
D3_HALF = 119.5
D3_Z = (-30.3, 48.2)

D2Q9 = {
    1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1),
    5: (1, 1), 6: (-1, 1), 7: (-1, -1), 8: (1, -1),
}
D2Q9_LABELS = {
    1: (305, 250), 2: (162, 102), 3: (17, 250), 4: (162, 398),
    5: (305, 101), 6: (18, 101), 7: (18, 396), 8: (305, 398),
}

D3Q19 = {
    1: (1, 0, 0), 2: (-1, 0, 0), 3: (0, 1, 0), 4: (0, -1, 0), 5: (0, 0, 1), 6: (0, 0, -1),
    7: (1, 1, 0), 8: (-1, -1, 0), 9: (1, 0, 1), 10: (-1, 0, -1), 11: (0, 1, 1), 12: (0, -1, -1),
    13: (1, -1, 0), 14: (-1, 1, 0), 15: (1, 0, -1), 16: (-1, 0, 1), 17: (0, 1, -1), 18: (0, -1, 1),
}
D3Q19_LABELS = {
    1: (334, 213), 2: (63, 213), 3: (198, 74), 4: (199, 352), 5: (152, 278), 6: (245, 147),
    7: (360, 86), 8: (62, 332), 9: (304, 260), 10: (87, 166), 11: (146, 123), 12: (252, 303),
    13: (341, 332), 14: (57, 94), 15: (372, 166), 16: (27, 260), 17: (229, 27), 18: (170, 399),
}
D3_RECTANGLES = (
    (14, 7, 13, 8),
    (10, 15, 9, 16),
    (17, 11, 18, 12),
)

D3Q27 = dict(D3Q19)
D3Q27.update({
    19: (1, 1, 1), 20: (-1, -1, -1), 21: (1, 1, -1), 22: (-1, -1, 1),
    23: (1, -1, 1), 24: (-1, 1, -1), 25: (-1, 1, 1), 26: (1, -1, -1),
})


def arrow(ax, start, end, color, lw=1.7, scale=20):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3))


def label(ax, xy, text, color='black', fontsize=FONT, halo=False):
    bbox = dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.9) if halo else None
    ax.text(*xy, text, ha='center', va='center', fontsize=fontsize, color=color, zorder=4, bbox=bbox)


def triad(ax, origin, length, depth=None):
    ox, oy = origin
    arrow(ax, (ox, oy), (ox + length, oy), AXIAL, lw=1.7, scale=16)
    arrow(ax, (ox, oy), (ox, oy - length), AXIAL, lw=1.7, scale=16)
    label(ax, (ox + length + 18, oy), 'x')
    label(ax, (ox, oy - length - 16), 'y')
    if depth is not None:
        arrow(ax, (ox, oy), (ox + depth[0], oy + depth[1]), AXIAL, lw=1.7, scale=16)
        label(ax, (ox + depth[0] - 12, oy + depth[1] + 16), 'z')


def d2q9(ax):
    ox, oy = D2_ORIGIN
    h = D2_HALF
    ax.plot([ox - h, ox + h, ox + h, ox - h, ox - h], [oy - h, oy - h, oy + h, oy + h, oy - h],
            color='black', lw=1.8, zorder=2, solid_joinstyle='miter')
    for i, (dx, dy) in D2Q9.items():
        axial = dx == 0 or dy == 0
        arrow(ax, (ox, oy), (ox + TIP * h * dx, oy - TIP * h * dy), AXIAL if axial else DIAG)
        label(ax, D2Q9_LABELS[i], str(i), AXIAL if axial else DIAG)
    label(ax, (160, 438), '(a) D2Q9')
    triad(ax, (350, 250), 55)


def d3q19(ax):
    ox, oy = D3_ORIGIN
    h = D3_HALF
    zx, zy = D3_Z

    def project(x, y, z):
        return ox + h * x + zx * z, oy - h * y + zy * z

    def point(i):
        return project(*D3Q19[i])

    corners = {(x, y, z): project(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)}
    for z in (-1, 1):
        ring = [corners[(-1, -1, z)], corners[(1, -1, z)], corners[(1, 1, z)], corners[(-1, 1, z)]]
        ax.add_patch(plt.Polygon(ring, closed=True, fill=False, ec='black', lw=1.8, zorder=2, joinstyle='miter'))
    for x in (-1, 1):
        for y in (-1, 1):
            a, b = corners[(x, y, 1)], corners[(x, y, -1)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=1.8, zorder=2, solid_capstyle='butt')

    for rect in D3_RECTANGLES:
        pts = [point(i) for i in rect] + [point(rect[0])]
        ax.plot(*zip(*pts), color='black', lw=1.3, ls=(0, (1, 3)), zorder=1, dash_capstyle='round')

    for i, (dx, dy, dz) in D3Q19.items():
        axial = abs(dx) + abs(dy) + abs(dz) == 1
        tx, ty = project(TIP * dx, TIP * dy, TIP * dz)
        arrow(ax, (ox, oy), (tx, ty), AXIAL if axial else DIAG)
        lx, ly = D3Q19_LABELS[i]
        label(ax, (SHIFT_B + lx, ly), str(i), AXIAL if axial else DIAG)
    label(ax, (SHIFT_B + 168, 438), '(b) D3Q19')
    triad(ax, (SHIFT_B + 437, 212), 55, depth=(-14, 22))


def d3q27(ax):
    """Painel (c): as 26 direções não nulas do D3Q27 (as 18 do D3Q19 e as 8 dos vértices do cubo)."""
    ox, oy = SHIFT_C + 199, D3_ORIGIN[1]
    h = 100.0
    zx, zy = -44.0, 64.0

    def project(x, y, z):
        return ox + h * x + zx * z, oy - h * y + zy * z

    corners = {(x, y, z): project(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)}
    for z in (-1, 1):
        ring = [corners[(-1, -1, z)], corners[(1, -1, z)], corners[(1, 1, z)], corners[(-1, 1, z)]]
        ax.add_patch(plt.Polygon(ring, closed=True, fill=False, ec='black', lw=1.8, zorder=2, joinstyle='miter'))
    for x in (-1, 1):
        for y in (-1, 1):
            a, b = corners[(x, y, 1)], corners[(x, y, -1)]
            ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=1.8, zorder=2, solid_capstyle='butt')

    for rect in D3_RECTANGLES:
        pts = [project(*D3Q27[i]) for i in rect] + [project(*D3Q27[rect[0]])]
        ax.plot(*zip(*pts), color='black', lw=1.3, ls=(0, (1, 3)), zorder=1, dash_capstyle='round')

    for i, (dx, dy, dz) in D3Q27.items():
        norm = abs(dx) + abs(dy) + abs(dz)
        color = AXIAL if norm == 1 else DIAG
        tx, ty = project(TIP * dx, TIP * dy, TIP * dz)
        arrow(ax, (ox, oy), (tx, ty), color)
        # rótulo além da ponta da seta, afastado do centro na direção projetada
        px, py = project(dx, dy, dz)
        vx, vy = px - ox, py - oy
        length = (vx * vx + vy * vy) ** 0.5
        offset = 34 if norm == 2 else 28
        label(ax, (px + offset * vx / length, py + offset * vy / length), str(i), color, fontsize=FONT - 3, halo=True)
    label(ax, (SHIFT_C + 168, 438), '(c) D3Q27')


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    d2q9(ax)
    d3q19(ax)
    d3q27(ax)
    fig.savefig(OUT, dpi=300, facecolor='white')
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

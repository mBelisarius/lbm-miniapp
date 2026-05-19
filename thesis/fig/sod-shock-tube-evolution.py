"""Redraws the wave pattern sketch of the Sod shock tube (Sod, 1978) in high resolution.

The layout follows the reference scan thesis/fig/sod-shock-tube-evolution-original.png, with
coordinates expressed in the pixels of that scan.

Usage: python sod-shock-tube-evolution.py
Writes sod-shock-tube-evolution.png next to this script.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 1448, 533
X1, X0, X2, X3, X4 = 388, 590, 685, 885, 1093
X_LEFT, X_RIGHT = 38, 1390
LW = 1.9
FONT = 15.5
DASHES = (10, 4.4)

plt.rcParams['mathtext.fontset'] = 'dejavusans'


def fan(y_start, y_end):
    """Smooth transition across the rarefaction fan, flat at its tail."""
    x = np.linspace(X1, X0, 100)
    s = (x - X1) / (X0 - X1)
    return x, y_start + (y_end - y_start) * (1 - (1 - s) ** 2)


def main():
    fig = plt.figure(figsize=(12, 12 * HEIGHT / WIDTH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis('off')

    line = dict(color='black', lw=LW, solid_capstyle='butt', solid_joinstyle='round', zorder=2)

    # Region boundaries
    for x, top, bottom in ((X1, 33, 478), (X0, 170, 470), (X2, 33, 478), (X3, 33, 478), (X4, 50, 478)):
        ax.plot([x, x], [top, bottom], color='black', lw=LW, dashes=DASHES, zorder=1)

    def level(y, x_start, x_end, ticks=()):
        ax.plot([x_start, x_end], [y, y], **line)
        for x in ticks:
            ax.plot([x, x], [y - 12, y + 12], **line)

    # Pressure
    level(173, X_LEFT, X1)
    ax.plot(*fan(173, 230), **line)
    level(232, X0, X4, ticks=(X4,))
    level(347, X4, X_RIGHT, ticks=(X4,))

    # Density
    level(265, X_LEFT, X1)
    ax.plot(*fan(270, 357), **line)
    level(360, X0, X3, ticks=(X3,))
    level(297, X3, X4, ticks=(X3, X4))
    level(406, X4, X_RIGHT, ticks=(X4,))
    ax.plot([X1, X0], [270, 357], ls='none', marker='o', ms=3, color='black', zorder=3)

    # Velocity
    level(463, X_LEFT, X_RIGHT)
    ax.plot([X1, X2], [463, 186], **line)
    level(186, X2, X4, ticks=(X4,))

    text = dict(fontsize=FONT, ha='center', va='center', family='monospace')
    for x, label in ((200, 'Região 1'), (508, 'Região 2'), (780, 'Região 3'), (980, 'Região 4'),
                     (1240, 'Região 5')):
        ax.text(x, 53, label, **text)

    labels = (
        (205, 143, r'p_1'), (205, 237, r'\rho_1'), (200, 433, r'u_1'),
        (762, 145, r'u_3'), (762, 203, r'p_3'), (762, 323, r'\rho_3'),
        (968, 150, r'u_4'), (968, 208, r'p_4'), (968, 263, r'\rho_4'),
        (1210, 311, r'p_5'), (1210, 376, r'\rho_5'), (1210, 436, r'u_5'),
        (376, 500, r'x_1'), (586, 499, r'x_0'), (676, 500, r'x_2'), (894, 500, r'x_3'), (1101, 500, r'x_4'),
    )
    for x, y, label in labels:
        ax.text(x, y, '$\\mathrm{' + label + '}$' if '\\rho' not in label else f'${label}$', **text)

    out = Path(__file__).with_name('sod-shock-tube-evolution.png')
    fig.savefig(out, dpi=300, facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

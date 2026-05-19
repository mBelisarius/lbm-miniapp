"""Redraws the initial condition sketch of the Sod shock tube (Sod, 1978) in high resolution.

The layout follows the reference scan thesis/fig/sod-shock-tube-initial-original.png, with
coordinates expressed in the pixels of that scan.

Usage: python sod-shock-tube-initial.py
Writes sod-shock-tube-initial.png next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt

WIDTH, HEIGHT = 1240, 690
X0 = 652
LW = 1.9
FONT = 17

plt.rcParams['mathtext.fontset'] = 'dejavusans'


def main():
    fig = plt.figure(figsize=(10, 10 * HEIGHT / WIDTH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis('off')

    line = dict(color='black', lw=LW, solid_capstyle='butt', zorder=2)

    # Membrane position
    ax.plot([X0, X0], [55, 618], color='black', lw=LW, dashes=(10, 4.6), zorder=2)

    # Left state (p1, rho1) and right state (p5, rho5)
    for y, x_start in ((313, 82), (407, 90)):
        ax.plot([x_start, X0], [y, y], **line)
        ax.plot([X0, X0], [y - 14, y + 14], **line)
    for y in (497, 561):
        ax.plot([X0, 1205], [y, y], **line)
        ax.plot([X0, X0], [y - 14, y + 14], **line)

    # Zero velocity baseline
    ax.plot([82, 1205], [617, 617], **line)

    text = dict(fontsize=FONT, ha='center', va='center', family='monospace')
    ax.text(360, 88, 'Região 1', **text)
    ax.text(893, 92, 'Região 5', **text)
    ax.text(418, 275, r'$\mathrm{p_1}$', **text)
    ax.text(418, 358, r'$\rho_1$', **text)
    ax.text(445, 577, r'$\mathrm{u_1\ =\ 0}$', **text)
    ax.text(812, 458, r'$\mathrm{p_5}$', **text)
    ax.text(820, 533, r'$\rho_5$', **text)
    ax.text(835, 580, r'$\mathrm{u_5\ =\ 0}$', **text)
    ax.text(655, 660, r'$\mathrm{x_0}$', **text)

    out = Path(__file__).with_name('sod-shock-tube-initial.png')
    fig.savefig(out, dpi=300, facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

"""Draws the heterogeneous computing node diagram of the literature review.

Usage: python hpc-heterogeneous-node.py
Writes hpc-heterogeneous-node.png next to this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

CPU_COLOR = '#d6e4f0'
GPU_COLOR = '#d9ead3'
MEM_COLOR = '#f3e9d2'
CORE_COLOR = '#8fb3d1'
MODEL_COLOR = '#ffffff'
EDGE = '#333333'


def box(ax, x, y, w, h, color, label=None, size=11, style='round,pad=0,rounding_size=0.8',
        lw=1.4, ls='-', weight='bold', label_y=None, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=color, ec=EDGE, lw=lw, ls=ls, zorder=z))
    if label:
        ax.text(x + w / 2, label_y if label_y is not None else y + h / 2, label, ha='center', va='center',
                fontsize=size, fontweight=weight, zorder=z + 1)


def pill(ax, x, y, w, label, h=2.8):
    box(ax, x, y, w, h, MODEL_COLOR, label, size=9.5, style='round,pad=0,rounding_size=1.4', lw=1.1,
        weight='normal', z=4)


def arrow(ax, p0, p1, style='<->', lw=1.8, color=EDGE):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=5))


def cores(ax, x, y, w, h, rows, cols, color=CORE_COLOR):
    gap = 0.8
    cw = (w - gap * (cols + 1)) / cols
    ch = (h - gap * (rows + 1)) / rows
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x + gap + c * (cw + gap), y + gap + r * (ch + gap)), cw, ch,
                                   fc=color, ec=EDGE, lw=0.8, zorder=3))


def device(ax, x, color, title, mem_label, grid, n_stack=1):
    for k in range(n_stack - 1, 0, -1):
        box(ax, x + 1.0 * k, 26 + 1.0 * k, 24, 14, color, lw=1.1, z=1)
    box(ax, x, 26, 24, 14, color, label_y=None)
    ax.text(x + 12, 38.2, title, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)
    cores(ax, x + 2, 27.2, 20, 8.2, *grid)
    box(ax, x, 10, 24, 8, MEM_COLOR, mem_label, size=10)
    arrow(ax, (x + 12, 18.3), (x + 12, 25.7))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 110)
    ax.set_ylim(2, 58)
    ax.set_aspect('equal')
    ax.axis('off')

    box(ax, 2, 4, 78, 52, '#fafafa', style='round,pad=0,rounding_size=1.5', ls=(0, (6, 4)), z=0)
    ax.text(4, 6, 'Nó de computação heterogêneo', fontsize=11, style='italic', va='center')

    # Host
    device(ax, 7, CPU_COLOR, 'CPU multinúcleo', 'Memória principal', (2, 4))
    pill(ax, 7.5, 47.5, 10.5, 'OpenMP')
    pill(ax, 19.5, 47.5, 10.5, 'MPI')
    ax.text(19, 52.6, 'Modelos de programação', ha='center', va='center', fontsize=9, color='#444444')

    # Accelerators
    device(ax, 49, GPU_COLOR, 'Acelerador(es)', 'Memória dedicada', (2, 8), n_stack=2)
    pill(ax, 49, 47.5, 7, 'CUDA')
    pill(ax, 56.7, 47.5, 8, 'OpenCL')
    pill(ax, 65.4, 47.5, 9.3, 'OpenACC')
    ax.text(61, 52.6, 'Modelos de programação', ha='center', va='center', fontsize=9, color='#444444')
    pill(ax, 49, 43.7, 8, 'Kokkos')
    pill(ax, 57.7, 43.7, 7, 'RAJA')

    # Interconnect between host and accelerator
    arrow(ax, (31.3, 33), (48.7, 33))
    ax.text(40, 35.3, 'PCIe', ha='center', va='center', fontsize=11, fontweight='bold')

    # Cluster
    for k in (2, 1):
        box(ax, 91 + 1.3 * k, 24 + 1.3 * k, 15, 18, '#ececec', lw=1.1, z=k * 0.1)
    box(ax, 91, 24, 15, 18, '#ececec', 'Outros nós\ndo agrupamento', size=9.5, weight='normal')
    arrow(ax, (80.3, 33), (90.7, 33))
    ax.text(85.5, 35.3, 'Rede', ha='center', va='center', fontsize=10, fontweight='bold')
    pill(ax, 81.6, 29.2, 7.8, 'MPI')

    out = Path(__file__).with_name('hpc-heterogeneous-node.png')
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Figure saved to {out}')


if __name__ == '__main__':
    main()

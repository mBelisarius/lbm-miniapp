"""Gera o diagrama roofline do miniaplicativo (FIGURA 14).

Os tetos da plataforma e a posição do miniaplicativo vêm de roofline-count.json,
produzido por roofline-count.py a partir do código-fonte dos kernels. O teto
em traço e ponto simula a mesma GPU com razão de 1:2 entre as taxas em precisão dupla
e simples, típica das GPUs de HPC, em lugar da razão de 1:64 da GPU real.

Para que a figura seja legível em impressão em preto e branco, cada plataforma
se distingue pelo estilo de linha do teto (CPU contínua, GPU tracejada, GPU hipotética em
traço e ponto) e pela forma do marcador (quadrado, círculo, triângulo), e a variante do
kernel, pelo preenchimento do marcador (cheio sem tabelas de propagação, vazado com elas).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

plt.rcParams['mathtext.fontset'] = 'dejavusans'

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).with_suffix('.png')
MODEL = json.loads((HERE / 'roofline-count.json').read_text())['model']

EDGE = '#333333'
BLUE = '#2f6ea5'
ORANGE = '#c4622d'
PURPLE = '#7b4f9d'
LW = 2.1
FS = 12

X_MIN, X_MAX = 0.1, 100.0
Y_MIN, Y_MAX = 10.0, 30000.0


def comma(v, _):
    return f'{v:g}'.replace('.', ',')


def fmt(v, nd=1):
    return f'{v:.{nd}f}'.replace('.', ',')


def main():
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=600, facecolor='white')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, which='major', color='#d9d9d9', lw=0.8, zorder=0)

    plat, var = MODEL['platform'], MODEL['variants']
    xs = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 400)
    # O teto hipotético coincide com o da GPU real na região inclinada; parte do ponto de transição real.
    x0 = plat['gpu']['peak_gflops'] / plat['gpu']['bw_gbs']
    xh = np.concatenate(([x0], xs[xs > x0]))
    roofs = (('cpu', ORANGE, 'Teto da CPU', '-', xs, 'right'),
             ('gpu', BLUE, 'Teto da GPU (razão 1:64)', (0, (5, 2.5)), xs, 'left'),
             ('gpu_1_2', PURPLE, 'Teto da GPU hipotética (razão 1:2)', (0, (7, 2, 1.5, 2)), xh, 'left'))
    for key, color, label, ls, x, side in roofs:
        p = plat[key]
        ax.plot(x, np.minimum(p['bw_gbs'] * x, p['peak_gflops']), color=color, lw=LW + 0.3, ls=ls, zorder=3,
                solid_capstyle='round', label=label)
        ridge = p['peak_gflops'] / p['bw_gbs']
        ax.plot([ridge, ridge], [Y_MIN, p['peak_gflops']], color=color, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(ridge * 1.06 if side == 'left' else ridge / 1.06, Y_MIN * 1.18,
                f'{fmt(ridge, 2 if ridge < 1 else 1)} op/byte', color=color, fontsize=FS - 2, ha=side, va='bottom',
                zorder=6)

    marks = (('cpu', 's', ORANGE, EDGE, f'Miniaplicativo na CPU ({fmt(var["cpu"]["ai"], 2)} op/byte)'),
             ('gpu', 'o', BLUE, EDGE, f'Miniaplicativo na GPU ({fmt(var["gpu"]["ai"], 2)} op/byte)'),
             ('gpu_tabelas', 'o', 'white', BLUE,
              f'GPU com tabelas de propagação ({fmt(var["gpu_tabelas"]["ai"], 2)} op/byte)'),
             ('gpu_1_2', '^', PURPLE, EDGE, 'Miniaplicativo na GPU hipotética'),
             ('gpu_tabelas_1_2', '^', 'white', PURPLE, 'GPU hipotética com tabelas de propagação'))
    sizes = {'s': 10, 'o': 11.5, '^': 12}
    for key, marker, mfc, mec, label in marks:
        v = var[key]
        ax.plot([v['ai']], [v['gflops']], marker, ms=sizes[marker], mfc=mfc, mec=mec,
                mew=1.5 if mfc != 'white' else 2.2, zorder=5, label=label)

    # Mesma intensidade aritmética nas duas GPUs: o ganho é o deslocamento vertical até o novo teto.
    for real, hyp in (('gpu', 'gpu_1_2'), ('gpu_tabelas', 'gpu_tabelas_1_2')):
        x, y0, y1 = var[real]['ai'], var[real]['gflops'], var[hyp]['gflops']
        ax.annotate('', xy=(x, y1), xytext=(x, y0), zorder=4,
                    arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.4, shrinkA=8, shrinkB=8, mutation_scale=12))
        ax.text(x * 1.1, np.sqrt(y0 * y1), f'×{fmt(var[hyp]["mlups"] / var[real]["mlups"], 1)}', color=PURPLE,
                fontsize=FS - 1, ha='left', va='center', zorder=6)

    for key, color, k, va in (('cpu', ORANGE, 1.07, 'bottom'), ('gpu', BLUE, 0.93, 'top'),
                              ('gpu_1_2', PURPLE, 1.07, 'bottom')):
        peak = plat[key]['peak_gflops'] / 1e3
        ax.text(X_MAX * 0.97, plat[key]['peak_gflops'] * k, f'{fmt(peak, 2 if peak < 10 else 1)} TFLOP/s',
                color=color, fontsize=FS - 1, ha='right', va=va, zorder=6)

    for key, color, x_lbl, nd in (('cpu', ORANGE, 0.13, 1), ('gpu', BLUE, 0.13, 0)):
        bw = MODEL['platform'][key]['bw_gbs']
        p0 = ax.transData.transform((x_lbl, bw * x_lbl))
        p1 = ax.transData.transform((x_lbl * 2, bw * x_lbl * 2))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(x_lbl, bw * x_lbl * 1.25, f'{fmt(bw, nd)} GB/s', color=color, fontsize=FS - 1, rotation=angle,
                rotation_mode='anchor', ha='left', va='bottom', zorder=6)

    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', labelsize=FS, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    ax.set_xlabel('Intensidade aritmética (operações por byte)', fontsize=FS + 1, labelpad=8)
    ax.set_ylabel('Desempenho (GFLOP/s, dupla precisão)', fontsize=FS + 1, labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.4)
        s.set_color(EDGE)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=FS - 2, frameon=True,
              framealpha=0.95, edgecolor='#999999', borderpad=0.7, labelspacing=0.5, columnspacing=1.6,
              handlelength=3.6)

    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

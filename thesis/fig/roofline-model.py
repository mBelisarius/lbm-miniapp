"""Gera o diagrama roofline do miniaplicativo (FIGURA 14), para o núcleo de cálculo atual.

Os tetos das plataformas (especificações do AMD Ryzen 9 9900X e da NVIDIA GeForce RTX 5070) vêm de
roofline-count.json, campo `platform`. O núcleo é o fatorado por eixo: FLOP = 770 operações e
BYTES = 320 bytes por nó e por passo (a CPU escreve as populações com armazenamento não temporal nas
malhas que excedem a cache, de modo que o tráfego é o mesmo da GPU), o que dá AI = 770/320 operações por byte.
O teto em traço e ponto simula a mesma GPU com razão de 1:2 entre as taxas em precisão dupla
e simples, típica das GPUs de HPC, em lugar da razão de 1:64 da GPU real.

As taxas obtidas vêm de ../campaign-results/protocol-otimizado.json (versão otimizada, N_x = 3e4): CPU com
doze núcleos (OpenMP) e GPU, medida em uma GPU de menor pico em dupla precisão (277 GFLOP/s) e projetada
à de referência pela razão entre os picos (482,3/277). O desempenho em GFLOP/s é a taxa de atualização vezes
as 770 operações do algoritmo; como esse número de operações é o do algoritmo fatorado, e não o de uma
execução com trabalho redundante, o eixo vertical não deve ser usado para comparar implementações com
contagens de operações distintas: para isso vale a taxa de atualização de nós (MLUPS).

Para que a figura seja legível em impressão em preto e branco, cada plataforma
se distingue pelo estilo de linha do teto (CPU contínua, GPU tracejada, GPU hipotética em
traço e ponto) e pela forma do marcador (quadrado, círculo, triângulo). Os marcadores vazados são os tetos
do modelo na intensidade aritmética do núcleo e os cheios, as taxas obtidas.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

plt.rcParams['mathtext.fontset'] = 'dejavusans'

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).with_suffix('.png')
MODEL = json.loads((HERE / 'roofline-count.json').read_text())['model']
PROTOCOL = json.loads((HERE.parent / 'campaign-results' / 'protocol-otimizado.json').read_text())['otimizada']
GPU_MEASURED_PEAK = 277.0  # GFLOP/s de FMA em dupla precisão da GPU de medição (microprograma)

EDGE = '#333333'
BLUE = '#2f6ea5'
ORANGE = '#c4622d'
PURPLE = '#7b4f9d'
LW = 2.1
FS = 12
FLOP = 770   # operações por nó e por passo do núcleo fatorado por eixo (Seção "Custo Computacional")
BYTES = 320  # tráfego por nó e por passo
AI = FLOP / BYTES

_gpu = [t['30000']['mlups'] for t in PROTOCOL['gpu_technologies'].values()]
_scale = MODEL['platform']['gpu']['peak_gflops'] / GPU_MEASURED_PEAK
OBTIDO = {'cpu': PROTOCOL['cpu_technologies']['OpenMP']['30000']['mlups'],
          'gpu': (min(_gpu) * _scale, max(_gpu) * _scale)}

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

    plat = MODEL['platform']
    xs = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 400)
    # O teto hipotético coincide com o da GPU real na região inclinada; parte do ponto de transição real.
    x0 = plat['gpu']['peak_gflops'] / plat['gpu']['bw_gbs']
    xh = np.concatenate(([x0], xs[xs > x0]))
    roofs = (('cpu', ORANGE, 'Teto da CPU', '-', xs, 'right'),
             ('gpu', BLUE, 'Teto da GPU (razão 1:64)', (0, (5, 2.5)), xs, 'left'),
             ('gpu_1_2', PURPLE, 'Teto da GPU hipotética (razão 1:2)', (0, (7, 2, 1.5, 2)), xh, 'left'))
    line_handles, symbol_handles = [], []
    for key, color, label, ls, x, side in roofs:
        p = plat[key]
        line_handles.extend(ax.plot(x, np.minimum(p['bw_gbs'] * x, p['peak_gflops']), color=color, lw=LW + 0.3,
                                    ls=ls, zorder=3, solid_capstyle='round', label=label))
        ridge = p['peak_gflops'] / p['bw_gbs']
        ax.plot([ridge, ridge], [Y_MIN, p['peak_gflops']], color=color, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(ridge * 1.06 if side == 'left' else ridge / 1.06, Y_MIN * 1.18,
                f'{fmt(ridge, 2 if ridge < 1 else 1)} op/byte', color=color, fontsize=FS - 2, ha=side, va='bottom',
                zorder=6)

    kernel = {}
    for key in ('cpu', 'gpu', 'gpu_1_2'):
        ceiling = min(plat[key]['bw_gbs'] * AI, plat[key]['peak_gflops'])
        kernel[key] = dict(gflops=ceiling, mlups=ceiling * 1e3 / FLOP)

    # Tetos do modelo na intensidade aritmética do núcleo (marcadores vazados). O da CPU, de 246 GFLOP/s, fica
    # na altura da taxa projetada da GPU: o quadrado vazado maior envolve o círculo cheio.
    ax.plot([AI, AI], [Y_MIN, kernel['gpu_1_2']['gflops']], color='#999999', lw=1.0, ls=(0, (1, 2)), zorder=2)
    ax.text(AI * 1.05, Y_MIN * 2.3, f'{fmt(AI, 2)} op/byte', color='#555555', fontsize=FS - 2, ha='left',
            va='bottom', zorder=6)
    for key, marker, color, name, size in (('cpu', 's', ORANGE, 'CPU', 17), ('gpu', 'o', BLUE, 'GPU', 11.5),
                                           ('gpu_1_2', '^', PURPLE, 'GPU hipotética', 12)):
        ax.plot([AI], [kernel[key]['gflops']], marker, ms=size, mfc='white', mec=color, mew=2.0, zorder=5)
        symbol_handles.append(Line2D([], [], ls='none', marker=marker, ms=10, mfc='white', mec=color, mew=2.0,
                                     label=f'{name}: teto do modelo ({fmt(kernel[key]["mlups"], 0)} MLUPS)'))

    # Taxas obtidas (marcadores cheios): CPU com doze núcleos; GPU projetada, faixa entre as tecnologias.
    cpu_att = OBTIDO['cpu'] * FLOP / 1e3
    ax.plot([AI], [cpu_att], 's', ms=10, mfc=ORANGE, mec=EDGE, mew=1.5, zorder=6)
    symbol_handles.append(Line2D([], [], ls='none', marker='s', ms=10, mfc=ORANGE, mec=EDGE, mew=1.5,
                                 label=f'CPU: obtida, 12 núcleos ({fmt(OBTIDO["cpu"], 0)} MLUPS)'))
    gpu_lo, gpu_hi = (v * FLOP / 1e3 for v in OBTIDO['gpu'])
    gpu_mid = np.sqrt(gpu_lo * gpu_hi)
    ax.errorbar([AI], [gpu_mid], yerr=[[gpu_mid - gpu_lo], [gpu_hi - gpu_mid]], fmt='o', ms=11.5, mfc=BLUE,
                mec=EDGE, mew=1.5, ecolor=BLUE, elinewidth=2.0, capsize=5, zorder=6)
    symbol_handles.append(Line2D([], [], ls='none', marker='o', ms=10, mfc=BLUE, mec=EDGE, mew=1.5,
                                 label=f'GPU: obtida, projeção ({fmt(OBTIDO["gpu"][0], 0)}\u2013'
                                       f'{fmt(OBTIDO["gpu"][1], 0)} MLUPS)'))
    ax.text(AI * 1.3, cpu_att, f'{fmt(100 * OBTIDO["cpu"] / kernel["cpu"]["mlups"], 0)}% do teto', color=ORANGE,
            fontsize=FS - 2, ha='left', va='center', zorder=6)
    ax.text(AI * 1.3, gpu_mid, f'{fmt(100 * OBTIDO["gpu"][0] / kernel["gpu"]["mlups"], 0)}\u2013'
                               f'{fmt(100 * OBTIDO["gpu"][1] / kernel["gpu"]["mlups"], 0)}% do teto',
            color=BLUE, fontsize=FS - 2, ha='left', va='center', zorder=6)

    # Mesma intensidade aritmética nas duas GPUs: o ganho é o deslocamento vertical até o novo teto.
    x, y0, y1 = AI, kernel['gpu']['gflops'], kernel['gpu_1_2']['gflops']
    ax.annotate('', xy=(x, y1), xytext=(x, y0), zorder=4,
                arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.4, shrinkA=8, shrinkB=8, mutation_scale=12))
    ax.text(x * 1.1, np.sqrt(y0 * y1), f'×{fmt(kernel["gpu_1_2"]["mlups"] / kernel["gpu"]["mlups"], 1)}',
            color=PURPLE, fontsize=FS - 1, ha='left', va='center', zorder=6)

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
    # Coluna esquerda: linhas (tetos); coluna direita: símbolos. Entradas em branco completam a coluna esquerda.
    blank = Line2D([], [], color='none')
    handles = line_handles + [blank] * (len(symbol_handles) - len(line_handles)) + symbol_handles
    labels = [h.get_label() for h in line_handles] + [' '] * (len(symbol_handles) - len(line_handles)) \
        + [h.get_label() for h in symbol_handles]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=FS - 2,
              frameon=True, framealpha=0.95, edgecolor='#999999', borderpad=0.7, labelspacing=0.5,
              columnspacing=1.6, handlelength=3.6)

    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

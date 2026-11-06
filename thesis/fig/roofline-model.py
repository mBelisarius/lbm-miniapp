"""Gera o diagrama roofline do miniaplicativo, para o núcleo de cálculo final.

O diagrama (roofline-model.png) apresenta a plataforma de referência (AMD Ryzen 9 9900X e NVIDIA GeForce RTX 5070),
com tetos das especificações dos fabricantes (roofline-final.json, campo `platform`), a GPU hipotética de razão 1:2
e as taxas da maior malha do protocolo (N_x = 5e4): CPU com doze núcleos e GPU (protocol-final.json).

O núcleo é o fatorado por eixo na forma de produto, com cinco iterações de Newton: FLOP operações
(roofline-final.json, contagem dinâmica do núcleo vetorial com a exponencial a 20 operações) e BYTES = 320 bytes por
nó e por passo (a CPU escreve as populações com armazenamento não temporal nas malhas que excedem a cache, de modo
que o tráfego é o mesmo da GPU), o que dá AI = FLOP/BYTES. O desempenho em GFLOP/s é a taxa de atualização vezes
FLOP; como FLOP é a contagem do núcleo adotado, o eixo vertical não deve ser usado para comparar implementações com
contagens de operações distintas (eliminar trabalho redundante desloca o ponto para baixo): para isso vale a taxa de
atualização de nós (MLUPS).

Para que a figura seja legível em impressão em preto e branco, cada plataforma se distingue pelo estilo de linha
do teto (CPU contínua, GPU tracejada, GPU hipotética em traço e ponto) e pela forma do marcador (quadrado, círculo,
triângulo). Os marcadores vazados são os tetos do modelo na intensidade aritmética do núcleo e os cheios, as taxas obtidas.
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
RESULTS = HERE.parent / 'campaign-results'
FINAL = json.loads((HERE / 'roofline-final.json').read_text())
MODEL = {'platform': FINAL['platform']}
PROTOCOL = json.loads((RESULTS / 'protocol-final.json').read_text())
LARGEST_NX = '50000'

EDGE = '#333333'
BLUE = '#2f6ea5'
ORANGE = '#c4622d'
PURPLE = '#7b4f9d'
LW = 2.1
FS = 12
FLOP = FINAL['flop_model']   # operações por nó e por passo do núcleo final (modelo, exponencial a 20 operações)
BYTES = 320                  # tráfego por nó e por passo
AI = FLOP / BYTES

X_MIN, X_MAX = 0.1, 100.0


def comma(v, _):
    return f'{v:g}'.replace('.', ',')


def fmt(v, nd=1):
    return f'{v:.{nd}f}'.replace('.', ',')


def _slope_label(ax, x_lbl, bw, color, nd):
    p0 = ax.transData.transform((x_lbl, bw * x_lbl))
    p1 = ax.transData.transform((x_lbl * 2, bw * x_lbl * 2))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x_lbl, bw * x_lbl * 1.25, f'{fmt(bw, nd)} GB/s', color=color, fontsize=FS - 1, rotation=angle,
            rotation_mode='anchor', ha='left', va='bottom', zorder=6)


def _frame(ax, y_min, y_max, y_ticks):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='major', color='#d9d9d9', lw=0.8, zorder=0)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(comma))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(comma))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', labelsize=FS, width=1.2, length=6)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    ax.set_xlabel('Intensidade aritmética (operações por byte)', fontsize=FS + 1, labelpad=8)
    ax.set_ylabel('Desempenho (GFLOP/s, dupla precisão)', fontsize=FS + 1, labelpad=8)
    for s in ax.spines.values():
        s.set_linewidth(1.4)
        s.set_color(EDGE)


def _legend(ax, line_handles, symbol_handles):
    # Coluna esquerda: linhas (tetos); coluna direita: símbolos. Entradas em branco completam a coluna esquerda.
    blank = Line2D([], [], color='none')
    pad = max(0, len(symbol_handles) - len(line_handles))
    handles = line_handles + [blank] * pad + symbol_handles
    labels = [h.get_label() for h in line_handles] + [' '] * pad + [h.get_label() for h in symbol_handles]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=FS - 2,
              frameon=True, framealpha=0.95, edgecolor='#999999', borderpad=0.7, labelspacing=0.5,
              columnspacing=1.6, handlelength=3.6)


def reference(out):
    """Plataforma de referência: tetos das especificações, GPU hipotética 1:2 e taxas da maior malha."""
    y_min, y_max = 10.0, 30000.0
    gpu_rates = [t[LARGEST_NX]['mlups'] for t in PROTOCOL['gpu_technologies'].values()]
    obtained = {'cpu': PROTOCOL['cpu_technologies']['OpenMP'][LARGEST_NX]['mlups'],
                'gpu': (min(gpu_rates), max(gpu_rates))}

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=600, facecolor='white')
    _frame(ax, y_min, y_max, [10, 100, 1000, 10000])
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
        ax.plot([ridge, ridge], [y_min, p['peak_gflops']], color=color, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(ridge * 1.06 if side == 'left' else ridge / 1.06, y_min * 1.18,
                f'{fmt(ridge, 2 if ridge < 1 else 1)} op/byte', color=color, fontsize=FS - 2, ha=side, va='bottom',
                zorder=6)

    kernel = {}
    for key in ('cpu', 'gpu', 'gpu_1_2'):
        ceiling = min(plat[key]['bw_gbs'] * AI, plat[key]['peak_gflops'])
        kernel[key] = dict(gflops=ceiling, mlups=ceiling * 1e3 / FLOP)

    ax.plot([AI, AI], [y_min, kernel['gpu_1_2']['gflops']], color='#999999', lw=1.0, ls=(0, (1, 2)), zorder=2)
    ax.text(AI * 1.05, y_min * 2.3, f'{fmt(AI, 2)} op/byte', color='#555555', fontsize=FS - 2, ha='left',
            va='bottom', zorder=6)
    for key, marker, color, name, size in (('cpu', 's', ORANGE, 'CPU', 17), ('gpu', 'o', BLUE, 'GPU', 11.5),
                                           ('gpu_1_2', '^', PURPLE, 'GPU hipotética', 12)):
        ax.plot([AI], [kernel[key]['gflops']], marker, ms=size, mfc='white', mec=color, mew=2.0, zorder=5)
        symbol_handles.append(Line2D([], [], ls='none', marker=marker, ms=10, mfc='white', mec=color, mew=2.0,
                                     label=f'{name}: teto do modelo ({fmt(kernel[key]["mlups"], 0)} MLUPS)'))

    cpu_att = obtained['cpu'] * FLOP / 1e3
    ax.plot([AI], [cpu_att], 's', ms=10, mfc=ORANGE, mec=EDGE, mew=1.5, zorder=6)
    symbol_handles.append(Line2D([], [], ls='none', marker='s', ms=10, mfc=ORANGE, mec=EDGE, mew=1.5,
                                 label=f'CPU: 12 núcleos ({fmt(obtained["cpu"], 0)} MLUPS)'))
    gpu_lo, gpu_hi = (v * FLOP / 1e3 for v in obtained['gpu'])
    gpu_mid = np.sqrt(gpu_lo * gpu_hi)
    ax.errorbar([AI], [gpu_mid], yerr=[[gpu_mid - gpu_lo], [gpu_hi - gpu_mid]], fmt='o', ms=11.5, mfc=BLUE,
                mec=EDGE, mew=1.5, ecolor=BLUE, elinewidth=2.0, capsize=5, zorder=6)
    symbol_handles.append(Line2D([], [], ls='none', marker='o', ms=10, mfc=BLUE, mec=EDGE, mew=1.5,
                                 label=f'GPU: sete tecnologias ({fmt(obtained["gpu"][0], 0)}\u2013'
                                       f'{fmt(obtained["gpu"][1], 0)} MLUPS)'))
    ax.text(AI * 1.3, cpu_att, f'{fmt(100 * obtained["cpu"] / kernel["cpu"]["mlups"], 0)}% do teto', color=ORANGE,
            fontsize=FS - 2, ha='left', va='center', zorder=6)
    ax.text(AI * 1.3, gpu_mid, f'{fmt(100 * obtained["gpu"][0] / kernel["gpu"]["mlups"], 0)}\u2013'
                               f'{fmt(100 * obtained["gpu"][1] / kernel["gpu"]["mlups"], 0)}% do teto',
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
    for key, color, nd in (('cpu', ORANGE, 1), ('gpu', BLUE, 0)):
        _slope_label(ax, 0.13, plat[key]['bw_gbs'], color, nd)

    _legend(ax, line_handles, symbol_handles)
    fig.savefig(out, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {out}')
    return kernel, obtained


def main():
    ref = reference(HERE / 'roofline-model.png')
    print(json.dumps({'reference': ref}, indent=2, default=float))


if __name__ == '__main__':
    main()

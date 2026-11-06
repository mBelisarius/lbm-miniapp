"""Gera a comparação entre as disposições de dados das duas formas do núcleo de cálculo (FIGURA 17).

Os dois painéis mostram a mesma malha de N_x colunas por N_y linhas (aqui N_y = 4). Os números são as
posições lineares das células no plano de uma direção. Em (a), forma por célula (i * ny + j): a
direção y é contígua, a propagação ao longo de x salta N_y endereços e um warp de 32 fluxos cobre 32
células consecutivas. Em (b), forma vetorial (j * pitch + i): a direção x é contígua, cada direção lê
uma sequência de passo unitário, as colunas de preenchimento completam o passo de linha e um bloco de
dezesseis células ocupa registradores vetoriais. A leitura não depende de cor: as células são caixas
retas de tom escuro, o preenchimento é branco com traço tracejado e as células destacadas têm borda
grossa.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

W = 8.5
EDGE = '#333333'
MEM = '#d9c79a'
FS = 10
NY = 4
CH = 0.42                  # altura das células
X0 = 1.0                   # início da malha
PH_A, PH_B = 3.4, 3.75     # alturas dos painéis


def cell(ax, x, y, w, label='', fs=8.5, ls='-', fill=MEM, lw=1.1, z=2):
    ax.add_patch(Rectangle((x, y), w, CH, facecolor=fill, edgecolor=EDGE, lw=lw, ls=ls, zorder=z))
    if label:
        ax.text(x + w / 2, y + CH / 2, label, ha='center', va='center', fontsize=fs, zorder=z + 1)


def brace(ax, xa, xb, y, up):
    d = 0.07 if up else -0.07
    ax.plot([xa, xa, xb, xb], [y, y + d, y + d, y], color=EDGE, lw=1.1, zorder=3)


def row_label(ax, y, txt):
    ax.text(X0 - 0.1, y + CH / 2, txt, ha='right', va='center', fontsize=FS - 1.5)


def panel_a(ax, y0):
    cw, ncol = 0.55, 10
    gb = y0 + 0.95
    gt = gb + NY * CH
    hi = {12, 16}
    for i in range(ncol):
        for j in range(NY):
            n = i * NY + j
            on = n in hi
            cell(ax, X0 + i * cw, gt - (j + 1) * CH, cw, str(n), fill='white' if on else MEM,
                 lw=2.2 if on else 1.1, z=4 if on else 2)
    ax.text(X0 + ncol * cw + 0.12, (gb + gt) / 2, '⋯', ha='left', va='center', fontsize=FS + 2)
    for j in range(NY):
        row_label(ax, gt - (j + 1) * CH, rf'$j={j}$')

    # coluna curta contígua
    bx = 0.45
    ax.plot([bx + 0.07, bx, bx, bx + 0.07], [gt, gt, gb, gb], color=EDGE, lw=1.1)
    ax.text(bx - 0.16, (gb + gt) / 2, rf'coluna $i$ ($N_y={NY}$ células)', rotation=90, ha='center',
            va='center', fontsize=FS - 1.5)

    # propagação ao longo de x
    xa, xb = X0 + 3.5 * cw, X0 + 4.5 * cw
    ax.add_patch(FancyArrowPatch((xa, gt + 0.03), (xb, gt + 0.03), connectionstyle='arc3,rad=-0.7',
                                 arrowstyle='-|>', mutation_scale=12, lw=1.5, color=EDGE, shrinkA=0, shrinkB=0,
                                 zorder=5))
    ax.text((xa + xb) / 2, gt + 0.5, 'propagação ao longo de $x$:\norigens distam $N_y$ endereços (12 → 16)',
            ha='center', va='center', fontsize=FS - 1, linespacing=1.2)

    # warp
    wa, wb = X0 + 0.03, X0 + 8 * cw - 0.03
    brace(ax, wa, wb, gb - 0.1, up=False)
    ax.text((wa + wb) / 2, gb - 0.4, 'warp: 32 fluxos sobre 32 endereços consecutivos', ha='center', va='center',
            fontsize=FS - 1)

    ax.text(W / 2, y0 + 0.12, '(a) Forma por célula: disposição i * ny + j', ha='center', va='center',
            fontsize=FS + 1)


def panel_b(ax, y0):
    cw, tw = 0.3, 0.42
    gb = y0 + 1.15
    gt = gb + NY * CH
    xe = X0 + 16 * cw + 0.45                       # início da cauda (após as reticências)
    tail = [(r'$N_x\!-\!1$', MEM, '-'), (r'$N_x$', 'white', (0, (3, 2))), (r'$\pi\!-\!1$', 'white', (0, (3, 2)))]
    xend = xe + 3 * tw
    for j in range(NY):
        y = gt - (j + 1) * CH
        row_label(ax, y, rf'$j={j}$')
        for k in range(16):
            lab = str(k) if j == 0 else ''
            if j and k == 0:
                lab = r'$\pi$' if j == 1 else rf'${j}\pi$'
            cell(ax, X0 + k * cw, y, cw, lab, fs=7.5)
        ax.text(X0 + 16 * cw + 0.22, y + CH / 2, '⋯', ha='center', va='center', fontsize=FS + 2)
        for t, (lab, fill, ls) in enumerate(tail):
            cell(ax, xe + t * tw, y, tw, lab if j == 0 else '', fs=7.5, fill=fill, ls=ls)

    # bloco de dezesseis células
    ba, bb = X0 + 0.03, X0 + 16 * cw - 0.03
    brace(ax, ba, bb, gt + 0.1, up=True)
    ax.text((ba + bb) / 2, gt + 0.5, 'bloco de 16 células em registradores vetoriais:\n'
            'cada direção lê 16 endereços consecutivos', ha='center', va='center', fontsize=FS - 1, linespacing=1.2)

    # preenchimento
    pa, pb = xe + tw + 0.03, xend - 0.03
    brace(ax, pa, pb, gb - 0.1, up=False)
    ax.text((pa + pb) / 2, gb - 0.5, 'preenchimento ($\\pi - N_x$ células):\ncalculadas e ignoradas',
            ha='center', va='center', fontsize=FS - 1, linespacing=1.2)

    # passo de linha
    xa = xend + 0.2
    y0c, y1c = gt - CH / 2, gt - 1.5 * CH
    ax.add_patch(FancyArrowPatch((xa, y0c), (xa, y1c), arrowstyle='<|-|>', mutation_scale=10, lw=1.2, color=EDGE,
                                 shrinkA=0, shrinkB=0, zorder=5))
    ax.text(xa + 0.1, (y0c + y1c) / 2, 'passo de\nlinha $\\pi$', ha='left', va='center', fontsize=FS - 2,
            linespacing=1.15)

    ax.text(W / 2, y0 + 0.12, '(b) Forma vetorial: disposição j * pitch + i', ha='center', va='center',
            fontsize=FS + 1)


def main():
    note_h = 0.35
    H = note_h + PH_A + 0.15 + PH_B + 0.05
    fig = plt.figure(figsize=(W, H), dpi=600, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    panel_b(ax, 0.05)
    panel_a(ax, 0.05 + PH_B + 0.15)
    ax.text(W / 2, H - 0.15, 'Os números são as posições lineares das células no plano de uma direção.',
            ha='center', va='center', fontsize=FS - 1, style='italic')
    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

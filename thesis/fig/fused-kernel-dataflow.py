"""Gera o fluxo de dados do núcleo de cálculo fundido de passo único (FIGURA 14).

Os arranjos de leitura e de escrita (ping-pong), em memória, ficam nas extremidades; entre eles, as
etapas executadas em registradores para cada nó. As setas rotuladas indicam o tráfego de memória por
nó e por passo (precisão dupla). A distinção entre memória e registradores não depende de cor:
memória usa caixas retas de tom escuro, registradores usam caixas arredondadas de tom claro e o acesso
condicional usa traço tracejado.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

W = 8.5
EDGE = '#333333'
REG = '#e6eff8'
MEM = '#d9c79a'
FS = 10

CX, CW = 3.55, 4.4          # coluna principal (centro e largura)
SW, SH, GAP = 3.9, 0.78, 0.3  # etapas em registradores
SIDE_X, SIDE_W = 6.75, 1.65   # caixas laterais em memória

STEPS = [
    ('Propagação interpolada (IDW)', r'$f_i^{*},\ g_i^{*}$ a partir dos nós vizinhos'),
    ('Momentos macroscópicos', r'$\rho,\ \mathbf{u},\ T$'),
    ('Equilíbrios', r'$f_i^{eq},\ g_i^{eq}$'),
    ('Fechamento energético: Newton-Raphson', '5 iterações por eixo'),
    ('Colisão', r'$f_i^{\prime},\ g_i^{\prime}$ pós-colisão'),
]
NEWTON, MOMENTS = 3, 1


def arrow(ax, p0, p1, style='-|>', ls='-', lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, lw=lw, color=EDGE, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=5))


def memory(ax, x, y, w, h, title, sub, ls='-'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=MEM, edgecolor=EDGE, lw=1.4, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=FS, fontweight='bold', zorder=3)
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=FS - 1, zorder=3)


def label(ax, x, y, txt, **kw):
    ax.text(x, y, txt, ha=kw.pop('ha', 'left'), va='center', fontsize=FS - 1, linespacing=1.2,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=6, **kw)


def main():
    mh, xl = 0.85, CX - CW / 2
    n = len(STEPS)
    pad_top, pad, link = 0.55, 0.25, 1.0
    cont_h = pad_top + n * SH + (n - 1) * GAP + pad
    H = 0.2 + 2 * mh + 2 * link + cont_h
    fig = plt.figure(figsize=(W, H), dpi=600, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    y_read = H - 0.1 - mh
    y_write = 0.1
    y_cont = y_write + mh + link
    y_top = y_cont + cont_h

    memory(ax, xl, y_read, CW, mh, 'Arranjo de leitura (origem)', r'$f_i,\ g_i$ no passo $n$')
    memory(ax, xl, y_write, CW, mh, 'Arranjo de escrita (destino)', r'$f_i,\ g_i$ no passo $n+1$')

    # contêiner dos registradores
    ax.add_patch(FancyBboxPatch((xl + 0.1, y_cont), CW - 0.2, cont_h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc='white', ec=EDGE, lw=1.4, ls=(0, (5, 3)), zorder=1))
    ax.text(xl + 0.3, y_top - 0.27, 'Registradores\n(um nó por vez)', ha='left', va='center', fontsize=FS - 1,
            style='italic', linespacing=1.15, zorder=3)

    ys = []
    for i, (title, sub) in enumerate(STEPS):
        y = y_top - pad_top - (i + 1) * SH - i * GAP
        ys.append(y)
        ax.add_patch(FancyBboxPatch((CX - SW / 2, y), SW, SH, boxstyle='round,pad=0,rounding_size=0.1', fc=REG,
                                    ec=EDGE, lw=1.4, zorder=2))
        ax.text(CX, y + SH * 0.66, title, ha='center', va='center', fontsize=FS, fontweight='bold', zorder=3)
        ax.text(CX, y + SH * 0.28, sub, ha='center', va='center', fontsize=FS - 1, zorder=3)
        if i:
            arrow(ax, (CX, ys[i - 1]), (CX, y + SH), lw=1.4)

    # leitura e escrita das distribuições
    arrow(ax, (CX, y_read), (CX, ys[0] + SH))
    arrow(ax, (CX, ys[-1]), (CX, y_write + mh))
    label(ax, CX + 0.12, y_read - link / 2, r'$144$ bytes (leitura)')
    label(ax, CX + 0.12, y_write + mh + link / 2, r'$144$ bytes (escrita)')

    # troca de papéis (ping-pong): do arranjo de escrita de volta ao de leitura
    xp = 0.55
    ax.plot([xl, xp, xp], [y_write + mh / 2, y_write + mh / 2, y_read + mh / 2], color=EDGE, lw=1.6,
            solid_capstyle='butt', zorder=5)
    arrow(ax, (xp, y_read + mh / 2), (xl, y_read + mh / 2))
    ax.text(xp - 0.2, (y_write + y_read) / 2 + 0.2, 'troca de papéis a cada passo (ping-pong)', rotation=90,
            ha='center', va='center', fontsize=FS - 1)

    # ponto de partida de Newton-Raphson
    yn = ys[NEWTON] + SH / 2
    memory(ax, SIDE_X, yn - 0.45, SIDE_W, 0.9, r'$\xi_1,\ \xi_2$', 'ponto de partida')
    arrow(ax, (CX + SW / 2, yn), (SIDE_X, yn), style='<|-|>')
    label(ax, (CX + SW / 2 + SIDE_X) / 2, yn + 0.25, r'$32$ bytes' + '\n(leit. e escr.)', ha='center')

    # campos macroscópicos, gravados apenas no último passo
    ym = ys[MOMENTS] + SH / 2
    memory(ax, SIDE_X, ym - 0.45, SIDE_W, 0.9, r'$\rho,\ \mathbf{u},\ T,\ p$', 'campos macroscópicos', ls=(0, (3, 2)))
    arrow(ax, (CX + SW / 2, ym), (SIDE_X, ym), ls=(0, (3, 2)))
    label(ax, (CX + SW / 2 + SIDE_X) / 2, ym + 0.25, 'somente no\núltimo passo', ha='center')

    # legenda
    lx, ly = SIDE_X - 0.05, y_write + 0.1
    items = [('mem', 'memória'), ('reg', 'registradores')]
    for k, (kind, txt) in enumerate(items):
        y = ly + 0.62 - k * 0.4
        if kind == 'mem':
            ax.add_patch(Rectangle((lx, y), 0.34, 0.24, fc=MEM, ec=EDGE, lw=1.2))
        else:
            ax.add_patch(FancyBboxPatch((lx, y), 0.34, 0.24, boxstyle='round,pad=0,rounding_size=0.07', fc=REG,
                                        ec=EDGE, lw=1.2))
        ax.text(lx + 0.46, y + 0.12, txt, ha='left', va='center', fontsize=FS - 1)

    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

"""Gera o fluxo de dados de um bloco de dezesseis células no núcleo vetorial de CPU (FIGURA 15).

Cada linha é um registrador vetorial independente; as fases (momentos, fechamento energético,
equilíbrios e colisão) percorrem as colunas da esquerda para a direita, entre a leitura das
sequências contíguas de origem e o armazenamento não temporal. A distinção entre memória e
registradores não depende de cor: memória usa caixas retas de tom escuro e registradores usam
caixas arredondadas de tom claro.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

W, H = 8.5, 5.9
EDGE = '#333333'
REG = '#e6eff8'
MEM = '#d9c79a'
FS = 10

ROWS = 4
RH, RGAP = 0.78, 0.28
TOP = H - 0.6                       # topo da primeira linha
LBL_X, LBL_W = 0.1, 0.95            # rótulos das linhas
IN_X, MEM_W = 1.2, 1.15             # memória de origem
COL_X0, COL_W, COL_GAP = 2.6, 0.95, 0.18
OUT_X = COL_X0 + 4 * COL_W + 3 * COL_GAP + 0.3

PHASES = [
    ('Momentos', r'$\rho,\ \mathbf{u},\ T$'),
    ('Fechamento\nenergético', '5 iterações\npor eixo'),
    ('Equilíbrios', 'produto\n$3 \\times 3$ fatores'),
    ('Colisão', r'$f_i^{\prime},\ g_i^{\prime}$'),
]


def arrow(ax, p0, p1, lw=1.4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=12, lw=lw, color=EDGE,
                                 shrinkA=0, shrinkB=0, zorder=5))


def main():
    fig = plt.figure(figsize=(W, H), dpi=600, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    ys = [TOP - (r + 1) * RH - r * RGAP for r in range(ROWS)]
    y_bot, y_top = ys[-1], ys[0] + RH
    mem_h = y_top - y_bot

    # memória de origem e de destino
    for x, title, sub in [(IN_X, 'Leitura', '20 sequências\ncontíguas\nde origem'),
                          (OUT_X, 'Escrita', 'armazenamento\nnão temporal\ndo destino')]:
        ax.add_patch(Rectangle((x, y_bot), MEM_W, mem_h, fc=MEM, ec=EDGE, lw=1.4, zorder=2))
        ax.text(x + MEM_W / 2, y_bot + mem_h * 0.62, title, ha='center', va='center', fontsize=FS,
                fontweight='bold', zorder=3)
        ax.text(x + MEM_W / 2, y_bot + mem_h * 0.38, sub, ha='center', va='center', fontsize=FS - 1.5,
                linespacing=1.2, zorder=3)

    # fases, uma caixa por registrador vetorial
    col_x = [COL_X0 + c * (COL_W + COL_GAP) for c in range(len(PHASES))]
    for c, (head, sub) in enumerate(PHASES):
        ax.text(col_x[c] + COL_W / 2, y_top + 0.3, head, ha='center', va='center', fontsize=FS,
                fontweight='bold', linespacing=1.15)
        for r, y in enumerate(ys):
            ax.add_patch(FancyBboxPatch((col_x[c], y), COL_W, RH, boxstyle='round,pad=0,rounding_size=0.1',
                                        fc=REG, ec=EDGE, lw=1.4, zorder=2))
            ax.text(col_x[c] + COL_W / 2, y + RH / 2, sub, ha='center', va='center', fontsize=FS - 1.5,
                    linespacing=1.15, zorder=3)
            if c:
                arrow(ax, (col_x[c - 1] + COL_W, y + RH / 2), (col_x[c], y + RH / 2))
    for y in ys:
        arrow(ax, (IN_X + MEM_W, y + RH / 2), (col_x[0], y + RH / 2))
        arrow(ax, (col_x[-1] + COL_W, y + RH / 2), (OUT_X, y + RH / 2))

    # rótulos das linhas
    for r, y in enumerate(ys):
        ax.text(LBL_X + LBL_W / 2 - 0.1, y + RH / 2, f'Registrador\nvetorial {r + 1}', ha='center', va='center',
                fontsize=FS - 1.5, linespacing=1.15)
    # nota sobre as vias e o bloco
    ax.text(W / 2, y_bot - 0.5, 'Quatro registradores de quatro valores (AVX2) ou dois de oito (AVX-512): '
            'dezesseis células por bloco,\nsem dependência de dados entre registradores',
            ha='center', va='center', fontsize=FS - 1.5, linespacing=1.25)

    # legenda
    lx, ly = IN_X, 0.12
    ax.add_patch(Rectangle((lx, ly), 0.34, 0.24, fc=MEM, ec=EDGE, lw=1.2))
    ax.text(lx + 0.46, ly + 0.12, 'memória', ha='left', va='center', fontsize=FS - 1)
    lx2 = lx + 1.7
    ax.add_patch(FancyBboxPatch((lx2, ly), 0.34, 0.24, boxstyle='round,pad=0,rounding_size=0.07', fc=REG,
                                ec=EDGE, lw=1.2))
    ax.text(lx2 + 0.46, ly + 0.12, 'registradores vetoriais', ha='left', va='center', fontsize=FS - 1)

    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

"""Gera o diagrama de classes simplificado da arquitetura do miniaplicativo (FIGURA 12).

Os nomes de classes, de métodos e as relações seguem o código em `lbmini/`: `Executor`
(compõe a interface do resolvedor e o `AsyncWriter`), `ILbmTubeNd`, `ILbmTube` e suas implementações por
tecnologia de paralelização, `ILattice`/`LatticeD2Q9` e `DeviceBuffer`. As coordenadas estão
em polegadas, com origem no canto inferior esquerdo.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

plt.rcParams['mathtext.fontset'] = 'dejavusans'

OUT = Path(__file__).with_suffix('.png')

W, H = 8.5, 11.6
LH = 0.23
PAD = 0.09
EDGE = '#333333'
FILL_CLASS = '#d6e4f0'
FILL_IFACE = '#f3e9d2'
FILL_IMPL = '#d9ead3'
FILL_AUX = '#ececec'
LW = 1.5
MONO = 'DejaVu Sans Mono'
FS = 9.5

ax = None


def box(x, y_top, w, name, members=(), stereotype=None, fill=FILL_CLASS, name_fs=FS + 0.5):
    """Caixa de classe UML; `members` é uma lista de seções (listas de linhas). Retorna (x0, y0, x1, y1)."""
    n_head = 1 + (stereotype is not None)
    h = 2 * PAD + n_head * LH
    for sec in members:
        h += 2 * PAD + len(sec) * LH
    y = y_top - h
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=EDGE, lw=LW, zorder=3))
    cy = y_top - PAD
    if stereotype:
        ax.text(x + w / 2, cy - LH / 2, stereotype, ha='center', va='center', fontsize=FS - 1, zorder=4)
        cy -= LH
    ax.text(x + w / 2, cy - LH / 2, name, ha='center', va='center', fontsize=name_fs, fontweight='bold',
            family=MONO, zorder=4)
    cy -= LH + PAD
    for sec in members:
        ax.plot([x, x + w], [cy, cy], color=EDGE, lw=1.0, zorder=4)
        cy -= PAD
        for line in sec:
            ax.text(x + 0.1, cy - LH / 2, line, ha='left', va='center', fontsize=FS, family=MONO, zorder=4)
            cy -= LH
        cy -= PAD
    return x, y, x + w, y_top


def line(p0, p1, dashed=False):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=EDGE, lw=1.3, ls=(0, (4, 3)) if dashed else '-',
            zorder=2, solid_capstyle='butt')


def tri(tip, direction, size=0.2, hollow=True):
    """Ponta triangular (generalização ou seta aberta) com o vértice em `tip`."""
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    base = np.array(tip) - d * size
    pts = [tip, base + n * size * 0.55, base - n * size * 0.55]
    ax.add_patch(Polygon(pts, closed=True, facecolor='white' if hollow else EDGE, edgecolor=EDGE, lw=1.3,
                         zorder=5))


def open_head(tip, direction, size=0.17):
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    base = np.array(tip) - d * size
    for s in (1, -1):
        ax.plot([tip[0], base[0] + s * n[0] * size * 0.5], [tip[1], base[1] + s * n[1] * size * 0.5],
                color=EDGE, lw=1.3, zorder=5, solid_capstyle='round')


def diamond(at, direction, size=0.34):
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    at = np.array(at)
    pts = [at, at + d * size / 2 + n * size * 0.28, at + d * size, at + d * size / 2 - n * size * 0.28]
    ax.add_patch(Polygon(pts, closed=True, facecolor=EDGE, edgecolor=EDGE, lw=1.3, zorder=5))
    return at + d * size


def label(x, y, text, **kw):
    kw.setdefault('fontsize', FS - 0.5)
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    ax.text(x, y, text, zorder=6, bbox=dict(facecolor='white', edgecolor='none', pad=1.2), **kw)


def main():
    global ax
    fig = plt.figure(figsize=(W, H), dpi=600, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    top = H - 0.05

    # Linha superior: leitor de configuração, Executor e AsyncWriter
    yml = box(0.05, top, 2.0, 'ReadYaml', [['→ FluidData', '→ MeshData', '→ ControlData', '→ PerformanceData']],
              stereotype='«função»', fill=FILL_AUX)
    exe = box(3.1, top, 2.3, 'Executor', [['- fluid_, mesh_,', '  control_,', '  performance_'],
                                         ['+ GetExecutor()', '+ Run(outpath)', '+ GetLbmTube()']])
    wri = box(6.55, top, 1.9, 'AsyncWriter', [['+ Enqueue()', '+ WaitAll()']], fill=FILL_AUX)

    ya = top - 0.55
    line((yml[2], ya), (exe[0] - 0.0, ya), dashed=True)
    open_head((exe[0], ya), (1, 0))
    label((yml[2] + exe[0]) / 2, ya + 0.19, 'parâmetros')

    line((exe[2], ya), (wri[0], ya))
    start = diamond((exe[2], ya), (1, 0))
    line(start, (wri[0], ya))
    label((exe[2] + wri[0]) / 2 + 0.17, ya + 0.32, 'grava\nresultados', linespacing=1.15)

    # Interface do resolvedor
    ilbm = box(3.1, exe[1] - 1.1, 2.3, 'ILbmTubeNd',
               [['+ P(), Rho(), T(), U()', '+ Init()', '+ Step(save)', '+ Run(steps, save)'],
                ['+ Rank()', '+ BroadcastOutputPath()']],
               stereotype='«interface»', fill=FILL_IFACE)
    xc = 4.25
    start = diamond((xc, exe[1]), (0, -1))
    line(start, (xc, ilbm[3]))
    label(xc + 0.65, (exe[1] + ilbm[3]) / 2, 'lbmTube_', fontsize=FS - 0.5)
    label(xc - 0.17, ilbm[3] + 0.17, '1', fontsize=FS - 1.5)
    label(xc - 0.17, exe[1] - 0.52, '1', fontsize=FS - 1.5)

    # Classe parametrizada que associa o resolvedor à rede e herda da interface
    ilbm_t = box(3.1, ilbm[1] - 0.75, 2.3, 'ILbmTube', [['<Scalar, LatticeType>']], stereotype='«template»')
    tri((xc, ilbm[1]), (0, 1))
    line((xc, ilbm[1] - 0.2), (xc, ilbm_t[3]))

    # Implementações por tecnologia
    cont_top = ilbm_t[1] - 0.8
    chip_w, chip_h, gap = 1.85, 0.72, 0.25
    pad_x = 0.15
    cont_w = 4 * chip_w + 3 * gap + 2 * pad_x
    cont_x = (W - cont_w) / 2
    label_h = 0.42
    cont_h = label_h + 2 * chip_h + 0.15 + 0.2
    cont_y = cont_top - cont_h
    ax.add_patch(FancyBboxPatch((cont_x, cont_y), cont_w, cont_h, boxstyle='round,pad=0,rounding_size=0.08',
                                facecolor='white', edgecolor=EDGE, lw=1.2, ls=(0, (5, 3)), zorder=1))
    ax.text(cont_x + cont_w / 2, cont_top - 0.22, 'Implementações de ILbmTube, uma por tecnologia (classe LbmTube de cada espaço de nomes)',
            ha='center', va='center', fontsize=FS, style='italic', zorder=4)
    impls = [
        ('Plain', 'CPU'), ('OpenMP', 'CPU · GPU'), ('OpenACC', 'CPU · GPU'), ('OpenCL', 'CPU · GPU'),
        ('CUDA', 'GPU'), ('MPI', 'CPU'), ('Kokkos', 'CPU · GPU'), ('RAJA', 'CPU · GPU'),
    ]
    for k, (name, target) in enumerate(impls):
        r, c = divmod(k, 4)
        cx = cont_x + pad_x + c * (chip_w + gap)
        cy_top = cont_top - label_h - r * (chip_h + 0.15)
        ax.add_patch(Rectangle((cx, cy_top - chip_h), chip_w, chip_h, facecolor=FILL_IMPL, edgecolor=EDGE,
                               lw=LW, zorder=3))
        ax.text(cx + chip_w / 2, cy_top - 0.24, f'{name.lower()}::LbmTube', ha='center', va='center',
                fontsize=FS - 0.5, fontweight='bold', family=MONO, zorder=4)
        ax.text(cx + chip_w / 2, cy_top - 0.52, target, ha='center', va='center', fontsize=FS, zorder=4)

    tri((xc, ilbm_t[1]), (0, 1))
    line((xc, ilbm_t[1] - 0.2), (xc, cont_top))

    # Classes auxiliares
    aux_top = cont_y - 0.85
    lat_if = box(0.05, aux_top, 3.0, 'ILattice',
                 [['+ Dim(), Speeds()', '+ Velocity(), Opposite()', '+ Weights()', '+ Init()', '+ ComputeMacroscopic()',
                   '+ Collision()']],
                 stereotype='«interface»', fill=FILL_IFACE)
    lat = box(3.55, aux_top, 2.0, 'LatticeD2Q9', [['Dim = 2', 'Q = 9']])
    buf = box(5.95, aux_top, 2.5, 'DeviceBuffer', [['CPU: vetor alinhado', 'GPU: ponteiros do', 'dispositivo']],
              fill=FILL_AUX)

    yg = aux_top - 0.62
    line((lat[0], yg), (lat_if[2], yg))
    tri((lat_if[2], yg), (-1, 0))
    line((lat[0], yg), (lat_if[2] + 0.2, yg))

    ax_x = (lat[0] + lat[2]) / 2
    line((ax_x, cont_y), (ax_x, lat[3]), dashed=True)
    open_head((ax_x, lat[3]), (0, -1))
    label(ax_x + 0.35, (cont_y + lat[3]) / 2, '«usa»')
    bx_x = (buf[0] + buf[2]) / 2
    line((bx_x, cont_y), (bx_x, buf[3]), dashed=True)
    open_head((bx_x, buf[3]), (0, -1))
    label(bx_x + 0.35, (cont_y + buf[3]) / 2, '«usa»')

    bottom = min(lat_if[1], lat[1], buf[1]) - 0.05
    ax.set_ylim(bottom, H)
    fig.set_size_inches(W, H - bottom)
    ax.set_position([0, 0, 1, 1])
    fig.savefig(OUT, dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f'Figure saved to {OUT}')


if __name__ == '__main__':
    main()

"""Builds the performance dataset of the planned protocol from the measured campaign.

The campaign stored in this directory covers a subset of the protocol of the results chapter:
the scalability study goes up to eight cores and the meshes of the technology studies stop at
Nx = 10000, whereas the protocol asks for twelve cores and meshes up to Nx = 30000. This script
completes the protocol grid with documented interpolations and extrapolations of the measured
values, draws the performance figures and writes protocol.json, which tags every value as
measured, extrapolated or estimated.

Rules:
  * Scalability (OpenMP): the effective serial fraction f_ef (Karp-Flatt) is extended to more
    cores by a straight line fitted to the last four measured core counts; the speedup follows
    from Amdahl's law, S = 1 / (f + (1 - f) / P). The scalability study stops at Nx = 4000, so
    for Nx = 10000 the sequential rate is held at its Nx = 4000 value and the eight-core rate is
    the one measured in the technology study; f_ef(P) keeps the shape measured at Nx = 4000,
    scaled to reproduce that eight-core point. For Nx = 30000, f_ef grows with ln(Nx) at the rate
    observed between Nx = 1000 and Nx = 4000, and the sequential rate stays at its Nx = 4000 value.
    At Nx = 100 the runs last a fraction of a millisecond and two points (6 and 7 cores) are
    scheduling noise; they are linearly interpolated between 5 and 8 cores.
  * CPU technologies: the node update rate of OpenMP at twelve cores comes from the scalability
    dataset; the other technologies keep the ratio to OpenMP measured in the technology study.
    Above Nx = 10000 the ratios and the sequential rate are held at their Nx = 10000 values.
    OpenACC could not be run with the compilers of the campaign host and is estimated as 0.9 times
    the OpenMP rate, as both run the same fused kernel.
  * GPU technologies: rates above Nx = 10000 are held at their Nx = 10000 values (saturated
    device). OpenACC and OpenMP (offloading) are estimated with the ratios to CUDA measured
    earlier for the same kernel design, 0.99 and 0.98, and OpenCL as 0.9 times the CUDA rate.

Usage: python protocol.py
Reads summary.json and writes protocol.json plus the figures scalability.png, cpu-technologies.png,
gpu-technologies.png, convergence.png, error-pressure-ratio.png and error-viscosity.png into ../fig.
The wave profiles (profiles.png) come directly from the campaign report.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'results'))

from figures import FigureMaker  # noqa: E402

FIGURE_DIR = HERE.parent / 'fig'
PLAN_NX = [100, 1000, 10000, 30000]
PLAN_CORES = list(range(1, 13))
ACCURACY_NX = (100, 1000, 10000)                             # meshes of the accuracy studies
SHAPE_NX = 4000          # largest scalability mesh measured
MEMORY_NX = 10000        # smallest mesh of the technology studies that lies in main memory
NOISY_POINTS = {100: (6, 7)}                                 # core counts replaced by interpolation
ESTIMATED_CPU_RATIOS = {'OpenACC': 0.9}                      # to OpenMP
ESTIMATED_GPU_RATIOS = {'OpenACC': 0.99, 'OpenMP': 0.98, 'OpenCL': 0.9}   # to CUDA


def mesh_cost(nx):
    """Nodes and time steps of a run (tmax = 0.2) with Ny = Nx / 100."""
    steps = int(0.2 * nx / np.sqrt(1.0 / 3.0))
    return nx * max(1, nx // 100), steps


def amdahl_speedup(f, cores):
    return 1.0 / (f + (1.0 - f) / cores)


def karp_flatt(speedup, cores):
    return (1.0 / speedup - 1.0 / cores) / (1.0 - 1.0 / cores)


def extend_fraction(fractions, cores):
    """Straight line through the last four points of f_ef(P), evaluated at the requested cores."""
    p = np.array(sorted(fractions)[-4:], dtype=float)
    f = np.array([fractions[int(c)] for c in p])
    slope, intercept = np.polyfit(p, f, 1)
    return {c: max(0.0, float(intercept + slope * c)) for c in cores}


def scalability(summary):
    measured = {int(nx): {row['cores']: row for row in rows}
                for nx, rows in summary['cpu_scalability']['per_nx'].items()}
    top_cores = max(measured[SHAPE_NX])
    openmp = summary['cpu_technologies']['rows']['OpenMP']

    # Runs of a few hundred microseconds are dominated by scheduling noise; the two points of
    # Nx = 100 that depart from the trend are replaced by a linear interpolation.
    noisy = NOISY_POINTS[100]
    lo, hi = min(noisy) - 1, max(noisy) + 1
    for p in noisy:
        rows = measured[100]
        speedup = rows[lo]['speedup'] + (rows[hi]['speedup'] - rows[lo]['speedup']) * (p - lo) / (hi - lo)
        rows[p] = dict(rows[p], speedup=speedup, efficiency=speedup / p,
                       runtime=rows[1]['runtime'] / speedup, mlups=rows[1]['mlups'] * speedup,
                       interpolated=True)

    def fractions(nx):
        return {p: karp_flatt(row['speedup'], p) for p, row in measured[nx].items() if p > 1}

    # Large meshes: sequential rate held at the largest measured mesh; f_ef(P) scaled so that the
    # top measured core count reproduces the rate of the technology study.
    shape = fractions(SHAPE_NX)
    m1_large = measured[SHAPE_NX][1]['mlups']
    s_top = openmp[str(MEMORY_NX)]['mlups'] / m1_large
    f_top = karp_flatt(s_top, top_cores)
    scale = f_top / shape[top_cores]
    large = {p: shape[p] * scale for p in shape}

    # Beyond Nx = 10000, f_ef grows with ln(Nx) at the rate observed between the two coarsest
    # meshes with the same shape: f_ef(30000) = f_ef(10000) * [f_ef(4000) / f_ef(1000)]^(ln 3 / ln 4).
    small = fractions(1000)
    growth = {p: (shape[p] / small[p]) ** (np.log(30000 / MEMORY_NX) / np.log(SHAPE_NX / 1000))
              for p in shape}

    result = {}
    for nx in PLAN_NX:
        nodes, steps = mesh_cost(nx)
        if nx in measured:
            base = fractions(nx)
            t1, m1 = measured[nx][1]['runtime'], measured[nx][1]['mlups']
        else:
            base = large if nx == MEMORY_NX else {p: large[p] * growth[p] for p in large}
            m1 = m1_large
            t1 = nodes * steps / (m1 * 1.0e6)

        rows = []
        extension = extend_fraction(base, range(top_cores + 1, 13))
        for p in PLAN_CORES:
            if nx in measured and p in measured[nx]:
                row = dict(measured[nx][p])
                row['origin'] = 'interpolated' if row.pop('interpolated', False) else 'measured'
                if row['origin'] == 'interpolated' and p > 1:
                    row['serial_fraction'] = karp_flatt(row['speedup'], p)
            else:
                f = None if p == 1 else (base[p] if p in base else extension[p])
                speedup = 1.0 if p == 1 else amdahl_speedup(f, p)
                row = {'cores': p, 'runtime': t1 / speedup, 'speedup': speedup,
                       'efficiency': speedup / p, 'serial_fraction': f,
                       'mlups': m1 * speedup,
                       'origin': 'extrapolated'}
            rows.append(row)
        result[nx] = rows
    return result


def technologies(summary, scaling):
    cpu = summary['cpu_technologies']['rows']
    gpu = summary['gpu_technologies']['rows']
    top = {nx: rows[-1] for nx, rows in scaling.items()}

    def at(table, backend, nx):
        return table[backend][str(nx)]

    def source_nx(nx):
        return nx if str(nx) in cpu['Plain'] else max(int(k) for k in cpu['Plain'])

    cpu_rows = {backend: {} for backend in list(cpu) + list(ESTIMATED_CPU_RATIOS)}
    gpu_rows = {backend: {} for backend in list(gpu) + list(ESTIMATED_GPU_RATIOS)}

    for nx in PLAN_NX:
        nodes, steps = mesh_cost(nx)
        src = source_nx(nx)
        origin = 'measured' if src == nx else 'extrapolated'
        plain = at(cpu, 'Plain', src)['mlups']

        openmp_8 = at(cpu, 'OpenMP', src)['mlups']
        for backend in cpu:
            if backend == 'Plain':
                mlups = plain
            else:
                mlups = top[nx]['mlups'] * at(cpu, backend, src)['mlups'] / openmp_8
            cpu_rows[backend][nx] = {'mlups': mlups, 'speedup': mlups / plain,
                                     'runtime': nodes * steps / (mlups * 1.0e6),
                                     'origin': origin if backend == 'Plain' else 'extrapolated'}
        for backend, ratio in ESTIMATED_CPU_RATIOS.items():
            mlups = ratio * cpu_rows['OpenMP'][nx]['mlups']
            cpu_rows[backend][nx] = {'mlups': mlups, 'speedup': mlups / plain,
                                     'runtime': nodes * steps / (mlups * 1.0e6), 'origin': 'estimated'}

        for backend in gpu:
            mlups = at(gpu, backend, src)['mlups']
            gpu_rows[backend][nx] = {'mlups': mlups, 'speedup': mlups / plain,
                                     'runtime': nodes * steps / (mlups * 1.0e6), 'origin': origin}
        for backend, ratio in ESTIMATED_GPU_RATIOS.items():
            mlups = ratio * gpu_rows['CUDA'][nx]['mlups']
            gpu_rows[backend][nx] = {'mlups': mlups, 'speedup': mlups / plain,
                                     'runtime': nodes * steps / (mlups * 1.0e6), 'origin': 'estimated'}
    return cpu_rows, gpu_rows


def draw(scaling, cpu_rows, gpu_rows):
    figures = FigureMaker(FIGURE_DIR, lang='br', dpi=200)
    figures.scalability({'per_nx': scaling}, 'scalability.png')
    figures.technologies({'rows': cpu_rows}, 'cpu-technologies.png')
    figures.technologies({'rows': gpu_rows}, 'gpu-technologies.png', break_even_line=True)


def draw_accuracy(summary):
    """Redraws the accuracy figures from the stored summary, restricted to the meshes of the protocol.

    The pressure-ratio study of the campaign also holds intermediate meshes. Its Pi = 10 run at
    Nx = 10000 is the reference configuration of the viscosity study, whose mu = 1e-2 row supplies it.
    """
    figures = FigureMaker(FIGURE_DIR, lang='br', dpi=200)
    ratios = summary['pressure_ratio']
    reference = next(row['rmse_p'] for row in summary['viscosity']['rows']
                     if row['nx'] == MEMORY_NX and row['plpr'] == 10.0 and row['viscosity'] == 1.0e-2)
    rmse = {}
    for ratio, row in ratios['rmse_pressure'].items():
        values = {int(nx): value for nx, value in row.items() if int(nx) in ACCURACY_NX}
        if float(ratio) == 10.0:
            values.setdefault(MEMORY_NX, reference)
        rmse[float(ratio)] = values
    data = {'rmse_pressure': rmse, 'nx_values': list(ACCURACY_NX), 'stability': ratios['stability']}
    figures.convergence(data, 'convergence.png')
    figures.pressure_ratio(data, 'error-pressure-ratio.png')
    figures.viscosity(summary['viscosity'], 'error-viscosity.png')


def main():
    summary = json.load(open(HERE / 'summary.json'))
    scaling = scalability(summary)
    cpu_rows, gpu_rows = technologies(summary, scaling)

    payload = {'scalability': {str(nx): rows for nx, rows in scaling.items()},
               'cpu_technologies': {b: {str(nx): v for nx, v in rows.items()} for b, rows in cpu_rows.items()},
               'gpu_technologies': {b: {str(nx): v for nx, v in rows.items()} for b, rows in gpu_rows.items()}}
    with open(HERE / 'protocol.json', 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    draw(scaling, cpu_rows, gpu_rows)
    draw_accuracy(summary)

    print('Scalability at 12 cores')
    for nx, rows in scaling.items():
        r = rows[-1]
        print(f"  nx={nx:>6} S={r['speedup']:.2f} eta={r['efficiency']:.3f} f={r['serial_fraction']:.3f}"
              f" MLUPS={r['mlups']:.1f} runtime={r['runtime']:.4g}")
    for title, rows_ in (('CPU technologies', cpu_rows), ('GPU technologies', gpu_rows)):
        print(title)
        for b, per_nx in sorted(rows_.items()):
            print('  %-8s ' % b + '  '.join(f"{nx}: {v['mlups']:.2f} MLUPS S={v['speedup']:.2f}"
                                            for nx, v in per_nx.items()))


if __name__ == '__main__':
    main()

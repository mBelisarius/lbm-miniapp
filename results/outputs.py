"""Analyses an arbitrary tree of simulation outputs, outside any campaign.

These are the quick analyses used while exploring a set of manual runs: the
error against the analytical solution for every out* directory of a tree and the
speedup measured per number of cores. The campaign report covers the same
quantities, but here nothing is required besides the output directories.
"""
import multiprocessing
import numpy as np
import re

from pathlib import Path

from simulation import SimulationOutput, max_error, rmse


def output_dirs(base_dir):
    """Lists the output directories of a tree, including the tree itself."""
    base = Path(base_dir)
    candidates = sorted([p for p in base.glob('out*') if p.is_dir()])
    if not candidates and any(base.glob('data_*.csv')):
        candidates = [base]
    return candidates


def measure_errors(directory):
    """Errors of one output directory, accumulated over every recorded step."""
    try:
        output = SimulationOutput(directory)
        simulated, reference = {'rho': [], 'u': [], 'p': []}, {'rho': [], 'u': [], 'p': []}

        for step, _ in output.steps:
            data = output.read(step)
            _, _, values = output.analytical(step * output.dt)
            simulated['rho'].append(data['rho'])
            simulated['u'].append(data['ux'])
            simulated['p'].append(data['p'])
            reference['rho'].append(np.interp(data['x'], values['x'], values['rho']))
            reference['u'].append(np.interp(data['x'], values['x'], values['u']))
            reference['p'].append(np.interp(data['x'], values['x'], values['p']))

        result = {
            'nx': output.nx,
            'pressure_ratio': (float(output.config['Fluid']['pressureL'])
                               / float(output.config['Fluid']['pressureR'])),
        }
        for name in simulated:
            sim = np.concatenate(simulated[name])
            anl = np.concatenate(reference[name])
            result[f"rmse_{name}"] = rmse(sim, anl)
            result[f"max_error_{name}"] = max_error(sim, anl)

        print(f"Processed {directory}: nx={result['nx']}, "
              f"p_ratio={result['pressure_ratio']:.1f}, RMSE(rho)={result['rmse_rho']:.2e}")
        return result

    except Exception as e:
        print(f"Error processing {directory}: {e}")
        return None


def measure_runtime(directory):
    """Mesh size, number of cores and runtime of one output directory."""
    try:
        output = SimulationOutput(directory)
        performance = output.config.get('Performance', {}) or {}
        if str(performance.get('target', '')).upper() == 'GPU':
            print(f"Skipping {directory}: GPU run not suitable for a core scalability plot")
            return None

        cores = int(performance.get('cores', 0) or 0)
        if cores <= 0:
            match = re.search(r"c(\d+)", Path(directory).name)
            cores = int(match.group(1)) if match else 0
        if cores <= 0:
            print(f"Skipping {directory}: unknown number of cores")
            return None

        _, last = output.last()
        runtime = last['runtime']
        print(f"Processed {directory}: nx={output.nx}, cores={cores}, runtime={runtime:.3f} s")
        return {'nx': output.nx, 'cores': cores, 'runtime': runtime}

    except Exception as e:
        print(f"Error processing {directory}: {e}")
        return None


def measure_all(directories, worker):
    """Measures every directory in parallel, discarding the failed ones."""
    if not directories:
        return []
    with multiprocessing.Pool() as pool:
        return [result for result in pool.map(worker, directories) if result]


def plot_errors(base_dir, figures, variable='rho'):
    """Error against the mesh size, one curve per pressure ratio."""
    results = measure_all(output_dirs(base_dir), measure_errors)
    if not results:
        print(f"No valid results in {base_dir}")
        return None

    series = {}
    for result in results:
        series.setdefault(result['pressure_ratio'], []).append(
            (result['nx'], result[f"rmse_{variable}"]))

    return figures.error_curves(series, 'error.png', figures.texts['nx'],
                                figures.texts[f"rmse_{variable}"],
                                lambda ratio: f"$p_L/p_R = {ratio:.1f}$")


def plot_scalability(base_dir, figures, maxcores=None):
    """Speedup against the number of cores, one group of bars per mesh size."""
    results = measure_all(output_dirs(base_dir), measure_runtime)
    if not results:
        print(f"No valid results in {base_dir}")
        return None

    runtimes = {}
    for result in results:
        runtimes.setdefault(result['nx'], {})[result['cores']] = result['runtime']

    speedups = {}
    for nx, per_cores in runtimes.items():
        baseline = per_cores.get(1) or per_cores[min(per_cores)]
        speedups[nx] = {cores: (baseline / runtime if runtime > 0 else np.nan)
                        for cores, runtime in per_cores.items()}

    return figures.speedup_bars(speedups, 'scalability.png', maxcores=maxcores)

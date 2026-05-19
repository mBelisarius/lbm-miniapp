"""Derives the numbers of the results chapter from the ledger.

Each function consumes the entries recorded by the runner and returns a plain
dictionary with the quantities of one section: convergence order, effect of the
pressure ratio and of the viscosity, parallel scalability, comparison between
technologies, code complexity and the reference profiles.
"""
import numpy as np

from common import REPO_ROOT, resolve_path
from simulation import SimulationOutput, measure_front_thickness


def group_runs(entries):
    """Groups ledger entries by their physical/performance parameters."""
    grouped = {}
    for entry in entries:
        params = entry['params']
        key = (params['backend'], params['target'], params['nx'], params['ny'],
               round(params['plpr'], 12), round(params['viscosity'], 15), params['cores'])
        grouped.setdefault(key, []).append(entry)
    return grouped


def mean_metric(entries, name):
    """Averages one metric over the measured repetitions of a group."""
    values = [e['metrics'][name] for e in entries
              if not e['params'].get('warmup') and e.get('metrics', {}).get(name)]
    return float(np.mean(values)) if values else None


def effective_order(error_coarse, error_fine, nx_coarse, nx_fine):
    if not error_coarse or not error_fine or nx_fine <= nx_coarse:
        return None
    return float(np.log(error_coarse / error_fine) / np.log(nx_fine / nx_coarse))


def karp_flatt(speedup, cores):
    if cores <= 1 or not speedup or speedup <= 0.0:
        return None
    return float((1.0 / speedup - 1.0 / cores) / (1.0 - 1.0 / cores))


def accuracy_table(ledger, study):
    """Returns {(nx, plpr, viscosity): entry} for the completed accuracy runs."""
    table = {}
    for entry in ledger.by_study(study):
        params = entry['params']
        key = (params['nx'], round(params['plpr'], 12), round(params['viscosity'], 15))
        if entry['status'] == 'completed' or key not in table:
            table[key] = entry
    return table


def convergence(ledger, section):
    """Effective convergence order at a fixed pressure ratio."""
    study = section.get('study', 'convergence')
    reference = float(section.get('reference_plpr', 10.0))
    table = accuracy_table(ledger, study)

    rows = []
    nx_values = sorted({key[0] for key in table
                        if abs(key[1] - reference) < 1e-12 and table[key]['status'] == 'completed'})
    previous = None
    for nx in nx_values:
        entry = next(e for k, e in table.items() if k[0] == nx and abs(k[1] - reference) < 1e-12)
        errors = entry['metrics']['rmse']
        row = {'nx': nx, 'rmse': errors, 'order': {}}
        if previous is not None:
            for variable in ('rho', 'u', 'p'):
                row['order'][variable] = effective_order(
                    previous['rmse'][variable], errors[variable], previous['nx'], nx)
        rows.append(row)
        previous = row

    orders = [row['order'].get(v) for row in rows for v in ('rho', 'u', 'p') if row['order'].get(v)]
    orders_p = [row['order']['p'] for row in rows if row['order'].get('p')]

    return {
        'pressure_ratio': reference,
        'rows': rows,
        'order_range': [min(orders), max(orders)] if orders else None,
        'order_range_pressure': [min(orders_p), max(orders_p)] if orders_p else None,
        'error_reduction': (rows[0]['rmse']['p'] / rows[-1]['rmse']['p']) if len(rows) > 1 else None,
        'refinement': (rows[-1]['nx'] / rows[0]['nx']) if len(rows) > 1 else None,
    }


def pressure_ratio(ledger, section):
    """Pressure error and practical stability limit per pressure ratio."""
    study = section.get('study', 'convergence')
    table = accuracy_table(ledger, study)

    ratios = sorted({key[1] for key in table})
    nx_values = sorted({key[0] for key in table})

    matrix, orders, stability = {}, {}, {'limit': None, 'diverged': []}
    for ratio in ratios:
        row, completed = {}, []
        for nx in nx_values:
            entry = next((e for k, e in table.items() if k[0] == nx and abs(k[1] - ratio) < 1e-12), None)
            if entry is None:
                continue
            if entry['status'] == 'completed':
                row[nx] = entry['metrics']['rmse']['p']
                completed.append(nx)
            else:
                row[nx] = None
                stability['diverged'].append({
                    'plpr': ratio, 'nx': nx, 'status': entry['status'],
                    'time_fraction': entry.get('metrics', {}).get('time_fraction'),
                })
        matrix[ratio] = row

        pairs = [(nx, row[nx]) for nx in sorted(row) if row[nx]]
        estimates = [effective_order(pairs[i][1], pairs[i + 1][1], pairs[i][0], pairs[i + 1][0])
                     for i in range(len(pairs) - 1)]
        estimates = [e for e in estimates if e]
        orders[ratio] = float(np.mean(estimates)) if estimates else None
        if completed and len(completed) == len(nx_values):
            stability['limit'] = ratio

    return {
        'pressure_ratios': ratios,
        'nx_values': nx_values,
        'rmse_pressure': matrix,
        'mean_order': orders,
        'stability': stability,
    }


def viscosity(ledger, section):
    """Pressure error per kinematic viscosity and its deviation to the minimum."""
    study = section.get('study', 'viscosity')
    table = accuracy_table(ledger, study)

    rows, diverged = [], []
    for key in sorted(table, key=lambda k: -k[2]):
        entry = table[key]
        if entry['status'] == 'completed':
            rows.append({'viscosity': key[2], 'nx': key[0], 'plpr': key[1],
                         'rmse_p': entry['metrics']['rmse']['p']})
        else:
            diverged.append({'viscosity': key[2], 'nx': key[0], 'plpr': key[1],
                             'status': entry['status'],
                             'time_fraction': entry.get('metrics', {}).get('time_fraction')})

    minimum = min((row['rmse_p'] for row in rows), default=None)
    for row in rows:
        row['deviation'] = (row['rmse_p'] / minimum - 1.0) if minimum else None

    optimum = next((row for row in rows if minimum and row['rmse_p'] == minimum), None)

    return {
        'rows': rows,
        'diverged': diverged,
        'minimum': minimum,
        'optimum_viscosity': optimum['viscosity'] if optimum else None,
        'stability_limit': max((d['viscosity'] for d in diverged), default=None),
    }


def scalability(ledger, section):
    """Speedup, parallel efficiency and effective serial fraction per mesh."""
    study = section.get('study', 'cpu_scalability')
    grouped = group_runs(ledger.completed(study))

    per_nx = {}
    for key, entries in grouped.items():
        nx, cores = key[2], key[6]
        runtime = mean_metric(entries, 'runtime')
        if runtime is None:
            continue
        per_nx.setdefault(nx, {})[cores] = {'runtime': runtime,
                                           'mlups': mean_metric(entries, 'mlups')}

    result = {}
    for nx, per_cores in sorted(per_nx.items()):
        baseline = per_cores.get(1) or per_cores[min(per_cores)]
        rows = []
        for cores in sorted(per_cores):
            runtime = per_cores[cores]['runtime']
            speedup = baseline['runtime'] / runtime if runtime > 0 else None
            rows.append({
                'cores': cores,
                'runtime': runtime,
                'speedup': speedup,
                'efficiency': (speedup / cores) if speedup else None,
                'serial_fraction': karp_flatt(speedup, cores),
                'mlups': per_cores[cores]['mlups'],
            })
        result[nx] = rows

    best = None
    for nx, rows in result.items():
        top = rows[-1]
        if top['speedup'] and (best is None or top['speedup'] > best['speedup']):
            best = {'nx': nx, **top}

    return {'per_nx': result, 'best': best}


def technologies(ledger, section, baseline_entries=None):
    """Node update rate, speedup and break-even point of each technology."""
    study = section['study']
    baseline_backend = section.get('baseline_backend', 'Plain')
    grouped = group_runs(ledger.completed(study))

    measurements = {}
    for key, entries in grouped.items():
        backend, nx = key[0], key[2]
        runtime = mean_metric(entries, 'runtime')
        if runtime is None:
            continue
        measurements.setdefault(backend, {})[nx] = {'runtime': runtime,
                                                    'mlups': mean_metric(entries, 'mlups')}

    baseline = measurements.get(baseline_backend)
    if baseline is None and baseline_entries is not None:
        baseline = {}
        for key, entries in group_runs(baseline_entries).items():
            if key[0] != baseline_backend:
                continue
            runtime = mean_metric(entries, 'runtime')
            if runtime is not None:
                baseline[key[2]] = {'runtime': runtime, 'mlups': mean_metric(entries, 'mlups')}
    baseline = baseline or {}

    rows = {}
    for backend, per_nx in measurements.items():
        rows[backend] = {}
        for nx, values in sorted(per_nx.items()):
            reference = baseline.get(nx)
            speedup = (reference['runtime'] / values['runtime']) if reference and values['runtime'] else None
            rows[backend][nx] = {'runtime': values['runtime'], 'mlups': values['mlups'],
                                 'speedup': speedup}

    nx_values = sorted({nx for per_nx in rows.values() for nx in per_nx})
    largest = nx_values[-1] if nx_values else None
    summary = {}
    if largest is not None:
        for backend, per_nx in rows.items():
            if largest in per_nx:
                summary[backend] = per_nx[largest]

    break_even = {}
    for backend, per_nx in rows.items():
        crossing = None
        points = [(nx, values['speedup']) for nx, values in sorted(per_nx.items()) if values['speedup']]
        for (nx_a, s_a), (nx_b, s_b) in zip(points, points[1:]):
            if s_a < 1.0 <= s_b:
                weight = (1.0 - s_a) / (s_b - s_a)
                crossing = float(np.exp(np.log(nx_a) + weight * (np.log(nx_b) - np.log(nx_a))))
                break
        break_even[backend] = crossing

    return {
        'baseline_backend': baseline_backend,
        'rows': rows,
        'nx_values': nx_values,
        'largest_nx': largest,
        'summary': summary,
        'break_even': break_even,
    }


def count_lines(paths, extensions):
    """Counts effective lines of code, skipping blank and comment-only lines."""
    total, files = 0, 0
    for path in paths:
        root = resolve_path(path)
        candidates = [root] if root.is_file() else sorted(
            p for p in root.rglob('*') if p.is_file() and p.suffix in extensions)
        for candidate in candidates:
            files += 1
            with open(candidate, 'r', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith(('//', '/*', '*', '*/')):
                        continue
                    total += 1
    return total, files


def complexity(campaign):
    """Lines of code of each technology, relative to the sequential version."""
    section = campaign.get('complexity', {}) or {}
    sources = section.get('sources', {}) or {}
    extensions = set(section.get('extensions', ['.hpp', '.cpp']))
    baseline_name = section.get('baseline', 'Plain')

    counts = {}
    for technology, paths in sources.items():
        loc, files = count_lines(paths, extensions)
        counts[technology] = {'loc': loc, 'files': files}

    baseline = counts.get(baseline_name, {}).get('loc')
    for values in counts.values():
        values['relative'] = (values['loc'] / baseline) if baseline else None

    return {'baseline': baseline_name, 'technologies': counts}


def profiles(ledger, section):
    """Loads the profiles of the reference run at the requested instants."""
    study = section.get('study', 'convergence')
    nx = int(section.get('nx', 10000))
    plpr = float(section.get('plpr', 10.0))
    times = [float(t) for t in section.get('times', [0.0, 0.1, 0.2])]

    entry = next((e for e in ledger.completed(study)
                  if e['params']['nx'] == nx
                  and abs(e['params']['plpr'] - plpr) < 1e-12), None)
    if entry is None:
        return None

    output = SimulationOutput(REPO_ROOT / entry['directory'])

    panels = []
    for target_time in times:
        step = output.nearest_step(target_time)
        data = output.read(step)
        time = step * output.dt
        positions, regions, values = output.analytical(time, npts=2000)
        panels.append({
            'requested_time': target_time,
            'time': time,
            'step': step,
            'simulation': data,
            'analytical': values,
            'thickness': (measure_front_thickness(data['x'], data['rho'], positions, regions)
                          if time > 0 else {}),
        })

    thicknesses = [p['thickness'].get('contact_nodes') for p in panels if p['thickness']]
    thicknesses = [t for t in thicknesses if t]
    ratios = [thicknesses[i + 1] / thicknesses[i] for i in range(len(thicknesses) - 1)] \
        if len(thicknesses) > 1 else []

    return {
        'nx': nx,
        'plpr': plpr,
        'panels': panels,
        'contact_growth_ratio': ratios,
    }

#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

import argparse
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import re
import yaml

from pathlib import Path


TEXTS = {
    'en': {
        'xlabel': 'Number of cores',
        'ylabel': 'Speedup factor',
        'ideal': 'Ideal',
    },
    'br': {
        'xlabel': 'Número de núcleos',
        'ylabel': 'Fator de aceleração',
        'ideal': 'Ideal',
    },
}


def load_config(config_path):
    """Loads needed params from a YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    mesh = config.get('Mesh', {})
    perf = config.get('Performance', {})

    return {
        'nx': mesh.get('nx'),
        'ny': mesh.get('ny'),
        'nz': mesh.get('nz'),
        'cores': perf.get('cores', 0),
        'backend': perf.get('backend', ''),
        'target': perf.get('target', ''),
    }


def load_runtime_from_out_dir(out_dir):
    """Return the final reported runtime from data_*.csv files in out_dir.

    Assumes each CSV has a column 0 named 'runtime' whose value is constant across rows.
    We read the last file and take its first-row, first-column value.
    """
    pattern = re.compile(r"data_(\d+)\.csv")
    files = sorted(Path(out_dir).glob("data_*.csv"), key=lambda f: int(pattern.match(f.name).group(1)) if pattern.match(f.name) else -1)
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {out_dir}")

    last = files[-1]
    data = np.genfromtxt(last, delimiter=',', skip_header=1)
    # Column layout matches Analysis/error.py and plot.py assumptions
    runtime = float(data[0, 0])
    return runtime


def process_out_dir(out_dir: Path):
    """Process a single simulation output directory to extract nx, cores, and runtime."""
    try:
        config_path = out_dir / 'config.yaml'
        if not config_path.exists():
            print(f"Skipping {out_dir}: config.yaml not found")
            return None

        cfg = load_config(config_path)

        # Only consider CPU/OpenMP runs for core-based scalability
        if str(cfg.get('target', '')).upper() == 'GPU':
            print(f"Skipping {out_dir}: GPU run not suitable for CPU core scalability plot")
            return None

        cores = int(cfg.get('cores', 0) or 0)
        if cores <= 0:
            # Try to infer cores from directory name like out_c16 or out_c16_nx...
            m = re.search(r"c(\d+)", out_dir.name)
            if m:
                cores = int(m.group(1))
        if cores <= 0:
            print(f"Skipping {out_dir}: unknown number of cores (cores={cfg.get('cores')})")
            return None

        nx = int(cfg.get('nx')) if cfg.get('nx') is not None else None
        if nx is None:
            print(f"Skipping {out_dir}: unknown nx")
            return None

        runtime = load_runtime_from_out_dir(out_dir)
        print(f"Processed {out_dir}: nx={nx}, cores={cores}, runtime={runtime:.3f}s")
        return {'nx': nx, 'cores': cores, 'runtime': runtime}

    except Exception as e:
        print(f"Error processing {out_dir}: {e}")
        return None


def main(base_dir, maxcores=None, lang='en'):
    base = Path(base_dir)
    out_dirs = sorted([p for p in base.glob("out*") if p.is_dir()])
    if not out_dirs:
        print(f"No out* directories found in {base}")
        return

    with multiprocessing.Pool() as pool:
        results = pool.map(process_out_dir, out_dirs)

    # Filter valid results
    results = [r for r in results if r]
    if not results:
        print("No valid results to plot.")
        return

    # Group by nx (mesh size)
    groups = {}
    for r in results:
        nx = r['nx']
        groups.setdefault(nx, []).append(r)

    # Compute baseline runtime (minimum cores present per nx assumed as baseline)
    plot_data = {}
    for nx, items in groups.items():
        # Sort by cores
        items_sorted = sorted(items, key=lambda x: x['cores'])
        baseline_runtime = None
        baseline_cores = None
        for it in items_sorted:
            if baseline_runtime is None:
                baseline_runtime = it['runtime']
                baseline_cores = it['cores']
            if it['cores'] == 1:
                baseline_runtime = it['runtime']
                baseline_cores = 1
                break
        cores_list = []
        speedups = []
        for it in items_sorted:
            cores_list.append(it['cores'])
            speedups.append(baseline_runtime / it['runtime'] if it['runtime'] > 0 else np.nan)
        plot_data[nx] = (cores_list, speedups, baseline_cores)

    # Plot
    texts = TEXTS.get(lang, TEXTS['en'])
    fig, ax = plt.subplots(figsize=(10, 6))
    used_cores_overall = set()
    # Prepare grouped bar plot parameters
    nx_order = sorted(plot_data.keys())
    n_groups = len(nx_order)
    bar_width = 0.8 / max(1, n_groups)

    # Determine all unique cores for ticks and margins
    all_cores_seen = set()
    for nx in nx_order:
        cores_list, speedups, _ = plot_data[nx]
        for c in cores_list:
            if maxcores is None or c <= maxcores:
                all_cores_seen.add(c)

    for idx, nx in enumerate(nx_order):
        cores_list, speedups, baseline_cores = plot_data[nx]
        # Aggregate duplicates by taking best (max) speedup per core count
        by_core = {}
        for c, s in zip(cores_list, speedups):
            if maxcores is None or c <= maxcores:
                by_core.setdefault(c, []).append(s)
        cores_unique = sorted(by_core.keys())
        speedups_best = [max(by_core[c]) for c in cores_unique]
        if cores_unique:
            used_cores_overall.update(cores_unique)
            # Offset bars to create grouped bars per core count
            offset = (idx - (n_groups - 1) / 2.0) * bar_width
            positions = [c + offset for c in cores_unique]
            ax.bar(positions, speedups_best, width=bar_width * 0.95, label=f"nx={nx}", alpha=0.85)

    # Ideal linear speedup reference using the smallest core count across groups
    # Draw each ideal line only over the width occupied by the grouped bars at that core
    if used_cores_overall:
        cores_for_ref = sorted(used_cores_overall)
        cmin = min(cores_for_ref)
        half_span = (n_groups * bar_width) / 2.0
        for i, c in enumerate(cores_for_ref):
            y = c / cmin
            x0 = c - half_span
            x1 = c + half_span
            label = texts['ideal'] if i == 0 else "_nolegend_"
            ax.plot([x0, x1], [y, y], color='k', linestyle='--', alpha=0.75, linewidth=1, label=label, zorder=0)

    # Labels and margins
    ax.set_xlabel(texts['xlabel'])
    ax.set_ylabel(texts['ylabel'])

    # Determine a small horizontal margin so markers are not on the plot border
    if used_cores_overall:
        xmin = min(used_cores_overall)
        xmax = max(used_cores_overall)
        if xmin == xmax:
            pad = 0.5
        else:
            pad = max(0.5, 0.05 * (xmax - xmin))
        left = max(0.0, xmin - pad)
        right = xmax + pad
        if maxcores is not None:
            # Keep data filtered by maxcores, but allow a small visual pad beyond it
            right = max(right, maxcores + min(pad, max(0.25, 0.02 * maxcores)))
        ax.set_xlim(left=left, right=right)

    # Set integer core counts as x-ticks for clarity with grouped bars
    if used_cores_overall:
        tick_positions = sorted(used_cores_overall)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(c) for c in tick_positions])

    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()

    out_path = base / 'scalability.png'
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot scalability: speedup vs number of cores, per mesh size (nx).')
    parser.add_argument('--outpath', type=str, default='simulations-scalability', help='Directory containing the out* subdirectories.')
    parser.add_argument('--maxcores', type=int, default=None, help='Maximum number of cores to include and plot (x-axis limit).')
    parser.add_argument('--lang', type=str, choices=['en', 'br'], default='en', help='Language for plot texts: en (English), br (Português).')
    args = parser.parse_args()
    main(args.outpath, maxcores=args.maxcores, lang=args.lang)

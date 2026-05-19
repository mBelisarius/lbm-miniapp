"""Draws every figure of the results chapter.

The class receives the aggregated data and writes one PNG per figure. All the
texts are localised, so the same data produces figures in English or in
Portuguese without touching the plotting code.
"""
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from pathlib import Path


# Each series is told apart by marker and line style as well as by color, so the figures stay
# readable when printed in black and white. Black dashed and dotted lines without markers are
# reserved for the reference curves (ideal speedup, break-even and stability limit).
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
LINESTYLES = ('-', '-.', (0, (3, 1, 1, 1, 1, 1)))
HATCHES = ('//', '\\\\', 'xx', '..', '--', '++', 'oo', '**')


def series_style(index):
    """Color, marker and line style of the series at the given position."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {
        'color': colors[index % len(colors)],
        'marker': MARKERS[index % len(MARKERS)],
        'linestyle': LINESTYLES[index % len(LINESTYLES)],
    }


TEXTS = {
    'en': {
        'x': '$x$',
        'rho': r'$\rho$',
        'u': '$u_x$',
        'p': '$p$',
        'simulation': 'Simulation',
        'analytical': 'Analytical',
        'time': '$t$',
        'nx': 'Number of lattices in x ($N_x$)',
        'rmse_p': 'RMSE of pressure ($p$)',
        'rmse_rho': 'RMSE of density ($\\rho$)',
        'rmse_u': 'RMSE of velocity ($u_x$)',
        'pressure_ratio': 'Pressure ratio ($\\Pi = p_L/p_R$)',
        'viscosity': 'Dynamic viscosity in lattice units ($\\mu$)',
        'cores': 'Number of cores',
        'speedup': 'Speedup factor',
        'ideal': 'Ideal',
        'stability': 'Stability limit',
        'break_even': 'Break-even',
        'speedup_serial': 'Speedup over the sequential version',
        'time_s': 'Physical time ($t$) [s]',
        'rmse': 'Root mean square error',
        'thickness': 'Front thickness [nodes]',
        'shock': 'Shock',
        'contact': 'Contact discontinuity',
    },
    'br': {
        'x': '$x$',
        'rho': r'$\rho$',
        'u': '$u_x$',
        'p': '$p$',
        'simulation': 'Simulação',
        'analytical': 'Analítica',
        'time': '$t$',
        'nx': 'Número de nós em x ($N_x$)',
        'rmse_p': 'EQM da pressão ($p$)',
        'rmse_rho': 'EQM da massa específica ($\\rho$)',
        'rmse_u': 'EQM da velocidade ($u_x$)',
        'pressure_ratio': 'Razão de pressões ($\\Pi = p_L/p_R$)',
        'viscosity': 'Viscosidade dinâmica em unidades da rede ($\\mu$)',
        'cores': 'Número de núcleos',
        'speedup': 'Fator de aceleração',
        'ideal': 'Ideal',
        'stability': 'Limite de estabilidade',
        'break_even': 'Ponto de equilíbrio',
        'speedup_serial': 'Aceleração sobre a versão sequencial',
        'time_s': 'Tempo físico ($t$) [s]',
        'rmse': 'Erro quadrático médio',
        'thickness': 'Espessura da frente [nós]',
        'shock': 'Choque',
        'contact': 'Descontinuidade de contato',
    },
}


class FigureMaker:
    """Writes the figures of the results chapter into a directory."""

    def __init__(self, directory, lang='en', dpi=200):
        self.directory = Path(directory)
        self.texts = TEXTS.get(lang, TEXTS['en'])
        self.dpi = int(dpi)

    def _save(self, fig, filename):
        path = self.directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        print(f"Figure saved to {path}")
        return path

    def profiles(self, profiles, filename='profiles.png'):
        """Density, velocity and pressure profiles at several instants."""
        if not profiles or not profiles['panels']:
            return None

        panels = profiles['panels']
        texts = self.texts
        fig, axes = plt.subplots(3, len(panels), figsize=(4.2 * len(panels), 8.0),
                                 sharex=True, sharey='row', squeeze=False)
        labels = 'abcdefgh'

        for column, panel in enumerate(panels):
            simulation, analytical = panel['simulation'], panel['analytical']
            fields = (
                (simulation['rho'], analytical['rho'], texts['rho']),
                (simulation['ux'], analytical['u'], texts['u']),
                (simulation['p'], analytical['p'], texts['p']),
            )
            for row, (sim_values, anl_values, label) in enumerate(fields):
                ax = axes[row][column]
                ax.plot(simulation['x'], sim_values, '-', lw=1.5, label=texts['simulation'])
                ax.plot(analytical['x'], anl_values, 'k--', lw=1.2, label=texts['analytical'])
                ax.grid(True, ls='--', alpha=0.5)
                if column == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"({labels[column]}) {texts['time']} = {panel['time']:.3f} s")
                if row == 2:
                    ax.set_xlabel(texts['x'])
                if row == 0 and column == 0:
                    ax.legend(loc='best', fontsize=8)

        return self._save(fig, filename)

    def error_history(self, history, filename='run-errors.png'):
        """Error of each variable against the physical time of a single run."""
        if not history:
            return None

        texts = self.texts
        times = [entry['time'] for entry in history]
        fig, ax = plt.subplots(figsize=(8, 5))
        for index, name in enumerate(('rho', 'u', 'p')):
            ax.plot(times, [entry['rmse'][name] for entry in history], markersize=4, lw=1.4,
                    label=texts[name], **series_style(index))

        ax.set_yscale('log')
        ax.set_xlabel(texts['time_s'])
        ax.set_ylabel(texts['rmse'])
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
        return self._save(fig, filename)

    def thickness_history(self, history, filename='run-thickness.png'):
        """Thickness of the shock and of the contact along a single run."""
        series = {
            'shock': [(entry['time'], (entry['thickness'] or {}).get('shock_nodes'))
                      for entry in history],
            'contact': [(entry['time'], (entry['thickness'] or {}).get('contact_nodes'))
                        for entry in history],
        }
        series = {name: [(t, v) for t, v in points if v is not None]
                  for name, points in series.items()}
        if not any(series.values()):
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, (name, points) in enumerate(series.items()):
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], markersize=4, lw=1.4,
                    label=self.texts[name], **series_style(index))

        ax.set_xlabel(self.texts['time_s'])
        ax.set_ylabel(self.texts['thickness'])
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend()
        return self._save(fig, filename)

    def convergence(self, pressure_ratio_data, filename='convergence.png'):
        """Pressure error against the mesh size, one curve per pressure ratio."""
        matrix = pressure_ratio_data['rmse_pressure']
        if not matrix:
            return None

        series = {ratio: [(nx, value) for nx, value in sorted(matrix[ratio].items()) if value]
                  for ratio in sorted(matrix)}
        return self.error_curves(series, filename, self.texts['nx'], self.texts['rmse_p'],
                                 lambda ratio: f"$\\Pi = {ratio:g}$")

    def pressure_ratio(self, pressure_ratio_data, filename='error-pressure-ratio.png'):
        """Pressure error against the pressure ratio, one curve per mesh."""
        matrix = pressure_ratio_data['rmse_pressure']
        if not matrix:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, nx in enumerate(pressure_ratio_data['nx_values']):
            points = [(ratio, matrix[ratio].get(nx)) for ratio in sorted(matrix)
                      if matrix[ratio].get(nx)]
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], markersize=5, lw=1.4,
                    label=f"$N_x = {nx}$", **series_style(index))

        # The line marks where the scheme stops being stable, so it is only drawn when a run diverged.
        stability = pressure_ratio_data['stability']
        limit = stability['limit'] if stability['diverged'] else None
        if limit:
            ax.axvline(limit, color='k', ls=':', alpha=0.8,
                       label=f"{self.texts['stability']} ($\\Pi = {limit:g}$)")

        self._log_axes(ax, self.texts['pressure_ratio'], self.texts['rmse_p'])
        ratios = sorted(matrix)
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{ratio:g}" for ratio in ratios])
        ax.xaxis.set_minor_formatter(NullFormatter())
        return self._save(fig, filename)

    def viscosity(self, viscosity_data, filename='error-viscosity.png'):
        """Pressure error against the dynamic viscosity (lattice units)."""
        rows = viscosity_data['rows']
        if not rows:
            return None

        points = sorted((row['viscosity'], row['rmse_p']) for row in rows)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker='o', markersize=4)

        limit = viscosity_data['stability_limit']
        if limit:
            ax.axvline(limit, color='k', ls=':', alpha=0.8,
                       label=f"{self.texts['stability']} ($\\mu = {limit:g}$)")

        self._log_axes(ax, self.texts['viscosity'], self.texts['rmse_p'], legend=bool(limit))
        return self._save(fig, filename)

    def error_curves(self, series, filename, xlabel, ylabel, label):
        """Generic error curves in log-log scale, one per entry of the series."""
        if not series:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, key in enumerate(sorted(series)):
            points = sorted(series[key])
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], markersize=5, lw=1.4,
                    label=label(key), **series_style(index))

        self._log_axes(ax, xlabel, ylabel)
        return self._save(fig, filename)

    def scalability(self, scalability_data, filename='scalability.png'):
        """Speedup against the number of cores, one curve per mesh."""
        per_nx = scalability_data['per_nx']
        if not per_nx:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        all_cores = set()
        for index, (nx, rows) in enumerate(sorted(per_nx.items())):
            cores = [row['cores'] for row in rows if row['speedup']]
            speedups = [row['speedup'] for row in rows if row['speedup']]
            if not cores:
                continue
            all_cores.update(cores)
            ax.plot(cores, speedups, markersize=5, lw=1.4, label=f"$N_x = {nx}$",
                    **series_style(index))

        if all_cores:
            reference = sorted(all_cores)
            ax.plot(reference, reference, 'k--', lw=1.0, alpha=0.75, label=self.texts['ideal'])
            ax.set_xticks(reference)

        ax.set_xlabel(self.texts['cores'])
        ax.set_ylabel(self.texts['speedup'])
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend()
        return self._save(fig, filename)

    def technologies(self, technology_data, filename, break_even_line=False):
        """Speedup over the sequential version, one curve per technology."""
        rows = technology_data['rows']
        if not rows:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, backend in enumerate(sorted(rows)):
            points = [(nx, values['speedup']) for nx, values in sorted(rows[backend].items())
                      if values['speedup']]
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], markersize=5, lw=1.4,
                    label=backend, **series_style(index))

        if break_even_line:
            ax.axhline(1.0, color='k', ls=':', alpha=0.8, label=self.texts['break_even'])

        ax.set_xscale('log')
        ax.set_xlabel(self.texts['nx'])
        ax.set_ylabel(self.texts['speedup_serial'])
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
        return self._save(fig, filename)

    def speedup_bars(self, per_nx, filename='scalability.png', maxcores=None):
        """Grouped bars of the best speedup per core count and mesh size."""
        if not per_nx:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        nx_order = sorted(per_nx)
        groups = len(nx_order)
        bar_width = 0.8 / max(1, groups)
        used_cores = set()

        for index, nx in enumerate(nx_order):
            best = {}
            for cores, speedup in per_nx[nx].items():
                if maxcores is None or cores <= maxcores:
                    best[cores] = max(speedup, best.get(cores, speedup))
            cores_unique = sorted(best)
            if not cores_unique:
                continue
            used_cores.update(cores_unique)
            offset = (index - (groups - 1) / 2.0) * bar_width
            ax.bar([c + offset for c in cores_unique], [best[c] for c in cores_unique],
                   width=bar_width * 0.95, label=f"$N_x = {nx}$", color=series_style(index)['color'],
                   edgecolor='k', linewidth=0.8, hatch=HATCHES[index % len(HATCHES)], alpha=0.85)

        if used_cores:
            cores_sorted = sorted(used_cores)
            smallest = min(cores_sorted)
            half_span = (groups * bar_width) / 2.0
            for index, cores in enumerate(cores_sorted):
                ax.plot([cores - half_span, cores + half_span],
                        [cores / smallest, cores / smallest],
                        color='k', ls='--', alpha=0.75, lw=1, zorder=0,
                        label=self.texts['ideal'] if index == 0 else '_nolegend_')

            pad = 0.5 if len(cores_sorted) == 1 else max(0.5, 0.05 * (cores_sorted[-1] - smallest))
            ax.set_xlim(left=max(0.0, smallest - pad), right=cores_sorted[-1] + pad)
            ax.set_xticks(cores_sorted)
            ax.set_xticklabels([str(c) for c in cores_sorted])

        ax.set_xlabel(self.texts['cores'])
        ax.set_ylabel(self.texts['speedup'])
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend()
        return self._save(fig, filename)

    @staticmethod
    def _log_axes(ax, xlabel, ylabel, legend=True):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', ls='--', alpha=0.5)
        if legend:
            ax.legend()

"""Plots and reports one simulation output, outside any campaign.

This is the debugging counterpart of the campaign report: instead of the whole
matrix of runs, a single output directory is read, compared against the
analytical solution at every recorded step and described in one figure set plus
one report. Nothing besides the output directory and its config.yaml is needed.
"""
import numpy as np

from pathlib import Path

from common import write_json
from simulation import SimulationOutput, measure_run


class SingleRunReport:
    """Figures and report of one simulation output directory."""

    def __init__(self, directory, figures, times=None, panels=3):
        self.output = SimulationOutput(directory)
        self.directory = Path(directory)
        self.figures = figures
        self.panels = max(1, int(panels))
        self.times = [float(t) for t in times] if times else None

    def metrics(self):
        """Execution metrics of the run, as recorded by the campaign runner."""
        return measure_run({'kind': 'accuracy'}, self.directory)

    def history(self):
        """Errors against the analytical solution for every recorded step."""
        history = []
        for step, _ in self.output.steps:
            time = step * self.output.dt
            if time <= 0.0:
                continue
            data = self.output.read(step)
            errors = self.output.errors(data, time, npts=2000)
            history.append({
                'step': step,
                'time': time,
                'rmse': errors['rmse'],
                'max_error': errors['max_error'],
                'thickness': errors['thickness'],
            })
        return history

    def profiles(self):
        """Profiles of the requested instants, in the layout used by the figures."""
        times = self.times if self.times is not None else self._default_times()

        panels = []
        for target_time in times:
            step = self.output.nearest_step(target_time)
            data = self.output.read(step)
            time = step * self.output.dt
            _, _, values = self.output.analytical(time, npts=2000)
            panels.append({
                'requested_time': target_time,
                'time': time,
                'step': step,
                'simulation': data,
                'analytical': values,
            })

        return {'nx': self.output.nx, 'panels': panels}

    def _default_times(self):
        """Spreads the requested instants over the recorded interval."""
        last = self.output.steps[-1][0] * self.output.dt
        if self.panels == 1:
            return [last]
        return list(np.linspace(0.0, last, self.panels))

    def build(self):
        """Writes the figures and the report; returns the paths written."""
        metrics = self.metrics()
        history = self.history()

        written = {
            'profiles': self.figures.profiles(self.profiles(), 'run-profiles.png'),
            'errors': self.figures.error_history(history, 'run-errors.png'),
            'thickness': self.figures.thickness_history(history, 'run-thickness.png'),
        }

        payload = {'directory': str(self.directory), 'output': str(self.output.out_dir),
                   'config': self.output.config, 'metrics': metrics, 'history': history,
                   'figures': {name: (str(path) if path else None)
                               for name, path in written.items()}}
        written['json'] = self._json(payload)
        written['markdown'] = self._markdown(metrics, history)
        return written

    def _json(self, payload):
        path = self.figures.directory / 'run-report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        print(f"Report written to {path}")
        return path

    def _markdown(self, metrics, history):
        path = self.figures.directory / 'run-report.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(self._lines(metrics, history)) + '\n')
        print(f"Report written to {path}")
        return path

    def _lines(self, metrics, history):
        config = self.output.config
        fluid = config.get('Fluid', {}) or {}
        performance = config.get('Performance', {}) or {}

        lines = [f"# Report of {self.output.out_dir}", '', '## Configuration', '',
                 '| quantity | value |', '|---|---|',
                 f"| nx | {self.output.nx} |",
                 f"| ny | {self.output.ny} |",
                 f"| pressure ratio | {float(fluid.get('pressureL', 0.0)) / float(fluid.get('pressureR', 1.0)):g} |",
                 f"| viscosity | {float(fluid.get('viscosity', 0.0)):.3e} |",
                 f"| gamma | {float(fluid.get('gamma', 0.0)):g} |",
                 f"| tmax | {self.output.tmax:g} |",
                 f"| backend | {performance.get('backend', '---')} |",
                 f"| target | {performance.get('target', '---')} |",
                 f"| cores | {performance.get('cores', '---')} |", '',
                 '## Execution', '', '| quantity | value |', '|---|---|',
                 f"| runtime (s) | {metrics['runtime']:.6f} |",
                 f"| expected steps | {metrics['steps']} |",
                 f"| last recorded step | {metrics['last_step']} |",
                 f"| physical time reached (s) | {metrics['time_reached']:.6f} |",
                 f"| fraction of tmax | {100 * metrics['time_fraction']:.1f}% |",
                 f"| MLUPS | " + (f"{metrics['mlups']:.1f} |" if metrics['mlups'] else '--- |'),
                 f"| diverged | {'yes' if metrics['diverged'] else 'no'} |", '']

        if history:
            lines += ['## Error against the analytical solution', '',
                      '| step | t (s) | RMSE(rho) | RMSE(u) | RMSE(p) | max(p) |'
                      ' shock (nodes) | contact (nodes) |',
                      '|---|---|---|---|---|---|---|---|']
            for entry in history:
                thickness = entry['thickness'] or {}
                shock, contact = thickness.get('shock_nodes'), thickness.get('contact_nodes')
                lines.append(
                    f"| {entry['step']} | {entry['time']:.6f} | "
                    f"{entry['rmse']['rho']:.3e} | {entry['rmse']['u']:.3e} | "
                    f"{entry['rmse']['p']:.3e} | {entry['max_error']['p']:.3e} | "
                    f"{'---' if shock is None else shock} | "
                    f"{'---' if contact is None else contact} |")
            lines.append('')

            last = history[-1]
            lines += [f"Final RMSE(p): {last['rmse']['p']:.3e} at t = {last['time']:.6f} s", '']

        return lines

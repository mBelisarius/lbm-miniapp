"""Animates the simulation against the analytical solution.

The whole output of one directory is loaded at once with Polars, normalised by
the initial left state and rendered as an animation: the center line of the
domain compared to the analytical solution and, for two-dimensional meshes, the
fields themselves.
"""
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pathlib import Path

from common import load_yaml
from simulation import STEP_PATTERN, analytical_solution, find_config, find_output_dir, simulation_dt

SCHEMA = {
    'runtime': pl.Float64,
    'x': pl.Float64, 'y': pl.Float64,
    'ux': pl.Float64, 'uy': pl.Float64, 'uz': pl.Float64,
    'density': pl.Float64, 'pressure': pl.Float64,
    'rho': pl.Float64, 'p': pl.Float64,
}


def read_all_steps(out_dir):
    """Loads every step of a directory using Polars for fast multithreaded I/O."""
    files = sorted(Path(out_dir).glob('data_*.csv'),
                   key=lambda f: int(STEP_PATTERN.match(f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {out_dir}")

    steps, runtimes, xs, ys, uxs, rhos, ps = [], [], [], [], [], [], []
    for path in files:
        frame = pl.read_csv(path, schema_overrides=SCHEMA)
        steps.append(int(STEP_PATTERN.match(path.name).group(1)))
        runtimes.append(frame['runtime'][0])
        xs.append(frame['x'].to_numpy())
        uxs.append(frame['ux'].to_numpy())
        ys.append(frame['y'].to_numpy() if 'y' in frame.columns else np.zeros_like(xs[-1]))
        rhos.append(frame['density'].to_numpy() if 'density' in frame.columns
                    else frame['rho'].to_numpy())
        ps.append(frame['pressure'].to_numpy() if 'pressure' in frame.columns
                  else frame['p'].to_numpy())

    return {
        'steps': np.array(steps),
        'runtime': np.array(runtimes),
        'x': np.array(xs),
        'y': np.array(ys),
        'ux': np.array(uxs),
        'rho': np.array(rhos),
        'p': np.array(ps),
    }


class SimulationMovie:
    """Animation of one simulation output directory."""

    def __init__(self, directory, config=None, speedup=1.0):
        self.out_dir = find_output_dir(directory)
        if self.out_dir is None:
            raise FileNotFoundError(f"No output directory inside {directory}")
        print(f"Reading from output directory: {self.out_dir}")

        config_path = config or find_config(directory, self.out_dir)
        if config_path is None:
            raise FileNotFoundError(f"No config.yaml for {directory}")
        self.config = load_yaml(config_path)

        self.speedup = float(speedup)
        self.data = read_all_steps(self.out_dir)
        self.dt = simulation_dt(self.config)
        self.x0 = float(self.config['Mesh']['lx']) / 2.0

        self._normalise()
        self._detect_shape()
        self._precompute_analytical()

    def _normalise(self):
        """Normalises the fields by the mean of the initial left state."""
        x_first = self.data['x'][0]
        left = x_first < self.x0
        if np.sum(left) < 3:
            left = np.arange(len(x_first)) < max(1, int(0.05 * len(x_first)))

        p_mean = float(np.mean(self.data['p'][0][left]))
        rho_mean = float(np.mean(self.data['rho'][0][left]))
        u_mean = float(np.mean(self.data['ux'][0][left]))
        if p_mean <= 0.0 or rho_mean <= 0.0:
            raise RuntimeError("Left-state mean pressure/density non-positive; can't normalize.")

        self.data['rho'] = self.data['rho'] / rho_mean
        self.data['p'] = self.data['p'] / p_mean
        self.data['ux'] = (self.data['ux'] - u_mean) / np.sqrt(p_mean / rho_mean)

    def _detect_shape(self):
        """Recovers (nx, ny) from the flat ordering of the output."""
        x_first = self.data['x'][0]
        ny = 1
        while ny < len(x_first) and x_first[ny] == x_first[0]:
            ny += 1
        self.ny = ny
        self.nx = len(x_first) // ny
        self.mid_y = ny // 2

    def _precompute_analytical(self):
        print("Pre-computing analytical solutions...")
        fluid = self.config['Fluid']
        self.anl_x = np.linspace(0.0, float(self.config['Mesh']['lx']), 1200)
        self.analytical = []
        for step in self.data['steps']:
            time = step * self.dt
            if time == 0.0:
                rho = np.where(self.anl_x < self.x0, float(fluid['densityL']), float(fluid['densityR']))
                p = np.where(self.anl_x < self.x0, float(fluid['pressureL']), float(fluid['pressureR']))
                u = np.zeros_like(self.anl_x)
            else:
                _, _, values = analytical_solution(self.config, time, npts=len(self.anl_x))
                rho, p, u = values['rho'], values['p'], values['u']
            self.analytical.append((time, rho, p, u))

    def _setup(self):
        """Builds the figure, the line plots and, for 2D meshes, the images."""
        label_kwargs = {'rotation': 0, 'va': 'center', 'ha': 'right'}
        x_min, x_max = float(np.min(self.data['x'][0])), float(np.max(self.data['x'][0]))
        y_min, y_max = float(np.min(self.data['y'][0])), float(np.max(self.data['y'][0]))
        y_lim = (y_min - 1.0, y_max + 1.0) if y_min == y_max else (y_min, y_max)

        if self.ny == 1:
            fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            line_axes = list(axes)
            field_axes = []
        else:
            fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')
            line_axes = [axes[row, 1] for row in range(3)]
            field_axes = [axes[row, 0] for row in range(3)]

        title = fig.suptitle('', fontsize=16, fontweight='bold')
        fields = (('rho', r"$\rho$", (0.0, 1.1)),
                  ('p', r"$p$", (0.0, 1.1)),
                  ('ux', r"$u_x$", (-0.2, 1.0)))

        lines, images = {}, {}
        for ax, (name, label, limits) in zip(line_axes, fields):
            lines[f"sim_{name}"] = ax.plot([], [], linestyle='', marker='o', markersize=3,
                                           label='Simulation')[0]
            lines[f"anl_{name}"] = ax.plot([], [], 'k--', lw=2, label='Analytical')[0]
            ax.set_xlim((x_min, x_max))
            ax.set_ylim(limits)
            ax.set_ylabel(label, labelpad=15, **label_kwargs)
            ax.legend()
        line_axes[-1].set_xlabel(r"$x$")

        for ax, (name, label, limits) in zip(field_axes, fields):
            initial = self.data[name][0].reshape((self.nx, self.ny)).T
            images[f"sim_{name}"] = ax.imshow(
                initial, origin='lower', extent=[x_min, x_max, y_lim[0], y_lim[1]],
                cmap='coolwarm', aspect='auto', vmin=limits[0], vmax=limits[1])
            ax.set_ylim(y_lim)
            ax.set_ylabel(r"$y$", labelpad=15, **label_kwargs)
            colorbar = fig.colorbar(images[f"sim_{name}"], ax=ax)
            colorbar.set_label(label, labelpad=15, **label_kwargs)
        if field_axes:
            field_axes[-1].set_xlabel(r"$x$")

        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        return fig, title, lines, images

    def render(self, filename='plot.webp', show=False):
        """Renders the animation and writes it next to the output directory."""
        fig, title, lines, images = self._setup()
        steps = self.data['steps']

        def artists():
            return tuple(lines.values()) + tuple(images.values()) + (title,)

        def init():
            for line in lines.values():
                line.set_data([], [])
            title.set_text('')
            return artists()

        def update(frame):
            time, rho_anl, p_anl, u_anl = self.analytical[frame]
            shape = (self.nx, self.ny)
            x_line = self.data['x'][frame].reshape(shape)[:, self.mid_y]

            for name, anl in (('rho', rho_anl), ('p', p_anl), ('ux', u_anl)):
                field = self.data[name][frame].reshape(shape)
                lines[f"sim_{name}"].set_data(x_line, field[:, self.mid_y])
                lines[f"anl_{name}"].set_data(self.anl_x, anl)
                if self.ny > 1:
                    images[f"sim_{name}"].set_data(field.T)

            title.set_text(f"Simulation State — Step {steps[frame]}, t = {time:.6f} s")
            return artists()

        print("Rendering animation...")
        step_time = (steps[1] - steps[0]) * self.dt if len(steps) > 1 else self.dt
        interval = step_time / self.speedup * 1000.0 if self.speedup > 0 else 0.0
        animation = manimation.FuncAnimation(fig, update, frames=len(steps), init_func=init,
                                            blit=False, interval=interval)

        path = self.out_dir / filename
        animation.save(path, writer='pillow')
        print(f"Plot saved to {path}")
        if show:
            plt.show()
        plt.close(fig)
        return path

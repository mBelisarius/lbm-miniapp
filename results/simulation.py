"""Reads the output of a single simulation and measures it.

Everything that depends on the layout written by the executable lives here: the
discovery of the output directory, the CSV columns, the time step reproduced
from the configuration, the comparison against the analytical solution and the
metrics derived from a finished run.
"""
import numpy as np
import re

from pathlib import Path

from analytical import solve
from common import CS2, load_yaml


STEP_PATTERN = re.compile(r"data_(\d+)\.csv")

COLUMN_ALIASES = {
    'runtime': ('runtime',),
    'x': ('x',),
    'y': ('y',),
    'ux': ('ux', 'u'),
    'rho': ('density', 'rho'),
    'p': ('pressure', 'p'),
}

# Layout written by Executor.cpp: runtime,x,y,ux,uy,density,pressure.
FALLBACK_COLUMNS = {'runtime': 0, 'x': 1, 'y': 2, 'ux': 3, 'rho': 5, 'p': 6}

REQUIRED_COLUMNS = ('runtime', 'x', 'ux', 'rho', 'p')


def simulation_dt(config):
    """Reproduces the time step used by Executor::Run."""
    fluid, mesh = config['Fluid'], config['Mesh']
    dx = float(mesh['lx']) / float(mesh['nx'])
    u_phys_ref = np.sqrt(float(fluid['gamma']) * float(fluid['pressureL']) / float(fluid['densityL']))
    u_lu_ref = np.sqrt(float(fluid['gamma']) * CS2)
    return dx * (u_lu_ref / u_phys_ref)


def total_steps(config):
    return int(float(config['Control']['tmax']) / simulation_dt(config))


def find_output_dir(directory):
    """Returns the directory actually written by the executable."""
    directory = Path(directory)
    if any(directory.glob('data_*.csv')):
        return directory

    candidates = sorted([p for p in directory.glob('out*') if p.is_dir()])
    for candidate in candidates:
        if any(candidate.glob('data_*.csv')):
            return candidate
    return candidates[0] if candidates else None


def find_config(directory, out_dir=None):
    """Returns the configuration file of a run, wherever it was written."""
    directory = Path(directory)
    candidates = [directory / 'config.yaml']
    if out_dir is not None:
        candidates.append(Path(out_dir) / 'config.yaml')
    candidates.append(directory.parent / 'config.yaml')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def list_step_files(out_dir):
    """Returns the (step, path) pairs of a directory, sorted by step."""
    files = []
    for path in Path(out_dir).glob('data_*.csv'):
        match = STEP_PATTERN.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def column_indices(csv_path, columns):
    """Maps the required fields to the columns of the CSV header."""
    with open(csv_path, 'r') as f:
        names = [name.strip().lower() for name in f.readline().strip().split(',')]

    index = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in names:
                index[field] = names.index(alias)
                break

    if any(field not in index for field in REQUIRED_COLUMNS):
        index = {field: position for field, position in FALLBACK_COLUMNS.items()
                 if position < columns}
    return index


def read_center_line(csv_path, ny=1):
    """Reads a CSV output and returns the center line of the domain."""
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"Empty output file: {csv_path}")

    index = column_indices(csv_path, data.shape[1])
    if any(field not in index for field in REQUIRED_COLUMNS):
        raise ValueError(f"Unexpected column layout in {csv_path}")

    runtime = float(data[0, index['runtime']])
    if ny > 1:
        data = data[ny // 2::ny]

    return {
        'runtime': runtime,
        'x': data[:, index['x']],
        'ux': data[:, index['ux']],
        'rho': data[:, index['rho']],
        'p': data[:, index['p']],
    }


def analytical_solution(config, time, npts=100000):
    """Evaluates the analytical Sod solution of a configuration at an instant."""
    fluid, mesh = config['Fluid'], config['Mesh']
    left = (float(fluid['pressureL']), float(fluid['densityL']), 0.0)
    right = (float(fluid['pressureR']), float(fluid['densityR']), 0.0)
    geometry = (0.0, float(mesh['lx']), float(mesh['lx']) / 2.0)
    return solve(left, right, geometry, time, gamma=float(fluid['gamma']), npts=npts)


def rmse(simulated, reference):
    return float(np.sqrt(np.mean((simulated - reference) ** 2)))


def max_error(simulated, reference):
    return float(np.max(np.abs(simulated - reference)))


def transition_nodes(x, values, center, left_value, right_value, half_window):
    """Counts the nodes spanning 10% to 90% of a jump around a wave front."""
    if not np.isfinite(half_window) or half_window <= 0.0:
        return None
    jump = right_value - left_value
    if abs(jump) < 1e-12:
        return None
    low = min(left_value, right_value) + 0.1 * abs(jump)
    high = min(left_value, right_value) + 0.9 * abs(jump)
    window = np.abs(x - center) <= half_window
    inside = window & (values > low) & (values < high)
    return int(np.count_nonzero(inside))


def measure_front_thickness(x, rho, positions, regions):
    """Measures the thickness, in nodes, of the shock and of the contact."""
    region3, region4, region5 = regions.get('Region 3'), regions.get('Region 4'), regions.get('Region 5')
    if not all(isinstance(r, tuple) for r in (region3, region4, region5)):
        return {}

    x_foot = positions['Foot of Rarefaction']
    x_contact = positions['Contact Discontinuity']
    x_shock = positions['Shock']

    contact_window = 0.5 * min(abs(x_contact - x_foot), abs(x_shock - x_contact))
    shock_window = 0.5 * abs(x_shock - x_contact)

    return {
        'contact_nodes': transition_nodes(x, rho, x_contact, region3[1], region4[1], contact_window),
        'shock_nodes': transition_nodes(x, rho, x_shock, region4[1], region5[1], shock_window),
    }


class SimulationOutput:
    """Output of one simulation, ready to be read and compared."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.out_dir = find_output_dir(self.directory)
        if self.out_dir is None:
            raise FileNotFoundError(f"No output directory inside {self.directory}")

        config_path = find_config(self.directory, self.out_dir)
        if config_path is None:
            raise FileNotFoundError(f"No config.yaml for {self.directory}")
        self.config = load_yaml(config_path)

        self.steps = list_step_files(self.out_dir)
        if not self.steps:
            raise FileNotFoundError(f"No data_*.csv files inside {self.out_dir}")

        self.dt = simulation_dt(self.config)
        self.expected_steps = total_steps(self.config)
        self.nx = int(self.config['Mesh']['nx'])
        self.ny = int(self.config['Mesh'].get('ny', 1) or 1)
        self.tmax = float(self.config['Control']['tmax'])

    def read(self, step):
        """Reads the center line of a given step."""
        path = dict(self.steps)[step]
        return read_center_line(path, self.ny)

    def last(self):
        """Returns the last recorded step and its center line."""
        step = self.steps[-1][0]
        return step, self.read(step)

    def nearest_step(self, time):
        """Returns the recorded step closest to a physical instant."""
        return min(self.steps, key=lambda item: abs(item[0] * self.dt - time))[0]

    def analytical(self, time, npts=100000):
        return analytical_solution(self.config, time, npts=npts)

    def errors(self, data, time, npts=100000):
        """Compares one center line against the analytical solution."""
        positions, regions, values = self.analytical(time, npts=npts)
        reference = {
            'rho': np.interp(data['x'], values['x'], values['rho']),
            'u': np.interp(data['x'], values['x'], values['u']),
            'p': np.interp(data['x'], values['x'], values['p']),
        }
        simulated = {'rho': data['rho'], 'u': data['ux'], 'p': data['p']}
        return {
            'rmse': {name: rmse(simulated[name], reference[name]) for name in simulated},
            'max_error': {name: max_error(simulated[name], reference[name]) for name in simulated},
            'thickness': measure_front_thickness(data['x'], data['rho'], positions, regions),
        }


def measure_run(run, run_dir):
    """Extracts every metric of a finished run from its output directory."""
    output = SimulationOutput(run_dir)
    last_step, last = output.last()

    finite = bool(np.all(np.isfinite(last['rho'])) and np.all(np.isfinite(last['p']))
                  and np.all(np.isfinite(last['ux'])))
    time_reached = last_step * output.dt
    time_fraction = time_reached / output.tmax
    complete = finite and last_step >= output.expected_steps

    cells = output.nx * output.ny
    runtime = last['runtime']
    mlups = (cells * output.expected_steps / runtime / 1.0e6) if runtime > 0.0 else None

    metrics = {
        'runtime': runtime,
        'steps': output.expected_steps,
        'last_step': last_step,
        'time_reached': time_reached,
        'time_fraction': time_fraction,
        'mlups': mlups,
        'cells': cells,
        'diverged': not complete,
    }

    if run.get('kind') == 'accuracy' and complete:
        metrics.update(output.errors(last, time_reached))

    return metrics

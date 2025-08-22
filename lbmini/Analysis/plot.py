#!/usr/bin/env python3
import matplotlib

matplotlib.use('TkAgg')

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re
import yaml

from pathlib import Path

from analytical import solve


def load_config(config_path):
    """Loads simulation parameters from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    fluid_config = config['Fluid']
    mesh_config = config['Mesh']

    cs2 = 1.0 / 3.0
    dx = mesh_config['lx'] / mesh_config['nx']
    u_phys_ref = np.sqrt(fluid_config['gamma'] * fluid_config['pressureL'] / fluid_config['densityL'])
    u_lu_ref = np.sqrt(fluid_config['gamma'] * cs2)
    sim_dt = dx * (u_lu_ref / u_phys_ref)

    params = {
        'sim_dt': sim_dt,
        'P_L': fluid_config['pressureL'], 'rho_L': fluid_config['densityL'], 'u_L': 0.0,
        'P_R': fluid_config['pressureR'], 'rho_R': fluid_config['densityR'], 'u_R': 0.0,
        'gamma': fluid_config['gamma'],
        'xl': 0.0, 'xr': mesh_config['lx'], 'x0': mesh_config['lx'] / 2.0
    }
    return params


def load_simulation_data(sim_path):
    """Loads all simulation data."""
    pattern = re.compile(r"data_(\d+)\.csv")
    files = sorted(sim_path.glob("data_*.csv"), key=lambda f: int(pattern.match(f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {sim_path}")

    first_data = np.genfromtxt(files[0], delimiter=',', skip_header=1)
    num_cols = first_data.shape[1]
    all_index = {'runtime': 0, 'x': 1, 'ux': 2, 'rho': num_cols - 2, 'p': num_cols - 1}
    if num_cols >= 6: all_index['uy'] = 3
    if num_cols >= 7: all_index['uz'] = 4

    data_arrays = {'steps': [], 'runtime': [], 'x': [], 'ux': [], 'rho': [], 'p': []}
    for f in files:
        match = pattern.match(f.name)
        step = int(match.group(1))
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        data_arrays['steps'].append(step)
        data_arrays['runtime'].append(data[0, all_index['runtime']])
        data_arrays['x'].append(data[:, all_index['x']])
        data_arrays['ux'].append(data[:, all_index['ux']])
        data_arrays['rho'].append(data[:, all_index['rho']])
        data_arrays['p'].append(data[:, all_index['p']])

    return data_arrays


def calculate_normalization_factors(all_p, all_rho, all_ux, all_x, x0):
    """Computes normalization factors from the simulation's initial left state."""
    x_first = all_x[0]
    left_mask = x_first < x0
    if np.sum(left_mask) < 3:
        left_mask = np.arange(len(x_first)) < max(1, int(0.05 * len(x_first)))

    p_sim_left_mean = np.mean(all_p[0][left_mask])
    rho_sim_left_mean = np.mean(all_rho[0][left_mask])
    u_sim_left_mean = np.mean(all_ux[0][left_mask])

    if p_sim_left_mean <= 0 or rho_sim_left_mean <= 0:
        raise RuntimeError("Left-state mean pressure/density non-positive; can't normalize reliably.")

    vel_scale_sim = np.sqrt(p_sim_left_mean / rho_sim_left_mean)

    return rho_sim_left_mean, p_sim_left_mean, u_sim_left_mean, vel_scale_sim


def setup_plot(x_lim):
    """Sets up the matplotlib figure and axes for the animation."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    lines = {
        'sim_rho': (ax1.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation"))[0],
        'anl_rho': (ax1.plot([], [], 'k--', lw=2, label="Analytical"))[0],
        'sim_p': (ax2.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation"))[0],
        'anl_p': (ax2.plot([], [], 'k--', lw=2, label="Analytical"))[0],
        'sim_u': (ax3.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation"))[0],
        'anl_u': (ax3.plot([], [], 'k--', lw=2, label="Analytical"))[0],
    }

    ax1.set_xlim(x_lim)
    ax1.set_ylim(0.0, 1.1)
    ax1.set_ylabel(r"$\rho$")
    ax1.legend()
    title = ax1.set_title("")

    ax2.set_ylim(0.0, 1.1)
    ax2.set_ylabel(r"$p$")
    ax2.legend()

    ax3.set_ylim(-0.2, 1.0)
    ax3.set_ylabel(r"$u_x$")
    ax3.set_xlabel(r"$x$")
    ax3.legend()

    return fig, title, lines


def main(speedup, config, outpath):
    # Finds the latest output
    output_path = Path(outpath)
    out_dirs = [p for p in output_path.glob("out*") if p.is_dir()]
    if not out_dirs:
        raise FileNotFoundError(f"No 'out*' directories found in {output_path}")

    out_dir = max(out_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Reading from latest output directory: {out_dir}")

    # Load configurations and data
    params = load_config(config)
    sim_data = load_simulation_data(out_dir)
    norm_factors = calculate_normalization_factors(sim_data['p'], sim_data['rho'], sim_data['ux'], sim_data['x'], params['x0'])
    rho_sim_left_mean, p_sim_left_mean, u_sim_left_mean, vel_scale_sim = norm_factors

    # Setup plotting
    anl_x = np.linspace(params['xl'], params['xr'], 1200)
    fig, title, lines = setup_plot(x_lim=(min(sim_data['x'][0]), max(sim_data['x'][0])))

    # Compute animation interval
    steps = sim_data['steps']
    dt_sim_per_frame = (steps[1] - steps[0]) * params['sim_dt'] if len(steps) > 1 else params['sim_dt']
    interval_ms = dt_sim_per_frame / speedup * 1000.0 if speedup > 0 else 0.0

    def init():
        for line in lines.values():
            line.set_data([], [])
        title.set_text("")
        return tuple(lines.values()) + (title,)

    def update(frame):
        sim_time = sim_data['steps'][frame] * params['sim_dt']

        x_sim = sim_data['x'][frame]
        rho_sim_dimless = sim_data['rho'][frame] / rho_sim_left_mean
        p_sim_dimless = sim_data['p'][frame] / p_sim_left_mean
        u_sim_dimless = (sim_data['ux'][frame] - u_sim_left_mean) / vel_scale_sim

        lines['sim_rho'].set_data(x_sim, rho_sim_dimless)
        lines['sim_p'].set_data(x_sim, p_sim_dimless)
        lines['sim_u'].set_data(x_sim, u_sim_dimless)

        left_state = (params['P_L'], params['rho_L'], params['u_L'])
        right_state = (params['P_R'], params['rho_R'], params['u_R'])
        geo = (params['xl'], params['xr'], params['x0'])

        if sim_time == 0.0:
            rho_an = np.where(anl_x < params['x0'], params['rho_L'], params['rho_R'])
            p_an = np.where(anl_x < params['x0'], params['P_L'], params['P_R'])
            u_an = np.zeros_like(anl_x)
        else:
            _, _, vals = solve(left_state, right_state, geo, sim_time, gamma=params['gamma'], npts=len(anl_x))
            rho_an, p_an, u_an = vals['rho'], vals['p'], vals['u']

        lines['anl_rho'].set_data(anl_x, rho_an)
        lines['anl_p'].set_data(anl_x, p_an)
        lines['anl_u'].set_data(anl_x, u_an)

        title.set_text(rf"Profiles — $t$ = {sim_time:.6f} s (Step {sim_data['steps'][frame]})")
        return tuple(lines.values()) + (title,)

    ani = animation.FuncAnimation(fig, update, frames=len(sim_data['steps']), init_func=init, blit=True, interval=interval_ms)

    ani.save(out_dir / "plot.webp", writer="pillow")
    print(f"Plot saved to {out_dir / 'plot.webp'}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Sod shock-tube analytical vs simulation results.")
    parser.add_argument('--speedup', type=float, default=1.0, help='Playback speedup factor. >1 for faster, <1 for slower.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config.yaml file.')
    parser.add_argument('--outpath', type=str, default='out', help='Path to the simulation output directory.')
    args = parser.parse_args()
    main(args.speedup, args.config, args.outpath)

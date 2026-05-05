#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import polars as pl
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

    return {
        'sim_dt': sim_dt,
        'P_L': fluid_config['pressureL'], 'rho_L': fluid_config['densityL'], 'u_L': 0.0,
        'P_R': fluid_config['pressureR'], 'rho_R': fluid_config['densityR'], 'u_R': 0.0,
        'gamma': fluid_config['gamma'],
        'xl': 0.0, 'xr': mesh_config['lx'], 'x0': mesh_config['lx'] / 2.0
    }


def load_simulation_data_fast(sim_path):
    """Loads simulation data using Polars for ultra-fast multithreaded I/O."""
    pattern = re.compile(r"data_(\d+)\.csv")
    files = sorted(sim_path.glob("data_*.csv"), key=lambda f: int(pattern.match(f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {sim_path}")

    steps, runtimes, xs, ys, uxs, rhos, ps = [], [], [], [], [], [], []

    for f in files:
        step = int(pattern.match(f.name).group(1))
        steps.append(step)

        df = pl.read_csv(f, schema_overrides={
            "runtime": pl.Float64,
            "x": pl.Float64, "y": pl.Float64,
            "ux": pl.Float64, "uy": pl.Float64, "uz": pl.Float64,
            "density": pl.Float64, "pressure": pl.Float64,
            "rho": pl.Float64, "p": pl.Float64
        })

        runtimes.append(df['runtime'][0])
        xs.append(df['x'].to_numpy())
        uxs.append(df['ux'].to_numpy())
        ys.append(df['y'].to_numpy() if 'y' in df.columns else np.zeros_like(xs[-1]))
        rhos.append(df['density'].to_numpy() if 'density' in df.columns else df['rho'].to_numpy())
        ps.append(df['pressure'].to_numpy() if 'pressure' in df.columns else df['p'].to_numpy())

    return {
        'steps': np.array(steps),
        'runtime': np.array(runtimes),
        'x': np.array(xs),
        'y': np.array(ys),
        'ux': np.array(uxs),
        'rho': np.array(rhos),
        'p': np.array(ps),
    }


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


def setup_plot(sim_data, nx, ny, extent, x_lim, y_lim):
    """Sets up the matplotlib figure and axes."""
    is_1d = (ny == 1)

    if is_1d:
        # For 1D simulations, omit the left column entirely
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        ax_rho_1d, ax_p_1d, ax_u_1d = axs[0], axs[1], axs[2]
    else:
        # Standard 2D layout
        fig, axs = plt.subplots(3, 2, figsize=(14, 8), sharex='col')
        ax_rho_2d, ax_rho_1d = axs[0, 0], axs[0, 1]
        ax_p_2d, ax_p_1d     = axs[1, 0], axs[1, 1]
        ax_u_2d, ax_u_1d     = axs[2, 0], axs[2, 1]

    title = fig.suptitle("", fontsize=16, fontweight='bold')
    label_kwargs = {'rotation': 0, 'va': 'center', 'ha': 'right'}

    # 1D Line Plots (Using only the center Y-slice)
    lines = {
        'sim_rho': ax_rho_1d.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation")[0],
        'anl_rho': ax_rho_1d.plot([], [], 'k--', lw=2, label="Analytical")[0],
        'sim_p': ax_p_1d.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation")[0],
        'anl_p': ax_p_1d.plot([], [], 'k--', lw=2, label="Analytical")[0],
        'sim_u': ax_u_1d.plot([], [], linestyle='', marker='o', markersize=3, label="Simulation")[0],
        'anl_u': ax_u_1d.plot([], [], 'k--', lw=2, label="Analytical")[0],
    }

    ax_rho_1d.set_xlim(x_lim)
    ax_rho_1d.set_ylim(0.0, 1.1)
    ax_rho_1d.set_ylabel(r"$\rho$", labelpad=15, **label_kwargs)
    ax_rho_1d.legend()

    ax_p_1d.set_ylim(0.0, 1.1)
    ax_p_1d.set_ylabel(r"$p$", labelpad=15, **label_kwargs)
    ax_p_1d.legend()

    ax_u_1d.set_ylim(-0.2, 1.0)
    ax_u_1d.set_ylabel(r"$u_x$", labelpad=15, **label_kwargs)
    ax_u_1d.set_xlabel(r"$x$")
    ax_u_1d.legend()

    images = {}

    # Left column: 2D Image Plots (Extremely Fast) ONLY if ny > 1
    if not is_1d:
        rho_init = sim_data['rho'][0].reshape((nx, ny)).T
        p_init   = sim_data['p'][0].reshape((nx, ny)).T
        u_init   = sim_data['ux'][0].reshape((nx, ny)).T

        img_kwargs = {'origin': 'lower', 'extent': extent, 'cmap': 'coolwarm', 'aspect': 'auto'}

        images = {
            'sim_rho': ax_rho_2d.imshow(rho_init, vmin=0.0, vmax=1.1, **img_kwargs),
            'sim_p': ax_p_2d.imshow(p_init, vmin=0.0, vmax=1.1, **img_kwargs),
            'sim_u': ax_u_2d.imshow(u_init, vmin=-0.2, vmax=1.0, **img_kwargs)
        }

        for ax in [ax_rho_2d, ax_p_2d, ax_u_2d]:
            ax.set_ylim(y_lim)
            ax.set_ylabel(r"$y$", labelpad=15, **label_kwargs)
        ax_u_2d.set_xlabel(r"$x$")

        cb_rho = fig.colorbar(images['sim_rho'], ax=ax_rho_2d)
        cb_rho.set_label(r"$\rho$", labelpad=15, **label_kwargs)

        cb_p = fig.colorbar(images['sim_p'], ax=ax_p_2d)
        cb_p.set_label(r"$p$", labelpad=15, **label_kwargs)

        cb_u = fig.colorbar(images['sim_u'], ax=ax_u_2d)
        cb_u.set_label(r"$u_x$", labelpad=15, **label_kwargs)

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    return fig, title, lines, images


def main(speedup, config, outpath):
    output_path = Path(outpath)
    out_dirs = [p for p in output_path.glob("out*") if p.is_dir()]
    if not out_dirs:
        raise FileNotFoundError(f"No 'out*' directories found in {output_path}")

    out_dir = max(out_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Reading from latest output directory: {out_dir}")

    params = load_config(config)
    sim_data = load_simulation_data_fast(out_dir)

    rho_l_mean, p_l_mean, u_l_mean, vel_scale = calculate_normalization_factors(
        sim_data['p'], sim_data['rho'], sim_data['ux'], sim_data['x'], params['x0']
    )

    sim_data['rho'] = sim_data['rho'] / rho_l_mean
    sim_data['p']   = sim_data['p'] / p_l_mean
    sim_data['ux']  = (sim_data['ux'] - u_l_mean) / vel_scale

    # Dynamically detect grid dimensions (nx, ny) from the flat array order
    x_first = sim_data['x'][0]
    ny = 1
    while ny < len(x_first) and x_first[ny] == x_first[0]:
        ny += 1
    nx = len(x_first) // ny
    mid_y = ny // 2  # The index to slice for the 1D plots

    x_min, x_max = np.min(sim_data['x'][0]), np.max(sim_data['x'][0])
    y_min, y_max = np.min(sim_data['y'][0]), np.max(sim_data['y'][0])
    y_lim = (y_min - 1.0, y_max + 1.0) if y_min == y_max else (y_min, y_max)
    extent = [x_min, x_max, y_lim[0], y_lim[1]]

    fig, title, lines, images = setup_plot(sim_data, nx, ny, extent, x_lim=(x_min, x_max), y_lim=y_lim)

    print("Pre-computing analytical solutions...")
    anl_x = np.linspace(params['xl'], params['xr'], 1200)
    left_state = (params['P_L'], params['rho_L'], params['u_L'])
    right_state = (params['P_R'], params['rho_R'], params['u_R'])
    geo = (params['xl'], params['xr'], params['x0'])

    analytical_cache = []
    for step in sim_data['steps']:
        sim_time = step * params['sim_dt']
        if sim_time == 0.0:
            rho_an = np.where(anl_x < params['x0'], params['rho_L'], params['rho_R'])
            p_an = np.where(anl_x < params['x0'], params['P_L'], params['P_R'])
            u_an = np.zeros_like(anl_x)
        else:
            _, _, vals = solve(left_state, right_state, geo, sim_time, gamma=params['gamma'], npts=len(anl_x))
            rho_an, p_an, u_an = vals['rho'], vals['p'], vals['u']
        analytical_cache.append((sim_time, rho_an, p_an, u_an))

    dt_sim_per_frame = (sim_data['steps'][1] - sim_data['steps'][0]) * params['sim_dt'] if len(sim_data['steps']) > 1 else params['sim_dt']
    interval_ms = dt_sim_per_frame / speedup * 1000.0 if speedup > 0 else 0.0

    def init():
        for line in lines.values():
            line.set_data([], [])

        title.set_text("")
        return tuple(lines.values()) + tuple(images.values()) + (title,)

    def update(frame):
        sim_time, rho_an, p_an, u_an = analytical_cache[frame]

        # Reshape current frame data to 2D
        rho_2d = sim_data['rho'][frame].reshape((nx, ny))
        p_2d   = sim_data['p'][frame].reshape((nx, ny))
        u_2d   = sim_data['ux'][frame].reshape((nx, ny))

        # Extract the 1D center slice
        x_1d = sim_data['x'][frame].reshape((nx, ny))[:, mid_y]

        # Update 1D line plots (Fast: rendering ~nx points)
        lines['sim_rho'].set_data(x_1d, rho_2d[:, mid_y])
        lines['sim_p'].set_data(x_1d, p_2d[:, mid_y])
        lines['sim_u'].set_data(x_1d, u_2d[:, mid_y])

        lines['anl_rho'].set_data(anl_x, rho_an)
        lines['anl_p'].set_data(anl_x, p_an)
        lines['anl_u'].set_data(anl_x, u_an)

        if ny > 1:
            # Update 2D images only if grid is 2D
            images['sim_rho'].set_data(rho_2d.T)
            images['sim_p'].set_data(p_2d.T)
            images['sim_u'].set_data(u_2d.T)

        title.set_text(f"Simulation State — Step {sim_data['steps'][frame]}, t = {sim_time:.6f} s")
        return tuple(lines.values()) + tuple(images.values()) + (title,)

    print("Rendering animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(sim_data['steps']), init_func=init, blit=False, interval=interval_ms)

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

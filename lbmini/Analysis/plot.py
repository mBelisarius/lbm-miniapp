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
from scipy import optimize


def sound_speed(gamma, pressure, density, dust_frac=0.0):
    scale = np.sqrt(1.0 - dust_frac)
    return np.sqrt(gamma * pressure / density) * scale


def shock_tube_function(p4, p1, p5, rho1, rho5, gamma, dust_frac=0.0):
    z = (p4 / p5 - 1.0)
    c1 = sound_speed(gamma, p1, rho1, dust_frac)
    c5 = sound_speed(gamma, p5, rho5, dust_frac)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    g2 = 2.0 * gamma

    fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1.0 + gp1 / g2 * z)
    fact = (1.0 - fact) ** (g2 / gm1)

    return p1 * fact - p4


def calculate_regions(pl, rhol, ul, pr, rhor, ur, gamma=1.4, dust_frac=0.0):
    rho1 = rhol
    p1 = pl
    u1 = ul
    rho5 = rhor
    p5 = pr
    u5 = ur

    if pl < pr:
        rho1 = rhor
        p1 = pr
        u1 = ur
        rho5 = rhol
        p5 = pl
        u5 = ul

    try:
        p4 = optimize.fsolve(
            shock_tube_function, p1,
            (p1, p5, rho1, rho5, gamma, dust_frac),
        )[0]
    except Exception:
        a = min(p1, p5) * 1e-8
        b = max(p1, p5) * 1e6

        def ftarget(p):
            return shock_tube_function(p, p1, p5, rho1, rho5, gamma, dust_frac)

        p4 = optimize.brentq(ftarget, a, b)

    z = (p4 / p5 - 1.0)
    c5 = sound_speed(gamma, p5, rho5, dust_frac)

    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gmfac1 = 0.5 * gm1 / gamma
    gmfac2 = 0.5 * gp1 / gamma

    fact = np.sqrt(1.0 + gmfac2 * z)

    u4 = c5 * z / (gamma * fact)
    rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z)

    w = c5 * fact

    p3 = p4
    u3 = u4
    rho3 = rho1 * (p3 / p1) ** (1.0 / gamma)

    return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w


def calc_positions(pl, pr, region1, region3, w, xi, t, gamma, dust_frac=0.0):
    p1, rho1 = region1[:2]
    p3, rho3, u3 = region3
    c1 = sound_speed(gamma, p1, rho1, dust_frac)
    c3 = sound_speed(gamma, p3, rho3, dust_frac)

    if pl > pr:
        xsh = xi + w * t
        xcd = xi + u3 * t
        xft = xi + (u3 - c3) * t
        xhd = xi - c1 * t
    else:
        xsh = xi - w * t
        xcd = xi - u3 * t
        xft = xi - (u3 - c3) * t
        xhd = xi + c1 * t

    return xhd, xft, xcd, xsh


def region_states(pl, pr, region1, region3, region4, region5):
    if pl > pr:
        return {'Region 1': region1,
                'Region 2': 'RAREFACTION',
                'Region 3': region3,
                'Region 4': region4,
                'Region 5': region5}
    else:
        return {'Region 1': region5,
                'Region 2': region4,
                'Region 3': region3,
                'Region 4': 'RAREFACTION',
                'Region 5': region1}


def create_arrays(pl, pr, xl, xr, positions, state1, state3, state4, state5, npts, gamma, t, xi, dust_frac=0.0):
    xhd, xft, xcd, xsh = positions
    p1, rho1, u1 = state1
    p3, rho3, u3 = state3
    p4, rho4, u4 = state4
    p5, rho5, u5 = state5
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    x_arr = np.linspace(xl, xr, npts)
    rho = np.zeros(npts, dtype=float)
    p = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    c1 = sound_speed(gamma, p1, rho1, dust_frac)

    if t == 0.0:
        for i, x in enumerate(x_arr):
            if x < xi:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
        return x_arr, p, rho, u

    if pl > pr:
        for i, x in enumerate(x_arr):
            if x < xhd:
                rho[i] = rho1
                p[i] = p1
                u[i] = u1
            elif x < xft:
                u_i = 2.0 / gp1 * (c1 + (x - xi) / t)
                fact = 1.0 - 0.5 * gm1 * u_i / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
                u[i] = u_i
            elif x < xcd:
                rho[i] = rho3
                p[i] = p3
                u[i] = u3
            elif x < xsh:
                rho[i] = rho4
                p[i] = p4
                u[i] = u4
            else:
                rho[i] = rho5
                p[i] = p5
                u[i] = u5
    else:
        for i, x in enumerate(x_arr):
            if x < xsh:
                rho[i] = rho5
                p[i] = p5
                u[i] = -u1
            elif x < xcd:
                rho[i] = rho4
                p[i] = p4
                u[i] = -u4
            elif x < xft:
                rho[i] = rho3
                p[i] = p3
                u[i] = -u3
            elif x < xhd:
                u_i = -2.0 / gp1 * (c1 + (xi - x) / t)
                fact = 1.0 + 0.5 * gm1 * u_i / c1
                rho[i] = rho1 * fact ** (2.0 / gm1)
                p[i] = p1 * fact ** (2.0 * gamma / gm1)
                u[i] = u_i
            else:
                rho[i] = rho1
                p[i] = p1
                u[i] = -u1

    return x_arr, p, rho, u


def solve(left_state, right_state, geometry, t, gamma=1.4, npts=500, dust_frac=0.0):
    pl, rhol, ul = left_state
    pr, rhor, ur = right_state
    xl, xr, xi = geometry

    if xl >= xr:
        raise ValueError("xl must be < xr")
    if not (xl < xi < xr):
        raise ValueError("xi must be between xl and xr")

    region1, region3, region4, region5, w = calculate_regions(pl, rhol, ul, pr, rhor, ur, gamma, dust_frac)
    regions = region_states(pl, pr, region1, region3, region4, region5)

    x_positions = calc_positions(pl, pr, region1, region3, w, xi, t, gamma, dust_frac)
    pos_description = ('Head of Rarefaction', 'Foot of Rarefaction', 'Contact Discontinuity', 'Shock')
    positions = dict(zip(pos_description, x_positions))

    x, p, rho, u = create_arrays(pl, pr, xl, xr, x_positions, region1, region3, region4, region5, npts, gamma, t, xi, dust_frac)
    energy = p / (rho * (gamma - 1.0))
    rho_total = rho / (1.0 - dust_frac)

    val_dict = {'x': x, 'p': p, 'rho': rho, 'u': u, 'energy': energy, 'rho_total': rho_total}
    return positions, regions, val_dict


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


def load_simulation_data(build_dir_path):
    """Finds the latest output and loads all simulation data."""
    build_dir = Path(build_dir_path)
    out_dirs = [p for p in build_dir.glob("out*") if p.is_dir()]
    if not out_dirs:
        raise FileNotFoundError(f"No 'out*' directories found in {build_dir}")

    out_dir = max(out_dirs, key=lambda p: p.stat().st_mtime)
    print(f"Reading from latest output directory: {out_dir}")

    pattern = re.compile(r"data_(\d+)\.csv")
    files = sorted(out_dir.glob("data_*.csv"), key=lambda f: int(pattern.match(f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {out_dir}")

    first_data = np.genfromtxt(files[0], delimiter=',', skip_header=1)
    num_cols = first_data.shape[1]
    all_index = {'x': 0, 'ux': 1, 'rho': num_cols - 2, 'p': num_cols - 1}
    if num_cols >= 5: all_index['uy'] = 4
    if num_cols >= 6: all_index['uz'] = 5

    data_arrays = {'steps': [], 'x': [], 'rho': [], 'p': [], 'ux': []}
    for f in files:
        match = pattern.match(f.name)
        step = int(match.group(1))
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        data_arrays['steps'].append(step)
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
    ax1.set_ylabel(r"$\rho / \rho_L$")
    ax1.legend()
    title = ax1.set_title("")

    ax2.set_ylim(0.0, 1.1)
    ax2.set_ylabel(r"$p / p_L$")
    ax2.legend()

    ax3.set_ylim(-0.2, 1.0)
    ax3.set_ylabel(r"$\frac{u}{\sqrt{p_L/\rho_L}}$")
    ax3.set_xlabel(r"$x$")
    ax3.legend()

    return fig, title, lines


def main(speedup, config_path, output_path):
    # Load configurations and data
    params = load_config(config_path)
    sim_data = load_simulation_data(output_path)
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
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Sod shock-tube analytical vs simulation results.")
    parser.add_argument('--speedup', type=float, default=1.0, help='Playback speedup factor. >1 for faster, <1 for slower.')
    parser.add_argument('--config-path', type=str, default='config.yaml', help='Path to the config.yaml file.')
    parser.add_argument('--output-path', type=str, default='out', help='Path to the simulation output directory.')
    args = parser.parse_args()
    main(args.speedup, args.config_path, args.output_path)

#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
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
        'xl': 0.0, 'xr': mesh_config['lx'], 'x0': mesh_config['lx'] / 2.0,
        'nx': mesh_config['nx']
    }
    return params


def load_simulation_data(out_dir_path):
    """Loads simulation data from all time steps from a single output directory."""
    out_dir = Path(out_dir_path)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {out_dir}")

    print(f"Reading from output directory: {out_dir}")

    pattern = re.compile(r"data_(\d+)\.csv")
    files = sorted(out_dir.glob("data_*.csv"), key=lambda f: int(pattern.match(f.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {out_dir}")

    sim_data_all_steps = []
    for f in files:
        match = pattern.match(f.name)
        step = int(match.group(1))

        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        num_cols = data.shape[1]
        all_index = {'x': 0, 'ux': 1, 'rho': num_cols - 2, 'p': num_cols - 1}

        sim_data_all_steps.append({
            'step': step,
            'x': data[:, all_index['x']],
            'ux': data[:, all_index['ux']],
            'rho': data[:, all_index['rho']],
            'p': data[:, all_index['p']],
        })

    return sim_data_all_steps


def calculate_errors(sim, anl):
    """Calculates RMSE and Max error."""
    rmse = np.sqrt(np.mean((sim - anl) ** 2))
    max_err = np.max(np.abs(sim - anl))
    return rmse, max_err


def main(base_dir):
    results = []
    out_dirs = sorted([p for p in Path(base_dir).glob("out*") if p.is_dir()])

    for out_dir in out_dirs:
        try:
            config_path = out_dir / 'config.yaml'
            if not config_path.exists():
                print(f"Skipping {out_dir}: config.yaml not found")
                continue

            params = load_config(config_path)
            sim_data_all_steps = load_simulation_data(out_dir)

            sim_rho_all, anl_rho_all = [], []
            sim_p_all, anl_p_all = [], []
            sim_u_all, anl_u_all = [], []

            for sim_data in sim_data_all_steps:
                sim_time = sim_data['step'] * params['sim_dt']

                # Get analytical solution on a fine grid
                left_state = (params['P_L'], params['rho_L'], params['u_L'])
                right_state = (params['P_R'], params['rho_R'], params['u_R'])
                geo = (params['xl'], params['xr'], params['x0'])
                _, _, an_vals = solve(left_state, right_state, geo, sim_time, gamma=params['gamma'], npts=10000)

                # Interpolate analytical solution to simulation grid points
                x_sim = sim_data['x']
                rho_an_interp = np.interp(x_sim, an_vals['x'], an_vals['rho'])
                p_an_interp = np.interp(x_sim, an_vals['x'], an_vals['p'])
                u_an_interp = np.interp(x_sim, an_vals['x'], an_vals['u'])

                # Append data for error calculation
                sim_rho_all.append(sim_data['rho'])
                anl_rho_all.append(rho_an_interp)
                sim_p_all.append(sim_data['p'])
                anl_p_all.append(p_an_interp)
                sim_u_all.append(sim_data['ux'])
                anl_u_all.append(u_an_interp)

            # Concatenate all data
            sim_rho_all = np.concatenate(sim_rho_all)
            anl_rho_all = np.concatenate(anl_rho_all)
            sim_p_all = np.concatenate(sim_p_all)
            anl_p_all = np.concatenate(anl_p_all)
            sim_u_all = np.concatenate(sim_u_all)
            anl_u_all = np.concatenate(anl_u_all)

            # Calculate errors
            rmse_rho, max_err_rho = calculate_errors(sim_rho_all, anl_rho_all)
            rmse_p, max_err_p = calculate_errors(sim_p_all, anl_p_all)
            rmse_u, max_err_u = calculate_errors(sim_u_all, anl_u_all)

            pressure_ratio = params['P_L'] / params['P_R']

            results.append({
                'nx': params['nx'],
                'pressure_ratio': pressure_ratio,
                'rmse_rho': rmse_rho, 'max_err_rho': max_err_rho,
                'rmse_p': rmse_p, 'max_err_p': max_err_p,
                'rmse_u': rmse_u, 'max_err_u': max_err_u
            })
            print(f"Processed {out_dir}: nx={params['nx']}, p_ratio={pressure_ratio:.1f}, RMSE_rho={rmse_rho:.2e}")

        except Exception as e:
            print(f"Error processing {out_dir}: {e}")

    # Plotting
    plot_data = {}
    for res in results:
        pr = res['pressure_ratio']
        if pr not in plot_data:
            plot_data[pr] = {'nx': [], 'rmse_rho': []}
        plot_data[pr]['nx'].append(res['nx'])
        plot_data[pr]['rmse_rho'].append(res['rmse_rho'])

    fig, ax = plt.subplots(figsize=(10, 6))
    for pr, data in sorted(plot_data.items()):
        # Sort data by nx before plotting
        sorted_data = sorted(zip(data['nx'], data['rmse_rho']))
        nxs = [d[0] for d in sorted_data]
        rmses = [d[1] for d in sorted_data]
        ax.plot(nxs, rmses, linestyle='',marker='o', markersize=4, label=f'$p_L/p_R = {pr:.1f}$')

    ax.set_xlabel('Number of lattices in x ($n_x$)')
    ax.set_ylabel('RMSE for Density ($\\rho$)')
    ax.set_title('Simulation Error vs. Grid Resolution')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--")
    ax.legend()

    plt.savefig(Path(base_dir) / 'error.png')
    print(f"Plot saved to {Path(base_dir) / 'error.png'}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Sod shock-tube simulation results and compare with analytical solutions.")
    parser.add_argument('--outpath', type=str, default='.', help='Directory containing the out* subdirectories.')
    args = parser.parse_args()
    main(args.outpath)

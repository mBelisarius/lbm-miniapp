import matplotlib
matplotlib.use('TkAgg')  # Force real window

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

from pathlib import Path
from scipy.optimize import brentq

sim_dt = 0.000272772  # Simulation time step (seconds)
speedup = 0.01  # >1 = faster playback, <1 = slower

# Find the latest output directory
build_dir = Path("./cmake-build-debug")
out_dirs = [p for p in build_dir.glob("out*") if p.is_dir()]
if not out_dirs:
    raise FileNotFoundError(f"No 'out*' directories found in {build_dir}")
out_dir = max(out_dirs, key=lambda p: p.stat().st_mtime)
print(f"Reading from latest output directory: {out_dir}")

pattern = re.compile(r"data_(\d+)\.csv")


def sod_analytical(x, t, gamma=1.4, P_L=1.0, P_R=0.1, rho_L=1.0, rho_R=0.125):
    """
    Computes the analytical solution for the Sod shock tube problem.
    Returns density, pressure, and velocity.
    """
    # Sound speed in the left state
    c_L = np.sqrt(gamma * P_L / rho_L)

    # Function to solve for P_star.
    # This function equates the velocity in the star region calculated from the
    # left (rarefaction wave) and right (shock wave) sides.
    def get_P_star_func(p_star):
        # Velocity in star region from rarefaction wave (left side)
        u_star_rarefaction = (2 * c_L / (gamma - 1)) * (1 - (p_star / P_L)**((gamma - 1) / (2 * gamma)))

        # Velocity in star region from shock wave (right side)
        u_star_shock = (p_star - P_R) * np.sqrt((2 / ((gamma + 1) * rho_R)) / (p_star + (gamma - 1) / (gamma + 1) * P_R))

        return u_star_rarefaction - u_star_shock

    # Find P_star using the Brent-Dekker root-finding method
    P_star = brentq(get_P_star_func, P_R, P_L)

    # Now calculate the rest of the solution using this P_star
    # Density in the star region (behind the contact discontinuity on the right of it)
    rho_star_R = rho_R * ((P_star / P_R + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * P_star / P_R + 1))
    # Density in the star region (behind the contact discontinuity on the left of it)
    rho_star_L = rho_L * (P_star / P_L)**(1 / gamma)

    # Velocity in the star region
    u_star = (2 * c_L / (gamma - 1)) * (1 - (P_star / P_L)**((gamma - 1) / (2 * gamma)))

    # Shock speed
    v_shock = u_star * (rho_star_R / rho_R) / (rho_star_R / rho_R - 1)

    # Positions of the waves
    x_shock = 0.5 + v_shock * t
    x_contact = 0.5 + u_star * t
    c_star_L = np.sqrt(gamma * P_star / rho_star_L)
    x_fan_head = 0.5 + (u_star - c_star_L) * t
    x_fan_tail = 0.5 - c_L * t

    # Compute the profiles
    rho = np.zeros_like(x)
    P = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, xi in enumerate(x):
        if t == 0:  # Initial condition
            if xi < 0.5:
                rho[i], P[i], u[i] = rho_L, P_L, 0.0
            else:
                rho[i], P[i], u[i] = rho_R, P_R, 0.0
            continue

        if xi < x_fan_tail:
            rho[i] = rho_L
            P[i] = P_L
            u[i] = 0.0
        elif xi < x_fan_head:
            # Rarefaction fan
            u_fan = (2 / (gamma + 1)) * (c_L + (xi - 0.5) / t)
            rho[i] = rho_L * (1 - (gamma - 1) / 2 * u_fan / c_L)**(2 / (gamma - 1))
            P[i] = P_L * (rho[i] / rho_L)**gamma
            u[i] = u_fan
        elif xi < x_contact:
            rho[i] = rho_star_L
            P[i] = P_star
            u[i] = u_star
        elif xi < x_shock:
            rho[i] = rho_star_R
            P[i] = P_star
            u[i] = u_star
        else:
            rho[i] = rho_R
            P[i] = P_R
            u[i] = 0.0

    return rho, P, u


def main():
    # Find and sort files by step number
    files = sorted(
        out_dir.glob("data_*.csv"),
        key=lambda f: int(pattern.match(f.name).group(1))
    )

    if not files:
        raise FileNotFoundError(f"No data_*.csv files found in {out_dir}")

    steps = []
    all_x = []
    all_rho = []
    all_p = []
    all_ux = []
    anl_x = np.linspace(0.0, 1.0, 1000)

    for f in files:
        match = pattern.match(f.name)
        step = int(match.group(1))
        # Use genfromtxt for CSV with header
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        all_x.append(data[:, 0])  # x
        all_ux.append(data[:, 1])  # ux
        all_p.append(data[:, 4])  # pressure
        all_rho.append(data[:, 5])  # density
        steps.append(step)

    # Animation interval in milliseconds
    dt_sim_per_frame = (steps[1] - steps[0]) * sim_dt if len(steps) > 1 else 0
    interval_ms = dt_sim_per_frame / speedup * 1000.0 if speedup > 0 else 0

    if len(steps) > 1:
        print(f"Detected step spacing: {steps[1] - steps[0]}")
    print(f"Simulation dt: {sim_dt}")
    print(f"Playback speedup: {speedup}x")

    # Plot setup
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Density plot
    line1_sim, = ax1.plot([], [], 'b-', lw=2, label="Simulation")
    line1_anl, = ax1.plot([], [], 'k--', lw=2, label="Analytical")
    ax1.set_xlim(min(all_x[0]), max(all_x[0]))
    ax1.set_ylim(0.0, 1.1)
    ax1.set_ylabel("Density")
    ax1.legend()
    title = ax1.set_title("")

    # Pressure plot
    line2_sim, = ax2.plot([], [], 'b-', lw=2, label="Simulation")
    line2_anl, = ax2.plot([], [], 'k--', lw=2, label="Analytical")
    ax2.set_ylim(0.0, 1.1)
    ax2.set_ylabel("Pressure")
    ax2.legend()

    # Velocity plot
    line3_sim, = ax3.plot([], [], 'b-', lw=2, label="Simulation")
    line3_anl, = ax3.plot([], [], 'k--', lw=2, label="Analytical")
    ax3.set_ylim(-0.1, 1.0)
    ax3.set_ylabel("Velocity (ux)")
    ax3.set_xlabel("x")
    ax3.legend()

    def init():
        line1_sim.set_data([], [])
        line1_anl.set_data([], [])
        line2_sim.set_data([], [])
        line2_anl.set_data([], [])
        line3_sim.set_data([], [])
        line3_anl.set_data([], [])
        title.set_text("")
        return line1_sim, line1_anl, line2_sim, line2_anl, line3_sim, line3_anl, title

    def update(frame):
        sim_time = steps[frame] * sim_dt

        # Update simulation data
        line1_sim.set_data(all_x[frame], all_rho[frame])
        line2_sim.set_data(all_x[frame], all_p[frame])
        line3_sim.set_data(all_x[frame], all_ux[frame])

        # Calculate and plot analytical solutions
        rho_analytical, p_analytical, u_analytical = sod_analytical(anl_x, sim_time)
        line1_anl.set_data(anl_x, rho_analytical)
        line2_anl.set_data(anl_x, p_analytical)
        line3_anl.set_data(anl_x, u_analytical)

        title.set_text(f"Profiles — t = {sim_time:.6f} s (Step {steps[frame]})")
        return line1_sim, line1_anl, line2_sim, line2_anl, line3_sim, line3_anl, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(steps),
        init_func=init, blit=True, interval=interval_ms
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

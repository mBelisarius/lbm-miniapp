import matplotlib
matplotlib.use('TkAgg')  # Force real window

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import re

sim_dt = 1.0e-3  # Simulation time step (seconds)
speedup = 1.0e-1  # >1 = faster playback, <1 = slower

out_dir = Path("./cmake-build-debug/out")
pattern = re.compile(r"density_(\d+)\.dat")


def main():
    # Find and sort files by step number
    files = sorted(
        out_dir.glob("density_*.dat"),
        key=lambda f: int(pattern.match(f.name).group(1))
    )

    if not files:
        raise FileNotFoundError(f"No density_*.dat files found in {out_dir}")

    steps = []
    all_x = []
    all_rho = []

    for f in files:
        match = pattern.match(f.name)
        step = int(match.group(1))
        data = np.loadtxt(f)
        all_x.append(data[:, 0])
        all_rho.append(data[:, 1])
        steps.append(step)

    # Animation interval in milliseconds
    dt_sim_per_frame = (steps[1] - steps[0]) * sim_dt
    interval_ms = dt_sim_per_frame / speedup * 1000.0

    print(f"Detected step spacing: {steps[1] - steps[0]}")
    print(f"Simulation dt: {sim_dt}")
    print(f"Playback speedup: {speedup}x")

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot([], [], 'b-', lw=2)
    ax.set_xlim(min(all_x[0]), max(all_x[0]))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(frame):
        line.set_data(all_x[frame], all_rho[frame])
        sim_time = steps[frame] * sim_dt
        title.set_text(f"Density Profile — t = {sim_time:.6f} s (Step {steps[frame]})")
        return line, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(steps),
        init_func=init, blit=True, interval=interval_ms
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

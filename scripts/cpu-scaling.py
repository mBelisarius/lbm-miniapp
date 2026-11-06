#!/usr/bin/env python3
"""CPU scaling driver: strong scaling of a backend and the machine's throughput ceiling.

Two modes, both on the Sod tube with ny = nx / ratio (see docs/performance.md):

  scaling   run the backend on 1..N cores, report MLUPS, speed-up and efficiency
            (thread placement: OMP_PLACES=cores, OMP_PROC_BIND=close);
  ceiling   run K *independent* single-thread copies at once, one per core, and
            report the summed MLUPS. There are no barriers and nothing is shared
            between the copies. This is a scaling diagnostic, not a roofline or a
            strict upper bound: independent copies use different working sets.
  paired    alternate --baseline-exe and --exe, discard --warmups runs, and
            retain all repetitions, executable hashes and measured provenance.

Examples (from the repository root, after building lbm-miniapp-main):

  scripts/cpu-scaling.py scaling --exe build/lbm-miniapp-main --backend OpenMP --nx 1000,5000 --cores 1,2,4,8
  scripts/cpu-scaling.py ceiling --exe build/lbm-miniapp-main --backend OpenMP --nx 5000 --copies 1,2,4,8

`--target GPU` runs the same sweep on the device (the core count is then meaningless and fixed at 1), which
gives the MLUPS-versus-mesh curve of a GPU backend. `--json FILE --label NAME` appends every result, tagged
with the label, to a JSON list; `thesis/campaign-results/protocol-final.py` draws the thesis figures from it.

Run it on a quiet machine: another job on the same cores makes the 8-core numbers meaningless
(`best` and `median` far apart are the symptom).
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

CONFIG = """Fluid:
    densityL: 1.0
    pressureL: 1.0
    densityR: 0.125
    pressureR: 0.1
    viscosity: 1.0e-2
    prandtl: 0.71
    gamma: 1.4
Mesh:
    lx: 1.0
    ly: 1.0
    lz: 1.0
    nx: {nx}
    ny: {ny}
    nz: 1
Control:
    tmax: {tmax}
    Ux: 0.25
    Uy: 0.0
    Uz: 0.0
    printStep: 1
Performance:
    target: {target}
    backend: {backend}
    cores: {cores}
"""


def parse(output):
    """(time-marching seconds, last time step) of one run, or None."""
    marching = re.search(r"Time-marching time: ([0-9.eE+-]+) s", output)
    steps = re.findall(r"Time step: (\d+)", output)
    if not marching or not steps:
        return None
    return float(marching.group(1)), int(steps[-1])


def write_config(directory, args, nx, backend, cores):
    path = os.path.join(directory, "config.yaml")
    with open(path, "w") as handle:
        handle.write(CONFIG.format(nx=nx, ny=max(1, nx // args.ratio), tmax=args.tmax, backend=backend, cores=cores,
                                   target=args.target))
    return path


def record(args, **fields):
    """Appends one result to the JSON file of `--json` (a list of records), tagged with `--label`."""
    if not args.json:
        return
    data = []
    if os.path.exists(args.json):
        with open(args.json) as handle:
            data = json.load(handle)
    data.append(dict(label=args.label, backend=args.backend, target=args.target, **fields))
    with open(args.json, "w") as handle:
        json.dump(data, handle, indent=1)


def run_threads(args, nx, cores, exe=None):
    """One run with `cores` threads (ranks for MPI); returns (seconds, steps) or None."""
    with tempfile.TemporaryDirectory() as tmp:
        config = write_config(tmp, args, nx, args.backend, cores)
        env = dict(os.environ, OMP_PLACES="cores", OMP_PROC_BIND="close",
                   OMP_NUM_THREADS=str(cores), ACC_NUM_CORES=str(cores))
        command = []
        if args.backend == "MPI":
            command = ["mpirun", "-np", str(cores), "--bind-to", "core", "--map-by", "core", "--oversubscribe"]
        command += [exe or args.exe, "--config", config, "--outpath", os.path.join(tmp, "out")]
        done = subprocess.run(command, env=env, capture_output=True, text=True, cwd=args.cwd or tmp)
        output = done.stdout + done.stderr
        result = parse(output)
        if done.returncode != 0 or result is None or result[0] <= 0 or result[1] <= 0:
            raise RuntimeError(f"benchmark failed ({' '.join(command)}):\n{output}")
        return result


def paired(args):
    """Alternate baseline/candidate order; retain every timing rather than only the winner."""
    executables = {"before": args.baseline_exe, "after": args.exe}
    hashes = {}
    for version, exe in executables.items():
        with open(exe, "rb") as handle:
            hashes[version] = hashlib.sha256(handle.read()).hexdigest()
    print(f"# paired {args.backend}/{args.target}: before -> after (median MLUPS; best-of-N MLUPS, less sensitive to a busy host)")
    for nx in args.nx:
        cells = nx * max(1, nx // args.ratio)
        for cores in args.cores:
            samples = {version: [] for version in executables}
            steps = None
            for repetition in range(args.warmups + args.reps):
                order = ("before", "after") if repetition % 2 == 0 else ("after", "before")
                for version in order:
                    seconds, count = run_threads(args, nx, cores, executables[version])
                    if steps is not None and count != steps:
                        raise RuntimeError(f"inconsistent step counts: {steps} and {count}")
                    steps = count
                    if repetition >= args.warmups:
                        samples[version].append(seconds)
            rates = {}
            best = {}
            for version, times in samples.items():
                median = statistics.median(times)
                rates[version] = cells * steps / median / 1e6
                best[version] = cells * steps / min(times) / 1e6
                record(
                    args, mode="paired", variant=version, nx=nx, ny=max(1, nx // args.ratio), cores=cores,
                    steps=steps, tmax=args.tmax, best=min(times), median=median,
                    mlups=cells * steps / min(times) / 1e6, mlups_median=rates[version],
                    samples_seconds=times, warmups=args.warmups,
                    executable=os.path.relpath(executables[version]), executable_sha256=hashes[version],
                    measured_at=datetime.now(timezone.utc).isoformat(), origin="measured",
                )
            print(f"{nx:7d} {cores:3d}: {rates['before']:8.2f} -> {rates['after']:8.2f} "
                  f"({rates['after'] / rates['before']:5.3f}x)   best {best['before']:8.2f} -> {best['after']:8.2f} "
                  f"({best['after'] / best['before']:5.3f}x)", flush=True)


def run_copies(args, nx, copies):
    """`copies` simultaneous single-thread runs, each pinned to its own core; returns [seconds], steps."""
    cpus = sorted(os.sched_getaffinity(0))[:: args.cpu_stride]
    if len(cpus) < copies:
        raise SystemExit(f"only {len(cpus)} cores available at stride {args.cpu_stride}, {copies} copies requested")
    tmps = [tempfile.TemporaryDirectory() for _ in range(copies)]
    procs = []
    for tmp, cpu in zip(tmps, cpus):
        config = write_config(tmp.name, args, nx, args.backend, 1)
        env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PLACES="cores", OMP_PROC_BIND="close")
        command = ["taskset", "-c", str(cpu), args.exe, "--config", config, "--outpath", os.path.join(tmp.name, "out")]
        procs.append(subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, cwd=args.cwd or tmp.name))
    results = [parse(p.communicate()[0]) for p in procs]
    for tmp in tmps:
        tmp.cleanup()
    if any(r is None for r in results):
        return None
    return [r[0] for r in results], results[0][1]


def scaling(args):
    print(f"# {args.backend} ({args.exe})")
    print(f"{'nx':>7} {'cores':>5} {'best[s]':>10} {'med[s]':>10} {'MLUPS':>8} {'steps':>6} {'speedup':>8} {'eff':>5}")
    for nx in args.nx:
        cells = nx * max(1, nx // args.ratio)
        reference = None
        for cores in args.cores:
            runs = [run_threads(args, nx, cores) for _ in range(args.reps)]
            runs = [r for r in runs if r]
            if not runs:
                print(f"{nx:>7} {cores:>5}  FAILED")
                continue
            best = min(r[0] for r in runs)
            steps = runs[0][1]
            reference = reference or (best, cores)
            speedup = reference[0] / best
            median = statistics.median(r[0] for r in runs)
            print(f"{nx:>7} {cores:>5} {best:>10.4f} {median:>10.4f} "
                  f"{cells * steps / best / 1e6:>8.2f} {steps:>6} {speedup:>8.2f} {speedup / (cores / reference[1]):>5.2f}")
            sys.stdout.flush()
            record(args, mode="scaling", nx=nx, cores=cores, best=best, median=median, steps=steps,
                   mlups=cells * steps / best / 1e6, mlups_median=cells * steps / median / 1e6)


def ceiling(args):
    print(f"# {args.backend} ({args.exe}): independent single-thread copies")
    for nx in args.nx:
        cells = nx * max(1, nx // args.ratio)
        for copies in args.copies:
            best = None
            for _ in range(args.reps):
                run = run_copies(args, nx, copies)
                if run and (best is None or sum(run[0]) < sum(best[0])):
                    best = run
            if best is None:
                print(f"nx {nx:>6} copies {copies:>2}: FAILED")
                continue
            times, steps = best
            rates = [cells * steps / t / 1e6 for t in times]
            print(f"nx {nx:>6} copies {copies:>2}: mean {statistics.mean(rates):6.2f} MLUPS/copy, "
                  f"min {min(rates):6.2f}, total {sum(rates):7.2f} MLUPS")
            sys.stdout.flush()
            record(args, mode="ceiling", nx=nx, copies=copies, steps=steps, mlups_total=sum(rates),
                   mlups_mean=statistics.mean(rates), mlups_min=min(rates))


def integers(text):
    return [int(v) for v in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["scaling", "ceiling", "paired"])
    parser.add_argument("--exe", required=True, help="path of lbm-miniapp-main")
    parser.add_argument("--baseline-exe", help="paired: unchanged executable built with the same toolchain")
    parser.add_argument("--warmups", type=int, default=1, help="paired: discarded runs per executable/configuration")
    parser.add_argument("--backend", default="OpenMP")
    parser.add_argument("--nx", type=integers, default=[1000], help="comma-separated mesh widths")
    parser.add_argument("--cores", type=integers, default=[1], help="scaling: comma-separated core counts")
    parser.add_argument("--copies", type=integers, default=[1], help="ceiling: comma-separated copy counts")
    parser.add_argument("--reps", type=int, default=3, help="repetitions; the best one is reported")
    parser.add_argument("--ratio", type=int, default=100, help="ny = nx / ratio")
    parser.add_argument("--tmax", type=float, default=0.05)
    parser.add_argument("--cpu-stride", type=int, default=2,
                        help="ceiling: logical CPUs per physical core (2 with SMT, so copies get distinct cores)")
    parser.add_argument("--cwd", default="", help="working directory of the runs (the OpenCL backend reads its "
                                                  "kernels from lbmini/Lbm/OpenCl relative to it)")
    parser.add_argument("--target", choices=["CPU", "GPU"], default="CPU", help="device of the runs")
    parser.add_argument("--json", default="", help="append every result to this JSON list")
    parser.add_argument("--label", default="", help="tag stored with every result of --json (e.g. old / new)")
    args = parser.parse_args()
    if args.reps < 1 or args.warmups < 0 or args.ratio < 1 or args.tmax <= 0:
        parser.error("reps, ratio and tmax must be positive; warmups must be nonnegative")
    if any(value < 1 for value in args.nx + args.cores + args.copies):
        parser.error("mesh widths and core/copy counts must be positive")
    if args.mode == "paired" and not args.baseline_exe:
        parser.error("paired requires --baseline-exe")
    args.exe = os.path.abspath(args.exe)
    if args.baseline_exe:
        args.baseline_exe = os.path.abspath(args.baseline_exe)
    if args.target == "GPU" and args.mode in ("scaling", "paired"):
        args.cores = [1]
    {"scaling": scaling, "ceiling": ceiling, "paired": paired}[args.mode](args)


if __name__ == "__main__":
    main()

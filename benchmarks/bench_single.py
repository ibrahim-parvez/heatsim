"""Benchmark every backend across grid sizes on one machine.

Writes one row per (backend, n) to results/single.csv. Each measurement picks
a step count that takes roughly --target seconds, runs it --repeats times and
keeps the median.

Effective bandwidth assumes the ideal stencil traffic of one read + one write
per cell per step (8 bytes in float32). Anything the caches don't catch on top
of that isn't counted, so this is a lower bound on actual DRAM traffic.

    python benchmarks/bench_single.py
    python benchmarks/bench_single.py --sizes 512 1024 2048 --backends numba metal-tiled
"""

import argparse
import csv
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heatsim.grid import initial_condition  # noqa: E402
from heatsim.solvers import ALL_BACKENDS, available_backends, get_solver  # noqa: E402

BYTES_PER_CELL = 8
DEFAULT_SIZES = [256, 512, 1024, 2048, 4096, 8192]


def time_steps(solver, steps):
    t0 = time.perf_counter()
    solver.advance(steps)
    return time.perf_counter() - t0


def pick_steps(solver, target):
    """Run a few steps to estimate the per-step cost, then scale up to hit `target` seconds."""
    steps = 2
    while True:
        dt = time_steps(solver, steps)
        if dt > 0.05 or steps >= 4096:
            break
        steps *= 4
    per_step = dt / steps
    return max(3, min(20000, int(target / per_step)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--backends", nargs="+", default=None, choices=ALL_BACKENDS)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--target", type=float, default=1.0, help="seconds per measurement")
    p.add_argument("--max-numpy", type=int, default=8192, help="skip numpy above this size")
    p.add_argument("--out", default="results/single.csv")
    args = p.parse_args()

    backends = args.backends or available_backends()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    print(f"machine: {platform.machine()} {platform.processor()}  numba threads: "
          f"{os.environ.get('NUMBA_NUM_THREADS', 'default')}")
    print(f"{'backend':12s} {'n':>6s} {'steps':>6s} {'ms/step':>9s} {'MLUPS':>8s} {'GB/s':>7s}")

    for backend in backends:
        solver = get_solver(backend, 0.24)
        solver.warmup()
        for n in args.sizes:
            if backend == "numpy" and n > args.max_numpy:
                continue
            solver.load(initial_condition("hotspot", n))
            steps = pick_steps(solver, args.target)
            times = [time_steps(solver, steps) for _ in range(args.repeats)]
            t = statistics.median(times)

            cells = (n - 2) ** 2
            row = {
                "backend": backend,
                "n": n,
                "steps": steps,
                "seconds": t,
                "ms_per_step": t / steps * 1e3,
                "mlups": cells * steps / t / 1e6,
                "gbps": n * n * BYTES_PER_CELL * steps / t / 1e9,
                "min_seconds": min(times),
            }
            rows.append(row)
            print(f"{backend:12s} {n:6d} {steps:6d} {row['ms_per_step']:9.3f} "
                  f"{row['mlups']:8.0f} {row['gbps']:7.1f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

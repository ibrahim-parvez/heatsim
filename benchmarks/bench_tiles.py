"""Sweep threadgroup shapes for the tiled Metal kernel (and the naive one for reference).

    python benchmarks/bench_tiles.py --n 4096
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heatsim.grid import initial_condition  # noqa: E402
from heatsim.solvers.metal_solver import MetalNaiveSolver, MetalTiledSolver  # noqa: E402

SHAPES = [(8, 8), (16, 8), (16, 16), (32, 4), (32, 8), (32, 16), (32, 32), (64, 4), (64, 8), (128, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="results/tiles.csv")
    args = p.parse_args()

    u0 = initial_condition("hotspot", args.n)
    rows = []
    for kind, cls in (("naive", MetalNaiveSolver), ("tiled", MetalTiledSolver)):
        for shape in SHAPES:
            if shape[0] * shape[1] > 1024:
                continue
            s = cls(0.24, shape)
            s.warmup()
            s.load(u0)
            s.advance(10)
            t0 = time.perf_counter()
            s.advance(args.steps)
            t = time.perf_counter() - t0
            gbps = args.n**2 * 8 * args.steps / t / 1e9
            rows.append({"kernel": kind, "tx": shape[0], "ty": shape[1], "ms_per_step": t / args.steps * 1e3, "gbps": gbps})
            print(f"{kind:6s} {shape[0]:4d}x{shape[1]:<4d} {t / args.steps * 1e3:7.3f} ms/step  {gbps:6.1f} GB/s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

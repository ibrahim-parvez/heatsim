"""One MPI scaling measurement. Appends a row to the CSV from rank 0.

Normally called from scripts/run_scaling.sh, but you can run it directly:

    mpirun -n 4 python benchmarks/bench_mpi.py --n 4096 --steps 200 --mode strong
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpi4py import MPI  # noqa: E402

from heatsim.mpi_solver import DistributedHeat  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--mode", choices=["strong", "weak"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="results/mpi.csv")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    sim = DistributedHeat(args.n, 0.24, "hotspot", comm)
    sim.advance(2)  # numba compile + first touch

    best = None
    for _ in range(args.repeats):
        sim.halo.time = 0.0
        sim.compute_time = 0.0
        comm.Barrier()
        t0 = MPI.Wtime()
        sim.advance(args.steps)
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        halo = comm.allreduce(sim.halo.time, op=MPI.MAX)
        compute = comm.allreduce(sim.compute_time, op=MPI.MAX)
        if best is None or elapsed < best[0]:
            best = (elapsed, halo, compute)

    if sim.rank == 0:
        elapsed, halo, compute = best
        size = comm.Get_size()
        row = {
            "mode": args.mode,
            "ranks": size,
            "px": sim.dims[0],
            "py": sim.dims[1],
            "n": args.n,
            "cells_per_rank": args.n * args.n // size,
            "steps": args.steps,
            "seconds": elapsed,
            "halo_seconds": halo,
            "compute_seconds": compute,
            "mlups": (args.n - 2) ** 2 * args.steps / elapsed / 1e6,
        }
        new = not os.path.exists(args.out)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{args.mode:6s} ranks={size:2d} dims={sim.dims} n={args.n:5d} "
              f"{elapsed:.3f}s  halo={halo / elapsed * 100:4.1f}%  {row['mlups']:.0f} MLUPS")


if __name__ == "__main__":
    main()

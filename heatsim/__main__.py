"""Command line entry point.

    python -m heatsim run --backend metal-tiled --n 1024 --steps 5000 --png results/out.png
    python -m heatsim run --n 512 --steps 20000 --init edge --gif results/edge.gif --frames 60
    python -m heatsim backends
"""

import argparse
import sys
import time

import numpy as np

from heatsim.grid import INITIAL_CONDITIONS, SimConfig, initial_condition
from heatsim.solvers import ALL_BACKENDS, available_backends, get_solver


def cmd_run(args):
    cfg = SimConfig(n=args.n, steps=args.steps, r=args.r, init=args.init)
    solver = get_solver(args.backend, cfg.r)
    solver.warmup()

    u0 = initial_condition(cfg.init, cfg.n)
    solver.load(u0)

    frames = []
    if args.gif:
        # split the run into chunks and grab a frame after each one
        chunk = max(1, cfg.steps // args.frames)
        frames.append((0, u0))
        done = 0
        t0 = time.perf_counter()
        while done < cfg.steps:
            k = min(chunk, cfg.steps - done)
            solver.advance(k)
            done += k
            frames.append((done, solver.fetch()))
        elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        solver.advance(cfg.steps)
        elapsed = time.perf_counter() - t0

    u = solver.fetch()
    cells = (cfg.n - 2) ** 2 * cfg.steps
    print(f"{args.backend}: n={cfg.n} steps={cfg.steps} init={cfg.init}")
    print(f"  {elapsed:.3f}s total, {elapsed / cfg.steps * 1e3:.3f} ms/step, {cells / elapsed / 1e6:.0f} MLUPS")
    print(f"  t_final={cfg.steps * cfg.dt:.4g}  min={u.min():.3f} max={u.max():.3f} mean={u.mean():.4f}")

    if args.npy:
        np.save(args.npy, u)
    if args.png or args.gif:
        from heatsim.plotting import save_animation, save_heatmap

        vmax = float(u0.max())
        if args.png:
            save_heatmap(u, args.png, title=f"{cfg.init}, n={cfg.n}, {cfg.steps} steps", vmax=vmax)
            print(f"  saved {args.png}")
        if args.gif:
            save_animation(frames, args.gif, vmax=vmax)
            print(f"  saved {args.gif}")


def cmd_backends(_):
    avail = set(available_backends())
    for b in ALL_BACKENDS:
        print(f"  {b:12s} {'ok' if b in avail else 'not available'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="heatsim")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a single simulation")
    run.add_argument("--backend", default="metal-tiled", choices=ALL_BACKENDS)
    run.add_argument("--n", type=int, default=1024)
    run.add_argument("--steps", type=int, default=2000)
    run.add_argument("--r", type=float, default=0.24)
    run.add_argument("--init", default="hotspot", choices=INITIAL_CONDITIONS)
    run.add_argument("--png", help="save final state as an image")
    run.add_argument("--gif", help="save an animation of the run")
    run.add_argument("--frames", type=int, default=40)
    run.add_argument("--npy", help="save final state as .npy")
    run.set_defaults(func=cmd_run)

    sub.add_parser("backends", help="list backends that work on this machine").set_defaults(func=cmd_backends)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())

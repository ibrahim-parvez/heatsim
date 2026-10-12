"""The function that actually runs a job. It executes inside a worker process,
so it only takes/returns plain picklable things."""

import os
import time

import numpy as np


def run_job(job_id, params, out_dir):
    from heatsim.grid import SimConfig, initial_condition
    from heatsim.plotting import save_heatmap
    from heatsim.solvers import get_solver

    cfg = SimConfig(n=params["n"], steps=params["steps"], r=params["r"], init=params["init"])
    solver = get_solver(params["backend"], cfg.r)
    solver.warmup()

    u0 = initial_condition(cfg.init, cfg.n)
    solver.load(u0)
    t0 = time.perf_counter()
    solver.advance(cfg.steps)
    runtime = time.perf_counter() - t0
    u = solver.fetch()

    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    np.save(os.path.join(job_dir, "final.npy"), u)
    save_heatmap(u, os.path.join(job_dir, "final.png"),
                 title=f"{cfg.init}, n={cfg.n}, {cfg.steps} steps", vmax=float(u0.max()))

    cells = (cfg.n - 2) ** 2 * cfg.steps
    return {
        "runtime_seconds": runtime,
        "cell_updates": cells,
        "mlups": cells / runtime / 1e6,
        "t_final": cfg.steps * cfg.dt,
        "stats": {"min": float(u.min()), "max": float(u.max()), "mean": float(u.mean())},
    }

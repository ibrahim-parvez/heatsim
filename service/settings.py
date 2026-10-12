import os


def _env(name, default, cast=str):
    val = os.environ.get(name)
    return cast(val) if val is not None else default


DEFAULT_BACKEND = _env("HEATSIM_BACKEND", "metal-tiled")
WORKERS = _env("HEATSIM_WORKERS", 1, int)
RESULTS_DIR = _env("HEATSIM_RESULTS_DIR", "results/jobs")
MAX_N = _env("HEATSIM_MAX_N", 4096, int)
MAX_STEPS = _env("HEATSIM_MAX_STEPS", 200_000, int)

# Pretend hourly price so the dashboard can show "cost per run". The default is
# roughly what a single small cloud GPU instance costs on demand. Running locally
# it's obviously free, this just shows what the same workload would cost.
COST_PER_HOUR = _env("HEATSIM_COST_PER_HOUR", 0.53, float)

"""Backend registry. Imports are lazy so a missing optional dependency (e.g. MLX
inside the Linux docker image) only breaks the backends that need it."""

import importlib

_BACKENDS = {
    "numpy": ("heatsim.solvers.numpy_solver", "NumpySolver"),
    "numba": ("heatsim.solvers.numba_solver", "NumbaSolver"),
    "mlx": ("heatsim.solvers.metal_solver", "MLXSolver"),
    "metal-naive": ("heatsim.solvers.metal_solver", "MetalNaiveSolver"),
    "metal-tiled": ("heatsim.solvers.metal_solver", "MetalTiledSolver"),
}

ALL_BACKENDS = tuple(_BACKENDS)


def get_solver(name: str, r: float, **kwargs):
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend '{name}', options are {ALL_BACKENDS}")
    module, cls = _BACKENDS[name]
    return getattr(importlib.import_module(module), cls)(r, **kwargs)


def available_backends():
    out = []
    for name, (module, _) in _BACKENDS.items():
        try:
            importlib.import_module(module)
            out.append(name)
        except Exception:
            pass
    return out

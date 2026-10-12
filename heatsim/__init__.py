"""2D heat diffusion solvers: NumPy, Numba, MLX and hand-written Metal kernels."""

from heatsim.grid import SimConfig, initial_condition
from heatsim.solvers import available_backends, get_solver

__all__ = [
    "SimConfig",
    "initial_condition",
    "available_backends",
    "get_solver",
]

__version__ = "0.1.0"

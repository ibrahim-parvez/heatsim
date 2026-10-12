"""Problem setup: config, initial conditions, and the stability limit.

The equation is u_t = alpha * (u_xx + u_yy) on the unit square, discretized
with the explicit 5-point (FTCS) scheme:

    u'[i,j] = u[i,j] + r * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4 u[i,j])

with r = alpha * dt / dx^2. The scheme is stable for r <= 0.25.
Boundary cells are Dirichlet (held fixed at whatever the initial condition put there).
"""

from dataclasses import dataclass

import numpy as np

DTYPE = np.float32
INITIAL_CONDITIONS = ("hotspot", "edge", "rings", "random")


@dataclass
class SimConfig:
    n: int = 1024
    steps: int = 1000
    alpha: float = 1.0
    r: float = 0.24  # alpha*dt/dx^2, keep it under 0.25
    init: str = "hotspot"

    def __post_init__(self):
        if self.n < 3:
            raise ValueError("grid needs at least 3 points per side")
        if not 0 < self.r <= 0.25:
            raise ValueError(f"r={self.r} is unstable, needs 0 < r <= 0.25")
        if self.init not in INITIAL_CONDITIONS:
            raise ValueError(f"unknown init '{self.init}', pick from {INITIAL_CONDITIONS}")

    @property
    def dx(self) -> float:
        return 1.0 / (self.n - 1)

    @property
    def dt(self) -> float:
        return self.r * self.dx**2 / self.alpha


def initial_condition(name: str, n: int, rows=None, cols=None, seed: int = 0) -> np.ndarray:
    """Build the initial field for an n x n grid.

    rows/cols are (start, stop) ranges in global indices so that MPI ranks can
    build just their own block without allocating the whole grid.
    """
    r0, r1 = rows if rows is not None else (0, n)
    c0, c1 = cols if cols is not None else (0, n)

    # coordinates in [0, 1]
    y = (np.arange(r0, r1, dtype=np.float64) / (n - 1))[:, None]
    x = (np.arange(c0, c1, dtype=np.float64) / (n - 1))[None, :]

    if name == "hotspot":
        u = 100.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01)
    elif name == "edge":
        # top edge held at 100, everything else starts cold
        u = np.zeros((r1 - r0, c1 - c0))
        if r0 == 0:
            u[0, :] = 100.0
    elif name == "rings":
        d = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        u = 50.0 * (1.0 + np.cos(40.0 * d)) * (d < 0.4)
    elif name == "random":
        # seed from the block position so the MPI and serial grids match
        u = np.empty((r1 - r0, c1 - c0))
        for k, i in enumerate(range(r0, r1)):
            u[k] = np.random.default_rng(seed + i).random(n)[c0:c1] * 100.0
    else:
        raise ValueError(f"unknown initial condition '{name}'")

    u = np.broadcast_to(u, (r1 - r0, c1 - c0)).astype(DTYPE)

    # zero the outer boundary, except for the 'edge' case where the top row is the heat source
    if name != "edge":
        u = u.copy()
        if r0 == 0:
            u[0, :] = 0
        if r1 == n:
            u[-1, :] = 0
        if c0 == 0:
            u[:, 0] = 0
        if c1 == n:
            u[:, -1] = 0
    return np.ascontiguousarray(u)

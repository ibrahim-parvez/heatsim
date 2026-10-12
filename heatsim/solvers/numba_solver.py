import numpy as np
from numba import njit, prange

from heatsim.solvers.base import Solver


@njit(parallel=True, fastmath=True, cache=True)
def _advance_parallel(u, v, r, steps):
    ny, nx = u.shape
    for _ in range(steps):
        for i in prange(1, ny - 1):
            for j in range(1, nx - 1):
                c = u[i, j]
                v[i, j] = c + r * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0 * c)
        u, v = v, u


@njit(fastmath=True, cache=True)
def step_block(u, v, r, i0, i1, j0, j1):
    """Serial update of u[i0:i1, j0:j1] into v. Used by the MPI solver (one core per rank)."""
    for i in range(i0, i1):
        for j in range(j0, j1):
            c = u[i, j]
            v[i, j] = c + r * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0 * c)


class NumbaSolver(Solver):
    """Multithreaded CPU version. Rows are split across cores with prange."""

    name = "numba"

    def load(self, u0):
        self.u = np.array(u0, dtype=np.float32, copy=True)
        self.v = self.u.copy()

    def advance(self, steps):
        _advance_parallel(self.u, self.v, self.r, steps)
        # the kernel ping-pongs internally, so after an odd number of steps the result is in v
        if steps % 2 == 1:
            self.u, self.v = self.v, self.u

    def fetch(self):
        return self.u.copy()

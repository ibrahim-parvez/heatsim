import numpy as np

from heatsim.solvers.base import Solver


def step_numpy(u: np.ndarray, out: np.ndarray, r: np.float32) -> None:
    c = u[1:-1, 1:-1]
    out[1:-1, 1:-1] = c + r * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * c)


class NumpySolver(Solver):
    """Vectorized NumPy baseline. Single threaded, allocates temporaries every step."""

    name = "numpy"

    def load(self, u0):
        self.u = np.array(u0, dtype=np.float32, copy=True)
        self.v = self.u.copy()  # boundary values never change, so copy them into both buffers

    def advance(self, steps):
        for _ in range(steps):
            step_numpy(self.u, self.v, self.r)
            self.u, self.v = self.v, self.u

    def fetch(self):
        return self.u.copy()

    def warmup(self):
        pass

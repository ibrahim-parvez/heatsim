import numpy as np


class Solver:
    """Common interface so the benchmark/service code doesn't care about the backend.

    load()    -> copy the initial field to wherever the backend works on it
    advance() -> run some steps and block until they're actually done
    fetch()   -> copy the current field back as a numpy array
    """

    name = "base"

    def __init__(self, r: float):
        self.r = np.float32(r)

    def load(self, u0: np.ndarray) -> None:
        raise NotImplementedError

    def advance(self, steps: int) -> None:
        raise NotImplementedError

    def fetch(self) -> np.ndarray:
        raise NotImplementedError

    def warmup(self) -> None:
        """Trigger any JIT compilation on a tiny grid so it isn't timed."""
        self.load(np.zeros((8, 8), dtype=np.float32))
        self.advance(2)

    def run(self, u0: np.ndarray, steps: int) -> np.ndarray:
        self.load(u0)
        self.advance(steps)
        return self.fetch()

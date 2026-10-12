"""GPU solvers for Apple Silicon.

Three versions:
  mlx          - plain MLX array ops (slicing), wrapped in mx.compile so it can fuse
  metal-naive  - hand written Metal kernel, every thread reads its 5 points from device memory
  metal-tiled  - same kernel but each threadgroup first loads a (TY+2) x (TX+2) tile into
                 threadgroup memory (Metal's version of CUDA __shared__), then computes from that

Metal doesn't support float64, so everything here is float32.
"""

import mlx.core as mx
import numpy as np

from heatsim.solvers.base import Solver

# how many steps to queue up lazily before forcing evaluation
EVAL_EVERY = 32


NAIVE_SOURCE = """
    int j = thread_position_in_grid.x;
    int i = thread_position_in_grid.y;
    int ny = u_shape[0];
    int nx = u_shape[1];
    if (i >= ny || j >= nx) return;

    int idx = i * nx + j;
    float c = u[idx];
    if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) {
        out[idx] = c;
        return;
    }
    out[idx] = c + r[0] * (u[idx - nx] + u[idx + nx] + u[idx - 1] + u[idx + 1] - 4.0f * c);
"""

# Grid is padded up to a multiple of the tile size so every threadgroup is full.
# Threads that land outside the domain still help load the tile (with clamped
# indices) and just skip the write - they can't return early because of the barrier.
TILED_SOURCE = """
    threadgroup float tile[TY + 2][TX + 2];

    int lx = thread_position_in_threadgroup.x;
    int ly = thread_position_in_threadgroup.y;
    int gx = thread_position_in_grid.x;
    int gy = thread_position_in_grid.y;
    int ny = u_shape[0];
    int nx = u_shape[1];

    int cx = min(gx, nx - 1);
    int cy = min(gy, ny - 1);
    int row = cy * nx;

    tile[ly + 1][lx + 1] = u[row + cx];
    if (lx == 0)      tile[ly + 1][0]      = u[row + max(cx - 1, 0)];
    if (lx == TX - 1) tile[ly + 1][TX + 1] = u[row + min(cx + 1, nx - 1)];
    if (ly == 0)      tile[0][lx + 1]      = u[max(cy - 1, 0) * nx + cx];
    if (ly == TY - 1) tile[TY + 1][lx + 1] = u[min(cy + 1, ny - 1) * nx + cx];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gx >= nx || gy >= ny) return;

    float c = tile[ly + 1][lx + 1];
    int idx = gy * nx + gx;
    if (gx == 0 || gy == 0 || gx == nx - 1 || gy == ny - 1) {
        out[idx] = c;
        return;
    }
    float lap = tile[ly][lx + 1] + tile[ly + 2][lx + 1]
              + tile[ly + 1][lx] + tile[ly + 1][lx + 2] - 4.0f * c;
    out[idx] = c + r[0] * lap;
"""

_naive_kernel = mx.fast.metal_kernel(
    name="heat_step_naive",
    input_names=["u", "r"],
    output_names=["out"],
    source=NAIVE_SOURCE,
)

_tiled_kernel = mx.fast.metal_kernel(
    name="heat_step_tiled",
    input_names=["u", "r"],
    output_names=["out"],
    source=TILED_SOURCE,
)


def _round_up(x, m):
    return (x + m - 1) // m * m


class _MetalBase(Solver):
    def load(self, u0):
        self.u = mx.array(np.asarray(u0, dtype=np.float32))
        self.r_arr = mx.array([float(self.r)], dtype=mx.float32)
        mx.eval(self.u, self.r_arr)

    def _step(self, u):
        raise NotImplementedError

    def advance(self, steps):
        u = self.u
        for k in range(steps):
            u = self._step(u)
            if (k + 1) % EVAL_EVERY == 0:
                mx.eval(u)
        mx.eval(u)
        mx.synchronize()
        self.u = u

    def fetch(self):
        return np.array(self.u)


class MetalNaiveSolver(_MetalBase):
    name = "metal-naive"

    def __init__(self, r, threadgroup=(32, 8)):
        super().__init__(r)
        self.tg = threadgroup

    def _step(self, u):
        ny, nx = u.shape
        return _naive_kernel(
            inputs=[u, self.r_arr],
            grid=(nx, ny, 1),
            threadgroup=(self.tg[0], self.tg[1], 1),
            output_shapes=[u.shape],
            output_dtypes=[mx.float32],
        )[0]


class MetalTiledSolver(_MetalBase):
    name = "metal-tiled"

    def __init__(self, r, tile=(32, 8)):
        super().__init__(r)
        self.tx, self.ty = tile

    def _step(self, u):
        ny, nx = u.shape
        return _tiled_kernel(
            inputs=[u, self.r_arr],
            template=[("TX", self.tx), ("TY", self.ty)],
            grid=(_round_up(nx, self.tx), _round_up(ny, self.ty), 1),
            threadgroup=(self.tx, self.ty, 1),
            output_shapes=[u.shape],
            output_dtypes=[mx.float32],
        )[0]


@mx.compile
def _mlx_step(u, r):
    c = u[1:-1, 1:-1]
    inner = c + r * (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4.0 * c)
    # stitch the fixed boundary back on
    top, bottom = u[:1, :], u[-1:, :]
    left, right = u[1:-1, :1], u[1:-1, -1:]
    middle = mx.concatenate([left, inner, right], axis=1)
    return mx.concatenate([top, middle, bottom], axis=0)


class MLXSolver(_MetalBase):
    """High level MLX version, mostly here to compare against the hand-written kernels."""

    name = "mlx"

    def _step(self, u):
        return _mlx_step(u, self.r_arr[0])

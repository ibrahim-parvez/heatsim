"""Distributed version using mpi4py with a 2D domain decomposition.

The global n x n grid is split over a px x py process grid (MPI_Dims_create picks
the shape). Each rank stores its block plus a 1-cell ghost layer:

    local array shape = (ly + 2, lx + 2)

Every step:
  1. halo exchange - send my edge rows/cols to the 4 neighbours, receive theirs into my ghosts
  2. update my owned cells with the 5-point stencil (numba, one core per rank)

Ranks on the outside of the domain have MPI.PROC_NULL as a neighbour, which turns the
send/recv into a no-op. Their outermost cells are physical boundary and never updated.

Run with something like:
    mpirun -n 4 python -m heatsim.mpi_solver --n 2048 --steps 500 --verify
"""

import argparse
import time

import numpy as np
from mpi4py import MPI

from heatsim.grid import initial_condition
from heatsim.solvers.numba_solver import step_block


def split(n, parts, idx):
    """Start/stop of chunk idx when splitting n points into `parts` nearly equal chunks."""
    base, extra = divmod(n, parts)
    start = idx * base + min(idx, extra)
    return start, start + base + (1 if idx < extra else 0)


class HaloExchanger:
    def __init__(self, cart, u):
        self.cart = cart
        self.up, self.down = cart.Shift(0, 1)
        self.left, self.right = cart.Shift(1, 1)
        ly, lx = u.shape[0] - 2, u.shape[1] - 2
        # columns aren't contiguous, so pack them into buffers
        self.send_l = np.empty(ly, dtype=u.dtype)
        self.send_r = np.empty(ly, dtype=u.dtype)
        self.recv_l = np.empty(ly, dtype=u.dtype)
        self.recv_r = np.empty(ly, dtype=u.dtype)
        self.time = 0.0

    def exchange(self, u):
        t0 = MPI.Wtime()
        c = self.cart
        # rows are contiguous so we can send straight out of the array
        c.Sendrecv(u[1, 1:-1], dest=self.up, sendtag=0, recvbuf=u[-1, 1:-1], source=self.down, recvtag=0)
        c.Sendrecv(u[-2, 1:-1], dest=self.down, sendtag=1, recvbuf=u[0, 1:-1], source=self.up, recvtag=1)

        self.send_l[:] = u[1:-1, 1]
        self.send_r[:] = u[1:-1, -2]
        c.Sendrecv(self.send_l, dest=self.left, sendtag=2, recvbuf=self.recv_r, source=self.right, recvtag=2)
        c.Sendrecv(self.send_r, dest=self.right, sendtag=3, recvbuf=self.recv_l, source=self.left, recvtag=3)
        if self.right != MPI.PROC_NULL:
            u[1:-1, -1] = self.recv_r
        if self.left != MPI.PROC_NULL:
            u[1:-1, 0] = self.recv_l
        self.time += MPI.Wtime() - t0


class DistributedHeat:
    def __init__(self, n, r, init="hotspot", comm=MPI.COMM_WORLD, dims=None):
        self.n = n
        self.r = np.float32(r)
        size = comm.Get_size()
        dims = dims or MPI.Compute_dims(size, 2)
        self.cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        self.rank = self.cart.Get_rank()
        self.dims = dims
        cy, cx = self.cart.Get_coords(self.rank)

        self.rows = split(n, dims[0], cy)
        self.cols = split(n, dims[1], cx)
        ly = self.rows[1] - self.rows[0]
        lx = self.cols[1] - self.cols[0]
        if ly < 1 or lx < 1:
            raise ValueError(f"grid {n} is too small for a {dims} process grid")

        self.u = np.zeros((ly + 2, lx + 2), dtype=np.float32)
        self.u[1:-1, 1:-1] = initial_condition(init, n, rows=self.rows, cols=self.cols)
        self.v = self.u.copy()

        # local index range of cells we actually update (skip the physical boundary)
        self.i0 = 2 if self.rows[0] == 0 else 1
        self.i1 = ly if self.rows[1] == n else ly + 1
        self.j0 = 2 if self.cols[0] == 0 else 1
        self.j1 = lx if self.cols[1] == n else lx + 1

        self.halo = HaloExchanger(self.cart, self.u)
        self.compute_time = 0.0

    def advance(self, steps):
        for _ in range(steps):
            self.halo.exchange(self.u)
            t0 = MPI.Wtime()
            step_block(self.u, self.v, self.r, self.i0, self.i1, self.j0, self.j1)
            self.compute_time += MPI.Wtime() - t0
            self.u, self.v = self.v, self.u

    def gather(self):
        """Collect the full grid on rank 0 (only for checking results, it's slow for big n)."""
        block = np.ascontiguousarray(self.u[1:-1, 1:-1])
        pieces = self.cart.gather((self.rows, self.cols, block), root=0)
        if self.rank != 0:
            return None
        full = np.empty((self.n, self.n), dtype=np.float32)
        for (r0, r1), (c0, c1), b in pieces:
            full[r0:r1, c0:c1] = b
        return full


def main():
    p = argparse.ArgumentParser(description="MPI heat diffusion")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--r", type=float, default=0.24)
    p.add_argument("--init", default="hotspot")
    p.add_argument("--verify", action="store_true", help="compare against the serial numpy solver")
    p.add_argument("--save", help="save the final grid to this .npy file (rank 0)")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    sim = DistributedHeat(args.n, args.r, args.init, comm)

    # one step first so numba compiles before we start timing
    sim.advance(1)
    comm.Barrier()
    t0 = MPI.Wtime()
    sim.advance(args.steps)
    comm.Barrier()
    elapsed = MPI.Wtime() - t0

    halo = comm.reduce(sim.halo.time, op=MPI.MAX, root=0)
    full = sim.gather() if (args.verify or args.save) else None

    if sim.rank == 0:
        cells = (args.n - 2) ** 2 * args.steps
        print(f"ranks={comm.Get_size()} dims={sim.dims} n={args.n} steps={args.steps}")
        print(f"time={elapsed:.3f}s  ({elapsed / args.steps * 1e3:.3f} ms/step)  "
              f"{cells / elapsed / 1e6:.1f} MLUPS  max halo time={halo:.3f}s")
        if args.save:
            np.save(args.save, full)
        if args.verify:
            from heatsim.solvers.numpy_solver import NumpySolver
            ref = NumpySolver(args.r).run(initial_condition(args.init, args.n), args.steps + 1)
            err = float(np.abs(full - ref).max())
            print(f"max abs diff vs numpy: {err:.3e}  {'OK' if err < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()

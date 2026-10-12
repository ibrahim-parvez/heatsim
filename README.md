# heatsim: 2D heat diffusion on CPU, Apple GPU and MPI

A 2D heat equation solver written five different ways: NumPy, multithreaded Numba, plain MLX, and two hand-written Metal kernels. On top of that there's an MPI version with 2D domain decomposition, and a small job service with Prometheus/Grafana monitoring.

The heat equation stencil is the standard memory-bound HPC benchmark. Each cell does very little math and a lot of loads and stores, so the question is how close each version gets to the hardware's memory bandwidth.

Everything runs locally on an M4 Pro MacBook (14 CPU cores: 10 performance + 4 efficiency, 20-core GPU, 273 GB/s unified memory).

<p align="center">
  <img src="docs/hotspot.gif" width="360" alt="heat spreading out from a hot spot">
  <img src="docs/edge.png" width="400" alt="steady state with the top edge held hot">
</p>

## Results at a glance

- **25× faster than NumPy** with a hand-written Metal kernel on a 4096² grid
- **209 GB/s effective memory bandwidth** at 4096² (76% of the M4 Pro's 273 GB/s peak), peaking at 235 GB/s (86%) at 1024²
- **96% parallel efficiency on 6 MPI ranks** (5.8× speedup), with 2D domain decomposition and halo exchange
- A job API with Prometheus metrics and a Grafana dashboard, all running in Docker Compose

## The math

$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

Discretized with forward Euler in time and the 5-point Laplacian in space (FTCS):

```
u'[i,j] = u[i,j] + r * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j]),   r = α·dt/dx²
```

It's only stable for `r <= 0.25`, so everything defaults to `r = 0.24`. The boundary is Dirichlet (edge cells are held fixed). Everything is float32 because Metal has no float64.

`tests/test_solvers.py` checks every backend against NumPy, and also checks NumPy against an exact solution: `sin(πx)·sin(πy)` is an eigenmode of the discrete Laplacian, so it should decay by exactly `1 - 8r·sin²(π·dx/2)` per step.

## Backends

| backend | what it is |
|---|---|
| `numpy` | vectorized slicing. Single threaded, allocates temporaries every step |
| `numba` | `@njit(parallel=True)`, rows split across cores with `prange` |
| `mlx` | the same slicing written in MLX and wrapped in `mx.compile` |
| `metal-naive` | custom Metal kernel, one thread per cell, each thread reads its 5 neighbours from device memory |
| `metal-tiled` | each threadgroup loads a `(TY+2)×(TX+2)` tile (with a 1-cell halo) into `threadgroup` memory, barriers, then computes from the tile |

The Metal kernels are in `heatsim/solvers/metal_solver.py` and are launched through `mx.fast.metal_kernel`. Threadgroup memory is Metal's version of CUDA `__shared__` memory, and `threadgroup_barrier` is `__syncthreads()`. The tiled kernel pads the grid up to a multiple of the tile size so every threadgroup is full. Out-of-range threads still help load the tile and just skip the write, because returning early before the barrier would be a bug.

## Setup

```bash
make setup                 # creates .venv and installs everything
source .venv/bin/activate
make test
```

MPI comes from the `mpich` pip package, which puts `mpirun` inside the venv, so no Homebrew is needed.

## Running it

```bash
python -m heatsim backends
python -m heatsim run --backend metal-tiled --n 2048 --steps 5000 --png results/out.png
python -m heatsim run --n 512 --steps 20000 --init edge --gif results/edge.gif
```

Initial conditions: `hotspot` (Gaussian in the middle), `edge` (top edge held at 100), `rings`, `random`.

## Phase 1: single device

```bash
make bench          # all backends, 256² .. 8192², writes results/single.csv
make bench-tiles    # threadgroup shape sweep for the metal kernels
make plots          # results/plots/*.png
```

Effective bandwidth counts 8 bytes per cell per step (one float read, one float written). That's the minimum possible traffic, so it's a lower bound and the standard way to report stencil performance.

**Results (M4 Pro):**

| backend | 1024² ms/step | 4096² ms/step | 4096² GB/s | % of 273 GB/s | speedup vs NumPy (4096²) |
|---|---|---|---|---|---|
| numpy | 0.901 | 16.41 | 8 | 3% | 1× |
| numba | 0.135 | 0.95 | 141 | 52% | 17.2× |
| mlx | 0.105 | 2.07 | 65 | 24% | 7.9× |
| metal-naive | 0.036 | 0.64 | 209 | 76% | **25.5×** |
| metal-tiled | 0.048 | 0.65 | 207 | 76% | 25.3× |

The best number overall was metal-naive at 1024²: 235 GB/s, which is 86% of peak.

![speedup](results/plots/speedup.png)
![bandwidth](results/plots/bandwidth.png)

Things I noticed:

- **Tiling didn't beat the naive kernel.** At 1024² the tiled kernel is actually about 35% slower (0.048 vs 0.036 ms/step), and from 2048² up the two are within 2% of each other. Apple GPUs have a large cache that already catches the neighbour reuse tiling is meant to provide, so the tile load and the barrier are pure overhead. On NVIDIA hardware, where the CUDA version of this trick comes from, the result would be different.
- **The GPU kernels plateau around 200–210 GB/s** once the grid stops fitting in cache (2048² and up). That's about 75% of the advertised peak, which is a typical ceiling for a streaming kernel.
- **Numba gets surprisingly close at the largest size.** At 8192² the CPU version reaches 166 GB/s, against about 203 GB/s for the GPU kernels, so the GPU is only 1.2× faster there. CPU and GPU share the same unified memory, so for a memory-bound problem the GPU's advantage shrinks to however much more bandwidth it can pull than the CPU cores.
- **Small grids are all overhead.** At 256², multithreaded Numba is slower than single-threaded NumPy (0.091 vs 0.062 ms/step) because starting the threads costs more than the work itself.
- **`mlx` (high-level ops) plateaus at about 65 GB/s.** Even with `mx.compile`, slicing the interior and concatenating the boundary back on means extra passes over memory that the hand-written kernel doesn't do.

## Phase 2: MPI domain decomposition

`heatsim/mpi_solver.py` splits the global grid over a `px × py` Cartesian process grid (`MPI.Compute_dims`). Each rank keeps its block plus a 1-cell ghost layer. Every step:

1. **Halo exchange.** `Sendrecv` with the up/down/left/right neighbours. Rows are contiguous and sent straight from the array; columns are packed into buffers first. Ranks on the outer edge get `MPI.PROC_NULL` as a neighbour, so those calls are no-ops.
2. **Update.** A serial Numba kernel updates the owned cells, skipping the physical boundary.

```bash
mpirun -n 4 python -m heatsim.mpi_solver --n 2048 --steps 500 --verify   # checks against serial numpy
make scaling    # strong + weak sweep over 1..14 ranks, writes results/mpi.csv
make plots
```

Each rank is pinned to one thread (`NUMBA_NUM_THREADS=1`), so ranks equal cores.

- **Strong scaling:** fixed 4096² grid, more ranks.
- **Weak scaling:** 2048² cells per rank, so the global grid grows like `2048·√p`.

**Results** (4096² grid for strong scaling, 2048² cells per rank for weak; best of 3 sweeps):

| ranks | process grid | strong speedup | strong efficiency | weak efficiency | halo % (strong) |
|---|---|---|---|---|---|
| 1 | 1×1 | 1.00× | 100% | 100% | 0% |
| 2 | 2×1 | 1.97× | 98% | 86% | 1% |
| 4 | 2×2 | 3.34× | 83% | 90% | 16% |
| 6 | 3×2 | **5.76×** | **96%** | 73% | 4% |
| 8 | 4×2 | 4.44× | 56% | 58% | 43% |
| 10 | 5×2 | 4.50× | 45% | 68% | 52% |
| 12 | 4×3 | 6.47× | 54% | 46% | 42% |
| 14 | 7×2 | 6.26× | 45% | 44% | 50% |

![strong](results/plots/strong_scaling.png)
![weak](results/plots/weak_scaling.png)
![halo](results/plots/halo_fraction.png)

What the numbers say:

- **Scaling is close to ideal up to 6 ranks** (96% strong efficiency), then falls off. 14 ranks only reach 6.3×.
- **The overall ceiling is memory bandwidth, not communication.** Each rank streams through about 1260 MLUPS × 8 bytes ≈ 10 GB/s, and the CPU cores together top out somewhere around 150 GB/s (see Numba in Phase 1). Past about 6 ranks, adding more cores just splits the same bandwidth more ways.
- **"Halo %" is mostly waiting, not sending.** It's the time spent inside `Sendrecv`, which includes waiting for a slower neighbour to arrive. The messages are tiny (one 4096-float row is 16 KB), so the jump to 40–50% beyond 6 ranks is load imbalance: one slow rank holds everyone up at the next exchange.
- **The dips at 4 and 8–10 ranks are reproducible, not noise.** All three sweeps gave the same shape (4 ranks always about 0.80 s, 8 and 10 always slower than 6). My best guess is how macOS places processes on the M4 Pro's two 5-core performance clusters. Each cluster shares an L2 cache and a link to memory, and macOS doesn't let you pin processes to cores, so some layouts end up with an unbalanced cluster or a rank on an efficiency core. I haven't confirmed this. Confirming it would need a machine where ranks can be pinned to specific cores (e.g. Linux with `--bind-to core`).
- **On a real cluster** each node brings its own memory bandwidth and ranks can be pinned, so scaling across nodes should hold up much better than scaling across cores on one laptop chip. The code doesn't change, only the `mpirun` line.

## Phase 3: job service + monitoring

A FastAPI service that accepts simulation jobs, runs them in a process pool, and exposes Prometheus metrics. Grafana has a provisioned dashboard showing throughput, runtime percentiles, queue depth/wait, and estimated cost per run.

```bash
make up            # docker compose: api (:8000) + prometheus (:9090) + grafana (:3000)
make load          # fire random jobs at it
open http://localhost:3000
```

```bash
curl -X POST localhost:8000/jobs -H 'content-type: application/json' \
     -d '{"n": 1024, "steps": 5000, "init": "hotspot"}'
curl localhost:8000/jobs/<id>
open http://localhost:8000/jobs/<id>/image
```

API docs are at `http://localhost:8000/docs`.

**Metal inside Docker:** Docker on macOS runs a Linux VM with no access to the GPU, so the containerized API falls back to the `numba` backend. To serve GPU jobs, run the API natively with `make serve` (port 8001). The Prometheus config already scrapes `host.docker.internal:8001`, so native jobs show up on the same dashboard under `job="heatsim-native"`. Try `python scripts/load_test.py --url http://localhost:8001` and compare the two.

**Cost per run:** running locally is free, so the service multiplies runtime by `HEATSIM_COST_PER_HOUR` (default $0.53/h, about the price of a small on-demand cloud GPU instance) to show what the same workload would cost if it were rented.

| metric | type |
|---|---|
| `heatsim_jobs_submitted_total{backend}` | counter |
| `heatsim_jobs_finished_total{backend,status}` | counter |
| `heatsim_jobs_queued`, `heatsim_jobs_running` | gauge |
| `heatsim_job_runtime_seconds{backend}` | histogram |
| `heatsim_job_queue_wait_seconds` | histogram |
| `heatsim_cell_updates_total{backend}` | counter (rate gives MLUPS) |
| `heatsim_job_cost_dollars{backend}` | histogram |

## Layout

```
heatsim/
  grid.py            config + initial conditions
  solvers/           numpy, numba, mlx/metal backends behind one interface
  mpi_solver.py      2D domain decomposition + halo exchange
  __main__.py        CLI
benchmarks/          single-device, tile sweep, MPI scaling, plotting
service/             FastAPI app, job queue, prometheus metrics
monitoring/          prometheus + grafana provisioning and dashboard
scripts/             scaling sweep, load generator
tests/
```

## Ideas for later

- Temporal blocking: do several timesteps per kernel launch while the tile is in threadgroup memory.
- Overlap the halo exchange with the interior update (`Isend`/`Irecv`, update the interior, wait, then update the edges).
- Run the MPI version on a real cluster with Slurm.
- A CUDA version of the tiled kernel (CuPy `RawKernel`) to compare against an NVIDIA GPU.

## License

MIT, see [LICENSE](LICENSE).

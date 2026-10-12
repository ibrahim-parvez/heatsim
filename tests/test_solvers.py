import numpy as np
import pytest

from heatsim.grid import SimConfig, initial_condition
from heatsim.solvers import available_backends, get_solver

BACKENDS = available_backends()
FAST = [b for b in BACKENDS if b != "numpy"]


def reference(u0, steps, r=0.24):
    return get_solver("numpy", r).run(u0, steps)


@pytest.mark.parametrize("backend", FAST)
@pytest.mark.parametrize("shape", [(64, 64), (101, 77), (33, 130), (5, 5)])
def test_matches_numpy(backend, shape):
    # odd shapes on purpose: not multiples of the threadgroup/tile size
    rng = np.random.default_rng(1)
    u0 = (rng.random(shape) * 100).astype(np.float32)
    steps = 37  # odd so the ping-pong buffers end up swapped
    out = get_solver(backend, 0.24).run(u0, steps)
    np.testing.assert_allclose(out, reference(u0, steps), atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_boundary_stays_fixed(backend):
    u0 = initial_condition("edge", 48)
    out = get_solver(backend, 0.24).run(u0, 200)
    np.testing.assert_array_equal(out[0], u0[0])
    np.testing.assert_array_equal(out[-1], u0[-1])
    np.testing.assert_array_equal(out[:, 0], u0[:, 0])
    np.testing.assert_array_equal(out[:, -1], u0[:, -1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_heat_spreads_and_max_principle(backend):
    u0 = initial_condition("hotspot", 64)
    out = get_solver(backend, 0.24).run(u0, 500)
    # diffusion can't create new extremes
    assert out.max() <= u0.max() + 1e-4
    assert out.min() >= -1e-4
    # peak goes down, and heat leaks out through the cold boundary
    assert out.max() < u0.max()
    assert out.sum() < u0.sum()


def test_matches_analytic_mode():
    """sin(pi x) sin(pi y) is an eigenmode of the discrete laplacian, so it should
    decay by exactly the same factor every step."""
    n, steps, r = 65, 300, 0.2
    x = np.linspace(0, 1, n)
    u0 = (np.sin(np.pi * x)[:, None] * np.sin(np.pi * x)[None, :]).astype(np.float32)
    u0[0] = u0[-1] = u0[:, 0] = u0[:, -1] = 0

    dx = 1 / (n - 1)
    lam = 1 - 8 * r * np.sin(np.pi * dx / 2) ** 2
    expected = u0 * lam**steps

    out = get_solver("numpy", r).run(u0, steps)
    np.testing.assert_allclose(out, expected, atol=1e-4)


def test_split_steps_same_as_one_call():
    u0 = initial_condition("rings", 80)
    for backend in FAST:
        a = get_solver(backend, 0.24).run(u0, 60)
        s = get_solver(backend, 0.24)
        s.load(u0)
        for _ in range(6):
            s.advance(10)
        np.testing.assert_allclose(s.fetch(), a, atol=1e-5)


def test_initial_condition_blocks_match_full():
    for name in ("hotspot", "edge", "rings", "random"):
        full = initial_condition(name, 50)
        block = initial_condition(name, 50, rows=(10, 35), cols=(0, 22))
        np.testing.assert_array_equal(block, full[10:35, 0:22])


def test_config_rejects_unstable_r():
    with pytest.raises(ValueError):
        SimConfig(r=0.3)

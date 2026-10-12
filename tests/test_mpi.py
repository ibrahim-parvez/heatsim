import os
import shutil
import subprocess
import sys

import pytest

from heatsim.mpi_solver import split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")


def test_split_covers_everything():
    for n in (10, 17, 1000):
        for parts in (1, 3, 4, 7):
            chunks = [split(n, parts, i) for i in range(parts)]
            assert chunks[0][0] == 0
            assert chunks[-1][1] == n
            for (a0, a1), (b0, _) in zip(chunks, chunks[1:]):
                assert a1 == b0
            sizes = [b - a for a, b in chunks]
            assert max(sizes) - min(sizes) <= 1


@pytest.mark.skipif(MPIRUN is None, reason="no mpirun on PATH")
@pytest.mark.parametrize("ranks", [1, 2, 4, 6])
@pytest.mark.parametrize("init", ["random", "edge"])
def test_mpi_matches_serial(ranks, init):
    cmd = [MPIRUN, "-n", str(ranks), sys.executable, "-m", "heatsim.mpi_solver",
           "--n", "97", "--steps", "40", "--init", init, "--verify"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout, res.stdout

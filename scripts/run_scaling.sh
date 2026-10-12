#!/usr/bin/env bash
# Strong + weak scaling sweep for the MPI solver on one machine.
#
#   ./scripts/run_scaling.sh                 # defaults
#   RANKS="1 2 4 8" STRONG_N=8192 ./scripts/run_scaling.sh
#
# Each rank runs single threaded so "ranks" == "cores used". The M4 Pro has
# 10 performance + 4 efficiency cores, so expect a kink after 10 ranks.

set -euo pipefail
cd "$(dirname "$0")/.."

RANKS=${RANKS:-"1 2 4 6 8 10 12 14"}
STRONG_N=${STRONG_N:-4096}
WEAK_BASE=${WEAK_BASE:-2048}   # grid side for a single rank in the weak test
STEPS=${STEPS:-200}
OUT=${OUT:-results/mpi.csv}
MPIRUN=${MPIRUN:-mpirun}

export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

rm -f "$OUT"

echo "== strong scaling (n=$STRONG_N) =="
for p in $RANKS; do
    "$MPIRUN" -n "$p" python benchmarks/bench_mpi.py --mode strong --n "$STRONG_N" --steps "$STEPS" --out "$OUT"
done

echo "== weak scaling ($WEAK_BASE^2 cells per rank) =="
for p in $RANKS; do
    # keep cells per rank fixed: global side grows like sqrt(p)
    n=$(python -c "import math; print(int(round($WEAK_BASE * math.sqrt($p))))")
    "$MPIRUN" -n "$p" python benchmarks/bench_mpi.py --mode weak --n "$n" --steps "$STEPS" --out "$OUT"
done

echo "results in $OUT"

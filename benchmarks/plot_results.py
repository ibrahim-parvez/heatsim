"""Turn the benchmark CSVs into the charts used in the README.

    python benchmarks/plot_results.py            # whatever CSVs exist in results/
    python benchmarks/plot_results.py --peak-bw 273
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed color per backend so it stays the same across every chart
COLORS = {
    "numpy": "#2a78d6",
    "numba": "#eb6834",
    "mlx": "#1baf7a",
    "metal-naive": "#eda100",
    "metal-tiled": "#e87ba4",
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


def _finish(fig, ax, path, title, subtitle=None):
    ax.set_title(title, pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")


def _size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}²" for s in sizes])
    ax.set_xlabel("grid size")


def plot_single(df, outdir, peak_bw):
    sizes = sorted(df.n.unique())
    backends = [b for b in COLORS if b in df.backend.unique()]

    # throughput
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for b in backends:
        d = df[df.backend == b].sort_values("n")
        ax.plot(d.n, d.mlups, marker="o", color=COLORS[b], label=b)
    _size_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("million cell updates / s")
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    _finish(fig, ax, f"{outdir}/throughput.png", "Throughput by backend")

    # speedup over numpy
    if "numpy" in backends:
        base = df[df.backend == "numpy"].set_index("n").ms_per_step
        fig, ax = plt.subplots(figsize=(7, 4.2))
        best = (0, None, None)
        for b in backends:
            if b == "numpy":
                continue
            d = df[df.backend == b].set_index("n").ms_per_step
            sp = (base / d).dropna()
            ax.plot(sp.index, sp.values, marker="o", color=COLORS[b], label=b)
            if sp.max() > best[0]:
                best = (sp.max(), sp.idxmax(), b)
        # only label the best point, labelling every line gets messy
        ax.annotate(f"{best[0]:.0f}× ({best[2]})", (best[1], best[0]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
        _size_axis(ax, sizes)
        ax.set_yscale("log")
        ax.set_ylabel("speedup vs NumPy")
        ax.legend(loc="upper left", fontsize=9)
        _finish(fig, ax, f"{outdir}/speedup.png", "Speedup over NumPy")

    # effective bandwidth
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for b in backends:
        d = df[df.backend == b].sort_values("n")
        ax.plot(d.n, d.gbps, marker="o", color=COLORS[b], label=b)
    ax.axhline(peak_bw, color=INK_2, linestyle="--", linewidth=1)
    ax.text(sizes[0], peak_bw, f" peak {peak_bw:.0f} GB/s", va="bottom", fontsize=9, color=INK_2)
    _size_axis(ax, sizes)
    ax.set_ylim(0, peak_bw * 1.15)
    ax.set_ylabel("effective bandwidth (GB/s)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    _finish(fig, ax, f"{outdir}/bandwidth.png", "Effective memory bandwidth",
            "8 bytes per cell per step (1 read + 1 write)")


def plot_mpi(df, outdir):
    strong = df[df["mode"] == "strong"].sort_values("ranks")
    weak = df[df["mode"] == "weak"].sort_values("ranks")
    color = "#2a78d6"

    if len(strong):
        t1 = strong[strong.ranks == strong.ranks.min()].seconds.iloc[0] * strong.ranks.min()
        sp = t1 / strong.seconds
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(strong.ranks, strong.ranks, color=INK_2, linestyle="--", linewidth=1, label="ideal")
        ax.plot(strong.ranks, sp, marker="o", color=color, label="measured")
        ax.set_xlabel("MPI ranks (cores)")
        ax.set_ylabel("speedup")
        ax.legend(loc="upper left", fontsize=9)
        n = strong.n.iloc[0]
        _finish(fig, ax, f"{outdir}/strong_scaling.png", "Strong scaling", f"fixed {n}² grid")

        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(strong.ranks, sp / strong.ranks * 100, marker="o", color=color)
        ax.set_ylim(0, 110)
        ax.set_xlabel("MPI ranks (cores)")
        ax.set_ylabel("parallel efficiency (%)")
        _finish(fig, ax, f"{outdir}/strong_efficiency.png", "Strong scaling efficiency", f"fixed {n}² grid")

    if len(weak):
        t1 = weak[weak.ranks == weak.ranks.min()].seconds.iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.axhline(100, color=INK_2, linestyle="--", linewidth=1)
        ax.plot(weak.ranks, t1 / weak.seconds * 100, marker="o", color=color)
        ax.set_ylim(0, 110)
        ax.set_xlabel("MPI ranks (cores)")
        ax.set_ylabel("weak scaling efficiency (%)")
        cpr = int(weak.cells_per_rank.iloc[0] ** 0.5)
        _finish(fig, ax, f"{outdir}/weak_scaling.png", "Weak scaling", f"~{cpr}² cells per rank")

    # where the time goes
    if len(strong):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        halo = strong.halo_seconds / strong.seconds * 100
        ax.bar(strong.ranks.astype(str), halo, color=color, width=0.6)
        ax.set_xlabel("MPI ranks (cores)")
        ax.set_ylabel("% of runtime in halo exchange")
        ax.grid(axis="x", visible=False)
        _finish(fig, ax, f"{outdir}/halo_fraction.png", "Communication overhead", "strong scaling runs, slowest rank")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/plots")
    p.add_argument("--peak-bw", type=float, default=273.0, help="peak DRAM bandwidth in GB/s (M4 Pro = 273)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    single = os.path.join(args.results, "single.csv")
    mpi = os.path.join(args.results, "mpi.csv")
    if os.path.exists(single):
        plot_single(pd.read_csv(single), args.out, args.peak_bw)
    if os.path.exists(mpi):
        plot_mpi(pd.read_csv(mpi), args.out)
    if not (os.path.exists(single) or os.path.exists(mpi)):
        print("no results found, run the benchmarks first")


if __name__ == "__main__":
    main()

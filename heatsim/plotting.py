import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

CMAP = "inferno"


def save_heatmap(u, path, title=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(u, cmap=CMAP, origin="upper", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="temperature")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_animation(frames, path, fps=12, vmax=None):
    """frames is a list of (step, 2D array)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax = vmax if vmax is not None else max(float(f.max()) for _, f in frames)
    im = ax.imshow(frames[0][1], cmap=CMAP, vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    def update(k):
        step, f = frames[k]
        im.set_data(f)
        title.set_text(f"step {step}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

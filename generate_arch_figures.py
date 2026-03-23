"""
Generate FUSE architecture mechanism diagram as PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path("/home/claude/fuse-repo/figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0f1729",
    "axes.facecolor": "#0f1729",
    "text.color": "#e2e8f0",
    "font.family": "monospace",
    "font.size": 10,
})

C = {
    "blue": "#38bdf8", "blue_bg": "#0c4a6e",
    "green": "#34d399", "green_bg": "#064e3b",
    "orange": "#fb923c", "orange_bg": "#7c2d12",
    "purple": "#a78bfa",
    "gray": "#64748b", "gray_bg": "#1e293b",
    "text": "#e2e8f0", "muted": "#94a3b8",
    "bg": "#0f1729", "card": "#1a2332",
    "border": "#334155",
}


def draw_box(ax, x, y, w, h, color, label, sublabel=None, alpha=0.25):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, alpha=alpha,
                                    edgecolor=color, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.06 if sublabel else 0), label,
            ha="center", va="center", fontsize=11, fontweight="bold", color=C["text"])
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.08, sublabel,
                ha="center", va="center", fontsize=9, color=C["muted"])


def draw_arrow(ax, x1, y1, x2, y2, color=None):
    c = color or C["muted"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))


def fig_mechanism():
    """FUSE two-phase mechanism diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(2.5, 5.8, "FUSE: two-phase sparse FFN execution", ha="center",
            fontsize=15, fontweight="bold", color=C["text"])
    ax.text(2.5, 5.6, "y = W_down(SiLU(W_gate · x) ⊙ W_up · x)", ha="center",
            fontsize=10, color=C["muted"], family="monospace")

    # Input
    draw_box(ax, 1.7, 5.1, 1.1, 0.35, C["gray"], "Input x", "d_model", 0.3)

    # Phase 1 region
    phase1 = mpatches.FancyBboxPatch((0.3, 3.45), 4.4, 1.55, boxstyle="round,pad=0.03",
                                      facecolor=C["blue"], alpha=0.06,
                                      edgecolor=C["blue"], linewidth=0.8, linestyle="--")
    ax.add_patch(phase1)
    ax.text(0.5, 4.85, "Phase 1 — TRACE", fontsize=11, fontweight="bold", color=C["blue"])
    ax.text(1.45, 4.85, "(W_gate always in RAM — 33% of FFN)", fontsize=9, color=C["muted"])

    # W_gate
    draw_arrow(ax, 2.25, 5.1, 2.25, 4.65)
    draw_box(ax, 1.3, 4.15, 1.9, 0.45, C["blue"], "W_gate", "d_ffn × d_model", 0.2)

    # Top-K
    draw_arrow(ax, 2.25, 4.15, 2.25, 3.85)
    draw_box(ax, 1.0, 3.55, 2.5, 0.28, C["blue"], "SiLU → top-K select fired neurons S", alpha=0.15)

    # Phase 2 region
    phase2 = mpatches.FancyBboxPatch((0.3, 1.0), 4.4, 2.3, boxstyle="round,pad=0.03",
                                      facecolor=C["green"], alpha=0.06,
                                      edgecolor=C["green"], linewidth=0.8, linestyle="--")
    ax.add_patch(phase2)
    ax.text(0.5, 3.15, "Phase 2 — SPARSE COMPUTE", fontsize=11, fontweight="bold", color=C["green"])
    ax.text(2.2, 3.15, "(W_up, W_down streamed from storage — only fired rows)", fontsize=9, color=C["muted"])

    # W_up sparse
    draw_arrow(ax, 1.6, 3.45, 1.1, 2.95)
    draw_box(ax, 0.4, 2.5, 1.4, 0.4, C["green"], "W_up[S]", "fired rows only", 0.2)

    # W_down sparse
    draw_arrow(ax, 2.9, 3.45, 3.4, 2.95)
    draw_box(ax, 2.7, 2.5, 1.6, 0.4, C["green"], "W_down[:, S]", "fired cols only", 0.2)

    # Multiply
    draw_arrow(ax, 1.1, 2.5, 1.6, 2.1)
    draw_arrow(ax, 2.25, 3.45, 2.05, 2.2, C["muted"])
    draw_box(ax, 1.3, 1.8, 1.5, 0.35, "#1d9e75", "gate[S] ⊙ up[S]", alpha=0.2)

    # Down multiply
    draw_arrow(ax, 2.8, 1.97, 3.2, 1.97)
    draw_arrow(ax, 3.5, 2.5, 3.5, 1.75)
    draw_box(ax, 3.0, 1.6, 1.5, 0.35, "#1d9e75", "W_down @ hidden", alpha=0.2)

    # Output
    draw_arrow(ax, 3.2, 1.6, 2.6, 1.25)
    draw_box(ax, 1.7, 1.05, 1.1, 0.3, C["gray"], "Output y", alpha=0.3)

    # Savings
    ax.text(2.5, 0.6, "At 50% sparsity: read 50% of W_up + W_down",
            ha="center", fontsize=10, color=C["green"], fontweight="bold")
    ax.text(2.5, 0.4, "= skip ~10,000 neurons per layer on DeepSeek-R1 7B",
            ha="center", fontsize=9, color=C["muted"])
    ax.text(2.5, 0.2, "= ~3.8 GB of weight I/O saved per token across 28 layers",
            ha="center", fontsize=9, color=C["muted"])

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fuse_mechanism.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print("  ✓ fuse_mechanism.png")


def fig_device_deployment():
    """Device deployment: what FUSE unlocks."""
    fig, ax = plt.subplots(figsize=(10, 4))

    devices = ["Phone\n6-8GB", "RPi 5\n8GB", "Laptop\n16GB", "Workstation\n32GB", "IoT\n4GB"]
    without = [3, 3, 7, 13, 1]
    withfuse = [7, 7, 13, 70, 4]

    x = np.arange(len(devices))
    w = 0.32

    b1 = ax.bar(x - w/2, without, w, label="Without FUSE", color="#f87171", alpha=0.5)
    b2 = ax.bar(x + w/2, withfuse, w, label="With FUSE", color="#34d399", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(devices, fontsize=10, color=C["muted"])
    ax.set_ylabel("Largest model (B params)", color=C["muted"])
    ax.set_title("What FUSE unlocks: run bigger models on every device class",
                 fontsize=13, fontweight="bold", color=C["text"], loc="left")
    ax.legend(fontsize=10, framealpha=0.2, labelcolor=C["text"],
              facecolor=C["bg"], edgecolor=C["border"])
    ax.set_yscale("log")
    ax.set_ylim(0.8, 100)
    ax.set_yticks([1, 3, 7, 13, 70])
    ax.set_yticklabels(["1B", "3B", "7B", "13B", "70B"], color=C["muted"])
    ax.tick_params(colors=C["muted"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["border"])
    ax.spines["bottom"].set_color(C["border"])
    ax.grid(axis="y", color=C["border"], alpha=0.3)

    for b, v in zip(b2, withfuse):
        ax.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v}B",
                ha="center", fontsize=9, fontweight="bold", color="#34d399")

    fig.tight_layout()
    fig.savefig(OUT / "device_deployment.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print("  ✓ device_deployment.png")


if __name__ == "__main__":
    print("Generating architecture figures...")
    fig_mechanism()
    fig_device_deployment()
    print("Done.")

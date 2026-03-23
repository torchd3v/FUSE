"""
Generate FUSE visualization charts as PNG files for the paper and README.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("/home/claude/fuse-repo/figures")
OUT.mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1729",
    "axes.facecolor": "#0f1729",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#cbd5e1",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#1e293b",
    "grid.linewidth": 0.5,
    "font.family": "monospace",
    "font.size": 10,
})

C = {
    "accent": "#38bdf8",
    "green": "#34d399",
    "orange": "#fb923c",
    "purple": "#a78bfa",
    "pink": "#f472b6",
    "red": "#f87171",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "border": "#334155",
    "bg": "#0f1729",
    "card": "#1a2332",
}

MODELS = {
    "TinyLlama SwiGLU": {
        "color": C["accent"], "sparsity": 48.0, "io": 1.92, "d_ffn": 5632, "n": 22, "profile": "U-shape",
        "layers": [42.7,42.1,40.0,39.4,38.5,37.6,34.8,36.0,33.3,29.2,31.8,30.9,29.0,29.8,29.7,32.5,31.5,30.8,34.5,36.8,41.0,37.7],
    },
    "TinyLlama ReLU": {
        "color": C["green"], "sparsity": 75.4, "io": 4.06, "d_ffn": 5632, "n": 22, "profile": "Inverted",
        "layers": [78.1,74.8,72.1,72.1,78.0,79.3,78.3,79.2,80.1,82.4,79.3,78.9,78.5,79.7,77.1,74.7,75.6,72.2,71.9,70.5,65.3,60.2],
    },
    "DeepSeek-R1 7B": {
        "color": C["orange"], "sparsity": 53.3, "io": 2.14, "d_ffn": 18944, "n": 28, "profile": "Cliff",
        "layers": [79.2,91.3,80.9,32.0,63.6,75.7,46.6,43.7,42.2,53.2,41.0,42.6,43.2,48.5,45.0,51.0,43.2,48.2,47.5,43.6,49.2,50.7,54.1,61.1,52.8,58.4,47.3,56.6],
    },
    "Qwen3.5-4B": {
        "color": C["purple"], "sparsity": 41.2, "io": 1.70, "d_ffn": 9216, "n": 32, "profile": "Descending",
        "layers": [60.7,51.6,57.9,49.5,44.2,39.4,39.0,36.0,38.4,39.6,43.4,42.5,41.4,42.5,45.5,45.7,43.8,44.2,43.2,41.9,37.5,39.6,33.9,37.9,31.6,26.8,26.2,29.4,33.7,36.3,40.4,53.8],
    },
    "Qwen3.5-9B": {
        "color": C["pink"], "sparsity": 44.3, "io": 1.80, "d_ffn": 12288, "n": 32, "profile": "Descending",
        "layers": [66.6,58.0,57.3,53.6,48.8,40.0,41.8,38.4,39.8,41.4,44.8,42.7,42.1,43.0,46.1,46.9,45.9,44.8,46.2,45.4,42.1,43.5,38.4,42.5,36.0,33.8,35.4,36.7,36.6,40.7,44.5,54.1],
    },
}


def fig1_layer_profiles():
    """All 5 models' per-layer sparsity profiles in one figure."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), gridspec_kw={"hspace": 0.35})
    fig.suptitle("Per-Layer Sparsity Profiles", fontsize=16, fontweight="bold",
                 color=C["text"], y=0.98)

    for ax, (name, m) in zip(axes, MODELS.items()):
        layers = m["layers"]
        x = np.arange(len(layers))
        bars = ax.bar(x, layers, color=m["color"], alpha=0.75, width=0.8,
                      edgecolor=m["color"], linewidth=0.3)

        # Highlight min/max
        imin = int(np.argmin(layers))
        imax = int(np.argmax(layers))
        bars[imin].set_edgecolor(C["red"])
        bars[imin].set_linewidth(2)
        bars[imax].set_edgecolor("#ffffff")
        bars[imax].set_linewidth(2)

        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, len(layers) - 0.4)
        ax.set_ylabel("Sparsity %", fontsize=9)
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_xticks(x[::max(1, len(x)//16)])
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

        # Title with stats
        avg = np.mean(layers)
        rng = max(layers) - min(layers)
        ax.set_title(
            f"{name}  —  avg {avg:.1f}%  •  range {rng:.1f}%  •  profile: {m['profile']}",
            fontsize=11, fontweight="bold", color=m["color"], loc="left", pad=8
        )

        # Annotate min/max
        ax.annotate(f"min {layers[imin]:.1f}%\n(L{imin})",
                    xy=(imin, layers[imin]), xytext=(imin + 2, max(8, layers[imin] - 12)),
                    fontsize=7, color=C["red"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.8))
        ax.annotate(f"max {layers[imax]:.1f}%\n(L{imax})",
                    xy=(imax, layers[imax]), xytext=(imax + 2, min(95, layers[imax] + 5)),
                    fontsize=7, color="#ffffff", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#ffffff", lw=0.8))

    fig.savefig(OUT / "layer_profiles.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ layer_profiles.png")


def fig2_sparsity_comparison():
    """Bar chart comparing avg sparsity and I/O reduction across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("FUSE Results Across Five Models", fontsize=14, fontweight="bold",
                 color=C["text"], y=1.02)

    names = list(MODELS.keys())
    colors = [MODELS[n]["color"] for n in names]
    sparsities = [MODELS[n]["sparsity"] for n in names]
    ios = [MODELS[n]["io"] for n in names]

    # Sparsity bars
    bars = ax1.barh(names, sparsities, color=colors, alpha=0.8, height=0.6,
                    edgecolor=[c for c in colors], linewidth=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Average Sparsity (%)")
    ax1.set_title("Average Sparsity", fontsize=12, fontweight="bold", color=C["text"], loc="left")
    ax1.grid(axis="x", alpha=0.3)
    ax1.tick_params(labelsize=9)
    ax1.invert_yaxis()
    for bar, val in zip(bars, sparsities):
        ax1.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=10, fontweight="bold", color=C["text"])

    # I/O reduction bars
    bars2 = ax2.barh(names, ios, color=colors, alpha=0.8, height=0.6,
                     edgecolor=[c for c in colors], linewidth=0.5)
    ax2.set_xlim(0, 5)
    ax2.set_xlabel("I/O Reduction (×)")
    ax2.set_title("I/O Reduction", fontsize=12, fontweight="bold", color=C["text"], loc="left")
    ax2.grid(axis="x", alpha=0.3)
    ax2.tick_params(labelsize=9)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, ios):
        ax2.text(val + 0.08, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}×", va="center", fontsize=10, fontweight="bold", color=C["text"])

    fig.tight_layout()
    fig.savefig(OUT / "sparsity_comparison.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ sparsity_comparison.png")


def fig3_neurons_fired():
    """Stacked bar: neurons fired vs skipped per model."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    names = list(MODELS.keys())
    fired = []
    skipped = []
    colors = []
    for n in names:
        m = MODELS[n]
        f = int(m["d_ffn"] * (1 - m["sparsity"] / 100))
        s = m["d_ffn"] - f
        fired.append(f)
        skipped.append(s)
        colors.append(m["color"])

    x = np.arange(len(names))
    w = 0.55

    b1 = ax.bar(x, fired, w, label="Neurons Fired", color=colors, alpha=0.85)
    b2 = ax.bar(x, skipped, w, bottom=fired, label="Neurons Skipped",
                color=C["muted"], alpha=0.45, edgecolor=C["border"], linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Neurons per Layer")
    ax.set_title("Neurons Fired vs Skipped per Layer",
                 fontsize=13, fontweight="bold", color=C["text"], loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=9)

    # Annotate totals
    for i, (f, s, n) in enumerate(zip(fired, skipped, names)):
        pct = s / (f + s) * 100
        ax.text(i, f + s + 200, f"{s:,} skipped\n({pct:.0f}%)",
                ha="center", fontsize=8, color=C["text"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "neurons_fired.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ neurons_fired.png")


def fig4_memory_savings():
    """Model size vs FUSE RAM requirement + device targets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: model memory
    models = ["LLaMA-7B", "LLaMA-13B", "LLaMA-70B"]
    full = [14, 26, 140]
    fuse = [5, 10, 35]
    x = np.arange(len(models))
    w = 0.32

    b1 = ax1.bar(x - w/2, full, w, label="Full Model (FP16)", color=C["red"], alpha=0.65)
    b2 = ax1.bar(x + w/2, fuse, w, label="FUSE RAM Needed", color=C["green"], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("Memory (GB)")
    ax1.set_title("Full Model vs FUSE RAM", fontsize=13, fontweight="bold",
                   color=C["text"], loc="left")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(axis="y", alpha=0.3)

    for b, v in zip(b1, full):
        ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v}GB",
                 ha="center", fontsize=9, fontweight="bold", color=C["red"])
    for b, v in zip(b2, fuse):
        ax1.text(b.get_x() + b.get_width()/2, v + 2, f"{v}GB",
                 ha="center", fontsize=9, fontweight="bold", color=C["green"])

    # Right: device targets
    devices = ["Phone\n(6-8GB)", "RPi 5\n(8GB)", "Laptop\n(16GB)", "Workstation\n(32GB)", "IoT\n(4GB)"]
    without = [3, 3, 7, 13, 1]
    withfuse = [7, 7, 13, 70, 4]

    x2 = np.arange(len(devices))
    b3 = ax2.bar(x2 - w/2, without, w, label="Without FUSE", color=C["red"], alpha=0.5)
    b4 = ax2.bar(x2 + w/2, withfuse, w, label="With FUSE", color=C["green"], alpha=0.85)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(devices, fontsize=9)
    ax2.set_ylabel("Largest Model (B params)")
    ax2.set_title("Device Deployment: Models Unlocked by FUSE",
                   fontsize=13, fontweight="bold", color=C["text"], loc="left")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, 100)
    ax2.set_yticks([1, 3, 7, 13, 70])
    ax2.set_yticklabels(["1B", "3B", "7B", "13B", "70B"])

    for b, v in zip(b4, withfuse):
        ax2.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v}B",
                 ha="center", fontsize=8, fontweight="bold", color=C["green"])

    fig.tight_layout()
    fig.savefig(OUT / "memory_savings.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ memory_savings.png")


if __name__ == "__main__":
    print("Generating FUSE figures...")
    fig1_layer_profiles()
    fig2_sparsity_comparison()
    fig3_neurons_fired()
    fig4_memory_savings()
    print(f"Done — {len(list(OUT.glob('*.png')))} figures in {OUT}/")

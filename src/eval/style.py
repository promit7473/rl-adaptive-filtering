"""Publication-grade plotting style — IEEE conference quality.

Aesthetic ported from regolith_entrapment_research with extensions for
adaptive-filtering plots: heatmaps, performance profiles, IQM bars.

Use:
    from src.eval.style import apply_style, color_for, ...
    apply_style()
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker


# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    # method-tier colours (perceptually distinct, color-blind friendly Brewer)
    "blue":         "#2166AC",
    "red":          "#D6604D",
    "green":        "#4DAC26",
    "purple":       "#8073AC",
    "orange":       "#E08214",
    "teal":         "#1B9E77",
    "yellow":       "#E6AB02",
    "magenta":      "#A6761D",
    # neutrals
    "gray":         "#666666",
    "lightgray":    "#BBBBBB",
    "black":        "#222222",
    # ui
    "background_light": "#FAFAFA",
    "grid":         "#DDDDDD",
    "axis":         "#444444",
    "text_dark":    "#222222",
}

PALETTE_ORDER = [PALETTE[k] for k in
                 ("blue", "red", "green", "purple", "orange", "teal", "yellow", "magenta")]

# semantic colour for each method — keep stable across all figures
METHOD_COLORS = {
    "LMS":            PALETTE["gray"],
    "NLMS":           PALETTE["orange"],
    "VSS-LMS":        PALETTE["green"],
    "RLS":            PALETTE["purple"],
    "CNN":            PALETTE["red"],
    "Meta-RL":        "#888888",         # de-emphasized — negative result
    "PPO-MLP":        PALETTE["blue"],
    "PPO-MLP (ours)": PALETTE["blue"],
    "RL":             PALETTE["blue"],
}

# canonical figure sizes (IEEE column = 3.5 in, two-column = 7.16 in)
FIG_SINGLE = (3.5, 2.6)         # one-column standard
FIG_SINGLE_TALL = (3.5, 4.2)    # one-column tall
FIG_DOUBLE = (7.16, 3.0)        # two-column standard
FIG_DOUBLE_TALL = (7.16, 4.2)   # two-column tall
FIG_DOUBLE_WIDE = (7.16, 2.6)   # two-column flat
FIG_HEATMAP = (7.16, 3.4)       # heatmap default


def apply_style() -> None:
    """Apply IEEE-publication quality matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       PALETTE["axis"],
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           PALETTE["grid"],
        "grid.linestyle":       "--",
        "grid.linewidth":       0.5,
        "grid.alpha":           0.7,
        "font.family":          "DejaVu Sans",
        "font.size":            9,
        "axes.titlesize":       10,
        "axes.titleweight":     "bold",
        "axes.titlepad":        6,
        "axes.labelsize":       9,
        "axes.labelcolor":      PALETTE["text_dark"],
        "axes.labelweight":     "normal",
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.color":          PALETTE["axis"],
        "ytick.color":          PALETTE["axis"],
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3.5,
        "ytick.major.size":     3.5,
        "legend.fontsize":      8,
        "legend.framealpha":    0.92,
        "legend.edgecolor":     PALETTE["lightgray"],
        "legend.fancybox":      False,
        "legend.borderpad":     0.4,
        "legend.handlelength":  1.6,
        "lines.linewidth":      1.6,
        "lines.solid_capstyle": "round",
        "patch.linewidth":      0.6,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.facecolor":    "white",
        "savefig.transparent":  False,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def remove_spines(ax, sides=("top", "right")) -> None:
    for s in sides:
        ax.spines[s].set_visible(False)


def color_for(method: str, fallback_idx: int = 0) -> str:
    return METHOD_COLORS.get(method, PALETTE_ORDER[fallback_idx % len(PALETTE_ORDER)])


def smooth_box(y, w: int = 20):
    """Boxcar smoother (no padding). Returns (smoothed, valid_offset)."""
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y, 0
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid"), w - 1


def shade_band(ax, x, y, color, *, w: int = 20, alpha_band: float = 0.18,
               alpha_raw: float = 0.0, label: str | None = None,
               smooth_lw: float = 1.8, zorder: int = 3) -> None:
    """Smoothed mean + ±1σ rolling band."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if alpha_raw > 0:
        ax.plot(x, y, color=color, alpha=alpha_raw, lw=0.6, zorder=zorder - 1)
    if w > 1 and len(y) > w:
        k = np.ones(w) / w
        mu = np.convolve(y, k, mode="valid")
        sq = np.convolve(y * y, k, mode="valid")
        sig = np.sqrt(np.maximum(sq - mu * mu, 0))
        xs = x[w - 1:]
        ax.fill_between(xs, mu - sig, mu + sig, color=color,
                        alpha=alpha_band, lw=0, zorder=zorder)
        ax.plot(xs, mu, color=color, lw=smooth_lw, label=label, zorder=zorder + 1)
    else:
        ax.plot(x, y, color=color, lw=smooth_lw, label=label, zorder=zorder)


def k_formatter():
    return ticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6
        else (f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}")
    )


def heatmap_text_color(value: float, vmin: float, vmax: float,
                       cmap_name: str = "viridis_r") -> str:
    """Choose black or white text based on cmap luminance at the cell value."""
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap(cmap_name)
    t = (value - vmin) / max(vmax - vmin, 1e-9)
    t = float(np.clip(t, 0.0, 1.0))
    rgba = cmap(t)
    # ITU-R BT.709 luminance
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if lum < 0.55 else "black"


def annotate_heatmap(ax, matrix, fmt: str = ".1f", cmap_name: str = "viridis_r",
                     fontsize: float = 7.5):
    """Per-cell value labels with luminance-aware contrast."""
    vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:{fmt}}",
                    color=heatmap_text_color(v, vmin, vmax, cmap_name),
                    ha="center", va="center", fontsize=fontsize)


def add_train_ood_divider(ax, n_train: int, *, axis: str = "x",
                          label_train: str = "Train (in-dist.)",
                          label_ood: str = "OOD (held-out)",
                          y_offset: float = -1.2):
    """Draw a thick dashed line and Train/OOD bracket labels above a heatmap."""
    if axis == "x":
        ax.axvline(n_train - 0.5, color="black", lw=1.8, ls="--", alpha=0.85)
        ax.text((n_train - 1) / 2, y_offset, label_train,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE["blue"])
        n_total = ax.get_xlim()[1] + 0.5
        ax.text((n_train + n_total - 1) / 2, y_offset, label_ood,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE["red"])

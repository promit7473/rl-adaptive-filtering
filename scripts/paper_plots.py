"""Single style source for the IEEE SPL paper figures.

Every paper figure (``fig_recovery``, ``fig_ablation``, ``fig_realworld``,
``fig_convergence``) imports :func:`apply_style` from here so they share ONE
consistent look:

* serif body type (Nimbus Roman -- Times-metric, matches the IEEEtran body),
* STIX math (serif-compatible),
* a single deep-red ``OURS`` colour for our method in every figure,
* thin, B&W-safe hatches as a second greyscale channel.

Keeping the style in one module is what guarantees the side-by-side figures
(zero-shot ECG + ablation) and the recovery panel render identically -- do not
re-introduce the DejaVu-Sans ``src/eval/style.py`` for paper figures.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── colour convention ───────────────────────────────────────────────────────
# Our method is the same deep red in EVERY figure (recovery, ablation, ECG).
OURS = "#D32F2F"

# Baseline / comparison colours -- deep but lightly-vibrant, B&W-safe once the
# thin hatches are added on top.
COL = {
    "NLMS":            "#2196F3",   # blue
    "NLMS (mu=0.1)":   "#90CAF9",   # light blue
    "RLS":             "#9C27B0",   # purple
    "VSS":             "#FF9800",   # orange
    "VSS-LMS":         "#FF9800",
    "VSS-LMS (Kwong)": "#FF9800",
    "Heuristic":       "#4CAF50",   # green
    "Meta-AF":         "#7E57C2",
    "gray":            "#777777",
}

# Edge colour drawn around hatched/filled bars.
HATCH_EDGE = "black"


def apply_style() -> None:
    """Apply the shared IEEE-SPL serif figure style (call once per figure)."""
    plt.rcParams.update({
        # canvas
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "axes.edgecolor":       "#333333",
        "axes.linewidth":       0.8,
        "axes.axisbelow":       True,
        # fonts: serif body + serif-compatible math, matches IEEEtran 8 pt
        "font.family":          "serif",
        "font.serif":           ["Nimbus Roman", "Times New Roman", "Times",
                                 "DejaVu Serif"],
        "mathtext.fontset":     "stix",
        "font.size":            8,
        "axes.titlesize":       8.6,
        "axes.titleweight":     "bold",
        "axes.labelsize":       8,
        "xtick.labelsize":      7,
        "ytick.labelsize":      7,
        "legend.fontsize":      6.2,
        # lines / patches
        "lines.linewidth":      1.4,
        "lines.solid_capstyle": "round",
        "patch.linewidth":      0.5,
        "hatch.linewidth":      0.45,
        # grid (figures enable per-axes, but keep defaults coherent)
        "grid.color":           "#E2E2E2",
        "grid.linewidth":       0.45,
        # output
        "savefig.dpi":          300,
        "savefig.facecolor":    "white",
        "savefig.transparent":  False,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def _despine(ax, sides=("top", "right")) -> None:
    """Hide the requested spines (default: top + right) for a clean look."""
    for s in sides:
        ax.spines[s].set_visible(False)


def styled_bars(ax, x, heights, width, *, color, hatch="", label=None,
                yerr=None, capsize=2, zorder=3):
    """Draw a bar group with the shared edge / hatch / error-bar styling."""
    return ax.bar(
        x, heights, width,
        color=color, hatch=hatch, label=label,
        edgecolor=HATCH_EDGE, linewidth=0.5, zorder=zorder,
        yerr=yerr, capsize=capsize, error_kw=dict(elinewidth=0.7),
    )

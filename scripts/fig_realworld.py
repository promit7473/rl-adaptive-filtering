"""Fig. 3 -- zero-shot ECG transfer (RL^2 Meta-RL policy), single panel.

Mirrors the house style of fig_ablation.py (shared scripts/paper_plots) so the
two figures sit balanced side-by-side on the page: same figsize, serif body,
deep-red 'ours', framed panel, bold title + italic subtitle.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from scripts.paper_plots import apply_style, OURS, HATCH_EDGE


def build_realworld_plot():
    apply_style()

    d = pd.read_csv(os.path.join(ROOT, "results", "realworld_ecg_multi.csv"))

    order_noise = ["gaussian", "impulsive", "burst", "regime_switch",
                   "powerline", "baseline_wander"]
    label_noise = ["Gaussian", "Impulsive", "Burst", "Reg.-Switch",
                   "Powerline\n(50 Hz)", "Baseline\nWander"]

    # method order: classical baselines first, our policy last (deep red)
    methods = ["NLMS", "NLMS (mu=0.1)", "VSS-LMS (Kwong)", "Heuristic", "Meta-RL"]
    m_labels = [r"NLMS ($\mu$=0.5)", r"NLMS ($\mu$=0.1)", "VSS-LMS (Kwong)",
                "Heuristic", "Meta-RL (ours)"]
    colors = ["#2196F3", "#90CAF9", "#FF9800", "#4CAF50", OURS]
    hatches = ["", "//", "xx", "..", ""]

    stats = (d.groupby(["noise", "method"])["ss_mse_db"]
               .agg(["mean", "std"]).reset_index())

    def get(noise, method, col):
        sel = stats[(stats.noise == noise) & (stats.method == method)]
        return sel[col].values[0]

    # match the ablation figure exactly: same width/height -> balanced pair
    fig, ax = plt.subplots(figsize=(3.45, 2.95))

    x = np.arange(len(order_noise))
    width = 0.15

    for i, (m, lbl, c, h) in enumerate(zip(methods, m_labels, colors, hatches)):
        means = [get(n, m, "mean") for n in order_noise]
        stds = [get(n, m, "std") for n in order_noise]
        ax.bar(x + (i - 2) * width, means, width, label=lbl,
               color=c, hatch=h, edgecolor=HATCH_EDGE, linewidth=0.5, zorder=3,
               yerr=stds, capsize=1.5, error_kw=dict(elinewidth=0.6))

    # ── 6.7 dB powerline win: clean callout in the empty bottom of the column ──
    pl_meta = get("powerline", "Meta-RL", "mean")     # -27.1
    pl_nlms = get("powerline", "NLMS", "mean")        # -20.4
    gap = pl_nlms - pl_meta
    x_meta = 4 + 2 * width                            # Meta-RL powerline bar
    ax.annotate(rf"Meta-RL: $+{gap:.1f}$ dB", xy=(x_meta, pl_meta),
                xytext=(x_meta - 0.02, -38.5),
                color=OURS, fontsize=6.0, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=OURS, lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.9,
                                shrinkA=3, shrinkB=2))

    ax.set_xticks(x)
    ax.set_xticklabels(label_noise, fontsize=7, fontweight="bold",
                       rotation=35, ha="right")
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylabel("Steady-state MSE (dB)", fontsize=8)
    ax.set_ylim(-58, 2)
    ax.axhline(0, color="black", lw=0.5)

    # framed "panel" look + bold title + italic subtitle (mirrors ablation)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("Zero-Shot ECG Transfer", fontweight="bold",
                 fontsize=8.6, pad=16)
    ax.text(0.5, 1.012, "synthetic-trained policy, deployed on real MIT-BIH",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.8, style="italic", color="#333333")
    ax.grid(True, axis="y", color="#E2E2E2", lw=0.45, ls=(0, (3, 3)))

    # legend in the clear lower-left region (gaussian/impulsive columns only
    # reach ~-28 dB, so the space below is empty); single column keeps it
    # narrow enough to clear the deep burst/regime-switch bars
    ax.legend(loc="lower left", ncol=1, frameon=True, fontsize=5.8,
              labelspacing=0.32, handletextpad=0.4, handlelength=1.1,
              borderpad=0.4, framealpha=0.96)

    fig.tight_layout(pad=0.3)

    out_dir = os.path.join(ROOT, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_realworld.pdf"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_realworld.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    print("Zero-shot ECG figure saved successfully!")


if __name__ == "__main__":
    build_realworld_plot()

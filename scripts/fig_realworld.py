"""Regenerate fig_realworld as ECG-only single-panel zero-shot transfer plot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.eval.style import apply_style

apply_style()

d = pd.read_csv("results/realworld_ecg_multi.csv")
order_noise = ["gaussian", "impulsive", "burst", "regime_switch", "powerline", "baseline_wander"]
label_noise = ["Gaussian", "Impulsive", "Burst", "Reg.-Switch", "Powerline\n(50 Hz)\u2605", "Baseline\nWander"]
methods = ["NLMS", "NLMS (mu=0.1)", "VSS-LMS (Kwong)", "Heuristic", "Meta-RL"]
m_labels = [r"NLMS ($\mu$=0.5)", r"NLMS ($\mu$=0.1)", "VSS-LMS\n(Kwong)", "Heuristic", "Meta-RL \u2605"]
colors = ["#4C9BD1", "#A6CFE8", "#F4B860", "#7FBF7F", "#E07A3D"]
hatches = ["", "//", "xx", "..", ""]

stats = d.groupby(["noise", "method"])["ss_mse_db"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(6.5, 3.4))
x = np.arange(len(order_noise))
w = 0.16
for i, (m, lbl, c, h) in enumerate(zip(methods, m_labels, colors, hatches)):
    means = [stats[(stats.noise == n) & (stats.method == m)]["mean"].values[0] for n in order_noise]
    stds = [stats[(stats.noise == n) & (stats.method == m)]["std"].values[0] for n in order_noise]
    offs = (i - 2) * w
    ax.bar(x + offs, means, w, yerr=stds, label=lbl, color=c, hatch=h,
           edgecolor="black", linewidth=0.5, capsize=2,
           error_kw=dict(elinewidth=0.7))

# annotate powerline win for Meta-RL
pl_meta = stats[(stats.noise == "powerline") & (stats.method == "Meta-RL")]["mean"].values[0]
pl_nlms = stats[(stats.noise == "powerline") & (stats.method == "NLMS")]["mean"].values[0]
gap = pl_nlms - pl_meta
ax.annotate(f"{gap:.1f} dB", xy=(4 + 2 * w, pl_meta), xytext=(4 + 2.4 * w, pl_meta + 5),
            color="#C0392B", fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=0.8))

ax.set_xticks(x)
ax.set_xticklabels(label_noise, fontsize=8)
ax.set_ylabel("Steady-state MSE (dB)")
ax.set_title("Zero-shot transfer to ECG (MIT-BIH) \u2014 policy trained on synthetic only",
             fontsize=9, fontweight="bold")
ax.axhline(0, color="black", lw=0.5)
ax.legend(loc="lower right", fontsize=7, ncol=3, frameon=True)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("paper/figures/fig_realworld.pdf", bbox_inches="tight")
plt.savefig("paper/figures/fig_realworld.png", bbox_inches="tight", dpi=200)
print("Saved fig_realworld.pdf/.png")

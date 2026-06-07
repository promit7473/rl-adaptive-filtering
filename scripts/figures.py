"""Regenerate paper figures from results CSVs.

Inputs (under --eval-dir, default results/v2_eval/):
  synthetic.csv, ecg.csv

Outputs (under --fig-dir, default paper/figures/):
  fig_synthetic.pdf  grouped bar plot of SS-MSE by method x family
  fig_ecg.pdf        ECG noise-type x method panel (mean +/- 95% CI)
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.eval.style import apply_style, remove_spines, PALETTE, PALETTE_ORDER

# ── Modern color palette & hatch patterns ─────────────────────────────────────
METHOD_COLORS = {
    "Fixed-Leaky":  "#888888",
    "Heuristic":    "#AAAAAA",
    "NLMS":         PALETTE["orange"],
    "RLS":          PALETTE["purple"],
    "PID-NLMS":     PALETTE["green"],
    "VSS-Kwong":    PALETTE["teal"],
    "IIR-Notch (50Hz)": PALETTE["yellow"],
    "Meta-RL":      PALETTE["blue"],
    "PPO-MLP":      PALETTE["red"],
}
METHOD_HATCHES = {
    "Fixed-Leaky":  "",
    "Heuristic":    "//",
    "NLMS":         "",
    "RLS":          "xx",
    "PID-NLMS":     "..",
    "VSS-Kwong":    "\\\\",
    "IIR-Notch (50Hz)": "||",
    "Meta-RL":      "",
    "PPO-MLP":      "",
}

# Pretty family names for x-axis
FAMILY_LABELS = {
    "alpha_stable":     "α-Stable",
    "burst":            "Burst",
    "chirp_interferer": "Chirp",
    "colored":          "Colored",
    "gaussian":         "Gaussian",
    "impulsive":        "Impulsive",
    "regime_switch":    "Regime-Sw.",
    "time_varying":     "TV-SNR",
}


def _bootstrap_ci(x, n=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fig_synthetic(csv_path, out_path):
    apply_style()
    df = pd.read_csv(csv_path)

    # Filter out divergent methods (inf / huge values)
    df = df[df["ss_mse_db"].abs() < 200]

    methods = sorted(df["method"].unique())
    families = sorted(df["family"].unique())

    fig, ax = plt.subplots(figsize=(7.16, 3.4))
    width = 0.75 / len(methods)
    x = np.arange(len(families))

    for i, m in enumerate(methods):
        means, errs_lo, errs_hi = [], [], []
        for f in families:
            sub = df[(df["method"] == m) & (df["family"] == f)]["ss_mse_db"].values
            mu = float(np.mean(sub)) if len(sub) else np.nan
            lo, hi = _bootstrap_ci(sub)
            means.append(mu)
            errs_lo.append(mu - lo)
            errs_hi.append(hi - mu)

        color = METHOD_COLORS.get(m, PALETTE_ORDER[i % len(PALETTE_ORDER)])
        hatch = METHOD_HATCHES.get(m, "")

        ax.bar(x + i * width, means, width, label=m,
               color=color, edgecolor="black", linewidth=0.5,
               hatch=hatch, yerr=[errs_lo, errs_hi],
               capsize=1.5, error_kw={"elinewidth": 0.6, "capthick": 0.6})

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(
        [FAMILY_LABELS.get(f, f) for f in families],
        rotation=30, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("SS MSE (dB)")
    ax.set_title("Steady-State MSE by Noise Family", fontweight="bold")
    remove_spines(ax)
    ax.legend(fontsize=6.5, loc="lower left", ncol=3, frameon=True,
              borderpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_ecg(csv_path, out_path):
    apply_style()
    df = pd.read_csv(csv_path)
    noises = sorted(df["noise"].unique())
    methods = sorted(df["method"].unique())

    fig, axes = plt.subplots(1, len(noises), figsize=(3.2 * len(noises), 3.5),
                             sharey=True)
    if len(noises) == 1:
        axes = [axes]

    width = 0.75 / len(methods)
    for ax, noise in zip(axes, noises):
        snrs = sorted(df[df["noise"] == noise]["snr_db"].unique())
        x_pos = np.arange(len(snrs))
        for i, m in enumerate(methods):
            means, lo, hi = [], [], []
            for snr in snrs:
                sub = df[(df["method"] == m) & (df["noise"] == noise)
                         & (df["snr_db"] == snr)]["ss_mse_db"].values
                mu = float(np.mean(sub)) if len(sub) else np.nan
                l, h = _bootstrap_ci(sub)
                means.append(mu)
                lo.append(mu - l)
                hi.append(h - mu)

            color = METHOD_COLORS.get(m, PALETTE_ORDER[i % len(PALETTE_ORDER)])
            hatch = METHOD_HATCHES.get(m, "")

            ax.bar(x_pos + i * width, means, width, label=m,
                   color=color, edgecolor="black", linewidth=0.5,
                   hatch=hatch, yerr=[lo, hi],
                   capsize=1.2, error_kw={"elinewidth": 0.5, "capthick": 0.5})

        ax.set_xticks(x_pos + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{int(s)}" for s in snrs])
        ax.set_xlabel(f"{noise}\nSNR (dB)")
        remove_spines(ax)

    axes[0].set_ylabel("SS MSE (dB)")
    axes[-1].legend(fontsize=6, loc="lower left", frameon=True)
    fig.suptitle("ECG Denoising: SS MSE by Noise Type & SNR", fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", default="results/v2_eval")
    p.add_argument("--fig-dir", default="paper/figures")
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    syn = os.path.join(args.eval_dir, "synthetic.csv")
    ecg = os.path.join(args.eval_dir, "ecg.csv")
    if os.path.exists(syn):
        fig_synthetic(syn, os.path.join(args.fig_dir, "fig_synthetic.pdf"))
    if os.path.exists(ecg):
        fig_ecg(ecg, os.path.join(args.fig_dir, "fig_ecg.pdf"))


if __name__ == "__main__":
    main()

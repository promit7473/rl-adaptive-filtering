"""Regenerate paper figures from results CSVs.

Inputs (under --eval-dir, default results/v2_eval/):
  synthetic.csv, ecg.csv

Outputs (under --fig-dir, default paper/figures/):
  fig_synthetic.pdf  bar/box plot of SS-MSE by method x family
  fig_ecg.pdf        ECG noise-type x method panel (mean +/- 95% CI)
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _bootstrap_ci(x, n=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fig_synthetic(csv_path, out_path):
    df = pd.read_csv(csv_path)
    methods = sorted(df["method"].unique())
    families = sorted(df["family"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(methods)
    x = np.arange(len(families))
    for i, m in enumerate(methods):
        means = []
        errs_lo = []
        errs_hi = []
        for f in families:
            sub = df[(df["method"] == m) & (df["family"] == f)]["ss_mse_db"].values
            mu = float(np.mean(sub)) if len(sub) else np.nan
            lo, hi = _bootstrap_ci(sub)
            means.append(mu); errs_lo.append(mu - lo); errs_hi.append(hi - mu)
        ax.bar(x + i * width, means, width, label=m,
               yerr=[errs_lo, errs_hi], capsize=2)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(families, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("SS MSE (dB)")
    ax.legend(fontsize=6, loc="lower left", ncol=2, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_ecg(csv_path, out_path):
    df = pd.read_csv(csv_path)
    noises = sorted(df["noise"].unique())
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(3 * len(noises), 3.5),
                             sharey=True)
    if len(noises) == 1:
        axes = [axes]
    width = 0.8 / len(methods)
    for ax, noise in zip(axes, noises):
        snrs = sorted(df[df["noise"] == noise]["snr_db"].unique())
        x = np.arange(len(snrs))
        for i, m in enumerate(methods):
            means, lo, hi = [], [], []
            for snr in snrs:
                sub = df[(df["method"] == m) & (df["noise"] == noise)
                         & (df["snr_db"] == snr)]["ss_mse_db"].values
                mu = float(np.mean(sub)) if len(sub) else np.nan
                l, h = _bootstrap_ci(sub)
                means.append(mu); lo.append(mu - l); hi.append(h - mu)
            ax.bar(x + i * width, means, width, label=m,
                   yerr=[lo, hi], capsize=1.5)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"{int(s)}" for s in snrs])
        ax.set_xlabel(f"{noise}\nSNR (dB)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("SS MSE (dB)")
    axes[-1].legend(fontsize=6, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
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

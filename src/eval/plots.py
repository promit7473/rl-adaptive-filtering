"""Plotting utilities — uses publication style from src.eval.style."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from .style import apply_style, remove_spines, color_for, PALETTE_ORDER, k_formatter

apply_style()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_signal_comparison(clean, noisy, recovered_dict: dict, out_path: str,
                           title: str = "", n_show: int = 800):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    t = np.arange(min(n_show, len(clean)))
    axes[0].plot(t, clean[:len(t)], label="clean", lw=1.4, color="#222222")
    axes[0].plot(t, noisy[:len(t)], label="noisy", lw=0.7, alpha=0.55,
                 color=color_for("NLMS"))
    axes[0].set_ylabel("amplitude")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(title)
    remove_spines(axes[0])

    for i, (name, rec) in enumerate(recovered_dict.items()):
        axes[1].plot(t, rec[:len(t)], label=name, lw=1.0,
                     color=color_for(name, i), alpha=0.9)
    axes[1].plot(t, clean[:len(t)], label="clean", lw=1.4, color="#222222", alpha=0.85)
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("sample")
    axes[1].legend(loc="upper right", ncol=3)
    remove_spines(axes[1])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_error_curves(errors: dict, out_path: str, title: str = "", smooth: int = 50):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, (name, e) in enumerate(errors.items()):
        e2 = e ** 2
        if smooth > 1 and len(e2) > smooth:
            kernel = np.ones(smooth) / smooth
            e2 = np.convolve(e2, kernel, mode="valid")
        ax.plot(10 * np.log10(e2 + 1e-12), label=name, lw=1.4,
                color=color_for(name, i))
    ax.set_xlabel("sample")
    ax.set_ylabel("instantaneous MSE (dB)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    remove_spines(ax)
    ax.xaxis.set_major_formatter(k_formatter())
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_metric(values: dict, out_path: str, ylabel: str = "", title: str = "",
                    err: dict | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names = list(values.keys())
    vals = [values[k] for k in names]
    colors = [color_for(n, i) for i, n in enumerate(names)]
    yerr = [err[k] for k in names] if err else None
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8,
                  yerr=yerr, capsize=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    remove_spines(ax)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmap(matrix: np.ndarray, row_labels, col_labels, out_path: str,
                 title: str = "", cbar_label: str = "", fmt: str = ".1f",
                 cmap: str = "viridis", invert: bool = False):
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(col_labels), 1.0 + 0.55 * len(row_labels)))
    data = -matrix if invert else matrix
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:{fmt}}", ha="center", va="center",
                    color="white" if (data[i, j] - data.min()) / (data.max() - data.min() + 1e-9) > 0.5 else "black",
                    fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

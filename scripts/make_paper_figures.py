"""Generate all paper figures — IEEE SPL publication quality.

Palette:   Okabe-Ito (colorblind-safe, grayscale-distinguishable)
Widths:    3.5" (single-column) or 7.16" (double-column)
DPI:       600 (line art) / 300 (raster fallback)
Fonts:     embedded PDF Type-42 via pdf.fonttype=42
Style:     SciencePlots 'science' + manual IEEE overrides

Usage:
    python3 scripts/make_paper_figures.py
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scienceplots  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── IEEE column widths (inches) ───────────────────────────────────────────────
W1 = 3.5    # single-column
W2 = 7.16   # double-column

# ── Okabe-Ito colorblind-safe palette ────────────────────────────────────────
OI = {
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "blue":     "#0072B2",
    "vermilion":"#D55E00",
    "purple":   "#CC79A7",
    "yellow":   "#F0E442",
    "black":    "#000000",
}

# Semantic method→colour assignment (consistent across ALL figures)
METHOD_COLOR = {
    "LMS":                 OI["black"],
    "NLMS":                OI["blue"],
    "VSS-LMS (Kwong)":     OI["orange"],
    "VSS-LMS (Aboulnasr)": OI["vermilion"],
    "VSS-LMS (Mathews)":   OI["purple"],
    "Heuristic Scheduler": OI["green"],
    "RLS":                 OI["sky"],
    "PPO-MLP":             OI["yellow"],
    "Meta-RL":             OI["vermilion"],   # hero — vivid
    "CNN (supervised)":    OI["purple"],
    # ECG bar-chart names
    "NLMS (mu=0.1)":       OI["sky"],
    "Heuristic":           OI["green"],
}

# Combined line-style + marker (for B&W printing)
METHOD_LS = {
    "LMS":                 (":",  "s",  1.1),
    "NLMS":                ("-",  "o",  1.4),
    "VSS-LMS (Kwong)":     ("--", "^",  1.2),
    "Heuristic Scheduler": ("-.", "D",  1.2),
    "PPO-MLP":             ("--", "v",  1.8),
    "Meta-RL":             ("-",  "*",  2.2),
    "CNN (supervised)":    (":",  "P",  1.4),
}

# Hatching for bar charts (B&W accessible)
METHOD_HATCH = {
    "NLMS":             "",
    "NLMS (mu=0.1)":    "///",
    "VSS-LMS (Kwong)":  "xxx",
    "Heuristic":        "...",
    "Heuristic Scheduler": "...",
    "PPO-MLP":          "---",
    "Meta-RL":          "",
    "CNN (supervised)": "|||",
}


# ── Global style ──────────────────────────────────────────────────────────────
def _apply_style() -> None:
    plt.style.use(["science"])          # SciencePlots base
    plt.rcParams.update({
        # figure
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        # spines
        "axes.edgecolor":       "#333333",
        "axes.linewidth":       0.8,
        # grid — light, behind data
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.color":           "#DDDDDD",
        "grid.linestyle":       "--",
        "grid.linewidth":       0.45,
        "grid.alpha":           1.0,
        # font (SciencePlots uses LaTeX; we override to DejaVu for portability)
        "text.usetex":          False,
        "font.family":          "DejaVu Sans",
        "font.size":            8.0,
        "axes.titlesize":       9.0,
        "axes.titleweight":     "bold",
        "axes.titlepad":        4.5,
        "axes.labelsize":       8.0,
        "axes.labelcolor":      "#111111",
        "xtick.labelsize":      7.5,
        "ytick.labelsize":      7.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.major.pad":      2.5,
        "ytick.major.pad":      2.5,
        # lines
        "lines.linewidth":      1.4,
        "lines.markersize":     5.0,
        "lines.solid_capstyle": "round",
        # legend
        "legend.fontsize":      7.0,
        "legend.framealpha":    0.93,
        "legend.edgecolor":     "#BBBBBB",
        "legend.fancybox":      False,
        "legend.borderpad":     0.4,
        "legend.handlelength":  2.2,
        "legend.labelspacing":  0.28,
        "legend.columnspacing": 1.0,
        # save
        "savefig.dpi":          600,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
        "savefig.facecolor":    "white",
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"))
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=600)
    plt.close(fig)
    print(f"    ✓  {stem}.{{pdf,png}}")


# ── Data helpers ──────────────────────────────────────────────────────────────
def _merge_rl(df: pd.DataFrame) -> pd.DataFrame:
    """Rename per-seed PPO-NLMS-s* / Meta-RL-s* rows to canonical names."""
    out = []
    for prefix, name in [("PPO-NLMS", "PPO-MLP"), ("Meta-RL", "Meta-RL")]:
        tmp = df[df.method.str.startswith(prefix)].copy()
        tmp["method"] = name
        out.append(tmp)
    clean = df[~df.method.str.contains(r"-s\d+", regex=True)].copy()
    return pd.concat([clean] + out, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — SNR sweep  (single-column, 3.5 × 2.6 in)
# ─────────────────────────────────────────────────────────────────────────────
SNR_PTS = [0, 5, 10, 15, 20]

def plot_snr_sweep(full_csv: str, sweep_csv: str, out_dir: str) -> None:
    """SS MSE vs input SNR for Gaussian noise — key methods only.

    Data strategy:
      SNR ∈ {0,5,15,20} — from sweep_csv (single seed each, no duplicate)
      SNR = 10           — from full_csv  (5 seeds per method, authoritative)
    """
    df_full  = _merge_rl(pd.read_csv(full_csv))
    df_sweep = _merge_rl(pd.read_csv(sweep_csv))

    # restrict to Gaussian
    df_full  = df_full [df_full .family == "gaussian"].copy()
    df_sweep = df_sweep[df_sweep.family == "gaussian"].copy()

    # use full_csv for SNR=10, sweep_csv for the other four
    df10   = df_full[df_full.snr_db == 10.0]
    df_sw  = df_sweep[df_sweep.snr_db.isin([0, 5, 15, 20])]
    combined = pd.concat([df10, df_sw], ignore_index=True)

    # methods to plot — exclude CNN (only has SNR=10)
    show = ["LMS", "NLMS", "VSS-LMS (Kwong)",
            "Heuristic Scheduler", "PPO-MLP", "Meta-RL"]

    fig, ax = plt.subplots(figsize=(W1, 2.6), constrained_layout=True)

    for method in show:
        sub = combined[combined.method == method]
        if sub.empty:
            continue
        grouped = sub.groupby("snr_db")["ss_mse_db"]
        mu  = grouped.mean().reindex(SNR_PTS)
        sem = grouped.sem ().reindex(SNR_PTS).fillna(0)

        col        = METHOD_COLOR.get(method, "#888888")
        ls, mk, lw = METHOD_LS.get(method, ("-", "o", 1.4))
        is_rl      = method in ("PPO-MLP", "Meta-RL")

        valid = ~mu.isna()
        xv = np.array(SNR_PTS)[valid.values]
        yv = mu.values[valid.values]
        se = sem.values[valid.values]

        ax.plot(xv, yv,
                color=col, lw=lw, ls=ls,
                marker=mk,
                markersize=6.5 if is_rl else 4.5,
                markerfacecolor=col if is_rl else "white",
                markeredgecolor=col,
                markeredgewidth=0.9 if not is_rl else 0,
                zorder=5 if is_rl else 2,
                label=method)

        if is_rl and len(se) > 0 and np.any(se > 0):
            ax.fill_between(xv, yv - se, yv + se,
                            color=col, alpha=0.14, lw=0, zorder=1)

    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("Steady-state MSE (dB)")
    ax.set_title("Performance vs. Input SNR (Gaussian Noise)")
    ax.set_xticks(SNR_PTS)
    ax.set_xlim(-1, 21)

    # annotate Meta-RL advantage over NLMS at SNR=0
    try:
        y_meta = combined[(combined.method=="Meta-RL") & (combined.snr_db==0)]["ss_mse_db"].mean()
        y_nlms = combined[(combined.method=="NLMS")    & (combined.snr_db==0)]["ss_mse_db"].mean()
        delta  = y_meta - y_nlms
        if np.isfinite(delta) and delta < 0:
            ax.annotate(f"{delta:+.1f} dB vs. NLMS",
                        xy=(0, y_meta),
                        xytext=(3.5, y_meta - 1.8),
                        fontsize=6.2, color=METHOD_COLOR["Meta-RL"],
                        arrowprops=dict(arrowstyle="-|>",
                                        color=METHOD_COLOR["Meta-RL"],
                                        lw=0.7, mutation_scale=7))
    except Exception:
        pass

    _despine(ax)
    leg = ax.legend(ncol=2, loc="upper right",
                    fontsize=6.5,
                    handlelength=2.0, borderpad=0.4, labelspacing=0.22,
                    columnspacing=1.0, framealpha=0.95)
    leg.get_frame().set_linewidth(0.5)

    _save(fig, out_dir, "fig_snr_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Real-world bar chart  (double-column, 7.16 × 2.9 in)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_ORDER = ["gaussian", "impulsive", "burst",
               "regime_switch", "powerline", "baseline_wander"]
NOISE_LABEL = {
    "gaussian":        "Gaussian",
    "impulsive":       "Impulsive",
    "burst":           "Burst",
    "regime_switch":   "Reg.-Switch",
    "powerline":       "Powerline\n(50 Hz)\u2605",
    "baseline_wander": "Baseline\nWander",
}
ECG_METHODS = ["NLMS", "NLMS (mu=0.1)", "VSS-LMS (Kwong)", "Heuristic", "Meta-RL"]
ECG_LABELS  = {
    "NLMS":             "NLMS (μ=0.5)",
    "NLMS (mu=0.1)":    "NLMS (μ=0.1)",
    "VSS-LMS (Kwong)":  "VSS-LMS\n(Kwong)",
    "Heuristic":        "Heuristic",
    "Meta-RL":          "Meta-RL ★",
}


def _bar_panel(ax, df: pd.DataFrame, methods: list[str],
               noise_order: list[str], title: str) -> None:
    n_n   = len(noise_order)
    n_m   = len(methods)
    width = 0.72 / n_m
    offs  = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    xs    = np.arange(n_n)

    for j, m in enumerate(methods):
        sub   = df[df.method == m]
        vals  = [sub[sub.noise == n]["ss_mse_db"].mean() for n in noise_order]
        errs  = [sub[sub.noise == n]["ss_mse_db"].std()  for n in noise_order]
        col   = METHOD_COLOR.get(m, "#888888")
        hatch = METHOD_HATCH.get(m, "")
        is_rl = (m == "Meta-RL")

        ax.bar(xs + offs[j], vals, width=width * 0.9,
               color=col,
               alpha=0.88 if is_rl else 0.65,
               hatch=hatch,
               edgecolor="#333333" if hatch else "white",
               linewidth=0.45,
               zorder=4 if is_rl else 3,
               label=ECG_LABELS.get(m, m))
        ax.errorbar(xs + offs[j], vals, yerr=errs,
                    fmt="none", color="#111111",
                    capsize=2.2, linewidth=0.8, zorder=5)

    ax.set_xticks(xs)
    ax.set_xticklabels([NOISE_LABEL.get(n, n) for n in noise_order],
                       fontsize=6.5, rotation=25, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Steady-state MSE (dB)")
    ax.set_title(title, pad=4.5)
    _despine(ax)


def plot_realworld(ecg_csv: str, speech_csv: str, out_dir: str) -> None:
    df_ecg = pd.read_csv(ecg_csv)
    if "noise_type" in df_ecg.columns and "noise" not in df_ecg.columns:
        df_ecg.rename(columns={"noise_type": "noise"}, inplace=True)

    has_sp = (os.path.exists(speech_csv) and os.path.getsize(speech_csv) > 20)
    df_sp  = pd.read_csv(speech_csv) if has_sp else pd.DataFrame()
    if not df_sp.empty and "noise_type" in df_sp.columns:
        df_sp.rename(columns={"noise_type": "noise"}, inplace=True)

    ncols  = 2 if not df_sp.empty else 1
    ht     = 2.9
    fig, axes = plt.subplots(1, ncols,
                             figsize=(W2 if ncols == 2 else W1 + 0.5, ht),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    avail  = df_ecg.method.unique().tolist()
    methods_ecg = [m for m in ECG_METHODS if m in avail]
    noise_ecg   = [n for n in NOISE_ORDER if n in df_ecg.noise.unique()]

    _bar_panel(axes[0], df_ecg, methods_ecg, noise_ecg,
               "(a)  ECG — MIT-BIH Record 100\n(zero-shot, trained on synthetic only)")

    # remove any auto-legend the panel may have added
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()

    # ★ powerline annotation — placed above bars, well clear of legend
    try:
        y_meta = df_ecg[(df_ecg.method=="Meta-RL") &
                        (df_ecg.noise=="powerline")]["ss_mse_db"].mean()
        y_nlms = df_ecg[(df_ecg.method=="NLMS") &
                        (df_ecg.noise=="powerline")]["ss_mse_db"].mean()
        pw_idx = noise_ecg.index("powerline")
        axes[0].annotate(f"+{y_meta - y_nlms:.1f} dB",
                         xy=(pw_idx, y_meta),
                         xytext=(pw_idx - 0.9, y_meta + 4.0),
                         fontsize=6.5, color=METHOD_COLOR["Meta-RL"],
                         fontweight="bold",
                         arrowprops=dict(arrowstyle="-|>",
                                         color=METHOD_COLOR["Meta-RL"],
                                         lw=0.7, mutation_scale=7))
    except Exception:
        pass

    if ncols == 2 and not df_sp.empty:
        sp_noises  = [n for n in ["gaussian","impulsive","burst","regime_switch"]
                      if n in df_sp.noise.unique()]
        methods_sp = [m for m in ECG_METHODS if m in df_sp.method.unique()]
        _bar_panel(axes[1], df_sp, methods_sp, sp_noises,
                   "(b)  Speech-like Signals\n(unseen during training)")
        if axes[1].get_legend() is not None:
            axes[1].get_legend().remove()

    # legend INSIDE axes[1] upper region — empty space (bars descend from 0)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = axes[1].legend(handles, labels,
                         loc="lower left", ncol=2,
                         fontsize=5.2, frameon=True, borderpad=0.2,
                         handlelength=1.0, handletextpad=0.4,
                         labelspacing=0.15, columnspacing=0.6,
                         framealpha=0.95)
    leg.get_frame().set_linewidth(0.35)

    fig.suptitle(
        "Zero-shot transfer to real signals — policy trained on synthetic noise only",
        fontsize=8.5, fontweight="bold")

    _save(fig, out_dir, "fig_realworld")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Method × Noise-family heatmap  (double-column)
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_ORDER = ["gaussian", "colored", "impulsive", "time_varying",
                "regime_switch", "alpha_stable", "burst", "chirp_interferer"]
FAMILY_LABEL = {
    "gaussian":        "Gaussian",
    "colored":         "Colored",
    "impulsive":       "Impulsive",
    "time_varying":    "TV-SNR",
    "regime_switch":   "Reg.-Sw.",
    "alpha_stable":    "α-Stable",
    "burst":           "Burst",
    "chirp_interferer":"Chirp",
}
HM_METHODS = ["LMS", "NLMS", "RLS",
               "VSS-LMS (Kwong)", "VSS-LMS (Aboulnasr)", "VSS-LMS (Mathews)",
               "Heuristic Scheduler", "Kalman Scheduler",
               "PPO-MLP", "Meta-RL", "CNN (supervised)"]
N_TRAIN = 5


def plot_heatmap(full_csv: str, out_dir: str, snr_db: float = 10.0) -> None:
    df = _merge_rl(pd.read_csv(full_csv))
    df = df[df.snr_db == snr_db]

    methods  = [m for m in HM_METHODS  if m in df.method.unique()]
    families = [f for f in FAMILY_ORDER if f in df.family.unique()]
    n_m, n_f = len(methods), len(families)

    # build matrix — cap diverged values at NaN for colormap
    mat = np.full((n_m, n_f), np.nan)
    div = np.zeros((n_m, n_f), dtype=bool)
    for i, m in enumerate(methods):
        for j, f in enumerate(families):
            vals = df[(df.method == m) & (df.family == f)]["ss_mse_db"].values
            if len(vals):
                v = float(np.mean(vals))
                if v > 0:
                    div[i, j] = True
                else:
                    mat[i, j] = v

    # color range from stable runs only
    stable = mat[np.isfinite(mat)]
    vmin = max(float(np.nanmin(stable)), -45.0) if len(stable) else -45
    vmax = min(float(np.nanmax(stable)),  -5.0) if len(stable) else  -5

    fig, ax = plt.subplots(figsize=(W2, 3.5), constrained_layout=True)

    # use 'RdYlGn': red=poor, yellow=mid, green=good (intuitive for MSE)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # annotate each cell
    for i in range(n_m):
        for j in range(n_f):
            if div[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor="#CCCCCC",
                    hatch="///", edgecolor="#999999", linewidth=0))
                ax.text(j, i, "DIV", ha="center", va="center",
                        fontsize=5.5, color="#444444", fontweight="bold")
            elif np.isfinite(mat[i, j]):
                # luminance-aware text colour
                t = (mat[i, j] - vmin) / max(vmax - vmin, 1e-6)
                tc = "white" if t < 0.45 else "#111111"
                ax.text(j, i, f"{mat[i, j]:.1f}",
                        ha="center", va="center", fontsize=6.5, color=tc)

    ax.set_xticks(range(n_f))
    ax.set_xticklabels([FAMILY_LABEL.get(f, f) for f in families],
                       fontsize=7.5, rotation=28, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(methods, fontsize=7.5)

    # train / OOD divider
    ax.axvline(N_TRAIN - 0.5, color="#111111", lw=1.4, ls="--")
    ylim = ax.get_ylim()
    ax.text(N_TRAIN / 2 - 0.5,      ylim[0] - 0.65,
            "In-distribution (train)",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            color=METHOD_COLOR["NLMS"])
    ax.text(N_TRAIN + (n_f - N_TRAIN) / 2 - 0.5, ylim[0] - 0.65,
            "Out-of-distribution (OOD)",
            ha="center", va="top", fontsize=7.5, fontweight="bold",
            color=METHOD_COLOR["Meta-RL"])

    # RL / classical separator
    rl_start = next((i for i, m in enumerate(methods)
                     if m in ("PPO-MLP", "Meta-RL")), None)
    if rl_start is not None:
        ax.axhline(rl_start - 0.5, color="#555555", lw=0.9, ls=":")

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.01, aspect=22)
    cb.set_label("Steady-state MSE (dB)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title(
        f"Steady-state MSE (dB) per (Method × Noise Family) at SNR = {snr_db:.0f} dB  "
        f"[bold = best unsupervised, DIV = diverged]",
        fontsize=8.5, fontweight="bold", pad=6)

    _save(fig, out_dir, "fig_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    full_csv   = os.path.join(ROOT, "results", "full_evaluation.csv")
    sweep_csv  = os.path.join(ROOT, "results", "snr_sweep.csv")
    ecg_csv    = os.path.join(ROOT, "results", "realworld_ecg.csv")
    speech_csv = os.path.join(ROOT, "results", "realworld_speech.csv")
    out_dir    = os.path.join(ROOT, "paper", "figures")

    _apply_style()
    print("Generating IEEE SPL publication figures …")

    print("  [1/3] SNR sweep (fig_snr_sweep) …")
    plot_snr_sweep(full_csv, sweep_csv, out_dir)

    print("  [2/3] Real-world bar chart (fig_realworld) …")
    plot_realworld(ecg_csv, speech_csv, out_dir)

    print("  [3/3] Heatmap (fig_heatmap) …")
    plot_heatmap(full_csv, out_dir)

    print(f"\nDone. → {out_dir}/")


if __name__ == "__main__":
    main()

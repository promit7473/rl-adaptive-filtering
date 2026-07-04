"""Fig. 3 -- zero-shot ECG transfer, paired improvement over NLMS.

Paired-difference design: for every (record, seed) pair the improvement
Delta = ss_mse_db(NLMS) - ss_mse_db(method) is computed per noise type,
and the plot shows mean +/- 95% CI of those paired differences. The zero
line IS the NLMS baseline, so the figure displays exactly the quantity
the paired Wilcoxon in Sec. IV-B tests (positive = better than NLMS).

Reads results/runs/eval/ecg.csv (benchmark.py output; see README for the
exact invocation -- method names must match METHODS below).
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

from scripts.paper_plots import apply_style, OURS

BASELINE = "NLMS"
# (method name in CSV, legend label, color, marker, dodge offset)
METHODS = [
    ("VSS-Kwong",      "VSS-LMS (Kwong)", "#FF9800", "s", -0.24),
    ("Meta-AF (BPTT)", r"Meta-AF$\dagger$", "#7E57C2", "D", -0.08),
    ("RL-only",        "RL-only (ours)",  "#2196F3", "^",  0.08),
    ("Hybrid (ours)",  "Hybrid (ours)",   OURS,      "o",  0.24),
]

NOISE_ORDER = ["gaussian", "impulsive", "burst", "regime_switch",
               "powerline", "baseline_wander"]
NOISE_LABEL = ["Gaussian", "Impulsive", "Burst", "Reg.-Switch",
               "Powerline\n(50 Hz)", "Baseline\nWander"]


def paired_improvement(d, noise, method):
    """Per-(record, seed) improvement of `method` over NLMS, in dB."""
    sub = d[d.noise == noise]
    a = sub[sub.method == BASELINE].set_index(["record", "seed"])["ss_mse_db"]
    b = sub[sub.method == method].set_index(["record", "seed"])["ss_mse_db"]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} paired (record, seed) rows for "
                         f"{method!r} vs {BASELINE!r} on {noise!r}")
    return (a.loc[common] - b.loc[common]).values  # positive = better


def build_realworld_plot(csv_path=None, out_dir=None, watermark=False):
    """Set watermark=True when rendering mock/preview data into the paper
    so the figure can never be mistaken for a result."""
    apply_style()

    csv_path = csv_path or os.path.join(ROOT, "results", "runs", "eval",
                                        "ecg.csv")
    if not os.path.exists(csv_path):
        # No silent fallback to pre-audit CSVs: the paper figure must come
        # from the current models or not be generated at all.
        raise SystemExit(
            f"{csv_path} not found. Generate it with:\n"
            "  PYTHONPATH=. python3 scripts/benchmark.py --skip-synthetic \\\n"
            "    --out-dir results/runs/eval \\\n"
            '    --hybrid-models "Hybrid (ours)=results/runs/hybrid_seed42/controller_final.pt" \\\n'
            '    --rl-models "RL-only=results/runs/rl_seed42/rl_seed42_final.zip"')
    d = pd.read_csv(csv_path)

    have = set(d.method)
    absent = [m for m, *_ in METHODS if m not in have] \
        + ([BASELINE] if BASELINE not in have else [])
    if absent:
        raise SystemExit(
            f"methods {absent} not in {csv_path} (found: {sorted(have)}); "
            "re-run benchmark.py with exactly these --hybrid-models/"
            "--rl-models names (see README)")

    # match the ablation figure exactly: same width/height -> balanced pair
    fig, ax = plt.subplots(figsize=(3.45, 2.95))
    x = np.arange(len(NOISE_ORDER))

    stats = {}
    for m, label, color, marker, off in METHODS:
        means, cis = [], []
        for n in NOISE_ORDER:
            diff = paired_improvement(d, n, m)
            mu = diff.mean()
            ci = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
            means.append(mu)
            cis.append(ci)
            stats[(m, n)] = (mu, ci, len(diff))
        is_ours = (m == "Hybrid (ours)")
        ax.errorbar(x + off, means, yerr=cis,
                    fmt=marker, ms=4.6 if is_ours else 3.6,
                    color=color, mec="black", mew=0.4,
                    ls="none", elinewidth=0.8, capsize=1.8,
                    zorder=5 if is_ours else 4, label=label)

    # zero line = NLMS baseline (the paired reference; y-label names it,
    # so no text label that could collide with data-dependent markers)
    ax.axhline(0, color="#333333", lw=0.8, ls=(0, (5, 2)), zorder=2)

    # callout on the headline result: hybrid on powerline
    ip = NOISE_ORDER.index("powerline")
    mu_pl = stats[("Hybrid (ours)", "powerline")][0]
    ax.annotate(rf"$+{mu_pl:.1f}$ dB", xy=(ip + 0.24, mu_pl),
                xytext=(ip + 0.24, mu_pl + 2.6),
                color=OURS, fontsize=6.2, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec=OURS, lw=0.6, alpha=0.95),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.8,
                                shrinkA=1, shrinkB=3))

    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LABEL, fontsize=7, fontweight="bold",
                       rotation=35, ha="right")
    ax.set_xlim(-0.55, len(NOISE_ORDER) - 0.45)
    ax.set_ylabel("Improvement over NLMS (dB)", fontsize=8)

    lo = min(mu - ci for (mu, ci, _) in stats.values())
    hi = max(mu + ci for (mu, ci, _) in stats.values())
    ax.set_ylim(np.floor(lo - 1.2), np.ceil(hi + 3.2))

    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("Zero-Shot ECG Transfer", fontweight="bold",
                 fontsize=8.6, pad=16)
    ax.text(0.5, 1.012, "paired improvement over NLMS on real MIT-BIH",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.8, style="italic", color="#333333")
    ax.grid(True, axis="y", color="#E2E2E2", lw=0.45, ls=(0, (3, 3)))

    ax.legend(loc="upper left", ncol=1, frameon=True, fontsize=5.8,
              labelspacing=0.32, handletextpad=0.35, borderpad=0.4,
              framealpha=0.96)

    if watermark:
        ax.text(0.5, 0.45, "MOCK DATA\nlayout preview only",
                transform=ax.transAxes, rotation=28, fontsize=15,
                color="#888888", alpha=0.38, fontweight="bold",
                ha="center", va="center", zorder=10)

    fig.tight_layout(pad=0.3)

    out_dir = out_dir or os.path.join(ROOT, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_realworld.pdf"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_realworld.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # print the numbers the paper text cites
    print("Paired improvement over NLMS (dB), mean [95% CI], n pairs:")
    for (m, n), (mu, ci, k) in stats.items():
        print(f"  {m:16s} {n:16s} {mu:+6.2f} [{ci:4.2f}]  n={k}")
    print("Zero-shot ECG figure saved successfully!")


if __name__ == "__main__":
    build_realworld_plot()

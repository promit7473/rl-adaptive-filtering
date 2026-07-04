"""Fig. 4 -- ablation as the COST of removing each auxiliary head.

Instead of 28 absolute bars (7 families x 4 variants), the figure shows
what the paper actually claims: the paired SS-MSE degradation of each
ablated variant relative to the full hybrid, Delta = variant - full in
dB per (signal, family, seed) episode (positive = worse). Horizontal
bars give the mean cost with a 95% CI whisker; small dots overlay the
per-family mean costs so the spread across noise families stays visible.

Reads results/ablations/ablation_eval.csv (run_ablations.py eval).
No hardcoded numbers.
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

from scripts.paper_plots import apply_style

FULL = "full"
# (variant key in CSV, axis label, color) -- plotted top to bottom
VARIANTS = [
    ("no_err", "$-\\hat{e}_{t+1}$\n(no error pred.)",  "#FF9800"),
    ("no_sig", "$-\\hat{d}_{t+1}$\n(no signal pred.)", "#2196F3"),
    ("no_aux", "$-$both\n(no aux heads)",              "#4CAF50"),
]

FAM_LABEL = {"gaussian": "Gaussian", "colored": "Colored",
             "impulsive": "Impulsive", "time_varying": "TV-SNR",
             "regime_switch": "Reg.-Sw", "alpha_stable": r"$\alpha$-Stable",
             "burst": "Burst", "chirp_interferer": "Chirp"}


def build_ablation_plot(csv_path=None, out_dir=None, watermark=False):
    """Set watermark=True when rendering mock/preview data into the paper
    so the figure can never be mistaken for a result."""
    apply_style()

    csv_path = csv_path or os.path.join(ROOT, "results", "ablations",
                                        "ablation_eval.csv")
    df = pd.read_csv(csv_path)
    df = df[df.snr_db == 10]

    absent = [k for k, *_ in VARIANTS + [(FULL,)] if k not in set(df.variant)]
    if absent:
        raise SystemExit(f"variants {absent} missing from {csv_path}; train "
                         "them with 'python scripts/run_ablations.py train' "
                         "and re-run 'run_ablations.py eval'")

    # paired degradation vs the full hybrid on identical episodes
    keys = ["signal", "family", "seed"]
    full = df[df.variant == FULL].set_index(keys)["ss_mse_db"]

    rows = []          # (variant, mean, ci)
    fam_costs = {}     # variant -> {family: mean cost}
    for key, _label, _color in VARIANTS:
        var = df[df.variant == key].set_index(keys)["ss_mse_db"]
        common = full.index.intersection(var.index)
        if len(common) < 3:
            raise SystemExit(f"only {len(common)} episodes shared between "
                             f"{key!r} and {FULL!r} in {csv_path}")
        delta = (var.loc[common] - full.loc[common])   # positive = worse
        mu = float(delta.mean())
        ci = 1.96 * float(delta.std(ddof=1)) / np.sqrt(len(delta))
        rows.append((key, mu, ci))
        fam_costs[key] = delta.groupby(level="family").mean()

    # match the ECG figure exactly: same width/height -> balanced pair
    fig, ax = plt.subplots(figsize=(3.45, 2.95))
    y = np.arange(len(VARIANTS))[::-1]   # first variant on top

    for (key, label, color), yi in zip(VARIANTS, y):
        mu, ci = next((m, c) for k, m, c in rows if k == key)
        ax.barh(yi, mu, height=0.52, color=color, edgecolor="black",
                linewidth=0.5, zorder=3,
                xerr=ci, capsize=2.4, error_kw=dict(elinewidth=0.8))
        # per-family mean costs as small jittered dots; the jitter skips a
        # center band so the dots never sit on the in-bar value label
        fams = fam_costs[key]
        half = (len(fams) + 1) // 2
        jit = np.concatenate([np.linspace(-0.21, -0.11, half),
                              np.linspace(0.11, 0.21, len(fams) - half)])
        ax.plot(fams.values, yi + jit, "o", ms=2.4, mfc="white",
                mec="#333333", mew=0.5, ls="none", zorder=4)
        # value label inside the bar (clear of the dot cloud, which sits
        # around the bar end); short bars fall back to beyond the dots
        if mu > 1.2:
            ax.text(0.18, yi, f"$+{mu:.1f}$ dB", va="center", ha="left",
                    fontsize=7, fontweight="bold", color="white", zorder=5)
        else:
            ax.text(max(mu + ci, fams.max()) + 0.3, yi, f"$+{mu:.1f}$ dB",
                    va="center", ha="left", fontsize=7, fontweight="bold",
                    color="#222222", zorder=5)

    # zero line = full hybrid (the paired reference)
    ax.axvline(0, color="#333333", lw=0.9, zorder=2)
    ax.text(-0.15, len(VARIANTS) - 0.45, "full\nhybrid", fontsize=5.8,
            color="#333333", ha="right", va="top", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([lab for _k, lab, _c in VARIANTS], fontsize=7,
                       fontweight="bold")
    ax.set_ylim(-0.55, len(VARIANTS) - 0.25)
    ax.set_xlabel("SS-MSE degradation vs. full hybrid (dB)", fontsize=8)

    xmax = max(mu + ci for _k, mu, ci in rows)
    xmax = max(xmax, max(f.max() for f in fam_costs.values()))
    xmin = min(0, min(f.min() for f in fam_costs.values()))
    ax.set_xlim(np.floor(xmin - 0.6), np.ceil(xmax + 0.6))

    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("Auxiliary-Head Ablation", fontweight="bold",
                 fontsize=8.6, pad=16)
    ax.text(0.5, 1.012,
            "cost of removing each head; $\\circ$ = per-family mean",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.8, style="italic", color="#333333")
    ax.grid(True, axis="x", color="#E2E2E2", lw=0.45, ls=(0, (3, 3)))

    if watermark:
        ax.text(0.5, 0.45, "MOCK DATA\nlayout preview only",
                transform=ax.transAxes, rotation=28, fontsize=15,
                color="#888888", alpha=0.38, fontweight="bold",
                ha="center", va="center", zorder=10)

    fig.tight_layout(pad=0.3)

    out_dir = out_dir or os.path.join(ROOT, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_ablation.pdf"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_ablation.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # print the numbers the paper text cites
    print("Paired degradation vs full hybrid (dB), SNR=10:")
    for key, mu, ci in rows:
        fams = ", ".join(f"{FAM_LABEL.get(f, f)} {v:+.1f}"
                         for f, v in fam_costs[key].items())
        print(f"  {key:8s} {mu:+5.2f} [{ci:4.2f}]   per-family: {fams}")


if __name__ == "__main__":
    build_ablation_plot()

#!/usr/bin/env python3
"""Generate Table I (LaTeX rows) from results/runs/eval/synthetic.csv.

Filters to SNR=10 dB, averages ss_mse_db over signals and seeds per
(method, family), bolds the best method per family, marks divergence.
Writes paper/tables/table1_generated.tex (the tabular body only).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

CSV = "results/runs/eval/synthetic.csv"
OUT = "paper/tables/table1_generated.tex"

FAMS = ["gaussian", "colored", "impulsive", "time_varying", "regime_switch",
        "alpha_stable", "burst", "chirp_interferer"]

# (csv name, latex label, group) — groups separated by \midrule
ROWS = [
    ("NLMS",           r"NLMS ($\mu{=}0.5$)",          0),
    ("RLS",            r"RLS ($\lambda{=}0.995$)",      0),
    ("VSS-Kwong",      r"VSS-LMS (Kwong)",              0),
    ("VSS-Aboulnasr",  r"VSS-LMS (Aboulnasr)",          0),
    ("PID-NLMS",       r"PID-NLMS",                     0),
    ("Fixed-Leaky",    r"Fixed-Leaky ($\mu{=}0.5$)",   0),
    ("Meta-AF",        r"Meta-AF$\dagger$",             1),
    ("RL-only",        r"RL-only (ours)",               2),
    ("BPTT-only (ours)", r"BPTT-only (ours)",           2),
    ("Hybrid (ours)",  r"\textbf{Hybrid BPTT+RL (ours)}", 2),
]


def main():
    df = pd.read_csv(CSV)
    df = df[df.snr_db == 10]

    methods = [m for m, _, _ in ROWS if m in set(df.method)]
    missing = [m for m, _, _ in ROWS if m not in set(df.method)]
    if missing:
        print(f"WARNING missing methods: {missing}")

    mean = df.pivot_table(index="method", columns="family",
                          values="ss_mse_db", aggfunc="mean")
    div = df.pivot_table(index="method", columns="family",
                         values="diverged", aggfunc="sum")
    overall = df.groupby("method")["ss_mse_db"].mean()

    # best (lowest) per family among non-diverged entries
    best = {}
    for f in FAMS:
        vals = {m: mean.loc[m, f] for m in methods
                if f in mean.columns and np.isfinite(mean.loc[m, f])
                and div.loc[m, f] == 0}
        best[f] = min(vals, key=vals.get) if vals else None
    best_mean = min({m: overall[m] for m in methods
                     if np.isfinite(overall[m])}, key=lambda m: overall[m])

    lines = []
    prev_group = 0
    for m, label, group in ROWS:
        if m not in methods:
            continue
        if group != prev_group:
            lines.append(r"\midrule")
            prev_group = group
        cells = [label]
        any_div = False
        for f in FAMS:
            v = mean.loc[m, f]
            d = int(div.loc[m, f]) if f in div.columns else 0
            if not np.isfinite(v) or v > 50:
                cell = r"{\emph{div.}${\uparrow}$}"
                any_div = True
            else:
                s = f"{v:+.1f}" if v > 0 else f"{v:.1f}"
                cell = f"${s}$"
                if d > 0:
                    cell = f"${s}{{\\uparrow}}$"
                    any_div = True
                elif best.get(f) == m:
                    cell = f"$\\mathbf{{{s}}}$"
            cells.append(cell)
        if any_div:
            cells.append(r"\emph{---}")
        else:
            v = overall[m]
            s = f"{v:.1f}"
            cells.append(f"$\\mathbf{{{s}}}$" if m == best_mean else f"${s}$")
        lines.append(" & ".join(cells) + r" \\")

    body = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(body)
    print(body)
    print(f"wrote {OUT}")

    # convenience: headline numbers for the text
    print("\n--- headline numbers (SNR=10) ---")
    for m in methods:
        print(f"{m:20s} mean={overall[m]:+.2f}")
    if "Hybrid (ours)" in methods and "Meta-AF" in methods:
        h = mean.loc["Hybrid (ours)"]; a = mean.loc["Meta-AF"]
        wins = [f for f in FAMS if h[f] < a[f]]
        print(f"Hybrid beats Meta-AF on {len(wins)}/8: {wins}")
        n = mean.loc["NLMS"]
        winsn = [f for f in FAMS if h[f] < n[f]]
        print(f"Hybrid beats NLMS on {len(winsn)}/8: {winsn}")


if __name__ == "__main__":
    main()

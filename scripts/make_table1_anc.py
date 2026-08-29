#!/usr/bin/env python3
"""Generate paper/tables/table1_generated.tex from results/runs/eval_anc/synthetic.csv.

Emits one LaTeX row per method at SNR=10 dB: mean steady-state residual (dB) per
family + overall mean, bolding the best method per column and flagging any method
with a divergent run (^). Column order matches paper.tex Table I.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CSV = "results/runs/eval_anc/synthetic.csv"
OUT = "paper/tables/table1_generated.tex"
# paper column order -> csv family key
COLS = [("Powerl.", "powerline"), ("Base.W.", "baseline_wander"),
        ("Echo", "echo"), ("RegSw", "regime_switch"),
        ("Chirp", "narrowband_chirp"), ("Echo-L", "echo_long")]
# display order / naming of methods (row order); ours last for emphasis
METHOD_ORDER = ["NLMS", "VSS-LMS", "IIR-Notch", "RLS(0.99)", "RLS(0.999)",
                "Kalman(Q=1e-6)", "Kalman(Q=1e-4)", "RL-only", "BPTT-only",
                "Hybrid (ours)"]


def main():
    if not os.path.exists(CSV):
        print(f"[make_table1_anc] {CSV} missing — run scripts/train_anc.py eval first")
        sys.exit(1)
    df = pd.read_csv(CSV)
    d10 = df[df.snr_db == 10]
    piv = d10.pivot_table(index="method", columns="family",
                          values="ss_res_db", aggfunc="mean")
    mean = d10.groupby("method")["ss_res_db"].mean()
    div = df.groupby("method")["diverged"].max()  # any diverged run

    # best (lowest) per column among present methods
    best = {csv: piv[csv].min() for _, csv in COLS if csv in piv.columns}
    best_mean = mean.min()

    methods = [m for m in METHOD_ORDER if m in piv.index]
    methods += [m for m in piv.index if m not in METHOD_ORDER]

    lines = []
    for m in methods:
        cells = []
        for _, csv in COLS:
            if csv in piv.columns and not np.isnan(piv.loc[m, csv]):
                v = piv.loc[m, csv]
                s = f"{v:.1f}"
                if abs(v - best[csv]) < 0.05:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            else:
                cells.append("--")
        mv = mean.get(m, np.nan)
        ms = f"{mv:.1f}" if not np.isnan(mv) else "--"
        if not np.isnan(mv) and abs(mv - best_mean) < 0.05:
            ms = f"\\textbf{{{ms}}}"
        name = m + ("$^{\\uparrow}$" if div.get(m, 0) else "")
        lines.append(f"{name} & " + " & ".join(cells) + f" & {ms} \\\\")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[make_table1_anc] wrote {len(lines)} rows -> {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

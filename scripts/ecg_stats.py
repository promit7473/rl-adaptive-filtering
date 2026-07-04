#!/usr/bin/env python3
"""Paired statistics for the ECG zero-shot claims (paper Sec. IV-B).

Reads results/runs/eval/ecg.csv (benchmark.py output) and reports, per
noise type, the paired Wilcoxon signed-rank test of ours vs NLMS over
(record, seed) pairs, plus the mean gap in dB.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from scipy.stats import wilcoxon

CSV = "results/runs/eval/ecg.csv"
OURS = sys.argv[1] if len(sys.argv) > 1 else "Hybrid (ours)"
BASE = "NLMS"


def main():
    df = pd.read_csv(CSV)
    print(f"methods: {sorted(df.method.unique())}")
    print(f"ours={OURS!r} vs base={BASE!r}\n")

    for m in (OURS, BASE):
        if m not in set(df.method):
            raise SystemExit(f"method {m!r} not in {CSV}; pass the name used "
                             "with benchmark.py --hybrid-models as argv[1]")

    overall = df[df.method == OURS]["ss_mse_db"].mean()
    print(f"{OURS} mean SS MSE over all conditions: {overall:.2f} dB\n")

    for noise in sorted(df.noise.unique()):
        sub = df[df.noise == noise]
        a = (sub[sub.method == OURS]
             .set_index(["record", "seed"])["ss_mse_db"].sort_index())
        b = (sub[sub.method == BASE]
             .set_index(["record", "seed"])["ss_mse_db"].sort_index())
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        if len(common) < 5:
            print(f"{noise:18s} insufficient pairs ({len(common)})")
            continue
        diff = (b - a)  # positive = ours better (lower MSE)
        try:
            stat, p = wilcoxon(a.values, b.values)
        except ValueError:
            stat, p = float("nan"), float("nan")
        print(f"{noise:18s} n={len(common):3d}  ours={a.mean():7.2f}  "
              f"NLMS={b.mean():7.2f}  gap={diff.mean():+5.2f} dB  "
              f"wilcoxon p={p:.2e}")


if __name__ == "__main__":
    main()

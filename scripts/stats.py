"""Paired statistics across methods (Wilcoxon + Holm-Bonferroni + Cohen's d).

Reads a results CSV with columns at minimum:
  method, ss_mse_db, and one or more grouping columns.
Pairs samples across methods by the *other* columns (family, snr_db, seed, ...)
so that paired tests are well-defined.

Usage:
  python3 scripts/stats.py results/v2_eval/synthetic.csv \\
      --reference Meta-RL --metric ss_mse_db --out results/v2_eval/stats.csv
  python3 scripts/stats.py results/v2_eval/ecg.csv \\
      --reference Meta-RL --group record noise snr_db seed
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d_paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 0 else float("nan")


def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, min(val, 1.0))
        adj[idx] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--reference", required=True,
                    help="method name to compare every other method against")
    ap.add_argument("--metric", default="ss_mse_db")
    ap.add_argument("--group", nargs="+", default=None,
                    help="columns that identify a paired sample (default: all "
                         "non-method, non-metric columns)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.group is None:
        skip = {"method", args.metric, "ss_mse", "ep_mse",
                "conv_time", "inference_time_ms"}
        args.group = [c for c in df.columns if c not in skip]

    ref = df[df["method"] == args.reference]
    if ref.empty:
        raise SystemExit(f"reference {args.reference!r} not in {args.csv}")
    others = sorted(m for m in df["method"].unique() if m != args.reference)

    rows, raw_p = [], []
    pair_cache = []
    for m in others:
        cmp = df[df["method"] == m]
        merged = ref.merge(cmp, on=args.group, suffixes=("_ref", "_cmp"))
        x = merged[f"{args.metric}_ref"].values  # reference (Meta-RL) values
        y = merged[f"{args.metric}_cmp"].values  # competitor values
        if len(x) < 5:
            print(f"[skip] {m}: only {len(x)} paired samples")
            continue
        # Lower SS-MSE-dB is better. Test: reference < competitor.
        try:
            stat, p = stats.wilcoxon(x, y, alternative="less")
        except ValueError:
            stat, p = (float("nan"), 1.0)
        d = cohens_d_paired(x, y)
        rows.append({
            "method": m,
            "n": len(x),
            "mean_diff_db": float(np.mean(x - y)),  # negative = ref better
            "median_diff_db": float(np.median(x - y)),
            "wilcoxon_stat": float(stat),
            "p_raw": float(p),
            "cohens_d": d,
        })
        raw_p.append(p)
        pair_cache.append(m)

    if rows:
        adj = holm_bonferroni(raw_p)
        for r, ap_ in zip(rows, adj):
            r["p_holm"] = float(ap_)

    out = args.out or args.csv.replace(".csv", "_stats.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} comparisons -> {out}")
    if rows:
        print(f"\n{args.reference} vs ... (mean Δ dB | p_holm | d):")
        for r in sorted(rows, key=lambda r: r["mean_diff_db"]):
            print(f"  {r['method']:<24s}  Δ={r['mean_diff_db']:+6.2f}  "
                  f"p={r['p_holm']:.2e}  d={r['cohens_d']:+.2f}")


if __name__ == "__main__":
    main()

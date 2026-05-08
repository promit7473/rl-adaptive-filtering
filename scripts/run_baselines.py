"""Run all classical baselines across train + OOD noise families. Save metrics + plots."""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from src.filters import make_filter, windowize
from src.eval import summarize, plot_signal_comparison, plot_error_curves, plot_bar_metric, ensure_dir


FILTERS = {
    "LMS":     ("lms",     dict(mu=0.01)),
    "NLMS":    ("nlms",    dict(mu=0.5)),
    "VSS-LMS": ("vss_lms", dict(mu_max=0.05, alpha=0.97, gamma=1e-3)),
    "RLS":     ("rls",     dict(forgetting=0.995)),
}


def run_one(noise_family: str, n: int, fs: float, snr_db: float, seed: int, order: int):
    rng = np.random.default_rng(seed)
    clean = make_signal("multitone", n=n, fs=fs, rng=rng)
    noise = make_noise(noise_family, clean, rng, snr_db=snr_db, fs=fs)
    noisy = clean + noise
    U = windowize(noisy, order=order)
    d = clean

    rows = []
    errors = {}
    recovered = {}
    for label, (name, kw) in FILTERS.items():
        f = make_filter(name, order=order, **kw)
        y, e = f.run(U, d)
        rec = noisy - e  # filter recovered = noisy - residual error proxy; we use y
        # `y` is the filter's prediction of clean; treat that as recovered estimate
        rec = y
        m = summarize(e, clean=clean, noisy=noisy, recovered=rec)
        m["filter"] = label
        m["noise"] = noise_family
        rows.append(m)
        errors[label] = e
        recovered[label] = rec
    return rows, errors, recovered, clean, noisy


def main(out_dir: str = "results/baselines",
         n: int = 4000, fs: float = 8000.0, snr_db: float = 10.0,
         seeds=(0, 1, 2), order: int = 16):
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    all_rows = []
    for fam in TRAIN_FAMILIES + OOD_FAMILIES:
        per_seed_rows = []
        last_errors, last_recovered, last_clean, last_noisy = None, None, None, None
        for s in seeds:
            rows, errors, recovered, clean, noisy = run_one(fam, n, fs, snr_db, s, order)
            for r in rows:
                r["seed"] = s
            per_seed_rows.extend(rows)
            last_errors, last_recovered, last_clean, last_noisy = errors, recovered, clean, noisy
        all_rows.extend(per_seed_rows)

        # plots from last seed
        plot_signal_comparison(last_clean, last_noisy, last_recovered,
                               os.path.join(plots_dir, f"signals_{fam}.png"),
                               title=f"signal recovery — {fam} @ {snr_db} dB SNR")
        plot_error_curves(last_errors,
                          os.path.join(plots_dir, f"errors_{fam}.png"),
                          title=f"learning curve — {fam}")

    df = pd.DataFrame(all_rows)
    df_csv = os.path.join(out_dir, "baselines_per_seed.csv")
    df.to_csv(df_csv, index=False)

    agg = df.groupby(["noise", "filter"]).agg(
        mse_mean=("ss_mse", "mean"),
        mse_std=("ss_mse", "std"),
        mse_db_mean=("ss_mse_db", "mean"),
        snr_imp_mean=("snr_improvement_db", "mean"),
        conv_time_mean=("conv_time", "mean"),
    ).reset_index()
    agg_csv = os.path.join(out_dir, "baselines_summary.csv")
    agg.to_csv(agg_csv, index=False)

    # bar chart: per-filter average steady-state MSE (dB) on OOD
    ood = agg[agg["noise"].isin(OOD_FAMILIES)]
    if len(ood):
        bar = ood.groupby("filter")["mse_db_mean"].mean().to_dict()
        plot_bar_metric(bar, os.path.join(plots_dir, "ood_avg_mse_db.png"),
                        ylabel="steady-state MSE (dB)",
                        title="avg OOD steady-state MSE (dB) per filter")

    print(f"\nWrote {df_csv}")
    print(f"Wrote {agg_csv}")
    print("\nSummary (steady-state MSE in dB, mean over seeds):")
    pivot = agg.pivot(index="noise", columns="filter", values="mse_db_mean").round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

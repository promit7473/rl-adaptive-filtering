#!/usr/bin/env python3
"""Ablation study: train hybrid variants at a reduced (1k-iter) budget and
evaluate them on the shared eval episodes at SNR=10.

Variants (paper Sec. IV-C):
  full       : full hybrid at the SAME reduced budget (fair comparison anchor)
  no_sig     : aux_signal_weight = 0  (remove d-hat head)
  no_err     : aux_error_weight  = 0  (remove e-hat head)
  no_aux     : both aux prediction heads removed
  no_sched   : no decaying base step-size schedule
  no_meta    : no RL^2 meta-episodes (LSTM state reset between episodes)

Usage:
  python scripts/run_ablations.py train full no_sig      # train a subset
  python scripts/run_ablations.py eval                   # eval all trained
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_num_threads(1)

ABL_DIR = "results/ablations"
N_ITERS = 1000

VARIANTS = {
    "full":     dict(),
    "no_sig":   dict(aux_signal_weight=0.0),
    "no_err":   dict(aux_error_weight=0.0),
    "no_aux":   dict(aux_signal_weight=0.0, aux_error_weight=0.0),
    "no_sched": dict(use_mu_schedule=False),
    "no_meta":  dict(meta_episode_len=1),
}

LABELS = {
    "full": "Full Hybrid",
    "no_sig": r"$-\hat{d}_{t+1}$",
    "no_err": r"$-\hat{e}_{t+1}$",
    "no_aux": "$-$both",
    "no_sched": "$-$schedule",
    "no_meta": r"$-$RL$^2$",
}


def base_cfg(seed=42):
    # Main-run config (train_pipeline.hybrid_cfg) at half the budget, with
    # the phase boundaries halved to match; everything else inherits from
    # the main run so ablation deltas isolate the ablated component.
    from scripts.train_pipeline import hybrid_cfg
    return hybrid_cfg(seed,
                      n_iters=N_ITERS,          # 1000 vs main 2000
                      rl_phase_start=200,       # main 400, scaled with budget
                      curriculum_ramp_iters=400,  # main 800, scaled with budget
                      eval_every=10_000)        # skip mid-train eval


def train(names):
    from dataclasses import replace
    from src.agents.hybrid_trainer import train_hybrid
    for name in names:
        if name not in VARIANTS:
            print(f"unknown variant {name}"); continue
        out = os.path.join(ABL_DIR, name)
        final = os.path.join(out, "controller_final.pt")
        if os.path.isfile(final):
            print(f"[abl] {name}: already trained, skipping")
            continue
        cfg = replace(base_cfg(), **VARIANTS[name])
        print(f"[abl] training {name} ...", flush=True)
        t0 = time.time()
        train_hybrid(cfg, out_dir=out)
        print(f"[abl] {name} done in {(time.time()-t0)/3600:.2f} h", flush=True)


def evaluate():
    import csv
    from scripts import benchmark as bm
    from scripts.train_pipeline import make_episodes, FS, N, ORDER

    episodes = {k: v for k, v in make_episodes().items() if k[2] == 10}
    rows = []
    for name in VARIANTS:
        path = os.path.join(ABL_DIR, name, "controller_final.pt")
        if not os.path.isfile(path):
            print(f"[abl] {name}: no checkpoint, skipping")
            continue
        ctrl, env_kw = bm._load_controller(path)
        print(f"[abl] evaluating {name} ({len(episodes)} episodes)", flush=True)
        for (sig, fam, snr, seed), (clean, noisy) in episodes.items():
            e, dt, divc = bm._run_controller_episode(
                ctrl, env_kw, clean, noisy, FS, N, ORDER)
            ss = float(np.mean(e[-N // 4:] ** 2))
            rows.append(dict(variant=name, signal=sig, family=fam,
                             snr_db=snr, seed=seed,
                             ss_mse=ss,
                             ss_mse_db=float(10 * np.log10(ss + 1e-12)),
                             divergence_count=divc))
    if not rows:
        raise SystemExit("[abl] no trained variants found in "
                         f"{ABL_DIR}; run 'run_ablations.py train' first")
    os.makedirs(ABL_DIR, exist_ok=True)
    out_csv = os.path.join(ABL_DIR, "ablation_eval.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[abl] wrote {len(rows)} rows -> {out_csv}")

    import pandas as pd
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="variant", columns="family",
                           values="ss_mse_db", aggfunc="mean").round(1)
    pivot["MEAN"] = df.groupby("variant")["ss_mse_db"].mean().round(1)
    print(pivot)
    pivot.to_csv(os.path.join(ABL_DIR, "ablation_pivot.csv"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    mode = sys.argv[1]
    if mode == "train":
        train(sys.argv[2:] or list(VARIANTS))
    elif mode == "eval":
        evaluate()
    else:
        print(f"unknown mode {mode}")

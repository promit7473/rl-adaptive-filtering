"""Evaluate Meta-RL policy on multiple MIT-BIH records (zero-shot)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import wfdb
from sb3_contrib import RecurrentPPO
from scripts.eval_realworld import (
    _resample, _normalize, _add_noise, _eval_segment, FS_TARGET, EPISODE_N, SEEDS, SNR_DB
)

RECORDS = ["100", "101", "103", "105", "115"]
META_PATH = "results/ppo_meta_v5/ppo_meta_seed442_final.zip"

meta = RecurrentPPO.load(META_PATH, device="cpu")

noise_types = ["gaussian", "powerline", "baseline_wander", "impulsive", "burst", "regime_switch"]
rows = []
for rec_id in RECORDS:
    print(f"[ECG] record {rec_id}")
    rec = wfdb.rdrecord(rec_id, pn_dir="mitdb")
    sig = _normalize(_resample(rec.p_signal[:, 0], int(rec.fs)))
    for seed in SEEDS:
        rng = np.random.default_rng(seed + int(rec_id))
        start = rng.integers(0, max(1, len(sig) - EPISODE_N))
        clean = sig[start: start + EPISODE_N]
        if len(clean) < EPISODE_N:
            clean = np.pad(clean, (0, EPISODE_N - len(clean)))
        for noise_kind in noise_types:
            noisy = _add_noise(clean, SNR_DB, rng, kind=noise_kind)
            scores = _eval_segment(clean, noisy, meta, None)
            for method, db in scores.items():
                rows.append(dict(record=rec_id, noise=noise_kind, seed=seed,
                                 method=method, ss_mse_db=db))
        print(f"  seed={seed} powerline meta={scores['Meta-RL']:.1f} nlms={scores['NLMS']:.1f}")

df = pd.DataFrame(rows)
df.to_csv("results/realworld_ecg_multi.csv", index=False)

# summary
print("\n=== Mean SS MSE (dB) per record × method ===")
print(df.groupby(['record','method']).ss_mse_db.mean().unstack().round(1))
print("\n=== Powerline only — Meta-RL vs NLMS gap ===")
pw = df[df.noise == 'powerline']
g = pw.groupby(['record','method']).ss_mse_db.mean().unstack()
g['gap_dB'] = g['NLMS'] - g['Meta-RL']
print(g[['NLMS','Meta-RL','gap_dB']].round(1))

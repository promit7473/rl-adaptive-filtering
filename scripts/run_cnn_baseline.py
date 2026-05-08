"""Train 1D CNN per train-noise-family, evaluate on held-out seeds AND on OOD families.

This reproduces the standard supervised-DL adaptive-filter baseline.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from src.supervised.cnn import train_cnn, cnn_predict, make_windows
from src.eval import summarize, ensure_dir


WINDOW = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gen_episode(noise_family: str, n: int, fs: float, snr_db: float, seed: int):
    rng = np.random.default_rng(seed)
    clean = make_signal("multitone", n=n, fs=fs, rng=rng)
    noise = make_noise(noise_family, clean, rng, snr_db=snr_db, fs=fs)
    return clean, clean + noise


def make_long_train_set(train_family: str, n_per_seed: int, fs: float, snr_db: float,
                         n_seeds: int):
    cleans, noisys = [], []
    for s in range(n_seeds):
        c, ny = gen_episode(train_family, n_per_seed, fs, snr_db, seed=1000 + s)
        cleans.append(c); noisys.append(ny)
    return np.concatenate(noisys), np.concatenate(cleans)


def main(out_dir: str = "results/cnn",
         n: int = 4000, fs: float = 8000.0, snr_db: float = 10.0,
         train_seeds: int = 8, eval_seeds=(0, 1, 2),
         epochs: int = 5):
    ensure_dir(out_dir)
    rows = []
    models = {}

    print(f"device={DEVICE}")
    # train one CNN per *train* noise family; also a "mixed" model for fair comparison with RL
    train_targets = list(TRAIN_FAMILIES) + ["mixed"]
    for tf in train_targets:
        print(f"[train] family={tf}")
        if tf == "mixed":
            n_per = n
            cleans, noisys = [], []
            for s in range(train_seeds):
                fam = np.random.default_rng(2000 + s).choice(TRAIN_FAMILIES)
                c, ny = gen_episode(str(fam), n_per, fs, snr_db, seed=3000 + s)
                cleans.append(c); noisys.append(ny)
            noisy_train = np.concatenate(noisys); clean_train = np.concatenate(cleans)
        else:
            noisy_train, clean_train = make_long_train_set(tf, n, fs, snr_db, train_seeds)
        model = train_cnn(noisy_train, clean_train, window=WINDOW, epochs=epochs,
                          device=DEVICE, verbose=True)
        models[tf] = model

    # evaluate every model on every family (matrix)
    print("\n[eval] CNN matrix")
    for tf, model in models.items():
        for ef in TRAIN_FAMILIES + OOD_FAMILIES:
            for s in eval_seeds:
                clean, noisy = gen_episode(ef, n, fs, snr_db, seed=s)
                pred = cnn_predict(model, noisy, WINDOW, device=DEVICE)
                e = clean - pred
                m = summarize(e, clean=clean, noisy=noisy, recovered=pred)
                m.update({"train_family": tf, "eval_family": ef, "seed": s, "model": "CNN"})
                rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "cnn_matrix.csv"), index=False)
    agg = df.groupby(["train_family", "eval_family"]).agg(
        mse_db_mean=("ss_mse_db", "mean"),
        snr_imp_mean=("snr_improvement_db", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, "cnn_summary.csv"), index=False)
    print("\nCNN steady-state MSE (dB) — rows: train family, cols: eval family")
    pivot = agg.pivot(index="train_family", columns="eval_family", values="mse_db_mean").round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

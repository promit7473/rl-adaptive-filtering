"""Unified evaluation harness for all methods.

Runs every method on every (family, SNR, seed) combination and produces
a single CSV with all metrics. This is the single source of truth for results.
"""
from __future__ import annotations
import os
import csv
import time
import numpy as np
from typing import Sequence

from ..signals.generators import make_signal
from ..noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from ..filters import (
    LMS, NLMS, VSSLMS, AboulnasrMayyasVSS, MathewsXieVSS,
    LMP, HeuristicMuScheduler, KalmanMuScheduler,
    RLS, windowize,
)
from ..supervised.cnn import Conv1DDenoiser, train_cnn, cnn_predict
from ..envs import AdaptiveFilterEnv, EnvConfig
from ..eval.metrics import steady_state_mse, convergence_time

ALL_FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)


def _make_signal_and_noise(family: str, n: int, fs: float, snr_db: float,
                           seed: int, signal_kind: str = "multitone"):
    rng = np.random.default_rng(seed)
    clean = make_signal(signal_kind, n=n, fs=fs, rng=rng)
    noise = make_noise(family, clean, rng, snr_db=snr_db, fs=fs)
    noisy = clean + noise
    return clean, noisy, rng


def _eval_classical_filter(filter_cls, filter_kwargs: dict, order: int,
                            noisy: np.ndarray, clean: np.ndarray,
                            family: str, snr_db: float, seed: int,
                            method_name: str) -> dict:
    U = windowize(noisy, order)
    d = clean
    filt = filter_cls(order=order, **filter_kwargs)
    t0 = time.perf_counter()
    y, e = filt.run(U, d)
    dt = time.perf_counter() - t0
    ss = steady_state_mse(e)
    ct = convergence_time(e)
    return {
        "method": method_name,
        "family": family,
        "snr_db": snr_db,
        "seed": seed,
        "ss_mse": ss,
        "ss_mse_db": 10 * np.log10(ss + 1e-12),
        "ep_mse": float(np.mean(e ** 2)),
        "conv_time": ct,
        "inference_time_ms": dt * 1000,
    }


def eval_all_classical(families: Sequence[str] = ALL_FAMILIES,
                       snr_db_values: Sequence[float] = (0, 5, 10, 15, 20),
                       seeds: Sequence[int] = (0, 1, 2, 3, 4),
                       n: int = 2000,
                       fs: float = 8000.0,
                       order: int = 16) -> list[dict]:
    rows = []
    classical_methods = {
        "LMS": (LMS, {"mu": 0.01, "leakage": 1.0}),
        "NLMS": (NLMS, {"mu": 0.5}),
        "VSS-LMS (Kwong)": (VSSLMS, {"mu_max": 0.05, "alpha": 0.97, "gamma": 1e-3}),
        "VSS-LMS (Aboulnasr)": (AboulnasrMayyasVSS, {"mu_max": 0.1, "alpha": 0.97}),
        "VSS-LMS (Mathews)": (MathewsXieVSS, {"mu_max": 0.1, "alpha": 0.97}),
        "RLS": (RLS, {"forgetting": 0.995}),
        "LMP (p=1.5)": (LMP, {"mu": 0.01, "p": 1.5}),
        "Heuristic Scheduler": (HeuristicMuScheduler, {"mu_base": 0.01}),
        "Kalman Scheduler": (KalmanMuScheduler, {"mu_init": 0.01}),
    }

    for method_name, (cls, kwargs) in classical_methods.items():
        print(f"Evaluating {method_name}...")
        for fam in families:
            for snr_db in snr_db_values:
                for seed in seeds:
                    clean, noisy, _ = _make_signal_and_noise(fam, n, fs, snr_db, seed)
                    row = _eval_classical_filter(
                        cls, kwargs, order, noisy, clean,
                        fam, snr_db, seed, method_name)
                    rows.append(row)
    return rows


def eval_grid_searched_lms(families: Sequence[str] = ALL_FAMILIES,
                           snr_db_values: Sequence[float] = (0, 5, 10, 15, 20),
                           seeds: Sequence[int] = (0, 1, 2, 3, 4),
                           n: int = 2000,
                           fs: float = 8000.0,
                           order: int = 16) -> list[dict]:
    """LMS with per-family grid-searched mu on a held-out validation seed."""
    mu_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    rows = []

    for fam in families:
        for snr_db in snr_db_values:
            best_mu = None
            best_ss = float("inf")
            for mu in mu_grid:
                clean, noisy, _ = _make_signal_and_noise(fam, n, fs, snr_db, seed=9999)
                U = windowize(noisy, order)
                filt = LMS(order=order, mu=mu)
                _, e = filt.run(U, clean)
                ss = steady_state_mse(e)
                if ss < best_ss:
                    best_ss = ss
                    best_mu = mu

            for seed in seeds:
                clean, noisy, _ = _make_signal_and_noise(fam, n, fs, snr_db, seed)
                row = _eval_classical_filter(
                    LMS, {"mu": best_mu, "leakage": 1.0}, order, noisy, clean,
                    fam, snr_db, seed, f"LMS (grid, mu={best_mu:.0e})")
                row["method"] = "LMS (per-family tuned)"
                rows.append(row)
    return rows


def eval_cnn(families: Sequence[str] = ALL_FAMILIES,
             snr_db_values: Sequence[float] = (0, 5, 10, 15, 20),
             seeds: Sequence[int] = (0, 1, 2, 3, 4),
             n: int = 2000,
             fs: float = 8000.0,
             window: int = 64,
             device: str = "cpu") -> list[dict]:
    """Evaluate CNN denoiser trained on a mixture of train families."""
    rows = []
    print("Training CNN on mixed train families...")
    rng_train = np.random.default_rng(12345)
    train_clean_list, train_noisy_list = [], []
    for fam in TRAIN_FAMILIES:
        for _ in range(20):
            c = make_signal("multitone", n=n, fs=fs, rng=rng_train)
            noise = make_noise(fam, c, rng_train, snr_db=10.0, fs=fs)
            train_clean_list.append(c)
            train_noisy_list.append(c + noise)
    train_clean = np.concatenate(train_clean_list)
    train_noisy = np.concatenate(train_noisy_list)
    cnn_model = train_cnn(train_noisy, train_clean, window=window,
                          epochs=50, device=device, verbose=True)

    print("Evaluating CNN...")
    for fam in families:
        for snr_db in snr_db_values:
            for seed in seeds:
                clean, noisy, _ = _make_signal_and_noise(fam, n, fs, snr_db, seed)
                t0 = time.perf_counter()
                pred = cnn_predict(cnn_model, noisy, window, device=device)
                dt = time.perf_counter() - t0
                e = clean - pred
                ss = steady_state_mse(e)
                ct = convergence_time(e)
                rows.append({
                    "method": "CNN (supervised)",
                    "family": fam,
                    "snr_db": snr_db,
                    "seed": seed,
                    "ss_mse": ss,
                    "ss_mse_db": 10 * np.log10(ss + 1e-12),
                    "ep_mse": float(np.mean(e ** 2)),
                    "conv_time": ct,
                    "inference_time_ms": dt * 1000,
                })
    return rows


def eval_rl_policies(model_paths: dict[str, str],
                     families: Sequence[str] = ALL_FAMILIES,
                     snr_db_values: Sequence[float] = (0, 5, 10, 15, 20),
                     eval_seeds: Sequence[int] = (0, 1, 2, 3, 4),
                     episode_len: int = 2000,
                     state_window: int = 16,
                     filter_order: int = 16,
                     signal_kind: str = "multitone",
                     device: str = "cpu") -> list[dict]:
    """Evaluate multiple RL policy checkpoints.

    model_paths: {method_name: path_to_zip}
    Each policy is evaluated on all (family, snr, seed) combinations.
    """
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    rows = []
    for method_name, model_path in model_paths.items():
        print(f"Evaluating RL policy: {method_name} from {model_path}")
        try:
            model = PPO.load(model_path, device=device)
            is_rec = False
        except Exception:
            model = RecurrentPPO.load(model_path, device=device)
            is_rec = True

        for fam in families:
            for snr_db in snr_db_values:
                for seed in eval_seeds:
                    env_cfg = EnvConfig(episode_len=episode_len,
                                        state_window=state_window,
                                        filter_order=filter_order)
                    env = AdaptiveFilterEnv(env_cfg, fixed_family=fam,
                                           fixed_signal=signal_kind,
                                           fixed_snr_db=snr_db, seed=seed)
                    obs, _ = env.reset(seed=seed)
                    lstm_state = None
                    episode_starts = np.ones((1,), dtype=bool) if is_rec else None
                    errors = []
                    done = False
                    steps = 0
                    t0 = time.perf_counter()
                    while not done and steps < episode_len:
                        if is_rec:
                            action, lstm_state = model.predict(
                                obs[None, :], state=lstm_state,
                                episode_start=episode_starts,
                                deterministic=True)
                            episode_starts = np.zeros((1,), dtype=bool)
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                        obs, _, term, trunc, info = env.step(
                            action[0] if is_rec else action)
                        errors.append(env.last_e)
                        done = term or trunc
                        steps += 1
                    dt = time.perf_counter() - t0

                    errors = np.array(errors)
                    ss = steady_state_mse(errors)
                    ct = convergence_time(errors)
                    rows.append({
                        "method": method_name,
                        "family": fam,
                        "snr_db": snr_db,
                        "seed": seed,
                        "ss_mse": ss,
                        "ss_mse_db": 10 * np.log10(ss + 1e-12),
                        "ep_mse": float(np.mean(errors ** 2)),
                        "conv_time": ct,
                        "inference_time_ms": dt * 1000,
                    })
    return rows


def run_full_evaluation(output_csv: str = "results/full_evaluation.csv",
                       model_paths: dict | None = None,
                       snr_db_values: Sequence[float] = (0, 5, 10, 15, 20),
                       seeds: Sequence[int] = (0, 1, 2, 3, 4),
                       device: str = "cpu") -> list[dict]:
    """Run the full evaluation suite and save to CSV."""
    all_rows = []

    print("=" * 60)
    print("PHASE 1: Classical baselines")
    print("=" * 60)
    all_rows.extend(eval_all_classical(snr_db_values=snr_db_values, seeds=seeds))

    print("\n" + "=" * 60)
    print("PHASE 2: Grid-searched LMS")
    print("=" * 60)
    all_rows.extend(eval_grid_searched_lms(snr_db_values=snr_db_values, seeds=seeds))

    print("\n" + "=" * 60)
    print("PHASE 3: CNN baseline")
    print("=" * 60)
    all_rows.extend(eval_cnn(snr_db_values=snr_db_values, seeds=seeds, device=device))

    if model_paths:
        print("\n" + "=" * 60)
        print("PHASE 4: RL policies")
        print("=" * 60)
        all_rows.extend(eval_rl_policies(model_paths, snr_db_values=snr_db_values,
                                          eval_seeds=seeds, device=device))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if all_rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {output_csv}")
    return all_rows

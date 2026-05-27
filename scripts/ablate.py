"""Ablation sweep for the v2 Meta-RL setup.

Trains a small grid of Meta-RL policies (1-2 seeds each, short budget) and
evaluates each on a fixed synthetic SNR x family grid. Writes ablations.csv
with per-cell SS-MSE-dB so you can drop a line in the paper showing each
design choice contributes.

Axes ablated (one at a time vs the v2 default):
  reward      : log_mse (default), softplus, neg_abs
  curriculum  : on (default), off (uniform family weights)
  lstm_hidden : 256 (default), 128, 64
  mu_max      : 2.0 (default), 1.0
"""
from __future__ import annotations
import argparse
import csv
import os
import time
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.envs import AdaptiveFilterEnv, EnvConfig
from src.noise.families import TRAIN_FAMILIES
from src.eval.metrics import steady_state_mse


CURRICULUM = {"gaussian": 1.0, "colored": 1.0, "impulsive": 2.0,
              "time_varying": 2.0, "regime_switch": 3.0}


def _cfg(reward="log_mse", curric=True, mu_max=2.0, fs=360.0, ep_len=2000):
    weights = ([CURRICULUM.get(f, 1.0) for f in TRAIN_FAMILIES]
               if curric else None)
    return EnvConfig(fs=fs, episode_len=ep_len, filter_order=16, state_window=16,
                     mu_min=0.005, mu_max=mu_max, leakage_min=0.80,
                     train_families=tuple(TRAIN_FAMILIES),
                     family_weights=tuple(weights) if weights else None,
                     reward_kind=reward)


def _train(cfg: EnvConfig, lstm_hidden: int, total_steps: int, seed: int,
           n_envs: int, device: str):
    vec = DummyVecEnv([
        (lambda i=i: Monitor(AdaptiveFilterEnv(cfg, seed=seed + i)))
        for i in range(n_envs)
    ])
    pk = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
              lstm_hidden_size=lstm_hidden, n_lstm_layers=1,
              shared_lstm=False, enable_critic_lstm=True)
    model = RecurrentPPO("MlpLstmPolicy", vec, n_steps=1024, batch_size=128,
                         ent_coef=0.005, verbose=0, device=device, seed=seed,
                         policy_kwargs=pk)
    model.learn(total_timesteps=total_steps)
    return model


def _eval(model, fs, ep_len, families, snrs, seeds):
    rows = []
    for fam in families:
        for snr in snrs:
            for sd in seeds:
                cfg = EnvConfig(fs=fs, episode_len=ep_len, filter_order=16)
                env = AdaptiveFilterEnv(cfg, fixed_family=fam,
                                        fixed_signal="multitone",
                                        fixed_snr_db=snr, seed=sd)
                obs, _ = env.reset(seed=sd)
                state = None; starts = np.ones((1,), dtype=bool)
                errs, done = [], False
                while not done:
                    a, state = model.predict(obs[None, :], state=state,
                                              episode_start=starts,
                                              deterministic=True)
                    starts = np.zeros((1,), dtype=bool)
                    obs, _, term, trunc, _ = env.step(a[0])
                    errs.append(env.last_e); done = term or trunc
                ss = steady_state_mse(np.array(errs))
                rows.append(dict(family=fam, snr_db=snr, seed=sd,
                                 ss_mse_db=10 * np.log10(ss + 1e-12)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/v2_ablations/ablations.csv")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = [
        ("default",        dict(reward="log_mse", curric=True,  mu_max=2.0), 256),
        ("reward=softplus",dict(reward="softplus", curric=True, mu_max=2.0), 256),
        ("reward=neg_abs", dict(reward="neg_abs", curric=True,  mu_max=2.0), 256),
        ("no_curriculum",  dict(reward="log_mse", curric=False, mu_max=2.0), 256),
        ("lstm=128",       dict(reward="log_mse", curric=True,  mu_max=2.0), 128),
        ("lstm=64",        dict(reward="log_mse", curric=True,  mu_max=2.0),  64),
        ("mu_max=1.0",     dict(reward="log_mse", curric=True,  mu_max=1.0), 256),
    ]
    families = list(TRAIN_FAMILIES)
    snrs = [0, 10, 20]
    eval_seeds = [0, 1, 2]

    all_rows = []
    for tag, kwargs, lstm in configs:
        t0 = time.perf_counter()
        print(f"[ablate] training {tag} (lstm={lstm}) ...")
        cfg = _cfg(**kwargs)
        model = _train(cfg, lstm, args.total_steps, args.seed,
                       args.n_envs, args.device)
        rows = _eval(model, cfg.fs, cfg.episode_len, families, snrs, eval_seeds)
        for r in rows:
            r["ablation"] = tag
            all_rows.append(r)
        print(f"[ablate] {tag} done in {(time.perf_counter() - t0) / 60:.1f} min")

    keys = sorted({k for r in all_rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

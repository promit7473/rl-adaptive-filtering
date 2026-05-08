"""Ablations:
  (a) single-noise training (no meta) — train one PPO per train family.
  (b) MLP (non-recurrent) policy — uses standard PPO instead of RecurrentPPO.

Compares against the meta-trained policy on the OOD test set.
"""
from __future__ import annotations
import os, sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs import AdaptiveFilterEnv, EnvConfig
from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
from src.eval import ensure_dir
from src.agents.train_ppo import train as train_recurrent
from src.agents.eval_ppo import evaluate_policy


def train_mlp(total_timesteps, n_envs, episode_len, fixed_family, out_dir, tag, seed, device):
    """Standard PPO (no recurrence) ablation."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def thunk_factory(i):
        def _thunk():
            env = AdaptiveFilterEnv(EnvConfig(episode_len=episode_len),
                                     fixed_family=fixed_family, seed=seed + i)
            return Monitor(env)
        return _thunk

    vec = DummyVecEnv([thunk_factory(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", vec, learning_rate=3e-4, n_steps=256, batch_size=256,
                verbose=0, device=device, seed=seed,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    path = os.path.join(out_dir, f"ppo_mlp_{tag}.zip")
    ensure_dir(out_dir)
    model.save(path)
    return path


def eval_mlp(model_path, families, seeds, snr_db, episode_len, state_window, filter_order, device):
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device=device)
    rows = []
    for fam in families:
        for s in seeds:
            env = AdaptiveFilterEnv(EnvConfig(episode_len=episode_len,
                                              state_window=state_window,
                                              filter_order=filter_order),
                                     fixed_family=fam, fixed_snr_db=snr_db, seed=s)
            obs, _ = env.reset(seed=s)
            errs = []; done = False; steps = 0
            while not done and steps < episode_len:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(a)
                errs.append(env.last_e)
                done = term or trunc; steps += 1
            errs = np.array(errs)
            ss = float(np.mean(errs[-int(len(errs) * 0.25):] ** 2))
            rows.append({
                "method": "PPO-MLP", "family": fam, "seed": s,
                "ss_mse": ss, "ss_mse_db": 10 * np.log10(ss + 1e-12),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=120_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--episode-len", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="results/ablations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--skip-mlp", action="store_true")
    args = p.parse_args()

    ensure_dir(args.out_dir)
    rows_all = []

    # (a) single-noise PPO recurrent — one per train family
    if not args.skip_single:
        for fam in TRAIN_FAMILIES:
            print(f"[ablation:single] training PPO on family={fam}")
            _, _, path = train_recurrent(
                total_timesteps=args.total_timesteps,
                n_envs=args.n_envs, episode_len=args.episode_len,
                fixed_family=fam, out_dir=args.out_dir, tag=f"single_{fam}",
                seed=args.seed, device=args.device,
            )
            print(f"  evaluating on full OOD set")
            rows = evaluate_policy(path, list(TRAIN_FAMILIES) + list(OOD_FAMILIES),
                                    seeds=(0, 1, 2),
                                    episode_len=args.episode_len, device=args.device)
            for r in rows:
                r["method"] = f"PPO-single({fam})"
            rows_all.extend(rows)

    # (b) MLP policy on meta task (no recurrence)
    if not args.skip_mlp:
        print(f"[ablation:mlp] training PPO-MLP on meta task")
        path = train_mlp(args.total_timesteps, args.n_envs, args.episode_len,
                          fixed_family=None, out_dir=args.out_dir,
                          tag="meta", seed=args.seed, device=args.device)
        rows = eval_mlp(path, list(TRAIN_FAMILIES) + list(OOD_FAMILIES),
                        seeds=(0, 1, 2), snr_db=10.0,
                        episode_len=args.episode_len, state_window=16,
                        filter_order=16, device=args.device)
        rows_all.extend(rows)

    df = pd.DataFrame(rows_all)
    df.to_csv(os.path.join(args.out_dir, "ablations_per_seed.csv"), index=False)
    summary = df.groupby(["method", "family"]).agg(
        mse_db_mean=("ss_mse_db", "mean"),
        mse_db_std=("ss_mse_db", "std"),
    ).reset_index()
    summary.to_csv(os.path.join(args.out_dir, "ablations_summary.csv"), index=False)
    print("\nDone. summary:")
    pivot = summary.pivot(index="family", columns="method", values="mse_db_mean").round(2)
    print(pivot.to_string())


if __name__ == "__main__":
    main()

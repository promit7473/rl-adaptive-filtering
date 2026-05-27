"""Train v2 Meta-RL adaptive filter (RecurrentPPO / PPO-MLP).

Defaults are tuned for the SPL submission:
  - 360 Hz native sampling (matches MIT-BIH ECG, no resample at eval time)
  - log-MSE reward (non-saturating; better gradient on hard burst noise)
  - mu in [0.005, 2.0], leakage in [0.80, 1.0]
  - LSTM-256 + (128,128) heads (~310k params)
  - Curriculum: 50 percent of episodes drawn from transient noise
    (regime_switch + impulsive + time_varying), 50 percent from
    stationary (gaussian + colored).
  - 5+ seeds; 1.5M steps per seed (RecurrentPPO) for budgeted overkill.

Usage:
    python3 scripts/train.py --policy meta --n-seeds 5 --total-steps 1500000
    python3 scripts/train.py --policy mlp  --n-seeds 5 --total-steps  800000
"""
from __future__ import annotations
import argparse
import csv
import os
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from src.envs import AdaptiveFilterEnv, EnvConfig
from src.noise.families import TRAIN_FAMILIES


# Curriculum: oversample transient/non-stationary regimes that hurt classical filters.
CURRICULUM_WEIGHTS = {
    "gaussian":      1.0,
    "colored":       1.0,
    "impulsive":     2.0,
    "time_varying":  2.0,
    "regime_switch": 3.0,
}


SIGNAL_KINDS = ("multitone", "am", "sine",
                "ecg_like", "random_pulses", "square_burst")
# Oversample transient-class signals so policy learns to distinguish
# high-amplitude desired-signal events from noise spikes.
SIGNAL_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def make_env_cfg(args) -> EnvConfig:
    weights = [CURRICULUM_WEIGHTS.get(f, 1.0) for f in TRAIN_FAMILIES]
    return EnvConfig(
        fs=args.fs,
        episode_len=args.episode_len,
        filter_order=args.filter_order,
        state_window=args.window,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        leakage_min=args.leakage_min,
        snr_db_options=tuple(args.snrs),
        train_families=tuple(TRAIN_FAMILIES),
        family_weights=tuple(weights),
        signal_kinds=SIGNAL_KINDS,
        signal_weights=SIGNAL_WEIGHTS,
        reward_kind=args.reward,
        reward_scale=1.0,
        normalize_input=True,
        convergence_bonus=args.convergence_bonus,
        robust_alpha=args.robust_alpha,
        robust_beta=args.robust_beta,
        no_reward_clip=args.no_reward_clip,
        terminal_ss_weight=args.terminal_ss_weight,
    )


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "episode_ss_mse" in info:
                self.records.append({
                    "step": int(self.num_timesteps),
                    "ss_mse": float(info["episode_ss_mse"]),
                    "ss_mse_db": float(10 * np.log10(info["episode_ss_mse"] + 1e-12)),
                    "ep_mse": float(info["episode_mse"]),
                    "family": info["task"]["family"],
                    "snr_db": info["task"]["snr_db"],
                    "divergence_count": info.get("divergence_count", 0),
                })
        return True


def _make_vec(cfg: EnvConfig, n_envs: int, seed: int, subproc: bool):
    def factory(i):
        def _f():
            return Monitor(AdaptiveFilterEnv(cfg, seed=seed + i))
        return _f
    fns = [factory(i) for i in range(n_envs)]
    return SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)


def train_one(args, seed: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cfg = make_env_cfg(args)
    vec = _make_vec(cfg, args.n_envs, seed, args.subproc)

    if args.policy == "meta":
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=getattr(args, 'n_lstm_layers', 2),
            shared_lstm=False,
            enable_critic_lstm=True,
        )
        model = RecurrentPPO(
            "MlpLstmPolicy", vec,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.batch_size, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
            verbose=1, device=args.device, seed=seed,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(out_dir, "tb"),
        )
    else:
        policy_kwargs = dict(net_arch=dict(pi=[args.mlp_hidden, args.mlp_hidden],
                                           vf=[args.mlp_hidden, args.mlp_hidden]))
        model = PPO(
            "MlpPolicy", vec,
            learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.batch_size, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range,
            ent_coef=args.ent_coef, verbose=1, device=args.device,
            seed=seed, policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(out_dir, "tb"),
        )

    cb_metrics = EpisodeMetricsCallback()
    cb_ckpt = CheckpointCallback(
        save_freq=max(args.total_steps // (10 * max(1, args.n_envs)), 1000),
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix=f"{args.policy}_seed{seed}",
    )
    model.learn(total_timesteps=args.total_steps,
                callback=[cb_metrics, cb_ckpt], progress_bar=False)
    final_path = os.path.join(out_dir, f"ppo_{args.policy}_seed{seed}_final.zip")
    model.save(final_path)

    rec_path = os.path.join(out_dir, f"train_records_seed{seed}.csv")
    if cb_metrics.records:
        with open(rec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cb_metrics.records[0].keys()))
            w.writeheader()
            w.writerows(cb_metrics.records)
    print(f"[seed {seed}] saved {final_path}")
    return final_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["meta", "mlp"], default="meta")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--total-steps", type=int, default=1_500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--device", default="auto")
    # env
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--episode-len", type=int, default=2000)
    p.add_argument("--filter-order", type=int, default=16)
    p.add_argument("--state-window", type=int, default=16)
    p.add_argument("--mu-min", type=float, default=0.005)
    p.add_argument("--mu-max", type=float, default=2.0)
    p.add_argument("--leakage-min", type=float, default=0.70)
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--reward", choices=["softplus", "log_mse", "neg_abs"],
                   default="log_mse")
    p.add_argument("--convergence-bonus", type=float, default=0.15)
    p.add_argument("--terminal-ss-weight", type=float, default=0.5)
    p.add_argument("--robust-alpha", type=float, default=0.05)
    p.add_argument("--robust-beta", type=float, default=0.02)
    p.add_argument("--no-reward-clip", action="store_true", default=True)
    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--n-lstm-layers", type=int, default=2)
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--window", type=int, default=4)
    args = p.parse_args()

    out_dir = args.out_dir or f"results/v2_{args.policy}"
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(args.n_seeds):
        seed = args.base_seed + 100 * i
        paths.append(train_one(args, seed=seed, out_dir=out_dir))

    with open(os.path.join(out_dir, "models.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"\nDone. {len(paths)} models in {out_dir}")


if __name__ == "__main__":
    main()

"""Train v3 hybrid BPTT+RL adaptive filter.

Three training modes:
  1. hybrid  — BPTT through differentiable NLMS + PPO RL fine-tuning
  2. bptt   — Pure BPTT (like Meta-AF but with aux losses)
  3. rl     — Pure RL (RecurrentPPO on V2 env with 11-D features + VecNormalize)

Plus multi-seed and multi-architecture support.

Usage:
    # Hybrid BPTT+RL (recommended — the best of both worlds)
    PYTHONPATH=. python3 scripts/train_v3.py --mode hybrid --n-seeds 5

    # Pure BPTT (Meta-AF killer)
    PYTHONPATH=. python3 scripts/train_v3.py --mode bptt --controller hybrid --n-iters 8000

    # Pure RL with enhanced env
    PYTHONPATH=. python3 scripts/train_v3.py --mode rl --n-seeds 5 --total-steps 2000000

    # Transformer controller (long-range attention)
    PYTHONPATH=. python3 scripts/train_v3.py --mode hybrid --controller transformer
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import numpy as np
import torch

from src.agents.hybrid_trainer import train_hybrid, HybridTrainConfig
from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES


CURRICULUM_WEIGHTS = {
    "gaussian": 1.0, "colored": 1.0, "impulsive": 2.0,
    "time_varying": 2.0, "regime_switch": 3.0,
    "alpha_stable": 2.0, "burst": 2.0, "chirp_interferer": 1.5,
}


def train_rl(args):
    """Train with RecurrentPPO on the V2 env with VecNormalize."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    SIGNAL_KINDS = ("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst")
    SIGNAL_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)

    all_families = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    fam_weights = [CURRICULUM_WEIGHTS.get(f, 1.0) for f in all_families]

    env_cfg = EnvConfigV2(
        fs=args.fs,
        episode_len=args.episode_len,
        filter_order=args.filter_order,
        state_window=args.state_window,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        leakage_min=args.leakage_min,
        leakage_max=1.0,
        snr_db_options=tuple(args.snrs),
        train_families=tuple(all_families),
        family_weights=tuple(fam_weights),
        signal_kinds=SIGNAL_KINDS,
        signal_weights=SIGNAL_WEIGHTS,
        reward_kind=args.reward,
        convergence_bonus=args.convergence_bonus,
        robust_alpha=args.robust_alpha,
        robust_beta=args.robust_beta,
    )

    def make_env(seed):
        def _thunk():
            return Monitor(AdaptiveFilterEnvV2(env_cfg, seed=seed))
        return _thunk

    out_dir = args.out_dir or f"results/v3_rl"
    os.makedirs(out_dir, exist_ok=True)

    for si in range(args.n_seeds):
        seed = args.base_seed + si * 100
        tag = f"rl_seed{seed}"
        print(f"\n{'='*60}")
        print(f"[rl] Training seed {si+1}/{args.n_seeds}: seed={seed}")
        print(f"{'='*60}\n")

        fns = [make_env(seed + i) for i in range(args.n_envs)]
        vec = DummyVecEnv(fns)
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0,
                           gamma=args.gamma, epsilon=1e-8)

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=2,
            shared_lstm=False,
            enable_critic_lstm=True,
        )

        model = RecurrentPPO(
            "MlpLstmPolicy", vec,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=args.device,
            seed=seed,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(out_dir, "tb"),
        )

        class MetricsCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.records = []
            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if isinstance(info, dict) and "episode_ss_mse" in info:
                        self.records.append({
                            "step": int(self.num_timesteps),
                            "ss_mse": float(info["episode_ss_mse"]),
                            "ss_mse_db": float(10 * np.log10(info["episode_ss_mse"] + 1e-12)),
                            "ep_mse": float(info["episode_mse"]),
                            "family": info.get("task", {}).get("family", ""),
                            "snr_db": info.get("task", {}).get("snr_db", float("nan")),
                        })
                return True

        cb_metrics = MetricsCallback()
        cb_ckpt = CheckpointCallback(
            save_freq=max(args.total_steps // (10 * max(1, args.n_envs)), 1000),
            save_path=os.path.join(out_dir, "checkpoints"),
            name_prefix=f"v3_{tag}",
        )

        model.learn(total_timesteps=args.total_steps,
                     callback=[cb_metrics, cb_ckpt], progress_bar=False)

        final_path = os.path.join(out_dir, f"v3_{tag}_final.zip")
        model.save(final_path)
        vec_path = os.path.join(out_dir, f"v3_{tag}_vecnormalize.pkl")
        vec.save(vec_path)

        if cb_metrics.records:
            rec_path = os.path.join(out_dir, f"train_records_seed{seed}.csv")
            with open(rec_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cb_metrics.records[0].keys()))
                w.writeheader()
                w.writerows(cb_metrics.records)

        print(f"[rl] seed={seed} saved {final_path}")


def train_bptt(args):
    """Train pure BPTT (like Meta-AF but with aux losses and wider leakage range)."""
    out_dir = args.out_dir or "results/v3_bptt"
    cfg = HybridTrainConfig(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        episode_len=args.episode_len,
        fs=args.fs,
        filter_order=args.filter_order,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        lam_min=args.leakage_min,
        lam_max=1.0,
        lr=args.lr,
        trunc_bptt=args.trunc_bptt,
        bptt_loss_weight=1.0,
        rl_loss_weight=0.0,
        aux_error_weight=args.aux_error_weight,
        aux_task_weight=args.aux_task_weight,
        convergence_bonus=args.convergence_bonus,
        robust_alpha=args.robust_alpha,
        controller_type=args.controller,
        lstm_hidden=args.lstm_hidden,
        n_lstm_layers=args.n_lstm_layers,
        device=args.device,
        seed=args.base_seed,
        save_every=args.save_every,
        eval_every=args.eval_every,
        curriculum_ramp_iters=args.curriculum_ramp,
        rl_phase_start=args.n_iters + 1,
    )
    for si in range(args.n_seeds):
        seed = args.base_seed + si * 100
        cfg.seed = seed
        sub_dir = os.path.join(out_dir, f"seed{seed}")
        print(f"\n[bptt] Training seed {si+1}/{args.n_seeds}: seed={seed}")
        train_hybrid(cfg, out_dir=sub_dir)


def train_hybrid_mode(args):
    """Train hybrid BPTT+RL."""
    out_dir = args.out_dir or "results/v3_hybrid"
    cfg = HybridTrainConfig(
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        episode_len=args.episode_len,
        fs=args.fs,
        filter_order=args.filter_order,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        lam_min=args.leakage_min,
        lam_max=1.0,
        lr=args.lr,
        trunc_bptt=args.trunc_bptt,
        bptt_loss_weight=args.bptt_weight,
        rl_loss_weight=args.rl_weight,
        aux_error_weight=args.aux_error_weight,
        aux_task_weight=args.aux_task_weight,
        convergence_bonus=args.convergence_bonus,
        robust_alpha=args.robust_alpha,
        controller_type=args.controller,
        lstm_hidden=args.lstm_hidden,
        n_lstm_layers=args.n_lstm_layers,
        device=args.device,
        seed=args.base_seed,
        save_every=args.save_every,
        eval_every=args.eval_every,
        curriculum_ramp_iters=args.curriculum_ramp,
        rl_phase_start=args.rl_phase_start,
    )
    for si in range(args.n_seeds):
        seed = args.base_seed + si * 100
        cfg.seed = seed
        sub_dir = os.path.join(out_dir, f"seed{seed}")
        print(f"\n[hybrid] Training seed {si+1}/{args.n_seeds}: seed={seed}")
        train_hybrid(cfg, out_dir=sub_dir)


def main():
    p = argparse.ArgumentParser(description="V3 training: hybrid BPTT+RL adaptive filter")
    p.add_argument("--mode", choices=["hybrid", "bptt", "rl"], default="hybrid")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)

    # Environment
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--episode-len", type=int, default=4000)
    p.add_argument("--filter-order", type=int, default=16)
    p.add_argument("--state-window", type=int, default=32)
    p.add_argument("--mu-min", type=float, default=0.005)
    p.add_argument("--mu-max", type=float, default=2.0)
    p.add_argument("--leakage-min", type=float, default=0.70)
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--reward", choices=["shaped_log_mse", "log_mse", "neg_abs"],
                    default="shaped_log_mse")
    p.add_argument("--convergence-bonus", type=float, default=0.15)
    p.add_argument("--robust-alpha", type=float, default=0.05)
    p.add_argument("--robust-beta", type=float, default=0.02)

    # BPTT / Hybrid
    p.add_argument("--n-iters", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--trunc-bptt", type=int, default=64)
    p.add_argument("--bptt-weight", type=float, default=1.0)
    p.add_argument("--rl-weight", type=float, default=0.3)
    p.add_argument("--aux-error-weight", type=float, default=0.2)
    p.add_argument("--aux-task-weight", type=float, default=0.1)
    p.add_argument("--curriculum-ramp", type=int, default=2000)
    p.add_argument("--rl-phase-start", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=250)

    # RL mode
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size-rl", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)

    # Architecture
    p.add_argument("--controller", choices=["hybrid", "lstm", "transformer"],
                    default="hybrid")
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--n-lstm-layers", type=int, default=2)

    # General
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    if not hasattr(args, 'batch_size_rl'):
        pass

    if args.mode == "hybrid":
        train_hybrid_mode(args)
    elif args.mode == "bptt":
        train_bptt(args)
    else:
        train_rl(args)


if __name__ == "__main__":
    main()

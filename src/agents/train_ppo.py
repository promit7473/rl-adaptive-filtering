"""Train Recurrent PPO (RL² style) and PPO-MLP on the meta adaptive-filter task.

Supports multi-seed training for proper statistical evaluation.
"""
from __future__ import annotations
import os
import csv
from typing import Sequence
import numpy as np
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from ..envs import AdaptiveFilterEnv, EnvConfig


def make_env_fn(env_cfg: EnvConfig, fixed_family: str | None, seed: int):
    def _thunk():
        env = AdaptiveFilterEnv(env_cfg, fixed_family=fixed_family, seed=seed)
        return Monitor(env)
    return _thunk


def build_vec_env(env_cfg: EnvConfig, n_envs: int, fixed_family: str | None,
                  seed: int, use_subproc: bool = False):
    fns = [make_env_fn(env_cfg, fixed_family, seed=seed + i) for i in range(n_envs)]
    return (SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns))


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.records: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode_ss_mse" in info:
                rec = {
                    "step": int(self.num_timesteps),
                    "ss_mse": float(info["episode_ss_mse"]),
                    "ss_mse_db": float(10 * np.log10(info["episode_ss_mse"] + 1e-12)),
                    "ep_mse": float(info["episode_mse"]),
                    "family": info.get("task", {}).get("family", ""),
                    "snr_db": info.get("task", {}).get("snr_db", float("nan")),
                    "divergence_count": info.get("divergence_count", 0),
                }
                self.records.append(rec)
                if self.verbose and len(self.records) % 50 == 0:
                    print(f"[ep {len(self.records)}] step={rec['step']:>8d}  "
                          f"ss_mse_db={rec['ss_mse_db']:+.2f}  fam={rec['family']}")
        return True


def train_mlp(total_timesteps: int = 600_000,
              n_envs: int = 8,
              episode_len: int = 2000,
              state_window: int = 16,
              filter_order: int = 16,
              learning_rate: float = 3e-4,
              n_steps: int = 2048,
              batch_size: int = 256,
              gamma: float = 0.99,
              gae_lambda: float = 0.95,
              clip_range: float = 0.2,
              ent_coef: float = 0.01,
              hidden_size: int = 128,
              fixed_family: str | None = None,
              out_dir: str = "results/ppo_mlp",
              tag: str = "mlp",
              seed: int = 42,
              device: str = "auto",
              use_subproc: bool = False) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    env_cfg = EnvConfig(episode_len=episode_len,
                        state_window=state_window,
                        filter_order=filter_order)
    vec_env = build_vec_env(env_cfg, n_envs, fixed_family, seed, use_subproc=use_subproc)

    policy_kwargs = dict(
        net_arch=dict(pi=[hidden_size, hidden_size], vf=[hidden_size, hidden_size]),
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    cb_metrics = EpisodeMetricsCallback(verbose=1)
    cb_ckpt = CheckpointCallback(save_freq=max(100_000 // max(1, n_envs), 1000),
                                 save_path=os.path.join(out_dir, "checkpoints"),
                                 name_prefix=f"ppo_{tag}")

    model.learn(total_timesteps=total_timesteps,
                callback=[cb_metrics, cb_ckpt],
                progress_bar=False,
                tb_log_name=tag)
    final_path = os.path.join(out_dir, f"ppo_{tag}_final.zip")
    model.save(final_path)

    rec_path = os.path.join(out_dir, f"train_records_{tag}.csv")
    _save_records(cb_metrics.records, rec_path)
    print(f"Saved {final_path} and {rec_path}")
    return model, cb_metrics.records, final_path


def train_recurrent(total_timesteps: int = 1_200_000,
                    n_envs: int = 8,
                    episode_len: int = 2000,
                    state_window: int = 16,
                    filter_order: int = 16,
                    hidden_size: int = 128,
                    learning_rate: float = 3e-4,
                    n_steps: int = 2048,
                    batch_size: int = 256,
                    gamma: float = 0.99,
                    gae_lambda: float = 0.95,
                    clip_range: float = 0.2,
                    ent_coef: float = 0.01,
                    fixed_family: str | None = None,
                    out_dir: str = "results/ppo_meta",
                    tag: str = "meta",
                    seed: int = 42,
                    device: str = "auto",
                    use_subproc: bool = False) -> tuple:
    os.makedirs(out_dir, exist_ok=True)
    env_cfg = EnvConfig(episode_len=episode_len,
                        state_window=state_window,
                        filter_order=filter_order)
    vec_env = build_vec_env(env_cfg, n_envs, fixed_family, seed, use_subproc=use_subproc)

    policy_kwargs = dict(
        net_arch=dict(pi=[hidden_size, hidden_size], vf=[hidden_size, hidden_size]),
        lstm_hidden_size=hidden_size,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        seed=seed,
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    cb_metrics = EpisodeMetricsCallback(verbose=1)
    cb_ckpt = CheckpointCallback(save_freq=max(100_000 // max(1, n_envs), 1000),
                                 save_path=os.path.join(out_dir, "checkpoints"),
                                 name_prefix=f"ppo_{tag}")

    model.learn(total_timesteps=total_timesteps,
                callback=[cb_metrics, cb_ckpt],
                progress_bar=False,
                tb_log_name=tag)
    final_path = os.path.join(out_dir, f"ppo_{tag}_final.zip")
    model.save(final_path)

    rec_path = os.path.join(out_dir, f"train_records_{tag}.csv")
    _save_records(cb_metrics.records, rec_path)
    print(f"Saved {final_path} and {rec_path}")
    return model, cb_metrics.records, final_path


def train_multi_seed(n_seeds: int = 5,
                     policy_type: str = "mlp",
                     base_seed: int = 42,
                     out_dir: str = "results",
                     **kwargs) -> dict:
    """Train multiple policies with different seeds for statistical evaluation."""
    results = {}
    for i in range(n_seeds):
        seed = base_seed + i * 100
        tag = f"{policy_type}_seed{seed}"
        sub_dir = os.path.join(out_dir, f"ppo_{policy_type}")
        print(f"\n{'='*60}")
        print(f"Training seed {i+1}/{n_seeds}: seed={seed}, tag={tag}")
        print(f"{'='*60}\n")

        if policy_type == "mlp":
            model, records, path = train_mlp(
                seed=seed, tag=tag, out_dir=sub_dir, **kwargs)
        else:
            model, records, path = train_recurrent(
                seed=seed, tag=tag, out_dir=sub_dir, **kwargs)
        results[seed] = {"model": model, "records": records, "path": path}
    return results


def _save_records(records: list[dict], path: str) -> None:
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

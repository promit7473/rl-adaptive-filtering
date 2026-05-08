"""Evaluate a trained PPO policy on a set of noise families.

Supports both PPO-MLP and RecurrentPPO (Meta-RL) models.
Returns per-(family, seed) metrics with proper steady-state MSE.
"""
from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from ..envs import AdaptiveFilterEnv, EnvConfig


def _is_recurrent(model) -> bool:
    return isinstance(model, RecurrentPPO)


def evaluate_policy(model_path: str,
                    families,
                    seeds=(0, 1, 2, 3, 4),
                    snr_db: float = 10.0,
                    episode_len: int = 2000,
                    state_window: int = 16,
                    filter_order: int = 16,
                    signal_kind: str = "multitone",
                    device: str = "cpu",
                    **kwargs) -> list[dict]:
    model = PPO.load(model_path, device=device) if "meta" not in model_path \
        else RecurrentPPO.load(model_path, device=device)
    is_rec = _is_recurrent(model)
    rows = []
    for fam in families:
        for s in seeds:
            env_cfg = EnvConfig(episode_len=episode_len,
                                state_window=state_window,
                                filter_order=filter_order)
            env = AdaptiveFilterEnv(env_cfg, fixed_family=fam,
                                   fixed_signal=signal_kind,
                                   fixed_snr_db=snr_db, seed=s)
            obs, _ = env.reset(seed=s)
            lstm_state = None
            episode_starts = np.ones((1,), dtype=bool)
            errors = []
            done = False
            steps = 0
            while not done and steps < episode_len:
                if is_rec:
                    action, lstm_state = model.predict(
                        obs[None, :], state=lstm_state,
                        episode_start=episode_starts,
                        deterministic=kwargs.get("deterministic", True),
                    )
                else:
                    action, _ = model.predict(
                        obs, deterministic=kwargs.get("deterministic", True),
                    )
                obs, _, term, trunc, info = env.step(action[0] if is_rec else action)
                errors.append(env.last_e)
                if is_rec:
                    episode_starts = np.zeros((1,), dtype=bool)
                done = term or trunc
                steps += 1
            errors = np.array(errors)
            ss = float(np.mean(errors[-int(len(errors) * 0.25):] ** 2))
            mse = float(np.mean(errors ** 2))
            rows.append({
                "family": fam,
                "seed": s,
                "ss_mse": ss,
                "ss_mse_db": 10 * np.log10(ss + 1e-12),
                "ep_mse": mse,
                "n_steps": steps,
            })
    return rows


def evaluate_policy_full_curves(model_path: str, families,
                                seeds=(0, 1, 2),
                                **kwargs) -> dict:
    model = PPO.load(model_path, device=kwargs.get("device", "cpu")) if "meta" not in model_path \
        else RecurrentPPO.load(model_path, device=kwargs.get("device", "cpu"))
    is_rec = _is_recurrent(model)
    out = {}
    snr_db = kwargs.get("snr_db", 10.0)
    episode_len = kwargs.get("episode_len", 2000)
    state_window = kwargs.get("state_window", 16)
    filter_order = kwargs.get("filter_order", 16)
    signal_kind = kwargs.get("signal_kind", "multitone")
    for fam in families:
        traces = []
        for s in seeds:
            env_cfg = EnvConfig(episode_len=episode_len,
                                state_window=state_window,
                                filter_order=filter_order)
            env = AdaptiveFilterEnv(env_cfg, fixed_family=fam,
                                   fixed_signal=signal_kind,
                                   fixed_snr_db=snr_db, seed=s)
            obs, _ = env.reset(seed=s)
            lstm_state = None
            episode_starts = np.ones((1,), dtype=bool)
            errors = []
            done = False
            steps = 0
            while not done and steps < episode_len:
                if is_rec:
                    action, lstm_state = model.predict(
                        obs[None, :], state=lstm_state,
                        episode_start=episode_starts,
                        deterministic=kwargs.get("deterministic", True),
                    )
                else:
                    action, _ = model.predict(
                        obs, deterministic=kwargs.get("deterministic", True),
                    )
                obs, _, term, trunc, info = env.step(action[0] if is_rec else action)
                errors.append(env.last_e)
                if is_rec:
                    episode_starts = np.zeros((1,), dtype=bool)
                done = term or trunc
                steps += 1
            traces.append(np.array(errors))
        out[fam] = traces
    return out

"""Shared eval helpers for hybrid/BPTT controllers and sb3 RL policies."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from ..agents.controller import LSTMController, HybridController, TransformerController
from ..envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
from ..kernel import FEAT_DIM
from .metrics import convergence_time, steady_state_mse


def load_controller(path):
    """Load a hybrid/BPTT controller + its training config from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = ckpt.get("config", None)
    g = lambda k, d: getattr(tcfg, k, d) if tcfg is not None else d
    ctrl_type = g('controller_type', 'hybrid')
    lstm_hidden = g('lstm_hidden', 256)
    n_lstm_layers = g('n_lstm_layers', 2)
    if ctrl_type == "hybrid":
        ctrl = HybridController(feat_dim=FEAT_DIM, hidden=lstm_hidden,
                                n_lstm_layers=n_lstm_layers, act_dim=2, n_families=8)
    elif ctrl_type == "transformer":
        ctrl = TransformerController(feat_dim=FEAT_DIM, d_model=lstm_hidden,
                                     n_heads=4, n_layers=4, act_dim=2)
    else:
        ctrl = LSTMController(feat_dim=FEAT_DIM, hidden=lstm_hidden,
                              n_lstm_layers=n_lstm_layers, act_dim=2)
    ctrl.load_state_dict(ckpt["state_dict"])
    ctrl.eval()
    env_kw = dict(mu_min=g('mu_min', 0.005), mu_max=g('mu_max', 2.0),
                  leakage_min=g('lam_min', 0.80), leakage_max=g('lam_max', 0.999),
                  mu_base_schedule=g('use_mu_schedule', True),
                  state_window=1)
    return ctrl, env_kw


def run_controller_episode(ctrl, env_kw, clean, noisy, fs, n, order):
    """Run a hybrid/BPTT controller on a preset episode; raw-unit errors."""
    env_cfg = EnvConfigV2(fs=fs, episode_len=n, filter_order=order, **env_kw)
    env = AdaptiveFilterEnvV2(env_cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    obs, _ = env.reset(seed=0)
    state = None
    errs = []
    t0 = time.perf_counter()
    done = False
    while not done:
        obs_t = torch.tensor(obs[-FEAT_DIM:], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action, state, *_ = ctrl(obs_t, state)
        obs, _, term, trunc, _ = env.step(action[0, 0].cpu().numpy())
        errs.append(env.last_e)
        done = term or trunc
    dt = (time.perf_counter() - t0) * 1000
    return np.array(errs) * env.norm_scale, dt, env.divergence_count


def run_rl_episode(model, is_rec, vec_norm, clean, noisy, fs, n, order):
    """Run an sb3 RL policy on a preset episode; raw-unit errors."""
    from scripts.train_pipeline import RL_ENV_KW
    eval_kw = {k: v for k, v in RL_ENV_KW.items()
               if k not in ("fs", "episode_len", "filter_order")}
    env_cfg = EnvConfigV2(fs=fs, episode_len=n, filter_order=order, **eval_kw)
    env = AdaptiveFilterEnvV2(env_cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    obs, _ = env.reset(seed=0)
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    lstm_state = None
    starts = np.ones((1,), dtype=bool) if is_rec else None
    errs = []
    t0 = time.perf_counter()
    done = False
    while not done:
        if is_rec:
            a, lstm_state = model.predict(obs[None, :], state=lstm_state,
                                          episode_start=starts, deterministic=True)
            starts = np.zeros((1,), dtype=bool)
            a_use = a[0]
        else:
            a_use, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a_use)
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        errs.append(env.last_e)
        done = term or trunc
    dt = (time.perf_counter() - t0) * 1000
    return np.array(errs) * env.norm_scale, dt, env.divergence_count


def load_rl(path):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    try:
        model = PPO.load(path, device="cpu"); is_rec = False
    except Exception:
        model = RecurrentPPO.load(path, device="cpu"); is_rec = True
    vec_path = path.replace("_final.zip", "_vecnormalize.pkl")
    vec_norm = None
    if os.path.exists(vec_path):
        try:
            # Dummy env must match the training obs space (RL_ENV_KW,
            # state_window=1 → FEAT_DIM); a default EnvConfigV2 is 44-dim and
            # VecNormalize.load rejects it.
            from scripts.train_pipeline import RL_ENV_KW
            dummy_env = DummyVecEnv(
                [lambda: AdaptiveFilterEnvV2(EnvConfigV2(**RL_ENV_KW))])
            vec_norm = VecNormalize.load(vec_path, dummy_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
        except Exception as ex:
            print(f"WARNING: failed to load VecNormalize stats from "
                  f"{vec_path} ({ex}); evaluating on RAW observations — "
                  f"RL results will not reflect the trained policy")
            vec_norm = None
    return model, is_rec, vec_norm


def metrics_row(method, e, dt_ms, **keys):
    """NaN-safe eval row; extra keys (family, signal, seed, …) pass through."""
    e = np.asarray(e, dtype=np.float64)
    finite = np.isfinite(e)
    n_nonfinite = int(np.sum(~finite))
    e_safe = np.where(finite, e, 1e6)
    ss = steady_state_mse(e_safe)
    ss_db = float(10 * np.log10(ss + 1e-12))
    row = dict(method=method)
    row.update(keys)
    row.update(
        ss_mse=float(ss), ss_mse_db=ss_db,
        ep_mse=float(np.mean(e_safe ** 2)),
        conv_time=float(convergence_time(e_safe)),
        inference_time_ms=float(dt_ms),
        diverged=int(n_nonfinite > 0 or ss_db > 10.0),
    )
    return row

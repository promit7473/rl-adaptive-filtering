"""Invariants that must hold between numpy/torch kernel copies and the env."""
import numpy as np
import torch

from src.kernel import (
    ActionBounds, FeatureState, FEAT_SCALE, MU_SCHEDULE_REF,
    decode_action_np, decode_action_torch, features_np, features_torch,
    mu_base_schedule,
)
from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
from src.filters.base import windowize


def test_decode_action_np_torch_match():
    bounds = ActionBounds()
    actions = np.array([
        [0.0, 0.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [0.3, -0.5],
        [1.5, -2.0],
    ], dtype=np.float64)
    for a in actions:
        mu_np, lam_np = decode_action_np(a, bounds)
        mu_t, lam_t = decode_action_torch(torch.tensor(a, dtype=torch.float64), bounds)
        np.testing.assert_allclose(mu_np, mu_t.item(), atol=1e-6)
        np.testing.assert_allclose(lam_np, lam_t.item(), atol=1e-6)


def test_features_np_torch_match():
    rng = np.random.default_rng(2)
    x = rng.normal(size=16)
    e = 0.3
    last_e = -0.1
    last2_e = 0.05
    ema_e2 = 0.2
    last_mu = 0.4
    last_lam = 0.9
    state = FeatureState(
        last_e=last_e, last_last_e=last2_e, ema_e2=ema_e2,
        last_mu=last_mu, last_lam=last_lam,
    )
    feat_np, st = features_np(e, x, last_mu, last_lam, state, feat_scale=FEAT_SCALE)

    feat_t, ema_t = features_torch(
        torch.tensor([e], dtype=torch.float64),
        torch.tensor(x, dtype=torch.float64).unsqueeze(0),
        torch.tensor([last_mu], dtype=torch.float64),
        torch.tensor([last_lam], dtype=torch.float64),
        torch.tensor([last_e], dtype=torch.float64),
        torch.tensor([last2_e], dtype=torch.float64),
        torch.tensor([ema_e2], dtype=torch.float64),
        feat_scale=FEAT_SCALE,
        filter_order=16,
    )
    np.testing.assert_allclose(feat_np, feat_t[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(st.ema_e2, ema_t[0].item(), atol=1e-6)
    assert feat_np.shape == (11,)


def test_mu_base_schedule_floor():
    s0 = mu_base_schedule(0)
    s999 = mu_base_schedule(999)
    s1000 = mu_base_schedule(1000)
    s5000 = mu_base_schedule(5000)
    assert s0 == 0.8 * (1.0 - 0.8 * 0 / MU_SCHEDULE_REF)
    assert s999 == 0.8 * (1.0 - 0.8 * 999 / MU_SCHEDULE_REF)
    assert s1000 == 0.8 * (1.0 - 0.8 * 1000 / MU_SCHEDULE_REF)
    assert s5000 == s1000
    assert s1000 == mu_base_schedule(MU_SCHEDULE_REF)


def test_env_step_features_match_kernel():
    rng = np.random.default_rng(0)
    n, fs, order = 64, 360.0, 16
    clean = np.sin(2 * np.pi * 10.0 * np.arange(n) / fs)
    noisy = clean + 0.1 * rng.normal(size=n)
    cfg = EnvConfigV2(episode_len=n, filter_order=order, state_window=1,
                      mu_base_schedule=False)
    env = AdaptiveFilterEnvV2(cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    env.reset()

    snap = dict(
        last_e=env.last_e,
        last_last_e=env.last_last_e,
        ema_e2=env.running_error_sq_ema,
        last_mu=env.last_mu,
        last_lam=env.last_lam,
        ema_alpha=env.cfg.ema_alpha,
    )
    env.step(np.array([0.2, -0.3], dtype=np.float32))

    state = FeatureState(**snap)
    feat, _ = features_np(
        env.last_e, env.x_buf, snap["last_mu"], snap["last_lam"], state,
        feat_scale=env.feat_scale,
    )
    np.testing.assert_allclose(env.feat_buf[-1], feat, atol=1e-6)


def test_windowize_matches_env_x_buf():
    rng = np.random.default_rng(1)
    n, order = 32, 16
    clean = rng.normal(size=n)
    noisy = clean + 0.1 * rng.normal(size=n)
    cfg = EnvConfigV2(episode_len=n, filter_order=order, state_window=1)
    env = AdaptiveFilterEnvV2(cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    env.reset()
    U = windowize(env.noisy, order)
    a = np.zeros(2, dtype=np.float32)
    for t in range(n):
        env.step(a)
        np.testing.assert_allclose(env.x_buf, U[t], atol=1e-12)

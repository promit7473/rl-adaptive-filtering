"""Gymnasium environment wrapping an adaptive filter + a noise sampler.

The agent controls (mu, leakage) of a leaky-NLMS filter at every sample.
Each episode samples a noise family + SNR (this is the meta-RL task distribution).

State (per step):
  - sliding window (W) of recent features:
    [e_t, e_t^2, delta_e_t, log_input_power, log_error_power,
     sign_e_t, autocorr_estimate]
  Stacked into a flat float vector of length 7*W.

Action (continuous, 2-D, in [-1, 1]):
  - a[0] -> mu in [mu_min, mu_max] via symmetric log-scale interpolation
    (NLMS-normalized step-size; mu in [0.01, 1.0] is the useful range)
  - a[1] -> leakage in [lambda_min, 1.0] via log-scale interpolation

Reward:
  - r = -softplus(e_t^2) / softplus(sigma_ref^2)
  Bounded, smooth, non-zero gradient everywhere. Monotone in e^2.
  Locally proportional to -e^2 for small errors.
  Saturates gracefully for large errors without killing gradient.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..signals.generators import make_signal
from ..noise.families import make_noise, TRAIN_FAMILIES

FEAT_DIM = 7


@dataclass
class EnvConfig:
    fs: float = 8000.0
    episode_len: int = 2000
    filter_order: int = 16
    state_window: int = 16
    mu_min: float = 0.01
    mu_max: float = 1.0
    leakage_min: float = 0.80
    snr_db_options: Sequence[float] = (0.0, 5.0, 10.0, 15.0, 20.0)
    train_families: Sequence[str] = field(default_factory=lambda: list(TRAIN_FAMILIES))
    signal_kinds: Sequence[str] = ("multitone", "am", "sine")
    reward_scale: float = 1.0
    max_error_penalty: float = 0.5
    softplus_beta: float = 1.0


def _softplus(x: float, beta: float = 1.0) -> float:
    if x > 30.0 / beta:
        return x
    return float(np.log1p(np.exp(beta * x))) / beta


def _decode_action(a: np.ndarray, cfg: EnvConfig) -> tuple[float, float]:
    a = np.clip(a, -1.0, 1.0)
    log_min, log_max = np.log(cfg.mu_min), np.log(cfg.mu_max)
    frac = (a[0] + 1.0) * 0.5
    mu = float(np.exp(log_min + frac * (log_max - log_min)))
    lam_log_min, lam_log_max = np.log(cfg.leakage_min), np.log(1.0)
    lam_frac = (a[1] + 1.0) * 0.5
    leakage = float(np.exp(lam_log_min + lam_frac * (lam_log_max - lam_log_min)))
    return mu, leakage


class AdaptiveFilterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None,
                 fixed_family: str | None = None,
                 fixed_signal: str | None = None,
                 fixed_snr_db: float | None = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.fixed_family = fixed_family
        self.fixed_signal = fixed_signal
        self.fixed_snr_db = fixed_snr_db
        self._seed_init = seed

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(FEAT_DIM * self.cfg.state_window,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._np_random: np.random.Generator = np.random.default_rng(seed)

        self.t = 0
        self.w = np.zeros(self.cfg.filter_order, dtype=np.float64)
        self.x_buf = np.zeros(self.cfg.filter_order, dtype=np.float64)
        self.feat_buf = np.zeros((self.cfg.state_window, FEAT_DIM), dtype=np.float32)
        self.clean: np.ndarray = np.zeros(0)
        self.noisy: np.ndarray = np.zeros(0)
        self.last_e = 0.0
        self.last_last_e = 0.0
        self.episode_errors: list[float] = []
        self.running_error_sq_ema: float = 0.0
        self.divergence_count: int = 0

    def _sample_episode(self) -> None:
        cfg = self.cfg
        rng = self._np_random
        sig_kind = self.fixed_signal or rng.choice(cfg.signal_kinds)
        family = self.fixed_family or rng.choice(cfg.train_families)
        snr = float(self.fixed_snr_db if self.fixed_snr_db is not None
                    else rng.choice(cfg.snr_db_options))
        if sig_kind == "multitone":
            base = rng.uniform(150.0, 400.0)
            freqs = [base, base * rng.uniform(1.5, 2.5), base * rng.uniform(2.5, 4.0)]
            amps = [1.0, rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.6)]
            self.clean = make_signal("multitone", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                                     freqs=freqs, amps=amps)
        elif sig_kind == "am":
            self.clean = make_signal("am", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                                     fc=rng.uniform(800.0, 1500.0),
                                     fm=rng.uniform(40.0, 120.0),
                                     mod_index=rng.uniform(0.3, 0.7))
        else:
            self.clean = make_signal("sine", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                                     freq=rng.uniform(150.0, 600.0))
        noise = make_noise(family, self.clean, rng, snr_db=snr, fs=cfg.fs)
        self.noisy = self.clean + noise
        self._task_meta = dict(family=family, snr_db=snr, signal=sig_kind)

    def reset(self, *, seed: Optional[int] = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._sample_episode()
        self.t = 0
        self.w[:] = 0.0
        self.x_buf[:] = 0.0
        self.feat_buf[:] = 0.0
        self.last_e = 0.0
        self.last_last_e = 0.0
        self.episode_errors = []
        self.running_error_sq_ema = 0.0
        self.divergence_count = 0
        return self._obs(), {"task": self._task_meta}

    def _obs(self) -> np.ndarray:
        return self.feat_buf.flatten()

    def _push_features(self, e: float, sig_pow: float, res_pow: float) -> None:
        de = e - self.last_e
        dde = de - (self.last_e - self.last_last_e)
        sign_e = float(np.sign(e))
        e_sq = e * e
        self.running_error_sq_ema = 0.99 * self.running_error_sq_ema + 0.01 * e_sq
        autocorr = (e * self.last_e) / (self.running_error_sq_ema + 1e-8)

        raw = np.array([
            e,
            e_sq,
            de,
            dde,
            np.log1p(sig_pow),
            np.log1p(res_pow),
            np.clip(autocorr, -1.0, 1.0),
        ], dtype=np.float32)

        scale = np.array([5.0, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32)
        feat = np.tanh(raw * scale)
        feat = np.clip(feat, -1.0, 1.0)

        self.last_last_e = self.last_e
        self.last_e = e
        self.feat_buf = np.roll(self.feat_buf, -1, axis=0)
        self.feat_buf[-1] = feat

    def step(self, action: np.ndarray):
        cfg = self.cfg
        mu, leakage = _decode_action(np.asarray(action, dtype=np.float32), cfg)

        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = self.noisy[self.t]
        d = self.clean[self.t]

        y = float(self.w @ self.x_buf)
        e = d - y

        input_norm = float(self.x_buf @ self.x_buf) + 1e-6
        self.w = leakage * self.w + (mu / input_norm) * e * self.x_buf

        w_norm = float(np.linalg.norm(self.w))
        max_w_norm = 100.0
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm
            self.divergence_count += 1
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
            self.divergence_count += 1

        sig_pow = float(np.mean(self.x_buf ** 2))
        res_pow = e * e
        self._push_features(e, sig_pow, res_pow)

        e_sq = e * e
        if not np.isfinite(e_sq):
            e_sq = 100.0
        sp = _softplus(e_sq, beta=cfg.softplus_beta)
        sp_ref = _softplus(1.0, beta=cfg.softplus_beta)
        raw_reward = -cfg.reward_scale * sp / sp_ref
        reward = float(np.clip(raw_reward, -10.0, 0.0))

        if abs(e) > 50.0:
            reward -= cfg.max_error_penalty

        self.episode_errors.append(e)
        self.t += 1
        terminated = self.t >= cfg.episode_len
        truncated = False
        info = {}
        if terminated:
            errs = np.array(self.episode_errors)
            info["task"] = self._task_meta
            info["episode_mse"] = float(np.mean(errs ** 2))
            info["episode_ss_mse"] = float(np.mean(errs[-int(cfg.episode_len * 0.25):] ** 2))
            info["divergence_count"] = self.divergence_count
        return self._obs(), reward, terminated, truncated, info

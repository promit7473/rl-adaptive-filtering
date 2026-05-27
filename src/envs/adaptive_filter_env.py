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
    family_weights: Optional[Sequence[float]] = None  # curriculum weights; None=uniform
    signal_kinds: Sequence[str] = ("multitone", "am", "sine",
                                   "ecg_like", "random_pulses", "square_burst")
    signal_weights: Optional[Sequence[float]] = None  # None=uniform
    reward_kind: str = "log_mse"  # "log_mse" | "softplus" | "neg_abs"
    reward_scale: float = 1.0
    max_error_penalty: float = 5.0
    softplus_beta: float = 1.0
    normalize_input: bool = True  # per-episode z-score noisy + clean
    divergence_penalty: float = 2.0
    per_tap_action: bool = False  # if True, action is (M log-mu + 1 log-lambda)
    robust_alpha: float = 0.0  # weight on -|median(last 64 errors)|; non-diff
    robust_beta: float = 0.0   # weight on -max(|last 64 errors|); non-diff
    convergence_bonus: float = 0.0  # reward shaping: bonus for MSE improvement
    terminal_ss_weight: float = 0.0  # bonus for low SS-MSE at episode end
    no_reward_clip: bool = False  # if True, remove [-10, 0] reward clipping


def _softplus(x: float, beta: float = 1.0) -> float:
    if x > 30.0 / beta:
        return x
    return float(np.log1p(np.exp(beta * x))) / beta


def _decode_action(a: np.ndarray, cfg: EnvConfig):
    """Returns (mu, leakage). mu is scalar if per_tap_action=False else
    np.ndarray of shape (filter_order,)."""
    a = np.clip(a, -1.0, 1.0)
    log_min, log_max = np.log(cfg.mu_min), np.log(cfg.mu_max)
    lam_log_min, lam_log_max = np.log(cfg.leakage_min), np.log(1.0)
    if cfg.per_tap_action:
        # last entry is leakage, first M entries are per-tap log-mu.
        mu_frac = (a[:cfg.filter_order] + 1.0) * 0.5
        mu = np.exp(log_min + mu_frac * (log_max - log_min)).astype(np.float64)
        lam_frac = (a[-1] + 1.0) * 0.5
        leakage = float(np.exp(lam_log_min + lam_frac * (lam_log_max - lam_log_min)))
        return mu, leakage
    else:
        frac = (a[0] + 1.0) * 0.5
        mu = float(np.exp(log_min + frac * (log_max - log_min)))
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
        act_dim = (self.cfg.filter_order + 1) if self.cfg.per_tap_action else 2
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(act_dim,), dtype=np.float32)
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
        self.running_error_ema: float = 0.0
        self.last_10_mse: list[float] = []
        self.divergence_count: int = 0

    def _sample_episode(self) -> None:
        cfg = self.cfg
        rng = self._np_random
        if self.fixed_signal is not None:
            sig_kind = self.fixed_signal
        elif cfg.signal_weights is not None:
            sw = np.asarray(cfg.signal_weights, dtype=float)
            sig_kind = str(rng.choice(cfg.signal_kinds, p=sw / sw.sum()))
        else:
            sig_kind = rng.choice(cfg.signal_kinds)
        if self.fixed_family is not None:
            family = self.fixed_family
        elif cfg.family_weights is not None:
            w = np.asarray(cfg.family_weights, dtype=float)
            family = str(rng.choice(cfg.train_families, p=w / w.sum()))
        else:
            family = rng.choice(cfg.train_families)
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
        elif sig_kind in ("ecg_like", "random_pulses", "square_burst"):
            self.clean = make_signal(sig_kind, n=cfg.episode_len, fs=cfg.fs, rng=rng)
        else:
            self.clean = make_signal("sine", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                                     freq=rng.uniform(150.0, 600.0))
        noise = make_noise(family, self.clean, rng, snr_db=snr, fs=cfg.fs)
        self.noisy = self.clean + noise
        if cfg.normalize_input:
            s = float(np.std(self.noisy)) + 1e-9
            self.noisy = self.noisy / s
            self.clean = self.clean / s
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
        # Bootstrap EMA from input variance so autocorr feature is meaningful
        # from step 1 (not the dirty first ~100 steps).
        self.running_error_sq_ema = float(np.var(self.noisy[:64])) + 1e-3
        self.running_error_ema = float(np.mean(np.abs(self.noisy[:64]))) + 1e-3
        self.last_10_mse = []
        self.divergence_count = 0
        self._diverged_this_step = False
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
        if cfg.per_tap_action:
            tap_norm = self.x_buf ** 2 + 1e-6
            self.w = leakage * self.w + (mu / tap_norm) * e * self.x_buf
        else:
            self.w = leakage * self.w + (mu / input_norm) * e * self.x_buf

        w_norm = float(np.linalg.norm(self.w))
        max_w_norm = 100.0
        self._diverged_this_step = False
        if w_norm > max_w_norm:
            self.w *= max_w_norm / w_norm
            self.divergence_count += 1
            self._diverged_this_step = True
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
            self.divergence_count += 1
            self._diverged_this_step = True

        sig_pow = float(np.mean(self.x_buf ** 2))
        res_pow = e * e
        self._push_features(e, sig_pow, res_pow)

        e_sq = e * e
        if not np.isfinite(e_sq):
            e_sq = 100.0
        if cfg.reward_kind == "log_mse":
            raw_reward = -cfg.reward_scale * float(np.log1p(e_sq))
        elif cfg.reward_kind == "neg_abs":
            raw_reward = -cfg.reward_scale * float(np.sqrt(e_sq))
        else:  # "softplus"
            sp = _softplus(e_sq, beta=cfg.softplus_beta)
            sp_ref = _softplus(1.0, beta=cfg.softplus_beta)
            raw_reward = -cfg.reward_scale * sp / sp_ref
        reward = float(raw_reward if cfg.no_reward_clip else np.clip(raw_reward, -10.0, 0.0))

        if abs(e) > 50.0:
            reward -= cfg.max_error_penalty
        if self._diverged_this_step:
            reward -= cfg.divergence_penalty

        # Convergence bonus: reward reduction in running MSE
        if cfg.convergence_bonus > 0.0 and len(self.last_10_mse) >= 2:
            prev_avg = float(np.mean(self.last_10_mse[-10:])) + 1e-8
            reward += cfg.convergence_bonus * (float(np.log1p(prev_avg)) - float(np.log1p(e_sq + 1e-8)))

        self.last_10_mse.append(e_sq)
        if len(self.last_10_mse) > 10:
            self.last_10_mse.pop(0)

        # Non-differentiable robust terms — RL can optimize, BPTT cannot.
        if (cfg.robust_alpha > 0.0 or cfg.robust_beta > 0.0) \
                and len(self.episode_errors) >= 64:
            window = np.asarray(self.episode_errors[-64:])
            if cfg.robust_alpha > 0.0:
                reward -= cfg.robust_alpha * float(np.abs(np.median(window)))
            if cfg.robust_beta > 0.0:
                reward -= cfg.robust_beta * float(np.max(np.abs(window)))

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
            # Terminal SS-MSE bonus: reward for low steady-state error
            if cfg.terminal_ss_weight > 0.0:
                ss_mse = info["episode_ss_mse"]
                reward += cfg.terminal_ss_weight * float(-np.log1p(ss_mse))
        return self._obs(), reward, terminated, truncated, info

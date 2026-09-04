"""V2 Gymnasium environment: 11-D features, convergence shaping, robust rewards.

Key upgrades over v1:
  - 11 features (was 7): adds mu, lambda, |grad|, delta_e_sign
  - Convergence speed bonus in reward (shaping)
  - Log-MSE reward with convergence shaping
  - Better feature normalization with learned scaling
  - Per-tap action with correct per-tap normalization
  - Configurable episode length (longer = better steady-state)
  - Proper VecNormalize compatibility
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..noise.families import TRAIN_FAMILIES, OOD_FAMILIES
from ..kernel import (
    FEAT_DIM, FEAT_SCALE, MU_SCHEDULE_GAIN, ActionBounds, FeatureState,
    decode_action_np, features_np, mu_base_schedule, nlms_update_np,
    sample_episode,
)

FEAT_DIM_V2 = FEAT_DIM


@dataclass
class EnvConfigV2:
    fs: float = 360.0
    episode_len: int = 4000
    filter_order: int = 16
    state_window: int = 4
    mu_min: float = 0.005
    mu_max: float = 2.0
    leakage_min: float = 0.80
    leakage_max: float = 0.999
    snr_db_options: Sequence[float] = (0.0, 5.0, 10.0, 15.0, 20.0)
    train_families: Sequence[str] = field(default_factory=lambda: list(TRAIN_FAMILIES) + list(OOD_FAMILIES))
    family_weights: Optional[Sequence[float]] = None
    signal_kinds: Sequence[str] = ("multitone", "am", "sine",
                                   "ecg_like", "random_pulses", "square_burst")
    signal_weights: Optional[Sequence[float]] = None
    reward_kind: str = "shaped_log_mse"
    reward_scale: float = 1.0
    max_error_penalty: float = 5.0
    normalize_input: bool = True
    divergence_penalty: float = 2.0
    per_tap_action: bool = False
    convergence_bonus: float = 0.15
    robust_alpha: float = 0.05
    robust_beta: float = 0.02
    terminal_ss_weight: float = 0.0
    no_reward_clip: bool = False
    mu_base_schedule: bool = False
    ema_alpha: float = 0.01
    feat_scale: Sequence[float] = FEAT_SCALE


def _decode_action_v2(a: np.ndarray, cfg: EnvConfigV2):
    """Decode action to (mu, leakage). Supports per-tap mu."""
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    bounds = ActionBounds(
        mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        lam_min=cfg.leakage_min, lam_max=cfg.leakage_max,
    )
    if cfg.per_tap_action:
        log_min, log_max = np.log(cfg.mu_min), np.log(cfg.mu_max)
        mu_frac = (a[:cfg.filter_order] + 1.0) * 0.5
        mu = np.exp(log_min + mu_frac * (log_max - log_min)).astype(np.float64)
        _, leakage = decode_action_np(np.array([0.0, a[-1]], dtype=np.float64), bounds)
        return mu, leakage
    return decode_action_np(a, bounds)


class AdaptiveFilterEnvV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfigV2 | None = None,
                 fixed_family: str | None = None,
                 fixed_signal: str | None = None,
                 fixed_snr_db: float | None = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or EnvConfigV2()
        self.fixed_family = fixed_family
        self.fixed_signal = fixed_signal
        self.fixed_snr_db = fixed_snr_db
        self._seed_init = seed
        self._preset: Optional[tuple[np.ndarray, np.ndarray]] = None
        self.norm_scale: float = 1.0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(FEAT_DIM_V2 * self.cfg.state_window,),
            dtype=np.float32,
        )
        act_dim = (self.cfg.filter_order + 1) if self.cfg.per_tap_action else 2
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(act_dim,), dtype=np.float32)
        self._np_random: np.random.Generator = np.random.default_rng(seed)

        self.t = 0
        self.w = np.zeros(self.cfg.filter_order, dtype=np.float64)
        self.x_buf = np.zeros(self.cfg.filter_order, dtype=np.float64)
        self.feat_buf = np.zeros((self.cfg.state_window, FEAT_DIM_V2), dtype=np.float32)
        self.clean: np.ndarray = np.zeros(0)
        self.noisy: np.ndarray = np.zeros(0)
        self.last_e = 0.0
        self.last_last_e = 0.0
        self.last_last2_e = 0.0
        self.last_mu = 0.1
        self.last_lam = 1.0
        self.episode_errors: list[float] = []
        self.running_error_sq_ema: float = 0.0
        self.running_error_ema: float = 0.0
        self.divergence_count: int = 0
        self._diverged_this_step: bool = False
        self.feat_scale = np.array(self.cfg.feat_scale, dtype=np.float32)

    def set_preset_episode(self, clean: np.ndarray, noisy: np.ndarray) -> None:
        """Inject a fixed (clean, noisy) pair so every method can be evaluated
        on the identical realization. Applied on the next reset()."""
        self._preset = (np.asarray(clean, dtype=np.float64),
                        np.asarray(noisy, dtype=np.float64))

    def _sample_episode(self) -> None:
        cfg = self.cfg
        rng = self._np_random

        if self._preset is not None:
            self.clean, self.noisy = self._preset[0].copy(), self._preset[1].copy()
            self.norm_scale = 1.0
            if cfg.normalize_input:
                s = float(np.std(self.noisy)) + 1e-9
                self.noisy = self.noisy / s
                self.clean = self.clean / s
                self.norm_scale = s
            self._task_meta = dict(family=self.fixed_family or "preset",
                                   snr_db=self.fixed_snr_db or 0.0,
                                   signal=self.fixed_signal or "preset")
            return

        kinds = [self.fixed_signal] if self.fixed_signal is not None else list(cfg.signal_kinds)
        if self.fixed_signal is not None:
            sw = np.array([1.0], dtype=float)
        elif cfg.signal_weights is not None:
            sw = np.asarray(cfg.signal_weights, dtype=float)
        else:
            sw = np.ones(len(kinds), dtype=float)

        fams = [self.fixed_family] if self.fixed_family is not None else list(cfg.train_families)
        if self.fixed_family is not None:
            fw = np.array([1.0], dtype=float)
        elif cfg.family_weights is not None:
            fw = np.asarray(cfg.family_weights, dtype=float)
        else:
            fw = None

        snrs = (self.fixed_snr_db,) if self.fixed_snr_db is not None else cfg.snr_db_options

        self.clean, self.noisy, family, snr, sig_kind, self.norm_scale = sample_episode(
            rng, cfg.episode_len, cfg.fs,
            train_families=fams,
            signal_kinds=kinds,
            signal_weights=sw,
            snr_options=snrs,
            family_weights=fw,
            normalize=cfg.normalize_input,
        )
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
        self.last_last2_e = 0.0
        self.last_mu = float(np.sqrt(self.cfg.mu_min * self.cfg.mu_max))
        self.last_lam = float(np.sqrt(self.cfg.leakage_min * self.cfg.leakage_max))
        self.episode_errors = []
        self.running_error_sq_ema = float(np.var(self.noisy[:64])) + 1e-3
        self.running_error_ema = float(np.mean(np.abs(self.noisy[:64]))) + 1e-3
        self.divergence_count = 0
        self._diverged_this_step = False
        self._push_features(0.0)
        return self._obs(), {"task": self._task_meta}

    def _obs(self) -> np.ndarray:
        return self.feat_buf.flatten()

    def _push_features(self, e: float) -> None:
        state = FeatureState(
            last_e=self.last_e,
            last_last_e=self.last_last_e,
            ema_e2=self.running_error_sq_ema,
            last_mu=float(np.mean(self.last_mu)) if isinstance(self.last_mu, np.ndarray)
            else float(self.last_mu),
            last_lam=float(self.last_lam),
            ema_alpha=self.cfg.ema_alpha,
        )
        feat, state = features_np(
            e, self.x_buf, state.last_mu, state.last_lam, state,
            feat_scale=self.feat_scale,
        )
        self.last_last2_e = self.last_last_e
        self.last_last_e = state.last_last_e
        self.last_e = state.last_e
        self.running_error_sq_ema = state.ema_e2
        self.running_error_ema = (
            (1.0 - self.cfg.ema_alpha) * self.running_error_ema
            + self.cfg.ema_alpha * abs(float(e))
        )
        self.feat_buf = np.roll(self.feat_buf, -1, axis=0)
        self.feat_buf[-1] = feat.astype(np.float32)

    def step(self, action: np.ndarray):
        cfg = self.cfg
        mu, leakage = _decode_action_v2(np.asarray(action, dtype=np.float32), cfg)
        if cfg.mu_base_schedule and not cfg.per_tap_action:
            base_mu = mu_base_schedule(self.t)
            mu = float(np.clip(base_mu + MU_SCHEDULE_GAIN * mu,
                               cfg.mu_min, cfg.mu_max))

        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = self.noisy[self.t]
        d = self.clean[self.t]

        y = float(self.w @ self.x_buf)
        e = d - y

        self._push_features(e)

        if cfg.per_tap_action:
            tap_norm = self.x_buf ** 2 + 1e-6
            self.w = leakage * self.w + (mu / tap_norm) * e * self.x_buf
            w_norm = float(np.linalg.norm(self.w))
            self._diverged_this_step = False
            if w_norm > 100.0:
                self.w *= 100.0 / w_norm
                self.divergence_count += 1
                self._diverged_this_step = True
        else:
            self.w = nlms_update_np(self.w, self.x_buf, e, mu, leakage)
            w_norm = float(np.linalg.norm(self.w))
            self._diverged_this_step = False
            if w_norm >= 100.0 - 1e-12:
                self.divergence_count += 1
                self._diverged_this_step = True
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
            self.divergence_count += 1
            self._diverged_this_step = True

        mu_val = float(np.mean(mu)) if isinstance(mu, np.ndarray) else float(mu)
        self.last_mu = mu_val
        self.last_lam = float(leakage)

        e_sq = e * e
        if not np.isfinite(e_sq):
            e_sq = 100.0

        if cfg.reward_kind == "shaped_log_mse":
            raw_reward = -cfg.reward_scale * float(np.log1p(e_sq))
            if len(self.episode_errors) >= 2:
                prev_avg = float(np.mean(np.array(self.episode_errors[-10:]) ** 2)) + 1e-8
                curr_avg = e_sq + 1e-8
                improvement = float(np.log1p(prev_avg) - np.log1p(curr_avg))
                raw_reward += cfg.convergence_bonus * improvement
        elif cfg.reward_kind == "log_mse":
            raw_reward = -cfg.reward_scale * float(np.log1p(e_sq))
        elif cfg.reward_kind == "neg_abs":
            raw_reward = -cfg.reward_scale * float(np.sqrt(e_sq))
        else:
            raw_reward = -cfg.reward_scale * float(np.log1p(e_sq))

        reward = float(raw_reward if cfg.no_reward_clip else np.clip(raw_reward, -10.0, 0.0))

        if abs(e) > 50.0:
            reward -= cfg.max_error_penalty
        if self._diverged_this_step:
            reward -= cfg.divergence_penalty

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
            if cfg.terminal_ss_weight > 0.0:
                ss_mse = info["episode_ss_mse"]
                reward += cfg.terminal_ss_weight * float(-np.log1p(ss_mse))
        return self._obs(), reward, terminated, truncated, info

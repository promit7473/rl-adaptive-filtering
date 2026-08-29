"""Reference-based ANC environment with a Kalman adaptive filter.

The agent controls the Kalman **process noise Q** (and measurement noise R) of a
reference-based interference canceller.  Crucially, the split between *observable*
and *supervised* quantities is enforced here:

  observable (drive the filter + the 11-D state, available at deployment):
      reference[t], primary[t], innovation e[t] = primary[t] - w^T x[t],
      Kalman gain, innovation variance, P-trace, the agent's own past (Q,R)

  supervised (used ONLY to shape the training reward / offline metric, never at
  deployment):
      clean[t]  ->  residual interference  r[t] = e[t] - clean[t]

Because the reference is independent of the clean signal, minimising innovation
power already minimises residual interference (Widrow's ANC principle); the clean
signal is used at train time only to sharpen the learning signal.

The observation is 11-D and squashed to [-1, 1].  The action is 2-D in [-1, 1],
decoded to (Q, R) by ``diff_kalman.decode_action_kalman`` -- the SAME decode used
by the BPTT trainer, so training and deployment are byte-identical.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..signals.generators import make_signal
from ..interference.families import make_interference, TRAIN_FAMILIES, OOD_FAMILIES
from ..filters.diff_kalman import DiffKalmanConfig, decode_action_kalman
import torch

FEAT_DIM = 11
SIGNAL_KINDS = ("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst")


@dataclass
class ANCEnvConfig:
    fs: float = 360.0
    episode_len: int = 1000
    filter_order: int = 16
    q_min: float = 1e-8
    q_max: float = 1e-3
    r_min: float = 1e-3
    r_max: float = 1e1
    p0: float = 1.0
    snr_db_options: Sequence[float] = (-5.0, 0.0, 5.0, 10.0, 15.0)
    train_families: Sequence[str] = field(default_factory=lambda: list(TRAIN_FAMILIES))
    family_weights: Optional[Sequence[float]] = None
    signal_kinds: Sequence[str] = SIGNAL_KINDS
    signal_weights: Optional[Sequence[float]] = None
    # reward shaping
    convergence_bonus: float = 0.15
    robust_alpha: float = 0.05
    terminal_ss_weight: float = 0.3
    # feature scaling (tanh gains)
    feat_scale: Sequence[float] = (3.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0,
                                   0.2, 0.2, 5.0, 1.0)


def _decode_qr_np(a: np.ndarray, cfg: ANCEnvConfig):
    a = np.clip(a, -1.0, 1.0)
    lo_q, hi_q = np.log(cfg.q_min), np.log(cfg.q_max)
    lo_r, hi_r = np.log(cfg.r_min), np.log(cfg.r_max)
    q = float(np.exp(lo_q + (a[0] + 1.0) * 0.5 * (hi_q - lo_q)))
    r = float(np.exp(lo_r + (a[1] + 1.0) * 0.5 * (hi_r - lo_r)))
    return q, r


class ANCKalmanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: ANCEnvConfig | None = None,
                 fixed_family: str | None = None,
                 fixed_signal: str | None = None,
                 fixed_snr_db: float | None = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or ANCEnvConfig()
        self.fixed_family = fixed_family
        self.fixed_signal = fixed_signal
        self.fixed_snr_db = fixed_snr_db
        self._preset: Optional[tuple] = None

        self.observation_space = spaces.Box(-1.0, 1.0, (FEAT_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._rng = np.random.default_rng(seed)

        M = self.cfg.filter_order
        self.w = np.zeros(M)
        self.x = np.zeros(M)
        self.P = np.eye(M) * self.cfg.p0
        self.feat_scale = np.asarray(self.cfg.feat_scale, np.float32)
        self._reset_state()

    # ---- episode sampling ----
    def set_preset_episode(self, clean, interference, reference):
        self._preset = (np.asarray(clean, float), np.asarray(interference, float),
                        np.asarray(reference, float))

    def _sample_episode(self):
        cfg = self.cfg
        rng = self._rng
        if self._preset is not None:
            clean, interf, ref = self._preset
            self._task = dict(family=self.fixed_family or "preset",
                              snr_db=self.fixed_snr_db or 0.0,
                              signal=self.fixed_signal or "preset")
        else:
            sk = (self.fixed_signal if self.fixed_signal
                  else _weighted_choice(rng, cfg.signal_kinds, cfg.signal_weights))
            fam = (self.fixed_family if self.fixed_family
                   else _weighted_choice(rng, cfg.train_families, cfg.family_weights))
            snr = float(self.fixed_snr_db if self.fixed_snr_db is not None
                        else rng.choice(cfg.snr_db_options))
            clean = make_signal(sk, n=cfg.episode_len, fs=cfg.fs, rng=rng)
            interf, ref = make_interference(fam, clean, rng, snr_db=snr, fs=cfg.fs)
            self._task = dict(family=fam, snr_db=snr, signal=sk)
        # normalise by primary std for stable features; scale stored for raw-unit eval
        primary = clean + interf
        s = float(np.std(primary)) + 1e-9
        self.norm_scale = s
        self.clean = clean / s
        self.interf = interf / s
        self.reference = ref / (np.std(ref) + 1e-9)
        self.primary = self.clean + self.interf

    def _reset_state(self):
        M = self.cfg.filter_order
        self.w[:] = 0.0
        self.x[:] = 0.0
        self.P = np.eye(M) * self.cfg.p0
        self.t = 0
        self.last_e = 0.0
        self.last_last_e = 0.0
        self.last_logq = np.log(np.sqrt(self.cfg.q_min * self.cfg.q_max))
        self.last_logr = np.log(np.sqrt(self.cfg.r_min * self.cfg.r_max))
        self.ema_e2 = 1.0
        self.errors: list[float] = []      # innovations (observable)
        self.residuals: list[float] = []   # e - clean (supervised)
        self.q_hist: list[float] = []      # per-step process noise (for figures)
        self.r_hist: list[float] = []
        self.last_e_raw = 0.0

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_episode()
        self._reset_state()
        return (self._features(0.0, 0.0, 0.0, self.last_logq, self.last_logr, 0.0, 0.0),
                {"task": self._task})

    # ---- features (observable only) ----
    # Mirrored exactly by kalman_trainer._anc_features for BPTT/eval parity.
    def _features(self, e, gain_norm, S, logq, logr, prev_e, prev_prev_e):
        de = e - prev_e
        e_sq = e * e
        self.ema_e2 = 0.99 * self.ema_e2 + 0.01 * e_sq
        autocorr = np.clip((e * prev_e) / (self.ema_e2 + 1e-8), -1.0, 1.0)
        ref_pow = float(self.x @ self.x) / self.cfg.filter_order
        raw = np.array([
            e,
            e_sq,
            de,
            np.log1p(ref_pow),
            np.log1p(max(S, 0.0)),
            np.log1p(e_sq),
            autocorr,
            logq - np.log(1e-6),   # centered log Q (own action)
            logr - np.log(1.0),    # centered log R
            gain_norm,
            np.sign(de),
        ], dtype=np.float32)
        return np.clip(np.tanh(raw * self.feat_scale), -1.0, 1.0)

    def step(self, action):
        cfg = self.cfg
        q, r = _decode_qr_np(np.asarray(action, np.float32), cfg)
        M = cfg.filter_order
        # advance reference tap buffer
        self.x = np.roll(self.x, 1)
        self.x[0] = self.reference[self.t]
        # Kalman predict/update (observable: primary, reference)
        self.P = self.P + q * np.eye(M)
        yhat = float(self.w @ self.x)
        e = self.primary[self.t] - yhat            # innovation == denoised output
        Px = self.P @ self.x
        S = float(self.x @ Px) + r
        K = Px / S
        self.w = self.w + K * e
        self.P = self.P - np.outer(K, self.x @ self.P)

        gain_norm = float(np.linalg.norm(K))
        ptrace = float(np.trace(self.P))
        residual = e - self.clean[self.t]           # supervised (train/metric only)

        # ---- reward: label-free innovation-power term + supervised residual term ----
        # The innovation term is deployable; the residual term (train only) sharpens
        # the signal by isolating exactly what the agent can control.
        e2 = e * e
        raw_reward = -np.log1p(residual * residual)
        if len(self.residuals) >= 2:
            prev = float(np.mean(np.array(self.residuals[-10:]) ** 2)) + 1e-8
            raw_reward += cfg.convergence_bonus * (np.log1p(prev) - np.log1p(residual * residual + 1e-8))
        reward = float(raw_reward)
        if cfg.robust_alpha > 0 and len(self.residuals) >= 64:
            reward -= cfg.robust_alpha * float(np.abs(np.median(self.residuals[-64:])))

        prev_e, prev_prev_e = self.last_e, self.last_last_e
        self.errors.append(e)
        self.residuals.append(residual)
        self.q_hist.append(q)
        self.r_hist.append(r)
        self.last_last_e = self.last_e
        self.last_e = e
        self.last_logq = np.log(q)
        self.last_logr = np.log(r)
        self.last_e_raw = e * self.norm_scale
        self.t += 1

        obs = self._features(e, gain_norm, S, np.log(q), np.log(r), prev_e, prev_prev_e)
        terminated = self.t >= cfg.episode_len
        info = {}
        if terminated:
            res = np.asarray(self.residuals)
            tail = int(cfg.episode_len * 0.25)
            info["task"] = self._task
            info["episode_ss_mse"] = float(np.mean(res[-tail:] ** 2))
            info["episode_mse"] = float(np.mean(res ** 2))
            if cfg.terminal_ss_weight > 0:
                reward += cfg.terminal_ss_weight * float(-np.log1p(info["episode_ss_mse"]))
        return obs, reward, terminated, False, info

    # convenience for eval harness: residual in raw units (denoising error)
    @property
    def last_residual_raw(self) -> float:
        return (self.last_e - self.clean[self.t - 1]) * self.norm_scale if self.t > 0 else 0.0


def _weighted_choice(rng, options, weights):
    if weights is None:
        return str(rng.choice(list(options)))
    w = np.asarray(weights, float)
    return str(rng.choice(list(options), p=w / w.sum()))

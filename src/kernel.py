"""Shared adaptive-filter kernel: decode, features, NLMS, schedule, sampling."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .signals.generators import make_signal
from .noise.families import make_noise

FEAT_DIM = 11
FEAT_SCALE = (5.0, 2.0, 5.0, 5.0, 1.0, 1.0, 1.0, 4.0, 10.0, 2.0, 1.0)
MU_SCHEDULE_GAIN = 0.3
MU_SCHEDULE_REF = 1000

SIGNAL_KINDS = ("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst")
SIGNAL_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=float)
SIGNAL_WEIGHTS = SIGNAL_WEIGHTS / SIGNAL_WEIGHTS.sum()
CURRICULUM_WEIGHTS = {
    "gaussian": 1.0, "colored": 1.5, "impulsive": 2.0,
    "time_varying": 2.0, "regime_switch": 3.0,
}
SNR_OPTIONS = (0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass
class ActionBounds:
    mu_min: float = 0.005
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 0.999


def decode_action_np(a: np.ndarray, bounds: ActionBounds) -> tuple[float, float]:
    """a in [-1, 1], length 2. log-interp μ and λ."""
    a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1)[:2], -1.0, 1.0)
    mu_frac = (a[0] + 1.0) * 0.5
    lam_frac = (a[1] + 1.0) * 0.5
    log_mu_min, log_mu_max = np.log(bounds.mu_min), np.log(bounds.mu_max)
    log_lam_min, log_lam_max = np.log(bounds.lam_min), np.log(bounds.lam_max)
    mu = float(np.exp(log_mu_min + mu_frac * (log_mu_max - log_mu_min)))
    lam = float(np.exp(log_lam_min + lam_frac * (log_lam_max - log_lam_min)))
    return mu, lam


def decode_action_torch(a: torch.Tensor, bounds: ActionBounds) -> tuple[torch.Tensor, torch.Tensor]:
    """a[..., 0]=μ, a[..., 1]=λ. Same log-interp as numpy."""
    a = torch.clamp(a, -1.0, 1.0)
    mu_frac = (a[..., 0] + 1.0) * 0.5
    lam_frac = (a[..., 1] + 1.0) * 0.5
    log_mu_min = a.new_tensor(np.log(bounds.mu_min))
    log_mu_max = a.new_tensor(np.log(bounds.mu_max))
    log_lam_min = a.new_tensor(np.log(bounds.lam_min))
    log_lam_max = a.new_tensor(np.log(bounds.lam_max))
    mu = torch.exp(log_mu_min + mu_frac * (log_mu_max - log_mu_min))
    lam = torch.exp(log_lam_min + lam_frac * (log_lam_max - log_lam_min))
    return mu, lam


def mu_base_schedule(t: int) -> float:
    """tt = min(t, MU_SCHEDULE_REF); return 0.8 * (1.0 - 0.8 * tt / MU_SCHEDULE_REF)."""
    tt = min(int(t), MU_SCHEDULE_REF)
    return 0.8 * (1.0 - 0.8 * tt / MU_SCHEDULE_REF)


def nlms_update_np(w, x, e, mu, lam, eps=1e-6, max_w_norm=100.0) -> np.ndarray:
    """w <- lam*w + (mu / (x·x+eps)) * e * x; clip ||w||."""
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    denom = float(np.dot(x, x)) + eps
    w_new = lam * w + (mu / denom) * e * x
    nrm = float(np.linalg.norm(w_new))
    if nrm > max_w_norm:
        w_new = w_new * (max_w_norm / nrm)
    return w_new


def nlms_update_torch(w, x_buf, e, mu, lam, eps=1e-6, max_w_norm=100.0) -> torch.Tensor:
    """Batched (B, M) equivalent."""
    input_norm = (x_buf * x_buf).sum(dim=-1) + eps
    w_new = lam.unsqueeze(-1) * w + (mu / input_norm).unsqueeze(-1) * e.unsqueeze(-1) * x_buf
    w_norm = torch.norm(w_new, dim=-1)
    clip_mask = w_norm > max_w_norm
    if clip_mask.any():
        scale = torch.where(
            clip_mask,
            max_w_norm / (w_norm + 1e-8),
            torch.ones_like(w_norm),
        )
        w_new = w_new * scale.unsqueeze(-1)
    return w_new


@dataclass
class FeatureState:
    last_e: float
    last_last_e: float
    ema_e2: float
    last_mu: float
    last_lam: float
    ema_alpha: float = 0.01


def features_np(e, x_buf, last_mu, last_lam, state: FeatureState,
                feat_scale=FEAT_SCALE) -> tuple[np.ndarray, FeatureState]:
    """Return 11-D tanh features using last_mu/last_lam, and the updated state."""
    e = float(e)
    x = np.asarray(x_buf, dtype=np.float64)
    last_mu = float(last_mu)
    last_lam = float(last_lam)
    scale = np.asarray(feat_scale, dtype=np.float64)

    e2 = e * e
    de = e - state.last_e
    dde = de - (state.last_e - state.last_last_e)
    ema_e2 = (1.0 - state.ema_alpha) * state.ema_e2 + state.ema_alpha * e2
    autocorr = np.clip(e * state.last_e / (ema_e2 + 1e-8), -1.0, 1.0)
    x_sq_sum = float(np.dot(x, x))
    sig_pow = float(np.mean(x * x)) if x.size else 0.0
    grad_norm = abs(last_mu * e) / np.sqrt(x_sq_sum + 1e-8)
    sign_de = float(np.sign(de))

    raw = np.array([
        e,
        e2,
        de,
        dde,
        np.log1p(sig_pow),
        np.log1p(e2),
        autocorr,
        last_mu - 0.5,
        last_lam - 0.85,
        grad_norm,
        sign_de,
    ], dtype=np.float64)
    feat = np.tanh(raw * scale)

    new_state = replace(
        state,
        last_e=e,
        last_last_e=state.last_e,
        ema_e2=float(ema_e2),
        last_mu=last_mu,
        last_lam=last_lam,
    )
    return feat, new_state


def features_torch(e, x_buf, last_mu, last_lam, last_e, last2_e, ema_e2,
                   feat_scale=FEAT_SCALE, ema_alpha=0.01, filter_order=16):
    """Batched (B, 11) equivalent. Returns (feat, new_ema_e2)."""
    scale = torch.as_tensor(feat_scale, dtype=e.dtype, device=e.device)

    e2 = e * e
    de = e - last_e
    dde = de - (last_e - last2_e)
    new_ema = (1.0 - ema_alpha) * ema_e2 + ema_alpha * e2
    autocorr = torch.clamp(e * last_e / (new_ema + 1e-8), -1.0, 1.0)
    x_sq_sum = (x_buf * x_buf).sum(dim=-1)
    sig_pow = x_buf.pow(2).mean(dim=-1)
    grad_norm = torch.abs(last_mu * e) / torch.sqrt(x_sq_sum + 1e-8)
    sign_de = torch.sign(de)

    raw = torch.stack([
        e,
        e2,
        de,
        dde,
        torch.log1p(sig_pow),
        torch.log1p(e2),
        autocorr,
        last_mu - 0.5,
        last_lam - 0.85,
        grad_norm,
        sign_de.to(dtype=e.dtype),
    ], dim=-1)
    feat = torch.tanh(raw * scale)
    return feat, new_ema


def sample_episode(rng, n, fs, *, train_families, curriculum_frac=1.0,
                   signal_kinds, signal_weights, snr_options,
                   family_weights=None) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Returns (clean, noisy, family_name, snr) after per-episode std-norm."""
    train_fams = list(train_families)
    if family_weights is not None:
        fam_weights = np.asarray(family_weights, dtype=float).copy()
    else:
        fam_weights = np.ones(len(train_fams), dtype=float)
    if curriculum_frac < 1.0:
        hard = {"regime_switch", "time_varying"}
        ramp = 0.25 + 0.75 * curriculum_frac
        for i, f in enumerate(train_fams):
            if f in hard:
                fam_weights[i] *= ramp
    fam_weights = fam_weights / fam_weights.sum()
    family = str(rng.choice(train_fams, p=fam_weights))

    kinds = list(signal_kinds)
    sw = np.asarray(signal_weights, dtype=float)
    sw = sw / sw.sum()
    sig_kind = str(rng.choice(kinds, p=sw))
    snr = float(rng.choice(np.asarray(list(snr_options), dtype=float)))

    if sig_kind == "multitone":
        base = rng.uniform(150.0, 400.0)
        clean = make_signal(
            "multitone", n=n, fs=fs, rng=rng,
            freqs=[base, base * rng.uniform(1.5, 2.5), base * rng.uniform(2.5, 4.0)],
            amps=[1.0, rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.6)],
        )
    elif sig_kind == "am":
        clean = make_signal(
            "am", n=n, fs=fs, rng=rng,
            fc=rng.uniform(800.0, 1500.0),
            fm=rng.uniform(40.0, 120.0),
            mod_index=rng.uniform(0.3, 0.7),
        )
    elif sig_kind in ("ecg_like", "random_pulses", "square_burst"):
        clean = make_signal(sig_kind, n=n, fs=fs, rng=rng)
    else:
        clean = make_signal("sine", n=n, fs=fs, rng=rng, freq=rng.uniform(150.0, 600.0))

    noise = make_noise(family, clean, rng, snr_db=snr, fs=fs)
    noisy = clean + noise
    s = float(np.std(noisy)) + 1e-9
    clean = clean / s
    noisy = noisy / s
    return clean, noisy, family, snr

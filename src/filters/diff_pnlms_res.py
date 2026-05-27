"""Differentiable PNLMS with learned residual correction.

The filter has TWO paths:
  1. Model-based path: PNLMS with per-tap step sizes + functional expansion
     w_{t+1} = lam * w_t + diag(mu_t/norm) * e_t * x_expanded_t
     y_filter = w^T x_expanded

  2. Learned residual path: controller outputs a scalar correction
     y_final = y_filter + delta_t

The residual delta_t is a *post-filter correction* — standard in speech
enhancement literature (Gustafsson et al., 2001; Srinivasan et al., 2006).
The residual handles what the linear filter structurally cannot: nonlinear
noise components, non-stationary artifacts, and model mismatch.

The key insight: if PNLMS already gives you -14 dB, the residual only
needs to learn the remaining -7 dB. Learning a residual is FAR easier
than learning the full -21 dB mapping from scratch (ResNet principle).

This is NOT a CNN in disguise:
  - The residual is a single scalar per timestep, not a full reconstruction
  - The model-based path (PNLMS) provides the bulk of the denoising
  - The residual is interpretable: it's the post-filter correction
  - The architecture is strictly causal and real-time (no look-ahead)

Action space: M + 2
  - mu_t[0..M-1]: per-tap step sizes
  - lam_t: leakage factor
  - delta_t: residual correction scalar
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class DiffPNLMSResConfig:
    order: int = 16
    mu_min: float = 0.001
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 1.0
    eps: float = 1e-6
    max_w_norm: float = 100.0
    use_fx: bool = True
    fx_order: int = 3
    delta_scale: float = 1.0


def decode_action_pnlms_res(a: torch.Tensor, cfg: DiffPNLMSResConfig):
    """Decode raw actions in [-1,1] to (mu_vec, lambda, delta).

    Args:
        a: (..., M+2) tensor — M per-tap mu, lambda, delta
    Returns:
        mu_vec: (..., M) per-tap step sizes
        lam: (...,) leakage
        delta: (...,) residual correction
    """
    a = torch.clamp(a, -1.0, 1.0)
    M = a.shape[-1] - 2

    log_mu_min = torch.tensor(np.log(cfg.mu_min), dtype=a.dtype, device=a.device)
    log_mu_max = torch.tensor(np.log(cfg.mu_max), dtype=a.dtype, device=a.device)
    log_lam_min = torch.tensor(np.log(cfg.lam_min), dtype=a.dtype, device=a.device)
    log_lam_max = torch.tensor(np.log(cfg.lam_max), dtype=a.dtype, device=a.device)

    mu_frac = (a[..., :M] + 1.0) * 0.5
    mu_vec = torch.exp(log_mu_min + mu_frac * (log_mu_max - log_mu_min))

    lam_frac = (a[..., -2] + 1.0) * 0.5
    lam = torch.exp(log_lam_min + lam_frac * (log_lam_max - log_lam_min))

    delta = a[..., -1] * cfg.delta_scale

    return mu_vec, lam, delta


def functional_expand(x_buf: torch.Tensor, fx_order: int = 3):
    parts = [x_buf]
    if fx_order >= 2:
        parts.append(x_buf * x_buf)
    if fx_order >= 3:
        parts.append(torch.sign(x_buf) * torch.sqrt(torch.abs(x_buf) + 1e-8))
    return torch.cat(parts, dim=-1)


class PNLMSResFilterWrapper:
    """NumPy inference wrapper for PNLMS+Residual."""

    def __init__(self, controller: nn.Module, cfg: DiffPNLMSResConfig | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or DiffPNLMSResConfig()
        self.controller = controller
        self.device = device
        self.order = self.cfg.order
        M = self.cfg.order
        self.M_eff = M * self.cfg.fx_order if self.cfg.use_fx else M
        self.w = np.zeros(self.M_eff, dtype=np.float64)
        self.x_buf = np.zeros(M, dtype=np.float64)
        self.state = None

    def reset(self) -> None:
        self.w[:] = 0.0
        self.x_buf[:] = 0.0
        self.state = None

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = u[0] if u.ndim == 0 else u[0]

        if self.cfg.use_fx:
            x_exp = self._fx_expand(self.x_buf)
        else:
            x_exp = self.x_buf

        y_filter = float(self.w @ x_exp)
        e = d - y_filter

        feat = self._build_feat(e, y_filter, float(self.x_buf @ self.x_buf))
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            a, self.state = self.controller(feat_t, self.state)[:2]
        a_np = a[0, 0].cpu().numpy()

        mu_vec, lam, delta = self._decode_action(a_np)

        y_final = y_filter + delta
        e_final = d - y_final

        tap_norm = float(x_exp @ x_exp) + self.cfg.eps
        self.w = lam * self.w + (mu_vec / tap_norm) * e_final * x_exp

        w_norm = float(np.linalg.norm(self.w))
        if w_norm > self.cfg.max_w_norm:
            self.w *= self.cfg.max_w_norm / w_norm

        return y_final, e_final

    def _fx_expand(self, x):
        parts = [x]
        if self.cfg.fx_order >= 2:
            parts.append(x * x)
        if self.cfg.fx_order >= 3:
            parts.append(np.sign(x) * np.sqrt(np.abs(x) + 1e-8))
        return np.concatenate(parts)

    def _decode_action(self, a):
        a = np.clip(a, -1.0, 1.0)
        M = self.order
        log_min, log_max = np.log(self.cfg.mu_min), np.log(self.cfg.mu_max)
        mu_frac = (a[:M] + 1.0) * 0.5
        mu_vec = np.exp(log_min + mu_frac * (log_max - log_min))
        if self.cfg.use_fx:
            mu_vec = np.repeat(mu_vec, self.cfg.fx_order)

        lam_log_min, lam_log_max = np.log(self.cfg.lam_min), np.log(self.cfg.lam_max)
        lam_frac = (a[-2] + 1.0) * 0.5
        lam = float(np.exp(lam_log_min + lam_frac * (lam_log_max - lam_log_min)))

        delta = float(a[-1]) * self.cfg.delta_scale
        return mu_vec, lam, delta

    def _build_feat(self, e, y_filter, sig_pow):
        M = self.order
        e_sq = e * e
        feat = np.zeros(12 + M, dtype=np.float32)
        feat[0] = np.tanh(e * 5.0)
        feat[1] = np.tanh(e_sq * 2.0)
        feat[2] = 0.0
        feat[3] = 0.0
        feat[4] = np.tanh(np.log1p(sig_pow / M + 1e-8))
        feat[5] = np.tanh(np.log1p(e_sq + 1e-8))
        feat[6] = 0.0
        feat[7] = np.tanh(-0.5 * 4.0)
        feat[8] = np.tanh((0.9 - 0.85) * 10.0)
        feat[9] = 0.0
        feat[10] = 0.0
        feat[11] = np.tanh(y_filter * 2.0)
        feat[12:] = np.tanh(self.w[:M] * 0.1)
        return feat

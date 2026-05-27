"""Fully differentiable Leaky-NLMS in PyTorch.

Enables BPTT through the filter update so gradient flows from loss
through (mu, lambda) -> w -> e -> loss. This is the core of the
hybrid BPTT+RL training: BPTT provides structured gradient signal,
RL (PPO) provides non-differentiable reward optimization.

The filter maintains:
  w_{t+1} = lambda_t * w_t + (mu_t / (x_t^T x_t + eps)) * e_t * x_t
  e_t     = d_t - w_t^T x_t
  y_t     = w_t^T x_t

All operations are differentiable w.r.t. (mu_t, lambda_t).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffNLMSConfig:
    order: int = 16
    mu_min: float = 0.005
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 1.0
    eps: float = 1e-6
    max_w_norm: float = 100.0


def decode_action_bptt(a: torch.Tensor, cfg: DiffNLMSConfig):
    """Decode raw actions in [-1,1] to (mu, lambda) via log-scale interp.

    Args:
        a: (..., 2) tensor of raw actions in [-1, 1]
    Returns:
        mu: (...,) tensor
        lam: (...,) tensor
    """
    a = torch.clamp(a, -1.0, 1.0)
    log_mu_min = torch.tensor(np.log(cfg.mu_min), dtype=a.dtype, device=a.device)
    log_mu_max = torch.tensor(np.log(cfg.mu_max), dtype=a.dtype, device=a.device)
    log_lam_min = torch.tensor(np.log(cfg.lam_min), dtype=a.dtype, device=a.device)
    log_lam_max = torch.tensor(np.log(cfg.lam_max), dtype=a.dtype, device=a.device)

    mu_frac = (a[..., 0] + 1.0) * 0.5
    mu = torch.exp(log_mu_min + mu_frac * (log_mu_max - log_mu_min))

    lam_frac = (a[..., 1] + 1.0) * 0.5
    lam = torch.exp(log_lam_min + lam_frac * (log_lam_max - log_lam_min))

    return mu, lam


class DifferentiableNLMS(nn.Module):
    """Batched differentiable NLMS filter.

    Processes B episodes in parallel, each of length T.
    The controller produces actions at each timestep; we step the filter
    and accumulate the BPTT loss.
    """
    def __init__(self, cfg: DiffNLMSConfig | None = None):
        super().__init__()
        self.cfg = cfg or DiffNLMSConfig()

    def forward(self, actions: torch.Tensor, noisy: torch.Tensor,
                clean: torch.Tensor, trunc_bptt: int = 64,
                per_tap: bool = False) -> dict:
        """Run the filter for B episodes of length T.

        Args:
            actions: (B, T, 2) or (B, T, order+1) raw controller outputs
            noisy: (B, T) noisy input signal
            clean: (B, T) desired clean signal
            trunc_bptt: truncation horizon for BPTT
            per_tap: if True, actions[..., :order] are per-tap mu
        Returns:
            dict with keys: errors, mu_seq, lam_seq, mse_loss (scalar)
        """
        B, T = noisy.shape
        M = self.cfg.order
        eps = self.cfg.eps

        w = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)
        x_buf = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)
        last_e = torch.zeros(B, dtype=noisy.dtype, device=noisy.device)
        last2_e = torch.zeros(B, dtype=noisy.dtype, device=noisy.device)
        ema_e2 = torch.zeros(B, dtype=noisy.dtype, device=noisy.device)

        errors = []
        mu_list = []
        lam_list = []
        grad_norms = []
        chunk_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)
        total_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy[:, t]
            d = clean[:, t]

            y = (w * x_buf).sum(dim=1)
            e = d - y

            if per_tap:
                mu, lam = decode_action_bptt(actions[:, t, :], self.cfg)
            else:
                mu, lam = decode_action_bptt(actions[:, t, :], self.cfg)

            input_norm = (x_buf * x_buf).sum(dim=1) + eps
            w = lam.unsqueeze(1) * w + (mu / input_norm).unsqueeze(1) * e.unsqueeze(1) * x_buf

            w_norm = torch.norm(w, dim=1)
            clip_mask = w_norm > self.cfg.max_w_norm
            if clip_mask.any():
                scale = torch.where(clip_mask, self.cfg.max_w_norm / (w_norm + 1e-8), torch.ones_like(w_norm))
                w = w * scale.unsqueeze(1)

            errors.append(e)
            mu_list.append(mu)
            lam_list.append(lam)
            chunk_loss = chunk_loss + (e * e).mean()

            if (t + 1) % trunc_bptt == 0 or t == T - 1:
                chunk_loss = chunk_loss / min(trunc_bptt, t + 1)
                chunk_loss.backward(retain_graph=False)
                total_loss = total_loss + chunk_loss.detach()
                chunk_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)
                w = w.detach()
                x_buf = x_buf.detach()
                last_e = last_e.detach()
                last2_e = last2_e.detach()
                ema_e2 = ema_e2.detach()

        return {
            "errors": torch.stack(errors, dim=1),
            "mu_seq": torch.stack(mu_list, dim=1),
            "lam_seq": torch.stack(lam_list, dim=1),
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def inference(self, actions: torch.Tensor, noisy: torch.Tensor,
                  clean: torch.Tensor, per_tap: bool = False) -> dict:
        """Non-gradient inference pass. Returns errors and filter output."""
        B, T = noisy.shape
        M = self.cfg.order
        eps = self.cfg.eps

        w = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)
        x_buf = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)
        errors = []
        outputs = []
        mu_list = []
        lam_list = []

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy[:, t]
            d = clean[:, t]

            y = (w * x_buf).sum(dim=1)
            e = d - y

            mu, lam = decode_action_bptt(actions[:, t, :], self.cfg)
            input_norm = (x_buf * x_buf).sum(dim=1) + eps
            w = lam.unsqueeze(1) * w + (mu / input_norm).unsqueeze(1) * e.unsqueeze(1) * x_buf

            w_norm = torch.norm(w, dim=1)
            clip_mask = w_norm > self.cfg.max_w_norm
            if clip_mask.any():
                scale = torch.where(clip_mask, self.cfg.max_w_norm / (w_norm + 1e-8), torch.ones_like(w_norm))
                w = w * scale.unsqueeze(1)

            errors.append(e)
            outputs.append(y)
            mu_list.append(mu)
            lam_list.append(lam)

        return {
            "errors": torch.stack(errors, dim=1),
            "outputs": torch.stack(outputs, dim=1),
            "mu_seq": torch.stack(mu_list, dim=1),
            "lam_seq": torch.stack(lam_list, dim=1),
        }


class DiffNLMSFilterWrapper:
    """NumPy inference wrapper around DifferentiableNLMS for eval compat."""
    def __init__(self, controller: nn.Module, cfg: DiffNLMSConfig | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or DiffNLMSConfig()
        self.controller = controller
        self.device = device
        self.order = self.cfg.order
        self.w = np.zeros(self.order, dtype=np.float64)
        self.x_buf = np.zeros(self.order, dtype=np.float64)
        self.state = None

    def reset(self) -> None:
        self.w[:] = 0.0
        self.x_buf[:] = 0.0
        self.state = None

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = u[0] if u.ndim == 0 else u[0]
        y = float(self.w @ self.x_buf)
        e = d - y

        feat = self._build_feat(e, float(self.x_buf @ self.x_buf))
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            a, self.state = self.controller(feat_t, self.state)[:2]
        a_np = a[0, 0].cpu().numpy()

        log_min, log_max = np.log(self.cfg.mu_min), np.log(self.cfg.mu_max)
        mu = float(np.exp(log_min + (a_np[0] + 1) * 0.5 * (log_max - log_min)))
        lam_log_min, lam_log_max = np.log(self.cfg.lam_min), np.log(1.0)
        lam = float(np.exp(lam_log_min + (a_np[1] + 1) * 0.5 * (lam_log_max - lam_log_min)))

        norm = float(self.x_buf @ self.x_buf) + self.cfg.eps
        self.w = lam * self.w + (mu / norm) * e * self.x_buf
        return y, e

    def _build_feat(self, e, sig_pow):
        e_sq = e * e
        return np.array([np.tanh(e * 5.0), np.tanh(np.log1p(sig_pow)),
                         np.tanh(np.log1p(e_sq))], dtype=np.float32)

"""Differentiable Proportionate NLMS (PNLMS) for BPTT+RL training.

Core idea: instead of a single scalar mu, the controller outputs per-tap
step sizes mu_t[0..M-1] plus a global leakage lambda. This lets the RL
agent learn *which taps to emphasize* for each noise regime — the key
expressiveness advantage over scalar-mu NLMS.

Filter update:
  w_{t+1} = lam_t * w_t + diag(mu_t / (x_t^2 + eps)) * e_t * x_t

where mu_t is a vector of M step sizes (one per tap).

The functional expansion adds nonlinear basis functions to the input
buffer, giving the linear filter access to second-order statistics:
  x_expanded = [x, x^2, sign(x)*sqrt(|x|)]
This is standard Functionally Expanded Adaptive Filtering (Sicuranza 2006).

Together, PNLMS + functional expansion + deep RL = the architecture that
can beat Meta-AF/CNN while remaining a real-time causal adaptive filter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


@dataclass
class DiffPNLMSConfig:
    order: int = 16
    mu_min: float = 0.001
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 1.0
    eps: float = 1e-6
    max_w_norm: float = 100.0
    use_fx: bool = True
    fx_order: int = 3


def decode_action_pnlms(a: torch.Tensor, cfg: DiffPNLMSConfig):
    """Decode raw actions in [-1,1] to (mu_vec, lambda).

    Args:
        a: (..., M+1) tensor — first M values are per-tap mu, last is lambda
    Returns:
        mu_vec: (..., M) tensor of per-tap step sizes
        lam: (...,) scalar leakage
    """
    a = torch.clamp(a, -1.0, 1.0)
    log_mu_min = torch.tensor(np.log(cfg.mu_min), dtype=a.dtype, device=a.device)
    log_mu_max = torch.tensor(np.log(cfg.mu_max), dtype=a.dtype, device=a.device)
    log_lam_min = torch.tensor(np.log(cfg.lam_min), dtype=a.dtype, device=a.device)
    log_lam_max = torch.tensor(np.log(cfg.lam_max), dtype=a.dtype, device=a.device)

    M = a.shape[-1] - 1
    mu_frac = (a[..., :M] + 1.0) * 0.5
    mu_vec = torch.exp(log_mu_min + mu_frac * (log_mu_max - log_mu_min))

    lam_frac = (a[..., -1] + 1.0) * 0.5
    lam = torch.exp(log_lam_min + lam_frac * (log_lam_max - log_lam_min))

    return mu_vec, lam


def functional_expand(x_buf: torch.Tensor, fx_order: int = 3):
    """Expand input buffer with nonlinear basis functions.

    Args:
        x_buf: (B, M) input buffer
        fx_order: number of expansion terms (1=linear only, 2=+x^2, 3=+sign*sqrt)
    Returns:
        (B, M*fx_order) expanded buffer
    """
    parts = [x_buf]
    if fx_order >= 2:
        parts.append(x_buf * x_buf)
    if fx_order >= 3:
        parts.append(torch.sign(x_buf) * torch.sqrt(torch.abs(x_buf) + 1e-8))
    return torch.cat(parts, dim=-1)


class DifferentiablePNLMS(nn.Module):
    """Batched differentiable PNLMS filter for BPTT training.

    The controller produces per-tap step sizes + leakage at each timestep.
    With functional expansion, the effective filter order is M*fx_order
    but the PNLMS action dimension is M+1 (not M*fx_order+1).
    The functional expansion is internal to the filter — the controller
    still outputs M+1 actions.
    """
    def __init__(self, cfg: DiffPNLMSConfig | None = None):
        super().__init__()
        self.cfg = cfg or DiffPNLMSConfig()

    def forward(self, actions: torch.Tensor, noisy: torch.Tensor,
                clean: torch.Tensor, trunc_bptt: int = 64) -> dict:
        """Run PNLMS filter for B episodes of length T.

        Args:
            actions: (B, T, M+1) raw controller outputs
            noisy: (B, T) noisy input signal
            clean: (B, T) desired clean signal
            trunc_bptt: truncation horizon for BPTT
        Returns:
            dict with keys: errors, mu_seq, lam_seq, total_loss
        """
        B, T = noisy.shape
        M = self.cfg.order
        eps = self.cfg.eps
        use_fx = self.cfg.use_fx
        fx_order = self.cfg.fx_order

        if use_fx:
            M_eff = M * fx_order
        else:
            M_eff = M

        w = torch.zeros(B, M_eff, dtype=noisy.dtype, device=noisy.device)
        x_buf = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)

        errors = []
        mu_list = []
        lam_list = []
        chunk_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)
        total_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy[:, t]
            d = clean[:, t]

            if use_fx:
                x_exp = functional_expand(x_buf, fx_order)
            else:
                x_exp = x_buf

            y = (w * x_exp).sum(dim=1)
            e = d - y

            mu_vec, lam = decode_action_pnlms(actions[:, t, :], self.cfg)

            if use_fx:
                tap_norm = (x_exp * x_exp).sum(dim=1, keepdim=True) + eps
            else:
                tap_norm = (x_buf * x_buf).sum(dim=1, keepdim=True) + eps

            if use_fx:
                mu_expanded = mu_vec.repeat(1, fx_order)
            else:
                mu_expanded = mu_vec

            w = lam.unsqueeze(1) * w + (mu_expanded / tap_norm) * e.unsqueeze(1) * x_exp

            w_norm = torch.norm(w, dim=1)
            clip_mask = w_norm > self.cfg.max_w_norm
            if clip_mask.any():
                scale = torch.where(clip_mask, self.cfg.max_w_norm / (w_norm + 1e-8),
                                    torch.ones_like(w_norm))
                w = w * scale.unsqueeze(1)

            errors.append(e)
            mu_list.append(mu_vec)
            lam_list.append(lam)
            chunk_loss = chunk_loss + (e * e).mean()

            if (t + 1) % trunc_bptt == 0 or t == T - 1:
                chunk_loss = chunk_loss / min(trunc_bptt, t + 1)
                chunk_loss.backward()
                total_loss = total_loss + chunk_loss.detach()
                chunk_loss = torch.tensor(0.0, dtype=noisy.dtype, device=noisy.device)
                w = w.detach()
                x_buf = x_buf.detach()

        return {
            "errors": torch.stack(errors, dim=1),
            "mu_seq": torch.stack(mu_list, dim=1),
            "lam_seq": torch.stack(lam_list, dim=1),
            "total_loss": total_loss,
        }

    @torch.no_grad()
    def inference(self, actions: torch.Tensor, noisy: torch.Tensor,
                  clean: torch.Tensor) -> dict:
        """Non-gradient inference pass."""
        B, T = noisy.shape
        M = self.cfg.order
        eps = self.cfg.eps
        use_fx = self.cfg.use_fx
        fx_order = self.cfg.fx_order

        if use_fx:
            M_eff = M * fx_order
        else:
            M_eff = M

        w = torch.zeros(B, M_eff, dtype=noisy.dtype, device=noisy.device)
        x_buf = torch.zeros(B, M, dtype=noisy.dtype, device=noisy.device)
        errors = []
        outputs = []
        mu_list = []
        lam_list = []

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy[:, t]
            d = clean[:, t]

            if use_fx:
                x_exp = functional_expand(x_buf, fx_order)
            else:
                x_exp = x_buf

            y = (w * x_exp).sum(dim=1)
            e = d - y

            mu_vec, lam = decode_action_pnlms(actions[:, t, :], self.cfg)

            if use_fx:
                tap_norm = (x_exp * x_exp).sum(dim=1, keepdim=True) + eps
            else:
                tap_norm = (x_buf * x_buf).sum(dim=1, keepdim=True) + eps

            if use_fx:
                mu_expanded = mu_vec.repeat(1, fx_order)
            else:
                mu_expanded = mu_vec

            w = lam.unsqueeze(1) * w + (mu_expanded / tap_norm) * e.unsqueeze(1) * x_exp

            w_norm = torch.norm(w, dim=1)
            clip_mask = w_norm > self.cfg.max_w_norm
            if clip_mask.any():
                scale = torch.where(clip_mask, self.cfg.max_w_norm / (w_norm + 1e-8),
                                    torch.ones_like(w_norm))
                w = w * scale.unsqueeze(1)

            errors.append(e)
            outputs.append(y)
            mu_list.append(mu_vec)
            lam_list.append(lam)

        return {
            "errors": torch.stack(errors, dim=1),
            "outputs": torch.stack(outputs, dim=1),
            "mu_seq": torch.stack(mu_list, dim=1),
            "lam_seq": torch.stack(lam_list, dim=1),
        }


class PNLMSFilterWrapper:
    """NumPy inference wrapper for evaluation compatibility."""

    def __init__(self, controller: nn.Module, cfg: DiffPNLMSConfig | None = None,
                 device: str = "cpu"):
        self.cfg = cfg or DiffPNLMSConfig()
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

        y = float(self.w @ x_exp)
        e = d - y

        feat = self._build_feat(e, float(self.x_buf @ self.x_buf))
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            a, self.state = self.controller(feat_t, self.state)[:2]
        a_np = a[0, 0].cpu().numpy()

        mu_vec, lam = self._decode_action(a_np)

        tap_norm = float(x_exp @ x_exp) + self.cfg.eps
        self.w = lam * self.w + (mu_vec / tap_norm) * e * x_exp

        w_norm = float(np.linalg.norm(self.w))
        if w_norm > self.cfg.max_w_norm:
            self.w *= self.cfg.max_w_norm / w_norm

        return y, e

    def _fx_expand(self, x: np.ndarray) -> np.ndarray:
        parts = [x]
        if self.cfg.fx_order >= 2:
            parts.append(x * x)
        if self.cfg.fx_order >= 3:
            parts.append(np.sign(x) * np.sqrt(np.abs(x) + 1e-8))
        return np.concatenate(parts)

    def _decode_action(self, a: np.ndarray):
        a = np.clip(a, -1.0, 1.0)
        M = self.order
        log_min, log_max = np.log(self.cfg.mu_min), np.log(self.cfg.mu_max)
        mu_frac = (a[:M] + 1.0) * 0.5
        mu_vec = np.exp(log_min + mu_frac * (log_max - log_min))

        lam_log_min, lam_log_max = np.log(self.cfg.lam_min), np.log(self.cfg.lam_max)
        lam_frac = (a[-1] + 1.0) * 0.5
        lam = float(np.exp(lam_log_min + lam_frac * (lam_log_max - lam_log_min)))

        if self.cfg.use_fx:
            mu_vec = np.repeat(mu_vec, self.cfg.fx_order)

        return mu_vec, lam

    def _build_feat(self, e, sig_pow):
        e_sq = e * e
        M = self.order
        feat = np.zeros(11 + M, dtype=np.float32)
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
        feat[11:] = np.tanh(self.w[:M] * 0.1)
        return feat

"""Meta-AF style baseline: LSTM controller trained by BPTT through leaky-NLMS.

This is a closer-in-spirit reimplementation of Casebeer et al. (2022).
A small LSTM observes (e_t, log_input_power, log_error_power) and outputs
(mu_t, lambda_t) for the leaky-NLMS update. The whole loop is differentiable
in PyTorch: we backprop the per-sample squared error through truncated BPTT
to update the LSTM. No RL — pure supervised meta-learning.

The point of including this baseline: reviewers will (correctly) ask why
we don't compare to Meta-AF directly. A faithful reimpl side-by-side lets
us argue from numbers, not slides, that an RL² controller does at least
as well as a BPTT one while never seeing labels at meta-test time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from .base import AdaptiveFilter, windowize


class _MetaAFController(nn.Module):
    def __init__(self, hidden: int = 64, in_dim: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=False)
        self.head = nn.Linear(hidden, 2)
        self.hidden = hidden

    def forward(self, x, state=None):
        # x: (T, B, in_dim)
        out, state = self.lstm(x, state)
        a = torch.tanh(self.head(out))  # (T, B, 2)
        return a, state


def _decode(a, mu_min, mu_max, lam_min):
    # a in [-1, 1]; identical mapping to env._decode_action.
    log_min = np.log(mu_min); log_max = np.log(mu_max)
    mu = torch.exp(log_min + (a[..., 0] + 1) * 0.5 * (log_max - log_min))
    lam_log_min = np.log(lam_min); lam_log_max = np.log(1.0)
    lam = torch.exp(lam_log_min + (a[..., 1] + 1) * 0.5 * (lam_log_max - lam_log_min))
    return mu, lam


def train_meta_af(env_factory, n_iters: int = 4000, batch_size: int = 8,
                  episode_len: int = 2000, order: int = 16,
                  mu_min: float = 0.005, mu_max: float = 2.0,
                  lam_min: float = 0.80, lr: float = 3e-4,
                  trunc_bptt: int = 64, device: str = "cpu", verbose: bool = True):
    """Train the LSTM controller by BPTT through the filter.

    env_factory(rng) -> (clean[N], noisy[N]) numpy arrays.
    """
    net = _MetaAFController().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    eps = 1e-6

    for it in range(n_iters):
        rng = np.random.default_rng(it)
        cleans, noisys = [], []
        for b in range(batch_size):
            c, n = env_factory(np.random.default_rng(it * 1000 + b))
            cleans.append(c); noisys.append(n)
        clean_t = torch.tensor(np.stack(cleans), dtype=torch.float32, device=device)  # (B, N)
        noisy_t = torch.tensor(np.stack(noisys), dtype=torch.float32, device=device)
        B, N = clean_t.shape

        w = torch.zeros(B, order, device=device)
        x_buf = torch.zeros(B, order, device=device)
        last_e = torch.zeros(B, device=device)
        state = None
        total_loss = 0.0
        opt.zero_grad()
        chunk_loss = torch.zeros((), device=device)

        for t in range(N):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy_t[:, t]
            d = clean_t[:, t]
            y = (w * x_buf).sum(dim=1)
            e = d - y
            sig_pow = (x_buf * x_buf).mean(dim=1)
            res_pow = e * e
            feat = torch.stack([
                torch.tanh(e * 5.0),
                torch.tanh(torch.log1p(sig_pow)),
                torch.tanh(torch.log1p(res_pow)),
            ], dim=-1).unsqueeze(0)  # (1, B, 3)
            a, state = net(feat, state)
            mu, lam = _decode(a[0], mu_min, mu_max, lam_min)
            norm = (x_buf * x_buf).sum(dim=1) + eps
            w = lam.unsqueeze(1) * w + (mu / norm).unsqueeze(1) * e.unsqueeze(1) * x_buf
            chunk_loss = chunk_loss + (e * e).mean()
            last_e = e.detach()

            if (t + 1) % trunc_bptt == 0 or t == N - 1:
                chunk_loss = chunk_loss / trunc_bptt
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                total_loss += float(chunk_loss.detach())
                chunk_loss = torch.zeros((), device=device)
                w = w.detach(); x_buf = x_buf.detach()
                state = (state[0].detach(), state[1].detach())

        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(f"[meta-af] iter {it:4d}  mean_chunk_mse={total_loss / max(1, N // trunc_bptt):.4f}")

    return net


@dataclass
class MetaAFFilter(AdaptiveFilter):
    """Inference-time wrapper around a trained _MetaAFController."""
    order: int = 16
    mu_min: float = 0.005
    mu_max: float = 2.0
    lam_min: float = 0.80
    leakage_floor: float = 0.80
    eps: float = 1e-6
    net: Optional[_MetaAFController] = None
    device: str = "cpu"

    state: object = field(default=None, init=False)

    def reset(self) -> None:
        super().reset()
        self.state = None

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        u_t = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        y = float((torch.as_tensor(self.w, dtype=torch.float32,
                                   device=self.device) * u_t).sum().item())
        e = float(d - y)
        sig_pow = float(np.mean(u * u))
        res_pow = e * e
        feat = torch.tensor([[np.tanh(e * 5.0),
                              np.tanh(np.log1p(sig_pow)),
                              np.tanh(np.log1p(res_pow))]],
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, self.state = self.net(feat, self.state)
        a_np = a[0, 0].cpu().numpy()
        log_min, log_max = np.log(self.mu_min), np.log(self.mu_max)
        mu = float(np.exp(log_min + (a_np[0] + 1) * 0.5 * (log_max - log_min)))
        lam_log_min, lam_log_max = np.log(self.lam_min), np.log(1.0)
        lam = float(np.exp(lam_log_min + (a_np[1] + 1) * 0.5 * (lam_log_max - lam_log_min)))
        norm = float(u @ u) + self.eps
        self.w = lam * self.w + (mu / norm) * e * u
        return y, e

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "MetaAFFilter":
        net = _MetaAFController().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        f = cls(net=net, device=device, **kwargs)
        return f

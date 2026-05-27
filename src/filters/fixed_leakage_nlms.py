"""Leaky-NLMS with a fixed leakage factor — the natural ablation against Meta-RL.

Meta-RL learns (mu, lambda) jointly. The honest static baseline is NLMS with a
fixed lambda < 1. We expose lambda in {0.9, 0.95, 0.99, 0.999} so a reviewer
sees the best static leakage we could find and that Meta-RL still wins.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .base import AdaptiveFilter


@dataclass
class FixedLeakageNLMS(AdaptiveFilter):
    order: int = 16
    mu: float = 0.5
    leakage: float = 0.99
    eps: float = 1e-6

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        norm = float(u @ u) + self.eps
        self.w = self.leakage * self.w + (self.mu / norm) * e * u
        return y, e

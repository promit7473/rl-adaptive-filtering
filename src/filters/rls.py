"""Recursive Least Squares (RLS) adaptive filter."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .base import AdaptiveFilter


@dataclass
class RLS(AdaptiveFilter):
    forgetting: float = 0.99       # lambda in (0, 1]
    delta: float = 1.0              # initial inverse-correlation regularization
    P: np.ndarray = field(init=False)

    def reset(self) -> None:
        super().reset()
        self.P = np.eye(self.order) / self.delta

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        u = u.reshape(-1, 1)
        Pu = self.P @ u                                 # (order, 1)
        denom = self.forgetting + float(u.T @ Pu)
        k = Pu / denom                                  # gain (order, 1)
        y = float(self.w @ u.ravel())
        e = d - y
        self.w = self.w + (k.ravel() * e)
        self.P = (self.P - k @ Pu.T) / self.forgetting
        return y, e

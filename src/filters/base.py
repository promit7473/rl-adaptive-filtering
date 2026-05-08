"""Base class for adaptive filters in a system-identification setup.

Setup: clean signal x[n] is passed through unknown plant -> desired d[n] = clean.
Filter sees noisy reference u[n] = x[n] + v[n] and tries to predict d[n].
We use the simpler noise-cancellation framing:
    primary input    : d[n] = clean[n] + v[n]    (signal corrupted by noise we want to remove)
    reference input  : r[n] = v_correlated[n]    (correlated reference of the noise)
For our experiments we use the **prediction** framing where the filter predicts
the next sample of the clean signal from a window of past noisy samples.

Each filter exposes:
    .reset()
    .step(u_window, d) -> (y, e)    one-sample update
    .run(U, d) -> (y[], e[])        full sequence
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AdaptiveFilter:
    order: int = 16
    w: np.ndarray = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.order, dtype=float)

    # subclasses override
    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        raise NotImplementedError

    def run(self, U: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """U: (N, order) windowed reference. d: (N,) desired."""
        N = U.shape[0]
        y = np.zeros(N)
        e = np.zeros(N)
        for n in range(N):
            yn, en = self.step(U[n], float(d[n]))
            y[n] = yn
            e[n] = en
        return y, e


def windowize(u: np.ndarray, order: int) -> np.ndarray:
    """Build sliding windows of size `order` from a 1D sequence (zero-padded prefix).
    Returns (N, order) where row n is [u[n], u[n-1], ..., u[n-order+1]].
    """
    N = u.shape[0]
    padded = np.concatenate([np.zeros(order - 1), u])
    out = np.lib.stride_tricks.sliding_window_view(padded, order)[:N][:, ::-1]
    return np.ascontiguousarray(out)

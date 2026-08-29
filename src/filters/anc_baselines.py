"""Classical reference-based ANC baselines.

Every baseline exposes ``run(reference, primary) -> e`` where ``e`` is the residual
(the denoised output).  Adaptation uses only ``reference`` and ``primary`` -- never
the clean signal -- so these are directly comparable to the learned controller.
The clean signal is applied by the eval harness afterwards to score residual MSE.

Included:
    NLMSANC          : reference-driven normalised LMS (fixed mu)
    VSSLMSANC        : Kwong-Johnston variable step-size LMS
    RLSANC           : RLS with a fixed forgetting factor (diverges at low ff --
                       the honest failure mode of a fixed-forgetting estimator)
    KalmanANC        : Kalman filter with a fixed process noise Q (the ceiling
                       baseline; no single Q wins across families)
    NotchANC         : cascaded IIR notch on the primary (reference-free, for
                       powerline only)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .diff_kalman import kalman_anc_numpy


def _windows(reference, order):
    x = np.zeros(order)
    for n in range(len(reference)):
        x = np.roll(x, 1)
        x[0] = reference[n]
        yield x


@dataclass
class NLMSANC:
    order: int = 16
    mu: float = 0.05
    eps: float = 1e-6

    def run(self, reference, primary):
        w = np.zeros(self.order)
        x = np.zeros(self.order)
        e = np.zeros(len(primary))
        for n in range(len(primary)):
            x = np.roll(x, 1); x[0] = reference[n]
            e[n] = primary[n] - w @ x
            w = w + (self.mu / (x @ x + self.eps)) * e[n] * x
        return e


@dataclass
class VSSLMSANC:
    """Kwong-Johnston variable step-size LMS (reference-driven)."""
    order: int = 16
    mu_max: float = 0.1
    alpha: float = 0.97
    gamma: float = 1e-3
    eps: float = 1e-6

    def run(self, reference, primary):
        w = np.zeros(self.order)
        x = np.zeros(self.order)
        mu = self.mu_max
        e = np.zeros(len(primary))
        for n in range(len(primary)):
            x = np.roll(x, 1); x[0] = reference[n]
            e[n] = primary[n] - w @ x
            mu = self.alpha * mu + self.gamma * e[n] ** 2
            mu = float(np.clip(mu, 0.0, self.mu_max))
            w = w + (mu / (x @ x + self.eps)) * e[n] * x
        return e


@dataclass
class RLSANC:
    order: int = 16
    forgetting: float = 0.999
    delta: float = 1.0

    def run(self, reference, primary):
        M = self.order
        w = np.zeros(M); x = np.zeros(M)
        P = np.eye(M) / self.delta
        ff = self.forgetting
        e = np.zeros(len(primary))
        for n in range(len(primary)):
            x = np.roll(x, 1); x[0] = reference[n]
            Pi = P @ x
            k = Pi / (ff + x @ Pi)
            e[n] = primary[n] - w @ x
            w = w + k * e[n]
            P = (P - np.outer(k, Pi)) / ff
            if not np.isfinite(w).all():          # windup -> divergence
                w = np.nan_to_num(w); P = np.eye(M) / self.delta
        return e


@dataclass
class KalmanANC:
    order: int = 16
    q: float = 1e-6
    r: float = 1.0
    p0: float = 1.0

    def run(self, reference, primary):
        return kalman_anc_numpy(reference, primary, order=self.order,
                                q=self.q, r=self.r, p0=self.p0)


@dataclass
class NotchANC:
    """Reference-free cascaded second-order IIR notch (powerline harmonics)."""
    f0: float = 50.0
    fs: float = 360.0
    Q: float = 30.0
    n_harm: int = 3

    def run(self, reference, primary):
        from scipy.signal import iirnotch, lfilter
        y = np.asarray(primary, float).copy()
        for k in range(1, self.n_harm + 1):
            f = self.f0 * k
            if f >= self.fs / 2:
                break
            b, a = iirnotch(f / (self.fs / 2), self.Q)
            y = lfilter(b, a, y)
        return y

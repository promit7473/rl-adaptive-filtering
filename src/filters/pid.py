"""PID-controlled step-size for leaky-NLMS — strong heuristic baseline.

Idea: treat the squared error e^2 as a process variable; drive it toward a
small target with a PID controller acting (in log-space) on the NLMS step
size mu. Adds a non-RL adaptive-mu baseline that is more sophisticated than
fixed-mu NLMS or the textbook Kwong-VSS, so reviewers cannot dismiss our
gains as "Meta-RL beats LMS from 1985".
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .base import AdaptiveFilter


@dataclass
class PIDLeakyNLMS(AdaptiveFilter):
    order: int = 16
    leakage: float = 0.999
    mu_init: float = 0.1
    mu_min: float = 0.005
    mu_max: float = 2.0
    # Controller acts on the *change* in smoothed log(e^2): if error is
    # *rising*, increase mu; if falling, decrease. This is a stable PD law
    # that needs no absolute target (which would otherwise wind up).
    kp: float = 0.10
    kd: float = 0.20
    ema_alpha: float = 0.05
    eps: float = 1e-8

    log_mu: float = field(init=False)
    ema_log_e2: float = field(init=False)
    prev_log_e2: float = field(init=False)

    def reset(self) -> None:
        super().reset()
        self.log_mu = float(np.log(self.mu_init))
        self.ema_log_e2 = 0.0
        self.prev_log_e2 = 0.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        log_e2 = float(np.log(e * e + self.eps))
        rise = log_e2 - self.ema_log_e2          # >0 -> error trending up
        deriv = log_e2 - self.prev_log_e2
        self.prev_log_e2 = log_e2
        self.ema_log_e2 = (1 - self.ema_alpha) * self.ema_log_e2 \
                          + self.ema_alpha * log_e2
        # Increase mu when error rises above its EMA; decay back toward
        # mu_init otherwise (mild leak keeps mu bounded).
        self.log_mu += self.kp * rise + self.kd * deriv
        self.log_mu += 0.001 * (np.log(self.mu_init) - self.log_mu)
        self.log_mu = float(np.clip(self.log_mu,
                                    np.log(self.mu_min), np.log(self.mu_max)))
        mu = float(np.exp(self.log_mu))
        norm = float(u @ u) + self.eps
        self.w = self.leakage * self.w + (mu / norm) * e * u
        return y, e

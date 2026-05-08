"""Extended LMS family: vanilla LMS, NLMS, VSS-LMS variants, LMP, heuristic schedulers."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .base import AdaptiveFilter


@dataclass
class LMS(AdaptiveFilter):
    mu: float = 0.01
    leakage: float = 1.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        self.w = self.leakage * self.w + self.mu * e * u
        return y, e


@dataclass
class NLMS(AdaptiveFilter):
    mu: float = 0.5
    eps: float = 1e-6
    leakage: float = 1.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        norm = float(u @ u) + self.eps
        self.w = self.leakage * self.w + (self.mu / norm) * e * u
        return y, e


@dataclass
class VSSLMS(AdaptiveFilter):
    """Variable Step-Size LMS (Kwong & Johnston, 1992)."""
    mu_max: float = 0.1
    mu_min: float = 1e-4
    alpha: float = 0.97
    gamma: float = 1e-3
    mu: float = field(init=False)

    def reset(self) -> None:
        super().reset()
        self.mu = self.mu_max

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        new_mu = self.alpha * self.mu + self.gamma * (e * e)
        self.mu = float(np.clip(new_mu, self.mu_min, self.mu_max))
        self.w = self.w + self.mu * e * u
        return y, e


@dataclass
class AboulnasrMayyasVSS(AdaptiveFilter):
    """Robust VSS-LMS (Aboulnasr & Mayyas, 1997).
    Uses squared error autocorrelation to distinguish noise from signal."""
    mu_max: float = 0.1
    mu_min: float = 1e-4
    alpha: float = 0.97
    gamma_p: float = 0.09
    gamma_n: float = 0.1
    mu: float = field(init=False)
    prev_e_sq: float = field(init=False, default=0.0)

    def reset(self) -> None:
        super().reset()
        self.mu = self.mu_max
        self.prev_e_sq = 0.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        e_sq = e * e
        p = e_sq * self.prev_e_sq
        if p >= 0:
            new_mu = self.alpha * self.mu + self.gamma_p * p
        else:
            new_mu = self.alpha * self.mu - self.gamma_n * abs(p)
        self.mu = float(np.clip(new_mu, self.mu_min, self.mu_max))
        self.w = self.w + self.mu * e * u
        self.prev_e_sq = e_sq
        return y, e


@dataclass
class MathewsXieVSS(AdaptiveFilter):
    """Gradient-adaptive step-size LMS (Mathews & Xie, 1993).
    Adapts mu based on cross-correlation of consecutive gradient estimates."""
    mu_max: float = 0.1
    mu_min: float = 1e-4
    alpha: float = 0.97
    rho: float = 1e-3
    mu: float = field(init=False)
    prev_e: float = field(init=False, default=0.0)
    prev_u: np.ndarray = field(init=False)

    def reset(self) -> None:
        super().reset()
        self.mu = self.mu_max
        self.prev_e = 0.0
        self.prev_u = np.zeros(self.order)

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        grad_corr = e * u * self.prev_e * self.prev_u
        new_mu = self.alpha * self.mu + self.rho * float(np.sum(grad_corr))
        self.mu = float(np.clip(new_mu, self.mu_min, self.mu_max))
        self.w = self.w + self.mu * e * u
        self.prev_e = e
        self.prev_u = u.copy()
        return y, e


@dataclass
class LMP(AdaptiveFilter):
    """Least Mean p-norm filter for impulsive / alpha-stable noise.
    Uses Lp norm instead of L2; p < 2 downweights large errors."""
    mu: float = 0.005
    p: float = 1.5
    eps: float = 1e-6
    leakage: float = 1.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        abs_e = abs(e) + self.eps
        gradient_weight = float(np.sign(e)) * (abs_e ** (self.p - 2.0))
        gradient_weight = np.clip(gradient_weight, -10.0, 10.0)
        self.w = self.leakage * self.w + self.mu * gradient_weight * e * u
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > 100.0:
            self.w *= 100.0 / w_norm
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        return y, e


@dataclass
class HeuristicMuScheduler(AdaptiveFilter):
    """Simple heuristic step-size scheduler — the natural baseline for RL comparison.
    
    Uses NLMS-style normalization to avoid input-power sensitivity.
    Rules:
    1. If error is very large relative to running average (impulse), drop mu
    2. If error is rising (bad tracking), increase mu
    3. If error is dropping (good convergence), slowly decay mu
    4. Apply leakage proportional to error magnitude
    """
    mu_base: float = 0.3
    mu_min: float = 1e-3
    mu_max: float = 1.0
    eps: float = 1e-6
    leakage_base: float = 1.0
    leakage_min: float = 1.0
    alpha_ema: float = 0.99
    mu: float = field(init=False)
    error_ema: float = field(init=False, default=0.0)
    prev_e_sq: float = field(init=False, default=0.0)

    def reset(self) -> None:
        super().reset()
        self.mu = self.mu_base
        self.error_ema = 0.0
        self.prev_e_sq = 0.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        e_sq = e * e
        norm = float(u @ u) + self.eps
        self.error_ema = self.alpha_ema * self.error_ema + (1.0 - self.alpha_ema) * e_sq
        ema_rms = np.sqrt(self.error_ema + 1e-8)

        if abs(e) > 5.0 * ema_rms:
            target_mu = max(self.mu * 0.1, self.mu_min)
        elif e_sq > self.prev_e_sq * 2.0 and e_sq > self.error_ema:
            target_mu = min(self.mu * 1.5, self.mu_max)
        elif e_sq < self.prev_e_sq * 0.5:
            target_mu = max(self.mu * 0.98, self.mu_min)
        else:
            target_mu = self.mu

        self.mu = float(np.clip(target_mu, self.mu_min, self.mu_max))

        err_ratio = min(abs(e) / (ema_rms + 1e-8), 3.0)
        leakage = self.leakage_base - (self.leakage_base - self.leakage_min) * (err_ratio / 3.0)
        leakage = max(leakage, self.leakage_min)

        self.w = leakage * self.w + (self.mu / norm) * e * u
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > 100.0:
            self.w *= 100.0 / w_norm
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_e_sq = e_sq
        return y, e


@dataclass
class KalmanMuScheduler(AdaptiveFilter):
    """Kalman-filter-based step-size estimation.
    
    Models mu as a random walk and estimates it from the error signal.
    Uses NLMS normalization for stability.
    """
    mu_init: float = 0.3
    q_mu: float = 1e-6
    r_mu: float = 1e-2
    eps: float = 1e-6
    leakage_base: float = 1.0
    leakage_min: float = 1.0
    mu: float = field(init=False)
    P_mu: float = field(init=False)
    error_ema: float = field(init=False, default=0.0)
    alpha_ema: float = 0.99

    def reset(self) -> None:
        super().reset()
        self.mu = self.mu_init
        self.P_mu = 1e-4
        self.error_ema = 0.0

    def step(self, u: np.ndarray, d: float) -> tuple[float, float]:
        y = float(self.w @ u)
        e = d - y
        e_sq = e * e
        norm = float(u @ u) + self.eps
        self.error_ema = self.alpha_ema * self.error_ema + (1.0 - self.alpha_ema) * e_sq

        innovation = e_sq - self.error_ema
        self.P_mu += self.q_mu
        self.P_mu = min(self.P_mu, 0.1)
        K = self.P_mu / (self.P_mu + self.r_mu + 1e-8)
        mu_update = K * np.sign(innovation) * min(abs(innovation), 10.0)
        self.mu = float(np.clip(self.mu + mu_update * 0.01, 1e-3, 1.0))
        self.P_mu = max((1.0 - K) * self.P_mu, 1e-10)

        ema_rms = np.sqrt(self.error_ema + 1e-8)
        err_ratio = min(abs(e) / (ema_rms + 1e-8), 3.0)
        leakage = self.leakage_base - (self.leakage_base - self.leakage_min) * (err_ratio / 3.0)
        leakage = max(leakage, self.leakage_min)

        self.w = leakage * self.w + (self.mu / norm) * e * u
        w_norm = float(np.linalg.norm(self.w))
        if w_norm > 100.0:
            self.w *= 100.0 / w_norm
        if not np.isfinite(self.w).all():
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        return y, e

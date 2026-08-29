"""Fully differentiable Kalman adaptive filter for reference-based ANC.

The filter estimates an unknown FIR path ``w`` from a reference signal to an
interference component, modelling ``w`` as a random walk::

    state model : w_{n+1} = w_n + q_n,   q_n ~ N(0, Q_n I)      (process noise)
    measurement : primary_n = w_n^T x_n + v_n,  v_n ~ N(0, R_n) (obs noise)

Per-step recursion (x_n = reference tap vector, e_n = innovation):

    P      <- P + Q_n I                 # predict (bounded diffusion -> no windup)
    e_n    =  primary_n - w^T x_n        # innovation == denoised output
    S_n    =  x_n^T P x_n + R_n
    K_n    =  P x_n / S_n
    w      <- w + K_n e_n
    P      <- P - K_n (x_n^T P)

The controller sets ``(Q_n, R_n)`` each step: ``Q_n`` is the continuous analog of
the LMS step-size (large -> fast tracking / high misadjustment, small -> low
misadjustment / slow tracking).  Unlike RLS with a forgetting factor, additive
process noise keeps ``P`` bounded, so the filter never suffers covariance windup
-- it is numerically stable for the entire action range.

Everything is differentiable w.r.t. (Q_n, R_n), so gradients flow
loss -> e_n -> w -> P -> (Q_n, R_n) -> controller for truncated BPTT.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class DiffKalmanConfig:
    order: int = 16
    q_min: float = 1e-8
    q_max: float = 1e-3
    r_min: float = 1e-3
    r_max: float = 1e1
    p0: float = 1.0          # initial state covariance (diagonal)
    max_w_norm: float = 1e3  # safety clamp (rarely active; Kalman is stable)


def decode_action_kalman(a: torch.Tensor, cfg: DiffKalmanConfig):
    """Decode raw actions in [-1, 1] to (Q, R) via log-scale interpolation.

    Args:
        a: (..., 2) tensor; a[...,0] -> log Q, a[...,1] -> log R.
    Returns:
        q: (...,) process-noise variance per step.
        r: (...,) measurement-noise variance.
    """
    a = torch.clamp(a, -1.0, 1.0)
    lo_q, hi_q = np.log(cfg.q_min), np.log(cfg.q_max)
    lo_r, hi_r = np.log(cfg.r_min), np.log(cfg.r_max)
    q = torch.exp(lo_q + (a[..., 0] + 1.0) * 0.5 * (hi_q - lo_q))
    r = torch.exp(lo_r + (a[..., 1] + 1.0) * 0.5 * (hi_r - lo_r))
    return q, r


def kalman_step(w, P, x, primary, q, r, max_w_norm: float = 1e3,
                return_aux: bool = False):
    """One batched, differentiable Kalman ANC update.

    Shapes (B = batch): w (B, M); P (B, M, M); x (B, M); primary (B,);
    q, r (B,).  Returns (w_new, P_new, e) with e the innovation (B,).  When
    ``return_aux`` is True, also returns ``(K, S)`` -- Kalman gain (B, M) and
    innovation variance (B,) -- which the controller state features consume.
    """
    B, M = w.shape
    eye = torch.eye(M, dtype=w.dtype, device=w.device).unsqueeze(0)
    # Predict: additive process noise keeps P bounded (no windup).
    P = P + q.view(B, 1, 1) * eye
    yhat = (w * x).sum(dim=1)              # (B,)
    e = primary - yhat                     # innovation / denoised output
    Px = torch.bmm(P, x.unsqueeze(2)).squeeze(2)          # (B, M)
    S = (x * Px).sum(dim=1) + r            # (B,)
    K = Px / S.unsqueeze(1)                # (B, M)
    w = w + K * e.unsqueeze(1)
    # Joseph-free covariance update: P <- P - K (x^T P)
    xP = torch.bmm(x.unsqueeze(1), P).squeeze(1)          # (B, M)
    P = P - torch.bmm(K.unsqueeze(2), xP.unsqueeze(1))    # (B, M, M)

    w_norm = torch.norm(w, dim=1)
    clip = w_norm > max_w_norm
    if clip.any():
        scale = torch.where(clip, max_w_norm / (w_norm + 1e-8),
                            torch.ones_like(w_norm))
        w = w * scale.unsqueeze(1)
    if return_aux:
        return w, P, e, K, S
    return w, P, e


def kalman_anc_numpy(ref, primary, order=16, q=1e-6, r=1.0, p0=1.0, q_sched=None):
    """Reference NumPy implementation (for parity tests and classical baselines).

    Cancels interference in ``primary`` using ``ref``; returns the residual/error
    sequence ``e`` (the denoised output).  Uses only ref + primary -- never clean.
    """
    ref = np.asarray(ref, float)
    primary = np.asarray(primary, float)
    N = len(primary)
    w = np.zeros(order)
    x = np.zeros(order)
    P = np.eye(order) * p0
    I = np.eye(order)
    e = np.zeros(N)
    for n in range(N):
        x = np.roll(x, 1)
        x[0] = ref[n]
        qn = q if q_sched is None else q_sched[n]
        P = P + qn * I
        e[n] = primary[n] - w @ x
        S = x @ P @ x + r
        K = P @ x / S
        w = w + K * e[n]
        P = P - np.outer(K, x @ P)
    return e

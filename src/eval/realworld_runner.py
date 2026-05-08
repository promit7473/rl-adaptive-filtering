"""Core engine: run any method on an arbitrary (clean, noisy) signal pair.

This is the sim-to-real evaluation engine. The policy was trained on synthetic
signals; here we feed it real ECG / speech / RF signals at inference time with
no retraining.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


FEAT_DIM = 7
WINDOW   = 16
MU_MIN, MU_MAX       = 0.01, 1.0
LAM_MIN, LAM_MAX     = 0.80, 1.0
FILTER_ORDER         = 16
SOFTPLUS_REF         = np.log1p(np.exp(1.0))   # softplus(1)


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(np.clip(x, -30, 30))))


def _decode_action(a: np.ndarray) -> tuple[float, float]:
    a = np.clip(a, -1.0, 1.0)
    frac_mu  = (a[0] + 1.0) * 0.5
    mu       = float(np.exp(np.log(MU_MIN) + frac_mu  * (np.log(MU_MAX) - np.log(MU_MIN))))
    frac_lam = (a[1] + 1.0) * 0.5
    lam      = float(np.exp(np.log(LAM_MIN) + frac_lam * (np.log(LAM_MAX) - np.log(LAM_MIN))))
    return mu, lam


def _build_features(e: float, last_e: float, last2_e: float,
                    ema_esq: float, u_norm_sq: float) -> np.ndarray:
    de   = e - last_e
    dde  = de - (last_e - last2_e)
    autocorr = (e * last_e) / (max(ema_esq, 1e-8))
    log_sig_pow = np.log1p(max(u_norm_sq / FILTER_ORDER, 1e-12))
    log_err_pow = np.log1p(e * e)
    return np.array([e, e*e, de, dde, log_sig_pow, log_err_pow, autocorr], dtype=np.float32)


@dataclass
class RunResult:
    errors:    np.ndarray   # per-sample |e[n]|
    mu_seq:    np.ndarray   # per-sample mu_t  (NaN for fixed-param methods)
    lam_seq:   np.ndarray   # per-sample lam_t
    ss_mse_db: float        # 10*log10(mean(e^2) over last 25%)


def run_nlms_filter(clean: np.ndarray, noisy: np.ndarray,
                    mu: float = 0.5, leakage: float = 1.0) -> RunResult:
    """Fixed-param leaky-NLMS."""
    N, M = len(clean), FILTER_ORDER
    w  = np.zeros(M)
    e_seq = np.zeros(N)
    for n in range(N):
        u = noisy[max(0, n-M+1):n+1][::-1]
        if len(u) < M: u = np.concatenate([np.zeros(M - len(u)), u])
        yhat    = w @ u
        e       = clean[n] - yhat
        norm    = u @ u + 1e-8
        w       = leakage * w + (mu / norm) * e * u
        e_seq[n] = e
    ss = int(0.75 * N)
    ss_mse = float(np.mean(e_seq[ss:] ** 2))
    db = 10 * np.log10(max(ss_mse, 1e-12))
    return RunResult(np.abs(e_seq), np.full(N, mu), np.full(N, leakage), db)


def run_vss_kwong(clean: np.ndarray, noisy: np.ndarray,
                  mu0: float = 0.1, alpha: float = 0.9,
                  gamma: float = 0.99) -> RunResult:
    """VSS-LMS Kwong-Johnston: mu adapts based on gradient correlation."""
    N, M = len(clean), FILTER_ORDER
    w, mu = np.zeros(M), mu0
    grad_prev = np.zeros(M)
    e_seq, mu_seq = np.zeros(N), np.zeros(N)
    for n in range(N):
        u = noisy[max(0, n-M+1):n+1][::-1]
        if len(u) < M: u = np.concatenate([np.zeros(M - len(u)), u])
        norm     = u @ u + 1e-8
        yhat     = w @ u
        e        = clean[n] - yhat
        grad     = e * u / norm
        corr     = float(np.dot(grad, grad_prev))
        mu       = float(np.clip(alpha * mu + (1 - alpha) * gamma * corr, 1e-5, 1.0))
        w        = w + mu * grad
        grad_prev = grad
        e_seq[n] = e
        mu_seq[n] = mu
    ss = int(0.75 * N)
    db = 10 * np.log10(max(float(np.mean(e_seq[ss:]**2)), 1e-12))
    return RunResult(np.abs(e_seq), mu_seq, np.ones(N), db)


def run_heuristic_scheduler(clean: np.ndarray, noisy: np.ndarray) -> RunResult:
    """Rule-based mu adaptation with NLMS normalization."""
    N, M = len(clean), FILTER_ORDER
    w    = np.zeros(M)
    mu   = 0.05
    ema  = 0.01
    e_seq, mu_seq = np.zeros(N), np.zeros(N)
    for n in range(N):
        u = noisy[max(0, n-M+1):n+1][::-1]
        if len(u) < M: u = np.concatenate([np.zeros(M - len(u)), u])
        norm  = u @ u + 1e-8
        yhat  = w @ u
        e     = clean[n] - yhat
        esq   = e * e
        ema   = 0.95 * ema + 0.05 * esq
        if   esq > 10 * ema: mu = max(mu * 0.5,  0.001)   # impulse: shrink fast
        elif esq > 2  * ema: mu = min(mu * 1.2,  0.8)     # rising: grow
        else:                mu = max(mu * 0.99, 0.005)    # quiet: decay slowly
        w       = w + (mu / norm) * e * u
        e_seq[n] = e
        mu_seq[n] = mu
    ss = int(0.75 * N)
    db = 10 * np.log10(max(float(np.mean(e_seq[ss:]**2)), 1e-12))
    return RunResult(np.abs(e_seq), mu_seq, np.ones(N), db)


def run_rl_policy(clean: np.ndarray, noisy: np.ndarray,
                  policy,                          # SB3 RecurrentPPO or PPO
                  recurrent: bool = True) -> RunResult:
    """Run trained RL policy on an arbitrary (clean, noisy) signal pair.

    Zero-shot: policy was never retrained on this signal type.
    """
    N, M = len(clean), FILTER_ORDER
    w        = np.zeros(M)
    feat_buf = np.zeros((WINDOW, FEAT_DIM), dtype=np.float32)
    last_e   = 0.0
    last2_e  = 0.0
    ema_esq  = 0.01
    e_seq    = np.zeros(N)
    mu_seq   = np.zeros(N)
    lam_seq  = np.zeros(N)

    # SB3 recurrent state
    lstm_states  = None
    episode_start = np.array([True])

    for n in range(N):
        u = noisy[max(0, n-M+1):n+1][::-1]
        if len(u) < M: u = np.concatenate([np.zeros(M - len(u)), u])

        yhat  = w @ u
        e     = clean[n] - yhat
        ema_esq = 0.99 * ema_esq + 0.01 * e * e

        # build feature, shift window
        feat = _build_features(e, last_e, last2_e, ema_esq, float(u @ u))
        feat_buf = np.roll(feat_buf, -1, axis=0)
        feat_buf[-1] = feat

        obs = feat_buf.flatten().astype(np.float32)

        if recurrent:
            action, lstm_states = policy.predict(
                obs, state=lstm_states,
                episode_start=episode_start, deterministic=True)
            episode_start = np.array([False])
        else:
            action, _ = policy.predict(obs, deterministic=True)

        mu, lam = _decode_action(action)
        norm    = u @ u + 1e-8
        w       = lam * w + (mu / norm) * e * u

        last2_e  = last_e
        last_e   = e
        e_seq[n] = e
        mu_seq[n] = mu
        lam_seq[n] = lam

    ss = int(0.75 * N)
    db = 10 * np.log10(max(float(np.mean(e_seq[ss:]**2)), 1e-12))
    return RunResult(np.abs(e_seq), mu_seq, lam_seq, db)

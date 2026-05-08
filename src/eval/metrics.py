"""Metrics for adaptive filtering evaluation."""
from __future__ import annotations
import numpy as np


def mse(e: np.ndarray) -> float:
    return float(np.mean(e ** 2))


def steady_state_mse(e: np.ndarray, frac: float = 0.25) -> float:
    tail = max(1, int(len(e) * frac))
    return float(np.mean(e[-tail:] ** 2))


def snr_improvement_db(clean: np.ndarray, noisy: np.ndarray, recovered: np.ndarray) -> float:
    """SNR_out - SNR_in in dB. recovered ~ clean estimate; noisy = clean + v."""
    eps = 1e-12
    snr_in = 10 * np.log10(np.mean(clean ** 2) / (np.mean((noisy - clean) ** 2) + eps) + eps)
    snr_out = 10 * np.log10(np.mean(clean ** 2) / (np.mean((recovered - clean) ** 2) + eps) + eps)
    return float(snr_out - snr_in)


def convergence_time(e: np.ndarray, target_db: float = -15.0,
                     window: int = 100) -> int:
    """Samples needed for moving-average MSE (in dB) to drop below target_db.
    Returns len(e) if never reached.
    """
    if window < 1:
        window = 1
    e2 = e ** 2
    if len(e2) < window:
        return len(e)
    csum = np.cumsum(e2)
    ma = (csum[window - 1:] - np.concatenate([[0.0], csum[:-window]])) / window
    ma_db = 10 * np.log10(ma + 1e-12)
    hits = np.where(ma_db < target_db)[0]
    return int(hits[0] + window) if len(hits) > 0 else len(e)


def summarize(e: np.ndarray, clean: np.ndarray | None = None,
              noisy: np.ndarray | None = None,
              recovered: np.ndarray | None = None) -> dict:
    out = {
        "mse": mse(e),
        "ss_mse": steady_state_mse(e),
        "ss_mse_db": 10 * np.log10(steady_state_mse(e) + 1e-12),
        "conv_time": convergence_time(e),
    }
    if clean is not None and noisy is not None and recovered is not None:
        out["snr_improvement_db"] = snr_improvement_db(clean, noisy, recovered)
    return out

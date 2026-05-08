"""Noise families.

Train (in-distribution): gaussian, colored (pink/brown), impulsive (Bernoulli-Gaussian),
time-varying SNR.

Held-out (OOD test): alpha-stable, burst, chirp interferer, regime-switch.

Each function returns a noise sequence with approximately the requested SNR (dB)
relative to the supplied clean signal. Some families (alpha-stable, burst) use
power matched to the average requested SNR over the sequence.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sps


# ---------- helpers ----------

def signal_power(x: np.ndarray) -> float:
    return float(np.mean(x ** 2) + 1e-12)


def scale_to_snr(noise: np.ndarray, clean: np.ndarray, snr_db: float) -> np.ndarray:
    ps = signal_power(clean)
    pn = signal_power(noise)
    target_pn = ps / (10 ** (snr_db / 10))
    return noise * np.sqrt(target_pn / pn)


# ---------- train families ----------

def gaussian(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(clean.shape[0])
    return scale_to_snr(n, clean, snr_db)


def colored(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
            color: str = "pink") -> np.ndarray:
    """1/f^alpha noise. color in {'pink' (alpha=1), 'brown' (alpha=2)}."""
    alpha = {"pink": 1.0, "brown": 2.0}[color]
    n = clean.shape[0]
    white = rng.standard_normal(n)
    # frequency-domain shaping
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = np.fft.rfft(white) / (freqs ** (alpha / 2.0))
    out = np.fft.irfft(spectrum, n=n)
    return scale_to_snr(out, clean, snr_db)


def impulsive(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
              p_impulse: float = 0.02, impulse_gain: float = 8.0) -> np.ndarray:
    """Bernoulli-Gaussian: Gaussian background + sparse high-amplitude spikes."""
    n = clean.shape[0]
    bg = rng.standard_normal(n)
    mask = rng.random(n) < p_impulse
    spikes = rng.standard_normal(n) * impulse_gain
    raw = bg + mask * spikes
    return scale_to_snr(raw, clean, snr_db)


def time_varying_snr(clean: np.ndarray, snr_db_low: float, snr_db_high: float,
                     rng: np.random.Generator, n_segments: int = 4) -> np.ndarray:
    """Piecewise-stationary Gaussian with SNR varying per segment."""
    n = clean.shape[0]
    bounds = np.linspace(0, n, n_segments + 1, dtype=int)
    out = np.zeros(n)
    for i in range(n_segments):
        a, b = bounds[i], bounds[i + 1]
        snr = rng.uniform(snr_db_low, snr_db_high)
        seg_clean = clean[a:b]
        seg_noise = rng.standard_normal(b - a)
        out[a:b] = scale_to_snr(seg_noise, seg_clean, snr) if signal_power(seg_clean) > 1e-9 else seg_noise
    return out


# ---------- held-out (OOD) families ----------

def alpha_stable(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
                 alpha: float = 1.5) -> np.ndarray:
    """Symmetric alpha-stable noise via CMS method (heavy-tailed)."""
    n = clean.shape[0]
    U = (rng.random(n) - 0.5) * np.pi
    W = -np.log(rng.random(n) + 1e-12)
    if abs(alpha - 1.0) < 1e-3:
        x = np.tan(U)
    else:
        x = (np.sin(alpha * U) / (np.cos(U) ** (1.0 / alpha))) * \
            (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha)
    # clip extreme values for stable power scaling
    x = np.clip(x, -50, 50)
    return scale_to_snr(x, clean, snr_db)


def burst(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
          burst_rate: float = 3.0, burst_len_range=(40, 200),
          burst_gain: float = 5.0) -> np.ndarray:
    """Bursty noise: quiet baseline + intermittent high-power Gaussian bursts."""
    n = clean.shape[0]
    out = 0.1 * rng.standard_normal(n)
    n_bursts = max(1, int(rng.poisson(burst_rate)))
    for _ in range(n_bursts):
        L = int(rng.integers(burst_len_range[0], burst_len_range[1]))
        start = int(rng.integers(0, max(1, n - L)))
        out[start:start + L] += burst_gain * rng.standard_normal(L)
    return scale_to_snr(out, clean, snr_db)


def chirp_interferer(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
                     fs: float = 8000.0,
                     f0_range=(100.0, 500.0), f1_range=(800.0, 2000.0)) -> np.ndarray:
    """Linear chirp added to mild Gaussian background."""
    n = clean.shape[0]
    f0 = rng.uniform(*f0_range)
    f1 = rng.uniform(*f1_range)
    t = np.arange(n) / fs
    T = n / fs
    k = (f1 - f0) / T
    chirp_sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    bg = 0.2 * rng.standard_normal(n)
    return scale_to_snr(chirp_sig + bg, clean, snr_db)


def regime_switch(clean: np.ndarray, snr_db_options, rng: np.random.Generator,
                  n_regimes: int = 3, family_pool=("gaussian", "colored", "impulsive")) -> np.ndarray:
    """Hard switches between different noise families and SNRs mid-episode."""
    n = clean.shape[0]
    bounds = np.sort(rng.choice(np.arange(n // 8, n - n // 8), size=n_regimes - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [n]])
    out = np.zeros(n)
    for i in range(n_regimes):
        a, b = int(bounds[i]), int(bounds[i + 1])
        seg_clean = clean[a:b]
        snr = float(rng.choice(np.atleast_1d(snr_db_options)))
        fam = rng.choice(family_pool)
        if fam == "gaussian":
            seg = gaussian(seg_clean, snr, rng) if len(seg_clean) > 0 else np.zeros(0)
        elif fam == "colored":
            seg = colored(seg_clean, snr, rng, color=rng.choice(["pink", "brown"])) if len(seg_clean) > 0 else np.zeros(0)
        else:
            seg = impulsive(seg_clean, snr, rng) if len(seg_clean) > 0 else np.zeros(0)
        out[a:b] = seg
    return out


# ---------- dispatch ----------

TRAIN_FAMILIES = ("gaussian", "colored", "impulsive", "time_varying", "regime_switch")
OOD_FAMILIES = ("alpha_stable", "burst", "chirp_interferer")


def make_noise(family: str, clean: np.ndarray, rng: np.random.Generator,
               snr_db: float = 10.0, **kwargs) -> np.ndarray:
    if family == "gaussian":
        return gaussian(clean, snr_db, rng)
    if family == "colored":
        return colored(clean, snr_db, rng, color=kwargs.get("color", "pink"))
    if family == "impulsive":
        return impulsive(clean, snr_db, rng,
                         p_impulse=kwargs.get("p_impulse", 0.02),
                         impulse_gain=kwargs.get("impulse_gain", 8.0))
    if family == "time_varying":
        return time_varying_snr(clean,
                                snr_db_low=kwargs.get("snr_db_low", 0.0),
                                snr_db_high=kwargs.get("snr_db_high", 20.0),
                                rng=rng,
                                n_segments=kwargs.get("n_segments", 4))
    if family == "alpha_stable":
        return alpha_stable(clean, snr_db, rng, alpha=kwargs.get("alpha", 1.5))
    if family == "burst":
        return burst(clean, snr_db, rng,
                     burst_rate=kwargs.get("burst_rate", 3.0),
                     burst_gain=kwargs.get("burst_gain", 5.0))
    if family == "chirp_interferer":
        return chirp_interferer(clean, snr_db, rng, fs=kwargs.get("fs", 8000.0))
    if family == "regime_switch":
        return regime_switch(clean,
                             snr_db_options=kwargs.get("snr_db_options", [0, 10, 20]),
                             rng=rng,
                             n_regimes=kwargs.get("n_regimes", 3))
    raise ValueError(f"Unknown noise family: {family}")

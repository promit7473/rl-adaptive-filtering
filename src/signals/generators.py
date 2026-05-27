"""Clean signal generators."""
from __future__ import annotations
import numpy as np


def sine(n: int, freq: float, fs: float, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase)


def multitone(n: int, freqs, amps, fs: float, phases=None) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    if phases is None:
        phases = np.zeros_like(freqs)
    t = np.arange(n) / fs
    out = np.zeros(n, dtype=float)
    for f, a, p in zip(freqs, amps, phases):
        out += a * np.sin(2 * np.pi * f * t + p)
    return out


def am_signal(n: int, fc: float, fm: float, fs: float, mod_index: float = 0.5) -> np.ndarray:
    t = np.arange(n) / fs
    carrier = np.sin(2 * np.pi * fc * t)
    msg = np.sin(2 * np.pi * fm * t)
    return (1.0 + mod_index * msg) * carrier


def chirp(n: int, f0: float, f1: float, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    T = n / fs
    k = (f1 - f0) / T
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.sin(phase)


def ecg_like(n: int, fs: float, rng: np.random.Generator,
             hr_bpm: float | None = None) -> np.ndarray:
    """Synthetic ECG: periodic Gaussian-pulse QRS spikes + smaller P/T waves.
    Trains the policy to handle high-amplitude transients that aren't noise."""
    if hr_bpm is None:
        hr_bpm = rng.uniform(50.0, 110.0)
    period_s = 60.0 / hr_bpm
    period_n = int(period_s * fs)
    t = np.arange(n) / fs
    out = np.zeros(n, dtype=float)
    qrs_w = max(2, int(0.04 * fs))
    p_w   = max(2, int(0.08 * fs))
    t_w   = max(2, int(0.16 * fs))
    centers = np.arange(period_n // 2, n, period_n)
    for c in centers:
        for off, amp, w in ((-int(0.16 * fs), 0.15, p_w),
                            (0,                1.0,  qrs_w),
                            (int(0.20 * fs),  0.30, t_w)):
            i = c + off
            if 0 <= i < n:
                lo, hi = max(0, i - 3 * w), min(n, i + 3 * w)
                tt = (np.arange(lo, hi) - i) / w
                out[lo:hi] += amp * np.exp(-0.5 * tt * tt)
    out += 0.02 * np.sin(2 * np.pi * rng.uniform(0.2, 0.5) * t)  # baseline wander
    out /= (np.std(out) + 1e-9)
    return out


def random_pulses(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Variable-shape, variable-rate pulse train. Generalizes ECG/speech/neural."""
    rate_hz = rng.uniform(1.0, 20.0)
    period_n = max(4, int(fs / rate_hz))
    out = np.zeros(n, dtype=float)
    i = int(rng.integers(0, period_n))
    while i < n:
        shape = rng.choice(("gauss", "tri", "exp"))
        amp = float(rng.uniform(0.5, 1.5)) * float(rng.choice((-1.0, 1.0)))
        w = max(2, int(rng.uniform(0.005, 0.05) * fs))
        lo, hi = max(0, i - 3 * w), min(n, i + 3 * w)
        tt = (np.arange(lo, hi) - i)
        if shape == "gauss":
            out[lo:hi] += amp * np.exp(-0.5 * (tt / w) ** 2)
        elif shape == "tri":
            out[lo:hi] += amp * np.maximum(0.0, 1.0 - np.abs(tt) / (1.5 * w))
        else:  # exp decay (one-sided)
            mask = tt >= 0
            out[lo:hi][mask] += amp * np.exp(-tt[mask] / w)
        i += int(period_n * rng.uniform(0.7, 1.3))
    out /= (np.std(out) + 1e-9)
    return out


def square_burst(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Gated tones with sharp on/off edges. Radar / motor / percussion-like."""
    f0 = float(rng.uniform(80.0, 800.0))
    t = np.arange(n) / fs
    tone = np.sin(2 * np.pi * f0 * t)
    burst_hz = float(rng.uniform(2.0, 15.0))
    duty = float(rng.uniform(0.2, 0.6))
    gate = (((t * burst_hz) % 1.0) < duty).astype(float)
    out = tone * gate
    out /= (np.std(out) + 1e-9)
    return out


def make_signal(kind: str, n: int, fs: float, rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Dispatch by name. Returns clean signal of length n."""
    if kind == "sine":
        return sine(n, kwargs.get("freq", 300.0), fs, kwargs.get("amp", 1.0))
    if kind == "multitone":
        freqs = kwargs.get("freqs", [200, 350, 700])
        amps = kwargs.get("amps", [1.0, 0.6, 0.4])
        return multitone(n, freqs, amps, fs)
    if kind == "am":
        return am_signal(n, kwargs.get("fc", 1200.0), kwargs.get("fm", 80.0), fs,
                         kwargs.get("mod_index", 0.5))
    if kind == "chirp":
        return chirp(n, kwargs.get("f0", 100.0), kwargs.get("f1", 1500.0), fs)
    if kind == "ecg_like":
        return ecg_like(n, fs, rng, hr_bpm=kwargs.get("hr_bpm"))
    if kind == "random_pulses":
        return random_pulses(n, fs, rng)
    if kind == "square_burst":
        return square_burst(n, fs, rng)
    raise ValueError(f"Unknown signal kind: {kind}")

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
    raise ValueError(f"Unknown signal kind: {kind}")

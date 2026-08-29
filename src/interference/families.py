"""Structured-interference families for reference-based ANC.

Each generator returns ``(interference, reference)``:

* ``interference`` is added to the clean signal to form the observable primary.
* ``reference`` is an observable signal correlated with the interference but
  statistically independent of the clean signal.  The adaptive filter identifies
  the (unknown, possibly time-varying) propagation path from ``reference`` to
  ``interference`` and subtracts its estimate, so the residual ``e = primary -
  w^T ref_window`` recovers the clean signal *without ever seeing it*.

``snr_db`` sets the input SNR = clean-power / interference-power.  The reference is
returned at unit power (its absolute gain is arbitrary; the filter absorbs it).

The families deliberately cover only interference that a reference can cancel:
    powerline, baseline_wander, echo  (train, stationary + slowly varying path)
    regime_switch                      (train, abrupt path change -- non-stationary)
    narrowband_chirp, echo_long        (held out for zero-shot generalisation)
"""
from __future__ import annotations
import numpy as np


# ---------------- helpers ----------------

def signal_power(x: np.ndarray) -> float:
    return float(np.mean(x ** 2) + 1e-12)


def _scale_to_snr(interference: np.ndarray, clean: np.ndarray,
                  snr_db: float) -> np.ndarray:
    """Scale interference so clean_power / interference_power == 10^(snr/10)."""
    ps = signal_power(clean)
    pi = signal_power(interference)
    target = ps / (10 ** (snr_db / 10.0))
    return interference * np.sqrt(target / pi)


def _unit_power(x: np.ndarray) -> np.ndarray:
    return x / (np.std(x) + 1e-9)


def _random_path(rng: np.random.Generator, taps: int) -> np.ndarray:
    """A short causal FIR propagation path with exponential decay + random sign."""
    h = rng.standard_normal(taps) * np.exp(-np.arange(taps) / max(1.0, taps / 3.0))
    h[0] += 1.0  # dominant direct path
    return h / (np.linalg.norm(h) + 1e-9)


def _apply_path(ref: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution ref * h, truncated to len(ref)."""
    return np.convolve(ref, h)[: len(ref)]


# ---------------- families ----------------

def powerline(clean, snr_db, rng, fs):
    """Mains interference: fundamental (50 or 60 Hz) + harmonics with an unknown
    coupling path and slow amplitude drift.  Reference carries the same harmonic
    lines at unit amplitude and unknown phase (a mains tap)."""
    n = len(clean)
    t = np.arange(n) / fs
    f0 = float(rng.choice([50.0, 60.0]))
    n_harm = int(rng.integers(2, 4))
    ks = np.arange(1, n_harm + 1)

    # reference: harmonic lines, unit-ish, reference phases
    ref = np.zeros(n)
    for k in ks:
        ref += np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))

    # interference: same lines, *different* per-harmonic gains & phases (the path)
    # plus a slow amplitude modulation so the path is mildly non-stationary.
    interf = np.zeros(n)
    for k in ks:
        gain = rng.uniform(0.3, 1.0) / k
        interf += gain * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    interf *= 1.0 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 2 * np.pi))

    return _scale_to_snr(interf, clean, snr_db), _unit_power(ref)


def baseline_wander(clean, snr_db, rng, fs):
    """Respiration-driven low-frequency drift.  Reference is a respiration-band
    signal; the drift is a scaled/phase-shifted version plus a small irreducible
    random-walk component."""
    n = len(clean)
    t = np.arange(n) / fs
    f_resp = rng.uniform(0.15, 0.4)  # ~9-24 breaths/min
    ref = (np.sin(2 * np.pi * f_resp * t + rng.uniform(0, 2 * np.pi))
           + 0.4 * np.sin(2 * np.pi * 2 * f_resp * t + rng.uniform(0, 2 * np.pi)))

    h = _random_path(rng, 4)
    drift = _apply_path(ref, h)
    walk = np.cumsum(rng.standard_normal(n)) / np.sqrt(n)  # small irreducible part
    interf = drift + 0.06 * walk
    return _scale_to_snr(interf, clean, snr_db), _unit_power(ref)


def echo(clean, snr_db, rng, fs, path_taps: int = 8):
    """Echo/crosstalk: a broadband reference source coupled through a short
    unknown FIR path (classic echo cancellation).  Fully cancellable when
    path_taps <= filter order."""
    n = len(clean)
    # coloured broadband reference (independent of clean)
    ref = rng.standard_normal(n)
    ref = np.convolve(ref, np.ones(3) / 3.0, mode="same")  # mild colouring
    ref = _unit_power(ref)
    h = _random_path(rng, path_taps)
    interf = _apply_path(ref, h)
    return _scale_to_snr(interf, clean, snr_db), ref


def regime_switch(clean, snr_db, rng, fs):
    """Non-stationary: the interference propagation path changes abruptly at one
    or two points mid-record (e.g. lead movement).  Reference is continuous; the
    controller must re-adapt quickly -- the case where a learned step-size wins."""
    n = len(clean)
    ref = rng.standard_normal(n)
    ref = np.convolve(ref, np.ones(3) / 3.0, mode="same")
    ref = _unit_power(ref)

    n_seg = int(rng.integers(2, 4))
    bounds = np.sort(rng.choice(np.arange(n // 6, n - n // 6),
                                size=n_seg - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [n]]).astype(int)
    interf = np.zeros(n)
    for i in range(n_seg):
        a, b = bounds[i], bounds[i + 1]
        h = _random_path(rng, int(rng.integers(4, 9)))
        seg = _apply_path(ref[a:b], h)
        interf[a:b] = seg
    return _scale_to_snr(interf, clean, snr_db), ref


def narrowband_chirp(clean, snr_db, rng, fs):
    """Held-out: a swept-frequency narrowband interferer (e.g. an RF/EMI tone
    drifting in frequency).  Reference is the tonal source; the path adds an
    unknown gain/phase."""
    n = len(clean)
    t = np.arange(n) / fs
    f0 = rng.uniform(20.0, 60.0)
    f1 = rng.uniform(80.0, 140.0)
    k = (f1 - f0) / (n / fs)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    ref = np.sin(phase + rng.uniform(0, 2 * np.pi))
    interf = rng.uniform(0.5, 1.0) * np.sin(phase + rng.uniform(0, 2 * np.pi))
    return _scale_to_snr(interf, clean, snr_db), _unit_power(ref)


def echo_long(clean, snr_db, rng, fs):
    """Held-out: echo through a longer path than seen in training (stresses the
    fixed filter order)."""
    return echo(clean, snr_db, rng, fs, path_taps=int(rng.integers(12, 20)))


# ---------------- dispatch ----------------

TRAIN_FAMILIES = ("powerline", "baseline_wander", "echo", "regime_switch")
OOD_FAMILIES = ("narrowband_chirp", "echo_long")
ALL_FAMILIES = TRAIN_FAMILIES + OOD_FAMILIES

_DISPATCH = {
    "powerline": powerline,
    "baseline_wander": baseline_wander,
    "echo": echo,
    "regime_switch": regime_switch,
    "narrowband_chirp": narrowband_chirp,
    "echo_long": echo_long,
}


def make_interference(family: str, clean: np.ndarray, rng: np.random.Generator,
                      snr_db: float = 10.0, fs: float = 360.0):
    """Return ``(interference, reference)`` for the given family.

    ``primary = clean + interference``; the filter cancels ``interference`` using
    ``reference`` alone.  Both outputs have ``len(clean)`` samples.
    """
    if family not in _DISPATCH:
        raise ValueError(f"Unknown interference family: {family!r}. "
                         f"Known: {list(_DISPATCH)}")
    interf, ref = _DISPATCH[family](clean, snr_db, rng, fs)
    return np.asarray(interf, dtype=float), np.asarray(ref, dtype=float)

"""Reference-based interference model for adaptive noise cancellation (ANC).

Unlike ``src.noise.families`` (which returns a single additive-noise vector for a
supervised noisy->clean setup), every generator here returns a *pair*::

    interference, reference = make_interference(family, clean, rng, snr_db, fs)

    primary[n]   = clean[n] + interference[n]     # observable sensor
    reference[n] = reference                       # observable, correlated w/ interference,
                                                   #   INDEPENDENT of clean

The adaptive filter estimates the interference from ``reference`` and subtracts it:
``e = primary - w^T ref_window``.  Because both ``primary`` and ``reference`` are
observable, ``e`` (the denoised output *and* the adaptation error) needs no clean
signal at run time.  The clean signal is used only offline to score MSE/SNR.

This is the classical Widrow adaptive-noise-cancelling / echo-cancellation
configuration, restricted to *structured* interference that admits a reference
(powerline, baseline wander, narrowband, echo).  White/impulsive/alpha-stable
noise is intentionally absent: there is no reference to cancel it, so claiming to
do so would be circular.
"""
from .families import (
    make_interference,
    TRAIN_FAMILIES,
    OOD_FAMILIES,
    ALL_FAMILIES,
)

__all__ = ["make_interference", "TRAIN_FAMILIES", "OOD_FAMILIES", "ALL_FAMILIES"]

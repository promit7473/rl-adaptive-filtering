"""IIR notch filter — strong specialised baseline for narrow-band interference.

Standard 2nd-order IIR notch at frequency f0 with quality factor Q.
Achieves -40 to -60 dB on a pure tone (e.g. 50 Hz powerline) with zero
training. Included as a fairness baseline: when the interfering frequency
is known a priori, a notch is the right tool. Meta-RL is general-purpose
and does not need to know the frequency in advance.
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sig


class IIRNotch:
    """Direct-form II 2nd-order IIR notch filter."""

    def __init__(self, f0: float = 50.0, fs: float = 360.0, Q: float = 30.0):
        self.f0 = float(f0)
        self.fs = float(fs)
        self.Q = float(Q)
        self.b, self.a = sig.iirnotch(self.f0 / (self.fs / 2.0), self.Q)
        self.zi = sig.lfilter_zi(self.b, self.a) * 0.0

    def reset(self) -> None:
        self.zi = sig.lfilter_zi(self.b, self.a) * 0.0

    def run(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = sig.lfilter(self.b, self.a, x, zi=self.zi)
        return np.asarray(y)

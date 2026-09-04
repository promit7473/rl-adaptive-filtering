"""Smoke tests for signal + noise generators."""
import numpy as np
from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES, signal_power


def test_signals_shapes():
    rng = np.random.default_rng(0)
    for kind in ["sine", "multitone", "am", "chirp"]:
        x = make_signal(kind, n=2000, fs=8000.0, rng=rng)
        assert x.shape == (2000,)
        assert np.isfinite(x).all()


def test_noise_families_snr():
    rng = np.random.default_rng(0)
    clean = make_signal("multitone", n=4000, fs=8000.0, rng=rng)
    for fam in TRAIN_FAMILIES + OOD_FAMILIES:
        kwargs = {"fs": 8000.0} if fam == "chirp_interferer" else {}
        noise = make_noise(fam, clean, rng, snr_db=10.0, **kwargs)
        assert noise.shape == clean.shape
        assert np.isfinite(noise).all(), fam
        # rough SNR sanity (some families like regime_switch / time_varying don't honor a single snr)
        if fam in ("gaussian", "colored", "impulsive", "alpha_stable", "burst", "chirp_interferer"):
            snr = 10 * np.log10(signal_power(clean) / signal_power(noise))
            assert abs(snr - 10.0) < 2.0, (fam, snr)


def test_chirp_interferer_below_nyquist_at_fs360():
    rng = np.random.default_rng(0)
    fs = 360.0
    n = 4096
    clean = make_signal("multitone", n=n, fs=fs, rng=rng)
    noise = make_noise("chirp_interferer", clean, rng, snr_db=10.0, fs=fs)
    spec = np.abs(np.fft.rfft(noise))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    peak_hz = float(freqs[int(np.argmax(spec))])
    assert peak_hz < fs / 2.0
    nyquist = fs / 2.0
    band = (freqs >= 0.05 * nyquist) & (freqs <= 0.45 * nyquist)
    assert spec[band].sum() > spec[~band].sum()


if __name__ == "__main__":
    test_signals_shapes()
    test_noise_families_snr()
    test_chirp_interferer_below_nyquist_at_fs360()
    print("OK")

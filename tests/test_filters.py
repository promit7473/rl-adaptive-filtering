"""Smoke + sanity tests for classical adaptive filters."""
import numpy as np
from src.signals.generators import make_signal
from src.noise.families import make_noise
from src.filters import make_filter, windowize


def build_problem(n=4000, fs=8000.0, snr_db=10.0, seed=0):
    rng = np.random.default_rng(seed)
    clean = make_signal("multitone", n=n, fs=fs, rng=rng)
    noise = make_noise("gaussian", clean, rng, snr_db=snr_db)
    noisy = clean + noise
    U = windowize(noisy, order=16)
    d = clean
    return U, d, clean, noisy


def steady_state_mse(e, frac=0.25):
    tail = int(len(e) * frac)
    return float(np.mean(e[-tail:] ** 2))


def test_all_filters_converge():
    U, d, clean, noisy = build_problem()
    raw_mse = float(np.mean((noisy - clean) ** 2))
    filter_configs = [
        ("lms", dict(mu=0.01)),
        ("nlms", dict(mu=0.5)),
        ("vss_lms", dict(mu_max=0.05)),
        ("aboulnasr_mayyas", dict(mu_max=0.1)),
        ("mathews_xie", dict(mu_max=0.1)),
        ("rls", dict(forgetting=0.995)),
        ("heuristic", dict(mu_base=0.01)),
        ("kalman_mu", dict(mu_init=0.01)),
    ]
    for name, kw in filter_configs:
        f = make_filter(name, order=16, **kw)
        y, e = f.run(U, d)
        ss = steady_state_mse(e)
        assert np.isfinite(ss), f"{name} produced non-finite MSE"
        assert ss < raw_mse * 1.5, f"{name} diverged: ss={ss:.4f} raw={raw_mse:.4f}"
        print(f"{name:20s}  steady-state MSE = {ss:.5f}  (raw {raw_mse:.5f})")

    # LMP is designed for impulsive noise, not Gaussian; test it separately
    rng = np.random.default_rng(0)
    clean_imp = make_signal("multitone", n=4000, fs=8000.0, rng=rng)
    noise_imp = make_noise("impulsive", clean_imp, rng, snr_db=10.0)
    noisy_imp = clean_imp + noise_imp
    U_imp = windowize(noisy_imp, order=16)
    f_lmp = make_filter("lmp", order=16, mu=0.001, p=1.5)
    y_lmp, e_lmp = f_lmp.run(U_imp, clean_imp)
    ss_lmp = steady_state_mse(e_lmp)
    assert np.isfinite(ss_lmp), f"LMP produced non-finite MSE"
    print(f"{'lmp':20s}  steady-state MSE = {ss_lmp:.5f}  (on impulsive noise)")


if __name__ == "__main__":
    test_all_filters_converge()
    print("All filter tests passed.")

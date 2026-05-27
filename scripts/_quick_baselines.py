#!/usr/bin/env python3
"""Quick classical baseline comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, time
from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from src.filters import NLMS, RLS, VSSLMS, PIDLeakyNLMS, FixedLeakageNLMS, windowize
from src.eval.metrics import steady_state_mse

FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
SNRS = [5, 10, 15]
N = 2000
FS = 360.0
ORDER = 16
SEEDS = [0, 1, 2]
SIGNALS = ["ecg_like", "random_pulses", "multitone", "square_burst"]

methods = [
    ("NLMS-mu0.1", lambda: NLMS(order=ORDER, mu=0.1)),
    ("NLMS-mu0.5", lambda: NLMS(order=ORDER, mu=0.5)),
    ("NLMS-mu1.0", lambda: NLMS(order=ORDER, mu=1.0)),
    ("RLS-099", lambda: RLS(order=ORDER, forgetting=0.99)),
    ("RLS-0995", lambda: RLS(order=ORDER, forgetting=0.995)),
    ("RLS-0999", lambda: RLS(order=ORDER, forgetting=0.999)),
    ("VSS-Kwong", lambda: VSSLMS(order=ORDER, mu_max=0.05, alpha=0.97, gamma=1e-3)),
    ("PID-NLMS", lambda: PIDLeakyNLMS(order=ORDER)),
    ("Fixed-Leaky", lambda: FixedLeakageNLMS(order=ORDER, mu=0.5, leakage=0.99)),
]

results = {}
for fam in FAMILIES:
    for snr in SNRS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + hash(fam) % 999)
            sig_kind = rng.choice(SIGNALS)
            clean = make_signal(sig_kind, N, fs=FS, rng=rng)
            noisy = clean + make_noise(fam, clean, rng, snr_db=snr, fs=FS)
            for name, factory in methods:
                filt = factory()
                U = windowize(noisy, ORDER)
                _, e = filt.run(U, clean)
                ss = steady_state_mse(e)
                ss_db = 10 * np.log10(ss + 1e-12)
                key = (fam, snr)
                if key not in results:
                    results[key] = {}
                if name not in results[key]:
                    results[key][name] = []
                results[key][name].append(ss_db)

hdr = "{:<18} {:>3} ".format("Family", "SNR")
display_methods = ["NLMS-mu0.5", "RLS-0995", "RLS-0999", "PID-NLMS", "VSS-Kwong", "Fixed-Leaky"]
for b in display_methods:
    hdr += "{:>14}".format(b)
print(hdr)
print("-" * len(hdr))

for key in sorted(results.keys()):
    fam, snr = key
    row = "{:<18} {:>3} ".format(fam, snr)
    for b in display_methods:
        vals = results[key].get(b, [])
        if vals:
            row += "{:>+14.2f}".format(np.mean(vals))
        else:
            row += "{:>14}".format("N/A")
    print(row)

print("\n=== AVERAGES ACROSS ALL CONDITIONS ===")
averages = {}
for b in [m[0] for m in methods]:
    all_vals = []
    for key in results:
        all_vals.extend(results[key].get(b, []))
    if all_vals:
        averages[b] = np.mean(all_vals)
        print("  {:<20} ss_mse_db = {:+.2f}".format(b, averages[b]))

"""Train the Meta-AF style baseline (BPTT through leaky-NLMS)."""
from __future__ import annotations
import argparse
import os
import numpy as np

from src.filters.meta_af import train_meta_af, MetaAFFilter, _MetaAFController
from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/v2_meta_af/meta_af.pt")
    p.add_argument("--n-iters", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--episode-len", type=int, default=2000)
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    families = list(TRAIN_FAMILIES)
    # Same signal mix and curriculum weights as Meta-RL training, so the
    # comparison isolates "RL vs BPTT" rather than "more diverse training data".
    signals = ["multitone", "am", "sine",
               "ecg_like", "random_pulses", "square_burst"]
    sig_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    sig_weights = sig_weights / sig_weights.sum()
    fam_weights = {"gaussian": 1.0, "colored": 1.0, "impulsive": 2.0,
                   "time_varying": 2.0, "regime_switch": 3.0}
    fw = np.array([fam_weights.get(f, 1.0) for f in families])
    fw = fw / fw.sum()
    snrs = [0.0, 5.0, 10.0, 15.0, 20.0]

    def env_factory(rng: np.random.Generator):
        sig_kind = str(rng.choice(signals, p=sig_weights))
        family = str(rng.choice(families, p=fw))
        snr = float(rng.choice(snrs))
        if sig_kind == "multitone":
            base = rng.uniform(150.0, 400.0)
            clean = make_signal("multitone", n=args.episode_len, fs=args.fs, rng=rng,
                                freqs=[base, base * 1.7, base * 3.1],
                                amps=[1.0, 0.6, 0.3])
        elif sig_kind == "am":
            clean = make_signal("am", n=args.episode_len, fs=args.fs, rng=rng,
                                fc=rng.uniform(800.0, 1500.0),
                                fm=rng.uniform(40.0, 120.0), mod_index=0.5)
        elif sig_kind in ("ecg_like", "random_pulses", "square_burst"):
            clean = make_signal(sig_kind, n=args.episode_len, fs=args.fs, rng=rng)
        else:
            clean = make_signal("sine", n=args.episode_len, fs=args.fs, rng=rng,
                                freq=rng.uniform(150.0, 600.0))
        noise = make_noise(family, clean, rng, snr_db=snr, fs=args.fs)
        # Match Meta-RL env's per-episode normalization for fair comparison.
        s = float(np.std(clean + noise)) + 1e-9
        return clean / s, (clean + noise) / s

    net = train_meta_af(env_factory, n_iters=args.n_iters,
                        batch_size=args.batch_size,
                        episode_len=args.episode_len,
                        device=args.device, verbose=True)
    torch.save({"state_dict": net.state_dict()}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

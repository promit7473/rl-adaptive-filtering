"""Train the Meta-AF style baseline (BPTT through leaky-NLMS)."""
from __future__ import annotations
import argparse
import os
import numpy as np

from src.filters.meta_af import train_meta_af
from src.noise.families import TRAIN_FAMILIES
from src.kernel import (
    sample_episode, SIGNAL_KINDS, SIGNAL_WEIGHTS, CURRICULUM_WEIGHTS, SNR_OPTIONS,
)
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/meta_af/meta_af.pt")
    p.add_argument("--n-iters", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--episode-len", type=int, default=2000)
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    families = list(TRAIN_FAMILIES)
    fam_weights = np.array([CURRICULUM_WEIGHTS.get(f, 1.0) for f in families], dtype=float)

    def env_factory(rng: np.random.Generator):
        clean, noisy, _, _, _, _ = sample_episode(
            rng, args.episode_len, args.fs,
            train_families=families,
            signal_kinds=SIGNAL_KINDS,
            signal_weights=SIGNAL_WEIGHTS,
            snr_options=SNR_OPTIONS,
            family_weights=fam_weights,
        )
        return clean, noisy

    net = train_meta_af(env_factory, n_iters=args.n_iters,
                        batch_size=args.batch_size,
                        episode_len=args.episode_len,
                        device=args.device, verbose=True)
    torch.save({"state_dict": net.state_dict()}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

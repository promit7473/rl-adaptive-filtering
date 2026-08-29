#!/usr/bin/env python3
"""Fig. 4 (zero-shot ECG): paired residual improvement over NLMS on real MIT-BIH.

The trained controller is applied without retraining to real ECG corrupted by
powerline (50 Hz mains reference) and baseline wander (respiration reference).
Bars show mean paired improvement over NLMS +/- 95% CI over (record, seed) pairs.
Requires a trained checkpoint and internet (wfdb streams MIT-BIH from PhysioNet).

Usage:
    PYTHONPATH=. python scripts/fig_realworld_anc.py \
        --model results/runs/anc_hybrid_seed42/controller_final.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.interference.families import make_interference
from src.envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig
from src.filters.anc_baselines import NLMSANC
from src.agents.controller import HybridController, LSTMController

FS, M = 360.0, 16


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = ck.get("config", None)
    g = lambda k, d: getattr(tcfg, k, d) if tcfg else d
    ctor = HybridController if g("controller_type", "hybrid") == "hybrid" else LSTMController
    ctrl = ctor(feat_dim=11, hidden=g("lstm_hidden", 256),
                n_lstm_layers=g("n_lstm_layers", 2), act_dim=2, n_families=g("n_families", 6))
    ctrl.load_state_dict(ck["state_dict"]); ctrl.eval()
    env_cfg = ANCEnvConfig(fs=FS, filter_order=M, q_min=g("q_min", 1e-8),
                           q_max=g("q_max", 1e-3), r_min=g("r_min", 1e-3), r_max=g("r_max", 1e1))
    return ctrl, env_cfg


def _ss_db(res, frac=0.25):
    tail = max(1, int(len(res) * frac))
    return 10 * np.log10(np.mean(np.asarray(res)[-tail:] ** 2) + 1e-12)


def _run_ctrl(ctrl, env_cfg, clean, interf, ref):
    import dataclasses
    cfg = ANCEnvConfig(**{**dataclasses.asdict(env_cfg), "episode_len": len(clean)})
    env = ANCKalmanEnv(cfg, fixed_family="ecg", fixed_signal="ecg", fixed_snr_db=10.0, seed=0)
    env.set_preset_episode(clean, interf, ref)
    obs, _ = env.reset(seed=0); state = None; done = False
    while not done:
        ot = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            a, state, *_ = ctrl(ot, state)
        obs, _, term, trunc, _ = env.step(a[0, 0].numpy()); done = term or trunc
    return np.asarray(env.residuals) * env.norm_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="paper/figures/fig_realworld.pdf")
    ap.add_argument("--records", nargs="+",
                    default=["100", "101", "103", "105", "115"])
    ap.add_argument("--noises", nargs="+", default=["powerline", "baseline_wander"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--len", type=int, default=10800)
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"model not found: {args.model}")
    try:
        import wfdb
    except ImportError:
        sys.exit("wfdb not installed (needed to stream MIT-BIH)")

    ctrl, env_cfg = _load(args.model)
    n = args.len
    improvements = {nz: [] for nz in args.noises}

    for rec in args.records:
        try:
            sig, _ = wfdb.rdsamp(rec, channels=[0], pn_dir="mitdb", sampto=n + 500)
        except Exception as e:
            print(f"[skip] {rec}: {e}"); continue
        clean = sig[500:500 + n, 0].astype(float)
        clean = (clean - clean.mean()) / (clean.std() + 1e-9)
        for nz in args.noises:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                interf, ref = make_interference(nz, clean, rng, snr_db=10.0, fs=FS)
                primary = clean + interf
                res_ctrl = _run_ctrl(ctrl, env_cfg, clean, interf, ref)
                res_nlms = NLMSANC(order=M, mu=0.05).run(ref, primary) - clean
                improvements[nz].append(_ss_db(res_nlms) - _ss_db(res_ctrl))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    labels, means, cis = [], [], []
    for nz in args.noises:
        v = np.asarray(improvements[nz])
        if len(v) == 0:
            continue
        labels.append(nz.replace("_", "\n"))
        means.append(v.mean())
        cis.append(1.96 * v.std() / np.sqrt(len(v)))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color="tab:red", alpha=0.8, capsize=3)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("residual improvement\nover NLMS (dB)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  (means: {dict(zip(labels, np.round(means,2)))})")


if __name__ == "__main__":
    main()

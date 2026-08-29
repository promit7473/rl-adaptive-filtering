#!/usr/bin/env python3
"""Fig. 3 (recovery): within-episode adaptation after an interference-path jump.

Shows the learned process noise Q_t spiking at a regime change and decaying as
the innovation re-whitens, with the residual recovering faster than fixed-Q
Kalman / RLS. Requires a trained hybrid checkpoint.

Usage:
    PYTHONPATH=. python scripts/fig_recovery_anc.py \
        --model results/runs/anc_hybrid_seed42/controller_final.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.signals.generators import make_signal
from src.interference.families import make_interference
from src.envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig
from src.filters.anc_baselines import KalmanANC, RLSANC
from src.agents.controller import HybridController, LSTMController

FS, N, M = 360.0, 3000, 16


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = ck.get("config", None)
    g = lambda k, d: getattr(tcfg, k, d) if tcfg else d
    ctor = HybridController if g("controller_type", "hybrid") == "hybrid" else LSTMController
    ctrl = ctor(feat_dim=11, hidden=g("lstm_hidden", 256),
                n_lstm_layers=g("n_lstm_layers", 2), act_dim=2, n_families=g("n_families", 6))
    ctrl.load_state_dict(ck["state_dict"]); ctrl.eval()
    env_cfg = ANCEnvConfig(fs=FS, episode_len=N, filter_order=M,
                           q_min=g("q_min", 1e-8), q_max=g("q_max", 1e-3),
                           r_min=g("r_min", 1e-3), r_max=g("r_max", 1e1))
    return ctrl, env_cfg


def _run_ctrl(ctrl, env_cfg, clean, interf, ref):
    env = ANCKalmanEnv(env_cfg, fixed_family="regime_switch",
                       fixed_signal="ecg_like", fixed_snr_db=5.0, seed=0)
    env.set_preset_episode(clean, interf, ref)
    obs, _ = env.reset(seed=0); state = None; done = False
    while not done:
        ot = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            a, state, *_ = ctrl(ot, state)
        obs, _, term, trunc, _ = env.step(a[0, 0].numpy()); done = term or trunc
    return (np.asarray(env.residuals) * env.norm_scale, np.asarray(env.q_hist),
            env.norm_scale)


def _db(res, win=100):
    r2 = res ** 2
    ma = np.convolve(r2, np.ones(win) / win, mode="same")
    return 10 * np.log10(ma + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="paper/figures/fig_recovery.pdf")
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        sys.exit(f"model not found: {args.model}")

    rng = np.random.default_rng(3)
    clean = make_signal("ecg_like", n=N, fs=FS, rng=rng)
    interf, ref = make_interference("regime_switch", clean, rng, snr_db=5.0, fs=FS)
    primary = clean + interf
    ctrl, env_cfg = _load(args.model)
    res_c, q_c, scale = _run_ctrl(ctrl, env_cfg, clean, interf, ref)
    res_k = (KalmanANC(order=M, q=1e-6).run(ref, primary) - clean)
    res_r = (RLSANC(order=M, forgetting=0.999).run(ref, primary) - clean)
    t = np.arange(N) / FS

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(t, _db(res_r), color="0.6", lw=1.0, label="RLS(0.999)")
    ax[0].plot(t, _db(res_k), color="tab:green", lw=1.0, label="Kalman $Q{=}10^{-6}$")
    ax[0].plot(t, _db(res_c), color="tab:red", lw=1.4, label="Hybrid (ours)")
    ax[0].set_ylabel("residual (dB)"); ax[0].legend(fontsize=6, loc="upper right")
    ax[1].semilogy(t, q_c, color="tab:red", lw=1.0)
    ax[1].set_ylabel("$Q_t$"); ax[1].set_xlabel("time (s)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

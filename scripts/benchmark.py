"""V3 evaluation: classical baselines + Meta-AF + v3 hybrid + v3 RL on synthetic & ECG.

Evaluates all methods on a comprehensive grid including OOD families,
multiple SNRs, and MIT-BIH ECG records. Outputs per-method comparison
tables with Wilcoxon tests.

Usage (method names must match what fig_realworld.py / ecg_stats.py expect):
    PYTHONPATH=. python3 scripts/benchmark.py \
        --out-dir results/runs/eval \
        --hybrid-models "Hybrid (ours)=results/runs/hybrid_seed42/controller_final.pt" \
        --rl-models "RL-only=results/runs/rl_seed42/rl_seed42_final.zip" \
        --meta-af-path results/meta_af/meta_af.pt

Note: sb3 checkpoints carry no env config, so --rl-models assumes the policy
was trained under train_pipeline.RL_ENV_KW (imported below). Legacy
checkpoints trained with other action bounds cannot be evaluated correctly.
"""
from __future__ import annotations
import argparse
import csv
import os
import time
import zlib
import numpy as np
import torch

from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from src.filters import (
    NLMS, RLS, VSSLMS, HeuristicMuScheduler,
    PIDLeakyNLMS, FixedLeakageNLMS, IIRNotch, MetaAFFilter, windowize,
)
from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
from src.eval.metrics import steady_state_mse, convergence_time
from src.agents.controller import LSTMController, HybridController, TransformerController
from scripts.train_pipeline import RL_ENV_KW, _episode_rng

ALL_FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)


def _row(method, family, snr, seed, e, dt_ms, **extra):
    # Same schema as train_pipeline.run_eval._row (both write synthetic.csv and
    # make_table1.py reads whichever one produced it): NaN-safe, incl. diverged.
    e = np.asarray(e, dtype=np.float64)
    finite = np.isfinite(e)
    n_nonfinite = int(np.sum(~finite))
    e_safe = np.where(finite, e, 1e6)
    ss = steady_state_mse(e_safe)
    ss_db = float(10 * np.log10(ss + 1e-12))
    return dict(method=method, family=family, snr_db=snr, seed=seed,
                ss_mse=float(ss), ss_mse_db=ss_db,
                ep_mse=float(np.mean(e_safe ** 2)),
                conv_time=float(convergence_time(e_safe)),
                inference_time_ms=float(dt_ms),
                diverged=int(n_nonfinite > 0 or ss_db > 10.0), **extra)


def _classical_run(filt, noisy, clean, order, single_input=False):
    t0 = time.perf_counter()
    is_meta_af = (filt.__class__.__name__ == "MetaAFFilter")
    if is_meta_af:
        s = float(np.std(noisy)) + 1e-9
        noisy_normalized = noisy / s
        clean_normalized = clean / s
    else:
        noisy_normalized = noisy
        clean_normalized = clean

    if single_input:
        y = filt.run(noisy_normalized)
        e = clean_normalized - y
    else:
        U = windowize(noisy_normalized, order)
        _, e = filt.run(U, clean_normalized)

    if is_meta_af:
        e = e * s

    return e, (time.perf_counter() - t0) * 1000


def _build_classical_methods(args, fs):
    methods = {
        "NLMS":            (lambda: NLMS(order=args.order, mu=0.5), False),
        "RLS":             (lambda: RLS(order=args.order, forgetting=0.995), False),
        "VSS-Kwong":       (lambda: VSSLMS(order=args.order, mu_max=0.05,
                                            alpha=0.97, gamma=1e-3), False),
        "Heuristic":       (lambda: HeuristicMuScheduler(order=args.order,
                                                          mu_base=0.01), False),
        "PID-NLMS":        (lambda: PIDLeakyNLMS(order=args.order), False),
        "Fixed-Leaky":     (lambda: FixedLeakageNLMS(order=args.order,
                                                       mu=0.5, leakage=0.99), False),
        "IIR-Notch (50Hz)":(lambda: IIRNotch(f0=50.0, fs=fs, Q=30.0), True),
    }
    if args.meta_af_path and os.path.exists(args.meta_af_path):
        methods["Meta-AF (BPTT)"] = (
            lambda: MetaAFFilter.load(args.meta_af_path, device="cpu",
                                       order=args.order), False)
    return methods


def eval_synthetic(args) -> list[dict]:
    methods = _build_classical_methods(args, args.fs)
    rows = []
    for sig_kind in args.signals:
        for fam in args.families:
            for snr in args.snrs:
                for seed in args.seeds:
                    # crc32-based rng shared with train_pipeline: stable
                    # across processes AND identical episodes across scripts.
                    rng = _episode_rng(sig_kind, fam, snr, seed)
                    clean = make_signal(sig_kind, n=args.n, fs=args.fs, rng=rng)
                    noisy = clean + make_noise(fam, clean, rng, snr_db=snr, fs=args.fs)
                    for name, (factory, single) in methods.items():
                        filt = factory()
                        e, dt = _classical_run(filt, noisy, clean, args.order,
                                                single_input=single)
                        rows.append(_row(name, fam, snr, seed, e, dt,
                                         signal=sig_kind))
    return rows


def _load_controller(path):
    """Load a hybrid/BPTT controller + its training config from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = ckpt.get("config", None)
    g = lambda k, d: getattr(tcfg, k, d) if tcfg is not None else d
    ctrl_type = g('controller_type', 'hybrid')
    lstm_hidden = g('lstm_hidden', 256)
    n_lstm_layers = g('n_lstm_layers', 2)
    if ctrl_type == "hybrid":
        ctrl = HybridController(feat_dim=11, hidden=lstm_hidden,
                                n_lstm_layers=n_lstm_layers, act_dim=2, n_families=8)
    elif ctrl_type == "transformer":
        ctrl = TransformerController(feat_dim=11, d_model=lstm_hidden,
                                     n_heads=4, n_layers=4, act_dim=2)
    else:
        ctrl = LSTMController(feat_dim=11, hidden=lstm_hidden,
                              n_lstm_layers=n_lstm_layers, act_dim=2)
    ctrl.load_state_dict(ckpt["state_dict"])
    ctrl.eval()
    env_kw = dict(mu_min=g('mu_min', 0.005), mu_max=g('mu_max', 2.0),
                  leakage_min=g('lam_min', 0.80), leakage_max=g('lam_max', 0.999),
                  mu_base_schedule=g('use_mu_schedule', True),
                  state_window=1)
    return ctrl, env_kw


# Single source of truth: the RL eval env is the RL training env
# (train_pipeline.RL_ENV_KW), minus the fields each episode overrides.
RL_EVAL_ENV_KW = {k: v for k, v in RL_ENV_KW.items()
                  if k not in ("fs", "episode_len", "filter_order")}


def _run_controller_episode(ctrl, env_kw, clean, noisy, fs, n, order):
    """Run a hybrid/BPTT controller on a preset episode; raw-unit errors."""
    env_cfg = EnvConfigV2(fs=fs, episode_len=n, filter_order=order, **env_kw)
    env = AdaptiveFilterEnvV2(env_cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    obs, _ = env.reset(seed=0)
    state = None
    errs = []
    t0 = time.perf_counter()
    done = False
    while not done:
        obs_t = torch.tensor(obs[-11:], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action, state, *_ = ctrl(obs_t, state)
        obs, _, term, trunc, _ = env.step(action[0, 0].cpu().numpy())
        errs.append(env.last_e)
        done = term or trunc
    dt = (time.perf_counter() - t0) * 1000
    return np.array(errs) * env.norm_scale, dt, env.divergence_count


def _run_rl_episode(model, is_rec, vec_norm, clean, noisy, fs, n, order):
    """Run an sb3 RL policy on a preset episode; raw-unit errors."""
    env_cfg = EnvConfigV2(fs=fs, episode_len=n, filter_order=order, **RL_EVAL_ENV_KW)
    env = AdaptiveFilterEnvV2(env_cfg, seed=0)
    env.set_preset_episode(clean, noisy)
    obs, _ = env.reset(seed=0)
    if vec_norm is not None:
        obs = vec_norm.normalize_obs(obs)
    lstm_state = None
    starts = np.ones((1,), dtype=bool) if is_rec else None
    errs = []
    t0 = time.perf_counter()
    done = False
    while not done:
        if is_rec:
            a, lstm_state = model.predict(obs[None, :], state=lstm_state,
                                          episode_start=starts, deterministic=True)
            starts = np.zeros((1,), dtype=bool)
            a_use = a[0]
        else:
            a_use, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(a_use)
        if vec_norm is not None:
            obs = vec_norm.normalize_obs(obs)
        errs.append(env.last_e)
        done = term or trunc
    dt = (time.perf_counter() - t0) * 1000
    return np.array(errs) * env.norm_scale, dt, env.divergence_count


def _load_rl(path):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    try:
        model = PPO.load(path, device="cpu"); is_rec = False
    except Exception:
        model = RecurrentPPO.load(path, device="cpu"); is_rec = True
    vec_path = path.replace("_final.zip", "_vecnormalize.pkl")
    vec_norm = None
    if os.path.exists(vec_path):
        try:
            # Dummy env must match the training obs space (RL_ENV_KW,
            # state_window=1 → 11-dim); a default EnvConfigV2 is 44-dim and
            # VecNormalize.load rejects it.
            dummy_env = DummyVecEnv(
                [lambda: AdaptiveFilterEnvV2(EnvConfigV2(**RL_ENV_KW))])
            vec_norm = VecNormalize.load(vec_path, dummy_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
        except Exception as ex:
            print(f"WARNING: failed to load VecNormalize stats from "
                  f"{vec_path} ({ex}); evaluating on RAW observations — "
                  f"RL results will not reflect the trained policy")
            vec_norm = None
    return model, is_rec, vec_norm


def eval_hybrid_models(args, model_paths: dict[str, str]) -> list[dict]:
    rows = []
    for name, path in model_paths.items():
        ctrl, env_kw = _load_controller(path)
        for sig_kind in args.signals:
            for fam in args.families:
                for snr in args.snrs:
                    for seed in args.seeds:
                        rng = _episode_rng(sig_kind, fam, snr, seed)
                        clean = make_signal(sig_kind, n=args.n, fs=args.fs, rng=rng)
                        noisy = clean + make_noise(fam, clean, rng,
                                                   snr_db=snr, fs=args.fs)
                        e, dt, _ = _run_controller_episode(
                            ctrl, env_kw, clean, noisy, args.fs, args.n, args.order)
                        rows.append(_row(name, fam, snr, seed, e, dt,
                                         signal=sig_kind))
    return rows


def eval_rl_v3(args, model_paths: dict[str, str]) -> list[dict]:
    rows = []
    for name, path in model_paths.items():
        model, is_rec, vec_norm = _load_rl(path)
        for sig_kind in args.signals:
            for fam in args.families:
                for snr in args.snrs:
                    for seed in args.seeds:
                        rng = _episode_rng(sig_kind, fam, snr, seed)
                        clean = make_signal(sig_kind, n=args.n, fs=args.fs, rng=rng)
                        noisy = clean + make_noise(fam, clean, rng,
                                                   snr_db=snr, fs=args.fs)
                        e, dt, _ = _run_rl_episode(
                            model, is_rec, vec_norm, clean, noisy,
                            args.fs, args.n, args.order)
                        rows.append(_row(name, fam, snr, seed, e, dt,
                                         signal=sig_kind))
    return rows


def eval_ecg_v3(args, hybrid_paths: dict, rl_paths: dict) -> list[dict]:
    try:
        import wfdb
    except ImportError:
        print("[ecg] wfdb not installed; skipping ECG eval")
        return []

    rows = []
    records = args.ecg_records
    fs = 360.0
    n = args.ecg_len

    classical = _build_classical_methods(args, fs)
    hybrid_loaded = {name: _load_controller(path)
                     for name, path in hybrid_paths.items()}
    rl_loaded = {name: _load_rl(path) for name, path in rl_paths.items()}

    def _scale(noise, clean, snr_db):
        ps = float(np.mean(clean ** 2))
        pn = float(np.mean(noise ** 2) + 1e-12)
        return noise * np.sqrt(ps / pn / 10 ** (snr_db / 10))

    def add_noise(clean, kind, snr_db, rng):
        t = np.arange(len(clean)) / fs
        if kind == "powerline":
            return _scale(np.sin(2 * np.pi * 50 * t), clean, snr_db)
        if kind == "baseline_wander":
            # respiratory-band drift: sum of slow sinusoids + random walk
            drift = (np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 2 * np.pi))
                     + 0.5 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi)))
            walk = np.cumsum(rng.standard_normal(len(clean))) / np.sqrt(len(clean))
            return _scale(drift + 0.3 * walk, clean, snr_db)
        if kind in ("gaussian", "impulsive", "burst", "regime_switch"):
            return make_noise(kind, clean, rng, snr_db=snr_db, fs=fs)
        return make_noise("gaussian", clean, rng, snr_db=snr_db, fs=fs)

    for rec_id in records:
        if "/" in rec_id:
            db, rname = rec_id.split("/", 1)
        else:
            db, rname = "mitdb", rec_id
        try:
            sig, _ = wfdb.rdsamp(rname, channels=[0], pn_dir=db, sampto=n + 500)
        except Exception as e:
            print(f"[ecg] failed to load {db}/{rname}: {e}")
            continue
        clean = sig[500:500 + n, 0].astype(float)
        clean = (clean - clean.mean()) / (clean.std() + 1e-9)

        for noise_kind in args.ecg_noises:
            for snr in args.ecg_snrs:
                for seed in args.seeds:
                    rng = np.random.default_rng(seed * 7919
                                                + zlib.crc32(noise_kind.encode()) % 1000)
                    noisy = clean + add_noise(clean, noise_kind, snr, rng)

                    def _ecg_row(mname, e, dt):
                        ss = steady_state_mse(np.asarray(e))
                        return dict(method=mname, record=rec_id,
                                    noise=noise_kind, snr_db=snr, seed=seed,
                                    ss_mse=float(ss),
                                    ss_mse_db=float(10 * np.log10(ss + 1e-12)),
                                    inference_time_ms=float(dt), db=db)

                    for mname, (factory, single) in classical.items():
                        filt = factory()
                        e, dt = _classical_run(filt, noisy, clean, args.order,
                                               single_input=single)
                        rows.append(_ecg_row(mname, e, dt))

                    for mname, (ctrl, env_kw) in hybrid_loaded.items():
                        e, dt, _ = _run_controller_episode(
                            ctrl, env_kw, clean, noisy, fs, n, args.order)
                        rows.append(_ecg_row(mname, e, dt))

                    for mname, (model, is_rec, vec_norm) in rl_loaded.items():
                        e, dt, _ = _run_rl_episode(
                            model, is_rec, vec_norm, clean, noisy,
                            fs, n, args.order)
                        rows.append(_ecg_row(mname, e, dt))
    return rows


def _save(rows, path):
    if not rows:
        print(f"[skip] no rows for {path}")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/benchmark")
    p.add_argument("--hybrid-models", nargs="*", default=[],
                   help="name=path.pt pairs for hybrid BPTT+RL models")
    p.add_argument("--rl-models", nargs="*", default=[],
                   help="name=path.zip pairs for RL models")
    p.add_argument("--families", nargs="+", default=ALL_FAMILIES)
    p.add_argument("--signals", nargs="+",
                   default=["multitone", "ecg_like",
                            "random_pulses", "square_burst"])
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--skip-ecg", action="store_true")
    p.add_argument("--ecg-records", nargs="+",
                   default=["mitdb/100", "mitdb/101", "mitdb/103",
                            "mitdb/105", "mitdb/115"])
    p.add_argument("--ecg-noises", nargs="+",
                   default=["powerline", "gaussian", "impulsive", "burst",
                            "baseline_wander", "regime_switch"])
    p.add_argument("--ecg-snrs", type=float, nargs="+", default=[10])
    p.add_argument("--ecg-len", type=int, default=10800)
    p.add_argument("--meta-af-path", default="results/meta_af/meta_af.pt")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    hybrid_paths = dict(s.split("=", 1) for s in args.hybrid_models) if args.hybrid_models else {}
    rl_paths = dict(s.split("=", 1) for s in args.rl_models) if args.rl_models else {}

    if not args.skip_synthetic:
        rows = eval_synthetic(args)
        if hybrid_paths:
            rows.extend(eval_hybrid_models(args, hybrid_paths))
        if rl_paths:
            rows.extend(eval_rl_v3(args, rl_paths))
        _save(rows, os.path.join(args.out_dir, "synthetic.csv"))

    if not args.skip_ecg:
        rows = eval_ecg_v3(args, hybrid_paths, rl_paths)
        _save(rows, os.path.join(args.out_dir, "ecg.csv"))


if __name__ == "__main__":
    main()

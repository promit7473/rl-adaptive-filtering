"""V3 evaluation: classical baselines + Meta-AF + v3 hybrid + v3 RL on synthetic & ECG.

Evaluates all methods on a comprehensive grid including OOD families,
multiple SNRs, and MIT-BIH ECG records. Outputs per-method comparison
tables with Wilcoxon tests.

Usage:
    PYTHONPATH=. python3 scripts/eval_v3.py \
        --out-dir results/benchmark \
        --hybrid-models Hybrid-BPTT-RL=results/v3_hybrid/seed42/controller_final.pt \
        --rl-models Meta-RL-v3=results/v3_rl/v3_rl_seed42_final.zip \
        --meta-af-path results/meta_af/meta_af.pt
"""
from __future__ import annotations
import argparse
import csv
import os
import time
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
from src.filters.diff_filter import DiffNLMSConfig, decode_action_bptt

ALL_FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)


def _row(method, family, snr, seed, e, dt_ms, **extra):
    e = np.asarray(e)
    ss = steady_state_mse(e)
    return dict(method=method, family=family, snr_db=snr, seed=seed,
                ss_mse=float(ss), ss_mse_db=float(10 * np.log10(ss + 1e-12)),
                ep_mse=float(np.mean(e ** 2)),
                conv_time=float(convergence_time(e)),
                inference_time_ms=float(dt_ms), **extra)


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
                    rng = np.random.default_rng(int(1e6) + seed * 1000
                                                + hash(fam + sig_kind) % 1000)
                    if sig_kind == "multitone":
                        clean = make_signal("multitone", n=args.n, fs=args.fs, rng=rng)
                    else:
                        clean = make_signal(sig_kind, n=args.n, fs=args.fs, rng=rng)
                    noisy = clean + make_noise(fam, clean, rng, snr_db=snr, fs=args.fs)
                    for name, (factory, single) in methods.items():
                        filt = factory()
                        e, dt = _classical_run(filt, noisy, clean, args.order,
                                                single_input=single)
                        rows.append(_row(name, fam, snr, seed, e, dt,
                                         signal=sig_kind))
    return rows


def eval_hybrid_models(args, model_paths: dict[str, str]) -> list[dict]:
    rows = []
    for name, path in model_paths.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", None)
        device = "cpu"

        if cfg_dict is not None and hasattr(cfg_dict, 'controller_type'):
            ctrl_type = cfg_dict.controller_type
            lstm_hidden = cfg_dict.lstm_hidden
            n_lstm_layers = cfg_dict.n_lstm_layers
        else:
            ctrl_type = "hybrid"
            lstm_hidden = 256
            n_lstm_layers = 2

        if ctrl_type == "hybrid":
            controller = HybridController(
                feat_dim=11, hidden=lstm_hidden,
                n_lstm_layers=n_lstm_layers, act_dim=2, n_families=8)
        elif ctrl_type == "transformer":
            controller = TransformerController(
                feat_dim=11, d_model=lstm_hidden, n_heads=4, n_layers=4, act_dim=2)
        else:
            controller = LSTMController(
                feat_dim=11, hidden=lstm_hidden,
                n_lstm_layers=n_lstm_layers, act_dim=2)

        controller.load_state_dict(ckpt["state_dict"])
        controller.eval()
        controller.to(device)

        for sig_kind in args.signals:
            for fam in args.families:
                for snr in args.snrs:
                    for seed in args.seeds:
                        env_cfg = EnvConfigV2(
                            fs=args.fs, episode_len=args.n,
                            filter_order=args.order,
                            mu_min=0.005, mu_max=2.0,
                            leakage_min=0.70, leakage_max=1.0,
                        )
                        env = AdaptiveFilterEnvV2(env_cfg, fixed_family=fam,
                                                    fixed_signal=sig_kind,
                                                    fixed_snr_db=snr, seed=seed)
                        obs, _ = env.reset(seed=seed)
                        state = None
                        errs = []
                        t0 = time.perf_counter()
                        done = False
                        steps = 0
                        while not done and steps < args.n:
                            obs_2d = obs.reshape(env_cfg.state_window, -1)[-1]
                            obs_t = torch.tensor(obs_2d, dtype=torch.float32,
                                                 device=device).unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():
                                action, state, *_ = controller(obs_t, state)
                            action_np = action[0, 0].cpu().numpy()
                            obs, _, term, trunc, _ = env.step(action_np)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        dt = (time.perf_counter() - t0) * 1000
                        rows.append(_row(name, fam, snr, seed, np.array(errs), dt,
                                         signal=sig_kind))
    return rows


def eval_rl_v3(args, model_paths: dict[str, str]) -> list[dict]:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

    rows = []
    for name, path in model_paths.items():
        try:
            model = PPO.load(path, device="cpu")
            is_rec = False
        except Exception:
            model = RecurrentPPO.load(path, device="cpu")
            is_rec = True

        vec_path = path.replace("_final.zip", "_vecnormalize.pkl")
        vec_norm = None
        if os.path.exists(vec_path):
            try:
                dummy_env = DummyVecEnv([lambda: AdaptiveFilterEnvV2(EnvConfigV2())])
                vec_norm = VecNormalize.load(vec_path, dummy_env)
                vec_norm.training = False
                vec_norm.norm_reward = False
            except Exception:
                vec_norm = None

        for sig_kind in args.signals:
            for fam in args.families:
                for snr in args.snrs:
                    for seed in args.seeds:
                        env_cfg = EnvConfigV2(
                            fs=args.fs, episode_len=args.n,
                            filter_order=args.order,
                        )
                        env = AdaptiveFilterEnvV2(env_cfg, fixed_family=fam,
                                                    fixed_signal=sig_kind,
                                                    fixed_snr_db=snr, seed=seed)
                        obs, _ = env.reset(seed=seed)
                        if vec_norm is not None:
                            obs = vec_norm.normalize_obs(obs)
                        lstm_state = None
                        episode_starts = np.ones((1,), dtype=bool) if is_rec else None
                        errs = []
                        t0 = time.perf_counter()
                        done = False
                        steps = 0
                        while not done and steps < args.n:
                            if is_rec:
                                action, lstm_state = model.predict(
                                    obs[None, :], state=lstm_state,
                                    episode_start=episode_starts,
                                    deterministic=True)
                                episode_starts = np.zeros((1,), dtype=bool)
                                a_use = action[0]
                            else:
                                a_use, _ = model.predict(obs, deterministic=True)
                            obs, _, term, trunc, _ = env.step(a_use)
                            if vec_norm is not None:
                                obs = vec_norm.normalize_obs(obs)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        dt = (time.perf_counter() - t0) * 1000
                        rows.append(_row(name, fam, snr, seed, np.array(errs), dt,
                                         signal=sig_kind))
    return rows


def eval_ecg_v3(args, hybrid_paths: dict, rl_paths: dict) -> list[dict]:
    try:
        import wfdb
    except ImportError:
        print("[ecg] wfdb not installed; skipping ECG eval")
        return []

    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    rows = []
    records = args.ecg_records
    fs = 360.0
    n = args.ecg_len

    classical = _build_classical_methods(args, fs)

    hybrid_loaded = {}
    for name, path in hybrid_paths.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", None)
        ctrl_type = getattr(cfg_dict, 'controller_type', 'hybrid') if cfg_dict else 'hybrid'
        lstm_hidden = getattr(cfg_dict, 'lstm_hidden', 256) if cfg_dict else 256
        n_lstm_layers = getattr(cfg_dict, 'n_lstm_layers', 2) if cfg_dict else 2

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
        hybrid_loaded[name] = ctrl

    rl_loaded = {}
    for name, path in rl_paths.items():
        try:
            m = PPO.load(path, device="cpu"); is_rec = False
        except Exception:
            m = RecurrentPPO.load(path, device="cpu"); is_rec = True
        rl_loaded[name] = (m, is_rec)

    def _scale(noise, clean, snr_db):
        ps = float(np.mean(clean ** 2))
        pn = float(np.mean(noise ** 2) + 1e-12)
        return noise * np.sqrt(ps / pn / 10 ** (snr_db / 10))

    def add_noise(clean, kind, snr_db, rng):
        if kind == "powerline":
            t = np.arange(len(clean)) / fs
            return _scale(np.sin(2 * np.pi * 50 * t), clean, snr_db)
        if kind == "gaussian":
            return make_noise("gaussian", clean, rng, snr_db=snr_db, fs=fs)
        if kind == "impulsive":
            return make_noise("impulsive", clean, rng, snr_db=snr_db, fs=fs)
        if kind == "burst":
            return make_noise("burst", clean, rng, snr_db=snr_db, fs=fs)
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
                    rng = np.random.default_rng(seed * 7919 + hash(noise_kind) % 1000)
                    noisy = clean + add_noise(clean, noise_kind, snr, rng)

                    # Classical
                    for mname, (factory, single) in classical.items():
                        filt = factory()
                        e, dt = _classical_run(filt, noisy, clean, args.order,
                                                single_input=single)
                        rows.append(dict(method=mname, record=rec_id,
                                         noise=noise_kind, snr_db=snr, seed=seed,
                                         ss_mse=float(steady_state_mse(e)),
                                         ss_mse_db=float(10 * np.log10(
                                             steady_state_mse(e) + 1e-12)),
                                         inference_time_ms=float(dt), db=db))

                    # Hybrid models
                    for mname, ctrl in hybrid_loaded.items():
                        env_cfg = EnvConfigV2(fs=fs, episode_len=n,
                                              filter_order=args.order,
                                              normalize_input=True)
                        env = AdaptiveFilterEnvV2(env_cfg, seed=seed)
                        s = float(np.std(noisy)) + 1e-9
                        env.noisy = noisy / s
                        env.clean = clean / s
                        env._task_meta = dict(family=noise_kind, snr_db=snr,
                                              signal="ecg")
                        env.t = 0
                        env.w[:] = 0.0
                        env.x_buf[:] = 0.0
                        env.feat_buf[:] = 0.0
                        env.last_e = env.last_last_e = env.last_last2_e = 0.0
                        env.episode_errors = []
                        env.running_error_sq_ema = float(np.var(env.noisy[:64])) + 1e-3
                        env.running_error_ema = float(np.mean(np.abs(env.noisy[:64]))) + 1e-3
                        env.divergence_count = 0
                        env._diverged_this_step = False
                        obs = env._obs()
                        state = None
                        errs = []
                        done = False
                        steps = 0
                        while not done and steps < n:
                            obs_2d = obs.reshape(env_cfg.state_window, -1)[-1]
                            obs_t = torch.tensor(obs_2d, dtype=torch.float32,
                                                  device=device).unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():
                                action, state, _, _, _, _ = ctrl(obs_t, state)
                            action_np = action[0, 0].cpu().numpy()
                            obs, _, term, trunc, _ = env.step(action_np)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        e = np.array(errs)[max(1, len(errs) // 20):] * s
                        rows.append(dict(method=mname, record=rec_id,
                                         noise=noise_kind, snr_db=snr, seed=seed,
                                         ss_mse=float(steady_state_mse(e)),
                                         ss_mse_db=float(10 * np.log10(
                                             steady_state_mse(e) + 1e-12)),
                                         inference_time_ms=0.0, db=db))

                    # RL v3 models
                    for mname, (model, is_rec) in rl_loaded.items():
                        env_cfg = EnvConfigV2(fs=fs, episode_len=n,
                                              filter_order=args.order,
                                              normalize_input=True)
                        env = AdaptiveFilterEnvV2(env_cfg, seed=seed)
                        s = float(np.std(noisy)) + 1e-9
                        env.noisy = noisy / s
                        env.clean = clean / s
                        env._task_meta = dict(family=noise_kind, snr_db=snr,
                                              signal="ecg")
                        env.t = 0
                        env.w[:] = 0.0
                        env.x_buf[:] = 0.0
                        env.feat_buf[:] = 0.0
                        env.last_e = env.last_last_e = env.last_last2_e = 0.0
                        env.episode_errors = []
                        env.running_error_sq_ema = float(np.var(env.noisy[:64])) + 1e-3
                        env.running_error_ema = float(np.mean(np.abs(env.noisy[:64]))) + 1e-3
                        env.divergence_count = 0
                        env._diverged_this_step = False
                        obs = env._obs()
                        lstm_state = None
                        starts = np.ones((1,), dtype=bool) if is_rec else None
                        errs = []
                        done = False
                        steps = 0
                        while not done and steps < n:
                            if is_rec:
                                a, lstm_state = model.predict(
                                    obs[None, :], state=lstm_state,
                                    episode_start=starts, deterministic=True)
                                starts = np.zeros((1,), dtype=bool)
                                a_use = a[0]
                            else:
                                a_use, _ = model.predict(obs, deterministic=True)
                            obs, _, term, trunc, _ = env.step(a_use)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        e = np.array(errs)[max(1, len(errs) // 20):] * s
                        rows.append(dict(method=mname, record=rec_id,
                                         noise=noise_kind, snr_db=snr, seed=seed,
                                         ss_mse=float(steady_state_mse(e)),
                                         ss_mse_db=float(10 * np.log10(
                                             steady_state_mse(e) + 1e-12)),
                                         inference_time_ms=0.0, db=db))
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
                            "mitdb/105", "mitdb/115",
                            "qtdb/sel100", "qtdb/sel102", "qtdb/sel103"])
    p.add_argument("--ecg-noises", nargs="+",
                   default=["powerline", "gaussian", "impulsive", "burst"])
    p.add_argument("--ecg-snrs", type=float, nargs="+", default=[0, 5, 10])
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

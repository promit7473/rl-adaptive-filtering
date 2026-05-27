"""Unified evaluation: classical + new baselines + Meta-RL on synthetic and ECG.

Outputs (under --out-dir, default results/v2_eval/):
  synthetic.csv    one row per (method, family, snr, seed)
  ecg.csv          one row per (method, record, noise, snr, seed)

Methods evaluated:
  Classical: NLMS, RLS, VSS-Kwong, Heuristic-Scheduler
  New:       PID-NLMS, Fixed-Leaky-NLMS (lambda=0.99), IIR-Notch (50 Hz)
  RL:        --rl-models name=path.zip [name=path.zip ...]

Usage:
  python3 scripts/eval.py --out-dir results/v2_eval \\
    --rl-models Meta-RL=results/v2_meta/ppo_meta_seed442_final.zip
"""
from __future__ import annotations
import argparse
import csv
import os
import time
import numpy as np

from src.signals.generators import make_signal
from src.noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from src.filters import (
    NLMS, RLS, VSSLMS, HeuristicMuScheduler,
    PIDLeakyNLMS, FixedLeakageNLMS, IIRNotch, MetaAFFilter, windowize,
)
from src.envs import AdaptiveFilterEnv, EnvConfig
from src.eval.metrics import steady_state_mse, convergence_time

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
    meta_af_path = getattr(args, 'meta_af_path', 'results/v2_meta_af/meta_af.pt')
    if meta_af_path and os.path.exists(meta_af_path):
        methods["Meta-AF (BPTT)"] = (
            lambda p=meta_af_path: MetaAFFilter.load(p, device="cpu",
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


def eval_rl(args, model_paths: dict[str, str]) -> list[dict]:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    rows = []
    for name, path in model_paths.items():
        try:
            model = PPO.load(path, device="cpu"); is_rec = False
        except Exception:
            model = RecurrentPPO.load(path, device="cpu"); is_rec = True
        per_tap = int(np.prod(model.action_space.shape)) == args.order + 1
        for sig_kind in args.signals:
          for fam in args.families:
            for snr in args.snrs:
                for seed in args.seeds:
                    cfg = EnvConfig(fs=args.fs, episode_len=args.n,
                                    filter_order=args.order,
                                    state_window=int(np.prod(model.observation_space.shape)) // 7,
                                    per_tap_action=per_tap)
                    env = AdaptiveFilterEnv(cfg, fixed_family=fam,
                                            fixed_signal=sig_kind,
                                            fixed_snr_db=snr, seed=seed)
                    obs, _ = env.reset(seed=seed)
                    state = None
                    starts = np.ones((1,), dtype=bool) if is_rec else None
                    errs = []
                    t0 = time.perf_counter()
                    done = False
                    while not done:
                        if is_rec:
                            a, state = model.predict(obs[None, :], state=state,
                                                     episode_start=starts,
                                                     deterministic=True)
                            starts = np.zeros((1,), dtype=bool)
                            a_use = a[0]
                        else:
                            a_use, _ = model.predict(obs, deterministic=True)
                        obs, _, term, trunc, _ = env.step(a_use)
                        errs.append(env.last_e)
                        done = term or trunc
                    dt = (time.perf_counter() - t0) * 1000
                    rows.append(_row(name, fam, snr, seed, np.array(errs), dt,
                                     signal=sig_kind))
    return rows


def eval_ecg(args, model_paths: dict[str, str]) -> list[dict]:
    """Zero-shot MIT-BIH eval. Adds 50 Hz powerline + Gaussian + impulsive noise."""
    try:
        import wfdb
    except ImportError:
        print("[ecg] wfdb not installed; skipping ECG eval")
        return []
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    rows = []
    records = args.ecg_records
    fs = 360.0  # MIT-BIH native
    n = args.ecg_len

    # NSTDB realistic noise records (baseline wander, muscle artifact,
    # electrode motion); cached on first load.
    nstdb_cache = {}

    def _nstdb(name, n_samples, rng):
        if name in nstdb_cache:
            sig = nstdb_cache[name]
        else:
            try:
                s, _ = wfdb.rdsamp(name, channels=[0], pn_dir="nstdb")
                sig = s[:, 0].astype(float)
            except Exception as e:
                print(f"[ecg] nstdb {name} failed: {e}; falling back to gaussian")
                return None
            nstdb_cache[name] = sig
        start = int(rng.integers(0, max(1, len(sig) - n_samples)))
        return sig[start:start + n_samples].copy()

    def _scale(noise, clean, snr_db):
        ps = float(np.mean(clean ** 2)); pn = float(np.mean(noise ** 2) + 1e-12)
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
        if kind in ("bw", "ma", "em"):  # NSTDB realistic noise
            v = _nstdb(kind, len(clean), rng)
            if v is None:
                return make_noise("gaussian", clean, rng, snr_db=snr_db, fs=fs)
            return _scale(v, clean, snr_db)
        raise ValueError(kind)

    rl_loaded = {}
    for name, path in model_paths.items():
        try:
            m = PPO.load(path, device="cpu"); is_rec = False
        except Exception:
            m = RecurrentPPO.load(path, device="cpu"); is_rec = True
        per_tap = int(np.prod(m.action_space.shape)) == args.order + 1
        rl_loaded[name] = (m, is_rec, per_tap)

    for rec_id in records:
        # rec_id may be "mitdb/100" or "qtdb/sel100" or just "100" -> mitdb
        if "/" in rec_id:
            db, name = rec_id.split("/", 1)
        else:
            db, name = "mitdb", rec_id
        try:
            sig, _ = wfdb.rdsamp(name, channels=[0], pn_dir=db, sampto=n + 500)
        except Exception as e:
            print(f"[ecg] failed to load {db}/{name}: {e}")
            continue
        clean = sig[500:500 + n, 0].astype(float)
        clean = (clean - clean.mean()) / (clean.std() + 1e-9)

        for noise_kind in args.ecg_noises:
            for snr in args.ecg_snrs:
                for seed in args.seeds:
                    rng = np.random.default_rng(seed * 7919 + hash(noise_kind) % 1000)
                    noisy = clean + add_noise(clean, noise_kind, snr, rng)

                    # Classical + Meta-AF baselines (built fresh per trial)
                    classical = {k: (factory(), single) for k, (factory, single)
                                 in _build_classical_methods(args, fs).items()}
                    for name, (filt, single) in classical.items():
                        e, dt = _classical_run(filt, noisy, clean, args.order,
                                                single_input=single)
                        rows.append(dict(method=name, record=rec_id,
                                         noise=noise_kind, snr_db=snr, seed=seed,
                                         ss_mse=float(steady_state_mse(e)),
                                         ss_mse_db=float(10 * np.log10(
                                             steady_state_mse(e) + 1e-12)),
                                         inference_time_ms=float(dt),
                                         db=db))

                    # RL policies — fresh env, fresh hidden state per cell
                    for name, (model, is_rec, per_tap) in rl_loaded.items():
                        cfg = EnvConfig(fs=fs, episode_len=n,
                                        filter_order=args.order,
                                        normalize_input=True,
                                        state_window=int(np.prod(model.observation_space.shape)) // 7,
                                        per_tap_action=per_tap)
                        env = AdaptiveFilterEnv(cfg, seed=seed)
                        # Match the env's per-episode normalization to keep
                        # feature distribution identical to training.
                        s = float(np.std(noisy)) + 1e-9
                        env.noisy = noisy / s
                        env.clean = clean / s
                        env._task_meta = dict(family=noise_kind, snr_db=snr,
                                              signal="ecg")
                        env.t = 0
                        env.w[:] = 0.0; env.x_buf[:] = 0.0
                        env.feat_buf[:] = 0.0
                        env.last_e = env.last_last_e = 0.0
                        env.episode_errors = []
                        env.running_error_sq_ema = float(
                            np.var(env.noisy[:64])) + 1e-3
                        env.divergence_count = 0
                        env._diverged_this_step = False
                        obs = env._obs()
                        state = None  # fresh LSTM hidden state per (rec,noise,snr,seed)
                        starts = np.ones((1,), dtype=bool) if is_rec else None
                        errs = []
                        done = False
                        while not done:
                            if is_rec:
                                a, state = model.predict(obs[None, :], state=state,
                                                          episode_start=starts,
                                                          deterministic=True)
                                starts = np.zeros((1,), dtype=bool)
                                a_use = a[0]
                            else:
                                a_use, _ = model.predict(obs, deterministic=True)
                            obs, _, term, trunc, _ = env.step(a_use)
                            errs.append(env.last_e)
                            done = term or trunc
                        # Discard first ~5% as warm-up; rescale errors back to
                        # original signal units so dB is comparable to classical.
                        e = np.array(errs)[max(1, len(errs)//20):] * s
                        rows.append(dict(method=name, record=rec_id,
                                         noise=noise_kind, snr_db=snr, seed=seed,
                                         ss_mse=float(steady_state_mse(e)),
                                         ss_mse_db=float(10 * np.log10(
                                             steady_state_mse(e) + 1e-12)),
                                         inference_time_ms=0.0,
                                         db=db))
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
    p.add_argument("--out-dir", default="results/v2_eval")
    p.add_argument("--rl-models", nargs="*", default=[],
                   help="name=path.zip pairs")
    p.add_argument("--families", nargs="+", default=ALL_FAMILIES)
    p.add_argument("--signals", nargs="+",
                   default=["multitone", "ecg_like",
                            "random_pulses", "square_burst"])
    p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--fs", type=float, default=360.0)
    p.add_argument("--order", type=int, default=16)
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--skip-ecg", action="store_true")
    p.add_argument("--ecg-records", nargs="+",
                   default=["mitdb/100", "mitdb/101", "mitdb/103",
                            "mitdb/105", "mitdb/115",
                            "qtdb/sel100", "qtdb/sel102", "qtdb/sel103"])
    p.add_argument("--ecg-noises", nargs="+",
                   default=["powerline", "gaussian", "impulsive", "burst",
                            "bw", "ma", "em"])
    p.add_argument("--ecg-snrs", type=float, nargs="+", default=[0, 5, 10])
    p.add_argument("--ecg-len", type=int, default=10800)  # 30 s @ 360 Hz
    p.add_argument("--meta-af-path", default="results/v2_meta_af/meta_af.pt",
                   help="trained Meta-AF (BPTT) checkpoint; ignored if missing")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_paths = dict(s.split("=", 1) for s in args.rl_models)

    if not args.skip_synthetic:
        rows = eval_synthetic(args)
        if model_paths:
            rows.extend(eval_rl(args, model_paths))
        _save(rows, os.path.join(args.out_dir, "synthetic.csv"))

    if not args.skip_ecg:
        rows = eval_ecg(args, model_paths)
        _save(rows, os.path.join(args.out_dir, "ecg.csv"))


if __name__ == "__main__":
    main()

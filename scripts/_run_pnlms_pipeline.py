#!/usr/bin/env python3
"""v4 PNLMS training pipeline — per-tap step sizes + functional expansion.

Trains the PNLMS controller that outputs M+1 continuous actions:
  - M per-tap step sizes (proportionate adaptation)
  - 1 global leakage factor

Then evaluates against classical baselines + Meta-AF + CNN.
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

RESULTS_DIR = "results/v4_pnlms"
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _find_latest_ckpt(out_dir):
    if not os.path.isdir(out_dir):
        return None
    best_path = None
    best_iter = -1
    for fname in os.listdir(out_dir):
        if fname.startswith("controller_it") and fname.endswith(".pt"):
            try:
                it = int(fname.replace("controller_it", "").replace(".pt", ""))
                if it > best_iter:
                    best_iter = it
                    best_path = os.path.join(out_dir, fname)
            except ValueError:
                pass
    if os.path.isfile(os.path.join(out_dir, "controller_final.pt")):
        return "__FINAL__"
    return best_path


def run_pnlms(seed):
    from src.agents.pnlms_trainer import train_pnlms, PNLMSTrainConfig
    cfg = PNLMSTrainConfig(
        n_iters=4000,
        batch_size=16,
        episode_len=2000,
        trunc_bptt=64,
        device='cuda',
        seed=seed,
        save_every=500,
        eval_every=200,
        rl_phase_start=800,
        rl_loss_weight=0.3,
        aux_signal_weight=0.3,
        aux_error_weight=0.2,
        aux_task_weight=0.1,
        convergence_bonus=0.15,
        robust_alpha=0.05,
        use_fx=True,
        fx_order=3,
        filter_order=16,
        lr=3e-4,
        curriculum_ramp_iters=1500,
        rl_n_envs=4,
        meta_episode_len=3,
        ppo_epochs=2,
        ppo_mb_size=64,
        proportionality_weight=0.01,
    )
    out = os.path.join(RESULTS_DIR, f"pnlms_seed{seed}")
    latest = _find_latest_ckpt(out)
    if latest == "__FINAL__":
        final_path = os.path.join(out, "controller_final.pt")
        log(f"Skipping PNLMS seed={seed} (already complete)")
        return final_path
    resume_from = latest if latest else None
    log(f"Starting PNLMS training seed={seed}, out={out}" +
        (f", resuming from {resume_from}" if resume_from else ""))
    controller, records, path = train_pnlms(cfg, out_dir=out, resume_from=resume_from)
    log(f"Finished PNLMS seed={seed}, path={path}")
    return path


def run_eval(model_paths):
    from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
    from src.signals.generators import make_signal
    from src.noise.families import make_noise
    from src.filters import NLMS, RLS, windowize
    from src.filters.diff_pnlms import PNLMSFilterWrapper, DiffPNLMSConfig, decode_action_pnlms
    from src.agents.pnlms_controller import PNLMSController
    from src.eval.metrics import steady_state_mse, convergence_time
    from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
    import csv

    ALL_FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    FAMILIES_FOR_EVAL = ["gaussian", "impulsive", "regime_switch", "burst", "alpha_stable"]
    SIGNALS_FOR_EVAL = ["multitone", "ecg_like", "random_pulses", "square_burst"]
    SNRS = [0, 5, 10, 15, 20]
    SEEDS = [0, 1, 2, 3, 4]
    FS = 360.0
    N = 2000
    ORDER = 16
    SW = 32

    eval_dir = os.path.join(RESULTS_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    rows = []

    def _row(method, family, snr, seed, e, dt_ms, **extra):
        e = np.asarray(e)
        ss = steady_state_mse(e)
        return dict(method=method, family=family, snr_db=snr, seed=seed,
                    ss_mse=float(ss), ss_mse_db=float(10 * np.log10(ss + 1e-12)),
                    ep_mse=float(np.mean(e ** 2)),
                    conv_time=float(convergence_time(e)),
                    inference_time_ms=float(dt_ms), **extra)

    # --- Classical baselines ---
    log("Evaluating classical baselines...")
    classical_methods = {
        "NLMS": (lambda: NLMS(order=ORDER, mu=0.5), False),
        "RLS": (lambda: RLS(order=ORDER, forgetting=0.995), False),
    }
    for sig_kind in SIGNALS_FOR_EVAL:
        for fam in FAMILIES_FOR_EVAL:
            for snr in SNRS:
                for seed in SEEDS:
                    rng = np.random.default_rng(int(1e6) + seed * 1000 + hash(fam + sig_kind) % 1000)
                    clean = make_signal(sig_kind, N, fs=FS, rng=rng) if sig_kind != "multitone" else \
                        make_signal("multitone", N, fs=FS, rng=rng)
                    noisy = clean + make_noise(fam, clean, rng, snr_db=snr, fs=FS)
                    for name, (factory, single) in classical_methods.items():
                        filt = factory()
                        if single:
                            y = filt.run(noisy)
                            e = clean - y
                        else:
                            U = windowize(noisy, ORDER)
                            _, e = filt.run(U, clean)
                        rows.append(_row(name, fam, snr, seed, e, 0.0, signal=sig_kind))

    # --- PNLMS models ---
    for name, path in model_paths.items():
        log(f"Evaluating {name}...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", None)
        M = getattr(cfg_dict, 'filter_order', 16) if cfg_dict else 16
        feat_dim = 11 + M

        ctrl = PNLMSController(
            feat_dim=feat_dim, hidden=getattr(cfg_dict, 'lstm_hidden', 256) if cfg_dict else 256,
            n_lstm_layers=getattr(cfg_dict, 'n_lstm_layers', 2) if cfg_dict else 2,
            filter_order=M,
            n_families=getattr(cfg_dict, 'n_families', 8) if cfg_dict else 8,
        )
        ctrl.load_state_dict(ckpt["state_dict"])
        ctrl.eval()

        diff_cfg = DiffPNLMSConfig(
            order=M,
            use_fx=getattr(cfg_dict, 'use_fx', True) if cfg_dict else True,
            fx_order=getattr(cfg_dict, 'fx_order', 3) if cfg_dict else 3,
        )

        for sig_kind in SIGNALS_FOR_EVAL:
            for fam in FAMILIES_FOR_EVAL:
                for snr in SNRS:
                    for seed in SEEDS:
                        env_cfg = EnvConfigV2(fs=FS, episode_len=N, filter_order=M,
                                              mu_min=0.001, mu_max=2.0,
                                              leakage_min=0.80, leakage_max=1.0,
                                              per_tap_action=True)
                        env = AdaptiveFilterEnvV2(env_cfg, fixed_family=fam,
                                                   fixed_signal=sig_kind,
                                                   fixed_snr_db=snr, seed=seed)
                        obs, _ = env.reset(seed=seed)
                        state = None
                        errs = []
                        t0 = time.perf_counter()
                        done = False
                        steps = 0
                        while not done and steps < N:
                            obs_2d = obs.reshape(SW, -1)[-1]
                            obs_t = torch.tensor(obs_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():
                                action, state, _, _, _, _ = ctrl(obs_t, state)
                            action_np = action[0, 0].cpu().numpy()
                            obs, _, term, trunc, _ = env.step(action_np)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        dt = (time.perf_counter() - t0) * 1000
                        rows.append(_row(name, fam, snr, seed, np.array(errs), dt, signal=sig_kind))

    # --- Save results ---
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        csv_path = os.path.join(eval_dir, "synthetic.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        log(f"Wrote {len(rows)} rows -> {csv_path}")

    # --- Summary ---
    import pandas as pd
    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby("method")["ss_mse_db"].agg(["mean", "std", "count"]).round(2)
        summary = summary.sort_values("mean")
        log(f"\n{'='*60}\nSUMMARY TABLE (ss_mse_db, lower is better):\n{summary}\n{'='*60}")
        summary.to_csv(os.path.join(eval_dir, "summary.csv"))

    log("Evaluation complete.")


if __name__ == "__main__":
    log("=" * 60)
    log("v4 PNLMS Pipeline Starting (resume-capable)")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
    log("Cleared CUDA cache")
    log("=" * 60)

    model_paths = {}

    for seed in [42, 142, 242]:
        try:
            p = run_pnlms(seed)
            model_paths[f"PNLMS-v4-s{seed}"] = p
        except Exception as e:
            log(f"ERROR PNLMS seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        torch.cuda.empty_cache()

    try:
        run_eval(model_paths)
    except Exception as e:
        log(f"ERROR eval: {e}")
        traceback.print_exc(file=open(LOG_FILE, "a"))

    log("Pipeline complete.")

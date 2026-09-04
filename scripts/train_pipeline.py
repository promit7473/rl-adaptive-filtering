#!/usr/bin/env python3
"""Production training + benchmarking pipeline.

Runs all training phases sequentially:
  1. Hybrid BPTT+RL (the main model)
  2. Pure BPTT baseline (Meta-AF comparison)
  3. RL-only baseline (RecurrentPPO)

Then evaluates everything (classical baselines + Meta-AF + our three
controllers) on IDENTICAL pre-generated episodes, in raw signal units,
so every Table I row comes from one consistent run.

Scientific guarantees enforced here:
  - Training samples ONLY the 5 train families; alpha_stable / burst /
    chirp_interferer are strictly held out (zero-shot OOD at eval).
  - The action decode (decaying base mu schedule) is identical in the
    BPTT phase, the PPO phase, and evaluation (env mu_base_schedule).
  - All methods are scored on the same (clean, noisy) realization per
    (signal, family, snr, seed) cell, with errors in raw units.
"""
import sys
import os
import time
import zlib
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

torch.set_num_threads(1)
torch.set_flush_denormal(True)

RESULTS_DIR = "results/runs"
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")
META_AF_PATH = "results/meta_af/meta_af.pt"

# Training budget (set after GPU timing test on the 5070 Ti:
# ~11 s/BPTT iter at episode_len=1000/batch=24, ~14 s PPO phase)
N_ITERS = 2000
RL_PHASE_START = 400

# One architecture everywhere: 2-layer LSTM-256 (~1M params), as stated
# in the paper abstract and Fig. 1.
LSTM_HIDDEN = 256
N_LSTM_LAYERS = 2

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def _find_latest_ckpt(out_dir):
    """Find the latest checkpoint in out_dir, or None."""
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


# ============================================================
# Phase 1: Hybrid BPTT+RL (main model)
# ============================================================
def hybrid_cfg(seed, **overrides):
    """Single source of truth for the hybrid training config.

    run_ablations.py derives its reduced-budget variants from this so
    ablation deltas measure the ablated component, not config drift.
    """
    from src.agents.hybrid_trainer import HybridTrainConfig
    kw = dict(
        n_iters=N_ITERS,
        batch_size=24,
        episode_len=1000,
        trunc_bptt=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        save_every=250,
        eval_every=200,
        rl_phase_start=RL_PHASE_START,
        rl_loss_weight=0.5,
        rl_every=2,
        aux_signal_weight=0.3,
        aux_error_weight=0.2,
        aux_task_weight=0.1,
        convergence_bonus=0.15,
        robust_alpha=0.05,
        controller_type='hybrid',
        lstm_hidden=LSTM_HIDDEN,
        n_lstm_layers=N_LSTM_LAYERS,
        feat_dim=11,
        lr=3e-4,
        curriculum_ramp_iters=800,
        rl_n_envs=4,
        meta_episode_len=3,
        ppo_epochs=4,
        ppo_mb_size=128,
        terminal_ss_weight=0.3,
        no_reward_clip=True,
        dropout=0.1,
        use_mu_schedule=True,
    )
    kw.update(overrides)
    return HybridTrainConfig(**kw)


def run_hybrid(seed):
    from src.agents.hybrid_trainer import train_hybrid
    cfg = hybrid_cfg(seed)
    out = os.path.join(RESULTS_DIR, f"hybrid_seed{seed}")
    latest = _find_latest_ckpt(out)
    if latest == "__FINAL__":
        final_path = os.path.join(out, "controller_final.pt")
        log(f"Skipping hybrid seed={seed} (already complete)")
        return final_path
    resume_from = latest if latest else None
    log(f"Starting hybrid training seed={seed}, out={out}" +
        (f", resuming from {resume_from}" if resume_from else ""))
    controller, records, path = train_hybrid(cfg, out_dir=out, resume_from=resume_from)
    log(f"Finished hybrid seed={seed}, path={path}")
    return path


# ============================================================
# Phase 2: Pure BPTT baseline
# ============================================================
def run_bptt(seed):
    from src.agents.hybrid_trainer import train_hybrid
    cfg = hybrid_cfg(seed, rl_phase_start=10**9, rl_loss_weight=0.0,
                     curriculum_ramp_iters=1000)
    out = os.path.join(RESULTS_DIR, f"bptt_seed{seed}")
    latest = _find_latest_ckpt(out)
    if latest == "__FINAL__":
        final_path = os.path.join(out, "controller_final.pt")
        log(f"Skipping BPTT seed={seed} (already complete)")
        return final_path
    resume_from = latest if latest else None
    log(f"Starting BPTT training seed={seed}, out={out}" +
        (f", resuming from {resume_from}" if resume_from else ""))
    controller, records, path = train_hybrid(cfg, out_dir=out, resume_from=resume_from)
    log(f"Finished BPTT seed={seed}, path={path}")
    return path


# ============================================================
# Phase 3: RL-only (RecurrentPPO on V2 env)
# ============================================================
# Action space + decode shared with the hybrid so the comparison
# isolates the training algorithm, not the action parameterization.
# episode_len == MU_SCHEDULE_REF (1000) so RL-only trains under the exact same
# base step-size curve as the BPTT phase, the hybrid RL phase, and eval.
RL_ENV_KW = dict(
    fs=360.0, episode_len=1000, filter_order=16,
    mu_min=0.005, mu_max=2.0, leakage_min=0.80, leakage_max=0.999,
    state_window=1,
    mu_base_schedule=True,
    reward_kind="shaped_log_mse",
    convergence_bonus=0.15, robust_alpha=0.05, robust_beta=0.02,
    terminal_ss_weight=0.3, no_reward_clip=True,
)


def run_rl(seed):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
    from src.noise.families import TRAIN_FAMILIES
    import csv

    # TRAIN FAMILIES ONLY — OOD stays held out (zero-shot claim).
    fam_weights = [1.0, 1.5, 2.0, 2.0, 3.0]
    sig_weights = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)

    env_cfg = EnvConfigV2(
        train_families=tuple(TRAIN_FAMILIES),
        family_weights=tuple(fam_weights),
        signal_kinds=("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst"),
        signal_weights=sig_weights,
        **RL_ENV_KW,
    )

    def make_env(s):
        def _f():
            return Monitor(AdaptiveFilterEnvV2(env_cfg, seed=s))
        return _f

    out = os.path.join(RESULTS_DIR, f"rl_seed{seed}")
    os.makedirs(out, exist_ok=True)

    final_path = os.path.join(out, f"rl_seed{seed}_final.zip")
    if os.path.isfile(final_path):
        log(f"Skipping RL seed={seed} (already complete)")
        return final_path

    log(f"Starting RL training seed={seed}, out={out}")

    fns = [make_env(seed + i) for i in range(8)]
    vec = DummyVecEnv(fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        lstm_hidden_size=LSTM_HIDDEN, n_lstm_layers=N_LSTM_LAYERS,
        shared_lstm=False, enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy", vec,
        learning_rate=1e-4, n_steps=2048, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(out, "tb"),
    )

    class MetricsCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records = []
        def _on_step(self):
            for info in self.locals.get("infos", []):
                if isinstance(info, dict) and "episode_ss_mse" in info:
                    self.records.append({
                        "step": int(self.num_timesteps),
                        "ss_mse": float(info["episode_ss_mse"]),
                        "ss_mse_db": float(10 * np.log10(info["episode_ss_mse"] + 1e-12)),
                        "family": info.get("task", {}).get("family", ""),
                    })
            return True

    cb_m = MetricsCB()
    cb_ckpt = CheckpointCallback(
        save_freq=200000, save_path=os.path.join(out, "checkpoints"),
        name_prefix=f"rl_seed{seed}")

    model.learn(total_timesteps=300_000,
                callback=[cb_m, cb_ckpt], progress_bar=False)

    final_path = os.path.join(out, f"rl_seed{seed}_final.zip")
    model.save(final_path)
    vec_path = os.path.join(out, f"rl_seed{seed}_vecnormalize.pkl")
    vec.save(vec_path)

    if cb_m.records:
        with open(os.path.join(out, f"train_records_seed{seed}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cb_m.records[0].keys()))
            w.writeheader(); w.writerows(cb_m.records)

    log(f"Finished RL seed={seed}, path={final_path}")
    return final_path


# ============================================================
# Phase 4: Evaluate everything on identical episodes
# ============================================================
FAMILIES_FOR_EVAL = ["gaussian", "colored", "impulsive", "time_varying",
                     "regime_switch", "alpha_stable", "burst", "chirp_interferer"]
SIGNALS_FOR_EVAL = ["multitone", "ecg_like", "random_pulses", "square_burst"]
SNRS = [0, 5, 10, 15, 20]
SEEDS = [0, 1, 2, 3, 4]
FS = 360.0
N = 4000
ORDER = 16


def _episode_rng(sig_kind, fam, snr, seed):
    # Stable (process-independent) seeding so episodes are reproducible.
    key = zlib.crc32(f"{sig_kind}|{fam}|{snr}".encode()) % 100000
    return np.random.default_rng(int(1e6) + seed * 1000_000 + key)


def make_episodes():
    """Pre-generate every eval episode once; all methods share them."""
    from src.signals.generators import make_signal
    from src.noise.families import make_noise
    episodes = {}
    for sig_kind in SIGNALS_FOR_EVAL:
        for fam in FAMILIES_FOR_EVAL:
            for snr in SNRS:
                for seed in SEEDS:
                    rng = _episode_rng(sig_kind, fam, snr, seed)
                    clean = make_signal(sig_kind, N, fs=FS, rng=rng)
                    noisy = clean + make_noise(fam, clean, rng, snr_db=snr, fs=FS)
                    episodes[(sig_kind, fam, snr, seed)] = (clean, noisy)
    return episodes


def run_eval(hybrid_paths, bptt_paths, rl_paths):
    from src.filters import (
        NLMS, RLS, VSSLMS, AboulnasrMayyasVSS, HeuristicMuScheduler,
        PIDLeakyNLMS, FixedLeakageNLMS, MetaAFFilter, windowize,
    )
    from src.eval.runner import (
        load_controller, run_controller_episode, run_rl_episode, load_rl, metrics_row,
    )
    import csv

    eval_dir = os.path.join(RESULTS_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    rows = []

    log("Pre-generating shared eval episodes...")
    episodes = make_episodes()

    def _row(method, sig, fam, snr, seed, e, dt_ms, divergence_count=0):
        return metrics_row(method, e, dt_ms, signal=sig, family=fam, snr_db=snr,
                           seed=seed, divergence_count=int(divergence_count))

    # --- Classical baselines + Meta-AF (raw units) ---
    classical_methods = {
        "NLMS": (lambda: NLMS(order=ORDER, mu=0.5)),
        "RLS": (lambda: RLS(order=ORDER, forgetting=0.995)),
        "VSS-Kwong": (lambda: VSSLMS(order=ORDER, mu_max=0.05, alpha=0.97, gamma=1e-3)),
        "VSS-Aboulnasr": (lambda: AboulnasrMayyasVSS(order=ORDER)),
        "Heuristic": (lambda: HeuristicMuScheduler(order=ORDER, mu_base=0.01)),
        "PID-NLMS": (lambda: PIDLeakyNLMS(order=ORDER)),
        "Fixed-Leaky": (lambda: FixedLeakageNLMS(order=ORDER, mu=0.5, leakage=0.99)),
    }
    if os.path.exists(META_AF_PATH):
        classical_methods["Meta-AF"] = (
            lambda: MetaAFFilter.load(META_AF_PATH, device="cpu", order=ORDER))
    else:
        log(f"WARNING: {META_AF_PATH} missing — Meta-AF row will be absent!")

    log("Evaluating classical baselines + Meta-AF...")
    for (sig_kind, fam, snr, seed), (clean, noisy) in episodes.items():
        for name, factory in classical_methods.items():
            filt = factory()
            t0 = time.perf_counter()
            if name == "Meta-AF":
                # Meta-AF trained on std-normalized inputs; rescale errors
                # back to raw units for a fair comparison.
                s = float(np.std(noisy)) + 1e-9
                U = windowize(noisy / s, ORDER)
                _, e = filt.run(U, clean / s)
                e = np.asarray(e) * s
            else:
                U = windowize(noisy, ORDER)
                _, e = filt.run(U, clean)
            dt = (time.perf_counter() - t0) * 1000
            rows.append(_row(name, sig_kind, fam, snr, seed, e, dt))

    # --- Our controllers (hybrid / BPTT), errors rescaled to raw units ---
    all_model_paths = {}
    all_model_paths.update(hybrid_paths)
    all_model_paths.update(bptt_paths)

    for name, path in all_model_paths.items():
        log(f"Evaluating {name}...")
        ctrl, env_kw = load_controller(path)

        for (sig_kind, fam, snr, seed), (clean, noisy) in episodes.items():
            e, dt, divc = run_controller_episode(
                ctrl, env_kw, clean, noisy, FS, N, ORDER)
            rows.append(_row(name, sig_kind, fam, snr, seed, e, dt,
                             divergence_count=divc))

    # --- RL model (sb3), errors rescaled to raw units ---
    for name, path in rl_paths.items():
        log(f"Evaluating {name}...")
        model, is_rec, vec_norm = load_rl(path)

        for (sig_kind, fam, snr, seed), (clean, noisy) in episodes.items():
            e, dt, divc = run_rl_episode(
                model, is_rec, vec_norm, clean, noisy, FS, N, ORDER)
            rows.append(_row(name, sig_kind, fam, snr, seed, e, dt,
                             divergence_count=divc))

    # --- Save results ---
    if rows:
        keys = list(rows[0].keys())
        csv_path = os.path.join(eval_dir, "synthetic.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        log(f"Wrote {len(rows)} rows -> {csv_path}")

    # --- Summary ---
    import pandas as pd
    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby("method")["ss_mse_db"].agg(["mean", "std", "count"]).round(2)
        summary = summary.sort_values("mean")
        log(f"\n{'='*60}\nSUMMARY (all SNRs, ss_mse_db, lower is better):\n{summary}\n{'='*60}")
        summary.to_csv(os.path.join(eval_dir, "summary.csv"))

        df10 = df[df.snr_db == 10]
        pivot = df10.pivot_table(index="method", columns="family",
                                 values="ss_mse_db", aggfunc="mean").round(1)
        pivot["MEAN"] = df10.groupby("method")["ss_mse_db"].mean().round(1)
        log(f"\nSNR=10 BY FAMILY (Table I source):\n{pivot}\n{'='*60}")
        pivot.to_csv(os.path.join(eval_dir, "table1_snr10.csv"))

    log("Evaluation complete.")


# ============================================================
# Main pipeline
# ============================================================
def _expected_paths(seed=42):
    hybrid = os.path.join(RESULTS_DIR, f"hybrid_seed{seed}", "controller_final.pt")
    bptt = os.path.join(RESULTS_DIR, f"bptt_seed{seed}", "controller_final.pt")
    rl = os.path.join(RESULTS_DIR, f"rl_seed{seed}", f"rl_seed{seed}_final.zip")
    return hybrid, bptt, rl


if __name__ == "__main__":
    phases = sys.argv[1:] or ["hybrid", "bptt", "rl", "eval"]
    log("=" * 60)
    log(f"Training pipeline starting (resume-capable), phases={phases}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        log("Running on CPU (CUDA not available)")
    log(f"Config: n_iters={N_ITERS}, LSTM {LSTM_HIDDEN}x{N_LSTM_LAYERS}")
    log("=" * 60)

    seed = 42
    hybrid_p, bptt_p, rl_p = _expected_paths(seed)

    if "hybrid" in phases:
        try:
            hybrid_p = run_hybrid(seed)
        except Exception as e:
            log(f"ERROR hybrid seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "bptt" in phases:
        try:
            bptt_p = run_bptt(seed)
        except Exception as e:
            log(f"ERROR BPTT seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "rl" in phases:
        try:
            rl_p = run_rl(seed)
        except Exception as e:
            log(f"ERROR RL seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "eval" in phases:
        hybrid_paths = {"Hybrid (ours)": hybrid_p} if os.path.isfile(hybrid_p) else {}
        bptt_paths = {"BPTT-only (ours)": bptt_p} if os.path.isfile(bptt_p) else {}
        rl_paths = {"RL-only": rl_p} if os.path.isfile(rl_p) else {}
        log(f"Eval models: {list(hybrid_paths) + list(bptt_paths) + list(rl_paths)}")
        try:
            run_eval(hybrid_paths, bptt_paths, rl_paths)
        except Exception as e:
            log(f"ERROR eval: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))

    log(f"Pipeline phases {phases} complete.")

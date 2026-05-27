#!/usr/bin/env python3
"""Full v3.1 production training pipeline.

Runs all training phases sequentially:
  1. Hybrid BPTT+RL (the main model)
  2. Pure BPTT baseline (Meta-AF comparison)
  3. RL-only baseline (RecurrentPPO)

Then evaluates everything against classical baselines + Meta-AF.
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

RESULTS_DIR = "results/beast"
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline.log")

os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ============================================================
# Phase 1: Hybrid BPTT+RL (main model, 3 seeds)
# ============================================================
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


def run_hybrid(seed):
    from src.agents.hybrid_trainer import train_hybrid, HybridTrainConfig
    cfg = HybridTrainConfig(
        n_iters=400,
        batch_size=16,
        episode_len=2000,
        trunc_bptt=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        save_every=200,
        eval_every=100,
        rl_phase_start=200,
        rl_loss_weight=0.5,
        aux_signal_weight=0.3,
        aux_error_weight=0.2,
        aux_task_weight=0.1,
        convergence_bonus=0.15,
        robust_alpha=0.05,
        controller_type='hybrid',
        lstm_hidden=512,
        n_lstm_layers=3,
        feat_dim=11,
        lr=3e-4,
        curriculum_ramp_iters=1500,
        rl_n_envs=6,
        meta_episode_len=4,
        ppo_epochs=4,
        ppo_mb_size=128,
        terminal_ss_weight=0.3,
        no_reward_clip=True,
        dropout=0.1,
    )
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
# Phase 2: Pure BPTT (Meta-AF killer, 2 seeds)
# ============================================================
def run_bptt(seed):
    from src.agents.hybrid_trainer import train_hybrid, HybridTrainConfig
    cfg = HybridTrainConfig(
        n_iters=400,
        batch_size=16,
        episode_len=2000,
        trunc_bptt=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=seed,
        save_every=200,
        eval_every=100,
        rl_phase_start=999999,
        rl_loss_weight=0.0,
        aux_signal_weight=0.3,
        aux_error_weight=0.2,
        aux_task_weight=0.1,
        convergence_bonus=0.15,
        robust_alpha=0.05,
        controller_type='hybrid',
        lstm_hidden=512,
        n_lstm_layers=3,
        feat_dim=11,
        lr=3e-4,
        curriculum_ramp_iters=2000,
        dropout=0.1,
    )
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
# Phase 3: RL-only (RecurrentPPO on V2 env, 2 seeds)
# ============================================================
def run_rl(seed):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
    from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
    import csv

    all_families = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    fam_weights = [1.0, 1.0, 2.0, 2.0, 3.0, 2.0, 2.0, 1.5]
    sig_weights = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)

    env_cfg = EnvConfigV2(
        fs=360.0, episode_len=2000, filter_order=16,
        mu_min=0.001, mu_max=3.0, leakage_min=0.50, leakage_max=1.0,
        state_window=1,
        train_families=tuple(all_families),
        family_weights=tuple(fam_weights),
        signal_kinds=("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst"),
        signal_weights=sig_weights,
        reward_kind="shaped_log_mse",
        convergence_bonus=0.15, robust_alpha=0.05, robust_beta=0.02,
        terminal_ss_weight=0.3, no_reward_clip=True,
    )

    def make_env(s):
        def _f():
            return Monitor(AdaptiveFilterEnvV2(env_cfg, seed=s))
        return _f

    out = os.path.join(RESULTS_DIR, f"rl_seed{seed}")
    os.makedirs(out, exist_ok=True)

    final_path = os.path.join(out, f"v31_rl_seed{seed}_final.zip")
    if os.path.isfile(final_path):
        log(f"Skipping RL seed={seed} (already complete)")
        return final_path

    log(f"Starting RL training seed={seed}, out={out}")

    # Use DummyVecEnv (no SubprocVecEnv for LSTM + VecNormalize compat)
    fns = [make_env(seed + i) for i in range(8)]
    vec = DummyVecEnv(fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        lstm_hidden_size=512, n_lstm_layers=3,
        shared_lstm=False, enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy", vec,
        learning_rate=1e-4, n_steps=2048, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        verbose=1, device='cuda', seed=seed,
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

    final_path = os.path.join(out, f"v31_rl_seed{seed}_final.zip")
    model.save(final_path)
    vec_path = os.path.join(out, f"v31_rl_seed{seed}_vecnormalize.pkl")
    vec.save(vec_path)

    if cb_m.records:
        with open(os.path.join(out, f"train_records_seed{seed}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cb_m.records[0].keys()))
            w.writeheader(); w.writerows(cb_m.records)

    log(f"Finished RL seed={seed}, path={final_path}")
    return final_path


# ============================================================
# Phase 4: Evaluate everything
# ============================================================
def run_eval(hybrid_paths, bptt_paths, rl_paths):
    from src.noise.families import TRAIN_FAMILIES, OOD_FAMILIES
    from src.signals.generators import make_signal
    from src.noise.families import make_noise
    from src.filters import (
        NLMS, RLS, VSSLMS, HeuristicMuScheduler,
        PIDLeakyNLMS, FixedLeakageNLMS, IIRNotch, MetaAFFilter, windowize,
    )
    from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2
    from src.eval.metrics import steady_state_mse, convergence_time
    from src.agents.controller import LSTMController, HybridController, TransformerController
    from src.filters.diff_filter import DiffNLMSConfig, decode_action_bptt
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    import csv

    ALL_FAMILIES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    FAMILIES_FOR_EVAL = ["gaussian", "colored", "impulsive", "time_varying",
                         "regime_switch", "alpha_stable", "burst", "chirp_interferer"]
    SIGNALS_FOR_EVAL = ["multitone", "ecg_like", "random_pulses", "square_burst"]
    SNRS = [0, 5, 10, 15, 20]
    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    FS = 360.0
    N = 4000
    ORDER = 16

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

    def _classical_run(filt, noisy, clean, order, single_input=False):
        t0 = time.perf_counter()
        if single_input:
            y = filt.run(noisy)
            e = clean - y
        else:
            U = windowize(noisy, order)
            _, e = filt.run(U, clean)
        return e, (time.perf_counter() - t0) * 1000

    # --- Classical baselines ---
    log("Evaluating classical baselines...")
    classical_methods = {
        "NLMS": (lambda: NLMS(order=ORDER, mu=0.5), False),
        "RLS": (lambda: RLS(order=ORDER, forgetting=0.995), False),
        "VSS-Kwong": (lambda: VSSLMS(order=ORDER, mu_max=0.05, alpha=0.97, gamma=1e-3), False),
        "Heuristic": (lambda: HeuristicMuScheduler(order=ORDER, mu_base=0.01), False),
        "PID-NLMS": (lambda: PIDLeakyNLMS(order=ORDER), False),
        "Fixed-Leaky": (lambda: FixedLeakageNLMS(order=ORDER, mu=0.5, leakage=0.99), False),
        "IIR-Notch-50Hz": (lambda: IIRNotch(f0=50.0, fs=FS, Q=30.0), True),
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
                        e, dt = _classical_run(filt, noisy, clean, ORDER, single_input=single)
                        rows.append(_row(name, fam, snr, seed, e, dt, signal=sig_kind))

    # --- Hybrid / BPTT models ---
    all_model_paths = {}
    all_model_paths.update(hybrid_paths)
    all_model_paths.update(bptt_paths)

    for name, path in all_model_paths.items():
        log(f"Evaluating {name}...")
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

        for sig_kind in SIGNALS_FOR_EVAL:
            for fam in FAMILIES_FOR_EVAL:
                for snr in SNRS:
                    for seed in SEEDS:
                        env_cfg = EnvConfigV2(fs=FS, episode_len=N, filter_order=ORDER,
                                              state_window=4, mu_min=0.005, mu_max=2.0,
                                              leakage_min=0.70, leakage_max=1.0)
                        env = AdaptiveFilterEnvV2(env_cfg, fixed_family=fam,
                                                  fixed_signal=sig_kind,
                                                  fixed_snr_db=snr, seed=seed)
                        obs, _ = env.reset(seed=seed)
                        feat_dim = 11
                        sw = obs.shape[0] // feat_dim
                        state = None
                        errs = []
                        t0 = time.perf_counter()
                        done = False
                        steps = 0
                        while not done and steps < N:
                            obs_2d = obs.reshape(sw, -1)[-1]
                            obs_t = torch.tensor(obs_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():
                                action, state, _, _, _, _, _ = ctrl(obs_t, state)
                            action_np = action[0, 0].cpu().numpy()
                            obs, _, term, trunc, _ = env.step(action_np)
                            errs.append(env.last_e)
                            done = term or trunc
                            steps += 1
                        dt = (time.perf_counter() - t0) * 1000
                        rows.append(_row(name, fam, snr, seed, np.array(errs), dt, signal=sig_kind))

    # --- RL models ---
    for name, path in rl_paths.items():
        log(f"Evaluating {name}...")
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

        for sig_kind in SIGNALS_FOR_EVAL:
            for fam in FAMILIES_FOR_EVAL:
                for snr in SNRS:
                    for seed in SEEDS:
                        env_cfg = EnvConfigV2(fs=FS, episode_len=N, filter_order=ORDER,
                                              state_window=1, mu_min=0.001, mu_max=3.0,
                                              leakage_min=0.50, leakage_max=1.0)
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
                        while not done and steps < N:
                            if is_rec:
                                a, lstm_state = model.predict(
                                    obs[None, :], state=lstm_state,
                                    episode_start=episode_starts,
                                    deterministic=True)
                                episode_starts = np.zeros((1,), dtype=bool)
                                a_use = a[0]
                            else:
                                a_use, _ = model.predict(obs, deterministic=True)
                            obs, _, term, trunc, _ = env.step(a_use)
                            if vec_norm is not None:
                                obs = vec_norm.normalize_obs(obs)
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
            w.writeheader(); w.writerows(rows)
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


# ============================================================
# Main pipeline
# ============================================================
if __name__ == "__main__":
    log("=" * 60)
    log("v3.1 Production Pipeline Starting (resume-capable)")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
    log("Cleared CUDA cache")
    log("=" * 60)

    hybrid_paths = {}
    bptt_paths = {}
    rl_paths = {}

    # Phase 1: Hybrid (1 seed for fast validation)
    for seed in [42]:
        try:
            p = run_hybrid(seed)
            hybrid_paths[f"Beast-Hybrid-s{seed}"] = p
        except Exception as e:
            log(f"ERROR hybrid seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        torch.cuda.empty_cache()

    # Phase 2: Pure BPTT (1 seed for fast validation)
    for seed in [42]:
        try:
            p = run_bptt(seed)
            bptt_paths[f"Beast-BPTT-s{seed}"] = p
        except Exception as e:
            log(f"ERROR BPTT seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        torch.cuda.empty_cache()

    # Phase 3: RL-only (1 seed for fast validation)
    for seed in [42]:
        try:
            p = run_rl(seed)
            rl_paths[f"Beast-RL-s{seed}"] = p
        except Exception as e:
            log(f"ERROR RL seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        torch.cuda.empty_cache()

    # Phase 4: Evaluate everything
    try:
        run_eval(hybrid_paths, bptt_paths, rl_paths)
    except Exception as e:
        log(f"ERROR eval: {e}")
        traceback.print_exc(file=open(LOG_FILE, "a"))

    log("Pipeline complete.")

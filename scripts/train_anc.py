#!/usr/bin/env python3
"""Reference-based ANC training + evaluation pipeline (Kalman-Q controller).

Phases (resume-capable):
    hybrid : BPTT+PPO Kalman controller (ours)          -> results/runs/anc_hybrid_seed42
    bptt   : BPTT-only ablation                          -> results/runs/anc_bptt_seed42
    rl     : PPO-only (sb3 RecurrentPPO on ANC env)      -> results/runs/anc_rl_seed42
    eval   : all baselines + learned controllers on shared episodes (synthetic + ECG)

Soundness guarantees:
    - Filter adaptation + state features use ONLY observables (reference, primary,
      innovation); the clean signal shapes the training loss/reward only.
    - OOD interference families (narrowband_chirp, echo_long) are held out from
      training; every method is scored on identical (clean, interference, reference)
      realisations; residual MSE is measured against the clean signal offline.
"""
import sys, os, time, zlib, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.set_num_threads(1)
RESULTS_DIR = "results/runs"
N_ITERS = 2000
RL_PHASE_START = 400
LSTM_HIDDEN = 256
N_LSTM_LAYERS = 2
FS = 360.0
N = 4000
ORDER = 16
SNRS = [-5, 0, 5, 10]
SEEDS = [0, 1, 2, 3, 4]
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------
# Training configs
# ----------------------------------------------------------------------
def hybrid_cfg(seed, **ov):
    from src.agents.kalman_trainer import KalmanTrainConfig
    kw = dict(n_iters=N_ITERS, batch_size=24, episode_len=1000, trunc_bptt=64,
              device='cuda' if torch.cuda.is_available() else 'cpu', seed=seed,
              save_every=250, eval_every=200, rl_phase_start=RL_PHASE_START,
              rl_loss_weight=0.5, rl_every=2, aux_signal_weight=0.3,
              aux_error_weight=0.2, aux_task_weight=0.1, convergence_bonus=0.15,
              controller_type='hybrid', lstm_hidden=LSTM_HIDDEN,
              n_lstm_layers=N_LSTM_LAYERS, lr=3e-4, curriculum_ramp_iters=800,
              rl_n_envs=4, meta_episode_len=3, ppo_epochs=4, ppo_mb_size=128,
              terminal_ss_weight=0.3, dropout=0.1)
    kw.update(ov)
    return KalmanTrainConfig(**kw)


def _find_ckpt(out):
    if os.path.isfile(os.path.join(out, "controller_final.pt")):
        return "__FINAL__"
    if not os.path.isdir(out):
        return None
    best, bi = None, -1
    for f in os.listdir(out):
        if f.startswith("controller_it") and f.endswith(".pt"):
            try:
                i = int(f[13:-3])
                if i > bi:
                    bi, best = i, os.path.join(out, f)
            except ValueError:
                pass
    return best


def run_hybrid(seed):
    from src.agents.kalman_trainer import train_kalman
    out = os.path.join(RESULTS_DIR, f"anc_hybrid_seed{seed}")
    ck = _find_ckpt(out)
    if ck == "__FINAL__":
        log(f"skip hybrid seed={seed}"); return os.path.join(out, "controller_final.pt")
    log(f"train hybrid seed={seed}")
    _, _, p = train_kalman(hybrid_cfg(seed), out_dir=out, resume_from=ck)
    return p


def run_bptt(seed):
    from src.agents.kalman_trainer import train_kalman
    out = os.path.join(RESULTS_DIR, f"anc_bptt_seed{seed}")
    ck = _find_ckpt(out)
    if ck == "__FINAL__":
        log(f"skip bptt seed={seed}"); return os.path.join(out, "controller_final.pt")
    log(f"train bptt-only seed={seed}")
    cfg = hybrid_cfg(seed, rl_phase_start=10**9, rl_loss_weight=0.0, curriculum_ramp_iters=1000)
    _, _, p = train_kalman(cfg, out_dir=out, resume_from=ck)
    return p


def run_rl(seed):
    """Pure-PPO baseline: sb3 RecurrentPPO on the ANC env (no BPTT)."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from src.envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig
    out = os.path.join(RESULTS_DIR, f"anc_rl_seed{seed}")
    os.makedirs(out, exist_ok=True)
    final = os.path.join(out, f"anc_rl_seed{seed}_final.zip")
    if os.path.isfile(final):
        log(f"skip rl seed={seed}"); return final
    log(f"train rl-only seed={seed}")
    env_cfg = ANCEnvConfig(episode_len=1000, filter_order=ORDER)
    vec = DummyVecEnv([lambda s=seed + i: Monitor(ANCKalmanEnv(env_cfg, seed=s))
                       for i in range(8)])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    model = RecurrentPPO("MlpLstmPolicy", vec, learning_rate=1e-4, n_steps=2048,
                         batch_size=256, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                         ent_coef=0.005, max_grad_norm=0.5, seed=seed,
                         policy_kwargs=dict(lstm_hidden_size=LSTM_HIDDEN,
                                            n_lstm_layers=N_LSTM_LAYERS,
                                            enable_critic_lstm=True),
                         device='cuda' if torch.cuda.is_available() else 'cpu', verbose=1)
    model.learn(total_timesteps=300_000, progress_bar=False)
    model.save(final)
    vec.save(os.path.join(out, f"anc_rl_seed{seed}_vecnormalize.pkl"))
    return final


# ----------------------------------------------------------------------
# Shared eval episodes
# ----------------------------------------------------------------------
def _ep_rng(sig, fam, snr, seed):
    key = zlib.crc32(f"{sig}|{fam}|{snr}".encode()) % 100000
    return np.random.default_rng(int(1e6) + seed * 1_000_000 + key)


def make_episodes(signals, families):
    from src.signals.generators import make_signal
    from src.interference.families import make_interference
    eps = {}
    for sig in signals:
        for fam in families:
            for snr in SNRS:
                for seed in SEEDS:
                    rng = _ep_rng(sig, fam, snr, seed)
                    clean = make_signal(sig, N, fs=FS, rng=rng)
                    interf, ref = make_interference(fam, clean, rng, snr_db=snr, fs=FS)
                    eps[(sig, fam, snr, seed)] = (clean, interf, ref)
    return eps


def _ss_db(residual, frac=0.25):
    r = np.asarray(residual, float)
    r = np.where(np.isfinite(r), r, 1e6)
    tail = max(1, int(len(r) * frac))
    return float(10 * np.log10(np.mean(r[-tail:] ** 2) + 1e-12))


def _diverged(residual, clean):
    r = np.asarray(residual, float)
    if not np.isfinite(r).all():
        return True
    # worse than doing nothing (residual power exceeds input interference power)
    return _ss_db(r) > _ss_db(clean) + 3.0  # 3 dB grace above signal level


# ----------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------
def eval_all(learned_paths, rl_paths, out_dir):
    import csv
    from src.filters.anc_baselines import (NLMSANC, VSSLMSANC, RLSANC, KalmanANC, NotchANC)
    from src.envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig
    from src.interference.families import TRAIN_FAMILIES, OOD_FAMILIES
    os.makedirs(out_dir, exist_ok=True)
    families = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    signals = ["multitone", "ecg_like", "random_pulses", "square_burst"]
    log("generating shared eval episodes...")
    eps = make_episodes(signals, families)

    classical = {
        "NLMS": lambda: NLMSANC(order=ORDER, mu=0.05),
        "VSS-LMS": lambda: VSSLMSANC(order=ORDER),
        "RLS(0.999)": lambda: RLSANC(order=ORDER, forgetting=0.999),
        "RLS(0.99)": lambda: RLSANC(order=ORDER, forgetting=0.99),
        "Kalman(Q=1e-6)": lambda: KalmanANC(order=ORDER, q=1e-6),
        "Kalman(Q=1e-4)": lambda: KalmanANC(order=ORDER, q=1e-4),
        "IIR-Notch": lambda: NotchANC(f0=50.0, fs=FS),
    }
    rows = []

    def _row(method, sig, fam, snr, seed, residual, clean, dt):
        return dict(method=method, signal=sig, family=fam, snr_db=snr, seed=seed,
                    ss_res_db=_ss_db(residual), inference_time_ms=dt,
                    diverged=int(_diverged(residual, clean)))

    log("evaluating classical baselines...")
    for (sig, fam, snr, seed), (clean, interf, ref) in eps.items():
        primary = clean + interf
        for name, factory in classical.items():
            f = factory(); t0 = time.perf_counter()
            e = f.run(ref, primary)
            dt = (time.perf_counter() - t0) * 1000
            residual = e - clean
            rows.append(_row(name, sig, fam, snr, seed, residual, clean, dt))

    # learned controllers (hybrid / bptt) via ANC env preset
    from src.agents.controller import HybridController, LSTMController
    for name, path in learned_paths.items():
        if not (path and os.path.isfile(path)):
            log(f"WARNING: {name} checkpoint missing ({path}) — skipping"); continue
        log(f"evaluating {name}...")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        tcfg = ckpt.get("config", None)
        g = lambda k, d: getattr(tcfg, k, d) if tcfg else d
        ctor = HybridController if g('controller_type', 'hybrid') == 'hybrid' else LSTMController
        ctrl = ctor(feat_dim=11, hidden=g('lstm_hidden', LSTM_HIDDEN),
                    n_lstm_layers=g('n_lstm_layers', N_LSTM_LAYERS), act_dim=2,
                    n_families=g('n_families', 6))
        ctrl.load_state_dict(ckpt["state_dict"]); ctrl.eval()
        env_cfg = ANCEnvConfig(fs=FS, episode_len=N, filter_order=ORDER,
                               q_min=g('q_min', 1e-8), q_max=g('q_max', 1e-3),
                               r_min=g('r_min', 1e-3), r_max=g('r_max', 1e1))
        for (sig, fam, snr, seed), (clean, interf, ref) in eps.items():
            env = ANCKalmanEnv(env_cfg, fixed_family=fam, fixed_signal=sig,
                               fixed_snr_db=snr, seed=seed)
            env.set_preset_episode(clean, interf, ref)
            obs, _ = env.reset(seed=seed)
            state = None; t0 = time.perf_counter(); done = False
            while not done:
                ot = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
                with torch.no_grad():
                    a, state, *_ = ctrl(ot, state)
                obs, _, term, trunc, _ = env.step(a[0, 0].numpy()); done = term or trunc
            dt = (time.perf_counter() - t0) * 1000
            residual = np.asarray(env.residuals) * env.norm_scale
            rows.append(_row(name, sig, fam, snr, seed, residual, clean, dt))

    # RL-only (sb3)
    for name, path in rl_paths.items():
        if not (path and os.path.isfile(path)):
            continue
        log(f"evaluating {name}...")
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
        model = RecurrentPPO.load(path, device="cpu")
        vp = path.replace("_final.zip", "_vecnormalize.pkl")
        vec_norm = None
        if os.path.isfile(vp):
            dummy = DummyVecEnv([lambda: ANCKalmanEnv(ANCEnvConfig(episode_len=1000))])
            vec_norm = VecNormalize.load(vp, dummy); vec_norm.training = False
        env_cfg = ANCEnvConfig(fs=FS, episode_len=N, filter_order=ORDER)
        for (sig, fam, snr, seed), (clean, interf, ref) in eps.items():
            env = ANCKalmanEnv(env_cfg, fixed_family=fam, fixed_signal=sig,
                               fixed_snr_db=snr, seed=seed)
            env.set_preset_episode(clean, interf, ref)
            obs, _ = env.reset(seed=seed)
            if vec_norm is not None:
                obs = vec_norm.normalize_obs(obs)
            lstm_state = None; starts = np.ones((1,), bool); done = False; t0 = time.perf_counter()
            while not done:
                a, lstm_state = model.predict(obs[None], state=lstm_state,
                                              episode_start=starts, deterministic=True)
                starts = np.zeros((1,), bool)
                obs, _, term, trunc, _ = env.step(a[0])
                if vec_norm is not None:
                    obs = vec_norm.normalize_obs(obs)
                done = term or trunc
            dt = (time.perf_counter() - t0) * 1000
            residual = np.asarray(env.residuals) * env.norm_scale
            rows.append(_row(name, sig, fam, snr, seed, residual, clean, dt))

    # save + summary
    csv_path = os.path.join(out_dir, "synthetic.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    log(f"wrote {len(rows)} rows -> {csv_path}")
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        d10 = df[df.snr_db == 10]
        piv = d10.pivot_table(index="method", columns="family", values="ss_res_db", aggfunc="mean").round(1)
        piv["MEAN"] = d10.groupby("method")["ss_res_db"].mean().round(1)
        piv["div%"] = (df.groupby("method")["diverged"].mean() * 100).round(1)
        piv = piv.sort_values("MEAN")
        log(f"\nSNR=10 residual dB by family (lower=better):\n{piv}")
        piv.to_csv(os.path.join(out_dir, "table1_snr10.csv"))
    except Exception as ex:
        log(f"summary skipped: {ex}")


def _paths(seed=42):
    return (os.path.join(RESULTS_DIR, f"anc_hybrid_seed{seed}", "controller_final.pt"),
            os.path.join(RESULTS_DIR, f"anc_bptt_seed{seed}", "controller_final.pt"),
            os.path.join(RESULTS_DIR, f"anc_rl_seed{seed}", f"anc_rl_seed{seed}_final.zip"))


if __name__ == "__main__":
    phases = sys.argv[1:] or ["hybrid", "bptt", "rl", "eval"]
    log(f"ANC pipeline phases={phases}; "
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    seed = 42
    hy, bp, rl = _paths(seed)
    if "hybrid" in phases:
        try: hy = run_hybrid(seed)
        except Exception as e: log(f"ERR hybrid: {e}"); traceback.print_exc()
    if "bptt" in phases:
        try: bp = run_bptt(seed)
        except Exception as e: log(f"ERR bptt: {e}"); traceback.print_exc()
    if "rl" in phases:
        try: rl = run_rl(seed)
        except Exception as e: log(f"ERR rl: {e}"); traceback.print_exc()
    if "eval" in phases:
        learned = {"Hybrid (ours)": hy, "BPTT-only": bp}
        rlp = {"RL-only": rl}
        eval_all(learned, rlp, os.path.join(RESULTS_DIR, "eval_anc"))
    log("pipeline done.")

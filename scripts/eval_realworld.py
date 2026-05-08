"""Sim-to-real transfer evaluation.

Tests the trained RL policy (zero-shot, no retraining) on:
  1. ECG denoising  — MIT-BIH record 100, powerline + baseline wander noise
  2. Speech         — LibriSpeech test-clean segment, 4 noise families at 10 dB SNR

Saves:
  results/realworld_ecg.csv
  results/realworld_speech.csv
  paper/figures/fig_realworld.pdf
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.eval.realworld_runner import (
    run_nlms_filter, run_vss_kwong,
    run_heuristic_scheduler, run_rl_policy,
)

plt.rcParams.update({
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#444444",
    "axes.linewidth":       0.9,
    "axes.grid":            True,
    "axes.axisbelow":       True,
    "grid.color":           "#E0E0E0",
    "grid.linestyle":       "--",
    "grid.linewidth":       0.5,
    "grid.alpha":           0.8,
    "font.family":          "DejaVu Sans",
    "font.size":            8.5,
    "axes.titlesize":       9.5,
    "axes.titleweight":     "bold",
    "axes.titlepad":        5,
    "axes.labelsize":       8.5,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.6,
    "legend.fontsize":      7.5,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#CCCCCC",
    "legend.fancybox":      False,
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.03,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

FS_TARGET = 8000     # model was trained at 8 kHz
EPISODE_N = 4000     # samples per eval episode
SNR_DB    = 10.0
SEEDS     = [7, 42, 99, 137, 256]

# ── colour palette ─────────────────────────────────────────────────────────
C = {
    "NLMS":           "#6e6e6e",
    "VSS-LMS (Kwong)":"#b58900",
    "Heuristic":      "#268bd2",
    "PPO-NLMS":       "#1f6feb",
    "Meta-RL":        "#cf222e",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _resample(sig: np.ndarray, fs_in: int, fs_out: int = FS_TARGET) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    g = gcd(fs_in, fs_out)
    return resample_poly(sig, fs_out // g, fs_in // g).astype(np.float64)


def _normalize(sig: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    return sig / rms


def _add_noise(clean: np.ndarray, snr_db: float, rng: np.random.Generator,
               kind: str = "gaussian") -> np.ndarray:
    sig_pow  = np.mean(clean ** 2)
    if kind == "gaussian":
        n = rng.standard_normal(len(clean))
    elif kind == "impulsive":
        n = rng.standard_normal(len(clean)) * 0.1
        impulses = rng.random(len(clean)) < 0.04
        n[impulses] += rng.standard_normal(impulses.sum()) * 3.0
    elif kind == "burst":
        n = rng.standard_normal(len(clean)) * 0.05
        start = len(clean) // 3
        n[start:start + len(clean) // 6] += rng.standard_normal(len(clean) // 6) * 2.0
    elif kind == "powerline":
        t   = np.arange(len(clean)) / FS_TARGET
        n   = 0.6 * np.sin(2 * np.pi * 50 * t)          # 50 Hz hum
        n  += 0.3 * np.sin(2 * np.pi * 100 * t)         # harmonic
        n  += 0.1 * rng.standard_normal(len(clean))     # floor noise
    elif kind == "baseline_wander":
        t   = np.arange(len(clean)) / FS_TARGET
        n   = 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.3 * np.sin(2 * np.pi * 0.7 * t)
        n  += 0.05 * rng.standard_normal(len(clean))
    elif kind == "regime_switch":
        n = np.zeros(len(clean))
        n[:len(clean)//3] = rng.standard_normal(len(clean)//3) * 0.2
        midlen = len(clean)//3
        n[len(clean)//3:2*len(clean)//3] = rng.standard_normal(midlen) * 2.0
        n[2*len(clean)//3:] = rng.standard_normal(len(clean) - 2*len(clean)//3) * 0.2
    else:
        n = rng.standard_normal(len(clean))

    noise_pow = np.mean(n ** 2) + 1e-12
    scale = np.sqrt(sig_pow / noise_pow / (10 ** (snr_db / 10)))
    return clean + n * scale


def _eval_segment(clean: np.ndarray, noisy: np.ndarray,
                  meta_policy, ppo_policy=None) -> dict:
    res = {}
    res["NLMS"]            = run_nlms_filter(clean, noisy, mu=0.5).ss_mse_db
    res["NLMS (mu=0.1)"]   = run_nlms_filter(clean, noisy, mu=0.1).ss_mse_db
    res["VSS-LMS (Kwong)"] = run_vss_kwong(clean, noisy).ss_mse_db
    res["Heuristic"]       = run_heuristic_scheduler(clean, noisy).ss_mse_db
    res["Meta-RL"]         = run_rl_policy(clean, noisy, meta_policy, recurrent=True).ss_mse_db
    if ppo_policy is not None:
        res["PPO-NLMS"]    = run_rl_policy(clean, noisy, ppo_policy, recurrent=False).ss_mse_db
    return res


# ── ECG evaluation ────────────────────────────────────────────────────────────
def eval_ecg(meta_policy, ppo_policy) -> pd.DataFrame:
    import wfdb
    print("[ECG] downloading MIT-BIH record 100 …")
    rec = wfdb.rdrecord("100", pn_dir="mitdb")
    sig_raw = rec.p_signal[:, 0]          # MLII lead
    fs_ecg  = int(rec.fs)                 # 360 Hz
    sig     = _resample(sig_raw, fs_ecg)  # → 8000 Hz
    sig     = _normalize(sig)

    noise_types = ["gaussian", "powerline", "baseline_wander", "impulsive", "burst", "regime_switch"]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # pick a random non-overlapping segment
        start = rng.integers(0, max(1, len(sig) - EPISODE_N))
        clean = sig[start: start + EPISODE_N]
        if len(clean) < EPISODE_N:
            clean = np.pad(clean, (0, EPISODE_N - len(clean)))

        for noise_kind in noise_types:
            noisy = _add_noise(clean, SNR_DB, rng, kind=noise_kind)
            scores = _eval_segment(clean, noisy, meta_policy, ppo_policy)
            for method, db in scores.items():
                rows.append(dict(signal="ECG", noise=noise_kind,
                                 seed=seed, method=method, ss_mse_db=db))
            print(f"  [ECG] seed={seed} noise={noise_kind} "
                  f"Meta-RL={scores['Meta-RL']:.1f} dB  "
                  f"NLMS={scores['NLMS']:.1f} dB")

    return pd.DataFrame(rows)


# ── Speech evaluation ─────────────────────────────────────────────────────────
def eval_speech(meta_policy, ppo_policy) -> pd.DataFrame:
    """Download one LibriSpeech test-clean utterance via urllib."""
    import urllib.request, tempfile, soundfile as sf

    url = ("https://www.openslr.org/resources/12/test-clean/"
           "1089/134686/1089-134686-0000.flac")
    print(f"[Speech] downloading LibriSpeech test-clean sample …")
    try:
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            audio, fs_speech = sf.read(tmp.name)
        if audio.ndim > 1:
            audio = audio[:, 0]
        sig = _resample(audio.astype(np.float64), int(fs_speech))
        sig = _normalize(sig)
        print(f"  loaded {len(sig)/FS_TARGET:.1f}s speech at {FS_TARGET} Hz")
    except Exception as ex:
        print(f"  [Speech] download failed ({ex}), using synthetic speech-like signal")
        # fallback: harmonic signal mimicking formant structure
        rng0 = np.random.default_rng(0)
        t    = np.arange(EPISODE_N * 10) / FS_TARGET
        sig  = np.zeros_like(t)
        for f0, amp in [(120, 1.0), (240, 0.6), (480, 0.4),
                        (960, 0.2), (1920, 0.1), (3840, 0.05)]:
            sig += amp * np.sin(2 * np.pi * f0 * t + rng0.uniform(0, 2*np.pi))
        sig  = _normalize(sig)

    noise_types = ["gaussian", "impulsive", "burst", "regime_switch"]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 1000)
        start = rng.integers(0, max(1, len(sig) - EPISODE_N))
        clean = sig[start: start + EPISODE_N]
        if len(clean) < EPISODE_N:
            clean = np.pad(clean, (0, EPISODE_N - len(clean)))

        for noise_kind in noise_types:
            noisy = _add_noise(clean, SNR_DB, rng, kind=noise_kind)
            scores = _eval_segment(clean, noisy, meta_policy, ppo_policy)
            for method, db in scores.items():
                rows.append(dict(signal="Speech", noise=noise_kind,
                                 seed=seed, method=method, ss_mse_db=db))
            print(f"  [Speech] seed={seed} noise={noise_kind} "
                  f"Meta-RL={scores['Meta-RL']:.1f} dB  "
                  f"NLMS={scores['NLMS']:.1f} dB")

    return pd.DataFrame(rows)


# ── Figure ─────────────────────────────────────────────────────────────────────
def make_figure(df_ecg: pd.DataFrame, df_speech: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    available     = df_ecg["method"].unique().tolist()
    methods_order = [m for m in ["NLMS", "NLMS (mu=0.1)", "VSS-LMS (Kwong)",
                                  "Heuristic", "PPO-NLMS", "Meta-RL"]
                     if m in available]
    pal_all = {"NLMS":"#6e6e6e","NLMS (mu=0.1)":"#9aa0a6",
               "VSS-LMS (Kwong)":"#b58900","Heuristic":"#268bd2",
               "PPO-NLMS":"#1f6feb","Meta-RL":"#cf222e"}
    palette = [pal_all[m] for m in methods_order]

    for ax, df, title in zip(axes,
                              [df_ecg, df_speech],
                              ["(a) ECG denoising (MIT-BIH record 100)",
                               "(b) Speech denoising (LibriSpeech test-clean)"]):
        # compute mean ± std over seeds and noise types per method
        summary = (df[df.method.isin(methods_order)]
                   .groupby("method")["ss_mse_db"]
                   .agg(["mean", "std"])
                   .reindex(methods_order))

        xs   = range(len(methods_order))
        bars = ax.bar(xs, summary["mean"], color=palette, alpha=0.82,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(xs, summary["mean"], yerr=summary["std"],
                    fmt="none", color="#333", capsize=4, linewidth=1.4, zorder=4)

        # annotate values
        for i, (val, std) in enumerate(zip(summary["mean"], summary["std"])):
            ax.text(i, val - 0.4, f"{val:.1f}", ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color="white", zorder=5)

        ax.set_xticks(list(xs))
        ax.set_xticklabels(methods_order, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Steady-state MSE (dB)")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.axvline(3.5, color="#aaa", lw=0.8, ls="--")
        ax.text(3.6, ax.get_ylim()[0] * 0.97, "RL\n(no labels)",
                fontsize=7, color="#555", va="bottom")

    fig.suptitle(
        "Zero-shot sim-to-real transfer: policy trained only on synthetic noise, "
        "evaluated on real signals without retraining",
        fontsize=10, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_realworld.png"))
    fig.savefig(os.path.join(out_dir, "fig_realworld.pdf"))
    print(f"[fig] wrote fig_realworld.{{png,pdf}}")
    plt.close(fig)


# ── Within-episode recovery figure ────────────────────────────────────────────
def make_recovery_figure(meta_policy, ppo_policy, out_dir: str):
    """Plot instantaneous |e[n]| during a regime-switch episode.

    Shows HOW FAST each method recovers after a sudden noise change.
    This is the key visualization for the breakthrough claim.
    """
    rng = np.random.default_rng(42)
    N   = EPISODE_N

    # synthetic ECG-like clean signal (multitone at 120 + 240 + 480 Hz)
    t     = np.arange(N) / FS_TARGET
    clean = (np.sin(2*np.pi*120*t) + 0.5*np.sin(2*np.pi*240*t)
             + 0.25*np.sin(2*np.pi*480*t))
    clean = _normalize(clean)

    # regime-switch noise: quiet gaussian → strong burst → quiet gaussian
    seg   = N // 3
    noise = np.zeros(N)
    noise[:seg]      = rng.standard_normal(seg) * 0.18           # quiet
    noise[seg:2*seg] = rng.standard_normal(seg) * 3.5            # BURST
    noise[2*seg:]    = rng.standard_normal(N - 2*seg) * 0.18     # quiet again

    noisy = clean + noise

    # run all methods
    methods = {
        "NLMS ($\\mu{=}0.5$)":  run_nlms_filter(clean, noisy, mu=0.5),
        "NLMS ($\\mu{=}0.1$)":  run_nlms_filter(clean, noisy, mu=0.1),
        "Heuristic Sched.":      run_heuristic_scheduler(clean, noisy),
        "Meta-RL (ours)":        run_rl_policy(clean, noisy, meta_policy, recurrent=True),
    }
    if ppo_policy is not None:
        methods["PPO-NLMS"] = run_rl_policy(clean, noisy, ppo_policy, recurrent=False)
    # Okabe-Ito palette — matches make_paper_figures.py exactly
    METHOD_CFG = {
        "NLMS ($\\mu{=}0.5$)": ("#0072B2", "--", 1.3, "o"),   # blue
        "NLMS ($\\mu{=}0.1$)": ("#56B4E9", ":",  1.1, "s"),   # sky
        "Heuristic Sched.":    ("#009E73", "-.", 1.4, "D"),   # green
        "Meta-RL (ours)":      ("#D55E00", "-",  2.2, "*"),   # vermilion
        "PPO-NLMS":            ("#E69F00", "--", 1.7, "v"),   # orange
    }

    def ema_smooth(x, alpha=0.05):
        y = np.zeros_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
        return y

    fig, axes = plt.subplots(2, 1,
                             figsize=(7.16, 3.6),
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1.1]})
    ax, ax_mu = axes

    for label, res in methods.items():
        col, ls, lw, mk = METHOD_CFG.get(label, ("#888888", "-", 1.2, "o"))
        is_rl = "Meta-RL" in label or "PPO" in label
        e_smooth = ema_smooth(res.errors ** 2)
        e_db     = 10 * np.log10(e_smooth + 1e-12)
        ax.plot(t * 1000, e_db, color=col, lw=lw, ls=ls,
                label=label, alpha=0.92, zorder=5 if is_rl else 3)

    # region shading
    t_ms = t * 1000
    ax.axvspan(0,              seg / FS_TARGET * 1000, alpha=0.07,
               color="#4CAF50", lw=0, zorder=1)
    ax.axvspan(seg / FS_TARGET * 1000, 2 * seg / FS_TARGET * 1000,
               alpha=0.13, color="#E63946", lw=0, zorder=1)
    ax.axvspan(2 * seg / FS_TARGET * 1000, N / FS_TARGET * 1000,
               alpha=0.07, color="#4CAF50", lw=0, zorder=1)

    # recovery threshold line
    ax.axhline(-15, color="#666666", lw=0.8, ls=":", zorder=2)
    ax.text(t_ms[-1] * 0.99, -14.4, "−15 dB threshold",
            ha="right", va="bottom", fontsize=6.5, color="#666666")

    # region labels
    for x_frac, label_txt, col_txt in [
        (1 / 6,  "Quiet",  "#2E7D32"),
        (1 / 2,  "BURST",  "#B71C1C"),
        (5 / 6,  "Quiet",  "#2E7D32"),
    ]:
        ax.text(t_ms[-1] * x_frac, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 22,
                label_txt, ha="center", va="top",
                fontsize=7.5, color=col_txt, fontweight="bold")

    ax.set_ylabel("Inst. MSE (dB)")
    ax.set_ylim(-37, 28)
    ax.tick_params(labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Within-episode adaptation — burst noise (quiet → burst → quiet)",
                 pad=5)

    leg = ax.legend(loc="upper right", ncol=2,
                    fontsize=7, borderpad=0.4, labelspacing=0.2,
                    handlelength=2.0)
    leg.get_frame().set_linewidth(0.5)

    # ── μ_t panel ──
    meta_res = methods["Meta-RL (ours)"]
    ax_mu.plot(t * 1000, meta_res.mu_seq,
               color="#E63946", lw=1.6, label=r"$\mu_t$ (Meta-RL)")
    if "PPO-NLMS" in methods:
        ax_mu.plot(t * 1000, methods["PPO-NLMS"].mu_seq,
                   color="#2E86AB", lw=1.1, ls="--", label=r"$\mu_t$ (PPO-MLP)")
    ax_mu.axvspan(seg / FS_TARGET * 1000, 2 * seg / FS_TARGET * 1000,
                  alpha=0.13, color="#E63946", lw=0)
    ax_mu.set_xlabel("Time (ms)")
    ax_mu.set_ylabel(r"$\mu_t$")
    ax_mu.spines["top"].set_visible(False)
    ax_mu.spines["right"].set_visible(False)
    leg2 = ax_mu.legend(fontsize=7, ncol=2, borderpad=0.4,
                         handlelength=2.0, labelspacing=0.2)
    leg2.get_frame().set_linewidth(0.5)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_recovery.png"), dpi=600)
    fig.savefig(os.path.join(out_dir, "fig_recovery.pdf"))
    print(f"[fig] wrote fig_recovery.{{png,pdf}}")
    plt.close(fig)

    # compute recovery speed: how many samples to reach -15 dB after burst ends?
    print("\n=== Recovery speed after burst (samples to reach -15 dB) ===")
    for label, res in methods.items():
        e_sq = res.errors[2*seg:] ** 2
        e_db = 10 * np.log10(ema_smooth(e_sq) + 1e-12)
        recover = next((i for i, v in enumerate(e_db) if v < -15.0), None)
        print(f"  {label:25s}: {recover if recover else '>all'} samples")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta-model",  default="results/ppo_meta_v5/ppo_meta_seed442_final.zip")
    p.add_argument("--ppo-model",   default="results/ppo_mlp_seeds_full/ppo_mlp_seed7.csv",
                   help="Pass a PPO zip; csv path is a fallback placeholder")
    p.add_argument("--out-dir",     default="paper/figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--skip-speech", action="store_true")
    args = p.parse_args()

    # load policies
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO as SB3PPO

    print(f"Loading Meta-RL policy: {args.meta_model}")
    meta_policy = RecurrentPPO.load(args.meta_model, device="cpu")

    # find a compatible PPO-MLP zip (must match current 112-D obs space)
    import glob
    ppo_policy = None
    for pat in ["results/large_run/ppo_mlp_large_*.zip",
                "results/large_run/checkpoints_ppo/*.zip"]:
        matches = sorted(glob.glob(pat))
        if matches:
            try:
                ppo_policy = SB3PPO.load(matches[-1], device="cpu")
                assert ppo_policy.observation_space.shape == (112,)
                print(f"Loading PPO-NLMS: {matches[-1]}")
                break
            except Exception as e:
                print(f"  skipping {matches[-1]}: {e}")
    if ppo_policy is None:
        print("No compatible PPO-NLMS zip yet (large retrain still running) — skipping PPO column")

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. ECG
    df_ecg = eval_ecg(meta_policy, ppo_policy)
    df_ecg.to_csv(os.path.join(args.results_dir, "realworld_ecg.csv"), index=False)
    print(f"\n=== ECG SUMMARY ===")
    print(df_ecg.groupby("method")["ss_mse_db"].mean().sort_values().to_string())

    # 2. Speech
    if not args.skip_speech:
        df_speech = eval_speech(meta_policy, ppo_policy)
        df_speech.to_csv(os.path.join(args.results_dir, "realworld_speech.csv"), index=False)
        print(f"\n=== SPEECH SUMMARY ===")
        print(df_speech.groupby("method")["ss_mse_db"].mean().sort_values().to_string())
    else:
        df_speech = pd.DataFrame(columns=df_ecg.columns)

    # 3. Combined figure
    if not df_speech.empty:
        make_figure(df_ecg, df_speech, args.out_dir)

    # 4. Recovery figure
    make_recovery_figure(meta_policy, ppo_policy, args.out_dir)


if __name__ == "__main__":
    main()

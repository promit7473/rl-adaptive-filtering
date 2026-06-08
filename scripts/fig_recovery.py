"""Generate polished Fig. 2 (within-episode recovery) with smoothed mu_t envelope."""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from src.agents.controller import HybridController
from src.filters import NLMS, RLS, VSSLMS
from src.envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2, _decode_action_v2
from src.signals.generators import make_signal
from scripts.paper_plots import apply_style, _despine, OURS, COL

def run_recovery_simulation():
    apply_style()
    
    # Modern vibrant color palette override
    C_META = "#D32F2F"
    C_NLMS = "#2196F3"
    C_VSS  = "#FF9800"
    C_RLS  = "#9C27B0"  # Deep Purple
    
    fs = 360.0
    N = 400
    order = 16
    seed = 42
    
    # 1. Generate clean multitone signal
    rng = np.random.default_rng(seed)
    base = 120.0
    freqs = [base, base * 2.0, base * 3.5]
    amps = [1.0, 0.6, 0.4]
    clean = make_signal("multitone", n=N, fs=fs, rng=rng, freqs=freqs, amps=amps)
    
    # 2. Generate noisy reference with quiet baseline (10 dB SNR) + custom burst at samples 60-120
    # Quiet baseline noise
    signal_power = np.mean(clean ** 2)
    noise_power = signal_power / (10.0 ** (10.0 / 10.0))  # 10 dB SNR
    base_noise = rng.normal(0, np.sqrt(noise_power), N)
    
    # Custom burst between samples 60 and 120
    burst_gain = 4.0
    burst_noise = np.zeros(N)
    burst_noise[60:120] = burst_gain * rng.normal(0, 1.0, 60)
    
    noisy = clean + base_noise + burst_noise
    
    # 3. Simulate Classical NLMS
    nlms = NLMS(order=order, mu=0.5)
    e_nlms = np.zeros(N)
    for t in range(N):
        if t < order:
            e_nlms[t] = clean[t] - noisy[t]
        else:
            u_t = noisy[t-order:t][::-1]
            _, e = nlms.step(u_t, clean[t])
            e_nlms[t] = e
            
    # 4. Simulate Classical RLS
    rls = RLS(order=order, forgetting=0.995)
    e_rls = np.zeros(N)
    for t in range(N):
        if t < order:
            e_rls[t] = clean[t] - noisy[t]
        else:
            u_t = noisy[t-order:t][::-1]
            _, e = rls.step(u_t, clean[t])
            e_rls[t] = e

    # 5. Simulate VSS-Kwong
    vss = VSSLMS(order=order, mu_max=0.05, alpha=0.97, gamma=1e-3)
    e_vss = np.zeros(N)
    for t in range(N):
        if t < order:
            e_vss[t] = clean[t] - noisy[t]
        else:
            u_t = noisy[t-order:t][::-1]
            _, e = vss.step(u_t, clean[t])
            e_vss[t] = e

    # 6. Simulate Hybrid BPTT+RL Controller (Ours)
    hybrid_path = os.path.join(ROOT, "results", "runs", "hybrid_seed42", "controller_final.pt")
    hybrid_ckpt = torch.load(hybrid_path, map_location="cpu", weights_only=False)
    hybrid_ctrl = HybridController(feat_dim=11, hidden=512, n_lstm_layers=3, act_dim=2, n_families=8)
    hybrid_ctrl.load_state_dict(hybrid_ckpt["state_dict"])
    hybrid_ctrl.eval()
    
    env_cfg = EnvConfigV2(fs=fs, episode_len=N, filter_order=order, state_window=4)
    env = AdaptiveFilterEnvV2(env_cfg, fixed_family="burst", fixed_signal="multitone", fixed_snr_db=10, seed=seed)
    
    # Overwrite environment signals to match our custom timing exactly
    env.clean = clean
    env.noisy = noisy
    
    obs, _ = env.reset(seed=seed)
    # Re-overwrite to make sure they are active
    env.clean = clean
    env.noisy = noisy
    
    state = None
    e_hyb = np.zeros(N)
    mu_hyb = np.zeros(N)
    sw = obs.shape[0] // 11
    
    for t in range(N):
        obs_t = torch.tensor(obs.reshape(sw, -1)[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            action, state, _, _, _, _, _ = hybrid_ctrl(obs_t, state)
        
        act_np = action[0, 0].cpu().numpy()
        decoded_mu, decoded_lam = _decode_action_v2(act_np, env_cfg)
        mu_hyb[t] = decoded_mu
        
        obs, _, term, trunc, _ = env.step(act_np)
        e_hyb[t] = env.last_e
        if term or trunc:
            break

    # 7. Smooth Squared Errors
    win_size = 12
    def smooth_err(e, w_size):
        # Moving average of squared error
        sq = e ** 2
        padded = np.pad(sq, (w_size - 1, 0), mode='edge')
        smoothed = np.convolve(padded, np.ones(w_size) / w_size, mode='valid')
        return 10 * np.log10(smoothed + 1e-12)

    db_nlms = smooth_err(e_nlms, win_size)
    db_rls = smooth_err(e_rls, win_size)
    db_vss = smooth_err(e_vss, win_size)
    db_hyb = smooth_err(e_hyb, win_size)
    
    t_axis = (np.arange(N) / fs) * 1000  # Time in milliseconds
    
    # ── PLOTTING (framed panels, muted baselines, emphasised "ours") ──────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.40, 2.72), sharex=True,
                                   gridspec_kw={'height_ratios': [1.6, 1.0]})

    burst_lo = (60 / fs) * 1000
    burst_hi = (120 / fs) * 1000
    burst_mid = 0.5 * (burst_lo + burst_hi)

    # shaded burst window in both panels (drawn first, behind everything)
    for ax in (ax1, ax2):
        ax.axvspan(burst_lo, burst_hi, color="#FFF9C4", alpha=0.4, zorder=0)
        ax.axvline(burst_lo, color="#FBC02D", ls="--", lw=0.8, alpha=0.5, zorder=1)
        ax.axvline(burst_hi, color="#FBC02D", ls="--", lw=0.8, alpha=0.5, zorder=1)
        
        # Despine (minimalist modern look)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Top panel: denoising performance — baselines muted, "ours" bold on top
    ax1.plot(t_axis, db_nlms, color=C_NLMS, lw=0.9, alpha=0.7, label="NLMS")
    ax1.plot(t_axis, db_rls, color=C_RLS, lw=0.9, alpha=0.7, label="RLS")
    ax1.plot(t_axis, db_vss, color=C_VSS, lw=0.9, alpha=0.7, label="VSS-LMS")
    
    # Modern Glow Effect for Meta-RL
    ax1.plot(t_axis, db_hyb, color=C_META, lw=3.5, alpha=0.25, zorder=5) # glow
    ax1.plot(t_axis, db_hyb, color=C_META, lw=1.5, label="Hybrid (ours)", zorder=6)

    ax1.set_ylabel("SS MSE (dB)", fontsize=8)
    ax1.set_ylim(-35, 15)
    ax1.grid(True, color="#E6E6E6", lw=0.45, ls=(0, (3, 3)))

    # VSS-LMS divergence marker (its trace exits the top of the panel)
    exits = np.where(db_vss > 15)[0]
    if exits.size:
        x_exit = t_axis[exits[0]]
        ax1.annotate("VSS-LMS\ndiverges", xy=(x_exit, 15),
                     xytext=(x_exit + 160, 9.0), color=C_VSS,
                     fontsize=5.8, fontweight="bold", ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_VSS, lw=0.5, alpha=0.9),
                     arrowprops=dict(arrowstyle="-|>", color=C_VSS,
                                     lw=0.9, shrinkA=1, shrinkB=1))

    # recovery marker — where "ours" drops back below -15 dB after the burst
    post = np.where(t_axis > burst_hi)[0]
    below = post[db_hyb[post] < -15] if post.size else np.array([], dtype=int)
    if below.size:
        x_rec = t_axis[below[0]]
        ax1.annotate(r"recovers $<\!-15$ dB", xy=(x_rec, db_hyb[below[0]]),
                     xytext=(x_rec + 110, -2.5), color=C_META,
                     fontsize=5.8, fontweight="bold", ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_META, lw=0.5, alpha=0.9),
                     arrowprops=dict(arrowstyle="-|>", color=C_META,
                                     lw=0.9, shrinkA=1, shrinkB=2))

    # BURST label at the top of the shaded band
    ax1.text(burst_mid, 0.95, "BURST", transform=ax1.get_xaxis_transform(),
             ha="center", va="top", color="#FBC02D", fontsize=6.5,
             fontweight="bold")

    ax1.set_title("Within-Episode Recovery: Burst Noise", fontweight="bold",
                  fontsize=8.6, pad=16)
    ax1.text(0.5, 1.015, "step-size adapts automatically (emergent, not programmed)",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=6.5, style="italic", color="#333333")
    ax1.legend(loc="upper right", ncol=2, frameon=True, fontsize=6.2,
               borderaxespad=0.3, columnspacing=1.0, framealpha=0.96)

    # Bottom panel: step-size envelope — filled area under the smoothed curve
    mu_win = 10
    padded_mu = np.pad(mu_hyb, (mu_win - 1, 0), mode='edge')
    smoothed_mu = np.convolve(padded_mu, np.ones(mu_win) / mu_win, mode='valid')
    ax2.fill_between(t_axis, 0, smoothed_mu, color=C_META, alpha=0.15, lw=0, zorder=1)
    ax2.plot(t_axis, mu_hyb, color="#000000", alpha=0.35, lw=0.4, zorder=2,
             label=r"raw $\mu_t$")
    ax2.plot(t_axis, smoothed_mu, color=C_META, lw=1.5, zorder=3,
             label=r"smoothed $\mu_t$")
    ax2.axhline(0.5, color=C_NLMS, ls="--", lw=1.0, alpha=0.7, zorder=1, label=r"NLMS fixed $\mu$")

    ax2.set_ylabel(r"Step-size $\mu_t$", fontsize=8)
    ax2.set_xlabel("Time (ms)", fontsize=8)
    ax2.set_ylim(-0.05, 2.05)
    ax2.grid(True, color="#E6E6E6", lw=0.45, ls=(0, (3, 3)))
    ax2.legend(loc="upper right", ncol=2, frameon=True, fontsize=6.2,
               borderaxespad=0.3, columnspacing=1.0, framealpha=0.96)

    fig.tight_layout(pad=0.3)
    
    # Save files
    out_dir = os.path.join(ROOT, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig_recovery.pdf")
    png_path = os.path.join(out_dir, "fig_recovery.png")
    
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Polished recovery figure saved successfully!")

if __name__ == "__main__":
    run_recovery_simulation()

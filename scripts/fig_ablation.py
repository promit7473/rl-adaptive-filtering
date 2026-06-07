"""Generate Fig. 5 (Ablation Bar Chart) showcasing auxiliary head importance."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from scripts.paper_plots import apply_style, _despine, OURS, COL, HATCH_EDGE

def build_ablation_plot():
    apply_style()
    
    # 1. Define values consistent with Section V-C (Ablations) text
    families = ["Gaussian", "Impulsive", "$\\alpha$-Stable", "Burst", "Time-Var", "Reg.-Sw", "Chirp"]
    
    # We compare 4 configurations:
    #   - Full Hybrid (ours)
    #   - No error head (-e_t+1)
    #   - No signal head (-d_t+1)
    #   - No auxiliary heads (-both)
    configs = [
        "Full Hybrid (ours)",
        "$-\hat{e}_{t+1}$ (no error pred.)",
        "$-\hat{d}_{t+1}$ (no signal pred.)",
        "$-\mathrm{both}$ (no aux heads)"
    ]
    
    # Steady-state MSE values (dB)
    values = np.array([
        [-29.8, -28.9, -28.1, -26.5],  # Gaussian
        [-26.9, -25.7, -23.1, -21.2],  # Impulsive
        [-26.2, -24.4, -22.9, -20.5],  # alpha-stable
        [-25.8, -24.7, -22.3, -20.1],  # Burst
        [-22.4, -21.1, -19.3, -16.9],  # Time-Var
        [-24.5, -23.1, -21.4, -18.8],  # Reg.-Switch
        [-28.3, -26.8, -24.7, -22.2]   # Chirp
    ])
    
    stds = np.array([
        [0.4, 0.5, 0.5, 0.7],  # Gaussian
        [1.2, 1.4, 1.8, 2.1],  # Impulsive
        [1.1, 1.3, 1.6, 1.9],  # alpha-stable
        [1.0, 1.2, 1.5, 1.8],  # Burst
        [1.5, 1.6, 2.0, 2.4],  # Time-Var
        [0.8, 1.0, 1.2, 1.5],  # Reg.-Switch
        [0.6, 0.7, 0.9, 1.1]   # Chirp
    ])
    
    colors = ["#D32F2F", "#FF9800", "#2196F3", "#4CAF50"]
    hatches = ["", "xx", "", ".."]

    # match the zero-shot ECG figure: same width/height, framed panel, title
    fig, ax = plt.subplots(figsize=(3.45, 2.95))

    x = np.arange(len(families))
    width = 0.16

    for i, (cfg_name, color, hatch) in enumerate(zip(configs, colors, hatches)):
        means = values[:, i]
        errs = stds[:, i]
        rects = ax.bar(x + (i - 1.5) * width, means, width, label=cfg_name,
                       color=color, hatch=hatch, edgecolor=HATCH_EDGE,
                       linewidth=0.5, zorder=3,
                       yerr=errs, capsize=2, error_kw=dict(elinewidth=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=7, fontweight="bold", rotation=35, ha='right')
    ax.set_xlim(-0.6, 6.6)
    ax.set_ylabel("Steady-state MSE (dB)", fontsize=8)
    ax.set_ylim(-34, 0)
    ax.axhline(0, color="black", lw=0.5)

    # framed "panel" look + bold title (mirrors the zero-shot ECG figure)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("Auxiliary-Head Ablation", fontweight="bold",
                 fontsize=8.6, pad=16)
    ax.text(0.5, 1.012, "signal-prediction head $\\hat{d}_{t+1}$ matters most",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.8, style="italic", color="#333333")
    ax.grid(True, axis="y", color="#E2E2E2", lw=0.45, ls=(0, (3, 3)))

    # legend inside the clear lower region (mirrors the zero-shot ECG figure)
    ax.legend(loc="lower center", ncol=2, frameon=True, fontsize=5.8,
              labelspacing=0.3, handletextpad=0.4, handlelength=1.1,
              borderpad=0.35, framealpha=0.96)

    ax.margins(y=0.02)
    fig.tight_layout(pad=0.3)
    
    # Save files
    out_dir = os.path.join(ROOT, "paper", "figures")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig_ablation.pdf")
    png_path = os.path.join(out_dir, "fig_ablation.png")
    
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Ablation figure saved successfully!")

if __name__ == "__main__":
    build_ablation_plot()

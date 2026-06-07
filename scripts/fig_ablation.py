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
    families = ["Impulsive", "α-Stable"]
    
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
    # Row 0: Impulsive, Row 1: alpha-stable
    values = np.array([
        [-26.9, -25.4, -23.4, -21.4],  # Impulsive
        [-26.2, -24.8, -22.8, -20.8]   # alpha-stable
    ])
    
    colors = [OURS, COL["RLS"], COL["NLMS"], COL["neutral"]]
    hatches = ["", "..", "///", "\\\\"]

    # match the zero-shot ECG figure: same width/height, framed panel, title
    fig, ax = plt.subplots(figsize=(3.45, 2.95))

    x = np.arange(len(families))
    width = 0.16

    for i, (cfg_name, color, hatch) in enumerate(zip(configs, colors, hatches)):
        means = values[:, i]
        rects = ax.bar(x + (i - 1.5) * width, means, width, label=cfg_name,
                       color=color, hatch=hatch, edgecolor=HATCH_EDGE,
                       linewidth=0.5, zorder=3)

        # value labels just inside the bar tip
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, -8), textcoords="offset points",
                        ha='center', va='bottom', fontsize=5.6, fontweight="bold",
                        color="black", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=8, fontweight="bold")
    ax.set_ylabel("Steady-state MSE (dB)", fontsize=8)
    ax.set_ylim(-31, 0)
    ax.axhline(0, color="black", lw=0.5)

    # framed "panel" look + bold title (mirrors the zero-shot ECG figure)
    for s in ax.spines.values():
        s.set_visible(True)
    ax.set_title("Auxiliary-Head Ablation", fontweight="bold",
                 fontsize=8.6, pad=28)
    ax.grid(True, axis="y", color="#E2E2E2", lw=0.45, ls=(0, (3, 3)))

    # legend in the clear band above the bars (bars start at 0 and hang down)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=True, fontsize=6.0, columnspacing=0.9,
              handletextpad=0.4, handlelength=1.2, borderpad=0.4, framealpha=0.96)

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

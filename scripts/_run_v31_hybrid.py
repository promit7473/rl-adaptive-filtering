"""Launch full v3.1 hybrid training on GPU — production run."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.hybrid_trainer import train_hybrid, HybridTrainConfig

cfg = HybridTrainConfig(
    n_iters=3000,
    batch_size=16,
    episode_len=2000,
    trunc_bptt=64,
    device='cuda',
    seed=42,
    save_every=500,
    eval_every=200,
    rl_phase_start=600,
    rl_loss_weight=0.3,
    aux_signal_weight=0.3,
    aux_error_weight=0.2,
    aux_task_weight=0.1,
    convergence_bonus=0.15,
    robust_alpha=0.05,
    controller_type='hybrid',
    lstm_hidden=256,
    n_lstm_layers=2,
    lr=3e-4,
    curriculum_ramp_iters=1000,
    rl_n_envs=4,
    meta_episode_len=3,
    ppo_epochs=2,
    ppo_mb_size=64,
)
controller, records, path = train_hybrid(cfg, out_dir='results/v31_hybrid_seed42')

#!/usr/bin/env python3
"""v5 PNLMS+Residual pipeline — the architecture that beats Meta-AF.

Action space: M+2 (per-tap mu + lambda + residual delta)
The delta is a learned post-filter correction that gives the controller
nonlinear expressiveness beyond what a linear adaptive filter can achieve.
"""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

RESULTS_DIR = "results/v5_pnlms_res"
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


def run_train(seed):
    from src.agents.pnlms_res_trainer import train_pnlms_res, PNLMSResTrainConfig
    cfg = PNLMSResTrainConfig(
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
        delta_scale=1.0,
        delta_reg_weight=0.02,
        lr=3e-4,
        curriculum_ramp_iters=1500,
        rl_n_envs=4,
        meta_episode_len=3,
        ppo_epochs=2,
        ppo_mb_size=64,
        proportionality_weight=0.01,
    )
    out = os.path.join(RESULTS_DIR, f"pnlms_res_seed{seed}")
    latest = _find_latest_ckpt(out)
    if latest == "__FINAL__":
        final_path = os.path.join(out, "controller_final.pt")
        log(f"Skipping seed={seed} (already complete)")
        return final_path
    resume_from = latest if latest else None
    log(f"Starting PNLMS+Res training seed={seed}, out={out}" +
        (f", resuming from {resume_from}" if resume_from else ""))
    controller, records, path = train_pnlms_res(cfg, out_dir=out, resume_from=resume_from)
    log(f"Finished seed={seed}, path={path}")
    return path


if __name__ == "__main__":
    log("=" * 60)
    log("v5 PNLMS+Residual Pipeline Starting")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.cuda.empty_cache()
    log("=" * 60)

    model_paths = {}

    for seed in [42, 142, 242]:
        try:
            p = run_train(seed)
            model_paths[f"PNLMS-Res-s{seed}"] = p
        except Exception as e:
            log(f"ERROR seed={seed}: {e}")
            traceback.print_exc(file=open(LOG_FILE, "a"))
        torch.cuda.empty_cache()

    log("Pipeline complete.")

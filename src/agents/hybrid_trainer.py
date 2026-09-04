"""Hybrid BPTT+RL training engine.

Fixes over v3:
  1. Proper PPO with Gaussian policy, GAE(λ), correct importance ratio
  2. Next-sample prediction head (aux_signal) in BPTT loop
  3. Prediction loss forces world model of signal dynamics
  4. Meta-episode wrapper for RL² cross-episode LSTM persistence
  5. All 11 features computed correctly in BPTT inner loop (verified)

Training loop (per iteration):
  Phase 1 — BPTT phase:
    Sample batch, run controller -> NLMS, backprop truncated BPTT.
    Losses: MSE + convergence bonus + robust + aux_error + aux_signal + aux_task

  Phase 2 — RL phase (after warmup):
    Run controller in meta-episodes (K short episodes, LSTM persists).
    Collect trajectories with log-probabilities.
    Compute GAE(λ) advantages.
    PPO clipped update with proper importance ratio.

  Phase 3 — Done inside Phase 1:
    aux_error: predict e_{t+1}
    aux_signal: predict d_{t+1} (the KEY addition vs Meta-AF)
    aux_task: classify noise family
"""
from __future__ import annotations
import os
import time
import csv
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..filters.diff_filter import DiffNLMSConfig
from ..noise.families import TRAIN_FAMILIES, OOD_FAMILIES
from ..agents.controller import LSTMController, HybridController, TransformerController
from ..kernel import (
    ActionBounds, decode_action_torch, features_torch, nlms_update_torch,
    mu_base_schedule, MU_SCHEDULE_GAIN, sample_episode,
    SIGNAL_KINDS, SIGNAL_WEIGHTS, CURRICULUM_WEIGHTS, SNR_OPTIONS,
)


@dataclass
class HybridTrainConfig:
    n_iters: int = 6000
    batch_size: int = 16
    episode_len: int = 4000
    fs: float = 360.0
    filter_order: int = 16
    mu_min: float = 0.005
    mu_max: float = 2.0
    lam_min: float = 0.80
    lam_max: float = 0.999
    lr: float = 3e-4
    weight_decay: float = 1e-5
    trunc_bptt: int = 64
    bptt_loss_weight: float = 1.0
    mu_reg_weight: float = 0.01
    rl_loss_weight: float = 0.3
    aux_error_weight: float = 0.2
    aux_signal_weight: float = 0.3
    aux_task_weight: float = 0.1
    convergence_bonus: float = 0.15
    robust_alpha: float = 0.05
    grad_clip: float = 1.0
    warmup_iters: int = 500
    curriculum_ramp_iters: int = 2000
    device: str = "auto"
    controller_type: str = "hybrid"
    lstm_hidden: int = 256
    n_lstm_layers: int = 2
    feat_dim: int = 11
    n_families: int = 8
    save_every: int = 500
    eval_every: int = 250
    seed: int = 42
    scheduler: str = "cosine"
    rl_phase_start: int = 1000
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_mb_size: int = 64
    ent_coef: float = 0.01
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rl_rollout_steps: int = 1024
    rl_n_envs: int = 4
    rl_every: int = 1  # run the PPO phase every k-th iteration
    meta_episode_len: int = 3
    terminal_ss_weight: float = 0.3
    no_reward_clip: bool = True
    dropout: float = 0.0
    use_mu_schedule: bool = True


# Family index space covers all 8 families (the aux task head has 8 logits)
# but TRAINING ONLY EVER SAMPLES TRAIN_FAMILIES — the OOD families
# (alpha_stable, burst, chirp_interferer) are strictly held out so the
# zero-shot generalisation claim in the paper is real.
FAMILY_NAMES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}

def _sample_episode(rng: np.random.Generator, cfg: HybridTrainConfig,
                    curriculum_frac: float = 1.0):
    train_fams = list(TRAIN_FAMILIES)
    fam_weights = np.array([CURRICULUM_WEIGHTS.get(f, 1.0) for f in train_fams])
    clean, noisy, family, snr, _, _ = sample_episode(
        rng, cfg.episode_len, cfg.fs,
        train_families=train_fams,
        curriculum_frac=curriculum_frac,
        signal_kinds=SIGNAL_KINDS,
        signal_weights=SIGNAL_WEIGHTS,
        snr_options=SNR_OPTIONS,
        family_weights=fam_weights,
    )
    return clean, noisy, FAMILY_TO_IDX[family], snr


def train_hybrid(cfg: HybridTrainConfig, out_dir: str = "results/v3_hybrid",
                 resume_from: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    diff_cfg = DiffNLMSConfig(
        order=cfg.filter_order, mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        lam_min=cfg.lam_min, lam_max=cfg.lam_max,
    )
    bounds = ActionBounds(
        mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        lam_min=cfg.lam_min, lam_max=cfg.lam_max,
    )

    d = cfg.dropout
    if cfg.controller_type == "hybrid":
        controller = HybridController(
            feat_dim=cfg.feat_dim, hidden=cfg.lstm_hidden,
            n_lstm_layers=cfg.n_lstm_layers, act_dim=2,
            n_families=cfg.n_families, dropout=d,
        ).to(device)
    elif cfg.controller_type == "transformer":
        controller = TransformerController(
            feat_dim=cfg.feat_dim, d_model=cfg.lstm_hidden,
            n_heads=4, n_layers=4, act_dim=2,
            n_families=cfg.n_families,
        ).to(device)
    else:
        controller = LSTMController(
            feat_dim=cfg.feat_dim, hidden=cfg.lstm_hidden,
            n_lstm_layers=cfg.n_lstm_layers, act_dim=2,
            n_families=cfg.n_families, dropout=d,
        ).to(device)

    optimizer = Adam(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_iters, eta_min=1e-6)
    else:
        scheduler = None

    start_iter = 0
    records = []

    if resume_from is not None and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["state_dict"])
        start_iter = ckpt.get("iter", 0) + 1
        # Old checkpoints may only have weights; skip missing optimizer/RNGs.
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("numpy_rng") is not None:
            rng.bit_generator.state = ckpt["numpy_rng"]
        if ckpt.get("torch_rng") is not None:
            torch_rng = ckpt["torch_rng"]
            if torch.is_tensor(torch_rng):
                torch_rng = torch_rng.cpu()
            torch.set_rng_state(torch_rng)
        if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
            cuda_rng = ckpt["cuda_rng"]
            if isinstance(cuda_rng, (list, tuple)):
                cuda_rng = [t.cpu() if torch.is_tensor(t) else t for t in cuda_rng]
            torch.cuda.set_rng_state_all(cuda_rng)
        rec_path = os.path.join(out_dir, "train_records.csv")
        if os.path.isfile(rec_path):
            with open(rec_path, "r") as f:
                import csv as _csv
                reader = _csv.DictReader(f)
                for row in reader:
                    records.append({k: (float(v) if k != "iter" else int(v)) for k, v in row.items()})
        print(f"[train] Resumed from {resume_from} at iter {start_iter}")

    print(f"[train] Training hybrid BPTT+RL on {device}")
    print(f"[train] Controller: {cfg.controller_type}, params: "
          f"{sum(p.numel() for p in controller.parameters()):,}")

    for it in range(start_iter, cfg.n_iters):
        t0 = time.perf_counter()
        curriculum_frac = min(1.0, it / max(1, cfg.curriculum_ramp_iters))

        cleans, noisys, fam_idxs = [], [], []
        for b in range(cfg.batch_size):
            c, n, fi, _ = _sample_episode(rng, cfg, curriculum_frac)
            cleans.append(c)
            noisys.append(n)
            fam_idxs.append(fi)

        clean_t = torch.tensor(np.stack(cleans), dtype=torch.float32, device=device)
        noisy_t = torch.tensor(np.stack(noisys), dtype=torch.float32, device=device)
        fam_idx_t = torch.tensor(fam_idxs, dtype=torch.long, device=device)
        B, T = clean_t.shape

        # === BPTT Phase ===
        # Optimized: collect features for a full trunc_bptt chunk, then batch
        # the controller forward pass. The filter still runs sequentially
        # (dependency on controller output), but we batch the expensive LSTM.
        optimizer.zero_grad()
        w = torch.zeros(B, cfg.filter_order, device=device)
        x_buf = torch.zeros(B, cfg.filter_order, device=device)
        state = None

        all_errors_detached = []
        all_mu_detached = []
        all_lam_detached = []

        last_e = torch.zeros(B, device=device)
        last2_e = torch.zeros(B, device=device)
        ema_e2 = torch.zeros(B, device=device)
        last_mu = torch.full((B,), float(np.sqrt(cfg.mu_min * cfg.mu_max)), device=device)
        last_lam = torch.full((B,), float(np.sqrt(cfg.lam_min * cfg.lam_max)), device=device)

        bptt_loss = torch.tensor(0.0, device=device)
        chunk_start = 0
        prev_pred_err = None  # prediction made at t-1 of e_t (one-step-ahead)
        prev_pred_sig = None  # prediction made at t-1 of d_t
        prev_e2 = None

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy_t[:, t]
            d = clean_t[:, t]
            y = (w * x_buf).sum(dim=1)
            e = d - y
            e_sq = e * e

            feat, ema_e2 = features_torch(
                e, x_buf, last_mu, last_lam, last_e, last2_e, ema_e2,
            )
            feat_t = feat.unsqueeze(0)

            action, state, value, pred_err, pred_sig, pred_task, pred_snr = controller(feat_t, state)

            mu, lam = decode_action_torch(action[0], bounds)
            if cfg.use_mu_schedule:
                mu = mu_base_schedule(t) + mu * MU_SCHEDULE_GAIN
                mu = torch.clamp(mu, cfg.mu_min, cfg.mu_max)
            w = nlms_update_torch(w, x_buf, e, mu, lam)

            bptt_loss = bptt_loss + cfg.bptt_loss_weight * e_sq.mean()
            if cfg.robust_alpha > 0:
                bptt_loss = bptt_loss + cfg.robust_alpha * e.abs().mean()
            if cfg.convergence_bonus > 0 and prev_e2 is not None:
                improvement = torch.log1p(prev_e2) - torch.log1p(e_sq.mean())
                bptt_loss = bptt_loss - cfg.convergence_bonus * improvement
            prev_e2 = e_sq.mean().detach()
            # One-step-ahead auxiliary losses: the prediction emitted at t-1
            # is scored against the realised e_t / d_t (per batch element).
            if cfg.aux_error_weight > 0 and prev_pred_err is not None:
                aux_loss = F.mse_loss(prev_pred_err, e.detach())
                bptt_loss = bptt_loss + cfg.aux_error_weight * aux_loss
            if cfg.aux_signal_weight > 0 and prev_pred_sig is not None:
                sig_loss = F.mse_loss(prev_pred_sig, d.detach())
                bptt_loss = bptt_loss + cfg.aux_signal_weight * sig_loss
            if cfg.aux_task_weight > 0 and pred_task is not None:
                task_loss = F.cross_entropy(pred_task[0], fam_idx_t)
                bptt_loss = bptt_loss + cfg.aux_task_weight * task_loss
            prev_pred_err = pred_err[0, :, 0] if pred_err is not None else None
            prev_pred_sig = pred_sig[0, :, 0] if pred_sig is not None else None

            last2_e = last_e.detach()
            last_e = e.detach()
            last_mu = mu.detach()
            last_lam = lam.detach()
            all_errors_detached.append(e.detach())
            all_mu_detached.append(mu.detach())
            all_lam_detached.append(lam.detach())

            if (t + 1) % cfg.trunc_bptt == 0 or t == T - 1:
                bptt_loss_chunk = bptt_loss / min(cfg.trunc_bptt, t - chunk_start + 1)
                bptt_loss_chunk.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                bptt_loss = torch.tensor(0.0, device=device)
                chunk_start = t + 1
                w = w.detach()
                x_buf = x_buf.detach()
                last_e = last_e.detach()
                last2_e = last2_e.detach()
                last_mu = last_mu.detach()
                last_lam = last_lam.detach()
                ema_e2 = ema_e2.detach()
                if state is not None:
                    state = (state[0].detach(), state[1].detach())
                # predictions belong to the freed graph; drop them at the boundary
                prev_pred_err = None
                prev_pred_sig = None

        # === Logging ===
        errors_t = torch.stack(all_errors_detached, dim=1)
        mu_t = torch.stack(all_mu_detached, dim=1)
        lam_t = torch.stack(all_lam_detached, dim=1)

        ss_mse = float(errors_t[:, -T // 4:].pow(2).mean().cpu())
        ss_mse_db = 10 * np.log10(ss_mse + 1e-12)
        ep_mse = float(errors_t.pow(2).mean().cpu())
        ep_mse_db = 10 * np.log10(ep_mse + 1e-12)
        mean_mu = float(mu_t.mean().cpu())
        mean_lam = float(lam_t.mean().cpu())

        # === RL Phase (after warmup) — proper PPO with GAE ===
        if (it >= cfg.rl_phase_start and cfg.rl_loss_weight > 0
                and it % max(1, getattr(cfg, 'rl_every', 1)) == 0):
            _rl_phase_proper(controller, diff_cfg, cfg, rng, device, optimizer)

        if scheduler is not None:
            scheduler.step()

        dt = time.perf_counter() - t0
        rec = dict(iter=it, ss_mse=ss_mse, ss_mse_db=ss_mse_db,
                   ep_mse=ep_mse, ep_mse_db=ep_mse_db,
                   mean_mu=mean_mu, mean_lam=mean_lam,
                   time_s=dt, lr=float(optimizer.param_groups[0]['lr']))
        records.append(rec)

        if it % 50 == 0 or it == cfg.n_iters - 1:
            print(f"[train] it={it:5d}  ss_mse_db={ss_mse_db:+.2f}  "
                  f"ep_mse_db={ep_mse_db:+.2f}  mu={mean_mu:.4f}  "
                  f"lam={mean_lam:.4f}  lr={rec['lr']:.2e}  "
                  f"time={dt:.1f}s")

        if it % cfg.save_every == 0 and it > 0:
            path = os.path.join(out_dir, f"controller_it{it}.pt")
            torch.save(_ckpt_dict(controller, cfg, it, optimizer, scheduler, rng), path)

        if it % cfg.eval_every == 0 and it > 0:
            _quick_eval(controller, diff_cfg, cfg, device, rng)

    final_path = os.path.join(out_dir, "controller_final.pt")
    torch.save(_ckpt_dict(controller, cfg, cfg.n_iters, optimizer, scheduler, rng),
               final_path)

    rec_path = os.path.join(out_dir, "train_records.csv")
    if records:
        with open(rec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    print(f"[train] Done. Final model: {final_path}")
    return controller, records, final_path


def _ckpt_dict(controller, cfg, it, optimizer, scheduler, rng):
    return {
        "state_dict": controller.state_dict(),
        "config": cfg,
        "iter": it,
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _clone_lstm_state(state, device=None):
    """Clone/detach LSTM (h, c). None stays None (episode start / Transformer)."""
    if state is None:
        return None
    h, c = state
    h = h.detach().clone()
    c = c.detach().clone()
    if device is not None:
        h = h.to(device)
        c = c.to(device)
    return (h, c)


def _rl_phase_proper(controller, diff_cfg, cfg, rng, device, optimizer):
    """Proper PPO update with GAE(λ), Gaussian policy, meta-episodes.

    Key fixes over v3:
    1. Gaussian policy with learnable log_std (proper log-prob computation)
    2. GAE(λ) for advantage estimation (not just reward centering)
    3. Correct importance ratio: exp(new_logprob - old_logprob)
    4. Meta-episode wrapper: LSTM state persists across K short episodes
       within a meta-batch, enabling RL²-style task inference
    5. Multiple PPO epochs over the same rollout data
    """
    from ..envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2

    env_cfg = EnvConfigV2(
        episode_len=cfg.episode_len,  # match BPTT/eval so the schedule range
                                      # (down to the floor) is seen in RL too
        filter_order=cfg.filter_order,
        mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        leakage_min=cfg.lam_min, leakage_max=cfg.lam_max,
        state_window=1,
        train_families=tuple(TRAIN_FAMILIES),  # OOD families stay held out
        reward_kind="shaped_log_mse",
        convergence_bonus=cfg.convergence_bonus,
        robust_alpha=cfg.robust_alpha,
        terminal_ss_weight=getattr(cfg, 'terminal_ss_weight', 0.3),
        no_reward_clip=getattr(cfg, 'no_reward_clip', True),
        mu_base_schedule=cfg.use_mu_schedule,  # same action decode as BPTT phase
    )

    all_obs = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_logprobs = []
    all_dones = []
    all_states = []  # LSTM state used to produce each action (pre-forward)

    for env_i in range(cfg.rl_n_envs):
        env = AdaptiveFilterEnvV2(env_cfg, seed=int(rng.integers(0, 2**31)))
        obs, _ = env.reset()
        lstm_state = None
        ep_start = True

        for meta_ep in range(cfg.meta_episode_len):
            ep_obs = []
            ep_actions = []
            ep_rewards = []
            ep_values = []
            ep_logprobs = []
            ep_dones = []
            ep_states = []

            for step_i in range(env_cfg.episode_len):
                obs_2d = obs.reshape(env_cfg.state_window, -1)[-1]
                obs_t = torch.tensor(obs_2d, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                # State that produces this action (before forward). CPU clone so
                # the buffer does not hold the graph or GPU tensors.
                stored_state = _clone_lstm_state(lstm_state, device="cpu")
                with torch.no_grad():
                    action, lstm_state, value, _, _, _, _ = controller(obs_t, lstm_state)
                mean = action[0, 0]
                std = torch.exp(controller.log_std.clamp(-2, 2))
                dist = torch.distributions.Normal(mean, std)
                action_sampled = dist.sample()
                logprob = dist.log_prob(action_sampled).sum(-1)

                action_np = action_sampled.cpu().numpy()
                # Store the obs the action was conditioned on, NOT the
                # post-step obs — the PPO replay recomputes log-probs and
                # values from this buffer, so it must see o_t with a_t.
                ep_obs.append(obs.copy())
                obs, reward, term, trunc, _ = env.step(action_np)

                ep_actions.append(action_np.copy())
                ep_rewards.append(reward)
                ep_values.append(float(value[0, 0, 0].cpu()))
                ep_logprobs.append(float(logprob.detach().cpu()))
                ep_dones.append(float(term or trunc))
                ep_states.append(stored_state)

                if term or trunc:
                    break

            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_rewards.extend(ep_rewards)
            all_values.extend(ep_values)
            all_logprobs.extend(ep_logprobs)
            all_dones.extend(ep_dones)
            all_states.extend(ep_states)

            # *** Meta-episode: DO NOT reset LSTM state ***
            # This is RL² — the LSTM carries task information across episodes.
            # Only reset the env, not the hidden state.
            obs, _ = env.reset()
            ep_start = False

    n_steps = len(all_rewards)
    if n_steps < 32:
        return

    # === Compute GAE(λ) ===
    rewards_t = torch.tensor(all_rewards, dtype=torch.float32, device=device)
    values_t = torch.tensor(all_values, dtype=torch.float32, device=device)
    dones_t = torch.tensor(all_dones, dtype=torch.float32, device=device)
    logprobs_t = torch.tensor(all_logprobs, dtype=torch.float32, device=device)

    advantages = torch.zeros(n_steps, dtype=torch.float32, device=device)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        # dones_t[t] == 1 means the episode ended AT step t, so masking with
        # (1 - dones_t[t]) stops both the bootstrap and the GAE recursion at
        # episode/env boundaries in this flat-concatenated buffer.
        non_terminal = 1.0 - dones_t[t]
        next_value = values_t[t + 1] if t < n_steps - 1 else 0.0
        delta = rewards_t[t] + cfg.gamma * next_value * non_terminal - values_t[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # === PPO update (multiple epochs) ===
    # Replay through LSTM with sequential segments so it has temporal context.
    # We split the rollout into chunks of ppo_mb_size, feed each chunk as
    # (T=chunk_len, B=1) through the LSTM. This preserves the sequential
    # nature of the policy — the LSTM sees real temporal dependencies,
    # not just independent samples.
    obs_arr = np.stack(all_obs)
    act_arr = np.stack(all_actions)

    # Windows must not straddle env starts: stored_states[c_start] is the
    # correct LSTM init only while the window stays in one env's stream.
    # Env starts store None (lstm_state = None at the env_i loop). Meta-episode
    # env.reset() does not zero the LSTM, so those steps stay in-stream.
    # TransformerController stores None everywhere — keep the original split.
    if all(s is None for s in all_states):
        chunk_len = min(cfg.ppo_mb_size, n_steps)
        n_chunks_rollout = max(1, n_steps // chunk_len)
        windows = [
            (ci * chunk_len, min((ci + 1) * chunk_len, n_steps))
            for ci in range(n_chunks_rollout)
        ]
    else:
        env_starts = [i for i, s in enumerate(all_states) if s is None and i > 0]
        boundaries = [0] + env_starts + [n_steps]
        windows = []
        for k in range(len(boundaries) - 1):
            seg_start, seg_end = boundaries[k], boundaries[k + 1]
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            win_len = min(cfg.ppo_mb_size, seg_len)
            for c_start in range(seg_start, seg_end, win_len):
                windows.append((c_start, min(c_start + win_len, seg_end)))

    for epoch in range(cfg.ppo_epochs):
        chunk_order = torch.randperm(len(windows), device=device)
        for ci in chunk_order:
            c_start, c_end = windows[int(ci)]
            if c_end - c_start < 4:
                continue

            chunk_obs = torch.tensor(obs_arr[c_start:c_end],
                                      dtype=torch.float32, device=device)
            feat_dim = cfg.feat_dim
            chunk_feat = chunk_obs.reshape(chunk_obs.shape[0], -1, feat_dim)[:, -1, :]
            feat_seq = chunk_feat.unsqueeze(1)  # (T, 1, feat_dim)
            chunk_act = torch.tensor(act_arr[c_start:c_end],
                                      dtype=torch.float32, device=device)
            chunk_adv = advantages[c_start:c_end]
            chunk_ret = returns[c_start:c_end]
            chunk_old_lp = logprobs_t[c_start:c_end]

            init_state = _clone_lstm_state(all_states[c_start], device=device)
            new_action, _, new_value, _, _, _, _ = controller(feat_seq, init_state)
            new_mean = new_action[:, 0]  # (T, act_dim)
            new_val = new_value[:, 0, 0]  # (T,)

            std = torch.exp(controller.log_std.clamp(-2, 2))
            dist = torch.distributions.Normal(new_mean, std)
            new_logprob = dist.log_prob(chunk_act).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = torch.exp(new_logprob - chunk_old_lp)
            surr1 = ratio * chunk_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip,
                                 1.0 + cfg.ppo_clip) * chunk_adv
            ppo_loss = -torch.min(surr1, surr2).mean()

            val_loss = F.mse_loss(new_val, chunk_ret)

            loss = ppo_loss + cfg.val_coef * val_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            (cfg.rl_loss_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.max_grad_norm)
            optimizer.step()


def _quick_eval(controller, diff_cfg, cfg, device, rng):
    from ..envs.adaptive_filter_env_v2 import AdaptiveFilterEnvV2, EnvConfigV2

    test_families = ["gaussian", "impulsive", "regime_switch", "burst", "alpha_stable"]
    for fam in test_families:
        env_cfg = EnvConfigV2(episode_len=2000, state_window=1,
                              mu_min=cfg.mu_min, mu_max=cfg.mu_max,
                              leakage_min=cfg.lam_min, leakage_max=cfg.lam_max,
                              mu_base_schedule=cfg.use_mu_schedule)
        env = AdaptiveFilterEnvV2(env_cfg, fixed_family=fam,
                                  fixed_snr_db=10.0, seed=42)
        obs, _ = env.reset()
        state = None
        errs = []
        feat_dim = 11
        sw = obs.shape[0] // feat_dim
        for _ in range(2000):
            obs_2d = obs.reshape(sw, -1)[-1]
            obs_t = torch.tensor(obs_2d, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                action, state, _, _, _, _, _ = controller(obs_t, state)
            action_np = action[0, 0].cpu().numpy()
            obs, _, term, trunc, _ = env.step(action_np)
            errs.append(env.last_e)
            if term or trunc:
                break
        ss_mse = float(np.mean(np.array(errs[-500:]) ** 2))
        ss_db = 10 * np.log10(ss_mse + 1e-12)
        print(f"  [eval] {fam:20s}  ss_mse_db={ss_db:+.2f}")

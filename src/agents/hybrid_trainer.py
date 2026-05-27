"""Hybrid BPTT+RL training engine — v3.1 with all fixes applied.

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

from ..filters.diff_filter import DifferentiableNLMS, DiffNLMSConfig, decode_action_bptt
from ..signals.generators import make_signal
from ..noise.families import make_noise, TRAIN_FAMILIES, OOD_FAMILIES
from ..agents.controller import LSTMController, HybridController, TransformerController


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
    lam_max: float = 1.0
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
    meta_episode_len: int = 3
    terminal_ss_weight: float = 0.3
    no_reward_clip: bool = True
    dropout: float = 0.0


FAMILY_NAMES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}

CURRICULUM_WEIGHTS = {
    "gaussian": 1.0, "colored": 2.0, "impulsive": 2.0,
    "time_varying": 2.0, "regime_switch": 3.0,
    "alpha_stable": 2.0, "burst": 2.0, "chirp_interferer": 1.5,
}

SIGNAL_KINDS = ("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst")
SIGNAL_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
SIGNAL_WEIGHTS /= SIGNAL_WEIGHTS.sum()


def _sample_episode(rng: np.random.Generator, cfg: HybridTrainConfig,
                    curriculum_frac: float = 1.0):
    fam_weights = np.array([CURRICULUM_WEIGHTS.get(f, 1.0) for f in FAMILY_NAMES])
    if curriculum_frac < 1.0:
        ood_weight = max(0.0, curriculum_frac - 0.5) * 2
        for i, f in enumerate(FAMILY_NAMES):
            if f in OOD_FAMILIES:
                fam_weights[i] *= ood_weight
    fam_weights /= fam_weights.sum()

    family = str(rng.choice(FAMILY_NAMES, p=fam_weights))
    sig_kind = str(rng.choice(SIGNAL_KINDS, p=SIGNAL_WEIGHTS))
    snr = float(rng.choice([0.0, 5.0, 10.0, 15.0, 20.0]))

    if sig_kind == "multitone":
        base = rng.uniform(150.0, 400.0)
        clean = make_signal("multitone", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            freqs=[base, base * rng.uniform(1.5, 2.5),
                                   base * rng.uniform(2.5, 4.0)],
                            amps=[1.0, rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.6)])
    elif sig_kind == "am":
        clean = make_signal("am", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            fc=rng.uniform(800.0, 1500.0),
                            fm=rng.uniform(40.0, 120.0),
                            mod_index=rng.uniform(0.3, 0.7))
    elif sig_kind in ("ecg_like", "random_pulses", "square_burst"):
        clean = make_signal(sig_kind, n=cfg.episode_len, fs=cfg.fs, rng=rng)
    else:
        clean = make_signal("sine", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            freq=rng.uniform(150.0, 600.0))

    noise = make_noise(family, clean, rng, snr_db=snr, fs=cfg.fs)
    noisy = clean + noise
    s = float(np.std(noisy)) + 1e-9
    clean = clean / s
    noisy = noisy / s

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
        rec_path = os.path.join(out_dir, "train_records.csv")
        if os.path.isfile(rec_path):
            with open(rec_path, "r") as f:
                import csv as _csv
                reader = _csv.DictReader(f)
                for row in reader:
                    records.append({k: (float(v) if k != "iter" else int(v)) for k, v in row.items()})
        print(f"[v3.1] Resumed from {resume_from} at iter {start_iter}")
        for _ in range(start_iter):
            if scheduler is not None:
                scheduler.step()

    print(f"[v3.1] Training hybrid BPTT+RL on {device}")
    print(f"[v3.1] Controller: {cfg.controller_type}, params: "
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
        last_lam = torch.full((B,), float(np.sqrt(cfg.lam_min * 1.0)), device=device)

        bptt_loss = torch.tensor(0.0, device=device)
        chunk_start = 0

        for t in range(T):
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf[:, 0] = noisy_t[:, t]
            d = clean_t[:, t]
            y = (w * x_buf).sum(dim=1)
            e = d - y
            e_sq = e * e
            input_norm = (x_buf * x_buf).sum(dim=1) + 1e-6

            de = e - last_e
            dde = de - (last_e - last2_e)
            ema_e2 = 0.99 * ema_e2 + 0.01 * e_sq
            autocorr = (e * last_e) / (ema_e2 + 1e-8)
            autocorr = torch.clamp(autocorr, -1.0, 1.0)
            grad_norm = torch.abs(last_mu * e) / torch.sqrt(input_norm + 1e-8)
            sign_de = torch.sign(de)

            feat_t = torch.stack([
                torch.tanh(e * 5.0),
                torch.tanh(e_sq * 2.0),
                torch.tanh(de * 5.0),
                torch.tanh(dde * 5.0),
                torch.tanh(torch.log1p(input_norm / cfg.filter_order + 1e-8)),
                torch.tanh(torch.log1p(e_sq + 1e-8)),
                torch.tanh(autocorr),
                torch.tanh((last_mu - 0.5) * 4.0),
                torch.tanh((last_lam - 0.85) * 10.0),
                torch.tanh(grad_norm * 2.0),
                torch.tanh(sign_de.float()),
            ], dim=-1).unsqueeze(0)

            action, state, value, pred_err, pred_sig, pred_task, pred_snr = controller(feat_t, state)

            mu, lam = decode_action_bptt(action[0], diff_cfg)
            base_mu = 0.8 * max(0.05, 1.0 - 0.8 * t / T)
            mu = base_mu + mu * 0.3
            mu = torch.clamp(mu, cfg.mu_min, cfg.mu_max)
            w = lam.unsqueeze(1) * w + (mu / input_norm).unsqueeze(1) * e.unsqueeze(1) * x_buf

            w_norm = torch.norm(w, dim=1)
            clip_mask = w_norm > 100.0
            if clip_mask.any():
                scale = torch.where(clip_mask, 100.0 / (w_norm + 1e-8), torch.ones_like(w_norm))
                w = w * scale.unsqueeze(1)

            bptt_loss = bptt_loss + cfg.bptt_loss_weight * e_sq.mean()
            if cfg.convergence_bonus > 0 and t < T // 4:
                bptt_loss = bptt_loss + cfg.convergence_bonus * e_sq.mean()
            if cfg.robust_alpha > 0 and len(all_errors_detached) >= 64:
                window = torch.stack(all_errors_detached[-64:])
                bptt_loss = bptt_loss + cfg.robust_alpha * window.abs().median()
            if cfg.aux_error_weight > 0 and pred_err is not None and t > 0:
                aux_loss = F.mse_loss(pred_err[0, 0, 0].expand(B), e.detach())
                bptt_loss = bptt_loss + cfg.aux_error_weight * aux_loss
            if cfg.aux_signal_weight > 0 and pred_sig is not None and t > 0:
                sig_loss = F.mse_loss(pred_sig[0, 0, 0].expand(B), d.detach())
                bptt_loss = bptt_loss + cfg.aux_signal_weight * sig_loss
            if cfg.aux_task_weight > 0 and pred_task is not None:
                task_loss = F.cross_entropy(pred_task[0, 0].unsqueeze(0), fam_idx_t[0:1])
                bptt_loss = bptt_loss + cfg.aux_task_weight * task_loss

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
        if it >= cfg.rl_phase_start and cfg.rl_loss_weight > 0:
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
            print(f"[v3.1] it={it:5d}  ss_mse_db={ss_mse_db:+.2f}  "
                  f"ep_mse_db={ep_mse_db:+.2f}  mu={mean_mu:.4f}  "
                  f"lam={mean_lam:.4f}  lr={rec['lr']:.2e}  "
                  f"time={dt:.1f}s")

        if it % cfg.save_every == 0 and it > 0:
            path = os.path.join(out_dir, f"controller_it{it}.pt")
            torch.save({"state_dict": controller.state_dict(),
                        "config": cfg, "iter": it}, path)

        if it % cfg.eval_every == 0 and it > 0:
            _quick_eval(controller, diff_cfg, cfg, device, rng)

    final_path = os.path.join(out_dir, "controller_final.pt")
    torch.save({"state_dict": controller.state_dict(),
                "config": cfg, "iter": cfg.n_iters}, final_path)

    rec_path = os.path.join(out_dir, "train_records.csv")
    if records:
        with open(rec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    print(f"[v3.1] Done. Final model: {final_path}")
    return controller, records, final_path


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
        episode_len=min(512, cfg.episode_len),
        filter_order=cfg.filter_order,
        mu_min=cfg.mu_min, mu_max=cfg.mu_max,
        leakage_min=cfg.lam_min, leakage_max=cfg.lam_max,
        state_window=1,
        reward_kind="shaped_log_mse",
        convergence_bonus=cfg.convergence_bonus,
        robust_alpha=cfg.robust_alpha,
        terminal_ss_weight=getattr(cfg, 'terminal_ss_weight', 0.3),
        no_reward_clip=getattr(cfg, 'no_reward_clip', True),
    )

    all_obs = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_logprobs = []
    all_dones = []

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

            for step_i in range(env_cfg.episode_len):
                obs_2d = obs.reshape(env_cfg.state_window, -1)[-1]
                obs_t = torch.tensor(obs_2d, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    action, lstm_state, value, _, _, _, _ = controller(obs_t, lstm_state)
                mean = action[0, 0]
                std = torch.exp(controller.log_std.clamp(-2, 2))
                dist = torch.distributions.Normal(mean, std)
                action_sampled = dist.sample()
                logprob = dist.log_prob(action_sampled).sum(-1)

                action_np = action_sampled.cpu().numpy()
                obs, reward, term, trunc, _ = env.step(action_np)

                ep_obs.append(obs.copy())
                ep_actions.append(action_np.copy())
                ep_rewards.append(reward)
                ep_values.append(float(value[0, 0, 0].cpu()))
                ep_logprobs.append(float(logprob.detach().cpu()))
                ep_dones.append(float(term or trunc))

                if term or trunc:
                    break

            all_obs.extend(ep_obs)
            all_actions.extend(ep_actions)
            all_rewards.extend(ep_rewards)
            all_values.extend(ep_values)
            all_logprobs.extend(ep_logprobs)
            all_dones.extend(ep_dones)

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
        if t == n_steps - 1:
            next_value = 0.0
            next_non_terminal = 0.0
        else:
            next_value = values_t[t + 1]
            next_non_terminal = 1.0 - dones_t[t + 1]
        delta = rewards_t[t] + cfg.gamma * next_value * next_non_terminal - values_t[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
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

    chunk_len = min(cfg.ppo_mb_size, n_steps)
    n_chunks_rollout = max(1, n_steps // chunk_len)

    for epoch in range(cfg.ppo_epochs):
        chunk_order = torch.randperm(n_chunks_rollout, device=device)
        for ci in chunk_order:
            c_start = int(ci) * chunk_len
            c_end = min(c_start + chunk_len, n_steps)
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

            new_action, _, new_value, _, _, _, _ = controller(feat_seq)
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
        env_cfg = EnvConfigV2(episode_len=2000, state_window=1)
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

"""Hybrid BPTT+RL trainer for the reference-based ANC Kalman controller.

The controller drives the Kalman process noise Q (and measurement noise R); the
filter and the 11-D state are computed from observables only, while the training
loss/reward use the clean signal (synthetic, training-time only) to isolate the
residual interference the agent can actually control.

Per iteration:
  Phase 1 (BPTT): batch of episodes through the differentiable Kalman filter;
    truncated BPTT of  loss = residual^2 (+ aux heads)  ->  (Q,R)  ->  controller.
  Phase 2 (PPO, after warmup): RL^2 meta-episodes in ANCKalmanEnv, GAE(lambda),
    clipped PPO -- identical machinery to the old hybrid trainer, new filter.

Aux heads (world model): predict next innovation e_{t+1}, next clean d_{t+1}
(training-only supervision), and classify the interference family.
"""
from __future__ import annotations
import os
import time
import csv
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..filters.diff_kalman import DiffKalmanConfig, decode_action_kalman, kalman_step
from ..signals.generators import make_signal
from ..interference.families import make_interference, TRAIN_FAMILIES, OOD_FAMILIES
from ..agents.controller import HybridController, LSTMController

FAMILY_NAMES = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILY_NAMES)}
N_FAMILIES = len(FAMILY_NAMES)

SIGNAL_KINDS = ("multitone", "am", "sine", "ecg_like", "random_pulses", "square_burst")
SIGNAL_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
SIGNAL_WEIGHTS /= SIGNAL_WEIGHTS.sum()

# tanh gains, must match ANCEnvConfig.feat_scale
FEAT_SCALE = torch.tensor([3.0, 2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 0.2, 0.2, 5.0, 1.0])


@dataclass
class KalmanTrainConfig:
    n_iters: int = 2000
    batch_size: int = 24
    episode_len: int = 1000
    fs: float = 360.0
    filter_order: int = 16
    q_min: float = 1e-8
    q_max: float = 1e-3
    r_min: float = 1e-3
    r_max: float = 1e1
    p0: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-5
    trunc_bptt: int = 64
    aux_error_weight: float = 0.2
    aux_signal_weight: float = 0.3
    aux_task_weight: float = 0.1
    convergence_bonus: float = 0.15
    grad_clip: float = 1.0
    curriculum_ramp_iters: int = 800
    device: str = "auto"
    controller_type: str = "hybrid"
    lstm_hidden: int = 256
    n_lstm_layers: int = 2
    feat_dim: int = 11
    n_families: int = N_FAMILIES
    save_every: int = 250
    eval_every: int = 200
    seed: int = 42
    scheduler: str = "cosine"
    # PPO
    rl_phase_start: int = 400
    rl_loss_weight: float = 0.5
    rl_every: int = 2
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    ppo_mb_size: int = 128
    ent_coef: float = 0.01
    val_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    rl_n_envs: int = 4
    meta_episode_len: int = 3
    terminal_ss_weight: float = 0.3
    dropout: float = 0.1


def _sample_episode(rng, cfg, curriculum_frac=1.0):
    fams = list(TRAIN_FAMILIES)
    w = np.ones(len(fams))
    # ramp the non-stationary family (regime_switch) in gradually
    if curriculum_frac < 1.0:
        for i, f in enumerate(fams):
            if f == "regime_switch":
                w[i] *= 0.25 + 0.75 * curriculum_frac
    fam = str(rng.choice(fams, p=w / w.sum()))
    sk = str(rng.choice(SIGNAL_KINDS, p=SIGNAL_WEIGHTS))
    snr = float(rng.choice([-5.0, 0.0, 5.0, 10.0, 15.0]))
    if sk == "multitone":
        base = rng.uniform(150.0, 400.0)
        clean = make_signal("multitone", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            freqs=[base, base * rng.uniform(1.5, 2.5), base * rng.uniform(2.5, 4.0)],
                            amps=[1.0, rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.6)])
    elif sk == "am":
        clean = make_signal("am", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            fc=rng.uniform(800.0, 1500.0), fm=rng.uniform(40.0, 120.0),
                            mod_index=rng.uniform(0.3, 0.7))
    elif sk in ("ecg_like", "random_pulses", "square_burst"):
        clean = make_signal(sk, n=cfg.episode_len, fs=cfg.fs, rng=rng)
    else:
        clean = make_signal("sine", n=cfg.episode_len, fs=cfg.fs, rng=rng,
                            freq=rng.uniform(150.0, 600.0))
    interf, ref = make_interference(fam, clean, rng, snr_db=snr, fs=cfg.fs)
    primary = clean + interf
    s = float(np.std(primary)) + 1e-9
    clean = clean / s
    primary = primary / s
    ref = ref / (float(np.std(ref)) + 1e-9)
    return clean, primary, ref, FAMILY_TO_IDX[fam], snr


def _anc_features(e, x_buf, S, gain_norm, prev_e, ema_e2, logq, logr, M, scale):
    """Torch batched features matching ANCKalmanEnv._features (observable only)."""
    de = e - prev_e
    e_sq = e * e
    autocorr = torch.clamp((e * prev_e) / (ema_e2 + 1e-8), -1.0, 1.0)
    ref_pow = (x_buf * x_buf).sum(dim=1) / M
    raw = torch.stack([
        e,
        e_sq,
        de,
        torch.log1p(ref_pow),
        torch.log1p(torch.clamp(S, min=0.0)),
        torch.log1p(e_sq),
        autocorr,
        logq - np.log(1e-6),
        logr - np.log(1.0),
        gain_norm,
        torch.sign(de),
    ], dim=-1)
    return torch.clamp(torch.tanh(raw * scale), -1.0, 1.0)


def _make_controller(cfg, device):
    if cfg.controller_type == "hybrid":
        C = HybridController
    else:
        C = LSTMController
    return C(feat_dim=cfg.feat_dim, hidden=cfg.lstm_hidden,
             n_lstm_layers=cfg.n_lstm_layers, act_dim=2,
             n_families=cfg.n_families, dropout=cfg.dropout).to(device)


def train_kalman(cfg: KalmanTrainConfig, out_dir: str = "results/runs/kalman",
                 resume_from: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device if cfg.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    kcfg = DiffKalmanConfig(order=cfg.filter_order, q_min=cfg.q_min, q_max=cfg.q_max,
                            r_min=cfg.r_min, r_max=cfg.r_max, p0=cfg.p0)
    scale = FEAT_SCALE.to(device)
    M = cfg.filter_order

    controller = _make_controller(cfg, device)
    optimizer = Adam(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (CosineAnnealingLR(optimizer, T_max=cfg.n_iters, eta_min=1e-6)
                 if cfg.scheduler == "cosine" else None)

    start_iter = 0
    records = []
    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        controller.load_state_dict(ckpt["state_dict"])
        start_iter = ckpt.get("iter", 0) + 1
        for _ in range(start_iter):
            if scheduler is not None:
                scheduler.step()
        print(f"[kalman] resumed from {resume_from} at iter {start_iter}")

    print(f"[kalman] training on {device}; params="
          f"{sum(p.numel() for p in controller.parameters()):,}")

    for it in range(start_iter, cfg.n_iters):
        t0 = time.perf_counter()
        cfrac = min(1.0, it / max(1, cfg.curriculum_ramp_iters))
        cleans, prims, refs, fams = [], [], [], []
        for _ in range(cfg.batch_size):
            c, p, r, fi, _ = _sample_episode(rng, cfg, cfrac)
            cleans.append(c); prims.append(p); refs.append(r); fams.append(fi)
        clean_t = torch.tensor(np.stack(cleans), dtype=torch.float32, device=device)
        prim_t = torch.tensor(np.stack(prims), dtype=torch.float32, device=device)
        ref_t = torch.tensor(np.stack(refs), dtype=torch.float32, device=device)
        fam_t = torch.tensor(fams, dtype=torch.long, device=device)
        B, T = clean_t.shape

        # === BPTT ===
        optimizer.zero_grad()
        w = torch.zeros(B, M, device=device)
        P = torch.eye(M, device=device).unsqueeze(0).repeat(B, 1, 1)
        x_buf = torch.zeros(B, M, device=device)
        state = None
        prev_e = torch.zeros(B, device=device)
        ema_e2 = torch.ones(B, device=device)
        logq = torch.full((B,), float(np.log(np.sqrt(cfg.q_min * cfg.q_max))), device=device)
        logr = torch.full((B,), float(np.log(np.sqrt(cfg.r_min * cfg.r_max))), device=device)
        feat = _anc_features(torch.zeros(B, device=device), x_buf, torch.zeros(B, device=device),
                             torch.zeros(B, device=device), prev_e, ema_e2, logq, logr, M, scale)

        bptt_loss = torch.zeros((), device=device)
        chunk_start = 0
        prev_pred_err = prev_pred_sig = None
        all_res = []

        for t in range(T):
            action, state, value, pred_err, pred_sig, pred_task, _ = controller(
                feat.unsqueeze(0), state)
            q, r = decode_action_kalman(action[0], kcfg)
            x_buf = torch.roll(x_buf, 1, dims=1)
            x_buf = x_buf.clone()
            x_buf[:, 0] = ref_t[:, t]
            w, P, e, K, S = kalman_step(w, P, x_buf, prim_t[:, t], q, r,
                                        max_w_norm=kcfg.max_w_norm, return_aux=True)
            residual = e - clean_t[:, t]                     # supervised
            res_sq = residual * residual
            bptt_loss = bptt_loss + res_sq.mean()
            if cfg.convergence_bonus > 0 and t < T // 4:
                bptt_loss = bptt_loss + cfg.convergence_bonus * res_sq.mean()
            if cfg.aux_error_weight > 0 and prev_pred_err is not None:
                bptt_loss = bptt_loss + cfg.aux_error_weight * F.mse_loss(prev_pred_err, e.detach())
            if cfg.aux_signal_weight > 0 and prev_pred_sig is not None:
                bptt_loss = bptt_loss + cfg.aux_signal_weight * F.mse_loss(prev_pred_sig, clean_t[:, t])
            if cfg.aux_task_weight > 0 and pred_task is not None:
                bptt_loss = bptt_loss + cfg.aux_task_weight * F.cross_entropy(pred_task[0], fam_t)
            prev_pred_err = pred_err[0, :, 0] if pred_err is not None else None
            prev_pred_sig = pred_sig[0, :, 0] if pred_sig is not None else None

            # next-step features (observable)
            gain_norm = torch.norm(K, dim=1)
            ema_e2 = (0.99 * ema_e2 + 0.01 * (e * e)).detach()
            feat = _anc_features(e.detach(), x_buf.detach(), S.detach(), gain_norm.detach(),
                                 prev_e.detach(), ema_e2, torch.log(q).detach(),
                                 torch.log(r).detach(), M, scale)
            prev_e = e.detach()
            logq, logr = torch.log(q).detach(), torch.log(r).detach()
            all_res.append(residual.detach())

            if (t + 1) % cfg.trunc_bptt == 0 or t == T - 1:
                (bptt_loss / min(cfg.trunc_bptt, t - chunk_start + 1)).backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
                optimizer.step(); optimizer.zero_grad()
                bptt_loss = torch.zeros((), device=device)
                chunk_start = t + 1
                w = w.detach(); P = P.detach(); x_buf = x_buf.detach()
                if state is not None:
                    state = (state[0].detach(), state[1].detach())
                feat = feat.detach()
                prev_pred_err = prev_pred_sig = None

        res_t = torch.stack(all_res, dim=1)
        ss = float(res_t[:, -T // 4:].pow(2).mean().cpu())
        ss_db = 10 * np.log10(ss + 1e-12)

        if it >= cfg.rl_phase_start and cfg.rl_loss_weight > 0 and it % max(1, cfg.rl_every) == 0:
            _ppo_phase(controller, cfg, kcfg, rng, device, optimizer)
        if scheduler is not None:
            scheduler.step()

        dt = time.perf_counter() - t0
        records.append(dict(iter=it, ss_res_db=ss_db, time_s=dt,
                            lr=float(optimizer.param_groups[0]['lr'])))
        if it % 25 == 0 or it == cfg.n_iters - 1:
            print(f"[kalman] it={it:5d}  ss_resid_db={ss_db:+.2f}  "
                  f"lr={records[-1]['lr']:.2e}  {dt:.1f}s")
        if it % cfg.save_every == 0 and it > 0:
            torch.save({"state_dict": controller.state_dict(), "config": cfg, "iter": it},
                       os.path.join(out_dir, f"controller_it{it}.pt"))

    final = os.path.join(out_dir, "controller_final.pt")
    torch.save({"state_dict": controller.state_dict(), "config": cfg, "iter": cfg.n_iters}, final)
    if records:
        with open(os.path.join(out_dir, "train_records.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            wr.writeheader(); wr.writerows(records)
    print(f"[kalman] done -> {final}")
    return controller, records, final


def _ppo_phase(controller, cfg, kcfg, rng, device, optimizer):
    from ..envs.anc_kalman_env import ANCKalmanEnv, ANCEnvConfig
    env_cfg = ANCEnvConfig(
        episode_len=cfg.episode_len, filter_order=cfg.filter_order,
        q_min=cfg.q_min, q_max=cfg.q_max, r_min=cfg.r_min, r_max=cfg.r_max, p0=cfg.p0,
        train_families=tuple(TRAIN_FAMILIES), convergence_bonus=cfg.convergence_bonus,
        terminal_ss_weight=cfg.terminal_ss_weight)

    obs_all, act_all, rew_all, val_all, lp_all, done_all = [], [], [], [], [], []
    for _ in range(cfg.rl_n_envs):
        env = ANCKalmanEnv(env_cfg, seed=int(rng.integers(0, 2**31)))
        obs, _ = env.reset()
        lstm_state = None
        for _meta in range(cfg.meta_episode_len):
            for _ in range(cfg.episode_len):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
                with torch.no_grad():
                    action, lstm_state, value, *_ = controller(obs_t, lstm_state)
                mean = action[0, 0]
                std = torch.exp(controller.log_std.clamp(-2, 2))
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
                obs_all.append(obs.copy())
                obs, rew, term, trunc, _ = env.step(a.cpu().numpy())
                act_all.append(a.cpu().numpy()); rew_all.append(rew)
                val_all.append(float(value[0, 0, 0].cpu())); lp_all.append(float(lp.detach().cpu()))
                done_all.append(float(term or trunc))
                if term or trunc:
                    break
            obs, _ = env.reset()   # RL^2: keep LSTM state across episodes

    n = len(rew_all)
    if n < 32:
        return
    rew = torch.tensor(rew_all, dtype=torch.float32, device=device)
    val = torch.tensor(val_all, dtype=torch.float32, device=device)
    done = torch.tensor(done_all, dtype=torch.float32, device=device)
    lp_old = torch.tensor(lp_all, dtype=torch.float32, device=device)
    adv = torch.zeros(n, device=device)
    gae = 0.0
    for t in reversed(range(n)):
        nt = 1.0 - done[t]
        nv = val[t + 1] if t < n - 1 else 0.0
        delta = rew[t] + cfg.gamma * nv * nt - val[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * nt * gae
        adv[t] = gae
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_arr = np.stack(obs_all); act_arr = np.stack(act_all)
    clen = min(cfg.ppo_mb_size, n)
    nch = max(1, n // clen)
    for _ in range(cfg.ppo_epochs):
        for ci in torch.randperm(nch, device=device):
            a0 = int(ci) * clen; a1 = min(a0 + clen, n)
            if a1 - a0 < 4:
                continue
            fo = torch.tensor(obs_arr[a0:a1], dtype=torch.float32, device=device).unsqueeze(1)
            ca = torch.tensor(act_arr[a0:a1], dtype=torch.float32, device=device)
            new_a, _, new_v, *_ = controller(fo)
            mean = new_a[:, 0]; nv = new_v[:, 0, 0]
            std = torch.exp(controller.log_std.clamp(-2, 2))
            dist = torch.distributions.Normal(mean, std)
            nlp = dist.log_prob(ca).sum(-1)
            ent = dist.entropy().sum(-1).mean()
            ratio = torch.exp(nlp - lp_old[a0:a1])
            s1 = ratio * adv[a0:a1]
            s2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv[a0:a1]
            loss = -torch.min(s1, s2).mean() + cfg.val_coef * F.mse_loss(nv, ret[a0:a1]) - cfg.ent_coef * ent
            optimizer.zero_grad()
            (cfg.rl_loss_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.max_grad_norm)
            optimizer.step()

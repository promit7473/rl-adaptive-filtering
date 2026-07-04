# Meta-Learned Step-Size and Leakage Control for Robust Adaptive Filtering

Code for the paper *"Meta-Learned Step-Size and Leakage Control for Robust
Adaptive Filtering"* — Meraj Hossain Promit, Maria Akter Jitu, Chandak Chakma
(under submission, IEEE Signal Processing Letters).

We meta-learn a recurrent controller that drives the **step-size $\mu_t$** and
**leakage $\lambda_t$** of a leaky-NLMS filter. It is trained **entirely on
synthetic noise** (no clean target needed at deployment) and transfers
**zero-shot** to real MIT-BIH ECG. All numbers, tables, and figures in the
paper are regenerated end-to-end by the pipeline below — nothing is
hand-typed (Table I is `\input` from `scripts/make_table1.py` output; the
ECG statistics come from `scripts/ecg_stats.py`).

## ⚠ Status (2026-07-05)

- A code review found and fixed two bugs in the hybrid trainer's PPO phase
  (rollout obs/action misalignment; GAE done-mask off by one). **Any
  `hybrid_seed*` or `results/ablations/` checkpoint trained before
  2026-07-04 is invalid — retrain them.** BPTT-only, RL-only (sb3's own
  PPO), and Meta-AF checkpoints are unaffected.
- The GAE mask in `src/agents/hybrid_trainer.py` is `1 - dones[t]` and that
  is **correct** for this buffer (`dones[t]=1` = episode ended *at* step t);
  do not "fix" it to the CleanRL-style `dones[t+1]`.
- Eval episodes are crc32-seeded (`train_pipeline._episode_rng`) and shared
  by every script, so all methods score on identical realizations and runs
  are reproducible across machines/processes.
- The paper compiles with a loud **PENDING** row in Table I until
  `train_pipeline.py eval` + `make_table1.py` have produced fresh numbers.

---

## What is ours vs. what is compared

**Our method (this work)** — `src/agents/`
| Name | What it is |
|------|------------|
| **Hybrid BPTT+RL (ours)** | The main controller. Truncated BPTT through a differentiable NLMS **+** PPO (RL² meta-episodes), with 3 auxiliary heads (error / signal / noise-class prediction). Trained label-free. Used for the zero-shot ECG transfer. |
| **BPTT-only (ours)** | Ablation of the above — same controller trained with BPTT only (no PPO). |
| **RL-only** | Ablation — sb3 RecurrentPPO with the same action decode (shares `train_pipeline.RL_ENV_KW`), no BPTT, no auxiliary heads. |

**Baselines we compare against** — `src/filters/`
- **Classical:** NLMS, RLS, VSS-LMS (Kwong / Aboulnasr / Mathews), PID-NLMS,
  Fixed-Leaky NLMS, IIR-Notch, Heuristic μ-scheduler.
- **Supervised:** **Meta-AF** (`src/filters/meta_af.py`) — back-propagates
  through a differentiable filter but **requires the clean signal at training
  time** (we do not).

Noise families: train = {gaussian, colored, impulsive, time-varying,
regime-switch}; held-out OOD = {α-stable, burst, chirp interferer}.

---

## Repository layout

```
rl-adaptive-filtering/
├── src/
│   ├── agents/
│   │   ├── controller.py          # LSTM / Hybrid / Transformer controllers (OURS)
│   │   └── hybrid_trainer.py      # Hybrid BPTT+RL training engine (OURS)
│   ├── filters/
│   │   ├── base.py                # AdaptiveFilter ABC + windowize()
│   │   ├── lms.py                 # LMS, NLMS, VSS variants, LMP, schedulers
│   │   ├── rls.py  notch.py  pid.py  fixed_leakage_nlms.py   # classical baselines
│   │   ├── meta_af.py             # Meta-AF supervised baseline (compared)
│   │   └── diff_filter.py         # differentiable leaky-NLMS for BPTT
│   ├── envs/adaptive_filter_env_v2.py   # Gymnasium env (μ, λ actions)
│   ├── signals/generators.py      # multitone, AM, sine, ECG-like, pulses, bursts
│   ├── noise/families.py          # all 8 noise families
│   └── eval/metrics.py            # steady-state MSE, convergence time
├── scripts/
│   ├── train_pipeline.py          # phase-aware: [hybrid|bptt|rl|eval]; shared configs (hybrid_cfg, RL_ENV_KW) + crc32 episode seeding
│   ├── train_meta_af.py           # train the supervised Meta-AF baseline
│   ├── benchmark.py               # evaluate any set of methods (synthetic + ECG)
│   ├── run_ablations.py           # train/eval the 6 ablation variants (Fig. 4)
│   ├── make_table1.py             # synthetic.csv -> paper/tables/table1_generated.tex
│   ├── ecg_stats.py               # paired Wilcoxon for the ECG claims
│   ├── paper_plots.py             # shared figure style (single source)
│   └── fig_recovery.py  fig_realworld.py  fig_ablation.py  fig_convergence.py
├── tests/                         # pytest: env, filters, signals/noise
├── results/                       # CSVs kept; results/runs/ and *.pt are gitignored
├── pyproject.toml  requirements.txt  LICENSE  README.md
```

> Large run artefacts (`results/runs/`, `*.pt`, `*.zip`) and generated paper
> assets (figures, `paper/tables/table1_generated.tex`) are gitignored —
> regenerate them with the scripts below. `paper/paper.tex` is tracked.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Python 3.10, PyTorch 2.x, stable-baselines3 + sb3-contrib. **A GPU is strongly
recommended** — full training is 2000 iterations (~13 s/iter on a 5070 Ti)
and is impractical on CPU. The single architecture used everywhere is the
2-layer LSTM-256 (~1.07M params), set by `LSTM_HIDDEN` / `N_LSTM_LAYERS` in
`scripts/train_pipeline.py`.

---

## Reproduce (full paper pipeline, in order)

```bash
# 1. (compared) train the supervised Meta-AF baseline  -> results/meta_af/meta_af.pt
PYTHONPATH=. python scripts/train_meta_af.py --n-iters 3000

# 2. train our controllers (each phase resumes/skips if already complete)
PYTHONPATH=. python scripts/train_pipeline.py hybrid   # -> results/runs/hybrid_seed42
PYTHONPATH=. python scripts/train_pipeline.py bptt     # -> results/runs/bptt_seed42
PYTHONPATH=. python scripts/train_pipeline.py rl       # -> results/runs/rl_seed42

# 3. synthetic eval on shared episodes -> results/runs/eval/synthetic.csv
PYTHONPATH=. python scripts/train_pipeline.py eval

# 4. Table I -> paper/tables/table1_generated.tex (paper.tex \inputs this)
PYTHONPATH=. python scripts/make_table1.py

# 5. ECG eval (needs internet: wfdb streams MIT-BIH from PhysioNet).
#    Method names must be EXACTLY these — fig_realworld.py / ecg_stats.py
#    validate them and fail loudly otherwise.
PYTHONPATH=. python scripts/benchmark.py --skip-synthetic \
    --out-dir results/runs/eval \
    --hybrid-models "Hybrid (ours)=results/runs/hybrid_seed42/controller_final.pt" \
    --rl-models "RL-only=results/runs/rl_seed42/rl_seed42_final.zip" \
    --meta-af-path results/meta_af/meta_af.pt
PYTHONPATH=. python scripts/ecg_stats.py         # paired Wilcoxon for Sec. IV-B

# 6. ablations (6 variants, ~4 h each) -> results/ablations/ablation_eval.csv
PYTHONPATH=. python scripts/run_ablations.py train
PYTHONPATH=. python scripts/run_ablations.py eval

# 7. paper figures (each fails loudly if its data/checkpoint is missing)
PYTHONPATH=. python scripts/fig_realworld.py     # zero-shot ECG transfer
PYTHONPATH=. python scripts/fig_ablation.py      # auxiliary-head ablation
PYTHONPATH=. python scripts/fig_recovery.py      # within-episode burst recovery
```

> **Note:** model checkpoints (`*.pt`, `*.zip`) are gitignored, so after cloning
> on a fresh machine you must retrain (step 1–2) before the benchmark/figures
> have models to load.

---

## Key design choices

| Item        | Setting |
|-------------|---------|
| Sampling    | 360 Hz native (matches MIT-BIH; no resample at eval) |
| Filter      | Leaky-NLMS, order M = 16, ε = 10⁻⁸ |
| Actions     | log-scaled μ ∈ [0.005, 2.0], λ ∈ [0.80, 0.999]; decaying base-μ schedule shared by BPTT/PPO/eval (`diff_filter.mu_base_schedule`) |
| State       | 11 features per step (state_window = 1), tanh-scaled |
| Controller  | LSTM 256×2 (~1.07M params) + actor/critic + 3 auxiliary heads |
| Training    | alternating truncated BPTT (τ = 64) + PPO (GAE-λ) from iter 400, RL² meta-episodes (K = 3) |
| Eval metric | steady-state MSE (last 25% of episode), raw units, mean over 5 seeds |

---

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see `LICENSE`.

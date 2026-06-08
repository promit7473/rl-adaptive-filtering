# Meta-Learned Step-Size and Leakage Control for Robust Adaptive Filtering

Code for the paper *"Meta-Learned Step-Size and Leakage Control for Robust
Adaptive Filtering"* — Meraj Hossain Promit, Maria Akter Jitu, Chandak Chakma
(under submission, IEEE Signal Processing Letters).

We meta-learn a recurrent controller that drives the **step-size $\mu_t$** and
**leakage $\lambda_t$** of a leaky-NLMS filter. It is trained **entirely on
synthetic noise** (no clean target needed at deployment) and transfers
**zero-shot** to real MIT-BIH ECG, gaining **+6.7 dB over NLMS on 50 Hz
powerline interference** (paired Wilcoxon, N = 25, p < 10⁻⁷).

---

## What is ours vs. what is compared

**Our method (this work)** — `src/agents/`
| Name | What it is |
|------|------------|
| **Hybrid BPTT+RL (ours)** | The main controller. Truncated BPTT through a differentiable NLMS **+** PPO, with 3 auxiliary heads (error / signal / noise-class prediction). Trained label-free. |
| **BPTT-only (ours)** | Ablation of the above — same controller trained with BPTT only (no PPO). |
| **RL-only** | Ablation — RecurrentPPO controller, no BPTT, no auxiliary heads. |
| **Meta-RL (RL²)** | The recurrent meta-policy used for the **zero-shot ECG** transfer. |

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
│   ├── train_pipeline.py          # train Hybrid + BPTT-only + RL-only, then benchmark
│   ├── train_meta_af.py           # train the supervised Meta-AF baseline
│   ├── benchmark.py               # evaluate any set of methods (synthetic + ECG)
│   ├── paper_plots.py             # shared figure style (single source)
│   └── fig_recovery.py  fig_realworld.py  fig_ablation.py  fig_convergence.py
├── tests/                         # pytest: env, filters, signals/noise
├── results/                       # CSVs kept; results/runs/ and *.pt are gitignored
├── pyproject.toml  requirements.txt  LICENSE  README.md
```

> `paper/` (LaTeX + figures) and large run artefacts (`results/runs/`, `*.pt`,
> `*.zip`) are gitignored — regenerate them with the scripts below.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Python 3.10, PyTorch 2.x, stable-baselines3 + sb3-contrib. **A GPU is strongly
recommended** — full training is ~6000 iterations and is impractical on CPU.

---

## Reproduce

```bash
# 1. (compared) train the supervised Meta-AF baseline  -> results/meta_af/meta_af.pt
PYTHONPATH=. python scripts/train_meta_af.py --n-iters 4000

# 2. train OUR controllers + the RL-only baseline, then benchmark them
#    (writes results/runs/{hybrid,bptt,rl}_seed42 and results/runs/eval/{synthetic,summary}.csv)
PYTHONPATH=. python scripts/train_pipeline.py
```
Model size is set in `scripts/train_pipeline.py` (`lstm_hidden`, `n_lstm_layers`):
the default is the 3-layer / 512 model; set `256` / `2` for the ~1M-param
lightweight variant.

```bash
# 3. (optional) standalone benchmark, e.g. SNR=10 synthetic only
PYTHONPATH=. python scripts/benchmark.py --snrs 10 --skip-ecg \
    --hybrid-models "Hybrid (ours)=results/runs/hybrid_seed42/controller_final.pt" \
                    "BPTT-only (ours)=results/runs/bptt_seed42/controller_final.pt" \
    --rl-models "RL-only=results/runs/rl_seed42/rl_seed42_final.zip" \
    --meta-af-path results/meta_af/meta_af.pt \
    --out-dir results/benchmark

# 4. paper figures (fig_recovery needs a trained checkpoint at results/runs/hybrid_seed42)
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
| Actions     | log-scaled μ ∈ [0.005, 2.0], λ ∈ [0.80, 1.0] |
| State       | 11 features × 32-step window = 352-D, tanh-scaled |
| Controller  | LSTM (512×3 default) + actor/critic + 3 auxiliary heads |
| Training    | truncated BPTT (τ = 64) warmup → PPO (GAE-λ), RL² meta-episodes |
| Eval metric | steady-state MSE (last 25% of episode), mean over seeds |

---

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see `LICENSE`.

# RL Adaptive Filtering

Code for **"Meta-Learned Step-Size and Leakage Control for Robust Adaptive
Filtering"** — Meraj Hossain Promit, Maria Akter Jitu, Chandak Chakma
(*under submission to IEEE Signal Processing Letters; engrXiv preprint*).

A recurrent meta-policy (RL²) controls the step-size μₜ and leakage λₜ of a
leaky-NLMS filter. Trained only on synthetic noise, it transfers zero-shot to
MIT-BIH ECG records and gains **6.7 dB over NLMS on 50 Hz powerline
interference** (paired Wilcoxon, N = 25, p < 10⁻⁷).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested on Python 3.10, PyTorch 2.3, stable-baselines3 2.3, sb3-contrib 2.3.

## Repository layout

```
rl-adaptive-filtering/
├── configs/
│   └── default.yaml            # env + RL hyperparams (v2)
├── scripts/
│   ├── train.py                # RecurrentPPO / PPO-MLP, multi-seed
│   ├── train_meta_af.py        # BPTT LSTM controller (Meta-AF reimpl baseline)
│   ├── eval.py                 # All baselines + Meta-RL on synthetic & ECG
│   ├── stats.py                # Wilcoxon + Holm-Bonferroni + Cohen's d
│   ├── ablate.py               # Reward / curriculum / LSTM / mu_max sweep
│   ├── figures.py              # Regenerate paper figures from CSVs
│   └── smoke.py                # 30-second sanity test
├── src/
│   ├── envs/
│   │   └── adaptive_filter_env.py   # Gymnasium env (mu, lambda actions)
│   ├── filters/
│   │   ├── base.py              # AdaptiveFilter ABC + windowize()
│   │   ├── lms.py               # LMS, NLMS, VSS variants, LMP, schedulers
│   │   ├── rls.py               # RLS
│   │   ├── notch.py             # IIR notch (50 Hz powerline baseline)
│   │   ├── pid.py               # PID-controlled mu (heuristic baseline)
│   │   └── fixed_leakage_nlms.py
│   ├── noise/families.py        # gaussian, colored, impulsive, time-varying,
│   │                            # regime-switch (+ OOD: alpha-stable, burst,
│   │                            # chirp interferer)
│   ├── signals/generators.py    # sine, multitone, AM, chirp
│   ├── agents/                  # legacy SB3 wrappers (kept for compat)
│   ├── eval/                    # metrics, bootstrap stats, plotting helpers
│   └── supervised/              # CNN denoiser (optional supervised baseline)
├── tests/                       # pytest suite for env, filters, signals
├── results/                     # CSVs + trained .zip policies (gitignored)
├── pyproject.toml
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

`paper/` (LaTeX sources + figures) is gitignored.

## Reproducing the paper

```bash
# 0. Smoke test (~30 s)
python3 scripts/smoke.py

# 1. Train Meta-RL (5 seeds, ~6 h/seed on a single A100)
PYTHONPATH=. python3 scripts/train.py --policy meta --n-seeds 5 \
    --total-steps 1500000 --out-dir results/v2_meta

# 2. Train PPO-MLP baseline (5 seeds, ~3 h/seed)
PYTHONPATH=. python3 scripts/train.py --policy mlp --n-seeds 5 \
    --total-steps 800000 --out-dir results/v2_mlp

# 3. Train Meta-AF (BPTT) baseline for direct comparison (~2 h)
PYTHONPATH=. python3 scripts/train_meta_af.py \
    --out results/v2_meta_af/meta_af.pt --n-iters 4000

# 4. Eval everything (synthetic + zero-shot MIT-BIH/QT-DB ECG with NSTDB noise)
PYTHONPATH=. python3 scripts/eval.py --out-dir results/v2_eval \
    --rl-models Meta-RL=results/v2_meta/ppo_meta_seed42_final.zip \
                PPO-MLP=results/v2_mlp/ppo_mlp_seed42_final.zip \
    --meta-af-path results/v2_meta_af/meta_af.pt

# 5. Paired stats (Wilcoxon + Holm-Bonferroni + Cohen's d)
PYTHONPATH=. python3 scripts/stats.py results/v2_eval/synthetic.csv \
    --reference Meta-RL --out results/v2_eval/synthetic_stats.csv
PYTHONPATH=. python3 scripts/stats.py results/v2_eval/ecg.csv \
    --reference Meta-RL --out results/v2_eval/ecg_stats.csv

# 6. Ablations (~3 h on GPU, 7 short training runs)
PYTHONPATH=. python3 scripts/ablate.py --total-steps 300000

# 7. Regenerate figures
PYTHONPATH=. python3 scripts/figures.py --eval-dir results/v2_eval \
    --fig-dir paper/figures
```

## Key design choices

| Item        | Setting                                             |
|-------------|-----------------------------------------------------|
| Sampling    | 360 Hz native (matches MIT-BIH; no resample at eval)|
| Filter      | Leaky-NLMS, order M = 16, ε = 10⁻⁸                  |
| Actions     | log-scaled μ ∈ [0.005, 2.0], λ ∈ [0.80, 1.0]        |
| State       | 7 features × 16-step window = 112-D, tanh-scaled    |
| Reward      | r = −log(1 + e²) (log-MSE; non-saturating)          |
| Curriculum  | 50% transient noise (regime-switch / impulsive / TV)|
| PPO-MLP     | (128, 128), ~62 k params, 0.8 M steps               |
| Meta-RL     | LSTM₂₅₆ → (128, 128), ~310 k params, 1.5 M steps    |
| Eval metric | IQM over (seed × family), 95 % bootstrap CI         |

## Tests

```bash
python3 -m pytest tests/ -v
```

## License

MIT. If you use this code, please cite the engrXiv preprint
(DOI added once issued) and the SPL paper if/when accepted.

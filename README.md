# RL Adaptive Filtering

Code for **"Meta-Learned Step-Size and Leakage Control for Robust Adaptive Filtering"**
Meraj Hossain Promit, Maria Akter Jitu, Chandak Chakma — *under submission to IEEE Signal Processing Letters; preprint on arXiv*.

A recurrent meta-policy (RL²) controls the step-size μₜ and leakage λₜ of a
leaky-NLMS filter. Trained only on synthetic noise, it transfers zero-shot to
MIT-BIH ECG records and gains **6.7 dB over NLMS on 50 Hz powerline
interference** (paired Wilcoxon, N=25, p < 10⁻⁷).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Tested on Python 3.10, PyTorch 2.3, stable-baselines3 2.3, sb3-contrib 2.3.

## Layout

```
configs/   YAML training/eval defaults
src/
  agents/      PPO + RecurrentPPO training
  envs/        Gymnasium adaptive-filter env
  filters/     LMS / NLMS / RLS / VSS baselines
  noise/       5 noise families (gaussian, impulsive, burst, regime-switch, baseline-wander)
  signals/     Synthetic clean signal generators
  eval/        Metrics, paired-stats, plotting
scripts/
  train_rl.py            Train PPO-MLP and/or Meta-RL
  run_baselines.py       Classical baselines
  eval_all.py            Full method-vs-noise eval
  eval_realworld.py      Zero-shot single-record ECG (record 100) + recovery
  eval_ecg_multi.py      Zero-shot 5-record ECG (paper Table II)
  make_paper_figures.py  Regenerate all PDFs from CSVs
  smoke_test.py          Sanity-check the env + training loop
results/   CSVs and trained policy zips (large run dirs are gitignored)
tests/     pytest suite
```

The `paper/` directory (LaTeX sources + figures) is gitignored.

## Reproducing the paper

```bash
# 1. Train (5 seeds; ~60 min PPO-MLP, ~120 min Meta-RL on a single GPU)
python3 scripts/train_rl.py --policy mlp  --n-seeds 5
python3 scripts/train_rl.py --policy meta --n-seeds 5

# 2. Synthetic eval (writes results/full_evaluation.csv)
python3 scripts/eval_all.py --snr 0 5 10 15 20 \
  --rl-models "Meta-RL=results/ppo_meta_v5/ppo_meta_seed442_final.zip"

# 3. Real ECG (MIT-BIH; auto-downloads via wfdb)
python3 scripts/eval_realworld.py  --meta-model results/ppo_meta_v5/ppo_meta_seed442_final.zip
python3 scripts/eval_ecg_multi.py            # 5 records × 5 seeds × 6 noises

# 4. Figures
python3 scripts/make_paper_figures.py
```

## Key design choices

| Item        | Setting                                             |
|-------------|-----------------------------------------------------|
| Filter      | Leaky-NLMS, order M = 16, ε = 10⁻⁸                  |
| Actions     | log-scaled μ ∈ [0.01, 1.0], λ ∈ [0.8, 1.0]          |
| State       | 7 features × 16-step window = 112-D, tanh-normalised |
| Reward      | r = −softplus(e²) / softplus(1)                     |
| PPO-MLP     | (128, 128) MLP, ~62k params, 600k steps             |
| Meta-RL     | LSTM₁₂₈ → MLP, ~108k params, 1.2M steps             |
| Eval metric | IQM over (seed × family), 95% bootstrap CI          |

## Tests

```bash
python3 -m pytest tests/ -v
```

## License

MIT. If you use this code, please cite the arXiv preprint (citation will be
added once the arXiv ID is assigned).

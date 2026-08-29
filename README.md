# Meta-RL Control of a Kalman Filter for Robust Interference Cancellation

Code for *"Meta-Reinforcement Learning of Kalman Process Noise for Robust
Adaptive Interference Cancellation"* — Meraj Hossain Promit, Maria Akter Jitu,
Chandak Chakma (under submission, IEEE Signal Processing Letters).

We learn a recurrent policy that sets the **process noise $Q_t$** (and
measurement noise $R_t$) of a **Kalman adaptive filter** performing
**reference-based interference cancellation**. The controller is trained by a
hybrid of truncated BPTT through the differentiable Kalman filter **+** PPO
(RL² meta-episodes), entirely on synthetic interference, and transfers
**zero-shot** to real MIT-BIH ECG.

## Why this is sound (the key design invariant)

The filter cancels interference using a **reference** correlated with the
interference but independent of the clean signal:

```
primary[n]   = clean[n] + interference[n]      (observable sensor)
reference[n] = correlated interference source   (observable)
ŷ[n]         = wₙᵀ · reference-window           (Kalman estimate of interference)
e[n]         = primary[n] − ŷ[n]  ≈ clean[n]    (denoised output AND adaptation error)
```

**Filter adaptation and the policy state use observables only** (`reference`,
`primary`, innovation `e`, Kalman gain, innovation variance). The **clean
signal is used only to shape the training loss/reward** (synthetic, train-time
only) — never at deployment. This is what makes the "no clean reference at
deployment / zero-shot on real ECG" claim actually true.

Because the Kalman predict step adds bounded process noise (`P ← P + QI`), the
filter is **unconditionally stable** — no covariance windup, unlike
fixed-forgetting RLS.

> **Scope:** reference-based ANC only applies to *structured* interference that
> admits a reference (powerline, baseline wander, echo, narrowband). White /
> impulsive / α-stable noise is intentionally excluded — no reference can
> cancel it.

## What is ours vs. what is compared

**Ours** — `src/agents/kalman_trainer.py` (hybrid BPTT+PPO), driving
`src/filters/diff_kalman.py` (differentiable Kalman filter, action = `(Q,R)`).
Ablations: **BPTT-only** and **RL-only** (sb3 RecurrentPPO on the ANC env).

**Baselines** — `src/filters/anc_baselines.py`: reference-driven NLMS,
variable-step LMS, RLS (fixed forgetting — shows divergence at low ff),
fixed-Q Kalman, cascaded IIR-notch.

**Interference families** — `src/interference/families.py`:
train = {powerline, baseline_wander, echo, regime_switch};
held-out (zero-shot) = {narrowband_chirp, echo_long}.

## Repository layout

```
src/
├── interference/families.py     # (interference, reference) generators — the ANC task
├── filters/
│   ├── diff_kalman.py           # differentiable Kalman filter, action=(Q,R)  (core)
│   └── anc_baselines.py         # NLMS / VSS / RLS / fixed-Q Kalman / notch baselines
├── envs/anc_kalman_env.py       # Gymnasium ANC env (observable state, (Q,R) action)
├── agents/
│   ├── kalman_trainer.py        # hybrid BPTT+PPO trainer  (OURS)
│   └── controller.py            # LSTM actor-critic + aux heads (reused)
└── signals/generators.py        # clean signal morphologies
scripts/
├── train_anc.py                 # pipeline: [hybrid|bptt|rl|eval]
└── make_table1_anc.py           # eval CSV -> paper/tables/table1_generated.tex
paper/paper.tex  paper/references.bib
```

> The older supervised noisy→clean code (`adaptive_filter_env_v2.py`,
> `hybrid_trainer.py`, `noise/families.py`, `meta_af.py`) is **deprecated** —
> it drove the LMS update with the clean signal at test time, which is why the
> paper was redesigned. Kept for reference only; not used by `train_anc.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```
Python 3.10, PyTorch 2.x, stable-baselines3 + sb3-contrib. GPU strongly
recommended (full training is 2000 iterations).

## Reproduce

```bash
# 1. train our controller + ablations (resume-capable)
PYTHONPATH=. python scripts/train_anc.py hybrid   # -> results/runs/anc_hybrid_seed42
PYTHONPATH=. python scripts/train_anc.py bptt      # ablation: BPTT only
PYTHONPATH=. python scripts/train_anc.py rl        # ablation: PPO only (sb3)

# 2. eval everything on shared episodes -> results/runs/eval_anc/synthetic.csv
PYTHONPATH=. python scripts/train_anc.py eval

# 3. Table I -> paper/tables/table1_generated.tex (paper.tex \inputs this)
PYTHONPATH=. python scripts/make_table1_anc.py

# 4. paper figures (need the trained checkpoint; ECG figure needs internet)
PYTHONPATH=. python scripts/fig_recovery_anc.py \
    --model results/runs/anc_hybrid_seed42/controller_final.pt
PYTHONPATH=. python scripts/fig_realworld_anc.py \
    --model results/runs/anc_hybrid_seed42/controller_final.pt
```

The paper compiles with a loud **PENDING** row in Table I until step 2–3 have
produced fresh numbers; all empirical values in `paper.tex` are wrapped in
`\pend{}` (render orange) until verified.

### Running training on a remote GPU (no manual file juggling)

```bash
# one-time: sync repo to the GPU box (checkpoints/figures are gitignored -> tiny)
rsync -az --exclude results/runs --exclude '*.pt' --exclude '*.zip' \
      ./ rmedu-pc:/tmp/rlaf/
# launch + monitor + pull results back
ssh rmedu-pc 'cd /tmp/rlaf && PYTHONPATH=. python scripts/train_anc.py hybrid bptt rl eval'
rsync -az rmedu-pc:/tmp/rlaf/results/runs/eval_anc ./results/runs/
```

## Tests

```bash
python -m pytest tests/ -q
```

## License
MIT — see `LICENSE`.

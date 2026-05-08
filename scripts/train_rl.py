"""Unified training script for PPO-MLP and Meta-RL policies.

Usage:
    python scripts/train_rl.py --policy mlp --n-seeds 5 --total-steps 600000
    python scripts/train_rl.py --policy meta --n-seeds 5 --total-steps 1200000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.train_ppo import train_mlp, train_recurrent, train_multi_seed


def main():
    parser = argparse.ArgumentParser(description="Train RL adaptive-filter controllers")
    parser.add_argument("--policy", type=str, default="mlp", choices=["mlp", "meta", "both"])
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of policy seeds")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--episode-len", type=int, default=2000)
    parser.add_argument("--state-window", type=int, default=16)
    parser.add_argument("--filter-order", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    common_kwargs = dict(
        n_envs=args.n_envs,
        episode_len=args.episode_len,
        state_window=args.state_window,
        filter_order=args.filter_order,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        device=args.device,
    )

    if args.policy in ("mlp", "both"):
        mlp_steps = args.total_steps or 600_000
        print(f"Training PPO-MLP: {args.n_seeds} seeds, {mlp_steps} steps each")
        train_multi_seed(
            n_seeds=args.n_seeds,
            policy_type="mlp",
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            total_timesteps=mlp_steps,
            hidden_size=args.hidden_size,
            **common_kwargs,
        )

    if args.policy in ("meta", "both"):
        meta_steps = args.total_steps or 1_200_000
        print(f"Training Meta-RL: {args.n_seeds} seeds, {meta_steps} steps each")
        train_multi_seed(
            n_seeds=args.n_seeds,
            policy_type="meta",
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            total_timesteps=meta_steps,
            hidden_size=args.hidden_size,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()

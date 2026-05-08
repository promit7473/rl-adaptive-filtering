"""Unified evaluation script for all methods.

Usage:
    python scripts/eval_all.py --snr 10
    python scripts/eval_all.py --snr 0 5 10 15 20
    python scripts/eval_all.py --rl-models results/ppo_mlp/mlp_seed42/ppo_mlp_seed42_seed42_final.zip
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.evaluation import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument("--snr", nargs="+", type=float, default=[10.0],
                        help="SNR values in dB")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="results/full_evaluation.csv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rl-models", nargs="*", default=[],
                        help="Paths to RL model .zip files as name=path pairs")
    args = parser.parse_args()

    model_paths = {}
    for item in args.rl_models:
        if "=" in item:
            name, path = item.split("=", 1)
            model_paths[name] = path
        else:
            name = os.path.splitext(os.path.basename(item))[0]
            model_paths[name] = item

    run_full_evaluation(
        output_csv=args.output,
        model_paths=model_paths if model_paths else None,
        snr_db_values=args.snr,
        seeds=args.seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()

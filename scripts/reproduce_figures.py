# scripts/reproduce_figures.py
"""
Recreate paper figures from saved evaluation data and configs.
Usage:
    python scripts/reproduce_figures.py --config configs/paper.yaml --out_dir figures
"""

import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml", help="Path to config")
    parser.add_argument("--out_dir", default="figures", help="Output directory for figures")
    parser.add_argument("--metrics_csv", default="results/metrics/weekly_metrics.csv", help="Evaluation CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Example: Pareto trade-off & sensitivity
    subprocess.run([
        "python", "scripts/plot_reward_profiles.py",
        "--config", args.config,
        "--metrics_csv", args.metrics_csv,
        "--out", os.path.join(args.out_dir, "RewardPareto.png")
    ])

    subprocess.run([
        "python", "scripts/plot_sensitivity_grid.py",
        "--config", args.config,
        "--metrics_csv", args.metrics_csv,
        "--out", os.path.join(args.out_dir, "RewardSensitivity.png")
    ])

    print(f"✅ Figures saved in {args.out_dir}")

if __name__ == "__main__":
    main()

"""
Plot reward-weight sensitivity (alpha, beta) and a Pareto (Yield vs Resource) plot
using REAL metrics only.

Inputs:
  - configs/paper.yaml (for reward profile weights & caps)
  - results/metrics/weekly_metrics.csv (your logged eval metrics; one row/episode)

Outputs:
  - figures/RewardSensitivity.png  (alpha/beta heatmap of mean reward)
  - figures/RewardPareto.png       (yield vs resource cloud + Pareto frontier)
  - figures/RewardSensitivity_grid.csv (the grid used to draw the heatmap)
  - figures/RewardPoints.csv          (the points used to draw the Pareto)

Assumptions about CSV columns:
  stem_elong, stem_thick, cum_trusses, heat, co2, elec_high, elec_low, irrig

Usage (from repo root):
  python scripts/plot_reward_profiles.py
  python scripts/plot_reward_profiles.py --profile equal_weights_v1 --alpha_min 0.2 --alpha_max 1.5 --alpha_steps 30 --beta_min 0.2 --beta_max 1.5 --beta_steps 30
  python scripts/plot_reward_profiles.py --config configs/paper.yaml --metrics_csv results/metrics/weekly_metrics.csv --out_dir figures/

You can also override caps if your CSV isn't normalized:
  --crop_max 1 1 1 --res_max 7 7 7 7
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from typing import Tuple

DEFAULT_CROP_MAX = np.array([1.0, 1.0, 1.0])        # stem_elong, stem_thick, cum_trusses
DEFAULT_RES_MAX  = np.array([7.0, 7.0, 7.0, 7.0])   # heat, co2, (elec_high+elec_low), irrig

REQUIRED_COLS = [
    "stem_elong", "stem_thick", "cum_trusses",
    "heat", "co2", "elec_high", "elec_low", "irrig"
]

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_profile(cfg: dict, name: str) -> dict:
    r = cfg.get("reward", {})
    profiles = r.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"Profile '{name}' not found in config.reward.profiles.")
    return profiles[name]

def compute_scores(df: pd.DataFrame,
                   crop_weights: np.ndarray,
                   crop_max: np.ndarray,
                   res_weights: np.ndarray,
                   res_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (crop_score, res_score) per-row based on your reward's components:
      crop_score  = sum_i w_i * (crop_i / crop_max_i)
      res_score   = sum_j w_j * (res_j  / res_max_j)
    where resources use [heat, co2, elec_high+elec_low, irrig].
    """
    # Validate columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Crop
    crop_raw = df[["stem_elong", "stem_thick", "cum_trusses"]].to_numpy(dtype=float)
    crop_norm = crop_raw / crop_max  # (N,3)
    crop_score = (crop_norm * crop_weights).sum(axis=1)

    # Resources (combine electricity channels)
    elec_sum = (df["elec_high"].to_numpy(dtype=float) + df["elec_low"].to_numpy(dtype=float)).reshape(-1, 1)
    res_raw = np.concatenate([
        df[["heat", "co2"]].to_numpy(dtype=float),
        elec_sum,
        df[["irrig"]].to_numpy(dtype=float)
    ], axis=1)  # (N,4)
    res_norm = res_raw / res_max
    res_score = (res_norm * res_weights).sum(axis=1)

    return crop_score, res_score

def reward_from_scores(alpha: float, beta: float, crop_score: np.ndarray, res_score: np.ndarray) -> np.ndarray:
    """
    Mean-field reward without extras to keep comparison clean:
      R = alpha * crop_score - beta * res_score
    (Bonuses/penalties are profile- and threshold-specific; include them
     in a second pass if you want, but keep the heatmap interpretable.)
    """
    return alpha * crop_score - beta * res_score

def pareto_frontier(yield_scores: np.ndarray, resource_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute non-dominated frontier with the convention:
      - Prefer higher yield (maximize)
      - Prefer lower resource (minimize)
    Returns frontier points sorted by yield ascending.
    """
    pts = np.column_stack([yield_scores, resource_scores])  # (N,2)
    # Sort by yield asc, then resource asc
    order = np.lexsort((resource_scores, yield_scores))
    pts = pts[order]

    frontier = []
    best_res = np.inf
    # Traverse from high to low yield to build lower envelope on resource
    for y, r in pts[::-1]:
        if r < best_res:
            frontier.append((y, r))
            best_res = r
    frontier = frontier[::-1]  # back to ascending yield
    fy = np.array([p[0] for p in frontier])
    fr = np.array([p[1] for p in frontier])
    return fy, fr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--metrics_csv", default="results/metrics/weekly_metrics.csv")
    p.add_argument("--profile", default=None, help="Profile name; defaults to reward.profile_active")
    p.add_argument("--alpha_min", type=float, default=0.2)
    p.add_argument("--alpha_max", type=float, default=1.5)
    p.add_argument("--alpha_steps", type=int, default=30)
    p.add_argument("--beta_min", type=float, default=0.2)
    p.add_argument("--beta_max", type=float, default=1.5)
    p.add_argument("--beta_steps", type=int, default=30)
    p.add_argument("--crop_max", nargs=3, type=float, default=None, help="Override crop caps")
    p.add_argument("--res_max",  nargs=4, type=float, default=None, help="Override resource caps")
    p.add_argument("--out_dir", default="figures/")
    p.add_argument("--heatmap_name", default="RewardSensitivity.png")
    p.add_argument("--pareto_name",  default="RewardPareto.png")
    args = p.parse_args()

    # Load config + profile
    cfg = load_config(args.config)
    prof_name = args.profile or cfg.get("reward", {}).get("profile_active", "equal_weights_v1")
    prof = get_profile(cfg, prof_name)

    crop_w = np.array(prof.get("crop_component", {}).get("weights", [1, 1, 1]), dtype=float)
    res_w  = np.array(prof.get("resource_component", {}).get("weights", [1, 1, 1, 1]), dtype=float)

    crop_max = np.array(args.crop_max) if args.crop_max else np.array(
        prof.get("crop_component", {}).get("max_vals", DEFAULT_CROP_MAX), dtype=float
    )
    res_max = np.array(args.res_max) if args.res_max else np.array(
        prof.get("resource_component", {}).get("max_vals", DEFAULT_RES_MAX), dtype=float
    )

    # Load metrics (REAL DATA ONLY)
    if not os.path.isfile(args.metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {args.metrics_csv}")
    df = pd.read_csv(args.metrics_csv)

    # Compute component scores once
    y_scores, r_scores = compute_scores(df, crop_w, crop_max, res_w, res_max)

    # Build alpha/beta grid
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    betas  = np.linspace(args.beta_min,  args.beta_max,  args.beta_steps)
    AA, BB = np.meshgrid(alphas, betas, indexing="xy")

    # Compute mean reward over episodes for each (alpha,beta)
    meanR = np.zeros_like(AA)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            RR = reward_from_scores(AA[i, j], BB[i, j], y_scores, r_scores)
            meanR[i, j] = float(np.mean(RR))

    # Ensure output dir
    os.makedirs(args.out_dir, exist_ok=True)

    # Save grid CSV
    grid_rows = []
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            grid_rows.append({"alpha": a, "beta": b, "mean_reward": meanR[i, j]})
    grid_path = os.path.join(args.out_dir, "RewardSensitivity_grid.csv")
    pd.DataFrame(grid_rows).to_csv(grid_path, index=False)

    # Plot heatmap (alpha on x, beta on y)
    plt.figure(figsize=(7, 5))
    im = plt.imshow(meanR, origin="lower",
                    extent=[alphas.min(), alphas.max(), betas.min(), betas.max()],
                    aspect="auto")
    plt.colorbar(im, label="Mean reward")
    plt.xlabel("alpha (crop weight)")
    plt.ylabel("beta (resource weight)")
    plt.title(f"Reward Sensitivity (Profile: {prof_name})")
    heatmap_path = os.path.join(args.out_dir, args.heatmap_name)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    print(f"Saved heatmap: {heatmap_path}")

    # Pareto: plot all points + non-dominated frontier
    pareto_points = pd.DataFrame({
        "yield_score": y_scores,
        "resource_score": r_scores
    })
    pts_path = os.path.join(args.out_dir, "RewardPoints.csv")
    pareto_points.to_csv(pts_path, index=False)

    fy, fr = pareto_frontier(y_scores, r_scores)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_scores, r_scores, s=18, alpha=0.7, label="Episodes")
    plt.plot(fy, fr, linewidth=2, label="Pareto frontier")
    plt.xlabel("Yield (normalized crop score)")
    plt.ylabel("Resource (normalized consumption score)")
    plt.title(f"Yield–Resource Trade-off (Profile: {prof_name})")
    plt.legend()
    pareto_path = os.path.join(args.out_dir, args.pareto_name)
    plt.tight_layout()
    plt.savefig(pareto_path, dpi=300)
    print(f"Saved Pareto: {pareto_path}")

if __name__ == "__main__":
    main()

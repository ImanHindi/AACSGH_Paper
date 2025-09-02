import argparse
from pathlib import Path
import csv

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3, PPO, SAC, DDPG

from load_config_and_init_model import load_yaml_config, make_env_from_config
from find_checkpoint import find_best_model, find_latest_checkpoint  # NEW

ALGOS = {"td3": TD3, "ppo": PPO, "sac": SAC, "ddpg": DDPG}


def load_model(algo: str, model_path: str, env):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(ALGOS.keys())}")
    cls = ALGOS[algo]
    return cls.load(model_path, env=env, device="auto")


def eval_episodes(model, env, n_eval_episodes: int, deterministic: bool):
    mean_r, std_r, rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        return_episode_rewards=True,
    )
    return mean_r, std_r, rewards, lengths


def maybe_save_csv(out_csv: str, rewards, lengths):
    if not out_csv:
        return
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "length"])
        for i, (r, l) in enumerate(zip(rewards, lengths), start=1):
            w.writerow([i, r, l])
    print(f"[INFO] Saved episode rewards to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved SB3 model on GreenhouseEnv.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to YAML config")
    parser.add_argument("--algo", type=str, required=True, choices=list(ALGOS.keys()), help="Algorithm of the saved model")
    parser.add_argument("--model_path", type=str, default="", help="Path to saved model .zip (best/final/checkpoint)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions during evaluation")
    parser.add_argument("--save_csv", type=str, default="", help="Optional: save per-episode rewards to CSV path")
    parser.add_argument("--auto_best", action="store_true", help="Auto-load best_model.zip if --model_path is empty")
    parser.add_argument("--auto_latest", action="store_true", help="Auto-load latest checkpoint if no best model")
    args = parser.parse_args()

    # Build env from config
    cfg = load_yaml_config(args.config)
    env = make_env_from_config(cfg)

    # Resolve model path with auto options
    resolved_path = args.model_path
    if not resolved_path:
        if args.auto_best:
            best = find_best_model(args.algo)
            if best:
                resolved_path = str(best)
                print(f"[INFO] Auto-selected best model: {resolved_path}")
            elif args.auto_latest:
                latest = find_latest_checkpoint(args.algo)
                if latest:
                    resolved_path = str(latest)
                    print(f"[INFO] Auto-selected latest checkpoint: {resolved_path}")
        if not resolved_path:
            raise FileNotFoundError("No --model_path provided and auto selection failed (no best/latest found).")

    # Load and evaluate
    print(f"[INFO] Loading {args.algo.upper()} model from: {resolved_path}")
    model = load_model(args.algo, resolved_path, env)
    mean_r, std_r, rewards, lengths = eval_episodes(model, env, args.episodes, args.deterministic)

    print(f"[RESULT] {args.algo.upper()} mean reward: {mean_r:.4f} ± {std_r:.4f}")
    print(f"[DETAIL] Per-episode rewards: {rewards}")

    maybe_save_csv(args.save_csv, rewards, lengths)


if __name__ == "__main__":
    main()

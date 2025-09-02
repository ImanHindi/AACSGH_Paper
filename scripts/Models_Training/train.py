import os
import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import TD3, PPO, SAC, DDPG

from load_config_and_init_model import load_env_and_model
from find_checkpoint import find_latest_checkpoint  


def algo_dirs(algo: str):
    algo = algo.lower()
    if algo == "td3":
        base = Path("./TD3_logs_fs")
    elif algo == "ppo":
        base = Path("./PPO_logs")
    elif algo == "sac":
        base = Path("./SAC_logs")
    elif algo == "ddpg":
        base = Path("./DDPG_logs")
    else:
        raise ValueError(f"Unknown algo {algo}")
    return {
        "base": base,
        "best": base / "best_model",
        "results": base / "results",
        "checkpoints": base / "checkpoints",
        "final": base / f"{algo}_final_model",
    }


def ensure_dirs(d):
    for _, v in d.items():
        if isinstance(v, Path):
            v.mkdir(parents=True, exist_ok=True)


def load_checkpoint_if_any(model, env, resume_path: str):
    if not resume_path:
        return model
    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"--resume_from path not found: {resume_path}")
    print(f"[INFO] Resuming from checkpoint: {resume_path}")
    cls = type(model)
    resumed = cls.load(str(resume_path), env=env, device=getattr(model, "device", "auto"))
    return resumed


def train_once(
    config_path: str,
    algo: str = "td3",
    total_timesteps: int = 100_000,
    eval_freq: int = 10_000,
    save_freq: int = 10_000,
    resume_from: str = "",
    deterministic_eval: bool = True,
    n_eval_episodes: int = 10,
    auto_resume: bool = True,   # NEW: Whether to auto-resume from latest checkpoint if no explicit path is given
):
    # 1) Build env + model
    env, model = load_env_and_model(config_path, algo)

    # 2) Folders
    dirs = algo_dirs(algo)
    ensure_dirs(dirs)

    # 3) Callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(dirs["best"]),
        log_path=str(dirs["results"]),
        eval_freq=eval_freq,
        deterministic=deterministic_eval,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(dirs["checkpoints"]),
        name_prefix=f"{algo.upper()}_GREENHOUSE",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 4) Auto-resume if requested and no explicit path
    if auto_resume and not resume_from:
        latest = find_latest_checkpoint(algo)
        if latest:
            print(f"[INFO] Auto-resuming from latest checkpoint: {latest}")
            resume_from = str(latest)

    model = load_checkpoint_if_any(model, env, resume_from)

    # 5) Train
    print(f"[INFO] Training {algo.upper()} for {total_timesteps:,} steps")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, checkpoint_callback])

    # 6) Save final
    final_path = f"{dirs['final']}.zip"
    print(f"[INFO] Saving final model to: {final_path}")
    model.save(dirs["final"])

    # 7) Evaluate final
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic_eval
    )
    print(f"[RESULT] {algo.upper()} mean reward: {mean_reward:.4f} ± {std_reward:.4f}")

    # 8) Evaluate best if present
    best_zip = dirs["best"] / "best_model.zip"
    if best_zip.exists():
        print(f"[INFO] Evaluating best saved model: {best_zip}")
        AlgoClass = {"td3": TD3, "ppo": PPO, "sac": SAC, "ddpg": DDPG}[algo.lower()]
        best_model = AlgoClass.load(str(best_zip), env=env, device=getattr(model, "device", "auto"))
        best_mean, best_std = evaluate_policy(
            best_model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic_eval
        )
        print(f"[RESULT] BEST {algo.upper()} mean reward: {best_mean:.4f} ± {best_std:.4f}")
    else:
        print(f"[WARN] No best_model.zip yet in {dirs['best']} — EvalCallback may not have triggered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SB3 model on GreenhouseEnv using YAML config.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to YAML config")
    parser.add_argument("--algo", type=str, default="td3", choices=["td3", "ppo", "sac", "ddpg"], help="Algorithm")
    parser.add_argument("--total_timesteps", type=int, default=80_000, help="Total timesteps to train")
    parser.add_argument("--eval_freq", type=int, default=10_000, help="Evaluate every N steps")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Checkpoint every N steps")
    parser.add_argument("--resume_from", type=str, default="", help="Path to a checkpoint .zip to resume from")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable auto-resume from latest checkpoint")
    parser.add_argument("--deterministic_eval", action="store_true", help="Use deterministic evaluation")
    parser.add_argument("--n_eval_episodes", type=int, default=10, help="Episodes for evaluation")
    args = parser.parse_args()

    train_once(
        config_path=args.config,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        resume_from=args.resume_from,
        deterministic_eval=args.deterministic_eval,
        n_eval_episodes=args.n_eval_episodes,
        auto_resume=not args.no_auto_resume,   # NEW
    )

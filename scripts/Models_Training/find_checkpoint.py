from pathlib import Path
import re
from typing import Optional, Tuple

ALGO_DIRS = {
    "td3":  {"base": Path("./TD3_logs_fs"), "prefix": "TD3_GREENHOUSE"},
    "ppo":  {"base": Path("./PPO_logs"),    "prefix": "PPO_GREENHOUSE"},
    "sac":  {"base": Path("./SAC_logs"),    "prefix": "SAC_GREENHOUSE"},
    "ddpg": {"base": Path("./DDPG_logs"),   "prefix": "DDPG_GREENHOUSE"},
}

STEP_RE = re.compile(r".*_(\d+)_steps\.zip$", re.IGNORECASE)

def _algo_paths(algo: str) -> Tuple[Path, Path, Path]:
    algo = algo.lower()
    if algo not in ALGO_DIRS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(ALGO_DIRS.keys())}")
    base = ALGO_DIRS[algo]["base"]
    return base, base / "checkpoints", base / "best_model"

def find_latest_checkpoint(algo: str) -> Optional[Path]:
    """
    Returns the path to the checkpoint with the largest step count, or None if none found.
    """
    _, ckpt_dir, _ = _algo_paths(algo)
    if not ckpt_dir.exists():
        return None

    best_steps = -1
    best_path: Optional[Path] = None

    for f in ckpt_dir.glob("*.zip"):
        m = STEP_RE.match(f.name)
        if not m:
            continue
        try:
            steps = int(m.group(1))
        except ValueError:
            continue
        if steps > best_steps:
            best_steps = steps
            best_path = f

    return best_path

def find_best_model(algo: str) -> Optional[Path]:
    """
    Returns the path to best_model.zip created by EvalCallback, or None if not present.
    """
    _, _, best_dir = _algo_paths(algo)
    best_zip = best_dir / "best_model.zip"
    return best_zip if best_zip.exists() else None

if __name__ == "__main__":
    # Quick manual test:
    for a in ALGO_DIRS.keys():
        print(f"--- {a.upper()} ---")
        print("latest checkpoint:", find_latest_checkpoint(a))
        print("best model:", find_best_model(a))

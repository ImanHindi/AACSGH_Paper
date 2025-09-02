# scripts/eval_and_log.py
# """
# How to run:

# cd C:\Users\Iman.Hindi\Desktop\AACSGH_Paper

# python scripts\eval_and_log.py ^
#   --config configs\paper.yaml ^
#   --algo td3 ^
#   --model_path path\to\your_trained_TD3_model.zip ^
#   --episodes 20 ^
#   --out results\metrics\weekly_metrics.csv

# """
import os, argparse, numpy as np, pandas as pd
from stable_baselines3 import TD3
from Models_Training.load_config_and_init_model import load_yaml_config, make_env_from_config
import gymnasium as gym

def run_episodes(env, model, n_episodes=20, seeds=None):
    rows = []
    seeds = seeds or list(range(n_episodes))
    for ep, s in enumerate(seeds):
        obs, info = env.reset(seed=s)
        done = False
        ep_crop, ep_res = [], []
        while not done:
            # Predict without attaching env to model (no replay buffer)
            action, _ = model.predict(obs, deterministic=True)  # TD3 loaded without env
            obs, reward, done, trunc, info = env.step(action)
            ep_crop.append(obs["crop_params"].reshape(-1))          # (3,)
            ep_res.append(obs["resource_consumption"].reshape(-1))  # (5,)
        crop_last = np.mean(np.vstack(ep_crop), axis=0)
        res_last  = np.mean(np.vstack(ep_res),  axis=0)
        rows.append({
            "episode": ep,
            "stem_elong":  crop_last[0],
            "stem_thick":  crop_last[1],
            "cum_trusses": crop_last[2],
            "heat":        res_last[0],
            "co2":         res_last[1],
            "elec_high":   res_last[2],
            "elec_low":    res_last[3],
            "irrig":       res_last[4],
        })
    return pd.DataFrame(rows)
import numpy as np
import gymnasium as gym

class ToFloat32(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self._convert_space(env.observation_space)

    def _convert_space(self, space):
        if isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.floating):
            low = space.low.astype(np.float32) if space.low.dtype.kind == "f" else space.low
            high = space.high.astype(np.float32) if space.high.dtype.kind == "f" else space.high
            return gym.spaces.Box(low=low, high=high, shape=space.shape, dtype=np.float32)
        elif isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict({k: self._convert_space(s) for k, s in space.items()})
        elif isinstance(space, gym.spaces.Tuple):
            return gym.spaces.Tuple(tuple(self._convert_space(s) for s in space.spaces))
        else:
            return space  # leave Discrete/MultiBinary/etc. unchanged

    def observation(self, obs):
        return self._cast_obs(obs, self.observation_space)

    def _cast_obs(self, obs, space):
        if isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.floating):
            return obs.astype(np.float32, copy=False)
        elif isinstance(space, gym.spaces.Dict):
            return {k: self._cast_obs(obs[k], space.spaces[k]) for k in space.spaces.keys()}
        elif isinstance(space, gym.spaces.Tuple):
            return tuple(self._cast_obs(o, s) for o, s in zip(obs, space.spaces))
        else:
            return obs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--algo", default="td3")  # kept for consistency/logging
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="results/metrics/weekly_metrics.csv")
    args = p.parse_args()

    # Build ONLY the env (no model here)
    cfg = load_yaml_config(args.config)
    env = make_env_from_config(cfg, algo_name=args.algo)  # your updated function that accepts algo_name


    # cast observations to float32 recursively (works for Dict/Tuple/Box)
    env = ToFloat32(env)

    # keep replay buffer small so it doesn't allocate GBs in RAM
    custom_objects = {
        "buffer_size": 100,      # tiny is fine for evaluation-only runs
        "learning_starts": 0,
    }

    # do NOT pass env to load (avoid immediate big buffer allocation)
    model = TD3.load(args.model_path, device="cpu", env=None, custom_objects=custom_objects)

    # now attach env (SB3 will build a small buffer using overridden values)
    model.set_env(env)


    # # Define the path to the best model directory
    # best_model_dir = r'C:\Users\Iman.Hindi\Desktop\AACSGH_Paper\models\weights'

    # # Check if the best model file exists
    # best_model_filename = 'TD3_greenhouse_final_model.zip'  # Change this to your actual best model filename if different
    # best_model_path = os.path.join(best_model_dir, best_model_filename)

    # # Load the best model
    # if os.path.exists(best_model_path):
    #     print(f"Loading the best model from {best_model_path}")
    #     best_model = TD3.load(best_model_path, env=env)
    # else:
    #     raise FileNotFoundError(f"No best model found at {best_model_path}")


    # Load trained TD3 WITHOUT env to avoid buffer allocation
    # model = TD3.load(args.model_path, env=env)  # or "cuda" if available; no env passed

    df = run_episodes(env, model, n_episodes=args.episodes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote:", args.out)

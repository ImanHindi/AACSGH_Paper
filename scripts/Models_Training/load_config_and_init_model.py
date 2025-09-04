# scripts/load_config_and_init_model.py

import os
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Any, Dict, Optional

# Keras / TF models for the estimators
from tensorflow.keras.models import load_model

# Gymnasium + SB3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3, PPO, SAC, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from sklearn.preprocessing import MinMaxScaler

def _get_active_reward_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    reward_cfg = cfg.get("reward", {})
    prof = reward_cfg.get("profile_active", "equal_weights_v1")
    return (reward_cfg.get("profiles", {}) or {}).get(prof, {})
# ----------------------
# 1) ENV: GreenhouseEnv
# ----------------------
class GreenhouseEnv(gym.Env):
    """
    Minimal version.
    Loads pre-trained estimators and uses weather data to create observations.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        crop_parameters_estimator,
        resource_consumption_estimator,
        gh_climate_estimator,
        weather_data: np.ndarray,
        max_steps: int = 23,
        action_count: int = 9,
        normalized_actions: bool = True,
        days_scale: float = 166.0,
        reward_cfg: Optional[Dict[str, Any]] = None,   # <-- NEW
    ):
        super().__init__()

        self.crop_parameters_estimator = crop_parameters_estimator
        self.resource_consumption_estimator = resource_consumption_estimator
        self.gh_climate_estimator = gh_climate_estimator

        self.max_steps = max_steps
        self.steps = 0
        self.action_count = action_count
        self.normalized_actions = normalized_actions
        self.days_scale = days_scale

        # Actions: flattened (2016 * action_count,)
        self.action_space = spaces.Box(
            low=0.0 if normalized_actions else -np.inf,
            high=1.0 if normalized_actions else np.inf,
            shape=(2016 * action_count,),
            dtype=np.float32,
        )
        rcfg = reward_cfg or {}
        self.alpha = float(rcfg.get("alpha", 1.0))
        self.beta  = float(rcfg.get("beta",  1.0))
        self.delta = float(rcfg.get("delta", 0.0))
        self.gamma = float(rcfg.get("gamma", 0.0))

        self.use_efficiency = bool(rcfg.get("use_efficiency_term", False))
        self.use_stability  = bool(rcfg.get("use_stability_term",  False))

        cc = rcfg.get("crop_component", {}) or {}
        self.crop_w  = np.array(cc.get("weights",  [1.0, 1.0, 1.0]), dtype=np.float32)
        self.crop_mv = np.array(cc.get("max_vals", [1.0, 1.0, 1.0]), dtype=np.float32)

        rc = rcfg.get("resource_component", {}) or {}
        self.res_w  = np.array(rc.get("weights",  [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        self.res_mv = np.array(rc.get("max_vals", [7.0, 7.0, 7.0, 7.0]), dtype=np.float32)

        self.th = rcfg.get("thresholds", {}) or {}
        self.ex = rcfg.get("extras", {}) or {}
        # Observations: 
        self.observation_space = spaces.Dict({
            "weather": spaces.Box(low=0, high=1, shape=(2016, 10), dtype=np.float32),
            "crop_params": spaces.Box(low=0, high=1, shape=(1, 3), dtype=np.float32),
            "resource_consumption": spaces.Box(low=0, high=7, shape=(1, 5), dtype=np.float32),
            "gh_climate": spaces.Box(low=-10, high=10, shape=(2016, 10), dtype=np.float32),
        })

        self.weather_data = weather_data
        # precompute days feature for the whole horizon
        self.days = np.array([(i // 288) / self.days_scale for i in range(2016 * max_steps)]).reshape(
            2016 * max_steps, 1
        )

        # Placeholders used in step/reset
        self.state = None
        self.daily_res_cons = None

        # Column lists 
        self.actions_sp = [
            "co2_vip",
            "int_white_vip",
            "pH_drain_PC",
            "scr_blck_vip",
            "scr_enrg_vip",
            "t_heat_vip",
            "t_ventlee_vip",
            "water_sup",
            "water_sup_intervals_vip_min",
            "days",
        ]

        self.GH_C_Out_columns = [
            "AssimLight",
            "BlackScr",
            "CO2air",
            "Cum_irr",
            "EC_drain_PC",
            "EnScr",
            "HumDef",
            "PipeGrow",
            "PipeLow",
            "Rhair",
            "Tair",
            "Tot_PAR",
            "Tot_PAR_Lamps",
            "VentLee",
            "Ventwind",
            "assim_vip",
            "co2_dos",
        ]

        self.CP_important_feature = [
            "Tair",
            "pH_drain_PC",
            "Cum_irr",
            "t_heat_vip",
            "water_sup",
            "Tot_PAR",
            "water_sup_intervals_vip_min",
            "PipeGrow",
            "EC_drain_PC",
            "BlackScr",
            "co2_dos",
            "Tot_PAR_Lamps",
            "scr_enrg_vip",
            "Rhair",
            "HumDef",
            "days",
        ]

        self.RC_important_feature = [
            "Cum_irr",
            "BlackScr",
            "water_sup_intervals_vip_min",
            "EC_drain_PC",
            "pH_drain_PC",
            "CO2air",
            "water_sup",
            "HumDef",
            "Rhair",
            "Tot_PAR",
        ]

        self.GH_C_important_feature = [
            "PARout",
            "Tout",
            "Iglob",
            "RadSum",
            "scr_enrg_vip",
            "t_heat_vip",
            "int_white_vip",
            "scr_blck_vip",
            "pH_drain_PC",
            "co2_vip",
            "t_ventlee_vip",
            "days",
        ]

        self.important_ghc = [
            "BlackScr",
            "CO2air",
            "Cum_irr",
            "EC_drain_PC",
            "PipeGrow",
            "HumDef",
            "Rhair",
            "Tair",
            "Tot_PAR",
            "Tot_PAR_Lamps",
        ]

        # Save weather column names to rebuild DataFrames
        self.w_columns = [f"w{i}" for i in range(10)]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.days = np.array([(i // 288) / self.days_scale for i in range(2016 * self.max_steps)]).reshape(
            2016 * self.max_steps, 1
        )

        initial_crop_params = np.zeros(3).reshape(1, 3) + np.random.uniform(0, 0.1, size=(1, 3))
        initial_resource_consumption = np.zeros(5).reshape(1, 5) + np.random.uniform(0, 0.1, size=(1, 5))
        initial_daily_resource_consumption = (
            np.zeros(shape=(7, 1, 5)).reshape(7, 1, 5) + np.random.uniform(0, 0.1, size=(7, 1, 5))
        )
        initial_gh_climate = np.zeros(shape=(1, 2016, 10)).reshape(2016, 10) + np.random.uniform(
            0, 0.1, size=(2016, 10)
        )

        self.state = {
            "weather": np.array(self.weather_data[self.steps * 2016 : (self.steps + 1) * 2016]),
            "crop_params": initial_crop_params,
            "resource_consumption": initial_resource_consumption,
            "gh_climate": initial_gh_climate,
        }
        self.daily_res_cons = initial_daily_resource_consumption
        return self.state, {}

    def step(self, action: np.ndarray):
        # Reshape actions (2016, 9)
        action = action.reshape((2016, self.action_count))

        # Build 'day' feature for this step window
        day = np.array(self.days[self.steps * 2016 : (self.steps + 1) * 2016]).reshape(2016, 1)
        self.steps += 1

        # Control setpoints with 'days' appended
        control_setpoints = np.concatenate([action, day], axis=1)
        control_df = pd.DataFrame(control_setpoints, columns=self.actions_sp)

        # Rebuild weather slice DataFrame using real column names
        weather_df = pd.DataFrame(self.state["weather"], columns=self.w_columns)

        # Predict greenhouse climate
        w_sp_data = pd.concat([weather_df, control_df], axis=1)
        GH_C_Estimator_Input = np.array(w_sp_data[self.GH_C_important_feature])
        ghclimate = self.gh_climate_estimator.predict(GH_C_Estimator_Input).reshape(2016, 17)
        gh_climate = pd.DataFrame(ghclimate, columns=self.GH_C_Out_columns)

        # Predict crop parameters (weekly)
        CP_Estimator_Input = np.array(pd.concat([gh_climate, control_df], axis=1)[self.CP_important_feature]).reshape(1, 2016, 16)
        weekly_crop_params = self.crop_parameters_estimator.predict(CP_Estimator_Input).reshape(1, 3)

        # Predict resource consumption (daily → aggregate weekly)
        RC_Estimator_Input = np.array(pd.concat([gh_climate, control_df], axis=1)[self.RC_important_feature]).reshape(7, 288, 10)
        daily_resource_consumption = self.resource_consumption_estimator.predict(RC_Estimator_Input)  # (7, 288, 5)
        self.daily_res_cons = daily_resource_consumption
        weekly_resource_consumption = daily_resource_consumption.sum(axis=0).reshape(1, 5)  # (1,5)

        # High resource flag like original
        high_rc = False
        for i in daily_resource_consumption[:]:
            if np.any(i >= 1):
                high_rc = True
                break

        # Update state for next step
        next_weather_slice = self.weather_data[self.steps * 2016 : (self.steps + 1) * 2016]
        self.state = {
            "weather": next_weather_slice,
            "crop_params": weekly_crop_params,
            "resource_consumption": weekly_resource_consumption,
            "gh_climate": np.array(gh_climate[self.important_ghc]).reshape(2016, 10),
        }

        # Reward (config-driven)
        reward = self.calculate_reward(
            weekly_crop_params[0],              # shape (3,)
            weekly_resource_consumption[0],     # shape (5,)
            action                               # (2016, 9)
        )

        # Done condition (mirror original)
        done = bool(
            (self.steps >= self.max_steps)
            or np.any(weekly_crop_params[0] < 0.2)
            or np.any(weekly_resource_consumption[0] > 7.0)
            or np.all(weekly_crop_params[0] < 0.5)
              
        )
        truncated = done
        info = {}
        return self.state, reward, done, truncated, info

    def calculate_reward(self, crop_params: np.ndarray, resource_consumption: np.ndarray, current_actions: np.ndarray) -> float:
        # crop_params: shape (3,)
        # resource_consumption: shape (5,) -> we use [heat, co2, elecHi, elecLow, irr] aggregated into 4 slots 
        # current_actions: shape (2016, action_count)

        # Crop reward (weighted normalized sum)
        crop_params = np.array(crop_params, dtype=float).reshape(-1)  # (3,)
        crop_norm   = np.divide(crop_params, self.crop_mv, out=np.zeros_like(crop_params), where=self.crop_mv!=0.0)
        crop_reward = float(np.sum(self.crop_w * crop_norm))

        # Resource penalty (heat, co2, elec_hi+elec_low, irr) 
        r = np.array(resource_consumption, dtype=float).reshape(-1)  # (5,)
        elec_sum = r[2] + r[3] if r.size >= 4 else 0.0
        res_vec4 = np.array([r[0], r[1], elec_sum, r[4]], dtype=float)   # (4,)
        res_norm = np.divide(res_vec4, self.res_mv, out=np.zeros_like(res_vec4), where=self.res_mv!=0.0)
        resource_penalty = float(np.sum(self.res_w * res_norm))

        # Penalties / bonuses from thresholds
        punishment, bonus = 0.0, 0.0
        # caps & lows
        cap = float(self.th.get("resource_cap", 7.0))
        low = float(self.th.get("crop_any_low", 0.5))

        pen = (self.ex.get("penalties", {}) or {})
        if pen.get("any_resource_gt_cap") is not None and np.any(res_vec4 > cap):
            punishment += float(pen["any_resource_gt_cap"])
        if pen.get("any_crop_lt_crop_any_low") is not None and np.any(crop_params < low):
            punishment += float(pen["any_crop_lt_crop_any_low"])
        if pen.get("all_crops_lt_crop_any_low") is not None and np.all(crop_params < low):
            punishment += float(pen["all_crops_lt_crop_any_low"])

        bon = (self.ex.get("bonuses", {}) or {})
        for th_str, val in (bon.get("any_crop_ge", {}) or {}).items():
            if np.any(crop_params >= float(th_str)):
                bonus += float(val)
        for th_str, val in (bon.get("all_crops_ge", {}) or {}).items():
            if np.all(crop_params >= float(th_str)):
                bonus += float(val)

        # Optional extra terms
        if self.use_efficiency:
            efficiency = crop_reward / (1.0 + resource_penalty)
            bonus += self.delta * float(efficiency)

        if self.use_stability:
            # temporal-difference stability penalty
            s = 0.01
            max_delta_action = 16.0
            diffs = np.abs(current_actions[:-1] - current_actions[1:])  # (2015, 9)
            stability = float(np.sum(diffs) / max_delta_action)
            stability_penalty = s * stability
            punishment -= self.gamma * stability_penalty


        # Final reward
        reward = self.alpha * crop_reward - self.beta * resource_penalty #+ bonus + punishment
        return float(reward)


# ----------------------
# 2) Config helpers
# ----------------------
def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_action_noise(noise_cfg: Dict[str, Any], action_dim: int):
    if not noise_cfg:
        return None

    noise_type = str(noise_cfg.get("type", "")).lower()
    if noise_type == "normal":
        sigma = float(noise_cfg.get("sigma", 0.1))
        return NormalActionNoise(
            mean=np.zeros(action_dim),
            sigma=sigma * np.ones(action_dim),
        )
    if noise_type.startswith("ornstein"):
        sigma = float(noise_cfg.get("sigma", 0.2))
        theta = float(noise_cfg.get("theta", 0.15))
        return OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(action_dim),
            sigma=sigma * np.ones(action_dim),
            theta=theta,
        )
    return None


# ----------------------
# 3) Build env + model
# ----------------------
def make_env_from_config(cfg: Dict[str, Any], algo_name: str = None) -> gym.Env:
    # --- Load estimators ---
    paths = cfg.get("paths", {})
    crop_path = paths.get("crop_parameters_estimator")
    rc_path = paths.get("resource_consumption_estimator")
    ghc_path = paths.get("gh_climate_estimator")
    reward_cfg = _get_active_reward_profile(cfg)

    if not (crop_path and rc_path and ghc_path):
        raise FileNotFoundError("Estimator paths are missing in config.paths")

    crop_estimator = load_model(crop_path)
    rc_estimator = load_model(rc_path)
    ghc_estimator = load_model(ghc_path)

    # --- Load & scale weather ---
# --- Load & scale weather ---
    weather_csv = paths.get("weather_csv")
    if not weather_csv or not os.path.exists(weather_csv):
        raise FileNotFoundError(f"Weather CSV not found: {weather_csv}")

    # Try to keep original index if present, otherwise fall back gracefully
    try:
        weather_df = pd.read_csv(weather_csv, index_col="%time")
    except Exception:
        weather_df = pd.read_csv(weather_csv)

    scaler_name = (cfg.get("scaling", {}) or {}).get("weather", "MinMaxScaler")
    if scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler in config: {scaler_name}")

    weather_arr = scaler.fit_transform(weather_df.values)
    if weather_arr.shape[1] != 10:
        raise ValueError(f"Expected 10 weather features, got {weather_arr.shape[1]}")

    # --- Env params ---
    env_cfg = cfg
    max_steps = int(env_cfg.get("max_steps", 23))
    action_count = int(env_cfg.get("actions", {}).get("count", 9))
    normalized_actions = bool(env_cfg.get("actions", {}).get("normalized", True))
    days_scale = float(env_cfg.get("days_feature", {}).get("scale", 166.0))

    env = GreenhouseEnv(
        crop_parameters_estimator=crop_estimator,
        resource_consumption_estimator=rc_estimator,
        gh_climate_estimator=ghc_estimator,
        weather_data=weather_arr,
        max_steps=max_steps,
        action_count=action_count,
        normalized_actions=normalized_actions,
        days_scale=days_scale,
        reward_cfg=reward_cfg,   # <-- pass it in

    )
    # Pass the real column names to the env
    env.w_columns = list(weather_df.columns)  # << keep headers for DataFrame reconstruction

    log_path = (
        (cfg.get("logging", {}) or {}).get("monitor_log")
        or ((cfg.get(algo_name or "", {}) or {}).get("monitor_log") if algo_name else None)
        or None
    )
    env = Monitor(env, filename=log_path)
    # You can uncomment to verify spaces:
    # check_env(env, warn=True)
    return env


def build_model_from_config(algo: str, env: gym.Env, cfg: Dict[str, Any]):
    algo = algo.lower().strip()
    if algo == "td3":
        td3_cfg = cfg.get("td3", {})
        noise = build_action_noise(td3_cfg.get("action_noise", {}), action_dim=env.action_space.shape[0])
        return TD3(
            td3_cfg.get("policy", "MultiInputPolicy"),
            env,
            learning_rate=float(td3_cfg.get("learning_rate", 1e-4)),
            buffer_size=int(td3_cfg.get("buffer_size", 100000)),
            batch_size=int(td3_cfg.get("batch_size", 256)),
            tau=float(td3_cfg.get("tau", 0.005)),
            gamma=float(td3_cfg.get("gamma", 0.99)),
            train_freq=(int(td3_cfg.get("train_freq", {}).get("n", 1)), td3_cfg.get("train_freq", {}).get("unit", "episode")),
            gradient_steps=int(td3_cfg.get("gradient_steps", 100)),
            learning_starts=int(td3_cfg.get("learning_starts", 1000)),
            policy_delay=int(td3_cfg.get("policy_delay", 2)),
            target_policy_noise=float(td3_cfg.get("target_policy_noise", 0.2)),
            target_noise_clip=float(td3_cfg.get("target_noise_clip", 0.5)),
            action_noise=noise,
            policy_kwargs=td3_cfg.get("policy_kwargs", {}),
            tensorboard_log=td3_cfg.get("tensorboard_log", "./TD3_logs_fs"),
            device=td3_cfg.get("device", "auto"),
            verbose=1,
        )

    if algo == "ppo":
        ppo_cfg = cfg.get("ppo", {})
        return PPO(
            ppo_cfg.get("policy", "MultiInputPolicy"),
            env,
            learning_rate=float(ppo_cfg.get("learning_rate", 1e-5)),
            n_steps=int(ppo_cfg.get("n_steps", 2048)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            n_epochs=int(ppo_cfg.get("n_epochs", 20)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            clip_range=float(ppo_cfg.get("clip_range", 0.2)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.05)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
            vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
            tensorboard_log=ppo_cfg.get("tensorboard_log", "./PPO_logs"),
            policy_kwargs=ppo_cfg.get("policy_kwargs", {}),
            device=ppo_cfg.get("device", "auto"),
            verbose=1,
        )

    if algo == "sac":
        sac_cfg = cfg.get("sac", {})
        return SAC(
            sac_cfg.get("policy", "MultiInputPolicy"),
            env,
            learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
            buffer_size=int(sac_cfg.get("buffer_size", 10000)),
            batch_size=int(sac_cfg.get("batch_size", 16)),
            tau=float(sac_cfg.get("tau", 0.005)),
            gamma=float(sac_cfg.get("gamma", 0.99)),
            train_freq=(int(sac_cfg.get("train_freq", {}).get("n", 1)), sac_cfg.get("train_freq", {}).get("unit", "episode")),
            gradient_steps=int(sac_cfg.get("gradient_steps", 100)),
            learning_starts=int(sac_cfg.get("learning_starts", 10000)),
            ent_coef=sac_cfg.get("ent_coef", "auto"),
            target_update_interval=int(sac_cfg.get("target_update_interval", 1)),
            policy_kwargs=sac_cfg.get("policy_kwargs", {}),
            tensorboard_log=sac_cfg.get("tensorboard_log", "./SAC_logs"),
            device=sac_cfg.get("device", "cuda"),
            verbose=1,
        )

    if algo == "ddpg":
        ddpg_cfg = cfg.get("ddpg", {})
        noise = build_action_noise(ddpg_cfg.get("action_noise", {}), action_dim=env.action_space.shape[0])
        return DDPG(
            ddpg_cfg.get("policy", "MultiInputPolicy"),
            env,
            learning_rate=float(ddpg_cfg.get("learning_rate", 1e-3)),
            buffer_size=int(ddpg_cfg.get("buffer_size", 10000)),
            batch_size=int(ddpg_cfg.get("batch_size", 128)),
            tau=float(ddpg_cfg.get("tau", 0.005)),
            gamma=float(ddpg_cfg.get("gamma", 0.98)),
            train_freq=(int(ddpg_cfg.get("train_freq", {}).get("n", 1)), ddpg_cfg.get("train_freq", {}).get("unit", "episode")),
            gradient_steps=int(ddpg_cfg.get("gradient_steps", 100)),
            learning_starts=int(ddpg_cfg.get("learning_starts", 100)),
            action_noise=noise,
            policy_kwargs=ddpg_cfg.get("policy_kwargs", {}),
            tensorboard_log=ddpg_cfg.get("tensorboard_log", "./DDPG_logs"),
            device=ddpg_cfg.get("device", "cuda"),
            verbose=1,
        )

    raise ValueError(f"Unknown algo '{algo}'. Use one of: td3, ppo, sac, ddpg.")


def load_env_and_model(config_path: str, algo: str) -> Tuple[gym.Env, Any]:
    cfg = load_yaml_config(config_path)
    env = make_env_from_config(cfg,algo)
    model = build_model_from_config(algo, env, cfg)
    return env, model


# ----------------------
# 4) Example CLI usage
# ----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load config and initialize SB3 model for GreenhouseEnv.")
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to YAML config")
    parser.add_argument("--algo", type=str, default="td3", choices=["td3", "ppo", "sac", "ddpg"], help="Algorithm")
    args = parser.parse_args()

    env, model = load_env_and_model(args.config, args.algo)
    print(f"Environment and {args.algo.upper()} model initialized.")
    # Example: short warm-up (comment out if you only want to init)
    # model.learn(total_timesteps=1000)

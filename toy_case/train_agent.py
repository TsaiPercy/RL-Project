"""SB3 PPO Agent 訓練腳本 — DoorKeyEnv(room_size=15)。

Per SPEC §5.4.1 Toy Case Agent Pool:
  - toy_strong_0: PPO 訓練至收斂 (~1M steps), success rate > 90%
  - toy_weak_0: PPO 提前停止 (~50K steps), success rate ~30-60%

Per SPEC §12 Experiment T:
  train_ppo("MiniGrid-DoorKey-15x15-v0", total_timesteps=1_000_000) → toy_strong_0.zip
  train_ppo("MiniGrid-DoorKey-15x15-v0", total_timesteps=50_000)    → toy_weak_0.zip

Usage:
  python -m toy_case.train_agent --agent strong
  python -m toy_case.train_agent --agent weak
  python -m toy_case.train_agent --agent both
"""

from __future__ import annotations

# Set project-local cache dirs before any ML library is imported.
from shared.env_setup import setup_project_cache
setup_project_cache()

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_CHECKPOINT_DIR = "checkpoints/agents"
DEFAULT_ENV_ID = "MiniGrid-DoorKey-15x15-v0"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """載入 YAML 配置。

    Args:
        config_path: 配置檔路徑。

    Returns:
        配置 dict。
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_id: str) -> callable:
    """建立 MiniGrid 環境工廠函數。

    Args:
        env_id: Gymnasium 環境 ID。

    Returns:
        環境建立函式（供 DummyVecEnv / SubprocVecEnv 使用）。
    """
    import gymnasium as gym
    import minigrid  # noqa: F401 — 註冊 MiniGrid 環境

    def _init() -> gym.Env:
        env = gym.make(env_id)
        from minigrid.wrappers import ImgObsWrapper
        env = ImgObsWrapper(env)
        return env

    return _init


def train_ppo(
    env_id: str,
    total_timesteps: int,
    save_path: str,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    seed: Optional[int] = 42,
    n_envs: int = 4,
) -> None:
    """訓練 SB3 PPO agent 並儲存。

    Args:
        env_id: Gymnasium 環境 ID。
        total_timesteps: 總訓練步數。
        save_path: 模型儲存路徑（不含 .zip 後綴）。
        learning_rate: PPO 學習率。
        n_steps: 每次 rollout 的步數。
        batch_size: Mini-batch 大小。
        n_epochs: PPO epoch 數。
        gamma: Discount factor。
        ent_coef: Entropy coefficient。
        seed: Random seed。
        n_envs: 平行環境數量。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    logger.info(
        "[train_agent] 開始訓練 | env=%s, timesteps=%d, save_path=%s",
        env_id, total_timesteps, save_path,
    )

    train_env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(env_id)])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path=os.path.join("logs", "toy_case"),
        eval_freq=max(total_timesteps // 20, 1000),
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)

    train_env.close()
    eval_env.close()

    logger.info("[train_agent] 訓練完成，模型儲存至: %s.zip", save_path)


def evaluate_agent(
    model_path: str,
    env_id: str,
    n_episodes: int = 100,
) -> dict:
    """評估已訓練的 agent。

    Args:
        model_path: 模型路徑（含或不含 .zip）。
        env_id: Gymnasium 環境 ID。
        n_episodes: 評估 episode 數。

    Returns:
        dict 包含 success_rate, mean_return, mean_length。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    eval_env = DummyVecEnv([make_env(env_id)])
    model = PPO.load(model_path, env=eval_env)

    successes = 0
    total_returns: list[float] = []
    total_lengths: list[int] = []

    obs = eval_env.reset()
    for _ in range(n_episodes):
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_return += reward[0]
            episode_length += 1

            if done[0]:
                if episode_return > 0:
                    successes += 1
                total_returns.append(episode_return)
                total_lengths.append(episode_length)
                obs = eval_env.reset()
                break

    eval_env.close()

    success_rate = successes / n_episodes
    mean_return = sum(total_returns) / len(total_returns) if total_returns else 0.0
    mean_length = sum(total_lengths) / len(total_lengths) if total_lengths else 0.0

    results = {
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_length": mean_length,
        "n_episodes": n_episodes,
    }

    logger.info(
        "[evaluate_agent] %s | success_rate=%.2f%%, mean_return=%.4f, mean_length=%.1f",
        model_path, success_rate * 100, mean_return, mean_length,
    )
    return results


def main() -> None:
    """主入口：依照 CLI 參數訓練 strong/weak/both agent。"""
    parser = argparse.ArgumentParser(
        description="Train SB3 PPO agents for toy case (Phase 0).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["strong", "weak", "both"],
        default="both",
        help="Which agent to train: strong, weak, or both.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save agent checkpoints.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    toy_config = config.get("toy_case", {})
    ppo_params = toy_config.get("ppo_hyperparams", {})

    env_id = toy_config.get("train_env", DEFAULT_ENV_ID)
    strong_steps = toy_config.get("strong_agent_steps", 1_000_000)
    weak_steps = toy_config.get("weak_agent_steps", 50_000)

    strong_path = os.path.join(args.checkpoint_dir, "toy_strong_0")
    weak_path = os.path.join(args.checkpoint_dir, "toy_weak_0")

    if not args.evaluate_only:
        if args.agent in ("strong", "both"):
            logger.info("=== 訓練 toy_strong_0 (%d steps) ===", strong_steps)
            train_ppo(
                env_id=env_id,
                total_timesteps=strong_steps,
                save_path=strong_path,
                learning_rate=ppo_params.get("learning_rate", 3e-4),
                n_steps=ppo_params.get("n_steps", 2048),
                batch_size=ppo_params.get("batch_size", 64),
                n_epochs=ppo_params.get("n_epochs", 10),
                gamma=ppo_params.get("gamma", 0.99),
                ent_coef=ppo_params.get("ent_coef", 0.01),
            )

        if args.agent in ("weak", "both"):
            logger.info("=== 訓練 toy_weak_0 (%d steps) ===", weak_steps)
            train_ppo(
                env_id=env_id,
                total_timesteps=weak_steps,
                save_path=weak_path,
                learning_rate=ppo_params.get("learning_rate", 3e-4),
                n_steps=ppo_params.get("n_steps", 2048),
                batch_size=ppo_params.get("batch_size", 64),
                n_epochs=ppo_params.get("n_epochs", 10),
                gamma=ppo_params.get("gamma", 0.99),
                ent_coef=ppo_params.get("ent_coef", 0.01),
            )

    logger.info("=== 評估 agent 表現 ===")
    if args.agent in ("strong", "both"):
        if os.path.exists(strong_path + ".zip"):
            strong_results = evaluate_agent(strong_path, env_id)
            print(f"\ntoy_strong_0 results: {strong_results}")
        else:
            logger.warning("toy_strong_0 checkpoint 不存在: %s.zip", strong_path)

    if args.agent in ("weak", "both"):
        if os.path.exists(weak_path + ".zip"):
            weak_results = evaluate_agent(weak_path, env_id)
            print(f"\ntoy_weak_0 results: {weak_results}")
        else:
            logger.warning("toy_weak_0 checkpoint 不存在: %s.zip", weak_path)


if __name__ == "__main__":
    main()

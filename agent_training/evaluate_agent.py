"""Evaluate a trained curriculum agent across BabyAI GoTo-family environments.

Per SPEC §5.4: validates that strong_0 > 80% win rate and that the
strong/weak gap is clearly visible before using agents as reward signals.

Usage:
  python -m agent_training.evaluate_agent \
      --agent-path checkpoints/agents/strong_0 \
      --envs all \
      --n-episodes 100

  python -m agent_training.evaluate_agent \
      --agent-path checkpoints/agents/weak_0 \
      --envs first3 \
      --n-episodes 100

  # Evaluate on a specific subset:
  python -m agent_training.evaluate_agent \
      --agent-path checkpoints/agents/strong_0 \
      --envs BabyAI-GoTo-v0,BabyAI-GoToObjMaze-v0 \
      --n-episodes 50
"""

from __future__ import annotations

from shared.env_setup import setup_project_cache

setup_project_cache()

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml

# Route BabyAI's rejection-sampling stdout chatter through logging (DEBUG).
# Must be called before any env is constructed.
from agent_training.baby_ai_silence import silence_baby_ai_rejection_logs

silence_baby_ai_rejection_logs()

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_N_EPISODES = 100


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate_agent(
    model_path: str,
    env_ids: list[str],
    n_episodes: int,
    mission_max_len: int,
    deterministic: bool = True,
) -> dict[str, dict]:
    """Evaluate an SB3 PPO agent on a list of BabyAI environments.

    For each env, runs ``n_episodes`` episodes and records success rate,
    mean return, and mean episode length.  A success is any episode with
    return > 0 (MiniGrid sparse reward: +1 on goal, 0 otherwise).

    Per SPEC §5.4 [impl-updated], envs use their native default sizes; no
    ``room_size`` override is applied.  Each env is wrapped with
    ``MissionTokenizer`` so the agent receives the same dict observation
    it was trained on.

    Args:
        model_path: Path to saved SB3 model (with or without .zip suffix).
        env_ids: List of Gymnasium env IDs to evaluate on.
        n_episodes: Number of evaluation episodes per environment.
        mission_max_len: Token sequence length used by ``MissionTokenizer``.
            Must match the value used during training.
        deterministic: Whether to use deterministic actions.

    Returns:
        Dict mapping env_id → {"success_rate", "mean_return", "mean_length",
        "n_episodes"}.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agent_training.train_curriculum import make_env_fn

    model_path_str = str(model_path)
    if not model_path_str.endswith(".zip"):
        model_path_str = model_path_str + ".zip"

    assert Path(model_path_str).exists(), (
        f"Checkpoint not found: {model_path_str}"
    )

    # Load model without binding to any env yet.
    model = PPO.load(model_path_str)

    results: dict[str, dict] = {}

    for env_id in env_ids:
        eval_env = DummyVecEnv([make_env_fn(env_id, mission_max_len)])
        model.set_env(eval_env)

        successes = 0
        total_returns: list[float] = []
        total_lengths: list[int] = []

        obs = eval_env.reset()
        episodes_done = 0
        episode_return = 0.0
        episode_length = 0

        while episodes_done < n_episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = eval_env.step(action)

            episode_return += float(reward[0])
            episode_length += 1

            if done[0]:
                if episode_return > 0.0:
                    successes += 1
                total_returns.append(episode_return)
                total_lengths.append(episode_length)
                episodes_done += 1
                episode_return = 0.0
                episode_length = 0
                obs = eval_env.reset()

        eval_env.close()

        success_rate = successes / n_episodes
        mean_return = float(sum(total_returns) / len(total_returns)) if total_returns else 0.0
        mean_length = float(sum(total_lengths) / len(total_lengths)) if total_lengths else 0.0

        results[env_id] = {
            "success_rate": success_rate,
            "mean_return": mean_return,
            "mean_length": mean_length,
            "n_episodes": n_episodes,
        }

        logger.info(
            "[evaluate_agent] %s | success=%.1f%% | mean_return=%.4f | mean_len=%.1f",
            env_id,
            success_rate * 100,
            mean_return,
            mean_length,
        )

    return results


def _print_results_table(
    agent_path: str,
    results: dict[str, dict],
) -> None:
    """Print a formatted summary table of evaluation results.

    Args:
        agent_path: Path shown in the header.
        results: Output of ``evaluate_agent()``.
    """
    col_w = 36
    print(f"\n{'=' * 70}")
    print(f"Agent: {agent_path}")
    print(f"{'=' * 70}")
    print(
        f"{'Environment':<{col_w}} {'Success%':>9} {'MeanReturn':>11} {'MeanLen':>9}"
    )
    print("-" * 70)
    for env_id, stats in results.items():
        print(
            f"{env_id:<{col_w}} "
            f"{stats['success_rate'] * 100:>8.1f}% "
            f"{stats['mean_return']:>11.4f} "
            f"{stats['mean_length']:>9.1f}"
        )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Config / env-list helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed config dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_env_ids(envs_arg: str, curriculum: list[dict]) -> list[str]:
    """Resolve the --envs CLI argument to a concrete list of env IDs.

    Args:
        envs_arg: One of ``"all"``, ``"first3"``, or a comma-separated list
            of env IDs.
        curriculum: Ordered list of curriculum env configs from config YAML.

    Returns:
        List of resolved env ID strings.
    """
    if envs_arg == "all":
        return [c["env"] for c in curriculum]
    if envs_arg == "first3":
        return [c["env"] for c in curriculum[:3]]
    # Treat as comma-separated list of explicit env IDs.
    return [e.strip() for e in envs_arg.split(",") if e.strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained curriculum agent across BabyAI GoTo-family "
            "environments.  Per SPEC §5.4."
        )
    )
    parser.add_argument(
        "--agent-path",
        type=str,
        required=True,
        help="Path to saved SB3 checkpoint (with or without .zip).",
    )
    parser.add_argument(
        "--envs",
        type=str,
        default="all",
        help=(
            "Which envs to evaluate on: 'all', 'first3', or a comma-separated "
            "list of env IDs.  Default: all."
        ),
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_EPISODES,
        help=f"Episodes per environment (default: {DEFAULT_N_EPISODES}).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config (used to resolve env list).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for agent evaluation."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    at_cfg = config["agent_training"]
    curriculum: list[dict] = at_cfg["curriculum"]
    mission_max_len: int = int(at_cfg["mission_max_len"])

    env_ids = resolve_env_ids(args.envs, curriculum)
    assert env_ids, f"No environments resolved from --envs '{args.envs}'"

    logger.info(
        "Evaluating %s on %d envs (%d episodes each)",
        args.agent_path,
        len(env_ids),
        args.n_episodes,
    )

    results = evaluate_agent(
        model_path=args.agent_path,
        env_ids=env_ids,
        n_episodes=args.n_episodes,
        mission_max_len=mission_max_len,
        deterministic=not args.stochastic,
    )

    _print_results_table(args.agent_path, results)


if __name__ == "__main__":
    main()

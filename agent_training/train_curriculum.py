"""BabyAI GoTo-family curriculum training for Phase 1-2 agent pool.

Per SPEC §5.4 (Agent Pool) and §11.4 (Pseudo Config):
  - strong_0: all 7 curriculum levels, effective_threshold = base + 0.05
  - weak_0:   first 3 curriculum levels, effective_threshold = base - 0.25
  - Held-out variants (strong_held_0, weak_held_0) use different seeds.

[impl-updated AT-4.x] Phase A→B observation + policy pipeline:
  Each BabyAI env is wrapped with ``MissionTokenizer`` which exposes a
  Dict obs ``{image: (7,7,3) int64, direction: (1,) int64,
  mission: (L,) int64}``.  Phase B policy is
  ``RecurrentPPO("MultiInputLstmPolicy")`` (sb3_contrib) with
  ``BabyAIDictExtractor`` (CNN + token mean-pool + direction embedding
  fusion) as the features extractor; an LSTM layer sits between the
  extractor output and the policy/value heads, providing cross-step
  memory for partial-obs maze navigation.

Usage:
  python -m agent_training.train_curriculum --agent strong
  python -m agent_training.train_curriculum --agent weak
  python -m agent_training.train_curriculum --agent strong --seed 99 --agent-id strong_held_0
"""

from __future__ import annotations

# Must come before any ML import — sets project-local cache dirs.
from shared.env_setup import setup_project_cache

setup_project_cache()

import argparse
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml

# Route BabyAI's rejection-sampling stdout chatter through logging (DEBUG).
# Must be called before any env is constructed.
from agent_training.baby_ai_silence import silence_baby_ai_rejection_logs

silence_baby_ai_rejection_logs()

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/default.yaml"
DEFAULT_CHECKPOINT_DIR = "checkpoints/agents"
DEFAULT_N_ENVS = 4
DEFAULT_EVAL_EPISODES = 50
EVAL_CHUNK_STEPS = 100_000


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env_fn(env_id: str, mission_max_len: int) -> Callable:
    """Return a zero-arg factory for a tokenized BabyAI env.

    Each env is constructed with its native default size (no ``room_size``
    override) and wrapped with :class:`MissionTokenizer` so the mission
    string and direction scalar survive into the policy. Per SPEC §5.4
    [impl-updated].

    Args:
        env_id: Gymnasium environment ID (e.g. "BabyAI-GoTo-v0").
        mission_max_len: Token sequence length for the mission field.

    Returns:
        Zero-arg callable that creates and wraps the environment.
    """
    import gymnasium as gym
    import minigrid  # noqa: F401 — registers MiniGrid + BabyAI envs

    from agent_training.wrappers import MissionTokenizer

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env = MissionTokenizer(env, max_len=mission_max_len)
        return env

    return _init


# ---------------------------------------------------------------------------
# CurriculumTrainer
# ---------------------------------------------------------------------------


class CurriculumTrainer:
    """Trains a single SB3 PPO agent through a BabyAI GoTo-family curriculum.

    The agent advances to the next curriculum level once its success rate
    on the current level meets ``effective_threshold = base + success_increase``.
    Training stops when all ``curriculum_levels`` are completed or the global
    ``max_timesteps`` budget is exhausted.

    Args:
        config: Full project config dict (from default.yaml).
        agent_type: ``"strong"`` or ``"weak"``.
        agent_id: Checkpoint name (e.g. ``"strong_0"``).
        checkpoint_dir: Directory to save checkpoints.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        config: dict,
        agent_type: str,
        agent_id: str,
        checkpoint_dir: str,
        seed: int,
    ) -> None:
        assert agent_type in ("strong", "weak"), (
            f"agent_type must be 'strong' or 'weak', got '{agent_type}'"
        )

        self._agent_id = agent_id
        self._checkpoint_dir = Path(checkpoint_dir)
        self._seed = seed

        at_cfg = config["agent_training"]
        agent_cfg = at_cfg[f"{agent_type}_agent"]

        self._curriculum: list[dict] = at_cfg["curriculum"]
        self._curriculum_levels: int = int(agent_cfg["curriculum_levels"])
        self._success_increase: float = float(agent_cfg["success_increase"])
        self._max_timesteps: int = int(agent_cfg["max_timesteps"])
        self._eval_episodes: int = int(
            at_cfg.get("eval_episodes", DEFAULT_EVAL_EPISODES)
        )
        self._eval_chunk_steps: int = int(
            at_cfg.get("eval_chunk_steps", EVAL_CHUNK_STEPS)
        )

        ppo = at_cfg.get("ppo_hyperparams", {})
        self._lr: float = float(ppo.get("learning_rate", 3e-4))
        self._n_steps: int = int(ppo.get("n_steps", 2048))
        self._batch_size: int = int(ppo.get("batch_size", 64))
        self._n_epochs: int = int(ppo.get("n_epochs", 10))
        self._gamma: float = float(ppo.get("gamma", 0.99))
        self._ent_coef: float = float(ppo.get("ent_coef", 0.01))
        self._features_dim: int = int(ppo["features_dim"])
        # Phase B LSTM hyperparams [impl-updated AT-B1].
        self._lstm_hidden_size: int = int(ppo.get("lstm_hidden_size", 256))
        self._n_lstm_layers: int = int(ppo.get("n_lstm_layers", 1))
        self._enable_critic_lstm: bool = bool(ppo.get("enable_critic_lstm", True))

        # Phase A dict-obs / mission-encoding parameters [impl-updated AT-4.x].
        from agent_training.wrappers import BABYAI_VOCAB

        self._mission_max_len: int = int(at_cfg["mission_max_len"])
        self._vocab_size: int = int(at_cfg["vocab_size"])
        self._text_embed_dim: int = int(at_cfg["text_embed_dim"])
        self._dir_embed_dim: int = int(at_cfg["dir_embed_dim"])
        # AT-4.5: per-channel symbolic-image embedding dims.
        self._obj_embed_dim: int = int(at_cfg["obj_embed_dim"])
        self._color_embed_dim: int = int(at_cfg["color_embed_dim"])
        self._state_embed_dim: int = int(at_cfg["state_embed_dim"])
        assert self._vocab_size == len(BABYAI_VOCAB), (
            f"config vocab_size ({self._vocab_size}) != len(BABYAI_VOCAB) "
            f"({len(BABYAI_VOCAB)}); update one to match the other"
        )

        assert self._curriculum_levels <= len(self._curriculum), (
            f"curriculum_levels ({self._curriculum_levels}) > "
            f"number of curriculum envs ({len(self._curriculum)})"
        )

        logger.info(
            "[CurriculumTrainer] agent_id=%s, type=%s, levels=%d/%d, "
            "success_increase=%.2f, max_timesteps=%d, seed=%d",
            agent_id,
            agent_type,
            self._curriculum_levels,
            len(self._curriculum),
            self._success_increase,
            self._max_timesteps,
            seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_level_preview(self, env_id: str, level_idx: int) -> None:
        """Render the first reset of env_id and save it as a PNG.

        Creates a temporary env with render_mode="rgb_array" (separate from
        the training vec-env) so the snapshot is always a fresh reset.
        Output path: {checkpoint_dir}/{agent_id}_level{N}_preview.png.

        Args:
            env_id: BabyAI gymnasium env ID.
            level_idx: Zero-based index of this curriculum level.
        """
        import gymnasium as gym

        # minigrid env registration is guaranteed by this point — caller must
        # invoke after _make_vec_env() so the import side-effect has run.
        try:
            from PIL import Image
        except ImportError:
            logger.warning("[train] Pillow not installed — skipping level preview.")
            return

        out_path = (
            self._checkpoint_dir
            / f"{self._agent_id}_level{level_idx + 1}_preview.png"
        )
        try:
            env = gym.make(env_id, render_mode="rgb_array")
            env.reset()
            rgb = env.render()  # (H, W, 3) uint8
            env.close()
            Image.fromarray(rgb).save(out_path)
            logger.info("[train] Level preview saved: %s", out_path)
        except Exception as exc:
            logger.warning(
                "[train] Could not save level preview for %s: %s", env_id, exc
            )


    def _eval_success_rate(
        self,
        model,  # RecurrentPPO instance — not typed to avoid import at module level
        env_id: str,
        n_episodes: int,
    ) -> float:
        """Evaluate ``model`` on ``env_id`` and return success rate.

        Success is defined as episode return > 0 (consistent with MiniGrid's
        sparse reward: +1 for reaching goal, 0 otherwise).

        Tracks LSTM hidden states across steps within each episode and resets
        them at episode boundaries via the ``episode_start`` flag, as required
        by ``RecurrentPPO.predict()`` [impl-updated AT-B2].

        Args:
            model: Trained sb3_contrib RecurrentPPO model.
            env_id: BabyAI gymnasium env ID to evaluate on.
            n_episodes: Number of evaluation episodes.

        Returns:
            Fraction of successful episodes in [0, 1].
        """
        from stable_baselines3.common.vec_env import DummyVecEnv

        eval_env = DummyVecEnv([make_env_fn(env_id, self._mission_max_len)])

        successes = 0
        obs = eval_env.reset()
        episodes_done = 0
        lstm_states = None  # (h, c) tuple; None triggers zero-init in RecurrentPPO
        episode_starts = np.ones((1,), dtype=bool)  # (n_envs=1,) — True on first step

        while episodes_done < n_episodes:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, reward, done, _info = eval_env.step(action)
            episode_starts = done  # RecurrentPPO zeros LSTM state when True
            if done[0]:
                if float(reward[0]) > 0.0:
                    successes += 1
                episodes_done += 1
                obs = eval_env.reset()

        eval_env.close()
        success_rate = successes / n_episodes
        return success_rate

    def _make_vec_env(self, env_id: str, n_envs: int):
        """Create a SubprocVecEnv for training.

        Args:
            env_id: BabyAI gymnasium env ID.
            n_envs: Number of parallel workers.

        Returns:
            SubprocVecEnv instance.
        """
        from stable_baselines3.common.vec_env import SubprocVecEnv

        return SubprocVecEnv(
            [make_env_fn(env_id, self._mission_max_len) for _ in range(n_envs)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, n_envs: int = DEFAULT_N_ENVS) -> str:
        """Run curriculum training and save the final checkpoint.

        Iterates through ``curriculum[:curriculum_levels]``.  For each level:
        1. Train in chunks of ``eval_chunk_steps`` steps.
        2. Evaluate success rate every chunk.
        3. Advance to the next level when ``rate >= effective_threshold``.
        4. Stop early if ``max_timesteps`` budget is exhausted.

        An intermediate checkpoint is saved after each level is cleared.
        The final checkpoint is saved at ``checkpoint_dir/{agent_id}.zip``.

        Args:
            n_envs: Number of parallel training environments.

        Returns:
            Path to the saved checkpoint (without ``.zip`` suffix).
        """
        # Phase B: RecurrentPPO from sb3_contrib [impl-updated AT-B2].
        from sb3_contrib import RecurrentPPO

        from agent_training.extractors import BabyAIDictExtractor

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model: Optional[RecurrentPPO] = None
        total_steps: int = 0
        first_learn_call: bool = True

        active_levels = self._curriculum[: self._curriculum_levels]

        for level_idx, level_cfg in enumerate(active_levels):
            env_id: str = level_cfg["env"]
            base_threshold: float = float(level_cfg["success_threshold"])
            effective_threshold: float = float(
                np.clip(base_threshold + self._success_increase, 0.0, 1.0)
            )

            logger.info(
                "[train] Level %d/%d: %s | effective_threshold=%.2f",
                level_idx + 1,
                self._curriculum_levels,
                env_id,
                effective_threshold,
            )

            train_env = self._make_vec_env(env_id, n_envs)
            self._save_level_preview(env_id, level_idx)

            if model is None:
                model = RecurrentPPO(
                    policy="MultiInputLstmPolicy",
                    env=train_env,
                    learning_rate=self._lr,
                    n_steps=self._n_steps,
                    batch_size=self._batch_size,
                    n_epochs=self._n_epochs,
                    gamma=self._gamma,
                    ent_coef=self._ent_coef,
                    seed=self._seed,
                    verbose=0,
                    policy_kwargs={
                        "features_extractor_class": BabyAIDictExtractor,
                        "features_extractor_kwargs": {
                            "features_dim": self._features_dim,
                            "vocab_size": self._vocab_size,
                            "text_embed_dim": self._text_embed_dim,
                            "dir_embed_dim": self._dir_embed_dim,
                            "obj_embed_dim": self._obj_embed_dim,
                            "color_embed_dim": self._color_embed_dim,
                            "state_embed_dim": self._state_embed_dim,
                        },
                        "lstm_hidden_size": self._lstm_hidden_size,
                        "n_lstm_layers": self._n_lstm_layers,
                        "enable_critic_lstm": self._enable_critic_lstm,
                    },
                )
            else:
                model.set_env(train_env)

            level_cleared = False

            while total_steps < self._max_timesteps:
                remaining = self._max_timesteps - total_steps
                chunk = min(self._eval_chunk_steps, remaining)

                model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=first_learn_call,
                )
                first_learn_call = False
                total_steps += chunk

                success_rate = self._eval_success_rate(
                    model, env_id, self._eval_episodes
                )
                logger.info(
                    "[train] Level %d (%s) | steps=%d | success=%.1f%% | threshold=%.1f%%",
                    level_idx + 1,
                    env_id,
                    total_steps,
                    success_rate * 100,
                    effective_threshold * 100,
                )

                if success_rate >= effective_threshold:
                    logger.info(
                        "[train] Level %d cleared! Moving to next level.",
                        level_idx + 1,
                    )
                    level_cleared = True
                    break

            train_env.close()

            # Save intermediate checkpoint regardless of whether cleared.
            intermediate_path = str(
                self._checkpoint_dir / f"{self._agent_id}_level{level_idx + 1}"
            )
            model.save(intermediate_path)
            logger.info(
                "[train] Saved intermediate checkpoint: %s.zip", intermediate_path
            )

            if not level_cleared:
                logger.warning(
                    "[train] Max timesteps (%d) exhausted at level %d (%s). "
                    "Agent may not have met the threshold.",
                    self._max_timesteps,
                    level_idx + 1,
                    env_id,
                )
                break

        assert model is not None, "No training occurred — curriculum may be empty."

        final_path = str(self._checkpoint_dir / self._agent_id)
        model.save(final_path)
        logger.info("[train] Training complete. Final checkpoint: %s.zip", final_path)
        return final_path


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load YAML config file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Parsed config dict.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BabyAI GoTo-family curriculum training for Phase 1-2 agent pool. "
            "Per SPEC §5.4."
        )
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["strong", "weak"],
        required=True,
        help="Agent type: 'strong' (all 7 levels, high threshold) or "
        "'weak' (first 3 levels, low threshold).",
    )
    parser.add_argument(
        "--agent-id",
        type=str,
        default=None,
        help="Checkpoint name (default: strong_0 or weak_0). "
        "Use e.g. strong_held_0 for held-out agents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Use a different seed for held-out agents.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory to save agent checkpoints.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help="Number of parallel training environments.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for curriculum training."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    agent_id = args.agent_id or f"{args.agent}_0"
    config = load_config(args.config)

    logger.info(
        "=== Curriculum Training: %s (agent_id=%s, seed=%d) ===",
        args.agent,
        agent_id,
        args.seed,
    )

    trainer = CurriculumTrainer(
        config=config,
        agent_type=args.agent,
        agent_id=agent_id,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    final_path = trainer.train(n_envs=args.n_envs)
    print(f"\nTraining complete. Checkpoint saved to: {final_path}.zip")


if __name__ == "__main__":
    main()

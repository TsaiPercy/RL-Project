"""BabyAI observation wrappers.

This module exposes ``MissionTokenizer``, a ``gym.ObservationWrapper`` that
converts the raw BabyAI dict observation::

    {"image": uint8 (7, 7, 3), "direction": int scalar, "mission": str}

into a tokenized dict observation suitable for SB3's ``MultiInputPolicy``::

    {"image":     uint8 (7, 7, 3),
     "direction": int64 (1,)         # value in {0, 1, 2, 3}
     "mission":   int64 (max_len,)}  # padded with [PAD]=0

Per SPEC §5.4 [impl-updated]: the prior ``ImgObsWrapper`` discarded the mission
string, leaving the agent unable to identify the goal object on
``BabyAI-GoTo-v0``.  Wrapping with ``MissionTokenizer`` restores that signal.

The vocabulary is hardcoded (see ``BABYAI_VOCAB``) — sufficient for the 9-stage
GoTo curriculum defined in ``config/default.yaml``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Hardcoded vocabulary for BabyAI GoTo-family missions.
#
# Source: mission templates in ``minigrid/envs/babyai/goto.py`` —
#   "go to [the|a] {color} {object}"
# with ``color ∈ {red, green, blue, purple, yellow, grey}`` and
# ``object ∈ {ball, box, key, door}``.
#
# Token order is load-bearing: id 0 must be ``[PAD]`` (used as
# ``padding_idx`` in the embedding) and id 1 is ``[UNK]`` (fallback for
# any out-of-vocab word).  Total = 16 tokens.
BABYAI_VOCAB: tuple[str, ...] = (
    "[PAD]",
    "[UNK]",
    "go",
    "to",
    "the",
    "a",
    "red",
    "green",
    "blue",
    "purple",
    "yellow",
    "grey",
    "ball",
    "box",
    "key",
    "door",
)

PAD_ID: int = 0
UNK_ID: int = 1


class MissionTokenizer(gym.ObservationWrapper):
    """Tokenize the BabyAI mission string into a fixed-length int64 array.

    The wrapper preserves the underlying ``image`` array verbatim and
    promotes the scalar ``direction`` to a length-1 int64 vector so the
    downstream Dict observation has a consistent ndarray shape per key
    (required by SB3's ``CombinedExtractor`` / ``MultiInputPolicy``).

    Args:
        env: A BabyAI / MiniGrid env whose observation space is a Dict
            containing at least the keys ``image``, ``direction``, and
            ``mission``.
        max_len: Token sequence length.  Missions longer than this are
            truncated; shorter missions are right-padded with ``PAD_ID``.

    The new observation space is::

        Dict({
            "image":     Box(uint8, (H, W, C)),
            "direction": Box(int64, (1,),       low=0, high=3),
            "mission":   Box(int64, (max_len,), low=0, high=vocab_size-1),
        })
    """

    def __init__(self, env: gym.Env, max_len: int) -> None:
        super().__init__(env)
        assert max_len > 0, f"max_len must be positive, got {max_len}"

        self._max_len: int = max_len
        self._token_id: dict[str, int] = {
            word: idx for idx, word in enumerate(BABYAI_VOCAB)
        }

        base_space = env.observation_space
        assert isinstance(base_space, spaces.Dict), (
            f"MissionTokenizer expects a Dict observation space, got {type(base_space)}"
        )
        for key in ("image", "direction", "mission"):
            assert key in base_space.spaces, (
                f"MissionTokenizer expects key '{key}' in observation space; "
                f"got {list(base_space.spaces.keys())}"
            )

        vocab_size: int = len(BABYAI_VOCAB)
        self.observation_space = spaces.Dict(
            {
                "image": base_space["image"],  # (H, W, C) uint8 — preserved
                "direction": spaces.Box(
                    low=0, high=3, shape=(1,), dtype=np.int64
                ),
                "mission": spaces.Box(
                    low=0,
                    high=vocab_size - 1,
                    shape=(max_len,),
                    dtype=np.int64,
                ),
            }
        )

    def observation(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Transform a single raw BabyAI dict obs into the tokenized form.

        Args:
            obs: Raw dict with keys ``image`` (uint8 ndarray),
                ``direction`` (int scalar), and ``mission`` (str).

        Returns:
            Dict with keys ``image`` (unchanged), ``direction`` (int64
            shape (1,)), ``mission`` (int64 shape (max_len,)).
        """
        mission_str: str = str(obs["mission"]).lower()
        words = mission_str.split()
        ids = [self._token_id.get(w, UNK_ID) for w in words]
        ids = ids[: self._max_len]
        ids = ids + [PAD_ID] * (self._max_len - len(ids))

        return {
            "image": obs["image"],  # (H, W, C) uint8
            "direction": np.asarray([int(obs["direction"])], dtype=np.int64),  # (1,)
            "mission": np.asarray(ids, dtype=np.int64),  # (max_len,)
        }

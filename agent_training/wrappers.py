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

# Bounds for BabyAI's symbolic image observation.
#
# BabyAI's ``image`` is NOT an RGB rendering — each cell is encoded as
# ``(object_type_id, color_id, state_id)``. Letting SB3 detect this as a
# ``uint8`` image and divide by 255 crushes every value to 0.000-0.024,
# starving the CNN of spatial signal. Setting the wrapped obs dtype to
# ``int64`` (instead of ``uint8``) escapes SB3's image-space heuristic so
# ``preprocess_obs`` only casts to float without normalising, and
# ``VecTransposeImage`` is not auto-applied.
#
# Bounds are taken from minigrid as of v3.0.0 with a small headroom buffer
# so a future minigrid upgrade adding a new object type or door state
# doesn't silently corrupt the embedding lookup:
#   minigrid.core.constants.OBJECT_TO_IDX has 11 entries (max id 10)
#   minigrid.core.constants.COLOR_TO_IDX  has 6  entries (max id 5)
#   door states are {open=0, closed=1, locked=2} (max id 2)
NUM_OBJECT_TYPES: int = 12  # minigrid OBJECT_TO_IDX (11) + 1 headroom
NUM_IMAGE_COLORS: int = 8  # minigrid COLOR_TO_IDX (6) + 2 headroom
NUM_OBJECT_STATES: int = 4  # door states (3) + 1 headroom


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
            "image":     Box(int64, (H, W, C),  low=0, high=max_id),
            "direction": Box(int64, (1,),       low=0, high=3),
            "mission":   Box(int64, (max_len,), low=0, high=vocab_size-1),
        })

    Note: ``image`` is intentionally exposed as ``int64`` (not ``uint8``)
    so SB3 does NOT classify it as an image space and skips both the
    ``VecTransposeImage`` auto-wrap and the ``/255`` normalisation in
    ``preprocess_obs``.  The downstream extractor is expected to treat
    each channel as a categorical id and embed it (per ``NUM_OBJECT_TYPES``,
    ``NUM_IMAGE_COLORS``, ``NUM_OBJECT_STATES`` declared at module top).
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
        # Reshape image space to int64 so SB3 stops treating it as an
        # RGB image (avoiding /255 normalisation that crushes symbolic
        # ids).  Cast happens inside ``observation()``.
        max_image_id: int = max(
            NUM_OBJECT_TYPES, NUM_IMAGE_COLORS, NUM_OBJECT_STATES
        ) - 1
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=max_image_id,
                    shape=base_space["image"].shape,  # (H, W, C)
                    dtype=np.int64,
                ),
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
            "image": obs["image"].astype(np.int64),  # (H, W, C) int64 (was uint8)
            "direction": np.asarray([int(obs["direction"])], dtype=np.int64),  # (1,)
            "mission": np.asarray(ids, dtype=np.int64),  # (max_len,)
        }

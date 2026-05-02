"""Silence BabyAI's rejection-sampling stdout chatter.

Background:
    BabyAI's level generator uses rejection sampling (see
    ``minigrid/envs/babyai/core/roomgrid_level.py``).  When a generated
    level has unreachable objects or invalid instructions, it raises
    ``RejectSampling``, the outer loop emits a raw ``print(...)`` and
    retries.  These messages are informational only — every rejected
    sample is automatically resampled, so training is unaffected — but
    with ``room_size=15`` (much larger than BabyAI's defaults of 6–8)
    the chatter can flood stdout during BabyAI maze training.

This module replaces the ``print`` reference inside BabyAI's
``roomgrid_level`` module with a thin wrapper that forwards messages to
a Python logger at DEBUG level.  Silent by default; re-enable by
setting the logger to DEBUG, e.g.::

    logging.getLogger("baby_ai.rejection").setLevel(logging.DEBUG)

Usage (call once at the top of an entry-point script, before any
BabyAI/MiniGrid env is created):

    from agent_training.baby_ai_silence import silence_baby_ai_rejection_logs
    silence_baby_ai_rejection_logs()
"""

from __future__ import annotations

import logging
from typing import Any

_REJECTION_LOGGER_NAME = "baby_ai.rejection"
_PATCHED_FLAG = "_rlvr_rejection_logs_patched"


def silence_baby_ai_rejection_logs() -> bool:
    """Redirect BabyAI's rejection-sampling ``print`` calls to logging.

    Idempotent — safe to call multiple times.  Returns ``True`` on
    successful patch, ``False`` if BabyAI is unavailable or already
    patched.

    Returns:
        ``True`` if the patch was applied this call, ``False`` otherwise.
    """
    try:
        from minigrid.envs.babyai.core import roomgrid_level
    except ImportError:
        # BabyAI not installed; nothing to silence.
        return False

    if getattr(roomgrid_level, _PATCHED_FLAG, False):
        return False

    rejection_logger = logging.getLogger(_REJECTION_LOGGER_NAME)

    def _logging_print(*args: Any, **_kwargs: Any) -> None:
        """Drop-in replacement for ``print`` that emits via logging."""
        message = " ".join(str(a) for a in args)
        rejection_logger.debug(message)

    roomgrid_level.print = _logging_print  # type: ignore[attr-defined]
    setattr(roomgrid_level, _PATCHED_FLAG, True)
    return True

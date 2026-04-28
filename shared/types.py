from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Module 1 – LLM Policy 相關
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """LLMPolicy.generate() 的回傳值。"""
    texts: list[str]            # 生成的原始文字
    log_probs: Tensor           # shape (batch, seq_len)
    token_ids: Tensor           # shape (batch, seq_len)


@dataclass
class GRPOBatch:
    """傳入 LLMPolicy.update() 的一個 training batch。"""
    token_ids: Tensor
    log_probs: Tensor           # 生成時的 log probs
    ref_log_probs: Tensor       # reference model 的 log probs
    rewards: Tensor             # shape (batch,)
    advantages: Tensor          # GRPO group-normalized advantages


# ---------------------------------------------------------------------------
# Module 2 – Level Parser & Game Environment 相關
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """GameEnvironment.parse_level() 的回傳值。"""
    success: bool
    level_config: Optional[dict] = None
    error_msg: Optional[str] = None


@dataclass
class Trajectory:
    """單次 agent rollout 的紀錄。"""
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    total_return: float
    success: bool
    length: int


@dataclass
class RolloutResult:
    """GameEnvironment.run_rollouts() 的回傳值。"""
    level_config: dict
    trajectories: dict[str, list[Trajectory]]
    # key = agent_id (e.g. "strong_0", "weak_0"), value = 該 agent 多次 rollout


# ---------------------------------------------------------------------------
# Module 3 – Reward Calculator & Evaluation 相關
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """RewardCalculator 的超參數。"""
    regret_weight: float = 1.0
    breadth_weight: float = 0.5
    playability_bonus: float = 1.0
    invalid_penalty: float = -1.0


@dataclass
class RewardOutput:
    """RewardCalculator.compute_reward() 的回傳值。"""
    total_reward: float
    regret: float
    strategy_breadth: float
    playable: bool
    breakdown: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """EvaluationSuite.evaluate() 的回傳值。"""
    playability_rate: float
    held_out_regret: dict       # mean, median, std, >threshold %
    solution_diversity: dict    # mean JSD across trajectories
    controllability: dict       # per-dimension Cohen's d (Phase 2+)
    raw_data: list[dict] = field(default_factory=list)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Module A – LLM Policy
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """LLMPolicy.generate() 的回傳值。"""
    texts: list[str]            # 生成的原始文字（ASCII grid + JSON）
    log_probs: Tensor           # shape (batch, seq_len)
    token_ids: Tensor           # shape (batch, seq_len)
    prompt_ids: Tensor          # shape (batch, prompt_len)


@dataclass
class GRPOBatch:
    """傳入 LLMPolicy.update() 的一個 training batch。"""
    token_ids: Tensor           # shape (batch, seq_len)
    prompt_ids: Tensor          # shape (batch, prompt_len)
    log_probs: Tensor           # 生成時的 log probs, shape (batch, seq_len)
    ref_log_probs: Tensor       # reference model 的 log probs, shape (batch, seq_len)
    rewards: Tensor             # shape (batch,)
    advantages: Tensor          # GRPO group-normalized advantages, shape (batch,)


# ---------------------------------------------------------------------------
# Module B – Level Parser & Game Environment
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """GameEnvironment.parse_level() 的回傳值。

    level_config 結構:
        {
            "width": 13,
            "height": 13,
            "grid": list[str],          # 13 行，每行 13 字元 ('W' 或 '.')
            "objects": list[dict],       # 物件清單，見下方說明
            "agent_start": {"x": int, "y": int, "dir": 0-3}
        }

    可用物件 (objects 內的 dict):
        - wall:  由 grid 中 'W' 定義，不出現在 objects 清單
        - floor: 由 grid 中 '.' 定義，不出現在 objects 清單
        - goal:  {"type": "goal", "x": int, "y": int}
                 終點，Agent 走到即過關。整個關卡恰好一個。
        - key:   {"type": "key", "x": int, "y": int, "color": str}
                 鑰匙，可撿起，用來開同色 locked door。
        - door:  {"type": "door", "x": int, "y": int, "color": str, "state": str}
                 門，state 可為 "open" / "closed" / "locked"（預設 "locked"）。
                 locked 需要同色 key 才能開啟；closed 可直接開啟。
        - ball:  {"type": "ball", "x": int, "y": int, "color": str}
                 球，可撿起或推動到相鄰空格。
        - box:   {"type": "box", "x": int, "y": int, "color": str}
                 箱子，可打開。可選 "contains" 欄位藏一個 key:
                 {"type": "box", ..., "contains": {"type": "key", "color": str}}

    顏色可用: red, green, blue, purple, yellow, grey
    座標範圍: x ∈ [0, 12], y ∈ [0, 12]
    agent_start.dir: 0=right, 1=down, 2=left, 3=up
    """
    success: bool
    level_config: Optional[dict] = None   # 見 type_examples.py 的 level_config 範例
    error_msg: Optional[str] = None


@dataclass
class Trajectory:
    """單次 agent rollout 的紀錄。"""
    states: list[np.ndarray]    # 每步的 observation (7×7 partially observable)
    actions: list[int]          # MiniGrid action ids
    rewards: list[float]        # 每步的 reward
    total_return: float         # 累計 return
    success: bool               # 是否通關
    length: int                 # trajectory 步數


@dataclass
class RolloutResult:
    """GameEnvironment.run_rollouts() 的回傳值。"""
    level_config: dict
    trajectories: dict[str, list[Trajectory]]
    # key = agent_id (e.g. "strong_0", "weak_0"), value = M 次 rollout


# ---------------------------------------------------------------------------
# Module C – Reward Calculator & Evaluation
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """RewardCalculator 的超參數。Per SPEC §5.2。"""
    regret_weight: float = 1.0
    playability_bonus: float = 1.0
    invalid_penalty: float = -1.0
    # strategy_breadth_weight: float = 0.0  # [deferred] — PD-1


@dataclass
class RewardOutput:
    """RewardCalculator.compute_reward() 的回傳值。Per SPEC §5.2。"""
    total_reward: float
    regret: float               # max(0, mean(V_strong) - mean(V_weak))
    playable: bool              # 至少一個 agent 通關
    breakdown: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """EvaluationSuite.evaluate() 的回傳值。Per SPEC §8。"""
    # --- 主要指標 ---
    parse_success_rate: float       # LLM 輸出成功解析為合法關卡的比例
    playability_rate: float         # 可通關關卡佔所有合法關卡的比例
    held_out_regret: dict           # {"mean": float, "median": float, "std": float}
    # --- 次要指標 (deferred) ---
    solution_diversity: Optional[dict] = None   # [deferred] — PD-5
    controllability: Optional[dict] = None      # Phase 2+ — SPEC §8
    # --- 元資料 ---
    eval_mode: str = "full"         # "quick" (training agents) or "full" (held-out agents)
    num_levels: int = 0             # 評估的關卡總數
    raw_data: list[dict] = field(default_factory=list)


@dataclass
class MetricsResult:
    """metrics.py 計算的指標彙總。供 evaluate.py 及 EvaluationSuite 使用。"""
    parse_success_rate: float       # 0~1，成功解析數 / 總數
    playability_rate: float         # 0~1，可通關數 / 成功解析數
    regret_stats: dict              # {"mean": float, "median": float, "std": float}
    total_levels: int               # 關卡總數
    parsed_levels: int              # 成功解析的關卡數
    playable_levels: int            # 可通關的關卡數

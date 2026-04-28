"""
各 dataclass 的範例實例，方便各模組開發時參照格式。
僅填入可確定的欄位；需要真實環境才能得知的（如 Trajectory 的 states）以 placeholder 標註。
"""

from shared.types import (
    ParseResult,
    RolloutResult,
    RewardConfig,
    RewardOutput,
    EvalReport,
)

# ===================================================================
# ParseResult 範例
# ===================================================================

# 成功解析
parse_success_example = ParseResult(
    success=True,
    level_config={
        "width": 8,
        "height": 8,
        "objects": [
            {"type": "wall", "x": 2, "y": 3},
            {"type": "wall", "x": 2, "y": 4},
            {"type": "wall", "x": 2, "y": 5},
            {"type": "key", "x": 1, "y": 5, "color": "yellow"},
            {"type": "door", "x": 4, "y": 4, "color": "yellow"},
            {"type": "goal", "x": 7, "y": 7},
        ],
        "agent_start": {"x": 0, "y": 0, "dir": 0},
    },
    error_msg=None,
)

# 解析失敗 — JSON 格式錯誤
parse_fail_json_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Invalid JSON: Expecting ',' delimiter at line 5 col 2",
)

# 解析失敗 — 語義錯誤（無 goal）
parse_fail_semantic_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Semantic error: level must contain exactly one 'goal' object",
)

# ===================================================================
# level_config 範例（LLM 輸出解析後的結構）
# ===================================================================

level_config_simple = {
    "width": 8,
    "height": 8,
    "objects": [
        {"type": "wall", "x": 2, "y": 3},
        {"type": "wall", "x": 2, "y": 4},
        {"type": "wall", "x": 2, "y": 5},
        {"type": "key", "x": 1, "y": 5, "color": "yellow"},
        {"type": "door", "x": 4, "y": 4, "color": "yellow"},
        {"type": "goal", "x": 7, "y": 7},
    ],
    "agent_start": {"x": 0, "y": 0, "dir": 0},
}

level_config_with_lava = {
    "width": 10,
    "height": 10,
    "objects": [
        {"type": "wall", "x": 3, "y": 0},
        {"type": "wall", "x": 3, "y": 1},
        {"type": "wall", "x": 3, "y": 2},
        {"type": "lava", "x": 5, "y": 5},
        {"type": "lava", "x": 5, "y": 6},
        {"type": "key", "x": 1, "y": 8, "color": "blue"},
        {"type": "door", "x": 7, "y": 3, "color": "blue"},
        {"type": "goal", "x": 9, "y": 9},
    ],
    "agent_start": {"x": 0, "y": 0, "dir": 0},
}

# ===================================================================
# Trajectory 範例
# ===================================================================
# ⚠ states 的具體內容取決於 MiniGrid observation wrapper，
#   這裡無法提供真實值，僅展示結構。
#   實際 states 由成員 B 的 environment wrapper 決定。
#
# 範例結構（僅供參考，不可直接執行）:
#
# trajectory_example = Trajectory(
#     states=[...],               # list[np.ndarray], 每步的 observation
#     actions=[2, 2, 0, 1, 2, 5], # MiniGrid action space: 0=left 1=right 2=forward 3=pickup 4=drop 5=toggle 6=done
#     rewards=[0, 0, 0, 0, 0, 1.0],
#     total_return=0.95,          # discounted or undiscounted 累計 return
#     success=True,
#     length=6,
# )

# ===================================================================
# RolloutResult 範例
# ===================================================================
# ⚠ 需要真實 Trajectory，這裡僅展示外層結構。
#
# rollout_result_example = RolloutResult(
#     level_config=level_config_simple,
#     trajectories={
#         "strong_0": [trajectory_0, trajectory_1, ...],   # num_rollouts_per_agent 次
#         "strong_1": [trajectory_0, trajectory_1, ...],
#         "weak_0":   [trajectory_0, trajectory_1, ...],
#         "weak_1":   [trajectory_0, trajectory_1, ...],
#     },
# )

# ===================================================================
# RewardConfig 範例
# ===================================================================

reward_config_default = RewardConfig(
    regret_weight=1.0,
    breadth_weight=0.5,
    playability_bonus=1.0,
    invalid_penalty=-1.0,
)

# ===================================================================
# RewardOutput 範例
# ===================================================================

# 正常可玩關卡
reward_output_good = RewardOutput(
    total_reward=2.3,
    regret=1.5,
    strategy_breadth=1.6,
    playable=True,
    breakdown={
        "playability_bonus": 1.0,
        "regret_component": 1.5,       # regret_weight * regret
        "breadth_component": 0.8,      # breadth_weight * strategy_breadth
    },
)

# 可解析但不可通關（所有 agent 都 fail）
reward_output_unplayable = RewardOutput(
    total_reward=0.0,
    regret=0.0,
    strategy_breadth=0.0,
    playable=False,
    breakdown={
        "reason": "no agent succeeded in any rollout",
    },
)

# 解析失敗
reward_output_invalid = RewardOutput(
    total_reward=-1.0,
    regret=0.0,
    strategy_breadth=0.0,
    playable=False,
    breakdown={
        "reason": "parse failure",
    },
)

# ===================================================================
# EvalReport 範例
# ===================================================================

eval_report_example = EvalReport(
    playability_rate=0.75,
    held_out_regret={
        "mean": 1.2,
        "median": 1.1,
        "std": 0.4,
        "above_threshold_pct": 0.6,   # regret > threshold 的比例
    },
    solution_diversity={
        "mean_jsd": 0.35,
    },
    controllability={},                # Phase 2+ 才會有值
    raw_data=[
        {
            "level_idx": 0,
            "playable": True,
            "regret": 1.5,
            "strategy_breadth": 1.6,
            "jsd": 0.32,
        },
        {
            "level_idx": 1,
            "playable": False,
            "regret": 0.0,
            "strategy_breadth": 0.0,
            "jsd": 0.0,
        },
    ],
)

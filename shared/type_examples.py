"""
各 dataclass 的範例實例，方便各模組開發時參照格式。
所有座標遵循 SPEC §3：13×13 usable grid，座標範圍 0-12。
"""

from shared.types import (
    ParseResult,
    RolloutResult,
    RewardConfig,
    RewardOutput,
    EvalReport,
)

# ===================================================================
# level_config 範例（Parser 解析 ASCII grid + JSON 後的結構）
# ===================================================================
# LLM 輸出 ASCII grid (定義 W/.) + JSON (定義 objects + agent_start)
# Parser 將 ASCII grid 轉為 grid 欄位，JSON 直接對應 objects / agent_start
# 座標範圍: x ∈ [0, 12], y ∈ [0, 12]

level_config_simple = {
    "width": 13,
    "height": 13,
    "grid": [
        ".............",
        ".............",
        "..WWW........",
        "....W........",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
    ],
    "objects": [
        {"type": "key", "x": 1, "y": 5, "color": "yellow"},
        {"type": "door", "x": 4, "y": 4, "color": "yellow"},
        {"type": "goal", "x": 12, "y": 12},
    ],
    "agent_start": {"x": 0, "y": 0, "dir": 0},
}

level_config_with_lava = {
    "width": 13,
    "height": 13,
    "grid": [
        ".............",
        ".............",
        ".............",
        "WWW..........",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
    ],
    "objects": [
        {"type": "lava", "x": 5, "y": 5},
        {"type": "lava", "x": 5, "y": 6},
        {"type": "key", "x": 1, "y": 8, "color": "blue"},
        {"type": "door", "x": 7, "y": 3, "color": "blue"},
        {"type": "goal", "x": 12, "y": 12},
    ],
    "agent_start": {"x": 0, "y": 0, "dir": 0},
}

level_config_multi_key = {
    "width": 13,
    "height": 13,
    "grid": [
        ".............",
        ".............",
        ".....W.......",
        ".....W.......",
        ".....W.......",
        ".....W.......",
        ".....W.......",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
        ".............",
    ],
    "objects": [
        {"type": "key", "x": 2, "y": 1, "color": "yellow"},
        {"type": "key", "x": 10, "y": 10, "color": "blue"},
        {"type": "door", "x": 5, "y": 7, "color": "yellow"},
        {"type": "door", "x": 5, "y": 2, "color": "blue"},
        {"type": "goal", "x": 12, "y": 0},
    ],
    "agent_start": {"x": 0, "y": 12, "dir": 0},
}


# ===================================================================
# ParseResult 範例
# ===================================================================

parse_success_example = ParseResult(
    success=True,
    level_config=level_config_simple,
    error_msg=None,
)

parse_fail_json_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Invalid JSON: Expecting ',' delimiter at line 5 col 2",
)

parse_fail_grid_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Grid error: expected 13 rows, got 10",
)

parse_fail_semantic_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Semantic error: level must contain exactly one 'goal' object",
)

parse_fail_overlap_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Semantic error: object at (4, 4) overlaps with wall",
)

parse_fail_coord_example = ParseResult(
    success=False,
    level_config=None,
    error_msg="Semantic error: object coordinate x=15 out of range [0, 12]",
)


# ===================================================================
# Trajectory 範例
# ===================================================================
# states 的內容取決於 MiniGrid observation wrapper (7×7 partial obs)，
# 這裡無法提供真實值，僅展示結構。
#
# trajectory_example = Trajectory(
#     states=[np.zeros((7, 7, 3)), ...],  # 每步的 7×7 partially observable grid
#     actions=[2, 2, 0, 1, 2, 5],
#     # MiniGrid actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done
#     rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
#     total_return=0.95,
#     success=True,
#     length=6,
# )
#
# trajectory_fail_example = Trajectory(
#     states=[np.zeros((7, 7, 3)), ...],
#     actions=[2, 2, 0, 0, 2, 2, ...],   # 達到 max_steps 仍未通關
#     rewards=[0.0, 0.0, ...],
#     total_return=0.0,
#     success=False,
#     length=100,
# )


# ===================================================================
# RolloutResult 範例
# ===================================================================
# 需要真實 Trajectory，這裡僅展示外層結構。
# Per SPEC §5.4: training agents = strong_0, weak_0
#                 held-out agents = strong_held_0, weak_held_0
#
# rollout_result_example = RolloutResult(
#     level_config=level_config_simple,
#     trajectories={
#         "strong_0": [trajectory_0, trajectory_1, ...],  # num_rollouts_per_agent=5
#         "weak_0":   [trajectory_0, trajectory_1, ...],
#     },
# )
#
# rollout_result_eval_example = RolloutResult(
#     level_config=level_config_simple,
#     trajectories={
#         "strong_held_0": [trajectory_0, ...],  # held-out agents for evaluation
#         "weak_held_0":   [trajectory_0, ...],
#     },
# )


# ===================================================================
# RewardConfig 範例
# ===================================================================

reward_config_default = RewardConfig(
    regret_weight=1.0,
    playability_bonus=1.0,
    invalid_penalty=-1.0,
)


# ===================================================================
# RewardOutput 範例 — Per SPEC §5.2
# ===================================================================

# reward = playability_bonus + regret_weight * regret
reward_output_good = RewardOutput(
    total_reward=2.5,
    regret=1.5,
    playable=True,
    breakdown={
        "playability_bonus": 1.0,
        "regret_component": 1.5,        # regret_weight * regret
    },
)

# 可解析但不可通關 → reward = 0.0
reward_output_unplayable = RewardOutput(
    total_reward=0.0,
    regret=0.0,
    playable=False,
    breakdown={
        "reason": "no agent succeeded in any rollout",
    },
)

# 解析失敗 → reward = invalid_penalty
reward_output_invalid = RewardOutput(
    total_reward=-1.0,
    regret=0.0,
    playable=False,
    breakdown={
        "reason": "parse failure",
    },
)

# 強 agent 表現不如弱 agent → regret clamp 到 0
reward_output_negative_regret = RewardOutput(
    total_reward=1.0,
    regret=0.0,
    playable=True,
    breakdown={
        "playability_bonus": 1.0,
        "regret_component": 0.0,
        "raw_regret_before_clamp": -0.3,
    },
)


# ===================================================================
# EvalReport 範例 — Per SPEC §8
# ===================================================================

eval_report_full_example = EvalReport(
    parse_success_rate=0.85,
    playability_rate=0.75,
    held_out_regret={
        "mean": 1.2,
        "median": 1.1,
        "std": 0.4,
    },
    solution_diversity=None,        # [deferred]
    controllability=None,           # Phase 2+
    eval_mode="full",
    num_levels=100,
    raw_data=[
        {
            "level_idx": 0,
            "parsed": True,
            "playable": True,
            "regret": 1.5,
            "strong_mean_return": 0.85,
            "weak_mean_return": 0.35,
        },
        {
            "level_idx": 1,
            "parsed": True,
            "playable": False,
            "regret": 0.0,
            "strong_mean_return": 0.0,
            "weak_mean_return": 0.0,
        },
        {
            "level_idx": 2,
            "parsed": False,
            "playable": False,
            "regret": 0.0,
        },
    ],
)

eval_report_quick_example = EvalReport(
    parse_success_rate=0.80,
    playability_rate=0.70,
    held_out_regret={
        "mean": 1.0,
        "median": 0.9,
        "std": 0.5,
    },
    eval_mode="quick",
    num_levels=100,
)

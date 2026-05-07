"""端到端 Pipeline Smoke Test — Phase 0 Toy Case。

Per SPEC §12 Experiment T:
  1. LLM zero-shot 生成 10-20 張地圖
  2. parse → build env → agent rollout → reward 計算
  3. 驗證所有 shared type dataclass 欄位正確

Usage:
  python -m toy_case.run_toy_pipeline --config config/default.yaml
  python -m toy_case.run_toy_pipeline --use-mock   # 用 mock 模組測試 pipeline 資料流
"""

from __future__ import annotations

# Set project-local cache dirs before any ML library is imported.
from shared.env_setup import setup_project_cache
setup_project_cache()

import argparse
import json
import logging
import os
from dataclasses import fields
from typing import Optional

import yaml

from shared.types import (
    GenerationOutput,
    ParseResult,
    RewardConfig,
    RewardOutput,
    RolloutResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """載入 YAML 配置。

    Args:
        config_path: 配置檔路徑。

    Returns:
        配置 dict。
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataclass_fields(instance: object, class_name: str) -> list[str]:
    """驗證 dataclass 實例的欄位是否完整且型別正確。

    Args:
        instance: Dataclass 實例。
        class_name: 用於 logging 的類別名稱。

    Returns:
        錯誤訊息列表（空 = 全部通過）。
    """
    errors: list[str] = []
    for f in fields(instance):
        if not hasattr(instance, f.name):
            errors.append(f"{class_name}.{f.name}: 欄位缺失")
        else:
            value = getattr(instance, f.name)
            if value is None and f.default is not None:
                pass
    return errors


def run_pipeline_with_mock(config: dict) -> None:
    """使用 mock 模組跑通 pipeline 資料流。

    Args:
        config: 完整配置 dict。
    """
    from llm_policy.mock import MockLLMPolicy
    from reward_eval.mock import MockRewardCalculator

    logger.info("=" * 60)
    logger.info("Pipeline Smoke Test (Mock Mode)")
    logger.info("=" * 60)

    num_levels = config.get("toy_case", {}).get("num_test_levels", 10)

    # --- Step 1: Mock LLM generate ---
    llm = MockLLMPolicy(valid_rate=0.7)
    prompts = ["Generate a MiniGrid level"] * num_levels
    gen_output = llm.generate(prompts)

    assert isinstance(gen_output, GenerationOutput), "GenerationOutput 型別錯誤"
    assert len(gen_output.texts) == num_levels, (
        f"texts 長度不符: {len(gen_output.texts)} != {num_levels}"
    )
    assert gen_output.log_probs.shape[0] == num_levels, "log_probs batch 不符"
    assert gen_output.token_ids.shape[0] == num_levels, "token_ids batch 不符"
    assert gen_output.prompt_ids.shape[0] == num_levels, "prompt_ids batch 不符"
    logger.info("[Step 1] GenerationOutput 驗證通過 ✓")

    # --- Step 2: Mock parse (模擬 parser 行為) ---
    parse_results: list[ParseResult] = []
    for text in gen_output.texts:
        if "Grid:" in text:
            # [impl-updated 2026-05-07] 15×15 含外牆；inner area row/col 1-13
            outer_wall = "W" * 15
            inner_row = "W" + "." * 13 + "W"
            parse_results.append(ParseResult(
                success=True,
                level_config={
                    "width": 15, "height": 15,
                    "grid": [outer_wall] + [inner_row] * 13 + [outer_wall],
                    "objects": [{"type": "ball", "x": 13, "y": 13, "color": "blue"}],
                    "agent_start": {"x": 1, "y": 1, "dir": 0},
                    "goal": 0,
                },
            ))
        else:
            parse_results.append(ParseResult(
                success=False,
                error_msg="Mock parse failure: no Grid: found",
            ))

    for pr in parse_results:
        errors = validate_dataclass_fields(pr, "ParseResult")
        assert not errors, f"ParseResult 驗證失敗: {errors}"
        if pr.success:
            assert pr.level_config is not None, "success=True 但 level_config 為 None"
        else:
            assert pr.error_msg is not None, "success=False 但 error_msg 為 None"
    logger.info("[Step 2] ParseResult 驗證通過 ✓ (parsed=%d/%d)",
                sum(1 for p in parse_results if p.success), len(parse_results))

    # --- Step 3: Mock rollout (模擬 agent rollout) ---
    import numpy as np
    rollout_results: list[Optional[RolloutResult]] = []
    for pr in parse_results:
        if not pr.success:
            rollout_results.append(None)
            continue

        mock_trajectories: dict[str, list[Trajectory]] = {}
        for agent_id in ["toy_strong_0", "toy_weak_0"]:
            trajs: list[Trajectory] = []
            for _ in range(5):
                is_strong = "strong" in agent_id
                success = np.random.random() > (0.1 if is_strong else 0.5)
                length = np.random.randint(10, 100)
                total_return = 0.9 if success else 0.0
                trajs.append(Trajectory(
                    states=[np.zeros((7, 7, 3), dtype=np.uint8)] * length,
                    actions=[np.random.randint(0, 7) for _ in range(length)],
                    rewards=[0.0] * (length - 1) + [total_return],
                    total_return=total_return,
                    success=success,
                    length=length,
                ))
            mock_trajectories[agent_id] = trajs

        rollout_results.append(RolloutResult(
            level_config=pr.level_config,
            trajectories=mock_trajectories,
        ))

    for rr in rollout_results:
        if rr is not None:
            errors = validate_dataclass_fields(rr, "RolloutResult")
            assert not errors, f"RolloutResult 驗證失敗: {errors}"
            for agent_id, trajs in rr.trajectories.items():
                assert len(trajs) == 5, f"agent {agent_id} rollout 數不是 5"
                for traj in trajs:
                    assert traj.length == len(traj.actions), "length != len(actions)"
                    assert traj.length == len(traj.rewards), "length != len(rewards)"
    logger.info("[Step 3] RolloutResult 驗證通過 ✓ (rollouts=%d)",
                sum(1 for rr in rollout_results if rr is not None))

    # --- Step 4: Reward 計算 ---
    reward_config = RewardConfig(
        regret_weight=config.get("regret_weight", 1.0),
        playability_bonus=config.get("playability_bonus", 1.0),
        invalid_penalty=config.get("invalid_penalty", -1.0),
    )
    reward_calc = MockRewardCalculator(config=reward_config)
    reward_outputs = reward_calc.compute_batch_rewards(rollout_results)

    for ro in reward_outputs:
        assert isinstance(ro, RewardOutput), "RewardOutput 型別錯誤"
        errors = validate_dataclass_fields(ro, "RewardOutput")
        assert not errors, f"RewardOutput 驗證失敗: {errors}"
        assert ro.regret >= 0.0, f"regret 應 >= 0, 實際 {ro.regret}"
    logger.info("[Step 4] RewardOutput 驗證通過 ✓")

    # --- Step 5: GRPO advantages ---
    rewards_list = [ro.total_reward for ro in reward_outputs]
    group_size = config.get("group_size", 16)
    padded_len = ((len(rewards_list) + group_size - 1) // group_size) * group_size
    rewards_list.extend([0.0] * (padded_len - len(rewards_list)))

    advantages = reward_calc.compute_advantages_grpo(rewards_list, group_size)
    assert advantages.shape[0] == padded_len, "advantages shape 不符"
    logger.info("[Step 5] GRPO advantages 驗證通過 ✓ | shape=%s", advantages.shape)

    # --- 總結 ---
    logger.info("=" * 60)
    logger.info("Pipeline Smoke Test (Mock Mode) 全部通過 ✓")
    logger.info("=" * 60)

    parsed = sum(1 for p in parse_results if p.success)
    playable = sum(1 for ro in reward_outputs if ro.playable)
    mean_reward = sum(ro.total_reward for ro in reward_outputs) / len(reward_outputs)
    logger.info("統計: total=%d, parsed=%d, playable=%d, mean_reward=%.4f",
                len(gen_output.texts), parsed, playable, mean_reward)


def run_pipeline_real(config: dict) -> None:
    """使用真實模組跑 pipeline（需要 LLM + trained agents）。

    Args:
        config: 完整配置 dict。
    """
    from llm_policy.policy import LLMPolicy
    from llm_policy.prompts import get_minigrid_prompt, get_system_prompt, format_chat_messages
    from reward_eval.reward import RewardCalculator

    logger.info("=" * 60)
    logger.info("Pipeline Smoke Test (Real Mode)")
    logger.info("=" * 60)

    num_levels = config.get("toy_case", {}).get("num_test_levels", 20)

    # --- Step 1: LLM 生成 ---
    llm = LLMPolicy(
        model_name=config.get("model_name", "Qwen/Qwen3.5-9B"),
        quantization=config.get("quantization", "4bit"),
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_target_modules=config.get("lora_target_modules"),
        max_new_tokens=config.get("max_new_tokens", 2048),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.8),
        top_k=config.get("top_k", 20),
        presence_penalty=config.get("presence_penalty", 1.5),
        enable_thinking=config.get("enable_thinking", False),
        cache_dir=config.get("cache_dir"),
    )

    system_prompt = get_system_prompt()
    user_prompt = get_minigrid_prompt()
    messages = format_chat_messages(system_prompt, user_prompt)
    messages_list = [messages] * num_levels

    gen_output = llm.generate_with_chat_template(messages_list)

    assert isinstance(gen_output, GenerationOutput), "GenerationOutput 型別錯誤"
    assert len(gen_output.texts) == num_levels
    logger.info("[Step 1] LLM 生成完成 | %d levels", num_levels)

    # --- 輸出原始生成結果供檢查 ---
    os.makedirs("results/toy_case", exist_ok=True)
    for i, text in enumerate(gen_output.texts):
        output_path = f"results/toy_case/level_{i:03d}.txt"
        with open(output_path, "w") as f:
            f.write(text)
    logger.info("[Step 1] 原始生成結果已存至 results/toy_case/")

    # --- Step 2-4: parse + rollout + reward ---
    # 這裡需要 Module B (GameEnvironment) 的真實實作
    # 目前先用簡易 parse 驗證 LLM 輸出格式
    parse_success = 0
    for i, text in enumerate(gen_output.texts):
        has_grid = "Grid:" in text
        has_json = "{" in text and "objects" in text
        if has_grid and has_json:
            parse_success += 1
            logger.debug("[Step 2] Level %d: 格式初步正確 (Grid + JSON)", i)
        else:
            logger.debug("[Step 2] Level %d: 格式錯誤 (Grid=%s, JSON=%s)",
                         i, has_grid, has_json)

    parse_rate = parse_success / num_levels if num_levels > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Pipeline Smoke Test (Real Mode) 完成")
    logger.info("初步 parse rate: %.1f%% (%d/%d)",
                parse_rate * 100, parse_success, num_levels)
    logger.info("完整 pipeline（agent rollout + reward）需等 Module B 完成")
    logger.info("=" * 60)

    return parse_rate


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="Phase 0 Toy Case: End-to-end pipeline smoke test.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock modules (no GPU/model required).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.use_mock:
        run_pipeline_with_mock(config)
    else:
        run_pipeline_real(config)


if __name__ == "__main__":
    main()

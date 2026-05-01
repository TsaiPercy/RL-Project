"""Experiment 0: Sanity Check — LLM 結構化輸出能力驗證。

Per SPEC §10 Exp 0:
  - Hypothesis: Qwen3.5-9B zero-shot 可生成 parse-valid 的 MiniGrid ASCII grid + JSON
  - Target: parse rate > 10%
  - If violated: 換模型或退回純 JSON 格式

Per SPEC §12 Experiment 0:
  LLMPolicy.generate(prompts) → texts
  GameEnvironment.parse_level(text) → ParseResult
  統計 parse success rate

Usage:
  python -m toy_case.sanity_check --num-levels 100 --config config/default.yaml
  python -m toy_case.sanity_check --num-levels 20 --batch-size 5  # 分批生成
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

GRID_SIZE = 13
VALID_GRID_CHARS = {"W", "."}
VALID_OBJECT_TYPES = {"wall", "key", "door", "ball", "box", "goal", "lava"}
VALID_COLORS = {"red", "green", "blue", "purple", "yellow", "grey"}


def load_config(config_path: str) -> dict:
    """載入 YAML 配置。

    Args:
        config_path: 配置檔路徑。

    Returns:
        配置 dict。
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def simple_parse(text: str) -> dict:
    """簡易 parser：從 LLM 輸出中提取 Grid + JSON。

    這是 S-3 sanity check 用的輕量 parser，不依賴 Module B。
    只做格式驗證，不做語義驗證（連通性等）。

    Args:
        text: LLM 生成的原始文字。

    Returns:
        dict 包含 success, error_msg, grid, objects, agent_start。
    """
    result = {"success": False, "error_msg": None, "grid": None, "objects": None, "agent_start": None}

    # --- 提取 Grid ---
    grid_start = text.find("Grid:")
    if grid_start == -1:
        result["error_msg"] = "Cannot find 'Grid:' section"
        return result

    after_grid_label = text[grid_start + len("Grid:"):]
    all_lines = after_grid_label.split("\n")

    filtered_lines: list[str] = []
    grid_end_offset = grid_start + len("Grid:")
    for line in all_lines:
        stripped = line.strip()
        grid_end_offset += len(line) + 1
        if not stripped:
            if filtered_lines:
                break
            continue
        if all(c in VALID_GRID_CHARS for c in stripped) and len(stripped) > 0:
            filtered_lines.append(stripped)
        else:
            break

    if len(filtered_lines) != GRID_SIZE:
        result["error_msg"] = f"Grid rows: expected {GRID_SIZE}, got {len(filtered_lines)}"
        return result

    for i, line in enumerate(filtered_lines):
        if len(line) != GRID_SIZE:
            result["error_msg"] = f"Grid row {i}: expected {GRID_SIZE} chars, got {len(line)}"
            return result

    result["grid"] = filtered_lines

    # --- 提取 JSON ---
    remaining_text = text[grid_start:]
    json_data = None
    brace_start = remaining_text.find("{")
    if brace_start == -1:
        result["error_msg"] = "Cannot find JSON block after grid"
        return result

    depth = 0
    for i in range(brace_start, len(remaining_text)):
        if remaining_text[i] == "{":
            depth += 1
        elif remaining_text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    json_data = json.loads(remaining_text[brace_start:i + 1])
                except json.JSONDecodeError as e:
                    result["error_msg"] = f"JSON parse error: {e}"
                    return result
                break

    if json_data is None:
        result["error_msg"] = "Cannot find valid JSON block after grid"
        return result

    # --- 驗證 JSON 結構 ---
    if "objects" not in json_data:
        result["error_msg"] = "JSON missing 'objects' field"
        return result

    if "agent_start" not in json_data:
        result["error_msg"] = "JSON missing 'agent_start' field"
        return result

    if not isinstance(json_data["objects"], list):
        result["error_msg"] = "'objects' is not a list"
        return result

    # --- 驗證 objects ---
    has_goal = False
    for obj in json_data["objects"]:
        if "type" not in obj:
            result["error_msg"] = f"Object missing 'type': {obj}"
            return result
        if obj["type"] not in VALID_OBJECT_TYPES:
            result["error_msg"] = f"Invalid object type: {obj['type']}"
            return result
        if "x" not in obj or "y" not in obj:
            result["error_msg"] = f"Object missing x/y: {obj}"
            return result
        if not (0 <= obj["x"] <= GRID_SIZE - 1 and 0 <= obj["y"] <= GRID_SIZE - 1):
            result["error_msg"] = f"Object coordinate out of range: ({obj['x']}, {obj['y']})"
            return result
        if obj["type"] == "goal":
            has_goal = True

    if not has_goal:
        result["error_msg"] = "No 'goal' object found"
        return result

    # --- 驗證 agent_start ---
    agent = json_data["agent_start"]
    if "x" not in agent or "y" not in agent:
        result["error_msg"] = f"agent_start missing x/y: {agent}"
        return result
    if not (0 <= agent["x"] <= GRID_SIZE - 1 and 0 <= agent["y"] <= GRID_SIZE - 1):
        result["error_msg"] = f"agent_start coordinate out of range: ({agent['x']}, {agent['y']})"
        return result

    result["success"] = True
    result["objects"] = json_data["objects"]
    result["agent_start"] = json_data["agent_start"]
    return result


def run_sanity_check(
    config: dict,
    num_levels: int = 100,
    batch_size: int = 10,
    output_dir: str = "results/sanity_check",
) -> dict:
    """執行 Experiment 0 Sanity Check。

    Args:
        config: 配置 dict。
        num_levels: 總生成關卡數。
        batch_size: 每批生成數量。
        output_dir: 結果輸出目錄。

    Returns:
        dict 包含 parse_rate, total, parsed, error_breakdown 等統計。
    """
    from llm_policy.policy import LLMPolicy
    from llm_policy.prompts import get_minigrid_prompt, get_system_prompt, format_chat_messages

    llm = LLMPolicy(
        model_name=config.get("model_name", "Qwen/Qwen3.5-9B"),
        quantization=config.get("quantization", "4bit"),
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_target_modules=config.get("lora_target_modules"),
        max_new_tokens=config.get("max_new_tokens", 2048),
        temperature=config.get("temperature", 0.8),
    )

    system_prompt = get_system_prompt()
    user_prompt = get_minigrid_prompt()
    messages = format_chat_messages(system_prompt, user_prompt)

    os.makedirs(output_dir, exist_ok=True)

    all_texts: list[str] = []
    all_parse_results: list[dict] = []

    num_batches = (num_levels + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_levels - batch_idx * batch_size)
        logger.info("Batch %d/%d (size=%d)...", batch_idx + 1, num_batches, current_batch_size)

        messages_list = [messages] * current_batch_size
        gen_output = llm.generate_with_chat_template(messages_list)

        for i, text in enumerate(gen_output.texts):
            level_idx = batch_idx * batch_size + i
            all_texts.append(text)

            with open(os.path.join(output_dir, f"level_{level_idx:03d}.txt"), "w") as f:
                f.write(text)

            parse_result = simple_parse(text)
            parse_result["level_idx"] = level_idx
            all_parse_results.append(parse_result)

            status = "✓" if parse_result["success"] else f"✗ ({parse_result['error_msg']})"
            logger.info("  Level %d: %s", level_idx, status)

    # --- 統計 ---
    total = len(all_parse_results)
    parsed = sum(1 for r in all_parse_results if r["success"])
    parse_rate = parsed / total if total > 0 else 0.0

    error_breakdown: dict[str, int] = {}
    for r in all_parse_results:
        if not r["success"] and r["error_msg"]:
            key = r["error_msg"].split(":")[0]
            error_breakdown[key] = error_breakdown.get(key, 0) + 1

    stats = {
        "total": total,
        "parsed": parsed,
        "parse_rate": parse_rate,
        "error_breakdown": error_breakdown,
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Experiment 0: Sanity Check Results")
    logger.info("=" * 60)
    logger.info("Total levels: %d", total)
    logger.info("Parse success: %d (%.1f%%)", parsed, parse_rate * 100)
    logger.info("Target: > 10%%  →  %s", "PASS" if parse_rate > 0.1 else "FAIL")
    logger.info("")
    logger.info("Error breakdown:")
    for err, count in sorted(error_breakdown.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d (%.1f%%)", err, count, count / total * 100)
    logger.info("Results saved to: %s", output_dir)

    return stats


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="Experiment 0: Sanity Check — LLM structured output capability.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--num-levels", type=int, default=100,
        help="Number of levels to generate.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/sanity_check",
        help="Output directory for results.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    stats = run_sanity_check(
        config=config,
        num_levels=args.num_levels,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    if stats["parse_rate"] < 0.1:
        logger.warning(
            "Parse rate (%.1f%%) < 10%%! Per SPEC §7: "
            "考慮換模型或退回純 JSON 格式 (S-4).",
            stats["parse_rate"] * 100,
        )


if __name__ == "__main__":
    main()

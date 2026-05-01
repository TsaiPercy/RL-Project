"""
run_baseline.py — Zero-shot / Few-shot baseline 實驗腳本（B-1）。

Per SPEC §10 Experiment 1-2:
  python baselines/run_baseline.py --mode zero_shot --num-levels 100
  python baselines/run_baseline.py --mode few_shot --examples-path results/few_shot_examples.json --num-levels 100

功能:
  1. zero_shot: 使用未微調的 LLM 直接生成
  2. few_shot: 使用指定的 few-shot examples 生成
  3. 收集 parse rate / playability / regret 指標
  4. 輸出結果到 results/ 目錄
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from shared.types import RewardConfig
from reward_eval.reward import RewardCalculator
from reward_eval.evaluation import EvaluationSuite
from reward_eval.visualization import (
    plotRegretHistogram,
    plotBaselineComparison,
)

# --- Logging 設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parseArgs() -> argparse.Namespace:
    """解析命令列參數。所有 baseline 設定透過 argparse 直接處理。"""
    parser = argparse.ArgumentParser(
        description="RLVR Baseline — Zero-shot / Few-shot 實驗",
    )
    parser.add_argument(
        "--mode", type=str, default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Baseline 模式: 'zero_shot' 或 'few_shot'",
    )
    parser.add_argument(
        "--num-levels", type=int, default=100,
        help="生成的關卡數量（預設 100）",
    )
    parser.add_argument(
        "--examples-path", type=str, default=None,
        help="Few-shot examples 的 JSON 檔案路徑（僅 few_shot 模式需要）",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="配置檔路徑（預設 config/default.yaml）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="結果輸出目錄（預設 results/）",
    )
    parser.add_argument(
        "--eval-mode", type=str, default="full",
        choices=["quick", "full"],
        help="評估模式: 'quick'（training agents）或 'full'（held-out agents）",
    )
    return parser.parse_args()


def loadConfig(configPath: str) -> dict:
    """載入 YAML 配置檔。"""
    with open(configPath, "r", encoding="utf-8") as configFile:
        config = yaml.safe_load(configFile)
    logger.info("[baseline] 載入配置檔: %s", configPath)
    return config


def loadFewShotExamples(examplesPath: str) -> list[str]:
    """載入 few-shot examples。

    Args:
        examplesPath: JSON 檔案路徑，內容為 list[str]。

    Returns:
        Few-shot example 字串列表。
    """
    with open(examplesPath, "r", encoding="utf-8") as examplesFile:
        examples = json.load(examplesFile)

    if not isinstance(examples, list):
        raise ValueError(f"Few-shot examples 檔案格式錯誤，預期 list，收到 {type(examples)}")

    logger.info("[baseline] 載入 %d 個 few-shot examples: %s", len(examples), examplesPath)
    return examples


def main() -> None:
    """Baseline 實驗主流程。"""
    args = parseArgs()

    # --- 驗證參數 ---
    if args.mode == "few_shot" and args.examples_path is None:
        logger.error("[baseline] few_shot 模式需要指定 --examples-path")
        sys.exit(1)

    logger.info(
        "[baseline] 開始 baseline 實驗 | mode=%s, num_levels=%d, eval_mode=%s",
        args.mode, args.num_levels, args.eval_mode,
    )

    # --- Step 1: 載入配置 ---
    config = loadConfig(args.config)

    # --- Step 2: 載入 few-shot examples（如果需要） ---
    fewShotExamples: list[str] = []
    if args.mode == "few_shot":
        fewShotExamples = loadFewShotExamples(args.examples_path)

    # --- Step 3: 初始化模組 ---
    # 注意：LLMPolicy 和 GameEnvironment 需要成員 A 和 B 的實作。
    # 以下為預留的初始化流程。

    # TODO: 待成員 A 完成後取消註解
    # from llm_policy.policy import LLMPolicy
    # llmPolicy = LLMPolicy(config)  # 不載入 checkpoint → zero-shot

    # TODO: 待成員 B 完成後取消註解
    # from game_env.environment import GameEnvironment
    # gameEnv = GameEnvironment(config)

    logger.warning(
        "[baseline] LLMPolicy 和 GameEnvironment 尚未整合。"
        "請在成員 A/B 完成後取消 TODO 區段的註解。"
    )

    rewardConfig = RewardConfig(
        regret_weight=config.get("regret_weight", 1.0),
        playability_bonus=config.get("playability_bonus", 1.0),
        invalid_penalty=config.get("invalid_penalty", -1.0),
    )
    rewardCalc = RewardCalculator(rewardConfig)

    # --- Step 4: 生成關卡 ---
    # TODO: 待整合後替換
    # if args.mode == "few_shot":
    #     prompts = [llmPolicy.get_prompt(few_shot_examples=fewShotExamples)
    #                for _ in range(args.num_levels)]
    # else:
    #     prompts = [llmPolicy.get_prompt() for _ in range(args.num_levels)]
    #
    # generationOutput = llmPolicy.generate(prompts)
    # llmOutputs = generationOutput.texts
    logger.warning("[baseline] 使用空白 LLM 輸出（等待 LLMPolicy 整合）")
    llmOutputs: list[str] = []

    if not llmOutputs:
        logger.error("[baseline] 無 LLM 輸出可評估。請先整合 LLMPolicy。")
        sys.exit(1)

    # --- Step 5: 執行 EvaluationSuite ---
    # TODO: 待整合後替換 gameEnv
    # evalSuite = EvaluationSuite(gameEnv, rewardCalc, config)
    # report = evalSuite.evaluate(llmOutputs, mode=args.eval_mode)

    # --- Step 6: 輸出結果 ---
    # outputDir = Path(args.output_dir)
    # outputDir.mkdir(parents=True, exist_ok=True)
    #
    # reportPath = outputDir / f"baseline_{args.mode}_report.json"
    # evalSuite.exportReport(report, str(reportPath))
    # logger.info("[baseline] Baseline 報告已儲存: %s", reportPath)
    #
    # # --- 視覺化 ---
    # regretValues = [
    #     entry["regret"] for entry in report.raw_data
    #     if entry.get("playable", False)
    # ]
    # if regretValues:
    #     plotRegretHistogram(
    #         regretValues,
    #         str(outputDir / f"baseline_{args.mode}_regret_histogram.png"),
    #         title=f"Regret Distribution ({args.mode} baseline)",
    #     )

    logger.info("[baseline] Baseline 實驗完成 | mode=%s", args.mode)


if __name__ == "__main__":
    main()

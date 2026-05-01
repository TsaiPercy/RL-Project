"""
evaluate.py — 獨立評估腳本（INT-2）。

Per SPEC §8, §12:
  python evaluate.py --checkpoint checkpoints/best --mode full --num-levels 100
  python evaluate.py --checkpoint checkpoints/best --mode quick --num-levels 100

功能:
  1. 載入指定 checkpoint 的 LLM Policy
  2. 生成 num_levels 個關卡
  3. 使用 EvaluationSuite 執行 quick 或 full evaluation
  4. 輸出 EvalReport（JSON + 視覺化圖表）
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
)

# --- Logging 設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parseArgs() -> argparse.Namespace:
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="RLVR Game Level Generation — 獨立評估腳本",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="LLM checkpoint 路徑（如 checkpoints/best）",
    )
    parser.add_argument(
        "--mode", type=str, default="full", choices=["quick", "full"],
        help="評估模式: 'quick'（training agents）或 'full'（held-out agents）",
    )
    parser.add_argument(
        "--num-levels", type=int, default=100,
        help="評估的關卡數量（預設 100）",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="配置檔路徑（預設 config/default.yaml）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="結果輸出目錄（預設 results/）",
    )
    return parser.parse_args()


def loadConfig(configPath: str) -> dict:
    """載入 YAML 配置檔。"""
    with open(configPath, "r", encoding="utf-8") as configFile:
        config = yaml.safe_load(configFile)
    logger.info("[evaluate] 載入配置檔: %s", configPath)
    return config


def main() -> None:
    """評估主流程。"""
    args = parseArgs()

    logger.info(
        "[evaluate] 開始評估 | checkpoint=%s, mode=%s, num_levels=%d",
        args.checkpoint, args.mode, args.num_levels,
    )

    # --- Step 1: 載入配置 ---
    config = loadConfig(args.config)

    # --- Step 2: 初始化模組 ---
    # 注意：LLMPolicy 和 GameEnvironment 需要成員 A 和 B 的實作。
    # 在他們完成前，可使用 mock 進行測試。
    # 以下為預留的初始化流程：

    # TODO: 待成員 A 完成 LLMPolicy 後取消註解
    # from llm_policy.policy import LLMPolicy
    # llmPolicy = LLMPolicy(config, checkpoint_path=args.checkpoint)

    # TODO: 待成員 B 完成 GameEnvironment 後取消註解
    # from game_env.environment import GameEnvironment
    # gameEnv = GameEnvironment(config)

    # 暫時使用 placeholder 提示
    logger.warning(
        "[evaluate] LLMPolicy 和 GameEnvironment 尚未整合。"
        "請在成員 A/B 完成後取消 TODO 區段的註解。"
    )

    # --- 初始化 RewardCalculator ---
    rewardConfig = RewardConfig(
        regret_weight=config.get("regret_weight", 1.0),
        playability_bonus=config.get("playability_bonus", 1.0),
        invalid_penalty=config.get("invalid_penalty", -1.0),
    )
    rewardCalc = RewardCalculator(rewardConfig)

    # --- Step 3: 生成關卡（需 LLMPolicy） ---
    # TODO: 待整合後替換
    # prompts = [llmPolicy.get_prompt() for _ in range(args.num_levels)]
    # generationOutput = llmPolicy.generate(prompts)
    # llmOutputs = generationOutput.texts
    logger.warning("[evaluate] 使用空白 LLM 輸出（等待 LLMPolicy 整合）")
    llmOutputs: list[str] = []

    if not llmOutputs:
        logger.error("[evaluate] 無 LLM 輸出可評估。請先整合 LLMPolicy。")
        sys.exit(1)

    # --- Step 4: 執行 EvaluationSuite ---
    # TODO: 待整合後替換 gameEnv
    # evalSuite = EvaluationSuite(gameEnv, rewardCalc, config)
    # report = evalSuite.evaluate(llmOutputs, mode=args.mode)

    # --- Step 5: 輸出結果 ---
    # outputDir = Path(args.output_dir)
    # outputDir.mkdir(parents=True, exist_ok=True)

    # reportPath = outputDir / f"eval_report_{args.mode}.json"
    # evalSuite.exportReport(report, str(reportPath))
    # logger.info("[evaluate] 評估報告已儲存: %s", reportPath)

    # --- Step 6: 視覺化 ---
    # regretValues = [
    #     entry["regret"] for entry in report.raw_data
    #     if entry.get("playable", False)
    # ]
    # if regretValues:
    #     plotRegretHistogram(
    #         regretValues,
    #         str(outputDir / f"regret_histogram_{args.mode}.png"),
    #         title=f"Regret Distribution ({args.mode} eval)",
    #     )

    logger.info("[evaluate] 評估完成")


if __name__ == "__main__":
    main()

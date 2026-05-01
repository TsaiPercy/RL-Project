"""
EvaluationSuite — 關卡評估套件（quick + full 模式）。

Per SPEC §8 Evaluation Protocol:
  - Quick eval: 使用 training agents（strong_0, weak_0），100 levels
  - Full eval:  使用 held-out agents（strong_held_0, weak_held_0），100 levels

Per SPEC §11 Module C:
  evaluate(llm_outputs: list[str], mode: str = "full") → EvalReport
  export_report(report: EvalReport, path: str)

EvaluationSuite 持有 GameEnvironment 參考（Option A），
內部呼叫 batch_evaluate() 取得 rollout 結果。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from shared.types import (
    EvalReport,
    RewardConfig,
    RewardOutput,
)
from reward_eval.reward import RewardCalculator
from reward_eval.metrics import (
    computeParseSuccessRate,
    computePlayabilityRate,
    computeRegretStats,
)

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """關卡評估套件，整合 Module B (GameEnvironment) 與 Module C (RewardCalculator)。

    Attributes:
        gameEnv:    Module B 的 GameEnvironment 實例（或 mock）。
        rewardCalc: RewardCalculator 實例。
        config:     評估相關配置（從 default.yaml 載入）。
    """

    def __init__(
        self,
        gameEnv,
        rewardCalc: RewardCalculator,
        config: dict,
    ) -> None:
        """初始化 EvaluationSuite。

        Args:
            gameEnv:    GameEnvironment 實例，需有 batch_evaluate() 方法。
            rewardCalc: RewardCalculator 實例。
            config:     字典，至少包含:
                        - eval_num_levels (int): 每次評估的關卡數
                        - num_rollouts_per_agent (int): 每個 agent 的 rollout 次數
                        - training_agents (list[str]): quick mode 使用的 agent ids
                        - held_out_agents (list[str]): full mode 使用的 agent ids
        """
        self.gameEnv = gameEnv
        self.rewardCalc = rewardCalc
        self.config = config

        logger.info(
            "[EvaluationSuite] 初始化完成 | eval_num_levels=%d, "
            "training_agents=%s, held_out_agents=%s",
            config.get("eval_num_levels", 100),
            config.get("training_agents", []),
            config.get("held_out_agents", []),
        )

    def evaluate(
        self,
        llmOutputs: list[str],
        mode: str = "full",
    ) -> EvalReport:
        """執行完整評估流程。

        Per SPEC §11 Module C:
          1. 呼叫 GameEnvironment.batch_evaluate(llmOutputs) 取得 rollouts
          2. 呼叫 RewardCalculator.compute_batch_rewards(rollouts) 計算 rewards
          3. 彙整指標 → EvalReport

        Args:
            llmOutputs: LLM 生成的原始文字列表。
            mode:       "quick"（training agents）或 "full"（held-out agents）。

        Returns:
            EvalReport，包含所有指標和原始資料。
        """
        if mode not in ("quick", "full"):
            raise ValueError(f"mode 必須為 'quick' 或 'full'，收到: '{mode}'")

        numLevels = len(llmOutputs)
        numRollouts = self.config.get("num_rollouts_per_agent", 5)

        logger.info(
            "[EvaluationSuite] evaluate 開始 | mode='%s', num_levels=%d",
            mode, numLevels,
        )

        # --- Step 1: 透過 GameEnvironment 執行 parse + rollout ---
        # batch_evaluate 回傳 list[RolloutResult | None]
        # mode 決定使用哪組 agent（由 GameEnvironment 內部處理，或透過 config 切換）
        rollouts = self.gameEnv.batch_evaluate(
            llmOutputs, numRollouts,
        )

        logger.debug(
            "[EvaluationSuite] batch_evaluate 完成 | "
            "rollout 結果數=%d, None 數=%d",
            len(rollouts),
            sum(1 for rollout in rollouts if rollout is None),
        )

        # --- Step 2: 計算 rewards ---
        rewardOutputs = self.rewardCalc.compute_batch_rewards(rollouts)

        # --- Step 3: 彙整指標 ---
        # 建立 ParseResult-like 判斷（rollout is None → parse failed）
        from shared.types import ParseResult
        parseResults = [
            ParseResult(success=(rollout is not None))
            for rollout in rollouts
        ]

        parseSuccessRate = computeParseSuccessRate(parseResults)
        playabilityRate = computePlayabilityRate(rewardOutputs, parseResults)
        regretStats = computeRegretStats(rewardOutputs)

        # --- Step 4: 組裝 raw_data ---
        rawData = self._buildRawData(rollouts, rewardOutputs)

        report = EvalReport(
            parse_success_rate=parseSuccessRate,
            playability_rate=playabilityRate,
            held_out_regret=regretStats,
            eval_mode=mode,
            num_levels=numLevels,
            raw_data=rawData,
        )

        logger.info(
            "[EvaluationSuite] evaluate 完成 | mode='%s', "
            "parse_rate=%.2f%%, playability=%.2f%%, "
            "regret_mean=%.4f",
            mode,
            parseSuccessRate * 100,
            playabilityRate * 100,
            regretStats["mean"],
        )

        return report

    def exportReport(self, report: EvalReport, path: str) -> None:
        """將 EvalReport 匯出為 JSON 檔案。

        Args:
            report: EvalReport 實例。
            path:   輸出檔案路徑。
        """
        outputPath = Path(path)
        outputPath.parent.mkdir(parents=True, exist_ok=True)

        reportDict = {
            "parse_success_rate": report.parse_success_rate,
            "playability_rate": report.playability_rate,
            "held_out_regret": report.held_out_regret,
            "solution_diversity": report.solution_diversity,
            "controllability": report.controllability,
            "eval_mode": report.eval_mode,
            "num_levels": report.num_levels,
            "raw_data": report.raw_data,
        }

        with open(outputPath, "w", encoding="utf-8") as outputFile:
            json.dump(reportDict, outputFile, indent=2, ensure_ascii=False)

        logger.info(
            "[EvaluationSuite] exportReport → %s (%.1f KB)",
            outputPath, outputPath.stat().st_size / 1024,
        )

    # ------------------------------------------------------------------
    # 內部輔助方法
    # ------------------------------------------------------------------

    def _buildRawData(
        self,
        rollouts: list,
        rewardOutputs: list[RewardOutput],
    ) -> list[dict]:
        """為每個關卡組裝 raw_data 記錄。

        Per SPEC §8 / type_examples.py 中的 EvalReport.raw_data 格式。
        """
        rawData = []
        for levelIdx, (rollout, rewardOutput) in enumerate(
            zip(rollouts, rewardOutputs)
        ):
            entry: dict = {
                "level_idx": levelIdx,
                "parsed": rollout is not None,
                "playable": rewardOutput.playable,
                "regret": rewardOutput.regret,
            }

            # 若有 rollout 結果，加入各 agent 的平均 return
            if rollout is not None:
                for agentId, trajectories in rollout.trajectories.items():
                    meanReturn = (
                        sum(trajectory.total_return for trajectory in trajectories)
                        / len(trajectories)
                        if trajectories
                        else 0.0
                    )
                    entry[f"{agentId}_mean_return"] = meanReturn

            rawData.append(entry)

        return rawData

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

    Per SPEC §5.4 / §8:
      - 內部維護 agent pool 資料結構（training pool + held-out pool）
      - quick mode → 使用 training agents
      - full  mode → 使用 held-out agents

    Attributes:
        gameEnv:      Module B 的 GameEnvironment 實例（或 mock）。
        rewardCalc:   RewardCalculator 實例。
        config:       評估相關配置（從 default.yaml 載入）。
        trainingPool: Training agent pool（list[dict]）。
        heldOutPool:  Held-out agent pool（list[dict]）。
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
                        - agent_pool_path (str): agent checkpoint 根目錄
                        - training_agents (list[str]): quick mode 使用的 agent ids
                        - held_out_agents (list[str]): full mode 使用的 agent ids
        """
        self.gameEnv = gameEnv
        self.rewardCalc = rewardCalc
        self.config = config

        # --- 建立 Agent Pool (Per SPEC §5.4) ---
        agentPoolPath = config.get("agent_pool_path", "checkpoints/agents/")
        trainingAgentIds = config.get("training_agents", ["strong_0", "weak_0"])
        heldOutAgentIds = config.get(
            "held_out_agents", ["strong_held_0", "weak_held_0"],
        )

        self.trainingPool = self._buildAgentPool(
            agentPoolPath, trainingAgentIds,
        )
        self.heldOutPool = self._buildAgentPool(
            agentPoolPath, heldOutAgentIds,
        )

        logger.info(
            "[EvaluationSuite] 初始化完成 | eval_num_levels=%d, "
            "training_pool=%s, held_out_pool=%s",
            config.get("eval_num_levels", 100),
            [agent["id"] for agent in self.trainingPool],
            [agent["id"] for agent in self.heldOutPool],
        )

    # ------------------------------------------------------------------
    # Agent Pool 建構
    # ------------------------------------------------------------------

    @staticmethod
    def _buildAgentPool(
        basePath: str,
        agentIds: list[str],
    ) -> list[dict]:
        """根據 agent IDs 建立 agent pool 資料結構。

        Per SPEC §5.4 Agent Pool 配置:
          - strong_0, weak_0           → training reward
          - strong_held_0, weak_held_0 → evaluation only

        每個 agent 包含:
          - id (str): agent 識別碼（如 "strong_0"）
          - path (str): checkpoint 檔案路徑（如 "checkpoints/agents/strong_0.zip"）
          - type (str): "strong" 或 "weak"

        Args:
            basePath:  checkpoint 根目錄。
            agentIds:  agent ID 列表。

        Returns:
            list[dict]，每個元素為一個 agent 的 metadata。
        """
        pool: list[dict] = []
        for agentId in agentIds:
            # 根據 ID 判斷 agent 類型（strong/weak）
            agentType = "strong" if "strong" in agentId else "weak"
            checkpointPath = f"{basePath}{agentId}.zip"

            pool.append({
                "id": agentId,
                "path": checkpointPath,
                "type": agentType,
            })

            logger.debug(
                "[EvaluationSuite] agent pool 加入 | id=%s, type=%s, path=%s",
                agentId, agentType, checkpointPath,
            )

        return pool

    def getAgentPool(self, mode: str) -> list[dict]:
        """根據 mode 取得對應的 agent pool。

        Args:
            mode: "quick" 或 "full"。

        Returns:
            list[dict]，agent pool 資料結構。

        Raises:
            ValueError: mode 不是 "quick" 或 "full"。
        """
        if mode == "quick":
            return self.trainingPool
        elif mode == "full":
            return self.heldOutPool
        else:
            raise ValueError(f"mode 必須為 'quick' 或 'full'，收到: '{mode}'")

    # ------------------------------------------------------------------
    # 評估主流程
    # ------------------------------------------------------------------

    def evaluate(
        self,
        llmOutputs: list[str],
        mode: str = "full",
    ) -> EvalReport:
        """執行完整評估流程。

        Per SPEC §8 / §11 Module C:
          1. 根據 mode 選擇 agent pool（quick → training, full → held-out）
          2. 呼叫 GameEnvironment.batch_evaluate(llmOutputs, agent_ids=...) 取得 rollouts
          3. 呼叫 RewardCalculator.compute_batch_rewards(rollouts) 計算 rewards
          4. 彙整指標 → EvalReport

        Args:
            llmOutputs: LLM 生成的原始文字列表。
            mode:       "quick"（training agents）或 "full"（held-out agents）。

        Returns:
            EvalReport，包含所有指標和原始資料。
        """
        # --- 選擇 agent pool (Per SPEC §8) ---
        agentPool = self.getAgentPool(mode)
        agentIds = [agent["id"] for agent in agentPool]

        numLevels = len(llmOutputs)
        numRollouts = self.config.get("num_rollouts_per_agent", 5)

        logger.info(
            "[EvaluationSuite] evaluate 開始 | mode='%s', num_levels=%d, "
            "agent_ids=%s, agent_paths=%s",
            mode, numLevels, agentIds,
            [agent["path"] for agent in agentPool],
        )

        # --- Step 1: 透過 GameEnvironment 執行 parse + rollout ---
        # batch_evaluate 回傳 list[RolloutResult | None]
        # agent_ids 指定此次 evaluation 使用的 agent pool
        rollouts = self.gameEnv.batch_evaluate(
            llmOutputs, numRollouts, agent_ids=agentIds,
        )

        logger.debug(
            "[EvaluationSuite] batch_evaluate 完成 | "
            "rollout 結果數=%d, None 數=%d, agent_ids=%s",
            len(rollouts),
            sum(1 for rollout in rollouts if rollout is None),
            agentIds,
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

    # ------------------------------------------------------------------
    # 報告匯出
    # ------------------------------------------------------------------

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


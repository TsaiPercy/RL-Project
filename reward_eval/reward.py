"""
RewardCalculator — 計算 training reward 與 GRPO advantages。

Per SPEC §5.2 Reward 設計:
  reward(ℓ) =
    invalid_penalty (-1.0)                         if 解析失敗 (rollout is None)
    0.0                                            if 可解析但無 agent 通關
    playability_bonus + regret_weight * regret(ℓ)  otherwise

  regret(ℓ) = max(0, mean(V_strong) - mean(V_weak))

Per SPEC §5.3 GRPO Update:
  組內 z-score normalize rewards → advantages
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from shared.types import (
    RewardConfig,
    RewardOutput,
    RolloutResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class RewardCalculator:
    """計算 regret-based reward 與 GRPO group-relative advantages。

    Attributes:
        config: RewardConfig，包含 regret_weight、playability_bonus、invalid_penalty。
    """

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        logger.info(
            "[RewardCalculator] 初始化完成 | regret_weight=%.2f, "
            "playability_bonus=%.2f, invalid_penalty=%.2f",
            config.regret_weight,
            config.playability_bonus,
            config.invalid_penalty,
        )

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def compute_reward(self, rollout: Optional[RolloutResult]) -> RewardOutput:
        """計算單一關卡的 reward。

        Args:
            rollout: GameEnvironment.run_rollouts() 的結果；
                     若為 None 代表 parse 失敗。

        Returns:
            RewardOutput，包含 total_reward、regret、playable、breakdown。
        """

        # --- Case 1: parse 失敗 ---
        if rollout is None:
            logger.debug("[RewardCalculator] compute_reward → parse 失敗, reward=%.2f", self.config.invalid_penalty)
            return RewardOutput(
                total_reward=self.config.invalid_penalty,
                regret=0.0,
                playable=False,
                breakdown={"reason": "parse failure"},
            )

        # --- 判斷 playability: 至少一個 agent（不論強弱）有通關 ---
        isPlayable = self._check_playability(rollout)

        # --- Case 2: 可解析但無 agent 通關 ---
        if not isPlayable:
            logger.debug("[RewardCalculator] compute_reward → 不可通關, reward=0.0")
            return RewardOutput(
                total_reward=0.0,
                regret=0.0,
                playable=False,
                breakdown={"reason": "no agent succeeded in any rollout"},
            )

        # --- Case 3: 可通關，計算 regret ---
        strongMeanReturn, weakMeanReturn = self._compute_agent_returns(rollout)
        rawRegret = strongMeanReturn - weakMeanReturn
        clampedRegret = max(0.0, rawRegret)

        regretComponent = self.config.regret_weight * clampedRegret
        totalReward = self.config.playability_bonus + regretComponent

        breakdown = {
            "playability_bonus": self.config.playability_bonus,
            "regret_component": regretComponent,
            "raw_regret_before_clamp": rawRegret,
            "strong_mean_return": strongMeanReturn,
            "weak_mean_return": weakMeanReturn,
        }

        logger.debug(
            "[RewardCalculator] compute_reward → playable=True, "
            "regret=%.4f (raw=%.4f), total_reward=%.4f",
            clampedRegret, rawRegret, totalReward,
        )

        return RewardOutput(
            total_reward=totalReward,
            regret=clampedRegret,
            playable=True,
            breakdown=breakdown,
        )

    def compute_batch_rewards(
        self, rollouts: list[Optional[RolloutResult]]
    ) -> list[RewardOutput]:
        """批次計算多個關卡的 reward。

        Args:
            rollouts: 每個元素對應一個關卡的 rollout 結果（None = parse 失敗）。

        Returns:
            與 rollouts 等長的 RewardOutput 列表。
        """
        results = [self.compute_reward(rollout) for rollout in rollouts]

        # --- 批次 debug 統計 ---
        totalCount = len(results)
        playableCount = sum(1 for rewardOutput in results if rewardOutput.playable)
        invalidCount = sum(
            1 for rewardOutput in results
            if rewardOutput.breakdown.get("reason") == "parse failure"
        )
        logger.info(
            "[RewardCalculator] compute_batch_rewards | total=%d, playable=%d, "
            "invalid=%d, unplayable=%d",
            totalCount, playableCount, invalidCount,
            totalCount - playableCount - invalidCount,
        )
        return results

    def compute_advantages_grpo(
        self, rewards: list[float], groupSize: int
    ) -> Tensor:
        """計算 GRPO group-relative advantages（z-score normalization）。

        Per SPEC §5.3:
          對 batch 中每個 prompt 的 group_size 個 reward，做組內 z-score normalize。
          advantages[i] = (rewards[i] - group_mean) / (group_std + eps)

        Args:
            rewards: 長度為 batch_size * group_size 的 reward 列表。
            groupSize: 每個 prompt 對應的 sample 數量。

        Returns:
            shape (len(rewards),) 的 Tensor，包含 normalized advantages。

        Raises:
            ValueError: rewards 長度不能被 groupSize 整除時。
        """
        if len(rewards) % groupSize != 0:
            raise ValueError(
                f"rewards 長度 ({len(rewards)}) 無法被 groupSize ({groupSize}) 整除"
            )

        eps = 1e-8
        rewardsTensor = torch.tensor(rewards, dtype=torch.float32)
        numGroups = len(rewards) // groupSize
        advantages = torch.zeros_like(rewardsTensor)

        for groupIdx in range(numGroups):
            startIdx = groupIdx * groupSize
            endIdx = startIdx + groupSize
            groupRewards = rewardsTensor[startIdx:endIdx]

            groupMean = groupRewards.mean()
            groupStd = groupRewards.std()

            # z-score normalization
            advantages[startIdx:endIdx] = (groupRewards - groupMean) / (groupStd + eps)

            logger.debug(
                "[RewardCalculator] GRPO group %d | mean=%.4f, std=%.4f",
                groupIdx, groupMean.item(), groupStd.item(),
            )

        logger.info(
            "[RewardCalculator] compute_advantages_grpo | "
            "num_groups=%d, group_size=%d, advantages_mean=%.4f, advantages_std=%.4f",
            numGroups, groupSize,
            advantages.mean().item(), advantages.std().item(),
        )

        return advantages

    # ------------------------------------------------------------------
    # 內部輔助方法
    # ------------------------------------------------------------------

    def _check_playability(self, rollout: RolloutResult) -> bool:
        """檢查關卡是否可通關：至少一個 agent 的至少一次 rollout 成功。"""
        for agentId, trajectories in rollout.trajectories.items():
            for trajectory in trajectories:
                if trajectory.success:
                    logger.debug(
                        "[RewardCalculator] _check_playability → agent '%s' 通關",
                        agentId,
                    )
                    return True
        return False

    def _compute_agent_returns(
        self, rollout: RolloutResult
    ) -> tuple[float, float]:
        """計算 strong 和 weak agent 的平均 return。

        依照 agent_id 名稱中包含 'strong' / 'weak' 來分類。
        V_a(ℓ) = (1/M) Σ total_return(trajectory_m)

        Returns:
            (strongMeanReturn, weakMeanReturn) 的 tuple。
        """
        strongReturns: list[float] = []
        weakReturns: list[float] = []

        for agentId, trajectories in rollout.trajectories.items():
            agentReturns = [trajectory.total_return for trajectory in trajectories]

            if "strong" in agentId:
                strongReturns.extend(agentReturns)
            elif "weak" in agentId:
                weakReturns.extend(agentReturns)
            else:
                logger.warning(
                    "[RewardCalculator] 未知 agent 類型: '%s'，跳過", agentId
                )

        # 避免除以零
        strongMeanReturn = (
            sum(strongReturns) / len(strongReturns) if strongReturns else 0.0
        )
        weakMeanReturn = (
            sum(weakReturns) / len(weakReturns) if weakReturns else 0.0
        )

        logger.debug(
            "[RewardCalculator] _compute_agent_returns | "
            "strong_mean=%.4f (%d rollouts), weak_mean=%.4f (%d rollouts)",
            strongMeanReturn, len(strongReturns),
            weakMeanReturn, len(weakReturns),
        )

        return strongMeanReturn, weakMeanReturn

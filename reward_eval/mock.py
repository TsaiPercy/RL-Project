"""
MockRewardCalculator — 供成員 A、B 獨立測試的 mock 模組。

API 簽名與 RewardCalculator 完全一致，回傳合理的隨機值。
Per work_assignment MC-6。
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
from torch import Tensor

from shared.types import (
    RewardConfig,
    RewardOutput,
    RolloutResult,
)

logger = logging.getLogger(__name__)


class MockRewardCalculator:
    """RewardCalculator 的 mock 實作，回傳隨機但結構正確的結果。

    用途：
    - 成員 A 測試 train.py 訓練迴圈的資料流
    - 成員 B 測試 GameEnvironment 的輸出格式

    Attributes:
        config: RewardConfig（與真實 RewardCalculator 相同的配置）。
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()
        logger.info("[MockRewardCalculator] 初始化完成（mock 模式）")

    def compute_reward(self, rollout: Optional[RolloutResult]) -> RewardOutput:
        """回傳隨機 reward。

        - rollout is None → invalid_penalty
        - 否則隨機決定 playable（70% 機率 True）
        """
        if rollout is None:
            logger.debug("[MockRewardCalculator] compute_reward → mock parse failure")
            return RewardOutput(
                total_reward=self.config.invalid_penalty,
                regret=0.0,
                playable=False,
                breakdown={"reason": "parse failure", "mock": True},
            )

        isPlayable = random.random() < 0.7

        if not isPlayable:
            logger.debug("[MockRewardCalculator] compute_reward → mock unplayable")
            return RewardOutput(
                total_reward=0.0,
                regret=0.0,
                playable=False,
                breakdown={"reason": "no agent succeeded in any rollout", "mock": True},
            )

        mockRegret = random.uniform(0.0, 2.0)
        regretComponent = self.config.regret_weight * mockRegret
        totalReward = self.config.playability_bonus + regretComponent

        logger.debug(
            "[MockRewardCalculator] compute_reward → mock playable, "
            "regret=%.4f, total_reward=%.4f",
            mockRegret, totalReward,
        )

        return RewardOutput(
            total_reward=totalReward,
            regret=mockRegret,
            playable=True,
            breakdown={
                "playability_bonus": self.config.playability_bonus,
                "regret_component": regretComponent,
                "mock": True,
            },
        )

    def compute_batch_rewards(
        self, rollouts: list[Optional[RolloutResult]]
    ) -> list[RewardOutput]:
        """批次回傳隨機 reward。"""
        results = [self.compute_reward(rollout) for rollout in rollouts]
        logger.info(
            "[MockRewardCalculator] compute_batch_rewards | total=%d (mock)",
            len(results),
        )
        return results

    def compute_advantages_grpo(
        self, rewards: list[float], groupSize: int
    ) -> Tensor:
        """回傳隨機 advantages（結構正確的 z-score）。

        注意：這裡使用與真實版相同的 z-score 計算邏輯，
        確保數學上正確，只是輸入是 mock rewards。
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
            advantages[startIdx:endIdx] = (groupRewards - groupMean) / (groupStd + eps)

        logger.info(
            "[MockRewardCalculator] compute_advantages_grpo | "
            "num_groups=%d, group_size=%d (mock)",
            numGroups, groupSize,
        )
        return advantages

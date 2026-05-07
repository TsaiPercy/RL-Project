"""MockLLMPolicy — 供成員 B、C 獨立測試的 mock 模組。

API 簽名與 LLMPolicy 完全一致，回傳合理的隨機值。
Per TODO MA-4: 撰寫 LLMPolicy mock（API 簽名正確 + 隨機值）。
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import torch
from torch import Tensor

from shared.types import GenerationOutput, GRPOBatch

logger = logging.getLogger(__name__)

MOCK_VALID_LEVEL = """\
Grid:
.............
.............
..WWW........
....W........
.............
.............
.............
.............
.............
.............
.............
.............
.............

{
  "objects": [
    {"type": "key", "x": 1, "y": 5, "color": "yellow"},
    {"type": "door", "x": 4, "y": 4, "color": "yellow"},
    {"type": "ball", "x": 12, "y": 12, "color": "blue"}
  ],
  "agent_start": {"x": 0, "y": 0, "dir": 0},
  "goal": 2
}"""

MOCK_INVALID_LEVEL = """\
This is not a valid level format.
Some random text that the parser will reject.
"""


class MockLLMPolicy:
    """LLMPolicy 的 mock 實作，回傳結構正確的隨機結果。

    用途：
    - 成員 B 測試 GameEnvironment 的解析與 rollout
    - 成員 C 測試 RewardCalculator 和 EvaluationSuite

    Attributes:
        valid_rate: Mock 生成中 parse-valid 輸出的比例。
        max_new_tokens: 模擬的最大生成長度。
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        valid_rate: float = 0.7,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> None:
        """初始化 MockLLMPolicy。

        Args:
            model_name: 模型名稱（僅用於 logging）。
            valid_rate: 生成的 valid level 比例 (0-1)。
            max_new_tokens: 模擬的 sequence 長度。
            **kwargs: 吸收其他參數以相容真實 LLMPolicy 簽名。
        """
        self.model_name = model_name
        self.valid_rate = valid_rate
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cpu")
        logger.info(
            "[MockLLMPolicy] 初始化完成（mock 模式）| valid_rate=%.2f",
            valid_rate,
        )

    def generate(self, prompts: list[str]) -> GenerationOutput:
        """Mock 生成：回傳固定的 valid/invalid level 文字。

        Args:
            prompts: 一批 prompt 字串。

        Returns:
            GenerationOutput（texts 為 mock level，log_probs/token_ids 為隨機）。
        """
        batch_size = len(prompts)
        seq_len = 128

        texts = []
        for _ in range(batch_size):
            if random.random() < self.valid_rate:
                texts.append(MOCK_VALID_LEVEL)
            else:
                texts.append(MOCK_INVALID_LEVEL)

        log_probs = torch.randn(batch_size, seq_len)  # (batch, seq_len)
        token_ids = torch.randint(0, 32000, (batch_size, seq_len))  # (batch, seq_len)
        prompt_ids = torch.randint(0, 32000, (batch_size, 32))  # (batch, prompt_len)

        logger.info(
            "[MockLLMPolicy] generate 完成 | batch_size=%d (mock)",
            batch_size,
        )

        return GenerationOutput(
            texts=texts,
            log_probs=log_probs,
            token_ids=token_ids,
            prompt_ids=prompt_ids,
        )

    def get_ref_log_probs(self, token_ids: Tensor, prompt_ids: Tensor) -> Tensor:
        """Mock reference log probs。

        Args:
            token_ids: shape (batch, gen_len).
            prompt_ids: shape (batch, prompt_len).

        Returns:
            Random tensor same shape as token_ids。
        """
        ref_log_probs = torch.randn_like(token_ids.float()) * 0.1  # (batch, gen_len)
        logger.debug("[MockLLMPolicy] get_ref_log_probs (mock)")
        return ref_log_probs

    def update(self, grpo_batch: GRPOBatch) -> dict:
        """Mock GRPO update：回傳隨機 metrics。

        Args:
            grpo_batch: GRPOBatch（不實際使用）。

        Returns:
            dict 包含 mock loss, kl, mean_reward。
        """
        metrics = {
            "loss": random.uniform(0.1, 2.0),
            "policy_loss": random.uniform(0.05, 1.5),
            "kl": random.uniform(0.0, 0.1),
            "mean_reward": grpo_batch.rewards.mean().item(),
        }
        logger.info(
            "[MockLLMPolicy] update 完成 | loss=%.4f (mock)", metrics["loss"],
        )
        return metrics

    def setup_optimizer(self, learning_rate: float, kl_coeff: float = 0.05) -> None:
        """Mock optimizer setup（no-op）。"""
        logger.info("[MockLLMPolicy] setup_optimizer (mock, no-op)")

    def save_checkpoint(self, path: str) -> None:
        """Mock checkpoint save（no-op）。"""
        logger.info("[MockLLMPolicy] save_checkpoint → %s (mock, no-op)", path)

    def generate_with_chat_template(
        self, messages_list: list[list[dict[str, str]]],
    ) -> GenerationOutput:
        """Mock chat template 生成。

        Args:
            messages_list: Chat messages（不實際使用 template）。

        Returns:
            同 generate() 的 mock 輸出。
        """
        prompts = [str(msgs) for msgs in messages_list]
        return self.generate(prompts)

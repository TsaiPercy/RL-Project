"""
metrics.py — 評估指標計算。

Per SPEC §8 主要指標:
  - Parse Success Rate: LLM 輸出成功解析為合法關卡的比例
  - Playability Rate:   可通關關卡佔所有合法關卡的比例
  - Regret:             regret 的統計量（mean, median, std）

所有函式接收 shared types（ParseResult, RewardOutput）作為輸入，
確保可以獨立於 Module A / B 進行測試。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from shared.types import (
    MetricsResult,
    ParseResult,
    RewardOutput,
)

logger = logging.getLogger(__name__)


def computeParseSuccessRate(parseResults: list[ParseResult]) -> float:
    """計算 parse success rate。

    parse_success_rate = 成功解析數 / 總數

    Args:
        parseResults: ParseResult 列表。

    Returns:
        0.0 ~ 1.0 之間的比例值。總數為 0 時回傳 0.0。
    """
    if not parseResults:
        logger.warning("[metrics] computeParseSuccessRate → 空列表, 回傳 0.0")
        return 0.0

    successCount = sum(1 for parseResult in parseResults if parseResult.success)
    rate = successCount / len(parseResults)

    logger.debug(
        "[metrics] computeParseSuccessRate | success=%d / total=%d = %.4f",
        successCount, len(parseResults), rate,
    )
    return rate


def computePlayabilityRate(
    rewardOutputs: list[RewardOutput],
    parseResults: list[ParseResult],
) -> float:
    """計算 playability rate。

    playability_rate = 可通關關卡數 / 成功解析的關卡數

    Args:
        rewardOutputs: RewardOutput 列表（與 parseResults 等長）。
        parseResults:  ParseResult 列表。

    Returns:
        0.0 ~ 1.0 之間的比例值。成功解析數為 0 時回傳 0.0。
    """
    parsedCount = sum(1 for parseResult in parseResults if parseResult.success)

    if parsedCount == 0:
        logger.warning("[metrics] computePlayabilityRate → 無成功解析, 回傳 0.0")
        return 0.0

    playableCount = sum(
        1 for rewardOutput in rewardOutputs if rewardOutput.playable
    )
    rate = playableCount / parsedCount

    logger.debug(
        "[metrics] computePlayabilityRate | playable=%d / parsed=%d = %.4f",
        playableCount, parsedCount, rate,
    )
    return rate


def computeRegretStats(rewardOutputs: list[RewardOutput]) -> dict:
    """計算 regret 統計量（僅計算 playable 的關卡）。

    Args:
        rewardOutputs: RewardOutput 列表。

    Returns:
        {"mean": float, "median": float, "std": float}
        若無 playable 關卡，所有值為 0.0。
    """
    playableRegrets = [
        rewardOutput.regret
        for rewardOutput in rewardOutputs
        if rewardOutput.playable
    ]

    if not playableRegrets:
        logger.warning("[metrics] computeRegretStats → 無 playable 關卡, 所有值為 0.0")
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    regretArray = np.array(playableRegrets)
    stats = {
        "mean": float(np.mean(regretArray)),
        "median": float(np.median(regretArray)),
        "std": float(np.std(regretArray)),
    }

    logger.debug(
        "[metrics] computeRegretStats | n=%d, mean=%.4f, median=%.4f, std=%.4f",
        len(playableRegrets), stats["mean"], stats["median"], stats["std"],
    )
    return stats


def computeAllMetrics(
    parseResults: list[ParseResult],
    rewardOutputs: list[RewardOutput],
) -> MetricsResult:
    """彙整所有指標，回傳 MetricsResult。

    Args:
        parseResults:  ParseResult 列表。
        rewardOutputs: RewardOutput 列表（與 parseResults 等長）。

    Returns:
        MetricsResult，包含所有指標和計數。
    """
    parseSuccessRate = computeParseSuccessRate(parseResults)
    playabilityRate = computePlayabilityRate(rewardOutputs, parseResults)
    regretStats = computeRegretStats(rewardOutputs)

    totalLevels = len(parseResults)
    parsedLevels = sum(1 for parseResult in parseResults if parseResult.success)
    playableLevels = sum(
        1 for rewardOutput in rewardOutputs if rewardOutput.playable
    )

    result = MetricsResult(
        parse_success_rate=parseSuccessRate,
        playability_rate=playabilityRate,
        regret_stats=regretStats,
        total_levels=totalLevels,
        parsed_levels=parsedLevels,
        playable_levels=playableLevels,
    )

    logger.info(
        "[metrics] computeAllMetrics | total=%d, parsed=%d (%.1f%%), "
        "playable=%d (%.1f%%), regret_mean=%.4f",
        totalLevels,
        parsedLevels, parseSuccessRate * 100,
        playableLevels, playabilityRate * 100,
        regretStats["mean"],
    )

    return result

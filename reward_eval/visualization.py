"""
visualization.py — 圖表生成。

Per SPEC §13 Visualization Opportunities:
  - Training: Reward curve, Parse success rate over time, Playability rate over time
  - Evaluation: Regret distribution (histogram), Baseline vs GRPO 對比
  - Ablation: Reward weight ablation 對比

使用 matplotlib 生成靜態圖表，儲存至指定路徑。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非互動式後端，適合 server / CI
import matplotlib.pyplot as plt
import numpy as np

from shared.types import EvalReport

logger = logging.getLogger(__name__)

# --- 全域樣式設定 ---
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def plotRewardCurve(
    rewardHistory: list[float],
    savePath: str,
    title: str = "Training Reward Curve",
    windowSize: int = 10,
) -> None:
    """繪製 training reward curve（含移動平均線）。

    Args:
        rewardHistory: 每個 iteration 的平均 reward。
        savePath:      圖片儲存路徑。
        title:         圖表標題。
        windowSize:    移動平均窗口大小。
    """
    fig, ax = plt.subplots()

    iterations = list(range(1, len(rewardHistory) + 1))
    ax.plot(iterations, rewardHistory, alpha=0.4, color="steelblue", label="Per iteration")

    # 移動平均
    if len(rewardHistory) >= windowSize:
        movingAvg = np.convolve(
            rewardHistory, np.ones(windowSize) / windowSize, mode="valid"
        )
        movingAvgIterations = list(range(windowSize, len(rewardHistory) + 1))
        ax.plot(
            movingAvgIterations, movingAvg,
            color="darkblue", linewidth=2,
            label=f"Moving avg (window={windowSize})",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title(title)
    ax.legend()

    _saveFigure(fig, savePath)
    logger.info("[visualization] plotRewardCurve → %s", savePath)


def plotRegretHistogram(
    regretValues: list[float],
    savePath: str,
    title: str = "Regret Distribution",
    bins: int = 30,
) -> None:
    """繪製 regret 分布直方圖。

    Args:
        regretValues: playable 關卡的 regret 值列表。
        savePath:     圖片儲存路徑。
        title:        圖表標題。
        bins:         直方圖的 bin 數量。
    """
    fig, ax = plt.subplots()

    ax.hist(regretValues, bins=bins, color="teal", edgecolor="white", alpha=0.8)

    # 標記 mean 和 median
    if regretValues:
        meanVal = np.mean(regretValues)
        medianVal = np.median(regretValues)
        ax.axvline(meanVal, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {meanVal:.3f}")
        ax.axvline(medianVal, color="orange", linestyle=":", linewidth=1.5, label=f"Median = {medianVal:.3f}")

    ax.set_xlabel("Regret")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()

    _saveFigure(fig, savePath)
    logger.info("[visualization] plotRegretHistogram → %s", savePath)


def plotBaselineComparison(
    baselineReport: EvalReport,
    grpoReport: EvalReport,
    savePath: str,
    title: str = "Baseline vs GRPO Comparison",
) -> None:
    """繪製 baseline vs GRPO 對比 bar chart。

    對比三項指標: Parse Success Rate, Playability Rate, Mean Regret。

    Args:
        baselineReport: baseline（zero-shot 或 few-shot）的 EvalReport。
        grpoReport:     GRPO 訓練後的 EvalReport。
        savePath:       圖片儲存路徑。
        title:          圖表標題。
    """
    metrics = ["Parse Success\nRate", "Playability\nRate", "Mean\nRegret"]
    baselineValues = [
        baselineReport.parse_success_rate,
        baselineReport.playability_rate,
        baselineReport.held_out_regret.get("mean", 0.0),
    ]
    grpoValues = [
        grpoReport.parse_success_rate,
        grpoReport.playability_rate,
        grpoReport.held_out_regret.get("mean", 0.0),
    ]

    fig, ax = plt.subplots()

    xPositions = np.arange(len(metrics))
    barWidth = 0.35

    baselineBars = ax.bar(
        xPositions - barWidth / 2, baselineValues, barWidth,
        label=f"Baseline ({baselineReport.eval_mode})",
        color="lightcoral", edgecolor="white",
    )
    grpoBars = ax.bar(
        xPositions + barWidth / 2, grpoValues, barWidth,
        label=f"GRPO ({grpoReport.eval_mode})",
        color="steelblue", edgecolor="white",
    )

    # 在 bar 上方標數值
    for bar in baselineBars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in grpoBars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xPositions)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()

    _saveFigure(fig, savePath)
    logger.info("[visualization] plotBaselineComparison → %s", savePath)


def plotTrainingProgress(
    metricsHistory: dict[str, list[float]],
    savePath: str,
    title: str = "Training Progress",
) -> None:
    """繪製訓練過程中的多項指標趨勢。

    Args:
        metricsHistory: 字典，key = 指標名稱，value = 每 iteration 的值列表。
                        例如 {"reward": [...], "parse_rate": [...], "playability": [...]}
        savePath:       圖片儲存路徑。
        title:          圖表標題。
    """
    numMetrics = len(metricsHistory)
    if numMetrics == 0:
        logger.warning("[visualization] plotTrainingProgress → 無指標資料")
        return

    fig, axes = plt.subplots(numMetrics, 1, figsize=(10, 4 * numMetrics), sharex=True)

    if numMetrics == 1:
        axes = [axes]

    colors = ["steelblue", "teal", "coral", "mediumpurple", "goldenrod"]

    for idx, (metricName, values) in enumerate(metricsHistory.items()):
        ax = axes[idx]
        iterations = list(range(1, len(values) + 1))
        color = colors[idx % len(colors)]

        ax.plot(iterations, values, color=color, linewidth=1.5)
        ax.set_ylabel(metricName)
        ax.set_title(f"{metricName} over iterations")

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    _saveFigure(fig, savePath)
    logger.info("[visualization] plotTrainingProgress → %s", savePath)


def plotAblationComparison(
    reports: dict[str, EvalReport],
    savePath: str,
    title: str = "Reward Weight Ablation",
) -> None:
    """繪製 ablation 實驗對比圖。

    Args:
        reports:  字典，key = 配置名稱（如 "regret_w=0.5"），value = EvalReport。
        savePath: 圖片儲存路徑。
        title:    圖表標題。
    """
    if not reports:
        logger.warning("[visualization] plotAblationComparison → 無報告資料")
        return

    configNames = list(reports.keys())
    parseRates = [report.parse_success_rate for report in reports.values()]
    playabilityRates = [report.playability_rate for report in reports.values()]
    meanRegrets = [
        report.held_out_regret.get("mean", 0.0) for report in reports.values()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Parse Success Rate
    axes[0].bar(configNames, parseRates, color="steelblue", edgecolor="white")
    axes[0].set_title("Parse Success Rate")
    axes[0].set_ylabel("Rate")
    axes[0].tick_params(axis="x", rotation=30)

    # Playability Rate
    axes[1].bar(configNames, playabilityRates, color="teal", edgecolor="white")
    axes[1].set_title("Playability Rate")
    axes[1].set_ylabel("Rate")
    axes[1].tick_params(axis="x", rotation=30)

    # Mean Regret
    axes[2].bar(configNames, meanRegrets, color="coral", edgecolor="white")
    axes[2].set_title("Mean Regret")
    axes[2].set_ylabel("Regret")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    _saveFigure(fig, savePath)
    logger.info("[visualization] plotAblationComparison → %s", savePath)


# ------------------------------------------------------------------
# 內部輔助
# ------------------------------------------------------------------

def _saveFigure(fig: plt.Figure, savePath: str) -> None:
    """儲存圖表並關閉 figure，避免記憶體洩漏。"""
    outputPath = Path(savePath)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputPath, bbox_inches="tight")
    plt.close(fig)
    logger.debug("[visualization] _saveFigure → %s", outputPath)

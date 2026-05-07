"""訓練 path 記憶體 smoke test — 驗證 GRPO update() 在 4090 撐得住。

Per SPEC §11.2 Module A [impl-updated 2026-05-07]: 完整訓練 path 包含
generate(compute_log_probs=True) → get_ref_log_probs → update()，本腳本
在不依賴 Module B / Module C（用 mock rewards）的情況下驗證這條 path
的記憶體峰值。

從 config 讀 model / lora / micro_batch_size 等設定；CLI 可覆寫 batch_size、
group_size、num_iterations、micro_batch_size 來縮小規模快速驗證。

Usage:
  # 先用最小規模確認能跑（B=1 prompt, G=2 → 2 條序列, micro=1）
  python -m toy_case.train_smoke_test --batch-size 1 --group-size 2 --num-iterations 2

  # 再用 config 真實規模試（B=4×G=16=64 序列, micro=1）
  python -m toy_case.train_smoke_test --num-iterations 1
"""

from __future__ import annotations

# Set project-local cache dirs before any ML library is imported.
from shared.env_setup import setup_project_cache
setup_project_cache()

import argparse
import logging
from typing import Optional

import torch
import yaml

from shared.types import GRPOBatch

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """載入 YAML 配置。"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_gpu_memory(tag: str) -> None:
    """印出當前 / 峰值 GPU 記憶體（GB）。"""
    if not torch.cuda.is_available():
        logger.info("[mem][%s] CUDA 不可用", tag)
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    logger.info(
        "[mem][%s] allocated=%.2f GB | reserved=%.2f GB | peak=%.2f GB",
        tag, allocated, reserved, peak,
    )


def build_mock_grpo_batch(
    prompt_ids: torch.Tensor,
    token_ids: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    group_size: int,
) -> GRPOBatch:
    """用真實生成的 token + mock rewards 建 GRPOBatch。

    rewards 從 N(0, 1) 抽樣；advantages 用同一 prompt group 內 z-score 標準化
    （等價 reward_eval.compute_advantages_grpo 的行為，但避免依賴 Module C）。

    Args:
        prompt_ids: shape (B, prompt_len).
        token_ids: shape (B, gen_len).
        log_probs: shape (B, gen_len).
        ref_log_probs: shape (B, gen_len).
        group_size: 每 group 多少條（B 必為 group_size 倍數）。

    Returns:
        GRPOBatch 實例。
    """
    actual_b = token_ids.shape[0]
    assert actual_b % group_size == 0, (
        f"batch size {actual_b} 不是 group_size {group_size} 的倍數"
    )

    rewards = torch.randn(actual_b, device=token_ids.device)

    n_groups = actual_b // group_size
    rewards_grouped = rewards.view(n_groups, group_size)
    means = rewards_grouped.mean(dim=1, keepdim=True)
    stds = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-6)
    advantages = ((rewards_grouped - means) / stds).view(actual_b)

    return GRPOBatch(
        token_ids=token_ids,
        prompt_ids=prompt_ids,
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        advantages=advantages,
    )


def run_smoke_test(
    config: dict,
    batch_size: int,
    group_size: int,
    num_iterations: int,
    micro_batch_size: Optional[int],
) -> None:
    """跑訓練 path：generate → ref → mock reward → update。

    Args:
        config: 完整 config dict。
        batch_size: prompts per iteration。
        group_size: samples per prompt（GRPO group size）。
        num_iterations: 跑幾次完整 update step。
        micro_batch_size: 傳給 update()；None 時走舊行為（不切）。
    """
    from llm_policy.policy import LLMPolicy
    from llm_policy.prompts import (
        format_chat_messages,
        get_minigrid_prompt,
        get_system_prompt,
    )

    logger.info("=" * 60)
    logger.info("訓練 smoke test 啟動")
    logger.info(
        "batch_size=%d, group_size=%d → B=%d 條/iter, micro=%s, iters=%d",
        batch_size, group_size, batch_size * group_size,
        "off" if micro_batch_size is None else str(micro_batch_size),
        num_iterations,
    )
    logger.info("=" * 60)

    llm = LLMPolicy(
        model_name=config.get("model_name", "Qwen/Qwen3.5-9B"),
        quantization=config.get("quantization", "4bit"),
        lora_rank=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 128),
        lora_target_modules=config.get("lora_target_modules"),
        max_new_tokens=config.get("max_new_tokens", 2048),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.8),
        top_k=config.get("top_k", 20),
        presence_penalty=config.get("presence_penalty", 1.5),
        enable_thinking=config.get("enable_thinking", False),
        cache_dir=config.get("cache_dir"),
    )
    llm.setup_optimizer(
        learning_rate=config.get("learning_rate", 1.0e-5),
        kl_coeff=config.get("kl_coeff", 0.05),
    )
    log_gpu_memory("model loaded")

    system_prompt = get_system_prompt()
    user_prompt = get_minigrid_prompt()
    messages = format_chat_messages(system_prompt, user_prompt)
    # 同 prompt 重複 (batch_size × group_size) 次 — GRPO 要求同 group 共享 prompt
    messages_list = [messages] * (batch_size * group_size)

    for it in range(num_iterations):
        logger.info("--- iteration %d/%d ---", it + 1, num_iterations)

        torch.cuda.reset_peak_memory_stats()

        # Step 1: generate（訓練 path → compute_log_probs=True）
        gen = llm.generate_with_chat_template(
            messages_list, compute_log_probs=True,
        )
        log_gpu_memory("after generate")

        # Step 2: reference log probs（用 disable_adapter 共用 base）
        ref_log_probs = llm.get_ref_log_probs(gen.token_ids, gen.prompt_ids)
        log_gpu_memory("after ref_log_probs")

        # Step 3: mock rewards + advantages
        batch = build_mock_grpo_batch(
            prompt_ids=gen.prompt_ids,
            token_ids=gen.token_ids,
            log_probs=gen.log_probs,
            ref_log_probs=ref_log_probs,
            group_size=group_size,
        )

        # Step 4: GRPO update（這是 OOM 最容易發生的地方）
        metrics = llm.update(batch, micro_batch_size=micro_batch_size)
        log_gpu_memory("after update")

        logger.info(
            "iter %d metrics: loss=%.4f, kl=%.4f, mean_reward=%.4f",
            it + 1, metrics["loss"], metrics["kl"], metrics["mean_reward"],
        )

    logger.info("=" * 60)
    logger.info("Smoke test PASSED")
    log_gpu_memory("final")
    logger.info("=" * 60)


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="Training-path smoke test for LLMPolicy.update() memory.",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size (prompts per iter). Default: config value.",
    )
    parser.add_argument(
        "--group-size", type=int, default=None,
        help="Override group_size. Default: config value.",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=1,
        help="Number of update steps to run (default: 1, just verify it doesn't OOM).",
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Override micro_batch_size. Default: config value.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 4)
    group_size = args.group_size if args.group_size is not None else config.get("group_size", 16)
    micro_batch_size = (
        args.micro_batch_size if args.micro_batch_size is not None
        else config.get("micro_batch_size")
    )

    run_smoke_test(
        config=config,
        batch_size=batch_size,
        group_size=group_size,
        num_iterations=args.num_iterations,
        micro_batch_size=micro_batch_size,
    )


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RLVR Game Level Generation** — 強化學習期末專案。使用 GRPO 微調 Qwen3.5-9B，使其生成 MiniGrid（Phase 1-2）和 MiniHack（Phase 3）遊戲關卡。訓練信號完全來自 game agent 的遊玩表現，不使用人類資料。

## Architecture

三個獨立模組，由 `train.py` 串接：

```
LLMPolicy.generate(prompts)           # Module A: llm_policy/
    → list[str] (raw LLM output)
GameEnvironment.batch_evaluate(texts) # Module B: game_env/
    → list[RolloutResult | None]
RewardCalculator.compute_batch_rewards(rollouts)  # Module C: reward_eval/
    → list[RewardOutput]
RewardCalculator.compute_advantages_grpo(rewards, group_size)
    → Tensor (GRPO normalized advantages)
LLMPolicy.update(GRPOBatch)           # back to Module A
```

每個模組的 API 規範在 `docs/sdd.md`，所有共用 dataclass 定義在 `shared/types.py`，範例實例在 `shared/type_examples.py`。

## Planned Directory Structure

```
llm_policy/          # Module A: LLMPolicy, GRPO update, prompt templates
game_env/            # Module B: parser, environment wrapper, agent training
  agents/            # SB3 agent training scripts
  minihack/          # Phase 3 MiniHack extension
reward_eval/         # Module C: RewardCalculator, EvaluationSuite, metrics
train.py             # Main training loop (Module A integrates all three)
evaluate.py          # Standalone evaluation script (Module C)
baselines/           # Zero-shot / few-shot baseline scripts
checkpoints/         # Model & agent checkpoints
  agents/            # Pretrained game agent checkpoints
config/              # YAML configs (default.yaml, minigrid.yaml, minihack.yaml)
shared/types.py      # Shared dataclasses (canonical source of truth)
```

## Key Data Contracts

**MiniGrid level format** (LLM output, defined by Module A, consumed by Module B) [impl-updated 2026-05-07]:
LLM 輸出完整 15×15 ASCII grid（含外牆 ring）+ 隨後的 JSON（objects + agent_start）。
外牆為固定 'W' ring（row/col 0 與 14），objects 與 agent_start 座標限制在 inner area `[1, 13]`。
詳見 `llm_policy/prompts.py:MINIGRID_LEVEL_PROMPT`。
```json
{"width": 15, "height": 15,
 "objects": [{"type": "wall|key|door|ball|box|goal|lava", "x": 2, "y": 3, "color": "yellow"}],
 "agent_start": {"x": 1, "y": 1, "dir": 0}}
```

**Agent pool IDs**: `strong_0`, `strong_1`, `weak_0`, `weak_1` (training); `strong_held_0`, `strong_held_1`, `weak_held_0` (eval only).

**Reward formula**:
```
reward = playability_bonus * I(playable)
       + regret_weight * (mean(strong_returns) - mean(weak_returns))
       + breadth_weight * mean(action_entropy(strong_trajectories))
# If parse fails: invalid_penalty (-1.0)
# If unsolvable: 0.0
```

## Config

All hyperparameters in `config/default.yaml`. Key values:
- `group_size: 4` — GRPO samples per prompt
- `kl_coeff: 0.05`, `learning_rate: 1e-5`, `batch_size: 16`
- `num_rollouts_per_agent: 5`
- `agent_pool_path: "checkpoints/agents/"`

## Development Conventions

- **Mock-first**: 各模組開發初期須先撰寫符合 SDD API 的 mock，讓其他模組可以獨立測試
- **Module integration**: 所有 API 變更須通知相關成員並更新 `docs/sdd.md`
- **Logging**: 使用 Python `logging` + Weights & Biases (`wandb_project: "rlvr-level-gen"`)
- **Checkpoints**: 存放於 `checkpoints/{experiment_name}/step_{n}/`
- **Branches**: main branch + personal feature branches，PR review 後 merge

## Key Dependencies

| 套件 | 用途 |
|------|------|
| Transformers + PEFT | Qwen3.5-9B 載入 + LoRA (rank=64, alpha=128) |
| TRL | GRPOTrainer（或自行實作 GRPO） |
| MiniGrid ≥2.3 | 遊戲環境 (Phase 1-2) |
| MiniHack ≥0.1.5 | 遊戲環境 (Phase 3) |
| Stable-Baselines3 ≥2.0 | 訓練 game agent pool |
| vLLM ≥0.4 | 加速 LLM 推理 (optional) |
| wandb | Experiment tracking |

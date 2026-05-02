# AIREADME.md — Antigravity 專案內部記憶文件

> 本文件僅供 Google Antigravity 自身閱讀，作為專案架構與狀態的唯一可信來源。

---

## 1. 專案概述

**RLVR Game Level Generation** — 使用 GRPO（Group Relative Policy Optimization）微調 Qwen3.5-Coder-9B，使其生成 MiniGrid 遊戲關卡。訓練信號完全來自 agent pool 的 regret index，不使用任何人類設計的關卡資料。

**核心假設**：LLM 預訓練知識 + GRPO 微調 → 生成能區分強弱玩家技巧水準的關卡。

## 2. 系統架構

```
train.py (orchestrator)
    │
    ├── Module A: llm_policy/     → LLMPolicy (Qwen3.5-9B QLoRA)
    │       ├── policy.py         → generate(), update(), get_ref_log_probs()
    │       ├── grpo.py           → GRPO update 邏輯（目前整合在 policy.py）
    │       └── prompts.py        → Prompt template
    │
    ├── Module B: game_env/       → GameEnvironment
    │       ├── parser.py         → ASCII grid + JSON 解析 + 驗證
    │       ├── environment.py    → MiniGrid 環境 + SubprocVecEnv rollout
    │       ├── wrappers.py       → BabyAI agent 相容 wrapper
    │       └── minihack/         → Phase 3 擴展
    │
    ├── Module C: reward_eval/    → RewardCalculator + EvaluationSuite
    │       ├── reward.py         → regret-based reward 計算
    │       ├── evaluation.py     → quick/full evaluation（agent pool 由 mode 選擇）
    │       ├── metrics.py        → parse rate / playability / regret 統計
    │       ├── visualization.py  → matplotlib 圖表生成
    │       └── mock.py           → MockRewardCalculator（測試用）
    │
    ├── shared/                   → 跨模組 contract
    │       ├── types.py          → 所有 shared dataclass
    │       ├── type_examples.py  → 範例實例
    │       └── env_setup.py      → project-local cache 設定
    │
    ├── toy_case/                 → Phase 0: Toy Case
    │       ├── train_agent.py    → SB3 PPO agent 訓練 (DoorKeyEnv)
    │       ├── run_toy_pipeline.py → 端到端 pipeline smoke test
    │       └── sanity_check.py   → Experiment 0 sanity check
    │
    └── agent_training/           → Phase 1-2: Curriculum Agent 訓練
            ├── train_curriculum.py → BabyAI GoTo curriculum 訓練
            └── evaluate_agent.py  → Agent 表現評估
```

### 模組間通訊

所有模組間通訊均透過 `train.py` 中介（indirect function call），模組不直接互相呼叫。資料流：

```
LLMPolicy.generate() → texts
    ↓
GameEnvironment.batch_evaluate(texts, num_rollouts, agent_ids=...) → list[RolloutResult | None]
    ↓
RewardCalculator.compute_batch_rewards(rollouts) → list[RewardOutput]
    ↓
RewardCalculator.compute_advantages_grpo(rewards, groupSize) → Tensor
    ↓
LLMPolicy.update(GRPOBatch)
```

## 3. Shared Types（Contract）

定義在 `shared/types.py`，所有跨模組邊界的資料結構：

| Type | Producer | Consumer |
|------|----------|----------|
| `GenerationOutput` | Module A | train.py → Module B |
| `GRPOBatch` | train.py | Module A |
| `ParseResult` | Module B | Module B / train.py |
| `Trajectory` | Module B | Module C |
| `RolloutResult` | Module B | Module C |
| `RewardConfig` | config/default.yaml | Module C |
| `RewardOutput` | Module C | train.py |
| `EvalReport` | Module C | evaluate.py / wandb |
| `MetricsResult` | Module C (metrics.py) | evaluate.py / EvaluationSuite |

## 4. 各模組實作狀態

### Module A: LLM Policy (`llm_policy/`)
- **狀態**: ✅ 基本完成（成員 A: YouZhe）
- **已完成**: `policy.py`（QLoRA 載入 + generate + update + get_ref_log_probs）, `prompts.py`（prompt template）, mock
- **待完成**: `grpo.py` 獨立抽出（目前整合在 policy.py）

### Module B: Game Environment (`game_env/`)
- **狀態**: ⬜ 未實作（成員 B 負責）
- **預期檔案**: `parser.py`, `environment.py`, `wrappers.py`
- **注意**: `batch_evaluate()` 需支援 `agent_ids` keyword argument（Per SPEC_MODIFICATION #15）

### Module C: Reward & Evaluation (`reward_eval/`)
- **狀態**: ✅ 全部完成（成員 C: Percy）
- **已完成**:
  - `reward.py` — `RewardCalculator`（compute_reward, compute_batch_rewards, compute_advantages_grpo）
  - `metrics.py` — `computeParseSuccessRate`, `computePlayabilityRate`, `computeRegretStats`, `computeAllMetrics`
  - `evaluation.py` — `EvaluationSuite`（持有 GameEnvironment 參考，evaluate, exportReport）
    - **已修正**: `mode` 根據 config 選擇 `training_agents` / `held_out_agents`，傳入 `batch_evaluate(agent_ids=...)`
  - `visualization.py` — 5 個圖表生成函式
  - `mock.py` — `MockRewardCalculator`（供 A/B 測試用）

### Shared Types (`shared/`)
- **狀態**: ✅ 完成
- 已包含 `MetricsResult`（成員 C 新增）
- 已包含 `prompt_ids`（成員 A 新增）

### 腳本
- `evaluate.py` — ✅ 完成（LLMPolicy/GameEnvironment 以 TODO 標記待整合）
- `baselines/run_baseline.py` — ✅ 完成（同上）
- `toy_case/` — ✅ 完成（成員 A: train_agent.py, run_toy_pipeline.py, sanity_check.py）
- `train.py` — ⬜ 未實作

## 5. SPEC 重大變更記錄

### 2026-05-02 更新
- **Agent Pool**: 從 BabyAI pretrained → 自訓練 BabyAI GoTo 家族 curriculum（9 關，由簡到難）
- **環境**: 統一 GoTo 任務（非多任務），環境配置由 `BabyAI-GoTo-v0` 至 `BabyAI-GoToObjMazeS7-v0`
- **Strong/Weak 差異**: 透過 `success_increase`（加法 delta）與 `curriculum_levels` 控制
- **Phase 0 Toy Case**: 新增 SB3 PPO on DoorKeyEnv(room_size=15) 作為 pipeline smoke test

## 6. 設計決策記錄

| 決策 | 選擇 | 理由 |
|------|------|------|
| EvaluationSuite 架構 | 持有 GameEnvironment 參考（Option A） | SPEC §11 設計 |
| EvaluationSuite mode | 根據 mode 傳 agent_ids 給 batch_evaluate | SPEC §8 要求 quick/full 使用不同 agent pool |
| MetricsResult 位置 | `shared/types.py` | 跨模組使用 |
| BaselineConfig | 移除，改用 argparse | 簡潔性 |
| GRPO update 位置 | 整合在 policy.py（成員 A 決策） | 緊密耦合 |
| 命名風格 | 函式/變數: camelCase，類別: PascalCase | 使用者規則 §1 |

## 7. 環境設定

- **Conda 環境名稱**: `RL_project`（團隊統一）
- **Python 版本**: 3.10
- **專案路徑**: `C:\homework\RL-Project`

## 8. 流程管理文件

本專案採用 SKILL.md（sdd-ai-coding）定義的流程：
- **SPEC.md**: 規範（`docs/SPEC.md`）
- **TODO.md**: 任務清單（`docs/TODO.md`）
- **IMPLEMENT_RECORD.md**: 自主決策記錄（`docs/IMPLEMENT_RECORD.md`）
- **SPEC_MODIFICATION.md**: SPEC 變更日誌（`docs/SPEC_MODIFICATION.md`）
- **AMBIGUITY.md**: 歧義記錄（`docs/AMBIGUITY.md`）

## 9. 待辦事項

### 整合待辦
- [ ] INT-3: 用 mock 模組跑通 train.py
- [ ] INT-4: 替換 mock 為真實模組
- [ ] 與成員 B 確認 `batch_evaluate()` 是否接受 `agent_ids` 參數

### 成員 C 後續實驗
- [ ] B-2~B-4: Baseline 實驗
- [ ] M-3, M-5: 評估實驗
- [ ] A-1~A-4: Ablation 實驗
- [ ] R-1, R-2: 彙整結果 + 視覺化

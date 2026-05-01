# AIREADME.md — Antigravity 專案內部記憶文件

> 本文件僅供 Google Antigravity 自身閱讀，作為專案架構與狀態的唯一可信來源。

---

## 1. 專案概述

**RLVR Game Level Generation** — 使用 GRPO（Group Relative Policy Optimization）微調 Qwen3.5-9B，使其生成 MiniGrid / MiniHack 遊戲關卡。訓練信號完全來自 agent pool 的 regret index，不使用人類設計資料。

**核心假設**：LLM 預訓練知識 + GRPO 微調 → 生成能區分強弱玩家技巧水準的關卡。

## 2. 系統架構

```
train.py (orchestrator)
    │
    ├── Module A: llm_policy/     → LLMPolicy (Qwen3.5-9B QLoRA)
    │       ├── policy.py         → generate(), update(), get_ref_log_probs()
    │       ├── grpo.py           → GRPO update 邏輯
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
    │       ├── evaluation.py     → quick/full evaluation 流程
    │       ├── metrics.py        → parse rate / playability / regret 統計
    │       ├── visualization.py  → matplotlib 圖表生成
    │       └── mock.py           → MockRewardCalculator（測試用）
    │
    └── shared/                   → 跨模組 contract
            ├── types.py          → 所有 shared dataclass
            └── type_examples.py  → 範例實例
```

### 模組間通訊

所有模組間通訊均透過 `train.py` 中介（indirect function call），模組不直接互相呼叫。資料流：

```
LLMPolicy.generate() → texts
    ↓
GameEnvironment.batch_evaluate(texts) → list[RolloutResult | None]
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
- **狀態**: ⬜ 未實作（成員 A 負責）
- **預期檔案**: `policy.py`, `grpo.py`, `prompts.py`

### Module B: Game Environment (`game_env/`)
- **狀態**: ⬜ 未實作（成員 B 負責）
- **預期檔案**: `parser.py`, `environment.py`, `wrappers.py`

### Module C: Reward & Evaluation (`reward_eval/`)
- **狀態**: ✅ 全部完成（成員 C 負責）
- **已完成檔案**:
  - `reward.py` — `RewardCalculator`（compute_reward, compute_batch_rewards, compute_advantages_grpo）
  - `metrics.py` — `computeParseSuccessRate`, `computePlayabilityRate`, `computeRegretStats`, `computeAllMetrics`
  - `evaluation.py` — `EvaluationSuite`（持有 GameEnvironment 參考，evaluate, exportReport）
  - `visualization.py` — `plotRewardCurve`, `plotRegretHistogram`, `plotBaselineComparison`, `plotTrainingProgress`, `plotAblationComparison`
  - `mock.py` — `MockRewardCalculator`（供 A/B 測試用）

### Shared Types (`shared/`)
- **狀態**: ✅ 完成
- `types.py` — 所有 9 個 shared dataclass
- `type_examples.py` — 各 type 的範例實例
- `__init__.py` — 匯出所有 types

### 腳本
- `evaluate.py` — ✅ 完成（LLMPolicy/GameEnvironment 以 TODO 標記待整合）
- `baselines/run_baseline.py` — ✅ 完成（同上）
- `train.py` — ⬜ 未實作（成員 A 負責）

## 5. 設計決策記錄

| 決策 | 選擇 | 理由 |
|------|------|------|
| EvaluationSuite 架構 | 持有 GameEnvironment 參考（Option A） | SPEC §11 設計，內部呼叫 batch_evaluate() |
| MetricsResult 位置 | `shared/types.py`（非 local type） | 跨 metrics.py / evaluate.py / EvaluationSuite 使用 |
| BaselineConfig | 移除，改用 argparse | run_baseline.py 為獨立腳本，argparse 直接處理設定更簡潔 |
| 命名風格 | 函式/變數: camelCase，類別: PascalCase | 遵循使用者規則 §1 |

## 6. 環境設定

- **Conda 環境名稱**: `RL_Final_Project`
- **Python 版本**: 3.10
- **已安裝依賴**: numpy, torch, pyyaml, matplotlib, scipy
- **Python 執行路徑**: `C:\Users\percy\miniconda3\envs\RL_Final_Project\python.exe`
- **專案路徑**: `C:\homework\RL-Project`

## 7. 待辦事項

### 成員 C 待完成
- [ ] B-2: 執行 Zero-shot baseline，收集指標（需 Module A/B 完成）
- [ ] B-3: 篩選 parse-valid 關卡作為 few-shot examples
- [ ] B-4: 執行 Few-shot baseline
- [ ] M-3: 訓練中期 quick eval
- [ ] M-5: Full eval + baseline 對比
- [ ] A-1~A-4: Ablation 實驗
- [ ] R-1, R-2: 彙整結果 + 最終視覺化

### 整合待辦
- [ ] INT-3: 用 mock 模組跑通 train.py
- [ ] INT-4: 替換 mock 為真實模組

## 8. 已知限制與技術債

1. `evaluate.py` 和 `baselines/run_baseline.py` 中 LLMPolicy/GameEnvironment 初始化為 TODO placeholder
2. `EvaluationSuite.evaluate()` 的 agent 切換（quick vs full）依賴 GameEnvironment 實作
3. `visualization.py` 使用 `Agg` backend（非互動式），適合 server/CI
4. `verify_imports.py` 為臨時驗證腳本，可在整合完成後移除

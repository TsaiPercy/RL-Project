# PM Dashboard — RLVR Game Level Generation

> **今日日期**: 2026-04-28
> **專案進度**: Phase 1 進行中

---

## 整體進度總覽

| 階段 | 週次 | 目標 | 狀態 |
|------|------|------|------|
| Phase 1 | Week 1 | Baseline Pipeline on MiniGrid | 🔄 進行中 |
| Phase 2 | Week 2 | GRPO Training on MiniGrid | ⏳ 未開始 |
| Phase 3 | Week 3 | Scale to MiniHack + Final Report | ⏳ 未開始 |

---

## Phase 1 — Week 1：Baseline Pipeline

### Day 1 (Mon) — 共同啟動日

| # | 任務 | 負責人 | 完成 |
|---|------|--------|------|
| 1 | 確認 `Trajectory` / `RolloutResult` class 具體定義與範例 | B | ⬜ |
| 2 | Sync meeting：確認所有 dataclass 定義，寫入 `shared/types.py`（含範例） | A, B, C | ⬜ |
| 3 | 確認 LLM 輸出 JSON 格式，A 提供 10 筆手寫範例給 B | A → B | ⬜ |
| 4 | 各自撰寫 mock 實作並 push 到各自 branch | A, B, C | ⬜ |
| 5 | 建立 Git repo、wandb project、共用 config | A | ⬜ |

**Day 1 產出物**:
- `shared/types.py` — 全部 dataclass 定義 ⬜
- `mock_game_env.py` — A 的 MockGameEnvironment ⬜
- `mock_reward.py` — B 的 MockRewardCalculator ⬜
- `mock_rollout_data.py` — C 的 generate_mock_rollouts ⬜
- `config/default.yaml` ⬜

---

### Day 2–3 (Tue–Wed)

#### 成員 A — LLM Policy

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A1 | 實作 `LLMPolicy`（載入 Qwen3.5-9B、generate、prompt template） | `llm_policy/policy.py`, `prompts.py` | 能在 single GPU 上 generate 一批關卡文字 | ⬜ |
| A2 | 撰寫 zero-shot baseline 腳本 | `baselines/run_baseline.py` | 用 mock env 跑通完整流程 | ⬜ |

#### 成員 B — Game Environment

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B1 | 實作 MiniGrid level parser + validator | `game_env/parser.py` | 能解析 A 的 10 筆範例，正確處理 edge cases | ⬜ |
| B2 | 實作 environment wrapper + rollout runner | `game_env/environment.py` | 能在 MiniGrid 上用 random agent 跑 rollout | ⬜ |
| B3 | 開始訓練 MiniGrid agents (PPO/DQN) | `game_env/agents/` | 訓練腳本可執行，開始跑 | ⬜ |

#### 成員 C — Reward & Evaluation

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C1 | 實作 `RewardCalculator`（regret + breadth + playability） | `reward_eval/reward.py` | 用 mock data 通過 unit test | ⬜ |
| C2 | 實作 `compute_advantages_grpo()` | `reward_eval/reward.py` | 數學正確性 unit test 通過 | ⬜ |
| C3 | 實作基本 evaluation metrics（Playability Rate, Regret） | `reward_eval/metrics.py` | 用 mock data 通過 | ⬜ |

---

### Day 4–5 (Thu–Fri)

#### 成員 A

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A3 | 用真實 `GameEnvironment` 替換 mock，跑 zero-shot baseline | baseline 結果 | 取得 playability rate + regret 數據 | ⬜ |
| A4 | 撰寫 few-shot (self-generated) baseline | 額外 baseline 結果 | 篩選 zero-shot 高品質輸出作 few-shot examples | ⬜ |

#### 成員 B

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B4 | Agent 訓練完成（training set 至少 4 個 agent） | agent checkpoints | strong agent win rate > 80% on random levels | ⬜ |
| B5 | 整合測試：A 的真實 LLM 輸出 → parser → rollout | 測試報告 | parse success rate 統計 + 修 bug | ⬜ |

#### 成員 C

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C4 | 接收 baseline 結果，跑完整 evaluation | Phase 1 evaluation report | Playability Rate, Regret 分布圖 | ⬜ |
| C5 | 實作 Solution Diversity (JSD) | `reward_eval/metrics.py` | 計算正確 | ⬜ |

---

### Phase 1 Checkpoint ✓

| 驗收項目 | 完成 |
|---------|------|
| Zero-shot baseline：Playability Rate, Held-out Regret, Solution Diversity 數據產出 | ⬜ |
| Few-shot baseline：同上三項數據 | ⬜ |
| Pipeline 端到端可跑通：prompt → LLM → parse → rollout → reward → log | ⬜ |
| Agent pool 訓練完成（至少 training set 所需的 4 個 agent） | ⬜ |
| 所有 module 的真實實作已替換 mock | ⬜ |

**預期結果**: Zero-shot playability rate 30–60%，regret 接近 0。

---

## Phase 2 — Week 2：GRPO Training on MiniGrid

### Day 6–7 (Mon–Tue)

#### 成員 A

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A5 | 實作 GRPO update（或整合 TRL GRPOTrainer） | `llm_policy/grpo.py` | 用 mock reward 跑通一個 training step | ⬜ |
| A6 | 實作 `train.py` 主迴圈，串接三個模組 | `train.py` | 端到端 1 iteration 跑通（真實模組） | ⬜ |

#### 成員 B

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B6 | 完成 held-out agent 訓練 | held-out checkpoints | 3 個 held-out agents ready | ⬜ |
| B7 | 優化 rollout 速度（多進程） | 更新 `environment.py` | batch_evaluate 延遲降低 ≥ 2x | ⬜ |

#### 成員 C

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C6 | 實作 Controllability metric (Cohen's d) | `reward_eval/metrics.py` | 用 mock data 通過 | ⬜ |
| C7 | 實作 `EvaluationSuite.evaluate()` 完整流程 | `reward_eval/evaluation.py` | 一鍵跑所有 metrics | ⬜ |

---

### Day 8–9 (Wed–Thu)

#### 成員 A

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A7 | 開始正式 GRPO 訓練（目標 ~200–500 iterations） | training logs + checkpoints | reward curve 有上升趨勢 | ⬜ |
| A8 | 監控訓練，調參（lr, kl_coeff, temperature, group_size） | wandb logs | 訓練穩定不 crash | ⬜ |

#### 成員 B

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B8 | 支援 A 的 debug（parser edge cases、環境問題） | bug fixes | parse success rate 穩定 | ⬜ |
| B9 | 研究 MiniHack des-file 格式（Phase 3 預備） | 格式文件 | 了解 des-file syntax | ⬜ |

#### 成員 C

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C8 | 用訓練中間 checkpoint 做 evaluation，觀察趨勢 | 中間報告 | 確認 metrics 隨訓練改善 | ⬜ |
| C9 | 設計 reward weight ablation 實驗（2–3 組不同 weight） | ablation config | 準備完成 | ⬜ |

---

### Day 10 (Fri)

| # | 任務 | 負責人 | 完成 |
|---|------|--------|------|
| 1 | 訓練收尾，儲存 best checkpoint | A | ⬜ |
| 2 | 確認 pipeline 穩定性統計（parse failure rate < 10%） | B | ⬜ |
| 3 | Phase 2 完整 evaluation（所有 5 項 metrics），與 baseline 對比表格 + 圖 | C | ⬜ |
| 4 | Phase 2 review meeting | 全員 | ⬜ |

---

### Phase 2 Checkpoint ✓

| 驗收項目 | 完成 |
|---------|------|
| GRPO 訓練完成，reward curve 呈上升趨勢 | ⬜ |
| 訓練後 Playability Rate 顯著高於 zero-shot baseline | ⬜ |
| Held-out Regret 顯著高於 baseline | ⬜ |
| Solution Diversity 不低於 baseline | ⬜ |
| Controllability 初步結果 | ⬜ |
| Ablation：不同 reward weight 的比較 | ⬜ |

**預期結果**: Playability Rate 提升至 70–90%，Regret 顯著高於 baseline。

---

## Phase 3 — Week 3：Scale to MiniHack + Final Report

### Day 11–12 (Mon–Tue)

#### 成員 A

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A9 | 設計 MiniHack prompt template（des-file 格式） | `llm_policy/prompts.py` 更新 | MiniHack prompt ready | ⬜ |
| A10 | 確認 Qwen3.5-9B 在 MiniHack 的 zero-shot 可行性 | zero-shot MiniHack baseline | parse success > 10% | ⬜ |

#### 成員 B

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B10 | 實作 MiniHack des-file parser | `game_env/minihack/parser.py` | 能解析合法 des-file | ⬜ |
| B11 | 實作 MiniHack environment wrapper | `game_env/minihack/environment.py` | rollout 可執行 | ⬜ |
| B12 | 訓練 MiniHack agent pool | MiniHack agent checkpoints | strong win rate > 50% | ⬜ |

#### 成員 C

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C10 | 開始設計 Human Study 問卷與介面 | `evaluation/human_study/` | Likert scale 問卷 draft | ⬜ |
| C11 | 將 evaluation suite 適配 MiniHack | 更新 `evaluation.py` | 支援雙環境 | ⬜ |

---

### Day 13–14 (Wed–Thu)

#### 成員 A

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| A11 | MiniHack GRPO 訓練 | training logs | reward 有改善趨勢 | ⬜ |
| A12 | 跑 reward ablation 補充實驗（如需） | ablation results | — | ⬜ |

#### 成員 B

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| B13 | 支援 MiniHack pipeline debug | bug fixes | 穩定執行 | ⬜ |
| B14 | 實作 PCGRL baseline（如時間允許） | PCGRL 結果 | — | ⬜ |

#### 成員 C

| # | 任務 | 產出 | 完成標準 | 完成 |
|---|------|------|---------|------|
| C12 | MiniHack evaluation（所有 metrics 產出） | 數據 | 完成 | ⬜ |
| C13 | 跨環境對比分析（MiniGrid vs MiniHack） | 對比報告 | — | ⬜ |

---

### Day 15 (Fri) — 收尾日

| # | 任務 | 負責人 | 完成 |
|---|------|--------|------|
| 1 | 整理所有訓練 log、checkpoint | A | ⬜ |
| 2 | 整理環境程式碼、agent checkpoint | B | ⬜ |
| 3 | 彙整 final report：所有 metrics、圖表、分析 | C | ⬜ |
| 4 | Final review + 報告撰寫分工 | 全員 | ⬜ |

---

### Phase 3 Checkpoint ✓

| 驗收項目 | 完成 |
|---------|------|
| MiniHack pipeline 端到端可跑 | ⬜ |
| MiniHack 上的 GRPO 訓練結果 | ⬜ |
| MiniGrid vs MiniHack 跨環境對比 | ⬜ |
| Human Study 至少有 preliminary 結果（或設計完成） | ⬜ |
| 完整實驗報告 | ⬜ |

---

## Agent Pool 訓練追蹤

| Agent ID | 類型 | 訓練步數 | 用途 | 狀態 |
|----------|------|---------|------|------|
| `strong_0` | PPO, seed=0 | 10M | Training reward | ⬜ |
| `strong_1` | PPO, seed=1 | 10M | Training reward | ⬜ |
| `weak_0` | PPO, seed=0 | 500K | Training reward | ⬜ |
| `weak_1` | PPO, seed=1 | 500K | Training reward | ⬜ |
| `strong_held_0` | PPO, seed=42 | 10M | Evaluation only | ⬜ |
| `strong_held_1` | DQN, seed=0 | 10M | Evaluation only | ⬜ |
| `weak_held_0` | PPO, seed=42 | 500K | Evaluation only | ⬜ |

---

## 成員間交接追蹤

| 交接項目 | From | To | 預計完成 | 狀態 |
|---------|------|----|---------|------|
| LLM 輸出 JSON 格式規範 + 10 筆範例 | A | B | Day 1 | ⬜ |
| `RolloutResult` / `Trajectory` dataclass 定義 | B | C | Day 1 | ⬜ |
| `GameEnvironment` class（符合 SDD API） | B | A | Day 4 | ⬜ |
| Training agents checkpoints（4 個） | B | A | Day 4–5 | ⬜ |
| `RewardCalculator` + `compute_advantages_grpo()` | C | A | Day 4 | ⬜ |
| Held-out agent checkpoints（3 個） | B | C | Day 7 | ⬜ |
| Phase 1 model checkpoint | A | C | Day 5 | ⬜ |
| Phase 2 best checkpoint | A | C | Day 10 | ⬜ |

---

## 關鍵風險追蹤

| 風險 | 可能性 | 觸發條件 | 應對方案 | 狀態 |
|------|--------|---------|---------|------|
| Parse success rate 太低（< 30%） | 中 | Day 4–5 整合測試 | 簡化格式、加 heuristic 修復、降低 grid size | 👀 監控中 |
| Agent 訓練超時 | 低 | Day 3 未開始跑 | 減少訓練步數先行，降低 rollout 次數 | 👀 監控中 |
| GRPO 訓練不穩定 | 中 | Day 8–9 reward 不上升 | 調降 lr、增大 group_size、換 PPO | 👀 監控中 |
| MiniHack des-file 太複雜 | 高 | Day 11 zero-shot < 10% | 限制 subset、用簡單任務、視為 negative result | 👀 監控中 |
| GPU 資源不足 | 低 | 訓練速度明顯過慢 | 減少 batch size、gradient accumulation、降 LoRA rank | 👀 監控中 |

---

*最後更新：2026-04-28*
*使用方式：任務完成後將 ⬜ 改為 ✅*

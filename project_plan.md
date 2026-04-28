# Project Timeline

## RLVR Game Level Generation — 三階段三週計畫

---

## 總覽

| 階段 | 週次 | 目標 | 驗證內容 |
|------|------|------|---------|
| Phase 1 | Week 1 | Baseline on Toy Case | Zero-shot Qwen3.5-9B → MiniGrid，確認 pipeline 可跑通 |
| Phase 2 | Week 2 | Our Method on Toy Case | Qwen3.5-9B + GRPO + regret reward → MiniGrid，驗證 RL 微調有效 |
| Phase 3 | Week 3 | Our Method on Hard Case | 同上方法 → MiniHack，驗證可擴展性 + 完整報告 |

---

## Phase 1 — Week 1：Baseline Pipeline

**目標**: 建立完整 pipeline，跑通 zero-shot baseline，取得基準數據。

### Day 1 (Mon) — 共同啟動日

| 時段 | 事項 | 參與人 |
|------|------|--------|
|凌晨| 確定`Trajectory` class, `RolloutResult` class具體長怎樣，給一個examlpe | B |
| 上午 | Sync meeting：確認所有 dataclass 定義 (`Trajectory`, `RolloutResult`, `RewardOutput`, `GenerationOutput`)，寫入 `shared/types.py`，要寫範例 | A, B, C |
| 上午 | 確認 LLM 輸出 JSON 格式，A 提供 10 筆手寫範例 | A → B |
| 下午 | 各自撰寫 mock 實作（見工作分配表），push 到各自 branch | A, B, C |
| 下午 | 建立 Git repo、wandb project、共用 config | A |

### Day 2-3 (Tue-Wed)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 實作 `LLMPolicy` (載入 Qwen3.5-9B, generate, prompt template) | `llm_policy/policy.py`, `prompts.py` | 能在 single GPU 上 generate 一批關卡文字 |
| A | 撰寫 zero-shot baseline 腳本 | `baselines/run_baseline.py` | 用 mock env 跑通完整流程 |
| B | 實作 MiniGrid level parser + validator | `game_env/parser.py` | 能解析 A 提供的 10 筆範例，正確處理 edge cases |
| B | 實作 environment wrapper + rollout runner | `game_env/environment.py` | 能在 MiniGrid 上用 random agent 跑 rollout |
| B | 開始訓練 MiniGrid agents (PPO/DQN) | `game_env/agents/` | 訓練腳本可執行，開始跑 |
| C | 實作 `RewardCalculator` (regret + breadth + playability) | `reward_eval/reward.py` | 用 mock data 通過 unit test |
| C | 實作 `compute_advantages_grpo()` | `reward_eval/reward.py` | 數學正確性 unit test 通過 |
| C | 實作基本 evaluation metrics (Playability Rate, Regret) | `reward_eval/metrics.py` | 用 mock data 通過 |

### Day 4-5 (Thu-Fri)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 用真實 `GameEnvironment` 替換 mock，跑 zero-shot baseline | baseline 結果 | 取得 playability rate + regret 數據 |
| A | 撰寫 few-shot (self-generated) baseline | 額外 baseline 結果 | 篩選 zero-shot 高品質輸出作為 few-shot examples |
| B | Agent 訓練完成（至少 training set 的 4 個 agent） | agent checkpoints | strong agent win rate > 80% on random levels |
| B | 整合測試：A 的真實 LLM 輸出 → parser → rollout | 測試報告 | parse success rate 統計 + 修 bug |
| C | 接收 baseline 結果，跑完整 evaluation | Phase 1 evaluation report | Playability Rate, Regret 分布圖 |
| C | 實作 Solution Diversity (JSD) | `reward_eval/metrics.py` | 計算正確 |

### Phase 1 Checkpoint ✓

**驗收標準**:
- [ ] Zero-shot baseline 的 Playability Rate, Held-out Regret, Solution Diversity 數據產出
- [ ] Few-shot baseline 同上
- [ ] Pipeline 端到端可跑通：prompt → LLM → parse → rollout → reward → log
- [ ] Agent pool 訓練完成（至少 training set 所需的 4 個 agent）
- [ ] 所有 module 的真實實作已替換 mock

**預期結果**: Zero-shot playability rate 預估 30-60%（很多輸出格式錯或不可通關），regret 接近 0（關卡沒有被優化過）。

---

## Phase 2 — Week 2：GRPO Training on MiniGrid

**目標**: 加入 GRPO 訓練，驗證 RL 微調能提升關卡品質。

### Day 6-7 (Mon-Tue)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 實作 GRPO update（或整合 TRL GRPOTrainer） | `llm_policy/grpo.py` | 用 mock reward 可以跑通一個 training step |
| A | 實作 `train.py` 主迴圈，串接三個模組 | `train.py` | 端到端 1 iteration 跑通（真實模組） |
| B | 完成 held-out agent 訓練 | held-out checkpoints | 3 個 held-out agents ready |
| B | 優化 rollout 速度（多進程） | 更新 `environment.py` | batch_evaluate 延遲降低 ≥2x |
| C | 實作 Controllability metric (Cohen's d) | `reward_eval/metrics.py` | 用 mock data 通過 |
| C | 實作 `EvaluationSuite.evaluate()` 完整流程 | `reward_eval/evaluation.py` | 一鍵跑所有 metrics |

### Day 8-9 (Wed-Thu)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 開始正式 GRPO 訓練 (目標 ~200-500 iterations) | training logs + checkpoints | reward curve 有上升趨勢 |
| A | 監控訓練，調參（lr, kl_coeff, temperature, group_size） | wandb logs | 訓練穩定不 crash |
| B | 支援 A 的 debug（parser edge cases、環境問題） | bug fixes | parse success rate 穩定 |
| B | 開始研究 MiniHack des-file 格式（Phase 3 預備） | 格式文件 | 了解 des-file syntax |
| C | 用訓練中間 checkpoint 做 evaluation，觀察趨勢 | 中間報告 | 確認 metrics 隨訓練改善 |
| C | 設計 reward weight ablation 實驗 | ablation config | 準備 2-3 組不同 weight |

### Day 10 (Fri)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 訓練收尾，儲存 best checkpoint | final checkpoint | 完成指定 iteration 數 |
| B | 確認 pipeline 穩定性統計 | 穩定性報告 | parse failure rate < 10% |
| C | Phase 2 完整 evaluation (所有 5 項 metrics) | Phase 2 evaluation report | 與 baseline 的對比表格 + 圖 |
| 全員 | Phase 2 review meeting | — | 確認結果、決定 Phase 3 策略 |

### Phase 2 Checkpoint ✓

**驗收標準**:
- [ ] GRPO 訓練完成，reward curve 呈上升趨勢
- [ ] 訓練後 Playability Rate 顯著高於 zero-shot baseline
- [ ] Held-out Regret 顯著高於 baseline（關卡有技巧深度）
- [ ] Solution Diversity 不低於 baseline（策略廣度未被犧牲）
- [ ] Controllability 初步結果
- [ ] Ablation: 不同 reward weight 的比較

**預期結果**: Playability Rate 提升至 70-90%，Regret 顯著高於 baseline，證明 GRPO 有效。

---

## Phase 3 — Week 3：Scale to MiniHack + Final Report

**目標**: 驗證方法在更複雜環境上的可擴展性，完成所有實驗與報告。

### Day 11-12 (Mon-Tue)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | 設計 MiniHack 的 prompt template（des-file 格式） | `llm_policy/prompts.py` 更新 | MiniHack prompt ready |
| A | 確認 Qwen3.5-9B 在 MiniHack 的 zero-shot 可行性 | zero-shot MiniHack baseline | parse success > 10% |
| B | 實作 MiniHack des-file parser | `game_env/minihack/parser.py` | 能解析合法 des-file |
| B | 實作 MiniHack environment wrapper | `game_env/minihack/environment.py` | rollout 可執行 |
| B | 訓練 MiniHack agent pool | MiniHack agent checkpoints | strong win rate > 50% |
| C | 開始設計 Human Study 問卷與介面 | `evaluation/human_study/` | Likert scale 問卷 draft |
| C | 將 evaluation suite 適配 MiniHack | 更新 `evaluation.py` | 支援雙環境 |

### Day 13-14 (Wed-Thu)

| 成員 | 工作 | 產出 | 完成標準 |
|------|------|------|---------|
| A | MiniHack GRPO 訓練 | training logs | reward 有改善趨勢 |
| A | 同時跑 reward ablation 補充實驗（如需） | ablation results | — |
| B | 支援 MiniHack pipeline debug | bug fixes | 穩定執行 |
| B | 實作 PCGRL baseline（如時間允許） | PCGRL 結果 | — |
| C | MiniHack evaluation | 數據 | 所有 metrics 產出 |
| C | 跨環境對比分析 (MiniGrid vs MiniHack) | 對比報告 | — |

### Day 15 (Fri) — 收尾日

| 成員 | 工作 | 產出 |
|------|------|------|
| A | 整理所有訓練 log、checkpoint | 完整實驗記錄 |
| B | 整理環境程式碼、agent checkpoint | 可復現的 codebase |
| C | 彙整 final report：所有 metrics、圖表、分析 | 完整實驗報告 |
| 全員 | Final review + 報告撰寫分工 | 最終報告 |

### Phase 3 Checkpoint ✓

**驗收標準**:
- [ ] MiniHack pipeline 端到端可跑
- [ ] MiniHack 上的 GRPO 訓練結果（即使改善幅度小也可）
- [ ] MiniGrid vs MiniHack 跨環境對比
- [ ] Human Study 至少有 preliminary 結果（或設計完成）
- [ ] 完整實驗報告

---

## 風險與應對

| 風險 | 可能性 | 影響 | 應對方案 |
|------|--------|------|---------|
| LLM 生成的 MiniGrid 關卡 parse success rate 太低 | 中 | 訓練無法開始 | 簡化關卡格式、加入格式修復 heuristic、降低 grid size |
| Agent 訓練需時太長 | 低 | 延遲 pipeline | 用較少訓練步數的 agent 先行、降低 rollout 次數 |
| GRPO 訓練不穩定 / reward 不上升 | 中 | Phase 2 結論不明確 | 調降 lr、增大 group_size、調整 reward weight、嘗試 PPO |
| MiniHack des-file 格式太複雜導致 LLM 無法生成 | 高 | Phase 3 效果差 | 限制 des-file subset、用更簡單的 MiniHack 任務、視為 negative result 報告 |
| GPU 資源不足 | 低 | 訓練速度慢 | 減少 batch size、用 gradient accumulation、LoRA 降 rank |

---

## 里程碑總結

```
Week 1                    Week 2                    Week 3
├─ Day 1: API 共識        ├─ Day 6: GRPO 實作       ├─ Day 11: MiniHack 適配
├─ Day 2-3: 各自開發      ├─ Day 7: train.py 完成   ├─ Day 12: MiniHack agents
├─ Day 4: Integration     ├─ Day 8-9: 正式訓練      ├─ Day 13-14: MiniHack 訓練
├─ Day 5: Baseline 結果   ├─ Day 10: Phase 2 結果   ├─ Day 15: Final report
│                         │                         │
▼ Checkpoint 1            ▼ Checkpoint 2            ▼ Checkpoint 3
  Pipeline 跑通              GRPO 有效                 可擴展性驗證
  Baseline 數據              vs Baseline 改善           完整報告
```

---

## GPU 資源分配建議

假設有 4-8 張 A100/H100：

| 用途 | GPU 數 | 時間段 |
|------|--------|--------|
| Agent 訓練 (B) | 1-2 | Week 1 Day 2-5 |
| LLM 推理 baseline (A) | 1 | Week 1 Day 4-5 |
| GRPO 訓練 (A) | 2-4 | Week 2 全週 |
| Rollout 環境 (B) | 1 | Week 2-3 持續 |
| MiniHack agent 訓練 (B) | 1-2 | Week 3 Day 11-12 |
| MiniHack GRPO 訓練 (A) | 2-4 | Week 3 Day 13-14 |

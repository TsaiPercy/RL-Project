# Project Plan — RLVR Game Level Generation

## 三週每日工作計畫

**起始日期**: 2026-05-04 (Mon)
**結束日期**: 2026-05-22 (Fri)
**硬體**: 單張 NVIDIA RTX 4090 (24 GB VRAM)

---

## 總覽

| 階段 | 週次 | 目標 | 驗證內容 |
|------|------|------|---------|
| Phase 1 | Week 1 (Day 1-5) | Pipeline 建設 + Sanity Check | 各模組實作完成、mock 整合跑通、Exp 0 Sanity Check |
| Phase 2 | Week 2 (Day 6-10) | 端到端整合 + GRPO 訓練 + Baseline | 真實整合、GRPO 訓練完成、Baseline 收集完成 |
| Phase 3 | Week 3 (Day 11-15) | MiniHack + Ablation + 報告 | MiniHack 實驗、消融分析、完整報告 |

---

## Week 1 — Phase 1：模組開發 + Sanity Check

### Day 1 (Mon 5/4) — 共同啟動 + 基礎設施

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | I-3 | 撰寫 `config/default.yaml` | 包含所有超參數，可被程式讀取 |
| A | I-5 | 建立 wandb project `rlvr-level-gen` | wandb project 可存取 |
| A | I-6 | 撰寫 config loading utility | YAML → Python dict，有基本 test |
| B | I-1 | 建立目錄結構 scaffold | 所有資料夾 + `__init__.py` 就位 |
| B | I-2 | 撰寫 `requirements.txt` / `environment.yml` | 可一鍵安裝所有依賴 |
| B | D-1 | 調查 BabyAI 可用 pretrained agent checkpoint | 產出 strong/weak agent 選擇方案 |
| C | I-4 | 更新 `shared/types.py` 共用 dataclass | 所有 dataclass 定義完成 + 範例 |
| **全員** | — | Sync meeting：確認 dataclass 定義、API 簽名、LLM 輸出格式 | 三人對 API 有共識 |

### Day 2 (Tue 5/5) — 核心模組開發 (上)

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | MA-2 | 實作 `llm_policy/prompts.py` — Prompt template | Prompt 包含遊戲規則 + 格式規範 |
| A | D-4 | 撰寫 LLM 輸出範例（≥10 筆） | 範例涵蓋各種物件組合，交付給成員 B |
| A | MA-1 | 開始實作 `llm_policy/policy.py` — LLMPolicy class | QLoRA 4-bit 載入成功 |
| B | D-2 | 下載/準備 BabyAI agent checkpoints | Checkpoints 放入 `checkpoints/agents/` |
| B | MB-1 | 實作 parser — ASCII grid 解析（W/. → wall/floor） | 可解析合法 ASCII grid |
| B | MB-2 | 實作 parser — JSON objects 解析 + schema 驗證 | 可解析 A 提供的 10 筆範例 |
| C | MC-1 | 實作 `RewardCalculator`（regret + playability） | 用 mock data 計算正確 |
| C | MC-2 | 實作 `compute_advantages_grpo()` | z-score normalization 數學正確 |

### Day 3 (Wed 5/6) — 核心模組開發 (下)

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | MA-1 | 完成 LLMPolicy class（generate + get_ref_log_probs） | 可在 4090 上 generate 一批文字 |
| A | MA-4 | 撰寫 LLMPolicy mock | API 簽名正確，回傳隨機值 |
| B | MB-3 | 實作 parser — 語義驗證（連通性、物件不重疊、座標 0-12） | 所有驗證規則實作完成 |
| B | MB-7 | 實作 `wrappers.py` — MiniGrid wrapper | 確保 BabyAI 7×7 obs 相容 |
| C | MC-3 | 實作 `metrics.py` — Playability Rate, Parse Success Rate, Regret | 各指標可獨立計算 |
| C | MC-6 | 撰寫 RewardCalculator mock | API 簽名正確，回傳隨機值 |

### Day 4 (Thu 5/7) — Mock + 環境驗證

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | S-2 | 設計 MiniGrid prompt template（最終版） | 結合 Day 2-3 的開發經驗微調 |
| A | S-1 | 測試 LLMPolicy.generate()（QLoRA 4-bit 載入 + 生成） | 在 4090 上完整跑通 |
| B | D-3 | 驗證 BabyAI agent 在隨機 MiniGrid 關卡上的表現 | Strong win rate > 80% |
| B | MB-8 | 撰寫 GameEnvironment mock | API 簽名正確，回傳隨機值 |
| B | MB-4 | 開始實作 GameEnvironment class（構建 MiniGrid 15×15） | 可從 ParseResult 構建環境 |
| C | MC-4 | 開始實作 EvaluationSuite（quick + full 模式） | 框架建立，quick 模式可用 |
| **全員** | INT-3 | 用三人的 mock 模組跑通 train.py 一個 iteration | **同步點**：資料流驗證通過 |

### Day 5 (Fri 5/8) — Sanity Check

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | S-3 | 執行 Experiment 0: Sanity Check（100+ 次生成） | Parse rate 統計產出，目標 > 10% |
| A | S-4 | 若 parse rate < 10%：嘗試純 JSON 或換模型 | 有應對方案 |
| B | MB-4 | 完成 GameEnvironment class | 完整環境構建 + 基本測試 |
| B | MB-5 | 開始實作 run_rollouts()（SubprocVecEnv） | 單一環境 rollout 可執行 |
| C | MC-4 | 完成 EvaluationSuite（full 模式） | Quick + Full 模式均可用 |
| C | MC-5 | 開始實作 visualization.py | Reward curve 基本圖表 |

#### Week 1 Checkpoint

- [ ] 三個模組核心功能實作完成（MA-1~2, MB-1~3, MC-1~3）
- [ ] 三人 mock 均已完成（MA-4, MB-8, MC-6）
- [ ] INT-3：mock 整合跑通一個 iteration
- [ ] S-3：Sanity Check 完成，parse rate > 10%
- [ ] BabyAI agent 驗證完成（D-3）

---

## Week 2 — Phase 2：整合 + GRPO 訓練 + Baseline

### Day 6 (Mon 5/11) — GRPO 實作 + 環境完善

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | MA-3 | 開始實作 `grpo.py` — GRPO update | 基本框架 + loss 計算邏輯 |
| B | MB-5 | 完成 run_rollouts()（SubprocVecEnv + BabyAI agent） | 多進程 rollout 正確執行 |
| B | MB-6 | 實作 batch_evaluate()（整合 parse + rollout） | 批次評估 pipeline 跑通 |
| C | INT-2 | 實作 `evaluate.py` — 獨立評估腳本 | Quick + Full 模式可執行 |
| C | B-1 | 實作 `baselines/run_baseline.py` — Zero-shot baseline | 腳本結構完成 |

### Day 7 (Tue 5/12) — GRPO 完成 + 整合

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | MA-3 | 完成 GRPO update 實作 | 用 mock reward 跑通一個 training step |
| A | INT-1 | 開始實作 `train.py` — 主訓練迴圈 | 串接 A + B + C 的框架 |
| B | — | 支援整合：修復 parser edge cases、環境 bug | 配合成員 A 的整合測試 |
| C | B-2 | 執行 Zero-shot baseline，收集 100 個關卡指標 | Playability / Regret / Parse rate 數據產出 |

### Day 8 (Wed 5/13) — 端到端整合

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | INT-1 | 完成 `train.py` | 端到端 1 iteration 可跑通 |
| B | — | 支援整合 debug + 優化 rollout 速度 | batch_evaluate 延遲可接受 |
| C | B-3 | 從 zero-shot 輸出篩選 parse-valid 關卡作為 few-shot examples | 篩選出高品質範例 |
| C | B-4 | 執行 Few-shot baseline，收集指標 | 與 zero-shot 對比數據 |
| **全員** | INT-4 | 替換 mock 為真實模組，端到端跑通 | **同步點**：全真實模組 pipeline 運作 |

### Day 9 (Thu 5/14) — 正式訓練啟動

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | M-1 | 開始 GRPO 訓練（~200-500 iterations） | 訓練啟動，reward curve 開始記錄 |
| A | M-2 | 監控訓練：reward curve, parse rate, playability | wandb dashboard 正常顯示 |
| B | — | 支援訓練中的環境 bug + parser edge case 修復 | 訓練過程穩定 |
| C | M-3 | 訓練中期做 quick eval（training agents, 100 levels） | 中間 checkpoint 評估結果 |
| C | MC-5 | 完成 visualization.py | Reward curve, regret histogram, baseline 對比圖 |

### Day 10 (Fri 5/15) — 訓練收尾 + Phase 2 總結

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | M-1 | 繼續 / 完成 GRPO 訓練 | 完成指定 iteration 數 |
| A | M-4 | 訓練收尾，儲存 best checkpoint | Best checkpoint 存入 `checkpoints/` |
| B | — | 確認 pipeline 穩定性，整理環境程式碼 | Parse failure rate 統計 |
| C | M-5 | Full eval（held-out agents, 100 levels），與 baseline 對比 | 完整 Phase 2 evaluation report |
| **全員** | — | Phase 2 Review Meeting：確認結果、決定 Phase 3 策略 | 對 Phase 3 方向有共識 |

#### Week 2 Checkpoint

- [ ] MA-3：GRPO update 實作完成
- [ ] INT-1：`train.py` 主迴圈完成
- [ ] INT-4：真實模組端到端整合跑通
- [ ] B-1~B-4：Zero-shot + Few-shot baseline 結果產出
- [ ] M-1~M-5：GRPO 訓練完成 + full eval 結果
- [ ] GRPO 訓練後 Playability Rate 與 Regret 高於 baseline

---

## Week 3 — Phase 3：MiniHack + Ablation + Report

### Day 11 (Mon 5/18) — MiniHack 準備 + Ablation 設計

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | M-2 | 若訓練未完成，繼續監控並完成訓練 | 最終 checkpoint 確認 |
| A | P3-3 | 決定 Phase 3 是否使用固定 prompt | 與成員 B 討論後決策 |
| B | P3-1 | 研究 MiniHack des-file 格式 | 產出格式文件 + 可行性評估 |
| B | P3-2 | 決定 MiniHack grid size / 關卡規格 | 確定環境配置 |
| C | A-1 | 設計 2-3 組 reward weight ablation 配置 | Ablation config 檔案準備完成 |
| C | A-3 | Controllability 實驗：設計帶控制指令的 prompt | 與成員 A 協作 prompt 設計 |

### Day 12 (Tue 5/19) — MiniHack 開發 + Ablation 執行

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | P3-7 | 設計 MiniHack prompt template | MiniHack prompt 可用 |
| B | P3-4 | 實作 `game_env/minihack/parser.py` | 可解析 MiniHack des-file |
| B | P3-5 | 開始實作 `game_env/minihack/environment.py` | 環境框架建立 |
| C | A-2 | 執行 reward weight ablation 實驗 | 2-3 組實驗結果產出 |
| C | A-4 | 執行 Controllability 實驗，計算 Cohen's d | Cohen's d 結果產出 |

### Day 13 (Wed 5/20) — MiniHack 整合

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | — | MiniHack zero-shot sanity check | MiniHack parse rate 評估 |
| B | P3-5 | 完成 MiniHack environment wrapper | Rollout 可在 MiniHack 上執行 |
| B | P3-6 | 準備 MiniHack agent pool | Agent 可在 MiniHack 上遊玩 |
| C | R-1 | 開始彙整所有實驗結果（MiniGrid baseline vs GRPO vs ablation） | 結果表格初版 |

### Day 14 (Thu 5/21) — MiniHack 訓練 + 報告

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | P3-8 | 執行 MiniHack GRPO 訓練（或 zero-shot/few-shot） | MiniHack 實驗結果產出 |
| B | P3-8 | 支援 MiniHack pipeline debug | 訓練過程穩定 |
| C | P3-8 | MiniHack evaluation | MiniHack 指標數據產出 |
| C | R-2 | 生成最終視覺化圖表 | 所有圖表完成 |

### Day 15 (Fri 5/22) — 收尾 + 報告撰寫

| 成員 | TODO # | 任務 | 完成標準 |
|------|--------|------|---------|
| A | R-4 | 撰寫報告 — Module A 部分（LLM Policy + GRPO 訓練） | 方法與結果完整描述 |
| B | R-4 | 撰寫報告 — Module B 部分（環境 + Agent + Parser） | 方法與結果完整描述 |
| C | R-1 | 完成最終結果彙整 | 所有數據確認 |
| C | R-4 | 撰寫報告 — Module C 部分（Evaluation + 分析） | 方法與結果完整描述 |
| **全員** | R-3 | MiniGrid vs MiniHack 跨環境對比分析 | 對比結論明確 |
| **全員** | R-4 | 報告整合 + 最終 review | 完整實驗報告交付 |

#### Week 3 Checkpoint

- [ ] P3-4, P3-5：MiniHack parser + environment 實作完成
- [ ] P3-8：MiniHack 實驗結果產出（即使改善幅度小）
- [ ] A-1~A-4：Ablation + Controllability 結果完成
- [ ] R-1~R-4：完整實驗報告交付

---

## 每人工作量總覽

### 成員 A（LLM Policy & Training）

| 週次 | 任務 | TODO # |
|------|------|--------|
| W1 | Config + wandb + config loader | I-3, I-5, I-6 |
| W1 | LLM 輸出範例 + prompt template | D-4, S-2, MA-2 |
| W1 | LLMPolicy 實作 + mock | MA-1, MA-4 |
| W1 | Sanity Check | S-1, S-3, S-4 |
| W2 | GRPO update + train.py | MA-3, INT-1 |
| W2 | GRPO 訓練 + 監控 + checkpoint | M-1, M-2, M-4 |
| W3 | MiniHack prompt + 訓練 | P3-3, P3-7, P3-8 |
| W3 | 報告撰寫 | R-4 |

### 成員 B（Game Environment & Agent）

| 週次 | 任務 | TODO # |
|------|------|--------|
| W1 | 目錄結構 + 依賴管理 | I-1, I-2 |
| W1 | Agent 準備 + 驗證 | D-1, D-2, D-3 |
| W1 | Parser 三階段（ASCII + JSON + 語義） | MB-1, MB-2, MB-3 |
| W1 | Wrapper + mock + GameEnvironment | MB-7, MB-8, MB-4 |
| W2 | Rollout + batch_evaluate | MB-5, MB-6 |
| W2 | 支援整合 debug | — |
| W3 | MiniHack 全套（研究 + parser + env + agent） | P3-1, P3-2, P3-4, P3-5, P3-6, P3-8 |
| W3 | 報告撰寫 | R-4 |

### 成員 C（Reward, Evaluation & Baselines）

| 週次 | 任務 | TODO # |
|------|------|--------|
| W1 | Shared types + Reward + GRPO advantages | I-4, MC-1, MC-2 |
| W1 | Metrics + mock + EvaluationSuite | MC-3, MC-6, MC-4 |
| W2 | evaluate.py + Baseline (zero-shot + few-shot) | INT-2, B-1, B-2, B-3, B-4 |
| W2 | Quick eval + visualization + full eval | M-3, MC-5, M-5 |
| W3 | Ablation + controllability | A-1, A-2, A-3, A-4 |
| W3 | 結果彙整 + 視覺化 + 報告 | R-1, R-2, R-3, R-4 |

---

## 關鍵同步點（Sync Points）

| 日期 | 事件 | 前置條件 | 參與人 |
|------|------|---------|--------|
| Day 1 | API 共識 Meeting | — | 全員 |
| Day 4 | INT-3：Mock 整合測試 | MA-4, MB-8, MC-6 三人 mock 就緒 | 全員 |
| Day 8 | INT-4：真實模組整合 | MA-1~3, MB-4~6, MC-1~2 完成 | 全員 |
| Day 10 | Phase 2 Review Meeting | M-1 訓練完成, M-5 eval 完成 | 全員 |
| Day 15 | 報告整合 + Final Review | 所有實驗完成 | 全員 |

## 關鍵依賴鏈

```
Day 1: I-1, I-2, I-4 (基礎設施)
  │
  ▼
Day 2-3: MA-1, MB-1~3, MC-1~2 (各模組核心)
  │
  ▼
Day 3-4: MA-4, MB-8, MC-6 (Mock 完成)
  │
  ▼
Day 4: INT-3 ← 三人 Mock 同步點
  │
  ▼
Day 5: S-3 ← MA-1 + MB-1~2 (Sanity Check)
  │
  ▼
Day 6-7: MA-3, MB-5~6 (GRPO + Rollout)
  │
  ▼
Day 7-8: INT-1 → INT-4 ← 真實整合同步點
  │
  ▼
Day 8: B-1~4 (Baselines，需 INT-4 完成)
  │
  ▼
Day 9-10: M-1 → M-4 → M-5 (訓練 → Eval)
  │
  ▼
Day 12-14: P3-4~8 (MiniHack pipeline)
  │
  ▼
Day 15: R-1~4 (報告)
```

---

## 風險與應對

| 風險 | 觸發時間 | 影響 | 應對方案 |
|------|---------|------|---------|
| S-3 parse rate < 10% | Day 5 | 訓練無法開始 | S-4：簡化格式 / 換模型（Day 5 當天處理） |
| BabyAI agent 表現不穩定（D-3 失敗） | Day 4 | Regret signal 不可靠 | 調整 agent 選擇或微調 agent |
| INT-4 整合失敗 | Day 8 | GRPO 訓練延遲 | 全員集中 debug，Day 9 前必須解決 |
| GRPO reward 不上升 | Day 9-10 | Phase 2 結論不明確 | 調降 lr、增大 group_size、調整 reward weight |
| MiniHack des-file 太複雜 | Day 11 | Phase 3 效果差 | 限制 des-file subset，視為 negative result |
| GPU VRAM 不足 | Day 2 / Day 9 | 推理或訓練 crash | 降低 LoRA rank、CPU offloading、減小 batch |

---

## GPU 資源分配（單張 4090）

| 時段 | 用途 | 占用者 | 估計 VRAM |
|------|------|--------|----------|
| Day 2-4 | LLM 推理測試（QLoRA 4-bit） | 成員 A | ~6-8 GB |
| Day 5 | Sanity Check 大量生成 | 成員 A | ~6-8 GB |
| Day 9-10 | GRPO 訓練 | 成員 A | ~15-18 GB |
| Day 12 | Ablation 實驗（if GPU available） | 成員 C | ~15-18 GB |
| Day 14 | MiniHack GRPO 訓練 | 成員 A | ~15-18 GB |

> Agent rollout (成員 B) 在 CPU 上用 SubprocVecEnv 執行，不佔 GPU。
> 成員 C 的 evaluation 需要 LLM 生成關卡時，與成員 A 協調 GPU 使用。

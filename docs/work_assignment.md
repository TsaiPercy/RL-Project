# Work Assignment — RLVR Game Level Generation

## 分工原則

依照三模組架構，每位成員各負責一個核心模組，共用任務（Infrastructure、Integration、Phase 3、Report）依相關性分配，確保工作量平衡。

| 成員 | 核心模組 | 職責概述 |
|------|---------|---------|
| **成員 A** | Module A: LLM Policy (`llm_policy/`) | LLM 載入、推理、GRPO 訓練、prompt 設計、訓練主迴圈 |
| **成員 B** | Module B: Game Environment (`game_env/`) | Level parser、環境構建、Agent pool 準備、rollout 執行 |
| **成員 C** | Module C: Reward & Evaluation (`reward_eval/`) | Reward 計算、評估指標、視覺化、baseline 實驗 |

---

## 成員 A — LLM Policy & Training

### Infrastructure
| # | Task | Notes |
|---|------|-------|
| I-3 | 撰寫 `config/default.yaml`（所有超參數） | 包含 LLM、GRPO、reward 等所有配置 |
| I-5 | 建立 wandb project `rlvr-level-gen` | 訓練監控基礎設施 |
| I-6 | 撰寫 config loading utility | YAML → Python dict/dataclass |

### Data Preparation
| # | Task | Notes |
|---|------|-------|
| D-4 | 撰寫 LLM 輸出範例（ASCII grid + JSON，≥10 筆） | 供 Parser 開發使用，需熟悉輸出格式 |

### Sanity Checks
| # | Task | Notes |
|---|------|-------|
| S-1 | 實作 LLMPolicy.generate()（QLoRA 4-bit 載入 + 生成） | 驗證 4090 可跑 |
| S-2 | 設計 MiniGrid prompt template（ASCII grid + JSON 格式） | 核心 prompt 設計 |
| S-3 | 執行 Experiment 0: Sanity Check（100+ 次生成，統計 parse rate） | 需成員 B 的 parser 配合 |
| S-4 | 若 parse rate < 10%：嘗試純 JSON 格式或換模型 | 風險緩解 |

### Module A: LLM Policy
| # | Task | Notes |
|---|------|-------|
| MA-1 | 實作 `llm_policy/policy.py` — LLMPolicy class | QLoRA 載入、generate、get_ref_log_probs |
| MA-2 | 實作 `llm_policy/prompts.py` — Prompt template 管理 | ASCII grid + JSON 格式規範 |
| MA-3 | 實作 `llm_policy/grpo.py` — GRPO update | 或整合 TRL GRPOTrainer |
| MA-4 | 撰寫 LLMPolicy mock | 供成員 B、C 獨立測試 |

### Integration
| # | Task | Notes |
|---|------|-------|
| INT-1 | 實作 `train.py` — 主訓練迴圈（串接 A + B + C） | Module A 作為 orchestrator |

### Main Method
| # | Task | Notes |
|---|------|-------|
| M-1 | 開始 GRPO 訓練（~200-500 iterations） | batch=4, group=16 |
| M-2 | 監控訓練：reward curve, parse rate, playability over time | 與成員 C 協作 |
| M-4 | 訓練收尾，儲存 best checkpoint | |

### Phase 3
| # | Task | Notes |
|---|------|-------|
| P3-3 | 決定 Phase 3 是否使用固定 prompt | 與成員 B 討論 |
| P3-7 | 設計 MiniHack prompt template | |

**成員 A 任務總計：20 項（15 獨立 + 5 協作）**

---

## 成員 B — Game Environment & Agent Pool

### Infrastructure
| # | Task | Notes |
|---|------|-------|
| I-1 | 建立目錄結構 scaffold | 按 SPEC §16 |
| I-2 | 撰寫 `requirements.txt` / `environment.yml` | 環境依賴管理 |

### Data Preparation
| # | Task | Notes |
|---|------|-------|
| D-1 | 調查 BabyAI 可用的 pretrained agent checkpoint | 確定 strong/weak 選擇 |
| D-2 | 下載/準備 BabyAI agent checkpoints | 放入 `checkpoints/agents/` |
| D-3 | 驗證 BabyAI agent 在隨機 MiniGrid 13×13 關卡上的表現 | Strong win rate 應 > 80% |

### Module B: Game Environment
| # | Task | Notes |
|---|------|-------|
| MB-1 | 實作 `game_env/parser.py` — ASCII grid 解析（W/. → wall/floor） | |
| MB-2 | 實作 `game_env/parser.py` — JSON objects 解析 + schema 驗證 | |
| MB-3 | 實作 `game_env/parser.py` — 語義驗證（連通性、起終點、物件不重疊、座標 0-12） | |
| MB-4 | 實作 `game_env/environment.py` — GameEnvironment class | 構建 MiniGrid 15×15 環境 |
| MB-5 | 實作 `game_env/environment.py` — run_rollouts() | SubprocVecEnv + BabyAI agent |
| MB-6 | 實作 `game_env/environment.py` — batch_evaluate() | 整合 parse + rollout |
| MB-7 | 實作 `game_env/wrappers.py` — MiniGrid wrapper | 確保 BabyAI agent 7×7 obs 相容 |
| MB-8 | 撰寫 GameEnvironment mock | 供成員 A、C 獨立測試 |

### Phase 3
| # | Task | Notes |
|---|------|-------|
| P3-1 | 研究 MiniHack des-file 格式 | |
| P3-2 | 決定 MiniHack grid size / 關卡規格 | |
| P3-4 | 實作 `game_env/minihack/parser.py` | |
| P3-5 | 實作 `game_env/minihack/environment.py` | |
| P3-6 | 準備 MiniHack agent pool | |

**成員 B 任務總計：20 項（18 獨立 + 2 協作）**

---

## 成員 C — Reward, Evaluation & Baselines

### Infrastructure
| # | Task | Notes |
|---|------|-------|
| I-4 | 更新 `shared/types.py` 共用 dataclass | 加入 eval mode 等新欄位 |

### Module C: Reward & Evaluation
| # | Task | Notes |
|---|------|-------|
| MC-1 | 實作 `reward_eval/reward.py` — RewardCalculator | regret + playability, regret clamp ≥ 0 |
| MC-2 | 實作 `reward_eval/reward.py` — compute_advantages_grpo() | group z-score normalization |
| MC-3 | 實作 `reward_eval/metrics.py` — Playability Rate, Parse Success Rate, Regret 計算 | |
| MC-4 | 實作 `reward_eval/evaluation.py` — EvaluationSuite | quick + full 模式 |
| MC-5 | 實作 `reward_eval/visualization.py` — reward curve, regret histogram, baseline 對比 | |
| MC-6 | 撰寫 RewardCalculator mock | 供成員 A、B 獨立測試 |

### Integration
| # | Task | Notes |
|---|------|-------|
| INT-2 | 實作 `evaluate.py` — 獨立評估腳本（quick + full 模式） | |

### Baseline Setup
| # | Task | Notes |
|---|------|-------|
| B-1 | 實作 `baselines/run_baseline.py` — Zero-shot baseline | |
| B-2 | 執行 Zero-shot baseline，收集指標 | 100 個關卡 |
| B-3 | 從 zero-shot 輸出篩選 parse-valid 關卡作為 few-shot examples | |
| B-4 | 執行 Few-shot baseline，收集指標 | |

### Main Method
| # | Task | Notes |
|---|------|-------|
| M-3 | 訓練中期做 quick eval（training agents, 100 levels） | |
| M-5 | Full eval（held-out agents, 100 levels），與 baseline 對比 | |

### Ablations
| # | Task | Notes |
|---|------|-------|
| A-1 | 設計 2-3 組 reward weight ablation 配置 | |
| A-2 | 執行 reward weight ablation 實驗 | |
| A-3 | Controllability 實驗：設計帶控制指令的 prompt | 與成員 A 協作 prompt 設計 |
| A-4 | 執行 Controllability 實驗，計算 Cohen's d | |

### Report & Wrap-up
| # | Task | Notes |
|---|------|-------|
| R-1 | 彙整所有實驗結果（baseline vs GRPO vs ablation） | |
| R-2 | 生成最終視覺化圖表 | |

**成員 C 任務總計：20 項（19 獨立 + 1 協作）**

---

## 共同協作任務

以下任務需三人共同參與：

| # | Task | 主導 | 備註 |
|---|------|------|------|
| INT-3 | 用 mock 模組跑通 train.py 一個 iteration | 成員 A 主導 | 三人 mock 需備妥 |
| INT-4 | 替換 mock 為真實模組，端到端跑通 | 成員 A 主導 | 全員 debug |
| P3-8 | 執行 MiniHack GRPO 訓練 + evaluation | 成員 A 主導 | 全員環境適配 |
| R-3 | MiniGrid vs MiniHack 跨環境對比分析 | 成員 C 主導 | 全員提供數據 |
| R-4 | 撰寫實驗報告 | 全員 | 各寫負責模組部分 |

---

## 時程與依賴關係

```
Week 1 (Phase 1): Pipeline 建設 + Sanity Check
├── 成員 A: MA-1~4, S-1, S-2, I-3, I-5, I-6, D-4
├── 成員 B: I-1, I-2, D-1~3, MB-1~3, MB-7, MB-8
├── 成員 C: I-4, MC-1~3, MC-6
└── 里程碑: 各模組 mock 完成，parser + LLM generate 可獨立運作
     → S-3 (Exp 0 Sanity Check) 可執行

Week 2 (Phase 2): 整合 + GRPO 訓練
├── 成員 A: INT-1, M-1, M-2, M-4
├── 成員 B: MB-4~6（環境 + rollout 完整實作）
├── 成員 C: MC-4~5, INT-2, B-1~4, M-3, M-5
├── 共同: INT-3, INT-4
└── 里程碑: 端到端 pipeline 跑通，GRPO 訓練完成，baseline 收集完成

Week 3 (Phase 3): MiniHack + Ablation + Report
├── 成員 A: P3-3, P3-7
├── 成員 B: P3-1, P3-2, P3-4~6
├── 成員 C: A-1~4, R-1, R-2
├── 共同: P3-8, R-3, R-4
└── 里程碑: MiniHack 實驗完成，所有結果彙整，報告撰寫完成
```

## 關鍵依賴（Blocking Dependencies）

| 下游任務 | 依賴（上游） | 備註 |
|---------|------------|------|
| S-3 (Sanity Check) | MA-1 (成員A) + MB-1~2 (成員B) | Parser + LLM 都需完成 |
| MB-5 (rollout) | D-1~2 (成員B 自行) + MB-7 (成員B 自行) | Agent + wrapper 需備妥 |
| INT-3 (mock 整合) | MA-4 + MB-8 + MC-6 (各成員 mock) | **三人同步點** |
| INT-4 (真實整合) | MA-1~3 + MB-4~6 + MC-1~2 | **三人同步點** |
| B-1~2 (baseline) | INT-4 完成 | 需完整 pipeline |
| M-1 (GRPO 訓練) | INT-4 完成 | 需完整 pipeline |
| M-5 (full eval) | M-1 完成 + MC-4 完成 | 訓練後才能 full eval |

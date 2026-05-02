# TODO — RLVR Game Level Generation

## Pending Decisions（待決策）

| # | 議題 | 影響範圍 | 備註 |
|---|------|---------|------|
| PD-1 | 是否在 reward 中加入 strategy breadth（action entropy） | Reward 計算, SPEC §5.2 | 目前僅用 regret；若加入需更新 RewardCalculator |
| PD-2 | Phase 3 MiniHack 是否使用固定 prompt | Prompt 設計, SPEC §5.1 | Phase 1-2 已確定用固定 prompt |
| PD-3 | MiniHack 的 grid size / 關卡規格 | 環境配置, SPEC §5.5 | 需等 Phase 3 開始時決定 |
| PD-4 | Mode collapse 防治：是否需要額外 diversity 機制 | Reward / Training, SPEC §5.2 | 先觀察 group_size=16 的效果 |
| PD-5 | Solution Diversity (JSD) 的具體計算方式 | Evaluation, SPEC §8 | [deferred] 先不實作 |
| PD-6 | Human Study 的規模與設計 | Evaluation, SPEC §8 | [deferred] 先不實作 |
| PD-7 | BabyAI strong/weak agent 的具體選擇 | Agent Pool, SPEC §5.4 | [obsolete: 改為自行訓練 curriculum agent，強弱差異由門檻與環境種類決定] |

---

## Infrastructure

| # | Task | Status | Notes |
|---|------|--------|-------|
| I-1 | 建立目錄結構 scaffold | ☐ | Per SPEC §16 |
| I-2 | 撰寫 `requirements.txt` / `environment.yml` | ☐ | Per SPEC §15，含 bitsandbytes for QLoRA |
| I-3 | 撰寫 `config/default.yaml`（所有超參數） | ☐ | Per SPEC §11 pseudo config |
| I-4 | 更新 `shared/types.py` 共用 dataclass | ☐ | 加入 eval mode 等新欄位；Per SPEC §11 |
| I-5 | 建立 wandb project `rlvr-level-gen` | ☐ | Per SPEC §11 |
| I-6 | 撰寫 config loading utility | ☐ | YAML → Python dict/dataclass |

## Data Preparation

| # | Task | Status | Notes |
|---|------|--------|-------|
| D-1 | 調查 BabyAI 可用的 pretrained agent checkpoint | ☐ | [obsolete: 改為自行訓練 curriculum agent] |
| D-2 | 下載/準備 BabyAI agent checkpoints | ☐ | [obsolete: 改為自行訓練 curriculum agent] |
| D-3 | 驗證 curriculum-trained agent 在隨機 MiniGrid 13×13 關卡上的表現 [updated] | ☐ | Strong win rate 應 > 80%；驗證 curriculum 訓練後 agent 在 LLM 生成關卡上的泛化能力 |
| D-4 | 撰寫 LLM 輸出範例（ASCII grid + JSON，≥10 筆） | ☐ | 供 Parser 開發使用；Per SPEC §3 |

## Agent Curriculum Training (Phase 1-2) [added]

| # | Task | Status | Notes |
|---|------|--------|-------|
| AT-1 | 實作 `agent_training/train_curriculum.py` — BabyAI 環境 curriculum 訓練腳本 | ☐ | Per SPEC §5.4；含 curriculum 晉級邏輯、success rate 門檻判斷 |
| AT-2 | 設計並配置 curriculum 成功率門檻（config 中各關卡門檻設定） | ☐ | Per SPEC §11.4；strong: 高門檻全 curriculum，weak: 低門檻部分 curriculum |
| AT-3 | 驗證各 BabyAI 環境支援 room_size=15 參數 [updated] | ☐ | BabyAI-GoTo-v0, BabyAI-GoToOpen-v0, BabyAI-GoToObjMaze-v0, BabyAI-GoToObjMazeOpen-v0, BabyAI-GoToObjMazeS4R2-v0, BabyAI-GoToObjMazeS4-v0, BabyAI-GoToObjMazeS5-v0, BabyAI-GoToObjMazeS6-v0, BabyAI-GoToObjMazeS7-v0；注意 S4-S7 系列可能已有內建 grid size，需確認 room_size 參數是否生效 |
| AT-4 | 訓練 strong_0：完整 curriculum（8 關），高門檻（~85-90%） | ☐ | 儲存至 `checkpoints/agents/strong_0.zip` |
| AT-5 | 訓練 weak_0：部分 curriculum（前 3 關），低門檻（~50-60%） | ☐ | 儲存至 `checkpoints/agents/weak_0.zip` |
| AT-6 | 訓練 held-out agents（strong_held_0, weak_held_0），使用不同 seed | ☐ | 儲存至 `checkpoints/agents/` |
| AT-7 | 實作 `agent_training/evaluate_agent.py` — 評估 agent 各環境 success rate | ☐ | 驗證強弱差異明顯 |

## Phase 0: Toy Case — Pipeline Smoke Test [added]

| # | Task | Status | Notes |
|---|------|--------|-------|
| TC-1 | 實作 `toy_case/train_agent.py` — SB3 PPO 訓練腳本（DoorKeyEnv(room_size=15)） [updated] | ☑ | Per SPEC §5.4.1；含 strong/weak 兩種訓練配置 |
| TC-2 | 訓練 toy_strong_0：PPO on `MiniGrid-DoorKey-15x15-v0`，~1M steps，目標 success rate > 90% [updated] | ☐ | 儲存至 `checkpoints/agents/toy_strong_0.zip` |
| TC-3 | 訓練 toy_weak_0：PPO on `MiniGrid-DoorKey-15x15-v0`，~50K steps，目標 success rate ~30-60% [updated] | ☐ | 儲存至 `checkpoints/agents/toy_weak_0.zip` |
| TC-4 | 驗證 strong/weak agent 表現差異（在 DoorKeyEnv 上跑 100 episodes，確認 success rate 差距明顯） | ☐ | Strong > 90%, Weak 30-60% |
| TC-5 | LLM zero-shot 生成 10-20 張地圖（使用 Experiment 0 的 LLM + prompt） | ☐ | 依賴 S-1, S-2 或可用簡化版 LLM 呼叫 |
| TC-6 | 實作 `toy_case/run_toy_pipeline.py` — 端到端 pipeline smoke test | ☑ | parse → env → rollout(toy agents) → reward；Per SPEC §12 Exp T |
| TC-7 | 執行 toy pipeline，驗證各 shared type dataclass 欄位正確 | ☑ | Mock 模式全部通過；真實模式需等 Module B |

## Sanity Checks

| # | Task | Status | Notes |
|---|------|--------|-------|
| S-1 | 實作 LLMPolicy.generate()（QLoRA 4-bit 載入 + 生成） | ☑ | Per SPEC §11 Module A；驗證 4090 可跑 |
| S-2 | 設計 MiniGrid prompt template（ASCII grid + JSON 格式） | ☑ | Per SPEC §3, §11 Module A |
| S-3 | 執行 Experiment 0: Sanity Check（100+ 次生成，統計 parse rate） | ☐ | 腳本已實作 (`toy_case/sanity_check.py`)；需 GPU 執行 |
| S-4 | 若 parse rate < 10%：嘗試純 JSON 格式或換模型 | ☐ | Per SPEC §7 風險緩解 |

## Module Implementation — Module A: LLM Policy

| # | Task | Status | Notes |
|---|------|--------|-------|
| MA-1 | 實作 `llm_policy/policy.py` — LLMPolicy class（QLoRA 載入、generate、get_ref_log_probs） | ☑ | Per SPEC §11 Module A |
| MA-2 | 實作 `llm_policy/prompts.py` — Prompt template 管理 | ☑ | Per SPEC §3, §11 |
| MA-3 | 實作 `llm_policy/grpo.py` — GRPO update（或整合 TRL GRPOTrainer） | ☐ | GRPO update 已整合在 policy.py 內；grpo.py 可後續獨立抽出 |
| MA-4 | 撰寫 LLMPolicy mock（API 簽名正確 + 隨機值） | ☑ | Per SPEC §11；供 B, C 獨立測試 |

## Module Implementation — Module B: Game Environment

| # | Task | Status | Notes |
|---|------|--------|-------|
| MB-1 | 實作 `game_env/parser.py` — ASCII grid 解析（W/. → wall/floor） | ☐ | Per SPEC §3, §11 Module B |
| MB-2 | 實作 `game_env/parser.py` — JSON objects 解析 + schema 驗證 | ☐ | Per SPEC §11 Module B |
| MB-3 | 實作 `game_env/parser.py` — 語義驗證（連通性、起終點、物件不重疊、座標 0-12） | ☐ | Per SPEC §11 Module B |
| MB-4 | 實作 `game_env/environment.py` — GameEnvironment class（構建 MiniGrid 15×15 環境） | ☐ | Per SPEC §11 Module B |
| MB-5 | 實作 `game_env/environment.py` — run_rollouts()（SubprocVecEnv + BabyAI agent） | ☐ | Per SPEC §11 Module B |
| MB-6 | 實作 `game_env/environment.py` — batch_evaluate()（整合 parse + rollout） | ☐ | Per SPEC §11 Module B |
| MB-7 | 實作 `game_env/wrappers.py` — MiniGrid wrapper（確保 BabyAI agent 7×7 obs 相容） | ☐ | Per SPEC §11 Module B |
| MB-8 | 撰寫 GameEnvironment mock（API 簽名正確 + 隨機值） | ☐ | Per SPEC §11；供 A, C 獨立測試 |

## Module Implementation — Module C: Reward & Evaluation

| # | Task | Status | Notes |
|---|------|--------|-------|
| MC-1 | 實作 `reward_eval/reward.py` — RewardCalculator（regret + playability, regret clamp ≥ 0） | ☐ | Per SPEC §5.2, §11 Module C |
| MC-2 | 實作 `reward_eval/reward.py` — compute_advantages_grpo()（group z-score normalization） | ☐ | Per SPEC §5.3, §11 |
| MC-3 | 實作 `reward_eval/metrics.py` — Playability Rate, Parse Success Rate, Regret 計算 | ☐ | Per SPEC §8 |
| MC-4 | 實作 `reward_eval/evaluation.py` — EvaluationSuite（quick + full 模式） | ☐ | Per SPEC §8, §11 Module C |
| MC-5 | 實作 `reward_eval/visualization.py` — reward curve, regret histogram, baseline 對比 | ☐ | Per SPEC §13 |
| MC-6 | 撰寫 RewardCalculator mock（API 簽名正確 + 隨機值） | ☐ | Per SPEC §11；供 A, B 獨立測試 |

## Integration

| # | Task | Status | Notes |
|---|------|--------|-------|
| INT-1 | 實作 `train.py` — 主訓練迴圈（串接 A + B + C） | ☐ | Per SPEC §11, §12 Exp 3 |
| INT-2 | 實作 `evaluate.py` — 獨立評估腳本（quick + full 模式） | ☐ | Per SPEC §8, §12 |
| INT-3 | 用 mock 模組跑通 train.py 一個 iteration | ☐ | 驗證模組間資料流 |
| INT-4 | 替換 mock 為真實模組，端到端跑通 | ☐ | Per SPEC §12 |

## Baseline Setup

| # | Task | Status | Notes |
|---|------|--------|-------|
| B-1 | 實作 `baselines/run_baseline.py` — Zero-shot baseline | ☐ | Per SPEC §10 Exp 1 |
| B-2 | 執行 Zero-shot baseline，收集 100 個關卡的 parse rate / playability / regret | ☐ | Per SPEC §10 Exp 1 |
| B-3 | 從 zero-shot 輸出篩選 parse-valid 關卡作為 few-shot examples | ☐ | Per SPEC §10 Exp 2 |
| B-4 | 執行 Few-shot baseline，收集同上指標 | ☐ | Per SPEC §10 Exp 2 |

## Main Method

| # | Task | Status | Notes |
|---|------|--------|-------|
| M-1 | 開始 GRPO 訓練（~200-500 iterations） | ☐ | Per SPEC §10 Exp 3；batch=4, group=16 |
| M-2 | 監控訓練：reward curve, parse rate, playability over time | ☐ | Per SPEC §13 |
| M-3 | 訓練中期做 quick eval（training agents, 100 levels） | ☐ | Per SPEC §8 |
| M-4 | 訓練收尾，儲存 best checkpoint | ☐ | Per SPEC §11 |
| M-5 | Full eval（held-out agents, 100 levels），與 baseline 對比 | ☐ | Per SPEC §8, §10 Exp 3 |

## Ablations

| # | Task | Status | Notes |
|---|------|--------|-------|
| A-1 | 設計 2-3 組 reward weight ablation 配置 | ☐ | Per SPEC §10 Exp 4 |
| A-2 | 執行 reward weight ablation 實驗 | ☐ | Per SPEC §10 Exp 4 |
| A-3 | Controllability 實驗：設計帶控制指令的 prompt | ☐ | Per SPEC §10 Exp 5；控制物件數量 |
| A-4 | 執行 Controllability 實驗，計算 Cohen's d | ☐ | Per SPEC §10 Exp 5 |

## Phase 3: MiniHack Extension

| # | Task | Status | Notes |
|---|------|--------|-------|
| P3-1 | 研究 MiniHack des-file 格式 | ☐ | Per SPEC §10 Exp 6 |
| P3-2 | 決定 MiniHack grid size / 關卡規格 | ☐ | Per PD-3 |
| P3-3 | 決定 Phase 3 是否使用固定 prompt | ☐ | Per PD-2 |
| P3-4 | 實作 `game_env/minihack/parser.py` | ☐ | Per SPEC §16 |
| P3-5 | 實作 `game_env/minihack/environment.py` | ☐ | Per SPEC §16 |
| P3-6 | 準備 MiniHack agent pool | ☐ | Per SPEC §5.4 |
| P3-7 | 設計 MiniHack prompt template | ☐ | Per SPEC §11 Module A |
| P3-8 | 執行 MiniHack GRPO 訓練 + evaluation | ☐ | Per SPEC §10 Exp 6 |

## Report & Wrap-up

| # | Task | Status | Notes |
|---|------|--------|-------|
| R-1 | 彙整所有實驗結果（baseline vs GRPO vs ablation） | ☐ | Per SPEC §8 |
| R-2 | 生成最終視覺化圖表 | ☐ | Per SPEC §13 |
| R-3 | MiniGrid vs MiniHack 跨環境對比分析 | ☐ | Per SPEC §10 Exp 6 |
| R-4 | 撰寫實驗報告 | ☐ | |

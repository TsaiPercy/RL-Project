# Implementation Record

## Session: 2026-05-01 — Member A (YouZhe): Sanity Checks + Phase 0
reviewed ✅
### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| S-1 | 實作 LLMPolicy.generate()（QLoRA 4-bit 載入 + 生成） | ☑ |
| S-2 | 設計 MiniGrid prompt template（ASCII grid + JSON 格式） | ☑ |
| MA-1 | 實作 `llm_policy/policy.py` — LLMPolicy class（QLoRA 載入、generate、get_ref_log_probs） | ☑ |
| MA-2 | 實作 `llm_policy/prompts.py` — Prompt template 管理 | ☑ |
| MA-4 | 撰寫 LLMPolicy mock（API 簽名正確 + 隨機值） | ☑ |
| TC-1 | 實作 `toy_case/train_agent.py` — SB3 PPO 訓練腳本（DoorKeyEnv(room_size=15)） | ☑ |
| TC-6 | 實作 `toy_case/run_toy_pipeline.py` — 端到端 pipeline smoke test | ☑ |
| TC-7 | 執行 toy pipeline，驗證各 shared type dataclass 欄位正確 | ☑ |
| S-3 (partial) | 實作 `toy_case/sanity_check.py` — Experiment 0 sanity check 腳本（含 simple parser） | ☑ (腳本完成，需 GPU 執行) |
| I-3 (partial) | 更新 `config/default.yaml` — 新增 toy_case section | ☑ |

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| MA-1 | GRPO update 邏輯整合在 `policy.py` 內，未獨立至 `grpo.py` | GRPO update 與 LLMPolicy 的 optimizer 和 model 緊密耦合，拆出反而增加跨模組傳遞成本。後續若需獨立可抽出。 |
| MA-1 | `get_ref_log_probs()` 需額外傳入 `prompt_ids` | Spec 原本只列 `token_ids`，但計算 log probs 需要完整 context（prompt + generated），故新增 `prompt_ids` 參數 |
| S-2 | Prompt 使用單一固定範例 + 詳細格式規範 | Per SPEC §3 Phase 1-2 使用固定 prompt；範例取自 SPEC §3 的官方範例 |
| TC-1 | 使用 `CnnPolicy` + `ImgObsWrapper` | SB3 PPO 需要 image observation；ImgObsWrapper 將 MiniGrid 的 dict obs 轉為 image tensor |
| TC-1 | 使用 `SubprocVecEnv` (4 envs) 訓練 + `DummyVecEnv` (1 env) 評估 | 平行化加速訓練，評估用單環境即可 |
| TC-6 | Mock 模式與 Real 模式分離 | Mock 模式不需 GPU/模型即可驗證 pipeline 資料流；Real 模式需等 Module B 完成才能跑完整 pipeline |
| S-3 | 實作 `simple_parse()` 輕量 parser | S-3 只需統計 parse rate，不需 Module B 的完整語義驗證；使用 brace-matching 取代 regex 處理嵌套 JSON |

### Spec Amendments
| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| `get_ref_log_probs` 簽名 | `get_ref_log_probs(token_ids: Tensor) → Tensor` | `get_ref_log_probs(token_ids: Tensor, prompt_ids: Tensor) → Tensor` | 計算 log probs 需要 prompt context 作為 conditioning input |

### Concerns
- **S-3 需 GPU 執行**: `toy_case/sanity_check.py` 腳本已完成，但需要 GPU + 足夠 VRAM 載入 Qwen3.5-9B（4-bit ~5GB）才能執行。命令：`python -m toy_case.sanity_check --num-levels 100`
- **TC-2/TC-3 需 GPU 執行**: Agent 訓練腳本已完成，但訓練需要時間（strong ~30min, weak ~5min）。命令：`python -m toy_case.train_agent --agent both`
- **S-4 備案**: 若 S-3 的 parse rate < 10%，`sanity_check.py` 會輸出警告，需手動檢查 `results/sanity_check/` 下的原始輸出來決定是否退回純 JSON 格式
- **Module B 依賴**: `run_toy_pipeline.py` 的 Real 模式中，完整 pipeline（parse → env → rollout）需等成員 B 完成 `game_env/parser.py` 和 `game_env/environment.py`
- **bitsandbytes / stable-baselines3 未安裝**: 當前環境缺少 `bitsandbytes`, `peft`, `stable-baselines3`, `minigrid` 等套件，需先執行 `pip install` 或建立 conda 環境（Per TODO I-2）

---

## Session: 2026-05-01~02 — Member C (Percy): Module C — Reward & Evaluation

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| MC-1 | 實作 `reward_eval/reward.py` — RewardCalculator（regret + playability, regret clamp ≥ 0） | ☑ |
| MC-2 | 實作 `reward_eval/reward.py` — compute_advantages_grpo()（group z-score normalization） | ☑ |
| MC-3 | 實作 `reward_eval/metrics.py` — Playability Rate, Parse Success Rate, Regret 計算 | ☑ |
| MC-4 | 實作 `reward_eval/evaluation.py` — EvaluationSuite（quick + full 模式） | ☑ |
| MC-5 | 實作 `reward_eval/visualization.py` — reward curve, regret histogram, baseline 對比 | ☑ |
| MC-6 | 撰寫 RewardCalculator mock（API 簽名正確 + 隨機值） | ☑ |
| INT-2 | 實作 `evaluate.py` — 獨立評估腳本（quick + full 模式） | ☑ |
| B-1 | 實作 `baselines/run_baseline.py` — Zero-shot baseline | ☑ |
| I-4 (partial) | 更新 `shared/types.py` — 新增 `MetricsResult` dataclass | ☑ |

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| MC-4 | EvaluationSuite 持有 GameEnvironment 參考（Option A） | Per SPEC §11，內部呼叫 `batch_evaluate()` 取得 rollout，簡化 evaluate() API |
| MC-4 | `evaluate()` 根據 `mode` 從 config 解析 `agent_ids`，傳入 `batch_evaluate(agent_ids=...)` | Per SPEC §8：quick 用 training agents、full 用 held-out agents；需與成員 B 確認 `batch_evaluate()` 是否接受 `agent_ids` 參數 |
| I-4 | `MetricsResult` 放在 `shared/types.py`（非 local type） | 跨 metrics.py / evaluate.py / EvaluationSuite 使用，屬 shared contract |
| B-1 | 移除 `BaselineConfig`，改用 `argparse` | `run_baseline.py` 為獨立腳本，argparse 直接處理設定更簡潔，避免過度設計 |
| MC-5 | `visualization.py` 使用 `Agg` backend | 非互動式後端，適合 server/CI 環境，避免 GUI 依賴 |

### Spec Amendments
| Change | Before | After | Rationale |
|--------|--------|-------|-----------|
| `batch_evaluate()` 簽名 | `batch_evaluate(llm_outputs, num_rollouts_per_agent)` | `batch_evaluate(llm_outputs, num_rollouts_per_agent, agent_ids=...)` | Per SPEC §8 mode 需選擇 agent pool；新增 `agent_ids` keyword arg |

### Concerns
- **`batch_evaluate()` 新增 `agent_ids` 參數**：需與成員 B 確認 `GameEnvironment.batch_evaluate()` 是否支援此參數。目前以 keyword argument 傳入，不影響已有呼叫方式。
- **環境名稱不一致**：成員 C 建立了 `RL_Final_Project` 環境，但團隊統一名稱為 `RL_project`。建議成員 C 重新建立或重命名環境。
- **TODO I-4 部分完成**：成員 C 新增了 `MetricsResult`，但成員 A 亦有獨立更新 `shared/types.py`（新增 `prompt_ids` 等）。目前兩邊的修改不衝突。

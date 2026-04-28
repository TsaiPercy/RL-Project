# 工作分配表

## RLVR Game Level Generation — 三人分工

---

## 分工原則

1. **模組獨立性**：每人負責一個完整模組，對外僅暴露 SDD 中定義的 API
2. **不需了解對方實作細節**：只需知道「輸入什麼、輸出什麼」
3. **Integration 責任歸屬**：成員 A 負責 `train.py` 串接，但各模組的 API 必須符合 SDD 規範
4. **Mock-first 開發**：Phase 1 第一天各自撰寫符合 API 的 mock，讓其他人可以獨立開發測試

---

## 成員 A — LLM Policy & Training Loop

### 核心職責
用 GRPO 微調 Qwen3.5-9B 生成遊戲關卡

### 具體工作項目

| 項目 | 說明 | 產出 |
|------|------|------|
| **A1. Prompt 設計** | 為 MiniGrid / MiniHack 設計 context prompt，定義 LLM 輸出格式（JSON） | `llm_policy/prompts.py` |
| **A2. LLMPolicy class** | 實作模型載入、LoRA 配置、generate()、get_ref_log_probs() | `llm_policy/policy.py` |
| **A3. GRPO 實作** | 實作 GRPO update (group-relative advantage + KL penalty)，或整合 TRL 的 GRPOTrainer | `llm_policy/grpo.py` |
| **A4. train.py** | 主訓練迴圈：串接 LLMPolicy + GameEnvironment + RewardCalculator | `train.py` |
| **A5. Baseline 腳本** | Zero-shot / Few-shot baseline 評估腳本 | `baselines/run_baseline.py` |
| **A6. Config 管理** | 統一 config 格式（YAML），方便實驗切換 | `config/` |
| **A7. Checkpoint & Logging** | wandb 整合、checkpoint 儲存/載入 | 整合於各檔案 |

### 需要從其他人取得的東西
- **從 B**: `GameEnvironment` 的可用實例（或 mock），呼叫 `batch_evaluate(list[str]) -> list[RolloutResult | None]`
- **從 C**: `RewardCalculator` 的可用實例（或 mock），呼叫 `compute_batch_rewards()` 和 `compute_advantages_grpo()`

### 需要提供給其他人的東西
- **給 B**: 確定的 LLM 輸出 JSON 格式（Day 1 提供範例）
- **給 C**: 訓練過程中的 checkpoint，供評估使用

### Mock 實作（Day 1 完成）

```python
# mock_game_env.py — 成員 A 用於獨立測試 training loop
class MockGameEnvironment:
    def batch_evaluate(self, llm_outputs):
        results = []
        for text in llm_outputs:
            # 隨機生成假的 rollout 結果
            mock_result = RolloutResult(
                level_config={"width": 8, "height": 8},
                trajectories={
                    "strong_0": [Trajectory(total_return=random.uniform(0.5, 1.0), success=True, ...)],
                    "weak_0": [Trajectory(total_return=random.uniform(0.0, 0.3), success=False, ...)],
                }
            )
            results.append(mock_result if random.random() > 0.2 else None)
        return results
```

---

## 成員 B — Game Environment & Agent Pool

### 核心職責
管理遊戲環境、解析 LLM 輸出為可執行關卡、訓練並管理 game agent pool

### 具體工作項目

| 項目 | 說明 | 產出 |
|------|------|------|
| **B1. Level Parser** | 解析 LLM 文字輸出 → JSON → MiniGrid/MiniHack 關卡物件。含 schema 驗證、語義驗證（連通性、起終點存在） | `game_env/parser.py` |
| **B2. Environment Wrapper** | 將 parsed level config 注入 MiniGrid/MiniHack，封裝為統一的 Gym 環境 | `game_env/environment.py` |
| **B3. Agent Training** | 用 Stable-Baselines3 在隨機生成的關卡上訓練 PPO/DQN agents，產出 weak/strong checkpoints | `game_env/agents/train_agent.py` |
| **B4. Rollout Runner** | 給定關卡 + agent，執行 rollout 並收集 Trajectory 資料。支援多進程平行化 | `game_env/environment.py` |
| **B5. batch_evaluate()** | 整合 parser + rollout，提供批次介面 | `game_env/environment.py` |
| **B6. MiniHack 擴展** (Phase 3) | 實作 MiniHack 的 des-file parser 與環境 wrapper | `game_env/minihack/` |

### 需要從其他人取得的東西
- **從 A**: LLM 輸出的 JSON 格式規範 + 至少 10 筆範例輸出（Day 1 取得）

### 需要提供給其他人的東西
- **給 A**: `GameEnvironment` class（符合 SDD API）
- **給 C**: `RolloutResult` 與 `Trajectory` dataclass 定義（Day 1 共同確定）
- **給 C**: 訓練好的 agent checkpoints（Phase 1 Week 1 Day 3 前完成）

### Mock 實作（Day 1 完成）

```python
# mock_reward.py — 成員 B 用於獨立測試 rollout 流程
class MockRewardCalculator:
    def compute_reward(self, rollout):
        if rollout is None:
            return RewardOutput(total_reward=-1.0, regret=0, strategy_breadth=0, playable=False)
        return RewardOutput(total_reward=random.uniform(0, 2), ...)
```

### Agent 訓練規格

| Agent | 演算法 | 訓練步數 | 環境 | 備註 |
|-------|--------|---------|------|------|
| strong PPO (×2 seeds) | PPO | 10M | 隨機 MiniGrid 8×8 | seed 0, 1 |
| weak PPO (×2 seeds) | PPO | 500K | 隨機 MiniGrid 8×8 | seed 0, 1 |
| held-out strong PPO | PPO | 10M | 同上 | seed 42, eval only |
| held-out strong DQN | DQN | 10M | 同上 | eval only |
| held-out weak PPO | PPO | 500K | 同上 | seed 42, eval only |

---

## 成員 C — Reward & Evaluation

### 核心職責
設計 reward function、實作完整 evaluation suite、產出實驗報告

### 具體工作項目

| 項目 | 說明 | 產出 |
|------|------|------|
| **C1. RewardCalculator** | 實作 regret index + strategy breadth + playability bonus 的 reward 計算 | `reward_eval/reward.py` |
| **C2. GRPO Advantage** | 實作 group-relative advantage normalization | `reward_eval/reward.py` |
| **C3. Evaluation Metrics** | 實作 5 項評估指標：Playability Rate, Held-out Regret, Solution Diversity (JSD), Controllability (Cohen's d), Human Eval 框架 | `reward_eval/metrics.py` |
| **C4. EvaluationSuite** | 整合所有 metrics，提供統一的 `evaluate()` 介面 | `reward_eval/evaluation.py` |
| **C5. 視覺化** | Reward curve, metric 分布圖, 關卡視覺化 | `reward_eval/visualization.py` |
| **C6. evaluate.py** | 獨立評估腳本：載入 checkpoint → 生成關卡 → 計算所有 metrics → 匯出報告 | `evaluate.py` |
| **C7. Reward Ablation** | 實驗不同 reward weight 組合的效果 | 與 A 協調 |
| **C8. Human Study 設計** | 設計問卷、關卡取樣策略、Likert scale 評分介面 | `evaluation/human_study/` |

### 需要從其他人取得的東西
- **從 B**: `RolloutResult` 與 `Trajectory` dataclass 定義（Day 1）
- **從 B**: 可用的 `GameEnvironment` 實例（或 mock），用於跑 held-out evaluation
- **從 A**: 各階段的 model checkpoint

### 需要提供給其他人的東西
- **給 A**: `RewardCalculator` class 與 `compute_advantages_grpo()` 方法
- **給全員**: 實驗報告與視覺化結果

### Mock 實作（Day 1 完成）

```python
# mock_rollout_data.py — 成員 C 用於獨立開發 reward 和 evaluation
def generate_mock_rollouts(n=100):
    """產生模擬的 RolloutResult 資料，含 edge cases"""
    results = []
    for _ in range(n):
        case = random.choice(["valid_easy", "valid_hard", "unparseable", "unsolvable"])
        if case == "unparseable":
            results.append(None)
        elif case == "unsolvable":
            results.append(RolloutResult(
                trajectories={"strong_0": [Trajectory(success=False, total_return=0.0, ...)], ...}
            ))
        else:
            # ... generate realistic trajectories
            results.append(rollout)
    return results
```

---

## 共同責任

| 項目 | 負責人 | 說明 |
|------|--------|------|
| Dataclass 定義 | 全員 Day 1 | `RolloutResult`, `Trajectory`, `RewardOutput`, `GenerationOutput` 等共用資料結構，放在 `shared/types.py` |
| Config schema | A (主), 全員 review | YAML config 格式 |
| Git workflow | 全員 | main branch + 個人 feature branch，PR review 後 merge |
| wandb project | A 建立 | 統一的 experiment tracking |
| README | 全員 | 環境安裝、執行方式 |

---

## 溝通協議

1. **Day 1 sync meeting**: 確認所有 dataclass 定義、JSON 格式、API signature
2. **每日 async update**: 在共用文件/群組中報告進度、blocker
3. **Phase 結束 sync**: 每個 Phase 結束時做 integration test + review
4. **API 變更**: 任何 API 簽名變更須通知相關成員並更新 SDD

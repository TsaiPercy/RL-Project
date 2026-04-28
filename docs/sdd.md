# Software Design Document (SDD)

## RLVR Game Level Generation — RL Fine-tuning LLM for Game Level Design

---

## 1. System Overview

本系統以 GRPO 微調  Qwen3.5-Coder-9B，使其生成遊戲關卡（MiniGrid / MiniHack）。訓練信號完全來自 game agent 的遊玩表現，不使用任何人類關卡資料。

### 1.1 系統架構圖

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │  LLM     │───▶│  Level   │───▶│  Game Environment │  │
│  │  Policy  │    │  Parser  │    │  Runner            │  │
│  │ (Qwen 9B)│    │          │    │  (MiniGrid/       │  │
│  └────▲─────┘    └──────────┘    │   MiniHack)       │  │
│       │                          └────────┬──────────┘  │
│       │                                   │              │
│       │          ┌──────────┐    ┌────────▼──────────┐  │
│       │          │  Reward  │◀───│  Agent Pool       │  │
│       └──────────│Calculator│    │  (weak / strong)  │  │
│    GRPO update   └──────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Evaluation Suite │
              │ (metrics, logs)  │
              └──────────────────┘
```

---

## 2. Module Specifications

### Module 1: LLM Policy (`llm_policy/`)

**負責人**: 成員 A

**功能**: 管理 Qwen3.5-Coder-9B 的載入、推理、與 RL 更新。

**模型規格**:
- Base model: `Qwen/Qwen3.5-9B`
- Precision: BF16
- LoRA: rank=64, alpha=128, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Framework: Hugging Face Transformers + PEFT + TRL (或 OpenRLHF)

**對外 API**:

```python
class LLMPolicy:
    def __init__(self, model_name: str, lora_config: dict):
        """載入 base model 與 LoRA adapter"""

    def generate(self, prompts: list[str], 
                 temperature: float = 0.8,
                 max_new_tokens: int = 2048) -> GenerationOutput:
        """
        批次生成關卡。
        
        Args:
            prompts: context prompt 列表（含遊戲規則 + 關卡格式規範）
            temperature: 取樣溫度
            max_new_tokens: 最大生成 token 數

        Returns:
            GenerationOutput:
                texts: list[str]          # 生成的原始文字
                log_probs: Tensor         # shape (batch, seq_len), 每個 token 的 log prob
                token_ids: Tensor         # shape (batch, seq_len)
        """

    def get_ref_log_probs(self, token_ids: Tensor) -> Tensor:
        """用 frozen reference model 計算 log probs，供 KL penalty 使用"""

    def update(self, grpo_batch: GRPOBatch) -> dict:
        """
        執行一步 GRPO 參數更新。
        
        Args:
            grpo_batch: 包含以下欄位的 dataclass
                token_ids: Tensor
                log_probs: Tensor          # 生成時的 log probs
                ref_log_probs: Tensor      # reference model 的 log probs
                rewards: Tensor            # shape (batch,)
                advantages: Tensor         # GRPO group-normalized advantages
        
        Returns:
            dict: {"loss": float, "kl": float, "mean_reward": float}
        """

    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...
```

**Prompt 格式規範** (成員 A 定義，成員 B 需遵循):

```
# MiniGrid Level Generation

You are a game level designer. Generate a MiniGrid level in the following JSON format:

{format_spec}

Rules:
- Grid size: {width}x{height}
- Available objects: {object_list}
- The level must have exactly one start position and one goal position
- All positions must be within grid bounds

Generate a level:
```

**Output 格式** (成員 A 定義，成員 B 的 Parser 需解析):

```json
{
  "width": 8,
  "height": 8,
  "objects": [
    {"type": "wall", "x": 2, "y": 3},
    {"type": "key", "x": 1, "y": 5, "color": "yellow"},
    {"type": "door", "x": 4, "y": 4, "color": "yellow"},
    {"type": "goal", "x": 7, "y": 7}
  ],
  "agent_start": {"x": 0, "y": 0, "dir": 0}
}
```

---

### Module 2: Level Parser & Game Environment (`game_env/`)

**負責人**: 成員 B

**功能**: 解析 LLM 輸出為合法關卡，執行 agent rollout，回傳 trajectory 資料。

**對外 API**:

```python
@dataclass
class ParseResult:
    success: bool                # 是否成功解析為合法關卡
    level_config: dict | None    # 解析後的關卡配置（成功時）
    error_msg: str | None        # 錯誤訊息（失敗時）

@dataclass 
class Trajectory:
    states: list[np.ndarray]     # 每步的 observation
    actions: list[int]           # agent 選擇的 action
    rewards: list[float]         # 每步的 environment reward
    total_return: float          # 累計 return
    success: bool                # 是否通關
    length: int                  # episode 步數

@dataclass
class RolloutResult:
    level_config: dict
    trajectories: dict[str, list[Trajectory]]  
    # key = agent_id ("strong_0", "weak_0", ...), value = 該 agent 多次 rollout

class GameEnvironment:
    def __init__(self, game: str, agent_pool_path: str):
        """
        Args:
            game: "minigrid" 或 "minihack"
            agent_pool_path: 預訓練 agent checkpoint 目錄
        """

    def parse_level(self, llm_output: str) -> ParseResult:
        """
        解析 LLM 的原始文字輸出為關卡配置。
        負責：
        1. 從文字中提取 JSON（處理 markdown code block 等）
        2. Schema 驗證
        3. 語義驗證（起點終點存在、物件不重疊、地圖連通性等）
        """

    def run_rollouts(self, level_config: dict,
                     num_rollouts_per_agent: int = 5) -> RolloutResult:
        """
        在指定關卡上，用 agent pool 中所有 agent 各跑 num_rollouts_per_agent 次。
        
        Returns:
            RolloutResult: 包含所有 agent 的 trajectory 資料
        """

    def batch_evaluate(self, llm_outputs: list[str],
                       num_rollouts_per_agent: int = 5) -> list[RolloutResult | None]:
        """
        批次處理：解析 + rollout。解析失敗的關卡回傳 None。
        內部會做平行化處理以加速。
        """
```

**Agent Pool 規格**:

| Agent ID | 類型 | 訓練程度 | 用途 |
|----------|------|---------|------|
| `strong_0` | PPO, seed=0 | 10M steps | Training reward |
| `strong_1` | PPO, seed=1 | 10M steps | Training reward |
| `weak_0` | PPO, seed=0 | 500K steps | Training reward |
| `weak_1` | PPO, seed=1 | 500K steps | Training reward |
| `strong_held_0` | PPO, seed=42 | 10M steps | Evaluation only |
| `strong_held_1` | DQN, seed=0 | 10M steps | Evaluation only |
| `weak_held_0` | PPO, seed=42 | 500K steps | Evaluation only |

- Training agents（前 4 個）用於計算 training reward
- Held-out agents（後 3 個）僅用於 evaluation，避免 overfitting

**MiniGrid 關卡 JSON Schema**:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["width", "height", "objects", "agent_start"],
  "properties": {
    "width": {"type": "integer", "minimum": 5, "maximum": 16},
    "height": {"type": "integer", "minimum": 5, "maximum": 16},
    "objects": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["type", "x", "y"],
        "properties": {
          "type": {"enum": ["wall", "key", "door", "ball", "box", "goal", "lava"]},
          "x": {"type": "integer"},
          "y": {"type": "integer"},
          "color": {"enum": ["red", "green", "blue", "purple", "yellow", "grey"]}
        }
      }
    },
    "agent_start": {
      "type": "object",
      "required": ["x", "y", "dir"],
      "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "dir": {"type": "integer", "minimum": 0, "maximum": 3}
      }
    }
  }
}
```

**MiniHack 關卡格式**: des-file 格式，成員 B 在 Phase 3 另行定義 schema。

---

### Module 3: Reward Calculator & Evaluation (`reward_eval/`)

**負責人**: 成員 C

**功能**: 接收 rollout 結果，計算 training reward 與 evaluation metrics。

**對外 API**:

```python
@dataclass
class RewardOutput:
    total_reward: float        # 最終 scalar reward（用於 GRPO）
    regret: float              # 強弱 agent return 差異
    strategy_breadth: float    # 強 agent action entropy
    playable: bool             # 是否可通關
    breakdown: dict            # 各項子指標明細

class RewardCalculator:
    def __init__(self, config: RewardConfig):
        """
        Args:
            config: RewardConfig dataclass，包含：
                - regret_weight: float = 1.0
                - breadth_weight: float = 0.5
                - playability_bonus: float = 1.0
                - invalid_penalty: float = -1.0
        """

    def compute_reward(self, rollout: RolloutResult | None) -> RewardOutput:
        """
        計算單一關卡的 reward。
        
        Reward = playability_bonus * I(playable) 
               + regret_weight * regret_index
               + breadth_weight * strategy_breadth
        
        若 rollout is None（解析失敗），回傳 invalid_penalty。
        
        regret_index = mean(strong_returns) - mean(weak_returns)
        strategy_breadth = mean(action_entropy(strong_agent_trajectories))
        """

    def compute_batch_rewards(self, rollouts: list[RolloutResult | None]) -> list[RewardOutput]:
        """批次計算 rewards"""

    def compute_advantages_grpo(self, rewards: list[float], group_size: int) -> Tensor:
        """
        GRPO group-relative advantage 計算。
        
        將 rewards 按 group_size 分組，組內 z-score normalize。
        
        Args:
            rewards: 所有 sample 的 reward 列表
            group_size: 每個 prompt 的 sample 數量（同一個 prompt 生成的多個關卡）
        
        Returns:
            Tensor: normalized advantages
        """


class EvaluationSuite:
    def __init__(self, game_env: GameEnvironment):
        """使用 held-out agents 進行評估"""

    def evaluate(self, llm_outputs: list[str]) -> EvalReport:
        """
        全面評估一組生成關卡。
        
        Returns:
            EvalReport:
                playability_rate: float
                held_out_regret: dict (mean, median, std, >threshold %)
                solution_diversity: dict (mean JSD across trajectories)
                controllability: dict (per-dimension Cohen's d) [Phase 2+]
                raw_data: list[dict]  # 每個關卡的詳細指標
        """

    def export_report(self, report: EvalReport, path: str):
        """匯出評估報告為 JSON + 視覺化圖表"""
```

**Reward 計算細節**:

```
# Regret Index
regret = (1/|A_strong|) Σ V^a_strong(ℓ) - (1/|A_weak|) Σ V^a_weak(ℓ)

其中 V^a(ℓ) = (1/M) Σ_{m=1}^{M} total_return(trajectory_m)

# Strategy Breadth
action_counts_a = Counter(trajectory.actions)  # 對每個 strong agent
action_dist_a = normalize(action_counts_a)
entropy_a = -Σ p log p
strategy_breadth = mean(entropy_a for a in A_strong)

# 無效關卡處理
if rollout is None:  # 解析失敗
    reward = invalid_penalty  # default: -1.0
elif not any(t.success for a, ts in rollout.trajectories.items() for t in ts):
    reward = 0.0  # 可解析但不可通關
else:
    reward = playability_bonus + regret_weight * regret + breadth_weight * breadth
```

---

## 3. Module 間 Interface 總覽

```
成員 A (LLM Policy)
    │
    │ 呼叫 generate() 產生 list[str]
    │
    ▼
成員 B (Game Environment)
    │ 
    │ A 呼叫 batch_evaluate(llm_outputs) 
    │ → 回傳 list[RolloutResult | None]
    │
    ▼
成員 C (Reward Calculator)
    │
    │ A 呼叫 compute_batch_rewards(rollouts)
    │ → 回傳 list[RewardOutput]
    │
    │ A 呼叫 compute_advantages_grpo(rewards, group_size)
    │ → 回傳 advantages Tensor
    │
    ▼
成員 A (LLM Policy)
    │
    │ 呼叫 update(grpo_batch) 更新參數
    │
    └── 一個 training iteration 結束
```

**Integration 由成員 A 負責**，在 `train.py` 中串接各模組：

```python
# train.py (成員 A 撰寫)
from llm_policy import LLMPolicy
from game_env import GameEnvironment
from reward_eval import RewardCalculator

def train_step(policy, game_env, reward_calc, prompts, group_size=4):
    # 1. 每個 prompt 生成 group_size 個關卡
    all_prompts = [p for p in prompts for _ in range(group_size)]
    gen_output = policy.generate(all_prompts)
    
    # 2. 解析 + rollout (成員 B 的模組)
    rollouts = game_env.batch_evaluate(gen_output.texts)
    
    # 3. 計算 reward (成員 C 的模組)
    reward_outputs = reward_calc.compute_batch_rewards(rollouts)
    rewards = [r.total_reward for r in reward_outputs]
    advantages = reward_calc.compute_advantages_grpo(rewards, group_size)
    
    # 4. 計算 ref log probs (KL penalty)
    ref_log_probs = policy.get_ref_log_probs(gen_output.token_ids)
    
    # 5. GRPO update
    batch = GRPOBatch(
        token_ids=gen_output.token_ids,
        log_probs=gen_output.log_probs,
        ref_log_probs=ref_log_probs,
        rewards=torch.tensor(rewards),
        advantages=advantages,
    )
    metrics = policy.update(batch)
    return metrics
```

---

## 4. 資料格式與 Shared Config

**共用 Config (`config/`)**:

```yaml
# config/default.yaml
game: "minigrid"
grid_size: 8

# LLM
model_name: "Qwen/Qwen3.5-9B"
lora_rank: 64
lora_alpha: 128
max_new_tokens: 2048
temperature: 0.8

# GRPO
group_size: 4              # 每個 prompt 取樣幾個關卡
kl_coeff: 0.05
learning_rate: 1e-5
batch_size: 16              # prompts per batch
num_iterations: 500

# Reward
regret_weight: 1.0
breadth_weight: 0.5
playability_bonus: 1.0
invalid_penalty: -1.0

# Rollout
num_rollouts_per_agent: 5

# Agent
agent_pool_path: "checkpoints/agents/"
```

**Logging 規範** (所有人遵循):
- 使用 Python `logging` module
- 使用 Weights & Biases (`wandb`) 記錄 training metrics
- Checkpoint 存放於 `checkpoints/{experiment_name}/step_{n}/`

---

## 5. 技術依賴

| 依賴 | 版本 | 用途 | 負責人 |
|------|------|------|--------|
| PyTorch | ≥2.1 | 框架 | 全員 |
| Transformers | ≥4.40 | LLM 載入 | A |
| PEFT | ≥0.10 | LoRA | A |
| TRL | ≥0.8 | GRPO trainer（或自行實作） | A |
| vLLM | ≥0.4 | 加速推理 (optional) | A |
| MiniGrid | ≥2.3 | 遊戲環境 | B |
| MiniHack | ≥0.1.5 | 遊戲環境 (Phase 3) | B |
| Stable-Baselines3 | ≥2.0 | 訓練 game agent | B |
| NumPy / SciPy | latest | 數值計算 | C |
| wandb | latest | Experiment tracking | 全員 |

---

## 6. 目錄結構

```
project/
├── config/
│   ├── default.yaml
│   ├── minigrid.yaml
│   └── minihack.yaml
├── llm_policy/              # 成員 A
│   ├── __init__.py
│   ├── policy.py            # LLMPolicy class
│   ├── grpo.py              # GRPO update 實作
│   └── prompts.py           # Prompt template 管理
├── game_env/                # 成員 B
│   ├── __init__.py
│   ├── parser.py            # Level parser + validator
│   ├── environment.py       # GameEnvironment class
│   ├── agents/              # Agent training scripts
│   │   ├── train_agent.py
│   │   └── configs/
│   └── wrappers.py          # 環境 wrapper
├── reward_eval/             # 成員 C
│   ├── __init__.py
│   ├── reward.py            # RewardCalculator
│   ├── evaluation.py        # EvaluationSuite
│   ├── metrics.py           # 各項指標計算
│   └── visualization.py     # 圖表生成
├── train.py                 # 主訓練迴圈 (成員 A 整合)
├── evaluate.py              # 獨立評估腳本 (成員 C)
├── checkpoints/
│   └── agents/              # 預訓練 agent checkpoints
├── logs/
└── results/
```

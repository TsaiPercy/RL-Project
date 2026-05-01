# RLVR Game Level Generation — Specification

## 1. Overview

本專案使用 GRPO（Group Relative Policy Optimization）微調 Qwen3.5-Coder-9B，使其生成 MiniGrid 與 MiniHack 遊戲關卡。訓練信號完全來自 game agent pool 的遊玩表現（regret index），不使用任何人類設計的關卡資料。目標是讓 LLM 生成能區分強弱玩家技巧水準的關卡。

## 2. Problem

現有的 procedural content generation (PCG) 方法多依賴人類設計資料或手工規則。PCGRL 雖然用 RL 直接訓練 agent 生成關卡，但缺乏可解釋性且無法利用預訓練語言模型的隱含遊戲設計知識。

本專案的 gap：
- 不依賴人類關卡資料，純粹用 agent 遊玩表現作為 reward
- 利用 LLM 的預訓練知識與可解釋的文字輸出，取代不可解釋的 PCGRL agent
- 以 regret（強弱 agent 表現差異）衡量關卡的技巧深度（skill-differentiation）

## 3. Modality

**輸入**：
- 固定的 text prompt（Phase 1-2），包含遊戲規則、關卡格式規範、物件列表
- 格式：純文字，~200-500 tokens

**輸出**：
- LLM 生成的關卡描述，包含兩部分：
  1. **ASCII Grid**（13×13 可用空間）：每個 cell 一個字元，定義牆壁（`W`）與地板（`.`）的基本地形
  2. **JSON**：定義物件放置位置（key、door、goal、lava 等）與 agent 起始位置
- 格式：結構化文字（ASCII grid + JSON），~200-1000 tokens
- MiniGrid 環境實際大小為 15×15（含外牆），可用空間 13×13

**輸出格式範例**：
```
Grid:
.............
.............
..WWW........
....W........
.............
.............
.............
.............
.............
.............
.............
.............
.............

{
  "objects": [
    {"type": "key", "x": 1, "y": 5, "color": "yellow"},
    {"type": "door", "x": 4, "y": 4, "color": "yellow"},
    {"type": "goal", "x": 12, "y": 12}
  ],
  "agent_start": {"x": 0, "y": 0, "dir": 0}
}
```

**中間表示**：
- `ParseResult`：解析後的關卡配置（dict）
- MiniGrid/MiniHack 環境實例
- `Trajectory`：agent rollout 紀錄（states, actions, rewards）
- `RolloutResult`：多個 agent 在同一關卡上的所有 trajectory

## 4. Core Idea

LLM 在大規模預訓練中接觸過遊戲設計相關知識，具有隱含的關卡設計能力。GRPO 微調可以在不需要 value function 的情況下（因 reward 稀疏，每個完整關卡才有一個 reward），透過 group-relative advantage normalization 穩定地引導 LLM 生成有技巧深度的關卡。

核心 reward signal 是 **regret index**：強 agent 與弱 agent 在同一關卡上的表現差異。高 regret 代表關卡能有效區分不同技巧水準的玩家。

## 5. Method

### 5.1 總體流程

```
每個 training iteration:
1. LLM Policy 接收固定 prompt，生成 batch_size × group_size = 64 個關卡描述
2. Level Parser 解析 ASCII grid + JSON → 驗證 → 構建 MiniGrid 環境
3. Agent Pool（1 strong + 1 weak BabyAI agent）各跑 5 次 rollout
4. Reward Calculator 計算 regret-based reward
5. GRPO 計算 group-relative advantages，更新 LLM LoRA 參數
```

### 5.2 Reward 設計

```
reward(ℓ) =
  invalid_penalty (-1.0)                           if 解析失敗
  0.0                                               if 可解析但無 agent 通關
  playability_bonus + regret_weight * regret(ℓ)     otherwise

regret(ℓ) = max(0, mean(V_strong(ℓ)) - mean(V_weak(ℓ)))
V_a(ℓ) = (1/M) Σ total_return(trajectory_m)  for agent a over M rollouts
```

- Regret clamp 到 0（負 regret 不懲罰）
- Playable 定義：至少一個 agent（不論強弱）能通關
- Strategy breadth 暫不使用（待決策）

### 5.3 GRPO Update

```
對 batch 中每個 prompt，生成 group_size (=16) 個關卡
組內 z-score normalize rewards → advantages
L = -E[advantage * log π(y|x)] + kl_coeff * KL(π || π_ref)
```

- 不需要 value network（reward 稀疏，不適合訓練 value function）
- KL penalty 使用 frozen reference model 的 log probs

### 5.4 Agent Pool

使用 BabyAI 預訓練 agent，7×7 partially observable 視野。

| Agent ID | 來源 | 類型 | 用途 |
|----------|------|------|------|
| `strong_0` | BabyAI pretrained | 強 agent | Training reward |
| `weak_0` | BabyAI pretrained | 弱 agent | Training reward |
| `strong_held_0` | BabyAI pretrained | 強 agent (held-out) | Evaluation only |
| `weak_held_0` | BabyAI pretrained | 弱 agent (held-out) | Evaluation only |

- Training agents（前 2 個）用於計算 training reward
- Held-out agents（後 2 個）僅用於完整 evaluation，避免 overfitting
- 強/弱 agent 的具體區分方式依 BabyAI 可用的 pretrained checkpoint 決定

#### 5.4.1 Toy Case Agent Pool [added]

Phase 0 使用 SB3 PPO 自行訓練的 agent，用於 pipeline smoke test。訓練環境為 MiniGrid `DoorKeyEnv(size=13)`。

| Agent ID | 來源 | 類型 | 用途 |
|----------|------|------|------|
| `toy_strong_0` | SB3 PPO on DoorKeyEnv(size=13), 充分訓練 | 強 agent | Toy case reward |
| `toy_weak_0` | SB3 PPO on DoorKeyEnv(size=13), 較少訓練步數 | 弱 agent | Toy case reward |

- 強 agent：PPO 訓練至收斂（success rate > 90% on DoorKeyEnv）
- 弱 agent：PPO 提前停止訓練（success rate ~30-60% on DoorKeyEnv）
- 兩者均使用 MiniGrid 的 `ImgObsWrapper`（fully observable image observation）或 `RGBImgPartialObsWrapper`（7×7 partial obs），依實際相容性決定
- 這些 agent 訓練於 DoorKeyEnv 的固定結構（key + locked door + goal），在 LLM 生成的自定義地圖上可能泛化能力有限——這對 smoke test 而言是可接受的，目的是驗證 pipeline 資料流，而非 agent 品質

### 5.5 階段計畫 [updated]

<!-- before: 三階段計畫（Phase 1-3） -->

| Phase | 環境 | 目標 | 時間 |
|-------|------|------|------|
| **Phase 0** [added] | MiniGrid DoorKeyEnv(size=13) | **Toy case**: SB3 PPO agent 訓練 + LLM zero-shot 生成 + 全 pipeline smoke test | Pre-Week 1 |
| Phase 1 | MiniGrid 15×15 (13×13 usable) | Zero-shot / Few-shot baseline，pipeline 跑通 | Week 1 |
| Phase 2 | MiniGrid 15×15 (13×13 usable) | GRPO 訓練，驗證 RL 微調有效 | Week 2 |
| Phase 3 | MiniHack | 擴展到更複雜環境，驗證可擴展性 | Week 3 |

#### Phase 0 詳細說明 [added]

**目的**：以最低成本跑通整個 pipeline（LLM 生成 → 解析 → 環境構建 → Agent rollout → Reward 計算），驗證模組間資料流正確，不追求結果品質。

**流程**：
1. 在 `DoorKeyEnv(size=13)` 上訓練 SB3 PPO agent（strong: 充分訓練; weak: 提前停止）
2. 用 LLM（Qwen3.5-Coder-9B）zero-shot 生成少量地圖（~10-20 張）
3. 將 LLM 生成的地圖送入 parser → 構建 MiniGrid 環境 → 用 toy agent 跑 rollout → 計算 reward
4. 驗證各模組的輸入輸出符合 shared types contract

**成功標準**：
- SB3 PPO strong agent 在 DoorKeyEnv 上 success rate > 90%
- Pipeline 端到端不報錯（即使 LLM 生成的地圖 parse rate 低、agent 表現差）
- 各模組產出的 dataclass 欄位完整且型別正確

## 6. Method Scope

**適用範圍**：
- Grid-based 遊戲環境（MiniGrid、MiniHack）
- 關卡可用結構化文字描述（ASCII grid + JSON / des-file）
- 有明確的 win/lose 條件（agent 可以計算 success 和 return）

**已知限制**：
- 固定 prompt 可能限制關卡多樣性（Phase 1-2）
- Regret 作為唯一 reward signal 可能導致 LLM 只學會一種「強弱差異大」的模式
- 13×13 可用空間限制了關卡的空間複雜度
- Agent pool 品質直接影響 reward signal 品質
- 單張 4090 限制了 batch size 與模型精度

**邊界條件**：
- 如果 LLM zero-shot parse rate < 10%，需要考慮換模型或簡化格式
- 如果 BabyAI strong agent 無法穩定通關隨機關卡，regret signal 不可靠

## 7. Risk Assessment

| Risk Factor | Severity | Likelihood | Mitigation |
|---|---|---|---|
| LLM zero-shot parse rate 過低（<10%） | H | M | Sanity check 實驗先行驗證；fallback 換模型或簡化 ASCII grid 格式 |
| GRPO 訓練不穩定 / reward 不上升 | H | M | 調降 lr、增大 group_size、調整 reward weight |
| LLM mode collapse（只生成少數模板） | M | M | 先依賴 GRPO group sampling（group_size=16）；後續考慮 diversity bonus |
| 單張 4090 VRAM 不足（24GB） | H | M | 使用 QLoRA 4-bit 量化、gradient checkpointing、小 batch size |
| MiniHack des-file 格式太複雜，LLM 無法生成 | H | H | 限制 des-file subset、使用更簡單的 MiniHack 任務、視為 negative result |
| BabyAI agent 在自定義關卡上表現不穩定 | M | M | 驗證 agent 在隨機生成關卡上的 win rate；必要時微調 agent |
| Regret 不適合作為唯一 reward（忽略關卡品質其他面向） | M | M | 加入 strategy breadth（待決策）；觀察生成關卡的質性分析 |
| ASCII grid + JSON 雙格式增加解析複雜度與 LLM 輸出錯誤率 | M | M | Parser 實作容錯機制；必要時退回純 JSON 格式 |
| Regret clamp 到 0 導致 LLM 無法從負 regret 關卡學到資訊 | L | M | 透過 GRPO group normalization，低 reward 仍有負 advantage 信號 |
| Rollout 速度瓶頸（每 iteration 640 次） | M | M | SubprocVecEnv 平行化；調整環境數量 |

## 8. Evaluation Protocol

### 主要指標

| 指標 | 定義 | 用途 |
|------|------|------|
| Playability Rate | 可通關關卡佔所有合法關卡的比例 | 基本品質 |
| Held-out Regret | 用 held-out agents 計算的 regret（mean, median, std） | 技巧深度（核心指標） |
| Parse Success Rate | LLM 輸出能成功解析為合法關卡的比例 | 格式正確性 |

### 次要指標（待決策）

| 指標 | 狀態 | 備註 |
|------|------|------|
| Solution Diversity (JSD) | [deferred] | 具體計算方式待定 |
| Strategy Breadth (action entropy) | [deferred] | 是否加入 reward 待決策 |
| Human Study | [deferred] | 規模與設計待定 |

### Controllability（Phase 2+）

- 控制維度：特定物件的數量（如 key 數量、wall 密度）
- 方法：在 prompt 中加入控制指令，以 Cohen's d 衡量效果
- 比較：有控制 vs 無控制的 prompt 生成結果

### Evaluation 模式

| 模式 | Agent | 關卡數 | 用途 |
|------|-------|--------|------|
| Quick eval | Training agents（strong_0, weak_0） | 100 | 訓練中定期檢查 |
| Full eval | Held-out agents（strong_held_0, weak_held_0） | 100 | Checkpoint 完整評估 |

### Baseline 方法

| Baseline | 說明 |
|----------|------|
| Zero-shot | Qwen3.5-Coder-9B 直接生成，不做任何微調 |
| Few-shot | 用 zero-shot 中篩選的 parse-valid 關卡作為 few-shot examples |

### 統計顯著性

對 held-out regret 等連續指標使用 t-test 或 Mann-Whitney U test 比較 baseline 與 GRPO 訓練後的差異。

## 9. Datasets

### Training Data

本專案不使用傳統 dataset。訓練信號來自：
- **Prompt**：固定的 context prompt（包含遊戲規則 + ASCII grid/JSON 格式規範）
- **Reward**：由 agent pool 遊玩生成關卡後即時計算

### Agent Pool

- **來源**：BabyAI 預訓練 agent（MiniGrid 環境）
- **Observation**：7×7 partially observable grid
- **Strong agent**：BabyAI 完整訓練的 agent
- **Weak agent**：BabyAI 較弱的 agent（較少訓練步數或較簡單架構）

#### Toy Case Agent Pool [added]

- **來源**：SB3 PPO 自行訓練（MiniGrid `DoorKeyEnv(size=13)`）
- **訓練環境**：`MiniGrid-DoorKey-13x13-v0`（agent 需撿鑰匙 → 開門 → 到達 goal）
- **Strong agent**：PPO 訓練至收斂（~500K-1M steps），DoorKeyEnv success rate > 90%
- **Weak agent**：PPO 提前停止（~50K-100K steps），DoorKeyEnv success rate ~30-60%
- **儲存位置**：`checkpoints/agents/toy_strong_0.zip`, `checkpoints/agents/toy_weak_0.zip`
- **注意**：這些 agent 僅用於 Phase 0 pipeline smoke test，不用於正式實驗

### Evaluation Data

- 評估時由 LLM 生成 100 個關卡
- Quick eval：使用 training agents
- Full eval：使用 held-out agents，避免 overfitting

## 10. Experiment Design

### Hypotheses

- **H1**：Qwen3.5-Coder-9B 能在 zero-shot 下生成 parse-valid 的 MiniGrid ASCII grid + JSON 關卡（parse rate > 10%）
- **H2**：GRPO 微調後，playability rate 與 regret 均高於 zero-shot baseline
- **H3**：Regret 提升不以犧牲 playability 為代價（playability rate 不低於 baseline）
- **H4**：方法可擴展到 MiniHack（更複雜環境仍能觀察到改善）

### Counter-hypotheses

- **CH1**：LLM 無法穩定生成 ASCII grid + JSON 格式，parse rate 極低（→ 模型選擇問題，或退回純 JSON）
- **CH2**：GRPO 訓練後 reward 不上升（→ 訓練設計問題：lr, kl_coeff, group_size）
- **CH3**：Training regret 上升但 held-out regret 不上升（→ overfitting to training agents）
- **CH4**：LLM mode collapse，生成高度重複的關卡（→ 需要 diversity 機制）

### Experiment Ordering [updated]

<!-- before: 採用 (a) build full pipeline first then ablate 策略，Phase 1 → 2 → 3 -->
採用 **(a) build full pipeline first then ablate** 策略，但在正式 Phase 1 之前先執行 **Phase 0 Toy Case** 以最低成本驗證 pipeline 資料流：
- Phase 0（Toy Case）：訓練 SB3 PPO agent + LLM zero-shot + 全 pipeline smoke test
- Phase 1：Zero-shot / Few-shot baseline（BabyAI agents）
- Phase 2：GRPO 訓練
- Phase 3：MiniHack 擴展
- 風險：如果核心假設（H1）不成立，會浪費 pipeline 建設時間。以 Phase 0 toy case + sanity check 實驗緩解。

### 實驗列表

#### Experiment T: Toy Case — Pipeline Smoke Test (Phase 0) [added]
- **Hypothesis**: 全 pipeline（LLM 生成 → Parser → MiniGrid 環境 → SB3 Agent rollout → Reward 計算）可端到端跑通，各模組間資料流符合 shared types contract
- **Expected result**: Pipeline 不報錯；SB3 PPO strong agent 在 DoorKeyEnv 上 success rate > 90%；LLM 生成的地圖至少部分可解析（parse rate 不限）
- **If violated**: 若 pipeline 資料流有斷裂 → 修復模組介面；若 SB3 agent 訓練失敗 → 檢查環境 wrapper 與超參數
- **Setup**:
  1. 訓練 SB3 PPO（strong + weak）on `MiniGrid-DoorKey-13x13-v0`
  2. LLM zero-shot 生成 10-20 張地圖
  3. 完整 pipeline：parse → build env → agent rollout → reward 計算
  4. 驗證所有 shared type dataclass 欄位正確
- **不追求**: agent 在 LLM 地圖上的表現品質、regret 的意義性

#### Experiment 0: Sanity Check — LLM 結構化輸出能力
- **Hypothesis**: Qwen3.5-Coder-9B 在 zero-shot 下能生成 parse-valid 的 MiniGrid ASCII grid + JSON（rate > 10%）
- **Expected result**: Parse success rate 10-50%
- **If violated**: 換模型（嘗試其他 Coder 模型）或退回純 JSON 格式
- **Setup**: LLM Policy 模組 + Parser，100+ 次生成，統計 parse rate

#### Experiment 1: Zero-shot Baseline
- **Hypothesis**: Zero-shot LLM 能生成部分可玩關卡，但 regret 接近 0（未針對技巧深度優化）
- **Expected result**: Playability rate 30-60%, Regret ≈ 0
- **If violated**: 若 playability 已很高，GRPO 的提升空間可能有限
- **Setup**: LLM Policy + Game Environment + Reward Calculator 完整 pipeline

#### Experiment 2: Few-shot Baseline
- **Hypothesis**: 用 zero-shot 篩選的 parse-valid 關卡作為 few-shot examples 能小幅提升 playability 和 regret
- **Expected result**: Playability rate 比 zero-shot 高 5-15%
- **If violated**: Few-shot 的知識遷移在此任務上可能無效
- **Setup**: 同 Experiment 1，prompt 包含從 zero-shot 篩選的 parse-valid examples

#### Experiment 3: GRPO Training on MiniGrid
- **Hypothesis**: GRPO 微調後，playability rate 和 regret 均顯著高於 baselines
- **Expected result**: Playability rate > 70%, Regret 顯著高於 baseline
- **If violated**: 若 reward curve 不上升 → 訓練設計問題；若 training regret 上升但 held-out 不上升 → overfitting
- **Setup**: 全部三個模組 + train.py，~200-500 iterations，batch_size=4, group_size=16

#### Experiment 4: Reward Weight Ablation
- **Hypothesis**: 不同 regret_weight 和 playability_bonus 比例會影響關卡品質的不同面向
- **Expected result**: 較高 regret_weight → 更高 regret 但可能降低 playability
- **If violated**: Reward weight 對結果影響不大 → regret 和 playability 可能高度相關
- **Setup**: 2-3 組不同 reward weight 配置

#### Experiment 5: Controllability (Phase 2+)
- **Hypothesis**: 在 prompt 中加入物件數量控制指令後，生成關卡的物件分布與指令一致
- **Expected result**: Cohen's d > 0.5（中等效果量）
- **If violated**: LLM 對 prompt 控制指令不敏感
- **Setup**: LLM Policy（加入控制 prompt）+ Game Environment + 統計分析

#### Experiment 6: MiniHack Extension (Phase 3)
- **Hypothesis**: 同樣的 GRPO pipeline 可擴展到 MiniHack
- **Expected result**: 觀察到 regret 改善（幅度可能小於 MiniGrid）
- **If violated**: MiniHack des-file 格式太複雜 → 視為 negative result 報告
- **Setup**: 全部模組適配 MiniHack

## 11. System Architecture & Components

### 架構圖

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │  LLM     │───▶│  Level   │───▶│  Game Environment │  │
│  │  Policy  │    │  Parser  │    │  Runner            │  │
│  │ (Qwen 9B)│    │ ASCII+JSON   │  (MiniGrid 15×15) │  │
│  └────▲─────┘    └──────────┘    └────────┬──────────┘  │
│       │                                   │              │
│       │          ┌──────────┐    ┌────────▼──────────┐  │
│       │          │  Reward  │◀───│  BabyAI Agent     │  │
│       └──────────│Calculator│    │  Pool (7×7 view)  │  │
│    GRPO update   └──────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Evaluation Suite │
              │ Quick / Full     │
              └──────────────────┘
```

### 模組間資料流

```
LLMPolicy.generate(prompts: list[str])
    → GenerationOutput(texts, log_probs, token_ids)

GameEnvironment.batch_evaluate(texts: list[str])
    → list[RolloutResult | None]

RewardCalculator.compute_batch_rewards(rollouts: list[RolloutResult | None])
    → list[RewardOutput]

RewardCalculator.compute_advantages_grpo(rewards: list[float], group_size: int)
    → Tensor (advantages)

LLMPolicy.update(GRPOBatch)
    → dict (loss, kl, mean_reward)
```

### 11.1 Shared Types (`shared/types.py`)

所有跨模組邊界的資料結構定義在 `shared/types.py`，作為模組間的 **contract**。修改任何 shared type 時須同步更新所有使用該 type 的模組。範例實例見 `shared/type_examples.py`。

```python
# shared/types.py — Shared type definitions

# ---------------------------------------------------------------------------
# Module A – LLM Policy
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutput:
    """LLMPolicy.generate() 的回傳值。"""
    texts: list[str]            # 生成的原始文字（ASCII grid + JSON）
    log_probs: Tensor           # shape (batch, seq_len) — 每個 token 的 log prob
    token_ids: Tensor           # shape (batch, seq_len)


@dataclass
class GRPOBatch:
    """傳入 LLMPolicy.update() 的一個 training batch。"""
    token_ids: Tensor           # shape (batch, seq_len)
    log_probs: Tensor           # 生成時的 log probs, shape (batch, seq_len)
    ref_log_probs: Tensor       # reference model 的 log probs, shape (batch, seq_len)
    rewards: Tensor             # shape (batch,)
    advantages: Tensor          # GRPO group-normalized advantages, shape (batch,)


# ---------------------------------------------------------------------------
# Module B – Level Parser & Game Environment
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """GameEnvironment.parse_level() 的回傳值。"""
    success: bool
    level_config: Optional[dict] = None   # 見 type_examples.py 的 level_config 範例
    error_msg: Optional[str] = None


@dataclass
class Trajectory:
    """單次 agent rollout 的紀錄。"""
    states: list[np.ndarray]    # 每步的 observation (7×7×3 partially observable)
    actions: list[int]          # MiniGrid action ids (0-6)
    rewards: list[float]        # 每步的 reward
    total_return: float         # 累計 return
    success: bool               # 是否通關
    length: int                 # trajectory 步數


@dataclass
class RolloutResult:
    """GameEnvironment.run_rollouts() 的回傳值。"""
    level_config: dict
    trajectories: dict[str, list[Trajectory]]
    # key = agent_id (e.g. "strong_0", "weak_0"), value = M 次 rollout


# ---------------------------------------------------------------------------
# Module C – Reward Calculator & Evaluation
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """RewardCalculator 的超參數。Per SPEC §5.2。"""
    regret_weight: float = 1.0
    playability_bonus: float = 1.0
    invalid_penalty: float = -1.0
    # strategy_breadth_weight: float = 0.0  # [deferred] — PD-1


@dataclass
class RewardOutput:
    """RewardCalculator.compute_reward() 的回傳值。Per SPEC §5.2。"""
    total_reward: float
    regret: float               # max(0, mean(V_strong) - mean(V_weak))
    playable: bool              # 至少一個 agent 通關
    breakdown: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """EvaluationSuite.evaluate() 的回傳值。Per SPEC §8。"""
    parse_success_rate: float       # LLM 輸出成功解析為合法關卡的比例
    playability_rate: float         # 可通關關卡佔所有合法關卡的比例
    held_out_regret: dict           # {"mean": float, "median": float, "std": float}
    solution_diversity: Optional[dict] = None   # [deferred]
    controllability: Optional[dict] = None      # Phase 2+
    eval_mode: str = "full"         # "quick" or "full"
    num_levels: int = 0
    raw_data: list[dict] = field(default_factory=list)
```

#### 各 Shared Type 的生產者 / 消費者 / 不變量

| Shared Type | 生產者 (Producer) | 消費者 (Consumer) | 不變量 (Invariants) |
|---|---|---|---|
| `GenerationOutput` | Module A: `LLMPolicy.generate()` | `train.py`（傳給 Module B） | `texts`, `log_probs`, `token_ids` 長度一致 = batch_size × group_size；`log_probs` 與 `token_ids` 的 seq_len 維度一致 |
| `GRPOBatch` | `train.py`（組裝） | Module A: `LLMPolicy.update()` | 所有 Tensor 的 batch 維度一致；`advantages` 經過 group-wise z-score normalization |
| `ParseResult` | Module B: `GameEnvironment.parse_level()` | Module B: `run_rollouts()`（取 `level_config`）, `train.py` | `success=True` 時 `level_config` 非 None 且通過全部驗證；`success=False` 時 `error_msg` 非 None |
| `Trajectory` | Module B: `GameEnvironment.run_rollouts()` | Module C: `RewardCalculator` | `length = len(actions) = len(rewards) = len(states)`；`total_return = sum(rewards)`；`actions` 值域 0-6（MiniGrid action space） |
| `RolloutResult` | Module B: `GameEnvironment.run_rollouts()` | Module C: `RewardCalculator`, `EvaluationSuite` | `trajectories` 的每個 agent_id 對應 `num_rollouts_per_agent` 條 `Trajectory`；`level_config` 與 `ParseResult.level_config` 一致 |
| `RewardConfig` | `config/default.yaml`（載入） | Module C: `RewardCalculator` | `invalid_penalty < 0`；`playability_bonus ≥ 0`；`regret_weight ≥ 0` |
| `RewardOutput` | Module C: `RewardCalculator.compute_reward()` | `train.py`（取 `total_reward` 計算 advantages） | `regret ≥ 0`（已 clamp）；若 `playable=False` 且非解析失敗則 `total_reward = 0.0`；若解析失敗則 `total_reward = invalid_penalty` |
| `EvalReport` | Module C: `EvaluationSuite.evaluate()` | `evaluate.py`, wandb logging | `0 ≤ parse_success_rate, playability_rate ≤ 1`；`eval_mode ∈ {"quick", "full"}`；`num_levels > 0` |

### 11.2 Module Specifications

### Module A: LLM Policy (`llm_policy/`)

**Class**: `LLMPolicy`

**職責**: 管理 Qwen3.5-Coder-9B 的載入（QLoRA 4-bit）、推理、GRPO 更新

**輸入/輸出**:
- `generate(prompts: list[str]) → GenerationOutput`
  - `texts: list[str]` — 生成的原始文字（ASCII grid + JSON）
  - `log_probs: Tensor (batch, seq_len)` — 每個 token 的 log prob
  - `token_ids: Tensor (batch, seq_len)`
- `get_ref_log_probs(token_ids: Tensor) → Tensor (batch, seq_len)`
- `update(grpo_batch: GRPOBatch) → dict`

**消費的 Shared Types**: `GRPOBatch`（from `train.py`）
**生產的 Shared Types**: `GenerationOutput`

**關鍵配置**: QLoRA 4-bit, LoRA rank=64, alpha=128, target modules: q/k/v/o/gate/up/down_proj

**依賴**: Transformers, PEFT, TRL（或自行實作 GRPO）, bitsandbytes

**子模組**:
- `policy.py` — LLMPolicy class
- `grpo.py` — GRPO update 邏輯
- `prompts.py` — Prompt template 管理（ASCII grid + JSON 格式規範）

### Module B: Game Environment (`game_env/`)

**Class**: `GameEnvironment`

**職責**: 解析 LLM 輸出、構建遊戲環境、使用 BabyAI agent 執行 rollout

**輸入/輸出**:
- `parse_level(llm_output: str) → ParseResult`
  - 從文字提取 ASCII grid + JSON → grid 驗證 → schema 驗證 → 語義驗證（起終點、連通性、物件不重疊、座標範圍 0-12）
- `run_rollouts(level_config: dict, num_rollouts_per_agent: int) → RolloutResult`
  - 使用 SubprocVecEnv 平行化 rollout
- `batch_evaluate(llm_outputs: list[str], num_rollouts_per_agent: int) → list[RolloutResult | None]`

**消費的 Shared Types**: `GenerationOutput.texts`（from Module A via `train.py`）
**生產的 Shared Types**: `ParseResult`, `Trajectory`, `RolloutResult`

**依賴**: MiniGrid ≥2.3, MiniHack ≥0.1.5 (Phase 3), BabyAI

**子模組**:
- `parser.py` — Level parser + validator（ASCII grid + JSON 解析）
- `environment.py` — GameEnvironment class, rollout runner (SubprocVecEnv)
- `wrappers.py` — 環境 wrapper（確保與 BabyAI agent 相容）
- `minihack/` — Phase 3 MiniHack 擴展

### Module C: Reward & Evaluation (`reward_eval/`)

**Class**: `RewardCalculator`, `EvaluationSuite`

**職責**: 計算 training reward、evaluation metrics、生成報告

**RewardCalculator 輸入/輸出**:
- `compute_reward(rollout: RolloutResult | None) → RewardOutput`
- `compute_batch_rewards(rollouts: list[...]) → list[RewardOutput]`
- `compute_advantages_grpo(rewards: list[float], group_size: int) → Tensor`

**EvaluationSuite 輸入/輸出**:
- `evaluate(llm_outputs: list[str], mode: str = "full") → EvalReport`
  - `mode="quick"`: 使用 training agents
  - `mode="full"`: 使用 held-out agents
- `export_report(report: EvalReport, path: str)`

**消費的 Shared Types**: `RolloutResult`, `Trajectory`, `RewardConfig`
**生產的 Shared Types**: `RewardOutput`, `EvalReport`

**依賴**: NumPy, SciPy, wandb

**子模組**:
- `reward.py` — RewardCalculator
- `evaluation.py` — EvaluationSuite（quick + full 模式）
- `metrics.py` — 各項指標實作
- `visualization.py` — 圖表生成

### 11.3 Inter-Module Communication

| From → To | 通訊方式 | 交換的 Shared Type | Sync/Async | 備註 |
|---|---|---|---|---|
| Module A → Module B | 間接（via `train.py` function call） | `GenerationOutput.texts` → `str` | sync | `train.py` 取出 texts 傳給 `batch_evaluate()` |
| Module B → Module C | 間接（via `train.py` function call） | `RolloutResult` | sync | `train.py` 取出 rollouts 傳給 `compute_batch_rewards()` |
| Module C → Module A | 間接（via `train.py` 組裝 `GRPOBatch`） | `RewardOutput.total_reward` → `GRPOBatch` | sync | `train.py` 從 rewards 計算 advantages，組裝 `GRPOBatch` |
| Module A → Module A | 內部 function call | `GenerationOutput.token_ids` → ref_log_probs | sync | `get_ref_log_probs()` 計算 KL penalty 所需 |
| Module C → wandb | callback / event | `EvalReport`, metrics dict | async | fire-and-forget logging |
| Config → All Modules | config-driven wiring | `RewardConfig`, hyperparams | — | `train.py` 載入 YAML 後分配給各模組 |

**設計選擇說明**：所有模組間通訊均透過 `train.py` 中介，模組之間不直接互相呼叫。這確保每個模組可獨立測試——只需用 shared types 建立 mock 即可。

### 11.4 Pseudo Config

```yaml
# === Game ===
game: "minigrid"                    # "minigrid" or "minihack"
grid_size: 13                       # usable space (total = grid_size + 2)

# === LLM Policy (Module A) ===
model_name: "Qwen/Qwen3.5-9B"
quantization: "4bit"                # QLoRA for 4090
lora_rank: 64
lora_alpha: 128
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
max_new_tokens: 2048
temperature: 0.8

# === GRPO Training ===
group_size: 16                      # samples per prompt
kl_coeff: 0.05
learning_rate: 1.0e-5
batch_size: 4                       # prompts per batch (total = 4 × 16 = 64 levels)
num_iterations: 500
gradient_accumulation_steps: 1

# === Reward (Module C) ===
regret_weight: 1.0
playability_bonus: 1.0
invalid_penalty: -1.0
# strategy_breadth_weight: 0.0     # [deferred]

# === Rollout (Module B) ===
num_rollouts_per_agent: 5
num_envs: 16                        # SubprocVecEnv parallel environments
agent_pool_path: "checkpoints/agents/"
training_agents:
  - strong_0                        # BabyAI pretrained strong
  - weak_0                          # BabyAI pretrained weak
held_out_agents:
  - strong_held_0                   # BabyAI pretrained strong (held-out)
  - weak_held_0                     # BabyAI pretrained weak (held-out)

# === Toy Case (Phase 0) === [added]
toy_case:
  train_env: "MiniGrid-DoorKey-13x13-v0"
  strong_agent_steps: 1_000_000       # PPO training steps for strong agent
  weak_agent_steps: 50_000            # PPO training steps for weak agent (early stop)
  toy_agents:
    - toy_strong_0
    - toy_weak_0
  num_test_levels: 20                 # LLM zero-shot levels for smoke test
  ppo_hyperparams:
    learning_rate: 3.0e-4
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    ent_coef: 0.01

# === Evaluation ===
eval_num_levels: 100                # levels per evaluation
eval_interval: 50                   # evaluate every N iterations

# === Logging ===
wandb_project: "rlvr-level-gen"
checkpoint_dir: "checkpoints/"
log_dir: "logs/"
```

## 12. Experiment Implementation Architecture

### Experiment T: Toy Case — Pipeline Smoke Test [added]

```
# Step 1: Train SB3 PPO agents
train_ppo("MiniGrid-DoorKey-13x13-v0", total_timesteps=1_000_000) → toy_strong_0.zip
train_ppo("MiniGrid-DoorKey-13x13-v0", total_timesteps=50_000)    → toy_weak_0.zip

# Step 2: LLM zero-shot generation
LLMPolicy.generate(prompts) → texts  (10-20 levels)

# Step 3: Full pipeline
for text in texts:
    ParseResult = GameEnvironment.parse_level(text)
    if ParseResult.success:
        RolloutResult = GameEnvironment.run_rollouts(
            level_config, agents=[toy_strong_0, toy_weak_0], num_rollouts=5)
        RewardOutput = RewardCalculator.compute_reward(RolloutResult)

# Step 4: Verify
assert all shared type fields are present and correctly typed
```

- 使用模組：A（generate only, 不 update）, B（完整）, C（compute_reward only）
- Agent：SB3 PPO（自行訓練），非 BabyAI
- 目的：驗證 pipeline 資料流，不追求結果品質
- 預計耗時：agent 訓練 ~30 min（DoorKeyEnv 較簡單），pipeline 跑通 ~10 min

### Experiment 0: Sanity Check

```
LLMPolicy.generate(prompts) → texts
GameEnvironment.parse_level(text) → ParseResult
統計 parse success rate
```

- 使用模組：A（generate only）, B（parse only）
- 排除：Reward Calculator, GRPO update, Agent rollout
- 目的：驗證 LLM + Parser 的基本可行性

### Experiment 1 & 2: Zero-shot / Few-shot Baseline

```
LLMPolicy.generate(prompts) → texts
GameEnvironment.batch_evaluate(texts) → rollouts
RewardCalculator.compute_batch_rewards(rollouts) → rewards
EvaluationSuite.evaluate(texts, mode="full") → EvalReport
```

- 使用模組：A（generate only, 不 update）, B（完整）, C（完整）
- 排除：GRPO update
- Few-shot 差異：prompt 中包含從 zero-shot 篩選的 parse-valid examples

### Experiment 3: GRPO Training

```
完整 training loop (train.py):
  每 iteration:
    LLMPolicy.generate(4 prompts × 16 samples = 64 levels)
    → GameEnvironment.batch_evaluate(64 levels, 2 agents × 5 rollouts = 640 rollouts)
    → RewardCalculator.compute_batch_rewards → compute_advantages_grpo
    → LLMPolicy.update

  每 eval_interval iterations:
    Quick eval (training agents, 100 levels)

  訓練結束:
    Full eval (held-out agents, 100 levels)
```

- 使用模組：A, B, C 全部
- 每 eval_interval iterations 做 quick eval checkpoint

### Experiment 4: Reward Weight Ablation

- 同 Experiment 3 架構
- 不同配置：調整 `regret_weight`, `playability_bonus`
- 2-3 組平行實驗

### Experiment 5: Controllability

- 同 Experiment 1 架構（generate + evaluate，不 train）
- LLM Policy 使用帶有控制指令的 prompt（例如「生成包含 3 把鑰匙的關卡」）
- 額外統計：物件數量分布 + Cohen's d

### Experiment 6: MiniHack Extension

- 同 Experiment 3 架構
- Module B 替換為 MiniHack 環境 + des-file parser
- Module A 更新 prompt template

## 13. Visualization Opportunities

| 類別 | 視覺化項目 | 用途 |
|------|-----------|------|
| Training | Reward curve (per iteration) | 監控訓練進度 |
| Training | KL divergence curve | 監控 policy 偏離程度 |
| Training | Parse success rate over time | 監控格式正確性趨勢 |
| Training | Playability rate over time | 監控關卡品質趨勢 |
| Evaluation | Regret distribution (histogram) | 分析關卡品質分布 |
| Evaluation | Baseline vs GRPO 對比 bar chart | 主要結果展示 |
| Evaluation | Reward weight ablation 對比 | 消融分析 |
| Level | MiniGrid 關卡渲染（grid 視覺化） | 定性分析生成品質 |
| Level | 生成關卡的物件統計分布 | 多樣性分析 |
| Agent | Strong vs Weak agent trajectory 視覺化 | 理解 regret 的來源 |
| Debug | 解析失敗案例的 LLM 原始輸出 | Debug parser |

## 14. VRAM Budget Estimation

硬體：單張 NVIDIA RTX 4090（24 GB VRAM）

| Operation | Model/Data Size | Est. VRAM | Notes |
|---|---|---|---|
| Qwen3.5-9B base (4-bit QLoRA) | 9B params × 0.5 bytes | ~5 GB | bitsandbytes NF4 |
| LoRA adapter (rank=64, BF16) | ~50M params × 2 bytes | ~0.1 GB | 可訓練參數 |
| Reference model (4-bit, frozen) | 9B params × 0.5 bytes | ~5 GB | KL penalty 計算 |
| Optimizer states (AdamW) | LoRA params × 8 bytes | ~0.4 GB | 僅 LoRA 參數 |
| Activations (batch=4, seq=2048) | — | ~3-5 GB | gradient checkpointing |
| KV cache (inference) | — | ~1-2 GB | 生成時 |
| Agent rollout | — | 0 GPU | CPU 執行 |
| **Total (training)** | — | **~15-18 GB** | 可放入 4090 |
| **Total (inference only)** | — | **~6-8 GB** | 餘裕充足 |

- 必須使用 4-bit 量化（QLoRA）+ gradient checkpointing
- Agent rollout 在 CPU 上用 SubprocVecEnv 執行
- 如仍超出：降低 LoRA rank 或使用 CPU offloading

## 15. Environment & Dependencies

**Python**: ≥ 3.10

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.1 | 核心框架 |
| transformers | ≥ 4.40 | LLM 載入與推理 |
| peft | ≥ 0.10 | LoRA / QLoRA |
| trl | ≥ 0.8 | GRPO Trainer（或自行實作） |
| bitsandbytes | ≥ 0.43 | 4-bit 量化 |
| vllm | ≥ 0.4 | 加速推理（optional） |
| minigrid | ≥ 2.3 | MiniGrid 環境 |
| babyai | latest | 預訓練 agent |
| minihack | ≥ 0.1.5 | MiniHack 環境（Phase 3） |
| numpy | latest | 數值計算 |
| scipy | latest | 統計檢定 |
| wandb | latest | Experiment tracking |
| pyyaml | latest | Config 讀取 |
| jsonschema | latest | JSON schema 驗證 |

**潛在衝突**：
- `minihack` 依賴 NLE（NetHack Learning Environment），可能需要額外系統套件
- `vllm` 與 `transformers` 版本需相容
- `babyai` 與 `minigrid` 版本需相容

**Conda 環境**：建議單一環境，MiniHack 若有衝突再分離。

## 16. Directory Structure

```
project/
├── config/                  # YAML 配置檔
│   ├── default.yaml         # 預設超參數（含所有可調參數）
│   ├── minigrid.yaml        # MiniGrid 特定配置覆寫
│   └── minihack.yaml        # MiniHack 特定配置（Phase 3）
├── llm_policy/              # Module A: LLM Policy
│   ├── __init__.py
│   ├── policy.py            # LLMPolicy class (QLoRA 載入 + generate + update)
│   ├── grpo.py              # GRPO update 實作
│   └── prompts.py           # Prompt template（ASCII grid + JSON 格式規範）
├── game_env/                # Module B: Game Environment
│   ├── __init__.py
│   ├── parser.py            # Level parser（ASCII grid + JSON 解析 + 驗證）
│   ├── environment.py       # GameEnvironment class + SubprocVecEnv rollout
│   ├── wrappers.py          # MiniGrid 環境 wrapper（BabyAI agent 相容）
│   └── minihack/            # Phase 3 MiniHack 擴展
│       ├── parser.py
│       └── environment.py
├── reward_eval/             # Module C: Reward & Evaluation
│   ├── __init__.py
│   ├── reward.py            # RewardCalculator（regret + playability）
│   ├── evaluation.py        # EvaluationSuite（quick + full 模式）
│   ├── metrics.py           # 指標計算（playability, regret, parse rate）
│   └── visualization.py     # 圖表生成
├── shared/                  # 共用程式碼
│   ├── types.py             # 共用 dataclass（canonical source）
│   └── type_examples.py     # Dataclass 範例實例
├── baselines/               # Baseline 腳本
│   └── run_baseline.py      # Zero-shot / Few-shot baseline
├── toy_case/                # [added] Phase 0: Toy Case 腳本
│   ├── train_agent.py       # SB3 PPO agent 訓練（DoorKeyEnv）
│   └── run_toy_pipeline.py  # 端到端 pipeline smoke test
├── train.py                 # 主訓練迴圈（Module A 整合 B + C）
├── evaluate.py              # 獨立評估腳本（quick + full 模式）
├── checkpoints/             # Model & agent checkpoints
│   └── agents/              # BabyAI 預訓練 + toy case agent checkpoints
├── logs/                    # 訓練日誌
├── results/                 # 實驗結果與報告
├── docs/                    # 文件
│   ├── SPEC.md
│   ├── TODO.md
│   └── AMBIGUITY.md
└── tests/                   # 測試
    ├── test_parser.py
    ├── test_reward.py
    └── test_integration.py
```

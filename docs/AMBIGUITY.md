# Ambiguity List — RLVR Game Level Generation

## Round 1（Research-Level）[resolved]

### Method & Scope

- **Q1**: 訓練時使用的 prompt 策略為何？
  - **A1**: Phase 1-2 使用固定 prompt，Phase 3 待決策 [resolved]

- **Q2**: Regret index 可能為負，如何處理？
  - **A2**: Clamp 到 0 [resolved]

- **Q3**: 是否有 curriculum 機制？
  - **A3**: 沒有，固定 13×13 grid（MiniGrid），MiniHack 待決策 [resolved]

- **Q4**: 如何防止 mode collapse？
  - **A4**: 先用 GRPO group sampling，後續再考慮 [resolved]

- **Q5**: Playable 的定義？
  - **A5**: 至少一個 agent（不論強弱）能通關 [resolved]

- **Q6**: MiniGrid 關卡 scope？
  - **A6**: 無限制。LLM 生成 ASCII grid（牆壁/地板地形）+ JSON（物件放置） [resolved]

### Evaluation

- **Q7**: 成功標準？
  - **A7**: 不需量化標準，跑出結果高即可 [resolved]

- **Q8**: Solution Diversity 計算方式？
  - **A8**: [deferred] 具體做法待決策

- **Q9**: Controllability 控制維度？
  - **A9**: 控制特定物件的數量 [resolved]

- **Q10**: Human Study 設計？
  - **A10**: [deferred] 規模與設計待定

- **Q11**: Baseline 列表？
  - **A11**: 僅 zero-shot 與 few-shot [resolved]

### Core Idea

- **Q12**: 為什麼 GRPO？
  - **A12**: Reward 稀疏，不適合訓練 value function [resolved]

- **Q13**: 為什麼 Qwen3.5-9B？
  - **A13**: 尚未驗證，需 sanity check。Fallback 是換模型 [resolved]

- **Q14**: 如何區分假設錯誤 vs 訓練設計問題？
  - **A14**: Training reward 不上升 → 訓練設計問題；Regret 上升但不泛化 → agent/regret 設計問題 [resolved]

- **Q15**: LLM vs PCGRL 的優勢？
  - **A15**: 可解釋性 + 利用預訓練知識 [resolved]

---

## Round 2（Implementation-Level）[resolved]

### Architecture & Data Format

- **Q16**: Grid + JSON 格式？
  - **A16**: ASCII grid（W=wall, .=floor），只有 wall 跟地面放在 grid，其他物件在 JSON [resolved]

- **Q17**: MiniGrid 13×13 的定義？
  - **A17**: 可用空間 13×13，整體 15×15（含外牆） [resolved]

- **Q18**: Agent observation 格式？
  - **A18**: BabyAI pretrained agent，7×7 partially observable [resolved]

### Infrastructure & Compute

- **Q19**: GPU 資源？
  - **A19**: 單張 RTX 4090（24GB） [resolved]

- **Q20**: Rollout 平行化策略？
  - **A20**: SubprocVecEnv，環境數量在 config 可調，預設 16 [resolved]

### Experiment Logistics

- **Q21**: 每 iteration 規模？
  - **A21**: batch_size=4, group_size=16, 2 agents, 5 rollouts → 640 rollouts/iteration [resolved]

- **Q22**: Few-shot examples 篩選？
  - **A22**: 從 zero-shot 輸出中選 parse-valid [resolved]

- **Q23**: Evaluation 關卡數與模式？
  - **A23**: 100 個關卡；Quick eval（training agents）+ Full eval（held-out agents） [resolved]

### Module Integration

- **Q24**: Mock 程度？
  - **A24**: API 簽名正確 + 回傳合理隨機值即可 [resolved]

- **Q25**: 版本控制策略？
  - **A25**: 直接 push 到 main [resolved]

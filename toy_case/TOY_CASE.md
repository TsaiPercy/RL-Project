# Toy Case — Phase 0 端到端煙霧測試

## 目的

`toy_case/` 是整個 RLVR Pipeline 的 **Phase 0 驗證工具包**，目標是在正式大規模訓練之前，用小規模實驗確認四件事：

1. **LLM 能生成有效格式的關卡** (Experiment 0: Sanity Check)
2. **Agent 能在 MiniGrid 環境中被訓練出強弱差異** (Agent Training)
3. **整條 Pipeline (生成 → 解析 → Rollout → Reward → GRPO) 的資料流能完整跑通** (Pipeline Smoke Test)
4. **訓練 path 在目標 GPU (4090, 24GB) 上記憶體不爆** (Training Memory Smoke Test) [added 2026-05-07]

---

## 檔案說明

| 檔案 | 功能 |
|------|------|
| `sanity_check.py` | Experiment 0：驗證 LLM zero-shot 生成 MiniGrid 關卡的 parse 成功率是否 > 10% |
| `train_agent.py` | 訓練 Toy Agent Pool：一個強 agent (`toy_strong_0`) 和一個弱 agent (`toy_weak_0`) |
| `run_toy_pipeline.py` | 端到端 Pipeline 煙霧測試，支援 Mock 模式和 Real 模式 |
| `train_smoke_test.py` | 訓練 path 記憶體驗證：跑一次完整 GRPO update step，確認 4090 撐得住 [added 2026-05-07] |

---

## 各腳本詳細說明

### 1. `sanity_check.py` — LLM 結構化輸出能力驗證

**對應 SPEC**: §10 Experiment 0, §12 Experiment 0

**假說**: Qwen3.5-9B zero-shot 能生成可被 parse 的 MiniGrid ASCII Grid + JSON 格式

**判斷標準**: parse rate > 10%，若未達標則考慮換模型或退回純 JSON 格式

**流程**:
1. 載入 LLM (Qwen3.5-9B, 4-bit 量化 + LoRA)
2. 使用系統 prompt 指示 LLM 生成 MiniGrid 關卡
3. 對生成的文字用 `simple_parse()` 進行格式驗證：
   - 是否包含 `Grid:` 段落，且為 15×15 的 ASCII grid（由 `W` 和 `.` 組成，外牆 ring 必須完整）[impl-updated 2026-05-07]
   - 是否包含合法 JSON，含 `objects` 和 `agent_start` 欄位
   - 物件類型是否合法（wall/key/door/ball/box/goal/lava）
   - 座標是否在範圍內
   - 是否至少有一個 goal
4. 統計 parse 成功率，輸出錯誤類別分布

**使用方式**:
```bash
python -m toy_case.sanity_check --num-levels 100 --config config/default.yaml
python -m toy_case.sanity_check --num-levels 20 --batch-size 5
```

**輸出**: `results/sanity_check/` 目錄下的各 level 文字檔 + `stats.json`

---

### 2. `train_agent.py` — 訓練 Toy Agent Pool

**對應 SPEC**: §5.4.1 Toy Case Agent Pool, §12 Experiment T

**目標**: 訓練出一對強弱 agent，用於後續 reward 計算中的 regret 信號

| Agent ID | 訓練步數 | 預期 Success Rate |
|----------|----------|-------------------|
| `toy_strong_0` | 1,000,000 steps | > 90% |
| `toy_weak_0` | 50,000 steps (提前停止) | ~30-60% |

**訓練設定**:
- 環境: `MiniGrid-DoorKey-15x15-v0`
- 演算法: SB3 PPO + CnnPolicy
- 觀察空間: `ImgObsWrapper` 包裝後的 7×7×3 image
- 4 個平行環境訓練 + 1 個環境做 evaluation callback

**使用方式**:
```bash
python -m toy_case.train_agent --agent both          # 訓練強+弱
python -m toy_case.train_agent --agent strong        # 只訓練強 agent
python -m toy_case.train_agent --agent weak          # 只訓練弱 agent
python -m toy_case.train_agent --evaluate-only       # 僅評估已存在的 checkpoint
```

**輸出**: `checkpoints/agents/toy_strong_0.zip` 和 `checkpoints/agents/toy_weak_0.zip`

---

### 3. `run_toy_pipeline.py` — 端到端 Pipeline 煙霧測試

**對應 SPEC**: §12 Experiment T

**目標**: 驗證三個模組 (LLM → Game Env → Reward) 之間的資料格式互通

#### Mock 模式 (`--use-mock`)

不需要 GPU 或真實模型，用 mock 模組測試資料流的 5 個步驟：

| 步驟 | 驗證內容 |
|------|---------|
| Step 1 | `MockLLMPolicy.generate()` → `GenerationOutput` 結構正確 |
| Step 2 | 模擬 parse → `ParseResult` 結構正確 |
| Step 3 | 模擬 agent rollout → `RolloutResult` + `Trajectory` 結構正確 |
| Step 4 | `MockRewardCalculator.compute_batch_rewards()` → `RewardOutput` 結構正確 |
| Step 5 | `compute_advantages_grpo()` → advantages tensor shape 正確 |

每一步都會驗證 dataclass 欄位的完整性和型別一致性。

#### Real 模式 (預設)

載入真實 LLM 生成關卡，進行初步 parse 驗證（完整 agent rollout 需等 Module B 完成）。

**使用方式**:
```bash
python -m toy_case.run_toy_pipeline --use-mock                    # Mock 模式（快速、無 GPU）
python -m toy_case.run_toy_pipeline --config config/default.yaml  # Real 模式（需要 GPU）
```

---

### 4. `train_smoke_test.py` — 訓練 path 記憶體驗證 [added 2026-05-07]

**對應 SPEC**: §11.2 Module A [impl-updated 2026-05-07]，Spec Amendments #20–#22

**目標**: 在不依賴 Module B / Module C 的情況下，完整跑一次 GRPO `update()`，驗證訓練 path 在目標 GPU (4090, 24GB) 上記憶體不爆且梯度可正常回傳到 LoRA。

**為什麼需要這支腳本**:
- `sanity_check.py` 只走推理 path（generate），**無法**驗證訓練是否撐得住。
- `train.py`（INT-1）尚未實作；要等 Module B / C 整合好再做端到端訓練測試太慢。
- LLM 訓練最容易爆 GPU 的時刻是 `update()` 的 forward + backward，這支腳本就是針對這個瓶頸。

**流程**（每個 iteration）:
```
generate(compute_log_probs=True)  ← 真實生成一批，並算 π_old log probs
   ↓
get_ref_log_probs(...)             ← 用 PEFT disable_adapter 算 π_ref
   ↓
mock rewards (~ N(0,1)) + 同 group 內 z-score 算 advantages
   ↓
update(grpo_batch, micro_batch_size=...)   ← GRPO 一步更新
```

每步印當前 / 峰值 GPU 記憶體（GB）。**不依賴** Module B 的 parser / env，也不依賴 Module C 的 reward；mock rewards 對訓練動態（loss / kl）無參考價值，但完全足以驗證**記憶體不爆 + 梯度有流通**。

**使用方式**:

```bash
# Step 1: 最小規模快速驗證能跑（B=1 prompt × G=2 = 2 條序列）
python -m toy_case.train_smoke_test \
    --batch-size 1 --group-size 2 --num-iterations 1

# Step 2: config 真實規模（讀 config 的 batch=4, group=16, micro=1）
python -m toy_case.train_smoke_test --num-iterations 1

# Step 3: 多跑幾步看記憶體是否穩定（不會隨 iter 數成長）
python -m toy_case.train_smoke_test --num-iterations 3
```

**CLI 參數**（皆為 optional，未指定則讀 config）:
- `--config` — 配置檔路徑（預設 `config/default.yaml`）
- `--batch-size` — 覆寫 `batch_size`
- `--group-size` — 覆寫 `group_size`
- `--micro-batch-size` — 覆寫 `micro_batch_size`（GPU 一次 forward+backward 幾條）
- `--num-iterations` — 跑幾次完整 update step（預設 1）

**輸出**: 印到 stdout，每步看到類似這樣的 log：
```
[mem][model loaded]        allocated=5.32 GB | reserved=5.50 GB | peak=5.45 GB
[mem][after generate]      allocated=8.12 GB | reserved=12.30 GB | peak=11.85 GB
[mem][after ref_log_probs] allocated=8.40 GB | ...
[mem][after update]        allocated=9.20 GB | reserved=14.50 GB | peak=14.20 GB
iter 1 metrics: loss=0.0123, kl=0.0001, mean_reward=0.0234
```

**判斷標準**:
- ✅ **PASS**: 跑完所有 iteration、`peak < 24 GB`、最後印出 `Smoke test PASSED`
- ❌ **FAIL**: 中途 OOM（`torch.cuda.OutOfMemoryError`）或梯度為 NaN

**FAIL 時的 fallback 順序**（從便宜到昂貴）:
1. 確認 `--micro-batch-size 1`（已是 config 預設）
2. 裝 flash-attn：`pip install flash-attn --no-build-isolation`，再省 1-2GB（MA-MEM-7）
3. 降 `max_new_tokens` 2048 → 1024（修 `config/default.yaml`，MA-MEM-9）
4. 改 optimizer 為 `bitsandbytes.optim.PagedAdamW8bit`（MA-MEM-8）

---

## 執行順序建議

```
1. python -m toy_case.train_agent --agent both              # 訓練 agent pool（CPU/GPU 皆可）
2. python -m toy_case.sanity_check --num-levels 20          # 驗證 LLM 推理 path（吃 GPU）
3. python -m toy_case.run_toy_pipeline --use-mock           # Mock 驗證模組間 dataclass 資料流
4. python -m toy_case.train_smoke_test \
       --batch-size 1 --group-size 2 --num-iterations 1     # 最小規模驗證訓練 path 不爆
5. python -m toy_case.train_smoke_test --num-iterations 1   # config 真實規模驗證 (4090 必看)
6. python -m toy_case.run_toy_pipeline                      # Real 模式端到端（需 Module B/C 完成）
```

> 步驟 4-5 通過後才可以放心啟動正式 GRPO 訓練（`train.py` / INT-1）。如果這兩步沒過，正式訓練必爆。

---

## 與主專案的關係

Toy Case 驗證通過後，代表：
- Module A (LLM Policy) 的 `generate()` 輸出格式正確
- Module B (Game Environment) 的輸入輸出 dataclass 格式被確認
- Module C (Reward Evaluation) 的 reward 和 GRPO advantage 計算邏輯可用
- 三個模組可以安心進行正式整合訓練

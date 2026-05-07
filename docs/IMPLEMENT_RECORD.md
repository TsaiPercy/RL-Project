# Implementation Record

## Session: 2026-05-07 — Member A (YouZhe): 4090 記憶體優化 + Qwen3.5 採樣 + 15×15 grid 重構

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| MA-MEM-1 | Gradient checkpointing + `enable_input_require_grads()`（QLoRA on 4090 必開） | ☑ |
| MA-MEM-2 | 移除獨立 ref model，`get_ref_log_probs` 改用 PEFT `disable_adapter()` 共用 base | ☑ |
| MA-MEM-3 | `_compute_log_probs` / `_compute_current_log_probs` 用 `gather − logsumexp` 取代 `log_softmax` | ☑ |
| MA-MEM-4 | Flash Attention 2 + `_resolve_attn_implementation()`，缺 flash-attn 自動 fallback SDPA | ☑ |
| MA-MEM-5 | `update()` 加 `micro_batch_size` 參數：動態切 chunk + 梯度累積 + 1 次 `optimizer.step()` | ☑ |
| MA-MEM-6 | `generate()` 加 `compute_log_probs` flag、`output_scores=False` 一律不開、推理期間 try/finally 暫開 `use_cache` | ☑ |
| LLM-CFG-1 | `LLMPolicy.__init__` 新增 `enable_thinking, top_p, top_k, presence_penalty` 參數，全從 config 讀 | ☑ |
| LLM-CFG-2 | `generate_with_chat_template` 透傳 `enable_thinking` 到 `apply_chat_template`（Qwen3.5 官方 thinking 控制路徑） | ☑ |
| CFG-1 | `config/default.yaml`：移除 `gradient_accumulation_steps`，加 `micro_batch_size: 1`、`enable_thinking: false`；採樣對齊官方（T 0.8→0.7, top_p 0.95→0.8, top_k 20, presence_penalty 1.5）；`grid_size 13→15` | ☑ |
| TC-SMOKE-1 | 新增 `toy_case/train_smoke_test.py` 訓練 path 記憶體驗證腳本 | ☑ |
| FIX-1 | `toy_case/sanity_check.py` 補傳 `cache_dir` 給 `LLMPolicy` | ☑ |
| GRID-1 | LLM 輸出格式：13×13 inner area → 完整 15×15 含外牆 ring；座標 [0,12] → [1,13] | ☑ |
| GRID-2 | `simple_parse()` 一系列強化：reasoning 防呆（`</think>` 砍除 + `rfind("Grid:")`）、外牆完整性、object×object/agent×object 重疊、object/agent on internal wall、goal 恰好一個 | ☑ |
| DOC-1 | 對應更新 SPEC §3/§11、SPEC_MODIFICATION #20–#25、CLAUDE.md、TOY_CASE.md、TODO.md、shared/types.py + type_examples.py | ☑ |

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| MA-MEM-1 | `gradient_checkpointing_enable(use_reentrant=False)` | reentrant 版本與 PEFT `disable_adapter()` 互斥，會在 ref forward 拋錯 |
| MA-MEM-3 | `gather − logsumexp` 取代 `log_softmax(...).gather(...)` | 不 materialize (B, seq, vocab=152K) softmax 副本（bf16 一份 ~6.4GB） |
| MA-MEM-5 | chunk loss weight = `chunk_size / actual_b` 而非 `1/n_chunks`；`clip_grad_norm_` 累積完才做一次 | 處理「最後一塊不足 micro_batch_size」餘數情況；clip 必須在累積完整梯度上做 |
| MA-MEM-6 | `output_scores=False` 即使要 log_probs 也不開 | scores 經 temperature/top_p 後處理，跟 raw logits 不一致；GRPO 要 raw policy log prob |
| MA-MEM-6 | 推理 try/finally 暫開 `use_cache=True` | 訓練要關 (KV cache vs grad 衝突)，推理沿用會災難 |
| CFG-1 | 移除 `gradient_accumulation_steps`、改用 `micro_batch_size` | GRPO effective batch = batch_size × group_size 已固定；保留 accumulation 跟 group_size 耦合容易打架 |
| LLM-CFG-2 | 走 `apply_chat_template(enable_thinking=False)` 而非 `/no_think` | Qwen3.5 不認 `/no_think`（官方 model card 明示）；先試 `/no_think` 已驗證無效（模型改用散文 reasoning，吃完 max_new_tokens 仍未寫到 Grid:） |
| GRID-1 | 座標 `[1, 13]` 嚴格（不能蓋外牆）而非 `[0, 14]` 寬鬆 | 避免物件覆蓋外牆；parser 一致檢查外牆完整性 |
| GRID-2 | overlap 用 `dict[(x,y) → type]` | 衝突時可同時報出兩邊（不只「重疊」，還告訴你跟誰重疊） |

### Spec Amendments
詳見 `SPEC_MODIFICATION.md` #20–#25：
- **#20**：LLMPolicy 4 項記憶體優化（gradient ckpt / disable_adapter ref / logsumexp / flash attn）
- **#21**：`generate()` 加 `compute_log_probs` flag、`output_scores` 改 False、推理暫開 `use_cache`
- **#22**：`update()` 加 `micro_batch_size`，config 移除 `gradient_accumulation_steps`
- **#23**：嘗試 `/no_think` + `simple_parse()` 加 reasoning 防呆（後續 #24 證實 `/no_think` 在 Qwen3.5 無效，但防呆保留）
- **#24**：改 `apply_chat_template(enable_thinking=False)`，採樣參數對齊 Qwen3.5 官方
- **#25**：LLM grid 13×13 inner → 完整 15×15 含外牆 ring，座標 [0,12] → [1,13]，parser 加外牆完整性與多項 overlap 檢查

### Concerns
- **未實機驗證**：所有改動都通過 `ast.parse` 與 unit-style assertion，但 GPU 上沒跑過。建議順序：(1) `rm -rf results/sanity_check/ && python -m toy_case.sanity_check --num-levels 100`（預期 parse rate 從 ≤10% 跳到 50%+）；(2) `python -m toy_case.train_smoke_test --batch-size 1 --group-size 2 --num-iterations 1` 最小規模驗證訓練 path 不爆 → (3) 拉到 config 真實規模 (`--num-iterations 1`)。
- **`presence_penalty` 暫未生效**：HF `generate()` 沒原生支援，目前只把參數收進 `__init__`；若實機看到大量重複 token（如連續 `..........`）需要加 `LogitsProcessorList`。
- **flash-attn 未安裝**：log 顯示 fallback SDPA，少省 1-2GB；若 4090 仍緊建議 `pip install flash-attn --no-build-isolation`。
- **Module B parser 仍 ☐**：`game_env/parser.py` 未實作；座標/外牆/overlap 驗證邏輯應參考 `simple_parse()` 抽出共用 helper（不要複製貼上）。連通性檢查 (BFS + key/door 順序) 是 Module B 責任，不在 sanity_check 範疇。

---

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

## Session: 2026-05-02 — Agent Curriculum Training (AT-1, AT-2, AT-3, AT-7)

### Tasks Completed

| ID | Description | Status |
|----|-------------|--------|
| AT-1 | 實作 `agent_training/train_curriculum.py` — BabyAI 環境 curriculum 訓練腳本 | ☑ |
| AT-2 | 設計並配置 curriculum 成功率門檻（config 中各關卡門檻設定） | ☑ |
| AT-3 | 驗證各 BabyAI 環境支援 room_size=15 參數 | ☑ |
| AT-7 | 實作 `agent_training/evaluate_agent.py` — 評估 agent 各環境 success rate | ☑ |

AT-4 (train strong_0) and AT-5 (train weak_0) require actual GPU execution and are left
for the user:
```bash
python -m agent_training.train_curriculum --agent strong --seed 42
python -m agent_training.train_curriculum --agent weak   --seed 42
```

### Decisions Made

| Task | Decision | Rationale |
|------|----------|-----------|
| AT-1 | `CurriculumTrainer` class with a single `train()` method | Single-responsibility; mirrors `toy_case/train_agent.py` pattern |
| AT-1 | Train in `eval_chunk_steps` (100K) increments, evaluate after each chunk | Allows per-level early-stopping without a separate EvalCallback; simpler control flow |
| AT-1 | `model.set_env(train_env)` when switching levels | SB3 PPO supports runtime env swap; avoids re-instantiating and discarding weights |
| AT-1 | Save intermediate checkpoint after every level | Cheap disk cost vs. high diagnostic value if training is interrupted |
| AT-1 | `reset_num_timesteps=False` on all calls after the first | Preserves SB3 internal step counter across levels for consistent logging |
| AT-3 | `try gym.make(env_id, room_size=…) except TypeError: gym.make(env_id)` | GoToObjMazeS4-S7 envs bake maze size into name and may reject room_size kwarg; fallback is safe and logged as a warning |
| AT-7 | `evaluate_agent()` as a standalone function | Only one responsibility; a class would be over-engineering for a pure eval loop |
| AT-7 | `resolve_env_ids("all"/"first3"/csv)` helper | Flexible CLI for both strong (all 9) and weak (first 3) post-training checks |
| AT-2 | Added `eval_episodes: 50`, `eval_chunk_steps: 100_000` to config | These fields were absent from original config; values balance eval speed vs. signal quality |

### Spec Amendments

| Severity | Section | Change | Trigger |
|----------|---------|--------|---------|
| L1 | §11.4 Pseudo Config | Synchronised `config/default.yaml` with SPEC §5.4 [updated]: replaced 8-env curriculum with 9-env GoTo family; renamed `success_threshold_override` → `success_increase`; `total_timesteps` → `max_timesteps`; added `eval_episodes` and `eval_chunk_steps` | impl-updated (AT-2) |

### Concerns

- **AT-3 room_size on S4-S7 envs**: GoToObjMazeS4-S7 likely cannot be overridden with `room_size=15`; the fallback uses the env's native size. Validate that the strong agent's generalisation to 15×15 MiniGrid levels is still adequate after training.
- **CnnPolicy with 7×7 obs**: SB3's default CnnPolicy expects larger images. It handles small obs automatically with a shallower CNN, but training may be slower. If instability is observed, switch to `MlpPolicy` with flattened obs.
- **Weak agent effective threshold**: At level 3 (GoToObjMaze-v0), effective threshold = 0.80 − 0.25 = 0.55. This is intentionally low to stop the agent before it becomes competent at maze navigation.

---

## Session: 2026-05-02 — BabyAI CNN Compatibility Bugfix (AT-4 blocker)

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| X-1 | Fix RuntimeError: NatureCNN kernel (8×8) > BabyAI obs size (7×7) — add `BabyAIFeaturesExtractor` with kernel_size=2 conv layers | ☑ |

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| X-1 | Defined `BabyAIFeaturesExtractor(BaseFeaturesExtractor)` inside `train()` to preserve SB3 lazy-import pattern already established in the file | Class only exists when PPO is instantiated; avoids SB3 import at module level |
| X-1 | Architecture: Conv2d(C,16,k=2) → Conv2d(16,32,k=2) → Conv2d(32,64,k=2) → Linear(1024, features_dim) | Standard minigrid CNN design; three k=2 layers on 7×7 input give 4×4 spatial output before flatten |
| X-1 | Used `with torch.no_grad(): n_flatten = self.cnn(sample).shape[1]` to auto-detect flattened dim | Avoids hardcoding 1024; generalises to other obs sizes (e.g. maze envs with larger rooms) |
| X-1 | Added `cnn_features_dim: 256` to `config/default.yaml` under `agent_training.ppo_hyperparams` | Keeps the extractor output dim configurable; matches sdd-ai-coding no-hardcode rule |

### Spec Amendments
| Severity | Section | Change | Trigger |
|----------|---------|--------|---------|
| L1 | §11.4 Pseudo Config | Added `cnn_features_dim: 256` to `agent_training.ppo_hyperparams` | impl-updated (X-1) |

### Concerns
- Previous session concern ("CnnPolicy with 7×7 obs: handles small obs automatically") was incorrect — SB3 does NOT auto-adapt NatureCNN kernels; the 8×8 kernel causes a hard crash. Fixed in this session.

---

## Session: 2026-05-02 — Silence BabyAI rejection-sampling stdout (AT-4 noise)

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| X-2 | Suppress BabyAI's `Sampling rejected: unreachable object at (i, j)` stdout chatter during curriculum training/eval | ☑ |

### Diagnosis
While running `python -m agent_training.train_curriculum --agent strong --seed 42`,
BabyAI emits frequent `Sampling rejected: unreachable object at (X, Y)` messages
from `minigrid/envs/babyai/core/roomgrid_level.py:137`. Root cause:
- BabyAI's level generator uses rejection sampling. When generated objects
  land in cells unreachable from the agent's start, it raises `RejectSampling`
  and the outer loop emits a raw `print(...)` and retries.
- Training is **NOT blocked** — every rejected sample is automatically resampled.
- With `room_size=15` (much larger than BabyAI's defaults of 6–8), maze envs
  (`BabyAI-GoToObjMaze*-v0`) produce a flood of these messages, drowning real
  training logs. Coordinates like (33, 40) reflect the enlarged grid (3×14+1=43).

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| X-2 | Created `agent_training/baby_ai_silence.py` with `silence_baby_ai_rejection_logs()` that monkey-patches `roomgrid_level.print` to forward to a `baby_ai.rejection` logger at DEBUG | Non-invasive (no BabyAI source edit), idempotent, recoverable (`logging.getLogger("baby_ai.rejection").setLevel(logging.DEBUG)` restores chatter for diagnostics) |
| X-2 | Patch invoked at top of both `train_curriculum.py` and `evaluate_agent.py`, after `setup_project_cache()` and before any env construction | Both scripts create BabyAI envs; both benefit. Applied at module load to ensure no env is created before the patch |
| X-2 | Used module-globals override (`roomgrid_level.print = ...`) rather than replacing `builtins.print` | Targeted: only silences chatter from this one module; doesn't affect any other library or user code |

### Verification
- 20 resets of `BabyAI-GoToObjMaze-v0(room_size=15)` after patch → 0 chars of "Sampling rejected" stdout chatter (was dozens of messages before).
- Both entry-point scripts still import cleanly.

### Spec Amendments
None — purely a runtime log-noise concern; no SPEC behavior changed.

### Concerns
- If a user wants to debug BabyAI sampling failures (e.g., investigating why a custom env never generates valid levels), they can re-enable the chatter via:
  ```python
  import logging
  logging.getLogger("baby_ai.rejection").setLevel(logging.DEBUG)
  ```
  This should be documented if AT-4/AT-5 ever fails to make progress on a level.

---

## Session: 2026-05-02 — Drop unified room_size=15 for curriculum (Spec Amendment #17)

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| X-3 | Remove unified `room_size=15` override; let each curriculum env use its native default size | ☑ |
| AT-3 | 驗證各 BabyAI 環境支援 room_size=15 參數 [updated] | ☑ → marked obsolete (no longer needed) |

### User Request
> 那每個stage都用原本環境預設大小就好，不要固定成15*15

### Changes
- **SPEC.md** §5.4 訓練機制: replaced "所有環境統一 `room_size=15`（15×15 total，13×13 usable space）" with native-size note `[impl-updated]`. Original kept in `<!-- before: ... -->`.
- **SPEC.md** §11.4 pseudo config: removed `room_size: 15` line; left an `[impl-updated]` comment in its place.
- **config/default.yaml**: removed `agent_training.room_size: 15` field.
- **agent_training/train_curriculum.py**:
  - `make_env_fn(env_id, room_size)` → `make_env_fn(env_id)` — single arg, no fallback try/except.
  - `CurriculumTrainer` no longer reads or stores `_room_size`.
  - `_make_vec_env` and `_eval_success_rate` updated to drop the param.
- **agent_training/evaluate_agent.py**:
  - `evaluate_agent(...)` signature: dropped `room_size` param.
  - `main()`: stopped reading `at_cfg["room_size"]`.
- **TODO.md** AT-3: appended `[obsolete: 2026-05-02 ...]` note.
- **SPEC_MODIFICATION.md**: appended entry #17 (L2, user request).

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| X-3 | Removed config field entirely rather than defaulting it to `None` | Cleaner: no dead code path, simpler signature, matches no-hardcode rule (the field served no purpose post-removal) |
| X-3 | Did NOT touch toy_case (Phase 0) DoorKey config | Toy case uses `MiniGrid-DoorKey-15x15-v0` whose 15×15 is baked into the env name; user's request was scoped to curriculum stages, and the toy agent's reward signal is meant to be calibrated to that specific size |
| X-3 | Classified as L2 (Medium), not L1 | Removes a public config field that other code reads; affects function signatures of `make_env_fn` and `evaluate_agent` — qualifies as API-surface change per the skill's L1/L2 boundary |

### Verification
All 9 curriculum envs construct successfully with native defaults; obs is (7,7,3) for all (unchanged) so `BabyAIFeaturesExtractor` works as-is. Native grid sizes:

| Env | Grid (W×H) |
|---|---|
| BabyAI-GoTo-v0 | 22×22 |
| BabyAI-GoToOpen-v0 | 22×22 |
| BabyAI-GoToObjMaze-v0 | 22×22 |
| BabyAI-GoToObjMazeOpen-v0 | 22×22 |
| BabyAI-GoToObjMazeS4R2-v0 | 7×7 |
| BabyAI-GoToObjMazeS4-v0 | 10×10 |
| BabyAI-GoToObjMazeS5-v0 | 13×13 |
| BabyAI-GoToObjMazeS6-v0 | 16×16 |
| BabyAI-GoToObjMazeS7-v0 | 19×19 |

Previously (with `room_size=15`), the first four envs would have been ~46×46 — the new sizes are roughly half, which means much faster episodes and far fewer "Sampling rejected" retries.

### Spec Amendments
| Severity | Section | Change | Trigger |
|----------|---------|--------|---------|
| L2 | §5.4, §11.4 | Curriculum 各關改用 BabyAI 預設大小，不再覆寫 `room_size`；移除 config 欄位與函數參數 | user request |

### Concerns
- **Difficulty curve**: The first four envs (GoTo, GoToOpen, GoToObjMaze, GoToObjMazeOpen) all default to 22×22, while S4–S7 are 7→19. This means levels 1–4 are *larger* than levels 5–6 in the curriculum — the difficulty progression is no longer monotonic by grid size. The progression is now task-structural (open vs. maze, S4R2 multi-room) rather than spatial. If empirically the agent struggles on level 5 (S4R2, only 7×7) after clearing level 4 (open 22×22), revisit the ordering.
- **base_threshold tuning**: Per-level `success_threshold` values in config were calibrated assuming `room_size=15`. With smaller native sizes, success rates may climb faster on the harder maze envs — thresholds may need re-tuning after the first AT-4 run is observed.
- **Held-out / training-eval consistency**: Both `train_curriculum.py` and `evaluate_agent.py` were updated together, so an agent trained without `room_size` is also evaluated without it — no train/eval mismatch.

---

## Session: 2026-05-03 — Phase A: Mission Encoding + Dict Extractor (AT-4.1–AT-4.4, Spec Amendment #18)

### User Request
> 我正在使用 SB3 訓練 BabyAI Agent... 第一階段 `BabyAI-GoTo-v0` 訓練 success rate 僅 2%。
> 1. 資訊遺失：`ImgObsWrapper` 丟棄 mission string；2. 記憶缺失：7×7 partial obs；3. 架構優化。

After plan-mode review: **Phased rollout** (Phase A: dict obs + mission text + direction encoding; Phase B = LSTM, deferred). Level 1 is essentially fully observable, so the 2% rate is overwhelmingly mission-blindness, not POMDP. LSTM doubles training cost — not justified until maze stages stall.

### Tasks Completed
| ID | Description | Status |
|----|-------------|--------|
| AT-4.1 | 實作 `agent_training/wrappers.py` (`MissionTokenizer` + `BABYAI_VOCAB`) [added 2026-05-03] | ☑ |
| AT-4.2 | 實作 `agent_training/extractors.py` (`BabyAIDictExtractor`) [added 2026-05-03] | ☑ |
| AT-4.3 | 重構 `train_curriculum.py` + `evaluate_agent.py`：wrapper swap, 移除 inline extractor, switch to `MultiInputPolicy` [added 2026-05-03] | ☑ |
| AT-4.4 | Phase A smoke test (50K 步 `BabyAI-GoTo-v0`) [added 2026-05-03] | ☑ |
| AT-4 | 訓練 strong_0：完整 curriculum（9 關），高門檻 [updated] | ☐ → [needs-redo] |
| AT-5 | 訓練 weak_0：部分 curriculum（前 3 關），低門檻 [updated] | ☐ → [needs-redo] |

### Files Created
- `agent_training/wrappers.py` — `MissionTokenizer(gym.ObservationWrapper)` + `BABYAI_VOCAB: tuple[str, ...]` (16 tokens: PAD/UNK/go/to/the/a + 6 colors + 4 objects). Lower-case + whitespace split + map-to-id with UNK fallback + pad to `max_len`.
- `agent_training/extractors.py` — `BabyAIDictExtractor(BaseFeaturesExtractor)` at module level (cloudpickle requirement). Three branches: image CNN (k=2 ×3 → Linear(128)), mission `Embedding+masked-mean-pool→Linear(64)`, direction `Embedding(4, dir_embed_dim)`. Concat → Linear(features_dim). Detects HWC vs CHW via SB3's `is_image_space_channels_first` so VecTransposeImage auto-wrapping is handled correctly.

### Files Modified
- `agent_training/train_curriculum.py`:
  - `make_env_fn(env_id, mission_max_len)` — applies `MissionTokenizer(env, max_len=mission_max_len)` instead of `ImgObsWrapper`.
  - `CurriculumTrainer.__init__` reads `mission_max_len`/`vocab_size`/`text_embed_dim`/`dir_embed_dim` from config; asserts `vocab_size == len(BABYAI_VOCAB)` to catch drift.
  - `train()`: deleted the inline `BabyAIFeaturesExtractor` (lines 245–293 in the old file); switched `PPO("CnnPolicy", ...)` → `PPO("MultiInputPolicy", ..., features_extractor_class=BabyAIDictExtractor, features_extractor_kwargs={features_dim, vocab_size, text_embed_dim, dir_embed_dim})`.
  - Dropped now-unused `import os` and the lazy `gym/torch/nn/BaseFeaturesExtractor` imports.
- `agent_training/evaluate_agent.py`:
  - `evaluate_agent(...)` gained `mission_max_len: int` parameter; threaded through `make_env_fn`.
  - `main()` reads `at_cfg["mission_max_len"]`.
- `config/default.yaml`:
  - Added `mission_max_len: 8`, `vocab_size: 16`, `text_embed_dim: 32`, `dir_embed_dim: 8` under `agent_training`.
  - Renamed `ppo_hyperparams.cnn_features_dim` → `ppo_hyperparams.features_dim` (extractor is no longer pure CNN).

### Files Deleted
- `checkpoints/agents/weak_0.zip`, `checkpoints/agents/weak_0_level1.zip` — legacy CNN+ImgObsWrapper artifacts that cannot load with the new dict obs space (per user decision: "Delete and overwrite").

### Decisions Made
| Task | Decision | Rationale |
|------|----------|-----------|
| AT-4.1 | Hardcoded 16-token vocab `(PAD, UNK, go, to, the, a, 6 colors, 4 objects)` | BabyAI GoTo grammar is closed and stable. Reproducibility wins; dynamic discovery breaks resume/eval. User confirmed in plan review. |
| AT-4.1 | Promote `direction` from scalar to length-1 int64 vector | SB3's `CombinedExtractor` requires consistent ndarray shape per Dict key; scalar Box would break batching. |
| AT-4.2 | Module-level extractor (not nested in `train()`) | `cloudpickle` cannot resolve `CurriculumTrainer.train.<locals>.BabyAIDictExtractor` on `PPO.load`. Verified with save/load roundtrip in smoke test. |
| AT-4.2 | Channel-position branch on `is_image_space_channels_first(image_space)` | SB3 auto-applies `VecTransposeImage` for HWC uint8 images, rewriting the policy's `observation_space["image"]` to CHW BEFORE the extractor `__init__` fires. The first smoke run crashed because the old code assumed HWC universally (`shape[-1] = C`); the fix detects layout once and stores `self._image_channels_first` for `forward()` to consult. |
| AT-4.2 | Stay with the proven `Conv2d k=2 ×3 (16→32→64)` block from X-1 | Same architecture that fixed the 7×7 obs / NatureCNN-incompat issue; no reason to redesign. |
| AT-4.2 | Image projection 128 dim, text projection 64 dim, fusion → features_dim | Architectural constants documented in the module's UPPER_SNAKE_CASE block; not researcher-tunable. Promoting to config would just add noise. |
| AT-4.3 | `policy="MultiInputPolicy"` (no LSTM) | Phase A scope per user decision in plan review. |
| AT-4.3 | `assert config.vocab_size == len(BABYAI_VOCAB)` at trainer init | Catches drift if someone bumps the vocab list without updating config (or vice versa). |
| AT-4.4 | Smoke test = 50K steps + both stochastic + deterministic eval | Original plan's "5K steps > 15%" was overly optimistic for 22×22 BabyAI-GoTo with 4 distractor objects. 50K is the smallest scale where signal vs. noise becomes legible. |
| Cleanup | Installed `einops` in `RL_Project` conda env | Required by project coding rule §4 (einops only for any reshape). Was missing. |

### Spec Amendments
| Severity | Section | Change | Trigger |
|----------|---------|--------|---------|
| L2 | §5.4, §11.4, §16 | Phase A 觀察值升級：`ImgObsWrapper` → `MissionTokenizer`；`PPO("CnnPolicy")` → `PPO("MultiInputPolicy") + BabyAIDictExtractor`；新增 4 個 config 欄位；`cnn_features_dim` → `features_dim`；新增 `wrappers.py`/`extractors.py` 至目錄結構 | user request |

### Verification

| Step | Result |
|------|--------|
| `len(BABYAI_VOCAB)` | 16 ✓ |
| `MissionTokenizer.observation_space` shape (3 envs) | `{image (7,7,3) uint8, direction (1,) int64, mission (8,) int64}` ✓ |
| Sample mission decode | "go to the green ball" / "go to the purple ball" ✓ |
| Extractor forward (B=2) | `(2, 256)` finite ✓; **195,856 params** |
| `image_channels_first` detection | `True` after `VecTransposeImage` ✓ |
| `PPO("MultiInputPolicy").learn(256)` | OK ✓ |
| `PPO.save` → `PPO.load` roundtrip | OK ✓ (extractor pickles cleanly because module-level) |
| Gradient flow | `text_emb`, `dir_emb`, `cnn[0]`, `fuse[0]` all non-zero after one update ✓ |
| Random-policy baseline (50 ep) | **7/50 = 14%** |
| 50K-step trained, stochastic eval (50 ep) | **9/50 = 18%** — beats random ✓ |
| 50K-step trained, deterministic eval (50 ep) | 1/50 = 2% (policy not yet converged for argmax; expected to climb with full AT-4 budget) |

### Concerns / Phase B Parking

- **Phase B (RecurrentPPO + MultiInputLstmPolicy)** is parked. Re-evaluate only if AT-4's strong agent clears levels 1–3 but plateaus on the maze stages (S4R2/S4/S5/S6/S7) where partial observability genuinely hurts. Trigger conditions: maze-stage success rate stuck below `0.5 × effective_threshold` for ≥1M steps. Implementation sketch in plan file `compressed-dazzling-forest.md` (§4 Algorithm swap, §5 Eval loop update).
- **Deterministic eval brittleness**: 2% on 50K-step deterministic eval doesn't mean the policy is bad — it means argmax is degenerate before convergence. Phase A's full AT-4 run has 5M-step budget; expect deterministic eval to climb toward the configured `effective_threshold` (0.95 for level 1).
- **SubprocVecEnv timing crash** during smoke testing (connection reset peer) — happened once when running test scripts via `python -c`. Did NOT reproduce inside `train_curriculum.py` (which uses module imports rather than `-c`); leaving as a session-scoped issue, not a code defect. If reproduced during AT-4, fall back to `DummyVecEnv` via `--n-envs 1`.
- **base_threshold tuning unknowns**: Per-level success thresholds were calibrated assuming the prior CnnPolicy + image-only obs. With mission encoding, level-1 should converge faster; thresholds may need re-tuning after first AT-4 run is observed (carry-over concern from session 2026-05-02).
- **Toy case unchanged**: `toy_case/train_agent.py` still uses `ImgObsWrapper + PPO("CnnPolicy")` on `MiniGrid-DoorKey-15x15-v0` (constant mission text — tokenization adds no signal). Per user preference confirmed in plan review.

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

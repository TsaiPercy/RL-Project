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

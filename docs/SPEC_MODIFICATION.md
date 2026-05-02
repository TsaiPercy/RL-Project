# Spec Modification Log

| # | Date | Section | Change Summary | Trigger | Type × Scope |
|---|------|---------|----------------|---------|---------------|
| 1 | 2026-05-01 | §5.4 Agent Pool | Added §5.4.1 Toy Case Agent Pool (SB3 PPO on DoorKeyEnv) | user request | Add × Cascading |
| 2 | 2026-05-01 | §5.5 階段計畫 | Inserted Phase 0 (Toy Case) before Phase 1; updated ordering description | user request | Add × Cascading |
| 3 | 2026-05-01 | §9 Datasets | Added Toy Case Agent Pool subsection under Agent Pool | user request | Add × Cascading |
| 4 | 2026-05-01 | §10 Experiment Design | Added Experiment T: Toy Case Pipeline Smoke Test; updated Experiment Ordering | user request | Add × Cascading |
| 5 | 2026-05-01 | §11.4 Pseudo Config | Added toy_case config block (train_env, agent steps, PPO hyperparams) | user request | Add × Cascading |
| 6 | 2026-05-01 | §12 Experiment Impl | Added Experiment T implementation architecture (4-step flow) | user request | Add × Cascading |
| 7 | 2026-05-01 | §16 Directory Structure | Added toy_case/ directory (train_agent.py, run_toy_pipeline.py) | user request | Add × Cascading |
| 8 | 2026-05-01 | §11.2 Module A | `get_ref_log_probs` 簽名新增 `prompt_ids` 參數 | impl-updated (MA-1) | L1 |
| 9 | 2026-05-01 | §16 Directory Structure, §11.4 Pseudo Config | 新增 `shared/env_setup.py`、`cache/`、`setup_env.sh` 至目錄；新增 Entry Point Convention（所有 entry point 頂端需呼叫 `setup_project_cache()`）；§11.4 加入 `cache_dir` 配置項 | user request | Add × Localized |
| 10 | 2026-05-01 | §5.4 Agent Pool | Agent 從 BabyAI pretrained 改為自行訓練 curriculum（BabyAI 環境 room_size=15，8 關由簡到難），強弱差異透過成功率門檻與訓練環境數量製造 | user request | Modify × Cascading |
| 11 | 2026-05-01 | §5.4.1, §9 Toy Case | DoorKeyEnv size=13 → room_size=15（統一為 15×15 total, 13×13 usable） | user request | Modify × Cascading |
| 12 | 2026-05-01 | §11.4 Pseudo Config | 新增 agent_training config block（curriculum 環境列表、成功率門檻、strong/weak 配置） | user request | Add × Cascading |
| 13 | 2026-05-01 | §16 Directory Structure | 新增 `agent_training/` 目錄（train_curriculum.py, evaluate_agent.py） | user request | Add × Localized |
| 14 | 2026-05-02 | §5.4, §11.4 | 將 8-task 多任務 curriculum 換為 9-env GoTo 家族單任務 curriculum；新增 per-env base threshold（0.90→0.60）；將 `success_threshold_override` 改為 `success_increase`（加法 delta）；將 `total_timesteps` 改為 `max_timesteps`；`curriculum_levels` 8→9 | user request | Modify × Cascading |
| 15 | 2026-05-02 | §11.2 Module B | `batch_evaluate()` 新增 optional `agent_ids` keyword argument，讓呼叫方指定使用哪組 agent pool（Per SPEC §8 quick/full mode） | impl-updated (MC-4) | L2 |

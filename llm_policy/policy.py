"""LLMPolicy — QLoRA 4-bit 載入 Qwen3.5-9B + generate + GRPO update。

Per SPEC §11 Module A:
  - generate(prompts) → GenerationOutput
  - get_ref_log_probs(token_ids) → Tensor
  - update(grpo_batch) → dict

Per SPEC §14: QLoRA 4-bit, LoRA rank=64, alpha=128.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from shared.types import GenerationOutput, GRPOBatch

logger = logging.getLogger(__name__)


class LLMPolicy:
    """管理 Qwen 模型的 QLoRA 載入、推理、GRPO 更新。

    Attributes:
        model: 量化後的 causal LM（含 LoRA adapter）。
        tokenizer: 對應的 tokenizer。
        ref_model: 已棄用，永遠為 None。Reference policy 改用
            `model.disable_adapter()` 取得，不再載入第二份權重。保留
            attribute 是為了 backward compat。
        device: 模型所在裝置。
    """

    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_target_modules: Optional[list[str]] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        enable_thinking: bool = False,
        device: str = "auto",
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
        gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        """初始化 LLMPolicy：載入模型 + LoRA adapter。

        Args:
            model_name: HuggingFace model ID (e.g. "Qwen/Qwen3.5-9B").
            quantization: 量化方式 ("4bit" or "none").
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_target_modules: LoRA target modules.
            max_new_tokens: 生成最大 token 數.
            temperature: Sampling temperature（Qwen3.5 non-thinking 建議 0.7）.
            top_p: Nucleus sampling 機率累積閾值（Qwen3.5 non-thinking 建議 0.8）.
            top_k: Top-k 採樣（Qwen3.5 non-thinking 建議 20）.
            presence_penalty: 已出現 token 的懲罰，抑制重複（Qwen3.5 建議 1.5）.
            enable_thinking: Qwen3.5 chat template 旗標；False 直接給最終答，
                不寫 reasoning。Qwen3.5 不支援 /no_think directive，
                必須走 apply_chat_template(enable_thinking=...) 這條路.
            device: 目標裝置 ("auto", "cuda", "cpu").
            load_in_4bit: 是否使用 4-bit 量化載入.
            cache_dir: HuggingFace cache directory. None = HF default.
            gradient_checkpointing: 是否啟用 activation checkpointing（QLoRA
                訓練 9B 在 4090 必開；節省 ~70% activation memory，代價是
                backward 多一次 forward，~30% 慢）.
            use_flash_attention: 是否使用 Flash Attention 2（attention 從
                O(seq^2) memory 降到 O(seq)）.若未安裝 flash-attn 會自動
                fallback 到 SDPA.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.cache_dir = cache_dir
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        self.model, self.tokenizer = self._load_model(
            model_name, quantization, load_in_4bit, device,
        )
        self.model = self._attach_lora(self.model)

        if gradient_checkpointing:
            # 必須在 attach LoRA 之後做。enable_input_require_grads() 讓
            # embedding 輸出 requires_grad=True，否則 4-bit base 凍結會
            # 切斷 autograd graph，LoRA grad 全變 None。
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.model.enable_input_require_grads()
            logger.info("[LLMPolicy] Gradient checkpointing 已啟用")

        # Reference model 用 PEFT 的 disable_adapter() 共用 base，
        # 不再載入第二份 9B 權重。
        self.ref_model = None
        self.device = next(self.model.parameters()).device

        logger.info(
            "[LLMPolicy] 初始化完成 | model=%s, device=%s, "
            "lora_rank=%d, lora_alpha=%d, max_new_tokens=%d, "
            "T=%.2f, top_p=%.2f, top_k=%d, presence_penalty=%.2f, "
            "enable_thinking=%s, grad_ckpt=%s, flash_attn=%s",
            model_name, self.device, lora_rank, lora_alpha, max_new_tokens,
            temperature, top_p, top_k, presence_penalty,
            enable_thinking, gradient_checkpointing, use_flash_attention,
        )

    def _resolve_attn_implementation(self) -> str:
        """選擇 attention 實作，優先 flash_attention_2，缺套件則退回 sdpa。

        Returns:
            "flash_attention_2" 或 "sdpa".
        """
        if not self.use_flash_attention:
            return "sdpa"
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            logger.warning(
                "[LLMPolicy] flash-attn 未安裝；fallback 到 sdpa。"
                "若要啟用 flash attention 2: pip install flash-attn --no-build-isolation",
            )
            return "sdpa"

    def _load_model(
        self,
        model_name: str,
        quantization: str,
        load_in_4bit: bool,
        device: str,
    ) -> tuple:
        """載入量化模型與 tokenizer。

        Returns:
            (model, tokenizer) tuple.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=self.cache_dir,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        attn_impl = self._resolve_attn_implementation()

        if quantization == "4bit" and load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
                attn_implementation=attn_impl,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
                attn_implementation=attn_impl,
            )

        model.config.use_cache = False
        logger.info(
            "[LLMPolicy] 模型載入完成: %s (quantization=%s, attn=%s)",
            model_name, quantization, attn_impl,
        )
        return model, tokenizer

    def _attach_lora(self, model) -> object:
        """為模型附加 LoRA adapter。

        Returns:
            附加 LoRA 後的模型。
        """
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=self.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "[LLMPolicy] LoRA 附加完成 | trainable=%d (%.2f%%)",
            trainable_params, trainable_params / total_params * 100,
        )
        return model

    def generate(
        self,
        prompts: list[str],
        compute_log_probs: bool = True,
    ) -> GenerationOutput:
        """使用 LLM 生成關卡描述。

        Args:
            prompts: 一批 prompt 字串。
            compute_log_probs: 是否計算每個 token 的 log prob。
                訓練時必須 True（GRPO 需要 π_old）；純推理（如 sanity check、
                evaluation）建議設 False — 可省一次全序列 forward + 一份
                (B, seq, vocab=152K) 的 logits 暫態（bf16 約 6.6GB at B=10）。
                False 時，回傳的 `GenerationOutput.log_probs` 是 zeros tensor。

        Returns:
            GenerationOutput 包含 texts, log_probs, token_ids。
        """
        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        # generate 期間暫時打開 KV cache（_load_model 為訓練設成 False，
        # 但推理沒有 KV cache 會讓每步重算整段 attention，記憶體與時間都爆）。
        prev_use_cache = self.model.config.use_cache
        self.model.config.use_cache = True

        try:
            with torch.no_grad():
                # Qwen3.5 non-thinking 官方建議採樣參數，皆從 self 讀（從
                # config 一路傳進來），不再 hardcode。presence_penalty 透過
                # generate() 的 `repetition_penalty` 不同——HF generate 用
                # `repetition_penalty` 是乘法懲罰；presence_penalty 是加法
                # 懲罰。HF generate 沒有 `presence_penalty` 參數，所以這裡
                # 用 `repetition_penalty=1.0`（中性）並把 presence_penalty
                # 留到 LogitsProcessor 處理（Optional，待加）。
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    # output_scores 會把每步 logits 都留下來（B × T × vocab，
                    # bf16 ≈ 6GB at B=10, T=2048）。只在需要 log_probs 時才開。
                    output_scores=False,
                )
        finally:
            self.model.config.use_cache = prev_use_cache

        generated_ids = outputs.sequences[:, input_length:]  # (batch, gen_len)
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if compute_log_probs:
            log_probs = self._compute_log_probs(
                inputs["input_ids"], generated_ids,
            )  # (batch, gen_len)
        else:
            log_probs = torch.zeros_like(generated_ids, dtype=torch.float32)

        logger.info(
            "[LLMPolicy] generate 完成 | batch_size=%d, avg_gen_len=%.1f, "
            "compute_log_probs=%s",
            len(prompts), generated_ids.shape[1], compute_log_probs,
        )

        return GenerationOutput(
            texts=texts,
            log_probs=log_probs,
            token_ids=generated_ids,
            prompt_ids=inputs["input_ids"],
        )

    @staticmethod
    def _gather_token_log_probs(logits: Tensor, token_ids: Tensor) -> Tensor:
        """以 logsumexp 取代 log_softmax 計算 selected-token log prob。

        數學等價 `log_softmax(logits).gather(token_ids)`，但**不**
        materialize `(batch, seq, vocab)` 的 log_softmax 副本（vocab 152K
        × bf16 一份就 6GB+）。改成先 gather logit、再扣 logsumexp，
        peak memory 直接砍掉一份 vocab-shape tensor。

        Args:
            logits: shape (batch, seq, vocab).
            token_ids: shape (batch, seq) — 要取 log prob 的 token ids.

        Returns:
            Log probs, shape (batch, seq).
        """
        gathered = logits.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(logits, dim=-1)
        return gathered - log_z

    def _compute_log_probs(
        self,
        input_ids: Tensor,
        generated_ids: Tensor,
    ) -> Tensor:
        """計算生成 token 的 log probabilities。

        Args:
            input_ids: 原始 prompt token ids, shape (batch, prompt_len).
            generated_ids: 生成的 token ids, shape (batch, gen_len).

        Returns:
            Log probs tensor, shape (batch, gen_len).
        """
        full_ids = torch.cat([input_ids, generated_ids], dim=1)  # (batch, total_len)

        with torch.no_grad():
            outputs = self.model(input_ids=full_ids)
            logits = outputs.logits  # (batch, total_len, vocab_size)

        prompt_len = input_ids.shape[1]
        gen_logits = logits[:, prompt_len - 1:-1, :]  # (batch, gen_len, vocab_size)
        return self._gather_token_log_probs(gen_logits, generated_ids)

    def get_ref_log_probs(self, token_ids: Tensor, prompt_ids: Tensor) -> Tensor:
        """使用 reference policy 計算 log probs（用於 KL penalty）。

        Reference 是「沒有 LoRA adapter 的 base model」，透過 PEFT 的
        `disable_adapter()` context manager 暫時關掉 adapter 後做 forward，
        不另外載入第二份 9B 權重。

        Args:
            token_ids: 生成的 token ids, shape (batch, gen_len).
            prompt_ids: Prompt token ids, shape (batch, prompt_len).

        Returns:
            Reference log probs, shape (batch, gen_len).
        """
        full_ids = torch.cat([prompt_ids, token_ids], dim=1)  # (batch, total_len)

        with torch.no_grad(), self.model.disable_adapter():
            outputs = self.model(input_ids=full_ids)
            logits = outputs.logits  # (batch, total_len, vocab_size)

        prompt_len = prompt_ids.shape[1]
        gen_logits = logits[:, prompt_len - 1:-1, :]  # (batch, gen_len, vocab_size)
        ref_log_probs = self._gather_token_log_probs(gen_logits, token_ids)

        logger.debug(
            "[LLMPolicy] get_ref_log_probs 完成 | shape=%s", ref_log_probs.shape,
        )
        return ref_log_probs

    @staticmethod
    def _slice_grpo_batch(batch: GRPOBatch, start: int, end: int) -> GRPOBatch:
        """沿 batch 維度切片 GRPOBatch；所有欄位的 dim0 是 batch。"""
        return GRPOBatch(
            token_ids=batch.token_ids[start:end],
            prompt_ids=batch.prompt_ids[start:end],
            log_probs=batch.log_probs[start:end],
            ref_log_probs=batch.ref_log_probs[start:end],
            rewards=batch.rewards[start:end],
            advantages=batch.advantages[start:end],
        )

    def _compute_chunk_loss(
        self, chunk: GRPOBatch,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """對一個 micro-batch 算 (loss, policy_loss, kl_div)，皆為 mean over chunk。

        Args:
            chunk: 已切片的 GRPOBatch，shape dim0 = micro_batch_size。

        Returns:
            (loss, policy_loss, kl_div) — 三個 scalar tensor。
        """
        current_log_probs = self._compute_current_log_probs(chunk)  # (m, gen_len)

        ratio = torch.exp(current_log_probs - chunk.log_probs)
        adv = chunk.advantages.unsqueeze(-1)  # (m, 1)
        policy_loss = -(adv * ratio).mean()

        log_ratio_ref_current = chunk.ref_log_probs - current_log_probs
        kl_div = (
            torch.exp(log_ratio_ref_current) - log_ratio_ref_current - 1.0
        ).mean()

        loss = policy_loss + self.kl_coeff * kl_div
        return loss, policy_loss, kl_div

    def update(
        self,
        grpo_batch: GRPOBatch,
        micro_batch_size: Optional[int] = None,
    ) -> dict:
        """執行一步 GRPO 更新。

        Per SPEC §5.3:
          L = -E[advantage * log π(y|x)] + kl_coeff * KL(π || π_ref)

        支援梯度累積：當 `micro_batch_size` 設定時，把 grpo_batch 沿
        dim0 切成 ceil(B / micro_batch_size) 個 chunk，逐個 forward+
        backward 並把梯度累加（loss 乘 chunk_weight），最後做一次
        optimizer.step()。數學上等同於一次處理 B 條（mean over B），
        但 peak memory 取決於 micro_batch_size 而非 B。

        Args:
            grpo_batch: 包含 token_ids, log_probs, ref_log_probs,
                rewards, advantages 的 batch (B = batch_size × group_size)。
            micro_batch_size: GPU 一次 forward+backward 處理幾條序列。
                None 或 >= B 時不切，與舊行為相同（向後相容）。建議
                4090 + Qwen 9B 設 1。

        Returns:
            dict 包含 loss, kl, mean_reward 等訓練指標（皆為 batch mean）。
        """
        self.model.train()

        if not hasattr(self, "optimizer"):
            raise RuntimeError(
                "Optimizer 未初始化。請先呼叫 setup_optimizer()。"
            )

        actual_b = grpo_batch.token_ids.shape[0]
        # micro=None 或 >= B 都走「一次處理」路徑
        no_microbatch = micro_batch_size is None or micro_batch_size >= actual_b

        self.optimizer.zero_grad()

        if no_microbatch:
            loss, policy_loss, kl_div = self._compute_chunk_loss(grpo_batch)
            loss.backward()
            metrics = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "kl": kl_div.item(),
            }
        else:
            # 動態切 chunk；最後一塊可能比 micro_batch_size 小（餘數）。
            # 為了讓「mean over B」的梯度等價，每 chunk 的 loss 要乘
            # (chunk_size / B)，這樣加總就是 (1/B) * Σ_i L_i。
            sum_loss = 0.0
            sum_policy_loss = 0.0
            sum_kl = 0.0
            for start in range(0, actual_b, micro_batch_size):
                end = min(start + micro_batch_size, actual_b)
                chunk = self._slice_grpo_batch(grpo_batch, start, end)
                chunk_size = end - start
                weight = chunk_size / actual_b

                chunk_loss, chunk_policy, chunk_kl = self._compute_chunk_loss(chunk)
                (chunk_loss * weight).backward()

                sum_loss += chunk_loss.item() * weight
                sum_policy_loss += chunk_policy.item() * weight
                sum_kl += chunk_kl.item() * weight

            metrics = {
                "loss": sum_loss,
                "policy_loss": sum_policy_loss,
                "kl": sum_kl,
            }

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics["mean_reward"] = grpo_batch.rewards.mean().item()

        logger.info(
            "[LLMPolicy] update 完成 | B=%d, micro=%s, loss=%.4f, "
            "kl=%.4f, mean_reward=%.4f",
            actual_b,
            "off" if no_microbatch else str(micro_batch_size),
            metrics["loss"], metrics["kl"], metrics["mean_reward"],
        )
        return metrics

    def _compute_current_log_probs(self, grpo_batch: GRPOBatch) -> Tensor:
        """計算 current policy 對 generated tokens 的 log probs。

        Args:
            grpo_batch: 包含 token_ids 的 batch。

        Returns:
            Current log probs, shape (batch, gen_len).
        """
        full_ids = torch.cat(
            [grpo_batch.prompt_ids, grpo_batch.token_ids], dim=1,
        )  # (batch, prompt_len + gen_len)

        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits  # (batch, total_len, vocab_size)

        prompt_len = grpo_batch.prompt_ids.shape[1]
        gen_logits = logits[
            :, prompt_len - 1:-1, :
        ]  # (batch, gen_len, vocab_size)

        return self._gather_token_log_probs(gen_logits, grpo_batch.token_ids)

    def setup_optimizer(self, learning_rate: float, kl_coeff: float = 0.05) -> None:
        """初始化 optimizer。

        Args:
            learning_rate: 學習率。
            kl_coeff: KL penalty 係數。
        """
        self.kl_coeff = kl_coeff
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )
        logger.info(
            "[LLMPolicy] Optimizer 初始化完成 | lr=%.2e, kl_coeff=%.4f",
            learning_rate, kl_coeff,
        )

    def save_checkpoint(self, path: str) -> None:
        """儲存 LoRA adapter checkpoint。

        Args:
            path: 儲存路徑。
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("[LLMPolicy] Checkpoint 儲存至: %s", path)

    def generate_with_chat_template(
        self,
        messages_list: list[list[dict[str, str]]],
        compute_log_probs: bool = True,
    ) -> GenerationOutput:
        """使用 chat template 格式生成（適合有 system prompt 的場景）。

        Qwen3.5 必須透過 `apply_chat_template(enable_thinking=...)` 控制
        thinking mode；`/no_think` directive 在 Qwen3.5 已被移除（Qwen3
        才認）。是否關 thinking 由 `self.enable_thinking` 決定（從 config
        傳進 `__init__`）。

        Args:
            messages_list: 一批 chat messages。
            compute_log_probs: 是否計算 token-level log prob，推理建議 False。

        Returns:
            GenerationOutput。
        """
        prompts = []
        for messages in messages_list:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            prompts.append(text)

        return self.generate(prompts, compute_log_probs=compute_log_probs)

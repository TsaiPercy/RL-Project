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
        ref_model: Frozen reference model（用於 KL penalty）。
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
        temperature: float = 0.8,
        device: str = "auto",
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        """初始化 LLMPolicy：載入模型 + LoRA adapter。

        Args:
            model_name: HuggingFace model ID (e.g. "Qwen/Qwen3.5-9B").
            quantization: 量化方式 ("4bit" or "none").
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            lora_target_modules: LoRA target modules.
            max_new_tokens: 生成最大 token 數.
            temperature: Sampling temperature.
            device: 目標裝置 ("auto", "cuda", "cpu").
            load_in_4bit: 是否使用 4-bit 量化載入.
            cache_dir: HuggingFace cache directory. None = HF default.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.cache_dir = cache_dir
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        self.model, self.tokenizer = self._load_model(
            model_name, quantization, load_in_4bit, device,
        )
        self.model = self._attach_lora(self.model)
        self.ref_model = self._load_reference_model(
            model_name, quantization, load_in_4bit, device,
        )
        self.device = next(self.model.parameters()).device

        logger.info(
            "[LLMPolicy] 初始化完成 | model=%s, device=%s, "
            "lora_rank=%d, lora_alpha=%d, max_new_tokens=%d",
            model_name, self.device, lora_rank, lora_alpha, max_new_tokens,
        )

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
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )

        model.config.use_cache = False
        logger.info("[LLMPolicy] 模型載入完成: %s (quantization=%s)", model_name, quantization)
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

    def _load_reference_model(
        self,
        model_name: str,
        quantization: str,
        load_in_4bit: bool,
        device: str,
    ) -> object:
        """載入 frozen reference model（用於 KL penalty）。

        Returns:
            Frozen reference model。
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if quantization == "4bit" and load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir,
            )

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        logger.info("[LLMPolicy] Reference model 載入完成 (frozen)")
        return ref_model

    def generate(self, prompts: list[str]) -> GenerationOutput:
        """使用 LLM 生成關卡描述。

        Args:
            prompts: 一批 prompt 字串。

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

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[:, input_length:]  # (batch, gen_len)
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        log_probs = self._compute_log_probs(
            inputs["input_ids"], generated_ids,
        )  # (batch, gen_len)

        logger.info(
            "[LLMPolicy] generate 完成 | batch_size=%d, avg_gen_len=%.1f",
            len(prompts), generated_ids.shape[1],
        )

        return GenerationOutput(
            texts=texts,
            log_probs=log_probs,
            token_ids=generated_ids,
            prompt_ids=inputs["input_ids"],
        )

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

        log_probs_all = torch.log_softmax(gen_logits, dim=-1)  # (batch, gen_len, vocab_size)
        log_probs = torch.gather(
            log_probs_all, dim=2, index=generated_ids.unsqueeze(-1),
        ).squeeze(-1)  # (batch, gen_len)

        return log_probs

    def get_ref_log_probs(self, token_ids: Tensor, prompt_ids: Tensor) -> Tensor:
        """使用 reference model 計算 log probs（用於 KL penalty）。

        Args:
            token_ids: 生成的 token ids, shape (batch, gen_len).
            prompt_ids: Prompt token ids, shape (batch, prompt_len).

        Returns:
            Reference log probs, shape (batch, gen_len).
        """
        full_ids = torch.cat([prompt_ids, token_ids], dim=1)  # (batch, total_len)

        with torch.no_grad():
            outputs = self.ref_model(input_ids=full_ids)
            logits = outputs.logits  # (batch, total_len, vocab_size)

        prompt_len = prompt_ids.shape[1]
        gen_logits = logits[:, prompt_len - 1:-1, :]  # (batch, gen_len, vocab_size)

        log_probs_all = torch.log_softmax(gen_logits, dim=-1)  # (batch, gen_len, vocab_size)
        ref_log_probs = torch.gather(
            log_probs_all, dim=2, index=token_ids.unsqueeze(-1),
        ).squeeze(-1)  # (batch, gen_len)

        logger.debug(
            "[LLMPolicy] get_ref_log_probs 完成 | shape=%s", ref_log_probs.shape,
        )
        return ref_log_probs

    def update(self, grpo_batch: GRPOBatch) -> dict:
        """執行一步 GRPO 更新。

        Per SPEC §5.3:
          L = -E[advantage * log π(y|x)] + kl_coeff * KL(π || π_ref)

        Args:
            grpo_batch: 包含 token_ids, log_probs, ref_log_probs,
                        rewards, advantages 的 batch。

        Returns:
            dict 包含 loss, kl, mean_reward 等訓練指標。
        """
        self.model.train()

        if not hasattr(self, "optimizer"):
            raise RuntimeError(
                "Optimizer 未初始化。請先呼叫 setup_optimizer()。"
            )

        current_log_probs = self._compute_current_log_probs(grpo_batch)  # (batch, gen_len)

        ratio = torch.exp(
            current_log_probs - grpo_batch.log_probs
        )  # (batch, gen_len)

        advantages = grpo_batch.advantages.unsqueeze(-1)  # (batch, 1)
        policy_loss = -(advantages * ratio).mean()

        log_ratio_ref_current = grpo_batch.ref_log_probs - current_log_probs
        kl_div = (
            torch.exp(log_ratio_ref_current) - log_ratio_ref_current - 1.0
        ).mean()  # scalar, non-negative KL approximation

        loss = policy_loss + self.kl_coeff * kl_div

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_div.item(),
            "mean_reward": grpo_batch.rewards.mean().item(),
        }

        logger.info(
            "[LLMPolicy] update 完成 | loss=%.4f, kl=%.4f, mean_reward=%.4f",
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

        log_probs_all = torch.log_softmax(gen_logits, dim=-1)  # (batch, gen_len, vocab)
        log_probs = torch.gather(
            log_probs_all, dim=2, index=grpo_batch.token_ids.unsqueeze(-1),
        ).squeeze(-1)  # (batch, gen_len)

        return log_probs

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
        self, messages_list: list[list[dict[str, str]]],
    ) -> GenerationOutput:
        """使用 chat template 格式生成（適合有 system prompt 的場景）。

        Args:
            messages_list: 一批 chat messages，每個元素是
                [{"role": "system", ...}, {"role": "user", ...}]。

        Returns:
            GenerationOutput。
        """
        prompts = []
        for messages in messages_list:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(text)

        return self.generate(prompts)

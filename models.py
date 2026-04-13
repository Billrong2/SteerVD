#!/usr/bin/env python3
# models.py
# Class-based wrapper for Hugging Face causal LMs (e.g., CodeLlama, Qwen2.5)
# with attention capture, pooled-attention summaries, and optional steering.
#
# Notes:
# - HUGGINGFACE_TOKEN and optional LLM_FORCE_4BIT may be read from the environment.
# - All other settings are hard-coded defaults; change them via .config(...).
# - Keep attn_implementation="eager" so attention tensors are exposed.
# - 70B BF16 typically needs large VRAM (e.g., 4x80GB). Set use_4bit=True to experiment on smaller GPUs.
# - Attention tensors scale ~O(L^2); long prompts or large max_new_tokens increase memory.

from __future__ import annotations
import copy
import os
import json
import pprint
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple, Optional, Sequence

from huggingface_hub import login
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from steering import SteeringConfig
from steering.backends import install_steering_hooks
from steering.pointer import PointerBiasProcessor, StepAdvanceProcessor
from steering.runtime import SteeringRuntime, create_runtime
from attn_postprocess import (
    POOLING_STRATEGIES,
    extract_full_decode_head_tensors,
    postprocess_generation_attentions,
)
from paths import resolve_artifact_path

# ----------------------------
# Hard-coded defaults (change via llama.config(...))
# ----------------------------
DEFAULT_MODEL_NAME = "codellama/CodeLlama-70b-Instruct-hf"
DEFAULT_CACHE_DIR  = ".cache/models"
DEFAULT_USE_4BIT   = False
DEFAULT_MAX_NEW    = 1024
DEFAULT_TEMP       = 0.7
DEFAULT_TOP_P      = 1.0
DEFAULT_TOP_K      = 7
DEFAULT_TOP_ATTN_K = 10
DEFAULT_MEM_FRAC   = 0.90
DEFAULT_KEY_SCOPE  = "prompt"   # "prompt" | "all"
DEFAULT_RENORMAL   = True
DEFAULT_MAX_DEVICES: Optional[int] = None

class _SanitizeLogitsProcessor(LogitsProcessor):
    """
    Guardrail against rare NaN/Inf logits causing `torch.multinomial` failures during sampling.

    This is especially useful for fp16 inference on some checkpoints. It does not change
    behavior when logits are already finite.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not torch.isfinite(scores).all():
            # Replace NaN/Inf with large finite values so softmax stays well-defined.
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        # Avoid extreme magnitudes that can still destabilize softmax on some stacks.
        return scores.clamp(min=-1e9, max=1e9)

class SteeredCausalLM:
    """
    Generic HF causal LM wrapper with attention capture and optional steering.

    Usage:
        lm = SteeredCausalLM()
        lm.login_hf()  # optional; reads HUGGINGFACE_TOKEN if set
        lm.config(
            model_name="codellama/CodeLlama-70b-Instruct-hf",
            cache_dir="/path/to/cache",
            use_4bit=True,
            max_new_tokens=128,
            temperature=0.7,
            top_p=1.0,
            top_k=7,
            top_attended_k=10,
            mem_fraction=0.90,
            key_scope="prompt",   # or "all"
            renormalize=True,
        )
        lm.build()
        result = lm.run_llama("def two_sum(...): ...", language="python")
        lm.free()
    """



    # ----------------------------
    # Construction / Configuration
    # ----------------------------
    def __init__(self):

        # hard-coded defaults (no env reads here)
        self.model_name: str = DEFAULT_MODEL_NAME
        self.cache_dir:  str = DEFAULT_CACHE_DIR
        self.use_4bit:   bool = DEFAULT_USE_4BIT
        self.max_new_tokens: int = DEFAULT_MAX_NEW
        self.temperature: float  = DEFAULT_TEMP
        self.top_p: float        = DEFAULT_TOP_P
        self.top_k: int          = DEFAULT_TOP_K
        self.top_attended_k: int = DEFAULT_TOP_ATTN_K
        self.mem_fraction: float = DEFAULT_MEM_FRAC
        self.key_scope: str      = DEFAULT_KEY_SCOPE
        self.renormalize: bool   = DEFAULT_RENORMAL
        self.max_devices: Optional[int] = DEFAULT_MAX_DEVICES

        # internals
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.steering_config: Optional[SteeringConfig] = None
        self._steering_runtime: Optional[SteeringRuntime] = None
        self._last_steering_runtime: Optional[SteeringRuntime] = None
        self._current_code_snippet: str = ""
        self._current_vocab_tokens: Sequence[dict] = []

    def config(
        self,
        *,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_4bit: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        top_attended_k: Optional[int] = None,
        mem_fraction: Optional[float] = None,
        key_scope: Optional[str] = None,           # "prompt" | "all"
        renormalize: Optional[bool] = None,
        max_devices: Optional[int] = None,
    ) -> "SteeredCausalLM":
        """Programmatic configuration (no environment variables used)."""
        if model_name is not None:     self.model_name = model_name
        if cache_dir is not None:      self.cache_dir = cache_dir
        if use_4bit is not None:       self.use_4bit = use_4bit
        if max_new_tokens is not None: self.max_new_tokens = max_new_tokens
        if temperature is not None:    self.temperature = temperature
        if top_p is not None:          self.top_p = top_p
        if top_k is not None:          self.top_k = top_k
        if top_attended_k is not None: self.top_attended_k = top_attended_k
        if mem_fraction is not None:   self.mem_fraction = mem_fraction
        if key_scope is not None:      self.key_scope = key_scope.lower()
        if renormalize is not None:    self.renormalize = renormalize
        if max_devices is not None:    self.max_devices = max_devices
        return self

    def set_steering_config(self, config: SteeringConfig) -> None:
        self.steering_config = config

    # ----------------------------
    # Auth (only HUGGINGFACE_TOKEN allowed from env)
    # ----------------------------
    def login_hf(self, token: Optional[str] = None):
        """
        Login to Hugging Face Hub. If token not provided explicitly,
        falls back to the HUGGINGFACE_TOKEN environment variable.
        """
        tok = token or os.getenv("HUGGINGFACE_TOKEN")
        if tok:
            login(token=tok)

    # ----------------------------
    # Build / Load
    # ----------------------------
    @staticmethod
    def _auto_max_memory_dict(fraction: float = 0.90) -> Dict[int, str]:
        """
        Build a max_memory dict for all visible GPUs using a fraction of total VRAM.
        Example return: {0: '71GiB', 1: '71GiB', 2: '71GiB', 3: '71GiB'}
        """
        if not torch.cuda.is_available():
            return {}
        n = torch.cuda.device_count()
        mm: Dict[int, str] = {}
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            allowed = int(props.total_memory * fraction)
            mm[i] = f"{allowed // (1024**3)}GiB"
        return mm

    def build(self):
        """Load tokenizer + model with attention enabled configuration."""
        print(f"Loading model: {self.model_name}")
        print(f"USE_4BIT={self.use_4bit}  |  GPUs visible={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        max_memory = self._auto_max_memory_dict(self.mem_fraction)
        selected_devices = None
        if torch.cuda.is_available() and max_memory:
            if self.max_devices is not None:
                visible = torch.cuda.device_count()
                use = max(1, min(self.max_devices, visible))
                selected_devices = list(range(use))
                max_memory = {k: v for k, v in max_memory.items() if k in selected_devices}
                print(f"Limiting to {use} GPU(s) (of {visible}).")
            print("Per-GPU max_memory cap:", max_memory)

        tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        tok.padding_side = "left"
        for k in ["hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads",
          "intermediate_size","rms_norm_eps","rope_theta","max_position_embeddings",
          "vocab_size","tie_word_embeddings"]:
            print(k, "=", getattr(tok, k, None))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id

        common = dict(
            device_map="auto",
            cache_dir=self.cache_dir,
            attn_implementation="eager",
        )
        if max_memory:
            common["max_memory"] = max_memory

        force_4bit = self.use_4bit or bool(int(os.environ.get("LLM_FORCE_4BIT", "0")))
        model = None
        if force_4bit:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    **common,
                )
                print("[Steering] Loaded model in 4-bit NF4 mode.")
            except (ImportError, AttributeError, RuntimeError) as err:
                print(f"[WARN] 4-bit load failed ({err}); falling back to full precision.")

        if model is None:
            if torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                **common,
            )
            print(f"[INFO] Loaded model with torch_dtype={dtype}.")

        model.eval()
        self.model = model
        self.tokenizer = tok

        # ensure model config pad/eos are set (required for HF >=4.38 generation)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tok.pad_token_id
        if model.config.eos_token_id is None:
            model.config.eos_token_id = tok.eos_token_id

        if self.steering_config and self.steering_config.enabled_levels:
            self._install_steering_hooks()

        # Optional: print device map
        try:
            print("\n=== Device map (layer -> GPU) ===")
            pprint.pprint(model.hf_device_map)
        except Exception:
            pass

    # ----------------------------
    # Utilities
    # ----------------------------
    # ----------------------------
    # Core: generate with attention
    # ----------------------------
    def _should_use_split_prefill(self, runtime: Optional[SteeringRuntime], prompt_len: int) -> bool:
        if runtime is None:
            return False
        if self.steering_config is None:
            return False
        if not self.steering_config.split_prefill:
            return False
        if not runtime.decode_only:
            return False
        if prompt_len < 2:
            return False
        return any(level in self.steering_config.enabled_levels for level in (1, 2))

    @staticmethod
    def _build_prompt(
        code_snippet: str,
        *,
        instruction: str,
        language: str,
        answer_prefix: str = "",
    ) -> str:
        suffix = answer_prefix or ""
        return f"{instruction}\n\n```{language}\n{code_snippet}\n```{suffix}"

    def _sample_next_token(
        self,
        scores: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        if not do_sample:
            return torch.argmax(scores, dim=-1, keepdim=True)

        scores = scores / max(float(temperature), 1e-6)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), scores.size(-1))
            cutoff = torch.topk(scores, k)[0][..., -1, None]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            remove = cdf > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _manual_decode_from_past(
        self,
        *,
        last_id: torch.Tensor,
        full_mask: torch.Tensor,
        initial_position_ids: torch.Tensor,
        past_key_values: Any,
        gen_cfg: GenerationConfig,
        logits_processor: LogitsProcessorList,
        record_attention: bool,
    ) -> SimpleNamespace:
        generated = last_id
        attentions: List[Any] = []
        past = past_key_values
        input_step = last_id
        dynamic_mask = full_mask
        dynamic_position_ids = initial_position_ids
        eos_ids = gen_cfg.eos_token_id
        if eos_ids is None:
            eos_set = set()
        elif isinstance(eos_ids, (list, tuple, set)):
            eos_set = {int(v) for v in eos_ids}
        else:
            eos_set = {int(eos_ids)}

        with torch.no_grad():
            for _ in range(int(gen_cfg.max_new_tokens)):
                outputs = self.model(
                    input_ids=input_step,
                    attention_mask=dynamic_mask,
                    position_ids=dynamic_position_ids,
                    past_key_values=past,
                    use_cache=True,
                    output_attentions=record_attention,
                    return_dict=True,
                )
                past = outputs.past_key_values
                if record_attention:
                    attentions.append(outputs.attentions)

                scores = outputs.logits[:, -1, :]
                scores = logits_processor(generated, scores)
                next_token = self._sample_next_token(
                    scores,
                    do_sample=bool(gen_cfg.do_sample),
                    temperature=float(gen_cfg.temperature or 1.0),
                    top_p=float(gen_cfg.top_p or 1.0),
                    top_k=gen_cfg.top_k,
                )
                generated = torch.cat([generated, next_token], dim=-1)
                input_step = next_token
                dynamic_position_ids = dynamic_position_ids + 1
                dynamic_mask = torch.cat(
                    [
                        dynamic_mask,
                        torch.ones((dynamic_mask.shape[0], 1), dtype=dynamic_mask.dtype, device=dynamic_mask.device),
                    ],
                    dim=-1,
                )
                if eos_set and all(int(tok.item()) in eos_set for tok in next_token.view(-1)):
                    break

        return SimpleNamespace(
            sequences=generated,
            attentions=tuple(attentions),
        )

    def _generate_with_split_prefill(
        self,
        *,
        enc: Dict[str, torch.Tensor],
        gen_cfg: GenerationConfig,
        logits_processor: LogitsProcessorList,
        record_attention: bool,
        runtime: Optional[SteeringRuntime],
    ) -> Tuple[Any, torch.Tensor]:
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if not bool(torch.all(attention_mask[:, -1] == 1)):
            raise ValueError(
                "Split-prefill requires the final token in each sample to be non-pad. "
                "Use left-padding or batch size 1."
            )

        prefix_ids = input_ids[:, :-1]
        last_id = input_ids[:, -1:]
        prefix_mask = attention_mask[:, :-1]
        full_mask = attention_mask
        if runtime:
            prefill_calls_before = runtime.steer_calls
        with torch.no_grad():
            prefill = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                return_dict=True,
            )
        if runtime and runtime.steer_calls != prefill_calls_before:
            raise AssertionError("Steering fired during prefix prefill, which should be decode-only.")

        # left-padded compatible position for the final prompt token
        pos_last = (full_mask.sum(dim=-1) - 1).clamp_min(0).unsqueeze(1)

        try:
            with torch.no_grad():
                out_tail = self.model.generate(
                    input_ids=last_id,
                    attention_mask=full_mask,
                    past_key_values=prefill.past_key_values,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                    output_attentions=record_attention,
                    output_scores=False,
                    use_cache=True,
                    logits_processor=logits_processor,
                )
        except (TypeError, ValueError, RuntimeError, IndexError):
            out_tail = self._manual_decode_from_past(
                last_id=last_id,
                full_mask=full_mask,
                initial_position_ids=pos_last,
                past_key_values=prefill.past_key_values,
                gen_cfg=gen_cfg,
                logits_processor=logits_processor,
                record_attention=record_attention,
            )

        full_sequences = torch.cat([prefix_ids, out_tail.sequences], dim=-1)
        out_tail.sequences = full_sequences
        return out_tail, full_sequences

    def _generate_with_attn(
        self,
        prompt: str,
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tokenizer is not None, "Call .build() first."

        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[-1]
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long)

        overrides = overrides or {}
        temperature = overrides.get("temperature", self.temperature)
        top_p = overrides.get("top_p", self.top_p)
        top_k = overrides.get("top_k", self.top_k)
        max_new_tokens = overrides.get("max_new_tokens", self.max_new_tokens)
        record_layers = bool(overrides.get("record_layers", True))
        record_attention = bool(overrides.get("record_attention", record_layers))
        record_layers = record_layers and record_attention
        do_sample = overrides.get("do_sample")
        if do_sample is None:
            do_sample = temperature > 0

        runtime: Optional[SteeringRuntime] = None
        logits_processor = LogitsProcessorList()
        if self.steering_config:
            self.steering_config.model_name = self.model_name
            prompt_ids = input_ids[0].tolist()
            prompt_tokens = self.tokenizer.convert_ids_to_tokens(prompt_ids)
            code_text = getattr(self, "_current_code_snippet", "")
            runtime = create_runtime(
                self.steering_config,
                prompt_token_ids=prompt_ids,
                prompt_tokens=prompt_tokens,
                code_text=code_text,
                vocab_tokens=self._current_vocab_tokens,
                prompt_text=prompt,
                prompt_attention_mask=enc["attention_mask"][0].tolist(),
            )
            runtime.start(max_new_tokens)
            self._steering_runtime = runtime
            if 5 in self.steering_config.enabled_levels:
                logits_processor.append(PointerBiasProcessor(runtime))
            else:
                logits_processor.append(StepAdvanceProcessor(runtime))
        logits_processor.append(_SanitizeLogitsProcessor())

        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=None if (top_k or 0) <= 0 else top_k,
            cache_dir=self.cache_dir,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        use_split_prefill = self._should_use_split_prefill(runtime, prompt_len)
        if use_split_prefill:
            out, sequences = self._generate_with_split_prefill(
                enc=enc,
                gen_cfg=gen_cfg,
                logits_processor=logits_processor,
                record_attention=record_attention,
                runtime=runtime,
            )
        else:
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    generation_config=gen_cfg,
                    return_dict_in_generate=True,
                    output_attentions=record_attention,   # capture attention only when requested
                    output_scores=False,
                    use_cache=True,
                    logits_processor=logits_processor,
                )
            sequences = out.sequences  # [batch, prompt_len + gen_len]

        if runtime:
            self._last_steering_runtime = runtime
            self._steering_runtime = None
        steering_debug = {
            "enabled": bool(runtime and runtime.enabled),
            "decode_only": bool(runtime and runtime.decode_only),
            "only_first_decode_step": bool(runtime and runtime.only_first_decode_step),
            "prompt_len": int(prompt_len),
            "steer_calls": int(runtime.steer_calls if runtime else 0),
            "blocked_prefill_calls": int(runtime.blocked_prefill_calls if runtime else 0),
            "blocked_q_len": int(runtime.blocked_q_len if runtime else 0),
            "blocked_kv_len": int(runtime.blocked_kv_len if runtime else 0),
            "blocked_layer": int(runtime.blocked_layer if runtime else 0),
            "blocked_disabled": int(runtime.blocked_disabled if runtime else 0),
            "current_step_kv_len": int(runtime.current_step_kv_len) if (runtime and runtime.current_step_kv_len is not None) else None,
            "split_prefill_used": bool(use_split_prefill),
            "recency_mix": bool(self.steering_config.recency_mix) if self.steering_config else False,
            "recency_rho": float(self.steering_config.recency_rho) if self.steering_config else 0.0,
            "recency_window": int(self.steering_config.recency_window) if self.steering_config else 0,
            "recency_apply_after_prompt": bool(self.steering_config.recency_apply_after_prompt) if self.steering_config else False,
            "recency_scope": str(self.steering_config.recency_scope) if self.steering_config else "",
            "head_subset_mode": str(self.steering_config.head_subset_mode) if self.steering_config else "none",
            "head_mask_apply_to": str(self.steering_config.head_mask_apply_to) if self.steering_config else "both",
            "head_mask_loaded": bool(runtime.head_mask_loaded) if runtime else False,
            "head_mask_error": runtime.head_mask_error if runtime else None,
            "head_mask_active_total": int(runtime.head_mask_active_total) if runtime else 0,
            "head_mask_active_by_layer": dict(runtime.head_mask_active_by_layer) if runtime else {},
            "head_subset_selected_heads": (
                dict(self.steering_config.head_subset_selected_heads)
                if self.steering_config
                else {}
            ),
            "head_subset_calibration": (
                dict(self.steering_config.head_subset_calibration)
                if self.steering_config
                else {}
            ),
            "head_stats": runtime.head_stats_payload() if runtime else None,
            "temporal_debug": list(runtime.temporal_debug) if runtime else [],
            "residual_scale_enabled": bool(self.steering_config.residual_scale) if self.steering_config else False,
            "residual_scale_mode": str(self.steering_config.residual_scale_mode) if self.steering_config else "static",
            "residual_scale_layer_start": (
                int(self.steering_config.residual_scale_layer_start)
                if (self.steering_config and self.steering_config.residual_scale_layer_start is not None)
                else None
            ),
            "residual_scale_layer_end": (
                int(self.steering_config.residual_scale_layer_end)
                if (self.steering_config and self.steering_config.residual_scale_layer_end is not None)
                else None
            ),
            "residual_calls_attn": int(runtime.residual_calls_attn) if runtime else 0,
            "residual_calls_mlp": int(runtime.residual_calls_mlp) if runtime else 0,
            "blocked_residual_prefill": int(runtime.blocked_residual_prefill) if runtime else 0,
            "blocked_residual_layer": int(runtime.blocked_residual_layer) if runtime else 0,
            "blocked_residual_disabled": int(runtime.blocked_residual_disabled) if runtime else 0,
            "blocked_residual_no_steer": int(runtime.blocked_residual_no_steer) if runtime else 0,
            "residual_debug": list(runtime.residual_debug) if runtime else [],
            "pointer_calls_total": int(runtime.pointer_calls_total) if runtime else 0,
            "pointer_bias_applied_steps": int(runtime.pointer_bias_applied_steps) if runtime else 0,
            "pointer_missing_attention_steps": int(runtime.pointer_missing_attention_steps) if runtime else 0,
            "pointer_beta_zero_steps": int(runtime.pointer_beta_zero_steps) if runtime else 0,
            "level_call_counts": runtime.level_call_counts_payload() if runtime else {},
            "level_event_trace": runtime.level_event_trace_payload() if runtime else [],
        }
        full_text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(sequences[0][:prompt_len], skip_special_tokens=True)
        generated_completion = self.tokenizer.decode(sequences[0][prompt_len:], skip_special_tokens=True)

        total_len = sequences.shape[-1]
        num_generated = total_len - prompt_len
        ids_all = sequences[0].tolist()
        tokens_all = self.tokenizer.convert_ids_to_tokens(ids_all)

        if not record_attention:
            return {
                "model": self.model_name,
                "use_4bit": self.use_4bit,
                "prompt_length_tokens": prompt_len,
                "num_generated_tokens": num_generated,
                "tokens_all": tokens_all,
                "token_ids_all": ids_all,
                "generated_text": full_text,
                "prompt_text": prompt_text,
                "generated_completion": generated_completion,
                "attention_by_generated_token": [],
                "pooled_attention_by_generated_token": [],
                "global_pooled_attention_over_prompt": {},
                "pooling_strategies": dict(POOLING_STRATEGIES),
                "key_scope": self.key_scope,
                "renormalized": self.renormalize,
                "layer_prompt_stats": None,
                "record_layers": False,
                "record_attention": False,
                "steering_debug": steering_debug,
                "full_decode_head_tensors": None,
                "generated_logits_path": None,
                "generated_logits_shape": None,
                "generated_logits_dtype": None,
                "logits_recorded": False,
            }

        attn_summaries = postprocess_generation_attentions(
            attentions=getattr(out, "attentions", None),
            tokens_all=tokens_all,
            prompt_len=prompt_len,
            num_generated=num_generated,
            key_scope=self.key_scope,
            renormalize=self.renormalize,
            top_attended_k=self.top_attended_k,
            record_layers=record_layers,
            model=self.model,
        )
        full_decode_head_tensors = None
        if record_layers:
            full_decode_head_tensors = extract_full_decode_head_tensors(
                attentions=getattr(out, "attentions", None),
                tokens_all=tokens_all,
                token_ids_all=ids_all,
                prompt_len=prompt_len,
                num_generated=num_generated,
            )

        return {
            "model": self.model_name,
            "use_4bit": self.use_4bit,
            "prompt_length_tokens": prompt_len,
            "num_generated_tokens": num_generated,
            "tokens_all": tokens_all,
            "token_ids_all": ids_all,
            "generated_text": full_text,
            "prompt_text": prompt_text,
            "generated_completion": generated_completion,
            # per-layer diagnostic (kept)
            "attention_by_generated_token": attn_summaries.attention_by_generated_token if record_layers else [],
            # per-step per-pool top-K
            "pooled_attention_by_generated_token": attn_summaries.pooled_attention_by_generated_token,
            # global per-pool distribution over prompt tokens (only when KEY_SCOPE='prompt')
            "global_pooled_attention_over_prompt": attn_summaries.global_pooled_attention_over_prompt,
            "pooling_strategies": dict(POOLING_STRATEGIES),
            "key_scope": self.key_scope,
            "renormalized": self.renormalize,
            "layer_prompt_stats": attn_summaries.layer_prompt_stats,
            "record_layers": record_layers,
            "record_attention": record_attention,
            "steering_debug": steering_debug,
            "full_decode_head_tensors": full_decode_head_tensors,
            "generated_logits_path": None,
            "generated_logits_shape": None,
            "generated_logits_dtype": None,
            "logits_recorded": False,
        }

    # ----------------------------
    # Public interface
    # ----------------------------
    def run_llama(
        self,
        code_snippet: str,
        instruction: Optional[str] = None,
        *,
        language: str = "java",
        answer_prefix: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        record_layers: Optional[bool] = None,
        record_attention: Optional[bool] = None,
        vocab_tokens: Optional[Sequence[dict]] = None,
        steering_code_snippet: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate with the configured causal LM using a caller-provided instruction.
        """
        if instruction is None:
            instruction = "Summarize what this Java function does. Be concise and accurate."

        prompt = self._build_prompt(
            code_snippet,
            instruction=instruction,
            language=language,
            answer_prefix=answer_prefix,
        )

        overrides: Dict[str, Any] = {}
        if max_new_tokens is not None:
            overrides["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            overrides["temperature"] = temperature
        if top_p is not None:
            overrides["top_p"] = top_p
        if top_k is not None:
            overrides["top_k"] = top_k
        if do_sample is not None:
            overrides["do_sample"] = do_sample
        if record_layers is not None:
            overrides["record_layers"] = record_layers
        if record_attention is not None:
            overrides["record_attention"] = record_attention

        self._current_code_snippet = steering_code_snippet if steering_code_snippet is not None else code_snippet
        self._current_vocab_tokens = vocab_tokens or []
        try:
            return self._generate_with_attn(prompt, overrides=overrides)
        finally:
            self._current_code_snippet = ""
            self._current_vocab_tokens = []

    # Steering integration -------------------------------------------------

    def _resolve_auto_mask_save_path(self, raw_path: Path, *, snippet_name: str, topk: int) -> Path:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        rendered = str(raw_path).format(
            snippet=snippet_name,
            model=safe_model,
            topk=topk,
            ts=time.strftime("%Y%m%d-%H%M%S"),
        )
        path = Path(rendered)
        if not path.is_absolute():
            path = resolve_artifact_path(Path(__file__).resolve().parent, path)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            return path / f"{safe_model}-{snippet_name}-topk{topk}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def calibrate_head_subset(
        self,
        *,
        code_snippet: str,
        instruction: str,
        language: str = "java",
        vocab_tokens: Optional[Sequence[dict]] = None,
        snippet_name: Optional[str] = None,
        steering_code_snippet: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Auto head subset calibration:
        - Collect per-layer/per-head agreement stats with betas forced to zero.
        - Select top-k heads per layer in the active attention-steering band.
        - Store in-memory mask on current steering config (no prebuilt JSON required).
        """
        if self.steering_config is None:
            raise RuntimeError("Head subset auto mode requires an active steering config.")

        base_cfg = self.steering_config
        if base_cfg.head_subset_mode != "auto":
            return {"mode": base_cfg.head_subset_mode, "active_total": 0, "layers_with_heads": 0}
        if not any(level in base_cfg.enabled_levels for level in (1, 2)):
            raise RuntimeError("Head subset auto mode requires active attention steering.")

        from steering.backends.common import compute_default_cutoffs

        runs = max(1, int(base_cfg.head_subset_calib_runs))
        calib_max_new = max(1, int(base_cfg.head_subset_calib_max_new_tokens))
        topk = max(1, int(base_cfg.head_subset_topk_per_layer))
        snippet_id = snippet_name or "snippet"

        calib_cfg = copy.deepcopy(base_cfg)
        calib_cfg.head_subset_mode = "none"  # no mask during calibration.
        calib_cfg.head_mask_path = None
        calib_cfg.head_mask_inline = None
        calib_cfg.head_subset_selected_heads = {}
        calib_cfg.head_subset_calibration = {}
        calib_cfg.collect_head_stats = True
        calib_cfg.collect_head_stats_first_decode_only = bool(base_cfg.head_subset_calib_first_decode_only)
        calib_cfg.beta_bias = 0.0
        calib_cfg.beta_post = 0.0
        calib_cfg.schedule = {}

        agg_sum: Optional[np.ndarray] = None
        agg_count: Optional[np.ndarray] = None
        valid_runs = 0

        prev_cfg = self.steering_config
        self.steering_config = calib_cfg
        try:
            for run_idx in range(1, runs + 1):
                result = self.run_llama(
                    code_snippet,
                    instruction=instruction,
                    language=language,
                    max_new_tokens=calib_max_new,
                    do_sample=True,
                    record_layers=False,
                    record_attention=False,
                    vocab_tokens=vocab_tokens,
                    steering_code_snippet=steering_code_snippet,
                )
                hs = (result.get("steering_debug") or {}).get("head_stats")
                if not hs:
                    continue
                run_sum = np.asarray(hs.get("sum", []), dtype=np.float64)
                run_count = np.asarray(hs.get("count", []), dtype=np.float64)
                if run_sum.ndim != 2 or run_count.ndim != 2:
                    continue
                if agg_sum is None:
                    agg_sum = np.zeros_like(run_sum, dtype=np.float64)
                    agg_count = np.zeros_like(run_count, dtype=np.float64)
                if agg_sum.shape != run_sum.shape:
                    raise RuntimeError(
                        f"Head stats shape mismatch during calibration: {agg_sum.shape} vs {run_sum.shape}"
                    )
                agg_sum += run_sum
                agg_count += run_count
                valid_runs += 1
        finally:
            self.steering_config = prev_cfg

        if agg_sum is None or agg_count is None or valid_runs == 0:
            raise RuntimeError("Step-4 auto calibration failed: no valid head stats collected.")

        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.divide(agg_sum, np.maximum(agg_count, 1.0), where=np.maximum(agg_count, 1.0) > 0)

        num_layers, num_heads = mean.shape
        cutoffs = compute_default_cutoffs(num_layers)
        layer_start = cutoffs.l12_start if base_cfg.steer_layer_start is None else int(base_cfg.steer_layer_start)
        layer_end = cutoffs.l12_end if base_cfg.steer_layer_end is None else int(base_cfg.steer_layer_end)
        layer_start = max(0, min(num_layers - 1, layer_start))
        layer_end = max(layer_start, min(num_layers - 1, layer_end))
        k = max(1, min(topk, num_heads))

        mask = np.zeros((num_layers, num_heads), dtype=bool)
        active_heads: Dict[str, List[int]] = {}
        for li in range(layer_start, layer_end + 1):
            candidates = np.where(agg_count[li] > 0)[0]
            if candidates.size == 0:
                continue
            scores = mean[li, candidates]
            order = np.argsort(-scores)
            selected = candidates[order[: min(k, candidates.size)]].astype(int).tolist()
            if not selected:
                continue
            mask[li, selected] = True
            active_heads[str(li)] = selected

        active_total = int(mask.sum())
        if active_total <= 0:
            raise RuntimeError("Step-4 auto calibration selected zero heads; aborting.")

        base_cfg.head_mask_inline = {
            "meta": {
                "model_name": self.model_name,
                "mode": "auto",
                "snippet": snippet_id,
            },
            "mask": mask.astype(np.int32).tolist(),
            "active_heads": active_heads,
        }
        base_cfg.head_subset_selected_heads = active_heads
        base_cfg.head_subset_calibration = {
            "mode": "auto",
            "calib_runs_requested": int(runs),
            "calib_runs_valid": int(valid_runs),
            "calib_max_new_tokens": int(calib_max_new),
            "collect_first_decode_only": bool(base_cfg.head_subset_calib_first_decode_only),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "topk_per_layer": int(k),
            "active_total": int(active_total),
            "layers_with_heads": int(len(active_heads)),
            "auto_save_path": None,
        }

        auto_save_path: Optional[str] = None
        if base_cfg.head_subset_auto_save:
            save_path = self._resolve_auto_mask_save_path(
                base_cfg.head_subset_auto_save,
                snippet_name=snippet_id,
                topk=k,
            )
            payload = {
                "meta": {
                    "snippet": snippet_id,
                    "model_name": self.model_name,
                    "mode": "auto",
                    "calib_runs_requested": int(runs),
                    "calib_runs_valid": int(valid_runs),
                    "calib_max_new_tokens": int(calib_max_new),
                    "collect_first_decode_only": bool(base_cfg.head_subset_calib_first_decode_only),
                    "layer_start": int(layer_start),
                    "layer_end": int(layer_end),
                    "topk_per_layer": int(k),
                },
                "shape": [int(num_layers), int(num_heads)],
                "mask": mask.astype(np.int32).tolist(),
                "active_heads": active_heads,
                "agree_mean": mean.tolist(),
                "agree_sum": agg_sum.tolist(),
                "agree_count": agg_count.astype(np.int64).tolist(),
            }
            save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            auto_save_path = str(save_path)
            print(f"[HeadSubset-Auto] saved mask -> {auto_save_path}")
            base_cfg.head_subset_calibration["auto_save_path"] = auto_save_path

        # Smoke check 1: auto mode should select non-empty heads.
        if int(mask.sum()) <= 0:
            raise RuntimeError("Step-4 smoke check failed: auto mode selected empty head set.")

        # Smoke check 2: when beta=0, auto mask must be a no-op.
        beta_zero = abs(float(base_cfg.beta_bias)) <= 1e-12 and abs(float(base_cfg.beta_post)) <= 1e-12
        noop_ok: Optional[bool] = None
        if beta_zero:
            cfg_none = copy.deepcopy(base_cfg)
            cfg_none.head_subset_mode = "none"
            cfg_none.head_mask_inline = None
            cfg_none.collect_head_stats = False

            cfg_auto = copy.deepcopy(base_cfg)
            cfg_auto.head_subset_mode = "auto"
            cfg_auto.collect_head_stats = False

            self.steering_config = cfg_none
            out_none = self.run_llama(
                code_snippet,
                instruction=instruction,
                language=language,
                max_new_tokens=min(calib_max_new, 16),
                do_sample=False,
                record_layers=False,
                record_attention=False,
                vocab_tokens=vocab_tokens,
            )
            self.steering_config = cfg_auto
            out_auto = self.run_llama(
                code_snippet,
                instruction=instruction,
                language=language,
                max_new_tokens=min(calib_max_new, 16),
                do_sample=False,
                record_layers=False,
                record_attention=False,
                vocab_tokens=vocab_tokens,
            )
            noop_ok = out_none.get("token_ids_all") == out_auto.get("token_ids_all")
            self.steering_config = base_cfg
            if not noop_ok:
                raise RuntimeError("Step-4 smoke check failed: auto head subset with beta=0 changed outputs.")

        return {
            "mode": "auto",
            "active_total": int(active_total),
            "layers_with_heads": int(len(active_heads)),
            "topk_per_layer": int(k),
            "calib_runs_requested": int(runs),
            "calib_runs_valid": int(valid_runs),
            "layer_start": int(layer_start),
            "layer_end": int(layer_end),
            "auto_save_path": auto_save_path,
            "beta_zero_noop_ok": noop_ok,
        }

    def _install_steering_hooks(self) -> None:
        assert self.model is not None
        assert self.steering_config is not None

        def runtime_getter():
            return self._steering_runtime

        backend = install_steering_hooks(self.model, runtime_getter, self.steering_config)
        print(f"[Steering] Installed backend={backend}.")

    def free(self):
        """Release model and VRAM."""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Backwards-compatible aliases (the codebase historically used `llama70b()` as
# the entrypoint for any causal LM; keep it to avoid touching other modules).
ModelRunner = SteeredCausalLM
llama70b = SteeredCausalLM

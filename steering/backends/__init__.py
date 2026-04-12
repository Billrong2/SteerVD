from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import SteeringConfig
    from ..runtime import SteeringRuntime


def _model_type(model: Any) -> str:
    cfg = getattr(model, "config", None)
    return str(getattr(cfg, "model_type", "") or "").lower()


def install_steering_hooks(
    model: Any,
    runtime_getter: Callable[[], Optional["SteeringRuntime"]],
    config: "SteeringConfig",
) -> str:
    """
    Install steering hooks for the given HF model.

    Returns the backend name ("llama", "qwen2", "qwen3", or "deepseek_v2").
    """

    model_type = _model_type(model)
    if model_type == "qwen2":
        from .qwen2_backend import install_qwen2_steering

        install_qwen2_steering(model, runtime_getter, config)
        return "qwen2"

    if model_type == "llama":
        from .llama_backend import install_llama_steering

        install_llama_steering(model, runtime_getter, config)
        return "llama"

    if model_type in {"qwen3", "qwen3_moe", "qwen3_next"}:
        from .qwen3_backend import install_qwen3_steering

        install_qwen3_steering(model, runtime_getter, config)
        return "qwen3"

    if model_type == "deepseek_v2":
        from .deepseek_v2_backend import install_deepseek_v2_steering

        install_deepseek_v2_steering(model, runtime_getter, config)
        return "deepseek_v2"

    raise ValueError(
        "Unsupported model_type="
        f"{model_type!r} for steering. Supported: 'llama' (CodeLlama/LLaMA), "
        "'qwen2' (Qwen2.5), 'qwen3'/'qwen3_moe'/'qwen3_next' (Qwen3 family), "
        "'deepseek_v2' (DeepSeek-Coder-V2)."
    )

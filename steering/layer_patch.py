"""
Legacy import shim.

The original implementation in this module was LLaMA/CodeLlama-specific.
It has been moved to `steering.backends.llama_backend` to keep architecture
patching separate per model family.
"""

from .backends.llama_backend import patch_decoder_layer

__all__ = ["patch_decoder_layer"]

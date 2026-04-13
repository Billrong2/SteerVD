from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class PriorContext:
    prompt_tokens: Sequence[str]
    code_text: str
    vocab_tokens: Sequence[dict]
    prompt_text: str = ""


class PriorProvider:
    def __init__(self, context: PriorContext) -> None:
        self.context = context

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        if vec.size == 0:
            return vec
        total = float(vec.sum())
        if total <= 0.0:
            return np.full(vec.shape, 1.0 / float(vec.size), dtype=float)
        return vec / total

    def vector(self, bin_idx: int, n_bins: int) -> np.ndarray:
        raise NotImplementedError


class PendingCodeGadgetPrior(PriorProvider):
    """
    Future steering prior for multi-gadget VulDeePecker-style extraction.

    The extractor exists in `steering.code_gadget`, but the final rule that
    reduces multiple gadgets into one prompt-aligned steering vector is still
    intentionally undefined. PrimeVul entry points block this path earlier; the
    class remains here as a defensive guard for any direct runtime usage.
    """

    def __init__(
        self,
        context: PriorContext,
        *,
        joern_cli_dir: Path | None = None,
        cache_dir: Path | None = None,
        slice_depth: int = 20,
        parallelism: int = 1,
        timeout_sec: int = 180,
        max_hops: int | None = None,
    ) -> None:
        super().__init__(context)
        del joern_cli_dir
        del cache_dir
        del slice_depth
        del parallelism
        del timeout_sec
        del max_hops
        raise RuntimeError(
            "code_gadget steering is not implemented yet: the final multi-gadget-to-prior "
            "reduction rule has not been defined."
        )

    def vector(self, bin_idx: int, n_bins: int) -> np.ndarray:
        del bin_idx
        del n_bins
        raise RuntimeError("PendingCodeGadgetPrior cannot produce a steering vector yet.")


PRIOR_REGISTRY = {
    "code_gadget": PendingCodeGadgetPrior,
}

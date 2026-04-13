from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .binning import BinAssignment, equal_count_bins
from .config import SteeringConfig
from .priors import PRIOR_REGISTRY, PriorContext


@dataclass
class SteeringState:
    bins: Sequence[BinAssignment]
    current_bin: int = 0


class SteeringManager:
    def __init__(
        self,
        config: SteeringConfig,
        prompt_tokens: Sequence[str],
        code_text: str,
        vocab_tokens: Sequence[dict],
        prompt_text: str = "",
    ):
        self.config = config
        self.config.load_schedule()
        prior_cls = PRIOR_REGISTRY[config.prior]
        prior_kwargs = {}
        if config.prior == "code_gadget":
            prior_kwargs.update(
                joern_cli_dir=config.joern_cli_dir,
                cache_dir=config.joern_cache_dir,
                slice_depth=config.joern_slice_depth,
                parallelism=config.joern_parallelism,
                timeout_sec=config.joern_timeout_sec,
                max_hops=config.joern_max_hops,
            )
        context = PriorContext(
            prompt_tokens=prompt_tokens,
            code_text=code_text,
            vocab_tokens=vocab_tokens,
            prompt_text=prompt_text,
        )
        self.prior_provider = prior_cls(context, **prior_kwargs)
        self.state: Optional[SteeringState] = None

    def init_bins(self, total_steps: int) -> None:
        self.state = SteeringState(bins=equal_count_bins(total_steps, self.config.n_bins))

    def step(self, step_idx: int) -> None:
        if not self.state:
            raise RuntimeError("SteeringManager not initialized")
        for idx, assignment in enumerate(self.state.bins):
            if assignment.start_step <= step_idx <= assignment.end_step:
                self.state.current_bin = idx
                break

    def prior_vector(self) -> np.ndarray:
        if not self.state:
            raise RuntimeError("SteeringManager not initialized")
        return self.prior_provider.vector(self.state.current_bin, len(self.state.bins))

    def coeffs(self):
        if not self.state:
            raise RuntimeError("SteeringManager not initialized")
        return self.config.coeff_for_bin(self.state.current_bin)

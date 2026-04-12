from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BinAssignment:
    start_step: int
    end_step: int
    indices: Sequence[int]


def equal_count_bins(step_count: int, n_bins: int) -> list[BinAssignment]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    indices = list(range(step_count))
    chunk = max(1, (step_count + n_bins - 1) // n_bins)
    bins: list[BinAssignment] = []
    for i in range(n_bins):
        chunk_indices = indices[i * chunk : (i + 1) * chunk]
        if not chunk_indices:
            break
        bins.append(
            BinAssignment(
                start_step=chunk_indices[0],
                end_step=chunk_indices[-1],
                indices=chunk_indices,
            )
        )
    return bins

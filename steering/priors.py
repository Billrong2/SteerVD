from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class PriorContext:
    prompt_tokens: Sequence[str]
    code_text: str
    vocab_tokens: Sequence[dict]
    prompt_text: str = ""
    prompt_token_offsets: Optional[Sequence[tuple[int, int]]] = None


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


class CodeGadgetPrior(PriorProvider):
    """
    Prompt-aligned prior derived from a single exported code gadget artifact.

    The runner selects one gadget directory per steered sample. This provider
    reads that gadget's `line_sequence` and projects the selected snippet lines
    back onto prompt tokens using prompt character offsets.
    """

    def __init__(
        self,
        context: PriorContext,
        *,
        artifact_path: Path | None = None,
    ) -> None:
        super().__init__(context)
        if artifact_path is None:
            raise RuntimeError("code_gadget prior requires `code_gadget_artifact_path`.")
        self.artifact_dir = Path(artifact_path).resolve()
        self.gadget_payload = self._load_gadget_payload(self.artifact_dir)
        self._prior_vec = self._build_prompt_prior()

    def vector(self, bin_idx: int, n_bins: int) -> np.ndarray:
        del bin_idx
        del n_bins
        return self._prior_vec.copy()

    @staticmethod
    def _load_gadget_payload(artifact_path: Path) -> dict[str, Any]:
        if artifact_path.is_dir():
            gadget_json = artifact_path / "gadget.json"
        else:
            gadget_json = artifact_path
        if not gadget_json.is_file():
            raise FileNotFoundError(f"Missing gadget artifact JSON: {gadget_json}")
        payload = json.loads(gadget_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Malformed gadget artifact payload at {gadget_json}")
        return payload

    @staticmethod
    def _line_char_spans(text: str) -> dict[int, tuple[int, int]]:
        spans: dict[int, tuple[int, int]] = {}
        cursor = 0
        lines = text.splitlines(keepends=True)
        if not lines and text:
            lines = [text]
        for line_idx, line in enumerate(lines, start=1):
            end = cursor + len(line)
            spans[line_idx] = (cursor, end)
            cursor = end
        if text.endswith(("\n", "\r")):
            spans[len(lines) + 1] = (cursor, cursor)
        return spans

    @staticmethod
    def _find_code_span(prompt_text: str, code_text: str) -> Optional[tuple[int, int]]:
        if not prompt_text or not code_text:
            return None
        start = prompt_text.find(code_text)
        if start >= 0:
            return start, start + len(code_text)
        stripped = code_text.strip("\n")
        if stripped:
            start = prompt_text.find(stripped)
            if start >= 0:
                return start, start + len(stripped)
        return None

    def _build_prompt_prior(self) -> np.ndarray:
        n_tokens = len(self.context.prompt_tokens)
        if n_tokens <= 0:
            return np.zeros((0,), dtype=float)

        offsets = list(self.context.prompt_token_offsets or [])
        if len(offsets) != n_tokens:
            return self._normalize(np.ones((n_tokens,), dtype=float))

        code_span = self._find_code_span(self.context.prompt_text, self.context.code_text)
        if code_span is None:
            return self._normalize(np.ones((n_tokens,), dtype=float))

        line_spans = self._line_char_spans(self.context.code_text)
        selected_lines = []
        for value in self.gadget_payload.get("line_sequence") or []:
            try:
                selected_lines.append(int(value))
            except Exception:
                continue
        if not selected_lines:
            raise ValueError("Gadget artifact is missing a usable `line_sequence`.")

        code_start, _ = code_span
        prompt_spans: list[tuple[int, int]] = []
        for line_idx in selected_lines:
            span = line_spans.get(line_idx)
            if span is None:
                continue
            start, end = span
            prompt_spans.append((code_start + start, code_start + end))
        if not prompt_spans:
            raise ValueError(
                "Gadget line sequence could not be aligned to the provided snippet text."
            )

        prior = np.zeros((n_tokens,), dtype=float)
        for token_idx, pair in enumerate(offsets):
            if pair is None or len(pair) != 2:
                continue
            tok_start, tok_end = int(pair[0]), int(pair[1])
            if tok_end <= tok_start:
                continue
            for span_start, span_end in prompt_spans:
                if tok_end > span_start and tok_start < span_end:
                    prior[token_idx] = 1.0
                    break

        if float(prior.sum()) <= 0.0:
            code_only = np.zeros((n_tokens,), dtype=float)
            code_start, code_end = code_span
            for token_idx, pair in enumerate(offsets):
                if pair is None or len(pair) != 2:
                    continue
                tok_start, tok_end = int(pair[0]), int(pair[1])
                if tok_end > code_start and tok_start < code_end:
                    code_only[token_idx] = 1.0
            prior = code_only

        if float(prior.sum()) <= 0.0:
            return self._normalize(np.ones((n_tokens,), dtype=float))
        return self._normalize(prior)


PRIOR_REGISTRY = {
    "code_gadget": CodeGadgetPrior,
}

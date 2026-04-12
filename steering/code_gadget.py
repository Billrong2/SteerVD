from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .joern_slice import (
    DEFAULT_CACHE_DIR as DEFAULT_JOERN_CACHE_DIR,
    DEFAULT_JOERN_CLI_DIR,
    extract_joern_variable_slices,
)


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "code_gadget"

CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
}
IDENTIFIER_BLACKLIST = CONTROL_KEYWORDS | {
    "true",
    "false",
    "null",
    "nullptr",
    "const",
    "static",
    "volatile",
    "unsigned",
    "signed",
    "struct",
    "class",
    "enum",
    "union",
    "void",
    "char",
    "short",
    "int",
    "long",
    "float",
    "double",
    "bool",
}
FORWARD_API_CALLS = {
    "recv",
    "recvfrom",
    "recvmsg",
    "read",
    "fread",
    "gets",
    "fgets",
    "getline",
    "getdelim",
    "scanf",
    "sscanf",
    "fscanf",
    "vscanf",
    "vfscanf",
    "vsscanf",
    "getenv",
}

CALL_NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*(?:(?:::|->|\.)[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
LINE_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)


@dataclass(frozen=True)
class CallCandidate:
    raw_name: str
    normalized_name: str
    line_number: int
    statement_text: str
    arg_texts: List[str]
    arg_identifiers: List[List[str]]


def _cache_key(
    *,
    code_text: str,
    default_direction: str,
    max_hops: Optional[int],
    slice_depth: int,
    parallelism: int,
    timeout_sec: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(code_text or "").encode("utf-8"))
    digest.update(str(default_direction or "").encode("utf-8"))
    digest.update(str(max_hops).encode("utf-8"))
    digest.update(str(int(slice_depth)).encode("utf-8"))
    digest.update(str(int(parallelism)).encode("utf-8"))
    digest.update(str(int(timeout_sec)).encode("utf-8"))
    return digest.hexdigest()


def _strip_comments(text: str) -> str:
    without_block = BLOCK_COMMENT_RE.sub("", str(text or ""))
    return LINE_COMMENT_RE.sub("", without_block)


def _source_lines(code_text: str) -> Dict[int, str]:
    return {idx: line.rstrip("\n") for idx, line in enumerate(str(code_text or "").splitlines(), start=1)}


def _normalize_call_name(raw_name: str) -> str:
    parts = re.split(r"::|->|\.", str(raw_name or ""))
    return parts[-1].strip() if parts else str(raw_name or "").strip()


def _normalize_identifier(raw_name: str) -> str:
    parts = re.split(r"::|->|\.", str(raw_name or ""))
    return parts[-1].strip().lower() if parts else str(raw_name or "").strip().lower()


def _find_matching_paren(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _split_top_level_args(text: str) -> List[str]:
    parts: List[str] = []
    depth_paren = depth_bracket = depth_brace = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_identifiers(expr: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for match in IDENT_RE.finditer(str(expr or "")):
        ident = match.group(0)
        lowered = ident.lower()
        if lowered in IDENTIFIER_BLACKLIST:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(ident)
    return out


def _looks_like_function_definition(statement_text: str) -> bool:
    stripped = str(statement_text or "").strip()
    if not stripped.endswith("{"):
        return False
    lowered = stripped.lower()
    if lowered.startswith(("if ", "if(", "for ", "for(", "while ", "while(", "switch ", "switch(", "catch ", "catch(")):
        return False
    if "=" in stripped and stripped.find("=") < stripped.find("("):
        return False
    return True


def _split_statements(code_text: str) -> List[Tuple[int, int, str]]:
    source_lines = str(code_text or "").splitlines()
    statements: List[Tuple[int, int, str]] = []
    buf: List[str] = []
    start_line: Optional[int] = None
    paren_depth = bracket_depth = brace_depth = 0

    for line_no, raw_line in enumerate(source_lines, start=1):
        line = raw_line.rstrip("\n")
        cleaned = _strip_comments(line)
        stripped = cleaned.strip()

        if start_line is None and stripped:
            start_line = line_no
        if start_line is None:
            continue

        buf.append(line)
        paren_depth += cleaned.count("(") - cleaned.count(")")
        bracket_depth += cleaned.count("[") - cleaned.count("]")
        brace_depth += cleaned.count("{") - cleaned.count("}")

        if stripped == "}":
            statements.append((start_line, line_no, "\n".join(buf).strip()))
            buf = []
            start_line = None
            continue

        should_close = False
        if paren_depth <= 0 and bracket_depth <= 0:
            if stripped.endswith(";"):
                should_close = True
            elif stripped.endswith("{") and _looks_like_function_definition("\n".join(buf)):
                should_close = True
        if should_close:
            statements.append((start_line, line_no, "\n".join(buf).strip()))
            buf = []
            start_line = None

    if buf and start_line is not None:
        statements.append((start_line, len(source_lines), "\n".join(buf).strip()))
    return statements


def _iter_call_candidates(code_text: str) -> Iterable[CallCandidate]:
    for start_line, _, statement_text in _split_statements(code_text):
        if not statement_text or _looks_like_function_definition(statement_text):
            continue
        cleaned = _strip_comments(statement_text)
        for match in CALL_NAME_RE.finditer(cleaned):
            raw_name = match.group(1).strip()
            normalized_name = _normalize_call_name(raw_name)
            if normalized_name.lower() in CONTROL_KEYWORDS:
                continue
            open_idx = cleaned.find("(", match.start())
            if open_idx < 0:
                continue
            close_idx = _find_matching_paren(cleaned, open_idx)
            if close_idx is None:
                continue
            arg_blob = cleaned[open_idx + 1 : close_idx]
            arg_texts = _split_top_level_args(arg_blob)
            call_line = start_line + cleaned[: match.start()].count("\n")
            yield CallCandidate(
                raw_name=raw_name,
                normalized_name=normalized_name,
                line_number=int(call_line),
                statement_text=statement_text,
                arg_texts=arg_texts,
                arg_identifiers=[_extract_identifiers(part) for part in arg_texts],
            )


def _call_direction(candidate: CallCandidate, *, default_direction: str) -> str:
    if candidate.normalized_name.lower() in FORWARD_API_CALLS:
        return "forward"
    return str(default_direction or "backward").lower()


def _build_line_scores_from_argument_slices(
    *,
    call_line: int,
    matched_slices: Sequence[Dict[str, Any]],
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for entry in matched_slices:
        for line_no, score in (entry.get("line_scores") or {}).items():
            key = int(line_no)
            scores[key] = float(scores.get(key, 0.0)) + float(score)
    scores[int(call_line)] = float(scores.get(int(call_line), 0.0)) + 1.0
    return {int(line_no): float(score) for line_no, score in sorted(scores.items()) if float(score) > 0.0}


def _slice_matches_argument(entry: Dict[str, Any], *, call_line: int, arg_identifiers: Sequence[str]) -> bool:
    if int(call_line) not in {int(x) for x in (entry.get("slice_lines") or []) if int(x) > 0}:
        return False
    if not arg_identifiers:
        return False
    variable_name = _normalize_identifier(str(entry.get("variable_name") or ""))
    if not variable_name:
        return False
    wanted = {_normalize_identifier(name) for name in arg_identifiers}
    return variable_name in wanted


def _summarize_slice_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "direction": payload.get("direction"),
        "num_graph_nodes": int(payload.get("num_graph_nodes", 0)),
        "num_graph_edges": int(payload.get("num_graph_edges", 0)),
        "num_variable_slices": int(payload.get("num_variable_slices", 0)),
        "num_selected_variable_slices": int(payload.get("num_selected_variable_slices", 0)),
        "meta": dict(payload.get("meta") or {}),
    }


def extract_code_gadget_payload(
    *,
    code_text: str,
    prompt_text: str = "",
    direction: str = "backward",
    joern_cli_dir: Path = DEFAULT_JOERN_CLI_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    joern_cache_dir: Path = DEFAULT_JOERN_CACHE_DIR,
    slice_depth: int = 20,
    parallelism: int = 1,
    timeout_sec: int = 180,
    include_control: bool = False,
    include_post_dominance: bool = False,
    max_hops: Optional[int] = None,
    sink_filter: Optional[str] = None,
) -> Dict[str, Any]:
    del sink_filter
    del include_post_dominance
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        _cache_key(
            code_text=code_text,
            default_direction=direction,
            max_hops=max_hops,
            slice_depth=slice_depth,
            parallelism=parallelism,
            timeout_sec=timeout_sec,
        )
        + ".json"
    )
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    source_map = _source_lines(code_text)
    call_candidates = list(_iter_call_candidates(code_text))
    directions_needed = sorted({_call_direction(candidate, default_direction=direction) for candidate in call_candidates} or {str(direction or "backward").lower()})

    payloads_by_direction: Dict[str, Dict[str, Any]] = {}
    joern_errors: Dict[str, str] = {}
    for active_direction in directions_needed:
        try:
            payloads_by_direction[active_direction] = extract_joern_variable_slices(
                code_text=str(code_text or ""),
                prompt_text=str(prompt_text or ""),
                direction=active_direction,
                joern_cli_dir=Path(joern_cli_dir),
                cache_dir=Path(joern_cache_dir),
                slice_depth=max(1, int(slice_depth)),
                parallelism=max(1, int(parallelism)),
                timeout_sec=max(1, int(timeout_sec)),
                include_control=bool(include_control),
                include_post_dominance=False,
                max_hops=None if max_hops is None else max(0, int(max_hops)),
                sink_filter=None,
                allow_empty=True,
            )
        except Exception as exc:
            joern_errors[active_direction] = str(exc)

    code_gadgets: List[Dict[str, Any]] = []
    for gadget_index, candidate in enumerate(call_candidates):
        active_direction = _call_direction(candidate, default_direction=direction)
        slice_payload = payloads_by_direction.get(active_direction, {})
        variable_slices = list(slice_payload.get("variable_slices") or [])

        matched_entries: List[Dict[str, Any]] = []
        argument_slice_counts: List[int] = []
        for arg_idents in candidate.arg_identifiers:
            matches: List[Dict[str, Any]] = []
            seen = set()
            for entry in variable_slices:
                if not isinstance(entry, dict):
                    continue
                if not _slice_matches_argument(entry, call_line=candidate.line_number, arg_identifiers=arg_idents):
                    continue
                key = (
                    str(entry.get("variable_key") or ""),
                    tuple(int(x) for x in (entry.get("anchor_lines") or []) if int(x) > 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                matches.append(entry)
            argument_slice_counts.append(int(len(matches)))
            matched_entries.extend(matches)

        line_scores = _build_line_scores_from_argument_slices(
            call_line=candidate.line_number,
            matched_slices=matched_entries,
        )
        line_sequence = sorted(set(int(candidate.line_number) for _ in [0]) | {int(line_no) for line_no in line_scores.keys() if int(line_no) > 0})
        code_gadget_text = "\n".join(source_map.get(line_no, "") for line_no in line_sequence if line_no in source_map)
        code_gadgets.append(
            {
                "gadget_index": int(gadget_index),
                "api_call_name": candidate.normalized_name,
                "raw_call_name": candidate.raw_name,
                "direction": active_direction,
                "call_line": int(candidate.line_number),
                "statement_text": candidate.statement_text,
                "arg_texts": list(candidate.arg_texts),
                "arg_identifiers": [list(names) for names in candidate.arg_identifiers],
                "argument_slice_counts": argument_slice_counts,
                "source_variable_slices": int(len(matched_entries)),
                "line_sequence": line_sequence,
                "line_scores": line_scores,
                "code_gadget": code_gadget_text,
            }
        )

    payload = {
        "direction": "mixed",
        "num_graph_nodes": int(max((payload.get("num_graph_nodes", 0) for payload in payloads_by_direction.values()), default=0)),
        "num_graph_edges": int(max((payload.get("num_graph_edges", 0) for payload in payloads_by_direction.values()), default=0)),
        "num_variable_slices": int(sum(int(payload.get("num_variable_slices", 0)) for payload in payloads_by_direction.values())),
        "num_selected_variable_slices": int(sum(int(payload.get("num_selected_variable_slices", 0)) for payload in payloads_by_direction.values())),
        "sink_filter": "",
        "sink_node_ids": [],
        "aggregate_line_scores": {},
        "variable_slices": [],
        "code_gadgets": code_gadgets,
        "selected_code_gadget": None,
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "code_gadget",
            "source_backend": "joern_slice",
            "construction": "vuldeepecker",
            "default_direction": str(direction or "backward").lower(),
            "include_control": bool(include_control),
            "gadget_count": int(len(code_gadgets)),
            "call_count": int(len(call_candidates)),
            "joern_errors": dict(joern_errors),
            "per_direction_summaries": {
                active_direction: _summarize_slice_payload(payload)
                for active_direction, payload in payloads_by_direction.items()
            },
            "slice_empty": len(code_gadgets) == 0,
            "empty_reason": None if code_gadgets else "no-api-call-gadgets",
        },
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

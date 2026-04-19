#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_vulnerable_snippets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_vulnerable_arg_slices"
STATE_FILENAME = "derive_state.json"
SUMMARY_FILENAME = "derive_summary.json"


@dataclass(frozen=True)
class ParentGadgetTarget:
    snippet_idx: str
    snippet_dir: Path
    gadget_dir: Path
    gadget_relpath: str
    gadget_index: int
    api_call_name: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _numeric_name_key(path: Path) -> Tuple[int, str]:
    parsed = _safe_int(path.name)
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:020d}")


def _gadget_key(path: Path) -> Tuple[int, str, str]:
    prefix = path.name.split("__", 1)[0]
    index_text = prefix.replace("gadget_", "", 1)
    parsed = _safe_int(index_text)
    if parsed is None:
        return (1, path.name, path.name)
    return (0, f"{parsed:06d}", path.name)


def _sanitize_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "", str(value or "").strip().lower())
    return text or "unknown_call"


def _write_state(path: Path, payload: Dict[str, Any]) -> None:
    state = dict(payload)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path, state)


def _iter_parent_gadgets(input_root: Path) -> Iterator[ParentGadgetTarget]:
    for snippet_dir in sorted((path for path in input_root.iterdir() if path.is_dir()), key=_numeric_name_key):
        status_path = snippet_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = _load_json(status_path)
        except Exception:
            continue
        if not isinstance(status, dict) or status.get("status") != "ok":
            continue

        snippet_idx = str(status.get("dataset_idx") or snippet_dir.name)
        for gadget_dir in sorted(
            (path for path in snippet_dir.iterdir() if path.is_dir() and path.name.startswith("gadget_")),
            key=_gadget_key,
        ):
            gadget_json_path = gadget_dir / "gadget.json"
            code_gadget_path = gadget_dir / "code_gadget.c"
            if not gadget_json_path.is_file() or not code_gadget_path.is_file():
                continue
            try:
                gadget_payload = _load_json(gadget_json_path)
            except Exception:
                continue
            if not isinstance(gadget_payload, dict):
                continue
            yield ParentGadgetTarget(
                snippet_idx=snippet_idx,
                snippet_dir=snippet_dir,
                gadget_dir=gadget_dir,
                gadget_relpath=str(gadget_dir.relative_to(input_root)),
                gadget_index=int(gadget_payload.get("gadget_index", 0)),
                api_call_name=str(gadget_payload.get("api_call_name") or "unknown_call"),
            )


def _load_snippet_lines(snippet_dir: Path) -> List[str]:
    snippet_path = snippet_dir / "snippet.c"
    if snippet_path.is_file():
        return snippet_path.read_text(encoding="utf-8").splitlines()
    row = _load_json(snippet_dir / "row.json")
    for key in ("func", "func_before", "code"):
        value = row.get(key)
        if value:
            return str(value).splitlines()
    return []


def _resolve_arg_index(argument_slice: Dict[str, Any], fallback_index: int) -> int:
    arg_text = str(argument_slice.get("arg_text") or "")
    if arg_text == "<return>":
        return 0
    for entry in argument_slice.get("flow_groups") or []:
        parsed = _safe_int(entry.get("arg_index"))
        if parsed is not None:
            return parsed
    return int(fallback_index)


def _normalized_unique_line_sequence(values: Sequence[Any], *, max_line: int) -> List[int]:
    ordered: List[int] = []
    seen: set[int] = set()
    for value in values:
        parsed = _safe_int(value)
        if parsed is None or parsed <= 0 or parsed > max_line:
            continue
        if parsed in seen:
            continue
        seen.add(parsed)
        ordered.append(int(parsed))
    ordered.sort()
    return ordered


def _snippet_span(gadget_payload: Dict[str, Any], *, snippet_line_count: int) -> Dict[str, int]:
    payload = gadget_payload.get("snippet_span") or {}
    start_line = _safe_int(payload.get("start_line")) or 1
    end_line = _safe_int(payload.get("end_line")) or snippet_line_count
    start_line = max(1, min(start_line, max(1, snippet_line_count)))
    end_line = max(start_line, min(end_line, max(1, snippet_line_count)))
    return {"start_line": start_line, "end_line": end_line}


def _coverage_from_lines(
    *,
    line_sequence: Sequence[int],
    snippet_lines: Sequence[str],
    snippet_span: Dict[str, int],
) -> Tuple[float, float]:
    start_line = int(snippet_span["start_line"])
    end_line = int(snippet_span["end_line"])
    nonempty_total = 0
    for line_no in range(start_line, end_line + 1):
        if 1 <= line_no <= len(snippet_lines) and str(snippet_lines[line_no - 1]).strip():
            nonempty_total += 1

    selected_nonempty = 0
    seen: set[int] = set()
    for line_no in line_sequence:
        if line_no in seen:
            continue
        seen.add(line_no)
        if not (start_line <= int(line_no) <= end_line):
            continue
        if 1 <= int(line_no) <= len(snippet_lines) and str(snippet_lines[int(line_no) - 1]).strip():
            selected_nonempty += 1

    coverage = round(float(selected_nonempty) / float(max(1, nonempty_total)), 6)
    coverage_weight = round(max(0.1, 1.0 - coverage), 6)
    return coverage, coverage_weight


def _render_code_gadget(line_sequence: Sequence[int], snippet_lines: Sequence[str]) -> str:
    rendered: List[str] = []
    for line_no in line_sequence:
        if 1 <= int(line_no) <= len(snippet_lines):
            rendered.append(snippet_lines[int(line_no) - 1])
    return "\n".join(rendered)


def _slice_dir_name(*, parent_gadget_index: int, arg_index: int, api_call_name: str, call_line: Optional[int]) -> str:
    call_slug = _sanitize_name(api_call_name)
    line_part = f"line_{int(call_line)}" if call_line is not None else "line_unknown"
    return f"slice_{int(parent_gadget_index):03d}__arg_{int(arg_index)}__{call_slug}__{line_part}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive per-argument slice steering units from exported Joern gadgets.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", choices=["on", "off"], default="on")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_every = max(1, int(args.checkpoint_every))
    state_path = output_root / STATE_FILENAME
    summary_path = output_root / SUMMARY_FILENAME
    resume = args.resume == "on"

    parents = list(_iter_parent_gadgets(input_root))
    offset = max(0, int(args.offset))
    if offset:
        parents = parents[offset:]
    if args.limit is not None:
        parents = parents[: max(0, int(args.limit))]

    state: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "resume": bool(resume),
        "target_total": int(len(parents)),
        "processed_parents": 0,
        "written_slices": 0,
        "skipped_existing": 0,
        "skipped_empty": 0,
        "last_snippet_idx": None,
        "last_parent_gadget_relpath": None,
    }

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(parents), desc="arg-slice-derive", unit="gadget")
        progress.set_postfix(written=0, skipped_existing=0, skipped_empty=0, snippet="-", gadget="-")

    try:
        for parent in parents:
            row_path = parent.snippet_dir / "row.json"
            status_path = parent.snippet_dir / "status.json"
            snippet_path = parent.snippet_dir / "snippet.c"
            parent_payload = dict(_load_json(parent.gadget_dir / "gadget.json"))
            snippet_lines = _load_snippet_lines(parent.snippet_dir)
            snippet_span = _snippet_span(parent_payload, snippet_line_count=len(snippet_lines))
            call_line = _safe_int(parent_payload.get("call_line"))
            output_snippet_dir = output_root / str(parent.snippet_idx)
            output_snippet_dir.mkdir(parents=True, exist_ok=True)
            if row_path.is_file():
                (output_snippet_dir / "row.json").write_text(row_path.read_text(encoding="utf-8"), encoding="utf-8")
            if status_path.is_file():
                (output_snippet_dir / "status.json").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
            if snippet_path.is_file():
                (output_snippet_dir / "snippet.c").write_text(snippet_path.read_text(encoding="utf-8"), encoding="utf-8")

            state["last_snippet_idx"] = str(parent.snippet_idx)
            state["last_parent_gadget_relpath"] = str(parent.gadget_relpath)

            for fallback_index, argument_slice in enumerate(parent_payload.get("argument_slices") or [], start=1):
                if not isinstance(argument_slice, dict):
                    continue
                arg_index = _resolve_arg_index(argument_slice, fallback_index)
                line_sequence = _normalized_unique_line_sequence(
                    argument_slice.get("line_sequence") or [],
                    max_line=len(snippet_lines),
                )
                if not line_sequence:
                    state["skipped_empty"] = int(state["skipped_empty"]) + 1
                    continue
                code_gadget = _render_code_gadget(line_sequence, snippet_lines)
                if not code_gadget.strip():
                    state["skipped_empty"] = int(state["skipped_empty"]) + 1
                    continue

                coverage, coverage_weight = _coverage_from_lines(
                    line_sequence=line_sequence,
                    snippet_lines=snippet_lines,
                    snippet_span=snippet_span,
                )
                slice_dir = output_snippet_dir / _slice_dir_name(
                    parent_gadget_index=parent.gadget_index,
                    arg_index=arg_index,
                    api_call_name=parent.api_call_name,
                    call_line=call_line,
                )
                gadget_json_path = slice_dir / "gadget.json"
                if resume and gadget_json_path.is_file():
                    state["skipped_existing"] = int(state["skipped_existing"]) + 1
                    continue

                slice_dir.mkdir(parents=True, exist_ok=True)
                slice_payload = {
                    "gadget_index": int(parent.gadget_index),
                    "unit_type": "argument_slice",
                    "parent_gadget_relpath": str(parent.gadget_relpath),
                    "parent_gadget_index": int(parent.gadget_index),
                    "api_call_name": str(parent_payload.get("api_call_name") or parent.api_call_name),
                    "raw_call_name": parent_payload.get("raw_call_name"),
                    "direction": str(parent_payload.get("direction") or "backward"),
                    "call_line": call_line,
                    "statement_text": parent_payload.get("statement_text"),
                    "arg_index": int(arg_index),
                    "arg_text": str(argument_slice.get("arg_text") or ""),
                    "arg_identifiers": list(argument_slice.get("arg_identifiers") or []),
                    "line_sequence": line_sequence,
                    "flow_count": int(_safe_int(argument_slice.get("flow_count")) or 0),
                    "flow_groups": list(argument_slice.get("flow_groups") or []),
                    "snippet_span": snippet_span,
                    "coverage": coverage,
                    "coverage_weight": coverage_weight,
                    "code_gadget": code_gadget,
                    "source_path": parent_payload.get("source_path"),
                }
                _write_json(gadget_json_path, slice_payload)
                (slice_dir / "code_gadget.c").write_text(code_gadget + "\n", encoding="utf-8")
                state["written_slices"] = int(state["written_slices"]) + 1

            state["processed_parents"] = int(state["processed_parents"]) + 1
            if progress is not None:
                progress.set_postfix(
                    written=int(state["written_slices"]),
                    skipped_existing=int(state["skipped_existing"]),
                    skipped_empty=int(state["skipped_empty"]),
                    snippet=parent.snippet_idx,
                    gadget=parent.gadget_index,
                )
                progress.update(1)
            if int(state["processed_parents"]) % checkpoint_every == 0:
                _write_state(state_path, state)
    finally:
        if progress is not None:
            progress.close()

    _write_state(state_path, state)
    summary = {
        "run": state,
        "counts": {
            "processed_parents": int(state["processed_parents"]),
            "written_slices": int(state["written_slices"]),
            "skipped_existing": int(state["skipped_existing"]),
            "skipped_empty": int(state["skipped_empty"]),
        },
    }
    _write_json(summary_path, summary)
    print(
        f"[arg-slice-derive] done processed_parents={state['processed_parents']} "
        f"written_slices={state['written_slices']} skipped_existing={state['skipped_existing']} "
        f"skipped_empty={state['skipped_empty']} output_root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

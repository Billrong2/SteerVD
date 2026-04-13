#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency at runtime
    tqdm = None

from steering.code_gadget import extract_code_gadget_payload


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Source" / "primevul_test.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_snippets"
DEFAULT_JOERN_CLI_DIR = Path("/people/cs/x/xxr230000/bin/joern/joern-cli")
DEFAULT_JOERN_CACHE_DIR = PROJECT_ROOT / ".cache" / "joern_slice"
DEFAULT_CODE_GADGET_CACHE_DIR = PROJECT_ROOT / ".cache" / "code_gadget"
DEFAULT_PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache" / "project_context"
DEFAULT_PROJECT_REPO_MAP_PATH = PROJECT_ROOT / "Source" / "project_repo_urls.json"
STATE_FILENAME = "export_state.json"
SUMMARY_FILENAME = "export_summary.json"


def _row_matches_target(row: Dict[str, Any], *, target_filter: str) -> bool:
    if target_filter == "all":
        return True
    target_value = row.get("target")
    try:
        target_int = int(target_value)
    except Exception:
        return False
    if target_filter == "vulnerable":
        return target_int == 1
    if target_filter == "safe":
        return target_int == 0
    raise ValueError(f"Unsupported target_filter: {target_filter!r}")


def _iter_jsonl_rows(path: Path, *, offset: int, limit: int | None, target_filter: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    exported = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < offset:
                continue
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Row {row_index} in {path} is not a JSON object.")
            if not _row_matches_target(payload, target_filter=target_filter):
                continue
            yield row_index, dict(payload)
            exported += 1
            if limit is not None and exported >= limit:
                break


def _count_jsonl_rows(path: Path, *, offset: int, limit: int | None, target_filter: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < offset:
                continue
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Row {row_index} in {path} is not a JSON object.")
            if not _row_matches_target(payload, target_filter=target_filter):
                continue
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _pick_code(row: Dict[str, Any]) -> str:
    for key in ("func", "func_before", "code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"No code field found in row keys: {list(row.keys())!r}")


def _snippet_folder_name(row: Dict[str, Any], *, row_index: int) -> str:
    value = row.get("idx")
    if value is None:
        return f"missing_idx__row_{row_index:05d}"
    text = str(value).strip()
    if not text:
        return f"missing_idx__row_{row_index:05d}"
    return text


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_call"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(str(text or ""), encoding="utf-8")


def _prepare_snippet_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": None,
            "dataset_path": None,
            "output_root": None,
            "strict_project_context": None,
            "target_filter": None,
            "processed": 0,
            "ok": 0,
            "empty": 0,
            "error": 0,
            "completed_snippets": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid exporter state file: {path}")
    payload.setdefault("completed_snippets", [])
    return payload


def _write_state(path: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(path, state)


def _write_summary(path: Path, state: Dict[str, Any]) -> None:
    summary = {
        "dataset_path": state.get("dataset_path"),
        "output_root": state.get("output_root"),
        "backend": state.get("backend"),
        "strict_project_context": state.get("strict_project_context"),
        "target_filter": state.get("target_filter"),
        "processed": int(state.get("processed", 0)),
        "ok": int(state.get("ok", 0)),
        "empty": int(state.get("empty", 0)),
        "error": int(state.get("error", 0)),
        "completed_snippet_count": int(len(state.get("completed_snippets") or [])),
        "updated_at": state.get("updated_at"),
    }
    _write_json(path, summary)


def _write_gadget_bundle(snippet_dir: Path, gadget: Dict[str, Any], *, fallback_index: int) -> None:
    gadget_index = int(gadget.get("gadget_index", fallback_index))
    call_name = _slugify(gadget.get("api_call_name"))
    call_line = int(gadget.get("call_line", -1))
    gadget_dir = snippet_dir / f"gadget_{gadget_index:03d}__{call_name}__line_{call_line}"
    gadget_dir.mkdir(parents=True, exist_ok=True)

    _write_json(gadget_dir / "gadget.json", gadget)
    _write_text(gadget_dir / "code_gadget.c", str(gadget.get("code_gadget") or ""))
    _write_text(gadget_dir / "symbolic_code_gadget.c", str(gadget.get("symbolic_code_gadget") or ""))
    _write_json(gadget_dir / "symbolic_tokens.json", gadget.get("symbolic_tokens") or [])
    _write_json(gadget_dir / "argument_slices.json", gadget.get("argument_slices") or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-snippet PrimeVul code gadgets into snippet-specific folders.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--joern-cli-dir", type=Path, default=DEFAULT_JOERN_CLI_DIR)
    parser.add_argument("--joern-cache-dir", type=Path, default=DEFAULT_JOERN_CACHE_DIR)
    parser.add_argument("--code-gadget-cache-dir", type=Path, default=DEFAULT_CODE_GADGET_CACHE_DIR)
    parser.add_argument("--project-source-root", type=Path, default=None)
    parser.add_argument("--project-cache-dir", type=Path, default=DEFAULT_PROJECT_CACHE_DIR)
    parser.add_argument("--project-repo-map-path", type=Path, default=DEFAULT_PROJECT_REPO_MAP_PATH)
    parser.add_argument("--strict-project-context", choices=["on", "off"], default="on")
    parser.add_argument("--target-filter", choices=["all", "vulnerable", "safe"], default="all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", choices=["on", "off"], default="on")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()
    output_root = args.output_root.resolve()
    strict_project_context = args.strict_project_context == "on"

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_FILENAME
    summary_path = output_root / SUMMARY_FILENAME
    checkpoint_every = max(1, int(args.checkpoint_every))

    if args.resume == "on":
        state = _load_state(state_path)
    else:
        state = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": None,
            "dataset_path": None,
            "output_root": None,
            "strict_project_context": None,
            "target_filter": None,
            "processed": 0,
            "ok": 0,
            "empty": 0,
            "error": 0,
            "completed_snippets": [],
        }

    state["dataset_path"] = str(dataset_path)
    state["output_root"] = str(output_root)
    state["backend"] = "joern"
    state["strict_project_context"] = bool(strict_project_context)
    state["target_filter"] = str(args.target_filter)
    completed_snippets: Set[str] = {
        str(item) for item in (state.get("completed_snippets") or []) if str(item).strip()
    }

    processed = int(state.get("processed", 0))
    ok_count = int(state.get("ok", 0))
    empty_count = int(state.get("empty", 0))
    error_count = int(state.get("error", 0))
    total_rows = _count_jsonl_rows(
        dataset_path,
        offset=max(0, int(args.offset)),
        limit=args.limit,
        target_filter=str(args.target_filter),
    )
    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=total_rows,
            initial=min(processed, total_rows),
            desc="code-gadget-export",
            unit="snippet",
        )
        progress.set_postfix(ok=ok_count, empty=empty_count, error=error_count)

    try:
        for row_index, row in _iter_jsonl_rows(
            dataset_path,
            offset=max(0, int(args.offset)),
            limit=args.limit,
            target_filter=str(args.target_filter),
        ):
            snippet_name = _snippet_folder_name(row, row_index=row_index)
            if snippet_name in completed_snippets:
                continue
            snippet_dir = output_root / snippet_name
            _prepare_snippet_dir(snippet_dir)

            code_text = ""
            try:
                code_text = _pick_code(row)
                _write_json(snippet_dir / "row.json", row)
                _write_text(snippet_dir / "snippet.c", code_text)

                payload = extract_code_gadget_payload(
                    code_text=code_text,
                    project_name=str(row.get("project") or ""),
                    commit_id=str(row.get("commit_id") or ""),
                    project_source_root=args.project_source_root,
                    project_cache_dir=args.project_cache_dir,
                    project_repo_map_path=args.project_repo_map_path,
                    strict_project_context=bool(strict_project_context),
                    prompt_text="```c\n" + code_text + "\n```",
                    joern_cli_dir=args.joern_cli_dir,
                    joern_cache_dir=args.joern_cache_dir,
                    cache_dir=args.code_gadget_cache_dir,
                )

                meta = dict(payload.get("meta") or {})
                gadgets: List[Dict[str, Any]] = [
                    dict(item) for item in (payload.get("code_gadgets") or []) if isinstance(item, dict)
                ]

                _write_json(snippet_dir / "meta.json", meta)
                for fallback_index, gadget in enumerate(gadgets):
                    _write_gadget_bundle(snippet_dir, gadget, fallback_index=fallback_index)

                status = {
                    "status": "ok" if gadgets else "empty",
                    "row_index": int(row_index),
                    "snippet_name": snippet_name,
                    "dataset_idx": row.get("idx"),
                    "gadget_count": int(len(gadgets)),
                    "strict_project_context": bool(strict_project_context),
                    "empty_reason": meta.get("empty_reason"),
                }
                _write_json(snippet_dir / "status.json", status)
                if gadgets:
                    ok_count += 1
                else:
                    empty_count += 1
                completed_snippets.add(snippet_name)
            except Exception as exc:
                error_payload = {
                    "row_index": int(row_index),
                    "snippet_name": snippet_name,
                    "dataset_idx": row.get("idx"),
                    "project": row.get("project"),
                    "commit_id": row.get("commit_id"),
                    "strict_project_context": bool(strict_project_context),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                if not (snippet_dir / "row.json").exists():
                    _write_json(snippet_dir / "row.json", row)
                if code_text and not (snippet_dir / "snippet.c").exists():
                    _write_text(snippet_dir / "snippet.c", code_text)
                _write_json(snippet_dir / "error.json", error_payload)
                _write_json(
                    snippet_dir / "status.json",
                    {
                        "status": "error",
                        "row_index": int(row_index),
                        "snippet_name": snippet_name,
                        "dataset_idx": row.get("idx"),
                        "gadget_count": 0,
                        "strict_project_context": bool(strict_project_context),
                    },
                )
                error_count += 1
                completed_snippets.add(snippet_name)

            processed += 1
            state["processed"] = int(processed)
            state["ok"] = int(ok_count)
            state["empty"] = int(empty_count)
            state["error"] = int(error_count)
            state["completed_snippets"] = sorted(completed_snippets)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(ok=ok_count, empty=empty_count, error=error_count)
            if processed % checkpoint_every == 0:
                _write_state(state_path, state)
                _write_summary(summary_path, state)
                if progress is None:
                    print(
                        f"[export] processed={processed} ok={ok_count} empty={empty_count} error={error_count}",
                        flush=True,
                    )
    finally:
        if progress is not None:
            progress.close()

    _write_state(state_path, state)
    _write_summary(summary_path, state)
    print(
        f"[export] done processed={processed} ok={ok_count} empty={empty_count} error={error_count} "
        f"output_root={output_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

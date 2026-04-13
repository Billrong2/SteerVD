#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from steering.code_gadget import extract_code_gadget_payload


def _load_jsonl_row(path: Path, row_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx != row_index:
                continue
            return dict(json.loads(line))
    raise IndexError(f"Row index {row_index} is out of range for {path}.")


def _pick_code(row: Dict[str, Any]) -> str:
    for key in ("func", "func_before", "code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"No code field found in row keys: {list(row.keys())!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect VulDeePecker-style code gadgets for a PrimeVul row.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--joern-cli-dir", type=Path, default=Path("/people/cs/x/xxr230000/bin/joern/joern-cli"))
    parser.add_argument("--joern-cache-dir", type=Path, default=Path(".cache/joern_slice"))
    parser.add_argument("--code-gadget-cache-dir", type=Path, default=Path(".cache/code_gadget"))
    parser.add_argument("--project-source-root", type=Path, default=None)
    parser.add_argument("--project-cache-dir", type=Path, default=Path(".cache/project_context"))
    parser.add_argument("--project-repo-map-path", type=Path, default=Path("Source/project_repo_urls.json"))
    parser.add_argument("--strict-project-context", choices=["on", "off"], default="on")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row = _load_jsonl_row(args.dataset_path, args.row_index)
    code = _pick_code(row)
    payload = extract_code_gadget_payload(
        code_text=code,
        project_name=str(row.get("project") or ""),
        commit_id=str(row.get("commit_id") or ""),
        project_source_root=args.project_source_root,
        project_cache_dir=args.project_cache_dir,
        project_repo_map_path=args.project_repo_map_path,
        strict_project_context=(args.strict_project_context == "on"),
        prompt_text="```c\n" + code + "\n```",
        joern_cli_dir=args.joern_cli_dir,
        joern_cache_dir=args.joern_cache_dir,
        cache_dir=args.code_gadget_cache_dir,
    )
    print(json.dumps(payload.get("meta") or {}, indent=2))
    gadgets: List[Dict[str, Any]] = [dict(item) for item in (payload.get("code_gadgets") or []) if isinstance(item, dict)]
    print(f"\ngadget_count={len(gadgets)}")
    for gadget in gadgets:
        print("\n" + "=" * 80)
        print(
            f"gadget_index={gadget.get('gadget_index')} "
            f"call={gadget.get('api_call_name')} "
            f"direction={gadget.get('direction')} "
            f"call_line={gadget.get('call_line')}"
        )
        print(f"arg_texts={gadget.get('arg_texts')}")
        print(
            "argument_slice_group_counts="
            f"{[len(item.get('flow_groups') or []) for item in (gadget.get('argument_slices') or []) if isinstance(item, dict)]}"
        )
        print("--- code_gadget ---")
        print(str(gadget.get("code_gadget") or ""))
        print("--- symbolic_code_gadget ---")
        print(str(gadget.get("symbolic_code_gadget") or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

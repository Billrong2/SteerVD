#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "Source" / "primevul_test_vulnerable_arg_slices"
DEFAULT_ALL_SNAPSHOT = PROJECT_ROOT / "artifacts" / "checkpoints" / "arg_slice_snapshot_all.json"
DEFAULT_POSITIVE_SNAPSHOT = PROJECT_ROOT / "artifacts" / "checkpoints" / "arg_slice_snapshot_positive.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_int(value: Any):
    try:
        return int(value)
    except Exception:
        return None


def _numeric_name_key(path: Path) -> Tuple[int, str]:
    parsed = _safe_int(path.name)
    if parsed is None:
        return (1, path.name)
    return (0, f"{parsed:020d}")


def _unit_key(path: Path) -> Tuple[int, str, str]:
    prefix = path.name.split("__", 1)[0]
    index_text = prefix
    for marker in ("gadget_", "slice_"):
        if index_text.startswith(marker):
            index_text = index_text.replace(marker, "", 1)
            break
    parsed = _safe_int(index_text)
    if parsed is None:
        return (1, path.name, path.name)
    return (0, f"{parsed:06d}", path.name)


def _iter_unit_dirs(input_root: Path) -> Iterator[Path]:
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
        for unit_dir in sorted((path for path in snippet_dir.iterdir() if path.is_dir()), key=_unit_key):
            if (unit_dir / "gadget.json").is_file():
                yield unit_dir


def _build_snapshot(input_root: Path) -> Tuple[List[str], List[str]]:
    all_relpaths: List[str] = []
    positive_relpaths: List[str] = []
    for unit_dir in _iter_unit_dirs(input_root):
        relpath = str(unit_dir.relative_to(input_root))
        all_relpaths.append(relpath)
        label_path = unit_dir / "gadget_label.json"
        if not label_path.is_file():
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            continue
        if str(label.get("pred_label") or "").strip().upper() == "VULNERABLE":
            positive_relpaths.append(relpath)
    return all_relpaths, positive_relpaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze all/positive relpath snapshots from labeled gadget or slice trees.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--all-snapshot-path", type=Path, default=DEFAULT_ALL_SNAPSHOT)
    parser.add_argument("--positive-snapshot-path", type=Path, default=DEFAULT_POSITIVE_SNAPSHOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    all_snapshot_path = args.all_snapshot_path.resolve()
    positive_snapshot_path = args.positive_snapshot_path.resolve()

    all_relpaths, positive_relpaths = _build_snapshot(input_root)
    common_meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_root": str(input_root),
    }
    _write_json(
        all_snapshot_path,
        {
            **common_meta,
            "label_filter": "all",
            "target_total": int(len(all_relpaths)),
            "gadget_relpaths": all_relpaths,
        },
    )
    _write_json(
        positive_snapshot_path,
        {
            **common_meta,
            "label_filter": "vulnerable",
            "target_total": int(len(positive_relpaths)),
            "gadget_relpaths": positive_relpaths,
        },
    )
    print(
        f"[freeze-snapshots] all={len(all_relpaths)} positive={len(positive_relpaths)} "
        f"input_root={input_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

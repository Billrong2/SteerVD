#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "primevul"
    / "Qwen_Qwen2.5-Coder-7B-Instruct"
    / "qwen25_primevul_full_baseline_revdcot"
    / "predictions.jsonl"
)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "Source" / "primevul_test.jsonl"

PAPER_BASELINES: Mapping[str, Dict[str, Any]] = {
    "revd_qwen25_cot_primevul": {
        "model": "Qwen2.5-Coder-7B-Instruct",
        "setting": "COT baseline",
        "accuracy": 49.77,
        "f1": 29.86,
    }
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a PrimeVul run and print the paper baseline plus per-trial accuracy."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to predictions.jsonl from a PrimeVul run.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the PrimeVul test split to report coverage against the full test set.",
    )
    parser.add_argument(
        "--paper-baseline",
        type=str,
        default="revd_qwen25_cot_primevul",
        choices=sorted(PAPER_BASELINES.keys()),
        help="Which paper baseline to print alongside the current run.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _compute_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    tp = fp = tn = fn = 0
    for row in records:
        gold = int(row["gold_label"])
        pred = row.get("pred_label")
        if pred is None:
            if gold == 1:
                fn += 1
            else:
                fp += 1
            continue
        pred = int(pred)
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _print_paper_baseline(baseline: Mapping[str, Any]) -> None:
    print("Paper baseline")
    print(f"  model: {baseline['model']}")
    print(f"  setting: {baseline['setting']}")
    print(f"  accuracy: {baseline['accuracy']:.2f}%")
    print(f"  f1: {baseline['f1']:.2f}%")
    print()


def _trial_key(record: Mapping[str, Any]) -> int:
    return int(record.get("sample_run", 0))


def _records_by_trial(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_trial_key(row)].append(row)
    return dict(grouped)


def _print_trial_metrics(
    *,
    label: str,
    records: Sequence[Dict[str, Any]],
    full_test_total: int | None,
) -> None:
    metrics = _compute_metrics(records)
    unique_snippets = len({str(row.get("sample_id")) for row in records})
    print(label)
    print(f"  snippets: {unique_snippets}")
    if full_test_total:
        print(f"  {unique_snippets}/{full_test_total}")
    print(f"  accuracy: {_percent(metrics['accuracy'])}")
    print(f"  f1: {_percent(metrics['f1'])}")
    print()


def main() -> None:
    args = _parse_args()
    predictions_path = args.predictions_path.resolve()
    dataset_path = args.dataset_path.resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    records = _load_jsonl(predictions_path)
    if not records:
        raise RuntimeError(f"No prediction rows found in {predictions_path}")

    baseline = PAPER_BASELINES[args.paper_baseline]
    full_test_total = _count_jsonl_rows(dataset_path)
    by_trial = _records_by_trial(records)

    _print_paper_baseline(baseline)

    _print_trial_metrics(
        label="Trial 1",
        records=by_trial.get(0, []),
        full_test_total=full_test_total,
    )
    _print_trial_metrics(
        label="Trial 2",
        records=by_trial.get(1, []),
        full_test_total=full_test_total,
    )
    _print_trial_metrics(
        label="All generated trials combined",
        records=records,
        full_test_total=full_test_total,
    )


if __name__ == "__main__":
    main()

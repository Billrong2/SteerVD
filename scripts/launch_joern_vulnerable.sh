#!/usr/bin/env bash
set -euo pipefail

python3 /home/cs/x/xxr230000/Steer_VD/primevul_export_code_gadgets.py \
  --dataset-path /home/cs/x/xxr230000/Steer_VD/Source/primevul_test.jsonl \
  --output-root /home/cs/x/xxr230000/Steer_VD/Source/primevul_test_vulnerable_snippets \
  --strict-project-context off \
  --target-filter vulnerable \
  --resume on \
  --checkpoint-every 25

#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python3 /home/cs/x/xxr230000/Steer_VD/primevul_steer_gadgets.py \
  --variant baseline \
  --input-root /home/cs/x/xxr230000/Steer_VD/Source/primevul_test_vulnerable_snippets \
  --snapshot-path /home/cs/x/xxr230000/Steer_VD/artifacts/checkpoints/primevul_gadget_snapshot_9472.json \
  --label-filter all \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --protocol revd_cot \
  --output-dir primevul_gadget_baseline \
  --run-name gadget_baseline_snapshot9472_revdcot_t768 \
  --samples-per-gadget 1 \
  --max-new-tokens 768 \
  --temperature 0.0 \
  --top-p 1.0 \
  --resume on \
  --checkpoint-every 25

#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=2 python3 /home/cs/x/xxr230000/Steer_VD/primevul_steer_gadgets.py \
  --input-root /home/cs/x/xxr230000/Steer_VD/Source/primevul_test_vulnerable_snippets \
  --label-filter all \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --protocol revd_cot \
  --run-name gadget_steered_all_revdcot_s2_l8_k4_b0p5_n12_full \
  --n-bins 12 \
  --beta-post 0.5 \
  --beta-bias 0.0 \
  --steer-last-n-layers 8 \
  --head-subset-mode auto \
  --head-subset-topk-per-layer 4 \
  --head-subset-calib-runs 1 \
  --head-subset-calib-max-new-tokens 1 \
  --samples-per-gadget 1 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --top-p 1.0 \
  --resume on \
  --checkpoint-every 25

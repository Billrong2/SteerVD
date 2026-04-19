#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=3 python3 /home/cs/x/xxr230000/Steer_VD/primevul_steer_gadgets.py \
  --variant steered \
  --input-root /home/cs/x/xxr230000/Steer_VD/Source/primevul_test_vulnerable_arg_slices \
  --snapshot-path /home/cs/x/xxr230000/Steer_VD/artifacts/checkpoints/arg_slice_snapshot_positive.json \
  --label-filter all \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --protocol revd_cot \
  --output-dir primevul_arg_slice_steered \
  --run-name arg_slice_steered_positive_revdcot_s2_l8_k4_b0p8_n12_t768 \
  --n-bins 12 \
  --beta-post 0.8 \
  --beta-bias 0.0 \
  --steer-last-n-layers 8 \
  --head-subset-mode auto \
  --head-subset-topk-per-layer 4 \
  --head-subset-calib-runs 1 \
  --head-subset-calib-max-new-tokens 1 \
  --samples-per-gadget 1 \
  --max-new-tokens 768 \
  --temperature 0.0 \
  --top-p 1.0 \
  --resume on \
  --checkpoint-every 25

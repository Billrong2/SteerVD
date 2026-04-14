#!/usr/bin/env bash
set -euo pipefail

python3 /home/cs/x/xxr230000/Steer_VD/primevul_steer_gadgets.py \
  --input-root /home/cs/x/xxr230000/Steer_VD/Source/primevul_test_vulnerable_snippets \
  --label-filter vulnerable \
  --protocol revd_cot \
  --run-name gadget_steered_vulnerable_revdcot_s2_l8_k4_b0p5_n12 \
  --n-bins 12 \
  --beta-post 0.5 \
  --beta-bias 0.0 \
  --max-new-tokens 128 \
  --resume on \
  --checkpoint-every 25

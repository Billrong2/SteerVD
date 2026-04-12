#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cs/x/xxr230000/Steer_VD"
MODEL_DIR="Qwen_Qwen2.5-Coder-7B-Instruct"
RUN_NAME="${1:-qwen25_primevul_full_baseline_revdcot}"
OUT_DIR="${ROOT}/artifacts/primevul/${MODEL_DIR}/${RUN_NAME}"
LOG_PATH="${OUT_DIR}.log"
PID_PATH="${OUT_DIR}.pid"
RUNNER_PATH="${OUT_DIR}.runner.sh"
SESSION_NAME="primevul_${RUN_NAME//[^A-Za-z0-9_]/_}"

mkdir -p "$(dirname "${OUT_DIR}")"

cd "${ROOT}"

CMD_STR=$(cat <<EOF
cd "${ROOT}" && \
export TOKENIZERS_PARALLELISM=false && \
export PYTHONUNBUFFERED=1 && \
exec python3 "${ROOT}/primevul_eval.py" \
  --dataset-path "${ROOT}/Source/primevul_test.jsonl" \
  --protocol revd_cot \
  --variant baseline \
  --prior code_gadget \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --cache-dir /home/cs/x/xxr230000/.cache/models \
  --gpu-ids 0 \
  --samples-per-snippet 1 \
  --do-sample off \
  --max-new-tokens 128 \
  --language c \
  --run-name "${RUN_NAME}" \
  --resume on \
  --checkpoint-every 25
EOF
)

echo "Launching detached baseline run: ${RUN_NAME}"
echo "Log: ${LOG_PATH}"
echo "PID file: ${PID_PATH}"
echo "tmux session: ${SESSION_NAME}"
echo "Steered code_gadget runs remain blocked until the multi-gadget reduction rule is defined."

cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${CMD_STR} > "${LOG_PATH}" 2>&1
EOF
chmod +x "${RUNNER_PATH}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  tmux list-panes -t "${SESSION_NAME}" -F '#{pane_pid}' | head -n 1 > "${PID_PATH}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" "bash '${RUNNER_PATH}'"
tmux list-panes -t "${SESSION_NAME}" -F '#{pane_pid}' | head -n 1 > "${PID_PATH}"
echo "PID: $(cat "${PID_PATH}")"


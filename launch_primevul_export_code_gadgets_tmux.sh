#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cs/x/xxr230000/Steer_VD"
OUTPUT_ROOT="${1:-${ROOT}/Source/primevul_test_snippets}"
STRICT_MODE="${2:-off}"
TARGET_FILTER="${3:-all}"
SESSION_BASE="$(basename "${OUTPUT_ROOT}")"
SESSION_NAME="primevul_export_${SESSION_BASE//[^A-Za-z0-9_]/_}"
LOG_PATH="${OUTPUT_ROOT}.log"
PID_PATH="${OUTPUT_ROOT}.pid"
RUNNER_PATH="${OUTPUT_ROOT}.runner.sh"

mkdir -p "$(dirname "${OUTPUT_ROOT}")"

cd "${ROOT}"

CMD_STR=$(cat <<EOF
cd "${ROOT}" && \
export TOKENIZERS_PARALLELISM=false && \
export PYTHONUNBUFFERED=1 && \
export JAVA_HOME="/usr" && \
export PATH="/usr/bin:\$PATH" && \
exec python3 "${ROOT}/primevul_export_code_gadgets.py" \
  --dataset-path "${ROOT}/Source/primevul_test.jsonl" \
  --output-root "${OUTPUT_ROOT}" \
  --strict-project-context "${STRICT_MODE}" \
  --target-filter "${TARGET_FILTER}" \
  --resume on \
  --checkpoint-every 25
EOF
)

echo "Launching detached code-gadget export"
echo "Output root: ${OUTPUT_ROOT}"
echo "Strict project context: ${STRICT_MODE}"
echo "Target filter: ${TARGET_FILTER}"
echo "Log: ${LOG_PATH}"
echo "PID file: ${PID_PATH}"
echo "tmux session: ${SESSION_NAME}"

cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
${CMD_STR} 2>&1 | tee "${LOG_PATH}"
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

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/NN_training}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
FOLLOWON_HOURS="${FOLLOWON_HOURS:-6}"
LOG_PATH="${LOG_PATH:-/workspace/v2a_pipeline/overnight_sweeps/followon_queue.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

echo "Queue watcher started at $(date)" >> "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" >> "${LOG_PATH}"
echo "Follow-on hours: ${FOLLOWON_HOURS}" >> "${LOG_PATH}"

while true; do
  ACTIVE_MATCHES="$(pgrep -af 'infra/runpod/run_overnight_experiments.py' 2>/dev/null || true)"
  ACTIVE_COUNT="$(printf '%s\n' "${ACTIVE_MATCHES}" | grep -v 'pgrep -af' | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "${ACTIVE_COUNT}" -eq 0 ]]; then
    echo "No active overnight runner detected at $(date). Launching follow-on preset." >> "${LOG_PATH}"
    cd "${REPO_ROOT}"
    PRESET=followon HOURS="${FOLLOWON_HOURS}" bash infra/runpod/launch_overnight_experiments.sh >> "${LOG_PATH}" 2>&1
    echo "Follow-on launch completed at $(date)." >> "${LOG_PATH}"
    exit 0
  fi

  echo "Active overnight runner count=${ACTIVE_COUNT} at $(date). Waiting ${WAIT_SECONDS}s." >> "${LOG_PATH}"
  sleep "${WAIT_SECONDS}"
done

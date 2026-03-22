#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/NN_training}"
SWEEPS_ROOT="${SWEEPS_ROOT:-/workspace/v2a_pipeline/overnight_sweeps}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
REQUIRED_LAUNCHES="${REQUIRED_LAUNCHES:-2}"
PRESET="${PRESET:-third}"
HOURS="${HOURS:-6}"
PILOT_EPOCHS="${PILOT_EPOCHS:-8}"
FULL_EPOCHS="${FULL_EPOCHS:-120}"
FINALISTS="${FINALISTS:-3}"
MIN_EPOCHS="${MIN_EPOCHS:-24}"
LOG_PATH="${LOG_PATH:-${SWEEPS_ROOT}/queue_${PRESET}.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

echo "Queued wave watcher started at $(date)" >> "${LOG_PATH}"
echo "Repo root: ${REPO_ROOT}" >> "${LOG_PATH}"
echo "Preset: ${PRESET}" >> "${LOG_PATH}"
echo "Required launches: ${REQUIRED_LAUNCHES}" >> "${LOG_PATH}"
echo "Hours: ${HOURS}" >> "${LOG_PATH}"
echo "Pilot epochs: ${PILOT_EPOCHS}" >> "${LOG_PATH}"
echo "Full epochs: ${FULL_EPOCHS}" >> "${LOG_PATH}"
echo "Finalists: ${FINALISTS}" >> "${LOG_PATH}"
echo "Min epochs: ${MIN_EPOCHS}" >> "${LOG_PATH}"

while true; do
  ACTIVE_MATCHES="$(pgrep -af 'infra/runpod/run_overnight_experiments.py' 2>/dev/null || true)"
  ACTIVE_COUNT="$(printf '%s\n' "${ACTIVE_MATCHES}" | grep -v 'pgrep -af' | sed '/^$/d' | wc -l | tr -d ' ')"
  LAUNCH_COUNT="$(find "${SWEEPS_ROOT}" -maxdepth 1 -name 'launch_*.log' | wc -l | tr -d ' ')"

  if [[ "${LAUNCH_COUNT}" -ge "${REQUIRED_LAUNCHES}" && "${ACTIVE_COUNT}" -eq 0 ]]; then
    echo "Conditions met at $(date): launches=${LAUNCH_COUNT}, active=${ACTIVE_COUNT}. Launching preset ${PRESET}." >> "${LOG_PATH}"
    cd "${REPO_ROOT}"
    PRESET="${PRESET}" HOURS="${HOURS}" PILOT_EPOCHS="${PILOT_EPOCHS}" FULL_EPOCHS="${FULL_EPOCHS}" FINALISTS="${FINALISTS}" MIN_EPOCHS="${MIN_EPOCHS}" \
      bash infra/runpod/launch_overnight_experiments.sh >> "${LOG_PATH}" 2>&1
    echo "Launch completed at $(date)." >> "${LOG_PATH}"
    exit 0
  fi

  echo "Waiting at $(date): launches=${LAUNCH_COUNT}, active=${ACTIVE_COUNT}, sleep=${WAIT_SECONDS}s." >> "${LOG_PATH}"
  sleep "${WAIT_SECONDS}"
done

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/NN_training}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
WORK_ROOT="${WORK_ROOT:-/workspace/v2a_pipeline}"
SWEEPS_ROOT="${SWEEPS_ROOT:-${WORK_ROOT}/overnight_sweeps}"
HOURS="${HOURS:-10}"
PRESET="${PRESET:-default}"
PILOT_EPOCHS="${PILOT_EPOCHS:-}"
FULL_EPOCHS="${FULL_EPOCHS:-}"
FINALISTS="${FINALISTS:-}"
MIN_EPOCHS="${MIN_EPOCHS:-}"

mkdir -p "${SWEEPS_ROOT}"

RUNNER="${REPO_ROOT}/infra/runpod/run_overnight_experiments.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${SWEEPS_ROOT}/launch_${STAMP}.log"
PID_PATH="${SWEEPS_ROOT}/launch_${STAMP}.pid"

echo "Launching overnight experiments"
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Sweeps root: ${SWEEPS_ROOT}"
echo "Hours: ${HOURS}"
echo "Preset: ${PRESET}"
if [[ -n "${PILOT_EPOCHS}" ]]; then
  echo "Pilot epochs: ${PILOT_EPOCHS}"
fi
if [[ -n "${FULL_EPOCHS}" ]]; then
  echo "Full epochs: ${FULL_EPOCHS}"
fi
if [[ -n "${FINALISTS}" ]]; then
  echo "Finalists: ${FINALISTS}"
fi
if [[ -n "${MIN_EPOCHS}" ]]; then
  echo "Min epochs: ${MIN_EPOCHS}"
fi
echo "Log: ${LOG_PATH}"

CMD=(
  "${PYTHON_BIN}" "${RUNNER}"
  --repo-root "${REPO_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --dataset-dir "${WORK_ROOT}/datasets/combined_long_10s_step500"
  --sweeps-root "${SWEEPS_ROOT}"
  --hours "${HOURS}"
  --preset "${PRESET}"
)

if [[ -n "${PILOT_EPOCHS}" ]]; then
  CMD+=(--pilot-epochs "${PILOT_EPOCHS}")
fi
if [[ -n "${FULL_EPOCHS}" ]]; then
  CMD+=(--full-epochs "${FULL_EPOCHS}")
fi
if [[ -n "${FINALISTS}" ]]; then
  CMD+=(--finalists "${FINALISTS}")
fi
if [[ -n "${MIN_EPOCHS}" ]]; then
  CMD+=(--min-epochs "${MIN_EPOCHS}")
fi

nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &

echo $! > "${PID_PATH}"
echo "PID: $(cat "${PID_PATH}")"

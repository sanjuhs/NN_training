#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/NN_training}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
WORK_ROOT="${WORK_ROOT:-/workspace/v2a_pipeline}"
COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${WORK_ROOT}/datasets/combined_long_10s_step500}"
RUNS_DIR="${RUNS_DIR:-${WORK_ROOT}/runs}"
RUN_NAME="${RUN_NAME:-conv_transformer_d320_l12_corrvar_std}"
RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
LAST_CHECKPOINT="${RUN_DIR}/${RUN_NAME}_last.pth"

mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1

echo "=== Deeper Conv Transformer Rerun ==="
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Combined dataset: ${COMBINED_DATASET_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Start time: $(date)"

RESUME_ARGS=()
if [[ -f "${LAST_CHECKPOINT}" ]]; then
  echo "Resuming from checkpoint: ${LAST_CHECKPOINT}"
  RESUME_ARGS+=(--resume-from "${LAST_CHECKPOINT}")
fi

if [[ ! -f "${COMBINED_DATASET_DIR}/audio_sequences.npy" ]]; then
  echo "Combined dataset not found at ${COMBINED_DATASET_DIR}"
  exit 1
fi

cd "${REPO_ROOT}"

"${PYTHON_BIN}" V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py \
  --data-dir "${COMBINED_DATASET_DIR}" \
  --variant conv_transformer \
  --epochs 60 \
  --batch-size 8 \
  --grad-accumulation 1 \
  --lr 1.5e-4 \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --temporal-weight 0.05 \
  --corr-weight 0.15 \
  --variance-weight 0.05 \
  --mouth-weight-scale 1.1 \
  --dropout 0.1 \
  --d-model 320 \
  --nhead 8 \
  --num-layers 12 \
  --ffn-dim 1280 \
  --conv-kernel-size 9 \
  --target-normalization standardize \
  --segment-aware-split \
  --val-fraction 0.15 \
  --patience 12 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --tf32 \
  --device cuda \
  --checkpoint-path "${RUN_DIR}/${RUN_NAME}_best.pth" \
  --last-checkpoint-path "${RUN_DIR}/${RUN_NAME}_last.pth" \
  --history-path "${RUN_DIR}/${RUN_NAME}_history.json" \
  --plot-path "${RUN_DIR}/${RUN_NAME}_history.png" \
  --curve-plot-path "${RUN_DIR}/${RUN_NAME}_curves.png" \
  --curve-sample-path "${RUN_DIR}/${RUN_NAME}_curves.json" \
  --summary-path "${RUN_DIR}/${RUN_NAME}_summary.json" \
  "${RESUME_ARGS[@]}"

"${PYTHON_BIN}" V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py \
  --checkpoint "${RUN_DIR}/${RUN_NAME}_best.pth" \
  --output "${RUN_DIR}/${RUN_NAME}.onnx" \
  --manifest "${RUN_DIR}/${RUN_NAME}.json" \
  --seq-len 1000

echo "Rerun complete at $(date)"

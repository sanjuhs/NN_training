#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/NN_training}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
WORK_ROOT="${WORK_ROOT:-/workspace/v2a_pipeline}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/root/v2a_pipeline_scratch}"
HF_ROOT="${HF_ROOT:-${SCRATCH_ROOT}/hf_downloads}"
PROCESSING_ROOT="${PROCESSING_ROOT:-${SCRATCH_ROOT}/processing}"
HF_DATASET_DIR="${HF_DATASET_DIR:-${WORK_ROOT}/datasets/hf_long_10s_step500}"
COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${WORK_ROOT}/datasets/combined_long_10s_step500}"
RUNS_DIR="${RUNS_DIR:-${WORK_ROOT}/runs}"

mkdir -p "${HF_ROOT}" "${PROCESSING_ROOT}" "${HF_DATASET_DIR}" "${COMBINED_DATASET_DIR}" "${RUNS_DIR}"
export PYTHONUNBUFFERED=1

echo "=== Full Transformer Pipeline ==="
echo "Repo root: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Work root: ${WORK_ROOT}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Start time: $(date)"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" V2A-over-training-old-nn/1_data_cleaning/download_hf_datasets.py \
  --dataset sanjuhs/longer-video-dataset \
  --dataset sanjuhs/ml_video_dataset \
  --dataset sanjuhs/test-video-dataset \
  --output-root "${HF_ROOT}"

"${PYTHON_BIN}" V2A-over-training-old-nn/1_data_cleaning/build_combined_long_context_dataset.py \
  --video-root "${HF_ROOT}/sanjuhs__longer-video-dataset" \
  --video-root "${HF_ROOT}/sanjuhs__ml_video_dataset" \
  --video-root "${HF_ROOT}/sanjuhs__test-video-dataset" \
  --working-dir "${PROCESSING_ROOT}" \
  --output-dir "${HF_DATASET_DIR}" \
  --window-ms 10000 \
  --overlap-ms 5000 \
  --blendshape-fps 30 \
  --use-gpu

"${PYTHON_BIN}" V2A-over-training-old-nn/2_architecture_training/merge_long_context_datasets.py \
  --input-dir "${REPO_ROOT}/V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500" \
  --input-dir "${HF_DATASET_DIR}" \
  --output-dir "${COMBINED_DATASET_DIR}"

train_variant() {
  local run_name="$1"
  local variant="$2"
  local d_model="$3"
  local nhead="$4"
  local num_layers="$5"
  local ffn_dim="$6"
  local batch_size="$7"
  local epochs="$8"
  local patience="$9"

  local run_dir="${RUNS_DIR}/${run_name}"
  mkdir -p "${run_dir}"

  echo "=== Training ${run_name} ==="

  "${PYTHON_BIN}" V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py \
    --data-dir "${COMBINED_DATASET_DIR}" \
    --variant "${variant}" \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --lr 2e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --temporal-weight 0.02 \
    --mouth-weight-scale 1.1 \
    --dropout 0.1 \
    --d-model "${d_model}" \
    --nhead "${nhead}" \
    --num-layers "${num_layers}" \
    --ffn-dim "${ffn_dim}" \
    --conv-kernel-size 9 \
    --segment-aware-split \
    --val-fraction 0.15 \
    --patience "${patience}" \
    --device cuda \
    --checkpoint-path "${run_dir}/${run_name}_best.pth" \
    --last-checkpoint-path "${run_dir}/${run_name}_last.pth" \
    --history-path "${run_dir}/${run_name}_history.json" \
    --plot-path "${run_dir}/${run_name}_history.png" \
    --summary-path "${run_dir}/${run_name}_summary.json"

  "${PYTHON_BIN}" V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py \
    --checkpoint "${run_dir}/${run_name}_best.pth" \
    --output "${run_dir}/${run_name}.onnx" \
    --manifest "${run_dir}/${run_name}.json" \
    --seq-len 1000
}

train_variant "baseline_d192_l6" "baseline" 192 6 6 768 2 30 8
train_variant "conv_transformer_d224_l8" "conv_transformer" 224 8 8 896 2 24 6
train_variant "gated_transformer_d224_l8" "gated_transformer" 224 8 8 896 2 24 6

if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "HF token detected; upload step can be added later."
else
  echo "No HF token detected on the pod. Skipping upload."
fi

echo "Pipeline complete at $(date)"

# Repository Guide

This repository has several experiments, but the active voice-to-animation pipeline lives in `V2A-over-training-old-nn`.

## Start Here

- [README.md](README.md): top-level overview and local demo entry points.
- [RUNPOD_TRAINING_REPORT.md](RUNPOD_TRAINING_REPORT.md): latest Runpod training results, Hugging Face upload status, synced checkpoints, and loss curves.
- [V2A-over-training-old-nn/README.md](V2A-over-training-old-nn/README.md): focused guide for the audio-to-blendshapes project.

## Most Important Folders

- [V2A-over-training-old-nn](V2A-over-training-old-nn): main audio-to-blendshapes codebase.
- [Online-demo](Online-demo): browser demos, calibration viewer, and Three.js model playback.
- [runpod_results](runpod_results): results synced back from Runpod, including `.pth`, `.onnx`, plots, and logs.
- [infra/runpod](infra/runpod): Runpod setup notes, SSH info, and the remote pipeline script.
- [infra/huggingface](infra/huggingface): notes for dataset upload workflow.
- [assets](assets): ONNX models and demo assets used by the browser UI.

## Where The Core Code Lives

### Data Extraction And Dataset Building

- [V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py](V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py): MediaPipe face/blendshape extraction.
- [V2A-over-training-old-nn/1_data_cleaning/2_extract_audio_features.py](V2A-over-training-old-nn/1_data_cleaning/2_extract_audio_features.py): audio feature extraction.
- [V2A-over-training-old-nn/1_data_cleaning/3_create_datset.py](V2A-over-training-old-nn/1_data_cleaning/3_create_datset.py): original dataset creation path.
- [V2A-over-training-old-nn/1_data_cleaning/download_hf_datasets.py](V2A-over-training-old-nn/1_data_cleaning/download_hf_datasets.py): pulls source videos from Hugging Face onto Runpod.
- [V2A-over-training-old-nn/1_data_cleaning/build_combined_long_context_dataset.py](V2A-over-training-old-nn/1_data_cleaning/build_combined_long_context_dataset.py): builds the cleaned long-context dataset from processed clips.

### Training

- [V2A-over-training-old-nn/blendshape_layout.py](V2A-over-training-old-nn/blendshape_layout.py): authoritative blendshape index mapping and mouth-channel groups.
- [V2A-over-training-old-nn/2_architecture_training/train.py](V2A-over-training-old-nn/2_architecture_training/train.py): original TCN training script, now using the corrected channel mapping.
- [V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py](V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py): older tiny-transformer training path.
- [V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py](V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py): newer transformer training entry point used on Runpod.
- [V2A-over-training-old-nn/2_architecture_training/models/audio_transformer_variants.py](V2A-over-training-old-nn/2_architecture_training/models/audio_transformer_variants.py): the new transformer variants trained in the latest run.
- [V2A-over-training-old-nn/2_architecture_training/merge_long_context_datasets.py](V2A-over-training-old-nn/2_architecture_training/merge_long_context_datasets.py): merges the existing and newly processed datasets.
- [V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py](V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py): ONNX export for the new transformer models.

### Inference And Demo

- [V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py](V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py): local inference for the tiny transformer.
- [Online-demo/mediapipe-calibration.html](Online-demo/mediapipe-calibration.html): side-by-side calibration viewer for source video vs MediaPipe-driven raccoon.
- [Online-demo/transformer-model.html](Online-demo/transformer-model.html): browser ONNX transformer demo.
- [Online-demo/comparison.html](Online-demo/comparison.html): side-by-side system comparison demo.

## Where The Important Outputs Are

### Latest Runpod Results

- [runpod_results/current_baseline_existing](runpod_results/current_baseline_existing): baseline retrain on the existing long-context dataset.
- [runpod_results/full_pipeline](runpod_results/full_pipeline): full Hugging Face download, MediaPipe extraction, combined dataset build, three-model training run, and ONNX exports.

### Specific Artifacts

- [runpod_results/current_baseline_existing/models/baseline_existing_best.pth](runpod_results/current_baseline_existing/models/baseline_existing_best.pth): best checkpoint from the baseline retrain on the old dataset.
- [runpod_results/current_baseline_existing/models/baseline_existing_best.onnx](runpod_results/current_baseline_existing/models/baseline_existing_best.onnx): ONNX export of that baseline retrain.
- [runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth): best checkpoint on the combined dataset.
- [runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx): ONNX export for the current best overall model.
- [runpod_results/full_pipeline/runs/conv_transformer_d224_l8/conv_transformer_d224_l8_best.pth](runpod_results/full_pipeline/runs/conv_transformer_d224_l8/conv_transformer_d224_l8_best.pth): conv-augmented transformer checkpoint.
- [runpod_results/full_pipeline/runs/gated_transformer_d224_l8/gated_transformer_d224_l8_best.pth](runpod_results/full_pipeline/runs/gated_transformer_d224_l8/gated_transformer_d224_l8_best.pth): gated transformer checkpoint.

## Best Docs To Read Next

- [RUNPOD_TRAINING_REPORT.md](RUNPOD_TRAINING_REPORT.md): if you want the latest status and model comparison.
- [V2A-over-training-old-nn/README_tiny_transformer.md](V2A-over-training-old-nn/README_tiny_transformer.md): if you want the earlier tiny-transformer background.
- [V2A-over-training-old-nn/documentation/TinyTransformer_Overfit_Architecture.md](V2A-over-training-old-nn/documentation/TinyTransformer_Overfit_Architecture.md): if you want the original long-context dataset story.
- [infra/runpod/runpod_info.md](infra/runpod/runpod_info.md): if you need the pod connection and volume notes.

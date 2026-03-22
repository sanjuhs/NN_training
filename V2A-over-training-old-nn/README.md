# V2A Over-Training Project

This folder contains the main voice-to-animation pipeline: audio in, `52 blendshapes + 7 pose values` out.

The original goal of this subproject was to overfit and stress-test the pipeline on a narrow dataset first. It now also contains the newer long-context transformer training path that was run remotely on Runpod.

## Start Here

- [../GUIDE.md](../GUIDE.md): top-level repository navigation.
- [../RUNPOD_TRAINING_REPORT.md](../RUNPOD_TRAINING_REPORT.md): latest Runpod status, model comparison, loss curves, and synced artifacts.
- [README_tiny_transformer.md](README_tiny_transformer.md): earlier transformer notes and local training flow.

## Folder Layout

### `1_data_cleaning`

Dataset preparation and MediaPipe processing:

- [1_data_cleaning/1_extract_blendshapes.py](1_data_cleaning/1_extract_blendshapes.py): extracts face landmarks, pose, and blendshape coefficients.
- [1_data_cleaning/2_extract_audio_features.py](1_data_cleaning/2_extract_audio_features.py): extracts frame-aligned audio features.
- [1_data_cleaning/3_create_datset.py](1_data_cleaning/3_create_datset.py): original dataset builder.
- [1_data_cleaning/download_hf_datasets.py](1_data_cleaning/download_hf_datasets.py): downloads source video datasets from Hugging Face on Runpod.
- [1_data_cleaning/build_combined_long_context_dataset.py](1_data_cleaning/build_combined_long_context_dataset.py): builds the cleaned long-context training dataset from processed clips.

### `2_architecture_training`

Model definitions, training, merging, and ONNX export:

- [blendshape_layout.py](blendshape_layout.py): corrected blendshape index mapping and grouped mouth channels.
- [2_architecture_training/train.py](2_architecture_training/train.py): original TCN training script with corrected channel weighting.
- [2_architecture_training/train_tiny_transformer.py](2_architecture_training/train_tiny_transformer.py): older tiny-transformer training entry point.
- [2_architecture_training/train_audio_transformer.py](2_architecture_training/train_audio_transformer.py): newer transformer training script used in the Runpod pipeline.
- [2_architecture_training/models/audio_transformer_variants.py](2_architecture_training/models/audio_transformer_variants.py): baseline, conv-transformer, and gated-transformer variants.
- [2_architecture_training/merge_long_context_datasets.py](2_architecture_training/merge_long_context_datasets.py): combines the existing and newly processed datasets.
- [2_architecture_training/export_audio_transformer_onnx.py](2_architecture_training/export_audio_transformer_onnx.py): exports the new transformer checkpoints to ONNX.

### `3_inference`

Inference scripts for saved models:

- [3_inference/inference.py](3_inference/inference.py)
- [3_inference/tiny_transformer_inference.py](3_inference/tiny_transformer_inference.py)

### `documentation`

Project notes and older architecture writeups:

- [documentation/TinyTransformer_Overfit_Architecture.md](documentation/TinyTransformer_Overfit_Architecture.md)
- [documentation/TinyTransformer_Quick_Status.md](documentation/TinyTransformer_Quick_Status.md)
- [documentation/TinyTransformer_Conference_FAQ.md](documentation/TinyTransformer_Conference_FAQ.md)
- [documentation/TinyTransformer_Runpod_Report.md](documentation/TinyTransformer_Runpod_Report.md)

## Current Best Outputs

The latest synced Runpod artifacts live outside this folder in [../runpod_results](../runpod_results).

Most important outputs:

- [../runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth](../runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth)
- [../runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx](../runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx)
- [../runpod_results/current_baseline_existing/models/baseline_existing_best.pth](../runpod_results/current_baseline_existing/models/baseline_existing_best.pth)
- [../runpod_results/current_baseline_existing/models/baseline_existing_best.onnx](../runpod_results/current_baseline_existing/models/baseline_existing_best.onnx)

## Browser And Calibration Demo

The demo pages are in [../Online-demo](../Online-demo).

Most useful pages:

- [../Online-demo/transformer-model.html](../Online-demo/transformer-model.html)
- [../Online-demo/comparison.html](../Online-demo/comparison.html)
- [../Online-demo/mediapipe-calibration.html](../Online-demo/mediapipe-calibration.html)

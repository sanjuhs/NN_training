# Runpod Training Report

This file records the latest remote training run that downloaded Hugging Face video data on Runpod, extracted MediaPipe blendshapes, built a combined long-context dataset, trained multiple transformer variants, exported ONNX, and synced the outputs back into this repository.

## Current Status

- Runpod processing completed successfully.
- The cleaned long-context dataset was uploaded to Hugging Face at [sanjuhs/audio-to-blendshapes-longer-dataset](https://huggingface.co/datasets/sanjuhs/audio-to-blendshapes-longer-dataset).
- The synced local outputs are in [runpod_results/current_baseline_existing](runpod_results/current_baseline_existing) and [runpod_results/full_pipeline](runpod_results/full_pipeline).
- The later overnight sweeps are synced under [runpod_results/overnight_sweeps](runpod_results/overnight_sweeps).
- The blendshape channel mapping issue was fixed before training by moving the mouth-channel logic to [V2A-over-training-old-nn/blendshape_layout.py](V2A-over-training-old-nn/blendshape_layout.py).
- Mouth-weighted losses were updated to use the corrected mouth channels with a `1.1x` mouth emphasis.

## Overnight Sweep Snapshot

After the main pipeline runs, we launched additional overnight sweeps on March 21-22, 2026. Those runs matter because they explored larger conv, gated, Conformer, and multi-scale variants with newer optimizer and loss combinations.

Two different checkpoints came out as the most useful:

### Best Mouth/Smile Checkpoint

Source: [runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16](runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16)

| Metric | Value |
| --- | --- |
| Variant | `conv_transformer` |
| Width / depth | `d_model=320`, `layers=12` |
| Best val loss | `0.549753` |
| Best mouth MAE | `0.029945` |
| Best mouth/jaw corr | `0.1892` |
| Best smile corr | `0.4157` |
| Best overall MAE | `0.049622` |

Why it matters:

- it gave the best smile correlation seen in the overnight runs
- it also gave the best mouth/jaw correlation among the finalized checkpoints
- this is still the strongest checkpoint if the goal is facial expressiveness rather than only minimizing the composite objective

### Best Composite-Loss Checkpoint

Source: [runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber](runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber)

| Metric | Value |
| --- | --- |
| Variant | `conv_transformer` |
| Width / depth | `d_model=320`, `layers=12` |
| Optimizer / base loss | `NAdam`, `Huber` |
| Best val loss | `0.406168` |
| Best mouth MAE | `0.031232` |
| Best mouth/jaw corr | `0.1862` |
| Best smile corr | `0.3814` |
| Best overall MAE | `0.051613` |

Why it matters:

- it achieved the lowest overnight composite validation objective
- it did not beat the earlier wave-1 model on smile correlation
- because this sweep switched to `Huber`, the absolute `val_loss` is not directly comparable to the earlier pure-`L1` overnight sweep

### What We Learned

- The recovered second wave did improve the standardized composite objective.
- The simpler conv transformer remained the strongest model on the mouth and smile correlation metrics that matter most visually.
- Larger gated, Conformer, and multi-scale pilots did not clearly beat the simpler conv model before we stopped.
- The next bottleneck is target quality and calibration, not just scaling the transformer further.

## What Was Uploaded To Hugging Face

The uploaded dataset is the cleaned training set only, not the raw videos.

Source on Runpod persistent volume:

- `/workspace/v2a_pipeline/datasets/hf_long_10s_step500`

Files uploaded into the Hugging Face dataset repository:

- `audio_sequences.npy`
- `target_sequences.npy`
- `vad_sequences.npy`
- `segment_ids.npy`
- `window_start_frames.npy`
- `dataset_metadata.json`
- `README.md`

Files not uploaded:

- raw source videos
- temporary extraction scratch directories under `/root/v2a_pipeline_scratch`
- model checkpoints
- plots and logs

## Dataset Summary

### Newly Processed Hugging Face Dataset

Source metadata: [runpod_results/full_pipeline/datasets/hf_long_10s_step500_metadata.json](runpod_results/full_pipeline/datasets/hf_long_10s_step500_metadata.json)

| Field | Value |
| --- | --- |
| Number of sequences | 2257 |
| Sequence length | 1000 frames |
| Sequence duration | 10 seconds |
| Step size | 500 frames |
| Overlap | 5 seconds |
| Audio feature dimension | 80 |
| Target dimension | 59 |
| Source videos processed | 10 |

### Combined Dataset Used For Final Training

Source metadata: [runpod_results/full_pipeline/datasets/combined_long_10s_step500_metadata.json](runpod_results/full_pipeline/datasets/combined_long_10s_step500_metadata.json)

| Field | Value |
| --- | --- |
| Number of sequences | 3055 |
| Existing dataset contribution | 798 sequences |
| Newly processed HF contribution | 2257 sequences |
| Existing unique segments | 83 |
| New unique segments | 10 |

## Data Quality Notes

MediaPipe extraction worked well on the `ml_video_dataset` clips and the short test clip, but three of the longer phone videos had weak face detection. This matters because target quality can bottleneck training quality even when the model itself improves.

Lowest face-detection rates observed in the pipeline log:

- `VID_20260320_083213340.mp4`: `48.0%`
- `VID_20260320_125631658.mp4`: `68.1%`
- `VID_20260320_132448358.mp4`: `54.2%`
- `VID_20260320_220651328.mp4`: `99.9%`
- `ml_video_dataset` clips: `94.0%` to `99.7%`
- `test.mp4`: `100.0%`

Log file: [runpod_results/full_pipeline/pipeline.log](runpod_results/full_pipeline/pipeline.log)

## Model Comparison

### Baseline Retrain On Existing Dataset

Summary: [runpod_results/current_baseline_existing/plots/baseline_existing_summary.json](runpod_results/current_baseline_existing/plots/baseline_existing_summary.json)

| Metric | Value |
| --- | --- |
| Variant | baseline |
| Params | 2,733,947 |
| Best epoch | 16 |
| Best val loss | 0.076364 |
| Val mouth MAE | 0.035716 |
| Val jaw-open corr | -0.003674 |
| Val smile corr | 0.003066 |
| Train loss trend | 0.091839 -> 0.063996 |
| Val loss trend | 0.078981 -> 0.076894 |

Artifacts:

- [runpod_results/current_baseline_existing/models/baseline_existing_best.pth](runpod_results/current_baseline_existing/models/baseline_existing_best.pth)
- [runpod_results/current_baseline_existing/models/baseline_existing_best.onnx](runpod_results/current_baseline_existing/models/baseline_existing_best.onnx)
- [runpod_results/current_baseline_existing/models/baseline_existing_best.json](runpod_results/current_baseline_existing/models/baseline_existing_best.json)

### Full Pipeline Models On Combined Dataset

| Model | Params | Best epoch | Best val loss | Val mouth MAE | Val jaw-open corr | Val smile corr |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline_d192_l6` | 2,733,947 | 24 | 0.063940 | 0.030038 | 0.214903 | 0.317995 |
| `conv_transformer_d224_l8` | 5,347,611 | 23 | 0.064167 | 0.029966 | 0.176654 | 0.294647 |
| `gated_transformer_d224_l8` | 6,535,707 | 19 | 0.064606 | 0.030202 | 0.205233 | 0.267998 |

Training trend snapshot:

- `baseline_d192_l6`: train `0.087289 -> 0.066400`, val `0.068773 -> 0.063944`
- `conv_transformer_d224_l8`: train `0.086748 -> 0.065936`, val `0.068657 -> 0.064170`
- `gated_transformer_d224_l8`: train `0.086272 -> 0.068152`, val `0.068777 -> 0.064657`

Interpretation:

- `baseline_d192_l6` is the best overall checkpoint by validation loss.
- `conv_transformer_d224_l8` has the best mouth MAE, but only by a small margin.
- `gated_transformer_d224_l8` did not beat the simpler baseline on this dataset.

## Best Current Checkpoint

The best current model from the full Runpod pipeline is:

- [runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth)
- [runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx)
- [runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.json](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.json)

## Loss Curves

### Existing-Dataset Baseline

![Existing baseline loss curve](runpod_results/current_baseline_existing/plots/baseline_existing_history.png)

### Combined-Dataset Baseline

![Full baseline loss curve](runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_history.png)

### Conv Transformer

![Conv transformer loss curve](runpod_results/full_pipeline/runs/conv_transformer_d224_l8/conv_transformer_d224_l8_history.png)

### Gated Transformer

![Gated transformer loss curve](runpod_results/full_pipeline/runs/gated_transformer_d224_l8/gated_transformer_d224_l8_history.png)

## Synced Local Artifact Locations

### Existing-Dataset Retrain

- [runpod_results/current_baseline_existing/baseline_existing.log](runpod_results/current_baseline_existing/baseline_existing.log)
- [runpod_results/current_baseline_existing/models](runpod_results/current_baseline_existing/models)
- [runpod_results/current_baseline_existing/plots](runpod_results/current_baseline_existing/plots)

### Full Pipeline

- [runpod_results/full_pipeline/pipeline.log](runpod_results/full_pipeline/pipeline.log)
- [runpod_results/full_pipeline/datasets](runpod_results/full_pipeline/datasets)
- [runpod_results/full_pipeline/runs/baseline_d192_l6](runpod_results/full_pipeline/runs/baseline_d192_l6)
- [runpod_results/full_pipeline/runs/conv_transformer_d224_l8](runpod_results/full_pipeline/runs/conv_transformer_d224_l8)
- [runpod_results/full_pipeline/runs/gated_transformer_d224_l8](runpod_results/full_pipeline/runs/gated_transformer_d224_l8)

## Code Paths Used In This Run

- [V2A-over-training-old-nn/blendshape_layout.py](V2A-over-training-old-nn/blendshape_layout.py)
- [V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py](V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py)
- [V2A-over-training-old-nn/1_data_cleaning/download_hf_datasets.py](V2A-over-training-old-nn/1_data_cleaning/download_hf_datasets.py)
- [V2A-over-training-old-nn/1_data_cleaning/build_combined_long_context_dataset.py](V2A-over-training-old-nn/1_data_cleaning/build_combined_long_context_dataset.py)
- [V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py](V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py)
- [V2A-over-training-old-nn/2_architecture_training/models/audio_transformer_variants.py](V2A-over-training-old-nn/2_architecture_training/models/audio_transformer_variants.py)
- [V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py](V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py)
- [infra/runpod/run_full_transformer_pipeline.sh](infra/runpod/run_full_transformer_pipeline.sh)

## Recommended Next Step

The next bottleneck is target quality, not model size. The training losses improved and the combined dataset clearly helped, but the weak face-detection rates on several long phone videos are still a real source of label noise. The best follow-up is to use [Online-demo/mediapipe-calibration.html](Online-demo/mediapipe-calibration.html) to inspect those clips and decide whether to recalibrate or exclude the weakest MediaPipe outputs before the next larger run.

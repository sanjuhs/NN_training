# NN_training

This repository contains multiple neural-network experiments. The current primary track is the `V2A-over-training-old-nn` pipeline: voice to facial blendshapes.

## Start Here

- [GUIDE.md](GUIDE.md): repository navigation and the most important files.
- [RUNPOD_TRAINING_REPORT.md](RUNPOD_TRAINING_REPORT.md): latest Runpod training results, model comparisons, synced checkpoints, and loss curves.
- [RUNPOD_OVERNIGHT_EXPERIMENTS.md](RUNPOD_OVERNIGHT_EXPERIMENTS.md): overnight sweep log from March 21-22, 2026, including the stalled handoff, recovered wave 2, and partial wave 3.
- [V2A-over-training-old-nn/README.md](V2A-over-training-old-nn/README.md): focused guide to the active audio-to-blendshapes project.

## Latest Snapshot

The latest overnight sweeps are synced under `runpod_results/overnight_sweeps`.

- Best mouth/smile checkpoint so far:
  - `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16`
  - best mouth MAE `0.029945`
  - best mouth/jaw corr `0.1892`
  - best smile corr `0.4157`
- Best composite-loss checkpoint so far:
  - `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`
  - best val loss `0.406168`
  - best mouth MAE `0.031232`
  - best mouth/jaw corr `0.1862`
  - best smile corr `0.3814`

The main lesson from these runs is that larger and newer variants can reduce the composite objective, but the strongest mouth and smile correlation still comes from the simpler conv transformer. That points back to target quality and calibration as the next bottleneck, not just architecture scale.

## Quick Start

Run the local demo server from the `Online-demo` folder:

```bash
cd /Users/sanjayprasads/Desktop/Coding/Python/NN_training/Online-demo
./run_demo.sh
```

Then open one of these pages:

- `http://127.0.0.1:8000/Online-demo/transformer-model.html`
- `http://127.0.0.1:8000/Online-demo/comparison.html`
- `http://127.0.0.1:8000/Online-demo/mediapipe-calibration.html`
- `http://127.0.0.1:8000/Online-demo/index.html`

Do not open the HTML files with `file://`.

## Online Demo Pages

- `transformer-model.html`
  - Loads the tiny transformer ONNX model.
  - Runs browser inference on audio and shows framewise blendshape output.
  - Exports results locally in the browser.

- `comparison.html`
  - Compares three systems on the same audio:
    - heuristic viseme baseline
    - old TCN neural network
    - tiny transformer neural network
  - Shows model stats for each:
    - architecture
    - parameter count
    - ONNX size
    - context/input-output notes
  - Shows side-by-side playback outputs for comparison.

- `mediapipe-calibration.html`
  - Uploads the original source video and matching MediaPipe JSON.
  - Plays the source video next to the raccoon head driven by extracted coefficients.
  - Can also run live webcam tracking in-browser and drive the raccoon directly.
  - Helps decide whether target calibration is needed before training.

## Demo Assets

The main browser demo assets live in [assets](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets):

- [best_tcn_model_train_50.onnx](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/best_tcn_model_train_50.onnx)
- [tiny_transformer_full10s_l1.onnx](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/tiny_transformer_full10s_l1.onnx)
- [sample-audio.wav](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/sample-audio.wav)

The browser pages try local assets first and then fall back to CDN/GitHub-hosted copies when needed.

## V2A Project

Goal:

- map voice to facial animation
- predict `52 blendshapes + 7 pose values`
- keep models small and practical for real-time use

Current focus:

- validate the pipeline with narrow overfit demos first
- compare the old TCN against a newer tiny transformer baseline
- keep the repo usable for browser demos and local experiments

Main working area:

- [V2A-over-training-old-nn](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn)
- [runpod_results](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/runpod_results)
- [infra/runpod](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/infra/runpod)

Important docs:

- [GUIDE.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/GUIDE.md)
- [RUNPOD_TRAINING_REPORT.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/RUNPOD_TRAINING_REPORT.md)
- [README_tiny_transformer.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/README_tiny_transformer.md)
- [TinyTransformer_Overfit_Architecture.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/documentation/TinyTransformer_Overfit_Architecture.md)
- [TinyTransformer_Quick_Status.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/documentation/TinyTransformer_Quick_Status.md)
- [TinyTransformer_Conference_FAQ.md](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/documentation/TinyTransformer_Conference_FAQ.md)
- [Planning Transformer Questions](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/Planning%20Transformer%20Questions/README.md)

## Notes

- The repository is being developed on macOS and then tested in browser and GPU environments as needed.
- Runpod is optional for heavier training; it is not required to run the demo pages locally.
- If the hosted demo shows a local asset `404`, the page should still fall back to the remote asset sources.

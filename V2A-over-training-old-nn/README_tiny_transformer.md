# Tiny Transformer README

## What Exists Now

There are now two transformer-ready datasets in this repo:

1. Small speaker-specific overfit set
   - path: `V2A-over-training-old-nn/data/extracted_features`
   - shape: `audio (12, 1000, 80)`
   - shape: `targets (12, 1000, 59)`
   - use this when you want the fastest possible memorization demo

2. Full long-context rebuilt set
   - path: `V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500`
   - shape: `audio (798, 1000, 80)`
   - shape: `targets (798, 1000, 59)`
   - use this when you want real `10-second` context across the full speaker archive

## Where The Full Long-Context Set Came From

The original larger training artifact in this repo is:

- `V2A-over-training-old-nn/2_architecture_training/data/train`

That artifact is stored as:

- `audio (40743, 23, 80)`
- `targets (40743, 23, 59)`
- `vad (40743, 23)`

Those are overlapping `23-frame` windows with an `11-frame` step.

The rebuild script stitches matching overlaps back into continuous segments, then cuts true `10-second` windows:

- source continuous segments found: `163`
- segments long enough for 10-second windows: `83`
- rebuilt output windows: `798`
- rebuilt window length: `1000 frames`
- rebuilt window step: `500 frames`
- feature rate used: `100 Hz`

## Rebuild The Full 10-Second Dataset

Use:

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/rebuild_long_context_dataset.py \
  --input-dir V2A-over-training-old-nn/2_architecture_training/data/train \
  --output-dir V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500 \
  --window-frames 1000 \
  --window-step-frames 500 \
  --min-segment-frames 1000 \
  --sample-rate-hz 100 \
  --cover-tail
```

Output files:

- `audio_sequences.npy`
- `target_sequences.npy`
- `vad_sequences.npy`
- `segment_ids.npy`
- `window_start_frames.npy`
- `dataset_metadata.json`

## Train On The Small 10-Second Overfit Set

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py \
  --data-dir V2A-over-training-old-nn/data/extracted_features \
  --epochs 400 \
  --batch-size 2 \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --num-layers 3 \
  --ffn-dim 256 \
  --temporal-weight 0.0 \
  --checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --last-checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_last.pth \
  --history-path V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_l1_history.json
```

## Train On The Full Rebuilt 10-Second Dataset

This is the proper long-context all-data command:

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py \
  --data-dir V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500 \
  --epochs 160 \
  --batch-size 2 \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --num-layers 3 \
  --ffn-dim 256 \
  --dropout 0.0 \
  --temporal-weight 0.0 \
  --val-sequences 0 \
  --val-fraction 0.15 \
  --segment-aware-split \
  --checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --last-checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_last.pth \
  --history-path V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_full10s_l1_history.json
```

Why this split is safe:

- it uses `segment_ids.npy`
- train and val are separated by reconstructed continuous segments
- no overlapping windows cross the split boundary

## Export To ONNX

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/export_tiny_transformer_onnx.py \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --output V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.onnx \
  --manifest V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.json \
  --seq-len 1000
```

## Local Inference Script

Inference script:

- `V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py`

Audio input example:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --audio V2A-over-training-old-nn/2_architecture_training/sample_audio/sample.wav \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/tiny_transformer_demo_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/tiny_transformer_demo_100hz.npy \
  --output-json V2A-over-training-old-nn/data/inference/tiny_transformer_demo_30fps.json \
  --max-duration 10 \
  --fixed-seconds 10 \
  --device cpu
```

Feature input example:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --features /path/to/feature_sequence.npy \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/tiny_transformer_features_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/tiny_transformer_features_100hz.npy \
  --fixed-seconds 10 \
  --device cpu
```

Fixed `10-second` output contract:

- model-rate output: `(1000, 59)`
- `30 FPS` export: `(300, 59)`

## Arbitrary-Length Audio

Yes, arbitrary-length audio is possible.

The model is trained on `10-second` windows, but inference can now run in chunks:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --audio /path/to/long_audio.wav \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/long_audio_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/long_audio_100hz.npy \
  --chunk-seconds 10 \
  --chunk-overlap-seconds 5 \
  --device cpu
```

What this does:

- split long audio features into overlapping `10-second` chunks
- run the transformer on each chunk
- blend overlapping predictions back into one continuous `100 Hz` sequence
- downsample the full sequence to `30 FPS`

If the input audio is `T` seconds long, the output will be approximately:

- `round(T * 30)` frames
- each frame has `59` values

So yes, the system can return a full-length `30 FPS` blendshape track for both short and long audio.

## Browser Path

Yes, the transformer works through `ONNX` with `onnxruntime-web` and `WASM`.

Do not open the HTML directly with `file://`.
Run a local server instead.

Recommended local demo command from the repo root:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/Online-demo/transformer-model.html
```

Comparison page:

```text
http://127.0.0.1:8000/Online-demo/comparison.html
```

Current browser files:

- transformer-only page: `Online-demo/transformer-model.html`
- comparison page: `Online-demo/comparison.html`
- ONNX asset: `assets/tiny_transformer_full10s_l1.onnx`
- ONNX manifest: `assets/tiny_transformer_full10s_l1.json`
- both pages now chunk long audio into overlapping `10-second` ONNX windows

The ONNX export keeps the expected browser contract:

- input name: `audio_features`
- output name: `blendshapes`
- output shape: `(batch, sequence_length, 59)`

## Which Model To Use

Recommended default now:

- `tiny_transformer_full10s_l1_best.pth`
- `tiny_transformer_full10s_l1.onnx`
- `assets/tiny_transformer_full10s_l1.onnx`

The transformer HTML pages now point to the full-data ONNX by default.

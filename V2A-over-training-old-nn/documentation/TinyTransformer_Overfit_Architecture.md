# Tiny Transformer Architecture

## Goal

Replace or supplement the old TCN path with a small transformer encoder that maps per-frame audio features to:

- `52` blendshapes
- `7` pose values

Total output per frame:

- `59 values`

## Input And Output Contract

Input tensor:

- shape: `(batch, sequence_length, 80)`

Output tensor:

- shape: `(batch, sequence_length, 59)`

Channel meaning:

- `0:52` -> blendshapes
- `52:59` -> pose

Output activation ranges:

- blendshapes use `sigmoid`
- pose uses `tanh * 0.2`

## Model File

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_model.py`

## Default Tiny Transformer Config

- `input_dim = 80`
- `output_dim = 59`
- `d_model = 128`
- `nhead = 4`
- `num_layers = 3`
- `dim_feedforward = 256`
- `dropout = 0.0` to `0.1`
- `max_seq_len = 1200`
- `pose_dims = 7`

Approximate size:

- `432,443` parameters
- about `1.65 MB` fp32

## Forward Structure

1. Project input mel features from `80 -> d_model`
2. Add sinusoidal positional encoding
3. Pass through `TransformerEncoder`
4. Apply output head back to `59` channels
5. Clamp ranges with blendshape and pose-specific activations

## Datasets

### Small Overfit Set

Path:

- `V2A-over-training-old-nn/data/extracted_features`

Shape:

- `audio (12, 1000, 80)`
- `targets (12, 1000, 59)`
- `vad (12, 1000)`

Use case:

- fastest speaker-specific memorization demo

### Full Rebuilt Long-Context Set

Path:

- `V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500`

Shape:

- `audio (798, 1000, 80)`
- `targets (798, 1000, 59)`
- `vad (798, 1000)`

Metadata sidecars:

- `segment_ids.npy`
- `window_start_frames.npy`

Use case:

- train on the full speaker archive while keeping real `10-second` context

## How The Full 10-Second Dataset Is Built

Source artifact:

- `V2A-over-training-old-nn/2_architecture_training/data/train`

Source shape:

- `audio (40743, 23, 80)`
- `targets (40743, 23, 59)`

Source sampling layout:

- window length: `23 frames`
- step size: `11 frames`

Rebuild script:

- `V2A-over-training-old-nn/2_architecture_training/rebuild_long_context_dataset.py`

Rebuild steps:

1. Compare overlapping frames between consecutive short windows
2. Start a new segment when overlap no longer matches
3. Stitch matching windows into continuous per-speaker timelines
4. Emit `1000-frame` windows with a `500-frame` step
5. Save segment ids for clean train/val splitting

Observed rebuild stats:

- continuous segments found: `163`
- segments kept for `10-second` windows: `83`
- output windows: `798`
- feature rate: `100 Hz`
- output window duration: `10 seconds`

## Training File

- `V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py`

## Training Defaults

- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR`
- gradient clipping: `1.0`
- mixed precision on CUDA: enabled

Loss options:

- `L1`
- `L1 + temporal smoothness`

Temporal term meaning:

- compute frame-to-frame differences for prediction and target
- apply `L1` to those deltas
- add it with a small weight such as `0.02`

## Split Strategy

### Small Overfit Set

The small `10-second` set overlaps heavily, so it uses a contiguous split with a dropped gap:

- train windows: `0..5`
- gap windows: `6..9`
- val windows: `10..11`

### Full Rebuilt Set

The rebuilt all-data set uses segment-aware splitting:

- split by `segment_ids.npy`
- reserve final reconstructed segments for validation
- never put overlapping windows from the same continuous segment in both train and val

Example validated split:

- train windows: `675`
- val windows: `123`
- train segments: `68`
- val segments: `15`

## Why This Architecture Fits The Project

- It is small enough for a budget GPU like a `3090`.
- It keeps the same per-frame input/output contract as the current pipeline.
- It supports exact `10-second` windows for offline and browser demos.
- It is simple enough to export to ONNX and run in `onnxruntime-web`.

## Inference Contract

Inference file:

- `V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py`

When `--fixed-seconds 10` is used:

- features are forced to exactly `1000` frames
- model output is exactly `(1000, 59)`
- exported `30 FPS` output is exactly `(300, 59)`

For arbitrary-length audio:

- use `--chunk-seconds 10 --chunk-overlap-seconds 5`
- inference runs the model over overlapping `10-second` chunks
- overlapping predictions are blended back into one continuous timeline
- the final output is then exported at `30 FPS` for the full clip duration

## ONNX Contract

Export file:

- `V2A-over-training-old-nn/2_architecture_training/export_tiny_transformer_onnx.py`

ONNX names:

- input: `audio_features`
- output: `blendshapes`

Browser path:

- `onnxruntime-web`
- `WASM`

Existing demo pages already prepared:

- `Online-demo/transformer-model.html`
- `Online-demo/comparison.html`

Both browser pages now use chunked ONNX inference for long audio instead of sending the full clip in one call.

## Recommended Experiments

1. Small overfit set with `L1 only`
2. Small overfit set with `L1 + temporal`
3. Full rebuilt set with `L1 only`
4. Full rebuilt set with `L1 + temporal` if the first pass is too jittery

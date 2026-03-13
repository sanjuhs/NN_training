# Tiny Transformer Quick Status

Date:

- `2026-03-13`

## GPU Confirmation

Runpod pod:

- GPU: `NVIDIA GeForce RTX 3090`
- VRAM: `24 GB`
- Torch: `2.8.0+cu128`
- CUDA available: `True`
- CUDA device count: `1`

Direct CUDA sanity check on the pod succeeded:

- tensor device: `cuda:0`

Important:

- if `nvidia-smi` shows `0%` utilization now, that means the pod is idle
- it does **not** mean training used CPU
- the saved training runs were launched with `--device cuda`
- the trainer log itself printed `Device: cuda`

Why CPU may still appear in some places:

- browser demo uses `onnxruntime-web` with `WASM`, which is browser CPU by default
- some local example commands used `--device cpu` only for portability on the Mac
- an idle pod shows `0%` GPU even if earlier training used the GPU correctly

## Model

File:

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_model.py`

Config:

- `input_dim = 80`
- `output_dim = 59`
- `d_model = 128`
- `nhead = 4`
- `num_layers = 3`
- `dim_feedforward = 256`
- `dropout = 0.0`
- `max_seq_len = 1200`

Size:

- `432,443` parameters
- about `1.65 MB`

Output:

- `52` blendshapes
- `7` pose values
- total `59` values per frame

## Dataset Used For The Proper Full Run

Original full artifact:

- `V2A-over-training-old-nn/2_architecture_training/data/train`
- shape: `audio (40743, 23, 80)`

Rebuilt long-context dataset:

- `V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500`
- shape: `audio (798, 1000, 80)`
- shape: `targets (798, 1000, 59)`
- usable segments: `83`
- usable duration: about `1.165 hours`

Clean split:

- train windows: `675`
- val windows: `123`

## Losses In Simple Terms

### L1 loss

This is just:

- prediction should be close to the real target value at every frame

If target is `0.8` and prediction is `0.5`, the error is `0.3`.

This was the best loss for the current full-data run.

### Temporal loss

This adds:

- not only match each frame
- also match how the face changes from one frame to the next

This can make motion smoother, but in this project it was slightly worse than plain L1 on validation.

## Results

### Best full-data model

- checkpoint: `tiny_transformer_full10s_l1_best.pth`
- best epoch: `14`
- best val loss: `0.07287662368147604`
- best train loss at that epoch: `0.06325812685198685`

Direct reconstruction metrics:

- train MAE: `0.06297550350427628`
- val MAE: `0.07299423962831497`

### Full-data temporal comparison

- checkpoint: `tiny_transformer_full10s_l1temp_best.pth`
- best epoch: `8`
- best val loss: `0.07439237220152732`

Conclusion:

- `L1 only` is better than `L1 + temporal` right now

## Output Behavior

### Fixed 10-second inference

Verified:

- input `10.0s`
- output `1000 x 59` at `100 Hz`
- output `300 x 59` at `30 FPS`

### Arbitrary-length inference

Supported now through chunking:

- split long audio into overlapping `10-second` chunks
- run transformer on each chunk
- blend overlaps
- export the full sequence at `30 FPS`

Verified:

- input `25.0s`
- output `2500 x 59` at `100 Hz`
- output `750 x 59` at `30 FPS`

## ONNX

Best ONNX export:

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.onnx`
- copied browser asset: `assets/tiny_transformer_full10s_l1.onnx`

Parity check:

- ONNX vs PyTorch max abs diff: `3.2782554626464844e-07`

## Demo Hookup

Updated pages:

- `Online-demo/transformer-model.html`
- `Online-demo/comparison.html`

Current behavior:

- both pages default to `tiny_transformer_full10s_l1.onnx`
- both pages can load local asset first
- both pages now use chunked ONNX inference for long audio

## What To Do Before A Demo

1. Push the repo changes, including `assets/tiny_transformer_full10s_l1.onnx`.
2. Make sure your deployment serves the updated `Online-demo` files.
3. Open the transformer page or comparison page.
4. Test one short audio file before showing it live.

## SSH And Pod Safety

Stopping the SSH session:

- safe, as long as no training job is currently running

Stopping the Runpod pod:

- this is what stops billing

Important:

- closing SSH does **not** stop the pod
- if no training is running, you can safely disconnect SSH now

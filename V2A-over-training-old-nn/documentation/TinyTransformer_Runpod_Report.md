# Tiny Transformer Runpod Report

Date:

- `2026-03-13`

## Objective

Train a tiny transformer on the existing 10-second speaker-specific dataset and verify:

- the model can overfit the narrow data regime
- fixed 10-second inference produces exact `300 x 59` output at `30 FPS`
- ONNX export stays compatible with the current demo interface

## Runpod Environment

- provider: Runpod
- template: PyTorch Environment
- GPU: `NVIDIA GeForce RTX 3090`
- VRAM: `24 GB`
- Python: `3.12.3`
- Torch: `2.8.0+cu128`

## Dataset Used

Path:

- `V2A-over-training-old-nn/data/extracted_features`

Shapes:

- audio: `(12, 1000, 80)`
- targets: `(12, 1000, 59)`
- vad: `(12, 1000)`

Split used:

- train windows: `0..5`
- dropped gap windows: `6..9`
- val windows: `10..11`

Reason:

- windows overlap by `8 seconds`, so the boundary gap is required to avoid leakage.

## Experiments

Common config:

- `d_model = 128`
- `nhead = 4`
- `num_layers = 3`
- `dim_feedforward = 256`
- `dropout = 0.0`
- `batch_size = 2`
- `lr = 5e-4`
- `epochs = 500`
- `patience = 120`
- `device = cuda`

### Experiment A

- name: `L1 only`
- temporal weight: `0.0`
- checkpoint: `models/tiny_transformer_l1_best.pth`

Result:

- best validation loss: `0.08262570947408676`
- best epoch: `58`
- early stop epoch: `178`

Direct reconstruction check:

- train MAE: `0.048005`
- val MAE: `0.082620`
- first train window MAE: `0.055157`
- first val window MAE: `0.076845`

### Experiment B

- name: `L1 + temporal`
- temporal weight: `0.02`
- checkpoint: `models/tiny_transformer_l1temp_best.pth`

Result:

- best validation loss: `0.08284632116556168`
- best epoch: `58`
- early stop epoch: `178`

Direct reconstruction check:

- train MAE: `0.047836`
- val MAE: `0.082707`
- first train window MAE: `0.055110`
- first val window MAE: `0.076930`

## Best Checkpoint

Recommended current checkpoint:

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth`

Reason:

- lowest held-out validation loss among the tested runs
- same output contract as the temporal version
- simplest behavior for the first demo pass

## Output Shape Verification

Fixed-length inference test:

- input audio: `V2A-over-training-old-nn/data/extracted_features/extracted_audio.wav`
- duration forced with: `--fixed-seconds 10 --max-duration 10`

Observed output:

- model-rate output: `(1000, 59)`
- exported `30 FPS` output: `(300, 59)`

This means:

- yes, for a fixed 10-second inference window the current transformer pipeline produces exactly:
  - `300 frames`
  - `59 values per frame`
  - total scalar outputs: `300 * 59 = 17,700`

Important qualification:

- this exact shape is guaranteed only when fixed-duration inference is used.
- that is why the inference script now supports `--fixed-seconds 10`.

## ONNX Verification

Exported files:

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1.json`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp.json`

ONNXRuntime parity check against PyTorch for the L1 model:

- PyTorch output shape: `(1, 1000, 59)`
- ONNX output shape: `(1, 1000, 59)`
- max absolute difference: `5.662441253662109e-07`
- mean absolute difference: `4.101993411609328e-08`

Interpretation:

- ONNX export is numerically consistent with the PyTorch checkpoint.

## What We Can And Cannot Claim

What is confirmed:

- the transformer trains correctly on GPU
- it fits the training regime much more closely than the held-out windows
- the fixed 10-second inference path gives exact `300 x 59` output
- the ONNX export is valid and numerically aligned with PyTorch

What is not safe to claim as a guarantee:

- that the model is perfectly overtrained in a visual sense
- that every output sequence looks good enough for demo playback without qualitative review
- that unseen audio will animate well

The current results support:

- a working speaker-specific overfit demo candidate

The current results do not support:

- a guarantee of natural motion on new audio without more testing

## Recommended Next Use

For the first demo:

1. Use `tiny_transformer_l1_best.pth`
2. Export or serve `tiny_transformer_l1.onnx`
3. Run inference with `--fixed-seconds 10`
4. Feed the resulting `300 x 59` output into the existing demo path

## Relevant Local Artifacts

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1.json`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp_best.pth`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp.json`
- `V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_l1_history.json`
- `V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_l1temp_history.json`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_l1_trainvoice_100hz.npy`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_l1_trainvoice_30fps.npy`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_l1_trainvoice_30fps.json`

## Full-Data Long-Context Run

After the small overfit pass, the full short-window dataset was rebuilt into true `10-second` windows.

Rebuilt dataset:

- source artifact: `V2A-over-training-old-nn/2_architecture_training/data/train`
- source shape: `audio (40743, 23, 80)`
- rebuilt artifact: `V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500`
- rebuilt shape: `audio (798, 1000, 80)`
- rebuilt shape: `targets (798, 1000, 59)`
- usable continuous segments: `83`
- usable long-context duration: about `1.165 hours`

Correct full-data split:

- train windows: `675`
- val windows: `123`
- train segments: `68`
- val segments: `15`

### Full-Data Experiment A

- name: `full10s_l1`
- checkpoint: `models/tiny_transformer_full10s_l1_best.pth`
- best epoch: `14`
- best validation loss: `0.07287662368147604`

Direct reconstruction check:

- train MAE: `0.06297550350427628`
- val MAE: `0.07299423962831497`
- first train window MAE: `0.06841298192739487`
- first val window MAE: `0.07889895886182785`

### Full-Data Experiment B

- name: `full10s_l1temp`
- checkpoint: `models/tiny_transformer_full10s_l1temp_best.pth`
- best epoch: `8`
- best validation loss: `0.07439237220152732`

Result:

- `L1 only` is still the better corrected full-data checkpoint

### Full-Data ONNX

Exported files:

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.json`

ONNX parity against PyTorch:

- PyTorch output shape: `(1, 1000, 59)`
- ONNX output shape: `(1, 1000, 59)`
- max absolute difference: `3.2782554626464844e-07`
- mean absolute difference: `3.289947869689058e-08`

### Full-Data Inference Checks

Fixed `10-second` inference with the corrected full-data checkpoint:

- output `100 Hz`: `(1000, 59)`
- output `30 FPS`: `(300, 59)`

Arbitrary-length chunked inference test:

- input features: `(2500, 80)` which is `25 seconds` at `100 Hz`
- output `100 Hz`: `(2500, 59)`
- output `30 FPS`: `(750, 59)`

Interpretation:

- yes, the trained transformer can produce a full-length blendshape track for longer clips
- the long-form path works by chunking into overlapping `10-second` windows and stitching the results

### Additional Local Artifacts

- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.onnx`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.json`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1temp_best.pth`
- `V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_full10s_l1_history.json`
- `V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_full10s_l1temp_history.json`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_full10s_l1_trainvoice_100hz.npy`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_full10s_l1_trainvoice_30fps.npy`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_full10s_l1_trainvoice_30fps.json`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_full10s_l1_long_100hz.npy`
- `V2A-over-training-old-nn/data/inference/tiny_transformer_full10s_l1_long_30fps.npy`

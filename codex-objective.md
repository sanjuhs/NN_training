# Codex Objective

## Project

Repo: `sanjuhs/NN_training`

Pipeline focus: `V2A-over-training-old-nn`

Primary goal: build a speaker-specific overfit demo that replaces or supplements the old TCN with a tiny transformer for audio-to-blendshape prediction, using the current repo preprocessing and dataset format with the smallest possible code changes.

## What We Are Optimizing For

- Working demo in a few hours, not generalization.
- Reuse existing mel-feature pipeline and existing target format.
- Keep the inference/export path compatible with the current ONNX demo flow where possible.
- Train on Runpod cheaply and access it over SSH from Codex.

## Current Repo Structure

- `V2A-over-training-old-nn/1_data_cleaning`
  - data extraction, MediaPipe blendshape extraction, audio mel extraction, dataset creation, Hugging Face upload script.
- `V2A-over-training-old-nn/2_architecture_training`
  - old TCN training code, old downloaded Hugging Face train/test datasets, model exports.
- `V2A-over-training-old-nn/3_inference`
  - Python inference pipeline for mel extraction, model inference, and JSON output.

## Existing Data Artifacts Found

### 1. Long-context local dataset already present

Path:

- `V2A-over-training-old-nn/data/extracted_features/audio_sequences.npy`
- `V2A-over-training-old-nn/data/extracted_features/target_sequences.npy`
- `V2A-over-training-old-nn/data/extracted_features/vad_sequences.npy`
- `V2A-over-training-old-nn/data/extracted_features/dataset_metadata.json`

Observed shapes:

- `audio_sequences.npy`: `(12, 1000, 80)`
- `target_sequences.npy`: `(12, 1000, 59)`
- `vad_sequences.npy`: `(12, 1000)`

Metadata:

- `sequence_length_frames`: `1000`
- `sequence_length_ms`: `10000`
- `step_size_frames`: `200`
- `overlap_ms`: `8000`
- `audio_feature_dim`: `80`
- `target_feature_dim`: `59`

Interpretation:

- This is the best artifact for the overfit demo because it already gives 10-second windows.
- It is small, speaker-specific, and fits the demo objective.
- The old `train.py` currently loads this root-level dataset, not the Hugging Face train split.

### 2. Short-window dataset produced by data cleaning

Path:

- `V2A-over-training-old-nn/1_data_cleaning/data/training_dataset`

Observed shapes:

- `audio_sequences.npy`: `(277, 24, 80)`
- `target_sequences.npy`: `(277, 24, 59)`
- `vad_sequences.npy`: `(277, 24)`

Interpretation:

- This is a short-context local dataset.
- It is not the best choice for the requested 10-second transformer overfit demo.

### 3. Hugging Face-style train/test copies already downloaded

Paths:

- `V2A-over-training-old-nn/2_architecture_training/data/train`
- `V2A-over-training-old-nn/2_architecture_training/data/test`

Observed shapes:

- Train `audio_sequences.npy`: `(40743, 23, 80)`
- Train `target_sequences.npy`: `(40743, 23, 59)`
- Train `vad_sequences.npy`: `(40743, 23)`
- Test `audio_sequences.npy`: `(277, 24, 80)`
- Test `target_sequences.npy`: `(277, 24, 59)`
- Test `vad_sequences.npy`: `(277, 24)`

Interpretation:

- These are short overlapping windows.
- Useful as legacy reference, but not the right first choice for a fast 10-second overfit demo.

## Dataset Format Findings

### Audio features

Source:

- `V2A-over-training-old-nn/1_data_cleaning/2_extract_audio_features.py`
- `V2A-over-training-old-nn/data/extracted_features/audio_features.json`

Confirmed:

- Audio is resampled to `16 kHz`.
- Features are `80-bin log-mel spectrogram` frames.
- `hop_length = 160` at `16 kHz`, so mel frame rate is `100 Hz`.
- `win_length = 400` and `n_fft = 512`.
- The raw test feature JSON contains keys:
  - `audio_path`
  - `sample_rate`
  - `duration_seconds`
  - `n_mels`
  - `hop_length`
  - `mel_frame_rate`
  - `n_frames`
  - `timestamps`
  - `mel_features`
  - `voice_activity`
  - `zero_crossing_rate`
  - `rms_energy`

### Targets

Source:

- `V2A-over-training-old-nn/1_data_cleaning/3_create_datset.py`
- `V2A-over-training-old-nn/data/extracted_features/blendshapes_and_pose.json`

Confirmed:

- Targets are not 52 only.
- Targets are `59` values per frame:
  - `52 blendshapes`
  - `7 head pose values`
- The current dataset builder explicitly concatenates `52 blendshapes + 7 pose`.
- For compatibility with the existing demo/inference path, the new transformer should default to outputting `59` values.

### Visual frame rate

Confirmed:

- Raw visual frames are extracted around `30 FPS`.
- They are interpolated onto the `100 Hz` audio timeline during dataset creation.
- Inference then downsamples predictions back to `30 FPS` for output.

## Hugging Face Repos Referenced In The Repo

From local repo references:

- Train dataset repo id: `sanjuhs/audio_to_blendshapes_main`
- Test dataset repo id: `sanjuhs/audio_to_blendshapes_test`

Direct URLs:

- `https://huggingface.co/datasets/sanjuhs/audio_to_blendshapes_main`
- `https://huggingface.co/datasets/sanjuhs/audio_to_blendshapes_test`

Notes:

- The repo download summary under `V2A-over-training-old-nn/2_architecture_training/data/train/download_summary.json` points to `sanjuhs/audio_to_blendshapes_main`.
- The upload script under `V2A-over-training-old-nn/1_data_cleaning/4_upload_dataset_hf.py` still points to `sanjuhs/audio_to_blendshapes_test`.
- For this overfit demo, we should use the already-present local 10-second dataset under `V2A-over-training-old-nn/data/extracted_features`, not the short-window Hugging Face split.

## Recommended Training Dataset For This Phase

Use:

- `V2A-over-training-old-nn/data/extracted_features`

Reason:

- It already supports 10-second context.
- It is small enough to memorize quickly.
- It matches the speaker-specific overfit objective.
- It minimizes preprocessing work.

## Recommended Tiny Transformer

First-pass model:

- `d_model = 128`
- `nhead = 4`
- `num_layers = 3`
- `dim_feedforward = 256`
- `dropout = 0.1`
- `input_dim = 80`
- `output_dim = 59`

Training defaults:

- Loss: `L1 + optional temporal smoothness`
- Gradient clipping: `1.0`
- Batch size target on 24 GB GPU: `2` to `4`
- Mixed precision: enabled on CUDA
- Save best checkpoint on validation loss
- Log train and val loss every epoch

## Leakage-Safe Split Strategy For The 10-Second Dataset

Important detail:

- The 10-second local dataset uses overlapping windows:
  - length `1000` frames
  - step `200` frames

Therefore:

- A naive random split is wrong.
- A naive contiguous split by window index is also still leakage-prone because neighboring windows overlap by `8 seconds`.

Recommended split:

- Use the current `(12, 1000, 80)` dataset.
- Split by contiguous window order.
- Insert a dropped gap between train and val large enough to remove overlap across the split boundary.

Practical first pass:

- Train on earliest contiguous windows.
- Drop a 4-window boundary gap.
- Validate on the final 2 windows.

This preserves the fast demo goal while avoiding split leakage.

## Recommended Runpod Setup

### Pod type

Use:

- `Runpod PyTorch` official template

Why:

- Official Runpod PyTorch templates already support SSH over exposed TCP.
- This minimizes setup time.
- Jupyter and standard ML tooling are already prepared.

### Recommended image/template choice

Best practical choice for this repo:

- Use the latest `Runpod PyTorch` template or a PyTorch image equivalent to:
  - `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

Reason:

- The repo already pins `torch==2.8.0` and `torchaudio==2.8.0`.
- Runpod documentation currently shows a PyTorch 2.8.0 CUDA 12.8.1 Ubuntu 24.04 base image for custom templates.
- This avoids unnecessary environment drift.

### Recommended GPU for a $19 credit budget

Primary recommendation:

- `1x RTX 3090 24 GB` on Community Cloud, on-demand if available.

Fallbacks:

- `1x RTX A5000 24 GB`
- `1x L4 24 GB`
- `1x RTX 4090 24 GB` if price difference is small and availability is better

Why 24 GB is enough:

- The dataset is tiny.
- The transformer is tiny.
- 10-second sequences of shape `(batch, 1000, 80)` are manageable on a 24 GB card.

Cost guidance:

- Runpod docs show a 1x RTX 3090 example with `uninterruptablePrice = 0.3` USD/hour and `minimumBidPrice = 0.163` USD/hour.
- With `$19`, even ignoring small storage charges, you have enough budget for a several-hour on-demand training session and many more hours if spot pricing is acceptable.

Safe recommendation:

- Use on-demand for the first real training run to avoid interruptions.
- Only use spot after checkpointing is verified.

### CPU / RAM / disk

Recommended minimums:

- `1 GPU`
- `8 to 16 vCPU`
- `32 GB RAM` preferred
- `50 GB container disk`
- `50 GB volume disk`

Why:

- Training itself is light.
- Disk is mostly for the repo, caches, exported checkpoints, and ONNX files.
- `/workspace` is the right persistent mount path on Runpod.

### Where to put Runpod config in this repo

Keep all Runpod-specific repo files under:

- `infra/runpod/`

Planned contents:

- `infra/runpod/README.md`
- `infra/runpod/pod.env.example`
- `infra/runpod/ssh-config-example`
- optionally `infra/runpod/create-pod.json`

Reason:

- Keeps infrastructure setup out of model code.
- Makes it easy to recreate the pod and SSH config later.

### Runpod pod settings to use

Recommended pod settings:

- Template: `Runpod PyTorch`
- Cloud: `Community` if cheaper and available, otherwise `Secure`
- GPU: `RTX 3090 24 GB` first choice
- Public IP: enabled
- SSH Terminal Access: enabled
- Exposed ports:
  - `22/tcp`
  - `8888/http`
- Volume mount path: `/workspace`
- Container disk: `50 GB`
- Volume disk: `50 GB`
- CUDA filter: choose a version compatible with the template, ideally `12.8` if using the Runpod PyTorch 2.8 image

### How Codex should SSH into the pod

Runpod side:

1. Add your public SSH key to your Runpod account.
2. Deploy the Pod with SSH Terminal Access enabled.
3. Open the Pod’s `Connect` tab.
4. Copy the `SSH over exposed TCP` command.

Local machine side:

1. Add a host entry to `~/.ssh/config`.
2. Point it at the Runpod IP and mapped SSH port.
3. Use that host alias from Codex.

Suggested `~/.ssh/config` entry:

```sshconfig
Host nn-training-runpod
    HostName <RUNPOD_PUBLIC_IP>
    User root
    Port <RUNPOD_TCP_PORT_22>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 120
```

Then connect with:

```bash
ssh nn-training-runpod
```

## PyTorch Recommendation

As of `2026-03-13`:

- PyTorch’s official get-started page lists `2.7.0` as the current `Stable` release.
- This repo currently pins `torch==2.8.0` and `torchaudio==2.8.0`.
- Runpod documentation currently shows a `torch 2.8.0 / CUDA 12.8.1` PyTorch base image.

Recommendation for this repo:

- Use `torch==2.8.0` on Runpod for this project right now.

Reason:

- Lowest friction with the current repo.
- Matches the current local requirements.
- Matches the current Runpod PyTorch image examples.
- No feature in this transformer plan requires downgrading to 2.7.0.

## Files We Intend To Add

- `codex-objective.md`
- `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_model.py`
- `V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py`
- `V2A-over-training-old-nn/2_architecture_training/export_tiny_transformer_onnx.py`
- `V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py`
- `infra/runpod/README.md`
- `infra/runpod/ssh-config-example`
- `infra/runpod/pod.env.example`

## Execution Order

1. Write the tiny transformer model.
2. Write a new training script that uses `V2A-over-training-old-nn/data/extracted_features`.
3. Use a leakage-safe contiguous split with a dropped boundary gap.
4. Save best checkpoint and training history.
5. Write inference for mel features or audio input.
6. Export ONNX with the same input/output contract:
   - input: `(batch, sequence_length, 80)`
   - output: `(batch, sequence_length, 59)`
7. Add exact Runpod and training commands.
8. Note the minimal change needed in the online demo:
   - either replace the old ONNX asset path
   - or update the default model URL/constants to the new transformer export

## Immediate Next Step

Implement the tiny transformer training, inference, and ONNX export scripts against the local 10-second dataset first, then add the Runpod helper files.

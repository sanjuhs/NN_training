# Runpod Setup

## Recommended Pod

- Template: `Runpod PyTorch`
- Cloud: `Community` first, `Secure` if availability is better
- GPU: `RTX 3090 24 GB`
- Good fallbacks: `RTX A5000 24 GB`, `L4 24 GB`, `RTX 4090 24 GB`
- Public IP: `Enabled`
- SSH Terminal Access: `Enabled`
- Ports:
  - `22/tcp`
  - `8888/http`
- Container disk: `50 GB`
- Volume disk: `50 GB`
- Volume mount path: `/workspace`

This project does not need a bigger GPU for the overfit demo. A single 24 GB GPU is enough.

## Why This Pod

- The dataset for the overfit demo is only `12 x 1000 x 80`.
- The transformer is tiny.
- We want the cheapest reliable GPU that can still train comfortably and export ONNX.

## Recommended Workspace Layout

Clone the repo into:

```bash
/workspace/NN_training
```

Use:

```bash
cd /workspace/NN_training
```

## SSH

1. Add your public key in Runpod account settings.
2. Deploy the pod with SSH enabled.
3. Copy the `SSH over exposed TCP` command from the pod `Connect` tab.
4. Add a host entry using `ssh-config-example`.

## Minimal Setup On The Pod

If the PyTorch template already has the right CUDA build, do not reinstall all of `torch` unless needed.

Recommended setup:

```bash
cd /workspace
python -m venv .venv
source /workspace/.venv/bin/activate
pip install --upgrade pip
git clone https://github.com/sanjuhs/NN_training.git
cd /workspace/NN_training
pip install numpy==1.26.4 librosa==0.11.0 scipy==1.16.1 matplotlib==3.10.5 tqdm==4.67.1 scikit-learn==1.7.1 soundfile==0.13.1 huggingface-hub==0.34.4 python-dotenv==1.1.1 moviepy==2.2.1 onnx onnxruntime
```

If `torch` is missing or mismatched:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## Training Command

```bash
cd /workspace/NN_training/V2A-over-training-old-nn/2_architecture_training
python train_tiny_transformer.py \
  --data-dir ../data/extracted_features \
  --epochs 400 \
  --batch-size 2 \
  --d-model 128 \
  --nhead 4 \
  --num-layers 3 \
  --ffn-dim 256 \
  --temporal-weight 0.02
```

## ONNX Export Command

```bash
cd /workspace/NN_training/V2A-over-training-old-nn/2_architecture_training
python export_tiny_transformer_onnx.py \
  --checkpoint models/tiny_transformer_overfit_best.pth \
  --output models/tiny_transformer_overfit.onnx \
  --manifest models/tiny_transformer_overfit.json \
  --seq-len 1000
```

## Inference Command

Feature input:

```bash
cd /workspace/NN_training/V2A-over-training-old-nn/3_inference
python tiny_transformer_inference.py \
  --features /path/to/one_feature_sequence.npy \
  --checkpoint ../2_architecture_training/models/tiny_transformer_overfit_best.pth \
  --output-npy ../data/inference/tiny_transformer_output_30fps.npy \
  --output-100hz-npy ../data/inference/tiny_transformer_output_100hz.npy
```

`/path/to/one_feature_sequence.npy` must have shape `(T, 80)` or `(1, T, 80)`.

Audio input:

```bash
cd /workspace/NN_training/V2A-over-training-old-nn/3_inference
python tiny_transformer_inference.py \
  --audio ../2_architecture_training/sample_audio/sample.wav \
  --checkpoint ../2_architecture_training/models/tiny_transformer_overfit_best.pth \
  --output-npy ../data/inference/tiny_transformer_audio_output_30fps.npy \
  --output-100hz-npy ../data/inference/tiny_transformer_audio_output_100hz.npy \
  --output-json ../data/inference/tiny_transformer_audio_output.json \
  --max-duration 10
```

## Online Demo Hookup

The current web demo expects:

- input name: `audio_features`
- output name: `blendshapes`
- output shape: `(batch, sequence_length, 59)`

The new export keeps that contract, so the minimal change is:

- replace the default ONNX file path with the new transformer ONNX

Current places to update if needed:

- `Online-demo/nn-model.html`
- `Online-demo/comparison.html`

If you copy the transformer ONNX over the old asset path instead, no HTML change is required.

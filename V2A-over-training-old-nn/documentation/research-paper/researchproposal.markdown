## Working Title

Audio-to-Blendshape Animation with Temporal Convolutional Networks: A Proof-of-Concept and Research Plan

### Authors

- Sanjay Prasad HS
- Affiliation: Independent Researcher (Citizen Scientist)
- Contact: sanjuhs123@gmail.com

### Abstract

We propose a Temporal Convolutional Network (TCN) approach for mapping speech audio to time-aligned facial blendshape and head pose trajectories for 3D character animation. Leveraging 80-bin Mel spectrograms at 100 Hz with auxiliary features (voice activity detection, zero-crossing rate, RMS), and synchronized visual targets at 30 Hz upsampled to 100 Hz by linear interpolation, our system predicts 52 blendshapes and 7 head pose parameters per frame. We present a proof-of-concept implementation using a compact 4-layer non-causal TCN and outline a research plan toward a full 10-layer causal TCN with 10+ second temporal receptive field and hybrid attention-TCN extensions. We detail dataset construction (~1.25 hours), loss design for temporal smoothness and silence handling, and real-time feasibility. We define quantitative metrics and user studies to compare against viseme-based pipelines and sequence models. This document serves both as an initial report and a structured roadmap for scaling to larger datasets (≈60 hours), broader expressivity, and production deployment.

## 1. Introduction

Natural audiovisual speech animation requires temporally coherent mapping from acoustic cues to facial muscle activations. Traditional pipelines rely on phoneme/viseme graphs or heuristics, which often produce rigid, off-timed motion. We investigate Temporal Convolutional Networks (TCNs) as an efficient, parallelizable alternative that captures multi-scale temporal dependencies with dilated convolutions and residual connections.

### 1.1 Problem Statement

- Predict time-varying facial blendshapes and head pose from speech audio with realistic timing and smoothness.
- Operate at real-time rates and generalize from limited data.

### 1.2 Contributions

- A complete audio-to-blendshape data pipeline with precise audio–visual synchronization at 100 Hz.
- A proof-of-concept 4-layer TCN baseline with multi-component losses (accuracy, temporal derivatives, silence, pose bounds).
- A research plan for a 10-layer causal TCN (10+ s receptive field) and hybrid attention integration for emotional context.
- An evaluation protocol against viseme pipelines and neural baselines, including quantitative metrics and human studies.

## 2. Related Work

### 2.1 Viseme/Phoneme-Driven Animation

Viseme rule-based systems and phoneme-to-pose mappings remain common in production but struggle with coarticulation, timing, and expressivity. We position our method as a data-driven alternative that directly predicts continuous muscle activations.

### 2.2 Sequence Models for Audiovisual Mapping

RNNs/LSTMs and Transformers model long-term dependencies but may be harder to deploy in low-latency contexts. TCNs offer long receptive fields with O(n) time and parallel training.

### 2.3 Dilated Convolutions and WaveNet

WaveNet popularized dilated causal convolutions for speech; we adopt similar dilation patterns for efficient temporal context while adapting the architecture for regression to blendshapes.

### 2.4 Talking-Head, Lip-Sync, and Blendshape Prediction

Neural lip-sync (e.g., SyncNet-based metrics), parametric animation (blendshape regression), and recent talking-head works inform our baselines and metrics.

[References placeholders; see Section 12]

## 3. Method

### 3.1 Data Representation and Synchronization

- Audio features at 100 Hz:
  - 80-bin Mel spectrograms (25 ms window, 10 ms hop, 16 kHz sample rate).
  - Aux features: voice activity detection (VAD via RMS), zero-crossing rate (ZCR), frame-wise RMS.
- Visual targets at 30 Hz:
  - 52 blendshapes + 7 head pose parameters (position x,y,z and rotation quaternion w,x,y,z).
- Synchronization:
  - Visual sequences linearly interpolated to 100 Hz to align with audio features.

```python
# Linear interpolation (conceptual)
sync_targets[:, i] = np.interp(sync_audio_timestamps, visual_timestamps, target_values)
```

### 3.2 Temporal Convolutional Networks and Dilation

- TCN backbone with 1D dilated convolutions, residual connections, and (optionally) depthwise separable convolutions for efficiency.
- Proof-of-concept (used): 4 layers, dilations [1, 2, 4, 8], ~0.15 s receptive field, centered padding (non-causal).
- Full-scale (planned): 10 layers, dilations [1, 2, 4, 8, 16, 32, 64, 128, 128, 128], ≈10.24 s receptive field at 100 Hz, causal padding for real-time streaming.
- Output activations: sigmoid for 52 blendshapes; tanh scaled by 0.2 for head pose.

### 3.3 Loss Functions

- Base loss: Smooth L1 (Huber) on all outputs.
- Temporal smoothness: first- and second-derivative losses to reduce jitter and enforce continuity.
- Silence loss: VAD-weighted penalty to keep mouth stable during non-speech.
- Pose clamping: penalty outside realistic pose bounds (±0.2 after tanh scaling).
- Staged schedule: introduce auxiliary losses progressively for stable convergence.

### 3.4 Training Procedure (POC)

- Optimizer: AdamW (lr, weight decay: TBD in final paper; current OneCycleLR schedule).
- Gradient clipping: max-norm 1.0.
- Sequence construction: 24 frames (240 ms) windows, 12-frame overlap (120 ms step).
- Epochs: 50 (POC); runtime ~2–3 hours on NVIDIA T4 (Google Colab).
- Batch size, channels, and dropout: TBD (document exact values from training logs).

### 3.5 Implementation Details

- Codebase: see `V2A-over-training-old-nn/2_architecture_training/` for training scripts and models.
- Models: `models/tcn_model.py` (POC), `models/tcn_10s_model.py` (design for full TCN).
- Plots and logs: `2_architecture_training/plots/` (loss curves, feature analyses).
- Inference prototype: `V2A-over-training-old-nn/3_inference/inference.py`.

## 4. Dataset

### 4.1 Source and Scale

- ~1.25 hours (≈75 minutes) of synchronized audio-visual data from 5 videos (~15 min each).
- Processed sequences: ≈40,743 sequences of 24 frames each (240 ms windows).
- Training/validation/test split: TBD (e.g., 80/10/10 by video).

### 4.2 Preparation

- Audio extraction: resampled to 16 kHz; MEL parameters as in Section 3.1.
- Visual extraction: MediaPipe blendshapes (52) and head pose (7) at target 30 FPS with frame skipping on high-FPS sources.
- Synchronization: linear interpolation to 100 Hz; quality filtering (≥50% face detection rate).

### 4.3 Release and Links

- Planned dataset artifacts hosted on Hugging Face (links TBD):
  - Audio features, target sequences, timestamps, metadata.
  - Train/val/test manifest files for reproducibility.

## 5. Baselines

- Viseme-driven pipeline with phoneme alignment and rule-based mapping to blendshapes.
- Linear regression from MELs to blendshapes.
- Recurrent baseline (GRU/LSTM) with comparable parameter count (TBD).
- Optional transformer-based temporal model (if compute permits).

## 6. Evaluation Protocol

### 6.1 Quantitative Metrics

- Mean Absolute Error (MAE): overall and per-region (jaw, lips, smile).
- Pearson correlation: key blendshapes (jawOpen, mouthSmile, lipFunnel, lipPucker).
- Dynamic Time Warping (DTW) distance: temporal alignment of lip aperture trajectories.
- Onset lag: cross-correlation lag between audio energy and jawOpen.
- Smoothness: L2 norm of first/second derivatives (lower is better, with realism caveat).
- Head pose MAE for 7 pose parameters.
- Optional lip-sync metrics (e.g., LSE-C/LSE-D) via surrogate mouth ROI if available.

### 6.2 Human Study

- Mean Opinion Score (MOS): naturalness, lip-sync accuracy, and expressivity (1–5 scale).
- Pairwise preference test: TCN vs viseme baseline, randomized trials.
- N ≈ 20–30 participants; report inter-rater reliability (Cronbach’s α).

### 6.3 Protocol Details

- Same audio inputs across methods; identical rig and rendering pipeline.
- Evaluate on held-out speakers and content.
- Report mean ± std and statistical significance (paired t-test or Wilcoxon).

## 7. Results (POC)

- Quantitative results: TBD. Targets for strong performance per Section 6.1: overall MAE < 0.05; jaw corr > 0.7; lips corr > 0.6; mouth MAE < 0.04.
- Qualitative: include sample animations and screenshots; note smoother coarticulation and reduced popping vs visemes.
- Runtime: proof-of-concept achieves real-time-capable inference on commodity GPU; mobile feasibility with separable convolutions (planned).

## 8. Ablation Studies (Planned)

- Architecture depth and dilation: 4 vs 10 layers; receptive field effects on sync and expressivity.
- Attention augmentation: TCN vs TCN+MHSA fusion.
- Loss components: remove temporal, silence, or pose penalties.
- Feature sets: MEL only vs MEL+VAD+ZCR+RMS.
- Sequence length and overlap: 240 ms vs multi-second windows.
- Dead neuron analysis: activation sparsity; prune or ablate low-magnitude neurons/filters and measure impact.

## 9. Scaling Plan

- Data: expand to ≈60 hours across multiple speakers, emotions, and languages.
- Model: 10-layer causal TCN with 256+ channels; maintain real-time streaming.
- Attention: integrate multi-head self-attention between TCN blocks for long-range emotional context.
- Modalities: extend to full-body animation (hands, torso) and prosody-aware head motion.
- Sampling rate: evaluate 50–100 Hz trade-offs; latency vs fidelity.
- Deployment: quantization and ONNX export for edge devices.

## 10. Implementation and Reproducibility

- Repository: this project; training scripts and configs under `2_architecture_training/`.
- Environment: Python; dependencies in `requirements.txt`; ONNX exports available.
- Seeds, hyperparameters, and exact checkpoints to be released with HF artifacts.
- Inference demo pages under `outputhtml/` for qualitative inspection.

## 11. Discussion

- Advantages over visemes: continuous control, better coarticulation, data-driven timing.
- Limitations: current data scale, short POC receptive field, non-causal training limiting live use.
- Ethical considerations: consent for facial data; potential misuse; demographic bias.

## 12. References (Placeholders)

- [1] Bai et al., “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,” 2018.
- [2] van den Oord et al., “WaveNet: A Generative Model for Raw Audio,” 2016.
- [3] Chung & Zisserman, “Out of time: automated lip sync in the wild,” 2016.
- [4] [Additional works on viseme-based animation].
- [5] [Talking-head and blendshape regression literature].
- [6] [Mobile-friendly depthwise separable convs].

## 13. Appendices

### A. Plots and Figures

- Loss curves and training overview:
  - `2_architecture_training/plots/training_overview_20250910_232322.png`
  - `2_architecture_training/plots/loss_breakdown_20250910_232323.png`
  - `2_architecture_training/plots/feature_analysis_20250910_232323.png`

### B. Example Data Structures

```json
{
  "sample_rate": 16000,
  "mel_frame_rate": 100.0,
  "n_frames": 3000,
  "mel_features": [[...]],
  "voice_activity": [...],
  "zero_crossing_rate": [...]
}
```

```json
{
  "frame_index": 0,
  "timestamp": 0.0,
  "blendshapes": { "jawOpen": 0.2, "mouthSmileLeft": 0.87 },
  "headPosition": { "x": 0.05, "y": -0.02, "z": 0.15 },
  "headRotation": { "w": 0.98, "x": 0.1, "y": 0.05, "z": 0.02 },
  "has_face": true
}
```

### C. Reproducibility Checklist (to finalize)

- Exact hyperparameters (channels, dropout, lr schedule) and seeds.
- Train/val/test split manifest and timestamps alignment scripts.
- Links to HF artifacts and ONNX models.

### D. Planned Large-Model Experiment Design

- 10 s sequences with 8 s overlap (2 s step) for full receptive field coverage.
- Multi-speaker training with speaker embeddings (optional).
- Attention blocks inserted between TCN stages; fusion via concatenation and MLP.

---

This research proposal consolidates the current proof-of-concept and outlines concrete next steps toward a robust, scalable audio-to-blendshape animation system.

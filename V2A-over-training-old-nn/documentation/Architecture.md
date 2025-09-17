### Architecture: Audio-to-Blendshapes (Voice-to-Action)

This document explains the full pipeline and the rationale behind the model and engineering choices. Code references point to the exact implementation used in this repo.

### 1) Data pipeline

- **Visual features (targets, 30 fps):**

  - We use MediaPipe Face Landmarker v2 to extract 52 blendshapes and a 7-dof head pose per frame (x, y, z, qw, qx, qy, qz) from video.
  - Implementation: `1_data_cleaning/1_extract_blendshapes.py`.
    - Extracts per-frame dictionaries and assembles `blendshapes_and_pose.json` with timestamps and session info.
    - Head pose is derived from the 4x4 facial transformation matrix and converted to translation + quaternion.

- **Audio features (inputs, 100 Hz):**

  - From the same video (or standalone audio), we extract a 16 kHz mono track and compute 80-dim log-mel spectrograms with 10 ms hop and 25 ms window.
  - Implementation: `1_data_cleaning/2_extract_audio_features.py`.
    - Yields mel frames at ~100 Hz, voice activity via RMS-thresholding, and auxiliary features.

- **Synchronization and sequence building (100 Hz alignment):**
  - We upsample the 30 fps visual stream to match the 100 Hz audio timeline via timestamp-based interpolation and then emit aligned sequences.
  - Implementation: `1_data_cleaning/3_create_datset.py`.
    - `synchronize_features(...)` interpolates 52 blendshapes + 7 pose to audio frame times.
    - `create_sequences(...)` windows sequences (default 240 ms length, 120 ms stride).
    - `normalize_features(...)` preserves natural ranges (mel dB clipping to [-80, 10]; blendshapes clipped to [0, 1]; pose lightly clipped to [-1, 1]).

### 2) Problem formulation

- Inputs: sequences of shape (T, 80) at 100 Hz.
- Outputs: sequences of shape (T, 59) at 100 Hz (52 blendshapes in [0,1] + 7 pose bounded), later downsampled to 30 fps for rendering.
- Causality: no future frames are used; deployment is streamable.

### 3) Model choice: TCN over RNN/LSTM

We initially considered LSTMs for sequential modeling. We chose a Temporal Convolutional Network (TCN) with dilated causal convolutions because:

- Parallelism and stable gradients vs. RNN recurrence; better throughput on GPUs.
- Flexible, controllable receptive field via dilation schedule to look back ~10 seconds, capturing prosodic context and coarticulation without peeking into the future.
- Robust temporal smoothing via residual connections and depthwise separable convs.

Implementation:

- `2_architecture_training/models/tcn_10s_model.py` defines a causal, dilated TCN with depthwise separable 1D convs, GELU activations, dropout, and residuals.
- Output head separates activation by semantic type: `sigmoid` for blendshapes, `tanh` (scaled) for pose.

Receptive field:

- With 80 mel features at 100 Hz and an exponentially increasing dilation schedule up to 128 across 10 layers, the receptive field spans ~10 s of past context.

### 4) Training pipeline

- Implementation: `2_architecture_training/train.py`.
- Model: `create_model()` from `tcn_10s_model.py` (10+ s receptive field). Alternative configs are provided for smaller memory.
- Loss: `AudioBlendshapeLoss` combines components:
  - Base: Smooth L1 (robust regression).
  - Temporal: first- and second-derivative consistency to match dynamics.
  - Silence weighting: during low RMS (VAD=0), prioritize accurate mouth closure.
  - Pose clamping: L2 + soft bounds to avoid drift.
- Optimization: AdamW + OneCycleLR, gradient clipping, batch training with train/val split.
- Metrics: MAE overall and for mouth indices, correlations on key tracks (jaw open, lip close, smile), pose MAE.

### 5) Inference and frame-rate conversion

- Implementation: `3_inference/inference.py`.
- Steps:
  1. Load audio/video; handle non-native formats via ffmpeg if needed.
  2. Compute log-mel features at 100 Hz with the same parameters as training; apply per-file normalization.
  3. Run the causal TCN at 100 Hz to produce (T, 59) predictions.
  4. Downsample to 30 fps for rendering by index selection aligned to timestamps.
- Output format mirrors `blendshapes_and_pose.json` for direct rendering.

### 6) Rationale and expected behavior

- Long-context causality: Prosody and coarticulation influence mouth and facial actions across seconds; the TCN captures this without future leakage.
- Output bounds: Blendshapes ∈ [0,1] improve stability and compatibility; pose is lightly bounded for realistic motion ranges.
- Temporal loss ensures smooth but responsive motions, avoiding jitter without lag.
- Silence handling reduces mouth flutter in pauses.

### 7) Related work (selected TCN references)

- Bai, Kolter, & Koltun (2018). “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.”
- Lea et al. (2017). “Temporal Convolutional Networks for Action Segmentation and Detection.”
- van den Oord et al. (2016). “WaveNet: A Generative Model for Raw Audio.”
- Lea et al. (2016). “Segmental Spatiotemporal CNNs for Fine-Grained Action Segmentation.”
- Bai et al. (2018) supplementary/implementations widely used for speech, audio tagging, time-series.

Note: Our task—regressing facial blendshapes from audio—shares the need for long causal context like speech models (prosody) and action segmentation (temporal structure). The TCN’s controllable receptive field and parallelism make it an apt choice over LSTMs in practice.

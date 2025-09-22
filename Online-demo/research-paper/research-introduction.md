## Research Introduction (Extended)

### Title

Audio-to-Blendshape Animation with Temporal Convolutional Networks

### Authors

- Sanjay Prasad HS
- Affiliation: Independent Researcher (Citizen Scientist)
- Contact: sanjuhs123@gmail.com

### 1. Background and Motivation

Animating speaking characters demands temporally accurate and expressive mapping from acoustic cues to facial muscle activations. Rule-based viseme systems—mapping phonemes to canonical mouth shapes—are widely used but often yield rigid motion, unstable coarticulation, and inaccurate timing. Data-driven neural methods promise smoother, context-aware animation. Temporal Convolutional Networks (TCNs) offer an attractive middle ground between RNNs and Transformers: they capture multi-scale temporal structure efficiently via dilated convolutions, train in parallel, and can be designed for causal, low-latency inference.

### 2. Problem Definition

We target frame-synchronous prediction of 52 blendshapes and 7 head pose values from speech audio. Inputs comprise 80-bin Mel spectrograms at 100 Hz complemented by auxiliary per-frame features (VAD, ZCR, RMS). Visual targets are extracted at 30 FPS and upsampled to 100 Hz by linear interpolation for perfect alignment. The system must:

- Preserve lip-sync timing, including rapid consonants and coarticulation.
- Produce smooth, natural trajectories without jitter.
- Generalize from limited training data and support real-time inference.

### 3. Approach Overview

Our proof-of-concept uses a compact 4-layer non-causal TCN with dilations [1,2,4,8] (≈0.15 s receptive field) and residual connections. The planned full system is a 10-layer causal TCN with dilations up to 128, achieving >10 s receptive field at 100 Hz. We consider depthwise separable convolutions for efficiency and hybridizing with attention blocks to capture longer-range emotional context.

### 4. Data and Synchronization

Audio is resampled to 16 kHz and windowed with 25 ms frames at 10 ms hops (100 Hz). Visual data (52 blendshapes + 7 pose) is tracked at 30 FPS with robust handling of dropped frames and quality filtering. Targets are linearly interpolated to 100 Hz to align with the audio features. Approximately 1.25 hours of synchronized data (≈40k overlapping 240 ms sequences) were prepared for the proof-of-concept.

### 5. Losses and Training Strategy

We adopt a multi-component loss: Smooth L1 for accuracy; first/second derivative penalties for temporal smoothness; VAD-weighted silence loss; and pose-range clamping. Training uses gradient clipping, residual connections, batch normalization, and OneCycleLR scheduling. Short overlapping sequences (240 ms, 120 ms step) improve stability and data efficiency.

### 6. Evaluation and Baselines

We will benchmark against viseme pipelines and neural baselines (linear, RNN/GRU, optional transformer), using MAE, per-region correlations, DTW timing distance, onset lag, and smoothness measures, plus human MOS/preference studies for perceptual quality.

### 7. Research Plan

We will scale training to ≈60 hours across multiple speakers and emotions, train the 10-layer causal TCN, and integrate multi-head attention between TCN blocks. We will conduct ablations on architecture depth, losses, feature sets, sequence length, and analyze dead neurons via activation sparsity and pruning. We will also examine sampling rate trade-offs (50–100 Hz) for latency vs fidelity and target mobile deployment via quantization and ONNX.

### 8. Contributions

- A synchronized audio-to-blendshape pipeline at 100 Hz with robust preprocessing.
- A TCN-based proof-of-concept and a roadmap to a real-time causal model with long memory.
- An evaluation protocol combining objective metrics and human judgments against viseme and neural baselines.
- A scaling and ablation plan to study model capacity, temporal context, and efficiency.

### 9. Limitations and Scope

The current proof-of-concept is trained on a small dataset with short temporal windows and non-causal padding. Results, while promising, are preliminary; the primary goal of this document is to establish a rigorous plan for comprehensive experiments and scaling.

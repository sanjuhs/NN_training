### Voice-to-Action (V2A): Objectives

This project converts human voice into facial action controls for a stylized character by predicting MediaPipe face blendshapes and head pose from audio. The original vision was “voice to realistic cartoon/animated character action.” The current scope focuses on Audio-to-Blendshapes, which remains faithful to the original intent: turning speech into continuous, frame-by-frame control signals that drive expressive facial animation.

### Why this approach

- **Expressiveness**: Speech prosody strongly correlates with facial dynamics (jaw opening, lip closures, smiles, eye blinks), enabling natural, speech-synchronized animation.
- **Real-time feasibility**: A causal Temporal Convolutional Network (TCN) with dilated convolutions models long context while remaining fast and streamable.
- **Compatibility**: Targets are the 52 MediaPipe blendshapes plus 7 head pose values (x, y, z, qw, qx, qy, qz). These are widely supported in DCC and game engines.

### Concrete goals

- **Inputs**: 16 kHz mono audio (or embedded video audio).
- **Features**: 80-dim log-mel spectrogram at 100 Hz (10 ms hop).
- **Outputs**: 59-dim sequences at 30 fps (52 blendshapes in [0,1], 7 pose values in a bounded range).
- **Latency**: Causal inference with a long receptive field (~10 s) for contextual realism but streamable.
- **Quality metrics**: Low MAE on key mouth AUs, high correlation on jaw/lip tracks, temporal smoothness without lag, stable pose.

### Scope of v1

- Data cleaning and feature extraction using MediaPipe (visual) and librosa (audio).
- Dataset alignment: upsample 30 fps visual targets to 100 Hz audio frames, then window into sequences.
- TCN architecture with depthwise separable dilated causal convs, residuals, and dual-headed activation (sigmoid for blendshapes, tanh-scaled for pose).
- Specialized training losses for base accuracy, temporal fidelity, silence behavior, and pose stability.
- Inference pipeline that consumes audio, runs the TCN, and downsamples to 30 fps for rendering.

### Non-goals (for now)

- Full-body gesture synthesis and gaze control.
- Phoneme alignment/viseme classification; we operate directly on acoustics.
- Multi-speaker identity disentanglement and explicit emotion control.

### Success criteria

- Mouth region MAE and correlations reach acceptable thresholds while preserving smoothness.
- Visual plausibility in the web demo (ONNX runtime) without noticeable jitter or drift.
- Robustness to variable loudness and short silences, preserving appropriate mouth closure.

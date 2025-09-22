## Research Abstract (Extended)

### Title

Audio-to-Blendshape Animation with Temporal Convolutional Networks

### Authors

- Sanjay Prasad HS
- Affiliation: Independent Researcher (Citizen Scientist)
- Contact: sanjuhs123@gmail.com

### Abstract

This work studies Temporal Convolutional Networks (TCNs) for predicting time-aligned facial blendshape and head pose trajectories directly from speech audio for controllable 3D character animation. We use 80-bin Mel spectrograms computed at 100 Hz (25 ms window, 10 ms hop, 16 kHz sampling) alongside auxiliary per-frame features (voice activity, zero-crossing rate, RMS) and align them with visual targets (52 blendshapes + 7 head pose values) extracted at 30 FPS and upsampled via linear interpolation to 100 Hz. Our proof-of-concept model is a compact 4-layer non-causal TCN that serves to validate the approach on ~1.25 hours of synchronized data. We design a multi-component loss combining Smooth L1 for accuracy, derivative-based temporal smoothness, VAD-weighted silence stability, and pose clamping for realism. We propose a comprehensive evaluation protocol including MAE, per-region correlations, DTW timing distance, onset lag, smoothness metrics, and human preference/MOS studies against viseme pipelines and neural baselines. We outline a research plan to scale to a 10-layer causal TCN with 10+ seconds of receptive field and hybrid attention blocks for emotional context modeling, combined with a larger multi-speaker dataset (~60 hours). Our objective is to demonstrate that TCNs can deliver more natural, temporally coherent animation than rule-based viseme approaches while remaining efficient enough for real-time deployment.

### Keywords

Temporal Convolutional Networks, Audio-Driven Animation, Blendshapes, Lip-Sync, Dilated Convolutions, Attention, Real-Time Graphics, Human Evaluation

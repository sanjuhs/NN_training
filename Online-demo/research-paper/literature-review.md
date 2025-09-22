## Literature Review (Expanded)

### Title

Audio-to-Blendshape Animation with Temporal Convolutional Networks

### Authors

- Sanjay Prasad HS
- Affiliation: Independent Researcher (Citizen Scientist)
- Contact: sanjuhs123@gmail.com

### 1. Classical Pipelines: Phonemes and Visemes

Rule-based lip-sync systems map phonemes to a discrete set of visemes and time these to aligned phoneme sequences, often from forced alignment. Strengths include interpretability and pipeline maturity. Limitations include:

- Poor coarticulation modeling (limited context windows).
- Discrete categories that fail to capture subtle articulatory variation.
- Latency and robustness issues in noisy alignment.

Representative systems: production toolchains (e.g., phoneme-to-viseme maps) and academic works on viseme sets and alignment strategies. [Citations TBD]

### 2. Parametric and Data-Driven Facial Animation

Approaches regress continuous facial parameters (blendshapes or landmarks) from audio, sometimes with auxiliary textual cues. Early methods use linear regression or HMMs; later methods adopt neural networks for nonlinear mapping and better context modeling. Advantages include continuous control and smoother trajectories; challenges include synchronization and temporal consistency. [Citations TBD]

### 3. Sequence Models: RNNs, TCNs, Transformers

- RNN/GRU/LSTM capture temporal dependencies sequentially but can suffer from vanishing gradients and limited parallelism.
- TCNs provide large receptive fields via dilated convolutions with residual connections, supporting efficient, parallel training and causal inference; strong results have been reported on general sequence modeling. [Bai et al., 2018]
- Transformers model long-range dependencies with attention but at O(n²) cost; they can be powerful but heavy for real-time applications. Hybrid TCN+attention architectures can combine local efficiency with global context.

### 4. Dilated Convolutions and WaveNet Influence

WaveNet demonstrated the effectiveness of dilated causal convolutions for audio generation, motivating TCN adoption in regression tasks where long receptive fields and low latency are beneficial. We reuse dilation schedules and residual blocks while adapting for regression outputs with bounded activations. [van den Oord et al., 2016]

### 5. Talking-Head Synthesis and Lip-Sync Metrics

Image-based talking-head methods synthesize pixels or landmarks conditioned on audio; while visually compelling, they often require large datasets and careful identity preservation. For evaluation, lip-sync confidence/distance metrics (e.g., LSE-C/LSE-D) have been proposed. For blendshape regression, MAE and correlation per key articulators (jaw, lips) are common, with DTW for timing. [Citations: SyncNet-like, Voca/Faceformer/NeRF-based works TBD]

### 6. Evaluation Practices in Animation

Beyond objective errors, human studies (MOS, pairwise preferences) are standard for perceptual quality. Protocols emphasize held-out speakers, identical rendering pipelines, and statistical significance testing. Smoothness must be balanced against responsiveness to avoid over-smoothing artifacts.

### 7. Gaps and Opportunities

- Many viseme systems lack continuous, data-driven coarticulation.
- RNN solutions can be latency-heavy; transformer solutions are compute-demanding.
- TCNs offer a pragmatic balance for real-time blendshape regression, especially when combined with attention for emotional context.
- Standardized benchmarks for blendshape animation remain limited; this work proposes a mixed metric + human protocol to compare against viseme and neural baselines.

### References (Placeholders)

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.
- van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio.
- [SyncNet and lip-sync metric papers].
- [Parametric facial animation and blendshape regression papers].
- [Talking-head synthesis pipelines and evaluation protocols].

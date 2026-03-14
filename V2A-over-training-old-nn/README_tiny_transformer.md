# Tiny Transformer README

## What Exists Now

There are now two transformer-ready datasets in this repo:

1. Small speaker-specific overfit set
   - path: `V2A-over-training-old-nn/data/extracted_features`
   - shape: `audio (12, 1000, 80)`
   - shape: `targets (12, 1000, 59)`
   - use this when you want the fastest possible memorization demo

2. Full long-context rebuilt set
   - path: `V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500`
   - shape: `audio (798, 1000, 80)`
   - shape: `targets (798, 1000, 59)`
   - use this when you want real `10-second` context across the full speaker archive

## Where The Full Long-Context Set Came From

The original larger training artifact in this repo is:

- `V2A-over-training-old-nn/2_architecture_training/data/train`

That artifact is stored as:

- `audio (40743, 23, 80)`
- `targets (40743, 23, 59)`
- `vad (40743, 23)`

Those are overlapping `23-frame` windows with an `11-frame` step.

The rebuild script stitches matching overlaps back into continuous segments, then cuts true `10-second` windows:

- source continuous segments found: `163`
- segments long enough for 10-second windows: `83`
- rebuilt output windows: `798`
- rebuilt window length: `1000 frames`
- rebuilt window step: `500 frames`
- feature rate used: `100 Hz`

## Rebuild The Full 10-Second Dataset

Use:

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/rebuild_long_context_dataset.py \
  --input-dir V2A-over-training-old-nn/2_architecture_training/data/train \
  --output-dir V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500 \
  --window-frames 1000 \
  --window-step-frames 500 \
  --min-segment-frames 1000 \
  --sample-rate-hz 100 \
  --cover-tail
```

Output files:

- `audio_sequences.npy`
- `target_sequences.npy`
- `vad_sequences.npy`
- `segment_ids.npy`
- `window_start_frames.npy`
- `dataset_metadata.json`

## Train On The Small 10-Second Overfit Set

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py \
  --data-dir V2A-over-training-old-nn/data/extracted_features \
  --epochs 400 \
  --batch-size 2 \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --num-layers 3 \
  --ffn-dim 256 \
  --temporal-weight 0.0 \
  --checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --last-checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_last.pth \
  --history-path V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_l1_history.json
```

## Train On The Full Rebuilt 10-Second Dataset

This is the proper long-context all-data command:

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py \
  --data-dir V2A-over-training-old-nn/2_architecture_training/data/train_long_10s_step500 \
  --epochs 160 \
  --batch-size 2 \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --num-layers 3 \
  --ffn-dim 256 \
  --dropout 0.0 \
  --temporal-weight 0.0 \
  --val-sequences 0 \
  --val-fraction 0.15 \
  --segment-aware-split \
  --checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --last-checkpoint-path V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_last.pth \
  --history-path V2A-over-training-old-nn/2_architecture_training/plots/tiny_transformer_full10s_l1_history.json
```

Why this split is safe:

- it uses `segment_ids.npy`
- train and val are separated by reconstructed continuous segments
- no overlapping windows cross the split boundary

## Export To ONNX

```bash
./.venv/bin/python V2A-over-training-old-nn/2_architecture_training/export_tiny_transformer_onnx.py \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --output V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.onnx \
  --manifest V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.json \
  --seq-len 1000
```

## Local Inference Script

Inference script:

- `V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py`

Audio input example:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --audio V2A-over-training-old-nn/2_architecture_training/sample_audio/sample.wav \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/tiny_transformer_demo_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/tiny_transformer_demo_100hz.npy \
  --output-json V2A-over-training-old-nn/data/inference/tiny_transformer_demo_30fps.json \
  --max-duration 10 \
  --fixed-seconds 10 \
  --device cpu
```

Feature input example:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --features /path/to/feature_sequence.npy \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/tiny_transformer_features_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/tiny_transformer_features_100hz.npy \
  --fixed-seconds 10 \
  --device cpu
```

Fixed `10-second` output contract:

- model-rate output: `(1000, 59)`
- `30 FPS` export: `(300, 59)`

## Arbitrary-Length Audio

Yes, arbitrary-length audio is possible.

The model is trained on `10-second` windows, but inference can now run in chunks:

```bash
./.venv/bin/python V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py \
  --audio /path/to/long_audio.wav \
  --checkpoint V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1_best.pth \
  --output-npy V2A-over-training-old-nn/data/inference/long_audio_30fps.npy \
  --output-100hz-npy V2A-over-training-old-nn/data/inference/long_audio_100hz.npy \
  --chunk-seconds 10 \
  --chunk-overlap-seconds 5 \
  --device cpu
```

What this does:

- split long audio features into overlapping `10-second` chunks
- run the transformer on each chunk
- blend overlapping predictions back into one continuous `100 Hz` sequence
- downsample the full sequence to `30 FPS`

If the input audio is `T` seconds long, the output will be approximately:

- `round(T * 30)` frames
- each frame has `59` values

So yes, the system can return a full-length `30 FPS` blendshape track for both short and long audio.

## Browser Path

Yes, the transformer works through `ONNX` with `onnxruntime-web` and `WASM`.

Do not open the HTML directly with `file://`.
Run a local server instead.

Recommended local demo command from the repo root:

```bash
python -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/Online-demo/transformer-model.html
```

Comparison page:

```text
http://127.0.0.1:8000/Online-demo/comparison.html
```

Current browser files:

- transformer-only page: `Online-demo/transformer-model.html`
- comparison page: `Online-demo/comparison.html`
- ONNX asset: `assets/tiny_transformer_full10s_l1.onnx`
- ONNX manifest: `assets/tiny_transformer_full10s_l1.json`
- both pages now chunk long audio into overlapping `10-second` ONNX windows

The ONNX export keeps the expected browser contract:

- input name: `audio_features`
- output name: `blendshapes`
- output shape: `(batch, sequence_length, 59)`

## Which Model To Use

Recommended default now:

- `tiny_transformer_full10s_l1_best.pth`
- `tiny_transformer_full10s_l1.onnx`
- `assets/tiny_transformer_full10s_l1.onnx`

The transformer HTML pages now point to the full-data ONNX by default.

## How To Scale This Properly

If you want this to become a much better model instead of just a working demo, the next gains will come more from data quality and data volume than from making the network slightly deeper.

Recommended order:

1. Increase clean aligned data
   - more speakers
   - more phonetic coverage
   - more emotion coverage
   - more head motion and speaking style variation

2. Improve target quality
   - fewer face-tracking failures
   - consistent camera framing
   - fewer dropped frames
   - cleaner pose stabilization

3. Train a better base model
   - pretrain a multi-speaker transformer on a larger corpus
   - then fine-tune per speaker for the final avatar

4. Improve losses and weighting
   - keep `L1` as the main anchor
   - add velocity / temporal loss
   - upweight mouth and jaw channels
   - separate pose loss from blendshape loss

5. Improve evaluation
   - hold out entire speakers or sessions
   - compare lip-sync quality, temporal smoothness, and expression stability

Practical model upgrades after the current tiny transformer:

- increase `d_model` from `128` to `192` or `256`
- increase depth from `3` to `4-6` layers
- keep `4-8` attention heads
- keep chunked long-context inference
- add speaker embeddings if you move to multi-speaker training
- later, swap raw log-mels for a stronger audio front-end such as HuBERT or wav2vec features

## Best Data Sources To Add

For this project, there are two broad buckets:

1. Best for final production quality
   - your own captured data for the target speaker
   - this is the most important data if the end goal is a speaker-specific avatar

2. Best for pretraining / broad coverage
   - public audio-visual or 3D speech-face datasets

Useful public sources:

- MEAD
  - large emotional talking-face dataset with `60` actors, `8` emotions, and `3` intensity levels
  - useful for expression diversity and emotional coverage
  - source: [MEAD GitHub](https://github.com/uniBruce/Mead)

- VOCA / VOCASET
  - 4D face dataset with about `29 minutes` of scans at `60 fps` from `12` speakers
  - useful for learning strong speech-to-face priors from 3D data
  - source: [VOCA official page](https://voca.is.tue.mpg.de/index.html)
  - source: [MPI-IS VOCA page](https://is.mpg.de/en/code/capture-learning-and-synthesis-of-3d-speaking-styles)

- BIWI 3D Audiovisual Corpus of Affective Communication
  - `1109` sentences from `14` speakers with synchronized audio and 3D face scans
  - useful for speech-driven 3D facial motion and affective speech
  - source: [ETH BIWI homepage reference via NIST catalog](https://tsapps.nist.gov/BDbC/Search/Details/492)
  - source: [BIWI license / access PDF](https://data.vision.ee.ethz.ch/cvl/datasets/B3DAC2/CorpusEULA.pdf)

- CREMA-D
  - `7442` clips from `91` actors with multimodal emotion labels
  - useful if you want more actor diversity and emotion labels
  - source: [CREMA-D official page](https://cheyneycomputerscience.github.io/CREMA-D/)

- RAVDESS
  - `7356` speech/song emotion recordings from `24` actors
  - useful for emotional facial movement diversity
  - source: [RAVDESS official dataset page](https://affectivedatascience.com/datasets)

- RAVDESS Facial Landmark Tracking
  - already tracked facial landmarks and pose for all RAVDESS trials
  - useful for bootstrapping without running your own first-pass tracker over all files
  - source: [RAVDESS tracking dataset page](https://affectivedatascience.com/datasets)

What I would actually do:

- first, collect more of your own speaker data
- second, add MEAD for emotion diversity
- third, add VOCASET / BIWI if you want stronger 3D motion priors
- fourth, use CREMA-D and RAVDESS as extra emotion/style coverage only after your core pipeline is stable

Important:

- many public datasets do not give you final `52 blendshapes + 7 pose` directly
- they give video, landmarks, or 3D geometry
- you still need a conversion pipeline into your repo’s target format

## Recommended Processing Pipeline

For every new dataset, the pipeline should be:

1. Ingest raw data
   - keep original video/audio untouched
   - save dataset-level metadata and licenses

2. Normalize media
   - standardize video fps
   - standardize audio sample rate
   - verify timestamps and dropped frames

3. Face tracking / target extraction
   - run MediaPipe Face Landmarker or your preferred tracker
   - export blendshape coefficients
   - export head pose / transformation matrices
   - mark frames with missing faces or low confidence

4. Audio feature extraction
   - resample to `16 kHz`
   - compute `80-bin` log-mel features
   - use `100 Hz` frame rate to match the current repo contract

5. Alignment
   - align audio features and facial targets frame-by-frame
   - trim mismatched starts / ends
   - save VAD or confidence masks

6. Quality filtering
   - drop clips with tracking failures
   - drop clips with desync
   - drop clips with extreme occlusion or profile views if your tracker is unstable there

7. Windowing
   - build `10-second` windows (`1000` frames at `100 Hz`)
   - start with `5-second` overlap

8. Split safely
   - split by speaker or session
   - never let overlapping windows leak across train / val / test

## Where Data Management Should Happen

Do not keep large raw datasets inside git.

Recommended split:

- Git repo
  - code
  - manifests
  - metadata
  - split definitions
  - small sample assets

- External storage
  - raw videos
  - raw audio
  - extracted tracking outputs
  - long window datasets
  - checkpoints

Recommended layout:

```text
/workspace/v2a_data/
  raw/
    mead/
    vocaset/
    biwi/
    crema_d/
    ravdess/
    custom_capture/
  processed/
    media_normalized/
    mediapipe_tracks/
    audio_features_100hz/
    aligned_sequences/
  windows/
    train_long_10s_step500/
    eval_sets/
  metadata/
    manifests/
    licenses/
    splits/
    quality_reports/
```

Inside the repo, keep only lightweight control files, for example:

```text
V2A-over-training-old-nn/
  documentation/
  1_data_cleaning/
  2_architecture_training/
  3_inference/
  data_management/
    manifests/
    split_defs/
    source_registry/
```

That way:

- heavy data lives outside git
- the repo still defines how data is built
- you can rebuild datasets deterministically

## Can Runpod Handle Data Processing Too?

Yes.

Runpod is fine for:

- downloading datasets
- running ffmpeg preprocessing
- running MediaPipe tracking
- extracting audio features
- building windowed training sets
- training
- exporting checkpoints and ONNX

The cleanest setup is:

- attach a Runpod network volume
- mount it at `/workspace`
- keep raw and processed data on that volume
- use the same volume across multiple Pods over time

According to Runpod’s current docs:

- network volumes persist independently of Pods
- they are typically mounted at `/workspace`
- they are meant for shared, persistent datasets and model files
- Runpod also exposes an S3-compatible API for network volumes in selected datacenters

Sources:

- [Runpod network volumes](https://docs.runpod.io/storage/network-volumes)
- [Runpod S3-compatible API](https://docs.runpod.io/storage/s3-api)
- [Runpod SSH docs](https://docs.runpod.io/pods/configuration/use-ssh)

My recommendation:

- local machine for quick code edits and light inspection
- Runpod for heavy preprocessing and all training
- network volume for raw + processed data
- sync only manifests, scripts, and small reports back into git

## Can MediaPipe Run On GPU?

Yes, but with an important caveat.

According to the current official MediaPipe Python docs:

- `mp.tasks.BaseOptions` supports `delegate`
- supported values are `CPU` and `GPU`
- Python GPU support is currently limited to `Ubuntu` platforms

That means a Runpod Ubuntu pod is the right place to try GPU-accelerated MediaPipe in Python.

Sources:

- [MediaPipe BaseOptions Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions)
- [MediaPipe Face Landmarker Python guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)

Important practical note:

- even with GPU inference, full preprocessing is not purely GPU-bound
- video decode, image resizing, file I/O, and writeback can still dominate runtime
- so the end-to-end pipeline may not scale linearly with GPU size

## Rough Time Expectations

This part is a working engineering estimate from the current workload, not a benchmark from the sources above.

For about `1.15 hours` of single-face source video, a first full pipeline pass on one Runpod pod is likely on the order of:

- media ingest + copy: `10-60 minutes` depending on upload speed
- decode / normalization: `30-120 minutes`
- MediaPipe tracking: `1-4 hours`
- audio feature extraction + alignment + window building: `15-60 minutes`
- training a stronger first full-data transformer run: `1-4 hours`

So a realistic first serious end-to-end pass is:

- roughly `3-9 hours`, depending mostly on:
  - video resolution
  - number of clips
  - face tracking stability
  - whether MediaPipe GPU actually helps your exact pipeline
  - storage / upload speed

Once the pipeline is cached and the raw data is already on a network volume, later runs should be much faster.

## Recommended Next Step

If you want to scale this project in a disciplined way, do this next:

1. define the permanent data layout on a Runpod network volume
2. ingest one additional public dataset, ideally `MEAD`
3. run a single standard preprocessing pipeline that outputs the repo format:
   - `audio_sequences.npy`
   - `target_sequences.npy`
   - `vad_sequences.npy`
   - `dataset_metadata.json`
4. pretrain a multi-speaker transformer
5. fine-tune on your target speaker

That is the cleanest path from the current demo to a genuinely stronger production model.

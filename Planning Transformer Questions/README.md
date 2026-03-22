# Planning Transformer Questions

This note answers the current planning questions for this repo in very simple terms.

## 1. Very Short Answers First

### Q1. Will MediaPipe work on the GPU?

Short answer:

- Yes, but only on the right setup.
- For Python Tasks, MediaPipe's official docs say GPU support is limited to Ubuntu platforms.
- For this repo as it is written now, the extraction script is effectively CPU-first because it does not enable the GPU delegate.

What that means for you:

- On your Mac: assume `no`.
- On an Ubuntu Runpod machine: assume `yes, if you explicitly enable the GPU delegate and it actually initializes correctly`.
- In the current repo code: assume `no until you change the extractor`.

Important repo detail:

- The current extractor is [`V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/1_data_cleaning/1_extract_blendshapes.py).
- It creates `BaseOptions` without a GPU delegate.
- It also uses `detect()` frame by frame, not `detect_for_video()`.

### Q2. If MediaPipe does work on GPU, how fast is it?

Short answer:

- There is no single guaranteed FPS number.
- In practice, for one face, a good Ubuntu GPU setup should be at least real-time and often a few-times real-time.
- Your full pipeline speed is not only MediaPipe. Video decoding, frame conversion, JSON writing, and bad frames also matter.

Safe planning estimate for offline extraction:

- `1 hour of video` can easily take anywhere from about `30 minutes to 2 hours` end to end, depending on resolution, cropping, and whether the GPU delegate is really active.
- If the pipeline falls back to CPU, it can be slower.

Best practical answer:

- Do not plan around theoretical FPS.
- Run one `10 minute` benchmark clip first.
- Measure total wall-clock time.
- Then scale from that number.

### Q3. Can we get perfect consistency?

Short answer:

- No, not perfect.
- Yes, good enough for training if you keep the pipeline fixed.

What gives you practical consistency:

- same MediaPipe version
- same model file
- same FPS
- same face crop policy
- same video resolution
- same script
- extract once
- save the outputs
- never re-extract unless the pipeline changes

That is the real answer:

- `perfect tracking` is unrealistic
- `perfect reproducibility of your processed dataset` is realistic

## 2. What This Repo Is Doing Right Now

The current transformer path in this repo uses:

- audio at `16 kHz`
- `80` log-mel features
- `10 ms` hop size
- audio feature rate of `100 Hz`
- output targets of `52 blendshapes + 7 pose values = 59 values`

Current transformer I/O:

- input shape: `(batch, sequence_length, 80)`
- output shape: `(batch, sequence_length, 59)`

For the main long-context transformer:

- sequence length is `1000`
- that means `10 seconds`
- so one input window is `(1, 1000, 80)`
- one output window is `(1, 1000, 59)`

Repo references:

- model: [`V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_model.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_model.py)
- trainer: [`V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/2_architecture_training/train_tiny_transformer.py)
- inference: [`V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/3_inference/tiny_transformer_inference.py)
- data sync: [`V2A-over-training-old-nn/1_data_cleaning/3_create_datset.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/1_data_cleaning/3_create_datset.py)

## 3. How The Transformer Works In Simple Terms

This model does **not** use a text tokenizer.

It takes audio numbers directly.

### Step-by-step

1. Start with raw audio.
2. Convert audio into log-mel features.
3. Every `10 ms`, you get one frame.
4. Each frame has `80` numbers.
5. Over `10 seconds`, you get `1000` frames.
6. So the transformer sees a matrix of `1000 x 80`.
7. The model projects each `80`-number frame into a hidden size of `128`.
8. Positional encoding is added so the model knows where each frame sits in time.
9. The transformer encoder looks across the full `10 second` window.
10. For each time step, it outputs `59` numbers.

### What the output means

For each frame:

- first `52` numbers = blendshapes
- last `7` numbers = head pose

In simple words:

- input = "what the audio looks like over time"
- output = "what the face should do over time"

### Very simple picture

```text
10-second audio
    ->
log-mel extraction
    ->
1000 frames x 80 numbers
    ->
linear projection to 128 dims
    ->
3 transformer encoder layers
    ->
1000 frames x 59 outputs
    ->
52 face controls + 7 pose controls per frame
```

### Does it look at the past 10 seconds?

Yes.

For the current `10 second` setup:

- the model sees the full `10 second` window at once
- so each output frame can use information from nearby and far-away audio inside that window

This is useful because:

- mouth shape depends on phoneme context
- expression depends on prosody
- head motion is often tied to phrase-level rhythm, not one frame only

## 4. If Some Face Parts Matter More, What Should We Do?

Right now, the model is a direct regression model.

That means:

- it does **not** classify mouth shapes first
- it directly predicts all `59` continuous values

If you care more about mouth, eyes, brows, and head pose, the right move is:

- keep the direct regression setup
- give more loss weight to the important channels

Simple weighting idea:

- mouth and jaw: highest weight
- head pose: medium-high weight
- eyebrows and eye blinks: medium weight
- cheeks and nose: lower weight

Extra improvements that help:

- stronger silence stability loss for mouth channels
- separate metrics for mouth only, eyes only, pose only
- clip filtering when face tracking is weak
- smoothing pose targets a little before training

## 5. How Much Of Your Own Data Should You Record?

If your goal is:

- not a general model for everybody
- but a strong demo with **your own face and your own voice**

then you do **not** need massive data first.

### Good simple targets

| Data amount | What it is good for |
| --- | --- |
| `1 hour` | weak proof of concept |
| `3 to 4 hours` | good single-person demo if the data is clean |
| `8 to 10 hours` | much better single-person model |
| `20+ hours` | starts becoming a serious scaling effort |

My practical answer for your current situation:

- `3 to 4 hours` is enough for a boss demo
- only if the recordings are clean, varied, and consistent

## 6. Do You Need More Words?

Short answer:

- Yes, but not just more words.
- You need better **coverage**.

What matters:

- different mouth shapes
- fast speech
- slow speech
- pauses
- questions
- emphasis
- soft speech
- emotional speech
- neutral speech

So do not only read random text.

Record blocks like this:

1. neutral reading
2. conversational reading
3. questions
4. excited delivery
5. calm delivery
6. short pauses and long pauses
7. numbers, names, dates
8. tongue-twister style lines
9. vowels and exaggerated articulation
10. natural monologue

## 7. How Many Expressions And How Much Head Movement?

You do not need to act out `100` expressions.

You need a small set of useful modes:

- neutral
- slight smile
- serious
- excited
- confused or questioning
- emphasis
- thoughtful pause
- silence with small natural motion

For head movement:

- include some left-right yaw
- include some up-down pitch
- include a little roll
- keep most clips mostly frontal
- do not spend too much time on extreme turns

Why:

- if you move too much, tracking quality drops
- if you never move, the pose channels stay weak

Best balance:

- `70%` mostly frontal and stable
- `20%` mild natural motion
- `10%` stronger motion for coverage

## 8. With Only 3 To 4 Hours, How Do You Make The Model Better?

This is the most important practical question.

With limited data, the biggest wins are:

1. cleaner recordings
2. tighter alignment
3. better coverage of speech styles
4. removing bad clips
5. making the model focus on the channels that matter most

With only `3 to 4 hours`, the best strategy is:

- build a very good single-speaker dataset
- overfit that cleanly
- show a strong demo
- do not try to become multi-speaker in the next few days

What helps most:

- fixed camera
- fixed lighting
- fixed framing
- clear microphone
- one face only
- little background noise
- no hard cuts
- no sunglasses
- no hand-over-face moments
- no heavy motion blur

## 9. Should You Use Runpod For Data Processing?

Short answer:

- Yes, it can help.
- But use Runpod as **temporary compute**, not as your main permanent storage.

Best use of Runpod:

- batch extraction
- syncing audio and face features
- dataset packing
- training
- quick experiments

What not to do:

- treat Runpod as your long-term source of truth

Better pattern:

1. upload raw clips to Runpod
2. process them there in one batch
3. save cleaned outputs
4. upload processed dataset to Hugging Face
5. also keep a local backup
6. stop the pod

## 10. Where Should Data Management Live?

Best answer:

- local storage or external disk for raw source files
- Hugging Face for processed reusable datasets
- Runpod only for temporary processing and training

Simple folder structure idea:

```text
data/
  raw_videos/
  raw_audio/
  extracted_features/
  aligned_sequences/
  training_dataset/
  manifests/
  quality_reports/
```

Also keep:

- `dataset_version.txt`
- one manifest CSV or Parquet file
- split files for train and val
- notes on which clips were rejected and why

## 11. Can You Upload This To Hugging Face?

Yes.

That is a good idea.

In fact, this repo already has a starting upload script:

- [`V2A-over-training-old-nn/1_data_cleaning/4_upload_dataset_hf.py`](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/V2A-over-training-old-nn/1_data_cleaning/4_upload_dataset_hf.py)

Best practical approach:

- upload processed features, not only raw video
- include metadata
- include a dataset card
- version the dataset clearly, for example `v1`, `v2`, `v3`

Best file formats:

- `Parquet` for metadata tables
- `npy` or `npz` for tensors if you want fast reuse inside Python
- `wav` and `mp4` only if you truly need raw media on the Hub

If storage is limited:

- keep raw media locally
- upload processed reusable features and manifests to Hugging Face

## 12. Where Should Other Data Come From Later?

For the next `3 to 4 days`, my answer is:

- do **not** spend your main effort collecting many outside datasets
- focus on your own high-quality speaker-specific data first

After the demo, useful outside sources include:

- `LRS3` for large talking-head speech data
- `VoxCeleb` for many speakers and in-the-wild variation
- `HDTF` for high-resolution talking-head clips
- `RAVDESS` for emotional audio-visual speech
- `MEAD` for controlled emotional talking-face data

Simple use for each:

- `LRS3`: speech coverage
- `VoxCeleb`: speaker diversity
- `HDTF`: cleaner talking-head visuals
- `RAVDESS`: emotion labels
- `MEAD`: emotion plus controlled capture

Important note:

- some of these have access restrictions or non-commercial limits
- check the license before building a product path around them

## 13. Is Runpod Worth It If Time Is Limited?

Yes, if you use it correctly.

The good use case is:

- process many clips in one burst
- train several experiments in one burst
- upload outputs away from the pod
- shut the pod down

So your plan is valid.

## 14. Best Cost-Effective Plan For The Next 3 To 4 Days

If the goal is:

- make the demo impressive
- get a green light

then this is the best practical plan.

### Day 1

- record `3 to 4 hours` of your own clean audio-video data
- keep camera, mic, framing, and lighting consistent
- split clips by speaking style
- throw away obviously bad takes

### Day 2

- run extraction
- inspect failure cases
- remove bad clips
- create the aligned dataset
- save one clean processed dataset version
- upload the processed dataset to Hugging Face

### Day 3

- train the current tiny transformer on the cleaned speaker-specific set
- run a few focused experiments, not too many
- pick the best checkpoint by both metric and visual quality

### Day 4

- export ONNX
- run the browser demo
- compare a few clips
- prepare the best examples for the boss

## 15. What Is The Fastest Way To Make The Demo Better Without A Giant Rewrite?

If I had to prioritize only a few things, I would choose:

1. better data
2. cleaner extraction
3. weighted attention to mouth and pose channels
4. stronger silence behavior
5. clean demo examples

The biggest mistake would be:

- spending all your time making the transformer bigger before fixing the data

## 16. Practical Recommendation For This Repo

For this exact repo, the highest-value next steps are:

1. keep the current tiny transformer baseline
2. build a clean single-speaker dataset with your own recordings
3. process in one batch on Ubuntu or Runpod
4. save processed outputs permanently
5. upload reusable data to Hugging Face
6. demo the best checkpoint, not every checkpoint

## 17. Final Bottom Line

If you want the shortest honest answer:

- `Yes`, MediaPipe can use GPU, but for Python you should treat that as an Ubuntu-only path and your current repo does not enable it yet.
- `3 to 4 hours` of your own clean data is enough for a strong single-person proof of concept.
- `Runpod is useful` for one-shot processing and training.
- `Hugging Face is the right place` to keep processed reusable datasets.
- For the next few days, `better data beats a bigger model`.

## 18. Phone Vs Webcam

Short answer:

- Yes, you can use your phone.
- In many cases, a phone is better than a webcam.

Why a phone can be better:

- better camera sensor
- better sharpness
- better low-light quality
- more stable exposure

What matters most for this project is not "4K" or "perfect cinema quality."

What matters most is:

- face is clear
- lighting is steady
- framing is steady
- face is not too small
- mouth and eyes are easy to see

For MediaPipe extraction, the exact original resolution matters less than:

- face visibility
- blur
- lighting
- occlusion
- framing consistency

So yes:

- a phone recording is completely fine
- a good phone recording is usually better than a bad webcam recording

## 19. Will Different Resolution Hurt The Blendshape Extraction?

Short answer:

- Not much, if the face is large and clear in frame.

What actually hurts extraction:

- blurry video
- dark lighting
- face too far from camera
- strong motion blur
- hand covering the face
- glasses glare
- extreme head turns

What does **not** matter much:

- whether it started as webcam or phone
- whether it is portrait or landscape
- whether it is somewhat higher or lower resolution

As long as:

- the face occupies a good chunk of the image
- the video is not noisy or blurry

then MediaPipe should still work well.

## 20. Landscape Or Vertical?

Short answer:

- Use `landscape`.

Why landscape is better here:

- easier to keep shoulders and head in frame
- easier to keep natural side-to-side motion without cutting off the face
- more standard for later processing and review
- easier to reuse for demos and future datasets

Vertical can still work, but it is less ideal for this pipeline.

So my recommendation is:

- record in `landscape`
- keep the head centered
- keep some space above the head
- keep shoulders visible if possible

## 21. What Resolution And FPS Should You Use?

Best practical setting:

- `1080p`
- `30 FPS`

Why:

- enough detail for tracking
- manageable file sizes
- consistent with the current extraction flow

You do **not** need `4K`.

In fact, `4K` is usually a bad tradeoff here because:

- files become much bigger
- transfer gets slower
- processing gets slower
- tracking quality usually does not improve enough to justify it

So the safe recommendation is:

- `1920 x 1080`
- `30 FPS`
- landscape

## 22. Is Landscape Heavier Than Portrait?

Short answer:

- Not inherently.

Important point:

- `1920 x 1080` and `1080 x 1920` have the same number of pixels.

So if one file is bigger, it is usually because of:

- bitrate variation
- compression differences
- lighting and texture detail
- how much motion is in the clip

Based on your test:

- portrait: `17.4 MB` for `11 s`
- landscape: `25.5 MB` for `11 s`

That does **not** prove landscape is always heavier.

It only proves:

- that specific landscape clip was encoded at a higher effective bitrate

## 23. Storage Estimate For 3 Hours

Using your two sample clips:

### Portrait-style sample

- `17.4 MB / 11 s`
- about `5.7 GB per hour`
- about `17.1 GB for 3 hours`

### Landscape-style sample

- `25.5 MB / 11 s`
- about `8.3 GB per hour`
- about `25.0 GB for 3 hours`

### Simple average estimate

- about `7.0 GB per hour`
- about `21 to 22 GB for 3 hours`

So yes:

- your `22 GB for 3 hours` estimate is reasonable

For safety, I would budget:

- `25 to 30 GB` for `3 hours`

That gives you some breathing room.

## 24. Can You Record It On Your Nothing Phone?

Yes.

If you have enough free space, recording on your phone is completely fine.

Your phone is a good temporary capture device.

Best practical workflow:

1. record on phone
2. move files off the phone after each session
3. keep one copy on your PC or external drive
4. process from PC or Runpod

I would **not** treat the phone as the permanent storage location.

Phones are good for capture.

They are not the best place for:

- long-term dataset storage
- versioning
- processing
- backup

## 25. Can You Keep The Videos On The Phone?

Yes, temporarily.

But I would not recommend keeping the full dataset only on the phone.

Why:

- easy to run out of space
- harder to organize versions
- harder to batch process
- risky if files are lost or deleted

Better rule:

- phone = capture
- PC/external drive = raw archive
- Hugging Face = processed reusable dataset

## 26. How Should You Transfer The Files?

Best simple options:

### Option A: Transfer to your PC

This is the best default.

Use:

- USB cable
- local Wi-Fi transfer
- Google Drive or similar only if needed

This is the easiest workflow for processing.

### Option B: Upload from phone to cloud storage first

This is okay if your PC storage is tight.

Use:

- Google Drive
- Google Cloud Storage
- Dropbox
- OneDrive

Then pull from there into PC or Runpod.

### Option C: Keep only on phone

Possible, but not recommended.

## 27. How Should You Upload To Hugging Face?

Best answer:

- do **not** upload the raw phone videos first unless you really need them there
- upload the processed dataset after extraction and alignment

Best pipeline:

1. record raw video on phone
2. transfer raw video to PC
3. run extraction
4. create aligned features and metadata
5. upload processed dataset to Hugging Face

This is better because:

- smaller than raw video in many cases
- easier to version
- easier to reuse for training
- easier to keep consistent

If you still want raw backups:

- keep raw videos on local storage or external drive

## 28. If You Want A Very Simple Capture Recommendation

Use this:

- device: your phone
- orientation: `landscape`
- resolution: `1080p`
- frame rate: `30 FPS`
- duration target: `3 to 4 hours total`
- storage budget: `25 to 35 GB free`
- workflow: phone -> PC -> processing -> Hugging Face

That is the simplest sensible setup.

## Sources

- MediaPipe Python `BaseOptions` docs: GPU delegate support is limited to Ubuntu platforms.
  - https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions
- MediaPipe GPU support docs: Linux desktop GPU use needs OpenGL ES `3.1+`.
  - https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support
- MediaPipe Face Landmarker Python docs: video/live modes use tracking to reduce latency.
  - https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- Hugging Face upload docs: `upload_folder()` supports dataset repos.
  - https://huggingface.co/docs/huggingface_hub/guides/upload
- Hugging Face dataset docs: Parquet is the recommended format for large datasets.
  - https://huggingface.co/docs/hub/datasets-adding
- LRS3:
  - https://www.robots.ox.ac.uk/~vgg/data/lip_reading/
- VoxCeleb:
  - https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- HDTF:
  - https://github.com/MRzzm/HDTF
- RAVDESS:
  - https://affectivedatascience.com/datasets
- MEAD:
  - https://wywu.github.io/projects/MEAD/MEAD.html

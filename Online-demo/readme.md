# Online Demo

## Run Locally

Start the local static server from inside `Online-demo`:

```bash
cd /Users/sanjayprasads/Desktop/Coding/Python/NN_training/Online-demo
./run_demo.sh
```

Or run the Python entrypoint directly:

```bash
cd /Users/sanjayprasads/Desktop/Coding/Python/NN_training/Online-demo
python3 server.py
```

Default port:

- `8000`

Custom port examples:

```bash
./run_demo.sh 8010
```

```bash
python3 server.py --port 8010
```

## Pages

Transformer-only page:

```text
http://127.0.0.1:8000/Online-demo/transformer-model.html
```

MediaPipe calibration page:

```text
http://127.0.0.1:8000/Online-demo/mediapipe-calibration.html
```

Three-way comparison page:

```text
http://127.0.0.1:8000/Online-demo/comparison.html
```

Landing page:

```text
http://127.0.0.1:8000/Online-demo/index.html
```

## What Each Page Does

`transformer-model.html`

- runs the tiny transformer ONNX model in the browser
- takes uploaded, recorded, or sample audio
- exports framewise blendshape output

`comparison.html`

- compares:
  - heuristic viseme baseline
  - old TCN ONNX model
  - tiny transformer ONNX model
- shows:
  - parameter count
  - ONNX size
  - architecture summary
  - side-by-side playback output

`mediapipe-calibration.html`

- uploads:
  - original source video
  - matching `blendshapes_and_pose.json`
- can also run:
  - live webcam tracking in-browser
- shows:
  - source video on the left
  - MediaPipe raccoon playback on the right
  - timestamp-synchronized frame scrubbing
  - mouth/jaw scaling for quick calibration checks

## Assets

The default demo assets live in:

- [assets/best_tcn_model_train_50.onnx](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/best_tcn_model_train_50.onnx)
- [assets/tiny_transformer_full10s_l1.onnx](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/tiny_transformer_full10s_l1.onnx)
- [assets/sample-audio.wav](/Users/sanjayprasads/Desktop/Coding/Python/NN_training/assets/sample-audio.wav)

The pages try local assets first. If the hosted site returns a `404` for `/assets/...`, the pages fall back to remote CDN/GitHub copies.

## Important

Do not open the HTML files with `file://`.

That breaks local asset loading and can also break browser security rules for model/audio fetching.

Use the local server instead.

## Stop The Server

Press:

```text
Ctrl+C
```

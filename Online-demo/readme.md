# Online Demo README

## What To Run

Run the demo from inside the `Online-demo` folder with:

```bash
./run_demo.sh
```

You can also run the Python server directly:

```bash
python3 server.py
```

## Why This Exists

Do not open the HTML files with `file://`.

That breaks local asset loading such as:

- `../assets/tiny_transformer_full10s_l1.onnx`

The local server fixes that by serving the **repo root** while letting you start it from the `Online-demo` folder.

## URLs

After starting the server, open:

```text
http://127.0.0.1:8000/Online-demo/transformer-model.html
```

Optional pages:

```text
http://127.0.0.1:8000/Online-demo/comparison.html
http://127.0.0.1:8000/Online-demo/index.html
```

## Port

Default port:

- `8000`

Custom port examples:

```bash
./run_demo.sh 8010
```

```bash
python3 server.py --port 8010
```

Then open:

```text
http://127.0.0.1:8010/Online-demo/transformer-model.html
```

## What The Server Does

`server.py` changes the working directory to the repository root and starts a standard Python static file server there.

That means:

- `Online-demo/...` is served correctly
- `assets/...` is served correctly
- the transformer ONNX file can load locally

## Quick Demo Checklist

1. Open a terminal.
2. `cd /Users/sanjayprasads/Desktop/Coding/Python/NN_training/Online-demo`
3. Run `./run_demo.sh`
4. Open `http://127.0.0.1:8000/Online-demo/transformer-model.html`
5. If the default model does not auto-load yet in production, test locally first with the local server

## Stop The Server

Press:

```text
Ctrl+C
```

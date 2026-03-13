#!/usr/bin/env python3
"""
Export the tiny transformer checkpoint to ONNX with the existing repo I/O names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from models.tiny_transformer_model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiny transformer checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit_best.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit.onnx",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit.json",
    )
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = create_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_dim = checkpoint["model_config"]["input_dim"]
    output_dim = checkpoint["model_config"]["output_dim"]
    dummy_input = torch.randn(1, args.seq_len, input_dim, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["audio_features"],
        output_names=["blendshapes"],
        dynamic_axes={
            "audio_features": {0: "batch_size", 1: "sequence_length"},
            "blendshapes": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    def maybe_relative(path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    manifest = {
        "model_type": "TinyTransformer_Audio_to_Blendshapes",
        "pytorch_source": maybe_relative(args.checkpoint, ROOT_PATH / "2_architecture_training"),
        "onnx_path": maybe_relative(args.output, ROOT_PATH / "2_architecture_training"),
        "input_shape": ["batch_size", "sequence_length", input_dim],
        "output_shape": ["batch_size", "sequence_length", output_dim],
        "input_name": "audio_features",
        "output_name": "blendshapes",
        "description": {
            "input": "80 mel-spectrogram features",
            "output": f"{output_dim} values per frame",
        },
        "model_config": checkpoint["model_config"],
        "model_info": checkpoint.get("model_info", {}),
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Tiny Transformer ONNX Export ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX output: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

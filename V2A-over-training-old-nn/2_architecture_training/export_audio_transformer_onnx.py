#!/usr/bin/env python3
"""
Export configurable audio transformer checkpoints to ONNX.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from models.audio_transformer_variants import clamp_output_to_natural_range, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export audio transformer checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


class NaturalRangeExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        output_space: str,
        pose_dims: int,
        pose_scale: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_space = output_space
        self.pose_dims = pose_dims
        self.pose_scale = pose_scale
        self.register_buffer(
            "target_mean",
            torch.tensor(target_mean.reshape(1, 1, -1), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "target_std",
            torch.tensor(target_std.reshape(1, 1, -1), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        output = self.model(audio_features)
        if self.output_space == "standardized":
            output = (output * self.target_std) + self.target_mean
        return clamp_output_to_natural_range(
            output,
            pose_dims=self.pose_dims,
            pose_scale=self.pose_scale,
        )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    base_model = create_model(checkpoint["model_config"])
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    target_norm = checkpoint.get("target_normalization", {})
    target_mean = np.asarray(
        target_norm.get("mean", np.zeros(checkpoint["model_config"]["output_dim"], dtype=np.float32)),
        dtype=np.float32,
    )
    target_std = np.asarray(
        target_norm.get("std", np.ones(checkpoint["model_config"]["output_dim"], dtype=np.float32)),
        dtype=np.float32,
    )
    output_space = checkpoint.get("output_space", "natural_range")
    pose_dims = int(checkpoint["model_config"].get("pose_dims", 7))
    pose_scale = float(checkpoint["model_config"].get("pose_scale", 0.2))

    model = NaturalRangeExportWrapper(
        model=base_model,
        target_mean=target_mean,
        target_std=target_std,
        output_space=output_space,
        pose_dims=pose_dims,
        pose_scale=pose_scale,
    )
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

    manifest = {
        "model_type": "AudioTransformerVariants",
        "onnx_path": str(args.output),
        "input_shape": ["batch_size", "sequence_length", input_dim],
        "output_shape": ["batch_size", "sequence_length", output_dim],
        "input_name": "audio_features",
        "output_name": "blendshapes",
        "model_config": checkpoint["model_config"],
        "model_info": checkpoint.get("model_info", {}),
        "output_space": "natural_range",
        "target_normalization": target_norm,
        "export_postprocess": {
            "destandardized": output_space == "standardized",
            "blendshape_clamp": [0.0, 1.0],
            "pose_clamp": [-pose_scale, pose_scale],
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Audio Transformer ONNX Export ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"ONNX output: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

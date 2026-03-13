#!/usr/bin/env python3
"""
Inference script for the tiny transformer audio-to-blendshape model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import torch

import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH / "2_architecture_training"))

from models.tiny_transformer_model import create_model


BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the tiny transformer overfit checkpoint."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--audio",
        type=Path,
        help="Audio file to convert to mel features before inference.",
    )
    source_group.add_argument(
        "--features",
        type=Path,
        help="Path to a .npy feature array with shape (T, 80) or (1, T, 80).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit_best.pth",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=ROOT_PATH / "data" / "inference" / "tiny_transformer_output_30fps.npy",
        help="30 FPS prediction output.",
    )
    parser.add_argument(
        "--output-100hz-npy",
        type=Path,
        default=ROOT_PATH / "data" / "inference" / "tiny_transformer_output_100hz.npy",
        help="100 Hz prediction output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON export in the existing frame-wise format.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument(
        "--fixed-seconds",
        type=float,
        default=None,
        help="Force the feature sequence to an exact duration, e.g. 10.0 seconds.",
    )
    parser.add_argument(
        "--features-already-normalized",
        action="store_true",
        help="Skip checkpoint mean/std normalization for .npy feature input.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Optional chunk size for long-form inference. Use 10.0 for 10-second windows.",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=5.0,
        help="Overlap between long-form inference chunks in seconds.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


def extract_mel_features(audio_path: Path, max_duration: float | None) -> Tuple[np.ndarray, np.ndarray]:
    sample_rate = 16000
    hop_length = 160
    win_length = 400
    n_fft = 512
    n_mels = 80

    audio, _sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        duration=max_duration,
    )
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        fmin=0,
        fmax=sample_rate // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    timestamps = librosa.frames_to_time(
        np.arange(log_mel.shape[0]),
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float32)
    return log_mel, timestamps


def resample_sequence(
    values: np.ndarray,
    target_length: int,
) -> np.ndarray:
    if len(values) == target_length:
        return values.astype(np.float32)
    if len(values) == 0:
        raise ValueError("Cannot resample an empty sequence.")

    old_positions = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)

    if values.ndim == 1:
        return np.interp(new_positions, old_positions, values).astype(np.float32)

    resampled = np.empty((target_length, values.shape[1]), dtype=np.float32)
    for feature_idx in range(values.shape[1]):
        resampled[:, feature_idx] = np.interp(
            new_positions,
            old_positions,
            values[:, feature_idx],
        )
    return resampled


def force_fixed_duration(
    features: np.ndarray,
    timestamps: np.ndarray,
    fixed_seconds: float | None,
    feature_fps: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if fixed_seconds is None:
        return features, timestamps

    target_frames = int(round(fixed_seconds * feature_fps))
    fixed_features = resample_sequence(features, target_frames)
    fixed_timestamps = (np.arange(target_frames, dtype=np.float32) / feature_fps).astype(
        np.float32
    )
    return fixed_features, fixed_timestamps


def load_feature_array(features_path: Path) -> np.ndarray:
    features = np.load(features_path).astype(np.float32)
    if features.ndim == 3:
        if features.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 3D input, got {features.shape}")
        features = features[0]
    if features.ndim != 2 or features.shape[-1] != 80:
        raise ValueError(
            f"Expected feature shape (T, 80) or (1, T, 80), got {features.shape}"
        )
    return features


def normalize_features(
    features: np.ndarray,
    checkpoint: Dict,
    skip_normalization: bool,
) -> np.ndarray:
    if skip_normalization:
        return features

    norm = checkpoint["audio_normalization"]
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, -1)
    return (features - mean) / std


def infer(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(tensor).squeeze(0).cpu().numpy()
    return predictions.astype(np.float32)


def infer_chunked(
    model: torch.nn.Module,
    features: np.ndarray,
    device: torch.device,
    chunk_frames: int,
    overlap_frames: int,
) -> np.ndarray:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if overlap_frames < 0:
        raise ValueError("overlap_frames cannot be negative")
    if overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be smaller than chunk_frames")

    total_frames = features.shape[0]
    if total_frames <= chunk_frames:
        return infer(model, features, device)

    step_frames = chunk_frames - overlap_frames
    starts = list(range(0, max(total_frames - chunk_frames + 1, 1), step_frames))
    tail_start = total_frames - chunk_frames
    if starts[-1] != tail_start:
        starts.append(tail_start)
    starts = sorted(set(starts))

    output_dim = model.config.output_dim
    merged = np.zeros((total_frames, output_dim), dtype=np.float32)
    weights = np.zeros((total_frames, 1), dtype=np.float32)

    window_weights = np.ones((chunk_frames,), dtype=np.float32)
    if overlap_frames > 0:
        ramp = np.linspace(0.0, 1.0, overlap_frames + 2, dtype=np.float32)[1:-1]
        window_weights[:overlap_frames] = ramp
        window_weights[-overlap_frames:] = ramp[::-1]

    for start in starts:
        end = start + chunk_frames
        chunk = features[start:end]
        chunk_predictions = infer(model, chunk, device)
        chunk_weight = window_weights[: len(chunk_predictions)].reshape(-1, 1)
        merged[start:end] += chunk_predictions * chunk_weight
        weights[start:end] += chunk_weight

    weights = np.clip(weights, 1e-6, None)
    return (merged / weights).astype(np.float32)


def downsample_predictions(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    target_fps: float,
    fixed_seconds: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    mel_frame_rate = 100.0
    if target_fps >= mel_frame_rate:
        return predictions, timestamps

    if fixed_seconds is not None:
        num_output_frames = int(round(fixed_seconds * target_fps))
        output_predictions = resample_sequence(predictions, num_output_frames)
        output_timestamps = (
            np.arange(num_output_frames, dtype=np.float32) / target_fps
        ).astype(np.float32)
        return output_predictions, output_timestamps

    ratio = mel_frame_rate / target_fps
    num_output_frames = max(1, int(round(len(predictions) / ratio)))
    indices = np.linspace(0, len(predictions) - 1, num_output_frames, dtype=np.int32)
    return predictions[indices], timestamps[indices]


def build_json_results(
    predictions_30fps: np.ndarray,
    timestamps_30fps: np.ndarray,
    checkpoint_path: Path,
    audio_path: Path | None,
) -> Dict:
    frames = []
    output_dim = predictions_30fps.shape[-1]

    for frame_idx, (timestamp, values) in enumerate(zip(timestamps_30fps, predictions_30fps)):
        blendshape_values = values[: min(52, output_dim)]
        blendshapes = {
            name: float(blendshape_values[idx]) if idx < len(blendshape_values) else 0.0
            for idx, name in enumerate(BLENDSHAPE_NAMES)
        }

        if output_dim >= 59:
            head_position = {
                "x": float(values[52]),
                "y": float(values[53]),
                "z": float(values[54]),
            }
            head_rotation = {
                "w": float(values[55]),
                "x": float(values[56]),
                "y": float(values[57]),
                "z": float(values[58]),
            }
        else:
            head_position = {"x": 0.0, "y": 0.0, "z": 0.0}
            head_rotation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

        frames.append(
            {
                "frame_index": frame_idx,
                "timestamp": int(timestamp * 1000),
                "blendshapes": blendshapes,
                "headPosition": head_position,
                "headRotation": head_rotation,
                "has_face": True,
                "sessionId": "tiny_transformer_inference",
            }
        )

    return {
        "sessionInfo": {
            "sessionId": "tiny_transformer_inference",
            "targetFPS": 30.0,
            "originalFPS": 100.0,
            "audioPath": str(audio_path) if audio_path else None,
            "modelPath": str(checkpoint_path),
            "inferenceMode": True,
        },
        "frameCount": len(frames),
        "failedFrames": 0,
        "failureRate": 0.0,
        "frames": frames,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    model = create_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.audio is not None:
        features, timestamps = extract_mel_features(args.audio, args.max_duration)
        features, timestamps = force_fixed_duration(
            features,
            timestamps,
            args.fixed_seconds,
        )
        normalized_features = normalize_features(features, checkpoint, skip_normalization=False)
    else:
        features = load_feature_array(args.features)
        timestamps = (np.arange(features.shape[0], dtype=np.float32) / 100.0).astype(np.float32)
        features, timestamps = force_fixed_duration(
            features,
            timestamps,
            args.fixed_seconds,
        )
        normalized_features = normalize_features(
            features,
            checkpoint,
            skip_normalization=args.features_already_normalized,
        )

    if args.chunk_seconds is not None:
        chunk_frames = int(round(args.chunk_seconds * 100.0))
        overlap_frames = int(round(args.chunk_overlap_seconds * 100.0))
        predictions_100hz = infer_chunked(
            model,
            normalized_features,
            device,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        )
    else:
        predictions_100hz = infer(model, normalized_features, device)
    predictions_30fps, timestamps_30fps = downsample_predictions(
        predictions_100hz,
        timestamps,
        args.fps,
        args.fixed_seconds,
    )

    args.output_100hz_npy.parent.mkdir(parents=True, exist_ok=True)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_100hz_npy, predictions_100hz.astype(np.float32))
    np.save(args.output_npy, predictions_30fps.astype(np.float32))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        results = build_json_results(
            predictions_30fps=predictions_30fps,
            timestamps_30fps=timestamps_30fps,
            checkpoint_path=args.checkpoint,
            audio_path=args.audio,
        )
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

    print("=== Tiny Transformer Inference ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input frames: {features.shape[0]} at 100 Hz")
    print(f"Output 100 Hz shape: {predictions_100hz.shape}")
    print(f"Output {args.fps:.1f} FPS shape: {predictions_30fps.shape}")
    if args.chunk_seconds is not None:
        print(
            f"Chunked inference: chunk={args.chunk_seconds:.2f}s "
            f"overlap={args.chunk_overlap_seconds:.2f}s"
        )
    if args.fixed_seconds is not None:
        print(
            f"Fixed duration: {args.fixed_seconds:.3f}s -> "
            f"{predictions_100hz.shape[0]} @100Hz and {predictions_30fps.shape[0]} @{args.fps:.1f}FPS"
        )
    print(f"Saved 100 Hz predictions to: {args.output_100hz_npy}")
    print(f"Saved {args.fps:.1f} FPS predictions to: {args.output_npy}")
    if args.output_json is not None:
        print(f"Saved JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()

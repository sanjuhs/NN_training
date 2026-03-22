#!/usr/bin/env python3
"""
Process a corpus of videos into a combined long-context audio-to-blendshape dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_CLEANING_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blendshape_module = load_module(
    "extract_blendshapes_module", DATA_CLEANING_DIR / "1_extract_blendshapes.py"
)
audio_module = load_module(
    "extract_audio_module", DATA_CLEANING_DIR / "2_extract_audio_features.py"
)
dataset_module = load_module(
    "create_dataset_module", DATA_CLEANING_DIR / "3_create_datset.py"
)

FaceBlendshapeExtractor = blendshape_module.FaceBlendshapeExtractor
AudioFeatureExtractor = audio_module.AudioFeatureExtractor
DatasetCreator = dataset_module.DatasetCreator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a long-context combined dataset from multiple video roots."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        action="append",
        required=True,
        help="Root directory containing videos. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        required=True,
        help="Persistent directory where per-video processing artifacts will be cached.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the combined dataset should be written.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=10000,
        help="Sequence length in milliseconds. Default is 10 seconds.",
    )
    parser.add_argument(
        "--overlap-ms",
        type=int,
        default=5000,
        help="Overlap between windows in milliseconds.",
    )
    parser.add_argument(
        "--blendshape-fps",
        type=int,
        default=30,
        help="Frame rate to use when extracting MediaPipe targets.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Request the MediaPipe GPU delegate when available.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=0,
        help="Process at most this many videos. 0 means no limit.",
    )
    return parser.parse_args()


def enumerate_videos(video_roots: List[Path]) -> List[Path]:
    videos: List[Path] = []
    for root in video_roots:
        if not root.exists():
            print(f"Skipping missing video root: {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(path)
    return videos


def clip_id_for(video_path: Path) -> str:
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:10]
    stem = video_path.stem.replace(" ", "_")
    return f"{stem}_{digest}"


def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    return json.loads((dataset_dir / "dataset_metadata.json").read_text())


def ensure_per_clip_dataset(
    video_path: Path,
    clip_dir: Path,
    face_extractor: Any,
    audio_extractor: Any,
    dataset_creator: Any,
    blendshape_fps: int,
) -> Path | None:
    dataset_dir = clip_dir / "dataset"
    ready_file = dataset_dir / "audio_sequences.npy"
    if ready_file.exists():
        print(f"Reusing cached clip dataset: {dataset_dir}")
        return dataset_dir

    features_dir = clip_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing clip ===")
    print(video_path)

    visual_data = face_extractor.extract_from_video(
        str(video_path),
        output_dir=str(features_dir),
        fps_limit=blendshape_fps,
    )
    audio_data = audio_extractor.extract_from_video(
        str(video_path),
        output_dir=str(features_dir),
    )

    synchronized = dataset_creator.synchronize_features(audio_data, visual_data)
    sequences = dataset_creator.create_sequences(synchronized)

    if len(sequences["audio_sequences"]) == 0:
        print(f"Skipping {video_path} because no valid sequences were created.")
        return None

    normalized = dataset_creator.normalize_features(sequences, output_dir=str(dataset_dir))
    dataset_creator.save_dataset(normalized, output_dir=str(dataset_dir))

    metadata = load_dataset_metadata(dataset_dir)
    metadata["source_video"] = str(video_path)
    metadata["features_dir"] = str(features_dir)
    (dataset_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))

    return dataset_dir


def main() -> None:
    args = parse_args()
    args.working_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = enumerate_videos(args.video_root)
    if args.limit_videos > 0:
        videos = videos[: args.limit_videos]

    if not videos:
        raise ValueError("No videos found under the provided roots.")

    face_extractor = FaceBlendshapeExtractor(use_gpu=args.use_gpu)
    audio_extractor = AudioFeatureExtractor()
    dataset_creator = DatasetCreator(
        sequence_length_ms=args.window_ms,
        overlap_ms=args.overlap_ms,
    )

    combined_audio: List[np.ndarray] = []
    combined_targets: List[np.ndarray] = []
    combined_vad: List[np.ndarray] = []
    segment_ids: List[np.ndarray] = []
    window_starts: List[np.ndarray] = []
    source_manifest: List[Dict[str, Any]] = []

    next_segment_id = 0
    for video_path in videos:
        clip_dir = args.working_dir / "clips" / clip_id_for(video_path)
        dataset_dir = ensure_per_clip_dataset(
            video_path=video_path,
            clip_dir=clip_dir,
            face_extractor=face_extractor,
            audio_extractor=audio_extractor,
            dataset_creator=dataset_creator,
            blendshape_fps=args.blendshape_fps,
        )
        if dataset_dir is None:
            continue

        audio = np.load(dataset_dir / "audio_sequences.npy").astype(np.float32)
        targets = np.load(dataset_dir / "target_sequences.npy").astype(np.float32)
        vad = np.load(dataset_dir / "vad_sequences.npy").astype(np.float32)
        metadata = load_dataset_metadata(dataset_dir)
        dataset_info = metadata["dataset_info"]

        segment_id_array = np.full(audio.shape[0], next_segment_id, dtype=np.int32)
        step_size_frames = int(dataset_info["step_size_frames"])
        window_start_array = np.arange(audio.shape[0], dtype=np.int32) * step_size_frames

        combined_audio.append(audio)
        combined_targets.append(targets)
        combined_vad.append(vad)
        segment_ids.append(segment_id_array)
        window_starts.append(window_start_array)
        source_manifest.append(
            {
                "segment_id": int(next_segment_id),
                "source_video": str(video_path),
                "dataset_dir": str(dataset_dir),
                "num_windows": int(audio.shape[0]),
                "sequence_length_frames": int(dataset_info["sequence_length_frames"]),
                "step_size_frames": int(step_size_frames),
            }
        )
        next_segment_id += 1

    if not combined_audio:
        raise ValueError("No clip datasets were produced.")

    audio_array = np.concatenate(combined_audio, axis=0)
    target_array = np.concatenate(combined_targets, axis=0)
    vad_array = np.concatenate(combined_vad, axis=0)
    segment_id_array = np.concatenate(segment_ids, axis=0)
    window_start_array = np.concatenate(window_starts, axis=0)

    np.save(args.output_dir / "audio_sequences.npy", audio_array)
    np.save(args.output_dir / "target_sequences.npy", target_array)
    np.save(args.output_dir / "vad_sequences.npy", vad_array)
    np.save(args.output_dir / "segment_ids.npy", segment_id_array)
    np.save(args.output_dir / "window_start_frames.npy", window_start_array)

    reference_metadata = load_dataset_metadata(Path(source_manifest[0]["dataset_dir"]))
    output_metadata = {
        "dataset_info": {
            **reference_metadata["dataset_info"],
            "num_sequences": int(audio_array.shape[0]),
            "sample_rate_hz": int(
                round(
                    reference_metadata["dataset_info"]["sequence_length_frames"]
                    * 1000
                    / reference_metadata["dataset_info"]["sequence_length_ms"]
                )
            ),
        },
        "normalization_method": reference_metadata.get("normalization_method"),
        "normalization_stats": reference_metadata.get("normalization_stats"),
        "source_manifest": source_manifest,
        "video_roots": [str(path) for path in args.video_root],
    }
    (args.output_dir / "dataset_metadata.json").write_text(json.dumps(output_metadata, indent=2))

    print("=== Combined Long-Context Dataset ===")
    print(f"Videos processed: {len(source_manifest)}")
    print(f"Audio shape: {audio_array.shape}")
    print(f"Target shape: {target_array.shape}")
    print(f"VAD shape: {vad_array.shape}")
    print(f"Unique segment ids: {len(np.unique(segment_id_array))}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()

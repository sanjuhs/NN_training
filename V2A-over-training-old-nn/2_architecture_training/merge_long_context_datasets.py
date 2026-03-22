#!/usr/bin/env python3
"""
Merge multiple long-context datasets that share the same tensor layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple long-context dataset directories into one dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="Dataset directory containing audio_sequences.npy and friends. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write the merged dataset.",
    )
    return parser.parse_args()


def load_dataset(input_dir: Path) -> Dict[str, np.ndarray | Dict]:
    payload = {
        "audio": np.load(input_dir / "audio_sequences.npy").astype(np.float32),
        "targets": np.load(input_dir / "target_sequences.npy").astype(np.float32),
        "vad": np.load(input_dir / "vad_sequences.npy").astype(np.float32),
        "segment_ids": (
            np.load(input_dir / "segment_ids.npy").astype(np.int32)
            if (input_dir / "segment_ids.npy").exists()
            else np.zeros(
                np.load(input_dir / "audio_sequences.npy").shape[0], dtype=np.int32
            )
        ),
        "window_starts": (
            np.load(input_dir / "window_start_frames.npy").astype(np.int32)
            if (input_dir / "window_start_frames.npy").exists()
            else np.zeros(
                np.load(input_dir / "audio_sequences.npy").shape[0], dtype=np.int32
            )
        ),
        "metadata": json.loads((input_dir / "dataset_metadata.json").read_text()),
    }
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged_audio: List[np.ndarray] = []
    merged_targets: List[np.ndarray] = []
    merged_vad: List[np.ndarray] = []
    merged_segment_ids: List[np.ndarray] = []
    merged_window_starts: List[np.ndarray] = []
    source_datasets: List[Dict[str, object]] = []

    next_segment_offset = 0
    reference_info = None

    for input_dir in args.input_dir:
        payload = load_dataset(input_dir)
        dataset_info = payload["metadata"]["dataset_info"]
        if reference_info is None:
            reference_info = dataset_info
        else:
            for field in ("sequence_length_frames", "audio_feature_dim", "target_feature_dim"):
                if int(dataset_info[field]) != int(reference_info[field]):
                    raise ValueError(
                        f"Incompatible dataset {input_dir}: field {field} does not match"
                    )

        segment_ids = payload["segment_ids"]
        if segment_ids.size > 0:
            unique_ids = np.unique(segment_ids)
            remap = {old: new for new, old in enumerate(unique_ids, start=next_segment_offset)}
            segment_ids = np.vectorize(remap.get)(segment_ids).astype(np.int32)
            next_segment_offset = max(remap.values()) + 1

        merged_audio.append(payload["audio"])
        merged_targets.append(payload["targets"])
        merged_vad.append(payload["vad"])
        merged_segment_ids.append(segment_ids)
        merged_window_starts.append(payload["window_starts"])
        source_datasets.append(
            {
                "input_dir": str(input_dir),
                "num_sequences": int(payload["audio"].shape[0]),
                "unique_segments": int(len(np.unique(segment_ids))),
            }
        )

    audio_array = np.concatenate(merged_audio, axis=0)
    target_array = np.concatenate(merged_targets, axis=0)
    vad_array = np.concatenate(merged_vad, axis=0)
    segment_id_array = np.concatenate(merged_segment_ids, axis=0)
    window_start_array = np.concatenate(merged_window_starts, axis=0)

    np.save(args.output_dir / "audio_sequences.npy", audio_array)
    np.save(args.output_dir / "target_sequences.npy", target_array)
    np.save(args.output_dir / "vad_sequences.npy", vad_array)
    np.save(args.output_dir / "segment_ids.npy", segment_id_array)
    np.save(args.output_dir / "window_start_frames.npy", window_start_array)

    metadata = {
        "dataset_info": {
            **reference_info,
            "num_sequences": int(audio_array.shape[0]),
        },
        "source_datasets": source_datasets,
    }
    (args.output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("=== Merged Long-Context Dataset ===")
    print(f"Output dir: {args.output_dir}")
    print(f"Audio shape: {audio_array.shape}")
    print(f"Target shape: {target_array.shape}")
    print(f"Unique segment ids: {len(np.unique(segment_id_array))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rebuild long-context training windows from the existing short-window dataset.

The current full training artifact stores overlapping 23-frame windows extracted with
an 11-frame step. This script stitches matching overlaps back into continuous segments,
then emits true long-context windows such as 10-second clips at 100 Hz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


ROOT_PATH = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a long-context dataset from overlapping short windows."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "data" / "train",
        help="Directory containing the existing short-window dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "data" / "train_long_10s_step500",
        help="Where to write the rebuilt long-context dataset.",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=1000,
        help="Output window length in frames. 1000 frames = 10 seconds at 100 Hz.",
    )
    parser.add_argument(
        "--window-step-frames",
        type=int,
        default=500,
        help="Stride between output windows in frames.",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=1000,
        help="Discard reconstructed continuous segments shorter than this.",
    )
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=1e-6,
        help="Maximum allowed overlap mismatch before starting a new segment.",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=1e-6,
        help="Maximum allowed target overlap mismatch before starting a new segment.",
    )
    parser.add_argument(
        "--cover-tail",
        action="store_true",
        help="Add a final window ending at the segment tail when stride does not land exactly.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=100,
        help="Frame rate for the synchronized per-frame features. The upstream extractor uses 100 Hz.",
    )
    return parser.parse_args()


def load_short_dataset(input_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    audio = np.load(input_dir / "audio_sequences.npy")
    targets = np.load(input_dir / "target_sequences.npy")
    vad = np.load(input_dir / "vad_sequences.npy")
    metadata = json.loads((input_dir / "dataset_metadata.json").read_text())
    return audio.astype(np.float32), targets.astype(np.float32), vad.astype(np.float32), metadata


def find_segments(
    audio: np.ndarray,
    targets: np.ndarray,
    vad: np.ndarray,
    sequence_length_frames: int,
    step_size_frames: int,
    audio_threshold: float,
    target_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
    overlap = sequence_length_frames - step_size_frames
    segments: List[Tuple[int, int]] = []
    breakpoints: List[Dict[str, float]] = []
    start = 0

    for idx in range(audio.shape[0] - 1):
        audio_diff = float(
            np.max(np.abs(audio[idx, step_size_frames:, :] - audio[idx + 1, :overlap, :]))
        )
        target_diff = float(
            np.max(np.abs(targets[idx, step_size_frames:, :] - targets[idx + 1, :overlap, :]))
        )
        vad_diff = float(np.max(np.abs(vad[idx, step_size_frames:] - vad[idx + 1, :overlap])))

        if audio_diff > audio_threshold or target_diff > target_threshold or vad_diff > 0.0:
            segments.append((start, idx))
            breakpoints.append(
                {
                    "after_sequence": idx,
                    "audio_overlap_max_abs_diff": audio_diff,
                    "target_overlap_max_abs_diff": target_diff,
                    "vad_overlap_max_abs_diff": vad_diff,
                }
            )
            start = idx + 1

    segments.append((start, audio.shape[0] - 1))
    return segments, breakpoints


def reconstruct_segment(
    audio: np.ndarray,
    targets: np.ndarray,
    vad: np.ndarray,
    start: int,
    end: int,
    sequence_length_frames: int,
    step_size_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = sequence_length_frames - step_size_frames
    audio_parts = [audio[start]]
    target_parts = [targets[start]]
    vad_parts = [vad[start]]

    for idx in range(start + 1, end + 1):
        audio_parts.append(audio[idx, overlap:, :])
        target_parts.append(targets[idx, overlap:, :])
        vad_parts.append(vad[idx, overlap:])

    return (
        np.concatenate(audio_parts, axis=0),
        np.concatenate(target_parts, axis=0),
        np.concatenate(vad_parts, axis=0),
    )


def build_windows_for_segment(
    audio_segment: np.ndarray,
    target_segment: np.ndarray,
    vad_segment: np.ndarray,
    segment_id: int,
    window_frames: int,
    window_step_frames: int,
    cover_tail: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    total_frames = audio_segment.shape[0]
    if total_frames < window_frames:
        return [], [], [], [], []

    starts = list(range(0, total_frames - window_frames + 1, window_step_frames))
    tail_start = total_frames - window_frames
    if cover_tail and starts and starts[-1] != tail_start:
        starts.append(tail_start)
    elif cover_tail and not starts:
        starts = [tail_start]

    starts = sorted(set(starts))

    audio_windows: List[np.ndarray] = []
    target_windows: List[np.ndarray] = []
    vad_windows: List[np.ndarray] = []
    segment_ids: List[int] = []
    window_starts: List[int] = []

    for start in starts:
        end = start + window_frames
        audio_windows.append(audio_segment[start:end])
        target_windows.append(target_segment[start:end])
        vad_windows.append(vad_segment[start:end])
        segment_ids.append(segment_id)
        window_starts.append(start)

    return audio_windows, target_windows, vad_windows, segment_ids, window_starts


def summarize(values: Sequence[int]) -> Dict[str, float]:
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values_np.min()) if len(values_np) else 0.0,
        "median": float(np.median(values_np)) if len(values_np) else 0.0,
        "mean": float(values_np.mean()) if len(values_np) else 0.0,
        "max": float(values_np.max()) if len(values_np) else 0.0,
    }


def main() -> None:
    args = parse_args()

    if args.window_frames <= 0:
        raise ValueError("--window-frames must be positive")
    if args.window_step_frames <= 0:
        raise ValueError("--window-step-frames must be positive")

    audio, targets, vad, metadata = load_short_dataset(args.input_dir)
    dataset_info = metadata["dataset_info"]
    source_seq_len = int(dataset_info["sequence_length_frames"])
    source_step = int(dataset_info["step_size_frames"])
    sample_rate_hz = int(args.sample_rate_hz)

    segments, breakpoints = find_segments(
        audio=audio,
        targets=targets,
        vad=vad,
        sequence_length_frames=source_seq_len,
        step_size_frames=source_step,
        audio_threshold=args.audio_threshold,
        target_threshold=args.target_threshold,
    )

    output_audio: List[np.ndarray] = []
    output_targets: List[np.ndarray] = []
    output_vad: List[np.ndarray] = []
    output_segment_ids: List[int] = []
    output_window_starts: List[int] = []
    segment_frame_lengths: List[int] = []
    kept_segment_ids: List[int] = []

    for segment_id, (start, end) in enumerate(segments):
        segment_audio, segment_targets, segment_vad = reconstruct_segment(
            audio=audio,
            targets=targets,
            vad=vad,
            start=start,
            end=end,
            sequence_length_frames=source_seq_len,
            step_size_frames=source_step,
        )
        segment_frames = int(segment_audio.shape[0])
        segment_frame_lengths.append(segment_frames)

        if segment_frames < args.min_segment_frames:
            continue

        (
            audio_windows,
            target_windows,
            vad_windows,
            segment_ids,
            window_starts,
        ) = build_windows_for_segment(
            audio_segment=segment_audio,
            target_segment=segment_targets,
            vad_segment=segment_vad,
            segment_id=segment_id,
            window_frames=args.window_frames,
            window_step_frames=args.window_step_frames,
            cover_tail=args.cover_tail,
        )

        if audio_windows:
            output_audio.extend(audio_windows)
            output_targets.extend(target_windows)
            output_vad.extend(vad_windows)
            output_segment_ids.extend(segment_ids)
            output_window_starts.extend(window_starts)
            kept_segment_ids.append(segment_id)

    if not output_audio:
        raise ValueError("No long-context windows were created. Lower the frame thresholds.")

    audio_array = np.stack(output_audio).astype(np.float32)
    target_array = np.stack(output_targets).astype(np.float32)
    vad_array = np.stack(output_vad).astype(np.float32)
    segment_id_array = np.asarray(output_segment_ids, dtype=np.int32)
    window_start_array = np.asarray(output_window_starts, dtype=np.int32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "audio_sequences.npy", audio_array)
    np.save(args.output_dir / "target_sequences.npy", target_array)
    np.save(args.output_dir / "vad_sequences.npy", vad_array)
    np.save(args.output_dir / "segment_ids.npy", segment_id_array)
    np.save(args.output_dir / "window_start_frames.npy", window_start_array)

    output_metadata = {
        "dataset_info": {
            "num_sequences": int(audio_array.shape[0]),
            "sequence_length_frames": int(args.window_frames),
            "sequence_length_ms": int(round(args.window_frames * 1000 / sample_rate_hz)),
            "step_size_frames": int(args.window_step_frames),
            "overlap_ms": int(
                round(max(args.window_frames - args.window_step_frames, 0) * 1000 / sample_rate_hz)
            ),
            "audio_feature_dim": int(audio_array.shape[-1]),
            "target_feature_dim": int(target_array.shape[-1]),
            "sample_rate_hz": int(sample_rate_hz),
        },
        "source_dataset_info": dataset_info,
        "reconstruction_info": {
            "source_sequence_count": int(audio.shape[0]),
            "reconstructed_segment_count": int(len(segments)),
            "kept_segment_count": int(len(set(kept_segment_ids))),
            "segment_frame_lengths": segment_frame_lengths,
            "segment_frame_length_summary": summarize(segment_frame_lengths),
            "kept_segment_ids": kept_segment_ids,
            "window_count_per_segment": {
                str(segment_id): int(np.sum(segment_id_array == segment_id))
                for segment_id in sorted(set(output_segment_ids))
            },
            "audio_threshold": args.audio_threshold,
            "target_threshold": args.target_threshold,
            "min_segment_frames": args.min_segment_frames,
            "cover_tail": args.cover_tail,
            "breakpoint_count": int(len(breakpoints)),
            "breakpoints_preview": breakpoints[:25],
        },
    }

    (args.output_dir / "dataset_metadata.json").write_text(json.dumps(output_metadata, indent=2))

    print("=== Rebuilt Long-Context Dataset ===")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Source audio shape: {audio.shape}")
    print(f"Reconstructed segments: {len(segments)}")
    print(f"Kept segments >= {args.min_segment_frames} frames: {len(set(kept_segment_ids))}")
    print(f"Output audio shape: {audio_array.shape}")
    print(f"Output target shape: {target_array.shape}")
    print(f"Window step: {args.window_step_frames} frames")
    print(f"Sample rate: {sample_rate_hz} Hz")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train a tiny transformer overfit model on the existing 10-second audio dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to import model definitions.
import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from models.tiny_transformer_model import TinyTransformerConfig, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny transformer overfit model for audio-to-blendshapes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_PATH / "data" / "extracted_features",
        help="Directory containing audio_sequences.npy, target_sequences.npy, and metadata.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit_best.pth",
        help="Path to save the best checkpoint.",
    )
    parser.add_argument(
        "--last-checkpoint-path",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "models" / "tiny_transformer_overfit_last.pth",
        help="Path to save the latest checkpoint.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ROOT_PATH / "2_architecture_training" / "plots" / "tiny_transformer_history.json",
        help="Where to write training history JSON.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=1200)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Used when --val-sequences is not set.",
    )
    parser.add_argument(
        "--val-sequences",
        type=int,
        default=2,
        help="Number of final windows to reserve for validation. Set 0 to use val-fraction.",
    )
    parser.add_argument(
        "--segment-aware-split",
        action="store_true",
        help="Prefer splitting by segment_ids.npy when that file exists.",
    )
    parser.add_argument(
        "--blendshapes-only",
        action="store_true",
        help="Train on the first 52 target channels only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=80,
        help="Early stop after this many epochs without validation improvement.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(
    data_dir: Path,
    blendshapes_only: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, np.ndarray | None]:
    audio = np.load(data_dir / "audio_sequences.npy").astype(np.float32)
    targets = np.load(data_dir / "target_sequences.npy").astype(np.float32)
    vad = np.load(data_dir / "vad_sequences.npy").astype(np.float32)

    with open(data_dir / "dataset_metadata.json", "r") as f:
        metadata = json.load(f)

    if blendshapes_only:
        targets = targets[..., :52]

    segment_ids_path = data_dir / "segment_ids.npy"
    segment_ids = (
        np.load(segment_ids_path).astype(np.int32) if segment_ids_path.exists() else None
    )

    return audio, targets, vad, metadata, segment_ids


def compute_window_split(
    num_sequences: int,
    sequence_length_frames: int,
    step_size_frames: int,
    val_sequences: int | None,
    val_fraction: float,
) -> Dict[str, List[int]]:
    if val_sequences is None:
        val_sequences = max(1, int(round(num_sequences * val_fraction)))

    if num_sequences < 3:
        raise ValueError(f"Expected at least 3 sequences, found {num_sequences}")

    if step_size_frames <= 0:
        gap_sequences = 0
    else:
        gap_sequences = max(0, math.ceil(sequence_length_frames / step_size_frames) - 1)

    val_sequences = max(1, min(val_sequences, num_sequences - 1))
    gap_sequences = min(gap_sequences, max(0, num_sequences - val_sequences - 1))
    train_end = num_sequences - val_sequences - gap_sequences

    if train_end < 1:
        raise ValueError(
            "Contiguous split left no training windows. Reduce val size or use a larger dataset."
        )

    train_indices = list(range(0, train_end))
    gap_indices = list(range(train_end, num_sequences - val_sequences))
    val_indices = list(range(num_sequences - val_sequences, num_sequences))

    return {
        "train_indices": train_indices,
        "gap_indices": gap_indices,
        "val_indices": val_indices,
        "gap_sequences": gap_sequences,
        "split_mode": "window_contiguous",
    }


def compute_segment_split(
    segment_ids: np.ndarray,
    val_sequences: int | None,
    val_fraction: float,
) -> Dict[str, List[int]]:
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids.npy must be a 1D array")

    ordered_segments = list(dict.fromkeys(segment_ids.tolist()))
    if len(ordered_segments) < 2:
        raise ValueError("Need at least two distinct segments for a segment-aware split")

    total_sequences = int(segment_ids.shape[0])
    target_val_sequences = (
        max(1, int(round(total_sequences * val_fraction)))
        if val_sequences is None
        else max(1, val_sequences)
    )

    segment_counts = {
        segment_id: int(np.sum(segment_ids == segment_id)) for segment_id in ordered_segments
    }
    val_segments: List[int] = []
    collected = 0

    for segment_id in reversed(ordered_segments):
        if len(val_segments) >= len(ordered_segments) - 1:
            break
        val_segments.append(segment_id)
        collected += segment_counts[segment_id]
        if collected >= target_val_sequences:
            break

    val_segment_set = set(val_segments)
    train_indices = [idx for idx, segment_id in enumerate(segment_ids) if segment_id not in val_segment_set]
    val_indices = [idx for idx, segment_id in enumerate(segment_ids) if segment_id in val_segment_set]

    if not train_indices or not val_indices:
        raise ValueError("Segment-aware split failed to produce both train and validation sets")

    return {
        "train_indices": train_indices,
        "gap_indices": [],
        "val_indices": val_indices,
        "gap_sequences": 0,
        "train_segments": [segment_id for segment_id in ordered_segments if segment_id not in val_segment_set],
        "val_segments": list(reversed(val_segments)),
        "split_mode": "segment_contiguous",
    }


def compute_split(
    num_sequences: int,
    sequence_length_frames: int,
    step_size_frames: int,
    val_sequences: int | None,
    val_fraction: float,
    segment_ids: np.ndarray | None,
    segment_aware_split: bool,
) -> Dict[str, List[int]]:
    if val_sequences is not None and val_sequences <= 0:
        val_sequences = None

    if segment_aware_split and segment_ids is not None and len(np.unique(segment_ids)) > 1:
        return compute_segment_split(
            segment_ids=segment_ids,
            val_sequences=val_sequences,
            val_fraction=val_fraction,
        )

    return compute_window_split(
        num_sequences=num_sequences,
        sequence_length_frames=sequence_length_frames,
        step_size_frames=step_size_frames,
        val_sequences=val_sequences,
        val_fraction=val_fraction,
    )


def normalize_audio(
    train_audio: np.ndarray,
    val_audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_audio.mean(axis=(0, 1), keepdims=True)
    std = train_audio.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train_audio - mean) / std, (val_audio - mean) / std, mean, std


class TemporalL1Loss(nn.Module):
    def __init__(self, temporal_weight: float = 0.02) -> None:
        super().__init__()
        self.temporal_weight = temporal_weight
        self.l1 = nn.L1Loss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        l1_loss = self.l1(predictions, targets)
        if predictions.size(1) > 1 and self.temporal_weight > 0:
            pred_delta = predictions[:, 1:] - predictions[:, :-1]
            target_delta = targets[:, 1:] - targets[:, :-1]
            temporal_loss = self.l1(pred_delta, target_delta)
        else:
            temporal_loss = predictions.new_tensor(0.0)

        total_loss = l1_loss + self.temporal_weight * temporal_loss
        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "temporal_loss": temporal_loss,
        }


def make_loader(
    audio: np.ndarray,
    targets: np.ndarray,
    vad: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(audio),
        torch.from_numpy(targets),
        torch.from_numpy(vad),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TemporalL1Loss,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_l1 = 0.0
    total_temporal = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for audio, targets, _vad in loader:
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(audio)
            loss_dict = criterion(predictions, targets)

        if is_train:
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_grad_norm += float(grad_norm)

        total_loss += float(loss_dict["loss"].detach().cpu())
        total_l1 += float(loss_dict["l1_loss"].detach().cpu())
        total_temporal += float(loss_dict["temporal_loss"].detach().cpu())
        num_batches += 1

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "l1_loss": total_l1 / max(num_batches, 1),
        "temporal_loss": total_temporal / max(num_batches, 1),
    }
    if is_train:
        metrics["grad_norm"] = total_grad_norm / max(num_batches, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    audio_mean: np.ndarray,
    audio_std: np.ndarray,
    metadata: Dict,
    split_info: Dict[str, List[int]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": asdict(model.config),
        "model_info": model.get_model_info(),
        "audio_normalization": {
            "mean": audio_mean.squeeze(0).squeeze(0).tolist(),
            "std": audio_std.squeeze(0).squeeze(0).tolist(),
        },
        "dataset_metadata": metadata,
        "split_info": split_info,
        "target_mode": "blendshapes_only" if args.blendshapes_only else "blendshapes_plus_pose",
        "training_args": vars(args),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    torch.save(checkpoint, path)


def write_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    audio, targets, vad, metadata, segment_ids = load_dataset(args.data_dir, args.blendshapes_only)
    dataset_info = metadata["dataset_info"]

    split_info = compute_split(
        num_sequences=audio.shape[0],
        sequence_length_frames=dataset_info["sequence_length_frames"],
        step_size_frames=dataset_info["step_size_frames"],
        val_sequences=args.val_sequences,
        val_fraction=args.val_fraction,
        segment_ids=segment_ids,
        segment_aware_split=args.segment_aware_split,
    )

    train_idx = split_info["train_indices"]
    val_idx = split_info["val_indices"]

    train_audio = audio[train_idx]
    train_targets = targets[train_idx]
    train_vad = vad[train_idx]

    val_audio = audio[val_idx]
    val_targets = targets[val_idx]
    val_vad = vad[val_idx]

    train_audio, val_audio, audio_mean, audio_std = normalize_audio(train_audio, val_audio)

    train_loader = make_loader(
        train_audio, train_targets, train_vad, batch_size=args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        val_audio, val_targets, val_vad, batch_size=args.batch_size, shuffle=False
    )

    model_config = TinyTransformerConfig(
        input_dim=train_audio.shape[-1],
        output_dim=train_targets.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        max_seq_len=max(args.max_seq_len, train_audio.shape[1]),
        pose_dims=0 if args.blendshapes_only else 7,
    )
    model = create_model(asdict(model_config)).to(device)

    criterion = TemporalL1Loss(temporal_weight=args.temporal_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    history = {
        "train_loss": [],
        "train_l1_loss": [],
        "train_temporal_loss": [],
        "train_grad_norm": [],
        "val_loss": [],
        "val_l1_loss": [],
        "val_temporal_loss": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0

    print("=== Tiny Transformer Overfit Training ===")
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Audio shape: {audio.shape}")
    print(f"Target shape: {targets.shape}")
    if segment_ids is not None:
        print(f"Distinct segment ids: {len(np.unique(segment_ids))}")
    print(f"Split mode: {split_info.get('split_mode', 'unknown')}")
    print(f"Split: train={train_idx}, gap={split_info['gap_indices']}, val={val_idx}")
    if "train_segments" in split_info or "val_segments" in split_info:
        print(f"Train segments: {split_info.get('train_segments', [])}")
        print(f"Val segments: {split_info.get('val_segments', [])}")
    print(f"Model: {json.dumps(model.get_model_info(), indent=2)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_l1_loss"].append(train_metrics["l1_loss"])
        history["train_temporal_loss"].append(train_metrics["temporal_loss"])
        history["train_grad_norm"].append(train_metrics["grad_norm"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_l1_loss"].append(val_metrics["l1_loss"])
        history["val_temporal_loss"].append(val_metrics["temporal_loss"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_metrics['loss']:.6f} "
            f"(l1={train_metrics['l1_loss']:.6f}, temp={train_metrics['temporal_loss']:.6f}) | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"(l1={val_metrics['l1_loss']:.6f}, temp={val_metrics['temporal_loss']:.6f}) | "
            f"grad={train_metrics['grad_norm']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        save_checkpoint(
            path=args.last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            audio_mean=audio_mean,
            audio_std=audio_std,
            metadata=metadata,
            split_info=split_info,
            args=args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                path=args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                audio_mean=audio_mean,
                audio_std=audio_std,
                metadata=metadata,
                split_info=split_info,
                args=args,
            )
            print(f"  Saved new best checkpoint to {args.checkpoint_path}")

        write_history(args.history_path, history)

        if epoch - best_epoch >= args.patience:
            print(
                f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement."
            )
            break

    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

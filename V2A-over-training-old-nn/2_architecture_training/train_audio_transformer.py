#!/usr/bin/env python3
"""
Train configurable transformer variants for audio-to-blendshape regression.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_PATH))

from blendshape_layout import (
    CURVE_DEBUG_CHANNEL_INDICES,
    CURVE_DEBUG_CHANNEL_NAMES,
    JAW_OPEN_INDEX,
    MOUTH_AND_JAW_INDICES,
    MOUTH_CLOSE_INDEX,
    POSE_INDICES,
    SMILE_INDICES,
)
from models.audio_transformer_variants import (
    AudioTransformerConfig,
    clamp_output_to_natural_range,
    create_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train configurable transformer variants for audio-to-blendshapes."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--last-checkpoint-path", type=Path, required=True)
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--plot-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--curve-plot-path", type=Path, default=None)
    parser.add_argument("--curve-sample-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "conv_transformer",
            "gated_transformer",
            "conv_gated_transformer",
            "conformer_transformer",
            "multiscale_transformer",
        ],
        default="baseline",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--max-run-epochs", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam", "nadam", "radam", "adamax"],
        default="adamw",
    )
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "warmup_cosine"],
        default="warmup_cosine",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-loss", choices=["l1", "huber"], default="l1")
    parser.add_argument("--huber-delta", type=float, default=0.75)
    parser.add_argument("--temporal-weight", type=float, default=0.02)
    parser.add_argument("--corr-weight", type=float, default=0.15)
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--mouth-weight-scale", type=float, default=1.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=1536)
    parser.add_argument("--conv-kernel-size", type=int, default=9)
    parser.add_argument("--max-seq-len", type=int, default=1200)
    parser.add_argument(
        "--target-normalization",
        choices=["none", "standardize"],
        default="none",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-sequences", type=int, default=0)
    parser.add_argument("--segment-aware-split", action="store_true")
    parser.add_argument("--blendshapes-only", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--patience", type=int, default=30)
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

    gap_sequences = 0 if step_size_frames <= 0 else max(
        0, math.ceil(sequence_length_frames / step_size_frames) - 1
    )
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
    ordered_segments = list(dict.fromkeys(segment_ids.tolist()))
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
    train_indices = [
        idx for idx, segment_id in enumerate(segment_ids) if segment_id not in val_segment_set
    ]
    val_indices = [
        idx for idx, segment_id in enumerate(segment_ids) if segment_id in val_segment_set
    ]
    return {
        "train_indices": train_indices,
        "gap_indices": [],
        "val_indices": val_indices,
        "gap_sequences": 0,
        "train_segments": [
            segment_id for segment_id in ordered_segments if segment_id not in val_segment_set
        ],
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
        return compute_segment_split(segment_ids, val_sequences, val_fraction)
    return compute_window_split(
        num_sequences, sequence_length_frames, step_size_frames, val_sequences, val_fraction
    )


def normalize_audio(
    train_audio: np.ndarray,
    val_audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_audio.mean(axis=(0, 1), keepdims=True)
    std = train_audio.std(axis=(0, 1), keepdims=True) + 1e-6
    return (train_audio - mean) / std, (val_audio - mean) / std, mean, std


def normalize_targets(
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if mode == "standardize":
        mean = train_targets.mean(axis=(0, 1), keepdims=True)
        std = train_targets.std(axis=(0, 1), keepdims=True)
        std = np.maximum(std, 1e-4).astype(np.float32)
        train_targets = (train_targets - mean) / std
        val_targets = (val_targets - mean) / std
        return train_targets, val_targets, mean.astype(np.float32), std.astype(np.float32), "standardized"

    zeros = np.zeros((1, 1, train_targets.shape[-1]), dtype=np.float32)
    ones = np.ones((1, 1, train_targets.shape[-1]), dtype=np.float32)
    return train_targets, val_targets, zeros, ones, "natural_range"


def restore_output_space(
    values: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    output_space: str,
    pose_dims: int,
    pose_scale: float,
) -> torch.Tensor:
    if output_space == "standardized":
        values = (values * target_std) + target_mean
    return clamp_output_to_natural_range(values, pose_dims=pose_dims, pose_scale=pose_scale)


class BalancedRegressionLoss(nn.Module):
    def __init__(
        self,
        temporal_weight: float,
        corr_weight: float,
        variance_weight: float,
        mouth_weight_scale: float,
        blendshape_dims: int,
        base_loss: str,
        huber_delta: float,
    ) -> None:
        super().__init__()
        self.temporal_weight = temporal_weight
        self.corr_weight = corr_weight
        self.variance_weight = variance_weight
        self.mouth_weight_scale = mouth_weight_scale
        self.blendshape_dims = blendshape_dims
        self.base_loss = base_loss
        self.huber_delta = huber_delta

    def _channel_weights(self, predictions: torch.Tensor) -> torch.Tensor:
        channel_weights = torch.ones(
            predictions.size(-1),
            device=predictions.device,
            dtype=predictions.dtype,
        )
        mouth_indices = [idx for idx in MOUTH_AND_JAW_INDICES if idx < predictions.size(-1)]
        if mouth_indices:
            channel_weights[mouth_indices] = self.mouth_weight_scale
        return channel_weights

    def _pointwise_error(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.base_loss == "huber":
            return torch.nn.functional.smooth_l1_loss(
                predictions,
                targets,
                beta=self.huber_delta,
                reduction="none",
            )
        return torch.abs(predictions - targets)

    def _channel_correlation_stats(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_flat = predictions.reshape(-1, predictions.size(-1))
        target_flat = targets.reshape(-1, targets.size(-1))

        pred_center = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        target_center = target_flat - target_flat.mean(dim=0, keepdim=True)

        pred_std = pred_center.square().mean(dim=0).sqrt()
        target_std = target_center.square().mean(dim=0).sqrt()
        valid = (pred_std > 1e-6) & (target_std > 1e-6)

        corr = torch.zeros_like(pred_std)
        if valid.any():
            covariance = (pred_center[:, valid] * target_center[:, valid]).mean(dim=0)
            corr_valid = covariance / (pred_std[valid] * target_std[valid] + 1e-6)
            corr[valid] = torch.clamp(corr_valid, -1.0, 1.0)
        return corr, valid, pred_std, target_std

    def _mean_corr(
        self,
        corr: torch.Tensor,
        valid: torch.Tensor,
        indices: List[int],
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if not indices:
            return fallback
        index_tensor = torch.tensor(indices, device=corr.device, dtype=torch.long)
        mask = valid[index_tensor]
        if mask.any():
            return corr[index_tensor][mask].mean()
        return fallback

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        channel_weights = self._channel_weights(predictions)
        mouth_indices = [idx for idx in MOUTH_AND_JAW_INDICES if idx < predictions.size(-1)]
        blendshape_indices = list(range(min(self.blendshape_dims, predictions.size(-1))))

        l1_loss = (self._pointwise_error(predictions, targets) * channel_weights.view(1, 1, -1)).mean()

        if predictions.size(1) > 1 and self.temporal_weight > 0:
            pred_delta = predictions[:, 1:] - predictions[:, :-1]
            target_delta = targets[:, 1:] - targets[:, :-1]
            temporal_loss = (
                torch.abs(pred_delta - target_delta) * channel_weights.view(1, 1, -1)
            ).mean()
        else:
            temporal_loss = predictions.new_tensor(0.0)

        corr, valid_corr, pred_std, target_std = self._channel_correlation_stats(
            predictions, targets
        )
        overall_corr = self._mean_corr(
            corr,
            valid_corr,
            blendshape_indices,
            fallback=predictions.new_tensor(1.0),
        )
        mouth_corr = self._mean_corr(
            corr,
            valid_corr,
            mouth_indices,
            fallback=overall_corr,
        )
        corr_score = (0.4 * overall_corr) + (0.6 * mouth_corr)
        corr_loss = 1.0 - corr_score

        variance_error = (pred_std - target_std).square()
        variance_loss = (
            variance_error * channel_weights
        ).sum() / channel_weights.sum().clamp_min(1e-6)

        total_loss = (
            l1_loss
            + (self.temporal_weight * temporal_loss)
            + (self.corr_weight * corr_loss)
            + (self.variance_weight * variance_loss)
        )
        return {
            "loss": total_loss,
            "l1_loss": l1_loss,
            "temporal_loss": temporal_loss,
            "corr_loss": corr_loss,
            "var_loss": variance_loss,
            "overall_corr_mean": overall_corr,
            "mouth_jaw_corr_mean": mouth_corr,
        }


def create_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.optim.Optimizer, str]:
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    fused_enabled = bool(args.fused_optimizer and device.type == "cuda")

    if args.optimizer == "adam":
        optimizer_cls = torch.optim.Adam
    elif args.optimizer == "nadam":
        optimizer_cls = torch.optim.NAdam
    elif args.optimizer == "radam":
        optimizer_cls = torch.optim.RAdam
    elif args.optimizer == "adamax":
        optimizer_cls = torch.optim.Adamax
    else:
        optimizer_cls = torch.optim.AdamW

    if fused_enabled and args.optimizer in {"adamw", "adam"}:
        try:
            optimizer = optimizer_cls(model.parameters(), fused=True, **optimizer_kwargs)
            return optimizer, f"{args.optimizer}(fused)"
        except TypeError:
            pass

    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    return optimizer, args.optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_epochs = max(args.epochs, 1)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=args.min_lr,
        )

    warmup_epochs = max(0, min(args.warmup_epochs, total_epochs - 1))
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.min_lr,
    )
    if warmup_epochs == 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(args.warmup_start_factor, 1e-4),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    pred_flat = predictions.reshape(-1, predictions.size(-1)).cpu().numpy()
    target_flat = targets.reshape(-1, targets.size(-1)).cpu().numpy()
    mae_per_channel = np.mean(np.abs(pred_flat - target_flat), axis=0)

    def safe_corr(index: int) -> float:
        if index >= pred_flat.shape[1]:
            return 0.0
        pred_column = pred_flat[:, index]
        target_column = target_flat[:, index]
        if np.std(pred_column) < 1e-8 or np.std(target_column) < 1e-8:
            return 0.0
        corr = np.corrcoef(pred_column, target_column)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    mouth_indices = [idx for idx in MOUTH_AND_JAW_INDICES if idx < mae_per_channel.shape[0]]
    smile_indices = [idx for idx in SMILE_INDICES if idx < mae_per_channel.shape[0]]
    pose_indices = [idx for idx in POSE_INDICES if idx < mae_per_channel.shape[0]]
    blendshape_indices = list(range(min(52, mae_per_channel.shape[0])))

    def mean_corr(indices: List[int]) -> float:
        values = [safe_corr(idx) for idx in indices]
        return float(np.mean(values)) if values else 0.0

    return {
        "overall_mae": float(np.mean(mae_per_channel)),
        "mouth_mae": float(np.mean(mae_per_channel[mouth_indices])) if mouth_indices else 0.0,
        "jaw_open_mae": float(mae_per_channel[JAW_OPEN_INDEX]) if JAW_OPEN_INDEX < mae_per_channel.shape[0] else 0.0,
        "mouth_close_mae": float(mae_per_channel[MOUTH_CLOSE_INDEX]) if MOUTH_CLOSE_INDEX < mae_per_channel.shape[0] else 0.0,
        "smile_mae": float(np.mean(mae_per_channel[smile_indices])) if smile_indices else 0.0,
        "pose_mae": float(np.mean(mae_per_channel[pose_indices])) if pose_indices else 0.0,
        "jaw_open_corr": safe_corr(JAW_OPEN_INDEX),
        "mouth_close_corr": safe_corr(MOUTH_CLOSE_INDEX),
        "smile_corr": mean_corr(smile_indices),
        "mouth_jaw_corr_mean": mean_corr(mouth_indices),
        "overall_blendshape_corr_mean": mean_corr(blendshape_indices),
    }


def make_loader(
    audio: np.ndarray,
    targets: np.ndarray,
    vad: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(audio),
        torch.from_numpy(targets),
        torch.from_numpy(vad),
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)
    return DataLoader(
        **loader_kwargs,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: BalancedRegressionLoss,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    grad_accumulation: int,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    output_space: str,
    pose_dims: int,
    pose_scale: float,
    collect_outputs: bool = False,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_l1 = 0.0
    total_temporal = 0.0
    total_corr = 0.0
    total_var = 0.0
    total_grad_norm = 0.0
    optimizer_steps = 0
    num_batches = 0
    predictions_buffer: List[torch.Tensor] = []
    targets_buffer: List[torch.Tensor] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (audio, targets, _vad) in enumerate(loader, start=1):
        audio = audio.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(audio)
            loss_dict = criterion(predictions, targets)

        if is_train:
            loss_for_backward = loss_dict["loss"] / max(grad_accumulation, 1)
            scaler.scale(loss_for_backward).backward()

            should_step = (batch_idx % max(grad_accumulation, 1) == 0) or (batch_idx == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norm_value = float(grad_norm)
                if not math.isfinite(grad_norm_value):
                    grad_norm_value = 0.0
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total_grad_norm += grad_norm_value
                optimizer_steps += 1

        total_loss += float(loss_dict["loss"].detach().cpu())
        total_l1 += float(loss_dict["l1_loss"].detach().cpu())
        total_temporal += float(loss_dict["temporal_loss"].detach().cpu())
        total_corr += float(loss_dict["corr_loss"].detach().cpu())
        total_var += float(loss_dict["var_loss"].detach().cpu())
        num_batches += 1

        if collect_outputs:
            natural_predictions = restore_output_space(
                predictions.detach(),
                target_mean=target_mean,
                target_std=target_std,
                output_space=output_space,
                pose_dims=pose_dims,
                pose_scale=pose_scale,
            )
            natural_targets = restore_output_space(
                targets.detach(),
                target_mean=target_mean,
                target_std=target_std,
                output_space=output_space,
                pose_dims=pose_dims,
                pose_scale=pose_scale,
            )
            predictions_buffer.append(natural_predictions.cpu())
            targets_buffer.append(natural_targets.cpu())

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "l1_loss": total_l1 / max(num_batches, 1),
        "temporal_loss": total_temporal / max(num_batches, 1),
        "corr_loss": total_corr / max(num_batches, 1),
        "var_loss": total_var / max(num_batches, 1),
    }
    if is_train:
        metrics["grad_norm"] = total_grad_norm / max(optimizer_steps, 1)
    if collect_outputs and predictions_buffer:
        metrics.update(
            compute_regression_metrics(
                predictions=torch.cat(predictions_buffer, dim=0),
                targets=torch.cat(targets_buffer, dim=0),
            )
        )
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
    target_mean: np.ndarray,
    target_std: np.ndarray,
    output_space: str,
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
        "target_normalization": {
            "mode": args.target_normalization,
            "mean": target_mean.squeeze(0).squeeze(0).tolist(),
            "std": target_std.squeeze(0).squeeze(0).tolist(),
        },
        "output_space": output_space,
        "dataset_metadata": metadata,
        "split_info": split_info,
        "target_mode": "blendshapes_only" if args.blendshapes_only else "blendshapes_plus_pose",
        "training_args": vars(args),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "curve_channels": CURVE_DEBUG_CHANNEL_NAMES,
    }
    torch.save(checkpoint, path)


def write_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def plot_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(12, 11))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(epochs, history["train_loss"], label="train_total")
    ax1.plot(epochs, history["val_loss"], label="val_total")
    ax1.plot(epochs, history["val_mouth_mae"], label="val_mouth_mae")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylabel("Loss / MAE")

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(epochs, history["train_corr_loss"], label="train_corr_loss")
    ax2.plot(epochs, history["val_corr_loss"], label="val_corr_loss")
    ax2.plot(epochs, history["train_var_loss"], label="train_var_loss")
    ax2.plot(epochs, history["val_var_loss"], label="val_var_loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylabel("Aux losses")

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(epochs, history["val_jaw_open_corr"], label="jaw_open_corr")
    ax3.plot(epochs, history["val_mouth_close_corr"], label="mouth_close_corr")
    ax3.plot(epochs, history["val_smile_corr"], label="smile_corr")
    ax3.plot(epochs, history["val_mouth_jaw_corr_mean"], label="mouth_jaw_corr_mean")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Correlation")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_summary(path: Path, history: Dict[str, List[float]], best_epoch: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if best_epoch <= 0:
        summary = {"best_epoch": 0, "best_val_loss": None}
    else:
        idx = best_epoch - 1
        summary = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_metrics": {key: values[idx] for key, values in history.items() if values},
        }
    path.write_text(json.dumps(summary, indent=2))


def write_curve_debug(
    plot_path: Path | None,
    sample_path: Path | None,
    checkpoint_path: Path,
    normalized_audio_sample: np.ndarray,
    normalized_target_sample: np.ndarray,
    sample_rate_hz: float,
    device: torch.device,
) -> None:
    if plot_path is None or sample_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_norm = checkpoint.get("target_normalization", {})
    target_mean = torch.tensor(
        np.asarray(target_norm.get("mean", []), dtype=np.float32).reshape(1, 1, -1),
        device=device,
    )
    target_std = torch.tensor(
        np.asarray(target_norm.get("std", []), dtype=np.float32).reshape(1, 1, -1),
        device=device,
    )
    output_space = checkpoint.get("output_space", "natural_range")
    pose_dims = int(checkpoint["model_config"].get("pose_dims", 7))
    pose_scale = float(checkpoint["model_config"].get("pose_scale", 0.2))

    audio_tensor = torch.from_numpy(normalized_audio_sample).to(device)
    target_tensor = torch.from_numpy(normalized_target_sample).to(device)

    with torch.no_grad():
        predictions = model(audio_tensor)
        predictions = restore_output_space(
            predictions,
            target_mean=target_mean,
            target_std=target_std,
            output_space=output_space,
            pose_dims=pose_dims,
            pose_scale=pose_scale,
        )
        targets = restore_output_space(
            target_tensor,
            target_mean=target_mean,
            target_std=target_std,
            output_space=output_space,
            pose_dims=pose_dims,
            pose_scale=pose_scale,
        )

    predictions_np = predictions.squeeze(0).cpu().numpy()
    targets_np = targets.squeeze(0).cpu().numpy()
    timestamps_sec = np.arange(predictions_np.shape[0], dtype=np.float32) / float(sample_rate_hz)

    channels = [
        (name, idx)
        for name, idx in zip(CURVE_DEBUG_CHANNEL_NAMES, CURVE_DEBUG_CHANNEL_INDICES)
        if idx < predictions_np.shape[1]
    ]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    payload = {
        "sample_rate_hz": float(sample_rate_hz),
        "timestamps_sec": timestamps_sec.tolist(),
        "channels": {},
    }

    for axis, (channel_name, channel_idx) in zip(axes, channels):
        axis.plot(timestamps_sec, targets_np[:, channel_idx], label="target", linewidth=1.2)
        axis.plot(timestamps_sec, predictions_np[:, channel_idx], label="prediction", linewidth=1.2)
        axis.set_title(channel_name)
        axis.grid(True, alpha=0.3)
        payload["channels"][channel_name] = {
            "index": int(channel_idx),
            "prediction": predictions_np[:, channel_idx].tolist(),
            "target": targets_np[:, channel_idx].tolist(),
        }

    for axis in axes[len(channels):]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    sample_path.write_text(json.dumps(payload, indent=2))


def load_history(path: Path) -> Dict[str, List[float]] | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

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
    train_targets, val_targets, target_mean, target_std, output_space = normalize_targets(
        train_targets,
        val_targets,
        mode=args.target_normalization,
    )

    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size

    train_loader = make_loader(
        train_audio,
        train_targets,
        train_vad,
        args.batch_size,
        True,
        args.num_workers,
        args.prefetch_factor,
    )
    val_loader = make_loader(
        val_audio,
        val_targets,
        val_vad,
        eval_batch_size,
        False,
        args.num_workers,
        args.prefetch_factor,
    )

    pose_dims = 0 if args.blendshapes_only else 7
    model_config = AudioTransformerConfig(
        input_dim=train_audio.shape[-1],
        output_dim=train_targets.shape[-1],
        variant=args.variant,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        max_seq_len=max(args.max_seq_len, train_audio.shape[1]),
        pose_dims=pose_dims,
        conv_kernel_size=args.conv_kernel_size,
        output_mode=output_space,
        activation_checkpointing=args.activation_checkpointing,
    )
    model = create_model(asdict(model_config)).to(device)

    target_mean_tensor = torch.from_numpy(target_mean).to(device=device, dtype=torch.float32)
    target_std_tensor = torch.from_numpy(target_std).to(device=device, dtype=torch.float32)

    criterion = BalancedRegressionLoss(
        temporal_weight=args.temporal_weight,
        corr_weight=args.corr_weight,
        variance_weight=args.variance_weight,
        mouth_weight_scale=args.mouth_weight_scale,
        blendshape_dims=min(52, train_targets.shape[-1]),
        base_loss=args.base_loss,
        huber_delta=args.huber_delta,
    )
    optimizer, optimizer_name = create_optimizer(model, args, device)
    scheduler = create_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")

    history = {
        "train_loss": [],
        "train_l1_loss": [],
        "train_temporal_loss": [],
        "train_corr_loss": [],
        "train_var_loss": [],
        "train_grad_norm": [],
        "val_loss": [],
        "val_l1_loss": [],
        "val_temporal_loss": [],
        "val_corr_loss": [],
        "val_var_loss": [],
        "val_overall_mae": [],
        "val_mouth_mae": [],
        "val_jaw_open_mae": [],
        "val_mouth_close_mae": [],
        "val_smile_mae": [],
        "val_pose_mae": [],
        "val_jaw_open_corr": [],
        "val_mouth_close_corr": [],
        "val_smile_corr": [],
        "val_mouth_jaw_corr_mean": [],
        "val_overall_blendshape_corr_mean": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    if args.resume_from is not None and args.resume_from.exists():
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        existing_history = load_history(args.history_path)
        if existing_history:
            for key in history:
                if key in existing_history and isinstance(existing_history[key], list):
                    history[key] = existing_history[key]
            if history["val_loss"]:
                best_epoch = int(np.argmin(history["val_loss"])) + 1
        print(f"Resuming from checkpoint: {args.resume_from}", flush=True)
        print(f"Resume epoch: {start_epoch}", flush=True)
        print(f"Best validation loss so far: {best_val_loss:.6f}", flush=True)

    print("=== Audio Transformer Training ===", flush=True)
    print(f"Variant: {args.variant}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Audio shape: {audio.shape}", flush=True)
    print(f"Target shape: {targets.shape}", flush=True)
    print(f"Split mode: {split_info.get('split_mode', 'unknown')}", flush=True)
    print(f"Output space: {output_space}", flush=True)
    print(f"Optimizer: {optimizer_name}", flush=True)
    print(f"Scheduler: {args.scheduler}", flush=True)
    print(f"Grad accumulation: {args.grad_accumulation}", flush=True)
    print(f"Eval batch size: {eval_batch_size}", flush=True)
    print(f"DataLoader workers: {args.num_workers}", flush=True)
    print(f"TF32 enabled: {bool(args.tf32 and device.type == 'cuda')}", flush=True)
    print(f"Model: {json.dumps(model.get_model_info(), indent=2)}", flush=True)
    print(f"Mouth weight scale: {args.mouth_weight_scale}", flush=True)

    final_epoch = args.epochs
    if args.max_run_epochs > 0:
        final_epoch = min(args.epochs, start_epoch + args.max_run_epochs - 1)
    print(f"Run epoch range: {start_epoch} -> {final_epoch}", flush=True)

    for epoch in range(start_epoch, final_epoch + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            grad_accumulation=args.grad_accumulation,
            target_mean=target_mean_tensor,
            target_std=target_std_tensor,
            output_space=output_space,
            pose_dims=pose_dims,
            pose_scale=model.config.pose_scale,
            collect_outputs=False,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            grad_accumulation=1,
            target_mean=target_mean_tensor,
            target_std=target_std_tensor,
            output_space=output_space,
            pose_dims=pose_dims,
            pose_scale=model.config.pose_scale,
            collect_outputs=True,
        )
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_l1_loss"].append(train_metrics["l1_loss"])
        history["train_temporal_loss"].append(train_metrics["temporal_loss"])
        history["train_corr_loss"].append(train_metrics["corr_loss"])
        history["train_var_loss"].append(train_metrics["var_loss"])
        history["train_grad_norm"].append(train_metrics["grad_norm"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_l1_loss"].append(val_metrics["l1_loss"])
        history["val_temporal_loss"].append(val_metrics["temporal_loss"])
        history["val_corr_loss"].append(val_metrics["corr_loss"])
        history["val_var_loss"].append(val_metrics["var_loss"])
        history["val_overall_mae"].append(val_metrics["overall_mae"])
        history["val_mouth_mae"].append(val_metrics["mouth_mae"])
        history["val_jaw_open_mae"].append(val_metrics["jaw_open_mae"])
        history["val_mouth_close_mae"].append(val_metrics["mouth_close_mae"])
        history["val_smile_mae"].append(val_metrics["smile_mae"])
        history["val_pose_mae"].append(val_metrics["pose_mae"])
        history["val_jaw_open_corr"].append(val_metrics["jaw_open_corr"])
        history["val_mouth_close_corr"].append(val_metrics["mouth_close_corr"])
        history["val_smile_corr"].append(val_metrics["smile_corr"])
        history["val_mouth_jaw_corr_mean"].append(val_metrics["mouth_jaw_corr_mean"])
        history["val_overall_blendshape_corr_mean"].append(val_metrics["overall_blendshape_corr_mean"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_mouth_mae={val_metrics['mouth_mae']:.6f} | "
            f"jaw_r={val_metrics['jaw_open_corr']:.4f} | "
            f"smile_r={val_metrics['smile_corr']:.4f} | "
            f"mouth_corr={val_metrics['mouth_jaw_corr_mean']:.4f} | "
            f"corr_loss={val_metrics['corr_loss']:.6f} | "
            f"var_loss={val_metrics['var_loss']:.6f} | "
            f"grad={train_metrics['grad_norm']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}",
            flush=True,
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

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
            target_mean=target_mean,
            target_std=target_std,
            output_space=output_space,
            metadata=metadata,
            split_info=split_info,
            args=args,
        )

        if is_best:
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
                target_mean=target_mean,
                target_std=target_std,
                output_space=output_space,
                metadata=metadata,
                split_info=split_info,
                args=args,
            )
            print(f"  Saved new best checkpoint to {args.checkpoint_path}", flush=True)

        write_history(args.history_path, history)
        plot_history(args.plot_path, history)
        write_summary(args.summary_path, history, best_epoch, best_val_loss)

        if epoch >= max(args.min_epochs, start_epoch) and epoch - best_epoch >= args.patience:
            print(
                f"Early stopping at epoch {epoch} after {args.patience} epochs without improvement.",
                flush=True,
            )
            break

    if best_epoch > 0 and len(val_audio) > 0:
        write_curve_debug(
            plot_path=args.curve_plot_path,
            sample_path=args.curve_sample_path,
            checkpoint_path=args.checkpoint_path,
            normalized_audio_sample=val_audio[:1],
            normalized_target_sample=val_targets[:1],
            sample_rate_hz=float(dataset_info.get("sample_rate_hz", 100)),
            device=device,
        )

    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()

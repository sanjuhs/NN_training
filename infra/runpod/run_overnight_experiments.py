#!/usr/bin/env python3
"""
Run an unattended overnight transformer sweep on RunPod.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentConfig:
    name: str
    variant: str
    d_model: int
    nhead: int
    num_layers: int
    ffn_dim: int
    batch_size: int
    eval_batch_size: int
    lr: float
    dropout: float
    conv_kernel_size: int
    corr_weight: float
    variance_weight: float
    warmup_epochs: int
    notes: str
    optimizer: str = "adamw"
    base_loss: str = "l1"
    huber_delta: float = 0.75
    patience: int = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overnight RunPod experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path("/workspace/NN_training"))
    parser.add_argument("--python-bin", type=Path, default=Path("/workspace/venv/bin/python"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/workspace/v2a_pipeline/datasets/combined_long_10s_step500"),
    )
    parser.add_argument(
        "--sweeps-root",
        type=Path,
        default=Path("/workspace/v2a_pipeline/overnight_sweeps"),
    )
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--pilot-epochs", type=int, default=6)
    parser.add_argument("--full-epochs", type=int, default=90)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=["default", "followon", "third"], default="default")
    return parser.parse_args()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def write_report_header(
    report_path: Path,
    sweep_id: str,
    deadline: datetime,
    dataset_dir: Path,
    baseline_summary_path: Path | None,
) -> None:
    lines = [
        "",
        f"## Overnight Sweep `{sweep_id}`",
        "",
        f"- Start: {isoformat(now_utc())}",
        f"- Deadline: {isoformat(deadline)}",
        f"- Dataset: `{dataset_dir}`",
    ]
    if baseline_summary_path is not None and baseline_summary_path.exists():
        baseline = json.loads(baseline_summary_path.read_text())
        metrics = baseline.get("best_metrics", {})
        lines.extend(
            [
                "- Baseline reference:",
                f"  - best_epoch: `{baseline.get('best_epoch')}`",
                f"  - best_val_loss: `{baseline.get('best_val_loss'):.6f}`",
                f"  - val_mouth_mae: `{metrics.get('val_mouth_mae', 0.0):.6f}`",
                f"  - val_mouth_jaw_corr_mean: `{metrics.get('val_mouth_jaw_corr_mean', 0.0):.4f}`",
                f"  - val_smile_corr: `{metrics.get('val_smile_corr', 0.0):.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Phase | Experiment | Best val_loss | Mouth MAE | Mouth corr | Smile corr | Score | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    append_line(report_path, "\n".join(lines))


def make_run_name(sweep_id: str, config: ExperimentConfig) -> str:
    return f"{sweep_id}__{config.name}"


def score_summary(summary: Dict) -> float:
    metrics = summary.get("best_metrics", {})
    val_loss = float(summary.get("best_val_loss", 1e9))
    mouth_corr = float(metrics.get("val_mouth_jaw_corr_mean", 0.0))
    smile_corr = float(metrics.get("val_smile_corr", 0.0))
    overall_corr = float(metrics.get("val_overall_blendshape_corr_mean", 0.0))
    mouth_mae = float(metrics.get("val_mouth_mae", 0.0))
    return val_loss - (0.05 * mouth_corr) - (0.03 * smile_corr) - (0.02 * overall_corr) + (
        0.02 * mouth_mae
    )


def train_command(
    args: argparse.Namespace,
    sweep_dir: Path,
    run_name: str,
    config: ExperimentConfig,
    max_run_epochs: int,
) -> List[str]:
    run_dir = sweep_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{run_name}_best.pth"
    last_checkpoint_path = run_dir / f"{run_name}_last.pth"
    history_path = run_dir / f"{run_name}_history.json"
    plot_path = run_dir / f"{run_name}_history.png"
    curve_plot_path = run_dir / f"{run_name}_curves.png"
    curve_sample_path = run_dir / f"{run_name}_curves.json"
    summary_path = run_dir / f"{run_name}_summary.json"

    cmd = [
        str(args.python_bin),
        str(args.repo_root / "V2A-over-training-old-nn/2_architecture_training/train_audio_transformer.py"),
        "--data-dir",
        str(args.dataset_dir),
        "--variant",
        config.variant,
        "--epochs",
        str(args.full_epochs),
        "--max-run-epochs",
        str(max_run_epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--batch-size",
        str(config.batch_size),
        "--eval-batch-size",
        str(config.eval_batch_size),
        "--grad-accumulation",
        "1",
        "--optimizer",
        config.optimizer,
        "--scheduler",
        "warmup_cosine",
        "--lr",
        str(config.lr),
        "--min-lr",
        "1e-6",
        "--warmup-epochs",
        str(config.warmup_epochs),
        "--warmup-start-factor",
        "0.1",
        "--weight-decay",
        "1e-4",
        "--fused-optimizer",
        "--grad-clip",
        "1.0",
        "--base-loss",
        config.base_loss,
        "--huber-delta",
        str(config.huber_delta),
        "--temporal-weight",
        "0.05",
        "--corr-weight",
        str(config.corr_weight),
        "--variance-weight",
        str(config.variance_weight),
        "--mouth-weight-scale",
        "1.1",
        "--dropout",
        str(config.dropout),
        "--d-model",
        str(config.d_model),
        "--nhead",
        str(config.nhead),
        "--num-layers",
        str(config.num_layers),
        "--ffn-dim",
        str(config.ffn_dim),
        "--conv-kernel-size",
        str(config.conv_kernel_size),
        "--target-normalization",
        "standardize",
        "--segment-aware-split",
        "--val-fraction",
        str(args.val_fraction),
        "--patience",
        str(config.patience),
        "--num-workers",
        "8",
        "--prefetch-factor",
        "4",
        "--tf32",
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--checkpoint-path",
        str(checkpoint_path),
        "--last-checkpoint-path",
        str(last_checkpoint_path),
        "--history-path",
        str(history_path),
        "--plot-path",
        str(plot_path),
        "--curve-plot-path",
        str(curve_plot_path),
        "--curve-sample-path",
        str(curve_sample_path),
        "--summary-path",
        str(summary_path),
    ]
    if last_checkpoint_path.exists():
        cmd.extend(["--resume-from", str(last_checkpoint_path)])
    return cmd


def export_command(args: argparse.Namespace, sweep_dir: Path, run_name: str) -> List[str]:
    run_dir = sweep_dir / run_name
    return [
        str(args.python_bin),
        str(args.repo_root / "V2A-over-training-old-nn/2_architecture_training/export_audio_transformer_onnx.py"),
        "--checkpoint",
        str(run_dir / f"{run_name}_best.pth"),
        "--output",
        str(run_dir / f"{run_name}.onnx"),
        "--manifest",
        str(run_dir / f"{run_name}.json"),
        "--seq-len",
        "1000",
    ]


def run_logged_command(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"[exit_code] {return_code}\n")
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)


def read_summary(summary_path: Path) -> Dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text())


def append_result_row(
    report_path: Path,
    phase: str,
    run_name: str,
    summary: Dict,
    config: ExperimentConfig,
) -> None:
    metrics = summary.get("best_metrics", {})
    row = (
        f"| {phase} | `{run_name}` | "
        f"{summary.get('best_val_loss', 0.0):.6f} | "
        f"{metrics.get('val_mouth_mae', 0.0):.6f} | "
        f"{metrics.get('val_mouth_jaw_corr_mean', 0.0):.4f} | "
        f"{metrics.get('val_smile_corr', 0.0):.4f} | "
        f"{score_summary(summary):.6f} | "
        f"{config.notes} |"
    )
    append_line(report_path, row)


def write_leaderboard(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def remaining_hours(deadline: datetime) -> float:
    return max((deadline - now_utc()).total_seconds() / 3600.0, 0.0)


def get_experiment_configs(preset: str) -> List[ExperimentConfig]:
    if preset == "third":
        return [
            ExperimentConfig(
                name="conformer_d384_l12_b12_huber",
                variant="conformer_transformer",
                d_model=384,
                nhead=8,
                num_layers=12,
                ffn_dim=1536,
                batch_size=12,
                eval_batch_size=4,
                lr=1.5e-4,
                dropout=0.08,
                conv_kernel_size=15,
                corr_weight=0.22,
                variance_weight=0.06,
                warmup_epochs=8,
                optimizer="adamw",
                base_loss="huber",
                huber_delta=0.75,
                patience=24,
                notes="Speech-oriented Conformer-style stack with Huber loss.",
            ),
            ExperimentConfig(
                name="conformer_d512_l16_b8_nadam_huber",
                variant="conformer_transformer",
                d_model=512,
                nhead=8,
                num_layers=16,
                ffn_dim=2048,
                batch_size=8,
                eval_batch_size=2,
                lr=1.2e-4,
                dropout=0.08,
                conv_kernel_size=15,
                corr_weight=0.24,
                variance_weight=0.06,
                warmup_epochs=10,
                optimizer="nadam",
                base_loss="huber",
                huber_delta=0.75,
                patience=28,
                notes="Larger Conformer-style run with NAdam.",
            ),
            ExperimentConfig(
                name="multiscale_d384_l12_b12_huber",
                variant="multiscale_transformer",
                d_model=384,
                nhead=8,
                num_layers=12,
                ffn_dim=1536,
                batch_size=12,
                eval_batch_size=4,
                lr=1.5e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.22,
                variance_weight=0.06,
                warmup_epochs=8,
                optimizer="adamw",
                base_loss="huber",
                huber_delta=0.75,
                patience=24,
                notes="Multi-scale local/global transformer fusion.",
            ),
            ExperimentConfig(
                name="multiscale_d512_l16_b8_radam_huber",
                variant="multiscale_transformer",
                d_model=512,
                nhead=8,
                num_layers=16,
                ffn_dim=2048,
                batch_size=8,
                eval_batch_size=2,
                lr=1.25e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.24,
                variance_weight=0.06,
                warmup_epochs=10,
                optimizer="radam",
                base_loss="huber",
                huber_delta=0.75,
                patience=28,
                notes="Larger multi-scale fusion model with RAdam.",
            ),
            ExperimentConfig(
                name="multiscale_d640_l18_b6_nadam_huber",
                variant="multiscale_transformer",
                d_model=640,
                nhead=10,
                num_layers=18,
                ffn_dim=2560,
                batch_size=6,
                eval_batch_size=2,
                lr=1.0e-4,
                dropout=0.08,
                conv_kernel_size=15,
                corr_weight=0.24,
                variance_weight=0.06,
                warmup_epochs=12,
                optimizer="nadam",
                base_loss="huber",
                huber_delta=0.75,
                patience=30,
                notes="Heavy multi-scale long-context run to use the remaining night budget.",
            ),
        ]

    if preset == "followon":
        return [
            ExperimentConfig(
                name="conv_d320_l12_b20_nadam_huber",
                variant="conv_transformer",
                d_model=320,
                nhead=8,
                num_layers=12,
                ffn_dim=1280,
                batch_size=20,
                eval_batch_size=4,
                lr=1.6e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.2,
                variance_weight=0.06,
                warmup_epochs=6,
                optimizer="nadam",
                base_loss="huber",
                huber_delta=0.75,
                notes="Large-batch control with NAdam and Huber.",
            ),
            ExperimentConfig(
                name="convgated_d512_l16_b10_nadam_huber",
                variant="conv_gated_transformer",
                d_model=512,
                nhead=8,
                num_layers=16,
                ffn_dim=2048,
                batch_size=10,
                eval_batch_size=4,
                lr=1.2e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.22,
                variance_weight=0.06,
                warmup_epochs=8,
                optimizer="nadam",
                base_loss="huber",
                huber_delta=0.75,
                notes="Hybrid conv+gated model with NAdam and Huber.",
            ),
            ExperimentConfig(
                name="conv_d512_l16_b10_radam_huber",
                variant="conv_transformer",
                d_model=512,
                nhead=8,
                num_layers=16,
                ffn_dim=2048,
                batch_size=10,
                eval_batch_size=4,
                lr=1.3e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.22,
                variance_weight=0.06,
                warmup_epochs=8,
                optimizer="radam",
                base_loss="huber",
                huber_delta=0.75,
                notes="Wider conv model with RAdam and Huber.",
            ),
            ExperimentConfig(
                name="gated_d384_l12_b16_nadam",
                variant="gated_transformer",
                d_model=384,
                nhead=8,
                num_layers=12,
                ffn_dim=1536,
                batch_size=16,
                eval_batch_size=4,
                lr=1.6e-4,
                dropout=0.08,
                conv_kernel_size=9,
                corr_weight=0.2,
                variance_weight=0.06,
                warmup_epochs=6,
                optimizer="nadam",
                base_loss="l1",
                notes="Pure gated FFN transformer with NAdam.",
            ),
        ]

    return [
        ExperimentConfig(
            name="conv_d320_l12_b16",
            variant="conv_transformer",
            d_model=320,
            nhead=8,
            num_layers=12,
            ffn_dim=1280,
            batch_size=16,
            eval_batch_size=4,
            lr=1.8e-4,
            dropout=0.08,
            conv_kernel_size=9,
            corr_weight=0.18,
            variance_weight=0.06,
            warmup_epochs=6,
            notes="Control rerun with warmup and larger batch.",
        ),
        ExperimentConfig(
            name="conv_d512_l16_b10",
            variant="conv_transformer",
            d_model=512,
            nhead=8,
            num_layers=16,
            ffn_dim=2048,
            batch_size=10,
            eval_batch_size=4,
            lr=1.4e-4,
            dropout=0.08,
            conv_kernel_size=9,
            corr_weight=0.18,
            variance_weight=0.06,
            warmup_epochs=8,
            notes="Wider/deeper conv transformer.",
        ),
        ExperimentConfig(
            name="convgated_d512_l16_b10",
            variant="conv_gated_transformer",
            d_model=512,
            nhead=8,
            num_layers=16,
            ffn_dim=2048,
            batch_size=10,
            eval_batch_size=4,
            lr=1.3e-4,
            dropout=0.08,
            conv_kernel_size=9,
            corr_weight=0.2,
            variance_weight=0.06,
            warmup_epochs=8,
            notes="Conv mixer plus gated FFN hybrid.",
        ),
        ExperimentConfig(
            name="conv_d640_l18_k15_b8",
            variant="conv_transformer",
            d_model=640,
            nhead=10,
            num_layers=18,
            ffn_dim=2560,
            batch_size=8,
            eval_batch_size=2,
            lr=1.1e-4,
            dropout=0.08,
            conv_kernel_size=15,
            corr_weight=0.2,
            variance_weight=0.06,
            warmup_epochs=8,
            notes="Deeper model with wider local kernel.",
        ),
    ]


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_dir}")

    started_at = now_utc()
    deadline = started_at + timedelta(hours=args.hours)
    sweep_id = started_at.strftime("overnight_%Y%m%d_%H%M%S")
    sweep_dir = args.sweeps_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.repo_root / "RUNPOD_OVERNIGHT_EXPERIMENTS.md"
    leaderboard_path = sweep_dir / "leaderboard.json"
    main_log_path = sweep_dir / "runner.log"
    baseline_summary_path = (
        Path("/workspace/v2a_pipeline/runs/conv_transformer_d320_l12_corrvar_std")
        / "conv_transformer_d320_l12_corrvar_std_summary.json"
    )

    write_report_header(
        report_path=report_path,
        sweep_id=sweep_id,
        deadline=deadline,
        dataset_dir=args.dataset_dir,
        baseline_summary_path=baseline_summary_path if baseline_summary_path.exists() else None,
    )

    configs = get_experiment_configs(args.preset)

    pilot_results: List[Dict] = []
    with open(main_log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"Starting sweep {sweep_id} at {isoformat(started_at)}\n")
        log_file.flush()

    for config in configs:
        if remaining_hours(deadline) <= 0.5:
            append_line(report_path, f"\nStopped before pilot `{config.name}` because the time budget was exhausted.")
            break

        run_name = make_run_name(sweep_id, config)
        pilot_log = sweep_dir / run_name / "pilot.log"
        append_line(report_path, f"\nPilot starting for `{run_name}` at {isoformat(now_utc())}.")
        cmd = train_command(
            args=args,
            sweep_dir=sweep_dir,
            run_name=run_name,
            config=config,
            max_run_epochs=args.pilot_epochs,
        )
        run_logged_command(cmd, cwd=args.repo_root, log_path=pilot_log)
        summary = read_summary(sweep_dir / run_name / f"{run_name}_summary.json")
        append_result_row(report_path, "pilot", run_name, summary, config)
        pilot_results.append(
            {
                "run_name": run_name,
                "config": asdict(config),
                "phase": "pilot",
                "summary": summary,
                "score": score_summary(summary),
            }
        )
        write_leaderboard(leaderboard_path, {"pilot_results": pilot_results})

    if not pilot_results:
        append_line(report_path, "\nNo pilot runs completed.")
        return

    finalists = sorted(pilot_results, key=lambda item: item["score"])[: max(args.finalists, 1)]
    finalist_names = ", ".join(f"`{item['run_name']}`" for item in finalists)
    append_line(report_path, f"\nFinalists selected: {finalist_names}")

    full_results: List[Dict] = []
    for finalist in finalists:
        if remaining_hours(deadline) <= 1.0:
            append_line(
                report_path,
                f"\nSkipping full continuation for `{finalist['run_name']}` because less than 1 hour remained.",
            )
            continue

        config = ExperimentConfig(**finalist["config"])
        run_name = finalist["run_name"]
        full_log = sweep_dir / run_name / "full.log"
        append_line(report_path, f"\nFull continuation starting for `{run_name}` at {isoformat(now_utc())}.")
        train_cmd = train_command(
            args=args,
            sweep_dir=sweep_dir,
            run_name=run_name,
            config=config,
            max_run_epochs=0,
        )
        run_logged_command(train_cmd, cwd=args.repo_root, log_path=full_log)

        export_log = sweep_dir / run_name / "export.log"
        run_logged_command(export_command(args, sweep_dir, run_name), cwd=args.repo_root, log_path=export_log)

        summary = read_summary(sweep_dir / run_name / f"{run_name}_summary.json")
        append_result_row(report_path, "full", run_name, summary, config)
        full_results.append(
            {
                "run_name": run_name,
                "config": asdict(config),
                "phase": "full",
                "summary": summary,
                "score": score_summary(summary),
            }
        )
        write_leaderboard(
            leaderboard_path,
            {
                "pilot_results": pilot_results,
                "full_results": full_results,
                "best_so_far": min(full_results, key=lambda item: item["score"]) if full_results else None,
            },
        )

    best_candidates = full_results if full_results else pilot_results
    best_result = min(best_candidates, key=lambda item: item["score"])
    best_metrics = best_result["summary"].get("best_metrics", {})
    append_line(report_path, "\n### Best Result")
    append_line(report_path, f"- Run: `{best_result['run_name']}`")
    append_line(report_path, f"- Phase reached: `{best_result['phase']}`")
    append_line(report_path, f"- best_val_loss: `{best_result['summary'].get('best_val_loss', 0.0):.6f}`")
    append_line(report_path, f"- val_mouth_mae: `{best_metrics.get('val_mouth_mae', 0.0):.6f}`")
    append_line(report_path, f"- val_mouth_jaw_corr_mean: `{best_metrics.get('val_mouth_jaw_corr_mean', 0.0):.4f}`")
    append_line(report_path, f"- val_smile_corr: `{best_metrics.get('val_smile_corr', 0.0):.4f}`")
    append_line(report_path, f"- End: {isoformat(now_utc())}")


if __name__ == "__main__":
    main()

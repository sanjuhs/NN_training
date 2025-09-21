#!/usr/bin/env python3
"""
Generate simple architecture diagrams for the V2A project as PNGs.

Outputs:
  - diagrams/tcn_blocks.png
  - diagrams/encoder_decoder.png

These are schematic, high-level visualizations, not exact graphs.
"""
import os
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).parent / "diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_tcn_blocks(path: Path):
    """Draw a high-level Dilated TCN stack diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axis_off()

    # Background blocks
    ax.add_patch(plt.Rectangle((0.05, 0.15), 0.9, 0.65, color=(1.0, 0.84, 0.73), alpha=0.6, ec=(0.9, 0.6, 0.4)))
    ax.text(0.06, 0.78, "Dilated TCN Blocks", fontsize=12, weight="bold")

    # Input row
    ax.text(0.02, 0.12, "Input: X", fontsize=11, color=(0.1, 0.4, 0.9), weight="bold")
    for i in range(12):
        ax.add_patch(plt.Circle((0.08 + i * 0.07, 0.18), 0.015, color=(0.2, 0.5, 0.95)))

    # Three blocks with increasing dilation annotations
    block_y = [0.25, 0.42, 0.59]
    labels = ["Block 1\nd={1,2,4}", "Block 2\nd={1,2,4}", "Block 3\nd={1,2,4}"]
    for y, lab in zip(block_y, labels):
        ax.add_patch(plt.Rectangle((0.1, y), 0.8, 0.12, color=(1.0, 0.9, 0.8), ec=(0.95, 0.75, 0.55)))
        ax.text(0.5, y + 0.06, lab, ha="center", va="center", fontsize=11)

    # Skip connections (red dashed)
    for y in block_y:
        ax.plot([0.1, 0.9], [y + 0.12, y + 0.12], ls="--", color="red", lw=1, alpha=0.6)

    # Output row
    ax.text(0.02, 0.84, "Predict: Y", fontsize=11, color=(0.1, 0.4, 0.9), weight="bold")
    for i in range(12):
        ax.add_patch(plt.Circle((0.08 + i * 0.07, 0.82), 0.015, color=(0.2, 0.7, 0.2)))

    ax.text(0.92, 0.84, "Softmax/Activation", fontsize=9, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_encoder_decoder(path: Path):
    """Draw a high-level U-Net-like encoder/decoder schematic."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()

    # Encoder background
    ax.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.36, color=(1.0, 0.8, 0.8), alpha=0.6, ec=(0.95, 0.6, 0.6)))
    ax.text(0.06, 0.41, "Encoder", fontsize=12, weight="bold")

    # Decoder background
    ax.add_patch(plt.Rectangle((0.05, 0.48), 0.9, 0.42, color=(0.8, 0.9, 1.0), alpha=0.6, ec=(0.55, 0.7, 0.95)))
    ax.text(0.06, 0.88, "Decoder", fontsize=12, weight="bold")

    # Input row
    ax.text(0.02, 0.06, "Input: X", fontsize=11, color=(0.1, 0.4, 0.9), weight="bold")
    for i in range(12):
        ax.add_patch(plt.Circle((0.08 + i * 0.07, 0.1), 0.013, color=(0.2, 0.5, 0.95)))

    # Encoder levels
    enc_levels = [0.18, 0.28, 0.38]
    for idx, y in enumerate(enc_levels):
        for i in range(6 + idx * 2):
            ax.add_patch(plt.Circle((0.2 + i * 0.07, y), 0.012, color=(0.4, 0.75, 0.3)))
        ax.text(0.12, y, f"E({idx+1}) Conv", fontsize=9, va="center", color=(0.4, 0.6, 0.2))

    # Bottleneck
    ax.add_patch(plt.Rectangle((0.45, 0.44), 0.1, 0.04, color=(0.7, 0.9, 0.5)))

    # Decoder levels
    dec_levels = [0.58, 0.7, 0.82]
    for idx, y in enumerate(dec_levels):
        for i in range(6 + (len(dec_levels)-idx-1) * 2):
            ax.add_patch(plt.Circle((0.2 + i * 0.07, y), 0.012, color=(1.0, 0.55, 0.15)))
        ax.text(0.12, y, f"D({idx+1}) Conv", fontsize=9, va="center", color=(0.9, 0.5, 0.2))

    # Skip arrows
    arrow_kw = dict(length_includes_head=True, head_width=0.012, head_length=0.02, color=(0.3, 0.5, 0.9), lw=1.5)
    for e_y, d_y in zip(enc_levels[::-1], dec_levels):
        ax.arrow(0.86, e_y + 0.01, 0.0, d_y - e_y - 0.02, **arrow_kw)

    # Output row
    ax.text(0.02, 0.93, "Predict: Y", fontsize=11, color=(0.1, 0.4, 0.9), weight="bold")
    for i in range(12):
        ax.add_patch(plt.Circle((0.08 + i * 0.07, 0.9), 0.013, color=(0.2, 0.7, 0.2)))
    ax.text(0.92, 0.93, "Softmax/Activation", fontsize=9, ha="right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    draw_tcn_blocks(OUTPUT_DIR / "tcn_blocks.png")
    draw_encoder_decoder(OUTPUT_DIR / "encoder_decoder.png")
    print(f"Saved diagrams to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()



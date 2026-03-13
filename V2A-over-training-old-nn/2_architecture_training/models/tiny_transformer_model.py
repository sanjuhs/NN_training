#!/usr/bin/env python3
"""
Tiny transformer encoder for audio-to-blendshape overfit experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TinyTransformerConfig:
    input_dim: int = 80
    output_dim: int = 59
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 1200
    pose_dims: int = 7
    pose_scale: float = 0.2


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed positional encoding sized for the 10-second overfit windows."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TinyAudioTransformer(nn.Module):
    """
    Small transformer encoder that preserves the current repo I/O contract.
    """

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply_output_constraints(self, output: torch.Tensor) -> torch.Tensor:
        if self.config.output_dim <= 52 or self.config.pose_dims <= 0:
            return torch.sigmoid(output)

        blendshape_dims = self.config.output_dim - self.config.pose_dims
        blendshapes = torch.sigmoid(output[..., :blendshape_dims])
        pose = torch.tanh(output[..., blendshape_dims:]) * self.config.pose_scale
        return torch.cat([blendshapes, pose], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input rank 3, got shape {tuple(x.shape)}")

        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.output_head(x)
        return self._apply_output_constraints(x)

    def get_model_info(self) -> Dict[str, Any]:
        model_size_mb = self.num_parameters * 4 / (1024 * 1024)
        info = asdict(self.config)
        info.update(
            {
                "architecture": "tiny_transformer_encoder",
                "num_parameters": self.num_parameters,
                "model_size_mb": model_size_mb,
            }
        )
        return info


def create_model(config: Optional[Dict[str, Any]] = None) -> TinyAudioTransformer:
    if config is None:
        model_config = TinyTransformerConfig()
    elif isinstance(config, TinyTransformerConfig):
        model_config = config
    else:
        valid_keys = {field.name for field in fields(TinyTransformerConfig)}
        filtered_config = {key: value for key, value in config.items() if key in valid_keys}
        model_config = TinyTransformerConfig(**filtered_config)
    return TinyAudioTransformer(model_config)

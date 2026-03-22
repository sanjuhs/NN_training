#!/usr/bin/env python3
"""
Configurable transformer variants for audio-to-blendshape regression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils


@dataclass
class AudioTransformerConfig:
    input_dim: int = 80
    output_dim: int = 59
    variant: str = "baseline"
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 1200
    pose_dims: int = 7
    pose_scale: float = 0.2
    conv_kernel_size: int = 9
    output_mode: str = "natural_range"
    activation_checkpointing: bool = False


def clamp_output_to_natural_range(
    output: torch.Tensor,
    pose_dims: int,
    pose_scale: float,
) -> torch.Tensor:
    if pose_dims <= 0 or output.size(-1) <= pose_dims:
        return torch.clamp(output, 0.0, 1.0)

    blendshape_dims = output.size(-1) - pose_dims
    blendshapes = torch.clamp(output[..., :blendshape_dims], 0.0, 1.0)
    pose = torch.clamp(output[..., blendshape_dims:], -pose_scale, pose_scale)
    return torch.cat([blendshapes, pose], dim=-1)


class SinusoidalPositionalEncoding(nn.Module):
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


class StandardFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, dim_feedforward * 2)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)
        return self.out(gated)


class LocalConvMixer(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = self.activation(y)
        y = self.pointwise(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = torch.nn.functional.glu(self.pointwise_in(y), dim=1)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class BaseTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        use_conv: bool,
        use_gated_ffn: bool,
        conv_kernel_size: int,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = LocalConvMixer(d_model, conv_kernel_size, dropout) if use_conv else None
        self.ffn_norm = nn.LayerNorm(d_model)
        if use_gated_ffn:
            self.ffn = GatedFeedForward(d_model, dim_feedforward, dropout)
        else:
            self.ffn = StandardFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)
        if self.conv is not None:
            x = x + self.conv(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        conv_kernel_size: int,
    ) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = StandardFeedForward(d_model, dim_feedforward, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = StandardFeedForward(d_model, dim_feedforward, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (0.5 * self.ffn1(self.ffn1_norm(x)))
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)
        x = x + self.conv(x)
        x = x + (0.5 * self.ffn2(self.ffn2_norm(x)))
        return self.output_norm(x)


class AudioTransformer(nn.Module):
    def __init__(self, config: AudioTransformerConfig) -> None:
        super().__init__()
        self.config = config

        use_conv = config.variant in {"conv_transformer", "conv_gated_transformer", "multiscale_transformer"}
        use_gated_ffn = config.variant in {"gated_transformer", "conv_gated_transformer"}
        if config.variant not in {
            "baseline",
            "conv_transformer",
            "gated_transformer",
            "conv_gated_transformer",
            "conformer_transformer",
            "multiscale_transformer",
        }:
            raise ValueError(f"Unsupported transformer variant: {config.variant}")

        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )
        self.input_norm = nn.LayerNorm(config.d_model)
        if config.variant == "conformer_transformer":
            self.blocks = nn.ModuleList(
                [
                    ConformerBlock(
                        d_model=config.d_model,
                        nhead=config.nhead,
                        dim_feedforward=config.dim_feedforward,
                        dropout=config.dropout,
                        conv_kernel_size=config.conv_kernel_size,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.global_blocks = None
            self.global_downsample = None
            self.global_projection = None
            self.fusion_gate = None
        elif config.variant == "multiscale_transformer":
            self.blocks = nn.ModuleList(
                [
                    BaseTransformerBlock(
                        d_model=config.d_model,
                        nhead=config.nhead,
                        dim_feedforward=config.dim_feedforward,
                        dropout=config.dropout,
                        use_conv=True,
                        use_gated_ffn=False,
                        conv_kernel_size=config.conv_kernel_size,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.global_blocks = nn.ModuleList(
                [
                    BaseTransformerBlock(
                        d_model=config.d_model,
                        nhead=config.nhead,
                        dim_feedforward=config.dim_feedforward,
                        dropout=config.dropout,
                        use_conv=False,
                        use_gated_ffn=True,
                        conv_kernel_size=config.conv_kernel_size,
                    )
                    for _ in range(max(1, config.num_layers // 2))
                ]
            )
            self.global_downsample = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
            self.global_projection = nn.Linear(config.d_model, config.d_model)
            self.fusion_gate = nn.Linear(config.d_model * 2, config.d_model)
        else:
            self.blocks = nn.ModuleList(
                [
                    BaseTransformerBlock(
                        d_model=config.d_model,
                        nhead=config.nhead,
                        dim_feedforward=config.dim_feedforward,
                        dropout=config.dropout,
                        use_conv=use_conv,
                        use_gated_ffn=use_gated_ffn,
                        conv_kernel_size=config.conv_kernel_size,
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.global_blocks = None
            self.global_downsample = None
            self.global_projection = None
            self.fusion_gate = None
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
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply_output_constraints(self, output: torch.Tensor) -> torch.Tensor:
        if self.config.output_mode == "standardized":
            return output
        if self.config.output_dim <= 52 or self.config.pose_dims <= 0:
            return torch.sigmoid(output)

        blendshape_dims = self.config.output_dim - self.config.pose_dims
        blendshapes = torch.sigmoid(output[..., :blendshape_dims])
        pose = torch.tanh(output[..., blendshape_dims:]) * self.config.pose_scale
        return torch.cat([blendshapes, pose], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        if self.config.variant == "multiscale_transformer":
            local = x
            for block in self.blocks:
                if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                    local = checkpoint_utils.checkpoint(block, local, use_reentrant=False)
                else:
                    local = block(local)

            pooled = self.global_downsample(x.transpose(1, 2)).transpose(1, 2)
            pooled = self.positional_encoding(pooled)
            for block in self.global_blocks:
                if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                    pooled = checkpoint_utils.checkpoint(block, pooled, use_reentrant=False)
                else:
                    pooled = block(pooled)
            pooled = self.global_projection(pooled)
            pooled = torch.repeat_interleave(pooled, repeats=2, dim=1)
            pooled = pooled[:, : local.size(1)]
            if pooled.size(1) < local.size(1):
                pad_len = local.size(1) - pooled.size(1)
                pooled = torch.cat([pooled, pooled[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([local, pooled], dim=-1)))
            x = local + (gate * pooled)
        else:
            for block in self.blocks:
                if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                    x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        x = self.output_head(x)
        return self._apply_output_constraints(x)

    def get_model_info(self) -> Dict[str, Any]:
        model_size_mb = self.num_parameters * 4 / (1024 * 1024)
        info = asdict(self.config)
        info.update(
            {
                "architecture": "audio_transformer_variants",
                "num_parameters": self.num_parameters,
                "model_size_mb": model_size_mb,
            }
        )
        return info


def create_model(config: Optional[Dict[str, Any]] = None) -> AudioTransformer:
    if config is None:
        model_config = AudioTransformerConfig()
    elif isinstance(config, AudioTransformerConfig):
        model_config = config
    else:
        valid_keys = {field.name for field in fields(AudioTransformerConfig)}
        filtered_config = {key: value for key, value in config.items() if key in valid_keys}
        model_config = AudioTransformerConfig(**filtered_config)
    return AudioTransformer(model_config)

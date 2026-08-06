from __future__ import annotations

"""Modular continuous-price forecaster for temporal/graph fault isolation.

The model deliberately separates four concerns:

1. continuous OHLCV representation;
2. within-asset temporal backbone (Transformer or official ModernTCN);
3. optional explicit cross-asset graph/spatial mixing;
4. a direct five-horizon continuous Close head.

Before the graph stage, assets are independent: the asset axis is folded into
the batch dimension by both temporal backbones.  No node embedding is used.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.models.dynamic_graph.contracts import (
    GraphConfig,
    GraphOutput,
    TemporalConfig,
)
from src.models.dynamic_graph.graph_learners import build_graph_learner
from src.models.dynamic_graph.modules import (
    IdentitySpatialModule,
    PerNodeTransformerEncoder,
    SpatialMessagePassing,
)
from src.models.modern_tcn import _TemporalEncodingModernTCNAdapter


TemporalBackboneType = Literal["transformer", "modern_tcn"]
OutputRepresentation = Literal[
    "normalised_close",
    "cumulative_log_change",
]
OutputHeadInitialisation = Literal["default", "zero"]
SpatialGateType = Literal["none", "fixed", "learned_scalar"]


@dataclass(frozen=True)
class ContinuousTemporalConfig:
    type: TemporalBackboneType = "transformer"
    d_model: int = 64

    # Transformer settings.
    num_layers: int = 1
    num_heads: int = 4
    feedforward_multiplier: int = 2
    dropout: float = 0.0
    relative_position_embedding: bool = True

    # Shared absolute session-position descriptor.
    session_position_encoding: bool = True

    # ModernTCN settings.  These match the selected per-asset OHLCV
    # baseline unless explicitly overridden.
    patch_size: int = 8
    patch_stride: int = 4
    modern_tcn_ffn_ratio: int = 1
    modern_tcn_num_blocks: int = 1
    modern_tcn_large_kernel: int = 51
    modern_tcn_small_kernel: int = 5
    modern_tcn_dropout: float = 0.05
    modern_tcn_head_dropout: float = 0.0

    def validate(self, *, context_length: int) -> None:
        if self.type not in {"transformer", "modern_tcn"}:
            raise ValueError(
                "temporal.type must be 'transformer' or 'modern_tcn'."
            )
        if self.d_model <= 0:
            raise ValueError("temporal.d_model must be positive.")
        if self.num_layers <= 0:
            raise ValueError("temporal.num_layers must be positive.")
        if self.num_heads <= 0:
            raise ValueError("temporal.num_heads must be positive.")
        if self.feedforward_multiplier <= 0:
            raise ValueError(
                "temporal.feedforward_multiplier must be positive."
            )
        if self.type == "transformer" and self.d_model % self.num_heads:
            raise ValueError(
                "Transformer d_model must be divisible by num_heads."
            )
        for name, value in {
            "dropout": self.dropout,
            "modern_tcn_dropout": self.modern_tcn_dropout,
            "modern_tcn_head_dropout": self.modern_tcn_head_dropout,
        }.items():
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"temporal.{name} must lie in [0,1).")
        if self.patch_size <= 0 or self.patch_stride <= 0:
            raise ValueError(
                "ModernTCN patch_size and patch_stride must be positive."
            )
        if self.patch_size < self.patch_stride:
            raise ValueError(
                "ModernTCN patch_size must be >= patch_stride."
            )
        if context_length % self.patch_stride != 0:
            raise ValueError(
                "context_length must be divisible by ModernTCN "
                "patch_stride for the selected one-stage contract."
            )
        if self.modern_tcn_ffn_ratio <= 0:
            raise ValueError("modern_tcn_ffn_ratio must be positive.")
        if self.modern_tcn_num_blocks <= 0:
            raise ValueError("modern_tcn_num_blocks must be positive.")


@dataclass(frozen=True)
class ContinuousForecasterConfig:
    num_nodes: int = 93
    context_length: int = 60
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    input_channels: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    target_channel: str = "close"
    output_representation: OutputRepresentation = "normalised_close"
    output_head_initialisation: OutputHeadInitialisation = "default"
    temporal: ContinuousTemporalConfig = ContinuousTemporalConfig()
    graph: GraphConfig = GraphConfig(type="none", num_heads=2)

    # Deterministic context-window absolute Close-return correlation graph.
    # ``None`` retains every non-self absolute correlation.
    dynamic_correlation_threshold: float | None = None
    dynamic_correlation_empty_row_policy: Literal[
        "error",
        "strongest",
    ] = "strongest"
    dynamic_correlation_eps: float = 1.0e-8

    spatial_num_layers: int = 1
    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0
    spatial_gate_type: SpatialGateType = "none"
    spatial_gate_initial_beta: float = 1.0
    head_dropout: float = 0.0

    def validate(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("horizons must be strictly increasing.")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique.")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Every horizon must be positive.")
        if not self.input_channels:
            raise ValueError("input_channels must not be empty.")
        if self.target_channel not in self.input_channels:
            raise ValueError(
                "target_channel must be present in input_channels."
            )
        if self.output_representation not in {
            "normalised_close",
            "cumulative_log_change",
        }:
            raise ValueError(
                "output_representation must be 'normalised_close' or "
                "'cumulative_log_change'."
            )
        if self.output_head_initialisation not in {"default", "zero"}:
            raise ValueError(
                "output_head_initialisation must be 'default' or 'zero'."
            )
        self.temporal.validate(context_length=self.context_length)
        self.graph.validate(
            num_nodes=self.num_nodes,
            d_model=self.temporal.d_model,
        )
        if self.graph.type not in {
            "none",
            "fixed",
            "free_static",
            "dynamic",
            "dynamic_correlation",
            "dynamic_base",
        }:
            raise ValueError(
                "Continuous forecasting supports graph.type in "
                "{'none','fixed','free_static','dynamic',"
                "'dynamic_correlation','dynamic_base'} only."
            )
        if self.graph.type == "dynamic_correlation":
            if self.graph.activation != "softmax":
                raise ValueError(
                    "dynamic_correlation requires graph.activation='softmax' "
                    "because it uses direct row normalisation."
                )
            if (
                self.dynamic_correlation_threshold is not None
                and not 0.0
                <= float(self.dynamic_correlation_threshold)
                <= 1.0
            ):
                raise ValueError(
                    "dynamic_correlation_threshold must be None or lie in "
                    "[0,1]."
                )
            if self.dynamic_correlation_empty_row_policy not in {
                "error",
                "strongest",
            }:
                raise ValueError(
                    "dynamic_correlation_empty_row_policy must be 'error' "
                    "or 'strongest'."
                )
            if (
                not math.isfinite(float(self.dynamic_correlation_eps))
                or float(self.dynamic_correlation_eps) <= 0.0
            ):
                raise ValueError(
                    "dynamic_correlation_eps must be finite and positive."
                )

        if self.spatial_num_layers <= 0:
            raise ValueError("spatial_num_layers must be positive.")
        if self.spatial_feedforward_multiplier <= 0:
            raise ValueError(
                "spatial_feedforward_multiplier must be positive."
            )
        if not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError("spatial_dropout must lie in [0,1).")
        if self.spatial_gate_type not in {
            "none",
            "fixed",
            "learned_scalar",
        }:
            raise ValueError(
                "spatial_gate_type must be 'none', 'fixed', or "
                "'learned_scalar'."
            )
        if not 0.0 <= float(self.spatial_gate_initial_beta) <= 1.0:
            raise ValueError(
                "spatial_gate_initial_beta must lie in [0,1]."
            )
        if self.graph.type == "none" and self.spatial_gate_type != "none":
            raise ValueError(
                "Graph-free models must use spatial_gate_type='none'."
            )
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0,1).")


@dataclass
class ContinuousForecastOutput:
    predictions: Tensor
    output_representation: OutputRepresentation
    temporal_hidden: Tensor
    graph_spatial_hidden: Tensor
    spatial_hidden: Tensor
    spatial_beta: Tensor | None
    graph: GraphOutput

    @property
    def predictions_normalised(self) -> Tensor:
        """Backward-compatible access for legacy normalised-Close runs."""
        if self.output_representation != "normalised_close":
            raise RuntimeError(
                "predictions_normalised is invalid when the model outputs "
                "cumulative log changes. Use output.predictions instead."
            )
        return self.predictions

    @property
    def predictions_cumulative_log_change(self) -> Tensor:
        if self.output_representation != "cumulative_log_change":
            raise RuntimeError(
                "The model is not configured for cumulative-log-change "
                "output."
            )
        return self.predictions

    def validate(self, config: ContinuousForecasterConfig) -> None:
        batch_size = int(self.temporal_hidden.shape[0])
        expected_prediction = (
            batch_size,
            len(config.horizons),
            config.num_nodes,
            1,
        )
        if tuple(self.predictions.shape) != expected_prediction:
            raise ValueError(
                "Unexpected prediction shape. "
                f"Expected {expected_prediction}, received "
                f"{tuple(self.predictions.shape)}."
            )
        if self.output_representation != config.output_representation:
            raise ValueError(
                "Output representation does not match model config."
            )
        if self.temporal_hidden.ndim != 4:
            raise ValueError(
                "temporal_hidden must have shape [B,L,N,D]."
            )
        if self.graph_spatial_hidden.shape != self.temporal_hidden.shape:
            raise ValueError(
                "graph_spatial_hidden must match temporal_hidden shape."
            )
        if self.spatial_hidden.shape != self.temporal_hidden.shape:
            raise ValueError(
                "spatial_hidden must match temporal_hidden shape."
            )
        if self.spatial_beta is not None:
            if self.spatial_beta.numel() != 1:
                raise ValueError("spatial_beta must be scalar.")
            beta_value = float(self.spatial_beta.detach().item())
            if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
                raise ValueError("spatial_beta must be finite in [0,1].")
        if self.temporal_hidden.shape[2:] != (
            config.num_nodes,
            config.temporal.d_model,
        ):
            raise ValueError("Unexpected temporal node/hidden dimensions.")
        self.graph.validate(
            batch_size=batch_size,
            num_heads=config.graph.num_heads,
            num_nodes=config.num_nodes,
        )


def build_context_session_features(
    *,
    context_start: Tensor,
    session_length: Tensor,
    context_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return ``[B,T,3]`` absolute session-position descriptors.

    The descriptor exactly matches the selected ModernTCN extension:

        p, sin(2πp), cos(2πp)
    """
    context_start = torch.as_tensor(context_start)
    session_length = torch.as_tensor(session_length)
    if context_start.ndim != 1 or session_length.ndim != 1:
        raise ValueError(
            "context_start and session_length must both have shape [B]."
        )
    if context_start.shape != session_length.shape:
        raise ValueError(
            "context_start and session_length must have the same shape."
        )
    if torch.any(context_start < 0) or torch.any(session_length <= 1):
        raise ValueError("Invalid context/session positions.")

    starts = context_start.to(device=device, dtype=torch.long)
    lengths = session_length.to(device=device, dtype=torch.long)
    positions = torch.arange(
        context_length,
        device=device,
        dtype=torch.long,
    )
    absolute = starts[:, None] + positions[None, :]
    if torch.any(absolute[:, -1] >= lengths):
        raise ValueError("A context window extends beyond its session.")
    denominator = (lengths - 1).to(dtype=dtype)[:, None]
    p = absolute.to(dtype=dtype) / denominator
    return torch.stack(
        [p, torch.sin(2.0 * math.pi * p), torch.cos(2.0 * math.pi * p)],
        dim=-1,
    )


def build_modern_tcn_patch_features(
    temporal_features: Tensor,
    *,
    patch_size: int,
    patch_stride: int,
) -> Tensor:
    """Match ModernTCN's replicated-final-value patch alignment."""
    if temporal_features.ndim != 3 or temporal_features.shape[-1] != 3:
        raise ValueError(
            "temporal_features must have shape [B,T,3]."
        )
    padding_length = patch_size - patch_stride
    padded = temporal_features
    if padding_length > 0:
        padded = torch.cat(
            [
                temporal_features,
                temporal_features[:, -1:, :].expand(
                    -1,
                    padding_length,
                    -1,
                ),
            ],
            dim=1,
        )
    patches = padded.unfold(1, patch_size, patch_stride)
    return patches.mean(dim=-1)


class DirectFlattenForecastHead(nn.Module):
    """Shared per-asset flatten head equivalent to ModernTCN's shared head."""

    def __init__(
        self,
        *,
        d_model: int,
        feature_length: int,
        num_horizons: int,
        dropout: float,
        initialisation: OutputHeadInitialisation = "default",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.feature_length = int(feature_length)
        self.num_horizons = int(num_horizons)
        self.linear = nn.Linear(
            self.d_model * self.feature_length,
            self.num_horizons,
        )
        self.dropout = nn.Dropout(dropout)
        if initialisation == "zero":
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        elif initialisation != "default":
            raise ValueError(
                "initialisation must be 'default' or 'zero'."
            )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,L,N,D].")
        batch_size, feature_length, num_nodes, d_model = hidden.shape
        if (feature_length, d_model) != (
            self.feature_length,
            self.d_model,
        ):
            raise ValueError("Unexpected hidden feature dimensions.")
        flattened = (
            hidden.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, num_nodes, -1)
        )
        predictions = self.dropout(self.linear(flattened))
        return predictions.permute(0, 2, 1).unsqueeze(-1).contiguous()


class TransformerContinuousBackbone(nn.Module):
    """Our shared causal per-asset Transformer on continuous OHLCV."""

    def __init__(
        self,
        *,
        config: ContinuousForecasterConfig,
    ) -> None:
        super().__init__()
        temporal = config.temporal
        self.context_length = int(config.context_length)
        self.num_nodes = int(config.num_nodes)
        self.d_model = int(temporal.d_model)
        self.session_position_encoding = bool(
            temporal.session_position_encoding
        )
        self.relative_position_embedding_enabled = bool(
            temporal.relative_position_embedding
        )

        self.input_projection = nn.Linear(
            len(config.input_channels),
            self.d_model,
        )
        if self.relative_position_embedding_enabled:
            self.relative_position_embedding: nn.Embedding | None = nn.Embedding(
                self.context_length,
                self.d_model,
            )
        else:
            self.relative_position_embedding = None
        if self.session_position_encoding:
            self.session_position_projection: nn.Linear | None = nn.Linear(
                3,
                self.d_model,
            )
        else:
            self.session_position_projection = None
        self.input_norm = nn.LayerNorm(self.d_model)
        self.encoder = PerNodeTransformerEncoder(
            d_model=self.d_model,
            config=TemporalConfig(
                type="transformer",
                num_layers=temporal.num_layers,
                num_heads=temporal.num_heads,
                feedforward_multiplier=temporal.feedforward_multiplier,
                dropout=temporal.dropout,
            ),
        )
        self.output_length = self.context_length

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C].")
        batch_size, num_steps, num_nodes, _ = x.shape
        if num_steps != self.context_length or num_nodes != self.num_nodes:
            raise ValueError("Unexpected Transformer input shape.")
        hidden = self.input_projection(x)
        if self.relative_position_embedding is not None:
            ids = torch.arange(num_steps, device=x.device)
            hidden = hidden + self.relative_position_embedding(ids).view(
                1,
                num_steps,
                1,
                self.d_model,
            )
        if self.session_position_projection is not None:
            features = build_context_session_features(
                context_start=context_start,
                session_length=session_length,
                context_length=num_steps,
                device=x.device,
                dtype=x.dtype,
            )
            hidden = hidden + self.session_position_projection(features).view(
                batch_size,
                num_steps,
                1,
                self.d_model,
            )
        return self.encoder(self.input_norm(hidden))


class ModernTCNContinuousBackbone(nn.Module):
    """Official per-asset ModernTCN feature extractor, reduced to Close state.

    OHLCV channels are variables within each asset.  The official ConvFFN2
    mixes those variables, so the selected Close feature stream can contain
    information from all five input channels.  Assets remain folded into the
    model batch and cannot interact before the explicit graph stage.
    """

    def __init__(
        self,
        *,
        config: ContinuousForecasterConfig,
    ) -> None:
        super().__init__()
        temporal = config.temporal
        self.context_length = int(config.context_length)
        self.num_nodes = int(config.num_nodes)
        self.input_channels = tuple(config.input_channels)
        self.close_index = self.input_channels.index(config.target_channel)
        self.d_model = int(temporal.d_model)
        self.patch_size = int(temporal.patch_size)
        self.patch_stride = int(temporal.patch_stride)
        self.session_position_encoding = bool(
            temporal.session_position_encoding
        )
        self.output_length = self.context_length // self.patch_stride

        project_root = Path(__file__).resolve().parents[2]
        modern_tcn_root = (
            project_root
            / "external"
            / "ModernTCN"
            / "ModernTCN-Long-term-forecasting"
        )
        if not modern_tcn_root.is_dir():
            raise FileNotFoundError(
                "Initialise external/ModernTCN before using the "
                "ModernTCN temporal backbone."
            )
        root_string = str(modern_tcn_root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
        from models.ModernTCN import Model as OfficialModernTCNModel

        official_config = SimpleNamespace(
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=temporal.modern_tcn_ffn_ratio,
            num_blocks=[temporal.modern_tcn_num_blocks],
            large_size=[temporal.modern_tcn_large_kernel],
            small_size=[temporal.modern_tcn_small_kernel],
            dims=[self.d_model] * 4,
            dw_dims=[self.d_model] * 4,
            enc_in=len(self.input_channels),
            small_kernel_merged=False,
            dropout=temporal.modern_tcn_dropout,
            head_dropout=temporal.modern_tcn_head_dropout,
            use_multi_scale=False,
            revin=0,
            affine=0,
            subtract_last=0,
            freq="t",
            seq_len=self.context_length,
            pred_len=len(config.horizons),
            individual=0,
            decomposition=0,
            kernel_size=25,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        official = OfficialModernTCNModel(official_config)
        if self.session_position_encoding:
            self.official_model: nn.Module = _TemporalEncodingModernTCNAdapter(
                official_model=official,
                hidden_dim=self.d_model,
            )
        else:
            self.official_model = official

    @property
    def _outer_official_model(self) -> nn.Module:
        if isinstance(
            self.official_model,
            _TemporalEncodingModernTCNAdapter,
        ):
            return self.official_model.official_model
        return self.official_model

    @property
    def _official_backbone(self) -> nn.Module:
        return self._outer_official_model.model

    def initialise_forecast_head(
        self,
        initialisation: OutputHeadInitialisation,
    ) -> None:
        if initialisation == "default":
            return
        if initialisation != "zero":
            raise ValueError(
                "initialisation must be 'default' or 'zero'."
            )
        linear_modules = [
            module
            for module in self._official_backbone.head.modules()
            if isinstance(module, nn.Linear)
        ]
        if not linear_modules:
            raise RuntimeError(
                "The official ModernTCN forecasting head exposes no "
                "nn.Linear module to initialise."
            )
        for module in linear_modules:
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C].")
        batch_size, num_steps, num_nodes, num_channels = x.shape
        if (num_steps, num_nodes, num_channels) != (
            self.context_length,
            self.num_nodes,
            len(self.input_channels),
        ):
            raise ValueError("Unexpected ModernTCN input shape.")

        per_asset = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * num_nodes, num_steps, num_channels)
        )
        model_channels_first = per_asset.permute(0, 2, 1).contiguous()

        if isinstance(
            self.official_model,
            _TemporalEncodingModernTCNAdapter,
        ):
            session_features = build_context_session_features(
                context_start=context_start,
                session_length=session_length,
                context_length=self.context_length,
                device=x.device,
                dtype=x.dtype,
            )
            patch_features = build_modern_tcn_patch_features(
                session_features,
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
            ).repeat_interleave(self.num_nodes, dim=0)
            features = self.official_model._forward_features_with_temporal_encoding(
                x=model_channels_first,
                temporal_patch_features=patch_features,
            )
        else:
            features = self._official_backbone.forward_feature(
                model_channels_first
            )

        expected = (
            batch_size * num_nodes,
            len(self.input_channels),
            self.d_model,
            self.output_length,
        )
        if tuple(features.shape) != expected:
            raise RuntimeError(
                "Unexpected official ModernTCN feature shape. "
                f"Expected {expected}, received {tuple(features.shape)}."
            )
        close_features = features[:, self.close_index]
        return (
            close_features.reshape(
                batch_size,
                num_nodes,
                self.d_model,
                self.output_length,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forecast(self, hidden: Tensor) -> Tensor:
        """Apply the exact official shared Flatten_Head to Close features."""
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,L,N,D].")
        batch_size, feature_length, num_nodes, d_model = hidden.shape
        if (feature_length, num_nodes, d_model) != (
            self.output_length,
            self.num_nodes,
            self.d_model,
        ):
            raise ValueError("Unexpected ModernTCN hidden shape.")
        official_head_input = (
            hidden.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(
                batch_size * num_nodes,
                1,
                self.d_model,
                self.output_length,
            )
        )
        values = self._official_backbone.head(official_head_input)
        if values.ndim != 3 or values.shape[1] != 1:
            raise RuntimeError("Unexpected official ModernTCN head output.")
        return (
            values[:, 0]
            .reshape(batch_size, num_nodes, -1)
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .contiguous()
        )


class SpatialBranchGate(nn.Module):
    """Blend temporal and graph-aware features through one scalar beta.

    The graph-aware branch is produced by the existing spatial module. The
    final representation supplied to the forecasting head is

        H = (1 - beta) * H_temporal + beta * H_graph.

    ``gate_type='none'`` recovers the previous full-spatial behaviour with
    beta=1. ``fixed`` keeps the configured beta constant, and
    ``learned_scalar`` optimises a sigmoid-parameterised scalar.
    """

    def __init__(
        self,
        *,
        gate_type: SpatialGateType,
        initial_beta: float,
    ) -> None:
        super().__init__()
        if gate_type not in {"none", "fixed", "learned_scalar"}:
            raise ValueError(f"Unsupported spatial gate type {gate_type!r}.")
        if not 0.0 <= float(initial_beta) <= 1.0:
            raise ValueError("initial_beta must lie in [0,1].")
        self.gate_type = gate_type
        self.initial_beta = float(initial_beta)

        if gate_type == "none":
            self.register_parameter("raw_beta", None)
            self.register_buffer("fixed_beta", None, persistent=False)
        elif gate_type == "fixed":
            self.register_parameter("raw_beta", None)
            self.register_buffer(
                "fixed_beta",
                torch.tensor(self.initial_beta, dtype=torch.float32),
                persistent=True,
            )
        else:
            epsilon = 1.0e-6
            clipped = min(max(self.initial_beta, epsilon), 1.0 - epsilon)
            raw = math.log(clipped / (1.0 - clipped))
            self.raw_beta = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
            self.register_buffer("fixed_beta", None, persistent=False)

    def beta(self, *, reference: Tensor | None = None) -> Tensor:
        if self.gate_type == "none":
            value = torch.tensor(1.0, dtype=torch.float32)
        elif self.gate_type == "fixed":
            if self.fixed_beta is None:
                raise RuntimeError("Fixed spatial beta is missing.")
            value = self.fixed_beta
        else:
            if self.raw_beta is None:
                raise RuntimeError("Learned spatial beta is missing.")
            value = torch.sigmoid(self.raw_beta)
        if reference is not None:
            value = value.to(device=reference.device, dtype=reference.dtype)
        return value

    def forward(
        self,
        temporal_hidden: Tensor,
        graph_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if temporal_hidden.shape != graph_hidden.shape:
            raise ValueError(
                "temporal_hidden and graph_hidden must have identical shapes."
            )
        beta = self.beta(reference=temporal_hidden)
        fused = (1.0 - beta) * temporal_hidden + beta * graph_hidden
        return fused, beta


class ContinuousForecaster(nn.Module):
    """Continuous Close forecaster with a swappable temporal backbone."""

    def __init__(
        self,
        config: ContinuousForecasterConfig,
        *,
        fixed_adjacency: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config

        if config.temporal.type == "transformer":
            self.temporal_backbone: nn.Module = TransformerContinuousBackbone(
                config=config
            )
        else:
            self.temporal_backbone = ModernTCNContinuousBackbone(
                config=config
            )

        self.graph_learner = build_graph_learner(
            config=config.graph,
            num_nodes=config.num_nodes,
            d_model=config.temporal.d_model,
            fixed_adjacency=fixed_adjacency,
            dynamic_correlation_threshold=(
                config.dynamic_correlation_threshold
            ),
            dynamic_correlation_empty_row_policy=(
                config.dynamic_correlation_empty_row_policy
            ),
            dynamic_correlation_eps=config.dynamic_correlation_eps,
        )
        if config.graph.type == "none":
            self.spatial_module: nn.Module = IdentitySpatialModule()
            self.spatial_gate: SpatialBranchGate | None = None
        else:
            self.spatial_module = SpatialMessagePassing(
                d_model=config.temporal.d_model,
                num_heads=config.graph.num_heads,
                num_layers=config.spatial_num_layers,
                feedforward_multiplier=(
                    config.spatial_feedforward_multiplier
                ),
                dropout=config.spatial_dropout,
            )
            self.spatial_gate = SpatialBranchGate(
                gate_type=config.spatial_gate_type,
                initial_beta=config.spatial_gate_initial_beta,
            )

        if config.temporal.type == "transformer":
            output_length = config.context_length
            self.direct_head: DirectFlattenForecastHead | None = (
                DirectFlattenForecastHead(
                    d_model=config.temporal.d_model,
                    feature_length=output_length,
                    num_horizons=len(config.horizons),
                    dropout=config.head_dropout,
                    initialisation=config.output_head_initialisation,
                )
            )
        else:
            self.direct_head = None
            if not isinstance(
                self.temporal_backbone,
                ModernTCNContinuousBackbone,
            ):
                raise TypeError("Unexpected temporal backbone type.")
            self.temporal_backbone.initialise_forecast_head(
                config.output_head_initialisation
            )

    def spatial_mixing_beta(self) -> Tensor | None:
        if self.spatial_gate is None:
            return None
        return self.spatial_gate.beta()

    def dynamic_graph_alpha(self) -> Tensor | None:
        method = getattr(self.graph_learner, "dynamic_residual_alpha", None)
        if method is None:
            return None
        return method()

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
        graph_context_values: Tensor | None = None,
    ) -> ContinuousForecastOutput:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C].")
        if tuple(x.shape[1:]) != (
            self.config.context_length,
            self.config.num_nodes,
            len(self.config.input_channels),
        ):
            raise ValueError(
                "x does not match the configured context/node/channel axes."
            )

        temporal_hidden = self.temporal_backbone(
            x,
            context_start=context_start,
            session_length=session_length,
        )
        if self.config.graph.type == "dynamic_correlation":
            graph = self.graph_learner(
                temporal_hidden,
                context_values=graph_context_values,
            )
        else:
            graph = self.graph_learner(temporal_hidden)
        if graph.selected is None:
            graph_spatial_hidden = self.spatial_module(temporal_hidden)
            spatial_hidden = graph_spatial_hidden
            spatial_beta = None
        else:
            graph_spatial_hidden = self.spatial_module(
                temporal_hidden,
                graph.selected,
            )
            if self.spatial_gate is None:
                raise RuntimeError("Graph model is missing its spatial gate.")
            spatial_hidden, spatial_beta = self.spatial_gate(
                temporal_hidden,
                graph_spatial_hidden,
            )

        if isinstance(
            self.temporal_backbone,
            ModernTCNContinuousBackbone,
        ):
            predictions = self.temporal_backbone.forecast(spatial_hidden)
        else:
            if self.direct_head is None:
                raise RuntimeError("Transformer direct head is missing.")
            predictions = self.direct_head(spatial_hidden)

        output = ContinuousForecastOutput(
            predictions=predictions,
            output_representation=self.config.output_representation,
            temporal_hidden=temporal_hidden,
            graph_spatial_hidden=graph_spatial_hidden,
            spatial_hidden=spatial_hidden,
            spatial_beta=spatial_beta,
            graph=graph,
        )
        output.validate(self.config)
        return output



@dataclass
class ContinuousRunEvaluation:
    """Complete saved-run inference and evaluation result."""

    run_dir: Path
    split_name: str
    checkpoint_epoch: int
    prediction_path: Path
    graph_path: Path | None
    metric_path: Path
    diagnostics_path: Path | None
    prediction_result: dict[str, Any]
    metric_results: dict[str, Any]
    metric_table: Any


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(
            f"Expected a JSON object in {path}; received "
            f"{type(values).__name__}."
        )
    return values


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_saved_run_device(
    device: str | torch.device,
) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    else:
        value = str(device).strip().lower()
        if value == "auto":
            if torch.cuda.is_available():
                value = "cuda"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                value = "mps"
            else:
                value = "cpu"
        resolved = torch.device(value)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but MPS is unavailable.")
    return resolved


def _normalise_saved_split_name(split_name: str) -> str:
    value = str(split_name).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    if value not in aliases:
        raise ValueError(
            "split_name must be train, validation/val, or test."
        )
    return aliases[value]


def _unwrap_saved_prediction_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved prediction artefact must be a mapping.")
    if "prediction_result" in payload:
        result = payload["prediction_result"]
    else:
        result = payload
    if not isinstance(result, Mapping):
        raise TypeError("prediction_result must be a mapping.")
    required = {
        "y_pred",
        "y_true",
        "last_context_target",
        "channels",
        "horizons",
        "sample_idx",
        "origin_idx",
        "target_indices",
    }
    missing = required.difference(result)
    if missing:
        raise KeyError(
            "Saved continuous prediction result is missing fields: "
            f"{sorted(missing)}"
        )
    return dict(result)


def evaluate_saved_continuous_forecaster_run(
    *,
    run_dir: str | Path,
    train_split: Mapping[str, Any],
    evaluation_split: Mapping[str, Any],
    split_name: str = "test",
    run_inference: bool = False,
    device: str | torch.device = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
    mixed_precision: bool | None = None,
    prediction_filename: str | None = None,
    metrics: Sequence[str] | None = None,
    bootstrap: bool = True,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> ContinuousRunEvaluation:
    """Run or reload one saved continuous forecaster on a data split.

    The run directory is authoritative. The helper loads its exact
    ``resolved_config.json`` and ``best_checkpoint.pt``; it never rebuilds the
    model from the current default YAML. When ``run_inference`` is true, the
    selected checkpoint is evaluated once and the predictions and graph
    artefacts are saved beside the checkpoint. When false, the saved prediction
    artefact is loaded directly and only the common ForecastEvaluator is run.

    No model fitting occurs in this function.
    """
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(run_path)

    resolved_split = _normalise_saved_split_name(split_name)
    metadata_path = run_path / "run_metadata.json"
    config_path = run_path / "resolved_config.json"
    checkpoint_path = run_path / "best_checkpoint.pt"
    for path in (metadata_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _load_json_object(metadata_path)
    if metadata.get("status") != "completed":
        raise RuntimeError(
            f"Continuous run is not marked completed: {run_path.name}"
        )
    resolved_config = _load_json_object(config_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Continuous best checkpoint must be a mapping.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Continuous checkpoint has no model_state_dict.")

    checkpoint_epoch = int(checkpoint["epoch"])
    checkpoint_best_epoch = int(
        checkpoint.get("best_epoch", checkpoint_epoch)
    )
    metadata_best_epoch = int(metadata["best_epoch"])
    if checkpoint_epoch != checkpoint_best_epoch:
        raise AssertionError(
            "best_checkpoint.pt does not represent its recorded best epoch: "
            f"epoch={checkpoint_epoch}, best_epoch={checkpoint_best_epoch}."
        )
    if checkpoint_epoch != metadata_best_epoch:
        raise AssertionError(
            "best_checkpoint.pt epoch differs from run_metadata.json: "
            f"checkpoint={checkpoint_epoch}, metadata={metadata_best_epoch}."
        )
    saved_checkpoint_config = checkpoint.get("resolved_config")
    if (
        saved_checkpoint_config is not None
        and saved_checkpoint_config != resolved_config
    ):
        raise ValueError(
            "best_checkpoint.pt and resolved_config.json describe different "
            "continuous models."
        )

    train_assets = [str(value) for value in train_split["asset_cols"]]
    evaluation_assets = [
        str(value) for value in evaluation_split["asset_cols"]
    ]
    if evaluation_assets != train_assets:
        raise ValueError(
            "Training and evaluation asset order differs."
        )
    metadata_assets = [str(value) for value in metadata["asset_cols"]]
    if metadata_assets != train_assets:
        raise ValueError(
            "Saved continuous model asset order differs from the supplied "
            "training/evaluation splits."
        )

    filename = (
        f"{resolved_split}_predictions.pt"
        if prediction_filename is None
        else str(prediction_filename)
    )
    if not filename.endswith(".pt"):
        raise ValueError("prediction_filename must end in .pt.")
    prediction_path = run_path / filename
    graph_path = run_path / f"{resolved_split}_graphs.pt"
    metric_path = run_path / f"{resolved_split}_metric_table.csv"
    diagnostics_path = run_path / f"{resolved_split}_diagnostics.json"

    if run_inference:
        # Local imports avoid creating a model/training import cycle while
        # reusing the production runner's exact dataset and decoding contract.
        from src.data.continuous_forecast_dataset import (
            build_continuous_dataset,
        )
        from src.models.dynamic_graph.fixed_graph_resource import (
            FixedGraphResource,
        )
        from src.training import run_continuous_forecaster as runner

        runner.validate_config(resolved_config)
        dataset_config = runner._dataset_config(resolved_config)
        dataset = build_continuous_dataset(
            dict(evaluation_split),
            config=dataset_config,
        )
        resolved_device = _resolve_saved_run_device(device)
        training_config = resolved_config["training"]
        loader = runner._build_loader(
            dataset,
            batch_size=(
                int(training_config["validation_batch_size"])
                if batch_size is None
                else int(batch_size)
            ),
            shuffle=False,
            num_workers=(
                int(training_config["num_workers"])
                if num_workers is None
                else int(num_workers)
            ),
            seed=int(training_config["seed"]) + 2,
            pin_memory=resolved_device.type == "cuda",
        )

        model_config = runner._model_config(
            resolved_config,
            num_nodes=len(train_assets),
        )
        fixed_adjacency = None
        if model_config.graph.type == "fixed":
            resource_path = run_path / "fixed_graph_resource.pt"
            if not resource_path.is_file():
                raise FileNotFoundError(resource_path)
            resource_payload = torch.load(
                resource_path,
                map_location="cpu",
                weights_only=False,
            )
            resource = FixedGraphResource.from_payload(resource_payload)
            if list(resource.asset_cols) != train_assets:
                raise ValueError(
                    "Fixed graph asset order differs from the model split."
                )
            fixed_adjacency = resource.adjacency

        model = ContinuousForecaster(
            model_config,
            fixed_adjacency=fixed_adjacency,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        model.to(resolved_device)
        model.eval()

        use_amp = (
            bool(training_config["mixed_precision"])
            if mixed_precision is None
            else bool(mixed_precision)
        ) and resolved_device.type == "cuda"
        validation = runner._run_validation(
            model=model,
            loader=loader,
            device=resolved_device,
            use_amp=use_amp,
            config=resolved_config,
            train_split=dict(train_split),
            asset_cols=train_assets,
            description=(
                f"{run_path.name} {resolved_split} selected-checkpoint inference"
            ),
        )
        prediction_result = dict(validation["prediction_result"])
        _atomic_torch_save(
            {
                "epoch": checkpoint_epoch,
                "prediction_result": prediction_result,
            },
            prediction_path,
        )
        _atomic_torch_save(
            {
                "epoch": checkpoint_epoch,
                "graph_artifacts": validation["graphs"],
            },
            graph_path,
        )
        _atomic_json_save(
            {
                "run_name": metadata.get("run_name", run_path.name),
                "split": resolved_split,
                "checkpoint_epoch": checkpoint_epoch,
                "device": str(resolved_device),
                "mixed_precision": use_amp,
                "windows": int(prediction_result["y_pred"].shape[0]),
                "prediction_path": str(prediction_path),
                "graph_path": str(graph_path),
                "inference_seconds": float(validation["seconds"]),
                "spatial_beta": validation.get("spatial_beta"),
                "dynamic_alpha": validation.get("dynamic_alpha"),
                "graph_summary": validation.get("graph_summary"),
            },
            diagnostics_path,
        )
    else:
        if not prediction_path.is_file():
            raise FileNotFoundError(
                "Saved continuous prediction artefact does not exist. "
                f"Set run_inference=True once to create it: {prediction_path}"
            )
        prediction_result = _unwrap_saved_prediction_result(
            torch.load(
                prediction_path,
                map_location="cpu",
                weights_only=False,
            )
        )

    from src.evaluation.metrics import ForecastEvaluator
    from src.utils.metric_tables import make_evaluation_table

    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
        train_split=dict(train_split),
    )
    requested_metrics = (
        evaluator.available_metrics
        if metrics is None
        else tuple(str(value) for value in metrics)
    )
    metric_results = evaluator.evaluate(
        metrics=requested_metrics,
        reduce_dims=(0, 2),
        bootstrap=bool(bootstrap),
        n_bootstrap=int(n_bootstrap),
        confidence_level=float(confidence_level),
        bootstrap_seed=int(bootstrap_seed),
    )
    metric_table = make_evaluation_table(
        metric_results=metric_results,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )
    metric_table.to_csv(metric_path, index=False)

    return ContinuousRunEvaluation(
        run_dir=run_path,
        split_name=resolved_split,
        checkpoint_epoch=checkpoint_epoch,
        prediction_path=prediction_path,
        graph_path=(graph_path if graph_path.is_file() else None),
        metric_path=metric_path,
        diagnostics_path=(
            diagnostics_path if diagnostics_path.is_file() else None
        ),
        prediction_result=prediction_result,
        metric_results=metric_results,
        metric_table=metric_table,
    )

def _cpu_smoke_test() -> None:
    torch.manual_seed(11)
    config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=12,
        horizons=(1, 5),
        input_channels=("open", "high", "low", "close", "volume"),
        temporal=ContinuousTemporalConfig(
            type="transformer",
            d_model=16,
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        graph=GraphConfig(
            type="none",
            num_heads=2,
            hidden_dim=16,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
    )
    model = ContinuousForecaster(config)
    x = torch.randn(2, 12, 4, 5)
    starts = torch.tensor([0, 5])
    lengths = torch.tensor([30, 30])
    output = model(x, context_start=starts, session_length=lengths)
    if tuple(output.predictions.shape) != (2, 2, 4, 1):
        raise AssertionError("Unexpected continuous forecast shape.")

    # No-graph temporal path must be asset independent.
    changed = x.clone()
    changed[:, :, 3] += 10.0
    model.eval()
    with torch.no_grad():
        reference = model(x, context_start=starts, session_length=lengths)
        perturbed = model(changed, context_start=starts, session_length=lengths)
    if not torch.allclose(
        reference.predictions[:, :, :3],
        perturbed.predictions[:, :, :3],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError("Temporal-only model mixed assets.")

    graph_config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=12,
        horizons=(1, 5),
        input_channels=config.input_channels,
        temporal=config.temporal,
        graph=GraphConfig(
            type="free_static",
            num_heads=2,
            hidden_dim=16,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
    )
    graph_model = ContinuousForecaster(graph_config)
    graph_output = graph_model(
        x,
        context_start=starts,
        session_length=lengths,
    )
    if graph_output.graph.selected is None:
        raise AssertionError("Graph run did not expose an adjacency.")
    if not torch.allclose(
        graph_output.graph.selected.sum(dim=-1),
        torch.ones(2, 2, 4),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError("Graph rows do not sum to one.")
    loss = graph_output.predictions.square().mean()
    loss.backward()
    gradient = graph_model.graph_learner.logits.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("Graph learner did not receive gradients.")


if __name__ == "__main__":
    _cpu_smoke_test()
    print("Continuous forecaster CPU smoke test passed.")

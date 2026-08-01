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

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

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
    temporal: ContinuousTemporalConfig = ContinuousTemporalConfig()
    graph: GraphConfig = GraphConfig(type="none", num_heads=2)
    spatial_num_layers: int = 1
    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0
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
        self.temporal.validate(context_length=self.context_length)
        self.graph.validate(
            num_nodes=self.num_nodes,
            d_model=self.temporal.d_model,
        )
        if self.graph.type not in {"none", "fixed", "free_static"}:
            raise ValueError(
                "The first continuous ladder supports graph.type in "
                "{'none','fixed','free_static'} only."
            )
        if self.spatial_num_layers <= 0:
            raise ValueError("spatial_num_layers must be positive.")
        if self.spatial_feedforward_multiplier <= 0:
            raise ValueError(
                "spatial_feedforward_multiplier must be positive."
            )
        if not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError("spatial_dropout must lie in [0,1).")
        if not 0.0 <= self.head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0,1).")


@dataclass
class ContinuousForecastOutput:
    predictions_normalised: Tensor
    temporal_hidden: Tensor
    spatial_hidden: Tensor
    graph: GraphOutput

    def validate(self, config: ContinuousForecasterConfig) -> None:
        batch_size = int(self.temporal_hidden.shape[0])
        expected_prediction = (
            batch_size,
            len(config.horizons),
            config.num_nodes,
            1,
        )
        if tuple(self.predictions_normalised.shape) != expected_prediction:
            raise ValueError(
                "Unexpected prediction shape. "
                f"Expected {expected_prediction}, received "
                f"{tuple(self.predictions_normalised.shape)}."
            )
        if self.temporal_hidden.ndim != 4:
            raise ValueError(
                "temporal_hidden must have shape [B,L,N,D]."
            )
        if self.spatial_hidden.shape != self.temporal_hidden.shape:
            raise ValueError(
                "spatial_hidden must match temporal_hidden shape."
            )
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
        )
        if config.graph.type == "none":
            self.spatial_module: nn.Module = IdentitySpatialModule()
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

        if config.temporal.type == "transformer":
            output_length = config.context_length
            self.direct_head: DirectFlattenForecastHead | None = (
                DirectFlattenForecastHead(
                    d_model=config.temporal.d_model,
                    feature_length=output_length,
                    num_horizons=len(config.horizons),
                    dropout=config.head_dropout,
                )
            )
        else:
            self.direct_head = None

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
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
        graph = self.graph_learner(temporal_hidden)
        if graph.selected is None:
            spatial_hidden = self.spatial_module(temporal_hidden)
        else:
            spatial_hidden = self.spatial_module(
                temporal_hidden,
                graph.selected,
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
            predictions_normalised=predictions,
            temporal_hidden=temporal_hidden,
            spatial_hidden=spatial_hidden,
            graph=graph,
        )
        output.validate(self.config)
        return output


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
    if tuple(output.predictions_normalised.shape) != (2, 2, 4, 1):
        raise AssertionError("Unexpected continuous forecast shape.")

    # No-graph temporal path must be asset independent.
    changed = x.clone()
    changed[:, :, 3] += 10.0
    model.eval()
    with torch.no_grad():
        reference = model(x, context_start=starts, session_length=lengths)
        perturbed = model(changed, context_start=starts, session_length=lengths)
    if not torch.allclose(
        reference.predictions_normalised[:, :, :3],
        perturbed.predictions_normalised[:, :, :3],
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
    loss = graph_output.predictions_normalised.square().mean()
    loss.backward()
    gradient = graph_model.graph_learner.logits.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("Graph learner did not receive gradients.")


if __name__ == "__main__":
    _cpu_smoke_test()
    print("Continuous forecaster CPU smoke test passed.")

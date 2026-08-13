from __future__ import annotations

"""Token-space counterpart of the continuous ModernTCN graph Round-2 grid.

The module deliberately reuses the established interlaced graph/spatial blocks
from :mod:`src.models.modern_tcn_graph_round2` while replacing continuous OHLCV
inputs and the direct price head with:

* learned embeddings of the original 1,024-way Kronos coarse ``s1`` IDs only;
* the same six temporal-stack definitions used by continuous Round 2;
* the same dynamic-only and correlation-prior-plus-state graph families;
* the project's structured-parallel Transformer future head;
* 60 parallel coarse-token distributions ``[B, 60, N, 1024]``.

No future token enters the context encoder or graph learner.  Every graph is
computed from the observed 60-token context and follows ``A[target, source]``.
"""

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Type

import torch
from torch import Tensor, nn

from src.models.dynamic_graph.contracts import TemporalConfig
from src.models.dynamic_graph.future_predictor import FutureTransformerLayer
from src.models.dynamic_graph.modules import PerNodeTransformerEncoder
from src.models.modern_tcn_graph_round1 import (
    align_state_embeddings_to_modern_tcn_patches,
)
from src.models.modern_tcn_graph_round2 import (
    GraphActivation,
    GraphFamily,
    InterlacedGraphSpatialBlock,
    PriorType,
    Round2BlockOutput,
    TemporalFamily,
    TransformerRefinementBlock,
)


@dataclass(frozen=True)
class ModernTCNGraphRound2TokenConfig:
    num_nodes: int
    context_length: int = 60
    prediction_length: int = 60
    evaluation_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    vocabulary_size: int = 1024

    temporal_family: TemporalFamily = "modern_tcn_transformer"
    num_transformer_blocks: int = 1

    modern_tcn_d_model: int = 32
    modern_tcn_patch_size: int = 8
    modern_tcn_patch_stride: int = 4
    modern_tcn_ffn_ratio: int = 1
    modern_tcn_num_blocks: int = 1
    modern_tcn_large_kernel: int = 15
    modern_tcn_small_kernel: int = 5
    modern_tcn_dropout: float = 0.05
    modern_tcn_head_dropout: float = 0.0

    transformer_d_model: int = 96
    transformer_num_layers: int = 1
    transformer_num_heads: int = 4
    transformer_feedforward_multiplier: int = 2
    transformer_dropout: float = 0.0
    transformer_relative_position_embedding: bool = True

    graph_family: GraphFamily = "dynamic_only"
    prior_type: PriorType = "none"
    graph_heads_per_block: tuple[int, ...] = (1, 1)
    graph_hidden_dims_per_block: tuple[int, ...] = (32, 96)
    graph_activations_per_block: tuple[GraphActivation, ...] = (
        "softmax",
        "sparsemax",
    )
    graph_initial_alpha: float = 0.5
    prior_scale: float = 4.0
    prior_jitter: float = 0.02
    prior_seed: int = 42

    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0
    spatial_initial_beta: float = 0.5

    future_predictor_num_layers: int = 1
    future_predictor_num_heads: int = 4
    future_predictor_feedforward_multiplier: int = 2
    future_predictor_dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("num_nodes", self.num_nodes),
            ("context_length", self.context_length),
            ("prediction_length", self.prediction_length),
            ("vocabulary_size", self.vocabulary_size),
            ("modern_tcn_d_model", self.modern_tcn_d_model),
            ("transformer_d_model", self.transformer_d_model),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.evaluation_horizons:
            raise ValueError("evaluation_horizons must not be empty.")
        horizons = tuple(int(value) for value in self.evaluation_horizons)
        if horizons != tuple(sorted(set(horizons))):
            raise ValueError("evaluation_horizons must be unique and increasing.")
        if horizons[0] <= 0 or horizons[-1] > self.prediction_length:
            raise ValueError("evaluation_horizons lie outside the future path.")
        if self.temporal_family not in {
            "modern_tcn_transformer",
            "transformer_only",
        }:
            raise ValueError(f"Unsupported temporal_family {self.temporal_family!r}.")
        if self.temporal_family == "modern_tcn_transformer":
            if int(self.num_transformer_blocks) < 0:
                raise ValueError("num_transformer_blocks cannot be negative.")
        elif int(self.num_transformer_blocks) <= 0:
            raise ValueError("transformer_only requires at least one block.")
        if self.temporal_family == "modern_tcn_transformer":
            if self.context_length % self.modern_tcn_patch_stride:
                raise ValueError(
                    "context_length must be divisible by ModernTCN patch stride."
                )
            if self.modern_tcn_patch_size < self.modern_tcn_patch_stride:
                raise ValueError("ModernTCN patch size must be >= patch stride.")
        if self.transformer_d_model % self.transformer_num_heads:
            raise ValueError("Transformer d_model must be divisible by heads.")
        if self.transformer_d_model % self.future_predictor_num_heads:
            raise ValueError(
                "Future-predictor d_model must be divisible by its heads."
            )
        if int(self.transformer_num_layers) <= 0:
            raise ValueError("transformer_num_layers must be positive.")
        if int(self.future_predictor_num_layers) < 0:
            raise ValueError("future_predictor_num_layers cannot be negative.")
        if self.graph_family not in {"dynamic_only", "prior_state"}:
            raise ValueError(f"Unsupported graph_family {self.graph_family!r}.")
        if self.prior_type not in {"none", "sector", "correlation", "uniform"}:
            raise ValueError(f"Unsupported prior_type {self.prior_type!r}.")
        if self.graph_family == "dynamic_only" and self.prior_type != "none":
            raise ValueError("dynamic_only requires prior_type='none'.")

        block_count = self.num_st_blocks
        schedules = (
            self.graph_heads_per_block,
            self.graph_hidden_dims_per_block,
            self.graph_activations_per_block,
        )
        if any(len(values) != block_count for values in schedules):
            raise ValueError("Every graph schedule must match num_st_blocks.")
        allowed_activations = {"softmax", "sparsemax", "entmax15"}
        if any(
            value not in allowed_activations
            for value in self.graph_activations_per_block
        ):
            raise ValueError(
                "Graph activations must be softmax, sparsemax, or entmax15."
            )
        for index, (heads, hidden) in enumerate(
            zip(
                self.graph_heads_per_block,
                self.graph_hidden_dims_per_block,
                strict=True,
            )
        ):
            if int(heads) <= 0 or int(hidden) <= 0 or int(hidden) % int(heads):
                raise ValueError(
                    f"Invalid graph heads/width in block {index}: "
                    f"heads={heads}, hidden={hidden}."
                )
        if not 0.0 < float(self.graph_initial_alpha) < 1.0:
            raise ValueError("graph_initial_alpha must lie strictly in (0,1).")
        if not 0.0 < float(self.spatial_initial_beta) < 1.0:
            raise ValueError("spatial_initial_beta must lie strictly in (0,1).")
        if not math.isfinite(float(self.prior_scale)) or self.prior_scale <= 0:
            raise ValueError("prior_scale must be finite and positive.")
        if not math.isfinite(float(self.prior_jitter)) or self.prior_jitter < 0:
            raise ValueError("prior_jitter must be finite and non-negative.")
        for name, value in (
            ("transformer_dropout", self.transformer_dropout),
            ("future_predictor_dropout", self.future_predictor_dropout),
            ("spatial_dropout", self.spatial_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1).")

    @property
    def num_st_blocks(self) -> int:
        return int(self.num_transformer_blocks) + (
            1 if self.temporal_family == "modern_tcn_transformer" else 0
        )

    @property
    def uses_state_pathway(self) -> bool:
        return self.graph_family == "prior_state"

    @property
    def uses_static_graph(self) -> bool:
        return self.graph_family == "prior_state"

    @property
    def block_d_models(self) -> tuple[int, ...]:
        if self.temporal_family == "modern_tcn_transformer":
            return (
                int(self.modern_tcn_d_model),
                *([int(self.transformer_d_model)] * self.num_transformer_blocks),
            )
        return tuple(
            [int(self.transformer_d_model)] * self.num_transformer_blocks
        )

    @property
    def feature_length(self) -> int:
        if self.temporal_family == "modern_tcn_transformer":
            return int(self.context_length) // int(self.modern_tcn_patch_stride)
        return int(self.context_length)

    @property
    def evaluation_indices(self) -> tuple[int, ...]:
        return tuple(int(value) - 1 for value in self.evaluation_horizons)


class CoarseS1Embedding(nn.Module):
    """Shared coarse-token embedding with dimension-specific input adapters.

    The original 1,024-way state table is shared across temporal families.
    Node and position embeddings are added only to the temporal input.  The
    unaugmented state embedding is retained separately for Dimitri-style graph
    scoring and spatial values in the ``prior_state`` family.
    """

    def __init__(self, config: ModernTCNGraphRound2TokenConfig) -> None:
        super().__init__()
        self.config = config
        raw_dim = int(config.transformer_d_model)
        modern_dim = int(config.modern_tcn_d_model)
        self.s1_embedding = nn.Embedding(config.vocabulary_size, raw_dim)
        self.modern_state_projection = nn.Linear(raw_dim, modern_dim)
        self.modern_node_embedding = nn.Embedding(config.num_nodes, modern_dim)
        self.modern_position_embedding = nn.Embedding(
            config.context_length, modern_dim
        )
        self.modern_norm = nn.LayerNorm(modern_dim)
        self.transformer_node_embedding = nn.Embedding(
            config.num_nodes, raw_dim
        )
        self.transformer_position_embedding = nn.Embedding(
            config.context_length, raw_dim
        )
        self.transformer_norm = nn.LayerNorm(raw_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.s1_embedding,
            self.modern_node_embedding,
            self.modern_position_embedding,
            self.transformer_node_embedding,
            self.transformer_position_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.modern_state_projection.reset_parameters()
        self.modern_norm.reset_parameters()
        self.transformer_norm.reset_parameters()

    def raw(self, s1_ids: Tensor) -> Tensor:
        values = torch.as_tensor(s1_ids)
        expected = (
            int(values.shape[0]),
            self.config.context_length,
            self.config.num_nodes,
        ) if values.ndim == 3 else None
        if values.ndim != 3 or tuple(values.shape) != expected:
            raise ValueError("s1_ids must have shape [B, context_length, N].")
        if values.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
            torch.uint8,
        }:
            raise TypeError("s1_ids must use an integer dtype.")
        values = values.long()
        if values.numel() and (
            int(values.min()) < 0 or int(values.max()) >= self.config.vocabulary_size
        ):
            raise ValueError("s1_ids lie outside the configured vocabulary.")
        return self.s1_embedding(values)

    def modern_inputs(self, s1_ids: Tensor) -> tuple[Tensor, Tensor]:
        raw96 = self.raw(s1_ids)
        raw32 = self.modern_state_projection(raw96)
        positions = self.modern_position_embedding(
            torch.arange(self.config.context_length, device=s1_ids.device)
        ).view(1, self.config.context_length, 1, -1)
        nodes = self.modern_node_embedding(
            torch.arange(self.config.num_nodes, device=s1_ids.device)
        ).view(1, 1, self.config.num_nodes, -1)
        return self.modern_norm(raw32 + positions + nodes), raw32

    def transformer_inputs(self, s1_ids: Tensor) -> tuple[Tensor, Tensor]:
        raw96 = self.raw(s1_ids)
        positions = self.transformer_position_embedding(
            torch.arange(self.config.context_length, device=s1_ids.device)
        ).view(1, self.config.context_length, 1, -1)
        nodes = self.transformer_node_embedding(
            torch.arange(self.config.num_nodes, device=s1_ids.device)
        ).view(1, 1, self.config.num_nodes, -1)
        return self.transformer_norm(raw96 + positions + nodes), raw96


class CoarseTokenModernTCNBackbone(nn.Module):
    """Official per-asset ModernTCN applied to D-dimensional s1 embeddings."""

    def __init__(
        self,
        config: ModernTCNGraphRound2TokenConfig,
        *,
        official_model_cls: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_length = int(config.context_length)
        self.num_nodes = int(config.num_nodes)
        self.d_model = int(config.modern_tcn_d_model)
        self.patch_size = int(config.modern_tcn_patch_size)
        self.patch_stride = int(config.modern_tcn_patch_stride)
        self.output_length = int(config.feature_length)
        self.num_variables = self.d_model

        if official_model_cls is None:
            project_root = Path(__file__).resolve().parents[2]
            modern_root = (
                project_root
                / "external"
                / "ModernTCN"
                / "ModernTCN-Long-term-forecasting"
            )
            if not modern_root.is_dir():
                raise FileNotFoundError(
                    "Initialise external/ModernTCN before using this backbone."
                )
            root_string = str(modern_root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            from models.ModernTCN import Model as OfficialModernTCNModel

            official_model_cls = OfficialModernTCNModel

        official_config = SimpleNamespace(
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=int(config.modern_tcn_ffn_ratio),
            num_blocks=[int(config.modern_tcn_num_blocks)],
            large_size=[int(config.modern_tcn_large_kernel)],
            small_size=[int(config.modern_tcn_small_kernel)],
            dims=[self.d_model] * 4,
            dw_dims=[self.d_model] * 4,
            enc_in=self.num_variables,
            small_kernel_merged=False,
            dropout=float(config.modern_tcn_dropout),
            head_dropout=float(config.modern_tcn_head_dropout),
            use_multi_scale=False,
            revin=0,
            affine=0,
            subtract_last=0,
            freq="t",
            seq_len=self.context_length,
            pred_len=self.config.prediction_length,
            individual=0,
            decomposition=0,
            kernel_size=25,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        self.official_model = official_model_cls(official_config)
        if not hasattr(self.official_model, "model") or not hasattr(
            self.official_model.model, "forward_feature"
        ):
            raise TypeError("Official ModernTCN must expose model.forward_feature().")
        self.variable_pool = nn.Linear(self.num_variables, 1, bias=False)
        nn.init.constant_(self.variable_pool.weight, 1.0 / self.num_variables)
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward(self, embedded: Tensor) -> Tensor:
        expected = (
            int(embedded.shape[0]),
            self.context_length,
            self.num_nodes,
            self.d_model,
        ) if embedded.ndim == 4 else None
        if embedded.ndim != 4 or tuple(embedded.shape) != expected:
            raise ValueError("embedded must have shape [B,C,N,D_modern].")
        batch = int(embedded.shape[0])
        per_asset = (
            embedded.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch * self.num_nodes, self.context_length, self.d_model)
        )
        channels_first = per_asset.permute(0, 2, 1).contiguous()
        features = self.official_model.model.forward_feature(channels_first)
        expected_features = (
            batch * self.num_nodes,
            self.num_variables,
            self.d_model,
            self.output_length,
        )
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                f"Unexpected ModernTCN feature shape {tuple(features.shape)}; "
                f"expected {expected_features}."
            )
        pooled = self.variable_pool(
            features.permute(0, 2, 3, 1).contiguous()
        ).squeeze(-1)
        hidden = (
            pooled.reshape(batch, self.num_nodes, self.d_model, self.output_length)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return self.output_norm(hidden)


class CoarseTokenTransformerInputBlock(nn.Module):
    def __init__(self, config: ModernTCNGraphRound2TokenConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PerNodeTransformerEncoder(
            d_model=config.transformer_d_model,
            config=TemporalConfig(
                type="transformer",
                num_layers=int(config.transformer_num_layers),
                num_heads=int(config.transformer_num_heads),
                feedforward_multiplier=int(
                    config.transformer_feedforward_multiplier
                ),
                dropout=float(config.transformer_dropout),
            ),
        )

    def forward(self, embedded: Tensor) -> Tensor:
        return self.encoder(embedded)


class CoarseStructuredParallelPredictor(nn.Module):
    """The established ordered-query structured-parallel future head."""

    def __init__(self, config: ModernTCNGraphRound2TokenConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = int(config.transformer_d_model)
        self.future_position_embedding = nn.Embedding(
            config.prediction_length, self.d_model
        )
        nn.init.normal_(self.future_position_embedding.weight, mean=0.0, std=0.02)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList(
            [
                FutureTransformerLayer(
                    d_model=self.d_model,
                    num_heads=int(config.future_predictor_num_heads),
                    feedforward_multiplier=int(
                        config.future_predictor_feedforward_multiplier
                    ),
                    dropout=float(config.future_predictor_dropout),
                )
                for _ in range(int(config.future_predictor_num_layers))
            ]
        )
        self.classifier = nn.Linear(self.d_model, config.vocabulary_size)

    @staticmethod
    def _flatten_nodes(values: Tensor) -> Tensor:
        batch, length, nodes, hidden = values.shape
        return (
            values.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch * nodes, length, hidden)
        )

    @staticmethod
    def _restore_nodes(values: Tensor, *, batch: int, nodes: int) -> Tensor:
        _, length, hidden = values.shape
        return (
            values.reshape(batch, nodes, length, hidden)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, context_memory: Tensor) -> tuple[Tensor, Tensor]:
        if context_memory.ndim != 4:
            raise ValueError("context_memory must have shape [B,L,N,D].")
        batch, _, nodes, hidden = map(int, context_memory.shape)
        if nodes != self.config.num_nodes or hidden != self.d_model:
            raise ValueError("context_memory does not match predictor config.")
        memory = self._flatten_nodes(context_memory)
        summary = context_memory[:, -1].reshape(batch * nodes, self.d_model)
        positions = self.future_position_embedding(
            torch.arange(self.config.prediction_length, device=context_memory.device)
        )
        future = summary[:, None] + positions[None]
        future = self.input_norm(future)
        for layer in self.layers:
            future = layer(
                future,
                memory,
                self_attention_mask=None,
            )
        restored = self._restore_nodes(future, batch=batch, nodes=nodes)
        logits = self.classifier(restored)
        expected = (
            batch,
            self.config.prediction_length,
            self.config.num_nodes,
            self.config.vocabulary_size,
        )
        if tuple(logits.shape) != expected:
            raise RuntimeError("Structured predictor returned an invalid shape.")
        if not torch.isfinite(logits).all():
            raise ValueError("Structured predictor logits are non-finite.")
        return logits, restored


@dataclass
class ModernTCNGraphRound2TokenOutput:
    s1_logits: Tensor
    selected_s1: Tensor
    future_hidden: Tensor
    block_outputs: tuple[Round2BlockOutput, ...]
    final_hidden: Tensor

    def validate(self, config: ModernTCNGraphRound2TokenConfig) -> None:
        batch = int(self.s1_logits.shape[0])
        expected_logits = (
            batch,
            config.prediction_length,
            config.num_nodes,
            config.vocabulary_size,
        )
        expected_ids = expected_logits[:-1]
        if tuple(self.s1_logits.shape) != expected_logits:
            raise ValueError("s1_logits has an unexpected shape.")
        if tuple(self.selected_s1.shape) != expected_ids:
            raise ValueError("selected_s1 has an unexpected shape.")
        if len(self.block_outputs) != config.num_st_blocks:
            raise ValueError("Unexpected number of ST-block outputs.")
        if not torch.isfinite(self.s1_logits).all():
            raise ValueError("s1_logits contains non-finite values.")
        if self.selected_s1.numel() and (
            int(self.selected_s1.min()) < 0
            or int(self.selected_s1.max()) >= config.vocabulary_size
        ):
            raise ValueError("selected_s1 lies outside the vocabulary.")
        for index, block in enumerate(self.block_outputs):
            graph = block.graph.selected
            expected_graph = (
                batch,
                config.graph_heads_per_block[index],
                config.num_nodes,
                config.num_nodes,
            )
            if graph is None or tuple(graph.shape) != expected_graph:
                raise ValueError(f"Block {index} graph has an unexpected shape.")


class ModernTCNGraphRound2TokenModel(nn.Module):
    """Twelve-grid token model with a common 60-position coarse head."""

    def __init__(
        self,
        config: ModernTCNGraphRound2TokenConfig,
        *,
        static_prior: Tensor | None,
        official_model_cls: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            config.uses_static_graph
            and config.prior_type in {"sector", "correlation"}
            and static_prior is None
        ):
            raise ValueError("A structured prior_state model requires static_prior.")
        if not config.uses_static_graph and static_prior is not None:
            raise ValueError("dynamic_only must not receive static_prior.")
        self.config = config
        self.embedding = CoarseS1Embedding(config)
        self.feature_length = int(config.feature_length)

        if config.temporal_family == "modern_tcn_transformer":
            self.modern_tcn_backbone: CoarseTokenModernTCNBackbone | None = (
                CoarseTokenModernTCNBackbone(
                    config,
                    official_model_cls=official_model_cls,
                )
            )
            self.transformer_input: CoarseTokenTransformerInputBlock | None = None
            if int(config.modern_tcn_d_model) == int(config.transformer_d_model):
                self.modern_to_transformer = nn.Identity()
            else:
                self.modern_to_transformer = nn.Sequential(
                    nn.Linear(
                        config.modern_tcn_d_model,
                        config.transformer_d_model,
                    ),
                    nn.LayerNorm(config.transformer_d_model),
                )
        else:
            self.modern_tcn_backbone = None
            self.transformer_input = CoarseTokenTransformerInputBlock(config)
            self.modern_to_transformer = None

        refinement_count = (
            config.num_transformer_blocks
            if config.temporal_family == "modern_tcn_transformer"
            else config.num_transformer_blocks - 1
        )
        self.temporal_refinements = nn.ModuleList(
            [
                TransformerRefinementBlock(
                    sequence_length=self.feature_length,
                    d_model=config.transformer_d_model,
                    num_layers=config.transformer_num_layers,
                    num_heads=config.transformer_num_heads,
                    feedforward_multiplier=config.transformer_feedforward_multiplier,
                    dropout=config.transformer_dropout,
                    relative_position_embedding=(
                        config.transformer_relative_position_embedding
                    ),
                )
                for _ in range(refinement_count)
            ]
        )

        block_priors = [
            static_prior if config.uses_static_graph else None
            for _ in range(config.num_st_blocks)
        ]
        self.graph_spatial_blocks = nn.ModuleList(
            [
                InterlacedGraphSpatialBlock(
                    d_model=config.block_d_models[index],
                    num_nodes=config.num_nodes,
                    num_heads=config.graph_heads_per_block[index],
                    graph_hidden_dim=config.graph_hidden_dims_per_block[index],
                    graph_activation=config.graph_activations_per_block[index],
                    graph_family=config.graph_family,
                    static_prior=block_priors[index],
                    initial_alpha=config.graph_initial_alpha,
                    prior_scale=config.prior_scale,
                    prior_jitter=config.prior_jitter,
                    prior_seed=config.prior_seed + index * 1009,
                    feedforward_multiplier=config.spatial_feedforward_multiplier,
                    dropout=config.spatial_dropout,
                    initial_beta=config.spatial_initial_beta,
                )
                for index in range(config.num_st_blocks)
            ]
        )
        self.future_predictor = CoarseStructuredParallelPredictor(config)

    def alphas(self) -> tuple[Tensor | None, ...]:
        return tuple(
            block.graph_learner.alpha() for block in self.graph_spatial_blocks
        )

    def betas(self) -> tuple[Tensor, ...]:
        return tuple(block.spatial_gate.beta() for block in self.graph_spatial_blocks)

    def graph_parameter_ids(self) -> set[int]:
        return {
            id(parameter)
            for block in self.graph_spatial_blocks
            for parameter in block.graph_learner.parameters()
            if parameter.requires_grad
        }

    def block_state_modules(self) -> tuple[nn.Module | None, ...]:
        if not self.config.uses_state_pathway:
            return tuple([None] * self.config.num_st_blocks)
        # The shared coarse state table and its D32 projection supply the raw
        # state pathway in every block.  The return is only for diagnostics.
        if self.config.temporal_family == "modern_tcn_transformer":
            return (
                self.embedding.modern_state_projection,
                *([self.embedding.s1_embedding] * self.config.num_transformer_blocks),
            )
        return tuple(
            [self.embedding.s1_embedding] * self.config.num_transformer_blocks
        )

    def forward(self, context_s1: Tensor) -> ModernTCNGraphRound2TokenOutput:
        config = self.config
        outputs: list[Round2BlockOutput] = []

        if config.temporal_family == "modern_tcn_transformer":
            if self.modern_tcn_backbone is None or self.modern_to_transformer is None:
                raise RuntimeError("ModernTCN-first modules are missing.")
            modern_input, raw32 = self.embedding.modern_inputs(context_s1)
            temporal = self.modern_tcn_backbone(modern_input)
            state32 = (
                align_state_embeddings_to_modern_tcn_patches(
                    raw32,
                    patch_size=config.modern_tcn_patch_size,
                    patch_stride=config.modern_tcn_patch_stride,
                ).contiguous()
                if config.uses_state_pathway
                else None
            )
            block0 = self.graph_spatial_blocks[0](
                temporal,
                state_hidden=state32,
            )
            outputs.append(block0)
            hidden = self.modern_to_transformer(block0.fused_hidden)
            raw96 = self.embedding.raw(context_s1)
            state96 = (
                align_state_embeddings_to_modern_tcn_patches(
                    raw96,
                    patch_size=config.modern_tcn_patch_size,
                    patch_stride=config.modern_tcn_patch_stride,
                ).contiguous()
                if config.uses_state_pathway
                else None
            )
            for block_index, temporal_block in enumerate(
                self.temporal_refinements,
                start=1,
            ):
                temporal = temporal_block(hidden)
                block = self.graph_spatial_blocks[block_index](
                    temporal,
                    state_hidden=state96,
                )
                outputs.append(block)
                hidden = block.fused_hidden
        else:
            if self.transformer_input is None:
                raise RuntimeError("Transformer input block is missing.")
            transformer_input, raw96 = self.embedding.transformer_inputs(context_s1)
            temporal = self.transformer_input(transformer_input)
            state96 = raw96 if config.uses_state_pathway else None
            block0 = self.graph_spatial_blocks[0](
                temporal,
                state_hidden=state96,
            )
            outputs.append(block0)
            hidden = block0.fused_hidden
            for block_index, temporal_block in enumerate(
                self.temporal_refinements,
                start=1,
            ):
                temporal = temporal_block(hidden)
                block = self.graph_spatial_blocks[block_index](
                    temporal,
                    state_hidden=state96,
                )
                outputs.append(block)
                hidden = block.fused_hidden

        logits, future_hidden = self.future_predictor(hidden)
        selected = logits.argmax(dim=-1)
        result = ModernTCNGraphRound2TokenOutput(
            s1_logits=logits,
            selected_s1=selected,
            future_hidden=future_hidden,
            block_outputs=tuple(outputs),
            final_hidden=hidden,
        )
        result.validate(config)
        return result


def token_round2_model_config_from_mapping(
    values: dict,
    *,
    num_nodes: int,
    vocabulary_size: int,
) -> ModernTCNGraphRound2TokenConfig:
    model = values["model"]
    temporal = model["temporal_stack"]
    modern = temporal["modern_tcn"]
    transformer = temporal["transformer"]
    graph = model["graph"]
    spatial = model["spatial"]
    prior = model["prior"]
    predictor = model["future_predictor"]
    data = values["data"]
    config = ModernTCNGraphRound2TokenConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        prediction_length=int(data["prediction_length"]),
        evaluation_horizons=tuple(
            int(value) for value in data["evaluation_horizons"]
        ),
        vocabulary_size=int(vocabulary_size),
        temporal_family=str(temporal["family"]),
        num_transformer_blocks=int(temporal["num_transformer_blocks"]),
        modern_tcn_d_model=int(modern["d_model"]),
        modern_tcn_patch_size=int(modern["patch_size"]),
        modern_tcn_patch_stride=int(modern["patch_stride"]),
        modern_tcn_ffn_ratio=int(modern["ffn_ratio"]),
        modern_tcn_num_blocks=int(modern["num_blocks"]),
        modern_tcn_large_kernel=int(modern["large_kernel"]),
        modern_tcn_small_kernel=int(modern["small_kernel"]),
        modern_tcn_dropout=float(modern["dropout"]),
        modern_tcn_head_dropout=float(modern["head_dropout"]),
        transformer_d_model=int(transformer["d_model"]),
        transformer_num_layers=int(transformer["num_layers"]),
        transformer_num_heads=int(transformer["num_heads"]),
        transformer_feedforward_multiplier=int(
            transformer["feedforward_multiplier"]
        ),
        transformer_dropout=float(transformer["dropout"]),
        transformer_relative_position_embedding=bool(
            transformer["relative_position_embedding"]
        ),
        graph_family=str(model["graph_family"]),
        prior_type=str(prior["type"]),
        graph_heads_per_block=tuple(
            int(value) for value in graph["num_heads_per_block"]
        ),
        graph_hidden_dims_per_block=tuple(
            int(value) for value in graph["hidden_dims_per_block"]
        ),
        graph_activations_per_block=tuple(
            str(value) for value in graph["activations_per_block"]
        ),
        graph_initial_alpha=float(graph["initial_alpha"]),
        prior_scale=float(prior["scale"]),
        prior_jitter=float(prior["jitter"]),
        prior_seed=int(prior["seed"]),
        spatial_feedforward_multiplier=int(spatial["feedforward_multiplier"]),
        spatial_dropout=float(spatial["dropout"]),
        spatial_initial_beta=float(spatial["initial_beta"]),
        future_predictor_num_layers=int(predictor["num_layers"]),
        future_predictor_num_heads=int(predictor["num_heads"]),
        future_predictor_feedforward_multiplier=int(
            predictor["feedforward_multiplier"]
        ),
        future_predictor_dropout=float(predictor["dropout"]),
    )
    config.validate()
    return config

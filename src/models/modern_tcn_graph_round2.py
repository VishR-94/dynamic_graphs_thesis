from __future__ import annotations

"""Interlaced ModernTCN/Transformer graph stacks for Round 2.

Round 2 keeps the selected continuous forecasting task and optimisation
protocol, then tests whether repeated temporal -> graph -> spatial blocks
produce progressively more useful graph representations.

Two graph families are supported:

``dynamic_only``
    Every block learns a context-conditioned graph with no static prior,
    no alpha gate, and no explicit state pathway.

``prior_state``
    Every block has a trainable static graph initialised from a sector,
    training-only absolute-correlation, random, or exact-uniform prior.  A learned alpha
    mixes this static graph with a dynamic graph, and the current continuous
    state is concatenated into both graph scoring and spatial values.

The graph used by message passing is always exposed.  Tensor orientation is
``A[target, source]`` throughout.
"""

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.models.continuous_forecaster import (
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
    DirectFlattenForecastHead,
    ModernTCNContinuousBackbone,
    SpatialBranchGate,
    build_context_session_features,
)
from src.models.dynamic_graph.contracts import GraphConfig, GraphOutput, TemporalConfig
from src.models.dynamic_graph.modules import GraphNormalizer, PerNodeTransformerEncoder
from src.models.modern_tcn_graph_round1 import (
    StateAwareSpatialMessagePassing,
    align_state_embeddings_to_modern_tcn_patches,
    build_v2_prior_logits,
)


TemporalFamily = Literal["modern_tcn_transformer", "transformer_only"]
GraphFamily = Literal["dynamic_only", "prior_state"]
PriorType = Literal["none", "sector", "correlation", "uniform"]
GraphActivation = Literal["softmax", "sparsemax"]


@dataclass(frozen=True)
class ModernTCNGraphRound2Config:
    num_nodes: int
    context_length: int
    horizons: tuple[int, ...]
    input_channels: tuple[str, ...]
    target_channel: str

    temporal_family: TemporalFamily
    num_transformer_blocks: int

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
    transformer_session_position_encoding: bool = True

    graph_family: GraphFamily = "dynamic_only"
    prior_type: PriorType = "none"
    graph_heads_per_block: tuple[int, ...] = (1, 1)
    graph_hidden_dims_per_block: tuple[int, ...] = (32, 96)
    graph_activations_per_block: tuple[GraphActivation, ...] = (
        "softmax",
        "sparsemax",
    )
    graph_initial_alpha: float = 0.25
    prior_scale: float = 4.0
    prior_jitter: float = 0.02
    prior_seed: int = 42

    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0
    spatial_initial_beta: float = 0.5
    head_dropout: float = 0.0

    def validate(self) -> None:
        if self.num_nodes <= 0 or self.context_length <= 0:
            raise ValueError("num_nodes and context_length must be positive.")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique, increasing, and non-empty.")
        if any(int(value) <= 0 for value in self.horizons):
            raise ValueError("Every forecast horizon must be positive.")
        if not self.input_channels or self.target_channel not in self.input_channels:
            raise ValueError("The target channel must occur in input_channels.")
        if self.temporal_family not in {
            "modern_tcn_transformer",
            "transformer_only",
        }:
            raise ValueError(f"Unsupported temporal family {self.temporal_family!r}.")
        if self.num_transformer_blocks <= 0:
            raise ValueError("num_transformer_blocks must be positive.")
        if self.temporal_family == "modern_tcn_transformer":
            if self.context_length % self.modern_tcn_patch_stride:
                raise ValueError(
                    "context_length must be divisible by ModernTCN patch stride."
                )
            if self.modern_tcn_patch_size < self.modern_tcn_patch_stride:
                raise ValueError("ModernTCN patch size must be >= patch stride.")
        if self.transformer_d_model % self.transformer_num_heads:
            raise ValueError("Transformer d_model must be divisible by heads.")
        if self.transformer_num_layers <= 0:
            raise ValueError("Transformer layers must be positive.")
        if self.graph_family not in {"dynamic_only", "prior_state"}:
            raise ValueError(f"Unsupported graph family {self.graph_family!r}.")
        if self.prior_type not in {"none", "sector", "correlation", "uniform"}:
            raise ValueError(f"Unsupported prior type {self.prior_type!r}.")
        if self.graph_family == "dynamic_only" and self.prior_type != "none":
            raise ValueError("Dynamic-only models must use prior_type='none'.")

        block_count = self.num_st_blocks
        schedules = {
            "graph_heads_per_block": self.graph_heads_per_block,
            "graph_hidden_dims_per_block": self.graph_hidden_dims_per_block,
            "graph_activations_per_block": self.graph_activations_per_block,
        }
        for name, values in schedules.items():
            if len(values) != block_count:
                raise ValueError(
                    f"{name} has length {len(values)}; expected {block_count}."
                )
        if self.graph_activations_per_block[-1] != "sparsemax":
            raise ValueError("The final Round-2 graph block must use sparsemax.")
        if any(value != "softmax" for value in self.graph_activations_per_block[:-1]):
            raise ValueError("All non-final Round-2 graph blocks must use softmax.")
        for index, (heads, hidden) in enumerate(
            zip(
                self.graph_heads_per_block,
                self.graph_hidden_dims_per_block,
                strict=True,
            )
        ):
            if int(heads) <= 0 or int(hidden) <= 0 or int(hidden) % int(heads):
                raise ValueError(
                    f"Invalid graph heads/hidden width in block {index}: "
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

        for index, d_model in enumerate(self.block_d_models):
            if d_model <= 0:
                raise ValueError(f"Invalid d_model in block {index}.")

    @property
    def num_st_blocks(self) -> int:
        return self.num_transformer_blocks + (
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
            return self.context_length // self.modern_tcn_patch_stride
        return self.context_length


@dataclass
class Round2BlockOutput:
    temporal_hidden: Tensor
    state_hidden: Tensor | None
    graph_spatial_hidden: Tensor
    fused_hidden: Tensor
    graph: GraphOutput
    alpha: Tensor | None
    beta: Tensor


@dataclass
class ModernTCNGraphRound2Output:
    predictions: Tensor
    block_outputs: tuple[Round2BlockOutput, ...]
    final_hidden: Tensor

    def validate(self, config: ModernTCNGraphRound2Config) -> None:
        if len(self.block_outputs) != config.num_st_blocks:
            raise ValueError("Unexpected number of ST-block outputs.")
        batch = int(self.predictions.shape[0])
        expected_predictions = (
            batch,
            len(config.horizons),
            config.num_nodes,
            1,
        )
        if tuple(self.predictions.shape) != expected_predictions:
            raise ValueError(
                f"predictions has shape {tuple(self.predictions.shape)}; "
                f"expected {expected_predictions}."
            )
        for index, block in enumerate(self.block_outputs):
            if block.temporal_hidden.ndim != 4:
                raise ValueError(f"Block {index} temporal hidden is not [B,L,N,D].")
            if block.graph_spatial_hidden.shape != block.temporal_hidden.shape:
                raise ValueError(f"Block {index} spatial hidden shape differs.")
            if block.fused_hidden.shape != block.temporal_hidden.shape:
                raise ValueError(f"Block {index} fused hidden shape differs.")
            if block.state_hidden is not None and (
                block.state_hidden.shape != block.temporal_hidden.shape
            ):
                raise ValueError(f"Block {index} state hidden shape differs.")
            block.graph.validate(
                batch_size=batch,
                num_heads=config.graph_heads_per_block[index],
                num_nodes=config.num_nodes,
            )
            if block.beta.numel() != 1:
                raise ValueError(f"Block {index} beta must be scalar.")
            if block.alpha is not None and block.alpha.numel() != 1:
                raise ValueError(f"Block {index} alpha must be scalar.")


class Round2WindowGraphLearner(nn.Module):
    """One context-window graph learner for an interlaced ST block."""

    def __init__(
        self,
        *,
        d_model: int,
        num_nodes: int,
        num_heads: int,
        graph_hidden_dim: int,
        activation: GraphActivation,
        graph_family: GraphFamily,
        static_prior: Tensor | None,
        initial_alpha: float,
        prior_scale: float,
        prior_jitter: float,
        prior_seed: int,
    ) -> None:
        super().__init__()
        if graph_hidden_dim % num_heads:
            raise ValueError("graph_hidden_dim must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        self.activation = activation
        self.graph_family = graph_family
        self.use_state_pathway = graph_family == "prior_state"
        scorer_dim = self.d_model * (2 if self.use_state_pathway else 1)

        self.q_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.k_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.normalizer = GraphNormalizer(
            GraphConfig(
                type="dynamic",
                num_heads=self.num_heads,
                hidden_dim=self.graph_hidden_dim,
                activation=activation,
                add_self_loops=False,
            )
        )

        if graph_family == "dynamic_only":
            self.register_parameter("static_logits", None)
            self.register_parameter("raw_alpha", None)
        else:
            if static_prior is None:
                generator = torch.Generator(device="cpu").manual_seed(int(prior_seed))
                initial_logits = torch.randn(
                    self.num_heads,
                    self.num_nodes,
                    self.num_nodes,
                    generator=generator,
                    dtype=torch.float32,
                ) * float(prior_jitter)
            else:
                values = torch.as_tensor(static_prior).detach().cpu().float()
                if tuple(values.shape) != (self.num_nodes, self.num_nodes):
                    raise ValueError("Static prior node axes differ from graph config.")
                initial_logits = build_v2_prior_logits(
                    values,
                    num_heads=self.num_heads,
                    scale=float(prior_scale),
                    jitter=float(prior_jitter),
                    seed=int(prior_seed),
                )
            self.static_logits = nn.Parameter(initial_logits)
            epsilon = 1.0e-6
            clipped = min(max(float(initial_alpha), epsilon), 1.0 - epsilon)
            self.raw_alpha = nn.Parameter(
                torch.tensor(
                    math.log(clipped / (1.0 - clipped)),
                    dtype=torch.float32,
                )
            )

    def alpha(self) -> Tensor | None:
        return None if self.raw_alpha is None else torch.sigmoid(self.raw_alpha)

    def static_adjacency(self) -> Tensor | None:
        if self.static_logits is None:
            return None
        return self.normalizer(self.static_logits.unsqueeze(0))

    def forward(
        self,
        temporal_hidden: Tensor,
        *,
        state_hidden: Tensor | None,
    ) -> GraphOutput:
        if temporal_hidden.ndim != 4:
            raise ValueError("temporal_hidden must have shape [B,L,N,D].")
        batch, _, nodes, hidden = map(int, temporal_hidden.shape)
        if (nodes, hidden) != (self.num_nodes, self.d_model):
            raise ValueError("temporal_hidden does not match graph configuration.")

        origin = temporal_hidden[:, -1]
        if self.use_state_pathway:
            if state_hidden is None or state_hidden.shape != temporal_hidden.shape:
                raise ValueError("prior_state graph requires matching state_hidden.")
            scorer = torch.cat([origin, state_hidden[:, -1]], dim=-1)
        else:
            if state_hidden is not None:
                raise ValueError("dynamic_only graph must not receive state_hidden.")
            scorer = origin

        queries = (
            self.q_proj(scorer)
            .view(batch, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(scorer)
            .view(batch, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        logits = (queries @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        dynamic = self.normalizer(logits)

        singleton_static = self.static_adjacency()
        alpha = self.alpha()
        if singleton_static is None:
            selected = dynamic
            base = None
        else:
            if alpha is None:
                raise RuntimeError("Static/dynamic graph mixture has no alpha.")
            base = singleton_static
            expanded = singleton_static.expand(batch, -1, -1, -1)
            alpha_value = alpha.to(dynamic.device, dynamic.dtype).view(1, 1, 1, 1)
            selected = (1.0 - alpha_value) * expanded + alpha_value * dynamic

        output = GraphOutput(
            selected=selected,
            per_layer=(selected,),
            base=base,
            dynamic=dynamic,
            alpha=alpha,
            logits=(logits if base is None else None),
        )
        output.validate(
            batch_size=batch,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class ContinuousTransformerInputBlock(nn.Module):
    """First pure-Transformer temporal block with continuous input embedding."""

    def __init__(
        self,
        *,
        input_channels: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        feedforward_multiplier: int,
        dropout: float,
        relative_position_embedding: bool,
        session_position_encoding: bool,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.relative_position_embedding_enabled = bool(
            relative_position_embedding
        )
        self.session_position_encoding = bool(session_position_encoding)
        self.state_projection = nn.Linear(int(input_channels), self.d_model)
        if self.relative_position_embedding_enabled:
            self.position_embedding: nn.Embedding | None = nn.Embedding(
                self.context_length,
                self.d_model,
            )
        else:
            self.position_embedding = None
        if self.session_position_encoding:
            self.session_projection: nn.Linear | None = nn.Linear(3, self.d_model)
        else:
            self.session_projection = None
        self.input_norm = nn.LayerNorm(self.d_model)
        self.encoder = PerNodeTransformerEncoder(
            d_model=self.d_model,
            config=TemporalConfig(
                type="transformer",
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                feedforward_multiplier=int(feedforward_multiplier),
                dropout=float(dropout),
            ),
        )

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or int(x.shape[1]) != self.context_length:
            raise ValueError("Unexpected continuous Transformer input shape.")
        state = self.state_projection(x)
        hidden = state
        if self.position_embedding is not None:
            ids = torch.arange(self.context_length, device=x.device)
            hidden = hidden + self.position_embedding(ids).view(
                1,
                self.context_length,
                1,
                self.d_model,
            )
        if self.session_projection is not None:
            features = build_context_session_features(
                context_start=context_start,
                session_length=session_length,
                context_length=self.context_length,
                device=x.device,
                dtype=x.dtype,
            )
            hidden = hidden + self.session_projection(features).view(
                int(x.shape[0]),
                self.context_length,
                1,
                self.d_model,
            )
        return self.encoder(self.input_norm(hidden)), state


class TransformerRefinementBlock(nn.Module):
    """One causal per-node Transformer used after an interlaced block."""

    def __init__(
        self,
        *,
        sequence_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        feedforward_multiplier: int,
        dropout: float,
        relative_position_embedding: bool,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        if relative_position_embedding:
            self.position_embedding: nn.Embedding | None = nn.Embedding(
                self.sequence_length,
                self.d_model,
            )
        else:
            self.position_embedding = None
        self.input_norm = nn.LayerNorm(self.d_model)
        self.encoder = PerNodeTransformerEncoder(
            d_model=self.d_model,
            config=TemporalConfig(
                type="transformer",
                num_layers=int(num_layers),
                num_heads=int(num_heads),
                feedforward_multiplier=int(feedforward_multiplier),
                dropout=float(dropout),
            ),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 4:
            raise ValueError("Transformer refinement input must be [B,L,N,D].")
        if (int(hidden.shape[1]), int(hidden.shape[-1])) != (
            self.sequence_length,
            self.d_model,
        ):
            raise ValueError("Unexpected Transformer refinement shape.")
        values = hidden
        if self.position_embedding is not None:
            ids = torch.arange(self.sequence_length, device=hidden.device)
            values = values + self.position_embedding(ids).view(
                1,
                self.sequence_length,
                1,
                self.d_model,
            )
        return self.encoder(self.input_norm(values))


class InterlacedGraphSpatialBlock(nn.Module):
    """Graph inference, state-aware message passing, and learned beta fusion."""

    def __init__(
        self,
        *,
        d_model: int,
        num_nodes: int,
        num_heads: int,
        graph_hidden_dim: int,
        graph_activation: GraphActivation,
        graph_family: GraphFamily,
        static_prior: Tensor | None,
        initial_alpha: float,
        prior_scale: float,
        prior_jitter: float,
        prior_seed: int,
        feedforward_multiplier: int,
        dropout: float,
        initial_beta: float,
    ) -> None:
        super().__init__()
        self.graph_learner = Round2WindowGraphLearner(
            d_model=d_model,
            num_nodes=num_nodes,
            num_heads=num_heads,
            graph_hidden_dim=graph_hidden_dim,
            activation=graph_activation,
            graph_family=graph_family,
            static_prior=static_prior,
            initial_alpha=initial_alpha,
            prior_scale=prior_scale,
            prior_jitter=prior_jitter,
            prior_seed=prior_seed,
        )
        self.spatial_module = StateAwareSpatialMessagePassing(
            d_model=d_model,
            num_heads=num_heads,
            graph_hidden_dim=graph_hidden_dim,
            feedforward_multiplier=feedforward_multiplier,
            dropout=dropout,
            use_state_pathway=(graph_family == "prior_state"),
        )
        self.spatial_gate = SpatialBranchGate(
            gate_type="learned_scalar",
            initial_beta=initial_beta,
        )

    def forward(
        self,
        temporal_hidden: Tensor,
        *,
        state_hidden: Tensor | None,
    ) -> Round2BlockOutput:
        graph = self.graph_learner(
            temporal_hidden,
            state_hidden=state_hidden,
        )
        graph_hidden = self.spatial_module(
            temporal_hidden,
            graph.selected,
            state_hidden=state_hidden,
        )
        fused, beta = self.spatial_gate(temporal_hidden, graph_hidden)
        return Round2BlockOutput(
            temporal_hidden=temporal_hidden,
            state_hidden=state_hidden,
            graph_spatial_hidden=graph_hidden,
            fused_hidden=fused,
            graph=graph,
            alpha=graph.alpha,
            beta=beta,
        )


class ModernTCNGraphRound2Model(nn.Module):
    """Configurable two-to-four-block Round-2 architecture."""

    def __init__(
        self,
        config: ModernTCNGraphRound2Config,
        *,
        static_prior: Tensor | None,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            config.uses_static_graph
            and config.prior_type in {"sector", "correlation"}
            and static_prior is None
        ):
            raise ValueError("Structured prior_state model requires static_prior.")
        if not config.uses_static_graph and static_prior is not None:
            raise ValueError("Dynamic-only model must not receive static_prior.")
        self.config = config
        self.feature_length = int(config.feature_length)

        self.modern_tcn_backbone: ModernTCNContinuousBackbone | None
        self.modern_state_projection: nn.Linear | None
        self.modern_to_transformer: nn.Module | None
        self.transformer_input: ContinuousTransformerInputBlock | None
        self.transformer_state_projection: nn.Linear | None

        if config.temporal_family == "modern_tcn_transformer":
            mtcn_forecaster = ContinuousForecasterConfig(
                num_nodes=config.num_nodes,
                context_length=config.context_length,
                horizons=config.horizons,
                input_channels=config.input_channels,
                target_channel=config.target_channel,
                output_representation="normalised_close",
                output_head_initialisation="default",
                temporal=ContinuousTemporalConfig(
                    type="modern_tcn",
                    d_model=config.modern_tcn_d_model,
                    num_layers=1,
                    num_heads=4,
                    feedforward_multiplier=2,
                    dropout=0.0,
                    relative_position_embedding=False,
                    session_position_encoding=False,
                    patch_size=config.modern_tcn_patch_size,
                    patch_stride=config.modern_tcn_patch_stride,
                    modern_tcn_ffn_ratio=config.modern_tcn_ffn_ratio,
                    modern_tcn_num_blocks=config.modern_tcn_num_blocks,
                    modern_tcn_large_kernel=config.modern_tcn_large_kernel,
                    modern_tcn_small_kernel=config.modern_tcn_small_kernel,
                    modern_tcn_dropout=config.modern_tcn_dropout,
                    modern_tcn_head_dropout=config.modern_tcn_head_dropout,
                ),
                graph=GraphConfig(type="none", num_heads=1),
                spatial_gate_type="none",
            )
            self.modern_tcn_backbone = ModernTCNContinuousBackbone(
                config=mtcn_forecaster
            )
            self.modern_state_projection = (
                nn.Linear(len(config.input_channels), config.modern_tcn_d_model)
                if config.uses_state_pathway
                else None
            )
            self.modern_to_transformer = nn.Sequential(
                nn.Linear(config.modern_tcn_d_model, config.transformer_d_model),
                nn.LayerNorm(config.transformer_d_model),
            )
            self.transformer_input = None
            self.transformer_state_projection = (
                nn.Linear(len(config.input_channels), config.transformer_d_model)
                if config.uses_state_pathway
                else None
            )
        else:
            self.modern_tcn_backbone = None
            self.modern_state_projection = None
            self.modern_to_transformer = None
            self.transformer_input = ContinuousTransformerInputBlock(
                input_channels=len(config.input_channels),
                context_length=config.context_length,
                d_model=config.transformer_d_model,
                num_layers=config.transformer_num_layers,
                num_heads=config.transformer_num_heads,
                feedforward_multiplier=config.transformer_feedforward_multiplier,
                dropout=config.transformer_dropout,
                relative_position_embedding=config.transformer_relative_position_embedding,
                session_position_encoding=config.transformer_session_position_encoding,
            )
            self.transformer_state_projection = None

        self.temporal_refinements = nn.ModuleList()
        refinement_count = (
            config.num_transformer_blocks
            if config.temporal_family == "modern_tcn_transformer"
            else config.num_transformer_blocks - 1
        )
        for _ in range(refinement_count):
            self.temporal_refinements.append(
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
        self.forecast_head = DirectFlattenForecastHead(
            d_model=config.transformer_d_model,
            feature_length=self.feature_length,
            num_horizons=len(config.horizons),
            dropout=config.head_dropout,
            initialisation="default",
        )

    def _patch_state(self, x: Tensor, projection: nn.Linear) -> Tensor:
        minute_state = projection(x)
        return align_state_embeddings_to_modern_tcn_patches(
            minute_state,
            patch_size=self.config.modern_tcn_patch_size,
            patch_stride=self.config.modern_tcn_patch_stride,
        ).contiguous()

    def alphas(self) -> tuple[Tensor | None, ...]:
        return tuple(
            block.graph_learner.alpha() for block in self.graph_spatial_blocks
        )

    def betas(self) -> tuple[Tensor, ...]:
        return tuple(block.spatial_gate.beta() for block in self.graph_spatial_blocks)

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> ModernTCNGraphRound2Output:
        config = self.config
        expected = (
            config.context_length,
            config.num_nodes,
            len(config.input_channels),
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x has shape {tuple(x.shape)}; expected [B,{expected[0]},"
                f"{expected[1]},{expected[2]}]."
            )

        outputs: list[Round2BlockOutput] = []
        if config.temporal_family == "modern_tcn_transformer":
            if self.modern_tcn_backbone is None:
                raise RuntimeError("ModernTCN backbone is missing.")
            temporal = self.modern_tcn_backbone(
                x,
                context_start=context_start,
                session_length=session_length,
            )
            state32 = (
                None
                if self.modern_state_projection is None
                else self._patch_state(x, self.modern_state_projection)
            )
            block0 = self.graph_spatial_blocks[0](
                temporal,
                state_hidden=state32,
            )
            outputs.append(block0)
            if self.modern_to_transformer is None:
                raise RuntimeError("ModernTCN-to-Transformer adapter is missing.")
            hidden = self.modern_to_transformer(block0.fused_hidden)
            state96 = (
                None
                if self.transformer_state_projection is None
                else self._patch_state(x, self.transformer_state_projection)
            )
            for refinement_index, temporal_block in enumerate(
                self.temporal_refinements,
                start=1,
            ):
                temporal = temporal_block(hidden)
                block = self.graph_spatial_blocks[refinement_index](
                    temporal,
                    state_hidden=state96,
                )
                outputs.append(block)
                hidden = block.fused_hidden
        else:
            if self.transformer_input is None:
                raise RuntimeError("Transformer input block is missing.")
            temporal, state96 = self.transformer_input(
                x,
                context_start=context_start,
                session_length=session_length,
            )
            state_for_graph = state96 if config.uses_state_pathway else None
            first = self.graph_spatial_blocks[0](
                temporal,
                state_hidden=state_for_graph,
            )
            outputs.append(first)
            hidden = first.fused_hidden
            for refinement_index, temporal_block in enumerate(
                self.temporal_refinements,
                start=1,
            ):
                temporal = temporal_block(hidden)
                block = self.graph_spatial_blocks[refinement_index](
                    temporal,
                    state_hidden=state_for_graph,
                )
                outputs.append(block)
                hidden = block.fused_hidden

        predictions = self.forecast_head(hidden)
        result = ModernTCNGraphRound2Output(
            predictions=predictions,
            block_outputs=tuple(outputs),
            final_hidden=hidden,
        )
        result.validate(config)
        return result

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
        if self.config.temporal_family == "modern_tcn_transformer":
            return (
                self.modern_state_projection,
                *([self.transformer_state_projection] * self.config.num_transformer_blocks),
            )
        if self.transformer_input is None:
            raise RuntimeError("Transformer input block is missing.")
        return tuple(
            [self.transformer_input.state_projection] * self.config.num_st_blocks
        )


def round2_model_config_from_mapping(
    values: Mapping[str, Any],
    *,
    num_nodes: int,
) -> ModernTCNGraphRound2Config:
    data = values["data"]
    model = values["model"]
    temporal = model["temporal_stack"]
    modern = temporal["modern_tcn"]
    transformer = temporal["transformer"]
    graph = model["graph"]
    spatial = model["spatial"]
    prior = model["prior"]
    config = ModernTCNGraphRound2Config(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
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
        transformer_session_position_encoding=bool(
            transformer["session_position_encoding"]
        ),
        graph_family=str(model["graph_family"]),
        prior_type=str(prior["type"]),
        graph_heads_per_block=tuple(int(value) for value in graph["num_heads_per_block"]),
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
        head_dropout=float(model["head_dropout"]),
    )
    config.validate()
    return config

from __future__ import annotations

"""Round-1 ModernTCN graph ablations.

This module preserves the selected ModernTCN temporal/forecasting path and
adds only three controlled graph variants:

1. dynamic-only graph (the current best architecture);
2. trainable prior-initialised static graph mixed with a dynamic graph;
3. the same mixture with the current continuous state exposed directly to
   both graph scoring and graph-weighted value propagation.

All graph tensors use the project convention ``A[target, source]``.
"""

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn

from src.models.continuous_forecaster import (
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
    ModernTCNContinuousBackbone,
    SpatialBranchGate,
)
from src.models.dynamic_graph.contracts import GraphConfig, GraphOutput
from src.models.dynamic_graph.modules import GraphNormalizer


Round1GraphVariant = Literal[
    "dynamic_only",
    "prior_mixture",
    "prior_mixture_state",
]


@dataclass(frozen=True)
class ModernTCNGraphRound1Config:
    forecaster: ContinuousForecasterConfig
    graph_variant: Round1GraphVariant = "dynamic_only"
    prior_scale: float = 4.0
    prior_jitter: float = 0.02
    prior_seed: int = 42

    def validate(self) -> None:
        self.forecaster.validate()
        if self.forecaster.temporal.type != "modern_tcn":
            raise ValueError("Round 1 requires the ModernTCN temporal backbone.")
        if self.forecaster.graph.activation != "softmax":
            raise ValueError("Round 1 uses softmax graphs only.")
        if self.forecaster.graph.add_self_loops:
            raise ValueError("Round 1 excludes graph self-edges.")
        if self.forecaster.spatial_gate_type != "learned_scalar":
            raise ValueError("Round 1 requires the learned scalar beta gate.")
        if self.graph_variant not in {
            "dynamic_only",
            "prior_mixture",
            "prior_mixture_state",
        }:
            raise ValueError(f"Unsupported graph variant {self.graph_variant!r}.")
        if not math.isfinite(float(self.prior_scale)) or self.prior_scale <= 0:
            raise ValueError("prior_scale must be finite and positive.")
        if not math.isfinite(float(self.prior_jitter)) or self.prior_jitter < 0:
            raise ValueError("prior_jitter must be finite and non-negative.")

    @property
    def uses_static_prior(self) -> bool:
        return self.graph_variant in {"prior_mixture", "prior_mixture_state"}

    @property
    def uses_state_pathway(self) -> bool:
        return self.graph_variant == "prior_mixture_state"


@dataclass
class ModernTCNGraphRound1Output:
    predictions: Tensor
    temporal_hidden: Tensor
    state_hidden: Tensor | None
    graph_spatial_hidden: Tensor
    fused_hidden: Tensor
    graph: GraphOutput
    alpha: Tensor | None
    beta: Tensor

    def validate(self, config: ModernTCNGraphRound1Config) -> None:
        forecaster = config.forecaster
        batch = int(self.temporal_hidden.shape[0])
        expected_prediction = (
            batch,
            len(forecaster.horizons),
            forecaster.num_nodes,
            1,
        )
        if tuple(self.predictions.shape) != expected_prediction:
            raise ValueError(
                f"predictions has shape {tuple(self.predictions.shape)}; "
                f"expected {expected_prediction}."
            )
        if self.temporal_hidden.ndim != 4:
            raise ValueError("temporal_hidden must have shape [B,L,N,D].")
        if self.graph_spatial_hidden.shape != self.temporal_hidden.shape:
            raise ValueError("graph_spatial_hidden must match temporal_hidden.")
        if self.fused_hidden.shape != self.temporal_hidden.shape:
            raise ValueError("fused_hidden must match temporal_hidden.")
        if self.state_hidden is not None and self.state_hidden.shape != self.temporal_hidden.shape:
            raise ValueError("state_hidden must match temporal_hidden.")
        if self.beta.numel() != 1:
            raise ValueError("beta must be scalar.")
        if self.alpha is not None and self.alpha.numel() != 1:
            raise ValueError("Round-1 alpha must be scalar.")
        self.graph.validate(
            batch_size=batch,
            num_heads=forecaster.graph.num_heads,
            num_nodes=forecaster.num_nodes,
        )


def build_v2_prior_logits(
    prior: Tensor,
    *,
    num_heads: int,
    scale: float,
    jitter: float,
    seed: int,
) -> Tensor:
    """Return Dimitri-V2-style initial base logits."""

    values = torch.as_tensor(prior).detach().cpu().float()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("prior must have square shape [N,N].")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("prior must be finite and non-negative.")
    if int(num_heads) <= 0:
        raise ValueError("num_heads must be positive.")
    if float(values.max().item()) <= 0:
        raise ValueError("prior must contain positive mass.")

    normalised = values / values.max().clamp_min(1.0e-6)
    base = float(scale) * (normalised - normalised.mean())
    result = base.unsqueeze(0).expand(int(num_heads), -1, -1).clone()
    if jitter:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        result += torch.randn(
            result.shape,
            generator=generator,
            dtype=result.dtype,
        ) * float(jitter)
    return result.contiguous()


class PriorMixedDynamicGraphLearner(nn.Module):
    """Dynamic graph with an optional trainable prior and direct alpha mix."""

    def __init__(
        self,
        *,
        d_model: int,
        num_nodes: int,
        num_heads: int,
        graph_hidden_dim: int,
        use_state_pathway: bool,
        static_prior: Tensor | None,
        initial_alpha: float,
        prior_scale: float,
        prior_jitter: float,
        prior_seed: int,
    ) -> None:
        super().__init__()
        if graph_hidden_dim % num_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_heads.")
        if not 0.0 <= float(initial_alpha) <= 1.0:
            raise ValueError("initial_alpha must lie in [0,1].")

        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        self.use_state_pathway = bool(use_state_pathway)
        scorer_dim = self.d_model * (2 if self.use_state_pathway else 1)

        self.q_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.k_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.normalizer = GraphNormalizer(
            GraphConfig(
                type="dynamic",
                num_heads=self.num_heads,
                hidden_dim=self.graph_hidden_dim,
                activation="softmax",
                add_self_loops=False,
            )
        )

        if static_prior is None:
            self.register_parameter("static_logits", None)
            self.register_parameter("raw_alpha", None)
        else:
            prior_values = torch.as_tensor(static_prior).detach().cpu().float()
            if tuple(prior_values.shape) != (self.num_nodes, self.num_nodes):
                raise ValueError("static_prior node axes differ from the configured graph.")
            initial_logits = build_v2_prior_logits(
                prior_values,
                num_heads=self.num_heads,
                scale=prior_scale,
                jitter=prior_jitter,
                seed=prior_seed,
            )
            self.static_logits = nn.Parameter(initial_logits)
            epsilon = 1.0e-6
            clipped = min(max(float(initial_alpha), epsilon), 1.0 - epsilon)
            raw = math.log(clipped / (1.0 - clipped))
            self.raw_alpha = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def alpha(self) -> Tensor | None:
        return None if self.raw_alpha is None else torch.sigmoid(self.raw_alpha)

    def static_adjacency(self) -> Tensor | None:
        if self.static_logits is None:
            return None
        return self.normalizer(self.static_logits.unsqueeze(0).float())

    def forward(
        self,
        temporal_hidden: Tensor,
        *,
        state_hidden: Tensor | None = None,
    ) -> GraphOutput:
        if temporal_hidden.ndim != 4:
            raise ValueError("temporal_hidden must have shape [B,L,N,D].")
        batch, _, nodes, hidden = map(int, temporal_hidden.shape)
        if (nodes, hidden) != (self.num_nodes, self.d_model):
            raise ValueError("temporal_hidden does not match the graph config.")

        origin = temporal_hidden[:, -1]
        if self.use_state_pathway:
            if state_hidden is None or state_hidden.shape != temporal_hidden.shape:
                raise ValueError("State-aware graph learner requires matching state_hidden.")
            scorer_input = torch.cat([origin, state_hidden[:, -1]], dim=-1)
        else:
            if state_hidden is not None:
                raise ValueError("Hidden-only graph learner should not receive state_hidden.")
            scorer_input = origin

        queries = (
            self.q_proj(scorer_input)
            .view(batch, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(scorer_input)
            .view(batch, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        dynamic_logits = (queries @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        dynamic = self.normalizer(dynamic_logits.float())

        singleton_static = self.static_adjacency()
        alpha = self.alpha()
        if singleton_static is None:
            selected = dynamic
            base = None
        else:
            base = singleton_static
            expanded = singleton_static.expand(batch, -1, -1, -1)
            alpha_view = alpha.to(dynamic.device, dynamic.dtype).view(1, 1, 1, 1)
            selected = (1.0 - alpha_view) * expanded + alpha_view * dynamic

        output = GraphOutput(
            selected=selected,
            per_layer=(selected,),
            base=base,
            dynamic=dynamic,
            alpha=alpha,
            logits=(dynamic_logits.float() if singleton_static is None else None),
        )
        output.validate(
            batch_size=batch,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class StateAwareSpatialMessagePassing(nn.Module):
    """Dimitri-style values with optional state concatenation."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        graph_hidden_dim: int,
        feedforward_multiplier: int,
        dropout: float,
        use_state_pathway: bool,
    ) -> None:
        super().__init__()
        if graph_hidden_dim % num_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        self.use_state_pathway = bool(use_state_pathway)
        value_input_dim = self.d_model * (2 if self.use_state_pathway else 1)

        self.value_projection = nn.Linear(value_input_dim, self.graph_hidden_dim)
        self.output_projection = nn.Linear(self.graph_hidden_dim, self.d_model)
        self.message_dropout = nn.Dropout(float(dropout))
        self.mix_norm = nn.LayerNorm(self.d_model)
        self.feedforward_norm = nn.LayerNorm(self.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(self.d_model, int(feedforward_multiplier) * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(feedforward_multiplier) * self.d_model, self.d_model),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        temporal_hidden: Tensor,
        adjacency: Tensor,
        *,
        state_hidden: Tensor | None = None,
    ) -> Tensor:
        if temporal_hidden.ndim != 4:
            raise ValueError("temporal_hidden must have shape [B,L,N,D].")
        batch, steps, nodes, hidden = map(int, temporal_hidden.shape)
        if hidden != self.d_model:
            raise ValueError("Unexpected temporal hidden dimension.")
        expected_graph = (batch, self.num_heads, nodes, nodes)
        if tuple(adjacency.shape) != expected_graph:
            raise ValueError(
                f"adjacency has shape {tuple(adjacency.shape)}; expected {expected_graph}."
            )

        if self.use_state_pathway:
            if state_hidden is None or state_hidden.shape != temporal_hidden.shape:
                raise ValueError("State-aware spatial values require matching state_hidden.")
            value_input = torch.cat([temporal_hidden, state_hidden], dim=-1)
        else:
            if state_hidden is not None:
                raise ValueError("Hidden-only spatial values should not receive state_hidden.")
            value_input = temporal_hidden

        values = (
            self.value_projection(value_input)
            .view(batch, steps, nodes, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        messages = torch.einsum(
            "bgij,btgjd->btgid",
            adjacency.to(device=values.device, dtype=values.dtype),
            values,
        )
        messages = (
            messages.permute(0, 1, 3, 2, 4)
            .reshape(batch, steps, nodes, self.graph_hidden_dim)
        )
        projected = self.output_projection(messages)
        mixed = self.mix_norm(temporal_hidden + self.message_dropout(projected))
        return self.feedforward_norm(mixed + self.feedforward(mixed))


def align_state_embeddings_to_modern_tcn_patches(
    state_embeddings: Tensor,
    *,
    patch_size: int,
    patch_stride: int,
) -> Tensor:
    """Select the current state at every ModernTCN patch endpoint.

    Dimitri's V2 scorer and spatial values receive the raw current state at
    the same temporal position as the contextual hidden state. ModernTCN
    represents overlapping patches rather than individual minutes, so the
    causal analogue is the final observed state in each patch. The standard
    final-value padding makes the final patch endpoint equal to the forecast
    origin.
    """

    if state_embeddings.ndim != 4:
        raise ValueError("state_embeddings must have shape [B,T,N,D].")
    if patch_size < patch_stride or patch_stride <= 0:
        raise ValueError("Invalid patch_size/patch_stride.")
    padding = int(patch_size) - int(patch_stride)
    values = state_embeddings
    if padding:
        values = torch.cat(
            [values, values[:, -1:].expand(-1, padding, -1, -1)],
            dim=1,
        )
    return values.unfold(1, int(patch_size), int(patch_stride))[..., -1]


class ModernTCNGraphRound1Model(nn.Module):
    """Selected ModernTCN backbone with one configurable graph/spatial block."""

    def __init__(
        self,
        config: ModernTCNGraphRound1Config,
        *,
        static_prior: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.uses_static_prior and static_prior is None:
            raise ValueError("A prior-mixture model requires static_prior.")
        if not config.uses_static_prior and static_prior is not None:
            raise ValueError("The dynamic-only control must not receive a prior.")

        self.config = config
        forecaster = config.forecaster
        self.temporal_backbone = ModernTCNContinuousBackbone(config=forecaster)
        if config.uses_state_pathway:
            self.state_projection: nn.Linear | None = nn.Linear(
                len(forecaster.input_channels), forecaster.temporal.d_model
            )
        else:
            self.state_projection = None

        self.graph_learner = PriorMixedDynamicGraphLearner(
            d_model=forecaster.temporal.d_model,
            num_nodes=forecaster.num_nodes,
            num_heads=forecaster.graph.num_heads,
            graph_hidden_dim=forecaster.graph.hidden_dim,
            use_state_pathway=config.uses_state_pathway,
            static_prior=(static_prior if config.uses_static_prior else None),
            initial_alpha=forecaster.graph.initial_alpha,
            prior_scale=config.prior_scale,
            prior_jitter=config.prior_jitter,
            prior_seed=config.prior_seed,
        )
        self.spatial_module = StateAwareSpatialMessagePassing(
            d_model=forecaster.temporal.d_model,
            num_heads=forecaster.graph.num_heads,
            graph_hidden_dim=forecaster.graph.hidden_dim,
            feedforward_multiplier=forecaster.spatial_feedforward_multiplier,
            dropout=forecaster.spatial_dropout,
            use_state_pathway=config.uses_state_pathway,
        )
        self.spatial_gate = SpatialBranchGate(
            gate_type=forecaster.spatial_gate_type,
            initial_beta=forecaster.spatial_gate_initial_beta,
        )
        self.temporal_backbone.initialise_forecast_head(
            forecaster.output_head_initialisation
        )

    def alpha(self) -> Tensor | None:
        return self.graph_learner.alpha()

    def beta(self) -> Tensor:
        return self.spatial_gate.beta()

    def _state_hidden(self, x: Tensor) -> Tensor | None:
        if self.state_projection is None:
            return None
        minute_states = self.state_projection(x)
        temporal = self.config.forecaster.temporal
        return align_state_embeddings_to_modern_tcn_patches(
            minute_states,
            patch_size=temporal.patch_size,
            patch_stride=temporal.patch_stride,
        ).contiguous()

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> ModernTCNGraphRound1Output:
        forecaster = self.config.forecaster
        if tuple(x.shape[1:]) != (
            forecaster.context_length,
            forecaster.num_nodes,
            len(forecaster.input_channels),
        ):
            raise ValueError("x does not match the configured [T,N,C] axes.")

        temporal_hidden = self.temporal_backbone(
            x,
            context_start=context_start,
            session_length=session_length,
        )
        state_hidden = self._state_hidden(x)
        if state_hidden is not None and state_hidden.shape != temporal_hidden.shape:
            raise RuntimeError(
                "ModernTCN state-patch alignment differs from temporal features."
            )
        graph = self.graph_learner(
            temporal_hidden,
            state_hidden=state_hidden,
        )
        graph_hidden = self.spatial_module(
            temporal_hidden,
            graph.selected,
            state_hidden=state_hidden,
        )
        fused_hidden, beta = self.spatial_gate(temporal_hidden, graph_hidden)
        predictions = self.temporal_backbone.forecast(fused_hidden)

        output = ModernTCNGraphRound1Output(
            predictions=predictions,
            temporal_hidden=temporal_hidden,
            state_hidden=state_hidden,
            graph_spatial_hidden=graph_hidden,
            fused_hidden=fused_hidden,
            graph=graph,
            alpha=graph.alpha,
            beta=beta,
        )
        output.validate(self.config)
        return output

    def graph_parameter_ids(self) -> set[int]:
        return {
            id(parameter)
            for parameter in self.graph_learner.parameters()
            if parameter.requires_grad
        }


def graph_component_summary(values: Tensor | None) -> dict[str, float | None]:
    if values is None:
        return {
            "mean_row_entropy": None,
            "mean_effective_neighbours": None,
            "mean_diagonal_weight": None,
            "maximum_edge_weight": None,
        }
    graph = torch.as_tensor(values).detach().float().clamp_min(1.0e-12)
    entropy = -(graph * graph.log()).sum(dim=-1)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
    return {
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_diagonal_weight": float(diagonal.mean().item()),
        "maximum_edge_weight": float(graph.max().item()),
    }


def round1_model_config_from_mapping(
    values: Mapping[str, Any],
    *,
    num_nodes: int,
) -> ModernTCNGraphRound1Config:
    """Build the typed Round-1 model config from the saved experiment JSON."""

    data = values["data"]
    model = values["model"]
    temporal = model["temporal"]
    graph = model["graph"]
    spatial = model["spatial"]
    prior = model["prior"]

    forecaster = ContinuousForecasterConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=tuple(int(item) for item in data["horizons"]),
        input_channels=tuple(str(item) for item in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        output_representation="normalised_close",
        output_head_initialisation="default",
        temporal=ContinuousTemporalConfig(
            type="modern_tcn",
            d_model=int(temporal["d_model"]),
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
            relative_position_embedding=False,
            session_position_encoding=bool(temporal["session_position_encoding"]),
            patch_size=int(temporal["patch_size"]),
            patch_stride=int(temporal["patch_stride"]),
            modern_tcn_ffn_ratio=int(temporal["ffn_ratio"]),
            modern_tcn_num_blocks=int(temporal["num_blocks"]),
            modern_tcn_large_kernel=int(temporal["large_kernel"]),
            modern_tcn_small_kernel=int(temporal["small_kernel"]),
            modern_tcn_dropout=float(temporal["dropout"]),
            modern_tcn_head_dropout=float(temporal["head_dropout"]),
        ),
        graph=GraphConfig(
            type="dynamic",
            num_heads=int(graph["num_heads"]),
            hidden_dim=int(graph["hidden_dim"]),
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=min(4, int(num_nodes) - 1),
            base_graph_type="free_static",
            gate_type=(
                "none" if str(model["variant"]) == "dynamic_only" else "learned_scalar"
            ),
            initial_alpha=float(graph["initial_alpha"]),
        ),
        spatial_num_layers=1,
        spatial_feedforward_multiplier=int(spatial["feedforward_multiplier"]),
        spatial_dropout=float(spatial["dropout"]),
        spatial_gate_type="learned_scalar",
        spatial_gate_initial_beta=float(spatial["initial_beta"]),
        head_dropout=0.0,
    )
    return ModernTCNGraphRound1Config(
        forecaster=forecaster,
        graph_variant=str(model["variant"]),
        prior_scale=float(prior["scale"]),
        prior_jitter=float(prior["jitter"]),
        prior_seed=int(prior["seed"]),
    )

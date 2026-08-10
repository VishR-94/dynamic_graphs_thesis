from __future__ import annotations

"""Stacked causal-Transformer ST blocks for dense five-horizon supervision.

This module is the final depth/capacity diagnostic built after the one-block
``TransformerDenseParallelGraphModel`` experiment.  Every block follows the
same causal interlaced contract:

    per-node causal Transformer
    -> state-aware dynamic graph + trainable random static graph
    -> convex alpha graph mixture
    -> state-aware spatial message passing
    -> learned beta temporal/spatial mixture

A direct five-horizon head is applied at every minute after the final ST block.
During public inference only the final minute is retained.  Graph orientation
is always ``A[target, source]``.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn

from src.models.continuous_forecaster import SpatialBranchGate
from src.models.dynamic_graph.contracts import GraphConfig, TemporalConfig
from src.models.dynamic_graph.modules import GraphNormalizer, PerNodeTransformerEncoder


GRAPH_ORIENTATION = "A[target, source]"


@dataclass(frozen=True)
class DenseTransformerDepthConfig:
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

    num_st_blocks: int = 1
    d_model: int = 64
    transformer_num_layers: int = 1
    transformer_num_heads: int = 4
    transformer_feedforward_multiplier: int = 2
    transformer_dropout: float = 0.0
    position_embedding: bool = False

    graph_heads_per_block: tuple[int, ...] = (1,)
    graph_hidden_dims_per_block: tuple[int, ...] = (64,)
    graph_activations_per_block: tuple[str, ...] = ("sparsemax",)
    graph_initial_alpha: float = 0.5
    random_static_jitter: float = 0.02
    graph_seed: int = 42

    spatial_initial_beta: float = 0.5
    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0

    def validate(self) -> None:
        if int(self.num_nodes) <= 1:
            raise ValueError("num_nodes must be greater than one.")
        if int(self.context_length) <= 0:
            raise ValueError("context_length must be positive.")
        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if tuple(sorted(set(int(value) for value in self.horizons))) != tuple(
            int(value) for value in self.horizons
        ):
            raise ValueError("horizons must be unique and increasing.")
        if any(int(value) <= 0 for value in self.horizons):
            raise ValueError("Every horizon must be positive.")
        if not self.input_channels or self.target_channel not in self.input_channels:
            raise ValueError("target_channel must occur in input_channels.")
        if int(self.num_st_blocks) <= 0:
            raise ValueError("num_st_blocks must be positive.")
        if int(self.d_model) <= 0:
            raise ValueError("d_model must be positive.")
        if int(self.transformer_num_layers) <= 0:
            raise ValueError("transformer_num_layers must be positive.")
        if int(self.transformer_num_heads) <= 0:
            raise ValueError("transformer_num_heads must be positive.")
        if int(self.d_model) % int(self.transformer_num_heads):
            raise ValueError("d_model must be divisible by Transformer heads.")
        if int(self.transformer_feedforward_multiplier) <= 0:
            raise ValueError("Transformer FF multiplier must be positive.")
        if not 0.0 <= float(self.transformer_dropout) < 1.0:
            raise ValueError("Transformer dropout must lie in [0,1).")
        if not 0.0 <= float(self.spatial_dropout) < 1.0:
            raise ValueError("Spatial dropout must lie in [0,1).")
        if not 0.0 < float(self.graph_initial_alpha) < 1.0:
            raise ValueError("graph_initial_alpha must lie strictly in (0,1).")
        if not 0.0 < float(self.spatial_initial_beta) < 1.0:
            raise ValueError("spatial_initial_beta must lie strictly in (0,1).")
        if not math.isfinite(float(self.random_static_jitter)) or float(
            self.random_static_jitter
        ) < 0.0:
            raise ValueError("random_static_jitter must be finite and non-negative.")

        schedules: dict[str, Sequence[object]] = {
            "graph_heads_per_block": self.graph_heads_per_block,
            "graph_hidden_dims_per_block": self.graph_hidden_dims_per_block,
            "graph_activations_per_block": self.graph_activations_per_block,
        }
        for name, values in schedules.items():
            if len(values) != int(self.num_st_blocks):
                raise ValueError(
                    f"{name} has length {len(values)}; "
                    f"expected {self.num_st_blocks}."
                )
        for index, (heads, hidden) in enumerate(
            zip(
                self.graph_heads_per_block,
                self.graph_hidden_dims_per_block,
                strict=True,
            )
        ):
            if int(heads) <= 0 or int(hidden) <= 0:
                raise ValueError(f"Block {index} graph width must be positive.")
            if int(hidden) % int(heads):
                raise ValueError(
                    f"Block {index}: graph_hidden_dim={hidden} is not "
                    f"divisible by graph_heads={heads}."
                )
        allowed = {"softmax", "sparsemax"}
        if any(value not in allowed for value in self.graph_activations_per_block):
            raise ValueError("Only softmax and sparsemax are supported here.")
        if self.graph_activations_per_block[-1] != "sparsemax":
            raise ValueError("The final graph block must use sparsemax.")
        if any(
            value != "softmax" for value in self.graph_activations_per_block[:-1]
        ):
            raise ValueError("Every non-final graph block must use softmax.")


@dataclass
class DenseLayerGraphOutput:
    selected: Tensor  # [B,T,G,N,N]
    dynamic: Tensor  # [B,T,G,N,N]
    base: Tensor  # [1,G,N,N]
    alpha: Tensor
    logits: Tensor  # [B,T,G,N,N]


@dataclass
class DenseTransformerSTBlockOutput:
    temporal_hidden: Tensor  # [B,T,N,D]
    state_hidden: Tensor  # [B,T,N,D]
    graph_spatial_hidden: Tensor  # [B,T,N,D]
    fused_hidden: Tensor  # [B,T,N,D]
    graph: DenseLayerGraphOutput
    beta: Tensor


@dataclass
class DenseTransformerDepthSequenceOutput:
    predictions: Tensor  # [B,T,H,N,1]
    block_outputs: tuple[DenseTransformerSTBlockOutput, ...]

    def final_predictions(self) -> Tensor:
        return self.predictions[:, -1].contiguous()


class DenseRandomStaticDynamicGraphLearner(nn.Module):
    """Per-minute state-aware graph with trainable random static logits."""

    def __init__(
        self,
        *,
        d_model: int,
        num_nodes: int,
        num_heads: int,
        graph_hidden_dim: int,
        activation: str,
        initial_alpha: float,
        random_static_jitter: float,
        seed: int,
    ) -> None:
        super().__init__()
        if int(graph_hidden_dim) % int(num_heads):
            raise ValueError("graph_hidden_dim must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_nodes = int(num_nodes)
        self.num_heads = int(num_heads)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        self.activation = str(activation)

        scorer_dim = 2 * self.d_model
        self.q_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.k_proj = nn.Linear(scorer_dim, self.graph_hidden_dim)
        self.normalizer = GraphNormalizer(
            GraphConfig(
                type="dynamic",
                num_heads=self.num_heads,
                hidden_dim=self.graph_hidden_dim,
                activation=self.activation,
                add_self_loops=False,
            )
        )

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial_logits = torch.randn(
            self.num_heads,
            self.num_nodes,
            self.num_nodes,
            generator=generator,
            dtype=torch.float32,
        ) * float(random_static_jitter)
        self.static_logits = nn.Parameter(initial_logits.contiguous())
        alpha = float(initial_alpha)
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(alpha / (1.0 - alpha)), dtype=torch.float32)
        )

    def alpha(self) -> Tensor:
        return torch.sigmoid(self.raw_alpha)

    def static_adjacency(self) -> Tensor:
        return self.normalizer(self.static_logits.unsqueeze(0))

    def forward(self, temporal: Tensor, state: Tensor) -> DenseLayerGraphOutput:
        if temporal.ndim != 4 or temporal.shape != state.shape:
            raise ValueError("temporal/state must match [B,T,N,D].")
        batch, steps, nodes, hidden = map(int, temporal.shape)
        if (nodes, hidden) != (self.num_nodes, self.d_model):
            raise ValueError("temporal/state axes differ from graph config.")

        scorer = torch.cat([temporal, state], dim=-1)
        flat = scorer.reshape(batch * steps, nodes, 2 * hidden)
        queries = (
            self.q_proj(flat)
            .view(batch * steps, nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(flat)
            .view(batch * steps, nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        logits_flat = (queries @ keys.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        dynamic_flat = self.normalizer(logits_flat)
        dynamic = dynamic_flat.reshape(
            batch,
            steps,
            self.num_heads,
            nodes,
            nodes,
        )
        logits = logits_flat.reshape_as(dynamic)

        base = self.static_adjacency()
        expanded_base = base.view(1, 1, self.num_heads, nodes, nodes).expand(
            batch,
            steps,
            self.num_heads,
            nodes,
            nodes,
        )
        alpha = self.alpha().to(dynamic.device, dynamic.dtype)
        selected = (1.0 - alpha) * expanded_base + alpha * dynamic
        return DenseLayerGraphOutput(
            selected=selected,
            dynamic=dynamic,
            base=base,
            alpha=alpha,
            logits=logits,
        )


class DenseStateAwareSpatialMessagePassing(nn.Module):
    """Use a different supplied graph at every sequence position."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        graph_hidden_dim: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(graph_hidden_dim) % int(num_heads):
            raise ValueError("graph_hidden_dim must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        self.value_projection = nn.Linear(2 * self.d_model, self.graph_hidden_dim)
        self.output_projection = nn.Linear(self.graph_hidden_dim, self.d_model)
        self.message_dropout = nn.Dropout(float(dropout))
        self.mix_norm = nn.LayerNorm(self.d_model)
        self.feedforward_norm = nn.LayerNorm(self.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(
                self.d_model,
                int(feedforward_multiplier) * self.d_model,
            ),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                int(feedforward_multiplier) * self.d_model,
                self.d_model,
            ),
            nn.Dropout(float(dropout)),
        )

    def forward(self, temporal: Tensor, adjacency: Tensor, state: Tensor) -> Tensor:
        if temporal.ndim != 4 or temporal.shape != state.shape:
            raise ValueError("temporal/state must match [B,T,N,D].")
        batch, steps, nodes, hidden = map(int, temporal.shape)
        expected = (batch, steps, self.num_heads, nodes, nodes)
        if tuple(adjacency.shape) != expected:
            raise ValueError(
                f"adjacency has shape {tuple(adjacency.shape)}; expected {expected}."
            )
        values = (
            self.value_projection(torch.cat([temporal, state], dim=-1))
            .view(batch, steps, nodes, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        messages = torch.einsum(
            "btgij,btgjd->btgid",
            adjacency.to(device=values.device, dtype=values.dtype),
            values,
        )
        messages = (
            messages.permute(0, 1, 3, 2, 4)
            .reshape(batch, steps, nodes, self.graph_hidden_dim)
        )
        projected = self.output_projection(messages)
        mixed = self.mix_norm(temporal + self.message_dropout(projected))
        return self.feedforward_norm(mixed + self.feedforward(mixed))


class DenseTransformerSTBlock(nn.Module):
    def __init__(
        self,
        *,
        config: DenseTransformerDepthConfig,
        block_index: int,
    ) -> None:
        super().__init__()
        self.block_index = int(block_index)
        self.input_norm = nn.LayerNorm(int(config.d_model))
        self.temporal_encoder = PerNodeTransformerEncoder(
            d_model=int(config.d_model),
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
        self.graph_learner = DenseRandomStaticDynamicGraphLearner(
            d_model=int(config.d_model),
            num_nodes=int(config.num_nodes),
            num_heads=int(config.graph_heads_per_block[self.block_index]),
            graph_hidden_dim=int(
                config.graph_hidden_dims_per_block[self.block_index]
            ),
            activation=str(
                config.graph_activations_per_block[self.block_index]
            ),
            initial_alpha=float(config.graph_initial_alpha),
            random_static_jitter=float(config.random_static_jitter),
            seed=int(config.graph_seed) + self.block_index * 1009,
        )
        self.spatial_module = DenseStateAwareSpatialMessagePassing(
            d_model=int(config.d_model),
            num_heads=int(config.graph_heads_per_block[self.block_index]),
            graph_hidden_dim=int(
                config.graph_hidden_dims_per_block[self.block_index]
            ),
            feedforward_multiplier=int(config.spatial_feedforward_multiplier),
            dropout=float(config.spatial_dropout),
        )
        self.spatial_gate = SpatialBranchGate(
            gate_type="learned_scalar",
            initial_beta=float(config.spatial_initial_beta),
        )

    def forward(self, hidden: Tensor, state: Tensor) -> DenseTransformerSTBlockOutput:
        temporal = self.temporal_encoder(self.input_norm(hidden))
        graph = self.graph_learner(temporal, state)
        spatial = self.spatial_module(temporal, graph.selected, state)
        fused, beta = self.spatial_gate(temporal, spatial)
        return DenseTransformerSTBlockOutput(
            temporal_hidden=temporal,
            state_hidden=state,
            graph_spatial_hidden=spatial,
            fused_hidden=fused,
            graph=graph,
            beta=beta,
        )


class StackedDenseTransformerGraphModel(nn.Module):
    """One-to-four interlaced dense Transformer ST blocks."""

    def __init__(self, config: DenseTransformerDepthConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.state_projection = nn.Linear(
            len(config.input_channels),
            int(config.d_model),
        )
        if bool(config.position_embedding):
            self.position_embedding: nn.Embedding | None = nn.Embedding(
                int(config.context_length),
                int(config.d_model),
            )
        else:
            self.position_embedding = None
        self.blocks = nn.ModuleList(
            [
                DenseTransformerSTBlock(config=config, block_index=index)
                for index in range(int(config.num_st_blocks))
            ]
        )
        self.forecast_head = nn.Linear(
            int(config.d_model),
            len(config.horizons),
        )

    def alphas(self) -> tuple[Tensor, ...]:
        return tuple(block.graph_learner.alpha() for block in self.blocks)

    def betas(self) -> tuple[Tensor, ...]:
        return tuple(block.spatial_gate.beta() for block in self.blocks)

    def graph_parameter_ids(self) -> set[int]:
        values: set[int] = set()
        for block in self.blocks:
            values.update(
                id(parameter)
                for parameter in block.graph_learner.parameters()
                if parameter.requires_grad
            )
        return values

    def initial_base_graphs(self) -> tuple[Tensor, ...]:
        return tuple(
            block.graph_learner.static_adjacency().detach().cpu().float()
            for block in self.blocks
        )

    def forward_dense(
        self,
        x: Tensor,
        *,
        context_start: Tensor | None = None,
        session_length: Tensor | None = None,
    ) -> DenseTransformerDepthSequenceOutput:
        del context_start, session_length
        expected_input = (
            int(x.shape[0]),
            int(self.config.context_length),
            int(self.config.num_nodes),
            len(self.config.input_channels),
        )
        if tuple(x.shape) != expected_input:
            raise ValueError(
                f"x has shape {tuple(x.shape)}; expected {expected_input}."
            )
        state = self.state_projection(x)
        hidden = state
        if self.position_embedding is not None:
            ids = torch.arange(int(self.config.context_length), device=x.device)
            hidden = hidden + self.position_embedding(ids).view(
                1,
                int(self.config.context_length),
                1,
                int(self.config.d_model),
            )

        outputs: list[DenseTransformerSTBlockOutput] = []
        for block in self.blocks:
            block_output = block(hidden, state)
            outputs.append(block_output)
            hidden = block_output.fused_hidden

        predictions = (
            self.forecast_head(hidden)
            .permute(0, 1, 3, 2)
            .unsqueeze(-1)
            .contiguous()
        )
        expected_predictions = (
            int(x.shape[0]),
            int(self.config.context_length),
            len(self.config.horizons),
            int(self.config.num_nodes),
            1,
        )
        if tuple(predictions.shape) != expected_predictions:
            raise RuntimeError(
                f"predictions has shape {tuple(predictions.shape)}; "
                f"expected {expected_predictions}."
            )
        return DenseTransformerDepthSequenceOutput(
            predictions=predictions,
            block_outputs=tuple(outputs),
        )

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor | None = None,
        session_length: Tensor | None = None,
    ) -> DenseTransformerDepthSequenceOutput:
        return self.forward_dense(
            x,
            context_start=context_start,
            session_length=session_length,
        )


def dense_transformer_depth_config_from_mapping(
    values: dict[str, object],
) -> DenseTransformerDepthConfig:
    model = dict(values["model"])  # type: ignore[arg-type]
    data = dict(values["data"])  # type: ignore[arg-type]
    temporal = dict(model["temporal"])  # type: ignore[arg-type]
    graph = dict(model["graph"])  # type: ignore[arg-type]
    spatial = dict(model["spatial"])  # type: ignore[arg-type]
    prior = dict(model["prior"])  # type: ignore[arg-type]
    config = DenseTransformerDepthConfig(
        num_nodes=int(model["num_nodes"]),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),  # type: ignore[arg-type]
        input_channels=tuple(str(value) for value in data["input_channels"]),  # type: ignore[arg-type]
        target_channel=str(data["target_channel"]),
        num_st_blocks=int(model["num_st_blocks"]),
        d_model=int(temporal["d_model"]),
        transformer_num_layers=int(temporal["num_layers"]),
        transformer_num_heads=int(temporal["num_heads"]),
        transformer_feedforward_multiplier=int(
            temporal["feedforward_multiplier"]
        ),
        transformer_dropout=float(temporal["dropout"]),
        position_embedding=bool(temporal.get("position_embedding", False)),
        graph_heads_per_block=tuple(
            int(value) for value in graph["num_heads_per_block"]  # type: ignore[arg-type]
        ),
        graph_hidden_dims_per_block=tuple(
            int(value) for value in graph["hidden_dims_per_block"]  # type: ignore[arg-type]
        ),
        graph_activations_per_block=tuple(
            str(value) for value in graph["activations_per_block"]  # type: ignore[arg-type]
        ),
        graph_initial_alpha=float(graph["initial_alpha"]),
        random_static_jitter=float(prior["jitter"]),
        graph_seed=int(prior["seed"]),
        spatial_initial_beta=float(spatial["initial_beta"]),
        spatial_feedforward_multiplier=int(spatial["feedforward_multiplier"]),
        spatial_dropout=float(spatial["dropout"]),
    )
    config.validate()
    return config

from __future__ import annotations

"""One-block graph forecasters for dense multi-horizon supervision.

The module isolates the final twelve-run diagnostic from the historical model
classes.  It preserves the winning one-block graph/spatial contract while
supporting two temporal backbones:

* the exact selected ModernTCN feature extractor and official flatten head;
* a causal per-node Transformer that exposes one hidden state, graph and direct
  five-horizon prediction at every minute.

Every graph variant retains the direct continuous-state pathway in both graph
scoring and spatial values.  Graph orientation is ``A[target, source]``.
"""

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from src.models.continuous_forecaster import (
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
    ModernTCNContinuousBackbone,
    SpatialBranchGate,
)
from src.models.dynamic_graph.contracts import GraphConfig, GraphOutput, TemporalConfig
from src.models.dynamic_graph.modules import GraphNormalizer, PerNodeTransformerEncoder
from src.models.modern_tcn_graph_round1 import (
    StateAwareSpatialMessagePassing,
    align_state_embeddings_to_modern_tcn_patches,
    build_v2_prior_logits,
)


TemporalBackbone = Literal["modern_tcn", "transformer"]
GraphVariant = Literal[
    "correlation_static_dynamic_state",
    "random_static_dynamic_state",
    "dynamic_state",
]


@dataclass(frozen=True)
class DenseParallelGraphModelConfig:
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
    temporal_backbone: TemporalBackbone = "modern_tcn"
    graph_variant: GraphVariant = "correlation_static_dynamic_state"

    # Selected ModernTCN architecture.
    modern_tcn_d_model: int = 32
    modern_tcn_patch_size: int = 8
    modern_tcn_patch_stride: int = 4
    modern_tcn_ffn_ratio: int = 1
    modern_tcn_num_blocks: int = 1
    modern_tcn_large_kernel: int = 15
    modern_tcn_small_kernel: int = 5
    modern_tcn_dropout: float = 0.05
    modern_tcn_head_dropout: float = 0.0

    # Causal Transformer control.
    transformer_d_model: int = 96
    transformer_num_layers: int = 1
    transformer_num_heads: int = 8
    transformer_feedforward_multiplier: int = 2
    transformer_dropout: float = 0.0
    transformer_position_embedding: bool = False

    # Shared graph/spatial contract.
    graph_num_heads: int = 1
    graph_hidden_dim: int = 32
    graph_activation: str = "softmax"
    graph_initial_alpha: float = 0.5
    spatial_initial_beta: float = 0.5
    spatial_feedforward_multiplier: int = 2
    spatial_dropout: float = 0.0
    prior_scale: float = 4.0
    prior_jitter: float = 0.02
    prior_seed: int = 42

    def validate(self) -> None:
        if int(self.num_nodes) <= 1 or int(self.context_length) <= 0:
            raise ValueError("num_nodes/context_length are invalid.")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be non-empty, unique and increasing.")
        if any(int(value) <= 0 for value in self.horizons):
            raise ValueError("Every horizon must be positive.")
        if not self.input_channels or self.target_channel not in self.input_channels:
            raise ValueError("target_channel must occur in input_channels.")
        if self.temporal_backbone not in {"modern_tcn", "transformer"}:
            raise ValueError(f"Unsupported temporal backbone {self.temporal_backbone!r}.")
        if self.graph_variant not in {
            "correlation_static_dynamic_state",
            "random_static_dynamic_state",
            "dynamic_state",
        }:
            raise ValueError(f"Unsupported graph variant {self.graph_variant!r}.")
        if self.graph_activation != "softmax":
            raise ValueError("This controlled experiment uses softmax graphs only.")
        if int(self.graph_num_heads) <= 0 or int(self.graph_hidden_dim) <= 0:
            raise ValueError("Graph heads/hidden dimension must be positive.")
        if int(self.graph_hidden_dim) % int(self.graph_num_heads):
            raise ValueError("graph_hidden_dim must be divisible by graph_num_heads.")
        if not 0.0 < float(self.graph_initial_alpha) < 1.0:
            raise ValueError("graph_initial_alpha must lie strictly in (0,1).")
        if not 0.0 < float(self.spatial_initial_beta) < 1.0:
            raise ValueError("spatial_initial_beta must lie strictly in (0,1).")
        if not math.isfinite(float(self.prior_scale)) or float(self.prior_scale) <= 0:
            raise ValueError("prior_scale must be finite and positive.")
        if not math.isfinite(float(self.prior_jitter)) or float(self.prior_jitter) < 0:
            raise ValueError("prior_jitter must be finite and non-negative.")
        if int(self.context_length) % int(self.modern_tcn_patch_stride):
            raise ValueError("context_length must be divisible by ModernTCN patch stride.")
        if int(self.modern_tcn_patch_size) < int(self.modern_tcn_patch_stride):
            raise ValueError("ModernTCN patch size must be >= patch stride.")
        if int(self.transformer_d_model) % int(self.transformer_num_heads):
            raise ValueError("Transformer d_model must be divisible by temporal heads.")
        if int(self.transformer_num_layers) <= 0:
            raise ValueError("Transformer num_layers must be positive.")
        if int(self.transformer_feedforward_multiplier) <= 0:
            raise ValueError("Transformer FF multiplier must be positive.")
        for value in (
            self.modern_tcn_dropout,
            self.modern_tcn_head_dropout,
            self.transformer_dropout,
            self.spatial_dropout,
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("All dropout values must lie in [0,1).")

    @property
    def d_model(self) -> int:
        return (
            int(self.modern_tcn_d_model)
            if self.temporal_backbone == "modern_tcn"
            else int(self.transformer_d_model)
        )

    @property
    def uses_static_graph(self) -> bool:
        return self.graph_variant != "dynamic_state"

    @property
    def prior_type(self) -> str:
        if self.graph_variant == "correlation_static_dynamic_state":
            return "correlation"
        if self.graph_variant == "random_static_dynamic_state":
            return "random"
        return "none"


@dataclass
class DenseSequenceGraphOutput:
    selected: Tensor  # [B,T,G,N,N]
    dynamic: Tensor  # [B,T,G,N,N]
    base: Tensor | None  # [1,G,N,N]
    alpha: Tensor | None
    logits: Tensor

    def final_graph_output(self) -> GraphOutput:
        selected = self.selected[:, -1].contiguous()
        dynamic = self.dynamic[:, -1].contiguous()
        return GraphOutput(
            selected=selected,
            per_layer=(selected,),
            base=self.base,
            dynamic=dynamic,
            alpha=self.alpha,
            logits=(self.logits[:, -1].contiguous() if self.base is None else None),
        )


@dataclass
class DenseParallelGraphOutput:
    predictions: Tensor  # [B,H,N,1]
    temporal_hidden: Tensor
    state_hidden: Tensor
    graph_spatial_hidden: Tensor
    fused_hidden: Tensor
    graph: GraphOutput
    alpha: Tensor | None
    beta: Tensor


@dataclass
class DenseParallelSequenceOutput:
    predictions: Tensor  # [B,T,H,N,1]
    temporal_hidden: Tensor
    state_hidden: Tensor
    graph_spatial_hidden: Tensor
    fused_hidden: Tensor
    graphs: DenseSequenceGraphOutput
    beta: Tensor

    def final_output(self) -> DenseParallelGraphOutput:
        graph = self.graphs.final_graph_output()
        return DenseParallelGraphOutput(
            predictions=self.predictions[:, -1].contiguous(),
            temporal_hidden=self.temporal_hidden[:, -1:].contiguous(),
            state_hidden=self.state_hidden[:, -1:].contiguous(),
            graph_spatial_hidden=self.graph_spatial_hidden[:, -1:].contiguous(),
            fused_hidden=self.fused_hidden[:, -1:].contiguous(),
            graph=graph,
            alpha=graph.alpha,
            beta=self.beta,
        )


class StateAwareStaticDynamicGraphLearner(nn.Module):
    """State-aware dynamic graph with optional random/structured static logits."""

    def __init__(
        self,
        *,
        config: DenseParallelGraphModelConfig,
        static_prior: Tensor | None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.d_model = int(config.d_model)
        self.num_nodes = int(config.num_nodes)
        self.num_heads = int(config.graph_num_heads)
        self.graph_hidden_dim = int(config.graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads

        scorer_dim = 2 * self.d_model
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

        if not config.uses_static_graph:
            if static_prior is not None:
                raise ValueError("Dynamic-only graph must not receive static_prior.")
            self.register_parameter("static_logits", None)
            self.register_parameter("raw_alpha", None)
        else:
            if config.prior_type == "correlation":
                if static_prior is None:
                    raise ValueError("Correlation graph requires static_prior.")
                values = torch.as_tensor(static_prior).detach().cpu().float()
                if tuple(values.shape) != (self.num_nodes, self.num_nodes):
                    raise ValueError("static_prior node axes differ from config.")
                logits = build_v2_prior_logits(
                    values,
                    num_heads=self.num_heads,
                    scale=float(config.prior_scale),
                    jitter=float(config.prior_jitter),
                    seed=int(config.prior_seed),
                )
            elif config.prior_type == "random":
                if static_prior is not None:
                    raise ValueError("Random static graph must not receive a prior.")
                generator = torch.Generator(device="cpu").manual_seed(
                    int(config.prior_seed)
                )
                logits = torch.randn(
                    self.num_heads,
                    self.num_nodes,
                    self.num_nodes,
                    generator=generator,
                    dtype=torch.float32,
                ) * float(config.prior_jitter)
            else:
                raise ValueError("A static graph requires correlation or random init.")
            self.static_logits = nn.Parameter(logits.contiguous())
            alpha = float(config.graph_initial_alpha)
            self.raw_alpha = nn.Parameter(
                torch.tensor(math.log(alpha / (1.0 - alpha)), dtype=torch.float32)
            )

    def alpha(self) -> Tensor | None:
        return None if self.raw_alpha is None else torch.sigmoid(self.raw_alpha)

    def static_adjacency(self) -> Tensor | None:
        if self.static_logits is None:
            return None
        return self.normalizer(self.static_logits.unsqueeze(0))

    def _compute(
        self,
        temporal: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
        """Compute graphs for ``[...,N,D]`` temporal/state tensors."""

        if temporal.shape != state.shape or temporal.shape[-2:] != (
            self.num_nodes,
            self.d_model,
        ):
            raise ValueError("temporal/state tensors do not match graph config.")
        leading = tuple(int(value) for value in temporal.shape[:-2])
        flat_count = math.prod(leading)
        scorer = torch.cat([temporal, state], dim=-1).reshape(
            flat_count,
            self.num_nodes,
            2 * self.d_model,
        )

        # Preserve the selected model's mixed-precision graph path exactly:
        # projections and Q/K multiplication follow the active autocast context,
        # while GraphNormalizer converts logits to float32 before softmax.
        queries = (
            self.q_proj(scorer)
            .view(flat_count, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(scorer)
            .view(flat_count, self.num_nodes, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        logits_flat = (queries @ keys.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        dynamic_flat = self.normalizer(logits_flat.float())

        dynamic = dynamic_flat.reshape(
            *leading,
            self.num_heads,
            self.num_nodes,
            self.num_nodes,
        )
        logits = logits_flat.reshape_as(dynamic)
        base = self.static_adjacency()
        alpha = self.alpha()
        if base is None:
            selected = dynamic
        else:
            if alpha is None:
                raise RuntimeError("Static/dynamic graph has no alpha parameter.")
            view_shape = [1] * len(leading) + [1, 1, 1]
            expanded_base = base.reshape(*view_shape[:-3], *base.shape[1:])
            expanded_base = expanded_base.expand_as(dynamic)
            alpha_value = alpha.to(dynamic.device, dynamic.dtype).reshape(
                *([1] * dynamic.ndim)
            )
            selected = (1.0 - alpha_value) * expanded_base + alpha_value * dynamic
        return selected, dynamic, base, alpha, logits

    def forward_window(
        self,
        temporal_hidden: Tensor,
        state_hidden: Tensor,
    ) -> GraphOutput:
        if temporal_hidden.ndim != 4 or temporal_hidden.shape != state_hidden.shape:
            raise ValueError("window hidden tensors must match [B,L,N,D].")
        selected, dynamic, base, alpha, logits = self._compute(
            temporal_hidden[:, -1],
            state_hidden[:, -1],
        )
        output = GraphOutput(
            selected=selected,
            per_layer=(selected,),
            base=base,
            dynamic=dynamic,
            alpha=alpha,
            logits=(logits if base is None else None),
        )
        output.validate(
            batch_size=int(temporal_hidden.shape[0]),
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output

    def forward_sequence(
        self,
        temporal_hidden: Tensor,
        state_hidden: Tensor,
    ) -> DenseSequenceGraphOutput:
        if temporal_hidden.ndim != 4 or temporal_hidden.shape != state_hidden.shape:
            raise ValueError("sequence hidden tensors must match [B,T,N,D].")
        selected, dynamic, base, alpha, logits = self._compute(
            temporal_hidden,
            state_hidden,
        )
        return DenseSequenceGraphOutput(
            selected=selected,
            dynamic=dynamic,
            base=base,
            alpha=alpha,
            logits=logits,
        )


class SequenceStateAwareSpatialMessagePassing(nn.Module):
    """Apply a different adjacency at every causal Transformer position."""

    def __init__(self, *, config: DenseParallelGraphModelConfig) -> None:
        super().__init__()
        self.d_model = int(config.d_model)
        self.num_heads = int(config.graph_num_heads)
        self.graph_hidden_dim = int(config.graph_hidden_dim)
        self.head_dim = self.graph_hidden_dim // self.num_heads
        input_dim = 2 * self.d_model
        self.value_projection = nn.Linear(input_dim, self.graph_hidden_dim)
        self.output_projection = nn.Linear(self.graph_hidden_dim, self.d_model)
        self.message_dropout = nn.Dropout(float(config.spatial_dropout))
        self.mix_norm = nn.LayerNorm(self.d_model)
        self.feedforward_norm = nn.LayerNorm(self.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(
                self.d_model,
                int(config.spatial_feedforward_multiplier) * self.d_model,
            ),
            nn.GELU(),
            nn.Dropout(float(config.spatial_dropout)),
            nn.Linear(
                int(config.spatial_feedforward_multiplier) * self.d_model,
                self.d_model,
            ),
            nn.Dropout(float(config.spatial_dropout)),
        )

    def _finish(self, temporal_hidden: Tensor, messages: Tensor) -> Tensor:
        projected = self.output_projection(messages)
        mixed = self.mix_norm(temporal_hidden + self.message_dropout(projected))
        return self.feedforward_norm(mixed + self.feedforward(mixed))

    def forward(
        self,
        temporal_hidden: Tensor,
        adjacency: Tensor,
        state_hidden: Tensor,
    ) -> Tensor:
        if temporal_hidden.ndim != 4 or temporal_hidden.shape != state_hidden.shape:
            raise ValueError("temporal/state must match [B,T,N,D].")
        batch, steps, nodes, hidden = map(int, temporal_hidden.shape)
        expected_graph = (batch, steps, self.num_heads, nodes, nodes)
        if tuple(adjacency.shape) != expected_graph:
            raise ValueError(
                f"adjacency has shape {tuple(adjacency.shape)}; "
                f"expected {expected_graph}."
            )
        values = (
            self.value_projection(torch.cat([temporal_hidden, state_hidden], dim=-1))
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
        return self._finish(temporal_hidden, messages)

    def forward_window(
        self,
        temporal_hidden: Tensor,
        adjacency: Tensor,
        state_hidden: Tensor,
    ) -> Tensor:
        if temporal_hidden.ndim != 4 or temporal_hidden.shape != state_hidden.shape:
            raise ValueError("temporal/state must match [B,T,N,D].")
        batch, steps, nodes, _ = map(int, temporal_hidden.shape)
        expected_graph = (batch, self.num_heads, nodes, nodes)
        if tuple(adjacency.shape) != expected_graph:
            raise ValueError(
                f"adjacency has shape {tuple(adjacency.shape)}; expected {expected_graph}."
            )
        value_input = torch.cat(
            [temporal_hidden[:, -1:], state_hidden[:, -1:]],
            dim=-1,
        )
        values = (
            self.value_projection(value_input)
            .view(batch, 1, nodes, self.num_heads, self.head_dim)
            .permute(0, 1, 3, 2, 4)
        )
        messages = torch.einsum(
            "bgij,btgjd->btgid",
            adjacency.to(device=values.device, dtype=values.dtype),
            values,
        )
        messages = (
            messages.permute(0, 1, 3, 2, 4)
            .reshape(batch, 1, nodes, self.graph_hidden_dim)
        )
        return self._finish(temporal_hidden[:, -1:], messages)


class ModernTCNDenseParallelGraphModel(nn.Module):
    """Exact selected ModernTCN path with the three controlled graph types."""

    def __init__(
        self,
        config: DenseParallelGraphModelConfig,
        *,
        static_prior: Tensor | None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.temporal_backbone != "modern_tcn":
            raise ValueError("ModernTCN model requires temporal_backbone='modern_tcn'.")
        self.config = config
        forecaster = ContinuousForecasterConfig(
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
        self.temporal_backbone = ModernTCNContinuousBackbone(config=forecaster)
        self.state_projection = nn.Linear(
            len(config.input_channels),
            config.modern_tcn_d_model,
        )
        self.graph_learner = StateAwareStaticDynamicGraphLearner(
            config=config,
            static_prior=static_prior,
        )
        self.spatial_module = StateAwareSpatialMessagePassing(
            d_model=config.modern_tcn_d_model,
            num_heads=config.graph_num_heads,
            graph_hidden_dim=config.graph_hidden_dim,
            feedforward_multiplier=config.spatial_feedforward_multiplier,
            dropout=config.spatial_dropout,
            use_state_pathway=True,
        )
        self.spatial_gate = SpatialBranchGate(
            gate_type="learned_scalar",
            initial_beta=config.spatial_initial_beta,
        )

    def alpha(self) -> Tensor | None:
        return self.graph_learner.alpha()

    def beta(self) -> Tensor:
        return self.spatial_gate.beta()

    def graph_parameter_ids(self) -> set[int]:
        return {
            id(parameter)
            for parameter in self.graph_learner.parameters()
            if parameter.requires_grad
        }

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> DenseParallelGraphOutput:
        temporal = self.temporal_backbone(
            x,
            context_start=context_start,
            session_length=session_length,
        )
        minute_state = self.state_projection(x)
        state = align_state_embeddings_to_modern_tcn_patches(
            minute_state,
            patch_size=self.config.modern_tcn_patch_size,
            patch_stride=self.config.modern_tcn_patch_stride,
        ).contiguous()
        if state.shape != temporal.shape:
            raise RuntimeError("ModernTCN state/temporal patch shapes differ.")
        graph = self.graph_learner.forward_window(temporal, state)
        spatial = self.spatial_module(
            temporal,
            graph.selected,
            state_hidden=state,
        )
        fused, beta = self.spatial_gate(temporal, spatial)
        predictions = self.temporal_backbone.forecast(fused)
        return DenseParallelGraphOutput(
            predictions=predictions,
            temporal_hidden=temporal,
            state_hidden=state,
            graph_spatial_hidden=spatial,
            fused_hidden=fused,
            graph=graph,
            alpha=graph.alpha,
            beta=beta,
        )


class TransformerDenseParallelGraphModel(nn.Module):
    """One causal Transformer ST block with per-minute graphs and forecasts."""

    def __init__(
        self,
        config: DenseParallelGraphModelConfig,
        *,
        static_prior: Tensor | None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.temporal_backbone != "transformer":
            raise ValueError("Transformer model requires temporal_backbone='transformer'.")
        self.config = config
        d_model = int(config.transformer_d_model)
        self.state_projection = nn.Linear(len(config.input_channels), d_model)
        if config.transformer_position_embedding:
            self.position_embedding: nn.Embedding | None = nn.Embedding(
                config.context_length,
                d_model,
            )
        else:
            self.position_embedding = None
        self.input_norm = nn.LayerNorm(d_model)
        self.temporal_encoder = PerNodeTransformerEncoder(
            d_model=d_model,
            config=TemporalConfig(
                type="transformer",
                num_layers=config.transformer_num_layers,
                num_heads=config.transformer_num_heads,
                feedforward_multiplier=config.transformer_feedforward_multiplier,
                dropout=config.transformer_dropout,
            ),
        )
        self.graph_learner = StateAwareStaticDynamicGraphLearner(
            config=config,
            static_prior=static_prior,
        )
        self.spatial_module = SequenceStateAwareSpatialMessagePassing(config=config)
        self.spatial_gate = SpatialBranchGate(
            gate_type="learned_scalar",
            initial_beta=config.spatial_initial_beta,
        )
        self.forecast_head = nn.Linear(d_model, len(config.horizons))

    def alpha(self) -> Tensor | None:
        return self.graph_learner.alpha()

    def beta(self) -> Tensor:
        return self.spatial_gate.beta()

    def graph_parameter_ids(self) -> set[int]:
        return {
            id(parameter)
            for parameter in self.graph_learner.parameters()
            if parameter.requires_grad
        }

    def forward_dense(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> DenseParallelSequenceOutput:
        del context_start, session_length  # positional sequence is within-window only.
        if tuple(x.shape[1:]) != (
            self.config.context_length,
            self.config.num_nodes,
            len(self.config.input_channels),
        ):
            raise ValueError("Transformer x does not match [T,N,C] config axes.")
        state = self.state_projection(x)
        hidden = state
        if self.position_embedding is not None:
            ids = torch.arange(self.config.context_length, device=x.device)
            hidden = hidden + self.position_embedding(ids).view(
                1,
                self.config.context_length,
                1,
                self.config.transformer_d_model,
            )
        temporal = self.temporal_encoder(self.input_norm(hidden))
        graphs = self.graph_learner.forward_sequence(temporal, state)
        spatial = self.spatial_module(temporal, graphs.selected, state)
        fused, beta = self.spatial_gate(temporal, spatial)
        predictions = (
            self.forecast_head(fused)
            .permute(0, 1, 3, 2)
            .unsqueeze(-1)
            .contiguous()
        )
        expected = (
            int(x.shape[0]),
            self.config.context_length,
            len(self.config.horizons),
            self.config.num_nodes,
            1,
        )
        if tuple(predictions.shape) != expected:
            raise RuntimeError(
                f"Dense Transformer predictions {tuple(predictions.shape)} != {expected}."
            )
        return DenseParallelSequenceOutput(
            predictions=predictions,
            temporal_hidden=temporal,
            state_hidden=state,
            graph_spatial_hidden=spatial,
            fused_hidden=fused,
            graphs=graphs,
            beta=beta,
        )

    def forward(
        self,
        x: Tensor,
        *,
        context_start: Tensor,
        session_length: Tensor,
    ) -> DenseParallelGraphOutput:
        del context_start, session_length
        if tuple(x.shape[1:]) != (
            self.config.context_length,
            self.config.num_nodes,
            len(self.config.input_channels),
        ):
            raise ValueError("Transformer x does not match [T,N,C] config axes.")
        state = self.state_projection(x)
        hidden = state
        if self.position_embedding is not None:
            ids = torch.arange(self.config.context_length, device=x.device)
            hidden = hidden + self.position_embedding(ids).view(
                1,
                self.config.context_length,
                1,
                self.config.transformer_d_model,
            )
        temporal = self.temporal_encoder(self.input_norm(hidden))
        graph = self.graph_learner.forward_window(temporal, state)
        spatial_final = self.spatial_module.forward_window(
            temporal,
            graph.selected,
            state,
        )
        temporal_final = temporal[:, -1:]
        fused_final, beta = self.spatial_gate(temporal_final, spatial_final)
        predictions = (
            self.forecast_head(fused_final[:, 0])
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .contiguous()
        )
        return DenseParallelGraphOutput(
            predictions=predictions,
            temporal_hidden=temporal,
            state_hidden=state,
            graph_spatial_hidden=spatial_final,
            fused_hidden=fused_final,
            graph=graph,
            alpha=graph.alpha,
            beta=beta,
        )


def build_dense_parallel_model(
    config: DenseParallelGraphModelConfig,
    *,
    static_prior: Tensor | None,
) -> nn.Module:
    if config.temporal_backbone == "modern_tcn":
        return ModernTCNDenseParallelGraphModel(config, static_prior=static_prior)
    return TransformerDenseParallelGraphModel(config, static_prior=static_prior)


def dense_parallel_config_from_mapping(
    values: dict,
    *,
    num_nodes: int,
) -> DenseParallelGraphModelConfig:
    data = values["data"]
    model = values["model"]
    temporal = model["temporal"]
    graph = model["graph"]
    spatial = model["spatial"]
    prior = model["prior"]
    return DenseParallelGraphModelConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        temporal_backbone=str(temporal["type"]),
        graph_variant=str(model["variant"]),
        modern_tcn_d_model=int(temporal.get("modern_tcn", {}).get("d_model", 32)),
        modern_tcn_patch_size=int(
            temporal.get("modern_tcn", {}).get("patch_size", 8)
        ),
        modern_tcn_patch_stride=int(
            temporal.get("modern_tcn", {}).get("patch_stride", 4)
        ),
        modern_tcn_ffn_ratio=int(
            temporal.get("modern_tcn", {}).get("ffn_ratio", 1)
        ),
        modern_tcn_num_blocks=int(
            temporal.get("modern_tcn", {}).get("num_blocks", 1)
        ),
        modern_tcn_large_kernel=int(
            temporal.get("modern_tcn", {}).get("large_kernel", 15)
        ),
        modern_tcn_small_kernel=int(
            temporal.get("modern_tcn", {}).get("small_kernel", 5)
        ),
        modern_tcn_dropout=float(
            temporal.get("modern_tcn", {}).get("dropout", 0.05)
        ),
        modern_tcn_head_dropout=float(
            temporal.get("modern_tcn", {}).get("head_dropout", 0.0)
        ),
        transformer_d_model=int(temporal.get("transformer", {}).get("d_model", 96)),
        transformer_num_layers=int(
            temporal.get("transformer", {}).get("num_layers", 1)
        ),
        transformer_num_heads=int(
            temporal.get("transformer", {}).get("num_heads", 8)
        ),
        transformer_feedforward_multiplier=int(
            temporal.get("transformer", {}).get("feedforward_multiplier", 2)
        ),
        transformer_dropout=float(
            temporal.get("transformer", {}).get("dropout", 0.0)
        ),
        transformer_position_embedding=bool(
            temporal.get("transformer", {}).get("position_embedding", False)
        ),
        graph_num_heads=int(graph["num_heads"]),
        graph_hidden_dim=int(graph["hidden_dim"]),
        graph_activation=str(graph["activation"]),
        graph_initial_alpha=float(graph["initial_alpha"]),
        spatial_initial_beta=float(spatial["initial_beta"]),
        spatial_feedforward_multiplier=int(spatial["feedforward_multiplier"]),
        spatial_dropout=float(spatial["dropout"]),
        prior_scale=float(prior["scale"]),
        prior_jitter=float(prior["jitter"]),
        prior_seed=int(prior["seed"]),
    )

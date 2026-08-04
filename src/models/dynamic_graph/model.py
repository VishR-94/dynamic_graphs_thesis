from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import build_model_config
from .contracts import (
    DynamicGraphModelConfig,
    GraphOutput,
    TokenForecastOutput,
)
from .future_predictor import (
    FutureTokenPrediction,
    TokenSelection,
    build_future_token_predictor,
    select_token_ids,
)
from .graph_learners import (
    BaseDyGraphDynamicBaseGraphLearner,
    BaseDyGraphDynamicGraphLearner,
    EmptyCorrelationRowPolicy,
    FixedGraphLearner,
    FreeStaticGraphLearner,
    MTGNNStaticGraphLearner,
    OracleGraphLearner,
    build_graph_learner,
)
from .modules import (
    HierarchicalTokenEmbedding,
    IdentitySpatialModule,
    SpatialMessagePassing,
    build_temporal_encoder,
)
from .modern_tcn_token import ModernTCNTokenEncoder


@dataclass
class GeneratedTokenForecast:
    """Generated future token path plus the standard model output.

    Shapes:
        token_ids:
            [B, prediction_length, N, 2], where the final axis is
            [s1, s2]. In coarse-only mode, the fine column is a
            deterministic zero placeholder ignored by coarse decoding.

        forecast:
            Standard :class:`TokenForecastOutput` containing logits,
            hidden states and graph artefacts.
    """

    token_ids: Tensor
    forecast: TokenForecastOutput

    def validate(
        self,
        config: DynamicGraphModelConfig,
        *,
        batch_size: int,
    ) -> None:
        expected = (
            batch_size,
            config.prediction_length,
            config.num_nodes,
            2,
        )

        if tuple(self.token_ids.shape) != expected:
            raise ValueError(
                "token_ids has shape "
                f"{tuple(self.token_ids.shape)}; expected {expected}."
            )

        if self.token_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        }:
            raise TypeError(
                "Generated token IDs must use an integer dtype."
            )

        if (
            not config.heads.predicts_s2
            and torch.any(self.token_ids[..., 1] != 0)
        ):
            raise ValueError(
                "Coarse-only generated paths must use a zero fine-token "
                "placeholder."
            )

        self.forecast.validate(
            config,
            batch_size=batch_size,
        )


@dataclass
class SampledGeneratedTokenForecast:
    """Multiple generated paths sharing one encoded context/graph.

    ``token_ids`` has shape ``[S, B, P, N, 2]``. The forecast logits and
    graph artefacts are shared because structured-parallel coarse logits are
    deterministic given the observed context; only categorical selection is
    repeated.
    """

    token_ids: Tensor
    forecast: TokenForecastOutput

    def validate(
        self,
        config: DynamicGraphModelConfig,
        *,
        sample_count: int,
        batch_size: int,
    ) -> None:
        expected = (
            int(sample_count),
            int(batch_size),
            config.prediction_length,
            config.num_nodes,
            2,
        )
        if tuple(self.token_ids.shape) != expected:
            raise ValueError(
                f"sampled token_ids has shape {tuple(self.token_ids.shape)}; "
                f"expected {expected}."
            )
        if not config.heads.predicts_s2 and torch.any(
            self.token_ids[..., 1] != 0
        ):
            raise ValueError(
                "Coarse-only sampled paths must use zero fine placeholders."
            )
        self.forecast.validate(config, batch_size=batch_size)


class TokenSpatialBranchGate(nn.Module):
    """FP32 scalar blend between temporal and graph-aware token features."""

    def __init__(self, *, gate_type: str, initial_beta: float) -> None:
        super().__init__()
        if gate_type not in {"none", "fixed", "learned_scalar"}:
            raise ValueError(f"Unsupported spatial gate type {gate_type!r}.")
        if not 0.0 <= float(initial_beta) <= 1.0:
            raise ValueError("initial_beta must lie in [0, 1].")
        self.gate_type = str(gate_type)
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

    def beta(self, *, device: torch.device | None = None) -> Tensor:
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
        if device is not None:
            value = value.to(device=device, dtype=torch.float32)
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
        beta = self.beta(device=temporal_hidden.device)
        device_type = temporal_hidden.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            fused_float = torch.lerp(
                temporal_hidden.float(),
                graph_hidden.float(),
                beta,
            )
        return fused_float.to(dtype=temporal_hidden.dtype), beta


class DynamicGraphTokenForecaster(nn.Module):
    """Shared token forecaster for real and synthetic experiments.

    Pipeline:

        context token IDs [B, T, N, 2]
        -> hierarchical token/node/position embedding
        -> causal per-node temporal encoder
        -> configured context-window graph learner
        -> explicit graph-weighted spatial message passing
        -> structured-parallel or autoregressive future predictor
        -> dense coarse-only or hierarchical s1/s2 logits for every minute

    Every graph follows ``A[target, source]``. The adjacency exposed in
    :class:`GraphOutput` is the exact adjacency supplied to
    :class:`SpatialMessagePassing`.

    Supported graph modes:

        - ``none``;
        - ``fixed`` (including a train-fitted absolute-correlation graph);
        - ``free_static`` (BaseDyGraph direct edge logits);
        - ``mtgnn_static``;
        - ``dynamic`` (BaseDyGraph Q/K graph from the final context state);
        - ``dynamic_base`` (BaseDyGraph convex static/dynamic combination);
        - ``oracle`` (synthetic experiments only).

    Fixed/correlation matrices are external resources. They must be fitted
    from training observations before model construction. The model never
    estimates a correlation graph from validation or test data.
    """

    def __init__(
        self,
        config: DynamicGraphModelConfig,
        *,
        fixed_adjacency: Tensor | None = None,
        correlation_matrix: Tensor | None = None,
        correlation_threshold: float | None = None,
        correlation_empty_row_policy: EmptyCorrelationRowPolicy = "error",
        oracle_graph: Tensor | None = None,
        modern_tcn_model_cls: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        config.validate()

        self.config = config
        self._validate_graph_resources(
            fixed_adjacency=fixed_adjacency,
            correlation_matrix=correlation_matrix,
            correlation_threshold=correlation_threshold,
            oracle_graph=oracle_graph,
        )

        self._fixed_adjacency = fixed_adjacency
        self._correlation_matrix = correlation_matrix
        self._correlation_threshold = correlation_threshold
        self._correlation_empty_row_policy = (
            correlation_empty_row_policy
        )
        self._stored_oracle_graph = oracle_graph

        if config.temporal.type == "modern_tcn":
            self.token_embedding: HierarchicalTokenEmbedding | None = None
            self.modern_tcn_encoder: ModernTCNTokenEncoder | None = (
                ModernTCNTokenEncoder(
                    config,
                    official_model_cls=modern_tcn_model_cls,
                )
            )
            self.temporal_blocks = nn.ModuleList()
        else:
            self.token_embedding = HierarchicalTokenEmbedding(config)
            self.modern_tcn_encoder = None
            self.temporal_blocks = nn.ModuleList(
                [
                    build_temporal_encoder(
                        d_model=config.d_model,
                        config=config.temporal,
                    )
                    for _ in range(config.num_st_blocks)
                ]
            )

        self.graph_learners = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        self.spatial_gates = nn.ModuleList()

        for _ in range(config.num_st_blocks):
            graph_learner, spatial_module = (
                self._build_graph_and_spatial_modules()
            )
            self.graph_learners.append(
                graph_learner
            )
            self.spatial_blocks.append(
                spatial_module
            )
            gate_type = (
                "none"
                if config.graph.type == "none"
                else config.spatial.gate_type
            )
            self.spatial_gates.append(
                TokenSpatialBranchGate(
                    gate_type=gate_type,
                    initial_beta=config.spatial.initial_beta,
                )
            )

        self.future_predictor = (
            build_future_token_predictor(
                config
            )
        )

        if config.backcast.enabled:
            self.backcast_head: nn.Linear | None = nn.Linear(
                config.d_model,
                config.backcast.num_channels,
            )
        else:
            self.backcast_head = None

    @classmethod
    def from_config(
        cls,
        experiment_config: Mapping[str, Any],
        *,
        fixed_adjacency: Tensor | None = None,
        correlation_matrix: Tensor | None = None,
        correlation_threshold: float | None = None,
        correlation_empty_row_policy: EmptyCorrelationRowPolicy = "error",
        oracle_graph: Tensor | None = None,
        modern_tcn_model_cls: type[nn.Module] | None = None,
    ) -> "DynamicGraphTokenForecaster":
        """Build the model from a resolved dynamic-graph config."""
        return cls(
            build_model_config(
                experiment_config
            ),
            fixed_adjacency=fixed_adjacency,
            correlation_matrix=correlation_matrix,
            correlation_threshold=correlation_threshold,
            correlation_empty_row_policy=(
                correlation_empty_row_policy
            ),
            oracle_graph=oracle_graph,
            modern_tcn_model_cls=modern_tcn_model_cls,
        )

    def _validate_graph_resources(
        self,
        *,
        fixed_adjacency: Tensor | None,
        correlation_matrix: Tensor | None,
        correlation_threshold: float | None,
        oracle_graph: Tensor | None,
    ) -> None:
        graph_type = self.config.graph.type

        if (
            fixed_adjacency is not None
            and correlation_matrix is not None
        ):
            raise ValueError(
                "Supply fixed_adjacency or correlation_matrix, not both."
            )

        if (
            correlation_matrix is None
            and correlation_threshold is not None
        ):
            raise ValueError(
                "correlation_threshold was supplied without a "
                "correlation_matrix."
            )

        if (
            correlation_matrix is not None
            and correlation_threshold is None
        ):
            raise ValueError(
                "correlation_matrix requires correlation_threshold."
            )

        if graph_type == "fixed":
            if (
                fixed_adjacency is None
                and correlation_matrix is None
            ):
                raise ValueError(
                    "graph.type='fixed' requires fixed_adjacency or "
                    "correlation_matrix."
                )
        elif graph_type == "dynamic_base":
            # With no external resource the BaseDyGraph learner builds a
            # directly learned free-static base. The graph learner itself
            # validates the configured base_graph_type.
            pass
        elif (
            fixed_adjacency is not None
            or correlation_matrix is not None
        ):
            raise ValueError(
                "Fixed/correlation graph resources are only valid for "
                "graph.type='fixed' or graph.type='dynamic_base'."
            )

        if graph_type != "oracle" and oracle_graph is not None:
            raise ValueError(
                "oracle_graph may only be stored when "
                "graph.type='oracle'."
            )

    def _build_graph_and_spatial_modules(
        self,
    ) -> tuple[nn.Module, nn.Module]:
        config = self.config

        graph_learner = build_graph_learner(
            config=config.graph,
            num_nodes=config.num_nodes,
            d_model=config.d_model,
            fixed_adjacency=self._fixed_adjacency,
            correlation_matrix=self._correlation_matrix,
            correlation_threshold=self._correlation_threshold,
            correlation_empty_row_policy=(
                self._correlation_empty_row_policy
            ),
            oracle_graph=self._stored_oracle_graph,
        )

        if config.graph.type == "none":
            spatial_module: nn.Module = (
                IdentitySpatialModule()
            )
        else:
            spatial_module = SpatialMessagePassing(
                d_model=config.d_model,
                num_heads=config.graph.num_heads,
                num_layers=config.spatial.num_layers,
                feedforward_multiplier=(
                    config.spatial.feedforward_multiplier
                ),
                dropout=config.spatial.dropout,
            )

        return graph_learner, spatial_module

    def _run_graph_learner(
        self,
        graph_learner: nn.Module,
        temporal_hidden: Tensor,
        *,
        oracle_graph: Tensor | None,
    ) -> GraphOutput:
        if self.config.graph.type == "oracle":
            if not isinstance(
                graph_learner,
                OracleGraphLearner,
            ):
                raise RuntimeError(
                    "Oracle graph mode did not construct an "
                    "OracleGraphLearner."
                )

            return graph_learner(
                temporal_hidden,
                oracle_graph=oracle_graph,
            )

        if oracle_graph is not None:
            raise ValueError(
                "oracle_graph was supplied to a learned or fixed graph "
                "mode. Hidden graph truth must never enter non-oracle "
                "models."
            )

        result = graph_learner(
            temporal_hidden
        )

        if not isinstance(
            result,
            GraphOutput,
        ):
            raise TypeError(
                "A graph learner returned an unexpected output type."
            )

        return result

    def _encode_context(
        self,
        token_ids: Tensor,
        *,
        oracle_graph: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, GraphOutput, Tensor | None]:
        """Encode observed tokens and expose the graphs used by the model.

        Returns:
            context_memory:
                Final graph-aware context sequence [B, T, N, D].

            context_hidden:
                Final graph-aware forecast-origin state [B, N, D].

            graph_output:
                Final selected graph plus every block-level selected graph.
        """
        if self.modern_tcn_encoder is not None:
            hidden = self.modern_tcn_encoder(token_ids)
            temporal_sequence = (hidden,)
        else:
            if self.token_embedding is None:
                raise RuntimeError("Hierarchical token embedding is missing.")
            hidden = self.token_embedding(token_ids)
            temporal_sequence = tuple(self.temporal_blocks)

        per_layer_graphs: list[Tensor | None] = []
        final_graph_output = GraphOutput(selected=None)
        final_spatial_beta: Tensor | None = None

        for layer_index, (graph_learner, spatial_block, spatial_gate) in enumerate(
            zip(
                self.graph_learners,
                self.spatial_blocks,
                self.spatial_gates,
                strict=True,
            )
        ):
            if self.modern_tcn_encoder is not None:
                if layer_index != 0:
                    raise RuntimeError(
                        "The token ModernTCN architecture supports one ST block."
                    )
                temporal_hidden = hidden
            else:
                temporal_block = temporal_sequence[layer_index]
                temporal_hidden = temporal_block(hidden)

            block_graph = self._run_graph_learner(
                graph_learner,
                temporal_hidden,
                oracle_graph=oracle_graph,
            )
            per_layer_graphs.append(block_graph.selected)

            if block_graph.selected is None:
                graph_hidden = spatial_block(temporal_hidden, None)
                hidden = graph_hidden
                final_spatial_beta = None
            else:
                graph_hidden = spatial_block(
                    temporal_hidden,
                    block_graph.selected,
                )
                hidden, beta_value = spatial_gate(
                    temporal_hidden,
                    graph_hidden,
                )
                final_spatial_beta = (
                    None
                    if spatial_gate.gate_type == "none"
                    else beta_value
                )

            final_graph_output = block_graph

        graph_output = GraphOutput(
            selected=final_graph_output.selected,
            per_layer=tuple(
                per_layer_graphs
            ),
            base=final_graph_output.base,
            dynamic=final_graph_output.dynamic,
            alpha=final_graph_output.alpha,
            logits=final_graph_output.logits,
        )

        batch_size = int(
            token_ids.shape[0]
        )

        graph_output.validate(
            batch_size=batch_size,
            num_heads=self.config.graph.num_heads,
            num_nodes=self.config.num_nodes,
        )

        context_hidden = hidden[
            :,
            -1,
            :,
            :,
        ]

        return (
            hidden,
            context_hidden,
            graph_output,
            final_spatial_beta,
        )

    def _build_forecast_output(
        self,
        *,
        prediction: FutureTokenPrediction,
        context_memory: Tensor,
        context_hidden: Tensor,
        graph_output: GraphOutput,
        spatial_beta: Tensor | None,
    ) -> TokenForecastOutput:
        backcast = (
            None
            if self.backcast_head is None
            else self.backcast_head(
                context_memory
            )
        )

        output = TokenForecastOutput(
            s1_logits=prediction.s1_logits,
            s2_logits=prediction.s2_logits,
            graph=graph_output,
            context_hidden=context_hidden,
            # This field stores the final graph-aware observed context
            # sequence consumed by FutureTokenPredictor. The contract
            # retains its historical ``temporal_hidden`` name.
            temporal_hidden=context_memory,
            future_hidden=prediction.future_hidden,
            spatial_beta=spatial_beta,
            backcast=backcast,
        )

        output.validate(
            self.config,
            batch_size=int(
                context_memory.shape[0]
            ),
        )

        return output

    def forward(
        self,
        token_ids: Tensor,
        *,
        target_s1: Tensor | None = None,
        target_s2: Tensor | None = None,
        oracle_graph: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> TokenForecastOutput:
        """Run the supervised training/evaluation forward path.

        In structured-parallel mode, true future tokens never enter the
        future hidden states or s1 logits. ``target_s1`` conditions only
        the same-position fine-token classifier under the current head
        policy.

        In autoregressive mode, the coarse target stream is always
        required. The fine stream is additionally required in full-token
        mode. Use :meth:`generate` for free-running inference.

        ``oracle_graph`` is accepted only when ``graph.type='oracle'``.
        """
        (
            context_memory,
            context_hidden,
            graph_output,
            spatial_beta,
        ) = self._encode_context(
            token_ids,
            oracle_graph=oracle_graph,
        )

        prediction = self.future_predictor(
            context_memory,
            s1_embedding=(
                None
                if self.token_embedding is None
                else self.token_embedding.s1_embedding
            ),
            s2_embedding=(
                None
                if self.token_embedding is None
                else self.token_embedding.s2_embedding
            ),
            target_s1=target_s1,
            target_s2=target_s2,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return self._build_forecast_output(
            prediction=prediction,
            context_memory=context_memory,
            context_hidden=context_hidden,
            graph_output=graph_output,
            spatial_beta=spatial_beta,
        )

    def spatial_mixing_beta(self) -> Tensor | None:
        """Return the final spatial branch gate value, when active."""
        if not self.spatial_gates:
            return None
        gate = self.spatial_gates[-1]
        if gate.gate_type == "none":
            return None
        return gate.beta()

    def generate_samples(
        self,
        token_ids: Tensor,
        *,
        sample_count: int,
        oracle_graph: Tensor | None = None,
        token_selection: TokenSelection = "sample",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> SampledGeneratedTokenForecast:
        """Generate multiple complete future paths for decoded averaging.

        The efficient shared-logit path is intentionally restricted to the
        final experiment contract: structured-parallel, coarse-only output.
        All 60 categorical positions are sampled for every path; prices are
        averaged only after the frozen coarse decoder.
        """
        if int(sample_count) <= 0:
            raise ValueError("sample_count must be positive.")
        if token_selection == "argmax" and int(sample_count) != 1:
            raise ValueError("argmax generation requires sample_count=1.")
        if self.config.future_predictor.type != "structured_parallel":
            raise ValueError(
                "Efficient multi-path generation currently requires the "
                "structured-parallel predictor."
            )
        if self.config.heads.predicts_s2:
            raise ValueError(
                "The final Monte Carlo path is coarse-only; s2 sampling is "
                "deliberately unsupported."
            )

        (
            context_memory,
            context_hidden,
            graph_output,
            spatial_beta,
        ) = self._encode_context(token_ids, oracle_graph=oracle_graph)

        prediction = self.future_predictor.generate(
            context_memory,
            s1_embedding=None,
            s2_embedding=None,
            token_selection="argmax",
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )
        forecast = self._build_forecast_output(
            prediction=prediction,
            context_memory=context_memory,
            context_hidden=context_hidden,
            graph_output=graph_output,
            spatial_beta=spatial_beta,
        )

        samples = []
        for _ in range(int(sample_count)):
            selected_s1 = select_token_ids(
                prediction.s1_logits,
                mode=token_selection,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            samples.append(
                torch.stack(
                    (selected_s1, torch.zeros_like(selected_s1)),
                    dim=-1,
                )
            )
        result = SampledGeneratedTokenForecast(
            token_ids=torch.stack(samples, dim=0),
            forecast=forecast,
        )
        result.validate(
            self.config,
            sample_count=int(sample_count),
            batch_size=int(token_ids.shape[0]),
        )
        return result

    def generate(
        self,
        token_ids: Tensor,
        *,
        oracle_graph: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> GeneratedTokenForecast:
        """Generate a complete future token path without true targets."""
        (
            context_memory,
            context_hidden,
            graph_output,
            spatial_beta,
        ) = self._encode_context(
            token_ids,
            oracle_graph=oracle_graph,
        )

        prediction = self.future_predictor.generate(
            context_memory,
            s1_embedding=(
                None
                if self.token_embedding is None
                else self.token_embedding.s1_embedding
            ),
            s2_embedding=(
                None
                if self.token_embedding is None
                else self.token_embedding.s2_embedding
            ),
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        forecast = self._build_forecast_output(
            prediction=prediction,
            context_memory=context_memory,
            context_hidden=context_hidden,
            graph_output=graph_output,
            spatial_beta=spatial_beta,
        )

        selected_s2 = (
            prediction.selected_s2
            if prediction.selected_s2 is not None
            else torch.zeros_like(
                prediction.selected_s1
            )
        )

        generated = GeneratedTokenForecast(
            token_ids=torch.stack(
                (
                    prediction.selected_s1,
                    selected_s2,
                ),
                dim=-1,
            ),
            forecast=forecast,
        )

        generated.validate(
            self.config,
            batch_size=int(
                token_ids.shape[0]
            ),
        )

        return generated


def _assert_nonzero_finite_gradient(
    parameter: nn.Parameter,
    *,
    name: str,
) -> None:
    gradient = parameter.grad

    if gradient is None:
        raise AssertionError(
            f"{name} did not receive a gradient."
        )

    if not torch.isfinite(
        gradient
    ).all():
        raise AssertionError(
            f"{name} received a non-finite gradient."
        )

    if gradient.abs().sum().item() == 0.0:
        raise AssertionError(
            f"{name} received only zero gradients."
        )


def _small_config(
    *,
    predictor_type: str = "structured_parallel",
    graph_type: str,
    base_graph_type: str = "free_static",
    gate_type: str = "learned_scalar",
    initial_alpha: float = 0.35,
    future_token_mode: str = "full",
) -> DynamicGraphModelConfig:
    from .contracts import (
        ForecastHeadConfig,
        FuturePredictorConfig,
        GraphConfig,
        TemporalConfig,
    )

    return DynamicGraphModelConfig(
        num_nodes=6,
        context_length=10,
        d_model=16,
        num_st_blocks=1,
        use_node_embedding=True,
        temporal=TemporalConfig(
            type="transformer",
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        graph=GraphConfig(
            type=graph_type,  # type: ignore[arg-type]
            num_heads=2,
            hidden_dim=8,
            activation="softmax",
            add_self_loops=False,
            mtgnn_embedding_dim=8,
            mtgnn_top_k=3,
            mtgnn_alpha=3.0,
            base_graph_type=base_graph_type,  # type: ignore[arg-type]
            gate_type=gate_type,  # type: ignore[arg-type]
            initial_alpha=initial_alpha,
        ),
        heads=ForecastHeadConfig(
            prediction_length=6,
            evaluation_horizons=(
                1,
                3,
                6,
            ),
            s1_vocabulary_size=32,
            s2_vocabulary_size=32,
            s2_loss_weight=(
                1.0
                if future_token_mode == "full"
                else 0.0
            ),
            future_token_mode=future_token_mode,  # type: ignore[arg-type]
            condition_s2_on_s1=True,
        ),
        future_predictor=FuturePredictorConfig(
            type=predictor_type,  # type: ignore[arg-type]
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
    )


def _ring_graph(
    *,
    num_nodes: int,
) -> Tensor:
    graph = torch.zeros(
        num_nodes,
        num_nodes,
    )

    for target in range(
        num_nodes
    ):
        graph[
            target,
            (target + 1) % num_nodes,
        ] = 2.0
        graph[
            target,
            (target + 2) % num_nodes,
        ] = 1.0

    return graph / graph.sum(
        dim=-1,
        keepdim=True,
    )


def _correlation_fixture() -> Tensor:
    correlation = torch.tensor(
        [
            [1.0, 0.8, -0.6, 0.1, 0.4, 0.2],
            [0.8, 1.0, 0.5, 0.3, 0.2, 0.7],
            [-0.6, 0.5, 1.0, 0.9, 0.1, 0.4],
            [0.1, 0.3, 0.9, 1.0, 0.8, 0.5],
            [0.4, 0.2, 0.1, 0.8, 1.0, 0.9],
            [0.2, 0.7, 0.4, 0.5, 0.9, 1.0],
        ],
        dtype=torch.float32,
    )

    return correlation


def _token_loss(
    output: TokenForecastOutput,
    target_s1: Tensor,
    target_s2: Tensor,
) -> Tensor:
    loss = F.cross_entropy(
        output.s1_logits.reshape(
            -1,
            output.s1_logits.shape[-1],
        ),
        target_s1.reshape(-1),
    )

    if output.s2_logits is not None:
        loss = loss + F.cross_entropy(
            output.s2_logits.reshape(
                -1,
                output.s2_logits.shape[-1],
            ),
            target_s2.reshape(-1),
        )

    return loss


def _assert_standard_output(
    output: TokenForecastOutput,
    *,
    config: DynamicGraphModelConfig,
    batch_size: int,
    graph_expected: bool,
) -> None:
    output.validate(
        config,
        batch_size=batch_size,
    )

    if graph_expected:
        if output.graph.selected is None:
            raise AssertionError(
                "Expected a selected graph."
            )
    elif output.graph.selected is not None:
        raise AssertionError(
            "No-graph mode exposed an adjacency."
        )

    if len(output.graph.per_layer) != config.num_st_blocks:
        raise AssertionError(
            "The output did not preserve one graph entry per block."
        )


def _cpu_smoke_test() -> None:
    torch.manual_seed(42)

    batch_size = 2
    context_length = 10
    prediction_length = 6
    num_nodes = 6
    vocabulary_size = 32

    tokens = torch.randint(
        0,
        vocabulary_size,
        (
            batch_size,
            context_length,
            num_nodes,
            2,
        ),
    )
    target_s1 = torch.randint(
        0,
        vocabulary_size,
        (
            batch_size,
            prediction_length,
            num_nodes,
        ),
    )
    target_s2 = torch.randint(
        0,
        vocabulary_size,
        (
            batch_size,
            prediction_length,
            num_nodes,
        ),
    )

    # No graph.
    no_graph_config = _small_config(
        graph_type="none",
    )
    no_graph_model = DynamicGraphTokenForecaster(
        no_graph_config
    )
    no_graph_output = no_graph_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        no_graph_output,
        config=no_graph_config,
        batch_size=batch_size,
        graph_expected=False,
    )

    # Supplied fixed graph.
    fixed_config = _small_config(
        graph_type="fixed",
    )
    fixed_model = DynamicGraphTokenForecaster(
        fixed_config,
        fixed_adjacency=_ring_graph(
            num_nodes=num_nodes
        ),
    )
    fixed_output = fixed_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        fixed_output,
        config=fixed_config,
        batch_size=batch_size,
        graph_expected=True,
    )
    if not isinstance(
        fixed_model.graph_learners[0],
        FixedGraphLearner,
    ):
        raise AssertionError(
            "Fixed graph mode built the wrong learner."
        )

    # Training-only absolute-correlation graph.
    correlation_model = DynamicGraphTokenForecaster(
        fixed_config,
        correlation_matrix=_correlation_fixture(),
        correlation_threshold=0.5,
    )
    correlation_output = correlation_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        correlation_output,
        config=fixed_config,
        batch_size=batch_size,
        graph_expected=True,
    )

    # BaseDyGraph free-static graph. Use the forecasting loss to prove
    # that the graph used by spatial mixing receives downstream signal.
    free_static_config = _small_config(
        graph_type="free_static",
    )
    free_static_model = DynamicGraphTokenForecaster(
        free_static_config
    )
    free_static_output = free_static_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _token_loss(
        free_static_output,
        target_s1,
        target_s2,
    ).backward()
    free_static_learner = (
        free_static_model.graph_learners[0]
    )
    if not isinstance(
        free_static_learner,
        FreeStaticGraphLearner,
    ):
        raise AssertionError(
            "Free-static mode built the wrong learner."
        )
    _assert_nonzero_finite_gradient(
        free_static_learner.logits,
        name="free-static graph logits",
    )

    # Existing MTGNN static path remains available as an ablation.
    mtgnn_config = _small_config(
        graph_type="mtgnn_static",
    )
    mtgnn_model = DynamicGraphTokenForecaster(
        mtgnn_config
    )
    if not isinstance(
        mtgnn_model.graph_learners[0],
        MTGNNStaticGraphLearner,
    ):
        raise AssertionError(
            "MTGNN mode built the wrong learner."
        )
    mtgnn_output = mtgnn_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        mtgnn_output,
        config=mtgnn_config,
        batch_size=batch_size,
        graph_expected=True,
    )

    # BaseDyGraph input-conditioned graph.
    dynamic_config = _small_config(
        graph_type="dynamic",
    )
    dynamic_model = DynamicGraphTokenForecaster(
        dynamic_config
    )
    dynamic_output = dynamic_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _token_loss(
        dynamic_output,
        target_s1,
        target_s2,
    ).backward()
    dynamic_learner = dynamic_model.graph_learners[0]
    if not isinstance(
        dynamic_learner,
        BaseDyGraphDynamicGraphLearner,
    ):
        raise AssertionError(
            "Dynamic mode built the wrong learner."
        )
    _assert_nonzero_finite_gradient(
        dynamic_learner.q_proj.weight,
        name="dynamic q projection",
    )
    _assert_nonzero_finite_gradient(
        dynamic_learner.k_proj.weight,
        name="dynamic k projection",
    )

    # BaseDyGraph learned free-static + dynamic convex graph.
    dynamic_base_config = _small_config(
        graph_type="dynamic_base",
        base_graph_type="free_static",
        gate_type="learned_per_head",
    )
    dynamic_base_model = DynamicGraphTokenForecaster(
        dynamic_base_config
    )
    dynamic_base_output = dynamic_base_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        dynamic_base_output,
        config=dynamic_base_config,
        batch_size=batch_size,
        graph_expected=True,
    )
    if any(
        value is None
        for value in (
            dynamic_base_output.graph.base,
            dynamic_base_output.graph.dynamic,
            dynamic_base_output.graph.alpha,
        )
    ):
        raise AssertionError(
            "Dynamic-base mode did not expose all graph components."
        )

    # Correlation + dynamic uses the same convex gate and preserves the
    # thresholded correlation support in its base graph.
    correlation_dynamic_model = DynamicGraphTokenForecaster(
        dynamic_base_config,
        correlation_matrix=_correlation_fixture(),
        correlation_threshold=0.5,
    )
    correlation_dynamic_output = correlation_dynamic_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        correlation_dynamic_output,
        config=dynamic_base_config,
        batch_size=batch_size,
        graph_expected=True,
    )

    # Oracle truth is accepted only in oracle mode and may vary by batch.
    oracle_config = _small_config(
        graph_type="oracle",
    )
    oracle_model = DynamicGraphTokenForecaster(
        oracle_config
    )
    oracle_truth = _ring_graph(
        num_nodes=num_nodes
    ).unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )

    try:
        oracle_model(
            tokens,
            target_s1=target_s1,
            target_s2=target_s2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Oracle mode accepted a batch without graph truth."
        )

    oracle_output = oracle_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
        oracle_graph=oracle_truth,
    )
    _assert_standard_output(
        oracle_output,
        config=oracle_config,
        batch_size=batch_size,
        graph_expected=True,
    )

    try:
        free_static_model(
            tokens,
            target_s1=target_s1,
            target_s2=target_s2,
            oracle_graph=oracle_truth,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "A learned graph mode consumed hidden oracle truth."
        )

    # Full parallel generation remains valid with the new graph wiring.
    free_static_model.eval()
    with torch.no_grad():
        generated = free_static_model.generate(
            tokens
        )
    generated.validate(
        free_static_config,
        batch_size=batch_size,
    )

    # Coarse-only mode keeps both observed context token streams but
    # predicts only s1, returns no s2 logits, and places a deterministic
    # zero placeholder in the generated pair tensor for coarse decoding.
    coarse_only_config = _small_config(
        graph_type="free_static",
        future_token_mode="coarse_only",
    )
    coarse_only_model = DynamicGraphTokenForecaster(
        coarse_only_config
    )
    coarse_only_output = coarse_only_model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    _assert_standard_output(
        coarse_only_output,
        config=coarse_only_config,
        batch_size=batch_size,
        graph_expected=True,
    )
    if coarse_only_output.s2_logits is not None:
        raise AssertionError(
            "Coarse-only model returned s2 logits."
        )

    coarse_only_model.eval()
    with torch.no_grad():
        coarse_only_generated = coarse_only_model.generate(
            tokens
        )
    coarse_only_generated.validate(
        coarse_only_config,
        batch_size=batch_size,
    )
    if torch.any(
        coarse_only_generated.token_ids[..., 1] != 0
    ):
        raise AssertionError(
            "Coarse-only generated path has a non-zero fine placeholder."
        )

    # The autoregressive no-graph path must remain unchanged.
    autoregressive_config = _small_config(
        predictor_type="autoregressive",
        graph_type="none",
    )
    autoregressive_model = DynamicGraphTokenForecaster(
        autoregressive_config
    )
    autoregressive_model.eval()
    with torch.no_grad():
        teacher_forced = autoregressive_model(
            tokens[:1],
            target_s1=target_s1[:1],
            target_s2=target_s2[:1],
        )
        autoregressive_generated = (
            autoregressive_model.generate(
                tokens[:1]
            )
        )
    _assert_standard_output(
        teacher_forced,
        config=autoregressive_config,
        batch_size=1,
        graph_expected=False,
    )

    print(
        "DYNAMIC GRAPH FORECASTER ALL-GRAPH CPU SMOKE TEST PASSED"
    )
    print(
        "Tested model wiring: none, fixed, absolute correlation, "
        "free static, MTGNN static, BaseDyGraph dynamic, learned "
        "static+dynamic, correlation+dynamic, and oracle."
    )
    print(
        "Generated token path:",
        tuple(generated.token_ids.shape),
    )
    print(
        "Autoregressive no-graph path:",
        tuple(autoregressive_generated.token_ids.shape),
    )


if __name__ == "__main__":
    _cpu_smoke_test()


from __future__ import annotations

from dataclasses import dataclass
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
)
from .graph_learners import MTGNNStaticGraphLearner
from .modules import (
    HierarchicalTokenEmbedding,
    IdentitySpatialModule,
    SpatialMessagePassing,
    build_temporal_encoder,
)


@dataclass
class GeneratedTokenForecast:
    """Generated future token path plus the standard model output.

    Shapes:
        token_ids:
            [B, prediction_length, N, 2], where the final axis is
            [s1, s2].

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

        self.forecast.validate(
            config,
            batch_size=batch_size,
        )


class DynamicGraphTokenForecaster(nn.Module):
    """Shared token forecaster for the real and synthetic experiments.

    Current implemented path:

        context token IDs [B, T, N, 2]
        -> hierarchical token/node/position embedding
        -> causal per-node temporal encoder
        -> MTGNN static graph learner
        -> explicit graph-weighted spatial message passing
        -> structured-parallel or autoregressive future predictor
        -> 60-step hierarchical s1/s2 logits

    One graph is inferred per context window and reused across every
    observed context position. The graph orientation is always:

        A[target, source]

    The graph exposed through ``GraphOutput`` is exactly the graph used
    by ``SpatialMessagePassing``.

    The current implementation supports the graph modes required for
    the initial real predictor comparison:

        - ``none``;
        - ``mtgnn_static``.

    The remaining fixed, oracle, free-static, dynamic and dynamic-base
    learners will be added behind the same private builder without
    changing this public model interface.
    """

    def __init__(
        self,
        config: DynamicGraphModelConfig,
    ) -> None:
        super().__init__()
        config.validate()

        self.config = config

        self.token_embedding = HierarchicalTokenEmbedding(
            config
        )

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

        for _ in range(config.num_st_blocks):
            graph_learner, spatial_module = (
                self._build_graph_and_spatial_modules(
                    config
                )
            )

            self.graph_learners.append(
                graph_learner
            )
            self.spatial_blocks.append(
                spatial_module
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
    ) -> "DynamicGraphTokenForecaster":
        """Build the model from a resolved dynamic-graph config."""
        return cls(
            build_model_config(
                experiment_config
            )
        )

    @staticmethod
    def _build_graph_and_spatial_modules(
        config: DynamicGraphModelConfig,
    ) -> tuple[nn.Module, nn.Module]:
        if config.graph.type == "none":
            return (
                _NoGraphLearner(
                    num_heads=config.graph.num_heads,
                    num_nodes=config.num_nodes,
                ),
                IdentitySpatialModule(),
            )

        if config.graph.type == "mtgnn_static":
            return (
                MTGNNStaticGraphLearner(
                    config=config.graph,
                    num_nodes=config.num_nodes,
                ),
                SpatialMessagePassing(
                    d_model=config.d_model,
                    num_heads=config.graph.num_heads,
                    num_layers=1,
                    feedforward_multiplier=(
                        config.temporal.feedforward_multiplier
                    ),
                    dropout=config.temporal.dropout,
                ),
            )

        raise NotImplementedError(
            "DynamicGraphTokenForecaster currently implements "
            "graph.type='none' and graph.type='mtgnn_static'. "
            f"Received {config.graph.type!r}."
        )

    def _encode_context(
        self,
        token_ids: Tensor,
    ) -> tuple[Tensor, Tensor, GraphOutput]:
        """Encode the observed token context and expose all graphs.

        Returns:
            context_memory:
                Final graph-aware context sequence [B, T, N, D].

            context_hidden:
                Final graph-aware forecast-origin state [B, N, D].

            graph_output:
                Final selected graph plus every block-level graph.
        """
        hidden = self.token_embedding(
            token_ids
        )

        per_layer_graphs: list[Tensor | None] = []
        final_graph_output = GraphOutput(
            selected=None,
        )

        for (
            temporal_block,
            graph_learner,
            spatial_block,
        ) in zip(
            self.temporal_blocks,
            self.graph_learners,
            self.spatial_blocks,
            strict=True,
        ):
            temporal_hidden = temporal_block(
                hidden
            )

            block_graph = graph_learner(
                temporal_hidden
            )

            per_layer_graphs.append(
                block_graph.selected
            )

            if block_graph.selected is None:
                hidden = spatial_block(
                    temporal_hidden,
                    None,
                )
            else:
                hidden = spatial_block(
                    temporal_hidden,
                    block_graph.selected,
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
        )

    def _build_forecast_output(
        self,
        *,
        prediction: FutureTokenPrediction,
        context_memory: Tensor,
        context_hidden: Tensor,
        graph_output: GraphOutput,
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
            # sequence consumed by FutureTokenPredictor. The existing
            # contract retains its historical ``temporal_hidden`` name.
            temporal_hidden=context_memory,
            future_hidden=prediction.future_hidden,
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
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> TokenForecastOutput:
        """Run the training/evaluation forward path.

        Structured-parallel prediction may omit targets. Supplying
        ``target_s1`` teacher-forces only the same-position fine-token
        classifier.

        Autoregressive training requires both target streams and uses a
        single teacher-forced causal pass. Use :meth:`generate` for
        free-running autoregressive inference.
        """
        (
            context_memory,
            context_hidden,
            graph_output,
        ) = self._encode_context(
            token_ids
        )

        prediction = self.future_predictor(
            context_memory,
            s1_embedding=(
                self.token_embedding.s1_embedding
            ),
            s2_embedding=(
                self.token_embedding.s2_embedding
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
        )

    def generate(
        self,
        token_ids: Tensor,
        *,
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
        ) = self._encode_context(
            token_ids
        )

        prediction = self.future_predictor.generate(
            context_memory,
            s1_embedding=(
                self.token_embedding.s1_embedding
            ),
            s2_embedding=(
                self.token_embedding.s2_embedding
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
        )

        generated = GeneratedTokenForecast(
            token_ids=torch.stack(
                (
                    prediction.selected_s1,
                    prediction.selected_s2,
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


class _NoGraphLearner(nn.Module):
    """Internal no-graph provider behind the graph-learner interface."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_nodes = int(num_nodes)

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        if context_hidden.ndim != 4:
            raise ValueError(
                "context_hidden must have shape [B, T, N, D]."
            )

        if int(context_hidden.shape[2]) != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes; received "
                f"{int(context_hidden.shape[2])}."
            )

        return GraphOutput(
            selected=None,
        )


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
    predictor_type: str,
    graph_type: str = "mtgnn_static",
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
            type=graph_type,
            num_heads=2,
            activation="softmax",
            add_self_loops=False,
            mtgnn_embedding_dim=8,
            mtgnn_top_k=3,
            mtgnn_alpha=3.0,
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
            s2_loss_weight=1.0,
            condition_s2_on_s1=True,
        ),
        future_predictor=FuturePredictorConfig(
            type=predictor_type,
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
    )


def _cpu_smoke_test() -> None:
    torch.manual_seed(42)

    config = _small_config(
        predictor_type="structured_parallel"
    )
    model = DynamicGraphTokenForecaster(
        config
    )

    batch_size = 2

    tokens = torch.randint(
        0,
        32,
        (
            batch_size,
            config.context_length,
            config.num_nodes,
            2,
        ),
    )
    target_s1 = torch.randint(
        0,
        32,
        (
            batch_size,
            config.prediction_length,
            config.num_nodes,
        ),
    )
    target_s2 = torch.randint(
        0,
        32,
        (
            batch_size,
            config.prediction_length,
            config.num_nodes,
        ),
    )

    model.train()
    output = model(
        tokens,
        target_s1=target_s1,
        target_s2=target_s2,
    )

    loss = (
        F.cross_entropy(
            output.s1_logits.reshape(
                -1,
                config.heads.s1_vocabulary_size,
            ),
            target_s1.reshape(-1),
        )
        + F.cross_entropy(
            output.s2_logits.reshape(
                -1,
                config.heads.s2_vocabulary_size,
            ),
            target_s2.reshape(-1),
        )
    )
    loss.backward()

    mtgnn_learner = model.graph_learners[0]

    if not isinstance(
        mtgnn_learner,
        MTGNNStaticGraphLearner,
    ):
        raise AssertionError(
            "The smoke test did not build the MTGNN graph learner."
        )

    _assert_nonzero_finite_gradient(
        mtgnn_learner.embedding_1[0].weight,
        name="MTGNN embedding_1[0]",
    )
    _assert_nonzero_finite_gradient(
        mtgnn_learner.embedding_2[0].weight,
        name="MTGNN embedding_2[0]",
    )

    model.eval()

    with torch.no_grad():
        generated = model.generate(
            tokens,
            token_selection="argmax",
        )

    expected_token_shape = (
        batch_size,
        config.prediction_length,
        config.num_nodes,
        2,
    )

    if tuple(
        generated.token_ids.shape
    ) != expected_token_shape:
        raise AssertionError(
            "Unexpected generated token shape."
        )

    if generated.forecast.graph.selected is None:
        raise AssertionError(
            "Generated forecast did not expose its selected graph."
        )

    expected_graph_shape = (
        batch_size,
        config.graph.num_heads,
        config.num_nodes,
        config.num_nodes,
    )

    if tuple(
        generated.forecast.graph.selected.shape
    ) != expected_graph_shape:
        raise AssertionError(
            "Unexpected selected graph shape."
        )

    if len(
        generated.forecast.graph.per_layer
    ) != config.num_st_blocks:
        raise AssertionError(
            "The output did not retain every block-level graph."
        )

    autoregressive_config = _small_config(
        predictor_type="autoregressive",
        graph_type="none",
    )
    autoregressive_model = (
        DynamicGraphTokenForecaster(
            autoregressive_config
        )
    )

    autoregressive_tokens = tokens[
        :1
    ]
    autoregressive_target_s1 = target_s1[
        :1
    ]
    autoregressive_target_s2 = target_s2[
        :1
    ]

    autoregressive_model.eval()

    with torch.no_grad():
        teacher_forced = autoregressive_model(
            autoregressive_tokens,
            target_s1=autoregressive_target_s1,
            target_s2=autoregressive_target_s2,
        )
        autoregressive_generated = (
            autoregressive_model.generate(
                autoregressive_tokens
            )
        )

    if tuple(
        teacher_forced.s1_logits.shape
    ) != (
        1,
        autoregressive_config.prediction_length,
        autoregressive_config.num_nodes,
        autoregressive_config.heads.s1_vocabulary_size,
    ):
        raise AssertionError(
            "Unexpected autoregressive teacher-forced shape."
        )

    if autoregressive_generated.forecast.graph.selected is not None:
        raise AssertionError(
            "The no-graph autoregressive ablation exposed a graph."
        )

    print(
        "Dynamic-graph token forecaster CPU smoke test passed."
    )
    print(
        "Training logits:",
        tuple(output.s1_logits.shape),
    )
    print(
        "Generated token path:",
        tuple(generated.token_ids.shape),
    )
    print(
        "Selected graph:",
        tuple(
            generated.forecast.graph.selected.shape
        ),
    )
    print(
        "MTGNN embedding gradient norm:",
        f"{mtgnn_learner.embedding_1[0].weight.grad.norm().item():.6f}",
    )
    print(
        "Autoregressive no-graph path:",
        tuple(
            autoregressive_generated.token_ids.shape
        ),
    )


if __name__ == "__main__":
    _cpu_smoke_test()

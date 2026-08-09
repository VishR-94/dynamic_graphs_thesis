from __future__ import annotations

"""Dense one-step graph-learning controls in token and price space.

The two BaseDyGraph controls preserve the pinned v1 architecture:

    learned state/node representation
    -> four causal Transformer / dynamic-graph / spatial blocks
    -> direct next-Close head at every predictor position

The token-input model uses native Kronos coarse ``s1`` IDs.  The continuous
model replaces only the discrete state embedding with a causal OHLCV
projection.  Both receive a 61-position teacher-forced sequence and expose the
60 predictor positions.  The appended h=1 observation is never visible to the
forecast-origin hidden state because every temporal block is causal.

The one-block ModernTCN controls are isolated adapters in this module.  Both
retain the attached model's state-aware graph scorer, state-aware spatial
values, and learned beta gate; their only graph difference is whether a
trainable random static graph is mixed with the dynamic graph.
"""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from src.models.basedygraph_official_adapter import (
    OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
    PINNED_BASEDYGRAPH_COMMIT,
    OfficialBaseDyGraphRunConfig,
    build_official_model_config,
    load_official_basedygraph_architecture_modules,
)
from src.models.continuous_forecaster import (
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
    ModernTCNContinuousBackbone,
    SpatialBranchGate,
)
from src.models.dynamic_graph.contracts import GraphConfig, GraphOutput
from src.models.modern_tcn_graph_round1 import (
    PriorMixedDynamicGraphLearner,
    StateAwareSpatialMessagePassing,
    align_state_embeddings_to_modern_tcn_patches,
)


DenseBaseDyGraphInputMode = Literal["token", "continuous"]


@dataclass(frozen=True)
class DenseBaseDyGraphV1PriceConfig:
    num_nodes: int
    input_mode: DenseBaseDyGraphInputMode
    context_length: int = 60
    input_channels: int = 5
    d_model: int = 96
    temporal_num_heads: int = 4
    temporal_num_layers: int = 1
    spatial_num_layers: int = 1
    feedforward_multiplier: int = 2
    graph_num_heads: int = 1
    graph_hidden_dim: int = 96
    num_st_blocks: int = 4
    dropout: float = 0.0
    spatial_dropout: float = 0.0
    vocabulary_size: int = 1024

    def validate(self) -> None:
        if self.input_mode not in {"token", "continuous"}:
            raise ValueError("input_mode must be 'token' or 'continuous'.")
        for name, value in (
            ("num_nodes", self.num_nodes),
            ("context_length", self.context_length),
            ("input_channels", self.input_channels),
            ("d_model", self.d_model),
            ("temporal_num_heads", self.temporal_num_heads),
            ("temporal_num_layers", self.temporal_num_layers),
            ("spatial_num_layers", self.spatial_num_layers),
            ("feedforward_multiplier", self.feedforward_multiplier),
            ("graph_num_heads", self.graph_num_heads),
            ("graph_hidden_dim", self.graph_hidden_dim),
            ("num_st_blocks", self.num_st_blocks),
            ("vocabulary_size", self.vocabulary_size),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.temporal_num_heads:
            raise ValueError("d_model must be divisible by temporal_num_heads.")
        if self.graph_hidden_dim % self.graph_num_heads:
            raise ValueError(
                "graph_hidden_dim must be divisible by graph_num_heads."
            )
        if self.d_model % self.graph_num_heads:
            raise ValueError("d_model must be divisible by graph_num_heads.")
        for name, value in (
            ("dropout", self.dropout),
            ("spatial_dropout", self.spatial_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1).")

    def official_run_config(self) -> OfficialBaseDyGraphRunConfig:
        self.validate()
        return OfficialBaseDyGraphRunConfig(
            num_states=int(self.vocabulary_size),
            num_nodes=int(self.num_nodes),
            context_length=int(self.context_length),
            d_model=int(self.d_model),
            nhead=int(self.temporal_num_heads),
            num_temporal_layers=int(self.temporal_num_layers),
            num_spatial_layers=int(self.spatial_num_layers),
            ff_mult=int(self.feedforward_multiplier),
            num_edge_heads=int(self.graph_num_heads),
            graph_hidden_dim=int(self.graph_hidden_dim),
            dropout=float(self.dropout),
            spatial_dropout=float(self.spatial_dropout),
            spatial_module_type="dynamic_graph",
            spatial_value="hidden",
            graph_activation="softmax",
            use_node_embedding=True,
            use_state_pair_bias=False,
            add_self_loops=False,
            symmetric_graph=False,
            num_st_blocks=int(self.num_st_blocks),
            first_spatial_module_type=None,
            st_block_post_norm=True,
            dummy_state_id=0,
        )


@dataclass(frozen=True)
class DenseBaseDyGraphV1PriceOutput:
    normalised_close: Tensor  # [B,T,N,1], T=context_length
    selected_graph: Tensor  # [B,G,N,N], final predictor position
    per_layer_graphs: tuple[Tensor, ...]  # each [B,G,N,N]
    graph_sequences: tuple[Tensor, ...]  # each [B,T,G,N,N]
    predictor_hidden: Tensor  # [B,T,N,D]


class _DenseBaseDyGraphV1PriceBase(nn.Module):
    graph_orientation = OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION

    def __init__(
        self,
        config: DenseBaseDyGraphV1PriceConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.official_modules = load_official_basedygraph_architecture_modules(
            external_source_dir,
            require_pinned_commit=True,
        )
        self.official_config = build_official_model_config(
            config.official_run_config(),
            external_source_dir=external_source_dir,
            official_modules=self.official_modules,
        )
        self.backbone = self.official_modules.model.DiscreteSTGraphBackbone(
            self.official_config
        )
        self.close_head = nn.Linear(config.d_model, 1)

    @property
    def external_commit(self) -> str | None:
        return self.official_modules.commit

    def graph_parameter_ids(self) -> set[int]:
        result: set[int] = set()
        blocks = getattr(self.backbone, "st_blocks", None)
        if blocks is None:
            raise RuntimeError("The official backbone exposes no ST blocks.")
        for block in blocks:
            scorer = getattr(block, "graph_scorer", None)
            if scorer is None:
                continue
            result.update(
                id(parameter)
                for parameter in scorer.parameters()
                if parameter.requires_grad
            )
        return result

    def _result(
        self,
        memory: Tensor,
        graph_sequences: tuple[Tensor, ...],
    ) -> DenseBaseDyGraphV1PriceOutput:
        # The 61st position is the appended teacher-forced h=1 target.  The
        # direct head and graph artefacts use only the 60 predictor positions.
        predictor_hidden = memory[:, :-1].contiguous()
        predictor_graphs = tuple(
            torch.as_tensor(values)[:, :-1].contiguous()
            for values in graph_sequences
        )
        if int(predictor_hidden.shape[1]) != self.config.context_length:
            raise RuntimeError("Unexpected dense predictor length.")
        normalised_close = self.close_head(predictor_hidden)
        per_layer = tuple(values[:, -1].contiguous() for values in predictor_graphs)
        if len(per_layer) != self.config.num_st_blocks:
            raise RuntimeError("Official backbone returned the wrong graph depth.")
        selected = per_layer[-1]
        return DenseBaseDyGraphV1PriceOutput(
            normalised_close=normalised_close,
            selected_graph=selected,
            per_layer_graphs=per_layer,
            graph_sequences=predictor_graphs,
            predictor_hidden=predictor_hidden,
        )

    @staticmethod
    def _raw_graph_sequences(output: dict[str, Any]) -> tuple[Tensor, ...]:
        raw_layers = output.get("block_graph_attns")
        if raw_layers is None:
            raw_layers = (output.get("graph_attn"),)
        result = tuple(
            torch.as_tensor(value)
            for value in raw_layers
            if value is not None
        )
        if not result:
            raise RuntimeError("The dynamic BaseDyGraph returned no graph.")
        return result


class BaseDyGraphV1TokenToPriceDense(_DenseBaseDyGraphV1PriceBase):
    """Official coarse-token backbone with a dense next-Close head."""

    def __init__(
        self,
        config: DenseBaseDyGraphV1PriceConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        if config.input_mode != "token":
            raise ValueError("Token-to-price model requires input_mode='token'.")
        super().__init__(config, external_source_dir=external_source_dir)

    def forward(
        self,
        context_s1: Tensor,
        *,
        first_future_s1: Tensor,
    ) -> DenseBaseDyGraphV1PriceOutput:
        context = torch.as_tensor(context_s1).long()
        future = torch.as_tensor(first_future_s1).long()
        expected_context = (
            self.config.context_length,
            self.config.num_nodes,
        )
        if context.ndim != 3 or tuple(context.shape[1:]) != expected_context:
            raise ValueError(
                "context_s1 must have shape "
                f"[B,{expected_context[0]},{expected_context[1]}]."
            )
        if future.ndim != 2 or int(future.shape[1]) != self.config.num_nodes:
            raise ValueError("first_future_s1 must have shape [B,N].")
        if context.numel() and (
            int(context.min()) < 0
            or int(context.max()) >= self.config.vocabulary_size
        ):
            raise ValueError("context_s1 contains an invalid token ID.")
        if future.numel() and (
            int(future.min()) < 0
            or int(future.max()) >= self.config.vocabulary_size
        ):
            raise ValueError("first_future_s1 contains an invalid token ID.")

        sequence = torch.cat((context, future.unsqueeze(1)), dim=1)
        state_ids = sequence.permute(0, 2, 1).contiguous()
        output = self.backbone(state_ids)
        memory = torch.as_tensor(output["spatial_repr"])
        return self._result(memory, self._raw_graph_sequences(output))


class BaseDyGraphV1ContinuousToPriceDense(_DenseBaseDyGraphV1PriceBase):
    """Continuous OHLCV state projection with the exact v1 ST stack."""

    def __init__(
        self,
        config: DenseBaseDyGraphV1PriceConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        if config.input_mode != "continuous":
            raise ValueError(
                "Continuous-to-price model requires input_mode='continuous'."
            )
        super().__init__(config, external_source_dir=external_source_dir)
        # Remove the unused discrete state table from the continuous adapter.
        self.backbone.state_embedding = nn.Identity()
        self.input_projection = nn.Linear(config.input_channels, config.d_model)

    def forward(
        self,
        continuous_teacher_sequence: Tensor,
    ) -> DenseBaseDyGraphV1PriceOutput:
        values = torch.as_tensor(continuous_teacher_sequence)
        expected = (
            self.config.context_length + 1,
            self.config.num_nodes,
            self.config.input_channels,
        )
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                "continuous_teacher_sequence must have shape "
                f"[B,{expected[0]},{expected[1]},{expected[2]}]."
            )

        # Mirror the official discrete path exactly: the raw state embedding
        # is supplied to spatial values, while the temporal stream starts from
        # state + node identity followed by the backbone pre-normalisation.
        value_embedding = self.input_projection(values)
        hidden = value_embedding
        node_embedding = getattr(self.backbone, "node_embedding", None)
        if node_embedding is not None:
            node_ids = torch.arange(
                self.config.num_nodes,
                device=values.device,
            )
            hidden = hidden + node_embedding(node_ids).view(
                1,
                1,
                self.config.num_nodes,
                self.config.d_model,
            )
        hidden = self.backbone.pre_norm(hidden)
        state_ids = torch.zeros(
            values.shape[0],
            self.config.num_nodes,
            self.config.context_length + 1,
            dtype=torch.long,
            device=values.device,
        )

        graphs: list[Tensor] = []
        blocks = getattr(self.backbone, "st_blocks", None)
        if blocks is None:
            raise RuntimeError("The official backbone exposes no ST blocks.")
        for block in blocks:
            temporal_bntd = hidden.permute(0, 2, 1, 3).contiguous()
            temporal_bntd = block.temporal_module(temporal_bntd)
            temporal = temporal_bntd.permute(0, 2, 1, 3).contiguous()
            scorer = getattr(block, "graph_scorer", None)
            if scorer is None:
                raise RuntimeError("The dense control requires a dynamic graph.")
            attention = scorer(temporal, state_ids)
            spatial = block.spatial_module(
                temporal,
                attention,
                e=value_embedding,
            )
            hidden = block.post_norm(spatial)
            graphs.append(torch.as_tensor(attention))

        memory = self.backbone.post_norm(hidden)
        return self._result(memory, tuple(graphs))



@dataclass(frozen=True)
class ModernTCNDenseOneStepConfig:
    """Attached one-block Close-only ModernTCN under two graph controls.

    The state projection is deliberately active for both controls.  This keeps
    the attached dense model fixed while changing only whether a trainable
    random static graph is present beside the dynamic graph.
    """

    forecaster: ContinuousForecasterConfig
    use_static_graph: bool
    random_static_logit_std: float = 0.02
    random_static_seed: int = 42

    def validate(self) -> None:
        self.forecaster.validate()
        if self.forecaster.temporal.type != "modern_tcn":
            raise ValueError("Dense ModernTCN controls require ModernTCN.")
        if tuple(self.forecaster.input_channels) != ("close",):
            raise ValueError("Dense ModernTCN controls are Close-only.")
        if tuple(self.forecaster.horizons) != (1,):
            raise ValueError("Dense ModernTCN controls predict one step.")
        if self.forecaster.graph.activation != "softmax":
            raise ValueError("Dense ModernTCN controls use softmax graphs.")
        if self.forecaster.graph.add_self_loops:
            raise ValueError("Dense ModernTCN controls add no self-loop matrix.")
        if self.forecaster.spatial_gate_type != "learned_scalar":
            raise ValueError("The attached dense model requires learned beta.")
        if float(self.random_static_logit_std) < 0.0:
            raise ValueError("random_static_logit_std must be non-negative.")


@dataclass
class ModernTCNDenseOneStepOutput:
    predictions: Tensor  # [B,1,N,1], cumulative log change
    temporal_hidden: Tensor  # [B,L,N,D]
    state_hidden: Tensor  # [B,L,N,D]
    graph_spatial_hidden: Tensor
    fused_hidden: Tensor
    graph: GraphOutput
    alpha: Tensor | None
    beta: Tensor


class ModernTCNDenseOneStepGraphModel(nn.Module):
    """One-block ModernTCN with state-aware dynamic graph message passing.

    ``use_static_graph=False`` is a state-aware dynamic-only graph.  When
    ``use_static_graph=True``, a trainable static graph is included and its
    logits are initialised from random noise by the runner; no sector or
    correlation matrix enters the model.
    """

    graph_orientation = OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION

    def __init__(
        self,
        config: ModernTCNDenseOneStepConfig,
        *,
        static_scaffold: Tensor | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.use_static_graph and static_scaffold is None:
            raise ValueError("Static+dynamic control requires a scaffold.")
        if not config.use_static_graph and static_scaffold is not None:
            raise ValueError("Dynamic-only control must not receive a scaffold.")

        self.config = config
        forecaster = config.forecaster
        self.temporal_backbone = ModernTCNContinuousBackbone(config=forecaster)
        self.state_projection = nn.Linear(
            len(forecaster.input_channels),
            forecaster.temporal.d_model,
        )
        self.graph_learner = PriorMixedDynamicGraphLearner(
            d_model=forecaster.temporal.d_model,
            num_nodes=forecaster.num_nodes,
            num_heads=forecaster.graph.num_heads,
            graph_hidden_dim=forecaster.graph.hidden_dim,
            use_state_pathway=True,
            static_prior=static_scaffold,
            initial_alpha=forecaster.graph.initial_alpha,
            prior_scale=1.0,
            prior_jitter=0.0,
            prior_seed=config.random_static_seed,
        )
        self.spatial_module = StateAwareSpatialMessagePassing(
            d_model=forecaster.temporal.d_model,
            num_heads=forecaster.graph.num_heads,
            graph_hidden_dim=forecaster.graph.hidden_dim,
            feedforward_multiplier=forecaster.spatial_feedforward_multiplier,
            dropout=forecaster.spatial_dropout,
            use_state_pathway=True,
        )
        self.spatial_gate = SpatialBranchGate(
            gate_type="learned_scalar",
            initial_beta=forecaster.spatial_gate_initial_beta,
        )
        self.temporal_backbone.initialise_forecast_head(
            forecaster.output_head_initialisation
        )

    def initialise_random_static_logits(self) -> None:
        """Initialise the optional trainable static graph without a prior."""

        logits = self.graph_learner.static_logits
        if logits is None:
            if self.config.use_static_graph:
                raise RuntimeError("Static control created no static logits.")
            return
        generator = torch.Generator(device="cpu").manual_seed(
            int(self.config.random_static_seed)
        )
        values = torch.randn(
            logits.shape,
            generator=generator,
            dtype=torch.float32,
        ) * float(self.config.random_static_logit_std)
        with torch.no_grad():
            logits.copy_(values.to(device=logits.device, dtype=logits.dtype))

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

    def _state_hidden(self, x: Tensor) -> Tensor:
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
    ) -> ModernTCNDenseOneStepOutput:
        forecaster = self.config.forecaster
        expected = (
            forecaster.context_length,
            forecaster.num_nodes,
            len(forecaster.input_channels),
        )
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x must have shape [B,{expected[0]},{expected[1]},"
                f"{expected[2]}]."
            )
        temporal_hidden = self.temporal_backbone(
            x,
            context_start=context_start,
            session_length=session_length,
        )
        state_hidden = self._state_hidden(x)
        if state_hidden.shape != temporal_hidden.shape:
            raise RuntimeError("State-patch alignment differs from ModernTCN.")
        graph = self.graph_learner(
            temporal_hidden,
            state_hidden=state_hidden,
        )
        graph_hidden = self.spatial_module(
            temporal_hidden,
            graph.selected,
            state_hidden=state_hidden,
        )
        fused_hidden, beta = self.spatial_gate(
            temporal_hidden,
            graph_hidden,
        )
        predictions = self.temporal_backbone.forecast(fused_hidden)
        expected_prediction = (
            int(x.shape[0]),
            1,
            forecaster.num_nodes,
            1,
        )
        if tuple(predictions.shape) != expected_prediction:
            raise RuntimeError(
                f"Prediction shape {tuple(predictions.shape)} != "
                f"{expected_prediction}."
            )
        return ModernTCNDenseOneStepOutput(
            predictions=predictions,
            temporal_hidden=temporal_hidden,
            state_hidden=state_hidden,
            graph_spatial_hidden=graph_hidden,
            fused_hidden=fused_hidden,
            graph=graph,
            alpha=graph.alpha,
            beta=beta,
        )


def modern_tcn_dense_config_from_mapping(
    values: dict[str, Any],
    *,
    num_nodes: int,
) -> ModernTCNDenseOneStepConfig:
    data = values["data"]
    model = values["model"]
    temporal = model["temporal"]
    graph = model["graph"]
    spatial = model["spatial"]
    prior = model["prior"]
    use_static = str(graph["type"]) == "static_dynamic_mixture"

    forecaster = ContinuousForecasterConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=(1,),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        output_representation=str(model["output_representation"]),
        output_head_initialisation=str(model["output_head_initialisation"]),
        temporal=ContinuousTemporalConfig(
            type="modern_tcn",
            d_model=int(temporal["d_model"]),
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
            relative_position_embedding=False,
            session_position_encoding=bool(
                temporal["session_position_encoding"]
            ),
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
            gate_type="learned_scalar" if use_static else "none",
            initial_alpha=float(graph["initial_alpha"]),
        ),
        spatial_num_layers=int(spatial["num_layers"]),
        spatial_feedforward_multiplier=int(spatial["feedforward_multiplier"]),
        spatial_dropout=float(spatial["dropout"]),
        spatial_gate_type=str(spatial["gate_type"]),
        spatial_gate_initial_beta=float(spatial["initial_beta"]),
        head_dropout=float(model.get("head_dropout", 0.0)),
    )
    result = ModernTCNDenseOneStepConfig(
        forecaster=forecaster,
        use_static_graph=use_static,
        random_static_logit_std=float(prior["jitter"]),
        random_static_seed=int(prior["seed"]),
    )
    result.validate()
    return result

def dense_basedygraph_config_from_mapping(
    values: dict[str, Any],
    *,
    num_nodes: int,
) -> DenseBaseDyGraphV1PriceConfig:
    model = values["model"]["official_basedygraph_v1"]
    data = values["data"]
    config = DenseBaseDyGraphV1PriceConfig(
        num_nodes=int(num_nodes),
        input_mode=str(data["input_mode"]),  # type: ignore[arg-type]
        context_length=int(data["context_length"]),
        input_channels=len(data["input_channels"]),
        d_model=int(model["d_model"]),
        temporal_num_heads=int(model["temporal_num_heads"]),
        temporal_num_layers=int(model["temporal_num_layers"]),
        spatial_num_layers=int(model["spatial_num_layers"]),
        feedforward_multiplier=int(model["feedforward_multiplier"]),
        graph_num_heads=int(model["graph_num_heads"]),
        graph_hidden_dim=int(model["graph_hidden_dim"]),
        num_st_blocks=int(model["num_st_blocks"]),
        dropout=float(model["dropout"]),
        spatial_dropout=float(model["spatial_dropout"]),
        vocabulary_size=int(data.get("s1_vocabulary_size", 1024)),
    )
    config.validate()
    return config


__all__ = [
    "PINNED_BASEDYGRAPH_COMMIT",
    "DenseBaseDyGraphV1PriceConfig",
    "DenseBaseDyGraphV1PriceOutput",
    "BaseDyGraphV1TokenToPriceDense",
    "BaseDyGraphV1ContinuousToPriceDense",
    "ModernTCNDenseOneStepConfig",
    "ModernTCNDenseOneStepOutput",
    "ModernTCNDenseOneStepGraphModel",
    "dense_basedygraph_config_from_mapping",
    "modern_tcn_dense_config_from_mapping",
]

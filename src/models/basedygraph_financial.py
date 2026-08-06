from __future__ import annotations

"""Financial forecasting adapters around the pinned official BaseDyGraph blocks.

The external BaseDyGraph source remains unmodified.  This module reuses its
causal per-node Transformer, graph scorers, interlaced ST blocks, spatial
message passing, residuals and normalisation.  Only task adapters are added:

* a 60-position coarse-token future-query head;
* an official teacher-forced one-step coarse-token head; and
* direct one- or multi-horizon continuous Close heads.

All graph tensors follow ``A[target, source]``.  Internally, the official
per-timestep dynamic graph has shape ``[B,T,G,N,N]``.  The optional per-window
adaptation computes one graph from the final observed context state and
broadcasts it over the observed sequence for spatial message passing.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import torch
from torch import Tensor, nn

from src.models.basedygraph_official_adapter import (
    OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
    OfficialBaseDyGraphRunConfig,
    build_official_model_config,
    load_official_basedygraph_modules,
)
from src.models.dynamic_graph.contracts import (
    BackcastConfig,
    DynamicGraphModelConfig,
    ForecastHeadConfig,
    FuturePredictorConfig,
    GraphConfig,
    GraphOutput,
    SpatialConfig,
    TemporalConfig,
    TokenForecastOutput,
    TokenLossConfig,
)
from src.models.dynamic_graph.future_predictor import (
    FutureTokenPrediction,
    TokenSelection,
    build_future_token_predictor,
    select_token_ids,
)
from src.models.dynamic_graph.model import (
    GeneratedTokenForecast,
    SampledGeneratedTokenForecast,
)


GraphScope = Literal["per_timestep", "per_window"]


@dataclass(frozen=True)
class BaseDyGraphGraphRegularisationConfig:
    """Equation-(38) graph-shape settings for the final ST block."""

    target_entropy: float | None = None
    target_entropy_weight: float = 0.0
    temporal_smooth_weight: float = 0.0
    direct_entropy_weight: float = 0.0
    warmup_epochs: int = 0
    layer: int = -1

    def validate(self, *, graph_scope: GraphScope) -> None:
        if self.target_entropy is not None and self.target_entropy < 0.0:
            raise ValueError("target_entropy cannot be negative.")
        for name, value in (
            ("target_entropy_weight", self.target_entropy_weight),
            ("temporal_smooth_weight", self.temporal_smooth_weight),
            ("direct_entropy_weight", self.direct_entropy_weight),
        ):
            if not torch.isfinite(torch.tensor(float(value))) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative.")
        if self.temporal_smooth_weight > 0.0 and graph_scope != "per_timestep":
            raise ValueError(
                "Temporal graph smoothness is defined only for a true "
                "per-timestep graph sequence."
            )
        if self.target_entropy_weight > 0.0 and self.target_entropy is None:
            raise ValueError(
                "A positive target-entropy weight requires target_entropy."
            )


@dataclass(frozen=True)
class BaseDyGraphFinancialConfig:
    """Complete model contract for one curiosity experiment."""

    mode: Literal["token", "continuous"]
    graph_type: Literal["static_graph", "dynamic_graph"]
    graph_scope: GraphScope = "per_timestep"
    context_length: int = 60
    prediction_length: int = 60
    evaluation_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    num_nodes: int = 93
    input_channels: int = 5
    d_model: int = 96
    temporal_heads: int = 4
    temporal_layers: int = 1
    spatial_layers: int = 1
    ff_mult: int = 2
    graph_heads: int = 2
    graph_hidden_dim: int = 64
    num_st_blocks: int = 3
    dropout: float = 0.0
    spatial_dropout: float = 0.0
    use_node_embedding: bool = True
    use_state_pair_bias: bool = False
    add_self_loops: bool = False
    symmetric_graph: bool = False
    graph_activation: str = "softmax"
    # Optional per-ST-block activations. An empty tuple preserves the legacy
    # behaviour and repeats ``graph_activation`` for every block.
    graph_activations: tuple[str, ...] = ()
    spatial_value: str = "hidden"
    st_block_post_norm: bool = True
    future_predictor_layers: int = 1
    future_predictor_heads: int = 4
    future_predictor_ff_mult: int = 2
    regularisation: BaseDyGraphGraphRegularisationConfig = field(
        default_factory=BaseDyGraphGraphRegularisationConfig
    )

    @property
    def resolved_graph_activations(self) -> tuple[str, ...]:
        """Return one graph activation for every interlaced ST block."""
        if self.graph_activations:
            return tuple(str(value) for value in self.graph_activations)
        return tuple(str(self.graph_activation) for _ in range(self.num_st_blocks))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the exact task adapter and graph contract.

        ``graph_activations`` is omitted when empty so checkpoints produced
        before layer-specific activations were added retain the same run
        signature and remain resumable.
        """
        values = asdict(self)
        if not self.graph_activations:
            values.pop("graph_activations", None)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BaseDyGraphFinancialConfig":
        payload = dict(values)
        regularisation = payload.get("regularisation", {})
        if not isinstance(regularisation, BaseDyGraphGraphRegularisationConfig):
            regularisation = BaseDyGraphGraphRegularisationConfig(**dict(regularisation))
        payload["regularisation"] = regularisation
        if "evaluation_horizons" in payload:
            payload["evaluation_horizons"] = tuple(
                int(value) for value in payload["evaluation_horizons"]
            )
        if "graph_activations" in payload:
            payload["graph_activations"] = tuple(
                str(value) for value in payload["graph_activations"]
            )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode not in {"token", "continuous"}:
            raise ValueError("mode must be 'token' or 'continuous'.")
        if self.graph_type not in {"static_graph", "dynamic_graph"}:
            raise ValueError("Unsupported BaseDyGraph graph type.")
        if self.graph_scope not in {"per_timestep", "per_window"}:
            raise ValueError("Unsupported graph scope.")
        if self.graph_type == "static_graph" and self.graph_scope != "per_timestep":
            raise ValueError("Static BaseDyGraph uses the official per-timestep broadcast.")
        if self.context_length <= 0 or self.prediction_length <= 0:
            raise ValueError("Context and prediction lengths must be positive.")
        if tuple(sorted(self.evaluation_horizons)) != self.evaluation_horizons:
            raise ValueError("evaluation_horizons must be strictly increasing.")
        if max(self.evaluation_horizons) > self.prediction_length:
            raise ValueError("Evaluation horizon exceeds prediction_length.")
        if self.num_nodes <= 1 or self.input_channels <= 0:
            raise ValueError("Invalid node/channel count.")
        if self.d_model <= 0 or self.d_model % self.temporal_heads != 0:
            raise ValueError("d_model must be divisible by temporal_heads.")
        if self.d_model % self.graph_heads != 0:
            raise ValueError("d_model must be divisible by graph_heads.")
        if self.graph_hidden_dim % self.graph_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by graph_heads.")
        if self.num_st_blocks <= 0:
            raise ValueError("num_st_blocks must be positive.")
        allowed_activations = {"softmax", "sparsemax", "entmax15", "gated"}
        activations = self.resolved_graph_activations
        if len(activations) != self.num_st_blocks:
            raise ValueError(
                "graph_activations must contain exactly one value per ST block."
            )
        unsupported = [value for value in activations if value not in allowed_activations]
        if unsupported:
            raise ValueError(f"Unsupported graph activations: {unsupported}.")
        if self.graph_type == "static_graph" and any(
            value != "softmax" for value in activations
        ):
            raise ValueError(
                "The pinned official StaticGraphScorer is softmax-only; "
                "layer-specific sparse activations require dynamic_graph."
            )
        if self.future_predictor_layers < 0:
            raise ValueError("future_predictor_layers cannot be negative.")
        if self.d_model % self.future_predictor_heads != 0:
            raise ValueError("d_model must be divisible by future predictor heads.")
        self.regularisation.validate(graph_scope=self.graph_scope)

    def official_run_config(self) -> OfficialBaseDyGraphRunConfig:
        self.validate()
        return OfficialBaseDyGraphRunConfig(
            num_states=1024,
            num_nodes=self.num_nodes,
            context_length=self.context_length,
            d_model=self.d_model,
            nhead=self.temporal_heads,
            num_temporal_layers=self.temporal_layers,
            num_spatial_layers=self.spatial_layers,
            ff_mult=self.ff_mult,
            num_edge_heads=self.graph_heads,
            graph_hidden_dim=self.graph_hidden_dim,
            dropout=self.dropout,
            spatial_dropout=self.spatial_dropout,
            spatial_module_type=self.graph_type,
            spatial_value=self.spatial_value,
            graph_activation=self.resolved_graph_activations[0],
            use_node_embedding=self.use_node_embedding,
            use_state_pair_bias=self.use_state_pair_bias,
            add_self_loops=self.add_self_loops,
            symmetric_graph=self.symmetric_graph,
            num_st_blocks=self.num_st_blocks,
            first_spatial_module_type=None,
            st_block_post_norm=self.st_block_post_norm,
            dummy_state_id=0,
        )

    def token_contract(self) -> DynamicGraphModelConfig:
        """Build the standard token-output contract used by shared decoding."""
        if self.mode != "token":
            raise ValueError("token_contract is available only in token mode.")
        graph_type = "free_static" if self.graph_type == "static_graph" else "dynamic"
        config = DynamicGraphModelConfig(
            num_nodes=self.num_nodes,
            context_length=self.context_length,
            d_model=self.d_model,
            num_st_blocks=self.num_st_blocks,
            use_node_embedding=self.use_node_embedding,
            token_input_representation="hierarchical_embedding",
            temporal=TemporalConfig(
                type="identity",
                num_layers=1,
                num_heads=self.temporal_heads,
                feedforward_multiplier=self.ff_mult,
                dropout=self.dropout,
            ),
            graph=GraphConfig(
                type=graph_type,
                num_heads=self.graph_heads,
                hidden_dim=self.graph_hidden_dim,
                activation=self.resolved_graph_activations[-1],
                add_self_loops=self.add_self_loops,
            ),
            spatial=SpatialConfig(
                num_layers=1,
                feedforward_multiplier=self.ff_mult,
                dropout=self.spatial_dropout,
                gate_type="none",
                initial_beta=1.0,
            ),
            heads=ForecastHeadConfig(
                prediction_length=self.prediction_length,
                evaluation_horizons=self.evaluation_horizons,
                s1_vocabulary_size=1024,
                s2_vocabulary_size=1024,
                s2_loss_weight=0.0,
                future_token_mode="coarse_only",
                s2_conditioning="predicted_s1",
            ),
            future_predictor=FuturePredictorConfig(
                type="structured_parallel",
                num_layers=self.future_predictor_layers,
                num_heads=self.future_predictor_heads,
                feedforward_multiplier=self.future_predictor_ff_mult,
                dropout=0.0,
            ),
            loss=TokenLossConfig(horizon_weighting="uniform"),
            backcast=BackcastConfig(enabled=False),
        )
        config.validate()
        return config


@dataclass
class BaseDyGraphContextEncoding:
    context_memory: Tensor  # [B,T,N,D]
    context_hidden: Tensor  # [B,N,D]
    graph: GraphOutput  # final-context graphs for normal analysis
    graph_sequences: tuple[Tensor | None, ...]  # [B,T,G,N,N]


@dataclass
class BaseDyGraphTokenTrainingOutput:
    forecast: TokenForecastOutput
    prediction: FutureTokenPrediction
    graph_sequences: tuple[Tensor | None, ...]


@dataclass
class BaseDyGraphTeacherForcedTokenOutput:
    """Teacher-forced next-token outputs over all context transitions."""

    s1_logits: Tensor  # [B,T,N,1024]
    forecast: TokenForecastOutput  # final unseen one-step forecast
    graph_sequences: tuple[Tensor | None, ...]  # [B,T,G,N,N]


@dataclass
class BaseDyGraphContinuousOutput:
    predictions: Tensor  # [B,H,N,1], normalised Close levels
    context_memory: Tensor
    context_hidden: Tensor
    graph: GraphOutput
    graph_sequences: tuple[Tensor | None, ...]


@dataclass
class GraphRegularisationResult:
    total: Tensor
    target_entropy_penalty: Tensor
    temporal_smoothness_penalty: Tensor
    direct_entropy_penalty: Tensor
    mean_row_entropy: Tensor
    mean_effective_neighbours: Tensor
    warmup_factor: float


def _normalise_layer_index(index: int, length: int) -> int:
    resolved = int(index)
    if resolved < 0:
        resolved += length
    if not 0 <= resolved < length:
        raise IndexError(f"Graph regularisation layer {index} is out of range.")
    return resolved


def graph_regularisation_loss(
    graph_sequences: tuple[Tensor | None, ...],
    config: BaseDyGraphGraphRegularisationConfig,
    *,
    epoch: int,
) -> GraphRegularisationResult:
    """Apply Equation (38) to the configured ST-block graph.

    The entropy terms average across batch, context time, graph heads and
    target rows.  The smoothness term preserves the Frobenius sum over the
    target/source matrix and averages across batch, adjacent context times and
    graph heads.
    """
    if not graph_sequences:
        zero = torch.tensor(0.0)
        return GraphRegularisationResult(zero, zero, zero, zero, zero, zero, 1.0)

    layer_index = _normalise_layer_index(config.layer, len(graph_sequences))
    graph = graph_sequences[layer_index]
    if graph is None:
        raise ValueError("The selected regularisation layer has no graph.")
    values = torch.as_tensor(graph).float().clamp_min(1.0e-12)
    if values.ndim != 5:
        raise ValueError("Graph sequence must have shape [B,T,G,N,N].")

    entropy_by_row = -(values * values.log()).sum(dim=-1)
    mean_entropy = entropy_by_row.mean()
    mean_effective = entropy_by_row.exp().mean()

    target_penalty = values.new_zeros(())
    if config.target_entropy is not None:
        target_penalty = (mean_entropy - float(config.target_entropy)).square()

    smooth_penalty = values.new_zeros(())
    if config.temporal_smooth_weight > 0.0:
        if int(values.shape[1]) < 2:
            raise ValueError("Temporal smoothness needs at least two graph steps.")
        differences = values[:, 1:] - values[:, :-1]
        smooth_penalty = differences.square().sum(dim=(-1, -2)).mean()

    direct_entropy_penalty = mean_entropy
    if config.warmup_epochs <= 0:
        warmup = 1.0
    else:
        warmup = min(max(int(epoch), 1) / float(config.warmup_epochs), 1.0)

    total = warmup * (
        float(config.target_entropy_weight) * target_penalty
        + float(config.temporal_smooth_weight) * smooth_penalty
        + float(config.direct_entropy_weight) * direct_entropy_penalty
    )
    return GraphRegularisationResult(
        total=total,
        target_entropy_penalty=target_penalty,
        temporal_smoothness_penalty=smooth_penalty,
        direct_entropy_penalty=direct_entropy_penalty,
        mean_row_entropy=mean_entropy,
        mean_effective_neighbours=mean_effective,
        warmup_factor=warmup,
    )


class _OfficialContextEncoderBase(nn.Module):
    """Shared official ST-block orchestration for token and continuous inputs."""

    def __init__(
        self,
        config: BaseDyGraphFinancialConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.official_modules = load_official_basedygraph_modules(external_source_dir)
        self.official_config = build_official_model_config(
            config.official_run_config(),
            external_source_dir=external_source_dir,
        )

    def _apply_layer_graph_activations(self, backbone: nn.Module) -> None:
        """Assign one activation to each official interlaced graph scorer.

        The pinned BaseDyGraph configuration exposes one global activation.
        For the financial diagnostics we preserve the official scorer classes
        and only replace each scorer's immutable config object after the
        backbone is constructed. No external source file is modified.

        Legacy configurations leave ``graph_activations`` empty. In that case
        the official backbone already received the single global activation,
        so no post-construction mutation is required. This also preserves old
        checkpoints and test doubles that do not expose a scorer ``cfg``.
        """
        financial_config = getattr(self, "financial_config", self.config)
        if not financial_config.graph_activations:
            return

        blocks = getattr(backbone, "st_blocks", None)
        if blocks is None:
            raise RuntimeError(
                "Layer-specific graph activations require interlaced ST blocks."
            )
        activations = financial_config.resolved_graph_activations
        if len(blocks) != len(activations):
            raise AssertionError(
                "Official ST-block count differs from graph_activations."
            )
        for block_index, (block, activation) in enumerate(
            zip(blocks, activations, strict=True)
        ):
            scorer = getattr(block, "graph_scorer", None)
            if scorer is None:
                continue
            scorer_config = getattr(scorer, "cfg", None)
            if scorer_config is None:
                raise RuntimeError(
                    f"ST block {block_index} graph scorer exposes no cfg."
                )
            try:
                scorer.cfg = replace(
                    scorer_config,
                    graph_activation=str(activation),
                )
            except (TypeError, ValueError):
                # Fallback for a non-dataclass configuration implementation.
                setattr(scorer_config, "graph_activation", str(activation))
                scorer.cfg = scorer_config

    @staticmethod
    def _final_context_graph(values: Tensor | None) -> Tensor | None:
        if values is None:
            return None
        graph = torch.as_tensor(values)
        if graph.ndim != 5:
            raise ValueError("Official graph must have shape [B,T,G,N,N].")
        return graph[:, -1].contiguous()

    def _manual_interlaced_forward(
        self,
        *,
        backbone: nn.Module,
        initial_hidden: Tensor,
        state_ids: Tensor,
        value_embedding: Tensor,
    ) -> tuple[Tensor, tuple[Tensor | None, ...]]:
        if getattr(backbone, "st_blocks", None) is None:
            raise RuntimeError("The financial adapter requires interlaced ST blocks.")
        financial_config = getattr(self, "financial_config", self.config)
        hidden = initial_hidden
        graph_sequences: list[Tensor | None] = []

        for block in backbone.st_blocks:
            temporal_bntd = hidden.permute(0, 2, 1, 3).contiguous()
            temporal_bntd = block.temporal_module(temporal_bntd)
            temporal = temporal_bntd.permute(0, 2, 1, 3).contiguous()

            scorer = getattr(block, "graph_scorer", None)
            if scorer is None:
                attention = None
                spatial = block.spatial_module(temporal, None, e=value_embedding)
            else:
                if (
                    financial_config.graph_type == "dynamic_graph"
                    and financial_config.graph_scope == "per_window"
                ):
                    final_hidden = temporal[:, -1:, :, :]
                    final_state = state_ids[:, :, -1:]
                    one_graph = scorer(final_hidden, final_state)
                    attention = one_graph.expand(
                        -1, temporal.shape[1], -1, -1, -1
                    ).contiguous()
                else:
                    attention = scorer(temporal, state_ids)
                spatial = block.spatial_module(
                    temporal,
                    attention,
                    e=value_embedding,
                )
            hidden = block.post_norm(spatial)
            graph_sequences.append(attention)

        return backbone.post_norm(hidden), tuple(graph_sequences)

    def _build_encoding(
        self,
        context_memory: Tensor,
        graph_sequences: tuple[Tensor | None, ...],
    ) -> BaseDyGraphContextEncoding:
        financial_config = getattr(self, "financial_config", self.config)
        per_layer = tuple(self._final_context_graph(value) for value in graph_sequences)
        selected = next((value for value in reversed(per_layer) if value is not None), None)
        graph_output = GraphOutput(selected=selected, per_layer=per_layer)
        graph_output.validate(
            batch_size=int(context_memory.shape[0]),
            num_heads=financial_config.graph_heads,
            num_nodes=financial_config.num_nodes,
        )
        return BaseDyGraphContextEncoding(
            context_memory=context_memory,
            context_hidden=context_memory[:, -1].contiguous(),
            graph=graph_output,
            graph_sequences=graph_sequences,
        )


class OfficialBaseDyGraphTokenContextEncoder(_OfficialContextEncoderBase):
    """Exact official s1-token BaseDyGraph context backbone."""

    def __init__(self, config: BaseDyGraphFinancialConfig, *, external_source_dir: str | None = None) -> None:
        if config.mode != "token":
            raise ValueError("Token context encoder requires mode='token'.")
        super().__init__(config, external_source_dir=external_source_dir)
        self.backbone = self.official_modules.model.DiscreteSTGraphBackbone(
            self.official_config
        )
        self._apply_layer_graph_activations(self.backbone)

    def forward(self, context_s1: Tensor) -> BaseDyGraphContextEncoding:
        values = torch.as_tensor(context_s1).long()
        expected = (self.config.context_length, self.config.num_nodes)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"context_s1 must have shape [B,{expected[0]},{expected[1]}]."
            )
        if values.numel() and (values.min() < 0 or values.max() >= 1024):
            raise ValueError("context_s1 contains an invalid Kronos s1 ID.")
        state_ids = values.permute(0, 2, 1).contiguous()

        if self.config.graph_scope == "per_timestep":
            output = self.backbone(state_ids)
            memory = torch.as_tensor(output["spatial_repr"])
            raw_layers = output.get("block_graph_attns")
            if raw_layers is None:
                raw_layers = (output.get("graph_attn"),)
            graph_sequences = tuple(
                None if value is None else torch.as_tensor(value)
                for value in raw_layers
            )
        else:
            initial = self.backbone._initial_embedding_bntd(state_ids)
            initial_btnd = initial.permute(0, 2, 1, 3).contiguous()
            value_embedding = self.backbone.state_embedding_btnd(state_ids)
            memory, graph_sequences = self._manual_interlaced_forward(
                backbone=self.backbone,
                initial_hidden=initial_btnd,
                state_ids=state_ids,
                value_embedding=value_embedding,
            )
        return self._build_encoding(memory, graph_sequences)


class OfficialBaseDyGraphCoarsePathForecaster(nn.Module):
    """Official BaseDyGraph context backbone plus a 60-position s1 head."""

    def __init__(self, config: BaseDyGraphFinancialConfig, *, external_source_dir: str | None = None) -> None:
        super().__init__()
        if config.mode != "token":
            raise ValueError("Coarse path forecaster requires token mode.")
        self.financial_config = config
        self.config = config.token_contract()
        self.context_encoder = OfficialBaseDyGraphTokenContextEncoder(
            config,
            external_source_dir=external_source_dir,
        )
        self.future_predictor = build_future_token_predictor(self.config)

    @property
    def external_commit(self) -> str | None:
        return self.context_encoder.official_modules.commit

    def _forecast_output(
        self,
        encoding: BaseDyGraphContextEncoding,
        prediction: FutureTokenPrediction,
    ) -> TokenForecastOutput:
        output = TokenForecastOutput(
            s1_logits=prediction.s1_logits,
            s2_logits=None,
            graph=encoding.graph,
            context_hidden=encoding.context_hidden,
            temporal_hidden=encoding.context_memory,
            future_hidden=prediction.future_hidden,
            spatial_beta=None,
            backcast=None,
        )
        output.validate(self.config, batch_size=int(encoding.context_memory.shape[0]))
        return output

    def forward(
        self,
        token_ids: Tensor,
        *,
        target_s1: Tensor | None = None,
        target_s2: Tensor | None = None,
        context_mean: Tensor | None = None,
        context_std: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> BaseDyGraphTokenTrainingOutput:
        del target_s2, context_mean, context_std
        pairs = torch.as_tensor(token_ids)
        if pairs.ndim != 4 or int(pairs.shape[-1]) != 2:
            raise ValueError("token_ids must have shape [B,T,N,2].")
        encoding = self.context_encoder(pairs[..., 0])
        prediction = self.future_predictor(
            encoding.context_memory,
            s1_embedding=None,
            s2_embedding=None,
            target_s1=target_s1,
            target_s2=None,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return BaseDyGraphTokenTrainingOutput(
            forecast=self._forecast_output(encoding, prediction),
            prediction=prediction,
            graph_sequences=encoding.graph_sequences,
        )

    def generate(
        self,
        token_ids: Tensor,
        *,
        context_mean: Tensor | None = None,
        context_std: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **_: Any,
    ) -> GeneratedTokenForecast:
        output = self.forward(
            token_ids,
            context_mean=context_mean,
            context_std=context_std,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        selected = output.prediction.selected_s1
        token_path = torch.stack((selected, torch.zeros_like(selected)), dim=-1)
        result = GeneratedTokenForecast(token_ids=token_path, forecast=output.forecast)
        result.validate(self.config, batch_size=int(token_ids.shape[0]))
        return result

    def generate_samples(
        self,
        token_ids: Tensor,
        *,
        sample_count: int,
        context_mean: Tensor | None = None,
        context_std: Tensor | None = None,
        token_selection: TokenSelection = "sample",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **_: Any,
    ) -> SampledGeneratedTokenForecast:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        output = self.forward(
            token_ids,
            context_mean=context_mean,
            context_std=context_std,
            token_selection="argmax",
        )
        samples: list[Tensor] = []
        for _sample_index in range(int(sample_count)):
            selected = select_token_ids(
                output.prediction.s1_logits,
                mode=token_selection,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            samples.append(torch.stack((selected, torch.zeros_like(selected)), dim=-1))
        result = SampledGeneratedTokenForecast(
            token_ids=torch.stack(samples, dim=0),
            forecast=output.forecast,
        )
        result.validate(
            self.config,
            sample_count=int(sample_count),
            batch_size=int(token_ids.shape[0]),
        )
        return result


class OfficialBaseDyGraphTokenToPriceForecaster(nn.Module):
    """Use the s1-token BaseDyGraph context backbone for direct Close prediction.

    The graph/temporal/spatial pathway is identical to the token-input
    BaseDyGraph context encoder.  Only the task head changes: the final context
    state is projected to one or more context-normalised Close levels.  The
    runner applies the causal context Close mean/std stored in the token cache
    to recover raw prices before computing cumulative-log-change MAE.
    """

    def __init__(
        self,
        config: BaseDyGraphFinancialConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        super().__init__()
        if config.mode != "token":
            raise ValueError("Token-to-price forecaster requires token mode.")
        if config.prediction_length != 1 or config.evaluation_horizons != (1,):
            raise ValueError(
                "The current token-to-price diagnostic predicts only horizon 1."
            )
        self.financial_config = config
        self.config = config
        self.context_encoder = OfficialBaseDyGraphTokenContextEncoder(
            config,
            external_source_dir=external_source_dir,
        )
        self.output_head = nn.Linear(
            config.d_model,
            len(config.evaluation_horizons),
        )

    @property
    def external_commit(self) -> str | None:
        return self.context_encoder.official_modules.commit

    def forward(self, token_ids: Tensor) -> BaseDyGraphContinuousOutput:
        pairs = torch.as_tensor(token_ids)
        if pairs.ndim != 4 or int(pairs.shape[-1]) != 2:
            raise ValueError("token_ids must have shape [B,T,N,2].")
        if int(pairs.shape[1]) != self.config.context_length:
            raise ValueError("Token context length differs from the model contract.")

        # This diagnostic deliberately retains only the native Kronos coarse
        # token, exactly like the teacher-forced token run.  s2 is not used.
        encoding = self.context_encoder(pairs[..., 0])
        predictions = self.output_head(encoding.context_hidden)
        predictions = predictions.permute(0, 2, 1).unsqueeze(-1).contiguous()
        return BaseDyGraphContinuousOutput(
            predictions=predictions,
            context_memory=encoding.context_memory,
            context_hidden=encoding.context_hidden,
            graph=encoding.graph,
            graph_sequences=encoding.graph_sequences,
        )


class OfficialBaseDyGraphTeacherForcedOneStepForecaster(
    _OfficialContextEncoderBase
):
    """Official one-step BaseDyGraph objective over a 60-minute token context.

    During training the first future coarse token is appended to the observed
    context. The official causal backbone then predicts each next token from
    the preceding position, giving 60 teacher-forced transitions:

    ``context[0] -> context[1]``, ..., ``context[59] -> future[0]``.

    The appended future token is never visible to the representation used for
    the final context-to-future prediction. Graph regularisation is likewise
    applied only to the 60 predictor positions and excludes the appended token.
    """

    def __init__(
        self,
        config: BaseDyGraphFinancialConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        if config.mode != "token":
            raise ValueError("Teacher-forced one-step forecaster needs token mode.")
        if config.prediction_length != 1 or config.evaluation_horizons != (1,):
            raise ValueError(
                "Teacher-forced one-step token mode requires prediction_length=1 "
                "and evaluation_horizons=(1,)."
            )
        super().__init__(config, external_source_dir=external_source_dir)
        self.financial_config = config
        self.config = config.token_contract()
        self.backbone = self.official_modules.model.DiscreteSTGraphBackbone(
            self.official_config
        )
        self._apply_layer_graph_activations(self.backbone)
        self.next_state_head = self.official_modules.model.NextStateHead(
            config.d_model,
            1024,
        )

    @property
    def external_commit(self) -> str | None:
        return self.official_modules.commit

    def _encode_sequence(
        self,
        sequence_s1: Tensor,
    ) -> tuple[Tensor, tuple[Tensor | None, ...]]:
        values = torch.as_tensor(sequence_s1).long()
        if values.ndim != 3 or int(values.shape[2]) != self.financial_config.num_nodes:
            raise ValueError("sequence_s1 must have shape [B,T,N].")
        if values.numel() and (values.min() < 0 or values.max() >= 1024):
            raise ValueError("sequence_s1 contains an invalid Kronos s1 ID.")
        state_ids = values.permute(0, 2, 1).contiguous()
        output = self.backbone(state_ids)
        memory = torch.as_tensor(output["spatial_repr"])
        raw_layers = output.get("block_graph_attns")
        if raw_layers is None:
            raw_layers = (output.get("graph_attn"),)
        graph_sequences = tuple(
            None if value is None else torch.as_tensor(value)
            for value in raw_layers
        )
        return memory, graph_sequences

    @staticmethod
    def teacher_targets(token_ids: Tensor, target_s1: Tensor) -> Tensor:
        pairs = torch.as_tensor(token_ids)
        future = torch.as_tensor(target_s1).long()
        if pairs.ndim != 4 or int(pairs.shape[-1]) != 2:
            raise ValueError("token_ids must have shape [B,T,N,2].")
        if future.ndim != 3 or int(future.shape[1]) < 1:
            raise ValueError("target_s1 must have shape [B,P,N] with P>=1.")
        context_s1 = pairs[..., 0].long()
        return torch.cat(
            (context_s1[:, 1:], future[:, :1]),
            dim=1,
        ).contiguous()

    def _one_step_forecast(
        self,
        encoding: BaseDyGraphContextEncoding,
        logits: Tensor,
    ) -> TokenForecastOutput:
        output = TokenForecastOutput(
            s1_logits=logits,
            s2_logits=None,
            graph=encoding.graph,
            context_hidden=encoding.context_hidden,
            temporal_hidden=encoding.context_memory,
            future_hidden=encoding.context_hidden.unsqueeze(1),
            spatial_beta=None,
            backcast=None,
        )
        output.validate(
            self.config,
            batch_size=int(encoding.context_memory.shape[0]),
        )
        return output

    def forward(
        self,
        token_ids: Tensor,
        *,
        target_s1: Tensor,
        target_s2: Tensor | None = None,
        context_mean: Tensor | None = None,
        context_std: Tensor | None = None,
        **_: Any,
    ) -> BaseDyGraphTeacherForcedTokenOutput:
        del target_s2, context_mean, context_std
        pairs = torch.as_tensor(token_ids)
        targets = torch.as_tensor(target_s1).long()
        if pairs.ndim != 4 or int(pairs.shape[-1]) != 2:
            raise ValueError("token_ids must have shape [B,T,N,2].")
        if int(pairs.shape[1]) != self.financial_config.context_length:
            raise ValueError("Token context length differs from the model contract.")
        if targets.ndim != 3 or int(targets.shape[1]) < 1:
            raise ValueError("target_s1 must contain the first future token.")

        teacher_sequence = torch.cat(
            (pairs[..., 0].long(), targets[:, :1]),
            dim=1,
        )
        full_memory, full_graphs = self._encode_sequence(teacher_sequence)
        logits = self.next_state_head(full_memory)
        logits = logits.permute(0, 2, 1, 3).contiguous()
        expected = (
            int(pairs.shape[0]),
            self.financial_config.context_length,
            self.financial_config.num_nodes,
            1024,
        )
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"Unexpected teacher-forced logit shape {tuple(logits.shape)}; "
                f"expected {expected}."
            )

        predictor_memory = full_memory[:, :-1].contiguous()
        predictor_graphs = tuple(
            None if graph is None else graph[:, :-1].contiguous()
            for graph in full_graphs
        )
        encoding = self._build_encoding(predictor_memory, predictor_graphs)
        forecast = self._one_step_forecast(
            encoding,
            logits[:, -1:].contiguous(),
        )
        return BaseDyGraphTeacherForcedTokenOutput(
            s1_logits=logits,
            forecast=forecast,
            graph_sequences=predictor_graphs,
        )

    def predict_next(self, token_ids: Tensor) -> TokenForecastOutput:
        pairs = torch.as_tensor(token_ids)
        if pairs.ndim != 4 or int(pairs.shape[-1]) != 2:
            raise ValueError("token_ids must have shape [B,T,N,2].")
        if int(pairs.shape[1]) != self.financial_config.context_length:
            raise ValueError("Token context length differs from the model contract.")
        memory, graph_sequences = self._encode_sequence(pairs[..., 0])
        encoding = self._build_encoding(memory, graph_sequences)
        logits = self.next_state_head.proj(encoding.context_hidden).unsqueeze(1)
        return self._one_step_forecast(encoding, logits)

    def generate(
        self,
        token_ids: Tensor,
        *,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **_: Any,
    ) -> GeneratedTokenForecast:
        forecast = self.predict_next(token_ids)
        selected = select_token_ids(
            forecast.s1_logits,
            mode=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_path = torch.stack((selected, torch.zeros_like(selected)), dim=-1)
        result = GeneratedTokenForecast(token_ids=token_path, forecast=forecast)
        result.validate(self.config, batch_size=int(token_ids.shape[0]))
        return result

    def generate_samples(
        self,
        token_ids: Tensor,
        *,
        sample_count: int,
        token_selection: TokenSelection = "sample",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        **_: Any,
    ) -> SampledGeneratedTokenForecast:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        forecast = self.predict_next(token_ids)
        samples = []
        for _ in range(int(sample_count)):
            selected = select_token_ids(
                forecast.s1_logits,
                mode=token_selection,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            samples.append(
                torch.stack((selected, torch.zeros_like(selected)), dim=-1)
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


class OfficialBaseDyGraphContinuousForecaster(_OfficialContextEncoderBase):
    """Continuous OHLCV adapter around the exact official interlaced blocks."""

    def __init__(self, config: BaseDyGraphFinancialConfig, *, external_source_dir: str | None = None) -> None:
        if config.mode != "continuous":
            raise ValueError("Continuous forecaster requires continuous mode.")
        super().__init__(config, external_source_dir=external_source_dir)
        self.backbone = self.official_modules.model.DiscreteSTGraphBackbone(
            self.official_config
        )
        self._apply_layer_graph_activations(self.backbone)
        # The discrete state embedding is not part of the continuous adapter.
        # Replace it so no unused trainable token table is carried by the model.
        self.backbone.state_embedding = nn.Identity()
        self.input_projection = nn.Linear(config.input_channels, config.d_model)
        self.output_head = nn.Linear(config.d_model, len(config.evaluation_horizons))

    @property
    def external_commit(self) -> str | None:
        return self.official_modules.commit

    def forward(self, x: Tensor) -> BaseDyGraphContinuousOutput:
        values = torch.as_tensor(x)
        expected = (
            self.config.context_length,
            self.config.num_nodes,
            self.config.input_channels,
        )
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                "x must have shape [B,T,N,C] with "
                f"T,N,C={expected}; observed {tuple(values.shape)}."
            )
        hidden = self.input_projection(values)
        if self.backbone.node_embedding is not None:
            node_ids = torch.arange(self.config.num_nodes, device=values.device)
            hidden = hidden + self.backbone.node_embedding(node_ids).view(
                1, 1, self.config.num_nodes, self.config.d_model
            )
        hidden = self.backbone.pre_norm(hidden)
        state_ids = torch.zeros(
            values.shape[0],
            self.config.num_nodes,
            self.config.context_length,
            dtype=torch.long,
            device=values.device,
        )
        memory, graph_sequences = self._manual_interlaced_forward(
            backbone=self.backbone,
            initial_hidden=hidden,
            state_ids=state_ids,
            value_embedding=hidden,
        )
        encoding = self._build_encoding(memory, graph_sequences)
        predictions = self.output_head(encoding.context_hidden)
        predictions = predictions.permute(0, 2, 1).unsqueeze(-1).contiguous()
        return BaseDyGraphContinuousOutput(
            predictions=predictions,
            context_memory=encoding.context_memory,
            context_hidden=encoding.context_hidden,
            graph=encoding.graph,
            graph_sequences=encoding.graph_sequences,
        )


def financial_graph_artifact_metadata(config: BaseDyGraphFinancialConfig) -> dict[str, Any]:
    return {
        "graph_type": (
            "static_graph" if config.graph_type == "static_graph" else "dynamic_graph"
        ),
        "graph_scope": config.graph_scope,
        "graph_orientation": OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
        "diagonal_policy": (
            "eligible_in_official_softmax; add_self_loops controls only "
            "additional identity mass"
        ),
        "graph_time_reduction": "final_observed_context_position",
        "num_st_blocks": config.num_st_blocks,
        "num_graph_heads": config.graph_heads,
        "graph_activations_by_layer": list(config.resolved_graph_activations),
    }

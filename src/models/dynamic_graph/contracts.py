from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Mapping, Sequence

import torch
from torch import Tensor


DEFAULT_PREDICTION_LENGTH = 60
DEFAULT_EVALUATION_HORIZONS = (1, 5, 15, 30, 60)
# Backwards-compatible alias used only by older notebook/config code.
DEFAULT_HORIZONS = DEFAULT_EVALUATION_HORIZONS
GRAPH_ORIENTATION = "row=target,column=source"

TemporalType = Literal[
    "identity",
    "transformer",
    "tcn",
    "modern_tcn",
]

TokenInputRepresentation = Literal[
    "hierarchical_embedding",
    "bsq_bits",
]

GraphType = Literal[
    "none",
    "fixed",
    "free_static",
    "mtgnn_static",
    "dynamic",
    "dynamic_correlation",
    "dynamic_base",
    "oracle",
]

GraphActivation = Literal[
    "softmax",
    "sparsemax",
    "entmax15",
    "gated",
]

StaticGraphType = Literal[
    "free_static",
    "mtgnn_static",
]

GateType = Literal[
    "none",
    "fixed",
    "learned_scalar",
    "learned_per_head",
]


FuturePredictorType = Literal[
    "structured_parallel",
    "autoregressive",
]

S2Conditioning = Literal[
    "true_s1",
    "predicted_s1",
]

FutureTokenMode = Literal[
    "full",
    "coarse_only",
]

HorizonWeighting = Literal[
    "uniform",
    "exponential_decay",
]


def _validate_probability(
    value: float,
    *,
    name: str,
    inclusive_upper: bool = True,
) -> None:
    value = float(value)

    upper_ok = (
        value <= 1.0
        if inclusive_upper
        else value < 1.0
    )

    if not 0.0 <= value or not upper_ok:
        upper_symbol = "]" if inclusive_upper else ")"
        raise ValueError(
            f"{name} must lie in [0, 1{upper_symbol}. "
            f"Received {value}."
        )


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer. "
            f"Received {value!r}."
        )


@dataclass(frozen=True)
class TemporalConfig:
    """Configuration shared by all node-wise temporal encoders.

    The temporal encoder must process each node independently:

        input:  [B, T, N, D]
        output: [B, T, N, D]

    Cross-node information is forbidden inside this module and first
    enters through the explicit graph/spatial stage.
    """

    type: TemporalType = "transformer"
    num_layers: int = 1
    num_heads: int = 4
    feedforward_multiplier: int = 2
    dropout: float = 0.0

    # TCN-only options.
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)

    # Official per-asset ModernTCN options.  Depending on
    # token_input_representation, the variable axis is either the exact
    # 20-dimensional post-BSQ code or the learned D-dimensional hierarchical
    # token embedding.
    modern_tcn_patch_size: int = 8
    modern_tcn_patch_stride: int = 4
    modern_tcn_ffn_ratio: int = 1
    modern_tcn_num_blocks: int = 1
    modern_tcn_large_kernel: int = 15
    modern_tcn_small_kernel: int = 5
    modern_tcn_dropout: float = 0.05

    def validate(
        self,
        *,
        d_model: int,
    ) -> None:
        if self.type not in {
            "identity",
            "transformer",
            "tcn",
            "modern_tcn",
        }:
            raise ValueError(
                f"Unsupported temporal type {self.type!r}."
            )

        _validate_positive_integer(
            self.num_layers,
            name="temporal.num_layers",
        )

        _validate_positive_integer(
            self.num_heads,
            name="temporal.num_heads",
        )

        _validate_positive_integer(
            self.feedforward_multiplier,
            name="temporal.feedforward_multiplier",
        )

        _validate_probability(
            self.dropout,
            name="temporal.dropout",
            inclusive_upper=False,
        )

        if (
            self.type == "transformer"
            and d_model % self.num_heads != 0
        ):
            raise ValueError(
                "d_model must be divisible by the number of "
                "Transformer heads."
            )

        if self.type == "modern_tcn":
            for name, value in {
                "modern_tcn_patch_size": self.modern_tcn_patch_size,
                "modern_tcn_patch_stride": self.modern_tcn_patch_stride,
                "modern_tcn_ffn_ratio": self.modern_tcn_ffn_ratio,
                "modern_tcn_num_blocks": self.modern_tcn_num_blocks,
                "modern_tcn_large_kernel": self.modern_tcn_large_kernel,
                "modern_tcn_small_kernel": self.modern_tcn_small_kernel,
            }.items():
                _validate_positive_integer(value, name=f"temporal.{name}")

            if self.modern_tcn_patch_size < self.modern_tcn_patch_stride:
                raise ValueError(
                    "temporal.modern_tcn_patch_size must be greater than "
                    "or equal to temporal.modern_tcn_patch_stride."
                )

            _validate_probability(
                self.modern_tcn_dropout,
                name="temporal.modern_tcn_dropout",
                inclusive_upper=False,
            )

        if self.type == "tcn":
            _validate_positive_integer(
                self.kernel_size,
                name="temporal.kernel_size",
            )

            if self.kernel_size < 2:
                raise ValueError(
                    "temporal.kernel_size must be at least 2 "
                    "for a causal TCN."
                )

            if not self.dilations:
                raise ValueError(
                    "temporal.dilations must not be empty."
                )

            for dilation in self.dilations:
                _validate_positive_integer(
                    dilation,
                    name="temporal dilation",
                )

    @property
    def tcn_receptive_field(self) -> int:
        """Return the receptive field of one convolution per dilation."""
        if self.type != "tcn":
            return 1

        return 1 + (
            self.kernel_size - 1
        ) * sum(self.dilations)

    def output_length(self, context_length: int) -> int:
        """Return the temporal feature length exposed to graph/prediction.

        Transformer, identity and causal-TCN encoders preserve all observed
        context positions. The one-stage ModernTCN path repeats the final
        value for patch padding, yielding exactly ``context / stride``
        patch positions for the supported session contract.
        """
        if self.type != "modern_tcn":
            return int(context_length)

        if int(context_length) % self.modern_tcn_patch_stride != 0:
            raise ValueError(
                "context_length must be divisible by the ModernTCN patch "
                "stride."
            )

        return int(context_length) // self.modern_tcn_patch_stride


@dataclass(frozen=True)
class SpatialConfig:
    """Configuration for graph message passing and temporal/spatial fusion."""

    num_layers: int = 1
    feedforward_multiplier: int = 2
    dropout: float = 0.0
    gate_type: Literal["none", "fixed", "learned_scalar"] = "none"
    initial_beta: float = 1.0

    def validate(self, *, graph_type: GraphType) -> None:
        _validate_positive_integer(self.num_layers, name="spatial.num_layers")
        _validate_positive_integer(
            self.feedforward_multiplier,
            name="spatial.feedforward_multiplier",
        )
        _validate_probability(
            self.dropout,
            name="spatial.dropout",
            inclusive_upper=False,
        )
        if self.gate_type not in {"none", "fixed", "learned_scalar"}:
            raise ValueError(
                "spatial.gate_type must be 'none', 'fixed', or "
                "'learned_scalar'."
            )
        _validate_probability(
            self.initial_beta,
            name="spatial.initial_beta",
        )
        if graph_type == "none" and self.gate_type != "none":
            raise ValueError(
                "Graph-free token models must use spatial.gate_type='none'."
            )


@dataclass(frozen=True)
class CloseScaleFeatureConfig:
    """Optional causal Close log-variance feature.

    The single raw feature is calculated from the observed 60-minute context
    only:

        log(context Close variance + eps)
        = log(context Close std ** 2 + eps)

    The training runner standardises it using statistics fitted over the
    training windows/assets only.  The resulting scalar is projected and added
    once before the first temporal module.
    """

    enabled: bool = False
    eps: float = 1.0e-6

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("close_scale_features.enabled must be boolean.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError(
                "close_scale_features.eps must be finite and positive."
            )


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for one context-window graph learner.

    All learned or supplied adjacencies follow:

        A[target, source]

    The public graph tensor is always:

        [B, graph_heads, N, N]

    even when the internal graph is global/static.
    """

    type: GraphType = "mtgnn_static"
    num_heads: int = 2
    hidden_dim: int = 32
    activation: GraphActivation = "softmax"
    add_self_loops: bool = False

    # MTGNN static graph options.
    mtgnn_embedding_dim: int = 16
    mtgnn_top_k: int = 4
    mtgnn_alpha: float = 3.0

    # Dynamic-base options.
    base_graph_type: StaticGraphType = "mtgnn_static"
    gate_type: GateType = "learned_scalar"
    initial_alpha: float = 0.5

    def validate(
        self,
        *,
        num_nodes: int,
        d_model: int,
    ) -> None:
        valid_graph_types = {
            "none",
            "fixed",
            "free_static",
            "mtgnn_static",
            "dynamic",
            "dynamic_correlation",
            "dynamic_base",
            "oracle",
        }

        if self.type not in valid_graph_types:
            raise ValueError(
                f"Unsupported graph type {self.type!r}."
            )

        _validate_positive_integer(
            self.num_heads,
            name="graph.num_heads",
        )

        _validate_positive_integer(
            self.hidden_dim,
            name="graph.hidden_dim",
        )

        if self.activation not in {
            "softmax",
            "sparsemax",
            "entmax15",
            "gated",
        }:
            raise ValueError(
                f"Unsupported graph activation "
                f"{self.activation!r}."
            )

        if (
            self.type in {
                "dynamic",
                "dynamic_base",
            }
            and d_model <= 0
        ):
            raise ValueError(
                "Dynamic graph learning requires a positive "
                "d_model."
            )

        _validate_positive_integer(
            self.mtgnn_embedding_dim,
            name="graph.mtgnn_embedding_dim",
        )

        _validate_positive_integer(
            self.mtgnn_top_k,
            name="graph.mtgnn_top_k",
        )

        maximum_neighbours = (
            num_nodes
            if self.add_self_loops
            else num_nodes - 1
        )

        if self.mtgnn_top_k > maximum_neighbours:
            raise ValueError(
                "graph.mtgnn_top_k exceeds the number of "
                "eligible source nodes."
            )

        if self.mtgnn_alpha <= 0:
            raise ValueError(
                "graph.mtgnn_alpha must be positive."
            )

        if self.base_graph_type not in {
            "free_static",
            "mtgnn_static",
        }:
            raise ValueError(
                "graph.base_graph_type must be 'free_static' "
                "or 'mtgnn_static'."
            )

        if self.gate_type not in {
            "none",
            "fixed",
            "learned_scalar",
            "learned_per_head",
        }:
            raise ValueError(
                f"Unsupported graph gate type "
                f"{self.gate_type!r}."
            )

        _validate_probability(
            self.initial_alpha,
            name="graph.initial_alpha",
        )


@dataclass(frozen=True)
class ForecastHeadConfig:
    """Configuration for the complete 60-step future token path.

    The model predicts every future minute required by the frozen
    contextual Kronos decoder. Final financial evaluation still uses
    only ``evaluation_horizons``.
    """

    prediction_length: int = DEFAULT_PREDICTION_LENGTH
    evaluation_horizons: tuple[int, ...] = (
        DEFAULT_EVALUATION_HORIZONS
    )
    s1_vocabulary_size: int = 1024
    s2_vocabulary_size: int = 1024
    s2_loss_weight: float = 1.0

    # ``full`` predicts and decodes both Kronos subtokens.
    # ``coarse_only`` optimises only s1 and uses the frozen tokenizer
    # coarse reconstruction branch for raw-price validation. Historical
    # context still contains both observed s1 and s2 IDs.
    future_token_mode: FutureTokenMode = "full"

    # Primary configuration used by new runs. During supervised training,
    # ``true_s1`` conditions the fine head on the same-position target
    # coarse token. ``predicted_s1`` uses the model-selected coarse token,
    # matching free-running generation.
    s2_conditioning: S2Conditioning = "true_s1"

    # Backwards-compatible field for older saved configs and smoke tests.
    # When provided, it takes precedence over ``s2_conditioning``:
    # True -> true_s1; False -> predicted_s1. New YAML files should use
    # ``s2_conditioning`` and omit this legacy field.
    condition_s2_on_s1: bool | None = None

    def validate(self) -> None:
        _validate_positive_integer(
            self.prediction_length,
            name="heads.prediction_length",
        )

        if not self.evaluation_horizons:
            raise ValueError(
                "At least one evaluation horizon is required."
            )

        resolved = tuple(
            int(horizon)
            for horizon in self.evaluation_horizons
        )

        if any(
            horizon <= 0
            for horizon in resolved
        ):
            raise ValueError(
                "Every evaluation horizon must be positive."
            )

        if len(set(resolved)) != len(resolved):
            raise ValueError(
                "Evaluation horizons must be unique."
            )

        if tuple(sorted(resolved)) != resolved:
            raise ValueError(
                "Evaluation horizons must be strictly increasing."
            )

        if max(resolved) > self.prediction_length:
            raise ValueError(
                "Evaluation horizons cannot exceed "
                "prediction_length."
            )

        _validate_positive_integer(
            self.s1_vocabulary_size,
            name="heads.s1_vocabulary_size",
        )

        _validate_positive_integer(
            self.s2_vocabulary_size,
            name="heads.s2_vocabulary_size",
        )

        if self.s2_loss_weight < 0:
            raise ValueError(
                "heads.s2_loss_weight cannot be negative."
            )

        if self.future_token_mode not in {
            "full",
            "coarse_only",
        }:
            raise ValueError(
                "heads.future_token_mode must be 'full' or "
                "'coarse_only'."
            )

        if (
            self.future_token_mode == "coarse_only"
            and self.s2_loss_weight != 0.0
        ):
            raise ValueError(
                "heads.s2_loss_weight must be 0.0 when "
                "heads.future_token_mode='coarse_only'."
            )

        if self.s2_conditioning not in {
            "true_s1",
            "predicted_s1",
        }:
            raise ValueError(
                "heads.s2_conditioning must be 'true_s1' or "
                "'predicted_s1'."
            )

        if (
            self.condition_s2_on_s1 is not None
            and not isinstance(
                self.condition_s2_on_s1,
                bool,
            )
        ):
            raise TypeError(
                "heads.condition_s2_on_s1 must be boolean or None."
            )

    @property
    def predicts_s2(self) -> bool:
        """Whether the future predictor should evaluate the fine head."""
        return self.future_token_mode == "full"

    @property
    def uses_fine_token_path(self) -> bool:
        """Whether future s2 is optimised and used for raw decoding."""
        return self.predicts_s2

    @property
    def resolved_s2_conditioning(self) -> S2Conditioning:
        """Return the active fine-head conditioning policy.

        ``condition_s2_on_s1`` is retained only so historical configs and
        checkpoints can still be reconstructed. New experiments should
        record the explicit ``s2_conditioning`` string.
        """
        if self.condition_s2_on_s1 is not None:
            return (
                "true_s1"
                if self.condition_s2_on_s1
                else "predicted_s1"
            )

        return self.s2_conditioning

    @property
    def evaluation_indices(self) -> tuple[int, ...]:
        """Zero-based indices into the dense future path."""
        return tuple(
            horizon - 1
            for horizon in self.evaluation_horizons
        )


@dataclass(frozen=True)
class FuturePredictorConfig:
    """Configuration shared by both future-token predictor variants.

    ``structured_parallel`` uses learned ordered future queries with
    bidirectional future-query self-attention.

    For ``structured_parallel``, ``num_layers=0`` disables the future
    Transformer stack. The predictor then applies LayerNorm directly to
    the final context summary plus learned future-position embeddings.

    ``autoregressive`` uses shifted future token-pair embeddings with
    causal self-attention. Training is teacher-forced; inference is
    sequential.
    """

    type: FuturePredictorType = "structured_parallel"
    num_layers: int = 2
    num_heads: int = 4
    feedforward_multiplier: int = 2
    dropout: float = 0.0

    def validate(
        self,
        *,
        d_model: int,
    ) -> None:
        if self.type not in {
            "structured_parallel",
            "autoregressive",
        }:
            raise ValueError(
                f"Unsupported future predictor type {self.type!r}."
            )

        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers < 0
        ):
            raise ValueError(
                "future_predictor.num_layers must be a non-negative "
                f"integer. Received {self.num_layers!r}."
            )

        if (
            self.type == "autoregressive"
            and self.num_layers == 0
        ):
            raise ValueError(
                "future_predictor.num_layers must be positive when "
                "future_predictor.type='autoregressive'."
            )

        _validate_positive_integer(
            self.num_heads,
            name="future_predictor.num_heads",
        )

        _validate_positive_integer(
            self.feedforward_multiplier,
            name="future_predictor.feedforward_multiplier",
        )

        _validate_probability(
            self.dropout,
            name="future_predictor.dropout",
            inclusive_upper=False,
        )

        if d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by the number of "
                "future-predictor attention heads."
            )


@dataclass(frozen=True)
class TokenLossConfig:
    """Weighting of the supervised future token positions.

    ``exponential_decay`` places the largest weight on minute 1, then
    halves the decaying component every ``exponential_half_life``
    positions. ``exponential_floor_weight`` retains a non-zero uniform
    contribution so the complete future token path remains supervised.
    The final weights always have mean one.
    """

    horizon_weighting: HorizonWeighting = "uniform"
    exponential_half_life: float = 5.0
    exponential_floor_weight: float = 0.25

    def validate(self) -> None:
        if self.horizon_weighting not in {
            "uniform",
            "exponential_decay",
        }:
            raise ValueError(
                "loss.horizon_weighting must be 'uniform' or "
                "'exponential_decay'."
            )

        if self.exponential_half_life <= 0:
            raise ValueError(
                "loss.exponential_half_life must be positive."
            )

        _validate_probability(
            self.exponential_floor_weight,
            name="loss.exponential_floor_weight",
            inclusive_upper=False,
        )


@dataclass(frozen=True)
class BackcastConfig:
    """Optional real-data-only continuous reconstruction objective."""

    enabled: bool = False
    loss_weight: float = 0.0
    num_channels: int = 5

    def validate(self) -> None:
        if self.loss_weight < 0:
            raise ValueError(
                "backcast.loss_weight cannot be negative."
            )

        _validate_positive_integer(
            self.num_channels,
            name="backcast.num_channels",
        )

        if not self.enabled and self.loss_weight != 0.0:
            raise ValueError(
                "Disabled backcasting must have loss_weight=0."
            )


@dataclass(frozen=True)
class DynamicGraphModelConfig:
    """Canonical configuration for synthetic and real token models."""

    num_nodes: int
    context_length: int = 60
    d_model: int = 64
    num_st_blocks: int = 1
    use_node_embedding: bool = True
    token_input_representation: TokenInputRepresentation = (
        "hierarchical_embedding"
    )

    temporal: TemporalConfig = field(
        default_factory=TemporalConfig
    )
    graph: GraphConfig = field(
        default_factory=GraphConfig
    )
    spatial: SpatialConfig = field(
        default_factory=SpatialConfig
    )
    close_scale_features: CloseScaleFeatureConfig = field(
        default_factory=CloseScaleFeatureConfig
    )
    heads: ForecastHeadConfig = field(
        default_factory=ForecastHeadConfig
    )
    future_predictor: FuturePredictorConfig = field(
        default_factory=FuturePredictorConfig
    )
    loss: TokenLossConfig = field(
        default_factory=TokenLossConfig
    )
    backcast: BackcastConfig = field(
        default_factory=BackcastConfig
    )

    def validate(self) -> None:
        _validate_positive_integer(
            self.num_nodes,
            name="num_nodes",
        )

        _validate_positive_integer(
            self.context_length,
            name="context_length",
        )

        _validate_positive_integer(
            self.d_model,
            name="d_model",
        )

        _validate_positive_integer(
            self.num_st_blocks,
            name="num_st_blocks",
        )

        if self.token_input_representation not in {
            "hierarchical_embedding",
            "bsq_bits",
        }:
            raise ValueError(
                "token_input_representation must be "
                "'hierarchical_embedding' or 'bsq_bits'."
            )

        self.temporal.validate(
            d_model=self.d_model,
        )

        if self.temporal.type == "modern_tcn":
            if self.num_st_blocks != 1:
                raise ValueError(
                    "The token ModernTCN architecture supports exactly one "
                    "temporal/graph/spatial block."
                )
            if (
                self.token_input_representation == "bsq_bits"
                and self.use_node_embedding
            ):
                raise ValueError(
                    "The exact post-BSQ ModernTCN control must not add a "
                    "node embedding before the per-asset backbone."
                )

        self.graph.validate(
            num_nodes=self.num_nodes,
            d_model=self.d_model,
        )

        if self.graph.type == "dynamic_correlation":
            raise ValueError(
                "graph.type='dynamic_correlation' is currently supported "
                "only by the continuous forecaster, where the graph is "
                "computed from observed raw Close values in each window."
            )

        self.spatial.validate(
            graph_type=self.graph.type,
        )

        self.close_scale_features.validate()

        self.heads.validate()

        self.future_predictor.validate(
            d_model=self.d_model,
        )

        self.loss.validate()
        self.backcast.validate()

        if self.temporal.type == "modern_tcn":
            if self.future_predictor.type != "structured_parallel":
                raise ValueError(
                    "The final token ModernTCN experiment requires the "
                    "structured-parallel 60-position predictor."
                )
            if self.heads.future_token_mode != "coarse_only":
                raise ValueError(
                    "The final token ModernTCN experiment predicts only the "
                    "coarse s1 subtoken."
                )
            if self.heads.s1_vocabulary_size != 1024:
                raise ValueError(
                    "The final token ModernTCN experiment uses the original "
                    "1024-way coarse vocabulary."
                )
            if self.backcast.enabled:
                raise ValueError(
                    "Backcasting is not implemented for patch-level token "
                    "ModernTCN features."
                )

        # Evaluate once here so invalid patch/stride contracts fail during
        # configuration validation rather than at the first GPU forward.
        _ = self.temporal_output_length

        if (
            self.context_length
            + self.heads.prediction_length
            > 512
        ):
            raise ValueError(
                "Context plus future path exceeds the current "
                "Kronos sequence limit of 512 positions."
            )

    @property
    def temporal_output_length(self) -> int:
        return self.temporal.output_length(self.context_length)

    @property
    def prediction_length(self) -> int:
        return int(self.heads.prediction_length)

    @property
    def num_evaluation_horizons(self) -> int:
        return len(self.heads.evaluation_horizons)

    @property
    def evaluation_indices(self) -> tuple[int, ...]:
        return self.heads.evaluation_indices

    @property
    def num_horizons(self) -> int:
        """Compatibility alias for dense future positions.

        New code should use ``prediction_length``.
        """
        return self.prediction_length


@dataclass
class GraphOutput:
    """Graphs and graph components exposed by the model.

    Shapes:
        selected:
            [B, G, N, N] or None.

        per_layer:
            Tuple containing one graph per interlaced block. Each
            non-null graph has shape [B, G, N, N].

        base:
            [B, G, N, N], [1, G, N, N], or None.

        dynamic:
            [B, G, N, N] or None.

        alpha:
            Scalar, [G], [B, G], or None.

        logits:
            Unnormalised selected graph logits with shape
            [B, G, N, N] or None.
    """

    selected: Tensor | None
    per_layer: tuple[Tensor | None, ...] = ()
    base: Tensor | None = None
    dynamic: Tensor | None = None
    alpha: Tensor | None = None
    logits: Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        num_heads: int,
        num_nodes: int,
        require_row_stochastic: bool = True,
        atol: float = 1.0e-5,
    ) -> None:
        expected = (
            batch_size,
            num_heads,
            num_nodes,
            num_nodes,
        )

        def validate_graph(
            graph: Tensor | None,
            *,
            name: str,
            allow_singleton_batch: bool = False,
        ) -> None:
            if graph is None:
                return

            valid_shapes = {expected}

            if allow_singleton_batch:
                valid_shapes.add(
                    (
                        1,
                        num_heads,
                        num_nodes,
                        num_nodes,
                    )
                )

            if tuple(graph.shape) not in valid_shapes:
                raise ValueError(
                    f"{name} has shape {tuple(graph.shape)}; "
                    f"expected one of {sorted(valid_shapes)}."
                )

            if not torch.isfinite(graph).all():
                raise ValueError(
                    f"{name} contains non-finite values."
                )

            if torch.any(graph < 0):
                raise ValueError(
                    f"{name} contains negative graph weights."
                )

            if require_row_stochastic:
                # Graph probabilities should normally remain float32 under
                # AMP, but accumulate the contract check in float32 as a
                # final safeguard against low-precision reductions.
                row_sums = graph.float().sum(dim=-1)
                row_deviation = (row_sums - 1.0).abs()
                maximum_deviation = float(
                    row_deviation.max().item()
                )

                if maximum_deviation > float(atol):
                    raise ValueError(
                        f"{name} is not row-stochastic: "
                        f"maximum row-sum deviation="
                        f"{maximum_deviation:.6g}, "
                        f"tolerance={float(atol):.6g}, "
                        f"dtype={graph.dtype}."
                    )

        validate_graph(
            self.selected,
            name="graph.selected",
        )

        validate_graph(
            self.base,
            name="graph.base",
            allow_singleton_batch=True,
        )

        validate_graph(
            self.dynamic,
            name="graph.dynamic",
        )

        for layer_idx, graph in enumerate(
            self.per_layer
        ):
            validate_graph(
                graph,
                name=f"graph.per_layer[{layer_idx}]",
            )

        if self.logits is not None:
            if tuple(self.logits.shape) != expected:
                raise ValueError(
                    "graph.logits has an unexpected shape."
                )

            if not torch.isfinite(
                self.logits
            ).all():
                raise ValueError(
                    "graph.logits contains non-finite values."
                )


@dataclass
class TokenForecastOutput:
    """Typed output of the direct sparse-horizon forecaster."""

    s1_logits: Tensor
    s2_logits: Tensor | None
    graph: GraphOutput
    context_hidden: Tensor
    temporal_hidden: Tensor
    future_hidden: Tensor
    spatial_beta: Tensor | None = None
    backcast: Tensor | None = None

    def validate(
        self,
        config: DynamicGraphModelConfig,
        *,
        batch_size: int,
    ) -> None:
        config.validate()

        expected_s1 = (
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.heads.s1_vocabulary_size,
        )

        expected_s2 = (
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.heads.s2_vocabulary_size,
        )

        if tuple(self.s1_logits.shape) != expected_s1:
            raise ValueError(
                "s1_logits has shape "
                f"{tuple(self.s1_logits.shape)}; "
                f"expected {expected_s1}."
            )

        if config.heads.predicts_s2:
            if self.s2_logits is None:
                raise ValueError(
                    "Full-token mode requires s2_logits."
                )

            if tuple(self.s2_logits.shape) != expected_s2:
                raise ValueError(
                    "s2_logits has shape "
                    f"{tuple(self.s2_logits.shape)}; "
                    f"expected {expected_s2}."
                )
        elif self.s2_logits is not None:
            raise ValueError(
                "Coarse-only mode must not return s2_logits."
            )

        expected_context_hidden = (
            batch_size,
            config.num_nodes,
            config.d_model,
        )

        if tuple(self.context_hidden.shape) != (
            expected_context_hidden
        ):
            raise ValueError(
                "context_hidden has shape "
                f"{tuple(self.context_hidden.shape)}; "
                f"expected {expected_context_hidden}."
            )

        expected_temporal_hidden = (
            batch_size,
            config.temporal_output_length,
            config.num_nodes,
            config.d_model,
        )

        if tuple(self.temporal_hidden.shape) != (
            expected_temporal_hidden
        ):
            raise ValueError(
                "temporal_hidden has shape "
                f"{tuple(self.temporal_hidden.shape)}; "
                f"expected {expected_temporal_hidden}."
            )

        expected_future_hidden = (
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.d_model,
        )

        if tuple(self.future_hidden.shape) != (
            expected_future_hidden
        ):
            raise ValueError(
                "future_hidden has shape "
                f"{tuple(self.future_hidden.shape)}; "
                f"expected {expected_future_hidden}."
            )

        if not torch.isfinite(
            self.future_hidden
        ).all():
            raise ValueError(
                "future_hidden contains non-finite values."
            )

        if config.backcast.enabled:
            expected_backcast = (
                batch_size,
                config.context_length,
                config.num_nodes,
                config.backcast.num_channels,
            )

            if self.backcast is None:
                raise ValueError(
                    "Backcasting is enabled but no backcast "
                    "tensor was returned."
                )

            if tuple(self.backcast.shape) != expected_backcast:
                raise ValueError(
                    "backcast has shape "
                    f"{tuple(self.backcast.shape)}; "
                    f"expected {expected_backcast}."
                )
        elif self.backcast is not None:
            raise ValueError(
                "A backcast tensor was returned while "
                "backcasting is disabled."
            )

        if not torch.isfinite(
            self.s1_logits
        ).all():
            raise ValueError(
                "s1_logits contains non-finite values."
            )

        if (
            self.s2_logits is not None
            and not torch.isfinite(
                self.s2_logits
            ).all()
        ):
            raise ValueError(
                "s2_logits contains non-finite values."
            )

        if self.spatial_beta is not None:
            beta = torch.as_tensor(self.spatial_beta).float()
            if beta.numel() != 1 or not torch.isfinite(beta).all():
                raise ValueError(
                    "spatial_beta must be one finite scalar."
                )
            if float(beta.item()) < 0.0 or float(beta.item()) > 1.0:
                raise ValueError(
                    "spatial_beta must lie in [0, 1]."
                )

        self.graph.validate(
            batch_size=batch_size,
            num_heads=config.graph.num_heads,
            num_nodes=config.num_nodes,
        )


def validate_token_batch(
    batch: Mapping[str, Tensor],
    config: DynamicGraphModelConfig,
    *,
    require_true_graph: bool = False,
) -> None:
    """Validate the common synthetic/real model-facing batch contract.

    Required shapes:
        tokens:
            [B, T, N, 2]

        target_s1:
            [B, prediction_length, N]

        target_s2:
            [B, prediction_length, N]

    Synthetic graph-recovery batches additionally contain:
        true_graph:
            [B, N, N]

        regime_id:
            [B]
    """
    config.validate()

    required = {
        "tokens",
        "target_s1",
        "target_s2",
    }

    missing = required - set(batch)

    if missing:
        raise KeyError(
            f"Batch is missing required keys: "
            f"{sorted(missing)}."
        )

    tokens = torch.as_tensor(
        batch["tokens"]
    )

    target_s1 = torch.as_tensor(
        batch["target_s1"]
    )

    target_s2 = torch.as_tensor(
        batch["target_s2"]
    )

    if tokens.ndim != 4:
        raise ValueError(
            "tokens must have shape [B, T, N, 2]."
        )

    batch_size = int(tokens.shape[0])

    expected_tokens = (
        batch_size,
        config.context_length,
        config.num_nodes,
        2,
    )

    if tuple(tokens.shape) != expected_tokens:
        raise ValueError(
            f"tokens has shape {tuple(tokens.shape)}; "
            f"expected {expected_tokens}."
        )

    expected_targets = (
        batch_size,
        config.prediction_length,
        config.num_nodes,
    )

    if tuple(target_s1.shape) != expected_targets:
        raise ValueError(
            "target_s1 has shape "
            f"{tuple(target_s1.shape)}; "
            f"expected {expected_targets}."
        )

    if tuple(target_s2.shape) != expected_targets:
        raise ValueError(
            "target_s2 has shape "
            f"{tuple(target_s2.shape)}; "
            f"expected {expected_targets}."
        )

    if (
        tokens[..., 0].min().item() < 0
        or tokens[..., 0].max().item()
        >= config.heads.s1_vocabulary_size
    ):
        raise ValueError(
            "Input s1 token IDs lie outside the configured "
            "vocabulary."
        )

    if (
        tokens[..., 1].min().item() < 0
        or tokens[..., 1].max().item()
        >= config.heads.s2_vocabulary_size
    ):
        raise ValueError(
            "Input s2 token IDs lie outside the configured "
            "vocabulary."
        )

    if (
        target_s1.min().item() < 0
        or target_s1.max().item()
        >= config.heads.s1_vocabulary_size
    ):
        raise ValueError(
            "target_s1 IDs lie outside the configured "
            "vocabulary."
        )

    if (
        target_s2.min().item() < 0
        or target_s2.max().item()
        >= config.heads.s2_vocabulary_size
    ):
        raise ValueError(
            "target_s2 IDs lie outside the configured "
            "vocabulary."
        )

    if require_true_graph:
        if "true_graph" not in batch:
            raise KeyError(
                "Synthetic graph-recovery batch is missing "
                "'true_graph'."
            )

        true_graph = torch.as_tensor(
            batch["true_graph"]
        )

        expected_graph = (
            batch_size,
            config.num_nodes,
            config.num_nodes,
        )

        if tuple(true_graph.shape) != expected_graph:
            raise ValueError(
                "true_graph has shape "
                f"{tuple(true_graph.shape)}; "
                f"expected {expected_graph}."
            )

        if torch.any(true_graph < 0):
            raise ValueError(
                "true_graph contains negative weights."
            )

        if not torch.allclose(
            true_graph.sum(dim=-1),
            torch.ones_like(
                true_graph.sum(dim=-1)
            ),
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "true_graph is not row-stochastic."
            )

        if "regime_id" not in batch:
            raise KeyError(
                "Synthetic graph-recovery batch is missing "
                "'regime_id'."
            )

        regime_id = torch.as_tensor(
            batch["regime_id"]
        )

        if tuple(regime_id.shape) != (
            batch_size,
        ):
            raise ValueError(
                "regime_id must have shape [B]."
            )


def _cpu_smoke_test() -> None:
    config = DynamicGraphModelConfig(
        num_nodes=16,
        context_length=60,
        d_model=32,
        num_st_blocks=2,
        temporal=TemporalConfig(
            type="transformer",
            num_layers=1,
            num_heads=4,
        ),
        graph=GraphConfig(
            type="dynamic_base",
            num_heads=2,
            hidden_dim=16,
            base_graph_type="mtgnn_static",
            mtgnn_top_k=4,
            gate_type="learned_scalar",
            initial_alpha=0.5,
        ),
        heads=ForecastHeadConfig(
            prediction_length=60,
            evaluation_horizons=(
                1,
                5,
                15,
                30,
                60,
            ),
        ),
        future_predictor=FuturePredictorConfig(
            type="structured_parallel",
            num_layers=2,
            num_heads=4,
        ),
        loss=TokenLossConfig(
            horizon_weighting="uniform",
        ),
        backcast=BackcastConfig(
            enabled=False,
            loss_weight=0.0,
        ),
    )

    config.validate()

    coarse_heads = ForecastHeadConfig(
        prediction_length=60,
        evaluation_horizons=(1, 5, 15, 30, 60),
        s2_loss_weight=0.0,
        future_token_mode="coarse_only",
    )
    coarse_heads.validate()

    if coarse_heads.predicts_s2:
        raise AssertionError(
            "Coarse-only head configuration still predicts s2."
        )

    try:
        ForecastHeadConfig(
            s2_loss_weight=1.0,
            future_token_mode="coarse_only",
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Coarse-only configuration accepted a non-zero s2 weight."
        )

    batch_size = 3

    batch = {
        "tokens": torch.randint(
            0,
            1024,
            (
                batch_size,
                config.context_length,
                config.num_nodes,
                2,
            ),
        ),
        "target_s1": torch.randint(
            0,
            1024,
            (
                batch_size,
                config.prediction_length,
                config.num_nodes,
            ),
        ),
        "target_s2": torch.randint(
            0,
            1024,
            (
                batch_size,
                config.prediction_length,
                config.num_nodes,
            ),
        ),
        "true_graph": torch.softmax(
            torch.randn(
                batch_size,
                config.num_nodes,
                config.num_nodes,
            ),
            dim=-1,
        ),
        "regime_id": torch.randint(
            0,
            3,
            (batch_size,),
        ),
    }

    validate_token_batch(
        batch,
        config,
        require_true_graph=True,
    )

    selected_graph = torch.softmax(
        torch.randn(
            batch_size,
            config.graph.num_heads,
            config.num_nodes,
            config.num_nodes,
        ),
        dim=-1,
    )

    output = TokenForecastOutput(
        s1_logits=torch.randn(
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.heads.s1_vocabulary_size,
        ),
        s2_logits=torch.randn(
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.heads.s2_vocabulary_size,
        ),
        graph=GraphOutput(
            selected=selected_graph,
            per_layer=(
                selected_graph,
                selected_graph.clone(),
            ),
            base=selected_graph[:1],
            dynamic=selected_graph,
            alpha=torch.tensor(0.5),
            logits=torch.randn_like(
                selected_graph
            ),
        ),
        context_hidden=torch.randn(
            batch_size,
            config.num_nodes,
            config.d_model,
        ),
        temporal_hidden=torch.randn(
            batch_size,
            config.context_length,
            config.num_nodes,
            config.d_model,
        ),
        future_hidden=torch.randn(
            batch_size,
            config.prediction_length,
            config.num_nodes,
            config.d_model,
        ),
        backcast=None,
    )

    output.validate(
        config,
        batch_size=batch_size,
    )

    print(
        "Dynamic-graph architecture contract CPU smoke "
        "test passed."
    )
    print(
        "Graph orientation:",
        GRAPH_ORIENTATION,
    )
    print(
        "Input tokens:",
        tuple(batch["tokens"].shape),
    )
    print(
        "Target s1/s2:",
        tuple(batch["target_s1"].shape),
    )
    print(
        "Evaluation indices:",
        config.evaluation_indices,
    )
    print(
        "Output s1 logits:",
        tuple(output.s1_logits.shape),
    )
    print(
        "Output s2 logits:",
        tuple(output.s2_logits.shape),
    )
    print(
        "Selected graph:",
        tuple(output.graph.selected.shape),
    )
    print(
        "TCN reference receptive field:",
        TemporalConfig(
            type="tcn"
        ).tcn_receptive_field,
    )


if __name__ == "__main__":
    _cpu_smoke_test()






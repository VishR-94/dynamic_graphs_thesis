from __future__ import annotations

"""Pinned BaseDyGraph-v1 coarse-token controls.

Both task variants use the official ``DiscreteSTGraphBackbone`` unchanged:

    learned coarse-state embedding + learned node embedding
    -> four interlaced causal Transformer / dynamic-graph / spatial blocks
    -> final LayerNorm representation

Only the prediction head differs:

``dense_one_step``
    The official ``NextStateHead`` predicts each next token in the
    teacher-forced sequence ``context + true_future_h1``.  For a 60-token
    context this yields 60 next-token targets.  Causality guarantees that the
    final context-to-horizon-1 logit cannot depend on the appended target.

``parallel_60``
    The official backbone receives only the observed context.  The project's
    established structured-parallel Transformer head predicts all 60 future
    coarse-token distributions jointly.

The graph tensors exposed here are the actual adjacencies used for spatial
message passing, with orientation ``A[target, source]``.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn

from src.models.basedygraph_official_adapter import (
    OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
    PINNED_BASEDYGRAPH_COMMIT,
    OfficialBaseDyGraphRunConfig,
    build_official_model_config,
    load_official_basedygraph_modules,
)
from src.models.modern_tcn_graph_round2_token import (
    CoarseStructuredParallelPredictor,
)


BaseDyGraphV1PredictionMode = Literal["dense_one_step", "parallel_60"]


@dataclass(frozen=True)
class BaseDyGraphV1TokenConfig:
    num_nodes: int
    context_length: int = 60
    prediction_length: int = 60
    evaluation_horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    vocabulary_size: int = 1024
    prediction_mode: BaseDyGraphV1PredictionMode = "dense_one_step"

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
            ("d_model", self.d_model),
            ("temporal_num_heads", self.temporal_num_heads),
            ("temporal_num_layers", self.temporal_num_layers),
            ("spatial_num_layers", self.spatial_num_layers),
            ("feedforward_multiplier", self.feedforward_multiplier),
            ("graph_num_heads", self.graph_num_heads),
            ("graph_hidden_dim", self.graph_hidden_dim),
            ("num_st_blocks", self.num_st_blocks),
            ("future_predictor_num_heads", self.future_predictor_num_heads),
            (
                "future_predictor_feedforward_multiplier",
                self.future_predictor_feedforward_multiplier,
            ),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.prediction_mode not in {"dense_one_step", "parallel_60"}:
            raise ValueError(
                f"Unsupported prediction_mode={self.prediction_mode!r}."
            )
        if self.d_model % self.temporal_num_heads:
            raise ValueError("d_model must be divisible by temporal_num_heads.")
        if self.graph_hidden_dim % self.graph_num_heads:
            raise ValueError(
                "graph_hidden_dim must be divisible by graph_num_heads."
            )
        if self.d_model % self.graph_num_heads:
            raise ValueError("d_model must be divisible by graph_num_heads.")
        horizons = tuple(int(value) for value in self.evaluation_horizons)
        if not horizons or horizons != tuple(sorted(set(horizons))):
            raise ValueError(
                "evaluation_horizons must be non-empty, unique, and increasing."
            )
        if horizons[0] <= 0 or horizons[-1] > self.prediction_length:
            raise ValueError("evaluation_horizons lie outside prediction_length.")
        if self.prediction_mode != "dense_one_step":
            if self.future_predictor_num_layers < 0:
                raise ValueError("future_predictor_num_layers cannot be negative.")
            if self.d_model % self.future_predictor_num_heads:
                raise ValueError(
                    "d_model must be divisible by future_predictor_num_heads."
                )
        for name, value in (
            ("dropout", self.dropout),
            ("spatial_dropout", self.spatial_dropout),
            ("future_predictor_dropout", self.future_predictor_dropout),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1).")

    @property
    def output_length(self) -> int:
        return (
            int(self.context_length)
            if self.prediction_mode == "dense_one_step"
            else int(self.prediction_length)
        )

    @property
    def public_horizons(self) -> tuple[int, ...]:
        return (
            (1,)
            if self.prediction_mode == "dense_one_step"
            else tuple(int(value) for value in self.evaluation_horizons)
        )

    @property
    def evaluation_indices(self) -> tuple[int, ...]:
        return tuple(int(value) - 1 for value in self.public_horizons)

    def official_run_config(self) -> OfficialBaseDyGraphRunConfig:
        """Exact v1 backbone settings shared by both comparison tasks."""

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
class BaseDyGraphV1TokenOutput:
    s1_logits: Tensor  # [B,60,N,K]
    selected_s1: Tensor  # [B,60,N]
    teacher_forced_targets: Tensor | None  # dense only [B,60,N]
    selected_graph: Tensor  # [B,G,N,N] at the forecast origin
    per_layer_graphs: tuple[Tensor, ...]  # each [B,G,N,N]
    graph_sequences: tuple[Tensor, ...]  # each [B,T,G,N,N]
    temporal_repr: Tensor  # [B,T,N,D]
    spatial_repr: Tensor  # [B,T,N,D]
    future_hidden: Tensor | None


class BaseDyGraphV1TokenModel(nn.Module):
    """Pinned official BaseDyGraph-v1 backbone with one controlled head."""

    graph_orientation = OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION

    def __init__(
        self,
        config: BaseDyGraphV1TokenConfig,
        *,
        external_source_dir: str | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.official_modules = load_official_basedygraph_modules(
            external_source_dir,
            require_pinned_commit=True,
        )
        self.official_config = build_official_model_config(
            config.official_run_config(),
            external_source_dir=external_source_dir,
        )
        self.backbone = self.official_modules.model.DiscreteSTGraphBackbone(
            self.official_config
        )
        if config.prediction_mode == "dense_one_step":
            self.next_state_head: nn.Module | None = (
                self.official_modules.model.NextStateHead(
                    config.d_model,
                    config.vocabulary_size,
                )
            )
            self.future_predictor: nn.Module | None = None
        else:
            self.next_state_head = None
            predictor_config = SimpleNamespace(
                transformer_d_model=int(config.d_model),
                prediction_length=int(config.prediction_length),
                num_nodes=int(config.num_nodes),
                vocabulary_size=int(config.vocabulary_size),
                future_predictor_num_layers=int(
                    config.future_predictor_num_layers
                ),
                future_predictor_num_heads=int(
                    config.future_predictor_num_heads
                ),
                future_predictor_feedforward_multiplier=int(
                    config.future_predictor_feedforward_multiplier
                ),
                future_predictor_dropout=float(
                    config.future_predictor_dropout
                ),
            )
            self.future_predictor = CoarseStructuredParallelPredictor(
                predictor_config
            )

    @property
    def external_commit(self) -> str | None:
        return self.official_modules.commit

    def graph_parameter_ids(self) -> set[int]:
        """IDs of trainable official Q/K graph-scorer parameters."""

        result: set[int] = set()
        blocks = getattr(self.backbone, "st_blocks", None)
        if blocks is None:
            scorers = (getattr(self.backbone, "graph_scorer", None),)
        else:
            scorers = tuple(
                getattr(block, "graph_scorer", None) for block in blocks
            )
        for scorer in scorers:
            if scorer is None:
                continue
            result.update(
                id(parameter)
                for parameter in scorer.parameters()
                if parameter.requires_grad
            )
        return result

    def _validate_context(self, context_s1: Tensor) -> Tensor:
        values = torch.as_tensor(context_s1)
        expected = (
            int(values.shape[0]),
            int(self.config.context_length),
            int(self.config.num_nodes),
        ) if values.ndim == 3 else None
        if values.ndim != 3 or tuple(values.shape) != expected:
            raise ValueError(
                "context_s1 must have shape [B,context_length,N]; "
                f"observed {tuple(values.shape)}."
            )
        values = values.long()
        if values.numel() and (
            int(values.min().item()) < 0
            or int(values.max().item()) >= self.config.vocabulary_size
        ):
            raise ValueError("context_s1 contains an out-of-vocabulary ID.")
        return values

    def _validate_first_future(self, values: Tensor, batch: int) -> Tensor:
        result = torch.as_tensor(values)
        expected = (batch, self.config.num_nodes)
        if result.ndim != 2 or tuple(result.shape) != expected:
            raise ValueError(
                f"first_future_s1 must have shape {expected}; "
                f"observed {tuple(result.shape)}."
            )
        result = result.long()
        if result.numel() and (
            int(result.min().item()) < 0
            or int(result.max().item()) >= self.config.vocabulary_size
        ):
            raise ValueError("first_future_s1 contains an out-of-vocabulary ID.")
        return result

    def _extract_graphs(
        self,
        official_output: Mapping[str, Any],
        *,
        context_index: int,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        raw_layers = official_output.get("block_graph_attns")
        if raw_layers is None:
            raw_selected = official_output.get("graph_attn")
            if raw_selected is None:
                raise RuntimeError("The dynamic BaseDyGraph returned no graph.")
            raw_layers = [raw_selected]
        graph_sequences = tuple(
            torch.as_tensor(values).contiguous()
            for values in raw_layers
            if values is not None
        )
        if len(graph_sequences) != self.config.num_st_blocks:
            raise RuntimeError(
                "Unexpected BaseDyGraph graph-layer count: "
                f"{len(graph_sequences)} vs {self.config.num_st_blocks}."
            )
        for values in graph_sequences:
            if values.ndim != 5:
                raise RuntimeError(
                    "Expected graph sequence [B,T,G,N,N], got "
                    f"{tuple(values.shape)}."
                )
            if context_index >= int(values.shape[1]):
                raise RuntimeError("Forecast-origin graph index is unavailable.")
        per_layer = tuple(
            values[:, context_index].contiguous() for values in graph_sequences
        )
        return per_layer[-1], per_layer, graph_sequences

    def forward(
        self,
        context_s1: Tensor,
        *,
        first_future_s1: Tensor | None = None,
    ) -> BaseDyGraphV1TokenOutput:
        context = self._validate_context(context_s1)
        batch = int(context.shape[0])

        if self.config.prediction_mode == "dense_one_step":
            if first_future_s1 is None:
                raise ValueError(
                    "dense_one_step requires the true first future token."
                )
            first_future = self._validate_first_future(
                first_future_s1,
                batch,
            )
            sequence = torch.cat(
                (context, first_future.unsqueeze(1)),
                dim=1,
            )  # [B,61,N]
            state_ids = sequence.permute(0, 2, 1).contiguous()
            official_output = self.backbone(state_ids)
            if self.next_state_head is None:
                raise RuntimeError("The official next-state head is unavailable.")
            official_logits = self.next_state_head(
                torch.as_tensor(official_output["spatial_repr"])
            )  # [B,N,60,K]
            logits = official_logits.permute(0, 2, 1, 3).contiguous()
            teacher_targets = sequence[:, 1:].contiguous()
            future_hidden = None
        else:
            if first_future_s1 is not None:
                raise ValueError("parallel_60 must not receive future tokens.")
            state_ids = context.permute(0, 2, 1).contiguous()
            official_output = self.backbone(state_ids)
            if self.future_predictor is None:
                raise RuntimeError(
                    "The structured-parallel predictor is unavailable."
                )
            logits, future_hidden = self.future_predictor(
                torch.as_tensor(official_output["spatial_repr"])
            )
            teacher_targets = None

        expected_logits = (
            batch,
            self.config.output_length,
            self.config.num_nodes,
            self.config.vocabulary_size,
        )
        if tuple(logits.shape) != expected_logits:
            raise RuntimeError(
                f"Unexpected logit shape {tuple(logits.shape)}; "
                f"expected {expected_logits}."
            )
        if not torch.isfinite(logits).all():
            raise ValueError("BaseDyGraph token logits contain non-finite values.")

        selected, per_layer, graph_sequences = self._extract_graphs(
            official_output,
            context_index=self.config.context_length - 1,
        )
        return BaseDyGraphV1TokenOutput(
            s1_logits=logits,
            selected_s1=logits.argmax(dim=-1),
            teacher_forced_targets=teacher_targets,
            selected_graph=selected,
            per_layer_graphs=per_layer,
            graph_sequences=graph_sequences,
            temporal_repr=torch.as_tensor(
                official_output["temporal_repr"]
            ).contiguous(),
            spatial_repr=torch.as_tensor(
                official_output["spatial_repr"]
            ).contiguous(),
            future_hidden=future_hidden,
        )


def basedygraph_v1_token_config_from_mapping(
    values: Mapping[str, Any],
    *,
    num_nodes: int,
    vocabulary_size: int,
) -> BaseDyGraphV1TokenConfig:
    data = values["data"]
    model = values["model"]
    architecture = model["official_basedygraph_v1"]
    predictor = model["future_predictor"]
    config = BaseDyGraphV1TokenConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        prediction_length=int(data["prediction_length"]),
        evaluation_horizons=tuple(
            int(value) for value in data["evaluation_horizons"]
        ),
        vocabulary_size=int(vocabulary_size),
        prediction_mode=str(model["prediction_mode"]),
        d_model=int(architecture["d_model"]),
        temporal_num_heads=int(architecture["temporal_num_heads"]),
        temporal_num_layers=int(architecture["temporal_num_layers"]),
        spatial_num_layers=int(architecture["spatial_num_layers"]),
        feedforward_multiplier=int(architecture["feedforward_multiplier"]),
        graph_num_heads=int(architecture["graph_num_heads"]),
        graph_hidden_dim=int(architecture["graph_hidden_dim"]),
        num_st_blocks=int(architecture["num_st_blocks"]),
        dropout=float(architecture["dropout"]),
        spatial_dropout=float(architecture["spatial_dropout"]),
        future_predictor_num_layers=int(predictor["num_layers"]),
        future_predictor_num_heads=int(predictor["num_heads"]),
        future_predictor_feedforward_multiplier=int(
            predictor["feedforward_multiplier"]
        ),
        future_predictor_dropout=float(predictor["dropout"]),
    )
    config.validate()
    return config


__all__ = [
    "BaseDyGraphV1PredictionMode",
    "BaseDyGraphV1TokenConfig",
    "BaseDyGraphV1TokenModel",
    "BaseDyGraphV1TokenOutput",
    "OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION",
    "PINNED_BASEDYGRAPH_COMMIT",
    "basedygraph_v1_token_config_from_mapping",
]

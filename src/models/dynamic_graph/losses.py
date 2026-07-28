from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .contracts import GraphOutput
from .future_predictor import FutureTokenLoss


GRAPH_ENTROPY_EPS = 1.0e-8


@dataclass(frozen=True)
class GraphRegularisationConfig:
    """BaseDyGraph-style graph-shape regularisation.

    The field names deliberately mirror the pinned BaseDyGraph
    implementation so experiment configs remain directly comparable.

    ``graph_entropy_reg`` minimises mean row entropy directly.

    ``graph_target_entropy_reg`` matches mean row entropy to
    ``graph_target_entropy``. When the target is ``None``, synthetic
    training may provide ``true_graph`` and use its mean row entropy.
    If neither is available, the current entropy is detached as the
    target, matching BaseDyGraph's no-op fallback.

    ``graph_temporal_smooth_reg`` penalises squared graph changes. The
    original BaseDyGraph model applies this between consecutive graph
    frames. This project learns one graph per forecast window, so the
    same penalty is applied only to explicitly supplied adjacent-window
    pairs. Unrelated shuffled batch rows are never compared.
    """

    graph_reg_layer: int = -1
    graph_reg_warmup_epochs: int = 0
    graph_entropy_reg: float = 0.0
    graph_target_entropy: float | None = None
    graph_target_entropy_reg: float = 0.0
    graph_temporal_smooth_reg: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "GraphRegularisationConfig":
        if values is None:
            config = cls()
            config.validate()
            return config

        allowed = {
            "graph_reg_layer",
            "graph_reg_warmup_epochs",
            "graph_entropy_reg",
            "graph_target_entropy",
            "graph_target_entropy_reg",
            "graph_temporal_smooth_reg",
        }

        unknown = set(values) - allowed

        if unknown:
            raise KeyError(
                "Unknown graph-regularisation fields: "
                f"{sorted(unknown)}."
            )

        config = cls(
            graph_reg_layer=int(
                values.get(
                    "graph_reg_layer",
                    -1,
                )
            ),
            graph_reg_warmup_epochs=int(
                values.get(
                    "graph_reg_warmup_epochs",
                    0,
                )
            ),
            graph_entropy_reg=float(
                values.get(
                    "graph_entropy_reg",
                    0.0,
                )
            ),
            graph_target_entropy=(
                None
                if values.get(
                    "graph_target_entropy",
                    None,
                )
                is None
                else float(
                    values[
                        "graph_target_entropy"
                    ]
                )
            ),
            graph_target_entropy_reg=float(
                values.get(
                    "graph_target_entropy_reg",
                    0.0,
                )
            ),
            graph_temporal_smooth_reg=float(
                values.get(
                    "graph_temporal_smooth_reg",
                    0.0,
                )
            ),
        )

        config.validate()

        return config

    def validate(self) -> None:
        if self.graph_reg_layer < -1:
            raise ValueError(
                "graph_reg_layer must be -1 or a non-negative "
                "spatio-temporal block index."
            )

        if self.graph_reg_warmup_epochs < 0:
            raise ValueError(
                "graph_reg_warmup_epochs cannot be negative."
            )

        for name, value in (
            (
                "graph_entropy_reg",
                self.graph_entropy_reg,
            ),
            (
                "graph_target_entropy_reg",
                self.graph_target_entropy_reg,
            ),
            (
                "graph_temporal_smooth_reg",
                self.graph_temporal_smooth_reg,
            ),
        ):
            if (
                not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )

        if self.graph_target_entropy is not None:
            target = float(
                self.graph_target_entropy
            )

            if (
                not math.isfinite(target)
                or target < 0.0
            ):
                raise ValueError(
                    "graph_target_entropy must be finite and "
                    "non-negative when supplied."
                )

    @property
    def enabled(self) -> bool:
        return any(
            coefficient > 0.0
            for coefficient in (
                self.graph_entropy_reg,
                self.graph_target_entropy_reg,
                self.graph_temporal_smooth_reg,
            )
        )


@dataclass
class GraphRegularisationLoss:
    """Graph regularisation and diagnostics for one model batch."""

    total: Tensor
    unscaled_total: Tensor
    mean_row_entropy: Tensor | None
    mean_effective_neighbours: Tensor | None
    entropy_loss: Tensor
    entropy_penalty: Tensor
    target_entropy: Tensor | None
    target_entropy_loss: Tensor
    target_entropy_penalty: Tensor
    temporal_smooth_loss: Tensor
    temporal_smooth_penalty: Tensor
    warmup_scale: Tensor
    valid_smoothing_pairs: int
    selected_layer: int | None

    def detached_log_values(
        self,
        *,
        prefix: str = "graph_reg",
    ) -> dict[str, float | int]:
        values: dict[
            str,
            float | int,
        ] = {
            f"{prefix}/loss": float(
                self.total.detach().cpu()
            ),
            f"{prefix}/unscaled_loss": float(
                self.unscaled_total.detach().cpu()
            ),
            f"{prefix}/entropy_loss": float(
                self.entropy_loss.detach().cpu()
            ),
            f"{prefix}/entropy_penalty": float(
                self.entropy_penalty.detach().cpu()
            ),
            f"{prefix}/target_entropy_loss": float(
                self.target_entropy_loss.detach().cpu()
            ),
            f"{prefix}/target_entropy_penalty": float(
                self.target_entropy_penalty.detach().cpu()
            ),
            f"{prefix}/temporal_smooth_loss": float(
                self.temporal_smooth_loss.detach().cpu()
            ),
            f"{prefix}/temporal_smooth_penalty": float(
                self.temporal_smooth_penalty.detach().cpu()
            ),
            f"{prefix}/warmup_scale": float(
                self.warmup_scale.detach().cpu()
            ),
            f"{prefix}/valid_smoothing_pairs": int(
                self.valid_smoothing_pairs
            ),
        }

        if self.mean_row_entropy is not None:
            values[
                f"{prefix}/mean_row_entropy"
            ] = float(
                self.mean_row_entropy.detach().cpu()
            )

        if self.mean_effective_neighbours is not None:
            values[
                f"{prefix}/mean_effective_neighbours"
            ] = float(
                self.mean_effective_neighbours
                .detach()
                .cpu()
            )

        if self.target_entropy is not None:
            values[
                f"{prefix}/target_entropy"
            ] = float(
                self.target_entropy.detach().cpu()
            )

        if self.selected_layer is not None:
            values[
                f"{prefix}/selected_layer"
            ] = int(
                self.selected_layer
            )

        return values


@dataclass
class DynamicGraphLoss:
    """Complete model objective with token, graph and backcast terms."""

    total: Tensor
    token: FutureTokenLoss
    graph: GraphRegularisationLoss
    backcast_loss: Tensor
    backcast_penalty: Tensor

    @property
    def s1(self) -> Tensor:
        return self.token.s1

    @property
    def s2(self) -> Tensor:
        return self.token.s2

    @property
    def s1_by_step(self) -> Tensor:
        return self.token.s1_by_step

    @property
    def s2_by_step(self) -> Tensor:
        return self.token.s2_by_step

    @property
    def weights(self) -> Tensor:
        return self.token.weights

    def detached_log_values(self) -> dict[str, float | int]:
        values: dict[
            str,
            float | int,
        ] = {
            "loss/total": float(
                self.total.detach().cpu()
            ),
            "loss/token_total": float(
                self.token.total.detach().cpu()
            ),
            "loss/s1": float(
                self.token.s1.detach().cpu()
            ),
            "loss/s2": float(
                self.token.s2.detach().cpu()
            ),
            "loss/backcast": float(
                self.backcast_loss.detach().cpu()
            ),
            "loss/backcast_penalty": float(
                self.backcast_penalty.detach().cpu()
            ),
        }

        values.update(
            self.graph.detached_log_values()
        )

        return values


def _validate_adjacency(
    adjacency: Tensor,
    *,
    name: str,
    require_row_stochastic: bool = True,
    atol: float = 1.0e-5,
) -> Tensor:
    values = torch.as_tensor(
        adjacency
    )

    if (
        values.ndim < 3
        or values.shape[-1]
        != values.shape[-2]
    ):
        raise ValueError(
            f"{name} must end with square [N, N] axes. "
            f"Received {tuple(values.shape)}."
        )

    if not values.dtype.is_floating_point:
        values = values.float()

    if not torch.isfinite(values).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if torch.any(values < 0):
        raise ValueError(
            f"{name} contains negative edge weights."
        )

    if require_row_stochastic:
        row_sums = values.sum(
            dim=-1
        )

        if not torch.allclose(
            row_sums,
            torch.ones_like(
                row_sums
            ),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                f"{name} is not row-stochastic."
            )

    return values


def graph_row_entropy(
    adjacency: Tensor,
    *,
    eps: float = GRAPH_ENTROPY_EPS,
) -> Tensor:
    """Return row entropy over the source-node axis.

    This deliberately mirrors BaseDyGraph's implementation:

        a = adjacency.clamp_min(1e-8)
        row_entropy = -(a * a.log()).sum(dim=-1)
    """
    if eps <= 0:
        raise ValueError(
            "eps must be positive."
        )

    values = _validate_adjacency(
        adjacency,
        name="adjacency",
    )

    safe_values = values.clamp_min(
        float(eps)
    )

    return -(
        safe_values
        * safe_values.log()
    ).sum(
        dim=-1
    )


def mean_graph_entropy(
    adjacency: Tensor,
    *,
    eps: float = GRAPH_ENTROPY_EPS,
) -> Tensor:
    return graph_row_entropy(
        adjacency,
        eps=eps,
    ).mean()


def mean_effective_neighbours(
    adjacency: Tensor,
    *,
    eps: float = GRAPH_ENTROPY_EPS,
) -> Tensor:
    return graph_row_entropy(
        adjacency,
        eps=eps,
    ).exp().mean()


def select_graph_for_regularisation(
    graph_output: GraphOutput,
    *,
    layer: int = -1,
) -> tuple[Tensor | None, int | None]:
    """Select the graph regularised by BaseDyGraph's layer policy.

    ``layer=-1`` selects the last non-null per-layer graph, then falls
    back to ``graph_output.selected``. A non-negative value selects one
    explicit spatio-temporal block.
    """
    if layer < -1:
        raise ValueError(
            "layer must be -1 or a non-negative index."
        )

    if layer >= 0:
        if layer >= len(
            graph_output.per_layer
        ):
            raise IndexError(
                f"Requested graph layer {layer}, but only "
                f"{len(graph_output.per_layer)} layer entries exist."
            )

        selected = graph_output.per_layer[
            layer
        ]

        if selected is None:
            raise ValueError(
                f"graph.per_layer[{layer}] is None and cannot "
                "be regularised."
            )

        return selected, layer

    for layer_index in range(
        len(graph_output.per_layer) - 1,
        -1,
        -1,
    ):
        candidate = graph_output.per_layer[
            layer_index
        ]

        if candidate is not None:
            return candidate, layer_index

    return graph_output.selected, None


def graph_regularisation_warmup_scale(
    *,
    current_epoch: int,
    warmup_epochs: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """BaseDyGraph warm-up: epoch zero receives ``1 / warmup``."""
    if current_epoch < 0:
        raise ValueError(
            "current_epoch cannot be negative."
        )

    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs cannot be negative."
        )

    if warmup_epochs <= 0:
        value = 1.0
    else:
        value = min(
            1.0,
            max(
                0.0,
                float(
                    current_epoch + 1
                )
                / float(warmup_epochs),
            ),
        )

    return torch.tensor(
        value,
        device=device,
        dtype=dtype,
    )


def build_adjacent_window_pairs(
    *,
    origin_idx: Tensor,
    expected_origin_delta: int,
    sample_idx: Tensor | None = None,
    trajectory_id: Tensor | None = None,
) -> Tensor:
    """Build exact adjacent-window pairs for graph smoothing.

    Real data use ``sample_idx`` to identify a trading session.
    Synthetic data use ``trajectory_id``. The pair order is independent
    of DataLoader ordering; only windows with the same group identifier
    and an exact origin difference are paired.
    """
    if expected_origin_delta <= 0:
        raise ValueError(
            "expected_origin_delta must be positive."
        )

    if (
        sample_idx is None
        and trajectory_id is None
    ):
        raise ValueError(
            "Provide sample_idx for real data or trajectory_id "
            "for synthetic data."
        )

    if (
        sample_idx is not None
        and trajectory_id is not None
    ):
        raise ValueError(
            "Provide only one of sample_idx and trajectory_id."
        )

    origins = torch.as_tensor(
        origin_idx
    ).long().reshape(-1)

    groups = torch.as_tensor(
        sample_idx
        if sample_idx is not None
        else trajectory_id
    ).long().reshape(-1)

    if groups.shape != origins.shape:
        raise ValueError(
            "Group IDs and origin_idx must have the same shape."
        )

    lookup: dict[
        tuple[int, int],
        int,
    ] = {}

    for row_index, (
        group_value,
        origin_value,
    ) in enumerate(
        zip(
            groups.detach().cpu().tolist(),
            origins.detach().cpu().tolist(),
            strict=True,
        )
    ):
        key = (
            int(group_value),
            int(origin_value),
        )

        if key in lookup:
            raise ValueError(
                "Duplicate [group, origin] metadata prevents "
                f"unambiguous smoothing pairs: {key}."
            )

        lookup[key] = row_index

    pairs: list[
        tuple[int, int]
    ] = []

    for (
        group_value,
        origin_value,
    ), earlier_index in lookup.items():
        later_key = (
            group_value,
            origin_value
            + int(
                expected_origin_delta
            ),
        )

        later_index = lookup.get(
            later_key
        )

        if later_index is not None:
            pairs.append(
                (
                    earlier_index,
                    later_index,
                )
            )

    pairs.sort()

    if not pairs:
        return torch.empty(
            (0, 2),
            dtype=torch.long,
            device=origins.device,
        )

    return torch.tensor(
        pairs,
        dtype=torch.long,
        device=origins.device,
    )


def window_graph_temporal_smoothness(
    adjacency: Tensor,
    adjacent_window_pairs: Tensor,
) -> Tensor:
    """Apply BaseDyGraph's squared-change penalty to valid windows."""
    values = _validate_adjacency(
        adjacency,
        name="adjacency",
    )

    if values.ndim != 4:
        raise ValueError(
            "Window-level graph smoothing expects adjacency with "
            "shape [B, G, N, N]."
        )

    pairs = torch.as_tensor(
        adjacent_window_pairs,
        device=values.device,
    ).long()

    if (
        pairs.ndim != 2
        or tuple(pairs.shape[1:])
        != (2,)
    ):
        raise ValueError(
            "adjacent_window_pairs must have shape [P, 2]."
        )

    if pairs.numel() == 0:
        return values.sum() * 0.0

    if (
        pairs.min().item() < 0
        or pairs.max().item()
        >= values.shape[0]
    ):
        raise IndexError(
            "adjacent_window_pairs contains an out-of-range "
            "batch index."
        )

    earlier = values.index_select(
        dim=0,
        index=pairs[:, 0],
    )

    later = values.index_select(
        dim=0,
        index=pairs[:, 1],
    )

    return (
        later
        - earlier
    ).square().mean()


def basedygraph_sequence_temporal_smoothness(
    adjacency_sequence: Tensor,
) -> Tensor:
    """Exact BaseDyGraph frame-to-frame smoothness formula.

    Args:
        adjacency_sequence:
            Graph sequence ``[B, T, G, N, N]``.
    """
    values = _validate_adjacency(
        adjacency_sequence,
        name="adjacency_sequence",
    )

    if values.ndim != 5:
        raise ValueError(
            "adjacency_sequence must have shape [B, T, G, N, N]."
        )

    if values.shape[1] <= 1:
        return values.sum() * 0.0

    return (
        values[:, 1:]
        - values[:, :-1]
    ).square().mean()


def _normalise_true_graph(
    true_graph: Tensor,
) -> Tensor:
    values = _validate_adjacency(
        true_graph,
        name="true_graph",
        require_row_stochastic=False,
    )

    row_mass = values.sum(
        dim=-1,
        keepdim=True,
    )

    if torch.any(
        row_mass <= 0
    ):
        raise ValueError(
            "true_graph contains an empty target row."
        )

    return values / row_mass


def compute_graph_regularisation(
    graph_output: GraphOutput,
    *,
    config: GraphRegularisationConfig,
    current_epoch: int,
    reference_tensor: Tensor | None = None,
    true_graph: Tensor | None = None,
    adjacent_window_pairs: Tensor | None = None,
    temporal_graph_sequence: Tensor | None = None,
) -> GraphRegularisationLoss:
    """Compute BaseDyGraph-style graph regularisation.

    Temporal smoothing has two valid modes:

    1. ``temporal_graph_sequence`` supplies the original BaseDyGraph
       ``[B, T, G, N, N]`` graph sequence.
    2. ``adjacent_window_pairs`` supplies valid pair indices for this
       project's one-graph-per-window ``[B, G, N, N]`` contract.

    Supplying unrelated shuffled batch rows as temporal neighbours is
    intentionally impossible through this interface.
    """
    config.validate()

    selected_graph, selected_layer = (
        select_graph_for_regularisation(
            graph_output,
            layer=config.graph_reg_layer,
        )
    )

    if selected_graph is None:
        if config.enabled:
            raise ValueError(
                "Graph regularisation is enabled, but the model "
                "returned no selected graph."
            )

        if reference_tensor is None:
            zero = torch.zeros(
                (),
                dtype=torch.float32,
            )
        else:
            zero = reference_tensor.sum() * 0.0

        one = zero + 1.0

        return GraphRegularisationLoss(
            total=zero,
            unscaled_total=zero,
            mean_row_entropy=None,
            mean_effective_neighbours=None,
            entropy_loss=zero,
            entropy_penalty=zero,
            target_entropy=None,
            target_entropy_loss=zero,
            target_entropy_penalty=zero,
            temporal_smooth_loss=zero,
            temporal_smooth_penalty=zero,
            warmup_scale=one,
            valid_smoothing_pairs=0,
            selected_layer=None,
        )

    adjacency = _validate_adjacency(
        selected_graph,
        name="selected_graph",
    )

    zero = adjacency.sum() * 0.0

    row_entropy = graph_row_entropy(
        adjacency
    )

    entropy = row_entropy.mean()
    effective_neighbours = row_entropy.exp().mean()

    entropy_loss = entropy
    entropy_penalty = (
        float(
            config.graph_entropy_reg
        )
        * entropy_loss
    )

    target_entropy: Tensor | None = None
    target_entropy_loss = zero
    target_entropy_penalty = zero

    if config.graph_target_entropy_reg > 0.0:
        if config.graph_target_entropy is not None:
            target_entropy = torch.as_tensor(
                float(
                    config.graph_target_entropy
                ),
                device=adjacency.device,
                dtype=adjacency.dtype,
            )
        elif true_graph is not None:
            normalised_truth = _normalise_true_graph(
                true_graph
            ).to(
                device=adjacency.device,
                dtype=adjacency.dtype,
            )

            target_entropy = mean_graph_entropy(
                normalised_truth
            ).detach()
        else:
            # This is BaseDyGraph's exact real-data fallback: the
            # target term is a no-op unless an explicit target or
            # synthetic true graph is available.
            target_entropy = entropy.detach()

        maximum_entropy = math.log(
            float(
                adjacency.shape[-1]
            )
        )

        if float(
            target_entropy.detach().cpu()
        ) > maximum_entropy + 1.0e-5:
            raise ValueError(
                "The target graph entropy exceeds log(num_nodes)."
            )

        target_entropy_loss = (
            entropy
            - target_entropy.detach()
        ).square()

        target_entropy_penalty = (
            float(
                config.graph_target_entropy_reg
            )
            * target_entropy_loss
        )

    temporal_smooth_loss = zero
    temporal_smooth_penalty = zero
    valid_smoothing_pairs = 0

    if config.graph_temporal_smooth_reg > 0.0:
        if (
            adjacent_window_pairs is not None
            and temporal_graph_sequence is not None
        ):
            raise ValueError(
                "Supply adjacent_window_pairs or "
                "temporal_graph_sequence, not both."
            )

        if temporal_graph_sequence is not None:
            temporal_smooth_loss = (
                basedygraph_sequence_temporal_smoothness(
                    temporal_graph_sequence
                )
            )

            valid_smoothing_pairs = max(
                0,
                int(
                    temporal_graph_sequence.shape[0]
                    * (
                        temporal_graph_sequence.shape[1]
                        - 1
                    )
                ),
            )
        elif adjacent_window_pairs is not None:
            pairs = torch.as_tensor(
                adjacent_window_pairs
            )

            valid_smoothing_pairs = int(
                pairs.shape[0]
            )

            if valid_smoothing_pairs == 0:
                raise ValueError(
                    "Temporal graph smoothing is enabled, but this "
                    "batch contains no valid adjacent-window pairs. "
                    "Use an adjacency-aware batch sampler rather than "
                    "comparing unrelated shuffled windows."
                )

            temporal_smooth_loss = (
                window_graph_temporal_smoothness(
                    adjacency,
                    pairs,
                )
            )
        else:
            raise ValueError(
                "Temporal graph smoothing is enabled, but neither "
                "adjacent_window_pairs nor temporal_graph_sequence "
                "was supplied."
            )

        temporal_smooth_penalty = (
            float(
                config.graph_temporal_smooth_reg
            )
            * temporal_smooth_loss
        )

    unscaled_total = (
        entropy_penalty
        + target_entropy_penalty
        + temporal_smooth_penalty
    )

    warmup_scale = (
        graph_regularisation_warmup_scale(
            current_epoch=current_epoch,
            warmup_epochs=(
                config.graph_reg_warmup_epochs
            ),
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
    )

    total = (
        unscaled_total
        * warmup_scale
    )

    if not torch.isfinite(total):
        raise RuntimeError(
            "Graph regularisation produced a non-finite loss."
        )

    return GraphRegularisationLoss(
        total=total,
        unscaled_total=unscaled_total,
        mean_row_entropy=entropy,
        mean_effective_neighbours=(
            effective_neighbours
        ),
        entropy_loss=entropy_loss,
        entropy_penalty=entropy_penalty,
        target_entropy=target_entropy,
        target_entropy_loss=(
            target_entropy_loss
        ),
        target_entropy_penalty=(
            target_entropy_penalty
        ),
        temporal_smooth_loss=(
            temporal_smooth_loss
        ),
        temporal_smooth_penalty=(
            temporal_smooth_penalty
        ),
        warmup_scale=warmup_scale,
        valid_smoothing_pairs=(
            valid_smoothing_pairs
        ),
        selected_layer=selected_layer,
    )


def combine_dynamic_graph_losses(
    token_loss: FutureTokenLoss,
    graph_loss: GraphRegularisationLoss,
    *,
    backcast_loss: Tensor | None = None,
    backcast_loss_weight: float = 0.0,
) -> DynamicGraphLoss:
    """Combine token, graph and optional backcast objectives."""
    if (
        not math.isfinite(
            float(
                backcast_loss_weight
            )
        )
        or backcast_loss_weight < 0.0
    ):
        raise ValueError(
            "backcast_loss_weight must be finite and non-negative."
        )

    if backcast_loss is None:
        resolved_backcast = (
            token_loss.total.sum()
            * 0.0
        )
    else:
        resolved_backcast = torch.as_tensor(
            backcast_loss,
            device=token_loss.total.device,
            dtype=token_loss.total.dtype,
        )

        if resolved_backcast.ndim != 0:
            raise ValueError(
                "backcast_loss must be scalar."
            )

        if not torch.isfinite(
            resolved_backcast
        ):
            raise ValueError(
                "backcast_loss is non-finite."
            )

    graph_total = graph_loss.total.to(
        device=token_loss.total.device,
        dtype=token_loss.total.dtype,
    )

    backcast_penalty = (
        float(
            backcast_loss_weight
        )
        * resolved_backcast
    )

    total = (
        token_loss.total
        + graph_total
        + backcast_penalty
    )

    if not torch.isfinite(total):
        raise RuntimeError(
            "The complete model loss is non-finite."
        )

    return DynamicGraphLoss(
        total=total,
        token=token_loss,
        graph=graph_loss,
        backcast_loss=resolved_backcast,
        backcast_penalty=backcast_penalty,
    )


def _row_softmax_without_self_loops(
    logits: Tensor,
) -> Tensor:
    num_nodes = int(
        logits.shape[-1]
    )

    diagonal = torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=logits.device,
    )

    return torch.softmax(
        logits.masked_fill(
            diagonal,
            -1.0e9,
        ),
        dim=-1,
    )


def _cpu_smoke_test() -> None:
    torch.manual_seed(42)

    batch_size = 4
    num_heads = 2
    num_nodes = 5

    logits = torch.nn.Parameter(
        torch.randn(
            batch_size,
            num_heads,
            num_nodes,
            num_nodes,
        )
        * 0.2
    )

    adjacency = (
        _row_softmax_without_self_loops(
            logits
        )
    )

    graph_output = GraphOutput(
        selected=adjacency,
        per_layer=(
            None,
            adjacency,
        ),
        logits=logits,
    )

    group_ids = torch.tensor(
        [
            0,
            0,
            0,
            1,
        ]
    )

    origin_idx = torch.tensor(
        [
            10,
            25,
            40,
            10,
        ]
    )

    pairs = build_adjacent_window_pairs(
        sample_idx=group_ids,
        origin_idx=origin_idx,
        expected_origin_delta=15,
    )

    if not torch.equal(
        pairs.cpu(),
        torch.tensor(
            [
                [0, 1],
                [1, 2],
            ]
        ),
    ):
        raise AssertionError(
            "Adjacent-window pair construction is incorrect."
        )

    config = GraphRegularisationConfig(
        graph_reg_layer=-1,
        graph_reg_warmup_epochs=4,
        graph_entropy_reg=0.05,
        graph_target_entropy=(
            math.log(2.5)
        ),
        graph_target_entropy_reg=0.1,
        graph_temporal_smooth_reg=0.2,
    )

    loss = compute_graph_regularisation(
        graph_output,
        config=config,
        current_epoch=0,
        adjacent_window_pairs=pairs,
    )

    if loss.selected_layer != 1:
        raise AssertionError(
            "The last non-null graph layer was not selected."
        )

    if not torch.allclose(
        loss.warmup_scale,
        torch.tensor(
            0.25,
            dtype=loss.warmup_scale.dtype,
        ),
    ):
        raise AssertionError(
            "Graph regularisation warm-up is incorrect."
        )

    if loss.valid_smoothing_pairs != 2:
        raise AssertionError(
            "The temporal smoothing pair count is incorrect."
        )

    if (
        not torch.isfinite(loss.total)
        or float(loss.total.detach()) <= 0.0
    ):
        raise AssertionError(
            "Graph regularisation is not finite and positive."
        )

    loss.total.backward()

    if (
        logits.grad is None
        or not torch.isfinite(
            logits.grad
        ).all()
        or float(
            logits.grad.abs().sum()
        )
        == 0.0
    ):
        raise AssertionError(
            "Graph regularisation did not reach graph logits."
        )

    entropy = mean_graph_entropy(
        adjacency.detach()
    )

    exact_target_config = (
        GraphRegularisationConfig(
            graph_target_entropy=float(
                entropy.detach()
            ),
            graph_target_entropy_reg=1.0,
        )
    )

    exact_target_loss = (
        compute_graph_regularisation(
            GraphOutput(
                selected=(
                    adjacency.detach()
                )
            ),
            config=exact_target_config,
            current_epoch=0,
        )
    )

    if float(
        exact_target_loss
        .target_entropy_loss
        .abs()
    ) > 1.0e-10:
        raise AssertionError(
            "Target-entropy matching should be zero at its target."
        )

    sequence = torch.stack(
        [
            adjacency.detach(),
            torch.roll(
                adjacency.detach(),
                shifts=1,
                dims=-1,
            ),
            adjacency.detach(),
        ],
        dim=1,
    )

    sequence_loss = (
        basedygraph_sequence_temporal_smoothness(
            sequence
        )
    )

    manual_sequence_loss = (
        sequence[:, 1:]
        - sequence[:, :-1]
    ).square().mean()

    if not torch.allclose(
        sequence_loss,
        manual_sequence_loss,
    ):
        raise AssertionError(
            "BaseDyGraph sequence smoothing is incorrect."
        )

    disabled = compute_graph_regularisation(
        GraphOutput(
            selected=None
        ),
        config=GraphRegularisationConfig(),
        current_epoch=0,
        reference_tensor=torch.tensor(
            2.0
        ),
    )

    if float(
        disabled.total
    ) != 0.0:
        raise AssertionError(
            "Disabled no-graph regularisation must be zero."
        )

    token_zero = torch.tensor(
        0.0
    )

    token_loss = FutureTokenLoss(
        total=torch.tensor(
            3.0
        ),
        s1=torch.tensor(
            2.0
        ),
        s2=torch.tensor(
            1.0
        ),
        s1_by_step=torch.ones(
            2
        ),
        s2_by_step=torch.ones(
            2
        ),
        weights=torch.ones(
            2
        ),
    )

    graph_zero = GraphRegularisationLoss(
        total=token_zero,
        unscaled_total=token_zero,
        mean_row_entropy=None,
        mean_effective_neighbours=None,
        entropy_loss=token_zero,
        entropy_penalty=token_zero,
        target_entropy=None,
        target_entropy_loss=token_zero,
        target_entropy_penalty=token_zero,
        temporal_smooth_loss=token_zero,
        temporal_smooth_penalty=token_zero,
        warmup_scale=torch.tensor(
            1.0
        ),
        valid_smoothing_pairs=0,
        selected_layer=None,
    )

    combined = combine_dynamic_graph_losses(
        token_loss,
        graph_zero,
        backcast_loss=torch.tensor(
            4.0
        ),
        backcast_loss_weight=0.5,
    )

    if not torch.allclose(
        combined.total,
        torch.tensor(
            5.0
        ),
    ):
        raise AssertionError(
            "Combined model loss is incorrect."
        )

    print(
        "DYNAMIC GRAPH LOSS CPU SMOKE TEST PASSED"
    )
    print(
        "Tested: entropy, target entropy, warm-up, "
        "adjacent-window smoothing, BaseDyGraph sequence "
        "smoothing, gradients, and composite loss."
    )


if __name__ == "__main__":
    _cpu_smoke_test()

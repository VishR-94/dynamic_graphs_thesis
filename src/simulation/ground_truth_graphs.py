from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


DEFAULT_REGIME_NAMES = (
    "bear",
    "neutral",
    "bull",
)

# Total number of active source neighbours in every target row.
#
# The default design deliberately makes the Bear graph dense, the
# Neutral graph moderately sparse, and the Bull graph sparse.
DEFAULT_TOTAL_IN_DEGREES = (
    8,  # bear
    4,  # neutral
    3,  # bull
)


@dataclass(frozen=True)
class GroundTruthGraphConfig:
    """Configuration for the three known regime graphs.

    Every graph is directed, non-negative, zero-diagonal and
    row-stochastic. The orientation is:

        adjacency[target, source]

    Each target row contains a shared backbone plus a regime-specific
    set of source nodes. Regime-specific supports are disjoint by
    construction, so the shared backbone is the only support common to
    all regimes.

    The default 16-node design gives:

        Bear:    8 incoming neighbours per target
        Neutral: 4 incoming neighbours per target
        Bull:    3 incoming neighbours per target

    of which 2 neighbours per target form the shared backbone.
    """

    num_nodes: int = 16
    regime_names: tuple[str, ...] = DEFAULT_REGIME_NAMES
    total_in_degrees: tuple[int, ...] = (
        DEFAULT_TOTAL_IN_DEGREES
    )
    shared_in_degree: int = 2
    weight_low: float = 0.5
    weight_high: float = 1.5


@dataclass(frozen=True)
class GroundTruthGraphSet:
    """The complete static set of known regime graphs.

    Attributes:
        graphs:
            Row-stochastic adjacencies with shape ``[R, N, N]``.

        support_masks:
            Boolean edge-support masks with shape ``[R, N, N]``.

        shared_support:
            Boolean support shared by all regimes, shape ``[N, N]``.

        regime_specific_support:
            Boolean regime-only support, shape ``[R, N, N]``.

        raw_weights:
            Positive pre-normalisation weights shared across regimes,
            shape ``[N, N]``.

        regime_names:
            Regime ordering aligned with the first graph dimension.

        total_in_degrees:
            Configured active neighbours per target row for each regime.

        shared_in_degree:
            Number of shared incoming neighbours per target.

        seed:
            Seed used to sample supports and raw edge weights.
    """

    graphs: Tensor
    support_masks: Tensor
    shared_support: Tensor
    regime_specific_support: Tensor
    raw_weights: Tensor
    regime_names: tuple[str, ...]
    total_in_degrees: tuple[int, ...]
    shared_in_degree: int
    seed: int
    config: GroundTruthGraphConfig

    @property
    def num_regimes(self) -> int:
        return int(self.graphs.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.graphs.shape[1])

    @property
    def mean_graph(self) -> Tensor:
        """Return the equally weighted average regime graph."""
        return self.graphs.mean(dim=0)

    def graph_for_regime(
        self,
        regime: int | str,
    ) -> Tensor:
        """Return one regime graph by integer ID or regime name."""
        if isinstance(regime, str):
            try:
                regime_id = self.regime_names.index(regime)
            except ValueError as error:
                raise KeyError(
                    f"Unknown regime name {regime!r}. Expected one of "
                    f"{self.regime_names}."
                ) from error
        else:
            regime_id = int(regime)

        if not 0 <= regime_id < self.num_regimes:
            raise IndexError(
                f"regime ID must lie in [0, {self.num_regimes - 1}]."
            )

        return self.graphs[regime_id]

    def select(
        self,
        regime_ids: Tensor,
    ) -> Tensor:
        """Select active graphs for arbitrary regime-ID tensor shapes.

        Args:
            regime_ids:
                Integer regime IDs with any shape, for example
                ``[B]`` or ``[B, T]``.

        Returns:
            Active graph tensor with shape
            ``[*regime_ids.shape, N, N]``.
        """
        regime_ids = torch.as_tensor(
            regime_ids,
            dtype=torch.long,
            device=self.graphs.device,
        )

        if regime_ids.numel() == 0:
            raise ValueError(
                "regime_ids must contain at least one value."
            )

        if (
            torch.any(regime_ids < 0)
            or torch.any(regime_ids >= self.num_regimes)
        ):
            raise ValueError(
                "regime_ids contains an unknown regime ID."
            )

        return self.graphs[regime_ids]

    def metadata(self) -> dict[str, Any]:
        """Return graph-generation metadata suitable for ``torch.save``."""
        return {
            "kind": "known_regime_graph_set",
            "config": asdict(self.config),
            "regime_names": list(self.regime_names),
            "regime_to_id": {
                name: idx
                for idx, name in enumerate(self.regime_names)
            },
            "orientation": "row=target,column=source",
            "weighting": (
                "shared positive raw edge weights, masked by each "
                "regime support and row-normalised"
            ),
            "support_design": (
                "shared backbone plus disjoint regime-specific edges"
            ),
            "seed": int(self.seed),
            "graphs": self.graphs.detach().cpu(),
            "support_masks": self.support_masks.detach().cpu(),
            "shared_support": self.shared_support.detach().cpu(),
            "regime_specific_support": (
                self.regime_specific_support.detach().cpu()
            ),
            "raw_weights": self.raw_weights.detach().cpu(),
            "mean_graph": self.mean_graph.detach().cpu(),
        }


@dataclass(frozen=True)
class GroundTruthGraphDiagnostics:
    """Structural diagnostics for the generated graph set."""

    active_in_degree: Tensor
    density: Tensor
    row_sum_min: Tensor
    row_sum_max: Tensor
    mean_row_entropy: Tensor
    min_row_entropy: Tensor
    max_row_entropy: Tensor
    mean_effective_neighbours: Tensor
    pairwise_support_jaccard: Tensor
    pairwise_weight_correlation: Tensor
    pairwise_weight_rmse: Tensor

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "active_in_degree": self.active_in_degree,
            "density": self.density,
            "row_sum_min": self.row_sum_min,
            "row_sum_max": self.row_sum_max,
            "mean_row_entropy": self.mean_row_entropy,
            "min_row_entropy": self.min_row_entropy,
            "max_row_entropy": self.max_row_entropy,
            "mean_effective_neighbours": (
                self.mean_effective_neighbours
            ),
            "pairwise_support_jaccard": (
                self.pairwise_support_jaccard
            ),
            "pairwise_weight_correlation": (
                self.pairwise_weight_correlation
            ),
            "pairwise_weight_rmse": self.pairwise_weight_rmse,
        }


class GroundTruthGraphGenerator:
    """Generate three related directed ground-truth graphs.

    The generator creates one shared incoming-neighbour backbone for
    every target node. Each regime then receives its own disjoint set
    of additional incoming edges.

    A single positive raw-weight matrix is sampled and reused across
    regimes. Each regime masks that matrix by its support and
    row-normalises the result. Consequently, a shared edge has the same
    pre-normalisation strength in every regime, while its final
    row-stochastic weight may change because each regime has a different
    number and set of active neighbours.
    """

    def __init__(
        self,
        config: GroundTruthGraphConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = (
            GroundTruthGraphConfig()
            if config is None
            else config
        )
        self.device = torch.device(device)
        self.dtype = dtype

        self._validate_config()

        self.regime_names = tuple(
            self.config.regime_names
        )
        self.total_in_degrees = tuple(
            int(value)
            for value in self.config.total_in_degrees
        )
        self.num_regimes = len(self.regime_names)
        self.num_nodes = int(self.config.num_nodes)
        self.shared_in_degree = int(
            self.config.shared_in_degree
        )

    def _validate_config(self) -> None:
        config = self.config

        if (
            isinstance(config.num_nodes, bool)
            or not isinstance(config.num_nodes, int)
            or config.num_nodes < 3
        ):
            raise ValueError(
                "num_nodes must be an integer of at least 3."
            )

        names = tuple(config.regime_names)

        if len(names) != 3:
            raise ValueError(
                "This experiment requires exactly three regimes."
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "regime_names must be unique."
            )

        if any(
            not isinstance(name, str) or not name
            for name in names
        ):
            raise ValueError(
                "Every regime name must be a non-empty string."
            )

        degrees = tuple(config.total_in_degrees)

        if len(degrees) != len(names):
            raise ValueError(
                "total_in_degrees must contain one value per regime."
            )

        if (
            isinstance(config.shared_in_degree, bool)
            or not isinstance(config.shared_in_degree, int)
            or config.shared_in_degree < 1
        ):
            raise ValueError(
                "shared_in_degree must be a positive integer."
            )

        maximum_degree = config.num_nodes - 1

        for regime_name, degree in zip(names, degrees):
            if (
                isinstance(degree, bool)
                or not isinstance(degree, int)
                or not (
                    config.shared_in_degree
                    <= degree
                    <= maximum_degree
                )
            ):
                raise ValueError(
                    f"Total in-degree for {regime_name!r} must lie "
                    f"between shared_in_degree="
                    f"{config.shared_in_degree} and {maximum_degree}."
                )

        required_distinct_sources = (
            config.shared_in_degree
            + sum(
                degree - config.shared_in_degree
                for degree in degrees
            )
        )

        if required_distinct_sources > maximum_degree:
            raise ValueError(
                "The requested shared plus disjoint regime-specific "
                "supports require "
                f"{required_distinct_sources} distinct sources per "
                f"target, but only {maximum_degree} are available. "
                "Reduce an in-degree, reduce shared_in_degree, or "
                "increase num_nodes."
            )

        if not (
            torch.isfinite(
                torch.tensor(
                    [
                        config.weight_low,
                        config.weight_high,
                    ],
                    dtype=torch.float64,
                )
            ).all()
        ):
            raise ValueError(
                "weight_low and weight_high must be finite."
            )

        if (
            config.weight_low <= 0
            or config.weight_high <= config.weight_low
        ):
            raise ValueError(
                "Require 0 < weight_low < weight_high."
            )

        if not self.dtype.is_floating_point:
            raise TypeError(
                "dtype must be a floating-point torch dtype."
            )

    def _make_generator(
        self,
        seed: int,
    ) -> torch.Generator:
        generator = torch.Generator(
            device="cpu"
        )
        generator.manual_seed(
            int(seed)
        )
        return generator

    def generate(
        self,
        *,
        seed: int = 7,
    ) -> GroundTruthGraphSet:
        """Generate the shared backbone and three regime graphs."""
        generator = self._make_generator(seed)

        shared_support = torch.zeros(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            dtype=torch.bool,
        )

        regime_specific_support = torch.zeros(
            (
                self.num_regimes,
                self.num_nodes,
                self.num_nodes,
            ),
            dtype=torch.bool,
        )

        all_nodes = torch.arange(
            self.num_nodes,
            dtype=torch.long,
        )

        for target in range(self.num_nodes):
            candidate_sources = all_nodes[
                all_nodes != target
            ]

            permutation = candidate_sources[
                torch.randperm(
                    candidate_sources.numel(),
                    generator=generator,
                )
            ]

            cursor = 0

            shared_sources = permutation[
                cursor:
                cursor + self.shared_in_degree
            ]
            cursor += self.shared_in_degree

            shared_support[
                target,
                shared_sources,
            ] = True

            for regime_idx, total_degree in enumerate(
                self.total_in_degrees
            ):
                extra_degree = (
                    total_degree
                    - self.shared_in_degree
                )

                extra_sources = permutation[
                    cursor:
                    cursor + extra_degree
                ]
                cursor += extra_degree

                regime_specific_support[
                    regime_idx,
                    target,
                    extra_sources,
                ] = True

        support_masks = (
            regime_specific_support
            | shared_support.unsqueeze(0)
        )

        random_weights = torch.rand(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            generator=generator,
            dtype=self.dtype,
        )

        raw_weights = (
            float(self.config.weight_low)
            + (
                float(self.config.weight_high)
                - float(self.config.weight_low)
            )
            * random_weights
        )

        diagonal = torch.arange(
            self.num_nodes
        )
        raw_weights[
            diagonal,
            diagonal,
        ] = 0.0

        unnormalised = (
            support_masks.to(self.dtype)
            * raw_weights.unsqueeze(0)
        )

        row_sums = unnormalised.sum(
            dim=-1,
            keepdim=True,
        )

        if torch.any(row_sums <= 0):
            raise RuntimeError(
                "At least one graph row has no positive edge mass."
            )

        graphs = (
            unnormalised
            / row_sums
        )

        return GroundTruthGraphSet(
            graphs=graphs.to(
                device=self.device,
                dtype=self.dtype,
            ),
            support_masks=support_masks.to(
                device=self.device,
            ),
            shared_support=shared_support.to(
                device=self.device,
            ),
            regime_specific_support=(
                regime_specific_support.to(
                    device=self.device,
                )
            ),
            raw_weights=raw_weights.to(
                device=self.device,
                dtype=self.dtype,
            ),
            regime_names=self.regime_names,
            total_in_degrees=self.total_in_degrees,
            shared_in_degree=self.shared_in_degree,
            seed=int(seed),
            config=self.config,
        )

    def diagnostics(
        self,
        graph_set: GroundTruthGraphSet,
    ) -> GroundTruthGraphDiagnostics:
        """Calculate graph density, entropy and pairwise separation."""
        graphs = graph_set.graphs.detach().cpu().to(
            torch.float64
        )
        support = graph_set.support_masks.detach().cpu()

        if tuple(graphs.shape) != (
            self.num_regimes,
            self.num_nodes,
            self.num_nodes,
        ):
            raise ValueError(
                "graph_set has a shape incompatible with this "
                "generator."
            )

        if support.shape != graphs.shape:
            raise ValueError(
                "support_masks must have the same shape as graphs."
            )

        active_in_degree = support.sum(
            dim=-1
        ).to(torch.long)

        density = (
            active_in_degree.to(torch.float64).mean(dim=1)
            / float(self.num_nodes - 1)
        )

        row_sums = graphs.sum(dim=-1)

        safe_graphs = graphs.clamp_min(
            torch.finfo(torch.float64).tiny
        )

        row_entropy = -(
            graphs
            * torch.log(safe_graphs)
        ).sum(dim=-1)

        effective_neighbours = torch.exp(
            row_entropy
        )

        pairwise_jaccard = torch.eye(
            self.num_regimes,
            dtype=torch.float64,
        )

        pairwise_correlation = torch.eye(
            self.num_regimes,
            dtype=torch.float64,
        )

        pairwise_rmse = torch.zeros(
            (
                self.num_regimes,
                self.num_regimes,
            ),
            dtype=torch.float64,
        )

        off_diagonal = ~torch.eye(
            self.num_nodes,
            dtype=torch.bool,
        )

        for left in range(self.num_regimes):
            for right in range(
                left + 1,
                self.num_regimes,
            ):
                left_support = support[
                    left
                ][off_diagonal]
                right_support = support[
                    right
                ][off_diagonal]

                intersection = (
                    left_support
                    & right_support
                ).sum().item()

                union = (
                    left_support
                    | right_support
                ).sum().item()

                jaccard = (
                    float(intersection / union)
                    if union > 0
                    else float("nan")
                )

                left_weights = graphs[
                    left
                ][off_diagonal]
                right_weights = graphs[
                    right
                ][off_diagonal]

                left_centred = (
                    left_weights
                    - left_weights.mean()
                )
                right_centred = (
                    right_weights
                    - right_weights.mean()
                )

                denominator = torch.sqrt(
                    left_centred.square().sum()
                    * right_centred.square().sum()
                )

                correlation = (
                    float(
                        (
                            left_centred
                            * right_centred
                        ).sum().item()
                        / denominator.item()
                    )
                    if denominator.item() > 0
                    else float("nan")
                )

                rmse = float(
                    torch.sqrt(
                        (
                            left_weights
                            - right_weights
                        ).square().mean()
                    ).item()
                )

                pairwise_jaccard[
                    left,
                    right,
                ] = jaccard
                pairwise_jaccard[
                    right,
                    left,
                ] = jaccard

                pairwise_correlation[
                    left,
                    right,
                ] = correlation
                pairwise_correlation[
                    right,
                    left,
                ] = correlation

                pairwise_rmse[
                    left,
                    right,
                ] = rmse
                pairwise_rmse[
                    right,
                    left,
                ] = rmse

        return GroundTruthGraphDiagnostics(
            active_in_degree=active_in_degree,
            density=density,
            row_sum_min=row_sums.min(dim=1).values,
            row_sum_max=row_sums.max(dim=1).values,
            mean_row_entropy=row_entropy.mean(dim=1),
            min_row_entropy=row_entropy.min(dim=1).values,
            max_row_entropy=row_entropy.max(dim=1).values,
            mean_effective_neighbours=(
                effective_neighbours.mean(dim=1)
            ),
            pairwise_support_jaccard=pairwise_jaccard,
            pairwise_weight_correlation=(
                pairwise_correlation
            ),
            pairwise_weight_rmse=pairwise_rmse,
        )


def format_graph_diagnostics(
    diagnostics: GroundTruthGraphDiagnostics,
    regime_names: Sequence[str],
) -> str:
    """Format graph diagnostics as a compact terminal report."""
    names = tuple(regime_names)

    if len(names) != diagnostics.density.numel():
        raise ValueError(
            "regime_names does not match the diagnostics."
        )

    lines = [
        "Ground-truth regime graph summary:",
    ]

    for regime_idx, regime_name in enumerate(names):
        in_degree = diagnostics.active_in_degree[
            regime_idx
        ]

        unique_degrees = torch.unique(
            in_degree
        ).tolist()

        lines.append(
            "  "
            f"{regime_name:>7} | "
            f"in-degree={unique_degrees} | "
            f"density={diagnostics.density[regime_idx].item():.4f} | "
            "entropy="
            f"{diagnostics.mean_row_entropy[regime_idx].item():.4f} | "
            "effective neighbours="
            f"{diagnostics.mean_effective_neighbours[regime_idx].item():.3f} | "
            "row-sum range="
            f"[{diagnostics.row_sum_min[regime_idx].item():.6f}, "
            f"{diagnostics.row_sum_max[regime_idx].item():.6f}]"
        )

    lines.extend(
        [
            "",
            "Pairwise support Jaccard:",
            str(
                diagnostics
                .pairwise_support_jaccard
                .numpy()
            ),
            "",
            "Pairwise weighted graph correlation:",
            str(
                diagnostics
                .pairwise_weight_correlation
                .numpy()
            ),
            "",
            "Pairwise off-diagonal RMSE:",
            str(
                diagnostics
                .pairwise_weight_rmse
                .numpy()
            ),
        ]
    )

    return "\n".join(lines)


def _cpu_smoke_test() -> None:
    """Run deterministic structural checks without a GPU."""
    config = GroundTruthGraphConfig(
        num_nodes=16,
        regime_names=DEFAULT_REGIME_NAMES,
        total_in_degrees=DEFAULT_TOTAL_IN_DEGREES,
        shared_in_degree=2,
    )

    generator = GroundTruthGraphGenerator(
        config,
        device="cpu",
    )

    graph_set = generator.generate(
        seed=7
    )

    if tuple(graph_set.graphs.shape) != (
        3,
        16,
        16,
    ):
        raise AssertionError(
            "Unexpected graph tensor shape."
        )

    if torch.any(graph_set.graphs < 0):
        raise AssertionError(
            "Ground-truth graphs contain negative weights."
        )

    diagonal = torch.diagonal(
        graph_set.graphs,
        dim1=-2,
        dim2=-1,
    )

    if not torch.equal(
        diagonal,
        torch.zeros_like(diagonal),
    ):
        raise AssertionError(
            "Ground-truth graphs contain self-loops."
        )

    if not torch.allclose(
        graph_set.graphs.sum(dim=-1),
        torch.ones(
            (3, 16),
            dtype=graph_set.graphs.dtype,
        ),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Ground-truth graph rows are not stochastic."
        )

    expected_degrees = torch.tensor(
        DEFAULT_TOTAL_IN_DEGREES,
        dtype=torch.long,
    ).unsqueeze(1).expand(-1, 16)

    actual_degrees = (
        graph_set.support_masks.sum(dim=-1)
    )

    if not torch.equal(
        actual_degrees.cpu(),
        expected_degrees,
    ):
        raise AssertionError(
            "Ground-truth graph in-degrees do not match the "
            "configuration."
        )

    expected_shared = torch.full(
        (16,),
        fill_value=2,
        dtype=torch.long,
    )

    if not torch.equal(
        graph_set.shared_support.sum(dim=-1).cpu(),
        expected_shared,
    ):
        raise AssertionError(
            "Shared backbone in-degree is incorrect."
        )

    for left in range(3):
        for right in range(left + 1, 3):
            overlap = (
                graph_set.regime_specific_support[left]
                & graph_set.regime_specific_support[right]
            )

            if torch.any(overlap):
                raise AssertionError(
                    "Regime-specific supports are not disjoint."
                )

    repeated = generator.generate(
        seed=7
    )

    if not torch.equal(
        graph_set.support_masks,
        repeated.support_masks,
    ):
        raise AssertionError(
            "Graph support is not deterministic under a fixed seed."
        )

    if not torch.equal(
        graph_set.graphs,
        repeated.graphs,
    ):
        raise AssertionError(
            "Graph weights are not deterministic under a fixed seed."
        )

    different = generator.generate(
        seed=8
    )

    if torch.equal(
        graph_set.graphs,
        different.graphs,
    ):
        raise AssertionError(
            "Different seeds produced identical graph sets."
        )

    selected_ids = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 0],
        ],
        dtype=torch.long,
    )

    selected = graph_set.select(
        selected_ids
    )

    if tuple(selected.shape) != (
        2,
        3,
        16,
        16,
    ):
        raise AssertionError(
            "Active-graph selection returned the wrong shape."
        )

    if not torch.equal(
        selected[0, 0],
        graph_set.graphs[0],
    ):
        raise AssertionError(
            "Active-graph selection used the wrong regime."
        )

    diagnostics = generator.diagnostics(
        graph_set
    )

    print(
        format_graph_diagnostics(
            diagnostics,
            graph_set.regime_names,
        )
    )
    print()
    print(
        "GroundTruthGraphGenerator CPU smoke test passed."
    )


if __name__ == "__main__":
    _cpu_smoke_test()

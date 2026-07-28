from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Literal, Sequence

import torch
from torch import Tensor


DEFAULT_REGIME_NAMES = (
    "bear",
    "neutral",
    "bull",
)

# Row = current regime, column = next regime.
#
# These are the rounded probabilities supplied in the three-state
# regime table. They are row-normalised once by BlockMarkovRegimeProcess
# before use because the printed rows sum to 0.999 or 1.001.
DEFAULT_TRANSITION_MATRIX = (
    (0.237, 0.027, 0.735),
    (0.046, 0.914, 0.041),
    (0.001, 0.145, 0.855),
)


InitialDistribution = (
    Literal["uniform", "stationary"]
    | tuple[float, ...]
)


@dataclass(frozen=True)
class RegimeProcessConfig:
    """Configuration for a blockwise hidden Markov regime process.

    One Markov transition is sampled between blocks. The active regime
    is held fixed for every timestep inside a block.

    The default ``block_length=120`` aligns one regime with a complete
    60-step model context plus the following 60-step target interval.
    It is deliberately configurable so later experiments can change the
    regime timescale without changing the implementation.
    """

    regime_names: tuple[str, ...] = DEFAULT_REGIME_NAMES
    transition_matrix: tuple[
        tuple[float, ...],
        ...,
    ] = DEFAULT_TRANSITION_MATRIX
    block_length: int = 120
    initial_distribution: InitialDistribution = "uniform"


@dataclass(frozen=True)
class RegimeSample:
    """Sampled hidden-regime paths.

    Attributes:
        block_regime_ids:
            Integer regime IDs with shape
            ``[num_trajectories, num_blocks]``.

        regime_ids:
            Regime IDs expanded to every timestep, with shape
            ``[num_trajectories, total_steps]``.

        block_start_indices:
            Start timestep of each regime block, shape ``[num_blocks]``.

        block_length:
            Number of timesteps represented by one block.

        total_steps:
            Number of returned timestep-level regime IDs.
    """

    block_regime_ids: Tensor
    regime_ids: Tensor
    block_start_indices: Tensor
    block_length: int
    total_steps: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary suitable for ``torch.save``."""
        return {
            "block_regime_ids": self.block_regime_ids,
            "regime_ids": self.regime_ids,
            "block_start_indices": self.block_start_indices,
            "block_length": int(self.block_length),
            "total_steps": int(self.total_steps),
        }


@dataclass(frozen=True)
class RegimeDiagnostics:
    """Empirical and theoretical diagnostics for sampled block paths."""

    transition_counts: Tensor
    empirical_transition_matrix: Tensor
    configured_transition_matrix: Tensor
    block_regime_frequency: Tensor
    stationary_distribution: Tensor
    run_counts: Tensor
    empirical_mean_dwell_blocks: Tensor
    empirical_mean_dwell_steps: Tensor
    theoretical_mean_dwell_blocks: Tensor
    theoretical_mean_dwell_steps: Tensor

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable diagnostics dictionary."""
        return {
            "transition_counts": self.transition_counts,
            "empirical_transition_matrix": (
                self.empirical_transition_matrix
            ),
            "configured_transition_matrix": (
                self.configured_transition_matrix
            ),
            "block_regime_frequency": self.block_regime_frequency,
            "stationary_distribution": self.stationary_distribution,
            "run_counts": self.run_counts,
            "empirical_mean_dwell_blocks": (
                self.empirical_mean_dwell_blocks
            ),
            "empirical_mean_dwell_steps": (
                self.empirical_mean_dwell_steps
            ),
            "theoretical_mean_dwell_blocks": (
                self.theoretical_mean_dwell_blocks
            ),
            "theoretical_mean_dwell_steps": (
                self.theoretical_mean_dwell_steps
            ),
        }


class BlockMarkovRegimeProcess:
    """Three-state blockwise Markov process for hidden graph regimes.

    The class is independent of Kronos and independent of the learned
    graph model. Its only role is to generate the hidden regime path
    that will later select one of the known ground-truth graphs.

    Matrix convention:
        ``transition_matrix[current_regime, next_regime]``.

    Default regime IDs:
        ``0 = bear``, ``1 = neutral``, ``2 = bull``.
    """

    def __init__(
        self,
        config: RegimeProcessConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = (
            RegimeProcessConfig()
            if config is None
            else config
        )
        self.device = torch.device(device)

        self._validate_config()

        self.regime_names = tuple(
            self.config.regime_names
        )
        self.num_regimes = len(
            self.regime_names
        )
        self.block_length = int(
            self.config.block_length
        )

        raw_transition = torch.tensor(
            self.config.transition_matrix,
            dtype=torch.float64,
            device=self.device,
        )

        self.raw_transition_matrix = (
            raw_transition.clone()
        )

        row_sums = raw_transition.sum(
            dim=1,
            keepdim=True,
        )

        self.transition_matrix = (
            raw_transition
            / row_sums
        )

        self.stationary_distribution = (
            self._compute_stationary_distribution()
        )

        self.initial_distribution = (
            self._resolve_initial_distribution()
        )

    def _validate_config(self) -> None:
        names = tuple(
            self.config.regime_names
        )

        if len(names) < 2:
            raise ValueError(
                "At least two regimes are required."
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

        if (
            isinstance(self.config.block_length, bool)
            or not isinstance(
                self.config.block_length,
                int,
            )
            or self.config.block_length <= 0
        ):
            raise ValueError(
                "block_length must be a positive integer."
            )

        matrix = self.config.transition_matrix
        num_regimes = len(names)

        if len(matrix) != num_regimes:
            raise ValueError(
                "transition_matrix row count must match "
                "regime_names."
            )

        if any(
            len(row) != num_regimes
            for row in matrix
        ):
            raise ValueError(
                "transition_matrix must be square with one "
                "row/column per regime."
            )

        matrix_tensor = torch.tensor(
            matrix,
            dtype=torch.float64,
        )

        if not torch.isfinite(
            matrix_tensor
        ).all():
            raise ValueError(
                "transition_matrix contains non-finite values."
            )

        if torch.any(matrix_tensor < 0):
            raise ValueError(
                "transition_matrix cannot contain negative "
                "probabilities."
            )

        if torch.any(
            matrix_tensor.sum(dim=1) <= 0
        ):
            raise ValueError(
                "Every transition row must have positive mass."
            )

        initial = self.config.initial_distribution

        if isinstance(initial, str):
            if initial not in {
                "uniform",
                "stationary",
            }:
                raise ValueError(
                    "initial_distribution must be 'uniform', "
                    "'stationary', or an explicit probability "
                    "tuple."
                )
        else:
            values = tuple(
                float(value)
                for value in initial
            )

            if len(values) != num_regimes:
                raise ValueError(
                    "Explicit initial_distribution length must "
                    "match the number of regimes."
                )

            initial_tensor = torch.tensor(
                values,
                dtype=torch.float64,
            )

            if (
                not torch.isfinite(
                    initial_tensor
                ).all()
                or torch.any(initial_tensor < 0)
                or initial_tensor.sum() <= 0
            ):
                raise ValueError(
                    "Explicit initial_distribution must contain "
                    "finite non-negative values with positive "
                    "total mass."
                )

    def _compute_stationary_distribution(
        self,
        *,
        tolerance: float = 1.0e-14,
        max_iterations: int = 100_000,
    ) -> Tensor:
        """Compute the stationary block distribution by power iteration."""
        probability = torch.full(
            (len(self.config.regime_names),),
            fill_value=(
                1.0
                / len(self.config.regime_names)
            ),
            dtype=torch.float64,
            device=self.device,
        )

        for _ in range(max_iterations):
            updated = (
                probability
                @ self.transition_matrix
            )

            if torch.max(
                torch.abs(
                    updated - probability
                )
            ).item() < tolerance:
                probability = updated
                break

            probability = updated
        else:
            raise RuntimeError(
                "Stationary-distribution power iteration did "
                "not converge."
            )

        return (
            probability
            / probability.sum()
        )

    def _resolve_initial_distribution(
        self,
    ) -> Tensor:
        initial = self.config.initial_distribution

        if initial == "uniform":
            probability = torch.ones(
                self.num_regimes,
                dtype=torch.float64,
                device=self.device,
            )
        elif initial == "stationary":
            probability = (
                self.stationary_distribution.clone()
            )
        else:
            probability = torch.tensor(
                initial,
                dtype=torch.float64,
                device=self.device,
            )

        return (
            probability
            / probability.sum()
        )

    def _make_generator(
        self,
        seed: int | None,
    ) -> torch.Generator:
        generator = torch.Generator(
            device=self.device
        )

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(
                int(seed)
            )

        return generator

    def sample_blocks(
        self,
        *,
        num_trajectories: int,
        num_blocks: int,
        seed: int | None = None,
    ) -> Tensor:
        """Sample block-level regime IDs.

        Returns:
            Integer tensor with shape
            ``[num_trajectories, num_blocks]``.
        """
        if (
            isinstance(num_trajectories, bool)
            or not isinstance(
                num_trajectories,
                int,
            )
            or num_trajectories <= 0
        ):
            raise ValueError(
                "num_trajectories must be a positive integer."
            )

        if (
            isinstance(num_blocks, bool)
            or not isinstance(num_blocks, int)
            or num_blocks <= 0
        ):
            raise ValueError(
                "num_blocks must be a positive integer."
            )

        generator = self._make_generator(
            seed
        )

        block_ids = torch.empty(
            (
                num_trajectories,
                num_blocks,
            ),
            dtype=torch.long,
            device=self.device,
        )

        initial_probabilities = (
            self.initial_distribution
            .expand(
                num_trajectories,
                -1,
            )
        )

        block_ids[:, 0] = torch.multinomial(
            initial_probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)

        for block_idx in range(
            1,
            num_blocks,
        ):
            current = block_ids[
                :,
                block_idx - 1,
            ]

            next_probabilities = (
                self.transition_matrix[
                    current
                ]
            )

            block_ids[:, block_idx] = (
                torch.multinomial(
                    next_probabilities,
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                )
                .squeeze(1)
            )

        return block_ids

    def expand_blocks(
        self,
        block_regime_ids: Tensor,
        *,
        total_steps: int | None = None,
    ) -> Tensor:
        """Expand each block ID over ``block_length`` timesteps."""
        if (
            block_regime_ids.ndim != 2
            or block_regime_ids.dtype
            not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.long,
            }
        ):
            raise ValueError(
                "block_regime_ids must be an integer tensor "
                "with shape [num_trajectories, num_blocks]."
            )

        if (
            torch.any(block_regime_ids < 0)
            or torch.any(
                block_regime_ids
                >= self.num_regimes
            )
        ):
            raise ValueError(
                "block_regime_ids contains an unknown regime ID."
            )

        maximum_steps = (
            block_regime_ids.shape[1]
            * self.block_length
        )

        resolved_steps = (
            maximum_steps
            if total_steps is None
            else int(total_steps)
        )

        if not 1 <= resolved_steps <= maximum_steps:
            raise ValueError(
                "total_steps must lie between 1 and the number "
                "of timesteps represented by the supplied blocks."
            )

        return (
            block_regime_ids
            .repeat_interleave(
                self.block_length,
                dim=1,
            )
            [:, :resolved_steps]
            .contiguous()
        )

    def sample(
        self,
        *,
        num_trajectories: int,
        total_steps: int,
        seed: int | None = None,
    ) -> RegimeSample:
        """Sample block-level and timestep-level regime paths."""
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
        ):
            raise ValueError(
                "total_steps must be a positive integer."
            )

        num_blocks = int(
            ceil(
                total_steps
                / self.block_length
            )
        )

        block_regime_ids = self.sample_blocks(
            num_trajectories=num_trajectories,
            num_blocks=num_blocks,
            seed=seed,
        )

        regime_ids = self.expand_blocks(
            block_regime_ids,
            total_steps=total_steps,
        )

        block_start_indices = torch.arange(
            num_blocks,
            dtype=torch.long,
            device=self.device,
        ) * self.block_length

        return RegimeSample(
            block_regime_ids=(
                block_regime_ids
            ),
            regime_ids=regime_ids,
            block_start_indices=(
                block_start_indices
            ),
            block_length=self.block_length,
            total_steps=int(total_steps),
        )

    def diagnostics(
        self,
        block_regime_ids: Tensor,
    ) -> RegimeDiagnostics:
        """Summarise transition frequencies and regime dwell times."""
        if block_regime_ids.ndim != 2:
            raise ValueError(
                "block_regime_ids must have shape "
                "[num_trajectories, num_blocks]."
            )

        block_regime_ids = (
            block_regime_ids
            .detach()
            .to(
                device="cpu",
                dtype=torch.long,
            )
        )

        if (
            torch.any(block_regime_ids < 0)
            or torch.any(
                block_regime_ids
                >= self.num_regimes
            )
        ):
            raise ValueError(
                "block_regime_ids contains an unknown regime ID."
            )

        flattened = block_regime_ids.reshape(
            -1
        )

        block_counts = torch.bincount(
            flattened,
            minlength=self.num_regimes,
        ).to(torch.float64)

        block_frequency = (
            block_counts
            / block_counts.sum()
        )

        if block_regime_ids.shape[1] > 1:
            current = block_regime_ids[
                :,
                :-1,
            ].reshape(-1)

            following = block_regime_ids[
                :,
                1:,
            ].reshape(-1)

            transition_codes = (
                current * self.num_regimes
                + following
            )

            transition_counts = torch.bincount(
                transition_codes,
                minlength=(
                    self.num_regimes
                    * self.num_regimes
                ),
            ).reshape(
                self.num_regimes,
                self.num_regimes,
            )
        else:
            transition_counts = torch.zeros(
                (
                    self.num_regimes,
                    self.num_regimes,
                ),
                dtype=torch.long,
            )

        row_totals = transition_counts.sum(
            dim=1,
            keepdim=True,
        )

        empirical_transition = torch.where(
            row_totals > 0,
            transition_counts.to(
                torch.float64
            )
            / row_totals.clamp_min(1),
            torch.full(
                (
                    self.num_regimes,
                    self.num_regimes,
                ),
                fill_value=float("nan"),
                dtype=torch.float64,
            ),
        )

        dwell_lengths: list[list[int]] = [
            []
            for _ in range(
                self.num_regimes
            )
        ]

        for trajectory in block_regime_ids:
            unique_ids, run_lengths = (
                torch.unique_consecutive(
                    trajectory,
                    return_counts=True,
                )
            )

            for regime_id, run_length in zip(
                unique_ids.tolist(),
                run_lengths.tolist(),
            ):
                dwell_lengths[regime_id].append(
                    int(run_length)
                )

        run_counts = torch.tensor(
            [
                len(values)
                for values in dwell_lengths
            ],
            dtype=torch.long,
        )

        empirical_mean_dwell_blocks = torch.tensor(
            [
                (
                    sum(values) / len(values)
                    if values
                    else float("nan")
                )
                for values in dwell_lengths
            ],
            dtype=torch.float64,
        )

        configured = (
            self.transition_matrix
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        diagonal = torch.diagonal(
            configured
        )

        theoretical_mean_dwell_blocks = (
            1.0
            / (1.0 - diagonal)
        )

        return RegimeDiagnostics(
            transition_counts=(
                transition_counts
            ),
            empirical_transition_matrix=(
                empirical_transition
            ),
            configured_transition_matrix=(
                configured
            ),
            block_regime_frequency=(
                block_frequency
            ),
            stationary_distribution=(
                self.stationary_distribution
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
            ),
            run_counts=run_counts,
            empirical_mean_dwell_blocks=(
                empirical_mean_dwell_blocks
            ),
            empirical_mean_dwell_steps=(
                empirical_mean_dwell_blocks
                * self.block_length
            ),
            theoretical_mean_dwell_blocks=(
                theoretical_mean_dwell_blocks
            ),
            theoretical_mean_dwell_steps=(
                theoretical_mean_dwell_blocks
                * self.block_length
            ),
        )

    def metadata(self) -> dict[str, Any]:
        """Return the immutable process specification for cache metadata."""
        return {
            "kind": "block_markov_regime_process",
            "config": asdict(self.config),
            "regime_names": list(
                self.regime_names
            ),
            "regime_to_id": {
                name: idx
                for idx, name in enumerate(
                    self.regime_names
                )
            },
            "transition_convention": (
                "row=current_regime,column=next_regime"
            ),
            "raw_transition_matrix": (
                self.raw_transition_matrix
                .detach()
                .cpu()
            ),
            "transition_matrix": (
                self.transition_matrix
                .detach()
                .cpu()
            ),
            "initial_distribution": (
                self.initial_distribution
                .detach()
                .cpu()
            ),
            "stationary_distribution": (
                self.stationary_distribution
                .detach()
                .cpu()
            ),
        }


def format_regime_diagnostics(
    diagnostics: RegimeDiagnostics,
    regime_names: Sequence[str],
) -> str:
    """Format diagnostics as a compact human-readable report."""
    names = tuple(regime_names)

    if len(names) != diagnostics.transition_counts.shape[0]:
        raise ValueError(
            "regime_names does not match the diagnostics."
        )

    lines = [
        "Configured transition matrix "
        "(row=current, column=next):",
        str(
            diagnostics
            .configured_transition_matrix
            .numpy()
        ),
        "",
        "Empirical transition matrix:",
        str(
            diagnostics
            .empirical_transition_matrix
            .numpy()
        ),
        "",
        "Regime summary:",
    ]

    for idx, name in enumerate(names):
        lines.append(
            "  "
            f"{name:>7} | "
            "block frequency="
            f"{diagnostics.block_regime_frequency[idx].item():.4f} | "
            "stationary="
            f"{diagnostics.stationary_distribution[idx].item():.4f} | "
            "mean dwell="
            f"{diagnostics.empirical_mean_dwell_blocks[idx].item():.3f} "
            "blocks / "
            f"{diagnostics.empirical_mean_dwell_steps[idx].item():.1f} "
            "steps | theoretical="
            f"{diagnostics.theoretical_mean_dwell_blocks[idx].item():.3f} "
            "blocks"
        )

    return "\n".join(lines)


def _cpu_smoke_test() -> None:
    """Run deterministic CPU checks without requiring Kronos or a GPU."""
    config = RegimeProcessConfig(
        block_length=120,
        initial_distribution="uniform",
    )

    process = BlockMarkovRegimeProcess(
        config,
        device="cpu",
    )

    # A large block-level sample is cheap because it is not expanded to
    # minute-level IDs. It gives a stable transition-frequency check.
    block_ids = process.sample_blocks(
        num_trajectories=4096,
        num_blocks=100,
        seed=7,
    )

    diagnostics = process.diagnostics(
        block_ids
    )

    finite_rows = torch.isfinite(
        diagnostics.empirical_transition_matrix
    ).all(dim=1)

    maximum_transition_error = torch.max(
        torch.abs(
            diagnostics.empirical_transition_matrix[
                finite_rows
            ]
            - diagnostics.configured_transition_matrix[
                finite_rows
            ]
        )
    ).item()

    if maximum_transition_error > 0.03:
        raise AssertionError(
            "Empirical transition matrix is unexpectedly far "
            "from the configured matrix: "
            f"max error={maximum_transition_error:.4f}."
        )

    sample = process.sample(
        num_trajectories=4,
        total_steps=365,
        seed=11,
    )

    if tuple(sample.regime_ids.shape) != (
        4,
        365,
    ):
        raise AssertionError(
            "Unexpected expanded regime shape."
        )

    reconstructed = process.expand_blocks(
        sample.block_regime_ids,
        total_steps=sample.total_steps,
    )

    if not torch.equal(
        sample.regime_ids,
        reconstructed,
    ):
        raise AssertionError(
            "Block expansion is not deterministic."
        )

    if not torch.allclose(
        process.transition_matrix.sum(dim=1),
        torch.ones(
            process.num_regimes,
            dtype=torch.float64,
        ),
    ):
        raise AssertionError(
            "Normalised transition rows do not sum to one."
        )

    print(
        format_regime_diagnostics(
            diagnostics,
            process.regime_names,
        )
    )
    print()
    print(
        "BlockMarkovRegimeProcess CPU smoke test passed."
    )


if __name__ == "__main__":
    _cpu_smoke_test()

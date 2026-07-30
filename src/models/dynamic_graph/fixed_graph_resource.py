from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from src.data.load_candle_data import compute_log_returns

from .graph_learners import (
    EmptyCorrelationRowPolicy,
    build_absolute_correlation_adjacency,
)


FixedGraphResourceType = Literal[
    "none",
    "absolute_return_correlation",
]

_RESOURCE_FORMAT = "absolute_return_correlation_fixed_graph_v1"


@dataclass(frozen=True)
class FixedGraphResourceConfig:
    """Configuration for a deterministic graph fitted outside the model.

    The current production resource is an absolute contemporaneous
    one-minute return-correlation graph fitted from the cleaned training
    split only. Keeping this separate from ``GraphConfig`` allows the same
    saved resource to be reused by a fixed graph or as a dynamic-base graph's
    stable base.
    """

    type: FixedGraphResourceType = "none"
    channel: str = "close"
    threshold: float = 0.18
    empty_row_policy: EmptyCorrelationRowPolicy = "error"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "FixedGraphResourceConfig":
        if values is None:
            return cls()
        return cls(**dict(values))

    @property
    def enabled(self) -> bool:
        return self.type != "none"

    def validate(
        self,
        *,
        graph_type: str,
        data_mode: str,
    ) -> None:
        if self.type not in {
            "none",
            "absolute_return_correlation",
        }:
            raise ValueError(
                f"Unsupported graph_resource.type {self.type!r}."
            )

        if self.empty_row_policy not in {
            "error",
            "strongest",
        }:
            raise ValueError(
                "graph_resource.empty_row_policy must be 'error' or "
                "'strongest'."
            )

        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError(
                "graph_resource.threshold must lie in [0, 1]."
            )

        if not str(self.channel).strip():
            raise ValueError(
                "graph_resource.channel must be non-empty."
            )

        if self.enabled and data_mode != "real":
            raise ValueError(
                "The absolute-return correlation resource is available "
                "only for real-data runs."
            )

        if self.enabled and graph_type not in {
            "fixed",
            "dynamic_base",
        }:
            raise ValueError(
                "A fixed graph resource may be used only with "
                "graph.type='fixed' or 'dynamic_base'."
            )

        if graph_type == "fixed" and not self.enabled:
            raise ValueError(
                "graph.type='fixed' requires graph_resource.type="
                "'absolute_return_correlation' in the production runner."
            )


@dataclass(frozen=True)
class FixedGraphResource:
    """Training-only fitted graph and its complete provenance."""

    format: str
    resource_type: str
    fitted_split: str
    channel: str
    threshold: float
    empty_row_policy: str
    add_self_loops: bool
    asset_cols: tuple[str, ...]
    session_days: tuple[str, ...]
    session_count: int
    return_observation_count: int
    correlation_matrix: Tensor
    adjacency: Tensor
    row_entropy: Tensor
    retained_neighbours: Tensor
    mean_row_entropy: float
    mean_effective_neighbours: float
    resource_hash: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "resource_type": self.resource_type,
            "fitted_split": self.fitted_split,
            "channel": self.channel,
            "threshold": self.threshold,
            "empty_row_policy": self.empty_row_policy,
            "add_self_loops": self.add_self_loops,
            "asset_cols": list(self.asset_cols),
            "session_days": list(self.session_days),
            "session_count": self.session_count,
            "return_observation_count": self.return_observation_count,
            "correlation_matrix": self.correlation_matrix,
            "adjacency": self.adjacency,
            "row_entropy": self.row_entropy,
            "retained_neighbours": self.retained_neighbours,
            "mean_row_entropy": self.mean_row_entropy,
            "mean_effective_neighbours": (
                self.mean_effective_neighbours
            ),
            "resource_hash": self.resource_hash,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "resource_type": self.resource_type,
            "fitted_split": self.fitted_split,
            "channel": self.channel,
            "threshold": self.threshold,
            "empty_row_policy": self.empty_row_policy,
            "add_self_loops": self.add_self_loops,
            "asset_cols": list(self.asset_cols),
            "session_days": list(self.session_days),
            "session_count": self.session_count,
            "return_observation_count": self.return_observation_count,
            "adjacency_shape": list(self.adjacency.shape),
            "correlation_shape": list(
                self.correlation_matrix.shape
            ),
            "mean_row_entropy": self.mean_row_entropy,
            "mean_effective_neighbours": (
                self.mean_effective_neighbours
            ),
            "mean_retained_neighbours": float(
                self.retained_neighbours
                .to(torch.float64)
                .mean()
                .item()
            ),
            "min_retained_neighbours": int(
                self.retained_neighbours.min().item()
            ),
            "max_retained_neighbours": int(
                self.retained_neighbours.max().item()
            ),
            "resource_hash": self.resource_hash,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "FixedGraphResource":
        resource = cls(
            format=str(payload["format"]),
            resource_type=str(payload["resource_type"]),
            fitted_split=str(payload["fitted_split"]),
            channel=str(payload["channel"]),
            threshold=float(payload["threshold"]),
            empty_row_policy=str(payload["empty_row_policy"]),
            add_self_loops=bool(payload["add_self_loops"]),
            asset_cols=tuple(str(x) for x in payload["asset_cols"]),
            session_days=tuple(
                str(x) for x in payload["session_days"]
            ),
            session_count=int(payload["session_count"]),
            return_observation_count=int(
                payload["return_observation_count"]
            ),
            correlation_matrix=torch.as_tensor(
                payload["correlation_matrix"],
                dtype=torch.float64,
            ).contiguous(),
            adjacency=torch.as_tensor(
                payload["adjacency"],
                dtype=torch.float32,
            ).contiguous(),
            row_entropy=torch.as_tensor(
                payload["row_entropy"],
                dtype=torch.float64,
            ).contiguous(),
            retained_neighbours=torch.as_tensor(
                payload["retained_neighbours"],
                dtype=torch.long,
            ).contiguous(),
            mean_row_entropy=float(payload["mean_row_entropy"]),
            mean_effective_neighbours=float(
                payload["mean_effective_neighbours"]
            ),
            resource_hash=str(payload["resource_hash"]),
        )
        resource.validate()
        return resource

    def validate(self) -> None:
        if self.format != _RESOURCE_FORMAT:
            raise ValueError(
                f"Unsupported fixed graph resource format {self.format!r}."
            )

        if self.resource_type != "absolute_return_correlation":
            raise ValueError(
                "Unsupported fixed graph resource type "
                f"{self.resource_type!r}."
            )

        if self.fitted_split != "train":
            raise ValueError(
                "A production fixed graph must be fitted from the "
                "training split only."
            )

        num_nodes = len(self.asset_cols)
        expected_square = (num_nodes, num_nodes)

        if tuple(self.correlation_matrix.shape) != expected_square:
            raise ValueError(
                "Fixed graph correlation matrix has the wrong shape."
            )

        if tuple(self.adjacency.shape) != expected_square:
            raise ValueError(
                "Fixed graph adjacency has the wrong shape."
            )

        if tuple(self.row_entropy.shape) != (num_nodes,):
            raise ValueError(
                "Fixed graph row entropy has the wrong shape."
            )

        if tuple(self.retained_neighbours.shape) != (num_nodes,):
            raise ValueError(
                "Fixed graph retained-neighbour vector has the wrong shape."
            )

        if not torch.isfinite(self.correlation_matrix).all():
            raise ValueError(
                "Fixed graph correlation matrix contains non-finite values."
            )

        if not torch.isfinite(self.adjacency).all():
            raise ValueError(
                "Fixed graph adjacency contains non-finite values."
            )

        if torch.any(self.adjacency < 0):
            raise ValueError(
                "Fixed graph adjacency contains negative values."
            )

        if not torch.allclose(
            self.adjacency.sum(dim=-1),
            torch.ones(num_nodes, dtype=self.adjacency.dtype),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise ValueError(
                "Fixed graph adjacency rows do not sum to one."
            )

        if not self.add_self_loops:
            diagonal = torch.diagonal(self.adjacency)
            if not torch.equal(
                diagonal,
                torch.zeros_like(diagonal),
            ):
                raise ValueError(
                    "Fixed graph adjacency diagonal is not zero."
                )

        observed_hash = _resource_hash(
            metadata=_hash_metadata(
                resource_type=self.resource_type,
                fitted_split=self.fitted_split,
                channel=self.channel,
                threshold=self.threshold,
                empty_row_policy=self.empty_row_policy,
                add_self_loops=self.add_self_loops,
                asset_cols=self.asset_cols,
                session_days=self.session_days,
                session_count=self.session_count,
                return_observation_count=(
                    self.return_observation_count
                ),
            ),
            correlation=self.correlation_matrix,
            adjacency=self.adjacency,
        )

        if observed_hash != self.resource_hash:
            raise ValueError(
                "Fixed graph resource hash does not match its contents."
            )

    def validate_against(
        self,
        *,
        config: FixedGraphResourceConfig,
        expected_asset_cols: Sequence[str],
        add_self_loops: bool,
    ) -> None:
        self.validate()

        if tuple(expected_asset_cols) != self.asset_cols:
            raise ValueError(
                "Fixed graph resource asset order differs from the token "
                "cache asset order."
            )

        if self.channel != config.channel:
            raise ValueError(
                "Fixed graph resource channel differs from config."
            )

        if not np.isclose(
            self.threshold,
            float(config.threshold),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Fixed graph resource threshold differs from config."
            )

        if self.empty_row_policy != config.empty_row_policy:
            raise ValueError(
                "Fixed graph resource empty-row policy differs from config."
            )

        if self.add_self_loops != bool(add_self_loops):
            raise ValueError(
                "Fixed graph resource self-loop policy differs from config."
            )


def _hash_metadata(
    *,
    resource_type: str,
    fitted_split: str,
    channel: str,
    threshold: float,
    empty_row_policy: str,
    add_self_loops: bool,
    asset_cols: Sequence[str],
    session_days: Sequence[str],
    session_count: int,
    return_observation_count: int,
) -> dict[str, Any]:
    return {
        "format": _RESOURCE_FORMAT,
        "resource_type": resource_type,
        "fitted_split": fitted_split,
        "channel": channel,
        "threshold": float(threshold),
        "empty_row_policy": empty_row_policy,
        "add_self_loops": bool(add_self_loops),
        "asset_cols": list(asset_cols),
        "session_days": list(session_days),
        "session_count": int(session_count),
        "return_observation_count": int(return_observation_count),
    }


def _resource_hash(
    *,
    metadata: Mapping[str, Any],
    correlation: Tensor,
    adjacency: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    for values in (correlation, adjacency):
        array = (
            values.detach()
            .cpu()
            .contiguous()
            .numpy()
        )
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))

    return digest.hexdigest()


def fit_absolute_return_correlation_resource(
    train_split: Mapping[str, Any],
    *,
    config: FixedGraphResourceConfig,
    expected_asset_cols: Sequence[str],
    add_self_loops: bool,
) -> FixedGraphResource:
    """Fit the deterministic graph from cleaned training sessions only."""
    if config.type != "absolute_return_correlation":
        raise ValueError(
            "fit_absolute_return_correlation_resource requires "
            "graph_resource.type='absolute_return_correlation'."
        )

    asset_cols = tuple(
        str(value) for value in train_split["asset_cols"]
    )
    expected_assets = tuple(str(value) for value in expected_asset_cols)

    if asset_cols != expected_assets:
        raise ValueError(
            "Raw training split and token cache asset order differ."
        )

    channels = tuple(str(value) for value in train_split["channels"])
    if config.channel not in channels:
        raise ValueError(
            f"Correlation channel {config.channel!r} is not present in "
            f"training channels {list(channels)}."
        )

    samples = train_split.get("samples")
    if not isinstance(samples, Sequence) or not samples:
        raise ValueError(
            "Training split contains no sessions."
        )

    returns_parts: list[Tensor] = []
    session_days: list[str] = []

    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Sequence) or len(sample) < 3:
            raise ValueError(
                f"Training sample {sample_index} has an invalid format."
            )

        x = torch.as_tensor(sample[0])
        if x.ndim != 3:
            raise ValueError(
                f"Training sample {sample_index} must have shape [T,N,D]."
            )

        if int(x.shape[1]) != len(asset_cols):
            raise ValueError(
                f"Training sample {sample_index} has the wrong asset axis."
            )

        values = compute_log_returns(
            x=x,
            split=dict(train_split),
            channels=[config.channel],
        ).to(torch.float64)

        if not torch.isfinite(values).all():
            raise ValueError(
                f"Training sample {sample_index} produced non-finite "
                "log returns."
            )

        returns_parts.append(values)
        session_days.append(str(sample[2]))

    returns = torch.cat(returns_parts, dim=0)
    if returns.ndim != 2:
        raise AssertionError(
            "Concatenated returns do not have shape [observations, assets]."
        )

    centred = returns - returns.mean(dim=0, keepdim=True)
    sum_squares = centred.square().sum(dim=0)
    zero_variance = sum_squares <= torch.finfo(torch.float64).eps

    if torch.any(zero_variance):
        names = [
            asset_cols[index]
            for index in torch.nonzero(
                zero_variance,
                as_tuple=False,
            ).flatten().tolist()
        ]
        raise ValueError(
            "Cannot fit a correlation graph because these training assets "
            f"have zero return variance: {names}."
        )

    covariance_numerator = centred.transpose(0, 1) @ centred
    denominator = torch.sqrt(
        sum_squares[:, None] * sum_squares[None, :]
    )
    correlation = (
        covariance_numerator / denominator
    ).clamp(-1.0, 1.0)
    correlation = 0.5 * (
        correlation + correlation.transpose(0, 1)
    )
    correlation.fill_diagonal_(1.0)

    expanded = build_absolute_correlation_adjacency(
        correlation,
        threshold=float(config.threshold),
        num_heads=1,
        add_self_loops=bool(add_self_loops),
        empty_row_policy=config.empty_row_policy,
    )
    adjacency = expanded[0, 0].contiguous()

    positive = adjacency > 0
    row_entropy = -torch.where(
        positive,
        adjacency.to(torch.float64)
        * torch.log(
            adjacency.to(torch.float64).clamp_min(1.0e-12)
        ),
        torch.zeros_like(adjacency, dtype=torch.float64),
    ).sum(dim=-1)
    retained_neighbours = positive.sum(dim=-1).to(torch.long)

    metadata = _hash_metadata(
        resource_type="absolute_return_correlation",
        fitted_split="train",
        channel=config.channel,
        threshold=float(config.threshold),
        empty_row_policy=config.empty_row_policy,
        add_self_loops=bool(add_self_loops),
        asset_cols=asset_cols,
        session_days=session_days,
        session_count=len(samples),
        return_observation_count=int(returns.shape[0]),
    )
    resource_hash = _resource_hash(
        metadata=metadata,
        correlation=correlation,
        adjacency=adjacency,
    )

    resource = FixedGraphResource(
        format=_RESOURCE_FORMAT,
        resource_type="absolute_return_correlation",
        fitted_split="train",
        channel=config.channel,
        threshold=float(config.threshold),
        empty_row_policy=config.empty_row_policy,
        add_self_loops=bool(add_self_loops),
        asset_cols=asset_cols,
        session_days=tuple(session_days),
        session_count=len(samples),
        return_observation_count=int(returns.shape[0]),
        correlation_matrix=correlation.contiguous(),
        adjacency=adjacency,
        row_entropy=row_entropy.contiguous(),
        retained_neighbours=retained_neighbours.contiguous(),
        mean_row_entropy=float(row_entropy.mean().item()),
        mean_effective_neighbours=float(
            row_entropy.exp().mean().item()
        ),
        resource_hash=resource_hash,
    )
    resource.validate_against(
        config=config,
        expected_asset_cols=expected_assets,
        add_self_loops=add_self_loops,
    )
    return resource


def _cpu_smoke_test() -> None:
    generator = torch.Generator().manual_seed(7)
    samples = []

    for day_index in range(3):
        returns = torch.randn(
            20,
            4,
            generator=generator,
        ) * 0.01
        close = 100.0 * torch.exp(
            torch.cat(
                [
                    torch.zeros(1, 4),
                    returns.cumsum(dim=0),
                ],
                dim=0,
            )
        )
        x = close.unsqueeze(-1)
        samples.append((x, {}, f"2024-01-{day_index + 1:02d}"))

    split = {
        "samples": samples,
        "asset_cols": ["A", "B", "C", "D"],
        "channels": ["close"],
    }
    config = FixedGraphResourceConfig(
        type="absolute_return_correlation",
        channel="close",
        threshold=0.0,
        empty_row_policy="error",
    )
    resource = fit_absolute_return_correlation_resource(
        split,
        config=config,
        expected_asset_cols=split["asset_cols"],
        add_self_loops=False,
    )

    if tuple(resource.adjacency.shape) != (4, 4):
        raise AssertionError("Unexpected smoke-test adjacency shape.")

    if not torch.allclose(
        resource.adjacency.sum(dim=-1),
        torch.ones(4),
    ):
        raise AssertionError("Smoke-test adjacency is not row-stochastic.")

    reloaded = FixedGraphResource.from_payload(
        resource.to_payload()
    )
    reloaded.validate_against(
        config=config,
        expected_asset_cols=split["asset_cols"],
        add_self_loops=False,
    )

    print("Fixed graph resource CPU smoke test passed.")


if __name__ == "__main__":
    _cpu_smoke_test()

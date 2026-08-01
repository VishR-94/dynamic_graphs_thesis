from __future__ import annotations

"""Continuous-price forecasting dataset helpers.

This module deliberately reuses :class:`WindowedCandleDataset` and the
canonical chronological/session-safe split contract.  It adds only one
matched-origin representation option: within-context log changes whose first
row is zero so the physical 60-minute forecasting origin is unchanged.
"""

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor

from src.data.data_generator import (
    ExampleDict,
    WindowContextNormaliser,
    WindowedCandleDataset,
)
from src.evaluation.prediction_transforms import (
    raw_to_cumulative_log_change,
)


InputRepresentation = Literal["raw", "context_log_change"]


@dataclass(frozen=True)
class ContinuousDatasetConfig:
    context_length: int = 60
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60)
    stride: int = 15
    input_channels: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    target_channels: tuple[str, ...] = ("close",)
    input_representation: InputRepresentation = "raw"
    eps: float = 1.0e-8
    clip: bool = False
    clip_min: float = -5.0
    clip_max: float = 5.0

    def validate(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if not self.horizons:
            raise ValueError("At least one forecast horizon is required.")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("horizons must be strictly increasing.")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique.")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("Every forecast horizon must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if not self.input_channels:
            raise ValueError("input_channels must not be empty.")
        if not self.target_channels:
            raise ValueError("target_channels must not be empty.")
        missing_targets = [
            channel
            for channel in self.target_channels
            if channel not in self.input_channels
        ]
        if missing_targets:
            raise ValueError(
                "Every target channel must also be an input channel. "
                f"Missing: {missing_targets}."
            )
        if self.input_representation not in {
            "raw",
            "context_log_change",
        }:
            raise ValueError(
                "input_representation must be 'raw' or "
                "'context_log_change'."
            )
        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.clip_min >= self.clip_max:
            raise ValueError("clip_min must be smaller than clip_max.")


class CumulativeLogChangeTargetAdapter:
    """Add the direct cumulative-log-change target to an example.

    The wrapped normaliser remains responsible for the model input and for
    the legacy normalised-Close target.  This adapter adds the exact target
    used by the direct-return model:

        target[h] = log(P[t + h]) - log(P[t])

    where ``P[t]`` is the final observed raw Close.  Only raw values already
    present in the supervised example are used; no future value enters input
    normalisation or any model feature.
    """

    def __init__(
        self,
        normaliser: Callable[[ExampleDict], ExampleDict],
        *,
        eps: float,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.normaliser = normaliser
        self.eps = float(eps)

    def __call__(self, example: ExampleDict) -> ExampleDict:
        output = self.normaliser(example)
        raw_target = torch.as_tensor(
            output["y_unnormalised"],
            dtype=torch.float32,
        )
        last_context_target = torch.as_tensor(
            output["last_context_target"],
            dtype=torch.float32,
        )
        output = dict(output)
        output["target_cumulative_log_change"] = (
            raw_to_cumulative_log_change(
                raw_target,
                last_context_target,
                eps=self.eps,
            )
        )
        return output


class MatchedContextLogChangeNormaliser:
    """Build matched-origin log-change inputs and raw-level Close targets.

    The base dataset first extracts the same raw physical context and targets
    used by the raw-price experiment.  This callable then transforms only the
    observed input context:

        x_change[0] = 0
        x_change[t] = log(x[t]) - log(x[t-1]),  t >= 1

    The resulting context retains length ``T`` and therefore retains exactly
    the same forecast origins and target timestamps as the raw experiment.
    Input log changes are normalised using their own context statistics.
    Future target levels are still normalised using the *raw observed context*
    target-channel mean and standard deviation for backwards-compatible
    normalised-Close experiments.  A separate outer adapter adds the direct
    cumulative-log-change target used by return-output experiments.
    """

    def __init__(
        self,
        *,
        eps: float = 1.0e-8,
        clip: bool = False,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if clip_min >= clip_max:
            raise ValueError("clip_min must be smaller than clip_max.")
        self.eps = float(eps)
        self.clip = bool(clip)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def __call__(self, example: ExampleDict) -> ExampleDict:
        raw_x = torch.as_tensor(example["x"], dtype=torch.float32)
        raw_y = torch.as_tensor(example["y"], dtype=torch.float32)

        if raw_x.ndim != 3:
            raise ValueError(
                "x must have shape [T,N,C]. "
                f"Received {tuple(raw_x.shape)}."
            )
        if raw_y.ndim != 3:
            raise ValueError(
                "y must have shape [H,N,C]. "
                f"Received {tuple(raw_y.shape)}."
            )
        if raw_x.shape[0] < 2:
            raise ValueError(
                "At least two context observations are required for "
                "within-context log changes."
            )

        log_values = torch.log(raw_x.clamp_min(self.eps))
        changes = torch.zeros_like(log_values)
        changes[1:] = log_values[1:] - log_values[:-1]

        input_mean = changes.mean(dim=0)
        input_std = changes.std(dim=0, unbiased=False).clamp_min(self.eps)
        input_log_std = torch.log(input_std)
        x_normalised = (changes - input_mean) / input_std

        input_channels = list(example["input_channels"])
        target_channels = list(example["target_channels"])
        target_positions = torch.tensor(
            [input_channels.index(channel) for channel in target_channels],
            dtype=torch.long,
            device=raw_x.device,
        )

        raw_context_mean = raw_x.mean(dim=0)
        raw_context_std = raw_x.std(dim=0, unbiased=False).clamp_min(
            self.eps
        )
        target_mean = raw_context_mean.index_select(1, target_positions)
        target_std = raw_context_std.index_select(1, target_positions)
        target_log_std = torch.log(target_std)
        y_normalised = (raw_y - target_mean) / target_std

        if self.clip:
            x_normalised = x_normalised.clamp(
                self.clip_min,
                self.clip_max,
            )
            y_normalised = y_normalised.clamp(
                self.clip_min,
                self.clip_max,
            )

        output = dict(example)
        output["x"] = x_normalised
        output["y"] = y_normalised
        output["y_unnormalised"] = raw_y
        output["norm_mean"] = input_mean
        output["norm_std"] = input_std
        output["norm_log_std"] = input_log_std
        output["target_norm_mean"] = target_mean
        output["target_norm_std"] = target_std
        output["target_norm_log_std"] = target_log_std
        output["input_representation"] = "context_log_change"
        return output


def build_continuous_dataset(
    split: dict[str, Any],
    *,
    config: ContinuousDatasetConfig,
) -> WindowedCandleDataset:
    """Build one continuous forecasting dataset under the canonical split."""
    config.validate()

    if config.input_representation == "raw":
        normaliser = WindowContextNormaliser(
            eps=config.eps,
            clip=config.clip,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            apply_to_target=True,
            include_stats=True,
        )
    else:
        normaliser = MatchedContextLogChangeNormaliser(
            eps=config.eps,
            clip=config.clip,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    normaliser_with_target = CumulativeLogChangeTargetAdapter(
        normaliser,
        eps=config.eps,
    )

    return WindowedCandleDataset(
        split=split,
        context_length=config.context_length,
        horizons=list(config.horizons),
        input_channels=list(config.input_channels),
        target_channels=list(config.target_channels),
        stride=config.stride,
        normaliser=normaliser_with_target,
    )


def _cpu_smoke_test() -> None:
    torch.manual_seed(3)
    channels = ["open", "high", "low", "close", "volume", "amount"]
    days = []
    for day_index in range(2):
        values = 10.0 + torch.rand(125, 3, len(channels))
        values[..., 4] *= 1000.0
        values[..., 5] = 0.0
        days.append((values, {}, f"2024-01-{day_index + 2:02d}"))

    split: dict[str, Any] = {
        "samples": days,
        "asset_cols": ["A", "B", "C"],
        "channels": channels,
    }

    raw_config = ContinuousDatasetConfig()
    log_config = ContinuousDatasetConfig(
        input_representation="context_log_change"
    )
    raw_dataset = build_continuous_dataset(split, config=raw_config)
    log_dataset = build_continuous_dataset(split, config=log_config)

    raw = raw_dataset[0]
    changed = log_dataset[0]
    if tuple(raw["x"].shape) != (60, 3, 5):
        raise AssertionError("Unexpected raw input shape.")
    if tuple(changed["x"].shape) != (60, 3, 5):
        raise AssertionError("Unexpected log-change input shape.")
    if not torch.equal(raw["target_indices"], changed["target_indices"]):
        raise AssertionError("Representations changed target timestamps.")
    if not torch.equal(raw["y_unnormalised"], changed["y_unnormalised"]):
        raise AssertionError("Representations changed raw targets.")
    torch.testing.assert_close(
        raw["target_cumulative_log_change"],
        changed["target_cumulative_log_change"],
        atol=0.0,
        rtol=0.0,
    )
    expected_target = raw_to_cumulative_log_change(
        raw["y_unnormalised"],
        raw["last_context_target"],
        eps=raw_config.eps,
    )
    torch.testing.assert_close(
        raw["target_cumulative_log_change"],
        expected_target,
        atol=0.0,
        rtol=0.0,
    )
    if not torch.allclose(
        changed["x"][0],
        (torch.zeros_like(changed["x"][0]) - changed["norm_mean"])
        / changed["norm_std"],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError("The first log-change row is not the zero state.")


if __name__ == "__main__":
    _cpu_smoke_test()
    print("Continuous dataset CPU smoke test passed.")

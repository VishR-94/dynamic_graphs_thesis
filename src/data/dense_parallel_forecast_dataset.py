from __future__ import annotations

"""Datasets for direct five-horizon dense-supervision experiments.

Two training contracts are required by the final graph-supervision diagnostic:

``stride1_fixed_context``
    Ordinary 60-minute contexts, five direct future Close targets, and a
    one-minute training stride.  Every sample is a fully observed 60-minute
    forecast origin.

``dense_prefix``
    Ordinary stride-15 outer windows.  For every position ``t`` inside the
    observed 60-minute context, the same five direct horizons are supervised:

        current position t -> t + [1, 5, 15, 30, 60]

    The model still receives only the observed 60-minute context.  Future rows
    are held separately as targets.  Context mean/std statistics are computed
    once from the full observed context and reused for all internal origins,
    matching the dense BaseDyGraph diagnostic convention used in this project.

All tensors follow the project asset convention and all target-index metadata
uses absolute within-session indices.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.load_candle_data import get_channel_index


@dataclass(frozen=True)
class DensePrefixDatasetConfig:
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
    target_channel: str = "close"
    eps: float = 1.0e-8
    clip: bool = False
    clip_min: float = -5.0
    clip_max: float = 5.0

    def validate(self) -> None:
        if int(self.context_length) <= 0:
            raise ValueError("context_length must be positive.")
        if not self.horizons:
            raise ValueError("horizons must not be empty.")
        if tuple(sorted(set(int(value) for value in self.horizons))) != tuple(
            int(value) for value in self.horizons
        ):
            raise ValueError("horizons must be unique and increasing.")
        if any(int(value) <= 0 for value in self.horizons):
            raise ValueError("Every horizon must be positive.")
        if int(self.stride) <= 0:
            raise ValueError("stride must be positive.")
        if not self.input_channels:
            raise ValueError("input_channels must not be empty.")
        if self.target_channel not in self.input_channels:
            raise ValueError("target_channel must occur in input_channels.")
        if float(self.eps) <= 0.0:
            raise ValueError("eps must be positive.")
        if float(self.clip_min) >= float(self.clip_max):
            raise ValueError("clip_min must be smaller than clip_max.")


class DensePrefixMultiHorizonDataset(Dataset[dict[str, Any]]):
    """Return one 60-origin dense multi-horizon training example.

    For one outer forecast origin the item contains:

    ``x``
        Full observed, context-normalised input, ``[T,N,C]``.

    ``dense_y_unnormalised``
        Raw Close target for every internal origin/horizon,
        ``[T,H,N,1]``.

    ``dense_target_cumulative_log_change``
        Exact target ``log(P[t+h]) - log(P[t])``, ``[T,H,N,1]``.

    The final internal origin ``t=T-1`` exactly matches the ordinary public
    forecasting task and is exposed through the standard ``y_unnormalised``,
    ``last_context_target`` and ``target_indices`` fields.
    """

    def __init__(
        self,
        split: dict[str, Any],
        *,
        config: DensePrefixDatasetConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.split = split
        self.config = config
        self.context_length = int(config.context_length)
        self.horizons = tuple(int(value) for value in config.horizons)
        self.stride = int(config.stride)
        self.input_channels = tuple(str(value) for value in config.input_channels)
        self.target_channel = str(config.target_channel)
        self.eps = float(config.eps)
        self.clip = bool(config.clip)
        self.clip_min = float(config.clip_min)
        self.clip_max = float(config.clip_max)
        self.asset_cols = tuple(str(value) for value in split["asset_cols"])

        self.input_channel_ids = [
            get_channel_index(split, channel) for channel in self.input_channels
        ]
        self.target_channel_id = get_channel_index(split, self.target_channel)
        self.target_input_position = self.input_channels.index(self.target_channel)
        self.index = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        values: list[tuple[int, int]] = []
        maximum_horizon = max(self.horizons)
        first_origin = self.context_length - 1
        for sample_index, (session, _, _) in enumerate(self.split["samples"]):
            session_length = int(torch.as_tensor(session).shape[0])
            last_origin = session_length - maximum_horizon - 1
            if last_origin < first_origin:
                continue
            values.extend(
                (sample_index, origin)
                for origin in range(first_origin, last_origin + 1, self.stride)
            )
        return values

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx, origin_idx = self.index[index]
        session, _, day = self.split["samples"][sample_idx]
        session = torch.as_tensor(session).float()
        session_length = int(session.shape[0])
        context_start = int(origin_idx) - self.context_length + 1
        context_end = int(origin_idx) + 1
        maximum_horizon = max(self.horizons)

        raw_context = session[
            context_start:context_end,
            :,
            self.input_channel_ids,
        ].float()
        expected_context = (
            self.context_length,
            len(self.asset_cols),
            len(self.input_channels),
        )
        if tuple(raw_context.shape) != expected_context:
            raise RuntimeError(
                f"Unexpected raw context shape {tuple(raw_context.shape)}; "
                f"expected {expected_context}."
            )

        mean = raw_context.mean(dim=0)
        std = raw_context.std(dim=0, unbiased=False).clamp_min(self.eps)
        x = (raw_context - mean) / std
        if self.clip:
            x = x.clamp(self.clip_min, self.clip_max)

        # The segment begins at the first observed context position and extends
        # far enough to contain the 60-minute target from the final origin.
        raw_close_segment = session[
            context_start : context_end + maximum_horizon,
            :,
            self.target_channel_id,
        ].float()
        expected_segment_length = self.context_length + maximum_horizon
        if tuple(raw_close_segment.shape) != (
            expected_segment_length,
            len(self.asset_cols),
        ):
            raise RuntimeError("Dense target segment has an unexpected shape.")

        internal_origins = torch.arange(self.context_length, dtype=torch.long)
        horizon_tensor = torch.tensor(self.horizons, dtype=torch.long)
        relative_target_indices = internal_origins[:, None] + horizon_tensor[None, :]
        dense_target_raw = raw_close_segment.index_select(
            0,
            relative_target_indices.reshape(-1),
        ).reshape(
            self.context_length,
            len(self.horizons),
            len(self.asset_cols),
            1,
        )
        dense_current_raw = raw_close_segment[: self.context_length].unsqueeze(-1)

        target_mean = mean[:, self.target_input_position : self.target_input_position + 1]
        target_std = std[:, self.target_input_position : self.target_input_position + 1]
        dense_target_normalised = (
            dense_target_raw - target_mean.view(1, 1, len(self.asset_cols), 1)
        ) / target_std.view(1, 1, len(self.asset_cols), 1)
        if self.clip:
            dense_target_normalised = dense_target_normalised.clamp(
                self.clip_min,
                self.clip_max,
            )

        dense_target_cumulative_log_change = (
            torch.log(dense_target_raw.clamp_min(self.eps))
            - torch.log(dense_current_raw[:, None].clamp_min(self.eps))
        )

        absolute_dense_target_indices = (
            torch.arange(
                context_start,
                context_end,
                dtype=torch.long,
            )[:, None]
            + horizon_tensor[None, :]
        )
        final_target_indices = absolute_dense_target_indices[-1].contiguous()

        return {
            "x": x.contiguous(),
            "context_unnormalised": raw_context.contiguous(),
            "dense_y_normalised": dense_target_normalised.contiguous(),
            "dense_y_unnormalised": dense_target_raw.contiguous(),
            "dense_current_close": dense_current_raw.contiguous(),
            "dense_target_cumulative_log_change": (
                dense_target_cumulative_log_change.contiguous()
            ),
            "dense_target_indices": absolute_dense_target_indices.contiguous(),
            # Standard final-origin forecasting contract.
            "y": dense_target_normalised[-1].contiguous(),
            "y_unnormalised": dense_target_raw[-1].contiguous(),
            "target_cumulative_log_change": (
                dense_target_cumulative_log_change[-1].contiguous()
            ),
            "last_context_target": dense_current_raw[-1].contiguous(),
            "target_indices": final_target_indices,
            "norm_mean": mean.contiguous(),
            "norm_std": std.contiguous(),
            "target_norm_mean": target_mean.contiguous(),
            "target_norm_std": target_std.contiguous(),
            "day": str(day),
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "origin_idx": torch.tensor(origin_idx, dtype=torch.long),
            "context_start": torch.tensor(context_start, dtype=torch.long),
            "context_end": torch.tensor(context_end, dtype=torch.long),
            "session_length": torch.tensor(session_length, dtype=torch.long),
            "input_channels": list(self.input_channels),
            "target_channels": [self.target_channel],
            "horizons": list(self.horizons),
            "asset_cols": list(self.asset_cols),
            "input_representation": "raw",
        }


def build_dense_prefix_dataset(
    split: dict[str, Any],
    *,
    context_length: int,
    horizons: Sequence[int],
    stride: int,
    input_channels: Sequence[str],
    target_channel: str = "close",
    eps: float = 1.0e-8,
    clip: bool = False,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> DensePrefixMultiHorizonDataset:
    return DensePrefixMultiHorizonDataset(
        split,
        config=DensePrefixDatasetConfig(
            context_length=int(context_length),
            horizons=tuple(int(value) for value in horizons),
            stride=int(stride),
            input_channels=tuple(str(value) for value in input_channels),
            target_channel=str(target_channel),
            eps=float(eps),
            clip=bool(clip),
            clip_min=float(clip_min),
            clip_max=float(clip_max),
        ),
    )


def right_aligned_prefix_batch(
    x: Tensor,
    prefix_indices: Sequence[int] | Tensor,
) -> Tensor:
    """Return zero-padded fixed-length prefixes for ModernTCN.

    ``x`` has shape ``[B,T,N,C]`` and the result has shape
    ``[B*K,T,N,C]`` where ``K`` is the number of requested zero-based internal
    origins.  Prefix ``t`` contains ``x[:, :t+1]`` right-aligned so its current
    observation occupies the final model position.  Padding is zero in already
    normalised space, i.e. the full-window mean state.
    """

    values = torch.as_tensor(x)
    if values.ndim != 4:
        raise ValueError("x must have shape [B,T,N,C].")
    batch, steps, nodes, channels = map(int, values.shape)
    indices = torch.as_tensor(prefix_indices, dtype=torch.long).flatten()
    if indices.numel() == 0:
        raise ValueError("prefix_indices must not be empty.")
    if torch.any(indices < 0) or torch.any(indices >= steps):
        raise ValueError("prefix_indices lie outside the context window.")

    outputs: list[Tensor] = []
    for raw_index in indices.tolist():
        prefix_length = int(raw_index) + 1
        padded = values.new_zeros(batch, steps, nodes, channels)
        padded[:, steps - prefix_length :] = values[:, :prefix_length]
        outputs.append(padded)
    return torch.cat(outputs, dim=0).contiguous()


def repeat_batch_for_prefixes(values: Tensor, prefix_count: int) -> Tensor:
    """Repeat batch-major values in the same order as prefix batching.

    ``right_aligned_prefix_batch`` concatenates complete batches prefix by
    prefix.  This helper produces ``[prefix0 batch, prefix1 batch, ...]`` for
    arbitrary batch-major metadata tensors.
    """

    tensor = torch.as_tensor(values)
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    if int(prefix_count) <= 0:
        raise ValueError("prefix_count must be positive.")
    repeats = [int(prefix_count)] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats).contiguous()

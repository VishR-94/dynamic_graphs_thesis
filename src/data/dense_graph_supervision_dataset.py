from __future__ import annotations

"""Datasets for dense one-step graph-supervision diagnostics.

The BaseDyGraph controls need two aligned views of the same canonical
forecast windows:

* the cached coarse ``s1`` sequence used by the token-input model; and
* the causally normalised continuous OHLCV sequence and raw Close levels used
  by both direct-price heads and the common price-space evaluator.

No window count is hard-coded.  Alignment is checked from the saved
``sample_idx``/``origin_idx`` metadata and from the five canonical target
indices.  The first future row is appended only as a teacher-forced target;
causal temporal masks ensure it cannot affect the forecast-origin hidden state
or graph.
"""

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)


DEFAULT_INPUT_CHANNELS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
DEFAULT_ALIGNMENT_HORIZONS: tuple[int, ...] = (1, 5, 15, 30, 60)


class AlignedTokenContinuousDenseDataset(Dataset[dict[str, Any]]):
    """Pair an origin-aligned token cache with the matching raw-price window."""

    def __init__(
        self,
        *,
        split: dict[str, Any],
        token_cache_path: str | Path,
        context_length: int = 60,
        stride: int = 15,
        alignment_horizons: Sequence[int] = DEFAULT_ALIGNMENT_HORIZONS,
        input_channels: Sequence[str] = DEFAULT_INPUT_CHANNELS,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.token_dataset = CachedTokenGraphDataset.from_path(token_cache_path)
        self.input_channels = tuple(str(value) for value in input_channels)
        self.alignment_horizons = tuple(int(value) for value in alignment_horizons)
        if not self.alignment_horizons:
            raise ValueError("alignment_horizons must not be empty.")
        if self.alignment_horizons[0] != 1:
            raise ValueError("The first alignment horizon must be 1 minute.")
        if "close" not in self.input_channels:
            raise ValueError("input_channels must contain Close.")
        self.close_index = self.input_channels.index("close")

        config = ContinuousDatasetConfig(
            context_length=int(context_length),
            horizons=self.alignment_horizons,
            stride=int(stride),
            input_channels=self.input_channels,
            # Full h=1 OHLCV is required for the continuous teacher-forced
            # input.  Only Close is used by the objective and evaluator.
            target_channels=self.input_channels,
            input_representation="raw",
            eps=float(eps),
            clip=False,
        )
        self.continuous_dataset = build_continuous_dataset(split, config=config)
        self.asset_cols = tuple(str(value) for value in split["asset_cols"])
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.eps = float(eps)

        if len(self.token_dataset) != len(self.continuous_dataset):
            raise ValueError(
                "Token and continuous windows differ: "
                f"{len(self.token_dataset)} != {len(self.continuous_dataset)}."
            )
        if tuple(self.token_dataset.asset_cols) != self.asset_cols:
            raise ValueError("Token and continuous asset ordering differs.")
        if self.token_dataset.context_length != self.context_length:
            raise ValueError("Token cache context length differs from the dataset.")

        # Check a deterministic subset up front.  Every item is checked again
        # when accessed, so this is a useful early diagnostic without loading
        # the complete cache into duplicated Python structures.
        if len(self):
            indices = sorted({0, len(self) // 2, len(self) - 1})
            for index in indices:
                self._aligned_pair(index)

    def __len__(self) -> int:
        return len(self.token_dataset)

    def _aligned_pair(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        token_item = self.token_dataset[index]
        continuous_item = self.continuous_dataset[index]

        token_sample = int(torch.as_tensor(token_item["sample_idx"]).item())
        continuous_sample = int(
            torch.as_tensor(continuous_item["sample_idx"]).item()
        )
        token_origin = int(torch.as_tensor(token_item["origin_idx"]).item())
        continuous_origin = int(
            torch.as_tensor(continuous_item["origin_idx"]).item()
        )
        if (token_sample, token_origin) != (
            continuous_sample,
            continuous_origin,
        ):
            raise ValueError(
                "Token/raw window alignment differs at index "
                f"{index}: token={(token_sample, token_origin)}, "
                f"continuous={(continuous_sample, continuous_origin)}."
            )

        token_targets = torch.as_tensor(token_item["target_indices"]).long()
        continuous_targets = torch.as_tensor(
            continuous_item["target_indices"]
        ).long()
        dense_positions = torch.tensor(
            [int(value) - 1 for value in self.alignment_horizons],
            dtype=torch.long,
        )
        selected_token_targets = token_targets.index_select(0, dense_positions)
        if not torch.equal(selected_token_targets, continuous_targets):
            raise ValueError(
                "Token and continuous target timestamps differ at window "
                f"{index}."
            )
        return token_item, continuous_item

    def __getitem__(self, index: int) -> dict[str, Any]:
        token_item, continuous_item = self._aligned_pair(index)

        x_normalised = torch.as_tensor(continuous_item["x"]).float()
        y_normalised = torch.as_tensor(continuous_item["y"]).float()
        raw_context = torch.as_tensor(
            continuous_item["context_unnormalised"]
        ).float()
        raw_future = torch.as_tensor(
            continuous_item["y_unnormalised"]
        ).float()
        norm_mean = torch.as_tensor(continuous_item["norm_mean"]).float()
        norm_std = torch.as_tensor(continuous_item["norm_std"]).float()

        expected_context = (
            self.context_length,
            len(self.asset_cols),
            len(self.input_channels),
        )
        if tuple(x_normalised.shape) != expected_context:
            raise ValueError(
                f"Unexpected continuous context shape {tuple(x_normalised.shape)}; "
                f"expected {expected_context}."
            )
        if tuple(y_normalised.shape[1:]) != expected_context[1:]:
            raise ValueError("Continuous target asset/channel axes differ.")

        continuous_teacher_sequence = torch.cat(
            (x_normalised, y_normalised[:1]),
            dim=0,
        ).contiguous()
        raw_close_sequence = torch.cat(
            (
                raw_context[..., self.close_index],
                raw_future[:1, ..., self.close_index],
            ),
            dim=0,
        ).contiguous()
        dense_target_normalised_close = torch.cat(
            (
                x_normalised[1:, ..., self.close_index],
                y_normalised[:1, ..., self.close_index],
            ),
            dim=0,
        ).unsqueeze(-1).contiguous()

        result = {
            "context_s1": torch.as_tensor(
                token_item["context_tokens"]
            )[..., 0].long(),
            "first_future_s1": torch.as_tensor(
                token_item["target_s1"]
            )[0].long(),
            "continuous_teacher_sequence": continuous_teacher_sequence,
            "dense_target_normalised_close": dense_target_normalised_close,
            "raw_close_sequence": raw_close_sequence,
            "close_norm_mean": norm_mean[:, self.close_index].contiguous(),
            "close_norm_std": norm_std[:, self.close_index].contiguous(),
            "last_context_close": raw_close_sequence[-2].unsqueeze(-1),
            "future_h1_close": raw_close_sequence[-1].view(
                1,
                len(self.asset_cols),
                1,
            ),
            "sample_idx": torch.as_tensor(
                continuous_item["sample_idx"]
            ).long(),
            "origin_idx": torch.as_tensor(
                continuous_item["origin_idx"]
            ).long(),
            "target_indices": torch.as_tensor(
                continuous_item["target_indices"]
            )[:1].long(),
            "context_start": torch.as_tensor(
                continuous_item["context_start"]
            ).long(),
            "session_length": torch.as_tensor(
                continuous_item["session_length"]
            ).long(),
            "day": str(continuous_item["day"]),
        }
        return result


def make_uniform_nonself_graph(num_nodes: int) -> Tensor:
    """Return a row-stochastic no-self graph with no economic information."""

    nodes = int(num_nodes)
    if nodes <= 1:
        raise ValueError("num_nodes must exceed one.")
    values = torch.ones(nodes, nodes, dtype=torch.float32)
    values.fill_diagonal_(0.0)
    return values / values.sum(dim=-1, keepdim=True)

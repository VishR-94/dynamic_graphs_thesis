from collections.abc import Sequence
from itertools import product
from typing import Any

import pandas as pd
import torch

from src.evaluation.metrics import mae, mse, rmse


def _as_list(value: Any | Sequence[Any] | None) -> list[Any] | None:
    """
    Convert a scalar/list/tuple selection into a list.

    None means average over that dimension.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def _get_index(
    value: Any,
    available_values: Sequence[Any],
    dim_name: str,
) -> int:
    """
    Get the index of a human-readable label.
    """
    available_values = list(available_values)

    if value not in available_values:
        raise ValueError(
            f"{dim_name} value {value} was requested, but it is not available. "
            f"Available values: {available_values}"
        )

    return available_values.index(value)


def _get_metric_function(metric: str):
    """
    Convert a metric name into the corresponding metric function.
    """
    metric = metric.lower()

    if metric == "mae":
        return mae

    if metric == "mse":
        return mse

    if metric == "rmse":
        return rmse

    raise ValueError(
        f"Unknown metric: {metric}. "
        "Expected one of: mae, mse, rmse."
    )


def make_metric_table(
    metric: str,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    horizons: Sequence[int],
    channels: Sequence[str],
    assets: Sequence[str] | None = None,
    horizon: int | Sequence[int] | None = None,
    asset: str | Sequence[str] | None = None,
    channel: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Create a readable metric table.

    Args:
        metric:
            One of:
                mae
                mse
                rmse

        y_pred:
            Prediction tensor with shape [B, H, N, C] or [H, N, C].

        y_true:
            Ground-truth tensor with shape [B, H, N, C] or [H, N, C].

        horizons:
            Horizon labels, e.g. [1, 5, 15, 30, 60].

        channels:
            Channel labels, e.g. ["open", "high", "low", "close", "volume"].

        assets:
            Asset labels, e.g. ["AAPL", "MSFT"].

            If None, assets are labelled by integer index.

        horizon:
            Horizons to display.

            If None, average over all horizons.

        asset:
            Assets to display.

            If None, average over all assets.

        channel:
            Channels to display.

            If None, average over all channels.

    Returns:
        pandas DataFrame.
    """
    metric_fn = _get_metric_function(metric)

    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred to have shape [B, H, N, C] or [H, N, C], "
            f"got {tuple(y_pred.shape)}."
        )

    if y_pred.shape != y_true.shape:
        raise ValueError(
            "y_pred and y_true must have the same shape. "
            f"Got {tuple(y_pred.shape)} and {tuple(y_true.shape)}."
        )

    _, num_horizons, num_assets, num_channels = y_pred.shape

    if len(horizons) != num_horizons:
        raise ValueError(
            f"len(horizons) must match prediction horizon dimension. "
            f"Got len(horizons)={len(horizons)} and H={num_horizons}."
        )

    if len(channels) != num_channels:
        raise ValueError(
            f"len(channels) must match prediction channel dimension. "
            f"Got len(channels)={len(channels)} and C={num_channels}."
        )

    if assets is None:
        assets = list(range(num_assets))

    if len(assets) != num_assets:
        raise ValueError(
            f"len(assets) must match prediction asset dimension. "
            f"Got len(assets)={len(assets)} and N={num_assets}."
        )

    requested_horizons = _as_list(horizon)
    requested_assets = _as_list(asset)
    requested_channels = _as_list(channel)

    horizon_labels = requested_horizons if requested_horizons is not None else [None]
    asset_labels = requested_assets if requested_assets is not None else [None]
    channel_labels = requested_channels if requested_channels is not None else [None]

    rows = []

    for horizon_label, asset_label, channel_label in product(
        horizon_labels,
        asset_labels,
        channel_labels,
    ):
        y_pred_selected = y_pred
        y_true_selected = y_true

        row = {}

        if horizon_label is not None:
            horizon_idx = _get_index(
                value=horizon_label,
                available_values=horizons,
                dim_name="horizon",
            )

            y_pred_selected = y_pred_selected[:, horizon_idx:horizon_idx + 1, :, :]
            y_true_selected = y_true_selected[:, horizon_idx:horizon_idx + 1, :, :]

            row["horizon"] = horizon_label

        if asset_label is not None:
            asset_idx = _get_index(
                value=asset_label,
                available_values=assets,
                dim_name="asset",
            )

            y_pred_selected = y_pred_selected[:, :, asset_idx:asset_idx + 1, :]
            y_true_selected = y_true_selected[:, :, asset_idx:asset_idx + 1, :]

            row["asset"] = asset_label

        if channel_label is not None:
            channel_idx = _get_index(
                value=channel_label,
                available_values=channels,
                dim_name="channel",
            )

            y_pred_selected = y_pred_selected[:, :, :, channel_idx:channel_idx + 1]
            y_true_selected = y_true_selected[:, :, :, channel_idx:channel_idx + 1]

            row["channel"] = channel_label

        value = metric_fn(
            y_pred=y_pred_selected,
            y_true=y_true_selected,
        )

        row[metric.upper()] = float(value.item())

        rows.append(row)

    return pd.DataFrame(rows)
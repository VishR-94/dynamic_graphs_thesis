from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch


def make_evaluation_table(
    metric_results: Mapping[str, torch.Tensor],
    horizons: Sequence[int],
    channels: Sequence[str],
) -> pd.DataFrame:
    """
    Convert evaluator metric results into a long-form table.

    Each metric tensor must have shape [H, C], where:
        H = number of forecast horizons
        C = number of target channels

    The returned table contains one row for each
    metric-horizon-channel combination.

    Args:
        metric_results:
            Mapping from metric name to a tensor with shape [H, C].

        horizons:
            Forecast-horizon labels in tensor order.

        channels:
            Channel labels in tensor order.

    Returns:
        DataFrame with columns:
            metric
            horizon
            channel
            value
    """
    if len(metric_results) == 0:
        raise ValueError(
            "metric_results must contain at least one metric."
        )

    horizons = list(horizons)
    channels = list(channels)

    expected_shape = (
        len(horizons),
        len(channels),
    )

    rows: list[dict[str, Any]] = []

    for metric_name, values in metric_results.items():
        if not isinstance(metric_name, str):
            raise TypeError(
                "Metric names must be strings. "
                f"Got {type(metric_name).__name__}."
            )

        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"Metric {metric_name!r} must be a torch.Tensor."
            )

        if values.shape != expected_shape:
            raise ValueError(
                f"Metric {metric_name!r} has shape "
                f"{tuple(values.shape)}. "
                f"Expected {expected_shape}."
            )

        values_cpu = values.detach().cpu()

        for horizon_idx, horizon in enumerate(horizons):
            for channel_idx, channel in enumerate(channels):
                rows.append(
                    {
                        "metric": metric_name,
                        "horizon": horizon,
                        "channel": channel,
                        "value": float(
                            values_cpu[
                                horizon_idx,
                                channel_idx,
                            ].item()
                        ),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "horizon",
            "channel",
            "value",
        ],
    )
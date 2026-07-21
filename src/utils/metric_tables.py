from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch


MetricResultValue = (
    torch.Tensor
    | Mapping[str, torch.Tensor]
)


def make_evaluation_table(
    metric_results: Mapping[
        str,
        MetricResultValue,
    ],
    horizons: Sequence[int],
    channels: Sequence[str],
) -> pd.DataFrame:
    """
    Convert evaluator metric results into a long-form table.

    Ordinary metric results must have the form:

        {
            metric_name: Tensor[H, C]
        }

    Bootstrapped metric results must have the form:

        {
            metric_name: {
                "value": Tensor[H, C],
                "bootstrap_mean": Tensor[H, C],
                "bootstrap_std": Tensor[H, C],
                "ci_lower": Tensor[H, C],
                "ci_upper": Tensor[H, C],
            }
        }

    Args:
        metric_results:
            Ordinary or bootstrapped evaluator results.

        horizons:
            Forecast-horizon labels in tensor order.

        channels:
            Channel labels in tensor order.

    Returns:
        For ordinary results, a DataFrame with columns:

            metric
            horizon
            channel
            value

        For bootstrapped results, a DataFrame with columns:

            metric
            horizon
            channel
            value
            bootstrap_mean
            bootstrap_std
            ci_lower
            ci_upper
    """
    if len(metric_results) == 0:
        raise ValueError(
            "metric_results must contain at least one metric."
        )

    horizons = list(horizons)
    channels = list(channels)

    if len(horizons) == 0:
        raise ValueError(
            "horizons must contain at least one value."
        )

    if len(channels) == 0:
        raise ValueError(
            "channels must contain at least one value."
        )

    expected_shape = (
        len(horizons),
        len(channels),
    )

    first_result = next(
        iter(metric_results.values())
    )

    if isinstance(first_result, torch.Tensor):
        result_mode = "ordinary"

    elif isinstance(first_result, Mapping):
        result_mode = "bootstrap"

    else:
        raise TypeError(
            "Each metric result must be either a torch.Tensor "
            "or a mapping of bootstrap summary tensors."
        )

    bootstrap_summary_names = (
        "value",
        "bootstrap_mean",
        "bootstrap_std",
        "ci_lower",
        "ci_upper",
    )

    required_bootstrap_keys = set(
        bootstrap_summary_names
    )

    rows: list[dict[str, Any]] = []

    for metric_name, metric_value in (
        metric_results.items()
    ):
        if not isinstance(metric_name, str):
            raise TypeError(
                "Metric names must be strings. "
                f"Got {type(metric_name).__name__}."
            )

        if result_mode == "ordinary":
            if not isinstance(
                metric_value,
                torch.Tensor,
            ):
                raise TypeError(
                    "metric_results cannot mix ordinary tensors "
                    "and bootstrap summary mappings."
                )

            if metric_value.shape != expected_shape:
                raise ValueError(
                    f"Metric {metric_name!r} has shape "
                    f"{tuple(metric_value.shape)}. "
                    f"Expected {expected_shape}."
                )

            values_cpu = (
                metric_value
                .detach()
                .cpu()
            )

            for horizon_idx, horizon in enumerate(
                horizons
            ):
                for channel_idx, channel in enumerate(
                    channels
                ):
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

            continue

        if not isinstance(
            metric_value,
            Mapping,
        ):
            raise TypeError(
                "metric_results cannot mix ordinary tensors "
                "and bootstrap summary mappings."
            )

        actual_bootstrap_keys = set(
            metric_value
        )

        if (
            actual_bootstrap_keys
            != required_bootstrap_keys
        ):
            missing_keys = (
                required_bootstrap_keys
                - actual_bootstrap_keys
            )

            unexpected_keys = (
                actual_bootstrap_keys
                - required_bootstrap_keys
            )

            raise ValueError(
                f"Bootstrap result for {metric_name!r} has "
                "incorrect summary keys. "
                f"Missing: {sorted(missing_keys)}. "
                f"Unexpected: {sorted(unexpected_keys)}."
            )

        summary_tensors: dict[
            str,
            torch.Tensor,
        ] = {}

        for summary_name in (
            bootstrap_summary_names
        ):
            summary_values = metric_value[
                summary_name
            ]

            if not isinstance(
                summary_values,
                torch.Tensor,
            ):
                raise TypeError(
                    f"Bootstrap summary "
                    f"{metric_name!r}[{summary_name!r}] "
                    "must be a torch.Tensor."
                )

            if summary_values.shape != expected_shape:
                raise ValueError(
                    f"Bootstrap summary "
                    f"{metric_name!r}[{summary_name!r}] "
                    f"has shape {tuple(summary_values.shape)}. "
                    f"Expected {expected_shape}."
                )

            summary_tensors[
                summary_name
            ] = (
                summary_values
                .detach()
                .cpu()
            )

        for horizon_idx, horizon in enumerate(
            horizons
        ):
            for channel_idx, channel in enumerate(
                channels
            ):
                row: dict[str, Any] = {
                    "metric": metric_name,
                    "horizon": horizon,
                    "channel": channel,
                }

                for summary_name in (
                    bootstrap_summary_names
                ):
                    row[summary_name] = float(
                        summary_tensors[
                            summary_name
                        ][
                            horizon_idx,
                            channel_idx,
                        ].item()
                    )

                rows.append(row)

    if result_mode == "ordinary":
        columns = [
            "metric",
            "horizon",
            "channel",
            "value",
        ]

    else:
        columns = [
            "metric",
            "horizon",
            "channel",
            "value",
            "bootstrap_mean",
            "bootstrap_std",
            "ci_lower",
            "ci_upper",
        ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )
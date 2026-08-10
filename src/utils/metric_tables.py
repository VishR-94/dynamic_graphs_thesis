from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import torch


MetricResultValue = (
    torch.Tensor
    | Mapping[str, torch.Tensor]
)

DEFAULT_METRIC_DISPLAY_NAMES = {
    "raw_mae": "Raw MAE",
    "raw_rmse": "Raw RMSE",
    "cumulative_log_change_mae": "Log MAE",
    "cumulative_log_change_median_absolute_error": "Log MedAE",
    "cumulative_log_change_p95_absolute_error": "Log P95 AE",
    "cumulative_log_change_rmse": "Log RMSE",
    "mase": "MASE",
    "relative_mae_vs_persistence": "Rel. MAE",
    "persistence_win_rate": "Win Rate",
    "cumulative_log_change_directional_accuracy": "Sign Acc.",
    "cumulative_log_change_pearson_correlation": "Pearson",
    "cumulative_log_change_cross_sectional_pearson_ic": "IC",
    "cumulative_log_change_cross_sectional_spearman_rank_ic": (
        "Rank IC"
    ),
    "cumulative_log_change_movement_magnitude_ratio": "MMR",
    "cumulative_log_change_temporal_absolute_correlation": (
        "AbsRet Corr"
    ),
}


DEFAULT_MODEL_DISPLAY_NAMES = {
    "persistence": "Persistence",
    "mean": "Mean",
    "arima": "ARIMA",
    "var": "VAR",
    "garch": "GARCH",
    "modern_tcn": "ModernTCN",
    "kronos": "Kronos",
}


DEFAULT_SUMMARY_METRICS = (
    "cumulative_log_change_mae",
    "cumulative_log_change_median_absolute_error",
    "cumulative_log_change_p95_absolute_error",
    "relative_mae_vs_persistence",
    "persistence_win_rate",
    "cumulative_log_change_directional_accuracy",
    "cumulative_log_change_pearson_correlation",
    "cumulative_log_change_cross_sectional_pearson_ic",
    "cumulative_log_change_cross_sectional_spearman_rank_ic",
    "cumulative_log_change_movement_magnitude_ratio",
    "cumulative_log_change_temporal_absolute_correlation",
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

    A metric that is intentionally not bootstrapped uses NaN tensors for
    the four bootstrap summaries. Notebook styling displays those entries
    as an em dash while preserving the ordinary full-sample value.

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


def make_baseline_summary_table(
    models_to_display: Sequence[str],
    namespace: Mapping[str, Any],
    channel: str = "close",
    metrics_to_display: Sequence[str] | None = None,
    metric_display_names: Mapping[str, str] | None = None,
    model_display_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Combine model evaluation tables into one horizon-by-model summary.

    Each model name is resolved from namespace using the convention:

        {model}_metric_table

    Only the ordinary full-test-set value is included. Bootstrap
    confidence intervals remain in the individual model tables.

    By default, the compact headline metric set in
    ``DEFAULT_SUMMARY_METRICS`` is displayed. MASE remains available
    in the evaluator and detailed tables and can be restored by passing
    it explicitly through ``metrics_to_display``.

    Missing metrics for a model are shown as NaN.
    """
    table_names = [
        f"{model}_metric_table"
        for model in models_to_display
    ]

    missing_tables = [
        table_name
        for table_name in table_names
        if table_name not in namespace
    ]

    if missing_tables:
        raise NameError(
            "Missing metric tables: "
            + ", ".join(missing_tables)
        )

    frames = []

    for model, table_name in zip(
        models_to_display,
        table_names,
    ):
        metric_table = namespace[table_name]

        model_frame = metric_table.loc[
            metric_table["channel"] == channel,
            [
                "metric",
                "horizon",
                "value",
            ],
        ].copy()

        model_frame["model"] = model
        frames.append(model_frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    metric_labels = dict(
        DEFAULT_METRIC_DISPLAY_NAMES
    )

    if metric_display_names is not None:
        metric_labels.update(
            metric_display_names
        )

    if metrics_to_display is None:
        metric_order = list(
            DEFAULT_SUMMARY_METRICS
        )
    else:
        metric_order = [
            str(metric_name)
            for metric_name in metrics_to_display
        ]

    if len(metric_order) == 0:
        raise ValueError(
            "metrics_to_display must contain at least one metric."
        )

    duplicate_metrics = {
        metric_name
        for metric_name in metric_order
        if metric_order.count(metric_name) > 1
    }

    if duplicate_metrics:
        raise ValueError(
            "metrics_to_display must not contain duplicates: "
            f"{sorted(duplicate_metrics)}."
        )

    horizons = sorted(
        combined["horizon"].unique().tolist()
    )

    full_index = pd.MultiIndex.from_product(
        [
            horizons,
            list(models_to_display),
        ],
        names=[
            "horizon",
            "model",
        ],
    )

    summary = (
        combined
        .pivot(
            index=[
                "horizon",
                "model",
            ],
            columns="metric",
            values="value",
        )
        .reindex(
            index=full_index,
            columns=metric_order,
        )
    )

    model_labels = dict(
        DEFAULT_MODEL_DISPLAY_NAMES
    )

    if model_display_names is not None:
        model_labels.update(
            model_display_names
        )

    summary.index = pd.MultiIndex.from_tuples(
        [
            (
                f"{int(horizon)} min",
                model_labels.get(
                    model,
                    model.replace(
                        "_",
                        " ",
                    ).title(),
                ),
            )
            for horizon, model in summary.index
        ],
        names=[
            "Horizon",
            "Model",
        ],
    )

    summary = summary.rename(
        columns={
            metric_name: metric_labels.get(
                metric_name,
                metric_name,
            )
            for metric_name in metric_order
        }
    )

    summary.columns.name = None

    return summary

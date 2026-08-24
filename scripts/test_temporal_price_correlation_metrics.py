"""CPU contract test for the two secondary temporal correlations."""

from __future__ import annotations

import math

import pandas as pd
import torch

from src.evaluation.metrics import (
    ForecastEvaluator,
    forecast_series_log_return_values,
)
from src.utils.metric_tables import (
    make_baseline_summary_table,
    make_evaluation_table,
)


RAW_PRICE_METRIC = "raw_price_temporal_pearson_correlation"
SERIES_RETURN_METRIC = (
    "forecast_series_log_return_temporal_pearson_correlation"
)


def _pearson(x: list[float], y: list[float]) -> float:
    x_tensor = torch.tensor(x, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64)
    x_centred = x_tensor - x_tensor.mean()
    y_centred = y_tensor - y_tensor.mean()
    denominator = torch.sqrt(
        x_centred.square().sum()
        * y_centred.square().sum()
    )
    return float(
        (x_centred * y_centred).sum()
        / denominator
    )


def _expected_raw_correlation(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    _, num_horizons, num_assets, num_channels = y_pred.shape
    output = torch.empty(
        (num_horizons, num_channels),
        dtype=torch.float64,
    )

    for horizon in range(num_horizons):
        for channel in range(num_channels):
            asset_values = []
            for asset in range(num_assets):
                asset_values.append(
                    _pearson(
                        y_pred[:, horizon, asset, channel].tolist(),
                        y_true[:, horizon, asset, channel].tolist(),
                    )
                )
            output[horizon, channel] = sum(asset_values) / len(asset_values)

    return output


def _expected_series_return_correlation(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    # Session 0 is regular.  Session 1 contains a deliberately missing
    # origin between rows 5 and 6.  Neither the overnight pair 3 -> 4 nor
    # the within-session gap 5 -> 6 is allowed to contribute.
    valid_pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (4, 5),
        (6, 7),
    ]

    _, num_horizons, num_assets, num_channels = y_pred.shape
    output = torch.empty(
        (num_horizons, num_channels),
        dtype=torch.float64,
    )

    for horizon in range(num_horizons):
        for channel in range(num_channels):
            asset_values = []
            for asset in range(num_assets):
                pred_returns = [
                    math.log(float(y_pred[current, horizon, asset, channel]))
                    - math.log(float(y_pred[previous, horizon, asset, channel]))
                    for previous, current in valid_pairs
                ]
                true_returns = [
                    math.log(float(y_true[current, horizon, asset, channel]))
                    - math.log(float(y_true[previous, horizon, asset, channel]))
                    for previous, current in valid_pairs
                ]
                asset_values.append(
                    _pearson(pred_returns, true_returns)
                )
            output[horizon, channel] = sum(asset_values) / len(asset_values)

    return output


def _prediction_result() -> dict[str, object]:
    # [B=8, H=2, N=2, C=1]
    y_true = torch.tensor(
        [
            [[[100.0], [50.0]], [[102.0], [55.0]]],
            [[[101.0], [49.0]], [[103.0], [54.0]]],
            [[[103.0], [50.0]], [[106.0], [56.0]]],
            [[[104.0], [52.0]], [[108.0], [57.0]]],
            [[[200.0], [80.0]], [[205.0], [84.0]]],
            [[[202.0], [79.0]], [[207.0], [83.0]]],
            [[[204.0], [81.0]], [[210.0], [86.0]]],
            [[[208.0], [82.0]], [[214.0], [88.0]]],
        ],
        dtype=torch.float64,
    )

    y_pred = torch.tensor(
        [
            [[[100.5], [49.5]], [[101.5], [54.5]]],
            [[[101.2], [49.2]], [[103.4], [54.2]]],
            [[[102.8], [50.4]], [[105.5], [55.5]]],
            [[[104.5], [51.6]], [[108.5], [57.5]]],
            [[[198.0], [79.5]], [[204.0], [83.5]]],
            [[[202.5], [79.2]], [[207.5], [83.2]]],
            [[[205.0], [80.5]], [[209.0], [85.5]]],
            [[[207.0], [82.4]], [[215.0], [87.4]]],
        ],
        dtype=torch.float64,
    )

    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "last_context_target": torch.tensor(
            [
                [[99.0], [49.0]],
                [[100.0], [49.5]],
                [[102.0], [49.0]],
                [[103.0], [51.0]],
                [[198.0], [79.0]],
                [[200.0], [79.5]],
                [[203.0], [80.0]],
                [[206.0], [81.0]],
            ],
            dtype=torch.float64,
        ),
        "channels": ["close"],
        "horizons": [1, 5],
        "sample_idx": torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=torch.long,
        ),
        "origin_idx": torch.tensor(
            [59, 74, 89, 104, 59, 74, 104, 119],
            dtype=torch.long,
        ),
        "output_space": "raw",
    }


def main() -> None:
    prediction_result = _prediction_result()
    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
    )

    metric_names = [
        RAW_PRICE_METRIC,
        SERIES_RETURN_METRIC,
    ]

    for metric_name in metric_names:
        if metric_name not in evaluator.available_metrics:
            raise AssertionError(
                f"Missing registered metric: {metric_name}"
            )

    ordinary = evaluator.evaluate(
        metrics=metric_names,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    y_pred = prediction_result["y_pred"]
    y_true = prediction_result["y_true"]
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)

    torch.testing.assert_close(
        ordinary[RAW_PRICE_METRIC],
        _expected_raw_correlation(y_pred, y_true),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        ordinary[SERIES_RETURN_METRIC],
        _expected_series_return_correlation(y_pred, y_true),
        rtol=1e-12,
        atol=1e-12,
    )

    pred_returns, true_returns, inferred_stride = (
        forecast_series_log_return_values(
            y_pred_raw=y_pred,
            y_true_raw=y_true,
            sample_idx=prediction_result["sample_idx"],
            origin_idx=prediction_result["origin_idx"],
        )
    )

    if inferred_stride != 15:
        raise AssertionError(
            f"Expected inferred stride 15, got {inferred_stride}."
        )

    excluded_rows = [0, 4, 6]
    included_rows = [1, 2, 3, 5, 7]

    if not torch.isnan(pred_returns[excluded_rows]).all():
        raise AssertionError(
            "Session openings and non-adjacent origin gaps must be NaN."
        )
    if not torch.isnan(true_returns[excluded_rows]).all():
        raise AssertionError(
            "True series returns used an invalid cross-session/gap pair."
        )
    if not torch.isfinite(pred_returns[included_rows]).all():
        raise AssertionError(
            "Valid within-session forecast returns are missing."
        )

    bootstrapped = evaluator.evaluate(
        metrics=metric_names,
        reduce_dims=(0, 2),
        bootstrap=True,
        n_bootstrap=200,
        bootstrap_seed=42,
    )

    for metric_name in metric_names:
        torch.testing.assert_close(
            bootstrapped[metric_name]["value"],
            ordinary[metric_name],
        )
        for summary_name in (
            "bootstrap_mean",
            "bootstrap_std",
            "ci_lower",
            "ci_upper",
        ):
            if not torch.isfinite(
                bootstrapped[metric_name][summary_name]
            ).all():
                raise AssertionError(
                    f"{metric_name} has non-finite {summary_name}."
                )

    metric_table = make_evaluation_table(
        metric_results=bootstrapped,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    ordinary_table = make_evaluation_table(
        metric_results=ordinary,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    summary = make_baseline_summary_table(
        models_to_display=["example"],
        namespace={
            "example_metric_table": ordinary_table,
        },
        metrics_to_display=metric_names,
    )

    expected_columns = {
        "Price Pearson",
        "Series-Return Pearson",
    }
    if set(summary.columns) != expected_columns:
        raise AssertionError(
            "Secondary temporal correlations are missing from the "
            f"summary table: {summary.columns.tolist()}"
        )

    if set(metric_table["metric"]) != set(metric_names):
        raise AssertionError(
            "Secondary temporal correlations are missing from the "
            "long-form evaluation table."
        )

    # Persistence has zero cumulative return at every origin, but its raw
    # forecast-price time series and the returns formed from that series
    # both vary through time and must therefore be defined.
    last_context = prediction_result["last_context_target"]
    assert isinstance(last_context, torch.Tensor)

    persistence_result = dict(prediction_result)
    persistence_result["y_pred"] = (
        last_context
        .unsqueeze(1)
        .expand_as(y_true)
        .clone()
    )

    persistence_metrics = ForecastEvaluator(
        prediction_result=persistence_result,
    ).evaluate(
        metrics=metric_names,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    for metric_name in metric_names:
        if not torch.isfinite(persistence_metrics[metric_name]).all():
            raise AssertionError(
                f"Persistence should have a defined {metric_name}."
            )

    # Backwards compatibility: raw-price correlation needs no window
    # metadata, while the return-series metric is registered only when the
    # required session/origin metadata are available.
    metadata_free = dict(prediction_result)
    metadata_free.pop("sample_idx")
    metadata_free.pop("origin_idx")
    metadata_free_evaluator = ForecastEvaluator(
        prediction_result=metadata_free,
    )

    if RAW_PRICE_METRIC not in metadata_free_evaluator.available_metrics:
        raise AssertionError(
            "Raw-price correlation should not require session metadata."
        )
    if SERIES_RETURN_METRIC in metadata_free_evaluator.available_metrics:
        raise AssertionError(
            "Series-return correlation must require session metadata."
        )

    print("Secondary temporal-correlation metric CPU test passed.")


if __name__ == "__main__":
    main()

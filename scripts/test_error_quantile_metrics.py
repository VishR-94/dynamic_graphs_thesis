"""CPU smoke test for cumulative-log-change median and P95 errors."""

from __future__ import annotations

import pandas as pd
import torch

from src.evaluation.metrics import ForecastEvaluator, reduce_quantile
from src.utils.metric_tables import (
    make_baseline_summary_table,
    make_evaluation_table,
)


def main() -> None:
    values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 100.0],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        reduce_quantile(values, 0.50),
        torch.tensor(3.0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        reduce_quantile(values, 0.95),
        torch.tensor(80.8, dtype=torch.float32),
    )

    log_errors = torch.tensor(
        [0.01, 0.02, 0.03, 0.04, 1.00],
        dtype=torch.float32,
    ).view(5, 1, 1, 1)

    prediction_result = {
        "y_pred": torch.exp(log_errors),
        "y_true": torch.ones_like(log_errors),
        "last_context_target": torch.ones(
            (5, 1, 1),
            dtype=torch.float32,
        ),
        "channels": ["close"],
        "horizons": [1],
        "sample_idx": torch.arange(5),
        "output_space": "raw",
    }

    evaluator = ForecastEvaluator(
        prediction_result=prediction_result
    )
    metric_names = [
        "cumulative_log_change_mae",
        "cumulative_log_change_median_absolute_error",
        "cumulative_log_change_p95_absolute_error",
    ]

    ordinary = evaluator.evaluate(
        metrics=metric_names,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    torch.testing.assert_close(
        ordinary["cumulative_log_change_mae"],
        torch.tensor([[0.22]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        ordinary[
            "cumulative_log_change_median_absolute_error"
        ],
        torch.tensor([[0.03]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        ordinary[
            "cumulative_log_change_p95_absolute_error"
        ],
        torch.tensor([[0.808]], dtype=torch.float32),
    )

    bootstrapped = evaluator.evaluate(
        metrics=metric_names,
        reduce_dims=(0, 2),
        bootstrap=True,
        n_bootstrap=50,
        bootstrap_seed=42,
    )

    if not torch.isnan(
        bootstrapped[
            "cumulative_log_change_median_absolute_error"
        ]["ci_lower"]
    ).all():
        raise AssertionError(
            "Median error should not have bootstrap intervals."
        )

    if not torch.isnan(
        bootstrapped[
            "cumulative_log_change_p95_absolute_error"
        ]["bootstrap_mean"]
    ).all():
        raise AssertionError(
            "P95 error should not have bootstrap summaries."
        )

    long_table = make_evaluation_table(
        metric_results=bootstrapped,
        horizons=[1],
        channels=["close"],
    )

    quantile_rows = long_table["metric"].isin(
        metric_names[1:]
    )
    if not pd.isna(
        long_table.loc[quantile_rows, "ci_lower"]
    ).all():
        raise AssertionError(
            "Quantile bootstrap columns should be NaN."
        )

    ordinary_table = make_evaluation_table(
        metric_results=ordinary,
        horizons=[1],
        channels=["close"],
    )
    summary = make_baseline_summary_table(
        models_to_display=["example"],
        namespace={
            "example_metric_table": ordinary_table,
        },
    )

    if "Log MedAE" not in summary.columns:
        raise AssertionError(
            "Headline table is missing Log MedAE."
        )
    if "Log P95 AE" not in summary.columns:
        raise AssertionError(
            "Headline table is missing Log P95 AE."
        )
    if "MASE" in summary.columns:
        raise AssertionError(
            "MASE should be omitted from the compact headline table."
        )

    print("Error-quantile metric CPU smoke test passed.")


if __name__ == "__main__":
    main()

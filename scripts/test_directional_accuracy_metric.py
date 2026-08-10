"""CPU contract test for cumulative-log-change directional accuracy."""

from __future__ import annotations

import torch

from src.evaluation.metrics import ForecastEvaluator
from src.utils.metric_tables import make_evaluation_table


def main() -> None:
    last = torch.full(
        (4, 3, 1),
        100.0,
        dtype=torch.float64,
    )

    true_changes = torch.tensor(
        [
            [[0.01, -0.01, 0.00], [0.02, -0.02, 0.01]],
            [[-0.01, 0.01, 0.00], [-0.02, 0.02, 0.01]],
            [[0.01, 0.01, -0.01], [0.02, 0.02, -0.01]],
            [[-0.01, -0.01, 0.01], [-0.02, -0.02, -0.01]],
        ],
        dtype=torch.float64,
    )

    predicted_changes = torch.tensor(
        [
            [[0.01, 0.01, 0.00], [0.02, -0.02, -0.01]],
            [[-0.01, 0.00, 0.01], [0.02, 0.02, 0.01]],
            [[-0.01, 0.01, -0.01], [0.02, -0.02, -0.01]],
            [[-0.01, 0.01, 0.00], [-0.02, -0.02, 0.01]],
        ],
        dtype=torch.float64,
    )

    prediction_result = {
        "y_pred": (
            last[:, None]
            * predicted_changes.exp().unsqueeze(-1)
        ),
        "y_true": (
            last[:, None]
            * true_changes.exp().unsqueeze(-1)
        ),
        "last_context_target": last,
        "channels": ["close"],
        "horizons": [1, 5],
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "output_space": "raw",
    }

    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
    )

    metric_name = (
        "cumulative_log_change_directional_accuracy"
    )

    ordinary = evaluator.evaluate(
        metrics=metric_name,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    expected = (
        torch.sign(predicted_changes)
        == torch.sign(true_changes)
    ).to(torch.float64).mean(dim=(0, 2)).reshape(2, 1)

    torch.testing.assert_close(
        ordinary[metric_name],
        expected,
    )

    # Explicitly verify the user's zero semantics.
    zero_semantics = ForecastEvaluator(
        prediction_result={
            "y_pred": torch.tensor(
                [[[[100.0], [100.0]]]],
                dtype=torch.float64,
            ),
            "y_true": torch.tensor(
                [[[[101.0], [100.0]]]],
                dtype=torch.float64,
            ),
            "last_context_target": torch.tensor(
                [[[100.0], [100.0]]],
                dtype=torch.float64,
            ),
            "channels": ["close"],
            "horizons": [1],
            "sample_idx": torch.tensor([0]),
            "output_space": "raw",
        }
    ).evaluate(
        metrics=metric_name,
        reduce_dims=(0, 2),
        bootstrap=False,
    )[metric_name]

    torch.testing.assert_close(
        zero_semantics,
        torch.tensor([[0.5]], dtype=torch.float64),
    )

    bootstrapped = evaluator.evaluate(
        metrics=metric_name,
        reduce_dims=(0, 2),
        bootstrap=True,
        n_bootstrap=100,
        bootstrap_seed=42,
    )

    torch.testing.assert_close(
        bootstrapped[metric_name]["value"],
        expected,
    )

    table = make_evaluation_table(
        metric_results=bootstrapped,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    if table["metric"].tolist() != [metric_name, metric_name]:
        raise AssertionError(
            "Directional accuracy is missing from the evaluation table."
        )

    print("Directional-accuracy metric CPU test passed.")


if __name__ == "__main__":
    main()

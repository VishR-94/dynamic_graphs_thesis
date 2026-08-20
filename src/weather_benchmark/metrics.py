from __future__ import annotations

"""Metrics matching Sonnet's executable weather evaluation code."""

from typing import Any

import numpy as np


def sonnet_linear_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mirror ``sonnet.utils.metrics.acc`` for the reported target series."""

    pred = np.asarray(predictions)
    true = np.asarray(targets)
    if pred.shape != true.shape:
        raise ValueError("predictions and targets must have identical shapes.")
    if pred.ndim == 3:
        pred = pred.astype(np.float32).reshape(pred.shape[0], -1)
        true = true.astype(np.float32).reshape(true.shape[0], -1)
    pred_anomaly = pred - pred.mean(axis=0)
    true_anomaly = true - true.mean(axis=0)
    numerator = np.sum(pred_anomaly * true_anomaly, axis=0)
    denominator = np.sqrt(np.sum(pred_anomaly**2, axis=0)) * np.sqrt(
        np.sum(true_anomaly**2, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return float(np.mean(values))


def sonnet_weather_smape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mirror the current Sonnet repository implementation exactly.

    The executable code sets ``a = min(targets) + 30`` and measures absolute
    distances from ``a`` in the denominator.  This is intentionally preserved
    even though it is not algebraically identical to Equation S1 in the paper.
    """

    pred = np.asarray(predictions)
    true = np.asarray(targets)
    if pred.shape != true.shape:
        raise ValueError("predictions and targets must have identical shapes.")
    offset = np.min(true) + 30.0
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 2.0 * np.abs(pred - true) / (
            np.abs(pred - offset) + np.abs(true - offset)
        )
    return float(np.mean(values) * 100.0)


def sonnet_reported_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Return the three weather metrics reported by Sonnet."""

    pred = np.asarray(predictions)
    true = np.asarray(targets)
    if pred.shape != true.shape:
        raise ValueError("predictions and targets must have identical shapes.")
    return {
        "mae": float(np.mean(np.abs(pred - true))),
        "r": sonnet_linear_correlation(pred, true),
        "smape": sonnet_weather_smape(pred, true),
    }


def weather_metric_payload(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
    central_node_index: int,
) -> dict[str, Any]:
    """Compute final-target headline metrics and supplementary sequence metrics.

    Inputs use ``[W, H, N, 1]``.  Headline values are evaluated at the final
    output position and central node, exactly as in the Sonnet weather tables.
    """

    pred = np.asarray(predictions)
    true = np.asarray(targets)
    if pred.shape != true.shape or pred.ndim != 4 or pred.shape[-1] != 1:
        raise ValueError("Expected matching [W,H,N,1] prediction tensors.")
    node = int(central_node_index)
    final_pred = pred[:, -1, node, 0]
    final_true = true[:, -1, node, 0]
    sequence_pred = pred[:, :, node, 0]
    sequence_true = true[:, :, node, 0]
    return {
        "reported": sonnet_reported_metrics(final_pred, final_true),
        "supplementary_sequence": sonnet_reported_metrics(
            sequence_pred, sequence_true
        ),
        "reported_scope": {
            "forecast_position": int(pred.shape[1] - 1),
            "forecast_step": int(pred.shape[1]),
            "central_node_index": node,
            "windows": int(pred.shape[0]),
            "unit": "Kelvin",
        },
    }

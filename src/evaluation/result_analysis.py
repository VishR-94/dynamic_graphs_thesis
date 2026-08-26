from __future__ import annotations

"""Conditional analysis of saved financial forecast results.

This module is deliberately read-only with respect to model artefacts.  It
loads saved raw-space prediction dictionaries, verifies that every model is
aligned to the same test observations, and provides stock-, volatility-,
time-of-day-, and daily-level performance diagnostics.

The module does not modify the existing evaluator or any training code.  All
plots return both the underlying tidy table and the Matplotlib figure so that
notebook outputs remain auditable and exportable.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
import json
import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import torch

from src.evaluation.dynamic_graph_evaluation import load_evaluation_artifacts
from src.evaluation.metrics import ForecastEvaluator, compute_mase_scale
from src.evaluation.prediction_transforms import raw_to_cumulative_log_change
from src.utils.company_profiles import make_asset_sector_mapping
from src.utils.metric_tables import DEFAULT_METRIC_DISPLAY_NAMES


TensorDict = dict[str, Any]
MetricDirection = Literal["lower", "higher", "target"]
OrderMode = Literal["volatility", "sector"]
YAxisMode = Literal["adaptive", "zero"]


_REQUIRED_PREDICTION_KEYS = {
    "y_pred",
    "y_true",
    "last_context_target",
    "channels",
    "horizons",
    "sample_idx",
    "origin_idx",
    "target_indices",
}


@dataclass(frozen=True)
class PredictionSource:
    """Describe where one model's saved prediction result should be loaded.

    Parameters
    ----------
    path:
        Path to either a ``.pt`` file or an evaluation run directory.
    kind:
        ``"auto"`` inspects the path.  ``"prediction_file"`` loads a PyTorch
        file and searches it for a prediction dictionary.  ``"evaluation_run"``
        uses :func:`load_evaluation_artifacts`.
    split / policy:
        Used only for evaluation-run directories.
    nested_key:
        Optional explicit key for files that wrap the prediction dictionary,
        e.g. ``"prediction_result"``.
    """

    path: str | Path
    kind: Literal["auto", "prediction_file", "evaluation_run"] = "auto"
    split: str = "test"
    policy: str = "best"
    nested_key: str | None = None


@dataclass(frozen=True)
class StockMetricSpec:
    """Metadata for a stock-level metric used in tables and plots."""

    name: str
    display_name: str
    direction: MetricDirection
    description: str
    reference_value: float | None = None
    percentage: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


STOCK_METRIC_SPECS: dict[str, StockMetricSpec] = {
    "cumulative_log_change_mae": StockMetricSpec(
        name="cumulative_log_change_mae",
        display_name="Cumulative-log-change MAE",
        direction="lower",
        description=(
            "Mean absolute error between predicted and realised cumulative "
            "log changes from the forecast origin."
        ),
        lower_bound=0.0,
    ),
    "cumulative_log_change_median_absolute_error": StockMetricSpec(
        name="cumulative_log_change_median_absolute_error",
        display_name="Cumulative-log-change median absolute error",
        direction="lower",
        description="Median absolute cumulative-log-change error.",
        lower_bound=0.0,
    ),
    "cumulative_log_change_p95_absolute_error": StockMetricSpec(
        name="cumulative_log_change_p95_absolute_error",
        display_name="Cumulative-log-change P95 absolute error",
        direction="lower",
        description="95th percentile cumulative-log-change absolute error.",
        lower_bound=0.0,
    ),
    "mase": StockMetricSpec(
        name="mase",
        display_name="MASE",
        direction="lower",
        description=(
            "Raw-space absolute error divided by the training-derived "
            "one-step naive scale."
        ),
        reference_value=1.0,
        lower_bound=0.0,
    ),
    "relative_mae_vs_persistence": StockMetricSpec(
        name="relative_mae_vs_persistence",
        display_name="Relative MAE vs persistence",
        direction="lower",
        description=(
            "Raw-price MAE divided by same-horizon persistence MAE.  Values "
            "below one beat persistence."
        ),
        reference_value=1.0,
        lower_bound=0.0,
    ),
    "mae_difference_vs_persistence": StockMetricSpec(
        name="mae_difference_vs_persistence",
        display_name="CLG-MAE minus persistence",
        direction="lower",
        description=(
            "Cumulative-log-change MAE minus persistence CLG-MAE.  Negative "
            "values favour the model."
        ),
        reference_value=0.0,
    ),
    "persistence_win_rate": StockMetricSpec(
        name="persistence_win_rate",
        display_name="Persistence win rate",
        direction="higher",
        description=(
            "Fraction of raw-price forecast elements with smaller absolute "
            "error than persistence; ties receive 0.5."
        ),
        reference_value=0.5,
        percentage=True,
        lower_bound=0.0,
        upper_bound=1.0,
    ),
    "cumulative_log_change_directional_accuracy": StockMetricSpec(
        name="cumulative_log_change_directional_accuracy",
        display_name="Directional accuracy",
        direction="higher",
        description="Fraction of cumulative-log-change signs predicted correctly.",
        reference_value=0.5,
        percentage=True,
        lower_bound=0.0,
        upper_bound=1.0,
    ),
    "cumulative_log_change_pearson_correlation": StockMetricSpec(
        name="cumulative_log_change_pearson_correlation",
        display_name="Cumulative-log-change Pearson",
        direction="higher",
        description=(
            "Temporal Pearson correlation between predicted and realised "
            "cumulative log changes for each stock."
        ),
        reference_value=0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
    ),
    "raw_price_temporal_pearson_correlation": StockMetricSpec(
        name="raw_price_temporal_pearson_correlation",
        display_name="Price Pearson",
        direction="higher",
        description=(
            "Temporal Pearson correlation between predicted and realised raw "
            "target-price series for each stock."
        ),
        reference_value=0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
    ),
    "forecast_series_log_return_temporal_pearson_correlation": StockMetricSpec(
        name="forecast_series_log_return_temporal_pearson_correlation",
        display_name="Series-return Pearson",
        direction="higher",
        description=(
            "Temporal Pearson correlation between within-session log changes "
            "of the horizon-aligned predicted and realised price series."
        ),
        reference_value=0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
    ),
    "cumulative_log_change_movement_magnitude_ratio": StockMetricSpec(
        name="cumulative_log_change_movement_magnitude_ratio",
        display_name="Movement-magnitude ratio",
        direction="target",
        description=(
            "Mean absolute predicted cumulative log change divided by mean "
            "absolute realised cumulative log change."
        ),
        reference_value=1.0,
        lower_bound=0.0,
    ),
    "cumulative_log_change_temporal_absolute_correlation": StockMetricSpec(
        name="cumulative_log_change_temporal_absolute_correlation",
        display_name="Absolute-return correlation",
        direction="higher",
        description=(
            "Temporal Pearson correlation between absolute predicted and "
            "absolute realised cumulative log changes for each stock."
        ),
        reference_value=0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
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
    "final_model": "GraphTCN",
    "GraphTCN": "GraphTCN",
    "Persistence": "Persistence",
}


_TIME_BUCKET_ORDER = ("Morning", "Midday", "Late session")
# Retained for backwards-compatible exports. New plots use the configurable
# equal-frequency bucket helpers below.
_VOLATILITY_TERCILE_ORDER = ("Low", "Medium", "High")


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_prediction_result(value: Any) -> bool:
    return isinstance(value, Mapping) and _REQUIRED_PREDICTION_KEYS.issubset(value)


def _find_prediction_result(value: Any, *, path_label: str) -> TensorDict:
    """Find one prediction dictionary inside a loaded PyTorch object."""

    if _is_prediction_result(value):
        return dict(value)

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{path_label} does not contain a prediction-result mapping."
        )

    preferred_keys = (
        "prediction_result",
        "test_prediction_result",
        "predictions",
        "test_predictions",
    )
    candidates: list[tuple[str, TensorDict]] = []

    for key in preferred_keys:
        child = value.get(key)
        if _is_prediction_result(child):
            candidates.append((key, dict(child)))

    if not candidates:
        for key, child in value.items():
            if _is_prediction_result(child):
                candidates.append((str(key), dict(child)))

    if len(candidates) == 1:
        return candidates[0][1]

    if len(candidates) > 1:
        raise ValueError(
            f"{path_label} contains multiple prediction dictionaries: "
            f"{[key for key, _ in candidates]}. Supply nested_key explicitly."
        )

    raise ValueError(
        f"No object with required keys {sorted(_REQUIRED_PREDICTION_KEYS)} "
        f"was found in {path_label}."
    )


def load_prediction_source(source: PredictionSource | str | Path | Mapping[str, Any]) -> TensorDict:
    """Load one saved raw-space prediction dictionary."""

    if _is_prediction_result(source):
        return dict(source)

    if isinstance(source, Mapping):
        return _find_prediction_result(source, path_label="in-memory source")

    if isinstance(source, (str, Path)):
        source = PredictionSource(path=source)

    if not isinstance(source, PredictionSource):
        raise TypeError(
            "Each model source must be a PredictionSource, path, or prediction mapping."
        )

    path = Path(source.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    kind = source.kind
    if kind == "auto":
        kind = "evaluation_run" if path.is_dir() else "prediction_file"

    if kind == "evaluation_run":
        if not path.is_dir():
            raise NotADirectoryError(path)
        artifacts = load_evaluation_artifacts(
            path,
            split=source.split,
            policy=source.policy,
        )
        return dict(artifacts.prediction_result)

    if kind != "prediction_file":
        raise ValueError(f"Unsupported prediction source kind: {kind!r}.")

    if not path.is_file():
        raise FileNotFoundError(path)

    loaded = _safe_torch_load(path)
    if source.nested_key is not None:
        if not isinstance(loaded, Mapping) or source.nested_key not in loaded:
            raise KeyError(
                f"{path} does not contain nested key {source.nested_key!r}."
            )
        loaded = loaded[source.nested_key]
        if not _is_prediction_result(loaded):
            raise ValueError(
                f"{path}[{source.nested_key!r}] is not a prediction result."
            )
        return dict(loaded)

    return _find_prediction_result(loaded, path_label=str(path))


def _as_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


def _parse_grain(grain: str) -> pd.Timedelta:
    value = str(grain).strip().lower()
    if value.endswith("min"):
        return pd.Timedelta(minutes=int(value[:-3]))
    if value.endswith("m"):
        return pd.Timedelta(minutes=int(value[:-1]))
    if value.endswith("h"):
        return pd.Timedelta(hours=int(value[:-1]))
    raise ValueError(f"Unsupported grain: {grain!r}.")


def _resolve_day_timestamps(
    *,
    sample_timestamps: Any,
    day: Any,
    num_bars: int,
    split: Mapping[str, Any],
) -> pd.DatetimeIndex:
    """Recover cleaned bar-close timestamps for one session."""

    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(sample_timestamps))
        if len(timestamps) == num_bars and not timestamps.isna().any():
            return timestamps
    except (TypeError, ValueError):
        pass

    interval = _parse_grain(split["grain"])
    first_bar_close = (
        pd.Timestamp(f"{pd.Timestamp(day).date()} {split['market_open']}")
        + interval
    )
    return pd.date_range(start=first_bar_close, periods=num_bars, freq=interval)


def _pearson_1d(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-15) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    x_valid = np.asarray(x[valid], dtype=np.float64)
    y_valid = np.asarray(y[valid], dtype=np.float64)
    x_centred = x_valid - x_valid.mean()
    y_centred = y_valid - y_valid.mean()
    denominator = math.sqrt(
        float(np.dot(x_centred, x_centred))
        * float(np.dot(y_centred, y_centred))
    )
    if not np.isfinite(denominator) or denominator <= eps:
        return float("nan")
    return float(np.dot(x_centred, y_centred) / denominator)


def _assetwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson through rows for each [H, N] column pair.

    ``x`` and ``y`` must have shape ``[B, H, N]``.  The result has shape
    ``[H, N]``.
    """

    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(
            f"Expected aligned [B,H,N] arrays, got {x.shape} and {y.shape}."
        )
    _, num_horizons, num_assets = x.shape
    result = np.full((num_horizons, num_assets), np.nan, dtype=np.float64)
    for horizon_idx in range(num_horizons):
        for asset_idx in range(num_assets):
            result[horizon_idx, asset_idx] = _pearson_1d(
                x[:, horizon_idx, asset_idx],
                y[:, horizon_idx, asset_idx],
            )
    return result


def _assign_equal_frequency_buckets(
    values: pd.Series,
    *,
    num_buckets: int,
) -> pd.Series:
    """Assign stable equal-frequency volatility buckets numbered low to high.

    Bucket ``1`` always contains the lowest-volatility stocks and bucket
    ``num_buckets`` the highest-volatility stocks. Ties are resolved using the
    existing stock order, making the assignment deterministic and ensuring
    bucket sizes differ by at most one stock.
    """

    if values.isna().any():
        raise ValueError("Volatility values contain missing entries.")
    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int):
        raise TypeError("num_buckets must be an integer.")
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1.")
    if num_buckets > len(values):
        raise ValueError(
            "num_buckets cannot exceed the number of available stocks "
            f"({len(values)})."
        )

    order = np.argsort(values.to_numpy(dtype=np.float64), kind="stable")
    ordered_buckets = (
        np.floor(np.arange(len(values), dtype=np.float64) * num_buckets / len(values))
        .astype(np.int64)
        + 1
    )
    assigned = np.empty(len(values), dtype=np.int64)
    assigned[order] = ordered_buckets
    return pd.Series(assigned, index=values.index, dtype="int64")


def _volatility_bucket_label(bucket: int, *, num_buckets: int) -> str:
    if num_buckets == 1:
        return "Bucket 1 (all stocks)"
    if bucket == 1:
        return "Bucket 1 (lowest)"
    if bucket == num_buckets:
        return f"Bucket {num_buckets} (highest)"
    return f"Bucket {bucket}"


def _with_volatility_buckets(
    characteristics: pd.DataFrame,
    *,
    num_buckets: int,
) -> pd.DataFrame:
    result = characteristics.copy()
    result["volatility_bucket"] = _assign_equal_frequency_buckets(
        result["test_median_realised_volatility"],
        num_buckets=num_buckets,
    )
    result["volatility_bucket_label"] = [
        _volatility_bucket_label(int(bucket), num_buckets=num_buckets)
        for bucket in result["volatility_bucket"]
    ]
    result["num_volatility_buckets"] = int(num_buckets)
    return result


def _assign_equal_frequency_terciles(values: pd.Series) -> pd.Series:
    """Assign the legacy low/medium/high field used by existing exports."""

    bucket = _assign_equal_frequency_buckets(values, num_buckets=3)
    mapping = {1: "Low", 2: "Medium", 3: "High"}
    return bucket.map(mapping).astype(str)


def _format_model_name(name: str) -> str:
    return DEFAULT_MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _normalise_model_names(
    model_names: str | Sequence[str] | None,
    *,
    available: Sequence[str],
) -> list[str]:
    if model_names is None:
        return list(available)
    if isinstance(model_names, str):
        names = [model_names]
    else:
        names = [str(value) for value in model_names]
    if not names:
        raise ValueError("At least one model must be selected.")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Available models: {list(available)}."
        )
    if len(set(names)) != len(names):
        raise ValueError("Model names must not be duplicated.")
    return names


def _normalise_horizons(
    horizons: int | Sequence[int] | None,
    *,
    available: Sequence[int],
) -> list[int]:
    if horizons is None:
        values = [int(value) for value in available]
    elif isinstance(horizons, (int, np.integer)):
        values = [int(horizons)]
    else:
        values = [int(value) for value in horizons]
    if not values:
        raise ValueError("At least one horizon must be selected.")
    missing = [value for value in values if value not in available]
    if missing:
        raise ValueError(
            f"Unknown horizons: {missing}. Available horizons: {list(available)}."
        )
    if len(set(values)) != len(values):
        raise ValueError("Horizon values must not be duplicated.")
    return values


def _time_to_minutes(value: str) -> int:
    timestamp = pd.Timestamp(f"2000-01-01 {value}")
    return int(timestamp.hour * 60 + timestamp.minute)


def _time_bucket_labels(
    target_timestamps: pd.DatetimeIndex,
    *,
    morning_cutoff: str,
    late_session_cutoff: str,
) -> np.ndarray:
    morning_minutes = _time_to_minutes(morning_cutoff)
    late_minutes = _time_to_minutes(late_session_cutoff)
    if morning_minutes >= late_minutes:
        raise ValueError("morning_cutoff must be earlier than late_session_cutoff.")
    target_minutes = target_timestamps.hour * 60 + target_timestamps.minute
    return np.where(
        target_minutes < morning_minutes,
        "Morning",
        np.where(target_minutes < late_minutes, "Midday", "Late session"),
    )


class FinancialResultAnalysis:
    """Aligned saved forecasts plus conditional performance diagnostics."""

    def __init__(
        self,
        *,
        prediction_results: Mapping[str, Mapping[str, Any]],
        train_split: Mapping[str, Any],
        val_split: Mapping[str, Any],
        test_split: Mapping[str, Any],
        company_profiles_path: str | Path | None = None,
        reference_model: str | None = None,
    ) -> None:
        if not prediction_results:
            raise ValueError("prediction_results cannot be empty.")

        self.train_split = dict(train_split)
        self.val_split = dict(val_split)
        self.test_split = dict(test_split)
        self.company_profiles_path = company_profiles_path

        self.prediction_results: dict[str, TensorDict] = {
            str(name): self._normalise_prediction_result(str(name), result)
            for name, result in prediction_results.items()
        }

        self.model_names = tuple(self.prediction_results)
        if reference_model is None:
            reference_model = self.model_names[0]
        if reference_model not in self.prediction_results:
            raise ValueError(
                f"reference_model {reference_model!r} is not in {self.model_names}."
            )
        self.reference_model = reference_model
        self._validate_alignment()

        reference = self.prediction_results[self.reference_model]
        self.assets = tuple(reference["asset_cols"])
        self.horizons = tuple(int(value) for value in reference["horizons"])
        self.channels = tuple(str(value) for value in reference["channels"])
        if "close" not in self.channels:
            raise ValueError(
                "Result analysis currently requires a 'close' target channel."
            )
        self.close_channel_idx = self.channels.index("close")
        self.sample_idx = reference["sample_idx"]
        self.origin_idx = reference["origin_idx"]
        self.target_indices = reference["target_indices"]

        self._target_timestamp_matrix = self._build_target_timestamp_matrix()
        self._per_stock_metric_cache: pd.DataFrame | None = None
        self._adf_cache: pd.DataFrame | None = None
        self._stock_characteristics_cache: pd.DataFrame | None = None

    @classmethod
    def from_sources(
        cls,
        *,
        model_sources: Mapping[
            str,
            PredictionSource | str | Path | Mapping[str, Any],
        ],
        train_split: Mapping[str, Any],
        val_split: Mapping[str, Any],
        test_split: Mapping[str, Any],
        company_profiles_path: str | Path | None = None,
        reference_model: str | None = None,
    ) -> "FinancialResultAnalysis":
        loaded = {
            str(name): load_prediction_source(source)
            for name, source in model_sources.items()
        }
        return cls(
            prediction_results=loaded,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            company_profiles_path=company_profiles_path,
            reference_model=reference_model,
        )

    def _normalise_prediction_result(
        self,
        model_name: str,
        value: Mapping[str, Any],
    ) -> TensorDict:
        missing = _REQUIRED_PREDICTION_KEYS.difference(value)
        if missing:
            raise KeyError(
                f"{model_name} is missing prediction keys: {sorted(missing)}."
            )

        result = dict(value)
        result["y_pred"] = _as_cpu_tensor(result["y_pred"], dtype=torch.float64)
        result["y_true"] = _as_cpu_tensor(result["y_true"], dtype=torch.float64)
        result["last_context_target"] = _as_cpu_tensor(
            result["last_context_target"], dtype=torch.float64
        )
        result["sample_idx"] = _as_cpu_tensor(result["sample_idx"], dtype=torch.long)
        result["origin_idx"] = _as_cpu_tensor(result["origin_idx"], dtype=torch.long)
        result["target_indices"] = _as_cpu_tensor(
            result["target_indices"], dtype=torch.long
        )
        result["channels"] = [str(value) for value in result["channels"]]
        result["horizons"] = [int(value) for value in result["horizons"]]
        result["asset_cols"] = list(
            result.get("asset_cols", self.test_split["asset_cols"])
        )

        y_pred = result["y_pred"]
        y_true = result["y_true"]
        last = result["last_context_target"]
        if y_pred.ndim != 4:
            raise ValueError(
                f"{model_name} y_pred must have shape [B,H,N,C], got {y_pred.shape}."
            )
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"{model_name} y_pred/y_true shapes differ: "
                f"{y_pred.shape} vs {y_true.shape}."
            )
        if last.shape != (y_pred.shape[0], y_pred.shape[2], y_pred.shape[3]):
            raise ValueError(
                f"{model_name} last_context_target is not aligned with y_pred."
            )
        if result["sample_idx"].shape != (y_pred.shape[0],):
            raise ValueError(f"{model_name} sample_idx has an invalid shape.")
        if result["origin_idx"].shape != (y_pred.shape[0],):
            raise ValueError(f"{model_name} origin_idx has an invalid shape.")
        if result["target_indices"].shape != (y_pred.shape[0], y_pred.shape[1]):
            raise ValueError(f"{model_name} target_indices has an invalid shape.")
        if len(result["channels"]) != y_pred.shape[3]:
            raise ValueError(f"{model_name} channel metadata is not aligned.")
        if len(result["horizons"]) != y_pred.shape[1]:
            raise ValueError(f"{model_name} horizon metadata is not aligned.")
        if len(result["asset_cols"]) != y_pred.shape[2]:
            raise ValueError(f"{model_name} asset metadata is not aligned.")
        if not torch.isfinite(y_pred).all():
            raise ValueError(f"{model_name} predictions contain NaN or infinity.")
        if not torch.isfinite(y_true).all():
            raise ValueError(f"{model_name} targets contain NaN or infinity.")
        return result

    def _validate_alignment(self) -> None:
        reference = self.prediction_results[self.reference_model]
        test_assets = list(self.test_split["asset_cols"])
        if reference["asset_cols"] != test_assets:
            raise ValueError(
                f"Reference model assets are not aligned with the test split."
            )

        for model_name, result in self.prediction_results.items():
            if result["asset_cols"] != reference["asset_cols"]:
                raise ValueError(f"{model_name} asset order differs from reference.")
            if result["channels"] != reference["channels"]:
                raise ValueError(f"{model_name} channels differ from reference.")
            if result["horizons"] != reference["horizons"]:
                raise ValueError(f"{model_name} horizons differ from reference.")
            for key in ("sample_idx", "origin_idx", "target_indices"):
                if not torch.equal(result[key], reference[key]):
                    raise ValueError(f"{model_name} {key} differs from reference.")
            if not torch.equal(result["y_true"], reference["y_true"]):
                max_difference = float(
                    torch.max(torch.abs(result["y_true"] - reference["y_true"])).item()
                )
                raise ValueError(
                    f"{model_name} y_true differs from reference; max diff={max_difference}."
                )
            if not torch.equal(
                result["last_context_target"], reference["last_context_target"]
            ):
                max_difference = float(
                    torch.max(
                        torch.abs(
                            result["last_context_target"]
                            - reference["last_context_target"]
                        )
                    ).item()
                )
                raise ValueError(
                    f"{model_name} last_context_target differs from reference; "
                    f"max diff={max_difference}."
                )

        max_sample_idx = int(reference["sample_idx"].max().item())
        if max_sample_idx >= len(self.test_split["samples"]):
            raise ValueError(
                "Prediction sample_idx exceeds the available test sessions."
            )

    def _build_target_timestamp_matrix(self) -> np.ndarray:
        num_windows = int(self.sample_idx.numel())
        num_horizons = len(self.horizons)
        result = np.empty((num_windows, num_horizons), dtype="datetime64[ns]")

        for sample_value in torch.unique(self.sample_idx, sorted=True).tolist():
            sample = int(sample_value)
            x_day, sample_timestamps, day = self.test_split["samples"][sample]
            timestamps = _resolve_day_timestamps(
                sample_timestamps=sample_timestamps,
                day=day,
                num_bars=int(torch.as_tensor(x_day).shape[0]),
                split=self.test_split,
            )
            rows = torch.where(self.sample_idx == sample)[0]
            indices = self.target_indices.index_select(0, rows).numpy()
            if (indices < 0).any() or (indices >= len(timestamps)).any():
                raise ValueError(
                    f"Target indices for sample {sample} fall outside its session."
                )
            result[rows.numpy(), :] = timestamps.values[indices]
        return result

    @property
    def target_timestamps(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._target_timestamp_matrix.copy(),
            columns=pd.Index(self.horizons, name="horizon"),
        )

    @property
    def session_dates(self) -> pd.Series:
        dates = [
            pd.Timestamp(self.test_split["samples"][int(idx)][2]).normalize()
            for idx in self.sample_idx.tolist()
        ]
        return pd.Series(dates, name="session_date")

    def alignment_manifest(self) -> pd.DataFrame:
        rows = []
        for model_name, result in self.prediction_results.items():
            rows.append(
                {
                    "model": model_name,
                    "display_name": _format_model_name(model_name),
                    "windows": int(result["y_pred"].shape[0]),
                    "horizons": tuple(result["horizons"]),
                    "assets": int(result["y_pred"].shape[2]),
                    "channels": tuple(result["channels"]),
                    "first_session": self.session_dates.iloc[0].date().isoformat(),
                    "last_session": self.session_dates.iloc[-1].date().isoformat(),
                    "nonpositive_predictions": int((result["y_pred"] <= 0).sum().item()),
                }
            )
        return pd.DataFrame(rows)

    def available_stock_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": spec.name,
                    "display_name": spec.display_name,
                    "direction": spec.direction,
                    "reference_value": spec.reference_value,
                    "description": spec.description,
                }
                for spec in STOCK_METRIC_SPECS.values()
            ]
        )

    def available_group_metrics(self) -> tuple[str, ...]:
        reference_result = self.prediction_results[self.reference_model]
        evaluator = ForecastEvaluator(
            prediction_result=reference_result,
            train_split=self.train_split,
        )
        return tuple(
            dict.fromkeys(
                (
                    *evaluator.available_metrics,
                    "mae_difference_vs_persistence",
                )
            )
        )

    def _daily_close_matrix(self, split: Mapping[str, Any]) -> tuple[pd.DatetimeIndex, np.ndarray]:
        channels = list(split["channels"])
        if "close" not in channels:
            raise ValueError("Split does not contain a close channel.")
        close_idx = channels.index("close")
        dates: list[pd.Timestamp] = []
        closes: list[np.ndarray] = []
        for x_day, _, day in split["samples"]:
            x_tensor = torch.as_tensor(x_day).detach().cpu().to(torch.float64)
            if x_tensor.ndim != 3 or x_tensor.shape[1] != len(self.assets):
                raise ValueError("Split daily tensor has an unexpected shape.")
            dates.append(pd.Timestamp(day).normalize())
            closes.append(x_tensor[:, :, close_idx].numpy())
        return pd.DatetimeIndex(dates), np.stack(closes, axis=0)

    @cached_property
    def daily_stock_realised_volatility(self) -> pd.DataFrame:
        dates, close = self._daily_close_matrix(self.test_split)
        log_close = np.log(np.clip(close, 1e-12, None))
        returns = np.diff(log_close, axis=1)
        rv = np.sqrt(np.sum(returns * returns, axis=1))
        frame = pd.DataFrame(rv, index=dates, columns=self.assets)
        frame.index.name = "session_date"
        return frame

    @cached_property
    def split_daily_market_volatility(self) -> pd.DataFrame:
        frames = []
        for split_name, split in (
            ("Train", self.train_split),
            ("Validation", self.val_split),
            ("Test", self.test_split),
        ):
            dates, close = self._daily_close_matrix(split)
            log_close = np.log(np.clip(close, 1e-12, None))
            returns = np.diff(log_close, axis=1)
            stock_rv = np.sqrt(np.sum(returns * returns, axis=1))
            market_rv = np.median(stock_rv, axis=1)
            frames.append(
                pd.DataFrame(
                    {
                        "split": split_name,
                        "session_date": dates,
                        "market_realised_volatility": market_rv,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    @property
    def adf_diagnostics(self) -> pd.DataFrame:
        if self._adf_cache is not None:
            return self._adf_cache.copy()

        dates, close = self._daily_close_matrix(self.test_split)
        daily_close = close[:, -1, :]
        log_daily_close = np.log(np.clip(daily_close, 1e-12, None))
        rows = []
        for asset_idx, ticker in enumerate(self.assets):
            series = log_daily_close[:, asset_idx]
            try:
                statistic, pvalue, used_lag, nobs, critical_values, icbest = adfuller(
                    series,
                    regression="c",
                    autolag="AIC",
                )
                row = {
                    "ticker": ticker,
                    "adf_statistic": float(statistic),
                    "adf_pvalue": float(pvalue),
                    "used_lag": int(used_lag),
                    "nobs": int(nobs),
                    "icbest": float(icbest),
                    "critical_1pct": float(critical_values["1%"]),
                    "critical_5pct": float(critical_values["5%"]),
                    "critical_10pct": float(critical_values["10%"]),
                    "reject_unit_root_1pct": bool(pvalue < 0.01),
                    "reject_unit_root_5pct": bool(pvalue < 0.05),
                    "reject_unit_root_10pct": bool(pvalue < 0.10),
                    "error": None,
                }
            except Exception as exc:  # statsmodels raises several data-specific errors
                row = {
                    "ticker": ticker,
                    "adf_statistic": np.nan,
                    "adf_pvalue": np.nan,
                    "used_lag": np.nan,
                    "nobs": len(dates),
                    "icbest": np.nan,
                    "critical_1pct": np.nan,
                    "critical_5pct": np.nan,
                    "critical_10pct": np.nan,
                    "reject_unit_root_1pct": False,
                    "reject_unit_root_5pct": False,
                    "reject_unit_root_10pct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
        self._adf_cache = pd.DataFrame(rows)
        return self._adf_cache.copy()

    @property
    def stock_characteristics(self) -> pd.DataFrame:
        if self._stock_characteristics_cache is not None:
            return self._stock_characteristics_cache.copy()

        median_rv = self.daily_stock_realised_volatility.median(axis=0)
        characteristics = pd.DataFrame(
            {
                "ticker": list(self.assets),
                "test_median_realised_volatility": [
                    float(median_rv[ticker]) for ticker in self.assets
                ],
            }
        )
        characteristics["volatility_tercile"] = _assign_equal_frequency_terciles(
            characteristics["test_median_realised_volatility"]
        )

        sectors = make_asset_sector_mapping(
            self.assets,
            company_profiles_path=self.company_profiles_path,
        ).rename(columns={"Ticker": "ticker", "Sector": "sector"})
        characteristics = characteristics.merge(
            sectors,
            on="ticker",
            how="left",
            validate="one_to_one",
        )
        characteristics = characteristics.merge(
            self.adf_diagnostics,
            on="ticker",
            how="left",
            validate="one_to_one",
        )
        self._stock_characteristics_cache = characteristics
        return characteristics.copy()

    def persistence_strength_summary(self) -> pd.DataFrame:
        characteristics = self.stock_characteristics
        adf_valid = characteristics["adf_pvalue"].notna()
        rows = [
            {
                "diagnostic": "Stocks with valid ADF test",
                "value": int(adf_valid.sum()),
            },
            {
                "diagnostic": "Fail to reject unit root at 5%",
                "value": int(
                    (adf_valid & ~characteristics["reject_unit_root_5pct"]).sum()
                ),
            },
            {
                "diagnostic": "Fraction failing to reject unit root at 5%",
                "value": float(
                    (adf_valid & ~characteristics["reject_unit_root_5pct"]).sum()
                    / max(int(adf_valid.sum()), 1)
                ),
            },
            {
                "diagnostic": "Median ADF p-value",
                "value": float(characteristics["adf_pvalue"].median()),
            },
            {
                "diagnostic": "Median test-period stock realised volatility",
                "value": float(
                    characteristics["test_median_realised_volatility"].median()
                ),
            },
        ]
        return pd.DataFrame(rows)

    def _reference_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference = self.prediction_results[self.reference_model]
        true_raw = reference["y_true"][..., self.close_channel_idx].numpy()
        last_raw = reference["last_context_target"][..., self.close_channel_idx].numpy()
        persistence_raw = np.repeat(last_raw[:, None, :], len(self.horizons), axis=1)
        return true_raw, last_raw, persistence_raw

    def _forecast_series_returns(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return adjacent within-session series changes and source-row mask.

        ``values`` has shape ``[B,H,N]``.  The returned array has shape
        ``[K,H,N]`` where K is the number of adjacent origin pairs.  The second
        result contains the row indices of each later observation.
        """

        if values.ndim != 3:
            raise ValueError("values must have shape [B,H,N].")
        sample_idx = self.sample_idx.numpy()
        origin_idx = self.origin_idx.numpy()
        unique_diffs = []
        for sample in np.unique(sample_idx):
            origins = np.sort(origin_idx[sample_idx == sample])
            if len(origins) > 1:
                unique_diffs.extend(np.diff(origins).tolist())
        positive_diffs = [int(value) for value in unique_diffs if value > 0]
        if not positive_diffs:
            raise ValueError("Could not infer a positive forecast-origin stride.")
        stride = int(pd.Series(positive_diffs).mode().iloc[0])

        previous_rows: list[int] = []
        current_rows: list[int] = []
        for sample in np.unique(sample_idx):
            rows = np.flatnonzero(sample_idx == sample)
            order = np.argsort(origin_idx[rows], kind="stable")
            rows = rows[order]
            if len(rows) < 2:
                continue
            adjacent = np.diff(origin_idx[rows]) == stride
            previous_rows.extend(rows[:-1][adjacent].tolist())
            current_rows.extend(rows[1:][adjacent].tolist())

        if not current_rows:
            empty = np.empty((0, values.shape[1], values.shape[2]), dtype=np.float64)
            return empty, np.empty((0,), dtype=np.int64)

        previous = np.asarray(previous_rows, dtype=np.int64)
        current = np.asarray(current_rows, dtype=np.int64)
        safe_values = np.clip(values, 1e-12, None)
        changes = np.log(safe_values[current]) - np.log(safe_values[previous])
        return changes, current

    def _compute_model_stock_metric_arrays(self, model_name: str) -> dict[str, np.ndarray]:
        result = self.prediction_results[model_name]
        pred_raw = result["y_pred"][..., self.close_channel_idx].numpy()
        true_raw, last_raw, persistence_raw = self._reference_arrays()

        pred_log_change = np.log(np.clip(pred_raw, 1e-12, None)) - np.log(
            np.clip(last_raw[:, None, :], 1e-12, None)
        )
        true_log_change = np.log(np.clip(true_raw, 1e-12, None)) - np.log(
            np.clip(last_raw[:, None, :], 1e-12, None)
        )
        persistence_log_change = np.zeros_like(true_log_change)

        clg_absolute_error = np.abs(pred_log_change - true_log_change)
        persistence_clg_absolute_error = np.abs(
            persistence_log_change - true_log_change
        )
        raw_absolute_error = np.abs(pred_raw - true_raw)
        persistence_raw_absolute_error = np.abs(persistence_raw - true_raw)

        mase_scale = compute_mase_scale(
            train_split=self.train_split,
            channels=["close"],
        ).detach().cpu().numpy()[:, 0]
        safe_mase_scale = np.where(mase_scale > 1e-12, mase_scale, np.nan)

        ties = np.isclose(
            raw_absolute_error,
            persistence_raw_absolute_error,
            rtol=1e-6,
            atol=1e-8,
        )
        wins = (raw_absolute_error < persistence_raw_absolute_error) & ~ties

        pred_series_return, later_rows = self._forecast_series_returns(pred_raw)
        true_series_return, true_later_rows = self._forecast_series_returns(true_raw)
        if not np.array_equal(later_rows, true_later_rows):
            raise AssertionError("Predicted and realised series-return rows differ.")

        realised_abs_mean = np.mean(np.abs(true_log_change), axis=0)
        movement_ratio = np.divide(
            np.mean(np.abs(pred_log_change), axis=0),
            realised_abs_mean,
            out=np.full_like(realised_abs_mean, np.nan, dtype=np.float64),
            where=realised_abs_mean > 1e-12,
        )

        result_arrays = {
            "cumulative_log_change_mae": clg_absolute_error.mean(axis=0),
            "cumulative_log_change_median_absolute_error": np.median(
                clg_absolute_error, axis=0
            ),
            "cumulative_log_change_p95_absolute_error": np.quantile(
                clg_absolute_error, 0.95, axis=0
            ),
            "mase": np.mean(raw_absolute_error / safe_mase_scale[None, None, :], axis=0),
            "relative_mae_vs_persistence": np.divide(
                raw_absolute_error.mean(axis=0),
                persistence_raw_absolute_error.mean(axis=0),
                out=np.full_like(
                    raw_absolute_error.mean(axis=0), np.nan, dtype=np.float64
                ),
                where=persistence_raw_absolute_error.mean(axis=0) > 1e-12,
            ),
            "mae_difference_vs_persistence": (
                clg_absolute_error.mean(axis=0)
                - persistence_clg_absolute_error.mean(axis=0)
            ),
            "persistence_win_rate": (wins.astype(np.float64) + 0.5 * ties).mean(axis=0),
            "cumulative_log_change_directional_accuracy": (
                np.sign(pred_log_change) == np.sign(true_log_change)
            ).mean(axis=0),
            "cumulative_log_change_pearson_correlation": _assetwise_pearson(
                pred_log_change, true_log_change
            ),
            "raw_price_temporal_pearson_correlation": _assetwise_pearson(
                pred_raw, true_raw
            ),
            "forecast_series_log_return_temporal_pearson_correlation": (
                _assetwise_pearson(pred_series_return, true_series_return)
                if pred_series_return.shape[0] > 0
                else np.full(
                    (len(self.horizons), len(self.assets)),
                    np.nan,
                    dtype=np.float64,
                )
            ),
            "cumulative_log_change_movement_magnitude_ratio": movement_ratio,
            "cumulative_log_change_temporal_absolute_correlation": _assetwise_pearson(
                np.abs(pred_log_change), np.abs(true_log_change)
            ),
        }
        return result_arrays

    def per_stock_metrics(
        self,
        *,
        model_names: str | Sequence[str] | None = None,
        metric_names: str | Sequence[str] | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return tidy stock-level metrics for selected models and metrics."""

        selected_models = _normalise_model_names(
            model_names, available=self.model_names
        )
        if metric_names is None:
            selected_metrics = list(STOCK_METRIC_SPECS)
        elif isinstance(metric_names, str):
            selected_metrics = [metric_names]
        else:
            selected_metrics = [str(value) for value in metric_names]
        unknown_metrics = [
            name for name in selected_metrics if name not in STOCK_METRIC_SPECS
        ]
        if unknown_metrics:
            raise ValueError(
                f"Metrics {unknown_metrics} are not defined at stock level. "
                "IC and Rank IC are cross-sectional and intentionally excluded."
            )

        if self._per_stock_metric_cache is None or refresh:
            characteristics = self.stock_characteristics
            rows: list[dict[str, Any]] = []
            for model_name in self.model_names:
                arrays = self._compute_model_stock_metric_arrays(model_name)
                for metric_name, values in arrays.items():
                    for horizon_idx, horizon in enumerate(self.horizons):
                        for asset_idx, ticker in enumerate(self.assets):
                            rows.append(
                                {
                                    "model": model_name,
                                    "model_display_name": _format_model_name(model_name),
                                    "metric": metric_name,
                                    "metric_display_name": STOCK_METRIC_SPECS[
                                        metric_name
                                    ].display_name,
                                    "horizon": int(horizon),
                                    "ticker": ticker,
                                    "value": float(values[horizon_idx, asset_idx]),
                                }
                            )
            cache = pd.DataFrame(rows).merge(
                characteristics,
                on="ticker",
                how="left",
                validate="many_to_one",
            )
            self._per_stock_metric_cache = cache

        mask = self._per_stock_metric_cache["model"].isin(selected_models) & (
            self._per_stock_metric_cache["metric"].isin(selected_metrics)
        )
        return self._per_stock_metric_cache.loc[mask].copy()

    def _subset_prediction_result(
        self,
        *,
        model_name: str,
        horizon: int,
        row_mask: np.ndarray,
    ) -> TensorDict:
        result = self.prediction_results[model_name]
        horizon_idx = self.horizons.index(int(horizon))
        rows = torch.as_tensor(np.flatnonzero(row_mask), dtype=torch.long)
        if rows.numel() == 0:
            raise ValueError("Cannot build an evaluator from an empty subset.")
        subset = {
            "y_pred": result["y_pred"].index_select(0, rows)[:, horizon_idx : horizon_idx + 1],
            "y_true": result["y_true"].index_select(0, rows)[:, horizon_idx : horizon_idx + 1],
            "last_context_target": result["last_context_target"].index_select(0, rows),
            "channels": list(result["channels"]),
            "horizons": [int(horizon)],
            "asset_cols": list(result["asset_cols"]),
            "sample_idx": result["sample_idx"].index_select(0, rows),
            "origin_idx": result["origin_idx"].index_select(0, rows),
            "target_indices": result["target_indices"].index_select(0, rows)[
                :, horizon_idx : horizon_idx + 1
            ],
            "output_space": "raw",
        }
        return subset

    def _evaluate_subset_metric(
        self,
        *,
        model_name: str,
        metric_name: str,
        horizon: int,
        row_mask: np.ndarray,
    ) -> float:
        if metric_name == "mae_difference_vs_persistence":
            model_mae = self._evaluate_subset_metric(
                model_name=model_name,
                metric_name="cumulative_log_change_mae",
                horizon=horizon,
                row_mask=row_mask,
            )
            persistence_mae = self._evaluate_subset_metric(
                model_name=self.reference_model,
                metric_name="cumulative_log_change_mae",
                horizon=horizon,
                row_mask=row_mask,
            )
            if not (np.isfinite(model_mae) and np.isfinite(persistence_mae)):
                return float("nan")
            return float(model_mae - persistence_mae)

        subset = self._subset_prediction_result(
            model_name=model_name,
            horizon=horizon,
            row_mask=row_mask,
        )
        evaluator = ForecastEvaluator(
            prediction_result=subset,
            train_split=self.train_split,
        )
        if metric_name not in evaluator.available_metrics:
            return float("nan")
        try:
            value = evaluator.evaluate(
                metrics=metric_name,
                reduce_dims=(0, 2),
                bootstrap=False,
            )[metric_name]
        except ValueError as exc:
            # Some conditional subsets are structurally too small for a
            # correlation.  For example, the H=30 late-session bucket has
            # one target per day, so no within-session series-return pair
            # exists.  Such combinations are undefined rather than errors.
            insufficient_data_messages = (
                "at least two forecast windows",
                "No within-session forecast pairs",
                "No forecast pairs were exactly one expected origin stride apart",
            )
            if any(message in str(exc) for message in insufficient_data_messages):
                return float("nan")
            raise
        return float(value[0, self.close_channel_idx].item())

    def time_of_day_metrics(
        self,
        *,
        model_names: str | Sequence[str] | None,
        metric_name: str,
        horizons: int | Sequence[int] | None = None,
        morning_cutoff: str = "12:00",
        late_session_cutoff: str = "15:30",
    ) -> pd.DataFrame:
        """Recompute a metric in target-time buckets for each model/horizon."""

        selected_models = _normalise_model_names(
            model_names, available=self.model_names
        )
        selected_horizons = _normalise_horizons(
            horizons, available=self.horizons
        )
        available_metrics = set(self.available_group_metrics())
        if metric_name not in available_metrics:
            raise ValueError(
                f"Unknown group metric {metric_name!r}. Available metrics: "
                f"{sorted(available_metrics)}."
            )

        rows: list[dict[str, Any]] = []
        for horizon in selected_horizons:
            horizon_idx = self.horizons.index(horizon)
            timestamps = pd.DatetimeIndex(self._target_timestamp_matrix[:, horizon_idx])
            bucket_labels = _time_bucket_labels(
                timestamps,
                morning_cutoff=morning_cutoff,
                late_session_cutoff=late_session_cutoff,
            )
            for bucket in _TIME_BUCKET_ORDER:
                row_mask = bucket_labels == bucket
                num_windows = int(row_mask.sum())
                num_sessions = int(
                    np.unique(self.sample_idx.numpy()[row_mask]).size
                )
                for model_name in selected_models:
                    if num_windows == 0:
                        value = float("nan")
                    else:
                        value = self._evaluate_subset_metric(
                            model_name=model_name,
                            metric_name=metric_name,
                            horizon=horizon,
                            row_mask=row_mask,
                        )
                    rows.append(
                        {
                            "model": model_name,
                            "model_display_name": _format_model_name(model_name),
                            "metric": metric_name,
                            "horizon": horizon,
                            "time_bucket": bucket,
                            "value": value,
                            "num_windows": num_windows,
                            "num_sessions": num_sessions,
                            "morning_cutoff": morning_cutoff,
                            "late_session_cutoff": late_session_cutoff,
                            "first_target_time": (
                                timestamps[row_mask].min().strftime("%H:%M")
                                if num_windows
                                else None
                            ),
                            "last_target_time": (
                                timestamps[row_mask].max().strftime("%H:%M")
                                if num_windows
                                else None
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def daily_model_metrics(
        self,
        *,
        model_names: str | Sequence[str] | None,
        metric_name: str,
        horizon: int,
    ) -> pd.DataFrame:
        selected_models = _normalise_model_names(
            model_names, available=self.model_names
        )
        selected_horizon = _normalise_horizons(
            horizon, available=self.horizons
        )[0]
        if metric_name not in set(self.available_group_metrics()):
            raise ValueError(f"Unknown metric: {metric_name!r}.")

        session_dates = self.session_dates
        rows = []
        for sample in sorted(np.unique(self.sample_idx.numpy()).tolist()):
            row_mask = self.sample_idx.numpy() == sample
            session_date = pd.Timestamp(
                self.test_split["samples"][int(sample)][2]
            ).normalize()
            daily_market_rv = float(
                self.daily_stock_realised_volatility.loc[session_date].median()
            )
            for model_name in selected_models:
                value = self._evaluate_subset_metric(
                    model_name=model_name,
                    metric_name=metric_name,
                    horizon=selected_horizon,
                    row_mask=row_mask,
                )
                rows.append(
                    {
                        "session_date": session_date,
                        "sample_idx": int(sample),
                        "model": model_name,
                        "model_display_name": _format_model_name(model_name),
                        "metric": metric_name,
                        "horizon": selected_horizon,
                        "value": value,
                        "daily_market_realised_volatility": daily_market_rv,
                        "num_windows": int(row_mask.sum()),
                    }
                )
        return pd.DataFrame(rows).sort_values(["session_date", "model"])

    def daily_metric_values(
        self,
        *,
        model_name: str,
        metric_name: str,
        horizon: int,
    ) -> pd.DataFrame:
        """Return one daily metric series for a model and horizon.

        Most metrics are evaluated directly for ``model_name`` using the
        standard :class:`ForecastEvaluator` registry.  The one custom metric,
        ``mae_difference_vs_persistence``, is already persistence-relative by
        definition and is calculated as daily cumulative-log-change MAE for
        the selected model minus daily cumulative-log-change MAE for the
        analysis reference model (normally ``Persistence``).

        No generic model-minus-benchmark transformation is applied.  In
        particular, correlation metrics are returned as the selected model's
        own daily correlations.
        """

        selected_model = _normalise_model_names(
            model_name, available=self.model_names
        )[0]
        selected_horizon = _normalise_horizons(
            horizon, available=self.horizons
        )[0]

        if metric_name == "mae_difference_vs_persistence":
            difference = self.daily_error_differences(
                model_name=selected_model,
                benchmark_name=self.reference_model,
                horizon=selected_horizon,
                metric_name="cumulative_log_change_mae",
            )
            table = difference[
                [
                    "session_date",
                    "horizon",
                    "model",
                    "daily_market_realised_volatility",
                    "num_windows",
                    "difference",
                ]
            ].rename(columns={"difference": "value"})
            table["metric"] = metric_name
            table["metric_display_name"] = STOCK_METRIC_SPECS[
                metric_name
            ].display_name
            table["reference_model"] = self.reference_model
            table["reference_value"] = 0.0
            table["is_persistence_relative"] = True
            return table.sort_values("session_date").reset_index(drop=True)

        if metric_name not in set(self.available_group_metrics()):
            available = sorted(
                set(self.available_group_metrics())
                | {"mae_difference_vs_persistence"}
            )
            raise ValueError(
                f"Unknown daily metric {metric_name!r}. Available metrics: "
                f"{available}."
            )

        table = self.daily_model_metrics(
            model_names=selected_model,
            metric_name=metric_name,
            horizon=selected_horizon,
        ).copy()
        if metric_name in DEFAULT_METRIC_DISPLAY_NAMES:
            metric_display_name = DEFAULT_METRIC_DISPLAY_NAMES[metric_name]
        elif metric_name in STOCK_METRIC_SPECS:
            metric_display_name = STOCK_METRIC_SPECS[metric_name].display_name
        else:
            metric_display_name = metric_name.replace("_", " ")
        table["metric_display_name"] = metric_display_name
        reference_value = (
            STOCK_METRIC_SPECS[metric_name].reference_value
            if metric_name in STOCK_METRIC_SPECS
            else None
        )
        table["reference_model"] = (
            self.reference_model
            if metric_name
            in {"relative_mae_vs_persistence", "persistence_win_rate"}
            else None
        )
        table["reference_value"] = reference_value
        table["is_persistence_relative"] = metric_name in {
            "relative_mae_vs_persistence",
            "persistence_win_rate",
        }
        return table.sort_values("session_date").reset_index(drop=True)

    def daily_error_differences(
        self,
        *,
        model_name: str,
        benchmark_name: str,
        horizon: int,
        metric_name: str = "cumulative_log_change_mae",
    ) -> pd.DataFrame:
        daily = self.daily_model_metrics(
            model_names=[model_name, benchmark_name],
            metric_name=metric_name,
            horizon=horizon,
        )
        pivot = daily.pivot(
            index=[
                "session_date",
                "daily_market_realised_volatility",
                "num_windows",
            ],
            columns="model",
            values="value",
        ).reset_index()
        pivot.columns.name = None
        pivot["model"] = model_name
        pivot["benchmark"] = benchmark_name
        pivot["metric"] = metric_name
        pivot["horizon"] = int(horizon)
        pivot["model_value"] = pivot[model_name]
        pivot["benchmark_value"] = pivot[benchmark_name]
        pivot["difference"] = pivot["model_value"] - pivot["benchmark_value"]
        pivot["cumulative_difference"] = pivot["difference"].cumsum()
        return pivot[
            [
                "session_date",
                "horizon",
                "metric",
                "model",
                "benchmark",
                "model_value",
                "benchmark_value",
                "difference",
                "cumulative_difference",
                "daily_market_realised_volatility",
                "num_windows",
            ]
        ]

    def export_tables(
        self,
        output_dir: str | Path,
        *,
        include_per_stock_metrics: bool = True,
    ) -> dict[str, Path]:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        tables = {
            "alignment_manifest": self.alignment_manifest(),
            "stock_characteristics": self.stock_characteristics,
            "adf_diagnostics": self.adf_diagnostics,
            "daily_stock_realised_volatility": (
                self.daily_stock_realised_volatility.reset_index().melt(
                    id_vars="session_date",
                    var_name="ticker",
                    value_name="realised_volatility",
                )
            ),
            "split_daily_market_volatility": self.split_daily_market_volatility,
        }
        if include_per_stock_metrics:
            tables["per_stock_metrics"] = self.per_stock_metrics()

        for name, table in tables.items():
            path = output / f"{name}.csv"
            table.to_csv(path, index=False)
            paths[name] = path

        manifest_path = output / "analysis_manifest.json"
        manifest = {
            "models": list(self.model_names),
            "reference_model": self.reference_model,
            "horizons": list(self.horizons),
            "assets": list(self.assets),
            "channels": list(self.channels),
            "windows": int(self.sample_idx.numel()),
            "test_sessions": int(len(self.test_split["samples"])),
            "stock_metric_names": list(STOCK_METRIC_SPECS),
            "group_metric_names": list(self.available_group_metrics()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        paths["analysis_manifest"] = manifest_path
        return paths


# ---------------------------------------------------------------------------
# Plotting and table helpers
# ---------------------------------------------------------------------------


def plot_split_volatility_distribution(
    analysis: FinancialResultAnalysis,
    *,
    figsize: tuple[float, float] = (9.0, 5.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compare daily cross-asset median realised volatility by split."""

    table = analysis.split_daily_market_volatility.copy()
    fig, ax = plt.subplots(figsize=figsize)
    split_order = ["Train", "Validation", "Test"]
    values = [
        table.loc[table["split"] == split_name, "market_realised_volatility"].to_numpy()
        for split_name in split_order
    ]
    ax.boxplot(values, labels=split_order, showfliers=True)
    ax.set_ylabel("Daily market realised volatility")
    ax.set_title("Daily cross-asset median realised volatility by split")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return table, fig


def plot_adf_pvalue_distribution(
    analysis: FinancialResultAnalysis,
    *,
    significance_level: float = 0.05,
    bins: int = 15,
    figsize: tuple[float, float] = (9.0, 5.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot the distribution of stock-level ADF p-values."""

    table = analysis.adf_diagnostics.copy()
    values = table["adf_pvalue"].dropna().to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)
    ax.axvline(
        significance_level,
        linestyle="--",
        linewidth=1.5,
        label=f"{significance_level:.0%} threshold",
    )
    ax.set_xlabel("ADF p-value on daily log closing prices")
    ax.set_ylabel("Number of stocks")
    ax.set_title("ADF unit-root diagnostic across test-set stocks")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return table, fig


def plot_adf_pvalues_by_stock(
    analysis: FinancialResultAnalysis,
    *,
    significance_level: float = 0.05,
    order_by: OrderMode = "volatility",
    num_volatility_buckets: int = 5,
    figsize: tuple[float, float] = (18.0, 5.5),
    show_ticker_labels: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot one ADF p-value bar per stock with logical stock ordering.

    Parameters
    ----------
    significance_level:
        Horizontal rejection threshold. A p-value below this line rejects the
        unit-root null at the selected level.
    order_by:
        ``"volatility"`` orders stocks from highest to lowest test-period
        median realised volatility. ``"sector"`` groups sectors alphabetically
        and orders stocks from highest to lowest volatility inside each sector.
    num_volatility_buckets:
        Number of equal-frequency volatility buckets. This controls the group
        separators for volatility ordering and the bucket metadata returned for
        either ordering mode.
    """

    if not 0.0 < float(significance_level) < 1.0:
        raise ValueError("significance_level must lie strictly between 0 and 1.")

    ordered, boundaries = _ordered_stock_rows(
        analysis,
        order_by=order_by,
        num_volatility_buckets=num_volatility_buckets,
    )
    # ``stock_characteristics`` already contains the validated ADF fields.
    table = ordered.copy()
    table["display_order"] = np.arange(len(table), dtype=np.int64)

    x_positions = table["display_order"].to_numpy(dtype=np.int64)
    values = table["adf_pvalue"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_positions, values)
    ax.axhline(
        float(significance_level),
        linestyle="--",
        linewidth=1.4,
        label=f"{significance_level:.0%} rejection threshold",
    )
    _decorate_stock_axis_groups(ax, boundaries)

    ax.set_xlim(-0.75, len(table) - 0.25)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("ADF p-value")
    ax.set_xlabel("Stock")
    if show_ticker_labels:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(table["ticker"], rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    ax.set_title(
        "ADF p-values on test-period daily log closing prices\n"
        f"Stocks ordered by {order_by}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return table, fig


def plot_metric_vs_adf(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    metric_name: str,
    horizon: int,
    significance_level: float = 0.05,
    annotate_extremes: int = 5,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Relate a stock-level performance metric to its ADF p-value."""

    metric = analysis.per_stock_metrics(
        model_names=model_name,
        metric_names=metric_name,
    )
    selected = metric.loc[metric["horizon"] == int(horizon)].copy()
    if selected.empty:
        raise ValueError("No matching stock metrics were found.")
    valid = selected["adf_pvalue"].notna() & selected["value"].notna()
    selected = selected.loc[valid].copy()

    pearson = _pearson_1d(
        selected["adf_pvalue"].to_numpy(),
        selected["value"].to_numpy(),
    )
    spearman = stats.spearmanr(
        selected["adf_pvalue"], selected["value"], nan_policy="omit"
    ).statistic

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(selected["adf_pvalue"], selected["value"], alpha=0.8)
    ax.axvline(significance_level, linestyle="--", linewidth=1.2)

    if len(selected) >= 2 and selected["adf_pvalue"].nunique() > 1:
        slope, intercept = np.polyfit(
            selected["adf_pvalue"].to_numpy(), selected["value"].to_numpy(), 1
        )
        x_grid = np.linspace(
            selected["adf_pvalue"].min(), selected["adf_pvalue"].max(), 100
        )
        ax.plot(x_grid, intercept + slope * x_grid, linewidth=1.4)

    if annotate_extremes > 0:
        extreme = pd.concat(
            [
                selected.nsmallest(annotate_extremes, "value"),
                selected.nlargest(annotate_extremes, "value"),
            ]
        ).drop_duplicates("ticker")
        for row in extreme.itertuples():
            ax.annotate(
                row.ticker,
                (row.adf_pvalue, row.value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    spec = STOCK_METRIC_SPECS[metric_name]
    ax.set_xlabel("ADF p-value on daily log closing prices")
    ax.set_ylabel(spec.display_name)
    ax.set_title(
        f"{_format_model_name(model_name)}: {spec.display_name} vs ADF p-value "
        f"at {horizon} minutes\nPearson={pearson:.3f}, Spearman={spearman:.3f}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    selected["pearson_across_stocks"] = pearson
    selected["spearman_across_stocks"] = spearman
    return selected, fig


def plot_persistence_headroom(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    metric_name: str,
    horizon: int,
    annotate_extremes: int = 5,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compare stock performance with persistence's realised-movement error."""

    model_metric = analysis.per_stock_metrics(
        model_names=model_name,
        metric_names=metric_name,
    )
    persistence_metric = analysis.per_stock_metrics(
        model_names=analysis.reference_model,
        metric_names="cumulative_log_change_mae",
    )
    model_metric = model_metric.loc[model_metric["horizon"] == int(horizon)].copy()
    persistence_metric = persistence_metric.loc[
        persistence_metric["horizon"] == int(horizon),
        ["ticker", "value"],
    ].rename(columns={"value": "persistence_clg_mae"})
    selected = model_metric.merge(
        persistence_metric,
        on="ticker",
        how="left",
        validate="one_to_one",
    )

    pearson = _pearson_1d(
        selected["persistence_clg_mae"].to_numpy(),
        selected["value"].to_numpy(),
    )
    spearman = stats.spearmanr(
        selected["persistence_clg_mae"], selected["value"], nan_policy="omit"
    ).statistic

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(selected["persistence_clg_mae"], selected["value"], alpha=0.8)
    if len(selected) >= 2 and selected["persistence_clg_mae"].nunique() > 1:
        slope, intercept = np.polyfit(
            selected["persistence_clg_mae"].to_numpy(),
            selected["value"].to_numpy(),
            1,
        )
        x_grid = np.linspace(
            selected["persistence_clg_mae"].min(),
            selected["persistence_clg_mae"].max(),
            100,
        )
        ax.plot(x_grid, intercept + slope * x_grid, linewidth=1.4)

    if annotate_extremes > 0:
        extreme = pd.concat(
            [
                selected.nsmallest(annotate_extremes, "value"),
                selected.nlargest(annotate_extremes, "value"),
            ]
        ).drop_duplicates("ticker")
        for row in extreme.itertuples():
            ax.annotate(
                row.ticker,
                (row.persistence_clg_mae, row.value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    spec = STOCK_METRIC_SPECS[metric_name]
    ax.set_xlabel("Persistence cumulative-log-change MAE (movement/headroom)")
    ax.set_ylabel(spec.display_name)
    ax.set_title(
        f"{_format_model_name(model_name)}: performance vs persistence headroom "
        f"at {horizon} minutes\nPearson={pearson:.3f}, Spearman={spearman:.3f}"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    selected["pearson_across_stocks"] = pearson
    selected["spearman_across_stocks"] = spearman
    return selected, fig


def _ordered_stock_rows(
    analysis: FinancialResultAnalysis,
    *,
    order_by: OrderMode,
    num_volatility_buckets: int = 5,
) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    characteristics = _with_volatility_buckets(
        analysis.stock_characteristics,
        num_buckets=num_volatility_buckets,
    )
    characteristics["original_position"] = np.arange(len(characteristics))
    boundaries: list[tuple[str, int, int]] = []

    if order_by == "volatility":
        ordered = characteristics.sort_values(
            ["test_median_realised_volatility", "ticker"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)
        for bucket in range(num_volatility_buckets, 0, -1):
            positions = np.flatnonzero(
                ordered["volatility_bucket"].to_numpy(dtype=np.int64) == bucket
            )
            if len(positions):
                boundaries.append(
                    (
                        _volatility_bucket_label(
                            bucket,
                            num_buckets=num_volatility_buckets,
                        ),
                        int(positions.min()),
                        int(positions.max()),
                    )
                )
        return ordered, boundaries

    if order_by != "sector":
        raise ValueError("order_by must be 'volatility' or 'sector'.")

    ordered = characteristics.sort_values(
        ["sector", "test_median_realised_volatility", "ticker"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    for sector, group in ordered.groupby("sector", sort=False):
        positions = group.index.to_numpy()
        boundaries.append((str(sector), int(positions.min()), int(positions.max())))
    return ordered, boundaries


def _decorate_stock_axis_groups(
    ax: plt.Axes,
    boundaries: Sequence[tuple[str, int, int]],
) -> None:
    """Draw group separators and labels above a stock-indexed x-axis."""

    for label, start, end in boundaries:
        if start > 0:
            ax.axvline(start - 0.5, linewidth=0.8, color="black", alpha=0.55)
        midpoint = (start + end) / 2.0
        ax.text(
            midpoint,
            1.01,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )


def _draw_group_mean_segments(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    boundaries: Sequence[tuple[str, int, int]],
    color: str = "red",
    linewidth: float = 1.6,
    legend_label: str = "Group mean",
) -> dict[str, float]:
    """Draw one horizontal mean segment across each visible stock group.

    The segment spans only the bars belonging to its volatility bucket or
    sector.  Non-finite stock metrics are excluded from that group's mean.
    The returned mapping makes the plotted values available for audit and
    testing.
    """

    metric_values = np.asarray(values, dtype=np.float64)
    group_means: dict[str, float] = {}
    legend_added = False

    for label, start, end in boundaries:
        group_values = metric_values[int(start) : int(end) + 1]
        finite_values = group_values[np.isfinite(group_values)]
        if finite_values.size == 0:
            continue

        group_mean = float(np.mean(finite_values))
        line, = ax.plot(
            [float(start) - 0.4, float(end) + 0.4],
            [group_mean, group_mean],
            color=color,
            linestyle="--",
            linewidth=float(linewidth),
            zorder=4,
            label=str(legend_label) if not legend_added else "_nolegend_",
        )
        line.set_gid("group_mean")
        group_means[str(label)] = group_mean
        legend_added = True

    return group_means


def _adaptive_metric_axis_limits(
    values: np.ndarray,
    *,
    spec: StockMetricSpec,
    padding_fraction: float = 0.08,
    minimum_relative_span: float = 0.04,
    reference_distance_multiplier: float = 2.0,
) -> tuple[float, float, bool]:
    """Return readable y-axis limits for a stock-level metric bar chart.

    The calculation is based on the finite values for one horizon.  A nearby
    benchmark/reference value is included because it is central to interpreting
    metrics such as relative MAE and persistence win rate.  A distant reference
    (for example zero when price correlations are all close to one) is not
    allowed to flatten the visible differences between stocks.

    Returns
    -------
    lower, upper, reference_visible
        Axis limits and whether the metric's reference value lies inside the
        resulting zoomed range.
    """

    if not 0.0 <= float(padding_fraction) < 1.0:
        raise ValueError("padding_fraction must lie in [0, 1).")
    if float(minimum_relative_span) <= 0.0:
        raise ValueError("minimum_relative_span must be positive.")
    if float(reference_distance_multiplier) < 0.0:
        raise ValueError("reference_distance_multiplier must be non-negative.")

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot determine y-axis limits without finite values.")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    scale_candidates = [abs(data_min), abs(data_max), np.finfo(np.float64).tiny]
    if spec.reference_value is not None and np.isfinite(spec.reference_value):
        scale_candidates.append(abs(float(spec.reference_value)))
    scale = max(scale_candidates)

    minimum_span = max(
        scale * float(minimum_relative_span),
        np.finfo(np.float64).eps * 128.0,
    )
    data_span = max(data_max - data_min, minimum_span)

    lower_core = data_min
    upper_core = data_max
    if data_max - data_min < minimum_span:
        midpoint = 0.5 * (data_min + data_max)
        lower_core = midpoint - 0.5 * minimum_span
        upper_core = midpoint + 0.5 * minimum_span

    reference = spec.reference_value
    if reference is not None and np.isfinite(reference):
        reference = float(reference)
        if reference < data_min:
            distance = data_min - reference
        elif reference > data_max:
            distance = reference - data_max
        else:
            distance = 0.0

        if distance <= float(reference_distance_multiplier) * data_span:
            lower_core = min(lower_core, reference)
            upper_core = max(upper_core, reference)

    core_span = max(upper_core - lower_core, minimum_span)
    padding = max(core_span * float(padding_fraction), minimum_span * 0.05)
    lower = lower_core - padding
    upper = upper_core + padding

    if spec.lower_bound is not None:
        lower = max(lower, float(spec.lower_bound))
    if spec.upper_bound is not None:
        upper = min(upper, float(spec.upper_bound))

    if not upper > lower:
        midpoint = 0.5 * (data_min + data_max)
        half_span = max(0.5 * minimum_span, np.finfo(np.float64).eps * 128.0)
        lower = midpoint - half_span
        upper = midpoint + half_span
        if spec.lower_bound is not None:
            lower = max(lower, float(spec.lower_bound))
        if spec.upper_bound is not None:
            upper = min(upper, float(spec.upper_bound))

    reference_visible = bool(
        reference is not None
        and np.isfinite(reference)
        and lower <= float(reference) <= upper
    )
    return float(lower), float(upper), reference_visible


def _zero_based_metric_axis_limits(
    values: np.ndarray,
    *,
    spec: StockMetricSpec,
    padding_fraction: float = 0.08,
) -> tuple[float, float, bool]:
    """Return a conventional zero-based axis for comparison/reproducibility."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot determine y-axis limits without finite values.")

    lower_core = min(0.0, float(np.min(finite)))
    upper_core = max(0.0, float(np.max(finite)))
    if spec.reference_value is not None and np.isfinite(spec.reference_value):
        lower_core = min(lower_core, float(spec.reference_value))
        upper_core = max(upper_core, float(spec.reference_value))

    span = max(
        upper_core - lower_core,
        max(abs(lower_core), abs(upper_core), 1.0) * 0.04,
    )
    padding = span * float(padding_fraction)
    lower = lower_core - padding if lower_core < 0.0 else 0.0
    upper = upper_core + padding

    if spec.lower_bound is not None:
        lower = max(lower, float(spec.lower_bound))
    if spec.upper_bound is not None:
        upper = min(upper, float(spec.upper_bound))

    reference_visible = bool(
        spec.reference_value is not None
        and lower <= float(spec.reference_value) <= upper
    )
    return float(lower), float(upper), reference_visible


def plot_stock_horizon_heatmap(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    metric_name: str,
    order_by: OrderMode = "volatility",
    num_volatility_buckets: int = 5,
    figsize: tuple[float, float] = (11.0, 20.0),
    cmap: str | None = None,
    show_ticker_labels: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot one stock-level metric over all stocks and horizons."""

    metrics = analysis.per_stock_metrics(
        model_names=model_name,
        metric_names=metric_name,
    )
    ordered, boundaries = _ordered_stock_rows(
        analysis,
        order_by=order_by,
        num_volatility_buckets=num_volatility_buckets,
    )
    ticker_order = ordered["ticker"].tolist()
    matrix = (
        metrics.pivot(index="ticker", columns="horizon", values="value")
        .reindex(index=ticker_order, columns=list(analysis.horizons))
    )
    spec = STOCK_METRIC_SPECS[metric_name]

    values = matrix.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The requested heatmap contains no finite values.")

    if cmap is None:
        cmap = "RdBu_r" if spec.reference_value is not None else "viridis"

    norm = None
    if (
        spec.reference_value is not None
        and float(np.nanmin(values)) < spec.reference_value < float(np.nanmax(values))
    ):
        norm = TwoSlopeNorm(
            vmin=float(np.nanmin(values)),
            vcenter=float(spec.reference_value),
            vmax=float(np.nanmax(values)),
        )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(analysis.horizons)))
    ax.set_xticklabels([str(value) for value in analysis.horizons])
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Stock")
    if show_ticker_labels:
        ax.set_yticks(np.arange(len(ticker_order)))
        ax.set_yticklabels(ticker_order, fontsize=7)
    else:
        ax.set_yticks([])

    group_midpoints: list[float] = []
    group_labels: list[str] = []
    for label, start, end in boundaries:
        if start > 0:
            ax.axhline(start - 0.5, linewidth=0.8, color="black", alpha=0.55)
        group_midpoints.append((start + end) / 2.0)
        group_labels.append(label)

    group_axis = ax.twinx()
    group_axis.set_ylim(ax.get_ylim())
    group_axis.set_yticks(group_midpoints)
    group_axis.set_yticklabels(group_labels, fontsize=7)
    group_axis.tick_params(axis="y", length=0, pad=4)
    for spine in group_axis.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=[ax, group_axis], pad=0.08)
    colorbar.set_label(spec.display_name)
    ax.set_title(
        f"{_format_model_name(model_name)} — {spec.display_name}\n"
        f"Stocks ordered by {order_by}"
    )

    tidy = metrics.merge(
        ordered[["ticker"]].assign(display_order=np.arange(len(ordered))),
        on="ticker",
        how="left",
        validate="many_to_one",
    ).sort_values(["display_order", "horizon"])
    return tidy, fig


def plot_stock_metric_by_horizon(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    metric_name: str,
    horizons: int | Sequence[int] | None = None,
    order_by: OrderMode = "volatility",
    num_volatility_buckets: int = 5,
    figsize_per_horizon: tuple[float, float] = (18.0, 5.5),
    show_ticker_labels: bool = True,
    y_axis_mode: YAxisMode = "adaptive",
    y_padding_fraction: float = 0.08,
    annotate_zoomed_axis: bool = True,
    show_group_means: bool = True,
) -> tuple[pd.DataFrame, dict[int, plt.Figure]]:
    """Plot one separate stock bar chart for every selected horizon.

    Each horizon receives an independent figure and y-axis scale. Volatility
    ordering is highest-to-lowest. Sector ordering groups sectors alphabetically
    and sorts stocks by decreasing realised volatility inside each sector.

    By default, ``y_axis_mode="adaptive"`` zooms each horizon to its finite
    stock-level value range, adds padding, respects the metric's mathematical
    bounds, and includes a nearby reference value.  This makes differences near
    one (relative MAE), 0.5 (win rate), or another local level readable.
    ``y_axis_mode="zero"`` restores a conventional zero-based bar-chart axis.

    When ``show_group_means=True``, a red dashed horizontal segment is drawn
    across every volatility bucket or sector.  Its height is the arithmetic
    mean of the finite stock-level metric values in that visible group for the
    current horizon.
    """

    selected_horizons = _normalise_horizons(
        horizons,
        available=analysis.horizons,
    )
    metrics = analysis.per_stock_metrics(
        model_names=model_name,
        metric_names=metric_name,
    )
    ordered, boundaries = _ordered_stock_rows(
        analysis,
        order_by=order_by,
        num_volatility_buckets=num_volatility_buckets,
    )
    ordered = ordered.copy()
    ordered["display_group"] = ""
    for label, start, end in boundaries:
        ordered.loc[int(start) : int(end), "display_group"] = str(label)

    ticker_order = ordered["ticker"].tolist()
    ordering_columns = [
        "ticker",
        "sector",
        "test_median_realised_volatility",
        "volatility_bucket",
        "volatility_bucket_label",
        "num_volatility_buckets",
        "display_group",
    ]
    tidy = metrics.loc[metrics["horizon"].isin(selected_horizons)].drop(
        columns=[
            column
            for column in ordering_columns[1:]
            if column in metrics.columns
        ],
        errors="ignore",
    ).merge(
        ordered[ordering_columns].assign(
            display_order=np.arange(len(ordered), dtype=np.int64)
        ),
        on="ticker",
        how="left",
        validate="many_to_one",
    ).sort_values(["horizon", "display_order"])
    tidy["display_group_mean"] = tidy.groupby(
        ["horizon", "display_group"],
        sort=False,
    )["value"].transform("mean")

    spec = STOCK_METRIC_SPECS[metric_name]
    if y_axis_mode not in {"adaptive", "zero"}:
        raise ValueError("y_axis_mode must be 'adaptive' or 'zero'.")
    if not 0.0 <= float(y_padding_fraction) < 1.0:
        raise ValueError("y_padding_fraction must lie in [0, 1).")

    figures: dict[int, plt.Figure] = {}
    for horizon in selected_horizons:
        horizon_table = (
            tidy.loc[tidy["horizon"] == int(horizon)]
            .set_index("ticker")
            .reindex(ticker_order)
            .reset_index()
        )
        values = horizon_table["value"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).any():
            raise ValueError(
                f"No finite {metric_name} values exist at horizon {horizon}."
            )

        x_positions = np.arange(len(horizon_table), dtype=np.int64)
        fig, ax = plt.subplots(figsize=figsize_per_horizon)
        ax.bar(x_positions, values)

        if y_axis_mode == "adaptive":
            y_lower, y_upper, reference_visible = _adaptive_metric_axis_limits(
                values,
                spec=spec,
                padding_fraction=y_padding_fraction,
            )
        else:
            y_lower, y_upper, reference_visible = _zero_based_metric_axis_limits(
                values,
                spec=spec,
                padding_fraction=y_padding_fraction,
            )
        ax.set_ylim(y_lower, y_upper)

        legend_required = False
        if spec.reference_value is not None and reference_visible:
            ax.axhline(
                float(spec.reference_value),
                linestyle="--",
                linewidth=1.3,
                label=f"Reference = {spec.reference_value:g}",
            )
            legend_required = True

        if show_group_means:
            group_means = _draw_group_mean_segments(
                ax,
                values=values,
                boundaries=boundaries,
                legend_label=(
                    "Volatility-bucket mean"
                    if order_by == "volatility"
                    else "Sector mean"
                ),
            )
            legend_required = legend_required or bool(group_means)

        if legend_required:
            ax.legend()
        _decorate_stock_axis_groups(ax, boundaries)

        if (
            annotate_zoomed_axis
            and y_axis_mode == "adaptive"
            and not (y_lower <= 0.0 <= y_upper)
        ):
            ax.text(
                0.995,
                0.015,
                "Adaptive y-axis (zero not shown)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                alpha=0.65,
            )

        ax.set_xlim(-0.75, len(horizon_table) - 0.25)
        ax.set_ylabel(spec.display_name)
        ax.set_xlabel("Stock")
        if show_ticker_labels:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                horizon_table["ticker"],
                rotation=90,
                fontsize=7,
            )
        else:
            ax.set_xticks([])
        ax.set_title(
            f"{_format_model_name(model_name)} — {spec.display_name} "
            f"at {horizon} minutes\nStocks ordered by {order_by}"
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figures[int(horizon)] = fig

    return tidy, figures


def plot_stock_metric_ecdf(
    analysis: FinancialResultAnalysis,
    *,
    model_names: str | Sequence[str],
    metric_name: str,
    horizon: int,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot empirical CDFs across stocks for a selected metric/horizon."""

    selected_models = _normalise_model_names(
        model_names, available=analysis.model_names
    )
    table = analysis.per_stock_metrics(
        model_names=selected_models,
        metric_names=metric_name,
    )
    table = table.loc[table["horizon"] == int(horizon)].copy()
    spec = STOCK_METRIC_SPECS[metric_name]

    fig, ax = plt.subplots(figsize=figsize)
    for model_name in selected_models:
        values = np.sort(
            table.loc[table["model"] == model_name, "value"]
            .dropna()
            .to_numpy(dtype=np.float64)
        )
        if len(values) == 0:
            continue
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, probabilities, where="post", label=_format_model_name(model_name))

    if spec.reference_value is not None:
        ax.axvline(spec.reference_value, linestyle="--", linewidth=1.2)
    ax.set_xlabel(spec.display_name)
    ax.set_ylabel("Fraction of stocks")
    ax.set_title(f"Stock-level ECDF at {horizon} minutes")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return table, fig


def make_top_bottom_stock_table(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    metric_name: str,
    horizon: int,
    top_k: int = 10,
    num_volatility_buckets: int = 5,
) -> pd.DataFrame:
    """Return a compact best/worst stock table for one metric.

    ``volatility_bucket`` is an equal-frequency test-period grouping. Bucket 1
    contains the lowest-volatility stocks and bucket K the highest-volatility
    stocks, where K is ``num_volatility_buckets``.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    table = analysis.per_stock_metrics(
        model_names=model_name,
        metric_names=metric_name,
    )
    table = table.loc[table["horizon"] == int(horizon)].copy()
    spec = STOCK_METRIC_SPECS[metric_name]

    if spec.direction == "lower":
        table["performance_score"] = table["value"]
    elif spec.direction == "higher":
        table["performance_score"] = -table["value"]
    else:
        assert spec.reference_value is not None
        table["performance_score"] = np.abs(table["value"] - spec.reference_value)

    best = table.nsmallest(top_k, "performance_score").copy()
    worst = table.nlargest(top_k, "performance_score").copy()
    best["group"] = "Top"
    worst["group"] = "Bottom"
    combined = pd.concat([best, worst], ignore_index=True)
    combined["rank_within_group"] = combined.groupby("group").cumcount() + 1

    bucket_table = _with_volatility_buckets(
        analysis.stock_characteristics,
        num_buckets=num_volatility_buckets,
    )[[
        "ticker",
        "volatility_bucket",
        "volatility_bucket_label",
        "num_volatility_buckets",
    ]]
    combined = combined.drop(
        columns=[
            "volatility_bucket",
            "volatility_bucket_label",
            "num_volatility_buckets",
        ],
        errors="ignore",
    ).merge(
        bucket_table,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    persistence = analysis.per_stock_metrics(
        model_names=analysis.reference_model,
        metric_names="cumulative_log_change_mae",
    )
    persistence = persistence.loc[
        persistence["horizon"] == int(horizon), ["ticker", "value"]
    ].rename(columns={"value": "persistence_clg_mae"})
    combined = combined.merge(
        persistence,
        on="ticker",
        how="left",
        validate="many_to_one",
    )
    return combined[
        [
            "group",
            "rank_within_group",
            "ticker",
            "sector",
            "volatility_bucket",
            "volatility_bucket_label",
            "num_volatility_buckets",
            "test_median_realised_volatility",
            "adf_statistic",
            "adf_pvalue",
            "value",
            "persistence_clg_mae",
        ]
    ]


def plot_metric_by_stock_volatility(
    analysis: FinancialResultAnalysis,
    *,
    model_names: str | Sequence[str],
    metric_name: str,
    horizons: int | Sequence[int] | None = None,
    num_volatility_buckets: int = 5,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot mean stock-level performance over configurable volatility buckets.

    The x-axis buckets are equal-frequency groups based on each stock's median
    daily realised volatility during the test period. Bucket 1 is the lowest-
    volatility group and bucket K the highest. One line is drawn for each
    selected ``(model, horizon)`` pair; no bars, quartiles, or error bars are
    shown.

    For ``relative_mae_vs_persistence`` only, the returned table additionally
    reports ``fraction_of_stocks_beating_persistence``. This is the proportion
    of stocks in the bucket whose stock-level relative MAE is strictly below 1.
    """

    selected_models = _normalise_model_names(
        model_names,
        available=analysis.model_names,
    )
    selected_horizons = _normalise_horizons(
        horizons,
        available=analysis.horizons,
    )
    table = analysis.per_stock_metrics(
        model_names=selected_models,
        metric_names=metric_name,
    )
    table = table.loc[table["horizon"].isin(selected_horizons)].copy()

    bucket_table = _with_volatility_buckets(
        analysis.stock_characteristics,
        num_buckets=num_volatility_buckets,
    )[[
        "ticker",
        "volatility_bucket",
        "volatility_bucket_label",
        "num_volatility_buckets",
    ]]
    table = table.drop(
        columns=[
            "volatility_bucket",
            "volatility_bucket_label",
            "num_volatility_buckets",
        ],
        errors="ignore",
    ).merge(
        bucket_table,
        on="ticker",
        how="left",
        validate="many_to_one",
    )

    summary_rows: list[dict[str, Any]] = []
    for (model_name, horizon, bucket), group in table.groupby(
        ["model", "horizon", "volatility_bucket"],
        observed=True,
        sort=True,
    ):
        valid = group.loc[group["value"].notna()].copy()
        values = valid["value"].to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue
        row: dict[str, Any] = {
            "model": model_name,
            "model_display_name": _format_model_name(model_name),
            "metric": metric_name,
            "metric_display_name": STOCK_METRIC_SPECS[metric_name].display_name,
            "horizon": int(horizon),
            "volatility_bucket": int(bucket),
            "volatility_bucket_label": _volatility_bucket_label(
                int(bucket),
                num_buckets=num_volatility_buckets,
            ),
            "num_volatility_buckets": int(num_volatility_buckets),
            "num_stocks": int(len(values)),
            "minimum_realised_volatility": float(
                valid["test_median_realised_volatility"].min()
            ),
            "mean_realised_volatility": float(
                valid["test_median_realised_volatility"].mean()
            ),
            "maximum_realised_volatility": float(
                valid["test_median_realised_volatility"].max()
            ),
            "mean_metric": float(np.mean(values)),
        }
        if metric_name == "relative_mae_vs_persistence":
            row["fraction_of_stocks_beating_persistence"] = float(
                np.mean(values < 1.0)
            )
        summary_rows.append(row)

    if not summary_rows:
        raise ValueError("No finite stock-level values were available to summarise.")
    summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "horizon", "volatility_bucket"]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    x_positions = np.arange(1, num_volatility_buckets + 1, dtype=np.int64)
    for model_name in selected_models:
        for horizon in selected_horizons:
            line = (
                summary.loc[
                    (summary["model"] == model_name)
                    & (summary["horizon"] == int(horizon))
                ]
                .set_index("volatility_bucket")
                .reindex(x_positions)
            )
            label = (
                f"{_format_model_name(model_name)} — {horizon}m"
                if len(selected_models) > 1
                else f"{horizon}m"
            )
            ax.plot(
                x_positions,
                line["mean_metric"].to_numpy(dtype=np.float64),
                marker="o",
                label=label,
            )

    spec = STOCK_METRIC_SPECS[metric_name]
    if spec.reference_value is not None:
        ax.axhline(
            float(spec.reference_value),
            linestyle="--",
            linewidth=1.2,
            label=f"Reference = {spec.reference_value:g}",
        )
    tick_labels = [
        (
            "1\n(lowest)"
            if bucket == 1 and num_volatility_buckets > 1
            else f"{bucket}\n(highest)"
            if bucket == num_volatility_buckets and num_volatility_buckets > 1
            else str(bucket)
        )
        for bucket in x_positions
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(
        "Equal-frequency stock-volatility bucket "
        "(test-period median daily realised volatility)"
    )
    ax.set_ylabel(f"Mean {spec.display_name} across stocks")
    ax.set_title("Mean stock-level performance by test-period volatility")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return summary, fig


def plot_time_of_day_metric(
    analysis: FinancialResultAnalysis,
    *,
    model_names: str | Sequence[str],
    metric_name: str,
    horizons: int | Sequence[int] | None,
    morning_cutoff: str = "12:00",
    late_session_cutoff: str = "15:30",
    figsize_per_horizon: tuple[float, float] = (9.0, 4.2),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot model metrics by predicted target time, not forecast-origin time."""

    selected_horizons = _normalise_horizons(
        horizons, available=analysis.horizons
    )
    table = analysis.time_of_day_metrics(
        model_names=model_names,
        metric_name=metric_name,
        horizons=selected_horizons,
        morning_cutoff=morning_cutoff,
        late_session_cutoff=late_session_cutoff,
    )

    num_horizons = len(selected_horizons)
    fig, axes = plt.subplots(
        nrows=num_horizons,
        ncols=1,
        figsize=(figsize_per_horizon[0], figsize_per_horizon[1] * num_horizons),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes[:, 0]
    model_order = _normalise_model_names(model_names, available=analysis.model_names)
    x_positions = np.arange(len(_TIME_BUCKET_ORDER))

    for axis, horizon in zip(axes_flat, selected_horizons, strict=True):
        horizon_table = table.loc[table["horizon"] == horizon]
        for model_name in model_order:
            model_table = (
                horizon_table.loc[horizon_table["model"] == model_name]
                .set_index("time_bucket")
                .reindex(_TIME_BUCKET_ORDER)
            )
            axis.plot(
                x_positions,
                model_table["value"].to_numpy(dtype=np.float64),
                marker="o",
                label=_format_model_name(model_name),
            )
        if metric_name in DEFAULT_METRIC_DISPLAY_NAMES:
            metric_display_name = DEFAULT_METRIC_DISPLAY_NAMES[metric_name]
        elif metric_name in STOCK_METRIC_SPECS:
            metric_display_name = STOCK_METRIC_SPECS[metric_name].display_name
        else:
            metric_display_name = metric_name.replace("_", " ")
        axis.set_ylabel(metric_display_name)
        if metric_name == "mae_difference_vs_persistence":
            axis.axhline(
                0.0,
                color="red",
                linestyle="--",
                linewidth=1.2,
                label="Persistence parity = 0",
            )
        axis.set_title(f"{horizon}-minute horizon")
        axis.grid(alpha=0.25)

    axes_flat[-1].set_xticks(x_positions)
    axes_flat[-1].set_xticklabels(_TIME_BUCKET_ORDER)
    axes_flat[-1].set_xlabel(
        "Predicted target bar-close time bucket "
        f"(<{morning_cutoff}, {morning_cutoff}–{late_session_cutoff}, "
        f">={late_session_cutoff})"
    )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.suptitle("Metric by actual prediction target time", y=1.002)
    fig.tight_layout()
    return table, fig


def make_time_of_day_metric_table(
    time_of_day_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return a compact wide table from ``time_of_day_metrics`` output."""

    required = {
        "horizon",
        "model_display_name",
        "time_bucket",
        "value",
        "num_windows",
        "num_sessions",
    }
    missing = required.difference(time_of_day_results.columns)
    if missing:
        raise KeyError(
            f"time_of_day_results is missing columns: {sorted(missing)}."
        )

    ordered = time_of_day_results.copy()
    ordered["time_bucket"] = pd.Categorical(
        ordered["time_bucket"],
        categories=list(_TIME_BUCKET_ORDER),
        ordered=True,
    )
    wide = ordered.pivot_table(
        index=["horizon", "model_display_name"],
        columns="time_bucket",
        values=["value", "num_windows", "num_sessions"],
        aggfunc="first",
        observed=False,
    )
    wide = wide.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    bucket_columns = []
    for bucket in _TIME_BUCKET_ORDER:
        for statistic in ("value", "num_windows", "num_sessions"):
            column = (bucket, statistic)
            if column in wide.columns:
                bucket_columns.append(column)
    return wide.reindex(columns=pd.MultiIndex.from_tuples(bucket_columns))


def summarise_daily_metric_vs_volatility(
    daily_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise a daily model metric's association with market volatility.

    The input is the table returned by
    :func:`plot_daily_metric_vs_volatility`.  ``pearson`` and ``spearman``
    quantify association between the selected model's daily metric and the
    session's cross-asset median realised volatility.
    """

    required = {
        "model",
        "horizon",
        "metric",
        "pearson",
        "spearman",
        "session_date",
    }
    missing = required.difference(daily_results.columns)
    if missing:
        raise KeyError(
            f"daily_results is missing columns: {sorted(missing)}."
        )
    first = daily_results.iloc[0]
    return pd.DataFrame(
        [
            {
                "model": first["model"],
                "horizon": int(first["horizon"]),
                "metric": first["metric"],
                "metric_display_name": first.get(
                    "metric_display_name",
                    DEFAULT_METRIC_DISPLAY_NAMES.get(
                        first["metric"], str(first["metric"]).replace("_", " ")
                    ),
                ),
                "pearson": float(first["pearson"]),
                "spearman": float(first["spearman"]),
                "sessions": int(daily_results["session_date"].nunique()),
                "is_persistence_relative": bool(
                    first.get("is_persistence_relative", False)
                ),
            }
        ]
    )


def summarise_daily_error_vs_volatility(
    daily_results: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-compatible alias for daily metric-volatility summaries."""

    return summarise_daily_metric_vs_volatility(daily_results)


def plot_daily_error_difference(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    benchmark_name: str,
    horizon: int,
    metric_name: str = "cumulative_log_change_mae",
    cumulative: bool = False,
    figsize: tuple[float, float] = (12.0, 5.5),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot daily model-minus-benchmark error differences."""

    table = analysis.daily_error_differences(
        model_name=model_name,
        benchmark_name=benchmark_name,
        horizon=horizon,
        metric_name=metric_name,
    )
    y_column = "cumulative_difference" if cumulative else "difference"
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(table["session_date"], table[y_column], marker="o", markersize=3)
    ax.axhline(0.0, linewidth=1.2, linestyle="--")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_xlabel("Test session")
    ax.set_ylabel(
        ("Cumulative " if cumulative else "")
        + f"{_format_model_name(model_name)} minus {_format_model_name(benchmark_name)}"
    )
    ax.set_title(
        f"Daily {DEFAULT_METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace('_', ' '))} "
        f"difference — {horizon} minutes"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return table, fig


def plot_daily_metric_vs_volatility(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    horizon: int,
    metric_name: str,
    annotate_extremes: int = 5,
    show_reference_line: bool = True,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Scatter one model's daily metric against daily market volatility.

    Parameters
    ----------
    model_name:
        Model whose own daily metric is plotted.
    horizon:
        Forecast horizon in minutes.
    metric_name:
        Any standard evaluator metric, or
        ``"mae_difference_vs_persistence"``.  Persistence-relative metrics
        remain persistence-relative by their own definitions; no additional
        benchmark subtraction is applied.  Correlation metrics are therefore
        plotted as the selected model's correlations, not as correlation
        differences versus persistence.
    annotate_extremes:
        Number of lowest and highest daily metric values to label.
    show_reference_line:
        Draw the metric's natural reference where one is defined, such as
        zero for CLG-MAE difference, one for relative MAE, or 0.5 for win
        rate.  This is a metric reference, not an extra benchmark operation.
    """

    table = analysis.daily_metric_values(
        model_name=model_name,
        metric_name=metric_name,
        horizon=horizon,
    ).copy()
    x = table["daily_market_realised_volatility"].to_numpy(dtype=np.float64)
    y = table["value"].to_numpy(dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x_valid = x[finite]
    y_valid = y[finite]

    pearson = _pearson_1d(x_valid, y_valid)
    if x_valid.size >= 2:
        spearman_result = stats.spearmanr(x_valid, y_valid, nan_policy="omit")
        spearman = float(spearman_result.statistic)
    else:
        spearman = float("nan")

    display_name = str(
        table["metric_display_name"].iloc[0]
        if "metric_display_name" in table.columns
        else DEFAULT_METRIC_DISPLAY_NAMES.get(
            metric_name, metric_name.replace("_", " ")
        )
    )
    reference_value = (
        table["reference_value"].iloc[0]
        if "reference_value" in table.columns
        else None
    )
    if pd.isna(reference_value):
        reference_value = None

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_valid, y_valid, alpha=0.8)

    if show_reference_line and reference_value is not None:
        ax.axhline(
            float(reference_value),
            linewidth=1.2,
            linestyle="--",
            label=f"Metric reference = {float(reference_value):g}",
        )

    if x_valid.size >= 2 and np.unique(x_valid).size > 1:
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        x_grid = np.linspace(np.min(x_valid), np.max(x_valid), 100)
        ax.plot(
            x_grid,
            intercept + slope * x_grid,
            linewidth=1.4,
            label="Linear trend",
        )

    if annotate_extremes > 0 and finite.any():
        finite_table = table.loc[finite].copy()
        extreme = pd.concat(
            [
                finite_table.nsmallest(annotate_extremes, "value"),
                finite_table.nlargest(annotate_extremes, "value"),
            ]
        ).drop_duplicates("session_date")
        for row in extreme.itertuples():
            ax.annotate(
                pd.Timestamp(row.session_date).strftime("%Y-%m-%d"),
                (row.daily_market_realised_volatility, row.value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Daily cross-asset median realised volatility")
    ax.set_ylabel(display_name)
    ax.set_title(
        f"{_format_model_name(model_name)} — Daily {display_name} vs realised "
        f"volatility at {horizon} minutes\n"
        f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    table["pearson"] = pearson
    table["spearman"] = spearman
    return table, fig


def plot_daily_error_vs_volatility(
    analysis: FinancialResultAnalysis,
    *,
    model_name: str,
    horizon: int,
    metric_name: str,
    annotate_extremes: int = 5,
    show_reference_line: bool = True,
    figsize: tuple[float, float] = (9.0, 6.0),
) -> tuple[pd.DataFrame, plt.Figure]:
    """Backward-compatible name for :func:`plot_daily_metric_vs_volatility`.

    The function no longer accepts a benchmark model.  It plots the selected
    model's daily metric directly against realised volatility.
    """

    return plot_daily_metric_vs_volatility(
        analysis,
        model_name=model_name,
        horizon=horizon,
        metric_name=metric_name,
        annotate_extremes=annotate_extremes,
        show_reference_line=show_reference_line,
        figsize=figsize,
    )


__all__ = [
    "DEFAULT_MODEL_DISPLAY_NAMES",
    "FinancialResultAnalysis",
    "PredictionSource",
    "STOCK_METRIC_SPECS",
    "load_prediction_source",
    "make_time_of_day_metric_table",
    "make_top_bottom_stock_table",
    "plot_adf_pvalue_distribution",
    "plot_adf_pvalues_by_stock",
    "plot_daily_error_difference",
    "plot_daily_error_vs_volatility",
    "plot_daily_metric_vs_volatility",
    "plot_metric_by_stock_volatility",
    "plot_metric_vs_adf",
    "plot_persistence_headroom",
    "plot_split_volatility_distribution",
    "plot_stock_horizon_heatmap",
    "plot_stock_metric_by_horizon",
    "plot_stock_metric_ecdf",
    "plot_time_of_day_metric",
    "summarise_daily_error_vs_volatility",
    "summarise_daily_metric_vs_volatility",
]

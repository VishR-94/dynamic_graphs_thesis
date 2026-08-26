from __future__ import annotations

"""Read-only evaluation helpers for the final GraphTCN weather experiments.

The functions in this module operate only on the prediction, metric, and graph
artifacts already exported by the fixed-architecture weather transfer runs.
They do not construct models, rerun inference, or modify saved experiment
files.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch

from src.weather_benchmark.artifacts import safe_torch_load
from src.weather_benchmark.final_transfer import (
    CITY_DISPLAY_NAMES,
    FINAL_TRANSFER_CITIES,
    FINAL_TRANSFER_HORIZONS,
    FINAL_TRANSFER_TEST_YEARS,
    SELECTED_MODERN_TCN_ARCHITECTURES,
    collect_selected_modern_tcn_transfer_metrics,
    selected_transfer_run_directory,
)


_CITY_ALIASES: dict[str, str] = {
    "capetown": "capetown",
    "cape town": "capetown",
    "ct": "capetown",
    "hongkong": "hongkong",
    "hong kong": "hongkong",
    "hk": "hongkong",
    "london": "london",
    "newyork": "newyork",
    "new york": "newyork",
    "nyc": "newyork",
    "singapore": "singapore",
}

_HORIZON_LABELS: dict[int, str] = {
    4: "1 day",
    12: "3 days",
    28: "7 days",
    120: "30 days",
}

_GRAPH_COMPONENT_ALIASES: dict[str, str] = {
    "selected": "selected",
    "mixed": "selected",
    "mixture": "selected",
    "dynamic": "dynamic",
    "base": "base",
    "static": "base",
}


def _canonical_city(city: str) -> str:
    key = " ".join(str(city).strip().lower().replace("_", " ").split())
    if key not in _CITY_ALIASES:
        raise ValueError(
            f"Unsupported city {city!r}. Expected one of: "
            f"{', '.join(CITY_DISPLAY_NAMES[value] for value in FINAL_TRANSFER_CITIES)}."
        )
    return _CITY_ALIASES[key]


def _resolve_cities(city: str | None) -> tuple[str, ...]:
    if city is None:
        return tuple(FINAL_TRANSFER_CITIES)
    return (_canonical_city(city),)


def _resolve_years(test_year: int | None) -> tuple[int, ...]:
    if test_year is None:
        return tuple(int(value) for value in FINAL_TRANSFER_TEST_YEARS)
    year = int(test_year)
    if year not in FINAL_TRANSFER_TEST_YEARS:
        raise ValueError(
            f"Unsupported test year {year}. Expected one of "
            f"{FINAL_TRANSFER_TEST_YEARS}."
        )
    return (year,)


def _resolve_horizons(horizon: int | None) -> tuple[int, ...]:
    if horizon is None:
        return tuple(int(value) for value in FINAL_TRANSFER_HORIZONS)
    value = int(horizon)
    if value not in FINAL_TRANSFER_HORIZONS:
        raise ValueError(
            f"Unsupported horizon {value}. Expected one of "
            f"{FINAL_TRANSFER_HORIZONS}."
        )
    return (value,)


def _run_directory(
    *,
    weather_root: str | Path,
    city: str,
    test_year: int,
    horizon: int,
) -> Path:
    specification = SELECTED_MODERN_TCN_ARCHITECTURES[int(horizon)]
    return selected_transfer_run_directory(
        output_root=Path(weather_root).expanduser(),
        city=city,
        test_year=int(test_year),
        architecture=specification,
    )


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required weather artifact not found:\n{path}")
    return path


def _load_prediction_result(run_directory: Path) -> dict[str, Any]:
    payload = safe_torch_load(
        _require_file(run_directory / "best_test_predictions.pt"),
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise TypeError("The saved prediction artifact is not a mapping.")
    result = payload.get("prediction_result", payload)
    if not isinstance(result, Mapping):
        raise TypeError("The prediction_result payload is not a mapping.")
    return dict(result)


def _load_graph_artifacts(run_directory: Path) -> dict[str, Any]:
    payload = safe_torch_load(
        _require_file(run_directory / "best_test_graphs.pt"),
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise TypeError("The saved graph artifact is not a mapping.")
    result = payload.get("graph_artifacts", payload)
    if not isinstance(result, Mapping):
        raise TypeError("The graph_artifacts payload is not a mapping.")
    return dict(result)


def _load_static_correlation_prior(run_directory: Path) -> dict[str, Any]:
    payload = safe_torch_load(
        _require_file(run_directory / "static_correlation_prior.pt"),
        map_location="cpu",
    )
    if not isinstance(payload, Mapping):
        raise TypeError("The saved static-correlation prior is not a mapping.")
    return dict(payload)


def _to_cpu_tensor(value: Any, *, name: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"Missing tensor field: {name}.")
    return torch.as_tensor(value).detach().cpu()


def _format_weather_table(table: pd.DataFrame, *, caption: str) -> Any:
    formatters = {
        column: ("{:.2f}" if column[1] == "ε%" else "{:.4f}")
        for column in table.columns
    }
    return (
        table.style
        .format(formatters, na_rep="—")
        .set_caption(caption)
        .set_properties(**{"text-align": "center", "padding": "4px 10px"})
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("caption-side", "top"),
                        ("font-size", "18px"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding-bottom", "8px"),
                    ],
                },
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("border-top", "2px solid black"),
                        ("border-bottom", "2px solid black"),
                        ("margin-bottom", "24px"),
                    ],
                },
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("font-weight", "bold"),
                        ("padding", "4px 10px"),
                        ("border-bottom", "1px solid #666"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("padding", "4px 10px"),
                    ],
                },
                {
                    "selector": "tbody tr",
                    "props": [("border-bottom", "1px solid #d9d9d9")],
                },
            ]
        )
    )


def show_weather_results(
    *,
    weather_root: str | Path,
    city: str | None = None,
    test_year: int | None = None,
    horizon: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Display central-node test metrics for any selected weather scope.

    Parameters
    ----------
    weather_root:
        Root of the saved weather artifacts, e.g.
        ``/content/drive/MyDrive/dissertation/weather``.
    city:
        A city name or ``None`` for all five cities.
    test_year:
        One of 2016, 2017, or 2018; ``None`` selects all three years.
    horizon:
        One of 4, 12, 28, or 120; ``None`` selects all four horizons.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One underlying multi-index result table for each selected city.
    """

    cities = _resolve_cities(city)
    years = _resolve_years(test_year)
    horizons = _resolve_horizons(horizon)

    metrics = collect_selected_modern_tcn_transfer_metrics(
        output_root=Path(weather_root).expanduser(),
        cities=cities,
        test_years=years,
        horizons=horizons,
    )
    if metrics.empty:
        raise RuntimeError("No weather result rows were found.")

    invalid = metrics.loc[
        ~metrics["status"].eq("completed")
        | metrics[["test_r", "test_mae", "test_smape"]].isna().any(axis=1)
    ]
    if not invalid.empty:
        raise RuntimeError(
            "Some requested weather runs are incomplete or missing metrics:\n"
            + invalid[
                [
                    "city",
                    "test_year",
                    "horizon",
                    "status",
                    "missing_artifacts",
                    "run_directory",
                ]
            ].to_string(index=False)
        )

    year_labels = [str(value) for value in years]
    column_groups = list(year_labels)
    if test_year is None and len(years) > 1:
        column_groups.append("Average")
    metric_labels = ("r", "MAE", "ε%")
    columns = pd.MultiIndex.from_product(
        [column_groups, metric_labels], names=(None, None)
    )

    tables: dict[str, pd.DataFrame] = {}
    for canonical in cities:
        table = pd.DataFrame(
            index=pd.Index(horizons, name="H"),
            columns=columns,
            dtype=float,
        )
        subset = metrics.loc[metrics["city"].eq(canonical)].copy()
        if subset.duplicated(["test_year", "horizon"]).any():
            raise RuntimeError(
                "Multiple result rows map to the same city/year/horizon."
            )
        for row in subset.itertuples(index=False):
            year_label = str(int(row.test_year))
            h = int(row.horizon)
            table.loc[h, (year_label, "r")] = float(row.test_r)
            table.loc[h, (year_label, "MAE")] = float(row.test_mae)
            table.loc[h, (year_label, "ε%")] = float(row.test_smape)

        if "Average" in column_groups:
            for metric_label in metric_labels:
                table.loc[:, ("Average", metric_label)] = table.loc[
                    :, [(year_label, metric_label) for year_label in year_labels]
                ].mean(axis=1)

        if table.isna().any().any():
            raise RuntimeError(
                f"The result table for {CITY_DISPLAY_NAMES[canonical]} "
                "contains missing values."
            )

        tables[canonical] = table
        try:
            from IPython.display import display

            display(
                _format_weather_table(
                    table,
                    caption=f"GraphTCN — {CITY_DISPLAY_NAMES[canonical]}",
                )
            )
        except ImportError:
            print(f"\nGraphTCN — {CITY_DISPLAY_NAMES[canonical]}")
            print(table)

    return tables


def _graph_scalar(
    artifacts: Mapping[str, Any],
    *,
    tensor_key: str,
    scalar_key: str,
    name: str,
) -> float:
    value = artifacts.get(scalar_key)
    if value is None:
        value = artifacts.get(tensor_key)
    tensor = _to_cpu_tensor(value, name=name).float().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(
            f"Expected one learned {name} value, received shape "
            f"{tuple(tensor.shape)}."
        )
    result = float(tensor.item())
    if not np.isfinite(result):
        raise ValueError(f"The learned {name} value is not finite: {result}.")
    return result


def show_weather_alpha_beta_table(
    *,
    weather_root: str | Path,
) -> pd.DataFrame:
    """Display final learned alpha and beta for all fixed GraphTCN runs.

    Rows are grouped by city with one sub-row per forecast horizon. Columns are
    grouped by test year and contain the learned static--dynamic mixture weight
    ``alpha`` and temporal--spatial mixture weight ``beta`` from the
    validation-selected checkpoint used to export the test artifacts.
    """

    years = tuple(int(value) for value in FINAL_TRANSFER_TEST_YEARS)
    horizons = tuple(int(value) for value in FINAL_TRANSFER_HORIZONS)
    columns = pd.MultiIndex.from_product(
        [[str(year) for year in years], ("α", "β")],
        names=(None, None),
    )
    index = pd.MultiIndex.from_tuples(
        [
            (CITY_DISPLAY_NAMES[city], horizon)
            for city in FINAL_TRANSFER_CITIES
            for horizon in horizons
        ],
        names=("City", "H"),
    )
    table = pd.DataFrame(index=index, columns=columns, dtype=float)

    for city in FINAL_TRANSFER_CITIES:
        display_name = CITY_DISPLAY_NAMES[city]
        for horizon in horizons:
            for year in years:
                run_directory = _run_directory(
                    weather_root=weather_root,
                    city=city,
                    test_year=year,
                    horizon=horizon,
                )
                artifacts = _load_graph_artifacts(run_directory)
                table.loc[(display_name, horizon), (str(year), "α")] = (
                    _graph_scalar(
                        artifacts,
                        tensor_key="alpha",
                        scalar_key="dynamic_alpha",
                        name="alpha",
                    )
                )
                table.loc[(display_name, horizon), (str(year), "β")] = (
                    _graph_scalar(
                        artifacts,
                        tensor_key="beta",
                        scalar_key="spatial_beta",
                        name="beta",
                    )
                )

    if table.isna().any().any():
        raise RuntimeError("The learned alpha/beta table contains missing values.")

    try:
        from IPython.display import display

        display(
            _format_weather_table(
                table,
                caption="GraphTCN — learned α and β values",
            )
        )
    except ImportError:
        print("\nGraphTCN — learned alpha and beta values")
        print(table)

    return table


def load_weather_forecast_series(
    *,
    weather_root: str | Path,
    city: str,
    horizon: int,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
) -> pd.DataFrame:
    """Load final-horizon central-node forecasts for plotting or analysis."""

    canonical = _canonical_city(city)
    h = _resolve_horizons(horizon)[0]
    years = tuple(int(value) for value in test_years)
    if not years or len(set(years)) != len(years):
        raise ValueError("test_years must contain distinct years.")
    invalid_years = [value for value in years if value not in FINAL_TRANSFER_TEST_YEARS]
    if invalid_years:
        raise ValueError(f"Unsupported test years: {invalid_years}.")

    frames: list[pd.DataFrame] = []
    for year in years:
        run_directory = _run_directory(
            weather_root=weather_root,
            city=canonical,
            test_year=year,
            horizon=h,
        )
        result = _load_prediction_result(run_directory)
        y_pred = _to_cpu_tensor(result.get("y_pred"), name="y_pred").float()
        y_true = _to_cpu_tensor(result.get("y_true"), name="y_true").float()
        target_times = _to_cpu_tensor(
            result.get("target_times_ns"), name="target_times_ns"
        ).long()

        if y_pred.shape != y_true.shape or y_pred.ndim != 4:
            raise ValueError(
                "Expected matching prediction and target tensors with shape "
                f"[W,H,N,1], received {tuple(y_pred.shape)} and "
                f"{tuple(y_true.shape)}."
            )
        if int(y_pred.shape[1]) != h:
            raise ValueError(
                f"H={h} run contains {y_pred.shape[1]} forecast positions."
            )
        if target_times.shape != y_pred.shape[:2]:
            raise ValueError(
                "target_times_ns must have shape [W,H], received "
                f"{tuple(target_times.shape)}."
            )

        central = int(result.get("central_node_index", 0))
        if not 0 <= central < int(y_pred.shape[2]):
            raise IndexError(f"Invalid central node index: {central}.")
        timestamps = pd.to_datetime(
            target_times[:, -1].numpy(), unit="ns", utc=False
        )
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "actual": y_true[:, -1, central, 0].numpy(),
                "prediction": y_pred[:, -1, central, 0].numpy(),
                "test_year": int(year),
                "city": canonical,
                "horizon": h,
                "window_index": np.arange(int(y_pred.shape[0]), dtype=int),
                "run_directory": str(run_directory),
            }
        )
        if frame["timestamp"].duplicated().any():
            raise RuntimeError(
                f"Duplicate final-target timestamps in {canonical}, {year}, H={h}."
            )
        if not frame["timestamp"].is_monotonic_increasing:
            frame = frame.sort_values("timestamp").reset_index(drop=True)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True).sort_values(
        "timestamp"
    ).reset_index(drop=True)
    duplicate_times = combined["timestamp"].duplicated(keep=False)
    if duplicate_times.any():
        duplicate_values = combined.loc[duplicate_times]
        for _, group in duplicate_values.groupby("timestamp"):
            if not np.allclose(group["actual"], group["actual"].iloc[0]):
                raise RuntimeError(
                    "Overlapping test-year artifacts disagree on the true target."
                )
        combined = combined.drop_duplicates("timestamp", keep="first")
    return combined.reset_index(drop=True)


def plot_weather_forecasts(
    *,
    weather_root: str | Path,
    city: str,
    horizon: int,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    figsize: tuple[float, float] = (18.0, 6.0),
    linewidth_actual: float = 1.4,
    linewidth_forecast: float = 1.0,
) -> tuple[pd.DataFrame, Figure, Axes]:
    """Plot central-node truth and final-horizon forecasts across test years."""

    data = load_weather_forecast_series(
        weather_root=weather_root,
        city=city,
        horizon=horizon,
        test_years=test_years,
    )
    canonical = _canonical_city(city)
    h = int(horizon)
    figure, axis = plt.subplots(figsize=figsize)
    axis.plot(
        data["timestamp"],
        data["actual"],
        color="black",
        linewidth=float(linewidth_actual),
        linestyle="dashed",
        label="Observed central T850",
        zorder=3,
    )
    for year in tuple(int(value) for value in test_years):
        subset = data.loc[data["test_year"].eq(year)]
        axis.plot(
            subset["timestamp"],
            subset["prediction"],
            linewidth=float(linewidth_forecast),
            label=f"GraphTCN forecast — {year}",
            alpha=0.9,
        )
    lead = _HORIZON_LABELS.get(h, f"{h} steps")
    axis.set_title(
        f"GraphTCN final-horizon forecasts — {CITY_DISPLAY_NAMES[canonical]} "
        f"— H={h} ({lead})"
    )
    axis.set_xlabel("Final forecast target time")
    axis.set_ylabel("Central-grid T850 (K)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    return data, figure, axis


def _resolve_graph_component(component: str) -> str:
    value = str(component).strip().lower()
    if value not in _GRAPH_COMPONENT_ALIASES:
        raise ValueError(
            "graph_component must be one of: selected, dynamic, or base/static."
        )
    return _GRAPH_COMPONENT_ALIASES[value]


def _collapse_graph_heads(
    tensor: torch.Tensor,
    *,
    head: str | int,
    component: str,
) -> torch.Tensor:
    """Return [W,N,N] or [N,N] after selecting/averaging graph heads."""

    value = tensor.float()
    if component in {"selected", "dynamic"}:
        if value.ndim == 3:  # [W,N,N]
            return value
        if value.ndim != 4:  # expected [W,G,N,N]
            raise ValueError(
                f"{component} graph must have shape [W,G,N,N] or [W,N,N]; "
                f"received {tuple(value.shape)}."
            )
        if head == "mean":
            return value.mean(dim=1)
        index = int(head)
        if not 0 <= index < int(value.shape[1]):
            raise IndexError(
                f"Graph head {index} is outside [0, {value.shape[1] - 1}]."
            )
        return value[:, index]

    # Static/base graph is saved once, normally as [G,N,N].
    if value.ndim == 2:
        return value
    if value.ndim != 3:
        raise ValueError(
            f"base graph must have shape [G,N,N] or [N,N]; "
            f"received {tuple(value.shape)}."
        )
    if head == "mean":
        return value.mean(dim=0)
    index = int(head)
    if not 0 <= index < int(value.shape[0]):
        raise IndexError(
            f"Graph head {index} is outside [0, {value.shape[0] - 1}]."
        )
    return value[index]


def _make_weather_graph_window_table(
    artifacts: Mapping[str, Any],
    *,
    num_windows: int,
) -> pd.DataFrame:
    """Return Graph-Hub-style day/window metadata for saved weather graphs."""

    origins_raw = artifacts.get("forecast_origin_times_ns")
    if origins_raw is None:
        origins_raw = artifacts.get("dates")
    if origins_raw is None:
        raise KeyError(
            "The graph artifact does not contain forecast-origin timestamps."
        )

    if torch.is_tensor(origins_raw) or isinstance(origins_raw, np.ndarray):
        origins = _to_cpu_tensor(
            origins_raw,
            name="forecast_origin_times_ns",
        ).long()
        if origins.ndim != 1 or int(origins.shape[0]) != int(num_windows):
            raise ValueError(
                "forecast_origin_times_ns must have shape [W], received "
                f"{tuple(origins.shape)} for W={num_windows}."
            )
        origin_times = pd.to_datetime(origins.numpy(), unit="ns", utc=False)
    else:
        origin_times = pd.to_datetime(list(origins_raw), utc=False)
        if len(origin_times) != int(num_windows):
            raise ValueError(
                "Saved origin timestamps do not match the graph count: "
                f"{len(origin_times)} versus {num_windows}."
            )

    targets_raw = artifacts.get("target_times_ns")
    if targets_raw is None:
        final_target_times = pd.DatetimeIndex([pd.NaT] * int(num_windows))
    else:
        targets = _to_cpu_tensor(targets_raw, name="target_times_ns").long()
        if targets.ndim == 1:
            final_targets = targets
        elif targets.ndim == 2:
            final_targets = targets[:, -1]
        else:
            raise ValueError(
                "target_times_ns must have shape [W] or [W,H], received "
                f"{tuple(targets.shape)}."
            )
        if int(final_targets.shape[0]) != int(num_windows):
            raise ValueError(
                "Saved final-target timestamps do not match the graph count."
            )
        final_target_times = pd.to_datetime(
            final_targets.numpy(),
            unit="ns",
            utc=False,
        )

    table = pd.DataFrame(
        {
            "global_window_index": np.arange(int(num_windows), dtype=int),
            "forecast_origin_time": origin_times,
            "final_target_time": final_target_times,
        }
    )
    table["day"] = table["forecast_origin_time"].dt.normalize()
    table["window_within_day"] = (
        table.groupby("day", sort=False).cumcount() + 1
    ).astype(int)
    return table


def _select_weather_graph_windows(
    table: pd.DataFrame,
    *,
    day: str | pd.Timestamp | None,
    window: int | None,
) -> tuple[pd.DataFrame, str]:
    """Apply the same day/window selection convention used by Graph Hub."""

    selected = table
    resolved_day: pd.Timestamp | None = None
    resolved_window: int | None = None

    if day is not None:
        resolved_day = pd.Timestamp(day).normalize()
        selected = selected.loc[selected["day"].eq(resolved_day)]

    if window is not None:
        resolved_window = int(window)
        if resolved_window <= 0:
            raise ValueError("window must be a positive, one-based integer.")
        selected = selected.loc[
            selected["window_within_day"].eq(resolved_window)
        ]

    if selected.empty:
        first_day = table["day"].min().date()
        last_day = table["day"].max().date()
        maximum_window = int(table["window_within_day"].max())
        raise ValueError(
            "No saved weather graphs match the requested selection. "
            f"Available origin dates run from {first_day} to {last_day}; "
            f"within-day windows are one-based and run up to {maximum_window}."
        )

    if resolved_day is None and resolved_window is None:
        description = f"mean across all {len(selected):,} test windows"
    elif resolved_day is not None and resolved_window is None:
        description = (
            f"mean across {len(selected):,} windows on "
            f"{resolved_day.date()}"
        )
    elif resolved_day is None and resolved_window is not None:
        description = (
            f"mean of window {resolved_window} across "
            f"{selected['day'].nunique():,} days"
        )
    else:
        description = (
            f"{resolved_day.date()}, window {resolved_window}"
        )

    return selected.reset_index(drop=True), description


def _draw_weather_graph_heatmap(
    *,
    axis: Axes,
    values: np.ndarray,
    node_order: Sequence[str],
    upper: float,
    annotate: bool,
    value_format: str,
    title: str,
) -> Any:
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    image = axis.imshow(
        values,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        vmin=0.0,
        vmax=upper,
    )
    axis.set_xticks(np.arange(len(node_order)))
    axis.set_yticks(np.arange(len(node_order)))
    axis.set_xticklabels(node_order)
    axis.set_yticklabels(node_order)
    axis.set_xlabel("Source node (influences target)")
    axis.set_ylabel("Target node (receives influence)")
    axis.set_title(title)

    if annotate:
        threshold = upper * 0.5
        for row in range(len(node_order)):
            for column in range(len(node_order)):
                value = float(values[row, column])
                if not np.isfinite(value):
                    continue
                axis.text(
                    column,
                    row,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
    return image


def plot_weather_graph(
    *,
    weather_root: str | Path,
    city: str,
    horizon: int,
    test_year: int,
    day: str | pd.Timestamp | None = None,
    window: int | None = None,
    graph_component: str = "selected",
    head: str | int = "mean",
    annotate: bool = True,
    figsize: tuple[float, float] = (8.5, 14.5),
    value_format: str = ".2f",
    vmax: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Figure, Axes]:
    """Plot weather graphs using the same day/window convention as Graph Hub.

    Selection rules
    ---------------
    ``day=None, window=None``
        Average every saved graph in the test split.
    ``day=<date>, window=None``
        Average every graph whose forecast origin falls on that date.
    ``day=None, window=<n>``
        Average the one-based within-day window ``n`` across all dates.
    ``day=<date>, window=<n>``
        Plot the exact graph for that date and within-day window.

    Rows are target nodes and columns are source nodes, i.e.
    ``A[target, source]``. The displayed colour map matches Graph Hub: white
    represents zero/masked self-edges and progressively darker red represents
    larger adjacency weights. The training-only, row-normalised static
    correlation prior for the same city and test-year split is displayed below
    the learned graph using the identical colour scale.
    """

    canonical = _canonical_city(city)
    year = _resolve_years(test_year)[0]
    h = _resolve_horizons(horizon)[0]
    component = _resolve_graph_component(graph_component)
    if head != "mean":
        head = int(head)

    run_directory = _run_directory(
        weather_root=weather_root,
        city=canonical,
        test_year=year,
        horizon=h,
    )
    artifacts = _load_graph_artifacts(run_directory)
    prior_artifacts = _load_static_correlation_prior(run_directory)
    node_order = list(artifacts.get("node_order", artifacts.get("asset_cols", [])))
    if not node_order:
        raise KeyError("The graph artifact does not contain node labels.")
    prior_node_order = list(prior_artifacts.get("node_order", []))
    if prior_node_order and prior_node_order != node_order:
        raise ValueError(
            "The static-correlation prior node order does not match the "
            "learned graph artifact."
        )
    prior_matrix = _to_cpu_tensor(
        prior_artifacts.get("row_normalised_prior"),
        name="row_normalised_prior",
    ).float()
    if prior_matrix.shape != (len(node_order), len(node_order)):
        raise ValueError(
            "The static-correlation prior shape does not match the node order: "
            f"{tuple(prior_matrix.shape)} versus {len(node_order)} nodes."
        )

    raw_component = artifacts.get(component)
    if raw_component is None:
        raise KeyError(
            f"The saved graph artifact does not contain component {component!r}."
        )
    collapsed = _collapse_graph_heads(
        _to_cpu_tensor(raw_component, name=component),
        head=head,
        component=component,
    )

    if component == "base":
        num_windows = int(
            _to_cpu_tensor(
                artifacts.get("forecast_origin_times_ns"),
                name="forecast_origin_times_ns",
            ).shape[0]
        )
    else:
        if collapsed.ndim != 3:
            raise ValueError(
                "Expected a window-indexed graph tensor with shape [W,N,N], "
                f"received {tuple(collapsed.shape)}."
            )
        num_windows = int(collapsed.shape[0])

    window_table = _make_weather_graph_window_table(
        artifacts,
        num_windows=num_windows,
    )
    selected_rows, selection_description = _select_weather_graph_windows(
        window_table,
        day=day,
        window=window,
    )

    selected_indices = torch.as_tensor(
        selected_rows["global_window_index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
    )
    if component == "base":
        matrix = collapsed
    else:
        matrix = collapsed.index_select(0, selected_indices).mean(dim=0)

    matrix = matrix.float()
    if matrix.shape != (len(node_order), len(node_order)):
        raise ValueError(
            "The plotted graph shape does not match the node order: "
            f"{tuple(matrix.shape)} versus {len(node_order)} nodes."
        )

    exact_window = len(selected_rows) == 1
    metadata: dict[str, Any] = {
        "city": canonical,
        "city_display_name": CITY_DISPLAY_NAMES[canonical],
        "test_year": year,
        "horizon": h,
        "graph_component": component,
        "head": head,
        "requested_day": None if day is None else str(pd.Timestamp(day).date()),
        "requested_window": None if window is None else int(window),
        "selection": selection_description,
        "selected_windows": int(len(selected_rows)),
        "selected_days": int(selected_rows["day"].nunique()),
        "forecast_origin_time": (
            selected_rows.loc[0, "forecast_origin_time"]
            if exact_window
            else None
        ),
        "final_target_time": (
            selected_rows.loc[0, "final_target_time"]
            if exact_window
            else None
        ),
        "graph_orientation": artifacts.get(
            "graph_orientation",
            artifacts.get("orientation", "A[target, source]"),
        ),
        "run_directory": str(run_directory),
        "static_correlation_prior_file": str(
            run_directory / "static_correlation_prior.pt"
        ),
        "static_correlation_prior_construction": prior_artifacts.get(
            "construction"
        ),
    }

    matrix_values = matrix.numpy()
    matrix_frame = pd.DataFrame(
        matrix_values,
        index=pd.Index(node_order, name="Target node"),
        columns=pd.Index(node_order, name="Source node"),
    )

    plotted_values = matrix_values.copy()
    prior_values = prior_matrix.numpy().copy()
    np.fill_diagonal(plotted_values, np.nan)
    np.fill_diagonal(prior_values, np.nan)
    combined_finite = np.concatenate(
        [
            plotted_values[np.isfinite(plotted_values)],
            prior_values[np.isfinite(prior_values)],
        ]
    )
    upper = (
        float(vmax)
        if vmax is not None
        else (float(np.max(combined_finite)) if combined_finite.size else 1.0)
    )
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        constrained_layout=True,
    )
    learned_axis, prior_axis = axes

    component_label = {
        "selected": "selected static–dynamic mixture",
        "dynamic": "dynamic graph",
        "base": "learned static graph",
    }[component]
    learned_title = (
        f"GraphTCN {component_label} — {CITY_DISPLAY_NAMES[canonical]} "
        f"— {year} — H={h}\n{selection_description}"
    )
    if exact_window:
        learned_title += (
            f"\norigin {metadata['forecast_origin_time']} — "
            f"target {metadata['final_target_time']}"
        )

    learned_image = _draw_weather_graph_heatmap(
        axis=learned_axis,
        values=plotted_values,
        node_order=node_order,
        upper=upper,
        annotate=annotate,
        value_format=value_format,
        title=learned_title,
    )
    _draw_weather_graph_heatmap(
        axis=prior_axis,
        values=prior_values,
        node_order=node_order,
        upper=upper,
        annotate=annotate,
        value_format=value_format,
        title=(
            "Training-only static correlation prior "
            f"— {CITY_DISPLAY_NAMES[canonical]} — test year {year}"
        ),
    )
    colourbar = figure.colorbar(
        learned_image,
        ax=(learned_axis, prior_axis),
        fraction=0.025,
        pad=0.02,
    )
    colourbar.set_label("Adjacency weight")
    return matrix_frame, metadata, figure, learned_axis


__all__ = [
    "load_weather_forecast_series",
    "plot_weather_forecasts",
    "plot_weather_graph",
    "show_weather_alpha_beta_table",
    "show_weather_results",
]

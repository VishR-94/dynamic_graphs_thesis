from __future__ import annotations

"""Fixed-architecture ModernTCN transfer across Sonnet weather cities/years.

This module is deliberately additive.  It reuses the existing weather data,
model, training, resume, metric, checkpoint and graph-export implementations.
The architecture is fixed from the Hong Kong 2018 validation sweep and is not
re-tuned for another city or test year.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import (
    MODEL_OUTPUT_DIRECTORIES,
    SUPPORTED_CITIES,
    WEATHER_HORIZON_TO_CONTEXT,
    WEATHER_NODES,
)
from .data import weather_node_column_map
from .runner import (
    ensure_weather_csv,
    modern_tcn_stride_width_run_suffix,
    preflight_weather_run,
    run_weather_suite,
)


FINAL_TRANSFER_CITIES: tuple[str, ...] = (
    "hongkong",
    "capetown",
    "london",
    "newyork",
    "singapore",
)
FINAL_TRANSFER_TEST_YEARS: tuple[int, ...] = (2016, 2017, 2018)
FINAL_TRANSFER_HORIZONS: tuple[int, ...] = (4, 12, 28, 120)
FINAL_TRANSFER_METRICS: tuple[str, ...] = ("mae", "r", "smape")
CITY_DISPLAY_NAMES: dict[str, str] = {
    "capetown": "Cape Town",
    "hongkong": "Hong Kong",
    "london": "London",
    "newyork": "New York",
    "singapore": "Singapore",
}


@dataclass(frozen=True)
class SelectedModernTCNArchitecture:
    """One validation-selected horizon-specific Graph-ModernTCN specification."""

    horizon: int
    large_kernel: int
    patch_stride: int
    d_model: int = 32
    graph_hidden_dim: int = 32
    patch_size: int = 8
    small_kernel: int = 5
    num_blocks: int = 1

    def __post_init__(self) -> None:
        horizon = int(self.horizon)
        if horizon not in WEATHER_HORIZON_TO_CONTEXT:
            raise ValueError(f"Unsupported weather horizon: {horizon}.")
        if int(self.large_kernel) < 5 or int(self.large_kernel) % 2 == 0:
            raise ValueError("large_kernel must be odd and at least 5.")
        if int(self.patch_stride) <= 0 or int(self.patch_stride) > int(self.patch_size):
            raise ValueError("patch_stride must be in [1, patch_size].")
        context = int(WEATHER_HORIZON_TO_CONTEXT[horizon])
        if context % int(self.patch_stride) != 0:
            raise ValueError(
                f"H={horizon} context {context} is not divisible by stride "
                f"{self.patch_stride}."
            )
        if int(self.d_model) <= 0 or int(self.graph_hidden_dim) <= 0:
            raise ValueError("d_model and graph_hidden_dim must be positive.")

    @property
    def context_length(self) -> int:
        return int(WEATHER_HORIZON_TO_CONTEXT[int(self.horizon)])

    @property
    def run_suffix(self) -> str:
        return modern_tcn_stride_width_run_suffix(
            kernel=int(self.large_kernel),
            patch_stride=int(self.patch_stride),
            d_model=int(self.d_model),
            graph_hidden_dim=int(self.graph_hidden_dim),
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["context_length"] = self.context_length
        values["run_suffix"] = self.run_suffix
        return values


SELECTED_MODERN_TCN_ARCHITECTURES: dict[int, SelectedModernTCNArchitecture] = {
    4: SelectedModernTCNArchitecture(
        horizon=4,
        large_kernel=7,
        patch_stride=2,
    ),
    12: SelectedModernTCNArchitecture(
        horizon=12,
        large_kernel=7,
        patch_stride=4,
    ),
    28: SelectedModernTCNArchitecture(
        horizon=28,
        large_kernel=15,
        patch_stride=8,
    ),
    120: SelectedModernTCNArchitecture(
        horizon=120,
        large_kernel=119,
        patch_stride=8,
    ),
}


def _normalise_unique_cities(cities: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value).lower().strip() for value in cities)
    if not values:
        raise ValueError("At least one city is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Cities must be unique: {values}.")
    invalid = [value for value in values if value not in SUPPORTED_CITIES]
    if invalid:
        raise ValueError(
            f"Unsupported cities {invalid}; expected values from {SUPPORTED_CITIES}."
        )
    return values


def _normalise_unique_ints(
    values: Sequence[int],
    *,
    name: str,
) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError(f"At least one {name} is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} values must be unique: {resolved}.")
    return resolved


def _validate_transfer_scope(
    *,
    cities: Sequence[str],
    test_years: Sequence[int],
    horizons: Sequence[int],
    architectures: Mapping[int, SelectedModernTCNArchitecture],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    resolved_cities = _normalise_unique_cities(cities)
    resolved_years = _normalise_unique_ints(test_years, name="test year")
    resolved_horizons = _normalise_unique_ints(horizons, name="horizon")
    unsupported = [
        value for value in resolved_horizons if value not in WEATHER_HORIZON_TO_CONTEXT
    ]
    if unsupported:
        raise ValueError(f"Unsupported weather horizons: {unsupported}.")
    missing = [value for value in resolved_horizons if value not in architectures]
    if missing:
        raise KeyError(f"No selected architecture is defined for horizons: {missing}.")
    for horizon in resolved_horizons:
        specification = architectures[int(horizon)]
        if int(specification.horizon) != int(horizon):
            raise ValueError(
                f"Architecture map key H={horizon} contains specification for "
                f"H={specification.horizon}."
            )
    return resolved_cities, resolved_years, resolved_horizons


def selected_transfer_run_directory(
    *,
    output_root: str | Path,
    city: str,
    test_year: int,
    architecture: SelectedModernTCNArchitecture,
) -> Path:
    return (
        Path(output_root).expanduser()
        / MODEL_OUTPUT_DIRECTORIES["modern_tcn_1st"]
        / str(city).lower().strip()
        / f"horizon_{int(architecture.horizon)}"
        / f"test_year_{int(test_year)}_{architecture.run_suffix}"
    )


def selected_transfer_plan(
    *,
    output_root: str | Path,
    data_cache_root: str | Path,
    cities: Sequence[str] = FINAL_TRANSFER_CITIES,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
    architectures: Mapping[
        int, SelectedModernTCNArchitecture
    ] = SELECTED_MODERN_TCN_ARCHITECTURES,
) -> pd.DataFrame:
    """Return the complete fixed-architecture city/year/horizon plan."""

    resolved_cities, resolved_years, resolved_horizons = _validate_transfer_scope(
        cities=cities,
        test_years=test_years,
        horizons=horizons,
        architectures=architectures,
    )
    output = Path(output_root).expanduser()
    cache = Path(data_cache_root).expanduser()
    rows: list[dict[str, Any]] = []
    for city in resolved_cities:
        for test_year in resolved_years:
            for horizon in resolved_horizons:
                specification = architectures[int(horizon)]
                run_directory = selected_transfer_run_directory(
                    output_root=output,
                    city=city,
                    test_year=test_year,
                    architecture=specification,
                )
                rows.append(
                    {
                        "city": city,
                        "city_display_name": CITY_DISPLAY_NAMES[city],
                        "test_year": int(test_year),
                        "validation_year": int(test_year) - 1,
                        "training_start_year": 1980,
                        "training_end_year": int(test_year) - 2,
                        "horizon": int(horizon),
                        "context_length": int(specification.context_length),
                        "large_kernel": int(specification.large_kernel),
                        "patch_size": int(specification.patch_size),
                        "patch_stride": int(specification.patch_stride),
                        "d_model": int(specification.d_model),
                        "graph_hidden_dim": int(specification.graph_hidden_dim),
                        "small_kernel": int(specification.small_kernel),
                        "num_blocks": int(specification.num_blocks),
                        "run_suffix": specification.run_suffix,
                        "data_path": str(cache / f"weather_{city}.csv"),
                        "run_directory": str(run_directory),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(resolved_cities) * len(resolved_years) * len(resolved_horizons)
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} plan rows, found {len(frame)}.")
    return frame


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_weather_city_csv(
    *,
    city: str,
    data_path: str | Path,
) -> dict[str, Any]:
    """Describe file integrity and exact duplicate spatial-node series.

    Duplicate nodes are reported but do not block execution.  This keeps the
    all-five-city experiment faithful to the supplied Sonnet city files while
    making known spatial degeneracies explicit in the saved provenance.
    """

    canonical = str(city).lower().strip()
    path = Path(data_path).expanduser().resolve()
    if canonical not in SUPPORTED_CITIES:
        raise ValueError(f"Unsupported city: {canonical}.")
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    node_map = weather_node_column_map(frame.columns)

    duplicate_pairs: list[str] = []
    nodes = list(WEATHER_NODES)
    node_values: dict[str, np.ndarray] = {
        node: frame.loc[:, list(node_map[node].values())].to_numpy()
        for node in nodes
    }
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            if np.array_equal(
                node_values[left], node_values[right], equal_nan=True
            ):
                duplicate_pairs.append(f"{left}={right}")

    # Count unique five-variable node trajectories directly rather than
    # subtracting pair counts, which would be wrong for groups of >2 copies.
    node_signatures = {
        hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        for values in node_values.values()
    }

    deltas = frame.index.to_series().diff().dropna()
    regular_six_hourly = bool(
        not deltas.empty and (deltas == pd.Timedelta(hours=6)).all()
    )
    return {
        "city": canonical,
        "city_display_name": CITY_DISPLAY_NAMES[canonical],
        "data_path": str(path),
        "sha256": _sha256(path),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "first_timestamp": frame.index[0].isoformat(),
        "last_timestamp": frame.index[-1].isoformat(),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "duplicate_columns": int(frame.columns.duplicated().sum()),
        "regular_six_hourly": regular_six_hourly,
        "exact_duplicate_node_pairs": "; ".join(duplicate_pairs),
        "exact_duplicate_node_pair_count": len(duplicate_pairs),
        "unique_spatial_node_count": len(node_signatures),
    }


def audit_all_weather_city_csvs(
    *,
    cities: Sequence[str],
    data_cache_root: str | Path,
) -> pd.DataFrame:
    resolved = _normalise_unique_cities(cities)
    rows: list[dict[str, Any]] = []
    for city in resolved:
        path = ensure_weather_csv(city, data_cache_root)
        rows.append(audit_weather_city_csv(city=city, data_path=path))
    return pd.DataFrame(rows)


def preflight_selected_modern_tcn_architectures(
    *,
    city: str,
    test_year: int,
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
    architectures: Mapping[
        int, SelectedModernTCNArchitecture
    ] = SELECTED_MODERN_TCN_ARCHITECTURES,
    device: str = "auto",
    train_batch_size: int = 16,
    validation_batch_size: int = 32,
    export_batch_size: int = 32,
    progress_update_interval: int = 50,
    prefetch_factor: int = 2,
    deterministic_runtime: bool = True,
) -> pd.DataFrame:
    _, _, resolved_horizons = _validate_transfer_scope(
        cities=(city,),
        test_years=(test_year,),
        horizons=horizons,
        architectures=architectures,
    )
    rows: list[dict[str, Any]] = []
    for horizon in resolved_horizons:
        specification = architectures[int(horizon)]
        values = preflight_weather_run(
            model_kind="modern_tcn_1st",
            city=city,
            test_year=int(test_year),
            horizon=int(horizon),
            data_path=data_path,
            output_root=output_root,
            project_root=project_root,
            device=device,
            modern_tcn_large_kernel=int(specification.large_kernel),
            modern_tcn_patch_stride=int(specification.patch_stride),
            modern_tcn_d_model=int(specification.d_model),
            modern_tcn_graph_hidden_dim=int(specification.graph_hidden_dim),
            train_batch_size=int(train_batch_size),
            validation_batch_size=int(validation_batch_size),
            export_batch_size=int(export_batch_size),
            run_suffix=specification.run_suffix,
            progress_update_interval=int(progress_update_interval),
            prefetch_factor=int(prefetch_factor),
            deterministic_runtime=bool(deterministic_runtime),
        )
        rows.append(values)
    return pd.DataFrame(rows)


def run_selected_modern_tcn_transfer(
    *,
    output_root: str | Path,
    data_cache_root: str | Path,
    project_root: str | Path,
    cities: Sequence[str] = FINAL_TRANSFER_CITIES,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
    architectures: Mapping[
        int, SelectedModernTCNArchitecture
    ] = SELECTED_MODERN_TCN_ARCHITECTURES,
    summary_directory: str | Path | None = None,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    skip_completed: bool = True,
    export_train_split: bool = True,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 0,
    continue_on_error: bool = False,
    train_batch_size: int = 16,
    validation_batch_size: int = 32,
    export_batch_size: int = 32,
    progress_update_interval: int = 50,
    prefetch_factor: int = 2,
    deterministic_runtime: bool = True,
) -> pd.DataFrame:
    """Train/evaluate the four fixed variants for every city and test year.

    The canonical run suffix is identical to the preceding stride/width sweep.
    Consequently, a completed selected Hong Kong 2018 run is safely reused when
    its full experiment signature matches; all other city/year runs are trained
    independently with their own Sonnet-aligned training and validation splits.
    """

    resolved_cities, resolved_years, resolved_horizons = _validate_transfer_scope(
        cities=cities,
        test_years=test_years,
        horizons=horizons,
        architectures=architectures,
    )
    output = Path(output_root).expanduser()
    cache = Path(data_cache_root).expanduser()
    project = Path(project_root).expanduser().resolve()
    summary_root = (
        Path(summary_directory).expanduser()
        if summary_directory is not None
        else output / "final_selected_modernTCN_transfer"
    )
    summary_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    plan = selected_transfer_plan(
        output_root=output,
        data_cache_root=cache,
        cities=resolved_cities,
        test_years=resolved_years,
        horizons=resolved_horizons,
        architectures=architectures,
    )
    _atomic_csv(plan, summary_root / "experiment_plan.csv")
    _atomic_json(
        {
            "experiment_family": "final_selected_modernTCN_transfer",
            "selection_source": (
                "Hong Kong 2018 validation-selected stride/width sweep"
            ),
            "retuning_per_city_or_year": False,
            "retrospective_earlier_year_note": (
                "The architecture was selected with Hong Kong validation year "
                "2017 for test year 2018. Applying it to test years 2016/2017 "
                "is a fixed-architecture retrospective robustness evaluation, "
                "not a temporally pristine hyperparameter-selection protocol."
            ),
            "cities": list(resolved_cities),
            "test_years": list(resolved_years),
            "horizons": list(resolved_horizons),
            "architectures": {
                str(horizon): architectures[horizon].to_dict()
                for horizon in resolved_horizons
            },
            "training": {
                "train_batch_size": int(train_batch_size),
                "validation_batch_size": int(validation_batch_size),
                "export_batch_size": int(export_batch_size),
                "max_epochs": int(max_epochs),
                "patience": int(patience),
                "num_workers": int(num_workers),
                "deterministic_runtime": bool(deterministic_runtime),
                "export_train_split": bool(export_train_split),
            },
            "total_runs": int(len(plan)),
        },
        summary_root / "experiment_manifest.json",
    )

    frames: list[pd.DataFrame] = []
    for city in resolved_cities:
        data_path = ensure_weather_csv(city, cache)
        for test_year in resolved_years:
            for horizon in resolved_horizons:
                specification = architectures[int(horizon)]
                print("#" * 96)
                print(
                    "Final fixed transfer | "
                    f"city={city} | test_year={test_year} | H={horizon} | "
                    f"K={specification.large_kernel} | "
                    f"stride={specification.patch_stride} | "
                    f"D={specification.d_model}"
                )
                frame = run_weather_suite(
                    model_kinds=("modern_tcn_1st",),
                    city=city,
                    test_year=int(test_year),
                    horizons=(int(horizon),),
                    data_path=data_path,
                    output_root=output,
                    project_root=project,
                    device=device,
                    resume=resume,
                    overwrite=overwrite,
                    skip_completed=skip_completed,
                    export_train_split=export_train_split,
                    max_epochs=max_epochs,
                    patience=patience,
                    num_workers=num_workers,
                    continue_on_error=continue_on_error,
                    modern_tcn_large_kernel=int(specification.large_kernel),
                    modern_tcn_patch_stride=int(specification.patch_stride),
                    modern_tcn_d_model=int(specification.d_model),
                    modern_tcn_graph_hidden_dim=int(
                        specification.graph_hidden_dim
                    ),
                    train_batch_size=int(train_batch_size),
                    validation_batch_size=int(validation_batch_size),
                    export_batch_size=int(export_batch_size),
                    run_suffix=specification.run_suffix,
                    progress_update_interval=int(progress_update_interval),
                    prefetch_factor=int(prefetch_factor),
                    deterministic_runtime=bool(deterministic_runtime),
                )
                frames.append(frame)
                progress = pd.concat(frames, ignore_index=True)
                _atomic_csv(progress, summary_root / "training_progress.csv")

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    _atomic_csv(result, summary_root / "training_summary.csv")
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def _metric(payload: Mapping[str, Any] | None, metric: str) -> float:
    if payload is None:
        return float("nan")
    reported = payload.get("reported", {})
    if not isinstance(reported, Mapping):
        return float("nan")
    value = reported.get(metric)
    return float("nan") if value is None else float(value)


def _resolved_architecture_from_config(values: Mapping[str, Any]) -> dict[str, int]:
    return {
        "large_kernel": int(values.get("modern_tcn_large_kernel", 15)),
        "patch_stride": int(values.get("modern_tcn_patch_stride", 4)),
        "d_model": int(values.get("modern_tcn_d_model", 32)),
        "graph_hidden_dim": int(
            values.get("modern_tcn_graph_hidden_dim", 32)
        ),
    }


def collect_selected_modern_tcn_transfer_metrics(
    *,
    output_root: str | Path,
    cities: Sequence[str] = FINAL_TRANSFER_CITIES,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
    architectures: Mapping[
        int, SelectedModernTCNArchitecture
    ] = SELECTED_MODERN_TCN_ARCHITECTURES,
) -> pd.DataFrame:
    """Collect validation/test metrics and artifact status from expected runs."""

    resolved_cities, resolved_years, resolved_horizons = _validate_transfer_scope(
        cities=cities,
        test_years=test_years,
        horizons=horizons,
        architectures=architectures,
    )
    output = Path(output_root).expanduser()
    rows: list[dict[str, Any]] = []
    required_relative_paths = (
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "best_validation_predictions.pt",
        "best_validation_graphs.pt",
        "best_validation_metrics.json",
        "best_test_predictions.pt",
        "best_test_graphs.pt",
        "best_test_metrics.json",
        "run_complete.json",
    )

    for city in resolved_cities:
        for test_year in resolved_years:
            for horizon in resolved_horizons:
                specification = architectures[int(horizon)]
                run_dir = selected_transfer_run_directory(
                    output_root=output,
                    city=city,
                    test_year=test_year,
                    architecture=specification,
                )
                completion = _read_json(run_dir / "run_complete.json")
                resolved_config = _read_json(run_dir / "resolved_config.json")
                validation_metrics = _read_json(
                    run_dir / "best_validation_metrics.json"
                )
                test_metrics = _read_json(run_dir / "best_test_metrics.json")

                architecture_match: bool | None = None
                architecture_error = ""
                if resolved_config is not None:
                    actual = _resolved_architecture_from_config(resolved_config)
                    expected = {
                        "large_kernel": int(specification.large_kernel),
                        "patch_stride": int(specification.patch_stride),
                        "d_model": int(specification.d_model),
                        "graph_hidden_dim": int(specification.graph_hidden_dim),
                    }
                    architecture_match = actual == expected
                    if not architecture_match:
                        architecture_error = (
                            f"expected={expected}; actual={actual}"
                        )

                missing_artifacts = [
                    relative
                    for relative in required_relative_paths
                    if not (run_dir / relative).is_file()
                ]
                completed = bool(
                    completion is not None
                    and completion.get("status") == "completed"
                )
                status = "completed" if completed else "incomplete"
                if architecture_match is False:
                    status = "configuration_mismatch"

                manifest = _read_json(run_dir / "data_manifest.json")
                test_windows = None
                if manifest is not None:
                    split_values = manifest.get("splits", {})
                    if isinstance(split_values, Mapping):
                        test_values = split_values.get("test", {})
                        if isinstance(test_values, Mapping):
                            test_windows = test_values.get("windows")

                rows.append(
                    {
                        "city": city,
                        "city_display_name": CITY_DISPLAY_NAMES[city],
                        "test_year": int(test_year),
                        "validation_year": int(test_year) - 1,
                        "training_end_year": int(test_year) - 2,
                        "horizon": int(horizon),
                        "context_length": int(specification.context_length),
                        "large_kernel": int(specification.large_kernel),
                        "patch_stride": int(specification.patch_stride),
                        "d_model": int(specification.d_model),
                        "graph_hidden_dim": int(specification.graph_hidden_dim),
                        "run_suffix": specification.run_suffix,
                        "status": status,
                        "architecture_match": architecture_match,
                        "architecture_error": architecture_error,
                        "best_epoch": (
                            completion.get("best_epoch")
                            if completion is not None
                            else None
                        ),
                        "best_validation_score": (
                            completion.get("best_validation_score")
                            if completion is not None
                            else None
                        ),
                        "validation_mae": _metric(validation_metrics, "mae"),
                        "validation_r": _metric(validation_metrics, "r"),
                        "validation_smape": _metric(
                            validation_metrics, "smape"
                        ),
                        "test_mae": _metric(test_metrics, "mae"),
                        "test_r": _metric(test_metrics, "r"),
                        "test_smape": _metric(test_metrics, "smape"),
                        "test_windows": test_windows,
                        "missing_artifact_count": len(missing_artifacts),
                        "missing_artifacts": "; ".join(missing_artifacts),
                        "run_directory": str(run_dir),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["city", "horizon", "test_year"]
    ).reset_index(drop=True)


def build_city_test_metric_table(
    metrics: pd.DataFrame,
    *,
    city: str,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
) -> pd.DataFrame:
    """Create the requested 4-row × 9-column location-specific test table."""

    canonical = str(city).lower().strip()
    years = _normalise_unique_ints(test_years, name="test year")
    resolved_horizons = _normalise_unique_ints(horizons, name="horizon")
    columns = pd.MultiIndex.from_product(
        [years, ("MAE", "r", "sMAPE")],
        names=("test_year", "metric"),
    )
    table = pd.DataFrame(
        np.nan,
        index=pd.Index(resolved_horizons, name="horizon"),
        columns=columns,
        dtype=float,
    )
    subset = metrics.loc[metrics["city"].eq(canonical)].copy()
    if subset.duplicated(["test_year", "horizon"]).any():
        duplicates = subset.loc[
            subset.duplicated(["test_year", "horizon"], keep=False),
            ["test_year", "horizon", "run_directory"],
        ]
        raise ValueError(
            "Multiple runs map to the same city/year/horizon table cell:\n"
            f"{duplicates.to_string(index=False)}"
        )
    for row in subset.itertuples(index=False):
        year = int(row.test_year)
        horizon = int(row.horizon)
        if year not in years or horizon not in resolved_horizons:
            continue
        table.loc[horizon, (year, "MAE")] = float(row.test_mae)
        table.loc[horizon, (year, "r")] = float(row.test_r)
        table.loc[horizon, (year, "sMAPE")] = float(row.test_smape)
    return table


def save_selected_transfer_summaries(
    *,
    metrics: pd.DataFrame,
    output_root: str | Path,
    summary_directory: str | Path | None = None,
    cities: Sequence[str] = FINAL_TRANSFER_CITIES,
    test_years: Sequence[int] = FINAL_TRANSFER_TEST_YEARS,
    horizons: Sequence[int] = FINAL_TRANSFER_HORIZONS,
) -> dict[str, Path]:
    """Save the long summary and one multi-header metric CSV per city."""

    output = Path(output_root).expanduser()
    summary_root = (
        Path(summary_directory).expanduser()
        if summary_directory is not None
        else output / "final_selected_modernTCN_transfer"
    )
    summary_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    long_path = summary_root / "metrics_long.csv"
    _atomic_csv(metrics, long_path)
    paths["metrics_long"] = long_path

    status_columns = [
        "city",
        "test_year",
        "horizon",
        "status",
        "architecture_match",
        "missing_artifact_count",
        "missing_artifacts",
        "run_directory",
    ]
    status_path = summary_root / "artifact_status.csv"
    _atomic_csv(metrics.loc[:, status_columns], status_path)
    paths["artifact_status"] = status_path

    nested: dict[str, Any] = {}
    for city in _normalise_unique_cities(cities):
        table = build_city_test_metric_table(
            metrics,
            city=city,
            test_years=test_years,
            horizons=horizons,
        )
        table_path = summary_root / f"test_metrics_{city}.csv"
        table.to_csv(table_path)
        paths[f"test_metrics_{city}"] = table_path
        nested[city] = {
            str(int(horizon)): {
                str(int(year)): {
                    metric: (
                        None
                        if pd.isna(table.loc[horizon, (year, label)])
                        else float(table.loc[horizon, (year, label)])
                    )
                    for metric, label in (
                        ("mae", "MAE"),
                        ("r", "r"),
                        ("smape", "sMAPE"),
                    )
                }
                for year in test_years
            }
            for horizon in horizons
        }

    json_path = summary_root / "test_metrics_by_city.json"
    _atomic_json(nested, json_path)
    paths["test_metrics_json"] = json_path
    return paths

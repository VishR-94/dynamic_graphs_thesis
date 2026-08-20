from __future__ import annotations

"""Exact Sonnet weather slicing/scaling plus graph-oriented tensor mapping.

The split boundaries, inclusive pandas slicing, input/target offsets and the
separate ``StandardScaler`` fits intentionally mirror Sonnet's executable
``CustomWeatherDataset``.  The resulting sample counts for Cape Town 2018 are
54,057 / 1,457 / 1,457 for every supported horizon.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset

from .config import (
    CENTRAL_NODE_INDEX,
    WEATHER_FEATURES,
    WEATHER_NODES,
    WeatherRunConfig,
)


SplitName = Literal["train", "validation", "test"]


def _timestamp_ns(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def weather_node_column_map(columns: list[str] | pd.Index) -> dict[str, dict[str, str]]:
    available = {str(value) for value in columns}
    result: dict[str, dict[str, str]] = {}
    for node in WEATHER_NODES:
        if node == "C":
            mapping = {
                "z500": "z500",
                "t850": "t850",
                "t2m": "t2m",
                "u10": "u10",
                "v10": "v10",
            }
        else:
            mapping = {
                "z500": f"z_{node}",
                "t850": f"t_{node}",
                "t2m": f"t2m_{node}",
                "u10": f"u10_{node}",
                "v10": f"v10_{node}",
            }
        missing = [value for value in mapping.values() if value not in available]
        if missing:
            raise KeyError(
                f"Weather CSV is missing columns for node {node}: {missing}."
            )
        result[node] = mapping
    return result


def _frame_to_node_feature_array(
    frame: pd.DataFrame,
    *,
    node_map: Mapping[str, Mapping[str, str]],
) -> np.ndarray:
    node_arrays: list[np.ndarray] = []
    for node in WEATHER_NODES:
        columns = [node_map[node][feature] for feature in WEATHER_FEATURES]
        node_arrays.append(frame.loc[:, columns].to_numpy(dtype=np.float32))
    return np.stack(node_arrays, axis=1).astype(np.float32, copy=False)


def _frame_to_t850_node_array(
    frame: pd.DataFrame,
    *,
    node_map: Mapping[str, Mapping[str, str]],
) -> np.ndarray:
    columns = [node_map[node]["t850"] for node in WEATHER_NODES]
    return frame.loc[:, columns].to_numpy(dtype=np.float32)


def _row_normalise_nonnegative(values: np.ndarray, *, eps: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Graph prior must be square.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("Graph prior must be finite and non-negative.")
    np.fill_diagonal(matrix, 0.0)
    row_mass = matrix.sum(axis=1, keepdims=True)
    empty = row_mass[:, 0] <= eps
    if np.any(empty):
        fallback = np.ones_like(matrix)
        np.fill_diagonal(fallback, 0.0)
        fallback /= fallback.sum(axis=1, keepdims=True)
        matrix[empty] = fallback[empty]
        row_mass = matrix.sum(axis=1, keepdims=True)
    return (matrix / np.maximum(row_mass, eps)).astype(np.float32)


@dataclass(frozen=True)
class SonnetAlignedSplit:
    name: SplitName
    input_positions: np.ndarray
    target_positions: np.ndarray
    context_length: int
    horizon: int

    def __post_init__(self) -> None:
        if self.input_positions.ndim != 1 or self.target_positions.ndim != 1:
            raise ValueError("Aligned split positions must be one-dimensional.")
        if len(self.input_positions) - int(self.context_length) != (
            len(self.target_positions) - int(self.horizon)
        ):
            raise ValueError("Input/target slices do not yield aligned window counts.")
        if self.sample_count <= 0:
            raise ValueError(f"Split {self.name} contains no windows.")

    @property
    def sample_count(self) -> int:
        return int(len(self.input_positions) - int(self.context_length) + 1)


class SonnetWeatherWindowDataset(Dataset[dict[str, Any]]):
    """Standard Sonnet outer windows mapped to ``[L, 9, 5]`` tensors."""

    def __init__(
        self,
        *,
        bundle: "SonnetWeatherDataBundle",
        split: SonnetAlignedSplit,
        dense_prefix: bool,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.split = split
        self.dense_prefix = bool(dense_prefix)
        self.context_length = int(split.context_length)
        self.horizon = int(split.horizon)
        self.num_nodes = len(WEATHER_NODES)

        # Relative targets for all causal internal origins.  For the final
        # origin, this is exactly the ordinary Sonnet H-step target sequence.
        origins = np.arange(self.context_length, dtype=np.int64)[:, None]
        future_steps = np.arange(1, self.horizon + 1, dtype=np.int64)[None, :]
        self._dense_relative_indices = origins + future_steps

        # Strong alignment check on first/middle/last windows.
        for item in sorted({0, len(self) // 2, len(self) - 1}):
            self._validate_alignment(item)

    def __len__(self) -> int:
        return self.split.sample_count

    def _global_starts(self, index: int) -> tuple[int, int]:
        input_start = int(self.split.input_positions[index])
        target_start = int(self.split.target_positions[index])
        return input_start, target_start

    def _validate_alignment(self, index: int) -> None:
        input_start, target_start = self._global_starts(index)
        expected_target_start = input_start + self.context_length
        if target_start != expected_target_start:
            raise RuntimeError(
                f"Split {self.split.name} window {index} is misaligned: "
                f"target starts at {target_start}, expected {expected_target_start}."
            )
        final_dense = input_start + self._dense_relative_indices[-1]
        ordinary = self.split.target_positions[index : index + self.horizon]
        if not np.array_equal(final_dense, ordinary):
            raise RuntimeError("Dense final-origin targets differ from Sonnet targets.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        input_start, target_start = self._global_starts(index)
        input_end = input_start + self.context_length
        target_end = target_start + self.horizon

        x = torch.from_numpy(
            self.bundle.input_nodes_scaled[input_start:input_end]
        ).float()
        y = torch.from_numpy(
            self.bundle.target_t850_nodes_scaled[target_start:target_end]
        ).float().unsqueeze(-1)
        y_raw = torch.from_numpy(
            self.bundle.target_t850_nodes_raw[target_start:target_end]
        ).float().unsqueeze(-1)
        last_context_raw = torch.from_numpy(
            self.bundle.target_t850_nodes_raw[input_end - 1]
        ).float().unsqueeze(-1)

        target_positions = self.split.target_positions[index : index + self.horizon]
        target_times_ns = torch.from_numpy(
            self.bundle.timestamps_ns[target_positions].copy()
        ).long()

        result: dict[str, Any] = {
            "x": x.contiguous(),
            "y": y.contiguous(),
            "y_unnormalised": y_raw.contiguous(),
            "last_context_target": last_context_raw.contiguous(),
            "sample_idx": torch.tensor(index, dtype=torch.long),
            "input_start_index": torch.tensor(input_start, dtype=torch.long),
            "forecast_origin_index": torch.tensor(input_end - 1, dtype=torch.long),
            "target_indices": torch.from_numpy(
                self.split.target_positions[index : index + self.horizon].copy()
            ).long(),
            "context_start_time_ns": torch.tensor(
                int(self.bundle.timestamps_ns[input_start]), dtype=torch.long
            ),
            "forecast_origin_time_ns": torch.tensor(
                int(self.bundle.timestamps_ns[input_end - 1]), dtype=torch.long
            ),
            "target_times_ns": target_times_ns,
            # The imported ModernTCN model requires these arguments even when
            # session-position encoding is disabled.  The values satisfy its
            # public shape/range contract without introducing a day boundary.
            "context_start": torch.tensor(0, dtype=torch.long),
            "session_length": torch.tensor(self.context_length, dtype=torch.long),
        }

        if self.dense_prefix:
            dense_positions = input_start + self._dense_relative_indices
            dense_y = torch.from_numpy(
                self.bundle.target_t850_nodes_scaled[dense_positions]
            ).float().unsqueeze(-1)
            result["dense_y"] = dense_y.contiguous()
            result["dense_target_indices"] = torch.from_numpy(
                dense_positions.copy()
            ).long()
        return result


@dataclass
class SonnetWeatherDataBundle:
    config: WeatherRunConfig
    raw_frame: pd.DataFrame
    reordered_columns: tuple[str, ...]
    node_map: dict[str, dict[str, str]]
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    input_nodes_scaled: np.ndarray
    target_t850_nodes_scaled: np.ndarray
    target_t850_nodes_raw: np.ndarray
    timestamps_ns: np.ndarray
    splits: dict[SplitName, SonnetAlignedSplit]
    source_abs_correlation: np.ndarray
    row_normalised_correlation_prior: np.ndarray
    input_fit_positions: np.ndarray
    target_fit_positions: np.ndarray

    def dataset(
        self,
        split: SplitName,
        *,
        dense_prefix: bool = False,
    ) -> SonnetWeatherWindowDataset:
        if split not in self.splits:
            raise KeyError(f"Unknown split {split!r}.")
        if dense_prefix and split != "train":
            raise ValueError("Dense-prefix supervision is a training-only dataset view.")
        return SonnetWeatherWindowDataset(
            bundle=self,
            split=self.splits[split],
            dense_prefix=dense_prefix,
        )

    @property
    def target_column_indices(self) -> np.ndarray:
        columns = list(self.reordered_columns)
        return np.asarray(
            [columns.index(self.node_map[node]["t850"]) for node in WEATHER_NODES],
            dtype=np.int64,
        )

    @property
    def target_mean(self) -> np.ndarray:
        return np.asarray(self.target_scaler.mean_, dtype=np.float64)[
            self.target_column_indices
        ].astype(np.float32)

    @property
    def target_scale(self) -> np.ndarray:
        return np.asarray(self.target_scaler.scale_, dtype=np.float64)[
            self.target_column_indices
        ].astype(np.float32)

    def inverse_target_tensor(self, values: Tensor) -> Tensor:
        tensor = torch.as_tensor(values)
        mean = torch.as_tensor(
            self.target_mean,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        scale = torch.as_tensor(
            self.target_scale,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        while mean.ndim < tensor.ndim - 1:
            mean = mean.unsqueeze(0)
            scale = scale.unsqueeze(0)
        # values end in [N, 1]; insert the singleton target-channel axis.
        mean = mean.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
        return tensor * scale + mean

    def manifest(self) -> dict[str, Any]:
        split_values: dict[str, Any] = {}
        for name, split in self.splits.items():
            first_input = int(split.input_positions[0])
            last_input = int(split.input_positions[-1])
            first_target = int(split.target_positions[0])
            last_target = int(split.target_positions[-1])
            first_final_target = int(split.target_positions[split.horizon - 1])
            split_values[name] = {
                "windows": split.sample_count,
                "input_rows": int(len(split.input_positions)),
                "target_rows": int(len(split.target_positions)),
                "input_first": str(self.raw_frame.index[first_input]),
                "input_last": str(self.raw_frame.index[last_input]),
                "target_first": str(self.raw_frame.index[first_target]),
                "target_last": str(self.raw_frame.index[last_target]),
                "first_final_target": str(self.raw_frame.index[first_final_target]),
                "last_final_target": str(self.raw_frame.index[last_target]),
            }
        return {
            "protocol": "Sonnet executable weatherDataloader.py",
            "city": self.config.city,
            "test_year": int(self.config.test_year),
            "validation_year": int(self.config.validation_year),
            "training_start_year": int(self.config.start_year),
            "training_end_year": int(self.config.training_end_year),
            "context_length": int(self.config.context_length),
            "forecast_length": int(self.config.horizon),
            "stride": 1,
            "sampling_frequency_hours": 6,
            "calendar_day_boundaries_enforced": False,
            "historical_context_may_cross_year_boundary": True,
            "source_csv": str(self.config.data_path.resolve()),
            "source_csv_sha256": _sha256(self.config.data_path.resolve()),
            "raw_rows": int(self.raw_frame.shape[0]),
            "raw_columns": int(self.raw_frame.shape[1]),
            "raw_first_timestamp": str(self.raw_frame.index[0]),
            "raw_last_timestamp": str(self.raw_frame.index[-1]),
            "reordered_columns": list(self.reordered_columns),
            "node_order": list(WEATHER_NODES),
            "feature_order": list(WEATHER_FEATURES),
            "central_node_index": CENTRAL_NODE_INDEX,
            "node_column_map": self.node_map,
            "normalisation": {
                "input": "separate StandardScaler fitted on exact Sonnet training input slice",
                "target": "separate StandardScaler fitted on exact Sonnet MT training target slice",
                "window_normalisation": False,
                "input_fit_first": str(
                    self.raw_frame.index[int(self.input_fit_positions[0])]
                ),
                "input_fit_last": str(
                    self.raw_frame.index[int(self.input_fit_positions[-1])]
                ),
                "input_fit_rows": int(len(self.input_fit_positions)),
                "target_fit_first": str(
                    self.raw_frame.index[int(self.target_fit_positions[0])]
                ),
                "target_fit_last": str(
                    self.raw_frame.index[int(self.target_fit_positions[-1])]
                ),
                "target_fit_rows": int(len(self.target_fit_positions)),
            },
            "correlation_prior": {
                "source": "absolute Pearson correlation of raw six-hour T850 first differences",
                "nominal_training_period_only": True,
                "diagonal_removed": True,
                "row_normalised": True,
            },
            "splits": split_values,
        }

    def save_data_artifacts(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(self.manifest(), handle, indent=2, sort_keys=True)
        with (run_dir / "column_and_node_map.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "node_order": list(WEATHER_NODES),
                    "feature_order": list(WEATHER_FEATURES),
                    "central_node_index": CENTRAL_NODE_INDEX,
                    "node_column_map": self.node_map,
                    "reordered_columns": list(self.reordered_columns),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        np.savez_compressed(
            run_dir / "scalers.npz",
            input_mean=np.asarray(self.input_scaler.mean_, dtype=np.float64),
            input_scale=np.asarray(self.input_scaler.scale_, dtype=np.float64),
            input_var=np.asarray(self.input_scaler.var_, dtype=np.float64),
            target_mean=np.asarray(self.target_scaler.mean_, dtype=np.float64),
            target_scale=np.asarray(self.target_scaler.scale_, dtype=np.float64),
            target_var=np.asarray(self.target_scaler.var_, dtype=np.float64),
            target_t850_mean=self.target_mean,
            target_t850_scale=self.target_scale,
            reordered_columns=np.asarray(self.reordered_columns, dtype=object),
            node_order=np.asarray(WEATHER_NODES, dtype=object),
            feature_order=np.asarray(WEATHER_FEATURES, dtype=object),
        )
        joblib.dump(
            {
                "input_scaler": self.input_scaler,
                "target_scaler": self.target_scaler,
                "reordered_columns": self.reordered_columns,
                "node_map": self.node_map,
            },
            run_dir / "scalers.joblib",
        )
        torch.save(
            {
                "graph_orientation": "A[target, source]",
                "node_order": list(WEATHER_NODES),
                "source_abs_correlation": torch.from_numpy(
                    self.source_abs_correlation
                ).float(),
                "row_normalised_prior": torch.from_numpy(
                    self.row_normalised_correlation_prior
                ).float(),
                "construction": "absolute correlation of raw T850 first differences over nominal training period",
            },
            run_dir / "static_correlation_prior.pt",
        )


def _positions_for_slice(
    full_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> np.ndarray:
    sliced = pd.Series(np.arange(len(full_index), dtype=np.int64), index=full_index).loc[
        start:end
    ]
    if sliced.empty:
        raise ValueError(f"Date slice {start} to {end} is empty.")
    return sliced.to_numpy(dtype=np.int64)


def _split_positions(
    index: pd.DatetimeIndex,
    *,
    config: WeatherRunConfig,
    split_name: SplitName,
) -> tuple[np.ndarray, np.ndarray]:
    one_step = pd.Timedelta(hours=6)
    context_delta = pd.Timedelta(hours=6 * int(config.context_length))
    forecast_delta = pd.Timedelta(hours=6 * int(config.horizon))

    if split_name == "train":
        start = pd.Timestamp(f"{int(config.start_year)}-01-01")
        end = pd.Timestamp(f"{int(config.training_end_year)}-12-31")
    elif split_name == "validation":
        start = pd.Timestamp(f"{int(config.validation_year)}-01-01")
        end = pd.Timestamp(f"{int(config.validation_year)}-12-31")
    elif split_name == "test":
        start = pd.Timestamp(f"{int(config.test_year)}-01-01")
        end = pd.Timestamp(f"{int(config.test_year)}-12-31")
    else:
        raise KeyError(split_name)

    input_positions = _positions_for_slice(
        index,
        start - context_delta - forecast_delta + one_step,
        end - forecast_delta,
    )
    target_positions = _positions_for_slice(
        index,
        start - forecast_delta + one_step,
        end,
    )
    return input_positions, target_positions


def build_weather_data_bundle(config: WeatherRunConfig) -> SonnetWeatherDataBundle:
    path = config.data_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Weather CSV not found: {path}")
    if config.city not in path.name.lower():
        raise ValueError(
            "Weather CSV filename does not contain the configured city name: "
            f"city={config.city!r}, path={path.name!r}."
        )

    raw = pd.read_csv(path, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    if not raw.index.is_unique:
        raise ValueError("Weather timestamps are not unique.")
    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        raise ValueError(f"Weather CSV contains duplicate columns: {duplicates}")
    if raw.isna().any().any():
        raise ValueError("Weather CSV contains missing values.")
    differences = raw.index.to_series().diff().dropna()
    if not (differences == pd.Timedelta(hours=6)).all():
        raise ValueError("Weather CSV is not a regular six-hourly time series.")

    node_map = weather_node_column_map(raw.columns)

    # Sonnet moves the central target column to the final flat-column position
    # before creating and scaling both input and MT target frames.
    target_column = "t850"
    reordered_columns = tuple(
        [str(value) for value in raw.columns if str(value) != target_column]
        + [target_column]
    )
    reordered = raw.loc[:, list(reordered_columns)].copy()

    splits: dict[SplitName, SonnetAlignedSplit] = {}
    for split_name in ("train", "validation", "test"):
        input_positions, target_positions = _split_positions(
            raw.index,
            config=config,
            split_name=split_name,
        )
        splits[split_name] = SonnetAlignedSplit(
            name=split_name,
            input_positions=input_positions,
            target_positions=target_positions,
            context_length=config.context_length,
            horizon=config.horizon,
        )

    input_fit_positions = splits["train"].input_positions
    target_fit_positions = splits["train"].target_positions
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    input_scaler.fit(reordered.iloc[input_fit_positions])
    target_scaler.fit(reordered.iloc[target_fit_positions])

    input_scaled_frame = pd.DataFrame(
        input_scaler.transform(reordered),
        index=reordered.index,
        columns=reordered.columns,
    )
    target_scaled_frame = pd.DataFrame(
        target_scaler.transform(reordered),
        index=reordered.index,
        columns=reordered.columns,
    )

    input_nodes_scaled = _frame_to_node_feature_array(
        input_scaled_frame,
        node_map=node_map,
    )
    target_t850_nodes_scaled = _frame_to_t850_node_array(
        target_scaled_frame,
        node_map=node_map,
    )
    target_t850_nodes_raw = _frame_to_t850_node_array(raw, node_map=node_map)

    # One horizon-independent, training-only prior per city/test year.
    nominal_start = pd.Timestamp(f"{int(config.start_year)}-01-01")
    nominal_end = pd.Timestamp(f"{int(config.training_end_year)}-12-31")
    nominal_positions = _positions_for_slice(raw.index, nominal_start, nominal_end)
    nominal_t850 = target_t850_nodes_raw[nominal_positions].astype(np.float64)
    first_differences = np.diff(nominal_t850, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(first_differences, rowvar=False)
    source_abs_correlation = np.nan_to_num(
        np.abs(correlation), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    np.fill_diagonal(source_abs_correlation, 0.0)
    prior = _row_normalise_nonnegative(source_abs_correlation)

    bundle = SonnetWeatherDataBundle(
        config=config,
        raw_frame=raw,
        reordered_columns=reordered_columns,
        node_map=node_map,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        input_nodes_scaled=input_nodes_scaled,
        target_t850_nodes_scaled=target_t850_nodes_scaled,
        target_t850_nodes_raw=target_t850_nodes_raw,
        timestamps_ns=_timestamp_ns(raw.index),
        splits=splits,
        source_abs_correlation=source_abs_correlation,
        row_normalised_correlation_prior=prior,
        input_fit_positions=input_fit_positions,
        target_fit_positions=target_fit_positions,
    )

    # Cape Town 2018 should reproduce the current Sonnet loader exactly.
    if config.city == "capetown" and int(config.test_year) == 2018:
        expected = {"train": 54057, "validation": 1457, "test": 1457}
        observed = {name: split.sample_count for name, split in splits.items()}
        if observed != expected:
            raise RuntimeError(
                f"Cape Town 2018 Sonnet window counts differ: {observed} != {expected}."
            )
    return bundle

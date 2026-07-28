from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.token_graph_dataset import (
    load_origin_aligned_token_cache,
    validate_origin_aligned_token_cache,
)


TokenGraphDataMode = Literal["auto", "real", "synthetic"]

REAL_TOKEN_GRAPH_REPRESENTATION = (
    "origin_aligned_kronos_forecasting_tokens"
)
SYNTHETIC_TOKEN_GRAPH_REPRESENTATION = (
    "kronos_basedygraph_window_tokens"
)
SYNTHETIC_TOKEN_GRAPH_CACHE_VERSION = 1
TOKEN_VOCABULARY_SIZE = 1024
GRAPH_ORIENTATION = "row=target,column=source"

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_REAL_WINDOW_FIELDS = (
    "context_mean",
    "context_std",
    "evaluation_true",
    "last_context_target",
    "sample_idx",
    "origin_idx",
    "target_indices",
)

_SYNTHETIC_WINDOW_FIELDS = (
    "true_graph",
    "regime_id",
    "trajectory_id",
    "origin_idx",
    "target_indices",
)


@dataclass(frozen=True)
class TokenGraphDataLoaders:
    """Production train/validation token datasets and DataLoaders."""

    train_dataset: "CachedTokenGraphDataset"
    validation_dataset: "CachedTokenGraphDataset"
    train_loader: DataLoader[dict[str, Any]]
    validation_loader: DataLoader[dict[str, Any]]


def _torch_load_mapping(
    path: str | Path,
) -> dict[str, Any]:
    resolved = Path(
        path
    ).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Token cache does not exist: {resolved}"
        )

    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(
            resolved,
            map_location="cpu",
        )

    if not isinstance(
        loaded,
        Mapping,
    ):
        raise TypeError(
            "Saved token cache must be a mapping."
        )

    return dict(
        loaded
    )


def _validate_integer_tensor(
    values: Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    minimum: int = 0,
    maximum_exclusive: int | None = None,
) -> None:
    if values.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"{name} must use an integer dtype, "
            f"got {values.dtype}."
        )

    if (
        expected_shape is not None
        and tuple(values.shape) != expected_shape
    ):
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"got {tuple(values.shape)}."
        )

    if values.numel() == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    observed_minimum = int(
        values.min().item()
    )
    observed_maximum = int(
        values.max().item()
    )

    if observed_minimum < minimum:
        raise ValueError(
            f"{name} contains a value below {minimum}."
        )

    if (
        maximum_exclusive is not None
        and observed_maximum >= maximum_exclusive
    ):
        raise ValueError(
            f"{name} contains a value outside "
            f"[0, {maximum_exclusive - 1}]."
        )


def _validate_graph_tensor(
    values: Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    atol: float = 1.0e-5,
) -> Tensor:
    graph = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if tuple(graph.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"got {tuple(graph.shape)}."
        )

    if not torch.isfinite(
        graph
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if torch.any(
        graph < -atol
    ):
        raise ValueError(
            f"{name} contains negative edge weights."
        )

    diagonal = torch.diagonal(
        graph,
        dim1=-2,
        dim2=-1,
    )

    if not torch.allclose(
        diagonal,
        torch.zeros_like(
            diagonal
        ),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(
            f"{name} must have a zero diagonal."
        )

    row_sums = graph.sum(
        dim=-1
    )

    if not torch.allclose(
        row_sums,
        torch.ones_like(
            row_sums
        ),
        atol=atol,
        rtol=atol,
    ):
        raise ValueError(
            f"{name} must be row-stochastic."
        )

    return graph.contiguous()


def _validate_asset_cols(
    values: Any,
    *,
    num_assets: int,
) -> tuple[str, ...]:
    asset_cols = tuple(
        str(value)
        for value in values
    )

    if len(asset_cols) != num_assets:
        raise ValueError(
            "asset_cols length does not match "
            "the asset dimension."
        )

    if len(set(asset_cols)) != len(
        asset_cols
    ):
        raise ValueError(
            "asset_cols must be unique."
        )

    return asset_cols


def validate_synthetic_token_graph_cache(
    cache: Mapping[str, Any],
) -> None:
    """Validate the fixed, pre-windowed synthetic training cache.

    The simulator may also save complete trajectories for provenance.
    The common training runner consumes this window cache:

        context_tokens:
            [W, C, N, 2]

        target_s1 / target_s2:
            [W, P, N]

        true_graph:
            [W, N, N]

        regime_id / trajectory_id / origin_idx:
            [W]

        regime_graphs:
            [R, N, N]
    """
    required = {
        "format_version",
        "representation",
        "context_tokens",
        "target_s1",
        "target_s2",
        "true_graph",
        "regime_id",
        "trajectory_id",
        "origin_idx",
        "regime_graphs",
        "asset_cols",
        "context_length",
        "prediction_length",
        "dense_horizons",
        "graph_orientation",
    }

    missing = required - set(
        cache
    )

    if missing:
        raise KeyError(
            "Synthetic token cache is missing keys: "
            f"{sorted(missing)}."
        )

    if str(
        cache["representation"]
    ) != SYNTHETIC_TOKEN_GRAPH_REPRESENTATION:
        raise ValueError(
            "Unexpected synthetic cache representation."
        )

    if int(
        cache["format_version"]
    ) != SYNTHETIC_TOKEN_GRAPH_CACHE_VERSION:
        raise ValueError(
            "Unsupported synthetic cache version."
        )

    if str(
        cache["graph_orientation"]
    ) != GRAPH_ORIENTATION:
        raise ValueError(
            "graph_orientation must be "
            f"{GRAPH_ORIENTATION!r}."
        )

    context_tokens = torch.as_tensor(
        cache["context_tokens"]
    )
    target_s1 = torch.as_tensor(
        cache["target_s1"]
    )
    target_s2 = torch.as_tensor(
        cache["target_s2"]
    )

    if (
        context_tokens.ndim != 4
        or context_tokens.shape[-1] != 2
    ):
        raise ValueError(
            "context_tokens must have shape "
            "[W, C, N, 2]."
        )

    (
        num_windows,
        context_length,
        num_assets,
        _,
    ) = context_tokens.shape

    prediction_length = int(
        cache["prediction_length"]
    )

    if num_windows <= 0:
        raise ValueError(
            "Synthetic cache contains no windows."
        )

    if int(
        cache["context_length"]
    ) != context_length:
        raise ValueError(
            "context_length metadata does not "
            "match context_tokens."
        )

    if prediction_length <= 0:
        raise ValueError(
            "prediction_length must be positive."
        )

    target_shape = (
        num_windows,
        prediction_length,
        num_assets,
    )

    _validate_integer_tensor(
        context_tokens,
        name="context_tokens",
        expected_shape=(
            num_windows,
            context_length,
            num_assets,
            2,
        ),
        maximum_exclusive=(
            TOKEN_VOCABULARY_SIZE
        ),
    )

    _validate_integer_tensor(
        target_s1,
        name="target_s1",
        expected_shape=target_shape,
        maximum_exclusive=(
            TOKEN_VOCABULARY_SIZE
        ),
    )

    _validate_integer_tensor(
        target_s2,
        name="target_s2",
        expected_shape=target_shape,
        maximum_exclusive=(
            TOKEN_VOCABULARY_SIZE
        ),
    )

    _validate_asset_cols(
        cache["asset_cols"],
        num_assets=num_assets,
    )

    dense_horizons = tuple(
        int(value)
        for value in cache[
            "dense_horizons"
        ]
    )

    if dense_horizons != tuple(
        range(
            1,
            prediction_length + 1,
        )
    ):
        raise ValueError(
            "dense_horizons must be exactly "
            "1..prediction_length."
        )

    regime_graphs = torch.as_tensor(
        cache["regime_graphs"]
    )

    if (
        regime_graphs.ndim != 3
        or regime_graphs.shape[0] <= 0
    ):
        raise ValueError(
            "regime_graphs must have shape "
            "[R, N, N]."
        )

    num_regimes = int(
        regime_graphs.shape[0]
    )

    regime_graphs = _validate_graph_tensor(
        regime_graphs,
        name="regime_graphs",
        expected_shape=(
            num_regimes,
            num_assets,
            num_assets,
        ),
    )

    true_graph = _validate_graph_tensor(
        torch.as_tensor(
            cache["true_graph"]
        ),
        name="true_graph",
        expected_shape=(
            num_windows,
            num_assets,
            num_assets,
        ),
    )

    regime_id = torch.as_tensor(
        cache["regime_id"]
    )
    trajectory_id = torch.as_tensor(
        cache["trajectory_id"]
    )
    origin_idx = torch.as_tensor(
        cache["origin_idx"]
    )

    _validate_integer_tensor(
        regime_id,
        name="regime_id",
        expected_shape=(
            num_windows,
        ),
        maximum_exclusive=num_regimes,
    )

    _validate_integer_tensor(
        trajectory_id,
        name="trajectory_id",
        expected_shape=(
            num_windows,
        ),
    )

    _validate_integer_tensor(
        origin_idx,
        name="origin_idx",
        expected_shape=(
            num_windows,
        ),
    )

    expected_graph = regime_graphs[
        regime_id.to(
            torch.long
        )
    ]

    if not torch.allclose(
        true_graph,
        expected_graph,
        atol=1.0e-6,
        rtol=1.0e-5,
    ):
        raise ValueError(
            "true_graph does not match "
            "regime_graphs[regime_id]."
        )

    if "target_indices" in cache:
        target_indices = torch.as_tensor(
            cache["target_indices"]
        )

        _validate_integer_tensor(
            target_indices,
            name="target_indices",
            expected_shape=(
                num_windows,
                prediction_length,
            ),
        )

        expected_indices = (
            origin_idx
            .to(torch.long)
            .unsqueeze(1)
            + torch.arange(
                1,
                prediction_length + 1,
                dtype=torch.long,
            ).unsqueeze(0)
        )

        if not torch.equal(
            target_indices.to(
                torch.long
            ),
            expected_indices,
        ):
            raise ValueError(
                "target_indices must equal "
                "origin_idx + 1..P."
            )


def validate_token_graph_cache(
    cache: Mapping[str, Any],
) -> None:
    representation = str(
        cache.get(
            "representation",
            "",
        )
    )

    if representation == (
        REAL_TOKEN_GRAPH_REPRESENTATION
    ):
        validate_origin_aligned_token_cache(
            cache
        )
        return

    if representation == (
        SYNTHETIC_TOKEN_GRAPH_REPRESENTATION
    ):
        validate_synthetic_token_graph_cache(
            cache
        )
        return

    raise ValueError(
        "Unknown token-cache representation. "
        f"Expected {REAL_TOKEN_GRAPH_REPRESENTATION!r} "
        f"or {SYNTHETIC_TOKEN_GRAPH_REPRESENTATION!r}; "
        f"received {representation!r}."
    )


def load_token_graph_cache(
    path: str | Path,
    *,
    data_mode: TokenGraphDataMode = "auto",
) -> dict[str, Any]:
    """Load a real cache through the existing loader or a synthetic cache."""
    if data_mode not in {
        "auto",
        "real",
        "synthetic",
    }:
        raise ValueError(
            "data_mode must be 'auto', "
            "'real', or 'synthetic'."
        )

    if data_mode == "real":
        return load_origin_aligned_token_cache(
            path
        )

    cache = _torch_load_mapping(
        path
    )

    representation = str(
        cache.get(
            "representation",
            "",
        )
    )

    if (
        data_mode == "synthetic"
        and representation
        != SYNTHETIC_TOKEN_GRAPH_REPRESENTATION
    ):
        raise ValueError(
            "Requested a synthetic cache but found "
            f"representation {representation!r}."
        )

    validate_token_graph_cache(
        cache
    )

    return cache


class CachedTokenGraphDataset(
    Dataset[dict[str, Any]]
):
    """Index already-materialised real or synthetic token windows.

    This class does not recreate raw windows, normalise candles, or
    call Kronos. ``WindowedCandleDataset`` and
    ``token_graph_dataset.py`` remain the sole raw-window and
    cache-generation layers.
    """

    def __init__(
        self,
        cache: Mapping[str, Any],
        *,
        source_path: str | Path | None = None,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(
            cache,
            Mapping,
        ):
            raise TypeError(
                "cache must be a mapping."
            )

        self.cache = dict(
            cache
        )

        self.source_path = (
            None
            if source_path is None
            else Path(
                source_path
            ).expanduser().resolve()
        )

        if validate:
            validate_token_graph_cache(
                self.cache
            )

        self.representation = str(
            self.cache[
                "representation"
            ]
        )

        self.data_mode: Literal[
            "real",
            "synthetic",
        ] = (
            "real"
            if self.representation
            == REAL_TOKEN_GRAPH_REPRESENTATION
            else "synthetic"
        )

        self.context_tokens = torch.as_tensor(
            self.cache[
                "context_tokens"
            ]
        ).contiguous()

        self.target_s1 = torch.as_tensor(
            self.cache[
                "target_s1"
            ]
        ).contiguous()

        self.target_s2 = torch.as_tensor(
            self.cache[
                "target_s2"
            ]
        ).contiguous()

        self._num_windows = int(
            self.context_tokens.shape[0]
        )

        self._window_tensors: dict[
            str,
            Tensor,
        ] = {}

        field_names = (
            _REAL_WINDOW_FIELDS
            if self.data_mode == "real"
            else _SYNTHETIC_WINDOW_FIELDS
        )

        for key in field_names:
            if key not in self.cache:
                continue

            values = torch.as_tensor(
                self.cache[key]
            )

            if (
                values.ndim == 0
                or values.shape[0]
                != self._num_windows
            ):
                raise ValueError(
                    f"Per-window field {key!r} "
                    "must have W at axis 0."
                )

            self._window_tensors[
                key
            ] = values.contiguous()

        if (
            self.data_mode == "synthetic"
            and "target_indices"
            not in self._window_tensors
        ):
            origin_idx = self._window_tensors[
                "origin_idx"
            ].to(
                torch.long
            )

            self._window_tensors[
                "target_indices"
            ] = (
                origin_idx.unsqueeze(1)
                + torch.arange(
                    1,
                    self.prediction_length + 1,
                    dtype=torch.long,
                ).unsqueeze(0)
            )

        dates = self.cache.get(
            "dates"
        )

        if dates is None:
            self._dates: (
                tuple[str, ...]
                | None
            ) = None
        else:
            if len(dates) != self._num_windows:
                raise ValueError(
                    "dates length does not match "
                    "the window dimension."
                )

            self._dates = tuple(
                str(value)
                for value in dates
            )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        data_mode: TokenGraphDataMode = "auto",
    ) -> "CachedTokenGraphDataset":
        resolved = Path(
            path
        ).expanduser().resolve()

        cache = load_token_graph_cache(
            resolved,
            data_mode=data_mode,
        )

        return cls(
            cache,
            source_path=resolved,
            validate=False,
        )

    def __len__(
        self,
    ) -> int:
        return self._num_windows

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        if not 0 <= index < self._num_windows:
            raise IndexError(
                f"Window index {index} is outside "
                f"[0, {self._num_windows - 1}]."
            )

        item: dict[str, Any] = {
            "window_idx": torch.tensor(
                index,
                dtype=torch.long,
            ),
            "context_tokens": (
                self.context_tokens[
                    index
                ].to(torch.long)
            ),
            "target_s1": (
                self.target_s1[
                    index
                ].to(torch.long)
            ),
            "target_s2": (
                self.target_s2[
                    index
                ].to(torch.long)
            ),
        }

        for (
            key,
            values,
        ) in self._window_tensors.items():
            selected = values[
                index
            ]

            if key in {
                "sample_idx",
                "origin_idx",
                "target_indices",
                "regime_id",
                "trajectory_id",
            }:
                selected = selected.to(
                    torch.long
                )
            else:
                selected = selected.to(
                    torch.float32
                )

            item[
                key
            ] = selected

        if self._dates is not None:
            item[
                "date"
            ] = self._dates[
                index
            ]

        return item

    @property
    def num_windows(
        self,
    ) -> int:
        return self._num_windows

    @property
    def context_length(
        self,
    ) -> int:
        return int(
            self.context_tokens.shape[
                1
            ]
        )

    @property
    def prediction_length(
        self,
    ) -> int:
        return int(
            self.target_s1.shape[
                1
            ]
        )

    @property
    def num_assets(
        self,
    ) -> int:
        return int(
            self.context_tokens.shape[
                2
            ]
        )

    @property
    def asset_cols(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            str(value)
            for value in self.cache[
                "asset_cols"
            ]
        )

    @property
    def evaluation_horizons(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.cache.get(
                "evaluation_horizons",
                (),
            )
        )

    @property
    def evaluation_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.cache.get(
                "evaluation_indices",
                (),
            )
        )

    @property
    def has_raw_evaluation_targets(
        self,
    ) -> bool:
        return (
            self.data_mode == "real"
            and "evaluation_true"
            in self._window_tensors
            and "last_context_target"
            in self._window_tensors
            and bool(
                self.evaluation_horizons
            )
        )

    @property
    def has_true_graph(
        self,
    ) -> bool:
        return (
            self.data_mode
            == "synthetic"
        )

    @property
    def trajectory_ids(
        self,
    ) -> Tensor | None:
        if self.data_mode != "synthetic":
            return None

        return torch.unique(
            self._window_tensors[
                "trajectory_id"
            ].to(torch.long)
        )

    def static_metadata(
        self,
    ) -> dict[str, Any]:
        keys = (
            "format_version",
            "representation",
            "asset_cols",
            "input_channels",
            "target_channels",
            "tokenizer_channels",
            "context_length",
            "prediction_length",
            "dense_horizons",
            "evaluation_horizons",
            "evaluation_indices",
            "stride",
            "amount_mode",
            "normalisation",
            "tokenizer_id",
            "tokenizer_revision",
            "graph_orientation",
            "regime_names",
            "generator_config",
        )

        metadata = {
            key: self.cache[
                key
            ]
            for key in keys
            if key in self.cache
        }

        metadata.update(
            {
                "data_mode": self.data_mode,
                "num_windows": (
                    self.num_windows
                ),
                "num_assets": (
                    self.num_assets
                ),
            }
        )

        if self.source_path is not None:
            metadata[
                "source_path"
            ] = str(
                self.source_path
            )

        return metadata


def _seed_worker(
    worker_id: int,
) -> None:
    del worker_id

    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(
        worker_seed
    )
    np.random.seed(
        worker_seed
    )


def build_token_graph_dataloader(
    dataset: CachedTokenGraphDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, Any]]:
    """Build one deterministic DataLoader over saved token windows."""
    if not isinstance(
        dataset,
        CachedTokenGraphDataset,
    ):
        raise TypeError(
            "dataset must be a "
            "CachedTokenGraphDataset."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if prefetch_factor <= 0:
        raise ValueError(
            "prefetch_factor must be positive."
        )

    if pin_memory is None:
        pin_memory = (
            torch.cuda.is_available()
        )

    if persistent_workers is None:
        persistent_workers = (
            num_workers > 0
        )

    if (
        persistent_workers
        and num_workers == 0
    ):
        raise ValueError(
            "persistent_workers requires "
            "num_workers > 0."
        )

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )

    loader_kwargs: dict[
        str,
        Any,
    ] = {
        "dataset": dataset,
        "batch_size": int(
            batch_size
        ),
        "shuffle": bool(
            shuffle
        ),
        "num_workers": int(
            num_workers
        ),
        "drop_last": bool(
            drop_last
        ),
        "pin_memory": bool(
            pin_memory
        ),
        "persistent_workers": bool(
            persistent_workers
        ),
        "worker_init_fn": (
            _seed_worker
            if num_workers > 0
            else None
        ),
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs[
            "prefetch_factor"
        ] = int(
            prefetch_factor
        )

    return DataLoader(
        **loader_kwargs
    )


def _validate_train_validation_compatibility(
    train_dataset: CachedTokenGraphDataset,
    validation_dataset: CachedTokenGraphDataset,
) -> None:
    comparisons = {
        "data_mode": (
            train_dataset.data_mode,
            validation_dataset.data_mode,
        ),
        "context_length": (
            train_dataset.context_length,
            validation_dataset.context_length,
        ),
        "prediction_length": (
            train_dataset.prediction_length,
            validation_dataset.prediction_length,
        ),
        "num_assets": (
            train_dataset.num_assets,
            validation_dataset.num_assets,
        ),
        "asset_cols": (
            train_dataset.asset_cols,
            validation_dataset.asset_cols,
        ),
        "evaluation_horizons": (
            train_dataset.evaluation_horizons,
            validation_dataset.evaluation_horizons,
        ),
        "evaluation_indices": (
            train_dataset.evaluation_indices,
            validation_dataset.evaluation_indices,
        ),
    }

    for (
        name,
        (
            train_value,
            validation_value,
        ),
    ) in comparisons.items():
        if train_value != validation_value:
            raise ValueError(
                "Training and validation caches "
                f"disagree for {name}: "
                f"{train_value!r} versus "
                f"{validation_value!r}."
            )

    if train_dataset.data_mode == "real":
        for key in (
            "input_channels",
            "target_channels",
            "tokenizer_channels",
            "tokenizer_id",
            "tokenizer_revision",
            "amount_mode",
            "normalisation",
        ):
            if train_dataset.cache.get(
                key
            ) != validation_dataset.cache.get(
                key
            ):
                raise ValueError(
                    "Real train/validation caches "
                    f"disagree for {key!r}."
                )

        return

    train_graphs = torch.as_tensor(
        train_dataset.cache[
            "regime_graphs"
        ]
    )

    validation_graphs = torch.as_tensor(
        validation_dataset.cache[
            "regime_graphs"
        ]
    )

    if not torch.equal(
        train_graphs,
        validation_graphs,
    ):
        raise ValueError(
            "Synthetic train/validation caches "
            "must share the exact frozen "
            "regime_graphs tensor."
        )

    train_ids = (
        train_dataset.trajectory_ids
    )
    validation_ids = (
        validation_dataset.trajectory_ids
    )

    if (
        train_ids is None
        or validation_ids is None
    ):
        raise RuntimeError(
            "Synthetic trajectory IDs are "
            "unavailable."
        )

    overlap = (
        set(
            train_ids.tolist()
        )
        & set(
            validation_ids.tolist()
        )
    )

    if overlap:
        raise ValueError(
            "Synthetic train/validation splits "
            "share trajectory IDs: "
            f"{sorted(overlap)}. Splits must "
            "be trajectory-disjoint."
        )


def build_token_graph_dataloaders(
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    *,
    data_mode: TokenGraphDataMode = "auto",
    train_batch_size: int,
    validation_batch_size: int | None = None,
    num_workers: int,
    seed: int,
    pin_memory: bool | None = None,
    drop_last_train: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> TokenGraphDataLoaders:
    """Load compatible real/synthetic caches and build both loaders."""
    if validation_batch_size is None:
        validation_batch_size = (
            train_batch_size
        )

    train_dataset = (
        CachedTokenGraphDataset.from_path(
            train_cache_path,
            data_mode=data_mode,
        )
    )

    validation_dataset = (
        CachedTokenGraphDataset.from_path(
            validation_cache_path,
            data_mode=data_mode,
        )
    )

    _validate_train_validation_compatibility(
        train_dataset,
        validation_dataset,
    )

    return TokenGraphDataLoaders(
        train_dataset=train_dataset,
        validation_dataset=(
            validation_dataset
        ),
        train_loader=(
            build_token_graph_dataloader(
                train_dataset,
                batch_size=(
                    train_batch_size
                ),
                shuffle=True,
                num_workers=num_workers,
                seed=seed,
                drop_last=(
                    drop_last_train
                ),
                pin_memory=pin_memory,
                persistent_workers=(
                    persistent_workers
                ),
                prefetch_factor=(
                    prefetch_factor
                ),
            )
        ),
        validation_loader=(
            build_token_graph_dataloader(
                validation_dataset,
                batch_size=(
                    validation_batch_size
                ),
                shuffle=False,
                num_workers=num_workers,
                seed=seed,
                pin_memory=pin_memory,
                persistent_workers=(
                    persistent_workers
                ),
                prefetch_factor=(
                    prefetch_factor
                ),
            )
        ),
    )


def build_real_token_graph_dataloaders(
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    **kwargs: Any,
) -> TokenGraphDataLoaders:
    """Explicit real-data alias for the common loader factory."""
    return build_token_graph_dataloaders(
        train_cache_path,
        validation_cache_path,
        data_mode="real",
        **kwargs,
    )


def build_synthetic_token_graph_dataloaders(
    train_cache_path: str | Path,
    validation_cache_path: str | Path,
    **kwargs: Any,
) -> TokenGraphDataLoaders:
    """Explicit synthetic-data alias for the common loader factory."""
    return build_token_graph_dataloaders(
        train_cache_path,
        validation_cache_path,
        data_mode="synthetic",
        **kwargs,
    )


def _assert_common_batch(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    context_length: int,
    prediction_length: int,
    num_assets: int,
) -> None:
    expected = {
        "context_tokens": (
            batch_size,
            context_length,
            num_assets,
            2,
        ),
        "target_s1": (
            batch_size,
            prediction_length,
            num_assets,
        ),
        "target_s2": (
            batch_size,
            prediction_length,
            num_assets,
        ),
        "window_idx": (
            batch_size,
        ),
    }

    for key, shape in expected.items():
        if (
            key not in batch
            or tuple(
                batch[key].shape
            ) != shape
        ):
            raise AssertionError(
                f"Unexpected {key} batch shape; "
                f"expected {shape}."
            )

        if batch[key].dtype != torch.long:
            raise TypeError(
                f"{key} must collate as "
                "torch.long."
            )


def _make_synthetic_fixture(
    *,
    trajectory_ids: tuple[int, ...],
) -> dict[str, Any]:
    context_length = 6
    prediction_length = 5
    num_assets = 4
    num_regimes = 2
    windows_per_trajectory = 2
    num_windows = (
        len(trajectory_ids)
        * windows_per_trajectory
    )

    regime_graphs = torch.zeros(
        num_regimes,
        num_assets,
        num_assets,
        dtype=torch.float32,
    )

    for regime in range(
        num_regimes
    ):
        for target in range(
            num_assets
        ):
            source = (
                target
                + regime
                + 1
            ) % num_assets

            regime_graphs[
                regime,
                target,
                source,
            ] = 1.0

    trajectory_id = torch.tensor(
        [
            value
            for value in trajectory_ids
            for _ in range(
                windows_per_trajectory
            )
        ],
        dtype=torch.long,
    )

    regime_id = (
        torch.arange(
            num_windows,
            dtype=torch.long,
        )
        % num_regimes
    )

    origin_idx = (
        torch.arange(
            num_windows,
            dtype=torch.long,
        )
        + context_length
        - 1
    )

    return {
        "format_version": (
            SYNTHETIC_TOKEN_GRAPH_CACHE_VERSION
        ),
        "representation": (
            SYNTHETIC_TOKEN_GRAPH_REPRESENTATION
        ),
        "context_tokens": torch.randint(
            0,
            TOKEN_VOCABULARY_SIZE,
            (
                num_windows,
                context_length,
                num_assets,
                2,
            ),
            dtype=torch.int16,
        ),
        "target_s1": torch.randint(
            0,
            TOKEN_VOCABULARY_SIZE,
            (
                num_windows,
                prediction_length,
                num_assets,
            ),
            dtype=torch.int16,
        ),
        "target_s2": torch.randint(
            0,
            TOKEN_VOCABULARY_SIZE,
            (
                num_windows,
                prediction_length,
                num_assets,
            ),
            dtype=torch.int16,
        ),
        "true_graph": (
            regime_graphs[
                regime_id
            ]
        ),
        "regime_id": regime_id,
        "trajectory_id": (
            trajectory_id
        ),
        "origin_idx": origin_idx,
        "regime_graphs": (
            regime_graphs
        ),
        "asset_cols": [
            f"node_{index}"
            for index in range(
                num_assets
            )
        ],
        "context_length": (
            context_length
        ),
        "prediction_length": (
            prediction_length
        ),
        "dense_horizons": list(
            range(
                1,
                prediction_length + 1,
            )
        ),
        "graph_orientation": (
            GRAPH_ORIENTATION
        ),
    }


def _run_synthetic_contract_smoke_test(
) -> None:
    train_dataset = (
        CachedTokenGraphDataset(
            _make_synthetic_fixture(
                trajectory_ids=(
                    100,
                    101,
                )
            )
        )
    )

    validation_dataset = (
        CachedTokenGraphDataset(
            _make_synthetic_fixture(
                trajectory_ids=(
                    200,
                )
            )
        )
    )

    _validate_train_validation_compatibility(
        train_dataset,
        validation_dataset,
    )

    batch = next(
        iter(
            build_token_graph_dataloader(
                train_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                seed=42,
                pin_memory=False,
            )
        )
    )

    _assert_common_batch(
        batch,
        batch_size=2,
        context_length=(
            train_dataset.context_length
        ),
        prediction_length=(
            train_dataset.prediction_length
        ),
        num_assets=(
            train_dataset.num_assets
        ),
    )

    if tuple(
        batch[
            "true_graph"
        ].shape
    ) != (
        2,
        4,
        4,
    ):
        raise AssertionError(
            "Synthetic true_graph shape "
            "is incorrect."
        )


def _run_real_cache_smoke_test(
    *,
    train_cache_path: Path,
    validation_cache_path: Path,
    train_batch_size: int,
    validation_batch_size: int,
    num_workers: int,
    seed: int,
) -> None:
    loaders = (
        build_real_token_graph_dataloaders(
            train_cache_path,
            validation_cache_path,
            train_batch_size=(
                train_batch_size
            ),
            validation_batch_size=(
                validation_batch_size
            ),
            num_workers=num_workers,
            seed=seed,
            pin_memory=False,
        )
    )

    train_batch = next(
        iter(
            loaders.train_loader
        )
    )

    validation_batch = next(
        iter(
            loaders.validation_loader
        )
    )

    _assert_common_batch(
        train_batch,
        batch_size=min(
            train_batch_size,
            len(
                loaders.train_dataset
            ),
        ),
        context_length=(
            loaders.train_dataset.context_length
        ),
        prediction_length=(
            loaders.train_dataset.prediction_length
        ),
        num_assets=(
            loaders.train_dataset.num_assets
        ),
    )

    _assert_common_batch(
        validation_batch,
        batch_size=min(
            validation_batch_size,
            len(
                loaders.validation_dataset
            ),
        ),
        context_length=(
            loaders.validation_dataset.context_length
        ),
        prediction_length=(
            loaders.validation_dataset.prediction_length
        ),
        num_assets=(
            loaders.validation_dataset.num_assets
        ),
    )

    expected_validation_indices = (
        torch.arange(
            validation_batch[
                "window_idx"
            ].shape[0],
            dtype=torch.long,
        )
    )

    if not torch.equal(
        validation_batch[
            "window_idx"
        ],
        expected_validation_indices,
    ):
        raise AssertionError(
            "Validation DataLoader did not "
            "preserve cache order."
        )

    repeated_batch = next(
        iter(
            build_token_graph_dataloader(
                loaders.train_dataset,
                batch_size=(
                    train_batch_size
                ),
                shuffle=True,
                num_workers=(
                    num_workers
                ),
                seed=seed,
                pin_memory=False,
            )
        )
    )

    if not torch.equal(
        train_batch[
            "window_idx"
        ],
        repeated_batch[
            "window_idx"
        ],
    ):
        raise AssertionError(
            "Training shuffle is not "
            "reproducible for a fixed seed."
        )

    print(
        "Train windows:",
        len(
            loaders.train_dataset
        ),
    )

    print(
        "Validation windows:",
        len(
            loaders.validation_dataset
        ),
    )

    print(
        "Context tokens:",
        tuple(
            train_batch[
                "context_tokens"
            ].shape
        ),
    )

    print(
        "Dense targets:",
        tuple(
            train_batch[
                "target_s1"
            ].shape
        ),
    )

    print(
        "Evaluation truth:",
        tuple(
            validation_batch[
                "evaluation_true"
            ].shape
        ),
    )


def _build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the production "
            "cached-token data layer."
        )
    )

    parser.add_argument(
        "--train-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--val-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


def main(
) -> None:
    args = (
        _build_argument_parser()
        .parse_args()
    )

    _run_real_cache_smoke_test(
        train_cache_path=(
            args.train_cache
        ),
        validation_cache_path=(
            args.val_cache
        ),
        train_batch_size=(
            args.train_batch_size
        ),
        validation_batch_size=(
            args.validation_batch_size
        ),
        num_workers=(
            args.num_workers
        ),
        seed=args.seed,
    )

    _run_synthetic_contract_smoke_test()

    print(
        "CACHED TOKEN GRAPH DATA LAYER "
        "TEST PASSED"
    )


if __name__ == "__main__":
    main()

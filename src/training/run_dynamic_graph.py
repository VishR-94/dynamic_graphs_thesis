from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import (
    CachedTokenGraphDataset,
    TokenGraphDataLoaders,
    build_token_graph_dataloader,
    build_token_graph_dataloaders,
)
from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.evaluation.metrics import ForecastEvaluator
from src.models.dynamic_graph.config import (
    load_dynamic_graph_config,
    validate_dynamic_graph_config,
)
from src.models.dynamic_graph.contracts import GraphOutput
from src.models.dynamic_graph.fixed_graph_resource import (
    FixedGraphResource,
    FixedGraphResourceConfig,
    fit_absolute_return_correlation_resource,
)
from src.models.dynamic_graph.future_predictor import (
    FutureTokenPrediction,
    compute_future_token_loss,
)
from src.models.dynamic_graph.losses import (
    DynamicGraphLoss,
    GraphRegularisationConfig,
    build_adjacent_window_pairs,
    combine_dynamic_graph_losses,
    compute_graph_regularisation,
)
from src.models.dynamic_graph.model import (
    DynamicGraphTokenForecaster,
    GeneratedTokenForecast,
    SampledGeneratedTokenForecast,
)
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]
PredictionResult = dict[str, Any]

REAL_REPRESENTATION = "origin_aligned_kronos_forecasting_tokens"
SYNTHETIC_REPRESENTATION = "kronos_basedygraph_window_tokens"
GRAPH_ORIENTATION = "row=target,column=source"
OHLCV_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)
CLOSE_CHANNEL_INDEX = OHLCV_CHANNELS.index("close")


@dataclass(frozen=True)
class TeacherForcedEpochMetrics:
    """Epoch-averaged supervised objective and graph diagnostics."""

    total_loss: float
    token_loss: float
    graph_regularisation_loss: float
    backcast_loss: float
    backcast_penalty: float

    s1_loss: float
    s2_loss: float
    s1_accuracy: float
    s2_accuracy: float

    graph_mean_row_entropy: float | None
    graph_mean_effective_neighbours: float | None
    graph_entropy_penalty: float
    graph_target_entropy_penalty: float
    graph_temporal_smooth_penalty: float
    graph_warmup_scale: float
    graph_valid_smoothing_pairs: int
    spatial_beta: float | None

    s1_loss_by_step: Tensor
    s2_loss_by_step: Tensor
    s1_accuracy_by_step: Tensor
    s2_accuracy_by_step: Tensor

    examples: int
    seconds: float


@dataclass(frozen=True)
class GenerationMetrics:
    s1_accuracy: float
    s2_accuracy: float
    s1_accuracy_by_step: Tensor
    s2_accuracy_by_step: Tensor
    examples: int
    seconds: float


@dataclass
class ValidationBundle:
    generation_metrics: GenerationMetrics
    token_artifacts: dict[str, Any]
    graph_artifacts: dict[str, Any]
    sampled_price_path_artifacts: dict[str, Any] | None
    prediction_result: PredictionResult | None
    metric_results: dict[str, Tensor] | None
    metric_table: pd.DataFrame | None
    primary_score: float | None
    diagnostics: dict[str, Any]


class GraphArtifactAccumulator:
    """Collect graph tensors without averaging windows, heads or layers."""

    def __init__(
        self,
        *,
        asset_cols: Sequence[str],
        graph_type: str,
        num_layers: int,
        num_heads: int,
    ) -> None:
        self.asset_cols = tuple(str(value) for value in asset_cols)
        self.graph_type = str(graph_type)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        self._selected: list[Tensor] = []
        self._base: list[Tensor] = []
        self._dynamic: list[Tensor] = []
        self._logits: list[Tensor] = []
        self._alpha: list[Tensor] = []
        self._spatial_beta: list[Tensor] = []
        self._per_layer: list[list[Tensor]] = [
            [] for _ in range(self.num_layers)
        ]

        self._window_idx: list[Tensor] = []
        self._sample_idx: list[Tensor] = []
        self._origin_idx: list[Tensor] = []
        self._target_indices: list[Tensor] = []
        self._regime_id: list[Tensor] = []
        self._trajectory_id: list[Tensor] = []
        self._true_graph: list[Tensor] = []
        self._dates: list[str] = []

        self._saw_selected = False
        self._saw_base = False
        self._saw_dynamic = False
        self._saw_logits = False
        self._saw_alpha = False
        self._saw_spatial_beta = False
        self._saw_true_graph = False

    @staticmethod
    def _cpu_float(values: Tensor) -> Tensor:
        return values.detach().cpu().to(torch.float32).contiguous()

    @staticmethod
    def _expand_graph_to_batch(
        values: Tensor,
        *,
        batch_size: int,
        name: str,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B, G, N, N] or [1, G, N, N]."
            )

        observed_batch = int(values.shape[0])

        if observed_batch == batch_size:
            return values

        if observed_batch == 1:
            return values.expand(batch_size, -1, -1, -1)

        raise ValueError(
            f"{name} batch dimension {observed_batch} cannot align with "
            f"batch size {batch_size}."
        )

    @staticmethod
    def _expand_alpha_to_batch(
        alpha: Tensor,
        *,
        batch_size: int,
    ) -> Tensor:
        alpha = alpha.detach()

        if alpha.ndim == 0:
            return alpha.reshape(1, 1).expand(batch_size, 1)

        if alpha.ndim == 1:
            return alpha.unsqueeze(0).expand(batch_size, -1)

        if alpha.ndim == 2:
            if int(alpha.shape[0]) == batch_size:
                return alpha
            if int(alpha.shape[0]) == 1:
                return alpha.expand(batch_size, -1)

        raise ValueError(
            "graph.alpha must be scalar, [G], [1, G], or [B, G]."
        )

    @staticmethod
    def _append_optional_index(
        destination: list[Tensor],
        batch: Mapping[str, Any],
        key: str,
    ) -> None:
        if key not in batch:
            return

        values = torch.as_tensor(batch[key]).detach().cpu().to(torch.long)
        destination.append(values.contiguous())

    def add(
        self,
        graph: GraphOutput,
        batch: Mapping[str, Any],
        *,
        batch_size: int,
        spatial_beta: Tensor | None = None,
    ) -> None:
        if len(graph.per_layer) != self.num_layers:
            raise ValueError(
                "The number of graph.per_layer tensors does not match "
                "the configured spatio-temporal block count."
            )

        if graph.selected is not None:
            self._saw_selected = True
            selected = self._expand_graph_to_batch(
                graph.selected,
                batch_size=batch_size,
                name="graph.selected",
            )
            self._selected.append(self._cpu_float(selected))

        if graph.base is not None:
            self._saw_base = True
            base = self._expand_graph_to_batch(
                graph.base,
                batch_size=batch_size,
                name="graph.base",
            )
            self._base.append(self._cpu_float(base))

        if graph.dynamic is not None:
            self._saw_dynamic = True
            dynamic = self._expand_graph_to_batch(
                graph.dynamic,
                batch_size=batch_size,
                name="graph.dynamic",
            )
            self._dynamic.append(self._cpu_float(dynamic))

        if graph.logits is not None:
            self._saw_logits = True
            logits = self._expand_graph_to_batch(
                graph.logits,
                batch_size=batch_size,
                name="graph.logits",
            )
            self._logits.append(self._cpu_float(logits))

        if graph.alpha is not None:
            self._saw_alpha = True
            alpha = self._expand_alpha_to_batch(
                graph.alpha,
                batch_size=batch_size,
            )
            self._alpha.append(self._cpu_float(alpha))

        if spatial_beta is not None:
            self._saw_spatial_beta = True
            beta = torch.as_tensor(spatial_beta).detach().float()
            if beta.numel() != 1 or not torch.isfinite(beta).all():
                raise ValueError("spatial_beta must be one finite scalar.")
            self._spatial_beta.append(
                beta.reshape(1, 1).expand(batch_size, 1).cpu().contiguous()
            )

        for layer_index, layer_graph in enumerate(graph.per_layer):
            if layer_graph is None:
                continue

            expanded = self._expand_graph_to_batch(
                layer_graph,
                batch_size=batch_size,
                name=f"graph.per_layer[{layer_index}]",
            )
            self._per_layer[layer_index].append(self._cpu_float(expanded))

        self._append_optional_index(self._window_idx, batch, "window_idx")
        self._append_optional_index(self._sample_idx, batch, "sample_idx")
        self._append_optional_index(self._origin_idx, batch, "origin_idx")
        self._append_optional_index(
            self._target_indices,
            batch,
            "target_indices",
        )
        self._append_optional_index(self._regime_id, batch, "regime_id")
        self._append_optional_index(
            self._trajectory_id,
            batch,
            "trajectory_id",
        )

        if "true_graph" in batch:
            self._saw_true_graph = True
            self._true_graph.append(
                self._cpu_float(torch.as_tensor(batch["true_graph"]))
            )

        if "date" in batch:
            dates = batch["date"]
            if isinstance(dates, str):
                self._dates.append(dates)
            else:
                self._dates.extend(str(value) for value in dates)

    @staticmethod
    def _concatenate_optional(
        values: list[Tensor],
        *,
        was_seen: bool,
        name: str,
    ) -> Tensor | None:
        if not was_seen:
            if values:
                raise RuntimeError(f"Unexpected accumulated values for {name}.")
            return None

        if not values:
            raise RuntimeError(f"No accumulated values were retained for {name}.")

        return torch.cat(values, dim=0).contiguous()

    @staticmethod
    def _concatenate_indices(values: list[Tensor]) -> Tensor | None:
        if not values:
            return None
        return torch.cat(values, dim=0).contiguous()

    def finalise(self) -> dict[str, Any]:
        per_layer: list[Tensor | None] = []

        for layer_values in self._per_layer:
            if layer_values:
                per_layer.append(torch.cat(layer_values, dim=0).contiguous())
            else:
                per_layer.append(None)

        artifacts: dict[str, Any] = {
            "graph_type": self.graph_type,
            "graph_orientation": GRAPH_ORIENTATION,
            "asset_cols": list(self.asset_cols),
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "selected": self._concatenate_optional(
                self._selected,
                was_seen=self._saw_selected,
                name="selected",
            ),
            "per_layer": tuple(per_layer),
            "base": self._concatenate_optional(
                self._base,
                was_seen=self._saw_base,
                name="base",
            ),
            "dynamic": self._concatenate_optional(
                self._dynamic,
                was_seen=self._saw_dynamic,
                name="dynamic",
            ),
            "alpha": self._concatenate_optional(
                self._alpha,
                was_seen=self._saw_alpha,
                name="alpha",
            ),
            "spatial_beta": self._concatenate_optional(
                self._spatial_beta,
                was_seen=self._saw_spatial_beta,
                name="spatial_beta",
            ),
            "logits": self._concatenate_optional(
                self._logits,
                was_seen=self._saw_logits,
                name="logits",
            ),
            "window_idx": self._concatenate_indices(self._window_idx),
            "sample_idx": self._concatenate_indices(self._sample_idx),
            "origin_idx": self._concatenate_indices(self._origin_idx),
            "target_indices": self._concatenate_indices(
                self._target_indices
            ),
            "regime_id": self._concatenate_indices(self._regime_id),
            "trajectory_id": self._concatenate_indices(
                self._trajectory_id
            ),
            "true_graph": self._concatenate_optional(
                self._true_graph,
                was_seen=self._saw_true_graph,
                name="true_graph",
            ),
            "dates": list(self._dates),
        }

        expected_windows: int | None = None

        for key in (
            "window_idx",
            "sample_idx",
            "origin_idx",
            "regime_id",
            "trajectory_id",
        ):
            values = artifacts[key]
            if values is not None:
                expected_windows = int(values.shape[0])
                break

        if expected_windows is None and artifacts["selected"] is not None:
            expected_windows = int(artifacts["selected"].shape[0])

        if expected_windows is not None:
            for key in (
                "selected",
                "base",
                "dynamic",
                "alpha",
                "spatial_beta",
                "logits",
                "true_graph",
            ):
                values = artifacts[key]
                if values is not None and int(values.shape[0]) != expected_windows:
                    raise ValueError(
                        f"Graph artifact {key!r} has {int(values.shape[0])} "
                        f"windows; expected {expected_windows}."
                    )

            for layer_index, values in enumerate(artifacts["per_layer"]):
                if values is not None and int(values.shape[0]) != expected_windows:
                    raise ValueError(
                        f"Graph layer {layer_index} has "
                        f"{int(values.shape[0])} windows; expected "
                        f"{expected_windows}."
                    )

            if artifacts["dates"] and len(artifacts["dates"]) != expected_windows:
                raise ValueError(
                    "Date metadata length does not match graph-window count."
                )

        return artifacts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train, validate and checkpoint the shared Kronos-token "
            "dynamic-graph forecaster."
        )
    )

    parser.add_argument(
        "--dynamic-config",
        type=Path,
        default=Path("configs/dynamic_graph.yaml"),
    )
    parser.add_argument(
        "--forecasting-config",
        type=Path,
        default=Path("configs/forecasting.yaml"),
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Raw candle directory required for real-data ForecastEvaluator "
            "metrics, including the training-derived MASE scale."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Named preset from --dynamic-config. When omitted, the "
            "YAML default_preset is used. The selected name is "
            "validated against the YAML presets mapping."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Repeatable nested config override. Values are parsed as YAML. "
            "Example: --set training.learning_rate=5e-5"
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--decode-series-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--validation-decode-every",
        type=int,
        default=1,
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from <run_dir>/last_checkpoint.pt.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load best_checkpoint.pt and regenerate validation artifacts.",
    )
    parser.add_argument(
        "--temperature-sweep",
        action="store_true",
        help=(
            "Load best_checkpoint.pt and run the configured inference-only "
            "10-path coarse-token temperature sweep. No weights are trained."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory before starting.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dynamic-graph-financial-forecasting",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])

    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable.")


def atomic_json_save(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            values,
            file,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        file.write("\n")

    os.replace(temporary, path)


def atomic_torch_save(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def atomic_csv_save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _set_nested_value(config: ConfigDict, path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]

    if not parts:
        raise ValueError("Override path must not be empty.")

    current: ConfigDict = config

    for part in parts[:-1]:
        existing = current.get(part)

        if not isinstance(existing, dict):
            raise KeyError(
                f"Cannot descend through config path {path!r}; "
                f"{part!r} is missing or is not a mapping."
            )

        current = existing

    leaf = parts[-1]

    if leaf not in current:
        raise KeyError(
            f"Config override path {path!r} does not refer to an existing key."
        )

    current[leaf] = value


def apply_overrides(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
    resolved = deepcopy(config)

    for expression in args.set:
        if "=" not in expression:
            raise ValueError(
                f"Invalid --set expression {expression!r}; expected PATH=VALUE."
            )

        path, raw_value = expression.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested_value(resolved, path.strip(), value)

    training = resolved["training"]

    explicit_values = {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.train_batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
        "mixed_precision": args.mixed_precision,
    }

    for key, value in explicit_values.items():
        if value is not None:
            training[key] = value

    validate_dynamic_graph_config(resolved)
    return resolved


def _validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value

def _mean_cumulative_log_change_mae_at_horizons(
    *,
    metric_results: Mapping[str, Tensor],
    available_horizons: Sequence[int],
    selected_horizons: Sequence[int],
) -> float:
    """Average decoded CLG-MAE over explicitly selected horizons."""

    available = tuple(
        int(horizon)
        for horizon in available_horizons
    )
    selected = tuple(
        int(horizon)
        for horizon in selected_horizons
    )

    if not selected:
        raise ValueError(
            "selected_horizons must contain at least one horizon."
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "selected_horizons must not contain duplicates."
        )

    horizon_to_index = {
        horizon: index
        for index, horizon in enumerate(available)
    }

    missing = [
        horizon
        for horizon in selected
        if horizon not in horizon_to_index
    ]

    if missing:
        raise ValueError(
            "Selected early-stopping horizons are not present in the "
            f"evaluation horizons. Missing={missing}, "
            f"available={list(available)}."
        )

    metric_name = "cumulative_log_change_mae"

    if metric_name not in metric_results:
        raise KeyError(
            f"Validation metrics do not contain {metric_name!r}."
        )

    values = (
        torch.as_tensor(
            metric_results[metric_name]
        )
        .detach()
        .to(torch.float64)
    )

    if values.ndim < 1:
        raise ValueError(
            "cumulative_log_change_mae must have a horizon dimension."
        )

    if int(values.shape[0]) != len(available):
        raise ValueError(
            "The cumulative-log-change MAE horizon dimension does not "
            "match the configured evaluation horizons. "
            f"Observed shape={tuple(values.shape)}, "
            f"horizons={list(available)}."
        )

    indices = torch.tensor(
        [
            horizon_to_index[horizon]
            for horizon in selected
        ],
        dtype=torch.long,
        device=values.device,
    )

    selected_values = values.index_select(
        dim=0,
        index=indices,
    )

    if not torch.isfinite(selected_values).all():
        raise ValueError(
            "The selected cumulative-log-change MAE values contain "
            "non-finite entries."
        )

    return float(
        selected_values.mean().item()
    )

def _validate_optional_window_limit(
    value: int | None,
    *,
    name: str,
) -> int | None:
    if value is None:
        return None

    return _validate_positive_int(value, name=name)


def _git_value(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _config_signature(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _worker_seed(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "drop_last": False,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
        "worker_init_fn": _worker_seed if num_workers > 0 else None,
        "generator": generator,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def limit_dataset(
    dataset: CachedTokenGraphDataset,
    limit: int | None,
) -> Dataset[dict[str, Any]]:
    if limit is None or limit >= len(dataset):
        return dataset

    return Subset(dataset, range(int(limit)))


def move_training_batch(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(batch["context_tokens"]).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
        torch.as_tensor(batch["target_s1"]).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
        torch.as_tensor(batch["target_s2"]).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ),
    )


def _future_prediction_for_loss(output: Any) -> FutureTokenPrediction:
    selected_s2 = (
        output.s2_logits.argmax(dim=-1)
        if output.s2_logits is not None
        else None
    )

    return FutureTokenPrediction(
        future_hidden=output.future_hidden,
        s1_logits=output.s1_logits,
        s2_logits=output.s2_logits,
        selected_s1=output.s1_logits.argmax(dim=-1),
        selected_s2=selected_s2,
    )


def _batch_true_graph(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    values = batch.get("true_graph")

    if values is None:
        return None

    return torch.as_tensor(
        values
    ).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )


def _batch_adjacent_window_pairs(
    batch: Mapping[str, Any],
    *,
    enabled: bool,
    expected_origin_delta: int,
) -> Tensor | None:
    if not enabled:
        return None

    if "origin_idx" not in batch:
        raise KeyError(
            "Temporal graph smoothing requires origin_idx metadata."
        )

    if "trajectory_id" in batch:
        return build_adjacent_window_pairs(
            origin_idx=torch.as_tensor(
                batch["origin_idx"]
            ),
            trajectory_id=torch.as_tensor(
                batch["trajectory_id"]
            ),
            expected_origin_delta=expected_origin_delta,
        )

    if "sample_idx" in batch:
        return build_adjacent_window_pairs(
            origin_idx=torch.as_tensor(
                batch["origin_idx"]
            ),
            sample_idx=torch.as_tensor(
                batch["sample_idx"]
            ),
            expected_origin_delta=expected_origin_delta,
        )

    raise KeyError(
        "Temporal graph smoothing requires sample_idx for real data "
        "or trajectory_id for synthetic data."
    )


def compute_model_loss(
    model: DynamicGraphTokenForecaster,
    output: Any,
    target_s1: Tensor,
    target_s2: Tensor,
    batch: Mapping[str, Any],
    *,
    graph_regularisation_config: GraphRegularisationConfig,
    current_epoch: int,
    expected_origin_delta: int,
) -> DynamicGraphLoss:
    """Compute the complete token, graph and optional backcast objective."""
    token_loss = compute_future_token_loss(
        _future_prediction_for_loss(output),
        target_s1,
        target_s2,
        loss_config=model.config.loss,
        evaluation_horizons=model.config.heads.evaluation_horizons,
        s2_loss_weight=model.config.heads.s2_loss_weight,
    )

    true_graph = _batch_true_graph(
        batch,
        device=output.s1_logits.device,
        dtype=output.s1_logits.dtype,
    )

    adjacent_window_pairs = _batch_adjacent_window_pairs(
        batch,
        enabled=(
            graph_regularisation_config
            .graph_temporal_smooth_reg
            > 0.0
        ),
        expected_origin_delta=expected_origin_delta,
    )

    graph_loss = compute_graph_regularisation(
        output.graph,
        config=graph_regularisation_config,
        current_epoch=current_epoch,
        reference_tensor=token_loss.total,
        true_graph=true_graph,
        adjacent_window_pairs=adjacent_window_pairs,
    )

    backcast_loss: Tensor | None = None

    if model.config.backcast.enabled:
        if "context_normalised_ohlcv" not in batch:
            raise KeyError(
                "Backcasting is enabled, but the cache does not contain "
                "context_normalised_ohlcv. Regenerate the cache with an "
                "explicit backcast target before enabling this ablation."
            )

        if output.backcast is None:
            raise RuntimeError(
                "Backcasting is enabled but the model returned None."
            )

        target = torch.as_tensor(
            batch["context_normalised_ohlcv"]
        ).to(
            device=output.backcast.device,
            dtype=output.backcast.dtype,
            non_blocking=True,
        )

        if tuple(target.shape) != tuple(
            output.backcast.shape
        ):
            raise ValueError(
                "Backcast target and output shapes differ: "
                f"{tuple(target.shape)} versus "
                f"{tuple(output.backcast.shape)}."
            )

        backcast_loss = torch.nn.functional.mse_loss(
            output.backcast,
            target,
        )

    return combine_dynamic_graph_losses(
        token_loss,
        graph_loss,
        backcast_loss=backcast_loss,
        backcast_loss_weight=(
            model.config.backcast.loss_weight
        ),
    )

def _autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def _new_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def _trainable_parameter_partition(
    model: DynamicGraphTokenForecaster,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition adjacency parameters from the rest of the token model.

    Only parameters owned by ``model.graph_learners`` use the dedicated
    graph learning rate. ModernTCN, the future-token head, spatial message
    passing and the learned spatial beta remain in the backbone group.
    """
    graph_parameters = [
        parameter
        for parameter in model.graph_learners.parameters()
        if parameter.requires_grad
    ]
    graph_ids = {id(parameter) for parameter in graph_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in graph_ids
    ]
    all_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(backbone_parameters) + len(graph_parameters) != len(all_parameters):
        raise AssertionError("Optimizer parameter partition lost parameters.")
    if {id(parameter) for parameter in backbone_parameters} & graph_ids:
        raise AssertionError("Optimizer parameter groups overlap.")
    return backbone_parameters, graph_parameters


def _build_optimizer(
    model: DynamicGraphTokenForecaster,
    *,
    optimizer_name: str,
    learning_rate: float,
    graph_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'.")
    optimizer_class = (
        torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    )
    backbone_parameters, graph_parameters = _trainable_parameter_partition(model)
    groups: list[dict[str, Any]] = [
        {
            "params": backbone_parameters,
            "lr": float(learning_rate),
            "base_lr": float(learning_rate),
            "name": "backbone",
        }
    ]
    if graph_parameters:
        groups.append(
            {
                "params": graph_parameters,
                "lr": float(graph_learning_rate),
                "base_lr": float(graph_learning_rate),
                "name": "graph",
            }
        )
    return optimizer_class(groups, weight_decay=float(weight_decay))


def _current_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {"backbone": None, "graph": None}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "backbone" if index == 0 else "graph"))
        if name in result:
            result[name] = float(group["lr"])
    if result["backbone"] is None:
        raise RuntimeError("Optimizer is missing its backbone parameter group.")
    return result


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: str,
    completed_epoch: int,
) -> dict[str, float | None]:
    if scheduler not in {"none", "modern_tcn_type3"}:
        raise ValueError(
            "training.scheduler must be 'none' or 'modern_tcn_type3'."
        )
    if scheduler == "none":
        return _current_learning_rates(optimizer)
    multiplier = (
        1.0 if int(completed_epoch) < 3 else 0.9 ** (int(completed_epoch) - 3)
    )
    for group in optimizer.param_groups:
        if "base_lr" not in group:
            raise RuntimeError(
                "Optimizer parameter group is missing base_lr; cannot "
                "preserve the backbone-to-graph learning-rate ratio."
            )
        group["lr"] = float(group["base_lr"]) * multiplier
    return _current_learning_rates(optimizer)


def synchronise_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_teacher_forced_epoch(
    *,
    model: DynamicGraphTokenForecaster,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    use_amp: bool,
    gradient_clip_norm: float,
    description: str,
    graph_regularisation_config: GraphRegularisationConfig,
    current_epoch: int,
    expected_origin_delta: int,
) -> TeacherForcedEpochMetrics:
    training = optimizer is not None
    model.train(training)

    synchronise_device(device)
    start = perf_counter()
    example_count = 0

    total_loss_sum = 0.0
    token_loss_sum = 0.0
    graph_regularisation_loss_sum = 0.0
    backcast_loss_sum = 0.0
    backcast_penalty_sum = 0.0

    s1_loss_sum = 0.0
    s2_loss_sum = 0.0

    graph_entropy_penalty_sum = 0.0
    graph_target_entropy_penalty_sum = 0.0
    graph_temporal_smooth_penalty_sum = 0.0
    graph_warmup_scale_sum = 0.0
    graph_valid_smoothing_pairs = 0

    graph_entropy_sum = 0.0
    graph_effective_neighbours_sum = 0.0
    graph_diagnostic_examples = 0
    spatial_beta_sum = 0.0
    spatial_beta_examples = 0

    prediction_length = model.config.prediction_length
    predict_s2 = model.config.heads.predicts_s2

    s1_loss_by_step_sum = torch.zeros(
        prediction_length,
        dtype=torch.float64,
    )
    s2_loss_by_step_sum = torch.zeros(
        prediction_length,
        dtype=torch.float64,
    )
    s1_correct_by_step = torch.zeros(
        prediction_length,
        dtype=torch.float64,
    )
    s2_correct_by_step = torch.zeros(
        prediction_length,
        dtype=torch.float64,
    )
    token_count_by_step = torch.zeros(
        prediction_length,
        dtype=torch.float64,
    )

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        (
            context_tokens,
            target_s1,
            target_s2,
        ) = move_training_batch(
            batch,
            device=device,
        )

        batch_size = int(
            context_tokens.shape[0]
        )
        num_nodes = int(
            context_tokens.shape[2]
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        grad_context = (
            torch.enable_grad()
            if training
            else torch.inference_mode()
        )

        with grad_context:
            with _autocast_context(
                device,
                use_amp,
            ):
                output = model(
                    context_tokens,
                    target_s1=target_s1,
                    target_s2=target_s2,
                )

                loss = compute_model_loss(
                    model,
                    output,
                    target_s1,
                    target_s2,
                    batch,
                    graph_regularisation_config=(
                        graph_regularisation_config
                    ),
                    current_epoch=current_epoch,
                    expected_origin_delta=(
                        expected_origin_delta
                    ),
                )

            if training:
                if use_amp:
                    scaler.scale(
                        loss.total
                    ).backward()
                    scaler.unscale_(
                        optimizer
                    )
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=(
                            gradient_clip_norm
                        ),
                    )
                    scaler.step(
                        optimizer
                    )
                    scaler.update()
                else:
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=(
                            gradient_clip_norm
                        ),
                    )
                    optimizer.step()

        example_count += batch_size

        total_loss_sum += (
            float(
                loss.total.detach().item()
            )
            * batch_size
        )
        token_loss_sum += (
            float(
                loss.token.total.detach().item()
            )
            * batch_size
        )
        graph_regularisation_loss_sum += (
            float(
                loss.graph.total.detach().item()
            )
            * batch_size
        )
        backcast_loss_sum += (
            float(
                loss.backcast_loss.detach().item()
            )
            * batch_size
        )
        backcast_penalty_sum += (
            float(
                loss.backcast_penalty
                .detach()
                .item()
            )
            * batch_size
        )

        s1_loss_sum += (
            float(
                loss.s1.detach().item()
            )
            * batch_size
        )
        s2_loss_sum += (
            float(
                loss.s2.detach().item()
            )
            * batch_size
        )

        graph_entropy_penalty_sum += (
            float(
                loss.graph.entropy_penalty
                .detach()
                .item()
            )
            * batch_size
        )
        graph_target_entropy_penalty_sum += (
            float(
                loss.graph.target_entropy_penalty
                .detach()
                .item()
            )
            * batch_size
        )
        graph_temporal_smooth_penalty_sum += (
            float(
                loss.graph.temporal_smooth_penalty
                .detach()
                .item()
            )
            * batch_size
        )
        graph_warmup_scale_sum += (
            float(
                loss.graph.warmup_scale
                .detach()
                .item()
            )
            * batch_size
        )
        graph_valid_smoothing_pairs += int(
            loss.graph.valid_smoothing_pairs
        )

        if (
            loss.graph.mean_row_entropy
            is not None
        ):
            if (
                loss.graph
                .mean_effective_neighbours
                is None
            ):
                raise RuntimeError(
                    "Graph entropy was returned without "
                    "effective-neighbour diagnostics."
                )

            graph_entropy_sum += (
                float(
                    loss.graph.mean_row_entropy
                    .detach()
                    .item()
                )
                * batch_size
            )
            graph_effective_neighbours_sum += (
                float(
                    loss.graph
                    .mean_effective_neighbours
                    .detach()
                    .item()
                )
                * batch_size
            )
            graph_diagnostic_examples += (
                batch_size
            )

        if output.spatial_beta is not None:
            beta_value = torch.as_tensor(
                output.spatial_beta
            ).detach().float()
            if beta_value.numel() != 1 or not torch.isfinite(beta_value).all():
                raise FloatingPointError(
                    "The learned spatial beta is non-finite."
                )
            spatial_beta_sum += float(beta_value.item()) * batch_size
            spatial_beta_examples += batch_size

        s1_loss_by_step_sum += (
            loss.s1_by_step
            .detach()
            .cpu()
            .to(
                torch.float64
            )
            * batch_size
        )
        s2_loss_by_step_sum += (
            loss.s2_by_step
            .detach()
            .cpu()
            .to(
                torch.float64
            )
            * batch_size
        )

        s1_predictions = (
            output.s1_logits
            .detach()
            .argmax(
                dim=-1
            )
        )
        s1_correct_by_step += (
            (
                s1_predictions
                == target_s1
            )
            .sum(
                dim=(
                    0,
                    2,
                )
            )
            .detach()
            .cpu()
            .to(
                torch.float64
            )
        )

        if predict_s2:
            if output.s2_logits is None:
                raise RuntimeError(
                    "Full-token mode returned no s2 logits."
                )

            s2_predictions = (
                output.s2_logits
                .detach()
                .argmax(dim=-1)
            )
            s2_correct_by_step += (
                (s2_predictions == target_s2)
                .sum(dim=(0, 2))
                .detach()
                .cpu()
                .to(torch.float64)
            )

        token_count_by_step += float(
            batch_size
            * num_nodes
        )

        progress.set_postfix(
            objective=(
                f"{float(loss.total.detach().item()):.4f}"
            ),
            token=(
                f"{float(loss.token.total.detach().item()):.4f}"
            ),
            refresh=False,
        )

    synchronise_device(
        device
    )

    if example_count == 0:
        raise RuntimeError(
            "The DataLoader yielded no examples."
        )

    s1_accuracy_by_step = (
        s1_correct_by_step
        / token_count_by_step.clamp_min(
            1
        )
    )
    if predict_s2:
        s2_accuracy_by_step = (
            s2_correct_by_step
            / token_count_by_step.clamp_min(1)
        )
        s2_accuracy = float(
            s2_correct_by_step.sum().item()
            / token_count_by_step
            .sum()
            .clamp_min(1)
            .item()
        )
    else:
        s2_accuracy_by_step = torch.full(
            (prediction_length,),
            float("nan"),
            dtype=torch.float64,
        )
        s2_accuracy = float("nan")

    if graph_diagnostic_examples > 0:
        mean_graph_entropy: (
            float | None
        ) = (
            graph_entropy_sum
            / graph_diagnostic_examples
        )
        mean_effective_neighbours: (
            float | None
        ) = (
            graph_effective_neighbours_sum
            / graph_diagnostic_examples
        )
    else:
        mean_graph_entropy = None
        mean_effective_neighbours = None

    return TeacherForcedEpochMetrics(
        total_loss=(
            total_loss_sum
            / example_count
        ),
        token_loss=(
            token_loss_sum
            / example_count
        ),
        graph_regularisation_loss=(
            graph_regularisation_loss_sum
            / example_count
        ),
        backcast_loss=(
            backcast_loss_sum
            / example_count
        ),
        backcast_penalty=(
            backcast_penalty_sum
            / example_count
        ),
        s1_loss=(
            s1_loss_sum
            / example_count
        ),
        s2_loss=(
            s2_loss_sum
            / example_count
        ),
        s1_accuracy=float(
            s1_correct_by_step.sum().item()
            / token_count_by_step
            .sum()
            .clamp_min(1)
            .item()
        ),
        s2_accuracy=s2_accuracy,
        graph_mean_row_entropy=(
            mean_graph_entropy
        ),
        graph_mean_effective_neighbours=(
            mean_effective_neighbours
        ),
        graph_entropy_penalty=(
            graph_entropy_penalty_sum
            / example_count
        ),
        graph_target_entropy_penalty=(
            graph_target_entropy_penalty_sum
            / example_count
        ),
        graph_temporal_smooth_penalty=(
            graph_temporal_smooth_penalty_sum
            / example_count
        ),
        graph_warmup_scale=(
            graph_warmup_scale_sum
            / example_count
        ),
        graph_valid_smoothing_pairs=(
            graph_valid_smoothing_pairs
        ),
        spatial_beta=(
            spatial_beta_sum / spatial_beta_examples
            if spatial_beta_examples
            else None
        ),
        s1_loss_by_step=(
            s1_loss_by_step_sum
            / example_count
        ),
        s2_loss_by_step=(
            s2_loss_by_step_sum
            / example_count
        ),
        s1_accuracy_by_step=(
            s1_accuracy_by_step
        ),
        s2_accuracy_by_step=(
            s2_accuracy_by_step
        ),
        examples=example_count,
        seconds=(
            perf_counter()
            - start
        ),
    )

def average_decoded_paths(decoded_paths: Tensor) -> Tensor:
    """Average complete decoded paths in continuous OHLCV space.

    Token IDs, bit codes and logits are deliberately never averaged.
    """
    values = torch.as_tensor(decoded_paths)
    if values.ndim != 5 or int(values.shape[-1]) != len(OHLCV_CHANNELS):
        raise ValueError(
            "decoded_paths must have shape [S, B, P, N, 5]."
        )
    if int(values.shape[0]) <= 0:
        raise ValueError("decoded_paths must contain at least one path.")
    return values.to(torch.float32).mean(dim=0).contiguous()


def _invalid_candle_mask(decoded_ohlcv: Tensor) -> Tensor:
    if decoded_ohlcv.ndim != 4 or int(decoded_ohlcv.shape[-1]) != 5:
        raise ValueError("decoded_ohlcv must have shape [B, P, N, 5].")

    open_values = decoded_ohlcv[..., 0]
    high_values = decoded_ohlcv[..., 1]
    low_values = decoded_ohlcv[..., 2]
    close_values = decoded_ohlcv[..., 3]
    volume_values = decoded_ohlcv[..., 4]

    return (
        ~torch.isfinite(decoded_ohlcv).all(dim=-1)
        | (open_values <= 0)
        | (high_values <= 0)
        | (low_values <= 0)
        | (close_values <= 0)
        | (high_values < torch.maximum(open_values, close_values))
        | (low_values > torch.minimum(open_values, close_values))
        | (high_values < low_values)
        | (volume_values < 0)
    )


def _append_cpu(values: list[Tensor], tensor: Any, *, dtype: torch.dtype) -> None:
    values.append(torch.as_tensor(tensor).detach().cpu().to(dtype).contiguous())


def generate_validation_artifacts(
    *,
    model: DynamicGraphTokenForecaster,
    loader: DataLoader[dict[str, Any]],
    dataset: CachedTokenGraphDataset,
    device: torch.device,
    use_amp: bool,
    decoding_config: Mapping[str, Any],
    tokenizer: KronosTokenizerAdapter | None,
    raw_train_split: Mapping[str, Any] | None,
    decode_series_batch_size: int,
    early_stopping_horizons: Sequence[int],
) -> ValidationBundle:
    """Generate, decode and evaluate deterministic or Monte Carlo paths.

    Standard training/checkpoint validation uses ``argmax`` with one path.
    The post-training temperature sweep uses ten complete sampled coarse-token
    paths. Every discrete path is decoded independently and only then averaged
    in raw continuous OHLCV space.
    """
    model.eval()
    synchronise_device(device)
    start = perf_counter()

    prediction_length = model.config.prediction_length
    predict_s2 = model.config.heads.predicts_s2
    s1_correct_by_step = torch.zeros(prediction_length, dtype=torch.float64)
    s2_correct_by_step = torch.zeros(prediction_length, dtype=torch.float64)
    token_count_by_step = torch.zeros(prediction_length, dtype=torch.float64)
    example_count = 0

    graph_accumulator = GraphArtifactAccumulator(
        asset_cols=dataset.asset_cols,
        graph_type=model.config.graph.type,
        num_layers=model.config.num_st_blocks,
        num_heads=model.config.graph.num_heads,
    )

    y_pred_parts: list[Tensor] = []
    y_true_parts: list[Tensor] = []
    last_context_parts: list[Tensor] = []
    sample_idx_parts: list[Tensor] = []
    origin_idx_parts: list[Tensor] = []
    target_indices_parts: list[Tensor] = []
    generated_s1_parts: list[Tensor] = []
    generated_s2_parts: list[Tensor] = []
    sampled_s1_evaluation_parts: list[Tensor] = []
    sampled_close_path_parts: list[Tensor] = []
    target_s1_parts: list[Tensor] = []
    target_s2_parts: list[Tensor] = []

    # Final ensemble diagnostics.
    invalid_dense_count = 0
    invalid_dense_total = 0
    invalid_evaluation_count = 0
    invalid_evaluation_total = 0
    nonpositive_close_count = 0

    # Individual sampled-path diagnostics. For one-path argmax these equal
    # the final-forecast diagnostics.
    sampled_invalid_dense_count = 0
    sampled_invalid_dense_total = 0
    sampled_invalid_evaluation_count = 0
    sampled_invalid_evaluation_total = 0

    evaluation_indices = torch.tensor(
        model.config.evaluation_indices,
        dtype=torch.long,
    )

    token_selection = str(decoding_config["token_selection"])
    temperature = float(decoding_config["temperature"])
    top_k = int(decoding_config["top_k"])
    top_p = float(decoding_config["top_p"])
    sample_count = int(decoding_config.get("sample_count", 1))

    if sample_count <= 0:
        raise ValueError("decoding.sample_count must be positive.")
    if token_selection == "argmax" and sample_count != 1:
        raise ValueError("Argmax validation requires sample_count=1.")
    if sample_count > 1 and predict_s2:
        raise ValueError(
            "Multi-path decoded averaging is intentionally coarse-only; "
            "future s2 sampling is disabled."
        )

    progress = tqdm(
        loader,
        desc=(
            "validation generation"
            if sample_count == 1
            else f"validation generation ({sample_count} sampled paths)"
        ),
        leave=False,
        dynamic_ncols=True,
    )

    with torch.inference_mode():
        for batch in progress:
            context_tokens, target_s1, target_s2 = move_training_batch(
                batch,
                device=device,
            )

            with _autocast_context(device, use_amp):
                if sample_count == 1:
                    generated_single: GeneratedTokenForecast = model.generate(
                        context_tokens,
                        token_selection=token_selection,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    sampled_tokens_device = (
                        generated_single.token_ids.unsqueeze(0)
                    )
                    forecast = generated_single.forecast
                else:
                    generated_multi: SampledGeneratedTokenForecast = (
                        model.generate_samples(
                            context_tokens,
                            sample_count=sample_count,
                            token_selection=token_selection,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        )
                    )
                    sampled_tokens_device = generated_multi.token_ids
                    forecast = generated_multi.forecast

            batch_size = int(context_tokens.shape[0])
            num_nodes = int(context_tokens.shape[2])
            sampled_tokens = (
                sampled_tokens_device.detach().cpu().to(torch.long)
            )
            if tuple(sampled_tokens.shape[:2]) != (
                sample_count,
                batch_size,
            ):
                raise RuntimeError("Generated sample and batch axes differ.")

            # Keep one representative path for backward-compatible saved
            # token artefacts, plus all sampled IDs only at the five reported
            # horizons (small enough for diagnostics).
            representative_tokens = sampled_tokens[0]
            generated_s1_parts.append(
                representative_tokens[..., 0].to(torch.int16).contiguous()
            )
            if predict_s2:
                generated_s2_parts.append(
                    representative_tokens[..., 1].to(torch.int16).contiguous()
                )
            sampled_s1_evaluation_parts.append(
                sampled_tokens[..., 0]
                .index_select(dim=2, index=evaluation_indices)
                .to(torch.int16)
                .contiguous()
            )

            target_s1_cpu = target_s1.detach().cpu().to(torch.long)
            target_s2_cpu = target_s2.detach().cpu().to(torch.long)
            target_s1_parts.append(target_s1_cpu.to(torch.int16).contiguous())
            target_s2_parts.append(target_s2_cpu.to(torch.int16).contiguous())

            s1_correct_by_step += (
                (sampled_tokens[..., 0] == target_s1_cpu.unsqueeze(0))
                .sum(dim=(0, 1, 3))
                .to(torch.float64)
            )
            if predict_s2:
                s2_correct_by_step += (
                    (sampled_tokens[..., 1] == target_s2_cpu.unsqueeze(0))
                    .sum(dim=(0, 1, 3))
                    .to(torch.float64)
                )
            token_count_by_step += float(sample_count * batch_size * num_nodes)
            example_count += batch_size

            graph_accumulator.add(
                forecast.graph,
                batch,
                batch_size=batch_size,
                spatial_beta=forecast.spatial_beta,
            )

            if dataset.data_mode != "real":
                continue

            if tokenizer is None or raw_train_split is None:
                raise RuntimeError(
                    "Real-data validation requires a loaded tokenizer and "
                    "raw training split."
                )

            context_tokens_for_decode = (
                context_tokens.detach().cpu().to(torch.long).clone()
            )
            context_tokens_for_decode[..., 0] = dataset.s1_to_kronos_ids(
                context_tokens_for_decode[..., 0]
            )
            sampled_s1_for_decode = dataset.s1_to_kronos_ids(
                sampled_tokens[..., 0]
            )
            context_mean = torch.as_tensor(batch["context_mean"])
            context_std = torch.as_tensor(batch["context_std"])

            if sample_count == 1:
                if predict_s2:
                    tokens_for_decode = representative_tokens.clone()
                    tokens_for_decode[..., 0] = sampled_s1_for_decode[0]
                    decoded_paths = tokenizer.decode_token_path(
                        context_tokens_for_decode,
                        tokens_for_decode,
                        mean=context_mean,
                        std=context_std,
                        series_batch_size=decode_series_batch_size,
                        return_full_path=False,
                    ).unsqueeze(0)
                else:
                    decoded_paths = tokenizer.decode_coarse_token_path(
                        context_tokens_for_decode,
                        sampled_s1_for_decode[0],
                        mean=context_mean,
                        std=context_std,
                        series_batch_size=decode_series_batch_size,
                        return_full_path=False,
                    ).unsqueeze(0)
            else:
                repeated_context = (
                    context_tokens_for_decode.unsqueeze(0)
                    .expand(sample_count, -1, -1, -1, -1)
                    .reshape(
                        sample_count * batch_size,
                        model.config.context_length,
                        num_nodes,
                        2,
                    )
                    .contiguous()
                )
                repeated_mean = (
                    context_mean.unsqueeze(0)
                    .expand(sample_count, -1, -1, -1)
                    .reshape(sample_count * batch_size, num_nodes, -1)
                    .contiguous()
                )
                repeated_std = (
                    context_std.unsqueeze(0)
                    .expand(sample_count, -1, -1, -1)
                    .reshape(sample_count * batch_size, num_nodes, -1)
                    .contiguous()
                )
                flat_future_s1 = sampled_s1_for_decode.reshape(
                    sample_count * batch_size,
                    prediction_length,
                    num_nodes,
                ).contiguous()
                decoded_flat = tokenizer.decode_coarse_token_path(
                    repeated_context,
                    flat_future_s1,
                    mean=repeated_mean,
                    std=repeated_std,
                    series_batch_size=decode_series_batch_size,
                    return_full_path=False,
                )
                decoded_paths = decoded_flat.reshape(
                    sample_count,
                    batch_size,
                    prediction_length,
                    num_nodes,
                    len(OHLCV_CHANNELS),
                )

            decoded_paths = decoded_paths.to(torch.float32)

            # Retain every decoded raw Close trajectory for downstream
            # predictive-distribution analysis.  The stored tensor is
            # close-only to keep the five-temperature, ten-path sweep
            # tractable on Drive while preserving the full 60-minute path.
            sampled_close_path_parts.append(
                decoded_paths[
                    ...,
                    CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
                ]
                .detach()
                .cpu()
                .contiguous()
            )

            sampled_invalid_dense = _invalid_candle_mask(
                decoded_paths.reshape(
                    sample_count * batch_size,
                    prediction_length,
                    num_nodes,
                    len(OHLCV_CHANNELS),
                )
            ).reshape(sample_count, batch_size, prediction_length, num_nodes)
            sampled_invalid_dense_count += int(
                sampled_invalid_dense.sum().item()
            )
            sampled_invalid_dense_total += int(
                sampled_invalid_dense.numel()
            )
            sampled_invalid_evaluation = sampled_invalid_dense.index_select(
                dim=2,
                index=evaluation_indices,
            )
            sampled_invalid_evaluation_count += int(
                sampled_invalid_evaluation.sum().item()
            )
            sampled_invalid_evaluation_total += int(
                sampled_invalid_evaluation.numel()
            )

            # This is the central Monte Carlo contract: average only after
            # each full categorical path has been decoded into continuous
            # raw OHLCV values.
            decoded_future = average_decoded_paths(decoded_paths)
            invalid_dense = _invalid_candle_mask(decoded_future)
            invalid_dense_count += int(invalid_dense.sum().item())
            invalid_dense_total += int(invalid_dense.numel())

            decoded_evaluation = decoded_future.index_select(
                dim=1,
                index=evaluation_indices,
            )
            invalid_evaluation = invalid_dense.index_select(
                dim=1,
                index=evaluation_indices,
            )
            invalid_evaluation_count += int(invalid_evaluation.sum().item())
            invalid_evaluation_total += int(invalid_evaluation.numel())

            y_pred = decoded_evaluation[
                ...,
                CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
            ]
            y_true = torch.as_tensor(batch["evaluation_true"]).to(
                torch.float32
            )[
                ...,
                CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
            ]
            last_context = torch.as_tensor(
                batch["last_context_target"]
            ).to(torch.float32)[
                ...,
                CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
            ]

            nonpositive_close_count += int((y_pred <= 0).sum().item())
            y_pred_parts.append(y_pred.contiguous())
            y_true_parts.append(y_true.contiguous())
            last_context_parts.append(last_context.contiguous())
            _append_cpu(sample_idx_parts, batch["sample_idx"], dtype=torch.long)
            _append_cpu(origin_idx_parts, batch["origin_idx"], dtype=torch.long)
            dense_target_indices = torch.as_tensor(batch["target_indices"]).to(
                torch.long
            )
            target_indices_parts.append(
                dense_target_indices.index_select(
                    dim=1,
                    index=evaluation_indices,
                ).contiguous()
            )

    synchronise_device(device)
    if example_count == 0:
        raise RuntimeError("Validation DataLoader yielded no examples.")

    s1_accuracy_by_step = s1_correct_by_step / token_count_by_step.clamp_min(1)
    if predict_s2:
        s2_accuracy_by_step = s2_correct_by_step / token_count_by_step.clamp_min(1)
        generated_s2_accuracy = float(
            s2_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        )
    else:
        s2_accuracy_by_step = torch.full(
            (prediction_length,),
            float("nan"),
            dtype=torch.float64,
        )
        generated_s2_accuracy = float("nan")

    generation_metrics = GenerationMetrics(
        s1_accuracy=float(
            s1_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        ),
        s2_accuracy=generated_s2_accuracy,
        s1_accuracy_by_step=s1_accuracy_by_step,
        s2_accuracy_by_step=s2_accuracy_by_step,
        examples=example_count,
        seconds=perf_counter() - start,
    )
    graph_artifacts = graph_accumulator.finalise()

    generated_s1 = torch.cat(generated_s1_parts, dim=0).contiguous()
    target_s1_all = torch.cat(target_s1_parts, dim=0).contiguous()
    target_s2_all = torch.cat(target_s2_parts, dim=0).contiguous()
    generated_s2 = (
        torch.cat(generated_s2_parts, dim=0).contiguous()
        if predict_s2
        else None
    )
    # [S, batch_1, H, N] parts -> [S, all_windows, H, N]
    sampled_s1_evaluation = torch.cat(
        sampled_s1_evaluation_parts,
        dim=1,
    ).contiguous()

    expected_token_shape = (
        example_count,
        prediction_length,
        len(dataset.asset_cols),
    )
    for name, values in {
        "generated_s1": generated_s1,
        "target_s1": target_s1_all,
        "target_s2": target_s2_all,
    }.items():
        if tuple(values.shape) != expected_token_shape:
            raise ValueError(
                f"{name} has shape {tuple(values.shape)}, "
                f"expected {expected_token_shape}."
            )
    if generated_s2 is not None and tuple(generated_s2.shape) != expected_token_shape:
        raise ValueError(
            f"generated_s2 has shape {tuple(generated_s2.shape)}, "
            f"expected {expected_token_shape}."
        )
    expected_sampled_eval = (
        sample_count,
        example_count,
        len(model.config.heads.evaluation_horizons),
        len(dataset.asset_cols),
    )
    if tuple(sampled_s1_evaluation.shape) != expected_sampled_eval:
        raise ValueError(
            "sampled_s1_evaluation has shape "
            f"{tuple(sampled_s1_evaluation.shape)}, expected "
            f"{expected_sampled_eval}."
        )

    sampled_price_path_artifacts: dict[str, Any] | None = None

    if dataset.data_mode == "real":
        if not sampled_close_path_parts:
            raise RuntimeError(
                "Real-data validation retained no decoded sampled paths."
            )

        # [S, batch_1, P, N, 1] parts -> [S, all_windows, P, N, 1]
        sampled_close_paths = torch.cat(
            sampled_close_path_parts,
            dim=1,
        ).contiguous()
        expected_sampled_close_shape = (
            sample_count,
            example_count,
            prediction_length,
            len(dataset.asset_cols),
            1,
        )
        if tuple(sampled_close_paths.shape) != expected_sampled_close_shape:
            raise ValueError(
                "sampled_close_paths has shape "
                f"{tuple(sampled_close_paths.shape)}, expected "
                f"{expected_sampled_close_shape}."
            )
        if not torch.isfinite(sampled_close_paths).all():
            raise ValueError("sampled_close_paths contains non-finite values.")

        sampled_close_evaluation = sampled_close_paths.index_select(
            dim=2,
            index=evaluation_indices,
        ).contiguous()
        ensemble_mean_close_path = sampled_close_paths.mean(
            dim=0,
        ).contiguous()

        sampled_price_path_artifacts = {
            "sampled_close_paths": sampled_close_paths,
            "sampled_close_paths_at_evaluation_horizons": (
                sampled_close_evaluation
            ),
            "ensemble_mean_close_path": ensemble_mean_close_path,
            "evaluation_true": torch.cat(
                y_true_parts,
                dim=0,
            ).contiguous(),
            "last_context_target": torch.cat(
                last_context_parts,
                dim=0,
            ).contiguous(),
            "sample_idx": graph_artifacts.get("sample_idx"),
            "origin_idx": graph_artifacts.get("origin_idx"),
            "dense_target_indices": graph_artifacts.get("target_indices"),
            "evaluation_target_indices": torch.cat(
                target_indices_parts,
                dim=0,
            ).contiguous(),
            "dates": graph_artifacts.get("dates"),
            "asset_cols": list(dataset.asset_cols),
            "future_steps": list(range(1, prediction_length + 1)),
            "evaluation_horizons": [
                int(value)
                for value in model.config.heads.evaluation_horizons
            ],
            "token_selection": token_selection,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "sample_count": sample_count,
            "output_space": "raw",
            "channel": "close",
            "path_dtype": str(sampled_close_paths.dtype),
            "graph_shared_across_sample_paths": True,
            "averaging_space": "decoded raw continuous Close",
        }

    token_artifacts: dict[str, Any] = {
        "generated_s1": generated_s1,
        "generated_s2": generated_s2,
        "sampled_s1_evaluation": sampled_s1_evaluation,
        "target_s1": target_s1_all,
        "target_s2": target_s2_all,
        "sample_idx": graph_artifacts.get("sample_idx"),
        "origin_idx": graph_artifacts.get("origin_idx"),
        "target_indices": graph_artifacts.get("target_indices"),
        "window_idx": graph_artifacts.get("window_idx"),
        "trajectory_id": graph_artifacts.get("trajectory_id"),
        "regime_id": graph_artifacts.get("regime_id"),
        "dates": graph_artifacts.get("dates"),
        "asset_cols": list(dataset.asset_cols),
        "prediction_length": int(prediction_length),
        "future_token_mode": model.config.heads.future_token_mode,
        "s2_conditioning": model.config.heads.resolved_s2_conditioning,
        "token_selection": token_selection,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "sample_count": sample_count,
        "token_dtype": "int16",
    }

    diagnostics: dict[str, Any] = {
        "future_token_mode": model.config.heads.future_token_mode,
        "s2_loss_weight": float(model.config.heads.s2_loss_weight),
        "token_selection": token_selection,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "sample_count": sample_count,
        "generated_s1_accuracy": generation_metrics.s1_accuracy,
        "generated_s2_accuracy": generation_metrics.s2_accuracy,
        "generation_seconds": generation_metrics.seconds,
        "validation_examples": generation_metrics.examples,
    }

    if dataset.data_mode != "real":
        return ValidationBundle(
            generation_metrics=generation_metrics,
            token_artifacts=token_artifacts,
            graph_artifacts=graph_artifacts,
            sampled_price_path_artifacts=None,
            prediction_result=None,
            metric_results=None,
            metric_table=None,
            primary_score=None,
            diagnostics=diagnostics,
        )

    prediction_result: PredictionResult = {
        "y_pred": torch.cat(y_pred_parts, dim=0).contiguous(),
        "y_true": torch.cat(y_true_parts, dim=0).contiguous(),
        "last_context_target": torch.cat(last_context_parts, dim=0).contiguous(),
        "sample_idx": torch.cat(sample_idx_parts, dim=0).contiguous(),
        "origin_idx": torch.cat(origin_idx_parts, dim=0).contiguous(),
        "target_indices": torch.cat(target_indices_parts, dim=0).contiguous(),
        "channels": ["close"],
        "horizons": list(model.config.heads.evaluation_horizons),
        "asset_cols": list(dataset.asset_cols),
        "output_space": "raw",
    }

    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
        train_split=dict(raw_train_split),
    )
    metric_results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
        bootstrap=False,
    )
    metric_table = make_evaluation_table(
        metric_results=metric_results,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )
    primary_score = _mean_cumulative_log_change_mae_at_horizons(
        metric_results=metric_results,
        available_horizons=evaluator.horizons,
        selected_horizons=early_stopping_horizons,
    )

    diagnostics.update(
        {
            "primary_metric": "mean_validation_cumulative_log_change_mae",
            "primary_horizons": [int(h) for h in early_stopping_horizons],
            "primary_score": primary_score,
            "ensemble_invalid_dense_candle_rate_percent": (
                100.0 * invalid_dense_count / max(invalid_dense_total, 1)
            ),
            "ensemble_invalid_evaluation_candle_rate_percent": (
                100.0
                * invalid_evaluation_count
                / max(invalid_evaluation_total, 1)
            ),
            # Backward-compatible names now describe the final ensemble.
            "invalid_dense_candle_rate_percent": (
                100.0 * invalid_dense_count / max(invalid_dense_total, 1)
            ),
            "invalid_evaluation_candle_rate_percent": (
                100.0
                * invalid_evaluation_count
                / max(invalid_evaluation_total, 1)
            ),
            "sample_path_invalid_dense_candle_rate_percent": (
                100.0
                * sampled_invalid_dense_count
                / max(sampled_invalid_dense_total, 1)
            ),
            "sample_path_invalid_evaluation_candle_rate_percent": (
                100.0
                * sampled_invalid_evaluation_count
                / max(sampled_invalid_evaluation_total, 1)
            ),
            "nonpositive_predicted_close_count": nonpositive_close_count,
        }
    )

    return ValidationBundle(
        generation_metrics=generation_metrics,
        token_artifacts=token_artifacts,
        graph_artifacts=graph_artifacts,
        sampled_price_path_artifacts=sampled_price_path_artifacts,
        prediction_result=prediction_result,
        metric_results=metric_results,
        metric_table=metric_table,
        primary_score=primary_score,
        diagnostics=diagnostics,
    )

def token_metric_table(
    *,
    model: DynamicGraphTokenForecaster,
    teacher_forced: TeacherForcedEpochMetrics,
    generated: GenerationMetrics,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluation_horizons = set(model.config.heads.evaluation_horizons)

    for step_index in range(model.config.prediction_length):
        horizon = step_index + 1
        rows.append(
            {
                "future_step": horizon,
                "is_evaluation_horizon": horizon in evaluation_horizons,
                "teacher_forced_s1_ce": float(
                    teacher_forced.s1_loss_by_step[step_index].item()
                ),
                "teacher_forced_s2_ce": float(
                    teacher_forced.s2_loss_by_step[step_index].item()
                ),
                "teacher_forced_s1_accuracy": float(
                    teacher_forced.s1_accuracy_by_step[step_index].item()
                ),
                "teacher_forced_s2_accuracy": float(
                    teacher_forced.s2_accuracy_by_step[step_index].item()
                ),
                "generated_s1_accuracy": float(
                    generated.s1_accuracy_by_step[step_index].item()
                ),
                "generated_s2_accuracy": float(
                    generated.s2_accuracy_by_step[step_index].item()
                ),
            }
        )

    return pd.DataFrame(rows)


def graph_summary(graph_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    selected = graph_artifacts.get("selected")

    if selected is None:
        return {
            "graph_present": False,
            "mean_row_entropy": None,
            "mean_effective_neighbours": None,
            "mean_nonzero_sources": None,
            "spatial_beta": None,
        }

    selected = torch.as_tensor(selected).to(torch.float64)
    positive = selected > 0
    entropy = -(
        torch.where(
            positive,
            selected * selected.clamp_min(1.0e-12).log(),
            torch.zeros_like(selected),
        )
    ).sum(dim=-1)

    spatial_beta = graph_artifacts.get("spatial_beta")
    beta_value = (
        None
        if spatial_beta is None
        else float(torch.as_tensor(spatial_beta).float().mean().item())
    )
    return {
        "graph_present": True,
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_nonzero_sources": float(
            positive.sum(dim=-1).to(torch.float64).mean().item()
        ),
        "spatial_beta": beta_value,
    }


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def build_checkpoint(
    *,
    model: DynamicGraphTokenForecaster,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    evaluations_without_improvement: int,
    history: list[dict[str, Any]],
    secondary_selection_state: Mapping[str, Mapping[str, Any]],
    run_signature: str,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(
            evaluations_without_improvement
        ),
        "history": list(history),
        "secondary_selection_state": deepcopy(
            dict(secondary_selection_state)
        ),
        "rng_state": capture_rng_state(),
        "run_signature": run_signature,
        "resolved_config": deepcopy(dict(resolved_config)),
        "run_metadata": deepcopy(dict(run_metadata)),
    }


def load_checkpoint(
    path: Path,
    *,
    model: DynamicGraphTokenForecaster,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    expected_signature: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if checkpoint.get("run_signature") != expected_signature:
        raise ValueError(
            "Checkpoint run signature differs from the requested run."
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _flatten_metric_results_for_logging(
    metric_results: Mapping[str, Tensor] | None,
    horizons: Sequence[int],
) -> dict[str, float]:
    if metric_results is None:
        return {}

    values: dict[str, float] = {}

    for metric_name, metric_tensor in metric_results.items():
        flat = metric_tensor.detach().cpu().reshape(-1)

        if flat.numel() != len(horizons):
            continue

        for horizon, value in zip(horizons, flat, strict=True):
            values[f"val/{metric_name}/h{horizon}"] = float(value.item())

    return values


def _save_best_validation_artifacts(
    *,
    run_dir: Path,
    epoch: int,
    bundle: ValidationBundle,
    teacher_forced: TeacherForcedEpochMetrics,
    model: DynamicGraphTokenForecaster,
) -> None:
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "prediction_result": bundle.prediction_result,
            "metric_results": bundle.metric_results,
            "diagnostics": bundle.diagnostics,
        },
        run_dir / "best_validation_predictions.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "graph_artifacts": bundle.graph_artifacts,
            "summary": graph_summary(bundle.graph_artifacts),
        },
        run_dir / "best_validation_graphs.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "token_artifacts": bundle.token_artifacts,
        },
        run_dir / "best_validation_tokens.pt",
    )
    if bundle.sampled_price_path_artifacts is not None:
        atomic_torch_save(
            {
                "epoch": int(epoch),
                "sampled_price_path_artifacts": (
                    bundle.sampled_price_path_artifacts
                ),
            },
            run_dir / "best_validation_sampled_price_paths.pt",
        )

    if bundle.metric_table is not None:
        atomic_csv_save(
            bundle.metric_table,
            run_dir / "best_validation_metric_table.csv",
        )

    atomic_csv_save(
        token_metric_table(
            model=model,
            teacher_forced=teacher_forced,
            generated=bundle.generation_metrics,
        ),
        run_dir / "best_validation_token_metrics.csv",
    )

    diagnostics = dict(bundle.diagnostics)
    diagnostics["epoch"] = int(epoch)
    diagnostics["graph_summary"] = graph_summary(bundle.graph_artifacts)
    diagnostics["validation_objective_loss"] = (
        teacher_forced.total_loss
    )
    diagnostics["validation_token_loss"] = (
        teacher_forced.token_loss
    )
    diagnostics["validation_graph_regularisation_loss"] = (
        teacher_forced.graph_regularisation_loss
    )
    diagnostics["validation_graph_mean_row_entropy"] = (
        teacher_forced.graph_mean_row_entropy
    )
    diagnostics["validation_graph_mean_effective_neighbours"] = (
        teacher_forced.graph_mean_effective_neighbours
    )
    diagnostics["validation_spatial_beta"] = teacher_forced.spatial_beta
    atomic_json_save(
        diagnostics,
        run_dir / "best_validation_diagnostics.json",
    )


def _temperature_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _save_inference_bundle(
    *,
    output_dir: Path,
    label: str,
    epoch: int,
    bundle: ValidationBundle,
) -> None:
    """Save one inference policy without overwriting checkpoint artefacts."""
    policy_dir = output_dir / label
    policy_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "prediction_result": bundle.prediction_result,
            "metric_results": bundle.metric_results,
            "diagnostics": bundle.diagnostics,
        },
        policy_dir / "validation_predictions.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "graph_artifacts": bundle.graph_artifacts,
            "summary": graph_summary(bundle.graph_artifacts),
        },
        policy_dir / "validation_graphs.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "token_artifacts": bundle.token_artifacts,
        },
        policy_dir / "validation_tokens.pt",
    )
    if bundle.sampled_price_path_artifacts is not None:
        atomic_torch_save(
            {
                "epoch": int(epoch),
                "sampled_price_path_artifacts": (
                    bundle.sampled_price_path_artifacts
                ),
            },
            policy_dir / "validation_sampled_price_paths.pt",
        )
    if bundle.metric_table is not None:
        atomic_csv_save(
            bundle.metric_table,
            policy_dir / "validation_metric_table.csv",
        )
    diagnostics = dict(bundle.diagnostics)
    diagnostics["epoch"] = int(epoch)
    diagnostics["graph_summary"] = graph_summary(bundle.graph_artifacts)
    atomic_json_save(
        diagnostics,
        policy_dir / "validation_diagnostics.json",
    )


def _temperature_result_row(
    *,
    label: str,
    temperature: float | None,
    sample_count: int,
    bundle: ValidationBundle,
    horizons: Sequence[int],
) -> dict[str, Any]:
    if bundle.metric_results is None or bundle.primary_score is None:
        raise RuntimeError("Temperature sweep requires real decoded metrics.")
    metric = torch.as_tensor(
        bundle.metric_results["cumulative_log_change_mae"]
    ).detach().cpu().reshape(-1)
    if metric.numel() != len(horizons):
        raise ValueError("Unexpected cumulative-log-change MAE shape.")
    row: dict[str, Any] = {
        "Policy": label,
        "Temperature": temperature,
        "Sample count": int(sample_count),
        "Mean Log MAE": float(bundle.primary_score),
        "Generation seconds": float(bundle.generation_metrics.seconds),
        "Generated s1 accuracy": float(bundle.generation_metrics.s1_accuracy),
        "Invalid ensemble candles (%)": float(
            bundle.diagnostics.get(
                "ensemble_invalid_dense_candle_rate_percent",
                bundle.diagnostics.get("invalid_dense_candle_rate_percent", float("nan")),
            )
        ),
        "Invalid sampled-path candles (%)": float(
            bundle.diagnostics.get(
                "sample_path_invalid_dense_candle_rate_percent",
                float("nan"),
            )
        ),
    }
    for horizon, value in zip(horizons, metric, strict=True):
        row[f"Log MAE — {int(horizon)} min"] = float(value.item())
    return row


def run_temperature_sweep(
    *,
    model: DynamicGraphTokenForecaster,
    loader: DataLoader[dict[str, Any]],
    dataset: CachedTokenGraphDataset,
    device: torch.device,
    use_amp: bool,
    tokenizer: KronosTokenizerAdapter,
    raw_train_split: Mapping[str, Any],
    decode_series_batch_size: int,
    temperature_config: Mapping[str, Any],
    output_dir: Path,
    checkpoint_epoch: int,
    evaluation_horizons: Sequence[int],
) -> pd.DataFrame:
    """Run deterministic argmax plus the inference-only 10-path sweep."""
    if model.config.future_predictor.type != "structured_parallel":
        raise ValueError("Temperature sweep requires structured_parallel.")
    if model.config.heads.future_token_mode != "coarse_only":
        raise ValueError("Temperature sweep requires coarse_only output.")

    raw_temperatures = temperature_config.get("temperatures")
    if not isinstance(raw_temperatures, (list, tuple)) or not raw_temperatures:
        raise ValueError("temperature_sweep.temperatures must be non-empty.")
    temperatures = tuple(float(value) for value in raw_temperatures)
    if any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
        raise ValueError("Every inference temperature must be finite and positive.")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("Temperature sweep values must be unique.")

    sample_count = int(temperature_config.get("sample_count", 10))
    top_k = int(temperature_config.get("top_k", 0))
    top_p = float(temperature_config.get("top_p", 0.9))
    sampling_seed = int(temperature_config.get("seed", 42))
    if sample_count <= 0:
        raise ValueError("temperature_sweep.sample_count must be positive.")
    if top_k < 0:
        raise ValueError("temperature_sweep.top_k cannot be negative.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("temperature_sweep.top_p must lie in (0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Argmax is a deterministic reference and is not eligible to define the
    # selected stochastic temperature.
    set_seed(sampling_seed)
    argmax_bundle = generate_validation_artifacts(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        use_amp=use_amp,
        decoding_config={
            "token_selection": "argmax",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "sample_count": 1,
        },
        tokenizer=tokenizer,
        raw_train_split=raw_train_split,
        decode_series_batch_size=decode_series_batch_size,
        early_stopping_horizons=evaluation_horizons,
    )
    _save_inference_bundle(
        output_dir=output_dir,
        label="argmax",
        epoch=checkpoint_epoch,
        bundle=argmax_bundle,
    )
    rows.append(
        _temperature_result_row(
            label="argmax",
            temperature=None,
            sample_count=1,
            bundle=argmax_bundle,
            horizons=evaluation_horizons,
        )
    )

    for temperature in temperatures:
        # Resetting the same seed gives every temperature the same RNG stream,
        # reducing Monte Carlo noise in the validation comparison.
        set_seed(sampling_seed)
        bundle = generate_validation_artifacts(
            model=model,
            loader=loader,
            dataset=dataset,
            device=device,
            use_amp=use_amp,
            decoding_config={
                "token_selection": "sample",
                "temperature": float(temperature),
                "top_k": top_k,
                "top_p": top_p,
                "sample_count": sample_count,
            },
            tokenizer=tokenizer,
            raw_train_split=raw_train_split,
            decode_series_batch_size=decode_series_batch_size,
            early_stopping_horizons=evaluation_horizons,
        )
        label = f"temperature_{_temperature_tag(temperature)}"
        _save_inference_bundle(
            output_dir=output_dir,
            label=label,
            epoch=checkpoint_epoch,
            bundle=bundle,
        )
        rows.append(
            _temperature_result_row(
                label=label,
                temperature=float(temperature),
                sample_count=sample_count,
                bundle=bundle,
                horizons=evaluation_horizons,
            )
        )

    results = pd.DataFrame(rows)
    atomic_csv_save(results, output_dir / "temperature_sweep_results.csv")
    sampled_results = results.loc[results["Temperature"].notna()].copy()
    winner = sampled_results.sort_values(
        ["Mean Log MAE", "Temperature"],
        ascending=[True, True],
    ).iloc[0]
    atomic_json_save(
        {
            "checkpoint_epoch": int(checkpoint_epoch),
            "selection_split": "September validation",
            "selection_metric": (
                "strict all-five-horizon mean cumulative-log-change MAE"
            ),
            "selected_temperature": float(winner["Temperature"]),
            "selected_policy": str(winner["Policy"]),
            "selected_score": float(winner["Mean Log MAE"]),
            "sample_count": sample_count,
            "top_k": top_k,
            "top_p": top_p,
            "sampling_seed": sampling_seed,
            "temperatures": list(temperatures),
            "temperature_affects_training": False,
            "averaging_space": "decoded raw continuous OHLCV",
        },
        output_dir / "temperature_selection.json",
    )
    return results.sort_values(
        ["Mean Log MAE", "Policy"],
        ascending=[True, True],
    ).reset_index(drop=True)

def _init_wandb(
    *,
    args: argparse.Namespace,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
):
    if args.wandb_mode == "disabled":
        return None

    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        tags=list(args.wandb_tags),
        config={
            "experiment": deepcopy(dict(resolved_config)),
            "runtime": deepcopy(dict(run_metadata)),
        },
    )


def _validate_cache_against_model(
    *,
    dataset: CachedTokenGraphDataset,
    model: DynamicGraphTokenForecaster,
    data_mode: str,
) -> None:
    if dataset.data_mode != data_mode:
        raise ValueError(
            f"Config data.mode={data_mode!r}, but the cache is "
            f"{dataset.data_mode!r}."
        )

    checks = {
        "context_length": (dataset.context_length, model.config.context_length),
        "prediction_length": (
            dataset.prediction_length,
            model.config.prediction_length,
        ),
        "num_assets": (dataset.num_assets, model.config.num_nodes),
        "s1_vocabulary_size": (
            dataset.s1_vocabulary_size,
            model.config.heads.s1_vocabulary_size,
        ),
    }

    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(
                f"Cache/model mismatch for {name}: {observed} versus {expected}."
            )

    if model.config.temporal.type == "modern_tcn":
        if dataset.s1_id_space != "kronos_original":
            raise ValueError(
                "The token ModernTCN experiment requires original Kronos "
                "s1 IDs. Remapped 150/250-token caches cannot reconstruct "
                "the exact post-BSQ 20-bit encoder code."
            )
        if dataset.s1_vocabulary_size != 1024:
            raise ValueError(
                "The token ModernTCN experiment requires the full 1024-way "
                "coarse vocabulary."
            )

    if data_mode == "real":
        if dataset.evaluation_horizons != tuple(
            model.config.heads.evaluation_horizons
        ):
            raise ValueError("Cache and model evaluation horizons differ.")

        if dataset.evaluation_indices != tuple(model.config.evaluation_indices):
            raise ValueError("Cache and model evaluation indices differ.")

        if not dataset.has_raw_evaluation_targets:
            raise ValueError(
                "Real cache lacks raw evaluation targets or context anchors."
            )


def _load_raw_training_split(
    data_dir: Path,
    *,
    expected_asset_cols: Sequence[str],
) -> dict[str, Any]:
    train_raw, val_raw, test_raw = load_candle_splits(data_dir)
    train_split, _, _ = clean_candle_splits(train_raw, val_raw, test_raw)

    if list(train_split["asset_cols"]) != list(expected_asset_cols):
        raise ValueError(
            "Raw training split and token cache asset ordering differ."
        )

    return train_split


def _load_or_create_fixed_graph_resource(
    *,
    run_dir: Path,
    raw_train_split: Mapping[str, Any],
    resource_config: FixedGraphResourceConfig,
    expected_asset_cols: Sequence[str],
    add_self_loops: bool,
    resume: bool,
    evaluate_only: bool,
) -> FixedGraphResource:
    """Fit once, persist, and reuse the exact training-only graph."""
    resource_path = run_dir / "fixed_graph_resource.pt"
    summary_path = run_dir / "fixed_graph_resource.json"

    if resource_path.is_file():
        try:
            payload = torch.load(
                resource_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                resource_path,
                map_location="cpu",
            )

        if not isinstance(payload, Mapping):
            raise TypeError(
                "Saved fixed graph resource must be a mapping."
            )

        resource = FixedGraphResource.from_payload(payload)
        resource.validate_against(
            config=resource_config,
            expected_asset_cols=expected_asset_cols,
            add_self_loops=add_self_loops,
        )
    else:
        if resume or evaluate_only:
            raise FileNotFoundError(
                "A resumed/evaluation fixed-graph run is missing "
                f"{resource_path}."
            )

        resource = fit_absolute_return_correlation_resource(
            raw_train_split,
            config=resource_config,
            expected_asset_cols=expected_asset_cols,
            add_self_loops=add_self_loops,
        )
        atomic_torch_save(
            resource.to_payload(),
            resource_path,
        )

    atomic_json_save(
        resource.metadata(),
        summary_path,
    )
    return resource


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")

    if args.evaluate_only and args.overwrite:
        raise ValueError("--evaluate-only and --overwrite cannot be combined.")

    if args.temperature_sweep and args.overwrite:
        raise ValueError("--temperature-sweep and --overwrite cannot be combined.")

    if args.temperature_sweep and args.evaluate_only:
        raise ValueError(
            "--temperature-sweep and --evaluate-only are separate modes."
        )

    if args.temperature_sweep and args.resume:
        raise ValueError("--temperature-sweep and --resume cannot be combined.")

    if args.validation_decode_every <= 0:
        raise ValueError("--validation-decode-every must be positive.")

    if args.decode_series_batch_size <= 0:
        raise ValueError("--decode-series-batch-size must be positive.")

    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative.")

    args.max_train_windows = _validate_optional_window_limit(
        args.max_train_windows,
        name="max_train_windows",
    )
    args.max_validation_windows = _validate_optional_window_limit(
        args.max_validation_windows,
        name="max_validation_windows",
    )

    dynamic_config_path = args.dynamic_config.expanduser().resolve()
    forecasting_config_path = args.forecasting_config.expanduser().resolve()
    train_cache_path = args.train_cache.expanduser().resolve()
    val_cache_path = args.val_cache.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    run_dir = output_root / args.run_name

    for path in (
        dynamic_config_path,
        forecasting_config_path,
        train_cache_path,
        val_cache_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)

    if (
        not args.resume
        and not args.evaluate_only
        and not args.temperature_sweep
        and run_dir.exists()
    ):
        if any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is not empty: {run_dir}. Use --resume or "
                "--overwrite explicitly."
            )

    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = load_dynamic_graph_config(
        dynamic_config_path,
        preset=args.preset,
    )
    resolved_config = apply_overrides(resolved_config, args)

    training_config = resolved_config["training"]
    decoding_config = resolved_config["decoding"]
    data_mode = str(resolved_config["data"]["mode"])

    model_values = resolved_config[
        "models"
    ][
        "dynamic_graph"
    ]

    graph_regularisation_config = (
        GraphRegularisationConfig.from_mapping(
            model_values.get(
                "graph_regularisation"
            )
        )
    )

    graph_values = model_values["graph"]
    graph_type = str(graph_values["type"])
    graph_add_self_loops = bool(
        graph_values["add_self_loops"]
    )
    graph_resource_config = (
        FixedGraphResourceConfig.from_mapping(
            resolved_config.get("graph_resource")
        )
    )
    graph_resource_config.validate(
        graph_type=graph_type,
        data_mode=data_mode,
    )

    if graph_type == "fixed":
        active_fixed_graph_penalties = {
            "graph_entropy_reg": (
                graph_regularisation_config.graph_entropy_reg
            ),
            "graph_target_entropy_reg": (
                graph_regularisation_config
                .graph_target_entropy_reg
            ),
            "graph_temporal_smooth_reg": (
                graph_regularisation_config
                .graph_temporal_smooth_reg
            ),
        }
        nonzero = {
            name: float(value)
            for name, value in active_fixed_graph_penalties.items()
            if float(value) != 0.0
        }
        if nonzero:
            raise ValueError(
                "A fixed graph cannot respond to graph regularisation. "
                "Set all graph-regularisation coefficients to zero. "
                f"Observed {nonzero}."
            )

    if data_mode not in {"real", "synthetic"}:
        raise ValueError("data.mode must be 'real' or 'synthetic'.")

    train_batch_size = _validate_positive_int(
        int(training_config["batch_size"]),
        name="training.batch_size",
    )
    validation_batch_size = (
        train_batch_size
        if args.validation_batch_size is None
        else _validate_positive_int(
            args.validation_batch_size,
            name="validation_batch_size",
        )
    )
    num_workers = int(training_config["num_workers"])
    if num_workers < 0:
        raise ValueError("training.num_workers cannot be negative.")

    max_epochs = _validate_positive_int(
        int(training_config["max_epochs"]),
        name="training.max_epochs",
    )
    patience = _validate_positive_int(
        int(training_config["patience"]),
        name="training.patience",
    )
    optimizer_name = str(
        training_config.get("optimizer", "adamw")
    ).strip().lower()
    scheduler_name = str(
        training_config.get("scheduler", "none")
    ).strip().lower()
    learning_rate = float(training_config["learning_rate"])
    graph_learning_rate = float(
        training_config.get("graph_learning_rate", learning_rate)
    )
    weight_decay = float(training_config["weight_decay"])
    gradient_clip_norm = float(training_config["gradient_clip_norm"])
    seed = int(training_config["seed"])

    raw_early_stopping_horizons = training_config.get(
        "early_stopping_horizons"
    )

    if not isinstance(
        raw_early_stopping_horizons,
        (list, tuple),
    ):
        raise TypeError(
            "training.early_stopping_horizons must be a list "
            "or tuple of integer horizons."
        )

    early_stopping_horizons = tuple(
        int(horizon)
        for horizon in raw_early_stopping_horizons
    )

    if not early_stopping_horizons:
        raise ValueError(
            "training.early_stopping_horizons must not be empty."
        )

    if any(
        horizon <= 0
        for horizon in early_stopping_horizons
    ):
        raise ValueError(
            "training.early_stopping_horizons must contain only "
            "positive integers."
        )

    if len(set(early_stopping_horizons)) != len(
        early_stopping_horizons
    ):
        raise ValueError(
            "training.early_stopping_horizons must not contain "
            "duplicate horizons."
        )

    early_stopping_metric = str(
        training_config.get(
            "early_stopping_metric",
            "",
        )
    ).strip()

    supported_early_stopping_metrics = {
        "decoded_cumulative_log_change_mae",
        "validation_token_loss",
    }

    if (
        early_stopping_metric
        not in supported_early_stopping_metrics
    ):
        raise ValueError(
            "training.early_stopping_metric must be one of "
            f"{sorted(supported_early_stopping_metrics)}. "
            f"Observed {early_stopping_metric!r}."
        )

    # Validation-token-loss selection is evaluated every epoch and is
    # deliberately decoupled from the much more expensive frozen-decoder
    # validation pass.  The selected CE checkpoint is decoded once after
    # training so its saved price-space artefacts match best_checkpoint.pt.

    if (
        data_mode == "synthetic"
        and early_stopping_metric
        == "decoded_cumulative_log_change_mae"
    ):
        raise ValueError(
            "Synthetic runs do not currently expose decoded raw-price "
            "metrics. Use training.early_stopping_metric="
            "'validation_token_loss' for synthetic data."
        )

    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'.")
    if scheduler_name not in {"none", "modern_tcn_type3"}:
        raise ValueError(
            "training.scheduler must be 'none' or 'modern_tcn_type3'."
        )
    if learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if graph_learning_rate <= 0:
        raise ValueError("training.graph_learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if gradient_clip_norm <= 0:
        raise ValueError("training.gradient_clip_norm must be positive.")

    checkpoint_token_selection = str(
        decoding_config.get("token_selection", "")
    )
    checkpoint_sample_count = int(
        decoding_config.get("sample_count", 1)
    )
    if checkpoint_token_selection != "argmax" or checkpoint_sample_count != 1:
        raise ValueError(
            "Training-time decoded validation diagnostics must use "
            "deterministic argmax decoding with decoding.sample_count=1. "
            "Temperature sampling is an inference-only sweep after the "
            "checkpoint is frozen."
        )

    device = resolve_device(args.device)
    requested_amp = bool(training_config["mixed_precision"])
    use_amp = requested_amp and device.type == "cuda"

    set_seed(seed)

    loaders: TokenGraphDataLoaders = build_token_graph_dataloaders(
        train_cache_path,
        val_cache_path,
        data_mode=data_mode,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    train_dataset_full = loaders.train_dataset
    validation_dataset_full = loaders.validation_dataset

    forecasting_config = load_yaml(forecasting_config_path)
    tokenizer: KronosTokenizerAdapter | None = None
    raw_train_split: dict[str, Any] | None = None
    fixed_graph_resource: FixedGraphResource | None = None
    fixed_adjacency: Tensor | None = None

    if data_mode == "real":
        if args.data_dir is None:
            raise ValueError("--data-dir is required when data.mode='real'.")

        data_dir = args.data_dir.expanduser().resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)

        raw_train_split = _load_raw_training_split(
            data_dir,
            expected_asset_cols=train_dataset_full.asset_cols,
        )

        if graph_resource_config.enabled:
            fixed_graph_resource = (
                _load_or_create_fixed_graph_resource(
                    run_dir=run_dir,
                    raw_train_split=raw_train_split,
                    resource_config=graph_resource_config,
                    expected_asset_cols=(
                        train_dataset_full.asset_cols
                    ),
                    add_self_loops=graph_add_self_loops,
                    resume=bool(args.resume),
                    evaluate_only=bool(
                        args.evaluate_only or args.temperature_sweep
                    ),
                )
            )
            fixed_adjacency = fixed_graph_resource.adjacency

        tokenizer = KronosTokenizerAdapter.from_config(
            forecasting_config,
            series_batch_size=args.decode_series_batch_size,
        ).load()

        expected_tokenizer = forecasting_config["models"]["kronos"]
        for cache_key, config_key in (
            ("tokenizer_id", "tokenizer_id"),
            ("tokenizer_revision", "tokenizer_revision"),
        ):
            cache_value = train_dataset_full.cache.get(cache_key)
            expected_value = expected_tokenizer[config_key]
            if cache_value != expected_value:
                raise ValueError(
                    f"Token cache {cache_key}={cache_value!r}, but the "
                    f"forecasting config expects {expected_value!r}."
                )
    else:
        data_dir = None

    model = DynamicGraphTokenForecaster.from_config(
        resolved_config,
        fixed_adjacency=fixed_adjacency,
    ).to(device)

    available_evaluation_horizons = tuple(
        int(horizon)
        for horizon in model.config.heads.evaluation_horizons
    )

    missing_early_stopping_horizons = [
        horizon
        for horizon in early_stopping_horizons
        if horizon not in available_evaluation_horizons
    ]

    if missing_early_stopping_horizons:
        raise ValueError(
            "training.early_stopping_horizons contains horizons that "
            "are not present in the model evaluation horizons. "
            f"Missing={missing_early_stopping_horizons}, "
            f"available={list(available_evaluation_horizons)}."
        )

    if (
        graph_regularisation_config
        .graph_temporal_smooth_reg
        > 0.0
        and model.config.graph.type
        not in {
            "dynamic",
            "dynamic_base",
        }
    ):
        raise ValueError(
            "graph_temporal_smooth_reg is only meaningful for "
            "graph.type='dynamic' or 'dynamic_base'."
        )

    expected_origin_delta = int(
        resolved_config[
            "forecasting"
        ][
            "stride"
        ]
    )

    _validate_cache_against_model(
        dataset=train_dataset_full,
        model=model,
        data_mode=data_mode,
    )
    _validate_cache_against_model(
        dataset=validation_dataset_full,
        model=model,
        data_mode=data_mode,
    )

    train_dataset = limit_dataset(
        train_dataset_full,
        args.max_train_windows,
    )
    validation_dataset = limit_dataset(
        validation_dataset_full,
        args.max_validation_windows,
    )

    validation_loader = build_loader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    optimizer = _build_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        graph_learning_rate=graph_learning_rate,
        weight_decay=weight_decay,
    )
    scaler = _new_grad_scaler(use_amp)

    repository_root = Path(__file__).resolve().parents[2]
    project_commit = _git_value(["rev-parse", "HEAD"], cwd=repository_root)
    git_status = _git_value(["status", "--short"], cwd=repository_root)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    run_metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "resolved_preset": resolved_config["resolved_preset"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dynamic_config_path": str(dynamic_config_path),
        "forecasting_config_path": str(forecasting_config_path),
        "train_cache_path": str(train_cache_path),
        "validation_cache_path": str(val_cache_path),
        "data_dir": None if data_dir is None else str(data_dir),
        "data_mode": data_mode,
        "device": str(device),
        "requested_mixed_precision": requested_amp,
        "active_cuda_amp": use_amp,
        "train_batch_size": train_batch_size,
        "validation_batch_size": validation_batch_size,
        "num_workers": num_workers,
        "max_epochs": max_epochs,
        "patience": patience,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "learning_rate": learning_rate,
        "graph_learning_rate": graph_learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "seed": seed,
        "validation_decode_every": args.validation_decode_every,
        "decode_series_batch_size": args.decode_series_batch_size,
        "max_train_windows": args.max_train_windows,
        "max_validation_windows": args.max_validation_windows,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "asset_cols": list(train_dataset_full.asset_cols),
        "s1_token_space": {
            "id_space": train_dataset_full.s1_id_space,
            "vocabulary_size": train_dataset_full.s1_vocabulary_size,
            "remapping_method": train_dataset_full.cache.get(
                "s1_remapping_method"
            ),
            "resource_hash": (
                train_dataset_full.s1_remapping_resource_hash
            ),
            "training_coverage_percent": train_dataset_full.cache.get(
                "s1_training_coverage_percent"
            ),
            "fallback_original_id": train_dataset_full.cache.get(
                "s1_fallback_original_id"
            ),
        },
        "project_git_commit": project_commit,
        "project_git_status": git_status,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "selection_metric": early_stopping_metric,
        "selection_horizons": (
            [
                int(horizon)
                for horizon in early_stopping_horizons
            ]
            if early_stopping_metric
            == "decoded_cumulative_log_change_mae"
            else None
        ),
        "fixed_graph_resource": (
            None
            if fixed_graph_resource is None
            else fixed_graph_resource.metadata()
        ),
        "graph_regularisation": {
            "graph_reg_layer": (
                graph_regularisation_config.graph_reg_layer
            ),
            "graph_reg_warmup_epochs": (
                graph_regularisation_config
                .graph_reg_warmup_epochs
            ),
            "graph_entropy_reg": (
                graph_regularisation_config.graph_entropy_reg
            ),
            "graph_target_entropy": (
                graph_regularisation_config.graph_target_entropy
            ),
            "graph_target_entropy_reg": (
                graph_regularisation_config
                .graph_target_entropy_reg
            ),
            "graph_temporal_smooth_reg": (
                graph_regularisation_config
                .graph_temporal_smooth_reg
            ),
        },
    }

    signature_config = deepcopy(resolved_config)
    # Temperature/sample-count choices are post-training inference policy,
    # not learned-model hyperparameters. Excluding this mapping allows a
    # frozen checkpoint to be evaluated at a different temperature grid.
    signature_config.pop("temperature_sweep", None)

    signature_values = {
        "resolved_config": signature_config,
        "train_cache": str(train_cache_path),
        "validation_cache": str(val_cache_path),
        "data_mode": data_mode,
        "asset_cols": list(train_dataset_full.asset_cols),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "project_git_commit": project_commit,
        "fixed_graph_resource_hash": (
            None
            if fixed_graph_resource is None
            else fixed_graph_resource.resource_hash
        ),
        "s1_id_space": train_dataset_full.s1_id_space,
        "s1_vocabulary_size": train_dataset_full.s1_vocabulary_size,
        "s1_remapping_resource_hash": (
            train_dataset_full.s1_remapping_resource_hash
        ),
    }
    run_signature = _config_signature(signature_values)
    run_metadata["run_signature"] = run_signature

    metadata_path = run_dir / "run_metadata.json"
    if (args.evaluate_only or args.temperature_sweep) and metadata_path.is_file():
        existing_metadata = load_json(metadata_path)
        existing_signature = existing_metadata.get("run_signature")
        if existing_signature is not None and existing_signature != run_signature:
            raise ValueError(
                "Saved run metadata signature differs from the requested "
                "evaluation configuration."
            )
        run_metadata = deepcopy(existing_metadata)
        run_metadata["run_signature"] = run_signature
        run_metadata["last_evaluated_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )
        if args.temperature_sweep:
            run_metadata["temperature_sweep_requested"] = True
    if not args.temperature_sweep:
        atomic_json_save(resolved_config, run_dir / "resolved_config.json")
    else:
        temperature_values = resolved_config.get("temperature_sweep")
        if isinstance(temperature_values, Mapping):
            atomic_json_save(
                dict(temperature_values),
                run_dir
                / "temperature_sweep"
                / "requested_temperature_sweep_config.json",
            )
    atomic_json_save(run_metadata, metadata_path)

    # best_checkpoint.pt is controlled by the configured early-stopping
    # metric.  The final tokenized ModernTCN preset selects the lowest
    # teacher-forced validation coarse-token cross-entropy.  Deterministic
    # argmax decoding remains a diagnostic and is regenerated once from the
    # selected checkpoint after training.
    best_checkpoint_path = run_dir / "best_checkpoint.pt"

    # Secondary checkpoints do not control patience. They preserve
    # alternative validation-selection views from the same run.
    best_all_horizons_checkpoint_path = (
        run_dir
        / "best_all_horizons_checkpoint.pt"
    )
    best_validation_ce_checkpoint_path = (
        run_dir
        / "best_validation_ce_checkpoint.pt"
    )

    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    history_path = run_dir / "history.csv"

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    evaluations_without_improvement = 0

    secondary_selection_state: dict[
        str,
        dict[str, Any],
    ] = {
        "all_horizons": {
            "best_score": math.inf,
            "best_epoch": 0,
        },
        "validation_ce": {
            "best_score": math.inf,
            "best_epoch": 0,
        },
    }

    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume because {last_checkpoint_path} does not exist."
            )

        checkpoint = load_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            expected_signature=run_signature,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        evaluations_without_improvement = int(
            checkpoint["evaluations_without_improvement"]
        )
        history = list(checkpoint.get("history", []))

        saved_secondary_selection_state = checkpoint.get(
            "secondary_selection_state"
        )

        if saved_secondary_selection_state is not None:
            if not isinstance(
                saved_secondary_selection_state,
                Mapping,
            ):
                raise TypeError(
                    "Checkpoint secondary_selection_state must be "
                    "a mapping."
                )

            for selection_name in (
                "all_horizons",
                "validation_ce",
            ):
                saved_entry = (
                    saved_secondary_selection_state.get(
                        selection_name
                    )
                )

                if not isinstance(saved_entry, Mapping):
                    raise ValueError(
                        "Checkpoint is missing valid secondary "
                        f"selection state for {selection_name!r}."
                    )

                secondary_selection_state[
                    selection_name
                ] = {
                    "best_score": float(
                        saved_entry["best_score"]
                    ),
                    "best_epoch": int(
                        saved_entry["best_epoch"]
                    ),
                }

        previous_metadata = checkpoint.get("run_metadata")
        if isinstance(previous_metadata, Mapping):
            original_started_at = previous_metadata.get("started_at_utc")
            if original_started_at is not None:
                run_metadata["started_at_utc"] = original_started_at
            run_metadata["resume_count"] = int(
                previous_metadata.get("resume_count", 0)
            ) + 1
            run_metadata["resumed_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
            atomic_json_save(run_metadata, run_dir / "run_metadata.json")

    wandb_run = _init_wandb(
        args=args,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
    )

    try:
        if args.temperature_sweep:
            if data_mode != "real":
                raise ValueError(
                    "Temperature sampling requires real decoded price metrics."
                )
            if tokenizer is None or raw_train_split is None:
                raise RuntimeError(
                    "Temperature sweep requires the frozen Kronos decoder "
                    "and raw training split."
                )
            if not best_checkpoint_path.is_file():
                raise FileNotFoundError(best_checkpoint_path)
            checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("run_signature") != run_signature:
                raise ValueError(
                    "Best checkpoint run signature differs from this run."
                )
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            model.to(device)
            checkpoint_epoch = int(checkpoint["epoch"])
            temperature_config = resolved_config.get("temperature_sweep")
            if not isinstance(temperature_config, Mapping):
                raise ValueError(
                    "Config must contain a temperature_sweep mapping."
                )
            sweep_results = run_temperature_sweep(
                model=model,
                loader=validation_loader,
                dataset=validation_dataset_full,
                device=device,
                use_amp=use_amp,
                tokenizer=tokenizer,
                raw_train_split=raw_train_split,
                decode_series_batch_size=args.decode_series_batch_size,
                temperature_config=temperature_config,
                output_dir=run_dir / "temperature_sweep",
                checkpoint_epoch=checkpoint_epoch,
                evaluation_horizons=available_evaluation_horizons,
            )
            run_metadata["temperature_sweep_completed_at_utc"] = (
                datetime.now(timezone.utc).isoformat()
            )
            run_metadata["temperature_sweep_results_path"] = str(
                run_dir
                / "temperature_sweep"
                / "temperature_sweep_results.csv"
            )
            atomic_json_save(run_metadata, metadata_path)
            print(sweep_results.to_string(index=False))
            print("TOKEN TEMPERATURE SWEEP PASSED")
            return

        if args.evaluate_only:
            if not best_checkpoint_path.is_file():
                raise FileNotFoundError(best_checkpoint_path)

            checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("run_signature") != run_signature:
                raise ValueError(
                    "Best checkpoint run signature differs from this run."
                )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(device)
            epoch = int(checkpoint["epoch"])

            teacher_forced = run_teacher_forced_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm,
                description="validation supervised token loss",
                graph_regularisation_config=(
                    graph_regularisation_config
                ),
                current_epoch=max(
                    0,
                    epoch - 1,
                ),
                expected_origin_delta=(
                    expected_origin_delta
                ),
            )
            bundle = generate_validation_artifacts(
                model=model,
                loader=validation_loader,
                dataset=validation_dataset_full,
                device=device,
                use_amp=use_amp,
                decoding_config=decoding_config,
                tokenizer=tokenizer,
                raw_train_split=raw_train_split,
                decode_series_batch_size=args.decode_series_batch_size,
                early_stopping_horizons=early_stopping_horizons,
            )
            _save_best_validation_artifacts(
                run_dir=run_dir,
                epoch=epoch,
                bundle=bundle,
                teacher_forced=teacher_forced,
                model=model,
            )
            print("DYNAMIC GRAPH EVALUATION-ONLY RUN PASSED")
            return

        if start_epoch > max_epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"which is not below max_epochs={max_epochs}."
            )

        for epoch in range(start_epoch, max_epochs + 1):
            current_learning_rates = _adjust_learning_rate(
                optimizer,
                scheduler=scheduler_name,
                completed_epoch=epoch - 1,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_loader = build_loader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                seed=seed + epoch - 1,
                pin_memory=device.type == "cuda",
            )

            epoch_start = perf_counter()
            train_metrics = run_teacher_forced_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm,
                description=f"epoch {epoch} training",
                graph_regularisation_config=(
                    graph_regularisation_config
                ),
                current_epoch=epoch - 1,
                expected_origin_delta=(
                    expected_origin_delta
                ),
            )
            validation_teacher = run_teacher_forced_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm,
                description=(
                    f"epoch {epoch} validation supervised token loss"
                ),
                graph_regularisation_config=(
                    graph_regularisation_config
                ),
                current_epoch=epoch - 1,
                expected_origin_delta=(
                    expected_origin_delta
                ),
            )

            validation_ce_score = float(
                validation_teacher.token_loss
            )

            if not math.isfinite(validation_ce_score):
                raise ValueError(
                    "Validation token CE is non-finite."
                )

            validation_ce_best_score = float(
                secondary_selection_state[
                    "validation_ce"
                ][
                    "best_score"
                ]
            )

            validation_ce_improved = (
                validation_ce_score
                < validation_ce_best_score
            )

            if validation_ce_improved:
                secondary_selection_state[
                    "validation_ce"
                ] = {
                    "best_score": float(
                        validation_ce_score
                    ),
                    "best_epoch": int(epoch),
                }

            should_decode = (
                epoch == 1
                or epoch == max_epochs
                or epoch % args.validation_decode_every == 0
            )

            bundle: ValidationBundle | None = None
            primary_score: float | None = (
                validation_ce_score
                if early_stopping_metric
                == "validation_token_loss"
                else None
            )

            all_horizons_score: float | None = None

            improved = False
            all_horizons_improved = False
            selection_was_evaluated = (
                early_stopping_metric
                == "validation_token_loss"
            )

            if should_decode:
                bundle = generate_validation_artifacts(
                    model=model,
                    loader=validation_loader,
                    dataset=validation_dataset_full,
                    device=device,
                    use_amp=use_amp,
                    decoding_config=decoding_config,
                    tokenizer=tokenizer,
                    raw_train_split=raw_train_split,
                    decode_series_batch_size=args.decode_series_batch_size,
                    early_stopping_horizons=early_stopping_horizons,
                )

                if (
                    early_stopping_metric
                    == "decoded_cumulative_log_change_mae"
                ):
                    if bundle.primary_score is None:
                        raise RuntimeError(
                            "Decoded cumulative-log-change MAE was "
                            "selected for early stopping, but validation "
                            "did not produce a decoded primary score."
                        )

                    primary_score = float(
                        bundle.primary_score
                    )
                    selection_was_evaluated = True

                elif (
                    early_stopping_metric
                    != "validation_token_loss"
                ):
                    raise AssertionError(
                        "Unsupported early-stopping metric passed "
                        "configuration validation."
                    )

                if data_mode == "real":
                    if bundle.metric_results is None:
                        raise RuntimeError(
                            "Real-data validation did not return "
                            "decoded metric results."
                        )

                    all_horizons_score = (
                        _mean_cumulative_log_change_mae_at_horizons(
                            metric_results=bundle.metric_results,
                            available_horizons=(
                                available_evaluation_horizons
                            ),
                            selected_horizons=(
                                available_evaluation_horizons
                            ),
                        )
                    )

                    all_horizons_best_score = float(
                        secondary_selection_state[
                            "all_horizons"
                        ][
                            "best_score"
                        ]
                    )

                    all_horizons_improved = (
                        all_horizons_score
                        < all_horizons_best_score
                    )

                    if all_horizons_improved:
                        secondary_selection_state[
                            "all_horizons"
                        ] = {
                            "best_score": float(
                                all_horizons_score
                            ),
                            "best_epoch": int(epoch),
                        }

            if selection_was_evaluated:
                if primary_score is None or not math.isfinite(
                    primary_score
                ):
                    raise ValueError(
                        "The configured primary validation score is "
                        "unavailable or non-finite."
                    )

                improved = (
                    primary_score
                    < (best_score - args.min_delta)
                )

                if improved:
                    best_score = primary_score
                    best_epoch = epoch
                    evaluations_without_improvement = 0

                    checkpoint = build_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        best_score=best_score,
                        best_epoch=best_epoch,
                        evaluations_without_improvement=(
                            evaluations_without_improvement
                        ),
                        history=history,
                        secondary_selection_state=(
                            secondary_selection_state
                        ),
                        run_signature=run_signature,
                        resolved_config=resolved_config,
                        run_metadata=run_metadata,
                    )
                    checkpoint[
                        "checkpoint_selection"
                    ] = {
                        "name": "primary",
                        "metric": early_stopping_metric,
                        "horizons": (
                            [
                                int(horizon)
                                for horizon in (
                                    early_stopping_horizons
                                )
                            ]
                            if early_stopping_metric
                            == "decoded_cumulative_log_change_mae"
                            else None
                        ),
                        "score": float(primary_score),
                        "epoch": int(epoch),
                    }
                    atomic_torch_save(
                        checkpoint,
                        best_checkpoint_path,
                    )

                    # A decoded bundle exists only on scheduled decoder
                    # validation epochs.  CE-selected checkpoints that do
                    # not coincide with one are decoded exactly once after
                    # training, from best_checkpoint.pt.
                    if bundle is not None:
                        _save_best_validation_artifacts(
                            run_dir=run_dir,
                            epoch=epoch,
                            bundle=bundle,
                            teacher_forced=validation_teacher,
                            model=model,
                        )
                else:
                    evaluations_without_improvement += 1

            if all_horizons_improved:
                if all_horizons_score is None:
                    raise AssertionError(
                        "all_horizons_improved is True but "
                        "all_horizons_score is unavailable."
                    )

                all_horizons_checkpoint = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    evaluations_without_improvement=(
                        evaluations_without_improvement
                    ),
                    history=history,
                    secondary_selection_state=(
                        secondary_selection_state
                    ),
                    run_signature=run_signature,
                    resolved_config=resolved_config,
                    run_metadata=run_metadata,
                )

                all_horizons_checkpoint[
                    "checkpoint_selection"
                ] = {
                    "name": "all_horizons",
                    "metric": (
                        "mean_validation_"
                        "cumulative_log_change_mae"
                    ),
                    "horizons": [
                        int(horizon)
                        for horizon in (
                            available_evaluation_horizons
                        )
                    ],
                    "score": float(
                        all_horizons_score
                    ),
                    "epoch": int(epoch),
                }

                atomic_torch_save(
                    all_horizons_checkpoint,
                    best_all_horizons_checkpoint_path,
                )

            # Preserve the CE view as a secondary checkpoint only when it
            # is not already the primary selection criterion.
            if (
                validation_ce_improved
                and early_stopping_metric
                != "validation_token_loss"
            ):
                validation_ce_checkpoint = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    evaluations_without_improvement=(
                        evaluations_without_improvement
                    ),
                    history=history,
                    secondary_selection_state=(
                        secondary_selection_state
                    ),
                    run_signature=run_signature,
                    resolved_config=resolved_config,
                    run_metadata=run_metadata,
                )

                validation_ce_checkpoint[
                    "checkpoint_selection"
                ] = {
                    "name": "validation_ce",
                    "metric": "validation_token_loss",
                    "horizons": None,
                    "score": float(
                        validation_ce_score
                    ),
                    "epoch": int(epoch),
                }

                atomic_torch_save(
                    validation_ce_checkpoint,
                    best_validation_ce_checkpoint_path,
                )

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": float(current_learning_rates["backbone"]),
                "backbone_learning_rate": float(
                    current_learning_rates["backbone"]
                ),
                "graph_learning_rate": current_learning_rates["graph"],
                "train_total_loss": train_metrics.total_loss,
                "train_token_loss": train_metrics.token_loss,
                "train_graph_regularisation_loss": (
                    train_metrics.graph_regularisation_loss
                ),
                "train_backcast_loss": train_metrics.backcast_loss,
                "train_backcast_penalty": train_metrics.backcast_penalty,
                "train_s1_loss": train_metrics.s1_loss,
                "train_s2_loss": train_metrics.s2_loss,
                "train_s1_accuracy": train_metrics.s1_accuracy,
                "train_s2_accuracy": train_metrics.s2_accuracy,
                "train_graph_mean_row_entropy": (
                    train_metrics.graph_mean_row_entropy
                ),
                "train_graph_mean_effective_neighbours": (
                    train_metrics.graph_mean_effective_neighbours
                ),
                "train_graph_entropy_penalty": (
                    train_metrics.graph_entropy_penalty
                ),
                "train_graph_target_entropy_penalty": (
                    train_metrics.graph_target_entropy_penalty
                ),
                "train_graph_temporal_smooth_penalty": (
                    train_metrics.graph_temporal_smooth_penalty
                ),
                "train_graph_warmup_scale": (
                    train_metrics.graph_warmup_scale
                ),
                "train_graph_valid_smoothing_pairs": (
                    train_metrics.graph_valid_smoothing_pairs
                ),
                "train_spatial_beta": train_metrics.spatial_beta,
                "validation_total_loss": validation_teacher.total_loss,
                "validation_token_loss": validation_teacher.token_loss,
                "validation_graph_regularisation_loss": (
                    validation_teacher.graph_regularisation_loss
                ),
                "validation_backcast_loss": (
                    validation_teacher.backcast_loss
                ),
                "validation_backcast_penalty": (
                    validation_teacher.backcast_penalty
                ),
                "validation_s1_loss": validation_teacher.s1_loss,
                "validation_s2_loss": validation_teacher.s2_loss,
                "validation_s1_accuracy": validation_teacher.s1_accuracy,
                "validation_s2_accuracy": validation_teacher.s2_accuracy,
                "validation_graph_mean_row_entropy": (
                    validation_teacher.graph_mean_row_entropy
                ),
                "validation_graph_mean_effective_neighbours": (
                    validation_teacher.graph_mean_effective_neighbours
                ),
                "validation_graph_entropy_penalty": (
                    validation_teacher.graph_entropy_penalty
                ),
                "validation_graph_target_entropy_penalty": (
                    validation_teacher.graph_target_entropy_penalty
                ),
                "validation_graph_temporal_smooth_penalty": (
                    validation_teacher.graph_temporal_smooth_penalty
                ),
                "validation_graph_warmup_scale": (
                    validation_teacher.graph_warmup_scale
                ),
                "validation_graph_valid_smoothing_pairs": (
                    validation_teacher.graph_valid_smoothing_pairs
                ),
                "validation_spatial_beta": validation_teacher.spatial_beta,
                "spatial_beta": validation_teacher.spatial_beta,
                "decoded_validation": should_decode,
                "primary_score": primary_score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "improved": improved,

                "all_horizons_score": (
                    all_horizons_score
                ),
                "all_horizons_improved": (
                    all_horizons_improved
                ),
                "best_all_horizons_score": float(
                    secondary_selection_state[
                        "all_horizons"
                    ][
                        "best_score"
                    ]
                ),
                "best_all_horizons_epoch": int(
                    secondary_selection_state[
                        "all_horizons"
                    ][
                        "best_epoch"
                    ]
                ),

                "validation_ce_selection_score": (
                    validation_ce_score
                ),
                "validation_ce_improved": (
                    validation_ce_improved
                ),
                "best_validation_ce_score": float(
                    secondary_selection_state[
                        "validation_ce"
                    ][
                        "best_score"
                    ]
                ),
                "best_validation_ce_epoch": int(
                    secondary_selection_state[
                        "validation_ce"
                    ][
                        "best_epoch"
                    ]
                ),
                "evaluations_without_improvement": (
                    evaluations_without_improvement
                ),
                "train_seconds": train_metrics.seconds,
                "validation_teacher_seconds": validation_teacher.seconds,
                "generation_seconds": (
                    None
                    if bundle is None
                    else bundle.generation_metrics.seconds
                ),
                "epoch_seconds": perf_counter() - epoch_start,
                "cuda_peak_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
                    if device.type == "cuda"
                    else None
                ),
                "cuda_peak_reserved_gib": (
                    float(torch.cuda.max_memory_reserved(device) / (1024 ** 3))
                    if device.type == "cuda"
                    else None
                ),
            }

            if bundle is not None:
                epoch_record.update(
                    bundle.diagnostics
                )
                epoch_record.update(
                    _flatten_metric_results_for_logging(
                        bundle.metric_results,
                        model.config.heads.evaluation_horizons,
                    )
                )

            # bundle.diagnostics contains the decoded score. Restore the
            # actual configured early-stopping score in case validation
            # token loss is being used instead.
            epoch_record["primary_score"] = primary_score
            epoch_record["selection_metric"] = (
                early_stopping_metric
            )
            epoch_record["selection_horizons"] = (
                json.dumps(
                    [
                        int(horizon)
                        for horizon in early_stopping_horizons
                    ]
                )
                if early_stopping_metric
                == "decoded_cumulative_log_change_mae"
                else None
            )

            history.append(epoch_record)
            atomic_csv_save(pd.DataFrame(history), history_path)

            last_checkpoint = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                evaluations_without_improvement=evaluations_without_improvement,
                history=history,
                secondary_selection_state=(
                    secondary_selection_state
                ),
                run_signature=run_signature,
                resolved_config=resolved_config,
                run_metadata=run_metadata,
            )
            atomic_torch_save(last_checkpoint, last_checkpoint_path)

            if wandb_run is not None:
                wandb_values = {
                    key: value
                    for key, value in epoch_record.items()
                    if isinstance(value, (int, float)) and value is not None
                }
                wandb_run.log(wandb_values, step=epoch)

            score_text = (
                "not decoded"
                if primary_score is None
                else f"{primary_score:.8f}"
            )
            print(
                f"epoch {epoch:>3}/{max_epochs} | "
                f"train_objective={train_metrics.total_loss:.6f} | "
                f"train_token={train_metrics.token_loss:.6f} | "
                f"val_token={validation_teacher.token_loss:.6f} | "
                f"primary={score_text} | "
                f"best_epoch={best_epoch} | "
                f"backbone_lr={float(current_learning_rates['backbone']):.3g} | "
                f"graph_lr={current_learning_rates['graph']} | "
                f"beta={validation_teacher.spatial_beta} | "
                f"seconds={epoch_record['epoch_seconds']:.1f}"
            )

            if (
                selection_was_evaluated
                and evaluations_without_improvement >= patience
            ):
                evaluation_unit = (
                    "validation epochs"
                    if early_stopping_metric
                    == "validation_token_loss"
                    else "decoded validations"
                )
                print(
                    "Early stopping: no primary-metric improvement across "
                    f"{evaluations_without_improvement} "
                    f"{evaluation_unit}."
                )
                break

        if best_epoch <= 0 or not best_checkpoint_path.is_file():
            raise RuntimeError("Training finished without a best checkpoint.")

        if early_stopping_metric == "validation_token_loss":
            # The CE-selected epoch may not coincide with a scheduled
            # frozen-decoder validation.  Regenerate deterministic argmax
            # price-space artefacts once from the exact selected checkpoint
            # so best_validation_*.pt/csv and the subsequent temperature
            # sweep all refer to the same model weights.
            selected_checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if selected_checkpoint.get("run_signature") != run_signature:
                raise ValueError(
                    "Best checkpoint run signature differs while "
                    "regenerating selected validation artefacts."
                )
            model.load_state_dict(
                selected_checkpoint["model_state_dict"],
                strict=True,
            )
            model.to(device)

            selected_teacher_forced = run_teacher_forced_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm,
                description=(
                    "selected CE checkpoint validation supervised "
                    "token loss"
                ),
                graph_regularisation_config=(
                    graph_regularisation_config
                ),
                current_epoch=max(0, best_epoch - 1),
                expected_origin_delta=(
                    expected_origin_delta
                ),
            )

            selected_bundle = generate_validation_artifacts(
                model=model,
                loader=validation_loader,
                dataset=validation_dataset_full,
                device=device,
                use_amp=use_amp,
                decoding_config=decoding_config,
                tokenizer=tokenizer,
                raw_train_split=raw_train_split,
                decode_series_batch_size=(
                    args.decode_series_batch_size
                ),
                early_stopping_horizons=(
                    early_stopping_horizons
                ),
            )

            _save_best_validation_artifacts(
                run_dir=run_dir,
                epoch=best_epoch,
                bundle=selected_bundle,
                teacher_forced=(
                    selected_teacher_forced
                ),
                model=model,
            )

            if selected_bundle.primary_score is not None:
                run_metadata[
                    "selected_checkpoint_argmax_decoded_score"
                ] = float(
                    selected_bundle.primary_score
                )
            run_metadata[
                "selected_checkpoint_artifacts_regenerated"
            ] = True
            run_metadata[
                "selected_checkpoint_artifacts_epoch"
            ] = int(best_epoch)

        run_metadata.update(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "epochs_completed": int(history[-1]["epoch"]),
                "status": "completed",
                "total_epoch_seconds": float(
                    sum(float(row["epoch_seconds"]) for row in history)
                ),
                "maximum_cuda_peak_allocated_gib": (
                    max(
                        float(row["cuda_peak_allocated_gib"])
                        for row in history
                        if row.get("cuda_peak_allocated_gib") is not None
                    )
                    if any(
                        row.get("cuda_peak_allocated_gib") is not None
                        for row in history
                    )
                    else None
                ),
                "maximum_cuda_peak_reserved_gib": (
                    max(
                        float(row["cuda_peak_reserved_gib"])
                        for row in history
                        if row.get("cuda_peak_reserved_gib") is not None
                    )
                    if any(
                        row.get("cuda_peak_reserved_gib") is not None
                        for row in history
                    )
                    else None
                ),
            }
        )
        atomic_json_save(run_metadata, run_dir / "run_metadata.json")

        if wandb_run is not None:
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_score"] = best_score

        print("Best epoch:", best_epoch)
        print("Best primary score:", f"{best_score:.8f}")
        print("Best checkpoint:", best_checkpoint_path)
        print("DYNAMIC GRAPH TRAINING RUN PASSED")

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()








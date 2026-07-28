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
    PREDICTOR_SELECTION_PRESETS,
    load_dynamic_graph_config,
    validate_dynamic_graph_config,
)
from src.models.dynamic_graph.contracts import GraphOutput
from src.models.dynamic_graph.future_predictor import (
    FutureTokenLoss,
    FutureTokenPrediction,
    compute_future_token_loss,
)
from src.models.dynamic_graph.model import (
    DynamicGraphTokenForecaster,
    GeneratedTokenForecast,
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
    total_loss: float
    s1_loss: float
    s2_loss: float
    s1_accuracy: float
    s2_accuracy: float
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
    graph_artifacts: dict[str, Any]
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
        choices=PREDICTOR_SELECTION_PRESETS,
        default=None,
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
    return FutureTokenPrediction(
        future_hidden=output.future_hidden,
        s1_logits=output.s1_logits,
        s2_logits=output.s2_logits,
        selected_s1=output.s1_logits.argmax(dim=-1),
        selected_s2=output.s2_logits.argmax(dim=-1),
    )


def compute_model_loss(
    model: DynamicGraphTokenForecaster,
    output: Any,
    target_s1: Tensor,
    target_s2: Tensor,
    batch: Mapping[str, Any],
) -> FutureTokenLoss:
    token_loss = compute_future_token_loss(
        _future_prediction_for_loss(output),
        target_s1,
        target_s2,
        loss_config=model.config.loss,
        evaluation_horizons=model.config.heads.evaluation_horizons,
        s2_loss_weight=model.config.heads.s2_loss_weight,
    )

    if model.config.backcast.enabled:
        if "context_normalised_ohlcv" not in batch:
            raise KeyError(
                "Backcasting is enabled, but the cache does not contain "
                "context_normalised_ohlcv. Regenerate the cache with an "
                "explicit backcast target before enabling this ablation."
            )

        if output.backcast is None:
            raise RuntimeError("Backcasting is enabled but the model returned None.")

        target = torch.as_tensor(batch["context_normalised_ohlcv"]).to(
            device=output.backcast.device,
            dtype=output.backcast.dtype,
            non_blocking=True,
        )

        if tuple(target.shape) != tuple(output.backcast.shape):
            raise ValueError(
                "Backcast target and output shapes differ: "
                f"{tuple(target.shape)} versus {tuple(output.backcast.shape)}."
            )

        backcast_loss = torch.nn.functional.mse_loss(output.backcast, target)
        total = token_loss.total + (
            float(model.config.backcast.loss_weight) * backcast_loss
        )

        return FutureTokenLoss(
            total=total,
            s1=token_loss.s1,
            s2=token_loss.s2,
            s1_by_step=token_loss.s1_by_step,
            s2_by_step=token_loss.s2_by_step,
            weights=token_loss.weights,
        )

    return token_loss


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
) -> TeacherForcedEpochMetrics:
    training = optimizer is not None
    model.train(training)

    synchronise_device(device)
    start = perf_counter()
    example_count = 0
    total_loss_sum = 0.0
    s1_loss_sum = 0.0
    s2_loss_sum = 0.0

    prediction_length = model.config.prediction_length
    s1_loss_by_step_sum = torch.zeros(prediction_length, dtype=torch.float64)
    s2_loss_by_step_sum = torch.zeros(prediction_length, dtype=torch.float64)
    s1_correct_by_step = torch.zeros(prediction_length, dtype=torch.float64)
    s2_correct_by_step = torch.zeros(prediction_length, dtype=torch.float64)
    token_count_by_step = torch.zeros(prediction_length, dtype=torch.float64)

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        context_tokens, target_s1, target_s2 = move_training_batch(
            batch,
            device=device,
        )

        batch_size = int(context_tokens.shape[0])
        num_nodes = int(context_tokens.shape[2])

        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = torch.enable_grad() if training else torch.inference_mode()

        with grad_context:
            with _autocast_context(device, use_amp):
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
                )

            if training:
                if use_amp:
                    scaler.scale(loss.total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    optimizer.step()

        example_count += batch_size
        total_loss_sum += float(loss.total.detach().item()) * batch_size
        s1_loss_sum += float(loss.s1.detach().item()) * batch_size
        s2_loss_sum += float(loss.s2.detach().item()) * batch_size
        s1_loss_by_step_sum += (
            loss.s1_by_step.detach().cpu().to(torch.float64) * batch_size
        )
        s2_loss_by_step_sum += (
            loss.s2_by_step.detach().cpu().to(torch.float64) * batch_size
        )

        s1_predictions = output.s1_logits.detach().argmax(dim=-1)
        s2_predictions = output.s2_logits.detach().argmax(dim=-1)

        s1_correct_by_step += (
            (s1_predictions == target_s1)
            .sum(dim=(0, 2))
            .detach()
            .cpu()
            .to(torch.float64)
        )
        s2_correct_by_step += (
            (s2_predictions == target_s2)
            .sum(dim=(0, 2))
            .detach()
            .cpu()
            .to(torch.float64)
        )
        token_count_by_step += float(batch_size * num_nodes)

        progress.set_postfix(
            loss=f"{float(loss.total.detach().item()):.4f}",
            refresh=False,
        )

    synchronise_device(device)

    if example_count == 0:
        raise RuntimeError("The DataLoader yielded no examples.")

    s1_accuracy_by_step = s1_correct_by_step / token_count_by_step.clamp_min(1)
    s2_accuracy_by_step = s2_correct_by_step / token_count_by_step.clamp_min(1)

    return TeacherForcedEpochMetrics(
        total_loss=total_loss_sum / example_count,
        s1_loss=s1_loss_sum / example_count,
        s2_loss=s2_loss_sum / example_count,
        s1_accuracy=float(
            s1_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        ),
        s2_accuracy=float(
            s2_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        ),
        s1_loss_by_step=s1_loss_by_step_sum / example_count,
        s2_loss_by_step=s2_loss_by_step_sum / example_count,
        s1_accuracy_by_step=s1_accuracy_by_step,
        s2_accuracy_by_step=s2_accuracy_by_step,
        examples=example_count,
        seconds=perf_counter() - start,
    )


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
) -> ValidationBundle:
    model.eval()
    synchronise_device(device)
    start = perf_counter()

    prediction_length = model.config.prediction_length
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

    invalid_dense_count = 0
    invalid_dense_total = 0
    invalid_evaluation_count = 0
    invalid_evaluation_total = 0
    nonpositive_close_count = 0

    evaluation_indices = torch.tensor(
        model.config.evaluation_indices,
        dtype=torch.long,
    )

    token_selection = str(decoding_config["token_selection"])
    temperature = float(decoding_config["temperature"])
    top_k = int(decoding_config["top_k"])
    top_p = float(decoding_config["top_p"])

    progress = tqdm(
        loader,
        desc="validation generation",
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
                generated: GeneratedTokenForecast = model.generate(
                    context_tokens,
                    token_selection=token_selection,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            batch_size = int(context_tokens.shape[0])
            num_nodes = int(context_tokens.shape[2])
            generated_tokens = generated.token_ids.detach().cpu().to(torch.long)

            s1_correct_by_step += (
                (generated_tokens[..., 0] == target_s1.detach().cpu())
                .sum(dim=(0, 2))
                .to(torch.float64)
            )
            s2_correct_by_step += (
                (generated_tokens[..., 1] == target_s2.detach().cpu())
                .sum(dim=(0, 2))
                .to(torch.float64)
            )
            token_count_by_step += float(batch_size * num_nodes)
            example_count += batch_size

            graph_accumulator.add(
                generated.forecast.graph,
                batch,
                batch_size=batch_size,
            )

            if dataset.data_mode != "real":
                continue

            if tokenizer is None or raw_train_split is None:
                raise RuntimeError(
                    "Real-data validation requires a loaded tokenizer and "
                    "raw training split."
                )

            decoded_future = tokenizer.decode_token_path(
                context_tokens.detach().cpu(),
                generated_tokens,
                mean=torch.as_tensor(batch["context_mean"]),
                std=torch.as_tensor(batch["context_std"]),
                series_batch_size=decode_series_batch_size,
                return_full_path=False,
            ).to(torch.float32)

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
    s2_accuracy_by_step = s2_correct_by_step / token_count_by_step.clamp_min(1)

    generation_metrics = GenerationMetrics(
        s1_accuracy=float(
            s1_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        ),
        s2_accuracy=float(
            s2_correct_by_step.sum().item()
            / token_count_by_step.sum().clamp_min(1).item()
        ),
        s1_accuracy_by_step=s1_accuracy_by_step,
        s2_accuracy_by_step=s2_accuracy_by_step,
        examples=example_count,
        seconds=perf_counter() - start,
    )

    graph_artifacts = graph_accumulator.finalise()

    diagnostics: dict[str, Any] = {
        "generated_s1_accuracy": generation_metrics.s1_accuracy,
        "generated_s2_accuracy": generation_metrics.s2_accuracy,
        "generation_seconds": generation_metrics.seconds,
        "validation_examples": generation_metrics.examples,
    }

    if dataset.data_mode != "real":
        return ValidationBundle(
            generation_metrics=generation_metrics,
            graph_artifacts=graph_artifacts,
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

    primary_values = metric_results["cumulative_log_change_mae"]
    primary_score = float(primary_values.mean().item())

    diagnostics.update(
        {
            "primary_metric": "mean_validation_cumulative_log_change_mae",
            "primary_score": primary_score,
            "invalid_dense_candle_rate_percent": (
                100.0 * invalid_dense_count / max(invalid_dense_total, 1)
            ),
            "invalid_evaluation_candle_rate_percent": (
                100.0
                * invalid_evaluation_count
                / max(invalid_evaluation_total, 1)
            ),
            "nonpositive_predicted_close_count": nonpositive_close_count,
        }
    )

    return ValidationBundle(
        generation_metrics=generation_metrics,
        graph_artifacts=graph_artifacts,
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

    return {
        "graph_present": True,
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_nonzero_sources": float(positive.sum(dim=-1).to(torch.float64).mean().item()),
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
    atomic_json_save(
        diagnostics,
        run_dir / "best_validation_diagnostics.json",
    )


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
    }

    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(
                f"Cache/model mismatch for {name}: {observed} versus {expected}."
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


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined.")

    if args.evaluate_only and args.overwrite:
        raise ValueError("--evaluate-only and --overwrite cannot be combined.")

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

    if not args.resume and not args.evaluate_only and run_dir.exists():
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
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    gradient_clip_norm = float(training_config["gradient_clip_norm"])
    seed = int(training_config["seed"])

    if learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")
    if gradient_clip_norm <= 0:
        raise ValueError("training.gradient_clip_norm must be positive.")

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

    model = DynamicGraphTokenForecaster.from_config(resolved_config).to(device)
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

    forecasting_config = load_yaml(forecasting_config_path)
    tokenizer: KronosTokenizerAdapter | None = None
    raw_train_split: dict[str, Any] | None = None

    if data_mode == "real":
        if args.data_dir is None:
            raise ValueError("--data-dir is required when data.mode='real'.")

        data_dir = args.data_dir.expanduser().resolve()
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)

        tokenizer = KronosTokenizerAdapter.from_config(
            forecasting_config,
            series_batch_size=args.decode_series_batch_size,
        ).load()
        raw_train_split = _load_raw_training_split(
            data_dir,
            expected_asset_cols=train_dataset_full.asset_cols,
        )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
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
        "learning_rate": learning_rate,
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
        "project_git_commit": project_commit,
        "project_git_status": git_status,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "selection_metric": (
            "mean_validation_cumulative_log_change_mae"
            if data_mode == "real"
            else "validation_teacher_forced_token_loss"
        ),
    }

    signature_values = {
        "resolved_config": resolved_config,
        "train_cache": str(train_cache_path),
        "validation_cache": str(val_cache_path),
        "data_mode": data_mode,
        "asset_cols": list(train_dataset_full.asset_cols),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "project_git_commit": project_commit,
    }
    run_signature = _config_signature(signature_values)
    run_metadata["run_signature"] = run_signature

    atomic_json_save(resolved_config, run_dir / "resolved_config.json")
    atomic_json_save(run_metadata, run_dir / "run_metadata.json")

    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    history_path = run_dir / "history.csv"

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    evaluations_without_improvement = 0

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
                description="validation teacher-forced",
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
            )
            validation_teacher = run_teacher_forced_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=gradient_clip_norm,
                description=f"epoch {epoch} validation teacher-forced",
            )

            should_decode = (
                epoch == 1
                or epoch == max_epochs
                or epoch % args.validation_decode_every == 0
            )

            bundle: ValidationBundle | None = None
            primary_score: float | None = None
            improved = False

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
                )

                primary_score = (
                    float(bundle.primary_score)
                    if data_mode == "real"
                    else float(validation_teacher.total_loss)
                )

                improved = primary_score < (best_score - args.min_delta)

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
                        run_signature=run_signature,
                        resolved_config=resolved_config,
                        run_metadata=run_metadata,
                    )
                    atomic_torch_save(checkpoint, best_checkpoint_path)
                    _save_best_validation_artifacts(
                        run_dir=run_dir,
                        epoch=epoch,
                        bundle=bundle,
                        teacher_forced=validation_teacher,
                        model=model,
                    )
                else:
                    evaluations_without_improvement += 1

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_total_loss": train_metrics.total_loss,
                "train_s1_loss": train_metrics.s1_loss,
                "train_s2_loss": train_metrics.s2_loss,
                "train_s1_accuracy": train_metrics.s1_accuracy,
                "train_s2_accuracy": train_metrics.s2_accuracy,
                "validation_total_loss": validation_teacher.total_loss,
                "validation_s1_loss": validation_teacher.s1_loss,
                "validation_s2_loss": validation_teacher.s2_loss,
                "validation_s1_accuracy": validation_teacher.s1_accuracy,
                "validation_s2_accuracy": validation_teacher.s2_accuracy,
                "decoded_validation": should_decode,
                "primary_score": primary_score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "improved": improved,
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
                epoch_record.update(bundle.diagnostics)
                epoch_record.update(
                    _flatten_metric_results_for_logging(
                        bundle.metric_results,
                        model.config.heads.evaluation_horizons,
                    )
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
                f"train={train_metrics.total_loss:.6f} | "
                f"val_token={validation_teacher.total_loss:.6f} | "
                f"primary={score_text} | "
                f"best_epoch={best_epoch} | "
                f"seconds={epoch_record['epoch_seconds']:.1f}"
            )

            if (
                should_decode
                and evaluations_without_improvement >= patience
            ):
                print(
                    "Early stopping: no primary-metric improvement across "
                    f"{evaluations_without_improvement} decoded validations."
                )
                break

        if best_epoch <= 0 or not best_checkpoint_path.is_file():
            raise RuntimeError("Training finished without a best checkpoint.")

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

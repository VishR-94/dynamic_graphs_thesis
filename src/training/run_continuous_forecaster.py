from __future__ import annotations

"""Train the modular continuous-price temporal/graph forecaster.

This runner is intentionally independent of the token-generation runner.  It
uses the canonical raw candle splits and ``WindowContextNormaliser``, predicts
Close directly at [1,5,15,30,60], and never invokes the Kronos tokenizer or
decoder.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.evaluation.metrics import ForecastEvaluator
from src.evaluation.prediction_transforms import inverse_window_normalisation
from src.models.continuous_forecaster import (
    ContinuousForecaster,
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
)
from src.models.dynamic_graph.contracts import GraphConfig
from src.models.dynamic_graph.fixed_graph_resource import (
    FixedGraphResource,
    FixedGraphResourceConfig,
    fit_absolute_return_correlation_resource,
)
from src.training.run_dynamic_graph import (
    _autocast_context,
    _move_optimizer_state,
    _new_grad_scaler,
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    set_seed,
    synchronise_device,
)
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the raw/log-change continuous Transformer or ModernTCN "
            "forecaster with optional fixed/free-static graph mixing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/continuous_forecasting.yaml"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Repeatable YAML-parsed nested override.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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


def _set_nested_value(config: ConfigDict, path: str, value: Any) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("Override path must not be empty.")
    current: ConfigDict = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise KeyError(
                f"Cannot descend through override path {path!r} at {part!r}."
            )
        current = current[part]
    leaf = parts[-1]
    if leaf not in current:
        raise KeyError(f"Unknown override path {path!r}.")
    current[leaf] = value


def load_resolved_config(path: Path, expressions: Sequence[str]) -> ConfigDict:
    resolved = deepcopy(load_yaml(path))
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(
                f"Invalid --set expression {expression!r}; expected PATH=VALUE."
            )
        path_string, raw_value = expression.split("=", 1)
        _set_nested_value(
            resolved,
            path_string.strip(),
            yaml.safe_load(raw_value),
        )
    validate_config(resolved)
    return resolved


def validate_config(config: Mapping[str, Any]) -> None:
    for key in ("data", "model", "training"):
        if key not in config or not isinstance(config[key], Mapping):
            raise KeyError(f"Config must contain a {key!r} mapping.")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    if str(data["target_channel"]) != "close":
        raise ValueError("The first continuous ladder predicts Close only.")
    if str(training["loss"]["type"]) not in {
        "mse",
        "cumulative_log_change_mae",
    }:
        raise ValueError("Unsupported continuous loss type.")
    if str(training["optimizer"]) not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be adam or adamw.")
    if str(training["scheduler"]) not in {
        "none",
        "modern_tcn_type3",
    }:
        raise ValueError("Unsupported scheduler.")
    if str(training["selection_metric"]) not in {
        "validation_loss",
        "mean_short_horizon_log_mae",
    }:
        raise ValueError("Unsupported checkpoint-selection metric.")
    if str(model["graph"]["type"]) not in {
        "none",
        "fixed",
        "free_static",
    }:
        raise ValueError("Unsupported graph type for the first ladder.")


def _dataset_config(config: Mapping[str, Any]) -> ContinuousDatasetConfig:
    data = config["data"]
    normalisation = config["normalisation"]
    return ContinuousDatasetConfig(
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        stride=int(data["stride"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channels=(str(data["target_channel"]),),
        input_representation=str(data["input_representation"]),
        eps=float(normalisation["eps"]),
        clip=bool(normalisation["clip"]),
        clip_min=float(normalisation["clip_min"]),
        clip_max=float(normalisation["clip_max"]),
    )


def _model_config(
    config: Mapping[str, Any],
    *,
    num_nodes: int,
) -> ContinuousForecasterConfig:
    data = config["data"]
    model = config["model"]
    temporal = model["temporal"]
    modern = temporal["modern_tcn"]
    graph = model["graph"]
    spatial = model["spatial"]
    return ContinuousForecasterConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        temporal=ContinuousTemporalConfig(
            type=str(temporal["type"]),
            d_model=int(temporal["d_model"]),
            num_layers=int(temporal["num_layers"]),
            num_heads=int(temporal["num_heads"]),
            feedforward_multiplier=int(
                temporal["feedforward_multiplier"]
            ),
            dropout=float(temporal["dropout"]),
            relative_position_embedding=bool(
                temporal["relative_position_embedding"]
            ),
            session_position_encoding=bool(
                temporal["session_position_encoding"]
            ),
            patch_size=int(modern["patch_size"]),
            patch_stride=int(modern["patch_stride"]),
            modern_tcn_ffn_ratio=int(modern["ffn_ratio"]),
            modern_tcn_num_blocks=int(modern["num_blocks"]),
            modern_tcn_large_kernel=int(modern["large_kernel"]),
            modern_tcn_small_kernel=int(modern["small_kernel"]),
            modern_tcn_dropout=float(modern["dropout"]),
            modern_tcn_head_dropout=float(modern["head_dropout"]),
        ),
        graph=GraphConfig(
            type=str(graph["type"]),
            num_heads=int(graph["num_heads"]),
            hidden_dim=int(graph["hidden_dim"]),
            activation=str(graph["activation"]),
            add_self_loops=bool(graph["add_self_loops"]),
            mtgnn_embedding_dim=16,
            mtgnn_top_k=min(4, num_nodes - 1),
            mtgnn_alpha=3.0,
            base_graph_type="free_static",
            gate_type="none",
            initial_alpha=0.5,
        ),
        spatial_num_layers=int(spatial["num_layers"]),
        spatial_feedforward_multiplier=int(
            spatial["feedforward_multiplier"]
        ),
        spatial_dropout=float(spatial["dropout"]),
        head_dropout=float(model["head_dropout"]),
    )


def _fixed_resource_config(config: Mapping[str, Any]) -> FixedGraphResourceConfig:
    values = config["model"]["fixed_graph_resource"]
    return FixedGraphResourceConfig(
        type=str(values["type"]),
        channel=str(values["channel"]),
        threshold=float(values["threshold"]),
        empty_row_policy=str(values["empty_row_policy"]),
    )


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("Window limits must be positive.")
    return Subset(dataset, range(int(limit)))


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": _seed_worker if num_workers else None,
        "persistent_workers": bool(num_workers),
    }
    if num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _prediction_raw(
    predictions_normalised: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tensor:
    mean = torch.as_tensor(batch["target_norm_mean"]).to(
        device=device,
        dtype=predictions_normalised.dtype,
        non_blocking=True,
    )
    std = torch.as_tensor(batch["target_norm_std"]).to(
        device=device,
        dtype=predictions_normalised.dtype,
        non_blocking=True,
    )
    return inverse_window_normalisation(
        y_norm=predictions_normalised,
        target_norm_mean=mean,
        target_norm_std=std,
    )


def _loss_values(
    predictions_normalised: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    loss_type: str,
    bps_scale: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Return optimisation loss and native reporting loss."""
    target_normalised = torch.as_tensor(batch["y"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    predictions_float = predictions_normalised.float()
    if loss_type == "mse":
        native = F.mse_loss(predictions_float, target_normalised)
        return native, native

    predicted_raw = _prediction_raw(
        predictions_float,
        batch,
        device=device,
    ).float().clamp_min(eps)
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).clamp_min(eps)
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).clamp_min(eps)
    predicted_change = torch.log(predicted_raw) - torch.log(last[:, None])
    true_change = torch.log(true_raw) - torch.log(last[:, None])
    native = torch.abs(predicted_change - true_change).mean()
    return native * float(bps_scale), native


def _build_optimizer(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    training = config["training"]
    cls = (
        torch.optim.Adam
        if training["optimizer"] == "adam"
        else torch.optim.AdamW
    )
    return cls(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    config: Mapping[str, Any],
    completed_epoch: int,
) -> float:
    training = config["training"]
    base = float(training["learning_rate"])
    scheduler = str(training["scheduler"])
    if scheduler == "none":
        return float(optimizer.param_groups[0]["lr"])
    learning_rate = (
        base
        if completed_epoch < 3
        else base * (0.9 ** (completed_epoch - 3))
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _graph_summary(graph: Tensor | None) -> dict[str, float | None]:
    if graph is None:
        return {
            "mean_row_entropy": None,
            "mean_effective_neighbours": None,
            "mean_diagonal_weight": None,
        }
    values = graph.detach().float().clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    return {
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_diagonal_weight": float(diagonal.mean().item()),
    }


def _run_train_epoch(
    *,
    model: ContinuousForecaster,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    description: str,
) -> dict[str, float | int | None]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_type = str(config["training"]["loss"]["type"])
    bps_scale = float(config["training"]["loss"]["bps_scale"])
    eps = float(config["normalisation"]["eps"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    optimisation_sum = 0.0
    native_sum = 0.0
    target_count = 0
    graph_entropy_sum = 0.0
    graph_effective_sum = 0.0
    graph_diag_sum = 0.0
    graph_batches = 0
    synchronise_device(device)
    start = perf_counter()

    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with _autocast_context(device, use_amp):
            output = model(
                x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
        optimisation_loss, native_loss = _loss_values(
            output.predictions_normalised,
            batch,
            device=device,
            loss_type=loss_type,
            bps_scale=bps_scale,
            eps=eps,
        )
        if not torch.isfinite(optimisation_loss):
            raise FloatingPointError("Non-finite training loss.")
        scaler.scale(optimisation_loss).backward()
        if clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        count = int(torch.as_tensor(batch["y"]).numel())
        optimisation_sum += float(optimisation_loss.detach().item()) * count
        native_sum += float(native_loss.detach().item()) * count
        target_count += count
        summary = _graph_summary(output.graph.selected)
        if summary["mean_row_entropy"] is not None:
            graph_entropy_sum += float(summary["mean_row_entropy"])
            graph_effective_sum += float(summary["mean_effective_neighbours"])
            graph_diag_sum += float(summary["mean_diagonal_weight"])
            graph_batches += 1
        progress.set_postfix(loss=f"{native_sum / target_count:.6g}")

    synchronise_device(device)
    return {
        "optimisation_loss": optimisation_sum / target_count,
        "native_loss": native_sum / target_count,
        "target_count": target_count,
        "graph_mean_row_entropy": (
            graph_entropy_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_effective_neighbours": (
            graph_effective_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_diagonal_weight": (
            graph_diag_sum / graph_batches if graph_batches else None
        ),
        "seconds": perf_counter() - start,
    }


def _run_validation(
    *,
    model: ContinuousForecaster,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    description: str,
) -> dict[str, Any]:
    model.eval()
    loss_type = str(config["training"]["loss"]["type"])
    bps_scale = float(config["training"]["loss"]["bps_scale"])
    eps = float(config["normalisation"]["eps"])
    native_sum = 0.0
    optimisation_sum = 0.0
    target_count = 0
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    graphs: list[Tensor] = []
    days: list[str] = []
    synchronise_device(device)
    start = perf_counter()

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        ):
            x = torch.as_tensor(batch["x"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _autocast_context(device, use_amp):
                output = model(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            optimisation_loss, native_loss = _loss_values(
                output.predictions_normalised,
                batch,
                device=device,
                loss_type=loss_type,
                bps_scale=bps_scale,
                eps=eps,
            )
            raw_prediction = _prediction_raw(
                output.predictions_normalised.float(),
                batch,
                device=device,
            )
            count = int(torch.as_tensor(batch["y"]).numel())
            optimisation_sum += float(optimisation_loss.item()) * count
            native_sum += float(native_loss.item()) * count
            target_count += count
            predictions.append(raw_prediction.detach().cpu())
            targets.append(
                torch.as_tensor(batch["y_unnormalised"]).float().cpu()
            )
            last_values.append(
                torch.as_tensor(batch["last_context_target"]).float().cpu()
            )
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).cpu())
            target_indices.append(
                torch.as_tensor(batch["target_indices"]).cpu()
            )
            if output.graph.selected is not None:
                graphs.append(output.graph.selected.detach().float().cpu())
            batch_days = batch["day"]
            if isinstance(batch_days, (list, tuple)):
                days.extend(str(value) for value in batch_days)
            else:
                days.extend(str(value) for value in list(batch_days))

    prediction_result = {
        "y_pred": torch.cat(predictions, dim=0),
        "y_true": torch.cat(targets, dim=0),
        "last_context_target": torch.cat(last_values, dim=0),
        "channels": [str(config["data"]["target_channel"])],
        "horizons": [int(value) for value in config["data"]["horizons"]],
        "asset_cols": list(asset_cols),
        "sample_idx": torch.cat(sample_indices, dim=0),
        "origin_idx": torch.cat(origin_indices, dim=0),
        "target_indices": torch.cat(target_indices, dim=0),
        "output_space": "raw",
    }
    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
        train_split=dict(train_split),
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
    graph_tensor = torch.cat(graphs, dim=0) if graphs else None
    synchronise_device(device)
    return {
        "optimisation_loss": optimisation_sum / target_count,
        "native_loss": native_sum / target_count,
        "target_count": target_count,
        "prediction_result": prediction_result,
        "metric_results": metric_results,
        "metric_table": metric_table,
        "graphs": {
            "selected": graph_tensor,
            "dates": days,
            "orientation": "A[target, source]",
        },
        "graph_summary": _graph_summary(graph_tensor),
        "seconds": perf_counter() - start,
    }


def _selection_score(validation: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    method = str(config["training"]["selection_metric"])
    if method == "validation_loss":
        return float(validation["native_loss"])
    available = tuple(int(value) for value in config["data"]["horizons"])
    selected = tuple(
        int(value)
        for value in config["training"]["selection_horizons"]
    )
    metric = validation["metric_results"]["cumulative_log_change_mae"]
    indices = [available.index(horizon) for horizon in selected]
    return float(metric[indices, 0].mean().item())


def _history_record(
    *,
    epoch: int,
    learning_rate: float,
    train_metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
    selection_score: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "training_optimisation_loss": float(
            train_metrics["optimisation_loss"]
        ),
        "training_loss": float(train_metrics["native_loss"]),
        "validation_optimisation_loss": float(
            validation["optimisation_loss"]
        ),
        "validation_loss": float(validation["native_loss"]),
        "selection_score": float(selection_score),
        "training_seconds": float(train_metrics["seconds"]),
        "validation_seconds": float(validation["seconds"]),
        "graph_mean_row_entropy": validation["graph_summary"][
            "mean_row_entropy"
        ],
        "graph_mean_effective_neighbours": validation["graph_summary"][
            "mean_effective_neighbours"
        ],
        "graph_mean_diagonal_weight": validation["graph_summary"][
            "mean_diagonal_weight"
        ],
    }
    for metric_name, values in validation["metric_results"].items():
        for horizon_index, horizon in enumerate(horizons):
            record[f"val_{metric_name}_h{horizon}"] = float(
                values[horizon_index, 0].item()
            )
    return record


def _signature(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(arguments: Sequence[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _prepare_run_dir(
    output_dir: Path,
    run_name: str,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Run directory is non-empty: {run_dir}. Use --resume or --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _checkpoint(
    *,
    model: ContinuousForecaster,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    without_improvement: int,
    history: list[dict[str, Any]],
    signature: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(without_improvement),
        "history": list(history),
        "run_signature": signature,
        "resolved_config": dict(config),
        "run_metadata": dict(metadata),
        "rng_state": capture_rng_state(),
    }


def _init_wandb(args: argparse.Namespace, config: Mapping[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested but wandb is missing.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        tags=list(args.wandb_tags),
        config=dict(config),
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    resolved = load_resolved_config(config_path, args.set)
    if args.mixed_precision is not None:
        resolved["training"]["mixed_precision"] = bool(args.mixed_precision)
    device = resolve_device(args.device)
    use_amp = bool(resolved["training"]["mixed_precision"]) and device.type == "cuda"
    seed = int(resolved["training"]["seed"])
    set_seed(seed)

    data_dir = args.data_dir.expanduser().resolve()
    train_raw, val_raw, test_raw = load_candle_splits(data_dir)
    train, val, _ = clean_candle_splits(train_raw, val_raw, test_raw)
    if list(train["asset_cols"]) != list(val["asset_cols"]):
        raise ValueError("Train/validation asset order differs.")

    dataset_config = _dataset_config(resolved)
    train_dataset_full = build_continuous_dataset(
        train,
        config=dataset_config,
    )
    val_dataset_full = build_continuous_dataset(
        val,
        config=dataset_config,
    )
    train_dataset = _limit_dataset(
        train_dataset_full,
        args.max_train_windows,
    )
    val_dataset = _limit_dataset(
        val_dataset_full,
        args.max_validation_windows,
    )

    training = resolved["training"]
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=int(training["validation_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed + 1,
        pin_memory=device.type == "cuda",
    )

    model_config = _model_config(
        resolved,
        num_nodes=len(train["asset_cols"]),
    )
    fixed_resource: FixedGraphResource | None = None
    fixed_adjacency: Tensor | None = None
    if model_config.graph.type == "fixed":
        resource_config = _fixed_resource_config(resolved)
        if resource_config.type != "absolute_return_correlation":
            raise ValueError(
                "graph.type='fixed' requires the training-only absolute "
                "return-correlation resource."
            )
        fixed_resource = fit_absolute_return_correlation_resource(
            train,
            config=resource_config,
            expected_asset_cols=train["asset_cols"],
            add_self_loops=model_config.graph.add_self_loops,
        )
        fixed_adjacency = fixed_resource.adjacency

    model = ContinuousForecaster(
        model_config,
        fixed_adjacency=fixed_adjacency,
    ).to(device)
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)

    project_root = Path(__file__).resolve().parents[2]
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    signature_values = {
        "config": resolved,
        "asset_cols": list(train["asset_cols"]),
        "train_days": [str(sample[2]) for sample in train["samples"]],
        "validation_days": [str(sample[2]) for sample in val["samples"]],
        "fixed_graph_hash": (
            None if fixed_resource is None else fixed_resource.resource_hash
        ),
    }
    run_signature = _signature(signature_values)
    metadata = {
        "status": "running",
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(["rev-parse", "HEAD"], project_root),
        "git_branch": _git_value(["branch", "--show-current"], project_root),
        "device": str(device),
        "mixed_precision": use_amp,
        "train_windows": len(train_dataset),
        "validation_windows": len(val_dataset),
        "train_sessions": len(train["samples"]),
        "validation_sessions": len(val["samples"]),
        "asset_cols": list(train["asset_cols"]),
        "input_representation": dataset_config.input_representation,
        "input_channels": list(dataset_config.input_channels),
        "target_channel": dataset_config.target_channels[0],
        "temporal_backbone": model_config.temporal.type,
        "graph_type": model_config.graph.type,
        "loss_type": training["loss"]["type"],
        "normalisation": "context-only per asset/channel",
        "cross_asset_path_before_graph": False,
        "fixed_graph_resource": (
            None if fixed_resource is None else fixed_resource.metadata()
        ),
        "run_signature": run_signature,
    }
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    if fixed_resource is not None:
        atomic_torch_save(
            fixed_resource.to_payload(),
            run_dir / "fixed_graph_resource.pt",
        )
        atomic_json_save(
            fixed_resource.metadata(),
            run_dir / "fixed_graph_resource.json",
        )

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        path = run_dir / "last_checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["run_signature"] != run_signature:
            raise ValueError("Resume signature differs from the requested run.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(
            checkpoint["evaluations_without_improvement"]
        )
        history = list(checkpoint["history"])
        restore_rng_state(checkpoint["rng_state"])

    wandb_run = _init_wandb(args, resolved)
    max_epochs = int(training["max_epochs"])
    patience = int(training["patience"])
    min_delta = float(training["min_delta"])
    horizons = tuple(int(value) for value in resolved["data"]["horizons"])

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = _run_train_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                config=resolved,
                description=f"train epoch {epoch}",
            )
            validation = _run_validation(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                config=resolved,
                train_split=train,
                asset_cols=train["asset_cols"],
                description=f"validation epoch {epoch}",
            )
            score = _selection_score(validation, resolved)
            record = _history_record(
                epoch=epoch,
                learning_rate=current_lr,
                train_metrics=train_metrics,
                validation=validation,
                selection_score=score,
                horizons=horizons,
            )
            history.append(record)
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")

            improved = score < best_score - min_delta
            if improved:
                best_score = score
                best_epoch = epoch
                without_improvement = 0
                atomic_torch_save(
                    validation["prediction_result"],
                    run_dir / "best_validation_predictions.pt",
                )
                atomic_torch_save(
                    validation["graphs"],
                    run_dir / "best_validation_graphs.pt",
                )
                atomic_csv_save(
                    validation["metric_table"],
                    run_dir / "best_validation_metric_table.csv",
                )
                atomic_json_save(
                    {
                        "epoch": epoch,
                        "selection_score": score,
                        "validation_loss": validation["native_loss"],
                        "graph_summary": validation["graph_summary"],
                    },
                    run_dir / "best_validation_diagnostics.json",
                )
                best_checkpoint = _checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    without_improvement=without_improvement,
                    history=history,
                    signature=run_signature,
                    config=resolved,
                    metadata=metadata,
                )
                atomic_torch_save(
                    best_checkpoint,
                    run_dir / "best_checkpoint.pt",
                )
            else:
                without_improvement += 1

            last_checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                without_improvement=without_improvement,
                history=history,
                signature=run_signature,
                config=resolved,
                metadata=metadata,
            )
            atomic_torch_save(last_checkpoint, run_dir / "last_checkpoint.pt")

            if wandb_run is not None:
                wandb_run.log(record, step=epoch)

            print(
                f"epoch={epoch} train={train_metrics['native_loss']:.6g} "
                f"val={validation['native_loss']:.6g} "
                f"selection={score:.6g} best={best_score:.6g} "
                f"best_epoch={best_epoch}"
            )

            _adjust_learning_rate(
                optimizer,
                config=resolved,
                completed_epoch=epoch,
            )
            if without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

        metadata = dict(metadata)
        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "epochs_completed": len(history),
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
    except BaseException:
        failed = dict(metadata)
        failed.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_json_save(failed, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Train one of four dense one-step graph-supervision diagnostics.

The experiment is intentionally test-selected and must not be reported as a
held-out result.  It exists to answer a structural question: does dense
one-step supervision recover intuitive financial relationships without an
economic graph prior?
"""

import argparse
import hashlib
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_graph_supervision_dataset import (
    AlignedTokenContinuousDenseDataset,
    make_uniform_nonself_graph,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.forecast_evaluator import ForecastEvaluator
from src.models.dense_one_step_graph_controls import (
    PINNED_BASEDYGRAPH_COMMIT,
    BaseDyGraphV1ContinuousToPriceDense,
    BaseDyGraphV1TokenToPriceDense,
    ModernTCNDenseOneStepGraphModel,
    dense_basedygraph_config_from_mapping,
    modern_tcn_dense_config_from_mapping,
)
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.training.run_dynamic_graph import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    resolve_device,
    set_seed,
)
from src.utils.metric_tables import make_evaluation_table


GRAPH_ORIENTATION = "A[target, source]"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one dense one-step graph-supervision control."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
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
        default="dynamic-graph-financial-forecasting-TEST-CONTAMINATED",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return values


def _validate_config(config: Mapping[str, Any]) -> None:
    for key in ("model_family", "experiment_family", "variant", "data", "model", "training"):
        if key not in config:
            raise KeyError(f"Missing config field {key!r}.")
    if str(config["model_family"]) != "dense_graph_supervision_control":
        raise ValueError("Unexpected model_family.")
    family = str(config["experiment_family"])
    if family not in {"basedygraph_v1", "modern_tcn"}:
        raise ValueError(f"Unsupported experiment_family {family!r}.")
    training = config["training"]
    if str(training["selection_split"]) != "test":
        raise ValueError("These curiosity runs must select on the test split.")
    if str(training["selection_metric"]) != (
        "forecast_origin_h1_cumulative_log_change_mae"
    ):
        raise ValueError("Unexpected selection metric.")
    if str(training["scheduler"]) != "modern_tcn_type3_delayed":
        raise ValueError("Expected delayed type-3 learning-rate schedule.")
    if int(training["patience"]) <= 0 or int(training["max_epochs"]) <= 0:
        raise ValueError("patience and max_epochs must be positive.")
    if family == "basedygraph_v1":
        architecture = config["model"]["official_basedygraph_v1"]
        if str(architecture["spatial_module_type"]) != "dynamic_graph":
            raise ValueError("BaseDyGraph controls require dynamic_graph.")
        if str(architecture["graph_activation"]) != "softmax":
            raise ValueError("BaseDyGraph v1 uses softmax in every block.")
        if int(architecture["num_st_blocks"]) != 4:
            raise ValueError("The BaseDyGraph controls require four ST blocks.")
    else:
        variant = str(config["model"]["variant"])
        if variant not in {"dynamic_state", "random_static_dynamic_state"}:
            raise ValueError("Unexpected ModernTCN graph variant.")
        if tuple(config["data"]["input_channels"]) != ("close",):
            raise ValueError("The attached dense ModernTCN model is Close-only.")


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _new_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": generator,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _schedule_epoch(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    decay_start_epoch: int,
    decay_factor: float,
) -> None:
    multiplier = (
        1.0
        if int(epoch) <= int(decay_start_epoch)
        else float(decay_factor) ** (int(epoch) - int(decay_start_epoch))
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def _learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float | None]:
    backbone = None
    graph = None
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "backbone" if index == 0 else "graph"))
        if name == "backbone":
            backbone = float(group["lr"])
        elif name == "graph":
            graph = float(group["lr"])
    if backbone is None:
        raise RuntimeError("Optimizer has no backbone parameter group.")
    return backbone, graph


def _parameter_partition(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    graph_ids = set(model.graph_parameter_ids())  # type: ignore[attr-defined]
    graph = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in graph_ids
    ]
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in graph_ids
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(expected) != len(graph) + len(backbone):
        raise AssertionError("Optimizer partition lost parameters.")
    if {id(value) for value in graph} & {id(value) for value in backbone}:
        raise AssertionError("Optimizer parameter groups overlap.")
    return backbone, graph


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    backbone, graph = _parameter_partition(model)
    groups: list[dict[str, Any]] = [
        {
            "params": backbone,
            "lr": float(training["learning_rate"]),
            "base_lr": float(training["learning_rate"]),
            "name": "backbone",
        }
    ]
    if graph:
        groups.append(
            {
                "params": graph,
                "lr": float(training["graph_learning_rate"]),
                "base_lr": float(training["graph_learning_rate"]),
                "name": "graph",
            }
        )
    return torch.optim.Adam(groups, weight_decay=float(training["weight_decay"]))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(values: Mapping[str, Any]) -> None:
    random.setstate(values["python"])
    np.random.set_state(values["numpy"])
    torch.set_rng_state(values["torch"])
    if torch.cuda.is_available() and values.get("cuda") is not None:
        torch.cuda.set_rng_state_all(values["cuda"])


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_epoch: int | None,
    best_score: float,
    epochs_without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": [dict(row) for row in history],
        "config": dict(config),
        "rng_state": _rng_state(),
    }


def _load_checkpoint(path: Path, *, map_location: torch.device) -> dict[str, Any]:
    values = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(values, dict):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    return values


def _prepare_run_dir(
    output_dir: Path,
    run_name: str,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    run_dir = Path(output_dir) / str(run_name)
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise RuntimeError(
            "Run directory is non-empty. Use --resume or --overwrite:\n"
            f"{run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_splits(data_dir: Path):
    raw_train, raw_validation, raw_test = load_candle_splits(data_dir)
    train, validation, test = clean_candle_splits(
        raw_train,
        raw_validation,
        raw_test,
    )
    reference = list(train["asset_cols"])
    if list(validation["asset_cols"]) != reference or list(test["asset_cols"]) != reference:
        raise ValueError("Canonical split asset ordering differs.")
    return train, validation, test


def _build_datasets(
    config: Mapping[str, Any],
    *,
    splits: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    cache_paths: Mapping[str, Path],
) -> tuple[dict[str, Dataset], dict[str, Dataset]]:
    train_split, validation_split, test_split = splits
    named_splits = {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }
    family = str(config["experiment_family"])
    if family == "basedygraph_v1":
        datasets = {
            name: AlignedTokenContinuousDenseDataset(
                split=split,
                token_cache_path=cache_paths[name],
                context_length=int(config["data"]["context_length"]),
                stride=int(config["data"]["export_stride"]),
                alignment_horizons=tuple(
                    int(value) for value in config["data"]["alignment_horizons"]
                ),
                input_channels=tuple(config["data"]["input_channels"]),
            )
            for name, split in named_splits.items()
        }
        return datasets, datasets

    context_length = int(config["data"]["context_length"])
    alignment_horizons = tuple(
        int(value) for value in config["data"]["alignment_horizons"]
    )
    training_stride = int(config["training"]["one_step_training_stride"])
    export_stride = int(config["data"]["export_stride"])
    input_channels = tuple(str(value) for value in config["data"]["input_channels"])
    train_config = ContinuousDatasetConfig(
        context_length=context_length,
        horizons=(1,),
        stride=training_stride,
        input_channels=input_channels,
        target_channels=("close",),
        input_representation="raw",
        eps=float(config["normalisation"]["eps"]),
        clip=bool(config["normalisation"]["clip"]),
    )
    export_config = ContinuousDatasetConfig(
        context_length=context_length,
        horizons=alignment_horizons,
        stride=export_stride,
        input_channels=input_channels,
        target_channels=("close",),
        input_representation="raw",
        eps=float(config["normalisation"]["eps"]),
        clip=bool(config["normalisation"]["clip"]),
    )
    training_datasets: dict[str, Dataset] = {
        "train": build_continuous_dataset(train_split, config=train_config),
        "validation": build_continuous_dataset(
            validation_split,
            config=export_config,
        ),
        "test": build_continuous_dataset(test_split, config=export_config),
    }
    export_datasets = {
        name: build_continuous_dataset(split, config=export_config)
        for name, split in named_splits.items()
    }
    return training_datasets, export_datasets


def _build_model(
    config: dict[str, Any],
    *,
    num_nodes: int,
    device: torch.device,
) -> nn.Module:
    family = str(config["experiment_family"])
    if family == "basedygraph_v1":
        model_config = dense_basedygraph_config_from_mapping(
            config,
            num_nodes=num_nodes,
        )
        if model_config.input_mode == "token":
            return BaseDyGraphV1TokenToPriceDense(model_config).to(device)
        return BaseDyGraphV1ContinuousToPriceDense(model_config).to(device)

    model_config = modern_tcn_dense_config_from_mapping(
        config,
        num_nodes=num_nodes,
    )
    scaffold = (
        make_uniform_nonself_graph(num_nodes)
        if model_config.use_static_graph
        else None
    )
    model = ModernTCNDenseOneStepGraphModel(
        model_config,
        static_scaffold=scaffold,
    ).to(device)
    model.initialise_random_static_logits()
    return model


def _base_batch_values(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    device: torch.device,
):
    variant = str(config["variant"])
    if variant == "token_to_price_dynamic":
        output = model(
            torch.as_tensor(batch["context_s1"]).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ),
            first_future_s1=torch.as_tensor(batch["first_future_s1"]).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ),
        )
    else:
        output = model(
            torch.as_tensor(batch["continuous_teacher_sequence"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        )

    mean = torch.as_tensor(batch["close_norm_mean"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    std = torch.as_tensor(batch["close_norm_std"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    raw_sequence = torch.as_tensor(batch["raw_close_sequence"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    predicted_raw = (
        output.normalised_close.float()
        * std[:, None, :, None]
        + mean[:, None, :, None]
    )
    current_raw = raw_sequence[:, :-1].unsqueeze(-1)
    target_raw = raw_sequence[:, 1:].unsqueeze(-1)
    eps = 1.0e-8
    predicted_change = (
        torch.log(predicted_raw.clamp_min(eps))
        - torch.log(current_raw.clamp_min(eps))
    )
    target_change = (
        torch.log(target_raw.clamp_min(eps))
        - torch.log(current_raw.clamp_min(eps))
    )
    absolute_error = (predicted_change - target_change).abs()
    return output, predicted_raw, target_raw, absolute_error


def _modern_batch_values(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
):
    x = torch.as_tensor(batch["x"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    context_start = torch.as_tensor(batch["context_start"]).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    session_length = torch.as_tensor(batch["session_length"]).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    output = model(
        x,
        context_start=context_start,
        session_length=session_length,
    )
    target_change = torch.as_tensor(batch["target_cumulative_log_change"])[
        :, :1
    ].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    absolute_error = (output.predictions.float() - target_change).abs()
    return output, absolute_error


def _train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_error = 0.0
    total_count = 0
    bps_scale = float(config["training"]["loss"]["bps_scale"])
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            if str(config["experiment_family"]) == "basedygraph_v1":
                _, _, _, absolute_error = _base_batch_values(
                    model,
                    batch,
                    config=config,
                    device=device,
                )
            else:
                _, absolute_error = _modern_batch_values(
                    model,
                    batch,
                    device=device,
                )
            loss = absolute_error.mean() * bps_scale
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(
            model.parameters(),
            float(config["training"]["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()

        total_error += float(absolute_error.detach().double().sum().item())
        total_count += int(absolute_error.numel())
        progress.set_postfix(log_mae=total_error / max(total_count, 1))
    if total_count <= 0:
        raise RuntimeError("Training loader produced no examples.")
    return {"dense_log_mae": total_error / total_count}


def _graph_stats(values: Tensor | None) -> dict[str, float | None]:
    return graph_component_summary(None if values is None else values.float())


def _evaluate_selection(
    *,
    model: nn.Module,
    loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    family = str(config["experiment_family"])
    total_error = 0.0
    total_count = 0
    per_layer_parts: list[list[Tensor]] | None = None
    dynamic_parts: list[Tensor] = []
    selected_parts: list[Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test selection", leave=False):
            with _autocast(device, use_amp):
                if str(config["experiment_family"]) == "basedygraph_v1":
                    output, _, _, dense_error = _base_batch_values(
                        model,
                        batch,
                        config=config,
                        device=device,
                    )
                    error = dense_error[:, -1:]
                    layer_graphs = output.per_layer_graphs
                    dynamic = output.selected_graph
                    selected = output.selected_graph
                else:
                    output, error = _modern_batch_values(
                        model,
                        batch,
                        device=device,
                    )
                    layer_graphs = tuple(output.graph.per_layer)
                    dynamic = output.graph.dynamic
                    selected = output.graph.selected
            total_error += float(error.detach().double().sum().item())
            total_count += int(error.numel())
            if per_layer_parts is None:
                per_layer_parts = [[] for _ in range(len(layer_graphs))]
            for index, graph in enumerate(layer_graphs):
                if graph is None:
                    continue
                per_layer_parts[index].append(
                    torch.as_tensor(graph).detach().cpu().float()
                )
            if family != "basedygraph_v1":
                selected_parts.append(
                    torch.as_tensor(selected).detach().cpu().float()
                )
                dynamic_parts.append(
                    torch.as_tensor(dynamic).detach().cpu().float()
                )
    if total_count <= 0 or per_layer_parts is None:
        raise RuntimeError("Selection loader produced no examples.")
    per_layer = tuple(
        torch.cat(parts, dim=0) if parts else None
        for parts in per_layer_parts
    )
    if family == "basedygraph_v1":
        final_graph = per_layer[-1]
        if final_graph is None:
            raise RuntimeError("The final BaseDyGraph block returned no graph.")
        dynamic_graph = final_graph
    else:
        final_graph = torch.cat(selected_parts, dim=0)
        dynamic_graph = torch.cat(dynamic_parts, dim=0)
    return {
        "h1_log_mae": total_error / total_count,
        "per_layer": per_layer,
        "selected_graph": final_graph,
        "dynamic_graph": dynamic_graph,
        "final_graph_summary": _graph_stats(final_graph),
    }


def _batch_dates(batch: Mapping[str, Any]) -> list[str]:
    values = batch["day"]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _export_split(
    *,
    model: nn.Module,
    dataset: Dataset,
    split_name: str,
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    config: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    loader = _build_loader(
        dataset,
        batch_size=int(config["training"]["export_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    model.eval()
    family = str(config["experiment_family"])
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    lasts: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    dates: list[str] = []
    per_layer_parts: list[list[Tensor]] | None = None
    dynamic_parts: list[Tensor] = []
    selected_parts: list[Tensor] = []
    singleton_static: Tensor | None = None
    invalid_prediction_count = 0
    prediction_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"export {split_name}", leave=False):
            with _autocast(device, use_amp):
                if str(config["experiment_family"]) == "basedygraph_v1":
                    output, predicted_dense, target_dense, _ = _base_batch_values(
                        model,
                        batch,
                        config=config,
                        device=device,
                    )
                    predicted_raw = predicted_dense[:, -1:]
                    true_raw = target_dense[:, -1:]
                    last = torch.as_tensor(batch["last_context_close"]).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    layer_graphs = output.per_layer_graphs
                    selected_graph = output.selected_graph
                    dynamic_graph = output.selected_graph
                    static_graph = None
                else:
                    output, _ = _modern_batch_values(
                        model,
                        batch,
                        device=device,
                    )
                    predicted_change = output.predictions.float()
                    last = torch.as_tensor(batch["last_context_target"]).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    predicted_raw = last.unsqueeze(1) * torch.exp(predicted_change)
                    true_raw = torch.as_tensor(batch["y_unnormalised"])[
                        :, :1
                    ].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    layer_graphs = tuple(output.graph.per_layer)
                    selected_graph = output.graph.selected
                    dynamic_graph = output.graph.dynamic
                    static_graph = output.graph.base

            invalid_prediction_count += int(
                (~torch.isfinite(predicted_raw) | (predicted_raw <= 0)).sum().item()
            )
            prediction_count += int(predicted_raw.numel())
            predictions.append(predicted_raw.detach().cpu().float())
            targets.append(true_raw.detach().cpu().float())
            lasts.append(last.detach().cpu().float())
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu().long())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).cpu().long())
            target_indices.append(
                torch.as_tensor(batch["target_indices"])[:, :1].cpu().long()
            )
            dates.extend(_batch_dates(batch))

            if per_layer_parts is None:
                per_layer_parts = [[] for _ in range(len(layer_graphs))]
            for index, graph in enumerate(layer_graphs):
                if graph is None:
                    continue
                per_layer_parts[index].append(
                    torch.as_tensor(graph)
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
            if family != "basedygraph_v1":
                selected_parts.append(
                    torch.as_tensor(selected_graph)
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
                dynamic_parts.append(
                    torch.as_tensor(dynamic_graph)
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
            if static_graph is not None and singleton_static is None:
                singleton_static = (
                    torch.as_tensor(static_graph)
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )

    if per_layer_parts is None:
        raise RuntimeError(f"No {split_name} export batches were produced.")
    y_pred = torch.cat(predictions, dim=0)
    y_true = torch.cat(targets, dim=0)
    last_context = torch.cat(lasts, dim=0)
    prediction_result = {
        "y_pred": y_pred,
        "y_true": y_true,
        "last_context_target": last_context,
        "channels": ["close"],
        "horizons": [1],
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

    per_layer = tuple(
        torch.cat(parts, dim=0) if parts else None
        for parts in per_layer_parts
    )
    if family == "basedygraph_v1":
        selected = per_layer[-1]
        if selected is None:
            raise RuntimeError("The final BaseDyGraph block returned no graph.")
        dynamic = selected
    else:
        selected = torch.cat(selected_parts, dim=0)
        dynamic = torch.cat(dynamic_parts, dim=0)
    layer_count = len(per_layer)
    graph_heads = int(
        config["model"]["official_basedygraph_v1"]["graph_num_heads"]
        if family == "basedygraph_v1"
        else config["model"]["graph"]["num_heads"]
    )
    saved_static = (
        None
        if singleton_static is None
        else singleton_static[0].contiguous()
    )
    alpha = (
        None
        if not hasattr(model, "alpha") or model.alpha() is None
        else model.alpha().detach().cpu().float().reshape(1)  # type: ignore[attr-defined]
    )
    beta = (
        None
        if not hasattr(model, "beta")
        else model.beta().detach().cpu().float().reshape(1)  # type: ignore[attr-defined]
    )
    graph_artifacts = {
        "graph_type": (
            "dynamic"
            if saved_static is None
            else "static_dynamic_mixture"
        ),
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(asset_cols),
        "num_layers": layer_count,
        "num_heads": graph_heads,
        "num_heads_per_layer": [graph_heads] * layer_count,
        "layer_head_counts": [graph_heads] * layer_count,
        "graph_activations_per_layer": ["softmax"] * layer_count,
        "selected_layer": layer_count - 1,
        "selected": selected,
        "per_layer": per_layer,
        "base": saved_static,
        "per_layer_base": tuple(
            [None] * (layer_count - 1) + [saved_static]
        ),
        "dynamic": dynamic,
        "per_layer_dynamic": (
            per_layer
            if family == "basedygraph_v1"
            else (dynamic,)
        ),
        "alpha": alpha,
        "alpha_per_layer": tuple(
            [None] * (layer_count - 1) + [alpha]
        ),
        "beta": beta,
        "beta_per_layer": (
            None
            if beta is None
            else tuple([None] * (layer_count - 1) + [beta])
        ),
        "dynamic_alpha": None if alpha is None else float(alpha.item()),
        "spatial_beta": None if beta is None else float(beta.item()),
        "spatial_gate_type": "none" if beta is None else "learned_scalar",
        "beta_trainable": beta is not None,
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "diagonal_policy": "eligible in scorer softmax; no extra identity matrix",
    }
    selected_summary = _graph_stats(selected.float())
    static_summary = _graph_stats(
        None if saved_static is None else saved_static.float()
    )
    dynamic_summary = _graph_stats(dynamic.float())
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(y_pred.shape[0]),
        "horizons": [1],
        "assets": int(y_pred.shape[2]),
        "input_mode": config["data"].get("input_mode", "continuous_close"),
        "dense_training_semantics": (
            "teacher-forced next-Close at every context transition"
            if family == "basedygraph_v1"
            else "stride-1 one-step Close forecasting windows"
        ),
        "invalid_prediction_fraction": (
            float(invalid_prediction_count) / max(prediction_count, 1)
        ),
        "alpha": graph_artifacts["dynamic_alpha"],
        "beta": graph_artifacts["spatial_beta"],
        "selected_graph": selected_summary,
        "static_graph": static_summary,
        "dynamic_graph": dynamic_summary,
        "blocks": [
            {
                "block": index,
                "selected_graph": _graph_stats(
                    None if per_layer[index] is None else per_layer[index].float()
                ),
                "activation": "softmax",
                "heads": graph_heads,
            }
            for index in range(layer_count)
        ],
        "graph_orientation": GRAPH_ORIENTATION,
    }
    return {
        "prediction_result": prediction_result,
        "graph_artifacts": graph_artifacts,
        "metric_table": metric_table,
        "diagnostics": diagnostics,
    }


def _save_export(
    run_dir: Path,
    *,
    split_name: str,
    values: Mapping[str, Any],
) -> None:
    epoch = int(values["diagnostics"]["checkpoint_epoch"])
    prediction_path = run_dir / f"best_{split_name}_predictions.pt"
    graph_path = run_dir / f"best_{split_name}_graphs.pt"
    metric_path = run_dir / f"best_{split_name}_metric_table.csv"
    diagnostics_path = run_dir / f"best_{split_name}_diagnostics.json"
    atomic_torch_save(
        {"epoch": epoch, "prediction_result": values["prediction_result"]},
        prediction_path,
    )
    atomic_torch_save(
        {"epoch": epoch, "graph_artifacts": values["graph_artifacts"]},
        graph_path,
    )
    atomic_csv_save(values["metric_table"], metric_path)
    atomic_json_save(values["diagnostics"], diagnostics_path)

    analysis_dir = run_dir / "analysis" / split_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction_path, analysis_dir / "predictions.pt")
    shutil.copy2(graph_path, analysis_dir / "graphs.pt")
    shutil.copy2(metric_path, analysis_dir / "metric_table.csv")
    shutil.copy2(diagnostics_path, analysis_dir / "diagnostics.json")


def _init_wandb(args: argparse.Namespace, config: Mapping[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("wandb is required for enabled logging.") from error
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=dict(config),
        tags=list(args.wandb_tags),
        mode=args.wandb_mode,
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_json(args.config)
    _validate_config(config)
    train_split, validation_split, test_split = _load_splits(args.data_dir)
    splits = (train_split, validation_split, test_split)
    cache_paths = {
        "train": args.train_cache,
        "validation": args.validation_cache,
        "test": args.test_cache,
    }
    training_datasets, export_datasets = _build_datasets(
        config,
        splits=splits,
        cache_paths=cache_paths,
    )
    for name, dataset in training_datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"The configured {name} dataset has no windows.")

    device = resolve_device(args.device)
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    set_seed(int(config["training"]["seed"]))
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    asset_cols = list(train_split["asset_cols"])
    model = _build_model(
        config,
        num_nodes=len(asset_cols),
        device=device,
    )
    optimizer = _build_optimizer(model, config)
    scaler = _new_scaler(use_amp)
    backbone_parameters, graph_parameters = _parameter_partition(model)

    signature_payload = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run_signature = hashlib.sha256(signature_payload).hexdigest()
    try:
        project_git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except Exception:
        project_git_commit = None

    metadata: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "run_signature": run_signature,
        "project_git_commit": project_git_commit,
        "model_family": "dense_graph_supervision_control",
        "experiment_family": config["experiment_family"],
        "variant": config["variant"],
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": config["training"]["selection_metric"],
        "asset_cols": asset_cols,
        "context_length": int(config["data"]["context_length"]),
        "reported_horizons": [1],
        "train_windows": len(training_datasets["train"]),
        "export_train_windows": len(export_datasets["train"]),
        "validation_windows": len(export_datasets["validation"]),
        "test_windows": len(export_datasets["test"]),
        "dense_targets_per_training_window": (
            int(config["data"]["context_length"])
            if str(config["experiment_family"]) == "basedygraph_v1"
            else 1
        ),
        "dense_training_scalar_targets_per_epoch": (
            len(training_datasets["train"])
            * (
                int(config["data"]["context_length"])
                if str(config["experiment_family"]) == "basedygraph_v1"
                else 1
            )
            * len(asset_cols)
        ),
        "forecast_origin_selection_horizon": 1,
        "optimizer": config["training"]["optimizer"],
        "scheduler": config["training"]["scheduler"],
        "learning_rate": float(config["training"]["learning_rate"]),
        "graph_learning_rate": float(config["training"]["graph_learning_rate"]),
        "scheduler_decay_start_epoch": int(
            config["training"]["scheduler_decay_start_epoch"]
        ),
        "scheduler_decay_factor": float(
            config["training"]["scheduler_decay_factor"]
        ),
        "mixed_precision": bool(config["training"]["mixed_precision"]),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "backbone_trainable_parameters": int(
            sum(parameter.numel() for parameter in backbone_parameters)
        ),
        "graph_trainable_parameters": int(
            sum(parameter.numel() for parameter in graph_parameters)
        ),
        "graph_orientation": GRAPH_ORIENTATION,
        "graph_regularisation": "none",
    }
    if str(config["experiment_family"]) == "basedygraph_v1":
        metadata.update(
            {
                "basedygraph_expected_commit": PINNED_BASEDYGRAPH_COMMIT,
                "basedygraph_observed_commit": getattr(model, "external_commit", None),
                "num_st_blocks": int(
                    config["model"]["official_basedygraph_v1"]["num_st_blocks"]
                ),
                "graph_type": "dynamic",
                "prior_type": "none",
                "alpha": None,
                "beta": None,
            }
        )
    else:
        metadata.update(
            {
                "num_st_blocks": 1,
                "graph_type": config["model"]["graph"]["type"],
                "prior_type": config["model"]["prior"]["type"],
                "state_pathway": True,
                "initial_alpha": (
                    None
                    if not hasattr(model, "alpha") or model.alpha() is None
                    else float(model.alpha().item())
                ),
                "initial_beta": float(model.beta().item()),
            }
        )

    if (
        str(config["experiment_family"]) == "modern_tcn"
        and hasattr(model, "graph_learner")
        and model.graph_learner.static_adjacency() is not None
    ):
        initial_static = (
            model.graph_learner.static_adjacency()[0]
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
        atomic_torch_save(
            {
                "prior_type": "random",
                "adjacency": initial_static,
                "asset_cols": asset_cols,
                "orientation": GRAPH_ORIENTATION,
                "fitted_on": "random trainable logits; no economic prior",
            },
            run_dir / "initial_graph_prior.pt",
        )
        pd.DataFrame(
            initial_static.mean(dim=0).numpy(),
            index=asset_cols,
            columns=asset_cols,
        ).to_csv(run_dir / "initial_graph_prior.csv")

    atomic_json_save(config, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        "This run uses the test split for checkpoint selection.\n",
        encoding="utf-8",
    )

    start_epoch = 1
    best_epoch: int | None = None
    best_score = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = _load_checkpoint(
            run_dir / "last_checkpoint.pt",
            map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = checkpoint["best_epoch"]
        best_score = float(checkpoint["best_score"])
        epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )
        history = [dict(row) for row in checkpoint["history"]]
        _restore_rng_state(checkpoint["rng_state"])

    train_loader = _build_loader(
        training_datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    test_loader = _build_loader(
        export_datasets["test"],
        batch_size=int(config["training"]["selection_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        seed=int(config["training"]["seed"]),
    )
    wandb_run = _init_wandb(args, config)

    try:
        for epoch in range(start_epoch, int(config["training"]["max_epochs"]) + 1):
            _schedule_epoch(
                optimizer,
                epoch=epoch,
                decay_start_epoch=int(
                    config["training"]["scheduler_decay_start_epoch"]
                ),
                decay_factor=float(
                    config["training"]["scheduler_decay_factor"]
                ),
            )
            training_values = _train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
            )
            selection = _evaluate_selection(
                model=model,
                loader=test_loader,
                config=config,
                device=device,
                use_amp=use_amp,
            )
            backbone_lr, graph_lr = _learning_rates(optimizer)
            alpha = (
                None
                if not hasattr(model, "alpha") or model.alpha() is None
                else float(model.alpha().item())
            )
            beta = (
                None
                if not hasattr(model, "beta")
                else float(model.beta().item())
            )
            row: dict[str, Any] = {
                "epoch": int(epoch),
                "backbone_learning_rate": backbone_lr,
                "graph_learning_rate": graph_lr,
                "train_dense_log_mae": float(training_values["dense_log_mae"]),
                "test_h1_log_mae": float(selection["h1_log_mae"]),
                "selection_score": float(selection["h1_log_mae"]),
                "alpha": alpha,
                "beta": beta,
                "test_final_graph_entropy": selection["final_graph_summary"].get(
                    "mean_row_entropy"
                ),
                "test_final_graph_effective_neighbours": selection[
                    "final_graph_summary"
                ].get("mean_effective_neighbours"),
            }
            for index, graph in enumerate(selection["per_layer"]):
                summary = _graph_stats(graph)
                row[f"block_{index}_selected_entropy"] = summary.get(
                    "mean_row_entropy"
                )
                row[f"block_{index}_selected_effective_neighbours"] = summary.get(
                    "mean_effective_neighbours"
                )
            history.append(row)
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")

            improved = float(selection["h1_log_mae"]) < (
                best_score - float(config["training"]["min_delta"])
            )
            if improved:
                best_score = float(selection["h1_log_mae"])
                best_epoch = int(epoch)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            state = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_epoch=best_epoch,
                best_score=best_score,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                config=config,
            )
            atomic_torch_save(state, run_dir / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(state, run_dir / "best_checkpoint.pt")

            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            print(
                f"Epoch {epoch}: train dense Log MAE="
                f"{training_values['dense_log_mae']:.8f}; "
                f"test h1 Log MAE={selection['h1_log_mae']:.8f}; "
                f"alpha={alpha}; beta={beta}"
            )
            if epochs_without_improvement >= int(config["training"]["patience"]):
                break

        if best_epoch is None or not (run_dir / "best_checkpoint.pt").is_file():
            raise RuntimeError("Training produced no selected checkpoint.")
        best = _load_checkpoint(run_dir / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(best["model_state_dict"], strict=True)
        best_epoch = int(best["epoch"])

        for split_name in ("train", "validation", "test"):
            values = _export_split(
                model=model,
                dataset=export_datasets[split_name],
                split_name=split_name,
                train_split=train_split,
                asset_cols=asset_cols,
                config=config,
                device=device,
                use_amp=use_amp,
                checkpoint_epoch=best_epoch,
            )
            _save_export(run_dir, split_name=split_name, values=values)

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_epoch": best_epoch,
                "best_score": float(best_score),
                "epochs_completed": int(history[-1]["epoch"]),
                "final_alpha": (
                    None
                    if not hasattr(model, "alpha") or model.alpha() is None
                    else float(model.alpha().item())
                ),
                "final_beta": (
                    None
                    if not hasattr(model, "beta")
                    else float(model.beta().item())
                ),
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
    except Exception:
        metadata["status"] = "failed"
        metadata["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

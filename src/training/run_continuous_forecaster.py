from __future__ import annotations

"""Train the modular continuous-price temporal/graph forecaster.

This runner is intentionally independent of the token-generation runner.  It
uses the canonical raw candle splits and ``WindowContextNormaliser`` and
never invokes the Kronos tokenizer or decoder.  The output representation is
explicitly configurable: legacy normalised Close levels or direct cumulative
Close log changes at [1,5,15,30,60].
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
from src.evaluation.prediction_transforms import (
    cumulative_log_change_to_raw,
    inverse_window_normalisation,
    raw_to_cumulative_log_change,
)
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
from src.models.dynamic_graph.losses import (
    GraphRegularisationConfig,
    compute_graph_regularisation,
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
            "forecaster with optional fixed, static, dynamic, or "
            "dynamic-base graph mixing."
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
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive.")
    if float(training["graph_learning_rate"]) <= 0.0:
        raise ValueError("training.graph_learning_rate must be positive.")
    if int(training["graph_diagnostics_batches_per_epoch"]) < 0:
        raise ValueError(
            "training.graph_diagnostics_batches_per_epoch cannot be negative."
        )
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
    output_representation = str(model["output_representation"])
    if output_representation not in {
        "normalised_close",
        "cumulative_log_change",
    }:
        raise ValueError("Unsupported model.output_representation.")
    output_initialisation = str(model["output_head_initialisation"])
    if output_initialisation not in {"default", "zero"}:
        raise ValueError("Unsupported model.output_head_initialisation.")
    if (
        output_representation == "cumulative_log_change"
        and str(training["loss"]["type"])
        != "cumulative_log_change_mae"
    ):
        raise ValueError(
            "Direct cumulative-log-change output currently requires "
            "training.loss.type=cumulative_log_change_mae."
        )
    graph_type = str(model["graph"]["type"])
    if graph_type not in {
        "none",
        "fixed",
        "free_static",
        "dynamic",
        "dynamic_correlation",
        "dynamic_base",
    }:
        raise ValueError("Unsupported graph type for continuous forecasting.")

    dynamic_correlation = model.get(
        "dynamic_correlation",
        {
            "threshold": None,
            "empty_row_policy": "strongest",
            "eps": 1.0e-8,
        },
    )
    if graph_type == "dynamic_correlation":
        threshold = dynamic_correlation.get("threshold")
        if threshold is not None and not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                "model.dynamic_correlation.threshold must be null or lie "
                "in [0,1]."
            )
        if str(dynamic_correlation.get("empty_row_policy")) not in {
            "error",
            "strongest",
        }:
            raise ValueError(
                "model.dynamic_correlation.empty_row_policy must be "
                "'error' or 'strongest'."
            )
        if float(dynamic_correlation.get("eps", 0.0)) <= 0.0:
            raise ValueError(
                "model.dynamic_correlation.eps must be positive."
            )
        if str(model["graph"]["activation"]) != "softmax":
            raise ValueError(
                "dynamic_correlation requires graph.activation='softmax'."
            )

    spatial = model["spatial"]
    if str(spatial["gate_type"]) not in {
        "none",
        "fixed",
        "learned_scalar",
    }:
        raise ValueError("Unsupported spatial gate type.")
    initial_beta = float(spatial["initial_beta"])
    if not 0.0 <= initial_beta <= 1.0:
        raise ValueError("model.spatial.initial_beta must lie in [0,1].")
    if graph_type == "none" and str(spatial["gate_type"]) != "none":
        raise ValueError(
            "Graph-free models must set model.spatial.gate_type=none."
        )

    graph_regularisation = GraphRegularisationConfig.from_mapping(
        model.get("graph_regularisation")
    )
    if graph_regularisation.graph_temporal_smooth_reg > 0.0:
        raise ValueError(
            "The continuous model emits one graph per context window, not "
            "a graph sequence; graph temporal smoothing must remain zero."
        )
    if graph_regularisation.enabled and graph_type != "free_static":
        raise ValueError(
            "Graph regularisation is supported only for the learned "
            "free-static graph in this ladder."
        )


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
    dynamic_correlation = model.get(
        "dynamic_correlation",
        {
            "threshold": None,
            "empty_row_policy": "strongest",
            "eps": 1.0e-8,
        },
    )
    spatial = model["spatial"]
    return ContinuousForecasterConfig(
        num_nodes=int(num_nodes),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        output_representation=str(model["output_representation"]),
        output_head_initialisation=str(
            model["output_head_initialisation"]
        ),
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
            base_graph_type=str(graph["base_graph_type"]),
            gate_type=str(graph["gate_type"]),
            initial_alpha=float(graph["initial_alpha"]),
        ),
        dynamic_correlation_threshold=(
            None
            if dynamic_correlation.get("threshold") is None
            else float(dynamic_correlation["threshold"])
        ),
        dynamic_correlation_empty_row_policy=str(
            dynamic_correlation.get("empty_row_policy", "strongest")
        ),
        dynamic_correlation_eps=float(
            dynamic_correlation.get("eps", 1.0e-8)
        ),
        spatial_num_layers=int(spatial["num_layers"]),
        spatial_feedforward_multiplier=int(
            spatial["feedforward_multiplier"]
        ),
        spatial_dropout=float(spatial["dropout"]),
        spatial_gate_type=str(spatial["gate_type"]),
        spatial_gate_initial_beta=float(spatial["initial_beta"]),
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
    predictions: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    output_representation: str,
) -> Tensor:
    predictions_float = predictions.float()
    if output_representation == "cumulative_log_change":
        last = torch.as_tensor(batch["last_context_target"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        return cumulative_log_change_to_raw(
            cumulative_log_change=predictions_float,
            last_context_target=last,
        )

    if output_representation != "normalised_close":
        raise ValueError(
            f"Unsupported output representation: {output_representation!r}."
        )
    mean = torch.as_tensor(batch["target_norm_mean"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    std = torch.as_tensor(batch["target_norm_std"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return inverse_window_normalisation(
        y_norm=predictions_float,
        target_norm_mean=mean,
        target_norm_std=std,
    )


def _loss_values(
    predictions: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    output_representation: str,
    loss_type: str,
    bps_scale: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Return optimisation loss and native reporting loss.

    ``native`` is always in the unscaled units used for checkpoint selection
    and history.  For direct cumulative-log-change output it is exactly the
    pointwise Log MAE objective.
    """
    predictions_float = predictions.float()

    if output_representation == "cumulative_log_change":
        if loss_type != "cumulative_log_change_mae":
            raise ValueError(
                "Direct cumulative-log-change output requires the "
                "cumulative_log_change_mae loss."
            )
        target_change = torch.as_tensor(
            batch["target_cumulative_log_change"]
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        native = F.l1_loss(predictions_float, target_change)
        return native * float(bps_scale), native

    if output_representation != "normalised_close":
        raise ValueError(
            f"Unsupported output representation: {output_representation!r}."
        )

    target_normalised = torch.as_tensor(batch["y"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if loss_type == "mse":
        native = F.mse_loss(predictions_float, target_normalised)
        return native, native

    predicted_raw = _prediction_raw(
        predictions_float,
        batch,
        device=device,
        output_representation=output_representation,
    ).clamp_min(eps)
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    predicted_change = raw_to_cumulative_log_change(
        predicted_raw,
        last,
        eps=eps,
    )
    true_change = raw_to_cumulative_log_change(
        true_raw,
        last,
        eps=eps,
    )
    native = F.l1_loss(predicted_change, true_change)
    return native * float(bps_scale), native

def _trainable_parameter_partition(
    model: ContinuousForecaster,
) -> tuple[list[Tensor], list[Tensor]]:
    """Return disjoint backbone and graph-learner parameter lists.

    Only parameters owned by ``model.graph_learner`` receive the dedicated
    graph learning rate. Spatial-message-passing parameters and the optional
    spatial beta gate remain in the backbone group because they transform
    representations rather than parameterise the adjacency itself.
    """
    graph_parameters = [
        parameter
        for parameter in model.graph_learner.parameters()
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
    model: ContinuousForecaster,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    training = config["training"]
    cls = (
        torch.optim.Adam
        if training["optimizer"] == "adam"
        else torch.optim.AdamW
    )
    backbone_parameters, graph_parameters = _trainable_parameter_partition(model)
    backbone_lr = float(training["learning_rate"])
    graph_lr = float(training["graph_learning_rate"])
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": backbone_parameters,
            "lr": backbone_lr,
            "base_lr": backbone_lr,
            "name": "backbone",
        }
    ]
    if graph_parameters:
        parameter_groups.append(
            {
                "params": graph_parameters,
                "lr": graph_lr,
                "base_lr": graph_lr,
                "name": "graph",
            }
        )
    return cls(
        parameter_groups,
        weight_decay=float(training["weight_decay"]),
    )


def _current_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "backbone": None,
        "graph": None,
    }
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "backbone" if index == 0 else f"group_{index}"))
        if name in values:
            values[name] = float(group["lr"])
    if values["backbone"] is None:
        raise RuntimeError("Optimizer is missing its backbone parameter group.")
    return values


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    config: Mapping[str, Any],
    completed_epoch: int,
) -> dict[str, float | None]:
    scheduler = str(config["training"]["scheduler"])
    if scheduler == "none":
        return _current_learning_rates(optimizer)
    multiplier = (
        1.0
        if completed_epoch < 3
        else 0.9 ** (completed_epoch - 3)
    )
    for group in optimizer.param_groups:
        if "base_lr" not in group:
            raise RuntimeError(
                "Optimizer parameter group is missing base_lr; cannot preserve "
                "the backbone-to-graph learning-rate ratio."
            )
        group["lr"] = float(group["base_lr"]) * multiplier
    return _current_learning_rates(optimizer)


def _gradient_norm_from_tensors(
    gradients: Sequence[Tensor | None],
) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        squared += float(gradient.detach().float().square().sum().item())
    return squared ** 0.5


def _parameter_gradient_norm(parameters: Sequence[Tensor]) -> float:
    return _gradient_norm_from_tensors(
        [parameter.grad for parameter in parameters]
    )


def _parameter_update_norm(
    before: Sequence[Tensor],
    parameters: Sequence[Tensor],
) -> float:
    if len(before) != len(parameters):
        raise ValueError("Parameter snapshot length differs from parameter list.")
    squared = 0.0
    for previous, parameter in zip(before, parameters, strict=True):
        difference = parameter.detach().float() - previous.float()
        squared += float(difference.square().sum().item())
    return squared ** 0.5


def _graph_summary(graph: Tensor | None) -> dict[str, float | None]:
    if graph is None:
        return {
            "mean_row_entropy": None,
            "mean_effective_neighbours": None,
            "mean_diagonal_weight": None,
            "maximum_edge_weight": None,
            "mean_top10_row_mass": None,
        }
    values = graph.detach().float().clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    top_k = min(10, int(values.shape[-1]))
    top10_mass = values.topk(top_k, dim=-1).values.sum(dim=-1)
    return {
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_diagonal_weight": float(diagonal.mean().item()),
        "maximum_edge_weight": float(values.max().item()),
        "mean_top10_row_mass": float(top10_mass.mean().item()),
    }


def _module_gradient_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    squared = 0.0
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        squared += float(gradient.detach().float().square().sum().item())
    return squared ** 0.5


def _scalar_value(value: Tensor | None) -> float | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().float()
    if tensor.numel() == 0:
        return None
    return float(tensor.mean().item())


def _expand_graph_component(
    graph: Tensor | None,
    *,
    batch_size: int,
) -> Tensor | None:
    if graph is None:
        return None
    values = torch.as_tensor(graph).detach().float().cpu()
    if values.ndim != 4:
        raise ValueError(
            "Graph components must have shape [B,G,N,N] or [1,G,N,N]."
        )
    if int(values.shape[0]) == 1 and batch_size > 1:
        values = values.expand(batch_size, -1, -1, -1)
    if int(values.shape[0]) != batch_size:
        raise ValueError(
            f"Graph batch axis is {int(values.shape[0])}; expected {batch_size}."
        )
    return values.contiguous()


def _graph_context_values(
    batch: Mapping[str, Any],
    *,
    model: ContinuousForecaster,
    device: torch.device,
) -> Tensor | None:
    """Return observed raw Close levels for dynamic correlation.

    The dataset field is derived only from the 60 observed context rows.
    Other graph types return ``None`` without transferring the extra tensor.
    """
    if model.config.graph.type != "dynamic_correlation":
        return None
    if "context_target_unnormalised" not in batch:
        raise KeyError(
            "The continuous dataset did not provide "
            "context_target_unnormalised required by dynamic_correlation."
        )
    values = torch.as_tensor(
        batch["context_target_unnormalised"]
    ).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if values.ndim != 4 or int(values.shape[-1]) != 1:
        raise ValueError(
            "context_target_unnormalised must have shape [B,T,N,1]."
        )
    return values[..., 0]


def _run_train_epoch(
    *,
    model: ContinuousForecaster,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    current_epoch: int,
    description: str,
) -> dict[str, float | int | None]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_type = str(config["training"]["loss"]["type"])
    bps_scale = float(config["training"]["loss"]["bps_scale"])
    eps = float(config["normalisation"]["eps"])
    output_representation = str(config["model"]["output_representation"])
    graph_regularisation_config = GraphRegularisationConfig.from_mapping(
        config["model"]["graph_regularisation"]
    )
    clip_norm = float(config["training"]["gradient_clip_norm"])
    diagnostics_batch_limit = int(
        config["training"]["graph_diagnostics_batches_per_epoch"]
    )
    _, graph_parameters = _trainable_parameter_partition(model)

    optimisation_sum = 0.0
    forecast_optimisation_sum = 0.0
    graph_regularisation_sum = 0.0
    graph_target_entropy_penalty_sum = 0.0
    native_sum = 0.0
    target_count = 0

    graph_entropy_sum = 0.0
    graph_effective_sum = 0.0
    graph_diag_sum = 0.0
    graph_max_edge_sum = 0.0
    graph_top10_mass_sum = 0.0
    graph_batches = 0
    spatial_beta_sum = 0.0
    spatial_beta_batches = 0
    dynamic_alpha_sum = 0.0
    dynamic_alpha_batches = 0

    combined_graph_gradient_norm_sum = 0.0
    combined_graph_gradient_batches = 0
    forecast_graph_gradient_norm_sum = 0.0
    regulariser_graph_gradient_norm_sum = 0.0
    graph_parameter_update_norm_sum = 0.0
    diagnostic_graph_batches = 0
    spatial_gate_gradient_norm_sum = 0.0
    spatial_gate_gradient_batches = 0

    synchronise_device(device)
    start = perf_counter()

    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for batch_index, batch in enumerate(progress):
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
                graph_context_values=_graph_context_values(
                    batch,
                    model=model,
                    device=device,
                ),
            )
        forecast_optimisation_loss, native_loss = _loss_values(
            output.predictions,
            batch,
            device=device,
            output_representation=output_representation,
            loss_type=loss_type,
            bps_scale=bps_scale,
            eps=eps,
        )
        graph_regularisation = compute_graph_regularisation(
            output.graph,
            config=graph_regularisation_config,
            current_epoch=current_epoch,
            reference_tensor=forecast_optimisation_loss,
        )
        optimisation_loss = (
            forecast_optimisation_loss + graph_regularisation.total
        )
        if not torch.isfinite(optimisation_loss):
            raise FloatingPointError("Non-finite training loss.")

        run_diagnostics = (
            bool(graph_parameters)
            and batch_index < diagnostics_batch_limit
        )
        graph_snapshot: list[Tensor] | None = None
        if run_diagnostics:
            forecast_gradients = torch.autograd.grad(
                forecast_optimisation_loss,
                graph_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if graph_regularisation.total.requires_grad:
                regulariser_gradients = torch.autograd.grad(
                    graph_regularisation.total,
                    graph_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            else:
                regulariser_gradients = tuple(None for _ in graph_parameters)
            forecast_graph_gradient_norm_sum += _gradient_norm_from_tensors(
                forecast_gradients
            )
            regulariser_graph_gradient_norm_sum += _gradient_norm_from_tensors(
                regulariser_gradients
            )
            graph_snapshot = [
                parameter.detach().float().clone()
                for parameter in graph_parameters
            ]

        scaler.scale(optimisation_loss).backward()
        scaler.unscale_(optimizer)

        if graph_parameters:
            combined_graph_gradient_norm_sum += _parameter_gradient_norm(
                graph_parameters
            )
            combined_graph_gradient_batches += 1
        if model.spatial_gate is not None:
            spatial_gate_gradient_norm_sum += _module_gradient_norm(
                model.spatial_gate
            )
            spatial_gate_gradient_batches += 1

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if run_diagnostics:
            if graph_snapshot is None:
                raise AssertionError("Graph diagnostic snapshot is missing.")
            graph_parameter_update_norm_sum += _parameter_update_norm(
                graph_snapshot,
                graph_parameters,
            )
            diagnostic_graph_batches += 1

        optimizer.zero_grad(set_to_none=True)

        target_key = (
            "target_cumulative_log_change"
            if output_representation == "cumulative_log_change"
            else "y"
        )
        count = int(torch.as_tensor(batch[target_key]).numel())
        optimisation_sum += float(optimisation_loss.detach().item()) * count
        forecast_optimisation_sum += float(
            forecast_optimisation_loss.detach().item()
        ) * count
        graph_regularisation_sum += float(
            graph_regularisation.total.detach().item()
        ) * count
        graph_target_entropy_penalty_sum += float(
            graph_regularisation.target_entropy_penalty.detach().item()
        ) * count
        native_sum += float(native_loss.detach().item()) * count
        target_count += count

        summary = _graph_summary(output.graph.selected)
        if summary["mean_row_entropy"] is not None:
            graph_entropy_sum += float(summary["mean_row_entropy"])
            graph_effective_sum += float(summary["mean_effective_neighbours"])
            graph_diag_sum += float(summary["mean_diagonal_weight"])
            graph_max_edge_sum += float(summary["maximum_edge_weight"])
            graph_top10_mass_sum += float(summary["mean_top10_row_mass"])
            graph_batches += 1
        beta_value = _scalar_value(output.spatial_beta)
        if beta_value is not None:
            spatial_beta_sum += beta_value
            spatial_beta_batches += 1
        alpha_value = _scalar_value(output.graph.alpha)
        if alpha_value is not None:
            dynamic_alpha_sum += alpha_value
            dynamic_alpha_batches += 1
        progress.set_postfix(loss=f"{native_sum / target_count:.6g}")

    synchronise_device(device)
    return {
        "optimisation_loss": optimisation_sum / target_count,
        "forecast_optimisation_loss": (
            forecast_optimisation_sum / target_count
        ),
        "graph_regularisation_loss": (
            graph_regularisation_sum / target_count
        ),
        "graph_target_entropy_penalty": (
            graph_target_entropy_penalty_sum / target_count
        ),
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
        "graph_maximum_edge_weight": (
            graph_max_edge_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_top10_row_mass": (
            graph_top10_mass_sum / graph_batches if graph_batches else None
        ),
        "spatial_beta": (
            spatial_beta_sum / spatial_beta_batches
            if spatial_beta_batches
            else None
        ),
        "dynamic_alpha": (
            dynamic_alpha_sum / dynamic_alpha_batches
            if dynamic_alpha_batches
            else None
        ),
        "graph_combined_gradient_norm": (
            combined_graph_gradient_norm_sum / combined_graph_gradient_batches
            if combined_graph_gradient_batches
            else 0.0
        ),
        "graph_forecast_gradient_norm": (
            forecast_graph_gradient_norm_sum / diagnostic_graph_batches
            if diagnostic_graph_batches
            else 0.0
        ),
        "graph_regulariser_gradient_norm": (
            regulariser_graph_gradient_norm_sum / diagnostic_graph_batches
            if diagnostic_graph_batches
            else 0.0
        ),
        "graph_parameter_update_norm": (
            graph_parameter_update_norm_sum / diagnostic_graph_batches
            if diagnostic_graph_batches
            else 0.0
        ),
        "spatial_gate_gradient_norm": (
            spatial_gate_gradient_norm_sum / spatial_gate_gradient_batches
            if spatial_gate_gradient_batches
            else 0.0
        ),
        "graph_diagnostic_batches": diagnostic_graph_batches,
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
    output_representation = str(config["model"]["output_representation"])
    native_sum = 0.0
    optimisation_sum = 0.0
    target_count = 0
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    selected_graphs: list[Tensor] = []
    base_graphs: list[Tensor] = []
    dynamic_graphs: list[Tensor] = []
    spatial_betas: list[Tensor] = []
    dynamic_alphas: list[Tensor] = []
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
                    graph_context_values=_graph_context_values(
                        batch,
                        model=model,
                        device=device,
                    ),
                )
            optimisation_loss, native_loss = _loss_values(
                output.predictions,
                batch,
                device=device,
                output_representation=output_representation,
                loss_type=loss_type,
                bps_scale=bps_scale,
                eps=eps,
            )
            raw_prediction = _prediction_raw(
                output.predictions,
                batch,
                device=device,
                output_representation=output_representation,
            )
            target_key = (
                "target_cumulative_log_change"
                if output_representation == "cumulative_log_change"
                else "y"
            )
            count = int(torch.as_tensor(batch[target_key]).numel())
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
            batch_size = int(raw_prediction.shape[0])
            selected_component = _expand_graph_component(
                output.graph.selected,
                batch_size=batch_size,
            )
            base_component = _expand_graph_component(
                output.graph.base,
                batch_size=batch_size,
            )
            dynamic_component = _expand_graph_component(
                output.graph.dynamic,
                batch_size=batch_size,
            )
            if selected_component is not None:
                selected_graphs.append(selected_component)
            if base_component is not None:
                base_graphs.append(base_component)
            if dynamic_component is not None:
                dynamic_graphs.append(dynamic_component)
            if output.spatial_beta is not None:
                spatial_betas.append(
                    output.spatial_beta.detach().float().cpu().reshape(1)
                )
            if output.graph.alpha is not None:
                dynamic_alphas.append(
                    torch.as_tensor(output.graph.alpha)
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )
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
    loss_metric_abs_difference: float | None = None
    if (
        output_representation == "cumulative_log_change"
        and loss_type == "cumulative_log_change_mae"
    ):
        evaluator_log_mae = float(
            metric_results["cumulative_log_change_mae"].mean().item()
        )
        native_validation_loss = native_sum / target_count
        loss_metric_abs_difference = abs(
            native_validation_loss - evaluator_log_mae
        )
        if loss_metric_abs_difference > 1.0e-6:
            raise AssertionError(
                "Direct cumulative-log-change validation loss differs from "
                "the ForecastEvaluator Log MAE. "
                f"Absolute difference: {loss_metric_abs_difference}."
            )
    selected_graph_tensor = (
        torch.cat(selected_graphs, dim=0) if selected_graphs else None
    )
    base_graph_tensor = torch.cat(base_graphs, dim=0) if base_graphs else None
    dynamic_graph_tensor = (
        torch.cat(dynamic_graphs, dim=0) if dynamic_graphs else None
    )
    spatial_beta_value = (
        float(torch.cat(spatial_betas).mean().item())
        if spatial_betas
        else None
    )
    dynamic_alpha_value = (
        float(torch.cat(dynamic_alphas).mean().item())
        if dynamic_alphas
        else None
    )
    synchronise_device(device)
    return {
        "optimisation_loss": optimisation_sum / target_count,
        "native_loss": native_sum / target_count,
        "target_count": target_count,
        "prediction_result": prediction_result,
        "metric_results": metric_results,
        "metric_table": metric_table,
        "graphs": {
            "selected": selected_graph_tensor,
            "base": base_graph_tensor,
            "dynamic": dynamic_graph_tensor,
            # ContinuousForecaster contains exactly one graph/spatial stage.
            # Preserve the historical top-level keys while also writing the
            # canonical per-layer schema expected by Graph Hub.
            "per_layer": (selected_graph_tensor,),
            "per_layer_base": (base_graph_tensor,),
            "per_layer_dynamic": (dynamic_graph_tensor,),
            "num_layers": 1,
            "selected_layer": 0,
            "num_heads_per_layer": [int(config["model"]["graph"]["num_heads"])],
            "layer_head_counts": [int(config["model"]["graph"]["num_heads"])],
            "spatial_beta": spatial_beta_value,
            "dynamic_alpha": dynamic_alpha_value,
            "dates": days,
            "orientation": "A[target, source]",
        },
        "graph_summary": _graph_summary(selected_graph_tensor),
        "spatial_beta": spatial_beta_value,
        "dynamic_alpha": dynamic_alpha_value,
        "loss_metric_abs_difference": loss_metric_abs_difference,
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
    learning_rates: Mapping[str, float | None],
    train_metrics: Mapping[str, Any],
    validation: Mapping[str, Any],
    selection_score: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    backbone_lr = learning_rates.get("backbone")
    graph_lr = learning_rates.get("graph")
    if backbone_lr is None:
        raise ValueError("Backbone learning rate is missing.")
    record: dict[str, Any] = {
        "epoch": int(epoch),
        # Backward-compatible alias used by older analysis cells.
        "learning_rate": float(backbone_lr),
        "backbone_learning_rate": float(backbone_lr),
        "graph_learning_rate": (
            None if graph_lr is None else float(graph_lr)
        ),
        "training_optimisation_loss": float(
            train_metrics["optimisation_loss"]
        ),
        "training_forecast_optimisation_loss": float(
            train_metrics.get(
                "forecast_optimisation_loss",
                train_metrics["optimisation_loss"],
            )
        ),
        "training_graph_regularisation_loss": float(
            train_metrics.get("graph_regularisation_loss", 0.0)
        ),
        "training_graph_target_entropy_penalty": float(
            train_metrics.get("graph_target_entropy_penalty", 0.0)
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
        "graph_maximum_edge_weight": validation["graph_summary"][
            "maximum_edge_weight"
        ],
        "graph_mean_top10_row_mass": validation["graph_summary"][
            "mean_top10_row_mass"
        ],
        "spatial_beta": validation.get("spatial_beta"),
        "dynamic_alpha": validation.get("dynamic_alpha"),
        "training_graph_gradient_norm": float(
            train_metrics.get("graph_combined_gradient_norm", 0.0)
        ),
        "training_graph_combined_gradient_norm": float(
            train_metrics.get("graph_combined_gradient_norm", 0.0)
        ),
        "training_graph_forecast_gradient_norm": float(
            train_metrics.get("graph_forecast_gradient_norm", 0.0)
        ),
        "training_graph_regulariser_gradient_norm": float(
            train_metrics.get("graph_regulariser_gradient_norm", 0.0)
        ),
        "training_graph_parameter_update_norm": float(
            train_metrics.get("graph_parameter_update_norm", 0.0)
        ),
        "training_graph_diagnostic_batches": int(
            train_metrics.get("graph_diagnostic_batches", 0)
        ),
        "training_spatial_gate_gradient_norm": float(
            train_metrics.get("spatial_gate_gradient_norm", 0.0)
        ),
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
    backbone_parameters, graph_parameters = _trainable_parameter_partition(model)
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
        "output_representation": model_config.output_representation,
        "output_head_initialisation": (
            model_config.output_head_initialisation
        ),
        "temporal_backbone": model_config.temporal.type,
        "graph_type": model_config.graph.type,
        "graph_heads": model_config.graph.num_heads,
        "graph_hidden_dim": model_config.graph.hidden_dim,
        "graph_activation": model_config.graph.activation,
        "graph_gate_type": model_config.graph.gate_type,
        "graph_initial_alpha": model_config.graph.initial_alpha,
        "spatial_gate_type": model_config.spatial_gate_type,
        "spatial_initial_beta": model_config.spatial_gate_initial_beta,
        "backbone_learning_rate": float(training["learning_rate"]),
        "graph_learning_rate": (
            float(training["graph_learning_rate"])
            if graph_parameters
            else None
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "backbone_trainable_parameters": int(
            sum(parameter.numel() for parameter in backbone_parameters)
        ),
        "graph_trainable_parameters": int(
            sum(parameter.numel() for parameter in graph_parameters)
        ),
        "graph_regularisation": dict(
            resolved["model"]["graph_regularisation"]
        ),
        "loss_type": training["loss"]["type"],
        "normalisation": "context-only per asset/channel",
        "cross_asset_path_before_graph": False,
        "fixed_graph_resource": (
            None if fixed_resource is None else fixed_resource.metadata()
        ),
        "dynamic_correlation": (
            dict(resolved["model"].get("dynamic_correlation", {}))
            if model_config.graph.type == "dynamic_correlation"
            else None
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
            current_learning_rates = _current_learning_rates(optimizer)
            train_metrics = _run_train_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                config=resolved,
                current_epoch=epoch,
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
                learning_rates=current_learning_rates,
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
                        "loss_metric_abs_difference": validation[
                            "loss_metric_abs_difference"
                        ],
                        "graph_summary": validation["graph_summary"],
                        "spatial_beta": validation.get("spatial_beta"),
                        "dynamic_alpha": validation.get("dynamic_alpha"),
                        "backbone_learning_rate": current_learning_rates.get(
                            "backbone"
                        ),
                        "graph_learning_rate": current_learning_rates.get(
                            "graph"
                        ),
                        "graph_combined_gradient_norm": train_metrics.get(
                            "graph_combined_gradient_norm"
                        ),
                        "graph_forecast_gradient_norm": train_metrics.get(
                            "graph_forecast_gradient_norm"
                        ),
                        "graph_regulariser_gradient_norm": train_metrics.get(
                            "graph_regulariser_gradient_norm"
                        ),
                        "graph_parameter_update_norm": train_metrics.get(
                            "graph_parameter_update_norm"
                        ),
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

            graph_lr_text = (
                "none"
                if current_learning_rates.get("graph") is None
                else f"{float(current_learning_rates['graph']):.3g}"
            )
            print(
                f"epoch={epoch} train={train_metrics['native_loss']:.6g} "
                f"val={validation['native_loss']:.6g} "
                f"selection={score:.6g} best={best_score:.6g} "
                f"best_epoch={best_epoch} "
                f"backbone_lr={float(current_learning_rates['backbone']):.3g} "
                f"graph_lr={graph_lr_text}"
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

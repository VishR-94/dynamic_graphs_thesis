from __future__ import annotations

"""Train the twelve dense-supervision graph curiosity experiments.

This runner is deliberately isolated from the dissertation's frozen valid
model-selection path.  It supports two temporal backbones (the selected
ModernTCN and a causal per-node Transformer), two training contracts
(stride-one fixed contexts and BaseDyGraph-style dense prefixes), and three
graph variants (correlation-initialised static+dynamic, random static+dynamic,
and dynamic-only).

Gradient updates always use the canonical January-August training split.
Checkpoint selection deliberately uses the October-December test split.  The
selected checkpoint is then exported over canonical train, September
validation, and test at the ordinary stride-15 forecast origins.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_parallel_forecast_dataset import (
    DensePrefixMultiHorizonDataset,
    build_dense_prefix_dataset,
    repeat_batch_for_prefixes,
    right_aligned_prefix_batch,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.evaluation.prediction_transforms import raw_to_cumulative_log_change
from src.models.dense_parallel_graph_models import (
    DenseParallelGraphModelConfig,
    ModernTCNDenseParallelGraphModel,
    TransformerDenseParallelGraphModel,
    build_dense_parallel_model,
    dense_parallel_config_from_mapping,
)
from src.models.graph_priors import build_absolute_correlation_graph_prior
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]
GRAPH_ORIENTATION = "A[target, source]"


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested accelerator without importing token/Kronos code."""

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
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serialisable."
    )


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


def synchronise_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one dense parallel graph-supervision curiosity model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
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


def _load_config(path: Path) -> ConfigDict:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected one JSON object in {path}.")
    _validate_config(values)
    return values


def _validate_config(config: Mapping[str, Any]) -> None:
    for key in ("data", "normalisation", "model", "training"):
        if not isinstance(config.get(key), Mapping):
            raise KeyError(f"Config must contain mapping {key!r}.")

    if str(config.get("model_family")) != "dense_parallel_graph_supervision":
        raise ValueError("Unexpected model_family for this runner.")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    horizons = tuple(int(value) for value in data["horizons"])
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("data.horizons must be non-empty, unique and increasing.")
    if int(data["context_length"]) <= 0 or int(data["export_stride"]) <= 0:
        raise ValueError("context_length/export_stride must be positive.")
    if str(data["target_channel"]).lower() != "close":
        raise ValueError("This experiment predicts Close only.")
    if str(data.get("input_representation", "raw")) != "raw":
        raise ValueError("The experiment uses context-normalised raw OHLCV.")

    temporal_type = str(model["temporal"]["type"])
    if temporal_type not in {"modern_tcn", "transformer"}:
        raise ValueError("temporal.type must be modern_tcn or transformer.")
    training_style = str(training["training_style"])
    if training_style not in {"stride1_fixed_context", "dense_prefix"}:
        raise ValueError("Unsupported training_style.")
    graph_variant = str(model["variant"])
    if graph_variant not in {
        "correlation_static_dynamic_state",
        "random_static_dynamic_state",
        "dynamic_state",
    }:
        raise ValueError("Unsupported graph variant.")
    if str(model["graph"]["activation"]) != "softmax":
        raise ValueError("This controlled grid uses softmax graphs.")
    if bool(model["graph"]["add_self_loops"]):
        raise ValueError("This controlled grid uses zero-diagonal graphs.")
    if int(model["graph"]["hidden_dim"]) % int(model["graph"]["num_heads"]):
        raise ValueError("graph.hidden_dim must be divisible by graph.num_heads.")
    if str(model["spatial"]["gate_type"]) != "learned_scalar":
        raise ValueError("Every run retains the learned beta gate.")
    if any(
        float(model["graph_regularisation"].get(key, 0.0)) != 0.0
        for key in (
            "graph_entropy_reg",
            "graph_target_entropy_reg",
            "graph_temporal_smooth_reg",
        )
    ):
        raise ValueError("These dense-supervision runs have no graph regularisation.")

    if str(training["selection_split"]) != "test":
        raise ValueError("This curiosity runner deliberately selects on test.")
    if tuple(int(value) for value in training["selection_horizons"]) != horizons:
        raise ValueError("Selection horizons must equal all configured horizons.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("The selected optimisation profile uses Adam.")
    if str(training["parameter_grouping"]) != "split":
        raise ValueError("The selected optimisation profile uses split LRs.")
    if str(training["scheduler"]) != "modern_tcn_type3_delayed":
        raise ValueError("The selected schedule is delayed type-3 decay.")
    if int(training["scheduler_decay_start_epoch"]) < 1:
        raise ValueError("scheduler_decay_start_epoch must be >= 1.")
    decay = float(training["scheduler_decay_factor"])
    if not math.isfinite(decay) or not 0.0 < decay <= 1.0:
        raise ValueError("scheduler_decay_factor must lie in (0,1].")
    loss = training["loss"]
    weights = tuple(float(value) for value in loss["horizon_weights"])
    references = tuple(float(value) for value in loss["horizon_reference_mae"])
    if len(weights) != len(horizons) or len(references) != len(horizons):
        raise ValueError("Loss weights/reference MAEs must match horizons.")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("Horizon weights must be positive and finite.")
    for key in (
        "training_stride",
        "batch_size",
        "selection_batch_size",
        "export_batch_size",
        "max_epochs",
        "patience",
        "prefix_chunk_size",
    ):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    for key in ("learning_rate", "graph_learning_rate"):
        if float(training[key]) <= 0.0:
            raise ValueError(f"training.{key} must be positive.")


def _continuous_dataset_config(
    config: Mapping[str, Any],
    *,
    stride: int,
) -> ContinuousDatasetConfig:
    data = config["data"]
    normalisation = config["normalisation"]
    return ContinuousDatasetConfig(
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        stride=int(stride),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channels=(str(data["target_channel"]),),
        input_representation="raw",
        eps=float(normalisation["eps"]),
        clip=bool(normalisation["clip"]),
        clip_min=float(normalisation["clip_min"]),
        clip_max=float(normalisation["clip_max"]),
    )


def _dense_dataset(
    split: Mapping[str, Any],
    config: Mapping[str, Any],
) -> DensePrefixMultiHorizonDataset:
    data = config["data"]
    normalisation = config["normalisation"]
    return build_dense_prefix_dataset(
        dict(split),
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        stride=int(config["training"]["training_stride"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        eps=float(normalisation["eps"]),
        clip=bool(normalisation["clip"]),
        clip_min=float(normalisation["clip_min"]),
        clip_max=float(normalisation["clip_max"]),
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "drop_last": False,
        "pin_memory": bool(pin_memory),
        "generator": generator,
        "worker_init_fn": _seed_worker if num_workers else None,
        "persistent_workers": bool(num_workers),
    }
    if num_workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _new_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _build_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    training = config["training"]
    graph_ids = set(model.graph_parameter_ids())  # type: ignore[attr-defined]
    graph_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in graph_ids
    ]
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in graph_ids
    ]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(expected) != len(graph_parameters) + len(backbone_parameters):
        raise AssertionError("Optimizer parameter partition lost parameters.")
    if not graph_parameters:
        raise RuntimeError("Every model must contain a trainable dynamic graph scorer.")
    return torch.optim.Adam(
        [
            {
                "params": backbone_parameters,
                "lr": float(training["learning_rate"]),
                "base_lr": float(training["learning_rate"]),
                "name": "backbone",
            },
            {
                "params": graph_parameters,
                "lr": float(training["graph_learning_rate"]),
                "base_lr": float(training["graph_learning_rate"]),
                "name": "graph",
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )


def _learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    values = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
    return {
        "backbone": values["backbone"],
        "graph": values["graph"],
    }


def _advance_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    training: Mapping[str, Any],
    completed_epoch: int,
) -> None:
    start = int(training["scheduler_decay_start_epoch"])
    factor = float(training["scheduler_decay_factor"])
    multiplier = (
        1.0
        if int(completed_epoch) < start
        else factor ** (int(completed_epoch) - start + 1)
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def _horizon_weights(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    dense: bool,
) -> Tensor:
    weights = torch.tensor(
        [float(value) for value in config["training"]["loss"]["horizon_weights"]],
        device=device,
        dtype=torch.float32,
    )
    if dense:
        return weights.view(1, 1, -1, 1, 1)
    return weights.view(1, -1, 1, 1)


def _normalised_to_raw(
    prediction: Tensor,
    *,
    target_mean: Tensor,
    target_std: Tensor,
    dense: bool,
) -> Tensor:
    prediction = prediction.float()
    mean = target_mean.float()
    std = target_std.float()
    if dense:
        return prediction * std[:, None, None, :, :] + mean[:, None, None, :, :]
    return prediction * std[:, None, :, :] + mean[:, None, :, :]


def _standard_absolute_error(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = torch.as_tensor(batch["target_norm_mean"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    std = torch.as_tensor(batch["target_norm_std"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    predicted_raw = _normalised_to_raw(
        prediction,
        target_mean=mean,
        target_std=std,
        dense=False,
    ).clamp_min(float(eps))
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    ).clamp_min(float(eps))
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    ).clamp_min(float(eps))
    predicted_change = raw_to_cumulative_log_change(
        predicted_raw, last, eps=float(eps)
    )
    true_change = raw_to_cumulative_log_change(true_raw, last, eps=float(eps))
    return predicted_raw, true_raw, last, (predicted_change - true_change).abs()


def _dense_absolute_error(
    prediction: Tensor,
    *,
    dense_true_raw: Tensor,
    dense_current_close: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    predicted_raw = _normalised_to_raw(
        prediction,
        target_mean=target_mean,
        target_std=target_std,
        dense=True,
    ).clamp_min(float(eps))
    true_raw = dense_true_raw.float().clamp_min(float(eps))
    current = dense_current_close.float().clamp_min(float(eps))
    predicted_change = torch.log(predicted_raw) - torch.log(current[:, :, None])
    true_change = torch.log(true_raw) - torch.log(current[:, :, None])
    return predicted_raw, (predicted_change - true_change).abs()


def _module_gradient_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return squared**0.5


def _scalar_gradient(parameter: nn.Parameter | None) -> float:
    if parameter is None or parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().float().abs().item())


def _graph_stats(graph: Tensor | None) -> tuple[float | None, float | None]:
    if graph is None:
        return None, None
    values = torch.as_tensor(graph).detach().float().clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    return float(entropy.mean().item()), float(entropy.exp().mean().item())


def _train_epoch_fixed_context(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    epoch: int,
) -> dict[str, float]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]) + int(epoch),
        pin_memory=device.type == "cuda",
    )
    weights = _horizon_weights(config, device=device, dense=False)
    eps = float(config["normalisation"]["eps"])
    bps_scale = float(training["loss"]["bps_scale"])
    model.train()
    unweighted_sum = 0.0
    weighted_sum = 0.0
    target_count = 0
    objective_sum = 0.0
    diagnostic_taken = False
    graph_gradient = alpha_gradient = beta_gradient = state_gradient = 0.0

    progress = tqdm(
        loader,
        desc=f"train fixed epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model(
                x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
        _, _, _, absolute_error = _standard_absolute_error(
            output.predictions,
            batch,
            device=device,
            eps=eps,
        )
        weighted_error = absolute_error * weights
        objective = weighted_error.mean() * bps_scale
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite fixed-context training loss.")
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            graph_gradient = _module_gradient_norm(model.graph_learner)
            alpha_gradient = _scalar_gradient(model.graph_learner.raw_alpha)
            beta_gradient = _scalar_gradient(model.spatial_gate.raw_beta)
            state_gradient = _module_gradient_norm(model.state_projection)
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(absolute_error.numel())
        unweighted_sum += float(absolute_error.detach().sum().item())
        weighted_sum += float(weighted_error.detach().sum().item())
        target_count += count
        objective_sum += float(objective.detach().item()) * count
        progress.set_postfix(native=f"{unweighted_sum / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Fixed-context training loader produced no targets.")
    return {
        "training_native_loss": unweighted_sum / target_count,
        "training_weighted_native_loss": weighted_sum / target_count,
        "training_objective_loss": objective_sum / target_count,
        "block_0_graph_gradient_norm": graph_gradient,
        "block_0_alpha_gradient_norm": alpha_gradient,
        "block_0_beta_gradient_norm": beta_gradient,
        "block_0_state_projection_gradient_norm": state_gradient,
    }


def _select_dense_tensor(
    values: Tensor,
    indices: Tensor,
) -> Tensor:
    """Select B×T values in prefix-major order to match prefix batching."""

    selected = values.index_select(1, indices)
    axes = [1, 0] + list(range(2, selected.ndim))
    return selected.permute(*axes).reshape(
        int(indices.numel()) * int(values.shape[0]),
        *values.shape[2:],
    ).contiguous()


def _train_epoch_dense_prefix_transformer(
    *,
    model: TransformerDenseParallelGraphModel,
    dataset: Dataset,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    epoch: int,
) -> dict[str, float]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]) + int(epoch),
        pin_memory=device.type == "cuda",
    )
    weights = _horizon_weights(config, device=device, dense=True)
    eps = float(config["normalisation"]["eps"])
    bps_scale = float(training["loss"]["bps_scale"])
    model.train()
    unweighted_sum = weighted_sum = objective_sum = 0.0
    target_count = 0
    diagnostic_taken = False
    graph_gradient = alpha_gradient = beta_gradient = state_gradient = 0.0

    progress = tqdm(
        loader,
        desc=f"train dense Transformer epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        dense_true = torch.as_tensor(batch["dense_y_unnormalised"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        dense_current = torch.as_tensor(batch["dense_current_close"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        mean = torch.as_tensor(batch["target_norm_mean"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        std = torch.as_tensor(batch["target_norm_std"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model.forward_dense(
                x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
        _, absolute_error = _dense_absolute_error(
            output.predictions,
            dense_true_raw=dense_true,
            dense_current_close=dense_current,
            target_mean=mean,
            target_std=std,
            eps=eps,
        )
        weighted_error = absolute_error * weights
        objective = weighted_error.mean() * bps_scale
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite dense Transformer training loss.")
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            graph_gradient = _module_gradient_norm(model.graph_learner)
            alpha_gradient = _scalar_gradient(model.graph_learner.raw_alpha)
            beta_gradient = _scalar_gradient(model.spatial_gate.raw_beta)
            state_gradient = _module_gradient_norm(model.state_projection)
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(absolute_error.numel())
        unweighted_sum += float(absolute_error.detach().sum().item())
        weighted_sum += float(weighted_error.detach().sum().item())
        target_count += count
        objective_sum += float(objective.detach().item()) * count
        progress.set_postfix(native=f"{unweighted_sum / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Dense Transformer loader produced no targets.")
    return {
        "training_native_loss": unweighted_sum / target_count,
        "training_weighted_native_loss": weighted_sum / target_count,
        "training_objective_loss": objective_sum / target_count,
        "block_0_graph_gradient_norm": graph_gradient,
        "block_0_alpha_gradient_norm": alpha_gradient,
        "block_0_beta_gradient_norm": beta_gradient,
        "block_0_state_projection_gradient_norm": state_gradient,
    }


def _train_epoch_dense_prefix_modern_tcn(
    *,
    model: ModernTCNDenseParallelGraphModel,
    dataset: Dataset,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    epoch: int,
) -> dict[str, float]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]) + int(epoch),
        pin_memory=device.type == "cuda",
    )
    context_length = int(config["data"]["context_length"])
    chunk_size = int(training["prefix_chunk_size"])
    weights = _horizon_weights(config, device=device, dense=False)
    eps = float(config["normalisation"]["eps"])
    bps_scale = float(training["loss"]["bps_scale"])
    model.train()
    unweighted_sum = weighted_sum = objective_sum = 0.0
    target_count = 0
    diagnostic_taken = False
    graph_gradient = alpha_gradient = beta_gradient = state_gradient = 0.0

    progress = tqdm(
        loader,
        desc=f"train dense ModernTCN epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        dense_true = torch.as_tensor(batch["dense_y_unnormalised"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        dense_current = torch.as_tensor(batch["dense_current_close"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        mean = torch.as_tensor(batch["target_norm_mean"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        std = torch.as_tensor(batch["target_norm_std"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        context_start = torch.as_tensor(batch["context_start"]).long()
        session_length = torch.as_tensor(batch["session_length"]).long()
        optimizer.zero_grad(set_to_none=True)

        batch_unweighted = batch_weighted = batch_objective = 0.0
        batch_count = 0
        for start in range(0, context_length, chunk_size):
            indices = torch.arange(
                start,
                min(start + chunk_size, context_length),
                dtype=torch.long,
            )
            prefix_count = int(indices.numel())
            prefix_x = right_aligned_prefix_batch(x, indices).to(device=device)
            prefix_context_start = repeat_batch_for_prefixes(
                context_start, prefix_count
            )
            prefix_session_length = repeat_batch_for_prefixes(
                session_length, prefix_count
            )
            prefix_true = _select_dense_tensor(dense_true, indices.to(device))
            prefix_current = _select_dense_tensor(
                dense_current, indices.to(device)
            )
            prefix_mean = repeat_batch_for_prefixes(mean, prefix_count)
            prefix_std = repeat_batch_for_prefixes(std, prefix_count)

            with _autocast_context(device, use_amp):
                output = model(
                    prefix_x,
                    context_start=prefix_context_start,
                    session_length=prefix_session_length,
                )
            predicted_raw = _normalised_to_raw(
                output.predictions,
                target_mean=prefix_mean,
                target_std=prefix_std,
                dense=False,
            ).clamp_min(eps)
            true_raw = prefix_true.clamp_min(eps)
            current = prefix_current.clamp_min(eps)
            predicted_change = torch.log(predicted_raw) - torch.log(current[:, None])
            true_change = torch.log(true_raw) - torch.log(current[:, None])
            absolute_error = (predicted_change - true_change).abs()
            weighted_error = absolute_error * weights
            chunk_fraction = prefix_count / context_length
            objective = weighted_error.mean() * bps_scale * chunk_fraction
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite dense ModernTCN training loss.")
            scaler.scale(objective).backward()

            count = int(absolute_error.numel())
            batch_unweighted += float(absolute_error.detach().sum().item())
            batch_weighted += float(weighted_error.detach().sum().item())
            batch_count += count
            # Undo chunk_fraction when accumulating the mean diagnostic.
            batch_objective += float(objective.detach().item()) * count / chunk_fraction

        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            graph_gradient = _module_gradient_norm(model.graph_learner)
            alpha_gradient = _scalar_gradient(model.graph_learner.raw_alpha)
            beta_gradient = _scalar_gradient(model.spatial_gate.raw_beta)
            state_gradient = _module_gradient_norm(model.state_projection)
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        unweighted_sum += batch_unweighted
        weighted_sum += batch_weighted
        target_count += batch_count
        objective_sum += batch_objective
        progress.set_postfix(native=f"{unweighted_sum / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Dense ModernTCN loader produced no targets.")
    return {
        "training_native_loss": unweighted_sum / target_count,
        "training_weighted_native_loss": weighted_sum / target_count,
        "training_objective_loss": objective_sum / target_count,
        "block_0_graph_gradient_norm": graph_gradient,
        "block_0_alpha_gradient_norm": alpha_gradient,
        "block_0_beta_gradient_norm": beta_gradient,
        "block_0_state_projection_gradient_norm": state_gradient,
    }


def _evaluate_selection(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    model.eval()
    eps = float(config["normalisation"]["eps"])
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    sums = torch.zeros(len(horizons), dtype=torch.float64)
    counts = torch.zeros(len(horizons), dtype=torch.float64)
    selected_entropies: list[float] = []
    selected_effective: list[float] = []
    dynamic_entropies: list[float] = []
    dynamic_effective: list[float] = []
    static_entropy = static_effective = None

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            with _autocast_context(device, use_amp):
                output = model(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            _, _, _, absolute_error = _standard_absolute_error(
                output.predictions,
                batch,
                device=device,
                eps=eps,
            )
            sums += absolute_error.detach().double().sum(dim=(0, 2, 3)).cpu()
            counts += torch.tensor(
                [absolute_error.shape[0] * absolute_error.shape[2] * absolute_error.shape[3]]
                * len(horizons),
                dtype=torch.float64,
            )
            entropy, effective = _graph_stats(output.graph.selected)
            if entropy is not None:
                selected_entropies.append(entropy)
                selected_effective.append(float(effective))
            entropy, effective = _graph_stats(output.graph.dynamic)
            if entropy is not None:
                dynamic_entropies.append(entropy)
                dynamic_effective.append(float(effective))
            if static_entropy is None and output.graph.base is not None:
                static_entropy, static_effective = _graph_stats(output.graph.base)

    if torch.any(counts <= 0):
        raise RuntimeError("Selection loader produced no targets.")
    by_horizon = (sums / counts).tolist()
    alpha = model.alpha()  # type: ignore[attr-defined]
    beta = model.beta()  # type: ignore[attr-defined]
    return {
        "selection_score": float(np.mean(by_horizon)),
        "by_horizon": {
            int(horizon): float(value)
            for horizon, value in zip(horizons, by_horizon)
        },
        "block_0_alpha": None if alpha is None else float(alpha.detach().item()),
        "block_0_beta": float(beta.detach().item()),
        "block_0_selected_entropy": (
            None if not selected_entropies else float(np.mean(selected_entropies))
        ),
        "block_0_selected_effective_neighbours": (
            None if not selected_effective else float(np.mean(selected_effective))
        ),
        "block_0_dynamic_entropy": (
            None if not dynamic_entropies else float(np.mean(dynamic_entropies))
        ),
        "block_0_dynamic_effective_neighbours": (
            None if not dynamic_effective else float(np.mean(dynamic_effective))
        ),
        "block_0_static_entropy": static_entropy,
        "block_0_static_effective_neighbours": static_effective,
    }


def _history_record(
    *,
    epoch: int,
    learning_rates: Mapping[str, float],
    train: Mapping[str, float],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    epoch_seconds: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "epoch_seconds": float(epoch_seconds),
        "temporal_backbone": str(config["model"]["temporal"]["type"]),
        "training_style": str(config["training"]["training_style"]),
        "graph_variant": str(config["model"]["variant"]),
        "backbone_learning_rate": float(learning_rates["backbone"]),
        "graph_learning_rate": float(learning_rates["graph"]),
        **dict(train),
        "selection_split": "test",
        "selection_score": float(selection["selection_score"]),
        "block_0_alpha": selection.get("block_0_alpha"),
        "block_0_beta": selection.get("block_0_beta"),
        "dynamic_alpha": selection.get("block_0_alpha"),
        "spatial_beta": selection.get("block_0_beta"),
        "block_0_selected_entropy": selection.get("block_0_selected_entropy"),
        "block_0_selected_effective_neighbours": selection.get(
            "block_0_selected_effective_neighbours"
        ),
        "block_0_static_entropy": selection.get("block_0_static_entropy"),
        "block_0_static_effective_neighbours": selection.get(
            "block_0_static_effective_neighbours"
        ),
        "block_0_dynamic_entropy": selection.get("block_0_dynamic_entropy"),
        "block_0_dynamic_effective_neighbours": selection.get(
            "block_0_dynamic_effective_neighbours"
        ),
        "test_graph_mean_row_entropy": selection.get("block_0_selected_entropy"),
        "test_graph_mean_effective_neighbours": selection.get(
            "block_0_selected_effective_neighbours"
        ),
    }
    for horizon, value in selection["by_horizon"].items():
        record[f"test_cumulative_log_change_mae_h{int(horizon)}"] = float(value)
    return record


def _export_selected_checkpoint(
    *,
    model: nn.Module,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.eval()
    eps = float(config["normalisation"]["eps"])
    horizons = [int(value) for value in config["data"]["horizons"]]
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    dates: list[str] = []
    selected_graphs: list[Tensor] = []
    dynamic_graphs: list[Tensor] = []
    singleton_static: Tensor | None = None

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"export {split_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            x = torch.as_tensor(batch["x"]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            with _autocast_context(device, use_amp):
                output = model(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            predicted_raw, true_raw, last, _ = _standard_absolute_error(
                output.predictions,
                batch,
                device=device,
                eps=eps,
            )
            predictions.append(predicted_raw.detach().cpu().float())
            targets.append(true_raw.detach().cpu().float())
            last_values.append(last.detach().cpu().float())
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).cpu())
            target_indices.append(torch.as_tensor(batch["target_indices"]).cpu())
            batch_days = batch["day"]
            if isinstance(batch_days, str):
                dates.append(batch_days)
            else:
                dates.extend(str(value) for value in batch_days)
            selected_graphs.append(
                output.graph.selected.detach().cpu().to(torch.float16).contiguous()
            )
            dynamic_graphs.append(
                output.graph.dynamic.detach().cpu().to(torch.float16).contiguous()
            )
            if output.graph.base is not None and singleton_static is None:
                singleton_static = (
                    output.graph.base.detach().cpu().to(torch.float16).contiguous()
                )

    prediction_result = {
        "y_pred": torch.cat(predictions, dim=0),
        "y_true": torch.cat(targets, dim=0),
        "last_context_target": torch.cat(last_values, dim=0),
        "channels": [str(config["data"]["target_channel"])],
        "horizons": horizons,
        "asset_cols": list(asset_cols),
        "sample_idx": torch.cat(sample_indices, dim=0).long(),
        "origin_idx": torch.cat(origin_indices, dim=0).long(),
        "target_indices": torch.cat(target_indices, dim=0).long(),
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

    selected = torch.cat(selected_graphs, dim=0)
    dynamic = torch.cat(dynamic_graphs, dim=0)
    saved_static = None if singleton_static is None else singleton_static[0].contiguous()
    alpha = model.alpha()  # type: ignore[attr-defined]
    beta = model.beta()  # type: ignore[attr-defined]
    graph_artifacts: dict[str, Any] = {
        "graph_type": str(config["model"]["graph"]["type"]),
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(asset_cols),
        "num_layers": 1,
        "num_heads": int(config["model"]["graph"]["num_heads"]),
        "num_heads_per_layer": [int(config["model"]["graph"]["num_heads"])],
        "layer_head_counts": [int(config["model"]["graph"]["num_heads"])],
        "selected_layer": 0,
        "selected": selected,
        "per_layer": (selected,),
        "base": saved_static,
        "per_layer_base": (saved_static,),
        "dynamic": dynamic,
        "per_layer_dynamic": (dynamic,),
        "alpha": None if alpha is None else alpha.detach().cpu().float().reshape(1),
        "beta": beta.detach().cpu().float().reshape(1),
        "dynamic_alpha": None if alpha is None else float(alpha.item()),
        "spatial_beta": float(beta.item()),
        "spatial_gate_type": "learned_scalar",
        "beta_trainable": True,
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "training_style": str(config["training"]["training_style"]),
        "temporal_backbone": str(config["model"]["temporal"]["type"]),
    }
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(prediction_result["y_pred"].shape[0]),
        "horizons": horizons,
        "assets": int(prediction_result["y_pred"].shape[2]),
        "alpha": graph_artifacts["dynamic_alpha"],
        "beta": graph_artifacts["spatial_beta"],
        "selected_graph": graph_component_summary(selected.float()),
        "static_graph": graph_component_summary(
            None if saved_static is None else saved_static.float()
        ),
        "dynamic_graph": graph_component_summary(dynamic.float()),
        "graph_orientation": GRAPH_ORIENTATION,
        "training_style": str(config["training"]["training_style"]),
        "temporal_backbone": str(config["model"]["temporal"]["type"]),
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
    root_prediction = run_dir / f"best_{split_name}_predictions.pt"
    root_graph = run_dir / f"best_{split_name}_graphs.pt"
    root_metric = run_dir / f"best_{split_name}_metric_table.csv"
    root_diagnostics = run_dir / f"best_{split_name}_diagnostics.json"
    epoch = int(values["diagnostics"]["checkpoint_epoch"])
    atomic_torch_save(
        {"epoch": epoch, "prediction_result": values["prediction_result"]},
        root_prediction,
    )
    atomic_torch_save(
        {"epoch": epoch, "graph_artifacts": values["graph_artifacts"]},
        root_graph,
    )
    atomic_csv_save(values["metric_table"], root_metric)
    atomic_json_save(values["diagnostics"], root_diagnostics)

    analysis_dir = run_dir / "analysis" / split_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_prediction, analysis_dir / "predictions.pt")
    shutil.copy2(root_graph, analysis_dir / "graphs.pt")
    shutil.copy2(root_metric, analysis_dir / "metric_table.csv")
    shutil.copy2(root_diagnostics, analysis_dir / "diagnostics.json")


def _prefix_graph_diagnostics(
    *,
    model: nn.Module,
    dataset: DensePrefixMultiHorizonDataset,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    checkpoint_epoch: int,
) -> None:
    requested = int(config["training"].get("prefix_graph_sample_windows", 0))
    sample_count = min(requested, len(dataset))
    if sample_count <= 0:
        return
    subset = Subset(dataset, list(range(sample_count)))
    loader = _build_loader(
        subset,
        batch_size=(
            min(sample_count, int(config["training"]["batch_size"]))
            if str(config["model"]["temporal"]["type"]) == "transformer"
            else 1
        ),
        shuffle=False,
        num_workers=0,
        seed=int(config["training"]["seed"]),
        pin_memory=device.type == "cuda",
    )
    context_length = int(config["data"]["context_length"])
    selected_windows: list[Tensor] = []
    dynamic_windows: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    dates: list[str] = []
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            x = torch.as_tensor(batch["x"]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if isinstance(model, TransformerDenseParallelGraphModel):
                with _autocast_context(device, use_amp):
                    output = model.forward_dense(
                        x,
                        context_start=batch["context_start"],
                        session_length=batch["session_length"],
                    )
                selected = output.graphs.selected
                dynamic = output.graphs.dynamic
            else:
                chunks_selected: list[Tensor] = []
                chunks_dynamic: list[Tensor] = []
                chunk_size = int(config["training"]["prefix_chunk_size"])
                for start in range(0, context_length, chunk_size):
                    indices = torch.arange(
                        start,
                        min(start + chunk_size, context_length),
                        dtype=torch.long,
                    )
                    prefix_count = int(indices.numel())
                    prefix_x = right_aligned_prefix_batch(x, indices)
                    with _autocast_context(device, use_amp):
                        output = model(
                            prefix_x,
                            context_start=repeat_batch_for_prefixes(
                                torch.as_tensor(batch["context_start"]).long(),
                                prefix_count,
                            ),
                            session_length=repeat_batch_for_prefixes(
                                torch.as_tensor(batch["session_length"]).long(),
                                prefix_count,
                            ),
                        )
                    batch_size = int(x.shape[0])
                    selected_chunk = output.graph.selected.reshape(
                        prefix_count,
                        batch_size,
                        *output.graph.selected.shape[1:],
                    ).permute(1, 0, 2, 3, 4)
                    dynamic_chunk = output.graph.dynamic.reshape(
                        prefix_count,
                        batch_size,
                        *output.graph.dynamic.shape[1:],
                    ).permute(1, 0, 2, 3, 4)
                    chunks_selected.append(selected_chunk)
                    chunks_dynamic.append(dynamic_chunk)
                selected = torch.cat(chunks_selected, dim=1)
                dynamic = torch.cat(chunks_dynamic, dim=1)
            selected_windows.append(selected.detach().cpu().to(torch.float16))
            dynamic_windows.append(dynamic.detach().cpu().to(torch.float16))
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).cpu())
            batch_days = batch["day"]
            if isinstance(batch_days, str):
                dates.append(batch_days)
            else:
                dates.extend(str(value) for value in batch_days)

    selected = torch.cat(selected_windows, dim=0)
    dynamic = torch.cat(dynamic_windows, dim=0)
    if int(selected.shape[1]) != context_length:
        raise RuntimeError("Prefix graph diagnostics lost internal origins.")
    rows: list[dict[str, Any]] = []
    final_selected = selected[:, -1].float()
    final_dynamic = dynamic[:, -1].float()
    for position in range(context_length):
        selected_summary = graph_component_summary(selected[:, position].float())
        dynamic_summary = graph_component_summary(dynamic[:, position].float())
        rows.append(
            {
                "prefix_length": position + 1,
                "selected_mean_row_entropy": selected_summary["mean_row_entropy"],
                "selected_effective_neighbours": selected_summary[
                    "mean_effective_neighbours"
                ],
                "dynamic_mean_row_entropy": dynamic_summary["mean_row_entropy"],
                "dynamic_effective_neighbours": dynamic_summary[
                    "mean_effective_neighbours"
                ],
                "selected_mean_absolute_distance_to_final": float(
                    (selected[:, position].float() - final_selected).abs().mean().item()
                ),
                "dynamic_mean_absolute_distance_to_final": float(
                    (dynamic[:, position].float() - final_dynamic).abs().mean().item()
                ),
            }
        )
    payload = {
        "epoch": int(checkpoint_epoch),
        "selected": selected,
        "dynamic": dynamic,
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": torch.cat(sample_indices).long(),
        "origin_idx": torch.cat(origin_indices).long(),
        "dates": dates,
        "prefix_lengths": torch.arange(1, context_length + 1),
        "orientation": GRAPH_ORIENTATION,
        "training_style": "dense_prefix",
    }
    atomic_torch_save(payload, run_dir / "best_train_prefix_graph_sample.pt")
    atomic_csv_save(
        pd.DataFrame(rows),
        run_dir / "best_train_prefix_graph_diagnostics.csv",
    )


def _signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_value(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _prepare_run_dir(
    output_dir: Path,
    run_name: str,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    run_dir = Path(output_dir) / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is non-empty: {run_dir}. Use --resume or --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    run_signature: str,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    training_complete: bool,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(without_improvement),
        "history": [dict(row) for row in history],
        "rng_state": capture_rng_state(),
        "run_signature": run_signature,
        "resolved_config": dict(config),
        "metadata": dict(metadata),
        "training_complete": bool(training_complete),
    }


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


def _write_initial_graphs(
    *,
    run_dir: Path,
    model: nn.Module,
    source_prior: Tensor | None,
    config: Mapping[str, Any],
    asset_cols: Sequence[str],
) -> None:
    initial = model.graph_learner.static_adjacency()  # type: ignore[attr-defined]
    if initial is None:
        return
    initial_cpu = initial.detach().cpu().float()[0]
    payload = {
        "adjacency": initial_cpu,
        "asset_cols": list(asset_cols),
        "orientation": GRAPH_ORIENTATION,
        "prior_type": str(config["model"]["prior"]["type"]),
        "prior_scale": float(config["model"]["prior"]["scale"]),
        "prior_jitter": float(config["model"]["prior"]["jitter"]),
    }
    atomic_torch_save(payload, run_dir / "initial_graph_prior.pt")
    atomic_csv_save(
        pd.DataFrame(initial_cpu.mean(dim=0).numpy(), index=asset_cols, columns=asset_cols),
        run_dir / "initial_graph_prior.csv",
    )
    if source_prior is not None:
        source = torch.as_tensor(source_prior).detach().cpu().float()
        atomic_torch_save(
            {
                "adjacency": source,
                "asset_cols": list(asset_cols),
                "orientation": GRAPH_ORIENTATION,
                "description": "training-only absolute Close-return correlation prior",
            },
            run_dir / "initial_graph_source.pt",
        )
        atomic_csv_save(
            pd.DataFrame(source.numpy(), index=asset_cols, columns=asset_cols),
            run_dir / "initial_graph_source.csv",
        )


def main() -> None:
    args = build_argument_parser().parse_args()
    resolved = _load_config(args.config)
    training = resolved["training"]
    project_root = Path(__file__).resolve().parents[2]
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    device = resolve_device(args.device)
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    seed = int(training["seed"])
    set_seed(seed)

    raw_train, raw_validation, raw_test = load_candle_splits(args.data_dir)
    train_split, validation_split, test_split = clean_candle_splits(
        raw_train, raw_validation, raw_test
    )
    asset_cols = [str(value) for value in train_split["asset_cols"]]
    for label, split in (("validation", validation_split), ("test", test_split)):
        if [str(value) for value in split["asset_cols"]] != asset_cols:
            raise ValueError(f"Train and {label} asset order differs.")

    export_config = _continuous_dataset_config(
        resolved,
        stride=int(resolved["data"]["export_stride"]),
    )
    export_datasets = {
        "train": build_continuous_dataset(train_split, config=export_config),
        "validation": build_continuous_dataset(validation_split, config=export_config),
        "test": build_continuous_dataset(test_split, config=export_config),
    }
    if str(training["training_style"]) == "stride1_fixed_context":
        training_dataset: Dataset = build_continuous_dataset(
            train_split,
            config=_continuous_dataset_config(
                resolved,
                stride=int(training["training_stride"]),
            ),
        )
    else:
        training_dataset = _dense_dataset(train_split, resolved)

    static_prior: Tensor | None = None
    if str(resolved["model"]["prior"]["type"]) == "correlation":
        static_prior = build_absolute_correlation_graph_prior(
            train_split,
            expected_asset_cols=asset_cols,
            threshold=None,
        )
    model_config = dense_parallel_config_from_mapping(
        resolved,
        num_nodes=len(asset_cols),
    )
    model = build_dense_parallel_model(model_config, static_prior=static_prior).to(device)
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)

    run_signature = _signature(resolved)
    now = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "status": "running",
        "created_at_utc": now,
        "model_family": "dense_parallel_graph_supervision",
        "experiment_family": "dense_parallel_graph_supervision",
        "run_name": args.run_name,
        "run_signature": run_signature,
        "project_git_commit": _git_value(["rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": _git_value(["branch", "--show-current"], cwd=project_root),
        "device": str(device),
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": str(training["selection_metric"]),
        "temporal_backbone": str(resolved["model"]["temporal"]["type"]),
        "training_style": str(training["training_style"]),
        "graph_variant": str(resolved["model"]["variant"]),
        "graph_type": str(resolved["model"]["graph"]["type"]),
        "prior_type": str(resolved["model"]["prior"]["type"]),
        "state_pathway": True,
        "context_length": int(resolved["data"]["context_length"]),
        "horizons": [int(value) for value in resolved["data"]["horizons"]],
        "training_stride": int(training["training_stride"]),
        "export_stride": int(resolved["data"]["export_stride"]),
        "input_channels": [str(value) for value in resolved["data"]["input_channels"]],
        "asset_cols": asset_cols,
        "training_windows": int(len(training_dataset)),
        "effective_dense_prefix_tasks": (
            int(len(training_dataset) * int(resolved["data"]["context_length"]))
            if str(training["training_style"]) == "dense_prefix"
            else int(len(training_dataset))
        ),
        "train_windows": int(len(export_datasets["train"])),
        "validation_windows": int(len(export_datasets["validation"])),
        "test_windows": int(len(export_datasets["test"])),
        "train_sessions": int(len(train_split["samples"])),
        "validation_sessions": int(len(validation_split["samples"])),
        "test_sessions": int(len(test_split["samples"])),
        "optimizer": "adam",
        "scheduler": "modern_tcn_type3_delayed",
        "scheduler_decay_start_epoch": int(training["scheduler_decay_start_epoch"]),
        "scheduler_decay_factor": float(training["scheduler_decay_factor"]),
        "learning_rate": float(training["learning_rate"]),
        "graph_learning_rate": float(training["graph_learning_rate"]),
        "mixed_precision": bool(use_amp),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "loss": dict(training["loss"]),
        "graph_heads": int(resolved["model"]["graph"]["num_heads"]),
        "graph_hidden_dim": int(resolved["model"]["graph"]["hidden_dim"]),
        "graph_initial_alpha": float(resolved["model"]["graph"]["initial_alpha"]),
        "spatial_initial_beta": float(resolved["model"]["spatial"]["initial_beta"]),
        "spatial_gate_type": str(resolved["model"]["spatial"]["gate_type"]),
        "forecast_strategy": "parallel_weighted",
        "output_representation": str(resolved["model"]["output_representation"]),
        "normalisation": "full observed context per asset/channel",
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "graph_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) in model.graph_parameter_ids()  # type: ignore[attr-defined]
            )
        ),
        "backbone_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) not in model.graph_parameter_ids()  # type: ignore[attr-defined]
            )
        ),
    }
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    _write_initial_graphs(
        run_dir=run_dir,
        model=model,
        source_prior=static_prior,
        config=resolved,
        asset_cols=asset_cols,
    )

    selection_loader = _build_loader(
        export_datasets["test"],
        batch_size=int(training["selection_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    training_complete = False
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(last_checkpoint_path)
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["run_signature"] != run_signature:
            raise ValueError("Resume checkpoint configuration differs.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(checkpoint["evaluations_without_improvement"])
        history = [dict(row) for row in checkpoint["history"]]
        training_complete = bool(checkpoint.get("training_complete", False))

    wandb_run = _init_wandb(args, resolved)
    last_epoch = max(0, start_epoch - 1)
    if not training_complete:
        for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
            epoch_start = perf_counter()
            learning_rates = _learning_rates(optimizer)
            if str(training["training_style"]) == "stride1_fixed_context":
                train_values = _train_epoch_fixed_context(
                    model=model,
                    dataset=training_dataset,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    config=resolved,
                    epoch=epoch,
                )
            elif isinstance(model, TransformerDenseParallelGraphModel):
                train_values = _train_epoch_dense_prefix_transformer(
                    model=model,
                    dataset=training_dataset,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    config=resolved,
                    epoch=epoch,
                )
            elif isinstance(model, ModernTCNDenseParallelGraphModel):
                train_values = _train_epoch_dense_prefix_modern_tcn(
                    model=model,
                    dataset=training_dataset,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    config=resolved,
                    epoch=epoch,
                )
            else:
                raise TypeError("Unexpected dense-prefix model type.")

            synchronise_device(device)
            selection = _evaluate_selection(
                model=model,
                loader=selection_loader,
                device=device,
                use_amp=use_amp,
                config=resolved,
                description=f"test selection epoch {epoch}",
            )
            synchronise_device(device)
            record = _history_record(
                epoch=epoch,
                learning_rates=learning_rates,
                train=train_values,
                selection=selection,
                config=resolved,
                epoch_seconds=perf_counter() - epoch_start,
            )
            history.append(record)
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")

            improved = float(selection["selection_score"]) < (
                best_score - float(training["min_delta"])
            )
            if improved:
                best_score = float(selection["selection_score"])
                best_epoch = int(epoch)
                without_improvement = 0
            else:
                without_improvement += 1

            # The selected checkpoint stores the model at this epoch.  The
            # resume checkpoint is written after advancing the schedule so it
            # contains the exact learning rates required by the next epoch.
            if improved:
                best_checkpoint_values = _checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    without_improvement=without_improvement,
                    history=history,
                    run_signature=run_signature,
                    config=resolved,
                    metadata=metadata,
                    training_complete=False,
                )
                atomic_torch_save(
                    best_checkpoint_values,
                    run_dir / "best_checkpoint.pt",
                )

            if wandb_run is not None:
                wandb_run.log(record, step=epoch)
            print(
                f"Epoch {epoch:03d} | train={train_values['training_native_loss']:.8f} "
                f"| test_mean={selection['selection_score']:.8f} "
                f"| alpha={selection.get('block_0_alpha')} "
                f"| beta={selection.get('block_0_beta'):.4f}"
            )
            last_epoch = int(epoch)
            _advance_schedule(
                optimizer,
                training=training,
                completed_epoch=epoch,
            )
            last_checkpoint_values = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                without_improvement=without_improvement,
                history=history,
                run_signature=run_signature,
                config=resolved,
                metadata=metadata,
                training_complete=False,
            )
            atomic_torch_save(last_checkpoint_values, last_checkpoint_path)
            if without_improvement >= int(training["patience"]):
                print(f"Early stopping after epoch {epoch}.")
                break

        if best_epoch <= 0 or not (run_dir / "best_checkpoint.pt").is_file():
            raise RuntimeError("Training produced no selected checkpoint.")
        training_complete = True
        final_last = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        final_last["training_complete"] = True
        atomic_torch_save(final_last, last_checkpoint_path)

    best_checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if best_checkpoint["run_signature"] != run_signature:
        raise ValueError("Best checkpoint configuration differs.")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model.to(device)
    checkpoint_epoch = int(best_checkpoint["epoch"])
    if checkpoint_epoch != int(best_checkpoint["best_epoch"]):
        raise AssertionError("best_checkpoint.pt is not its recorded best epoch.")

    for split_name, split in (
        ("train", train_split),
        ("validation", validation_split),
        ("test", test_split),
    ):
        loader = _build_loader(
            export_datasets[split_name],
            batch_size=int(training["export_batch_size"]),
            shuffle=False,
            num_workers=int(training["num_workers"]),
            seed=seed,
            pin_memory=device.type == "cuda",
        )
        values = _export_selected_checkpoint(
            model=model,
            loader=loader,
            split_name=split_name,
            device=device,
            use_amp=use_amp,
            config=resolved,
            train_split=train_split,
            asset_cols=asset_cols,
            checkpoint_epoch=checkpoint_epoch,
        )
        _save_export(run_dir, split_name=split_name, values=values)

    if str(training["training_style"]) == "dense_prefix":
        if not isinstance(training_dataset, DensePrefixMultiHorizonDataset):
            raise TypeError("Dense-prefix training dataset has the wrong type.")
        _prefix_graph_diagnostics(
            model=model,
            dataset=training_dataset,
            run_dir=run_dir,
            device=device,
            use_amp=use_amp,
            config=resolved,
            checkpoint_epoch=checkpoint_epoch,
        )

    alpha = model.alpha()  # type: ignore[attr-defined]
    beta = model.beta()  # type: ignore[attr-defined]
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": checkpoint_epoch,
            "best_score": float(best_checkpoint["best_score"]),
            "epochs_completed": int(last_epoch or best_checkpoint["epoch"]),
            "final_alpha": None if alpha is None else float(alpha.detach().item()),
            "final_beta": float(beta.detach().item()),
        }
    )
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    if wandb_run is not None:
        wandb_run.finish()
    print("Completed:", run_dir)
    print("Best epoch:", checkpoint_epoch)
    print("Best test mean Log MAE:", float(best_checkpoint["best_score"]))


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Train and export the deliberately test-selected Round-2 depth grid.

Training gradients use the canonical January-August split.  Checkpoint
selection and early stopping use October-December test mean cumulative-log-
change MAE over every configured horizon.  The selected checkpoint is then
exported chronologically over train, September validation, and test.

The runner derives every window/tensor size from the configured datasets and
saved outputs.  No fixed expected window count is an execution gate.
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.evaluation.prediction_transforms import (
    inverse_window_normalisation,
    raw_to_cumulative_log_change,
)
from src.models.graph_priors import (
    build_absolute_correlation_graph_prior,
    build_sector_graph_prior,
)
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.models.modern_tcn_graph_round2 import (
    ModernTCNGraphRound2Config,
    ModernTCNGraphRound2Model,
    round2_model_config_from_mapping,
)
from src.training.run_dynamic_graph import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    set_seed,
)
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]
GRAPH_ORIENTATION = "A[target, source]"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one ModernTCN/Transformer Round-2 graph stack."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--company-profiles", type=Path, default=None)
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
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    _validate_config(values)
    return values


def _validate_config(config: Mapping[str, Any]) -> None:
    for key in ("data", "normalisation", "model", "training"):
        if not isinstance(config.get(key), Mapping):
            raise KeyError(f"Config must contain mapping {key!r}.")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    horizons = tuple(int(value) for value in data["horizons"])
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("data.horizons must be non-empty, unique, increasing.")
    if int(data["context_length"]) <= 0 or int(data["stride"]) <= 0:
        raise ValueError("context_length and stride must be positive.")
    if str(data["target_channel"]).lower() != "close":
        raise ValueError("Round 2 predicts Close only.")
    if str(data.get("input_representation", "raw")) != "raw":
        raise ValueError("Round 2 uses context-normalised raw OHLCV input.")

    if str(model["graph_family"]) not in {"dynamic_only", "prior_state"}:
        raise ValueError("Unsupported Round-2 graph family.")
    graph = model["graph"]
    heads = tuple(int(value) for value in graph["num_heads_per_block"])
    hidden = tuple(int(value) for value in graph["hidden_dims_per_block"])
    activations = tuple(str(value) for value in graph["activations_per_block"])
    if not heads or len(heads) != len(hidden) or len(heads) != len(activations):
        raise ValueError("Round-2 graph schedules must have equal non-zero length.")
    if activations[-1] != "sparsemax" or any(
        value != "softmax" for value in activations[:-1]
    ):
        raise ValueError(
            "Round 2 requires softmax in non-final blocks and sparsemax final."
        )
    for index, (num_heads, graph_hidden) in enumerate(
        zip(heads, hidden, strict=True)
    ):
        if num_heads <= 0 or graph_hidden <= 0 or graph_hidden % num_heads:
            raise ValueError(
                f"Invalid graph head/width schedule in block {index}."
            )
    if bool(graph.get("add_self_loops", False)):
        raise ValueError("Round 2 excludes graph self-edges.")
    if str(model["spatial"]["gate_type"]) != "learned_scalar":
        raise ValueError("Every Round-2 block must retain learned beta.")
    if any(
        float(model["graph_regularisation"].get(key, 0.0)) != 0.0
        for key in (
            "graph_entropy_reg",
            "graph_target_entropy_reg",
            "graph_temporal_smooth_reg",
        )
    ):
        raise ValueError("Round 2 intentionally uses no graph regularisation.")

    if str(training["selection_split"]) != "test":
        raise ValueError("This isolated curiosity runner selects on test.")
    if tuple(int(value) for value in training["selection_horizons"]) != horizons:
        raise ValueError("Selection horizons must equal all configured horizons.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("Round 2 preserves the winning Adam optimiser.")
    if str(training["parameter_grouping"]) != "split":
        raise ValueError("Round 2 preserves split backbone/graph LR groups.")
    if str(training["scheduler"]).lower() != "modern_tcn_type3":
        raise ValueError("Round 2 preserves the ModernTCN type-3 schedule.")
    for key in (
        "learning_rate",
        "graph_learning_rate",
        "batch_size",
        "selection_batch_size",
        "export_batch_size",
        "max_epochs",
        "patience",
    ):
        if float(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")


def _dataset_config(config: Mapping[str, Any]) -> ContinuousDatasetConfig:
    data = config["data"]
    normalisation = config["normalisation"]
    return ContinuousDatasetConfig(
        context_length=int(data["context_length"]),
        horizons=tuple(int(value) for value in data["horizons"]),
        stride=int(data["stride"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channels=(str(data["target_channel"]),),
        input_representation="raw",
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
    if enabled:
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


def _parameter_partition(
    model: ModernTCNGraphRound2Model,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    graph_ids = model.graph_parameter_ids()
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
    expected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(expected) != len(graph_parameters) + len(backbone_parameters):
        raise AssertionError("Optimizer parameter partition lost parameters.")
    if {id(value) for value in graph_parameters} & {
        id(value) for value in backbone_parameters
    }:
        raise AssertionError("Optimizer parameter groups overlap.")
    return backbone_parameters, graph_parameters


def _build_optimizer(
    model: ModernTCNGraphRound2Model,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
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


def _learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {"backbone": None, "graph": None}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "backbone" if index == 0 else "graph"))
        if name in result:
            result[name] = float(group["lr"])
    return result


def _advance_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    completed_epoch: int,
) -> None:
    multiplier = (
        1.0
        if int(completed_epoch) < 3
        else 0.9 ** (int(completed_epoch) - 3)
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def _normalised_prediction_to_raw(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tensor:
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
        y_norm=prediction.float(),
        target_norm_mean=mean,
        target_norm_std=std,
    )


def _forecast_errors(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    predicted_raw = _normalised_prediction_to_raw(
        prediction,
        batch,
        device=device,
    ).clamp_min(float(eps))
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).clamp_min(float(eps))
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).clamp_min(float(eps))
    predicted_change = raw_to_cumulative_log_change(
        predicted_raw,
        last,
        eps=float(eps),
    )
    true_change = raw_to_cumulative_log_change(
        true_raw,
        last,
        eps=float(eps),
    )
    absolute_error = (predicted_change - true_change).abs()
    return predicted_raw, true_raw, last, absolute_error


def _module_gradient_norm(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    squared = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum().item())
    return squared**0.5


def _scalar_gradient(value: nn.Parameter | None) -> float:
    if value is None or value.grad is None:
        return 0.0
    return float(value.grad.detach().float().abs().item())


def _graph_stats_accumulator() -> dict[str, float]:
    return {"entropy_sum": 0.0, "effective_sum": 0.0, "row_count": 0.0}


def _add_graph_stats(
    accumulator: dict[str, float],
    graph: Tensor | None,
    *,
    batch_size: int,
) -> None:
    if graph is None:
        return
    values = torch.as_tensor(graph).detach().float()
    if int(values.shape[0]) == 1 and batch_size > 1:
        values = values.expand(batch_size, -1, -1, -1)
    values = values.clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    accumulator["entropy_sum"] += float(entropy.sum().item())
    accumulator["effective_sum"] += float(entropy.exp().sum().item())
    accumulator["row_count"] += float(entropy.numel())


def _final_graph_stats(
    accumulator: Mapping[str, float],
) -> dict[str, float | None]:
    count = float(accumulator["row_count"])
    if count <= 0:
        return {"entropy": None, "effective_neighbours": None}
    return {
        "entropy": float(accumulator["entropy_sum"]) / count,
        "effective_neighbours": float(accumulator["effective_sum"]) / count,
    }


def _train_epoch(
    *,
    model: ModernTCNGraphRound2Model,
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
    model.train()
    eps = float(config["normalisation"]["eps"])
    bps_scale = float(training["loss"]["bps_scale"])
    total_absolute_error = 0.0
    optimisation_sum = 0.0
    target_count = 0
    diagnostic_taken = False
    diagnostics: dict[str, float] = {}

    progress = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model(
                x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
        _, _, _, absolute_error = _forecast_errors(
            output.predictions,
            batch,
            device=device,
            eps=eps,
        )
        native_loss = absolute_error.mean()
        optimisation_loss = native_loss * bps_scale
        if not torch.isfinite(optimisation_loss):
            raise FloatingPointError("Non-finite Round-2 training loss.")

        scaler.scale(optimisation_loss).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            state_modules = model.block_state_modules()
            for index, block in enumerate(model.graph_spatial_blocks):
                diagnostics[
                    f"block_{index}_graph_gradient_norm"
                ] = _module_gradient_norm(block.graph_learner)
                diagnostics[
                    f"block_{index}_alpha_gradient_norm"
                ] = _scalar_gradient(block.graph_learner.raw_alpha)
                diagnostics[
                    f"block_{index}_beta_gradient_norm"
                ] = _scalar_gradient(block.spatial_gate.raw_beta)
                diagnostics[
                    f"block_{index}_state_projection_gradient_norm"
                ] = _module_gradient_norm(state_modules[index])
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(absolute_error.numel())
        total_absolute_error += float(absolute_error.sum().item())
        optimisation_sum += float(optimisation_loss.item()) * count
        target_count += count
        progress.set_postfix(
            native=f"{total_absolute_error / max(target_count, 1):.6g}"
        )

    if target_count <= 0:
        raise RuntimeError("Training loader produced no targets.")
    return {
        "training_native_loss": total_absolute_error / target_count,
        "training_objective_loss": optimisation_sum / target_count,
        **diagnostics,
    }


def _evaluate_selection(
    *,
    model: ModernTCNGraphRound2Model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    model.eval()
    eps = float(config["normalisation"]["eps"])
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    horizon_sum = torch.zeros(len(horizons), dtype=torch.float64)
    horizon_count = torch.zeros(len(horizons), dtype=torch.float64)
    block_count = len(config["model"]["graph"]["num_heads_per_block"])
    selected_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    static_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    dynamic_stats = [_graph_stats_accumulator() for _ in range(block_count)]

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
            _, _, _, absolute_error = _forecast_errors(
                output.predictions,
                batch,
                device=device,
                eps=eps,
            )
            error = absolute_error.detach().double().cpu()
            horizon_sum += error.sum(dim=(0, 2, 3))
            horizon_count += torch.full(
                (len(horizons),),
                float(error.shape[0] * error.shape[2] * error.shape[3]),
                dtype=torch.float64,
            )
            batch_size = int(error.shape[0])
            for index, block in enumerate(output.block_outputs):
                _add_graph_stats(
                    selected_stats[index],
                    block.graph.selected,
                    batch_size=batch_size,
                )
                _add_graph_stats(
                    static_stats[index],
                    block.graph.base,
                    batch_size=batch_size,
                )
                _add_graph_stats(
                    dynamic_stats[index],
                    block.graph.dynamic,
                    batch_size=batch_size,
                )

    if torch.any(horizon_count <= 0):
        raise RuntimeError("Selection loader produced no targets.")
    by_horizon = horizon_sum / horizon_count
    result: dict[str, Any] = {
        "selection_score": float(by_horizon.mean().item()),
        "by_horizon": {
            int(horizon): float(value)
            for horizon, value in zip(
                horizons,
                by_horizon.tolist(),
                strict=True,
            )
        },
    }
    alphas = model.alphas()
    betas = model.betas()
    for index in range(block_count):
        selected_summary = _final_graph_stats(selected_stats[index])
        static_summary = _final_graph_stats(static_stats[index])
        dynamic_summary = _final_graph_stats(dynamic_stats[index])
        result[f"block_{index}_alpha"] = (
            None if alphas[index] is None else float(alphas[index].item())
        )
        result[f"block_{index}_beta"] = float(betas[index].item())
        result[f"block_{index}_selected_entropy"] = selected_summary[
            "entropy"
        ]
        result[
            f"block_{index}_selected_effective_neighbours"
        ] = selected_summary["effective_neighbours"]
        result[f"block_{index}_static_entropy"] = static_summary["entropy"]
        result[
            f"block_{index}_static_effective_neighbours"
        ] = static_summary["effective_neighbours"]
        result[f"block_{index}_dynamic_entropy"] = dynamic_summary["entropy"]
        result[
            f"block_{index}_dynamic_effective_neighbours"
        ] = dynamic_summary["effective_neighbours"]
    return result


def _history_record(
    *,
    epoch: int,
    learning_rates: Mapping[str, float | None],
    train: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "backbone_learning_rate": learning_rates.get("backbone"),
        "graph_learning_rate": learning_rates.get("graph"),
        "optimizer": "adam",
        "scheduler": "modern_tcn_type3",
        "parameter_grouping": "split",
        **dict(train),
        "selection_score": float(selection["selection_score"]),
        "selection_split": "test",
    }
    for horizon, value in selection["by_horizon"].items():
        record[f"test_cumulative_log_change_mae_h{int(horizon)}"] = float(
            value
        )
    block_count = len(config["model"]["graph"]["num_heads_per_block"])
    for index in range(block_count):
        for suffix in (
            "alpha",
            "beta",
            "selected_entropy",
            "selected_effective_neighbours",
            "static_entropy",
            "static_effective_neighbours",
            "dynamic_entropy",
            "dynamic_effective_neighbours",
        ):
            record[f"block_{index}_{suffix}"] = selection.get(
                f"block_{index}_{suffix}"
            )
    # Final-block aliases keep simple existing plots readable.
    last = block_count - 1
    record["dynamic_alpha"] = selection.get(f"block_{last}_alpha")
    record["spatial_beta"] = selection.get(f"block_{last}_beta")
    record["test_graph_mean_row_entropy"] = selection.get(
        f"block_{last}_selected_entropy"
    )
    record["test_graph_mean_effective_neighbours"] = selection.get(
        f"block_{last}_selected_effective_neighbours"
    )
    return record


def _export_selected_checkpoint(
    *,
    model: ModernTCNGraphRound2Model,
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
    block_count = model.config.num_st_blocks
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    dates: list[str] = []
    selected_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    dynamic_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    singleton_static: list[Tensor | None] = [None] * block_count

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"export {split_name}",
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
            predicted_raw, true_raw, last, _ = _forecast_errors(
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

            for index, block in enumerate(output.block_outputs):
                selected_tensor = (
                    torch.as_tensor(block.graph.selected)
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
                selected_lists[index].append(selected_tensor)
                if model.config.graph_family == "dynamic_only":
                    # selected and dynamic are exactly the same tensor.  Avoid
                    # a second in-memory copy and re-use the saved object.
                    dynamic_lists[index].append(selected_tensor)
                else:
                    dynamic_lists[index].append(
                        torch.as_tensor(block.graph.dynamic)
                        .detach()
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
                    )
                if block.graph.base is not None and singleton_static[index] is None:
                    singleton_static[index] = (
                        torch.as_tensor(block.graph.base)
                        .detach()
                        .cpu()
                        .to(torch.float16)
                        .contiguous()
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

    per_layer_selected = tuple(
        torch.cat(values, dim=0) for values in selected_lists
    )
    if model.config.graph_family == "dynamic_only":
        per_layer_dynamic = per_layer_selected
    else:
        per_layer_dynamic = tuple(
            torch.cat(values, dim=0) for values in dynamic_lists
        )
    per_layer_base = tuple(
        None if values is None else values[0].contiguous()
        for values in singleton_static
    )
    alphas = tuple(
        None
        if value is None
        else value.detach().cpu().float().reshape(1)
        for value in model.alphas()
    )
    betas = torch.stack(
        [value.detach().cpu().float().reshape(()) for value in model.betas()]
    )
    final_alpha = alphas[-1]
    final_beta = betas[-1:].contiguous()
    graph_artifacts: dict[str, Any] = {
        "graph_type": str(config["model"]["graph_family"]),
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(asset_cols),
        "num_layers": block_count,
        "num_heads": int(model.config.graph_heads_per_block[-1]),
        "num_heads_per_layer": list(model.config.graph_heads_per_block),
        "layer_head_counts": list(model.config.graph_heads_per_block),
        "graph_activations_per_layer": list(
            model.config.graph_activations_per_block
        ),
        "selected_layer": block_count - 1,
        "selected": per_layer_selected[-1],
        "per_layer": per_layer_selected,
        "base": per_layer_base[-1],
        "per_layer_base": per_layer_base,
        "dynamic": per_layer_dynamic[-1],
        "per_layer_dynamic": per_layer_dynamic,
        "alpha": final_alpha,
        "alpha_per_layer": alphas,
        "beta": final_beta,
        "beta_per_layer": betas,
        "dynamic_alpha": (
            None if final_alpha is None else float(final_alpha.item())
        ),
        "spatial_beta": float(final_beta.item()),
        "spatial_gate_type": "learned_scalar",
        "beta_trainable": True,
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
    }

    block_diagnostics: list[dict[str, Any]] = []
    for index in range(block_count):
        block_diagnostics.append(
            {
                "block": index,
                "activation": model.config.graph_activations_per_block[index],
                "heads": model.config.graph_heads_per_block[index],
                "alpha": (
                    None if alphas[index] is None else float(alphas[index].item())
                ),
                "beta": float(betas[index].item()),
                "selected_graph": graph_component_summary(
                    per_layer_selected[index].float()
                ),
                "static_graph": graph_component_summary(
                    None
                    if per_layer_base[index] is None
                    else per_layer_base[index].float()
                ),
                "dynamic_graph": graph_component_summary(
                    per_layer_dynamic[index].float()
                ),
            }
        )
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(prediction_result["y_pred"].shape[0]),
        "horizons": horizons,
        "assets": int(prediction_result["y_pred"].shape[2]),
        "temporal_family": model.config.temporal_family,
        "graph_family": model.config.graph_family,
        "blocks": block_diagnostics,
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
    root_prediction = run_dir / f"best_{split_name}_predictions.pt"
    root_graph = run_dir / f"best_{split_name}_graphs.pt"
    root_metric = run_dir / f"best_{split_name}_metric_table.csv"
    root_diagnostics = run_dir / f"best_{split_name}_diagnostics.json"

    checkpoint_epoch = int(values["diagnostics"]["checkpoint_epoch"])
    atomic_torch_save(
        {
            "epoch": checkpoint_epoch,
            "prediction_result": values["prediction_result"],
        },
        root_prediction,
    )
    atomic_torch_save(
        {
            "epoch": checkpoint_epoch,
            "graph_artifacts": values["graph_artifacts"],
        },
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
    model: ModernTCNGraphRound2Model,
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


def _save_initial_graphs(
    run_dir: Path,
    *,
    model: ModernTCNGraphRound2Model,
    source_prior: Tensor | None,
    prior_type: str,
    asset_cols: Sequence[str],
    sectors: Sequence[str] | None,
) -> None:
    per_layer = tuple(
        None
        if block.graph_learner.static_adjacency() is None
        else block.graph_learner.static_adjacency()[0].detach().cpu().float()
        for block in model.graph_spatial_blocks
    )
    if not any(value is not None for value in per_layer):
        return
    payload = {
        "prior_type": prior_type,
        "source_prior": source_prior,
        "initial_base_graphs_per_layer": per_layer,
        "asset_cols": list(asset_cols),
        "sectors": None if sectors is None else list(sectors),
        "orientation": GRAPH_ORIENTATION,
    }
    atomic_torch_save(payload, run_dir / "initial_graph_prior.pt")
    atomic_json_save(
        {
            "prior_type": prior_type,
            "asset_cols": list(asset_cols),
            "sectors": None if sectors is None else list(sectors),
            "orientation": GRAPH_ORIENTATION,
            "num_layers": len(per_layer),
            "fitted_on": (
                "random trainable logits"
                if prior_type == "none"
                else (
                    "company_profiles.csv"
                    if prior_type == "sector"
                    else "canonical January-August training Close returns only"
                )
            ),
        },
        run_dir / "initial_graph_prior.json",
    )
    if source_prior is not None:
        pd.DataFrame(
            torch.as_tensor(source_prior).cpu().numpy(),
            index=asset_cols,
            columns=asset_cols,
        ).to_csv(run_dir / "initial_graph_prior.csv")


def main() -> None:
    args = build_argument_parser().parse_args()
    resolved = _load_config(args.config)
    training = resolved["training"]
    data_config = _dataset_config(resolved)

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
        raw_train,
        raw_validation,
        raw_test,
    )
    asset_cols = list(train_split["asset_cols"])
    if asset_cols != list(validation_split["asset_cols"]):
        raise ValueError("Train and validation asset order differs.")
    if asset_cols != list(test_split["asset_cols"]):
        raise ValueError("Train and test asset order differs.")

    datasets = {
        "train": build_continuous_dataset(train_split, config=data_config),
        "validation": build_continuous_dataset(
            validation_split,
            config=data_config,
        ),
        "test": build_continuous_dataset(test_split, config=data_config),
    }
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"The configured {name} dataset has no windows.")

    model_config: ModernTCNGraphRound2Config = round2_model_config_from_mapping(
        resolved,
        num_nodes=len(asset_cols),
    )
    prior_type = str(resolved["model"]["prior"]["type"])
    source_prior: Tensor | None = None
    sectors: list[str] | None = None
    if model_config.uses_static_graph:
        if prior_type == "sector":
            if args.company_profiles is None:
                raise ValueError("A sector prior requires --company-profiles.")
            source_prior, sectors = build_sector_graph_prior(
                asset_cols,
                args.company_profiles,
            )
        elif prior_type == "correlation":
            source_prior = build_absolute_correlation_graph_prior(
                train_split,
                expected_asset_cols=asset_cols,
            )
        elif prior_type == "none":
            source_prior = None
        else:
            raise ValueError(f"Unsupported prior type {prior_type!r}.")
    elif prior_type != "none":
        raise ValueError("Dynamic-only family must use prior_type='none'.")

    model = ModernTCNGraphRound2Model(
        model_config,
        static_prior=source_prior,
    ).to(device)
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)

    signature_values = {
        "config": resolved,
        "asset_cols": asset_cols,
        "train_dates": [str(sample[2]) for sample in train_split["samples"]],
        "validation_dates": [
            str(sample[2]) for sample in validation_split["samples"]
        ],
        "test_dates": [str(sample[2]) for sample in test_split["samples"]],
    }
    run_signature = _signature(signature_values)
    backbone_parameters, graph_parameters = _parameter_partition(model)
    created = datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "status": "running",
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": (
            "strict mean cumulative-log-change MAE over all configured horizons"
        ),
        "created_at_utc": created,
        "run_name": args.run_name,
        "run_signature": run_signature,
        "project_git_commit": _git_value(["rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": _git_value(
            ["branch", "--show-current"], cwd=project_root
        ),
        "device": str(device),
        "mixed_precision": use_amp,
        "asset_cols": asset_cols,
        "train_sessions": len(train_split["samples"]),
        "validation_sessions": len(validation_split["samples"]),
        "test_sessions": len(test_split["samples"]),
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "context_length": int(resolved["data"]["context_length"]),
        "stride": int(resolved["data"]["stride"]),
        "horizons": [int(value) for value in resolved["data"]["horizons"]],
        "model_family": "modern_tcn_graph_round2",
        "temporal_family": model_config.temporal_family,
        "num_transformer_blocks": model_config.num_transformer_blocks,
        "num_st_blocks": model_config.num_st_blocks,
        "graph_family": model_config.graph_family,
        "graph_heads_per_block": list(model_config.graph_heads_per_block),
        "graph_hidden_dims_per_block": list(
            model_config.graph_hidden_dims_per_block
        ),
        "graph_activations_per_block": list(
            model_config.graph_activations_per_block
        ),
        "prior_type": prior_type,
        "prior_scale": float(model_config.prior_scale),
        "prior_jitter": float(model_config.prior_jitter),
        "state_pathway": model_config.uses_state_pathway,
        "alpha_initial": float(model_config.graph_initial_alpha),
        "beta_initial": float(model_config.spatial_initial_beta),
        "optimizer": "adam",
        "parameter_grouping": "split",
        "scheduler": "modern_tcn_type3",
        "backbone_learning_rate": float(training["learning_rate"]),
        "graph_learning_rate": float(training["graph_learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "backbone_trainable_parameters": int(
            sum(parameter.numel() for parameter in backbone_parameters)
        ),
        "graph_trainable_parameters": int(
            sum(parameter.numel() for parameter in graph_parameters)
        ),
    }
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        "This run uses the test split for checkpoint selection.\n",
        encoding="utf-8",
    )
    _save_initial_graphs(
        run_dir,
        model=model,
        source_prior=source_prior,
        prior_type=prior_type,
        asset_cols=asset_cols,
        sectors=sectors,
    )

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    training_complete = False
    last_epoch = 0
    if args.resume:
        checkpoint_path = run_dir / "last_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["run_signature"] != run_signature:
            raise ValueError("Resume signature differs from requested run.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        last_epoch = int(checkpoint["epoch"])
        start_epoch = last_epoch + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(checkpoint["evaluations_without_improvement"])
        history = list(checkpoint["history"])
        training_complete = bool(checkpoint.get("training_complete", False))
        restore_rng_state(checkpoint["rng_state"])

    selection_loader = _build_loader(
        datasets["test"],
        batch_size=int(training["selection_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    wandb_run = _init_wandb(args, resolved)

    try:
        if not training_complete:
            max_epochs = int(training["max_epochs"])
            patience = int(training["patience"])
            min_delta = float(training["min_delta"])
            for epoch in range(start_epoch, max_epochs + 1):
                last_epoch = epoch
                current_lrs = _learning_rates(optimizer)
                train_values = _train_epoch(
                    model=model,
                    dataset=datasets["train"],
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=use_amp,
                    config=resolved,
                    epoch=epoch,
                )
                selection = _evaluate_selection(
                    model=model,
                    loader=selection_loader,
                    device=device,
                    use_amp=use_amp,
                    config=resolved,
                    description=f"test selection epoch {epoch}",
                )
                score = float(selection["selection_score"])
                record = _history_record(
                    epoch=epoch,
                    learning_rates=current_lrs,
                    train=train_values,
                    selection=selection,
                    config=resolved,
                )
                history.append(record)
                atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")

                improved = score < best_score - min_delta
                if improved:
                    best_score = score
                    best_epoch = epoch
                    without_improvement = 0
                    atomic_torch_save(
                        _checkpoint(
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
                        ),
                        run_dir / "best_checkpoint.pt",
                    )
                else:
                    without_improvement += 1

                _advance_schedule(optimizer, completed_epoch=epoch)
                atomic_torch_save(
                    _checkpoint(
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
                    ),
                    run_dir / "last_checkpoint.pt",
                )
                if wandb_run is not None:
                    wandb_run.log(record, step=epoch)

                final_index = model.config.num_st_blocks - 1
                print(
                    f"epoch={epoch} train={train_values['training_native_loss']:.8g} "
                    f"test_mean={score:.8g} best={best_score:.8g} "
                    f"best_epoch={best_epoch} "
                    f"final_alpha={selection.get(f'block_{final_index}_alpha')} "
                    f"final_beta={selection[f'block_{final_index}_beta']:.4f} "
                    f"backbone_lr={current_lrs['backbone']:.3g} "
                    f"graph_lr={current_lrs['graph']:.3g}"
                )
                if without_improvement >= patience:
                    print(f"Early stopping after epoch {epoch}.")
                    break

            if best_epoch <= 0 or not (run_dir / "best_checkpoint.pt").is_file():
                raise RuntimeError("Training produced no selected checkpoint.")
            training_complete = True
            atomic_torch_save(
                _checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=last_epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    without_improvement=without_improvement,
                    history=history,
                    run_signature=run_signature,
                    config=resolved,
                    metadata=metadata,
                    training_complete=True,
                ),
                run_dir / "last_checkpoint.pt",
            )

        best_checkpoint = torch.load(
            run_dir / "best_checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        if best_checkpoint["run_signature"] != run_signature:
            raise ValueError("Best-checkpoint signature differs from requested run.")
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        model.to(device)
        best_epoch = int(best_checkpoint["best_epoch"])
        best_score = float(best_checkpoint["best_score"])

        for split_index, split_name in enumerate(("train", "validation", "test")):
            loader = _build_loader(
                datasets[split_name],
                batch_size=int(training["export_batch_size"]),
                shuffle=False,
                num_workers=int(training["num_workers"]),
                seed=seed + 1000 + split_index,
                pin_memory=device.type == "cuda",
            )
            exported = _export_selected_checkpoint(
                model=model,
                loader=loader,
                split_name=split_name,
                device=device,
                use_amp=use_amp,
                config=resolved,
                train_split=train_split,
                asset_cols=asset_cols,
                checkpoint_epoch=best_epoch,
            )
            _save_export(run_dir, split_name=split_name, values=exported)

        final_alphas = [
            None if value is None else float(value.item()) for value in model.alphas()
        ]
        final_betas = [float(value.item()) for value in model.betas()]
        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": int(last_epoch),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "final_alphas": final_alphas,
                "final_betas": final_betas,
                "final_alpha": final_alphas[-1],
                "final_beta": final_betas[-1],
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        print("ROUND-2 RUN COMPLETE")
        print("Run:", args.run_name)
        print("Best epoch:", best_epoch)
        print("Best test all-horizon mean Log MAE:", best_score)
    except BaseException as error:
        failed = dict(metadata)
        failed.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "best_epoch": int(best_epoch),
                "best_score": (
                    None if not np.isfinite(best_score) else float(best_score)
                ),
                "epochs_completed": int(last_epoch),
            }
        )
        atomic_json_save(failed, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

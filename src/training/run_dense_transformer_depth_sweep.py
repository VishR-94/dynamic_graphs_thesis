from __future__ import annotations

"""Train one stacked dense-Transformer graph model from the 12-run sweep."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data.continuous_forecast_dataset import build_continuous_dataset
from src.data.dense_parallel_forecast_dataset import (
    DensePrefixMultiHorizonDataset,
    build_dense_prefix_dataset,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.models.dense_transformer_depth_sweep import (
    GRAPH_ORIENTATION,
    DenseTransformerDepthSequenceOutput,
    StackedDenseTransformerGraphModel,
    dense_transformer_depth_config_from_mapping,
)
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.training.run_dense_parallel_graph_supervision import (
    _advance_schedule,
    _autocast_context,
    _build_loader,
    _continuous_dataset_config,
    _dense_absolute_error,
    _git_value,
    _horizon_weights,
    _init_wandb,
    _learning_rates,
    _move_optimizer_state,
    _new_grad_scaler,
    _prepare_run_dir,
    _signature,
    _standard_absolute_error,
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    set_seed,
    synchronise_device,
)
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one stacked dense Transformer graph curiosity model."
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
    if str(config.get("model_family")) != "dense_transformer_depth_sweep":
        raise ValueError("Unexpected model_family for this runner.")

    data = config["data"]
    model = config["model"]
    training = config["training"]
    horizons = tuple(int(value) for value in data["horizons"])
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("data.horizons must be non-empty, unique and increasing.")
    if int(data["context_length"]) <= 0:
        raise ValueError("data.context_length must be positive.")
    if int(data["dense_prefix_outer_stride"]) <= 0:
        raise ValueError("dense_prefix_outer_stride must be positive.")
    if int(data["export_stride"]) <= 0:
        raise ValueError("export_stride must be positive.")
    if str(data["target_channel"]).lower() != "close":
        raise ValueError("This sweep predicts Close only.")
    if str(data.get("input_representation", "raw")) != "raw":
        raise ValueError("This sweep uses context-normalised raw OHLCV.")

    if str(model["variant"]) != "uniform_static_dynamic_state":
        raise ValueError("This sweep fixes uniform static + dynamic + state.")
    if str(model["prior"]["type"]) != "uniform":
        raise ValueError("Every graph branch must start from a neutral uniform adjacency.")
    if str(model["graph"]["type"]) != "static_dynamic_mixture":
        raise ValueError("Every run must contain static and dynamic graphs.")
    if bool(model["graph"].get("add_self_loops", False)):
        raise ValueError("This sweep uses zero-diagonal graphs.")
    blocks = int(model["num_st_blocks"])
    if blocks <= 0:
        raise ValueError("model.num_st_blocks must be positive.")
    heads = tuple(int(value) for value in model["graph"]["num_heads_per_block"])
    widths = tuple(int(value) for value in model["graph"]["hidden_dims_per_block"])
    activations = tuple(str(value) for value in model["graph"]["activations_per_block"])
    if not (len(heads) == len(widths) == len(activations) == blocks):
        raise ValueError("Per-block graph schedules must match num_st_blocks.")
    if activations[-1] != "sparsemax" or any(value != "softmax" for value in activations[:-1]):
        raise ValueError("Graph activations must be softmax ... sparsemax.")
    for index, (head_count, width) in enumerate(zip(heads, widths, strict=True)):
        if head_count <= 0 or width <= 0 or width % head_count:
            raise ValueError(
                f"Invalid graph heads/width at block {index}: "
                f"heads={head_count}, width={width}."
            )
    if str(model["spatial"]["gate_type"]) != "learned_scalar":
        raise ValueError("Every block must retain the learned beta gate.")
    if any(
        float(model["graph_regularisation"].get(key, 0.0)) != 0.0
        for key in (
            "graph_entropy_reg",
            "graph_target_entropy_reg",
            "graph_temporal_smooth_reg",
        )
    ):
        raise ValueError("This depth sweep has no graph regularisation.")

    if str(training["training_style"]) != "dense_prefix":
        raise ValueError("This runner supports dense-prefix training only.")
    if str(training["selection_split"]) != "test":
        raise ValueError("This curiosity sweep deliberately selects on test.")
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
    factor = float(training["scheduler_decay_factor"])
    if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("scheduler_decay_factor must lie in (0,1].")
    loss = training["loss"]
    weights = tuple(float(value) for value in loss["horizon_weights"])
    references = tuple(float(value) for value in loss["horizon_reference_mae"])
    if len(weights) != len(horizons) or len(references) != len(horizons):
        raise ValueError("Loss weights/reference MAEs must match horizons.")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("Horizon weights must be positive and finite.")
    for key in (
        "batch_size",
        "selection_batch_size",
        "export_batch_size",
        "max_epochs",
        "patience",
    ):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    for key in ("learning_rate", "graph_learning_rate"):
        if float(training[key]) <= 0.0:
            raise ValueError(f"training.{key} must be positive.")

    # Validate the exact model tensor contract as the final config check.
    dense_transformer_depth_config_from_mapping(dict(config))


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
        stride=int(data["dense_prefix_outer_stride"]),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channel=str(data["target_channel"]),
        eps=float(normalisation["eps"]),
        clip=bool(normalisation["clip"]),
        clip_min=float(normalisation["clip_min"]),
        clip_max=float(normalisation["clip_max"]),
    )


def _parameter_partition(
    model: StackedDenseTransformerGraphModel,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    graph_ids = model.graph_parameter_ids()
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
        raise AssertionError("Optimizer parameter partition lost parameters.")
    if {id(value) for value in graph} & {id(value) for value in backbone}:
        raise AssertionError("Optimizer parameter groups overlap.")
    if not graph or not backbone:
        raise RuntimeError("Both optimizer parameter groups must be non-empty.")
    return backbone, graph


def _build_optimizer(
    model: StackedDenseTransformerGraphModel,
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    training = config["training"]
    backbone, graph = _parameter_partition(model)
    return torch.optim.Adam(
        [
            {
                "params": backbone,
                "lr": float(training["learning_rate"]),
                "base_lr": float(training["learning_rate"]),
                "name": "backbone",
            },
            {
                "params": graph,
                "lr": float(training["graph_learning_rate"]),
                "base_lr": float(training["graph_learning_rate"]),
                "name": "graph",
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )


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
    if count <= 0.0:
        return {"entropy": None, "effective_neighbours": None}
    return {
        "entropy": float(accumulator["entropy_sum"]) / count,
        "effective_neighbours": float(accumulator["effective_sum"]) / count,
    }


def _train_epoch(
    *,
    model: StackedDenseTransformerGraphModel,
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
    diagnostics: dict[str, float] = {}

    progress = tqdm(
        loader,
        desc=f"train dense depth epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        x = torch.as_tensor(batch["x"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        dense_true = torch.as_tensor(batch["dense_y_unnormalised"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        dense_current = torch.as_tensor(batch["dense_current_close"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
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
            raise FloatingPointError("Non-finite dense depth-sweep training loss.")
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)

        if not diagnostic_taken:
            diagnostics["shared_state_projection_gradient_norm"] = (
                _module_gradient_norm(model.state_projection)
            )
            for index, block in enumerate(model.blocks):
                diagnostics[f"block_{index}_graph_gradient_norm"] = (
                    _module_gradient_norm(block.graph_learner)
                )
                diagnostics[f"block_{index}_alpha_gradient_norm"] = (
                    _scalar_gradient(block.graph_learner.raw_alpha)
                )
                diagnostics[f"block_{index}_beta_gradient_norm"] = (
                    _scalar_gradient(block.spatial_gate.raw_beta)
                )
            diagnostic_taken = True

        clip = float(training["gradient_clip_norm"])
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(absolute_error.numel())
        unweighted_sum += float(absolute_error.detach().sum().item())
        weighted_sum += float(weighted_error.detach().sum().item())
        objective_sum += float(objective.detach().item()) * count
        target_count += count
        progress.set_postfix(native=f"{unweighted_sum / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Dense training loader produced no targets.")
    return {
        "training_native_loss": unweighted_sum / target_count,
        "training_weighted_native_loss": weighted_sum / target_count,
        "training_objective_loss": objective_sum / target_count,
        **diagnostics,
    }


def _evaluate_selection(
    *,
    model: StackedDenseTransformerGraphModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    model.eval()
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    eps = float(config["normalisation"]["eps"])
    horizon_sum = torch.zeros(len(horizons), dtype=torch.float64)
    horizon_count = torch.zeros(len(horizons), dtype=torch.float64)
    block_count = int(config["model"]["num_st_blocks"])
    selected_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    static_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    dynamic_stats = [_graph_stats_accumulator() for _ in range(block_count)]

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _autocast_context(device, use_amp):
                output = model.forward_dense(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            predictions = output.final_predictions()
            _, _, _, absolute_error = _standard_absolute_error(
                predictions,
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
                    block.graph.selected[:, -1],
                    batch_size=batch_size,
                )
                _add_graph_stats(
                    static_stats[index],
                    block.graph.base,
                    batch_size=batch_size,
                )
                _add_graph_stats(
                    dynamic_stats[index],
                    block.graph.dynamic[:, -1],
                    batch_size=batch_size,
                )

    if torch.any(horizon_count <= 0):
        raise RuntimeError("Selection loader produced no targets.")
    by_horizon = horizon_sum / horizon_count
    result: dict[str, Any] = {
        "selection_score": float(by_horizon.mean().item()),
        "by_horizon": {
            int(horizon): float(value)
            for horizon, value in zip(horizons, by_horizon.tolist(), strict=True)
        },
    }
    alphas = model.alphas()
    betas = model.betas()
    for index in range(block_count):
        selected_summary = _final_graph_stats(selected_stats[index])
        static_summary = _final_graph_stats(static_stats[index])
        dynamic_summary = _final_graph_stats(dynamic_stats[index])
        result[f"block_{index}_alpha"] = float(alphas[index].detach().item())
        result[f"block_{index}_beta"] = float(betas[index].detach().item())
        result[f"block_{index}_selected_entropy"] = selected_summary["entropy"]
        result[f"block_{index}_selected_effective_neighbours"] = (
            selected_summary["effective_neighbours"]
        )
        result[f"block_{index}_static_entropy"] = static_summary["entropy"]
        result[f"block_{index}_static_effective_neighbours"] = (
            static_summary["effective_neighbours"]
        )
        result[f"block_{index}_dynamic_entropy"] = dynamic_summary["entropy"]
        result[f"block_{index}_dynamic_effective_neighbours"] = (
            dynamic_summary["effective_neighbours"]
        )
    return result


def _history_record(
    *,
    epoch: int,
    learning_rates: Mapping[str, float | None],
    train: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: Mapping[str, Any],
    epoch_seconds: float,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "backbone_learning_rate": learning_rates.get("backbone"),
        "graph_learning_rate": learning_rates.get("graph"),
        "optimizer": "adam",
        "scheduler": "modern_tcn_type3_delayed",
        "parameter_grouping": "split",
        "epoch_seconds": float(epoch_seconds),
        **dict(train),
        "selection_score": float(selection["selection_score"]),
        "selection_split": "test",
    }
    for horizon, value in selection["by_horizon"].items():
        record[f"test_cumulative_log_change_mae_h{int(horizon)}"] = float(value)
    block_count = int(config["model"]["num_st_blocks"])
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
    model: StackedDenseTransformerGraphModel,
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
    horizons = [int(value) for value in config["data"]["horizons"]]
    eps = float(config["normalisation"]["eps"])
    block_count = int(config["model"]["num_st_blocks"])

    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    dates: list[str] = []
    selected_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    dynamic_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    singleton_static: list[Tensor | None] = [None for _ in range(block_count)]

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
                output = model.forward_dense(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            predicted_raw, true_raw, last, _ = _standard_absolute_error(
                output.final_predictions(),
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
                selected_lists[index].append(
                    block.graph.selected[:, -1]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
                dynamic_lists[index].append(
                    block.graph.dynamic[:, -1]
                    .detach()
                    .cpu()
                    .to(torch.float16)
                    .contiguous()
                )
                if singleton_static[index] is None:
                    singleton_static[index] = (
                        block.graph.base.detach().cpu().to(torch.float16).contiguous()
                    )

    prediction_result = {
        "y_pred": torch.cat(predictions, dim=0),
        "y_true": torch.cat(targets, dim=0),
        "last_context_target": torch.cat(last_values, dim=0),
        "channels": [str(config["data"]["target_channel"])],
        "horizons": horizons,
        "asset_cols": list(asset_cols),
        "sample_idx": torch.cat(sample_indices).long(),
        "origin_idx": torch.cat(origin_indices).long(),
        "target_indices": torch.cat(target_indices).long(),
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

    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    per_layer_base = tuple(
        None if value is None else value[0].contiguous()
        for value in singleton_static
    )
    alphas = tuple(value.detach().cpu().float().reshape(1) for value in model.alphas())
    betas = torch.stack(
        [value.detach().cpu().float().reshape(()) for value in model.betas()]
    )
    final_alpha = alphas[-1]
    final_beta = betas[-1:].contiguous()
    graph_config = config["model"]["graph"]
    graph_artifacts: dict[str, Any] = {
        "graph_type": "static_dynamic_mixture",
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(asset_cols),
        "num_layers": block_count,
        "num_heads": int(graph_config["num_heads_per_block"][-1]),
        "num_heads_per_layer": list(graph_config["num_heads_per_block"]),
        "layer_head_counts": list(graph_config["num_heads_per_block"]),
        "graph_hidden_dims_per_layer": list(
            graph_config["hidden_dims_per_block"]
        ),
        "graph_activations_per_layer": list(
            graph_config["activations_per_block"]
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
        "dynamic_alpha": float(final_alpha.item()),
        "spatial_beta": float(final_beta.item()),
        "spatial_gate_type": "learned_scalar",
        "beta_trainable": True,
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
    }

    blocks: list[dict[str, Any]] = []
    for index in range(block_count):
        blocks.append(
            {
                "block": index,
                "activation": graph_config["activations_per_block"][index],
                "heads": graph_config["num_heads_per_block"][index],
                "graph_hidden_dim": graph_config["hidden_dims_per_block"][index],
                "alpha": float(alphas[index].item()),
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
        "temporal_family": "transformer",
        "training_style": "dense_prefix",
        "graph_family": "uniform_static_dynamic_state",
        "blocks": blocks,
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


def _save_initial_graphs(
    run_dir: Path,
    *,
    model: StackedDenseTransformerGraphModel,
    asset_cols: Sequence[str],
) -> None:
    per_layer = model.initial_base_graphs()
    payload = {
        "prior_type": "uniform",
        "source_prior": None,
        "initial_base_graphs_per_layer": per_layer,
        "asset_cols": list(asset_cols),
        "orientation": GRAPH_ORIENTATION,
        "fitted_on": "neutral zero logits",
    }
    atomic_torch_save(payload, run_dir / "initial_graph_prior.pt")
    atomic_json_save(
        {
            "prior_type": "uniform",
            "asset_cols": list(asset_cols),
            "orientation": GRAPH_ORIENTATION,
            "num_layers": len(per_layer),
            "fitted_on": "neutral zero logits",
        },
        run_dir / "initial_graph_prior.json",
    )
    for index, graph in enumerate(per_layer):
        averaged = torch.as_tensor(graph)[0].mean(dim=0).numpy()
        pd.DataFrame(averaged, index=asset_cols, columns=asset_cols).to_csv(
            run_dir / f"initial_static_graph_block_{index}.csv"
        )


def _prefix_graph_diagnostics(
    *,
    model: StackedDenseTransformerGraphModel,
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        seed=int(config["training"]["seed"]),
        pin_memory=device.type == "cuda",
    )
    block_count = int(config["model"]["num_st_blocks"])
    selected_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    dynamic_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    dates: list[str] = []
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            x = torch.as_tensor(batch["x"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _autocast_context(device, use_amp):
                output = model.forward_dense(
                    x,
                    context_start=batch["context_start"],
                    session_length=batch["session_length"],
                )
            for index, block in enumerate(output.block_outputs):
                selected_lists[index].append(
                    block.graph.selected.detach().cpu().to(torch.float16)
                )
                dynamic_lists[index].append(
                    block.graph.dynamic.detach().cpu().to(torch.float16)
                )
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).cpu())
            batch_days = batch["day"]
            if isinstance(batch_days, str):
                dates.append(batch_days)
            else:
                dates.extend(str(value) for value in batch_days)

    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    per_layer_base = tuple(
        graph[0].to(torch.float16).contiguous()
        for graph in model.initial_base_graphs()
    )
    rows: list[dict[str, Any]] = []
    context_length = int(config["data"]["context_length"])
    for block_index in range(block_count):
        selected = per_layer_selected[block_index].float()
        dynamic = per_layer_dynamic[block_index].float()
        final_selected = selected[:, -1]
        final_dynamic = dynamic[:, -1]
        for position in range(context_length):
            selected_summary = graph_component_summary(selected[:, position])
            dynamic_summary = graph_component_summary(dynamic[:, position])
            rows.append(
                {
                    "block": block_index,
                    "prefix_length": position + 1,
                    "selected_mean_row_entropy": selected_summary[
                        "mean_row_entropy"
                    ],
                    "selected_effective_neighbours": selected_summary[
                        "mean_effective_neighbours"
                    ],
                    "dynamic_mean_row_entropy": dynamic_summary[
                        "mean_row_entropy"
                    ],
                    "dynamic_effective_neighbours": dynamic_summary[
                        "mean_effective_neighbours"
                    ],
                    "selected_mean_absolute_distance_to_final": float(
                        (selected[:, position] - final_selected).abs().mean().item()
                    ),
                    "dynamic_mean_absolute_distance_to_final": float(
                        (dynamic[:, position] - final_dynamic).abs().mean().item()
                    ),
                }
            )

    payload = {
        "epoch": int(checkpoint_epoch),
        "per_layer": per_layer_selected,
        "per_layer_dynamic": per_layer_dynamic,
        "per_layer_base": per_layer_base,
        "alpha_per_layer": tuple(
            value.detach().cpu().float().reshape(1) for value in model.alphas()
        ),
        "beta_per_layer": torch.stack(
            [value.detach().cpu().float().reshape(()) for value in model.betas()]
        ),
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


def _checkpoint(
    *,
    model: StackedDenseTransformerGraphModel,
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
        raw_train,
        raw_validation,
        raw_test,
    )
    asset_cols = list(train_split["asset_cols"])
    if asset_cols != list(validation_split["asset_cols"]) or asset_cols != list(
        test_split["asset_cols"]
    ):
        raise ValueError("Asset ordering differs across chronological splits.")

    # The node count is a derived data property, not a notebook execution gate.
    resolved = json.loads(json.dumps(resolved))
    resolved["model"]["num_nodes"] = len(asset_cols)
    _validate_config(resolved)
    model_config = dense_transformer_depth_config_from_mapping(resolved)

    training_dataset = _dense_dataset(train_split, resolved)
    export_stride = int(resolved["data"]["export_stride"])
    export_config = _continuous_dataset_config(resolved, stride=export_stride)
    export_datasets = {
        "train": build_continuous_dataset(train_split, config=export_config),
        "validation": build_continuous_dataset(
            validation_split,
            config=export_config,
        ),
        "test": build_continuous_dataset(test_split, config=export_config),
    }

    model = StackedDenseTransformerGraphModel(model_config).to(device)
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)
    run_signature = _signature(resolved)

    metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "run_signature": run_signature,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _git_value(["rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": _git_value(
            ["branch", "--show-current"], cwd=project_root
        ),
        "model_family": "dense_transformer_depth_sweep",
        "experiment_family": "dense_transformer_depth_sweep",
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": str(training["selection_metric"]),
        "training_style": "dense_prefix",
        "temporal_backbone": "transformer",
        "num_st_blocks": int(model_config.num_st_blocks),
        "d_model": int(model_config.d_model),
        "transformer_num_layers": int(model_config.transformer_num_layers),
        "transformer_num_heads": int(model_config.transformer_num_heads),
        "graph_heads_per_layer": list(model_config.graph_heads_per_block),
        "graph_hidden_dims_per_layer": list(
            model_config.graph_hidden_dims_per_block
        ),
        "graph_activations_per_layer": list(
            model_config.graph_activations_per_block
        ),
        "graph_type": "static_dynamic_mixture",
        "prior_type": "uniform",
        "initial_static_graph": "uniform_off_diagonal",
        "initial_dynamic_graph": "uniform_off_diagonal",
        "state_pathway": True,
        "graph_initial_alpha": float(model_config.graph_initial_alpha),
        "spatial_initial_beta": float(model_config.spatial_initial_beta),
        "context_length": int(model_config.context_length),
        "horizons": list(model_config.horizons),
        "asset_cols": list(asset_cols),
        "input_channels": list(model_config.input_channels),
        "train_sessions": len(train_split["samples"]),
        "validation_sessions": len(validation_split["samples"]),
        "test_sessions": len(test_split["samples"]),
        "dense_training_windows": len(training_dataset),
        "train_windows": len(export_datasets["train"]),
        "validation_windows": len(export_datasets["validation"]),
        "test_windows": len(export_datasets["test"]),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "graph_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
                and id(parameter) in model.graph_parameter_ids()
            )
        ),
    }
    metadata["backbone_trainable_parameters"] = (
        metadata["trainable_parameters"] - metadata["graph_trainable_parameters"]
    )
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    _save_initial_graphs(run_dir, model=model, asset_cols=asset_cols)

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
            train_values = _train_epoch(
                model=model,
                dataset=training_dataset,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                config=resolved,
                epoch=epoch,
            )
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

            if improved:
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

            if wandb_run is not None:
                wandb_run.log(record, step=epoch)
            alpha_text = ", ".join(
                f"a{index}={selection[f'block_{index}_alpha']:.3f}"
                for index in range(int(model_config.num_st_blocks))
            )
            beta_text = ", ".join(
                f"b{index}={selection[f'block_{index}_beta']:.3f}"
                for index in range(int(model_config.num_st_blocks))
            )
            print(
                f"Epoch {epoch:03d} | train={train_values['training_native_loss']:.8f} "
                f"| test_mean={selection['selection_score']:.8f} "
                f"| {alpha_text} | {beta_text}"
            )
            last_epoch = int(epoch)
            _advance_schedule(
                optimizer,
                training=training,
                completed_epoch=epoch,
            )
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
                last_checkpoint_path,
            )
            if without_improvement >= int(training["patience"]):
                print(f"Early stopping after epoch {epoch}.")
                break

        if best_epoch <= 0 or not (run_dir / "best_checkpoint.pt").is_file():
            raise RuntimeError("Training produced no selected checkpoint.")
        training_complete = True
        final_last = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
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

    _prefix_graph_diagnostics(
        model=model,
        dataset=training_dataset,
        run_dir=run_dir,
        device=device,
        use_amp=use_amp,
        config=resolved,
        checkpoint_epoch=checkpoint_epoch,
    )

    alphas = [float(value.detach().item()) for value in model.alphas()]
    betas = [float(value.detach().item()) for value in model.betas()]
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": checkpoint_epoch,
            "best_score": float(best_checkpoint["best_score"]),
            "epochs_completed": int(last_epoch or best_checkpoint["epoch"]),
            "final_alpha_per_layer": alphas,
            "final_beta_per_layer": betas,
            "final_alpha": alphas[-1],
            "final_beta": betas[-1],
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

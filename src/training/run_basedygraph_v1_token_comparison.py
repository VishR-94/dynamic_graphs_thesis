from __future__ import annotations

"""Train the two pinned BaseDyGraph-v1 coarse-token controls.

This is an intentionally test-selected curiosity experiment:

* January-August token windows supply gradient updates;
* October-December test windows control early stopping and checkpoint choice;
* September validation is exported only after checkpoint selection.

``dense_one_step`` uses the official next-state head and maximises mean top-1
accuracy over every teacher-forced transition in ``context + true_future_h1``.
``parallel_60`` uses the same official backbone with the project's structured
parallel head and maximises mean top-1 accuracy over all 60 future positions.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.models.basedygraph_v1_token_comparison import (
    PINNED_BASEDYGRAPH_COMMIT,
    BaseDyGraphV1TokenModel,
    basedygraph_v1_token_config_from_mapping,
)
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.training.run_dynamic_graph import (
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    resolve_device,
    restore_rng_state,
    set_seed,
)
from src.training.run_modern_tcn_graph_round2_token import (
    TOP_K_VALUES,
    _add_graph_stats,
    _autocast_context,
    _build_loader,
    _checkpoint,
    _final_graph_stats,
    _git_value,
    _graph_stats_accumulator,
    _init_wandb,
    _learning_rates,
    _move_optimizer_state,
    _new_grad_scaler,
    _prepare_run_dir,
    _save_export,
    _set_schedule_for_epoch,
    _signature,
    _token_batch_sums,
)


GRAPH_ORIENTATION = "A[target, source]"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one pinned BaseDyGraph-v1 token control."
    )
    parser.add_argument("--config", type=Path, required=True)
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


def _load_config(path: Path) -> dict[str, Any]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    _validate_config(values)
    return values


def _validate_config(values: Mapping[str, Any]) -> None:
    for key in ("data", "model", "models", "training"):
        if not isinstance(values.get(key), Mapping):
            raise KeyError(f"Config must contain mapping {key!r}.")
    data = values["data"]
    model = values["model"]
    architecture = model["official_basedygraph_v1"]
    training = values["training"]
    if str(data["input_token_stream"]) != "s1" or str(
        data["target_token_stream"]
    ) != "s1":
        raise ValueError("This comparison uses coarse s1 input and output only.")
    if int(data["s1_vocabulary_size"]) != 1024:
        raise ValueError("The comparison uses the original 1,024-way s1 space.")
    mode = str(model["prediction_mode"])
    if mode not in {"dense_one_step", "parallel_60"}:
        raise ValueError(f"Unsupported prediction mode {mode!r}.")
    if str(architecture["spatial_module_type"]) != "dynamic_graph":
        raise ValueError("The BaseDyGraph-v1 controls require dynamic_graph.")
    if str(architecture["spatial_value"]) != "hidden":
        raise ValueError("The exact v1 path uses hidden-only spatial values.")
    if str(architecture["graph_activation"]) != "softmax":
        raise ValueError("The exact v1 path uses softmax in every ST block.")
    if int(architecture["num_st_blocks"]) != 4:
        raise ValueError("These two controls use four ST blocks.")
    if bool(architecture["use_state_pair_bias"]):
        raise ValueError("The exact v1 control excludes state-pair bias.")
    if bool(architecture["add_self_loops"]):
        raise ValueError("The exact v1 control adds no extra identity matrix.")
    if str(training["selection_split"]) != "test":
        raise ValueError("This curiosity run selects on the test cache.")
    expected_metric = (
        "dense_teacher_forced_mean_top1_accuracy"
        if mode == "dense_one_step"
        else "mean_top1_accuracy_over_all_future_steps"
    )
    if str(training["selection_metric"]) != expected_metric:
        raise ValueError("Unexpected checkpoint-selection metric.")
    if str(training["selection_direction"]) != "maximise":
        raise ValueError("Top-1 checkpoint selection must be maximised.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("The controls preserve the current Adam optimiser.")
    if str(training["scheduler"]) != "modern_tcn_type3_delayed":
        raise ValueError("The controls preserve the delayed LR schedule.")
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


def _load_datasets(args: argparse.Namespace) -> dict[str, CachedTokenGraphDataset]:
    datasets = {
        "train": CachedTokenGraphDataset.from_path(args.train_cache),
        "validation": CachedTokenGraphDataset.from_path(args.validation_cache),
        "test": CachedTokenGraphDataset.from_path(args.test_cache),
    }
    reference = datasets["train"]
    for split_name, dataset in datasets.items():
        if dataset.data_mode != "real":
            raise ValueError(f"{split_name} cache must be real data.")
        if dataset.asset_cols != reference.asset_cols:
            raise ValueError(f"{split_name} asset order differs from train.")
        if dataset.context_length != reference.context_length:
            raise ValueError(f"{split_name} context length differs from train.")
        if dataset.prediction_length != reference.prediction_length:
            raise ValueError(f"{split_name} prediction length differs from train.")
        if dataset.s1_id_space != "kronos_original":
            raise ValueError(f"{split_name} cache is not in original s1 space.")
        if dataset.s1_vocabulary_size != 1024:
            raise ValueError(f"{split_name} cache is not 1,024-way s1.")
    return datasets


def _batch_inputs(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    context = torch.as_tensor(batch["context_tokens"])[..., 0].to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    future = torch.as_tensor(batch["target_s1"]).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    return context, future


def _forward_batch(
    model: BaseDyGraphV1TokenModel,
    context: Tensor,
    future: Tensor,
):
    if model.config.prediction_mode == "dense_one_step":
        output = model(context, first_future_s1=future[:, 0])
        target = output.teacher_forced_targets
        if target is None:
            raise RuntimeError("Dense BaseDyGraph returned no teacher targets.")
        return output, target
    return model(context), future


def _parameter_partition(
    model: BaseDyGraphV1TokenModel,
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
    all_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(all_trainable) != len(backbone) + len(graph):
        raise AssertionError("Optimizer partition lost trainable parameters.")
    if {id(value) for value in backbone} & {id(value) for value in graph}:
        raise AssertionError("Optimizer parameter groups overlap.")
    return backbone, graph


def _build_optimizer(
    model: BaseDyGraphV1TokenModel,
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
    return torch.optim.Adam(
        groups,
        weight_decay=float(training["weight_decay"]),
    )


def _train_epoch(
    *,
    model: BaseDyGraphV1TokenModel,
    dataset: Dataset,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    config: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
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
    total_ce = 0.0
    total_correct = 0.0
    total_count = 0.0
    graph_stats = [
        _graph_stats_accumulator() for _ in range(model.config.num_st_blocks)
    ]
    progress = tqdm(
        loader,
        desc=f"train epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        context, future = _batch_inputs(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output, target = _forward_batch(model, context, future)
        sums = _token_batch_sums(
            output.s1_logits,
            target,
            top_k_values=(1,),
        )
        batch_count = float(sums["count_by_step"].sum().item())
        loss = sums["ce_sum_by_step"].sum().float() / batch_count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BaseDyGraph token loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip = float(training["gradient_clip_norm"])
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_ce += float(sums["ce_sum_by_step"].sum().item())
        total_correct += float(
            sums["top1_correct_by_step"].sum().item()
        )
        total_count += batch_count
        for index, graph in enumerate(output.per_layer_graphs):
            _add_graph_stats(
                graph_stats[index],
                graph,
                batch_size=int(context.shape[0]),
            )
        progress.set_postfix(ce=f"{total_ce / max(total_count, 1):.4f}")

    if total_count <= 0:
        raise RuntimeError("Training loader produced no token targets.")
    result: dict[str, Any] = {
        "training_mean_cross_entropy": total_ce / total_count,
        "training_mean_top1_accuracy": total_correct / total_count,
    }
    for index, accumulator in enumerate(graph_stats):
        values = _final_graph_stats(accumulator)
        result[f"training_block_{index}_selected_entropy"] = values["entropy"]
        result[
            f"training_block_{index}_selected_effective_neighbours"
        ] = values["effective_neighbours"]
    return result


def _evaluate(
    *,
    model: BaseDyGraphV1TokenModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    description: str,
    top_k_values: Sequence[int] = TOP_K_VALUES,
) -> dict[str, Any]:
    model.eval()
    length = int(model.config.output_length)
    ce_sum = torch.zeros(length, dtype=torch.float64)
    count = torch.zeros(length, dtype=torch.float64)
    correct = {
        int(k): torch.zeros(length, dtype=torch.float64)
        for k in top_k_values
    }
    graph_stats = [
        _graph_stats_accumulator() for _ in range(model.config.num_st_blocks)
    ]
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        ):
            context, future = _batch_inputs(batch, device=device)
            with _autocast_context(device, use_amp):
                output, target = _forward_batch(model, context, future)
            sums = _token_batch_sums(
                output.s1_logits,
                target,
                top_k_values=top_k_values,
            )
            ce_sum += sums["ce_sum_by_step"].detach().cpu()
            count += sums["count_by_step"].detach().cpu()
            for k in top_k_values:
                correct[int(k)] += sums[
                    f"top{int(k)}_correct_by_step"
                ].detach().cpu()
            for index, graph in enumerate(output.per_layer_graphs):
                _add_graph_stats(
                    graph_stats[index],
                    graph,
                    batch_size=int(context.shape[0]),
                )
    if torch.any(count <= 0):
        raise RuntimeError("Evaluation produced an empty token position.")
    cross_entropy = ce_sum / count
    accuracy = {k: correct[k] / count for k in correct}
    forecast_index = (
        length - 1
        if model.config.prediction_mode == "dense_one_step"
        else 0
    )
    result: dict[str, Any] = {
        "mean_cross_entropy": float(cross_entropy.mean().item()),
        "mean_top1_accuracy": float(accuracy[1].mean().item()),
        "forecast_h1_cross_entropy": float(
            cross_entropy[forecast_index].item()
        ),
        "forecast_h1_top1_accuracy": float(
            accuracy[1][forecast_index].item()
        ),
        "cross_entropy_by_step": cross_entropy,
        **{
            f"top{k}_accuracy_by_step": values
            for k, values in accuracy.items()
        },
    }
    for index, accumulator in enumerate(graph_stats):
        values = _final_graph_stats(accumulator)
        result[f"block_{index}_selected_entropy"] = values["entropy"]
        result[
            f"block_{index}_selected_effective_neighbours"
        ] = values["effective_neighbours"]
    return result


def _history_record(
    *,
    epoch: int,
    learning_rates: Mapping[str, float | None],
    train: Mapping[str, Any],
    selection: Mapping[str, Any],
    model: BaseDyGraphV1TokenModel,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "backbone_learning_rate": learning_rates.get("backbone"),
        "graph_learning_rate": learning_rates.get("graph"),
        **dict(train),
        "selection_score": float(selection["mean_top1_accuracy"]),
        "test_mean_top1_accuracy": float(selection["mean_top1_accuracy"]),
        "test_mean_cross_entropy": float(selection["mean_cross_entropy"]),
        "test_forecast_h1_top1_accuracy": float(
            selection["forecast_h1_top1_accuracy"]
        ),
        "test_forecast_h1_cross_entropy": float(
            selection["forecast_h1_cross_entropy"]
        ),
    }
    prefix = (
        "test_dense_transition"
        if model.config.prediction_mode == "dense_one_step"
        else "test_future"
    )
    ce = torch.as_tensor(selection["cross_entropy_by_step"])
    top1 = torch.as_tensor(selection["top1_accuracy_by_step"])
    for step in range(model.config.output_length):
        position = step + 1
        record[f"{prefix}_cross_entropy_{position}"] = float(ce[step].item())
        record[f"{prefix}_top1_accuracy_{position}"] = float(
            top1[step].item()
        )
    for block in range(model.config.num_st_blocks):
        record[f"block_{block}_alpha"] = None
        record[f"block_{block}_beta"] = None
        record[f"block_{block}_selected_entropy"] = selection.get(
            f"block_{block}_selected_entropy"
        )
        record[
            f"block_{block}_selected_effective_neighbours"
        ] = selection.get(
            f"block_{block}_selected_effective_neighbours"
        )
    return record


def _token_metric_table(
    evaluation: Mapping[str, Any],
    *,
    model: BaseDyGraphV1TokenModel,
) -> pd.DataFrame:
    ce = torch.as_tensor(evaluation["cross_entropy_by_step"])
    rows: list[dict[str, Any]] = []
    for step in range(model.config.output_length):
        if model.config.prediction_mode == "dense_one_step":
            semantic = (
                "forecast_h1"
                if step == model.config.output_length - 1
                else "teacher_forced_context_transition"
            )
            public_horizon = 1 if semantic == "forecast_h1" else None
        else:
            semantic = "future_horizon"
            public_horizon = step + 1
        row: dict[str, Any] = {
            "position": step + 1,
            "semantic": semantic,
            "public_horizon": public_horizon,
            "cross_entropy": float(ce[step].item()),
        }
        for k in TOP_K_VALUES:
            row[f"top{k}_accuracy"] = float(
                torch.as_tensor(evaluation[f"top{k}_accuracy_by_step"])[
                    step
                ].item()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _public_metric_long_table(
    detailed: pd.DataFrame,
    *,
    model: BaseDyGraphV1TokenModel,
) -> pd.DataFrame:
    selected = (
        detailed.loc[detailed["semantic"] == "forecast_h1"]
        if model.config.prediction_mode == "dense_one_step"
        else detailed.loc[
            detailed["public_horizon"].isin(model.config.evaluation_horizons)
        ]
    )
    metric_names = {
        "cross_entropy": "coarse_s1_cross_entropy",
        "top1_accuracy": "coarse_s1_top1_accuracy",
        "top3_accuracy": "coarse_s1_top3_accuracy",
        "top5_accuracy": "coarse_s1_top5_accuracy",
        "top10_accuracy": "coarse_s1_top10_accuracy",
    }
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        horizon = int(row.public_horizon)
        for source, metric in metric_names.items():
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "channel": "s1",
                    "value": float(getattr(row, source)),
                }
            )
    return pd.DataFrame(rows)


def _export_selected_checkpoint(
    *,
    model: BaseDyGraphV1TokenModel,
    loader: DataLoader,
    dataset: CachedTokenGraphDataset,
    split_name: str,
    device: torch.device,
    use_amp: bool,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.eval()
    predicted_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    sample_parts: list[Tensor] = []
    origin_parts: list[Tensor] = []
    target_index_parts: list[Tensor] = []
    dates: list[str] = []
    per_layer_parts: list[list[Tensor]] = [
        [] for _ in range(model.config.num_st_blocks)
    ]
    top10_id_parts: list[Tensor] = []
    top10_probability_parts: list[Tensor] = []
    true_probability_parts: list[Tensor] = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"export BaseDyGraph-v1 {split_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            context, future = _batch_inputs(batch, device=device)
            with _autocast_context(device, use_amp):
                output, target = _forward_batch(model, context, future)
            predicted_parts.append(
                output.selected_s1.detach().cpu().to(torch.int16)
            )
            target_parts.append(target.detach().cpu().to(torch.int16))
            last_parts.append(
                context[:, -1].detach().cpu().to(torch.int16).unsqueeze(-1)
            )
            sample_parts.append(
                torch.as_tensor(
                    batch.get("sample_idx", batch["window_idx"])
                ).cpu().long()
            )
            origin_parts.append(
                torch.as_tensor(batch["origin_idx"]).cpu().long()
            )
            target_index_parts.append(
                torch.as_tensor(batch["target_indices"]).cpu().long()
            )
            batch_dates = batch.get("date")
            if batch_dates is None:
                dates.extend([""] * int(context.shape[0]))
            elif isinstance(batch_dates, str):
                dates.append(batch_dates)
            else:
                dates.extend(str(value) for value in batch_dates)

            public_indices = torch.tensor(
                (
                    [model.config.output_length - 1]
                    if model.config.prediction_mode == "dense_one_step"
                    else model.config.evaluation_indices
                ),
                dtype=torch.long,
                device=device,
            )
            public_logits = output.s1_logits.index_select(
                1,
                public_indices,
            ).float()
            probabilities = torch.softmax(public_logits, dim=-1)
            top_probabilities, top_ids = probabilities.topk(10, dim=-1)
            public_targets = target.index_select(1, public_indices)
            top10_id_parts.append(top_ids.detach().cpu().to(torch.int16))
            top10_probability_parts.append(
                top_probabilities.detach().cpu().to(torch.float16)
            )
            true_probability_parts.append(
                probabilities.gather(-1, public_targets.unsqueeze(-1))
                .squeeze(-1)
                .detach()
                .cpu()
                .to(torch.float16)
            )
            for index, graph in enumerate(output.per_layer_graphs):
                per_layer_parts[index].append(
                    graph.detach().cpu().to(torch.float16).contiguous()
                )

    predicted = torch.cat(predicted_parts, dim=0)
    target = torch.cat(target_parts, dim=0)
    sample_idx = torch.cat(sample_parts, dim=0)
    origin_idx = torch.cat(origin_parts, dim=0)
    raw_target_indices = torch.cat(target_index_parts, dim=0)
    if model.config.prediction_mode == "dense_one_step":
        public_indices_cpu = torch.tensor(
            [model.config.output_length - 1],
            dtype=torch.long,
        )
        public_target_indices = raw_target_indices[:, :1]
        public_horizons = [1]
    else:
        public_indices_cpu = torch.tensor(
            model.config.evaluation_indices,
            dtype=torch.long,
        )
        public_target_indices = raw_target_indices.index_select(
            1,
            public_indices_cpu,
        )
        public_horizons = list(model.config.evaluation_horizons)
    public_predicted = predicted.index_select(1, public_indices_cpu)
    public_target = target.index_select(1, public_indices_cpu)
    last_context = torch.cat(last_parts, dim=0)

    prediction_result = {
        "y_pred": public_predicted.unsqueeze(-1),
        "y_true": public_target.unsqueeze(-1),
        "last_context_target": last_context,
        "channels": ["s1"],
        "horizons": public_horizons,
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": public_target_indices,
        "output_space": "token_id",
    }
    dense_prediction_result = {
        "y_pred": predicted.unsqueeze(-1),
        "y_true": target.unsqueeze(-1),
        "last_context_target": last_context,
        "channels": ["s1"],
        "horizons": list(range(1, model.config.output_length + 1)),
        "reported_horizons": public_horizons,
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": (
            raw_target_indices
            if model.config.prediction_mode == "parallel_60"
            else None
        ),
        "output_space": "token_id",
        "axis_semantics": (
            "teacher_forced_next_token_transition"
            if model.config.prediction_mode == "dense_one_step"
            else "future_horizon"
        ),
    }

    evaluation = _evaluate(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        description=f"token metrics {split_name}",
        top_k_values=TOP_K_VALUES,
    )
    token_metric_table = _token_metric_table(evaluation, model=model)
    metric_table = _public_metric_long_table(
        token_metric_table,
        model=model,
    )

    per_layer = tuple(
        torch.cat(values, dim=0) for values in per_layer_parts
    )
    graph_artifacts = {
        "graph_type": "dynamic",
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(dataset.asset_cols),
        "num_layers": model.config.num_st_blocks,
        "num_heads": model.config.graph_num_heads,
        "num_heads_per_layer": [
            model.config.graph_num_heads
        ] * model.config.num_st_blocks,
        "layer_head_counts": [
            model.config.graph_num_heads
        ] * model.config.num_st_blocks,
        "graph_activations_per_layer": [
            "softmax"
        ] * model.config.num_st_blocks,
        "selected_layer": model.config.num_st_blocks - 1,
        "selected": per_layer[-1],
        "per_layer": per_layer,
        "base": None,
        "per_layer_base": tuple([None] * model.config.num_st_blocks),
        "dynamic": per_layer[-1],
        "per_layer_dynamic": per_layer,
        "alpha": None,
        "alpha_per_layer": tuple([None] * model.config.num_st_blocks),
        "beta": None,
        "beta_per_layer": None,
        "dynamic_alpha": None,
        "spatial_beta": None,
        "spatial_gate_type": "none",
        "beta_trainable": False,
        "dates": dates,
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": public_target_indices,
        "diagonal_policy": (
            "eligible_in_dynamic_softmax; no extra identity matrix"
        ),
    }
    token_artifacts = {
        "predicted_s1": public_predicted,
        "generated_s1": public_predicted,
        "target_s1": public_target,
        "dense_predicted_s1": predicted,
        "dense_target_s1": target,
        "top10_s1_ids_at_reported_horizons": torch.cat(
            top10_id_parts,
            dim=0,
        ),
        "top10_s1_probabilities_at_reported_horizons": torch.cat(
            top10_probability_parts,
            dim=0,
        ),
        "true_s1_probability_at_reported_horizons": torch.cat(
            true_probability_parts,
            dim=0,
        ),
        "evaluation_horizons": public_horizons,
        "prediction_length": model.config.output_length,
        "prediction_mode": model.config.prediction_mode,
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": public_target_indices,
        "dates": dates,
        "asset_cols": list(dataset.asset_cols),
        "token_selection": "argmax",
        "future_token_mode": "coarse_only",
        "input_token_stream": "s1",
        "output_token_stream": "s1",
    }
    blocks = [
        {
            "block": index,
            "activation": "softmax",
            "heads": model.config.graph_num_heads,
            "selected_graph": graph_component_summary(
                per_layer[index].float()
            ),
            "dynamic_graph": graph_component_summary(
                per_layer[index].float()
            ),
            "static_graph": graph_component_summary(None),
        }
        for index in range(model.config.num_st_blocks)
    ]
    final_summary = blocks[-1]["selected_graph"]
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(predicted.shape[0]),
        "prediction_mode": model.config.prediction_mode,
        "selection_semantics": (
            f"mean top-1 across {model.config.output_length} dense "
            "teacher-forced next-token transitions"
            if model.config.prediction_mode == "dense_one_step"
            else f"mean top-1 across {model.config.output_length} future horizons"
        ),
        "mean_selection_top1_accuracy": float(
            evaluation["mean_top1_accuracy"]
        ),
        "mean_cross_entropy": float(evaluation["mean_cross_entropy"]),
        "forecast_h1_top1_accuracy": float(
            evaluation["forecast_h1_top1_accuracy"]
        ),
        "forecast_h1_cross_entropy": float(
            evaluation["forecast_h1_cross_entropy"]
        ),
        "mean_future_top1_accuracy": (
            None
            if model.config.prediction_mode == "dense_one_step"
            else float(evaluation["mean_top1_accuracy"])
        ),
        "final_graph_entropy": final_summary.get("mean_row_entropy"),
        "final_graph_effective_neighbours": final_summary.get(
            "mean_effective_neighbours"
        ),
        "blocks": blocks,
        "graph_orientation": GRAPH_ORIENTATION,
    }
    return {
        "prediction_result": prediction_result,
        "dense_token_prediction_result": dense_prediction_result,
        "graph_artifacts": graph_artifacts,
        "token_artifacts": token_artifacts,
        "metric_table": metric_table,
        "token_metric_table": token_metric_table,
        "diagnostics": diagnostics,
    }


def _build_model(
    *,
    config: dict[str, Any],
    dataset: CachedTokenGraphDataset,
    device: torch.device,
) -> BaseDyGraphV1TokenModel:
    config["models"]["dynamic_graph"]["num_nodes"] = dataset.num_assets
    model_config = basedygraph_v1_token_config_from_mapping(
        config,
        num_nodes=dataset.num_assets,
        vocabulary_size=dataset.s1_vocabulary_size,
    )
    return BaseDyGraphV1TokenModel(model_config).to(device)


def main() -> None:
    args = build_argument_parser().parse_args()
    resolved = _load_config(args.config)
    datasets = _load_datasets(args)
    reference = datasets["train"]
    if int(resolved["data"]["context_length"]) != reference.context_length:
        raise ValueError("Configured context length differs from the cache.")
    if int(resolved["data"]["prediction_length"]) != reference.prediction_length:
        raise ValueError("Configured prediction length differs from the cache.")

    device = resolve_device(args.device)
    training = resolved["training"]
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    set_seed(int(training["seed"]))
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    model = _build_model(
        config=resolved,
        dataset=reference,
        device=device,
    )
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)
    run_signature = _signature(resolved)
    project_root = Path(__file__).resolve().parents[2]
    backbone_parameters, graph_parameters = _parameter_partition(model)
    metadata: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "run_signature": run_signature,
        "model_family": "official_basedygraph_v1_token_comparison",
        "prediction_mode": model.config.prediction_mode,
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": training["selection_metric"],
        "asset_cols": list(reference.asset_cols),
        "context_length": reference.context_length,
        "prediction_length": model.config.output_length,
        "cache_future_length": reference.prediction_length,
        "reported_horizons": list(model.config.public_horizons),
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "temporal_family": "official_basedygraph_v1_transformer",
        "graph_family": "dynamic_only",
        "graph_type": "dynamic",
        "prior_type": "none",
        "num_st_blocks": model.config.num_st_blocks,
        "graph_heads": model.config.graph_num_heads,
        "graph_heads_per_layer": [
            model.config.graph_num_heads
        ] * model.config.num_st_blocks,
        "graph_hidden_dims_per_layer": [
            model.config.graph_hidden_dim
        ] * model.config.num_st_blocks,
        "graph_activations_per_layer": [
            "softmax"
        ] * model.config.num_st_blocks,
        "state_pathway": False,
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
        "basedygraph_expected_commit": PINNED_BASEDYGRAPH_COMMIT,
        "basedygraph_observed_commit": model.external_commit,
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "scheduler_decay_start_epoch": int(
            training["scheduler_decay_start_epoch"]
        ),
        "scheduler_decay_factor": float(training["scheduler_decay_factor"]),
        "mixed_precision": bool(use_amp),
        "device": str(device),
        "train_cache_path": str(args.train_cache),
        "validation_cache_path": str(args.validation_cache),
        "test_cache_path": str(args.test_cache),
        "project_git_commit": _git_value(
            ["rev-parse", "HEAD"],
            cwd=project_root,
        ),
        "project_git_branch": _git_value(
            ["branch", "--show-current"],
            cwd=project_root,
        ),
    }
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        "This curiosity run uses October-December test tokens for checkpoint selection.\n",
        encoding="utf-8",
    )

    start_epoch = 1
    best_score = -float("inf")
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
            raise ValueError("Resume signature differs from the requested run.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        last_epoch = int(checkpoint["epoch"])
        start_epoch = last_epoch + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(
            checkpoint["evaluations_without_improvement"]
        )
        history = list(checkpoint["history"])
        training_complete = bool(checkpoint.get("training_complete", False))
        restore_rng_state(checkpoint["rng_state"])

    selection_loader = _build_loader(
        datasets["test"],
        batch_size=int(training["selection_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    wandb_run = _init_wandb(args, resolved)
    try:
        if not training_complete:
            for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
                last_epoch = epoch
                _set_schedule_for_epoch(
                    optimizer,
                    epoch=epoch,
                    decay_start_epoch=int(
                        training["scheduler_decay_start_epoch"]
                    ),
                    decay_factor=float(training["scheduler_decay_factor"]),
                )
                learning_rates = _learning_rates(optimizer)
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
                selection = _evaluate(
                    model=model,
                    loader=selection_loader,
                    device=device,
                    use_amp=use_amp,
                    description=f"test selection epoch {epoch}",
                    top_k_values=(1,),
                )
                score = float(selection["mean_top1_accuracy"])
                record = _history_record(
                    epoch=epoch,
                    learning_rates=learning_rates,
                    train=train_values,
                    selection=selection,
                    model=model,
                )
                history.append(record)
                atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
                improved = score > best_score + float(training["min_delta"])
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
                print(
                    f"epoch={epoch} "
                    f"train_ce={train_values['training_mean_cross_entropy']:.6f} "
                    f"test_selection_top1={score:.6f} "
                    f"test_h1_top1={selection['forecast_h1_top1_accuracy']:.6f} "
                    f"best={best_score:.6f} best_epoch={best_epoch} "
                    f"backbone_lr={learning_rates['backbone']:.3g} "
                    f"graph_lr={learning_rates['graph']:.3g}"
                )
                if without_improvement >= int(training["patience"]):
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
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        model.to(device)
        best_epoch = int(best_checkpoint["best_epoch"])
        best_score = float(best_checkpoint["best_score"])

        for split_name in ("train", "validation", "test"):
            export_loader = _build_loader(
                datasets[split_name],
                batch_size=int(training["export_batch_size"]),
                shuffle=False,
                num_workers=int(training["num_workers"]),
                seed=int(training["seed"]),
                pin_memory=device.type == "cuda",
            )
            values = _export_selected_checkpoint(
                model=model,
                loader=export_loader,
                dataset=datasets[split_name],
                split_name=split_name,
                device=device,
                use_amp=use_amp,
                checkpoint_epoch=best_epoch,
            )
            _save_export(
                run_dir,
                split_name=split_name,
                values=values,
            )

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "epochs_completed": int(last_epoch),
                "training_complete": True,
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        print("Completed BaseDyGraph-v1 token run:", run_dir)
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "best_epoch": int(best_epoch),
                "best_score": (
                    None if not np.isfinite(best_score) else float(best_score)
                ),
                "epochs_completed": int(last_epoch),
                "training_complete": bool(training_complete),
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

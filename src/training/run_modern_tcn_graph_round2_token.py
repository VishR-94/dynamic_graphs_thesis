from __future__ import annotations

"""Train, export, and optionally decode the 12-model coarse-token Round-2 grid.

This is an intentionally test-selected curiosity experiment.  Gradient updates
use the canonical January-August token cache; checkpoint selection and early
stopping maximise mean top-1 coarse-token accuracy across all 60 future minutes
on the October-December test cache.  September validation is exported only as a
post-selection diagnostic.

The optional ``--decode-sampled`` mode loads one completed model, samples ten
complete 60-token paths, decodes each path independently with the frozen Kronos
coarse decoder, averages in raw price space, and writes price artefacts for
validation and test only.
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
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.models.dynamic_graph.future_predictor import select_token_ids
from src.models.graph_priors import (
    build_absolute_correlation_graph_prior,
    build_sector_graph_prior,
)
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.models.modern_tcn_graph_round2_token import (
    ModernTCNGraphRound2TokenModel,
    token_round2_model_config_from_mapping,
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
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]
GRAPH_ORIENTATION = "A[target, source]"
TOP_K_VALUES = (1, 3, 5, 10)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or decode one coarse-token ModernTCN Round-2 model."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--company-profiles", type=Path, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decode-sampled", action="store_true")
    parser.add_argument("--forecasting-config", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--decode-series-batch-size", type=int, default=64)
    parser.add_argument(
        "--decode-splits", nargs="+", default=("validation", "test")
    )
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online", "offline"), default="disabled"
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
    for key in ("data", "model", "training"):
        if not isinstance(config.get(key), Mapping):
            raise KeyError(f"Config must contain mapping {key!r}.")
    data = config["data"]
    model = config["model"]
    training = config["training"]
    context = int(data["context_length"])
    prediction = int(data["prediction_length"])
    horizons = tuple(int(value) for value in data["evaluation_horizons"])
    if context <= 0 or prediction <= 0:
        raise ValueError("context_length and prediction_length must be positive.")
    if horizons != tuple(sorted(set(horizons))) or not horizons:
        raise ValueError("evaluation_horizons must be unique and increasing.")
    if horizons[0] <= 0 or horizons[-1] > prediction:
        raise ValueError("evaluation_horizons lie outside prediction_length.")
    if str(data["input_token_stream"]) != "s1" or str(
        data["target_token_stream"]
    ) != "s1":
        raise ValueError("This sweep is coarse-s1 input and output only.")
    if int(data["s1_vocabulary_size"]) != 1024:
        raise ValueError("This sweep uses the original 1,024-way Kronos s1 space.")

    if str(model["graph_family"]) not in {"dynamic_only", "prior_state"}:
        raise ValueError("Unsupported graph family.")
    graph = model["graph"]
    heads = tuple(int(value) for value in graph["num_heads_per_block"])
    hidden = tuple(int(value) for value in graph["hidden_dims_per_block"])
    activations = tuple(str(value) for value in graph["activations_per_block"])
    if not heads or len(heads) != len(hidden) or len(heads) != len(activations):
        raise ValueError("Graph schedules must have equal non-zero length.")
    if activations[-1] != "sparsemax" or any(
        value != "softmax" for value in activations[:-1]
    ):
        raise ValueError("Non-final graphs must be softmax and final sparsemax.")
    if bool(graph.get("add_self_loops", False)):
        raise ValueError("This controlled sweep excludes graph self-edges.")
    if str(model["spatial"]["gate_type"]) != "learned_scalar":
        raise ValueError("Every ST block must retain a learned beta gate.")
    predictor = model["future_predictor"]
    if str(predictor["type"]) != "structured_parallel":
        raise ValueError("This sweep uses the structured-parallel predictor.")

    if str(training["selection_split"]) != "test":
        raise ValueError("This curiosity sweep selects on the test cache.")
    if str(training["selection_metric"]) != (
        "mean_top1_accuracy_over_all_future_steps"
    ):
        raise ValueError("Unexpected checkpoint-selection metric.")
    if str(training["selection_direction"]) != "maximise":
        raise ValueError("Token accuracy selection must be maximised.")
    if str(training["optimizer"]).lower() != "adam":
        raise ValueError("The sweep preserves Adam.")
    if str(training["parameter_grouping"]) != "split":
        raise ValueError("The sweep preserves split backbone/graph LRs.")
    if str(training["scheduler"]) != "modern_tcn_type3_delayed":
        raise ValueError("The sweep uses the delayed type-3 schedule.")
    if int(training["scheduler_decay_start_epoch"]) <= 0:
        raise ValueError("scheduler_decay_start_epoch must be positive.")
    if not 0.0 < float(training["scheduler_decay_factor"]) <= 1.0:
        raise ValueError("scheduler_decay_factor must lie in (0,1].")
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


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
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
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _parameter_partition(
    model: ModernTCNGraphRound2TokenModel,
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
        raise AssertionError("Optimizer partition lost parameters.")
    if {id(value) for value in graph} & {id(value) for value in backbone}:
        raise AssertionError("Optimizer parameter groups overlap.")
    return backbone, graph


def _build_optimizer(
    model: ModernTCNGraphRound2TokenModel,
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


def _set_schedule_for_epoch(
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


def _learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float | None]:
    values: dict[str, float | None] = {"backbone": None, "graph": None}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", "backbone" if index == 0 else "graph"))
        if name in values:
            values[name] = float(group["lr"])
    return values


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


def _batch_tokens(
    batch: Mapping[str, Any], *, device: torch.device
) -> tuple[Tensor, Tensor]:
    context = torch.as_tensor(batch["context_tokens"])[..., 0].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    target = torch.as_tensor(batch["target_s1"]).to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return context, target


def _token_batch_sums(
    logits: Tensor,
    target: Tensor,
    *,
    top_k_values: Sequence[int],
) -> dict[str, Tensor]:
    if logits.shape[:-1] != target.shape:
        raise ValueError("Token logits and target shapes do not align.")
    batch, prediction, nodes, vocabulary = map(int, logits.shape)
    losses = F.cross_entropy(
        logits.reshape(-1, vocabulary).float(),
        target.reshape(-1),
        reduction="none",
    ).reshape(batch, prediction, nodes)
    result: dict[str, Tensor] = {
        "ce_sum_by_step": losses.sum(dim=(0, 2)).double(),
        "count_by_step": torch.full(
            (prediction,), batch * nodes, dtype=torch.float64, device=logits.device
        ),
    }
    maximum_k = min(max(int(value) for value in top_k_values), vocabulary)
    indices = logits.topk(maximum_k, dim=-1).indices
    matches = indices.eq(target.unsqueeze(-1))
    for value in top_k_values:
        k = min(int(value), vocabulary)
        result[f"top{int(value)}_correct_by_step"] = (
            matches[..., :k].any(dim=-1).sum(dim=(0, 2)).double()
        )
    return result


def _train_epoch(
    *,
    model: ModernTCNGraphRound2TokenModel,
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
    ce_sum = 0.0
    top1_sum = 0.0
    count = 0.0
    block_count = model.config.num_st_blocks
    selected_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    dynamic_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    static_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    diagnostic_taken = False
    gradient_diagnostics: dict[str, float] = {}

    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False, dynamic_ncols=True)
    for batch in progress:
        context, target = _batch_tokens(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model(context)
        sums = _token_batch_sums(output.s1_logits, target, top_k_values=(1,))
        batch_count = float(sums["count_by_step"].sum().item())
        loss = sums["ce_sum_by_step"].sum().float() / batch_count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite token training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            state_modules = model.block_state_modules()
            for index, block in enumerate(model.graph_spatial_blocks):
                gradient_diagnostics[
                    f"block_{index}_graph_gradient_norm"
                ] = _module_gradient_norm(block.graph_learner)
                gradient_diagnostics[
                    f"block_{index}_alpha_gradient_norm"
                ] = _scalar_gradient(block.graph_learner.raw_alpha)
                gradient_diagnostics[
                    f"block_{index}_beta_gradient_norm"
                ] = _scalar_gradient(block.spatial_gate.raw_beta)
                gradient_diagnostics[
                    f"block_{index}_state_embedding_gradient_norm"
                ] = _module_gradient_norm(state_modules[index])
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        ce_sum += float(sums["ce_sum_by_step"].sum().item())
        top1_sum += float(sums["top1_correct_by_step"].sum().item())
        count += batch_count
        batch_size = int(context.shape[0])
        for index, block in enumerate(output.block_outputs):
            _add_graph_stats(selected_stats[index], block.graph.selected, batch_size=batch_size)
            _add_graph_stats(dynamic_stats[index], block.graph.dynamic, batch_size=batch_size)
            _add_graph_stats(static_stats[index], block.graph.base, batch_size=batch_size)
        progress.set_postfix(ce=f"{ce_sum / max(count, 1):.4f}")

    if count <= 0:
        raise RuntimeError("Training loader produced no tokens.")
    result: dict[str, Any] = {
        "training_mean_cross_entropy": ce_sum / count,
        "training_mean_top1_accuracy": top1_sum / count,
        **gradient_diagnostics,
    }
    for index in range(block_count):
        for component, accumulators in (
            ("selected", selected_stats),
            ("dynamic", dynamic_stats),
            ("static", static_stats),
        ):
            stats = _final_graph_stats(accumulators[index])
            result[f"training_block_{index}_{component}_entropy"] = stats[
                "entropy"
            ]
            result[
                f"training_block_{index}_{component}_effective_neighbours"
            ] = stats["effective_neighbours"]
    return result


def _evaluate_token_split(
    *,
    model: ModernTCNGraphRound2TokenModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    description: str,
    top_k_values: Sequence[int] = TOP_K_VALUES,
) -> dict[str, Any]:
    model.eval()
    prediction = model.config.prediction_length
    ce_sum = torch.zeros(prediction, dtype=torch.float64)
    count = torch.zeros(prediction, dtype=torch.float64)
    correct = {
        int(k): torch.zeros(prediction, dtype=torch.float64)
        for k in top_k_values
    }
    block_count = model.config.num_st_blocks
    selected_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    dynamic_stats = [_graph_stats_accumulator() for _ in range(block_count)]
    static_stats = [_graph_stats_accumulator() for _ in range(block_count)]

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            context, target = _batch_tokens(batch, device=device)
            with _autocast_context(device, use_amp):
                output = model(context)
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
            batch_size = int(context.shape[0])
            for index, block in enumerate(output.block_outputs):
                _add_graph_stats(
                    selected_stats[index], block.graph.selected, batch_size=batch_size
                )
                _add_graph_stats(
                    dynamic_stats[index], block.graph.dynamic, batch_size=batch_size
                )
                _add_graph_stats(
                    static_stats[index], block.graph.base, batch_size=batch_size
                )

    if torch.any(count <= 0):
        raise RuntimeError("Evaluation produced an empty future step.")
    ce = ce_sum / count
    accuracy = {k: correct[k] / count for k in correct}
    result: dict[str, Any] = {
        "mean_cross_entropy": float(ce.mean().item()),
        "mean_top1_accuracy": float(accuracy[1].mean().item()),
        "cross_entropy_by_step": ce,
        **{f"top{k}_accuracy_by_step": values for k, values in accuracy.items()},
    }
    for index in range(block_count):
        result[f"block_{index}_alpha"] = (
            None
            if model.alphas()[index] is None
            else float(model.alphas()[index].detach().cpu().item())
        )
        result[f"block_{index}_beta"] = float(
            model.betas()[index].detach().cpu().item()
        )
        for component, accumulators in (
            ("selected", selected_stats),
            ("dynamic", dynamic_stats),
            ("static", static_stats),
        ):
            stats = _final_graph_stats(accumulators[index])
            result[f"block_{index}_{component}_entropy"] = stats["entropy"]
            result[
                f"block_{index}_{component}_effective_neighbours"
            ] = stats["effective_neighbours"]
    return result


def _history_record(
    *,
    epoch: int,
    learning_rates: Mapping[str, float | None],
    train: Mapping[str, Any],
    selection: Mapping[str, Any],
    model: ModernTCNGraphRound2TokenModel,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "backbone_learning_rate": learning_rates.get("backbone"),
        "graph_learning_rate": learning_rates.get("graph"),
        **dict(train),
        "selection_score": float(selection["mean_top1_accuracy"]),
        "test_mean_top1_accuracy": float(selection["mean_top1_accuracy"]),
        "test_mean_cross_entropy": float(selection["mean_cross_entropy"]),
    }
    ce = torch.as_tensor(selection["cross_entropy_by_step"])
    top1 = torch.as_tensor(selection["top1_accuracy_by_step"])
    for step in range(model.config.prediction_length):
        horizon = step + 1
        record[f"test_cross_entropy_h{horizon}"] = float(ce[step].item())
        record[f"test_top1_accuracy_h{horizon}"] = float(top1[step].item())
    for index in range(model.config.num_st_blocks):
        for name in (
            "alpha",
            "beta",
            "selected_entropy",
            "selected_effective_neighbours",
            "dynamic_entropy",
            "dynamic_effective_neighbours",
            "static_entropy",
            "static_effective_neighbours",
        ):
            record[f"block_{index}_{name}"] = selection.get(
                f"block_{index}_{name}"
            )
    return record


def _token_metric_table(
    evaluation: Mapping[str, Any],
    *,
    reported_horizons: Sequence[int],
) -> pd.DataFrame:
    prediction = len(torch.as_tensor(evaluation["cross_entropy_by_step"]))
    reported = {int(value) for value in reported_horizons}
    rows = []
    for step in range(prediction):
        horizon = step + 1
        row = {
            "future_step": horizon,
            "is_reported_horizon": horizon in reported,
            "cross_entropy": float(
                torch.as_tensor(evaluation["cross_entropy_by_step"])[step].item()
            ),
        }
        for k in TOP_K_VALUES:
            row[f"top{k}_accuracy"] = float(
                torch.as_tensor(evaluation[f"top{k}_accuracy_by_step"])[
                    step
                ].item()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _token_metric_long_table(
    detailed: pd.DataFrame,
) -> pd.DataFrame:
    """Convert dense coarse-token metrics to Graph-Hub's long-table schema."""
    required = {
        "future_step",
        "cross_entropy",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "top10_accuracy",
    }
    missing = required - set(detailed.columns)
    if missing:
        raise KeyError(
            "Detailed token metric table is missing columns "
            f"{sorted(missing)}."
        )
    metric_columns = {
        "cross_entropy": "coarse_s1_cross_entropy",
        "top1_accuracy": "coarse_s1_top1_accuracy",
        "top3_accuracy": "coarse_s1_top3_accuracy",
        "top5_accuracy": "coarse_s1_top5_accuracy",
        "top10_accuracy": "coarse_s1_top10_accuracy",
    }
    rows: list[dict[str, Any]] = []
    for row in detailed.itertuples(index=False):
        horizon = int(row.future_step)
        for source, metric in metric_columns.items():
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "channel": "s1",
                    "value": float(getattr(row, source)),
                }
            )
    return pd.DataFrame(rows)


def _temperature_label(value: float) -> str:
    return f"temperature_{float(value):g}".replace(".", "p").replace("-", "m")


def _export_selected_checkpoint(
    *,
    model: ModernTCNGraphRound2TokenModel,
    loader: DataLoader,
    dataset: CachedTokenGraphDataset,
    split_name: str,
    device: torch.device,
    use_amp: bool,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.eval()
    block_count = model.config.num_st_blocks
    predicted_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    sample_parts: list[Tensor] = []
    origin_parts: list[Tensor] = []
    target_index_parts: list[Tensor] = []
    dates: list[str] = []
    top10_id_parts: list[Tensor] = []
    top10_probability_parts: list[Tensor] = []
    true_probability_parts: list[Tensor] = []
    selected_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    dynamic_lists: list[list[Tensor]] = [[] for _ in range(block_count)]
    singleton_static: list[Tensor | None] = [None] * block_count

    evaluation_indices = torch.tensor(
        model.config.evaluation_indices,
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"export token {split_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            context, target = _batch_tokens(batch, device=device)
            with _autocast_context(device, use_amp):
                output = model(context)
            predicted_parts.append(output.selected_s1.detach().cpu().to(torch.int16))
            target_parts.append(target.detach().cpu().to(torch.int16))
            last_parts.append(context[:, -1].detach().cpu().to(torch.int16).unsqueeze(-1))
            sample_parts.append(torch.as_tensor(batch.get("sample_idx", batch["window_idx"])).cpu().long())
            origin_parts.append(torch.as_tensor(batch["origin_idx"]).cpu().long())
            target_index_parts.append(torch.as_tensor(batch["target_indices"]).cpu().long())
            batch_dates = batch.get("date")
            if batch_dates is None:
                dates.extend([""] * int(context.shape[0]))
            elif isinstance(batch_dates, str):
                dates.append(batch_dates)
            else:
                dates.extend(str(value) for value in batch_dates)

            eval_logits = output.s1_logits.index_select(1, evaluation_indices).float()
            probabilities = torch.softmax(eval_logits, dim=-1)
            top_prob, top_ids = probabilities.topk(10, dim=-1)
            top10_id_parts.append(top_ids.detach().cpu().to(torch.int16))
            top10_probability_parts.append(
                top_prob.detach().cpu().to(torch.float16)
            )
            true_eval = target.index_select(1, evaluation_indices)
            true_probability_parts.append(
                probabilities.gather(-1, true_eval.unsqueeze(-1))
                .squeeze(-1)
                .detach()
                .cpu()
                .to(torch.float16)
            )

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

    predicted = torch.cat(predicted_parts, dim=0)
    target = torch.cat(target_parts, dim=0)
    windows = int(predicted.shape[0])
    dense_horizons = list(range(1, model.config.prediction_length + 1))
    sample_idx = torch.cat(sample_parts, dim=0)
    origin_idx = torch.cat(origin_parts, dim=0)
    dense_target_indices = torch.cat(target_index_parts, dim=0)
    evaluation_indices_cpu = torch.tensor(
        model.config.evaluation_indices,
        dtype=torch.long,
    )
    public_prediction_result = {
        "y_pred": predicted.index_select(1, evaluation_indices_cpu).unsqueeze(-1),
        "y_true": target.index_select(1, evaluation_indices_cpu).unsqueeze(-1),
        "last_context_target": torch.cat(last_parts, dim=0),
        "channels": ["s1"],
        "horizons": list(model.config.evaluation_horizons),
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": dense_target_indices.index_select(
            1, evaluation_indices_cpu
        ),
        "output_space": "token_id",
    }
    dense_token_prediction_result = {
        "y_pred": predicted.unsqueeze(-1),
        "y_true": target.unsqueeze(-1),
        "last_context_target": torch.cat(last_parts, dim=0),
        "channels": ["s1"],
        "horizons": dense_horizons,
        "reported_horizons": list(model.config.evaluation_horizons),
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": dense_target_indices,
        "output_space": "token_id",
    }

    # Re-evaluate metrics from the selected IDs and top-k summaries without
    # saving the enormous complete [W,60,N,1024] logit tensor.
    evaluation = _evaluate_token_split(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        description=f"token metrics {split_name}",
        top_k_values=TOP_K_VALUES,
    )
    token_metric_table = _token_metric_table(
        evaluation,
        reported_horizons=model.config.evaluation_horizons,
    )
    metric_table = _token_metric_long_table(token_metric_table)

    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = (
        per_layer_selected
        if model.config.graph_family == "dynamic_only"
        else tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    )
    per_layer_base = tuple(
        None if values is None else values[0].contiguous()
        for values in singleton_static
    )
    alphas = tuple(
        None if value is None else value.detach().cpu().float().reshape(1)
        for value in model.alphas()
    )
    betas = torch.stack(
        [value.detach().cpu().float().reshape(()) for value in model.betas()]
    )
    graph_artifacts = {
        "graph_type": str(model.config.graph_family),
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(dataset.asset_cols),
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
        "alpha": alphas[-1],
        "alpha_per_layer": alphas,
        "beta": betas[-1:].contiguous(),
        "beta_per_layer": betas,
        "dynamic_alpha": (
            None if alphas[-1] is None else float(alphas[-1].item())
        ),
        "spatial_beta": float(betas[-1].item()),
        "spatial_gate_type": "learned_scalar",
        "beta_trainable": True,
        "dates": dates,
        "sample_idx": public_prediction_result["sample_idx"],
        "origin_idx": public_prediction_result["origin_idx"],
        "target_indices": public_prediction_result["target_indices"],
    }
    token_artifacts = {
        "predicted_s1": predicted,
        "generated_s1": predicted,
        "target_s1": target,
        "top10_s1_ids_at_reported_horizons": torch.cat(
            top10_id_parts, dim=0
        ),
        "top10_s1_probabilities_at_reported_horizons": torch.cat(
            top10_probability_parts, dim=0
        ),
        "true_s1_probability_at_reported_horizons": torch.cat(
            true_probability_parts, dim=0
        ),
        "evaluation_horizons": list(model.config.evaluation_horizons),
        "prediction_length": model.config.prediction_length,
        "sample_idx": public_prediction_result["sample_idx"],
        "origin_idx": public_prediction_result["origin_idx"],
        "target_indices": public_prediction_result["target_indices"],
        "dates": dates,
        "asset_cols": list(dataset.asset_cols),
        "token_selection": "argmax",
        "future_token_mode": "coarse_only",
        "input_token_stream": "s1",
        "output_token_stream": "s1",
    }
    block_diagnostics = []
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
        "windows": windows,
        "prediction_length": model.config.prediction_length,
        "reported_horizons": list(model.config.evaluation_horizons),
        "mean_top1_accuracy_all_60": float(evaluation["mean_top1_accuracy"]),
        "mean_cross_entropy_all_60": float(evaluation["mean_cross_entropy"]),
        "temporal_family": model.config.temporal_family,
        "graph_family": model.config.graph_family,
        "blocks": block_diagnostics,
        "graph_orientation": GRAPH_ORIENTATION,
    }
    return {
        "prediction_result": public_prediction_result,
        "dense_token_prediction_result": dense_token_prediction_result,
        "graph_artifacts": graph_artifacts,
        "token_artifacts": token_artifacts,
        "metric_table": metric_table,
        "token_metric_table": token_metric_table,
        "diagnostics": diagnostics,
    }


def _save_export(
    run_dir: Path,
    *,
    split_name: str,
    values: Mapping[str, Any],
) -> None:
    epoch = int(values["diagnostics"]["checkpoint_epoch"])
    root_prediction = run_dir / f"best_{split_name}_predictions.pt"
    root_graph = run_dir / f"best_{split_name}_graphs.pt"
    root_tokens = run_dir / f"best_{split_name}_tokens.pt"
    root_token_prediction = run_dir / f"best_{split_name}_token_predictions.pt"
    root_metric = run_dir / f"best_{split_name}_metric_table.csv"
    root_token_metric = run_dir / f"best_{split_name}_token_metric_table.csv"
    root_diagnostics = run_dir / f"best_{split_name}_diagnostics.json"
    prediction_payload = {
        "epoch": epoch,
        "prediction_result": values["prediction_result"],
    }
    dense_token_prediction_payload = {
        "epoch": epoch,
        "prediction_result": values["dense_token_prediction_result"],
    }
    token_payload = {
        "epoch": epoch,
        "token_artifacts": values["token_artifacts"],
    }
    atomic_torch_save(prediction_payload, root_prediction)
    atomic_torch_save(dense_token_prediction_payload, root_token_prediction)
    atomic_torch_save(
        {"epoch": epoch, "graph_artifacts": values["graph_artifacts"]},
        root_graph,
    )
    atomic_torch_save(token_payload, root_tokens)
    atomic_csv_save(values["metric_table"], root_metric)
    atomic_csv_save(values["token_metric_table"], root_token_metric)
    atomic_json_save(values["diagnostics"], root_diagnostics)

    analysis = run_dir / "analysis" / split_name
    analysis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_prediction, analysis / "predictions.pt")
    shutil.copy2(root_graph, analysis / "graphs.pt")
    shutil.copy2(root_tokens, analysis / "tokens.pt")
    shutil.copy2(root_metric, analysis / "metric_table.csv")
    shutil.copy2(root_token_metric, analysis / "token_metric_table.csv")
    shutil.copy2(root_diagnostics, analysis / "diagnostics.json")


def _signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_value(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _prepare_run_dir(
    output_dir: Path, run_name: str, *, overwrite: bool, resume: bool
) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        metadata = run_dir / "run_metadata.json"
        checkpoint = run_dir / "best_checkpoint.pt"
        if metadata.is_file() and checkpoint.is_file():
            values = json.loads(metadata.read_text(encoding="utf-8"))
            if values.get("status") == "completed":
                raise FileExistsError(f"Completed run already exists: {run_dir}")
        raise FileExistsError(
            f"Non-empty run directory requires --resume or --overwrite: {run_dir}"
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(without_improvement),
        "history": [dict(value) for value in history],
        "run_signature": str(run_signature),
        "resolved_config": dict(config),
        "run_metadata": dict(metadata),
        "training_complete": bool(training_complete),
        "rng_state": capture_rng_state(),
    }


def _init_wandb(args: argparse.Namespace, config: Mapping[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        tags=list(args.wandb_tags),
        config=dict(config),
    )


def _load_datasets(args: argparse.Namespace) -> dict[str, CachedTokenGraphDataset]:
    datasets = {
        "train": CachedTokenGraphDataset.from_path(args.train_cache),
        "validation": CachedTokenGraphDataset.from_path(args.validation_cache),
        "test": CachedTokenGraphDataset.from_path(args.test_cache),
    }
    reference = datasets["train"]
    for split, dataset in datasets.items():
        if dataset.data_mode != "real":
            raise ValueError(f"{split} cache must be a real-data token cache.")
        if dataset.asset_cols != reference.asset_cols:
            raise ValueError(f"{split} asset order differs from train.")
        if dataset.context_length != reference.context_length:
            raise ValueError(f"{split} context length differs from train.")
        if dataset.prediction_length != reference.prediction_length:
            raise ValueError(f"{split} prediction length differs from train.")
        if dataset.s1_id_space != "kronos_original":
            raise ValueError(f"{split} cache is not in original Kronos s1 space.")
        if dataset.s1_vocabulary_size != 1024:
            raise ValueError(f"{split} cache is not 1,024-way s1.")
    return datasets


def _load_raw_splits(args: argparse.Namespace):
    raw_train, raw_validation, raw_test = load_candle_splits(args.data_dir)
    return clean_candle_splits(raw_train, raw_validation, raw_test)


def _source_prior(
    *,
    config: Mapping[str, Any],
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    company_profiles: Path | None,
) -> tuple[Tensor | None, list[str] | None]:
    graph_family = str(config["model"]["graph_family"])
    prior_type = str(config["model"]["prior"]["type"])
    if graph_family == "dynamic_only":
        return None, None
    if prior_type == "correlation":
        return (
            build_absolute_correlation_graph_prior(
                train_split,
                expected_asset_cols=asset_cols,
                threshold=None,
            ),
            None,
        )
    if prior_type == "sector":
        if company_profiles is None:
            raise ValueError("Sector prior requires --company-profiles.")
        result = build_sector_graph_prior(
            company_profiles,
            expected_asset_cols=asset_cols,
        )
        if isinstance(result, tuple):
            return result[0], list(result[1])
        return result, None
    if prior_type == "none":
        return None, None
    raise ValueError(f"Unsupported prior type {prior_type!r}.")


def _save_initial_prior(
    run_dir: Path,
    *,
    source_prior: Tensor | None,
    prior_type: str,
    asset_cols: Sequence[str],
    sectors: Sequence[str] | None,
) -> None:
    payload = {
        "prior_type": str(prior_type),
        "adjacency": (
            None
            if source_prior is None
            else torch.as_tensor(source_prior).detach().cpu().float()
        ),
        "asset_cols": list(asset_cols),
        "sectors": None if sectors is None else list(sectors),
        "graph_orientation": GRAPH_ORIENTATION,
    }
    atomic_torch_save(payload, run_dir / "initial_graph_prior.pt")


def _build_model(
    *,
    resolved: Mapping[str, Any],
    dataset: CachedTokenGraphDataset,
    source_prior: Tensor | None,
    device: torch.device,
) -> ModernTCNGraphRound2TokenModel:
    # The Graph-Hub mirror is completed only after the cache has established N.
    resolved["models"]["dynamic_graph"]["num_nodes"] = dataset.num_assets
    model_config = token_round2_model_config_from_mapping(
        dict(resolved),
        num_nodes=dataset.num_assets,
        vocabulary_size=dataset.s1_vocabulary_size,
    )
    return ModernTCNGraphRound2TokenModel(
        model_config,
        static_prior=(source_prior if model_config.uses_static_graph else None),
    ).to(device)


def _invalid_candle_mask(decoded_ohlcv: Tensor) -> Tensor:
    values = torch.as_tensor(decoded_ohlcv)
    open_values, high, low, close, volume = [values[..., index] for index in range(5)]
    return (
        ~torch.isfinite(values).all(dim=-1)
        | (open_values <= 0)
        | (high <= 0)
        | (low <= 0)
        | (close <= 0)
        | (high < torch.maximum(open_values, close))
        | (low > torch.minimum(open_values, close))
        | (high < low)
        | (volume < 0)
    )


def _decode_sampled_split(
    *,
    model: ModernTCNGraphRound2TokenModel,
    dataset: CachedTokenGraphDataset,
    split_name: str,
    device: torch.device,
    use_amp: bool,
    tokenizer: KronosTokenizerAdapter,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    sampling_seed: int,
    batch_size: int,
    num_workers: int,
    decode_series_batch_size: int,
    train_split: Mapping[str, Any],
    checkpoint_epoch: int,
    run_dir: Path,
) -> None:
    if sample_count <= 0 or temperature <= 0:
        raise ValueError("sample_count and temperature must be positive.")
    if top_k < 0 or not 0.0 < top_p <= 1.0:
        raise ValueError("Invalid top-k/top-p sampling controls.")
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=sampling_seed,
        pin_memory=device.type == "cuda",
    )
    set_seed(int(sampling_seed))
    model.eval()
    windows = len(dataset)
    prediction = model.config.prediction_length
    nodes = model.config.num_nodes
    sampled_tokens = torch.empty(
        sample_count, windows, prediction, nodes, dtype=torch.int16
    )
    sampled_close = torch.empty(
        sample_count, windows, prediction, nodes, 1, dtype=torch.float32
    )
    cursor = 0
    invalid_sample_count = 0
    invalid_sample_total = 0
    invalid_ensemble_count = 0
    invalid_ensemble_total = 0

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"decode {split_name} ({sample_count} paths)",
            leave=False,
            dynamic_ncols=True,
        ):
            context, _ = _batch_tokens(batch, device=device)
            with _autocast_context(device, use_amp):
                output = model(context)
            samples = torch.stack(
                [
                    select_token_ids(
                        output.s1_logits,
                        mode="sample",
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    for _ in range(sample_count)
                ],
                dim=0,
            ).detach().cpu().long()
            current = int(context.shape[0])
            start, stop = cursor, cursor + current
            sampled_tokens[:, start:stop] = samples.to(torch.int16)

            context_pairs = torch.as_tensor(batch["context_tokens"]).cpu().long()
            means = torch.as_tensor(batch["context_mean"]).cpu().float()
            stds = torch.as_tensor(batch["context_std"]).cpu().float()
            expanded_context = (
                context_pairs.unsqueeze(0)
                .expand(sample_count, -1, -1, -1, -1)
                .reshape(sample_count * current, model.config.context_length, nodes, 2)
            )
            expanded_future = samples.reshape(
                sample_count * current, prediction, nodes
            )
            expanded_mean = (
                means.unsqueeze(0)
                .expand(sample_count, -1, -1, -1)
                .reshape(sample_count * current, nodes, 6)
            )
            expanded_std = (
                stds.unsqueeze(0)
                .expand(sample_count, -1, -1, -1)
                .reshape(sample_count * current, nodes, 6)
            )
            decoded = tokenizer.decode_coarse_token_path(
                expanded_context,
                expanded_future,
                mean=expanded_mean,
                std=expanded_std,
                series_batch_size=decode_series_batch_size,
                return_full_path=False,
            ).reshape(sample_count, current, prediction, nodes, 5)
            sampled_close[:, start:stop] = decoded[..., 3:4]
            invalid = _invalid_candle_mask(decoded)
            invalid_sample_count += int(invalid.sum().item())
            invalid_sample_total += int(invalid.numel())
            ensemble = decoded.float().mean(dim=0)
            ensemble_invalid = _invalid_candle_mask(ensemble)
            invalid_ensemble_count += int(ensemble_invalid.sum().item())
            invalid_ensemble_total += int(ensemble_invalid.numel())
            cursor = stop
    if cursor != windows:
        raise RuntimeError(f"Decoded {cursor} windows; expected {windows}.")
    if not torch.isfinite(sampled_close).all():
        raise ValueError("Decoded Close paths contain non-finite values.")

    evaluation_indices = torch.tensor(model.config.evaluation_indices, dtype=torch.long)
    ensemble_dense = sampled_close.mean(dim=0)
    y_pred = ensemble_dense.index_select(1, evaluation_indices)
    y_true = torch.as_tensor(dataset.cache["evaluation_true"])[..., 3:4].float()
    last = torch.as_tensor(dataset.cache["last_context_target"])[..., 3:4].float()
    target_indices_dense = torch.as_tensor(dataset.cache["target_indices"]).long()
    target_indices_eval = target_indices_dense.index_select(1, evaluation_indices)
    prediction_result = {
        "y_pred": y_pred,
        "y_true": y_true,
        "last_context_target": last,
        "sample_idx": torch.as_tensor(dataset.cache["sample_idx"]).long(),
        "origin_idx": torch.as_tensor(dataset.cache["origin_idx"]).long(),
        "target_indices": target_indices_eval,
        "channels": ["close"],
        "horizons": list(model.config.evaluation_horizons),
        "asset_cols": list(dataset.asset_cols),
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
    sampled_artifacts = {
        "sampled_s1_paths": sampled_tokens,
        "sampled_close_paths": sampled_close,
        "sampled_close_paths_at_evaluation_horizons": sampled_close.index_select(
            2, evaluation_indices
        ),
        "ensemble_mean_close_path": ensemble_dense,
        "evaluation_true": y_true,
        "last_context_target": last,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "dense_target_indices": target_indices_dense,
        "evaluation_target_indices": target_indices_eval,
        "dates": list(dataset.cache.get("dates", [])),
        "asset_cols": list(dataset.asset_cols),
        "future_steps": list(range(1, prediction + 1)),
        "evaluation_horizons": list(model.config.evaluation_horizons),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "sample_count": int(sample_count),
        "sampling_seed": int(sampling_seed),
        "averaging_space": "decoded raw continuous Close",
        "graph_shared_across_sample_paths": True,
    }
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "sample_count": int(sample_count),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "sample_path_invalid_candle_rate_percent": (
            100.0 * invalid_sample_count / max(invalid_sample_total, 1)
        ),
        "ensemble_invalid_candle_rate_percent": (
            100.0 * invalid_ensemble_count / max(invalid_ensemble_total, 1)
        ),
    }

    policy = _temperature_label(temperature)
    policy_root = run_dir / "temperature_sweep" / policy
    analysis_root = run_dir / "analysis" / split_name / policy
    policy_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    root_prediction = policy_root / f"{split_name}_predictions.pt"
    root_paths = policy_root / f"{split_name}_sampled_price_paths.pt"
    root_metrics = policy_root / f"{split_name}_metric_table.csv"
    root_diag = policy_root / f"{split_name}_diagnostics.json"
    root_tokens = policy_root / f"{split_name}_tokens.pt"
    atomic_torch_save(
        {"epoch": checkpoint_epoch, "prediction_result": prediction_result},
        root_prediction,
    )
    atomic_torch_save(
        {"sampled_price_path_artifacts": sampled_artifacts}, root_paths
    )
    atomic_torch_save(
        {
            "epoch": checkpoint_epoch,
            "token_artifacts": {
                "sampled_s1_paths": sampled_tokens,
                "sampled_s1_evaluation": sampled_tokens.index_select(
                    2, evaluation_indices
                ),
                "target_s1": torch.as_tensor(dataset.cache["target_s1"]).to(torch.int16),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "sample_count": int(sample_count),
            },
        },
        root_tokens,
    )
    atomic_csv_save(metric_table, root_metrics)
    atomic_json_save(diagnostics, root_diag)
    graph_source = run_dir / f"best_{split_name}_graphs.pt"
    if not graph_source.is_file():
        raise FileNotFoundError(graph_source)
    shutil.copy2(root_prediction, analysis_root / "predictions.pt")
    shutil.copy2(graph_source, analysis_root / "graphs.pt")
    shutil.copy2(root_paths, analysis_root / "sampled_price_paths.pt")
    shutil.copy2(root_metrics, analysis_root / "metric_table.csv")
    shutil.copy2(root_diag, analysis_root / "diagnostics.json")
    shutil.copy2(root_tokens, analysis_root / "tokens.pt")
    print(f"Decoded {split_name} price metrics saved to {policy_root}")


def _run_decode_mode(
    args: argparse.Namespace,
    resolved: ConfigDict,
    datasets: Mapping[str, CachedTokenGraphDataset],
    train_split: Mapping[str, Any],
    source_prior: Tensor | None,
    device: torch.device,
) -> None:
    if args.forecasting_config is None:
        raise ValueError("--forecasting-config is required for --decode-sampled.")
    run_dir = args.output_dir.expanduser().resolve() / args.run_name
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Completed run metadata/checkpoint is missing.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("Only a completed model can be decoded.")
    model = _build_model(
        resolved=resolved,
        dataset=datasets["train"],
        source_prior=source_prior,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    tokenizer = KronosTokenizerAdapter.from_config(
        load_yaml(args.forecasting_config),
        series_batch_size=args.decode_series_batch_size,
    ).load()
    use_amp = bool(resolved["training"]["mixed_precision"]) and device.type == "cuda"
    allowed = {"validation", "test"}
    requested = tuple(str(value) for value in args.decode_splits)
    if not requested or any(value not in allowed for value in requested):
        raise ValueError("decode_splits must contain validation and/or test.")
    for split_name in requested:
        _decode_sampled_split(
            model=model,
            dataset=datasets[split_name],
            split_name=split_name,
            device=device,
            use_amp=use_amp,
            tokenizer=tokenizer,
            sample_count=args.sample_count,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sampling_seed=args.sampling_seed,
            batch_size=int(resolved["training"]["export_batch_size"]),
            num_workers=int(resolved["training"]["num_workers"]),
            decode_series_batch_size=args.decode_series_batch_size,
            train_split=train_split,
            checkpoint_epoch=int(metadata["best_epoch"]),
            run_dir=run_dir,
        )
    selection = {
        "selected_policy": _temperature_label(args.temperature),
        "selected_temperature": float(args.temperature),
        "sample_count": int(args.sample_count),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "sampling_seed": int(args.sampling_seed),
        "decoded_splits": list(requested),
    }
    atomic_json_save(selection, run_dir / "temperature_sweep" / "temperature_selection.json")


def main() -> None:
    args = build_argument_parser().parse_args()
    resolved = _load_config(args.config)
    datasets = _load_datasets(args)
    data = resolved["data"]
    reference = datasets["train"]
    if reference.context_length != int(data["context_length"]):
        raise ValueError("Configured context length differs from token cache.")
    if reference.prediction_length != int(data["prediction_length"]):
        raise ValueError("Configured prediction length differs from token cache.")
    if tuple(reference.evaluation_horizons) != tuple(
        int(value) for value in data["evaluation_horizons"]
    ):
        raise ValueError("Configured reported horizons differ from token cache.")

    train_split, validation_split, test_split = _load_raw_splits(args)
    del validation_split, test_split
    if list(reference.asset_cols) != list(train_split["asset_cols"]):
        raise ValueError("Raw training asset order differs from token cache.")
    source_prior, sectors = _source_prior(
        config=resolved,
        train_split=train_split,
        asset_cols=reference.asset_cols,
        company_profiles=args.company_profiles,
    )
    device = resolve_device(args.device)
    set_seed(int(resolved["training"]["seed"]))

    if args.decode_sampled:
        _run_decode_mode(
            args,
            resolved,
            datasets,
            train_split,
            source_prior,
            device,
        )
        return

    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    resolved["models"]["dynamic_graph"]["num_nodes"] = reference.num_assets
    run_signature = _signature(resolved)
    model = _build_model(
        resolved=resolved,
        dataset=reference,
        source_prior=source_prior,
        device=device,
    )
    optimizer = _build_optimizer(model, resolved)
    training = resolved["training"]
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = _new_grad_scaler(use_amp)
    project_root = Path(__file__).resolve().parents[2]
    backbone, graph = _parameter_partition(model)
    metadata: dict[str, Any] = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "run_signature": run_signature,
        "model_family": "modern_tcn_graph_round2_token",
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": training["selection_metric"],
        "asset_cols": list(reference.asset_cols),
        "context_length": reference.context_length,
        "prediction_length": reference.prediction_length,
        "reported_horizons": list(reference.evaluation_horizons),
        "all_selection_horizons": list(range(1, reference.prediction_length + 1)),
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "temporal_family": model.config.temporal_family,
        "graph_family": model.config.graph_family,
        "prior_type": model.config.prior_type,
        "graph_heads_per_layer": list(model.config.graph_heads_per_block),
        "graph_hidden_dims_per_layer": list(
            model.config.graph_hidden_dims_per_block
        ),
        "graph_activations_per_layer": list(
            model.config.graph_activations_per_block
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "num_st_blocks": int(model.config.num_st_blocks),
        "state_pathway": bool(model.config.uses_state_pathway),
        "backbone_trainable_parameters": int(
            sum(parameter.numel() for parameter in backbone)
        ),
        "graph_trainable_parameters": int(sum(parameter.numel() for parameter in graph)),
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "scheduler_decay_start_epoch": int(
            training["scheduler_decay_start_epoch"]
        ),
        "scheduler_decay_factor": float(training["scheduler_decay_factor"]),
        "mixed_precision": bool(use_amp),
        "device": str(device),
        "project_git_commit": _git_value(["rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": _git_value(["branch", "--show-current"], cwd=project_root),
    }
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        "This curiosity run uses the test split for architecture and checkpoint selection.\n",
        encoding="utf-8",
    )
    _save_initial_prior(
        run_dir,
        source_prior=source_prior,
        prior_type=model.config.prior_type,
        asset_cols=reference.asset_cols,
        sectors=sectors,
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
        seed=int(training["seed"]),
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
                _set_schedule_for_epoch(
                    optimizer,
                    epoch=epoch,
                    decay_start_epoch=int(training["scheduler_decay_start_epoch"]),
                    decay_factor=float(training["scheduler_decay_factor"]),
                )
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
                selection = _evaluate_token_split(
                    model=model,
                    loader=selection_loader,
                    device=device,
                    use_amp=use_amp,
                    description=f"test token selection epoch {epoch}",
                    top_k_values=(1,),
                )
                score = float(selection["mean_top1_accuracy"])
                record = _history_record(
                    epoch=epoch,
                    learning_rates=current_lrs,
                    train=train_values,
                    selection=selection,
                    model=model,
                )
                history.append(record)
                atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
                improved = score > best_score + min_delta
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
                final_index = model.config.num_st_blocks - 1
                print(
                    f"epoch={epoch} train_ce={train_values['training_mean_cross_entropy']:.6f} "
                    f"test_mean_top1={score:.6f} best={best_score:.6f} "
                    f"best_epoch={best_epoch} "
                    f"alpha={selection.get(f'block_{final_index}_alpha')} "
                    f"beta={selection[f'block_{final_index}_beta']:.4f} "
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
            run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
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
                seed=int(training["seed"]) + 1000 + split_index,
                pin_memory=device.type == "cuda",
            )
            exported = _export_selected_checkpoint(
                model=model,
                loader=loader,
                dataset=datasets[split_name],
                split_name=split_name,
                device=device,
                use_amp=use_amp,
                checkpoint_epoch=best_epoch,
            )
            _save_export(run_dir, split_name=split_name, values=exported)

        final_alphas = [
            None if value is None else float(value.detach().cpu().item())
            for value in model.alphas()
        ]
        final_betas = [float(value.detach().cpu().item()) for value in model.betas()]
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
        print("TOKEN ROUND-2 RUN COMPLETE")
        print("Run:", args.run_name)
        print("Best epoch:", best_epoch)
        print("Best test mean top-1 accuracy over all 60 steps:", best_score)
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
            }
        )
        atomic_json_save(failed, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

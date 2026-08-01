from __future__ import annotations

"""Train the pinned official BaseDyGraph model on one-minute Kronos s1 targets.

The trainable network is imported directly from ``external/BaseDyGraph``. The
only task adaptation is supplied by ``OfficialBaseDyGraphOneStep``:

    context s1 [B, 60, N]
    -> official BaseDyGraph forward
    -> final context-to-next-state logits [B, N, 1024]

No future token is passed to the trainable model. Validation decodes the one
predicted native Kronos coarse token through the existing frozen coarse decoder
and evaluates the ordinary raw-price ForecastEvaluator metrics at horizon 1.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import (
    CachedTokenGraphDataset,
    build_token_graph_dataloaders,
)
from src.evaluation.metrics import ForecastEvaluator
from src.models.basedygraph_official_adapter import (
    OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
    PINNED_BASEDYGRAPH_COMMIT,
    OfficialBaseDyGraphOneStep,
    OfficialBaseDyGraphOneStepOutput,
    OfficialBaseDyGraphRunConfig,
    assert_official_one_step_parity,
)
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.training.run_dynamic_graph import (
    _autocast_context,
    _load_raw_training_split,
    _new_grad_scaler,
    _move_optimizer_state,
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


CLOSE_CHANNEL_INDEX = 3
TOKEN_VOCABULARY_SIZE = 1024


@dataclass(frozen=True)
class EpochClassificationMetrics:
    loss: float
    accuracy: float
    examples: int
    token_targets: int
    graph_mean_row_entropy: float | None
    graph_mean_effective_neighbours: float | None
    graph_mean_diagonal_weight: float | None
    seconds: float


@dataclass(frozen=True)
class ValidationArtifacts:
    classification: EpochClassificationMetrics
    prediction_result: dict[str, Any]
    metric_results: dict[str, Tensor]
    metric_table: pd.DataFrame
    primary_score: float
    token_artifacts: dict[str, Any]
    graph_artifacts: dict[str, Any]
    diagnostics: dict[str, Any]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the pinned official BaseDyGraph implementation on one-step s1 prediction."
    )

    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forecasting-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-series-batch-size", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-validation-windows", type=int, default=None)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=1)
    parser.add_argument("--spatial-layers", type=int, default=1)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--graph-heads", type=int, default=2)
    parser.add_argument("--graph-hidden-dim", type=int, default=64)
    parser.add_argument("--num-st-blocks", type=int, default=1)
    parser.add_argument(
        "--spatial-module-type",
        choices=("none", "static_graph", "dynamic_graph", "dynamic_base"),
        default="static_graph",
    )
    parser.add_argument(
        "--first-spatial-module-type",
        choices=("same", "none", "static_graph", "dynamic_graph", "dynamic_base"),
        default="same",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--spatial-dropout", type=float, default=0.0)
    parser.add_argument("--spatial-value", choices=("hidden", "state_embedding", "concat"), default="hidden")
    parser.add_argument("--graph-activation", choices=("softmax", "sparsemax", "entmax15", "gated"), default="softmax")
    parser.add_argument("--use-node-embedding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-state-pair-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-self-loops", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--symmetric-graph", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", type=str, default="dynamic-graph-financial-forecasting")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=())

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_positive(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _limit_dataset(dataset: CachedTokenGraphDataset, limit: int | None) -> Dataset[dict[str, Any]]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("Window limits must be positive.")
    return Subset(dataset, range(int(limit)))




def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    """Build a deterministic loader for a full dataset or ``Subset`` view."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "generator": generator,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def _signature(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(arguments: Sequence[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _prepare_run_dir(output_dir: Path, run_name: str, overwrite: bool, resume: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") == "completed":
                raise FileExistsError(f"Run is already completed: {run_dir}")
        raise FileExistsError(
            f"Run directory is non-empty: {run_dir}. Use --resume or --overwrite."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _validate_dataset(dataset: CachedTokenGraphDataset, *, name: str) -> None:
    if dataset.data_mode != "real":
        raise ValueError(f"{name} cache must be a real-data cache.")
    if dataset.s1_id_space != "kronos_original":
        raise ValueError(
            f"{name} cache must use original Kronos s1 IDs, not {dataset.s1_id_space!r}."
        )
    if dataset.s1_vocabulary_size != TOKEN_VOCABULARY_SIZE:
        raise ValueError(f"{name} cache must have a 1,024-class s1 vocabulary.")
    if dataset.context_length != 60:
        raise ValueError(f"{name} cache must contain 60 context positions.")
    if dataset.prediction_length < 1:
        raise ValueError(f"{name} cache has no future target.")
    if not dataset.has_raw_evaluation_targets:
        raise ValueError(f"{name} cache lacks raw evaluation targets.")
    if 1 not in dataset.evaluation_horizons:
        raise ValueError(f"{name} cache does not expose horizon 1.")


def _batch_inputs(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    context_pairs = torch.as_tensor(batch["context_tokens"])
    context_s1 = context_pairs[..., 0].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    target_s1 = torch.as_tensor(batch["target_s1"])[:, 0].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return context_s1, target_s1


def _graph_batch_metrics(graph: Tensor | None) -> tuple[float, float, float, int]:
    if graph is None:
        return 0.0, 0.0, 0.0, 0
    values = graph.detach().float().clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    rows = int(entropy.numel())
    return (
        float(entropy.sum().item()),
        float(entropy.exp().sum().item()),
        float(diagonal.sum().item()),
        rows,
    )


def _run_classification_epoch(
    *,
    model: OfficialBaseDyGraphOneStep,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    use_amp: bool,
    gradient_clip_norm: float,
    description: str,
) -> EpochClassificationMetrics:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    total_examples = 0
    entropy_sum = 0.0
    effective_sum = 0.0
    diagonal_sum = 0.0
    graph_rows = 0
    start = perf_counter()

    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    context_manager = torch.enable_grad if training else torch.inference_mode
    with context_manager():
        for batch in progress:
            context_s1, target_s1 = _batch_inputs(batch, device)
            with _autocast_context(device, use_amp):
                output = model(context_s1)
                loss = F.cross_entropy(
                    output.s1_logits.float().reshape(-1, model.num_states),
                    target_s1.reshape(-1),
                )

            if training:
                scaler.scale(loss).backward()
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            target_count = int(target_s1.numel())
            total_loss += float(loss.detach().item()) * target_count
            total_correct += int((output.predicted_s1 == target_s1).sum().item())
            total_targets += target_count
            total_examples += int(context_s1.shape[0])
            ent, eff, diag, rows = _graph_batch_metrics(output.selected_graph)
            entropy_sum += ent
            effective_sum += eff
            diagonal_sum += diag
            graph_rows += rows

            progress.set_postfix(
                loss=f"{total_loss / max(total_targets, 1):.4f}",
                acc=f"{total_correct / max(total_targets, 1):.4f}",
            )

    if total_targets == 0:
        raise RuntimeError("DataLoader yielded no token targets.")

    return EpochClassificationMetrics(
        loss=total_loss / total_targets,
        accuracy=total_correct / total_targets,
        examples=total_examples,
        token_targets=total_targets,
        graph_mean_row_entropy=(entropy_sum / graph_rows if graph_rows else None),
        graph_mean_effective_neighbours=(effective_sum / graph_rows if graph_rows else None),
        graph_mean_diagonal_weight=(diagonal_sum / graph_rows if graph_rows else None),
        seconds=perf_counter() - start,
    )


def _invalid_candle_mask(decoded: Tensor) -> Tensor:
    open_values = decoded[..., 0]
    high_values = decoded[..., 1]
    low_values = decoded[..., 2]
    close_values = decoded[..., 3]
    volume_values = decoded[..., 4]
    return (
        ~torch.isfinite(decoded).all(dim=-1)
        | (open_values <= 0)
        | (high_values <= 0)
        | (low_values <= 0)
        | (close_values <= 0)
        | (high_values < torch.maximum(open_values, close_values))
        | (low_values > torch.minimum(open_values, close_values))
        | (high_values < low_values)
        | (volume_values < 0)
    )


def _graph_summary(graph_artifacts: Mapping[str, Any]) -> dict[str, Any]:
    selected = graph_artifacts.get("selected")
    if selected is None:
        return {"graph_present": False}
    graph = torch.as_tensor(selected).float().clamp_min(1.0e-12)
    entropy = -(graph * graph.log()).sum(dim=-1)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
    return {
        "graph_present": True,
        "mean_row_entropy": float(entropy.mean().item()),
        "mean_effective_neighbours": float(entropy.exp().mean().item()),
        "mean_diagonal_weight": float(diagonal.mean().item()),
        "maximum_edge_weight": float(graph.max().item()),
    }


def _validation_artifacts(
    *,
    model: OfficialBaseDyGraphOneStep,
    loader: DataLoader[dict[str, Any]],
    dataset: CachedTokenGraphDataset,
    device: torch.device,
    use_amp: bool,
    tokenizer: KronosTokenizerAdapter,
    raw_train_split: Mapping[str, Any],
    decode_series_batch_size: int,
) -> ValidationArtifacts:
    model.eval()
    synchronise_device(device)
    start = perf_counter()

    total_loss = 0.0
    total_correct = 0
    total_targets = 0
    total_examples = 0
    entropy_sum = 0.0
    effective_sum = 0.0
    diagonal_sum = 0.0
    graph_rows = 0

    y_pred_parts: list[Tensor] = []
    y_true_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    sample_parts: list[Tensor] = []
    origin_parts: list[Tensor] = []
    target_index_parts: list[Tensor] = []
    generated_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    window_parts: list[Tensor] = []
    dates: list[str] = []
    selected_graph_parts: list[Tensor] = []
    per_layer_parts: list[list[Tensor]] = [list() for _ in range(model.num_st_blocks)]
    invalid_count = 0
    invalid_total = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="validation + one-step decode", leave=False, dynamic_ncols=True):
            context_s1, target_s1 = _batch_inputs(batch, device)
            with _autocast_context(device, use_amp):
                output: OfficialBaseDyGraphOneStepOutput = model(context_s1)
                loss = F.cross_entropy(
                    output.s1_logits.float().reshape(-1, model.num_states),
                    target_s1.reshape(-1),
                )

            generated = output.predicted_s1.detach().cpu().long()
            target_cpu = target_s1.detach().cpu().long()
            count = int(target_cpu.numel())
            total_loss += float(loss.item()) * count
            total_correct += int((generated == target_cpu).sum().item())
            total_targets += count
            total_examples += int(generated.shape[0])

            ent, eff, diag, rows = _graph_batch_metrics(output.selected_graph)
            entropy_sum += ent
            effective_sum += eff
            diagonal_sum += diag
            graph_rows += rows

            context_pairs = torch.as_tensor(batch["context_tokens"]).long()
            decoded = tokenizer.decode_coarse_token_path(
                context_pairs,
                generated.unsqueeze(1),
                mean=torch.as_tensor(batch["context_mean"]),
                std=torch.as_tensor(batch["context_std"]),
                series_batch_size=decode_series_batch_size,
                return_full_path=False,
            ).float()
            if tuple(decoded.shape[1:]) != (1, dataset.num_assets, 5):
                raise RuntimeError(f"Unexpected one-step decoded shape: {tuple(decoded.shape)}")

            invalid = _invalid_candle_mask(decoded)
            invalid_count += int(invalid.sum().item())
            invalid_total += int(invalid.numel())

            horizon_one_index = dataset.evaluation_horizons.index(1)
            y_pred_parts.append(
                decoded[..., CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1]
            )
            y_true_parts.append(
                torch.as_tensor(batch["evaluation_true"]).float()[
                    :,
                    horizon_one_index : horizon_one_index + 1,
                    :,
                    CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
                ]
            )
            last_parts.append(
                torch.as_tensor(batch["last_context_target"]).float()[
                    ...,
                    CLOSE_CHANNEL_INDEX:CLOSE_CHANNEL_INDEX + 1,
                ]
            )
            sample_parts.append(torch.as_tensor(batch["sample_idx"]).long().cpu())
            origin_parts.append(torch.as_tensor(batch["origin_idx"]).long().cpu())
            # target_indices is the dense 1..P future path; column zero is
            # therefore the first unseen minute independently of the sparse
            # evaluation-horizon ordering.
            target_index_parts.append(
                torch.as_tensor(batch["target_indices"]).long().cpu()[:, 0:1]
            )
            window_parts.append(torch.as_tensor(batch["window_idx"]).long().cpu())
            generated_parts.append(generated.to(torch.int16).unsqueeze(1))
            target_parts.append(target_cpu.to(torch.int16).unsqueeze(1))
            if "date" in batch:
                dates.extend(str(value) for value in batch["date"])

            if output.selected_graph is not None:
                selected_graph_parts.append(output.selected_graph.detach().cpu().float())
            for layer_index, layer_graph in enumerate(output.per_layer_graphs):
                if layer_graph is not None:
                    per_layer_parts[layer_index].append(layer_graph.detach().cpu().float())

    synchronise_device(device)
    if total_targets == 0:
        raise RuntimeError("Validation DataLoader yielded no targets.")

    classification = EpochClassificationMetrics(
        loss=total_loss / total_targets,
        accuracy=total_correct / total_targets,
        examples=total_examples,
        token_targets=total_targets,
        graph_mean_row_entropy=(entropy_sum / graph_rows if graph_rows else None),
        graph_mean_effective_neighbours=(effective_sum / graph_rows if graph_rows else None),
        graph_mean_diagonal_weight=(diagonal_sum / graph_rows if graph_rows else None),
        seconds=perf_counter() - start,
    )

    prediction_result: dict[str, Any] = {
        "y_pred": torch.cat(y_pred_parts, dim=0).contiguous(),
        "y_true": torch.cat(y_true_parts, dim=0).contiguous(),
        "last_context_target": torch.cat(last_parts, dim=0).contiguous(),
        "sample_idx": torch.cat(sample_parts, dim=0).contiguous(),
        "origin_idx": torch.cat(origin_parts, dim=0).contiguous(),
        "target_indices": torch.cat(target_index_parts, dim=0).contiguous(),
        "channels": ["close"],
        "horizons": [1],
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
    primary_tensor = metric_results["cumulative_log_change_mae"].reshape(-1)
    if primary_tensor.numel() != 1:
        raise RuntimeError("One-step Log MAE did not resolve to one scalar.")
    primary_score = float(primary_tensor.item())

    selected_graph = (
        torch.cat(selected_graph_parts, dim=0).contiguous()
        if selected_graph_parts
        else None
    )
    per_layer: list[Tensor | None] = [
        torch.cat(parts, dim=0).contiguous() if parts else None
        for parts in per_layer_parts
    ]
    graph_artifacts: dict[str, Any] = {
        "selected": selected_graph,
        "base": None,
        "dynamic": None,
        "per_layer": per_layer,
        "asset_cols": list(dataset.asset_cols),
        "graph_type": model.run_config.spatial_module_type,
        "graph_orientation": OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
        "diagonal_policy": "eligible_in_official_softmax; no extra identity added",
        "window_idx": torch.cat(window_parts, dim=0).contiguous(),
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "dates": dates,
    }
    token_artifacts = {
        "generated_s1": torch.cat(generated_parts, dim=0).contiguous(),
        "target_s1": torch.cat(target_parts, dim=0).contiguous(),
        "asset_cols": list(dataset.asset_cols),
        "prediction_length": 1,
        "model_family": "official_basedygraph",
        "token_dtype": "int16",
    }
    diagnostics = {
        "primary_score": primary_score,
        "primary_metric": "cumulative_log_change_mae",
        "primary_horizons": [1],
        "validation_s1_accuracy": classification.accuracy,
        "validation_cross_entropy": classification.loss,
        "invalid_candle_rate_percent": 100.0 * invalid_count / max(invalid_total, 1),
        "graph_summary": _graph_summary(graph_artifacts),
        "validation_examples": classification.examples,
        "validation_seconds": classification.seconds,
    }
    return ValidationArtifacts(
        classification=classification,
        prediction_result=prediction_result,
        metric_results=metric_results,
        metric_table=metric_table,
        primary_score=primary_score,
        token_artifacts=token_artifacts,
        graph_artifacts=graph_artifacts,
        diagnostics=diagnostics,
    )


def _flatten_metrics(metric_results: Mapping[str, Tensor]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, values in metric_results.items():
        flat = torch.as_tensor(values).detach().cpu().reshape(-1)
        if flat.numel() == 1:
            flattened[f"val/{name}/h1"] = float(flat.item())
    return flattened


def _build_checkpoint(
    *,
    model: OfficialBaseDyGraphOneStep,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    patience_count: int,
    history: list[dict[str, Any]],
    run_signature: str,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(patience_count),
        "history": list(history),
        "rng_state": capture_rng_state(),
        "run_signature": run_signature,
        "resolved_config": dict(resolved_config),
        "run_metadata": dict(run_metadata),
    }


def _load_checkpoint(
    path: Path,
    *,
    model: OfficialBaseDyGraphOneStep,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    expected_signature: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("run_signature") != expected_signature:
        raise ValueError("Checkpoint signature differs from the requested run.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _init_wandb(args: argparse.Namespace, resolved_config: Mapping[str, Any], metadata: Mapping[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        tags=list(args.wandb_tags),
        config={"experiment": dict(resolved_config), "runtime": dict(metadata)},
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    max_epochs = _validate_positive(args.max_epochs, "max_epochs")
    patience = _validate_positive(args.patience, "patience")
    train_batch_size = _validate_positive(args.train_batch_size, "train_batch_size")
    validation_batch_size = _validate_positive(args.validation_batch_size, "validation_batch_size")
    decode_batch = _validate_positive(args.decode_series_batch_size, "decode_series_batch_size")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip_norm < 0:
        raise ValueError("Invalid optimisation hyperparameter.")

    device = resolve_device(args.device)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    set_seed(args.seed)
    run_dir = _prepare_run_dir(args.output_dir, args.run_name, args.overwrite, args.resume)

    loaders = build_token_graph_dataloaders(
        args.train_cache,
        args.val_cache,
        data_mode="real",
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    _validate_dataset(loaders.train_dataset, name="Training")
    _validate_dataset(loaders.validation_dataset, name="Validation")

    train_dataset = _limit_dataset(loaders.train_dataset, args.max_train_windows)
    validation_dataset = _limit_dataset(loaders.validation_dataset, args.max_validation_windows)
    train_loader = _build_loader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    first_spatial = None if args.first_spatial_module_type == "same" else args.first_spatial_module_type
    model_config = OfficialBaseDyGraphRunConfig(
        num_states=TOKEN_VOCABULARY_SIZE,
        num_nodes=loaders.train_dataset.num_assets,
        context_length=loaders.train_dataset.context_length,
        d_model=args.d_model,
        nhead=args.temporal_heads,
        num_temporal_layers=args.temporal_layers,
        num_spatial_layers=args.spatial_layers,
        ff_mult=args.ff_mult,
        num_edge_heads=args.graph_heads,
        graph_hidden_dim=args.graph_hidden_dim,
        dropout=args.dropout,
        spatial_dropout=args.spatial_dropout,
        spatial_module_type=args.spatial_module_type,
        spatial_value=args.spatial_value,
        graph_activation=args.graph_activation,
        use_node_embedding=args.use_node_embedding,
        use_state_pair_bias=args.use_state_pair_bias,
        add_self_loops=args.add_self_loops,
        symmetric_graph=args.symmetric_graph,
        num_st_blocks=args.num_st_blocks,
        first_spatial_module_type=first_spatial,
    )
    model = OfficialBaseDyGraphOneStep(
        model_config,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_t_max=max_epochs,
    ).to(device)

    # Mandatory architecture parity check on a real cached mini-batch.
    parity_batch = next(iter(validation_loader))
    parity_context = torch.as_tensor(parity_batch["context_tokens"])[..., 0].to(device).long()
    parity = assert_official_one_step_parity(model, parity_context[:1])

    official_optim = model.configure_official_optimizers()
    optimizer = official_optim["optimizer"]
    scheduler_config = official_optim.get("lr_scheduler")
    scheduler = None
    if args.scheduler == "cosine":
        if not isinstance(scheduler_config, Mapping):
            raise RuntimeError("Official optimiser configuration has no scheduler mapping.")
        scheduler = scheduler_config["scheduler"]
    scaler = _new_grad_scaler(use_amp)

    forecasting_config = load_yaml(args.forecasting_config.expanduser().resolve())
    tokenizer = KronosTokenizerAdapter.from_config(
        forecasting_config,
        series_batch_size=decode_batch,
    ).load()
    raw_train_split = _load_raw_training_split(
        args.data_dir.expanduser().resolve(),
        expected_asset_cols=loaders.train_dataset.asset_cols,
    )

    repository_root = Path(__file__).resolve().parents[2]
    project_commit = _git_value(["rev-parse", "HEAD"], repository_root)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    compatible_graph_type = {
        "none": "none",
        "static_graph": "static_graph",
        "dynamic_graph": "dynamic_graph",
        "dynamic_base": "dynamic_base",
    }[args.spatial_module_type]
    resolved_config: dict[str, Any] = {
        "model_family": "official_basedygraph",
        "official_basedygraph": model_config.to_dict(),
        "training": {
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "gradient_clip_norm": args.gradient_clip_norm,
            "seed": args.seed,
            "mixed_precision": use_amp,
        },
        # Compatibility summary for the existing saved-metric/graph notebook.
        "models": {
            "dynamic_graph": {
                "num_nodes": model_config.num_nodes,
                "d_model": model_config.d_model,
                "num_st_blocks": model_config.num_st_blocks,
                "temporal": {
                    "type": "transformer",
                    "num_layers": model_config.num_temporal_layers,
                    "num_heads": model_config.nhead,
                    "feedforward_multiplier": model_config.ff_mult,
                    "dropout": model_config.dropout,
                },
                "graph": {
                    "type": compatible_graph_type,
                    "num_heads": model_config.num_edge_heads,
                    "hidden_dim": model_config.graph_hidden_dim,
                    "activation": model_config.graph_activation,
                    "add_self_loops": model_config.add_self_loops,
                    "diagonal_is_eligible": True,
                },
                "future_predictor": {"type": "official_direct_next_state_head", "num_layers": 0},
                "heads": {
                    "prediction_length": 1,
                    "evaluation_horizons": [1],
                    "s1_vocabulary_size": TOKEN_VOCABULARY_SIZE,
                    "future_token_mode": "coarse_only",
                },
            }
        },
    }
    run_metadata: dict[str, Any] = {
        "run_name": args.run_name,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "official_basedygraph",
        "project_commit": project_commit,
        "basedygraph_expected_commit": PINNED_BASEDYGRAPH_COMMIT,
        "basedygraph_observed_commit": model.external_commit,
        "train_cache_path": str(args.train_cache.expanduser().resolve()),
        "validation_cache_path": str(args.val_cache.expanduser().resolve()),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "forecasting_config_path": str(args.forecasting_config.expanduser().resolve()),
        "device": str(device),
        "active_cuda_amp": use_amp,
        "train_batch_size": train_batch_size,
        "validation_batch_size": validation_batch_size,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "asset_cols": list(loaders.train_dataset.asset_cols),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "graph_orientation": OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION,
        "graph_diagonal_policy": "eligible in scorer softmax; add_self_loops controls only extra identity addition",
        "adapter_trainable_parameters": 0,
        "parity_test": parity,
    }
    signature_payload = {
        "resolved_config": resolved_config,
        "train_cache": str(args.train_cache.expanduser().resolve()),
        "val_cache": str(args.val_cache.expanduser().resolve()),
        "basedygraph_commit": model.external_commit,
    }
    run_signature = _signature(signature_payload)
    run_metadata["run_signature"] = run_signature
    atomic_json_save(resolved_config, run_dir / "resolved_config.json")
    atomic_json_save(run_metadata, run_dir / "run_metadata.json")

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    patience_count = 0
    start_epoch = 1
    last_checkpoint_path = run_dir / "last_checkpoint.pt"
    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(last_checkpoint_path)
        checkpoint = _load_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            expected_signature=run_signature,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_count = int(checkpoint["evaluations_without_improvement"])
        history = list(checkpoint.get("history", []))

    wandb_run = _init_wandb(args, resolved_config, run_metadata)
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start = perf_counter()
            train_metrics = _run_classification_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                gradient_clip_norm=args.gradient_clip_norm,
                description=f"epoch {epoch} train",
            )
            validation = _validation_artifacts(
                model=model,
                loader=validation_loader,
                dataset=loaders.validation_dataset,
                device=device,
                use_amp=use_amp,
                tokenizer=tokenizer,
                raw_train_split=raw_train_split,
                decode_series_batch_size=decode_batch,
            )

            improved = validation.primary_score < best_score - 1.0e-12
            if improved:
                best_score = validation.primary_score
                best_epoch = epoch
                patience_count = 0
            else:
                patience_count += 1

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_s1_accuracy": train_metrics.accuracy,
                "validation_loss": validation.classification.loss,
                "validation_s1_accuracy": validation.classification.accuracy,
                "generated_s1_accuracy": validation.classification.accuracy,
                "validation_primary_score": validation.primary_score,
                "graph_mean_row_entropy": validation.classification.graph_mean_row_entropy,
                "graph_mean_effective_neighbours": validation.classification.graph_mean_effective_neighbours,
                "graph_mean_diagonal_weight": validation.classification.graph_mean_diagonal_weight,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "evaluations_without_improvement": patience_count,
                "epoch_seconds": perf_counter() - epoch_start,
            }
            record.update(_flatten_metrics(validation.metric_results))
            history.append(record)
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")

            # Step the optional official cosine scheduler before checkpointing so
            # a resumed run starts with exactly the same next-epoch LR as an
            # uninterrupted run. The history row above intentionally records the
            # LR that was used during the epoch just completed.
            if scheduler is not None:
                scheduler.step()

            checkpoint = _build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                patience_count=patience_count,
                history=history,
                run_signature=run_signature,
                resolved_config=resolved_config,
                run_metadata=run_metadata,
            )
            atomic_torch_save(checkpoint, last_checkpoint_path)

            if improved:
                atomic_torch_save(checkpoint, run_dir / "best_checkpoint.pt")
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "prediction_result": validation.prediction_result,
                        "metric_results": validation.metric_results,
                        "diagnostics": validation.diagnostics,
                    },
                    run_dir / "best_validation_predictions.pt",
                )
                atomic_torch_save(
                    {"epoch": epoch, "token_artifacts": validation.token_artifacts},
                    run_dir / "best_validation_tokens.pt",
                )
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "graph_artifacts": validation.graph_artifacts,
                        "summary": _graph_summary(validation.graph_artifacts),
                    },
                    run_dir / "best_validation_graphs.pt",
                )
                atomic_csv_save(validation.metric_table, run_dir / "best_validation_metric_table.csv")
                diagnostics = dict(validation.diagnostics)
                diagnostics["epoch"] = epoch
                atomic_json_save(diagnostics, run_dir / "best_validation_diagnostics.json")

            print(
                f"Epoch {epoch:03d}: train CE={train_metrics.loss:.6f}, "
                f"train acc={train_metrics.accuracy:.4%}, "
                f"val CE={validation.classification.loss:.6f}, "
                f"val acc={validation.classification.accuracy:.4%}, "
                f"val Log MAE={validation.primary_score:.9f}, "
                f"best={best_score:.9f} (epoch {best_epoch})"
            )

            if wandb_run is not None:
                wandb_run.log(record, step=epoch)

            if patience_count >= patience:
                print(f"Early stopping after {patience} non-improving epochs.")
                break

        run_metadata["status"] = "completed"
        run_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_metadata["best_epoch"] = best_epoch
        run_metadata["best_score"] = best_score
        run_metadata["epochs_completed"] = len(history)
        atomic_json_save(run_metadata, run_dir / "run_metadata.json")
    except Exception:
        run_metadata["status"] = "failed"
        run_metadata["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_save(run_metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

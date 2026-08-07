from __future__ import annotations

"""Dimitri BaseDyGraph-V2 with native s1 token input and direct price output.

This diagnostic keeps the exact x0jhc0tx four-block dual-fusion backbone and
replaces only the 1,024-way next-token head with a scalar next-Close head.  The
model is trained densely at every teacher-forced one-step transition, while
checkpoint selection uses the one-minute cumulative-log-change MAE at the end
of the configured context on the selected test split. The selected membership
can be Dimitri's physical files or the dissertation's canonical chronological
repartition.

The run is deliberately test-selected and must not be reported as held-out
performance.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dimitri_anchor_tokens import file_sha256
from src.data.dimitri_token_price import (
    DimitriTokenPriceDataset,
    DimitriTokenPriceWindowSpec,
    load_token_price_splits,
    make_clean_physical_split,
    normalise_split_mode,
    validate_token_split,
)
from src.evaluation.metrics import ForecastEvaluator
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_SOURCE_HASHES,
    DIMITRI_TOKEN_PRICE_CONTRACT,
    DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT,
    DIMITRI_X0_CONFIG,
    DIMITRI_X0_TRAINING,
    build_absolute_correlation_prior,
    build_sector_prior,
    extract_base_graph_logits,
    extract_dynamic_alphas,
    initialise_base_graphs_from_prior,
    instantiate_dimitri_token_to_price_model,
    parameter_count,
    resolved_per_block_contract,
    verify_dimitri_source_snapshot,
)
from src.utils.metric_tables import make_evaluation_table


GRAPH_ORIENTATION = "row=target,column=source"
PUBLIC_HORIZONS = (1,)
EPS = 1.0e-8


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _atomic_json(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _project_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        return None


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _run_signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_entropy(attention: Tensor, eps: float = 1.0e-12) -> Tensor:
    values = attention.float()
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(eps)
    values = values.clamp_min(eps)
    return -(values * values.log()).sum(dim=-1)


def _graph_diagnostics(graphs: Sequence[Tensor], origin_position: int) -> dict[str, Any]:
    if len(graphs) != 4:
        raise AssertionError(f"Expected four graph layers; observed {len(graphs)}.")
    rows: dict[str, Any] = {}
    for layer_index, graph in enumerate(graphs):
        values = torch.as_tensor(graph).float()
        if values.ndim != 5:
            raise ValueError("Graph tensors must have shape [B,T,H,N,N].")
        origin = values[:, origin_position]
        entropy = _row_entropy(origin)
        rows[f"layer_{layer_index}_origin_entropy"] = float(entropy.mean().item())
        rows[f"layer_{layer_index}_origin_effective_neighbours"] = float(
            entropy.exp().mean().item()
        )
        rows[f"layer_{layer_index}_origin_zero_fraction"] = float(
            (origin == 0).float().mean().item()
        )
        rows[f"layer_{layer_index}_all_time_entropy"] = float(
            _row_entropy(values).mean().item()
        )
    return rows


def _inverse_close(
    predicted_normalised: Tensor,
    close_mean: Tensor,
    close_std: Tensor,
) -> Tensor:
    return (
        predicted_normalised.squeeze(-1).float()
        * close_std[:, None, :].float().clamp_min(1.0e-8)
        + close_mean[:, None, :].float()
    )


def _dense_one_step_loss(
    predicted_normalised: Tensor,
    raw_close: Tensor,
    close_mean: Tensor,
    close_std: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return bps-scaled objective and native dense one-step CLG-MAE."""
    predicted_raw = _inverse_close(predicted_normalised, close_mean, close_std)
    current_raw = raw_close[:, :, :-1].transpose(1, 2).float()
    true_next_raw = raw_close[:, :, 1:].transpose(1, 2).float()
    if tuple(predicted_raw.shape) != tuple(true_next_raw.shape):
        raise AssertionError(
            f"Predicted/true sequence shapes differ: {tuple(predicted_raw.shape)} "
            f"vs {tuple(true_next_raw.shape)}."
        )
    predicted_change = (
        predicted_raw.clamp_min(EPS).log() - current_raw.clamp_min(EPS).log()
    )
    true_change = (
        true_next_raw.clamp_min(EPS).log() - current_raw.clamp_min(EPS).log()
    )
    native = F.l1_loss(predicted_change, true_change)
    return native * 10_000.0, native


def _origin_values(
    *,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    context_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    origin_position = int(context_length - 1)
    predicted_normalised = torch.as_tensor(output["next_close_normalised"])
    if origin_position >= predicted_normalised.shape[1]:
        raise IndexError("Forecast origin lies outside next-close outputs.")
    close_mean = torch.as_tensor(batch["close_mean"]).to(device=device).float()
    close_std = torch.as_tensor(batch["close_std"]).to(device=device).float()
    predicted_raw = _inverse_close(
        predicted_normalised[:, origin_position : origin_position + 1],
        close_mean,
        close_std,
    ).unsqueeze(-1)  # [B,1,N,1]
    raw_close = torch.as_tensor(batch["raw_close"]).to(device=device).float()
    last_raw = raw_close[:, :, origin_position].unsqueeze(-1)
    true_raw = raw_close[:, :, origin_position + 1].unsqueeze(1).unsqueeze(-1)
    return predicted_raw, true_raw, last_raw


def _build_loader(
    dataset: DimitriTokenPriceDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def _training_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    context_length: int,
) -> dict[str, Any]:
    model.train()
    native_sum = 0.0
    count = 0
    diagnostics_sum: dict[str, float] = {}
    batches = 0
    start = perf_counter()

    for batch in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        state_ids = torch.as_tensor(batch["state_ids"]).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        raw_close = torch.as_tensor(batch["raw_close"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        close_mean = torch.as_tensor(batch["close_mean"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        close_std = torch.as_tensor(batch["close_std"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(state_ids)
        objective, native = _dense_one_step_loss(
            output["next_close_normalised"],
            raw_close,
            close_mean,
            close_std,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite dense token-to-price objective.")
        objective.backward()
        optimizer.step()

        target_count = int(raw_close[:, :, 1:].numel())
        native_sum += float(native.detach().item()) * target_count
        count += target_count
        batch_diag = _graph_diagnostics(
            output["block_graph_attns"],
            origin_position=context_length - 1,
        )
        for key, value in batch_diag.items():
            diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value)
        batches += 1

    result: dict[str, Any] = {
        "dense_native_log_mae": native_sum / max(count, 1),
        "dense_objective_bps": 10_000.0 * native_sum / max(count, 1),
        "seconds": perf_counter() - start,
    }
    for key, value in diagnostics_sum.items():
        result[key] = value / max(batches, 1)
    return result


@torch.inference_mode()
def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    context_length: int,
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    retain_graphs: bool,
    description: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    window_starts: list[Tensor] = []
    dates: list[str] = []
    per_layer_parts: list[list[Tensor]] = [[], [], [], []]
    aggregate_sums: list[Tensor | None] = [None, None, None, None]
    aggregate_counts = [0, 0, 0, 0]
    per_window_entropy: list[list[Tensor]] = [[], [], [], []]
    dense_native_sum = 0.0
    dense_count = 0
    start = perf_counter()

    for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        state_ids = torch.as_tensor(batch["state_ids"]).to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )
        raw_close = torch.as_tensor(batch["raw_close"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        close_mean = torch.as_tensor(batch["close_mean"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        close_std = torch.as_tensor(batch["close_std"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        output = model(state_ids)
        _, dense_native = _dense_one_step_loss(
            output["next_close_normalised"],
            raw_close,
            close_mean,
            close_std,
        )
        dense_targets = int(raw_close[:, :, 1:].numel())
        dense_native_sum += float(dense_native.item()) * dense_targets
        dense_count += dense_targets

        predicted, true, last = _origin_values(
            output=output,
            batch=batch,
            context_length=context_length,
            device=device,
        )
        predictions.append(predicted.detach().cpu().contiguous())
        targets.append(true.detach().cpu().contiguous())
        last_values.append(last.detach().cpu().contiguous())
        sample_idx = torch.as_tensor(batch["sample_idx"]).long().cpu()
        window_start = torch.as_tensor(batch["window_start"]).long().cpu()
        origin_idx = window_start + int(context_length - 1)
        sample_indices.append(sample_idx)
        window_starts.append(window_start)
        origin_indices.append(origin_idx)
        target_indices.append((origin_idx + 1).unsqueeze(-1))
        batch_dates = batch["window_date"]
        if isinstance(batch_dates, str):
            dates.append(batch_dates)
        else:
            dates.extend(str(value) for value in batch_dates)

        block_graphs = output.get("block_graph_attns") or []
        if len(block_graphs) != 4:
            raise AssertionError("Expected four graph layers.")
        for layer_index, graph in enumerate(block_graphs):
            values = torch.as_tensor(graph).float()
            if values.ndim != 5:
                raise ValueError("Graph must have shape [B,T,H,N,N].")
            if retain_graphs:
                per_layer_parts[layer_index].append(
                    values[:, context_length - 1].detach().cpu().to(torch.float16)
                )
            aggregate = values.sum(dim=(0, 1, 2)).detach().cpu().double()
            aggregate_sums[layer_index] = (
                aggregate
                if aggregate_sums[layer_index] is None
                else aggregate_sums[layer_index] + aggregate
            )
            aggregate_counts[layer_index] += int(
                values.shape[0] * values.shape[1] * values.shape[2]
            )
            per_window_entropy[layer_index].append(
                _row_entropy(values[:, context_length - 1])
                .mean(dim=(1, 2))
                .detach()
                .cpu()
                .float()
            )

    prediction_result = {
        "task_type": "dimitri_v2_token_input_direct_price_one_minute",
        "y_pred": torch.cat(predictions, dim=0),
        "y_true": torch.cat(targets, dim=0),
        "last_context_target": torch.cat(last_values, dim=0),
        "sample_idx": torch.cat(sample_indices, dim=0),
        "origin_idx": torch.cat(origin_indices, dim=0),
        "target_indices": torch.cat(target_indices, dim=0),
        "window_start": torch.cat(window_starts, dim=0),
        "window_date": dates,
        "dates": dates,
        "asset_cols": list(asset_cols),
        "channels": ["close"],
        "horizons": [1],
        "output_space": "raw",
        "context_length": int(context_length),
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
    selection_score = float(
        metric_results["cumulative_log_change_mae"].reshape(-1)[0].item()
    )

    graph_artifacts = None
    if retain_graphs:
        per_layer = [torch.cat(parts, dim=0) for parts in per_layer_parts]
        aggregates = [
            (values / max(aggregate_counts[index], 1)).float()
            for index, values in enumerate(aggregate_sums)
            if values is not None
        ]
        if len(aggregates) != 4:
            raise AssertionError("Missing all-time graph aggregates.")
        graph_artifacts = {
            "selected": per_layer[-1],
            "per_layer": per_layer,
            "per_layer_all_time_aggregate": aggregates,
            "per_layer_all_time_counts": aggregate_counts,
            "per_layer_window_entropy": [
                torch.cat(parts, dim=0) for parts in per_window_entropy
            ],
            "dynamic_alpha_per_layer": extract_dynamic_alphas(model),
            "base_graph_logits_per_layer": extract_base_graph_logits(model),
            "graph_orientation": GRAPH_ORIENTATION,
            "asset_cols": list(asset_cols),
            "dates": dates,
            "sample_idx": prediction_result["sample_idx"],
            "origin_idx": prediction_result["origin_idx"],
            "window_start": prediction_result["window_start"],
            "saved_graph_time": (
                f"forecast-origin token position {context_length - 1}"
            ),
            "aggregate_graph_scope": (
                "all sequence positions, all heads and all split windows"
            ),
            "graph_heads_per_layer": [
                int(values.shape[1]) for values in per_layer
            ],
        }

    return {
        "selection_score": selection_score,
        "dense_native_log_mae": dense_native_sum / max(dense_count, 1),
        "prediction_result": prediction_result,
        "metric_results": metric_results,
        "metric_table": metric_table,
        "graph_artifacts": graph_artifacts,
        "seconds": perf_counter() - start,
    }


def _save_split(
    *,
    run_dir: Path,
    split: str,
    epoch: int,
    evaluation: Mapping[str, Any],
    prior_metadata: Mapping[str, Any],
) -> None:
    public_split = "validation" if split == "val" else split
    prediction_wrapper = {
        "epoch": int(epoch),
        "prediction_result": evaluation["prediction_result"],
    }
    graph_artifacts = dict(evaluation["graph_artifacts"])
    graph_artifacts["prior"] = dict(prior_metadata)
    graph_wrapper = {
        "epoch": int(epoch),
        "graph_artifacts": graph_artifacts,
    }
    predictions_path = run_dir / f"best_{public_split}_predictions.pt"
    graphs_path = run_dir / f"best_{public_split}_graphs.pt"
    metrics_path = run_dir / f"best_{public_split}_metric_table.csv"
    diagnostics_path = run_dir / f"best_{public_split}_diagnostics.json"
    _atomic_torch(prediction_wrapper, predictions_path)
    _atomic_torch(graph_wrapper, graphs_path)
    _atomic_csv(pd.DataFrame(evaluation["metric_table"]), metrics_path)
    _atomic_json(
        {
            "split": public_split,
            "epoch": int(epoch),
            "selection_score": float(evaluation["selection_score"]),
            "dense_native_log_mae": float(evaluation["dense_native_log_mae"]),
            "seconds": float(evaluation["seconds"]),
            "prior": dict(prior_metadata),
        },
        diagnostics_path,
    )
    analysis_dir = run_dir / "analysis" / public_split
    _copy_or_link(predictions_path, analysis_dir / "predictions.pt")
    _copy_or_link(graphs_path, analysis_dir / "graphs.pt")
    _copy_or_link(metrics_path, analysis_dir / "metric_table.csv")
    _copy_or_link(diagnostics_path, analysis_dir / "diagnostics.json")


def _checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    best_epoch: int,
    without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    signature: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(without_improvement),
        "history": [dict(row) for row in history],
        "run_signature": signature,
        "rng_state": _rng_state(),
    }


def _load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    expected_signature: str,
    restore_rng: bool,
) -> dict[str, Any]:
    payload = _torch_load(path)
    if payload.get("run_signature") != expected_signature:
        raise ValueError("Saved checkpoint run signature differs.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng:
        _restore_rng(payload["rng_state"])
    return payload


def _build_prior(
    *,
    prior_type: str,
    asset_cols: Sequence[str],
    company_profiles: Path | None,
    clean_train_split: Mapping[str, Any],
    correlation_threshold: float | None,
    split_mode: str,
) -> tuple[Tensor, dict[str, Any]]:
    if prior_type == "sector":
        if company_profiles is None or not company_profiles.is_file():
            raise FileNotFoundError(
                "Sector prior requires --company-profiles pointing to company_profiles.csv."
            )
        prior, labels = build_sector_prior(
            asset_cols,
            company_profiles,
            level="sector",
            self_loops=False,
            off_block=0.0,
        )
        metadata = {
            "type": "sector",
            "company_profiles": str(company_profiles),
            "company_profiles_sha256": file_sha256(company_profiles),
            "labels": labels,
            "self_loops": False,
            "off_block": 0.0,
        }
    elif prior_type == "correlation":
        prior = build_absolute_correlation_prior(
            clean_train_split,
            asset_cols=asset_cols,
            threshold=correlation_threshold,
        )
        metadata = {
            "type": "absolute_close_return_correlation",
            "fit_split": f"{normalise_split_mode(split_mode)}_train_only",
            "threshold": correlation_threshold,
            "self_loops": False,
        }
    elif prior_type == "none":
        num_nodes = len(asset_cols)
        prior = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
        prior.fill_diagonal_(0.0)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        metadata = {"type": "none_uniform_nonself"}
    else:
        raise ValueError("prior_type must be sector, correlation, or none.")
    metadata.update(
        {
            "shape": list(prior.shape),
            "row_sum_max_error": float((prior.sum(dim=-1) - 1.0).abs().max().item()),
            "minimum": float(prior.min().item()),
            "maximum": float(prior.max().item()),
        }
    )
    return prior, metadata


def _resolved_config(
    *,
    args: argparse.Namespace,
    asset_cols: Sequence[str],
    per_block: Mapping[str, Sequence[Any]],
    prior_metadata: Mapping[str, Any],
    token_summaries: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_family": "dimitri_basedygraph_v2_token_to_price",
        "model": {
            "family": "dimitri_basedygraph_v2",
            "input_representation": "native_kronos_s1_tokens",
            "output_representation": "context_normalised_close",
            "output_head": "Linear(96,1)_dense_next_close",
            "num_st_blocks": 4,
            "graph": {
                "type": "dual_fusion",
                "scope": "per_timestep",
                "num_heads": int(per_block["num_edge_heads"][-1]),
                "num_heads_per_layer": [
                    int(value) for value in per_block["num_edge_heads"]
                ],
                "hidden_dims_per_layer": [
                    int(value) for value in per_block["graph_hidden_dims"]
                ],
                "activations_by_layer": list(per_block["activations"]),
                "add_self_loops": False,
                "orientation": GRAPH_ORIENTATION,
                "slow_window": 32,
                "fast_window": 4,
                "learned_base_graph": True,
                "base_prior": dict(prior_metadata),
            },
            "temporal": {
                "type": "causal_transformer",
                "d_model": 96,
                "num_layers_per_block": 1,
                "num_heads": 4,
                "context_window": 180,
                "feedforward_multiplier": 1,
            },
        },
        "dimitri_basedygraph_v2": dict(DIMITRI_X0_CONFIG),
        "data": {
            "context_length": int(args.context_length),
            "teacher_forced_continuation_length": int(args.continuation_length),
            "sequence_length": int(args.context_length + args.continuation_length),
            "stride": int(args.stride),
            "horizons": [1],
            "training_transitions": "all_teacher_forced_next_steps",
            "forecast_origin_position": int(args.context_length - 1),
            "split_mode": normalise_split_mode(args.split_mode),
            "physical_split_membership_preserved": args.split_mode == "physical",
            "canonical_chronological_repartition": args.split_mode == "canonical",
            "asset_cols": list(asset_cols),
            "token_summaries": dict(token_summaries),
        },
        "training": {
            "loss": "dense_one_step_cumulative_log_change_mae",
            "loss_bps_scale": 10_000.0,
            "selection_metric": "test_origin_1m_cumulative_log_change_mae",
            "selection_split": f"{normalise_split_mode(args.split_mode)}_test",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": int(args.max_epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "seed": int(args.seed),
            "precision": "float32",
            "graph_regularisation": "none",
            "do_not_report": True,
        },
        "source_hashes": dict(DIMITRI_SOURCE_HASHES),
        "per_block": {key: list(value) for key, value in per_block.items()},
    }


def _init_wandb(args: argparse.Namespace, config: Mapping[str, Any]) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        tags=[
            "DO-NOT-REPORT",
            "dimitri-v2",
            "token-input",
            "price-output",
            f"context-{args.context_length}",
            f"split-{args.split_mode}",
            f"prior-{args.graph_prior}",
            "test-selected",
        ],
        config=dict(config),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dimitri V2 token-input/direct-price one-minute diagnostic."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split-mode",
        choices=("physical", "canonical"),
        default="physical",
        help=(
            "physical preserves the three stored file memberships; canonical "
            "combines them and applies the project date boundaries"
        ),
    )
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--company-profiles", type=Path)
    parser.add_argument("--context-length", type=int, default=180)
    parser.add_argument("--continuation-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument(
        "--graph-prior",
        choices=("sector", "correlation", "none"),
        default="sector",
    )
    parser.add_argument("--prior-scale", type=float, default=4.0)
    parser.add_argument("--prior-jitter", type=float, default=0.02)
    parser.add_argument("--correlation-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--selection-batch-size", type=int, default=1)
    parser.add_argument("--export-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="BaseDyGraph V2 Token Price TEST-CONTAMINATED",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if args.context_length <= 0 or args.continuation_length <= 0 or args.stride <= 0:
        raise ValueError("Context, continuation and stride must be positive.")
    if args.context_length + args.continuation_length > 512:
        raise ValueError("Sequence length exceeds Dimitri V2 max_seq_len=512.")
    if args.batch_size <= 0 or args.selection_batch_size <= 0 or args.export_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    args.split_mode = normalise_split_mode(args.split_mode)
    device = torch.device(args.device)
    spec = DimitriTokenPriceWindowSpec(
        context_length=args.context_length,
        continuation_length=args.continuation_length,
        stride=args.stride,
    )
    source_hashes = verify_dimitri_source_snapshot()
    token_summaries = {
        split: validate_token_split(
            args.token_dir / f"{split}.pt",
            split_name=split,
            split_mode=args.split_mode,
            spec=spec,
        )
        for split in ("train", "val", "test")
    }

    raw_splits = load_token_price_splits(
        args.data_dir,
        split_mode=args.split_mode,
    )
    clean_train_split = make_clean_physical_split(raw_splits["train"])
    datasets = {
        split: DimitriTokenPriceDataset(
            token_path=args.token_dir / f"{split}.pt",
            raw_split=raw_splits[split],
            split_name=split,
            split_mode=args.split_mode,
            spec=spec,
        )
        for split in ("train", "val", "test")
    }
    asset_cols = datasets["train"].asset_cols
    if not all(dataset.asset_cols == asset_cols for dataset in datasets.values()):
        raise ValueError("Token-price split asset orders differ.")

    prior, prior_metadata = _build_prior(
        prior_type=args.graph_prior,
        asset_cols=asset_cols,
        company_profiles=args.company_profiles,
        clean_train_split=clean_train_split,
        correlation_threshold=args.correlation_threshold,
        split_mode=args.split_mode,
    )

    _set_seed(args.seed)
    model = instantiate_dimitri_token_to_price_model()
    observed_parameters = parameter_count(model)
    if observed_parameters != DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT:
        raise AssertionError(
            f"Token-to-price parameter count {observed_parameters:,} differs from "
            f"expected {DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT:,}."
        )
    prior_initialisation = initialise_base_graphs_from_prior(
        model,
        prior,
        scale=args.prior_scale,
        jitter=args.prior_jitter,
        seed=args.seed,
    )
    prior_metadata.update(
        {
            "initialisation": prior_initialisation,
            "learnable": True,
        }
    )
    per_block = resolved_per_block_contract(model.cfg)

    signature_payload = {
        "contract": DIMITRI_TOKEN_PRICE_CONTRACT,
        "context": spec.to_dict(),
        "split_mode": args.split_mode,
        "graph_prior": prior_metadata,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "source_hashes": source_hashes,
        "token_hashes": {
            split: token_summaries[split]["sha256"]
            for split in token_summaries
        },
    }
    signature = _run_signature(signature_payload)
    run_dir = args.output_dir / args.run_name
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"{run_dir} is complete; nothing to do.")
            return

    resolved_config = _resolved_config(
        args=args,
        asset_cols=asset_cols,
        per_block=per_block,
        prior_metadata=prior_metadata,
        token_summaries=token_summaries,
    )
    _atomic_json(resolved_config, run_dir / "resolved_config.json")
    _atomic_torch(
        {
            "prior": prior,
            "metadata": prior_metadata,
            "asset_cols": asset_cols,
        },
        run_dir / "initial_graph_prior.pt",
    )
    pd.DataFrame(prior.numpy(), index=asset_cols, columns=asset_cols).to_csv(
        run_dir / "initial_graph_prior.csv"
    )

    metadata: dict[str, Any] = {
        "status": "running",
        "run_name": args.run_name,
        "experiment_contract": DIMITRI_TOKEN_PRICE_CONTRACT,
        "model_family": "dimitri_basedygraph_v2_token_to_price",
        "data_split_mode": args.split_mode,
        "selection_split": "test",
        "selection_split_contract": f"{args.split_mode}_test",
        "selection_metric": "test_origin_1m_cumulative_log_change_mae",
        "test_set_contaminated": True,
        "do_not_report": True,
        "project_commit": _project_commit(),
        "source_hashes": source_hashes,
        "run_signature": signature,
        "asset_cols": asset_cols,
        "graph_orientation": GRAPH_ORIENTATION,
        "graph_prior": prior_metadata,
        "data_dir": str(args.data_dir),
        "token_dir": str(args.token_dir),
        "context_length": args.context_length,
        "continuation_length": args.continuation_length,
        "sequence_length": spec.sequence_length,
        "stride": args.stride,
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["val"]),
        "test_windows": len(datasets["test"]),
        "parameter_count": observed_parameters,
        "trainable_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "started_at_utc": _utc_now(),
    }
    _atomic_json(metadata, metadata_path)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
    )
    train_loader = _build_loader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = _build_loader(
        datasets["test"],
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint = _load_checkpoint(
            run_dir / "last_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_signature=signature,
            restore_rng=True,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(checkpoint["evaluations_without_improvement"])
        history = [dict(row) for row in checkpoint["history"]]

    wandb_run = _init_wandb(args, resolved_config)
    try:
        for epoch in range(start_epoch, args.max_epochs + 1):
            train_metrics = _training_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                context_length=args.context_length,
            )
            test_evaluation = _evaluate(
                model=model,
                loader=test_loader,
                device=device,
                context_length=args.context_length,
                train_split=clean_train_split,
                asset_cols=asset_cols,
                retain_graphs=False,
                description=f"TEST selection epoch {epoch}",
            )
            score = float(test_evaluation["selection_score"])
            improved = score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                without_improvement = 0
            else:
                without_improvement += 1

            row: dict[str, Any] = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "training_dense_one_step_log_mae": train_metrics[
                    "dense_native_log_mae"
                ],
                "training_objective_bps": train_metrics["dense_objective_bps"],
                "test_origin_1m_log_mae": score,
                "test_dense_one_step_log_mae": test_evaluation[
                    "dense_native_log_mae"
                ],
                "selection_score": score,
                "validation_loss": score,
                "best_score_after_epoch": best_score,
                "best_epoch_after_epoch": best_epoch,
                "selection_split": "test",
                "train_seconds": train_metrics["seconds"],
                "test_selection_seconds": test_evaluation["seconds"],
            }
            row.update(
                {
                    f"training_{key}": value
                    for key, value in train_metrics.items()
                    if key.startswith("layer_")
                }
            )
            for layer_index, alpha_values in enumerate(extract_dynamic_alphas(model)):
                if alpha_values:
                    row[f"alpha_layer_{layer_index}_mean"] = float(
                        np.mean(alpha_values)
                    )
            history.append(row)
            _atomic_csv(pd.DataFrame(history), run_dir / "history.csv")
            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                without_improvement=without_improvement,
                history=history,
                signature=signature,
            )
            _atomic_torch(checkpoint, run_dir / "last_checkpoint.pt")
            if improved:
                _atomic_torch(checkpoint, run_dir / "best_checkpoint.pt")
            if wandb_run is not None:
                wandb_run.log(row, step=epoch)
            scheduler.step()
            if without_improvement >= args.patience:
                break

        best_checkpoint = _load_checkpoint(
            run_dir / "best_checkpoint.pt",
            model=model,
            optimizer=None,
            scheduler=None,
            expected_signature=signature,
            restore_rng=False,
        )
        export_scores: dict[str, float] = {}
        for split in ("train", "test", "val"):
            loader = _build_loader(
                datasets[split],
                batch_size=args.export_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            evaluation = _evaluate(
                model=model,
                loader=loader,
                device=device,
                context_length=args.context_length,
                train_split=clean_train_split,
                asset_cols=asset_cols,
                retain_graphs=True,
                description=f"selected checkpoint {split} export",
            )
            _save_split(
                run_dir=run_dir,
                split=split,
                epoch=int(best_checkpoint["epoch"]),
                evaluation=evaluation,
                prior_metadata=prior_metadata,
            )
            export_scores["validation" if split == "val" else split] = float(
                evaluation["selection_score"]
            )

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "epochs_completed": int(history[-1]["epoch"]),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "postselection_price_scores": export_scores,
                "analysis_splits_saved": ["train", "validation", "test"],
                "dynamic_alphas": extract_dynamic_alphas(model),
            }
        )
        _atomic_json(metadata, metadata_path)
        print("Completed:", run_dir)
        print("Data split mode:", args.split_mode)
        print("Best test-selected epoch:", best_epoch)
        print("Best test one-minute Log MAE:", best_score)
        print("Post-selection scores:", export_scores)
    except Exception:
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
            }
        )
        _atomic_json(metadata, metadata_path)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

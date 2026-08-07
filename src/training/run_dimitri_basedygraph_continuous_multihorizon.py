from __future__ import annotations

"""Dimitri BaseDyGraph-V2 continuous-input direct multi-horizon forecasting.

This runner is the canonical-grid extension of the completed one-minute
continuous-price diagnostic.  It preserves the exact Dimitri V2 dual-fusion
backbone and graph-prior initialisation while changing only the forecasting
contract:

* model input: the observed context only;
* public output: direct Close forecasts at configurable future horizons;
* objective: equally weighted cumulative-log-change MAE across those horizons;
* checkpoint selection: the same mean metric on the deliberately used test
  split.

No future candle is supplied to temporal encoding, graph inference, or spatial
message passing.  The stored continuation exists only to provide targets and
causal inverse-normalisation statistics from the observed context.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dimitri_continuous_price import (
    DIMITRI_CONTINUOUS_PRICE_CONTRACT as DATA_CONTRACT,
    DimitriContinuousPriceDataset,
    build_continuous_price_datasets,
)
from src.data.dimitri_token_price import (
    DimitriTokenPriceWindowSpec,
    make_clean_physical_split,
    normalise_split_mode,
)
from src.evaluation.metrics import ForecastEvaluator
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_CONTINUOUS_MULTI_HORIZON_CONTRACT,
    DIMITRI_SOURCE_HASHES,
    build_absolute_correlation_prior,
    build_sector_prior,
    dimitri_continuous_multi_horizon_parameter_count,
    extract_base_graph_logits,
    extract_dynamic_alphas,
    initialise_base_graphs_from_prior,
    instantiate_dimitri_continuous_multi_horizon_model,
    parameter_count,
    resolved_per_block_contract,
    verify_dimitri_source_snapshot,
)
from src.training.run_dimitri_basedygraph_continuous_price import (
    GRAPH_ORIENTATION,
    _atomic_csv,
    _atomic_json,
    _atomic_torch,
    _build_loader,
    _checkpoint,
    _copy_or_link,
    _graph_diagnostics,
    _load_checkpoint,
    _project_commit,
    _row_entropy,
    _run_signature,
    _set_seed,
    _utc_now,
)
from src.utils.metric_tables import make_evaluation_table


EPS = 1.0e-8
DEFAULT_HORIZONS = (1, 5, 15, 30, 60)


def _normalise_horizons(
    values: Sequence[int],
    *,
    continuation_length: int,
) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in values)
    if not horizons:
        raise ValueError("At least one evaluation horizon is required.")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("Horizons must be unique and strictly increasing.")
    if any(value <= 0 for value in horizons):
        raise ValueError("Horizons must be positive.")
    if horizons[-1] > int(continuation_length):
        raise ValueError(
            f"Maximum horizon {horizons[-1]} exceeds continuation length "
            f"{continuation_length}."
        )
    return horizons


def _inverse_close(
    predicted_normalised: Tensor,
    close_mean: Tensor,
    close_std: Tensor,
) -> Tensor:
    """Convert [B,H,N,1] normalised Close values to raw [B,H,N]."""
    values = predicted_normalised.squeeze(-1).float()
    return (
        values
        * close_std[:, None, :].float().clamp_min(1.0e-8)
        + close_mean[:, None, :].float()
    )


def _multi_horizon_values(
    *,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    context_length: int,
    horizons: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return raw predictions/targets and the last observed Close."""
    horizon_values = tuple(int(value) for value in horizons)
    predicted_normalised = torch.as_tensor(output["future_close_normalised"])
    expected_h = len(horizon_values)
    if predicted_normalised.ndim != 4 or predicted_normalised.shape[1] != expected_h:
        raise ValueError(
            "future_close_normalised must have shape [B,H,N,1], got "
            f"{tuple(predicted_normalised.shape)}."
        )

    close_mean = torch.as_tensor(batch["close_mean"]).to(device=device).float()
    close_std = torch.as_tensor(batch["close_std"]).to(device=device).float()
    predicted_raw = _inverse_close(
        predicted_normalised,
        close_mean,
        close_std,
    ).unsqueeze(-1)

    raw_close = torch.as_tensor(batch["raw_close"]).to(device=device).float()
    origin_position = int(context_length - 1)
    target_positions = torch.tensor(
        [origin_position + value for value in horizon_values],
        dtype=torch.long,
        device=device,
    )
    if int(target_positions[-1]) >= int(raw_close.shape[-1]):
        raise IndexError(
            "Target positions exceed the stored context-plus-continuation "
            f"window: last target={int(target_positions[-1])}, "
            f"length={int(raw_close.shape[-1])}."
        )
    true_raw = raw_close.index_select(-1, target_positions).transpose(1, 2).unsqueeze(-1)
    last_raw = raw_close[:, :, origin_position].unsqueeze(-1)
    return predicted_raw, true_raw, last_raw


def _multi_horizon_loss(
    *,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    context_length: int,
    horizons: Sequence[int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return bps-scaled mean loss, native mean, and native per-horizon MAE."""
    predicted_raw, true_raw, last_raw = _multi_horizon_values(
        output=output,
        batch=batch,
        context_length=context_length,
        horizons=horizons,
        device=device,
    )
    base = last_raw[:, None, :, :]
    predicted_change = (
        predicted_raw.clamp_min(EPS).log() - base.clamp_min(EPS).log()
    )
    true_change = true_raw.clamp_min(EPS).log() - base.clamp_min(EPS).log()
    absolute_error = (predicted_change - true_change).abs().squeeze(-1)
    per_horizon = absolute_error.mean(dim=(0, 2))
    native = per_horizon.mean()
    return native * 10_000.0, native, per_horizon


def _training_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    context_length: int,
    horizons: Sequence[int],
) -> dict[str, Any]:
    model.train()
    horizon_values = tuple(int(value) for value in horizons)
    error_sums = torch.zeros(len(horizon_values), dtype=torch.float64)
    element_counts = torch.zeros(len(horizon_values), dtype=torch.float64)
    diagnostics_sum: dict[str, float] = {}
    batches = 0
    start = perf_counter()

    for batch in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        complete_values = torch.as_tensor(batch["continuous_values"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        context_values = complete_values[:, :, :context_length, :].contiguous()
        optimizer.zero_grad(set_to_none=True)
        output = model(context_values)
        objective, native, per_horizon = _multi_horizon_loss(
            output=output,
            batch=batch,
            context_length=context_length,
            horizons=horizon_values,
            device=device,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite multi-horizon price objective.")
        objective.backward()
        optimizer.step()

        batch_size = int(context_values.shape[0])
        num_nodes = int(context_values.shape[1])
        count = batch_size * num_nodes
        error_sums += per_horizon.detach().cpu().double() * count
        element_counts += count

        batch_diag = _graph_diagnostics(
            output["block_graph_attns"],
            origin_position=context_length - 1,
        )
        for key, value in batch_diag.items():
            diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value)
        batches += 1

    per_horizon_values = error_sums / element_counts.clamp_min(1.0)
    result: dict[str, Any] = {
        "mean_native_log_mae": float(per_horizon_values.mean().item()),
        "objective_bps": float(10_000.0 * per_horizon_values.mean().item()),
        "seconds": perf_counter() - start,
    }
    for horizon, value in zip(horizon_values, per_horizon_values.tolist(), strict=True):
        result[f"log_mae_h{horizon}"] = float(value)
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
    horizons: Sequence[int],
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    retain_graphs: bool,
    description: str,
) -> dict[str, Any]:
    model.eval()
    horizon_values = tuple(int(value) for value in horizons)
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
    error_sums = torch.zeros(len(horizon_values), dtype=torch.float64)
    element_counts = torch.zeros(len(horizon_values), dtype=torch.float64)
    start = perf_counter()

    for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        complete_values = torch.as_tensor(batch["continuous_values"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        context_values = complete_values[:, :, :context_length, :].contiguous()
        output = model(context_values)
        _objective, _native, per_horizon = _multi_horizon_loss(
            output=output,
            batch=batch,
            context_length=context_length,
            horizons=horizon_values,
            device=device,
        )
        batch_size = int(context_values.shape[0])
        num_nodes = int(context_values.shape[1])
        count = batch_size * num_nodes
        error_sums += per_horizon.detach().cpu().double() * count
        element_counts += count

        predicted, true, last = _multi_horizon_values(
            output=output,
            batch=batch,
            context_length=context_length,
            horizons=horizon_values,
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
        target_indices.append(
            origin_idx[:, None]
            + torch.tensor(horizon_values, dtype=torch.long)[None, :]
        )
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
            if int(values.shape[1]) != int(context_length):
                raise AssertionError(
                    f"Layer {layer_index} graph length {values.shape[1]} "
                    f"differs from context length {context_length}."
                )
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
        "task_type": "dimitri_v2_continuous_input_direct_price_multi_horizon",
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
        "horizons": list(horizon_values),
        "output_space": "raw",
        "context_length": int(context_length),
        "forecast_strategy": "direct_parallel_from_final_context",
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
    selection_values = torch.as_tensor(
        metric_results["cumulative_log_change_mae"]
    ).reshape(-1)
    if int(selection_values.numel()) != len(horizon_values):
        raise AssertionError(
            "Cumulative-log-change MAE did not return one value per horizon."
        )
    selection_score = float(selection_values.mean().item())
    direct_per_horizon = error_sums / element_counts.clamp_min(1.0)
    graph_diagnostics: dict[str, float] = {}
    for layer_index, parts in enumerate(per_window_entropy):
        if not parts:
            raise AssertionError(f"Missing layer-{layer_index} graph entropy.")
        entropy_values = torch.cat(parts, dim=0)
        graph_diagnostics[f"layer_{layer_index}_origin_entropy"] = float(
            entropy_values.mean().item()
        )
        graph_diagnostics[
            f"layer_{layer_index}_origin_effective_neighbours"
        ] = float(entropy_values.exp().mean().item())

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
                f"forecast-origin context position {context_length - 1}"
            ),
            "aggregate_graph_scope": (
                "all observed context positions, all heads and all split windows"
            ),
            "graph_heads_per_layer": [
                int(values.shape[1]) for values in per_layer
            ],
        }

    return {
        "selection_score": selection_score,
        "native_log_mae": float(direct_per_horizon.mean().item()),
        "per_horizon_log_mae": {
            int(horizon): float(value)
            for horizon, value in zip(
                horizon_values,
                direct_per_horizon.tolist(),
                strict=True,
            )
        },
        "graph_diagnostics": graph_diagnostics,
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
            "mean_cumulative_log_change_mae": float(evaluation["native_log_mae"]),
            "per_horizon_cumulative_log_change_mae": dict(
                evaluation["per_horizon_log_mae"]
            ),
            "graph_diagnostics": dict(evaluation["graph_diagnostics"]),
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
                "Sector prior requires --company-profiles pointing to "
                "company_profiles.csv."
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
            "row_sum_max_error": float(
                (prior.sum(dim=-1) - 1.0).abs().max().item()
            ),
            "minimum": float(prior.min().item()),
            "maximum": float(prior.max().item()),
        }
    )
    return prior, metadata


def _resolved_config(
    *,
    args: argparse.Namespace,
    horizons: Sequence[int],
    asset_cols: Sequence[str],
    per_block: Mapping[str, Sequence[Any]],
    prior_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    horizon_values = tuple(int(value) for value in horizons)
    return {
        "model_family": "dimitri_basedygraph_v2_continuous_multi_horizon",
        "model": {
            "family": "dimitri_basedygraph_v2",
            "input_representation": "context_normalised_ohlcva_continuous",
            "output_representation": "context_normalised_close",
            "output_head": (
                f"Linear(96,{len(horizon_values)})_direct_parallel_"
                "from_final_context"
            ),
            "forecast_strategy": "direct_parallel_from_final_context",
            "num_st_blocks": 4,
            "graph": {
                "type": "dual_fusion",
                "scope": "per_timestep_within_observed_context",
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
                "configured_context_window": 180,
                "effective_context_length": int(args.context_length),
                "feedforward_multiplier": 1,
            },
        },
        "dimitri_basedygraph_v2": {
            "d_model": 96,
            "num_st_blocks": 4,
            "temporal_heads": 4,
            "temporal_layers": 1,
            "temporal_context_window": 180,
            "graph_heads_per_layer": [
                int(value) for value in per_block["num_edge_heads"]
            ],
            "graph_hidden_dims_per_layer": [
                int(value) for value in per_block["graph_hidden_dims"]
            ],
            "graph_activations": list(per_block["activations"]),
            "slow_window": 32,
            "fast_window": 4,
            "scorer_value": "concat",
            "spatial_value": "concat",
            "learned_base_graph": True,
            "initial_alpha": 0.75,
        },
        # Compatibility view used by the current Graph Hub architecture table.
        "basedygraph_financial": {
            "d_model": 96,
            "num_st_blocks": 4,
            "temporal_layers": 1,
            "temporal_heads": 4,
            "ff_mult": 1,
            "graph_type": "dual_fusion",
            "graph_scope": "per_timestep_within_observed_context",
            "graph_heads": int(per_block["num_edge_heads"][-1]),
            "graph_hidden_dim": int(per_block["graph_hidden_dims"][-1]),
            "graph_activations": list(per_block["activations"]),
            "spatial_layers": 1,
            "prediction_length": len(horizon_values),
            "regularisation": {
                "target_entropy": None,
                "target_entropy_weight": 0.0,
                "temporal_smooth_weight": 0.0,
            },
        },
        "data": {
            "context_length": int(args.context_length),
            "target_continuation_length": int(args.continuation_length),
            "sequence_length": int(args.context_length + args.continuation_length),
            "model_input_length": int(args.context_length),
            "stride": int(args.stride),
            "horizons": list(horizon_values),
            "forecast_origin_position": int(args.context_length - 1),
            "target_positions_in_window": [
                int(args.context_length - 1 + value)
                for value in horizon_values
            ],
            "split_mode": normalise_split_mode(args.split_mode),
            "physical_split_membership_preserved": args.split_mode == "physical",
            "canonical_chronological_repartition": args.split_mode == "canonical",
            "asset_cols": list(asset_cols),
            "input_channels": ["open", "high", "low", "close", "volume", "amount"],
            "normalisation": {
                "statistics": "observed context rows only",
                "std_correction": 1,
                "clip": [-5.0, 5.0],
                "amount_forced_to_zero": True,
            },
        },
        "training": {
            "loss": {
                "type": "cumulative_log_change_mae",
                "horizon_weighting": "uniform",
                "bps_scale": 10_000.0,
            },
            "selection_metric": "test_five_horizon_mean_cumulative_log_change_mae",
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
            "continuous-input",
            "direct-five-horizon-price",
            f"context-{args.context_length}",
            f"stride-{args.stride}",
            f"split-{args.split_mode}",
            f"prior-{args.graph_prior}",
            "test-selected",
        ],
        config=dict(config),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dimitri V2 continuous-input direct multi-horizon price diagnostic."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split-mode",
        choices=("physical", "canonical"),
        default="canonical",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--company-profiles", type=Path)
    parser.add_argument("--context-length", type=int, default=60)
    parser.add_argument("--continuation-length", type=int, default=60)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
    )
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
        default="BaseDyGraph V2 Multi-Horizon TEST-CONTAMINATED",
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
        raise ValueError("Stored window length exceeds 512 positions.")
    if args.batch_size <= 0 or args.selection_batch_size <= 0 or args.export_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

    args.split_mode = normalise_split_mode(args.split_mode)
    horizons = _normalise_horizons(
        args.horizons,
        continuation_length=args.continuation_length,
    )
    device = torch.device(args.device)
    spec = DimitriTokenPriceWindowSpec(
        context_length=args.context_length,
        continuation_length=args.continuation_length,
        stride=args.stride,
    )
    source_hashes = verify_dimitri_source_snapshot()
    raw_splits, datasets = build_continuous_price_datasets(
        args.data_dir,
        split_mode=args.split_mode,
        spec=spec,
    )
    clean_train_split = make_clean_physical_split(raw_splits["train"])
    asset_cols = datasets["train"].asset_cols
    if not all(dataset.asset_cols == asset_cols for dataset in datasets.values()):
        raise ValueError("Continuous-price split asset orders differ.")

    prior, prior_metadata = _build_prior(
        prior_type=args.graph_prior,
        asset_cols=asset_cols,
        company_profiles=args.company_profiles,
        clean_train_split=clean_train_split,
        correlation_threshold=args.correlation_threshold,
        split_mode=args.split_mode,
    )

    _set_seed(args.seed)
    model = instantiate_dimitri_continuous_multi_horizon_model(
        input_channels=6,
        evaluation_horizons=horizons,
    )
    observed_parameters = parameter_count(model)
    expected_parameters = dimitri_continuous_multi_horizon_parameter_count(
        len(horizons)
    )
    if observed_parameters != expected_parameters:
        raise AssertionError(
            f"Multi-horizon parameter count {observed_parameters:,} differs "
            f"from expected {expected_parameters:,}."
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
        "contract": DIMITRI_CONTINUOUS_MULTI_HORIZON_CONTRACT,
        "context": spec.to_dict(),
        "horizons": list(horizons),
        "forecast_strategy": "direct_parallel_from_final_context",
        "split_mode": args.split_mode,
        "graph_prior": prior_metadata,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "source_hashes": source_hashes,
        "data_contract": DATA_CONTRACT,
        "input_representation": "context_normalised_ohlcva_continuous",
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
        horizons=horizons,
        asset_cols=asset_cols,
        per_block=per_block,
        prior_metadata=prior_metadata,
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
        "experiment_contract": DIMITRI_CONTINUOUS_MULTI_HORIZON_CONTRACT,
        "model_family": "dimitri_basedygraph_v2_continuous_multi_horizon",
        "data_split_mode": args.split_mode,
        "selection_split": "test",
        "selection_split_contract": f"{args.split_mode}_test",
        "selection_metric": "test_five_horizon_mean_cumulative_log_change_mae",
        "evaluation_horizons": list(horizons),
        "forecast_strategy": "direct_parallel_from_final_context",
        "test_set_contaminated": True,
        "do_not_report": True,
        "project_commit": _project_commit(),
        "project_git_commit": _project_commit(),
        "source_hashes": source_hashes,
        "run_signature": signature,
        "asset_cols": asset_cols,
        "graph_orientation": GRAPH_ORIENTATION,
        "graph_prior": prior_metadata,
        "data_dir": str(args.data_dir),
        "input_representation": "context_normalised_ohlcva_continuous",
        "input_channels": ["open", "high", "low", "close", "volume", "amount"],
        "context_length": args.context_length,
        "continuation_length": args.continuation_length,
        "sequence_length": spec.sequence_length,
        "model_input_length": args.context_length,
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
                horizons=horizons,
            )
            test_evaluation = _evaluate(
                model=model,
                loader=test_loader,
                device=device,
                context_length=args.context_length,
                horizons=horizons,
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
                "training_mean_log_mae": train_metrics["mean_native_log_mae"],
                "training_objective_bps": train_metrics["objective_bps"],
                "test_mean_log_mae": score,
                "selection_score": score,
                "validation_loss": score,
                "best_score_after_epoch": best_score,
                "best_epoch_after_epoch": best_epoch,
                "selection_split": "test",
                "train_seconds": train_metrics["seconds"],
                "test_selection_seconds": test_evaluation["seconds"],
                "training_graph_mean_row_entropy": train_metrics[
                    "layer_3_origin_entropy"
                ],
                "training_graph_mean_effective_neighbours": train_metrics[
                    "layer_3_origin_effective_neighbours"
                ],
                "test_graph_mean_row_entropy": test_evaluation[
                    "graph_diagnostics"
                ]["layer_3_origin_entropy"],
                "test_graph_mean_effective_neighbours": test_evaluation[
                    "graph_diagnostics"
                ]["layer_3_origin_effective_neighbours"],
            }
            for horizon in horizons:
                row[f"training_log_mae_h{horizon}"] = train_metrics[
                    f"log_mae_h{horizon}"
                ]
                row[f"test_log_mae_h{horizon}"] = test_evaluation[
                    "per_horizon_log_mae"
                ][int(horizon)]
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
                horizons=horizons,
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
        print("Best test-selected epoch:", best_epoch)
        print("Best test five-horizon mean Log MAE:", best_score)
        print("Post-selection mean scores:", export_scores)
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

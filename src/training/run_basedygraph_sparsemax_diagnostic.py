from __future__ import annotations

"""Four deliberately test-selected BaseDyGraph sparsemax diagnostics.

The experiment preserves the pinned BaseDyGraph temporal/graph/spatial blocks,
uses four interlaced ST blocks and one graph head per block, and changes only
the final graph activation to sparsemax.  The first three blocks remain
softmax.  Checkpoint selection uses the chronological October--December test
split; every output is therefore marked DO_NOT_REPORT.
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

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.evaluation.prediction_transforms import inverse_window_normalisation
from src.models.basedygraph_financial import (
    BaseDyGraphFinancialConfig,
    BaseDyGraphGraphRegularisationConfig,
    BaseDyGraphTeacherForcedTokenOutput,
    OfficialBaseDyGraphContinuousForecaster,
    OfficialBaseDyGraphTeacherForcedOneStepForecaster,
    financial_graph_artifact_metadata,
    graph_regularisation_loss,
)
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.training.run_basedygraph_financial import (
    _checkpoint,
    _git_value,
    _load_checkpoint,
    _model_parameter_count,
    _optimizer,
    _prepare_run_dir,
    _save_continuous_bundle,
    _save_token_bundle,
    _signature,
)
from src.training.run_continuous_forecaster import _loss_values
from src.training.run_dynamic_graph import (
    GraphArtifactAccumulator,
    _autocast_context,
    _load_raw_training_split,
    _move_optimizer_state,
    _new_grad_scaler,
    atomic_csv_save,
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    generate_validation_artifacts,
    graph_summary,
    move_context_statistics,
    move_training_batch,
    resolve_device,
    restore_rng_state,
    set_seed,
    synchronise_device,
)
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


RUNNER_VERSION = 1
CONTEXT_LENGTH = 60
NUM_NODES = 93
INPUT_CHANNELS = ("open", "high", "low", "close", "volume")
ONE_MINUTE_HORIZONS = (1,)
MULTI_HORIZONS = (1, 5, 15, 30, 60)
ROLLOUT_LENGTH = 60
S1_VOCABULARY_SIZE = 1024
GRAPH_ORIENTATION = "row=target,column=source"
LAYER_ACTIVATIONS = ("softmax", "softmax", "softmax", "sparsemax")


@dataclass(frozen=True)
class SparsemaxDiagnosticSpec:
    name: str
    run_name: str
    label: str
    mode: str
    forecast_strategy: str
    config: BaseDyGraphFinancialConfig
    selection_metric: str
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_name": self.run_name,
            "label": self.label,
            "mode": self.mode,
            "forecast_strategy": self.forecast_strategy,
            "config": self.config.to_dict(),
            "selection_metric": self.selection_metric,
            "tags": list(self.tags),
        }


def _regularisation(weight: float) -> BaseDyGraphGraphRegularisationConfig:
    return BaseDyGraphGraphRegularisationConfig(
        target_entropy=3.0,
        target_entropy_weight=float(weight),
        temporal_smooth_weight=0.01,
        direct_entropy_weight=0.0,
        warmup_epochs=5,
        layer=-1,
    )


def make_experiment_specs() -> tuple[SparsemaxDiagnosticSpec, ...]:
    """Return phase-1 first, then phase-2, in the requested run order."""
    shared: dict[str, Any] = {
        "context_length": CONTEXT_LENGTH,
        "num_nodes": NUM_NODES,
        "input_channels": len(INPUT_CHANNELS),
        "d_model": 96,
        "temporal_heads": 4,
        "temporal_layers": 1,
        "spatial_layers": 1,
        "ff_mult": 2,
        "graph_heads": 1,
        "graph_hidden_dim": 64,
        "num_st_blocks": 4,
        "dropout": 0.0,
        "spatial_dropout": 0.0,
        "use_node_embedding": True,
        "use_state_pair_bias": False,
        "add_self_loops": False,
        "symmetric_graph": False,
        "graph_activation": "softmax",
        "graph_activations": LAYER_ACTIVATIONS,
        "spatial_value": "hidden",
        "st_block_post_norm": True,
        "future_predictor_layers": 0,
        "future_predictor_heads": 4,
        "future_predictor_ff_mult": 2,
        "graph_type": "dynamic_graph",
        "graph_scope": "per_timestep",
    }

    # Run 2 is intentionally first because it is much faster to inspect.
    specs = (
        SparsemaxDiagnosticSpec(
            name="continuous_one_minute",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_1m_dynamic_d96_st4_g1_"
                "final_sparsemax_h3_lam1_smooth0p01"
            ),
            label=(
                "Continuous BaseDyGraph, 1-minute direct forecast, "
                "four ST blocks, one graph head, final sparsemax"
            ),
            mode="continuous",
            forecast_strategy="direct_one_step",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                prediction_length=1,
                evaluation_horizons=ONE_MINUTE_HORIZONS,
                regularisation=_regularisation(1.0),
                **shared,
            ),
            selection_metric="test_1m_cumulative_log_change_mae",
            tags=("basedygraph", "continuous", "one-minute", "sparsemax"),
        ),
        SparsemaxDiagnosticSpec(
            name="token_teacher_forced_one_minute",
            run_name=(
                "DO_NOT_REPORT_basedygraph_token_1m_teacher_forced_d96_st4_g1_"
                "final_sparsemax_h3_lam0p05_smooth0p01"
            ),
            label=(
                "Token BaseDyGraph, teacher-forced next-s1 objective, "
                "four ST blocks, one graph head, final sparsemax"
            ),
            mode="token",
            forecast_strategy="teacher_forced_one_step",
            config=BaseDyGraphFinancialConfig(
                mode="token",
                prediction_length=1,
                evaluation_horizons=ONE_MINUTE_HORIZONS,
                regularisation=_regularisation(0.05),
                **shared,
            ),
            selection_metric="test_teacher_forced_next_s1_cross_entropy",
            tags=("basedygraph", "token", "teacher-forced", "sparsemax"),
        ),
        SparsemaxDiagnosticSpec(
            name="continuous_parallel_sixty_minute",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_60m_parallel_d96_st4_g1_"
                "final_sparsemax_h3_lam1_smooth0p01"
            ),
            label=(
                "Continuous BaseDyGraph, parallel five-horizon forecast, "
                "four ST blocks, one graph head, final sparsemax"
            ),
            mode="continuous",
            forecast_strategy="parallel",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                prediction_length=60,
                evaluation_horizons=MULTI_HORIZONS,
                regularisation=_regularisation(1.0),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=("basedygraph", "continuous", "parallel", "sparsemax"),
        ),
        SparsemaxDiagnosticSpec(
            name="continuous_autoregressive_sixty_minute",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_60m_autoregressive_d96_"
                "st4_g1_final_sparsemax_h3_lam1_smooth0p01"
            ),
            label=(
                "Continuous BaseDyGraph, 60-step autoregressive rollout, "
                "four ST blocks, one graph head, final sparsemax"
            ),
            mode="continuous",
            forecast_strategy="autoregressive",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                prediction_length=1,
                evaluation_horizons=ONE_MINUTE_HORIZONS,
                regularisation=_regularisation(1.0),
                **shared,
            ),
            selection_metric="test_autoregressive_five_horizon_mean_log_mae",
            tags=("basedygraph", "continuous", "autoregressive", "sparsemax"),
        ),
    )
    for spec in specs:
        spec.config.validate()
        if spec.config.resolved_graph_activations != LAYER_ACTIVATIONS:
            raise AssertionError("Layer-specific activation contract changed.")
        if spec.config.num_st_blocks != 4 or spec.config.graph_heads != 1:
            raise AssertionError("Dimitri-matched capacity contract changed.")
    return specs


EXPERIMENT_SPECS = make_experiment_specs()
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
PHASE1_EXPERIMENTS = (
    "continuous_one_minute",
    "token_teacher_forced_one_minute",
)
PHASE2_EXPERIMENTS = (
    "continuous_parallel_sixty_minute",
    "continuous_autoregressive_sixty_minute",
)


class OneStepTokenDataset(Dataset[dict[str, Any]]):
    """A causal one-step view over the existing 60-step token cache."""

    def __init__(self, base: CachedTokenGraphDataset) -> None:
        if base.data_mode != "real":
            raise ValueError("OneStepTokenDataset requires a real-data cache.")
        if base.context_length != CONTEXT_LENGTH or base.prediction_length < 1:
            raise ValueError("Unexpected underlying token-cache geometry.")
        self.base = base
        self.data_mode = base.data_mode

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base[index])
        item["target_s1"] = torch.as_tensor(item["target_s1"])[:1].contiguous()
        item["target_s2"] = torch.as_tensor(item["target_s2"])[:1].contiguous()
        if "target_indices" in item:
            item["target_indices"] = torch.as_tensor(item["target_indices"])[:1].contiguous()
        if "evaluation_true" in item:
            item["evaluation_true"] = torch.as_tensor(item["evaluation_true"])[:1].contiguous()
        return item

    @property
    def context_length(self) -> int:
        return self.base.context_length

    @property
    def prediction_length(self) -> int:
        return 1

    @property
    def num_assets(self) -> int:
        return self.base.num_assets

    @property
    def asset_cols(self) -> tuple[str, ...]:
        return self.base.asset_cols

    @property
    def evaluation_horizons(self) -> tuple[int, ...]:
        return (1,)

    @property
    def evaluation_indices(self) -> tuple[int, ...]:
        return (0,)

    @property
    def s1_id_space(self) -> str:
        return self.base.s1_id_space

    @property
    def s1_vocabulary_size(self) -> int:
        return self.base.s1_vocabulary_size

    @property
    def has_raw_evaluation_targets(self) -> bool:
        return self.base.has_raw_evaluation_targets

    def s1_to_kronos_ids(self, values: Tensor) -> Tensor:
        return self.base.s1_to_kronos_ids(values)


# ---------------------------------------------------------------------------
# Generic runtime helpers
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one test-selected BaseDyGraph sparsemax diagnostic."
    )
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_BY_NAME), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forecasting-config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--validation-cache", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--selection-batch-size", type=int, default=2)
    parser.add_argument("--export-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-series-batch-size", type=int, default=64)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-selection-windows", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("disabled", "online", "offline"), default="disabled")
    parser.add_argument("--wandb-project", type=str, default="dynamic-graph-financial-forecasting-TEST-CONTAMINATED")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_value(arguments: Sequence[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _limit_dataset(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None or int(limit) >= len(dataset):
        return dataset
    if int(limit) <= 0:
        raise ValueError("Window limits must be positive.")
    return Subset(dataset, range(int(limit)))


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(int(seed))
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


def _graph_diagnostics(
    graph_sequences: tuple[Tensor | None, ...],
) -> tuple[float | None, float | None, float | None, float | None]:
    selected = next((value for value in reversed(graph_sequences) if value is not None), None)
    if selected is None:
        return None, None, None, None
    graph = torch.as_tensor(selected).detach().float()
    safe = graph.clamp_min(1.0e-12)
    entropy = -(graph * safe.log()).sum(dim=-1)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
    zero_fraction = (graph == 0).to(torch.float32).mean()
    return (
        float(entropy.mean().item()),
        float(entropy.exp().mean().item()),
        float(diagonal.mean().item()),
        float(zero_fraction.item()),
    )


def _mean(sum_value: float, count: int) -> float:
    if count <= 0:
        raise RuntimeError("No values were accumulated.")
    return float(sum_value) / float(count)


def _batch_for_graph(batch: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(batch)
    if "date" not in values and "day" in values:
        values["date"] = values["day"]
    return values


# ---------------------------------------------------------------------------
# Token objective and post-selection inference
# ---------------------------------------------------------------------------


def _run_token_epoch(
    *,
    model: OfficialBaseDyGraphTeacherForcedOneStepForecaster,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    use_amp: bool,
    epoch: int,
    training: bool,
    gradient_clip_norm: float,
    description: str,
) -> dict[str, Any]:
    model.train(training)
    context_manager = nullcontext() if training else torch.inference_mode()
    ce_sum = objective_sum = regularisation_sum = 0.0
    target_penalty_sum = smooth_penalty_sum = 0.0
    correct = token_count = examples = 0
    entropy_sum = effective_sum = diagonal_sum = zero_fraction_sum = 0.0
    graph_batches = 0
    start = perf_counter()

    with context_manager:
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            context_tokens, target_s1, _target_s2 = move_training_batch(batch, device=device)
            if training:
                assert optimizer is not None and scaler is not None
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, use_amp):
                output: BaseDyGraphTeacherForcedTokenOutput = model(
                    context_tokens,
                    target_s1=target_s1,
                )
                teacher_targets = model.teacher_targets(context_tokens, target_s1)
                ce = F.cross_entropy(
                    output.s1_logits.reshape(-1, S1_VOCABULARY_SIZE),
                    teacher_targets.reshape(-1),
                )
                regularisation = graph_regularisation_loss(
                    output.graph_sequences,
                    model.financial_config.regularisation,
                    epoch=epoch,
                )
                objective = ce + regularisation.total
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite token objective.")
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            count = int(teacher_targets.numel())
            batch_size = int(teacher_targets.shape[0])
            ce_sum += float(ce.detach().item()) * count
            objective_sum += float(objective.detach().item()) * count
            regularisation_sum += float(regularisation.total.detach().item()) * count
            target_penalty_sum += float(regularisation.target_entropy_penalty.detach().item()) * count
            smooth_penalty_sum += float(regularisation.temporal_smoothness_penalty.detach().item()) * count
            predicted = output.s1_logits.detach().argmax(dim=-1)
            correct += int((predicted == teacher_targets).sum().item())
            token_count += count
            examples += batch_size
            entropy, effective, diagonal, zero_fraction = _graph_diagnostics(
                output.graph_sequences
            )
            if entropy is not None:
                entropy_sum += entropy
                effective_sum += float(effective)
                diagonal_sum += float(diagonal)
                zero_fraction_sum += float(zero_fraction)
                graph_batches += 1

    return {
        "token_loss": _mean(ce_sum, token_count),
        "objective_loss": _mean(objective_sum, token_count),
        "graph_regularisation_loss": _mean(regularisation_sum, token_count),
        "target_entropy_penalty": _mean(target_penalty_sum, token_count),
        "temporal_smoothness_penalty": _mean(smooth_penalty_sum, token_count),
        "s1_accuracy": float(correct) / float(max(token_count, 1)),
        "examples": examples,
        "tokens": token_count,
        "graph_mean_row_entropy": entropy_sum / graph_batches if graph_batches else None,
        "graph_mean_effective_neighbours": effective_sum / graph_batches if graph_batches else None,
        "graph_mean_diagonal_weight": diagonal_sum / graph_batches if graph_batches else None,
        "graph_zero_fraction": zero_fraction_sum / graph_batches if graph_batches else None,
        "seconds": perf_counter() - start,
    }


def _token_postselection_exports(
    *,
    model: OfficialBaseDyGraphTeacherForcedOneStepForecaster,
    run_dir: Path,
    epoch: int,
    datasets: Mapping[str, OneStepTokenDataset],
    loaders: Mapping[str, DataLoader[Any]],
    device: torch.device,
    use_amp: bool,
    forecasting_config: Mapping[str, Any],
    raw_train_split: Mapping[str, Any],
    decode_series_batch_size: int,
) -> dict[str, float]:
    tokenizer = KronosTokenizerAdapter.from_config(
        dict(forecasting_config),
        series_batch_size=decode_series_batch_size,
    ).load()
    scores: dict[str, float] = {}
    for split in ("train", "test", "validation"):
        sampled = split == "validation"
        policy = "temperature_1" if sampled else "argmax"
        bundle = generate_validation_artifacts(
            model=model,
            loader=loaders[split],
            dataset=datasets[split],
            device=device,
            use_amp=use_amp,
            decoding_config={
                "token_selection": "sample" if sampled else "argmax",
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 0.9 if sampled else 1.0,
                "sample_count": 10 if sampled else 1,
            },
            tokenizer=tokenizer,
            raw_train_split=raw_train_split,
            decode_series_batch_size=decode_series_batch_size,
            early_stopping_horizons=(1,),
        )
        _save_token_bundle(
            run_dir=run_dir,
            split=split,
            policy=policy,
            epoch=epoch,
            bundle=bundle,
        )
        if bundle.primary_score is not None:
            scores[split] = float(bundle.primary_score)
    return scores


# ---------------------------------------------------------------------------
# Continuous objectives and evaluation
# ---------------------------------------------------------------------------


def _continuous_dataset_config(horizons: tuple[int, ...]) -> ContinuousDatasetConfig:
    config = ContinuousDatasetConfig(
        context_length=CONTEXT_LENGTH,
        horizons=horizons,
        stride=15,
        input_channels=INPUT_CHANNELS,
        target_channels=("close",),
        input_representation="raw",
        eps=1.0e-8,
        clip=False,
        clip_min=-5.0,
        clip_max=5.0,
    )
    config.validate()
    return config


def _run_continuous_train_epoch(
    *,
    model: OfficialBaseDyGraphContinuousForecaster,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    use_amp: bool,
    epoch: int,
    gradient_clip_norm: float,
    description: str,
) -> dict[str, Any]:
    model.train()
    native_sum = objective_sum = regularisation_sum = 0.0
    target_penalty_sum = smooth_penalty_sum = 0.0
    target_count = 0
    entropy_sum = effective_sum = diagonal_sum = zero_fraction_sum = 0.0
    graph_batches = 0
    start = perf_counter()

    for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        x = torch.as_tensor(batch["x"]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model(x)
            forecast_optimisation, native_loss = _loss_values(
                output.predictions,
                batch,
                device=device,
                output_representation="normalised_close",
                loss_type="cumulative_log_change_mae",
                bps_scale=10000.0,
                eps=1.0e-8,
            )
            regularisation = graph_regularisation_loss(
                output.graph_sequences,
                model.config.regularisation,
                epoch=epoch,
            )
            objective = forecast_optimisation + regularisation.total
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite continuous objective.")
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        count = int(torch.as_tensor(batch["y"]).numel())
        native_sum += float(native_loss.detach().item()) * count
        objective_sum += float(objective.detach().item()) * count
        regularisation_sum += float(regularisation.total.detach().item()) * count
        target_penalty_sum += float(regularisation.target_entropy_penalty.detach().item()) * count
        smooth_penalty_sum += float(regularisation.temporal_smoothness_penalty.detach().item()) * count
        target_count += count
        entropy, effective, diagonal, zero_fraction = _graph_diagnostics(
            output.graph_sequences
        )
        if entropy is not None:
            entropy_sum += entropy
            effective_sum += float(effective)
            diagonal_sum += float(diagonal)
            zero_fraction_sum += float(zero_fraction)
            graph_batches += 1

    return {
        "forecast_native_loss": _mean(native_sum, target_count),
        "objective_loss": _mean(objective_sum, target_count),
        "graph_regularisation_loss": _mean(regularisation_sum, target_count),
        "target_entropy_penalty": _mean(target_penalty_sum, target_count),
        "temporal_smoothness_penalty": _mean(smooth_penalty_sum, target_count),
        "graph_mean_row_entropy": entropy_sum / graph_batches if graph_batches else None,
        "graph_mean_effective_neighbours": effective_sum / graph_batches if graph_batches else None,
        "graph_mean_diagonal_weight": diagonal_sum / graph_batches if graph_batches else None,
        "graph_zero_fraction": zero_fraction_sum / graph_batches if graph_batches else None,
        "seconds": perf_counter() - start,
    }


def _inverse_prediction(
    predictions: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tensor:
    mean = torch.as_tensor(batch["target_norm_mean"]).to(device=device, dtype=torch.float32)
    std = torch.as_tensor(batch["target_norm_std"]).to(device=device, dtype=torch.float32)
    return inverse_window_normalisation(
        y_norm=predictions.float(),
        target_norm_mean=mean,
        target_norm_std=std,
    )


def _finalise_continuous_bundle(
    *,
    predictions: list[Tensor],
    targets: list[Tensor],
    last_values: list[Tensor],
    sample_indices: list[Tensor],
    origin_indices: list[Tensor],
    target_indices: list[Tensor],
    horizons: tuple[int, ...],
    asset_cols: Sequence[str],
    train_split: Mapping[str, Any],
    graphs: Mapping[str, Any] | None,
    graph_diagnostics: Mapping[str, Any],
    seconds: float,
) -> dict[str, Any]:
    prediction_result = {
        "y_pred": torch.cat(predictions, dim=0).contiguous(),
        "y_true": torch.cat(targets, dim=0).contiguous(),
        "last_context_target": torch.cat(last_values, dim=0).contiguous(),
        "sample_idx": torch.cat(sample_indices, dim=0).contiguous(),
        "origin_idx": torch.cat(origin_indices, dim=0).contiguous(),
        "target_indices": torch.cat(target_indices, dim=0).contiguous(),
        "channels": ["close"],
        "horizons": list(horizons),
        "asset_cols": list(asset_cols),
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
    score = float(metric_results["cumulative_log_change_mae"].mean().item())
    return {
        "native_loss": score,
        "selection_score": score,
        "prediction_result": prediction_result,
        "metric_results": metric_results,
        "metric_table": metric_table,
        "graphs": graphs,
        "graph_summary": dict(graph_diagnostics),
        "seconds": float(seconds),
    }


def _evaluate_continuous_direct(
    *,
    model: OfficialBaseDyGraphContinuousForecaster,
    loader: DataLoader[Any],
    device: torch.device,
    use_amp: bool,
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    horizons: tuple[int, ...],
    description: str,
    retain_graphs: bool = True,
) -> dict[str, Any]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    graph_accumulator = (
        GraphArtifactAccumulator(
            asset_cols=asset_cols,
            graph_type="dynamic",
            num_layers=model.config.num_st_blocks,
            num_heads=model.config.graph_heads,
        )
        if retain_graphs
        else None
    )
    entropy_sum = effective_sum = diagonal_sum = zero_fraction_sum = 0.0
    graph_batches = 0
    start = perf_counter()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(device=device, dtype=torch.float32)
            with _autocast_context(device, use_amp):
                output = model(x)
            raw_prediction = _inverse_prediction(output.predictions, batch, device=device)
            predictions.append(raw_prediction.detach().cpu().contiguous())
            targets.append(torch.as_tensor(batch["y_unnormalised"]).float().cpu())
            last_values.append(torch.as_tensor(batch["last_context_target"]).float().cpu())
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).long().cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).long().cpu())
            target_indices.append(torch.as_tensor(batch["target_indices"]).long().cpu())
            entropy, effective, diagonal, zero_fraction = _graph_diagnostics(
                output.graph_sequences
            )
            if entropy is not None:
                entropy_sum += entropy
                effective_sum += float(effective)
                diagonal_sum += float(diagonal)
                zero_fraction_sum += float(zero_fraction)
                graph_batches += 1
            if graph_accumulator is not None:
                graph_accumulator.add(
                    output.graph,
                    _batch_for_graph(batch),
                    batch_size=int(raw_prediction.shape[0]),
                    spatial_beta=None,
                )

    if graph_accumulator is not None:
        graphs = graph_accumulator.finalise()
        graphs.update(financial_graph_artifact_metadata(model.config))
        graph_diagnostics = graph_summary(graphs)
        graph_diagnostics["mean_zero_fraction"] = (
            zero_fraction_sum / graph_batches if graph_batches else None
        )
    else:
        graphs = None
        graph_diagnostics = {
            "graph_present": graph_batches > 0,
            "mean_row_entropy": entropy_sum / graph_batches if graph_batches else None,
            "mean_effective_neighbours": effective_sum / graph_batches if graph_batches else None,
            "mean_diagonal_weight": diagonal_sum / graph_batches if graph_batches else None,
            "mean_zero_fraction": zero_fraction_sum / graph_batches if graph_batches else None,
        }
    return _finalise_continuous_bundle(
        predictions=predictions,
        targets=targets,
        last_values=last_values,
        sample_indices=sample_indices,
        origin_indices=origin_indices,
        target_indices=target_indices,
        horizons=horizons,
        asset_cols=asset_cols,
        train_split=train_split,
        graphs=graphs,
        graph_diagnostics=graph_diagnostics,
        seconds=perf_counter() - start,
    )


def _normalise_raw_context(raw_context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean = raw_context.mean(dim=1)
    std = raw_context.std(dim=1, unbiased=False).clamp_min(1.0e-8)
    normalised = (raw_context - mean.unsqueeze(1)) / std.unsqueeze(1)
    return normalised, mean, std


def _append_synthetic_candle(raw_context: Tensor, next_close: Tensor) -> Tensor:
    """Shift one step using only the model Close and previous observed state."""
    if raw_context.ndim != 4 or int(raw_context.shape[-1]) != 5:
        raise ValueError("raw_context must have shape [B,T,N,5].")
    if next_close.ndim != 2:
        raise ValueError("next_close must have shape [B,N].")
    previous_close = raw_context[:, -1, :, 3]
    previous_volume = raw_context[:, -1, :, 4]
    next_open = previous_close
    next_high = torch.maximum(next_open, next_close)
    next_low = torch.minimum(next_open, next_close)
    candle = torch.stack(
        (next_open, next_high, next_low, next_close, previous_volume),
        dim=-1,
    )
    return torch.cat((raw_context[:, 1:], candle.unsqueeze(1)), dim=1).contiguous()


def _evaluate_continuous_autoregressive(
    *,
    model: OfficialBaseDyGraphContinuousForecaster,
    loader: DataLoader[Any],
    device: torch.device,
    use_amp: bool,
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    description: str,
    retain_graphs: bool = True,
) -> dict[str, Any]:
    model.eval()
    evaluation_indices = torch.tensor([0, 4, 14, 29, 59], device=device)
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    graph_accumulator = (
        GraphArtifactAccumulator(
            asset_cols=asset_cols,
            graph_type="dynamic",
            num_layers=model.config.num_st_blocks,
            num_heads=model.config.graph_heads,
        )
        if retain_graphs
        else None
    )
    entropy_sum = effective_sum = diagonal_sum = zero_fraction_sum = 0.0
    graph_batches = 0
    start = perf_counter()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            raw_context = torch.as_tensor(batch["context_unnormalised"]).to(
                device=device,
                dtype=torch.float32,
            )
            dense_close: list[Tensor] = []
            initial_output = None
            for step in range(ROLLOUT_LENGTH):
                x_normalised, mean, std = _normalise_raw_context(raw_context)
                with _autocast_context(device, use_amp):
                    output = model(x_normalised)
                if tuple(output.predictions.shape[1:]) != (1, NUM_NODES, 1):
                    raise RuntimeError("Autoregressive model must emit one Close step.")
                next_close = (
                    output.predictions[:, 0, :, 0].float() * std[:, :, 3]
                    + mean[:, :, 3]
                )
                if not torch.isfinite(next_close).all() or (next_close <= 0).any():
                    raise FloatingPointError(
                        f"Invalid autoregressive Close prediction at rollout step {step + 1}."
                    )
                dense_close.append(next_close.unsqueeze(-1))
                if initial_output is None:
                    initial_output = output
                raw_context = _append_synthetic_candle(raw_context, next_close)

            dense = torch.stack(dense_close, dim=1)
            evaluation = dense.index_select(dim=1, index=evaluation_indices)
            predictions.append(evaluation.detach().cpu().contiguous())
            targets.append(torch.as_tensor(batch["y_unnormalised"]).float().cpu())
            last_values.append(torch.as_tensor(batch["last_context_target"]).float().cpu())
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).long().cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).long().cpu())
            target_indices.append(torch.as_tensor(batch["target_indices"]).long().cpu())

            if initial_output is None:
                raise RuntimeError("Autoregressive rollout produced no initial output.")
            entropy, effective, diagonal, zero_fraction = _graph_diagnostics(
                initial_output.graph_sequences
            )
            if entropy is not None:
                entropy_sum += entropy
                effective_sum += float(effective)
                diagonal_sum += float(diagonal)
                zero_fraction_sum += float(zero_fraction)
                graph_batches += 1
            if graph_accumulator is not None:
                graph_accumulator.add(
                    initial_output.graph,
                    _batch_for_graph(batch),
                    batch_size=int(evaluation.shape[0]),
                    spatial_beta=None,
                )

    if graph_accumulator is not None:
        graphs = graph_accumulator.finalise()
        graphs.update(financial_graph_artifact_metadata(model.config))
        graphs["forecast_rollout"] = "60-step causal synthetic-OHLCV autoregressive"
        graph_diagnostics = graph_summary(graphs)
        graph_diagnostics["mean_zero_fraction"] = (
            zero_fraction_sum / graph_batches if graph_batches else None
        )
    else:
        graphs = None
        graph_diagnostics = {
            "graph_present": graph_batches > 0,
            "mean_row_entropy": entropy_sum / graph_batches if graph_batches else None,
            "mean_effective_neighbours": effective_sum / graph_batches if graph_batches else None,
            "mean_diagonal_weight": diagonal_sum / graph_batches if graph_batches else None,
            "mean_zero_fraction": zero_fraction_sum / graph_batches if graph_batches else None,
        }
    return _finalise_continuous_bundle(
        predictions=predictions,
        targets=targets,
        last_values=last_values,
        sample_indices=sample_indices,
        origin_indices=origin_indices,
        target_indices=target_indices,
        horizons=MULTI_HORIZONS,
        asset_cols=asset_cols,
        train_split=train_split,
        graphs=graphs,
        graph_diagnostics=graph_diagnostics,
        seconds=perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# Metadata and training orchestration
# ---------------------------------------------------------------------------


def _resolved_config_payload(
    *,
    spec: SparsemaxDiagnosticSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_horizons = list(spec.config.evaluation_horizons)
    reported_horizons = (
        list(MULTI_HORIZONS)
        if spec.forecast_strategy == "autoregressive"
        else model_horizons
    )
    payload: dict[str, Any] = {
        "runner": "src.training.run_basedygraph_sparsemax_diagnostic",
        "runner_version": RUNNER_VERSION,
        "model_family": "official_basedygraph_financial",
        "experiment": spec.name,
        "run_name": args.run_name or spec.run_name,
        "selection_split": "test",
        "test_set_contaminated": True,
        "do_not_report": True,
        "forecast_strategy": spec.forecast_strategy,
        "basedygraph_financial": spec.config.to_dict(),
        "data": {
            "context_length": CONTEXT_LENGTH,
            "model_prediction_length": spec.config.prediction_length,
            "reported_horizons": reported_horizons,
            "rollout_length": ROLLOUT_LENGTH if spec.forecast_strategy == "autoregressive" else None,
            "stride": 15,
            "input_channels": list(INPUT_CHANNELS),
            "target_channel": "close",
            "data_dir": str(args.data_dir),
            "train_cache": None if args.train_cache is None else str(args.train_cache),
            "validation_cache": None if args.validation_cache is None else str(args.validation_cache),
            "test_cache": None if args.test_cache is None else str(args.test_cache),
        },
        "training": {
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "train_batch_size": int(args.train_batch_size),
            "selection_batch_size": int(args.selection_batch_size),
            "export_batch_size": int(args.export_batch_size),
            "num_workers": int(args.num_workers),
            "mixed_precision": bool(args.mixed_precision),
            "seed": int(args.seed),
            "optimizer": "adamw",
            "scheduler": "none",
            "selection_metric": spec.selection_metric,
        },
        "graph": {
            "type": "dynamic",
            "scope": "per_timestep",
            "num_heads": 1,
            "hidden_dim": 64,
            "activations_by_layer": list(LAYER_ACTIVATIONS),
            "regularisation": asdict(spec.config.regularisation),
            "orientation": GRAPH_ORIENTATION,
        },
    }
    if spec.mode == "token":
        payload["models"] = {
            "dynamic_graph": {
                "num_nodes": NUM_NODES,
                "context_length": CONTEXT_LENGTH,
                "d_model": 96,
                "num_st_blocks": 4,
                "graph": {
                    "type": "dynamic",
                    "num_heads": 1,
                    "hidden_dim": 64,
                    "activation": "sparsemax",
                    "activations_by_layer": list(LAYER_ACTIVATIONS),
                    "add_self_loops": False,
                },
                "heads": {
                    "prediction_length": 1,
                    "evaluation_horizons": [1],
                    "future_token_mode": "coarse_only",
                    "s1_vocabulary_size": 1024,
                    "s2_loss_weight": 0.0,
                },
                "future_predictor": {
                    "type": "official_direct_next_state_head",
                    "num_layers": 0,
                },
            }
        }
        payload["sampling"] = {
            "validation_temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "sample_count": 10,
        }
    else:
        payload["model"] = {
            "output_representation": "normalised_close",
            "num_st_blocks": 4,
            "graph": {
                "type": "dynamic",
                "num_heads": 1,
                "hidden_dim": 64,
                "activation": "sparsemax",
                "activations_by_layer": list(LAYER_ACTIVATIONS),
                "add_self_loops": False,
            },
            "temporal": {
                "type": "official_basedygraph_transformer",
                "d_model": 96,
                "num_layers": 1,
                "num_heads": 4,
            },
            "forecast_strategy": spec.forecast_strategy,
        }
    return payload


def _init_wandb(args: argparse.Namespace, spec: SparsemaxDiagnosticSpec) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name or spec.run_name,
        tags=[*spec.tags, *args.wandb_tags, "DO-NOT-REPORT"],
        config=_resolved_config_payload(spec=spec, args=args),
    )


def _token_history_record(
    *,
    epoch: int,
    train: Mapping[str, Any],
    test: Mapping[str, Any],
    best_score: float,
    best_epoch: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "training_token_loss": float(train["token_loss"]),
        "training_objective_loss": float(train["objective_loss"]),
        "training_graph_regularisation_loss": float(train["graph_regularisation_loss"]),
        "training_graph_target_entropy_penalty": float(train["target_entropy_penalty"]),
        "training_graph_temporal_smooth_penalty": float(train["temporal_smoothness_penalty"]),
        "training_s1_accuracy": float(train["s1_accuracy"]),
        "training_graph_mean_row_entropy": train["graph_mean_row_entropy"],
        "training_graph_mean_effective_neighbours": train["graph_mean_effective_neighbours"],
        "training_graph_zero_fraction": train["graph_zero_fraction"],
        "test_token_loss": float(test["token_loss"]),
        "test_s1_accuracy": float(test["s1_accuracy"]),
        "test_graph_mean_row_entropy": test["graph_mean_row_entropy"],
        "test_graph_mean_effective_neighbours": test["graph_mean_effective_neighbours"],
        "test_graph_zero_fraction": test["graph_zero_fraction"],
        "validation_token_loss": float(test["token_loss"]),
        "validation_s1_accuracy": float(test["s1_accuracy"]),
        "selection_score": float(test["token_loss"]),
        "best_score_after_epoch": float(best_score),
        "best_epoch_after_epoch": int(best_epoch),
        "selection_split": "test",
        "train_seconds": float(train["seconds"]),
        "test_selection_seconds": float(test["seconds"]),
    }


def _continuous_history_record(
    *,
    epoch: int,
    train: Mapping[str, Any],
    test: Mapping[str, Any],
    best_score: float,
    best_epoch: int,
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "training_native_log_mae": float(train["forecast_native_loss"]),
        "training_objective_loss": float(train["objective_loss"]),
        "training_graph_regularisation_loss": float(train["graph_regularisation_loss"]),
        "training_graph_target_entropy_penalty": float(train["target_entropy_penalty"]),
        "training_graph_temporal_smooth_penalty": float(train["temporal_smoothness_penalty"]),
        "training_graph_mean_row_entropy": train["graph_mean_row_entropy"],
        "training_graph_mean_effective_neighbours": train["graph_mean_effective_neighbours"],
        "training_graph_zero_fraction": train["graph_zero_fraction"],
        "test_mean_log_mae": float(test["selection_score"]),
        "test_graph_mean_row_entropy": test["graph_summary"].get("mean_row_entropy"),
        "test_graph_mean_effective_neighbours": test["graph_summary"].get("mean_effective_neighbours"),
        "test_graph_zero_fraction": test["graph_summary"].get("mean_zero_fraction"),
        "selection_score": float(test["selection_score"]),
        "validation_loss": float(test["selection_score"]),
        "best_score_after_epoch": float(best_score),
        "best_epoch_after_epoch": int(best_epoch),
        "selection_split": "test",
        "train_seconds": float(train["seconds"]),
        "test_selection_seconds": float(test["seconds"]),
    }
    metric = test["metric_results"]["cumulative_log_change_mae"].reshape(-1)
    for horizon, value in zip(horizons, metric, strict=True):
        record[f"test_cumulative_log_change_mae_h{horizon}"] = float(value.item())
        record[f"val_cumulative_log_change_mae_h{horizon}"] = float(value.item())
    return record


def _run_token_experiment(
    *,
    spec: SparsemaxDiagnosticSpec,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    run_signature: str,
    metadata: dict[str, Any],
    forecasting_config: Mapping[str, Any],
) -> None:
    for name, path in (("train", args.train_cache), ("validation", args.validation_cache), ("test", args.test_cache)):
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} token cache: {path}")

    base_datasets = {
        "train": CachedTokenGraphDataset.from_path(args.train_cache),
        "validation": CachedTokenGraphDataset.from_path(args.validation_cache),
        "test": CachedTokenGraphDataset.from_path(args.test_cache),
    }
    datasets = {name: OneStepTokenDataset(value) for name, value in base_datasets.items()}
    if not (datasets["train"].asset_cols == datasets["validation"].asset_cols == datasets["test"].asset_cols):
        raise ValueError("Token-cache asset orders differ.")

    train_dataset = _limit_dataset(datasets["train"], args.max_train_windows)
    test_dataset = _limit_dataset(datasets["test"], args.max_selection_windows)
    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = OfficialBaseDyGraphTeacherForcedOneStepForecaster(spec.config).to(device)
    total_parameters, trainable_parameters = _model_parameter_count(model)
    optimizer = _optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scaler = _new_grad_scaler(enabled=use_amp)
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
            scaler=scaler,
            device=device,
            expected_signature=run_signature,
            restore_rng=True,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(checkpoint["evaluations_without_improvement"])
        history = [dict(row) for row in checkpoint["history"]]

    metadata.update(
        {
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "asset_cols": list(datasets["train"].asset_cols),
            "train_windows": len(datasets["train"]),
            "test_selection_windows": len(datasets["test"]),
            "validation_windows": len(datasets["validation"]),
            "basedygraph_observed_commit": model.external_commit,
        }
    )
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    wandb_run = _init_wandb(args, spec)
    try:
        for epoch in range(start_epoch, int(args.max_epochs) + 1):
            train_metrics = _run_token_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch,
                training=True,
                gradient_clip_norm=args.gradient_clip_norm,
                description=f"teacher-forced train epoch {epoch}",
            )
            test_metrics = _run_token_epoch(
                model=model,
                loader=test_loader,
                device=device,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                epoch=epoch,
                training=False,
                gradient_clip_norm=args.gradient_clip_norm,
                description=f"TEST teacher-forced selection epoch {epoch}",
            )
            score = float(test_metrics["token_loss"])
            improved = score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                without_improvement = 0
            else:
                without_improvement += 1
            history.append(
                _token_history_record(
                    epoch=epoch,
                    train=train_metrics,
                    test=test_metrics,
                    best_score=best_score,
                    best_epoch=best_epoch,
                )
            )
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                without_improvement=without_improvement,
                history=history,
                run_signature=run_signature,
                spec=spec,
            )
            atomic_torch_save(checkpoint, run_dir / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(checkpoint, run_dir / "best_checkpoint.pt")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/token_ce": train_metrics["token_loss"],
                        "test_selection/token_ce": score,
                        "train/graph_entropy": train_metrics["graph_mean_row_entropy"],
                        "train/graph_zero_fraction": train_metrics["graph_zero_fraction"],
                    },
                    step=epoch,
                )
            if without_improvement >= int(args.patience):
                break

        best_checkpoint = _load_checkpoint(
            run_dir / "best_checkpoint.pt",
            model=model,
            optimizer=None,
            scaler=None,
            device=device,
            expected_signature=run_signature,
            restore_rng=False,
        )
        export_loaders = {
            split: _build_loader(
                dataset,
                batch_size=args.export_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=args.seed,
                pin_memory=pin_memory,
            )
            for split, dataset in datasets.items()
        }
        raw_train_split = _load_raw_training_split(
            args.data_dir,
            expected_asset_cols=datasets["train"].asset_cols,
        )
        export_scores = _token_postselection_exports(
            model=model,
            run_dir=run_dir,
            epoch=int(best_checkpoint["epoch"]),
            datasets=datasets,
            loaders=export_loaders,
            device=device,
            use_amp=use_amp,
            forecasting_config=forecasting_config,
            raw_train_split=raw_train_split,
            decode_series_batch_size=args.decode_series_batch_size,
        )
        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "epochs_completed": int(history[-1]["epoch"]),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_split": "test",
                "postselection_price_scores": export_scores,
                "analysis_splits_saved": ["train", "validation", "test"],
                "validation_sampling": {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "sample_count": 10,
                },
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
    except Exception:
        metadata.update({"status": "failed", "failed_at_utc": _utc_now(), "best_epoch": best_epoch, "best_score": best_score})
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_continuous_experiment(
    *,
    spec: SparsemaxDiagnosticSpec,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    run_signature: str,
    metadata: dict[str, Any],
) -> None:
    train_raw, val_raw, test_raw = load_candle_splits(args.data_dir)
    train_split, validation_split, test_split = clean_candle_splits(train_raw, val_raw, test_raw)

    training_horizons = (
        MULTI_HORIZONS if spec.forecast_strategy == "parallel" else ONE_MINUTE_HORIZONS
    )
    evaluation_horizons = (
        MULTI_HORIZONS if spec.forecast_strategy in {"parallel", "autoregressive"} else ONE_MINUTE_HORIZONS
    )
    train_dataset_full = build_continuous_dataset(
        train_split,
        config=_continuous_dataset_config(training_horizons),
    )
    evaluation_datasets = {
        "train": build_continuous_dataset(train_split, config=_continuous_dataset_config(evaluation_horizons)),
        "validation": build_continuous_dataset(validation_split, config=_continuous_dataset_config(evaluation_horizons)),
        "test": build_continuous_dataset(test_split, config=_continuous_dataset_config(evaluation_horizons)),
    }
    train_dataset = _limit_dataset(train_dataset_full, args.max_train_windows)
    test_selection_dataset = _limit_dataset(evaluation_datasets["test"], args.max_selection_windows)
    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    test_loader = _build_loader(
        test_selection_dataset,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = OfficialBaseDyGraphContinuousForecaster(spec.config).to(device)
    total_parameters, trainable_parameters = _model_parameter_count(model)
    optimizer = _optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    scaler = _new_grad_scaler(enabled=use_amp)
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
            scaler=scaler,
            device=device,
            expected_signature=run_signature,
            restore_rng=True,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        without_improvement = int(checkpoint["evaluations_without_improvement"])
        history = [dict(row) for row in checkpoint["history"]]

    metadata.update(
        {
            "parameter_count": total_parameters,
            "trainable_parameter_count": trainable_parameters,
            "asset_cols": list(train_split["asset_cols"]),
            "train_windows": len(train_dataset_full),
            "test_selection_windows": len(evaluation_datasets["test"]),
            "validation_windows": len(evaluation_datasets["validation"]),
            "basedygraph_observed_commit": model.external_commit,
            "autoregressive_candle_bridge": (
                {
                    "open": "previous close",
                    "close": "model prediction",
                    "high": "max(open, close)",
                    "low": "min(open, close)",
                    "volume": "previous volume",
                    "future_truth_used": False,
                }
                if spec.forecast_strategy == "autoregressive"
                else None
            ),
        }
    )
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    wandb_run = _init_wandb(args, spec)
    try:
        for epoch in range(start_epoch, int(args.max_epochs) + 1):
            train_metrics = _run_continuous_train_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                epoch=epoch,
                gradient_clip_norm=args.gradient_clip_norm,
                description=f"train epoch {epoch}",
            )
            if spec.forecast_strategy == "autoregressive":
                test_bundle = _evaluate_continuous_autoregressive(
                    model=model,
                    loader=test_loader,
                    device=device,
                    use_amp=use_amp,
                    train_split=train_split,
                    asset_cols=train_split["asset_cols"],
                    description=f"TEST autoregressive selection epoch {epoch}",
                    retain_graphs=False,
                )
            else:
                test_bundle = _evaluate_continuous_direct(
                    model=model,
                    loader=test_loader,
                    device=device,
                    use_amp=use_amp,
                    train_split=train_split,
                    asset_cols=train_split["asset_cols"],
                    horizons=evaluation_horizons,
                    description=f"TEST selection epoch {epoch}",
                    retain_graphs=False,
                )
            score = float(test_bundle["selection_score"])
            improved = score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                without_improvement = 0
            else:
                without_improvement += 1
            history.append(
                _continuous_history_record(
                    epoch=epoch,
                    train=train_metrics,
                    test=test_bundle,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    horizons=evaluation_horizons,
                )
            )
            atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
            checkpoint = _checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                without_improvement=without_improvement,
                history=history,
                run_signature=run_signature,
                spec=spec,
            )
            atomic_torch_save(checkpoint, run_dir / "last_checkpoint.pt")
            if improved:
                atomic_torch_save(checkpoint, run_dir / "best_checkpoint.pt")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/log_mae": train_metrics["forecast_native_loss"],
                        "test_selection/mean_log_mae": score,
                        "train/graph_entropy": train_metrics["graph_mean_row_entropy"],
                        "train/graph_zero_fraction": train_metrics["graph_zero_fraction"],
                    },
                    step=epoch,
                )
            if without_improvement >= int(args.patience):
                break

        best_checkpoint = _load_checkpoint(
            run_dir / "best_checkpoint.pt",
            model=model,
            optimizer=None,
            scaler=None,
            device=device,
            expected_signature=run_signature,
            restore_rng=False,
        )
        export_loaders = {
            split: _build_loader(
                dataset,
                batch_size=args.export_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                seed=args.seed,
                pin_memory=pin_memory,
            )
            for split, dataset in evaluation_datasets.items()
        }
        export_scores: dict[str, float] = {}
        for split in ("train", "test", "validation"):
            if spec.forecast_strategy == "autoregressive":
                bundle = _evaluate_continuous_autoregressive(
                    model=model,
                    loader=export_loaders[split],
                    device=device,
                    use_amp=use_amp,
                    train_split=train_split,
                    asset_cols=train_split["asset_cols"],
                    description=f"selected checkpoint {split} autoregressive export",
                    retain_graphs=True,
                )
            else:
                bundle = _evaluate_continuous_direct(
                    model=model,
                    loader=export_loaders[split],
                    device=device,
                    use_amp=use_amp,
                    train_split=train_split,
                    asset_cols=train_split["asset_cols"],
                    horizons=evaluation_horizons,
                    description=f"selected checkpoint {split} export",
                    retain_graphs=True,
                )
            _save_continuous_bundle(
                run_dir=run_dir,
                split=split,
                epoch=int(best_checkpoint["epoch"]),
                bundle=bundle,
            )
            export_scores[split] = float(bundle["selection_score"])
        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "epochs_completed": int(history[-1]["epoch"]),
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_split": "test",
                "postselection_price_scores": export_scores,
                "analysis_splits_saved": ["train", "validation", "test"],
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
    except Exception:
        metadata.update({"status": "failed", "failed_at_utc": _utc_now(), "best_epoch": best_epoch, "best_score": best_score})
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    args = build_argument_parser().parse_args()
    spec = EXPERIMENT_BY_NAME[args.experiment]
    run_name = args.run_name or spec.run_name
    if not run_name.startswith("DO_NOT_REPORT"):
        raise ValueError("The test-selected runner requires a DO_NOT_REPORT name.")
    if args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("max_epochs and patience must be positive.")
    repository_root = Path(__file__).resolve().parents[2]
    project_commit = _git_value(("rev-parse", "HEAD"), repository_root)
    device = resolve_device(args.device)
    use_amp = bool(args.mixed_precision and device.type == "cuda")
    set_seed(args.seed)
    run_dir = _prepare_run_dir(
        args.output_dir,
        run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    resolved = _resolved_config_payload(spec=spec, args=args)
    resolved["run_name"] = run_name
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    signature_payload = {
        "runner_version": RUNNER_VERSION,
        "experiment": spec.name,
        "run_name": run_name,
        "model": spec.config.to_dict(),
        "forecast_strategy": spec.forecast_strategy,
        "training": resolved["training"],
        "data": resolved["data"],
    }
    run_signature = _signature(signature_payload)
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at_utc": _utc_now(),
        "run_name": run_name,
        "experiment": spec.name,
        "label": spec.label,
        "model_family": "official_basedygraph_financial",
        "mode": spec.mode,
        "forecast_strategy": spec.forecast_strategy,
        "selection_split": "test",
        "selection_metric": spec.selection_metric,
        "test_set_contaminated": True,
        "do_not_report": True,
        "run_signature": run_signature,
        "project_commit": project_commit,
        "device": str(device),
        "requested_mixed_precision": bool(args.mixed_precision),
        "active_cuda_amp": use_amp,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "graph_orientation": GRAPH_ORIENTATION,
        "graph_scope": "per_timestep",
        "graph_activations_by_layer": list(LAYER_ACTIVATIONS),
        "graph_regularisation": asdict(spec.config.regularisation),
        "num_st_blocks": 4,
        "num_graph_heads": 1,
        "context_length": CONTEXT_LENGTH,
        "model_prediction_length": spec.config.prediction_length,
        "reported_horizons": (
            list(MULTI_HORIZONS)
            if spec.forecast_strategy == "autoregressive"
            else list(spec.config.evaluation_horizons)
        ),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "warning": (
            "Checkpoint selection used the October-December test split. "
            "This is a contaminated diagnostic and must not be reported as "
            "held-out test performance."
        ),
    }
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(metadata["warning"] + "\n", encoding="utf-8")

    forecasting_config = load_yaml(args.forecasting_config)
    if spec.mode == "token":
        _run_token_experiment(
            spec=spec,
            args=args,
            run_dir=run_dir,
            device=device,
            use_amp=use_amp,
            run_signature=run_signature,
            metadata=metadata,
            forecasting_config=forecasting_config,
        )
    else:
        _run_continuous_experiment(
            spec=spec,
            args=args,
            run_dir=run_dir,
            device=device,
            use_amp=use_amp,
            run_signature=run_signature,
            metadata=metadata,
        )


if __name__ == "__main__":
    main()

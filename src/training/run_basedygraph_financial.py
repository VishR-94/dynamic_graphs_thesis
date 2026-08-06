from __future__ import annotations

"""Run the seven deliberately test-selected BaseDyGraph diagnostics.

This runner preserves the pinned official BaseDyGraph interlaced backbone and
adds only the financial task adapters in ``src.models.basedygraph_financial``.
It is intentionally a curiosity runner: gradients use the chronological
January--August training split, while checkpoint selection and patience use the
October--December test split.  Every run is therefore marked ``do_not_report``
and ``test_set_contaminated`` in its metadata.

After training, the selected checkpoint is evaluated over train, validation and
test.  Standard root artefacts and canonical ``analysis/<split>`` bundles are
saved so the existing Graph Hub can analyse every selected graph.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
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
    BaseDyGraphTokenTrainingOutput,
    OfficialBaseDyGraphCoarsePathForecaster,
    OfficialBaseDyGraphContinuousForecaster,
    financial_graph_artifact_metadata,
    graph_regularisation_loss,
)
from src.models.kronos_tokenizer import KronosTokenizerAdapter
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
    build_loader,
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
GRAPH_ORIENTATION = "row=target,column=source"
INPUT_CHANNELS = ("open", "high", "low", "close", "volume")
HORIZONS = (1, 5, 15, 30, 60)
CONTEXT_LENGTH = 60
PREDICTION_LENGTH = 60
NUM_NODES = 93
S1_VOCABULARY_SIZE = 1024


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    run_name: str
    label: str
    mode: str
    config: BaseDyGraphFinancialConfig
    selection_metric: str
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_name": self.run_name,
            "label": self.label,
            "mode": self.mode,
            "config": self.config.to_dict(),
            "selection_metric": self.selection_metric,
            "tags": list(self.tags),
        }


def _regularisation(
    *,
    target_entropy: float | None,
    target_weight: float,
    smooth_weight: float,
    warmup_epochs: int,
) -> BaseDyGraphGraphRegularisationConfig:
    return BaseDyGraphGraphRegularisationConfig(
        target_entropy=target_entropy,
        target_entropy_weight=target_weight,
        temporal_smooth_weight=smooth_weight,
        direct_entropy_weight=0.0,
        warmup_epochs=warmup_epochs,
        layer=-1,
    )


def make_experiment_specs() -> tuple[ExperimentSpec, ...]:
    """Return the locked seven-run curiosity ladder."""

    shared: dict[str, Any] = {
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "evaluation_horizons": HORIZONS,
        "num_nodes": NUM_NODES,
        "input_channels": len(INPUT_CHANNELS),
        "d_model": 96,
        "temporal_heads": 4,
        "temporal_layers": 1,
        "spatial_layers": 1,
        "ff_mult": 2,
        "graph_heads": 2,
        "graph_hidden_dim": 64,
        "num_st_blocks": 3,
        "dropout": 0.0,
        "spatial_dropout": 0.0,
        "use_node_embedding": True,
        "use_state_pair_bias": False,
        "add_self_loops": False,
        "symmetric_graph": False,
        "graph_activation": "softmax",
        "spatial_value": "hidden",
        "st_block_post_norm": True,
        "future_predictor_layers": 1,
        "future_predictor_heads": 4,
        "future_predictor_ff_mult": 2,
    }

    specs = (
        ExperimentSpec(
            name="token_static_h3",
            run_name="DO_NOT_REPORT_basedygraph_token_static_d96_st3_h3_lam0p05",
            label="Token BaseDyGraph static, H*=3.0",
            mode="token",
            config=BaseDyGraphFinancialConfig(
                mode="token",
                graph_type="static_graph",
                graph_scope="per_timestep",
                regularisation=_regularisation(
                    target_entropy=3.0,
                    target_weight=0.05,
                    smooth_weight=0.0,
                    warmup_epochs=0,
                ),
                **shared,
            ),
            selection_metric="test_coarse_token_cross_entropy",
            tags=("basedygraph", "token", "static", "test-selected"),
        ),
        ExperimentSpec(
            name="token_dynamic_step_eq38",
            run_name=(
                "DO_NOT_REPORT_basedygraph_token_dynamic_step_d96_st3_"
                "h3_lam0p05_smooth0p01"
            ),
            label="Token BaseDyGraph dynamic per-step, Eq. 38",
            mode="token",
            config=BaseDyGraphFinancialConfig(
                mode="token",
                graph_type="dynamic_graph",
                graph_scope="per_timestep",
                regularisation=_regularisation(
                    target_entropy=3.0,
                    target_weight=0.05,
                    smooth_weight=0.01,
                    warmup_epochs=5,
                ),
                **shared,
            ),
            selection_metric="test_coarse_token_cross_entropy",
            tags=(
                "basedygraph",
                "token",
                "dynamic-step",
                "eq38",
                "test-selected",
            ),
        ),
        ExperimentSpec(
            name="continuous_static_h3",
            run_name="DO_NOT_REPORT_basedygraph_continuous_static_d96_st3_h3_lam1",
            label="Continuous BaseDyGraph static, H*=3.0",
            mode="continuous",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                graph_type="static_graph",
                graph_scope="per_timestep",
                regularisation=_regularisation(
                    target_entropy=3.0,
                    target_weight=1.0,
                    smooth_weight=0.0,
                    warmup_epochs=0,
                ),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=("basedygraph", "continuous", "static", "test-selected"),
        ),
        ExperimentSpec(
            name="continuous_dynamic_step_no_reg",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_dynamic_step_"
                "d96_st3_no_reg"
            ),
            label="Continuous BaseDyGraph dynamic per-step, no regularisation",
            mode="continuous",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                graph_type="dynamic_graph",
                graph_scope="per_timestep",
                regularisation=_regularisation(
                    target_entropy=None,
                    target_weight=0.0,
                    smooth_weight=0.0,
                    warmup_epochs=0,
                ),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=(
                "basedygraph",
                "continuous",
                "dynamic-step",
                "no-reg",
                "test-selected",
            ),
        ),
        ExperimentSpec(
            name="continuous_dynamic_step_eq38",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_dynamic_step_"
                "d96_st3_h3_lam1_smooth0p01"
            ),
            label="Continuous BaseDyGraph dynamic per-step, Eq. 38",
            mode="continuous",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                graph_type="dynamic_graph",
                graph_scope="per_timestep",
                regularisation=_regularisation(
                    target_entropy=3.0,
                    target_weight=1.0,
                    smooth_weight=0.01,
                    warmup_epochs=5,
                ),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=(
                "basedygraph",
                "continuous",
                "dynamic-step",
                "eq38",
                "test-selected",
            ),
        ),
        ExperimentSpec(
            name="continuous_dynamic_window_no_reg",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_dynamic_window_"
                "d96_st3_no_reg"
            ),
            label="Continuous BaseDyGraph dynamic per-window, no regularisation",
            mode="continuous",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                graph_type="dynamic_graph",
                graph_scope="per_window",
                regularisation=_regularisation(
                    target_entropy=None,
                    target_weight=0.0,
                    smooth_weight=0.0,
                    warmup_epochs=0,
                ),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=(
                "basedygraph",
                "continuous",
                "dynamic-window",
                "no-reg",
                "test-selected",
            ),
        ),
        ExperimentSpec(
            name="continuous_dynamic_window_h3",
            run_name=(
                "DO_NOT_REPORT_basedygraph_continuous_dynamic_window_"
                "d96_st3_h3_lam1"
            ),
            label="Continuous BaseDyGraph dynamic per-window, H*=3.0",
            mode="continuous",
            config=BaseDyGraphFinancialConfig(
                mode="continuous",
                graph_type="dynamic_graph",
                graph_scope="per_window",
                regularisation=_regularisation(
                    target_entropy=3.0,
                    target_weight=1.0,
                    smooth_weight=0.0,
                    warmup_epochs=5,
                ),
                **shared,
            ),
            selection_metric="test_five_horizon_mean_log_mae",
            tags=(
                "basedygraph",
                "continuous",
                "dynamic-window",
                "target-entropy",
                "test-selected",
            ),
        ),
    )
    for spec in specs:
        spec.config.validate()
    if len({spec.name for spec in specs}) != len(specs):
        raise AssertionError("Duplicate BaseDyGraph experiment names.")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise AssertionError("Duplicate BaseDyGraph run names.")
    return specs


EXPERIMENT_SPECS = make_experiment_specs()
EXPERIMENT_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one deliberately test-selected BaseDyGraph financial diagnostic."
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_BY_NAME),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forecasting-config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--validation-cache", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument(
        "--mixed-precision", action=argparse.BooleanOptionalAction, default=True
    )
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
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "online", "offline"), default="disabled"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dynamic-graph-financial-forecasting-TEST-CONTAMINATED",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _model_parameter_count(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_score: float,
    best_epoch: int,
    without_improvement: int,
    history: Sequence[Mapping[str, Any]],
    run_signature: str,
    spec: ExperimentSpec,
) -> dict[str, Any]:
    return {
        "version": RUNNER_VERSION,
        "run_signature": run_signature,
        "experiment": spec.name,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "evaluations_without_improvement": int(without_improvement),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "history": [dict(row) for row in history],
        "resolved_config": spec.to_dict(),
    }


def _load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    device: torch.device,
    expected_signature: str,
    restore_rng: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("run_signature") != expected_signature:
        raise ValueError("Checkpoint signature differs from the requested run.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
    if scaler is not None:
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    if restore_rng:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def _average(values: float, count: int) -> float:
    if count <= 0:
        raise RuntimeError("No examples were accumulated.")
    return float(values) / float(count)


def _graph_diagnostics_from_sequences(
    graph_sequences: tuple[Tensor | None, ...],
) -> tuple[float | None, float | None, float | None]:
    selected = next((value for value in reversed(graph_sequences) if value is not None), None)
    if selected is None:
        return None, None, None
    graph = torch.as_tensor(selected).detach().float().clamp_min(1.0e-12)
    entropy = -(graph * graph.log()).sum(dim=-1)
    diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
    return (
        float(entropy.mean().item()),
        float(entropy.exp().mean().item()),
        float(diagonal.mean().item()),
    )


def _run_token_epoch(
    *,
    model: OfficialBaseDyGraphCoarsePathForecaster,
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
    if training and (optimizer is None or scaler is None):
        raise ValueError("Training requires optimizer and scaler.")

    ce_sum = 0.0
    total_objective_sum = 0.0
    regularisation_sum = 0.0
    target_penalty_sum = 0.0
    smooth_penalty_sum = 0.0
    direct_entropy_penalty_sum = 0.0
    correct = 0
    token_count = 0
    examples = 0
    entropy_sum = 0.0
    effective_sum = 0.0
    diagonal_sum = 0.0
    graph_batches = 0
    warmup_sum = 0.0

    synchronise_device(device)
    start = perf_counter()
    context = nullcontext() if training else torch.inference_mode()

    with context:
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            context_tokens, target_s1, _target_s2 = move_training_batch(
                batch,
                device=device,
            )
            context_mean, context_std = move_context_statistics(batch, device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, use_amp):
                output: BaseDyGraphTokenTrainingOutput = model(
                    context_tokens,
                    target_s1=target_s1,
                    context_mean=context_mean,
                    context_std=context_std,
                )
                logits = output.forecast.s1_logits
                ce = F.cross_entropy(
                    logits.reshape(-1, S1_VOCABULARY_SIZE),
                    target_s1.reshape(-1),
                )
                regularisation = graph_regularisation_loss(
                    output.graph_sequences,
                    model.financial_config.regularisation,
                    epoch=epoch,
                )
                objective = ce + regularisation.total

            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite token training objective.")

            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(objective).backward()
                scaler.unscale_(optimizer)
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()

            count = int(target_s1.numel())
            batch_size = int(target_s1.shape[0])
            ce_sum += float(ce.detach().item()) * count
            total_objective_sum += float(objective.detach().item()) * count
            regularisation_sum += float(regularisation.total.detach().item()) * count
            target_penalty_sum += (
                float(regularisation.target_entropy_penalty.detach().item()) * count
            )
            smooth_penalty_sum += (
                float(regularisation.temporal_smoothness_penalty.detach().item()) * count
            )
            direct_entropy_penalty_sum += (
                float(regularisation.direct_entropy_penalty.detach().item()) * count
            )
            warmup_sum += float(regularisation.warmup_factor) * batch_size
            predicted = logits.detach().argmax(dim=-1)
            correct += int((predicted == target_s1).sum().item())
            token_count += count
            examples += batch_size

            entropy, effective, diagonal = _graph_diagnostics_from_sequences(
                output.graph_sequences
            )
            if entropy is not None:
                entropy_sum += entropy
                effective_sum += float(effective)
                diagonal_sum += float(diagonal)
                graph_batches += 1

    synchronise_device(device)
    return {
        "token_loss": _average(ce_sum, token_count),
        "objective_loss": _average(total_objective_sum, token_count),
        "graph_regularisation_loss": _average(regularisation_sum, token_count),
        "target_entropy_penalty": _average(target_penalty_sum, token_count),
        "temporal_smoothness_penalty": _average(smooth_penalty_sum, token_count),
        "direct_entropy_penalty": _average(direct_entropy_penalty_sum, token_count),
        "s1_accuracy": float(correct) / float(max(token_count, 1)),
        "examples": int(examples),
        "tokens": int(token_count),
        "graph_mean_row_entropy": (
            entropy_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_effective_neighbours": (
            effective_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_diagonal_weight": (
            diagonal_sum / graph_batches if graph_batches else None
        ),
        "graph_warmup_factor": warmup_sum / max(examples, 1),
        "seconds": perf_counter() - start,
    }


def _continuous_dataset_config() -> ContinuousDatasetConfig:
    config = ContinuousDatasetConfig(
        context_length=CONTEXT_LENGTH,
        horizons=HORIZONS,
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
    forecast_sum = 0.0
    objective_sum = 0.0
    regularisation_sum = 0.0
    target_penalty_sum = 0.0
    smooth_penalty_sum = 0.0
    target_count = 0
    entropy_sum = 0.0
    effective_sum = 0.0
    diagonal_sum = 0.0
    graph_batches = 0
    synchronise_device(device)
    start = perf_counter()

    for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        x = torch.as_tensor(batch["x"]).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
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
            raise FloatingPointError("Non-finite continuous training objective.")
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        count = int(torch.as_tensor(batch["y"]).numel())
        forecast_sum += float(native_loss.detach().item()) * count
        objective_sum += float(objective.detach().item()) * count
        regularisation_sum += float(regularisation.total.detach().item()) * count
        target_penalty_sum += (
            float(regularisation.target_entropy_penalty.detach().item()) * count
        )
        smooth_penalty_sum += (
            float(regularisation.temporal_smoothness_penalty.detach().item()) * count
        )
        target_count += count
        entropy, effective, diagonal = _graph_diagnostics_from_sequences(
            output.graph_sequences
        )
        if entropy is not None:
            entropy_sum += entropy
            effective_sum += float(effective)
            diagonal_sum += float(diagonal)
            graph_batches += 1

    synchronise_device(device)
    return {
        "forecast_native_loss": _average(forecast_sum, target_count),
        "objective_loss": _average(objective_sum, target_count),
        "graph_regularisation_loss": _average(regularisation_sum, target_count),
        "target_entropy_penalty": _average(target_penalty_sum, target_count),
        "temporal_smoothness_penalty": _average(smooth_penalty_sum, target_count),
        "graph_mean_row_entropy": (
            entropy_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_effective_neighbours": (
            effective_sum / graph_batches if graph_batches else None
        ),
        "graph_mean_diagonal_weight": (
            diagonal_sum / graph_batches if graph_batches else None
        ),
        "seconds": perf_counter() - start,
    }


def _continuous_prediction_raw(
    predictions: Tensor,
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
        y_norm=predictions.float(),
        target_norm_mean=mean,
        target_norm_std=std,
    )


def _batch_for_graph_accumulator(batch: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(batch)
    if "date" not in values and "day" in values:
        values["date"] = values["day"]
    return values


def _evaluate_continuous(
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
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    target_indices: list[Tensor] = []
    native_sum = 0.0
    target_count = 0
    graph_accumulator = (
        GraphArtifactAccumulator(
            asset_cols=asset_cols,
            graph_type=(
                "free_static"
                if model.config.graph_type == "static_graph"
                else "dynamic"
            ),
            num_layers=model.config.num_st_blocks,
            num_heads=model.config.graph_heads,
        )
        if retain_graphs
        else None
    )
    graph_entropy_sum = 0.0
    graph_effective_sum = 0.0
    graph_diagonal_sum = 0.0
    graph_batches = 0
    synchronise_device(device)
    start = perf_counter()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _autocast_context(device, use_amp):
                output = model(x)
            _optimisation, native_loss = _loss_values(
                output.predictions,
                batch,
                device=device,
                output_representation="normalised_close",
                loss_type="cumulative_log_change_mae",
                bps_scale=10000.0,
                eps=1.0e-8,
            )
            raw_prediction = _continuous_prediction_raw(
                output.predictions,
                batch,
                device=device,
            )
            count = int(torch.as_tensor(batch["y"]).numel())
            native_sum += float(native_loss.item()) * count
            target_count += count
            predictions.append(raw_prediction.detach().cpu().contiguous())
            targets.append(
                torch.as_tensor(batch["y_unnormalised"]).float().cpu().contiguous()
            )
            last_values.append(
                torch.as_tensor(batch["last_context_target"])
                .float()
                .cpu()
                .contiguous()
            )
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).long().cpu())
            origin_indices.append(torch.as_tensor(batch["origin_idx"]).long().cpu())
            target_indices.append(
                torch.as_tensor(batch["target_indices"]).long().cpu()
            )
            entropy, effective, diagonal = _graph_diagnostics_from_sequences(
                output.graph_sequences
            )
            if entropy is not None:
                graph_entropy_sum += entropy
                graph_effective_sum += float(effective)
                graph_diagonal_sum += float(diagonal)
                graph_batches += 1
            if graph_accumulator is not None:
                graph_accumulator.add(
                    output.graph,
                    _batch_for_graph_accumulator(batch),
                    batch_size=int(raw_prediction.shape[0]),
                    spatial_beta=None,
                )

    prediction_result = {
        "y_pred": torch.cat(predictions, dim=0).contiguous(),
        "y_true": torch.cat(targets, dim=0).contiguous(),
        "last_context_target": torch.cat(last_values, dim=0).contiguous(),
        "sample_idx": torch.cat(sample_indices, dim=0).contiguous(),
        "origin_idx": torch.cat(origin_indices, dim=0).contiguous(),
        "target_indices": torch.cat(target_indices, dim=0).contiguous(),
        "channels": ["close"],
        "horizons": list(HORIZONS),
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
    native_loss = _average(native_sum, target_count)
    if abs(score - native_loss) > 1.0e-6:
        raise AssertionError(
            "Continuous native CLG-MAE differs from ForecastEvaluator mean. "
            f"native={native_loss}, evaluator={score}."
        )
    if graph_accumulator is not None:
        graphs = graph_accumulator.finalise()
        graphs.update(financial_graph_artifact_metadata(model.config))
        graph_diagnostics = graph_summary(graphs)
    else:
        graphs = None
        graph_diagnostics = {
            "graph_present": graph_batches > 0,
            "mean_row_entropy": (
                graph_entropy_sum / graph_batches if graph_batches else None
            ),
            "mean_effective_neighbours": (
                graph_effective_sum / graph_batches if graph_batches else None
            ),
            "mean_diagonal_weight": (
                graph_diagonal_sum / graph_batches if graph_batches else None
            ),
        }
    synchronise_device(device)
    return {
        "native_loss": native_loss,
        "selection_score": score,
        "prediction_result": prediction_result,
        "metric_results": metric_results,
        "metric_table": metric_table,
        "graphs": graphs,
        "graph_summary": graph_diagnostics,
        "seconds": perf_counter() - start,
    }


def _save_continuous_bundle(
    *,
    run_dir: Path,
    split: str,
    epoch: int,
    bundle: Mapping[str, Any],
) -> None:
    root_prefix = f"best_{split}_"
    atomic_torch_save(
        bundle["prediction_result"],
        run_dir / f"{root_prefix}predictions.pt",
    )
    atomic_torch_save(
        bundle["graphs"],
        run_dir / f"{root_prefix}graphs.pt",
    )
    atomic_csv_save(
        bundle["metric_table"],
        run_dir / f"{root_prefix}metric_table.csv",
    )
    atomic_json_save(
        {
            "epoch": int(epoch),
            "selection_split": "test",
            "evaluated_split": split,
            "selection_score": float(bundle["selection_score"]),
            "native_loss": float(bundle["native_loss"]),
            "graph_summary": bundle["graph_summary"],
            "seconds": float(bundle["seconds"]),
            "do_not_report": True,
            "test_set_contaminated": True,
        },
        run_dir / f"{root_prefix}diagnostics.json",
    )

    analysis_dir = run_dir / "analysis" / split
    analysis_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "prediction_result": bundle["prediction_result"],
            "metric_results": bundle["metric_results"],
        },
        analysis_dir / "predictions.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "graph_artifacts": bundle["graphs"],
            "summary": bundle["graph_summary"],
        },
        analysis_dir / "graphs.pt",
    )
    atomic_csv_save(bundle["metric_table"], analysis_dir / "metric_table.csv")
    atomic_json_save(
        {
            "epoch": int(epoch),
            "split": split,
            "selection_split": "test",
            "selection_score": float(bundle["selection_score"]),
            "graph_summary": bundle["graph_summary"],
            "do_not_report": True,
            "test_set_contaminated": True,
        },
        analysis_dir / "diagnostics.json",
    )


def _save_token_bundle(
    *,
    run_dir: Path,
    split: str,
    policy: str,
    epoch: int,
    bundle: Any,
) -> None:
    if bundle.prediction_result is None or bundle.metric_table is None:
        raise RuntimeError("Real-data token export returned no price predictions.")
    root_prefix = f"best_{split}_"
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "prediction_result": bundle.prediction_result,
            "metric_results": bundle.metric_results,
            "diagnostics": bundle.diagnostics,
        },
        run_dir / f"{root_prefix}predictions.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "graph_artifacts": bundle.graph_artifacts,
            "summary": graph_summary(bundle.graph_artifacts),
        },
        run_dir / f"{root_prefix}graphs.pt",
    )
    atomic_torch_save(
        {"epoch": int(epoch), "token_artifacts": bundle.token_artifacts},
        run_dir / f"{root_prefix}tokens.pt",
    )
    if bundle.sampled_price_path_artifacts is not None:
        atomic_torch_save(
            {
                "epoch": int(epoch),
                "sampled_price_path_artifacts": bundle.sampled_price_path_artifacts,
            },
            run_dir / f"{root_prefix}sampled_price_paths.pt",
        )
    atomic_csv_save(bundle.metric_table, run_dir / f"{root_prefix}metric_table.csv")
    diagnostics = dict(bundle.diagnostics)
    diagnostics.update(
        {
            "epoch": int(epoch),
            "split": split,
            "policy": policy,
            "graph_summary": graph_summary(bundle.graph_artifacts),
            "do_not_report": True,
            "test_set_contaminated": True,
        }
    )
    atomic_json_save(diagnostics, run_dir / f"{root_prefix}diagnostics.json")

    policy_dir = run_dir / "analysis" / split / policy
    policy_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "prediction_result": bundle.prediction_result,
            "metric_results": bundle.metric_results,
            "diagnostics": bundle.diagnostics,
        },
        policy_dir / "predictions.pt",
    )
    atomic_torch_save(
        {
            "epoch": int(epoch),
            "graph_artifacts": bundle.graph_artifacts,
            "summary": graph_summary(bundle.graph_artifacts),
        },
        policy_dir / "graphs.pt",
    )
    atomic_torch_save(
        {"epoch": int(epoch), "token_artifacts": bundle.token_artifacts},
        policy_dir / "tokens.pt",
    )
    if bundle.sampled_price_path_artifacts is not None:
        atomic_torch_save(
            {
                "epoch": int(epoch),
                "sampled_price_path_artifacts": bundle.sampled_price_path_artifacts,
            },
            policy_dir / "sampled_price_paths.pt",
        )
    atomic_csv_save(bundle.metric_table, policy_dir / "metric_table.csv")
    atomic_json_save(diagnostics, policy_dir / "diagnostics.json")
    atomic_json_save(
        {
            "selected_policy": policy,
            "selected_temperature": (
                1.0 if policy == "temperature_1" else None
            ),
            "selection_split": "test",
            "do_not_report": True,
        },
        run_dir / "analysis" / split / "temperature_selection.json",
    )


def _token_postselection_exports(
    *,
    model: OfficialBaseDyGraphCoarsePathForecaster,
    run_dir: Path,
    epoch: int,
    datasets: Mapping[str, CachedTokenGraphDataset],
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
        decoding = {
            "token_selection": "sample" if sampled else "argmax",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.9 if sampled else 1.0,
            "sample_count": 10 if sampled else 1,
        }
        bundle = generate_validation_artifacts(
            model=model,
            loader=loaders[split],
            dataset=datasets[split],
            device=device,
            use_amp=use_amp,
            decoding_config=decoding,
            tokenizer=tokenizer,
            raw_train_split=raw_train_split,
            decode_series_batch_size=decode_series_batch_size,
            early_stopping_horizons=HORIZONS,
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


def _resolved_config_payload(
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Serialise both the exact adapter and the standard analysis schema."""
    graph_type = (
        "free_static"
        if spec.config.graph_type == "static_graph"
        else "dynamic"
    )
    training = {
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
    }
    data = {
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "horizons": list(HORIZONS),
        "stride": 15,
        "input_channels": list(INPUT_CHANNELS),
        "target_channel": "close",
        "train_cache": None if args.train_cache is None else str(args.train_cache),
        "validation_cache": (
            None if args.validation_cache is None else str(args.validation_cache)
        ),
        "test_cache": None if args.test_cache is None else str(args.test_cache),
        "data_dir": str(args.data_dir),
    }
    payload: dict[str, Any] = {
        "runner": "src.training.run_basedygraph_financial",
        "runner_version": RUNNER_VERSION,
        "model_family": "official_basedygraph_financial",
        "experiment": spec.name,
        "run_name": args.run_name or spec.run_name,
        "selection_split": "test",
        "test_set_contaminated": True,
        "do_not_report": True,
        "basedygraph_financial": spec.config.to_dict(),
        "data": data,
        "training": training,
        "sampling": {
            "validation_temperature": 1.0,
            "top_p": 0.9,
            "top_k": 0,
            "sample_count": 10,
        },
    }
    if spec.mode == "token":
        payload["models"] = {
            "dynamic_graph": {
                "num_nodes": spec.config.num_nodes,
                "context_length": spec.config.context_length,
                "d_model": spec.config.d_model,
                "num_st_blocks": spec.config.num_st_blocks,
                "graph": {
                    "type": graph_type,
                    "num_heads": spec.config.graph_heads,
                    "hidden_dim": spec.config.graph_hidden_dim,
                    "activation": spec.config.graph_activation,
                    "add_self_loops": spec.config.add_self_loops,
                },
                "heads": {
                    "prediction_length": spec.config.prediction_length,
                    "evaluation_horizons": list(spec.config.evaluation_horizons),
                    "future_token_mode": "coarse_only",
                    "s1_vocabulary_size": S1_VOCABULARY_SIZE,
                    "s2_loss_weight": 0.0,
                },
                "future_predictor": {
                    "type": "structured_parallel",
                    "num_layers": spec.config.future_predictor_layers,
                    "num_heads": spec.config.future_predictor_heads,
                },
            }
        }
        training["early_stopping_metric"] = "test_token_loss"
        training["selection_metric"] = spec.selection_metric
    else:
        payload["model"] = {
            "output_representation": "normalised_close",
            "graph": {
                "type": graph_type,
                "num_heads": spec.config.graph_heads,
                "hidden_dim": spec.config.graph_hidden_dim,
                "activation": spec.config.graph_activation,
                "add_self_loops": spec.config.add_self_loops,
            },
            "temporal": {
                "type": "official_basedygraph_transformer",
                "d_model": spec.config.d_model,
                "num_layers": spec.config.temporal_layers,
                "num_heads": spec.config.temporal_heads,
            },
            "num_st_blocks": spec.config.num_st_blocks,
        }
        training["selection_metric"] = spec.selection_metric
    return payload


def _init_wandb(args: argparse.Namespace, spec: ExperimentSpec) -> Any | None:
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
        "training_graph_regularisation_loss": float(
            train["graph_regularisation_loss"]
        ),
        "training_graph_target_entropy_penalty": float(
            train["target_entropy_penalty"]
        ),
        "training_graph_temporal_smooth_penalty": float(
            train["temporal_smoothness_penalty"]
        ),
        "training_s1_accuracy": float(train["s1_accuracy"]),
        "training_graph_mean_row_entropy": train["graph_mean_row_entropy"],
        "training_graph_mean_effective_neighbours": train[
            "graph_mean_effective_neighbours"
        ],
        "training_graph_mean_diagonal_weight": train[
            "graph_mean_diagonal_weight"
        ],
        "training_graph_warmup_factor": float(train["graph_warmup_factor"]),
        "test_token_loss": float(test["token_loss"]),
        "test_s1_accuracy": float(test["s1_accuracy"]),
        "test_graph_mean_row_entropy": test["graph_mean_row_entropy"],
        "test_graph_mean_effective_neighbours": test[
            "graph_mean_effective_neighbours"
        ],
        "test_graph_mean_diagonal_weight": test["graph_mean_diagonal_weight"],
        # Explicit legacy aliases: these values are TEST selection metrics.
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
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "training_native_log_mae": float(train["forecast_native_loss"]),
        "training_objective_loss": float(train["objective_loss"]),
        "training_graph_regularisation_loss": float(
            train["graph_regularisation_loss"]
        ),
        "training_graph_target_entropy_penalty": float(
            train["target_entropy_penalty"]
        ),
        "training_graph_temporal_smooth_penalty": float(
            train["temporal_smoothness_penalty"]
        ),
        "training_graph_mean_row_entropy": train["graph_mean_row_entropy"],
        "training_graph_mean_effective_neighbours": train[
            "graph_mean_effective_neighbours"
        ],
        "training_graph_mean_diagonal_weight": train[
            "graph_mean_diagonal_weight"
        ],
        "test_native_log_mae": float(test["native_loss"]),
        "test_mean_log_mae": float(test["selection_score"]),
        "test_graph_mean_row_entropy": test["graph_summary"].get(
            "mean_row_entropy"
        ),
        "test_graph_mean_effective_neighbours": test["graph_summary"].get(
            "mean_effective_neighbours"
        ),
        # Explicit legacy aliases: these values are TEST selection metrics.
        "selection_score": float(test["selection_score"]),
        "validation_loss": float(test["native_loss"]),
        "best_score_after_epoch": float(best_score),
        "best_epoch_after_epoch": int(best_epoch),
        "selection_split": "test",
        "train_seconds": float(train["seconds"]),
        "test_selection_seconds": float(test["seconds"]),
    }
    metric = test["metric_results"]["cumulative_log_change_mae"].reshape(-1)
    for horizon, value in zip(HORIZONS, metric, strict=True):
        record[f"test_cumulative_log_change_mae_h{horizon}"] = float(value.item())
        record[f"val_cumulative_log_change_mae_h{horizon}"] = float(value.item())
    return record


def _run_token_experiment(
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    run_signature: str,
    metadata: dict[str, Any],
    forecasting_config: Mapping[str, Any],
) -> None:
    for name, path in (
        ("train", args.train_cache),
        ("validation", args.validation_cache),
        ("test", args.test_cache),
    ):
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} token cache: {path}")

    datasets = {
        "train": CachedTokenGraphDataset.from_path(args.train_cache),
        "validation": CachedTokenGraphDataset.from_path(args.validation_cache),
        "test": CachedTokenGraphDataset.from_path(args.test_cache),
    }
    for name, dataset in datasets.items():
        if dataset.data_mode != "real":
            raise ValueError(f"{name} token cache is not real-data.")
        if dataset.context_length != CONTEXT_LENGTH:
            raise ValueError(f"{name} token context length differs.")
        if dataset.prediction_length != PREDICTION_LENGTH:
            raise ValueError(f"{name} token prediction length differs.")
        if len(dataset.asset_cols) != NUM_NODES:
            raise ValueError(f"{name} token asset count differs.")
    if not (
        datasets["train"].asset_cols
        == datasets["validation"].asset_cols
        == datasets["test"].asset_cols
    ):
        raise ValueError("Token cache asset orders differ.")

    train_dataset = _limit_dataset(datasets["train"], args.max_train_windows)
    test_dataset = _limit_dataset(datasets["test"], args.max_selection_windows)
    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.selection_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = OfficialBaseDyGraphCoarsePathForecaster(spec.config).to(device)
    optimizer = _optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = _new_grad_scaler(use_amp)
    total_params, trainable_params = _model_parameter_count(model)
    metadata.update(
        {
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "asset_cols": list(datasets["train"].asset_cols),
            "basedygraph_observed_commit": model.external_commit,
            "train_windows": len(train_dataset),
            "test_selection_windows": len(test_dataset),
            "validation_windows": len(datasets["validation"]),
        }
    )
    atomic_json_save(metadata, run_dir / "run_metadata.json")

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint_path = run_dir / "last_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _load_checkpoint(
            checkpoint_path,
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
                description=f"train epoch {epoch}",
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
                description=f"TEST selection epoch {epoch}",
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
                        "test_selection/token_ce": test_metrics["token_loss"],
                        "train/graph_regularisation": train_metrics[
                            "graph_regularisation_loss"
                        ],
                        "train/graph_entropy": train_metrics[
                            "graph_mean_row_entropy"
                        ],
                    },
                    step=epoch,
                )
            if without_improvement >= int(args.patience):
                break

        if best_epoch <= 0:
            raise RuntimeError("No token checkpoint was selected.")
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
            split: build_loader(
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
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
    except Exception:
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _run_continuous_experiment(
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    use_amp: bool,
    run_signature: str,
    metadata: dict[str, Any],
) -> None:
    train_raw, val_raw, test_raw = load_candle_splits(args.data_dir)
    train_split, validation_split, test_split = clean_candle_splits(
        train_raw, val_raw, test_raw
    )
    dataset_config = _continuous_dataset_config()
    datasets = {
        "train": build_continuous_dataset(train_split, config=dataset_config),
        "validation": build_continuous_dataset(
            validation_split, config=dataset_config
        ),
        "test": build_continuous_dataset(test_split, config=dataset_config),
    }
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

    model = OfficialBaseDyGraphContinuousForecaster(spec.config).to(device)
    optimizer = _optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = _new_grad_scaler(use_amp)
    total_params, trainable_params = _model_parameter_count(model)
    metadata.update(
        {
            "parameter_count": total_params,
            "trainable_parameter_count": trainable_params,
            "asset_cols": list(train_split["asset_cols"]),
            "basedygraph_observed_commit": model.external_commit,
            "train_windows": len(train_dataset),
            "test_selection_windows": len(test_dataset),
            "validation_windows": len(datasets["validation"]),
        }
    )
    atomic_json_save(metadata, run_dir / "run_metadata.json")

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        checkpoint_path = run_dir / "last_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = _load_checkpoint(
            checkpoint_path,
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
            test_bundle = _evaluate_continuous(
                model=model,
                loader=test_loader,
                device=device,
                use_amp=use_amp,
                train_split=train_split,
                asset_cols=train_split["asset_cols"],
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
                        "train/graph_regularisation": train_metrics[
                            "graph_regularisation_loss"
                        ],
                        "train/graph_entropy": train_metrics[
                            "graph_mean_row_entropy"
                        ],
                    },
                    step=epoch,
                )
            if without_improvement >= int(args.patience):
                break

        if best_epoch <= 0:
            raise RuntimeError("No continuous checkpoint was selected.")
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
        export_scores: dict[str, float] = {}
        for split in ("train", "test", "validation"):
            bundle = _evaluate_continuous(
                model=model,
                loader=export_loaders[split],
                device=device,
                use_amp=use_amp,
                train_split=train_split,
                asset_cols=train_split["asset_cols"],
                description=f"selected checkpoint {split} export",
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
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    spec = EXPERIMENT_BY_NAME[args.experiment]
    run_name = args.run_name or spec.run_name
    if not run_name.startswith("DO_NOT_REPORT"):
        raise ValueError(
            "This deliberately test-selected runner requires a DO_NOT_REPORT run name."
        )
    if args.max_epochs <= 0 or args.patience <= 0:
        raise ValueError("max_epochs and patience must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimiser settings.")
    if args.train_batch_size <= 0 or args.selection_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")

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
    resolved_payload = _resolved_config_payload(spec=spec, args=args)
    resolved_payload["run_name"] = run_name
    atomic_json_save(resolved_payload, run_dir / "resolved_config.json")

    signature_payload = {
        "runner_version": RUNNER_VERSION,
        "experiment": spec.name,
        "run_name": run_name,
        "model": spec.config.to_dict(),
        "training": resolved_payload["training"],
        "data": resolved_payload["data"],
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
        "graph_diagonal_policy": (
            "eligible in official scorer softmax; add_self_loops controls only "
            "extra identity addition"
        ),
        "graph_scope": spec.config.graph_scope,
        "graph_regularisation": asdict(spec.config.regularisation),
        "forecast_horizons": list(HORIZONS),
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "train_batch_size": int(args.train_batch_size),
        "selection_batch_size": int(args.selection_batch_size),
        "export_batch_size": int(args.export_batch_size),
        "warning": (
            "Checkpoint selection used the October-December test split. "
            "This run is a contaminated diagnostic and must not be reported "
            "as held-out test performance."
        ),
    }
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        metadata["warning"] + "\n",
        encoding="utf-8",
    )

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

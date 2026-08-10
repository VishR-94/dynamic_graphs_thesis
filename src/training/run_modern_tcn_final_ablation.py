from __future__ import annotations

"""Run final ModernTCN graph diagnostic ablations.

This runner reuses the winning one-block ModernTCN graph architecture and adds
only two forecast/loss protocols:

* parallel_weighted: train the existing five-horizon OHLCV-input head with
  inverse-reference horizon weights while selecting on the ordinary unweighted
  five-horizon mean;
* autoregressive_close_only: train a Close-only one-step log-return model on
  dense one-step windows and roll the predicted Close forward for 60 minutes.
  The recurrent model never fabricates Open, High, Low, or Volume channels.

The runner remains deliberately test-selected and marks every output as
``DO_NOT_REPORT``.
"""

import argparse
import copy
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

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.models.graph_priors import (
    build_absolute_correlation_graph_prior,
    build_sector_graph_prior,
)
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Config,
    ModernTCNGraphRound1Model,
    graph_component_summary,
    round1_model_config_from_mapping,
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
from src.training.run_modern_tcn_graph_round1 import (
    GRAPH_ORIENTATION,
    _advance_schedule,
    _autocast_context,
    _build_loader,
    _build_optimizer,
    _checkpoint,
    _dataset_config,
    _evaluate_selection,
    _final_graph_stats,
    _forecast_errors,
    _git_value,
    _graph_stats_accumulator,
    _history_record,
    _init_wandb,
    _learning_rates,
    _load_config,
    _module_gradient_norm,
    _move_optimizer_state,
    _new_grad_scaler,
    _parameter_partition,
    _prepare_run_dir,
    _save_export,
    _scalar_gradient,
    _signature,
)
from src.utils.metric_tables import make_evaluation_table


ConfigDict = dict[str, Any]


AUTOREGRESSIVE_CLOSE_ONLY = "autoregressive_close_only"
PARALLEL_WEIGHTED = "parallel_weighted"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one final ModernTCN graph diagnostic ablation."
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


def _strategy(config: Mapping[str, Any]) -> str:
    return str(config["training"].get("forecast_strategy", PARALLEL_WEIGHTED))


def _is_close_only_autoregressive(config: Mapping[str, Any]) -> bool:
    return _strategy(config) == AUTOREGRESSIVE_CLOSE_ONLY


def _model_config_for_strategy(
    config: Mapping[str, Any],
    *,
    num_nodes: int,
) -> ModernTCNGraphRound1Config:
    values = copy.deepcopy(dict(config))
    if _is_close_only_autoregressive(config):
        values["data"] = dict(values["data"])
        values["data"]["horizons"] = [1]
        input_channels = tuple(
            str(value) for value in values["data"]["input_channels"]
        )
        if input_channels != ("close",):
            raise ValueError(
                "Close-only autoregression requires data.input_channels=['close']."
            )
        if str(values["model"].get("output_representation")) != (
            "cumulative_log_change"
        ):
            raise ValueError(
                "Close-only autoregression requires direct cumulative-log-change "
                "output."
            )
    return round1_model_config_from_mapping(values, num_nodes=num_nodes)


def _training_dataset_config(config: Mapping[str, Any]) -> ContinuousDatasetConfig:
    if not _is_close_only_autoregressive(config):
        return _dataset_config(config)
    data = config["data"]
    normalisation = config["normalisation"]
    return ContinuousDatasetConfig(
        context_length=int(data["context_length"]),
        horizons=(1,),
        stride=int(config["training"].get("one_step_training_stride", 1)),
        input_channels=tuple(str(value) for value in data["input_channels"]),
        target_channels=(str(data["target_channel"]),),
        input_representation="raw",
        eps=float(normalisation["eps"]),
        clip=bool(normalisation["clip"]),
        clip_min=float(normalisation["clip_min"]),
        clip_max=float(normalisation["clip_max"]),
    )


def _horizon_weights(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tensor | None:
    loss_values = config["training"].get("loss", {})
    weights = loss_values.get("horizon_weights")
    if weights is None:
        return None
    result = torch.tensor([float(value) for value in weights], device=device)
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    if result.numel() != len(horizons):
        raise ValueError("loss.horizon_weights must match data.horizons.")
    if not torch.isfinite(result).all() or torch.any(result <= 0):
        raise ValueError("loss.horizon_weights must be positive and finite.")
    return result.view(1, -1, 1, 1)


def _train_epoch_weighted_parallel(
    *,
    model: ModernTCNGraphRound1Model,
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
    weights = _horizon_weights(config, device=device)
    model.train()
    eps = float(config["normalisation"]["eps"])
    bps_scale = float(training["loss"]["bps_scale"])
    total_unweighted_error = 0.0
    total_weighted_error = 0.0
    target_count = 0
    weighted_count = 0.0
    optimisation_sum = 0.0
    diagnostic_taken = False
    graph_gradient = 0.0
    alpha_gradient = 0.0
    beta_gradient = 0.0
    state_gradient = 0.0

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
        unweighted_loss = absolute_error.mean()
        if weights is None:
            weighted_error = absolute_error
        else:
            weighted_error = absolute_error * weights
        weighted_loss = weighted_error.mean()
        optimisation_loss = weighted_loss * bps_scale
        if not torch.isfinite(optimisation_loss):
            raise FloatingPointError("Non-finite weighted parallel loss.")

        scaler.scale(optimisation_loss).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            graph_gradient = _module_gradient_norm(model.graph_learner)
            alpha_gradient = _scalar_gradient(model.graph_learner.raw_alpha)
            beta_gradient = _scalar_gradient(model.spatial_gate.raw_beta)
            state_gradient = _module_gradient_norm(model.state_projection)
            diagnostic_taken = True
        clip = float(training["gradient_clip_norm"])
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        count = int(absolute_error.numel())
        total_unweighted_error += float(absolute_error.sum().item())
        total_weighted_error += float(weighted_error.sum().item())
        target_count += count
        weighted_count += float(weighted_error.numel())
        optimisation_sum += float(optimisation_loss.item()) * count
        progress.set_postfix(native=f"{total_unweighted_error / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Training loader produced no targets.")
    return {
        "training_native_loss": total_unweighted_error / target_count,
        "training_weighted_native_loss": total_weighted_error / max(weighted_count, 1.0),
        "training_objective_loss": optimisation_sum / target_count,
        "block_0_graph_gradient_norm": graph_gradient,
        "block_0_alpha_gradient_norm": alpha_gradient,
        "block_0_beta_gradient_norm": beta_gradient,
        "block_0_state_projection_gradient_norm": state_gradient,
    }


def _autoregressive_rollout_uses_amp(
    config: Mapping[str, Any],
    *,
    training_amp_enabled: bool,
) -> bool:
    """Return whether free-running rollout may use CUDA FP16 autocast.

    Training AMP and rollout AMP are separate decisions.  Autoregressive
    rollout repeatedly feeds model output back into the next context.  Small
    scale errors can therefore grow across steps, and FP16 Q/K matmuls can
    overflow before the same computation would become non-finite in FP32.
    The default is consequently FP32 rollout, even when training uses AMP.
    """

    requested = bool(
        config["training"].get(
            "autoregressive_rollout_mixed_precision",
            False,
        )
    )
    return bool(training_amp_enabled and requested)


def _finite_tensor_summary(values: Tensor) -> str:
    tensor = torch.as_tensor(values).detach().float()
    finite = tensor[torch.isfinite(tensor)]
    non_finite = int((~torch.isfinite(tensor)).sum().item())
    if finite.numel() == 0:
        return f"shape={tuple(tensor.shape)}, finite=0, non_finite={non_finite}"
    return (
        f"shape={tuple(tensor.shape)}, finite={finite.numel()}, "
        f"non_finite={non_finite}, min={float(finite.min().item()):.6g}, "
        f"max={float(finite.max().item()):.6g}"
    )


def _require_finite_rollout_tensor(
    name: str,
    values: Tensor,
    *,
    step: int,
) -> None:
    if torch.isfinite(values).all():
        return
    raise FloatingPointError(
        f"Autoregressive rollout produced non-finite {name} at step "
        f"{int(step)}. {_finite_tensor_summary(values)}"
    )


def _normalise_raw_context(
    raw_context: Tensor,
    *,
    input_channels: Sequence[str],
    target_channel: str,
    eps: float,
    clip: bool,
    clip_min: float,
    clip_max: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mean = raw_context.mean(dim=1)
    std = raw_context.std(dim=1, unbiased=False).clamp_min(float(eps))
    x = (raw_context - mean.unsqueeze(1)) / std.unsqueeze(1)
    if clip:
        x = x.clamp(float(clip_min), float(clip_max))
    target_position = list(input_channels).index(str(target_channel))
    target_mean = mean[:, :, target_position : target_position + 1]
    target_std = std[:, :, target_position : target_position + 1]
    return x, target_mean, target_std


def _require_strictly_positive_rollout_tensor(
    name: str,
    values: Tensor,
    *,
    step: int,
) -> None:
    tensor = torch.as_tensor(values)
    _require_finite_rollout_tensor(name, tensor, step=step)
    non_positive = tensor <= 0
    if not torch.any(non_positive):
        return
    count = int(non_positive.sum().item())
    minimum = float(tensor.min().item())
    raise FloatingPointError(
        f"Autoregressive rollout produced {count} non-positive values in "
        f"{name} at step {int(step)}; minimum={minimum:.6g}. No price "
        "floor or persistence fallback is applied."
    )


def _next_close_from_log_return(
    previous_close: Tensor,
    predicted_log_return: Tensor,
    *,
    step: int,
) -> Tensor:
    """Convert one-step log returns to strictly positive next Close values.

    The calculation is performed in float64 so the recurrent price state does
    not accumulate avoidable float32 exponentiation error.  It contains no
    clipping: a non-finite or non-positive result is reported as model
    divergence rather than silently replaced by a numerical floor.
    """

    previous = torch.as_tensor(previous_close, dtype=torch.float64)
    log_return = torch.as_tensor(predicted_log_return, dtype=torch.float64)
    if previous.shape != log_return.shape:
        raise ValueError(
            "previous_close and predicted_log_return must have matching shapes."
        )
    _require_strictly_positive_rollout_tensor(
        "previous Close",
        previous,
        step=step,
    )
    _require_finite_rollout_tensor(
        "predicted one-step log return",
        log_return,
        step=step,
    )
    next_close = previous * torch.exp(log_return)
    _require_strictly_positive_rollout_tensor(
        "reconstructed next Close",
        next_close,
        step=step,
    )
    return next_close

def _autoregressive_rollout(
    *,
    model: ModernTCNGraphRound1Model,
    batch: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Any]:
    """Roll a Close-only one-step log-return model forward causally.

    ``raw_context`` contains one channel only.  At every step the observed or
    generated Close history is context-normalised, the model predicts the next
    one-step cumulative log change, and the next Close is reconstructed as
    ``previous_close * exp(predicted_log_return)``.  No unmodelled candle field
    is manufactured.
    """

    data = config["data"]
    normalisation = config["normalisation"]
    input_channels = tuple(str(value) for value in data["input_channels"])
    target_channel = str(data["target_channel"])
    if input_channels != ("close",) or target_channel != "close":
        raise ValueError(
            "Close-only autoregression requires one input/target channel: close."
        )
    if str(config["model"].get("output_representation")) != (
        "cumulative_log_change"
    ):
        raise ValueError(
            "Close-only rollout expects direct one-step cumulative-log-change "
            "model output."
        )

    horizons = tuple(int(value) for value in data["horizons"])
    rollout_length = int(
        config["training"].get("autoregressive_rollout_length", max(horizons))
    )
    if rollout_length < max(horizons):
        raise ValueError("autoregressive_rollout_length is smaller than max horizon.")

    # Retain the recurrent price state in float64.  Only the normalised tensor
    # passed through the neural network is converted to float32.
    raw_context = torch.as_tensor(batch["context_unnormalised"]).to(
        device=device,
        dtype=torch.float64,
        non_blocking=True,
    )
    expected_axes = (
        int(config["data"]["context_length"]),
        int(raw_context.shape[2]),
        1,
    )
    if tuple(raw_context.shape[1:]) != expected_axes:
        raise ValueError(
            "Close-only context has unexpected [T,N,C] axes: "
            f"{tuple(raw_context.shape[1:])}; expected {expected_axes}."
        )
    context_start = torch.as_tensor(batch["context_start"]).long()
    session_length = torch.as_tensor(batch["session_length"]).long()
    close_predictions: list[Tensor] = []
    log_return_predictions: list[Tensor] = []
    first_output = None
    rollout_use_amp = _autoregressive_rollout_uses_amp(
        config,
        training_amp_enabled=use_amp,
    )
    _require_strictly_positive_rollout_tensor(
        "initial Close context",
        raw_context,
        step=0,
    )

    for step in range(rollout_length):
        rollout_step = int(step) + 1
        x_norm, _, _ = _normalise_raw_context(
            raw_context,
            input_channels=input_channels,
            target_channel=target_channel,
            eps=float(normalisation["eps"]),
            clip=bool(normalisation["clip"]),
            clip_min=float(normalisation["clip_min"]),
            clip_max=float(normalisation["clip_max"]),
        )
        x_model = x_norm.to(dtype=torch.float32)
        _require_finite_rollout_tensor(
            "normalised Close context",
            x_model,
            step=rollout_step,
        )
        try:
            with _autocast_context(device, rollout_use_amp):
                output = model(
                    x_model,
                    context_start=context_start + int(step),
                    session_length=session_length,
                )
        except ValueError as error:
            if "non-finite" not in str(error):
                raise
            raise FloatingPointError(
                "Close-only autoregressive model forward became non-finite at "
                f"rollout step {rollout_step}; rollout_amp={rollout_use_amp}. "
                f"Normalised input: {_finite_tensor_summary(x_model)}. "
                f"Original error: {error}"
            ) from error
        if first_output is None:
            first_output = output

        predicted_log_return = output.predictions[:, 0].to(dtype=torch.float64)
        _require_finite_rollout_tensor(
            "predicted one-step log return",
            predicted_log_return,
            step=rollout_step,
        )
        previous_close = raw_context[:, -1]
        next_close = _next_close_from_log_return(
            previous_close,
            predicted_log_return,
            step=rollout_step,
        )
        close_predictions.append(next_close)
        log_return_predictions.append(predicted_log_return)
        raw_context = torch.cat(
            [raw_context[:, 1:], next_close.unsqueeze(1)],
            dim=1,
        )

    if first_output is None:
        raise RuntimeError("Autoregressive rollout produced no steps.")
    dense_close = torch.stack(close_predictions, dim=1)
    dense_log_return = torch.stack(log_return_predictions, dim=1)
    return dense_close, dense_log_return, first_output

def _select_rollout_horizons(dense: Tensor, horizons: Sequence[int]) -> Tensor:
    indices = torch.tensor([int(h) - 1 for h in horizons], device=dense.device)
    return dense.index_select(1, indices)


def _autoregressive_errors(
    *,
    model: ModernTCNGraphRound1Model,
    batch: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]:
    dense_close, dense_log_return, first_output = _autoregressive_rollout(
        model=model,
        batch=batch,
        device=device,
        use_amp=use_amp,
        config=config,
    )
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    predicted_raw = _select_rollout_horizons(dense_close, horizons)
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device,
        dtype=torch.float64,
        non_blocking=True,
    )
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device,
        dtype=torch.float64,
        non_blocking=True,
    )
    _require_strictly_positive_rollout_tensor(
        "autoregressive horizon predictions",
        predicted_raw,
        step=max(horizons),
    )
    _require_strictly_positive_rollout_tensor(
        "real forecast targets",
        true_raw,
        step=0,
    )
    _require_strictly_positive_rollout_tensor(
        "last observed Close",
        last,
        step=0,
    )
    predicted_change = torch.log(predicted_raw) - torch.log(last.unsqueeze(1))
    true_change = torch.log(true_raw) - torch.log(last.unsqueeze(1))
    absolute_error = (predicted_change - true_change).abs()
    _require_finite_rollout_tensor(
        "autoregressive cumulative-log-change errors",
        absolute_error,
        step=max(horizons),
    )
    return (
        predicted_raw,
        true_raw,
        last,
        absolute_error,
        dense_close,
        dense_log_return,
        first_output,
    )

def _one_step_log_return_errors(
    prediction: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return true one-step log change and absolute direct-head error."""

    predicted_change = torch.as_tensor(prediction).to(
        device=device,
        dtype=torch.float32,
    )
    true_raw = torch.as_tensor(batch["y_unnormalised"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    last = torch.as_tensor(batch["last_context_target"]).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    if tuple(predicted_change.shape) != tuple(true_raw.shape):
        raise ValueError(
            "Direct one-step prediction and target shapes differ: "
            f"{tuple(predicted_change.shape)} versus {tuple(true_raw.shape)}."
        )
    _require_strictly_positive_rollout_tensor(
        "one-step training targets",
        true_raw,
        step=1,
    )
    _require_strictly_positive_rollout_tensor(
        "one-step training last Close",
        last,
        step=0,
    )
    true_change = torch.log(true_raw) - torch.log(last.unsqueeze(1))
    absolute_error = (predicted_change - true_change).abs()
    _require_finite_rollout_tensor(
        "one-step log-return training errors",
        absolute_error,
        step=1,
    )
    return true_change, absolute_error


def _train_epoch_autoregressive_one_step(
    *,
    model: ModernTCNGraphRound1Model,
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
    bps_scale = float(training["loss"]["bps_scale"])
    total_absolute_error = 0.0
    target_count = 0
    optimisation_sum = 0.0
    diagnostic_taken = False
    graph_gradient = 0.0
    alpha_gradient = 0.0
    beta_gradient = 0.0
    state_gradient = 0.0

    progress = tqdm(
        loader,
        desc=f"train one-step epoch {epoch}",
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
        _, absolute_error = _one_step_log_return_errors(
            output.predictions,
            batch,
            device=device,
        )
        native_loss = absolute_error.mean()
        optimisation_loss = native_loss * bps_scale
        if not torch.isfinite(optimisation_loss):
            raise FloatingPointError("Non-finite one-step training loss.")

        scaler.scale(optimisation_loss).backward()
        scaler.unscale_(optimizer)
        if not diagnostic_taken:
            graph_gradient = _module_gradient_norm(model.graph_learner)
            alpha_gradient = _scalar_gradient(model.graph_learner.raw_alpha)
            beta_gradient = _scalar_gradient(model.spatial_gate.raw_beta)
            state_gradient = _module_gradient_norm(model.state_projection)
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
        progress.set_postfix(native=f"{total_absolute_error / max(target_count, 1):.6g}")

    if target_count <= 0:
        raise RuntimeError("Training loader produced no one-step targets.")
    return {
        "training_native_loss": total_absolute_error / target_count,
        "training_objective_loss": optimisation_sum / target_count,
        "block_0_graph_gradient_norm": graph_gradient,
        "block_0_alpha_gradient_norm": alpha_gradient,
        "block_0_beta_gradient_norm": beta_gradient,
        "block_0_state_projection_gradient_norm": state_gradient,
    }


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


def _evaluate_selection_autoregressive(
    *,
    model: ModernTCNGraphRound1Model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    config: Mapping[str, Any],
    description: str,
) -> dict[str, Any]:
    model.eval()
    horizons = tuple(int(value) for value in config["data"]["horizons"])
    horizon_sum = torch.zeros(len(horizons), dtype=torch.float64)
    horizon_count = torch.zeros(len(horizons), dtype=torch.float64)
    selected_stats = _graph_stats_accumulator()
    static_stats = _graph_stats_accumulator()
    dynamic_stats = _graph_stats_accumulator()

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        ):
            _, _, _, absolute_error, _, _, first_output = _autoregressive_errors(
                model=model,
                batch=batch,
                device=device,
                use_amp=use_amp,
                config=config,
            )
            error = absolute_error.detach().double().cpu()
            horizon_sum += error.sum(dim=(0, 2, 3))
            horizon_count += torch.full(
                (len(horizons),),
                float(error.shape[0] * error.shape[2] * error.shape[3]),
                dtype=torch.float64,
            )
            batch_size = int(error.shape[0])
            _add_graph_stats(selected_stats, first_output.graph.selected, batch_size=batch_size)
            _add_graph_stats(static_stats, first_output.graph.base, batch_size=batch_size)
            _add_graph_stats(dynamic_stats, first_output.graph.dynamic, batch_size=batch_size)

    if torch.any(horizon_count <= 0):
        raise RuntimeError("Selection loader produced no autoregressive targets.")
    by_horizon = horizon_sum / horizon_count
    selected_summary = _final_graph_stats(selected_stats)
    static_summary = _final_graph_stats(static_stats)
    dynamic_summary = _final_graph_stats(dynamic_stats)
    return {
        "selection_score": float(by_horizon.mean().item()),
        "by_horizon": {
            int(horizon): float(value)
            for horizon, value in zip(horizons, by_horizon.tolist(), strict=True)
        },
        "block_0_alpha": None if model.alpha() is None else float(model.alpha().item()),
        "block_0_beta": float(model.beta().item()),
        "block_0_beta_trainable": bool(model.spatial_gate.raw_beta is not None),
        "block_0_selected_entropy": selected_summary["entropy"],
        "block_0_selected_effective_neighbours": selected_summary["effective_neighbours"],
        "block_0_static_entropy": static_summary["entropy"],
        "block_0_static_effective_neighbours": static_summary["effective_neighbours"],
        "block_0_dynamic_entropy": dynamic_summary["entropy"],
        "block_0_dynamic_effective_neighbours": dynamic_summary["effective_neighbours"],
    }


def _rollout_step_diagnostics(
    dense_close: Tensor,
    dense_log_return: Tensor,
    last_close: Tensor,
) -> list[dict[str, float | int]]:
    """Summarise recurrent scale at every rollout step without hiding tails."""

    close = torch.as_tensor(dense_close, dtype=torch.float64)
    one_step = torch.as_tensor(dense_log_return, dtype=torch.float64)
    last = torch.as_tensor(last_close, dtype=torch.float64).unsqueeze(1)
    if close.shape != one_step.shape:
        raise ValueError("Dense Close and one-step-return paths must match.")
    if tuple(last.shape[0:1] + last.shape[2:]) != tuple(
        close.shape[0:1] + close.shape[2:]
    ):
        raise ValueError("last_close axes are incompatible with dense rollout.")
    cumulative = torch.log(close) - torch.log(last)
    rows: list[dict[str, float | int]] = []
    for step in range(int(close.shape[1])):
        raw_values = close[:, step].reshape(-1)
        one_step_abs = one_step[:, step].abs().reshape(-1)
        cumulative_abs = cumulative[:, step].abs().reshape(-1)
        rows.append(
            {
                "step": step + 1,
                "mean_absolute_one_step_log_return": float(
                    one_step_abs.mean().item()
                ),
                "p95_absolute_one_step_log_return": float(
                    torch.quantile(one_step_abs, 0.95).item()
                ),
                "maximum_absolute_one_step_log_return": float(
                    one_step_abs.max().item()
                ),
                "mean_absolute_cumulative_log_change": float(
                    cumulative_abs.mean().item()
                ),
                "p95_absolute_cumulative_log_change": float(
                    torch.quantile(cumulative_abs, 0.95).item()
                ),
                "maximum_absolute_cumulative_log_change": float(
                    cumulative_abs.max().item()
                ),
                "minimum_predicted_close": float(raw_values.min().item()),
                "maximum_predicted_close": float(raw_values.max().item()),
            }
        )
    return rows


def _export_autoregressive_checkpoint(
    *,
    model: ModernTCNGraphRound1Model,
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
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    dense_rollouts: list[Tensor] = []
    dense_log_returns: list[Tensor] = []
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
            desc=f"export autoregressive {split_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            (
                predicted_raw,
                true_raw,
                last,
                _,
                dense,
                dense_log_return,
                first_output,
            ) = _autoregressive_errors(
                model=model,
                batch=batch,
                device=device,
                use_amp=use_amp,
                config=config,
            )
            predictions.append(predicted_raw.detach().cpu().float())
            targets.append(true_raw.detach().cpu().float())
            dense_rollouts.append(dense.detach().cpu().float())
            dense_log_returns.append(
                dense_log_return.detach().cpu().float()
            )
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
                torch.as_tensor(first_output.graph.selected)
                .detach()
                .cpu()
                .to(torch.float16)
                .contiguous()
            )
            dynamic_graphs.append(
                torch.as_tensor(first_output.graph.dynamic)
                .detach()
                .cpu()
                .to(torch.float16)
                .contiguous()
            )
            if first_output.graph.base is not None and singleton_static is None:
                singleton_static = (
                    torch.as_tensor(first_output.graph.base)
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
        "dense_autoregressive_close_path": torch.cat(dense_rollouts, dim=0),
        "dense_autoregressive_one_step_log_returns": torch.cat(
            dense_log_returns,
            dim=0,
        ),
        "dense_autoregressive_horizons": list(range(1, int(max(horizons)) + 1)),
        "autoregressive_input_channels": ["close"],
        "autoregressive_feedback_channels": ["close"],
        "autoregressive_output_representation": "one_step_cumulative_log_change",
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
        "alpha": None if model.alpha() is None else model.alpha().detach().cpu().float().reshape(1),
        "beta": model.beta().detach().cpu().float().reshape(1),
        "dynamic_alpha": None if model.alpha() is None else float(model.alpha().item()),
        "spatial_beta": float(model.beta().item()),
        "spatial_gate_type": str(config["model"]["spatial"]["gate_type"]),
        "beta_trainable": bool(model.spatial_gate.raw_beta is not None),
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "forecast_strategy": AUTOREGRESSIVE_CLOSE_ONLY,
    }
    selected_summary = graph_component_summary(selected.float())
    static_summary = graph_component_summary(None if saved_static is None else saved_static.float())
    dynamic_summary = graph_component_summary(dynamic.float())
    dense_close_all = prediction_result["dense_autoregressive_close_path"]
    dense_log_return_all = prediction_result[
        "dense_autoregressive_one_step_log_returns"
    ]
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(prediction_result["y_pred"].shape[0]),
        "horizons": horizons,
        "assets": int(prediction_result["y_pred"].shape[2]),
        "forecast_strategy": AUTOREGRESSIVE_CLOSE_ONLY,
        "input_channels": ["close"],
        "feedback_channels": ["close"],
        "output_representation": "one_step_cumulative_log_change",
        "rollout_step_diagnostics": _rollout_step_diagnostics(
            dense_close_all,
            dense_log_return_all,
            prediction_result["last_context_target"],
        ),
        "alpha": graph_artifacts["dynamic_alpha"],
        "beta": graph_artifacts["spatial_beta"],
        "spatial_gate_type": graph_artifacts["spatial_gate_type"],
        "beta_trainable": graph_artifacts["beta_trainable"],
        "selected_graph": selected_summary,
        "static_graph": static_summary,
        "dynamic_graph": dynamic_summary,
        "graph_orientation": GRAPH_ORIENTATION,
    }
    return {
        "prediction_result": prediction_result,
        "graph_artifacts": graph_artifacts,
        "metric_table": metric_table,
        "diagnostics": diagnostics,
    }


def _make_metadata(
    *,
    args: argparse.Namespace,
    resolved: Mapping[str, Any],
    model: ModernTCNGraphRound1Model,
    model_config: ModernTCNGraphRound1Config,
    datasets: Mapping[str, Dataset],
    train_split: Mapping[str, Any],
    validation_split: Mapping[str, Any],
    test_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    run_signature: str,
    device: torch.device,
    use_amp: bool,
    project_root: Path,
    created: str,
) -> dict[str, Any]:
    backbone_parameters, graph_parameters = _parameter_partition(model)
    return {
        "status": "running",
        "do_not_report": True,
        "test_set_contaminated": True,
        "selection_split": "test",
        "selection_metric": str(resolved["training"]["selection_metric"]),
        "created_at_utc": created,
        "run_name": args.run_name,
        "run_signature": run_signature,
        "project_git_commit": _git_value(["rev-parse", "HEAD"], cwd=project_root),
        "project_git_branch": _git_value(["branch", "--show-current"], cwd=project_root),
        "device": str(device),
        "mixed_precision": use_amp,
        "autoregressive_rollout_mixed_precision": (
            None
            if not _is_close_only_autoregressive(resolved)
            else _autoregressive_rollout_uses_amp(
                resolved,
                training_amp_enabled=use_amp,
            )
        ),
        "asset_cols": list(asset_cols),
        "train_sessions": len(train_split["samples"]),
        "validation_sessions": len(validation_split["samples"]),
        "test_sessions": len(test_split["samples"]),
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "one_step_train_windows": len(datasets.get("one_step_train", datasets["train"])),
        "context_length": int(resolved["data"]["context_length"]),
        "stride": int(resolved["data"]["stride"]),
        "horizons": [int(value) for value in resolved["data"]["horizons"]],
        "input_channels": [
            str(value) for value in resolved["data"]["input_channels"]
        ],
        "output_representation": str(
            resolved["model"].get("output_representation", "normalised_close")
        ),
        "forecast_strategy": _strategy(resolved),
        "autoregressive_feedback_channels": (
            ["close"]
            if _is_close_only_autoregressive(resolved)
            else None
        ),
        "model_family": "modern_tcn_graph_final_ablation",
        "optimisation_profile": str(resolved["training"].get("optimisation_profile", "round1")),
        "optimizer": str(resolved["training"]["optimizer"]),
        "parameter_grouping": str(resolved["training"].get("parameter_grouping", "split")),
        "scheduler": str(resolved["training"]["scheduler"]),
        "scheduler_t_max": int(resolved["training"].get("scheduler_t_max", 0)),
        "scheduler_decay_start_epoch": (
            None
            if resolved["training"].get("scheduler_decay_start_epoch") is None
            else int(resolved["training"]["scheduler_decay_start_epoch"])
        ),
        "scheduler_decay_factor": (
            None
            if resolved["training"].get("scheduler_decay_factor") is None
            else float(resolved["training"]["scheduler_decay_factor"])
        ),
        "weight_decay": float(resolved["training"]["weight_decay"]),
        "gradient_clip_norm": float(resolved["training"]["gradient_clip_norm"]),
        "temporal_backbone": "modern_tcn",
        "graph_variant": str(resolved["model"]["variant"]),
        "graph_type": str(resolved["model"]["graph"]["type"]),
        "graph_activation": str(
            resolved["model"]["graph"]["activation"]
        ),
        "graph_heads": int(resolved["model"]["graph"]["num_heads"]),
        "graph_hidden_dim": int(resolved["model"]["graph"]["hidden_dim"]),
        "prior_type": str(resolved["model"]["prior"]["type"]),
        "prior_scale": float(resolved["model"]["prior"]["scale"]),
        "prior_jitter": float(resolved["model"]["prior"]["jitter"]),
        "state_pathway": model_config.uses_state_pathway,
        "spatial_gate_type": str(resolved["model"]["spatial"]["gate_type"]),
        "spatial_initial_beta": float(resolved["model"]["spatial"]["initial_beta"]),
        "graph_initial_alpha": (
            float(resolved["model"]["graph"]["initial_alpha"])
            if model_config.uses_static_graph
            else None
        ),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "backbone_trainable_parameters": int(sum(parameter.numel() for parameter in backbone_parameters)),
        "graph_trainable_parameters": int(sum(parameter.numel() for parameter in graph_parameters)),
    }


def _build_static_prior(
    *,
    resolved: Mapping[str, Any],
    train_split: Mapping[str, Any],
    asset_cols: Sequence[str],
    company_profiles: Path | None,
) -> tuple[Tensor | None, list[str] | None]:
    prior_type = str(resolved["model"]["prior"]["type"])
    if prior_type in {"none", "random"}:
        return None, None
    if prior_type == "sector":
        if company_profiles is None:
            raise ValueError("A sector prior requires --company-profiles.")
        return build_sector_graph_prior(asset_cols, company_profiles)
    if prior_type == "correlation":
        return (
            build_absolute_correlation_graph_prior(
                train_split,
                expected_asset_cols=asset_cols,
            ),
            None,
        )
    raise ValueError(f"Unsupported prior type {prior_type!r}.")


def main() -> None:
    args = build_argument_parser().parse_args()
    resolved = _load_config(args.config)
    strategy = _strategy(resolved)
    if strategy not in {AUTOREGRESSIVE_CLOSE_ONLY, PARALLEL_WEIGHTED}:
        raise ValueError(
            "forecast_strategy must be 'autoregressive_close_only' or "
            "'parallel_weighted'."
        )

    training = resolved["training"]
    public_data_config = _dataset_config(resolved)
    train_data_config = _training_dataset_config(resolved)
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
    if list(train_split["asset_cols"]) != list(validation_split["asset_cols"]):
        raise ValueError("Train and validation asset order differs.")
    if list(train_split["asset_cols"]) != list(test_split["asset_cols"]):
        raise ValueError("Train and test asset order differs.")
    asset_cols = list(train_split["asset_cols"])

    public_datasets = {
        "train": build_continuous_dataset(train_split, config=public_data_config),
        "validation": build_continuous_dataset(validation_split, config=public_data_config),
        "test": build_continuous_dataset(test_split, config=public_data_config),
    }
    train_dataset = (
        build_continuous_dataset(train_split, config=train_data_config)
        if strategy == AUTOREGRESSIVE_CLOSE_ONLY
        else public_datasets["train"]
    )
    datasets: dict[str, Dataset] = dict(public_datasets)
    datasets["one_step_train"] = train_dataset
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(f"The configured {name} dataset has no windows.")

    static_prior, sectors = _build_static_prior(
        resolved=resolved,
        train_split=train_split,
        asset_cols=asset_cols,
        company_profiles=args.company_profiles,
    )
    model_config = _model_config_for_strategy(resolved, num_nodes=len(asset_cols))
    model = ModernTCNGraphRound1Model(model_config, static_prior=static_prior).to(device)
    optimizer = _build_optimizer(model, resolved)
    scaler = _new_grad_scaler(use_amp)

    signature_values = {
        "config": resolved,
        "model_horizons": list(model_config.forecaster.horizons),
        "train_dataset": {
            "context_length": train_data_config.context_length,
            "horizons": list(train_data_config.horizons),
            "stride": train_data_config.stride,
        },
        "asset_cols": asset_cols,
        "train_dates": [str(sample[2]) for sample in train_split["samples"]],
        "validation_dates": [str(sample[2]) for sample in validation_split["samples"]],
        "test_dates": [str(sample[2]) for sample in test_split["samples"]],
    }
    run_signature = _signature(signature_values)
    created = datetime.now(timezone.utc).isoformat()
    metadata = _make_metadata(
        args=args,
        resolved=resolved,
        model=model,
        model_config=model_config,
        datasets=datasets,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        asset_cols=asset_cols,
        run_signature=run_signature,
        device=device,
        use_amp=use_amp,
        project_root=project_root,
        created=created,
    )
    atomic_json_save(resolved, run_dir / "resolved_config.json")
    atomic_json_save(metadata, run_dir / "run_metadata.json")
    (run_dir / "DO_NOT_REPORT.txt").write_text(
        "This run uses the test split for checkpoint selection.\n",
        encoding="utf-8",
    )
    initial_static = model.graph_learner.static_adjacency()
    if initial_static is not None:
        prior_type = str(resolved["model"]["prior"]["type"])
        initial_adjacency = initial_static.detach().cpu().float()[0]
        initial_prior_payload = {
            "prior_type": prior_type,
            "adjacency": initial_adjacency,
            "static_logits": (
                None
                if model.graph_learner.static_logits is None
                else model.graph_learner.static_logits.detach().cpu().float()
            ),
            "asset_cols": asset_cols,
            "sectors": sectors,
            "orientation": GRAPH_ORIENTATION,
            "graph_activation": str(
                resolved["model"]["graph"]["activation"]
            ),
            "fitted_on": (
                "company_profiles.csv"
                if prior_type == "sector"
                else (
                    "canonical January-August training Close returns only"
                    if prior_type == "correlation"
                    else "random trainable static logits; no economic prior"
                )
            ),
        }
        atomic_torch_save(
            initial_prior_payload,
            run_dir / "initial_graph_prior.pt",
        )
        pd.DataFrame(
            initial_adjacency.mean(dim=0).numpy(),
            index=asset_cols,
            columns=asset_cols,
        ).to_csv(run_dir / "initial_graph_prior.csv")
        atomic_json_save(
            {
                key: value
                for key, value in initial_prior_payload.items()
                if key not in {"adjacency", "static_logits"}
            },
            run_dir / "initial_graph_prior.json",
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
        public_datasets["test"],
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
                if strategy == AUTOREGRESSIVE_CLOSE_ONLY:
                    train_values = _train_epoch_autoregressive_one_step(
                        model=model,
                        dataset=train_dataset,
                        device=device,
                        optimizer=optimizer,
                        scaler=scaler,
                        use_amp=use_amp,
                        config=resolved,
                        epoch=epoch,
                    )
                    selection = _evaluate_selection_autoregressive(
                        model=model,
                        loader=selection_loader,
                        device=device,
                        use_amp=use_amp,
                        config=resolved,
                        description=f"autoregressive test selection epoch {epoch}",
                    )
                else:
                    train_values = _train_epoch_weighted_parallel(
                        model=model,
                        dataset=train_dataset,
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
                        description=f"weighted-parallel test selection epoch {epoch}",
                    )
                score = float(selection["selection_score"])
                record = _history_record(
                    epoch=epoch,
                    learning_rates=current_lrs,
                    train=train_values,
                    selection=selection,
                    config=resolved,
                )
                record["forecast_strategy"] = strategy
                history.append(record)
                atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
                improved = score < best_score - min_delta
                if improved:
                    best_score = score
                    best_epoch = epoch
                    without_improvement = 0
                    best_checkpoint = _checkpoint(
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
                    atomic_torch_save(best_checkpoint, run_dir / "best_checkpoint.pt")
                else:
                    without_improvement += 1

                _advance_schedule(optimizer, training=training, completed_epoch=epoch)
                last_checkpoint = _checkpoint(
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
                atomic_torch_save(last_checkpoint, run_dir / "last_checkpoint.pt")
                if wandb_run is not None:
                    wandb_run.log(record, step=epoch)
                graph_lr = current_lrs.get("graph")
                print(
                    f"epoch={epoch} strategy={strategy} "
                    f"train={train_values['training_native_loss']:.8g} "
                    f"test_mean={score:.8g} best={best_score:.8g} "
                    f"best_epoch={best_epoch} alpha={selection['block_0_alpha']} "
                    f"beta={selection['block_0_beta']:.4f} "
                    f"backbone_lr={current_lrs['backbone']:.3g} "
                    f"graph_lr={graph_lr if graph_lr is not None else 'none'}"
                )
                if without_improvement >= patience:
                    print(f"Early stopping after epoch {epoch}.")
                    break

            if best_epoch <= 0 or not (run_dir / "best_checkpoint.pt").is_file():
                raise RuntimeError("Training produced no selected checkpoint.")
            training_complete = True
            final_last = _checkpoint(
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
            )
            atomic_torch_save(final_last, run_dir / "last_checkpoint.pt")

        best_checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        if best_checkpoint["run_signature"] != run_signature:
            raise ValueError("Best-checkpoint signature differs from requested run.")
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        model.to(device)
        best_epoch = int(best_checkpoint["best_epoch"])
        best_score = float(best_checkpoint["best_score"])

        split_values = {"train": train_split, "validation": validation_split, "test": test_split}
        for split_index, split_name in enumerate(("train", "validation", "test")):
            loader = _build_loader(
                public_datasets[split_name],
                batch_size=int(training["export_batch_size"]),
                shuffle=False,
                num_workers=int(training["num_workers"]),
                seed=seed + 1000 + split_index,
                pin_memory=device.type == "cuda",
            )
            if strategy == AUTOREGRESSIVE_CLOSE_ONLY:
                exported = _export_autoregressive_checkpoint(
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
            else:
                from src.training.run_modern_tcn_graph_round1 import _export_selected_checkpoint

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
                exported["diagnostics"]["forecast_strategy"] = strategy
            _save_export(run_dir, split_name=split_name, values=exported)
            if strategy == AUTOREGRESSIVE_CLOSE_ONLY:
                rollout_table = pd.DataFrame(
                    exported["diagnostics"]["rollout_step_diagnostics"]
                )
                rollout_path = (
                    run_dir
                    / f"best_{split_name}_rollout_step_diagnostics.csv"
                )
                atomic_csv_save(rollout_table, rollout_path)
                analysis_path = run_dir / "analysis" / split_name
                analysis_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(
                    rollout_path,
                    analysis_path / "rollout_step_diagnostics.csv",
                )

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": int(last_epoch),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                "final_alpha": None if model.alpha() is None else float(model.alpha().item()),
                "final_beta": float(model.beta().item()),
                "final_beta_trainable": bool(model.spatial_gate.raw_beta is not None),
            }
        )
        atomic_json_save(metadata, run_dir / "run_metadata.json")
        print("FINAL MODERNTCN GRAPH ABLATION COMPLETE")
        print("Run:", args.run_name)
        print("Strategy:", strategy)
        print("Best epoch:", best_epoch)
        print("Best test five-horizon mean Log MAE:", best_score)
    except BaseException as error:
        failed = dict(metadata)
        failed.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "best_epoch": int(best_epoch),
                "best_score": None if not np.isfinite(best_score) else float(best_score),
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

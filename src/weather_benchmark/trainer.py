from __future__ import annotations

"""Training, resume, early stopping and artifact export for weather runs."""

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.dense_transformer_depth_sweep import (
    GRAPH_ORIENTATION,
    DenseTransformerDepthSequenceOutput,
    StackedDenseTransformerGraphModel,
)
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Model,
    ModernTCNGraphRound1Output,
    graph_component_summary,
)

from .artifacts import (
    atomic_history_save,
    atomic_json_save,
    atomic_metric_csv,
    atomic_torch_save,
    capture_rng_state,
    environment_manifest,
    restore_rng_state,
    safe_torch_load,
    utc_now,
)
from .config import CENTRAL_NODE_INDEX, WEATHER_FEATURES, WEATHER_NODES, WeatherRunConfig
from .data import SonnetWeatherDataBundle
from .metrics import weather_metric_payload
from .models import (
    WeatherModelBundle,
    graph_parameter_ids,
    model_alphas,
    model_betas,
    parameter_counts,
)


@dataclass
class TrainingResult:
    run_directory: Path
    best_epoch: int
    best_validation_score: float
    stopped_early: bool
    completed: bool
    test_metrics: dict[str, Any]


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        # Improve repeatability for the controlled weather sweep without
        # enabling strict deterministic-algorithm errors in the imported
        # ModernTCN implementation.  The resolved flag is saved in every run.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        # Prefer deterministic implementations where PyTorch can provide them.
        # ``warn_only`` keeps third-party ModernTCN operations usable while
        # surfacing any operation for which deterministic execution is not
        # available on the current Colab stack.
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def _signature(values: Mapping[str, Any]) -> str:
    serialised = json.dumps(dict(values), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def _amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _new_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _build_optimizer(model: nn.Module, config: WeatherRunConfig) -> torch.optim.Optimizer:
    graph_ids = graph_parameter_ids(model)
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
    if not graph or not backbone:
        raise RuntimeError("Both graph and backbone optimizer groups must be non-empty.")
    return torch.optim.Adam(
        [
            {
                "params": backbone,
                "lr": float(config.backbone_learning_rate),
                "base_lr": float(config.backbone_learning_rate),
                "name": "backbone",
            },
            {
                "params": graph,
                "lr": float(config.graph_learning_rate),
                "base_lr": float(config.graph_learning_rate),
                "name": "graph",
            },
        ],
        weight_decay=float(config.weight_decay),
    )


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        result[name] = float(group["lr"])
    return result


def _advance_delayed_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    completed_epoch: int,
    decay_start_epoch: int,
    decay_factor: float,
) -> None:
    multiplier = (
        1.0
        if int(completed_epoch) < int(decay_start_epoch)
        else float(decay_factor)
        ** (int(completed_epoch) - int(decay_start_epoch) + 1)
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return total**0.5


class WeightedScalarAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def add(self, name: str, value: float, *, weight: float = 1.0) -> None:
        if not math.isfinite(float(value)):
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(value) * float(weight)
        self.weights[name] = self.weights.get(name, 0.0) + float(weight)

    def update(self, values: Mapping[str, float], *, weight: float = 1.0) -> None:
        for name, value in values.items():
            self.add(name, float(value), weight=weight)

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.weights[name]
            for name in self.sums
            if self.weights.get(name, 0.0) > 0.0
        }


def _graph_summary(values: Tensor) -> dict[str, float]:
    summary = graph_component_summary(torch.as_tensor(values).detach().float())
    return {
        name: float(value)
        for name, value in summary.items()
        if value is not None
    }


def _rms(values: Tensor) -> float:
    tensor = torch.as_tensor(values).detach().float()
    return float(torch.sqrt(torch.mean(tensor.square())).item())


def _batch_model_diagnostics(
    output: ModernTCNGraphRound1Output | DenseTransformerDepthSequenceOutput,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if isinstance(output, ModernTCNGraphRound1Output):
        selected = output.graph.selected
        dynamic = output.graph.dynamic
        if dynamic is None:
            raise RuntimeError("ModernTCN output is missing its dynamic graph.")
        for prefix, graph in (("selected", selected), ("dynamic", dynamic)):
            for name, value in _graph_summary(graph).items():
                values[f"graph_{prefix}_{name}"] = value
        if output.graph.base is not None:
            base = output.graph.base
            for name, value in _graph_summary(base).items():
                values[f"graph_static_{name}"] = value
            expanded = base.expand(selected.shape[0], -1, -1, -1)
            values["graph_dynamic_static_l1"] = float(
                torch.mean(torch.abs(dynamic.float() - expanded.float())).item()
            )
        values["temporal_hidden_rms"] = _rms(output.temporal_hidden)
        values["graph_spatial_hidden_rms"] = _rms(output.graph_spatial_hidden)
        values["fused_hidden_rms"] = _rms(output.fused_hidden)
        return values

    if isinstance(output, DenseTransformerDepthSequenceOutput):
        for block_index, block in enumerate(output.block_outputs, start=1):
            selected = block.graph.selected[:, -1]
            dynamic = block.graph.dynamic[:, -1]
            base = block.graph.base
            for prefix, graph in (
                ("selected", selected),
                ("dynamic", dynamic),
                ("static", base),
            ):
                for name, value in _graph_summary(graph).items():
                    values[f"block{block_index}_graph_{prefix}_{name}"] = value
            expanded = base.view(1, *base.shape[1:]).expand(
                selected.shape[0], -1, -1, -1
            )
            values[f"block{block_index}_graph_dynamic_static_l1"] = float(
                torch.mean(torch.abs(dynamic.float() - expanded.float())).item()
            )
            values[f"block{block_index}_temporal_hidden_rms"] = _rms(
                block.temporal_hidden
            )
            values[f"block{block_index}_graph_spatial_hidden_rms"] = _rms(
                block.graph_spatial_hidden
            )
            values[f"block{block_index}_fused_hidden_rms"] = _rms(
                block.fused_hidden
            )
        return values
    raise TypeError(type(output))


def _final_predictions(
    output: ModernTCNGraphRound1Output | DenseTransformerDepthSequenceOutput,
) -> Tensor:
    if isinstance(output, ModernTCNGraphRound1Output):
        return output.predictions
    if isinstance(output, DenseTransformerDepthSequenceOutput):
        return output.final_predictions()
    raise TypeError(type(output))


def _forward_model(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> ModernTCNGraphRound1Output | DenseTransformerDepthSequenceOutput:
    x = torch.as_tensor(batch["x"]).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    if isinstance(model, ModernTCNGraphRound1Model):
        return model(
            x,
            context_start=torch.as_tensor(batch["context_start"]),
            session_length=torch.as_tensor(batch["session_length"]),
        )
    if isinstance(model, StackedDenseTransformerGraphModel):
        return model.forward_dense(x)
    raise TypeError(type(model))


def _training_loss(
    output: ModernTCNGraphRound1Output | DenseTransformerDepthSequenceOutput,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(output, ModernTCNGraphRound1Output):
        target = torch.as_tensor(batch["y"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        prediction = output.predictions
        loss = F.mse_loss(prediction.float(), target.float())
        final_pred = prediction[:, -1, CENTRAL_NODE_INDEX, 0]
        final_true = target[:, -1, CENTRAL_NODE_INDEX, 0]
        return loss, final_pred, final_true
    if isinstance(output, DenseTransformerDepthSequenceOutput):
        target = torch.as_tensor(batch["dense_y"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        prediction = output.predictions
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Dense prediction/target shape mismatch: {prediction.shape} vs {target.shape}."
            )
        loss = F.mse_loss(prediction.float(), target.float())
        final_pred = prediction[:, -1, -1, CENTRAL_NODE_INDEX, 0]
        final_true = target[:, -1, -1, CENTRAL_NODE_INDEX, 0]
        return loss, final_pred, final_true
    raise TypeError(type(output))


def _loader_worker_options(config: WeatherRunConfig) -> dict[str, Any]:
    if int(config.num_workers) <= 0:
        return {}
    return {
        "persistent_workers": False,
        "prefetch_factor": int(config.prefetch_factor),
    }


def _make_train_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    config: WeatherRunConfig,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + int(epoch))
    return DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory and torch.cuda.is_available()),
        drop_last=False,
        **_loader_worker_options(config),
    )


def _make_eval_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    config: WeatherRunConfig,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        pin_memory=bool(config.pin_memory and torch.cuda.is_available()),
        drop_last=False,
        **_loader_worker_options(config),
    )


def _train_epoch(
    *,
    model: nn.Module,
    dataset: Dataset[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: WeatherRunConfig,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    loader = _make_train_loader(dataset, config=config, epoch=epoch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    diagnostics = WeightedScalarAccumulator()
    loss_sum = 0.0
    loss_count = 0
    final_sse = 0.0
    final_count = 0
    optimizer_steps = 0

    graph_ids = graph_parameter_ids(model)
    graph_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in graph_ids
    ]
    diagnostic_taken = False

    progress = tqdm(
        loader,
        desc=f"epoch {epoch:03d} train",
        leave=False,
        dynamic_ncols=True,
    )
    total_batches = len(loader)
    update_interval = max(1, int(config.progress_update_interval))
    for batch_index, batch in enumerate(progress, start=1):
        optimizer.zero_grad(set_to_none=True)
        batch_size = int(torch.as_tensor(batch["x"]).shape[0])
        with _amp_context(device, use_amp):
            output = _forward_model(model, batch, device=device)
            loss, final_pred, final_true = _training_loss(
                output, batch, device=device
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        graph_grad = (
            _gradient_norm(graph_parameters) if not diagnostic_taken else None
        )
        total_grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.gradient_clip_norm)
        )
        if not diagnostic_taken:
            diagnostics.update(_batch_model_diagnostics(output), weight=1.0)
            diagnostics.add(
                "graph_gradient_norm",
                float(graph_grad if graph_grad is not None else 0.0),
                weight=1.0,
            )
            diagnostics.add(
                "total_gradient_norm_before_clip",
                float(torch.as_tensor(total_grad).detach().item()),
                weight=1.0,
            )
            diagnostic_taken = True
        scaler.step(optimizer)
        scaler.update()
        optimizer_steps += 1

        loss_sum += float(loss.detach().item()) * batch_size
        loss_count += batch_size
        final_sse += float(
            torch.sum((final_pred.detach().float() - final_true.detach().float()) ** 2).item()
        )
        final_count += int(final_pred.numel())
        if batch_index % update_interval == 0 or batch_index == total_batches:
            progress.set_postfix(loss=f"{float(loss.detach().item()):.5f}")

    if loss_count <= 0 or final_count <= 0:
        raise RuntimeError("Training loader produced no examples.")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_duration = max(time.perf_counter() - train_started, 1.0e-12)
    result = {
        "train_loss": loss_sum / loss_count,
        "train_central_final_horizon_mse": final_sse / final_count,
        "train_duration_seconds": train_duration,
        "train_examples_per_second": float(loss_count) / train_duration,
        "train_optimizer_steps": float(optimizer_steps),
        "train_batches_per_second": float(optimizer_steps) / train_duration,
        **diagnostics.means(),
    }
    if device.type == "cuda":
        result["train_peak_cuda_memory_allocated_gib"] = float(
            torch.cuda.max_memory_allocated(device) / (1024**3)
        )
        result["train_peak_cuda_memory_reserved_gib"] = float(
            torch.cuda.max_memory_reserved(device) / (1024**3)
        )
    for index, alpha in enumerate(model_alphas(model), start=1):
        result[f"alpha_block_{index}"] = float(alpha.detach().item())
    for index, beta in enumerate(model_betas(model), start=1):
        result[f"beta_block_{index}"] = float(beta.detach().item())
    return result


def _validate_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    data: SonnetWeatherDataBundle,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    sequence_sse = 0.0
    sequence_count = 0
    final_sse = 0.0
    final_count = 0
    final_pred_raw: list[np.ndarray] = []
    final_true_raw: list[np.ndarray] = []
    diagnostics = WeightedScalarAccumulator()

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"epoch {epoch:03d} validation",
            leave=False,
            dynamic_ncols=True,
        ):
            batch_size = int(torch.as_tensor(batch["x"]).shape[0])
            target = torch.as_tensor(batch["y"]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            with _amp_context(device, use_amp):
                output = _forward_model(model, batch, device=device)
            prediction = _final_predictions(output).float()
            difference = prediction - target.float()
            sequence_sse += float(torch.sum(difference.square()).item())
            sequence_count += int(difference.numel())
            central_pred = prediction[:, -1, CENTRAL_NODE_INDEX, 0]
            central_true = target[:, -1, CENTRAL_NODE_INDEX, 0]
            final_sse += float(torch.sum((central_pred - central_true).square()).item())
            final_count += int(central_pred.numel())

            target_scale = float(data.target_scale[CENTRAL_NODE_INDEX])
            target_mean = float(data.target_mean[CENTRAL_NODE_INDEX])
            final_pred_raw.append(
                (central_pred.detach().cpu().numpy() * target_scale + target_mean)
                .astype(np.float32)
            )
            final_true_raw.append(
                (central_true.detach().cpu().numpy() * target_scale + target_mean)
                .astype(np.float32)
            )
            diagnostics.update(_batch_model_diagnostics(output), weight=batch_size)

    if sequence_count <= 0 or final_count <= 0:
        raise RuntimeError("Validation loader produced no examples.")
    raw_metrics = weather_metric_payload(
        predictions=np.concatenate(final_pred_raw)[:, None, None, None],
        targets=np.concatenate(final_true_raw)[:, None, None, None],
        central_node_index=0,
    )["reported"]
    return {
        "validation_sequence_all_node_mse": sequence_sse / sequence_count,
        "validation_central_final_horizon_mse": final_sse / final_count,
        "validation_central_final_horizon_mae_kelvin": float(raw_metrics["mae"]),
        "validation_central_final_horizon_r": float(raw_metrics["r"]),
        "validation_central_final_horizon_smape": float(raw_metrics["smape"]),
        **{f"validation_{name}": value for name, value in diagnostics.means().items()},
    }


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: WeatherRunConfig,
    run_signature: str,
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "resolved_config": config.to_dict(),
        "run_signature": run_signature,
        "best_validation_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "history": history,
        "rng_state": capture_rng_state(),
        "saved_at_utc": utc_now(),
    }


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    expected_signature: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location=device)
    if str(payload.get("run_signature")) != str(expected_signature):
        raise RuntimeError(
            f"Checkpoint signature differs from the resolved run: {path}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("grad_scaler_state_dict") is not None:
        scaler.load_state_dict(payload["grad_scaler_state_dict"])
    if payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])
    return payload


def _prepare_run_metadata(
    *,
    run_dir: Path,
    config: WeatherRunConfig,
    model_bundle: WeatherModelBundle,
    data: SonnetWeatherDataBundle,
    device: torch.device,
    project_root: Path,
    run_signature: str,
) -> dict[str, Any]:
    values = {
        "status": "running",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "run_signature": run_signature,
        "selection_split": "validation",
        "selection_metric": "central-node final-origin final-horizon normalised MSE",
        "test_used_for_selection": False,
        "reported_test_metrics": ["mae", "r", "smape"],
        "reported_scope": (
            f"central {config.city} T850 at the final forecast position"
        ),
        "model": model_bundle.model_config,
        "training": {
            "optimizer": "Adam",
            "backbone_learning_rate": float(config.backbone_learning_rate),
            "graph_learning_rate": float(config.graph_learning_rate),
            "weight_decay": float(config.weight_decay),
            "batch_size": int(config.batch_size),
            "validation_batch_size": int(config.validation_batch_size),
            "export_batch_size": int(config.export_batch_size),
            "max_epochs": int(config.max_epochs),
            "patience": int(config.patience),
            "min_delta": float(config.min_delta),
            "gradient_clip_norm": float(config.gradient_clip_norm),
            "mixed_precision": bool(config.mixed_precision and device.type == "cuda"),
            "scheduler": "modern_tcn_type3_delayed",
            "scheduler_decay_start_epoch": int(config.scheduler_decay_start_epoch),
            "scheduler_decay_factor": float(config.scheduler_decay_factor),
            "dense_prefix_training": bool(config.dense_prefix_training),
            "num_workers": int(config.num_workers),
            "prefetch_factor": (
                int(config.prefetch_factor) if int(config.num_workers) > 0 else None
            ),
            "progress_update_interval": int(config.progress_update_interval),
            "cache_causal_masks": bool(config.cache_causal_masks),
            "deterministic_runtime": bool(config.deterministic_runtime),
            "torch_deterministic_algorithms_enabled": (
                bool(torch.are_deterministic_algorithms_enabled())
                if hasattr(torch, "are_deterministic_algorithms_enabled")
                else None
            ),
            "loss": "normalised-space MSE over all output steps and all nodes",
            "dense_prefix_scope": (
                "all context origins, all forecast steps, all nodes"
                if config.dense_prefix_training
                else None
            ),
            "training_diagnostic_scope": (
                "losses use every batch; gradient and hidden/graph diagnostics "
                "use the first deterministic training batch of each epoch"
            ),
            "seed": int(config.seed),
        },
        "data_summary": data.manifest(),
        "parameter_counts": parameter_counts(model_bundle.model),
        "environment": environment_manifest(project_root, device),
    }
    atomic_json_save(values, run_dir / "metadata.json")
    return values


def _save_initial_artifacts(
    *,
    run_dir: Path,
    config: WeatherRunConfig,
    model_bundle: WeatherModelBundle,
    data: SonnetWeatherDataBundle,
    project_root: Path,
    device: torch.device,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_save(config.to_dict(), run_dir / "resolved_config.json")
    data.save_data_artifacts(run_dir)
    atomic_json_save(
        parameter_counts(model_bundle.model), run_dir / "parameter_counts.json"
    )
    atomic_json_save(
        environment_manifest(project_root, device), run_dir / "environment.json"
    )
    atomic_torch_save(
        model_bundle.initial_graph_payload,
        run_dir / "initial_graphs.pt",
    )


def _split_graph_artifacts(
    *,
    model: nn.Module,
    selected_lists: list[list[Tensor]],
    dynamic_lists: list[list[Tensor]],
    singleton_static: list[Tensor | None],
    prediction_result: Mapping[str, Any],
) -> dict[str, Any]:
    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    per_layer_base = tuple(
        None if value is None else value[0].contiguous()
        for value in singleton_static
    )
    alphas = tuple(value.detach().cpu().float().reshape(1) for value in model_alphas(model))
    betas = torch.stack(
        [value.detach().cpu().float().reshape(()) for value in model_betas(model)]
    )
    final_alpha = None if not alphas else alphas[-1]
    final_beta = betas[-1:].contiguous()
    if isinstance(model, ModernTCNGraphRound1Model):
        graph_activations = [str(model.config.forecaster.graph.activation)]
        graph_hidden_dims = [int(model.config.forecaster.graph.hidden_dim)]
    elif isinstance(model, StackedDenseTransformerGraphModel):
        graph_activations = [
            str(value) for value in model.config.graph_activations_per_block
        ]
        graph_hidden_dims = [
            int(value) for value in model.config.graph_hidden_dims_per_block
        ]
    else:
        raise TypeError(type(model))
    return {
        "graph_type": "static_dynamic_mixture",
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(WEATHER_NODES),
        "node_order": list(WEATHER_NODES),
        "num_layers": len(per_layer_selected),
        "num_heads": int(per_layer_selected[-1].shape[1]),
        "num_heads_per_layer": [int(value.shape[1]) for value in per_layer_selected],
        "layer_head_counts": [int(value.shape[1]) for value in per_layer_selected],
        "graph_hidden_dims_per_layer": graph_hidden_dims,
        "graph_activations_per_layer": graph_activations,
        "selected_layer": len(per_layer_selected) - 1,
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
        "dynamic_alpha": None if final_alpha is None else float(final_alpha.item()),
        "spatial_beta": float(final_beta.item()),
        "spatial_gate_type": "learned_scalar",
        "beta_trainable": True,
        "dates": list(prediction_result["forecast_origin_times_iso"]),
        "forecast_origin_times_ns": prediction_result["forecast_origin_times_ns"],
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "target_times_ns": prediction_result["target_times_ns"],
    }


def _export_split(
    *,
    model: nn.Module,
    dataset: Dataset[dict[str, Any]],
    split_name: str,
    config: WeatherRunConfig,
    data: SonnetWeatherDataBundle,
    device: torch.device,
    use_amp: bool,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.eval()
    loader = _make_eval_loader(
        dataset,
        batch_size=int(config.export_batch_size),
        config=config,
    )
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    last_values: list[Tensor] = []
    sample_indices: list[Tensor] = []
    origin_indices: list[Tensor] = []
    forecast_origin_times: list[Tensor] = []
    target_indices: list[Tensor] = []
    target_times: list[Tensor] = []

    block_count = 1 if isinstance(model, ModernTCNGraphRound1Model) else 3
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
            with _amp_context(device, use_amp):
                output = _forward_model(model, batch, device=device)
            prediction_norm = _final_predictions(output).float()
            target_norm = torch.as_tensor(batch["y"]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            prediction_raw = data.inverse_target_tensor(prediction_norm)
            target_raw = data.inverse_target_tensor(target_norm)
            predictions.append(prediction_raw.detach().cpu().float())
            targets.append(target_raw.detach().cpu().float())
            last_values.append(
                torch.as_tensor(batch["last_context_target"]).cpu().float()
            )
            sample_indices.append(torch.as_tensor(batch["sample_idx"]).cpu().long())
            origin_indices.append(
                torch.as_tensor(batch["forecast_origin_index"]).cpu().long()
            )
            forecast_origin_times.append(
                torch.as_tensor(batch["forecast_origin_time_ns"]).cpu().long()
            )
            target_indices.append(
                torch.as_tensor(batch["target_indices"]).cpu().long()
            )
            target_times.append(torch.as_tensor(batch["target_times_ns"]).cpu().long())

            if isinstance(output, ModernTCNGraphRound1Output):
                selected_lists[0].append(
                    output.graph.selected.detach().cpu().to(torch.float16).contiguous()
                )
                if output.graph.dynamic is None:
                    raise RuntimeError("ModernTCN export is missing dynamic graphs.")
                dynamic_lists[0].append(
                    output.graph.dynamic.detach().cpu().to(torch.float16).contiguous()
                )
                if singleton_static[0] is None and output.graph.base is not None:
                    singleton_static[0] = (
                        output.graph.base.detach().cpu().to(torch.float16).contiguous()
                    )
            elif isinstance(output, DenseTransformerDepthSequenceOutput):
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
            else:
                raise TypeError(type(output))

    y_pred = torch.cat(predictions, dim=0)
    y_true = torch.cat(targets, dim=0)
    origins_ns = torch.cat(forecast_origin_times, dim=0)
    origin_iso = [
        pd.Timestamp(int(value), unit="ns").isoformat()
        for value in origins_ns.tolist()
    ]
    prediction_result: dict[str, Any] = {
        "y_pred": y_pred,
        "y_true": y_true,
        "last_context_target": torch.cat(last_values, dim=0),
        "channels": ["t850"],
        "horizons": list(range(1, int(config.horizon) + 1)),
        "asset_cols": list(WEATHER_NODES),
        "node_order": list(WEATHER_NODES),
        "central_node_index": CENTRAL_NODE_INDEX,
        "sample_idx": torch.cat(sample_indices, dim=0),
        "origin_idx": torch.cat(origin_indices, dim=0),
        "forecast_origin_times_ns": origins_ns,
        "forecast_origin_times_iso": origin_iso,
        "target_indices": torch.cat(target_indices, dim=0),
        "target_times_ns": torch.cat(target_times, dim=0),
        "output_space": "raw_kelvin",
        "sampling_frequency_hours": 6,
    }
    metric_payload = weather_metric_payload(
        predictions=y_pred.numpy(),
        targets=y_true.numpy(),
        central_node_index=CENTRAL_NODE_INDEX,
    )
    graph_artifacts = _split_graph_artifacts(
        model=model,
        selected_lists=selected_lists,
        dynamic_lists=dynamic_lists,
        singleton_static=singleton_static,
        prediction_result=prediction_result,
    )

    layer_summaries: list[dict[str, Any]] = []
    for index, (selected, dynamic, base) in enumerate(
        zip(
            graph_artifacts["per_layer"],
            graph_artifacts["per_layer_dynamic"],
            graph_artifacts["per_layer_base"],
            strict=True,
        ),
        start=1,
    ):
        layer_summaries.append(
            {
                "block": index,
                "alpha": float(graph_artifacts["alpha_per_layer"][index - 1].item()),
                "beta": float(graph_artifacts["beta_per_layer"][index - 1].item()),
                "selected_graph": graph_component_summary(selected.float()),
                "dynamic_graph": graph_component_summary(dynamic.float()),
                "static_graph": graph_component_summary(
                    None if base is None else base.float()
                ),
            }
        )
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": int(y_pred.shape[0]),
        "forecast_length": int(y_pred.shape[1]),
        "nodes": int(y_pred.shape[2]),
        "graph_orientation": GRAPH_ORIENTATION,
        "saved_graph_position": "final context position / forecast origin",
        "layers": layer_summaries,
        "metrics": metric_payload,
    }
    return {
        "prediction_result": prediction_result,
        "graph_artifacts": graph_artifacts,
        "metrics": metric_payload,
        "diagnostics": diagnostics,
    }


def _save_split_export(
    *,
    run_dir: Path,
    split_name: str,
    checkpoint_epoch: int,
    values: Mapping[str, Any],
) -> None:
    prediction_path = run_dir / f"best_{split_name}_predictions.pt"
    graph_path = run_dir / f"best_{split_name}_graphs.pt"
    metrics_path = run_dir / f"best_{split_name}_metrics.json"
    metric_table_path = run_dir / f"best_{split_name}_metric_table.csv"
    diagnostics_path = run_dir / f"best_{split_name}_diagnostics.json"

    atomic_torch_save(
        {
            "epoch": int(checkpoint_epoch),
            "prediction_result": values["prediction_result"],
        },
        prediction_path,
    )
    atomic_torch_save(
        {
            "epoch": int(checkpoint_epoch),
            "graph_artifacts": values["graph_artifacts"],
        },
        graph_path,
    )
    atomic_json_save(values["metrics"], metrics_path)
    atomic_metric_csv(values["metrics"], metric_table_path)
    atomic_json_save(values["diagnostics"], diagnostics_path)

    analysis_dir = run_dir / "analysis" / split_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction_path, analysis_dir / "predictions.pt")
    shutil.copy2(graph_path, analysis_dir / "graphs.pt")
    shutil.copy2(metrics_path, analysis_dir / "metrics.json")
    shutil.copy2(metric_table_path, analysis_dir / "metric_table.csv")
    shutil.copy2(diagnostics_path, analysis_dir / "diagnostics.json")


def train_weather_model(
    *,
    config: WeatherRunConfig,
    data: SonnetWeatherDataBundle,
    model_bundle: WeatherModelBundle,
    project_root: Path,
) -> TrainingResult:
    run_dir = config.run_directory
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints / "best.pt"
    last_path = checkpoints / "last.pt"
    completion_path = run_dir / "run_complete.json"
    run_signature = _signature(
        {
            "resolved_config": config.to_dict(),
            "data_manifest": data.manifest(),
            "model_config": model_bundle.model_config,
        }
    )

    if completion_path.is_file() and config.skip_completed and not config.overwrite:
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if str(completed.get("run_signature")) != run_signature:
            raise FileExistsError(
                "The completed run directory belongs to a different resolved "
                f"experiment: {run_dir}. Use a different output root or set "
                "overwrite=True deliberately."
            )
        return TrainingResult(
            run_directory=run_dir,
            best_epoch=int(completed["best_epoch"]),
            best_validation_score=float(completed["best_validation_score"]),
            stopped_early=bool(completed["stopped_early"]),
            completed=True,
            test_metrics=dict(completed.get("test_metrics", {})),
        )

    if config.overwrite and run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        checkpoints.mkdir(parents=True, exist_ok=True)
    elif not config.resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError(
            f"Run directory already contains checkpoints and resume=False: {run_dir}"
        )

    device = resolve_device(config.device)
    use_amp = bool(config.mixed_precision) and device.type == "cuda"
    set_seed(
        config.seed,
        deterministic=bool(config.deterministic_runtime),
    )

    model = model_bundle.model.to(device)
    optimizer = _build_optimizer(model, config)
    scaler = _new_grad_scaler(use_amp)
    _save_initial_artifacts(
        run_dir=run_dir,
        config=config,
        model_bundle=model_bundle,
        data=data,
        project_root=project_root,
        device=device,
    )
    metadata = _prepare_run_metadata(
        run_dir=run_dir,
        config=config,
        model_bundle=model_bundle,
        data=data,
        device=device,
        project_root=project_root,
        run_signature=run_signature,
    )

    training_dataset = data.dataset(
        "train", dense_prefix=bool(config.dense_prefix_training)
    )
    validation_dataset = data.dataset("validation", dense_prefix=False)
    validation_loader = _make_eval_loader(
        validation_dataset,
        batch_size=int(config.validation_batch_size),
        config=config,
    )

    start_epoch = 1
    best_score = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    stopped_early = False

    if config.resume and last_path.is_file():
        payload = _load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_signature=run_signature,
            device=device,
        )
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_validation_score"])
        best_epoch = int(payload["best_epoch"])
        bad_epochs = int(payload["bad_epochs"])
        history = [dict(value) for value in payload.get("history", [])]
        metadata["resumed_at_utc"] = utc_now()
        metadata["resumed_from_epoch"] = int(payload["epoch"])
        atomic_json_save(metadata, run_dir / "metadata.json")
        if bad_epochs >= int(config.patience):
            stopped_early = True

    try:
        epoch_range = (
            range(0)
            if stopped_early
            else range(start_epoch, int(config.max_epochs) + 1)
        )
        for epoch in epoch_range:
            epoch_started = time.perf_counter()
            rates_used = _learning_rates(optimizer)
            train_values = _train_epoch(
                model=model,
                dataset=training_dataset,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
            )
            validation_values = _validate_epoch(
                model=model,
                loader=validation_loader,
                data=data,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
            )
            selection_score = float(
                validation_values["validation_central_final_horizon_mse"]
            )
            improved = selection_score < best_score - float(config.min_delta)
            if improved:
                best_score = selection_score
                best_epoch = int(epoch)
                bad_epochs = 0
            else:
                bad_epochs += 1

            _advance_delayed_schedule(
                optimizer,
                completed_epoch=epoch,
                decay_start_epoch=int(config.scheduler_decay_start_epoch),
                decay_factor=float(config.scheduler_decay_factor),
            )
            rates_next = _learning_rates(optimizer)
            epoch_values: dict[str, Any] = {
                "epoch": int(epoch),
                "epoch_duration_seconds": time.perf_counter() - epoch_started,
                "selection_score": selection_score,
                "selection_improved": bool(improved),
                "best_validation_score": float(best_score),
                "best_epoch": int(best_epoch),
                "early_stopping_bad_epochs": int(bad_epochs),
                "learning_rate_backbone_used": rates_used.get("backbone"),
                "learning_rate_graph_used": rates_used.get("graph"),
                "learning_rate_backbone_next": rates_next.get("backbone"),
                "learning_rate_graph_next": rates_next.get("graph"),
                **train_values,
                **validation_values,
            }
            history.append(epoch_values)
            atomic_history_save(history, run_dir)

            checkpoint = _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                run_signature=run_signature,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                history=history,
            )
            atomic_torch_save(checkpoint, last_path)
            if improved:
                atomic_torch_save(checkpoint, best_path)

            metadata.update(
                {
                    "updated_at_utc": utc_now(),
                    "last_completed_epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_validation_score": float(best_score),
                    "early_stopping_bad_epochs": int(bad_epochs),
                }
            )
            atomic_json_save(metadata, run_dir / "metadata.json")
            print(
                f"[{config.model_kind} | {config.city} | H={config.horizon}] "
                f"epoch={epoch} train={train_values['train_loss']:.6f} "
                f"val_final={selection_score:.6f} best={best_score:.6f} "
                f"bad={bad_epochs}/{config.patience}"
            )
            if bad_epochs >= int(config.patience):
                stopped_early = True
                break

        if not best_path.is_file():
            raise RuntimeError("Training completed without a best checkpoint.")

        best_payload = _load_checkpoint(
            best_path,
            model=model,
            optimizer=None,
            scaler=None,
            expected_signature=run_signature,
            device=device,
        )
        best_epoch = int(best_payload["best_epoch"])
        best_score = float(best_payload["best_validation_score"])

        split_names = ["validation", "test"]
        if config.export_train_split:
            split_names.insert(0, "train")
        exported_metrics: dict[str, Any] = {}
        for split_name in split_names:
            dataset = data.dataset(split_name, dense_prefix=False)  # type: ignore[arg-type]
            values = _export_split(
                model=model,
                dataset=dataset,
                split_name=split_name,
                config=config,
                data=data,
                device=device,
                use_amp=use_amp,
                checkpoint_epoch=best_epoch,
            )
            _save_split_export(
                run_dir=run_dir,
                split_name=split_name,
                checkpoint_epoch=best_epoch,
                values=values,
            )
            exported_metrics[split_name] = values["metrics"]

        test_metrics = dict(exported_metrics["test"])
        completion = {
            "status": "completed",
            "completed_at_utc": utc_now(),
            "run_signature": run_signature,
            "model_kind": config.model_kind,
            "city": config.city,
            "test_year": int(config.test_year),
            "horizon": int(config.horizon),
            "context_length": int(config.context_length),
            "run_suffix": config.run_suffix,
            "modern_tcn_large_kernel": (
                int(config.modern_tcn_large_kernel)
                if config.model_kind == "modern_tcn_1st"
                else None
            ),
            "train_batch_size": int(config.batch_size),
            "validation_batch_size": int(config.validation_batch_size),
            "export_batch_size": int(config.export_batch_size),
            "best_epoch": int(best_epoch),
            "best_validation_score": float(best_score),
            "stopped_early": bool(stopped_early),
            "selection_split": "validation",
            "test_used_for_selection": False,
            "test_metrics": test_metrics,
            "run_directory": str(run_dir),
        }
        atomic_json_save(completion, completion_path)
        metadata.update(completion)
        metadata["status"] = "completed"
        atomic_json_save(metadata, run_dir / "metadata.json")
        return TrainingResult(
            run_directory=run_dir,
            best_epoch=best_epoch,
            best_validation_score=best_score,
            stopped_early=stopped_early,
            completed=True,
            test_metrics=test_metrics,
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        atomic_json_save(metadata, run_dir / "metadata.json")
        raise

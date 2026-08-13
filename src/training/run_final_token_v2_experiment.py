from __future__ import annotations

"""Train/export/decode the final token and BaseDyGraph-V2 comparisons.

Four model kinds are supported by one resumable runner:

``modern_tcn_token``
    Token counterpart of the selected one-block ModernTCN correlation-prior
    architecture.  It predicts the complete 60-step coarse-s1 path once from
    the final observed origin.

``dense_transformer_token``
    Token counterpart of the winning D64/three-ST-block dense Transformer.
    It supports both the hybrid dense-five/final-60 objective and the full
    60-origin x 60-future-position objective used by the final experiment.

``dimitri_v2_token``
    Dimitri's default four-block V2 backbone at context 60, using the same
    hybrid dense-five plus final-sixty coarse-token objective.

``dimitri_v2_continuous``
    The same V2 backbone with a direct five-horizon price head applied at all
    60 causal origins.  Public evaluation uses the final origin.

All curiosity models select checkpoints on the October-December test split.
Graph orientation is ``A[target, source]``.
"""

import argparse
import gc
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from time import perf_counter
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_parallel_forecast_dataset import (
    DensePrefixDatasetConfig,
    DensePrefixMultiHorizonDataset,
)
from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.evaluation.metrics import ForecastEvaluator
from src.models.dynamic_graph.future_predictor import (
    token_selection_probabilities,
)
from src.models.final_token_v2_models import (
    DenseTokenBackboneOutput,
    DenseTransformerTokenForecaster,
    DimitriV2DenseContinuousForecaster,
    DimitriV2DenseTokenForecaster,
    GRAPH_ORIENTATION,
)
from src.models.graph_priors import build_absolute_correlation_graph_prior
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.models.modern_tcn_graph_round1 import graph_component_summary
from src.models.modern_tcn_graph_round2_token import (
    ModernTCNGraphRound2TokenModel,
    token_round2_model_config_from_mapping,
)
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


TOP_K_VALUES = (1, 3, 5, 10)
TOKEN_MODEL_KINDS = {
    "modern_tcn_token",
    "dense_transformer_token",
    "dimitri_v2_token",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, default=None)
    parser.add_argument("--validation-cache", type=Path, default=None)
    parser.add_argument("--test-cache", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decode-sampled", action="store_true")
    parser.add_argument(
        "--backfill-probability-aggregates",
        action="store_true",
        help=(
            "Run selected-checkpoint inference only and add exact raw-model "
            "and post-sampling-policy probability aggregates to an existing "
            "decoded temperature policy. Existing samples, decoded prices, "
            "and metric tables are preserved."
        ),
    )
    parser.add_argument("--forecasting-config", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--decode-series-batch-size", type=int, default=64)
    parser.add_argument("--decode-splits", nargs="+", default=("validation", "test"))
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return values


def _atomic_torch_save(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def _atomic_json_save(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv_save(values: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    values.to_csv(temporary, index=False)
    temporary.replace(path)


def _signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(values: Mapping[str, Any]) -> None:
    random.setstate(values["python"])
    np.random.set_state(values["numpy"])
    torch.set_rng_state(values["torch"])
    if torch.cuda.is_available() and values.get("cuda") is not None:
        torch.cuda.set_rng_state_all(values["cuda"])


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _new_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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


def _prepare_run_dir(output_dir: Path, run_name: str, *, overwrite: bool, resume: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        metadata_path = run_dir / "run_metadata.json"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        if metadata_path.is_file() and checkpoint_path.is_file():
            metadata = _load_json(metadata_path)
            if metadata.get("status") == "completed":
                raise FileExistsError(f"Completed run already exists: {run_dir}")
        raise FileExistsError(f"Non-empty run requires --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_token_datasets(args: argparse.Namespace) -> dict[str, CachedTokenGraphDataset]:
    paths = {
        "train": args.train_cache,
        "validation": args.validation_cache,
        "test": args.test_cache,
    }
    if any(path is None for path in paths.values()):
        raise ValueError("All three token caches are required for token models.")
    datasets = {
        name: CachedTokenGraphDataset.from_path(path, data_mode="real")
        for name, path in paths.items()
    }
    reference = datasets["train"]
    for name, dataset in datasets.items():
        if dataset.context_length != reference.context_length:
            raise ValueError(f"{name} context length differs from train.")
        if dataset.prediction_length != reference.prediction_length:
            raise ValueError(f"{name} prediction length differs from train.")
        if dataset.asset_cols != reference.asset_cols:
            raise ValueError(f"{name} asset order differs from train.")
        if dataset.s1_id_space != "kronos_original":
            raise ValueError(f"{name} cache is not in original Kronos ID space.")
    return datasets


def _load_raw_splits(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = load_candle_splits(data_dir)
    return clean_candle_splits(*raw)


def _token_batch(batch: Mapping[str, Any], *, device: torch.device) -> tuple[Tensor, Tensor]:
    context = torch.as_tensor(batch["context_tokens"])[..., 0].to(
        device=device, dtype=torch.long, non_blocking=True
    )
    target = torch.as_tensor(batch["target_s1"]).to(
        device=device, dtype=torch.long, non_blocking=True
    )
    return context, target


def _all_origins_full_path_token_targets(
    context: Tensor,
    future: Tensor,
) -> Tensor:
    """Return every causal origin's complete future-token path.

    Given an observed context ``[x_0, ..., x_{T-1}]`` and saved continuation
    ``[x_T, ..., x_{T+P-1}]``, origin ``t`` is supervised against
    ``[x_{t+1}, ..., x_{t+P}]``.  With ``T=P=60`` the result has shape
    ``[B,60,60,N]`` and the final origin exactly equals the ordinary saved
    future path.
    """
    if context.ndim != 3 or future.ndim != 3:
        raise ValueError("context/future must have shape [B,T,N].")
    if (
        int(context.shape[0]) != int(future.shape[0])
        or int(context.shape[2]) != int(future.shape[2])
    ):
        raise ValueError("Context and future batches/assets do not align.")
    context_steps = int(context.shape[1])
    future_steps = int(future.shape[1])
    if context_steps <= 0 or future_steps <= 0:
        raise ValueError("Context and future paths must be non-empty.")

    combined = torch.cat([context, future], dim=1)
    origins = torch.arange(
        context_steps,
        device=context.device,
        dtype=torch.long,
    )[:, None]
    offsets = torch.arange(
        1,
        future_steps + 1,
        device=context.device,
        dtype=torch.long,
    )[None, :]
    indices = origins + offsets
    if int(indices.max().item()) >= int(combined.shape[1]):
        raise ValueError("Dense target index exceeds context+future data.")
    targets = combined.index_select(1, indices.reshape(-1)).reshape(
        int(context.shape[0]),
        context_steps,
        future_steps,
        int(context.shape[2]),
    )
    if not torch.equal(targets[:, -1], future):
        raise AssertionError("Final-origin dense target differs from future path.")
    return targets


def _hybrid_dense_token_targets(
    context: Tensor,
    future: Tensor,
    *,
    auxiliary_horizons: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Build hybrid dense-token targets without materialising a 60x60 cube.

    Returns
    -------
    auxiliary_targets:
        ``[B, T-1, H_aux, N]``.  Each internal causal origin predicts only
        the configured dissertation horizons.
    final_path_targets:
        ``[B, P, N]``.  The final observed origin predicts the complete
        ordered future path required by the frozen Kronos decoder.
    """
    if context.ndim != 3 or future.ndim != 3:
        raise ValueError("context/future must have shape [B,T,N].")
    if int(context.shape[0]) != int(future.shape[0]) or int(context.shape[2]) != int(future.shape[2]):
        raise ValueError("Context and future batches/assets do not align.")

    horizons = tuple(int(value) for value in auxiliary_horizons)
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("Auxiliary horizons must be unique and increasing.")
    if horizons[0] <= 0:
        raise ValueError("Auxiliary horizons must be positive.")

    context_steps = int(context.shape[1])
    future_steps = int(future.shape[1])
    internal_origins = context_steps - 1
    if internal_origins <= 0:
        raise ValueError("Dense auxiliary supervision requires at least two context steps.")
    if horizons[-1] > future_steps:
        raise ValueError("Auxiliary horizon exceeds the saved future path.")

    combined = torch.cat([context, future], dim=1)
    offsets = torch.as_tensor(horizons, device=context.device, dtype=torch.long)
    indices = (
        torch.arange(internal_origins, device=context.device, dtype=torch.long)[:, None]
        + offsets[None]
    )
    if int(indices.max().item()) >= int(combined.shape[1]):
        raise ValueError("Dense auxiliary target index exceeds context+future data.")

    auxiliary = combined.index_select(1, indices.reshape(-1)).reshape(
        int(context.shape[0]),
        internal_origins,
        len(horizons),
        int(context.shape[2]),
    )
    return auxiliary, future


def _token_sums(logits: Tensor, target: Tensor, top_k_values: Sequence[int]) -> dict[str, Tensor]:
    if logits.shape[:-1] != target.shape:
        raise ValueError("Token logits and targets do not align.")
    batch, prediction, nodes, vocabulary = map(int, logits.shape)
    losses = F.cross_entropy(
        logits.reshape(-1, vocabulary).float(),
        target.reshape(-1),
        reduction="none",
    ).reshape(batch, prediction, nodes)
    values: dict[str, Tensor] = {
        "ce_sum_by_step": losses.sum(dim=(0, 2)).double(),
        "count_by_step": torch.full(
            (prediction,), batch * nodes, dtype=torch.float64, device=logits.device
        ),
    }
    maximum = min(max(int(k) for k in top_k_values), vocabulary)
    matches = logits.topk(maximum, dim=-1).indices.eq(target.unsqueeze(-1))
    for k in top_k_values:
        values[f"top{int(k)}_correct_by_step"] = (
            matches[..., : min(int(k), vocabulary)].any(dim=-1).sum(dim=(0, 2)).double()
        )
    return values


def _graph_entropy(graph: Tensor | None) -> tuple[float | None, float | None]:
    if graph is None:
        return None, None
    values = torch.as_tensor(graph).detach().float().clamp_min(1.0e-12)
    entropy = -(values * values.log()).sum(dim=-1)
    return float(entropy.mean().item()), float(entropy.exp().mean().item())


def _accumulate_graph_statistics(
    accumulators: list[dict[str, float]],
    graphs: Sequence[Tensor],
) -> None:
    """Accumulate row-entropy diagnostics over every saved window/head/row."""
    if not accumulators:
        accumulators.extend(
            {"entropy_sum": 0.0, "effective_sum": 0.0, "count": 0.0}
            for _ in graphs
        )
    if len(accumulators) != len(graphs):
        raise ValueError("Graph-layer count changed between evaluation batches.")
    for accumulator, raw_graph in zip(accumulators, graphs, strict=True):
        graph = torch.as_tensor(raw_graph).detach().float().clamp_min(1.0e-12)
        entropy = -(graph * graph.log()).sum(dim=-1)
        accumulator["entropy_sum"] += float(entropy.sum().item())
        accumulator["effective_sum"] += float(entropy.exp().sum().item())
        accumulator["count"] += float(entropy.numel())


def _generic_graph_batch(
    output: DenseTokenBackboneOutput,
    *,
    final_origin_only: bool,
) -> dict[str, Any]:
    selected = tuple(
        graph[:, -1] if final_origin_only else graph
        for graph in output.selected_graphs
    )
    dynamic = tuple(
        None
        if graph is None
        else (graph[:, -1] if final_origin_only else graph)
        for graph in output.dynamic_graphs
    )
    slow = tuple(
        None
        if graph is None
        else (graph[:, -1] if final_origin_only else graph)
        for graph in output.slow_graphs
    )
    return {
        "selected": selected,
        "dynamic": dynamic,
        "base": output.base_graphs,
        "slow": slow,
        "alphas": output.alphas,
        "betas": output.betas,
    }


def _modern_graph_batch(output: Any) -> dict[str, Any]:
    return {
        "selected": tuple(block.graph.selected for block in output.block_outputs),
        "dynamic": tuple(block.graph.dynamic for block in output.block_outputs),
        "base": tuple(block.graph.base for block in output.block_outputs),
        "slow": tuple(None for _ in output.block_outputs),
        "alphas": tuple(block.graph.alpha for block in output.block_outputs),
        "betas": tuple(block.beta for block in output.block_outputs),
    }


def _build_model(
    *,
    config: Mapping[str, Any],
    token_dataset: CachedTokenGraphDataset | None,
    train_split: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    kind = str(config["model_kind"])
    if kind == "modern_tcn_token":
        if token_dataset is None:
            raise ValueError("Token dataset is required.")
        prior_type = str(config["model"]["prior"]["type"])
        if prior_type == "correlation":
            prior = build_absolute_correlation_graph_prior(
                train_split,
                expected_asset_cols=token_dataset.asset_cols,
                threshold=None,
            )
        elif prior_type in {"none", "uniform"}:
            prior = None
        else:
            raise ValueError(
                "The final ModernTCN token path supports only correlation "
                f"or uniform static initialisation; got {prior_type!r}."
            )
        model_config = token_round2_model_config_from_mapping(
            dict(config),
            num_nodes=token_dataset.num_assets,
            vocabulary_size=1024,
        )
        return ModernTCNGraphRound2TokenModel(
            model_config,
            static_prior=prior,
        ).to(device)
    if kind == "dense_transformer_token":
        if token_dataset is None:
            raise ValueError("Token dataset is required.")
        return DenseTransformerTokenForecaster(
            num_nodes=token_dataset.num_assets,
            context_length=token_dataset.context_length,
            prediction_length=token_dataset.prediction_length,
        ).to(device)
    if kind == "dimitri_v2_token":
        if token_dataset is None:
            raise ValueError("Token dataset is required.")
        return DimitriV2DenseTokenForecaster(
            num_nodes=token_dataset.num_assets,
            context_length=token_dataset.context_length,
            prediction_length=token_dataset.prediction_length,
        ).to(device)
    if kind == "dimitri_v2_continuous":
        return DimitriV2DenseContinuousForecaster(
            num_nodes=len(train_split["asset_cols"]),
            context_length=int(config["data"]["context_length"]),
            horizons=tuple(int(value) for value in config["data"]["horizons"]),
            input_channels=len(config["data"]["input_channels"]),
        ).to(device)
    raise ValueError(f"Unsupported model_kind {kind!r}.")


def _graph_parameter_ids(model: nn.Module) -> set[int]:
    function = getattr(model, "graph_parameter_ids", None)
    return set() if function is None else set(function())


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> tuple[torch.optim.Optimizer, Any | None]:
    training = config["training"]
    if str(training["parameter_grouping"]) == "shared":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training["scheduler_t_max"]),
            eta_min=0.0,
        )
        return optimizer, scheduler
    graph_ids = _graph_parameter_ids(model)
    graph = [p for p in model.parameters() if p.requires_grad and id(p) in graph_ids]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in graph_ids]
    if not graph or not backbone:
        raise RuntimeError("Split optimizer requires non-empty graph and backbone groups.")
    optimizer = torch.optim.Adam(
        [
            {"params": backbone, "lr": float(training["learning_rate"]), "base_lr": float(training["learning_rate"]), "name": "backbone"},
            {"params": graph, "lr": float(training["graph_learning_rate"]), "base_lr": float(training["graph_learning_rate"]), "name": "graph"},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    return optimizer, None


def _set_delayed_schedule(optimizer: torch.optim.Optimizer, config: Mapping[str, Any], epoch: int) -> None:
    training = config["training"]
    if str(training["scheduler"]) != "modern_tcn_type3_delayed":
        return
    start = int(training["scheduler_decay_start_epoch"])
    factor = float(training["scheduler_decay_factor"])
    multiplier = 1.0 if int(epoch) <= start else factor ** (int(epoch) - start)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


def _current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {str(group.get("name", "shared")): float(group["lr"]) for group in optimizer.param_groups}


def _forward_token_final(model: nn.Module, kind: str, context: Tensor, *, include_components: bool) -> tuple[Tensor, dict[str, Any]]:
    if kind == "modern_tcn_token":
        output = model(context)
        return output.s1_logits, _modern_graph_batch(output)
    logits, backbone = model.forward_final(context, include_components=include_components) if kind == "dimitri_v2_token" else model.forward_final(context)
    return logits, _generic_graph_batch(backbone, final_origin_only=True)


def _all_origins_full_path_backward(
    *,
    model: nn.Module,
    kind: str,
    context: Tensor,
    future: Tensor,
    loss_config: Mapping[str, Any],
    scaler: Any,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Backpropagate exact CE over all origins and all future positions.

    Origins are processed in configurable vectorised chunks.  Each chunk loss
    is multiplied by ``chunk_origins / total_origins`` before backward, so the
    accumulated gradient is exactly the mean over the complete target tensor
    ``[B,T,P,N]``.  Chunking changes only execution/memory, never the objective.
    """
    if kind not in {"dense_transformer_token", "dimitri_v2_token"}:
        raise ValueError(
            "Full dense all-origin token training requires a causal "
            "sequence-output token backbone."
        )
    expected_steps = int(loss_config["future_steps_per_origin"])
    if expected_steps != int(future.shape[1]):
        raise ValueError(
            "future_steps_per_origin differs from the saved future path."
        )
    total_origins = int(context.shape[1])
    chunk_size = int(loss_config.get("origin_chunk_size", 1))
    if chunk_size <= 0:
        raise ValueError("origin_chunk_size must be positive.")
    chunk_size = min(chunk_size, total_origins)

    with _autocast_context(device, use_amp):
        backbone = (
            model.forward_backbone(context, include_components=False)
            if kind == "dimitri_v2_token"
            else model.forward_backbone(context)
        )
    targets = _all_origins_full_path_token_targets(context, future)

    all_ce_sum = 0.0
    all_correct_sum = 0.0
    all_count = 0
    objective_value = 0.0
    final_ce_sum = 0.0
    final_correct_sum = 0.0
    final_count = 0

    for start in range(0, total_origins, chunk_size):
        stop = min(start + chunk_size, total_origins)
        origin_indices = torch.arange(
            start,
            stop,
            device=context.device,
            dtype=torch.long,
        )
        with _autocast_context(device, use_amp):
            logits = model.future_predictor.forward_origins(
                backbone.hidden,
                origin_indices,
            )
            target = targets[:, start:stop]
            raw_chunk_loss = F.cross_entropy(
                logits.reshape(-1, int(logits.shape[-1])).float(),
                target.reshape(-1),
            )
            chunk_weight = float(stop - start) / float(total_origins)
            weighted_chunk_loss = raw_chunk_loss * chunk_weight

        scaler.scale(weighted_chunk_loss).backward(
            retain_graph=stop < total_origins
        )

        raw_value = float(raw_chunk_loss.detach().item())
        chunk_count = int(target.numel())
        all_ce_sum += raw_value * chunk_count
        all_correct_sum += float(
            logits.argmax(dim=-1).eq(target).sum().item()
        )
        all_count += chunk_count
        objective_value += raw_value * chunk_weight

        if start <= total_origins - 1 < stop:
            local_index = total_origins - 1 - start
            final_logits = logits[:, local_index]
            final_target = target[:, local_index]
            final_loss = F.cross_entropy(
                final_logits.reshape(-1, int(final_logits.shape[-1])).float(),
                final_target.reshape(-1),
            )
            final_count = int(final_target.numel())
            final_ce_sum = float(final_loss.detach().item()) * final_count
            final_correct_sum = float(
                final_logits.argmax(dim=-1).eq(final_target).sum().item()
            )
            del final_logits, final_target, final_loss

        del (
            logits,
            target,
            raw_chunk_loss,
            weighted_chunk_loss,
            origin_indices,
        )

    if all_count <= 0 or final_count <= 0:
        raise RuntimeError("Full dense token batch produced no targets.")
    del targets, backbone
    return {
        "objective": objective_value,
        "all_origins_ce_sum": all_ce_sum,
        "all_origins_correct_sum": all_correct_sum,
        "all_origins_count": float(all_count),
        "final_ce_sum": final_ce_sum,
        "final_correct_sum": final_correct_sum,
        "final_count": float(final_count),
        "origin_chunk_size": float(chunk_size),
    }


def probe_full_dense_transformer_batch_candidates(
    *,
    candidates: Sequence[Sequence[int]],
    device: torch.device | str = "cuda",
    num_nodes: int = 93,
    context_length: int = 60,
    prediction_length: int = 60,
    mixed_precision: bool = True,
    memory_safety_fraction: float = 0.92,
    seed: int = 42,
    stop_after_first_safe: bool = True,
) -> list[dict[str, Any]]:
    """Probe physical batch/origin-chunk pairs using the real training path.

    The candidates are evaluated in the supplied order.  A candidate is marked
    safe only when a complete forward/backward/Adam step succeeds and peak CUDA
    allocation stays below ``memory_safety_fraction`` of device VRAM.  The
    function never changes the scientific objective or model architecture.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The full dense batch probe requires CUDA.")
    if not 0.0 < float(memory_safety_fraction) <= 1.0:
        raise ValueError("memory_safety_fraction must lie in (0,1].")
    parsed: list[tuple[int, int]] = []
    for value in candidates:
        if len(value) != 2:
            raise ValueError("Each candidate must be (batch_size, chunk_size).")
        batch_size, chunk_size = (int(value[0]), int(value[1]))
        if batch_size <= 0 or chunk_size <= 0:
            raise ValueError("Candidate sizes must be positive.")
        parsed.append((batch_size, chunk_size))
    if not parsed:
        raise ValueError("At least one batch candidate is required.")

    total_gib = float(
        torch.cuda.get_device_properties(device).total_memory / (1024**3)
    )
    records: list[dict[str, Any]] = []
    for batch_size, chunk_size in parsed:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model: nn.Module | None = None
        optimizer: torch.optim.Optimizer | None = None
        context: Tensor | None = None
        future: Tensor | None = None
        values: dict[str, float] | None = None
        scaler: Any | None = None
        try:
            _set_seed(int(seed))
            model = DenseTransformerTokenForecaster(
                num_nodes=int(num_nodes),
                context_length=int(context_length),
                prediction_length=int(prediction_length),
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=2.5e-4,
                weight_decay=0.0,
            )
            scaler = _new_grad_scaler(
                bool(mixed_precision) and device.type == "cuda"
            )
            context = torch.randint(
                0,
                1024,
                (batch_size, context_length, num_nodes),
                device=device,
                dtype=torch.long,
            )
            future = torch.randint(
                0,
                1024,
                (batch_size, prediction_length, num_nodes),
                device=device,
                dtype=torch.long,
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = perf_counter()
            values = _all_origins_full_path_backward(
                model=model,
                kind="dense_transformer_token",
                context=context,
                future=future,
                loss_config={
                    "future_steps_per_origin": int(prediction_length),
                    "origin_chunk_size": int(chunk_size),
                },
                scaler=scaler,
                device=device,
                use_amp=bool(mixed_precision),
            )
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            peak_gib = float(
                torch.cuda.max_memory_allocated(device) / (1024**3)
            )
            safe = peak_gib <= float(memory_safety_fraction) * total_gib
            records.append(
                {
                    "batch_size": batch_size,
                    "origin_chunk_size": chunk_size,
                    "status": "safe" if safe else "fits_but_above_margin",
                    "seconds_per_step": elapsed,
                    "examples_per_second": batch_size / max(elapsed, 1.0e-9),
                    "peak_cuda_gib": peak_gib,
                    "total_cuda_gib": total_gib,
                    "peak_fraction": peak_gib / total_gib,
                    "objective": float(values["objective"]),
                }
            )
            if safe and stop_after_first_safe:
                break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            message = str(exc).lower()
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and (
                "out of memory" not in message
                and "cuda error: out of memory" not in message
            ):
                raise
            records.append(
                {
                    "batch_size": batch_size,
                    "origin_chunk_size": chunk_size,
                    "status": "cuda_oom",
                    "seconds_per_step": None,
                    "examples_per_second": None,
                    "peak_cuda_gib": float(
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                    ),
                    "total_cuda_gib": total_gib,
                    "peak_fraction": None,
                    "objective": None,
                }
            )
        finally:
            del model, optimizer, context, future, values, scaler
            gc.collect()
            torch.cuda.empty_cache()

    if not any(value["status"] == "safe" for value in records):
        raise RuntimeError(
            "No full-dense batch candidate satisfied the CUDA memory margin. "
            "Include a smaller pair such as (1,1)."
        )
    return records


def _train_token_epoch(
    *,
    model: nn.Module,
    kind: str,
    dataset: Dataset,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
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
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    loss_config = training["loss"]
    dense = bool(loss_config.get("dense_origins", False))
    dense_objective_name = (
        str(loss_config.get("dense_objective", "")) if dense else ""
    )
    model.train()

    final_ce_sum = final_correct_sum = final_count = 0.0
    auxiliary_ce_sum = auxiliary_correct_sum = auxiliary_count = 0.0
    objective_sum = objective_weight = 0.0

    for batch in tqdm(loader, desc=f"train token epoch {epoch}", leave=False, dynamic_ncols=True):
        context, future = _token_batch(batch, device=device)
        optimizer.zero_grad(set_to_none=True)

        if not dense:
            with _autocast_context(device, use_amp):
                logits, _ = _forward_token_final(
                    model, kind, context, include_components=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, 1024).float(), future.reshape(-1)
                )
            scaler.scale(loss).backward()
            batch_count = int(future.numel())
            final_ce_sum += float(loss.detach().item()) * batch_count
            final_correct_sum += float(logits.argmax(dim=-1).eq(future).sum().item())
            final_count += batch_count
            objective_sum += float(loss.detach().item()) * int(context.shape[0])
            objective_weight += int(context.shape[0])
        else:
            objective_name = str(loss_config.get("dense_objective", ""))
            if objective_name == "all_60_future_positions_per_origin":
                batch_values = _all_origins_full_path_backward(
                    model=model,
                    kind=kind,
                    context=context,
                    future=future,
                    loss_config=loss_config,
                    scaler=scaler,
                    device=device,
                    use_amp=use_amp,
                )
                final_ce_sum += batch_values["final_ce_sum"]
                final_correct_sum += batch_values["final_correct_sum"]
                final_count += batch_values["final_count"]
                auxiliary_ce_sum += batch_values["all_origins_ce_sum"]
                auxiliary_correct_sum += batch_values[
                    "all_origins_correct_sum"
                ]
                auxiliary_count += batch_values["all_origins_count"]
                objective_sum += batch_values["objective"] * int(
                    context.shape[0]
                )
                objective_weight += int(context.shape[0])
            elif objective_name == "internal_five_horizons_plus_final_full_path":
                auxiliary_horizons = tuple(
                    int(value) for value in loss_config["dense_auxiliary_horizons"]
                )
                auxiliary_weight = float(loss_config["dense_auxiliary_weight"])
                if auxiliary_weight < 0.0:
                    raise ValueError("dense_auxiliary_weight must be non-negative.")
                expected_final_steps = int(loss_config["final_origin_future_steps"])
                if expected_final_steps != int(future.shape[1]):
                    raise ValueError(
                        "Final-origin path length differs from the configured loss."
                    )

                with _autocast_context(device, use_amp):
                    backbone = (
                        model.forward_backbone(context, include_components=False)
                        if kind == "dimitri_v2_token"
                        else model.forward_backbone(context)
                    )
                    final_logits = model.future_predictor.forward_origin(
                        backbone.hidden,
                        int(context.shape[1]) - 1,
                    )
                    final_loss = F.cross_entropy(
                        final_logits.reshape(-1, 1024).float(), future.reshape(-1)
                    )

                auxiliary_target, final_target = _hybrid_dense_token_targets(
                    context,
                    future,
                    auxiliary_horizons=auxiliary_horizons,
                )
                # ``final_target`` is returned explicitly so the target contract is
                # audited in one place rather than inferred independently here.
                if not torch.equal(final_target, future):
                    raise AssertionError("Final-origin target path was altered.")

                internal_origins = int(auxiliary_target.shape[1])
                query_indices = torch.as_tensor(
                    [value - 1 for value in auxiliary_horizons],
                    device=context.device,
                    dtype=torch.long,
                )

                final_loss_value = float(final_loss.detach().item())
                batch_final_count = int(final_target.numel())
                final_ce_sum += final_loss_value * batch_final_count
                final_correct_sum += float(
                    final_logits.argmax(dim=-1).eq(final_target).sum().item()
                )
                final_count += batch_final_count
                del final_logits

                auxiliary_losses: list[Tensor] = []
                batch_auxiliary_loss_sum = 0.0
                for origin in range(internal_origins):
                    with _autocast_context(device, use_amp):
                        auxiliary_logits = model.future_predictor.forward_origin(
                            backbone.hidden,
                            origin,
                            future_position_indices=query_indices,
                        )
                        target = auxiliary_target[:, origin]
                        raw_auxiliary_loss = F.cross_entropy(
                            auxiliary_logits.reshape(-1, 1024).float(),
                            target.reshape(-1),
                        )
                    auxiliary_losses.append(raw_auxiliary_loss)
                    raw_value = float(raw_auxiliary_loss.detach().item())
                    batch_auxiliary_loss_sum += raw_value
                    batch_count = int(target.numel())
                    auxiliary_ce_sum += raw_value * batch_count
                    auxiliary_correct_sum += float(
                        auxiliary_logits.argmax(dim=-1).eq(target).sum().item()
                    )
                    auxiliary_count += batch_count
                    del auxiliary_logits, target

                auxiliary_mean_loss = torch.stack(auxiliary_losses).mean()
                objective_loss = final_loss + auxiliary_weight * auxiliary_mean_loss
                batch_auxiliary_mean = batch_auxiliary_loss_sum / internal_origins
                batch_objective = final_loss_value + (
                    auxiliary_weight * batch_auxiliary_mean
                )
                scaler.scale(objective_loss).backward()
                objective_sum += batch_objective * int(context.shape[0])
                objective_weight += int(context.shape[0])
                del (
                    auxiliary_losses,
                    auxiliary_mean_loss,
                    objective_loss,
                    final_loss,
                    auxiliary_target,
                    final_target,
                    backbone,
                )
            else:
                raise ValueError(
                    "Unsupported dense token objective: " + repr(objective_name)
                )

        scaler.unscale_(optimizer)
        clip = training.get("gradient_clip_norm")
        if clip is not None and float(clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
        scaler.step(optimizer)
        scaler.update()

    if final_count <= 0 or objective_weight <= 0:
        raise RuntimeError("Token training produced no final-path targets.")

    result = {
        "training_objective": objective_sum / objective_weight,
        "training_mean_cross_entropy": final_ce_sum / final_count,
        "training_mean_top1_accuracy": final_correct_sum / final_count,
        "training_final_path_cross_entropy": final_ce_sum / final_count,
        "training_final_path_top1_accuracy": final_correct_sum / final_count,
    }
    if dense:
        if auxiliary_count <= 0:
            raise RuntimeError("Dense token training produced no dense targets.")
        if dense_objective_name == "all_60_future_positions_per_origin":
            result.update(
                {
                    "training_dense_all_origins_cross_entropy": (
                        auxiliary_ce_sum / auxiliary_count
                    ),
                    "training_dense_all_origins_top1_accuracy": (
                        auxiliary_correct_sum / auxiliary_count
                    ),
                    "training_dense_origin_count": float(
                        config["data"]["context_length"]
                    ),
                    "training_dense_future_steps_per_origin": float(
                        loss_config["future_steps_per_origin"]
                    ),
                    "training_dense_origin_chunk_size": float(
                        loss_config.get("origin_chunk_size", 1)
                    ),
                }
            )
        else:
            result.update(
                {
                    "training_dense_auxiliary_cross_entropy": (
                        auxiliary_ce_sum / auxiliary_count
                    ),
                    "training_dense_auxiliary_top1_accuracy": (
                        auxiliary_correct_sum / auxiliary_count
                    ),
                    "training_dense_auxiliary_weight": float(
                        loss_config["dense_auxiliary_weight"]
                    ),
                }
            )
    return result


def _evaluate_token(
    *,
    model: nn.Module,
    kind: str,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    description: str,
) -> dict[str, Any]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["selection_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    prediction = int(config["data"]["prediction_length"])
    ce_sum = torch.zeros(prediction, dtype=torch.float64)
    count = torch.zeros(prediction, dtype=torch.float64)
    correct = {k: torch.zeros(prediction, dtype=torch.float64) for k in TOP_K_VALUES}
    graph_batch: dict[str, Any] | None = None
    graph_statistics: list[dict[str, float]] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            context, target = _token_batch(batch, device=device)
            with _autocast_context(device, use_amp):
                logits, graph_batch = _forward_token_final(
                    model, kind, context, include_components=False
                )
            sums = _token_sums(logits, target, TOP_K_VALUES)
            ce_sum += sums["ce_sum_by_step"].cpu()
            count += sums["count_by_step"].cpu()
            for k in TOP_K_VALUES:
                correct[k] += sums[f"top{k}_correct_by_step"].cpu()
            _accumulate_graph_statistics(graph_statistics, graph_batch["selected"])
    if torch.any(count <= 0):
        raise RuntimeError("Token evaluation produced an empty step.")
    ce = ce_sum / count
    accuracy = {k: correct[k] / count for k in TOP_K_VALUES}
    result: dict[str, Any] = {
        "mean_cross_entropy": float(ce.mean().item()),
        "mean_top1_accuracy": float(accuracy[1].mean().item()),
        "cross_entropy_by_step": ce,
        **{f"top{k}_accuracy_by_step": value for k, value in accuracy.items()},
    }
    for index, accumulator in enumerate(graph_statistics):
        if accumulator["count"] <= 0:
            continue
        result[f"block_{index}_selected_entropy"] = (
            accumulator["entropy_sum"] / accumulator["count"]
        )
        result[f"block_{index}_selected_effective_neighbours"] = (
            accumulator["effective_sum"] / accumulator["count"]
        )
    if graph_batch is not None:
        for index, alpha in enumerate(graph_batch["alphas"]):
            result[f"block_{index}_alpha"] = (
                None
                if alpha is None
                else float(torch.as_tensor(alpha).detach().float().mean().item())
            )
        for index, beta in enumerate(graph_batch["betas"]):
            result[f"block_{index}_beta"] = (
                None
                if beta is None
                else float(torch.as_tensor(beta).detach().float().mean().item())
            )
    return result


def _dense_price_error(
    predictions_normalised: Tensor,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mean = torch.as_tensor(batch["target_norm_mean"]).to(device=device, dtype=torch.float32)
    std = torch.as_tensor(batch["target_norm_std"]).to(device=device, dtype=torch.float32)
    true_raw = torch.as_tensor(batch["dense_y_unnormalised"]).to(device=device, dtype=torch.float32)
    current = torch.as_tensor(batch["dense_current_close"]).to(device=device, dtype=torch.float32)
    predicted_raw = predictions_normalised * std[:, None, None] + mean[:, None, None]
    predicted_change = torch.log(predicted_raw.clamp_min(eps)) - torch.log(current[:, :, None].clamp_min(eps))
    true_change = torch.log(true_raw.clamp_min(eps)) - torch.log(current[:, :, None].clamp_min(eps))
    return (predicted_change - true_change).abs(), predicted_raw, true_raw


def _train_continuous_epoch(
    *,
    model: DimitriV2DenseContinuousForecaster,
    dataset: Dataset,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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
    weights = torch.tensor(training["loss"]["horizon_weights"], device=device).view(1, 1, -1, 1, 1)
    eps = float(config["normalisation"]["eps"])
    bps = float(training["loss"]["bps_scale"])
    model.train()
    unweighted_sum = weighted_sum = count = 0.0
    for batch in tqdm(loader, desc=f"train V2 price epoch {epoch}", leave=False, dynamic_ncols=True):
        x = torch.as_tensor(batch["x"]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_dense(x, include_components=False)
        error, _, _ = _dense_price_error(output.predictions, batch, device=device, eps=eps)
        objective = (error * weights).mean() * bps
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite V2 continuous loss.")
        objective.backward()
        optimizer.step()
        unweighted_sum += float(error.sum().item())
        weighted_sum += float((error * weights).sum().item())
        count += float(error.numel())
    return {
        "training_unweighted_mean_log_mae": unweighted_sum / count,
        "training_weighted_mean_log_mae": weighted_sum / count,
    }


def _continuous_dataset(split: Mapping[str, Any], config: Mapping[str, Any]) -> DensePrefixMultiHorizonDataset:
    return DensePrefixMultiHorizonDataset(
        dict(split),
        config=DensePrefixDatasetConfig(
            context_length=int(config["data"]["context_length"]),
            horizons=tuple(int(value) for value in config["data"]["horizons"]),
            stride=int(config["data"]["stride"]),
            input_channels=tuple(config["data"]["input_channels"]),
            target_channel=str(config["data"]["target_channel"]),
            eps=float(config["normalisation"]["eps"]),
            clip=bool(config["normalisation"]["clip"]),
            clip_min=float(config["normalisation"]["clip_min"]),
            clip_max=float(config["normalisation"]["clip_max"]),
        ),
    )


def _evaluate_continuous(
    *,
    model: DimitriV2DenseContinuousForecaster,
    dataset: Dataset,
    config: Mapping[str, Any],
    device: torch.device,
    description: str,
) -> dict[str, Any]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["selection_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    eps = float(config["normalisation"]["eps"])
    sums = torch.zeros(len(config["data"]["horizons"]), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    graph_statistics: list[dict[str, float]] = []
    graph_batch: dict[str, Any] | None = None
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(device=device, dtype=torch.float32)
            output = model.forward_dense(x, include_components=False)
            final = output.final_predictions()
            mean = torch.as_tensor(batch["target_norm_mean"]).to(device=device, dtype=torch.float32)
            std = torch.as_tensor(batch["target_norm_std"]).to(device=device, dtype=torch.float32)
            predicted_raw = final * std[:, None] + mean[:, None]
            true_raw = torch.as_tensor(batch["y_unnormalised"]).to(device=device, dtype=torch.float32)
            last = torch.as_tensor(batch["last_context_target"]).to(device=device, dtype=torch.float32)
            error = (
                (torch.log(predicted_raw.clamp_min(eps)) - torch.log(last[:, None].clamp_min(eps)))
                - (torch.log(true_raw.clamp_min(eps)) - torch.log(last[:, None].clamp_min(eps)))
            ).abs()
            sums += error.sum(dim=(0, 2, 3)).double().cpu()
            counts += torch.full_like(sums, int(error.shape[0]) * int(error.shape[2]))
            graph_batch = _generic_graph_batch(output.backbone, final_origin_only=True)
            _accumulate_graph_statistics(graph_statistics, graph_batch["selected"])
    by_horizon = sums / counts
    result: dict[str, Any] = {
        "mean_log_mae": float(by_horizon.mean().item()),
        "log_mae_by_horizon": by_horizon,
    }
    for index, accumulator in enumerate(graph_statistics):
        if accumulator["count"] <= 0:
            continue
        result[f"block_{index}_selected_entropy"] = (
            accumulator["entropy_sum"] / accumulator["count"]
        )
        result[f"block_{index}_selected_effective_neighbours"] = (
            accumulator["effective_sum"] / accumulator["count"]
        )
    if graph_batch is not None:
        for index, alpha in enumerate(graph_batch["alphas"]):
            result[f"block_{index}_alpha"] = (
                None
                if alpha is None
                else float(torch.as_tensor(alpha).detach().float().mean().item())
            )
    return result


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "rng_state": _capture_rng_state(),
        "config_signature": _signature(config),
    }


def _token_metric_tables(
    evaluation: Mapping[str, Any],
    reported_horizons: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-60 token metrics and the five-horizon Graph-Hub table."""
    cross_entropy = torch.as_tensor(evaluation["cross_entropy_by_step"])
    prediction_length = int(cross_entropy.numel())
    all_rows: list[dict[str, float | int]] = []
    for index in range(prediction_length):
        row: dict[str, float | int] = {
            "horizon": index + 1,
            "cross_entropy": float(cross_entropy[index]),
        }
        for k in TOP_K_VALUES:
            row[f"top{k}_accuracy"] = float(
                torch.as_tensor(evaluation[f"top{k}_accuracy_by_step"])[index]
            )
        all_rows.append(row)
    wide = pd.DataFrame(all_rows)

    public = {int(value) for value in reported_horizons}
    long_rows: list[dict[str, float | int | str]] = []
    for row in all_rows:
        horizon = int(row["horizon"])
        if horizon not in public:
            continue
        for metric, value in row.items():
            if metric == "horizon":
                continue
            long_rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "channel": "s1",
                    "value": float(value),
                }
            )
    return wide, pd.DataFrame(long_rows)


def _static_graph_for_artifact(values: Any) -> Tensor | None:
    """Store one global static graph as [G,N,N], never as one window.

    Several model implementations expose a global base adjacency with a
    broadcast batch singleton [1,G,N,N].  Graph Hub interprets rank-four
    tensors as [W,G,N,N], so retaining that singleton would falsely claim the
    graph has one saved window.
    """
    if values is None:
        return None
    tensor = torch.as_tensor(values).detach().cpu().float()
    if tensor.ndim == 4:
        if int(tensor.shape[0]) <= 0:
            raise ValueError("Static graph has an empty leading dimension.")
        reference = tensor[0]
        if int(tensor.shape[0]) > 1:
            expanded = reference.unsqueeze(0).expand_as(tensor)
            if not torch.allclose(tensor, expanded, atol=2.0e-6, rtol=0.0):
                raise ValueError(
                    "A purported static graph varies across the batch/window axis."
                )
        tensor = reference
    if tensor.ndim not in {2, 3}:
        raise ValueError(
            "Static graph must resolve to [N,N] or [G,N,N], got "
            f"{tuple(tensor.shape)}."
        )
    return tensor.contiguous()


def _append_graph_parts(storage: list[list[Tensor]], graphs: Sequence[Tensor]) -> None:
    if not storage:
        storage.extend([] for _ in graphs)
    for index, graph in enumerate(graphs):
        storage[index].append(torch.as_tensor(graph).detach().cpu().float())


def _export_token_split(
    *,
    model: nn.Module,
    kind: str,
    dataset: CachedTokenGraphDataset,
    split_name: str,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["export_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    predicted_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    top10_id_parts: list[Tensor] = []
    top10_prob_parts: list[Tensor] = []
    true_prob_parts: list[Tensor] = []
    selected_lists: list[list[Tensor]] = []
    dynamic_lists: list[list[Tensor]] = []
    slow_lists: list[list[Tensor]] = []
    base_singletons: list[Tensor | None] = []
    alpha_values: tuple[Any, ...] = ()
    beta_values: tuple[Any, ...] = ()
    sample_parts: list[Tensor] = []
    origin_parts: list[Tensor] = []
    target_index_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    dates: list[str] = []
    reported = torch.tensor([int(value) - 1 for value in config["data"]["evaluation_horizons"]], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"export token {split_name}", leave=False, dynamic_ncols=True):
            context, target = _token_batch(batch, device=device)
            with _autocast_context(device, use_amp):
                logits, graphs = _forward_token_final(model, kind, context, include_components=True)
            probabilities = logits.float().softmax(dim=-1)
            top_prob, top_ids = probabilities.topk(10, dim=-1)
            predicted_parts.append(logits.argmax(dim=-1).cpu().to(torch.int16))
            target_parts.append(target.cpu().to(torch.int16))
            top10_id_parts.append(top_ids.index_select(1, reported.to(top_ids.device)).cpu().to(torch.int16))
            top10_prob_parts.append(top_prob.index_select(1, reported.to(top_prob.device)).cpu().float())
            true_prob = probabilities.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            true_prob_parts.append(true_prob.index_select(1, reported.to(true_prob.device)).cpu().float())
            _append_graph_parts(selected_lists, graphs["selected"])
            _append_graph_parts(dynamic_lists, [g if g is not None else s for g, s in zip(graphs["dynamic"], graphs["selected"], strict=True)])
            _append_graph_parts(slow_lists, [g if g is not None else s for g, s in zip(graphs["slow"], graphs["selected"], strict=True)])
            if not base_singletons:
                base_singletons = [_static_graph_for_artifact(g) for g in graphs["base"]]
                alpha_values = tuple(graphs["alphas"])
                beta_values = tuple(graphs["betas"])
            sample_parts.append(torch.as_tensor(batch["sample_idx"]).long())
            origin_parts.append(torch.as_tensor(batch["origin_idx"]).long())
            target_index_parts.append(torch.as_tensor(batch["target_indices"]).long())
            last_parts.append(
                torch.as_tensor(batch["context_tokens"])[:, -1, :, 0].long()
            )
            batch_dates = batch.get("date")
            if batch_dates is not None:
                dates.extend(str(value) for value in batch_dates)
    predicted = torch.cat(predicted_parts, dim=0)
    target = torch.cat(target_parts, dim=0)
    dense_target_indices = torch.cat(target_index_parts, dim=0)
    sample_idx = torch.cat(sample_parts, dim=0)
    origin_idx = torch.cat(origin_parts, dim=0)
    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    per_layer_slow = tuple(torch.cat(values, dim=0) for values in slow_lists)
    alphas = tuple(None if value is None else torch.as_tensor(value).detach().cpu().float().reshape(-1) for value in alpha_values)
    betas = tuple(None if value is None else torch.as_tensor(value).detach().cpu().float().reshape(-1) for value in beta_values)
    evaluation = _evaluate_token(
        model=model,
        kind=kind,
        dataset=dataset,
        config=config,
        device=device,
        description=f"token metrics {split_name}",
    )
    token_metric_table, metric_table = _token_metric_tables(evaluation, config["data"]["evaluation_horizons"])
    public_prediction = {
        "y_pred": predicted.index_select(1, reported).unsqueeze(-1),
        "y_true": target.index_select(1, reported).unsqueeze(-1),
        "last_context_target": torch.cat(last_parts, dim=0).unsqueeze(-1),
        "channels": ["s1"],
        "horizons": list(config["data"]["evaluation_horizons"]),
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": dense_target_indices.index_select(1, reported),
        "output_space": "token_id",
    }
    dense_prediction = {
        "y_pred": predicted.unsqueeze(-1),
        "y_true": target.unsqueeze(-1),
        "last_context_target": public_prediction["last_context_target"],
        "channels": ["s1"],
        "horizons": list(range(1, int(config["data"]["prediction_length"]) + 1)),
        "reported_horizons": list(config["data"]["evaluation_horizons"]),
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": dense_target_indices,
        "output_space": "token_id",
    }
    graph_artifacts = {
        "graph_type": str(config["model_kind"]),
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(dataset.asset_cols),
        "num_layers": len(per_layer_selected),
        "num_heads": int(per_layer_selected[-1].shape[1]),
        "num_heads_per_layer": [int(value.shape[1]) for value in per_layer_selected],
        "layer_head_counts": [int(value.shape[1]) for value in per_layer_selected],
        "selected_layer": len(per_layer_selected) - 1,
        "selected": per_layer_selected[-1],
        "per_layer": per_layer_selected,
        "base": base_singletons[-1],
        "per_layer_base": tuple(base_singletons),
        "dynamic": per_layer_dynamic[-1],
        "per_layer_dynamic": per_layer_dynamic,
        "slow": per_layer_slow[-1],
        "per_layer_slow": per_layer_slow,
        "alpha": alphas[-1],
        "alpha_per_layer": alphas,
        "beta": betas[-1],
        "beta_per_layer": betas,
        "dates": dates,
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": public_prediction["target_indices"],
    }
    token_artifacts = {
        "predicted_s1": predicted,
        "generated_s1": predicted,
        "target_s1": target,
        "top10_s1_ids_at_reported_horizons": torch.cat(top10_id_parts, dim=0),
        "top10_s1_probabilities_at_reported_horizons": torch.cat(top10_prob_parts, dim=0),
        "true_s1_probability_at_reported_horizons": torch.cat(true_prob_parts, dim=0),
        "evaluation_horizons": list(config["data"]["evaluation_horizons"]),
        "prediction_length": int(config["data"]["prediction_length"]),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": public_prediction["target_indices"],
        "dates": dates,
        "asset_cols": list(dataset.asset_cols),
        "token_selection": "argmax",
        "future_token_mode": "coarse_only",
        "input_token_stream": "s1",
        "output_token_stream": "s1",
    }
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": len(dataset),
        "prediction_length": int(config["data"]["prediction_length"]),
        "reported_horizons": list(config["data"]["evaluation_horizons"]),
        "mean_top1_accuracy_all_60": float(evaluation["mean_top1_accuracy"]),
        "mean_cross_entropy_all_60": float(evaluation["mean_cross_entropy"]),
        "model_kind": kind,
        "graph_orientation": GRAPH_ORIENTATION,
        "blocks": [
            {
                "block": index,
                "alpha": None if alphas[index] is None else float(alphas[index].mean().item()),
                "beta": None if betas[index] is None else float(betas[index].mean().item()),
                "selected_graph": graph_component_summary(per_layer_selected[index]),
                "base_graph": graph_component_summary(base_singletons[index]),
                "dynamic_graph": graph_component_summary(per_layer_dynamic[index]),
            }
            for index in range(len(per_layer_selected))
        ],
    }
    return {
        "prediction_result": public_prediction,
        "dense_prediction_result": dense_prediction,
        "graph_artifacts": graph_artifacts,
        "token_artifacts": token_artifacts,
        "metric_table": metric_table,
        "token_metric_table": token_metric_table,
        "diagnostics": diagnostics,
    }


def _save_token_export(run_dir: Path, split_name: str, values: Mapping[str, Any]) -> None:
    epoch = int(values["diagnostics"]["checkpoint_epoch"])
    paths = {
        "predictions": run_dir / f"best_{split_name}_predictions.pt",
        "token_predictions": run_dir / f"best_{split_name}_token_predictions.pt",
        "graphs": run_dir / f"best_{split_name}_graphs.pt",
        "tokens": run_dir / f"best_{split_name}_tokens.pt",
        "metrics": run_dir / f"best_{split_name}_metric_table.csv",
        "token_metrics": run_dir / f"best_{split_name}_token_metric_table.csv",
        "diagnostics": run_dir / f"best_{split_name}_diagnostics.json",
    }
    _atomic_torch_save({"epoch": epoch, "prediction_result": values["prediction_result"]}, paths["predictions"])
    _atomic_torch_save({"epoch": epoch, "prediction_result": values["dense_prediction_result"]}, paths["token_predictions"])
    _atomic_torch_save({"epoch": epoch, "graph_artifacts": values["graph_artifacts"]}, paths["graphs"])
    _atomic_torch_save({"epoch": epoch, "token_artifacts": values["token_artifacts"]}, paths["tokens"])
    _atomic_csv_save(values["metric_table"], paths["metrics"])
    _atomic_csv_save(values["token_metric_table"], paths["token_metrics"])
    _atomic_json_save(values["diagnostics"], paths["diagnostics"])
    analysis = run_dir / "analysis" / split_name
    analysis.mkdir(parents=True, exist_ok=True)
    for key, source in paths.items():
        target_name = {
            "predictions": "predictions.pt",
            "token_predictions": "token_predictions.pt",
            "graphs": "graphs.pt",
            "tokens": "tokens.pt",
            "metrics": "metric_table.csv",
            "token_metrics": "token_metric_table.csv",
            "diagnostics": "diagnostics.json",
        }[key]
        shutil.copy2(source, analysis / target_name)


def _export_continuous_split(
    *,
    model: DimitriV2DenseContinuousForecaster,
    dataset: DensePrefixMultiHorizonDataset,
    split_name: str,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_epoch: int,
    train_split: Mapping[str, Any],
    bootstrap: bool,
) -> dict[str, Any]:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["export_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    prediction_parts: list[Tensor] = []
    true_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    sample_parts: list[Tensor] = []
    origin_parts: list[Tensor] = []
    target_index_parts: list[Tensor] = []
    selected_lists: list[list[Tensor]] = []
    dynamic_lists: list[list[Tensor]] = []
    slow_lists: list[list[Tensor]] = []
    base_singletons: list[Tensor | None] = []
    alpha_values: tuple[Any, ...] = ()
    dates: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"export V2 price {split_name}", leave=False, dynamic_ncols=True):
            x = torch.as_tensor(batch["x"]).to(device=device, dtype=torch.float32)
            output = model.forward_dense(x, include_components=True)
            final = output.final_predictions()
            mean = torch.as_tensor(batch["target_norm_mean"]).to(device=device, dtype=torch.float32)
            std = torch.as_tensor(batch["target_norm_std"]).to(device=device, dtype=torch.float32)
            predicted_raw = final * std[:, None] + mean[:, None]
            prediction_parts.append(predicted_raw.cpu())
            true_parts.append(torch.as_tensor(batch["y_unnormalised"]).float())
            last_parts.append(torch.as_tensor(batch["last_context_target"]).float())
            sample_parts.append(torch.as_tensor(batch["sample_idx"]).long())
            origin_parts.append(torch.as_tensor(batch["origin_idx"]).long())
            target_index_parts.append(torch.as_tensor(batch["target_indices"]).long())
            graph_values = _generic_graph_batch(output.backbone, final_origin_only=True)
            _append_graph_parts(selected_lists, graph_values["selected"])
            _append_graph_parts(dynamic_lists, [g if g is not None else s for g, s in zip(graph_values["dynamic"], graph_values["selected"], strict=True)])
            _append_graph_parts(slow_lists, [g if g is not None else s for g, s in zip(graph_values["slow"], graph_values["selected"], strict=True)])
            if not base_singletons:
                base_singletons = [_static_graph_for_artifact(g) for g in graph_values["base"]]
                alpha_values = tuple(graph_values["alphas"])
            batch_dates = batch.get("day")
            if batch_dates is not None:
                dates.extend(str(value) for value in batch_dates)
    prediction_result = {
        "y_pred": torch.cat(prediction_parts, dim=0),
        "y_true": torch.cat(true_parts, dim=0),
        "last_context_target": torch.cat(last_parts, dim=0),
        "sample_idx": torch.cat(sample_parts, dim=0),
        "origin_idx": torch.cat(origin_parts, dim=0),
        "target_indices": torch.cat(target_index_parts, dim=0),
        "channels": ["close"],
        "horizons": list(config["data"]["horizons"]),
        "asset_cols": list(dataset.asset_cols),
        "output_space": "raw",
    }
    evaluator = ForecastEvaluator(prediction_result=prediction_result, train_split=dict(train_split))
    results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
        bootstrap=bool(bootstrap),
        n_bootstrap=10000,
        bootstrap_seed=42,
    )
    metric_table = make_evaluation_table(results, evaluator.horizons, evaluator.channels)
    per_layer_selected = tuple(torch.cat(values, dim=0) for values in selected_lists)
    per_layer_dynamic = tuple(torch.cat(values, dim=0) for values in dynamic_lists)
    per_layer_slow = tuple(torch.cat(values, dim=0) for values in slow_lists)
    alphas = tuple(None if value is None else torch.as_tensor(value).detach().cpu().float().reshape(-1) for value in alpha_values)
    graph_artifacts = {
        "graph_type": "dimitri_v2_dual_fusion",
        "graph_orientation": GRAPH_ORIENTATION,
        "orientation": GRAPH_ORIENTATION,
        "asset_cols": list(dataset.asset_cols),
        "num_layers": len(per_layer_selected),
        "num_heads": int(per_layer_selected[-1].shape[1]),
        "num_heads_per_layer": [int(value.shape[1]) for value in per_layer_selected],
        "layer_head_counts": [int(value.shape[1]) for value in per_layer_selected],
        "selected_layer": len(per_layer_selected) - 1,
        "selected": per_layer_selected[-1],
        "per_layer": per_layer_selected,
        "base": base_singletons[-1],
        "per_layer_base": tuple(base_singletons),
        "dynamic": per_layer_dynamic[-1],
        "per_layer_dynamic": per_layer_dynamic,
        "slow": per_layer_slow[-1],
        "per_layer_slow": per_layer_slow,
        "alpha": alphas[-1],
        "alpha_per_layer": alphas,
        "beta": None,
        "beta_per_layer": tuple(None for _ in per_layer_selected),
        "dates": dates,
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
    }
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "windows": len(dataset),
        "model_kind": "dimitri_v2_continuous",
        "graph_orientation": GRAPH_ORIENTATION,
        "bootstrap_sessions": 10000 if bootstrap else 0,
    }
    return {
        "prediction_result": prediction_result,
        "graph_artifacts": graph_artifacts,
        "metric_table": metric_table,
        "diagnostics": diagnostics,
    }


def _save_continuous_export(run_dir: Path, split_name: str, values: Mapping[str, Any]) -> None:
    epoch = int(values["diagnostics"]["checkpoint_epoch"])
    prediction = run_dir / f"best_{split_name}_predictions.pt"
    graph = run_dir / f"best_{split_name}_graphs.pt"
    metric = run_dir / f"best_{split_name}_metric_table.csv"
    diagnostics = run_dir / f"best_{split_name}_diagnostics.json"
    _atomic_torch_save({"epoch": epoch, "prediction_result": values["prediction_result"]}, prediction)
    _atomic_torch_save({"epoch": epoch, "graph_artifacts": values["graph_artifacts"]}, graph)
    _atomic_csv_save(values["metric_table"], metric)
    _atomic_json_save(values["diagnostics"], diagnostics)
    analysis = run_dir / "analysis" / split_name
    analysis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prediction, analysis / "predictions.pt")
    shutil.copy2(graph, analysis / "graphs.pt")
    shutil.copy2(metric, analysis / "metric_table.csv")
    shutil.copy2(diagnostics, analysis / "diagnostics.json")


def _invalid_candle_mask(decoded: Tensor) -> Tensor:
    values = torch.as_tensor(decoded)
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


def _probability_policy_label(temperature: float) -> str:
    return f"temperature_{float(temperature):g}".replace(".", "p")


def _finalise_probability_aggregates(
    *,
    raw_probability_sum: Tensor,
    sampling_probability_sum: Tensor,
    raw_max_probability_sum: Tensor,
    sampling_max_probability_sum: Tensor,
    window_count: int,
    future_steps: int,
) -> dict[str, Any]:
    """Return exact all-window probability means in the policy schema."""

    if int(window_count) <= 0:
        raise ValueError("Probability aggregation requires at least one window.")
    raw_mean = (raw_probability_sum / float(window_count)).to(torch.float32)
    sampling_mean = (
        sampling_probability_sum / float(window_count)
    ).to(torch.float32)
    raw_mean_max = (
        raw_max_probability_sum / float(window_count)
    ).to(torch.float32)
    sampling_mean_max = (
        sampling_max_probability_sum / float(window_count)
    ).to(torch.float32)
    if int(raw_mean.shape[0]) != int(future_steps):
        raise ValueError(
            "Probability future-step axis differs from prediction length."
        )
    for label, values in (
        ("raw model", raw_mean),
        ("sampling policy", sampling_mean),
    ):
        if values.ndim != 3:
            raise ValueError(
                f"{label} probability aggregate must have shape [P,N,V]."
            )
        row_sum_error = float(
            (values.sum(dim=-1) - 1.0).abs().max().item()
        )
        if row_sum_error > 2.0e-5:
            raise RuntimeError(
                f"{label} probability aggregate is not normalised; "
                f"maximum error={row_sum_error:.3e}."
            )
    return {
        "raw_model_mean_probability": raw_mean,
        "sampling_policy_mean_probability": sampling_mean,
        "raw_model_mean_max_probability": raw_mean_max,
        "sampling_policy_mean_max_probability": sampling_mean_max,
        "probability_future_steps": list(range(1, int(future_steps) + 1)),
        "probability_aggregate_window_count": int(window_count),
        "probability_aggregate_scope": (
            "mean over every decoded split window; future-step and asset "
            "axes retained"
        ),
        "probability_aggregate_schema_version": 1,
    }


def _compute_token_probability_aggregates(
    *,
    model: nn.Module,
    kind: str,
    dataset: CachedTokenGraphDataset,
    config: Mapping[str, Any],
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    description: str,
) -> dict[str, Any]:
    """Infer exact probabilities only; do not sample or call the decoder."""

    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["export_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        pin_memory=device.type == "cuda",
    )
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    raw_sum: Tensor | None = None
    sampling_sum: Tensor | None = None
    raw_max_sum: Tensor | None = None
    sampling_max_sum: Tensor | None = None
    window_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=description,
            leave=False,
            dynamic_ncols=True,
        ):
            context, _ = _token_batch(batch, device=device)
            with _autocast_context(device, use_amp):
                logits, _ = _forward_token_final(
                    model,
                    kind,
                    context,
                    include_components=False,
                )
            raw = torch.softmax(logits.float(), dim=-1)
            sampling = token_selection_probabilities(
                logits.float(),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
            if raw_sum is None:
                shape = tuple(int(value) for value in raw.shape[1:])
                raw_sum = torch.zeros(shape, dtype=torch.float64)
                sampling_sum = torch.zeros(shape, dtype=torch.float64)
                raw_max_sum = torch.zeros(shape[:-1], dtype=torch.float64)
                sampling_max_sum = torch.zeros(shape[:-1], dtype=torch.float64)
            raw_sum.add_(raw.sum(dim=0).detach().cpu().to(torch.float64))
            sampling_sum.add_(
                sampling.sum(dim=0).detach().cpu().to(torch.float64)
            )
            raw_max_sum.add_(
                raw.max(dim=-1).values.sum(dim=0)
                .detach().cpu().to(torch.float64)
            )
            sampling_max_sum.add_(
                sampling.max(dim=-1).values.sum(dim=0)
                .detach().cpu().to(torch.float64)
            )
            window_count += int(context.shape[0])
    if (
        raw_sum is None
        or sampling_sum is None
        or raw_max_sum is None
        or sampling_max_sum is None
        or int(window_count) != len(dataset)
    ):
        raise RuntimeError(
            "Probability backfill did not cover the complete split."
        )
    return _finalise_probability_aggregates(
        raw_probability_sum=raw_sum,
        sampling_probability_sum=sampling_sum,
        raw_max_probability_sum=raw_max_sum,
        sampling_max_probability_sum=sampling_max_sum,
        window_count=window_count,
        future_steps=int(config["data"]["prediction_length"]),
    )


def _load_existing_policy_token_payload(
    *,
    run_dir: Path,
    split_name: str,
    policy: str,
) -> tuple[dict[str, Any], Path, Path]:
    primary = (
        run_dir
        / "temperature_sweep"
        / policy
        / f"{split_name}_tokens.pt"
    )
    analysis = run_dir / "analysis" / split_name / policy / "tokens.pt"
    source = primary if primary.is_file() else analysis
    if not source.is_file():
        raise FileNotFoundError(
            "Probability backfill requires an existing sampled-policy token "
            f"artifact for {split_name}/{policy}: {primary}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("token_artifacts"), dict
    ):
        raise TypeError(f"Invalid sampled token payload: {source}")
    return payload, primary, analysis


def _backfill_probability_aggregates(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    model: nn.Module,
    datasets: Mapping[str, CachedTokenGraphDataset],
    device: torch.device,
    run_dir: Path,
) -> None:
    """Repair old sampled policies without resampling or re-decoding prices."""

    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    requested = tuple(str(value) for value in args.decode_splits)
    if not requested or any(
        value not in {"validation", "test"} for value in requested
    ):
        raise ValueError(
            "decode_splits must contain validation and/or test for backfill."
        )
    policy = _probability_policy_label(float(args.temperature))
    for split_name in requested:
        payload, primary, analysis = _load_existing_policy_token_payload(
            run_dir=run_dir,
            split_name=split_name,
            policy=policy,
        )
        token_artifacts = payload["token_artifacts"]
        expected_policy = {
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
        }
        for key, expected in expected_policy.items():
            observed = token_artifacts.get(key)
            mismatch = observed is None
            if isinstance(expected, float) and observed is not None:
                mismatch = not math.isclose(
                    float(observed),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            elif observed is not None:
                mismatch = int(observed) != expected
            if mismatch:
                raise ValueError(
                    f"Existing {split_name}/{policy} policy has "
                    f"{key}={observed!r}; requested {expected!r}. "
                    "Refusing to attach mismatched probabilities to the "
                    "saved sampled paths."
                )
        aggregates = _compute_token_probability_aggregates(
            model=model,
            kind=str(config["model_kind"]),
            dataset=datasets[split_name],
            config=config,
            device=device,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            description=f"probability backfill {split_name}",
        )
        token_artifacts.update(aggregates)
        payload["probability_backfilled_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _atomic_torch_save(payload, primary)
        analysis.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, analysis)

        sampled_path = (
            run_dir
            / "temperature_sweep"
            / policy
            / f"{split_name}_sampled_price_paths.pt"
        )
        if sampled_path.is_file():
            sampled_payload = torch.load(
                sampled_path,
                map_location="cpu",
                weights_only=False,
            )
            sampled_values = sampled_payload.get(
                "sampled_price_path_artifacts"
            )
            if isinstance(sampled_values, dict):
                sampled_values.update(aggregates)
                _atomic_torch_save(sampled_payload, sampled_path)
                sampled_analysis = (
                    run_dir
                    / "analysis"
                    / split_name
                    / policy
                    / "sampled_price_paths.pt"
                )
                sampled_analysis.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sampled_path, sampled_analysis)
        print(
            f"Backfilled exact probabilities: {run_dir.name} / "
            f"{split_name} / {policy}"
        )


def _decode_token_split(
    *,
    model: nn.Module,
    kind: str,
    dataset: CachedTokenGraphDataset,
    split_name: str,
    config: Mapping[str, Any],
    device: torch.device,
    tokenizer: KronosTokenizerAdapter,
    args: argparse.Namespace,
    train_split: Mapping[str, Any],
    checkpoint_epoch: int,
    run_dir: Path,
) -> None:
    training = config["training"]
    loader = _build_loader(
        dataset,
        batch_size=int(training["export_batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=int(args.sampling_seed),
        pin_memory=device.type == "cuda",
    )
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    _set_seed(int(args.sampling_seed))
    windows = len(dataset)
    prediction = int(config["data"]["prediction_length"])
    nodes = dataset.num_assets
    sampled_tokens = torch.empty(args.sample_count, windows, prediction, nodes, dtype=torch.int16)
    sampled_close = torch.empty(args.sample_count, windows, prediction, nodes, 1, dtype=torch.float32)

    # Probability aggregates are accumulated over every saved window while
    # preserving the future-step and asset axes.  This is the exact quantity
    # required by Graph Hub: E_window[p(token | context)] for every token.
    # We deliberately do not save the enormous per-window 1,024-way tensor.
    raw_probability_sum: Tensor | None = None
    sampling_probability_sum: Tensor | None = None
    raw_max_probability_sum: Tensor | None = None
    sampling_max_probability_sum: Tensor | None = None
    probability_window_count = 0

    cursor = 0
    invalid_paths = invalid_path_total = invalid_ensemble = invalid_ensemble_total = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"decode {split_name}", leave=False, dynamic_ncols=True):
            context, _ = _token_batch(batch, device=device)
            with _autocast_context(device, use_amp):
                logits, _ = _forward_token_final(model, kind, context, include_components=False)

            # Raw checkpoint probabilities before any sampling policy.
            raw_probabilities = torch.softmax(logits.float(), dim=-1)

            # Exact post-temperature/top-k/top-p categorical distribution used
            # by the sampler.  Computing it once avoids repeating the full
            # filtering and softmax operation for each of the ten paths.
            sampling_probabilities = token_selection_probabilities(
                logits.float(),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
            )

            if raw_probability_sum is None:
                aggregate_shape = tuple(int(value) for value in raw_probabilities.shape[1:])
                raw_probability_sum = torch.zeros(
                    aggregate_shape, dtype=torch.float64
                )
                sampling_probability_sum = torch.zeros(
                    aggregate_shape, dtype=torch.float64
                )
                max_shape = aggregate_shape[:-1]
                raw_max_probability_sum = torch.zeros(
                    max_shape, dtype=torch.float64
                )
                sampling_max_probability_sum = torch.zeros(
                    max_shape, dtype=torch.float64
                )

            raw_probability_sum.add_(
                raw_probabilities.sum(dim=0).detach().cpu().to(torch.float64)
            )
            sampling_probability_sum.add_(
                sampling_probabilities.sum(dim=0).detach().cpu().to(torch.float64)
            )
            raw_max_probability_sum.add_(
                raw_probabilities.max(dim=-1).values.sum(dim=0)
                .detach().cpu().to(torch.float64)
            )
            sampling_max_probability_sum.add_(
                sampling_probabilities.max(dim=-1).values.sum(dim=0)
                .detach().cpu().to(torch.float64)
            )

            current = int(context.shape[0])
            probability_window_count += current

            vocabulary_size = int(sampling_probabilities.shape[-1])
            flat_probabilities = sampling_probabilities.reshape(
                -1, vocabulary_size
            )
            sampled_flat = torch.multinomial(
                flat_probabilities,
                num_samples=int(args.sample_count),
                replacement=True,
            )
            samples = (
                sampled_flat
                .reshape(current, prediction, nodes, int(args.sample_count))
                .permute(3, 0, 1, 2)
                .contiguous()
                .cpu()
                .long()
            )
            start, stop = cursor, cursor + current
            sampled_tokens[:, start:stop] = samples.to(torch.int16)
            context_pairs = torch.as_tensor(batch["context_tokens"]).cpu().long()
            means = torch.as_tensor(batch["context_mean"]).cpu().float()
            stds = torch.as_tensor(batch["context_std"]).cpu().float()
            expanded_context = context_pairs.unsqueeze(0).expand(args.sample_count, -1, -1, -1, -1).reshape(args.sample_count * current, dataset.context_length, nodes, 2)
            expanded_future = samples.reshape(args.sample_count * current, prediction, nodes)
            expanded_mean = means.unsqueeze(0).expand(args.sample_count, -1, -1, -1).reshape(args.sample_count * current, nodes, 6)
            expanded_std = stds.unsqueeze(0).expand(args.sample_count, -1, -1, -1).reshape(args.sample_count * current, nodes, 6)
            decoded = tokenizer.decode_coarse_token_path(
                expanded_context,
                expanded_future,
                mean=expanded_mean,
                std=expanded_std,
                series_batch_size=int(args.decode_series_batch_size),
                return_full_path=False,
            ).reshape(args.sample_count, current, prediction, nodes, 5)
            sampled_close[:, start:stop] = decoded[..., 3:4]
            invalid = _invalid_candle_mask(decoded)
            invalid_paths += int(invalid.sum().item())
            invalid_path_total += int(invalid.numel())
            ensemble_invalid_mask = _invalid_candle_mask(decoded.float().mean(dim=0))
            invalid_ensemble += int(ensemble_invalid_mask.sum().item())
            invalid_ensemble_total += int(ensemble_invalid_mask.numel())
            cursor = stop
    if cursor != windows or not torch.isfinite(sampled_close).all():
        raise RuntimeError("Sampled decoding did not produce a complete finite path.")
    if (
        raw_probability_sum is None
        or sampling_probability_sum is None
        or raw_max_probability_sum is None
        or sampling_max_probability_sum is None
        or probability_window_count != windows
    ):
        raise RuntimeError(
            "Probability aggregation did not cover every decoded window."
        )

    raw_mean_probability = (
        raw_probability_sum / float(probability_window_count)
    ).to(torch.float32)
    sampling_mean_probability = (
        sampling_probability_sum / float(probability_window_count)
    ).to(torch.float32)
    raw_mean_max_probability = (
        raw_max_probability_sum / float(probability_window_count)
    ).to(torch.float32)
    sampling_mean_max_probability = (
        sampling_max_probability_sum / float(probability_window_count)
    ).to(torch.float32)

    for label, values in (
        ("raw model", raw_mean_probability),
        ("sampling policy", sampling_mean_probability),
    ):
        row_sum_error = float(
            (values.sum(dim=-1) - 1.0).abs().max().item()
        )
        if row_sum_error > 2.0e-5:
            raise RuntimeError(
                f"{label} probability aggregate is not normalised; "
                f"maximum error={row_sum_error:.3e}."
            )

    evaluation_indices = torch.tensor([int(value) - 1 for value in config["data"]["evaluation_horizons"]], dtype=torch.long)
    ensemble_dense = sampled_close.mean(dim=0)
    prediction_result = {
        "y_pred": ensemble_dense.index_select(1, evaluation_indices),
        "y_true": torch.as_tensor(dataset.cache["evaluation_true"])[..., 3:4].float(),
        "last_context_target": torch.as_tensor(dataset.cache["last_context_target"])[..., 3:4].float(),
        "sample_idx": torch.as_tensor(dataset.cache["sample_idx"]).long(),
        "origin_idx": torch.as_tensor(dataset.cache["origin_idx"]).long(),
        "target_indices": torch.as_tensor(dataset.cache["target_indices"]).long().index_select(1, evaluation_indices),
        "channels": ["close"],
        "horizons": list(config["data"]["evaluation_horizons"]),
        "asset_cols": list(dataset.asset_cols),
        "output_space": "raw",
    }
    evaluator = ForecastEvaluator(prediction_result=prediction_result, train_split=dict(train_split))
    metric_results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
        bootstrap=True,
        n_bootstrap=10000,
        bootstrap_seed=42,
    )
    metric_table = make_evaluation_table(metric_results, evaluator.horizons, evaluator.channels)
    policy = f"temperature_{float(args.temperature):g}".replace(".", "p")
    policy_root = run_dir / "temperature_sweep" / policy
    analysis = run_dir / "analysis" / split_name / policy
    policy_root.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    sampled_artifacts = {
        "sampled_s1_paths": sampled_tokens,
        "sampled_close_paths": sampled_close,
        "sampled_close_paths_at_evaluation_horizons": sampled_close.index_select(
            2, evaluation_indices
        ),
        "ensemble_mean_close_path": ensemble_dense,
        "evaluation_true": prediction_result["y_true"],
        "last_context_target": prediction_result["last_context_target"],
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "dense_target_indices": torch.as_tensor(
            dataset.cache["target_indices"]
        ).long(),
        "evaluation_target_indices": prediction_result["target_indices"],
        "dates": list(dataset.cache.get("dates", [])),
        "asset_cols": list(dataset.asset_cols),
        "future_steps": list(range(1, prediction + 1)),
        "evaluation_horizons": list(config["data"]["evaluation_horizons"]),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "sample_count": int(args.sample_count),
        "sampling_seed": int(args.sampling_seed),
        "averaging_space": "decoded raw continuous Close",
        "raw_model_mean_probability": raw_mean_probability,
        "sampling_policy_mean_probability": sampling_mean_probability,
        "raw_model_mean_max_probability": raw_mean_max_probability,
        "sampling_policy_mean_max_probability": sampling_mean_max_probability,
        "probability_aggregate_window_count": int(probability_window_count),
        "probability_aggregate_scope": (
            "mean over every decoded split window; future-step and asset axes retained"
        ),
        "probability_aggregate_schema_version": 1,
    }
    diagnostics = {
        "split": split_name,
        "checkpoint_epoch": int(checkpoint_epoch),
        "sample_path_invalid_candle_rate_percent": 100.0 * invalid_paths / max(invalid_path_total, 1),
        "ensemble_invalid_candle_rate_percent": 100.0 * invalid_ensemble / max(invalid_ensemble_total, 1),
        "bootstrap_sessions": 10000,
        "probability_aggregate_window_count": int(probability_window_count),
        "raw_probability_row_sum_max_error": float(
            (raw_mean_probability.sum(dim=-1) - 1.0).abs().max().item()
        ),
        "sampling_probability_row_sum_max_error": float(
            (sampling_mean_probability.sum(dim=-1) - 1.0).abs().max().item()
        ),
    }
    paths = {
        "predictions": policy_root / f"{split_name}_predictions.pt",
        "sampled": policy_root / f"{split_name}_sampled_price_paths.pt",
        "metrics": policy_root / f"{split_name}_metric_table.csv",
        "diagnostics": policy_root / f"{split_name}_diagnostics.json",
        "tokens": policy_root / f"{split_name}_tokens.pt",
    }
    _atomic_torch_save({"epoch": checkpoint_epoch, "prediction_result": prediction_result}, paths["predictions"])
    _atomic_torch_save({"sampled_price_path_artifacts": sampled_artifacts}, paths["sampled"])
    _atomic_torch_save({"epoch": checkpoint_epoch, "token_artifacts": {
        "sampled_s1_paths": sampled_tokens,
        "sampled_s1_evaluation": sampled_tokens.index_select(2, evaluation_indices),
        "raw_model_mean_probability": raw_mean_probability,
        "sampling_policy_mean_probability": sampling_mean_probability,
        "raw_model_mean_max_probability": raw_mean_max_probability,
        "sampling_policy_mean_max_probability": sampling_mean_max_probability,
        "probability_future_steps": list(range(1, prediction + 1)),
        "probability_aggregate_window_count": int(probability_window_count),
        "probability_aggregate_scope": (
            "mean over every decoded split window; future-step and asset axes retained"
        ),
        "probability_aggregate_schema_version": 1,
        "target_s1": torch.as_tensor(dataset.cache["target_s1"]).to(torch.int16),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "sample_count": int(args.sample_count),
        "evaluation_horizons": list(config["data"]["evaluation_horizons"]),
        "asset_cols": list(dataset.asset_cols),
        "sample_idx": prediction_result["sample_idx"],
        "origin_idx": prediction_result["origin_idx"],
        "target_indices": prediction_result["target_indices"],
        "dates": list(dataset.cache.get("dates", [])),
    }}, paths["tokens"])
    _atomic_csv_save(metric_table, paths["metrics"])
    _atomic_json_save(diagnostics, paths["diagnostics"])
    graph_source = run_dir / f"best_{split_name}_graphs.pt"
    if not graph_source.is_file():
        raise FileNotFoundError(graph_source)
    shutil.copy2(paths["predictions"], analysis / "predictions.pt")
    shutil.copy2(paths["sampled"], analysis / "sampled_price_paths.pt")
    shutil.copy2(paths["metrics"], analysis / "metric_table.csv")
    shutil.copy2(paths["diagnostics"], analysis / "diagnostics.json")
    shutil.copy2(paths["tokens"], analysis / "tokens.pt")
    shutil.copy2(graph_source, analysis / "graphs.pt")


def _metadata(
    *,
    config: Mapping[str, Any],
    run_name: str,
    model: nn.Module,
    best_epoch: int,
    best_score: float,
    epochs_completed: int,
    datasets: Mapping[str, Dataset],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    def git_value(args: Sequence[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    reference_dataset = datasets["train"]
    asset_cols = tuple(
        str(value)
        for value in getattr(reference_dataset, "asset_cols")
    )
    model_values = config.get("model", {})
    graph_values = (
        model_values.get("graph", {})
        if isinstance(model_values, Mapping)
        else {}
    )
    temporal_values = (
        model_values.get("temporal", {})
        if isinstance(model_values, Mapping)
        else {}
    )
    if not temporal_values and isinstance(model_values, Mapping):
        temporal_stack = model_values.get("temporal_stack", {})
        if isinstance(temporal_stack, Mapping):
            modern = temporal_stack.get("modern_tcn", {})
            transformer = temporal_stack.get("transformer", {})
            if str(temporal_stack.get("family", "")).startswith("modern_tcn"):
                temporal_values = modern if isinstance(modern, Mapping) else {}
            elif isinstance(transformer, Mapping):
                temporal_values = transformer

    graph_heads = graph_values.get("num_heads_per_block")
    if graph_heads is None:
        graph_heads = graph_values.get("num_heads_per_layer")
    if graph_heads is None and graph_values.get("num_heads") is not None:
        graph_heads = [int(graph_values["num_heads"])]
    graph_widths = graph_values.get("hidden_dims_per_block")
    if graph_widths is None and graph_values.get("hidden_dim") is not None:
        graph_widths = [int(graph_values["hidden_dim"])]
    graph_activations = graph_values.get("activations_per_block")
    if graph_activations is None and graph_values.get("activation") is not None:
        graph_activations = [str(graph_values["activation"])]
    parameter_count = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "status": "completed",
        "run_name": run_name,
        "model_family": str(config["model_family"]),
        "model_kind": str(config["model_kind"]),
        "run_signature": _signature(config),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "epochs_completed": int(epochs_completed),
        "selection_split": "test",
        "selection_metric": str(config["training"]["selection_metric"]),
        "dense_token_objective": config["training"].get("loss", {}).get(
            "dense_objective"
        ),
        "dense_auxiliary_horizons": config["training"].get("loss", {}).get(
            "dense_auxiliary_horizons"
        ),
        "dense_auxiliary_weight": config["training"].get("loss", {}).get(
            "dense_auxiliary_weight"
        ),
        "final_origin_future_steps": config["training"].get("loss", {}).get(
            "final_origin_future_steps"
        ),
        "dense_future_steps_per_origin": config["training"].get(
            "loss", {}
        ).get("future_steps_per_origin"),
        "dense_origin_chunk_size": config["training"].get(
            "loss", {}
        ).get("origin_chunk_size"),
        "training_batch_size": int(config["training"]["batch_size"]),
        "selection_batch_size": int(
            config["training"]["selection_batch_size"]
        ),
        "export_batch_size": int(config["training"]["export_batch_size"]),
        "test_set_contaminated": True,
        "do_not_report": True,
        "trainable_parameters": parameter_count,
        "trainable_parameter_count": parameter_count,
        "asset_cols": list(asset_cols),
        "num_nodes": len(asset_cols),
        "context_length": int(config["data"]["context_length"]),
        "prediction_length": config["data"].get("prediction_length"),
        "horizons": list(
            config["data"].get(
                "evaluation_horizons",
                config["data"].get("horizons", []),
            )
        ),
        "num_st_blocks": (
            None if graph_heads is None else len(tuple(graph_heads))
        ),
        "graph_type": graph_values.get("type", graph_values.get("graph_type")),
        "graph_heads": (
            None if graph_heads is None else int(tuple(graph_heads)[-1])
        ),
        "graph_heads_per_layer": (
            None if graph_heads is None else [int(value) for value in graph_heads]
        ),
        "graph_hidden_dims_per_layer": (
            None if graph_widths is None else [int(value) for value in graph_widths]
        ),
        "graph_activations_per_layer": (
            None
            if graph_activations is None
            else [str(value) for value in graph_activations]
        ),
        "d_model": temporal_values.get("d_model"),
        "transformer_num_heads": temporal_values.get("num_heads"),
        "transformer_num_layers": temporal_values.get("num_layers"),
        "prior_type": (
            model_values.get("prior", {}).get("type")
            if isinstance(model_values, Mapping)
            and isinstance(model_values.get("prior"), Mapping)
            else None
        ),
        "state_pathway": str(config["model_kind"]) in {
            "modern_tcn_token",
            "dense_transformer_token",
            "dimitri_v2_token",
            "dimitri_v2_continuous",
        },
        "train_windows": len(datasets["train"]),
        "validation_windows": len(datasets["validation"]),
        "test_windows": len(datasets["test"]),
        "project_git_commit": git_value(("rev-parse", "HEAD")),
        "project_git_branch": git_value(("branch", "--show-current")),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _train(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: nn.Module,
    datasets: Mapping[str, Dataset],
    train_split: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
) -> None:
    training = config["training"]
    kind = str(config["model_kind"])
    optimizer, scheduler = _build_optimizer(model, config)
    use_amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = _new_grad_scaler(use_amp)
    maximise = str(training["selection_direction"]) == "maximise"
    best_score = -math.inf if maximise else math.inf
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1
    history: list[dict[str, Any]] = []
    last_path = run_dir / "last_checkpoint.pt"
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(last_path)
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config_signature"] != _signature(config):
            raise AssertionError("Resume config differs from saved checkpoint.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"])
        history_path = run_dir / "history.csv"
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict("records")

    # A previous process can finish optimisation and then fail during the much
    # larger selected-checkpoint export/decoder phase.  Resuming such a run
    # must not reopen an already exhausted early-stopping budget.
    if bad_epochs >= int(training["patience"]):
        print(
            "The resumed checkpoint had already satisfied early stopping: "
            f"bad_epochs={bad_epochs}, patience={int(training['patience'])}. "
            "Skipping further optimisation and proceeding to best-checkpoint "
            "export."
        )
        start_epoch = int(training["max_epochs"]) + 1

    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        if str(training["scheduler"]) == "modern_tcn_type3_delayed":
            _set_delayed_schedule(optimizer, config, epoch)
        if kind in TOKEN_MODEL_KINDS:
            train_values = _train_token_epoch(
                model=model,
                kind=kind,
                dataset=datasets["train"],
                config=config,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                epoch=epoch,
            )
            selection = _evaluate_token(
                model=model,
                kind=kind,
                dataset=datasets["test"],
                config=config,
                device=device,
                description="test selection",
            )
            score = float(selection["mean_top1_accuracy"])
            record = {
                "epoch": epoch,
                **_current_lrs(optimizer),
                **train_values,
                "selection_score": score,
                "test_mean_top1_accuracy": score,
                "test_mean_cross_entropy": float(selection["mean_cross_entropy"]),
            }
            for step in range(int(config["data"]["prediction_length"])):
                record[f"test_top1_accuracy_h{step + 1}"] = float(selection["top1_accuracy_by_step"][step])
                record[f"test_cross_entropy_h{step + 1}"] = float(selection["cross_entropy_by_step"][step])
            for key, value in selection.items():
                if key.startswith("block_"):
                    record[key] = value
        else:
            train_values = _train_continuous_epoch(
                model=model,
                dataset=datasets["train"],
                config=config,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
            selection = _evaluate_continuous(
                model=model,
                dataset=datasets["test"],
                config=config,
                device=device,
                description="test selection",
            )
            score = float(selection["mean_log_mae"])
            record = {
                "epoch": epoch,
                **_current_lrs(optimizer),
                **train_values,
                "selection_score": score,
                "test_mean_log_mae": score,
            }
            for horizon, value in zip(config["data"]["horizons"], selection["log_mae_by_horizon"], strict=True):
                record[f"test_cumulative_log_change_mae_h{int(horizon)}"] = float(value)
            for key, value in selection.items():
                if key.startswith("block_"):
                    record[key] = value
        improved = score > best_score + float(training["min_delta"]) if maximise else score < best_score - float(training["min_delta"])
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_score": score,
                    "config_signature": _signature(config),
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            bad_epochs += 1
        history.append(record)
        _atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
        # Store the scheduler state for the *next* epoch.  Saving before this
        # step would repeat one cosine LR after an interrupted/resumed V2 run.
        if scheduler is not None:
            scheduler.step()
        _atomic_torch_save(
            _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                config=config,
            ),
            last_path,
        )
        print(json.dumps({"epoch": epoch, "score": score, "best": best_score, "bad_epochs": bad_epochs}))
        if bad_epochs >= int(training["patience"]):
            break
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    for split_name in ("train", "validation", "test"):
        if kind in TOKEN_MODEL_KINDS:
            values = _export_token_split(
                model=model,
                kind=kind,
                dataset=datasets[split_name],
                split_name=split_name,
                config=config,
                device=device,
                checkpoint_epoch=int(checkpoint["epoch"]),
            )
            _save_token_export(run_dir, split_name, values)
        else:
            values = _export_continuous_split(
                model=model,
                dataset=datasets[split_name],
                split_name=split_name,
                config=config,
                device=device,
                checkpoint_epoch=int(checkpoint["epoch"]),
                train_split=train_split,
                bootstrap=split_name in {"validation", "test"},
            )
            _save_continuous_export(run_dir, split_name, values)
    metadata = _metadata(
        config=config,
        run_name=args.run_name,
        model=model,
        best_epoch=int(checkpoint["epoch"]),
        best_score=float(checkpoint["selection_score"]),
        epochs_completed=int(history[-1]["epoch"]),
        datasets=datasets,
    )
    _atomic_json_save(metadata, run_dir / "run_metadata.json")


def _decode(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    model: nn.Module,
    datasets: Mapping[str, CachedTokenGraphDataset],
    train_split: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
) -> None:
    if str(config["model_kind"]) not in TOKEN_MODEL_KINDS:
        raise ValueError("Only token models can be decoded.")
    if args.forecasting_config is None:
        raise ValueError("--forecasting-config is required for decoding.")
    metadata = _load_json(run_dir / "run_metadata.json")
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    tokenizer = KronosTokenizerAdapter.from_config(
        load_yaml(args.forecasting_config),
        series_batch_size=int(args.decode_series_batch_size),
    ).load()
    requested = tuple(str(value) for value in args.decode_splits)
    if not requested or any(value not in {"validation", "test"} for value in requested):
        raise ValueError("decode_splits must contain validation and/or test.")
    for split_name in requested:
        _decode_token_split(
            model=model,
            kind=str(config["model_kind"]),
            dataset=datasets[split_name],
            split_name=split_name,
            config=config,
            device=device,
            tokenizer=tokenizer,
            args=args,
            train_split=train_split,
            checkpoint_epoch=int(metadata["best_epoch"]),
            run_dir=run_dir,
        )
    _atomic_json_save(
        {
            "selected_policy": "temperature_1",
            "selected_temperature": float(args.temperature),
            "sample_count": int(args.sample_count),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "sampling_seed": int(args.sampling_seed),
            "decoded_splits": list(requested),
        },
        run_dir / "temperature_sweep" / "temperature_selection.json",
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_json(args.config)
    model_kind = str(config.get("model_kind", ""))
    allowed_kinds = {*TOKEN_MODEL_KINDS, "dimitri_v2_continuous"}
    if model_kind not in allowed_kinds:
        raise ValueError(
            f"Unsupported model_kind {model_kind!r}; expected one of "
            f"{sorted(allowed_kinds)}."
        )
    device = _resolve_device(args.device)
    _set_seed(int(config["training"]["seed"]))
    if args.decode_sampled and args.backfill_probability_aggregates:
        raise ValueError(
            "--decode-sampled and --backfill-probability-aggregates are "
            "mutually exclusive."
        )
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=(
            args.resume
            or args.decode_sampled
            or args.backfill_probability_aggregates
        ),
    )
    train_split, validation_split, test_split = _load_raw_splits(args.data_dir)
    raw_splits = {"train": train_split, "validation": validation_split, "test": test_split}
    token_datasets: dict[str, CachedTokenGraphDataset] | None = None
    if str(config["model_kind"]) in TOKEN_MODEL_KINDS:
        token_datasets = _load_token_datasets(args)
        datasets: Mapping[str, Dataset] = token_datasets
        reference_token = token_datasets["train"]
    else:
        datasets = {name: _continuous_dataset(split, config) for name, split in raw_splits.items()}
        reference_token = None
    model = _build_model(
        config=config,
        token_dataset=reference_token,
        train_split=train_split,
        device=device,
    )
    if args.backfill_probability_aggregates:
        if token_datasets is None:
            raise ValueError(
                "Continuous V2 has no token-probability backfill mode."
            )
        _backfill_probability_aggregates(
            args=args,
            config=config,
            model=model,
            datasets=token_datasets,
            device=device,
            run_dir=run_dir,
        )
        return
    if args.decode_sampled:
        if token_datasets is None:
            raise ValueError("Continuous V2 has no token-decoding mode.")
        _decode(
            args=args,
            config=config,
            model=model,
            datasets=token_datasets,
            train_split=train_split,
            device=device,
            run_dir=run_dir,
        )
        return
    _atomic_json_save(config, run_dir / "resolved_config.json")
    _train(
        args=args,
        config=config,
        model=model,
        datasets=datasets,
        train_split=train_split,
        device=device,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()

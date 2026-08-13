from __future__ import annotations

"""Post-train the frozen Kronos coarse decoder for two frozen token forecasters.

The forecaster generates ten fixed coarse-token paths.  Only the pretrained
coarse reconstruction decoder is optimised.  The loss is the stretched-
exponential weighted cumulative-log-change MAE of the ten-path decoded ensemble
at every future minute 1..60.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.evaluation.metrics import ForecastEvaluator
from src.models.dynamic_graph.future_predictor import token_selection_probabilities
from src.models.kronos_decoder_post_training import (
    TrainableKronosCoarseDecoder,
    decoder_state_dict_cpu,
)
from src.training.run_final_token_v2_experiment import (
    _build_model as _build_frozen_forecaster,
    _forward_token_final,
    _load_raw_splits,
)
from src.utils.config import load_yaml
from src.utils.metric_tables import make_evaluation_table


EVALUATION_HORIZONS = (1, 5, 15, 30, 60)
POLICY_NAME = "temperature_1"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--forecasting-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-sample-cache", action="store_true")
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


def _autocast(device: torch.device, enabled: bool):
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
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    values: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "generator": generator,
        "worker_init_fn": _seed_worker if num_workers else None,
        "persistent_workers": bool(num_workers),
    }
    if num_workers:
        values["prefetch_factor"] = 2
    return DataLoader(**values)


def _prepare_run_dir(output_dir: Path, run_name: str, *, overwrite: bool, resume: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_name
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        metadata = run_dir / "run_metadata.json"
        checkpoint = run_dir / "best_checkpoint.pt"
        if metadata.is_file() and checkpoint.is_file():
            if _load_json(metadata).get("status") == "completed":
                raise FileExistsError(f"Completed run already exists: {run_dir}")
        raise FileExistsError(f"Non-empty run requires --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _token_datasets(args: argparse.Namespace) -> dict[str, CachedTokenGraphDataset]:
    paths = {
        "train": args.train_cache,
        "validation": args.validation_cache,
        "test": args.test_cache,
    }
    datasets = {
        name: CachedTokenGraphDataset.from_path(path, data_mode="real")
        for name, path in paths.items()
    }
    reference = datasets["train"]
    for name, dataset in datasets.items():
        if dataset.context_length != 60 or dataset.prediction_length != 60:
            raise ValueError(f"{name} token cache is not the C60/P60 contract.")
        if dataset.asset_cols != reference.asset_cols:
            raise ValueError(f"{name} asset order differs from train.")
        if dataset.s1_id_space != "kronos_original":
            raise ValueError(f"{name} is not in original Kronos s1 ID space.")
    return datasets


def _sample_cache_signature(
    *,
    source_identity: Mapping[str, Any],
    split: str,
    sampling: Mapping[str, Any],
    windows: int,
) -> str:
    values = {
        "source": dict(source_identity),
        "split": str(split),
        "sampling": dict(sampling),
        "windows": int(windows),
    }
    serialised = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _generate_sample_cache(
    *,
    source_dir: Path,
    source_config: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    dataset: CachedTokenGraphDataset,
    split_name: str,
    train_split: Mapping[str, Any],
    sampling: Mapping[str, Any],
    batch_size: int,
    num_workers: int,
    mixed_precision: bool,
    device: torch.device,
    cache_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    source_identity = {
        "folder": source_dir.name,
        "run_name": source_metadata.get("run_name", source_dir.name),
        "run_signature": source_metadata.get("run_signature"),
        "best_epoch": int(source_metadata["best_epoch"]),
        "model_kind": source_config.get("model_kind"),
    }
    signature = _sample_cache_signature(
        source_identity=source_identity,
        split=split_name,
        sampling=sampling,
        windows=len(dataset),
    )
    if cache_path.is_file() and not overwrite:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, Mapping) and cached.get("cache_signature") == signature:
            return dict(cached)

    model = _build_frozen_forecaster(
        config=source_config,
        token_dataset=dataset,
        train_split=train_split,
        device=device,
    )
    checkpoint = torch.load(
        source_dir / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=int(sampling["seed"]),
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    sample_count = int(sampling["sample_count"])
    prediction_length = dataset.prediction_length
    num_assets = dataset.num_assets
    sampled = torch.empty(
        sample_count,
        len(dataset),
        prediction_length,
        num_assets,
        dtype=torch.int16,
    )
    raw_sum: Tensor | None = None
    policy_sum: Tensor | None = None
    cursor = 0
    _set_seed(int(sampling["seed"]))
    use_amp = bool(mixed_precision) and device.type == "cuda"
    model_kind = str(source_config["model_kind"])

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"sample frozen {source_dir.name} {split_name}",
            leave=False,
            dynamic_ncols=True,
        ):
            context = torch.as_tensor(batch["context_tokens"])[..., 0].to(
                device=device, dtype=torch.long, non_blocking=True
            )
            with _autocast(device, use_amp):
                logits, _ = _forward_token_final(
                    model,
                    model_kind,
                    context,
                    include_components=False,
                )
            logits = logits.float()
            raw_probability = logits.softmax(dim=-1)
            policy_probability = token_selection_probabilities(
                logits,
                temperature=float(sampling["temperature"]),
                top_k=int(sampling["top_k"]),
                top_p=float(sampling["top_p"]),
            )
            if raw_sum is None:
                aggregate_shape = tuple(int(value) for value in logits.shape[1:])
                raw_sum = torch.zeros(aggregate_shape, dtype=torch.float64)
                policy_sum = torch.zeros(aggregate_shape, dtype=torch.float64)
            raw_sum.add_(raw_probability.sum(dim=0).cpu().double())
            policy_sum.add_(policy_probability.sum(dim=0).cpu().double())

            batch_size_actual = int(context.shape[0])
            vocabulary = int(policy_probability.shape[-1])
            flat = policy_probability.reshape(-1, vocabulary)
            sampled_flat = torch.multinomial(
                flat,
                num_samples=sample_count,
                replacement=True,
            )
            samples = (
                sampled_flat.reshape(
                    batch_size_actual,
                    prediction_length,
                    num_assets,
                    sample_count,
                )
                .permute(3, 0, 1, 2)
                .contiguous()
                .cpu()
                .to(torch.int16)
            )
            sampled[:, cursor : cursor + batch_size_actual] = samples
            cursor += batch_size_actual

    if cursor != len(dataset) or raw_sum is None or policy_sum is None:
        raise RuntimeError("Frozen-forecaster sampling did not cover the split.")
    raw_mean = (raw_sum / float(len(dataset))).float()
    policy_mean = (policy_sum / float(len(dataset))).float()

    values = {
        "cache_signature": signature,
        "source_forecaster": source_identity,
        "split": split_name,
        "sampled_s1_paths": sampled,
        "raw_model_mean_probability": raw_mean,
        "sampling_policy_mean_probability": policy_mean,
        "sample_count": sample_count,
        "temperature": float(sampling["temperature"]),
        "top_k": int(sampling["top_k"]),
        "top_p": float(sampling["top_p"]),
        "sampling_seed": int(sampling["seed"]),
        "sample_idx": torch.as_tensor(dataset.cache["sample_idx"]).long(),
        "origin_idx": torch.as_tensor(dataset.cache["origin_idx"]).long(),
        "target_indices": torch.as_tensor(dataset.cache["target_indices"]).long(),
        "dates": list(dataset.cache.get("dates", [])),
        "asset_cols": list(dataset.asset_cols),
    }
    _atomic_torch_save(values, cache_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


class DecoderPostTrainingDataset(Dataset):
    """Join one token cache, sampled paths, and raw all-60 Close targets."""

    def __init__(
        self,
        *,
        token_dataset: CachedTokenGraphDataset,
        sampled_cache: Mapping[str, Any],
        raw_split: Mapping[str, Any],
    ) -> None:
        self.token_dataset = token_dataset
        self.sampled_s1_paths = torch.as_tensor(
            sampled_cache["sampled_s1_paths"]
        ).contiguous()
        if self.sampled_s1_paths.ndim != 4:
            raise ValueError("sampled_s1_paths must have shape [S,W,P,N].")
        if int(self.sampled_s1_paths.shape[1]) != len(token_dataset):
            raise ValueError("Sample-cache window count differs from token cache.")
        self.raw_split = raw_split
        self.close_channel = list(raw_split["channels"]).index("close")

    def __len__(self) -> int:
        return len(self.token_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token_item = self.token_dataset[index]
        sample_idx = int(torch.as_tensor(token_item["sample_idx"]).item())
        origin_idx = int(torch.as_tensor(token_item["origin_idx"]).item())
        target_indices = torch.as_tensor(token_item["target_indices"]).long()
        day_values = torch.as_tensor(
            self.raw_split["samples"][sample_idx][0]
        ).float()
        future_close = day_values.index_select(0, target_indices)[
            :, :, self.close_channel
        ]
        last_close = day_values[origin_idx, :, self.close_channel]
        return {
            "context_tokens": token_item["context_tokens"],
            "context_mean": token_item["context_mean"],
            "context_std": token_item["context_std"],
            "sampled_s1_paths": self.sampled_s1_paths[:, index].long(),
            "true_future_close": future_close,
            "last_context_close": last_close,
            "sample_idx": token_item["sample_idx"],
            "origin_idx": token_item["origin_idx"],
            "target_indices": target_indices,
            "date": token_item.get("date"),
        }


class WeightedAll60Loss:
    def __init__(self, weights: Sequence[float]) -> None:
        values = torch.tensor(tuple(float(value) for value in weights), dtype=torch.float32)
        if tuple(values.shape) != (60,):
            raise ValueError("The decoder loss requires exactly 60 weights.")
        if torch.any(values <= 0):
            raise ValueError("All decoder loss weights must be positive.")
        self.weights = values / values.sum()

    def __call__(
        self,
        predicted_close: Tensor,
        true_close: Tensor,
        last_close: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if predicted_close.shape != true_close.shape:
            raise ValueError("Predicted and true Close tensors must match.")
        if predicted_close.ndim != 3 or int(predicted_close.shape[1]) != 60:
            raise ValueError("Close paths must have shape [B,60,N].")
        if tuple(last_close.shape) != (
            int(predicted_close.shape[0]),
            int(predicted_close.shape[2]),
        ):
            raise ValueError("last_close must have shape [B,N].")
        if torch.any(predicted_close <= 0) or torch.any(true_close <= 0) or torch.any(last_close <= 0):
            raise FloatingPointError("Cumulative log-change loss received non-positive prices.")
        predicted_change = predicted_close.log() - last_close.log().unsqueeze(1)
        true_change = true_close.log() - last_close.log().unsqueeze(1)
        absolute_error = (predicted_change - true_change).abs()
        weights = self.weights.to(device=absolute_error.device, dtype=absolute_error.dtype)
        loss = (absolute_error * weights.view(1, 60, 1)).sum(dim=1).mean()
        return loss, absolute_error


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    future_paths = torch.as_tensor(batch["sampled_s1_paths"]).permute(1, 0, 2, 3)
    return {
        "context_s1": torch.as_tensor(batch["context_tokens"])[..., 0].to(
            device=device, dtype=torch.long, non_blocking=True
        ),
        "future_paths": future_paths.to(
            device=device, dtype=torch.long, non_blocking=True
        ),
        "mean": torch.as_tensor(batch["context_mean"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "std": torch.as_tensor(batch["context_std"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "true_close": torch.as_tensor(batch["true_future_close"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "last_close": torch.as_tensor(batch["last_context_close"]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
    }


def _decode_ensemble(
    *,
    decoder: TrainableKronosCoarseDecoder,
    values: Mapping[str, Tensor],
    sample_chunk_size: int,
    device: torch.device,
    mixed_precision: bool,
    keep_paths: bool,
) -> tuple[Tensor, Tensor | None, int, int]:
    future_paths = values["future_paths"]
    sample_count = int(future_paths.shape[0])
    path_sum: Tensor | None = None
    saved_parts: list[Tensor] = []
    invalid = total = 0
    for start in range(0, sample_count, sample_chunk_size):
        stop = min(start + sample_chunk_size, sample_count)
        with _autocast(device, mixed_precision and device.type == "cuda"):
            decoded = decoder.decode_paths(
                context_s1=values["context_s1"],
                future_s1_paths=future_paths[start:stop],
                mean=values["mean"],
                std=values["std"],
                future_only=True,
            )
        decoded = decoded.float()
        invalid_mask = (
            (decoded[..., 1] < torch.maximum(decoded[..., 0], decoded[..., 3]))
            | (decoded[..., 2] > torch.minimum(decoded[..., 0], decoded[..., 3]))
            | (decoded[..., 4] < 0)
        )
        invalid += int(invalid_mask.sum().item())
        total += int(invalid_mask.numel())
        close = decoded[..., 3]
        path_sum = close.sum(dim=0) if path_sum is None else path_sum + close.sum(dim=0)
        if keep_paths:
            saved_parts.append(close.detach().cpu().unsqueeze(-1))
    if path_sum is None:
        raise RuntimeError("No sampled decoder paths were processed.")
    ensemble = path_sum / float(sample_count)
    paths = torch.cat(saved_parts, dim=0) if keep_paths else None
    return ensemble, paths, invalid, total


def _train_epoch(
    *,
    decoder: TrainableKronosCoarseDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loss_function: WeightedAll60Loss,
    sample_chunk_size: int,
    device: torch.device,
    mixed_precision: bool,
    gradient_clip_norm: float,
    epoch: int,
) -> dict[str, float]:
    # Eval mode disables decoder dropout, making the no-grad first pass and
    # gradient replay pass exactly the same deterministic function.
    decoder.eval()
    total_loss = 0.0
    total_windows = 0
    gradient_norm_sum = 0.0
    batches = 0

    for raw_batch in tqdm(
        loader,
        desc=f"decoder train epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    ):
        values = _batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ensemble, _, _, _ = _decode_ensemble(
                decoder=decoder,
                values=values,
                sample_chunk_size=sample_chunk_size,
                device=device,
                mixed_precision=mixed_precision,
                keep_paths=False,
            )

        ensemble_leaf = ensemble.detach().float().requires_grad_(True)
        loss, _ = loss_function(
            ensemble_leaf,
            values["true_close"],
            values["last_close"],
        )
        gradient_at_mean = torch.autograd.grad(loss, ensemble_leaf)[0].detach()

        sample_count = int(values["future_paths"].shape[0])
        for start in range(0, sample_count, sample_chunk_size):
            stop = min(start + sample_chunk_size, sample_count)
            with _autocast(device, mixed_precision and device.type == "cuda"):
                decoded = decoder.decode_paths(
                    context_s1=values["context_s1"],
                    future_s1_paths=values["future_paths"][start:stop],
                    mean=values["mean"],
                    std=values["std"],
                    future_only=True,
                )
                decoded_close = decoded[..., 3].float()
                surrogate = (
                    decoded_close
                    * gradient_at_mean.unsqueeze(0)
                    / float(sample_count)
                ).sum()
            scaler.scale(surrogate).backward()

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), float(gradient_clip_norm)
        )
        scaler.step(optimizer)
        scaler.update()

        current = int(values["true_close"].shape[0])
        total_loss += float(loss.detach().item()) * current
        total_windows += current
        gradient_norm_sum += float(torch.as_tensor(gradient_norm).item())
        batches += 1

    return {
        "loss": total_loss / max(total_windows, 1),
        "gradient_norm": gradient_norm_sum / max(batches, 1),
    }


@torch.no_grad()
def _evaluate_weighted_loss(
    *,
    decoder: TrainableKronosCoarseDecoder,
    loader: DataLoader,
    loss_function: WeightedAll60Loss,
    sample_chunk_size: int,
    device: torch.device,
    mixed_precision: bool,
    description: str,
) -> dict[str, Any]:
    decoder.eval()
    total_loss = 0.0
    total_windows = 0
    error_sum = torch.zeros(60, dtype=torch.float64)
    error_count = 0
    invalid = total = 0
    for raw_batch in tqdm(loader, desc=description, leave=False, dynamic_ncols=True):
        values = _batch_to_device(raw_batch, device)
        ensemble, _, batch_invalid, batch_total = _decode_ensemble(
            decoder=decoder,
            values=values,
            sample_chunk_size=sample_chunk_size,
            device=device,
            mixed_precision=mixed_precision,
            keep_paths=False,
        )
        loss, errors = loss_function(
            ensemble,
            values["true_close"],
            values["last_close"],
        )
        current = int(values["true_close"].shape[0])
        total_loss += float(loss.item()) * current
        total_windows += current
        error_sum += errors.sum(dim=(0, 2)).cpu().double()
        error_count += int(errors.shape[0] * errors.shape[2])
        invalid += batch_invalid
        total += batch_total
    return {
        "weighted_loss": total_loss / max(total_windows, 1),
        "clg_mae_by_horizon": (error_sum / max(error_count, 1)).float(),
        "invalid_sample_candle_rate_percent": 100.0 * invalid / max(total, 1),
    }


def _checkpoint_payload(
    *,
    epoch: int,
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    config_signature: str,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "config_signature": str(config_signature),
        "rng_state": _capture_rng_state(),
    }


def _re_epoch_wrapped_artifact(source: Path, destination: Path, *, epoch: int, nested_key: str) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Invalid source artefact: {source}")
    nested = payload.get(nested_key)
    if not isinstance(nested, Mapping):
        nested = payload
    values = dict(nested)
    values["source_forecaster_epoch"] = payload.get("epoch")
    _atomic_torch_save({"epoch": int(epoch), nested_key: values}, destination)


def _copy_forecaster_artifacts(
    *,
    source_dir: Path,
    run_dir: Path,
    split_name: str,
    decoder_epoch: int,
) -> None:
    graph_source = source_dir / f"best_{split_name}_graphs.pt"
    token_source = source_dir / f"best_{split_name}_tokens.pt"
    token_prediction_source = source_dir / f"best_{split_name}_token_predictions.pt"
    if not token_prediction_source.is_file():
        token_prediction_source = source_dir / f"best_{split_name}_predictions.pt"
    token_metric_source = source_dir / f"best_{split_name}_token_metric_table.csv"
    for path in (graph_source, token_source, token_prediction_source, token_metric_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    _re_epoch_wrapped_artifact(
        graph_source,
        run_dir / f"best_{split_name}_graphs.pt",
        epoch=decoder_epoch,
        nested_key="graph_artifacts",
    )
    _re_epoch_wrapped_artifact(
        token_source,
        run_dir / f"best_{split_name}_tokens.pt",
        epoch=decoder_epoch,
        nested_key="token_artifacts",
    )
    _re_epoch_wrapped_artifact(
        token_prediction_source,
        run_dir / f"best_{split_name}_token_predictions.pt",
        epoch=decoder_epoch,
        nested_key="prediction_result",
    )
    shutil.copy2(
        token_metric_source,
        run_dir / f"best_{split_name}_token_metric_table.csv",
    )


def _save_analysis_copy(run_dir: Path, split_name: str) -> None:
    analysis = run_dir / "analysis" / split_name
    analysis.mkdir(parents=True, exist_ok=True)
    mapping = {
        f"best_{split_name}_predictions.pt": "predictions.pt",
        f"best_{split_name}_graphs.pt": "graphs.pt",
        f"best_{split_name}_tokens.pt": "tokens.pt",
        f"best_{split_name}_token_predictions.pt": "token_predictions.pt",
        f"best_{split_name}_metric_table.csv": "metric_table.csv",
        f"best_{split_name}_token_metric_table.csv": "token_metric_table.csv",
        f"best_{split_name}_diagnostics.json": "diagnostics.json",
    }
    for source_name, target_name in mapping.items():
        shutil.copy2(run_dir / source_name, analysis / target_name)


@torch.no_grad()
def _export_split(
    *,
    decoder: TrainableKronosCoarseDecoder,
    dataset: DecoderPostTrainingDataset,
    token_dataset: CachedTokenGraphDataset,
    sample_cache: Mapping[str, Any],
    split_name: str,
    source_dir: Path,
    run_dir: Path,
    train_split: Mapping[str, Any],
    decoder_epoch: int,
    config: Mapping[str, Any],
    device: torch.device,
    baseline: bool,
) -> dict[str, Any]:
    training = config["training"]
    loader = _loader(
        dataset,
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        seed=int(training["seed"]),
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    loss_function = WeightedAll60Loss(config["loss"]["weights"])
    sample_chunk_size = int(training["sample_chunk_size"])
    mixed_precision = bool(training["mixed_precision"])

    ensemble_parts: list[Tensor] = []
    true_parts: list[Tensor] = []
    last_parts: list[Tensor] = []
    sampled_parts: list[Tensor] = []
    invalid = total = 0
    for raw_batch in tqdm(
        loader,
        desc=("baseline" if baseline else "post-trained") + f" export {split_name}",
        leave=False,
        dynamic_ncols=True,
    ):
        values = _batch_to_device(raw_batch, device)
        keep_paths = True
        ensemble, paths, batch_invalid, batch_total = _decode_ensemble(
            decoder=decoder,
            values=values,
            sample_chunk_size=sample_chunk_size,
            device=device,
            mixed_precision=mixed_precision,
            keep_paths=keep_paths,
        )
        ensemble_parts.append(ensemble.cpu().unsqueeze(-1))
        true_parts.append(values["true_close"].cpu().unsqueeze(-1))
        last_parts.append(values["last_close"].cpu().unsqueeze(-1))
        if paths is not None:
            sampled_parts.append(paths)
        invalid += batch_invalid
        total += batch_total

    ensemble_dense = torch.cat(ensemble_parts, dim=0)
    true_dense = torch.cat(true_parts, dim=0)
    last_close = torch.cat(last_parts, dim=0)
    sampled_close = (
        torch.cat(sampled_parts, dim=1)
        if sampled_parts
        else None
    )
    weights = torch.tensor(config["loss"]["weights"], dtype=torch.float32)
    all_loss, all_errors = loss_function(
        ensemble_dense[..., 0], true_dense[..., 0], last_close[..., 0]
    )
    indices = torch.tensor([value - 1 for value in EVALUATION_HORIZONS], dtype=torch.long)
    target_indices = torch.as_tensor(token_dataset.cache["target_indices"]).long()
    prediction_result = {
        "y_pred": ensemble_dense.index_select(1, indices),
        "y_true": true_dense.index_select(1, indices),
        "last_context_target": last_close,
        "sample_idx": torch.as_tensor(token_dataset.cache["sample_idx"]).long(),
        "origin_idx": torch.as_tensor(token_dataset.cache["origin_idx"]).long(),
        "target_indices": target_indices.index_select(1, indices),
        "channels": ["close"],
        "horizons": list(EVALUATION_HORIZONS),
        "asset_cols": list(token_dataset.asset_cols),
        "output_space": "raw",
        "decoder_checkpoint_epoch": int(decoder_epoch),
        "source_forecaster_epoch": int(config["source_forecaster"]["best_epoch"]),
    }
    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
        train_split=dict(train_split),
    )
    metric_results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
        bootstrap=split_name in {"validation", "test"},
        n_bootstrap=10000,
        bootstrap_seed=42,
    )
    metric_table = make_evaluation_table(
        metric_results,
        evaluator.horizons,
        evaluator.channels,
    )
    all60_table = pd.DataFrame(
        {
            "horizon": list(range(1, 61)),
            "weight": weights.tolist(),
            "cumulative_log_change_mae": all_errors.mean(dim=(0, 2)).tolist(),
        }
    )
    all60_table["weighted_contribution"] = (
        all60_table["weight"] * all60_table["cumulative_log_change_mae"]
    )
    diagnostics = {
        "split": split_name,
        "decoder_checkpoint_epoch": int(decoder_epoch),
        "source_forecaster_epoch": int(config["source_forecaster"]["best_epoch"]),
        "weighted_all_60_clg_mae": float(all_loss.item()),
        "sample_path_invalid_candle_rate_percent": 100.0 * invalid / max(total, 1),
        "baseline_decoder": bool(baseline),
        "sample_count": int(config["sampling"]["sample_count"]),
        "temperature": float(config["sampling"]["temperature"]),
        "top_p": float(config["sampling"]["top_p"]),
        "top_k": int(config["sampling"]["top_k"]),
    }

    root = run_dir / "baseline_decoder" if baseline else run_dir
    root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {"epoch": int(decoder_epoch), "prediction_result": prediction_result},
        root / f"best_{split_name}_predictions.pt",
    )
    _atomic_csv_save(metric_table, root / f"best_{split_name}_metric_table.csv")
    _atomic_csv_save(all60_table, root / f"best_{split_name}_all_60_loss_table.csv")
    _atomic_json_save(diagnostics, root / f"best_{split_name}_diagnostics.json")

    if not baseline:
        sampled_artifacts = {
            "sampled_s1_paths": torch.as_tensor(sample_cache["sampled_s1_paths"]),
            "sampled_close_paths": sampled_close,
            "sampled_close_paths_at_evaluation_horizons": (
                None if sampled_close is None else sampled_close.index_select(2, indices)
            ),
            "ensemble_mean_close_path": ensemble_dense,
            "evaluation_true": prediction_result["y_true"],
            "last_context_target": last_close,
            "sample_idx": prediction_result["sample_idx"],
            "origin_idx": prediction_result["origin_idx"],
            "dense_target_indices": target_indices,
            "evaluation_target_indices": prediction_result["target_indices"],
            "dates": list(token_dataset.cache.get("dates", [])),
            "asset_cols": list(token_dataset.asset_cols),
            "future_steps": list(range(1, 61)),
            "evaluation_horizons": list(EVALUATION_HORIZONS),
            "temperature": float(config["sampling"]["temperature"]),
            "top_k": int(config["sampling"]["top_k"]),
            "top_p": float(config["sampling"]["top_p"]),
            "sample_count": int(config["sampling"]["sample_count"]),
            "sampling_seed": int(config["sampling"]["seed"]),
            "averaging_space": "post-trained decoded raw continuous Close",
            "raw_model_mean_probability": torch.as_tensor(
                sample_cache["raw_model_mean_probability"]
            ).float(),
            "sampling_policy_mean_probability": torch.as_tensor(
                sample_cache["sampling_policy_mean_probability"]
            ).float(),
        }
        policy_root = run_dir / "temperature_sweep" / POLICY_NAME
        analysis_policy = run_dir / "analysis" / split_name / POLICY_NAME
        policy_root.mkdir(parents=True, exist_ok=True)
        analysis_policy.mkdir(parents=True, exist_ok=True)
        prediction_path = policy_root / f"{split_name}_predictions.pt"
        sampled_path = policy_root / f"{split_name}_sampled_price_paths.pt"
        metrics_path = policy_root / f"{split_name}_metric_table.csv"
        diagnostics_path = policy_root / f"{split_name}_diagnostics.json"
        tokens_path = policy_root / f"{split_name}_tokens.pt"
        _atomic_torch_save(
            {"epoch": int(decoder_epoch), "prediction_result": prediction_result},
            prediction_path,
        )
        if sampled_close is not None:
            _atomic_torch_save(
                {"sampled_price_path_artifacts": sampled_artifacts},
                sampled_path,
            )
        token_artifacts = {
            "sampled_s1_paths": torch.as_tensor(sample_cache["sampled_s1_paths"]),
            "sampled_s1_evaluation": torch.as_tensor(
                sample_cache["sampled_s1_paths"]
            ).index_select(2, indices),
            "raw_model_mean_probability": sampled_artifacts[
                "raw_model_mean_probability"
            ],
            "sampling_policy_mean_probability": sampled_artifacts[
                "sampling_policy_mean_probability"
            ],
            "target_s1": torch.as_tensor(token_dataset.cache["target_s1"]).to(torch.int16),
            "temperature": float(config["sampling"]["temperature"]),
            "top_k": int(config["sampling"]["top_k"]),
            "top_p": float(config["sampling"]["top_p"]),
            "sample_count": int(config["sampling"]["sample_count"]),
            "evaluation_horizons": list(EVALUATION_HORIZONS),
            "asset_cols": list(token_dataset.asset_cols),
            "sample_idx": prediction_result["sample_idx"],
            "origin_idx": prediction_result["origin_idx"],
            "target_indices": prediction_result["target_indices"],
            "dates": list(token_dataset.cache.get("dates", [])),
        }
        _atomic_torch_save(
            {"epoch": int(decoder_epoch), "token_artifacts": token_artifacts},
            tokens_path,
        )
        _atomic_csv_save(metric_table, metrics_path)
        _atomic_json_save(diagnostics, diagnostics_path)

        _copy_forecaster_artifacts(
            source_dir=source_dir,
            run_dir=run_dir,
            split_name=split_name,
            decoder_epoch=decoder_epoch,
        )
        _atomic_torch_save(
            torch.load(run_dir / f"best_{split_name}_graphs.pt", map_location="cpu", weights_only=False),
            policy_root / f"{split_name}_graphs.pt",
        )
        _save_analysis_copy(run_dir, split_name)
        policy_copies = [
            (prediction_path, analysis_policy / "predictions.pt"),
            (metrics_path, analysis_policy / "metric_table.csv"),
            (diagnostics_path, analysis_policy / "diagnostics.json"),
            (tokens_path, analysis_policy / "tokens.pt"),
            (policy_root / f"{split_name}_graphs.pt", analysis_policy / "graphs.pt"),
        ]
        if sampled_path.is_file():
            policy_copies.append(
                (sampled_path, analysis_policy / "sampled_price_paths.pt")
            )
        for source, target in policy_copies:
            shutil.copy2(source, target)

    return {
        "weighted_loss": float(all_loss.item()),
        "metric_table": metric_table,
        "all60_table": all60_table,
    }


def _metadata(
    *,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    decoder: TrainableKronosCoarseDecoder,
    best_epoch: int,
    best_score: float,
    epochs_completed: int,
    asset_cols: Sequence[str],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    graph_values = source_config.get("model", {}).get("graph", {})
    if not graph_values:
        graph_values = source_config.get("models", {}).get("dynamic_graph", {}).get("graph", {})
    return {
        "status": "completed",
        "model_family": "kronos_decoder_post_training_token",
        "run_name": config.get("run_name"),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "epochs_completed": int(epochs_completed),
        "selection_split": "validation",
        "selection_metric": config["training"]["selection_metric"],
        "source_forecaster_run_name": source_metadata.get("run_name"),
        "source_forecaster_run_signature": source_metadata.get("run_signature"),
        "source_forecaster_best_epoch": int(source_metadata["best_epoch"]),
        "source_forecaster_frozen": True,
        "tokenizer_encoder_frozen": True,
        "quantizer_frozen": True,
        "decoder_trainable_parameters": decoder.trainable_parameter_count(),
        "trainable_parameters": decoder.trainable_parameter_count(),
        "asset_cols": list(asset_cols),
        "graph_type": source_metadata.get("graph_type", graph_values.get("type")),
        "graph_heads": source_metadata.get("graph_heads", graph_values.get("num_heads", 1)),
        "graph_heads_per_layer": source_metadata.get(
            "graph_heads_per_layer",
            graph_values.get("num_heads_per_block") or graph_values.get("num_heads_per_layer"),
        ),
        "test_set_contaminated": True,
        "do_not_report": True,
        "project_git_commit": commit,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _resolved_config(
    *,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
) -> dict[str, Any]:
    values = deepcopy(dict(source_config))
    values["model_family"] = "kronos_decoder_post_training_token"
    values["decoder_post_training"] = deepcopy(dict(config))
    values["do_not_report"] = True
    values["test_set_contaminated"] = True
    training = values.setdefault("training", {})
    if isinstance(training, dict):
        training["early_stopping_metric"] = config["training"]["selection_metric"]
        training["selection_metric"] = config["training"]["selection_metric"]
        training["selection_split"] = "validation"
    return values


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_json(args.config)
    observed_signature = str(config.get("config_signature", ""))
    signature_payload = dict(config)
    signature_payload.pop("config_signature", None)
    expected_signature = hashlib.sha256(
        json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed_signature != expected_signature:
        raise ValueError(
            "Decoder post-training config signature is invalid: "
            f"expected {expected_signature}, observed {observed_signature}."
        )
    config["run_name"] = args.run_name
    device = _resolve_device(args.device)
    training = config["training"]
    _set_seed(int(training["seed"]))
    run_dir = _prepare_run_dir(
        args.output_dir,
        args.run_name,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    _atomic_json_save(config, run_dir / "decoder_post_training_config.json")

    source_dir = Path(config["source_forecaster"]["path"]).expanduser().resolve()
    source_config = _load_json(source_dir / "resolved_config.json")
    source_metadata = _load_json(source_dir / "run_metadata.json")
    token_datasets = _token_datasets(args)
    train_split, validation_split, test_split = _load_raw_splits(args.data_dir)
    raw_splits = {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }

    sample_cache_root = run_dir / "sample_cache"
    sample_caches: dict[str, dict[str, Any]] = {}
    decoder_datasets: dict[str, DecoderPostTrainingDataset] = {}
    for split_name in ("train", "validation", "test"):
        cache_path = sample_cache_root / f"{split_name}_sampled_s1.pt"
        sample_cache = _generate_sample_cache(
            source_dir=source_dir,
            source_config=source_config,
            source_metadata=source_metadata,
            dataset=token_datasets[split_name],
            split_name=split_name,
            train_split=train_split,
            sampling=config["sampling"],
            batch_size=int(training["forecaster_batch_size"]),
            num_workers=int(training["num_workers"]),
            mixed_precision=bool(training["mixed_precision"]),
            device=device,
            cache_path=cache_path,
            overwrite=args.overwrite_sample_cache,
        )
        sample_caches[split_name] = sample_cache
        decoder_datasets[split_name] = DecoderPostTrainingDataset(
            token_dataset=token_datasets[split_name],
            sampled_cache=sample_cache,
            raw_split=raw_splits[split_name],
        )

    decoder = TrainableKronosCoarseDecoder.from_forecasting_config(
        load_yaml(args.forecasting_config)
    ).to(device)
    decoder.eval()
    initial_state_path = run_dir / "initial_pretrained_decoder_state.pt"
    if not initial_state_path.is_file():
        _atomic_torch_save(decoder_state_dict_cpu(decoder), initial_state_path)

    loss_function = WeightedAll60Loss(config["loss"]["weights"])
    weight_table = pd.DataFrame(
        {
            "horizon": list(range(1, 61)),
            "normalised_weight": list(config["loss"]["weights"]),
        }
    )
    anchors = config["loss"]["loss_ratio_anchors_vs_h1"]
    weight_table["anchor_loss_ratio_vs_h1"] = weight_table["horizon"].astype(str).map(anchors)
    weight_table["anchor_inverse_ratio"] = 1.0 / weight_table["anchor_loss_ratio_vs_h1"]
    _atomic_csv_save(weight_table, run_dir / "decoder_loss_weights.csv")

    train_loader = _loader(
        decoder_datasets["train"],
        batch_size=int(training["train_batch_size"]),
        shuffle=True,
        seed=int(training["seed"]),
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    validation_loader = _loader(
        decoder_datasets["validation"],
        batch_size=int(training["evaluation_batch_size"]),
        shuffle=False,
        seed=int(training["seed"]),
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
    )
    scaler = _new_grad_scaler(
        bool(training["mixed_precision"]) and device.type == "cuda"
    )
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1
    last_path = run_dir / "last_checkpoint.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config_signature") != config["config_signature"]:
            raise ValueError("Resume checkpoint config signature differs.")
        decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        _restore_rng_state(checkpoint["rng_state"])
        history_path = run_dir / "history.csv"
        if history_path.is_file():
            history = pd.read_csv(history_path).to_dict("records")

    if bad_epochs >= int(training["patience"]):
        start_epoch = int(training["max_epochs"]) + 1
        print(
            "The resumed decoder checkpoint had already exhausted patience; "
            "skipping additional optimisation."
        )

    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        started = perf_counter()
        train_values = _train_epoch(
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_function=loss_function,
            sample_chunk_size=int(training["sample_chunk_size"]),
            device=device,
            mixed_precision=bool(training["mixed_precision"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            epoch=epoch,
        )
        validation_values = _evaluate_weighted_loss(
            decoder=decoder,
            loader=validation_loader,
            loss_function=loss_function,
            sample_chunk_size=int(training["sample_chunk_size"]),
            device=device,
            mixed_precision=bool(training["mixed_precision"]),
            description=f"decoder validation epoch {epoch}",
        )
        score = float(validation_values["weighted_loss"])
        improved = score < best_score - float(training["min_delta"])
        if improved:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(
                {
                    "epoch": int(epoch),
                    "decoder_state_dict": decoder.state_dict(),
                    "selection_score": score,
                    "config_signature": config["config_signature"],
                },
                run_dir / "best_checkpoint.pt",
            )
        else:
            bad_epochs += 1
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "train_weighted_all_60_clg_mae": float(train_values["loss"]),
            "validation_weighted_all_60_clg_mae": score,
            "decoder_gradient_norm": float(train_values["gradient_norm"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_score": float(best_score),
            "bad_epochs": int(bad_epochs),
            "epoch_seconds": perf_counter() - started,
            "validation_invalid_sample_candle_rate_percent": float(
                validation_values["invalid_sample_candle_rate_percent"]
            ),
        }
        for index, value in enumerate(
            torch.as_tensor(validation_values["clg_mae_by_horizon"]), start=1
        ):
            record[f"validation_clg_mae_h{index}"] = float(value.item())
        history.append(record)
        _atomic_csv_save(pd.DataFrame(history), run_dir / "history.csv")
        _atomic_torch_save(
            _checkpoint_payload(
                epoch=epoch,
                decoder=decoder,
                optimizer=optimizer,
                best_score=best_score,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                config_signature=config["config_signature"],
            ),
            last_path,
        )
        print(json.dumps({"epoch": epoch, "score": score, "best": best_score, "bad_epochs": bad_epochs}))
        if bad_epochs >= int(training["patience"]):
            break

    best_checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    decoder.load_state_dict(best_checkpoint["decoder_state_dict"], strict=True)
    decoder.to(device)
    best_epoch = int(best_checkpoint["epoch"])
    best_score = float(best_checkpoint["selection_score"])

    # Save a Graph-Hub-compatible resolved configuration before exports.
    resolved = _resolved_config(config=config, source_config=source_config)
    _atomic_json_save(resolved, run_dir / "resolved_config.json")

    # Baseline and post-trained evaluations use the exact same fixed ten paths.
    post_results: dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        post_results[split_name] = _export_split(
            decoder=decoder,
            dataset=decoder_datasets[split_name],
            token_dataset=token_datasets[split_name],
            sample_cache=sample_caches[split_name],
            split_name=split_name,
            source_dir=source_dir,
            run_dir=run_dir,
            train_split=train_split,
            decoder_epoch=best_epoch,
            config=config,
            device=device,
            baseline=False,
        )

    initial_state = torch.load(initial_state_path, map_location="cpu", weights_only=False)
    decoder.load_state_dict(initial_state, strict=True)
    decoder.to(device)
    baseline_results: dict[str, Any] = {}
    for split_name in ("validation", "test"):
        baseline_results[split_name] = _export_split(
            decoder=decoder,
            dataset=decoder_datasets[split_name],
            token_dataset=token_datasets[split_name],
            sample_cache=sample_caches[split_name],
            split_name=split_name,
            source_dir=source_dir,
            run_dir=run_dir,
            train_split=train_split,
            decoder_epoch=0,
            config=config,
            device=device,
            baseline=True,
        )

    metadata = _metadata(
        config=config,
        source_config=source_config,
        source_metadata=source_metadata,
        decoder=decoder,
        best_epoch=best_epoch,
        best_score=best_score,
        epochs_completed=int(history[-1]["epoch"]) if history else best_epoch,
        asset_cols=token_datasets["train"].asset_cols,
    )
    metadata["run_name"] = args.run_name
    metadata["validation_baseline_weighted_loss"] = float(
        baseline_results["validation"]["weighted_loss"]
    )
    metadata["validation_posttrained_weighted_loss"] = float(
        post_results["validation"]["weighted_loss"]
    )
    metadata["test_baseline_weighted_loss"] = float(
        baseline_results["test"]["weighted_loss"]
    )
    metadata["test_posttrained_weighted_loss"] = float(
        post_results["test"]["weighted_loss"]
    )
    _atomic_json_save(metadata, run_dir / "run_metadata.json")
    _atomic_json_save(
        {
            "selected_policy": POLICY_NAME,
            "selected_temperature": float(config["sampling"]["temperature"]),
            "sample_count": int(config["sampling"]["sample_count"]),
            "top_k": int(config["sampling"]["top_k"]),
            "top_p": float(config["sampling"]["top_p"]),
            "sampling_seed": int(config["sampling"]["seed"]),
            "decoded_splits": ["train", "validation", "test"],
        },
        run_dir / "temperature_sweep" / "temperature_selection.json",
    )
    comparison = pd.DataFrame(
        [
            {
                "split": split,
                "baseline_weighted_all_60_clg_mae": baseline_results[split]["weighted_loss"],
                "posttrained_weighted_all_60_clg_mae": post_results[split]["weighted_loss"],
                "improvement": baseline_results[split]["weighted_loss"] - post_results[split]["weighted_loss"],
            }
            for split in ("validation", "test")
        ]
    )
    _atomic_csv_save(comparison, run_dir / "decoder_post_training_comparison.csv")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.models.kronos import KronosBaseline
from src.utils.config import load_yaml


ConfigDict = dict[str, Any]
SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]

PROGRESS_SCHEMA_VERSION = 1
STATIC_PREDICTION_KEYS = (
    "channels",
    "horizons",
    "asset_cols",
    "output_space",
)
TENSOR_PREDICTION_KEYS = (
    "y_pred",
    "y_true",
    "sample_idx",
    "origin_idx",
    "target_indices",
    "last_context_target",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen zero-shot Kronos inference and cache the raw "
            "project prediction dictionary."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting.yaml"),
        help="Path to the forecasting YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.pt, val.pt, and test.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which prediction and metadata files are saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional output stem. When omitted, a timestamped name "
            "is generated."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
        help="Split on which predictions are generated.",
    )
    parser.add_argument(
        "--confirm-test-run",
        action="store_true",
        help=(
            "Required when --evaluation-split test is used. This guards "
            "against accidental test-set inspection."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override models.kronos.inference.device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default=None,
        help=(
            "Override models.kronos.inference.dtype. float16 uses "
            "CUDA automatic mixed precision and therefore requires "
            "a CUDA device."
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Override the number of stochastic trajectories.",
    )
    parser.add_argument(
        "--series-batch-size",
        type=int,
        default=None,
        help=(
            "Override the number of flattened asset-window series passed "
            "to each official predict_batch() call."
        ),
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Number of project forecast windows prepared together.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "Optional number of forecast windows to process. Use only "
            "for smoke tests and runtime benchmarks."
        ),
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help=(
            "Optional number of leading assets to retain. Use only for "
            "smoke tests and runtime benchmarks."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional inference-temperature override.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k override.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p override.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Atomically save cumulative progress after this many "
            "forecast windows. Set to 0 to disable checkpointing."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a compatible cumulative progress checkpoint when "
            "one exists; otherwise begin a fresh checkpointed run."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing prediction, metadata, and progress files "
            "before beginning a fresh run."
        ),
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if (
        args.evaluation_split == "test"
        and not args.confirm_test_run
    ):
        raise ValueError(
            "A test run requires --confirm-test-run. Do not use test "
            "results while selecting inference settings."
        )

    if args.window_batch_size <= 0:
        raise ValueError(
            "--window-batch-size must be greater than zero."
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers must be non-negative."
        )

    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError(
            "--max-examples must be greater than zero."
        )

    if args.checkpoint_every < 0:
        raise ValueError(
            "--checkpoint-every must be non-negative."
        )

    if args.resume and args.checkpoint_every <= 0:
        raise ValueError(
            "--resume requires --checkpoint-every to be greater "
            "than zero."
        )

    if args.resume and args.overwrite:
        raise ValueError(
            "--resume and --overwrite are mutually exclusive."
        )

    if (
        args.checkpoint_every > 0
        and args.checkpoint_every % args.window_batch_size != 0
    ):
        raise ValueError(
            "--checkpoint-every must be an exact multiple of "
            "--window-batch-size so checkpoint boundaries do not "
            "split a project prediction batch."
        )


def validate_paths(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    config_path = config_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    missing_files = [
        filename
        for filename in ("train.pt", "val.pt", "test.pt")
        if not (data_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing cached split files: {missing_files}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return config_path, data_dir, output_dir


def apply_cli_overrides(
    config: ConfigDict,
    args: argparse.Namespace,
) -> ConfigDict:
    resolved = deepcopy(config)

    kronos_config = resolved.setdefault(
        "models",
        {},
    ).setdefault(
        "kronos",
        {},
    )
    inference = kronos_config.setdefault(
        "inference",
        {},
    )

    overrides = {
        "device": args.device,
        "dtype": args.dtype,
        "sample_count": args.sample_count,
        "series_batch_size": args.series_batch_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    for key, value in overrides.items():
        if value is not None:
            inference[key] = value

    return resolved


def retain_first_assets(
    split: SplitDict,
    max_assets: int | None,
) -> SplitDict:
    if max_assets is None:
        return split

    if max_assets <= 0:
        raise ValueError(
            "--max-assets must be greater than zero."
        )

    total_assets = len(split["asset_cols"])
    if max_assets > total_assets:
        raise ValueError(
            f"--max-assets={max_assets} exceeds the available "
            f"{total_assets} assets."
        )

    reduced = dict(split)
    reduced["asset_cols"] = list(
        split["asset_cols"][:max_assets]
    )
    reduced["samples"] = [
        (
            x_day[:, :max_assets, :].clone(),
            aux,
            day,
        )
        for x_day, aux, day in split["samples"]
    ]

    return reduced


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def first_parameter_dtype(module: Any) -> str | None:
    if module is None:
        return None

    try:
        parameter = next(module.parameters())
    except StopIteration:
        return None

    return str(parameter.dtype).removeprefix("torch.")


def cuda_memory_snapshot(
    device: torch.device,
) -> dict[str, float | str]:
    properties = torch.cuda.get_device_properties(device)

    return {
        "gpu_name": properties.name,
        "gpu_total_memory_gib": (
            properties.total_memory / (1024 ** 3)
        ),
        "cuda_memory_allocated_gib": (
            torch.cuda.memory_allocated(device) / (1024 ** 3)
        ),
        "cuda_memory_reserved_gib": (
            torch.cuda.memory_reserved(device) / (1024 ** 3)
        ),
        "cuda_peak_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        ),
        "cuda_peak_memory_reserved_gib": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        ),
    }


def merge_cuda_memory_snapshots(
    previous: dict[str, float | str] | None,
    current: dict[str, float | str] | None,
) -> dict[str, float | str] | None:
    if current is None:
        return previous

    if previous is None:
        return dict(current)

    merged = dict(current)

    for key in (
        "cuda_peak_memory_allocated_gib",
        "cuda_peak_memory_reserved_gib",
    ):
        merged[key] = max(
            float(previous[key]),
            float(current[key]),
        )

    return merged


def build_output_paths(
    output_dir: Path,
    run_name: str | None,
    evaluation_split: str,
) -> tuple[Path, Path, Path]:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"kronos_{evaluation_split}_{timestamp}"

    run_name = run_name.strip()
    if not run_name:
        raise ValueError("--run-name must not be empty.")

    prediction_path = output_dir / f"{run_name}_predictions.pt"
    metadata_path = output_dir / f"{run_name}_metadata.json"
    progress_path = output_dir / f"{run_name}_progress.pt"

    return prediction_path, metadata_path, progress_path


def temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def atomic_torch_save(
    value: Any,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path(path)

    if temp_path.exists():
        temp_path.unlink()

    try:
        with temp_path.open("wb") as file:
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json_save(
    value: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_path(path)

    if temp_path.exists():
        temp_path.unlink()

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                value,
                file,
                indent=2,
                sort_keys=True,
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def prepare_output_paths(
    *,
    prediction_path: Path,
    metadata_path: Path,
    progress_path: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    paths = (
        prediction_path,
        metadata_path,
        progress_path,
    )

    for path in paths:
        temp_path = temporary_path(path)
        if temp_path.exists():
            temp_path.unlink()

    if overwrite:
        for path in paths:
            if path.exists():
                path.unlink()
        return

    if resume:
        if not progress_path.exists() and (
            prediction_path.exists() or metadata_path.exists()
        ):
            raise FileExistsError(
                "Final output files exist but no resumable progress "
                "checkpoint is available. Use a new --run-name or "
                "explicitly use --overwrite for a fresh run."
            )
        return

    existing = [
        path
        for path in paths
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Output file(s) already exist. Use --resume for a "
            "compatible progress checkpoint or --overwrite for a "
            f"fresh run: {existing}"
        )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
    missing = required - set(state)
    if missing:
        raise KeyError(
            f"Saved RNG state is missing keys: {sorted(missing)}"
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    saved_cuda_state = state["torch_cuda"]
    if saved_cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The progress checkpoint contains CUDA RNG state, "
                "but CUDA is unavailable in the resumed runtime."
            )

        if len(saved_cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the saved progress "
                "checkpoint. Expected "
                f"{len(saved_cuda_state)}, observed "
                f"{torch.cuda.device_count()}."
            )

        torch.cuda.set_rng_state_all(
            [saved.cpu() for saved in saved_cuda_state]
        )


def expected_window_index(
    *,
    split: SplitDict,
    context_length: int,
    horizons: list[int],
    stride: int,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_indices: list[int] = []
    origin_indices: list[int] = []
    target_indices: list[list[int]] = []

    max_horizon = max(horizons)

    for sample_idx, (x_day, _, _) in enumerate(split["samples"]):
        first_origin = context_length - 1
        last_origin = int(x_day.shape[0]) - max_horizon - 1

        if last_origin < first_origin:
            continue

        for origin_idx in range(
            first_origin,
            last_origin + 1,
            stride,
        ):
            sample_indices.append(sample_idx)
            origin_indices.append(origin_idx)
            target_indices.append(
                [origin_idx + horizon for horizon in horizons]
            )

            if len(sample_indices) == limit:
                return (
                    torch.tensor(sample_indices, dtype=torch.long),
                    torch.tensor(origin_indices, dtype=torch.long),
                    torch.tensor(target_indices, dtype=torch.long),
                )

    if len(sample_indices) != limit:
        raise RuntimeError(
            "Could not reconstruct the requested number of expected "
            f"windows. Requested {limit}, reconstructed "
            f"{len(sample_indices)}."
        )

    return (
        torch.tensor(sample_indices, dtype=torch.long),
        torch.tensor(origin_indices, dtype=torch.long),
        torch.tensor(target_indices, dtype=torch.long),
    )


def prediction_summary(
    prediction_result: PredictionDict,
) -> dict[str, Any]:
    y_pred = prediction_result["y_pred"]
    y_true = prediction_result["y_true"]

    return {
        "num_windows": int(y_pred.shape[0]),
        "num_horizons": int(y_pred.shape[1]),
        "num_assets": int(y_pred.shape[2]),
        "num_channels": int(y_pred.shape[3]),
        "y_pred_shape": list(y_pred.shape),
        "y_true_shape": list(y_true.shape),
        "y_pred_finite": bool(torch.isfinite(y_pred).all()),
        "y_true_finite": bool(torch.isfinite(y_true).all()),
        "minimum_predicted_close": float(y_pred.min()),
        "maximum_predicted_close": float(y_pred.max()),
        "non_positive_predicted_close_count": int(
            (y_pred <= 0).sum()
        ),
    }


def validate_prediction_result(
    *,
    prediction_result: PredictionDict,
    expected_num_windows: int,
    model: KronosBaseline,
    split: SplitDict,
) -> None:
    required = set(STATIC_PREDICTION_KEYS) | set(
        TENSOR_PREDICTION_KEYS
    )
    missing = required - set(prediction_result)
    if missing:
        raise KeyError(
            "Prediction result is missing keys: "
            f"{sorted(missing)}"
        )

    expected_shape = (
        expected_num_windows,
        len(model.horizons),
        len(split["asset_cols"]),
        len(model.target_channels),
    )

    if tuple(prediction_result["y_pred"].shape) != expected_shape:
        raise ValueError(
            "Unexpected y_pred shape. Expected "
            f"{expected_shape}, observed "
            f"{tuple(prediction_result['y_pred'].shape)}."
        )

    if tuple(prediction_result["y_true"].shape) != expected_shape:
        raise ValueError(
            "Unexpected y_true shape. Expected "
            f"{expected_shape}, observed "
            f"{tuple(prediction_result['y_true'].shape)}."
        )

    expected_last_context_shape = (
        expected_num_windows,
        len(split["asset_cols"]),
        len(model.target_channels),
    )
    if tuple(
        prediction_result["last_context_target"].shape
    ) != expected_last_context_shape:
        raise ValueError(
            "Unexpected last_context_target shape. Expected "
            f"{expected_last_context_shape}, observed "
            f"{tuple(prediction_result['last_context_target'].shape)}."
        )

    if tuple(prediction_result["sample_idx"].shape) != (
        expected_num_windows,
    ):
        raise ValueError("Unexpected sample_idx shape.")

    if tuple(prediction_result["origin_idx"].shape) != (
        expected_num_windows,
    ):
        raise ValueError("Unexpected origin_idx shape.")

    if tuple(prediction_result["target_indices"].shape) != (
        expected_num_windows,
        len(model.horizons),
    ):
        raise ValueError("Unexpected target_indices shape.")

    if list(prediction_result["channels"]) != list(
        model.target_channels
    ):
        raise ValueError("Prediction channels do not match the model.")

    if list(prediction_result["horizons"]) != list(model.horizons):
        raise ValueError("Prediction horizons do not match the model.")

    if list(prediction_result["asset_cols"]) != list(
        split["asset_cols"]
    ):
        raise ValueError(
            "Prediction asset order does not match the split."
        )

    if prediction_result["output_space"] != "raw":
        raise ValueError("Kronos predictions must be in raw space.")

    if not torch.isfinite(prediction_result["y_pred"]).all():
        raise ValueError("Predictions contain non-finite values.")

    if not torch.isfinite(prediction_result["y_true"]).all():
        raise ValueError("Targets contain non-finite values.")

    if (prediction_result["y_pred"] <= 0).any():
        raise ValueError(
            "Predictions contain non-positive close values."
        )

    (
        expected_sample_idx,
        expected_origin_idx,
        expected_target_indices,
    ) = expected_window_index(
        split=split,
        context_length=model.context_length,
        horizons=model.horizons,
        stride=model.stride,
        limit=expected_num_windows,
    )

    if not torch.equal(
        prediction_result["sample_idx"].cpu().long(),
        expected_sample_idx,
    ):
        raise ValueError(
            "Saved sample_idx does not match the chronological "
            "dataset prefix."
        )

    if not torch.equal(
        prediction_result["origin_idx"].cpu().long(),
        expected_origin_idx,
    ):
        raise ValueError(
            "Saved origin_idx does not match the chronological "
            "dataset prefix."
        )

    if not torch.equal(
        prediction_result["target_indices"].cpu().long(),
        expected_target_indices,
    ):
        raise ValueError(
            "Saved target_indices do not match origin + horizon."
        )


def split_signature(split: SplitDict) -> dict[str, Any]:
    return {
        "asset_cols": [str(value) for value in split["asset_cols"]],
        "channels": [str(value) for value in split["channels"]],
        "session_days": [
            str(day)
            for _, _, day in split["samples"]
        ],
        "session_shapes": [
            [int(dimension) for dimension in x_day.shape]
            for x_day, _, _ in split["samples"]
        ],
        "market_open": str(split.get("market_open")),
        "market_close": str(split.get("market_close")),
    }


def build_run_signature(
    *,
    args: argparse.Namespace,
    model: KronosBaseline,
    prediction_split: SplitDict,
    total_split_windows: int,
    target_num_windows: int,
    project_revision: str | None,
    kronos_revision: str | None,
) -> dict[str, Any]:
    gpu_name: str | None = None
    if model.device is not None and model.device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(torch.device(model.device))

    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "evaluation_split": args.evaluation_split,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "tokenizer_id": model.tokenizer_id,
        "tokenizer_revision": model.tokenizer_revision,
        "project_git_revision": project_revision,
        "kronos_git_revision": kronos_revision,
        "device": model.device,
        "gpu_name": gpu_name,
        "dtype": model.dtype,
        "context_length": model.context_length,
        "horizons": list(model.horizons),
        "stride": model.stride,
        "input_channels": list(model.input_channels),
        "target_channels": list(model.target_channels),
        "temperature": model.temperature,
        "top_k": model.top_k,
        "top_p": model.top_p,
        "sample_count": model.sample_count,
        "seed": model.seed,
        "max_context": model.max_context,
        "clip": model.clip,
        "series_batch_size": model.series_batch_size,
        "window_batch_size": args.window_batch_size,
        "num_workers": args.num_workers,
        "max_examples": args.max_examples,
        "max_assets": args.max_assets,
        "checkpoint_every": args.checkpoint_every,
        "total_split_windows": total_split_windows,
        "target_num_windows": target_num_windows,
        "split": split_signature(prediction_split),
        "runtime": {
            "python": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "torch": package_version("torch"),
            "cuda_runtime": torch.version.cuda,
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version(
                "huggingface-hub"
            ),
            "safetensors": package_version("safetensors"),
            "einops": package_version("einops"),
        },
    }


def signature_differences(
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    differences: list[str] = []

    all_keys = sorted(set(expected) | set(observed))
    for key in all_keys:
        if key not in expected:
            differences.append(f"unexpected key {key!r}")
            continue

        if key not in observed:
            differences.append(f"missing key {key!r}")
            continue

        expected_value = expected[key]
        observed_value = observed[key]

        if isinstance(expected_value, dict) and isinstance(
            observed_value,
            dict,
        ):
            nested = signature_differences(
                expected_value,
                observed_value,
            )
            differences.extend(
                [f"{key}.{difference}" for difference in nested]
            )
        elif expected_value != observed_value:
            differences.append(
                f"{key}: expected {expected_value!r}, "
                f"observed {observed_value!r}"
            )

    return differences


def load_progress_checkpoint(
    *,
    progress_path: Path,
    expected_signature: dict[str, Any],
    model: KronosBaseline,
    prediction_split: SplitDict,
) -> dict[str, Any]:
    progress = torch.load(
        progress_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(progress, dict):
        raise TypeError("Progress checkpoint must contain a dictionary.")

    if progress.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported progress-checkpoint schema. Expected "
            f"{PROGRESS_SCHEMA_VERSION}, observed "
            f"{progress.get('schema_version')!r}."
        )

    observed_signature = progress.get("run_signature")
    if not isinstance(observed_signature, dict):
        raise TypeError(
            "Progress checkpoint does not contain a run signature."
        )

    differences = signature_differences(
        expected_signature,
        observed_signature,
    )
    if differences:
        formatted = "\n".join(
            f"  - {difference}"
            for difference in differences[:20]
        )
        raise ValueError(
            "The existing progress checkpoint is incompatible with "
            "the requested run:\n"
            f"{formatted}"
        )

    prediction_result = progress.get("prediction_result")
    if not isinstance(prediction_result, dict):
        raise TypeError(
            "Progress checkpoint does not contain prediction_result."
        )

    next_window_index = int(progress.get("next_window_index", -1))
    if next_window_index < 0:
        raise ValueError(
            "Progress checkpoint contains an invalid next_window_index."
        )

    if int(prediction_result["y_pred"].shape[0]) != next_window_index:
        raise ValueError(
            "Progress next_window_index does not equal the number of "
            "saved predictions."
        )

    validate_prediction_result(
        prediction_result=prediction_result,
        expected_num_windows=next_window_index,
        model=model,
        split=prediction_split,
    )

    if "rng_state" not in progress:
        raise KeyError(
            "Progress checkpoint does not contain RNG state."
        )

    return progress


def save_progress_checkpoint(
    *,
    progress_path: Path,
    status: str,
    run_signature: dict[str, Any],
    prediction_result: PredictionDict,
    next_window_index: int,
    run_started_at: str,
    cumulative_prediction_seconds: float,
    checkpoint_count: int,
    resume_count: int,
    cuda_memory_peak: dict[str, float | str] | None,
    final_prediction_path: Path | None = None,
    final_metadata_path: Path | None = None,
) -> None:
    progress = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": status,
        "run_signature": run_signature,
        "prediction_result": prediction_result,
        "next_window_index": next_window_index,
        "rng_state": capture_rng_state(),
        "run_started_at": run_started_at,
        "updated_at": datetime.now().isoformat(),
        "cumulative_prediction_seconds": (
            cumulative_prediction_seconds
        ),
        "checkpoint_count": checkpoint_count,
        "resume_count": resume_count,
        "cuda_memory_peak": cuda_memory_peak,
        "final_prediction_path": (
            str(final_prediction_path)
            if final_prediction_path is not None
            else None
        ),
        "final_metadata_path": (
            str(final_metadata_path)
            if final_metadata_path is not None
            else None
        ),
    }

    atomic_torch_save(
        progress,
        progress_path,
    )


def progress_line(
    *,
    completed: int,
    total: int,
    prediction_seconds: float,
) -> str:
    seconds_per_window = (
        prediction_seconds / completed
        if completed > 0
        else float("nan")
    )
    remaining_seconds = max(total - completed, 0) * seconds_per_window

    return (
        f"completed {completed}/{total} windows | "
        f"inference {prediction_seconds / 3600:.2f} h | "
        f"ETA {remaining_seconds / 3600:.2f} h"
    )


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_arguments(args)

    config_path, data_dir, output_dir = validate_paths(
        args.config,
        args.data_dir,
        args.output_dir,
    )

    (
        prediction_path,
        metadata_path,
        progress_path,
    ) = build_output_paths(
        output_dir=output_dir,
        run_name=args.run_name,
        evaluation_split=args.evaluation_split,
    )

    prepare_output_paths(
        prediction_path=prediction_path,
        metadata_path=metadata_path,
        progress_path=progress_path,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    config = load_yaml(config_path)
    resolved_config = apply_cli_overrides(
        config,
        args,
    )

    print("Loading and cleaning chronological splits...")
    train_raw, val_raw, test_raw = load_candle_splits(data_dir)
    train, val, test = clean_candle_splits(
        train_raw,
        val_raw,
        test_raw,
    )

    train = retain_first_assets(train, args.max_assets)
    val = retain_first_assets(val, args.max_assets)
    test = retain_first_assets(test, args.max_assets)

    prediction_split = (
        val
        if args.evaluation_split == "val"
        else test
    )

    model = KronosBaseline.from_config(resolved_config)

    print("Preparing frozen Kronos checkpoints...")
    fit_start = perf_counter()
    model.fit(
        train_split=train,
        val_split=val,
    )
    fit_seconds = perf_counter() - fit_start

    total_split_windows = model.prediction_window_count(
        prediction_split
    )
    target_num_windows = (
        total_split_windows
        if args.max_examples is None
        else min(total_split_windows, args.max_examples)
    )

    project_revision = git_revision(Path.cwd())
    kronos_revision = git_revision(
        Path.cwd() / "external" / "Kronos"
    )

    run_signature = build_run_signature(
        args=args,
        model=model,
        prediction_split=prediction_split,
        total_split_windows=total_split_windows,
        target_num_windows=target_num_windows,
        project_revision=project_revision,
        kronos_revision=kronos_revision,
    )

    print(
        "Generating predictions:",
        {
            "split": args.evaluation_split,
            "device": model.device,
            "dtype": model.dtype,
            "sample_count": model.sample_count,
            "series_batch_size": model.series_batch_size,
            "window_batch_size": args.window_batch_size,
            "target_windows": target_num_windows,
            "checkpoint_every": args.checkpoint_every,
            "resume": args.resume,
            "max_assets": args.max_assets,
        },
    )

    cuda_device: torch.device | None = None
    cuda_memory_before_prediction: dict[str, float | str] | None = None
    cuda_memory_peak: dict[str, float | str] | None = None

    if model.device is not None and model.device.startswith("cuda"):
        cuda_device = torch.device(model.device)
        torch.cuda.synchronize(cuda_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        cuda_memory_before_prediction = cuda_memory_snapshot(cuda_device)

    run_started_at = datetime.now().isoformat()
    cumulative_prediction_seconds = 0.0
    checkpoint_count = 0
    resume_count = 0
    start_index = 0
    prediction_batches: list[PredictionDict] = []
    seed_rng = True

    if args.resume and progress_path.exists():
        progress = load_progress_checkpoint(
            progress_path=progress_path,
            expected_signature=run_signature,
            model=model,
            prediction_split=prediction_split,
        )

        start_index = int(progress["next_window_index"])
        if start_index > target_num_windows:
            raise ValueError(
                "Progress checkpoint contains more windows than the "
                "requested run."
            )

        prediction_batches = [progress["prediction_result"]]
        cumulative_prediction_seconds = float(
            progress.get("cumulative_prediction_seconds", 0.0)
        )
        checkpoint_count = int(progress.get("checkpoint_count", 0))
        resume_count = int(progress.get("resume_count", 0)) + 1
        run_started_at = str(progress.get("run_started_at"))
        cuda_memory_peak = progress.get("cuda_memory_peak")

        restore_rng_state(progress["rng_state"])
        seed_rng = False

        print(
            "Resuming compatible checkpoint:",
            progress_path,
        )
        print(
            progress_line(
                completed=start_index,
                total=target_num_windows,
                prediction_seconds=cumulative_prediction_seconds,
            )
        )
    elif args.resume:
        print(
            "No progress checkpoint exists; starting a fresh "
            "checkpointed run."
        )

    remaining_windows = target_num_windows - start_index

    if remaining_windows > 0:
        prediction_iterator = model.iter_prediction_batches(
            split=prediction_split,
            batch_size=args.window_batch_size,
            num_workers=args.num_workers,
            start_index=start_index,
            max_examples=remaining_windows,
            seed_rng=seed_rng,
        )

        completed = start_index
        next_checkpoint = (
            (
                completed // args.checkpoint_every
                + 1
            )
            * args.checkpoint_every
            if args.checkpoint_every > 0
            else None
        )

        while True:
            batch_start = perf_counter()

            try:
                prediction_batch = next(prediction_iterator)
            except StopIteration:
                break

            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)

            cumulative_prediction_seconds += (
                perf_counter() - batch_start
            )

            prediction_batches.append(prediction_batch)
            completed += int(prediction_batch["y_pred"].shape[0])

            if completed > target_num_windows:
                raise RuntimeError(
                    "Prediction iterator produced more windows than "
                    "requested."
                )

            should_checkpoint = (
                args.checkpoint_every > 0
                and (
                    completed >= int(next_checkpoint)
                    or completed == target_num_windows
                )
            )

            if should_checkpoint:
                cumulative_result = (
                    model.concatenate_prediction_batches(
                        prediction_batches
                    )
                )
                validate_prediction_result(
                    prediction_result=cumulative_result,
                    expected_num_windows=completed,
                    model=model,
                    split=prediction_split,
                )
                prediction_batches = [cumulative_result]

                current_memory = (
                    cuda_memory_snapshot(cuda_device)
                    if cuda_device is not None
                    else None
                )
                cuda_memory_peak = merge_cuda_memory_snapshots(
                    cuda_memory_peak,
                    current_memory,
                )

                checkpoint_count += 1
                save_progress_checkpoint(
                    progress_path=progress_path,
                    status="in_progress",
                    run_signature=run_signature,
                    prediction_result=cumulative_result,
                    next_window_index=completed,
                    run_started_at=run_started_at,
                    cumulative_prediction_seconds=(
                        cumulative_prediction_seconds
                    ),
                    checkpoint_count=checkpoint_count,
                    resume_count=resume_count,
                    cuda_memory_peak=cuda_memory_peak,
                )

                print(
                    "Saved progress:",
                    progress_path,
                )
                print(
                    progress_line(
                        completed=completed,
                        total=target_num_windows,
                        prediction_seconds=(
                            cumulative_prediction_seconds
                        ),
                    )
                )

                while (
                    next_checkpoint is not None
                    and next_checkpoint <= completed
                ):
                    next_checkpoint += args.checkpoint_every

    if not prediction_batches:
        raise RuntimeError(
            "Kronos inference produced no prediction windows."
        )

    prediction_result = model.concatenate_prediction_batches(
        prediction_batches
    )
    validate_prediction_result(
        prediction_result=prediction_result,
        expected_num_windows=target_num_windows,
        model=model,
        split=prediction_split,
    )

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        current_memory = cuda_memory_snapshot(cuda_device)
        cuda_memory_peak = merge_cuda_memory_snapshots(
            cuda_memory_peak,
            current_memory,
        )

    summary = prediction_summary(prediction_result)

    run_metadata = {
        "created_at": datetime.now().isoformat(),
        "run_started_at": run_started_at,
        "evaluation_split": args.evaluation_split,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "prediction_path": str(prediction_path),
        "metadata_path": str(metadata_path),
        "progress_path": (
            str(progress_path)
            if args.checkpoint_every > 0
            else None
        ),
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "tokenizer_id": model.tokenizer_id,
        "tokenizer_revision": model.tokenizer_revision,
        "device": model.device,
        "dtype": model.dtype,
        "precision_mode": (
            "cuda_autocast_float16"
            if model.dtype == "float16"
            else "full_float32"
        ),
        "model_parameter_dtype": first_parameter_dtype(model.model),
        "tokenizer_parameter_dtype": first_parameter_dtype(
            model.tokenizer
        ),
        "cuda_memory_before_prediction": (
            cuda_memory_before_prediction
        ),
        "cuda_memory_after_prediction": cuda_memory_peak,
        "context_length": model.context_length,
        "horizons": model.horizons,
        "input_channels": model.input_channels,
        "target_channels": model.target_channels,
        "temperature": model.temperature,
        "top_k": model.top_k,
        "top_p": model.top_p,
        "sample_count": model.sample_count,
        "seed": model.seed,
        "max_context": model.max_context,
        "clip": model.clip,
        "series_batch_size": model.series_batch_size,
        "window_batch_size": args.window_batch_size,
        "num_workers": args.num_workers,
        "max_examples": args.max_examples,
        "max_assets": args.max_assets,
        "checkpoint_every": args.checkpoint_every,
        "resume_enabled": args.resume,
        "resume_count": resume_count,
        "checkpoint_count": checkpoint_count,
        "total_split_windows": total_split_windows,
        "fit_seconds": fit_seconds,
        "prediction_seconds": cumulative_prediction_seconds,
        "prediction_summary": summary,
        "project_git_revision": project_revision,
        "kronos_git_revision": kronos_revision,
        "package_versions": run_signature["runtime"],
        "run_signature": run_signature,
        "resolved_kronos_config": resolved_config[
            "models"
        ]["kronos"],
    }

    cache = {
        "prediction_result": prediction_result,
        "run_metadata": run_metadata,
    }

    atomic_torch_save(cache, prediction_path)
    atomic_json_save(run_metadata, metadata_path)

    if args.checkpoint_every > 0:
        checkpoint_count += 1
        save_progress_checkpoint(
            progress_path=progress_path,
            status="completed",
            run_signature=run_signature,
            prediction_result=prediction_result,
            next_window_index=target_num_windows,
            run_started_at=run_started_at,
            cumulative_prediction_seconds=(
                cumulative_prediction_seconds
            ),
            checkpoint_count=checkpoint_count,
            resume_count=resume_count,
            cuda_memory_peak=cuda_memory_peak,
            final_prediction_path=prediction_path,
            final_metadata_path=metadata_path,
        )

    print()
    print("Prediction summary:", summary)
    print("fit seconds:", round(fit_seconds, 3))
    print(
        "prediction seconds:",
        round(cumulative_prediction_seconds, 3),
    )
    print("resume count:", resume_count)
    print("checkpoint count:", checkpoint_count)

    if cuda_memory_peak is not None:
        print(
            "CUDA peak allocated GiB:",
            round(
                float(
                    cuda_memory_peak[
                        "cuda_peak_memory_allocated_gib"
                    ]
                ),
                3,
            ),
        )
        print(
            "CUDA peak reserved GiB:",
            round(
                float(
                    cuda_memory_peak[
                        "cuda_peak_memory_reserved_gib"
                    ]
                ),
                3,
            ),
        )
        print(
            "GPU total memory GiB:",
            round(
                float(cuda_memory_peak["gpu_total_memory_gib"]),
                3,
            ),
        )

    print("saved predictions:", prediction_path)
    print("saved metadata:", metadata_path)

    if args.checkpoint_every > 0:
        print("saved completed progress:", progress_path)

    print("KRONOS CACHED INFERENCE RUN PASSED")


if __name__ == "__main__":
    main()

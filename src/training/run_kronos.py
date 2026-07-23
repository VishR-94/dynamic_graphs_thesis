from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.models.kronos import KronosBaseline
from src.utils.config import load_yaml


ConfigDict = dict[str, Any]
SplitDict = dict[str, Any]


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
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be overwritten.",
    )

    return parser


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
    """Return the dtype of the first parameter in a module."""
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
    """Return current and peak CUDA memory statistics in GiB."""
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


def build_output_paths(
    output_dir: Path,
    run_name: str | None,
    evaluation_split: str,
) -> tuple[Path, Path]:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"kronos_{evaluation_split}_{timestamp}"

    run_name = run_name.strip()
    if not run_name:
        raise ValueError("--run-name must not be empty.")

    prediction_path = output_dir / f"{run_name}_predictions.pt"
    metadata_path = output_dir / f"{run_name}_metadata.json"

    return prediction_path, metadata_path


def validate_output_paths(
    prediction_path: Path,
    metadata_path: Path,
    overwrite: bool,
) -> None:
    existing = [
        path
        for path in (prediction_path, metadata_path)
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Output file(s) already exist. Use --overwrite to replace "
            f"them: {existing}"
        )


def prediction_summary(
    prediction_result: dict[str, Any],
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


def main() -> None:
    args = build_argument_parser().parse_args()

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

    config_path, data_dir, output_dir = validate_paths(
        args.config,
        args.data_dir,
        args.output_dir,
    )

    prediction_path, metadata_path = build_output_paths(
        output_dir=output_dir,
        run_name=args.run_name,
        evaluation_split=args.evaluation_split,
    )
    validate_output_paths(
        prediction_path=prediction_path,
        metadata_path=metadata_path,
        overwrite=args.overwrite,
    )

    config = load_yaml(config_path)
    resolved_config = apply_cli_overrides(
        config,
        args,
    )

    print("Loading and cleaning chronological splits...")
    train_raw, val_raw, test_raw = load_candle_splits(
        data_dir
    )
    train, val, test = clean_candle_splits(
        train_raw,
        val_raw,
        test_raw,
    )

    train = retain_first_assets(
        train,
        args.max_assets,
    )
    val = retain_first_assets(
        val,
        args.max_assets,
    )
    test = retain_first_assets(
        test,
        args.max_assets,
    )

    prediction_split = (
        val
        if args.evaluation_split == "val"
        else test
    )

    model = KronosBaseline.from_config(
        resolved_config
    )

    print("Preparing frozen Kronos checkpoints...")
    fit_start = perf_counter()
    model.fit(
        train_split=train,
        val_split=val,
    )
    fit_seconds = perf_counter() - fit_start

    print(
        "Generating predictions:",
        {
            "split": args.evaluation_split,
            "device": model.device,
            "dtype": model.dtype,
            "sample_count": model.sample_count,
            "series_batch_size": model.series_batch_size,
            "window_batch_size": args.window_batch_size,
            "max_examples": args.max_examples,
            "max_assets": args.max_assets,
        },
    )

    cuda_device: torch.device | None = None
    cuda_memory_before_prediction: dict[str, float | str] | None = None
    cuda_memory_after_prediction: dict[str, float | str] | None = None

    if model.device is not None and model.device.startswith("cuda"):
        cuda_device = torch.device(model.device)

        torch.cuda.synchronize(cuda_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)

        cuda_memory_before_prediction = cuda_memory_snapshot(
            cuda_device
        )

    prediction_start = perf_counter()

    prediction_result = model.predict(
        split=prediction_split,
        batch_size=args.window_batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_examples,
    )

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)

    prediction_seconds = (
        perf_counter() - prediction_start
    )

    if cuda_device is not None:
        cuda_memory_after_prediction = cuda_memory_snapshot(
            cuda_device
        )

    summary = prediction_summary(
        prediction_result
    )

    run_metadata = {
        "created_at": datetime.now().isoformat(),
        "evaluation_split": args.evaluation_split,
        "config_path": str(config_path),
        "data_dir": str(data_dir),
        "prediction_path": str(prediction_path),
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
        "model_parameter_dtype": first_parameter_dtype(
            model.model
        ),
        "tokenizer_parameter_dtype": first_parameter_dtype(
            model.tokenizer
        ),
        "cuda_memory_before_prediction": (
            cuda_memory_before_prediction
        ),
        "cuda_memory_after_prediction": (
            cuda_memory_after_prediction
        ),
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
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_summary": summary,
        "project_git_revision": git_revision(Path.cwd()),
        "kronos_git_revision": git_revision(
            Path.cwd() / "external" / "Kronos"
        ),
        "package_versions": {
            "python": (
                f"{__import__('sys').version_info.major}."
                f"{__import__('sys').version_info.minor}."
                f"{__import__('sys').version_info.micro}"
            ),
            "torch": package_version("torch"),
            "pandas": package_version("pandas"),
            "numpy": package_version("numpy"),
            "huggingface_hub": package_version(
                "huggingface-hub"
            ),
            "safetensors": package_version("safetensors"),
            "einops": package_version("einops"),
        },
        "resolved_kronos_config": resolved_config[
            "models"
        ]["kronos"],
    }

    cache = {
        "prediction_result": prediction_result,
        "run_metadata": run_metadata,
    }

    torch.save(
        cache,
        prediction_path,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_metadata,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print("Prediction summary:", summary)
    print(
        "fit seconds:",
        round(fit_seconds, 3),
    )
    print(
        "prediction seconds:",
        round(prediction_seconds, 3),
    )

    if cuda_memory_after_prediction is not None:
        print(
            "CUDA peak allocated GiB:",
            round(
                float(
                    cuda_memory_after_prediction[
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
                    cuda_memory_after_prediction[
                        "cuda_peak_memory_reserved_gib"
                    ]
                ),
                3,
            ),
        )
        print(
            "GPU total memory GiB:",
            round(
                float(
                    cuda_memory_after_prediction[
                        "gpu_total_memory_gib"
                    ]
                ),
                3,
            ),
        )

    print("saved predictions:", prediction_path)
    print("saved metadata:", metadata_path)
    print("KRONOS CACHED INFERENCE RUN PASSED")


if __name__ == "__main__":
    main()
from __future__ import annotations

"""Public orchestration API for the additive weather benchmark package."""

from contextlib import nullcontext
from pathlib import Path
import shutil
import time
import urllib.request
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    MODERN_TCN_KERNEL_GRID_BY_HORIZON,
    ModelKind,
    SUPPORTED_CITIES,
    WEATHER_HORIZON_TO_CONTEXT,
    WeatherRunConfig,
)
from .data import build_weather_data_bundle
from .models import build_weather_model, parameter_counts
from .trainer import TrainingResult, resolve_device, set_seed, train_weather_model


SONNET_WEATHER_URL = (
    "https://raw.githubusercontent.com/ClaudiaShu/Sonnet/"
    "main/datasets/weatherbench/weather_{city}.csv"
)


def ensure_weather_csv(city: str, cache_directory: str | Path) -> Path:
    """Download the official Sonnet city CSV once and return its local path."""

    canonical = str(city).lower().strip()
    if canonical not in SUPPORTED_CITIES:
        raise ValueError(
            f"Unsupported city {canonical!r}; expected one of {SUPPORTED_CITIES}."
        )
    root = Path(cache_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"weather_{canonical}.csv"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    temporary = destination.with_suffix(".csv.part")
    request = urllib.request.Request(
        SONNET_WEATHER_URL.format(city=canonical),
        headers={"User-Agent": "dynamic-graphs-thesis-weather-benchmark"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        if temporary.stat().st_size <= 0:
            raise RuntimeError("Downloaded weather CSV is empty.")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def make_weather_run_config(
    *,
    model_kind: ModelKind,
    city: str,
    test_year: int,
    horizon: int,
    data_path: str | Path,
    output_root: str | Path,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    skip_completed: bool = True,
    export_train_split: bool = True,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 0,
    modern_tcn_large_kernel: int = 15,
    train_batch_size: int | None = None,
    validation_batch_size: int | None = None,
    export_batch_size: int | None = None,
    run_suffix: str | None = None,
    cache_causal_masks: bool = False,
    progress_update_interval: int = 1,
    prefetch_factor: int = 2,
) -> WeatherRunConfig:
    return WeatherRunConfig(
        model_kind=model_kind,
        city=city,
        test_year=int(test_year),
        horizon=int(horizon),
        data_path=Path(data_path),
        output_root=Path(output_root),
        device=device,
        resume=bool(resume),
        overwrite=bool(overwrite),
        skip_completed=bool(skip_completed),
        export_train_split=bool(export_train_split),
        max_epochs=int(max_epochs),
        patience=int(patience),
        num_workers=int(num_workers),
        modern_tcn_large_kernel=int(modern_tcn_large_kernel),
        train_batch_size_override=(
            None if train_batch_size is None else int(train_batch_size)
        ),
        validation_batch_size_override=(
            None if validation_batch_size is None else int(validation_batch_size)
        ),
        export_batch_size_override=(
            None if export_batch_size is None else int(export_batch_size)
        ),
        run_suffix=run_suffix,
        cache_causal_masks=bool(cache_causal_masks),
        progress_update_interval=int(progress_update_interval),
        prefetch_factor=int(prefetch_factor),
    )


def preflight_weather_run(
    *,
    model_kind: ModelKind,
    city: str,
    test_year: int,
    horizon: int,
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    device: str = "auto",
    modern_tcn_large_kernel: int = 15,
    train_batch_size: int | None = None,
    validation_batch_size: int | None = None,
    export_batch_size: int | None = None,
    run_suffix: str | None = None,
    cache_causal_masks: bool = False,
    progress_update_interval: int = 1,
    prefetch_factor: int = 2,
) -> dict[str, Any]:
    """Build one dataset/model and run a no-gradient shape check."""

    config = make_weather_run_config(
        model_kind=model_kind,
        city=city,
        test_year=test_year,
        horizon=horizon,
        data_path=data_path,
        output_root=output_root,
        device=device,
        export_train_split=False,
        modern_tcn_large_kernel=modern_tcn_large_kernel,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        export_batch_size=export_batch_size,
        run_suffix=run_suffix,
        cache_causal_masks=cache_causal_masks,
        progress_update_interval=progress_update_interval,
        prefetch_factor=prefetch_factor,
    )
    set_seed(config.seed)
    data = build_weather_data_bundle(config)
    model_bundle = build_weather_model(config, data)
    resolved_device = resolve_device(device)
    model = model_bundle.model.to(resolved_device).eval()
    dataset = data.dataset("train", dense_prefix=config.dense_prefix_training)
    sample = dataset[0]
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    with torch.inference_mode():
        x = batch["x"].to(resolved_device)
        if model_kind == "modern_tcn_1st":
            output = model(
                x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
            prediction_shape = list(output.predictions.shape)
            graph_shapes = [list(output.graph.selected.shape)]
        else:
            output = model.forward_dense(x)
            prediction_shape = list(output.predictions.shape)
            graph_shapes = [
                list(block.graph.selected[:, -1].shape)
                for block in output.block_outputs
            ]
    return {
        "model_kind": model_kind,
        "city": city,
        "test_year": int(test_year),
        "horizon": int(horizon),
        "context_length": int(config.context_length),
        "modern_tcn_large_kernel": (
            int(config.modern_tcn_large_kernel)
            if model_kind == "modern_tcn_1st"
            else None
        ),
        "run_suffix": config.run_suffix,
        "train_batch_size": int(config.batch_size),
        "validation_batch_size": int(config.validation_batch_size),
        "export_batch_size": int(config.export_batch_size),
        "cache_causal_masks": bool(config.cache_causal_masks),
        "run_directory": str(config.run_directory),
        "window_counts": {
            name: split.sample_count for name, split in data.splits.items()
        },
        "input_shape": list(batch["x"].shape),
        "training_target_shape": list(
            batch["dense_y"].shape
            if config.dense_prefix_training
            else batch["y"].shape
        ),
        "prediction_shape": prediction_shape,
        "saved_final_graph_shapes": graph_shapes,
        "parameter_counts": parameter_counts(model),
        "device": str(resolved_device),
        "project_root": str(Path(project_root).resolve()),
    }


def probe_weather_training_batch(
    *,
    model_kind: ModelKind,
    city: str,
    test_year: int,
    horizon: int,
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    device: str = "auto",
    modern_tcn_large_kernel: int = 15,
    train_batch_size: int | None = None,
    validation_batch_size: int | None = None,
    export_batch_size: int | None = None,
    run_suffix: str | None = None,
    cache_causal_masks: bool = False,
) -> dict[str, Any]:
    """Run one real forward/backward batch without updating parameters.

    This is intended as a Colab memory/throughput preflight for the accelerated
    3ST batch size.  It writes no checkpoint or experiment artifact.
    """

    config = make_weather_run_config(
        model_kind=model_kind,
        city=city,
        test_year=test_year,
        horizon=horizon,
        data_path=data_path,
        output_root=output_root,
        device=device,
        export_train_split=False,
        modern_tcn_large_kernel=modern_tcn_large_kernel,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        export_batch_size=export_batch_size,
        run_suffix=run_suffix,
        cache_causal_masks=cache_causal_masks,
    )
    set_seed(config.seed)
    data = build_weather_data_bundle(config)
    model_bundle = build_weather_model(config, data)
    resolved_device = resolve_device(device)
    model = model_bundle.model.to(resolved_device).train()
    dataset = data.dataset("train", dense_prefix=config.dense_prefix_training)
    loader = DataLoader(
        dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(resolved_device.type == "cuda"),
        drop_last=False,
    )
    batch = next(iter(loader))
    x = torch.as_tensor(batch["x"]).to(
        device=resolved_device, dtype=torch.float32, non_blocking=True
    )
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(resolved_device)
        torch.cuda.synchronize(resolved_device)
    started = time.perf_counter()
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if resolved_device.type == "cuda"
        else nullcontext()
    )
    with amp:
        if model_kind == "modern_tcn_1st":
            output = model(
                x,
                context_start=torch.as_tensor(batch["context_start"]),
                session_length=torch.as_tensor(batch["session_length"]),
            )
            prediction = output.predictions
            target = torch.as_tensor(batch["y"]).to(
                device=resolved_device, dtype=torch.float32, non_blocking=True
            )
        else:
            output = model.forward_dense(x)
            prediction = output.predictions
            target = torch.as_tensor(batch["dense_y"]).to(
                device=resolved_device, dtype=torch.float32, non_blocking=True
            )
        loss = F.mse_loss(prediction.float(), target.float())
    loss.backward()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = time.perf_counter() - started
    result = {
        "model_kind": model_kind,
        "horizon": int(horizon),
        "context_length": int(config.context_length),
        "batch_size": int(x.shape[0]),
        "input_shape": list(x.shape),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "loss": float(loss.detach().item()),
        "forward_backward_seconds": float(elapsed),
        "peak_cuda_memory_allocated_gib": (
            float(torch.cuda.max_memory_allocated(resolved_device) / (1024**3))
            if resolved_device.type == "cuda"
            else None
        ),
        "peak_cuda_memory_reserved_gib": (
            float(torch.cuda.max_memory_reserved(resolved_device) / (1024**3))
            if resolved_device.type == "cuda"
            else None
        ),
        "run_directory": str(config.run_directory),
        "project_root": str(Path(project_root).resolve()),
    }
    model.zero_grad(set_to_none=True)
    del output, prediction, target, loss, x, batch, loader, dataset, model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_weather_experiment(
    *,
    model_kind: ModelKind,
    city: str,
    test_year: int,
    horizon: int,
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    skip_completed: bool = True,
    export_train_split: bool = True,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 0,
    modern_tcn_large_kernel: int = 15,
    train_batch_size: int | None = None,
    validation_batch_size: int | None = None,
    export_batch_size: int | None = None,
    run_suffix: str | None = None,
    cache_causal_masks: bool = False,
    progress_update_interval: int = 1,
    prefetch_factor: int = 2,
) -> TrainingResult:
    config = make_weather_run_config(
        model_kind=model_kind,
        city=city,
        test_year=test_year,
        horizon=horizon,
        data_path=data_path,
        output_root=output_root,
        device=device,
        resume=resume,
        overwrite=overwrite,
        skip_completed=skip_completed,
        export_train_split=export_train_split,
        max_epochs=max_epochs,
        patience=patience,
        num_workers=num_workers,
        modern_tcn_large_kernel=modern_tcn_large_kernel,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        export_batch_size=export_batch_size,
        run_suffix=run_suffix,
        cache_causal_masks=cache_causal_masks,
        progress_update_interval=progress_update_interval,
        prefetch_factor=prefetch_factor,
    )
    set_seed(config.seed)
    data = build_weather_data_bundle(config)
    model_bundle = build_weather_model(config, data)
    return train_weather_model(
        config=config,
        data=data,
        model_bundle=model_bundle,
        project_root=Path(project_root).expanduser().resolve(),
    )


def _summary_row(
    *,
    result: TrainingResult,
    model_kind: ModelKind,
    city: str,
    test_year: int,
    horizon: int,
    modern_tcn_large_kernel: int,
    run_suffix: str | None,
    train_batch_size: int,
) -> dict[str, Any]:
    reported = result.test_metrics.get("reported", {})
    return {
        "model_kind": model_kind,
        "city": city,
        "test_year": int(test_year),
        "horizon": int(horizon),
        "context_length": WEATHER_HORIZON_TO_CONTEXT[int(horizon)],
        "modern_tcn_large_kernel": (
            int(modern_tcn_large_kernel)
            if model_kind == "modern_tcn_1st"
            else None
        ),
        "run_suffix": run_suffix,
        "train_batch_size": int(train_batch_size),
        "status": "completed",
        "best_epoch": int(result.best_epoch),
        "best_validation_score": float(result.best_validation_score),
        "test_mae": reported.get("mae"),
        "test_r": reported.get("r"),
        "test_smape": reported.get("smape"),
        "run_directory": str(result.run_directory),
    }


def run_weather_suite(
    *,
    model_kinds: Sequence[ModelKind],
    city: str,
    test_year: int,
    horizons: Sequence[int],
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    skip_completed: bool = True,
    export_train_split: bool = True,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 0,
    continue_on_error: bool = False,
    modern_tcn_large_kernel: int = 15,
    train_batch_size: int | None = None,
    validation_batch_size: int | None = None,
    export_batch_size: int | None = None,
    run_suffix: str | None = None,
    cache_causal_masks: bool = False,
    progress_update_interval: int = 1,
    prefetch_factor: int = 2,
) -> pd.DataFrame:
    """Run the requested model/horizon grid sequentially with auto-resume."""

    invalid_horizons = [
        int(value) for value in horizons if int(value) not in WEATHER_HORIZON_TO_CONTEXT
    ]
    if invalid_horizons:
        raise ValueError(f"Unsupported weather horizons: {invalid_horizons}")
    rows: list[dict[str, Any]] = []
    for model_kind in model_kinds:
        resolved_train_batch_size = (
            int(train_batch_size)
            if train_batch_size is not None
            else (16 if model_kind == "modern_tcn_1st" else 1)
        )
        for horizon in horizons:
            print("=" * 88)
            print(
                f"Starting {model_kind} | city={city} | test_year={test_year} | "
                f"H={int(horizon)} | L={WEATHER_HORIZON_TO_CONTEXT[int(horizon)]} | "
                f"kernel={int(modern_tcn_large_kernel) if model_kind == 'modern_tcn_1st' else '-'} | "
                f"batch={resolved_train_batch_size} | suffix={run_suffix or '-'}"
            )
            try:
                result = run_weather_experiment(
                    model_kind=model_kind,
                    city=city,
                    test_year=int(test_year),
                    horizon=int(horizon),
                    data_path=data_path,
                    output_root=output_root,
                    project_root=project_root,
                    device=device,
                    resume=resume,
                    overwrite=overwrite,
                    skip_completed=skip_completed,
                    export_train_split=export_train_split,
                    max_epochs=max_epochs,
                    patience=patience,
                    num_workers=num_workers,
                    modern_tcn_large_kernel=modern_tcn_large_kernel,
                    train_batch_size=train_batch_size,
                    validation_batch_size=validation_batch_size,
                    export_batch_size=export_batch_size,
                    run_suffix=run_suffix,
                    cache_causal_masks=cache_causal_masks,
                    progress_update_interval=progress_update_interval,
                    prefetch_factor=prefetch_factor,
                )
                rows.append(
                    _summary_row(
                        result=result,
                        model_kind=model_kind,
                        city=city,
                        test_year=int(test_year),
                        horizon=int(horizon),
                        modern_tcn_large_kernel=int(modern_tcn_large_kernel),
                        run_suffix=run_suffix,
                        train_batch_size=resolved_train_batch_size,
                    )
                )
            except Exception as error:
                rows.append(
                    {
                        "model_kind": model_kind,
                        "city": city,
                        "test_year": int(test_year),
                        "horizon": int(horizon),
                        "context_length": WEATHER_HORIZON_TO_CONTEXT[int(horizon)],
                        "modern_tcn_large_kernel": (
                            int(modern_tcn_large_kernel)
                            if model_kind == "modern_tcn_1st"
                            else None
                        ),
                        "run_suffix": run_suffix,
                        "train_batch_size": resolved_train_batch_size,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if not continue_on_error:
                    raise
    return pd.DataFrame(rows)


def _validate_kernel_grid(
    horizons: Sequence[int],
    kernel_grid: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    resolved: dict[int, tuple[int, ...]] = {}
    for horizon in horizons:
        horizon_value = int(horizon)
        if horizon_value not in WEATHER_HORIZON_TO_CONTEXT:
            raise ValueError(f"Unsupported weather horizon: {horizon_value}")
        if horizon_value not in kernel_grid:
            raise KeyError(f"Kernel grid has no entry for H={horizon_value}.")
        kernels = tuple(int(value) for value in kernel_grid[horizon_value])
        if len(kernels) != 3 or len(set(kernels)) != 3:
            raise ValueError(
                f"H={horizon_value} must have exactly three distinct kernels; "
                f"received {kernels}."
            )
        if any(value < 5 or value % 2 == 0 for value in kernels):
            raise ValueError(
                f"H={horizon_value} kernels must be odd and at least 5: {kernels}."
            )
        resolved[horizon_value] = kernels
    return resolved


def run_modern_tcn_kernel_sweep(
    *,
    city: str,
    test_year: int,
    horizons: Sequence[int],
    data_path: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    kernel_grid: Mapping[int, Sequence[int]] = MODERN_TCN_KERNEL_GRID_BY_HORIZON,
    device: str = "auto",
    resume: bool = True,
    overwrite: bool = False,
    skip_completed: bool = True,
    export_train_split: bool = True,
    max_epochs: int = 100,
    patience: int = 10,
    num_workers: int = 0,
    continue_on_error: bool = False,
    progress_update_interval: int = 1,
    prefetch_factor: int = 2,
) -> pd.DataFrame:
    """Run three ModernTCN kernel candidates per requested weather horizon.

    Every candidate is isolated under
    ``horizon_H/test_year_YEAR_kernel_K``.  Candidate ranking in the returned
    table is computed from the validation selection score only.
    """

    resolved_grid = _validate_kernel_grid(horizons, kernel_grid)
    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        horizon_value = int(horizon)
        for kernel in resolved_grid[horizon_value]:
            frame = run_weather_suite(
                model_kinds=("modern_tcn_1st",),
                city=city,
                test_year=int(test_year),
                horizons=(horizon_value,),
                data_path=data_path,
                output_root=output_root,
                project_root=project_root,
                device=device,
                resume=resume,
                overwrite=overwrite,
                skip_completed=skip_completed,
                export_train_split=export_train_split,
                max_epochs=max_epochs,
                patience=patience,
                num_workers=num_workers,
                continue_on_error=continue_on_error,
                modern_tcn_large_kernel=kernel,
                run_suffix=f"kernel_{kernel}",
                progress_update_interval=progress_update_interval,
                prefetch_factor=prefetch_factor,
            )
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result["validation_rank_within_horizon"] = pd.NA
    completed = result["status"].eq("completed")
    result.loc[completed, "validation_rank_within_horizon"] = (
        result.loc[completed]
        .groupby("horizon")["best_validation_score"]
        .rank(method="min", ascending=True)
    )
    return result.sort_values(
        ["horizon", "validation_rank_within_horizon", "modern_tcn_large_kernel"],
        na_position="last",
    ).reset_index(drop=True)

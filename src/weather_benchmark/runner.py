from __future__ import annotations

"""Public orchestration API for the additive weather benchmark package."""

from pathlib import Path
import shutil
import urllib.request
from typing import Any, Sequence

import pandas as pd
import torch

from .config import ModelKind, SUPPORTED_CITIES, WEATHER_HORIZON_TO_CONTEXT, WeatherRunConfig
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
) -> pd.DataFrame:
    """Run the requested model/horizon grid sequentially with auto-resume."""

    invalid_horizons = [
        int(value) for value in horizons if int(value) not in WEATHER_HORIZON_TO_CONTEXT
    ]
    if invalid_horizons:
        raise ValueError(f"Unsupported weather horizons: {invalid_horizons}")
    rows: list[dict[str, Any]] = []
    for model_kind in model_kinds:
        for horizon in horizons:
            print("=" * 88)
            print(
                f"Starting {model_kind} | city={city} | test_year={test_year} | "
                f"H={int(horizon)} | L={WEATHER_HORIZON_TO_CONTEXT[int(horizon)]}"
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
                )
                reported = result.test_metrics.get("reported", {})
                rows.append(
                    {
                        "model_kind": model_kind,
                        "city": city,
                        "test_year": int(test_year),
                        "horizon": int(horizon),
                        "context_length": WEATHER_HORIZON_TO_CONTEXT[int(horizon)],
                        "status": "completed",
                        "best_epoch": int(result.best_epoch),
                        "best_validation_score": float(result.best_validation_score),
                        "test_mae": reported.get("mae"),
                        "test_r": reported.get("r"),
                        "test_smape": reported.get("smape"),
                        "run_directory": str(result.run_directory),
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "model_kind": model_kind,
                        "city": city,
                        "test_year": int(test_year),
                        "horizon": int(horizon),
                        "context_length": WEATHER_HORIZON_TO_CONTEXT[int(horizon)],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if not continue_on_error:
                    raise
    return pd.DataFrame(rows)

"""Additive weather benchmark support for the frozen graph architectures."""

from .audit import audit_modern_tcn_kernel15_replications
from .config import (
    CENTRAL_NODE_INDEX,
    MODERN_TCN_KERNEL_GRID_BY_HORIZON,
    MODERN_TCN_PATCH_STRIDE_GRID_BY_HORIZON,
    MODERN_TCN_SELECTED_KERNEL_BY_HORIZON,
    MODERN_TCN_WIDTH_GRID,
    MODEL_OUTPUT_DIRECTORIES,
    SUPPORTED_CITIES,
    WEATHER_FEATURES,
    WEATHER_HORIZON_TO_CONTEXT,
    WEATHER_NODES,
    WeatherRunConfig,
)
from .runner import (
    ensure_weather_csv,
    make_weather_run_config,
    modern_tcn_stride_width_run_suffix,
    preflight_weather_run,
    probe_weather_training_batch,
    run_modern_tcn_kernel_sweep,
    run_modern_tcn_stride_width_sweep,
    run_weather_experiment,
    run_weather_suite,
)

__all__ = [
    "CENTRAL_NODE_INDEX",
    "MODERN_TCN_KERNEL_GRID_BY_HORIZON",
    "MODERN_TCN_PATCH_STRIDE_GRID_BY_HORIZON",
    "MODERN_TCN_SELECTED_KERNEL_BY_HORIZON",
    "MODERN_TCN_WIDTH_GRID",
    "MODEL_OUTPUT_DIRECTORIES",
    "SUPPORTED_CITIES",
    "WEATHER_FEATURES",
    "WEATHER_HORIZON_TO_CONTEXT",
    "WEATHER_NODES",
    "WeatherRunConfig",
    "audit_modern_tcn_kernel15_replications",
    "ensure_weather_csv",
    "make_weather_run_config",
    "modern_tcn_stride_width_run_suffix",
    "preflight_weather_run",
    "probe_weather_training_batch",
    "run_modern_tcn_kernel_sweep",
    "run_modern_tcn_stride_width_sweep",
    "run_weather_experiment",
    "run_weather_suite",
]

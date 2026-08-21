from __future__ import annotations

"""Command-line runner for the additive Sonnet weather benchmark package."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weather_benchmark.config import (  # noqa: E402
    MODERN_TCN_KERNEL_GRID_BY_HORIZON,
    SUPPORTED_CITIES,
    WEATHER_HORIZON_TO_CONTEXT,
)
from src.weather_benchmark.runner import (  # noqa: E402
    ensure_weather_csv,
    run_modern_tcn_kernel_sweep,
    run_weather_suite,
)


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Horizons must be comma-separated integers.") from error
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon is required.")
    invalid = [item for item in horizons if item not in WEATHER_HORIZON_TO_CONTEXT]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported horizons {invalid}; expected {tuple(WEATHER_HORIZON_TO_CONTEXT)}."
        )
    return horizons


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen 1ST ModernTCN and/or 3ST Transformer architectures "
            "on the Sonnet weather benchmark."
        )
    )
    parser.add_argument(
        "--model",
        choices=("modern_tcn_1st", "transformer_3st", "all"),
        default="all",
    )
    parser.add_argument("--city", choices=SUPPORTED_CITIES, default="capetown")
    parser.add_argument("--test-year", type=int, default=2018)
    parser.add_argument(
        "--horizons",
        type=_parse_horizons,
        default=tuple(WEATHER_HORIZON_TO_CONTEXT),
        help="Comma-separated subset of 4,12,28,120.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root weather directory containing model/city/horizon/year folders.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Existing official Sonnet city CSV. Downloaded automatically when omitted.",
    )
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "sonnet_weather",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--modern-tcn-kernel-sweep",
        action="store_true",
        help="Run the three-kernel validation grid for every requested horizon.",
    )
    parser.add_argument("--modern-tcn-large-kernel", type=int, default=15)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--export-batch-size", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument(
        "--cache-causal-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress-update-interval", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--export-train-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    model_kinds = (
        ("modern_tcn_1st", "transformer_3st")
        if args.model == "all"
        else (args.model,)
    )
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else ensure_weather_csv(args.city, args.data_cache)
    )
    if args.modern_tcn_kernel_sweep:
        if args.model not in {"modern_tcn_1st", "all"}:
            raise ValueError("Kernel sweep requires --model modern_tcn_1st or all.")
        summary = run_modern_tcn_kernel_sweep(
            city=args.city,
            test_year=args.test_year,
            horizons=args.horizons,
            data_path=data_path,
            output_root=args.output_root,
            project_root=PROJECT_ROOT,
            kernel_grid=MODERN_TCN_KERNEL_GRID_BY_HORIZON,
            device=args.device,
            resume=args.resume,
            overwrite=args.overwrite,
            skip_completed=args.skip_completed,
            export_train_split=args.export_train_split,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            continue_on_error=args.continue_on_error,
            progress_update_interval=args.progress_update_interval,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        summary = run_weather_suite(
            model_kinds=model_kinds,
            city=args.city,
            test_year=args.test_year,
            horizons=args.horizons,
            data_path=data_path,
            output_root=args.output_root,
            project_root=PROJECT_ROOT,
            device=args.device,
            resume=args.resume,
            overwrite=args.overwrite,
            skip_completed=args.skip_completed,
            export_train_split=args.export_train_split,
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            continue_on_error=args.continue_on_error,
            modern_tcn_large_kernel=args.modern_tcn_large_kernel,
            train_batch_size=args.train_batch_size,
            validation_batch_size=args.validation_batch_size,
            export_batch_size=args.export_batch_size,
            run_suffix=args.run_suffix,
            cache_causal_masks=args.cache_causal_masks,
            progress_update_interval=args.progress_update_interval,
            prefetch_factor=args.prefetch_factor,
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / (
        f"suite_summary_{args.city}_test{args.test_year}.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved suite summary: {summary_path}")


if __name__ == "__main__":
    main()

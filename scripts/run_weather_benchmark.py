from __future__ import annotations

"""Command-line runner for the additive Sonnet weather benchmark package."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weather_benchmark.config import (  # noqa: E402
    SUPPORTED_CITIES,
    WEATHER_HORIZON_TO_CONTEXT,
)
from src.weather_benchmark.runner import ensure_weather_csv, run_weather_suite  # noqa: E402


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

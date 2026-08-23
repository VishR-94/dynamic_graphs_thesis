from __future__ import annotations

"""CLI for the fixed Graph-ModernTCN all-city/all-test-year transfer."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd

from src.weather_benchmark.final_transfer import (
    FINAL_TRANSFER_CITIES,
    FINAL_TRANSFER_HORIZONS,
    FINAL_TRANSFER_TEST_YEARS,
    CITY_DISPLAY_NAMES,
    audit_all_weather_city_csvs,
    build_city_test_metric_table,
    collect_selected_modern_tcn_transfer_metrics,
    run_selected_modern_tcn_transfer,
    save_selected_transfer_summaries,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--summary-directory", type=Path, default=None)
    parser.add_argument("--cities", nargs="+", default=list(FINAL_TRANSFER_CITIES))
    parser.add_argument(
        "--test-years",
        nargs="+",
        type=int,
        default=list(FINAL_TRANSFER_TEST_YEARS),
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(FINAL_TRANSFER_HORIZONS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-update-interval", type=int, default=50)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-completed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--export-train-split", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic-runtime",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary_root = (
        args.summary_directory
        if args.summary_directory is not None
        else args.output_root / "final_selected_modernTCN_transfer"
    )
    summary_root.mkdir(parents=True, exist_ok=True)

    quality = audit_all_weather_city_csvs(
        cities=args.cities,
        data_cache_root=args.data_cache_root,
    )
    quality.to_csv(summary_root / "city_data_quality.csv", index=False)
    print(quality.to_string(index=False))

    run_selected_modern_tcn_transfer(
        output_root=args.output_root,
        data_cache_root=args.data_cache_root,
        project_root=args.project_root,
        summary_directory=summary_root,
        cities=args.cities,
        test_years=args.test_years,
        horizons=args.horizons,
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
        deterministic_runtime=args.deterministic_runtime,
    )

    metrics = collect_selected_modern_tcn_transfer_metrics(
        output_root=args.output_root,
        cities=args.cities,
        test_years=args.test_years,
        horizons=args.horizons,
    )
    paths = save_selected_transfer_summaries(
        metrics=metrics,
        output_root=args.output_root,
        summary_directory=summary_root,
        cities=args.cities,
        test_years=args.test_years,
        horizons=args.horizons,
    )

    for city in args.cities:
        print("\n" + CITY_DISPLAY_NAMES[str(city).lower().strip()])
        table = build_city_test_metric_table(
            metrics,
            city=city,
            test_years=args.test_years,
            horizons=args.horizons,
        )
        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(table.to_string(float_format=lambda value: f"{value:.4f}"))

    incomplete = metrics.loc[~metrics["status"].eq("completed")]
    if not incomplete.empty:
        raise RuntimeError(
            "Some expected runs are incomplete:\n"
            + incomplete[
                ["city", "test_year", "horizon", "status", "run_directory"]
            ].to_string(index=False)
        )
    print("\nSaved summaries:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

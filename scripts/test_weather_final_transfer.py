from __future__ import annotations

"""Contract tests for the fixed all-city weather transfer orchestration."""

from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.weather_benchmark.final_transfer import (
    FINAL_TRANSFER_CITIES,
    FINAL_TRANSFER_HORIZONS,
    FINAL_TRANSFER_TEST_YEARS,
    SELECTED_MODERN_TCN_ARCHITECTURES,
    build_city_test_metric_table,
    collect_selected_modern_tcn_transfer_metrics,
    selected_transfer_plan,
    selected_transfer_run_directory,
)


def test_selected_architectures() -> None:
    expected = {
        4: (7, 2, 32, 32),
        12: (7, 4, 32, 32),
        28: (15, 8, 32, 32),
        120: (119, 8, 32, 32),
    }
    actual = {
        horizon: (
            spec.large_kernel,
            spec.patch_stride,
            spec.d_model,
            spec.graph_hidden_dim,
        )
        for horizon, spec in SELECTED_MODERN_TCN_ARCHITECTURES.items()
    }
    assert actual == expected


def test_plan_has_60_unique_runs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plan = selected_transfer_plan(
            output_root=root / "weather",
            data_cache_root=root / "cache",
        )
    assert len(plan) == 5 * 3 * 4 == 60
    assert plan["run_directory"].nunique() == 60
    assert set(plan["city"]) == set(FINAL_TRANSFER_CITIES)
    assert set(plan["test_year"]) == set(FINAL_TRANSFER_TEST_YEARS)
    assert set(plan["horizon"]) == set(FINAL_TRANSFER_HORIZONS)


def test_hongkong_2018_reuses_selected_sweep_path() -> None:
    spec = SELECTED_MODERN_TCN_ARCHITECTURES[28]
    path = selected_transfer_run_directory(
        output_root=Path("/tmp/weather"),
        city="hongkong",
        test_year=2018,
        architecture=spec,
    )
    assert path.name == (
        "test_year_2018_kernel_15_stride_8_dmodel_32_graphdim_32"
    )


def test_city_metric_table_contract() -> None:
    rows = []
    for year in FINAL_TRANSFER_TEST_YEARS:
        for horizon in FINAL_TRANSFER_HORIZONS:
            rows.append(
                {
                    "city": "hongkong",
                    "test_year": year,
                    "horizon": horizon,
                    "test_mae": float(year + horizon),
                    "test_r": float(horizon) / 120.0,
                    "test_smape": float(year - 2000),
                    "run_directory": f"run_{year}_{horizon}",
                }
            )
    frame = pd.DataFrame(rows)
    table = build_city_test_metric_table(frame, city="hongkong")
    assert table.shape == (4, 9)
    assert isinstance(table.columns, pd.MultiIndex)
    assert table.index.tolist() == list(FINAL_TRANSFER_HORIZONS)
    assert table.columns.tolist() == [
        (year, metric)
        for year in FINAL_TRANSFER_TEST_YEARS
        for metric in ("MAE", "r", "sMAPE")
    ]
    assert np.isfinite(table.to_numpy()).all()



def test_metric_collector_resolves_default_width_fields() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = SELECTED_MODERN_TCN_ARCHITECTURES[12]
        run_dir = selected_transfer_run_directory(
            output_root=root,
            city="hongkong",
            test_year=2018,
            architecture=spec,
        )
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "modern_tcn_large_kernel": 7,
                    # stride=4, d_model=32 and graph_dim=32 are omitted by the
                    # legacy-compatible config serializer.
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_complete.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "best_epoch": 9,
                    "best_validation_score": 0.25,
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "reported": {"mae": 1.1, "r": 0.8, "smape": 7.2}
        }
        (run_dir / "best_validation_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (run_dir / "best_test_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (run_dir / "data_manifest.json").write_text(
            json.dumps({"splits": {"test": {"windows": 1457}}}),
            encoding="utf-8",
        )
        for relative in (
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "best_validation_predictions.pt",
            "best_validation_graphs.pt",
            "best_test_predictions.pt",
            "best_test_graphs.pt",
        ):
            path = run_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")

        frame = collect_selected_modern_tcn_transfer_metrics(
            output_root=root,
            cities=("hongkong",),
            test_years=(2018,),
            horizons=(12,),
        )
    row = frame.iloc[0]
    assert row["status"] == "completed"
    assert bool(row["architecture_match"])
    assert int(row["missing_artifact_count"]) == 0
    assert float(row["test_mae"]) == 1.1
    assert int(row["test_windows"]) == 1457

def main() -> None:
    tests = (
        test_selected_architectures,
        test_plan_has_60_unique_runs,
        test_hongkong_2018_reuses_selected_sweep_path,
        test_city_metric_table_contract,
        test_metric_collector_resolves_default_width_fields,
    )
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")


if __name__ == "__main__":
    main()

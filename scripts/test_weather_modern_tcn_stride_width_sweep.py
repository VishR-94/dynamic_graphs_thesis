from __future__ import annotations

"""Contract tests for the additive weather ModernTCN stride/width sweep.

The tests do not instantiate the external ModernTCN submodule or read the
weather CSV.  They verify the configuration, directory isolation, coupled
width contract and 36-run orchestration independently of GPU availability.
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weather_benchmark.config import (
    MODERN_TCN_PATCH_STRIDE_GRID_BY_HORIZON,
    MODERN_TCN_SELECTED_KERNEL_BY_HORIZON,
    MODERN_TCN_WIDTH_GRID,
    WeatherRunConfig,
)
from src.weather_benchmark.runner import (
    modern_tcn_stride_width_run_suffix,
    run_modern_tcn_stride_width_sweep,
)


class WeatherStrideWidthConfigTests(unittest.TestCase):
    def test_default_config_retains_legacy_signature_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = WeatherRunConfig(
                model_kind="modern_tcn_1st",
                city="hongkong",
                test_year=2018,
                horizon=28,
                data_path=root / "weather_hongkong.csv",
                output_root=root / "weather",
            )
            payload = config.to_dict()

        self.assertNotIn("modern_tcn_patch_stride", payload)
        self.assertNotIn("modern_tcn_d_model", payload)
        self.assertNotIn("modern_tcn_graph_hidden_dim", payload)
        self.assertNotIn("deterministic_runtime", payload)
        self.assertEqual(config.modern_tcn_patch_stride, 4)
        self.assertEqual(config.modern_tcn_d_model, 32)
        self.assertEqual(config.modern_tcn_graph_hidden_dim, 32)

    def test_sweep_config_records_non_default_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suffix = modern_tcn_stride_width_run_suffix(
                kernel=119,
                patch_stride=2,
                d_model=128,
                graph_hidden_dim=128,
            )
            config = WeatherRunConfig(
                model_kind="modern_tcn_1st",
                city="hongkong",
                test_year=2018,
                horizon=120,
                data_path=root / "weather_hongkong.csv",
                output_root=root / "weather",
                modern_tcn_large_kernel=119,
                modern_tcn_patch_stride=2,
                modern_tcn_d_model=128,
                modern_tcn_graph_hidden_dim=128,
                deterministic_runtime=True,
                run_suffix=suffix,
            )
            payload = config.to_dict()

        self.assertEqual(payload["modern_tcn_large_kernel"], 119)
        self.assertEqual(payload["modern_tcn_patch_stride"], 2)
        self.assertEqual(payload["modern_tcn_d_model"], 128)
        self.assertEqual(payload["modern_tcn_graph_hidden_dim"], 128)
        self.assertTrue(payload["deterministic_runtime"])
        self.assertTrue(
            str(config.run_directory).endswith(
                "horizon_120/"
                "test_year_2018_kernel_119_stride_2_"
                "dmodel_128_graphdim_128"
            )
        )

    def test_invalid_stride_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "divisible"):
                WeatherRunConfig(
                    model_kind="modern_tcn_1st",
                    city="hongkong",
                    test_year=2018,
                    horizon=28,
                    data_path=root / "weather_hongkong.csv",
                    output_root=root / "weather",
                    modern_tcn_patch_stride=3,
                )


class WeatherStrideWidthSweepTests(unittest.TestCase):
    def test_default_grid_dispatches_36_isolated_coupled_runs(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_suite(**kwargs):
            calls.append(dict(kwargs))
            horizon = int(kwargs["horizons"][0])
            stride = int(kwargs["modern_tcn_patch_stride"])
            width = int(kwargs["modern_tcn_d_model"])
            score = horizon + stride / 100.0 + width / 10000.0
            return pd.DataFrame(
                [
                    {
                        "model_kind": "modern_tcn_1st",
                        "city": kwargs["city"],
                        "test_year": int(kwargs["test_year"]),
                        "horizon": horizon,
                        "context_length": {4: 28, 12: 28, 28: 56, 120: 240}[
                            horizon
                        ],
                        "modern_tcn_large_kernel": int(
                            kwargs["modern_tcn_large_kernel"]
                        ),
                        "modern_tcn_patch_stride": stride,
                        "modern_tcn_d_model": width,
                        "modern_tcn_graph_hidden_dim": int(
                            kwargs["modern_tcn_graph_hidden_dim"]
                        ),
                        "run_suffix": kwargs["run_suffix"],
                        "train_batch_size": 16,
                        "deterministic_runtime": bool(
                            kwargs["deterministic_runtime"]
                        ),
                        "status": "completed",
                        "best_epoch": 1,
                        "best_validation_score": score,
                        "test_mae": score + 0.1,
                        "test_r": 0.9,
                        "test_smape": 1.0,
                        "run_directory": str(kwargs["run_suffix"]),
                    }
                ]
            )

        with patch(
            "src.weather_benchmark.runner.run_weather_suite",
            side_effect=fake_suite,
        ):
            result = run_modern_tcn_stride_width_sweep(
                city="hongkong",
                test_year=2018,
                horizons=(4, 12, 28, 120),
                data_path=Path("weather_hongkong.csv"),
                output_root=Path("weather"),
                project_root=Path("."),
                train_batch_size=16,
                validation_batch_size=32,
                export_batch_size=32,
                deterministic_runtime=True,
            )

        self.assertEqual(len(calls), 36)
        self.assertEqual(len(result), 36)
        self.assertEqual(
            len(
                {
                    (int(call["horizons"][0]), str(call["run_suffix"]))
                    for call in calls
                }
            ),
            36,
        )
        self.assertTrue(
            all(
                int(call["modern_tcn_d_model"])
                == int(call["modern_tcn_graph_hidden_dim"])
                for call in calls
            )
        )
        self.assertTrue(all(bool(call["deterministic_runtime"]) for call in calls))
        self.assertTrue(all(int(call["train_batch_size"]) == 16 for call in calls))
        self.assertTrue(
            all(int(call["validation_batch_size"]) == 32 for call in calls)
        )
        self.assertTrue(all(int(call["export_batch_size"]) == 32 for call in calls))

        observed: dict[int, set[tuple[int, int, int]]] = {}
        for call in calls:
            horizon = int(call["horizons"][0])
            observed.setdefault(horizon, set()).add(
                (
                    int(call["modern_tcn_large_kernel"]),
                    int(call["modern_tcn_patch_stride"]),
                    int(call["modern_tcn_d_model"]),
                )
            )

        for horizon in (4, 12, 28, 120):
            expected = {
                (
                    int(MODERN_TCN_SELECTED_KERNEL_BY_HORIZON[horizon]),
                    int(stride),
                    int(width),
                )
                for stride in MODERN_TCN_PATCH_STRIDE_GRID_BY_HORIZON[horizon]
                for width in MODERN_TCN_WIDTH_GRID
            }
            self.assertEqual(observed[horizon], expected)
            horizon_rows = result.loc[result["horizon"].eq(horizon)]
            self.assertEqual(
                sorted(horizon_rows["validation_rank_within_horizon"].astype(int)),
                list(range(1, 10)),
            )
            self.assertEqual(
                sorted(horizon_rows["test_mae_rank_within_horizon"].astype(int)),
                list(range(1, 10)),
            )


if __name__ == "__main__":
    unittest.main()

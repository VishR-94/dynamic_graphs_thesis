from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluation.result_analysis import (
    FinancialResultAnalysis,
    make_top_bottom_stock_table,
    plot_adf_pvalues_by_stock,
    plot_daily_error_difference,
    plot_daily_metric_vs_volatility,
    plot_metric_by_stock_volatility,
    plot_metric_vs_adf,
    plot_persistence_headroom,
    plot_split_volatility_distribution,
    plot_stock_metric_by_horizon,
    plot_stock_metric_ecdf,
    plot_time_of_day_metric,
)


ASSETS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
CHANNELS = ["open", "high", "low", "close", "volume", "amount"]
HORIZONS = [1, 5, 15, 30, 60]
CONTEXT = 60
STRIDE = 15


def _make_split(start: str, days: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = []
    dates = pd.bdate_range(start=start, periods=days)
    base = np.linspace(25.0, 150.0, len(ASSETS))
    previous = base.copy()

    for day_idx, day in enumerate(dates):
        market = rng.normal(0.0, 0.00055 + 0.00002 * (day_idx % 5), size=390)
        idiosyncratic = rng.normal(0.0, 0.00035, size=(390, len(ASSETS)))
        returns = market[:, None] + idiosyncratic
        close = previous[None, :] * np.exp(np.cumsum(returns, axis=0))
        previous = close[-1]
        open_ = close * np.exp(rng.normal(0.0, 0.0001, size=close.shape))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0002, close.shape)))
        low = np.minimum(open_, close) / (1.0 + np.abs(rng.normal(0, 0.0002, close.shape)))
        volume = rng.lognormal(mean=11.0, sigma=0.4, size=close.shape)
        amount = volume * close
        x = np.stack([open_, high, low, close, volume, amount], axis=-1)
        timestamps = pd.date_range(
            pd.Timestamp(day.date()) + pd.Timedelta(hours=9, minutes=31),
            periods=390,
            freq="1min",
        )
        samples.append((torch.tensor(x, dtype=torch.float32), timestamps, str(day.date())))

    return {
        "samples": samples,
        "dropped_days": [],
        "asset_cols": list(ASSETS),
        "channels": list(CHANNELS),
        "grain": "1m",
        "market_open": "09:30",
        "market_close": "16:00",
        "fill_method": "none",
        "T": 390,
        "F": len(ASSETS),
        "D": len(CHANNELS),
    }


def _make_prediction_result(test_split: dict, *, quality: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    close_idx = CHANNELS.index("close")
    y_true = []
    last = []
    sample_indices = []
    origin_indices = []
    target_indices = []

    for sample_idx, (x_day, _, _) in enumerate(test_split["samples"]):
        first_origin = CONTEXT - 1
        last_origin = x_day.shape[0] - max(HORIZONS) - 1
        for origin in range(first_origin, last_origin + 1, STRIDE):
            targets = [origin + horizon for horizon in HORIZONS]
            y_true.append(x_day[targets, :, close_idx : close_idx + 1])
            last.append(x_day[origin, :, close_idx : close_idx + 1])
            sample_indices.append(sample_idx)
            origin_indices.append(origin)
            target_indices.append(targets)

    y_true_tensor = torch.stack(y_true).to(torch.float64)
    last_tensor = torch.stack(last).to(torch.float64)
    true_return = torch.log(y_true_tensor) - torch.log(last_tensor[:, None])
    noise = torch.tensor(
        rng.normal(0.0, 0.00035, size=true_return.shape),
        dtype=torch.float64,
    )
    predicted_return = quality * true_return + (1.0 - quality) * noise
    y_pred = last_tensor[:, None] * torch.exp(predicted_return)

    return {
        "y_pred": y_pred,
        "y_true": y_true_tensor,
        "last_context_target": last_tensor,
        "channels": ["close"],
        "horizons": list(HORIZONS),
        "asset_cols": list(ASSETS),
        "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
        "origin_idx": torch.tensor(origin_indices, dtype=torch.long),
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "output_space": "raw",
    }


def _make_persistence(test_split: dict) -> dict:
    result = _make_prediction_result(test_split, quality=1.0, seed=99)
    result["y_pred"] = result["last_context_target"][:, None].expand_as(
        result["y_true"]
    ).clone()
    return result


def main() -> None:
    train = _make_split("2024-01-02", 40, seed=1)
    val = _make_split("2024-03-01", 12, seed=2)
    test = _make_split("2024-04-01", 30, seed=3)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        profiles = pd.DataFrame(
            {
                "ticker": ASSETS,
                "name": ASSETS,
                "c3": range(len(ASSETS)),
                "c4": range(len(ASSETS)),
                "c5": range(len(ASSETS)),
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy", "Energy"],
            }
        )
        profiles_path = root / "company_profiles.csv"
        profiles.to_csv(profiles_path, index=False)

        analysis = FinancialResultAnalysis(
            prediction_results={
                "Persistence": _make_persistence(test),
                "GraphTCN": _make_prediction_result(test, quality=0.8, seed=4),
                "ModernTCN": _make_prediction_result(test, quality=0.6, seed=5),
            },
            train_split=train,
            val_split=val,
            test_split=test,
            company_profiles_path=profiles_path,
            reference_model="Persistence",
        )

        assert analysis.alignment_manifest().shape[0] == 3
        assert analysis.target_timestamps[60].dt.strftime("%H:%M").max() == "16:00"
        assert analysis.target_timestamps[30].dt.strftime("%H:%M").max() == "15:30"
        assert analysis.target_timestamps[1].dt.strftime("%H:%M").max() == "15:01"
        assert analysis.stock_characteristics.shape[0] == len(ASSETS)
        assert set(analysis.stock_characteristics["volatility_tercile"]) == {
            "Low", "Medium", "High"
        }

        stock_metrics = analysis.per_stock_metrics(
            model_names=["Persistence", "GraphTCN"],
            metric_names=[
                "cumulative_log_change_mae",
                "relative_mae_vs_persistence",
                "raw_price_temporal_pearson_correlation",
                "forecast_series_log_return_temporal_pearson_correlation",
            ],
        )
        expected_rows = 2 * 4 * len(HORIZONS) * len(ASSETS)
        assert len(stock_metrics) == expected_rows

        h60_time = analysis.time_of_day_metrics(
            model_names=["Persistence", "GraphTCN"],
            metric_name="cumulative_log_change_mae",
            horizons=60,
        )
        assert set(h60_time["time_bucket"]) == {
            "Morning", "Midday", "Late session"
        }
        assert h60_time.loc[
            h60_time["time_bucket"] == "Late session", "num_windows"
        ].min() > 0

        h60_difference = analysis.time_of_day_metrics(
            model_names=["Persistence", "GraphTCN"],
            metric_name="mae_difference_vs_persistence",
            horizons=60,
        )
        assert "mae_difference_vs_persistence" in analysis.available_group_metrics()
        for bucket in ("Morning", "Midday", "Late session"):
            base_bucket = h60_time.loc[h60_time["time_bucket"] == bucket]
            difference_bucket = h60_difference.loc[
                h60_difference["time_bucket"] == bucket
            ]
            persistence_mae = float(
                base_bucket.loc[
                    base_bucket["model"] == "Persistence", "value"
                ].iloc[0]
            )
            graphtcn_mae = float(
                base_bucket.loc[
                    base_bucket["model"] == "GraphTCN", "value"
                ].iloc[0]
            )
            persistence_difference = float(
                difference_bucket.loc[
                    difference_bucket["model"] == "Persistence", "value"
                ].iloc[0]
            )
            graphtcn_difference = float(
                difference_bucket.loc[
                    difference_bucket["model"] == "GraphTCN", "value"
                ].iloc[0]
            )
            assert np.isclose(persistence_difference, 0.0)
            assert np.isclose(
                graphtcn_difference,
                graphtcn_mae - persistence_mae,
            )

        h1_time = analysis.time_of_day_metrics(
            model_names=["GraphTCN"],
            metric_name="cumulative_log_change_mae",
            horizons=1,
        )
        assert h1_time.loc[
            h1_time["time_bucket"] == "Late session", "num_windows"
        ].iloc[0] == 0

        h30_series_return = analysis.time_of_day_metrics(
            model_names=["GraphTCN"],
            metric_name=(
                "forecast_series_log_return_temporal_pearson_correlation"
            ),
            horizons=30,
        )
        late_series_value = h30_series_return.loc[
            h30_series_return["time_bucket"] == "Late session", "value"
        ].iloc[0]
        assert np.isnan(late_series_value)

        daily = analysis.daily_error_differences(
            model_name="GraphTCN",
            benchmark_name="Persistence",
            horizon=60,
        )
        assert len(daily) == len(test["samples"])
        assert np.isfinite(daily["difference"]).all()

        top_bottom = make_top_bottom_stock_table(
            analysis,
            model_name="GraphTCN",
            metric_name="relative_mae_vs_persistence",
            horizon=60,
            top_k=2,
            num_volatility_buckets=4,
        )
        assert len(top_bottom) == 4
        assert top_bottom["volatility_bucket"].between(1, 4).all()
        assert (top_bottom["num_volatility_buckets"] == 4).all()

        adf_ordered, adf_figure = plot_adf_pvalues_by_stock(
            analysis,
            order_by="volatility",
            num_volatility_buckets=4,
            figsize=(9, 4),
        )
        assert adf_ordered["volatility_bucket"].nunique() == 4
        ordered_volatility = adf_ordered[
            "test_median_realised_volatility"
        ].to_numpy()
        assert np.all(ordered_volatility[:-1] >= ordered_volatility[1:])
        plt.close(adf_figure)

        stock_bar_table, stock_bar_figures = plot_stock_metric_by_horizon(
            analysis,
            model_name="GraphTCN",
            metric_name="relative_mae_vs_persistence",
            horizons=HORIZONS,
            order_by="volatility",
            num_volatility_buckets=4,
            figsize_per_horizon=(9, 4),
        )
        assert set(stock_bar_figures) == set(HORIZONS)
        assert stock_bar_table["volatility_bucket"].nunique() == 4
        for horizon, figure in stock_bar_figures.items():
            axis = figure.axes[0]
            lower, upper = axis.get_ylim()
            finite_values = np.asarray(
                [patch.get_height() for patch in axis.patches],
                dtype=np.float64,
            )
            assert lower > 0.0
            assert upper > np.nanmax(finite_values)
            assert upper - lower < 0.5
            assert any(
                "Adaptive y-axis" in text.get_text()
                for text in axis.texts
            )

            group_mean_lines = [
                line
                for line in axis.lines
                if line.get_color() == "red"
                and line.get_linestyle() == "--"
            ]
            assert len(group_mean_lines) == 4
            expected_group_means = (
                stock_bar_table.loc[
                    stock_bar_table["horizon"] == int(horizon)
                ]
                .groupby("volatility_bucket", sort=True)["value"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            observed_group_means = np.asarray(
                [float(line.get_ydata()[0]) for line in group_mean_lines],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                np.sort(observed_group_means),
                np.sort(expected_group_means),
                rtol=1e-12,
                atol=1e-12,
            )
            plt.close(figure)

        _, zero_based_figures = plot_stock_metric_by_horizon(
            analysis,
            model_name="GraphTCN",
            metric_name="relative_mae_vs_persistence",
            horizons=[1],
            order_by="volatility",
            num_volatility_buckets=4,
            figsize_per_horizon=(9, 4),
            y_axis_mode="zero",
        )
        assert zero_based_figures[1].axes[0].get_ylim()[0] == 0.0
        plt.close(zero_based_figures[1])

        _, price_correlation_figures = plot_stock_metric_by_horizon(
            analysis,
            model_name="GraphTCN",
            metric_name="raw_price_temporal_pearson_correlation",
            horizons=[1],
            order_by="volatility",
            num_volatility_buckets=4,
            figsize_per_horizon=(9, 4),
        )
        price_axis = price_correlation_figures[1].axes[0]
        assert price_axis.get_ylim()[0] > 0.9
        assert price_axis.get_ylim()[1] <= 1.0
        plt.close(price_correlation_figures[1])

        sector_bar_table, sector_bar_figures = plot_stock_metric_by_horizon(
            analysis,
            model_name="GraphTCN",
            metric_name="cumulative_log_change_mae",
            horizons=[1, 60],
            order_by="sector",
            num_volatility_buckets=4,
            figsize_per_horizon=(9, 4),
        )
        assert set(sector_bar_figures) == {1, 60}
        assert sector_bar_table["volatility_bucket"].nunique() == 4
        expected_sector_count = int(sector_bar_table["sector"].nunique())
        for horizon, figure in sector_bar_figures.items():
            axis = figure.axes[0]
            group_mean_lines = [
                line
                for line in axis.lines
                if line.get_color() == "red"
                and line.get_linestyle() == "--"
            ]
            assert len(group_mean_lines) == expected_sector_count
            expected_group_means = (
                sector_bar_table.loc[
                    sector_bar_table["horizon"] == int(horizon)
                ]
                .groupby("sector", sort=True)["value"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            observed_group_means = np.asarray(
                [float(line.get_ydata()[0]) for line in group_mean_lines],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                np.sort(observed_group_means),
                np.sort(expected_group_means),
                rtol=1e-12,
                atol=1e-12,
            )
            plt.close(figure)

        volatility_summary, volatility_figure = plot_metric_by_stock_volatility(
            analysis,
            model_names=["GraphTCN", "ModernTCN"],
            metric_name="relative_mae_vs_persistence",
            horizons=[1, 60],
            num_volatility_buckets=4,
        )
        assert set(volatility_summary["horizon"]) == {1, 60}
        assert set(volatility_summary["volatility_bucket"]) == {1, 2, 3, 4}
        assert "mean_metric" in volatility_summary.columns
        assert "fraction_of_stocks_beating_persistence" in volatility_summary.columns
        assert "fraction_better_than_reference" not in volatility_summary.columns
        assert "median" not in volatility_summary.columns
        assert "q25" not in volatility_summary.columns
        assert "q75" not in volatility_summary.columns
        plt.close(volatility_figure)

        # Daily metric-versus-volatility uses the selected model's own
        # metric. Correlations are not converted into differences versus a
        # benchmark.
        daily_direct = analysis.daily_model_metrics(
            model_names="GraphTCN",
            metric_name="cumulative_log_change_pearson_correlation",
            horizon=60,
        ).sort_values("session_date").reset_index(drop=True)
        daily_scatter, daily_scatter_figure = plot_daily_metric_vs_volatility(
            analysis,
            model_name="GraphTCN",
            horizon=60,
            metric_name="cumulative_log_change_pearson_correlation",
            annotate_extremes=0,
        )
        assert "benchmark" not in daily_scatter.columns
        assert np.allclose(
            daily_scatter["value"].to_numpy(dtype=np.float64),
            daily_direct["value"].to_numpy(dtype=np.float64),
            equal_nan=True,
        )
        assert not daily_scatter["is_persistence_relative"].any()
        plt.close(daily_scatter_figure)

        # The explicit CLG-MAE difference remains available as a metric whose
        # own definition is relative to persistence.
        daily_difference = analysis.daily_error_differences(
            model_name="GraphTCN",
            benchmark_name="Persistence",
            horizon=60,
            metric_name="cumulative_log_change_mae",
        ).sort_values("session_date").reset_index(drop=True)
        difference_scatter, difference_figure = plot_daily_metric_vs_volatility(
            analysis,
            model_name="GraphTCN",
            horizon=60,
            metric_name="mae_difference_vs_persistence",
            annotate_extremes=0,
        )
        assert np.allclose(
            difference_scatter["value"].to_numpy(dtype=np.float64),
            daily_difference["difference"].to_numpy(dtype=np.float64),
            equal_nan=True,
        )
        assert difference_scatter["is_persistence_relative"].all()
        assert set(difference_scatter["reference_model"]) == {"Persistence"}
        plt.close(difference_figure)

        plot_functions = [
            lambda: plot_split_volatility_distribution(analysis),
            lambda: plot_metric_vs_adf(
                analysis,
                model_name="GraphTCN",
                metric_name="relative_mae_vs_persistence",
                horizon=60,
            ),
            lambda: plot_persistence_headroom(
                analysis,
                model_name="GraphTCN",
                metric_name="relative_mae_vs_persistence",
                horizon=60,
            ),
            lambda: plot_stock_metric_ecdf(
                analysis,
                model_names=["GraphTCN", "ModernTCN"],
                metric_name="relative_mae_vs_persistence",
                horizon=60,
            ),
            lambda: plot_time_of_day_metric(
                analysis,
                model_names=["Persistence", "GraphTCN"],
                metric_name="mae_difference_vs_persistence",
                horizons=[30, 60],
            ),
            lambda: plot_daily_error_difference(
                analysis,
                model_name="GraphTCN",
                benchmark_name="Persistence",
                horizon=60,
            ),
            lambda: plot_daily_metric_vs_volatility(
                analysis,
                model_name="GraphTCN",
                horizon=60,
                metric_name="cumulative_log_change_pearson_correlation",
            ),
        ]
        for function in plot_functions:
            _, figure = function()
            assert figure is not None
            plt.close(figure)

        exported = analysis.export_tables(root / "analysis_output")
        assert all(path.is_file() for path in exported.values())

        broken = _make_prediction_result(test, quality=0.7, seed=6)
        broken["origin_idx"] = broken["origin_idx"].clone()
        broken["origin_idx"][0] += 1
        try:
            FinancialResultAnalysis(
                prediction_results={
                    "Persistence": _make_persistence(test),
                    "Broken": broken,
                },
                train_split=train,
                val_split=val,
                test_split=test,
                company_profiles_path=profiles_path,
                reference_model="Persistence",
            )
        except ValueError as exc:
            assert "origin_idx" in str(exc)
        else:
            raise AssertionError("Misaligned predictions were not rejected.")

    print("Result-analysis synthetic contract test passed.")


if __name__ == "__main__":
    main()

"""CPU contracts for the unified final-model analysis helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

from src.evaluation.dynamic_graph_evaluation import (
    detect_run_kind,
    load_analysis_artifacts,
    load_sampled_path_bundle,
    make_comparative_metrics_table,
    make_graph_snapshot_connections_table,
    make_graph_snapshot_summary_table,
    make_graph_window_table,
    make_predictive_coverage_by_asset_table,
    make_predictive_coverage_table,
    make_probabilistic_score_table,
    make_run_overview_table,
    make_unified_model_summary_table,
    make_sample_rank_histogram_table,
    make_temperature_sweep_table,
    plot_coverage_calibration,
    plot_graph_snapshot,
    plot_point_forecast_window,
    plot_sampled_price_paths,
    select_graph_snapshot,
)
from src.utils.metric_tables import DEFAULT_SUMMARY_METRICS


HORIZONS = (1, 5, 15, 30, 60)
ASSETS = ("AAA", "BBB", "CCC")
DATES = ("2024-09-03", "2024-09-03", "2024-09-04", "2024-09-04")


def _save_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _metric_table() -> pd.DataFrame:
    rows = []
    for metric_index, metric in enumerate(DEFAULT_SUMMARY_METRICS):
        for horizon_index, horizon in enumerate(HORIZONS):
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "channel": "close",
                    "value": 0.001 + 1.0e-5 * metric_index + 1.0e-6 * horizon_index,
                }
            )
    return pd.DataFrame(rows)


def _prediction_result() -> dict:
    windows = len(DATES)
    last = torch.tensor(
        [[[100.0], [50.0], [25.0]]] * windows,
        dtype=torch.float32,
    )
    changes = torch.linspace(0.0001, 0.002, len(HORIZONS)).reshape(1, -1, 1, 1)
    true = last.unsqueeze(1) * torch.exp(changes)
    true = true.expand(windows, -1, len(ASSETS), -1).contiguous()
    pred = true * 1.0001
    return {
        "y_pred": pred,
        "y_true": true,
        "last_context_target": last,
        "channels": ["close"],
        "horizons": list(HORIZONS),
        "asset_cols": list(ASSETS),
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "origin_idx": torch.tensor([59, 74, 59, 74]),
        "target_indices": torch.tensor(
            [[60, 64, 74, 89, 119]] * windows
        ),
        "output_space": "raw",
    }


def _graphs() -> dict:
    windows = len(DATES)
    base = torch.tensor(
        [
            [0.0, 0.7, 0.3],
            [0.2, 0.0, 0.8],
            [0.6, 0.4, 0.0],
        ],
        dtype=torch.float32,
    )
    values = torch.stack(
        [
            base,
            torch.tensor(
                [[0.0, 0.6, 0.4], [0.3, 0.0, 0.7], [0.5, 0.5, 0.0]]
            ),
            base,
            torch.tensor(
                [[0.0, 0.8, 0.2], [0.1, 0.0, 0.9], [0.7, 0.3, 0.0]]
            ),
        ],
        dim=0,
    ).unsqueeze(1)
    return {
        "graph_type": "dynamic",
        "graph_orientation": "A[target, source]",
        "asset_cols": list(ASSETS),
        "selected": values,
        "base": None,
        "dynamic": values,
        "per_layer": (values,),
        "alpha": None,
        "spatial_beta": torch.full((windows, 1), 0.48),
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "origin_idx": torch.tensor([59, 74, 59, 74]),
        "target_indices": torch.tensor([[60, 61, 62, 63, 64]] * windows),
        "dates": list(DATES),
    }


def _write_continuous_run(root: Path) -> Path:
    run = root / "continuous"
    run.mkdir()
    _save_json(
        run / "resolved_config.json",
        {
            "data": {"horizons": list(HORIZONS), "target_channel": "close"},
            "model": {
                "output_representation": "normalised_close",
                "temporal": {"type": "modern_tcn", "d_model": 32},
                "graph": {
                    "type": "dynamic",
                    "num_heads": 1,
                    "hidden_dim": 32,
                    "add_self_loops": False,
                },
                "spatial": {"gate_type": "learned_scalar", "initial_beta": 0.5},
            },
            "training": {"selection_metric": "validation_loss"},
        },
    )
    _save_json(
        run / "run_metadata.json",
        {"status": "completed", "asset_cols": list(ASSETS), "best_epoch": 3},
    )
    pd.DataFrame({"epoch": [1, 2, 3]}).to_csv(run / "history.csv", index=False)
    torch.save(_prediction_result(), run / "best_validation_predictions.pt")
    continuous_graph = dict(_graphs())
    continuous_graph["orientation"] = continuous_graph.pop("graph_orientation")
    continuous_graph["spatial_beta"] = 0.48
    continuous_graph["dynamic_alpha"] = None
    torch.save(continuous_graph, run / "best_validation_graphs.pt")
    _metric_table().to_csv(run / "best_validation_metric_table.csv", index=False)
    return run


def _write_token_run(root: Path) -> Path:
    run = root / "token"
    run.mkdir()
    _save_json(
        run / "resolved_config.json",
        {
            "models": {
                "dynamic_graph": {
                    "num_nodes": len(ASSETS),
                    "num_st_blocks": 1,
                    "d_model": 32,
                    "temporal": {"type": "modern_tcn"},
                    "graph": {
                        "type": "dynamic",
                        "num_heads": 1,
                        "hidden_dim": 32,
                        "add_self_loops": False,
                    },
                    "spatial": {"gate_type": "learned_scalar", "initial_beta": 0.5},
                    "heads": {
                        "evaluation_horizons": list(HORIZONS),
                        "prediction_length": 60,
                    },
                }
            },
            "training": {"early_stopping_metric": "validation_token_loss"},
        },
    )
    _save_json(
        run / "run_metadata.json",
        {"status": "completed", "asset_cols": list(ASSETS), "best_epoch": 4},
    )
    pd.DataFrame({"epoch": [1, 2, 3, 4]}).to_csv(run / "history.csv", index=False)

    temperature_root = run / "temperature_sweep"
    policy = temperature_root / "temperature_0p6"
    policy.mkdir(parents=True)
    _save_json(
        temperature_root / "temperature_selection.json",
        {"selected_policy": "temperature_0p6", "selected_temperature": 0.6},
    )
    pd.DataFrame(
        [
            {"Policy": "argmax", "Temperature": np.nan, "Mean Log MAE": 0.002},
            {"Policy": "temperature_0p6", "Temperature": 0.6, "Mean Log MAE": 0.001},
        ]
    ).to_csv(temperature_root / "temperature_sweep_results.csv", index=False)

    prediction = _prediction_result()
    torch.save(
        {"epoch": 4, "prediction_result": prediction},
        policy / "validation_predictions.pt",
    )
    torch.save(
        {"epoch": 4, "graph_artifacts": _graphs()},
        policy / "validation_graphs.pt",
    )
    _metric_table().to_csv(policy / "validation_metric_table.csv", index=False)

    samples = 10
    windows = len(DATES)
    steps = 60
    base = prediction["last_context_target"].unsqueeze(0).unsqueeze(2)
    minute_returns = torch.linspace(0.0001, 0.003, steps).reshape(1, 1, steps, 1, 1)
    offsets = torch.linspace(-0.0004, 0.0004, samples).reshape(samples, 1, 1, 1, 1)
    sampled_paths = base * torch.exp(minute_returns + offsets)
    sampled_paths = sampled_paths.expand(-1, windows, -1, len(ASSETS), -1).contiguous()
    eval_indices = torch.tensor([0, 4, 14, 29, 59])
    sampled_eval = sampled_paths.index_select(2, eval_indices)
    artifacts = {
        "sampled_close_paths": sampled_paths,
        "sampled_close_paths_at_evaluation_horizons": sampled_eval,
        "ensemble_mean_close_path": sampled_paths.mean(dim=0),
        "evaluation_true": prediction["y_true"],
        "last_context_target": prediction["last_context_target"],
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "origin_idx": torch.tensor([59, 74, 59, 74]),
        "dense_target_indices": torch.tensor([list(range(60, 120))] * windows),
        "evaluation_target_indices": prediction["target_indices"],
        "dates": list(DATES),
        "asset_cols": list(ASSETS),
        "future_steps": list(range(1, 61)),
        "evaluation_horizons": list(HORIZONS),
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 0.9,
        "sample_count": samples,
    }
    torch.save(
        {"epoch": 4, "sampled_price_path_artifacts": artifacts},
        policy / "validation_sampled_price_paths.pt",
    )
    return run


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        continuous = _write_continuous_run(root)
        token = _write_token_run(root)

        assert detect_run_kind(continuous) == "continuous"
        assert detect_run_kind(token) == "token"
        continuous_artifacts = load_analysis_artifacts(continuous)
        token_artifacts = load_analysis_artifacts(token, policy="selected_temperature")
        assert continuous_artifacts.policy == "best"
        assert token_artifacts.policy == "temperature_0p6"

        continuous_summary = make_unified_model_summary_table(continuous)
        token_summary = make_unified_model_summary_table(token)
        assert continuous_summary.loc["Run family", "Value"] == "continuous"
        assert token_summary.loc["Run family", "Value"] == "token"
        overview = make_run_overview_table(
            {"Continuous": continuous, "Token": token},
            policies={"Token": "selected_temperature"},
        )
        assert len(overview) == 2
        metrics = make_comparative_metrics_table(
            {"Continuous": continuous, "Token": token},
            policies={"Token": "selected_temperature"},
        )
        assert len(metrics) == 10

        windows = make_graph_window_table(token, policy="selected_temperature")
        assert windows["Window within date"].tolist() == [1, 2, 1, 2]
        snapshot = select_graph_snapshot(
            token,
            policy="selected_temperature",
            date="2024-09-03",
            window_within_date=2,
        )
        assert snapshot.global_window_index == 1
        summary = make_graph_snapshot_summary_table(snapshot)
        assert float(summary.loc[0, "Mean effective neighbours"]) > 1.0
        connections = make_graph_snapshot_connections_table(snapshot, top_n=2)
        assert len(connections) == len(ASSETS)
        figure, _, plotted = plot_graph_snapshot(snapshot, cluster=False)
        assert plotted.shape == (len(ASSETS), len(ASSETS))
        figure.clf()

        bundle = load_sampled_path_bundle(token, policy="selected_temperature")
        assert bundle.sample_count == 10
        coverage = make_predictive_coverage_table(
            token,
            policy="selected_temperature",
        )
        assert set(coverage["Horizon"]) == set(HORIZONS)
        assert coverage["Empirical coverage"].between(0.0, 1.0).all()
        asset_coverage = make_predictive_coverage_by_asset_table(
            token,
            policy="selected_temperature",
            nominal_coverage=0.8,
        )
        assert len(asset_coverage) == len(ASSETS) * len(HORIZONS)
        rank_table = make_sample_rank_histogram_table(
            token,
            policy="selected_temperature",
        )
        assert len(rank_table) == len(HORIZONS) * 11
        scores = make_probabilistic_score_table(
            token,
            policy="selected_temperature",
        )
        assert (scores["Empirical CRPS (log-return bps)"] >= 0.0).all()
        calibration_figure, _ = plot_coverage_calibration(coverage)
        calibration_figure.clf()
        temperature_table = make_temperature_sweep_table(token)
        assert temperature_table["Selected"].sum() == 1

        figure, _, path_table = plot_sampled_price_paths(
            token,
            policy="selected_temperature",
            asset="AAA",
            date="2024-09-03",
            window_within_date=1,
        )
        assert path_table.shape[0] == 60
        figure.clf()
        figure, _, point_table = plot_point_forecast_window(
            continuous,
            asset="AAA",
            date="2024-09-03",
            window_within_date=1,
        )
        assert len(point_table) == 5
        figure.clf()

    print("Unified dynamic-graph evaluation contracts passed.")


if __name__ == "__main__":
    main()

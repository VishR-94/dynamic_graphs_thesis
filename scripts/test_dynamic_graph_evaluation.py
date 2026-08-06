"""CPU contracts for the final models-to-use Graph Hub analysis API."""

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
    analyse_graph,
    analyse_graph_entropy,
    analyse_graph_window,
    analyse_sector_graph,
    analyse_training_history,
    discover_models,
    load_evaluation_artifacts,
    load_model_sampled_path_bundle,
    make_model_architecture_comparison,
    make_model_artifact_audit,
    make_model_metric_comparison,
    make_predictive_coverage_report,
    make_evaluation_window_table,
    plot_point_forecast_comparison,
    plot_point_forecast_example,
    plot_training_diagnostics,
    resolve_evaluation_artifact_paths,
    resolve_model_folder,
    resolve_models_to_use_root,
    style_model_artifact_audit,
    style_model_metric_comparison,
    style_numeric_table,
)
from src.utils.metric_tables import DEFAULT_SUMMARY_METRICS


HORIZONS = (1, 5, 15, 30, 60)
ASSETS = ("AAA", "BBB", "CCC")
DATES = ("2024-09-03", "2024-09-03", "2024-09-04", "2024-09-04")


def _save_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _metric_table(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for metric_index, metric in enumerate(DEFAULT_SUMMARY_METRICS):
        for horizon_index, horizon in enumerate(HORIZONS):
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "channel": "close",
                    "value": (
                        offset
                        + 0.001
                        + 1.0e-5 * metric_index
                        + 1.0e-6 * horizon_index
                    ),
                }
            )
    return pd.DataFrame(rows)


def _prediction_result(scale: float = 1.0001) -> dict:
    windows = len(DATES)
    last = torch.tensor(
        [[[100.0], [50.0], [25.0]]] * windows,
        dtype=torch.float32,
    )
    changes = torch.linspace(0.0001, 0.002, len(HORIZONS)).reshape(1, -1, 1, 1)
    true = last.unsqueeze(1) * torch.exp(changes)
    true = true.expand(windows, -1, len(ASSETS), -1).contiguous()
    pred = true * scale
    return {
        "y_pred": pred,
        "y_true": true,
        "last_context_target": last,
        "channels": ["close"],
        "horizons": list(HORIZONS),
        "asset_cols": list(ASSETS),
        "sample_idx": torch.tensor([0, 0, 1, 1]),
        "origin_idx": torch.tensor([59, 74, 59, 74]),
        "target_indices": torch.tensor([[60, 64, 74, 89, 119]] * windows),
        "output_space": "raw",
    }


def _graph_values(dynamic: bool = True) -> torch.Tensor:
    base = torch.tensor(
        [
            [0.0, 0.7, 0.3],
            [0.2, 0.0, 0.8],
            [0.6, 0.4, 0.0],
        ],
        dtype=torch.float32,
    )
    if not dynamic:
        return base.unsqueeze(0).unsqueeze(0).expand(len(DATES), 1, -1, -1).clone()
    return torch.stack(
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


def _graphs(graph_type: str, orientation: str = "A[target, source]") -> dict:
    dynamic = graph_type == "dynamic"
    values = _graph_values(dynamic=dynamic)
    return {
        "graph_type": graph_type,
        "graph_orientation": orientation,
        "asset_cols": list(ASSETS),
        "selected": values,
        "base": None,
        "dynamic": values if dynamic else None,
        "per_layer": (values,),
        "alpha": None,
        "spatial_beta": torch.full((len(DATES), 1), 0.48),
        "dates": list(DATES),
    }


def _continuous_config(graph_type: str) -> dict:
    return {
        "data": {
            "horizons": list(HORIZONS),
            "target_channel": "close",
            "context_length": 60,
            "input_representation": "raw",
        },
        "model": {
            "output_representation": "normalised_close",
            "temporal": {
                "type": "modern_tcn",
                "d_model": 32,
                "modern_tcn": {
                    "num_blocks": 1,
                    "patch_size": 8,
                    "patch_stride": 4,
                    "large_kernel": 15,
                    "small_kernel": 5,
                    "ffn_ratio": 1,
                },
            },
            "graph": {
                "type": graph_type,
                "num_heads": 1,
                "hidden_dim": 32,
                "add_self_loops": False,
            },
            "spatial": {
                "num_layers": 1,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
            },
        },
        "training": {
            "selection_metric": "validation_loss",
            "learning_rate": 1.0e-4,
            "graph_learning_rate": 5.0e-4,
            "loss": {"type": "cumulative_log_change_mae"},
        },
    }


def _write_history(run: Path, token: bool = False) -> None:
    if token:
        frame = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4],
                "train_token_loss": [5.0, 4.8, 4.7, 4.6],
                "validation_token_loss": [4.9, 4.7, 4.6, 4.55],
                "validation_graph_mean_row_entropy": [3.0, 2.9, 2.8, 2.7],
                "spatial_beta": [0.50, 0.49, 0.48, 0.47],
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "training_loss": [0.002, 0.0018, 0.0017],
                "selection_score": [0.0019, 0.0017, 0.0016],
                "graph_mean_row_entropy": [3.2, 3.1, 3.0],
                "spatial_beta": [0.50, 0.49, 0.48],
            }
        )
    frame.to_csv(run / "history.csv", index=False)


def _write_continuous_run(root: Path, name: str, graph_type: str, offset: float) -> Path:
    run = root / name
    run.mkdir(parents=True)
    _save_json(run / "resolved_config.json", _continuous_config(graph_type))
    _save_json(
        run / "run_metadata.json",
        {
            "status": "completed",
            "asset_cols": list(ASSETS),
            "best_epoch": 3,
            "trainable_parameter_count": 1234,
            "project_git_commit": "abc",
        },
    )
    _write_history(run)
    prediction = _prediction_result(scale=1.0001 + offset)
    torch.save(prediction, run / "best_validation_predictions.pt")
    graph = _graphs(graph_type)
    # Reproduce the historical continuous schema: dates and orientation are in
    # the graph file, while origin/sample indices live in predictions.
    graph["orientation"] = graph.pop("graph_orientation")
    graph.pop("sample_idx", None)
    graph.pop("origin_idx", None)
    torch.save(graph, run / "best_validation_graphs.pt")
    _metric_table(offset).to_csv(run / "best_validation_metric_table.csv", index=False)

    # Canonical final-checkpoint train bundle.
    train_dir = run / "analysis" / "train"
    train_dir.mkdir(parents=True)
    torch.save({"epoch": 3, "prediction_result": prediction}, train_dir / "predictions.pt")
    torch.save({"epoch": 3, "graph_artifacts": graph}, train_dir / "graphs.pt")
    _metric_table(offset + 5.0e-5).to_csv(train_dir / "metric_table.csv", index=False)

    # Canonical frozen test bundle.
    test_dir = run / "analysis" / "test"
    test_dir.mkdir(parents=True)
    torch.save(prediction, test_dir / "predictions.pt")
    torch.save(graph, test_dir / "graphs.pt")
    _metric_table(offset + 1.0e-4).to_csv(test_dir / "metric_table.csv", index=False)
    return run


def _token_config() -> dict:
    return {
        "models": {
            "dynamic_graph": {
                "num_nodes": len(ASSETS),
                "num_st_blocks": 1,
                "d_model": 32,
                "context_length": 60,
                "token_input_representation": "bsq_bits",
                "temporal": {
                    "type": "modern_tcn",
                    "modern_tcn": {
                        "num_blocks": 1,
                        "patch_size": 8,
                        "patch_stride": 4,
                        "large_kernel": 15,
                        "small_kernel": 5,
                        "ffn_ratio": 1,
                    },
                },
                "graph": {
                    "type": "dynamic",
                    "num_heads": 1,
                    "hidden_dim": 32,
                    "add_self_loops": False,
                },
                "spatial": {
                    "num_layers": 1,
                    "gate_type": "learned_scalar",
                    "initial_beta": 0.5,
                },
                "heads": {
                    "evaluation_horizons": list(HORIZONS),
                    "prediction_length": 60,
                    "future_token_mode": "coarse_only",
                },
                "future_predictor": {"type": "structured_parallel"},
            }
        },
        "training": {
            "early_stopping_metric": "validation_token_loss",
            "learning_rate": 1.0e-4,
            "graph_learning_rate": 5.0e-4,
        },
    }


def _sampled_artifacts(prediction: dict) -> dict:
    samples = 10
    windows = len(DATES)
    steps = 60
    base = prediction["last_context_target"].unsqueeze(0).unsqueeze(2)
    minute_returns = torch.linspace(0.0001, 0.003, steps).reshape(1, 1, steps, 1, 1)
    offsets = torch.linspace(-0.0004, 0.0004, samples).reshape(samples, 1, 1, 1, 1)
    paths = base * torch.exp(minute_returns + offsets)
    paths = paths.expand(-1, windows, -1, len(ASSETS), -1).contiguous()
    eval_indices = torch.tensor([0, 4, 14, 29, 59])
    return {
        "sampled_close_paths": paths,
        "sampled_close_paths_at_evaluation_horizons": paths.index_select(2, eval_indices),
        "ensemble_mean_close_path": paths.mean(dim=0),
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


def _write_token_run(root: Path) -> Path:
    run = root / "tokenized_dynamic"
    run.mkdir(parents=True)
    _save_json(run / "resolved_config.json", _token_config())
    _save_json(
        run / "run_metadata.json",
        {
            "status": "completed",
            "asset_cols": list(ASSETS),
            "best_epoch": 4,
            "trainable_parameter_count": 2345,
            "project_git_commit": "def",
        },
    )
    _write_history(run, token=True)
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
    torch.save({"epoch": 4, "prediction_result": prediction}, policy / "validation_predictions.pt")
    # The production token runner historically wrote this spelling.
    token_graph = _graphs("dynamic", orientation="row=target,column=source")
    token_graph["sample_idx"] = torch.tensor([0, 0, 1, 1])
    token_graph["origin_idx"] = torch.tensor([59, 74, 59, 74])
    torch.save({"epoch": 4, "graph_artifacts": token_graph}, policy / "validation_graphs.pt")
    _metric_table().to_csv(policy / "validation_metric_table.csv", index=False)
    sampled = _sampled_artifacts(prediction)
    torch.save(
        {"epoch": 4, "sampled_price_path_artifacts": sampled},
        policy / "validation_sampled_price_paths.pt",
    )

    train_dir = run / "analysis" / "train"
    train_dir.mkdir(parents=True)
    torch.save({"epoch": 4, "prediction_result": prediction}, train_dir / "predictions.pt")
    torch.save({"epoch": 4, "graph_artifacts": token_graph}, train_dir / "graphs.pt")
    _metric_table(1.5e-4).to_csv(train_dir / "metric_table.csv", index=False)

    # Canonical frozen test bundle: no policy-specific folder required.
    test_dir = run / "analysis" / "test"
    test_dir.mkdir(parents=True)
    torch.save({"epoch": 4, "prediction_result": prediction}, test_dir / "predictions.pt")
    torch.save({"epoch": 4, "graph_artifacts": token_graph}, test_dir / "graphs.pt")
    _metric_table(2.0e-4).to_csv(test_dir / "metric_table.csv", index=False)
    torch.save(
        {"epoch": 4, "sampled_price_path_artifacts": sampled},
        test_dir / "sampled_price_paths.pt",
    )
    return run


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "models_to_use"
        root.mkdir()
        dynamic = _write_continuous_run(root, "continuous_dynamic", "dynamic", 0.0)
        static = _write_continuous_run(root, "continuous_static", "free_static", 1.0e-5)
        correlation = _write_continuous_run(root, "continuous_correlation", "fixed", 2.0e-5)
        token = _write_token_run(root)
        arbitrary = _write_continuous_run(
            root, "weighted_loss_dynamic_experiment", "dynamic", 3.0e-5
        )

        profiles_path = Path(temporary) / "company_profiles.csv"
        pd.DataFrame(
            {
                "Ticker": ASSETS,
                "Company Name": ASSETS,
                "Country": ["US"] * len(ASSETS),
                "State": ["CA"] * len(ASSETS),
                "Exchange": ["X"] * len(ASSETS),
                "Sector": ["Technology", "Financial Services", "Technology"],
            }
        ).to_csv(profiles_path, index=False)

        assert resolve_models_to_use_root(root) == root.resolve()
        discovered = discover_models(models_root=root)
        assert set(discovered["Folder"]) == {
            "continuous_dynamic",
            "continuous_static",
            "continuous_correlation",
            "tokenized_dynamic",
            "weighted_loss_dynamic_experiment",
        }
        assert resolve_model_folder("continuous_dynamic", models_root=root) == dynamic.resolve()

        token_paths = resolve_evaluation_artifact_paths(
            token,
            split="validation",
            policy="auto",
        )
        assert token_paths.policy == "temperature_0p6"
        test_paths = resolve_evaluation_artifact_paths(
            token,
            split="test",
            policy="auto",
        )
        assert test_paths.predictions.name == "predictions.pt"
        train_paths = resolve_evaluation_artifact_paths(
            dynamic,
            split="train",
            policy=None,
        )
        assert train_paths.predictions.name == "predictions.pt"
        val_alias_paths = resolve_evaluation_artifact_paths(
            dynamic,
            split="val",
            policy=None,
        )
        assert val_alias_paths.predictions.name == "best_validation_predictions.pt"

        validation = load_evaluation_artifacts(
            token,
            split="validation",
            policy="auto",
            require_graph=True,
            require_metrics=True,
            require_sampled_paths=True,
        )
        assert validation.graph_artifacts is not None
        assert validation.graph_artifacts["graph_orientation"] == "A[target, source]"
        assert tuple(validation.graph_artifacts["selected"].shape) == (4, 1, 3, 3)
        test = load_evaluation_artifacts(
            token,
            split="test",
            policy="auto",
            require_sampled_paths=True,
        )
        assert test.split == "test"

        models = {
            "Continuous dynamic": dynamic,
            "Continuous static": static,
            "Continuous correlation": correlation,
            "Tokenized dynamic": token,
        }
        policies = {"Tokenized dynamic": "auto"}
        audit = make_model_artifact_audit(models, split="validation", policies=policies)
        assert audit["Ready"].all()
        assert int(audit.loc[audit["Model"] == "Tokenized dynamic", "Sample count"].iloc[0]) == 10
        style_model_artifact_audit(audit)._repr_html_()

        architecture = make_model_architecture_comparison(models)
        assert set(architecture.columns) == set(models)
        metrics = make_model_metric_comparison(models, split="validation", policies=policies)
        assert len(metrics) == len(models) * len(HORIZONS)
        style_model_metric_comparison(metrics, caption="test")._repr_html_()
        style_numeric_table(
            pd.DataFrame({"Text": ["x"], "Value": [1.2345]}),
            caption="mixed",
        )._repr_html_()

        window_table = make_evaluation_window_table(dynamic, split="validation")
        first = window_table.iloc[0]
        assert first["Window within date"] == 1
        assert first["Forecast origin time"] == "10:30"
        assert first["Time window"] == "context 09:31–10:30; forecast 10:31–11:30"

        graph_report = analyse_graph_window(
            dynamic,
            split="validation",
            date="2024-09-03",
            window_within_date=1,
            top_n=2,
            cluster=False,
        )
        assert graph_report.snapshot.global_window_index == 0
        assert "10:30" in graph_report.axes.get_title()
        assert len(graph_report.connections) == len(ASSETS)
        graph_report.figure.clf()

        all_graph = analyse_graph(
            dynamic,
            split="train",
            day=None,
            window=None,
            top_n=2,
            cluster=False,
        )
        assert len(all_graph.graph.selected_windows) == len(DATES)
        assert all_graph.graph.selection_description.startswith("all 4 saved windows")
        assert int(
            all_graph.top_source_frequency[
                "Targets for which source is in top 2"
            ].max()
        ) <= len(ASSETS) - 1
        all_graph.adjacency_figure.clf()
        all_graph.frequency_figure.clf()

        day_graph = analyse_graph(
            dynamic, split="validation", day="2024-09-03", window=None, cluster=False
        )
        assert len(day_graph.graph.selected_windows) == 2
        day_graph.adjacency_figure.clf()
        day_graph.frequency_figure.clf()

        window_graph = analyse_graph(
            dynamic, split="validation", day=None, window=2, cluster=False
        )
        assert len(window_graph.graph.selected_windows) == 2
        window_graph.adjacency_figure.clf()
        window_graph.frequency_figure.clf()

        random_graph = analyse_graph(
            dynamic, split="validation", day="random", window="random", random_seed=7
        )
        assert len(random_graph.graph.selected_windows) == 1
        random_graph.adjacency_figure.clf()
        random_graph.frequency_figure.clf()

        entropy_report = analyse_graph_entropy(dynamic, split="train")
        assert len(entropy_report.day_values) == 2
        assert len(entropy_report.asset_summary) == len(ASSETS)
        assert entropy_report.day_summary["Highest-entropy day"].notna().all()
        entropy_report.figure.clf()

        sector_report = analyse_sector_graph(
            dynamic,
            split="validation",
            day=None,
            window=None,
            company_profiles_path=profiles_path,
        )
        assert sector_report.sector_adjacency.shape == (2, 2)
        assert sector_report.asset_sector_adjacency.shape == (3, 2)
        np.testing.assert_allclose(
            sector_report.sector_adjacency.sum(axis=1).to_numpy(), 1.0
        )
        sector_report.sector_figure.clf()
        sector_report.asset_sector_figure.clf()

        forecast_report = plot_point_forecast_comparison(
            models,
            split="validation",
            policies=policies,
            asset="AAA",
            date="2024-09-03",
            window_within_date=1,
        )
        assert len(forecast_report.values) == len(models) * len(HORIZONS)
        assert "10:30" in forecast_report.axes.get_title()
        forecast_report.figure.clf()
        random_forecast = plot_point_forecast_example(
            models,
            split="validation",
            policies=policies,
            asset="random",
            day="random",
            window="random",
            random_seed=9,
        )
        assert len(random_forecast.values) == len(models) * len(HORIZONS)
        random_forecast.figure.clf()

        bundle = load_model_sampled_path_bundle(token, split="validation", policy="auto")
        assert bundle.sample_count == 10
        coverage = make_predictive_coverage_report(
            token,
            split="validation",
            policy="auto",
            nominal_coverages=(0.5, 0.8, 0.9),
            asset_coverage=0.8,
        )
        assert coverage.overall["Empirical coverage"].between(0.0, 1.0).all()
        assert len(coverage.by_asset) == len(ASSETS) * len(HORIZONS)
        assert len(coverage.rank_histogram) == len(HORIZONS) * 11
        coverage.calibration_figure.clf()

        history_figure, _, history = plot_training_diagnostics(token)
        assert "validation_token_loss" in history.columns
        history_figure.clf()
        training_report = analyse_training_history(token)
        assert training_report.entropy_figure is not None
        assert training_report.beta_figure is not None
        assert training_report.entropy_axes is not training_report.beta_axes
        training_report.objective_figure.clf()
        training_report.entropy_figure.clf()
        training_report.beta_figure.clf()

        # A reversed orientation remains a hard error; it is never silently
        # transposed by analysis code.
        bad_graph = _graphs("dynamic", orientation="row=source,column=target")
        bad_dir = token / "analysis" / "validation"
        bad_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 4, "prediction_result": _prediction_result()}, bad_dir / "predictions.pt")
        torch.save({"epoch": 4, "graph_artifacts": bad_graph}, bad_dir / "graphs.pt")
        try:
            load_evaluation_artifacts(token, split="validation", policy="default", require_graph=True)
        except ValueError as error:
            assert "A[source, target]" in str(error)
        else:
            raise AssertionError("Reversed graph orientation was not rejected.")

    print("Final Graph Hub evaluation contracts passed.")


if __name__ == "__main__":
    main()

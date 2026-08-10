from __future__ import annotations

"""Focused contracts for the final continuous completion/sparsemax work."""

import json
from pathlib import Path
import tempfile

import pandas as pd
import torch
import yaml

from src.evaluation.dynamic_graph_evaluation import (
    load_evaluation_artifacts,
    select_graph,
)
from src.models.continuous_forecaster import (
    ContinuousForecaster,
)
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Model,
    PriorMixedDynamicGraphLearner,
    round1_model_config_from_mapping,
)
from src.training import run_continuous_forecaster as continuous_runner
from src.training.final_continuous_artifact_completion import (
    canonicalise_single_layer_graph_payload,
    complete_continuous_run_artifacts,
)
from src.training.final_sparsemax_graph_specs import (
    make_final_sparsemax_graph_specs,
)
from src.training.modern_tcn_final_two_runs_specs import (
    make_final_two_run_specs,
)


def _assert_stochastic(values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise AssertionError("Graph contains non-finite values.")
    if (values < 0).any():
        raise AssertionError("Graph contains negative weights.")
    torch.testing.assert_close(
        values.sum(dim=-1),
        torch.ones_like(values.sum(dim=-1)),
        atol=1.0e-6,
        rtol=0.0,
    )
    diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    torch.testing.assert_close(
        diagonal,
        torch.zeros_like(diagonal),
        atol=0.0,
        rtol=0.0,
    )


def _test_specs() -> None:
    specs = make_final_sparsemax_graph_specs()
    if len(specs) != 2 or len({spec.run_name for spec in specs}) != 2:
        raise AssertionError("Expected two unique sparsemax specifications.")
    dynamic, random_static = specs
    if dynamic.variant != "dynamic_only_state":
        raise AssertionError("Run 1 is not the state-aware dynamic control.")
    if random_static.variant != "random_static_mixture_state":
        raise AssertionError("Run 2 is not the random-static state control.")
    for spec in specs:
        config = spec.config
        if config["model"]["graph"]["activation"] != "sparsemax":
            raise AssertionError("Sparsemax was not applied.")
        if config["model"]["graph_regularisation"]["graph_entropy_reg"] != 0.0:
            raise AssertionError("Unexpected graph regularisation.")
        if config["training"]["forecast_strategy"] != "parallel_weighted":
            raise AssertionError("Forecast strategy differs from the winner.")
        if config["training"]["scheduler"] != "modern_tcn_type3_delayed":
            raise AssertionError("Delayed schedule was not preserved.")
        weights = config["training"]["loss"]["horizon_weights"]
        if len(weights) != 5 or any(float(value) <= 0 for value in weights):
            raise AssertionError("Weighted loss was not preserved.")
    if dynamic.config["model"]["prior"]["type"] != "none":
        raise AssertionError("Dynamic-only run contains a static prior.")
    if random_static.config["model"]["prior"]["type"] != "random":
        raise AssertionError("Random-static run does not use random logits.")

    reference, _ = make_final_two_run_specs()
    allowed = {
        ("model", "variant"),
        ("model", "graph", "type"),
        ("model", "graph", "activation"),
        ("model", "graph", "gate_type"),
        ("model", "prior", "type"),
        ("model", "prior", "description"),
        ("training", "optimisation_profile"),
    }

    def flatten(value, prefix=()):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                result.update(flatten(item, prefix + (str(key),)))
            return result
        return {prefix: value}

    reference_flat = flatten(reference.config)
    for spec in specs:
        candidate_flat = flatten(spec.config)
        changed = {
            key
            for key in set(reference_flat) | set(candidate_flat)
            if reference_flat.get(key) != candidate_flat.get(key)
        }
        if not changed.issubset(allowed):
            raise AssertionError(
                f"Unexpected config changes for {spec.run_name}: "
                f"{sorted(changed - allowed)}"
            )


def _test_graph_learners() -> None:
    torch.manual_seed(9)
    temporal = torch.randn(3, 5, 6, 8)
    state = torch.randn_like(temporal)

    dynamic = PriorMixedDynamicGraphLearner(
        d_model=8,
        num_nodes=6,
        num_heads=1,
        graph_hidden_dim=8,
        graph_activation="sparsemax",
        use_state_pathway=True,
        use_static_graph=False,
        static_prior=None,
        random_static_initialisation=False,
        initial_alpha=0.5,
        prior_scale=4.0,
        prior_jitter=0.02,
        prior_seed=42,
    )
    output = dynamic(temporal, state_hidden=state)
    if output.base is not None or output.alpha is not None:
        raise AssertionError("Dynamic-only learner exposed a static branch.")
    _assert_stochastic(output.selected)
    output.selected.square().mean().backward()
    if dynamic.q_proj.weight.grad is None or dynamic.k_proj.weight.grad is None:
        raise AssertionError("Dynamic sparsemax graph received no gradients.")

    random_static = PriorMixedDynamicGraphLearner(
        d_model=8,
        num_nodes=6,
        num_heads=1,
        graph_hidden_dim=8,
        graph_activation="sparsemax",
        use_state_pathway=True,
        use_static_graph=True,
        static_prior=None,
        random_static_initialisation=True,
        initial_alpha=0.5,
        prior_scale=4.0,
        prior_jitter=0.02,
        prior_seed=42,
    )
    initial_logits = random_static.static_logits.detach().clone()
    if float(initial_logits.std().item()) <= 0.0:
        raise AssertionError("Random static logits are not random.")
    output = random_static(temporal, state_hidden=state)
    if output.base is None or output.alpha is None:
        raise AssertionError("Random-static learner is missing base/alpha.")
    _assert_stochastic(output.base)
    _assert_stochastic(output.dynamic)
    _assert_stochastic(output.selected)
    output.selected.square().mean().backward()
    if random_static.static_logits.grad is None:
        raise AssertionError("Random static logits received no gradient.")
    if random_static.raw_alpha.grad is None:
        raise AssertionError("Alpha received no gradient.")


def _write_json(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def _test_graph_hub_historical_single_layer() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "continuous_dynamic"
        run_dir.mkdir()
        assets = ["A", "B", "C"]
        horizons = [1, 5]
        resolved = {
            "data": {
                "context_length": 4,
                "stride": 1,
                "horizons": horizons,
                "input_channels": ["open", "high", "low", "close", "volume"],
                "target_channel": "close",
            },
            "model": {
                "temporal": {"type": "modern_tcn"},
                "graph": {
                    "type": "dynamic",
                    "num_heads": 1,
                    "hidden_dim": 4,
                    "activation": "softmax",
                    "add_self_loops": False,
                },
                "spatial": {"gate_type": "learned_scalar"},
            },
            "training": {"selection_metric": "validation_loss"},
        }
        metadata = {
            "status": "completed",
            "best_epoch": 2,
            "asset_cols": assets,
            "graph_heads": 1,
            "graph_type": "dynamic",
        }
        _write_json(run_dir / "resolved_config.json", resolved)
        _write_json(run_dir / "run_metadata.json", metadata)

        windows = 2
        y_true = torch.full((windows, 2, 3, 1), 100.0)
        prediction = {
            "epoch": 2,
            "prediction_result": {
                "y_pred": y_true.clone(),
                "y_true": y_true,
                "last_context_target": torch.full((windows, 3, 1), 99.0),
                "channels": ["close"],
                "horizons": horizons,
                "asset_cols": assets,
                "sample_idx": torch.tensor([0, 1]),
                "origin_idx": torch.tensor([3, 3]),
                "target_indices": torch.tensor([[4, 8], [4, 8]]),
                "output_space": "raw",
            },
        }
        adjacency = torch.tensor(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ).view(1, 1, 3, 3).expand(windows, -1, -1, -1).clone()
        graph = {
            "epoch": 2,
            "graph_artifacts": {
                "selected": adjacency,
                "base": None,
                "dynamic": adjacency,
                "dates": ["2024-10-01", "2024-10-02"],
                "orientation": "A[target, source]",
            },
        }
        torch.save(prediction, run_dir / "test_predictions.pt")
        torch.save(graph, run_dir / "test_graphs.pt")
        pd.DataFrame(
            [
                {
                    "metric": "cumulative_log_change_mae",
                    "horizon": horizon,
                    "channel": "close",
                    "value": 0.0,
                }
                for horizon in horizons
            ]
        ).to_csv(run_dir / "test_metric_table.csv", index=False)

        artifacts = load_evaluation_artifacts(
            run_dir,
            split="test",
            policy="best",
            require_graph=True,
        )
        per_layer = artifacts.graph_artifacts.get("per_layer")
        if not isinstance(per_layer, tuple) or len(per_layer) != 1:
            raise AssertionError("Historical graph was not promoted to layer 0.")
        selected = select_graph(
            run_dir,
            split="test",
            policy="best",
            component="selected",
            layer=0,
            head=0,
        )
        if selected.adjacency.shape != (3, 3):
            raise AssertionError("Historical layer-0 graph selection failed.")

        canonical = canonicalise_single_layer_graph_payload(
            graph,
            checkpoint_epoch=2,
            resolved_config=resolved,
            run_metadata=metadata,
            prediction_result=prediction["prediction_result"],
        )
        saved = canonical["graph_artifacts"]
        if len(saved["per_layer"]) != 1 or saved["selected_layer"] != 0:
            raise AssertionError("Canonical graph completion is incomplete.")


def _test_full_artifact_completion() -> None:
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        run_dir = root / "continuous_dynamic"
        data_dir.mkdir()
        run_dir.mkdir()

        config = yaml.safe_load(
            (repository / "configs" / "continuous_forecasting.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["data"].update(
            {"context_length": 4, "stride": 1, "horizons": [1, 2]}
        )
        config["model"]["temporal"].update(
            {
                "type": "transformer",
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
                "relative_position_embedding": False,
                "session_position_encoding": False,
            }
        )
        config["model"]["graph"].update(
            {
                "type": "dynamic",
                "num_heads": 1,
                "hidden_dim": 8,
                "activation": "softmax",
                "gate_type": "none",
            }
        )
        config["model"]["spatial"].update(
            {"gate_type": "learned_scalar", "initial_beta": 0.5}
        )
        config["training"].update(
            {
                "batch_size": 2,
                "validation_batch_size": 2,
                "num_workers": 0,
                "mixed_precision": False,
                "selection_horizons": [1, 2],
            }
        )
        config["training"]["loss"]["type"] = "cumulative_log_change_mae"
        continuous_runner.validate_config(config)

        assets = ["A", "B", "C"]
        channels = ["open", "high", "low", "close", "volume", "amount"]

        def make_sample(day: str, seed: int):
            generator = torch.Generator().manual_seed(seed)
            level = 100.0 + torch.arange(11).float().view(-1, 1, 1) * 0.01
            values = level.expand(11, 3, 6).clone()
            values += torch.rand((11, 3, 6), generator=generator) * 0.01
            values[..., 4:] = 1000.0 + torch.rand(
                (11, 3, 2), generator=generator
            ) * 10.0
            return values, {}, day

        split_metadata = {
            "asset_cols": assets,
            "channels": channels,
            "grain": "1m",
            "market_open": "09:30",
            "market_close": "16:00",
            "fill_method": "none",
            "T": 11,
            "F": 6,
            "D": 3,
            "dropped_days": [],
        }
        stored_splits = (
            {**split_metadata, "samples": [make_sample("2024-08-30", 1)]},
            {**split_metadata, "samples": [make_sample("2024-09-03", 2)]},
            {**split_metadata, "samples": [make_sample("2024-10-03", 3)]},
        )
        for filename, split in zip(
            ("train.pt", "val.pt", "test.pt"), stored_splits, strict=True
        ):
            torch.save(split, data_dir / filename)

        model = ContinuousForecaster(
            continuous_runner._model_config(config, num_nodes=len(assets))
        )
        torch.save(
            {
                "epoch": 1,
                "best_epoch": 1,
                "best_score": 0.1,
                "model_state_dict": model.state_dict(),
                "resolved_config": config,
            },
            run_dir / "best_checkpoint.pt",
        )
        _write_json(run_dir / "resolved_config.json", config)
        _write_json(
            run_dir / "run_metadata.json",
            {
                "status": "completed",
                "best_epoch": 1,
                "run_name": "synthetic_continuous_dynamic",
                "asset_cols": assets,
                "graph_heads": 1,
                "graph_type": "dynamic",
            },
        )

        completion = complete_continuous_run_artifacts(
            run_dir=run_dir,
            data_dir=data_dir,
            device="cpu",
            batch_size=2,
            num_workers=0,
            mixed_precision=False,
            bootstrap=False,
        )
        if set(completion.audit["Split"]) != {"train", "validation", "test"}:
            raise AssertionError("Completion did not audit all three splits.")
        for split_name in ("train", "validation", "test"):
            analysis_dir = run_dir / "analysis" / split_name
            for filename in (
                "predictions.pt",
                "graphs.pt",
                "metric_table.csv",
                "diagnostics.json",
            ):
                if not (analysis_dir / filename).is_file():
                    raise FileNotFoundError(analysis_dir / filename)
            selected = select_graph(
                run_dir,
                split=split_name,
                policy="best",
                component="selected",
                layer=0,
                head=0,
            )
            if selected.adjacency.shape != (3, 3):
                raise AssertionError("Completed graph is not selectable at layer 0.")


def _optional_real_modern_tcn_forward() -> bool:
    root = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ModernTCN"
        / "ModernTCN-Long-term-forecasting"
    )
    if not root.is_dir() or not any(root.iterdir()):
        return False
    for spec in make_final_sparsemax_graph_specs():
        config = round1_model_config_from_mapping(spec.config, num_nodes=4)
        model = ModernTCNGraphRound1Model(config, static_prior=None).eval()
        x = torch.randn(2, 60, 4, 5)
        with torch.no_grad():
            output = model(
                x,
                context_start=torch.tensor([0, 1]),
                session_length=torch.tensor([390, 390]),
            )
        if tuple(output.predictions.shape) != (2, 5, 4, 1):
            raise AssertionError("Unexpected real ModernTCN prediction shape.")
        _assert_stochastic(output.graph.selected)
    return True


def main() -> None:
    _test_specs()
    _test_graph_learners()
    _test_graph_hub_historical_single_layer()
    _test_full_artifact_completion()
    real_forward = _optional_real_modern_tcn_forward()
    print("Sparsemax specification contracts passed.")
    print("Sparsemax graph learner contracts passed.")
    print("Historical single-layer Graph Hub contract passed.")
    print("Full frozen-run artifact completion contract passed.")
    print(
        "Real ModernTCN sparsemax forward: "
        + ("passed" if real_forward else "deferred until submodule checkout")
    )


if __name__ == "__main__":
    main()

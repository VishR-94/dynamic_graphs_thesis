from __future__ import annotations

"""CPU contracts for the four dense graph-supervision diagnostics.

The specification and tensor helpers run without external submodules.  When
BaseDyGraph and ModernTCN have been initialised (as they are in the supplied
Colab notebook), the test also executes the real model forwards and verifies
causality, graph depth, row stochasticity, and gradient flow.
"""

from pathlib import Path
import json
import tempfile

import pandas as pd
import torch

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_graph_supervision_dataset import (
    AlignedTokenContinuousDenseDataset,
    make_uniform_nonself_graph,
)
from src.evaluation.dynamic_graph_evaluation import (
    load_evaluation_artifacts,
    load_unified_run_info,
    select_graph,
)
from src.models.dense_one_step_graph_controls import (
    BaseDyGraphV1ContinuousToPriceDense,
    BaseDyGraphV1TokenToPriceDense,
    ModernTCNDenseOneStepGraphModel,
    dense_basedygraph_config_from_mapping,
    modern_tcn_dense_config_from_mapping,
)
from src.training.dense_graph_supervision_specs import (
    make_dense_graph_supervision_specs,
)


def _assert_graph(graph: torch.Tensor, *, batch: int, heads: int, nodes: int) -> None:
    expected = (batch, heads, nodes, nodes)
    if tuple(graph.shape) != expected:
        raise AssertionError(f"Graph shape {tuple(graph.shape)} != {expected}.")
    if not torch.isfinite(graph).all() or torch.any(graph < 0):
        raise AssertionError("Graph is not finite/non-negative.")
    torch.testing.assert_close(
        graph.float().sum(dim=-1),
        torch.ones(batch, heads, nodes),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def _spec_contract() -> tuple:
    specs = make_dense_graph_supervision_specs()
    if len(specs) != 4:
        raise AssertionError("Expected four dense graph controls.")
    if len({spec.run_name for spec in specs}) != 4:
        raise AssertionError("Dense-control run names are not unique.")
    expected = (
        "token_to_price_dynamic",
        "price_to_price_dynamic",
        "modern_tcn_dynamic_state",
        "modern_tcn_random_static_dynamic_state",
    )
    if tuple(spec.variant for spec in specs) != expected:
        raise AssertionError("Dense-control variant ordering changed.")

    for spec in specs[:2]:
        architecture = spec.config["model"]["official_basedygraph_v1"]
        if architecture["num_st_blocks"] != 4:
            raise AssertionError("BaseDyGraph control is not four blocks.")
        if architecture["d_model"] != 96:
            raise AssertionError("BaseDyGraph width changed.")
        if architecture["graph_num_heads"] != 1:
            raise AssertionError("BaseDyGraph graph-head count changed.")
        if architecture["graph_activation"] != "softmax":
            raise AssertionError("BaseDyGraph graph activation changed.")
        if architecture["spatial_module_type"] != "dynamic_graph":
            raise AssertionError("BaseDyGraph control is not dynamic-only.")
        if spec.config["training"]["selection_split"] != "test":
            raise AssertionError("BaseDyGraph control is not test-selected.")

    dynamic, random_static = specs[2:]
    if dynamic.config["model"]["variant"] != "dynamic_state":
        raise AssertionError("ModernTCN state-aware dynamic control changed.")
    if random_static.config["model"]["variant"] != "random_static_dynamic_state":
        raise AssertionError("Random-static ModernTCN state pathway changed.")
    if random_static.config["model"]["prior"]["type"] != "random":
        raise AssertionError("Random-static control acquired an economic prior.")
    for spec in specs:
        if spec.config["training"]["scheduler_decay_start_epoch"] != 15:
            raise AssertionError("Delayed schedule no longer begins at epoch 15.")
        if spec.config["training"]["patience"] != 10:
            raise AssertionError("Patience changed.")
    return specs


def _helper_contract() -> None:
    graph = make_uniform_nonself_graph(5)
    if tuple(graph.shape) != (5, 5):
        raise AssertionError("Uniform graph shape changed.")
    torch.testing.assert_close(
        graph.sum(dim=-1),
        torch.ones(5),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.diagonal(graph),
        torch.zeros(5),
        atol=0.0,
        rtol=0.0,
    )



def _aligned_dataset_contract() -> None:
    time_points = 140
    nodes = 3
    channel_names = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    base = torch.arange(time_points, dtype=torch.float32).view(-1, 1, 1) + 100.0
    candles = base.repeat(1, nodes, len(channel_names))
    candles[..., 1] += 1.0
    candles[..., 2] -= 1.0
    candles[..., 4] = 1000.0
    candles[..., 5] = 0.0
    split = {
        "samples": [(candles, None, "2024-01-02")],
        "asset_cols": ["A", "B", "C"],
        "channels": channel_names,
    }
    continuous = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=60,
            horizons=(1, 5, 15, 30, 60),
            stride=15,
            input_channels=("open", "high", "low", "close", "volume"),
            target_channels=("open", "high", "low", "close", "volume"),
            input_representation="raw",
            clip=False,
        ),
    )
    windows = len(continuous)
    items = [continuous[index] for index in range(windows)]
    sample_idx = torch.tensor([int(item["sample_idx"]) for item in items])
    origin_idx = torch.tensor([int(item["origin_idx"]) for item in items])
    target_indices = torch.stack(
        [
            torch.arange(
                int(item["origin_idx"]) + 1,
                int(item["origin_idx"]) + 61,
            )
            for item in items
        ]
    )
    cache = {
        "format_version": 2,
        "representation": "origin_aligned_kronos_forecasting_tokens",
        "s1_id_space": "kronos_original",
        "s1_vocabulary_size": 1024,
        "context_tokens": torch.randint(0, 1024, (windows, 60, nodes, 2)),
        "target_s1": torch.randint(0, 1024, (windows, 60, nodes)),
        "target_s2": torch.randint(0, 1024, (windows, 60, nodes)),
        "context_mean": torch.zeros(windows, nodes, 6),
        "context_std": torch.ones(windows, nodes, 6),
        "evaluation_true": torch.ones(windows, 5, nodes, 5),
        "last_context_target": torch.ones(windows, nodes, 5),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": target_indices,
        "dates": ["2024-01-02"] * windows,
        "asset_cols": ["A", "B", "C"],
        "input_channels": ["open", "high", "low", "close", "volume"],
        "target_channels": ["open", "high", "low", "close", "volume"],
        "context_length": 60,
        "prediction_length": 60,
        "dense_horizons": tuple(range(1, 61)),
        "evaluation_horizons": (1, 5, 15, 30, 60),
        "evaluation_indices": (0, 4, 14, 29, 59),
        "tokenizer_channels": channel_names,
        "future_clipping_rate_percent_by_step_channel": torch.zeros(60, 6),
        "future_clipping_rate_percent_by_channel": torch.zeros(6),
        "context_clipping_rate_percent_by_step_channel": torch.zeros(60, 6),
        "context_clipping_rate_percent_by_channel": torch.zeros(6),
    }
    with tempfile.TemporaryDirectory() as directory:
        cache_path = Path(directory) / "cache.pt"
        torch.save(cache, cache_path)
        dataset = AlignedTokenContinuousDenseDataset(
            split=split,
            token_cache_path=cache_path,
            context_length=60,
            stride=15,
        )
        if len(dataset) != windows:
            raise AssertionError("Aligned dataset window count changed.")
        item = dataset[0]
        expected_shapes = {
            "context_s1": (60, nodes),
            "first_future_s1": (nodes,),
            "continuous_teacher_sequence": (61, nodes, 5),
            "dense_target_normalised_close": (60, nodes, 1),
            "raw_close_sequence": (61, nodes),
            "last_context_close": (nodes, 1),
            "future_h1_close": (1, nodes, 1),
            "target_indices": (1,),
        }
        for key, expected in expected_shapes.items():
            observed = tuple(torch.as_tensor(item[key]).shape)
            if observed != expected:
                raise AssertionError(f"{key} shape {observed} != {expected}.")



def _graph_hub_schema_contract(specs: tuple) -> None:
    """Verify that every new run schema and graph payload is Graph-Hub readable."""

    nodes = 3
    windows = 2
    asset_cols = ["A", "B", "C"]
    graph = make_uniform_nonself_graph(nodes).view(1, 1, nodes, nodes)
    selected = graph.expand(windows, -1, -1, -1).contiguous()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for spec in specs:
            run_dir = root / spec.run_name
            run_dir.mkdir(parents=True)
            config = json.loads(json.dumps(spec.config))
            metadata = {
                "status": "completed",
                "best_epoch": 1,
                "asset_cols": asset_cols,
                "graph_heads": 1,
                "graph_type": (
                    "static_dynamic_mixture"
                    if spec.variant == "modern_tcn_random_static_dynamic_state"
                    else "dynamic"
                ),
                "selection_metric": "forecast_origin_h1_cumulative_log_change_mae",
            }
            (run_dir / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            prediction_result = {
                "y_pred": torch.full((windows, 1, nodes, 1), 101.0),
                "y_true": torch.full((windows, 1, nodes, 1), 101.5),
                "last_context_target": torch.full((windows, nodes, 1), 100.0),
                "channels": ["close"],
                "horizons": [1],
                "asset_cols": asset_cols,
                "sample_idx": torch.tensor([0, 1]),
                "origin_idx": torch.tensor([59, 74]),
                "target_indices": torch.tensor([[60], [75]]),
                "output_space": "raw",
            }
            layer_count = 4 if spec.family == "basedygraph_v1" else 1
            static = (
                graph[0]
                if spec.variant == "modern_tcn_random_static_dynamic_state"
                else None
            )
            graph_artifacts = {
                "graph_type": metadata["graph_type"],
                "graph_orientation": "A[target, source]",
                "orientation": "A[target, source]",
                "asset_cols": asset_cols,
                "num_layers": layer_count,
                "num_heads": 1,
                "num_heads_per_layer": [1] * layer_count,
                "layer_head_counts": [1] * layer_count,
                "selected_layer": layer_count - 1,
                "selected": selected,
                "per_layer": tuple(selected.clone() for _ in range(layer_count)),
                "base": static,
                "per_layer_base": tuple(
                    [None] * (layer_count - 1) + [static]
                ),
                "dynamic": selected,
                "per_layer_dynamic": tuple(
                    selected.clone() for _ in range(layer_count)
                ),
                "alpha": None,
                "beta": None,
                "dates": ["2024-10-01", "2024-10-02"],
                "sample_idx": prediction_result["sample_idx"],
                "origin_idx": prediction_result["origin_idx"],
                "target_indices": prediction_result["target_indices"],
            }
            torch.save(
                {"epoch": 1, "prediction_result": prediction_result},
                run_dir / "best_test_predictions.pt",
            )
            torch.save(
                {"epoch": 1, "graph_artifacts": graph_artifacts},
                run_dir / "best_test_graphs.pt",
            )
            pd.DataFrame(
                [
                    {
                        "metric": "cumulative_log_change_mae",
                        "horizon": 1,
                        "channel": "close",
                        "value": 0.001,
                    }
                ]
            ).to_csv(run_dir / "best_test_metric_table.csv", index=False)

            info = load_unified_run_info(run_dir)
            if info.run_kind != "continuous" or info.horizons != (1,):
                raise AssertionError("Dense control is not a continuous h1 Graph-Hub run.")
            artifacts = load_evaluation_artifacts(
                run_dir,
                split="test",
                require_graph=True,
                require_metrics=True,
            )
            if tuple(artifacts.graph_artifacts["selected"].shape) != (
                windows,
                1,
                nodes,
                nodes,
            ):
                raise AssertionError("Graph Hub changed the selected graph shape.")
            snapshot = select_graph(
                run_dir,
                split="test",
                day=None,
                window=None,
                component="selected",
                layer=-1,
                head=0,
                random_seed=42,
            )
            if tuple(snapshot.adjacency.shape) != (nodes, nodes):
                raise AssertionError("Graph Hub could not select the saved graph.")

def _optional_basedygraph_forward(specs: tuple, repository: Path) -> bool:
    required = [
        repository / "external" / "BaseDyGraph" / "src" / name
        for name in ("utilities.py", "modules.py", "model.py")
    ]
    if not all(path.is_file() for path in required):
        return False

    token_config = dense_basedygraph_config_from_mapping(
        specs[0].config,
        num_nodes=5,
    )
    token_config = token_config.__class__(
        **{
            **token_config.__dict__,
            "context_length": 7,
            "d_model": 16,
            "temporal_num_heads": 4,
            "graph_num_heads": 2,
            "graph_hidden_dim": 16,
            "num_st_blocks": 4,
            "vocabulary_size": 32,
        }
    )
    torch.manual_seed(5)
    model = BaseDyGraphV1TokenToPriceDense(token_config)
    context = torch.randint(0, 32, (2, 7, 5))
    future_a = torch.randint(0, 32, (2, 5))
    future_b = (future_a + 1) % 32
    output_a = model(context, first_future_s1=future_a)
    output_b = model(context, first_future_s1=future_b)
    if tuple(output_a.normalised_close.shape) != (2, 7, 5, 1):
        raise AssertionError("Token-to-price dense output shape changed.")
    if len(output_a.per_layer_graphs) != 4:
        raise AssertionError("Token-to-price model lost graph layers.")
    # The appended future token is only a target; it cannot change any
    # predictor output or forecast-origin graph.
    torch.testing.assert_close(
        output_a.normalised_close,
        output_b.normalised_close,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    for graph_a, graph_b in zip(
        output_a.per_layer_graphs,
        output_b.per_layer_graphs,
    ):
        _assert_graph(graph_a, batch=2, heads=2, nodes=5)
        torch.testing.assert_close(graph_a, graph_b, atol=1.0e-6, rtol=1.0e-6)

    continuous_config = token_config.__class__(
        **{
            **token_config.__dict__,
            "input_mode": "continuous",
            "input_channels": 5,
        }
    )
    continuous = BaseDyGraphV1ContinuousToPriceDense(continuous_config)
    sequence = torch.randn(2, 8, 5, 5)
    changed_sequence = sequence.clone()
    changed_sequence[:, -1] = changed_sequence[:, -1] + 100.0
    continuous_output = continuous(sequence)
    continuous_changed = continuous(changed_sequence)
    if tuple(continuous_output.normalised_close.shape) != (2, 7, 5, 1):
        raise AssertionError("Continuous dense output shape changed.")
    torch.testing.assert_close(
        continuous_output.normalised_close,
        continuous_changed.normalised_close,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    for graph, changed_graph in zip(
        continuous_output.per_layer_graphs,
        continuous_changed.per_layer_graphs,
    ):
        _assert_graph(graph, batch=2, heads=2, nodes=5)
        torch.testing.assert_close(
            graph, changed_graph, atol=1.0e-6, rtol=1.0e-6
        )

    loss = (
        output_a.normalised_close.square().mean()
        + continuous_output.normalised_close.square().mean()
    )
    loss.backward()
    token_graph_gradient = sum(
        float(parameter.grad.detach().abs().sum().item())
        for name, parameter in model.named_parameters()
        if "graph_scorer" in name and parameter.grad is not None
    )
    continuous_graph_gradient = sum(
        float(parameter.grad.detach().abs().sum().item())
        for name, parameter in continuous.named_parameters()
        if "graph_scorer" in name and parameter.grad is not None
    )
    if token_graph_gradient <= 0 or continuous_graph_gradient <= 0:
        raise AssertionError("Dense price losses did not reach graph scorers.")
    return True


def _optional_modern_forward(specs: tuple, repository: Path) -> bool:
    required = (
        repository
        / "external"
        / "ModernTCN"
        / "ModernTCN-Long-term-forecasting"
        / "models"
        / "ModernTCN.py"
    )
    if not required.is_file():
        return False

    torch.manual_seed(11)
    dynamic_config = modern_tcn_dense_config_from_mapping(
        specs[2].config,
        num_nodes=5,
    )
    dynamic = ModernTCNDenseOneStepGraphModel(
        dynamic_config,
        static_scaffold=None,
    )
    random_config = modern_tcn_dense_config_from_mapping(
        specs[3].config,
        num_nodes=5,
    )
    random_static = ModernTCNDenseOneStepGraphModel(
        random_config,
        static_scaffold=make_uniform_nonself_graph(5),
    )
    random_static.initialise_random_static_logits()

    x = torch.randn(2, 60, 5, 1)
    context_start = torch.zeros(2, dtype=torch.long)
    session_length = torch.full((2,), 390, dtype=torch.long)
    for model in (dynamic, random_static):
        # The production one-step head is deliberately zero-initialised.
        # Give it a non-zero test weight so this contract tests the complete
        # graph-to-loss gradient path rather than only the first head update.
        with torch.no_grad():
            for module in model.temporal_backbone._official_backbone.head.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.normal_(mean=0.0, std=0.02)
        output = model(
            x,
            context_start=context_start,
            session_length=session_length,
        )
        if tuple(output.predictions.shape) != (2, 1, 5, 1):
            raise AssertionError("ModernTCN one-step output shape changed.")
        _assert_graph(output.graph.selected, batch=2, heads=1, nodes=5)
        if output.state_hidden is None or output.state_hidden.shape != output.temporal_hidden.shape:
            raise AssertionError("ModernTCN control lost the state pathway.")
        output.predictions.square().mean().backward()
        graph_gradient = sum(
            float(parameter.grad.detach().abs().sum().item())
            for parameter in model.graph_learner.parameters()
            if parameter.grad is not None
        )
        if graph_gradient <= 0:
            raise AssertionError("ModernTCN loss did not reach graph learner.")
    return True


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    specs = _spec_contract()
    _helper_contract()
    _aligned_dataset_contract()
    _graph_hub_schema_contract(specs)
    basedygraph_ran = _optional_basedygraph_forward(specs, repository)
    modern_ran = _optional_modern_forward(specs, repository)
    print("Dense graph-supervision configuration contracts passed.")
    print("BaseDyGraph forward contracts:", "passed" if basedygraph_ran else "deferred until submodule init")
    print("ModernTCN forward contracts:", "passed" if modern_ran else "deferred until submodule init")


if __name__ == "__main__":
    main()

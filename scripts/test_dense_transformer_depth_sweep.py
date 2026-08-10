from __future__ import annotations

"""Fast contracts for the 12-run stacked dense Transformer sweep."""

import json
from pathlib import Path
import tempfile

import pandas as pd
import torch

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_parallel_forecast_dataset import build_dense_prefix_dataset
from src.evaluation.dynamic_graph_evaluation import (
    load_evaluation_artifacts,
    select_graph,
)
from src.models.dense_transformer_depth_sweep import (
    DenseTransformerDepthConfig,
    StackedDenseTransformerGraphModel,
)
from src.training.dense_transformer_depth_specs import (
    DEFAULT_PROFILES,
    make_dense_transformer_depth_specs,
)
from src.training.run_dense_parallel_graph_supervision import (
    _build_loader,
    _new_grad_scaler,
)
from src.training.run_dense_transformer_depth_sweep import (
    _build_optimizer,
    _evaluate_selection,
    _train_epoch,
    _validate_config,
)


def _tiny_model_config(depth: int = 3) -> DenseTransformerDepthConfig:
    return DenseTransformerDepthConfig(
        num_nodes=5,
        context_length=6,
        horizons=(1, 2),
        input_channels=("open", "high", "low", "close", "volume"),
        target_channel="close",
        num_st_blocks=depth,
        d_model=12,
        transformer_num_layers=1,
        transformer_num_heads=3,
        transformer_feedforward_multiplier=2,
        transformer_dropout=0.0,
        position_embedding=False,
        graph_heads_per_block=tuple([2] * (depth - 1) + [1]),
        graph_hidden_dims_per_block=tuple([12] * depth),
        graph_activations_per_block=tuple(["softmax"] * (depth - 1) + ["sparsemax"]),
        graph_initial_alpha=0.5,
        spatial_initial_beta=0.5,
        spatial_feedforward_multiplier=2,
        spatial_dropout=0.0,
    )


def _tiny_mapping(depth: int = 2) -> dict:
    model_config = _tiny_model_config(depth)
    return {
        "model_family": "dense_transformer_depth_sweep",
        "experiment_family": "dense_transformer_depth_sweep",
        "data": {
            "context_length": 6,
            "horizons": [1, 2],
            "dense_prefix_outer_stride": 1,
            "export_stride": 1,
            "input_channels": ["open", "high", "low", "close", "volume"],
            "target_channel": "close",
            "input_representation": "raw",
        },
        "normalisation": {
            "eps": 1.0e-8,
            "clip": False,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "model": {
            "num_nodes": 5,
            "num_st_blocks": depth,
            "variant": "uniform_static_dynamic_state",
            "temporal": {
                "type": "transformer",
                "d_model": 12,
                "num_layers": 1,
                "num_heads": 3,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "position_embedding": False,
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "num_heads": model_config.graph_heads_per_block[-1],
                "num_heads_per_block": list(model_config.graph_heads_per_block),
                "num_heads_per_layer": list(model_config.graph_heads_per_block),
                "hidden_dim": model_config.graph_hidden_dims_per_block[-1],
                "hidden_dims_per_block": list(
                    model_config.graph_hidden_dims_per_block
                ),
                "activations_per_block": list(
                    model_config.graph_activations_per_block
                ),
                "activation": "sparsemax",
                "add_self_loops": False,
                "initial_alpha": 0.5,
            },
            "spatial": {
                "num_layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
            },
            "prior": {
                "type": "uniform",
                "static_logits": "zeros",
                "dynamic_logits": "zeros_at_initialisation",
                "diagonal": "excluded",
            },
            "graph_regularisation": {
                "graph_reg_layer": -1,
                "graph_reg_warmup_epochs": 0,
                "graph_entropy_reg": 0.0,
                "graph_target_entropy": None,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
            "output_representation": "normalised_close",
            "output_head_initialisation": "default",
        },
        "training": {
            "training_style": "dense_prefix",
            "optimizer": "adam",
            "parameter_grouping": "split",
            "scheduler": "modern_tcn_type3_delayed",
            "scheduler_decay_start_epoch": 15,
            "scheduler_decay_factor": 0.9,
            "learning_rate": 2.5e-4,
            "graph_learning_rate": 5.0e-4,
            "weight_decay": 0.0,
            "batch_size": 1,
            "selection_batch_size": 1,
            "export_batch_size": 1,
            "num_workers": 0,
            "max_epochs": 2,
            "patience": 1,
            "min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "seed": 42,
            "selection_split": "test",
            "selection_horizons": [1, 2],
            "selection_metric": (
                "unweighted_mean_five_horizon_cumulative_log_change_mae"
            ),
            "loss": {
                "type": "cumulative_log_change_mae",
                "bps_scale": 10000.0,
                "horizon_weighting": "inverse_reference_mae",
                "horizon_reference_mae": [0.001, 0.002],
                "horizon_weights": [4.0 / 3.0, 2.0 / 3.0],
            },
            "prefix_graph_sample_windows": 1,
        },
    }


def _tiny_split() -> dict:
    torch.manual_seed(11)
    steps = 12
    nodes = 5
    base = torch.linspace(50.0, 51.0, steps).view(steps, 1).repeat(1, nodes)
    noise = torch.randn(steps, nodes) * 0.01
    close = (base + noise).clamp_min(1.0)
    open_values = close * (1.0 + torch.randn_like(close) * 1.0e-4)
    high = torch.maximum(open_values, close) * 1.0002
    low = torch.minimum(open_values, close) * 0.9998
    volume = torch.full_like(close, 1000.0)
    amount = torch.zeros_like(close)
    session = torch.stack([open_values, high, low, close, volume, amount], dim=-1)
    return {
        "samples": [(session, None, "2024-01-02")],
        "asset_cols": [f"A{index}" for index in range(nodes)],
        "channels": ["open", "high", "low", "close", "volume", "amount"],
    }


def _test_grid() -> None:
    specs = make_dense_transformer_depth_specs()
    assert len(specs) == 12
    assert len({spec.run_name for spec in specs}) == 12
    assert {spec.depth for spec in specs} == {1, 2, 3, 4}
    assert {spec.profile_id for spec in specs} == {
        "d64_t4_g1",
        "d96_t6_g2to1",
        "d96_v2like_t4_g6to1",
    }
    v2 = next(profile for profile in DEFAULT_PROFILES if "v2like" in profile.profile_id)
    assert v2.temporal_heads == 4
    assert v2.graph_heads(4) == (6, 6, 6, 1)
    assert v2.graph_hidden_dims(4) == (192, 192, 192, 96)
    for spec in specs:
        depth = spec.depth
        activations = tuple(spec.config["model"]["graph"]["activations_per_block"])
        assert activations == tuple(["softmax"] * (depth - 1) + ["sparsemax"])
        assert spec.config["model"]["variant"] == "uniform_static_dynamic_state"
        assert spec.config["model"]["prior"]["type"] == "uniform"
        assert "uniformstatic" in spec.run_name
        _validate_config(spec.config)


def _test_forward_causality_and_gradients() -> None:
    torch.manual_seed(5)
    config = _tiny_model_config(depth=3)
    model = StackedDenseTransformerGraphModel(config).eval()
    x = torch.randn(2, 6, 5, 5)
    output = model.forward_dense(x)
    assert tuple(output.predictions.shape) == (2, 6, 2, 5, 1)
    assert len(output.block_outputs) == 3

    # All graph branches and therefore the selected graph must begin from the
    # exact same neutral off-diagonal uniform adjacency, regardless of whether
    # the block uses softmax or sparsemax.
    expected_uniform = torch.full((5, 5), 1.0 / 4.0)
    expected_uniform.fill_diagonal_(0.0)
    for block_output in output.block_outputs:
        heads = int(block_output.graph.selected.shape[2])
        expected_base = expected_uniform.view(1, 1, 5, 5).expand(
            1, heads, 5, 5
        )
        expected_dynamic = expected_uniform.view(1, 1, 1, 5, 5).expand(
            2, 6, heads, 5, 5
        )
        torch.testing.assert_close(
            block_output.graph.base, expected_base, atol=1.0e-7, rtol=0.0
        )
        torch.testing.assert_close(
            block_output.graph.dynamic, expected_dynamic, atol=1.0e-7, rtol=0.0
        )
        torch.testing.assert_close(
            block_output.graph.selected, expected_dynamic, atol=1.0e-7, rtol=0.0
        )

    for index, block in enumerate(output.block_outputs):
        heads = config.graph_heads_per_block[index]
        assert tuple(block.graph.selected.shape) == (2, 6, heads, 5, 5)
        assert tuple(block.graph.dynamic.shape) == (2, 6, heads, 5, 5)
        assert tuple(block.graph.base.shape) == (1, heads, 5, 5)
        torch.testing.assert_close(
            block.graph.selected.sum(dim=-1),
            torch.ones_like(block.graph.selected.sum(dim=-1)),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        diagonal = torch.diagonal(block.graph.selected, dim1=-2, dim2=-1)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal), atol=0, rtol=0)

    changed = x.clone()
    changed[:, 3:] = changed[:, 3:] + 10.0 * torch.randn_like(changed[:, 3:])
    with torch.inference_mode():
        original = model.forward_dense(x)
        perturbed = model.forward_dense(changed)
    torch.testing.assert_close(
        original.predictions[:, 2],
        perturbed.predictions[:, 2],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    for original_block, changed_block in zip(
        original.block_outputs, perturbed.block_outputs, strict=True
    ):
        torch.testing.assert_close(
            original_block.graph.selected[:, 2],
            changed_block.graph.selected[:, 2],
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    model.train()
    output = model.forward_dense(x)
    loss = output.predictions.square().mean()
    loss.backward()
    assert model.state_projection.weight.grad is not None
    for block in model.blocks:
        assert block.graph_learner.q_proj.weight.grad is not None
        assert block.graph_learner.k_proj.weight.grad is not None
        assert float(block.graph_learner.k_proj.weight.grad.norm().item()) > 0.0
        assert block.graph_learner.static_logits.grad is not None
        assert float(block.graph_learner.static_logits.grad.norm().item()) > 0.0
        assert block.graph_learner.raw_alpha.grad is not None
        assert block.spatial_gate.raw_beta is not None
        assert block.spatial_gate.raw_beta.grad is not None

    clone = StackedDenseTransformerGraphModel(config)
    clone.load_state_dict(model.state_dict(), strict=True)


def _test_tiny_training_and_selection() -> None:
    mapping = _tiny_mapping(depth=2)
    _validate_config(mapping)
    split = _tiny_split()
    dense = build_dense_prefix_dataset(
        split,
        context_length=6,
        horizons=(1, 2),
        stride=1,
        input_channels=("open", "high", "low", "close", "volume"),
        target_channel="close",
    )
    export = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=6,
            horizons=(1, 2),
            stride=1,
            input_channels=("open", "high", "low", "close", "volume"),
            target_channels=("close",),
            input_representation="raw",
        ),
    )
    model = StackedDenseTransformerGraphModel(_tiny_model_config(depth=2))
    optimizer = _build_optimizer(model, mapping)
    scaler = _new_grad_scaler(False)
    train_values = _train_epoch(
        model=model,
        dataset=dense,
        device=torch.device("cpu"),
        optimizer=optimizer,
        scaler=scaler,
        use_amp=False,
        config=mapping,
        epoch=1,
    )
    assert train_values["training_native_loss"] >= 0.0
    loader = _build_loader(
        export,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        seed=42,
        pin_memory=False,
    )
    selection = _evaluate_selection(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        use_amp=False,
        config=mapping,
        description="test selection",
    )
    assert selection["selection_score"] >= 0.0
    assert "block_1_selected_entropy" in selection


def _row_stochastic(windows: int, heads: int, nodes: int) -> torch.Tensor:
    values = torch.rand(windows, heads, nodes, nodes)
    diagonal = torch.eye(nodes, dtype=torch.bool).view(1, 1, nodes, nodes)
    values = values.masked_fill(diagonal, 0.0)
    return values / values.sum(dim=-1, keepdim=True)


def _test_graph_hub_schema() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "depth_model"
        analysis = run_dir / "analysis" / "test"
        analysis.mkdir(parents=True)
        nodes = 5
        windows = 3
        horizons = [1, 2]
        assets = [f"A{index}" for index in range(nodes)]
        selected0 = _row_stochastic(windows, 2, nodes)
        selected1 = _row_stochastic(windows, 1, nodes)
        dynamic0 = _row_stochastic(windows, 2, nodes)
        dynamic1 = _row_stochastic(windows, 1, nodes)
        base0 = _row_stochastic(1, 2, nodes)[0]
        base1 = _row_stochastic(1, 1, nodes)[0]
        predictions = {
            "y_pred": torch.ones(windows, 2, nodes, 1) * 100.0,
            "y_true": torch.ones(windows, 2, nodes, 1) * 100.1,
            "last_context_target": torch.ones(windows, nodes, 1) * 100.0,
            "channels": ["close"],
            "horizons": horizons,
            "asset_cols": assets,
            "sample_idx": torch.arange(windows),
            "origin_idx": torch.tensor([59, 74, 89]),
            "target_indices": torch.tensor([[60, 61], [75, 76], [90, 91]]),
            "output_space": "raw",
        }
        graphs = {
            "graph_type": "static_dynamic_mixture",
            "orientation": "A[target, source]",
            "graph_orientation": "A[target, source]",
            "asset_cols": assets,
            "num_layers": 2,
            "num_heads": 1,
            "num_heads_per_layer": [2, 1],
            "layer_head_counts": [2, 1],
            "graph_hidden_dims_per_layer": [12, 12],
            "graph_activations_per_layer": ["softmax", "sparsemax"],
            "selected_layer": 1,
            "selected": selected1,
            "per_layer": (selected0, selected1),
            "base": base1,
            "per_layer_base": (base0, base1),
            "dynamic": dynamic1,
            "per_layer_dynamic": (dynamic0, dynamic1),
            "alpha": torch.tensor([0.5]),
            "alpha_per_layer": (torch.tensor([0.5]), torch.tensor([0.5])),
            "beta": torch.tensor([0.5]),
            "beta_per_layer": torch.tensor([0.5, 0.5]),
            "dates": ["2024-10-01", "2024-10-02", "2024-10-03"],
            "sample_idx": predictions["sample_idx"],
            "origin_idx": predictions["origin_idx"],
            "target_indices": predictions["target_indices"],
        }
        torch.save(
            {"epoch": 2, "prediction_result": predictions},
            analysis / "predictions.pt",
        )
        torch.save(
            {"epoch": 2, "graph_artifacts": graphs},
            analysis / "graphs.pt",
        )
        pd.DataFrame(
            [{"metric": "cumulative_log_change_mae", "horizon": 1, "channel": "close", "value": 0.001}]
        ).to_csv(analysis / "metric_table.csv", index=False)
        (analysis / "diagnostics.json").write_text(
            json.dumps({"checkpoint_epoch": 2}), encoding="utf-8"
        )
        mapping = _tiny_mapping(depth=2)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(mapping), encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "best_epoch": 2,
                    "epochs_completed": 3,
                    "model_family": "dense_transformer_depth_sweep",
                    "run_name": run_dir.name,
                    "asset_cols": assets,
                    "horizons": horizons,
                    "context_length": 6,
                }
            ),
            encoding="utf-8",
        )

        loaded = load_evaluation_artifacts(
            run_dir,
            split="test",
            policy=None,
            require_graph=True,
        )
        assert tuple(loaded.graph_artifacts["selected"].shape) == (windows, 1, nodes, nodes)
        layer0 = select_graph(
            run_dir,
            split="test",
            day=None,
            window=None,
            component="selected",
            layer=0,
            head=0,
        )
        assert tuple(layer0.adjacency.shape) == (nodes, nodes)
        base1_view = select_graph(
            run_dir,
            split="test",
            day=None,
            window=None,
            component="base",
            layer=1,
            head=0,
        )
        assert tuple(base1_view.adjacency.shape) == (nodes, nodes)


def main() -> None:
    _test_grid()
    _test_forward_causality_and_gradients()
    _test_tiny_training_and_selection()
    _test_graph_hub_schema()
    print("Dense Transformer depth-sweep contracts passed.")


if __name__ == "__main__":
    main()

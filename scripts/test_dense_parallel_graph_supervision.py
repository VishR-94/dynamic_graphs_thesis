from __future__ import annotations

"""Fast contracts for the twelve dense parallel graph-supervision runs."""

import json
from pathlib import Path
import sys
import tempfile
import types

import pandas as pd
import torch
from torch import nn

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.data.dense_parallel_forecast_dataset import (
    build_dense_prefix_dataset,
    repeat_batch_for_prefixes,
    right_aligned_prefix_batch,
)
from src.models.dense_parallel_graph_models import (
    DenseParallelGraphModelConfig,
    ModernTCNDenseParallelGraphModel,
    TransformerDenseParallelGraphModel,
)
from src.models.graph_priors import build_absolute_correlation_graph_prior
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Model,
    round1_model_config_from_mapping,
)
from src.training.dense_parallel_graph_specs import (
    DEFAULT_REFERENCE_MAE,
    inverse_reference_weights,
    make_dense_parallel_graph_specs,
)
from src.evaluation.dynamic_graph_evaluation import (
    discover_models,
    load_evaluation_artifacts,
    make_model_artifact_audit,
    make_unified_model_summary_table,
    select_graph,
)
from src.training.run_dense_parallel_graph_supervision import (
    _advance_schedule,
    _build_optimizer,
    _dense_absolute_error,
    _normalised_to_raw,
    _select_dense_tensor,
    _validate_config,
)


def _install_fake_modern_tcn() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ModernTCN"
        / "ModernTCN-Long-term-forecasting"
    )
    root.mkdir(parents=True, exist_ok=True)

    package = types.ModuleType("models")
    package.__path__ = []
    module = types.ModuleType("models.ModernTCN")

    class FakeHead(nn.Module):
        def __init__(self, *, d_model: int, length: int, horizons: int) -> None:
            super().__init__()
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_model * length, horizons)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.linear(self.flatten(values))

    class FakeInner(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.patch_size = int(config.patch_size)
            self.patch_stride = int(config.patch_stride)
            self.padding = self.patch_size - self.patch_stride
            self.d_model = int(config.dims[0])
            self.output_length = int(config.seq_len) // self.patch_stride
            self.stem = nn.Linear(self.patch_size, self.d_model)
            self.head = FakeHead(
                d_model=self.d_model,
                length=self.output_length,
                horizons=int(config.pred_len),
            )

        def forward_feature(self, values: torch.Tensor) -> torch.Tensor:
            if self.padding:
                values = torch.cat(
                    [
                        values,
                        values[..., -1:].expand(*values.shape[:-1], self.padding),
                    ],
                    dim=-1,
                )
            patches = values.unfold(-1, self.patch_size, self.patch_stride)
            features = self.stem(patches)
            return features.permute(0, 1, 3, 2).contiguous()

    class FakeModel(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.model = FakeInner(config)

    module.Model = FakeModel
    package.ModernTCN = module
    sys.modules["models"] = package
    sys.modules["models.ModernTCN"] = module


def _synthetic_split(*, num_nodes: int = 4, length: int = 78) -> dict:
    torch.manual_seed(91)
    channels = ["open", "high", "low", "close", "volume", "amount"]
    samples = []
    for day_index in range(3):
        base = 80.0 + torch.cumsum(0.02 * torch.randn(length, num_nodes), dim=0)
        open_price = base + 0.003 * torch.randn_like(base)
        close = base + 0.003 * torch.randn_like(base)
        high = torch.maximum(open_price, close) + 0.01
        low = torch.minimum(open_price, close) - 0.01
        volume = 1000.0 + 20.0 * torch.rand_like(base)
        amount = torch.zeros_like(base)
        samples.append(
            (
                torch.stack(
                    [open_price, high, low, close, volume, amount],
                    dim=-1,
                ),
                {},
                f"2024-01-{day_index + 2:02d}",
            )
        )
    return {
        "samples": samples,
        "asset_cols": [f"A{index}" for index in range(num_nodes)],
        "channels": channels,
    }


def _collate(dataset, count: int = 2) -> dict:
    items = [dataset[index] for index in range(count)]
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return result


def _model_config(
    *,
    temporal: str,
    graph_variant: str,
    context_length: int = 8,
    horizons: tuple[int, ...] = (1, 2),
) -> DenseParallelGraphModelConfig:
    return DenseParallelGraphModelConfig(
        num_nodes=4,
        context_length=context_length,
        horizons=horizons,
        input_channels=("open", "high", "low", "close", "volume"),
        target_channel="close",
        temporal_backbone=temporal,  # type: ignore[arg-type]
        graph_variant=graph_variant,  # type: ignore[arg-type]
        modern_tcn_d_model=8,
        modern_tcn_patch_size=4,
        modern_tcn_patch_stride=2,
        modern_tcn_ffn_ratio=1,
        modern_tcn_num_blocks=1,
        modern_tcn_large_kernel=3,
        modern_tcn_small_kernel=3,
        modern_tcn_dropout=0.0,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_feedforward_multiplier=2,
        transformer_dropout=0.0,
        transformer_position_embedding=True,
        graph_num_heads=1,
        graph_hidden_dim=8,
        graph_activation="softmax",
        graph_initial_alpha=0.5,
        spatial_initial_beta=0.5,
        spatial_feedforward_multiplier=2,
        spatial_dropout=0.0,
        prior_scale=4.0,
        prior_jitter=0.02,
        prior_seed=42,
    )


def _assert_graph(graph: torch.Tensor, *, steps: int | None = None) -> None:
    expected_ndim = 5 if steps is not None else 4
    if graph.ndim != expected_ndim:
        raise AssertionError(f"Unexpected graph rank {graph.ndim}.")
    torch.testing.assert_close(
        graph.float().sum(dim=-1),
        torch.ones_like(graph.float().sum(dim=-1)),
        atol=1.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.diagonal(graph.float(), dim1=-2, dim2=-1),
        torch.zeros_like(torch.diagonal(graph.float(), dim1=-2, dim2=-1)),
        atol=0.0,
        rtol=0.0,
    )
    if steps is not None and int(graph.shape[1]) != steps:
        raise AssertionError("Dense graph sequence has the wrong number of steps.")


def _round1_mapping() -> dict:
    return {
        "data": {
            "context_length": 8,
            "horizons": [1, 2],
            "stride": 1,
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
            "output_representation": "normalised_close",
            "output_head_initialisation": "default",
            "variant": "prior_mixture_state",
            "temporal": {
                "type": "modern_tcn",
                "d_model": 8,
                "num_blocks": 1,
                "patch_size": 4,
                "patch_stride": 2,
                "ffn_ratio": 1,
                "large_kernel": 3,
                "small_kernel": 3,
                "dropout": 0.0,
                "head_dropout": 0.0,
                "session_position_encoding": False,
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "num_heads": 1,
                "hidden_dim": 8,
                "activation": "softmax",
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
            "prior": {"type": "correlation", "scale": 4.0, "jitter": 0.02, "seed": 42},
            "graph_regularisation": {
                "graph_entropy_reg": 0.0,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
        },
        "training": {
            "selection_split": "test",
            "selection_horizons": [1, 2],
            "optimizer": "adam",
            "parameter_grouping": "split",
            "scheduler": "modern_tcn_type3_delayed",
            "scheduler_decay_start_epoch": 15,
            "scheduler_decay_factor": 0.9,
            "learning_rate": 2.5e-4,
            "graph_learning_rate": 5.0e-4,
            "weight_decay": 0.0,
            "batch_size": 2,
            "selection_batch_size": 2,
            "export_batch_size": 2,
            "max_epochs": 3,
            "patience": 2,
        },
    }


def main() -> None:
    _install_fake_modern_tcn()

    # Exact selected loss weights.
    observed_weights = inverse_reference_weights(DEFAULT_REFERENCE_MAE)
    expected_weights = [
        3.7295707385901125,
        1.7489229046582941,
        1.039473644407472,
        0.7471142661463033,
        0.5377548425463323,
    ]
    torch.testing.assert_close(
        torch.tensor(observed_weights),
        torch.tensor(expected_weights),
        atol=1.0e-12,
        rtol=0.0,
    )

    specs = make_dense_parallel_graph_specs(
        context_length=8,
        horizons=(1, 2),
        export_stride=2,
        stride1_training_stride=1,
        dense_prefix_outer_stride=2,
        reference_mae=(0.1, 0.2),
        modern_tcn_d_model=8,
        modern_tcn_patch_size=4,
        modern_tcn_patch_stride=2,
        modern_tcn_large_kernel=3,
        modern_tcn_small_kernel=3,
        transformer_d_model=16,
        transformer_num_layers=1,
        transformer_num_heads=4,
        graph_hidden_dim=8,
        stride1_batch_size=2,
        dense_prefix_modern_tcn_batch_size=1,
        dense_prefix_transformer_batch_size=2,
        selection_batch_size=2,
        export_batch_size=2,
        prefix_chunk_size=2,
        max_epochs=3,
        patience=2,
    )
    if len(specs) != 12 or len({spec.run_name for spec in specs}) != 12:
        raise AssertionError("The twelve-run grid is incomplete or non-unique.")
    combinations = {
        (spec.temporal_backbone, spec.training_style, spec.graph_variant)
        for spec in specs
    }
    if len(combinations) != 12:
        raise AssertionError("The complete 2×2×3 grid was not generated.")
    for spec in specs:
        _validate_config(spec.config)
        if spec.config["training"]["selection_split"] != "test":
            raise AssertionError("A run is not test selected.")
        if spec.config["model"]["graph_regularisation"]["graph_entropy_reg"] != 0.0:
            raise AssertionError("Graph regularisation was enabled unexpectedly.")

    split = _synthetic_split(length=18)
    dense_dataset = build_dense_prefix_dataset(
        split,
        context_length=8,
        horizons=(1, 2),
        stride=2,
        input_channels=("open", "high", "low", "close", "volume"),
    )
    item = dense_dataset[0]
    if tuple(item["x"].shape) != (8, 4, 5):
        raise AssertionError("Dense-prefix input shape is wrong.")
    if tuple(item["dense_y_unnormalised"].shape) != (8, 2, 4, 1):
        raise AssertionError("Dense-prefix target shape is wrong.")
    if item["dense_target_indices"][0].tolist() != [1, 2]:
        raise AssertionError("The first internal origin has wrong targets.")
    if item["dense_target_indices"][-1].tolist() != [8, 9]:
        raise AssertionError("The final internal origin has wrong targets.")
    torch.testing.assert_close(
        item["y_unnormalised"], item["dense_y_unnormalised"][-1]
    )

    x = torch.arange(2 * 8 * 4 * 5, dtype=torch.float32).reshape(2, 8, 4, 5)
    prefixes = right_aligned_prefix_batch(x, (0, 3, 7))
    if tuple(prefixes.shape) != (6, 8, 4, 5):
        raise AssertionError("Prefix batching shape is wrong.")
    torch.testing.assert_close(prefixes[0, -1], x[0, 0])
    torch.testing.assert_close(prefixes[1, -1], x[1, 0])
    torch.testing.assert_close(prefixes[2, -4:], x[0, :4])
    torch.testing.assert_close(prefixes[4], x[0])
    repeated = repeat_batch_for_prefixes(torch.tensor([10, 20]), 3)
    if repeated.tolist() != [10, 20, 10, 20, 10, 20]:
        raise AssertionError("Prefix metadata order differs from prefix inputs.")

    # Dense tensor extraction uses the same prefix-major order.
    values = torch.arange(2 * 8 * 2).reshape(2, 8, 2)
    indices = torch.tensor([0, 3, 7])
    selected = _select_dense_tensor(values, indices)
    expected = torch.stack(
        [values[0, 0], values[1, 0], values[0, 3], values[1, 3], values[0, 7], values[1, 7]]
    )
    torch.testing.assert_close(selected, expected)

    prior = build_absolute_correlation_graph_prior(
        split,
        expected_asset_cols=split["asset_cols"],
        threshold=None,
    )

    # Transformer dense forward produces one graph/prediction per minute and
    # its last position is exactly the ordinary fixed-context forward.
    transformer_config = _model_config(
        temporal="transformer",
        graph_variant="correlation_static_dynamic_state",
    )
    torch.manual_seed(7)
    transformer = TransformerDenseParallelGraphModel(
        transformer_config,
        static_prior=prior,
    ).eval()
    batch = _collate(dense_dataset, count=2)
    batch_x = batch["x"].float()
    dense_output = transformer.forward_dense(
        batch_x,
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    fixed_output = transformer(
        batch_x,
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    if tuple(dense_output.predictions.shape) != (2, 8, 2, 4, 1):
        raise AssertionError("Dense Transformer prediction shape is wrong.")
    _assert_graph(dense_output.graphs.selected, steps=8)
    _assert_graph(dense_output.graphs.dynamic, steps=8)
    torch.testing.assert_close(
        fixed_output.predictions,
        dense_output.predictions[:, -1],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        fixed_output.graph.selected,
        dense_output.graphs.selected[:, -1],
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    # Causal contract: changing future context rows cannot change an earlier
    # hidden prediction or graph.
    altered = batch_x.clone()
    altered[:, 4:] += 100.0
    altered_output = transformer.forward_dense(
        altered,
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    torch.testing.assert_close(
        dense_output.predictions[:, 3],
        altered_output.predictions[:, 3],
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    torch.testing.assert_close(
        dense_output.graphs.selected[:, 3],
        altered_output.graphs.selected[:, 3],
        atol=2.0e-6,
        rtol=2.0e-6,
    )

    # All graph variants are valid and the loss reaches state, graph, alpha and beta.
    for graph_variant in (
        "correlation_static_dynamic_state",
        "random_static_dynamic_state",
        "dynamic_state",
    ):
        config = _model_config(temporal="transformer", graph_variant=graph_variant)
        static = prior if graph_variant.startswith("correlation") else None
        model = TransformerDenseParallelGraphModel(config, static_prior=static)
        output = model.forward_dense(
            batch_x,
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
        _assert_graph(output.graphs.selected, steps=8)
        _assert_graph(output.graphs.dynamic, steps=8)
        if graph_variant == "dynamic_state":
            if output.graphs.base is not None or output.graphs.alpha is not None:
                raise AssertionError("Dynamic-only graph exposed static/alpha state.")
        else:
            if output.graphs.base is None or output.graphs.alpha is None:
                raise AssertionError("Static/dynamic graph lost base/alpha state.")
        loss = output.predictions.square().mean()
        loss.backward()
        if model.graph_learner.q_proj.weight.grad is None:
            raise AssertionError("Prediction loss did not reach graph scoring.")
        if model.state_projection.weight.grad is None:
            raise AssertionError("Prediction loss did not reach state projection.")
        if model.spatial_gate.raw_beta is None or model.spatial_gate.raw_beta.grad is None:
            raise AssertionError("Prediction loss did not reach beta.")
        if graph_variant != "dynamic_state":
            if model.graph_learner.raw_alpha is None or model.graph_learner.raw_alpha.grad is None:
                raise AssertionError("Prediction loss did not reach alpha.")

    # Dense target inversion/error uses the full-window target statistics and
    # is zero when predictions equal the saved normalised targets.
    dense_true = batch["dense_y_unnormalised"].float()
    target_mean = batch["target_norm_mean"].float()
    target_std = batch["target_norm_std"].float()
    exact_normalised = (
        dense_true
        - target_mean[:, None, None, :, :]
    ) / target_std[:, None, None, :, :]
    reconstructed = _normalised_to_raw(
        exact_normalised,
        target_mean=target_mean,
        target_std=target_std,
        dense=True,
    )
    torch.testing.assert_close(reconstructed, dense_true, atol=1.0e-5, rtol=1.0e-6)
    _, exact_error = _dense_absolute_error(
        exact_normalised,
        dense_true_raw=dense_true,
        dense_current_close=batch["dense_current_close"].float(),
        target_mean=target_mean,
        target_std=target_std,
        eps=1.0e-8,
    )
    torch.testing.assert_close(exact_error, torch.zeros_like(exact_error), atol=2.0e-7, rtol=0.0)

    # ModernTCN implementation is an exact architectural clone of the selected
    # Round-1 model for the correlation/static/state variant.
    round1_values = _round1_mapping()
    round1_config = round1_model_config_from_mapping(round1_values, num_nodes=4)
    new_config = _model_config(
        temporal="modern_tcn",
        graph_variant="correlation_static_dynamic_state",
    )
    torch.manual_seed(123)
    historical = ModernTCNGraphRound1Model(round1_config, static_prior=prior).eval()
    torch.manual_seed(123)
    dense_modern_tcn = ModernTCNDenseParallelGraphModel(
        new_config,
        static_prior=prior,
    ).eval()
    continuous_dataset = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=8,
            horizons=(1, 2),
            stride=2,
            input_channels=("open", "high", "low", "close", "volume"),
            target_channels=("close",),
        ),
    )
    continuous_batch = _collate(continuous_dataset, count=2)
    historical_output = historical(
        continuous_batch["x"].float(),
        context_start=continuous_batch["context_start"],
        session_length=continuous_batch["session_length"],
    )
    new_output = dense_modern_tcn(
        continuous_batch["x"].float(),
        context_start=continuous_batch["context_start"],
        session_length=continuous_batch["session_length"],
    )
    torch.testing.assert_close(new_output.predictions, historical_output.predictions)
    torch.testing.assert_close(new_output.graph.selected, historical_output.graph.selected)
    torch.testing.assert_close(new_output.graph.dynamic, historical_output.graph.dynamic)
    torch.testing.assert_close(new_output.fused_hidden, historical_output.fused_hidden)

    # Split parameter groups and delayed schedule preserve the selected profile.
    config_values = specs[0].config
    optimizer = _build_optimizer(transformer, config_values)
    if len(optimizer.param_groups) != 2:
        raise AssertionError("Expected separate backbone and graph groups.")
    _advance_schedule(
        optimizer,
        training=config_values["training"],
        completed_epoch=14,
    )
    before = [float(group["lr"]) for group in optimizer.param_groups]
    _advance_schedule(
        optimizer,
        training=config_values["training"],
        completed_epoch=15,
    )
    after = [float(group["lr"]) for group in optimizer.param_groups]
    torch.testing.assert_close(
        torch.tensor(before),
        torch.tensor([2.5e-4, 5.0e-4]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.tensor(after),
        torch.tensor([2.25e-4, 4.5e-4]),
        atol=1.0e-12,
        rtol=0.0,
    )

    # The generic saved-artifact schema is readable by Graph Hub without a
    # model-family-specific config path.
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary) / "dense_parallel_contract"
        analysis = run / "analysis" / "train"
        analysis.mkdir(parents=True)
        saved_config = specs[0].config
        (run / "resolved_config.json").write_text(
            json.dumps(saved_config, indent=2),
            encoding="utf-8",
        )
        (run / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "asset_cols": ["A0", "A1", "A2", "A3"],
                    "best_epoch": 1,
                    "model_family": "dense_parallel_graph_supervision",
                    "temporal_backbone": specs[0].temporal_backbone,
                    "training_style": specs[0].training_style,
                    "graph_type": saved_config["model"]["graph"]["type"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        windows = 2
        horizons = saved_config["data"]["horizons"]
        last = torch.full((windows, 4, 1), 100.0)
        true = last[:, None] * torch.exp(
            torch.tensor([0.001, 0.002]).view(1, 2, 1, 1)
        )
        prediction = {
            "y_pred": true.clone(),
            "y_true": true,
            "last_context_target": last,
            "channels": ["close"],
            "horizons": horizons,
            "asset_cols": ["A0", "A1", "A2", "A3"],
            "sample_idx": torch.tensor([0, 1]),
            "origin_idx": torch.tensor([7, 7]),
            "target_indices": torch.tensor([[8, 9], [8, 9]]),
            "output_space": "raw",
        }
        graph_values = transformer.graph_learner.forward_window(
            torch.randn(windows, 3, 4, 16),
            torch.randn(windows, 3, 4, 16),
        )
        graph_payload = {
            "graph_type": saved_config["model"]["graph"]["type"],
            "graph_orientation": "A[target, source]",
            "orientation": "A[target, source]",
            "asset_cols": ["A0", "A1", "A2", "A3"],
            "num_layers": 1,
            "num_heads": 1,
            "num_heads_per_layer": [1],
            "layer_head_counts": [1],
            "selected_layer": 0,
            "selected": graph_values.selected.detach(),
            "per_layer": (graph_values.selected.detach(),),
            "base": graph_values.base.detach()[0] if graph_values.base is not None else None,
            "per_layer_base": (
                graph_values.base.detach()[0] if graph_values.base is not None else None,
            ),
            "dynamic": graph_values.dynamic.detach(),
            "per_layer_dynamic": (graph_values.dynamic.detach(),),
            "alpha": graph_values.alpha.detach().reshape(1) if graph_values.alpha is not None else None,
            "beta": torch.tensor([0.5]),
            "dates": ["2024-01-02", "2024-01-03"],
            "sample_idx": prediction["sample_idx"],
            "origin_idx": prediction["origin_idx"],
            "target_indices": prediction["target_indices"],
        }
        torch.save(
            {"epoch": 1, "prediction_result": prediction},
            analysis / "predictions.pt",
        )
        torch.save(
            {"epoch": 1, "graph_artifacts": graph_payload},
            analysis / "graphs.pt",
        )
        pd.DataFrame(
            [
                {
                    "metric": "cumulative_log_change_mae",
                    "horizon": int(horizon),
                    "channel": "close",
                    "value": 0.001,
                }
                for horizon in horizons
            ]
        ).to_csv(analysis / "metric_table.csv", index=False)
        artifacts = load_evaluation_artifacts(
            run,
            split="train",
            policy=None,
            require_graph=True,
            require_metrics=True,
        )
        if tuple(artifacts.graph_artifacts["selected"].shape) != (2, 1, 4, 4):
            raise AssertionError("Graph Hub changed the selected graph shape.")
        selected_graph = select_graph(
            run,
            split="train",
            policy=None,
            day=None,
            window=None,
            component="selected",
            layer=-1,
            head=0,
        )
        if selected_graph.adjacency.shape != (4, 4):
            raise AssertionError("Graph Hub could not select the saved graph.")
        discovered = discover_models(models_root=Path(temporary))
        if len(discovered) != 1 or discovered.iloc[0]["Issue"] is not None:
            raise AssertionError(
                "Graph Hub discover_models could not read the new run schema: "
                f"{discovered.to_dict(orient='records')}"
            )
        audit = make_model_artifact_audit(
            {"Dense parallel contract": run},
            split="train",
        )
        if len(audit) != 1 or not bool(audit.iloc[0]["Ready"]):
            raise AssertionError(
                "Graph Hub artifact audit rejected the new run schema: "
                f"{audit.to_dict(orient='records')}"
            )
        summary = make_unified_model_summary_table(run)
        if "Temporal backbone" not in summary.index:
            raise AssertionError("Graph Hub architecture summary lost temporal metadata.")

    # Run names carry a configuration hash, so changed exposed controls cannot
    # silently reuse prior completed folders.
    changed_specs = make_dense_parallel_graph_specs(transformer_dropout=0.1)
    if {spec.run_name for spec in changed_specs} & {
        spec.run_name for spec in make_dense_parallel_graph_specs()
    }:
        raise AssertionError("Changed Transformer dropout collided with old run names.")

    print("Dense parallel graph-supervision contracts passed.")


if __name__ == "__main__":
    main()

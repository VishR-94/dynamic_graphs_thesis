from __future__ import annotations

"""Fast contract tests for the continuous temporal/graph forecasting path."""

from pathlib import Path
import math

import torch
import torch.nn.functional as F

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.evaluation.prediction_transforms import (
    cumulative_log_change_to_raw,
    inverse_window_normalisation,
    raw_to_cumulative_log_change,
)
from src.models.continuous_forecaster import (
    ContinuousForecaster,
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
)
from src.models.dynamic_graph.contracts import GraphConfig
from src.models.dynamic_graph.graph_learners import (
    FixedGraphLearner,
    build_window_absolute_correlation_adjacency,
)
from src.models.dynamic_graph.losses import (
    GraphRegularisationConfig,
    compute_graph_regularisation,
)
from src.training.run_continuous_forecaster import (
    _adjust_learning_rate,
    _build_optimizer,
    _current_learning_rates,
    _graph_context_values,
    _loss_values,
    _trainable_parameter_partition,
)


def _synthetic_split() -> dict:
    torch.manual_seed(91)
    channels = ["open", "high", "low", "close", "volume", "amount"]
    samples = []
    for day_index in range(3):
        base = 50.0 + torch.cumsum(
            0.02 * torch.randn(390, 4),
            dim=0,
        )
        open_price = base + 0.01 * torch.randn_like(base)
        close = base + 0.01 * torch.randn_like(base)
        high = torch.maximum(open_price, close) + 0.02
        low = torch.minimum(open_price, close) - 0.02
        volume = 1000.0 + 20.0 * torch.rand_like(base)
        amount = torch.zeros_like(base)
        values = torch.stack(
            [open_price, high, low, close, volume, amount],
            dim=-1,
        )
        samples.append((values, {}, f"2024-01-{day_index + 2:02d}"))
    return {
        "samples": samples,
        "asset_cols": ["A", "B", "C", "D"],
        "channels": channels,
    }


def _batch(dataset, batch_size: int = 2) -> dict:
    items = [dataset[index] for index in range(batch_size)]
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


def main() -> None:
    split = _synthetic_split()
    dataset_config = ContinuousDatasetConfig(
        context_length=60,
        horizons=(1, 5, 15, 30, 60),
        stride=15,
    )
    dataset = build_continuous_dataset(split, config=dataset_config)
    batch = _batch(dataset)

    # Normalisation round trip.
    recovered = inverse_window_normalisation(
        batch["y"],
        batch["target_norm_mean"],
        batch["target_norm_std"],
    )
    torch.testing.assert_close(
        recovered,
        batch["y_unnormalised"],
        atol=1.0e-5,
        rtol=1.0e-6,
    )
    expected_target_change = raw_to_cumulative_log_change(
        batch["y_unnormalised"],
        batch["last_context_target"],
    )
    torch.testing.assert_close(
        batch["target_cumulative_log_change"],
        expected_target_change,
        atol=0.0,
        rtol=0.0,
    )

    temporal = ContinuousTemporalConfig(
        type="transformer",
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    no_graph_config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=60,
        horizons=(1, 5, 15, 30, 60),
        temporal=temporal,
        graph=GraphConfig(
            type="none",
            num_heads=2,
            hidden_dim=16,
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
    )
    model = ContinuousForecaster(no_graph_config).eval()
    output = model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    if tuple(output.predictions.shape) != (2, 5, 4, 1):
        raise AssertionError("Unexpected forecast shape.")

    # MSE parity.
    optimisation, native = _loss_values(
        output.predictions,
        batch,
        device=torch.device("cpu"),
        output_representation="normalised_close",
        loss_type="mse",
        bps_scale=10000.0,
        eps=1.0e-8,
    )
    expected_mse = F.mse_loss(
        output.predictions.float(),
        batch["y"].float(),
    )
    torch.testing.assert_close(native, expected_mse)
    torch.testing.assert_close(optimisation, expected_mse)

    # CLG-MAE parity against the common transform.
    optimisation, native = _loss_values(
        output.predictions,
        batch,
        device=torch.device("cpu"),
        output_representation="normalised_close",
        loss_type="cumulative_log_change_mae",
        bps_scale=10000.0,
        eps=1.0e-8,
    )
    raw_prediction = inverse_window_normalisation(
        output.predictions.float(),
        batch["target_norm_mean"],
        batch["target_norm_std"],
    ).clamp_min(1.0e-8)
    predicted_change = raw_to_cumulative_log_change(
        raw_prediction,
        batch["last_context_target"],
    )
    true_change = raw_to_cumulative_log_change(
        batch["y_unnormalised"],
        batch["last_context_target"],
    )
    expected_clg = torch.abs(predicted_change - true_change).mean()
    torch.testing.assert_close(native, expected_clg)
    torch.testing.assert_close(optimisation, 10000.0 * expected_clg)

    # Direct cumulative-log-change output and persistence initialisation.
    direct_config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=60,
        horizons=(1, 5, 15, 30, 60),
        output_representation="cumulative_log_change",
        output_head_initialisation="zero",
        temporal=temporal,
        graph=GraphConfig(
            type="none",
            num_heads=2,
            hidden_dim=16,
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
    )
    direct_model = ContinuousForecaster(direct_config).eval()
    direct_output = direct_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    torch.testing.assert_close(
        direct_output.predictions,
        torch.zeros_like(direct_output.predictions),
        atol=0.0,
        rtol=0.0,
    )
    persistence_raw = cumulative_log_change_to_raw(
        direct_output.predictions,
        batch["last_context_target"],
    )
    expected_persistence = batch["last_context_target"][:, None].expand_as(
        persistence_raw
    )
    torch.testing.assert_close(
        persistence_raw,
        expected_persistence,
        atol=0.0,
        rtol=0.0,
    )
    direct_optimisation, direct_native = _loss_values(
        direct_output.predictions,
        batch,
        device=torch.device("cpu"),
        output_representation="cumulative_log_change",
        loss_type="cumulative_log_change_mae",
        bps_scale=10000.0,
        eps=1.0e-8,
    )
    expected_direct = F.l1_loss(
        direct_output.predictions.float(),
        batch["target_cumulative_log_change"].float(),
    )
    torch.testing.assert_close(direct_native, expected_direct)
    torch.testing.assert_close(
        direct_optimisation,
        10000.0 * expected_direct,
    )
    reconstructed_change = raw_to_cumulative_log_change(
        persistence_raw,
        batch["last_context_target"],
    )
    torch.testing.assert_close(
        reconstructed_change,
        direct_output.predictions,
        atol=1.0e-7,
        rtol=0.0,
    )

    # Raw and log-change input views must share the exact direct target.
    log_dataset = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=60,
            horizons=(1, 5, 15, 30, 60),
            stride=15,
            input_representation="context_log_change",
        ),
    )
    log_batch = _batch(log_dataset)
    torch.testing.assert_close(
        batch["target_cumulative_log_change"],
        log_batch["target_cumulative_log_change"],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        batch["y_unnormalised"],
        log_batch["y_unnormalised"],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        batch["context_target_unnormalised"],
        log_batch["context_target_unnormalised"],
        atol=0.0,
        rtol=0.0,
    )

    # Deterministic per-window absolute Close-return correlation.
    graph_context = batch["context_target_unnormalised"][..., 0]
    unthresholded = build_window_absolute_correlation_adjacency(
        graph_context,
        threshold=None,
        num_heads=1,
        add_self_loops=False,
        empty_row_policy="strongest",
    )
    thresholded = build_window_absolute_correlation_adjacency(
        graph_context,
        threshold=0.18,
        num_heads=1,
        add_self_loops=False,
        empty_row_policy="strongest",
    )
    for adjacency in (unthresholded, thresholded):
        if tuple(adjacency.shape) != (2, 1, 4, 4):
            raise AssertionError(
                "Unexpected dynamic-correlation graph shape."
            )
        torch.testing.assert_close(
            adjacency.sum(dim=-1),
            torch.ones(2, 1, 4),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.diagonal(adjacency, dim1=-2, dim2=-1),
            torch.zeros(2, 1, 4),
            atol=0.0,
            rtol=0.0,
        )
    if int((thresholded > 0).sum()) > int((unthresholded > 0).sum()):
        raise AssertionError(
            "Thresholding increased dynamic-correlation support."
        )
    if torch.allclose(
        unthresholded[0],
        unthresholded[1],
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise AssertionError(
            "Per-window correlation graphs did not change across windows."
        )

    dynamic_correlation_config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=60,
        horizons=(1, 5, 15, 30, 60),
        temporal=temporal,
        graph=GraphConfig(
            type="dynamic_correlation",
            num_heads=1,
            hidden_dim=16,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
        dynamic_correlation_threshold=0.18,
        dynamic_correlation_empty_row_policy="strongest",
        spatial_gate_type="learned_scalar",
        spatial_gate_initial_beta=0.5,
    )
    dynamic_correlation_model = ContinuousForecaster(
        dynamic_correlation_config
    )
    runner_graph_context = _graph_context_values(
        batch,
        model=dynamic_correlation_model,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(
        runner_graph_context,
        graph_context,
        atol=0.0,
        rtol=0.0,
    )
    dynamic_correlation_output = dynamic_correlation_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
        graph_context_values=runner_graph_context,
    )
    torch.testing.assert_close(
        dynamic_correlation_output.graph.selected,
        thresholded,
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        dynamic_correlation_output.graph.dynamic,
        thresholded,
        atol=1.0e-6,
        rtol=0.0,
    )
    if any(
        parameter.requires_grad
        for parameter in dynamic_correlation_model.graph_learner.parameters()
    ):
        raise AssertionError(
            "Deterministic dynamic correlation unexpectedly has graph "
            "parameters."
        )

    # Asset independence before graph.
    changed_x = batch["x"].clone()
    changed_x[:, :, 3] += 10.0
    with torch.no_grad():
        changed_output = model(
            changed_x,
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
    torch.testing.assert_close(
        output.predictions[:, :, :3],
        changed_output.predictions[:, :, :3],
        atol=1.0e-6,
        rtol=0.0,
    )

    # Explicit graph path and forecasting gradients.
    graph_config = ContinuousForecasterConfig(
        num_nodes=4,
        context_length=60,
        horizons=(1, 5, 15, 30, 60),
        temporal=temporal,
        graph=GraphConfig(
            type="free_static",
            num_heads=2,
            hidden_dim=16,
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
        spatial_gate_type="learned_scalar",
        spatial_gate_initial_beta=0.1,
    )
    graph_model = ContinuousForecaster(graph_config)
    graph_output = graph_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    graph_loss = F.mse_loss(
        graph_output.predictions,
        batch["y"],
    )
    graph_loss.backward()
    gradient = graph_model.graph_learner.logits.grad
    if gradient is None or float(gradient.norm().item()) <= 0.0:
        raise AssertionError("Forecast loss did not reach graph logits.")
    if graph_output.spatial_beta is None:
        raise AssertionError("Learned spatial gate did not expose beta.")
    torch.testing.assert_close(
        graph_output.spatial_beta.float(),
        torch.tensor(0.1),
        atol=1.0e-6,
        rtol=0.0,
    )
    if (
        graph_model.spatial_gate is None
        or graph_model.spatial_gate.raw_beta is None
        or graph_model.spatial_gate.raw_beta.grad is None
        or float(graph_model.spatial_gate.raw_beta.grad.abs().item()) <= 0.0
    ):
        raise AssertionError("Forecast loss did not reach the spatial gate.")

    graph_model.zero_grad(set_to_none=True)
    graph_output = graph_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    graph_regularisation = compute_graph_regularisation(
        graph_output.graph,
        config=GraphRegularisationConfig(
            graph_target_entropy=0.5,
            graph_target_entropy_reg=0.05,
        ),
        current_epoch=1,
        reference_tensor=graph_output.predictions.sum() * 0.0,
    )
    graph_regularisation.total.backward()
    regularisation_gradient = graph_model.graph_learner.logits.grad
    if (
        regularisation_gradient is None
        or float(regularisation_gradient.norm().item()) <= 0.0
    ):
        raise AssertionError(
            "Target-entropy regularisation did not reach graph logits."
        )


    # Dedicated optimizer parameter groups and schedule-ratio preservation.
    optimizer_config = {
        "training": {
            "optimizer": "adam",
            "learning_rate": 1.0e-4,
            "graph_learning_rate": 2.0e-3,
            "weight_decay": 0.0,
            "scheduler": "modern_tcn_type3",
        }
    }
    optimizer = _build_optimizer(graph_model, optimizer_config)
    learning_rates = _current_learning_rates(optimizer)
    if not math.isclose(float(learning_rates["backbone"]), 1.0e-4):
        raise AssertionError("Unexpected backbone learning rate.")
    if not math.isclose(float(learning_rates["graph"]), 2.0e-3):
        raise AssertionError("Unexpected graph learning rate.")
    backbone_parameters, graph_parameters = _trainable_parameter_partition(
        graph_model
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_parameter_ids = {
        id(parameter)
        for parameter in (*backbone_parameters, *graph_parameters)
    }
    if optimizer_parameter_ids != expected_parameter_ids:
        raise AssertionError("Optimizer parameter groups are incomplete.")
    scheduled = _adjust_learning_rate(
        optimizer,
        config=optimizer_config,
        completed_epoch=5,
    )
    expected_multiplier = 0.9 ** 2
    if not math.isclose(
        float(scheduled["backbone"]),
        1.0e-4 * expected_multiplier,
        rel_tol=1.0e-12,
    ):
        raise AssertionError("Backbone scheduler multiplier is incorrect.")
    if not math.isclose(
        float(scheduled["graph"]),
        2.0e-3 * expected_multiplier,
        rel_tol=1.0e-12,
    ):
        raise AssertionError("Graph scheduler did not preserve the LR ratio.")

    # Convergence contract: a meaningful graph LR must move a near-uniform
    # softmax graph toward a low target entropy. A non-zero gradient alone is
    # insufficient and was the failure mode this change fixes.
    entropy_model = ContinuousForecaster(graph_config)
    entropy_optimizer = _build_optimizer(entropy_model, optimizer_config)
    fixed_hidden = torch.randn(2, 3, 4, 16)
    with torch.no_grad():
        initial_graph = entropy_model.graph_learner(fixed_hidden)
        initial_values = initial_graph.selected.clamp_min(1.0e-12)
        initial_entropy = float(
            (-(initial_values * initial_values.log()).sum(dim=-1))
            .mean()
            .item()
        )
    for _ in range(300):
        entropy_optimizer.zero_grad(set_to_none=True)
        graph_output_only = entropy_model.graph_learner(fixed_hidden)
        regularisation = compute_graph_regularisation(
            graph_output_only,
            config=GraphRegularisationConfig(
                graph_target_entropy=0.35,
                graph_target_entropy_reg=1.0,
            ),
            current_epoch=1,
            reference_tensor=entropy_model.graph_learner.logits.sum() * 0.0,
        )
        regularisation.total.backward()
        entropy_optimizer.step()
    with torch.no_grad():
        final_graph = entropy_model.graph_learner(fixed_hidden)
        final_values = final_graph.selected.clamp_min(1.0e-12)
        final_entropy = float(
            (-(final_values * final_values.log()).sum(dim=-1))
            .mean()
            .item()
        )
    if final_entropy >= initial_entropy - 0.25:
        raise AssertionError(
            "Dedicated graph LR did not materially reduce graph entropy: "
            f"initial={initial_entropy:.6f}, final={final_entropy:.6f}."
        )

    # AMP graph-precision regression: fixed row-stochastic graphs must remain
    # float32 first-class outputs even when the temporal representation is
    # low precision. This prevents false stochasticity failures under CUDA AMP.
    fixed_config = GraphConfig(
        type="fixed",
        num_heads=1,
        hidden_dim=16,
        activation="softmax",
        add_self_loops=False,
        mtgnn_top_k=2,
        base_graph_type="free_static",
        gate_type="none",
        initial_alpha=0.25,
    )
    fixed_adjacency = torch.tensor(
        [
            [0.0, 0.2, 0.3, 0.5],
            [0.1, 0.0, 0.6, 0.3],
            [0.7, 0.2, 0.0, 0.1],
            [0.25, 0.25, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    fixed_learner = FixedGraphLearner(
        config=fixed_config,
        num_nodes=4,
        adjacency=fixed_adjacency,
    )
    low_precision_hidden = torch.randn(
        2,
        3,
        4,
        16,
        dtype=torch.float16,
    )
    fixed_output = fixed_learner(low_precision_hidden)
    if fixed_output.selected is None:
        raise AssertionError("Fixed graph output is missing.")
    if fixed_output.selected.dtype != torch.float32:
        raise AssertionError(
            "Fixed graph probabilities must remain float32 under AMP."
        )
    torch.testing.assert_close(
        fixed_output.selected.sum(dim=-1),
        torch.ones(2, 1, 4),
        atol=1.0e-6,
        rtol=0.0,
    )

    # Dynamic graph and dynamic-base graph contracts.
    for graph_type in ("dynamic", "dynamic_base"):
        dynamic_config = ContinuousForecasterConfig(
            num_nodes=4,
            context_length=60,
            horizons=(1, 5, 15, 30, 60),
            temporal=temporal,
            graph=GraphConfig(
                type=graph_type,
                num_heads=1,
                hidden_dim=16,
                add_self_loops=False,
                mtgnn_top_k=2,
                base_graph_type="free_static",
                gate_type=(
                    "learned_scalar"
                    if graph_type == "dynamic_base"
                    else "none"
                ),
                initial_alpha=0.25,
            ),
            spatial_gate_type="learned_scalar",
            spatial_gate_initial_beta=0.1,
        )
        dynamic_model = ContinuousForecaster(dynamic_config)
        dynamic_output = dynamic_model(
            batch["x"],
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
        if tuple(dynamic_output.graph.selected.shape) != (2, 1, 4, 4):
            raise AssertionError(
                f"Unexpected {graph_type} graph shape."
            )
        torch.testing.assert_close(
            dynamic_output.graph.selected.sum(dim=-1),
            torch.ones(2, 1, 4),
            atol=1.0e-6,
            rtol=0.0,
        )
        dynamic_loss = F.mse_loss(dynamic_output.predictions, batch["y"])
        dynamic_loss.backward()
        q_gradient = dynamic_model.graph_learner.q_proj.weight.grad
        if q_gradient is None or float(q_gradient.norm().item()) <= 0.0:
            raise AssertionError(
                f"Forecast loss did not reach {graph_type} Q projection."
            )
        if graph_type == "dynamic_base":
            alpha = dynamic_model.dynamic_graph_alpha()
            if alpha is None:
                raise AssertionError("Dynamic-base alpha is missing.")
            torch.testing.assert_close(
                alpha.float().mean(),
                torch.tensor(0.25),
                atol=1.0e-6,
                rtol=0.0,
            )
            raw_alpha = dynamic_model.graph_learner.dynamic_residual_raw
            if raw_alpha is None or raw_alpha.grad is None:
                raise AssertionError(
                    "Forecast loss did not reach dynamic-base alpha."
                )

    # Optional official ModernTCN shape check when the submodule is present.
    modern_root = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ModernTCN"
        / "ModernTCN-Long-term-forecasting"
    )
    if modern_root.is_dir() and any(modern_root.iterdir()):
        modern_config = ContinuousForecasterConfig(
            num_nodes=4,
            context_length=60,
            horizons=(1, 5, 15, 30, 60),
            temporal=ContinuousTemporalConfig(
                type="modern_tcn",
                d_model=16,
                session_position_encoding=True,
                patch_size=8,
                patch_stride=4,
                modern_tcn_ffn_ratio=1,
                modern_tcn_num_blocks=1,
                modern_tcn_large_kernel=15,
                modern_tcn_small_kernel=3,
                modern_tcn_dropout=0.0,
                modern_tcn_head_dropout=0.0,
            ),
            graph=GraphConfig(
                type="none",
                num_heads=2,
                hidden_dim=16,
                add_self_loops=False,
                mtgnn_top_k=2,
            ),
        )
        modern_model = ContinuousForecaster(modern_config).eval()
        modern_output = modern_model(
            batch["x"],
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
        if tuple(modern_output.temporal_hidden.shape) != (2, 15, 4, 16):
            raise AssertionError("Unexpected ModernTCN hidden shape.")
        if tuple(modern_output.predictions.shape) != (2, 5, 4, 1):
            raise AssertionError("Unexpected ModernTCN forecast shape.")

        # No-graph parity against the complete selected official path.
        backbone = modern_model.temporal_backbone
        per_asset = (
            batch["x"]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(2 * 4, 60, 5)
        )
        session_features = backbone.__class__.__module__  # keep mypy quiet
        del session_features
        from src.models.continuous_forecaster import (
            build_context_session_features,
            build_modern_tcn_patch_features,
        )
        time_features = build_context_session_features(
            context_start=batch["context_start"],
            session_length=batch["session_length"],
            context_length=60,
            device=per_asset.device,
            dtype=per_asset.dtype,
        )
        patch_features = build_modern_tcn_patch_features(
            time_features,
            patch_size=8,
            patch_stride=4,
        ).repeat_interleave(4, dim=0)
        with torch.no_grad():
            official_all = backbone.official_model(
                per_asset,
                patch_features,
            )
        official_close = (
            official_all[:, :, 3]
            .reshape(2, 4, 5)
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .contiguous()
        )
        torch.testing.assert_close(
            modern_output.predictions,
            official_close,
            atol=1.0e-6,
            rtol=0.0,
        )

        changed_modern_x = batch["x"].clone()
        changed_modern_x[:, :, 3] += 10.0
        with torch.no_grad():
            changed_modern = modern_model(
                changed_modern_x,
                context_start=batch["context_start"],
                session_length=batch["session_length"],
            )
        torch.testing.assert_close(
            modern_output.predictions[:, :, :3],
            changed_modern.predictions[:, :, :3],
            atol=1.0e-6,
            rtol=0.0,
        )

    print("Continuous forecasting contract tests passed.")


if __name__ == "__main__":
    main()

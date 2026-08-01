from __future__ import annotations

"""Fast contract tests for the continuous temporal/graph forecasting path."""

from pathlib import Path

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
from src.training.run_continuous_forecaster import _loss_values


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

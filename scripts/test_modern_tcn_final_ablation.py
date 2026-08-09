from __future__ import annotations

import math

import torch

from src.training.modern_tcn_final_two_runs_specs import (
    DEFAULT_HORIZON_REFERENCE_MAE,
    make_final_two_run_specs,
)
from src.training.run_modern_tcn_final_ablation import (
    _autoregressive_rollout_uses_amp,
    _normalise_raw_context,
    _select_rollout_horizons,
    _synthetic_next_candle,
)


def test_final_specs() -> None:
    autoreg, weighted = make_final_two_run_specs()
    assert autoreg.config["training"]["forecast_strategy"] == "autoregressive"
    assert (
        autoreg.config["training"][
            "autoregressive_rollout_mixed_precision"
        ]
        is False
    )
    assert autoreg.config["training"]["one_step_training_stride"] == 1
    assert autoreg.config["data"]["horizons"] == [1, 5, 15, 30, 60]
    assert weighted.config["training"]["forecast_strategy"] == "parallel_weighted"
    weights = weighted.config["training"]["loss"]["horizon_weights"]
    assert len(weights) == 5
    expected_mean = sum(DEFAULT_HORIZON_REFERENCE_MAE) / len(
        DEFAULT_HORIZON_REFERENCE_MAE
    )
    for weight, reference in zip(weights, DEFAULT_HORIZON_REFERENCE_MAE, strict=True):
        assert math.isclose(weight, expected_mean / reference, rel_tol=1e-12)
    assert "a0p5_b0p5" in autoreg.run_name
    assert "a0p5_b0p5" in weighted.run_name


def test_rollout_helpers() -> None:
    torch.manual_seed(11)
    raw = 10.0 + torch.rand(2, 60, 3, 5)
    channels = ("open", "high", "low", "close", "volume")
    x, mean, std = _normalise_raw_context(
        raw,
        input_channels=channels,
        target_channel="close",
        eps=1e-8,
        clip=False,
        clip_min=-5.0,
        clip_max=5.0,
    )
    assert tuple(x.shape) == tuple(raw.shape)
    assert tuple(mean.shape) == (2, 3, 1)
    assert tuple(std.shape) == (2, 3, 1)
    torch.testing.assert_close(x.mean(dim=1), torch.zeros_like(x.mean(dim=1)), atol=1e-5, rtol=0)

    next_close = raw[:, -1, :, 3:4] * 1.001
    next_candle = _synthetic_next_candle(
        raw,
        next_close,
        input_channels=channels,
        eps=1e-8,
    )
    assert tuple(next_candle.shape) == (2, 3, 5)
    torch.testing.assert_close(next_candle[:, :, 0], raw[:, -1, :, 3])
    torch.testing.assert_close(next_candle[:, :, 3], next_close.squeeze(-1))
    assert torch.all(next_candle[:, :, 1] >= next_candle[:, :, 0])
    assert torch.all(next_candle[:, :, 1] >= next_candle[:, :, 3])
    assert torch.all(next_candle[:, :, 2] <= next_candle[:, :, 0])
    assert torch.all(next_candle[:, :, 2] <= next_candle[:, :, 3])

    dense = torch.arange(2 * 60 * 3 * 1).view(2, 60, 3, 1)
    selected = _select_rollout_horizons(dense, (1, 5, 15, 30, 60))
    assert tuple(selected.shape) == (2, 5, 3, 1)
    torch.testing.assert_close(selected[:, 0], dense[:, 0])
    torch.testing.assert_close(selected[:, -1], dense[:, 59])



def test_rollout_precision_contract() -> None:
    autoreg, _ = make_final_two_run_specs()
    config = autoreg.config
    assert not _autoregressive_rollout_uses_amp(
        config,
        training_amp_enabled=True,
    )
    assert not _autoregressive_rollout_uses_amp(
        config,
        training_amp_enabled=False,
    )

    opt_in = {
        **config,
        "training": {
            **config["training"],
            "autoregressive_rollout_mixed_precision": True,
        },
    }
    assert _autoregressive_rollout_uses_amp(
        opt_in,
        training_amp_enabled=True,
    )
    assert not _autoregressive_rollout_uses_amp(
        opt_in,
        training_amp_enabled=False,
    )


def main() -> None:
    test_final_specs()
    test_rollout_helpers()
    test_rollout_precision_contract()
    print("ModernTCN final ablation contracts passed.")


if __name__ == "__main__":
    main()

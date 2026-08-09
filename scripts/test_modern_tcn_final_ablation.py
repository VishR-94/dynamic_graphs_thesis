from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import torch

from scripts.test_modern_tcn_graph_round1 import _install_fake_modern_tcn
from src.models.modern_tcn_graph_round1 import ModernTCNGraphRound1Model
from src.training.modern_tcn_final_two_runs_specs import (
    DEFAULT_HORIZON_REFERENCE_MAE,
    make_final_two_run_specs,
)
from src.training.run_modern_tcn_final_ablation import (
    AUTOREGRESSIVE_CLOSE_ONLY,
    PARALLEL_WEIGHTED,
    _autoregressive_rollout,
    _autoregressive_rollout_uses_amp,
    _model_config_for_strategy,
    _next_close_from_log_return,
    _normalise_raw_context,
    _one_step_log_return_errors,
    _rollout_step_diagnostics,
    _select_rollout_horizons,
)


def test_final_specs() -> None:
    weighted, autoreg = make_final_two_run_specs()

    # The notebook must run the weighted experiment first.
    assert weighted.config["training"]["forecast_strategy"] == PARALLEL_WEIGHTED
    assert weighted.config["data"]["input_channels"] == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    weights = weighted.config["training"]["loss"]["horizon_weights"]
    assert len(weights) == 5
    expected_mean = sum(DEFAULT_HORIZON_REFERENCE_MAE) / len(
        DEFAULT_HORIZON_REFERENCE_MAE
    )
    for weight, reference in zip(
        weights,
        DEFAULT_HORIZON_REFERENCE_MAE,
        strict=True,
    ):
        assert math.isclose(weight, expected_mean / reference, rel_tol=1e-12)

    assert (
        autoreg.config["training"]["forecast_strategy"]
        == AUTOREGRESSIVE_CLOSE_ONLY
    )
    assert autoreg.config["data"]["input_channels"] == ["close"]
    assert autoreg.config["model"]["output_representation"] == (
        "cumulative_log_change"
    )
    assert autoreg.config["model"]["output_head_initialisation"] == "zero"
    assert autoreg.config["training"]["autoregressive_feedback_channels"] == [
        "close"
    ]
    assert (
        autoreg.config["training"]["autoregressive_rollout_mixed_precision"]
        is False
    )
    assert autoreg.config["training"]["one_step_training_stride"] == 1
    assert autoreg.config["data"]["horizons"] == [1, 5, 15, 30, 60]
    assert "close_only_autoreg" in autoreg.run_name
    assert "a0p5_b0p5" in autoreg.run_name
    assert "a0p5_b0p5" in weighted.run_name

    model_config = _model_config_for_strategy(autoreg.config, num_nodes=3)
    assert model_config.forecaster.horizons == (1,)
    assert model_config.forecaster.input_channels == ("close",)
    assert model_config.forecaster.output_representation == (
        "cumulative_log_change"
    )
    assert model_config.forecaster.output_head_initialisation == "zero"


def test_close_only_model_contract() -> None:
    _install_fake_modern_tcn()
    _, autoreg = make_final_two_run_specs()
    model_config = _model_config_for_strategy(autoreg.config, num_nodes=3)
    prior = torch.ones(3, 3) - torch.eye(3)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    model = ModernTCNGraphRound1Model(
        model_config,
        static_prior=prior,
    )
    x = torch.randn(2, 60, 3, 1)
    output = model(
        x,
        context_start=torch.tensor([0, 15]),
        session_length=torch.tensor([390, 390]),
    )
    assert tuple(output.predictions.shape) == (2, 1, 3, 1)
    # A zero-initialised direct log-return head starts from persistence.
    torch.testing.assert_close(
        output.predictions,
        torch.zeros_like(output.predictions),
        atol=0.0,
        rtol=0.0,
    )
    assert output.graph.selected is not None
    torch.testing.assert_close(
        output.graph.selected.sum(dim=-1),
        torch.ones_like(output.graph.selected.sum(dim=-1)),
        atol=2e-6,
        rtol=0.0,
    )


def test_close_only_normalisation_and_reconstruction() -> None:
    torch.manual_seed(11)
    raw = 10.0 + torch.rand(2, 60, 3, 1, dtype=torch.float64)
    x, mean, std = _normalise_raw_context(
        raw,
        input_channels=("close",),
        target_channel="close",
        eps=1e-8,
        clip=False,
        clip_min=-5.0,
        clip_max=5.0,
    )
    assert tuple(x.shape) == tuple(raw.shape)
    assert tuple(mean.shape) == (2, 3, 1)
    assert tuple(std.shape) == (2, 3, 1)
    torch.testing.assert_close(
        x.mean(dim=1),
        torch.zeros_like(x.mean(dim=1)),
        atol=1e-9,
        rtol=1e-7,
    )

    previous = torch.full((2, 3, 1), 100.0, dtype=torch.float64)
    log_return = torch.full_like(previous, math.log(1.001))
    next_close = _next_close_from_log_return(
        previous,
        log_return,
        step=1,
    )
    torch.testing.assert_close(
        next_close,
        torch.full_like(previous, 100.1),
        atol=1e-7,
        rtol=1e-9,
    )

    try:
        _next_close_from_log_return(
            torch.zeros_like(previous),
            log_return,
            step=1,
        )
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Non-positive previous Close was silently accepted.")


def test_direct_one_step_log_return_loss() -> None:
    last = torch.tensor([[[100.0], [50.0]]])
    true_change = torch.tensor([[[[0.001], [-0.002]]]])
    target = last.unsqueeze(1) * torch.exp(true_change)
    prediction = true_change + 0.0003
    batch = {
        "y_unnormalised": target,
        "last_context_target": last,
    }
    observed_true, absolute_error = _one_step_log_return_errors(
        prediction,
        batch,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(observed_true, true_change)
    torch.testing.assert_close(
        absolute_error,
        torch.full_like(absolute_error, 0.0003),
        atol=2e-7,
        rtol=0,
    )


class _ConstantReturnModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(
        self,
        x: torch.Tensor,
        *,
        context_start: torch.Tensor,
        session_length: torch.Tensor,
    ) -> SimpleNamespace:
        del context_start, session_length
        batch, _, nodes, _ = x.shape
        prediction = torch.full(
            (batch, 1, nodes, 1),
            self.value,
            dtype=x.dtype,
            device=x.device,
        )
        graph = SimpleNamespace(selected=None, base=None, dynamic=None)
        return SimpleNamespace(predictions=prediction, graph=graph)


def test_close_only_rollout_contract() -> None:
    _, autoreg = make_final_two_run_specs()
    config = copy.deepcopy(autoreg.config)
    config["data"]["horizons"] = [1, 2, 3]
    config["training"]["autoregressive_rollout_length"] = 3

    batch = {
        "context_unnormalised": torch.full((2, 60, 3, 1), 100.0),
        "context_start": torch.tensor([0, 15]),
        "session_length": torch.tensor([390, 390]),
    }
    one_step = math.log(1.001)
    close_path, return_path, first_output = _autoregressive_rollout(
        model=_ConstantReturnModel(one_step),  # type: ignore[arg-type]
        batch=batch,
        device=torch.device("cpu"),
        use_amp=False,
        config=config,
    )
    assert first_output is not None
    assert tuple(close_path.shape) == (2, 3, 3, 1)
    assert tuple(return_path.shape) == (2, 3, 3, 1)
    expected = torch.tensor(
        [100.0 * 1.001, 100.0 * 1.001**2, 100.0 * 1.001**3],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        close_path[0, :, 0, 0],
        expected,
        atol=1e-7,
        rtol=1e-9,
    )
    torch.testing.assert_close(
        return_path,
        torch.full_like(return_path, one_step),
        atol=1e-9,
        rtol=1e-7,
    )

    selected = _select_rollout_horizons(close_path, (1, 3))
    assert tuple(selected.shape) == (2, 2, 3, 1)
    torch.testing.assert_close(selected[:, 0], close_path[:, 0])
    torch.testing.assert_close(selected[:, 1], close_path[:, 2])


def test_rollout_diagnostics_contract() -> None:
    last = torch.full((2, 3, 1), 100.0)
    returns = torch.full((2, 3, 3, 1), math.log(1.001))
    close = torch.empty_like(returns)
    close[:, 0] = last * 1.001
    close[:, 1] = last * 1.001**2
    close[:, 2] = last * 1.001**3
    rows = _rollout_step_diagnostics(close, returns, last)
    assert len(rows) == 3
    assert rows[0]["step"] == 1
    assert rows[-1]["step"] == 3
    assert rows[-1]["maximum_predicted_close"] > rows[0][
        "maximum_predicted_close"
    ]


def test_rollout_precision_contract() -> None:
    _, autoreg = make_final_two_run_specs()
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
    test_close_only_model_contract()
    test_close_only_normalisation_and_reconstruction()
    test_direct_one_step_log_return_loss()
    test_close_only_rollout_contract()
    test_rollout_diagnostics_contract()
    test_rollout_precision_contract()
    print("ModernTCN final ablation contracts passed.")


if __name__ == "__main__":
    main()

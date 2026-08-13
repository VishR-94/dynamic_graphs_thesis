from __future__ import annotations

"""Focused contracts for the eight-run batch-size-one ModernTCN grid."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.models.modern_tcn_graph_round1 import PriorMixedDynamicGraphLearner
from src.models.modern_tcn_graph_round2 import Round2WindowGraphLearner
from src.training.final_batch1_modern_tcn_grid_specs import (
    make_final_batch1_modern_tcn_grid_specs,
)
from src.training.run_final_token_v2_experiment import _build_model


EXPECTED_WEIGHTS = (
    3.7295707385901125,
    1.7489229046582941,
    1.039473644407472,
    0.7471142661463033,
    0.5377548425463323,
)


def _uniform_off_diagonal(nodes: int) -> torch.Tensor:
    result = torch.ones(nodes, nodes, dtype=torch.float32)
    result.fill_diagonal_(0.0)
    return result / float(nodes - 1)


def _test_grid_contract() -> None:
    specs = make_final_batch1_modern_tcn_grid_specs()
    observed = [
        (spec.model_space, spec.static_initialisation, spec.graph_activation)
        for spec in specs
    ]
    expected = [
        ("continuous", "uniform", "softmax"),
        ("continuous", "correlation", "softmax"),
        ("continuous", "uniform", "sparsemax"),
        ("continuous", "correlation", "sparsemax"),
        ("token", "uniform", "softmax"),
        ("token", "correlation", "softmax"),
        ("token", "uniform", "sparsemax"),
        ("token", "correlation", "sparsemax"),
    ]
    assert observed == expected
    assert len({spec.run_name for spec in specs}) == 8

    for spec in specs:
        training = spec.config["training"]
        for key in (
            "batch_size",
            "selection_batch_size",
            "export_batch_size",
        ):
            assert int(training[key]) == 1, (spec.run_name, key)
        if "validation_batch_size" in training:
            assert int(training["validation_batch_size"]) == 1

        graph = spec.config["model"]["graph"]
        assert str(graph["activation"]) == spec.graph_activation
        assert int(graph.get("num_heads", 1)) == 1
        assert int(graph.get("hidden_dim", 32)) == 32
        assert float(graph["initial_alpha"]) == 0.5
        assert float(spec.config["model"]["spatial"]["initial_beta"]) == 0.5
        assert spec.config["model"]["temporal" if spec.model_space == "continuous" else "temporal_stack"]

        prior = spec.config["model"]["prior"]
        assert str(prior["type"]) == spec.static_initialisation
        if spec.static_initialisation == "uniform":
            assert float(prior["jitter"]) == 0.0
        else:
            assert float(prior["jitter"]) == 0.02
            assert prior.get("threshold") is None

        if spec.model_space == "continuous":
            assert spec.config["model"]["variant"] in {
                "uniform_static_mixture_state",
                "prior_mixture_state",
            }
            loss = training["loss"]
            assert loss["type"] == "cumulative_log_change_mae"
            assert loss["horizon_weighting"] == "inverse_reference_mae"
            assert tuple(float(v) for v in loss["horizon_weights"]) == EXPECTED_WEIGHTS
        else:
            assert spec.config["model_kind"] == "modern_tcn_token"
            assert spec.config["model"]["graph_family"] == "prior_state"
            loss = training["loss"]
            assert loss == {
                "type": "coarse_s1_cross_entropy",
                "horizon_weighting": "uniform",
                "dense_origins": False,
            }
            assert training["selection_metric"] == (
                "mean_top1_accuracy_over_all_60_future_steps"
            )


def _test_uniform_static_initialisation() -> None:
    nodes = 7
    expected = _uniform_off_diagonal(nodes)
    for activation in ("softmax", "sparsemax"):
        continuous = PriorMixedDynamicGraphLearner(
            d_model=8,
            num_nodes=nodes,
            num_heads=1,
            graph_hidden_dim=8,
            use_state_pathway=True,
            static_prior=None,
            initial_alpha=0.5,
            prior_scale=4.0,
            prior_jitter=0.0,
            prior_seed=42,
            graph_activation=activation,
            use_static_graph=True,
            random_static_initialisation=True,
        )
        observed_continuous = continuous.static_adjacency()
        assert observed_continuous is not None
        torch.testing.assert_close(
            observed_continuous[0, 0], expected, atol=1.0e-7, rtol=0.0
        )

        token = Round2WindowGraphLearner(
            d_model=8,
            num_nodes=nodes,
            num_heads=1,
            graph_hidden_dim=8,
            activation=activation,
            graph_family="prior_state",
            static_prior=None,
            initial_alpha=0.5,
            prior_scale=4.0,
            prior_jitter=0.0,
            prior_seed=42,
        )
        observed_token = token.static_adjacency()
        assert observed_token is not None
        torch.testing.assert_close(
            observed_token[0, 0], expected, atol=1.0e-7, rtol=0.0
        )


class _CapturedTokenModel:
    last_static_prior = object()

    def __init__(self, config, *, static_prior):
        self.config = config
        type(self).last_static_prior = static_prior

    def to(self, device):
        del device
        return self


def _test_token_prior_routing() -> None:
    specs = make_final_batch1_modern_tcn_grid_specs()
    uniform = next(
        spec
        for spec in specs
        if spec.model_space == "token"
        and spec.static_initialisation == "uniform"
        and spec.graph_activation == "softmax"
    )
    correlation = next(
        spec
        for spec in specs
        if spec.model_space == "token"
        and spec.static_initialisation == "correlation"
        and spec.graph_activation == "softmax"
    )
    dataset = SimpleNamespace(num_assets=3, asset_cols=["A", "B", "C"])
    train_split = {"asset_cols": ["A", "B", "C"], "samples": []}

    with patch(
        "src.training.run_final_token_v2_experiment.ModernTCNGraphRound2TokenModel",
        _CapturedTokenModel,
    ), patch(
        "src.training.run_final_token_v2_experiment.build_absolute_correlation_graph_prior",
        side_effect=AssertionError("Uniform model must not build correlation."),
    ):
        _build_model(
            config=uniform.config,
            token_dataset=dataset,
            train_split=train_split,
            device=torch.device("cpu"),
        )
    assert _CapturedTokenModel.last_static_prior is None

    sentinel = torch.eye(3)
    with patch(
        "src.training.run_final_token_v2_experiment.ModernTCNGraphRound2TokenModel",
        _CapturedTokenModel,
    ), patch(
        "src.training.run_final_token_v2_experiment.build_absolute_correlation_graph_prior",
        return_value=sentinel,
    ) as builder:
        _build_model(
            config=correlation.config,
            token_dataset=dataset,
            train_split=train_split,
            device=torch.device("cpu"),
        )
    builder.assert_called_once()
    assert _CapturedTokenModel.last_static_prior is sentinel


def main() -> None:
    _test_grid_contract()
    _test_uniform_static_initialisation()
    _test_token_prior_routing()
    print("Final batch-1 ModernTCN grid contracts passed.")


if __name__ == "__main__":
    main()

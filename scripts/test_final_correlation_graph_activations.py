from __future__ import annotations

"""Focused contracts for the final correlation-prior activation controls."""

from pathlib import Path
import sys
import types

import torch
from torch import nn

from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Model,
    PriorMixedDynamicGraphLearner,
    round1_model_config_from_mapping,
)
from src.training.final_correlation_graph_activation_specs import (
    make_final_correlation_graph_activation_specs,
)
from src.training.modern_tcn_final_two_runs_specs import make_final_two_run_specs
from src.training.run_modern_tcn_graph_round1 import _validate_config


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
                        values[..., -1:].expand(
                            *values.shape[:-1], self.padding
                        ),
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


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_flatten(item, prefix + (str(key),)))
        return result
    return {prefix: value}


def _assert_graph(values: torch.Tensor) -> None:
    if not torch.isfinite(values).all():
        raise AssertionError("Graph contains non-finite values.")
    if torch.any(values < 0):
        raise AssertionError("Graph contains negative values.")
    torch.testing.assert_close(
        values.sum(dim=-1),
        torch.ones_like(values.sum(dim=-1)),
        atol=1.0e-5,
        rtol=0.0,
    )
    diagonal = torch.diagonal(values, dim1=-2, dim2=-1)
    torch.testing.assert_close(
        diagonal,
        torch.zeros_like(diagonal),
        atol=0.0,
        rtol=0.0,
    )


def _test_specs() -> tuple:
    specs = make_final_correlation_graph_activation_specs()
    if len(specs) != 2:
        raise AssertionError("Expected sparsemax and entmax15 specifications.")
    observed = tuple(
        spec.config["model"]["graph"]["activation"] for spec in specs
    )
    if observed != ("sparsemax", "entmax15"):
        raise AssertionError(f"Unexpected activation order: {observed}.")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise AssertionError("Activation run names are not unique.")

    reference, _ = make_final_two_run_specs(prior_type="correlation")
    reference_flat = _flatten(reference.config)
    for spec in specs:
        if spec.variant != "prior_mixture_state":
            raise AssertionError("The selected state-aware variant changed.")
        if spec.prior_type != "correlation":
            raise AssertionError("The correlation prior was not preserved.")
        _validate_config(spec.config)
        candidate_flat = _flatten(spec.config)
        changed = {
            key
            for key in set(reference_flat) | set(candidate_flat)
            if reference_flat.get(key) != candidate_flat.get(key)
        }
        if changed != {("model", "graph", "activation")}:
            raise AssertionError(
                f"Unexpected controlled changes for {spec.run_name}: "
                f"{sorted(changed)}"
            )
    return specs


def _test_graph_learners() -> None:
    torch.manual_seed(7)
    temporal = torch.randn(3, 5, 6, 8)
    state = torch.randn_like(temporal)
    prior = torch.rand(6, 6)
    prior.fill_diagonal_(0.0)
    prior = prior / prior.sum(dim=-1, keepdim=True)

    for activation in ("sparsemax", "entmax15"):
        learner = PriorMixedDynamicGraphLearner(
            d_model=8,
            num_nodes=6,
            num_heads=1,
            graph_hidden_dim=8,
            graph_activation=activation,
            use_state_pathway=True,
            use_static_graph=True,
            static_prior=prior,
            random_static_initialisation=False,
            initial_alpha=0.5,
            prior_scale=4.0,
            prior_jitter=0.02,
            prior_seed=42,
        )
        output = learner(temporal, state_hidden=state)
        if output.base is None or output.alpha is None:
            raise AssertionError(f"{activation} lost the static branch.")
        _assert_graph(output.base)
        _assert_graph(output.dynamic)
        _assert_graph(output.selected)

        loss = output.selected.square().mean()
        loss.backward()
        for name, parameter in (
            ("query", learner.q_proj.weight),
            ("key", learner.k_proj.weight),
            ("static", learner.static_logits),
            ("alpha", learner.raw_alpha),
        ):
            if parameter is None or parameter.grad is None:
                raise AssertionError(
                    f"{activation} produced no {name} gradient."
                )
            if not torch.isfinite(parameter.grad).all():
                raise AssertionError(
                    f"{activation} produced a non-finite {name} gradient."
                )


def _test_model_forward(specs) -> None:
    _install_fake_modern_tcn()
    torch.manual_seed(11)
    static_prior = torch.rand(4, 4)
    static_prior.fill_diagonal_(0.0)
    static_prior = static_prior / static_prior.sum(dim=-1, keepdim=True)
    batch = {
        "x": torch.randn(2, 60, 4, 5),
        "context_start": torch.zeros(2, dtype=torch.long),
        "session_length": torch.full((2,), 390, dtype=torch.long),
    }

    for spec in specs:
        config = round1_model_config_from_mapping(spec.config, num_nodes=4)
        model = ModernTCNGraphRound1Model(
            config,
            static_prior=static_prior,
        )
        output = model(
            batch["x"],
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
        if tuple(output.predictions.shape) != (2, 5, 4, 1):
            raise AssertionError("Unexpected five-horizon prediction shape.")
        _assert_graph(output.graph.selected)
        _assert_graph(output.graph.dynamic)
        if output.graph.base is None:
            raise AssertionError("Correlation-prior model lost its base graph.")
        _assert_graph(output.graph.base)
        output.predictions.square().mean().backward()
        if model.graph_learner.static_logits.grad is None:
            raise AssertionError("Forecast loss did not reach static logits.")
        if model.graph_learner.raw_alpha.grad is None:
            raise AssertionError("Forecast loss did not reach alpha.")


def main() -> None:
    specs = _test_specs()
    _test_graph_learners()
    _test_model_forward(specs)
    print("Final correlation graph-activation contracts passed.")


if __name__ == "__main__":
    main()

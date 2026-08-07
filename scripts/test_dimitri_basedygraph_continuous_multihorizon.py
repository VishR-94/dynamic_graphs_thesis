from __future__ import annotations

"""CPU contracts for Dimitri-V2 direct multi-horizon continuous forecasting."""

import torch

from src.models.dimitri_basedygraph_v2 import (
    dimitri_continuous_multi_horizon_parameter_count,
    initialise_base_graphs_from_prior,
    instantiate_dimitri_continuous_multi_horizon_model,
    instantiate_dimitri_continuous_to_price_model,
    parameter_count,
    resolved_per_block_contract,
)
from src.training.run_dimitri_basedygraph_continuous_multihorizon import (
    _multi_horizon_loss,
    _multi_horizon_values,
    _normalise_horizons,
)


HORIZONS = (1, 2, 4)


def _test_horizon_contract() -> None:
    assert _normalise_horizons(HORIZONS, continuation_length=4) == HORIZONS
    for invalid in ((2, 1), (1, 1), (0, 1), (1, 5)):
        try:
            _normalise_horizons(invalid, continuation_length=4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid horizon contract passed: {invalid}")


def _test_model_shape_and_shared_initialisation() -> None:
    torch.manual_seed(101)
    one_step = instantiate_dimitri_continuous_to_price_model().eval()
    torch.manual_seed(101)
    multi = instantiate_dimitri_continuous_multi_horizon_model(
        evaluation_horizons=HORIZONS,
    ).eval()

    assert parameter_count(multi) == dimitri_continuous_multi_horizon_parameter_count(
        len(HORIZONS)
    )
    contract = resolved_per_block_contract(multi.cfg)
    assert contract["activations"] == ["softmax", "softmax", "softmax", "sparsemax"]
    assert contract["num_edge_heads"] == [6, 6, 6, 1]
    assert contract["graph_hidden_dims"] == [192, 192, 192, 96]

    # The complete shared reference backbone and continuous input adapter retain
    # the one-step model's seed-aligned initialisation.
    one_parameters = dict(one_step.backbone.named_parameters())
    multi_parameters = dict(multi.backbone.named_parameters())
    assert one_parameters.keys() == multi_parameters.keys()
    for name in one_parameters:
        torch.testing.assert_close(
            one_parameters[name],
            multi_parameters[name],
            atol=0.0,
            rtol=0.0,
        )
    torch.testing.assert_close(
        one_step.next_close_head.weight[0],
        multi.future_close_head.weight[0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        one_step.next_close_head.bias[0],
        multi.future_close_head.bias[0],
        atol=0.0,
        rtol=0.0,
    )

    context = torch.randn(1, 93, 4, 6)
    context[..., 5] = 0.0
    with torch.no_grad():
        output = multi(context)
    assert tuple(output["future_close_normalised"].shape) == (1, 3, 93, 1)
    assert tuple(output["spatial_repr"].shape) == (1, 4, 93, 96)
    expected_heads = (6, 6, 6, 1)
    for graph, heads in zip(output["block_graph_attns"], expected_heads, strict=True):
        assert tuple(graph.shape) == (1, 4, heads, 93, 93)
        torch.testing.assert_close(
            graph.float().sum(dim=-1),
            torch.ones_like(graph.float().sum(dim=-1)),
            atol=2.0e-5,
            rtol=0.0,
        )
    assert (output["block_graph_attns"][-1] == 0).any()

    prior = torch.ones(93, 93)
    prior.fill_diagonal_(0.0)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    prior_result = initialise_base_graphs_from_prior(
        multi,
        prior,
        scale=4.0,
        jitter=0.02,
        seed=0,
    )
    assert prior_result["count"] == 4


def _test_loss_targets_and_graph_gradients() -> None:
    torch.manual_seed(23)
    model = instantiate_dimitri_continuous_multi_horizon_model(
        evaluation_horizons=HORIZONS,
    ).train()
    context = torch.randn(1, 93, 4, 6)
    context[..., 5] = 0.0
    raw_close = 100.0 * torch.exp(torch.randn(1, 93, 8) * 0.001)
    close_mean = raw_close[:, :, :4].mean(dim=-1)
    close_std = raw_close[:, :, :4].std(dim=-1).clamp_min(1.0e-3)
    batch = {
        "raw_close": raw_close,
        "close_mean": close_mean,
        "close_std": close_std,
    }

    output = model(context)
    predicted, true, last = _multi_horizon_values(
        output=output,
        batch=batch,
        context_length=4,
        horizons=HORIZONS,
        device=torch.device("cpu"),
    )
    assert tuple(predicted.shape) == (1, 3, 93, 1)
    assert tuple(true.shape) == (1, 3, 93, 1)
    assert tuple(last.shape) == (1, 93, 1)
    torch.testing.assert_close(true[:, 0, :, 0], raw_close[:, :, 4])
    torch.testing.assert_close(true[:, 1, :, 0], raw_close[:, :, 5])
    torch.testing.assert_close(true[:, 2, :, 0], raw_close[:, :, 7])

    objective, native, per_horizon = _multi_horizon_loss(
        output=output,
        batch=batch,
        context_length=4,
        horizons=HORIZONS,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(objective)
    assert torch.isfinite(native)
    assert tuple(per_horizon.shape) == (3,)
    torch.testing.assert_close(objective, native * 10_000.0)
    torch.testing.assert_close(native, per_horizon.mean())
    objective.backward()

    graph_gradient = 0.0
    for block in model.backbone.st_blocks:
        for parameter in block.graph_scorer.parameters():
            if parameter.grad is not None:
                graph_gradient += float(parameter.grad.detach().abs().sum().item())
    assert graph_gradient > 0.0


def main() -> None:
    _test_horizon_contract()
    _test_model_shape_and_shared_initialisation()
    _test_loss_targets_and_graph_gradients()
    print("Dimitri V2 continuous direct multi-horizon contracts passed.")


if __name__ == "__main__":
    main()

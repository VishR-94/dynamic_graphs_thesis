from __future__ import annotations

"""CPU contracts for the Dimitri-V2 token-input/direct-price diagnostic."""

from pathlib import Path
from tempfile import TemporaryDirectory
import csv

import torch

import src.data.dimitri_token_price as token_price_data
from src.data.dimitri_token_price import (
    DimitriTokenPriceWindowSpec,
    exact_window_starts,
    load_token_price_splits,
    normalise_split_mode,
)
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT,
    build_absolute_correlation_prior,
    build_sector_prior,
    initialise_base_graphs_from_prior,
    instantiate_dimitri_token_to_price_model,
    parameter_count,
    resolved_per_block_contract,
)
from src.training.run_dimitri_basedygraph_token_price import (
    _dense_one_step_loss,
    _origin_values,
)


ASSETS = [f"A{i:02d}" for i in range(93)]


def _fake_clean_train_split() -> dict:
    torch.manual_seed(11)
    samples = []
    base = torch.linspace(20.0, 200.0, 93)
    for day in range(3):
        innovations = torch.randn(40, 93) * 0.001
        close = base[None] * torch.exp(innovations.cumsum(dim=0))
        values = torch.zeros(40, 93, 6)
        values[..., 0] = close
        values[..., 1] = close * 1.001
        values[..., 2] = close * 0.999
        values[..., 3] = close
        values[..., 4] = 1_000.0
        values[..., 5] = 0.0
        samples.append((values, None, f"2024-01-{day + 2:02d}"))
    return {
        "samples": samples,
        "asset_cols": ASSETS,
        "channels": ["open", "high", "low", "close", "volume", "amount"],
    }


def _test_split_mode_contracts() -> None:
    assert normalise_split_mode("PHYSICAL") == "physical"
    assert normalise_split_mode(" canonical ") == "canonical"
    try:
        normalise_split_mode("wrong")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid split mode was accepted.")

    train = _fake_clean_train_split()
    validation = {**train, "samples": list(train["samples"][:1])}
    test = {**train, "samples": list(train["samples"][1:])}

    original_canonical = token_price_data.load_candle_splits
    original_physical = token_price_data.load_physical_candle_split
    try:
        token_price_data.load_candle_splits = lambda _path: (
            train,
            validation,
            test,
        )
        token_price_data.load_physical_candle_split = (
            lambda _path, split: {
                "train": train,
                "val": validation,
                "test": test,
            }[split]
        )
        canonical = load_token_price_splits(
            "/unused",
            split_mode="canonical",
        )
        physical = load_token_price_splits(
            "/unused",
            split_mode="physical",
        )
    finally:
        token_price_data.load_candle_splits = original_canonical
        token_price_data.load_physical_candle_split = original_physical

    assert len(canonical["train"]["samples"]) == 3
    assert len(canonical["val"]["samples"]) == 1
    assert len(canonical["test"]["samples"]) == 2
    assert len(physical["train"]["samples"]) == 3


def _test_window_contracts() -> None:
    c180 = DimitriTokenPriceWindowSpec(180, 30, 30)
    c60 = DimitriTokenPriceWindowSpec(60, 30, 30)
    assert c180.sequence_length == 210
    assert c60.sequence_length == 90
    assert exact_window_starts(390, c180) == [0, 30, 60, 90, 120, 150, 180]
    assert exact_window_starts(390, c60) == list(range(0, 301, 30))


def _test_priors() -> None:
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "company_profiles.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["Ticker", "Sector", "Industry"],
            )
            writer.writeheader()
            for index, ticker in enumerate(ASSETS):
                writer.writerow(
                    {
                        "Ticker": ticker,
                        "Sector": f"Sector{index // 10}",
                        "Industry": f"Industry{index // 5}",
                    }
                )
        sector, labels = build_sector_prior(ASSETS, path)
        assert tuple(sector.shape) == (93, 93)
        torch.testing.assert_close(sector.sum(-1), torch.ones(93))
        assert len(labels) == 93

    corr = build_absolute_correlation_prior(
        _fake_clean_train_split(),
        asset_cols=ASSETS,
        threshold=None,
    )
    assert tuple(corr.shape) == (93, 93)
    torch.testing.assert_close(corr.sum(-1), torch.ones(93), atol=1e-6, rtol=0)
    assert torch.count_nonzero(torch.diagonal(corr)) == 0

    model = instantiate_dimitri_token_to_price_model()
    before = [
        block.graph_scorer.base_graph.detach().clone()
        for block in model.backbone.st_blocks
    ]
    result = initialise_base_graphs_from_prior(
        model,
        sector,
        scale=4.0,
        jitter=0.02,
        seed=0,
    )
    assert result["count"] == 4
    for old, block in zip(before, model.backbone.st_blocks, strict=True):
        assert not torch.equal(old, block.graph_scorer.base_graph)


def _test_model_flow_and_causality() -> None:
    torch.manual_seed(7)
    model = instantiate_dimitri_token_to_price_model().eval()
    assert parameter_count(model) == DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT
    contract = resolved_per_block_contract(model.cfg)
    assert contract["activations"] == ["softmax", "softmax", "softmax", "sparsemax"]
    assert contract["num_edge_heads"] == [6, 6, 6, 1]
    assert contract["graph_hidden_dims"] == [192, 192, 192, 96]

    state = torch.randint(0, 1024, (1, 93, 6))
    with torch.no_grad():
        output = model(state)
    assert tuple(output["next_close_normalised"].shape) == (1, 5, 93, 1)
    expected_heads = [6, 6, 6, 1]
    for graph, heads in zip(output["block_graph_attns"], expected_heads, strict=True):
        assert tuple(graph.shape) == (1, 6, heads, 93, 93)
        torch.testing.assert_close(
            graph.float().sum(-1),
            torch.ones_like(graph.float().sum(-1)),
            atol=2e-5,
            rtol=0,
        )
    assert (output["block_graph_attns"][-1] == 0).any()

    changed = state.clone()
    changed[:, :, 4:] = torch.randint(0, 1024, changed[:, :, 4:].shape)
    with torch.no_grad():
        changed_output = model(changed)
    # Context length 4: representation at index 3 predicts index 4 and must not
    # depend on tokens at index 4 or later.
    torch.testing.assert_close(
        output["next_close_normalised"][:, 3],
        changed_output["next_close_normalised"][:, 3],
        atol=1e-6,
        rtol=0,
    )
    for left, right in zip(
        output["block_graph_attns"],
        changed_output["block_graph_attns"],
        strict=True,
    ):
        torch.testing.assert_close(left[:, 3], right[:, 3], atol=1e-6, rtol=0)


def _test_loss_and_gradients() -> None:
    torch.manual_seed(17)
    model = instantiate_dimitri_token_to_price_model().train()
    state = torch.randint(0, 1024, (1, 93, 5))
    raw_close = 100.0 * torch.exp(torch.randn(1, 93, 5) * 0.001)
    mean = raw_close[:, :, :4].mean(-1)
    std = raw_close[:, :, :4].std(-1).clamp_min(1e-3)
    output = model(state)
    objective, native = _dense_one_step_loss(
        output["next_close_normalised"],
        raw_close,
        mean,
        std,
    )
    assert torch.isfinite(objective) and torch.isfinite(native)
    objective.backward()
    graph_gradient = 0.0
    for block in model.backbone.st_blocks:
        for parameter in block.graph_scorer.parameters():
            if parameter.grad is not None:
                graph_gradient += float(parameter.grad.detach().abs().sum().item())
    assert graph_gradient > 0.0

    batch = {
        "raw_close": raw_close,
        "close_mean": mean,
        "close_std": std,
    }
    predicted, true, last = _origin_values(
        output=output,
        batch=batch,
        context_length=4,
        device=torch.device("cpu"),
    )
    assert tuple(predicted.shape) == (1, 1, 93, 1)
    assert tuple(true.shape) == (1, 1, 93, 1)
    assert tuple(last.shape) == (1, 93, 1)


def main() -> None:
    _test_split_mode_contracts()
    _test_window_contracts()
    _test_priors()
    _test_model_flow_and_causality()
    _test_loss_and_gradients()
    print("Dimitri V2 token-input/direct-price contracts passed.")


if __name__ == "__main__":
    main()

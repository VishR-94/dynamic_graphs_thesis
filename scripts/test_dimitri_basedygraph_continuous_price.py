from __future__ import annotations

"""CPU contracts for Dimitri-V2 continuous-input/direct-price diagnostic."""

from pathlib import Path
from tempfile import TemporaryDirectory
import csv

import torch

import src.data.dimitri_continuous_price as continuous_data
from src.data.dimitri_continuous_price import (
    DimitriContinuousPriceDataset,
    build_continuous_price_datasets,
)
from src.data.dimitri_token_price import DimitriTokenPriceWindowSpec
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_CONTINUOUS_PRICE_EXPECTED_PARAMETER_COUNT,
    build_sector_prior,
    initialise_base_graphs_from_prior,
    instantiate_dimitri_continuous_to_price_model,
    instantiate_dimitri_token_to_price_model,
    parameter_count,
    resolved_per_block_contract,
)
from src.training.run_dimitri_basedygraph_continuous_price import (
    _dense_one_step_loss,
    _origin_values,
)


ASSETS = [f"A{i:02d}" for i in range(93)]
CHANNELS = ["open", "high", "low", "close", "volume", "amount"]


def _fake_raw_split(*, sessions: int = 2, rows: int = 13) -> dict:
    """Return raw sessions including the one previous-session first row."""
    torch.manual_seed(19)
    values_list = []
    base = torch.linspace(20.0, 200.0, 93)
    for day in range(sessions):
        innovations = torch.randn(rows, 93) * 0.001
        close = base[None] * torch.exp(innovations.cumsum(dim=0))
        values = torch.zeros(rows, 93, 6)
        values[..., 0] = close * (1.0 - 0.0002)
        values[..., 1] = close * (1.0 + 0.0010)
        values[..., 2] = close * (1.0 - 0.0010)
        values[..., 3] = close
        values[..., 4] = 1_000.0 + torch.arange(rows)[:, None]
        values[..., 5] = 123.0  # must be overwritten to zero by the dataset
        values_list.append((values, None, f"2024-01-{day + 2:02d}"))
    return {
        "samples": values_list,
        "asset_cols": ASSETS,
        "channels": CHANNELS,
    }


def _test_dataset_contract() -> None:
    spec = DimitriTokenPriceWindowSpec(4, 2, 2)
    raw = _fake_raw_split(sessions=1, rows=13)  # 12 clean rows -> starts 0,2,4,6
    dataset = DimitriContinuousPriceDataset(
        raw_split=raw,
        split_name="train",
        split_mode="canonical",
        spec=spec,
    )
    assert len(dataset) == 4
    item = dataset[0]
    values = item["continuous_values"]
    assert tuple(values.shape) == (93, 6, 6)
    assert tuple(item["raw_close"].shape) == (93, 6)
    assert tuple(item["close_mean"].shape) == (93,)
    assert tuple(item["close_std"].shape) == (93,)
    assert torch.count_nonzero(values[..., 5]) == 0

    summary = dataset.summary()
    assert summary.split == "train"
    assert summary.split_mode == "canonical"
    assert summary.sessions == 1
    assert summary.windows == 4
    assert summary.assets == 93
    assert summary.channels == 6
    assert summary.context_length == 4
    assert summary.continuation_length == 2
    assert summary.sequence_length == 6
    assert summary.stride == 2
    assert summary.windows_per_session_min == 4
    assert summary.windows_per_session_max == 4
    assert summary.first_date == "2024-01-02"
    assert summary.last_date == "2024-01-02"
    assert summary.to_dict()["Windows"] == 4

    # First four rows are normalised with their own sample mean/std.
    context = values[:, :4]
    torch.testing.assert_close(
        context[..., :5].mean(dim=1),
        torch.zeros(93, 5),
        atol=1e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        context[..., :5].std(dim=1, correction=1),
        torch.ones(93, 5),
        atol=5e-3,
        rtol=0,
    )

    clean = raw["samples"][0][0][1:]
    expected_close = clean[:6, :, 3].transpose(0, 1)
    torch.testing.assert_close(item["raw_close"], expected_close)
    torch.testing.assert_close(
        item["close_mean"],
        expected_close[:, :4].mean(dim=1),
    )
    torch.testing.assert_close(
        item["close_std"],
        expected_close[:, :4].std(dim=1, correction=1),
    )


def _test_dataset_builder() -> None:
    spec = DimitriTokenPriceWindowSpec(4, 2, 2)
    train = _fake_raw_split(sessions=2, rows=13)
    validation = _fake_raw_split(sessions=1, rows=13)
    test = _fake_raw_split(sessions=3, rows=13)
    original = continuous_data.load_token_price_splits
    try:
        continuous_data.load_token_price_splits = lambda *_args, **_kwargs: {
            "train": train,
            "val": validation,
            "test": test,
        }
        raw, datasets = build_continuous_price_datasets(
            "/unused",
            split_mode="canonical",
            spec=spec,
        )
    finally:
        continuous_data.load_token_price_splits = original
    assert len(raw["train"]["samples"]) == 2
    assert len(datasets["train"]) == 8
    assert len(datasets["val"]) == 4
    assert len(datasets["test"]) == 12




def _test_shared_initialisation_parity() -> None:
    """Only the input adapter should differ at initialisation."""
    torch.manual_seed(101)
    token_model = instantiate_dimitri_token_to_price_model().eval()
    torch.manual_seed(101)
    continuous_model = instantiate_dimitri_continuous_to_price_model().eval()

    torch.testing.assert_close(
        token_model.next_close_head.weight,
        continuous_model.next_close_head.weight,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        token_model.next_close_head.bias,
        continuous_model.next_close_head.bias,
        atol=0.0,
        rtol=0.0,
    )

    token_reference = token_model.backbone
    continuous_reference = continuous_model.backbone.reference
    token_parameters = dict(token_reference.named_parameters())
    continuous_parameters = dict(continuous_reference.named_parameters())
    # The continuous adapter deliberately removes only the discrete state table.
    assert "state_embedding.weight" in token_parameters
    assert "state_embedding.weight" not in continuous_parameters
    for name, values in continuous_parameters.items():
        torch.testing.assert_close(
            values,
            token_parameters[name],
            atol=0.0,
            rtol=0.0,
        )

def _test_model_flow_and_causality() -> None:
    torch.manual_seed(7)
    model = instantiate_dimitri_continuous_to_price_model().eval()
    assert parameter_count(model) == DIMITRI_CONTINUOUS_PRICE_EXPECTED_PARAMETER_COUNT
    contract = resolved_per_block_contract(model.cfg)
    assert contract["activations"] == ["softmax", "softmax", "softmax", "sparsemax"]
    assert contract["num_edge_heads"] == [6, 6, 6, 1]
    assert contract["graph_hidden_dims"] == [192, 192, 192, 96]

    values = torch.randn(1, 93, 6, 6)
    values[..., 5] = 0.0
    with torch.no_grad():
        output = model(values)
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

    changed = values.clone()
    changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:])
    changed[..., 5] = 0.0
    with torch.no_grad():
        changed_output = model(changed)
    # Context length 4: representation at position 3 predicts position 4 and
    # cannot depend on continuous inputs at position 4 or later.
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


def _test_loss_gradients_and_prior() -> None:
    torch.manual_seed(23)
    model = instantiate_dimitri_continuous_to_price_model().train()
    values = torch.randn(1, 93, 5, 6)
    values[..., 5] = 0.0
    raw_close = 100.0 * torch.exp(torch.randn(1, 93, 5) * 0.001)
    mean = raw_close[:, :, :4].mean(-1)
    std = raw_close[:, :, :4].std(-1).clamp_min(1e-3)
    output = model(values)
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
        sector, _labels = build_sector_prior(ASSETS, path)
    result = initialise_base_graphs_from_prior(
        model,
        sector,
        scale=4.0,
        jitter=0.02,
        seed=0,
    )
    assert result["count"] == 4


def main() -> None:
    _test_dataset_contract()
    _test_dataset_builder()
    _test_shared_initialisation_parity()
    _test_model_flow_and_causality()
    _test_loss_gradients_and_prior()
    print("Dimitri V2 continuous-input/direct-price contracts passed.")


if __name__ == "__main__":
    main()

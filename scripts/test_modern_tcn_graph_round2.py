from __future__ import annotations

"""Fast CPU contracts for the two-family Round-2 depth experiment."""

from pathlib import Path
import sys
import types

import torch
from torch import nn

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.models.modern_tcn_graph_round2 import (
    ModernTCNGraphRound2Model,
    round2_model_config_from_mapping,
)
from src.training.modern_tcn_round2_specs import make_round2_specs
from src.training.run_modern_tcn_graph_round2 import (
    _build_loader,
    _build_optimizer,
    _export_selected_checkpoint,
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.flatten(x))

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

        def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
            if self.padding:
                x = torch.cat(
                    [x, x[..., -1:].expand(*x.shape[:-1], self.padding)],
                    dim=-1,
                )
            patches = x.unfold(-1, self.patch_size, self.patch_stride)
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



def _synthetic_split(nodes: int = 4) -> dict:
    torch.manual_seed(23)
    samples = []
    for day_index in range(3):
        base = 50.0 + torch.cumsum(0.01 * torch.randn(48, nodes), dim=0)
        open_price = base + 0.002 * torch.randn_like(base)
        close = base + 0.002 * torch.randn_like(base)
        high = torch.maximum(open_price, close) + 0.01
        low = torch.minimum(open_price, close) - 0.01
        volume = 1000.0 + 10.0 * torch.rand_like(base)
        amount = torch.zeros_like(base)
        values = torch.stack(
            [open_price, high, low, close, volume, amount],
            dim=-1,
        )
        samples.append((values, {}, f"2024-01-{day_index + 2:02d}"))
    return {
        "samples": samples,
        "asset_cols": [f"A{index}" for index in range(nodes)],
        "channels": ["open", "high", "low", "close", "volume", "amount"],
    }


def _batch(*, batch: int, context: int, nodes: int, channels: int) -> dict:
    torch.manual_seed(5)
    return {
        "x": torch.randn(batch, context, nodes, channels),
        "context_start": torch.arange(batch, dtype=torch.long),
        "session_length": torch.full(
            (batch,), context + 20, dtype=torch.long
        ),
    }


def _prior(nodes: int) -> torch.Tensor:
    values = torch.zeros(nodes, nodes)
    for row in range(nodes):
        values[row, (row + 1) % nodes] = 0.7
        values[row, (row + 2) % nodes] = 0.3
    return values


def _assert_graphs(output, config) -> None:
    if len(output.block_outputs) != config.num_st_blocks:
        raise AssertionError("Wrong number of block outputs.")
    for index, block in enumerate(output.block_outputs):
        graph = block.graph.selected.detach().float()
        expected = (
            output.predictions.shape[0],
            config.graph_heads_per_block[index],
            config.num_nodes,
            config.num_nodes,
        )
        if tuple(graph.shape) != expected:
            raise AssertionError(
                f"Block {index} graph {tuple(graph.shape)} != {expected}."
            )
        torch.testing.assert_close(
            graph.sum(dim=-1),
            torch.ones_like(graph.sum(dim=-1)),
            atol=2.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.diagonal(graph, dim1=-2, dim2=-1),
            torch.zeros_like(torch.diagonal(graph, dim1=-2, dim2=-1)),
            atol=0.0,
            rtol=0.0,
        )
    final = output.block_outputs[-1].graph.selected.detach()
    if not torch.any(final == 0):
        raise AssertionError("Final sparsemax graph produced no exact zeros.")


def main() -> None:
    _install_fake_modern_tcn()

    specs = make_round2_specs(
        prior_type="sector",
        graph_heads=1,
        context_length=16,
        stride=3,
        horizons=(1, 3),
    )
    if len(specs) != 12 or len({spec.run_name for spec in specs}) != 12:
        raise AssertionError("Round 2 did not create twelve unique runs.")
    family_counts = {
        family: sum(spec.graph_family == family for spec in specs)
        for family in ("dynamic_only", "prior_state")
    }
    if family_counts != {"dynamic_only": 6, "prior_state": 6}:
        raise AssertionError(f"Unexpected graph-family counts {family_counts}.")
    expected_temporal = {
        ("modern_tcn_transformer", 1),
        ("modern_tcn_transformer", 2),
        ("modern_tcn_transformer", 3),
        ("transformer_only", 2),
        ("transformer_only", 3),
        ("transformer_only", 4),
    }
    observed_temporal = {
        (spec.temporal_family, spec.num_transformer_blocks)
        for spec in specs
        if spec.graph_family == "dynamic_only"
    }
    if observed_temporal != expected_temporal:
        raise AssertionError(f"Unexpected temporal grid {observed_temporal}.")

    # One integer is repeated in every block; depth-specific schedules are
    # also directly configurable from the notebook without code changes.
    scheduled = make_round2_specs(
        prior_type="correlation",
        graph_heads={2: (2, 1), 3: (2, 2, 1), 4: (2, 2, 2, 1)},
        context_length=16,
        stride=5,
        horizons=(1, 2),
    )
    for spec in scheduled:
        expected = {2: (2, 1), 3: (2, 2, 1), 4: (2, 2, 2, 1)}[
            spec.num_st_blocks
        ]
        if spec.graph_heads_per_block != expected:
            raise AssertionError("Configurable graph-head schedule was lost.")

    # Validate every generated config and its final sparsemax schedule.
    for spec in specs:
        _validate_config(spec.config)
        if spec.graph_activations_per_block[-1] != "sparsemax":
            raise AssertionError("Final graph block is not sparsemax.")
        if any(
            activation != "softmax"
            for activation in spec.graph_activations_per_block[:-1]
        ):
            raise AssertionError("A non-final graph block is not softmax.")
        if any(
            float(spec.config["model"]["graph_regularisation"][key]) != 0.0
            for key in (
                "graph_entropy_reg",
                "graph_target_entropy_reg",
                "graph_temporal_smooth_reg",
            )
        ):
            raise AssertionError("Graph regularisation was enabled.")

    nodes = 4
    prior = _prior(nodes)
    batch = _batch(batch=2, context=16, nodes=nodes, channels=5)
    representative = [
        next(
            spec
            for spec in specs
            if spec.temporal_family == "modern_tcn_transformer"
            and spec.num_transformer_blocks == 1
            and spec.graph_family == "dynamic_only"
        ),
        next(
            spec
            for spec in specs
            if spec.temporal_family == "modern_tcn_transformer"
            and spec.num_transformer_blocks == 3
            and spec.graph_family == "prior_state"
        ),
        next(
            spec
            for spec in specs
            if spec.temporal_family == "transformer_only"
            and spec.num_transformer_blocks == 2
            and spec.graph_family == "dynamic_only"
        ),
        next(
            spec
            for spec in specs
            if spec.temporal_family == "transformer_only"
            and spec.num_transformer_blocks == 4
            and spec.graph_family == "prior_state"
        ),
    ]

    for spec in representative:
        config = round2_model_config_from_mapping(spec.config, num_nodes=nodes)
        torch.manual_seed(42)
        model = ModernTCNGraphRound2Model(
            config,
            static_prior=(prior if config.uses_static_graph else None),
        )
        output = model(**batch)
        expected_prediction = (2, 2, nodes, 1)
        if tuple(output.predictions.shape) != expected_prediction:
            raise AssertionError(
                f"{spec.run_name} output {tuple(output.predictions.shape)} "
                f"!= {expected_prediction}."
            )
        _assert_graphs(output, config)

        if config.graph_family == "dynamic_only":
            if any(value is not None for value in model.alphas()):
                raise AssertionError("Dynamic-only model unexpectedly has alpha.")
            if any(
                block.graph.base is not None for block in output.block_outputs
            ):
                raise AssertionError("Dynamic-only model unexpectedly has base graph.")
        else:
            for alpha in model.alphas():
                if alpha is None:
                    raise AssertionError("prior_state model is missing alpha.")
                torch.testing.assert_close(
                    alpha,
                    torch.tensor(0.25),
                    atol=1.0e-6,
                    rtol=0.0,
                )
            # Forecast gradients must reach state exposure, every graph
            # learner, every alpha, and every beta.
            output.predictions.square().mean().backward()
            for index, block in enumerate(model.graph_spatial_blocks):
                required = {
                    "Q": block.graph_learner.q_proj.weight.grad,
                    "K": block.graph_learner.k_proj.weight.grad,
                    "value": block.spatial_module.value_projection.weight.grad,
                    "alpha": block.graph_learner.raw_alpha.grad,
                    "beta": block.spatial_gate.raw_beta.grad,
                }
                for name, gradient in required.items():
                    if gradient is None or float(gradient.norm().item()) <= 0.0:
                        raise AssertionError(
                            f"Forecast loss did not reach block {index} {name}."
                        )
            state_modules = model.block_state_modules()
            if any(module is None for module in state_modules):
                raise AssertionError("prior_state model lost a state module.")
            if not any(
                parameter.grad is not None
                and float(parameter.grad.detach().norm().item()) > 0.0
                for module in set(state_modules)
                for parameter in module.parameters()
            ):
                raise AssertionError("Forecast loss did not reach state projection.")

        optimizer = _build_optimizer(model, spec.config)
        if not isinstance(optimizer, torch.optim.Adam):
            raise AssertionError("Round 2 did not preserve Adam.")
        if len(optimizer.param_groups) != 2:
            raise AssertionError("Round 2 did not preserve split LR groups.")
        rates = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
        if rates != {"backbone": 2.5e-4, "graph": 5.0e-4}:
            raise AssertionError(f"Unexpected Round-2 LRs: {rates}.")

    # Selected-checkpoint export retains every layer and derives window and
    # horizon axes from the actual dataset rather than notebook constants.
    export_spec = next(
        spec
        for spec in specs
        if spec.temporal_family == "transformer_only"
        and spec.num_transformer_blocks == 2
        and spec.graph_family == "prior_state"
    )
    export_config = round2_model_config_from_mapping(
        export_spec.config,
        num_nodes=nodes,
    )
    export_model = ModernTCNGraphRound2Model(
        export_config,
        static_prior=prior,
    )
    split = _synthetic_split(nodes)
    dataset = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=16,
            horizons=(1, 3),
            stride=5,
        ),
    )
    loader = _build_loader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=0,
        seed=42,
        pin_memory=False,
    )
    exported = _export_selected_checkpoint(
        model=export_model,
        loader=loader,
        split_name="train",
        device=torch.device("cpu"),
        use_amp=False,
        config=export_spec.config,
        train_split=split,
        asset_cols=split["asset_cols"],
        checkpoint_epoch=1,
    )
    prediction = exported["prediction_result"]
    graphs = exported["graph_artifacts"]
    if int(prediction["y_pred"].shape[0]) != len(dataset):
        raise AssertionError("Exported prediction window count is not dataset-derived.")
    if tuple(prediction["horizons"]) != (1, 3):
        raise AssertionError("Exported horizons differ from the configured task.")
    if len(graphs["per_layer"]) != 2:
        raise AssertionError("Export did not retain every graph layer.")
    if len(graphs["per_layer_base"]) != 2 or len(graphs["per_layer_dynamic"]) != 2:
        raise AssertionError("Export did not retain base/dynamic components.")
    if tuple(graphs["beta_per_layer"].shape) != (2,):
        raise AssertionError("Export did not retain one beta per block.")
    if len(graphs["alpha_per_layer"]) != 2:
        raise AssertionError("Export did not retain one alpha per block.")

    # The production runner contains no fixed dataset-window-count gates.
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "training"
        / "run_modern_tcn_graph_round2.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("EXPECTED_TRAIN_WINDOWS", "EXPECTED_VALIDATION_WINDOWS", "EXPECTED_TEST_WINDOWS"):
        if forbidden in source:
            raise AssertionError(f"Runner contains hard-coded gate {forbidden}.")

    print("ModernTCN graph Round-2 contracts passed.")


if __name__ == "__main__":
    main()

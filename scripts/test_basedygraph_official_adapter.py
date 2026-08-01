from __future__ import annotations

"""CPU smoke tests for the direct official BaseDyGraph one-step adapter.

Run from the repository root after initialising submodules:

    python3 -m scripts.test_basedygraph_official_adapter

The test imports the pinned external implementation, does not modify it, and
checks the only project-specific adaptation: selecting the final observed
context state as the predictor of the first unseen state.
"""

from pathlib import Path
from types import ModuleType
import sys

import torch
import torch.nn.functional as F

from src.models.basedygraph_official_adapter import (
    OfficialBaseDyGraphOneStep,
    OfficialBaseDyGraphRunConfig,
    assert_official_one_step_parity,
)


def _assert_graph(graph: torch.Tensor | None, *, batch: int, time_nodes: int, heads: int) -> None:
    if graph is None:
        raise AssertionError("Expected an official graph tensor.")
    expected = (batch, heads, time_nodes, time_nodes)
    if tuple(graph.shape) != expected:
        raise AssertionError(f"Graph shape {tuple(graph.shape)} != {expected}.")
    row_sums = graph.float().sum(dim=-1)
    torch.testing.assert_close(
        row_sums,
        torch.ones_like(row_sums),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def _run_case(config: OfficialBaseDyGraphRunConfig) -> None:
    torch.manual_seed(7)
    model = OfficialBaseDyGraphOneStep(
        config,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        scheduler_t_max=4,
    )

    # The wrapper must introduce no trainable operation of its own. Every
    # parameter name is rooted in the untouched official model.
    unexpected = [
        name
        for name, _ in model.named_parameters()
        if not name.startswith("official_model.")
    ]
    if unexpected:
        raise AssertionError(f"Adapter introduced trainable parameters: {unexpected}")

    batch = 2
    context = torch.randint(
        0,
        config.num_states,
        (batch, config.context_length, config.num_nodes),
        dtype=torch.long,
    )
    target = torch.randint(
        0,
        config.num_states,
        (batch, config.num_nodes),
        dtype=torch.long,
    )

    parity = assert_official_one_step_parity(model, context)
    if parity["dummy_invariance_max_abs_difference"] > 1.0e-5:
        raise AssertionError("Dummy-token invariance failed.")
    if parity["direct_reference_max_abs_difference"] > 1.0e-5:
        raise AssertionError("Official full-forward parity failed.")

    model.train()
    output = model(context)
    expected_logits = (batch, config.num_nodes, config.num_states)
    if tuple(output.s1_logits.shape) != expected_logits:
        raise AssertionError(
            f"Logit shape {tuple(output.s1_logits.shape)} != {expected_logits}."
        )
    if tuple(output.temporal_final.shape) != (
        batch,
        config.num_nodes,
        config.d_model,
    ):
        raise AssertionError("Unexpected final temporal representation shape.")
    if tuple(output.spatial_final.shape) != (
        batch,
        config.num_nodes,
        config.d_model,
    ):
        raise AssertionError("Unexpected final spatial representation shape.")

    if config.spatial_module_type == "none":
        if output.selected_graph is not None:
            raise AssertionError("No-graph model returned a selected graph.")
    else:
        _assert_graph(
            output.selected_graph,
            batch=batch,
            time_nodes=config.num_nodes,
            heads=config.num_edge_heads,
        )

    expected_layers = config.num_st_blocks if config.num_st_blocks > 1 else 1
    if len(output.per_layer_graphs) != expected_layers:
        raise AssertionError(
            f"Expected {expected_layers} layer graphs, observed "
            f"{len(output.per_layer_graphs)}."
        )

    loss = F.cross_entropy(
        output.s1_logits.float().reshape(-1, config.num_states),
        target.reshape(-1),
    )
    loss.backward()

    graph_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if (
            "graph_scorer" in name
            and parameter.requires_grad
            and parameter.grad is not None
        )
    ]
    if config.spatial_module_type != "none":
        if not graph_gradients:
            raise AssertionError("No forecasting gradient reached the official graph scorer.")
        total_graph_gradient = sum(
            float(gradient.detach().float().abs().sum().item())
            for gradient in graph_gradients
        )
        if not total_graph_gradient > 0.0:
            raise AssertionError("Official graph-scorer gradient is exactly zero.")

    configured = model.configure_official_optimizers()
    if "optimizer" not in configured or "lr_scheduler" not in configured:
        raise AssertionError("Official optimizer/scheduler contract changed.")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    required = [
        repository_root / "external" / "BaseDyGraph" / "src" / name
        for name in ("utilities.py", "modules.py", "model.py")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Initialise the pinned BaseDyGraph submodule before this test. "
            f"Missing: {missing}"
        )

    # BaseDyGraph and Kronos both use generic top-level module names in
    # their official repositories. Prove that the isolated loader restores
    # any pre-existing bindings instead of corrupting the Kronos process.
    sentinel_model = ModuleType("model")
    sentinel_utilities = ModuleType("utilities")
    sentinel_modules = ModuleType("modules")
    missing = object()
    previous_aliases = {
        name: sys.modules.get(name, missing)
        for name in ("model", "utilities", "modules")
    }
    sys.modules["model"] = sentinel_model
    sys.modules["utilities"] = sentinel_utilities
    sys.modules["modules"] = sentinel_modules

    cases = (
        OfficialBaseDyGraphRunConfig(
            num_states=32,
            num_nodes=5,
            context_length=7,
            d_model=16,
            nhead=4,
            num_temporal_layers=1,
            num_spatial_layers=1,
            ff_mult=2,
            num_edge_heads=2,
            graph_hidden_dim=16,
            spatial_module_type="static_graph",
            num_st_blocks=1,
        ),
        OfficialBaseDyGraphRunConfig(
            num_states=32,
            num_nodes=5,
            context_length=7,
            d_model=24,
            nhead=4,
            num_temporal_layers=1,
            num_spatial_layers=1,
            ff_mult=2,
            num_edge_heads=2,
            graph_hidden_dim=16,
            spatial_module_type="static_graph",
            num_st_blocks=3,
        ),
        OfficialBaseDyGraphRunConfig(
            num_states=32,
            num_nodes=5,
            context_length=7,
            d_model=16,
            nhead=4,
            num_temporal_layers=2,
            num_spatial_layers=1,
            ff_mult=4,
            num_edge_heads=2,
            graph_hidden_dim=16,
            spatial_module_type="dynamic_graph",
            num_st_blocks=1,
        ),
    )

    try:
        for config in cases:
            _run_case(config)

        if sys.modules.get("model") is not sentinel_model:
            raise AssertionError("The BaseDyGraph loader replaced top-level 'model'.")
        if sys.modules.get("utilities") is not sentinel_utilities:
            raise AssertionError("The BaseDyGraph loader did not restore 'utilities'.")
        if sys.modules.get("modules") is not sentinel_modules:
            raise AssertionError("The BaseDyGraph loader did not restore 'modules'.")
    finally:
        for name, previous in previous_aliases.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    print("Official BaseDyGraph one-step adapter CPU smoke tests passed.")


if __name__ == "__main__":
    main()
from __future__ import annotations

"""CPU contracts for the two pinned BaseDyGraph-v1 token controls.

Run after initialising the external submodules:

    python3 -m scripts.test_basedygraph_v1_token_comparison
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.basedygraph_v1_token_comparison import (
    BaseDyGraphV1TokenConfig,
    BaseDyGraphV1TokenModel,
)
from src.training.basedygraph_v1_token_comparison_specs import (
    make_basedygraph_v1_token_comparison_specs,
)


def _assert_graph(
    graph: torch.Tensor,
    *,
    batch: int,
    heads: int,
    nodes: int,
) -> None:
    expected = (batch, heads, nodes, nodes)
    if tuple(graph.shape) != expected:
        raise AssertionError(f"Graph shape {tuple(graph.shape)} != {expected}.")
    if not torch.isfinite(graph).all():
        raise AssertionError("Graph contains non-finite values.")
    if torch.any(graph < 0):
        raise AssertionError("Graph contains negative values.")
    torch.testing.assert_close(
        graph.float().sum(dim=-1),
        torch.ones(batch, heads, nodes),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def _dense_case() -> None:
    torch.manual_seed(7)
    config = BaseDyGraphV1TokenConfig(
        num_nodes=5,
        context_length=7,
        prediction_length=7,
        evaluation_horizons=(1, 3, 7),
        vocabulary_size=32,
        prediction_mode="dense_one_step",
        d_model=16,
        temporal_num_heads=4,
        temporal_num_layers=1,
        feedforward_multiplier=2,
        graph_num_heads=2,
        graph_hidden_dim=16,
        num_st_blocks=4,
    )
    model = BaseDyGraphV1TokenModel(config)
    model.eval()
    batch = 2
    context = torch.randint(0, 32, (batch, 7, 5))
    future_a = torch.randint(0, 32, (batch, 5))
    future_b = (future_a + 1) % 32

    output_a = model(context, first_future_s1=future_a)
    output_b = model(context, first_future_s1=future_b)
    if tuple(output_a.s1_logits.shape) != (batch, 7, 5, 32):
        raise AssertionError("Unexpected dense next-state logit shape.")
    if output_a.teacher_forced_targets is None:
        raise AssertionError("Dense model returned no teacher-forced targets.")
    if tuple(output_a.teacher_forced_targets.shape) != (batch, 7, 5):
        raise AssertionError("Unexpected dense target shape.")
    if len(output_a.per_layer_graphs) != 4:
        raise AssertionError("Dense model did not expose four graph layers.")
    if len(output_a.graph_sequences) != 4:
        raise AssertionError("Dense model did not expose graph sequences.")
    for graph, sequence in zip(
        output_a.per_layer_graphs,
        output_a.graph_sequences,
    ):
        _assert_graph(graph, batch=batch, heads=2, nodes=5)
        if tuple(sequence.shape) != (batch, 8, 2, 5, 5):
            raise AssertionError(
                f"Unexpected dense graph-sequence shape {tuple(sequence.shape)}."
            )

    # The appended true h1 token is a target only.  It must not change any
    # logit or forecast-origin graph because all temporal blocks are causal.
    torch.testing.assert_close(
        output_a.s1_logits,
        output_b.s1_logits,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    for graph_a, graph_b in zip(
        output_a.per_layer_graphs,
        output_b.per_layer_graphs,
    ):
        torch.testing.assert_close(graph_a, graph_b, atol=1.0e-6, rtol=1.0e-6)
    if torch.equal(
        output_a.teacher_forced_targets[:, -1],
        output_b.teacher_forced_targets[:, -1],
    ):
        raise AssertionError("The final dense target did not change.")

    model.train()
    output = model(context, first_future_s1=future_a)
    target = output.teacher_forced_targets
    if target is None:
        raise AssertionError("Dense training target disappeared.")
    loss = F.cross_entropy(
        output.s1_logits.float().reshape(-1, 32),
        target.reshape(-1),
    )
    loss.backward()
    graph_gradient = sum(
        float(parameter.grad.detach().abs().sum().item())
        for name, parameter in model.named_parameters()
        if "graph_scorer" in name and parameter.grad is not None
    )
    if not graph_gradient > 0:
        raise AssertionError("Dense CE did not reach the graph scorers.")


def _parallel_case() -> None:
    torch.manual_seed(9)
    config = BaseDyGraphV1TokenConfig(
        num_nodes=5,
        context_length=7,
        prediction_length=7,
        evaluation_horizons=(1, 3, 7),
        vocabulary_size=32,
        prediction_mode="parallel_60",
        d_model=16,
        temporal_num_heads=4,
        temporal_num_layers=1,
        feedforward_multiplier=2,
        graph_num_heads=2,
        graph_hidden_dim=16,
        num_st_blocks=4,
        future_predictor_num_layers=1,
        future_predictor_num_heads=4,
        future_predictor_feedforward_multiplier=2,
    )
    model = BaseDyGraphV1TokenModel(config)
    context = torch.randint(0, 32, (2, 7, 5))
    output = model(context)
    if tuple(output.s1_logits.shape) != (2, 7, 5, 32):
        raise AssertionError("Unexpected parallel logit shape.")
    if output.teacher_forced_targets is not None:
        raise AssertionError("Parallel model exposed teacher-forced targets.")
    if output.future_hidden is None:
        raise AssertionError("Structured predictor returned no future hidden state.")
    if len(output.per_layer_graphs) != 4:
        raise AssertionError("Parallel model did not expose four graph layers.")
    for graph, sequence in zip(output.per_layer_graphs, output.graph_sequences):
        _assert_graph(graph, batch=2, heads=2, nodes=5)
        if tuple(sequence.shape) != (2, 7, 2, 5, 5):
            raise AssertionError("Unexpected parallel graph-sequence shape.")


def _spec_case() -> None:
    dense, parallel = make_basedygraph_v1_token_comparison_specs(
        context_length=60,
        prediction_length=60,
        evaluation_horizons=(1, 5, 15, 30, 60),
        d_model=96,
        temporal_num_layers=1,
        temporal_num_heads=4,
        feedforward_multiplier=2,
        graph_num_heads=1,
        graph_hidden_dim=96,
        num_st_blocks=4,
    )
    if dense.run_name == parallel.run_name:
        raise AssertionError("The two controls share a run name.")
    if dense.prediction_mode != "dense_one_step":
        raise AssertionError("Dense specification mode changed.")
    if parallel.prediction_mode != "parallel_60":
        raise AssertionError("Parallel specification mode changed.")
    for spec in (dense, parallel):
        architecture = spec.config["model"]["official_basedygraph_v1"]
        if architecture["num_st_blocks"] != 4:
            raise AssertionError("Specification does not contain four ST blocks.")
        if architecture["d_model"] != 96:
            raise AssertionError("Specification does not retain D=96.")
        if architecture["graph_num_heads"] != 1:
            raise AssertionError("Specification does not retain one graph head.")
        if architecture["graph_hidden_dim"] != 96:
            raise AssertionError("Specification graph width changed.")
        if architecture["graph_activation"] != "softmax":
            raise AssertionError("BaseDyGraph-v1 graph activation changed.")
        if architecture["spatial_module_type"] != "dynamic_graph":
            raise AssertionError("Specification is not dynamic-only.")
        if spec.config["training"]["selection_split"] != "test":
            raise AssertionError("Specification is not test-selected.")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    required = [
        repository / "external" / "BaseDyGraph" / "src" / name
        for name in ("utilities.py", "modules.py", "model.py")
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Initialise the pinned BaseDyGraph submodule before this test. "
            f"Missing: {missing}"
        )
    _dense_case()
    _parallel_case()
    _spec_case()
    print("BaseDyGraph-v1 token comparison contracts passed.")


if __name__ == "__main__":
    main()

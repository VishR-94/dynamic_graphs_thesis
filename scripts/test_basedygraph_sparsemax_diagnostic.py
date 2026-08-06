from __future__ import annotations

"""CPU contracts for the four BaseDyGraph sparsemax diagnostics."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import src.models.basedygraph_financial as financial
from src.models.basedygraph_financial import (
    BaseDyGraphFinancialConfig,
    BaseDyGraphGraphRegularisationConfig,
    OfficialBaseDyGraphContinuousForecaster,
    OfficialBaseDyGraphTeacherForcedOneStepForecaster,
    graph_regularisation_loss,
)
from src.training.run_basedygraph_sparsemax_diagnostic import (
    EXPERIMENT_SPECS,
    LAYER_ACTIVATIONS,
    PHASE1_EXPERIMENTS,
    PHASE2_EXPERIMENTS,
    _append_synthetic_candle,
)


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_values = shifted.sort(dim=dim, descending=True).values
    ranks = torch.arange(
        1,
        int(logits.shape[dim]) + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    view = [1] * logits.ndim
    view[dim] = -1
    ranks = ranks.view(view)
    cumulative = sorted_values.cumsum(dim=dim)
    support = 1 + ranks * sorted_values > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_sum = cumulative.gather(dim, support_size - 1)
    tau = (tau_sum - 1) / support_size.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class _Temporal(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor) -> Tensor:
        # Pointwise in time, hence causal for this contract fixture.
        return values + 0.05 * self.projection(values)


class _DynamicScorer(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            graph_activation=str(config.graph_activation),
        )
        self.heads = int(config.num_edge_heads)
        self.edge_dim = int(config.graph_hidden_dim) // self.heads
        self.q = nn.Linear(config.d_model, config.graph_hidden_dim, bias=False)
        self.k = nn.Linear(config.d_model, config.graph_hidden_dim, bias=False)

    def forward(self, hidden: Tensor, state_ids: Tensor) -> Tensor:
        del state_ids
        batch, steps, nodes, _ = hidden.shape
        query = self.q(hidden).reshape(
            batch,
            steps,
            nodes,
            self.heads,
            self.edge_dim,
        ).permute(0, 1, 3, 2, 4)
        key = self.k(hidden).reshape(
            batch,
            steps,
            nodes,
            self.heads,
            self.edge_dim,
        ).permute(0, 1, 3, 2, 4)
        logits = torch.einsum("btgid,btgjd->btgij", query, key)
        logits = logits / (self.edge_dim ** 0.5)
        activation = str(self.cfg.graph_activation)
        if activation == "softmax":
            return torch.softmax(logits, dim=-1)
        if activation == "sparsemax":
            # The scale makes exact zeros deterministic in the small fixture.
            return _sparsemax(12.0 * logits, dim=-1)
        raise ValueError(f"Unsupported fixture activation: {activation}")


class _Spatial(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor | None,
        *,
        e: Tensor,
    ) -> Tensor:
        del e
        if adjacency is None:
            return hidden
        messages = torch.einsum(
            "btij,btjd->btid",
            adjacency.mean(dim=2),
            hidden,
        )
        return hidden + 0.1 * self.output(messages)


class _Block(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.temporal_module = _Temporal(config.d_model)
        self.graph_scorer = _DynamicScorer(config)
        self.spatial_module = _Spatial(config.d_model)
        self.post_norm = nn.LayerNorm(config.d_model)


class _FakeBackbone(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.state_embedding = nn.Embedding(config.num_states, config.d_model)
        self.node_embedding = nn.Embedding(config.num_nodes, config.d_model)
        self.pre_norm = nn.LayerNorm(config.d_model)
        self.post_norm = nn.LayerNorm(config.d_model)
        self.st_blocks = nn.ModuleList(
            [_Block(config) for _ in range(config.num_st_blocks)]
        )

    def _initial_embedding_bntd(self, state_ids: Tensor) -> Tensor:
        values = self.state_embedding(state_ids)
        node_ids = torch.arange(state_ids.shape[1], device=state_ids.device)
        values = values + self.node_embedding(node_ids).view(
            1,
            -1,
            1,
            values.shape[-1],
        )
        return self.pre_norm(values)

    def state_embedding_btnd(self, state_ids: Tensor) -> Tensor:
        return self.state_embedding(state_ids).permute(0, 2, 1, 3).contiguous()

    def forward(self, state_ids: Tensor) -> dict[str, object]:
        hidden = self._initial_embedding_bntd(state_ids).permute(0, 2, 1, 3)
        embedding = self.state_embedding_btnd(state_ids)
        graphs = []
        for block in self.st_blocks:
            temporal = block.temporal_module(
                hidden.permute(0, 2, 1, 3)
            ).permute(0, 2, 1, 3)
            graph = block.graph_scorer(temporal, state_ids)
            hidden = block.post_norm(
                block.spatial_module(temporal, graph, e=embedding)
            )
            graphs.append(graph)
        return {
            "spatial_repr": self.post_norm(hidden),
            "block_graph_attns": tuple(graphs),
            "graph_attn": graphs[-1],
        }


class _FakeNextStateHead(nn.Module):
    def __init__(self, d_model: int, num_states: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_states)

    def forward(self, hidden: Tensor) -> Tensor:
        # Official contract: [B,T,N,D] -> [B,N,T-1,K].
        logits = self.proj(hidden[:, :-1])
        return logits.permute(0, 2, 1, 3).contiguous()


class _FakeModelModule:
    DiscreteSTGraphBackbone = _FakeBackbone
    NextStateHead = _FakeNextStateHead


def _fake_official_config(run_config: object, **_: object) -> SimpleNamespace:
    return SimpleNamespace(
        num_states=run_config.num_states,
        num_nodes=run_config.num_nodes,
        d_model=run_config.d_model,
        num_edge_heads=run_config.num_edge_heads,
        graph_hidden_dim=run_config.graph_hidden_dim,
        num_st_blocks=run_config.num_st_blocks,
        spatial_module_type=run_config.spatial_module_type,
        graph_activation=run_config.graph_activation,
    )


def _patch_official():
    modules = SimpleNamespace(
        model=_FakeModelModule,
        commit="fake-pinned-commit",
    )
    return (
        patch.object(
            financial,
            "load_official_basedygraph_modules",
            return_value=modules,
        ),
        patch.object(
            financial,
            "build_official_model_config",
            side_effect=_fake_official_config,
        ),
    )


def _small_config(*, mode: str, horizons: tuple[int, ...]) -> BaseDyGraphFinancialConfig:
    return BaseDyGraphFinancialConfig(
        mode=mode,
        graph_type="dynamic_graph",
        graph_scope="per_timestep",
        context_length=6,
        prediction_length=max(horizons),
        evaluation_horizons=horizons,
        num_nodes=5,
        input_channels=5,
        d_model=12,
        temporal_heads=3,
        temporal_layers=1,
        spatial_layers=1,
        ff_mult=2,
        graph_heads=1,
        graph_hidden_dim=8,
        num_st_blocks=4,
        graph_activation="softmax",
        graph_activations=LAYER_ACTIVATIONS,
        future_predictor_layers=0,
        future_predictor_heads=3,
        regularisation=BaseDyGraphGraphRegularisationConfig(
            target_entropy=1.4,
            target_entropy_weight=0.5,
            temporal_smooth_weight=0.01,
            warmup_epochs=5,
        ),
    )


def _assert_graph_rows(graph: Tensor) -> None:
    assert torch.isfinite(graph).all()
    assert (graph >= 0).all()
    expected = torch.ones_like(graph.sum(dim=-1))
    assert torch.allclose(graph.sum(dim=-1), expected, atol=1.0e-5)


def test_spec_contract() -> None:
    assert len(EXPERIMENT_SPECS) == 4
    assert PHASE1_EXPERIMENTS == (
        "continuous_one_minute",
        "token_teacher_forced_one_minute",
    )
    assert PHASE2_EXPERIMENTS == (
        "continuous_parallel_sixty_minute",
        "continuous_autoregressive_sixty_minute",
    )
    for spec in EXPERIMENT_SPECS:
        assert spec.run_name.startswith("DO_NOT_REPORT")
        assert spec.config.d_model == 96
        assert spec.config.num_st_blocks == 4
        assert spec.config.graph_heads == 1
        assert spec.config.graph_hidden_dim == 64
        assert spec.config.resolved_graph_activations == LAYER_ACTIVATIONS
        assert spec.config.regularisation.layer == -1


def test_serialisation_compatibility() -> None:
    legacy = BaseDyGraphFinancialConfig(mode="continuous", graph_type="dynamic_graph")
    assert "graph_activations" not in legacy.to_dict()
    current = _small_config(mode="continuous", horizons=(1,))
    payload = current.to_dict()
    assert payload["graph_activations"] == list(LAYER_ACTIVATIONS) or tuple(
        payload["graph_activations"]
    ) == LAYER_ACTIVATIONS
    reconstructed = BaseDyGraphFinancialConfig.from_dict(payload)
    assert reconstructed.resolved_graph_activations == LAYER_ACTIVATIONS


def test_teacher_forced_one_step_contract() -> None:
    patch_load, patch_config = _patch_official()
    with patch_load, patch_config:
        config = _small_config(mode="token", horizons=(1,))
        model = OfficialBaseDyGraphTeacherForcedOneStepForecaster(config)
        token_ids = torch.randint(0, 1024, (2, 6, 5, 2))
        future_a = torch.randint(0, 1024, (2, 1, 5))
        future_b = (future_a + 17) % 1024

        output_a = model(token_ids, target_s1=future_a)
        output_b = model(token_ids, target_s1=future_b)

        assert tuple(output_a.s1_logits.shape) == (2, 6, 5, 1024)
        assert tuple(output_a.forecast.s1_logits.shape) == (2, 1, 5, 1024)
        assert len(output_a.graph_sequences) == 4
        targets = model.teacher_targets(token_ids, future_a)
        assert torch.equal(targets[:, :-1], token_ids[:, 1:, :, 0])
        assert torch.equal(targets[:, -1:], future_a)

        # Causal final-context forecast is invariant to the appended target.
        assert torch.allclose(
            output_a.forecast.s1_logits,
            output_b.forecast.s1_logits,
            atol=1.0e-6,
            rtol=1.0e-5,
        )

        for layer, graph in enumerate(output_a.graph_sequences):
            assert graph is not None
            assert tuple(graph.shape) == (2, 6, 1, 5, 5)
            _assert_graph_rows(graph)
            if layer < 3:
                assert bool((graph > 0).all())
            else:
                assert bool((graph == 0).any())

        loss = F.cross_entropy(
            output_a.s1_logits.reshape(-1, 1024),
            targets.reshape(-1),
        )
        regularisation = graph_regularisation_loss(
            output_a.graph_sequences,
            config.regularisation,
            epoch=5,
        )
        (loss + regularisation.total).backward()
        graph_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if (".q." in name or ".k." in name)
            and parameter.grad is not None
        )
        assert graph_gradient > 0.0

        generated = model.generate_samples(
            token_ids,
            sample_count=10,
            token_selection="sample",
            temperature=1.0,
            top_k=0,
            top_p=0.9,
        )
        assert tuple(generated.token_ids.shape) == (10, 2, 1, 5, 2)


def test_continuous_one_and_multi_horizon_contracts() -> None:
    patch_load, patch_config = _patch_official()
    with patch_load, patch_config:
        for horizons in ((1,), (1, 3, 5)):
            config = _small_config(mode="continuous", horizons=horizons)
            model = OfficialBaseDyGraphContinuousForecaster(config)
            values = torch.randn(2, 6, 5, 5)
            output = model(values)
            assert tuple(output.predictions.shape) == (
                2,
                len(horizons),
                5,
                1,
            )
            assert len(output.graph_sequences) == 4
            assert output.graph.selected is not None
            assert tuple(output.graph.selected.shape) == (2, 1, 5, 5)
            for layer, graph in enumerate(output.graph_sequences):
                assert graph is not None
                assert tuple(graph.shape) == (2, 6, 1, 5, 5)
                _assert_graph_rows(graph)
                if layer < 3:
                    assert bool((graph > 0).all())
                else:
                    assert bool((graph == 0).any())

            objective = output.predictions.square().mean()
            regularisation = graph_regularisation_loss(
                output.graph_sequences,
                config.regularisation,
                epoch=5,
            )
            (objective + regularisation.total).backward()
            graph_gradient = sum(
                float(parameter.grad.abs().sum())
                for name, parameter in model.named_parameters()
                if (".q." in name or ".k." in name)
                and parameter.grad is not None
            )
            assert graph_gradient > 0.0


def test_autoregressive_candle_bridge() -> None:
    raw = torch.zeros(2, 6, 5, 5)
    raw[..., 3] = 100.0
    raw[..., 4] = 7.0
    next_close = torch.tensor(
        [
            [101.0, 99.0, 100.5, 100.0, 102.0],
            [98.0, 103.0, 100.0, 101.0, 99.5],
        ]
    )
    shifted = _append_synthetic_candle(raw, next_close)
    assert tuple(shifted.shape) == tuple(raw.shape)
    candle = shifted[:, -1]
    assert torch.equal(candle[..., 0], torch.full_like(next_close, 100.0))
    assert torch.equal(candle[..., 3], next_close)
    assert torch.equal(candle[..., 1], torch.maximum(candle[..., 0], next_close))
    assert torch.equal(candle[..., 2], torch.minimum(candle[..., 0], next_close))
    assert torch.equal(candle[..., 4], torch.full_like(next_close, 7.0))



def test_real_official_integration_if_available() -> None:
    """Exercise ST4/G1/layer activations against the pinned submodule."""
    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "external" / "BaseDyGraph" / "src"
    required = tuple(
        source_dir / name
        for name in ("utilities.py", "modules.py", "model.py")
    )
    if not all(path.is_file() for path in required):
        print(
            "Pinned BaseDyGraph submodule absent; "
            "skipping sparsemax real-adapter integration."
        )
        return

    torch.manual_seed(29)
    token_config = BaseDyGraphFinancialConfig(
        mode="token",
        graph_type="dynamic_graph",
        graph_scope="per_timestep",
        context_length=6,
        prediction_length=1,
        evaluation_horizons=(1,),
        num_nodes=5,
        d_model=16,
        temporal_heads=4,
        temporal_layers=1,
        spatial_layers=1,
        ff_mult=2,
        graph_heads=1,
        graph_hidden_dim=16,
        num_st_blocks=4,
        graph_activation="softmax",
        graph_activations=LAYER_ACTIVATIONS,
        future_predictor_layers=0,
        future_predictor_heads=4,
        regularisation=BaseDyGraphGraphRegularisationConfig(
            target_entropy=1.4,
            target_entropy_weight=0.05,
            temporal_smooth_weight=0.01,
            warmup_epochs=5,
        ),
    )
    token_model = OfficialBaseDyGraphTeacherForcedOneStepForecaster(
        token_config,
        external_source_dir=str(source_dir),
    )
    tokens = torch.randint(0, 1024, (2, 6, 5, 2))
    future = torch.randint(0, 1024, (2, 1, 5))
    token_output = token_model(tokens, target_s1=future)
    assert tuple(token_output.s1_logits.shape) == (2, 6, 5, 1024)
    assert tuple(token_output.forecast.s1_logits.shape) == (2, 1, 5, 1024)
    observed_activations = tuple(
        str(block.graph_scorer.cfg.graph_activation)
        for block in token_model.backbone.st_blocks
    )
    assert observed_activations == LAYER_ACTIVATIONS
    for graph in token_output.graph_sequences:
        assert graph is not None
        _assert_graph_rows(graph)

    continuous_payload = token_config.to_dict()
    continuous_payload["mode"] = "continuous"
    continuous_payload["input_channels"] = 5
    continuous_config = BaseDyGraphFinancialConfig.from_dict(continuous_payload)
    continuous_model = OfficialBaseDyGraphContinuousForecaster(
        continuous_config,
        external_source_dir=str(source_dir),
    )
    continuous_output = continuous_model(torch.randn(2, 6, 5, 5))
    assert tuple(continuous_output.predictions.shape) == (2, 1, 5, 1)
    observed_activations = tuple(
        str(block.graph_scorer.cfg.graph_activation)
        for block in continuous_model.backbone.st_blocks
    )
    assert observed_activations == LAYER_ACTIVATIONS


def main() -> None:
    test_spec_contract()
    test_serialisation_compatibility()
    test_teacher_forced_one_step_contract()
    test_continuous_one_and_multi_horizon_contracts()
    test_autoregressive_candle_bridge()
    test_real_official_integration_if_available()
    print("BaseDyGraph four-block sparsemax diagnostic contracts passed.")


if __name__ == "__main__":
    main()

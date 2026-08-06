from __future__ import annotations

"""CPU contracts for the financial BaseDyGraph curiosity runner."""

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import src.models.basedygraph_financial as financial
from src.models.basedygraph_financial import (
    BaseDyGraphFinancialConfig,
    BaseDyGraphGraphRegularisationConfig,
    OfficialBaseDyGraphCoarsePathForecaster,
    OfficialBaseDyGraphContinuousForecaster,
    graph_regularisation_loss,
)
from src.training.run_basedygraph_financial import (
    EXPERIMENT_SPECS,
    _resolved_config_payload,
    _save_continuous_bundle,
    _save_token_bundle,
)
from src.evaluation.dynamic_graph_evaluation import (
    load_evaluation_artifacts,
    load_unified_run_info,
)


class _Temporal(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor) -> Tensor:
        return values + 0.05 * self.projection(values)


class _StaticScorer(nn.Module):
    def __init__(self, heads: int, nodes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(heads, nodes, nodes))
        nn.init.normal_(self.logits, std=0.05)

    def forward(self, hidden: Tensor, state_ids: Tensor) -> Tensor:
        del state_ids
        batch, steps = int(hidden.shape[0]), int(hidden.shape[1])
        adjacency = torch.softmax(self.logits, dim=-1)
        return adjacency.view(1, 1, *adjacency.shape).expand(
            batch, steps, -1, -1, -1
        )


class _DynamicScorer(nn.Module):
    def __init__(self, d_model: int, heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.edge_dim = hidden_dim // heads
        self.q = nn.Linear(d_model, hidden_dim, bias=False)
        self.k = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, hidden: Tensor, state_ids: Tensor) -> Tensor:
        del state_ids
        batch, steps, nodes, _ = hidden.shape
        query = self.q(hidden).reshape(
            batch, steps, nodes, self.heads, self.edge_dim
        ).permute(0, 1, 3, 2, 4)
        key = self.k(hidden).reshape(
            batch, steps, nodes, self.heads, self.edge_dim
        ).permute(0, 1, 3, 2, 4)
        logits = torch.einsum("btgid,btgjd->btgij", query, key)
        logits = logits / (self.edge_dim ** 0.5)
        return torch.softmax(logits, dim=-1)


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
        mean_graph = adjacency.mean(dim=2)
        messages = torch.einsum("btij,btjd->btid", mean_graph, hidden)
        return hidden + 0.1 * self.output(messages)


class _Block(nn.Module):
    def __init__(self, config: SimpleNamespace, graph_type: str) -> None:
        super().__init__()
        self.temporal_module = _Temporal(config.d_model)
        if graph_type == "static_graph":
            self.graph_scorer = _StaticScorer(
                config.num_edge_heads,
                config.num_nodes,
            )
        elif graph_type == "dynamic_graph":
            self.graph_scorer = _DynamicScorer(
                config.d_model,
                config.num_edge_heads,
                config.graph_hidden_dim,
            )
        else:
            self.graph_scorer = None
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
            [
                _Block(config, config.spatial_module_type)
                for _ in range(config.num_st_blocks)
            ]
        )

    def _initial_embedding_bntd(self, state_ids: Tensor) -> Tensor:
        values = self.state_embedding(state_ids)
        node_ids = torch.arange(state_ids.shape[1], device=state_ids.device)
        values = values + self.node_embedding(node_ids).view(1, -1, 1, values.shape[-1])
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


class _FakeModelModule:
    DiscreteSTGraphBackbone = _FakeBackbone


def _fake_official_config(run_config: object, **_: object) -> SimpleNamespace:
    return SimpleNamespace(
        num_states=run_config.num_states,
        num_nodes=run_config.num_nodes,
        d_model=run_config.d_model,
        num_edge_heads=run_config.num_edge_heads,
        graph_hidden_dim=run_config.graph_hidden_dim,
        num_st_blocks=run_config.num_st_blocks,
        spatial_module_type=run_config.spatial_module_type,
    )


def _patch_official():
    modules = SimpleNamespace(model=_FakeModelModule, commit="fake-pinned-commit")
    return (
        patch.object(financial, "load_official_basedygraph_modules", return_value=modules),
        patch.object(financial, "build_official_model_config", side_effect=_fake_official_config),
    )


def _assert_graph(graph: Tensor, *, batch: int, heads: int, nodes: int) -> None:
    assert tuple(graph.shape) == (batch, heads, nodes, nodes)
    assert torch.isfinite(graph).all()
    assert (graph >= 0).all()
    assert torch.allclose(
        graph.sum(dim=-1),
        torch.ones(batch, heads, nodes),
        atol=1.0e-5,
    )


def test_spec_matrix() -> None:
    assert len(EXPERIMENT_SPECS) == 7
    assert sum(spec.mode == "token" for spec in EXPERIMENT_SPECS) == 2
    assert sum(spec.mode == "continuous" for spec in EXPERIMENT_SPECS) == 5
    for spec in EXPERIMENT_SPECS:
        assert spec.run_name.startswith("DO_NOT_REPORT")
        assert spec.config.d_model == 96
        assert spec.config.num_st_blocks == 3
        assert spec.config.graph_heads == 2
        assert spec.config.graph_hidden_dim == 64
        assert spec.config.evaluation_horizons == (1, 5, 15, 30, 60)
    window_reg = next(
        spec for spec in EXPERIMENT_SPECS
        if spec.name == "continuous_dynamic_window_h3"
    )
    assert window_reg.config.regularisation.temporal_smooth_weight == 0.0


def test_regularisation() -> None:
    torch.manual_seed(5)
    raw = torch.randn(2, 6, 2, 5, 5, requires_grad=True)
    graph = torch.softmax(raw, dim=-1)
    config = BaseDyGraphGraphRegularisationConfig(
        target_entropy=1.4,
        target_entropy_weight=1.0,
        temporal_smooth_weight=0.01,
        direct_entropy_weight=0.0,
        warmup_epochs=5,
        layer=-1,
    )
    result = graph_regularisation_loss((graph,), config, epoch=2)
    assert math.isclose(result.warmup_factor, 0.4)
    assert result.total.requires_grad
    result.total.backward()
    assert raw.grad is not None
    assert float(raw.grad.abs().sum()) > 0.0


def test_token_and_continuous_shapes() -> None:
    patch_load, patch_config = _patch_official()
    with patch_load, patch_config:
        token_config = BaseDyGraphFinancialConfig(
            mode="token",
            graph_type="dynamic_graph",
            graph_scope="per_timestep",
            context_length=6,
            prediction_length=5,
            evaluation_horizons=(1, 3, 5),
            num_nodes=5,
            d_model=12,
            temporal_heads=3,
            graph_heads=2,
            graph_hidden_dim=8,
            num_st_blocks=3,
            future_predictor_heads=3,
            regularisation=BaseDyGraphGraphRegularisationConfig(
                target_entropy=1.4,
                target_entropy_weight=0.05,
                temporal_smooth_weight=0.01,
                warmup_epochs=5,
            ),
        )
        model = OfficialBaseDyGraphCoarsePathForecaster(token_config)
        token_ids = torch.randint(0, 1024, (2, 6, 5, 2))
        targets = torch.randint(0, 1024, (2, 5, 5))
        output = model(token_ids, target_s1=targets)
        assert tuple(output.forecast.s1_logits.shape) == (2, 5, 5, 1024)
        assert len(output.graph_sequences) == 3
        _assert_graph(output.forecast.graph.selected, batch=2, heads=2, nodes=5)
        loss = F.cross_entropy(
            output.forecast.s1_logits.reshape(-1, 1024), targets.reshape(-1)
        )
        regularisation = graph_regularisation_loss(
            output.graph_sequences,
            token_config.regularisation,
            epoch=5,
        )
        (loss + regularisation.total).backward()
        q_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if ".q." in name and parameter.grad is not None
        )
        assert q_grad > 0.0

        window_config = BaseDyGraphFinancialConfig(
            mode="continuous",
            graph_type="dynamic_graph",
            graph_scope="per_window",
            context_length=6,
            prediction_length=5,
            evaluation_horizons=(1, 3, 5),
            num_nodes=5,
            input_channels=5,
            d_model=12,
            temporal_heads=3,
            graph_heads=2,
            graph_hidden_dim=8,
            num_st_blocks=3,
            future_predictor_heads=3,
            regularisation=BaseDyGraphGraphRegularisationConfig(
                target_entropy=1.4,
                target_entropy_weight=1.0,
                temporal_smooth_weight=0.0,
            ),
        )
        continuous = OfficialBaseDyGraphContinuousForecaster(window_config)
        values = torch.randn(2, 6, 5, 5)
        continuous_output = continuous(values)
        assert tuple(continuous_output.predictions.shape) == (2, 3, 5, 1)
        _assert_graph(
            continuous_output.graph.selected,
            batch=2,
            heads=2,
            nodes=5,
        )
        for sequence in continuous_output.graph_sequences:
            assert sequence is not None
            reference = sequence[:, :1].expand_as(sequence)
            assert torch.allclose(sequence, reference)



def test_real_official_integration_if_available() -> None:
    """Exercise the new adapters against the pinned submodule when present.

    Repository archives used by lightweight CI may omit submodule contents;
    Colab initialises the pinned checkout before running this suite, so the
    actual official block API and exact context-backbone parity are tested
    there before any expensive financial run starts.
    """

    repository_root = Path(__file__).resolve().parents[1]
    source_dir = repository_root / "external" / "BaseDyGraph" / "src"
    required = tuple(source_dir / name for name in ("utilities.py", "modules.py", "model.py"))
    if not all(path.is_file() for path in required):
        print("Pinned BaseDyGraph submodule absent; skipping real-adapter integration.")
        return

    torch.manual_seed(17)
    token_config = BaseDyGraphFinancialConfig(
        mode="token",
        graph_type="dynamic_graph",
        graph_scope="per_timestep",
        context_length=6,
        prediction_length=5,
        evaluation_horizons=(1, 3, 5),
        num_nodes=5,
        d_model=16,
        temporal_heads=4,
        graph_heads=2,
        graph_hidden_dim=16,
        num_st_blocks=3,
        future_predictor_heads=4,
        regularisation=BaseDyGraphGraphRegularisationConfig(
            target_entropy=1.4,
            target_entropy_weight=0.05,
            temporal_smooth_weight=0.01,
            warmup_epochs=5,
        ),
    )
    token_model = OfficialBaseDyGraphCoarsePathForecaster(
        token_config,
        external_source_dir=str(source_dir),
    ).eval()
    token_ids = torch.randint(0, 1024, (2, 6, 5, 2))
    with torch.inference_mode():
        encoding = token_model.context_encoder(token_ids[..., 0])
        state_ids = token_ids[..., 0].permute(0, 2, 1).contiguous()
        direct = token_model.context_encoder.backbone(state_ids)
        torch.testing.assert_close(
            encoding.context_memory,
            torch.as_tensor(direct["spatial_repr"]),
            atol=0.0,
            rtol=0.0,
        )
        generated = token_model(token_ids)
    assert tuple(generated.forecast.s1_logits.shape) == (2, 5, 5, 1024)
    assert len(generated.graph_sequences) == 3
    for sequence in generated.graph_sequences:
        assert sequence is not None
        assert tuple(sequence.shape) == (2, 6, 2, 5, 5)
    _assert_graph(generated.forecast.graph.selected, batch=2, heads=2, nodes=5)

    continuous_config = BaseDyGraphFinancialConfig(
        mode="continuous",
        graph_type="dynamic_graph",
        graph_scope="per_window",
        context_length=6,
        prediction_length=5,
        evaluation_horizons=(1, 3, 5),
        num_nodes=5,
        input_channels=5,
        d_model=16,
        temporal_heads=4,
        graph_heads=2,
        graph_hidden_dim=16,
        num_st_blocks=3,
        future_predictor_heads=4,
        regularisation=BaseDyGraphGraphRegularisationConfig(
            target_entropy=1.4,
            target_entropy_weight=1.0,
        ),
    )
    continuous_model = OfficialBaseDyGraphContinuousForecaster(
        continuous_config,
        external_source_dir=str(source_dir),
    ).eval()
    with torch.inference_mode():
        continuous_output = continuous_model(torch.randn(2, 6, 5, 5))
    assert tuple(continuous_output.predictions.shape) == (2, 3, 5, 1)
    for sequence in continuous_output.graph_sequences:
        assert sequence is not None
        assert tuple(sequence.shape) == (2, 6, 2, 5, 5)
        torch.testing.assert_close(sequence, sequence[:, :1].expand_as(sequence))
    _assert_graph(continuous_output.graph.selected, batch=2, heads=2, nodes=5)

def test_standard_artifact_layout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        prediction_result = {
            "y_pred": torch.ones(2, 5, 3, 1),
            "y_true": torch.ones(2, 5, 3, 1),
            "last_context_target": torch.ones(2, 3, 1),
            "sample_idx": torch.tensor([0, 1]),
            "origin_idx": torch.tensor([59, 74]),
            "target_indices": torch.ones(2, 5, dtype=torch.long),
            "channels": ["close"],
            "horizons": [1, 5, 15, 30, 60],
            "asset_cols": ["A", "B", "C"],
            "output_space": "raw",
        }
        graph = torch.full((2, 1, 3, 3), 1.0 / 3.0)
        graphs = {
            "selected": graph,
            "per_layer": (graph, graph, graph),
            "graph_orientation": "row=target,column=source",
            "asset_cols": ["A", "B", "C"],
            "num_layers": 3,
            "num_heads": 1,
        }
        metric_table = pd.DataFrame(
            {
                "metric": ["cumulative_log_change_mae"],
                "horizon": [1],
                "channel": ["close"],
                "value": [0.0],
            }
        )
        bundle = {
            "prediction_result": prediction_result,
            "metric_results": {"cumulative_log_change_mae": torch.zeros(5, 1)},
            "metric_table": metric_table,
            "graphs": graphs,
            "graph_summary": {},
            "selection_score": 0.0,
            "native_loss": 0.0,
            "seconds": 0.0,
        }
        _save_continuous_bundle(
            run_dir=run_dir,
            split="train",
            epoch=4,
            bundle=bundle,
        )
        required = (
            run_dir / "best_train_predictions.pt",
            run_dir / "best_train_graphs.pt",
            run_dir / "best_train_metric_table.csv",
            run_dir / "analysis/train/predictions.pt",
            run_dir / "analysis/train/graphs.pt",
            run_dir / "analysis/train/metric_table.csv",
        )
        assert all(path.is_file() for path in required)
        wrapper = torch.load(
            run_dir / "analysis/train/predictions.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert wrapper["epoch"] == 4
        assert "prediction_result" in wrapper




def test_standard_token_artifact_layout() -> None:
    with tempfile.TemporaryDirectory() as directory:
        run_dir = Path(directory)
        assets = [f"A{index}" for index in range(93)]
        prediction_result = {
            "y_pred": torch.ones(2, 5, 93, 1),
            "y_true": torch.ones(2, 5, 93, 1),
            "last_context_target": torch.ones(2, 93, 1),
            "sample_idx": torch.tensor([0, 1]),
            "origin_idx": torch.tensor([59, 74]),
            "target_indices": torch.tensor(
                [[60, 64, 74, 89, 119], [75, 79, 89, 104, 134]],
                dtype=torch.long,
            ),
            "channels": ["close"],
            "horizons": [1, 5, 15, 30, 60],
            "asset_cols": assets,
            "output_space": "raw",
        }
        graph = torch.full((2, 2, 93, 93), 1.0 / 93.0)
        graph_artifacts = {
            "selected": graph,
            "per_layer": (graph, graph, graph),
            "graph_orientation": "row=target,column=source",
            "asset_cols": assets,
            "num_layers": 3,
            "num_heads": 2,
        }
        metric_table = pd.DataFrame(
            {
                "metric": ["cumulative_log_change_mae"],
                "horizon": [1],
                "channel": ["close"],
                "value": [0.0],
            }
        )
        bundle = SimpleNamespace(
            prediction_result=prediction_result,
            metric_results={"cumulative_log_change_mae": torch.zeros(5, 1)},
            metric_table=metric_table,
            graph_artifacts=graph_artifacts,
            token_artifacts={"sampled_s1": torch.zeros(1, dtype=torch.long)},
            sampled_price_path_artifacts={
                "sampled_close_paths": torch.ones(1, 2, 60, 93, 1)
            },
            diagnostics={},
        )
        _save_token_bundle(
            run_dir=run_dir,
            split="test",
            policy="argmax",
            epoch=4,
            bundle=bundle,
        )
        required = (
            run_dir / "best_test_predictions.pt",
            run_dir / "best_test_graphs.pt",
            run_dir / "best_test_tokens.pt",
            run_dir / "best_test_sampled_price_paths.pt",
            run_dir / "best_test_metric_table.csv",
            run_dir / "analysis/test/argmax/predictions.pt",
            run_dir / "analysis/test/argmax/graphs.pt",
            run_dir / "analysis/test/argmax/tokens.pt",
            run_dir / "analysis/test/argmax/sampled_price_paths.pt",
            run_dir / "analysis/test/argmax/metric_table.csv",
            run_dir / "analysis/test/temperature_selection.json",
        )
        assert all(path.is_file() for path in required)
        selection = json.loads(
            (run_dir / "analysis/test/temperature_selection.json").read_text(
                encoding="utf-8"
            )
        )
        assert selection["selected_policy"] == "argmax"

        arguments = SimpleNamespace(
            learning_rate=1.0e-4,
            weight_decay=0.0,
            max_epochs=50,
            patience=8,
            gradient_clip_norm=1.0,
            train_batch_size=2,
            selection_batch_size=2,
            export_batch_size=8,
            num_workers=0,
            mixed_precision=True,
            seed=42,
            train_cache=Path("train.pt"),
            validation_cache=Path("val.pt"),
            test_cache=Path("test.pt"),
            data_dir=Path("data"),
            run_name=None,
        )
        config = _resolved_config_payload(
            spec=EXPERIMENT_SPECS[0],
            args=arguments,
        )
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "asset_cols": assets,
                    "best_epoch": 4,
                }
            ),
            encoding="utf-8",
        )
        loaded = load_evaluation_artifacts(
            run_dir,
            split="test",
            policy=None,
            require_graph=True,
            require_metrics=True,
            require_sampled_paths=True,
        )
        assert loaded.paths.policy == "argmax"
        assert loaded.prediction_result["y_pred"].shape == (2, 5, 93, 1)
        assert loaded.graph_artifacts is not None
        assert loaded.graph_artifacts["selected"].shape == (2, 2, 93, 93)


def test_graph_hub_schema() -> None:
    arguments = SimpleNamespace(
        learning_rate=1.0e-4,
        weight_decay=0.0,
        max_epochs=50,
        patience=8,
        gradient_clip_norm=1.0,
        train_batch_size=2,
        selection_batch_size=2,
        export_batch_size=8,
        num_workers=0,
        mixed_precision=True,
        seed=42,
        train_cache=Path("train.pt"),
        validation_cache=Path("val.pt"),
        test_cache=Path("test.pt"),
        data_dir=Path("data"),
        run_name=None,
    )
    for spec in (EXPERIMENT_SPECS[0], EXPERIMENT_SPECS[2]):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            config = _resolved_config_payload(spec=spec, args=arguments)
            (run_dir / "resolved_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "asset_cols": [f"A{index}" for index in range(93)],
                        "best_epoch": 1,
                    }
                ),
                encoding="utf-8",
            )
            info = load_unified_run_info(run_dir)
            assert info.run_kind == ("token" if spec.mode == "token" else "continuous")
            assert info.num_nodes == 93
            assert info.num_heads == 2
            assert info.horizons == (1, 5, 15, 30, 60)


def main() -> None:
    test_spec_matrix()
    test_regularisation()
    test_token_and_continuous_shapes()
    test_real_official_integration_if_available()
    test_standard_artifact_layout()
    test_standard_token_artifact_layout()
    test_graph_hub_schema()
    print("Financial BaseDyGraph curiosity contracts passed.")


if __name__ == "__main__":
    main()

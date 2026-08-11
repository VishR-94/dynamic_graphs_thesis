from __future__ import annotations

"""Focused CPU contracts for the final token and BaseDyGraph-V2 notebook.

The suite intentionally avoids the external ModernTCN checkout by using the
same small fake feature extractor as the established token Round-2 tests.  The
vendored Dimitri V2 source *is* executed and compared against its unchanged
reference backbone.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import torch
from torch import Tensor, nn

from src.data.cached_token_graph_dataset import (
    CachedTokenGraphDataset,
    REAL_TOKEN_GRAPH_REPRESENTATION,
)
from src.evaluation.dynamic_graph_evaluation import (
    analyse_coarse_token_predictive_distribution,
    analyse_coarse_token_topk,
    analyse_graph,
    load_evaluation_artifacts,
    load_model_sampled_path_bundle,
    make_model_artifact_audit,
)
from src.models.final_token_v2_models import (
    DIMITRI_NOTEBOOK_DEFAULTS,
    DenseTransformerTokenForecaster,
    DimitriV2DenseContinuousForecaster,
    DimitriV2DenseTokenForecaster,
    load_dimitri_v2_architecture,
)
from src.models.modern_tcn_graph_round2_token import (
    ModernTCNGraphRound2TokenModel,
    token_round2_model_config_from_mapping,
)
from src.training.final_token_v2_specs import make_final_token_v2_specs
from src.training.run_final_token_v2_experiment import (
    _decode_token_split,
    _hybrid_dense_token_targets,
    _train_token_epoch,
    _export_continuous_split,
    _export_token_split,
    _metadata,
    _new_grad_scaler,
    _save_continuous_export,
    _save_token_export,
)


class _FakeModernTCNInner(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.patch_stride = int(config.patch_stride)
        self.padding = self.patch_size - self.patch_stride
        self.d_model = int(config.dims[0])
        self.stem = nn.Linear(self.patch_size, self.d_model)

    def forward_feature(self, values: Tensor) -> Tensor:
        if self.padding:
            values = torch.cat(
                [
                    values,
                    values[..., -1:].expand(*values.shape[:-1], self.padding),
                ],
                dim=-1,
            )
        patches = values.unfold(-1, self.patch_size, self.patch_stride)
        features = self.stem(patches)
        return features.permute(0, 1, 3, 2).contiguous()


class _FakeOfficialModernTCN(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.model = _FakeModernTCNInner(config)


class _FakeCoarseDecoder:
    def decode_coarse_token_path(
        self,
        context_token_ids: Tensor,
        future_s1_ids: Tensor,
        *,
        mean: Tensor,
        std: Tensor,
        series_batch_size: int,
        return_full_path: bool,
    ) -> Tensor:
        del context_token_ids, mean, std, series_batch_size
        if return_full_path:
            raise AssertionError("Only the future path should be requested.")
        ids = torch.as_tensor(future_s1_ids).float()
        close = 100.0 + ids * 1.0e-5
        open_values = close
        high = close + 0.01
        low = close - 0.01
        volume = torch.full_like(close, 1000.0)
        return torch.stack((open_values, high, low, close, volume), dim=-1)


def _uniform_graph(nodes: int) -> Tensor:
    values = torch.ones(nodes, nodes) - torch.eye(nodes)
    return values / values.sum(dim=-1, keepdim=True)


def _token_cache(
    *, windows: int = 3, context: int = 4, prediction: int = 4, nodes: int = 3
) -> dict:
    torch.manual_seed(71)
    evaluation_horizons = (1, 2, 4)
    context_tokens = torch.randint(
        0, 1024, (windows, context, nodes, 2), dtype=torch.int16
    )
    target_s1 = torch.randint(
        0, 1024, (windows, prediction, nodes), dtype=torch.int16
    )
    target_s2 = torch.randint(
        0, 1024, (windows, prediction, nodes), dtype=torch.int16
    )
    last_close = 100.0 + 0.1 * torch.arange(nodes).float()
    last = torch.zeros(windows, nodes, 5)
    last[..., 0] = last_close
    last[..., 1] = last_close + 0.01
    last[..., 2] = last_close - 0.01
    last[..., 3] = last_close
    last[..., 4] = 1000.0
    truth = torch.zeros(windows, len(evaluation_horizons), nodes, 5)
    for index, horizon in enumerate(evaluation_horizons):
        close = last_close + 0.001 * horizon
        truth[:, index, :, 0] = close
        truth[:, index, :, 1] = close + 0.01
        truth[:, index, :, 2] = close - 0.01
        truth[:, index, :, 3] = close
        truth[:, index, :, 4] = 1000.0
    origin = torch.arange(windows) + context - 1
    target_indices = origin[:, None] + torch.arange(1, prediction + 1)[None]
    return {
        "format_version": 2,
        "representation": REAL_TOKEN_GRAPH_REPRESENTATION,
        "context_tokens": context_tokens,
        "target_s1": target_s1,
        "target_s2": target_s2,
        "context_mean": torch.zeros(windows, nodes, 6),
        "context_std": torch.ones(windows, nodes, 6),
        "evaluation_true": truth,
        "last_context_target": last,
        "sample_idx": torch.arange(windows),
        "origin_idx": origin,
        "target_indices": target_indices,
        "dates": [f"2024-01-{index + 2:02d}" for index in range(windows)],
        "asset_cols": [f"A{index}" for index in range(nodes)],
        "input_channels": ["open", "high", "low", "close", "volume"],
        "target_channels": ["close"],
        "context_length": context,
        "prediction_length": prediction,
        "dense_horizons": list(range(1, prediction + 1)),
        "evaluation_horizons": list(evaluation_horizons),
        "evaluation_indices": [value - 1 for value in evaluation_horizons],
        "tokenizer_channels": [
            "open", "high", "low", "close", "volume", "amount"
        ],
        "future_clipping_rate_percent_by_step_channel": torch.zeros(prediction, 6),
        "future_clipping_rate_percent_by_channel": torch.zeros(6),
        "context_clipping_rate_percent_by_step_channel": torch.zeros(context, 6),
        "context_clipping_rate_percent_by_channel": torch.zeros(6),
        "s1_id_space": "kronos_original",
        "s1_vocabulary_size": 1024,
    }


def _raw_split(*, nodes: int = 3, sessions: int = 3, length: int = 12) -> dict:
    torch.manual_seed(17)
    samples = []
    for day in range(sessions):
        close = 100.0 + torch.cumsum(0.01 * torch.randn(length, nodes), dim=0)
        open_values = close + 0.001 * torch.randn_like(close)
        high = torch.maximum(open_values, close) + 0.01
        low = torch.minimum(open_values, close) - 0.01
        volume = 1000.0 + 5.0 * torch.rand_like(close)
        amount = torch.zeros_like(close)
        values = torch.stack(
            (open_values, high, low, close, volume, amount), dim=-1
        )
        samples.append((values, {}, f"2024-01-{day + 2:02d}"))
    return {
        "samples": samples,
        "asset_cols": [f"A{index}" for index in range(nodes)],
        "channels": ["open", "high", "low", "close", "volume", "amount"],
    }


def _small_dense_config(config: dict) -> dict:
    values = deepcopy(config)
    values["data"]["context_length"] = 4
    values["data"]["prediction_length"] = 4
    values["data"]["evaluation_horizons"] = [1, 2, 4]
    values["data"]["stride"] = 2
    values["model"]["num_nodes"] = 3
    values["models"]["dynamic_graph"]["num_nodes"] = 3
    values["models"]["dynamic_graph"]["heads"]["prediction_length"] = 4
    values["models"]["dynamic_graph"]["heads"]["evaluation_horizons"] = [1, 2, 4]
    values["models"]["dynamic_graph"]["future_predictor"]["prediction_length"] = 4
    values["training"]["batch_size"] = 1
    values["training"]["selection_batch_size"] = 1
    values["training"]["export_batch_size"] = 1
    values["training"]["num_workers"] = 0
    values["training"]["mixed_precision"] = False
    if values["training"]["loss"].get("dense_origins", False):
        values["training"]["loss"]["dense_auxiliary_horizons"] = [1, 2, 4]
        values["training"]["loss"]["final_origin_future_steps"] = 4
    return values


def _small_modern_config(config: dict) -> dict:
    values = deepcopy(config)
    values["data"]["context_length"] = 4
    values["data"]["prediction_length"] = 4
    values["data"]["evaluation_horizons"] = [1, 2, 4]
    values["model"]["temporal_stack"]["modern_tcn"]["patch_size"] = 2
    values["model"]["temporal_stack"]["modern_tcn"]["patch_stride"] = 2
    values["model"]["future_predictor"]["prediction_length"] = 4
    values["models"]["dynamic_graph"]["num_nodes"] = 3
    values["models"]["dynamic_graph"]["heads"]["prediction_length"] = 4
    values["models"]["dynamic_graph"]["heads"]["evaluation_horizons"] = [1, 2, 4]
    values["models"]["dynamic_graph"]["future_predictor"]["prediction_length"] = 4
    return values


def _small_continuous_config(config: dict) -> dict:
    values = deepcopy(config)
    values["data"]["context_length"] = 4
    values["data"]["horizons"] = [1, 2, 4]
    values["data"]["stride"] = 2
    values["training"]["loss"]["horizon_reference_mae"] = [0.1, 0.2, 0.4]
    values["training"]["loss"]["horizon_weights"] = [1.7142857143, 0.8571428571, 0.4285714286]
    values["training"]["batch_size"] = 1
    values["training"]["selection_batch_size"] = 1
    values["training"]["export_batch_size"] = 1
    values["training"]["num_workers"] = 0
    return values


def _test_specs() -> None:
    specs = make_final_token_v2_specs()
    if len(specs) != 4 or len({spec.run_name for spec in specs}) != 4:
        raise AssertionError("Expected four unique final-comparison specifications.")
    by_kind = {spec.model_kind: spec for spec in specs}
    expected = {
        "modern_tcn_token",
        "dense_transformer_token",
        "dimitri_v2_token",
        "dimitri_v2_continuous",
    }
    if set(by_kind) != expected:
        raise AssertionError(f"Unexpected model kinds: {sorted(by_kind)}")

    modern = by_kind["modern_tcn_token"].config
    if modern["model"]["temporal_stack"]["num_transformer_blocks"] != 0:
        raise AssertionError("ModernTCN counterpart gained a Transformer refinement.")
    if modern["model"]["graph"]["activations_per_block"] != ["softmax"]:
        raise AssertionError("Selected ModernTCN graph activation changed.")
    if modern["model"]["prior"]["type"] != "correlation":
        raise AssertionError("Selected correlation prior was lost.")
    if modern["training"]["loss"]["horizon_weighting"] != "uniform":
        raise AssertionError("Token CE must not be horizon downweighted.")

    dense = by_kind["dense_transformer_token"].config
    if dense["model"]["num_st_blocks"] != 3:
        raise AssertionError("Winning dense Transformer must have three ST blocks.")
    if dense["model"]["temporal"] != {
        "type": "transformer",
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_multiplier": 2,
        "dropout": 0.0,
        "position_embedding": False,
    }:
        raise AssertionError("Dense Transformer temporal contract changed.")
    if dense["model"]["graph"]["num_heads_per_block"] != [1, 1, 1]:
        raise AssertionError("Dense Transformer graph-head schedule changed.")
    if dense["model"]["graph"]["activations_per_block"] != [
        "softmax", "softmax", "sparsemax"
    ]:
        raise AssertionError("Dense Transformer activation schedule changed.")
    dense_loss = dense["training"]["loss"]
    if not dense_loss["dense_origins"]:
        raise AssertionError("Dense token supervision is disabled.")
    if dense_loss["dense_objective"] != (
        "internal_five_horizons_plus_final_full_path"
    ):
        raise AssertionError("Dense token objective is not the hybrid contract.")
    if dense_loss["dense_auxiliary_horizons"] != [1, 5, 15, 30, 60]:
        raise AssertionError("Dense auxiliary horizons changed.")
    if dense_loss["dense_auxiliary_weight"] != 1.0:
        raise AssertionError("Dense auxiliary loss weight changed.")
    if dense_loss["final_origin_future_steps"] != 60:
        raise AssertionError("Final origin must predict all 60 future steps.")

    for kind in ("dimitri_v2_token", "dimitri_v2_continuous"):
        values = by_kind[kind].config
        defaults = values["model"]["dimitri_defaults"]
        for key, expected_value in DIMITRI_NOTEBOOK_DEFAULTS.items():
            if defaults[key] != expected_value:
                raise AssertionError(f"{kind}: Dimitri default {key} changed.")
        if defaults["temporal_context_window"] != 60:
            raise AssertionError("Dimitri context length is not 60.")
        if values["model"]["graph"]["num_heads_per_block"] != [6, 6, 6, 1]:
            raise AssertionError("Resolved V2 graph-head schedule changed.")
        if values["model"]["graph"]["activations_per_block"] != [
            "softmax", "softmax", "softmax", "sparsemax"
        ]:
            raise AssertionError("Resolved V2 activation schedule changed.")

    if by_kind["dimitri_v2_token"].config["training"]["selection_metric"] != (
        "mean_top1_accuracy_over_all_60_future_steps"
    ):
        raise AssertionError("V2 token selection metric changed.")
    v2_token_loss = by_kind["dimitri_v2_token"].config["training"]["loss"]
    if v2_token_loss["dense_objective"] != (
        "internal_five_horizons_plus_final_full_path"
    ):
        raise AssertionError("V2 token dense objective changed.")
    if v2_token_loss["dense_auxiliary_horizons"] != [1, 5, 15, 30, 60]:
        raise AssertionError("V2 token auxiliary horizons changed.")
    if v2_token_loss["final_origin_future_steps"] != 60:
        raise AssertionError("V2 token final path length changed.")
    if by_kind["dimitri_v2_continuous"].config["training"]["loss"][
        "horizon_weighting"
    ] != "inverse_reference_mae":
        raise AssertionError("V2 continuous weighted loss changed.")


def _test_hybrid_dense_targets() -> None:
    context = torch.tensor([[[0], [1], [2], [3]]])
    future = torch.tensor([[[4], [5], [6], [7]]])
    auxiliary, final_path = _hybrid_dense_token_targets(
        context,
        future,
        auxiliary_horizons=(1, 2, 4),
    )
    expected_auxiliary = torch.tensor(
        [
            [
                [[1], [2], [4]],
                [[2], [3], [5]],
                [[3], [4], [6]],
            ]
        ]
    )
    torch.testing.assert_close(auxiliary, expected_auxiliary)
    torch.testing.assert_close(final_path, future)


def _assert_row_stochastic(graph: Tensor) -> None:
    torch.testing.assert_close(
        graph.sum(dim=-1), torch.ones_like(graph.sum(dim=-1)), atol=3.0e-5, rtol=0.0
    )


def _test_dense_transformer() -> None:
    torch.manual_seed(5)
    model = DenseTransformerTokenForecaster(
        num_nodes=3, context_length=4, prediction_length=4
    )
    context = torch.randint(0, 1024, (1, 4, 3))
    initial = model.forward_backbone(context)
    if len(initial.selected_graphs) != 3:
        raise AssertionError("Dense Transformer lost ST blocks.")
    uniform = _uniform_graph(3).view(1, 1, 1, 3, 3)
    for graph in initial.selected_graphs:
        _assert_row_stochastic(graph)
        torch.testing.assert_close(graph, uniform.expand_as(graph), atol=2.0e-6, rtol=0.0)

    changed = context.clone()
    changed[:, 2:] = (changed[:, 2:] + 137) % 1024
    other = model.forward_backbone(changed)
    for first, second in zip(initial.selected_graphs, other.selected_graphs, strict=True):
        torch.testing.assert_close(first[:, :2], second[:, :2], atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(initial.hidden[:, :2], other.hidden[:, :2], atol=2.0e-6, rtol=0.0)
    first_logits = model.future_predictor.forward_origin(initial.hidden, 1)
    second_logits = model.future_predictor.forward_origin(other.hidden, 1)
    torch.testing.assert_close(first_logits, second_logits, atol=2.0e-6, rtol=0.0)
    auxiliary_logits = model.future_predictor.forward_origin(
        initial.hidden,
        1,
        future_position_indices=(0, 1, 3),
    )
    if tuple(auxiliary_logits.shape) != (1, 3, 3, 1024):
        raise AssertionError("Selected-position token head has the wrong shape.")

    target = torch.randint(0, 1024, (1, 4, 3))
    loss = torch.nn.functional.cross_entropy(first_logits.reshape(-1, 1024), target.reshape(-1))
    loss.backward()
    if model.future_predictor.classifier.weight.grad is None:
        raise AssertionError("Future token classifier received no gradient.")
    for block in model.blocks:
        learner = block.graph_learner
        if learner.static_logits.grad is None:
            raise AssertionError("Static graph received no gradient.")
        if learner.k_proj.weight.grad is None:
            raise AssertionError("Dynamic graph key projection received no gradient.")
        if learner.raw_alpha.grad is None or block.spatial_gate.raw_beta.grad is None:
            raise AssertionError("Alpha or beta gate received no gradient.")


def _test_hybrid_dense_training_step() -> None:
    specs = {spec.model_kind: spec for spec in make_final_token_v2_specs()}
    config = _small_dense_config(specs["dense_transformer_token"].config)
    dataset = CachedTokenGraphDataset(_token_cache(), validate=False)
    model = DenseTransformerTokenForecaster(
        num_nodes=3,
        context_length=4,
        prediction_length=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    values = _train_token_epoch(
        model=model,
        kind="dense_transformer_token",
        dataset=dataset,
        config=config,
        optimizer=optimizer,
        scaler=_new_grad_scaler(False),
        device=torch.device("cpu"),
        epoch=1,
    )
    required = {
        "training_objective",
        "training_final_path_cross_entropy",
        "training_final_path_top1_accuracy",
        "training_dense_auxiliary_cross_entropy",
        "training_dense_auxiliary_top1_accuracy",
    }
    if not required.issubset(values):
        raise AssertionError(
            "Hybrid dense training diagnostics are incomplete: "
            f"{sorted(set(required).difference(values))}"
        )
    if not all(torch.isfinite(torch.tensor(float(values[key]))) for key in required):
        raise AssertionError("Hybrid dense training produced a non-finite metric.")


def _test_v2() -> None:
    load_dimitri_v2_architecture()
    torch.manual_seed(13)
    model = DimitriV2DenseTokenForecaster(
        num_nodes=3, context_length=4, prediction_length=4
    )
    model.eval()
    context = torch.randint(0, 1024, (1, 4, 3))
    manual = model.forward_backbone(context, include_components=True)
    reference = model.backbone.reference(
        context.permute(0, 2, 1).contiguous()
    )
    torch.testing.assert_close(
        manual.hidden, reference["spatial_repr"], atol=2.0e-6, rtol=0.0
    )
    if len(manual.selected_graphs) != 4:
        raise AssertionError("V2 must expose four graph layers.")
    expected_heads = (6, 6, 6, 1)
    for index, (observed, expected) in enumerate(
        zip(manual.selected_graphs, reference["block_graph_attns"], strict=True)
    ):
        torch.testing.assert_close(observed, expected, atol=2.0e-6, rtol=0.0)
        if int(observed.shape[2]) != expected_heads[index]:
            raise AssertionError("V2 graph-head schedule changed.")
        _assert_row_stochastic(observed)

    changed = context.clone()
    changed[:, 2:] = (changed[:, 2:] + 219) % 1024
    other = model.forward_backbone(changed, include_components=False)
    torch.testing.assert_close(manual.hidden[:, :2], other.hidden[:, :2], atol=3.0e-6, rtol=0.0)
    for first, second in zip(manual.selected_graphs, other.selected_graphs, strict=True):
        torch.testing.assert_close(first[:, :2], second[:, :2], atol=3.0e-6, rtol=0.0)
    logits = model.future_predictor.forward_origin(manual.hidden, 1)
    other_logits = model.future_predictor.forward_origin(other.hidden, 1)
    torch.testing.assert_close(logits, other_logits, atol=3.0e-6, rtol=0.0)

    target = torch.randint(0, 1024, (1, 4, 3))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 1024), target.reshape(-1))
    loss.backward()
    if model.future_predictor.classifier.weight.grad is None:
        raise AssertionError("V2 future head received no gradient.")
    for block in model.backbone.reference.st_blocks:
        scorer = block.graph_scorer
        gradients = [parameter.grad for parameter in scorer.parameters() if parameter.requires_grad]
        if not any(value is not None for value in gradients):
            raise AssertionError("A V2 graph scorer received no gradient.")

    continuous = DimitriV2DenseContinuousForecaster(
        num_nodes=3, context_length=4, horizons=(1, 2, 4), input_channels=5
    )
    x = torch.randn(1, 4, 3, 5)
    output = continuous.forward_dense(x, include_components=True)
    if tuple(output.predictions.shape) != (1, 4, 3, 3, 1):
        raise AssertionError(f"Unexpected V2 price output {tuple(output.predictions.shape)}")
    if tuple(output.final_predictions().shape) != (1, 3, 3, 1):
        raise AssertionError("V2 final-origin price head has the wrong shape.")


def _build_small_modern(spec_config: dict) -> ModernTCNGraphRound2TokenModel:
    values = _small_modern_config(spec_config)
    config = token_round2_model_config_from_mapping(
        values, num_nodes=3, vocabulary_size=1024
    )
    return ModernTCNGraphRound2TokenModel(
        config,
        static_prior=_uniform_graph(3),
        official_model_cls=_FakeOfficialModernTCN,
    )


def _save_minimum_run_files(
    *, run_dir: Path, config: dict, metadata: dict, model: nn.Module
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        __import__("json").dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, run_dir / "best_checkpoint.pt")
    pd.DataFrame([{"epoch": 1, "selection_score": 0.1}]).to_csv(
        run_dir / "history.csv", index=False
    )


def _test_exports_and_graph_hub() -> None:
    specs = {spec.model_kind: spec for spec in make_final_token_v2_specs()}
    cache = _token_cache()
    dataset = CachedTokenGraphDataset(cache, validate=False)
    raw = _raw_split()
    dense_config = _small_dense_config(specs["dense_transformer_token"].config)
    dense_model = DenseTransformerTokenForecaster(
        num_nodes=3, context_length=4, prediction_length=4
    )
    datasets = {"train": dataset, "validation": dataset, "test": dataset}

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "dense-token"
        metadata = _metadata(
            config=dense_config,
            run_name="dense-token",
            model=dense_model,
            best_epoch=1,
            best_score=0.1,
            epochs_completed=1,
            datasets=datasets,
        )
        _save_minimum_run_files(
            run_dir=run_dir, config=dense_config, metadata=metadata, model=dense_model
        )
        for split_name in ("train", "validation", "test"):
            values = _export_token_split(
                model=dense_model,
                kind="dense_transformer_token",
                dataset=dataset,
                split_name=split_name,
                config=dense_config,
                device=torch.device("cpu"),
                checkpoint_epoch=1,
            )
            _save_token_export(run_dir, split_name, values)

        audit = make_model_artifact_audit({"dense-token": run_dir}, split="test")
        if not bool(audit.iloc[0]["Ready"]):
            raise AssertionError(f"Graph Hub audit failed: {audit.iloc[0]['Issue']}")
        artifacts = load_evaluation_artifacts(
            run_dir, split="test", policy="best", require_graph=True, require_metrics=True
        )
        if tuple(artifacts.prediction_result["y_pred"].shape) != (3, 3, 3, 1):
            raise AssertionError("Token public prediction shape is wrong.")
        report = analyse_graph(
            run_dir,
            split="test",
            policy="best",
            component="selected",
            layer=2,
            head=0,
            day=None,
            window=None,
            cluster=False,
        )
        if report.plotted_adjacency.shape != (3, 3):
            raise AssertionError("Graph Hub returned the wrong adjacency shape.")
        topk = analyse_coarse_token_topk(
            run_dir,
            split="test",
            source="best",
            max_windows=None,
            top_k_values=(1, 3, 5, 10),
            horizons=(1, 2, 4),
            random_seed=42,
        )
        if set(int(value) for value in topk.index) != {1, 2, 4}:
            raise AssertionError("Token Top-K analysis lost reported horizons.")

        args = SimpleNamespace(
            sample_count=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            decode_series_batch_size=8,
        )
        _decode_token_split(
            model=dense_model,
            kind="dense_transformer_token",
            dataset=dataset,
            split_name="validation",
            config=dense_config,
            device=torch.device("cpu"),
            tokenizer=_FakeCoarseDecoder(),
            args=args,
            train_split=raw,
            checkpoint_epoch=1,
            run_dir=run_dir,
        )
        sampled = load_model_sampled_path_bundle(
            run_dir, split="validation", policy="temperature_1"
        )
        if tuple(sampled.sampled_close_paths.shape) != (2, 3, 4, 3, 1):
            raise AssertionError("Decoded sampled Close paths have the wrong shape.")
        distribution = analyse_coarse_token_predictive_distribution(
            run_dir,
            split="validation",
            policy="temperature_1",
            asset=0,
            horizon=1,
            plot_top_n=5,
            bars=(
                "mean_probability",
                "hard_prediction_frequency",
                "training_target_frequency",
                "sampled_frequency",
            ),
            max_windows=None,
            random_seed=42,
        )
        if distribution["distribution_table"].empty:
            raise AssertionError("Token predictive-distribution analysis is empty.")

        # The one-block ModernTCN token counterpart must also construct with no
        # Transformer refinement and retain its selected graph contract.
        modern_model = _build_small_modern(specs["modern_tcn_token"].config)
        context = torch.randint(0, 1024, (1, 4, 3))
        modern_output = modern_model(context)
        if tuple(modern_output.s1_logits.shape) != (1, 4, 3, 1024):
            raise AssertionError("ModernTCN token counterpart has the wrong output shape.")
        if len(modern_output.block_outputs) != 1:
            raise AssertionError("ModernTCN token counterpart is not one ST block.")

        # Continuous V2 export uses the common raw-price metric and Graph-Hub
        # contracts on the same direct five-horizon tensor.
        continuous_config = _small_continuous_config(
            specs["dimitri_v2_continuous"].config
        )
        from src.training.run_final_token_v2_experiment import _continuous_dataset

        continuous_dataset = _continuous_dataset(raw, continuous_config)
        continuous_model = DimitriV2DenseContinuousForecaster(
            num_nodes=3, context_length=4, horizons=(1, 2, 4), input_channels=5
        )
        continuous_dir = root / "v2-price"
        continuous_datasets = {
            "train": continuous_dataset,
            "validation": continuous_dataset,
            "test": continuous_dataset,
        }
        continuous_metadata = _metadata(
            config=continuous_config,
            run_name="v2-price",
            model=continuous_model,
            best_epoch=1,
            best_score=0.2,
            epochs_completed=1,
            datasets=continuous_datasets,
        )
        _save_minimum_run_files(
            run_dir=continuous_dir,
            config=continuous_config,
            metadata=continuous_metadata,
            model=continuous_model,
        )
        exported = _export_continuous_split(
            model=continuous_model,
            dataset=continuous_dataset,
            split_name="train",
            config=continuous_config,
            device=torch.device("cpu"),
            checkpoint_epoch=1,
            train_split=raw,
            bootstrap=False,
        )
        _save_continuous_export(continuous_dir, "train", exported)
        loaded = load_evaluation_artifacts(
            continuous_dir,
            split="train",
            policy="best",
            require_graph=True,
            require_metrics=True,
        )
        if tuple(loaded.prediction_result["y_pred"].shape[1:]) != (3, 3, 1):
            raise AssertionError("V2 continuous export has the wrong shape.")
        price_report = analyse_graph(
            continuous_dir,
            split="train",
            policy="best",
            component="selected",
            layer=-1,
            head=0,
            day=None,
            window=None,
            cluster=False,
        )
        if price_report.plotted_adjacency.shape != (3, 3):
            raise AssertionError("V2 continuous graph loading failed.")


def main() -> None:
    _test_specs()
    _test_hybrid_dense_targets()
    _test_dense_transformer()
    _test_hybrid_dense_training_step()
    _test_v2()
    _test_exports_and_graph_hub()
    print("Final token and BaseDyGraph-V2 comparison contracts passed.")


if __name__ == "__main__":
    main()

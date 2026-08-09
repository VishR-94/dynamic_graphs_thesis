from __future__ import annotations

"""CPU contracts for the 12-model coarse-token Round-2 sweep."""

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
    analyse_graph,
    load_model_sampled_path_bundle,
    make_model_artifact_audit,
)
from src.models.modern_tcn_graph_round2_token import (
    ModernTCNGraphRound2TokenModel,
    token_round2_model_config_from_mapping,
)
from src.training.modern_tcn_round2_token_specs import (
    make_token_round2_specs,
)
from src.training.run_modern_tcn_graph_round2_token import (
    _build_loader,
    _decode_sampled_split,
    _export_selected_checkpoint,
    _save_export,
    _set_schedule_for_epoch,
    _token_batch_sums,
    _token_metric_long_table,
    _token_metric_table,
    _validate_config,
)
from src.training.run_dynamic_graph import atomic_json_save


class _FakeModernTCNInner(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.patch_stride = int(config.patch_stride)
        self.padding = self.patch_size - self.patch_stride
        self.d_model = int(config.dims[0])
        self.stem = nn.Linear(self.patch_size, self.d_model)

    def forward_feature(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Fake ModernTCN expects [B,M,T].")
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
            raise AssertionError("The runner must request only the future path.")
        ids = torch.as_tensor(future_s1_ids).float()
        close = 100.0 + ids * 1.0e-4
        open_values = close
        high = close + 0.01
        low = close - 0.01
        volume = torch.full_like(close, 1000.0)
        return torch.stack((open_values, high, low, close, volume), dim=-1)


def _cache(*, windows: int = 4, context: int = 8, prediction: int = 4, nodes: int = 4) -> dict:
    torch.manual_seed(91)
    evaluation_horizons = (1, 2, 4)
    evaluation_indices = tuple(value - 1 for value in evaluation_horizons)
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
    for horizon_index, horizon in enumerate(evaluation_horizons):
        close = last_close + horizon * 0.001
        truth[:, horizon_index, :, 0] = close
        truth[:, horizon_index, :, 1] = close + 0.01
        truth[:, horizon_index, :, 2] = close - 0.01
        truth[:, horizon_index, :, 3] = close
        truth[:, horizon_index, :, 4] = 1000.0
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
        "evaluation_indices": list(evaluation_indices),
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


def _raw_train_split(nodes: int = 4) -> dict:
    torch.manual_seed(19)
    samples = []
    for day in range(3):
        close = 100.0 + torch.cumsum(0.01 * torch.randn(20, nodes), dim=0)
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


def _prior(nodes: int = 4) -> Tensor:
    values = torch.ones(nodes, nodes) - torch.eye(nodes)
    return values / values.sum(dim=-1, keepdim=True)


def _small_specs():
    return make_token_round2_specs(
        prior_type="correlation",
        graph_heads=1,
        context_length=8,
        prediction_length=4,
        evaluation_horizons=(1, 2, 4),
        transformer_d_model=8,
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_feedforward_multiplier=2,
        future_predictor_num_layers=1,
        future_predictor_num_heads=2,
        future_predictor_feedforward_multiplier=2,
        modern_tcn_graph_dim_per_head=4,
        transformer_graph_dim_per_head=8,
        train_batch_size=2,
        selection_batch_size=2,
        export_batch_size=2,
        max_epochs=3,
        patience=2,
    )


def _build_small_model(spec, *, nodes: int = 4) -> ModernTCNGraphRound2TokenModel:
    resolved = spec.config
    resolved["models"]["dynamic_graph"]["num_nodes"] = nodes
    config = token_round2_model_config_from_mapping(
        resolved,
        num_nodes=nodes,
        vocabulary_size=1024,
    )
    source = _prior(nodes) if config.uses_static_graph else None
    return ModernTCNGraphRound2TokenModel(
        config,
        static_prior=source,
        official_model_cls=_FakeOfficialModernTCN,
    )


def _assert_model_contract(model: ModernTCNGraphRound2TokenModel) -> None:
    torch.manual_seed(7)
    context = torch.randint(
        0,
        model.config.vocabulary_size,
        (2, model.config.context_length, model.config.num_nodes),
    )
    output = model(context)
    expected_logits = (
        2,
        model.config.prediction_length,
        model.config.num_nodes,
        model.config.vocabulary_size,
    )
    if tuple(output.s1_logits.shape) != expected_logits:
        raise AssertionError(
            f"Token logits {tuple(output.s1_logits.shape)} != {expected_logits}."
        )
    for block_index, block in enumerate(output.block_outputs):
        graph = block.graph.selected.detach().float()
        expected_graph = (
            2,
            model.config.graph_heads_per_block[block_index],
            model.config.num_nodes,
            model.config.num_nodes,
        )
        if tuple(graph.shape) != expected_graph:
            raise AssertionError("Saved graph shape differs from block schedule.")
        torch.testing.assert_close(
            graph.sum(dim=-1),
            torch.ones_like(graph.sum(dim=-1)),
            atol=2.0e-6,
            rtol=0.0,
        )
        diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diagonal,
            torch.zeros_like(diagonal),
            atol=0.0,
            rtol=0.0,
        )
    if not torch.any(output.block_outputs[-1].graph.selected == 0):
        raise AssertionError("Final sparsemax graph produced no exact zeros.")

    target = torch.randint(
        0,
        model.config.vocabulary_size,
        output.selected_s1.shape,
    )
    loss = torch.nn.functional.cross_entropy(
        output.s1_logits.reshape(-1, model.config.vocabulary_size),
        target.reshape(-1),
    )
    loss.backward()
    if model.embedding.s1_embedding.weight.grad is None:
        raise AssertionError("Coarse state embedding received no gradient.")
    if model.config.uses_state_pathway:
        for block in model.graph_spatial_blocks:
            if block.graph_learner.q_proj.weight.grad is None:
                raise AssertionError("State-aware graph scorer received no gradient.")


def main() -> None:
    specs = _small_specs()
    if len(specs) != 12 or len({spec.run_name for spec in specs}) != 12:
        raise AssertionError("Token Round 2 did not create twelve unique runs.")
    family_counts = {
        family: sum(spec.graph_family == family for spec in specs)
        for family in ("dynamic_only", "prior_state")
    }
    if family_counts != {"dynamic_only": 6, "prior_state": 6}:
        raise AssertionError(f"Unexpected graph-family counts {family_counts}.")
    temporal_grid = {
        (spec.temporal_family, spec.num_transformer_blocks)
        for spec in specs
        if spec.graph_family == "dynamic_only"
    }
    expected_grid = {
        ("modern_tcn_transformer", 1),
        ("modern_tcn_transformer", 2),
        ("modern_tcn_transformer", 3),
        ("transformer_only", 2),
        ("transformer_only", 3),
        ("transformer_only", 4),
    }
    if temporal_grid != expected_grid:
        raise AssertionError(f"Unexpected temporal grid {temporal_grid}.")
    for spec in specs:
        _validate_config(spec.config)
        if spec.config["data"]["input_token_stream"] != "s1":
            raise AssertionError("The sweep is not coarse-only input.")
        if spec.config["data"]["target_token_stream"] != "s1":
            raise AssertionError("The sweep is not coarse-only output.")
        if spec.config["model"]["future_predictor"]["type"] != "structured_parallel":
            raise AssertionError("Structured-parallel predictor was lost.")
        if spec.config["training"]["selection_split"] != "test":
            raise AssertionError("The curiosity sweep is not test-selected.")

    scheduled = make_token_round2_specs(
        prior_type="correlation",
        graph_heads={2: (2, 1), 3: (2, 2, 1), 4: (2, 2, 2, 1)},
        context_length=8,
        prediction_length=4,
        evaluation_horizons=(1, 2, 4),
        transformer_d_model=8,
        transformer_num_heads=2,
        future_predictor_num_heads=2,
        modern_tcn_graph_dim_per_head=4,
        transformer_graph_dim_per_head=8,
    )
    expected_schedules = {2: (2, 1), 3: (2, 2, 1), 4: (2, 2, 2, 1)}
    for spec in scheduled:
        if spec.graph_heads_per_block != expected_schedules[spec.num_st_blocks]:
            raise AssertionError("Configurable graph-head schedule was lost.")

    dynamic_spec = next(
        spec
        for spec in specs
        if spec.temporal_family == "modern_tcn_transformer"
        and spec.num_transformer_blocks == 1
        and spec.graph_family == "dynamic_only"
    )
    prior_spec = next(
        spec
        for spec in specs
        if spec.temporal_family == "modern_tcn_transformer"
        and spec.num_transformer_blocks == 1
        and spec.graph_family == "prior_state"
    )
    _assert_model_contract(_build_small_model(dynamic_spec))
    _assert_model_contract(_build_small_model(prior_spec))

    # Selection is the unweighted mean Top-1 accuracy over every future step.
    logits = torch.full((1, 4, 1, 4), -5.0)
    target = torch.tensor([[[0], [1], [2], [3]]])
    for step in range(4):
        logits[0, step, 0, step] = 5.0
    sums = _token_batch_sums(logits, target, top_k_values=(1, 3))
    if float(sums["top1_correct_by_step"].sum().item()) != 4.0:
        raise AssertionError("Top-1 accumulation is incorrect.")

    evaluation = {
        "cross_entropy_by_step": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "top1_accuracy_by_step": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "top3_accuracy_by_step": torch.tensor([0.2, 0.3, 0.4, 0.5]),
        "top5_accuracy_by_step": torch.tensor([0.3, 0.4, 0.5, 0.6]),
        "top10_accuracy_by_step": torch.tensor([0.4, 0.5, 0.6, 0.7]),
    }
    detailed = _token_metric_table(evaluation, reported_horizons=(1, 2, 4))
    long = _token_metric_long_table(detailed)
    if set(long.columns) != {"metric", "horizon", "channel", "value"}:
        raise AssertionError("Graph-Hub token metric schema is invalid.")
    if len(long) != 4 * 5:
        raise AssertionError("Long token metric table has the wrong row count.")

    parameter = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "lr": 1.0, "base_lr": 1.0, "name": "backbone"}]
    )
    _set_schedule_for_epoch(
        optimizer, epoch=15, decay_start_epoch=15, decay_factor=0.9
    )
    if optimizer.param_groups[0]["lr"] != 1.0:
        raise AssertionError("Epoch 15 should retain the full learning rate.")
    _set_schedule_for_epoch(
        optimizer, epoch=16, decay_start_epoch=15, decay_factor=0.9
    )
    if abs(float(optimizer.param_groups[0]["lr"]) - 0.9) > 1.0e-12:
        raise AssertionError("Epoch 16 should be the first decayed epoch.")

    cache = _cache()
    dataset = CachedTokenGraphDataset(cache, validate=False)
    model = _build_small_model(prior_spec)
    loader = _build_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        seed=42,
        pin_memory=False,
    )
    exported = _export_selected_checkpoint(
        model=model,
        loader=loader,
        dataset=dataset,
        split_name="train",
        device=torch.device("cpu"),
        use_amp=False,
        checkpoint_epoch=1,
    )
    if tuple(exported["prediction_result"]["y_pred"].shape) != (4, 3, 4, 1):
        raise AssertionError("Public token prediction must expose reported horizons.")
    if tuple(exported["dense_token_prediction_result"]["y_pred"].shape) != (4, 4, 4, 1):
        raise AssertionError("Dense token prediction must retain all future steps.")

    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / prior_spec.run_name
        run_dir.mkdir(parents=True)
        resolved = prior_spec.config
        resolved["models"]["dynamic_graph"]["num_nodes"] = dataset.num_assets
        atomic_json_save(resolved, run_dir / "resolved_config.json")
        atomic_json_save(
            {
                "status": "completed",
                "model_family": "modern_tcn_graph_round2_token",
                "asset_cols": list(dataset.asset_cols),
                "best_epoch": 1,
                "epochs_completed": 1,
                "graph_family": "prior_state",
                "graph_heads_per_layer": list(prior_spec.graph_heads_per_block),
                "trainable_parameter_count": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "project_git_commit": "test",
            },
            run_dir / "run_metadata.json",
        )
        _save_export(run_dir, split_name="train", values=exported)

        # Save the same selected checkpoint outputs as validation so the
        # sampled-path loader can be tested without another model pass.
        validation_loader = _build_loader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            seed=43,
            pin_memory=False,
        )
        validation_export = _export_selected_checkpoint(
            model=model,
            loader=validation_loader,
            dataset=dataset,
            split_name="validation",
            device=torch.device("cpu"),
            use_amp=False,
            checkpoint_epoch=1,
        )
        _save_export(run_dir, split_name="validation", values=validation_export)

        audit = make_model_artifact_audit(
            {"token-round2": run_dir}, split="train"
        )
        if not bool(audit.iloc[0]["Ready"]):
            raise AssertionError(f"Graph Hub audit failed: {audit.iloc[0]['Issue']}")
        report = analyse_graph(
            run_dir,
            split="train",
            component="selected",
            layer=-1,
            head=0,
            day=None,
            window=None,
            cluster=False,
        )
        if report.plotted_adjacency.shape != (4, 4):
            raise AssertionError("Graph Hub returned the wrong adjacency shape.")

        _decode_sampled_split(
            model=model,
            dataset=dataset,
            split_name="validation",
            device=torch.device("cpu"),
            use_amp=False,
            tokenizer=_FakeCoarseDecoder(),
            sample_count=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            batch_size=2,
            num_workers=0,
            decode_series_batch_size=8,
            train_split=_raw_train_split(),
            checkpoint_epoch=1,
            run_dir=run_dir,
        )
        sampled_bundle = load_model_sampled_path_bundle(
            run_dir,
            split="validation",
            policy="temperature_1",
        )
        if tuple(sampled_bundle.sampled_close_paths.shape) != (2, 4, 4, 4, 1):
            raise AssertionError("Saved sampled Close paths have the wrong shape.")
        tokens_payload = torch.load(
            run_dir
            / "analysis"
            / "validation"
            / "temperature_1"
            / "tokens.pt",
            map_location="cpu",
            weights_only=False,
        )["token_artifacts"]
        if tuple(tokens_payload["sampled_s1_evaluation"].shape) != (2, 4, 3, 4):
            raise AssertionError("Sampled evaluation tokens have the wrong shape.")

    print("ModernTCN token-space Round-2 contracts passed.")


if __name__ == "__main__":
    main()

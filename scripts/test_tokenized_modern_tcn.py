from __future__ import annotations

"""CPU contracts for the final coarse-token ModernTCN graph experiment."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import yaml
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.dynamic_graph.contracts import (
    CloseScaleFeatureConfig,
    DynamicGraphModelConfig,
    ForecastHeadConfig,
    FuturePredictorConfig,
    GraphConfig,
    SpatialConfig,
    TemporalConfig,
)
from src.models.dynamic_graph.modern_tcn_token import (
    KRONOS_CODEBOOK_DIM,
    token_ids_to_bsq_codes,
)
from src.models.dynamic_graph.model import DynamicGraphTokenForecaster
import src.training.run_dynamic_graph as token_runner
from src.training.run_dynamic_graph import (
    _raw_close_scale_features,
    average_decoded_paths,
    fit_close_scale_feature_standardisation,
    generate_validation_artifacts,
)


class _FakeModernTCNBackbone(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.num_variables = int(config.enc_in)
        self.d_model = int(config.dims[0])
        self.patch_size = int(config.patch_size)
        self.patch_stride = int(config.patch_stride)
        self.stem = nn.Conv1d(
            1,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
        )
        self.variable_gain = nn.Parameter(
            torch.ones(self.num_variables, self.d_model)
        )

    def forward_feature(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Fake ModernTCN expects [B, M, T].")
        batch_size, variables, _ = x.shape
        padding = self.patch_size - self.patch_stride
        if padding > 0:
            x = torch.cat(
                [x, x[..., -1:].expand(-1, -1, padding)],
                dim=-1,
            )
        patches = self.stem(
            x.reshape(batch_size * variables, 1, x.shape[-1])
        )
        patches = patches.reshape(
            batch_size,
            variables,
            self.d_model,
            patches.shape[-1],
        )
        return patches * self.variable_gain[None, :, :, None]


class _FakeOfficialModernTCN(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.model = _FakeModernTCNBackbone(config)


def _config() -> DynamicGraphModelConfig:
    return DynamicGraphModelConfig(
        num_nodes=4,
        context_length=8,
        d_model=8,
        num_st_blocks=1,
        use_node_embedding=False,
        token_input_representation="bsq_bits",
        temporal=TemporalConfig(
            type="modern_tcn",
            num_layers=1,
            num_heads=2,
            feedforward_multiplier=2,
            dropout=0.0,
            modern_tcn_patch_size=4,
            modern_tcn_patch_stride=2,
            modern_tcn_ffn_ratio=1,
            modern_tcn_num_blocks=1,
            modern_tcn_large_kernel=3,
            modern_tcn_small_kernel=3,
            modern_tcn_dropout=0.0,
        ),
        graph=GraphConfig(
            type="dynamic",
            num_heads=1,
            hidden_dim=4,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=3,
            base_graph_type="free_static",
            gate_type="none",
            initial_alpha=0.5,
        ),
        spatial=SpatialConfig(
            num_layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
            gate_type="learned_scalar",
            initial_beta=0.5,
        ),
        heads=ForecastHeadConfig(
            prediction_length=6,
            evaluation_horizons=(1, 3, 6),
            s1_vocabulary_size=1024,
            s2_vocabulary_size=1024,
            s2_loss_weight=0.0,
            future_token_mode="coarse_only",
            s2_conditioning="true_s1",
        ),
        future_predictor=FuturePredictorConfig(
            type="structured_parallel",
            num_layers=1,
            num_heads=2,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
    )




def _embedded_config(
    *,
    temporal_type: str = "modern_tcn",
    graph_type: str = "dynamic",
    num_st_blocks: int = 1,
    scale_features: bool = False,
) -> DynamicGraphModelConfig:
    temporal = TemporalConfig(
        type=temporal_type,
        num_layers=1,
        num_heads=2,
        feedforward_multiplier=2,
        dropout=0.0,
        modern_tcn_patch_size=4,
        modern_tcn_patch_stride=2,
        modern_tcn_ffn_ratio=1,
        modern_tcn_num_blocks=1,
        modern_tcn_large_kernel=3,
        modern_tcn_small_kernel=3,
        modern_tcn_dropout=0.0,
    )
    return DynamicGraphModelConfig(
        num_nodes=4,
        context_length=8,
        d_model=8,
        num_st_blocks=num_st_blocks,
        use_node_embedding=True,
        token_input_representation="hierarchical_embedding",
        temporal=temporal,
        graph=GraphConfig(
            type=graph_type,
            num_heads=1,
            hidden_dim=4,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=3,
            base_graph_type="free_static",
            gate_type="none",
            initial_alpha=0.5,
        ),
        spatial=SpatialConfig(
            num_layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
            gate_type="learned_scalar",
            initial_beta=0.5,
        ),
        close_scale_features=CloseScaleFeatureConfig(
            enabled=scale_features,
            eps=1.0e-6,
        ),
        heads=ForecastHeadConfig(
            prediction_length=6,
            evaluation_horizons=(1, 3, 6),
            s1_vocabulary_size=1024,
            s2_vocabulary_size=1024,
            s2_loss_weight=0.0,
            future_token_mode="coarse_only",
            s2_conditioning="true_s1",
        ),
        future_predictor=FuturePredictorConfig(
            type="structured_parallel",
            num_layers=1,
            num_heads=2,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
    )


def test_hierarchical_embedding_modern_tcn_contract() -> None:
    torch.manual_seed(21)
    config = _embedded_config()
    model = DynamicGraphTokenForecaster(
        config,
        modern_tcn_model_cls=_FakeOfficialModernTCN,
    )
    context = torch.randint(
        0,
        1024,
        (2, config.context_length, config.num_nodes, 2),
    )
    target_s1 = torch.randint(
        0,
        1024,
        (2, config.prediction_length, config.num_nodes),
    )
    output = model(context, target_s1=target_s1)
    output.validate(config, batch_size=2)
    assert tuple(output.temporal_hidden.shape) == (2, 4, 4, 8)
    assert model.token_embedding is not None
    assert model.modern_tcn_encoder is not None
    assert model.modern_tcn_encoder.num_token_variables == config.d_model

    loss = F.cross_entropy(
        output.s1_logits.reshape(-1, 1024),
        target_s1.reshape(-1),
    )
    loss.backward()
    for name, parameter in {
        "s1 embedding": model.token_embedding.s1_embedding.weight,
        "s2 embedding": model.token_embedding.s2_embedding.weight,
        "node embedding": model.token_embedding.node_embedding.weight,
        "position embedding": model.token_embedding.position_embedding.weight,
        "ModernTCN stem": model.modern_tcn_encoder.official_model.model.stem.weight,
    }.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"{name} received no finite gradient.")
        if parameter.grad.abs().sum().item() == 0.0:
            raise AssertionError(f"{name} received only zero gradients.")


def test_three_interlaced_transformer_blocks() -> None:
    torch.manual_seed(22)
    config = _embedded_config(
        temporal_type="transformer",
        graph_type="dynamic",
        num_st_blocks=3,
    )
    model = DynamicGraphTokenForecaster(config)
    context = torch.randint(
        0, 1024, (2, config.context_length, config.num_nodes, 2)
    )
    target_s1 = torch.randint(
        0, 1024, (2, config.prediction_length, config.num_nodes)
    )
    output = model(context, target_s1=target_s1)
    output.validate(config, batch_size=2)
    assert len(model.temporal_blocks) == 3
    assert len(model.graph_learners) == 3
    assert len(model.spatial_blocks) == 3
    assert len(model.spatial_gates) == 3
    assert len(output.graph.per_layer) == 3
    assert all(graph is not None for graph in output.graph.per_layer)


class _ScaleDataset:
    data_mode = "real"

    def __init__(self) -> None:
        means = torch.ones(3, 4, 6)
        stds = torch.ones(3, 4, 6)
        means[..., 3] = torch.tensor(
            [100.0, 200.0, 400.0, 800.0]
        ).view(1, 4).expand(3, -1)
        stds[..., 3] = means[..., 3] * torch.tensor(
            [0.001, 0.002, 0.004]
        ).view(3, 1)
        self.cache = {"context_mean": means, "context_std": stds}


def test_close_scale_feature_contract() -> None:
    dataset = _ScaleDataset()
    center, scale, metadata = fit_close_scale_feature_standardisation(
        dataset, eps=1.0e-6
    )
    raw = _raw_close_scale_features(
        dataset.cache["context_mean"],
        dataset.cache["context_std"],
        eps=1.0e-6,
    )
    assert tuple(raw.shape) == (3, 4, 1)
    expected_raw = torch.log(
        dataset.cache["context_std"][..., 3].square().add(1.0e-6)
    ).unsqueeze(-1)
    torch.testing.assert_close(raw, expected_raw)
    assert tuple(center.shape) == (1,)
    assert tuple(scale.shape) == (1,)
    assert metadata["feature_contract"] == "close_log_variance_v2"
    assert metadata["feature_names"] == ["log_context_close_variance"]
    assert torch.isfinite(center).all()
    assert torch.isfinite(scale).all() and torch.all(scale > 0)
    assert metadata["fit_split"] == "training windows only"

    torch.manual_seed(23)
    config = _embedded_config(
        temporal_type="transformer",
        graph_type="free_static",
        num_st_blocks=1,
        scale_features=True,
    )
    model = DynamicGraphTokenForecaster(
        config,
        close_scale_feature_center=center,
        close_scale_feature_scale=scale,
    )
    context = torch.randint(
        0, 1024, (2, config.context_length, config.num_nodes, 2)
    )
    target_s1 = torch.randint(
        0, 1024, (2, config.prediction_length, config.num_nodes)
    )
    means = dataset.cache["context_mean"][:2]
    stds = dataset.cache["context_std"][:2]
    output = model(
        context,
        target_s1=target_s1,
        context_mean=means,
        context_std=stds,
    )
    loss = F.cross_entropy(
        output.s1_logits.reshape(-1, 1024),
        target_s1.reshape(-1),
    )
    loss.backward()
    if model.close_scale_embedding is None:
        raise AssertionError("Scale embedding was not constructed.")
    gradient = model.close_scale_embedding.projection.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("Scale projection received no finite gradient.")
    if gradient.abs().sum().item() == 0.0:
        raise AssertionError("Scale projection received only zero gradients.")

    with torch.inference_mode():
        shifted_mean = means.clone()
        shifted_mean[..., 3] *= 2.0
        unchanged = model(
            context,
            target_s1=target_s1,
            context_mean=shifted_mean,
            context_std=stds,
        )
        shifted_std = stds.clone()
        shifted_std[..., 3] *= 2.0
        changed = model(
            context,
            target_s1=target_s1,
            context_mean=means,
            context_std=shifted_std,
        )
    torch.testing.assert_close(
        output.s1_logits.detach(),
        unchanged.s1_logits,
        atol=0.0,
        rtol=0.0,
    )
    if torch.equal(output.s1_logits.detach(), changed.s1_logits):
        raise AssertionError("Changing Close variance changed no logits.")

def test_runner_json_loader_contract() -> None:
    """Inference-only modes can reload an existing run metadata file."""

    with TemporaryDirectory() as directory:
        metadata_path = (
            Path(directory)
            / "run_metadata.json"
        )
        expected = {
            "status": "completed",
            "run_signature": "test-signature",
        }

        token_runner.atomic_json_save(
            expected,
            metadata_path,
        )

        observed = token_runner.load_json(
            metadata_path
        )

        assert observed == expected


def test_evaluation_signature_allows_code_only_commit_change() -> None:
    """Legacy checkpoints remain evaluable after a code-only bug fix."""

    resolved_config = {
        "training": {"learning_rate": 1.0e-4},
        "models": {"dynamic_graph": {"d_model": 32}},
        "temperature_sweep": {"temperatures": [0.6]},
    }
    payload = token_runner._evaluation_compatibility_payload(
        resolved_config=resolved_config,
        train_cache="/cache/train.pt",
        validation_cache="/cache/val.pt",
        data_mode="real",
        asset_cols=("A", "B"),
        train_windows=3,
        validation_windows=2,
        fixed_graph_resource_hash=None,
        s1_id_space="kronos_original",
        s1_vocabulary_size=1024,
        s1_remapping_resource_hash=None,
    )
    expected = token_runner._config_signature(payload)

    with TemporaryDirectory() as directory:
        run_dir = Path(directory)
        token_runner.atomic_json_save(
            resolved_config,
            run_dir / "resolved_config.json",
        )
        metadata = {
            "run_signature": "legacy-signature-containing-old-commit",
            "project_git_commit": "old-commit",
            "train_cache_path": "/cache/train.pt",
            "validation_cache_path": "/cache/val.pt",
            "data_mode": "real",
            "asset_cols": ["A", "B"],
            "train_windows": 3,
            "validation_windows": 2,
            "fixed_graph_resource": None,
            "s1_token_space": {
                "id_space": "kronos_original",
                "vocabulary_size": 1024,
                "resource_hash": None,
            },
        }
        observed = token_runner._saved_evaluation_compatibility_signature(
            run_dir=run_dir,
            existing_metadata=metadata,
        )
        assert observed == expected

        changed_temperature_config = {
            **resolved_config,
            "temperature_sweep": {"temperatures": [0.3, 0.8]},
        }
        explicitly_disabled_scale = {
            **resolved_config,
            "models": {
                "dynamic_graph": {
                    "d_model": 32,
                    "close_scale_features": {
                        "enabled": False,
                        "eps": 1.0e-6,
                    },
                }
            },
        }
        disabled_scale_payload = token_runner._evaluation_compatibility_payload(
            resolved_config=explicitly_disabled_scale,
            train_cache="/cache/train.pt",
            validation_cache="/cache/val.pt",
            data_mode="real",
            asset_cols=("A", "B"),
            train_windows=3,
            validation_windows=2,
            fixed_graph_resource_hash=None,
            s1_id_space="kronos_original",
            s1_vocabulary_size=1024,
            s1_remapping_resource_hash=None,
        )
        assert token_runner._config_signature(disabled_scale_payload) == expected

        changed_payload = token_runner._evaluation_compatibility_payload(
            resolved_config=changed_temperature_config,
            train_cache="/cache/train.pt",
            validation_cache="/cache/val.pt",
            data_mode="real",
            asset_cols=("A", "B"),
            train_windows=3,
            validation_windows=2,
            fixed_graph_resource_hash=None,
            s1_id_space="kronos_original",
            s1_vocabulary_size=1024,
            s1_remapping_resource_hash=None,
        )
        assert token_runner._config_signature(changed_payload) == expected

        incompatible_payload = token_runner._evaluation_compatibility_payload(
            resolved_config={
                **resolved_config,
                "models": {"dynamic_graph": {"d_model": 64}},
            },
            train_cache="/cache/train.pt",
            validation_cache="/cache/val.pt",
            data_mode="real",
            asset_cols=("A", "B"),
            train_windows=3,
            validation_windows=2,
            fixed_graph_resource_hash=None,
            s1_id_space="kronos_original",
            s1_vocabulary_size=1024,
            s1_remapping_resource_hash=None,
        )
        assert token_runner._config_signature(incompatible_payload) != expected


def test_temperature_policy_resume_record_contract() -> None:
    """A completed policy is reused only for an identical request."""

    with TemporaryDirectory() as directory:
        output_dir = Path(directory)
        label = "temperature_0p6"
        policy_dir = output_dir / label
        policy_dir.mkdir(parents=True)

        for filename in (
            "validation_predictions.pt",
            "validation_graphs.pt",
            "validation_tokens.pt",
            "validation_metric_table.csv",
            "validation_diagnostics.json",
            "validation_sampled_price_paths.pt",
        ):
            (policy_dir / filename).write_bytes(b"test")

        expected_result = {
            "Policy": label,
            "Temperature": 0.6,
            "Sample count": 10,
            "Mean Log MAE": 0.001,
        }
        token_runner._save_temperature_result_record(
            output_dir=output_dir,
            label=label,
            temperature=0.6,
            sample_count=10,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            checkpoint_epoch=7,
            result=expected_result,
        )

        observed = token_runner._load_reusable_temperature_result(
            output_dir=output_dir,
            label=label,
            temperature=0.6,
            sample_count=10,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            checkpoint_epoch=7,
        )
        assert observed == expected_result

        changed_temperature = token_runner._load_reusable_temperature_result(
            output_dir=output_dir,
            label=label,
            temperature=0.8,
            sample_count=10,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            checkpoint_epoch=7,
        )
        assert changed_temperature is None

        (policy_dir / "validation_sampled_price_paths.pt").unlink()
        incomplete = token_runner._load_reusable_temperature_result(
            output_dir=output_dir,
            label=label,
            temperature=0.6,
            sample_count=10,
            top_k=0,
            top_p=0.9,
            sampling_seed=42,
            checkpoint_epoch=7,
        )
        assert incomplete is None


def test_final_preset_uses_validation_ce_selection() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "dynamic_graph.yaml"
    payload = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )
    preset = payload["presets"][
        "modern_tcn_dynamic_coarse_mc10"
    ]
    training = preset["training"]
    decoding = preset["decoding"]
    temperature_sweep = payload["temperature_sweep"]

    assert training["early_stopping_metric"] == (
        "validation_token_loss"
    )
    assert decoding["token_selection"] == "argmax"
    assert int(decoding["sample_count"]) == 1
    assert int(temperature_sweep["sample_count"]) == 10

    embedded = payload["presets"]["hierarchical_embedding_coarse_ce"]
    embedded_model = embedded["models"]["dynamic_graph"]
    assert embedded_model["token_input_representation"] == (
        "hierarchical_embedding"
    )
    assert embedded_model["future_predictor"]["num_layers"] == 1
    assert embedded["training"]["early_stopping_metric"] == (
        "validation_token_loss"
    )

    runner_source = (
        repo_root
        / "src"
        / "training"
        / "run_dynamic_graph.py"
    ).read_text(encoding="utf-8")
    assert (
        "validation_token_loss early stopping currently requires"
        not in runner_source
    )
    assert (
        "selected_checkpoint_artifacts_regenerated"
        in runner_source
    )

def test_post_bsq_code_contract() -> None:
    token_ids = torch.tensor(
        [[[[1, 2], [1023, 0]]]],
        dtype=torch.long,
    )
    code = token_ids_to_bsq_codes(token_ids)
    assert tuple(code.shape) == (1, 1, 2, 20)
    scale = 1.0 / (KRONOS_CODEBOOK_DIM ** 0.5)
    expected_first = torch.tensor(
        [1.0] + [-1.0] * 9 + [-1.0, 1.0] + [-1.0] * 8
    ) * scale
    torch.testing.assert_close(code[0, 0, 0], expected_first)
    torch.testing.assert_close(
        code.square().sum(dim=-1),
        torch.ones(1, 1, 2),
    )


def test_model_shapes_gradients_and_sampling() -> None:
    torch.manual_seed(42)
    config = _config()
    model = DynamicGraphTokenForecaster(
        config,
        modern_tcn_model_cls=_FakeOfficialModernTCN,
    )
    context = torch.randint(
        0,
        1024,
        (2, config.context_length, config.num_nodes, 2),
    )
    target_s1 = torch.randint(
        0,
        1024,
        (2, config.prediction_length, config.num_nodes),
    )
    target_s2 = torch.randint(
        0,
        1024,
        (2, config.prediction_length, config.num_nodes),
    )
    output = model(context, target_s1=target_s1, target_s2=target_s2)
    output.validate(config, batch_size=2)
    assert tuple(output.temporal_hidden.shape) == (2, 4, 4, 8)
    assert tuple(output.s1_logits.shape) == (2, 6, 4, 1024)
    assert output.s2_logits is None
    assert output.graph.selected is not None
    assert tuple(output.graph.selected.shape) == (2, 1, 4, 4)
    assert output.spatial_beta is not None

    loss = F.cross_entropy(
        output.s1_logits.reshape(-1, 1024),
        target_s1.reshape(-1),
    )
    loss.backward()
    required_parameters = {
        "ModernTCN stem": model.modern_tcn_encoder.official_model.model.stem.weight,
        "token-variable pool": model.modern_tcn_encoder.variable_pool.weight,
        "dynamic query": model.graph_learners[0].q_proj.weight,
        "future s1 head": model.future_predictor.token_heads.s1_classifier.weight,
        "spatial beta": model.spatial_gates[0].raw_beta,
    }
    for name, parameter in required_parameters.items():
        if parameter is None or parameter.grad is None:
            raise AssertionError(f"{name} received no gradient.")
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"{name} received a non-finite gradient.")
        if parameter.grad.abs().sum().item() == 0.0:
            raise AssertionError(f"{name} received only zero gradients.")

    torch.manual_seed(123)
    first = model.generate_samples(
        context,
        sample_count=10,
        token_selection="sample",
        temperature=0.6,
        top_k=0,
        top_p=0.9,
    )
    torch.manual_seed(123)
    second = model.generate_samples(
        context,
        sample_count=10,
        token_selection="sample",
        temperature=0.6,
        top_k=0,
        top_p=0.9,
    )
    assert tuple(first.token_ids.shape) == (10, 2, 6, 4, 2)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.count_nonzero(first.token_ids[..., 1]).item() == 0


def test_decoded_continuous_average() -> None:
    paths = torch.tensor(
        [
            [[[[10.0, 11.0, 9.0, 10.5, 100.0]]]],
            [[[[12.0, 13.0, 11.0, 12.5, 120.0]]]],
        ]
    )
    averaged = average_decoded_paths(paths)
    expected = torch.tensor([[[[11.0, 12.0, 10.0, 11.5, 110.0]]]])
    torch.testing.assert_close(averaged, expected)



class _FakeTokenDataset:
    data_mode = "real"
    asset_cols = ("A", "B", "C", "D")

    @staticmethod
    def s1_to_kronos_ids(values: Tensor) -> Tensor:
        return torch.as_tensor(values).long()


class _FakeCoarseDecoder:
    def decode_coarse_token_path(
        self,
        context_tokens: Tensor,
        future_s1: Tensor,
        *,
        mean: Tensor,
        std: Tensor,
        series_batch_size: int,
        return_full_path: bool,
    ) -> Tensor:
        del context_tokens, mean, std, series_batch_size, return_full_path
        close = future_s1.float() + 100.0
        return torch.stack(
            (
                close,
                close + 1.0,
                close - 1.0,
                close,
                torch.ones_like(close),
            ),
            dim=-1,
        )


class _FakeEvaluator:
    available_metrics = ("cumulative_log_change_mae",)

    def __init__(self, *, prediction_result, train_split) -> None:
        del train_split
        self.prediction_result = prediction_result
        self.horizons = tuple(prediction_result["horizons"])
        self.channels = tuple(prediction_result["channels"])

    def evaluate(self, *, metrics, reduce_dims, bootstrap):
        del metrics, reduce_dims, bootstrap
        predicted = self.prediction_result["y_pred"].float().clamp_min(1.0e-6)
        target = self.prediction_result["y_true"].float().clamp_min(1.0e-6)
        last = self.prediction_result["last_context_target"].float().clamp_min(1.0e-6)
        predicted_change = predicted.log() - last.unsqueeze(1).log()
        true_change = target.log() - last.unsqueeze(1).log()
        value = (predicted_change - true_change).abs().mean(dim=(0, 2, 3))
        return {"cumulative_log_change_mae": value}


def _fake_metric_table(*, metric_results, horizons, channels):
    rows = []
    values = torch.as_tensor(
        metric_results["cumulative_log_change_mae"]
    ).reshape(-1)
    for horizon, value in zip(horizons, values, strict=True):
        rows.append(
            {
                "metric": "cumulative_log_change_mae",
                "horizon": int(horizon),
                "channel": str(channels[0]),
                "value": float(value.item()),
            }
        )
    import pandas as pd
    return pd.DataFrame(rows)


def test_ten_path_decode_then_average_contract() -> None:
    torch.manual_seed(7)
    config = _config()
    model = DynamicGraphTokenForecaster(
        config,
        modern_tcn_model_cls=_FakeOfficialModernTCN,
    )
    batch_size = 2
    context = torch.randint(
        0,
        1024,
        (batch_size, config.context_length, config.num_nodes, 2),
    )
    target_s1 = torch.randint(
        0,
        1024,
        (batch_size, config.prediction_length, config.num_nodes),
    )
    target_s2 = torch.randint(
        0,
        1024,
        (batch_size, config.prediction_length, config.num_nodes),
    )
    evaluation_true = torch.full(
        (batch_size, 3, config.num_nodes, 5),
        200.0,
    )
    evaluation_true[..., 4] = 1.0
    last_context = torch.full(
        (batch_size, config.num_nodes, 5),
        190.0,
    )
    last_context[..., 4] = 1.0
    batch = {
        "context_tokens": context,
        "target_s1": target_s1,
        "target_s2": target_s2,
        "context_mean": torch.zeros(batch_size, config.num_nodes, 6),
        "context_std": torch.ones(batch_size, config.num_nodes, 6),
        "evaluation_true": evaluation_true,
        "last_context_target": last_context,
        "sample_idx": torch.arange(batch_size),
        "origin_idx": torch.full((batch_size,), 59),
        "target_indices": torch.tensor(
            [[60, 61, 62, 63, 64, 65]] * batch_size
        ),
        "date": ["2024-09-03", "2024-09-04"],
    }

    original_evaluator = token_runner.ForecastEvaluator
    original_table = token_runner.make_evaluation_table
    token_runner.ForecastEvaluator = _FakeEvaluator
    token_runner.make_evaluation_table = _fake_metric_table
    try:
        torch.manual_seed(1234)
        bundle = generate_validation_artifacts(
            model=model,
            loader=[batch],
            dataset=_FakeTokenDataset(),
            device=torch.device("cpu"),
            use_amp=False,
            decoding_config={
                "token_selection": "sample",
                "temperature": 0.6,
                "top_k": 0,
                "top_p": 0.9,
                "sample_count": 10,
            },
            tokenizer=_FakeCoarseDecoder(),
            raw_train_split={"unused": True},
            decode_series_batch_size=64,
            early_stopping_horizons=(1, 3, 6),
        )
    finally:
        token_runner.ForecastEvaluator = original_evaluator
        token_runner.make_evaluation_table = original_table

    sampled = bundle.token_artifacts["sampled_s1_evaluation"].float()
    expected_close = sampled.mean(dim=0).unsqueeze(-1) + 100.0
    observed_close = bundle.prediction_result["y_pred"]
    torch.testing.assert_close(observed_close, expected_close)

    path_artifacts = bundle.sampled_price_path_artifacts
    if path_artifacts is None:
        raise AssertionError("Sampled price-path artifacts were not retained.")
    sampled_close_paths = path_artifacts["sampled_close_paths"]
    sampled_close_evaluation = path_artifacts[
        "sampled_close_paths_at_evaluation_horizons"
    ]
    assert tuple(sampled_close_paths.shape) == (10, 2, 6, 4, 1)
    assert tuple(sampled_close_evaluation.shape) == (10, 2, 3, 4, 1)
    torch.testing.assert_close(
        sampled_close_evaluation,
        sampled.unsqueeze(-1) + 100.0,
    )
    torch.testing.assert_close(
        path_artifacts["ensemble_mean_close_path"].index_select(
            dim=1,
            index=torch.tensor([0, 2, 5]),
        ),
        observed_close,
    )
    assert path_artifacts["sample_count"] == 10
    assert path_artifacts["channel"] == "close"
    assert path_artifacts["output_space"] == "raw"

    assert bundle.token_artifacts["sample_count"] == 10
    assert bundle.diagnostics["sample_count"] == 10
    assert tuple(bundle.graph_artifacts["spatial_beta"].shape) == (2, 1)


def main() -> None:
    test_runner_json_loader_contract()
    test_evaluation_signature_allows_code_only_commit_change()
    test_temperature_policy_resume_record_contract()
    test_final_preset_uses_validation_ce_selection()
    test_post_bsq_code_contract()
    test_model_shapes_gradients_and_sampling()
    test_hierarchical_embedding_modern_tcn_contract()
    test_three_interlaced_transformer_blocks()
    test_close_scale_feature_contract()
    test_decoded_continuous_average()
    test_ten_path_decode_then_average_contract()
    print("Tokenized ModernTCN graph contract tests passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Specifications for the final token and BaseDyGraph-V2 comparison notebook."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.models.final_token_v2_models import DIMITRI_NOTEBOOK_DEFAULTS
from src.training.dense_parallel_graph_specs import inverse_reference_weights


REFERENCE_MAE: tuple[float, ...] = (
    0.00036854,
    0.00078591,
    0.00132230,
    0.00183974,
    0.00255599,
)


@dataclass(frozen=True)
class FinalComparisonSpec:
    run_name: str
    label: str
    model_kind: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "model_kind": self.model_kind,
            "config": self.config,
        }


def _hash(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _spec(*, model_kind: str, label: str, prefix: str, config: dict[str, Any]) -> FinalComparisonSpec:
    signature = _hash(config)
    return FinalComparisonSpec(
        run_name=f"{prefix}_{signature}",
        label=label,
        model_kind=model_kind,
        config=config,
    )


def _token_analysis_mirror(
    *,
    num_nodes: int,
    d_model: int,
    temporal: Mapping[str, Any],
    graph: Mapping[str, Any],
    spatial: Mapping[str, Any],
    prediction_length: int,
    evaluation_horizons: Sequence[int],
    future_predictor: Mapping[str, Any],
    token_input_representation: str = "coarse_s1_embedding",
) -> dict[str, Any]:
    """Return the established Graph-Hub token schema.

    The training constructors use the more explicit top-level ``model``
    configuration.  Graph Hub historically reads ``models.dynamic_graph``.
    Saving both schemas prevents analysis code from guessing a model class and
    keeps the on-disk contract self-describing.
    """

    graph_values = dict(graph)
    graph_values.setdefault(
        "num_heads",
        int(tuple(graph_values["num_heads_per_block"])[-1]),
    )
    graph_values.setdefault(
        "hidden_dim",
        int(tuple(graph_values["hidden_dims_per_block"])[-1]),
    )
    graph_values.setdefault(
        "activation",
        str(tuple(graph_values["activations_per_block"])[-1]),
    )
    return {
        "dynamic_graph": {
            "num_nodes": int(num_nodes),
            "d_model": int(d_model),
            "token_input_representation": str(token_input_representation),
            "temporal": dict(temporal),
            "graph": graph_values,
            "spatial": dict(spatial),
            "heads": {
                "future_token_mode": "coarse_only",
                "prediction_length": int(prediction_length),
                "evaluation_horizons": [
                    int(value) for value in evaluation_horizons
                ],
                "s1_vocabulary_size": 1024,
            },
            "future_predictor": dict(future_predictor),
        }
    }


def _v2_graph_schema(defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the resolved four-block V2 graph schedule to Graph Hub."""

    blocks = int(defaults["num_st_blocks"])
    early_heads = tuple(int(v) for v in defaults["num_edge_heads_per_block"])
    early_widths = tuple(int(v) for v in defaults["graph_hidden_dim_per_block"])
    early_activations = tuple(str(v) for v in defaults["graph_activation_per_block"])
    if blocks == 1:
        heads = (early_heads[-1],)
        widths = (early_widths[-1],)
        activations = (early_activations[-1],)
    else:
        heads = (*([early_heads[0]] * (blocks - 2)), early_heads[-2], early_heads[-1])
        widths = (*([early_widths[0]] * (blocks - 2)), early_widths[-2], early_widths[-1])
        activations = (*([early_activations[0]] * (blocks - 2)), early_activations[-2], early_activations[-1])
    return {
        "type": "dimitri_v2_dual_fusion",
        "graph_type": "dimitri_v2_dual_fusion",
        "num_heads": int(heads[-1]),
        "num_heads_per_block": list(heads),
        "hidden_dim": int(widths[-1]),
        "hidden_dims_per_block": list(widths),
        "activation": str(activations[-1]),
        "activations_per_block": list(activations),
        "add_self_loops": bool(defaults["add_self_loops"]),
        "initial_alpha": float(defaults["dynamic_residual_init"]),
    }


def make_final_token_v2_specs(
    *,
    context_length: int = 60,
    prediction_length: int = 60,
    evaluation_horizons: Sequence[int] = (1, 5, 15, 30, 60),
    vocabulary_size: int = 1024,
    seed: int = 42,
) -> tuple[FinalComparisonSpec, ...]:
    horizons = tuple(int(value) for value in evaluation_horizons)
    if horizons != tuple(sorted(set(horizons))) or not horizons:
        raise ValueError("evaluation_horizons must be unique and increasing.")
    if horizons[-1] > int(prediction_length):
        raise ValueError("evaluation_horizons exceed prediction_length.")

    token_data = {
        "context_length": int(context_length),
        "prediction_length": int(prediction_length),
        "evaluation_horizons": list(horizons),
        "input_token_stream": "s1",
        "target_token_stream": "s1",
        "s1_vocabulary_size": int(vocabulary_size),
        "stride": 15,
    }
    continuous_data = {
        "context_length": int(context_length),
        "horizons": list(horizons),
        "stride": 15,
        "input_channels": ["open", "high", "low", "close", "volume"],
        "target_channel": "close",
        "input_representation": "raw",
    }

    delayed_token_training = {
        "optimizer": "adam",
        "parameter_grouping": "split",
        "learning_rate": 2.5e-4,
        "graph_learning_rate": 5.0e-4,
        "weight_decay": 0.0,
        "scheduler": "modern_tcn_type3_delayed",
        "scheduler_decay_start_epoch": 15,
        "scheduler_decay_factor": 0.9,
        "max_epochs": 100,
        "patience": 10,
        "min_delta": 0.0,
        "batch_size": 2,
        "selection_batch_size": 2,
        "export_batch_size": 2,
        "num_workers": 0,
        "gradient_clip_norm": 1.0,
        "mixed_precision": True,
        "seed": int(seed),
        "selection_split": "test",
        "selection_metric": "mean_top1_accuracy_over_all_60_future_steps",
        "early_stopping_metric": "mean_top1_accuracy_over_all_60_future_steps",
        "selection_direction": "maximise",
        "loss": {
            "type": "coarse_s1_cross_entropy",
            "horizon_weighting": "uniform",
            "dense_origins": False,
        },
    }

    modern_config: dict[str, Any] = {
        "model_family": "final_modern_tcn_token_counterpart",
        "model_kind": "modern_tcn_token",
        "do_not_report": True,
        "test_set_contaminated": True,
        "data": dict(token_data),
        "model": {
            "graph_family": "prior_state",
            "temporal_stack": {
                "family": "modern_tcn_transformer",
                "num_transformer_blocks": 0,
                "modern_tcn": {
                    "d_model": 32,
                    "patch_size": 8,
                    "patch_stride": 4,
                    "ffn_ratio": 1,
                    "num_blocks": 1,
                    "large_kernel": 15,
                    "small_kernel": 5,
                    "dropout": 0.05,
                    "head_dropout": 0.0,
                },
                # This width is also the structured future-head width.  There
                # is no Transformer refinement block in this model.
                "transformer": {
                    "d_model": 32,
                    "num_layers": 1,
                    "num_heads": 4,
                    "feedforward_multiplier": 2,
                    "dropout": 0.0,
                    "relative_position_embedding": False,
                },
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "num_heads": 1,
                "num_heads_per_block": [1],
                "hidden_dim": 32,
                "hidden_dims_per_block": [32],
                "activation": "softmax",
                "activations_per_block": ["softmax"],
                "add_self_loops": False,
                "initial_alpha": 0.5,
            },
            "prior": {
                "type": "correlation",
                "scale": 4.0,
                "jitter": 0.02,
                "seed": int(seed),
                "threshold": None,
            },
            "spatial": {
                "num_layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
            },
            "future_predictor": {
                "type": "structured_parallel",
                "num_layers": 1,
                "num_heads": 4,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
            },
            "graph_regularisation": {
                "graph_entropy_reg": 0.0,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
        },
        "training": dict(delayed_token_training),
    }
    modern_config["models"] = _token_analysis_mirror(
        num_nodes=93,
        d_model=32,
        temporal={
            "type": "modern_tcn",
            "d_model": 32,
            "num_layers": 1,
            "num_heads": 4,
            "modern_tcn": dict(
                modern_config["model"]["temporal_stack"]["modern_tcn"]
            ),
        },
        graph=modern_config["model"]["graph"],
        spatial=modern_config["model"]["spatial"],
        prediction_length=prediction_length,
        evaluation_horizons=horizons,
        future_predictor=modern_config["model"]["future_predictor"],
    )

    dense_training = dict(delayed_token_training)
    dense_training.update(
        {
            "batch_size": 1,
            "selection_batch_size": 2,
            "export_batch_size": 2,
            "loss": {
                "type": "coarse_s1_cross_entropy",
                "horizon_weighting": "uniform",
                "dense_origins": True,
                "future_steps_per_origin": int(prediction_length),
                "origin_chunk_size": 1,
            },
        }
    )
    dense_transformer_config: dict[str, Any] = {
        "model_family": "final_dense_transformer_token_counterpart",
        "model_kind": "dense_transformer_token",
        "do_not_report": True,
        "test_set_contaminated": True,
        "data": dict(token_data),
        "model": {
            "num_nodes": 93,
            "num_st_blocks": 3,
            "temporal": {
                "type": "transformer",
                "d_model": 64,
                "num_layers": 1,
                "num_heads": 4,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "position_embedding": False,
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "num_heads_per_block": [1, 1, 1],
                "hidden_dims_per_block": [64, 64, 64],
                "activations_per_block": ["softmax", "softmax", "sparsemax"],
                "add_self_loops": False,
                "initial_alpha": 0.5,
            },
            "prior": {
                "type": "uniform",
                "static_logits": "zeros",
                "dynamic_logits": "zeros_at_initialisation",
            },
            "spatial": {
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "initial_beta": 0.5,
            },
            "future_predictor": {
                "type": "structured_parallel_per_causal_origin",
                "prediction_length": int(prediction_length),
                "num_layers": 1,
                "num_heads": 4,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
            },
            "graph_regularisation": {
                "graph_entropy_reg": 0.0,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
        },
        "training": dense_training,
    }
    dense_transformer_config["models"] = _token_analysis_mirror(
        num_nodes=93,
        d_model=64,
        temporal=dense_transformer_config["model"]["temporal"],
        graph=dense_transformer_config["model"]["graph"],
        spatial=dense_transformer_config["model"]["spatial"],
        prediction_length=prediction_length,
        evaluation_horizons=horizons,
        future_predictor=dense_transformer_config["model"]["future_predictor"],
    )

    v2_defaults = dict(DIMITRI_NOTEBOOK_DEFAULTS)
    v2_defaults["temporal_context_window"] = int(context_length)
    v2_token_training = {
        "optimizer": "adamw",
        "parameter_grouping": "shared",
        "learning_rate": 0.0012,
        "weight_decay": 0.0001,
        "scheduler": "cosine_annealing",
        "scheduler_t_max": 120,
        "max_epochs": 120,
        "patience": 15,
        "min_delta": 0.0,
        "batch_size": 4,
        "selection_batch_size": 2,
        "export_batch_size": 2,
        "num_workers": 0,
        "gradient_clip_norm": None,
        "mixed_precision": False,
        "seed": 0,
        "selection_split": "test",
        "selection_metric": "mean_top1_accuracy_over_all_60_future_steps",
        "early_stopping_metric": "mean_top1_accuracy_over_all_60_future_steps",
        "selection_direction": "maximise",
        "loss": {
            "type": "coarse_s1_cross_entropy",
            "horizon_weighting": "uniform",
            "dense_origins": True,
            "future_steps_per_origin": int(prediction_length),
            "origin_chunk_size": 1,
        },
    }
    v2_token_config: dict[str, Any] = {
        "model_family": "dimitri_basedygraph_v2_dense_token",
        "model_kind": "dimitri_v2_token",
        "do_not_report": True,
        "test_set_contaminated": True,
        "data": dict(token_data),
        "model": {
            "graph_family": "dimitri_v2_dual_fusion",
            "num_st_blocks": int(v2_defaults["num_st_blocks"]),
            "dimitri_defaults": v2_defaults,
            "temporal": {
                "type": "transformer",
                "d_model": int(v2_defaults["d_model"]),
                "num_layers": int(v2_defaults["num_temporal_layers"]),
                "num_heads": int(v2_defaults["nhead"]),
                "feedforward_multiplier": int(v2_defaults["ff_mult"]),
                "dropout": float(v2_defaults["dropout"]),
            },
            "graph": _v2_graph_schema(v2_defaults),
            "spatial": {
                "num_layers": int(v2_defaults["num_spatial_layers"]),
                "dropout": float(v2_defaults["spatial_dropout"]),
                "gate_type": "none",
                "initial_beta": None,
                "module_type": str(v2_defaults["spatial_module_type"]),
            },
            "prior": {
                "type": str(v2_defaults["graph_prior_level"]),
                "scale": float(v2_defaults["graph_prior_scale"]),
                "learnable": bool(v2_defaults["graph_prior_learnable"]),
            },
            "forecast_head": {
                "type": "structured_parallel_per_causal_origin",
                "prediction_length": int(prediction_length),
                "vocabulary_size": int(vocabulary_size),
                "num_layers": 1,
                "num_heads": 4,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
            },
        },
        "training": v2_token_training,
    }
    v2_token_config["models"] = _token_analysis_mirror(
        num_nodes=93,
        d_model=int(v2_defaults["d_model"]),
        temporal=v2_token_config["model"]["temporal"],
        graph=v2_token_config["model"]["graph"],
        spatial=v2_token_config["model"]["spatial"],
        prediction_length=prediction_length,
        evaluation_horizons=horizons,
        future_predictor=v2_token_config["model"]["forecast_head"],
    )

    v2_continuous_training = {
        "optimizer": "adamw",
        "parameter_grouping": "shared",
        "learning_rate": 0.0012,
        "weight_decay": 0.0001,
        "scheduler": "cosine_annealing",
        "scheduler_t_max": 120,
        "max_epochs": 120,
        "patience": 15,
        "min_delta": 0.0,
        "batch_size": 4,
        "selection_batch_size": 4,
        "export_batch_size": 4,
        "num_workers": 0,
        "gradient_clip_norm": None,
        "mixed_precision": False,
        "seed": 0,
        "selection_split": "test",
        "selection_metric": "unweighted_mean_five_horizon_cumulative_log_change_mae",
        "selection_direction": "minimise",
        "loss": {
            "type": "cumulative_log_change_mae",
            "bps_scale": 10000.0,
            "horizon_weighting": "inverse_reference_mae",
            "horizon_reference_mae": list(REFERENCE_MAE),
            "horizon_weights": list(inverse_reference_weights(REFERENCE_MAE)),
            "dense_origins": True,
        },
    }
    v2_continuous_config: dict[str, Any] = {
        "model_family": "dimitri_basedygraph_v2_dense_continuous",
        "model_kind": "dimitri_v2_continuous",
        "do_not_report": True,
        "test_set_contaminated": True,
        "data": dict(continuous_data),
        "normalisation": {
            "eps": 1.0e-8,
            "clip": False,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "model": {
            "graph_family": "dimitri_v2_dual_fusion",
            "num_st_blocks": int(v2_defaults["num_st_blocks"]),
            "dimitri_defaults": v2_defaults,
            "temporal": {
                "type": "transformer",
                "d_model": int(v2_defaults["d_model"]),
                "num_layers": int(v2_defaults["num_temporal_layers"]),
                "num_heads": int(v2_defaults["nhead"]),
                "feedforward_multiplier": int(v2_defaults["ff_mult"]),
                "dropout": float(v2_defaults["dropout"]),
            },
            "graph": _v2_graph_schema(v2_defaults),
            "spatial": {
                "num_layers": int(v2_defaults["num_spatial_layers"]),
                "dropout": float(v2_defaults["spatial_dropout"]),
                "gate_type": "none",
                "initial_beta": None,
                "module_type": str(v2_defaults["spatial_module_type"]),
            },
            "prior": {
                "type": str(v2_defaults["graph_prior_level"]),
                "scale": float(v2_defaults["graph_prior_scale"]),
                "learnable": bool(v2_defaults["graph_prior_learnable"]),
            },
            "forecast_head": {
                "type": "direct_five_horizon_at_every_causal_origin",
                "horizons": list(horizons),
            },
        },
        "training": v2_continuous_training,
    }

    specs = (
        _spec(
            model_kind="modern_tcn_token",
            label="Selected one-block ModernTCN graph architecture in coarse-s1 space",
            prefix="final_tok_mtg_d32_st1_abscorr_state_softmax",
            config=modern_config,
        ),
        _spec(
            model_kind="dense_transformer_token",
            label="Winning D64 three-block dense Transformer in coarse-s1 space",
            prefix="final_tok_dense_tr_d64_t4_g1_st3_uniform",
            config=dense_transformer_config,
        ),
        _spec(
            model_kind="dimitri_v2_token",
            label="Dimitri BaseDyGraph-V2 defaults, dense coarse-s1 forecasting",
            prefix="final_tok_dimitri_v2_c60_s15_dense60x60",
            config=v2_token_config,
        ),
        _spec(
            model_kind="dimitri_v2_continuous",
            label="Dimitri BaseDyGraph-V2 defaults, dense direct five-horizon price forecasting",
            prefix="final_px_dimitri_v2_c60_s15_dense5",
            config=v2_continuous_config,
        ),
    )

    signatures = [_hash(spec.config) for spec in specs]
    if len(set(signatures)) != len(signatures):
        raise AssertionError("Different final-comparison labels resolved to identical configs.")
    return specs


def save_specs(path: str | Path, specs: Sequence[FinalComparisonSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )

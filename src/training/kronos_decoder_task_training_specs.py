from __future__ import annotations

"""Specifications for task-specific Kronos-initialised Close decoders.

These runs preserve the frozen-forecaster, fixed-ten-path, all-60-loss and
Graph-Hub artifact contracts from the decoder post-training experiment.  The
only intentional change is optimisation: the pretrained Kronos decoder is used
as an initialisation and trained with a larger working learning rate, one epoch
of linear warm-up, and validation-driven plateau reductions.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.kronos_decoder_post_training_specs import (
    DEFAULT_LOSS_RATIO_ANCHORS,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_WEIGHT_SCALE,
    DecoderPostTrainingSpec,
    stretched_exponential_weights,
)


def _load_json(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return values


def _hash(values: Mapping[str, Any]) -> str:
    serialised = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _source_identity(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    config_path = source_dir / "resolved_config.json"
    metadata_path = source_dir / "run_metadata.json"
    checkpoint_path = source_dir / "best_checkpoint.pt"
    for path in (config_path, metadata_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    resolved = _load_json(config_path)
    metadata = _load_json(metadata_path)
    model_kind = str(resolved.get("model_kind", ""))
    if model_kind not in {"modern_tcn_token", "dense_transformer_token"}:
        raise ValueError(
            f"Unsupported source model_kind {model_kind!r} in {source_dir}."
        )

    return {
        "folder": source_dir.name,
        "path": str(source_dir),
        "model_kind": model_kind,
        "run_name": str(metadata.get("run_name", source_dir.name)),
        "run_signature": str(
            metadata.get("run_signature")
            or resolved.get("run_signature")
            or ""
        ),
        "best_epoch": int(metadata["best_epoch"]),
        "best_score": float(metadata["best_score"]),
        "source_config_signature": _hash(resolved),
    }


def _make_config(
    *,
    source_dir: Path,
    label: str,
    run_prefix: str,
    max_epochs: int,
    patience: int,
    train_batch_size: int,
    evaluation_batch_size: int,
    forecaster_batch_size: int,
    sample_chunk_size: int,
    learning_rate: float,
    warmup_start_learning_rate: float,
    warmup_epochs: int,
    weight_decay: float,
    plateau_factor: float,
    plateau_patience: int,
    minimum_learning_rate: float,
    gradient_clip_norm: float,
    mixed_precision: bool,
    seed: int,
    sample_count: int,
    temperature: float,
    top_k: int,
    top_p: float,
    weighting_scale: float,
    weighting_power: float,
) -> DecoderPostTrainingSpec:
    source = _source_identity(source_dir)

    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive.")
    if min(train_batch_size, evaluation_batch_size, forecaster_batch_size) <= 0:
        raise ValueError("All batch sizes must be positive.")
    if sample_chunk_size <= 0 or sample_chunk_size > sample_count:
        raise ValueError("sample_chunk_size must lie in [1, sample_count].")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 < warmup_start_learning_rate <= learning_rate:
        raise ValueError(
            "warmup_start_learning_rate must lie in (0, learning_rate]."
        )
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if not 0.0 < plateau_factor < 1.0:
        raise ValueError("plateau_factor must lie strictly between 0 and 1.")
    if plateau_patience <= 0:
        raise ValueError("plateau_patience must be positive.")
    if not 0.0 < minimum_learning_rate <= learning_rate:
        raise ValueError(
            "minimum_learning_rate must lie in (0, learning_rate]."
        )
    if gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")

    weights = stretched_exponential_weights(
        scale=weighting_scale,
        power=weighting_power,
        normalise=True,
    )

    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_family": "kronos_task_specific_coarse_decoder_training",
        "do_not_report": True,
        "test_set_contaminated_source_forecaster": True,
        "source_forecaster": source,
        "data": {
            "context_length": 60,
            "prediction_length": 60,
            "evaluation_horizons": [1, 5, 15, 30, 60],
            "token_stream": "coarse_s1",
            "target_channel": "close",
        },
        "sampling": {
            "sample_count": int(sample_count),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "seed": int(seed),
            "fixed_paths_across_decoder_epochs": True,
        },
        "decoder": {
            "initialisation": "pretrained_kronos_coarse_decoder",
            "training_intent": (
                "task_specific_close_decoder_initialised_from_pretrained_kronos"
            ),
            "conservative_fine_tuning": False,
            "trainable_scope": (
                "complete_coarse_reconstruction_branch:"
                "post_quant_embed_pre+decoder_stack+head"
            ),
            "forecasting_model_frozen": True,
            "tokenizer_encoder_frozen": True,
            "quantizer_and_token_definitions_frozen": True,
            "straight_through_estimator": False,
            "training_mode": "deterministic_eval_mode_with_gradients",
        },
        "loss": {
            "type": "stretched_exponential_weighted_all_60_clg_mae",
            "horizons": list(range(1, 61)),
            "loss_ratio_anchors_vs_h1": {
                str(key): float(value)
                for key, value in DEFAULT_LOSS_RATIO_ANCHORS.items()
            },
            "weight_function": "exp(-scale * (horizon - 1) ** power)",
            "weight_scale": float(weighting_scale),
            "weight_power": float(weighting_power),
            "weights_normalised_to_sum_one": True,
            "weights": list(weights),
            "ensemble_space": "decoded_raw_close",
            "ensemble_size": int(sample_count),
        },
        "training": {
            "optimisation_profile": (
                "task_specific_decoder_adamw_linear_warmup_plateau"
            ),
            "optimizer": "adamw",
            "max_learning_rate": float(learning_rate),
            "initial_learning_rate": float(warmup_start_learning_rate),
            "minimum_learning_rate": float(minimum_learning_rate),
            "weight_decay": float(weight_decay),
            "adam_betas": [0.9, 0.999],
            "adam_eps": 1.0e-8,
            "scheduler": "warmup_reduce_on_plateau",
            "warmup_start_learning_rate": float(warmup_start_learning_rate),
            "warmup_epochs": int(warmup_epochs),
            "plateau_mode": "min",
            "plateau_factor": float(plateau_factor),
            "plateau_patience": int(plateau_patience),
            "plateau_threshold": 0.0,
            "plateau_threshold_mode": "abs",
            "plateau_cooldown": 0,
            "scheduler_step_unit": (
                "optimizer_step_during_warmup_then_validation_epoch"
            ),
            "gradient_clip_norm": float(gradient_clip_norm),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": 0.0,
            "train_batch_size": int(train_batch_size),
            "evaluation_batch_size": int(evaluation_batch_size),
            "forecaster_batch_size": int(forecaster_batch_size),
            "sample_chunk_size": int(sample_chunk_size),
            "num_workers": 0,
            "mixed_precision": bool(mixed_precision),
            "seed": int(seed),
            "selection_split": "validation",
            "selection_metric": (
                "stretched_exponential_weighted_all_60_"
                "cumulative_log_change_mae"
            ),
        },
        "artifacts": {
            "copy_source_token_metrics": True,
            "copy_source_graphs": True,
            "save_train_validation_test_price_metrics": True,
            "save_all_ten_decoded_paths": True,
            "graph_hub_layout": True,
        },
    }

    signature = _hash(config)
    config["config_signature"] = signature
    run_name = f"{run_prefix}_{source['folder']}_{signature[:12]}"
    return DecoderPostTrainingSpec(
        label=label,
        run_name=run_name,
        source_forecaster_dir=source_dir.expanduser().resolve(),
        config=config,
    )


def make_decoder_task_training_specs(
    *,
    modern_tcn_source_dir: str | Path,
    dense_transformer_source_dir: str | Path,
    max_epochs: int = 100,
    patience: int = 10,
    train_batch_size: int = 1,
    evaluation_batch_size: int = 1,
    forecaster_batch_size: int = 2,
    sample_chunk_size: int = 2,
    learning_rate: float = 5.0e-4,
    warmup_start_learning_rate: float = 5.0e-5,
    warmup_epochs: int = 1,
    weight_decay: float = 0.1,
    plateau_factor: float = 0.5,
    plateau_patience: int = 3,
    minimum_learning_rate: float = 5.0e-6,
    gradient_clip_norm: float = 2.0,
    mixed_precision: bool = True,
    seed: int = 42,
    sample_count: int = 10,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    weighting_scale: float = DEFAULT_WEIGHT_SCALE,
    weighting_power: float = DEFAULT_WEIGHT_POWER,
) -> tuple[DecoderPostTrainingSpec, DecoderPostTrainingSpec]:
    """Build the two task-specific decoder-training runs."""
    common = {
        "max_epochs": max_epochs,
        "patience": patience,
        "train_batch_size": train_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "forecaster_batch_size": forecaster_batch_size,
        "sample_chunk_size": sample_chunk_size,
        "learning_rate": learning_rate,
        "warmup_start_learning_rate": warmup_start_learning_rate,
        "warmup_epochs": warmup_epochs,
        "weight_decay": weight_decay,
        "plateau_factor": plateau_factor,
        "plateau_patience": plateau_patience,
        "minimum_learning_rate": minimum_learning_rate,
        "gradient_clip_norm": gradient_clip_norm,
        "mixed_precision": mixed_precision,
        "seed": seed,
        "sample_count": sample_count,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "weighting_scale": weighting_scale,
        "weighting_power": weighting_power,
    }
    specs = (
        _make_config(
            source_dir=Path(modern_tcn_source_dir),
            label=(
                "ModernTCN token forecaster + task-specific "
                "Kronos-initialised Close decoder"
            ),
            run_prefix="tasktrain_close_decoder_moderntcn",
            **common,
        ),
        _make_config(
            source_dir=Path(dense_transformer_source_dir),
            label=(
                "Dense Transformer token forecaster + task-specific "
                "Kronos-initialised Close decoder"
            ),
            run_prefix="tasktrain_close_decoder_dense_transformer",
            **common,
        ),
    )
    signatures = [spec.config_signature for spec in specs]
    if len(set(signatures)) != len(signatures):
        raise AssertionError(
            "The two task-specific decoder runs resolved to identical configs."
        )
    return specs


def save_specs(
    path: str | Path,
    specs: Sequence[DecoderPostTrainingSpec],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )

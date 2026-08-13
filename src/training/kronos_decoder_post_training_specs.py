from __future__ import annotations

"""Specification builder for the final Kronos decoder post-training runs."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_LOSS_RATIO_ANCHORS = {
    1: 1.0,
    5: 1.976,
    15: 3.253,
    30: 4.498,
    60: 6.243,
}

# Least-squares fit of exp(-a * (h-1)^b) to the inverse anchor ratios.
DEFAULT_WEIGHT_SCALE = 0.41773697
DEFAULT_WEIGHT_POWER = 0.37551955


@dataclass(frozen=True)
class DecoderPostTrainingSpec:
    label: str
    run_name: str
    source_forecaster_dir: Path
    config: dict[str, Any]

    @property
    def config_signature(self) -> str:
        return str(self.config["config_signature"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "run_name": self.run_name,
            "source_forecaster_dir": str(self.source_forecaster_dir),
            "config": deepcopy(self.config),
        }


def _load_json(path: Path) -> dict[str, Any]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return values


def _hash(values: Mapping[str, Any]) -> str:
    serialised = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def stretched_exponential_weights(
    *,
    horizons: Sequence[int] = tuple(range(1, 61)),
    scale: float = DEFAULT_WEIGHT_SCALE,
    power: float = DEFAULT_WEIGHT_POWER,
    normalise: bool = True,
) -> tuple[float, ...]:
    import math

    horizons = tuple(int(value) for value in horizons)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers.")
    if scale <= 0.0 or power <= 0.0:
        raise ValueError("scale and power must be positive.")

    values = tuple(
        math.exp(-float(scale) * float(horizon - 1) ** float(power))
        for horizon in horizons
    )
    if normalise:
        total = sum(values)
        values = tuple(value / total for value in values)
    return values


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

    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_family": "kronos_coarse_decoder_post_training",
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
            "weights": list(
                stretched_exponential_weights(
                    scale=weighting_scale,
                    power=weighting_power,
                    normalise=True,
                )
            ),
            "ensemble_space": "decoded_raw_close",
            "ensemble_size": int(sample_count),
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": float(learning_rate),
            "weight_decay": 0.0,
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


def make_decoder_post_training_specs(
    *,
    modern_tcn_source_dir: str | Path,
    dense_transformer_source_dir: str | Path,
    max_epochs: int = 30,
    patience: int = 5,
    train_batch_size: int = 1,
    evaluation_batch_size: int = 1,
    forecaster_batch_size: int = 2,
    sample_chunk_size: int = 2,
    learning_rate: float = 1.0e-5,
    gradient_clip_norm: float = 1.0,
    mixed_precision: bool = True,
    seed: int = 42,
    sample_count: int = 10,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    weighting_scale: float = DEFAULT_WEIGHT_SCALE,
    weighting_power: float = DEFAULT_WEIGHT_POWER,
) -> tuple[DecoderPostTrainingSpec, DecoderPostTrainingSpec]:
    common = {
        "max_epochs": max_epochs,
        "patience": patience,
        "train_batch_size": train_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "forecaster_batch_size": forecaster_batch_size,
        "sample_chunk_size": sample_chunk_size,
        "learning_rate": learning_rate,
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
            label="ModernTCN token forecaster + post-trained Kronos decoder",
            run_prefix="posttrain_decoder_moderntcn",
            **common,
        ),
        _make_config(
            source_dir=Path(dense_transformer_source_dir),
            label="Dense Transformer token forecaster + post-trained Kronos decoder",
            run_prefix="posttrain_decoder_dense_transformer",
            **common,
        ),
    )
    signatures = [spec.config_signature for spec in specs]
    if len(set(signatures)) != len(signatures):
        raise AssertionError("The two decoder runs resolved to identical configs.")
    return specs


def save_specs(path: str | Path, specs: Sequence[DecoderPostTrainingSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )

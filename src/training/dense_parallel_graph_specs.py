from __future__ import annotations

"""Specifications and summaries for the twelve dense-training graph runs."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd


TemporalBackbone = Literal["modern_tcn", "transformer"]
TrainingStyle = Literal["stride1_fixed_context", "dense_prefix"]
GraphVariant = Literal[
    "correlation_static_dynamic_state",
    "random_static_dynamic_state",
    "dynamic_state",
]

DEFAULT_REFERENCE_MAE: tuple[float, ...] = (
    0.00036854,
    0.00078591,
    0.00132230,
    0.00183974,
    0.00255599,
)


@dataclass(frozen=True)
class DenseParallelRunSpec:
    run_name: str
    label: str
    temporal_backbone: TemporalBackbone
    training_style: TrainingStyle
    graph_variant: GraphVariant
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "temporal_backbone": self.temporal_backbone,
            "training_style": self.training_style,
            "graph_variant": self.graph_variant,
            "config": self.config,
        }


def _float_tag(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _hash(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def inverse_reference_weights(reference_mae: Sequence[float]) -> list[float]:
    values = [float(value) for value in reference_mae]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("reference_mae must contain positive values.")
    mean_value = sum(values) / len(values)
    return [mean_value / value for value in values]


def make_dense_parallel_graph_specs(
    *,
    context_length: int = 60,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    export_stride: int = 15,
    stride1_training_stride: int = 1,
    dense_prefix_outer_stride: int = 15,
    input_channels: Sequence[str] = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    ),
    target_channel: str = "close",
    reference_mae: Sequence[float] = DEFAULT_REFERENCE_MAE,
    alpha_initial: float = 0.5,
    beta_initial: float = 0.5,
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    graph_heads: int = 1,
    graph_hidden_dim: int = 32,
    modern_tcn_d_model: int = 32,
    modern_tcn_patch_size: int = 8,
    modern_tcn_patch_stride: int = 4,
    modern_tcn_ffn_ratio: int = 1,
    modern_tcn_num_blocks: int = 1,
    modern_tcn_large_kernel: int = 15,
    modern_tcn_small_kernel: int = 5,
    modern_tcn_dropout: float = 0.05,
    transformer_d_model: int = 96,
    transformer_num_layers: int = 1,
    transformer_num_heads: int = 8,
    transformer_feedforward_multiplier: int = 2,
    transformer_dropout: float = 0.0,
    transformer_position_embedding: bool = False,
    backbone_learning_rate: float = 2.5e-4,
    graph_learning_rate: float = 5.0e-4,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    max_epochs: int = 100,
    patience: int = 10,
    stride1_batch_size: int = 16,
    dense_prefix_modern_tcn_batch_size: int = 2,
    dense_prefix_transformer_batch_size: int = 2,
    selection_batch_size: int = 32,
    export_batch_size: int = 32,
    prefix_chunk_size: int = 8,
    prefix_graph_sample_windows: int = 8,
    mixed_precision: bool = True,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
) -> tuple[DenseParallelRunSpec, ...]:
    horizons = tuple(int(value) for value in horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be non-empty, unique and increasing.")
    if len(reference_mae) != len(horizons):
        raise ValueError("reference_mae must contain one value per horizon.")
    for name, value in {
        "context_length": context_length,
        "export_stride": export_stride,
        "stride1_training_stride": stride1_training_stride,
        "dense_prefix_outer_stride": dense_prefix_outer_stride,
        "graph_heads": graph_heads,
        "graph_hidden_dim": graph_hidden_dim,
        "max_epochs": max_epochs,
        "patience": patience,
        "stride1_batch_size": stride1_batch_size,
        "dense_prefix_modern_tcn_batch_size": dense_prefix_modern_tcn_batch_size,
        "dense_prefix_transformer_batch_size": dense_prefix_transformer_batch_size,
        "selection_batch_size": selection_batch_size,
        "export_batch_size": export_batch_size,
        "prefix_chunk_size": prefix_chunk_size,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(graph_hidden_dim) % int(graph_heads):
        raise ValueError("graph_hidden_dim must be divisible by graph_heads.")
    if int(transformer_d_model) % int(transformer_num_heads):
        raise ValueError("Transformer d_model must be divisible by heads.")
    if int(context_length) % int(modern_tcn_patch_stride):
        raise ValueError("context_length must be divisible by ModernTCN patch stride.")
    if not 0.0 < float(alpha_initial) < 1.0:
        raise ValueError("alpha_initial must lie in (0,1).")
    if not 0.0 < float(beta_initial) < 1.0:
        raise ValueError("beta_initial must lie in (0,1).")

    weights = inverse_reference_weights(reference_mae)
    horizon_tag = "-".join(str(value) for value in horizons)
    common_data = {
        "context_length": int(context_length),
        "horizons": list(horizons),
        "reported_horizons": list(horizons),
        "stride": int(export_stride),
        "export_stride": int(export_stride),
        "input_channels": [str(value) for value in input_channels],
        "target_channel": str(target_channel),
        "input_representation": "raw",
        "normalisation_clip": False,
        "normalisation_eps": 1.0e-8,
    }
    common_model = {
        "output_representation": "normalised_close",
        "output_head_initialisation": "default",
        "temporal": {
            # This field is overwritten for each run and makes Graph Hub's
            # architecture summary independent of model-family-specific paths.
            "type": None,
            "d_model": None,
            "num_layers": None,
            "num_heads": None,
            "feedforward_multiplier": None,
            "dropout": None,
            "modern_tcn": {
                "d_model": int(modern_tcn_d_model),
                "patch_size": int(modern_tcn_patch_size),
                "patch_stride": int(modern_tcn_patch_stride),
                "ffn_ratio": int(modern_tcn_ffn_ratio),
                "num_blocks": int(modern_tcn_num_blocks),
                "large_kernel": int(modern_tcn_large_kernel),
                "small_kernel": int(modern_tcn_small_kernel),
                "dropout": float(modern_tcn_dropout),
                "head_dropout": 0.0,
            },
            "transformer": {
                "d_model": int(transformer_d_model),
                "num_layers": int(transformer_num_layers),
                "num_heads": int(transformer_num_heads),
                "feedforward_multiplier": int(transformer_feedforward_multiplier),
                "dropout": float(transformer_dropout),
                "position_embedding": bool(transformer_position_embedding),
            },
        },
        "graph": {
            "type": None,
            "num_heads": int(graph_heads),
            "num_heads_per_layer": [int(graph_heads)],
            "hidden_dim": int(graph_hidden_dim),
            "activation": "softmax",
            "add_self_loops": False,
            "initial_alpha": float(alpha_initial),
        },
        "spatial": {
            "num_layers": 1,
            "feedforward_multiplier": 2,
            "dropout": 0.0,
            "gate_type": "learned_scalar",
            "initial_beta": float(beta_initial),
        },
        "prior": {
            "type": None,
            "scale": float(prior_scale),
            "jitter": float(prior_jitter),
            "seed": int(seed),
            "correlation_threshold": None,
        },
        "graph_regularisation": {
            "graph_reg_layer": -1,
            "graph_reg_warmup_epochs": 0,
            "graph_entropy_reg": 0.0,
            "graph_target_entropy": None,
            "graph_target_entropy_reg": 0.0,
            "graph_temporal_smooth_reg": 0.0,
        },
        "head_dropout": 0.0,
        "variant": None,
    }
    common_training = {
        "optimizer": "adam",
        "parameter_grouping": "split",
        "scheduler": "modern_tcn_type3_delayed",
        "scheduler_decay_start_epoch": int(decay_start_epoch),
        "scheduler_decay_factor": float(decay_factor),
        "learning_rate": float(backbone_learning_rate),
        "graph_learning_rate": float(graph_learning_rate),
        "weight_decay": 0.0,
        "selection_batch_size": int(selection_batch_size),
        "export_batch_size": int(export_batch_size),
        "num_workers": 0,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "min_delta": 0.0,
        "gradient_clip_norm": float(gradient_clip_norm),
        "mixed_precision": bool(mixed_precision),
        "seed": int(seed),
        "selection_split": "test",
        "selection_horizons": list(horizons),
        "selection_metric": "unweighted_mean_five_horizon_cumulative_log_change_mae",
        "loss": {
            "type": "cumulative_log_change_mae",
            "bps_scale": 10000.0,
            "horizon_weighting": "inverse_reference_mae",
            "horizon_reference_mae": [float(value) for value in reference_mae],
            "horizon_weights": [float(value) for value in weights],
        },
        "prefix_chunk_size": int(prefix_chunk_size),
        "prefix_graph_sample_windows": int(prefix_graph_sample_windows),
    }

    specs: list[DenseParallelRunSpec] = []
    for temporal_backbone in ("modern_tcn", "transformer"):
        for training_style in ("stride1_fixed_context", "dense_prefix"):
            for graph_variant in (
                "correlation_static_dynamic_state",
                "random_static_dynamic_state",
                "dynamic_state",
            ):
                config = json.loads(
                    json.dumps(
                        {
                            "model_family": "dense_parallel_graph_supervision",
                            "experiment_family": "dense_parallel_graph_supervision",
                            "temporal_backbone": temporal_backbone,
                            "training_style": training_style,
                            "graph_variant": graph_variant,
                            "data": common_data,
                            "normalisation": {
                                "eps": 1.0e-8,
                                "clip": False,
                                "clip_min": -5.0,
                                "clip_max": 5.0,
                            },
                            "model": common_model,
                            "training": common_training,
                        }
                    )
                )
                config["model"]["temporal"].update(
                    {
                        "type": temporal_backbone,
                        "d_model": (
                            int(modern_tcn_d_model)
                            if temporal_backbone == "modern_tcn"
                            else int(transformer_d_model)
                        ),
                        "num_layers": (
                            1
                            if temporal_backbone == "modern_tcn"
                            else int(transformer_num_layers)
                        ),
                        "num_heads": (
                            4
                            if temporal_backbone == "modern_tcn"
                            else int(transformer_num_heads)
                        ),
                        "feedforward_multiplier": (
                            2
                            if temporal_backbone == "modern_tcn"
                            else int(transformer_feedforward_multiplier)
                        ),
                        "dropout": (
                            float(modern_tcn_dropout)
                            if temporal_backbone == "modern_tcn"
                            else float(transformer_dropout)
                        ),
                    }
                )
                config["model"]["variant"] = graph_variant
                if graph_variant == "correlation_static_dynamic_state":
                    graph_type = "static_dynamic_mixture"
                    prior_type = "correlation"
                    graph_tag = "corr"
                elif graph_variant == "random_static_dynamic_state":
                    graph_type = "static_dynamic_mixture"
                    prior_type = "random"
                    graph_tag = "randomstatic"
                else:
                    graph_type = "dynamic"
                    prior_type = "none"
                    graph_tag = "dynamic"
                config["model"]["graph"]["type"] = graph_type
                config["model"]["prior"]["type"] = prior_type

                if training_style == "stride1_fixed_context":
                    training_stride = int(stride1_training_stride)
                    batch_size = int(stride1_batch_size)
                    style_tag = f"stride{training_stride}"
                    style_label = "stride-one fixed 60-minute contexts"
                else:
                    training_stride = int(dense_prefix_outer_stride)
                    batch_size = (
                        int(dense_prefix_modern_tcn_batch_size)
                        if temporal_backbone == "modern_tcn"
                        else int(dense_prefix_transformer_batch_size)
                    )
                    style_tag = f"denseprefix_s{training_stride}"
                    style_label = "60 internal dense-prefix origins per outer window"
                config["training"]["training_style"] = training_style
                config["training"]["training_stride"] = training_stride
                config["training"]["batch_size"] = batch_size
                config["training"]["optimisation_profile"] = (
                    "delayed_decay_inverse_reference_dense_parallel"
                )

                temporal_tag = (
                    f"mtg_d{int(modern_tcn_d_model)}"
                    if temporal_backbone == "modern_tcn"
                    else (
                        f"tr_d{int(transformer_d_model)}_"
                        f"l{int(transformer_num_layers)}_h{int(transformer_num_heads)}"
                    )
                )
                base_name = (
                    f"densemh_{temporal_tag}_{style_tag}_{graph_tag}_"
                    f"a{_float_tag(alpha_initial)}_b{_float_tag(beta_initial)}_"
                    f"g{int(graph_heads)}_gh{int(graph_hidden_dim)}_"
                    f"c{int(context_length)}_e{int(export_stride)}_h{horizon_tag}"
                )
                run_name = f"{base_name}_cfg{_hash(config)}"
                label = (
                    f"{temporal_backbone.replace('_', ' ').title()} — "
                    f"{style_label} — {graph_variant.replace('_', ' ')}"
                )
                specs.append(
                    DenseParallelRunSpec(
                        run_name=run_name,
                        label=label,
                        temporal_backbone=temporal_backbone,  # type: ignore[arg-type]
                        training_style=training_style,  # type: ignore[arg-type]
                        graph_variant=graph_variant,  # type: ignore[arg-type]
                        config=config,
                    )
                )

    if len(specs) != 12 or len({spec.run_name for spec in specs}) != 12:
        raise AssertionError("Expected twelve unique dense-parallel specifications.")
    return tuple(specs)


def save_specs(path: str | Path, specs: Sequence[DenseParallelRunSpec]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_config(path: str | Path, spec: DenseParallelRunSpec) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return values


def run_is_complete(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    metadata = directory / "run_metadata.json"
    checkpoint = directory / "best_checkpoint.pt"
    return (
        metadata.is_file()
        and checkpoint.is_file()
        and load_json(metadata).get("status") == "completed"
    )


def summarise_runs(
    output_root: str | Path,
    specs: Sequence[DenseParallelRunSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        history_path = directory / "history.csv"
        if not metadata_path.is_file() or not history_path.is_file():
            missing.append(spec.run_name)
            continue
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            missing.append(spec.run_name)
            continue
        history = pd.read_csv(history_path)
        best_epoch = int(metadata["best_epoch"])
        selected = history.loc[history["epoch"] == best_epoch]
        if len(selected) != 1:
            raise AssertionError(
                f"Expected one best-epoch row for {spec.run_name}; found {len(selected)}."
            )
        row = selected.iloc[0]
        result: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Temporal backbone": spec.temporal_backbone,
            "Training style": spec.training_style,
            "Graph variant": spec.graph_variant,
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Mean test Log MAE": float(row["selection_score"]),
            "Alpha": row.get("block_0_alpha"),
            "Beta": row.get("block_0_beta"),
            "Selected graph entropy": row.get("block_0_selected_entropy"),
            "Dynamic graph entropy": row.get("block_0_dynamic_entropy"),
            "Static graph entropy": row.get("block_0_static_entropy"),
            "Trainable parameters": metadata.get("trainable_parameters"),
            "Training windows": metadata.get("training_windows"),
            "Export train windows": metadata.get("train_windows"),
        }
        for horizon in spec.config["data"]["horizons"]:
            result[f"Log MAE — {int(horizon)} min"] = float(
                row[f"test_cumulative_log_change_mae_h{int(horizon)}"]
            )
        rows.append(result)
    if require_all and missing:
        raise FileNotFoundError(
            "Missing or incomplete runs:\n" + "\n".join(f"  - {name}" for name in missing)
        )
    if not rows:
        raise RuntimeError("No completed dense-parallel runs were found.")
    return pd.DataFrame(rows).sort_values(
        ["Mean test Log MAE", "Run"]
    ).reset_index(drop=True)

from __future__ import annotations

"""Configuration helpers for the two BaseDyGraph-v1 token controls."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.models.basedygraph_v1_token_comparison import (
    BaseDyGraphV1PredictionMode,
)


@dataclass(frozen=True)
class BaseDyGraphV1TokenRunSpec:
    run_name: str
    label: str
    prediction_mode: BaseDyGraphV1PredictionMode
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "prediction_mode": self.prediction_mode,
            "config": self.config,
        }


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _make_config(
    *,
    prediction_mode: BaseDyGraphV1PredictionMode,
    context_length: int,
    prediction_length: int,
    evaluation_horizons: Sequence[int],
    d_model: int,
    temporal_num_layers: int,
    temporal_num_heads: int,
    feedforward_multiplier: int,
    graph_num_heads: int,
    graph_hidden_dim: int,
    num_st_blocks: int,
    dropout: float,
    spatial_dropout: float,
    future_predictor_num_layers: int,
    future_predictor_num_heads: int,
    future_predictor_feedforward_multiplier: int,
    future_predictor_dropout: float,
    max_epochs: int,
    patience: int,
    train_batch_size: int,
    selection_batch_size: int,
    export_batch_size: int,
    num_workers: int,
    backbone_learning_rate: float,
    graph_learning_rate: float,
    decay_start_epoch: int,
    decay_factor: float,
    mixed_precision: bool,
    gradient_clip_norm: float,
    seed: int,
) -> dict[str, Any]:
    public_horizons = (
        [1]
        if prediction_mode == "dense_one_step"
        else [int(value) for value in evaluation_horizons]
    )
    dense_output_length = (
        int(context_length)
        if prediction_mode == "dense_one_step"
        else int(prediction_length)
    )
    predictor_type = (
        "official_dense_next_state_head"
        if prediction_mode == "dense_one_step"
        else "structured_parallel"
    )
    config: dict[str, Any] = {
        "model_family": "official_basedygraph_v1_token_comparison",
        "data": {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "model_output_length": dense_output_length,
            "evaluation_horizons": [int(value) for value in evaluation_horizons],
            "public_horizons": public_horizons,
            "input_token_stream": "s1",
            "target_token_stream": "s1",
            "s1_vocabulary_size": 1024,
            "s1_id_space": "kronos_original",
        },
        "model": {
            "prediction_mode": prediction_mode,
            "official_basedygraph_v1": {
                "d_model": int(d_model),
                "temporal_num_layers": int(temporal_num_layers),
                "temporal_num_heads": int(temporal_num_heads),
                "spatial_num_layers": 1,
                "feedforward_multiplier": int(feedforward_multiplier),
                "graph_num_heads": int(graph_num_heads),
                "graph_hidden_dim": int(graph_hidden_dim),
                "num_st_blocks": int(num_st_blocks),
                "dropout": float(dropout),
                "spatial_dropout": float(spatial_dropout),
                "spatial_module_type": "dynamic_graph",
                "spatial_value": "hidden",
                "graph_activation": "softmax",
                "use_node_embedding": True,
                "use_state_pair_bias": False,
                "add_self_loops": False,
                "symmetric_graph": False,
                "st_block_post_norm": True,
            },
            "future_predictor": {
                "type": predictor_type,
                "num_layers": (
                    0
                    if prediction_mode == "dense_one_step"
                    else int(future_predictor_num_layers)
                ),
                "num_heads": int(future_predictor_num_heads),
                "feedforward_multiplier": int(
                    future_predictor_feedforward_multiplier
                ),
                "dropout": float(future_predictor_dropout),
            },
            "graph_regularisation": {
                "graph_reg_layer": -1,
                "graph_reg_warmup_epochs": 0,
                "graph_entropy_reg": 0.0,
                "graph_target_entropy": None,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
        },
        # Compatibility mirror used only by the existing Graph Hub metadata
        # parser.  The actual constructor contract is model.official_basedygraph_v1.
        "models": {
            "dynamic_graph": {
                "num_nodes": 0,
                "d_model": int(d_model),
                "num_st_blocks": int(num_st_blocks),
                "temporal": {
                    "type": "transformer",
                    "d_model": int(d_model),
                    "num_layers": int(temporal_num_layers),
                    "num_heads": int(temporal_num_heads),
                    "feedforward_multiplier": int(feedforward_multiplier),
                    "dropout": float(dropout),
                },
                "graph": {
                    "type": "dynamic",
                    "num_heads": int(graph_num_heads),
                    "num_heads_per_layer": [
                        int(graph_num_heads)
                    ] * int(num_st_blocks),
                    "hidden_dim": int(graph_hidden_dim),
                    "hidden_dims_per_layer": [
                        int(graph_hidden_dim)
                    ] * int(num_st_blocks),
                    "activation": "softmax",
                    "activations_per_layer": [
                        "softmax"
                    ] * int(num_st_blocks),
                    "add_self_loops": False,
                },
                "heads": {
                    "evaluation_horizons": public_horizons,
                    "prediction_length": dense_output_length,
                    "future_token_mode": "coarse_only",
                    "s1_vocabulary_size": 1024,
                },
                "future_predictor": {
                    "type": predictor_type,
                    "num_layers": (
                        0
                        if prediction_mode == "dense_one_step"
                        else int(future_predictor_num_layers)
                    ),
                },
            }
        },
        "training": {
            "optimizer": "adam",
            "parameter_grouping": "split",
            "scheduler": "modern_tcn_type3_delayed",
            "scheduler_decay_start_epoch": int(decay_start_epoch),
            "scheduler_decay_factor": float(decay_factor),
            "learning_rate": float(backbone_learning_rate),
            "graph_learning_rate": float(graph_learning_rate),
            "weight_decay": 0.0,
            "batch_size": int(train_batch_size),
            "selection_batch_size": int(selection_batch_size),
            "export_batch_size": int(export_batch_size),
            "num_workers": int(num_workers),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "min_delta": 0.0,
            "gradient_clip_norm": float(gradient_clip_norm),
            "mixed_precision": bool(mixed_precision),
            "seed": int(seed),
            "selection_split": "test",
            "selection_direction": "maximise",
            "selection_metric": (
                "dense_teacher_forced_mean_top1_accuracy"
                if prediction_mode == "dense_one_step"
                else "mean_top1_accuracy_over_all_future_steps"
            ),
            "early_stopping_metric": (
                "dense_teacher_forced_mean_top1_accuracy"
                if prediction_mode == "dense_one_step"
                else "mean_top1_accuracy_over_all_future_steps"
            ),
            "loss": {
                "type": "coarse_s1_cross_entropy",
                "horizon_weighting": "uniform",
            },
        },
    }
    return config


def make_basedygraph_v1_token_comparison_specs(
    *,
    context_length: int = 60,
    prediction_length: int = 60,
    evaluation_horizons: Sequence[int] = (1, 5, 15, 30, 60),
    d_model: int = 96,
    temporal_num_layers: int = 1,
    temporal_num_heads: int = 4,
    feedforward_multiplier: int = 2,
    graph_num_heads: int = 1,
    graph_hidden_dim: int = 96,
    num_st_blocks: int = 4,
    dropout: float = 0.0,
    spatial_dropout: float = 0.0,
    future_predictor_num_layers: int = 1,
    future_predictor_num_heads: int = 4,
    future_predictor_feedforward_multiplier: int = 2,
    future_predictor_dropout: float = 0.0,
    max_epochs: int = 100,
    patience: int = 10,
    train_batch_size: int = 2,
    selection_batch_size: int = 2,
    export_batch_size: int = 2,
    num_workers: int = 0,
    backbone_learning_rate: float = 2.5e-4,
    graph_learning_rate: float = 5.0e-4,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    mixed_precision: bool = True,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
) -> tuple[BaseDyGraphV1TokenRunSpec, BaseDyGraphV1TokenRunSpec]:
    specs: list[BaseDyGraphV1TokenRunSpec] = []
    labels = {
        "dense_one_step": (
            "BaseDyGraph v1 — four dynamic ST blocks — dense teacher-forced "
            "one-step head"
        ),
        "parallel_60": (
            "BaseDyGraph v1 — four dynamic ST blocks — structured-parallel "
            "60-minute head"
        ),
    }
    for mode in ("dense_one_step", "parallel_60"):
        config = _make_config(
            prediction_mode=mode,
            context_length=context_length,
            prediction_length=prediction_length,
            evaluation_horizons=evaluation_horizons,
            d_model=d_model,
            temporal_num_layers=temporal_num_layers,
            temporal_num_heads=temporal_num_heads,
            feedforward_multiplier=feedforward_multiplier,
            graph_num_heads=graph_num_heads,
            graph_hidden_dim=graph_hidden_dim,
            num_st_blocks=num_st_blocks,
            dropout=dropout,
            spatial_dropout=spatial_dropout,
            future_predictor_num_layers=future_predictor_num_layers,
            future_predictor_num_heads=future_predictor_num_heads,
            future_predictor_feedforward_multiplier=(
                future_predictor_feedforward_multiplier
            ),
            future_predictor_dropout=future_predictor_dropout,
            max_epochs=max_epochs,
            patience=patience,
            train_batch_size=train_batch_size,
            selection_batch_size=selection_batch_size,
            export_batch_size=export_batch_size,
            num_workers=num_workers,
            backbone_learning_rate=backbone_learning_rate,
            graph_learning_rate=graph_learning_rate,
            decay_start_epoch=decay_start_epoch,
            decay_factor=decay_factor,
            mixed_precision=mixed_precision,
            gradient_clip_norm=gradient_clip_norm,
            seed=seed,
        )
        signature = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:10]
        mode_tag = "dense1_tf" if mode == "dense_one_step" else "parallel60"
        run_name = (
            f"basedygraph_v1_tr4_d{int(d_model)}_tl{int(temporal_num_layers)}_"
            f"th{int(temporal_num_heads)}_ff{int(feedforward_multiplier)}_"
            f"dynamic_g{int(graph_num_heads)}_gh{int(graph_hidden_dim)}_"
            f"{mode_tag}_ds{int(decay_start_epoch)}_df{_float_tag(decay_factor)}_"
            f"cfg{signature}"
        )
        specs.append(
            BaseDyGraphV1TokenRunSpec(
                run_name=run_name,
                label=labels[mode],
                prediction_mode=mode,
                config=config,
            )
        )
    return specs[0], specs[1]


def save_run_config(
    path: str | Path,
    spec: BaseDyGraphV1TokenRunSpec,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_specs(
    path: str | Path,
    specs: Sequence[BaseDyGraphV1TokenRunSpec],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [spec.to_dict() for spec in specs],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


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
    specs: Sequence[BaseDyGraphV1TokenRunSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        diagnostics_path = directory / "best_test_diagnostics.json"
        if not (metadata_path.is_file() and diagnostics_path.is_file()):
            missing.append(spec.run_name)
            continue
        metadata = load_json(metadata_path)
        diagnostics = load_json(diagnostics_path)
        if metadata.get("status") != "completed":
            missing.append(spec.run_name)
            continue
        rows.append(
            {
                "Run": spec.run_name,
                "Task": spec.prediction_mode,
                "Best epoch": int(metadata["best_epoch"]),
                "Epochs completed": int(metadata["epochs_completed"]),
                "Selection top-1 accuracy": float(metadata["best_score"]),
                "Forecast h1 top-1 accuracy": float(
                    diagnostics["forecast_h1_top1_accuracy"]
                ),
                "Mean future top-1 accuracy": diagnostics.get(
                    "mean_future_top1_accuracy"
                ),
                "Final graph entropy": diagnostics.get(
                    "final_graph_entropy"
                ),
                "Final graph effective neighbours": diagnostics.get(
                    "final_graph_effective_neighbours"
                ),
                "Run directory": str(directory),
            }
        )
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed BaseDyGraph-v1 token runs: "
            + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed BaseDyGraph-v1 token runs were found.")
    return pd.DataFrame(rows)


__all__ = [
    "BaseDyGraphV1TokenRunSpec",
    "load_json",
    "make_basedygraph_v1_token_comparison_specs",
    "run_is_complete",
    "save_run_config",
    "save_specs",
    "summarise_runs",
]

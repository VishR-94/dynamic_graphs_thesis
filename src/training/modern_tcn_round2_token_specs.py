from __future__ import annotations

"""Specifications and result summaries for the 12-model token Round-2 sweep."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.models.modern_tcn_graph_round2 import GraphFamily, PriorType, TemporalFamily


@dataclass(frozen=True)
class TokenRound2RunSpec:
    run_name: str
    label: str
    temporal_family: TemporalFamily
    num_transformer_blocks: int
    num_st_blocks: int
    graph_family: GraphFamily
    prior_type: PriorType
    graph_heads_per_block: tuple[int, ...]
    graph_hidden_dims_per_block: tuple[int, ...]
    graph_activations_per_block: tuple[str, ...]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "temporal_family": self.temporal_family,
            "num_transformer_blocks": self.num_transformer_blocks,
            "num_st_blocks": self.num_st_blocks,
            "graph_family": self.graph_family,
            "prior_type": self.prior_type,
            "graph_heads_per_block": list(self.graph_heads_per_block),
            "graph_hidden_dims_per_block": list(
                self.graph_hidden_dims_per_block
            ),
            "graph_activations_per_block": list(
                self.graph_activations_per_block
            ),
            "config": self.config,
        }


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _head_schedule(
    graph_heads: int | Mapping[int, Sequence[int]],
    *,
    block_count: int,
) -> tuple[int, ...]:
    if isinstance(graph_heads, Mapping):
        if int(block_count) not in graph_heads:
            raise KeyError(
                "GRAPH_HEAD_SCHEDULES must contain every requested ST depth; "
                f"missing depth {block_count}."
            )
        values = tuple(int(value) for value in graph_heads[int(block_count)])
    else:
        values = tuple([int(graph_heads)] * int(block_count))
    if len(values) != int(block_count):
        raise ValueError(
            f"Graph-head schedule {values} has length {len(values)}; "
            f"expected {block_count}."
        )
    if any(value <= 0 for value in values):
        raise ValueError("Every graph-head count must be positive.")
    return values


def _base_config(
    *,
    context_length: int,
    prediction_length: int,
    evaluation_horizons: Sequence[int],
    prior_type: PriorType,
    prior_scale: float,
    prior_jitter: float,
    alpha_initial: float,
    beta_initial: float,
    seed: int,
    transformer_d_model: int,
    transformer_num_layers: int,
    transformer_num_heads: int,
    transformer_feedforward_multiplier: int,
    transformer_dropout: float,
    future_predictor_num_layers: int,
    future_predictor_num_heads: int,
    future_predictor_feedforward_multiplier: int,
    future_predictor_dropout: float,
    max_epochs: int,
    patience: int,
    train_batch_size: int,
    selection_batch_size: int,
    export_batch_size: int,
    backbone_learning_rate: float,
    graph_learning_rate: float,
    decay_start_epoch: int,
    decay_factor: float,
    mixed_precision: bool,
    gradient_clip_norm: float,
    num_workers: int,
) -> dict[str, Any]:
    horizons = tuple(int(value) for value in evaluation_horizons)
    return {
        "model_family": "modern_tcn_graph_round2_token",
        "data": {
            "context_length": int(context_length),
            "prediction_length": int(prediction_length),
            "evaluation_horizons": list(horizons),
            "input_token_stream": "s1",
            "target_token_stream": "s1",
            "s1_vocabulary_size": 1024,
            "s1_id_space": "kronos_original",
        },
        "model": {
            "graph_family": "dynamic_only",
            "temporal_stack": {
                "family": "modern_tcn_transformer",
                "num_transformer_blocks": 1,
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
                "transformer": {
                    "d_model": int(transformer_d_model),
                    "num_layers": int(transformer_num_layers),
                    "num_heads": int(transformer_num_heads),
                    "feedforward_multiplier": int(
                        transformer_feedforward_multiplier
                    ),
                    "dropout": float(transformer_dropout),
                    "relative_position_embedding": True,
                },
            },
            "graph": {
                "num_heads_per_block": [1, 1],
                "hidden_dims_per_block": [32, int(transformer_d_model)],
                "activations_per_block": ["softmax", "sparsemax"],
                "initial_alpha": float(alpha_initial),
                "add_self_loops": False,
            },
            "spatial": {
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": float(beta_initial),
            },
            "prior": {
                "type": str(prior_type),
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
            "future_predictor": {
                "type": "structured_parallel",
                "num_layers": int(future_predictor_num_layers),
                "num_heads": int(future_predictor_num_heads),
                "feedforward_multiplier": int(
                    future_predictor_feedforward_multiplier
                ),
                "dropout": float(future_predictor_dropout),
            },
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
            "selection_metric": "mean_top1_accuracy_over_all_future_steps",
            "early_stopping_metric": "mean_top1_accuracy_over_all_future_steps",
            "selection_direction": "maximise",
            "loss": {
                "type": "coarse_s1_cross_entropy",
                "horizon_weighting": "uniform",
            },
        },
    }


def _make_spec(
    *,
    base: Mapping[str, Any],
    temporal_family: TemporalFamily,
    num_transformer_blocks: int,
    graph_family: GraphFamily,
    prior_type: PriorType,
    graph_heads: int | Mapping[int, Sequence[int]],
    modern_tcn_graph_dim_per_head: int,
    transformer_graph_dim_per_head: int,
) -> TokenRound2RunSpec:
    if temporal_family == "modern_tcn_transformer":
        block_count = 1 + int(num_transformer_blocks)
        temporal_tag = f"mtg_t{int(num_transformer_blocks)}"
        temporal_label = (
            "ModernTCN first ST block + "
            f"{int(num_transformer_blocks)} Transformer ST block(s)"
        )
    elif temporal_family == "transformer_only":
        block_count = int(num_transformer_blocks)
        temporal_tag = f"tr{block_count}"
        temporal_label = f"{block_count} Transformer ST block(s)"
    else:
        raise ValueError(f"Unsupported temporal_family {temporal_family!r}.")
    if block_count < 2:
        raise ValueError("Token Round 2 requires at least two ST blocks.")

    heads = _head_schedule(graph_heads, block_count=block_count)
    hidden_dims = tuple(
        int(heads[index])
        * int(
            modern_tcn_graph_dim_per_head
            if temporal_family == "modern_tcn_transformer" and index == 0
            else transformer_graph_dim_per_head
        )
        for index in range(block_count)
    )
    activations = tuple(["softmax"] * (block_count - 1) + ["sparsemax"])

    values = json.loads(json.dumps(base))
    values["model"]["temporal_stack"]["family"] = temporal_family
    values["model"]["temporal_stack"]["num_transformer_blocks"] = int(
        num_transformer_blocks
    )
    values["model"]["graph_family"] = graph_family
    values["model"]["prior"]["type"] = (
        "none" if graph_family == "dynamic_only" else str(prior_type)
    )
    values["model"]["graph"]["num_heads_per_block"] = list(heads)
    values["model"]["graph"]["hidden_dims_per_block"] = list(hidden_dims)
    values["model"]["graph"]["activations_per_block"] = list(activations)

    # Graph-Hub-compatible token schema.  The full architecture remains in
    # ``model``; this lightweight mirror avoids requiring price predictions.
    values["models"] = {
        "dynamic_graph": {
            "num_nodes": None,
            "d_model": int(
                values["model"]["temporal_stack"]["transformer"]["d_model"]
            ),
            "num_st_blocks": int(block_count),
            "token_input_representation": "coarse_s1_embedding",
            "temporal": {
                "type": str(temporal_family),
                "num_layers": int(block_count),
            },
            "graph": {
                "type": (
                    "dynamic"
                    if graph_family == "dynamic_only"
                    else "dynamic_base"
                ),
                "num_heads": int(heads[-1]),
                "num_heads_per_layer": list(heads),
                "hidden_dim": int(hidden_dims[-1]),
                "hidden_dims_per_layer": list(hidden_dims),
                "activation": str(activations[-1]),
                "activations_per_layer": list(activations),
                "add_self_loops": False,
                "initial_alpha": float(values["model"]["graph"]["initial_alpha"]),
            },
            "spatial": {
                "gate_type": "learned_scalar",
                "initial_beta": float(
                    values["model"]["spatial"]["initial_beta"]
                ),
            },
            "heads": {
                "prediction_length": int(values["data"]["prediction_length"]),
                "evaluation_horizons": list(
                    values["data"]["evaluation_horizons"]
                ),
                "dense_horizons": list(
                    range(1, int(values["data"]["prediction_length"]) + 1)
                ),
                "reported_horizons": list(
                    values["data"]["evaluation_horizons"]
                ),
                "future_token_mode": "coarse_only",
                "s1_vocabulary_size": int(values["data"]["s1_vocabulary_size"]),
                "s2_vocabulary_size": 1024,
                "s2_loss_weight": 0.0,
            },
            "future_predictor": dict(values["model"]["future_predictor"]),
        }
    }

    graph_tag = (
        "dynamic"
        if graph_family == "dynamic_only"
        else f"{prior_type}_state"
    )
    heads_tag = "g" + "-".join(str(value) for value in heads)
    transformer = values["model"]["temporal_stack"]["transformer"]
    predictor = values["model"]["future_predictor"]
    run_name = (
        f"tok_r2_{temporal_tag}_{graph_tag}_{heads_tag}_"
        f"td{int(transformer['d_model'])}_tl{int(transformer['num_layers'])}_"
        f"th{int(transformer['num_heads'])}_tf{int(transformer['feedforward_multiplier'])}_"
        f"ph{int(predictor['num_heads'])}_pl{int(predictor['num_layers'])}_"
        f"c{int(values['data']['context_length'])}_p{int(values['data']['prediction_length'])}"
    )
    if graph_family == "prior_state":
        run_name += (
            f"_a{_float_tag(values['model']['graph']['initial_alpha'])}"
            f"_b{_float_tag(values['model']['spatial']['initial_beta'])}"
            f"_ps{_float_tag(values['model']['prior']['scale'])}"
            f"_pj{_float_tag(values['model']['prior']['jitter'])}"
        )
    config_signature = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    run_name += f"_cfg{config_signature}"

    label = f"{temporal_label} — " + (
        "dynamic-only graph"
        if graph_family == "dynamic_only"
        else (
            f"{prior_type} prior + dynamic graph + coarse-state pathway"
        )
    )
    return TokenRound2RunSpec(
        run_name=run_name,
        label=label,
        temporal_family=temporal_family,
        num_transformer_blocks=int(num_transformer_blocks),
        num_st_blocks=int(block_count),
        graph_family=graph_family,
        prior_type=("none" if graph_family == "dynamic_only" else prior_type),
        graph_heads_per_block=heads,
        graph_hidden_dims_per_block=hidden_dims,
        graph_activations_per_block=activations,
        config=values,
    )


def make_token_round2_specs(
    *,
    prior_type: PriorType = "correlation",
    graph_heads: int | Mapping[int, Sequence[int]] = 1,
    context_length: int = 60,
    prediction_length: int = 60,
    evaluation_horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    alpha_initial: float = 0.5,
    beta_initial: float = 0.5,
    seed: int = 42,
    transformer_d_model: int = 96,
    transformer_num_layers: int = 1,
    transformer_num_heads: int = 4,
    transformer_feedforward_multiplier: int = 2,
    transformer_dropout: float = 0.0,
    future_predictor_num_layers: int = 1,
    future_predictor_num_heads: int = 4,
    future_predictor_feedforward_multiplier: int = 2,
    future_predictor_dropout: float = 0.0,
    modern_tcn_graph_dim_per_head: int = 32,
    transformer_graph_dim_per_head: int = 96,
    max_epochs: int = 100,
    patience: int = 10,
    train_batch_size: int = 2,
    selection_batch_size: int = 2,
    export_batch_size: int = 2,
    backbone_learning_rate: float = 2.5e-4,
    graph_learning_rate: float = 5.0e-4,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    mixed_precision: bool = True,
    gradient_clip_norm: float = 1.0,
    num_workers: int = 0,
) -> tuple[TokenRound2RunSpec, ...]:
    """Return the exact six temporal stacks for both graph families."""
    if prior_type not in {"sector", "correlation", "none"}:
        raise ValueError("prior_type must be sector, correlation, or none.")
    base = _base_config(
        context_length=context_length,
        prediction_length=prediction_length,
        evaluation_horizons=evaluation_horizons,
        prior_type=prior_type,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        alpha_initial=alpha_initial,
        beta_initial=beta_initial,
        seed=seed,
        transformer_d_model=transformer_d_model,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_feedforward_multiplier=transformer_feedforward_multiplier,
        transformer_dropout=transformer_dropout,
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
        backbone_learning_rate=backbone_learning_rate,
        graph_learning_rate=graph_learning_rate,
        decay_start_epoch=decay_start_epoch,
        decay_factor=decay_factor,
        mixed_precision=mixed_precision,
        gradient_clip_norm=gradient_clip_norm,
        num_workers=num_workers,
    )
    temporal_definitions: tuple[tuple[TemporalFamily, int], ...] = (
        ("modern_tcn_transformer", 1),
        ("modern_tcn_transformer", 2),
        ("modern_tcn_transformer", 3),
        ("transformer_only", 2),
        ("transformer_only", 3),
        ("transformer_only", 4),
    )
    result = tuple(
        _make_spec(
            base=base,
            temporal_family=temporal_family,
            num_transformer_blocks=transformer_blocks,
            graph_family=graph_family,
            prior_type=prior_type,
            graph_heads=graph_heads,
            modern_tcn_graph_dim_per_head=modern_tcn_graph_dim_per_head,
            transformer_graph_dim_per_head=transformer_graph_dim_per_head,
        )
        for graph_family in ("dynamic_only", "prior_state")
        for temporal_family, transformer_blocks in temporal_definitions
    )
    if len(result) != 12 or len({spec.run_name for spec in result}) != 12:
        raise AssertionError("Expected twelve unique token Round-2 specs.")
    return result


def save_specs(path: str | Path, specs: Sequence[TokenRound2RunSpec]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_run_config(path: str | Path, spec: TokenRound2RunSpec) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
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
    specs: Sequence[TokenRound2RunSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        history_path = directory / "history.csv"
        metric_path = directory / "best_test_token_metric_table.csv"
        if not (
            metadata_path.is_file()
            and history_path.is_file()
            and metric_path.is_file()
        ):
            missing.append(spec.run_name)
            continue
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            missing.append(spec.run_name)
            continue
        history = pd.read_csv(history_path)
        best_epoch = int(metadata["best_epoch"])
        selected = history.loc[pd.to_numeric(history["epoch"]) == best_epoch]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one best-epoch row for {spec.run_name}; "
                f"found {len(selected)}."
            )
        best = selected.iloc[0]
        metrics = pd.read_csv(metric_path)
        by_step = metrics.set_index("future_step")
        row: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Temporal family": spec.temporal_family,
            "Transformer blocks": spec.num_transformer_blocks,
            "ST blocks": spec.num_st_blocks,
            "Graph family": spec.graph_family,
            "Prior": spec.prior_type,
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Mean test top-1 accuracy — all future steps": float(
                metadata["best_score"]
            ),
            "Mean test cross-entropy — all 60": float(
                best["test_mean_cross_entropy"]
            ),
            "Run directory": str(directory),
        }
        for horizon in spec.config["data"]["evaluation_horizons"]:
            row[f"Top-1 accuracy — {int(horizon)} min"] = float(
                by_step.loc[int(horizon), "top1_accuracy"]
            )
        for block_index in range(spec.num_st_blocks):
            row[f"Block {block_index} alpha"] = best.get(
                f"block_{block_index}_alpha"
            )
            row[f"Block {block_index} beta"] = best.get(
                f"block_{block_index}_beta"
            )
            row[f"Block {block_index} selected entropy"] = best.get(
                f"block_{block_index}_selected_entropy"
            )
        rows.append(row)
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed token Round-2 runs: " + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed token Round-2 runs were found.")
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["Mean test top-1 accuracy — all future steps", "Run"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

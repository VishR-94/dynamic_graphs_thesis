from __future__ import annotations

"""Specifications and summaries for the two-family Round-2 depth grid."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


TemporalFamily = Literal["modern_tcn_transformer", "transformer_only"]
GraphFamily = Literal["dynamic_only", "prior_state"]
PriorType = Literal["none", "sector", "correlation"]


@dataclass(frozen=True)
class Round2RunSpec:
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
            "num_transformer_blocks": int(self.num_transformer_blocks),
            "num_st_blocks": int(self.num_st_blocks),
            "graph_family": self.graph_family,
            "prior_type": self.prior_type,
            "graph_heads_per_block": [
                int(value) for value in self.graph_heads_per_block
            ],
            "graph_hidden_dims_per_block": [
                int(value) for value in self.graph_hidden_dims_per_block
            ],
            "graph_activations_per_block": list(
                self.graph_activations_per_block
            ),
            "config": self.config,
        }


def _float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _base_config(
    *,
    context_length: int,
    stride: int,
    horizons: Sequence[int],
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
) -> dict[str, Any]:
    return {
        "data": {
            "context_length": int(context_length),
            "stride": int(stride),
            "horizons": [int(value) for value in horizons],
            "input_channels": ["open", "high", "low", "close", "volume"],
            "target_channel": "close",
            "input_representation": "raw",
        },
        "normalisation": {
            "eps": 1.0e-8,
            "clip": False,
            "clip_min": -5.0,
            "clip_max": 5.0,
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
                    "session_position_encoding": True,
                },
            },
            "graph": {
                "num_heads_per_block": [1, 1],
                "hidden_dims_per_block": [32, transformer_d_model],
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
            "output_representation": "normalised_close",
        },
        "training": {
            "optimizer": "adam",
            "parameter_grouping": "split",
            "scheduler": "modern_tcn_type3",
            "learning_rate": 2.5e-4,
            "graph_learning_rate": 5.0e-4,
            "weight_decay": 0.0,
            "batch_size": 16,
            "selection_batch_size": 32,
            "validation_batch_size": 32,
            "export_batch_size": 32,
            "num_workers": 0,
            "max_epochs": 100,
            "patience": 10,
            "min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": True,
            "seed": int(seed),
            "selection_split": "test",
            "selection_metric": (
                "mean_all_horizon_cumulative_log_change_mae"
            ),
            "selection_horizons": [int(value) for value in horizons],
            "loss": {
                "type": "cumulative_log_change_mae",
                "bps_scale": 10000.0,
            },
        },
    }


def _resolve_head_schedule(
    graph_heads: int | Sequence[int],
    *,
    block_count: int,
) -> tuple[int, ...]:
    if isinstance(graph_heads, int):
        values = tuple([int(graph_heads)] * int(block_count))
    else:
        values = tuple(int(value) for value in graph_heads)
        if len(values) != int(block_count):
            raise ValueError(
                f"Graph-head schedule has {len(values)} values; "
                f"expected {block_count}."
            )
    if any(value <= 0 for value in values):
        raise ValueError("Every graph-head count must be positive.")
    return values


def _make_spec(
    *,
    base: Mapping[str, Any],
    temporal_family: TemporalFamily,
    num_transformer_blocks: int,
    graph_family: GraphFamily,
    prior_type: PriorType,
    graph_heads: int | Sequence[int],
    modern_tcn_graph_dim_per_head: int,
    transformer_graph_dim_per_head: int,
) -> Round2RunSpec:

    modern_tcn_d_model = int(
        base["model"]["temporal_stack"]["modern_tcn"]["d_model"]
    )

    transformer_d_model = int(
        base["model"]["temporal_stack"]["transformer"]["d_model"]
    )
    if temporal_family == "modern_tcn_transformer":
        block_count = 1 + int(num_transformer_blocks)
        temporal_tag = f"mtg_t{int(num_transformer_blocks)}"
        temporal_label = (
            "ModernTCN first block + "
            f"{int(num_transformer_blocks)} Transformer block(s)"
        )
        block_dims = (
            modern_tcn_d_model,
            *(
                [transformer_d_model]
                * int(num_transformer_blocks)
            ),
        )
    elif temporal_family == "transformer_only":
        block_count = int(num_transformer_blocks)
        temporal_tag = f"tr{block_count}"
        temporal_label = f"{block_count} Transformer ST block(s)"
        block_dims = tuple(
            [transformer_d_model] * block_count
        )
    else:
        raise ValueError(f"Unsupported temporal family {temporal_family!r}.")
    if block_count < 2:
        raise ValueError("Round 2 requires at least two ST blocks.")

    heads = _resolve_head_schedule(graph_heads, block_count=block_count)
    hidden_dims = tuple(
        int(heads[index])
        * int(
            modern_tcn_graph_dim_per_head
            if temporal_family == "modern_tcn_transformer" and index == 0
            else transformer_graph_dim_per_head
        )
        for index in range(block_count)
    )
    activations = tuple(
        ["softmax"] * (block_count - 1) + ["sparsemax"]
    )

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
    values["model"]["block_d_models"] = list(block_dims)

    graph_tag = "dynamic" if graph_family == "dynamic_only" else (
        f"{prior_type}_state"
    )
    prior_tag = ""
    if graph_family == "prior_state":
        prior_tag = f"_ps{_float_tag(values['model']['prior']['scale'])}"
    head_tag = "g" + "-".join(str(value) for value in heads)
    run_name = f"r2_{temporal_tag}_{graph_tag}_{head_tag}{prior_tag}"
    label = f"{temporal_label} — " + (
        "dynamic-only graphs"
        if graph_family == "dynamic_only"
        else f"{prior_type} prior + dynamic graphs + state pathway"
    )

    return Round2RunSpec(
        run_name=run_name,
        label=label,
        temporal_family=temporal_family,
        num_transformer_blocks=int(num_transformer_blocks),
        num_st_blocks=block_count,
        graph_family=graph_family,
        prior_type=("none" if graph_family == "dynamic_only" else prior_type),
        graph_heads_per_block=heads,
        graph_hidden_dims_per_block=hidden_dims,
        graph_activations_per_block=activations,
        config=values,
    )


def make_round2_specs(
    *,
    prior_type: PriorType = "sector",
    graph_heads: int | Mapping[int, Sequence[int]] = 1,
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    alpha_initial: float = 0.25,
    beta_initial: float = 0.5,
    seed: int = 42,
    transformer_d_model: int = 96,
    transformer_num_layers: int = 1,
    transformer_num_heads: int = 4,
    transformer_feedforward_multiplier: int = 2,
    transformer_dropout: float = 0.0,
    modern_tcn_graph_dim_per_head: int = 32,
    transformer_graph_dim_per_head: int = 96,
) -> tuple[Round2RunSpec, ...]:
    """Return six temporal stacks for each of two graph families.

    ``graph_heads`` may be one integer, repeated in every block, or a mapping
    from total ST depth to an explicit schedule.  The notebook defaults to one
    head in every block, but a later schedule such as ``{4: (6,6,6,1)}`` is a
    direct configuration change rather than a code edit.
    """

    if prior_type not in {"sector", "correlation", "none"}:
        raise ValueError("prior_type must be sector, correlation, or none.")

    base = _base_config(
        context_length=context_length,
        stride=stride,
        horizons=horizons,
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
    )

    temporal_definitions: list[tuple[TemporalFamily, int]] = [
        ("modern_tcn_transformer", 1),
        ("modern_tcn_transformer", 2),
        ("modern_tcn_transformer", 3),
        ("transformer_only", 2),
        ("transformer_only", 3),
        ("transformer_only", 4),
    ]
    result: list[Round2RunSpec] = []
    for graph_family in ("dynamic_only", "prior_state"):
        for temporal_family, transformer_blocks in temporal_definitions:
            total_blocks = transformer_blocks + (
                1 if temporal_family == "modern_tcn_transformer" else 0
            )
            if isinstance(graph_heads, Mapping):
                schedule: int | Sequence[int] = graph_heads.get(
                    total_blocks,
                    1,
                )
            else:
                schedule = graph_heads
            result.append(
                _make_spec(
                    base=base,
                    temporal_family=temporal_family,
                    num_transformer_blocks=transformer_blocks,
                    graph_family=graph_family,
                    prior_type=prior_type,
                    graph_heads=schedule,
                    modern_tcn_graph_dim_per_head=(
                        modern_tcn_graph_dim_per_head
                    ),
                    transformer_graph_dim_per_head=(
                        transformer_graph_dim_per_head
                    ),
                )
            )
    if len(result) != 12 or len({spec.run_name for spec in result}) != 12:
        raise AssertionError("Expected twelve unique Round-2 specifications.")
    return tuple(result)


def save_specs(path: str | Path, specs: Sequence[Round2RunSpec]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_run_config(path: str | Path, spec: Round2RunSpec) -> Path:
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
        raise TypeError(f"Expected a JSON object in {path}.")
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
    specs: Sequence[Round2RunSpec],
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
        selected = history.loc[pd.to_numeric(history["epoch"]) == best_epoch]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one selected row for {spec.run_name}; "
                f"found {len(selected)}."
            )
        history_row = selected.iloc[0]
        row: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Temporal family": spec.temporal_family,
            "Transformer blocks": spec.num_transformer_blocks,
            "ST blocks": spec.num_st_blocks,
            "Graph family": spec.graph_family,
            "Prior": spec.prior_type,
            "Graph heads": str(spec.graph_heads_per_block),
            "Graph activations": str(spec.graph_activations_per_block),
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Trainable parameters": int(metadata.get("trainable_parameters", 0)),
            "Backbone parameters": int(
                metadata.get("backbone_trainable_parameters", 0)
            ),
            "Graph parameters": int(
                metadata.get("graph_trainable_parameters", 0)
            ),
            "Mean test Log MAE": float(history_row["selection_score"]),
            "Run directory": str(directory),
        }
        for horizon in spec.config["data"]["horizons"]:
            row[f"Log MAE — {int(horizon)} min"] = float(
                history_row[
                    f"test_cumulative_log_change_mae_h{int(horizon)}"
                ]
            )
        for block_index in range(spec.num_st_blocks):
            row[f"Block {block_index} alpha"] = history_row.get(
                f"block_{block_index}_alpha"
            )
            row[f"Block {block_index} beta"] = history_row.get(
                f"block_{block_index}_beta"
            )
            row[f"Block {block_index} selected entropy"] = history_row.get(
                f"block_{block_index}_selected_entropy"
            )
            row[f"Block {block_index} dynamic entropy"] = history_row.get(
                f"block_{block_index}_dynamic_entropy"
            )
            row[f"Block {block_index} static entropy"] = history_row.get(
                f"block_{block_index}_static_entropy"
            )
        rows.append(row)
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed Round-2 runs: " + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed Round-2 runs were found.")
    return (
        pd.DataFrame(rows)
        .sort_values(["Mean test Log MAE", "Run"])
        .reset_index(drop=True)
    )

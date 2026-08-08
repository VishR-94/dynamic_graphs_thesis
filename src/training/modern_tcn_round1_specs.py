from __future__ import annotations

"""Experiment specifications for the ModernTCN graph Round-1 ladder."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


PriorType = Literal["none", "sector", "correlation"]
Variant = Literal["dynamic_only", "prior_mixture", "prior_mixture_state"]


@dataclass(frozen=True)
class Round1RunSpec:
    run_name: str
    label: str
    variant: Variant
    prior_type: PriorType
    graph_heads: int
    graph_hidden_dim: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "variant": self.variant,
            "prior_type": self.prior_type,
            "graph_heads": int(self.graph_heads),
            "graph_hidden_dim": int(self.graph_hidden_dim),
            "config": self.config,
        }


def _base_config(
    *,
    context_length: int,
    stride: int,
    horizons: Sequence[int],
    prior_type: PriorType,
    prior_scale: float,
    prior_jitter: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "data": {
            "context_length": int(context_length),
            "stride": int(stride),
            "horizons": [int(value) for value in horizons],
            "input_channels": ["open", "high", "low", "close", "volume"],
            "target_channel": "close",
            "input_representation": "raw",
            "normalisation_eps": 1.0e-8,
            "normalisation_clip": False,
        },
        "normalisation": {
            "eps": 1.0e-8,
            "clip": False,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "model": {
            "variant": "dynamic_only",
            "output_representation": "normalised_close",
            "output_head_initialisation": "default",
            "temporal": {
                "type": "modern_tcn",
                "d_model": 32,
                "patch_size": 8,
                "patch_stride": 4,
                "ffn_ratio": 1,
                "num_blocks": 1,
                "large_kernel": 15,
                "small_kernel": 5,
                "dropout": 0.05,
                "head_dropout": 0.0,
                "session_position_encoding": False,
                "modern_tcn": {
                    "patch_size": 8,
                    "patch_stride": 4,
                    "ffn_ratio": 1,
                    "num_blocks": 1,
                    "large_kernel": 15,
                    "small_kernel": 5,
                    "dropout": 0.05,
                    "head_dropout": 0.0,
                },
            },
            "graph": {
                "type": "dynamic",
                "num_heads": 1,
                "num_heads_per_layer": [1],
                "hidden_dim": 32,
                "activation": "softmax",
                "add_self_loops": False,
                "gate_type": "none",
                "initial_alpha": 0.25,
            },
            "spatial": {
                "num_layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
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
        },
        "training": {
            "optimizer": "adam",
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
            "selection_metric": "mean_five_horizon_cumulative_log_change_mae",
            "selection_horizons": [int(value) for value in horizons],
            "loss": {
                "type": "cumulative_log_change_mae",
                "bps_scale": 10000.0,
            },
            "loss_bps_scale": 10000.0,
        },
    }


def _with_variant(
    base: Mapping[str, Any],
    *,
    variant: Variant,
    prior_type: PriorType,
    graph_heads: int,
    graph_hidden_dim: int,
) -> dict[str, Any]:
    values = json.loads(json.dumps(base))
    values["model"]["variant"] = str(variant)
    values["model"]["prior"]["type"] = str(prior_type)
    values["model"]["graph"]["type"] = (
        "dynamic" if variant == "dynamic_only" else "static_dynamic_mixture"
    )
    values["model"]["graph"]["gate_type"] = (
        "none" if variant == "dynamic_only" else "learned_scalar"
    )
    values["model"]["graph"]["num_heads"] = int(graph_heads)
    values["model"]["graph"]["num_heads_per_layer"] = [int(graph_heads)]
    values["model"]["graph"]["hidden_dim"] = int(graph_hidden_dim)
    return values


def make_round1_specs(
    *,
    prior_type: Literal["sector", "correlation"] = "sector",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    seed: int = 42,
) -> tuple[Round1RunSpec, ...]:
    if prior_type not in {"sector", "correlation"}:
        raise ValueError("prior_type must be 'sector' or 'correlation'.")
    base = _base_config(
        context_length=context_length,
        stride=stride,
        horizons=horizons,
        prior_type=prior_type,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        seed=seed,
    )
    suffix = f"c{int(context_length)}_s{int(stride)}"
    prior_tag = "sector" if prior_type == "sector" else "abscorr"
    return (
        Round1RunSpec(
            run_name=f"r1_control_dynamic_g1_{suffix}",
            label="R1-A — exact ModernTCN dynamic-only control",
            variant="dynamic_only",
            prior_type="none",
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="dynamic_only",
                prior_type="none",
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
        Round1RunSpec(
            run_name=f"r1_{prior_tag}_static_dynamic_g1_{suffix}",
            label="R1-B — prior-initialised static + dynamic graph",
            variant="prior_mixture",
            prior_type=prior_type,
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="prior_mixture",
                prior_type=prior_type,
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
        Round1RunSpec(
            run_name=f"r1_{prior_tag}_static_dynamic_state_g1_{suffix}",
            label=(
                "R1-C — prior/static/dynamic graph with state-aware scorer "
                "and spatial values"
            ),
            variant="prior_mixture_state",
            prior_type=prior_type,
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="prior_mixture_state",
                prior_type=prior_type,
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
    )


def make_six_head_ablation_spec(
    winner: Round1RunSpec,
    *,
    graph_heads: int = 6,
    per_head_dim: int = 32,
) -> Round1RunSpec:
    graph_hidden_dim = int(graph_heads) * int(per_head_dim)
    values = _with_variant(
        winner.config,
        variant=winner.variant,
        prior_type=winner.prior_type,
        graph_heads=graph_heads,
        graph_hidden_dim=graph_hidden_dim,
    )
    return Round1RunSpec(
        run_name=f"{winner.run_name}_g{graph_heads}_gh{graph_hidden_dim}",
        label=(
            f"Round-1 winner with {graph_heads} graph heads "
            f"({per_head_dim} dimensions per head)"
        ),
        variant=winner.variant,
        prior_type=winner.prior_type,
        graph_heads=int(graph_heads),
        graph_hidden_dim=graph_hidden_dim,
        config=values,
    )


def save_specs(path: str | Path, specs: Sequence[Round1RunSpec]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_run_config(path: str | Path, spec: Round1RunSpec) -> Path:
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
    specs: Sequence[Round1RunSpec],
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
                f"Expected one selected history row for {spec.run_name}; "
                f"found {len(selected)}."
            )
        row = selected.iloc[0]
        result: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Variant": spec.variant,
            "Prior": spec.prior_type,
            "Graph heads": spec.graph_heads,
            "Graph hidden dim": spec.graph_hidden_dim,
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Mean test Log MAE": float(row["selection_score"]),
            "Alpha": row.get("block_0_alpha"),
            "Beta": row.get("block_0_beta"),
            "Selected graph entropy": row.get("block_0_selected_entropy"),
            "Static graph entropy": row.get("block_0_static_entropy"),
            "Dynamic graph entropy": row.get("block_0_dynamic_entropy"),
            "Run directory": str(directory),
        }
        for horizon in spec.config["data"]["horizons"]:
            result[f"Log MAE — {int(horizon)} min"] = float(
                row[f"test_cumulative_log_change_mae_h{int(horizon)}"]
            )
        rows.append(result)
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed Round-1 runs: " + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed Round-1 runs were found.")
    return (
        pd.DataFrame(rows)
        .sort_values(["Mean test Log MAE", "Run"])
        .reset_index(drop=True)
    )

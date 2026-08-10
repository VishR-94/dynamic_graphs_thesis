from __future__ import annotations

"""Immutable specifications for the 12-run dense Transformer depth sweep."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.training.dense_parallel_graph_specs import inverse_reference_weights


DEFAULT_REFERENCE_MAE: tuple[float, ...] = (
    0.00036854,
    0.00078591,
    0.00132230,
    0.00183974,
    0.00255599,
)


@dataclass(frozen=True)
class DenseTransformerProfile:
    profile_id: str
    label: str
    d_model: int
    temporal_heads: int
    temporal_layers: int
    feedforward_multiplier: int
    early_graph_heads: int
    final_graph_heads: int
    early_graph_hidden_dim: int
    final_graph_hidden_dim: int

    def graph_heads(self, depth: int) -> tuple[int, ...]:
        if int(depth) <= 0:
            raise ValueError("depth must be positive.")
        if int(depth) == 1:
            return (int(self.final_graph_heads),)
        return (
            *([int(self.early_graph_heads)] * (int(depth) - 1)),
            int(self.final_graph_heads),
        )

    def graph_hidden_dims(self, depth: int) -> tuple[int, ...]:
        if int(depth) <= 0:
            raise ValueError("depth must be positive.")
        if int(depth) == 1:
            return (int(self.final_graph_hidden_dim),)
        return (
            *([int(self.early_graph_hidden_dim)] * (int(depth) - 1)),
            int(self.final_graph_hidden_dim),
        )


DEFAULT_PROFILES: tuple[DenseTransformerProfile, ...] = (
    DenseTransformerProfile(
        profile_id="d64_t4_g1",
        label=(
            "D64, four temporal heads, one graph head in every block"
        ),
        d_model=64,
        temporal_heads=4,
        temporal_layers=1,
        feedforward_multiplier=2,
        early_graph_heads=1,
        final_graph_heads=1,
        early_graph_hidden_dim=64,
        final_graph_hidden_dim=64,
    ),
    DenseTransformerProfile(
        profile_id="d96_t6_g2to1",
        label=(
            "D96, six temporal heads, two early graph heads and one final head"
        ),
        d_model=96,
        temporal_heads=6,
        temporal_layers=1,
        feedforward_multiplier=2,
        early_graph_heads=2,
        final_graph_heads=1,
        early_graph_hidden_dim=96,
        final_graph_hidden_dim=96,
    ),
    DenseTransformerProfile(
        profile_id="d96_v2like_t4_g6to1",
        label=(
            "D96 V2-like width/head schedule: four temporal heads, "
            "six early graph heads and one final head"
        ),
        d_model=96,
        # The exact Dimitri/BaseDyGraph-V2 configuration used in this project
        # has nhead=4, not 8.  The graph-head triplet resolves to [6,...,1].
        temporal_heads=4,
        temporal_layers=1,
        feedforward_multiplier=2,
        early_graph_heads=6,
        final_graph_heads=1,
        # Match the V2 spatial-width triplet: 192 in early/internal blocks and
        # 96 in the final sparsemax block.
        early_graph_hidden_dim=192,
        final_graph_hidden_dim=96,
    ),
)


@dataclass(frozen=True)
class DenseTransformerDepthRunSpec:
    run_name: str
    label: str
    profile_id: str
    depth: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "profile_id": self.profile_id,
            "depth": int(self.depth),
            "config": self.config,
        }


def _hash(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def make_dense_transformer_depth_specs(
    *,
    profiles: Sequence[DenseTransformerProfile] = DEFAULT_PROFILES,
    depths: Sequence[int] = (1, 2, 3, 4),
    context_length: int = 60,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    dense_prefix_outer_stride: int = 15,
    export_stride: int = 15,
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
    position_embedding: bool = False,
    transformer_dropout: float = 0.0,
    spatial_feedforward_multiplier: int = 2,
    spatial_dropout: float = 0.0,
    backbone_learning_rate: float = 2.5e-4,
    graph_learning_rate: float = 5.0e-4,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    max_epochs: int = 100,
    patience: int = 10,
    train_batch_size: int = 1,
    selection_batch_size: int = 2,
    export_batch_size: int = 2,
    prefix_graph_sample_windows: int = 2,
    mixed_precision: bool = True,
    gradient_clip_norm: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DenseTransformerDepthRunSpec, ...]:
    horizons_tuple = tuple(int(value) for value in horizons)
    if tuple(sorted(set(horizons_tuple))) != horizons_tuple:
        raise ValueError("horizons must be unique and increasing.")
    if tuple(int(value) for value in depths) != tuple(sorted(set(depths))):
        raise ValueError("depths must be unique and increasing.")
    if any(int(value) <= 0 for value in depths):
        raise ValueError("Every depth must be positive.")
    if len(reference_mae) != len(horizons_tuple):
        raise ValueError("reference_mae must match horizons.")
    weights = inverse_reference_weights(reference_mae)

    specs: list[DenseTransformerDepthRunSpec] = []
    for profile in profiles:
        if int(profile.d_model) % int(profile.temporal_heads):
            raise ValueError(
                f"Profile {profile.profile_id}: d_model must be divisible by "
                "temporal heads."
            )
        for depth in depths:
            depth = int(depth)
            graph_heads = profile.graph_heads(depth)
            graph_hidden_dims = profile.graph_hidden_dims(depth)
            activations = (("softmax",) if depth == 1 else tuple(["softmax"] * (depth - 1) + ["sparsemax"]))
            for index, (heads, hidden) in enumerate(
                zip(graph_heads, graph_hidden_dims, strict=True)
            ):
                if hidden % heads:
                    raise ValueError(
                        f"Profile {profile.profile_id}, depth {depth}, block "
                        f"{index}: graph width {hidden} is not divisible by "
                        f"heads {heads}."
                    )

            config: dict[str, Any] = {
                "model_family": "dense_transformer_depth_sweep",
                "experiment_family": "dense_transformer_depth_sweep",
                "do_not_report": True,
                "test_set_contaminated": True,
                "data": {
                    "context_length": int(context_length),
                    "horizons": list(horizons_tuple),
                    "dense_prefix_outer_stride": int(dense_prefix_outer_stride),
                    "export_stride": int(export_stride),
                    "input_channels": [str(value) for value in input_channels],
                    "target_channel": str(target_channel),
                    "input_representation": "raw",
                },
                "normalisation": {
                    "eps": 1.0e-8,
                    "clip": False,
                    "clip_min": -5.0,
                    "clip_max": 5.0,
                },
                "model": {
                    "num_nodes": 93,
                    "num_st_blocks": depth,
                    "variant": "uniform_static_dynamic_state",
                    "temporal": {
                        "type": "transformer",
                        "d_model": int(profile.d_model),
                        "num_layers": int(profile.temporal_layers),
                        "num_heads": int(profile.temporal_heads),
                        "feedforward_multiplier": int(
                            profile.feedforward_multiplier
                        ),
                        "dropout": float(transformer_dropout),
                        "position_embedding": bool(position_embedding),
                    },
                    "graph": {
                        "type": "static_dynamic_mixture",
                        "num_heads": int(graph_heads[-1]),
                        "num_heads_per_block": list(graph_heads),
                        "num_heads_per_layer": list(graph_heads),
                        "hidden_dim": int(graph_hidden_dims[-1]),
                        "hidden_dims_per_block": list(graph_hidden_dims),
                        "activations_per_block": list(activations),
                        "activation": str(activations[-1]),
                        "add_self_loops": False,
                        "initial_alpha": float(alpha_initial),
                    },
                    "spatial": {
                        "num_layers": 1,
                        "feedforward_multiplier": int(
                            spatial_feedforward_multiplier
                        ),
                        "dropout": float(spatial_dropout),
                        "gate_type": "learned_scalar",
                        "initial_beta": float(beta_initial),
                    },
                    "prior": {
                        "type": "uniform",
                        "static_logits": "zeros",
                        "dynamic_logits": "zeros_at_initialisation",
                        "diagonal": "excluded",
                    },
                    "graph_regularisation": {
                        "graph_reg_layer": -1,
                        "graph_reg_warmup_epochs": 0,
                        "graph_entropy_reg": 0.0,
                        "graph_target_entropy": None,
                        "graph_target_entropy_reg": 0.0,
                        "graph_temporal_smooth_reg": 0.0,
                    },
                    "output_representation": "normalised_close",
                    "output_head_initialisation": "default",
                },
                "training": {
                    "training_style": "dense_prefix",
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
                    "selection_horizons": list(horizons_tuple),
                    "selection_metric": (
                        "unweighted_mean_five_horizon_"
                        "cumulative_log_change_mae"
                    ),
                    "loss": {
                        "type": "cumulative_log_change_mae",
                        "bps_scale": 10000.0,
                        "horizon_weighting": "inverse_reference_mae",
                        "horizon_reference_mae": [
                            float(value) for value in reference_mae
                        ],
                        "horizon_weights": [float(value) for value in weights],
                    },
                    "prefix_graph_sample_windows": int(
                        prefix_graph_sample_windows
                    ),
                    "optimisation_profile": (
                        "delayed_decay_inverse_reference_dense_transformer_depth"
                    ),
                },
            }
            signature = _hash(config)
            run_name = (
                f"dense_tr_{profile.profile_id}_st{depth}_"
                f"a{_float_tag(alpha_initial)}_b{_float_tag(beta_initial)}_"
                f"uniformstatic_{signature}"
            )
            label = f"{profile.label}; {depth} ST block(s)"
            specs.append(
                DenseTransformerDepthRunSpec(
                    run_name=run_name,
                    label=label,
                    profile_id=profile.profile_id,
                    depth=depth,
                    config=config,
                )
            )

    expected = len(tuple(profiles)) * len(tuple(depths))
    if len(specs) != expected:
        raise AssertionError(f"Expected {expected} specs; generated {len(specs)}.")
    if len({spec.run_name for spec in specs}) != len(specs):
        raise AssertionError("Duplicate run names were generated.")
    return tuple(specs)


def save_specs(
    path: str | Path,
    specs: Sequence[DenseTransformerDepthRunSpec],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_config(
    path: str | Path,
    spec: DenseTransformerDepthRunSpec,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_is_complete(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    metadata_path = directory / "run_metadata.json"
    checkpoint_path = directory / "best_checkpoint.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        return False
    return load_json(metadata_path).get("status") == "completed"


def summarise_runs(
    output_root: str | Path,
    specs: Sequence[DenseTransformerDepthRunSpec],
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        run_dir = Path(output_root) / spec.run_name
        metadata_path = run_dir / "run_metadata.json"
        history_path = run_dir / "history.csv"
        if not metadata_path.is_file() or not history_path.is_file():
            missing.append(spec.run_name)
            continue
        metadata = load_json(metadata_path)
        history = pd.read_csv(history_path)
        if metadata.get("status") != "completed" or history.empty:
            missing.append(spec.run_name)
            continue
        best_epoch = int(metadata["best_epoch"])
        selected = history.loc[history["epoch"] == best_epoch]
        if len(selected) != 1:
            raise AssertionError(
                f"Expected one best-epoch row for {spec.run_name}; "
                f"found {len(selected)}."
            )
        best = selected.iloc[0]
        config = spec.config
        graph = config["model"]["graph"]
        temporal = config["model"]["temporal"]
        row: dict[str, Any] = {
            "Run": spec.run_name,
            "Profile": spec.profile_id,
            "ST blocks": int(spec.depth),
            "D": int(temporal["d_model"]),
            "Temporal heads": int(temporal["num_heads"]),
            "Transformer layers per ST block": int(temporal["num_layers"]),
            "Graph heads by block": tuple(graph["num_heads_per_block"]),
            "Graph widths by block": tuple(graph["hidden_dims_per_block"]),
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Mean test Log MAE": float(best["selection_score"]),
            "Trainable parameters": int(metadata["trainable_parameters"]),
        }
        for horizon in config["data"]["horizons"]:
            row[f"Log MAE — {int(horizon)} min"] = float(
                best[f"test_cumulative_log_change_mae_h{int(horizon)}"]
            )
        for block in range(int(spec.depth)):
            row[f"Alpha block {block}"] = float(best[f"block_{block}_alpha"])
            row[f"Beta block {block}"] = float(best[f"block_{block}_beta"])
            row[f"Entropy block {block}"] = float(
                best[f"block_{block}_selected_entropy"]
            )
        rows.append(row)

    if require_all and missing:
        raise RuntimeError(
            "Missing or incomplete depth-sweep runs:\n" + "\n".join(missing)
        )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(["Mean test Log MAE", "Run"]).reset_index(drop=True)

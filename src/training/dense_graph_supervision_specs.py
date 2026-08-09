from __future__ import annotations

"""Specifications for the four dense-supervision graph diagnostics."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd


DenseControlFamily = Literal["basedygraph_v1", "modern_tcn"]
DenseControlVariant = Literal[
    "token_to_price_dynamic",
    "price_to_price_dynamic",
    "modern_tcn_dynamic_state",
    "modern_tcn_random_static_dynamic_state",
]


@dataclass(frozen=True)
class DenseGraphSupervisionRunSpec:
    run_name: str
    label: str
    family: DenseControlFamily
    variant: DenseControlVariant
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "family": self.family,
            "variant": self.variant,
            "config": self.config,
        }


def _base_training(
    *,
    batch_size: int,
    selection_batch_size: int,
    export_batch_size: int,
    backbone_learning_rate: float,
    graph_learning_rate: float,
    max_epochs: int,
    patience: int,
    decay_start_epoch: int,
    decay_factor: float,
    mixed_precision: bool,
    gradient_clip_norm: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "optimizer": "adam",
        "parameter_grouping": "split",
        "scheduler": "modern_tcn_type3_delayed",
        "scheduler_decay_start_epoch": int(decay_start_epoch),
        "scheduler_decay_factor": float(decay_factor),
        "learning_rate": float(backbone_learning_rate),
        "graph_learning_rate": float(graph_learning_rate),
        "weight_decay": 0.0,
        "batch_size": int(batch_size),
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
        "selection_direction": "minimise",
        "selection_metric": "forecast_origin_h1_cumulative_log_change_mae",
        "loss": {
            "type": "dense_one_step_cumulative_log_change_mae",
            "bps_scale": 10000.0,
        },
    }


def _short_config_hash(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]



def make_dense_graph_supervision_specs(
    *,
    context_length: int = 60,
    alignment_horizons: Sequence[int] = (1, 5, 15, 30, 60),
    export_stride: int = 15,
    modern_tcn_training_stride: int = 1,
    d_model: int = 96,
    temporal_num_layers: int = 1,
    temporal_num_heads: int = 4,
    feedforward_multiplier: int = 2,
    graph_num_heads: int = 1,
    graph_hidden_dim: int = 96,
    num_st_blocks: int = 4,
    max_epochs: int = 100,
    patience: int = 10,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    seed: int = 42,
) -> tuple[DenseGraphSupervisionRunSpec, ...]:
    """Return the two v1 and two one-block ModernTCN diagnostics."""

    horizons = tuple(int(value) for value in alignment_horizons)
    if horizons != tuple(sorted(set(horizons))) or not horizons:
        raise ValueError("alignment_horizons must be unique and increasing.")
    if horizons[0] != 1:
        raise ValueError("alignment_horizons must begin at one minute.")
    if int(context_length) <= 0 or int(export_stride) <= 0:
        raise ValueError("context_length and export_stride must be positive.")
    if int(modern_tcn_training_stride) <= 0:
        raise ValueError("modern_tcn_training_stride must be positive.")

    basedy_training = _base_training(
        batch_size=2,
        selection_batch_size=2,
        export_batch_size=2,
        backbone_learning_rate=2.5e-4,
        graph_learning_rate=5.0e-4,
        max_epochs=max_epochs,
        patience=patience,
        decay_start_epoch=decay_start_epoch,
        decay_factor=decay_factor,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        seed=seed,
    )
    modern_training = _base_training(
        batch_size=16,
        selection_batch_size=32,
        export_batch_size=32,
        backbone_learning_rate=2.5e-4,
        graph_learning_rate=5.0e-4,
        max_epochs=max_epochs,
        patience=patience,
        decay_start_epoch=decay_start_epoch,
        decay_factor=decay_factor,
        mixed_precision=True,
        gradient_clip_norm=1.0,
        seed=seed,
    )
    modern_training["one_step_training_stride"] = int(
        modern_tcn_training_stride
    )

    basedy_architecture = {
        "d_model": int(d_model),
        "temporal_num_layers": int(temporal_num_layers),
        "temporal_num_heads": int(temporal_num_heads),
        "spatial_num_layers": 1,
        "feedforward_multiplier": int(feedforward_multiplier),
        "graph_num_heads": int(graph_num_heads),
        "graph_hidden_dim": int(graph_hidden_dim),
        "num_st_blocks": int(num_st_blocks),
        "dropout": 0.0,
        "spatial_dropout": 0.0,
        "spatial_module_type": "dynamic_graph",
        "spatial_value": "hidden",
        "graph_activation": "softmax",
        "use_node_embedding": True,
        "use_state_pair_bias": False,
        "add_self_loops": False,
        "symmetric_graph": False,
        "st_block_post_norm": True,
    }

    specs: list[DenseGraphSupervisionRunSpec] = []
    for input_mode, variant, label, input_channels in (
        (
            "token",
            "token_to_price_dynamic",
            "BaseDyGraph v1 dense one-step — coarse s1 input, direct price output",
            ["open", "high", "low", "close", "volume"],
        ),
        (
            "continuous",
            "price_to_price_dynamic",
            "BaseDyGraph v1 dense one-step — OHLCV input, direct price output",
            ["open", "high", "low", "close", "volume"],
        ),
    ):
        config = {
            "model_family": "dense_graph_supervision_control",
            "experiment_family": "basedygraph_v1",
            "variant": variant,
            "data": {
                "context_length": int(context_length),
                "alignment_horizons": list(horizons),
                "horizons": [1],
                "export_stride": int(export_stride),
                "input_mode": input_mode,
                "input_channels": input_channels,
                "target_channel": "close",
                "s1_vocabulary_size": 1024,
                "s1_id_space": "kronos_original",
                "normalisation": "context-only per asset/channel",
            },
            "model": {
                "official_basedygraph_v1": dict(basedy_architecture),
                # Analysis mirror used by Graph Hub.  The constructor reads
                # official_basedygraph_v1; this mapping only describes the
                # saved continuous-price graph artefacts.
                "temporal": {
                    "type": "transformer",
                    "d_model": int(d_model),
                    "num_layers": int(temporal_num_layers),
                    "num_heads": int(temporal_num_heads),
                    "feedforward_multiplier": int(feedforward_multiplier),
                    "dropout": 0.0,
                },
                "graph": {
                    "type": "dynamic",
                    "num_heads": int(graph_num_heads),
                    "num_heads_per_layer": [int(graph_num_heads)]
                    * int(num_st_blocks),
                    "hidden_dim": int(graph_hidden_dim),
                    "activation": "softmax",
                    "add_self_loops": False,
                },
                "spatial": {
                    "num_layers": 1,
                    "feedforward_multiplier": int(feedforward_multiplier),
                    "dropout": 0.0,
                    "gate_type": "none",
                },
                "graph_regularisation": {
                    "graph_entropy_reg": 0.0,
                    "graph_target_entropy": None,
                    "graph_target_entropy_reg": 0.0,
                    "graph_temporal_smooth_reg": 0.0,
                },
                "output_representation": "normalised_close",
                "output_head_initialisation": "default",
            },
            # Compatibility mirror for Graph Hub descriptions.  The actual
            # constructor reads model.official_basedygraph_v1.
            "models": {
                "dynamic_graph": {
                    "d_model": int(d_model),
                    "num_st_blocks": int(num_st_blocks),
                    "temporal": {
                        "type": "transformer",
                        "d_model": int(d_model),
                        "num_layers": int(temporal_num_layers),
                        "num_heads": int(temporal_num_heads),
                        "feedforward_multiplier": int(feedforward_multiplier),
                        "dropout": 0.0,
                    },
                    "graph": {
                        "type": "dynamic",
                        "num_heads": int(graph_num_heads),
                        "num_heads_per_layer": [int(graph_num_heads)]
                        * int(num_st_blocks),
                        "hidden_dim": int(graph_hidden_dim),
                        "hidden_dims_per_layer": [int(graph_hidden_dim)]
                        * int(num_st_blocks),
                        "activation": "softmax",
                        "activations_per_layer": ["softmax"]
                        * int(num_st_blocks),
                        "add_self_loops": False,
                    },
                    "heads": {
                        "evaluation_horizons": [1],
                        "prediction_length": 1,
                    },
                }
            },
            "training": dict(basedy_training),
        }
        run_name = (
            "dense_bdgv1_"
            + ("token_in_price_out" if input_mode == "token" else "price_in_price_out")
            + f"_dynamic_d{int(d_model)}_st{int(num_st_blocks)}_g{int(graph_num_heads)}"
            + f"_c{int(context_length)}_s{int(export_stride)}"
            + f"_cfg{_short_config_hash(config)}"
        )
        specs.append(
            DenseGraphSupervisionRunSpec(
                run_name=run_name,
                label=label,
                family="basedygraph_v1",
                variant=variant,  # type: ignore[arg-type]
                config=config,
            )
        )

    modern_base = {
        "model_family": "dense_graph_supervision_control",
        "experiment_family": "modern_tcn",
        "data": {
            "context_length": int(context_length),
            "horizons": [1],
            "alignment_horizons": list(horizons),
            "export_stride": int(export_stride),
            "input_channels": ["close"],
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
            "output_representation": "cumulative_log_change",
            "output_head_initialisation": "zero",
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
                "num_heads": 1,
                "num_heads_per_layer": [1],
                "hidden_dim": 32,
                "activation": "softmax",
                "add_self_loops": False,
                "initial_alpha": 0.5,
            },
            "spatial": {
                "num_layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
            },
            "prior": {
                "type": "none",
                "scale": 4.0,
                "jitter": 0.02,
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
        "training": dict(modern_training),
    }

    dynamic = json.loads(json.dumps(modern_base))
    dynamic["variant"] = "modern_tcn_dynamic_state"
    dynamic["model"]["variant"] = "dynamic_state"
    dynamic["model"]["graph"].update(
        {"type": "dynamic", "gate_type": "none"}
    )

    random_static = json.loads(json.dumps(modern_base))
    random_static["variant"] = "modern_tcn_random_static_dynamic_state"
    random_static["model"]["variant"] = "random_static_dynamic_state"
    random_static["model"]["graph"].update(
        {"type": "static_dynamic_mixture", "gate_type": "learned_scalar"}
    )
    random_static["model"]["prior"].update(
        {
            "type": "random",
            "description": "trainable random static logits; no sector/correlation prior",
        }
    )

    specs.extend(
        [
            DenseGraphSupervisionRunSpec(
                run_name=(
                    "dense_mtg_close_dynamic_state_g1_h32_"
                    f"c{int(context_length)}_trainstride{int(modern_tcn_training_stride)}"
                    f"_cfg{_short_config_hash(dynamic)}"
                ),
                label=(
                    "One-block ModernTCN dense one-step — state-aware dynamic graph only"
                ),
                family="modern_tcn",
                variant="modern_tcn_dynamic_state",
                config=dynamic,
            ),
            DenseGraphSupervisionRunSpec(
                run_name=(
                    "dense_mtg_close_random_static_dynamic_state_"
                    f"a0p5_b0p5_g1_h32_c{int(context_length)}_"
                    f"trainstride{int(modern_tcn_training_stride)}"
                    f"_cfg{_short_config_hash(random_static)}"
                ),
                label=(
                    "One-block ModernTCN dense one-step — random static + dynamic, state-aware"
                ),
                family="modern_tcn",
                variant="modern_tcn_random_static_dynamic_state",
                config=random_static,
            ),
        ]
    )
    return tuple(specs)


def save_specs(
    path: str | Path,
    specs: Sequence[DenseGraphSupervisionRunSpec],
) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_config(path: str | Path, spec: DenseGraphSupervisionRunSpec) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
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


def config_hash(values: dict[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def summarise_runs(
    output_root: str | Path,
    specs: Sequence[DenseGraphSupervisionRunSpec],
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
        history = pd.read_csv(history_path)
        best_epoch = int(metadata["best_epoch"])
        selected = history.loc[history["epoch"] == best_epoch]
        if len(selected) != 1:
            raise AssertionError(
                f"Expected one best epoch row for {spec.run_name}."
            )
        best = selected.iloc[0]
        rows.append(
            {
                "Run": spec.run_name,
                "Label": spec.label,
                "Family": spec.family,
                "Variant": spec.variant,
                "Status": metadata.get("status"),
                "Best epoch": best_epoch,
                "Epochs completed": int(metadata["epochs_completed"]),
                "Test h1 Log MAE": float(best["test_h1_log_mae"]),
                "Train dense Log MAE": float(best["train_dense_log_mae"]),
                "Final graph entropy": best.get("test_final_graph_entropy"),
                "Final graph effective neighbours": best.get(
                    "test_final_graph_effective_neighbours"
                ),
                "Alpha": best.get("alpha"),
                "Beta": best.get("beta"),
            }
        )
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed runs: " + ", ".join(sorted(missing))
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["Test h1 Log MAE", "Run"]
        ).reset_index(drop=True)
    return result

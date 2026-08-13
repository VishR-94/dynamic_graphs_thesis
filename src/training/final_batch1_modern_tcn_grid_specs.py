from __future__ import annotations

"""Batch-size-one ModernTCN graph grid in continuous and token space.

The grid intentionally holds the selected one-block ModernTCN architecture,
optimiser, learning-rate schedule, graph/state pathways, alpha/beta
initialisation, data splits, context, stride, and public horizons fixed.

Controlled factors
------------------
* representation / objective: continuous weighted price forecasting versus
  coarse-s1 token forecasting;
* static initialisation: exact uniform off-diagonal logits versus the
  training-only unthresholded absolute-correlation prior;
* graph normalisation: softmax versus sparsemax.

Every physical training batch contains one forecast window.  Selection and
export loaders also use batch size one so the saved configurations contain no
ambiguous secondary batch fields.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.training.final_token_v2_specs import make_final_token_v2_specs
from src.training.modern_tcn_final_two_runs_specs import make_final_two_run_specs


ModelSpace = Literal["continuous", "token"]
StaticInitialisation = Literal["uniform", "correlation"]
GraphActivation = Literal["softmax", "sparsemax"]


@dataclass(frozen=True)
class Batch1ModernTCNGridSpec:
    run_name: str
    label: str
    model_space: ModelSpace
    static_initialisation: StaticInitialisation
    graph_activation: GraphActivation
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "model_space": self.model_space,
            "static_initialisation": self.static_initialisation,
            "graph_activation": self.graph_activation,
            "config": self.config,
        }


def _deepcopy_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values))


def _signature(values: Mapping[str, Any]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _set_all_loader_batches_to_one(config: dict[str, Any]) -> None:
    training = config["training"]
    for key in (
        "batch_size",
        "selection_batch_size",
        "validation_batch_size",
        "export_batch_size",
    ):
        if key in training:
            training[key] = 1


def _continuous_reference_config(*, seed: int) -> dict[str, Any]:
    weighted, _ = make_final_two_run_specs(
        prior_type="correlation",
        context_length=60,
        stride=15,
        horizons=(1, 5, 15, 30, 60),
        alpha_initial=0.5,
        beta_initial=0.5,
        prior_scale=4.0,
        prior_jitter=0.02,
        decay_start_epoch=15,
        decay_factor=0.9,
        seed=int(seed),
    )
    values = _deepcopy_mapping(weighted.config)
    values["experiment_family"] = "batch1_modern_tcn_graph_grid"
    values["do_not_report"] = True
    values["test_set_contaminated"] = True
    _set_all_loader_batches_to_one(values)
    return values


def _token_reference_config(*, seed: int) -> dict[str, Any]:
    references = make_final_token_v2_specs(seed=int(seed))
    matches = [
        spec
        for spec in references
        if spec.model_kind == "modern_tcn_token"
    ]
    if len(matches) != 1:
        raise AssertionError(
            "Expected exactly one selected ModernTCN token reference."
        )
    values = _deepcopy_mapping(matches[0].config)
    values["model_family"] = "batch1_modern_tcn_token_graph_grid"
    values["do_not_report"] = True
    values["test_set_contaminated"] = True
    _set_all_loader_batches_to_one(values)

    loss = values["training"]["loss"]
    loss.clear()
    loss.update(
        {
            "type": "coarse_s1_cross_entropy",
            "horizon_weighting": "uniform",
            "dense_origins": False,
        }
    )
    values["training"]["selection_metric"] = (
        "mean_top1_accuracy_over_all_60_future_steps"
    )
    values["training"]["early_stopping_metric"] = (
        "mean_top1_accuracy_over_all_60_future_steps"
    )
    values["training"]["selection_direction"] = "maximise"
    return values


def _configure_continuous_variant(
    base: Mapping[str, Any],
    *,
    initialisation: StaticInitialisation,
    activation: GraphActivation,
    seed: int,
) -> dict[str, Any]:
    values = _deepcopy_mapping(base)
    model = values["model"]
    model["graph"].update(
        {
            "type": "static_dynamic_mixture",
            "activation": str(activation),
            "gate_type": "learned_scalar",
            "initial_alpha": 0.5,
        }
    )
    model["spatial"].update(
        {
            "gate_type": "learned_scalar",
            "initial_beta": 0.5,
        }
    )
    model["graph_regularisation"].update(
        {
            "graph_entropy_reg": 0.0,
            "graph_target_entropy_reg": 0.0,
            "graph_temporal_smooth_reg": 0.0,
        }
    )

    if initialisation == "correlation":
        model["variant"] = "prior_mixture_state"
        model["prior"].update(
            {
                "type": "correlation",
                "scale": 4.0,
                "jitter": 0.02,
                "seed": int(seed),
                "threshold": None,
                "description": (
                    "training-only absolute Close-return correlation; "
                    "diagonal removed; no threshold"
                ),
            }
        )
    elif initialisation == "uniform":
        model["variant"] = "uniform_static_mixture_state"
        model["prior"].update(
            {
                "type": "uniform",
                "scale": 4.0,
                "jitter": 0.0,
                "seed": int(seed),
                "threshold": None,
                "description": (
                    "exact zero trainable static logits; uniform "
                    "off-diagonal adjacency at initialisation"
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported initialisation {initialisation!r}.")

    values["training"]["optimisation_profile"] = (
        "batch1_round1_delayed_decay_parallel_weighted"
    )
    return values


def _configure_token_variant(
    base: Mapping[str, Any],
    *,
    initialisation: StaticInitialisation,
    activation: GraphActivation,
    seed: int,
) -> dict[str, Any]:
    values = _deepcopy_mapping(base)
    model = values["model"]
    model["graph_family"] = "prior_state"
    model["graph"].update(
        {
            "type": "static_dynamic_mixture",
            "num_heads": 1,
            "num_heads_per_block": [1],
            "hidden_dim": 32,
            "hidden_dims_per_block": [32],
            "activation": str(activation),
            "activations_per_block": [str(activation)],
            "initial_alpha": 0.5,
            "add_self_loops": False,
        }
    )
    model["spatial"].update(
        {
            "gate_type": "learned_scalar",
            "initial_beta": 0.5,
        }
    )
    model["graph_regularisation"].update(
        {
            "graph_entropy_reg": 0.0,
            "graph_target_entropy_reg": 0.0,
            "graph_temporal_smooth_reg": 0.0,
        }
    )

    if initialisation == "correlation":
        model["prior"].update(
            {
                "type": "correlation",
                "scale": 4.0,
                "jitter": 0.02,
                "seed": int(seed),
                "threshold": None,
                "description": (
                    "training-only absolute Close-return correlation; "
                    "diagonal removed; no threshold"
                ),
            }
        )
    elif initialisation == "uniform":
        model["prior"].update(
            {
                "type": "uniform",
                "scale": 4.0,
                "jitter": 0.0,
                "seed": int(seed),
                "threshold": None,
                "description": (
                    "exact zero trainable static logits; uniform "
                    "off-diagonal adjacency at initialisation"
                ),
            }
        )
    else:
        raise ValueError(f"Unsupported initialisation {initialisation!r}.")

    # Keep the historical Graph Hub mirror aligned with the real model config.
    mirror = values["models"]["dynamic_graph"]
    mirror["graph"].update(
        {
            "type": "static_dynamic_mixture",
            "num_heads": 1,
            "num_heads_per_block": [1],
            "hidden_dim": 32,
            "hidden_dims_per_block": [32],
            "activation": str(activation),
            "activations_per_block": [str(activation)],
            "initial_alpha": 0.5,
            "add_self_loops": False,
        }
    )
    mirror["prior"] = _deepcopy_mapping(model["prior"])
    mirror["spatial"]["initial_beta"] = 0.5
    return values


def make_final_batch1_modern_tcn_grid_specs(
    *,
    seed: int = 42,
) -> tuple[Batch1ModernTCNGridSpec, ...]:
    continuous_base = _continuous_reference_config(seed=int(seed))
    token_base = _token_reference_config(seed=int(seed))

    result: list[Batch1ModernTCNGridSpec] = []
    ordered_variants = (
        ("uniform", "softmax"),
        ("correlation", "softmax"),
        ("uniform", "sparsemax"),
        ("correlation", "sparsemax"),
    )
    for model_space, base in (
        ("continuous", continuous_base),
        ("token", token_base),
    ):
        for initialisation, activation in ordered_variants:
            if model_space == "continuous":
                config = _configure_continuous_variant(
                    base,
                    initialisation=initialisation,
                    activation=activation,
                    seed=int(seed),
                )
                objective = "weighted_price"
            else:
                config = _configure_token_variant(
                    base,
                    initialisation=initialisation,
                    activation=activation,
                    seed=int(seed),
                )
                objective = "uniform_ce_parallel60"

            signature = _signature(config)
            initial_tag = "uniform" if initialisation == "uniform" else "abscorr"
            prefix = "b1_px" if model_space == "continuous" else "b1_tok"
            run_name = (
                f"{prefix}_mtg_d32_st1_{initial_tag}_{activation}_"
                f"a0p5_b0p5_{objective}_{signature}"
            )
            label = (
                f"Batch-1 {model_space} ModernTCN — "
                f"{initialisation} static + dynamic, {activation}"
            )
            result.append(
                Batch1ModernTCNGridSpec(
                    run_name=run_name,
                    label=label,
                    model_space=model_space,  # type: ignore[arg-type]
                    static_initialisation=initialisation,  # type: ignore[arg-type]
                    graph_activation=activation,  # type: ignore[arg-type]
                    config=config,
                )
            )
    signatures = [_signature(spec.config) for spec in result]
    if len(set(signatures)) != len(signatures):
        raise AssertionError("Batch-1 grid contains duplicate resolved configs.")
    if len(result) != 8:
        raise AssertionError(f"Expected eight run specs; found {len(result)}.")
    return tuple(result)


def save_specs(path: str | Path, specs: Sequence[Batch1ModernTCNGridSpec]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_run_config(path: str | Path, spec: Batch1ModernTCNGridSpec) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return values


def run_is_complete(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    metadata_path = directory / "run_metadata.json"
    checkpoint_path = directory / "best_checkpoint.pt"
    return (
        metadata_path.is_file()
        and checkpoint_path.is_file()
        and load_json(metadata_path).get("status") == "completed"
    )


def summarise_continuous_runs(
    output_root: str | Path,
    specs: Sequence[Batch1ModernTCNGridSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.model_space != "continuous":
            continue
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        metric_path = directory / "best_test_metric_table.csv"
        if not metadata_path.is_file() or not metric_path.is_file():
            continue
        metadata = load_json(metadata_path)
        table = pd.read_csv(metric_path)
        selected = table.loc[
            table["metric"].astype(str).eq("cumulative_log_change_mae")
            & table["channel"].astype(str).str.lower().eq("close")
        ].copy()
        selected["horizon"] = pd.to_numeric(selected["horizon"]).astype(int)
        by_horizon = {
            int(row.horizon): float(row.value)
            for row in selected.itertuples(index=False)
        }
        horizons = tuple(int(value) for value in spec.config["data"]["horizons"])
        rows.append(
            {
                "Run": spec.run_name,
                "Static init": spec.static_initialisation,
                "Activation": spec.graph_activation,
                "Best epoch": int(metadata["best_epoch"]),
                "Mean test Log MAE": float(
                    sum(by_horizon[h] for h in horizons) / len(horizons)
                ),
                "Final alpha": metadata.get("final_alpha"),
                "Final beta": metadata.get("final_beta"),
                **{
                    f"Log MAE — {h} min": by_horizon[h]
                    for h in horizons
                },
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Mean test Log MAE", "Run"]
    ).reset_index(drop=True)


def summarise_token_runs(
    output_root: str | Path,
    specs: Sequence[Batch1ModernTCNGridSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.model_space != "token":
            continue
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        metric_path = directory / "best_test_token_metric_table.csv"
        if not metadata_path.is_file() or not metric_path.is_file():
            continue
        metadata = load_json(metadata_path)
        table = pd.read_csv(metric_path)
        top1 = table.loc[
            table["metric"].astype(str).eq("top1_accuracy")
        ].copy()
        top1["horizon"] = pd.to_numeric(top1["horizon"]).astype(int)
        by_minute = {
            int(row.horizon): float(row.value)
            for row in top1.itertuples(index=False)
        }
        rows.append(
            {
                "Run": spec.run_name,
                "Static init": spec.static_initialisation,
                "Activation": spec.graph_activation,
                "Best epoch": int(metadata["best_epoch"]),
                "Mean test Top-1 — all 60": float(
                    sum(by_minute.values()) / len(by_minute)
                ),
                **{
                    f"Top-1 — {h} min": by_minute[h]
                    for h in (1, 5, 15, 30, 60)
                },
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Mean test Top-1 — all 60", "Run"],
        ascending=[False, True],
    ).reset_index(drop=True)

from __future__ import annotations

"""Final correlation-prior graph-activation ablation specifications.

The selected weighted-parallel one-block ModernTCN configuration is retained
exactly.  The only architectural change is the row-normalisation applied to
both the trainable correlation-initialised static logits and the dynamic Q/K
logits:

* ``sparsemax``;
* ``entmax15`` (1.5-entmax).

Both runs keep the state pathway, learned alpha/beta gates, inverse-reference
horizon weights, delayed learning-rate schedule, and test-set checkpoint
selection used by the selected continuous model.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from src.training.modern_tcn_final_two_runs_specs import (
    DEFAULT_HORIZON_REFERENCE_MAE,
    make_final_two_run_specs,
)
from src.training.modern_tcn_round1_specs import Round1RunSpec


CorrelationActivation = Literal["sparsemax", "entmax15"]
_SUPPORTED_ACTIVATIONS = frozenset({"sparsemax", "entmax15"})


def _deepcopy_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values))


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def make_final_correlation_graph_activation_specs(
    *,
    activations: Sequence[CorrelationActivation] = (
        "sparsemax",
        "entmax15",
    ),
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    alpha_initial: float = 0.5,
    beta_initial: float = 0.5,
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    reference_mae: Sequence[float] = DEFAULT_HORIZON_REFERENCE_MAE,
    seed: int = 42,
) -> tuple[Round1RunSpec, ...]:
    """Return correlation-prior activation controls in the requested order."""

    activation_values = tuple(str(value) for value in activations)
    if not activation_values:
        raise ValueError("activations must not be empty.")
    if len(set(activation_values)) != len(activation_values):
        raise ValueError("activations must be unique.")
    unsupported = sorted(set(activation_values) - _SUPPORTED_ACTIVATIONS)
    if unsupported:
        raise ValueError(
            "Unsupported final correlation activation(s): "
            + ", ".join(unsupported)
        )

    horizons = tuple(int(value) for value in horizons)
    if not horizons or horizons != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be non-empty, unique, and increasing.")

    weighted_reference, _ = make_final_two_run_specs(
        prior_type="correlation",
        context_length=int(context_length),
        stride=int(stride),
        horizons=horizons,
        alpha_initial=float(alpha_initial),
        beta_initial=float(beta_initial),
        prior_scale=float(prior_scale),
        prior_jitter=float(prior_jitter),
        decay_start_epoch=int(decay_start_epoch),
        decay_factor=float(decay_factor),
        reference_mae=tuple(float(value) for value in reference_mae),
        seed=int(seed),
    )

    horizon_tag = "-".join(str(value) for value in horizons)
    shared = (
        f"d32_k1_p8s4_lk15_g1_h32_a{_float_tag(alpha_initial)}_"
        f"b{_float_tag(beta_initial)}_ps{_float_tag(prior_scale)}_"
        f"pj{_float_tag(prior_jitter)}_ds{int(decay_start_epoch)}_"
        f"df{_float_tag(decay_factor)}_c{int(context_length)}_"
        f"s{int(stride)}_h{horizon_tag}"
    )

    specs: list[Round1RunSpec] = []
    for activation in activation_values:
        config = _deepcopy_mapping(weighted_reference.config)

        # Preserve the selected architecture exactly.  The activation is the
        # only value inside the model/training configuration that changes.
        config["model"]["graph"]["activation"] = activation

        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        label_activation = (
            "1.5-entmax" if activation == "entmax15" else "sparsemax"
        )
        specs.append(
            replace(
                weighted_reference,
                run_name=(
                    f"final_abscorr_{activation}_state_{shared}_cfg{digest}"
                ),
                label=(
                    "Weighted parallel ModernTCN — correlation-initialised "
                    f"static + dynamic graph with {label_activation}"
                ),
                variant="prior_mixture_state",
                prior_type="correlation",
                config=config,
                ablation_family="final_correlation_activation_ablation",
            )
        )

    return tuple(specs)


def save_specs(path: str | Path, specs: Sequence[Round1RunSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_config(path: str | Path, spec: Round1RunSpec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected one JSON object in {path}.")
    return values


def run_is_complete(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    metadata = run_dir / "run_metadata.json"
    checkpoint = run_dir / "best_checkpoint.pt"
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
        run_dir = Path(output_root) / spec.run_name
        metadata_path = run_dir / "run_metadata.json"
        metric_path = run_dir / "best_test_metric_table.csv"
        if not metadata_path.is_file() or not metric_path.is_file():
            missing.append(spec.run_name)
            continue

        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            missing.append(spec.run_name)
            continue

        table = pd.read_csv(metric_path)
        selected = table.loc[
            table["metric"].astype(str).eq("cumulative_log_change_mae")
            & table["channel"].astype(str).str.lower().eq("close")
        ].copy()
        selected["horizon"] = pd.to_numeric(
            selected["horizon"], errors="raise"
        ).astype(int)
        by_horizon = {
            int(row.horizon): float(row.value)
            for row in selected.itertuples(index=False)
        }
        horizons = tuple(int(value) for value in spec.config["data"]["horizons"])
        if set(by_horizon) != set(horizons):
            raise ValueError(f"Incomplete test horizons for {spec.run_name}.")

        rows.append(
            {
                "Run": spec.run_name,
                "Label": spec.label,
                "Graph activation": spec.config["model"]["graph"]["activation"],
                "Prior type": spec.prior_type,
                "Best epoch": int(metadata["best_epoch"]),
                "Epochs completed": int(metadata["epochs_completed"]),
                "Mean test Log MAE": float(
                    sum(by_horizon[horizon] for horizon in horizons)
                    / len(horizons)
                ),
                "Final alpha": metadata.get("final_alpha"),
                "Final beta": metadata.get("final_beta"),
                **{
                    f"Log MAE — {horizon} min": by_horizon[horizon]
                    for horizon in horizons
                },
            }
        )

    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed correlation-activation runs: "
            + ", ".join(missing)
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Mean test Log MAE", "Run"]
    ).reset_index(drop=True)

from __future__ import annotations

"""Final one-block sparsemax graph ablation specifications.

The two runs in this module are cloned from the selected weighted-parallel
ModernTCN configuration. Only the graph branch changes:

1. state-aware dynamic-only sparsemax;
2. state-aware random-static + dynamic sparsemax.

Every data, temporal, forecasting-head, weighted-loss, optimiser, scheduler and
selection setting remains inherited from the selected configuration.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.training.modern_tcn_final_two_runs_specs import (
    DEFAULT_HORIZON_REFERENCE_MAE,
    make_final_two_run_specs,
)
from src.training.modern_tcn_round1_specs import Round1RunSpec


def _deepcopy_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values))


def make_final_sparsemax_graph_specs(
    *,
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    alpha_initial: float = 0.5,
    beta_initial: float = 0.5,
    random_static_logit_std: float = 0.02,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    reference_mae: Sequence[float] = DEFAULT_HORIZON_REFERENCE_MAE,
    seed: int = 42,
) -> tuple[Round1RunSpec, Round1RunSpec]:
    """Return dynamic-only then random-static sparsemax specifications."""

    weighted_reference, _ = make_final_two_run_specs(
        prior_type="correlation",
        context_length=int(context_length),
        stride=int(stride),
        horizons=tuple(int(value) for value in horizons),
        alpha_initial=float(alpha_initial),
        beta_initial=float(beta_initial),
        prior_scale=4.0,
        prior_jitter=float(random_static_logit_std),
        decay_start_epoch=int(decay_start_epoch),
        decay_factor=float(decay_factor),
        reference_mae=tuple(float(value) for value in reference_mae),
        seed=int(seed),
    )

    dynamic_config = _deepcopy_mapping(weighted_reference.config)
    dynamic_config["model"]["variant"] = "dynamic_only_state"
    dynamic_config["model"]["graph"].update(
        {
            "type": "dynamic",
            "activation": "sparsemax",
            "gate_type": "none",
        }
    )
    dynamic_config["model"]["prior"]["type"] = "none"
    dynamic_config["training"]["optimisation_profile"] = (
        "round1_delayed_decay_parallel_weighted_sparsemax_dynamic_state"
    )

    random_config = _deepcopy_mapping(weighted_reference.config)
    random_config["model"]["variant"] = "random_static_mixture_state"
    random_config["model"]["graph"].update(
        {
            "type": "static_dynamic_mixture",
            "activation": "sparsemax",
            "gate_type": "learned_scalar",
        }
    )
    random_config["model"]["prior"].update(
        {
            "type": "random",
            "scale": 4.0,
            "jitter": float(random_static_logit_std),
            "seed": int(seed),
            "description": (
                "independent Gaussian trainable static logits; "
                "no sector/correlation information"
            ),
        }
    )
    random_config["training"]["optimisation_profile"] = (
        "round1_delayed_decay_parallel_weighted_sparsemax_random_static_state"
    )

    horizon_tag = "-".join(str(int(value)) for value in horizons)
    shared = (
        f"d32_k1_p8s4_lk15_g1_h32_a{float(alpha_initial):g}_"
        f"b{float(beta_initial):g}_rs{float(random_static_logit_std):g}_"
        f"ds{int(decay_start_epoch)}_df{float(decay_factor):g}_"
        f"c{int(context_length)}_s{int(stride)}_h{horizon_tag}"
    ).replace(".", "p")
    dynamic_hash = hashlib.sha256(
        json.dumps(dynamic_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    random_hash = hashlib.sha256(
        json.dumps(random_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    dynamic = replace(
        weighted_reference,
        run_name=(
            f"final_sparsemax_dynamic_state_{shared}_cfg{dynamic_hash}"
        ),
        label=(
            "Weighted parallel ModernTCN — state-aware dynamic-only sparsemax"
        ),
        variant="dynamic_only_state",
        prior_type="none",
        config=dynamic_config,
        ablation_family="final_sparsemax_graph_ablation",
    )
    random_static = replace(
        weighted_reference,
        run_name=(
            f"final_sparsemax_random_static_state_{shared}_cfg{random_hash}"
        ),
        label=(
            "Weighted parallel ModernTCN — random static + dynamic sparsemax, "
            "state-aware"
        ),
        variant="random_static_mixture_state",
        prior_type="random",
        config=random_config,
        ablation_family="final_sparsemax_graph_ablation",
    )
    return dynamic, random_static


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
        raise TypeError(f"Expected JSON object in {path}.")
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
                "Graph variant": spec.variant,
                "Graph activation": spec.config["model"]["graph"]["activation"],
                "Prior type": spec.prior_type,
                "Best epoch": int(metadata["best_epoch"]),
                "Mean test Log MAE": float(
                    sum(by_horizon[h] for h in horizons) / len(horizons)
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
            "Missing completed sparsemax runs: " + ", ".join(missing)
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Mean test Log MAE", "Run"]
    ).reset_index(drop=True)

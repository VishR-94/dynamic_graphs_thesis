from __future__ import annotations

"""Final ModernTCN graph ablation specifications.

The helper in this module deliberately reuses the winning Round-1 delayed-
decay alpha/beta configuration and changes only the forecast/loss protocol for
these last two diagnostic runs:

1. one-step training followed by autoregressive 60-minute rollout;
2. parallel five-horizon training with horizon-scaled loss.
"""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.training.modern_tcn_round1_specs import (
    Round1RunSpec,
    _float_tag,
    make_alpha_beta_delayed_decay_sweep_specs,
)


DEFAULT_HORIZON_REFERENCE_MAE: tuple[float, ...] = (
    0.00036854,
    0.00078591,
    0.00132230,
    0.00183974,
    0.00255599,
)


def _deepcopy_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values))


def _safe_name_part(values: Sequence[int]) -> str:
    return "-".join(str(int(value)) for value in values)


def _normalised_inverse_reference_weights(
    reference_mae: Sequence[float],
) -> list[float]:
    values = [float(value) for value in reference_mae]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("reference_mae must contain positive values.")
    mean_value = sum(values) / len(values)
    return [mean_value / value for value in values]


def _winner_base_spec(
    *,
    prior_type: str,
    context_length: int,
    stride: int,
    horizons: Sequence[int],
    alpha_initial: float,
    beta_initial: float,
    prior_scale: float,
    prior_jitter: float,
    decay_start_epoch: int,
    decay_factor: float,
    seed: int,
) -> Round1RunSpec:
    specs = make_alpha_beta_delayed_decay_sweep_specs(
        alpha_initials=(float(alpha_initial),),
        beta_initials=(float(beta_initial),),
        prior_type=prior_type,  # type: ignore[arg-type]
        context_length=int(context_length),
        stride=int(stride),
        horizons=tuple(int(value) for value in horizons),
        prior_scale=float(prior_scale),
        prior_jitter=float(prior_jitter),
        decay_start_epoch=int(decay_start_epoch),
        decay_factor=float(decay_factor),
        seed=int(seed),
    )
    if len(specs) != 1:
        raise AssertionError("Expected exactly one winning base specification.")
    return specs[0]


def make_final_two_run_specs(
    *,
    prior_type: str = "correlation",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    alpha_initial: float = 0.5,
    beta_initial: float = 0.5,
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    one_step_training_stride: int = 1,
    reference_mae: Sequence[float] = DEFAULT_HORIZON_REFERENCE_MAE,
    seed: int = 42,
) -> tuple[Round1RunSpec, Round1RunSpec]:
    """Return autoregressive and horizon-weighted parallel run specs."""

    horizons = tuple(int(value) for value in horizons)
    if horizons != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be unique and increasing.")
    if max(horizons) <= 0:
        raise ValueError("horizons must be positive.")
    if int(one_step_training_stride) <= 0:
        raise ValueError("one_step_training_stride must be positive.")
    if len(reference_mae) != len(horizons):
        raise ValueError("reference_mae must contain one value per horizon.")

    base = _winner_base_spec(
        prior_type=str(prior_type),
        context_length=int(context_length),
        stride=int(stride),
        horizons=horizons,
        alpha_initial=float(alpha_initial),
        beta_initial=float(beta_initial),
        prior_scale=float(prior_scale),
        prior_jitter=float(prior_jitter),
        decay_start_epoch=int(decay_start_epoch),
        decay_factor=float(decay_factor),
        seed=int(seed),
    )

    weights = _normalised_inverse_reference_weights(reference_mae)
    horizon_tag = _safe_name_part(horizons)
    prior_tag = "abscorr" if str(prior_type) == "correlation" else str(prior_type)
    shared_suffix = (
        f"{prior_tag}_a{_float_tag(alpha_initial)}_b{_float_tag(beta_initial)}_"
        f"ps{_float_tag(prior_scale)}_pj{_float_tag(prior_jitter)}_"
        f"ds{int(decay_start_epoch)}_df{_float_tag(decay_factor)}_"
        f"c{int(context_length)}_s{int(stride)}_h{horizon_tag}"
    )

    autoregressive_config = _deepcopy_mapping(base.config)
    autoregressive_config["training"]["forecast_strategy"] = "autoregressive"
    autoregressive_config["training"]["one_step_training_stride"] = int(
        one_step_training_stride
    )
    autoregressive_config["training"]["autoregressive_rollout_length"] = int(
        max(horizons)
    )
    autoregressive_config["training"]["selection_metric"] = (
        "autoregressive_mean_five_horizon_cumulative_log_change_mae"
    )
    autoregressive_config["training"]["optimisation_profile"] = (
        "round1_delayed_decay_autoregressive"
    )

    weighted_config = _deepcopy_mapping(base.config)
    weighted_config["training"]["forecast_strategy"] = "parallel_weighted"
    weighted_config["training"]["loss"]["horizon_weighting"] = (
        "inverse_reference_mae"
    )
    weighted_config["training"]["loss"]["horizon_reference_mae"] = [
        float(value) for value in reference_mae
    ]
    weighted_config["training"]["loss"]["horizon_weights"] = weights
    weighted_config["training"]["selection_metric"] = (
        "mean_five_horizon_cumulative_log_change_mae"
    )
    weighted_config["training"]["optimisation_profile"] = (
        "round1_delayed_decay_parallel_weighted"
    )

    autoregressive = replace(
        base,
        run_name=(
            "final_autoreg_one_step_stride"
            f"{int(one_step_training_stride)}_roll{int(max(horizons))}_"
            f"{shared_suffix}"
        ),
        label=(
            "Final ablation — one-step training, autoregressive 60-minute rollout"
        ),
        config=autoregressive_config,
    )
    weighted = replace(
        base,
        run_name=f"final_parallel_weighted_{shared_suffix}",
        label=(
            "Final ablation — parallel five-horizon head with inverse-reference "
            "horizon loss weights"
        ),
        config=weighted_config,
    )
    return autoregressive, weighted


def save_specs(path: Path, specs: Sequence[Round1RunSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_run_config(path: Path, spec: Round1RunSpec) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return values


def run_is_complete(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    return (
        metadata_path.is_file()
        and checkpoint_path.is_file()
        and load_json(metadata_path).get("status") == "completed"
    )


def summarise_runs(
    output_root: Path,
    specs: Sequence[Round1RunSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = Path(output_root) / spec.run_name
        metadata_path = run_dir / "run_metadata.json"
        history_path = run_dir / "history.csv"
        if not metadata_path.is_file() or not history_path.is_file():
            rows.append(
                {
                    "Run": spec.run_name,
                    "Label": spec.label,
                    "Status": "missing",
                }
            )
            continue
        metadata = load_json(metadata_path)
        history = pd.read_csv(history_path)
        best_epoch = metadata.get("best_epoch")
        if metadata.get("status") != "completed" or best_epoch is None:
            rows.append(
                {
                    "Run": spec.run_name,
                    "Label": spec.label,
                    "Status": metadata.get("status", "unknown"),
                    "Best epoch": best_epoch,
                    "Best score": metadata.get("best_score"),
                }
            )
            continue
        best_rows = history.loc[history["epoch"] == int(best_epoch)]
        if len(best_rows) != 1:
            raise AssertionError(
                f"Expected one best-epoch row for {spec.run_name}; "
                f"found {len(best_rows)}."
            )
        best = best_rows.iloc[0]
        row: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Status": metadata.get("status"),
            "Strategy": spec.config["training"].get("forecast_strategy", "parallel"),
            "Best epoch": int(best_epoch),
            "Epochs completed": metadata.get("epochs_completed"),
            "Mean test Log MAE": float(metadata.get("best_score")),
            "Alpha": metadata.get("final_alpha"),
            "Beta": metadata.get("final_beta"),
        }
        for horizon in spec.config["data"]["horizons"]:
            column = f"test_cumulative_log_change_mae_h{int(horizon)}"
            if column in best:
                row[f"Log MAE — {int(horizon)} min"] = float(best[column])
        rows.append(row)
    result = pd.DataFrame(rows)
    if "Mean test Log MAE" in result.columns:
        result = result.sort_values(
            ["Mean test Log MAE", "Run"],
            na_position="last",
        ).reset_index(drop=True)
    return result

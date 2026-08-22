from __future__ import annotations

"""Artifact-level reproducibility checks for weather benchmark replications.

The audit is deliberately read-only.  It compares the original unsuffixed
ModernTCN weather run with its ``kernel_15`` replication and verifies that the
model contract, optimisation settings, data, scalers, prior, targets and metric
calculation are identical.  Differences in learned predictions are then
separated from data/evaluation mismatches.
"""

from pathlib import Path
import json
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .artifacts import safe_torch_load
from .config import MODEL_OUTPUT_DIRECTORIES
from .metrics import weather_metric_payload


_CORE_CONFIG_KEYS: tuple[str, ...] = (
    "model_kind",
    "city",
    "test_year",
    "horizon",
    "start_year",
    "seed",
    "max_epochs",
    "patience",
    "min_delta",
    "weight_decay",
    "gradient_clip_norm",
    "mixed_precision",
    "backbone_learning_rate",
    "graph_learning_rate",
    "scheduler_decay_factor",
    "prior_scale",
    "prior_jitter",
    "prior_seed",
    "context_length",
    "validation_year",
    "training_end_year",
    "batch_size",
    "validation_batch_size",
    "export_batch_size",
    "scheduler_decay_start_epoch",
    "dense_prefix_training",
    "forecast_steps",
    "node_names",
    "feature_names",
    "central_node_index",
)

_TRAINING_SEMANTIC_KEYS: tuple[str, ...] = (
    "optimizer",
    "backbone_learning_rate",
    "graph_learning_rate",
    "weight_decay",
    "batch_size",
    "validation_batch_size",
    "export_batch_size",
    "max_epochs",
    "patience",
    "min_delta",
    "gradient_clip_norm",
    "mixed_precision",
    "scheduler",
    "scheduler_decay_start_epoch",
    "scheduler_decay_factor",
    "dense_prefix_training",
    "loss",
    "dense_prefix_scope",
    "seed",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _subset(values: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: values.get(key) for key in keys}


def _normalised_manifest(values: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(values)
    # The cache location can differ across Colab sessions; the SHA-256 is the
    # content identity that matters.
    result.pop("source_csv", None)
    return result


def _payload_equal(left: Any, right: Any, *, ignored_keys: frozenset[str] = frozenset()) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
            left.detach().cpu(), right.detach().cpu()
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        left_keys = {str(key) for key in left.keys()} - ignored_keys
        right_keys = {str(key) for key in right.keys()} - ignored_keys
        if left_keys != right_keys:
            return False
        return all(
            _payload_equal(left[key], right[key], ignored_keys=ignored_keys)
            for key in left_keys
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            return False
        return len(left) == len(right) and all(
            _payload_equal(a, b, ignored_keys=ignored_keys)
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def _npz_equal(left_path: Path, right_path: Path) -> bool:
    with np.load(left_path, allow_pickle=True) as left, np.load(
        right_path, allow_pickle=True
    ) as right:
        if set(left.files) != set(right.files):
            return False
        return all(np.array_equal(left[key], right[key]) for key in left.files)


def _prediction_payload(run_dir: Path, split: str) -> dict[str, Any]:
    payload = safe_torch_load(
        run_dir / f"best_{split}_predictions.pt", map_location="cpu"
    )
    result = payload.get("prediction_result")
    if not isinstance(result, Mapping):
        raise KeyError(f"Prediction artifact is malformed: {run_dir}, split={split}")
    return dict(result)


def _prediction_alignment_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "y_true",
        "last_context_target",
        "sample_idx",
        "origin_idx",
        "forecast_origin_times_ns",
        "target_indices",
        "target_times_ns",
        "horizons",
        "node_order",
        "central_node_index",
    )
    return all(_payload_equal(left.get(key), right.get(key)) for key in keys)


def _metric_values_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for scope in ("reported", "supplementary_sequence"):
        left_scope = left.get(scope)
        right_scope = right.get(scope)
        if not isinstance(left_scope, Mapping) or not isinstance(right_scope, Mapping):
            return False
        for metric in ("mae", "r", "smape"):
            if metric not in left_scope or metric not in right_scope:
                return False
            if not np.isclose(
                float(left_scope[metric]),
                float(right_scope[metric]),
                rtol=1.0e-7,
                atol=1.0e-7,
                equal_nan=True,
            ):
                return False
    return True


def _metrics_recompute(run_dir: Path, split: str) -> tuple[bool, dict[str, Any]]:
    prediction = _prediction_payload(run_dir, split)
    y_pred = torch.as_tensor(prediction["y_pred"]).cpu().numpy()
    y_true = torch.as_tensor(prediction["y_true"]).cpu().numpy()
    recomputed = weather_metric_payload(
        predictions=y_pred,
        targets=y_true,
        central_node_index=int(prediction["central_node_index"]),
    )
    saved = _read_json(run_dir / f"best_{split}_metrics.json")
    return _metric_values_close(recomputed, saved), recomputed


def _checkpoint_structure(run_dir: Path) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    payload = safe_torch_load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(f"Checkpoint has no model_state_dict: {run_dir}")
    return tuple(
        (str(name), tuple(int(value) for value in tensor.shape), str(tensor.dtype))
        for name, tensor in sorted(state.items())
    )


def _history_summary(run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(run_dir / "epoch_history.csv")
    if frame.empty:
        return {
            "epochs": 0,
            "epoch1_train_loss": np.nan,
            "epoch1_selection_score": np.nan,
        }
    first = frame.sort_values("epoch").iloc[0]
    return {
        "epochs": int(frame["epoch"].max()),
        "epoch1_train_loss": float(first["train_loss"]),
        "epoch1_selection_score": float(first["selection_score"]),
    }


def _first_material_history_divergence(
    left_dir: Path,
    right_dir: Path,
    *,
    tolerance: float = 1.0e-6,
) -> int | None:
    left = pd.read_csv(left_dir / "epoch_history.csv")
    right = pd.read_csv(right_dir / "epoch_history.csv")
    columns = ["epoch", "train_loss", "selection_score"]
    merged = left[columns].merge(
        right[columns], on="epoch", suffixes=("_original", "_kernel15")
    )
    for row in merged.itertuples(index=False):
        train_delta = abs(float(row.train_loss_original) - float(row.train_loss_kernel15))
        val_delta = abs(
            float(row.selection_score_original) - float(row.selection_score_kernel15)
        )
        if train_delta > tolerance or val_delta > tolerance:
            return int(row.epoch)
    return None


def _git_changed_files(
    project_root: Path | None,
    left_commit: str | None,
    right_commit: str | None,
) -> str | None:
    if project_root is None or not left_commit or not right_commit:
        return None
    if left_commit == right_commit:
        return ""
    try:
        output = subprocess.check_output(
            ["git", "diff", "--name-only", left_commit, right_commit],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return "; ".join(line.strip() for line in output.splitlines() if line.strip())


def audit_modern_tcn_kernel15_replications(
    *,
    output_root: str | Path,
    city: str,
    test_year: int,
    horizons: Sequence[int] = (4, 12, 28, 120),
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Compare original and kernel-15 ModernTCN runs without modifying them.

    A row is marked ``substantive_artifacts_match`` only when the architecture,
    optimisation contract, data manifest, scalers, correlation prior, initial
    graph, parameter counts, target tensors/indices, checkpoint structure and
    saved metric calculation all agree.  Learned predictions and best epochs
    are intentionally *not* required to match.
    """

    root = Path(output_root).expanduser().resolve()
    git_root = None if project_root is None else Path(project_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []

    for horizon in (int(value) for value in horizons):
        base = (
            root
            / MODEL_OUTPUT_DIRECTORIES["modern_tcn_1st"]
            / str(city).lower().strip()
            / f"horizon_{horizon}"
        )
        original_dir = base / f"test_year_{int(test_year)}"
        kernel15_dir = base / f"test_year_{int(test_year)}_kernel_15"
        required = (
            "run_complete.json",
            "resolved_config.json",
            "metadata.json",
            "environment.json",
            "data_manifest.json",
            "scalers.npz",
            "static_correlation_prior.pt",
            "initial_graphs.pt",
            "parameter_counts.json",
            "epoch_history.csv",
            "checkpoints/best.pt",
            "best_validation_predictions.pt",
            "best_validation_metrics.json",
            "best_test_predictions.pt",
            "best_test_metrics.json",
        )
        missing_original = [name for name in required if not (original_dir / name).is_file()]
        missing_kernel15 = [name for name in required if not (kernel15_dir / name).is_file()]
        if missing_original or missing_kernel15:
            rows.append(
                {
                    "horizon": horizon,
                    "available": False,
                    "missing_original": "; ".join(missing_original),
                    "missing_kernel15": "; ".join(missing_kernel15),
                    "original_run_directory": str(original_dir),
                    "kernel15_run_directory": str(kernel15_dir),
                }
            )
            continue

        original_config = _read_json(original_dir / "resolved_config.json")
        kernel15_config = _read_json(kernel15_dir / "resolved_config.json")
        original_metadata = _read_json(original_dir / "metadata.json")
        kernel15_metadata = _read_json(kernel15_dir / "metadata.json")
        original_manifest = _read_json(original_dir / "data_manifest.json")
        kernel15_manifest = _read_json(kernel15_dir / "data_manifest.json")
        original_env = _read_json(original_dir / "environment.json")
        kernel15_env = _read_json(kernel15_dir / "environment.json")
        original_complete = _read_json(original_dir / "run_complete.json")
        kernel15_complete = _read_json(kernel15_dir / "run_complete.json")

        original_validation = _prediction_payload(original_dir, "validation")
        kernel15_validation = _prediction_payload(kernel15_dir, "validation")
        original_test = _prediction_payload(original_dir, "test")
        kernel15_test = _prediction_payload(kernel15_dir, "test")

        original_metric_ok, original_test_recomputed = _metrics_recompute(
            original_dir, "test"
        )
        kernel15_metric_ok, kernel15_test_recomputed = _metrics_recompute(
            kernel15_dir, "test"
        )

        original_initial = safe_torch_load(
            original_dir / "initial_graphs.pt", map_location="cpu"
        )
        kernel15_initial = safe_torch_load(
            kernel15_dir / "initial_graphs.pt", map_location="cpu"
        )
        original_prior = safe_torch_load(
            original_dir / "static_correlation_prior.pt", map_location="cpu"
        )
        kernel15_prior = safe_torch_load(
            kernel15_dir / "static_correlation_prior.pt", map_location="cpu"
        )

        core_config_equal = _subset(original_config, _CORE_CONFIG_KEYS) == _subset(
            kernel15_config, _CORE_CONFIG_KEYS
        )
        model_config_equal = original_metadata.get("model") == kernel15_metadata.get(
            "model"
        )
        training_semantics_equal = _subset(
            original_metadata.get("training", {}), _TRAINING_SEMANTIC_KEYS
        ) == _subset(kernel15_metadata.get("training", {}), _TRAINING_SEMANTIC_KEYS)
        data_manifest_equal = _normalised_manifest(
            original_manifest
        ) == _normalised_manifest(kernel15_manifest)
        scaler_equal = _npz_equal(
            original_dir / "scalers.npz", kernel15_dir / "scalers.npz"
        )
        prior_equal = _payload_equal(original_prior, kernel15_prior)
        # The sweep added an informational ``large_kernel`` field to this
        # artifact; it does not alter the actual graph initialisation.
        initial_graph_equal = _payload_equal(
            original_initial,
            kernel15_initial,
            ignored_keys=frozenset({"large_kernel"}),
        )
        parameter_counts_equal = _read_json(
            original_dir / "parameter_counts.json"
        ) == _read_json(kernel15_dir / "parameter_counts.json")
        validation_alignment_equal = _prediction_alignment_equal(
            original_validation, kernel15_validation
        )
        test_alignment_equal = _prediction_alignment_equal(original_test, kernel15_test)
        checkpoint_structure_equal = _checkpoint_structure(
            original_dir
        ) == _checkpoint_structure(kernel15_dir)

        substantive = all(
            (
                core_config_equal,
                model_config_equal,
                training_semantics_equal,
                data_manifest_equal,
                scaler_equal,
                prior_equal,
                initial_graph_equal,
                parameter_counts_equal,
                validation_alignment_equal,
                test_alignment_equal,
                checkpoint_structure_equal,
                original_metric_ok,
                kernel15_metric_ok,
            )
        )

        original_history = _history_summary(original_dir)
        kernel15_history = _history_summary(kernel15_dir)
        original_reported = original_test_recomputed["reported"]
        kernel15_reported = kernel15_test_recomputed["reported"]
        left_commit = original_env.get("project_git_commit")
        right_commit = kernel15_env.get("project_git_commit")

        rows.append(
            {
                "horizon": horizon,
                "available": True,
                "substantive_artifacts_match": substantive,
                "core_config_equal": core_config_equal,
                "model_config_equal": model_config_equal,
                "training_semantics_equal": training_semantics_equal,
                "data_manifest_equal": data_manifest_equal,
                "source_csv_sha256_equal": original_manifest.get(
                    "source_csv_sha256"
                )
                == kernel15_manifest.get("source_csv_sha256"),
                "scalers_exact": scaler_equal,
                "correlation_prior_exact": prior_equal,
                "initial_graph_exact": initial_graph_equal,
                "parameter_counts_equal": parameter_counts_equal,
                "validation_targets_and_indices_exact": validation_alignment_equal,
                "test_targets_and_indices_exact": test_alignment_equal,
                "checkpoint_structure_equal": checkpoint_structure_equal,
                "original_test_metrics_recompute": original_metric_ok,
                "kernel15_test_metrics_recompute": kernel15_metric_ok,
                "original_best_epoch": int(original_complete["best_epoch"]),
                "kernel15_best_epoch": int(kernel15_complete["best_epoch"]),
                "original_best_validation_mse": float(
                    original_complete["best_validation_score"]
                ),
                "kernel15_best_validation_mse": float(
                    kernel15_complete["best_validation_score"]
                ),
                "validation_mse_delta_kernel15_minus_original": float(
                    kernel15_complete["best_validation_score"]
                )
                - float(original_complete["best_validation_score"]),
                "original_test_mae": float(original_reported["mae"]),
                "kernel15_test_mae": float(kernel15_reported["mae"]),
                "test_mae_delta_kernel15_minus_original": float(
                    kernel15_reported["mae"]
                )
                - float(original_reported["mae"]),
                "original_test_r": float(original_reported["r"]),
                "kernel15_test_r": float(kernel15_reported["r"]),
                "original_test_smape": float(original_reported["smape"]),
                "kernel15_test_smape": float(kernel15_reported["smape"]),
                "original_epoch1_train_loss": original_history["epoch1_train_loss"],
                "kernel15_epoch1_train_loss": kernel15_history[
                    "epoch1_train_loss"
                ],
                "original_epoch1_validation_mse": original_history[
                    "epoch1_selection_score"
                ],
                "kernel15_epoch1_validation_mse": kernel15_history[
                    "epoch1_selection_score"
                ],
                "first_material_history_divergence_epoch": _first_material_history_divergence(
                    original_dir, kernel15_dir
                ),
                "same_torch_version": original_env.get("torch")
                == kernel15_env.get("torch"),
                "same_cuda_version": original_env.get("cuda_version")
                == kernel15_env.get("cuda_version"),
                "same_cudnn_version": original_env.get("cudnn_version")
                == kernel15_env.get("cudnn_version"),
                "same_gpu_name": original_env.get("gpu_name")
                == kernel15_env.get("gpu_name"),
                "original_git_commit": left_commit,
                "kernel15_git_commit": right_commit,
                "git_changed_files_between_runs": _git_changed_files(
                    git_root,
                    None if left_commit is None else str(left_commit),
                    None if right_commit is None else str(right_commit),
                ),
                "original_run_directory": str(original_dir),
                "kernel15_run_directory": str(kernel15_dir),
            }
        )

    return pd.DataFrame(rows)

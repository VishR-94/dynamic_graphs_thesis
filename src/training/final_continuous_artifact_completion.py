from __future__ import annotations

"""Complete frozen continuous-model artefacts for the final Graph Hub.

This module never trains a model. It reloads one completed run's exact
``resolved_config.json`` and ``best_checkpoint.pt``, performs selected-
checkpoint inference over canonical train/validation/test splits, computes the
complete common metric suite, and writes both historical and canonical Graph
Hub layouts.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.data.load_candle_data import clean_candle_splits, load_candle_splits
from src.models.continuous_forecaster import (
    ContinuousRunEvaluation,
    evaluate_saved_continuous_forecaster_run,
)


GRAPH_ORIENTATION = "A[target, source]"


@dataclass(frozen=True)
class ContinuousArtifactCompletion:
    run_dir: Path
    audit: pd.DataFrame
    evaluations: dict[str, ContinuousRunEvaluation]
    manifest_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return values


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json_save(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(values), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _unwrap_graph_payload(payload: Any) -> tuple[int | None, dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved graph artefact must be a mapping.")
    epoch_value = payload.get("epoch")
    epoch = None if epoch_value is None else int(epoch_value)
    nested = payload.get("graph_artifacts")
    graph = dict(nested if isinstance(nested, Mapping) else payload)
    return epoch, graph


def canonicalise_single_layer_graph_payload(
    payload: Any,
    *,
    checkpoint_epoch: int,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    prediction_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a canonical one-layer graph wrapper without altering weights."""

    saved_epoch, graph = _unwrap_graph_payload(payload)
    if saved_epoch is not None and int(saved_epoch) != int(checkpoint_epoch):
        raise ValueError(
            "Saved graph epoch differs from the selected checkpoint: "
            f"{saved_epoch} vs {checkpoint_epoch}."
        )

    model = resolved_config["model"]
    graph_config = model["graph"]
    num_heads = int(graph_config.get("num_heads", run_metadata["graph_heads"]))
    asset_cols = [str(value) for value in run_metadata["asset_cols"]]

    graph["graph_type"] = str(
        graph.get("graph_type", run_metadata.get("graph_type", graph_config["type"]))
    )
    graph["graph_orientation"] = GRAPH_ORIENTATION
    graph["orientation"] = GRAPH_ORIENTATION
    graph["asset_cols"] = asset_cols
    graph["num_layers"] = 1
    graph["num_heads"] = num_heads
    graph["num_heads_per_layer"] = [num_heads]
    graph["layer_head_counts"] = [num_heads]
    graph["selected_layer"] = 0

    for component, layer_key in (
        ("selected", "per_layer"),
        ("base", "per_layer_base"),
        ("dynamic", "per_layer_dynamic"),
    ):
        graph[layer_key] = (graph.get(component),)

    for key in ("sample_idx", "origin_idx", "target_indices"):
        if graph.get(key) is None and prediction_result.get(key) is not None:
            graph[key] = prediction_result[key]

    if graph.get("alpha") is None and graph.get("dynamic_alpha") is not None:
        graph["alpha"] = torch.tensor(
            [float(graph["dynamic_alpha"])], dtype=torch.float32
        )
    if graph.get("beta") is None and graph.get("spatial_beta") is not None:
        graph["beta"] = torch.tensor(
            [float(graph["spatial_beta"])], dtype=torch.float32
        )

    return {
        "epoch": int(checkpoint_epoch),
        "graph_artifacts": graph,
    }


def _copy_to_canonical_layout(
    *,
    run_dir: Path,
    split_name: str,
    checkpoint_epoch: int,
    evaluation: ContinuousRunEvaluation,
    resolved_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> dict[str, Path]:
    prediction_source = evaluation.prediction_path
    graph_source = evaluation.graph_path
    metric_source = evaluation.metric_path
    diagnostics_source = evaluation.diagnostics_path
    if graph_source is None or not graph_source.is_file():
        raise FileNotFoundError(
            f"Selected-checkpoint inference saved no graph for {split_name}."
        )
    if diagnostics_source is None or not diagnostics_source.is_file():
        raise FileNotFoundError(
            f"Selected-checkpoint inference saved no diagnostics for {split_name}."
        )

    prediction_payload = torch.load(
        prediction_source,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(prediction_payload, Mapping):
        raise TypeError("Prediction payload must be a mapping.")
    prediction_result = prediction_payload.get(
        "prediction_result", prediction_payload
    )
    if not isinstance(prediction_result, Mapping):
        raise TypeError("prediction_result must be a mapping.")

    canonical_graph = canonicalise_single_layer_graph_payload(
        torch.load(graph_source, map_location="cpu", weights_only=False),
        checkpoint_epoch=checkpoint_epoch,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        prediction_result=prediction_result,
    )

    # Upgrade the historical root graph in place and write the explicit
    # best-checkpoint spelling used by newer runners.
    _atomic_torch_save(canonical_graph, graph_source)
    best_prediction = run_dir / f"best_{split_name}_predictions.pt"
    best_graph = run_dir / f"best_{split_name}_graphs.pt"
    best_metric = run_dir / f"best_{split_name}_metric_table.csv"
    best_diagnostics = run_dir / f"best_{split_name}_diagnostics.json"
    shutil.copy2(prediction_source, best_prediction)
    _atomic_torch_save(canonical_graph, best_graph)
    shutil.copy2(metric_source, best_metric)
    shutil.copy2(diagnostics_source, best_diagnostics)

    analysis_dir = run_dir / "analysis" / split_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_prediction, analysis_dir / "predictions.pt")
    shutil.copy2(best_graph, analysis_dir / "graphs.pt")
    shutil.copy2(best_metric, analysis_dir / "metric_table.csv")
    shutil.copy2(best_diagnostics, analysis_dir / "diagnostics.json")

    return {
        "predictions": best_prediction,
        "graphs": best_graph,
        "metrics": best_metric,
        "diagnostics": best_diagnostics,
        "analysis": analysis_dir,
    }


def complete_continuous_run_artifacts(
    *,
    run_dir: str | Path,
    data_dir: str | Path,
    splits: Sequence[str] = ("train", "validation", "test"),
    device: str = "auto",
    batch_size: int | None = None,
    num_workers: int = 0,
    mixed_precision: bool | None = None,
    bootstrap: bool = True,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> ContinuousArtifactCompletion:
    """Generate every final Graph Hub artefact from the frozen checkpoint."""

    run_path = Path(run_dir).expanduser().resolve()
    data_path = Path(data_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(run_path)
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)

    resolved_config = _load_json(run_path / "resolved_config.json")
    run_metadata = _load_json(run_path / "run_metadata.json")
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("best_checkpoint.pt must be a mapping.")
    checkpoint_epoch = int(checkpoint["epoch"])
    if checkpoint_epoch != int(run_metadata["best_epoch"]):
        raise ValueError(
            "Frozen best checkpoint epoch differs from run_metadata.best_epoch."
        )

    raw_train, raw_validation, raw_test = load_candle_splits(data_path)
    train_split, validation_split, test_split = clean_candle_splits(
        raw_train,
        raw_validation,
        raw_test,
    )
    split_lookup = {
        "train": train_split,
        "validation": validation_split,
        "test": test_split,
    }

    requested: list[str] = []
    for value in splits:
        name = str(value).strip().lower()
        if name == "val":
            name = "validation"
        if name not in split_lookup:
            raise ValueError("splits may contain train, validation/val, or test.")
        if name not in requested:
            requested.append(name)

    evaluations: dict[str, ContinuousRunEvaluation] = {}
    path_records: dict[str, dict[str, str]] = {}
    for split_name in requested:
        evaluation = evaluate_saved_continuous_forecaster_run(
            run_dir=run_path,
            train_split=train_split,
            evaluation_split=split_lookup[split_name],
            split_name=split_name,
            run_inference=True,
            device=device,
            batch_size=batch_size,
            num_workers=int(num_workers),
            mixed_precision=mixed_precision,
            prediction_filename=f"{split_name}_predictions.pt",
            metrics=None,
            bootstrap=bool(bootstrap),
            n_bootstrap=int(n_bootstrap),
            confidence_level=float(confidence_level),
            bootstrap_seed=int(bootstrap_seed),
        )
        evaluations[split_name] = evaluation
        paths = _copy_to_canonical_layout(
            run_dir=run_path,
            split_name=split_name,
            checkpoint_epoch=checkpoint_epoch,
            evaluation=evaluation,
            resolved_config=resolved_config,
            run_metadata=run_metadata,
        )
        path_records[split_name] = {
            key: str(value) for key, value in paths.items()
        }

    # Exercise the same loaders used by Graph Hub after writing the files.
    from src.evaluation.dynamic_graph_evaluation import (
        load_evaluation_artifacts,
        select_graph,
    )

    rows: list[dict[str, Any]] = []
    for split_name in requested:
        artifacts = load_evaluation_artifacts(
            run_path,
            split=split_name,
            policy="best",
            require_graph=True,
            require_metrics=True,
        )
        selected = select_graph(
            run_path,
            split=split_name,
            policy="best",
            day=None,
            window=None,
            component="selected",
            layer=0,
            head="mean",
        )
        predictions = torch.as_tensor(
            artifacts.prediction_result["y_pred"]
        )
        per_layer = artifacts.graph_artifacts.get("per_layer")
        rows.append(
            {
                "Split": split_name,
                "Checkpoint epoch": artifacts.epoch,
                "Windows": int(predictions.shape[0]),
                "Horizons": tuple(int(v) for v in artifacts.info.horizons),
                "Assets": int(predictions.shape[2]),
                "Graph layers": (
                    len(per_layer) if isinstance(per_layer, Sequence) else None
                ),
                "Graph heads": int(artifacts.info.num_heads),
                "Mean row entropy": selected.mean_window_row_entropy,
                "Effective neighbours": selected.mean_window_effective_neighbours,
                "Metrics": artifacts.metric_table is not None,
                "Ready": True,
            }
        )

    audit = pd.DataFrame(rows)
    audit_path = run_path / "graph_hub_artifact_audit.csv"
    audit.to_csv(audit_path, index=False)
    manifest = {
        "run_name": run_metadata.get("run_name", run_path.name),
        "run_dir": str(run_path),
        "checkpoint_epoch": checkpoint_epoch,
        "training_performed": False,
        "source_of_truth": [
            "resolved_config.json",
            "run_metadata.json",
            "best_checkpoint.pt",
        ],
        "splits": requested,
        "bootstrap": bool(bootstrap),
        "n_bootstrap": int(n_bootstrap),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(bootstrap_seed),
        "paths": path_records,
        "audit": str(audit_path),
    }
    manifest_path = run_path / "graph_hub_artifact_completion.json"
    _atomic_json_save(manifest, manifest_path)
    return ContinuousArtifactCompletion(
        run_dir=run_path,
        audit=audit,
        evaluations=evaluations,
        manifest_path=manifest_path,
    )

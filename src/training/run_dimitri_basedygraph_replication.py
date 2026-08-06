from __future__ import annotations

"""Reproduce Dimitri's x0jhc0tx model, then retrain it with test selection.

The runner first performs a non-negotiable frozen-checkpoint parity check on
Dimitri's original physical validation split.  Only after the supplied
checkpoint reproduces its saved next-token accuracy does the runner initialise
the same architecture from scratch, train on the physical train split, and use
the physical test split for checkpoint selection as requested by this
post-freeze curiosity experiment.

The retrained result is deliberately contaminated and is marked DO NOT REPORT.
The physical validation split is evaluated only after checkpoint selection.
"""

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dimitri_anchor_tokens import (
    DIMITRI_EXPECTED_WINDOWS,
    DIMITRI_SEQUENCE_LENGTH,
    file_sha256,
    validate_dimitri_anchor_token_split,
)
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_SOURCE_HASHES,
    DIMITRI_X0_CONFIG,
    DIMITRI_X0_EXPECTED_PARAMETER_COUNT,
    DIMITRI_X0_EXPECTED_VALIDATION_ACCURACY,
    DIMITRI_X0_INFERRED_BATCH_SIZE,
    DIMITRI_X0_TRAINING,
    extract_dynamic_alphas,
    import_dimitri_basedygraph,
    instantiate_exact_x0_model,
    load_dimitri_checkpoint,
    parameter_count,
    resolved_per_block_contract,
    verify_dimitri_source_snapshot,
)


EXPERIMENT_CONTRACT = "dimitri_basedygraph_v2_x0_test_selected_v1"
GRAPH_ORIENTATION = "row=target,column=source"
EXPECTED_ASSETS = 93
EXPECTED_SEQUENCE_LENGTH = 210


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _atomic_json(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def _copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _project_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        return None


def _metric_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _load_token_payload(path: Path, split: str) -> dict[str, Any]:
    """Load only fields needed by training/export after validating the full cache."""
    validate_dimitri_anchor_token_split(
        path,
        split_name=split,
        require_expected_window_count=True,
    )
    payload = _torch_load(path)
    state_ids = torch.as_tensor(payload["s1"]).long().contiguous()
    expected_shape = (
        DIMITRI_EXPECTED_WINDOWS[split],
        EXPECTED_ASSETS,
        EXPECTED_SEQUENCE_LENGTH,
    )
    if tuple(state_ids.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected {split} s1 shape {tuple(state_ids.shape)}; "
            f"expected {expected_shape}."
        )
    return {
        "s1": state_ids,
        "window_date": list(payload["window_date"]),
        "window_start": torch.as_tensor(payload["window_start"]).long(),
        "sample_idx": torch.as_tensor(payload["sample_idx"]).long(),
        "asset_cols": list(payload["asset_cols"]),
        "channels": list(payload["channels"]),
        "cache_sha256": file_sha256(path),
        "cache_path": str(path),
    }


def _row_entropy(attention: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    probabilities = attention.float()
    probabilities = probabilities / probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    probabilities = probabilities.clamp_min(eps)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def _make_evaluation_loader(
    state_ids: torch.Tensor,
    *,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, values: torch.Tensor) -> None:
            self.values = values.long()

        def __len__(self) -> int:
            return int(self.values.shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {"state_ids": self.values[index]}

    return torch.utils.data.DataLoader(
        _Dataset(state_ids),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def evaluate_token_model(
    *,
    model: torch.nn.Module,
    token_payload: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    collect_final_context_graphs: bool,
    aggregate_window_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate all next-token transitions and stream graph summaries.

    Full per-timestep graphs are not retained for every window.  The standard
    split artefact stores the final sequence-position graph from every layer and
    every window.  In parallel, the function streams an adjacency average over
    windows, all 210 positions and all heads, matching Dimitri's aggregate plot.
    """
    state_ids = torch.as_tensor(token_payload["s1"]).long().contiguous()
    loader = _make_evaluation_loader(
        state_ids,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = model.to(device).eval()

    total_nll = 0.0
    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_targets = 0
    predictive_entropy_sum = 0.0

    final_prediction_parts: list[torch.Tensor] = []
    final_target_parts: list[torch.Tensor] = []
    final_graph_parts: list[list[torch.Tensor]] = [[], [], [], []]
    window_entropy_parts: list[list[torch.Tensor]] = [[], [], [], []]
    aggregate_sums: list[torch.Tensor | None] = [None, None, None, None]
    aggregate_counts = [0, 0, 0, 0]

    selected_indices = (
        None
        if aggregate_window_indices is None
        else {int(value) for value in aggregate_window_indices}
    )
    global_window = 0

    for batch in loader:
        values = batch["state_ids"].to(device, non_blocking=True)
        output = model(values)
        logits = output["next_state_logits"]  # [B,N,T-1,K]
        targets = values[:, :, 1:]

        total_nll += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            ).item()
        )
        predicted = logits.argmax(dim=-1)
        total_correct += int((predicted == targets).sum().item())
        top5_indices = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        total_top3 += int(
            (top5_indices[..., :3] == targets.unsqueeze(-1)).any(-1).sum().item()
        )
        total_top5 += int(
            (top5_indices == targets.unsqueeze(-1)).any(-1).sum().item()
        )
        total_targets += int(targets.numel())

        log_probability = F.log_softmax(logits, dim=-1)
        predictive_entropy_sum += float(
            (-(log_probability.exp() * log_probability).sum(dim=-1)).sum().item()
        )

        final_prediction_parts.append(predicted[:, :, -1].detach().cpu().short())
        final_target_parts.append(targets[:, :, -1].detach().cpu().short())

        block_graphs = output.get("block_graph_attns") or []
        if len(block_graphs) != 4:
            raise AssertionError(
                f"Expected four block graphs; observed {len(block_graphs)}."
            )

        batch_indices = list(range(global_window, global_window + values.shape[0]))
        for layer_index, graph in enumerate(block_graphs):
            if graph is None or graph.ndim != 5:
                raise AssertionError(
                    f"Layer {layer_index} graph must be [B,T,H,N,N]; "
                    f"observed {None if graph is None else tuple(graph.shape)}."
                )
            if tuple(graph.shape[-2:]) != (EXPECTED_ASSETS, EXPECTED_ASSETS):
                raise AssertionError("Graph node dimensions differ from 93 x 93.")
            if collect_final_context_graphs:
                final_graph_parts[layer_index].append(
                    graph[:, -1].detach().cpu().to(torch.float16)
                )
            window_entropy_parts[layer_index].append(
                _row_entropy(graph).mean(dim=(1, 2, 3)).detach().cpu().float()
            )

            if selected_indices is None:
                selected_mask = torch.ones(
                    values.shape[0],
                    dtype=torch.bool,
                    device=device,
                )
            else:
                selected_mask = torch.tensor(
                    [index in selected_indices for index in batch_indices],
                    dtype=torch.bool,
                    device=device,
                )
            if selected_mask.any():
                selected_graphs = graph[selected_mask].float()
                contribution = selected_graphs.sum(dim=(0, 1, 2)).detach().cpu().double()
                if aggregate_sums[layer_index] is None:
                    aggregate_sums[layer_index] = contribution
                else:
                    aggregate_sums[layer_index] += contribution
                aggregate_counts[layer_index] += int(
                    selected_graphs.shape[0]
                    * selected_graphs.shape[1]
                    * selected_graphs.shape[2]
                )

        global_window += int(values.shape[0])
        del output, logits, targets, predicted, top5_indices, log_probability

    if total_targets <= 0:
        raise RuntimeError("Evaluation produced no next-token targets.")

    aggregates: list[torch.Tensor | None] = []
    for layer_index, values in enumerate(aggregate_sums):
        count = aggregate_counts[layer_index]
        aggregates.append(None if values is None or count == 0 else (values / count).float())

    mean_predictive_entropy = predictive_entropy_sum / total_targets
    return {
        "metrics": {
            "cross_entropy": total_nll / total_targets,
            "accuracy": total_correct / total_targets,
            "top3_accuracy": total_top3 / total_targets,
            "top5_accuracy": total_top5 / total_targets,
            "predictive_entropy": mean_predictive_entropy,
            "predictive_perplexity": math.exp(mean_predictive_entropy),
            "targets": total_targets,
        },
        "final_predicted_s1": torch.cat(final_prediction_parts, dim=0).long(),
        "final_true_s1": torch.cat(final_target_parts, dim=0).long(),
        "per_layer_final_context": (
            [torch.cat(parts, dim=0) for parts in final_graph_parts]
            if collect_final_context_graphs
            else None
        ),
        "per_layer_window_entropy": [
            torch.cat(parts, dim=0) for parts in window_entropy_parts
        ],
        "per_layer_all_time_aggregate": aggregates,
        "per_layer_all_time_counts": aggregate_counts,
        "dynamic_alphas": extract_dynamic_alphas(model),
        "aggregate_window_indices": (
            None
            if aggregate_window_indices is None
            else [int(value) for value in aggregate_window_indices]
        ),
    }


class EpochHistoryCallback:
    @staticmethod
    def build(lightning: Any, history_path: Path) -> Any:
        class _HistoryCallback(lightning.Callback):
            def __init__(self) -> None:
                super().__init__()
                self.rows: list[dict[str, Any]] = []
                if history_path.is_file():
                    self.rows = pd.read_csv(history_path).to_dict("records")

            def _save(self) -> None:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                frame = pd.DataFrame(self.rows)
                temporary = history_path.with_suffix(".csv.tmp")
                frame.to_csv(temporary, index=False)
                os.replace(temporary, history_path)

            def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
                metrics = trainer.callback_metrics
                epoch = int(trainer.current_epoch)
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "learning_rate": float(
                        trainer.optimizers[0].param_groups[0]["lr"]
                    ),
                }
                mappings = {
                    "training_cross_entropy": "train/pred_loss",
                    "training_objective_loss": "train/loss",
                    "training_s1_accuracy": "train/acc",
                    "test_selection_cross_entropy": "val/pred_loss",
                    "test_selection_objective_loss": "val/loss",
                    "test_selection_s1_accuracy": "val/acc",
                    "training_perplexity": "predictive/train/perplexity",
                    "test_selection_perplexity": "predictive/val/perplexity",
                    "training_top3_accuracy": "predictive/train/top3_acc",
                    "test_selection_top3_accuracy": "predictive/val/top3_acc",
                    "training_top5_accuracy": "predictive/train/top5_acc",
                    "test_selection_top5_accuracy": "predictive/val/top5_acc",
                    "training_selected_graph_entropy": "graph_selected/train/entropy",
                    "test_selection_selected_graph_entropy": "graph_selected/val/entropy",
                }
                for output_name, metric_name in mappings.items():
                    if metric_name in metrics:
                        row[output_name] = _metric_scalar(metrics[metric_name])

                for layer_index in range(4):
                    for source_stage, label in (("train", "training"), ("val", "test_selection")):
                        metric_name = (
                            f"graph_layers/layer_{layer_index:02d}/"
                            f"{source_stage}/entropy"
                        )
                        if metric_name in metrics:
                            row[f"layer_{layer_index}_{label}_entropy"] = _metric_scalar(
                                metrics[metric_name]
                            )
                    alpha_name = f"graph_mix/val/layer_{layer_index:02d}/alpha_mean"
                    if alpha_name in metrics:
                        row[f"layer_{layer_index}_test_selection_alpha_mean"] = _metric_scalar(
                            metrics[alpha_name]
                        )

                self.rows = [
                    existing
                    for existing in self.rows
                    if int(existing.get("epoch", -1)) != epoch
                ]
                self.rows.append(row)
                self.rows.sort(key=lambda item: int(item["epoch"]))
                self._save()

        return _HistoryCallback()


def _metric_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": f"next_s1_{name}",
                "horizon": 1,
                "channel": "s1",
                "value": float(value),
            }
            for name, value in metrics.items()
            if name != "targets"
        ]
    )


def _save_split_artifacts(
    *,
    run_dir: Path,
    split: str,
    epoch: int,
    token_payload: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    per_block_contract: Mapping[str, Sequence[Any]],
) -> None:
    public_split = "validation" if split == "val" else split
    state_ids = torch.as_tensor(token_payload["s1"]).long()
    windows = int(state_ids.shape[0])
    predicted = torch.as_tensor(evaluation["final_predicted_s1"]).long()
    truth = torch.as_tensor(evaluation["final_true_s1"]).long()
    if tuple(predicted.shape) != (windows, EXPECTED_ASSETS):
        raise AssertionError("Final predicted-token shape differs from [W,N].")
    if tuple(truth.shape) != tuple(predicted.shape):
        raise AssertionError("Final predicted and true token shapes differ.")

    window_start = torch.as_tensor(token_payload["window_start"]).long()
    sample_idx = torch.as_tensor(token_payload["sample_idx"]).long()
    # The final supervised transition is sequence position 208 -> 209.
    origin_idx = window_start + (EXPECTED_SEQUENCE_LENGTH - 2)
    target_indices = (origin_idx + 1).unsqueeze(-1)

    prediction_result = {
        "task_type": "teacher_forced_next_s1",
        "y_pred": predicted[:, None, :, None],
        "y_true": truth[:, None, :, None],
        "last_context_target": state_ids[:, :, -2].unsqueeze(-1),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": target_indices,
        "window_date": [str(value) for value in token_payload["window_date"]],
        "dates": [str(value) for value in token_payload["window_date"]],
        "window_start": window_start,
        "asset_cols": list(token_payload["asset_cols"]),
        "channels": ["s1"],
        "horizons": [1],
        "output_space": "token_id",
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "teacher_forced_transitions": EXPECTED_SEQUENCE_LENGTH - 1,
    }
    prediction_wrapper = {"epoch": int(epoch), "prediction_result": prediction_result}

    per_layer = evaluation["per_layer_final_context"]
    if not isinstance(per_layer, Sequence) or len(per_layer) != 4:
        raise AssertionError("Expected four final-context graph layers.")
    expected_heads = [int(value) for value in per_block_contract["num_edge_heads"]]
    for layer_index, graph in enumerate(per_layer):
        expected = (windows, expected_heads[layer_index], EXPECTED_ASSETS, EXPECTED_ASSETS)
        if tuple(torch.as_tensor(graph).shape) != expected:
            raise AssertionError(
                f"Layer {layer_index} graph shape {tuple(torch.as_tensor(graph).shape)} "
                f"!= {expected}."
            )

    graph_artifacts = {
        "selected": per_layer[-1],
        "per_layer": list(per_layer),
        "per_layer_all_time_aggregate": evaluation["per_layer_all_time_aggregate"],
        "per_layer_all_time_counts": evaluation["per_layer_all_time_counts"],
        "per_layer_window_entropy": evaluation["per_layer_window_entropy"],
        "per_layer_selected_window_entropy": (
            evaluation["per_layer_window_entropy"]
            if evaluation.get("aggregate_window_indices") is None
            else [
                torch.as_tensor(values)[evaluation["aggregate_window_indices"]]
                for values in evaluation["per_layer_window_entropy"]
            ]
        ),
        "dynamic_alpha_per_layer": evaluation["dynamic_alphas"],
        "graph_orientation": GRAPH_ORIENTATION,
        "asset_cols": list(token_payload["asset_cols"]),
        "saved_graph_time": "final position of each 210-token sequence",
        "aggregate_graph_scope": (
            "all sequence positions and all heads; all split windows"
            if evaluation.get("aggregate_window_indices") is None
            else "all sequence positions and all heads; selected diagnostic windows"
        ),
        "aggregate_window_indices": evaluation.get("aggregate_window_indices"),
        "graph_activations_per_layer": list(per_block_contract["activations"]),
        "graph_heads_per_layer": expected_heads,
        "graph_hidden_dims_per_layer": [
            int(value) for value in per_block_contract["graph_hidden_dims"]
        ],
    }
    graph_wrapper = {"epoch": int(epoch), "graph_artifacts": graph_artifacts}

    predictions_path = run_dir / f"best_{public_split}_predictions.pt"
    graphs_path = run_dir / f"best_{public_split}_graphs.pt"
    metrics_path = run_dir / f"best_{public_split}_metric_table.csv"
    diagnostics_path = run_dir / f"best_{public_split}_diagnostics.json"

    _atomic_torch(prediction_wrapper, predictions_path)
    _atomic_torch(graph_wrapper, graphs_path)
    metric_table = _metric_table(evaluation["metrics"])
    temporary_metric = metrics_path.with_suffix(".csv.tmp")
    metric_table.to_csv(temporary_metric, index=False)
    os.replace(temporary_metric, metrics_path)
    _atomic_json(
        {
            "split": public_split,
            "epoch": int(epoch),
            "metrics": dict(evaluation["metrics"]),
            "windows": windows,
            "graph_layers": 4,
            "graph_heads_per_layer": expected_heads,
            "graph_orientation": GRAPH_ORIENTATION,
            "dynamic_alpha_per_layer": evaluation["dynamic_alphas"],
            "token_cache_path": token_payload["cache_path"],
            "token_cache_sha256": token_payload["cache_sha256"],
        },
        diagnostics_path,
    )

    analysis_dir = run_dir / "analysis" / public_split
    _copy_or_link(predictions_path, analysis_dir / "predictions.pt")
    _copy_or_link(graphs_path, analysis_dir / "graphs.pt")
    _copy_or_link(metrics_path, analysis_dir / "metric_table.csv")
    _copy_or_link(diagnostics_path, analysis_dir / "diagnostics.json")


def _load_reusable_parity(
    *,
    parity_path: Path,
    checkpoint_path: Path,
    val_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not parity_path.is_file():
        return None
    values = json.loads(parity_path.read_text(encoding="utf-8"))
    if not values.get("passed"):
        return None
    if values.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        return None
    if values.get("validation_token_cache_sha256") != val_payload["cache_sha256"]:
        return None
    if values.get("source_hashes") != DIMITRI_SOURCE_HASHES:
        return None
    return values


def frozen_checkpoint_parity(
    *,
    checkpoint_path: Path,
    val_payload: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    diagnostic_indices: Sequence[int],
    tolerance: float,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parity_path = output_dir / "frozen_checkpoint_parity.json"
    if not force:
        reusable = _load_reusable_parity(
            parity_path=parity_path,
            checkpoint_path=checkpoint_path,
            val_payload=val_payload,
        )
        if reusable is not None:
            print("Reusing completed frozen-checkpoint parity:", parity_path)
            return reusable

    model = instantiate_exact_x0_model()
    checkpoint = load_dimitri_checkpoint(model, checkpoint_path)
    evaluation = evaluate_token_model(
        model=model,
        token_payload=val_payload,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        collect_final_context_graphs=False,
        aggregate_window_indices=diagnostic_indices,
    )
    observed = float(evaluation["metrics"]["accuracy"])
    difference = observed - DIMITRI_X0_EXPECTED_VALIDATION_ACCURACY
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "validation_token_cache_sha256": val_payload["cache_sha256"],
        "expected_validation_accuracy": DIMITRI_X0_EXPECTED_VALIDATION_ACCURACY,
        "observed_validation_accuracy": observed,
        "difference": difference,
        "tolerance": float(tolerance),
        "passed": abs(difference) <= tolerance,
        "metrics": evaluation["metrics"],
        "dynamic_alpha_per_layer": evaluation["dynamic_alphas"],
        "diagnostic_window_indices": [int(value) for value in diagnostic_indices],
        "source_hashes": verify_dimitri_source_snapshot(),
    }
    _atomic_json(result, parity_path)
    _atomic_torch(
        {
            "per_layer": evaluation["per_layer_all_time_aggregate"],
            "per_layer_counts": evaluation["per_layer_all_time_counts"],
            "per_layer_window_entropy": evaluation["per_layer_window_entropy"],
            "per_layer_selected_window_entropy": [
                torch.as_tensor(values)[list(diagnostic_indices)]
                for values in evaluation["per_layer_window_entropy"]
            ],
            "dynamic_alpha_per_layer": evaluation["dynamic_alphas"],
            "diagnostic_window_indices": list(diagnostic_indices),
            "diagnostic_window_dates": [
                str(val_payload["window_date"][index])
                for index in diagnostic_indices
            ],
            "asset_cols": list(val_payload["asset_cols"]),
            "graph_orientation": GRAPH_ORIENTATION,
            "aggregation": "mean over selected windows, all 210 positions, all heads",
        },
        output_dir / "frozen_checkpoint_dimitri_aggregate_graphs.pt",
    )

    if not result["passed"]:
        raise AssertionError(
            "Frozen x0jhc0tx parity failed. Expected validation accuracy "
            f"{DIMITRI_X0_EXPECTED_VALIDATION_ACCURACY:.9f}; observed "
            f"{observed:.9f}; difference {difference:+.9f}. Do not retrain "
            "until tokenizer revision, physical data and source parity are resolved."
        )
    return result


def _analysis_resolved_config(
    *,
    asset_cols: Sequence[str],
    source_hashes: Mapping[str, str],
    checkpoint_sha256: str,
    args: argparse.Namespace,
    per_block: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    """Create an exact provenance record plus a Graph-Hub-compatible schema."""
    return {
        "contract": EXPERIMENT_CONTRACT,
        "model_family": "dimitri_basedygraph_v2",
        # Graph Hub-compatible token model summary.
        "models": {
            "dynamic_graph": {
                "num_nodes": EXPECTED_ASSETS,
                "d_model": 96,
                "context_length": 180,
                "temporal": {
                    "type": "transformer",
                    "num_st_blocks": 4,
                    "num_layers_per_block": 1,
                    "num_heads": 4,
                    "feedforward_multiplier": 1,
                    "context_window": 180,
                    "dropout": 0.0,
                },
                "graph": {
                    "type": "dual_fusion",
                    "base_graph_type": "free_static",
                    "activation": "sparsemax",
                    "activation_per_block": list(per_block["activations"]),
                    "num_heads": int(per_block["num_edge_heads"][-1]),
                    "num_heads_per_block": [
                        int(value) for value in per_block["num_edge_heads"]
                    ],
                    "hidden_dim": int(per_block["graph_hidden_dims"][-1]),
                    "hidden_dim_per_block": [
                        int(value) for value in per_block["graph_hidden_dims"]
                    ],
                    "add_self_loops": False,
                    "diagonal_is_eligible": True,
                    "slow_window": 32,
                    "fast_window": 4,
                    "dynamic_residual_gate": "per_head",
                    "dynamic_residual_initial_alpha": 0.75,
                    "dynamic_residual_mix": "strict_convex",
                    "scorer_value": "concat",
                    "spatial_value": "concat",
                },
                "heads": {
                    "evaluation_horizons": [1],
                    "prediction_length": 1,
                    "future_token_mode": "coarse_only",
                    "s1_vocabulary_size": 1024,
                },
                "future_predictor": {
                    "type": "official_direct_next_state_head",
                    "num_layers": 0,
                },
                "loss": {
                    "type": "teacher_forced_next_s1_cross_entropy",
                    "teacher_forced_transitions": 209,
                },
            }
        },
        # Full exact source configuration, unchanged from the supplied checkpoint.
        "dimitri_basedygraph_v2": dict(DIMITRI_X0_CONFIG),
        "training": {
            **DIMITRI_X0_TRAINING,
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "early_stopping_metric": "test_next_s1_accuracy",
            "selection_split": "physical_test_December_2024",
            "do_not_report": True,
        },
        "data": {
            "token_contract": "tokens_anchor_amt0",
            "context_length": 180,
            "teacher_forced_continuation_length": 30,
            "sequence_length": 210,
            "stride": 30,
            "horizons": [1],
            "physical_split_membership_preserved": True,
            "train_windows": DIMITRI_EXPECTED_WINDOWS["train"],
            "validation_windows": DIMITRI_EXPECTED_WINDOWS["val"],
            "test_windows": DIMITRI_EXPECTED_WINDOWS["test"],
            "asset_cols": list(asset_cols),
        },
        "source_hashes": dict(source_hashes),
        "reference_checkpoint_sha256": checkpoint_sha256,
        "per_block": {key: list(values) for key, values in per_block.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exact Dimitri BaseDyGraph-V2 frozen parity and test-selected retraining."
        )
    )
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--export-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--diagnostic-windows", type=int, default=12)
    parser.add_argument("--diagnostic-stride", type=int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="BaseDyGraph Kronos TEST-CONTAMINATED",
    )
    parser.add_argument("--wandb-group", type=str, default="x0-exact-test-selected")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--force-parity", action="store_true")
    return parser


def _import_lightning() -> Any:
    import lightning.pytorch as pl

    return pl


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if args.batch_size != DIMITRI_X0_INFERRED_BATCH_SIZE:
        raise ValueError(
            "The supplied checkpoint global_step proves exact batch_size=1."
        )
    if args.seed != 0:
        raise ValueError("Exact x0jhc0tx replication uses seed 0.")
    if args.max_epochs != 120:
        raise ValueError("Exact x0jhc0tx optimisation uses max_epochs=120.")
    if args.patience != 15:
        raise ValueError("Exact x0jhc0tx optimisation uses patience=15.")
    if not math.isclose(args.learning_rate, 0.0012, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Exact x0jhc0tx optimisation uses learning_rate=0.0012.")
    if not math.isclose(args.weight_decay, 0.0001, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Exact x0jhc0tx optimisation uses weight_decay=0.0001.")

    imported = import_dimitri_basedygraph()
    lightning = _import_lightning()
    source_hashes = verify_dimitri_source_snapshot()
    per_block = resolved_per_block_contract(imported["ModelConfig"](**DIMITRI_X0_CONFIG))

    run_dir = args.output_dir / args.run_name
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(f"{run_dir} is complete; nothing to do.")
            return

    token_payloads = {
        split: _load_token_payload(args.token_dir / f"{split}.pt", split)
        for split in ("train", "val", "test")
    }
    asset_cols = list(token_payloads["train"]["asset_cols"])
    for split, payload in token_payloads.items():
        if list(payload["asset_cols"]) != asset_cols:
            raise AssertionError(f"{split} asset ordering differs from training.")

    diagnostic_indices = list(
        range(0, DIMITRI_EXPECTED_WINDOWS["val"], args.diagnostic_stride)
    )[: args.diagnostic_windows]
    parity = frozen_checkpoint_parity(
        checkpoint_path=args.reference_checkpoint,
        val_payload=token_payloads["val"],
        output_dir=run_dir / "frozen_reference",
        device=torch.device(args.device),
        batch_size=args.export_batch_size,
        num_workers=args.num_workers,
        diagnostic_indices=diagnostic_indices,
        tolerance=args.parity_tolerance,
        force=args.force_parity,
    )
    print(
        "Frozen checkpoint parity passed:",
        f"accuracy={parity['observed_validation_accuracy']:.9f}",
    )
    if args.parity_only:
        return

    resolved_config = _analysis_resolved_config(
        asset_cols=asset_cols,
        source_hashes=source_hashes,
        checkpoint_sha256=file_sha256(args.reference_checkpoint),
        args=args,
        per_block=per_block,
    )
    _atomic_json(resolved_config, run_dir / "resolved_config.json")

    started = perf_counter()
    metadata: dict[str, Any] = {
        "status": "running",
        "run_name": args.run_name,
        "experiment_contract": EXPERIMENT_CONTRACT,
        "model_family": "dimitri_basedygraph_v2",
        "selection_split": "test",
        "selection_metric": "test_next_s1_accuracy",
        "test_set_contaminated": True,
        "do_not_report": True,
        "project_commit": _project_commit(),
        "asset_cols": asset_cols,
        "graph_orientation": GRAPH_ORIENTATION,
        "reference_parity": parity,
    }
    _atomic_json(metadata, metadata_path)

    try:
        lightning.seed_everything(args.seed, workers=True)
        DataModule = imported["DiscreteStateDataModule"]
        data_module = DataModule(
            train_tensor=token_payloads["train"]["s1"],
            val_tensor=token_payloads["test"]["s1"],
            test_tensor=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = instantiate_exact_x0_model(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_t_max=None,
        )
        if parameter_count(model) != DIMITRI_X0_EXPECTED_PARAMETER_COUNT:
            raise AssertionError("Exact model parameter count differs.")

        callbacks_dir = run_dir / "lightning_checkpoints"
        callbacks_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = lightning.callbacks.ModelCheckpoint(
            dirpath=callbacks_dir,
            filename="best-{epoch:03d}",
            monitor="val/acc",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
        early_stopping = lightning.callbacks.EarlyStopping(
            monitor="val/acc",
            mode="max",
            patience=args.patience,
        )
        history_callback = EpochHistoryCallback.build(
            lightning,
            run_dir / "history.csv",
        )

        logger: Any = False
        if args.wandb_mode != "disabled":
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                project=args.wandb_project,
                name=args.run_name,
                group=args.wandb_group,
                offline=args.wandb_mode == "offline",
                config=resolved_config,
                tags=["DO-NOT-REPORT", "TEST-SELECTION-CONTAMINATED", "x0-exact"],
            )

        trainer = lightning.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu" if args.device.startswith("cuda") else "cpu",
            devices=1,
            precision="32-true",
            logger=logger,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, early_stopping, history_callback],
            enable_progress_bar=True,
            num_sanity_val_steps=0,
            deterministic=False,
            log_every_n_steps=10,
        )

        training_complete_path = run_dir / "training_complete.json"
        if training_complete_path.is_file() and not args.overwrite:
            training_complete = json.loads(
                training_complete_path.read_text(encoding="utf-8")
            )
            best_lightning_path = Path(training_complete["best_lightning_checkpoint"])
            last_lightning_path = Path(training_complete["last_lightning_checkpoint"])
            if not best_lightning_path.is_file() or not last_lightning_path.is_file():
                raise FileNotFoundError("A recorded completed Lightning checkpoint is missing.")
            print("Training already finished; resuming selected-checkpoint export.")
        else:
            resume_path = callbacks_dir / "last.ckpt"
            trainer.fit(
                model,
                datamodule=data_module,
                ckpt_path=str(resume_path) if resume_path.is_file() else None,
            )
            best_lightning_path = Path(checkpoint_callback.best_model_path)
            last_lightning_path = Path(checkpoint_callback.last_model_path)
            if not best_lightning_path.is_file():
                raise FileNotFoundError(best_lightning_path)
            if not last_lightning_path.is_file():
                raise FileNotFoundError(last_lightning_path)
            _atomic_json(
                {
                    "best_lightning_checkpoint": str(best_lightning_path),
                    "last_lightning_checkpoint": str(last_lightning_path),
                    "best_test_selection_accuracy": _metric_scalar(
                        checkpoint_callback.best_model_score
                    ),
                },
                training_complete_path,
            )

        selected_model = instantiate_exact_x0_model(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_t_max=None,
        )
        best_blob = _torch_load(best_lightning_path)
        selected_model.load_state_dict(best_blob["state_dict"], strict=True)
        best_epoch = int(best_blob["epoch"])

        best_score: float | None = None
        for callback_values in best_blob.get("callbacks", {}).values():
            if (
                isinstance(callback_values, dict)
                and callback_values.get("monitor") == "val/acc"
                and callback_values.get("best_model_score") is not None
            ):
                best_score = _metric_scalar(callback_values["best_model_score"])
                break
        if best_score is None:
            history = pd.read_csv(run_dir / "history.csv")
            row = history.loc[history["epoch"] == best_epoch]
            if len(row) != 1:
                raise AssertionError("Could not recover selected test accuracy.")
            best_score = float(row.iloc[0]["test_selection_s1_accuracy"])

        standard_best = {
            "epoch": best_epoch,
            "best_score": best_score,
            "selection_split": "test",
            "selection_metric": "next_s1_accuracy",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in selected_model.state_dict().items()
            },
            "config": resolved_config,
            "source_lightning_checkpoint": str(best_lightning_path),
        }
        _atomic_torch(standard_best, run_dir / "best_checkpoint.pt")

        last_blob = _torch_load(last_lightning_path)
        _atomic_torch(
            {
                "epoch": int(last_blob["epoch"]),
                "best_score": best_score,
                "selection_split": "test",
                "selection_metric": "next_s1_accuracy",
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in last_blob["state_dict"].items()
                },
                "config": resolved_config,
                "source_lightning_checkpoint": str(last_lightning_path),
            },
            run_dir / "last_checkpoint.pt",
        )

        split_results: dict[str, Any] = {}
        for split in ("train", "val", "test"):
            print(f"Exporting selected checkpoint on {split}...", flush=True)
            evaluation = evaluate_token_model(
                model=selected_model,
                token_payload=token_payloads[split],
                device=torch.device(args.device),
                batch_size=args.export_batch_size,
                num_workers=args.num_workers,
                collect_final_context_graphs=True,
                aggregate_window_indices=(diagnostic_indices if split == "val" else None),
            )
            _save_split_artifacts(
                run_dir=run_dir,
                split=split,
                epoch=best_epoch,
                token_payload=token_payloads[split],
                evaluation=evaluation,
                per_block_contract=per_block,
            )
            split_results["validation" if split == "val" else split] = {
                "metrics": evaluation["metrics"],
                "dynamic_alpha_per_layer": evaluation["dynamic_alphas"],
            }
            del evaluation

        metadata.update(
            {
                "status": "completed",
                "best_epoch": best_epoch,
                "best_score": best_score,
                "epochs_completed": int(last_blob["epoch"]) + 1,
                "parameter_count": parameter_count(selected_model),
                "train_windows": DIMITRI_EXPECTED_WINDOWS["train"],
                "validation_windows": DIMITRI_EXPECTED_WINDOWS["val"],
                "test_windows": DIMITRI_EXPECTED_WINDOWS["test"],
                "split_results": split_results,
                "dynamic_alpha_per_layer": extract_dynamic_alphas(selected_model),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        _atomic_json(metadata, metadata_path)
        print("Completed:", run_dir)
        print("Best epoch:", best_epoch)
        print("Best test-selection accuracy:", best_score)
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": repr(error),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        _atomic_json(metadata, metadata_path)
        raise


if __name__ == "__main__":
    main()

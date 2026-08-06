from __future__ import annotations

"""Compact diagnostics for the exact Dimitri BaseDyGraph-V2 replication."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class DimitriAggregateGraphReport:
    summary: pd.DataFrame
    figure: Figure
    axes: np.ndarray


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _load_sector_map(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Ticker" not in fieldnames or "Sector" not in fieldnames:
            raise ValueError(
                f"{path} must contain Ticker and Sector columns; observed {fieldnames}."
            )
        mapping: dict[str, str] = {}
        for row in reader:
            ticker = str(row["Ticker"]).strip()
            sector = str(row["Sector"]).strip()
            if ticker:
                mapping[ticker] = sector or "Unknown"
    return mapping


def _entropy_excluding_diagonal(adjacency: np.ndarray) -> float:
    values = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(values, 0.0)
    values /= np.clip(values.sum(axis=1, keepdims=True), 1.0e-12, None)
    return float(np.mean(-(values * np.log(values + 1.0e-12)).sum(axis=1)))


def _normalise_aggregate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "graph_artifacts" in payload:
        payload = payload["graph_artifacts"]
    per_layer = payload.get("per_layer_all_time_aggregate")
    if per_layer is None:
        per_layer = payload.get("per_layer")
    if not isinstance(per_layer, Sequence) or isinstance(per_layer, (str, bytes)):
        raise ValueError("Aggregate graph payload contains no per-layer matrices.")
    matrices = []
    for layer, values in enumerate(per_layer):
        if values is None:
            raise ValueError(f"Layer {layer} aggregate is missing.")
        tensor = torch.as_tensor(values).detach().cpu().double()
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError(
                f"Layer {layer} aggregate must be [N,N], got {tuple(tensor.shape)}."
            )
        matrices.append(tensor.numpy())

    entropy_values = payload.get("per_layer_selected_window_entropy")
    if entropy_values is None:
        entropy_values = payload.get("per_layer_window_entropy")
    if entropy_values is None:
        per_instance_entropy = [float("nan")] * len(matrices)
    else:
        per_instance_entropy = [
            float(torch.as_tensor(values).float().mean().item())
            for values in entropy_values
        ]
    return {
        "matrices": matrices,
        "per_instance_entropy": per_instance_entropy,
        "counts": payload.get("per_layer_counts")
        or payload.get("per_layer_all_time_counts"),
        "asset_cols": payload.get("asset_cols"),
        "dynamic_alpha_per_layer": payload.get("dynamic_alpha_per_layer"),
        "diagnostic_window_indices": payload.get("diagnostic_window_indices"),
        "diagnostic_window_dates": payload.get("diagnostic_window_dates"),
        "aggregation": payload.get("aggregation")
        or payload.get("aggregate_graph_scope"),
    }


def plot_dimitri_sector_sorted_aggregates(
    graph_path: str | Path,
    *,
    company_profiles_path: str | Path,
    asset_cols: Sequence[str] | None = None,
    title: str = "Dimitri-style BaseDyGraph aggregate graphs",
    figsize: tuple[float, float] = (24.0, 6.5),
) -> DimitriAggregateGraphReport:
    """Plot one sector-sorted aggregate matrix per interlaced ST block.

    The plotted quantity matches Dimitri's aggregate view: each layer is averaged
    across the selected windows, all sequence positions and all graph heads.  The
    entropy shown in each title is instead the mean per-instance row entropy, as
    in the training diagnostics; entropy of the already averaged matrix is listed
    separately because averaging different sparse supports makes it look denser.
    """
    graph_path = Path(graph_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    payload = _torch_load(graph_path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {graph_path}.")
    values = _normalise_aggregate_payload(payload)

    resolved_assets = asset_cols or values["asset_cols"]
    if resolved_assets is None:
        raise ValueError("asset_cols must be supplied or saved in the graph payload.")
    assets = [str(value) for value in resolved_assets]
    matrices = values["matrices"]
    if any(matrix.shape != (len(assets), len(assets)) for matrix in matrices):
        raise ValueError("Aggregate matrix dimensions differ from asset_cols.")

    sector_map = _load_sector_map(company_profiles_path)
    missing = [asset for asset in assets if asset not in sector_map]
    if missing:
        raise ValueError(f"Company profile file has no sector for assets: {missing}")
    sectors = [sector_map[asset] for asset in assets]
    order = np.argsort(np.asarray(sectors, dtype=object), kind="stable")
    sorted_sectors = [sectors[index] for index in order]
    boundaries = [
        index
        for index in range(1, len(sorted_sectors))
        if sorted_sectors[index] != sorted_sectors[index - 1]
    ]

    figure, axes = plt.subplots(1, len(matrices), figsize=figsize, squeeze=False)
    flat_axes = axes.reshape(-1)
    summary_rows: list[dict[str, Any]] = []

    counts = values["counts"] or [None] * len(matrices)
    alphas = values["dynamic_alpha_per_layer"] or [[] for _ in matrices]
    for layer, (matrix, axis) in enumerate(zip(matrices, flat_axes)):
        sorted_matrix = matrix[np.ix_(order, order)].copy()
        np.fill_diagonal(sorted_matrix, np.nan)
        finite = sorted_matrix[np.isfinite(sorted_matrix)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        image = axis.imshow(
            sorted_matrix,
            cmap="Reds",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        for boundary in boundaries:
            axis.axhline(boundary - 0.5, color="0.55", linewidth=0.35)
            axis.axvline(boundary - 0.5, color="0.55", linewidth=0.35)
        axis.set_xticks([])
        axis.set_yticks([])
        per_instance_entropy = float(values["per_instance_entropy"][layer])
        aggregate_entropy = _entropy_excluding_diagonal(matrix)
        effective_sources = (
            float(np.exp(per_instance_entropy))
            if np.isfinite(per_instance_entropy)
            else float("nan")
        )
        axis.set_title(
            f"block {layer}\n"
            f"per-instance H={per_instance_entropy:.3f} "
            f"(~{effective_sources:.1f} sources)",
            fontsize=10,
        )
        axis.set_xlabel("source (sector-sorted)")
        if layer == 0:
            axis.set_ylabel("target (sector-sorted)")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        summary_rows.append(
            {
                "Block": layer,
                "Activation": (
                    "sparsemax" if layer == len(matrices) - 1 else "softmax"
                ),
                "Instantaneous graphs averaged": (
                    None if layer >= len(counts) else counts[layer]
                ),
                "Mean per-instance row entropy": per_instance_entropy,
                "Effective sources per row": effective_sources,
                "Entropy of aggregate matrix": aggregate_entropy,
                "Learned fast-graph alpha mean": (
                    float(np.mean(alphas[layer]))
                    if layer < len(alphas) and len(alphas[layer])
                    else np.nan
                ),
            }
        )

    figure.suptitle(
        title
        + "\n"
        + "aggregate = mean across selected windows, all timesteps and all heads",
        y=1.03,
    )
    figure.tight_layout()
    return DimitriAggregateGraphReport(
        summary=pd.DataFrame(summary_rows),
        figure=figure,
        axes=axes,
    )


def load_replication_summary(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        result = metadata.get("split_results", {}).get(split)
        if not isinstance(result, Mapping):
            continue
        metrics = result.get("metrics", {})
        rows.append(
            {
                "Split": split,
                "Windows": metadata.get(f"{split}_windows"),
                "Next-s1 CE": metrics.get("cross_entropy"),
                "Next-s1 accuracy": metrics.get("accuracy"),
                "Top-3 accuracy": metrics.get("top3_accuracy"),
                "Top-5 accuracy": metrics.get("top5_accuracy"),
                "Predictive perplexity": metrics.get("predictive_perplexity"),
            }
        )
    return pd.DataFrame(rows)

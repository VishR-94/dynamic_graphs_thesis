"""Reusable diagnostics for saved dynamic-graph forecasting runs.

The public functions accept a run directory and load the artefacts they need
from that directory. Graph orientation is always ``A[target, source]``.

Current public API
------------------
``make_model_summary_table`` / ``style_model_summary_table``
    Summarise the saved architecture, token target, training loss and exact
    trainable parameter count.

``plot_learned_graph``
    Plot a checkpoint graph with hierarchical clustering and readable asset
    labels. Static graphs can be read from ``last_checkpoint.pt``. Saved
    validation graph artefacts can be used for static or dynamic models.

``make_metrics_table`` / ``style_metrics_table``
    Build and format either the best or last validation metric table.

``make_top_neighbours_table`` / ``style_top_neighbours_table``
    Report either the stocks that influence each stock or the stocks each
    stock influences.

``analyse_absolute_correlation_graph``
    Build the exact production fixed-correlation adjacency from a candle-data
    split, plot it, display its top-neighbour table, and report row entropy.

``analyse_spatial_branch_reliance``
    Measure the relative scale, alignment, and immediate post-normalisation
    effect of the temporal residual and graph-message branches.

``analyse_graph_gate_sweep``
    Scale the trained graph-message branch from zero to full strength and
    measure end-to-end token and optional decoded-price sensitivity.

``analyse_graph_topology_counterfactuals``
    Compare the trained adjacency with zero-message, uniform, and row-wise
    shuffled graph interventions to distinguish graph-branch use from
    reliance on the specific topology.
"""

from __future__ import annotations

EVALUATION_MODULE_VERSION = "2026-07-30-consolidated-v7-graph-reliance"

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.models.dynamic_graph.config import build_model_config
from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.load_candle_data import clean_candle_split
from src.models.dynamic_graph.graph_learners import (
    EmptyCorrelationRowPolicy,
    build_absolute_correlation_adjacency,
    build_graph_learner,
)
from src.models.dynamic_graph.model import DynamicGraphTokenForecaster
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.evaluation.prediction_transforms import raw_to_cumulative_log_change
from src.utils.config import load_yaml
from src.utils.metric_tables import DEFAULT_METRIC_DISPLAY_NAMES
from src.visualization.candle_plots import (
    compute_return_correlation_matrix,
    reorder_correlation_matrix,
)


CheckpointSource = Literal[
    "last",
    "best",
    "best_validation",
]
MetricsSource = Literal[
    "best",
    "last",
]
GraphComponent = Literal[
    "selected",
    "base",
    "dynamic",
]
NeighbourDirection = Literal[
    "impacted_by",
    "impacts",
]
HeadSelection = int | Literal["mean"]


@dataclass(frozen=True)
class RunInfo:
    """Minimal, validated metadata for one saved run."""

    run_dir: Path
    resolved_config: dict[str, Any]
    run_metadata: dict[str, Any]
    history: pd.DataFrame
    asset_cols: tuple[str, ...]
    graph_type: str
    num_nodes: int
    num_heads: int
    num_st_blocks: int
    add_self_loops: bool


@dataclass(frozen=True)
class LoadedGraph:
    """A graph tensor and the provenance used to construct it."""

    values: Tensor  # [G, N, N]
    asset_cols: tuple[str, ...]
    graph_type: str
    source: str
    checkpoint_epoch: int | None
    component: str


@dataclass(frozen=True)
class CorrelationGraphDiagnostics:
    """Outputs from one thresholded absolute-correlation graph analysis."""

    threshold: float
    correlation_matrix: pd.DataFrame
    adjacency: pd.DataFrame
    top_neighbours: pd.DataFrame
    row_entropy: pd.Series
    retained_neighbours: pd.Series
    summary: pd.DataFrame
    figure: Figure
    axes: Axes

    @property
    def mean_row_entropy(self) -> float:
        return float(self.summary.loc[0, "Mean row entropy"])

    @property
    def mean_effective_neighbours(self) -> float:
        return float(self.summary.loc[0, "Mean effective neighbours"])


def _resolve_run_dir(run_dir: str | Path) -> Path:
    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return loaded


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_run_info(run_dir: str | Path) -> RunInfo:
    """Load the small set of files shared by all run diagnostics."""

    run_path = _resolve_run_dir(run_dir)
    resolved_config = _load_json(run_path / "resolved_config.json")
    run_metadata = _load_json(run_path / "run_metadata.json")

    history_path = run_path / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)

    model_config = resolved_config["models"]["dynamic_graph"]
    graph_config = model_config["graph"]

    asset_cols = tuple(str(value) for value in run_metadata["asset_cols"])
    num_nodes = int(model_config["num_nodes"])

    if len(asset_cols) != num_nodes:
        raise ValueError(
            "run_metadata.asset_cols does not match the configured node count."
        )
    if len(set(asset_cols)) != len(asset_cols):
        raise ValueError("run_metadata.asset_cols contains duplicate names.")

    return RunInfo(
        run_dir=run_path,
        resolved_config=resolved_config,
        run_metadata=run_metadata,
        history=history,
        asset_cols=asset_cols,
        graph_type=str(graph_config["type"]),
        num_nodes=num_nodes,
        num_heads=int(graph_config["num_heads"]),
        num_st_blocks=int(model_config["num_st_blocks"]),
        add_self_loops=bool(graph_config.get("add_self_loops", False)),
    )



def _checkpoint_path(run_dir: Path, source: CheckpointSource) -> Path:
    if source == "last":
        return run_dir / "last_checkpoint.pt"
    if source in {"best", "best_validation"}:
        return run_dir / "best_checkpoint.pt"
    raise ValueError(f"Unsupported checkpoint source {source!r}.")


def _normalise_layer_index(layer: int, num_layers: int) -> int:
    resolved = int(layer)
    if resolved < 0:
        resolved += num_layers
    if not 0 <= resolved < num_layers:
        raise IndexError(
            f"Graph layer {layer} is outside the {num_layers}-layer model."
        )
    return resolved


def _graph_learner_state(
    model_state: Mapping[str, Tensor],
    *,
    layer_index: int,
) -> dict[str, Tensor]:
    prefix = f"graph_learners.{layer_index}."
    state = {
        key[len(prefix):]: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    if not state:
        raise KeyError(
            f"Checkpoint contains no graph-learner state for layer {layer_index}."
        )
    return state


def _fixed_resource_from_state(
    graph_type: str,
    learner_state: Mapping[str, Tensor],
) -> Tensor | None:
    if graph_type == "fixed":
        values = learner_state.get("_singleton_adjacency")
        if values is None:
            raise KeyError(
                "The fixed-graph checkpoint does not contain "
                "_singleton_adjacency."
            )
        return torch.as_tensor(values)

    if graph_type == "dynamic_base":
        values = learner_state.get("_fixed_base_adjacency")
        if values is not None:
            return torch.as_tensor(values)

    return None


def _load_saved_model(
    info: RunInfo,
    *,
    source: CheckpointSource,
) -> tuple[
    DynamicGraphTokenForecaster,
    Any,
    dict[str, Any],
]:
    """Reconstruct one saved model exactly enough to count parameters.

    Fixed graph tensors are recovered from the checkpoint because they are
    external resources rather than ordinary YAML values. The model state is
    then loaded strictly so a parameter count is never silently calculated
    from an architecture that differs from the saved run.
    """

    checkpoint = _torch_load(
        _checkpoint_path(
            info.run_dir,
            source,
        )
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")

    checkpoint_config = checkpoint.get("resolved_config")
    if isinstance(checkpoint_config, Mapping):
        experiment_config = dict(checkpoint_config)
    else:
        experiment_config = info.resolved_config

    model_config = build_model_config(experiment_config)

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise KeyError("Checkpoint does not contain model_state_dict.")

    fixed_adjacency: Tensor | None = None
    if model_config.graph.type in {
        "fixed",
        "dynamic_base",
    }:
        first_learner_state = _graph_learner_state(
            model_state,
            layer_index=0,
        )
        fixed_adjacency = _fixed_resource_from_state(
            model_config.graph.type,
            first_learner_state,
        )

        if (
            model_config.graph.type == "fixed"
            and fixed_adjacency is None
        ):
            raise KeyError(
                "The fixed-graph checkpoint does not contain its saved "
                "adjacency resource."
            )

    stored_oracle_key = (
        "graph_learners.0._stored_oracle_graph"
    )
    oracle_graph = (
        torch.as_tensor(model_state[stored_oracle_key])
        if stored_oracle_key in model_state
        else None
    )

    model = DynamicGraphTokenForecaster.from_config(
        experiment_config,
        fixed_adjacency=fixed_adjacency,
        oracle_graph=oracle_graph,
    )

    try:
        model.load_state_dict(
            dict(model_state),
            strict=True,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "The current code cannot reconstruct the saved model exactly. "
            "Use the repository revision recorded by the run before "
            "reporting its trainable parameter count."
        ) from error

    return model, model_config, experiment_config


def _describe_temporal(model_config: Any) -> str:
    temporal = model_config.temporal
    block_suffix = (
        f"; repeated in {model_config.num_st_blocks} ST block"
        f"{'s' if model_config.num_st_blocks != 1 else ''}"
    )

    if temporal.type == "transformer":
        return (
            "Causal per-node Transformer "
            f"({temporal.num_layers} layer"
            f"{'s' if temporal.num_layers != 1 else ''}, "
            f"{temporal.num_heads} heads, d_model={model_config.d_model}"
            f"{block_suffix})"
        )

    if temporal.type == "tcn":
        return (
            "Causal per-node TCN "
            f"(kernel={temporal.kernel_size}, "
            f"dilations={list(temporal.dilations)}, "
            f"receptive field={temporal.tcn_receptive_field}"
            f"{block_suffix})"
        )

    if temporal.type == "identity":
        return f"Identity temporal encoder ({block_suffix.lstrip('; ')})"

    return str(temporal.type)


def _describe_graph(model_config: Any) -> str:
    graph = model_config.graph
    common = (
        f"{graph.num_heads} heads; activation={graph.activation}; "
        f"self-loops={'on' if graph.add_self_loops else 'off'}"
    )

    if graph.type == "none":
        return "No graph learner"
    if graph.type == "fixed":
        return f"Fixed supplied graph ({common})"
    if graph.type == "free_static":
        return (
            "BaseDyGraph free-static direct edge logits "
            f"({common})"
        )
    if graph.type == "mtgnn_static":
        return (
            "MTGNN static node-embedding graph "
            f"(top-k={graph.mtgnn_top_k}, alpha={graph.mtgnn_alpha:g}; "
            f"{common})"
        )
    if graph.type == "dynamic":
        return (
            "BaseDyGraph window-conditioned Q/K graph "
            f"(hidden={graph.hidden_dim}; {common})"
        )
    if graph.type == "dynamic_base":
        return (
            "BaseDyGraph dynamic-base graph "
            f"(base={graph.base_graph_type}, gate={graph.gate_type}, "
            f"initial alpha={graph.initial_alpha:g}; {common})"
        )
    if graph.type == "oracle":
        return f"Oracle supplied true graph ({common})"
    return str(graph.type)


def _describe_spatial(model_config: Any) -> str:
    if model_config.graph.type == "none":
        return "Identity spatial path (no cross-node mixing)"

    blocks = int(model_config.num_st_blocks)
    return (
        "BaseDyGraph-style graph-weighted value aggregation "
        f"({model_config.graph.num_heads} heads; residual + LayerNorm + "
        f"FFN; 1 spatial layer per ST block; {blocks} ST block"
        f"{'s' if blocks != 1 else ''})"
    )


def _describe_output_head(model_config: Any) -> str:
    future = model_config.future_predictor

    if future.type == "structured_parallel":
        if int(future.num_layers) == 0:
            return (
                "Direct parallel position-conditioned token head "
                "(final graph-aware state + learned future positions; "
                "future Transformer disabled)"
            )
        return (
            "Structured-parallel future Transformer "
            f"({future.num_layers} layer"
            f"{'s' if future.num_layers != 1 else ''}, "
            f"{future.num_heads} heads)"
        )

    if future.type == "autoregressive":
        return (
            "Autoregressive causal future Transformer "
            f"({future.num_layers} layer"
            f"{'s' if future.num_layers != 1 else ''}, "
            f"{future.num_heads} heads)"
        )

    return str(future.type)


def _describe_token_target(model_config: Any) -> str:
    heads = model_config.heads

    if heads.future_token_mode == "coarse_only":
        return (
            "Observed context uses s1+s2; predicts coarse s1 only "
            f"(vocabulary={heads.s1_vocabulary_size}); coarse Kronos decode"
        )

    conditioning = str(heads.resolved_s2_conditioning).replace("_", " ")
    return (
        "Observed context and future use s1+s2; hierarchical prediction "
        f"(s1 vocabulary={heads.s1_vocabulary_size}, "
        f"s2 vocabulary={heads.s2_vocabulary_size}; "
        f"s2 conditioned on {conditioning}); full Kronos decode"
    )


def _describe_position_weighting(loss_values: Mapping[str, Any]) -> str:
    weighting = str(loss_values.get("horizon_weighting", "uniform"))

    if weighting == "uniform":
        return "uniform future-position weights"

    if weighting == "exponential_decay":
        half_life = float(loss_values["exponential_half_life"])
        floor = float(loss_values["exponential_floor_weight"])
        return (
            "exponential-decay future weights "
            f"(half-life={half_life:g}, floor={floor:g})"
        )

    if weighting == "gaussian_mixture":
        sigma = float(loss_values.get("gaussian_sigma", float("nan")))
        peak_mass = float(
            loss_values.get("gaussian_peak_mass", float("nan"))
        )
        return (
            "legacy Gaussian-envelope future weights "
            f"(sigma={sigma:g}, peak mass={peak_mass:g})"
        )

    return f"{weighting} future-position weights"


def _describe_training_loss(
    model_config: Any,
    experiment_config: Mapping[str, Any],
) -> str:
    model_values = experiment_config["models"]["dynamic_graph"]
    loss_values = model_values["loss"]
    graph_reg = model_values.get("graph_regularisation", {})

    if model_config.heads.future_token_mode == "coarse_only":
        token_loss = "s1 cross-entropy"
    else:
        token_loss = (
            "s1 cross-entropy + "
            f"{model_config.heads.s2_loss_weight:g}×s2 cross-entropy"
        )

    parts = [
        token_loss,
        _describe_position_weighting(loss_values),
    ]

    target_coefficient = float(
        graph_reg.get("graph_target_entropy_reg", 0.0)
    )
    target_entropy = graph_reg.get("graph_target_entropy")
    if target_coefficient > 0.0 and target_entropy is not None:
        parts.append(
            "target-entropy regularisation "
            f"(lambda={target_coefficient:g}, H*={float(target_entropy):g})"
        )

    direct_entropy = float(graph_reg.get("graph_entropy_reg", 0.0))
    if direct_entropy > 0.0:
        parts.append(
            f"direct entropy regularisation (lambda={direct_entropy:g})"
        )

    temporal_smooth = float(
        graph_reg.get("graph_temporal_smooth_reg", 0.0)
    )
    if temporal_smooth > 0.0:
        parts.append(
            f"graph temporal smoothing (lambda={temporal_smooth:g})"
        )

    if model_config.backcast.enabled:
        parts.append(
            "backcast loss "
            f"(lambda={model_config.backcast.loss_weight:g})"
        )

    return "; ".join(parts)


def make_model_summary_table(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "last",
) -> pd.DataFrame:
    """Return a compact architecture and training summary for one run.

    The trainable count is reconstructed from the saved configuration and
    checkpoint, and excludes the frozen Kronos tokenizer/decoder because
    those modules are not trainable parts of the forecasting model.
    """

    info = load_run_info(run_dir)
    model, model_config, experiment_config = _load_saved_model(
        info,
        source=source,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    values = {
        "Temporal encoder": _describe_temporal(model_config),
        "Graph learner": _describe_graph(model_config),
        "Spatial module": _describe_spatial(model_config),
        "Output head": _describe_output_head(model_config),
        "Token target": _describe_token_target(model_config),
        "Training loss": _describe_training_loss(
            model_config,
            experiment_config,
        ),
        "Trainable parameters": (
            f"{trainable_parameters:,} "
            f"({trainable_parameters / 1_000_000:.3f}M; "
            "excludes frozen Kronos tokenizer/decoder)"
        ),
    }

    table = pd.DataFrame.from_dict(
        values,
        orient="index",
        columns=["Configuration"],
    )
    table.index.name = "Component"
    table.attrs["checkpoint_source"] = source
    table.attrs["trainable_parameters"] = int(trainable_parameters)
    return table


def style_model_summary_table(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "last",
    caption: str | None = None,
) -> pd.io.formats.style.Styler:
    """Return a clean formatted model-summary table."""

    table = make_model_summary_table(
        run_dir,
        source=source,
    )
    run_name = _resolve_run_dir(run_dir).name

    return (
        table.style
        .set_caption(caption or f"{run_name} — Model Summary")
        .set_properties(
            subset=["Configuration"],
            **{
                "text-align": "left",
                "white-space": "normal",
            },
        )
    )


def _load_static_checkpoint_graph(
    info: RunInfo,
    *,
    source: CheckpointSource,
    component: GraphComponent,
    layer: int,
) -> LoadedGraph:
    """Read a global graph directly from a saved checkpoint.

    Dynamic selected graphs are input-conditioned and therefore do not exist
    as one checkpoint matrix. For those models, use ``source='best_validation'``
    to load the saved validation graphs and average or select windows.
    """

    checkpoint = _torch_load(_checkpoint_path(info.run_dir, source))
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")

    checkpoint_config = checkpoint.get("resolved_config")
    if isinstance(checkpoint_config, Mapping):
        experiment_config = dict(checkpoint_config)
    else:
        experiment_config = info.resolved_config

    model_config = build_model_config(experiment_config)
    layer_index = _normalise_layer_index(layer, model_config.num_st_blocks)

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise KeyError("Checkpoint does not contain model_state_dict.")

    learner_state = _graph_learner_state(
        model_state,
        layer_index=layer_index,
    )

    graph_type = model_config.graph.type

    if graph_type in {"dynamic", "oracle"}:
        raise ValueError(
            f"graph.type={graph_type!r} has no single checkpoint graph. "
            "Use source='best_validation' to inspect saved validation graphs."
        )

    if graph_type == "dynamic_base" and component != "base":
        raise ValueError(
            "A dynamic-base selected/dynamic graph is input-conditioned. "
            "Use source='best_validation', or request component='base'."
        )

    if graph_type == "none":
        raise ValueError("This run has graph.type='none'.")

    fixed_resource = _fixed_resource_from_state(graph_type, learner_state)

    learner = build_graph_learner(
        config=model_config.graph,
        num_nodes=model_config.num_nodes,
        d_model=model_config.d_model,
        fixed_adjacency=fixed_resource,
    )
    learner.load_state_dict(learner_state, strict=True)
    learner.eval()

    if graph_type == "dynamic_base" and component == "base":
        if not hasattr(learner, "singleton_base_adjacency"):
            raise TypeError("Dynamic-base learner does not expose its base graph.")
        graph = learner.singleton_base_adjacency()  # type: ignore[attr-defined]
    else:
        dummy_context = torch.zeros(
            1,
            1,
            model_config.num_nodes,
            model_config.d_model,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            output = learner(dummy_context)

        if component == "selected":
            graph = output.selected
        elif component == "base":
            graph = output.base
        else:
            graph = output.dynamic

        if graph is None:
            raise ValueError(
                f"Graph component {component!r} is unavailable for "
                f"graph.type={graph_type!r}."
            )

    graph_tensor = torch.as_tensor(graph).detach().cpu().to(torch.float64)
    if graph_tensor.ndim != 4 or int(graph_tensor.shape[0]) != 1:
        raise ValueError(
            "Checkpoint graph must have shape [1, G, N, N]; "
            f"observed {tuple(graph_tensor.shape)}."
        )

    return LoadedGraph(
        values=graph_tensor[0].contiguous(),
        asset_cols=info.asset_cols,
        graph_type=str(graph_type),
        source=source,
        checkpoint_epoch=int(checkpoint.get("epoch", 0)),
        component=component,
    )


def _saved_validation_graph_path(run_dir: Path) -> Path:
    path = run_dir / "best_validation_graphs.pt"
    if not path.is_file():
        raise FileNotFoundError(
            "Saved validation graph artefacts were not found at "
            f"{path}."
        )
    return path


def _load_saved_validation_graph(
    info: RunInfo,
    *,
    component: GraphComponent,
    layer: int,
    window_index: int | None,
) -> LoadedGraph:
    artifact = _torch_load(_saved_validation_graph_path(info.run_dir))
    if not isinstance(artifact, Mapping):
        raise TypeError("best_validation_graphs.pt must contain a mapping.")

    graph_artifacts = artifact.get("graph_artifacts")
    if not isinstance(graph_artifacts, Mapping):
        raise KeyError(
            "best_validation_graphs.pt does not contain graph_artifacts."
        )

    artifact_assets = tuple(
        str(value) for value in graph_artifacts.get("asset_cols", [])
    )
    if artifact_assets and artifact_assets != info.asset_cols:
        raise ValueError(
            "Saved graph artefact asset order differs from run_metadata."
        )

    if component == "selected" and layer != -1:
        per_layer = graph_artifacts.get("per_layer")
        if not isinstance(per_layer, Sequence):
            raise KeyError("Saved graph artefacts do not contain per_layer.")
        layer_index = _normalise_layer_index(layer, len(per_layer))
        values = per_layer[layer_index]
    else:
        values = graph_artifacts.get(component)

    if values is None:
        raise ValueError(
            f"Saved graph component {component!r} is unavailable for this run."
        )

    tensor = torch.as_tensor(values).detach().cpu().to(torch.float64)
    if tensor.ndim != 4:
        raise ValueError(
            "Saved validation graph must have shape [W, G, N, N]; "
            f"observed {tuple(tensor.shape)}."
        )

    if window_index is None:
        graph = tensor.mean(dim=0)
        source_label = "best_validation_mean"
    else:
        resolved_window = int(window_index)
        if resolved_window < 0:
            resolved_window += int(tensor.shape[0])
        if not 0 <= resolved_window < int(tensor.shape[0]):
            raise IndexError(
                f"window_index {window_index} is outside the saved "
                f"{int(tensor.shape[0])} validation windows."
            )
        graph = tensor[resolved_window]
        source_label = f"best_validation_window_{resolved_window}"

    return LoadedGraph(
        values=graph.contiguous(),
        asset_cols=info.asset_cols,
        graph_type=str(graph_artifacts.get("graph_type", info.graph_type)),
        source=source_label,
        checkpoint_epoch=int(artifact.get("epoch", 0)),
        component=component,
    )


def load_learned_graph(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "last",
    component: GraphComponent = "selected",
    layer: int = -1,
    window_index: int | None = None,
) -> LoadedGraph:
    """Load a graph from a run directory.

    Parameters
    ----------
    source:
        ``"last"`` reads the global graph from ``last_checkpoint.pt``.
        This is appropriate for free-static, fixed and MTGNN-static runs.

        ``"best"`` reads the global graph from ``best_checkpoint.pt``.

        ``"best_validation"`` reads ``best_validation_graphs.pt``. For
        dynamic graphs, ``window_index=None`` returns the mean selected graph
        over all saved validation windows; an integer selects one window.

    component:
        ``"selected"`` is always the graph supplied to spatial message
        passing. ``"base"`` and ``"dynamic"`` are available only when the
        run saved or exposes them.
    """

    info = load_run_info(run_dir)

    if source == "best_validation":
        return _load_saved_validation_graph(
            info,
            component=component,
            layer=layer,
            window_index=window_index,
        )

    if window_index is not None:
        raise ValueError(
            "window_index is only valid with source='best_validation'."
        )

    return _load_static_checkpoint_graph(
        info,
        source=source,
        component=component,
        layer=layer,
    )


def _select_graph_head(graph: Tensor, head: HeadSelection) -> Tensor:
    values = torch.as_tensor(graph).detach().cpu().to(torch.float64)
    if values.ndim != 3:
        raise ValueError("Graph must have shape [G, N, N].")

    if head == "mean":
        return values.mean(dim=0)

    head_index = int(head)
    if head_index < 0:
        head_index += int(values.shape[0])
    if not 0 <= head_index < int(values.shape[0]):
        raise IndexError(
            f"Graph head {head} is outside {int(values.shape[0])} heads."
        )
    return values[head_index]


def _cluster_graph_order(
    adjacency: np.ndarray,
    *,
    method: str = "average",
) -> np.ndarray:
    """Cluster a directed graph using its symmetrised relationship strength.

    The clustering is used only to order the heatmap. The plotted matrix
    remains directed and is not symmetrised.
    """

    num_nodes = int(adjacency.shape[0])
    if num_nodes <= 1:
        return np.arange(num_nodes)

    similarity = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(similarity, 0.0)

    maximum = float(np.nanmax(similarity))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.arange(num_nodes)

    similarity = np.clip(similarity / maximum, 0.0, 1.0)
    distance = 1.0 - similarity
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(distance, checks=False)
    return leaves_list(linkage(condensed, method=method))


def plot_learned_graph(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "last",
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    window_index: int | None = None,
    cluster: bool = True,
    cluster_method: str = "average",
    plot_adjacency: bool = True,
    figsize: tuple[float, float] = (13.0, 11.0),
    tick_fontsize: float = 8.0,
    title_fontsize: float = 12.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot a learned graph with all asset names visible.

    Graph orientation is always ``A[target, source]``: rows are receiving
    target assets and columns are influencing source assets.

    Parameters
    ----------
    plot_adjacency:
        When ``True`` (the default), display the actual non-negative
        row-stochastic adjacency supplied to spatial message passing. White
        represents zero and darker red represents a larger edge weight.

        When ``False``, display each edge relative to uniform routing over
        the eligible sources. Red means above uniform, white means uniform,
        and blue means below uniform. Blue does *not* mean a negative edge.

    Notes
    -----
    Clustering changes only the display order. The directed adjacency values
    themselves are not symmetrised or modified.
    """

    info = load_run_info(run_dir)
    loaded = load_learned_graph(
        run_dir,
        source=source,
        component=component,
        layer=layer,
        window_index=window_index,
    )
    adjacency = _select_graph_head(loaded.values, head)

    matrix = adjacency.numpy().astype(np.float64, copy=True)
    labels = np.asarray(loaded.asset_cols, dtype=object)
    num_nodes = int(matrix.shape[0])

    if matrix.shape != (num_nodes, num_nodes):
        raise ValueError("Selected graph head must be square.")

    if cluster:
        order = _cluster_graph_order(
            matrix,
            method=cluster_method,
        )
        matrix = matrix[np.ix_(order, order)]
        labels = labels[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if plot_adjacency:
        values_to_plot = matrix.copy()

        if not info.add_self_loops:
            np.fill_diagonal(values_to_plot, np.nan)

        finite = values_to_plot[np.isfinite(values_to_plot)]
        maximum = float(np.max(finite)) if finite.size else 1.0
        if not np.isfinite(maximum) or maximum <= 0.0:
            maximum = 1.0

        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("white")
        image = ax.imshow(
            values_to_plot,
            cmap=cmap,
            vmin=0.0,
            vmax=maximum,
            interpolation="nearest",
            aspect="equal",
        )
        colourbar_label = "Adjacency weight"
        display_mode = "adjacency"

    else:
        eligible_sources = (
            num_nodes
            if info.add_self_loops
            else num_nodes - 1
        )
        if eligible_sources <= 0:
            raise ValueError(
                "A deviation-from-uniform plot requires at least one "
                "eligible source per target."
            )

        neutral = 1.0 / eligible_sources
        values_to_plot = matrix.copy() - neutral

        if not info.add_self_loops:
            np.fill_diagonal(values_to_plot, np.nan)

        finite = values_to_plot[np.isfinite(values_to_plot)]
        limit = (
            float(np.max(np.abs(finite)))
            if finite.size
            else 1.0
        )
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0

        cmap = plt.get_cmap("bwr").copy()
        cmap.set_bad("white")
        image = ax.imshow(
            values_to_plot,
            cmap=cmap,
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0.0,
                vmax=limit,
            ),
            interpolation="nearest",
            aspect="equal",
        )
        colourbar_label = (
            "Adjacency weight minus uniform weight"
        )
        display_mode = "deviation from uniform"

    ax.set_xticks(np.arange(num_nodes))
    ax.set_yticks(np.arange(num_nodes))
    ax.set_xticklabels(
        labels,
        rotation=90,
        ha="center",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        labels,
        fontsize=tick_fontsize,
    )
    ax.set_xlabel("Source asset (influences target)")
    ax.set_ylabel("Target asset (receives influence)")

    head_label = (
        "mean across heads"
        if head == "mean"
        else f"head {int(head)}"
    )
    cluster_label = (
        "clustered"
        if cluster
        else "original asset order"
    )
    ax.set_title(
        f"{Path(run_dir).expanduser().name}\n"
        f"{loaded.component} graph — {loaded.source} — "
        f"{head_label} — {cluster_label} — {display_mode}",
        fontsize=title_fontsize,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.03,
    )
    colorbar.set_label(colourbar_label)
    fig.tight_layout()

    plotted = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    plotted.index.name = "Target"
    plotted.columns.name = "Source"
    return fig, ax, plotted

def _load_best_metrics_long_table(run_path: Path) -> tuple[pd.DataFrame, int | None]:
    metric_path = run_path / "best_validation_metric_table.csv"
    if not metric_path.is_file():
        raise FileNotFoundError(metric_path)

    long_table = pd.read_csv(metric_path)

    checkpoint_epoch: int | None = None
    checkpoint_path = run_path / "best_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(checkpoint_path)
        if isinstance(checkpoint, Mapping) and checkpoint.get("epoch") is not None:
            checkpoint_epoch = int(checkpoint["epoch"])

    return long_table, checkpoint_epoch


def _load_last_metrics_long_table(
    run_path: Path,
    *,
    channel: str,
) -> tuple[pd.DataFrame, int]:
    if str(channel) != "close":
        raise ValueError(
            "Last-checkpoint metrics are reconstructed from history.csv, "
            "whose logged metric columns use the close-only evaluation "
            "contract. Use channel='close' or source='best'."
        )

    history_path = run_path / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)

    checkpoint = _torch_load(run_path / "last_checkpoint.pt")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("last_checkpoint.pt must contain a mapping.")
    checkpoint_epoch = int(checkpoint["epoch"])

    history = pd.read_csv(history_path)
    metric_columns: list[tuple[str, str, int]] = []

    for column in history.columns:
        if not isinstance(column, str) or not column.startswith("val/"):
            continue

        prefix, separator, horizon_text = column.rpartition("/h")
        if not separator or not horizon_text.isdigit():
            continue

        metric_name = prefix[len("val/"):]
        metric_columns.append((column, metric_name, int(horizon_text)))

    if not metric_columns:
        raise ValueError(
            "history.csv contains no per-horizon validation metric columns."
        )

    epoch_values = pd.to_numeric(history["epoch"], errors="coerce")
    eligible = history.loc[epoch_values <= checkpoint_epoch].copy()

    column_names = [column for column, _, _ in metric_columns]
    numeric_metrics = eligible[column_names].apply(
        pd.to_numeric,
        errors="coerce",
    )
    decoded_rows = eligible.loc[numeric_metrics.notna().any(axis=1)]
    if decoded_rows.empty:
        raise ValueError(
            "history.csv contains no decoded validation row at or before "
            f"last checkpoint epoch {checkpoint_epoch}."
        )

    row = decoded_rows.sort_values("epoch").iloc[-1]
    epoch = int(row["epoch"])

    records: list[dict[str, Any]] = []
    for column, metric_name, horizon in metric_columns:
        value = pd.to_numeric(
            pd.Series([row[column]]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(value):
            continue
        records.append(
            {
                "metric": metric_name,
                "horizon": int(horizon),
                "channel": "close",
                "value": float(value),
            }
        )

    return pd.DataFrame.from_records(records), epoch


def make_metrics_table(
    run_dir: str | Path,
    *,
    source: MetricsSource = "best",
    channel: str = "close",
) -> pd.DataFrame:
    """Return best- or last-validation metrics in baseline-table format.

    ``source='best'`` reads ``best_validation_metric_table.csv``.
    ``source='last'`` reconstructs the most recent decoded validation metrics
    from ``history.csv``.
    """

    run_path = _resolve_run_dir(run_dir)

    if source == "best":
        long_table, epoch = _load_best_metrics_long_table(run_path)
    elif source == "last":
        long_table, epoch = _load_last_metrics_long_table(
            run_path,
            channel=channel,
        )
    else:
        raise ValueError("source must be 'best' or 'last'.")

    required = {"metric", "horizon", "channel", "value"}
    missing = required - set(long_table.columns)
    if missing:
        raise ValueError(
            f"Metric table is missing columns: {sorted(missing)}."
        )

    selected = long_table.loc[
        long_table["channel"].astype(str) == str(channel),
        ["metric", "horizon", "value"],
    ].copy()
    if selected.empty:
        available = sorted(long_table["channel"].astype(str).unique())
        raise ValueError(
            f"Channel {channel!r} is unavailable. Available channels: {available}."
        )

    observed_metrics = list(pd.unique(selected["metric"]))
    preferred = [
        metric
        for metric in DEFAULT_METRIC_DISPLAY_NAMES
        if metric in observed_metrics
    ]
    extras = [metric for metric in observed_metrics if metric not in preferred]
    metric_order = preferred + extras

    table = selected.pivot(index="horizon", columns="metric", values="value")
    table = table.reindex(columns=metric_order).sort_index()
    table = table.rename(
        columns={
            metric: DEFAULT_METRIC_DISPLAY_NAMES.get(metric, metric)
            for metric in metric_order
        }
    )
    table.index = [f"{int(horizon)} min" for horizon in table.index]
    table.index.name = "Horizon"
    table.columns.name = None
    table.attrs["metrics_source"] = source
    table.attrs["epoch"] = epoch
    return table


def style_metrics_table(
    run_dir: str | Path,
    *,
    source: MetricsSource = "best",
    channel: str = "close",
    caption: str | None = None,
) -> pd.io.formats.style.Styler:
    """Return a clean pandas Styler matching the baseline summary style."""

    table = make_metrics_table(
        run_dir,
        source=source,
        channel=channel,
    )
    run_name = _resolve_run_dir(run_dir).name
    epoch = table.attrs.get("epoch")
    epoch_label = f", epoch {epoch}" if epoch is not None else ""
    source_label = "Best" if source == "best" else "Last"

    return (
        table.style
        .format("{:.6g}", na_rep="—")
        .set_caption(
            caption
            or f"{run_name} — {source_label} Validation Results{epoch_label}"
        )
    )


def make_top_neighbours_table(
    run_dir: str | Path,
    *,
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
    source: CheckpointSource = "last",
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    window_index: int | None = None,
) -> pd.DataFrame:
    """Return the top positive-weight neighbours for every stock.

    ``direction='impacted_by'``
        For each stock treated as a target, rank the source stocks that
        influence it. This reads a row of ``A[target, source]``.

    ``direction='impacts'``
        For each stock treated as a source, rank the target stocks it
        influences. This reads a column of ``A[target, source]``.

    Zero-weight entries are not reported as neighbours. If a sparse graph has
    fewer than ``top_n`` positive edges for a stock, the remaining table cells
    are left blank.
    """

    loaded = load_learned_graph(
        run_dir,
        source=source,
        component=component,
        layer=layer,
        window_index=window_index,
    )
    adjacency = _select_graph_head(
        loaded.values,
        head,
    ).numpy()

    table = _top_neighbours_from_adjacency(
        adjacency,
        loaded.asset_cols,
        top_n=top_n,
        direction=direction,
    )
    table.attrs["source"] = loaded.source
    table.attrs["component"] = loaded.component
    table.attrs["head"] = head
    return table

def style_top_neighbours_table(
    run_dir: str | Path,
    *,
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
    source: CheckpointSource = "last",
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    window_index: int | None = None,
    caption: str | None = None,
) -> pd.io.formats.style.Styler:
    """Return a formatted top-neighbours table."""

    table = make_top_neighbours_table(
        run_dir,
        top_n=top_n,
        direction=direction,
        source=source,
        component=component,
        layer=layer,
        head=head,
        window_index=window_index,
    )
    run_name = _resolve_run_dir(run_dir).name
    relation = (
        "Stocks that influence each stock"
        if direction == "impacted_by"
        else "Stocks influenced by each stock"
    )
    weight_columns = [column for column in table.columns if column.startswith("Weight ")]

    return (
        table.style
        .format({column: "{:.6f}" for column in weight_columns}, na_rep="—")
        .set_caption(caption or f"{run_name} — {relation}")
    )


def _prepare_correlation_split(
    split: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a clean 390-row candle split without modifying the input."""

    if not isinstance(split, Mapping):
        raise TypeError("split must be a candle-data split mapping.")

    required = {"samples", "asset_cols", "channels"}
    missing = required - set(split)
    if missing:
        raise KeyError(f"split is missing keys: {sorted(missing)}.")

    samples = split["samples"]
    if not isinstance(samples, Sequence) or len(samples) == 0:
        raise ValueError("split['samples'] must be a non-empty sequence.")

    first_sample = samples[0]
    if not isinstance(first_sample, Sequence) or len(first_sample) < 1:
        raise ValueError("Each split sample must contain a candle tensor.")

    first_tensor = torch.as_tensor(first_sample[0])
    if first_tensor.ndim != 3:
        raise ValueError(
            "Each candle tensor must have shape [T, N, C]."
        )

    observed_steps = int(first_tensor.shape[0])
    if observed_steps == 391:
        prepared = clean_candle_split(dict(split))
    elif observed_steps == 390:
        prepared = dict(split)
    else:
        raise ValueError(
            "Expected raw 391-row or clean 390-row sessions; "
            f"observed T={observed_steps}."
        )

    expected_nodes = len(prepared["asset_cols"])
    expected_channels = len(prepared["channels"])

    for sample_index, sample in enumerate(prepared["samples"]):
        values = torch.as_tensor(sample[0])
        expected_shape = (390, expected_nodes, expected_channels)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Sample {sample_index} has shape {tuple(values.shape)}, "
                f"expected {expected_shape}."
            )
        if not torch.isfinite(values.float()).all():
            raise ValueError(
                f"Sample {sample_index} contains non-finite values."
            )

    return prepared


def _top_neighbours_from_adjacency(
    adjacency: np.ndarray,
    labels: Sequence[str],
    *,
    top_n: int,
    direction: NeighbourDirection,
) -> pd.DataFrame:
    """Build the compact neighbour table for an in-memory adjacency."""

    if direction not in {"impacted_by", "impacts"}:
        raise ValueError("direction must be 'impacted_by' or 'impacts'.")

    matrix = np.asarray(adjacency, dtype=np.float64)
    names = [str(value) for value in labels]
    num_nodes = len(names)

    if matrix.shape != (num_nodes, num_nodes):
        raise ValueError(
            "adjacency shape does not match the supplied asset labels."
        )

    requested = int(top_n)
    if requested <= 0:
        raise ValueError("top_n must be positive.")
    count = min(requested, num_nodes - 1)

    records: list[dict[str, Any]] = []

    for stock_index, stock in enumerate(names):
        scores = (
            matrix[stock_index].copy()
            if direction == "impacted_by"
            else matrix[:, stock_index].copy()
        )
        scores[stock_index] = -np.inf

        positive_indices = np.flatnonzero(scores > 0.0)
        ranked = positive_indices[
            np.argsort(scores[positive_indices])[::-1]
        ][:count]

        record: dict[str, Any] = {"Stock": stock}
        for rank in range(1, count + 1):
            if rank <= len(ranked):
                neighbour_index = int(ranked[rank - 1])
                record[f"Neighbour {rank}"] = names[neighbour_index]
                record[f"Weight {rank}"] = float(scores[neighbour_index])
            else:
                record[f"Neighbour {rank}"] = None
                record[f"Weight {rank}"] = np.nan
        records.append(record)

    table = pd.DataFrame.from_records(records).set_index("Stock")
    table.attrs["direction"] = direction
    table.attrs["graph_orientation"] = "A[target, source]"
    return table


def analyse_absolute_correlation_graph(
    split: Mapping[str, Any],
    *,
    threshold: float,
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    empty_row_policy: EmptyCorrelationRowPolicy = "error",
    cluster: bool = True,
    cluster_method: str = "average",
    figsize: tuple[float, float] = (13.0, 11.0),
    tick_fontsize: float = 8.0,
    title_fontsize: float = 12.0,
    display_outputs: bool = True,
    plot_adjacency: bool = True,
) -> CorrelationGraphDiagnostics:
    """Build and inspect the production absolute-correlation fixed graph.

    The function performs the same transformation as
    :func:`build_absolute_correlation_adjacency`:

    1. calculate within-session one-step log returns for ``channel``;
    2. calculate the cross-asset Pearson correlation matrix;
    3. take absolute correlations and set the diagonal to zero;
    4. remove values strictly below ``threshold``;
    5. handle empty rows according to ``empty_row_policy``;
    6. divide every retained target row by its row sum.

    The production fixed-correlation learner uses L1 row normalisation at
    step 6, not a softmax over raw correlation values. Calling the production
    helper here ensures that this diagnostic and the eventual forecasting
    model use the same adjacency.

    Graph orientation is ``A[target, source]``. ``direction='impacted_by'``
    ranks each target row; ``direction='impacts'`` ranks each source column.

    ``plot_adjacency=True`` displays the actual non-negative row-stochastic
    adjacency: white is zero and darker red is a larger message-passing
    weight. ``plot_adjacency=False`` displays edge weights relative to a
    uniform allocation over the eligible non-self sources; blue then means
    below uniform, not a negative edge.
    """

    prepared = _prepare_correlation_split(split)

    correlation, asset_labels = compute_return_correlation_matrix(
        split=prepared,
        channel=channel,
        sample_indices=sample_indices,
        assets=assets,
    )

    labels = tuple(str(value) for value in asset_labels)
    if len(labels) < 2:
        raise ValueError(
            "Correlation-graph analysis requires at least two assets."
        )

    adjacency_4d = build_absolute_correlation_adjacency(
        torch.as_tensor(correlation, dtype=torch.float32),
        threshold=float(threshold),
        num_heads=1,
        add_self_loops=False,
        empty_row_policy=empty_row_policy,
    )
    adjacency_tensor = (
        adjacency_4d[0, 0]
        .detach()
        .cpu()
        .to(torch.float64)
    )

    if not torch.allclose(
        adjacency_tensor.sum(dim=-1),
        torch.ones(len(labels), dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Correlation adjacency rows do not sum to one."
        )

    diagonal = torch.diagonal(adjacency_tensor)
    if not torch.equal(
        diagonal,
        torch.zeros_like(diagonal),
    ):
        raise AssertionError(
            "Correlation adjacency diagonal is not zero."
        )

    positive = adjacency_tensor > 0.0
    row_entropy_tensor = -torch.where(
        positive,
        adjacency_tensor
        * torch.log(
            adjacency_tensor.clamp_min(1.0e-12)
        ),
        torch.zeros_like(adjacency_tensor),
    ).sum(dim=-1)
    retained_tensor = positive.sum(dim=-1)

    mean_row_entropy = float(
        row_entropy_tensor.mean().item()
    )
    mean_effective_neighbours = float(
        row_entropy_tensor.exp().mean().item()
    )

    adjacency = adjacency_tensor.numpy()
    correlation_array = np.asarray(
        correlation,
        dtype=np.float64,
    )
    display_labels = np.asarray(
        labels,
        dtype=object,
    )

    if cluster:
        _, _, order = reorder_correlation_matrix(
            correlation_array,
            list(labels),
            cluster_by_abs=True,
            method=cluster_method,
        )
        plot_matrix = adjacency[np.ix_(order, order)]
        plot_labels = display_labels[order]
    else:
        plot_matrix = adjacency.copy()
        plot_labels = display_labels.copy()

    num_nodes = len(labels)
    fig, ax = plt.subplots(figsize=figsize)

    if plot_adjacency:
        values_to_plot = plot_matrix.copy()
        np.fill_diagonal(values_to_plot, np.nan)

        finite = values_to_plot[np.isfinite(values_to_plot)]
        maximum = float(np.max(finite)) if finite.size else 1.0
        if not np.isfinite(maximum) or maximum <= 0.0:
            maximum = 1.0

        cmap = plt.get_cmap("Reds").copy()
        cmap.set_bad("white")
        image = ax.imshow(
            values_to_plot,
            cmap=cmap,
            vmin=0.0,
            vmax=maximum,
            interpolation="nearest",
            aspect="equal",
        )
        colourbar_label = "Adjacency weight"
        display_mode = "adjacency"

    else:
        neutral = 1.0 / (num_nodes - 1)
        values_to_plot = plot_matrix.copy() - neutral
        np.fill_diagonal(values_to_plot, np.nan)

        finite = values_to_plot[np.isfinite(values_to_plot)]
        limit = (
            float(np.max(np.abs(finite)))
            if finite.size
            else 1.0
        )
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0

        cmap = plt.get_cmap("bwr").copy()
        cmap.set_bad("white")
        image = ax.imshow(
            values_to_plot,
            cmap=cmap,
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0.0,
                vmax=limit,
            ),
            interpolation="nearest",
            aspect="equal",
        )
        colourbar_label = (
            "Adjacency weight minus uniform weight"
        )
        display_mode = "deviation from uniform"

    ax.set_xticks(np.arange(num_nodes))
    ax.set_yticks(np.arange(num_nodes))
    ax.set_xticklabels(
        plot_labels,
        rotation=90,
        ha="center",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        plot_labels,
        fontsize=tick_fontsize,
    )
    ax.set_xlabel("Source asset (influences target)")
    ax.set_ylabel("Target asset (receives influence)")
    ax.set_title(
        f"Absolute {channel} log-return graph ({display_mode})\n"
        f"threshold={float(threshold):.4g}; "
        f"mean row entropy={mean_row_entropy:.4f}",
        fontsize=title_fontsize,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.03,
    )
    colorbar.set_label(colourbar_label)
    fig.tight_layout()

    top_neighbours = _top_neighbours_from_adjacency(
        adjacency,
        labels,
        top_n=top_n,
        direction=direction,
    )

    row_entropy = pd.Series(
        row_entropy_tensor.numpy(),
        index=labels,
        name="Row entropy",
    )
    row_entropy.index.name = "Target"

    retained_neighbours = pd.Series(
        retained_tensor.numpy(),
        index=labels,
        name="Retained neighbours",
    )
    retained_neighbours.index.name = "Target"

    summary = pd.DataFrame(
        [
            {
                "Threshold": float(threshold),
                "Assets": num_nodes,
                "Mean retained neighbours": float(
                    retained_tensor
                    .to(torch.float64)
                    .mean()
                    .item()
                ),
                "Min retained neighbours": int(
                    retained_tensor.min().item()
                ),
                "Max retained neighbours": int(
                    retained_tensor.max().item()
                ),
                "Mean row entropy": mean_row_entropy,
                "Mean effective neighbours": (
                    mean_effective_neighbours
                ),
            }
        ]
    )

    correlation_frame = pd.DataFrame(
        correlation_array,
        index=labels,
        columns=labels,
    )
    correlation_frame.index.name = "Asset"
    correlation_frame.columns.name = "Asset"

    adjacency_frame = pd.DataFrame(
        adjacency,
        index=labels,
        columns=labels,
    )
    adjacency_frame.index.name = "Target"
    adjacency_frame.columns.name = "Source"

    result = CorrelationGraphDiagnostics(
        threshold=float(threshold),
        correlation_matrix=correlation_frame,
        adjacency=adjacency_frame,
        top_neighbours=top_neighbours,
        row_entropy=row_entropy,
        retained_neighbours=retained_neighbours,
        summary=summary,
        figure=fig,
        axes=ax,
    )

    if display_outputs:
        try:
            from IPython.display import display
        except ImportError as exc:
            raise RuntimeError(
                "display_outputs=True requires IPython. Set "
                "display_outputs=False outside a notebook."
            ) from exc

        plt.show()
        display(
            summary.style
            .format(
                {
                    "Threshold": "{:.4f}",
                    "Mean retained neighbours": "{:.2f}",
                    "Mean row entropy": "{:.4f}",
                    "Mean effective neighbours": "{:.2f}",
                }
            )
            .set_caption(
                "Absolute-correlation graph summary"
            )
        )

        weight_columns = [
            column
            for column in top_neighbours.columns
            if column.startswith("Weight ")
        ]
        relation = (
            "Stocks that influence each stock"
            if direction == "impacted_by"
            else "Stocks influenced by each stock"
        )
        display(
            top_neighbours.style
            .format(
                {
                    column: "{:.6f}"
                    for column in weight_columns
                },
                na_rep="—",
            )
            .set_caption(relation)
        )

    return result


__all__ = [
    "EVALUATION_MODULE_VERSION",
    "CorrelationGraphDiagnostics",
    "LoadedGraph",
    "RunInfo",
    "load_run_info",
    "load_learned_graph",
    "make_model_summary_table",
    "style_model_summary_table",
    "plot_learned_graph",
    "make_metrics_table",
    "style_metrics_table",
    "make_top_neighbours_table",
    "style_top_neighbours_table",
    "analyse_absolute_correlation_graph",
]
# ---------------------------------------------------------------------------
# Spatial-branch and graph-topology reliance diagnostics
# ---------------------------------------------------------------------------

GraphInterventionName = Literal[
    "actual",
    "zero_message",
    "uniform",
    "shuffled",
]


@dataclass(frozen=True)
class SpatialBranchRelianceDiagnostics:
    """Local diagnostics at one spatial residual layer.

    ``summary`` describes the distribution over all selected
    ``[window, context_time, asset]`` vectors. ``per_asset`` and
    ``per_context_position`` contain the corresponding means.
    """

    run_dir: Path
    checkpoint_source: str
    checkpoint_epoch: int | None
    block_index: int
    spatial_layer_index: int
    window_indices: tuple[int, ...]
    summary: pd.DataFrame
    per_asset: pd.DataFrame
    per_context_position: pd.DataFrame


@dataclass(frozen=True)
class GraphInterventionDiagnostics:
    """End-to-end token and optional decoded-price intervention results."""

    run_dir: Path
    checkpoint_source: str
    checkpoint_epoch: int | None
    block_index: int
    spatial_layer_index: int
    window_indices: tuple[int, ...]
    summary: pd.DataFrame
    per_horizon: pd.DataFrame
    decoded: bool


_TOKENIZER_CACHE: dict[tuple[str, str, int], KronosTokenizerAdapter] = {}


def _resolve_diagnostic_device(
    device: str | torch.device | None,
) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    elif torch.cuda.is_available():
        resolved = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        resolved = torch.device("mps")
    else:
        resolved = torch.device("cpu")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")

    return resolved


def _diagnostic_autocast(
    device: torch.device,
    use_amp: bool,
):
    if use_amp and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def _resolve_validation_cache_path(
    info: RunInfo,
    validation_cache_path: str | Path | None,
) -> Path:
    candidates: list[Path] = []

    if validation_cache_path is not None:
        candidates.append(
            Path(validation_cache_path).expanduser()
        )

    recorded = info.run_metadata.get("validation_cache_path")
    if recorded:
        recorded_path = Path(str(recorded)).expanduser()
        candidates.append(recorded_path)

        # Colab paths recorded in run metadata do not exist on the local
        # machine. In the canonical directory layout the run lives under
        # final_model/initial_test/<run>, while token caches live under
        # final_model/tokens/.
        candidates.append(
            info.run_dir.parent.parent
            / "tokens"
            / recorded_path.name
        )

    candidates.append(
        info.run_dir.parent.parent
        / "tokens"
        / "origin_aligned_val_tokens.pt"
    )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "Could not locate the validation token cache. Checked:\n"
        + "\n".join(checked)
        + "\nPass validation_cache_path explicitly if the cache is elsewhere."
    )


def _resolve_forecasting_config_path(
    info: RunInfo,
    forecasting_config_path: str | Path | None,
) -> Path:
    candidates: list[Path] = []

    if forecasting_config_path is not None:
        candidates.append(
            Path(forecasting_config_path).expanduser()
        )

    recorded = info.run_metadata.get("forecasting_config_path")
    if recorded:
        candidates.append(
            Path(str(recorded)).expanduser()
        )

    candidates.append(
        Path(__file__).resolve().parents[2]
        / "configs"
        / "forecasting.yaml"
    )

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        "Could not locate forecasting.yaml. Checked:\n"
        + "\n".join(checked)
        + "\nPass forecasting_config_path explicitly if it is elsewhere."
    )


def _load_validation_dataset_for_run(
    info: RunInfo,
    *,
    validation_cache_path: str | Path | None,
) -> CachedTokenGraphDataset:
    path = _resolve_validation_cache_path(
        info,
        validation_cache_path,
    )
    dataset = CachedTokenGraphDataset.from_path(
        path,
        data_mode=str(info.run_metadata.get("data_mode", "auto")),
    )

    if dataset.asset_cols != info.asset_cols:
        raise ValueError(
            "Validation-cache asset order differs from run_metadata.asset_cols."
        )
    if dataset.num_assets != info.num_nodes:
        raise ValueError(
            "Validation-cache asset count differs from the saved model."
        )

    return dataset


def _select_diagnostic_window_indices(
    dataset: CachedTokenGraphDataset,
    *,
    window_indices: Sequence[int] | None,
    max_windows: int | None,
) -> tuple[int, ...]:
    if window_indices is not None:
        selected = tuple(int(value) for value in window_indices)
        if not selected:
            raise ValueError("window_indices must not be empty.")
    else:
        if max_windows is None or int(max_windows) >= len(dataset):
            selected = tuple(range(len(dataset)))
        else:
            count = int(max_windows)
            if count <= 0:
                raise ValueError("max_windows must be positive or None.")
            # Even spacing covers the full validation period rather than
            # taking only the first sessions.
            selected = tuple(
                int(value)
                for value in np.linspace(
                    0,
                    len(dataset) - 1,
                    num=count,
                    dtype=np.int64,
                )
            )

    if len(set(selected)) != len(selected):
        raise ValueError("Selected validation window indices are not unique.")
    if min(selected) < 0 or max(selected) >= len(dataset):
        raise IndexError(
            "A diagnostic window index lies outside the validation cache."
        )

    return selected


def _build_diagnostic_loader(
    dataset: CachedTokenGraphDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _checkpoint_epoch(
    info: RunInfo,
    source: CheckpointSource,
) -> int | None:
    checkpoint = _torch_load(
        _checkpoint_path(info.run_dir, source)
    )
    if not isinstance(checkpoint, Mapping):
        return None
    value = checkpoint.get("epoch")
    return None if value is None else int(value)


def _normalise_block_and_spatial_layer(
    model: DynamicGraphTokenForecaster,
    *,
    block_index: int,
    spatial_layer_index: int,
) -> tuple[int, int, Any]:
    if model.config.graph.type == "none":
        raise ValueError(
            "Graph-reliance diagnostics require a model with a spatial graph."
        )

    block = _normalise_layer_index(
        int(block_index),
        len(model.spatial_blocks),
    )
    spatial_block = model.spatial_blocks[block]

    if not hasattr(spatial_block, "layers"):
        raise TypeError(
            "The selected spatial block does not expose message-passing layers."
        )

    layer = _normalise_layer_index(
        int(spatial_layer_index),
        len(spatial_block.layers),
    )

    return block, layer, spatial_block.layers[layer]


def _feature_scale(values: Tensor) -> Tensor:
    values = values.float()
    centred = values - values.mean(dim=-1, keepdim=True)
    return centred.square().mean(dim=-1).sqrt()


def _feature_cosine(first: Tensor, second: Tensor) -> Tensor:
    first = first.float()
    second = second.float()
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    return F.cosine_similarity(first, second, dim=-1, eps=1.0e-8)


def _distribution_summary(
    values: Tensor,
    *,
    metric: str,
) -> dict[str, float | str]:
    flat = values.detach().float().reshape(-1).cpu()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {
            "Metric": metric,
            "Mean": float("nan"),
            "Median": float("nan"),
            "P10": float("nan"),
            "P90": float("nan"),
        }

    quantiles = torch.quantile(
        finite,
        torch.tensor([0.10, 0.50, 0.90]),
    )
    return {
        "Metric": metric,
        "Mean": float(finite.mean().item()),
        "Median": float(quantiles[1].item()),
        "P10": float(quantiles[0].item()),
        "P90": float(quantiles[2].item()),
    }


def analyse_spatial_branch_reliance(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "best",
    validation_cache_path: str | Path | None = None,
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = 64,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    block_index: int = 0,
    spatial_layer_index: int = 0,
) -> SpatialBranchRelianceDiagnostics:
    """Measure temporal-residual and graph-message scale before mixing.

    The clean interpretation "own history versus cross-asset message" applies
    to the first ST block when graph self-loops are disabled. In later blocks,
    the residual input can already contain cross-node information introduced
    by earlier spatial blocks.
    """

    info = load_run_info(run_dir)
    dataset = _load_validation_dataset_for_run(
        info,
        validation_cache_path=validation_cache_path,
    )
    selected_indices = _select_diagnostic_window_indices(
        dataset,
        window_indices=window_indices,
        max_windows=max_windows,
    )
    loader = _build_diagnostic_loader(
        dataset,
        selected_indices,
        batch_size=batch_size,
    )

    resolved_device = _resolve_diagnostic_device(device)
    active_amp = (
        bool(info.run_metadata.get("active_cuda_amp", False))
        if use_amp is None
        else bool(use_amp)
    )
    active_amp = active_amp and resolved_device.type == "cuda"

    model, _, _ = _load_saved_model(info, source=source)
    model = model.to(resolved_device).eval()
    block, layer_number, spatial_layer = _normalise_block_and_spatial_layer(
        model,
        block_index=block_index,
        spatial_layer_index=spatial_layer_index,
    )

    collected: dict[str, list[Tensor]] = {
        "Temporal feature scale": [],
        "Graph branch feature scale": [],
        "Bias-free graph signal scale": [],
        "Graph / temporal scale ratio": [],
        "Temporal–graph cosine": [],
        "Post-mix LayerNorm graph effect RMS": [],
        "Full spatial-layer graph effect RMS": [],
    }

    def capture_components(
        module: torch.nn.Module,
        inputs: tuple[Tensor, Tensor],
        output: Tensor,
    ) -> None:
        hidden, adjacency = inputs
        batch_count, time_count, node_count, hidden_dim = hidden.shape

        values = (
            module.value_projection(hidden)
            .view(
                batch_count,
                time_count,
                node_count,
                module.num_heads,
                module.head_dim,
            )
            .permute(0, 1, 3, 2, 4)
        )
        head_messages = torch.einsum(
            "bgij,btgjd->btgid",
            adjacency,
            values,
        )
        joined = (
            head_messages
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_count, time_count, node_count, hidden_dim)
        )
        projected_message = module.message_dropout(
            module.output_projection(joined)
        )

        # Bias-free signal isolates the part that depends on routed source
        # representations rather than the value/output projection biases.
        bias_free_values = F.linear(
            hidden,
            module.value_projection.weight,
            bias=None,
        )
        bias_free_values = (
            bias_free_values
            .view(
                batch_count,
                time_count,
                node_count,
                module.num_heads,
                module.head_dim,
            )
            .permute(0, 1, 3, 2, 4)
        )
        bias_free_messages = torch.einsum(
            "bgij,btgjd->btgid",
            adjacency,
            bias_free_values,
        )
        bias_free_joined = (
            bias_free_messages
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_count, time_count, node_count, hidden_dim)
        )
        bias_free_signal = F.linear(
            bias_free_joined,
            module.output_projection.weight,
            bias=None,
        )

        temporal_scale = _feature_scale(hidden)
        message_scale = _feature_scale(projected_message)
        bias_free_scale = _feature_scale(bias_free_signal)
        scale_ratio = message_scale / temporal_scale.clamp_min(1.0e-8)
        branch_cosine = _feature_cosine(hidden, projected_message)

        full_mixed = module.mix_norm(hidden + projected_message)
        residual_mixed = module.mix_norm(hidden)
        post_mix_effect = (
            full_mixed.float() - residual_mixed.float()
        ).square().mean(dim=-1).sqrt()

        residual_only_output = module.feedforward_norm(
            residual_mixed + module.feedforward(residual_mixed)
        )
        full_layer_effect = (
            output.float() - residual_only_output.float()
        ).square().mean(dim=-1).sqrt()

        tensors = {
            "Temporal feature scale": temporal_scale,
            "Graph branch feature scale": message_scale,
            "Bias-free graph signal scale": bias_free_scale,
            "Graph / temporal scale ratio": scale_ratio,
            "Temporal–graph cosine": branch_cosine,
            "Post-mix LayerNorm graph effect RMS": post_mix_effect,
            "Full spatial-layer graph effect RMS": full_layer_effect,
        }

        for name, values_for_metric in tensors.items():
            collected[name].append(values_for_metric.detach().cpu())


    hook = spatial_layer.register_forward_hook(capture_components)
    try:
        with torch.inference_mode():
            for batch in loader:
                context_tokens = batch["context_tokens"].to(
                    resolved_device,
                    dtype=torch.long,
                )
                with _diagnostic_autocast(resolved_device, active_amp):
                    model.generate(
                        context_tokens,
                        token_selection="argmax",
                    )
    finally:
        hook.remove()

    summary_rows: list[dict[str, float | str]] = []
    for name, parts in collected.items():
        values = torch.cat(
            [part.reshape(-1) for part in parts],
            dim=0,
        )
        summary_rows.append(
            _distribution_summary(values, metric=name)
        )

    summary = pd.DataFrame(summary_rows).set_index("Metric")

    concatenated = {
        name: torch.cat(parts, dim=0).float()
        for name, parts in collected.items()
    }

    per_asset = pd.DataFrame(
        {
            name: values.mean(dim=(0, 1)).numpy()
            for name, values in concatenated.items()
        },
        index=info.asset_cols,
    )
    per_asset.index.name = "Asset"

    context_length = next(iter(concatenated.values())).shape[1]
    per_context_position = pd.DataFrame(
        {
            name: values.mean(dim=(0, 2)).numpy()
            for name, values in concatenated.items()
        },
        index=np.arange(1, context_length + 1),
    )
    per_context_position.index.name = "Context minute"

    return SpatialBranchRelianceDiagnostics(
        run_dir=info.run_dir,
        checkpoint_source=str(source),
        checkpoint_epoch=_checkpoint_epoch(info, source),
        block_index=block,
        spatial_layer_index=layer_number,
        window_indices=selected_indices,
        summary=summary,
        per_asset=per_asset,
        per_context_position=per_context_position,
    )


def _load_tokenizer_for_diagnostics(
    info: RunInfo,
    dataset: CachedTokenGraphDataset,
    *,
    forecasting_config_path: str | Path | None,
    series_batch_size: int,
) -> KronosTokenizerAdapter:
    config_path = _resolve_forecasting_config_path(
        info,
        forecasting_config_path,
    )
    forecasting_config = load_yaml(config_path)
    kronos_config = forecasting_config["models"]["kronos"]

    expected_id = dataset.cache.get("tokenizer_id")
    expected_revision = dataset.cache.get("tokenizer_revision")
    if expected_id is not None and expected_id != kronos_config["tokenizer_id"]:
        raise ValueError(
            "Validation cache and forecasting.yaml use different tokenizer IDs."
        )
    if (
        expected_revision is not None
        and expected_revision != kronos_config["tokenizer_revision"]
    ):
        raise ValueError(
            "Validation cache and forecasting.yaml use different tokenizer revisions."
        )

    key = (
        str(kronos_config["tokenizer_id"]),
        str(kronos_config["tokenizer_revision"]),
        int(series_batch_size),
    )
    if key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[key] = KronosTokenizerAdapter.from_config(
            forecasting_config,
            series_batch_size=int(series_batch_size),
        ).load()

    return _TOKENIZER_CACHE[key]


def _invalid_decoded_candle_mask(decoded_ohlcv: Tensor) -> Tensor:
    open_values = decoded_ohlcv[..., 0]
    high_values = decoded_ohlcv[..., 1]
    low_values = decoded_ohlcv[..., 2]
    close_values = decoded_ohlcv[..., 3]
    volume_values = decoded_ohlcv[..., 4]

    return (
        ~torch.isfinite(decoded_ohlcv).all(dim=-1)
        | (open_values <= 0)
        | (high_values <= 0)
        | (low_values <= 0)
        | (close_values <= 0)
        | (high_values < torch.maximum(open_values, close_values))
        | (low_values > torch.minimum(open_values, close_values))
        | (high_values < low_values)
        | (volume_values < 0)
    )


def _decode_intervention_tokens(
    *,
    model: DynamicGraphTokenForecaster,
    tokenizer: KronosTokenizerAdapter,
    batch: Mapping[str, Any],
    context_tokens_cpu: Tensor,
    generated_token_ids_cpu: Tensor,
    evaluation_indices: Tensor,
    series_batch_size: int,
) -> tuple[Tensor, Tensor, int, int]:
    if model.config.heads.predicts_s2:
        decoded = tokenizer.decode_token_path(
            context_tokens_cpu,
            generated_token_ids_cpu,
            mean=torch.as_tensor(batch["context_mean"]),
            std=torch.as_tensor(batch["context_std"]),
            series_batch_size=int(series_batch_size),
            return_full_path=False,
        )
    else:
        decoded = tokenizer.decode_coarse_token_path(
            context_tokens_cpu,
            generated_token_ids_cpu[..., 0],
            mean=torch.as_tensor(batch["context_mean"]),
            std=torch.as_tensor(batch["context_std"]),
            series_batch_size=int(series_batch_size),
            return_full_path=False,
        )

    decoded = decoded.to(torch.float32)
    invalid = _invalid_decoded_candle_mask(decoded)
    evaluation = decoded.index_select(dim=1, index=evaluation_indices)

    y_pred = evaluation[..., 3:4]
    y_true = torch.as_tensor(batch["evaluation_true"]).to(torch.float32)[
        ..., 3:4
    ]
    last_context = torch.as_tensor(batch["last_context_target"]).to(
        torch.float32
    )[..., 3:4]

    predicted_change = raw_to_cumulative_log_change(
        y_raw=y_pred,
        last_context_target=last_context,
    )
    true_change = raw_to_cumulative_log_change(
        y_raw=y_true,
        last_context_target=last_context,
    )
    absolute_error = (predicted_change - true_change).abs()

    return (
        absolute_error.sum(dim=(0, 2, 3)).to(torch.float64),
        torch.full(
            (absolute_error.shape[1],),
            float(absolute_error.shape[0] * absolute_error.shape[2]),
            dtype=torch.float64,
        ),
        int(invalid.sum().item()),
        int(invalid.numel()),
    )


def _new_intervention_accumulator(
    num_horizons: int,
) -> dict[str, Any]:
    return {
        "logit_squared_sum": 0.0,
        "logit_count": 0,
        "kl_sum": 0.0,
        "kl_count": 0,
        "argmax_change_count": 0,
        "token_count": 0,
        "correct_count": 0,
        "horizon_correct": torch.zeros(num_horizons, dtype=torch.float64),
        "horizon_argmax_change": torch.zeros(
            num_horizons,
            dtype=torch.float64,
        ),
        "horizon_token_count": torch.zeros(
            num_horizons,
            dtype=torch.float64,
        ),
        "clg_absolute_error": torch.zeros(
            num_horizons,
            dtype=torch.float64,
        ),
        "clg_count": torch.zeros(num_horizons, dtype=torch.float64),
        "invalid_count": 0,
        "invalid_total": 0,
    }


def _rowwise_shuffle_adjacency(
    adjacency: Tensor,
    *,
    add_self_loops: bool,
    seed: int,
) -> Tensor:
    num_nodes = int(adjacency.shape[-1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    gather_indices = torch.empty(
        (num_nodes, num_nodes),
        dtype=torch.long,
    )
    all_sources = torch.arange(num_nodes, dtype=torch.long)

    for target in range(num_nodes):
        off_diagonal = all_sources[all_sources != target]
        shuffled = off_diagonal[
            torch.randperm(off_diagonal.numel(), generator=generator)
        ]
        mapping = all_sources.clone()
        mapping[off_diagonal] = shuffled
        if add_self_loops:
            mapping[target] = target
        else:
            mapping[target] = target
        gather_indices[target] = mapping

    view_shape = [1] * (adjacency.ndim - 2) + [num_nodes, num_nodes]
    indices = gather_indices.to(adjacency.device).view(view_shape)
    indices = indices.expand_as(adjacency)
    shuffled_adjacency = torch.gather(adjacency, dim=-1, index=indices)

    if not add_self_loops:
        diagonal = torch.arange(num_nodes, device=adjacency.device)
        shuffled_adjacency[..., diagonal, diagonal] = 0.0
        shuffled_adjacency = shuffled_adjacency / shuffled_adjacency.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)

    return shuffled_adjacency


def _uniform_adjacency_like(
    adjacency: Tensor,
    *,
    add_self_loops: bool,
) -> Tensor:
    num_nodes = int(adjacency.shape[-1])
    if add_self_loops:
        return torch.full_like(adjacency, 1.0 / num_nodes)

    values = torch.ones_like(adjacency)
    diagonal = torch.arange(num_nodes, device=adjacency.device)
    values[..., diagonal, diagonal] = 0.0
    return values / float(num_nodes - 1)


def _install_spatial_intervention(
    spatial_layer: torch.nn.Module,
    *,
    intervention: str,
    gamma: float | None,
    add_self_loops: bool,
    shuffle_seed: int,
):
    handles: list[Any] = []

    if intervention == "actual":
        return handles

    if intervention == "zero_message" or gamma is not None:
        scale = 0.0 if intervention == "zero_message" else float(gamma)

        def scale_output(
            module: torch.nn.Module,
            inputs: tuple[Tensor, ...],
            output: Tensor,
        ) -> Tensor:
            del module, inputs
            return output * scale

        handles.append(
            spatial_layer.output_projection.register_forward_hook(scale_output)
        )
        return handles

    if intervention in {"uniform", "shuffled"}:

        def replace_adjacency(
            module: torch.nn.Module,
            inputs: tuple[Tensor, Tensor],
        ) -> tuple[Tensor, Tensor]:
            del module
            hidden, adjacency = inputs
            if intervention == "uniform":
                replacement = _uniform_adjacency_like(
                    adjacency,
                    add_self_loops=add_self_loops,
                )
            else:
                replacement = _rowwise_shuffle_adjacency(
                    adjacency,
                    add_self_loops=add_self_loops,
                    seed=shuffle_seed,
                )
            return hidden, replacement

        handles.append(
            spatial_layer.register_forward_pre_hook(replace_adjacency)
        )
        return handles

    raise ValueError(f"Unsupported graph intervention {intervention!r}.")


def _remove_hooks(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.remove()


def _run_graph_intervention_diagnostics(
    run_dir: str | Path,
    *,
    source: CheckpointSource,
    interventions: Sequence[tuple[str, str, float | None]],
    validation_cache_path: str | Path | None,
    forecasting_config_path: str | Path | None,
    window_indices: Sequence[int] | None,
    max_windows: int | None,
    batch_size: int,
    device: str | torch.device | None,
    use_amp: bool | None,
    block_index: int,
    spatial_layer_index: int,
    decode: bool,
    decode_series_batch_size: int,
    shuffle_seed: int,
) -> GraphInterventionDiagnostics:
    info = load_run_info(run_dir)
    dataset = _load_validation_dataset_for_run(
        info,
        validation_cache_path=validation_cache_path,
    )
    if decode and dataset.data_mode != "real":
        raise ValueError("Decoded price diagnostics require a real-data cache.")

    selected_indices = _select_diagnostic_window_indices(
        dataset,
        window_indices=window_indices,
        max_windows=max_windows,
    )
    loader = _build_diagnostic_loader(
        dataset,
        selected_indices,
        batch_size=batch_size,
    )

    resolved_device = _resolve_diagnostic_device(device)
    active_amp = (
        bool(info.run_metadata.get("active_cuda_amp", False))
        if use_amp is None
        else bool(use_amp)
    )
    active_amp = active_amp and resolved_device.type == "cuda"

    model, _, _ = _load_saved_model(info, source=source)
    model = model.to(resolved_device).eval()
    block, layer_number, spatial_layer = _normalise_block_and_spatial_layer(
        model,
        block_index=block_index,
        spatial_layer_index=spatial_layer_index,
    )

    horizons = tuple(int(value) for value in dataset.evaluation_horizons)
    evaluation_indices = torch.tensor(
        dataset.evaluation_indices,
        dtype=torch.long,
    )
    num_horizons = len(horizons)
    if num_horizons == 0:
        raise ValueError("Validation cache does not define evaluation horizons.")

    tokenizer = (
        _load_tokenizer_for_diagnostics(
            info,
            dataset,
            forecasting_config_path=forecasting_config_path,
            series_batch_size=decode_series_batch_size,
        )
        if decode
        else None
    )

    accumulators = {
        label: _new_intervention_accumulator(num_horizons)
        for label, _, _ in interventions
    }

    with torch.inference_mode():
        for batch in loader:
            context_tokens_cpu = torch.as_tensor(
                batch["context_tokens"],
                dtype=torch.long,
            )
            target_s1_cpu = torch.as_tensor(
                batch["target_s1"],
                dtype=torch.long,
            )
            context_tokens = context_tokens_cpu.to(resolved_device)

            # The unmodified model is the common reference for every
            # intervention in this batch.
            with _diagnostic_autocast(resolved_device, active_amp):
                reference_generated = model.generate(
                    context_tokens,
                    token_selection="argmax",
                )
            reference_logits = (
                reference_generated.forecast.s1_logits.detach().float()
            )
            reference_ids = (
                reference_generated.token_ids[..., 0].detach().cpu()
            )
            reference_log_probability = F.log_softmax(
                reference_logits,
                dim=-1,
            )
            reference_probability = reference_log_probability.exp()

            for label, intervention, gamma in interventions:
                if intervention == "actual":
                    generated = reference_generated
                else:
                    handles = _install_spatial_intervention(
                        spatial_layer,
                        intervention=intervention,
                        gamma=gamma,
                        add_self_loops=info.add_self_loops,
                        shuffle_seed=shuffle_seed,
                    )
                    try:
                        with _diagnostic_autocast(resolved_device, active_amp):
                            generated = model.generate(
                                context_tokens,
                                token_selection="argmax",
                            )
                    finally:
                        _remove_hooks(handles)

                logits = generated.forecast.s1_logits.detach().float()
                selected_ids_cpu = generated.token_ids[..., 0].detach().cpu()
                accumulator = accumulators[label]

                logit_difference = logits - reference_logits
                accumulator["logit_squared_sum"] += float(
                    logit_difference.square().sum().item()
                )
                accumulator["logit_count"] += int(logit_difference.numel())

                candidate_log_probability = F.log_softmax(logits, dim=-1)
                kl_values = (
                    reference_probability
                    * (reference_log_probability - candidate_log_probability)
                ).sum(dim=-1)
                accumulator["kl_sum"] += float(kl_values.sum().item())
                accumulator["kl_count"] += int(kl_values.numel())

                differences = selected_ids_cpu != reference_ids
                accumulator["argmax_change_count"] += int(
                    differences.sum().item()
                )
                accumulator["token_count"] += int(differences.numel())

                correct = selected_ids_cpu == target_s1_cpu
                accumulator["correct_count"] += int(correct.sum().item())

                horizon_ids = selected_ids_cpu.index_select(
                    dim=1,
                    index=evaluation_indices,
                )
                reference_horizon_ids = reference_ids.index_select(
                    dim=1,
                    index=evaluation_indices,
                )
                target_horizon_ids = target_s1_cpu.index_select(
                    dim=1,
                    index=evaluation_indices,
                )

                accumulator["horizon_correct"] += (
                    (horizon_ids == target_horizon_ids)
                    .sum(dim=(0, 2))
                    .to(torch.float64)
                )
                accumulator["horizon_argmax_change"] += (
                    (horizon_ids != reference_horizon_ids)
                    .sum(dim=(0, 2))
                    .to(torch.float64)
                )
                accumulator["horizon_token_count"] += float(
                    horizon_ids.shape[0] * horizon_ids.shape[2]
                )

                if decode:
                    if tokenizer is None:
                        raise AssertionError("Tokenizer was not loaded.")
                    generated_ids_cpu = generated.token_ids.detach().cpu()
                    (
                        absolute_error,
                        error_count,
                        invalid_count,
                        invalid_total,
                    ) = _decode_intervention_tokens(
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        context_tokens_cpu=context_tokens_cpu,
                        generated_token_ids_cpu=generated_ids_cpu,
                        evaluation_indices=evaluation_indices,
                        series_batch_size=decode_series_batch_size,
                    )
                    accumulator["clg_absolute_error"] += absolute_error
                    accumulator["clg_count"] += error_count
                    accumulator["invalid_count"] += invalid_count
                    accumulator["invalid_total"] += invalid_total

    summary_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []

    for label, _, gamma in interventions:
        accumulator = accumulators[label]
        row: dict[str, Any] = {
            "Intervention": label,
            "Gamma": gamma,
            "s1 accuracy": (
                accumulator["correct_count"]
                / max(accumulator["token_count"], 1)
            ),
            "s1 argmax change vs actual": (
                accumulator["argmax_change_count"]
                / max(accumulator["token_count"], 1)
            ),
            "s1 logit RMS delta vs actual": (
                accumulator["logit_squared_sum"]
                / max(accumulator["logit_count"], 1)
            ) ** 0.5,
            "Mean KL(actual || intervention)": (
                accumulator["kl_sum"]
                / max(accumulator["kl_count"], 1)
            ),
        }

        clg_values = accumulator["clg_absolute_error"] / accumulator[
            "clg_count"
        ].clamp_min(1.0)

        if decode:
            for horizon_value in (1, 5):
                if horizon_value in horizons:
                    index = horizons.index(horizon_value)
                    row[f"h{horizon_value} CLG-MAE"] = float(
                        clg_values[index].item()
                    )
            if 1 in horizons and 5 in horizons:
                row["Mean h1/h5 CLG-MAE"] = float(
                    (
                        clg_values[horizons.index(1)]
                        + clg_values[horizons.index(5)]
                    ).item()
                    / 2.0
                )
            row["Invalid candle rate (%)"] = (
                100.0
                * accumulator["invalid_count"]
                / max(accumulator["invalid_total"], 1)
            )

        summary_rows.append(row)

        horizon_accuracy = accumulator["horizon_correct"] / accumulator[
            "horizon_token_count"
        ].clamp_min(1.0)
        horizon_change = accumulator[
            "horizon_argmax_change"
        ] / accumulator["horizon_token_count"].clamp_min(1.0)

        for index, horizon in enumerate(horizons):
            horizon_row: dict[str, Any] = {
                "Intervention": label,
                "Horizon": int(horizon),
                "s1 accuracy": float(horizon_accuracy[index].item()),
                "s1 argmax change vs actual": float(
                    horizon_change[index].item()
                ),
            }
            if decode:
                horizon_row["Decoded CLG-MAE"] = float(
                    clg_values[index].item()
                )
            horizon_rows.append(horizon_row)

    summary = pd.DataFrame(summary_rows).set_index("Intervention")
    per_horizon = pd.DataFrame(horizon_rows).set_index(
        ["Intervention", "Horizon"]
    )

    return GraphInterventionDiagnostics(
        run_dir=info.run_dir,
        checkpoint_source=str(source),
        checkpoint_epoch=_checkpoint_epoch(info, source),
        block_index=block,
        spatial_layer_index=layer_number,
        window_indices=selected_indices,
        summary=summary,
        per_horizon=per_horizon,
        decoded=bool(decode),
    )


def analyse_graph_gate_sweep(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "best",
    gammas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    validation_cache_path: str | Path | None = None,
    forecasting_config_path: str | Path | None = None,
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = 32,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    block_index: int = 0,
    spatial_layer_index: int = 0,
    decode: bool = False,
    decode_series_batch_size: int = 64,
) -> GraphInterventionDiagnostics:
    """Scale the trained graph-message branch from zero to full strength."""

    gamma_values = tuple(float(value) for value in gammas)
    if not gamma_values:
        raise ValueError("gammas must not be empty.")
    if any(value < 0.0 for value in gamma_values):
        raise ValueError("gammas must be non-negative.")
    if 1.0 not in gamma_values:
        raise ValueError("gammas must contain 1.0 as the trained reference.")

    interventions = tuple(
        (
            f"gamma={value:g}",
            "actual" if value == 1.0 else "gamma",
            value,
        )
        for value in gamma_values
    )

    return _run_graph_intervention_diagnostics(
        run_dir,
        source=source,
        interventions=interventions,
        validation_cache_path=validation_cache_path,
        forecasting_config_path=forecasting_config_path,
        window_indices=window_indices,
        max_windows=max_windows,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        block_index=block_index,
        spatial_layer_index=spatial_layer_index,
        decode=decode,
        decode_series_batch_size=decode_series_batch_size,
        shuffle_seed=0,
    )


def analyse_graph_topology_counterfactuals(
    run_dir: str | Path,
    *,
    source: CheckpointSource = "best",
    interventions: Sequence[GraphInterventionName] = (
        "actual",
        "zero_message",
        "uniform",
        "shuffled",
    ),
    validation_cache_path: str | Path | None = None,
    forecasting_config_path: str | Path | None = None,
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = 32,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    block_index: int = 0,
    spatial_layer_index: int = 0,
    decode: bool = False,
    decode_series_batch_size: int = 64,
    shuffle_seed: int = 42,
) -> GraphInterventionDiagnostics:
    """Compare the trained topology with generic or destroyed graph paths.

    ``shuffled`` preserves every row's multiset of weights and therefore its
    entropy, but reassigns those weights to different source assets.
    """

    names = tuple(str(value) for value in interventions)
    supported = {"actual", "zero_message", "uniform", "shuffled"}
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(f"Unsupported interventions: {unknown}.")
    if "actual" not in names:
        raise ValueError("interventions must include 'actual'.")

    definitions = tuple((name, name, None) for name in names)

    return _run_graph_intervention_diagnostics(
        run_dir,
        source=source,
        interventions=definitions,
        validation_cache_path=validation_cache_path,
        forecasting_config_path=forecasting_config_path,
        window_indices=window_indices,
        max_windows=max_windows,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        block_index=block_index,
        spatial_layer_index=spatial_layer_index,
        decode=decode,
        decode_series_batch_size=decode_series_batch_size,
        shuffle_seed=int(shuffle_seed),
    )


def plot_graph_gate_sweep(
    result: GraphInterventionDiagnostics,
    *,
    metric: str = "s1 argmax change vs actual",
    figsize: tuple[float, float] = (7.0, 4.0),
    marker: str = "o",
) -> tuple[Figure, Axes]:
    """Plot one gate-sweep metric against the graph-message multiplier."""

    if "Gamma" not in result.summary.columns:
        raise ValueError("The supplied result is not a graph-gate sweep.")
    if metric not in result.summary.columns:
        raise KeyError(
            f"Metric {metric!r} is not available. "
            f"Available={list(result.summary.columns)}."
        )

    values = result.summary.dropna(subset=["Gamma"]).sort_values("Gamma")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values["Gamma"], values[metric], marker=marker)
    ax.set_xlabel("Graph-message multiplier γ")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Graph-branch sensitivity — {result.run_dir.name}\n"
        f"checkpoint={result.checkpoint_source}, "
        f"epoch={result.checkpoint_epoch}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


# Final public export list. This appears at the end so the graph-reliance
# diagnostics added below the original evaluation helpers are exported too.
__all__ = [
    "EVALUATION_MODULE_VERSION",
    "RunInfo",
    "LoadedGraph",
    "CorrelationGraphDiagnostics",
    "SpatialBranchRelianceDiagnostics",
    "GraphInterventionDiagnostics",
    "GraphInterventionName",
    "load_run_info",
    "load_learned_graph",
    "make_model_summary_table",
    "style_model_summary_table",
    "plot_learned_graph",
    "make_metrics_table",
    "style_metrics_table",
    "make_top_neighbours_table",
    "style_top_neighbours_table",
    "analyse_absolute_correlation_graph",
    "analyse_spatial_branch_reliance",
    "analyse_graph_gate_sweep",
    "analyse_graph_topology_counterfactuals",
    "plot_graph_gate_sweep",
]


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

EVALUATION_MODULE_VERSION = "2026-08-05-v11-final-graph-hub"

import json
import re
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from src.models.dynamic_graph.config import build_model_config
from src.data.cached_token_graph_dataset import CachedTokenGraphDataset
from src.data.load_candle_data import (
    clean_candle_split,
    get_channel_index,
    load_candle_splits,
)
from src.models.dynamic_graph.graph_learners import (
    EmptyCorrelationRowPolicy,
    build_absolute_correlation_adjacency,
    build_graph_learner,
)
from src.models.dynamic_graph.model import DynamicGraphTokenForecaster
from src.models.kronos_tokenizer import KronosTokenizerAdapter
from src.evaluation.metrics import ForecastEvaluator
from src.evaluation.prediction_transforms import raw_to_cumulative_log_change
from src.utils.config import load_yaml
from src.utils.metric_tables import (
    DEFAULT_METRIC_DISPLAY_NAMES,
    DEFAULT_SUMMARY_METRICS,
    make_evaluation_table,
)
from src.utils.company_profiles import (
    make_asset_sector_mapping,
    make_sector_group_order,
    resolve_company_profiles_path,
)
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

    scale_center_key = "close_scale_embedding.feature_center"
    scale_scale_key = "close_scale_embedding.feature_scale"
    close_scale_feature_center = (
        torch.as_tensor(model_state[scale_center_key])
        if scale_center_key in model_state
        else None
    )
    close_scale_feature_scale = (
        torch.as_tensor(model_state[scale_scale_key])
        if scale_scale_key in model_state
        else None
    )

    model = DynamicGraphTokenForecaster.from_config(
        experiment_config,
        fixed_adjacency=fixed_adjacency,
        oracle_graph=oracle_graph,
        close_scale_feature_center=close_scale_feature_center,
        close_scale_feature_scale=close_scale_feature_scale,
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

    if graph_type in {"dynamic", "dynamic_correlation", "oracle"}:
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


def _augment_best_metrics_from_saved_predictions(
    run_path: Path,
    long_table: pd.DataFrame,
    *,
    requested_metrics: Sequence[str],
) -> pd.DataFrame:
    """Recompute missing best-checkpoint metrics from saved predictions.

    Older runs predate some evaluation-only metrics.  Their exact saved
    prediction paths are authoritative and can be re-evaluated without
    regenerating tokens or touching model weights.  MASE is excluded from
    this fallback because it requires the training-derived scale.
    """
    observed = set(
        long_table.get(
            "metric",
            pd.Series(dtype=str),
        )
        .astype(str)
        .tolist()
    )

    missing = [
        metric_name
        for metric_name in requested_metrics
        if metric_name not in observed
    ]

    if not missing:
        return long_table

    prediction_path = (
        run_path
        / "best_validation_predictions.pt"
    )

    if not prediction_path.is_file():
        return long_table

    payload = _torch_load(
        prediction_path
    )

    if not isinstance(payload, Mapping):
        raise TypeError(
            "best_validation_predictions.pt must contain a mapping."
        )

    prediction_result = payload.get(
        "prediction_result"
    )

    if not isinstance(
        prediction_result,
        Mapping,
    ):
        raise KeyError(
            "best_validation_predictions.pt does not contain a "
            "prediction_result mapping."
        )

    evaluator = ForecastEvaluator(
        prediction_result=dict(
            prediction_result
        )
    )

    backfillable_metrics = {
        "cumulative_log_change_median_absolute_error",
        "cumulative_log_change_p95_absolute_error",
        "cumulative_log_change_directional_accuracy",
    }

    recomputable = [
        metric_name
        for metric_name in missing
        if (
            metric_name in backfillable_metrics
            and metric_name in evaluator.available_metrics
        )
    ]

    if not recomputable:
        return long_table

    results = evaluator.evaluate(
        metrics=recomputable,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    additional = make_evaluation_table(
        metric_results=results,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    combined = pd.concat(
        [
            long_table,
            additional,
        ],
        ignore_index=True,
    )

    return combined.drop_duplicates(
        subset=[
            "metric",
            "horizon",
            "channel",
        ],
        keep="first",
    )


def make_metrics_table(
    run_dir: str | Path,
    *,
    source: MetricsSource = "best",
    channel: str = "close",
    metrics_to_display: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return best- or last-validation metrics in headline-table format.

    ``source='best'`` reads the saved validation table and, for older
    runs, recomputes any missing evaluation-only metrics from the exact
    saved prediction path. ``source='last'`` reconstructs metrics logged
    in ``history.csv``; metrics absent from an older history are shown as
    missing rather than regenerated from a different checkpoint.

    The default metric list includes mean, median and 95th-percentile
    cumulative-log-change absolute error and omits MASE from the compact
    display.  MASE remains implemented and can be restored by passing it
    explicitly in ``metrics_to_display``.
    """
    run_path = _resolve_run_dir(
        run_dir
    )

    if metrics_to_display is None:
        metric_order = list(
            DEFAULT_SUMMARY_METRICS
        )
    else:
        metric_order = [
            str(metric_name)
            for metric_name in metrics_to_display
        ]

    if not metric_order:
        raise ValueError(
            "metrics_to_display must contain at least one metric."
        )

    duplicate_metrics = {
        metric_name
        for metric_name in metric_order
        if metric_order.count(metric_name) > 1
    }

    if duplicate_metrics:
        raise ValueError(
            "metrics_to_display must not contain duplicates: "
            f"{sorted(duplicate_metrics)}."
        )

    if source == "best":
        long_table, epoch = (
            _load_best_metrics_long_table(
                run_path
            )
        )
        long_table = (
            _augment_best_metrics_from_saved_predictions(
                run_path,
                long_table,
                requested_metrics=metric_order,
            )
        )
    elif source == "last":
        long_table, epoch = (
            _load_last_metrics_long_table(
                run_path,
                channel=channel,
            )
        )
    else:
        raise ValueError(
            "source must be 'best' or 'last'."
        )

    required = {
        "metric",
        "horizon",
        "channel",
        "value",
    }
    missing = required - set(
        long_table.columns
    )

    if missing:
        raise ValueError(
            "Metric table is missing columns: "
            f"{sorted(missing)}."
        )

    selected = long_table.loc[
        long_table["channel"].astype(str)
        == str(channel),
        [
            "metric",
            "horizon",
            "value",
        ],
    ].copy()

    if selected.empty:
        available = sorted(
            long_table["channel"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            f"Channel {channel!r} is unavailable. "
            f"Available channels: {available}."
        )

    table = selected.pivot(
        index="horizon",
        columns="metric",
        values="value",
    )
    table = (
        table
        .reindex(
            columns=metric_order
        )
        .sort_index()
    )
    table = table.rename(
        columns={
            metric_name: (
                DEFAULT_METRIC_DISPLAY_NAMES.get(
                    metric_name,
                    metric_name,
                )
            )
            for metric_name in metric_order
        }
    )
    table.index = [
        f"{int(horizon)} min"
        for horizon in table.index
    ]
    table.index.name = "Horizon"
    table.columns.name = None
    table.attrs["metrics_source"] = source
    table.attrs["epoch"] = epoch
    table.attrs["metrics_to_display"] = tuple(
        metric_order
    )
    return table


def style_metrics_table(
    run_dir: str | Path,
    *,
    source: MetricsSource = "best",
    channel: str = "close",
    metrics_to_display: Sequence[str] | None = None,
    caption: str | None = None,
) -> pd.io.formats.style.Styler:
    """Return a clean pandas Styler matching the baseline summary."""
    table = make_metrics_table(
        run_dir,
        source=source,
        channel=channel,
        metrics_to_display=metrics_to_display,
    )
    run_name = _resolve_run_dir(
        run_dir
    ).name
    epoch = table.attrs.get(
        "epoch"
    )
    epoch_label = (
        f", epoch {epoch}"
        if epoch is not None
        else ""
    )
    source_label = (
        "Best"
        if source == "best"
        else "Last"
    )

    formatters = {
        column: (
            "{:.2%}"
            if column == "Sign Acc."
            else "{:.6g}"
        )
        for column in table.columns
    }

    return (
        table.style
        .format(
            formatters,
            na_rep="—",
        )
        .set_caption(
            caption
            or (
                f"{run_name} — {source_label} "
                f"Validation Results{epoch_label}"
            )
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
    dataset: CachedTokenGraphDataset,
    batch: Mapping[str, Any],
    context_tokens_cpu: Tensor,
    generated_token_ids_cpu: Tensor,
    evaluation_indices: Tensor,
    series_batch_size: int,
) -> tuple[Tensor, Tensor, int, int]:
    # The trainable model may operate in a compact retained-s1 ID space
    # (for example 0...149), whereas the frozen Kronos decoder always
    # expects native coarse-token IDs in 0...1023.  Convert both the
    # observed context and generated future back to native IDs before
    # either full or coarse-only decoding.  For ordinary 1024-token
    # caches this method is an identity operation.
    context_tokens_for_decode = context_tokens_cpu.clone().to(torch.long)
    generated_tokens_for_decode = (
        generated_token_ids_cpu.clone().to(torch.long)
    )

    context_tokens_for_decode[..., 0] = dataset.s1_to_kronos_ids(
        context_tokens_for_decode[..., 0]
    )
    generated_tokens_for_decode[..., 0] = dataset.s1_to_kronos_ids(
        generated_tokens_for_decode[..., 0]
    )

    if model.config.heads.predicts_s2:
        decoded = tokenizer.decode_token_path(
            context_tokens_for_decode,
            generated_tokens_for_decode,
            mean=torch.as_tensor(batch["context_mean"]),
            std=torch.as_tensor(batch["context_std"]),
            series_batch_size=int(series_batch_size),
            return_full_path=False,
        )
    else:
        decoded = tokenizer.decode_coarse_token_path(
            context_tokens_for_decode,
            generated_tokens_for_decode[..., 0],
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
                        dataset=dataset,
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

def analyse_token_predictive_distribution(
    run_dir: str | Path,
    *,
    asset: str | int | None = None,
    horizon: int | None = None,
    source: MetricsSource = "best",
    validation_cache_path: str | Path | None = None,
    training_cache_path: str | Path | None = None,
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    plot_top_n: int = 40,
    figsize: tuple[float, float] = (14.0, 5.0),
) -> dict[str, Any]:
    """Analyse free-running coarse-token probabilities and hard predictions.

    ``asset=None`` pools all assets; otherwise pass a ticker or zero-based
    asset index. ``horizon=None`` pools all future minutes; otherwise pass a
    one-based future minute in ``[1, prediction_length]``.

    Probabilities are the raw temperature-1 softmax of generated ``s1``
    logits. Hard-token frequencies use argmax generation. Training-target
    frequencies use ``target_s1`` over the same asset/horizon scope.
    """

    if source not in {"best", "last"}:
        raise ValueError("source must be 'best' or 'last'.")

    if batch_size <= 0 or plot_top_n <= 0:
        raise ValueError(
            "batch_size and plot_top_n must be positive."
        )

    info = load_run_info(
        run_dir
    )

    if asset is None:
        asset_index = None
        asset_label = "All assets"

    elif (
        isinstance(
            asset,
            int,
        )
        and not isinstance(
            asset,
            bool,
        )
    ):
        asset_index = int(
            asset
        )

        if not 0 <= asset_index < info.num_nodes:
            raise IndexError(
                f"Asset index {asset_index} is out of range."
            )

        asset_label = info.asset_cols[
            asset_index
        ]

    elif isinstance(
        asset,
        str,
    ):
        if asset not in info.asset_cols:
            raise KeyError(
                f"Unknown asset {asset!r}."
            )

        asset_index = info.asset_cols.index(
            asset
        )
        asset_label = asset

    else:
        raise TypeError(
            "asset must be None, a ticker, "
            "or an integer index."
        )

    validation_dataset = (
        _load_validation_dataset_for_run(
            info,
            validation_cache_path=(
                validation_cache_path
            ),
        )
    )

    selected_indices = (
        _select_diagnostic_window_indices(
            validation_dataset,
            window_indices=window_indices,
            max_windows=max_windows,
        )
    )

    validation_loader = (
        _build_diagnostic_loader(
            validation_dataset,
            selected_indices,
            batch_size=batch_size,
        )
    )

    model, _, _ = _load_saved_model(
        info,
        source=source,
    )

    prediction_length = int(
        model.config.prediction_length
    )

    vocabulary_size = int(
        model.config.heads.s1_vocabulary_size
    )

    if horizon is None:
        horizon_index = None
        horizon_label: int | str = (
            f"All {prediction_length} future minutes"
        )

    else:
        horizon = int(
            horizon
        )

        if not 1 <= horizon <= prediction_length:
            raise ValueError(
                "horizon must lie in "
                f"[1, {prediction_length}]."
            )

        horizon_index = horizon - 1
        horizon_label = horizon

    training_candidates: list[
        Path
    ] = []

    if training_cache_path is not None:
        training_candidates.append(
            Path(
                training_cache_path
            ).expanduser()
        )

    recorded_train = (
        info.run_metadata.get(
            "train_cache_path"
        )
    )

    if recorded_train:
        recorded_path = Path(
            str(
                recorded_train
            )
        ).expanduser()

        training_candidates.extend(
            [
                recorded_path,
                (
                    info.run_dir.parent.parent
                    / "tokens"
                    / recorded_path.name
                ),
            ]
        )

    training_candidates.append(
        (
            info.run_dir.parent.parent
            / "tokens"
            / "origin_aligned_train_tokens.pt"
        )
    )

    resolved_training_path = next(
        (
            candidate.resolve()
            for candidate in training_candidates
            if candidate.resolve().is_file()
        ),
        None,
    )

    if resolved_training_path is None:
        raise FileNotFoundError(
            "Could not locate the training token cache. "
            "Pass training_cache_path explicitly."
        )

    training_dataset = (
        CachedTokenGraphDataset.from_path(
            resolved_training_path,
            data_mode=str(
                info.run_metadata.get(
                    "data_mode",
                    "auto",
                )
            ),
        )
    )

    if (
        training_dataset.asset_cols
        != info.asset_cols
    ):
        raise ValueError(
            "Training-cache asset order differs "
            "from the run."
        )

    if (
        training_dataset.prediction_length
        != prediction_length
    ):
        raise ValueError(
            "Training-cache prediction length "
            "differs from the run."
        )

    # Match the training-token frequency scope to the
    # selected validation prediction scope.
    training_targets = (
        training_dataset.target_s1
    )

    if horizon_index is not None:
        training_targets = training_targets[
            :,
            horizon_index : horizon_index + 1,
        ]

    if asset_index is not None:
        training_targets = training_targets[
            :,
            :,
            asset_index : asset_index + 1,
        ]

    flat_training_targets = (
        training_targets
        .reshape(-1)
        .to(torch.long)
    )

    if (
        int(
            flat_training_targets.min()
        ) < 0
        or int(
            flat_training_targets.max()
        ) >= vocabulary_size
    ):
        raise ValueError(
            "Training targets are incompatible with "
            "the saved s1 vocabulary."
        )

    training_counts = torch.bincount(
        flat_training_targets,
        minlength=vocabulary_size,
    ).to(torch.float64)

    training_total = int(
        flat_training_targets.numel()
    )

    training_frequency = (
        training_counts
        / training_total
    )

    resolved_device = (
        _resolve_diagnostic_device(
            device
        )
    )

    active_amp = (
        bool(
            info.run_metadata.get(
                "active_cuda_amp",
                False,
            )
        )
        if use_amp is None
        else bool(
            use_amp
        )
    ) and resolved_device.type == "cuda"

    model = (
        model
        .to(resolved_device)
        .eval()
    )

    probability_sum = torch.zeros(
        vocabulary_size,
        dtype=torch.float64,
    )

    hard_counts = torch.zeros(
        vocabulary_size,
        dtype=torch.float64,
    )

    entropy_sum = 0.0
    maximum_probability_sum = 0.0
    validation_total = 0

    with torch.inference_mode():
        for batch in validation_loader:
            context_tokens = batch[
                "context_tokens"
            ].to(
                resolved_device
            )

            oracle_graph = None

            if (
                model.config.graph.type
                == "oracle"
            ):
                oracle_graph = batch[
                    "true_graph"
                ].to(
                    resolved_device
                )

            with _diagnostic_autocast(
                resolved_device,
                active_amp,
            ):
                generated = model.generate(
                    context_tokens,
                    oracle_graph=oracle_graph,
                    token_selection="argmax",
                )

            logits = (
                generated
                .forecast
                .s1_logits
            )

            hard_ids = (
                generated
                .token_ids[
                    ...,
                    0,
                ]
            )

            if horizon_index is not None:
                logits = logits[
                    :,
                    horizon_index : horizon_index + 1,
                ]

                hard_ids = hard_ids[
                    :,
                    horizon_index : horizon_index + 1,
                ]

            if asset_index is not None:
                logits = logits[
                    :,
                    :,
                    asset_index : asset_index + 1,
                ]

                hard_ids = hard_ids[
                    :,
                    :,
                    asset_index : asset_index + 1,
                ]

            flat_logits = (
                logits
                .reshape(
                    -1,
                    vocabulary_size,
                )
                .float()
            )

            flat_hard_ids = (
                hard_ids
                .reshape(-1)
                .to(torch.long)
            )

            probabilities = F.softmax(
                flat_logits,
                dim=-1,
            )

            log_probabilities = F.log_softmax(
                flat_logits,
                dim=-1,
            )

            probability_sum += (
                probabilities
                .sum(0)
                .cpu()
                .to(torch.float64)
            )

            hard_counts += (
                torch.bincount(
                    flat_hard_ids,
                    minlength=vocabulary_size,
                )
                .cpu()
                .to(torch.float64)
            )

            entropy_sum += float(
                (
                    -(
                        probabilities
                        * log_probabilities
                    )
                    .sum(-1)
                )
                .sum()
                .cpu()
            )

            maximum_probability_sum += float(
                probabilities
                .max(-1)
                .values
                .sum()
                .cpu()
            )

            validation_total += int(
                flat_logits.shape[0]
            )

    mean_probability = (
        probability_sum
        / validation_total
    )

    hard_frequency = (
        hard_counts
        / validation_total
    )

    def distribution_entropy(
        values: Tensor,
    ) -> float:
        positive = values > 0
        values = values[
            positive
        ]

        return float(
            -(
                values
                * values.log()
            )
            .sum()
            .item()
        )

    mean_position_entropy = (
        entropy_sum
        / validation_total
    )

    mean_distribution_entropy = (
        distribution_entropy(
            mean_probability
        )
    )

    hard_entropy = (
        distribution_entropy(
            hard_frequency
        )
    )

    predicted_mask = (
        hard_counts > 0
    )

    summary = pd.DataFrame(
        [
            {
                "Checkpoint": source,
                "Asset": asset_label,
                "Horizon": horizon_label,
                "Validation windows": (
                    len(
                        selected_indices
                    )
                ),
                "Validation predictions": (
                    validation_total
                ),
                "Training targets": (
                    training_total
                ),
                "Mean per-position entropy (nats)": (
                    mean_position_entropy
                ),
                "Mean per-position perplexity": (
                    np.exp(
                        mean_position_entropy
                    )
                ),
                "Entropy of mean distribution (nats)": (
                    mean_distribution_entropy
                ),
                "Perplexity of mean distribution": (
                    np.exp(
                        mean_distribution_entropy
                    )
                ),
                "Hard-prediction entropy (nats)": (
                    hard_entropy
                ),
                "Hard-prediction effective vocabulary": (
                    np.exp(
                        hard_entropy
                    )
                ),
                "Distinct hard-predicted tokens": (
                    int(
                        predicted_mask.sum()
                    )
                ),
                "Largest hard-prediction share (%)": (
                    float(
                        hard_frequency.max()
                        * 100.0
                    )
                ),
                "Mean maximum token probability (%)": (
                    100.0
                    * maximum_probability_sum
                    / validation_total
                ),
            }
        ]
    )

    # Only include tokens that won the hard argmax
    # at least once.
    predicted_ids = torch.nonzero(
        predicted_mask
    ).flatten()

    token_table = pd.DataFrame(
        {
            "Token ID": (
                predicted_ids.numpy()
            ),
            "Predicted Count": (
                hard_counts[
                    predicted_ids
                ]
                .to(torch.int64)
                .numpy()
            ),
            "Predicted Frequency (%)": (
                hard_frequency[
                    predicted_ids
                ]
                .numpy()
                * 100.0
            ),
            "Mean Predictive Probability (%)": (
                mean_probability[
                    predicted_ids
                ]
                .numpy()
                * 100.0
            ),
            "Training Target Count": (
                training_counts[
                    predicted_ids
                ]
                .to(torch.int64)
                .numpy()
            ),
            "Training Target Frequency (%)": (
                training_frequency[
                    predicted_ids
                ]
                .numpy()
                * 100.0
            ),
        }
    ).sort_values(
        [
            "Predicted Frequency (%)",
            "Token ID",
        ],
        ascending=[
            False,
            True,
        ],
        ignore_index=True,
    )

    ranked_ids = torch.argsort(
        mean_probability,
        descending=True,
    )

    shown = min(
        plot_top_n,
        vocabulary_size,
    )

    shown_ids = (
        ranked_ids[
            :shown
        ]
        .numpy()
    )

    x = np.arange(
        shown
    )

    width = 0.27

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.bar(
        x - width,
        (
            mean_probability[
                shown_ids
            ]
            .numpy()
            * 100.0
        ),
        width,
        label="Mean model probability",
    )

    ax.bar(
        x,
        (
            hard_frequency[
                shown_ids
            ]
            .numpy()
            * 100.0
        ),
        width,
        label="Hard prediction frequency",
    )

    ax.bar(
        x + width,
        (
            training_frequency[
                shown_ids
            ]
            .numpy()
            * 100.0
        ),
        width,
        label="Training target frequency",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        shown_ids,
        rotation=90,
    )

    ax.set_xlabel(
        "Coarse token ID"
    )

    ax.set_ylabel(
        "Percentage"
    )

    ax.set_title(
        "Coarse-token predictive distribution\n"
        f"asset={asset_label}; "
        f"horizon={horizon_label}; "
        f"checkpoint={source}"
    )

    ax.legend()

    fig.tight_layout()
    plt.close(fig)

    token_index = pd.Index(
        np.arange(
            vocabulary_size
        ),
        name="Token ID",
    )

    return {
        "summary": summary,
        "token_table": token_table,
        "mean_predictive_distribution": (
            pd.Series(
                mean_probability.numpy(),
                index=token_index,
                name=(
                    "Mean predictive probability"
                ),
            )
        ),
        "hard_prediction_distribution": (
            pd.Series(
                hard_frequency.numpy(),
                index=token_index,
                name=(
                    "Hard prediction frequency"
                ),
            )
        ),
        "training_target_distribution": (
            pd.Series(
                training_frequency.numpy(),
                index=token_index,
                name=(
                    "Training target frequency"
                ),
            )
        ),
        "validation_window_indices": (
            selected_indices
        ),
        "figure": fig,
        "axes": ax,
    }

def analyse_s1_topk_accuracy_by_horizon(
    run_dir: str | Path,
    *,
    asset: str | int | None = None,
    source: MetricsSource = "best",
    validation_cache_path: str | Path | None = None,
    training_cache_path: str | Path | None = None,
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    top_k_values: Sequence[int] = (1, 2, 5, 10),
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return free-running s1 Top-k accuracy and excess over a marginal baseline.

    The marginal baseline is fitted from training targets only. At each future
    horizon, under the same asset filter, it always proposes the k most frequent
    training tokens. The visible subcolumns are model accuracy and model accuracy
    minus marginal-baseline accuracy, in percentage points.
    """

    if source not in {"best", "last"}:
        raise ValueError("source must be 'best' or 'last'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    info = load_run_info(run_dir)

    if asset is None:
        asset_index, asset_label = None, "All assets"
    elif isinstance(asset, int) and not isinstance(asset, bool):
        asset_index = int(asset)
        if not 0 <= asset_index < info.num_nodes:
            raise IndexError(f"Asset index {asset_index} is out of range.")
        asset_label = info.asset_cols[asset_index]
    elif isinstance(asset, str):
        if asset not in info.asset_cols:
            raise KeyError(f"Unknown asset {asset!r}.")
        asset_index, asset_label = info.asset_cols.index(asset), asset
    else:
        raise TypeError(
            "asset must be None, a ticker, or a zero-based integer index."
        )

    validation_dataset = _load_validation_dataset_for_run(
        info,
        validation_cache_path=validation_cache_path,
    )
    selected_indices = _select_diagnostic_window_indices(
        validation_dataset,
        window_indices=window_indices,
        max_windows=max_windows,
    )
    validation_loader = _build_diagnostic_loader(
        validation_dataset,
        selected_indices,
        batch_size=batch_size,
    )

    model, _, _ = _load_saved_model(info, source=source)
    prediction_length = int(model.config.prediction_length)
    vocabulary_size = int(model.config.heads.s1_vocabulary_size)

    top_ks = tuple(int(value) for value in top_k_values)
    if not top_ks or len(set(top_ks)) != len(top_ks):
        raise ValueError("top_k_values must be non-empty and unique.")
    if any(k < 1 or k > vocabulary_size for k in top_ks):
        raise ValueError(f"top_k_values must lie in [1, {vocabulary_size}].")
    maximum_k = max(top_ks)

    selected_horizons = (
        tuple(range(1, prediction_length + 1))
        if horizons is None
        else tuple(int(value) for value in horizons)
    )
    if not selected_horizons or len(set(selected_horizons)) != len(
        selected_horizons
    ):
        raise ValueError("horizons must be non-empty and unique.")
    invalid_horizons = [
        value
        for value in selected_horizons
        if not 1 <= value <= prediction_length
    ]
    if invalid_horizons:
        raise ValueError(
            f"Horizons outside [1, {prediction_length}]: {invalid_horizons}."
        )
    selected_horizon_indices = [horizon - 1 for horizon in selected_horizons]

    # Resolve the training cache; the empirical ranking is fitted only on train.
    candidates: list[Path] = []
    if training_cache_path is not None:
        candidates.append(Path(training_cache_path).expanduser())

    recorded_train = info.run_metadata.get("train_cache_path")
    if recorded_train:
        recorded_path = Path(str(recorded_train)).expanduser()
        candidates.extend(
            [
                recorded_path,
                info.run_dir.parent.parent / "tokens" / recorded_path.name,
            ]
        )

    candidates.append(
        info.run_dir.parent.parent
        / "tokens"
        / "origin_aligned_train_tokens.pt"
    )

    checked: list[str] = []
    resolved_training_path: Path | None = None
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_file():
            resolved_training_path = candidate
            break

    if resolved_training_path is None:
        raise FileNotFoundError(
            "Could not locate the training token cache. Checked:\n"
            + "\n".join(checked)
        )

    training_dataset = CachedTokenGraphDataset.from_path(
        resolved_training_path,
        data_mode=str(info.run_metadata.get("data_mode", "auto")),
    )
    if training_dataset.asset_cols != info.asset_cols:
        raise ValueError("Training-cache asset order differs from the run.")
    if training_dataset.num_assets != info.num_nodes:
        raise ValueError("Training-cache asset count differs from the run.")
    if training_dataset.prediction_length != prediction_length:
        raise ValueError("Training-cache prediction length differs from the run.")

    training_targets = training_dataset.target_s1.to(torch.long)
    if asset_index is not None:
        training_targets = training_targets[
            :, :, asset_index : asset_index + 1
        ]
    if (
        int(training_targets.min()) < 0
        or int(training_targets.max()) >= vocabulary_size
    ):
        raise ValueError(
            "Training targets contain IDs outside the model s1 vocabulary."
        )

    # rank[h, token] is the token's zero-based training-frequency rank at h.
    # Ties are broken deterministically by ascending token ID.
    marginal_rank = torch.empty(
        (prediction_length, vocabulary_size),
        dtype=torch.long,
    )
    marginal_order = torch.empty_like(marginal_rank)
    token_ids = np.arange(vocabulary_size, dtype=np.int64)

    for horizon_index in range(prediction_length):
        counts = torch.bincount(
            training_targets[:, horizon_index, :].reshape(-1),
            minlength=vocabulary_size,
        ).cpu().numpy()
        order = torch.from_numpy(
            np.lexsort((token_ids, -counts)).copy()
        ).to(torch.long)
        marginal_order[horizon_index] = order
        marginal_rank[horizon_index, order] = torch.arange(vocabulary_size)

    resolved_device = _resolve_diagnostic_device(device)
    active_amp = (
        bool(info.run_metadata.get("active_cuda_amp", False))
        if use_amp is None
        else bool(use_amp)
    ) and resolved_device.type == "cuda"

    model = model.to(resolved_device).eval()
    marginal_rank = marginal_rank.to(resolved_device)

    count_shape = (prediction_length, len(top_ks))
    model_correct = torch.zeros(count_shape, dtype=torch.float32)
    marginal_correct = torch.zeros(count_shape, dtype=torch.float32)
    totals = torch.zeros(prediction_length, dtype=torch.float32)

    with torch.inference_mode():
        for batch in validation_loader:
            context_tokens = batch["context_tokens"].to(resolved_device)
            target_s1 = batch["target_s1"].to(resolved_device)
            oracle_graph = (
                batch["true_graph"].to(resolved_device)
                if model.config.graph.type == "oracle"
                else None
            )

            with _diagnostic_autocast(resolved_device, active_amp):
                generated = model.generate(
                    context_tokens,
                    oracle_graph=oracle_graph,
                    token_selection="argmax",
                )

            logits = generated.forecast.s1_logits
            if tuple(logits.shape[:3]) != tuple(target_s1.shape):
                raise ValueError(
                    "s1 logits and targets have incompatible shapes: "
                    f"{tuple(logits.shape)} vs {tuple(target_s1.shape)}."
                )
            if int(logits.shape[-1]) != vocabulary_size:
                raise ValueError("Unexpected s1 vocabulary dimension.")

            if asset_index is not None:
                logits = logits[:, :, asset_index : asset_index + 1, :]
                target_s1 = target_s1[
                    :, :, asset_index : asset_index + 1
                ]

            model_top_ids = torch.topk(
                logits.float(),
                k=maximum_k,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices
            model_matches = model_top_ids.eq(target_s1.unsqueeze(-1))

            target_marginal_rank = (
                marginal_rank
                .unsqueeze(0)
                .expand(target_s1.shape[0], -1, -1)
                .gather(dim=2, index=target_s1.to(torch.long))
            )

            totals += float(target_s1.shape[0] * target_s1.shape[2])

            for column_index, k in enumerate(top_ks):
                model_correct[:, column_index] += (
                    model_matches[..., :k]
                    .any(dim=-1)
                    .to(torch.float32)
                    .sum(dim=(0, 2))
                    .cpu()
                )
                marginal_correct[:, column_index] += (
                    (target_marginal_rank < k)
                    .to(torch.float32)
                    .sum(dim=(0, 2))
                    .cpu()
                )

    if (totals <= 0).any():
        raise RuntimeError("At least one horizon received no validation cases.")

    model_accuracy = model_correct / totals.unsqueeze(-1) * 100.0
    marginal_accuracy = marginal_correct / totals.unsqueeze(-1) * 100.0
    excess = model_accuracy - marginal_accuracy

    # Both Top-k curves must be non-decreasing as k increases.
    sorted_columns = sorted(enumerate(top_ks), key=lambda item: item[1])
    for (previous_index, previous_k), (current_index, current_k) in zip(
        sorted_columns,
        sorted_columns[1:],
    ):
        for name, values in (
            ("model", model_accuracy),
            ("marginal baseline", marginal_accuracy),
        ):
            if (
                values[:, current_index] + 1.0e-12
                < values[:, previous_index]
            ).any():
                raise AssertionError(
                    f"{name} Top-k accuracy decreased from "
                    f"k={previous_k} to k={current_k}."
                )

    model_selected = model_accuracy[selected_horizon_indices].numpy()
    marginal_selected = marginal_accuracy[selected_horizon_indices].numpy()
    excess_selected = excess[selected_horizon_indices].numpy()

    data: dict[tuple[str, str], np.ndarray] = {}
    for column_index, k in enumerate(top_ks):
        data[(f"Top-{k}", "Model Accuracy (%)")] = (
            model_selected[:, column_index]
        )
        data[(f"Top-{k}", "Excess vs Marginal (pp)")] = (
            excess_selected[:, column_index]
        )

    table = pd.DataFrame(
        data,
        index=pd.Index(
            selected_horizons,
            name="Future horizon (minutes)",
        ),
    )
    table.columns = pd.MultiIndex.from_tuples(
        table.columns,
        names=("Candidate set", "Metric"),
    )

    # Baseline values stay accessible without adding a third visible subcolumn.
    table.attrs.update(
        {
            "run_directory": str(info.run_dir),
            "checkpoint": source,
            "asset": asset_label,
            "validation_windows": len(selected_indices),
            "training_cache_path": str(resolved_training_path),
            "marginal_baseline_accuracy_percent": {
                int(horizon): {
                    f"Top-{k}": float(
                        marginal_selected[row_index, column_index]
                    )
                    for column_index, k in enumerate(top_ks)
                }
                for row_index, horizon in enumerate(selected_horizons)
            },
            "training_top_token_ids_by_horizon": {
                int(horizon): marginal_order[
                    horizon - 1, :maximum_k
                ].tolist()
                for horizon in selected_horizons
            },
        }
    )

    return table

# ---------------------------------------------------------------------------
# Unified final-model analysis API
# ---------------------------------------------------------------------------

AnalysisRunKind = Literal["continuous", "token"]
AnalysisPolicy = str


@dataclass(frozen=True)
class UnifiedRunInfo:
    """Validated metadata shared by continuous and tokenized final models."""

    run_dir: Path
    run_kind: AnalysisRunKind
    resolved_config: dict[str, Any]
    run_metadata: dict[str, Any]
    asset_cols: tuple[str, ...]
    horizons: tuple[int, ...]
    graph_type: str
    num_nodes: int
    num_heads: int
    num_heads_per_layer: tuple[int, ...] | None
    add_self_loops: bool
    selection_metric: str | None


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Saved prediction, graph, metric and optional Monte Carlo artefacts."""

    info: UnifiedRunInfo
    policy: str
    epoch: int | None
    prediction_result: dict[str, Any]
    graph_artifacts: dict[str, Any]
    metric_table: pd.DataFrame
    sampled_price_paths: dict[str, Any] | None
    policy_dir: Path


@dataclass(frozen=True)
class GraphSnapshot:
    """One graph matrix selected by date/window and optional graph head."""

    adjacency: pd.DataFrame
    run_name: str
    run_kind: AnalysisRunKind
    policy: str
    component: str
    head: HeadSelection
    global_window_index: int
    date: str | None
    window_within_date: int | None
    sample_idx: int | None
    origin_idx: int | None
    mean_row_entropy: float
    mean_effective_neighbours: float
    median_effective_neighbours: float
    maximum_edge_weight: float
    mean_top5_row_mass: float
    graph_type: str
    add_self_loops: bool


@dataclass(frozen=True)
class SampledPathBundle:
    """Validated decoded Monte Carlo Close-price paths."""

    run_dir: Path
    policy: str
    sampled_close_paths: Tensor  # [S, W, P, N, 1]
    sampled_close_paths_at_evaluation_horizons: Tensor  # [S, W, H, N, 1]
    ensemble_mean_close_path: Tensor  # [W, P, N, 1]
    evaluation_true: Tensor  # [W, H, N, 1]
    last_context_target: Tensor  # [W, N, 1]
    sample_idx: Tensor | None
    origin_idx: Tensor | None
    dense_target_indices: Tensor | None
    evaluation_target_indices: Tensor | None
    dates: tuple[str, ...]
    asset_cols: tuple[str, ...]
    future_steps: tuple[int, ...]
    evaluation_horizons: tuple[int, ...]
    temperature: float | None
    top_k: int
    top_p: float
    sample_count: int


def detect_run_kind(run_dir: str | Path) -> AnalysisRunKind:
    """Return ``continuous`` or ``token`` from the resolved config schema."""

    run_path = _resolve_run_dir(run_dir)
    resolved = _load_json(run_path / "resolved_config.json")

    models = resolved.get("models")
    model_family = str(resolved.get("model_family", "")).lower()
    data_values = resolved.get("data")
    token_schema = (
        isinstance(models, Mapping)
        and isinstance(models.get("dynamic_graph"), Mapping)
    )
    token_data = (
        isinstance(data_values, Mapping)
        and (
            data_values.get("input_token_stream") is not None
            or data_values.get("target_token_stream") is not None
        )
    )

    # New token-space runners retain a detailed top-level ``model`` mapping
    # for their own constructor while also writing the established
    # ``models.dynamic_graph`` analysis mirror.  Prefer the explicit token
    # identity before applying the historical continuous-schema fallback.
    if token_schema and ("token" in model_family or token_data):
        return "token"

    if isinstance(resolved.get("model"), Mapping):
        return "continuous"

    if token_schema:
        return "token"

    raise ValueError(
        "Could not identify the saved run family from resolved_config.json. "
        "Expected either top-level 'model' (continuous) or "
        "models.dynamic_graph (tokenized)."
    )


def _positive_int_tuple(
    values: Any,
    *,
    name: str,
) -> tuple[int, ...] | None:
    """Return an optional positive-integer schedule from saved metadata."""

    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of positive integers.")
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _resolve_graph_schema(
    *,
    model_values: Mapping[str, Any],
    graph_values: Mapping[str, Any],
    resolved: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> tuple[str, int, tuple[int, ...] | None]:
    """Resolve graph type/head metadata across historical model schemas.

    Graph Hub analyses saved graph tensors.  Configuration metadata is only
    used to describe those tensors, so a new runner must not become
    unreadable merely because it stores a per-block head schedule instead of
    the historical scalar ``model.graph.num_heads`` field.
    """

    schedule_sources = (
        (graph_values.get("num_heads_per_block"), "model.graph.num_heads_per_block"),
        (graph_values.get("num_heads_per_layer"), "model.graph.num_heads_per_layer"),
        (graph_values.get("layer_head_counts"), "model.graph.layer_head_counts"),
        (metadata.get("graph_heads_per_block"), "run_metadata.graph_heads_per_block"),
        (metadata.get("graph_heads_per_layer"), "run_metadata.graph_heads_per_layer"),
    )
    per_layer: tuple[int, ...] | None = None
    for raw_values, name in schedule_sources:
        if raw_values is not None:
            per_layer = _positive_int_tuple(raw_values, name=name)
            break
    if per_layer is None:
        per_block_values = resolved.get("per_block")
        if isinstance(per_block_values, Mapping):
            per_layer = _positive_int_tuple(
                per_block_values.get("num_edge_heads"),
                name="per_block.num_edge_heads",
            )

    scalar_heads = graph_values.get("num_heads")
    if scalar_heads is None:
        scalar_heads = metadata.get("graph_heads")
    if scalar_heads is None and per_layer is not None:
        scalar_heads = per_layer[-1]
    if scalar_heads is None:
        raise KeyError(
            "No graph-head metadata was found. Expected model.graph.num_heads "
            "or a per-block/per-layer graph-head schedule."
        )
    num_heads = int(scalar_heads)
    if num_heads <= 0:
        raise ValueError("The final graph-head count must be positive.")
    if per_layer is not None and per_layer[-1] != num_heads:
        raise ValueError(
            "The final per-layer graph-head count does not match the saved "
            f"scalar head count: {per_layer[-1]} vs {num_heads}."
        )

    graph_type_value = graph_values.get("type")
    if graph_type_value is None:
        graph_type_value = graph_values.get("graph_type")
    if graph_type_value is None:
        graph_family = model_values.get(
            "graph_family",
            metadata.get("graph_family"),
        )
        family_aliases = {
            "dynamic_only": "dynamic",
            "prior_state": "static_dynamic_mixture",
        }
        graph_type_value = family_aliases.get(
            str(graph_family),
            graph_family,
        )
    if graph_type_value is None:
        graph_type_value = metadata.get("graph_type", "unknown")

    return str(graph_type_value), num_heads, per_layer


def load_unified_run_info(run_dir: str | Path) -> UnifiedRunInfo:
    """Load run metadata without assuming a continuous or tokenized model."""

    run_path = _resolve_run_dir(run_dir)
    resolved = _load_json(run_path / "resolved_config.json")
    metadata = _load_json(run_path / "run_metadata.json")
    run_kind = detect_run_kind(run_path)

    asset_values = metadata.get("asset_cols")
    if not isinstance(asset_values, Sequence) or isinstance(asset_values, str):
        raise ValueError("run_metadata.asset_cols must be a sequence.")
    asset_cols = tuple(str(value) for value in asset_values)
    if len(set(asset_cols)) != len(asset_cols):
        raise ValueError("run_metadata.asset_cols contains duplicate names.")

    if run_kind == "continuous":
        model_values = resolved["model"]
        graph_values = model_values["graph"]

        # Historical continuous runs store their reported horizons at
        # ``data.horizons``.  The BaseDyGraph financial diagnostics must
        # distinguish the model's native one-step head from the horizons
        # reported after an autoregressive rollout, so they store the public
        # evaluation contract at ``data.reported_horizons`` instead.  Treat
        # both spellings as equivalent analysis metadata.  The final fallback
        # supports the earlier BaseDyGraph financial runner, whose exact task
        # configuration is also serialised under ``basedygraph_financial``.
        data_values = resolved.get("data", {})
        if not isinstance(data_values, Mapping):
            raise TypeError("resolved_config.data must be a mapping.")
        horizon_values = data_values.get("horizons")
        if horizon_values is None:
            horizon_values = data_values.get("reported_horizons")
        if horizon_values is None:
            basedygraph_values = resolved.get("basedygraph_financial", {})
            if isinstance(basedygraph_values, Mapping):
                horizon_values = basedygraph_values.get("evaluation_horizons")
        if horizon_values is None:
            raise KeyError(
                "The continuous resolved configuration contains no public "
                "evaluation horizons. Expected data.horizons, "
                "data.reported_horizons, or "
                "basedygraph_financial.evaluation_horizons."
            )

        num_nodes = len(asset_cols)
        selection_metric = str(
            resolved.get("training", {}).get("selection_metric", "")
        ) or None
    else:
        model_values = resolved["models"]["dynamic_graph"]
        graph_values = model_values["graph"]
        horizon_values = model_values["heads"]["evaluation_horizons"]
        num_nodes = int(model_values["num_nodes"])
        selection_metric = str(
            resolved.get("training", {}).get("early_stopping_metric", "")
        ) or None

    if len(asset_cols) != num_nodes:
        raise ValueError(
            "Saved asset order does not match the configured node count: "
            f"{len(asset_cols)} vs {num_nodes}."
        )

    horizons = tuple(int(value) for value in horizon_values)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Evaluation horizons must be positive integers.")

    graph_type, num_heads, num_heads_per_layer = _resolve_graph_schema(
        model_values=model_values,
        graph_values=graph_values,
        resolved=resolved,
        metadata=metadata,
    )

    return UnifiedRunInfo(
        run_dir=run_path,
        run_kind=run_kind,
        resolved_config=resolved,
        run_metadata=metadata,
        asset_cols=asset_cols,
        horizons=horizons,
        graph_type=graph_type,
        num_nodes=num_nodes,
        num_heads=num_heads,
        num_heads_per_layer=num_heads_per_layer,
        add_self_loops=bool(graph_values.get("add_self_loops", False)),
        selection_metric=selection_metric,
    )


def _resolve_analysis_policy(
    run_dir: Path,
    policy: AnalysisPolicy,
) -> tuple[str, Path, dict[str, Path]]:
    """Resolve best-checkpoint or temperature-policy artefact paths."""

    requested = str(policy).strip()
    if not requested:
        raise ValueError("policy cannot be empty.")

    if requested in {"best", "best_validation"}:
        return (
            "best",
            run_dir,
            {
                "predictions": run_dir / "best_validation_predictions.pt",
                "graphs": run_dir / "best_validation_graphs.pt",
                "metrics": run_dir / "best_validation_metric_table.csv",
                "sampled_paths": (
                    run_dir / "best_validation_sampled_price_paths.pt"
                ),
            },
        )

    temperature_root = run_dir / "temperature_sweep"
    if requested == "selected_temperature":
        selection = _load_json(temperature_root / "temperature_selection.json")
        selected_policy = selection.get("selected_policy")
        if not isinstance(selected_policy, str) or not selected_policy:
            raise ValueError(
                "temperature_selection.json does not contain selected_policy."
            )
        requested = selected_policy

    policy_dir = temperature_root / requested
    return (
        requested,
        policy_dir,
        {
            "predictions": policy_dir / "validation_predictions.pt",
            "graphs": policy_dir / "validation_graphs.pt",
            "metrics": policy_dir / "validation_metric_table.csv",
            "sampled_paths": policy_dir / "validation_sampled_price_paths.pt",
        },
    )


def _unwrap_prediction_result(payload: Any) -> tuple[dict[str, Any], int | None]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved prediction artefact must be a mapping.")

    epoch_value = payload.get("epoch")
    epoch = None if epoch_value is None else int(epoch_value)
    nested = payload.get("prediction_result")
    values = nested if isinstance(nested, Mapping) else payload

    required = {
        "y_pred",
        "y_true",
        "last_context_target",
        "horizons",
        "channels",
    }
    missing = required - set(values)
    if missing:
        raise KeyError(
            "Saved prediction artefact is missing keys: "
            f"{sorted(missing)}."
        )
    return dict(values), epoch


def _unwrap_graph_artifacts(payload: Any) -> tuple[dict[str, Any], int | None]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved graph artefact must be a mapping.")

    epoch_value = payload.get("epoch")
    epoch = None if epoch_value is None else int(epoch_value)
    nested = payload.get("graph_artifacts")
    values = nested if isinstance(nested, Mapping) else payload
    return dict(values), epoch


def _unwrap_sampled_price_paths(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved sampled-path artefact must be a mapping.")
    nested = payload.get("sampled_price_path_artifacts")
    values = nested if isinstance(nested, Mapping) else payload
    return dict(values)


def _validate_prediction_result(
    prediction_result: Mapping[str, Any],
    info: UnifiedRunInfo,
) -> int:
    y_pred = torch.as_tensor(prediction_result["y_pred"])
    y_true = torch.as_tensor(prediction_result["y_true"])
    if y_pred.shape != y_true.shape or y_pred.ndim != 4:
        raise ValueError(
            "Saved predictions and targets must share shape [W,H,N,C]."
        )
    windows, horizons, nodes, _ = map(int, y_pred.shape)
    if nodes != info.num_nodes:
        raise ValueError("Saved prediction node count differs from metadata.")
    if horizons != len(info.horizons):
        raise ValueError("Saved prediction horizon count differs from config.")
    if tuple(int(value) for value in prediction_result["horizons"]) != info.horizons:
        raise ValueError("Saved prediction horizons differ from config.")
    if not torch.isfinite(y_pred).all() or not torch.isfinite(y_true).all():
        raise ValueError("Saved prediction artefact contains non-finite values.")
    return windows


def _saved_graph_row_sum_tolerance(dtype: torch.dtype) -> float:
    """Return a storage-dtype-aware row-sum tolerance.

    Large Dimitri V2 graph artefacts are intentionally stored in ``float16``
    to keep the train/validation/test exports tractable.  A row that was
    exactly stochastic in FP32 can differ from one by a few times ``1e-4``
    after that storage cast.  Only that bounded quantisation drift is accepted;
    downstream loaders renormalise accepted rows in FP32 before calculating
    entropy, sector mass or connection summaries.
    """

    if dtype == torch.float16:
        return 1.0e-3
    if dtype == torch.bfloat16:
        return 1.0e-2
    return 2.0e-4


def _validate_graph_artifacts(
    graph_artifacts: Mapping[str, Any],
    info: UnifiedRunInfo,
    expected_windows: int,
) -> None:
    orientation = graph_artifacts.get(
        "graph_orientation",
        graph_artifacts.get("orientation", "A[target, source]"),
    )
    _canonical_graph_orientation(orientation)

    artifact_assets = graph_artifacts.get("asset_cols")
    if artifact_assets is not None and tuple(map(str, artifact_assets)) != info.asset_cols:
        raise ValueError("Saved graph asset order differs from run metadata.")

    for key in ("selected", "base", "dynamic"):
        values = graph_artifacts.get(key)
        if values is None:
            continue
        tensor = torch.as_tensor(values)
        if tensor.ndim != 4:
            raise ValueError(
                f"Graph component {key!r} must have shape [W,G,N,N]."
            )
        if int(tensor.shape[0]) != expected_windows:
            raise ValueError(
                f"Graph component {key!r} has {int(tensor.shape[0])} "
                f"windows; expected {expected_windows}."
            )
        if tuple(tensor.shape[-2:]) != (info.num_nodes, info.num_nodes):
            raise ValueError(f"Graph component {key!r} has wrong node shape.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Graph component {key!r} contains non-finite values.")
        row_sums = tensor.float().sum(dim=-1)
        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=_saved_graph_row_sum_tolerance(tensor.dtype),
            rtol=0.0,
        ):
            maximum_error = float((row_sums - 1.0).abs().max().item())
            raise ValueError(
                f"Graph component {key!r} is not row-stochastic; "
                f"maximum row-sum error={maximum_error:.3e}."
            )

    dates = graph_artifacts.get("dates")
    if dates is not None and len(dates) != expected_windows:
        raise ValueError("Saved graph dates do not align with window count.")


def load_analysis_artifacts(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "best",
) -> AnalysisArtifacts:
    """Load one saved inference policy for either final-model family."""

    info = load_unified_run_info(run_dir)
    resolved_policy, policy_dir, paths = _resolve_analysis_policy(
        info.run_dir,
        policy,
    )

    for key in ("predictions", "graphs", "metrics"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    prediction_result, prediction_epoch = _unwrap_prediction_result(
        _torch_load(paths["predictions"])
    )
    graph_artifacts, graph_epoch = _unwrap_graph_artifacts(
        _torch_load(paths["graphs"])
    )
    metric_table = pd.read_csv(paths["metrics"])
    sampled_paths = (
        _unwrap_sampled_price_paths(_torch_load(paths["sampled_paths"]))
        if paths["sampled_paths"].is_file()
        else None
    )

    windows = _validate_prediction_result(prediction_result, info)
    # Use the same normalising path as the current Graph Hub API.  This
    # converts storage-compressed FP16 graphs to FP32 and corrects only the
    # tiny row-sum drift introduced by serialisation.
    graph_artifacts = _normalise_graph_payload(
        graph_artifacts,
        info=info,
        prediction_result=prediction_result,
        windows=windows,
    )

    if prediction_epoch is not None and graph_epoch is not None:
        if prediction_epoch != graph_epoch:
            raise ValueError(
                "Prediction and graph artefacts come from different epochs."
            )
    epoch = prediction_epoch if prediction_epoch is not None else graph_epoch

    return AnalysisArtifacts(
        info=info,
        policy=resolved_policy,
        epoch=epoch,
        prediction_result=prediction_result,
        graph_artifacts=graph_artifacts,
        metric_table=metric_table,
        sampled_price_paths=sampled_paths,
        policy_dir=policy_dir,
    )


def _first_scalar(values: Any) -> float | None:
    if values is None:
        return None
    tensor = torch.as_tensor(values).detach().float().reshape(-1)
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return None
    return float(finite.mean().item())


def make_run_overview_table(
    runs: Mapping[str, str | Path],
    *,
    policies: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarise model family, checkpoint policy and graph diagnostics."""

    rows: list[dict[str, Any]] = []
    for label, run_dir in runs.items():
        policy = "best" if policies is None else policies.get(label, "best")
        artifacts = load_analysis_artifacts(run_dir, policy=policy)
        graph = artifacts.graph_artifacts
        selected = graph.get("selected")
        entropy = None
        effective = None
        if selected is not None:
            tensor = torch.as_tensor(selected).float().clamp_min(1.0e-12)
            row_entropy = -(tensor * tensor.log()).sum(dim=-1)
            entropy = float(row_entropy.mean().item())
            effective = float(row_entropy.exp().mean().item())

        temperature = None
        sample_count = None
        if artifacts.sampled_price_paths is not None:
            temperature = artifacts.sampled_price_paths.get("temperature")
            sample_count = artifacts.sampled_price_paths.get("sample_count")

        rows.append(
            {
                "Model": str(label),
                "Run": artifacts.info.run_dir.name,
                "Family": artifacts.info.run_kind,
                "Policy": artifacts.policy,
                "Epoch": artifacts.epoch,
                "Selection metric": artifacts.info.selection_metric,
                "Graph type": artifacts.info.graph_type,
                "Graph heads": artifacts.info.num_heads,
                "Spatial beta": _first_scalar(graph.get("spatial_beta")),
                "Dynamic alpha": _first_scalar(
                    graph.get("alpha", graph.get("dynamic_alpha"))
                ),
                "Mean row entropy": entropy,
                "Mean effective neighbours": effective,
                "Temperature": temperature,
                "Sample count": sample_count,
            }
        )
    return pd.DataFrame(rows)


def make_comparative_metrics_table(
    runs: Mapping[str, str | Path],
    *,
    policies: Mapping[str, str] | None = None,
    channel: str = "close",
    metrics_to_display: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one full horizon-by-model metric table for mixed run families."""

    metric_order = list(
        DEFAULT_SUMMARY_METRICS
        if metrics_to_display is None
        else [str(value) for value in metrics_to_display]
    )
    frames: list[pd.DataFrame] = []

    for label, run_dir in runs.items():
        policy = "best" if policies is None else policies.get(label, "best")
        artifacts = load_analysis_artifacts(run_dir, policy=policy)
        long_table = artifacts.metric_table
        required = {"metric", "horizon", "channel", "value"}
        missing = required - set(long_table.columns)
        if missing:
            raise ValueError(
                f"Metric table for {label!r} is missing {sorted(missing)}."
            )
        selected = long_table.loc[
            long_table["channel"].astype(str).str.lower().eq(channel.lower()),
            ["metric", "horizon", "value"],
        ].copy()
        selected["horizon"] = pd.to_numeric(
            selected["horizon"], errors="raise"
        ).astype(int)
        wide = (
            selected.pivot(index="horizon", columns="metric", values="value")
            .reindex(index=artifacts.info.horizons, columns=metric_order)
            .rename(
                columns={
                    name: DEFAULT_METRIC_DISPLAY_NAMES.get(name, name)
                    for name in metric_order
                }
            )
        )
        wide.insert(0, "Model", str(label))
        wide.insert(1, "Policy", artifacts.policy)
        wide.insert(2, "Horizon", [f"{int(value)} min" for value in wide.index])
        frames.append(wide.reset_index(drop=True))

    if not frames:
        raise ValueError("At least one run must be supplied.")
    result = pd.concat(frames, ignore_index=True)
    return result.set_index(["Model", "Policy", "Horizon"])


def _normalise_date(value: Any) -> str:
    return pd.Timestamp(str(value)).date().isoformat()


def make_graph_window_table(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "best",
) -> pd.DataFrame:
    """List saved graph windows with a one-based window number per date."""

    artifacts = load_analysis_artifacts(run_dir, policy=policy)
    graph = artifacts.graph_artifacts
    selected = graph.get("selected")
    if selected is None:
        raise ValueError("This policy does not contain a selected graph.")
    windows = int(torch.as_tensor(selected).shape[0])

    raw_dates = graph.get("dates") or [None] * windows
    if len(raw_dates) != windows:
        raise ValueError("Graph date metadata length does not match graph windows.")

    def optional_vector(name: str) -> list[int | None]:
        values = graph.get(name)
        if values is None:
            return [None] * windows
        tensor = torch.as_tensor(values).reshape(-1)
        if int(tensor.numel()) != windows:
            raise ValueError(f"Graph metadata {name!r} has wrong length.")
        return [int(value) for value in tensor.tolist()]

    table = pd.DataFrame(
        {
            "Global window index": np.arange(windows, dtype=int),
            "Date": [
                None if value is None else _normalise_date(value)
                for value in raw_dates
            ],
            "Sample index": optional_vector("sample_idx"),
            "Origin index": optional_vector("origin_idx"),
        }
    )
    if table["Date"].notna().any():
        table["Window within date"] = (
            table.groupby("Date", dropna=False).cumcount() + 1
        )
    else:
        table["Window within date"] = np.arange(1, windows + 1)
    return table[
        [
            "Global window index",
            "Date",
            "Window within date",
            "Sample index",
            "Origin index",
        ]
    ]


def _resolve_saved_window(
    window_table: pd.DataFrame,
    *,
    date: str | None,
    window_within_date: int | None,
    global_window_index: int | None,
    origin_idx: int | None,
) -> pd.Series:
    provided = sum(
        value is not None
        for value in (global_window_index, origin_idx)
    )
    if provided > 1:
        raise ValueError("Specify at most one of global_window_index/origin_idx.")

    if global_window_index is not None:
        rows = window_table.loc[
            window_table["Global window index"] == int(global_window_index)
        ]
    elif origin_idx is not None:
        rows = window_table.loc[window_table["Origin index"] == int(origin_idx)]
        if date is not None:
            rows = rows.loc[rows["Date"] == _normalise_date(date)]
    else:
        if date is None:
            raise ValueError(
                "Specify date plus window_within_date, or a global index."
            )
        resolved_window = 1 if window_within_date is None else int(window_within_date)
        if resolved_window <= 0:
            raise ValueError("window_within_date is one-based and must be positive.")
        rows = window_table.loc[
            (window_table["Date"] == _normalise_date(date))
            & (window_table["Window within date"] == resolved_window)
        ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one saved window, found "
            f"{len(rows)}. Inspect make_graph_window_table(...)."
        )
    return rows.iloc[0]


def select_graph_snapshot(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "best",
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    date: str | None = None,
    window_within_date: int | None = None,
    global_window_index: int | None = None,
    origin_idx: int | None = None,
) -> GraphSnapshot:
    """Select the exact saved adjacency for a date and one-based window."""

    artifacts = load_analysis_artifacts(run_dir, policy=policy)
    graph = artifacts.graph_artifacts
    window_table = make_graph_window_table(run_dir, policy=policy)
    row = _resolve_saved_window(
        window_table,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=origin_idx,
    )
    absolute_index = int(row["Global window index"])

    if layer != -1:
        per_layer_key = {
            "selected": "per_layer",
            "base": "per_layer_base",
            "dynamic": "per_layer_dynamic",
        }[component]
        per_layer = graph.get(per_layer_key)
        if not isinstance(per_layer, Sequence):
            raise ValueError(
                f"Saved graph artefact does not contain {per_layer_key}."
            )
        layer_index = _normalise_layer_index(layer, len(per_layer))
        values = per_layer[layer_index]
    else:
        values = graph.get(component)
    if values is None:
        raise ValueError(f"Graph component {component!r} is unavailable.")

    tensor = torch.as_tensor(values).detach().cpu().double()
    if tensor.ndim != 4:
        raise ValueError("Saved graph must have shape [W,G,N,N].")
    graph_heads = tensor[absolute_index]
    adjacency = _select_graph_head(graph_heads, head)
    row_entropy = -(
        adjacency.clamp_min(1.0e-12)
        * adjacency.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    top_k = min(5, adjacency.shape[-1] - (0 if artifacts.info.add_self_loops else 1))
    top5_mass = adjacency.topk(top_k, dim=-1).values.sum(dim=-1)

    frame = pd.DataFrame(
        adjacency.numpy(),
        index=artifacts.info.asset_cols,
        columns=artifacts.info.asset_cols,
    )
    frame.index.name = "Target"
    frame.columns.name = "Source"

    date_value = None if pd.isna(row["Date"]) else str(row["Date"])
    window_value = (
        None
        if pd.isna(row["Window within date"])
        else int(row["Window within date"])
    )
    sample_value = (
        None if pd.isna(row["Sample index"]) else int(row["Sample index"])
    )
    origin_value = (
        None if pd.isna(row["Origin index"]) else int(row["Origin index"])
    )

    return GraphSnapshot(
        adjacency=frame,
        run_name=artifacts.info.run_dir.name,
        run_kind=artifacts.info.run_kind,
        policy=artifacts.policy,
        component=component,
        head=head,
        global_window_index=absolute_index,
        date=date_value,
        window_within_date=window_value,
        sample_idx=sample_value,
        origin_idx=origin_value,
        mean_row_entropy=float(row_entropy.mean().item()),
        mean_effective_neighbours=float(row_entropy.exp().mean().item()),
        median_effective_neighbours=float(row_entropy.exp().median().item()),
        maximum_edge_weight=float(adjacency.max().item()),
        mean_top5_row_mass=float(top5_mass.mean().item()),
        graph_type=artifacts.info.graph_type,
        add_self_loops=artifacts.info.add_self_loops,
    )


def make_graph_snapshot_summary_table(snapshot: GraphSnapshot) -> pd.DataFrame:
    """Return one row of interpretable graph concentration diagnostics."""

    return pd.DataFrame(
        [
            {
                "Run": snapshot.run_name,
                "Family": snapshot.run_kind,
                "Policy": snapshot.policy,
                "Graph type": snapshot.graph_type,
                "Component": snapshot.component,
                "Head": snapshot.head,
                "Date": snapshot.date,
                "Window within date": snapshot.window_within_date,
                "Global window index": snapshot.global_window_index,
                "Origin index": snapshot.origin_idx,
                "Mean row entropy": snapshot.mean_row_entropy,
                "Mean effective neighbours": snapshot.mean_effective_neighbours,
                "Median effective neighbours": snapshot.median_effective_neighbours,
                "Maximum edge weight": snapshot.maximum_edge_weight,
                "Mean top-5 row mass": snapshot.mean_top5_row_mass,
            }
        ]
    )


def make_graph_snapshot_connections_table(
    snapshot: GraphSnapshot,
    *,
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
) -> pd.DataFrame:
    """Rank the strongest connections in one exact adjacency snapshot."""

    adjacency = snapshot.adjacency.to_numpy(dtype=np.float64)
    labels = tuple(str(value) for value in snapshot.adjacency.index)
    return _top_neighbours_from_adjacency(
        adjacency,
        labels,
        top_n=top_n,
        direction=direction,
    )


def plot_graph_snapshot(
    snapshot: GraphSnapshot,
    *,
    cluster: bool = False,
    cluster_method: str = "average",
    figsize: tuple[float, float] = (13.0, 11.0),
    tick_fontsize: float = 8.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot the actual adjacency weights for one date/window snapshot."""

    matrix = snapshot.adjacency.to_numpy(dtype=np.float64, copy=True)
    labels = np.asarray(snapshot.adjacency.index, dtype=object)
    if cluster:
        order = _cluster_graph_order(matrix, method=cluster_method)
        matrix = matrix[np.ix_(order, order)]
        labels = labels[order]

    plotted_values = matrix.copy()
    if not snapshot.add_self_loops:
        np.fill_diagonal(plotted_values, np.nan)
    finite = plotted_values[np.isfinite(plotted_values)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    image = ax.imshow(
        plotted_values,
        cmap=cmap,
        vmin=0.0,
        vmax=maximum,
        interpolation="nearest",
        aspect="equal",
    )
    count = len(labels)
    ax.set_xticks(np.arange(count))
    ax.set_yticks(np.arange(count))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    ax.set_xlabel("Source asset (influences target)")
    ax.set_ylabel("Target asset (receives influence)")
    date_label = snapshot.date or "date unavailable"
    window_label = (
        "?" if snapshot.window_within_date is None else snapshot.window_within_date
    )
    ax.set_title(
        f"{snapshot.run_name} — {snapshot.policy}\n"
        f"{date_label}, window {window_label}; "
        f"mean entropy={snapshot.mean_row_entropy:.4f}, "
        f"effective neighbours={snapshot.mean_effective_neighbours:.2f}"
    )
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colourbar.set_label("Adjacency weight")
    fig.tight_layout()
    plotted = pd.DataFrame(matrix, index=labels, columns=labels)
    plotted.index.name = "Target"
    plotted.columns.name = "Source"
    return fig, ax, plotted


def load_sampled_path_bundle(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
) -> SampledPathBundle:
    """Load and validate all decoded Monte Carlo Close-price paths."""

    artifacts = load_analysis_artifacts(run_dir, policy=policy)
    values = artifacts.sampled_price_paths
    if values is None:
        raise ValueError(
            f"Policy {artifacts.policy!r} contains no saved sampled paths."
        )

    required = {
        "sampled_close_paths",
        "sampled_close_paths_at_evaluation_horizons",
        "ensemble_mean_close_path",
        "evaluation_true",
        "last_context_target",
        "asset_cols",
        "future_steps",
        "evaluation_horizons",
        "sample_count",
    }
    missing = required - set(values)
    if missing:
        raise KeyError(f"Sampled-path artefact is missing {sorted(missing)}.")

    sampled = torch.as_tensor(values["sampled_close_paths"]).float()
    sampled_eval = torch.as_tensor(
        values["sampled_close_paths_at_evaluation_horizons"]
    ).float()
    ensemble = torch.as_tensor(values["ensemble_mean_close_path"]).float()
    true = torch.as_tensor(values["evaluation_true"]).float()
    last = torch.as_tensor(values["last_context_target"]).float()

    if sampled.ndim != 5 or sampled.shape[-1] != 1:
        raise ValueError("sampled_close_paths must have shape [S,W,P,N,1].")
    sample_count, windows, steps, nodes, _ = map(int, sampled.shape)
    if sample_count != int(values["sample_count"]):
        raise ValueError("Saved sample count differs from path tensor.")
    if nodes != artifacts.info.num_nodes:
        raise ValueError("Sampled path node count differs from run metadata.")
    expected_eval = (sample_count, windows, len(artifacts.info.horizons), nodes, 1)
    if tuple(sampled_eval.shape) != expected_eval:
        raise ValueError(
            "sampled_close_paths_at_evaluation_horizons has shape "
            f"{tuple(sampled_eval.shape)}, expected {expected_eval}."
        )
    if tuple(ensemble.shape) != (windows, steps, nodes, 1):
        raise ValueError("ensemble_mean_close_path has inconsistent shape.")
    if tuple(true.shape) != (windows, len(artifacts.info.horizons), nodes, 1):
        raise ValueError("evaluation_true has inconsistent shape.")
    if tuple(last.shape) != (windows, nodes, 1):
        raise ValueError("last_context_target has inconsistent shape.")
    for name, tensor in {
        "sampled paths": sampled,
        "sampled evaluation paths": sampled_eval,
        "ensemble mean": ensemble,
        "truth": true,
        "last context": last,
    }.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values.")
        if (tensor <= 0).any():
            raise ValueError(f"{name} contains non-positive Close values.")

    torch.testing.assert_close(
        ensemble,
        sampled.mean(dim=0),
        atol=2.0e-5,
        rtol=1.0e-6,
    )

    def optional_long(name: str) -> Tensor | None:
        value = values.get(name)
        return None if value is None else torch.as_tensor(value).long()

    dates = tuple(str(value) for value in (values.get("dates") or []))
    if dates and len(dates) != windows:
        raise ValueError("Sampled-path dates do not align with windows.")

    return SampledPathBundle(
        run_dir=artifacts.info.run_dir,
        policy=artifacts.policy,
        sampled_close_paths=sampled,
        sampled_close_paths_at_evaluation_horizons=sampled_eval,
        ensemble_mean_close_path=ensemble,
        evaluation_true=true,
        last_context_target=last,
        sample_idx=optional_long("sample_idx"),
        origin_idx=optional_long("origin_idx"),
        dense_target_indices=optional_long("dense_target_indices"),
        evaluation_target_indices=optional_long("evaluation_target_indices"),
        dates=dates,
        asset_cols=tuple(str(value) for value in values["asset_cols"]),
        future_steps=tuple(int(value) for value in values["future_steps"]),
        evaluation_horizons=tuple(
            int(value) for value in values["evaluation_horizons"]
        ),
        temperature=(
            None if values.get("temperature") is None else float(values["temperature"])
        ),
        top_k=int(values.get("top_k", 0)),
        top_p=float(values.get("top_p", 1.0)),
        sample_count=sample_count,
    )


def _sampled_return_tensors(bundle: SampledPathBundle) -> tuple[Tensor, Tensor]:
    last = bundle.last_context_target.unsqueeze(0).unsqueeze(2)
    sampled_returns = torch.log(
        bundle.sampled_close_paths_at_evaluation_horizons / last
    )
    true_returns = torch.log(
        bundle.evaluation_true / bundle.last_context_target.unsqueeze(1)
    )
    return sampled_returns, true_returns


def make_predictive_coverage_table(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
    nominal_coverages: Sequence[float] = (0.5, 0.8, 0.9),
    include_sample_range: bool = True,
) -> pd.DataFrame:
    """Compute empirical central-interval coverage in log-return space."""

    bundle = load_sampled_path_bundle(run_dir, policy=policy)
    sampled_returns, true_returns = _sampled_return_tensors(bundle)
    rows: list[dict[str, Any]] = []

    interval_specs: list[tuple[str, float | None, float, float]] = []
    for coverage in nominal_coverages:
        value = float(coverage)
        if not 0.0 < value < 1.0:
            raise ValueError("nominal_coverages must lie strictly in (0,1).")
        tail = 0.5 * (1.0 - value)
        interval_specs.append((f"Central {100*value:g}%", value, tail, 1.0 - tail))
    if include_sample_range:
        interval_specs.append(("Sample min-max", None, 0.0, 1.0))

    for label, nominal, lower_q, upper_q in interval_specs:
        lower = torch.quantile(sampled_returns, lower_q, dim=0)
        upper = torch.quantile(sampled_returns, upper_q, dim=0)
        covered = (true_returns >= lower) & (true_returns <= upper)
        below = true_returns < lower
        above = true_returns > upper
        width_bps = 10000.0 * (upper - lower)

        for horizon_index, horizon in enumerate(bundle.evaluation_horizons):
            coverage_value = float(covered[:, horizon_index].float().mean().item())
            rows.append(
                {
                    "Policy": bundle.policy,
                    "Temperature": bundle.temperature,
                    "Sample count": bundle.sample_count,
                    "Horizon": int(horizon),
                    "Interval": label,
                    "Nominal coverage": nominal,
                    "Empirical coverage": coverage_value,
                    "Coverage gap": (
                        None if nominal is None else coverage_value - nominal
                    ),
                    "Below interval": float(
                        below[:, horizon_index].float().mean().item()
                    ),
                    "Above interval": float(
                        above[:, horizon_index].float().mean().item()
                    ),
                    "Mean interval width (log-return bps)": float(
                        width_bps[:, horizon_index].mean().item()
                    ),
                    "Median interval width (log-return bps)": float(
                        width_bps[:, horizon_index].median().item()
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_probabilistic_score_table(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
) -> pd.DataFrame:
    """Report empirical CRPS, ensemble error and sample dispersion by horizon."""

    bundle = load_sampled_path_bundle(run_dir, policy=policy)
    sampled, truth = _sampled_return_tensors(bundle)
    ensemble_mean = sampled.mean(dim=0)
    sample_median = sampled.median(dim=0).values
    term_one = (sampled - truth.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (sampled[:, None] - sampled[None, :]).abs().mean(dim=(0, 1))
    crps = term_one - 0.5 * pairwise
    dispersion = sampled.std(dim=0, unbiased=False)

    rows: list[dict[str, Any]] = []
    for index, horizon in enumerate(bundle.evaluation_horizons):
        rows.append(
            {
                "Policy": bundle.policy,
                "Temperature": bundle.temperature,
                "Sample count": bundle.sample_count,
                "Horizon": int(horizon),
                "Empirical CRPS (log-return bps)": float(
                    10000.0 * crps[:, index].mean().item()
                ),
                "Ensemble-mean Log MAE": float(
                    (ensemble_mean[:, index] - truth[:, index]).abs().mean().item()
                ),
                "Sample-median Log MAE": float(
                    (sample_median[:, index] - truth[:, index]).abs().mean().item()
                ),
                "Mean predictive std (log-return bps)": float(
                    10000.0 * dispersion[:, index].mean().item()
                ),
                "Median predictive std (log-return bps)": float(
                    10000.0 * dispersion[:, index].median().item()
                ),
            }
        )
    return pd.DataFrame(rows)


def make_temperature_sweep_table(run_dir: str | Path) -> pd.DataFrame:
    """Load the saved inference-temperature ranking and mark the winner."""

    run_path = _resolve_run_dir(run_dir)
    root = run_path / "temperature_sweep"
    table_path = root / "temperature_sweep_results.csv"
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    table = pd.read_csv(table_path)
    selection = _load_json(root / "temperature_selection.json")
    selected_policy = str(selection["selected_policy"])
    table.insert(0, "Selected", table["Policy"].astype(str).eq(selected_policy))
    return table.sort_values(["Mean Log MAE", "Policy"]).reset_index(drop=True)


def plot_coverage_calibration(
    coverage_table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot nominal versus empirical central-interval coverage by horizon."""

    required = {"Horizon", "Nominal coverage", "Empirical coverage"}
    missing = required - set(coverage_table.columns)
    if missing:
        raise ValueError(f"coverage_table is missing {sorted(missing)}.")
    selected = coverage_table.loc[coverage_table["Nominal coverage"].notna()]
    if selected.empty:
        raise ValueError("coverage_table contains no nominal intervals.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for horizon, group in selected.groupby("Horizon"):
        ordered = group.sort_values("Nominal coverage")
        ax.plot(
            ordered["Nominal coverage"],
            ordered["Empirical coverage"],
            marker="o",
            label=f"{int(horizon)} min",
        )
    limits = [0.0, 1.0]
    ax.plot(limits, limits, linestyle="--", linewidth=1.0, label="Ideal")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Nominal central coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive interval calibration")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def _resolve_bundle_window(
    bundle: SampledPathBundle,
    *,
    date: str | None,
    window_within_date: int | None,
    global_window_index: int | None,
) -> int:
    windows = int(bundle.sampled_close_paths.shape[1])
    dates = list(bundle.dates) if bundle.dates else [None] * windows
    table = pd.DataFrame(
        {
            "Global window index": np.arange(windows, dtype=int),
            "Date": [None if value is None else _normalise_date(value) for value in dates],
        }
    )
    table["Window within date"] = table.groupby("Date", dropna=False).cumcount() + 1
    row = _resolve_saved_window(
        table.assign(**{"Sample index": np.nan, "Origin index": np.nan}),
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=None,
    )
    return int(row["Global window index"])


def _load_true_candle_window(
    *,
    data_dir: str | Path,
    bundle: SampledPathBundle,
    window_index: int,
    asset_index: int,
    split: Literal["validation", "test"] = "validation",
) -> tuple[Tensor | None, Tensor | None, str | None]:
    """Load context and dense true future Close when raw data are available."""

    if bundle.sample_idx is None or bundle.dense_target_indices is None:
        return None, None, None
    train_raw, val_raw, test_raw = load_candle_splits(data_dir)
    raw = val_raw if split == "validation" else test_raw
    cleaned = clean_candle_split(raw)
    sample_index = int(bundle.sample_idx.reshape(-1)[window_index].item())
    if not 0 <= sample_index < len(cleaned["samples"]):
        raise IndexError("Saved sample_idx is outside the requested split.")
    x, _, day = cleaned["samples"][sample_index]
    close_index = get_channel_index(cleaned, "close")
    origin = (
        None
        if bundle.origin_idx is None
        else int(bundle.origin_idx.reshape(-1)[window_index].item())
    )
    context = (
        None
        if origin is None
        else torch.as_tensor(x[: origin + 1, asset_index, close_index]).float()
    )
    dense_indices = bundle.dense_target_indices[window_index].long()
    future = torch.as_tensor(x[dense_indices, asset_index, close_index]).float()
    return context, future, _normalise_date(day)


def plot_sampled_price_paths(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
    asset: str,
    date: str | None = None,
    window_within_date: int | None = None,
    global_window_index: int | None = None,
    data_dir: str | Path | None = None,
    split: Literal["validation", "test"] = "validation",
    context_points: int = 30,
    figsize: tuple[float, float] = (12.0, 6.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot all decoded paths, their mean and the realised Close path."""

    bundle = load_sampled_path_bundle(run_dir, policy=policy)
    if asset not in bundle.asset_cols:
        raise ValueError(
            f"Asset {asset!r} is unavailable. Example assets: "
            f"{list(bundle.asset_cols[:10])}."
        )
    asset_index = bundle.asset_cols.index(asset)
    window_index = _resolve_bundle_window(
        bundle,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
    )

    paths = bundle.sampled_close_paths[:, window_index, :, asset_index, 0]
    mean_path = bundle.ensemble_mean_close_path[window_index, :, asset_index, 0]
    eval_true = bundle.evaluation_true[window_index, :, asset_index, 0]
    last_value = bundle.last_context_target[window_index, asset_index, 0]

    context = None
    true_future = None
    if data_dir is not None:
        context, true_future, _ = _load_true_candle_window(
            data_dir=data_dir,
            bundle=bundle,
            window_index=window_index,
            asset_index=asset_index,
            split=split,
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    future_x = np.arange(1, paths.shape[1] + 1)
    for sample_index in range(paths.shape[0]):
        ax.plot(
            future_x,
            paths[sample_index].numpy(),
            linewidth=0.9,
            alpha=0.35,
        )
    ax.plot(future_x, mean_path.numpy(), linewidth=2.4, label="Ensemble mean")

    if true_future is not None:
        ax.plot(future_x, true_future.numpy(), linewidth=2.2, label="True future")
    else:
        ax.scatter(
            np.asarray(bundle.evaluation_horizons),
            eval_true.numpy(),
            marker="x",
            s=55,
            label="True evaluation horizons",
        )

    if context is not None and context_points > 0:
        shown = context[-int(context_points):]
        context_x = np.arange(-len(shown) + 1, 1)
        ax.plot(context_x, shown.numpy(), linewidth=1.8, label="Observed context")
    else:
        ax.scatter([0], [float(last_value)], s=35, label="Last observed Close")

    ax.axvline(0, linestyle="--", linewidth=1.0)
    date_label = (
        _normalise_date(bundle.dates[window_index])
        if bundle.dates
        else "date unavailable"
    )
    ax.set_title(
        f"{asset} — {date_label} — {bundle.policy} — "
        f"{bundle.sample_count} decoded paths"
    )
    ax.set_xlabel("Minutes relative to forecast origin")
    ax.set_ylabel("Raw Close price")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path_table = pd.DataFrame(
        paths.T.numpy(),
        index=pd.Index(future_x, name="Future minute"),
        columns=[f"Path {index + 1}" for index in range(paths.shape[0])],
    )
    path_table["Ensemble mean"] = mean_path.numpy()
    if true_future is not None:
        path_table["True future"] = true_future.numpy()
    else:
        path_table["True future"] = np.nan
        for horizon, value in zip(bundle.evaluation_horizons, eval_true, strict=True):
            path_table.loc[int(horizon), "True future"] = float(value.item())
    return fig, ax, path_table


def plot_point_forecast_window(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "best",
    asset: str,
    date: str | None = None,
    window_within_date: int | None = None,
    global_window_index: int | None = None,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot point forecasts and truths at the five evaluation horizons."""

    artifacts = load_analysis_artifacts(run_dir, policy=policy)
    table = make_graph_window_table(run_dir, policy=policy)
    row = _resolve_saved_window(
        table,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=None,
    )
    window_index = int(row["Global window index"])
    if asset not in artifacts.info.asset_cols:
        raise ValueError(f"Asset {asset!r} is unavailable.")
    asset_index = artifacts.info.asset_cols.index(asset)
    y_pred = torch.as_tensor(artifacts.prediction_result["y_pred"])
    y_true = torch.as_tensor(artifacts.prediction_result["y_true"])
    last = torch.as_tensor(artifacts.prediction_result["last_context_target"])
    pred = y_pred[window_index, :, asset_index, 0].float()
    truth = y_true[window_index, :, asset_index, 0].float()
    last_value = float(last[window_index, asset_index, 0].item())
    horizons = np.asarray(artifacts.info.horizons, dtype=int)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(horizons, pred.numpy(), marker="o", label="Prediction")
    ax.plot(horizons, truth.numpy(), marker="o", label="Truth")
    ax.scatter([0], [last_value], label="Last observed Close")
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Raw Close price")
    ax.set_title(
        f"{asset} — {row['Date']} window {int(row['Window within date'])} — "
        f"{artifacts.policy}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    values = pd.DataFrame(
        {
            "Horizon": horizons,
            "Prediction": pred.numpy(),
            "Truth": truth.numpy(),
            "Absolute error": (pred - truth).abs().numpy(),
        }
    ).set_index("Horizon")
    return fig, ax, values


def make_predictive_coverage_by_asset_table(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
    nominal_coverage: float = 0.8,
) -> pd.DataFrame:
    """Report central-interval coverage and width for every asset/horizon."""

    coverage = float(nominal_coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("nominal_coverage must lie strictly in (0,1).")
    bundle = load_sampled_path_bundle(run_dir, policy=policy)
    sampled, truth = _sampled_return_tensors(bundle)
    tail = 0.5 * (1.0 - coverage)
    lower = torch.quantile(sampled, tail, dim=0)
    upper = torch.quantile(sampled, 1.0 - tail, dim=0)
    covered = (truth >= lower) & (truth <= upper)
    width_bps = 10000.0 * (upper - lower)

    rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(bundle.evaluation_horizons):
        for asset_index, asset in enumerate(bundle.asset_cols):
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "Nominal coverage": coverage,
                    "Empirical coverage": float(
                        covered[:, horizon_index, asset_index, 0]
                        .float()
                        .mean()
                        .item()
                    ),
                    "Mean interval width (log-return bps)": float(
                        width_bps[:, horizon_index, asset_index, 0]
                        .mean()
                        .item()
                    ),
                    "Median interval width (log-return bps)": float(
                        width_bps[:, horizon_index, asset_index, 0]
                        .median()
                        .item()
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_sample_rank_histogram_table(
    run_dir: str | Path,
    *,
    policy: AnalysisPolicy = "selected_temperature",
) -> pd.DataFrame:
    """Return truth ranks among sampled paths for a finite-ensemble check.

    With ``S`` paths the rank takes values 0..S and counts how many sampled
    cumulative returns are strictly below the realised cumulative return.
    A calibrated continuous predictive distribution would be approximately
    uniform over these ``S+1`` bins, subject to the small sample size.
    """

    bundle = load_sampled_path_bundle(run_dir, policy=policy)
    sampled, truth = _sampled_return_tensors(bundle)
    ranks = (sampled < truth.unsqueeze(0)).sum(dim=0)
    rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(bundle.evaluation_horizons):
        horizon_ranks = ranks[:, horizon_index].reshape(-1)
        total = int(horizon_ranks.numel())
        for rank in range(bundle.sample_count + 1):
            count = int((horizon_ranks == rank).sum().item())
            rows.append(
                {
                    "Horizon": int(horizon),
                    "Rank": int(rank),
                    "Count": count,
                    "Frequency": count / max(total, 1),
                    "Ideal frequency": 1.0 / (bundle.sample_count + 1),
                }
            )
    return pd.DataFrame(rows)


def make_unified_model_summary_table(run_dir: str | Path) -> pd.DataFrame:
    """Summarise the saved final architecture without reconstructing weights."""

    info = load_unified_run_info(run_dir)
    resolved = info.resolved_config
    metadata = info.run_metadata
    rows: list[tuple[str, Any]] = [
        ("Run family", info.run_kind),
        ("Run name", info.run_dir.name),
        ("Assets", info.num_nodes),
        ("Evaluation horizons", list(info.horizons)),
        ("Selection metric", info.selection_metric),
    ]

    if info.run_kind == "continuous":
        model = resolved["model"]
        graph = model["graph"]
        training = resolved.get("training", {})
        basedygraph = resolved.get("basedygraph_financial")
        round2 = (
            metadata.get("model_family") == "modern_tcn_graph_round2"
            or isinstance(model.get("temporal_stack"), Mapping)
        )

        if isinstance(basedygraph, Mapping):
            regularisation = basedygraph.get("regularisation", {})
            if not isinstance(regularisation, Mapping):
                regularisation = {}
            activations = basedygraph.get("graph_activations")
            if activations is None:
                activation = basedygraph.get(
                    "graph_activation", graph.get("activation")
                )
                activations = [activation] * int(
                    basedygraph.get(
                        "num_st_blocks", model.get("num_st_blocks", 1)
                    )
                )
            rows.extend(
                [
                    ("Input representation", "context-normalised OHLCV"),
                    ("Temporal backbone", "official BaseDyGraph Transformer"),
                    ("Hidden dimension", basedygraph.get("d_model")),
                    ("Interlaced ST blocks", basedygraph.get("num_st_blocks")),
                    ("Temporal layers per block", basedygraph.get("temporal_layers")),
                    ("Temporal attention heads", basedygraph.get("temporal_heads")),
                    ("Temporal FF multiplier", basedygraph.get("ff_mult")),
                    ("Graph type", graph.get("type", basedygraph.get("graph_type"))),
                    ("Graph scope", basedygraph.get("graph_scope")),
                    ("Graph heads", basedygraph.get("graph_heads", graph.get("num_heads"))),
                    ("Graph hidden dimension", basedygraph.get("graph_hidden_dim", graph.get("hidden_dim"))),
                    ("Graph activations by layer", list(activations)),
                    ("Spatial layers per block", basedygraph.get("spatial_layers")),
                    ("Forecast strategy", resolved.get("forecast_strategy", model.get("forecast_strategy"))),
                    ("Model prediction length", basedygraph.get("prediction_length")),
                    ("Output representation", model.get("output_representation")),
                    ("Target graph entropy", regularisation.get("target_entropy")),
                    ("Target-entropy weight", regularisation.get("target_entropy_weight")),
                    ("Temporal-smoothness weight", regularisation.get("temporal_smooth_weight")),
                    ("Backbone learning rate", training.get("learning_rate")),
                ]
            )
        elif round2:
            temporal_stack = model.get("temporal_stack", {})
            if not isinstance(temporal_stack, Mapping):
                temporal_stack = {}
            modern = temporal_stack.get("modern_tcn", {})
            if not isinstance(modern, Mapping):
                modern = {}
            transformer = temporal_stack.get("transformer", {})
            if not isinstance(transformer, Mapping):
                transformer = {}
            spatial = model.get("spatial", {})
            if not isinstance(spatial, Mapping):
                spatial = {}
            prior = model.get("prior", {})
            if not isinstance(prior, Mapping):
                prior = {}
            loss = training.get("loss", {})
            if not isinstance(loss, Mapping):
                loss = {}
            rows.extend(
                [
                    ("Input representation", resolved["data"].get("input_representation")),
                    ("Temporal stack family", temporal_stack.get("family")),
                    ("Interlaced ST blocks", metadata.get("num_st_blocks", (
                        None
                        if info.num_heads_per_layer is None
                        else len(info.num_heads_per_layer)
                    ))),
                    ("ModernTCN hidden dimension", modern.get("d_model")),
                    ("ModernTCN blocks", modern.get("num_blocks")),
                    ("ModernTCN patch / stride", f"{modern.get('patch_size')} / {modern.get('patch_stride')}"),
                    ("ModernTCN large / small kernel", f"{modern.get('large_kernel')} / {modern.get('small_kernel')}"),
                    ("Transformer hidden dimension", transformer.get("d_model")),
                    ("Transformer layers per ST block", transformer.get("num_layers")),
                    ("Transformer temporal heads", transformer.get("num_heads")),
                    ("Transformer FF multiplier", transformer.get("feedforward_multiplier")),
                    ("Transformer dropout", transformer.get("dropout")),
                    ("Graph family", model.get("graph_family")),
                    ("Graph type", info.graph_type),
                    ("Graph heads by block", list(info.num_heads_per_layer or (info.num_heads,))),
                    ("Graph hidden dimensions by block", graph.get("hidden_dims_per_block")),
                    ("Graph activations by block", graph.get("activations_per_block")),
                    ("Prior type", prior.get("type")),
                    ("Prior scale", prior.get("scale")),
                    ("Prior jitter", prior.get("jitter")),
                    ("State pathway", metadata.get("state_pathway")),
                    ("Initial graph alpha", graph.get("initial_alpha")),
                    ("Spatial gate", spatial.get("gate_type")),
                    ("Initial spatial beta", spatial.get("initial_beta")),
                    ("Output representation", model.get("output_representation")),
                    ("Training loss", loss.get("type")),
                    ("Backbone learning rate", training.get("learning_rate")),
                    ("Graph learning rate", training.get("graph_learning_rate")),
                ]
            )
        else:
            temporal = model["temporal"]
            modern = temporal.get("modern_tcn", {})
            spatial = model["spatial"]
            loss = training.get("loss", {})
            rows.extend(
                [
                    ("Input representation", resolved["data"].get("input_representation")),
                    ("Temporal backbone", temporal.get("type")),
                    ("Hidden dimension", temporal.get("d_model")),
                    ("ModernTCN blocks", modern.get("num_blocks")),
                    ("Patch / stride", f"{modern.get('patch_size')} / {modern.get('patch_stride')}"),
                    ("Large / small kernel", f"{modern.get('large_kernel')} / {modern.get('small_kernel')}"),
                    ("ModernTCN FFN ratio", modern.get("ffn_ratio")),
                    ("Graph type", graph.get("type")),
                    ("Graph heads", graph.get("num_heads")),
                    ("Graph hidden dimension", graph.get("hidden_dim")),
                    ("Spatial layers", spatial.get("num_layers")),
                    ("Spatial gate", spatial.get("gate_type")),
                    ("Initial spatial beta", spatial.get("initial_beta")),
                    ("Output representation", model.get("output_representation")),
                    ("Training loss", loss.get("type")),
                    ("Backbone learning rate", training.get("learning_rate")),
                    ("Graph learning rate", training.get("graph_learning_rate")),
                ]
            )
            if str(graph.get("type")) == "dynamic_correlation":
                dynamic_correlation = model.get("dynamic_correlation", {})
                rows.extend(
                    [
                        (
                            "Dynamic-correlation threshold",
                            dynamic_correlation.get("threshold"),
                        ),
                        (
                            "Dynamic-correlation empty-row policy",
                            dynamic_correlation.get("empty_row_policy"),
                        ),
                        (
                            "Dynamic-correlation epsilon",
                            dynamic_correlation.get("eps"),
                        ),
                    ]
                )
    else:
        model = resolved["models"]["dynamic_graph"]
        temporal = model["temporal"]
        modern = temporal.get("modern_tcn", {})
        graph = model["graph"]
        spatial = model["spatial"]
        heads = model["heads"]
        future = model.get("future_predictor", {})
        training = resolved["training"]
        rows.extend(
            [
                ("Token input representation", model.get("token_input_representation")),
                ("Temporal backbone", temporal.get("type")),
                ("Hidden dimension", model.get("d_model")),
                ("ModernTCN blocks", modern.get("num_blocks")),
                ("Patch / stride", f"{modern.get('patch_size')} / {modern.get('patch_stride')}"),
                ("Large / small kernel", f"{modern.get('large_kernel')} / {modern.get('small_kernel')}"),
                ("ModernTCN FFN ratio", modern.get("ffn_ratio")),
                ("Graph type", graph.get("type")),
                ("Graph heads", graph.get("num_heads")),
                ("Graph hidden dimension", graph.get("hidden_dim")),
                ("Spatial layers", spatial.get("num_layers")),
                ("Spatial gate", spatial.get("gate_type")),
                ("Initial spatial beta", spatial.get("initial_beta")),
                ("Future token mode", heads.get("future_token_mode")),
                ("Prediction length", heads.get("prediction_length")),
                ("Future predictor", future.get("type")),
                ("Backbone learning rate", training.get("learning_rate")),
                ("Graph learning rate", training.get("graph_learning_rate")),
            ]
        )

    rows.extend(
        [
            ("Trainable parameters", metadata.get("trainable_parameter_count")),
            ("Training project commit", metadata.get("project_git_commit")),
        ]
    )
    return pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field")

# ============================================================================
# Models-to-use analysis API
# ============================================================================
#
# This API is intentionally path-driven rather than preset-driven.  Every
# public function accepts a model directory, an evaluation split, and (where
# relevant) an inference policy.  It supports the historical validation
# artefact layout and the canonical split layout used for frozen test outputs.

EvaluationSplit = Literal["train", "validation", "test"]
EvaluationSplitInput = Literal["train", "training", "val", "validation", "test"]


@dataclass(frozen=True)
class EvaluationArtifactPaths:
    """Resolved files for one model, split, and inference policy."""

    run_dir: Path
    split: EvaluationSplit
    policy: str
    policy_dir: Path
    predictions: Path
    graphs: Path | None
    metrics: Path | None
    sampled_paths: Path | None
    tokens: Path | None


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Validated saved outputs for one model/split/policy combination."""

    info: UnifiedRunInfo
    split: EvaluationSplit
    policy: str
    epoch: int | None
    prediction_result: dict[str, Any]
    graph_artifacts: dict[str, Any] | None
    metric_table: pd.DataFrame | None
    sampled_price_paths: dict[str, Any] | None
    token_artifacts: dict[str, Any] | None
    paths: EvaluationArtifactPaths


@dataclass(frozen=True)
class GraphWindowReport:
    """Graph plot and tables for one exact saved forecast window."""

    snapshot: GraphSnapshot
    summary: pd.DataFrame
    connections: pd.DataFrame
    window: pd.DataFrame
    figure: Figure
    axes: Axes
    plotted_adjacency: pd.DataFrame


@dataclass(frozen=True)
class PointForecastReport:
    """Point-forecast comparison for one date/window/asset."""

    figure: Figure
    axes: Axes
    values: pd.DataFrame
    window: pd.DataFrame


@dataclass(frozen=True)
class SamplePathWindowReport:
    """Ten-path forecast plot and values for one date/window/asset."""

    figure: Figure
    axes: Axes
    paths: pd.DataFrame
    horizons: pd.DataFrame
    window: pd.DataFrame


@dataclass(frozen=True)
class PredictiveCoverageReport:
    """Coverage, sharpness, CRPS, and finite-ensemble rank diagnostics."""

    overall: pd.DataFrame
    by_asset: pd.DataFrame
    probabilistic_scores: pd.DataFrame
    rank_histogram: pd.DataFrame
    calibration_figure: Figure
    calibration_axes: Axes


def resolve_models_to_use_root(root: str | Path | None = None) -> Path:
    """Resolve the single user-managed ``models_to_use`` directory.

    The helper checks the local Google Drive mount used on the author's Mac and
    the Colab mount.  Passing ``root`` bypasses auto-detection.
    """

    if root is not None:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"models_to_use directory does not exist: {path}")
        return path

    candidates = (
        Path(
            "/Users/vishalruparelia/Library/CloudStorage/"
            "GoogleDrive-vishal@autonomous-fox.ai/My Drive/"
            "dissertation/models_to_use"
        ),
        Path("/content/drive/MyDrive/dissertation/models_to_use"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate dissertation/models_to_use. Checked:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
    )


def resolve_model_folder(
    model: str | Path,
    *,
    models_root: str | Path | None = None,
) -> Path:
    """Resolve a direct run path or a folder name under ``models_to_use``."""

    direct = Path(model).expanduser()
    if direct.is_dir():
        return direct.resolve()

    root = resolve_models_to_use_root(models_root)
    candidate = root / str(model)
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Model folder {model!r} was not found under {root}."
        )
    return candidate.resolve()


def normalise_evaluation_split(split: EvaluationSplitInput | str) -> EvaluationSplit:
    """Return the canonical saved-artifact split name.

    Public Graph Hub functions accept ``"train"``, ``"val"``/``"validation"``
    and ``"test"``.  The analysis layer never launches inference.  When the
    requested split has not been saved, the loader raises a file error listing
    the paths it checked.
    """

    value = str(split).strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    if value not in aliases:
        raise ValueError(
            "split must be one of 'train', 'val'/'validation', or 'test'."
        )
    return aliases[value]  # type: ignore[return-value]


def discover_models(
    *,
    models_root: str | Path | None = None,
) -> pd.DataFrame:
    """List every direct child directory under ``models_to_use``.

    No folder-name whitelist is used.  A folder is shown even when it is
    incomplete, because the purpose of discovery is to expose what is present,
    not to silently decide which experiments count as models.
    """

    root = resolve_models_to_use_root(models_root)
    rows: list[dict[str, Any]] = []
    for folder in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        metadata_path = folder / "run_metadata.json"
        config_path = folder / "resolved_config.json"
        history_path = folder / "history.csv"
        row: dict[str, Any] = {
            "Folder": folder.name,
            "Path": str(folder.resolve()),
            "Run metadata": metadata_path.is_file(),
            "Resolved config": config_path.is_file(),
            "History": history_path.is_file(),
            "Best checkpoint": (folder / "best_checkpoint.pt").is_file(),
            "Last checkpoint": (folder / "last_checkpoint.pt").is_file(),
            "Saved train analysis": (folder / "analysis" / "train").is_dir(),
            "Saved validation analysis": (folder / "analysis" / "validation").is_dir(),
            "Saved test analysis": (folder / "analysis" / "test").is_dir(),
            "Family": None,
            "Status": None,
            "Graph type": None,
            "Issue": None,
        }
        try:
            if metadata_path.is_file():
                metadata = _load_json(metadata_path)
                row["Status"] = metadata.get("status")
            if metadata_path.is_file() and config_path.is_file():
                info = load_unified_run_info(folder)
                row["Family"] = info.run_kind
                row["Graph type"] = info.graph_type
        except Exception as error:
            row["Issue"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def _temperature_policy_name(temperature: float) -> str:
    text = f"{float(temperature):g}".replace("-", "m").replace(".", "p")
    return f"temperature_{text}"


def _selected_policy_from_json(path: Path) -> str | None:
    if not path.is_file():
        return None
    values = _load_json(path)
    for key in ("selected_policy", "policy", "selected_policy_name"):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    temperature = values.get("selected_temperature")
    if temperature is not None:
        return _temperature_policy_name(float(temperature))
    return None


def _selected_temperature_policy(run_dir: Path, split: EvaluationSplit) -> str | None:
    candidates = (
        run_dir / "analysis" / split / "temperature_selection.json",
        run_dir / f"{split}_inference" / "temperature_selection.json",
        run_dir / "temperature_sweep" / f"{split}_temperature_selection.json",
        run_dir / "temperature_sweep" / "temperature_selection.json",
        run_dir / "_experiment_control" / "temperature_selection.json",
    )
    for candidate in candidates:
        resolved = _selected_policy_from_json(candidate)
        if resolved is not None:
            return resolved
    return None


def _artifact_paths_in_directory(
    directory: Path,
    *,
    split: EvaluationSplit,
    policy: str,
    prefix: str,
) -> EvaluationArtifactPaths:
    return EvaluationArtifactPaths(
        run_dir=directory,
        split=split,
        policy=policy,
        policy_dir=directory,
        predictions=directory / f"{prefix}predictions.pt",
        graphs=directory / f"{prefix}graphs.pt",
        metrics=directory / f"{prefix}metric_table.csv",
        sampled_paths=directory / f"{prefix}sampled_price_paths.pt",
        tokens=directory / f"{prefix}tokens.pt",
    )



def _candidate_artifact_layouts(
    run_dir: Path,
    *,
    split: EvaluationSplit,
    policy: str,
) -> list[EvaluationArtifactPaths]:
    """Return supported saved-artifact layouts in priority order."""

    candidates: list[EvaluationArtifactPaths] = []

    # Canonical frozen-analysis layout.  An explicit inference policy must
    # never be shadowed by the split's base/best bundle.  The old ordering
    # checked ``analysis/<split>/predictions.pt`` first even when callers
    # requested ``temperature_1``, so sampled-path lookups silently loaded the
    # argmax bundle and then reported that no sampled paths existed.
    canonical_direct = run_dir / "analysis" / split
    base_policies = {"best", f"best_{split}", "default"}
    if policy in base_policies:
        candidates.append(
            _artifact_paths_in_directory(
                canonical_direct,
                split=split,
                policy=policy,
                prefix="",
            )
        )
    candidates.append(
        _artifact_paths_in_directory(
            canonical_direct / policy,
            split=split,
            policy=policy,
            prefix="",
        )
    )

    # Direct split-labelled files in the model root are also base-policy
    # artefacts and must not override an explicitly requested temperature.
    if policy in base_policies:
        candidates.append(
            _artifact_paths_in_directory(
                run_dir,
                split=split,
                policy=policy,
                prefix=f"{split}_",
            )
        )

    # Historical best-checkpoint layouts.  Validation is the established
    # production spelling; train/training are accepted for final-checkpoint
    # inference saved later by the baselines notebook.
    historical_prefixes: tuple[str, ...]
    if split == "validation":
        historical_prefixes = ("best_validation_",)
    elif split == "train":
        historical_prefixes = ("best_train_", "best_training_")
    else:
        historical_prefixes = ("best_test_",)

    if policy in {"best", f"best_{split}", "default"}:
        for prefix in historical_prefixes:
            candidates.append(
                _artifact_paths_in_directory(
                    run_dir,
                    split=split,
                    policy="best",
                    prefix=prefix,
                )
            )

    # Temperature-policy layouts used by the tokenized runner and by frozen
    # split inference.  The historical validation runner uses validation_*
    # filenames under temperature_sweep/<policy>.
    policy_directories = (
        run_dir / "temperature_sweep" / policy,
        run_dir / f"{split}_inference" / policy,
        run_dir / "analysis" / split / policy,
    )
    for directory in policy_directories:
        for prefix in (f"{split}_", ""):
            candidates.append(
                _artifact_paths_in_directory(
                    directory,
                    split=split,
                    policy=policy,
                    prefix=prefix,
                )
            )
    if split == "validation":
        candidates.append(
            _artifact_paths_in_directory(
                run_dir / "temperature_sweep" / policy,
                split=split,
                policy=policy,
                prefix="validation_",
            )
        )

    # Remove duplicate prediction paths while preserving priority.
    unique: list[EvaluationArtifactPaths] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.predictions in seen:
            continue
        seen.add(candidate.predictions)
        unique.append(candidate)
    return unique




def _available_saved_policies(
    run_dir: Path,
    *,
    split: EvaluationSplit,
) -> tuple[str, ...]:
    """Return policy directories that contain a prediction artefact."""

    names: set[str] = set()
    roots = (
        run_dir / "temperature_sweep",
        run_dir / f"{split}_inference",
        run_dir / "analysis" / split,
    )
    for root in roots:
        if not root.is_dir():
            continue
        for directory in root.iterdir():
            if not directory.is_dir():
                continue
            prefixes = ("", f"{split}_", "validation_" if split == "validation" else "")
            if any(
                prefix and (directory / f"{prefix}predictions.pt").is_file()
                for prefix in prefixes
            ) or (directory / "predictions.pt").is_file():
                names.add(directory.name)
    return tuple(sorted(names))


def resolve_evaluation_artifact_paths(
    run_dir: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
) -> EvaluationArtifactPaths:
    """Resolve saved analysis files without depending on a folder name.

    ``policy=None`` is the transparent automatic mode used by the final Graph
    Hub.  A saved temperature-selection manifest is honoured.  Otherwise a
    validation ``best_validation_*`` bundle or a direct train/test bundle is
    used.  If several tokenized policies exist but no selection manifest says
    which one is final, the function raises and lists the available policies.
    """

    run_path = _resolve_run_dir(run_dir)
    resolved_split = normalise_evaluation_split(split)
    requested = None if policy is None else str(policy).strip()
    if requested == "":
        raise ValueError("policy cannot be an empty string; use None for automatic selection.")

    automatic_request = requested in {None, "auto", "selected_temperature"}
    selected = _selected_temperature_policy(run_path, resolved_split)
    if automatic_request:
        if selected is not None:
            requested = selected
        elif resolved_split == "validation" and (
            run_path / "best_validation_predictions.pt"
        ).is_file():
            requested = "best"
        else:
            available = _available_saved_policies(
                run_path,
                split=resolved_split,
            )
            if len(available) == 1:
                requested = available[0]
            elif len(available) > 1:
                raise ValueError(
                    "Several saved inference policies are available but no "
                    "selection manifest identifies the final one. Pass policy="
                    f" explicitly. Available policies: {list(available)}"
                )
            else:
                requested = "default"

    assert requested is not None
    candidates = _candidate_artifact_layouts(
        run_path,
        split=resolved_split,
        policy=requested,
    )
    for candidate in candidates:
        if candidate.predictions.is_file():
            return EvaluationArtifactPaths(
                run_dir=run_path,
                split=candidate.split,
                policy=("best" if candidate.policy.startswith("best_") else candidate.policy),
                policy_dir=candidate.policy_dir,
                predictions=candidate.predictions,
                graphs=(candidate.graphs if candidate.graphs and candidate.graphs.is_file() else None),
                metrics=(candidate.metrics if candidate.metrics and candidate.metrics.is_file() else None),
                sampled_paths=(
                    candidate.sampled_paths
                    if candidate.sampled_paths and candidate.sampled_paths.is_file()
                    else None
                ),
                tokens=(candidate.tokens if candidate.tokens and candidate.tokens.is_file() else None),
            )

    # Historical selected-temperature manifests sometimes cover validation
    # only, while train/test retain the canonical direct analysis bundle.  In
    # automatic mode, fall back to that direct bundle only after the selected
    # policy-specific paths have been exhausted.  An explicitly requested
    # policy remains strict and never falls back silently.
    fallback_candidates: list[EvaluationArtifactPaths] = []
    if automatic_request and requested not in {"default", "best"}:
        fallback_candidates = _candidate_artifact_layouts(
            run_path,
            split=resolved_split,
            policy="default",
        )
        for candidate in fallback_candidates:
            if candidate.predictions.is_file():
                return EvaluationArtifactPaths(
                    run_dir=run_path,
                    split=candidate.split,
                    policy=(
                        "best"
                        if candidate.policy.startswith("best_")
                        else candidate.policy
                    ),
                    policy_dir=candidate.policy_dir,
                    predictions=candidate.predictions,
                    graphs=(
                        candidate.graphs
                        if candidate.graphs and candidate.graphs.is_file()
                        else None
                    ),
                    metrics=(
                        candidate.metrics
                        if candidate.metrics and candidate.metrics.is_file()
                        else None
                    ),
                    sampled_paths=(
                        candidate.sampled_paths
                        if candidate.sampled_paths
                        and candidate.sampled_paths.is_file()
                        else None
                    ),
                    tokens=(
                        candidate.tokens
                        if candidate.tokens and candidate.tokens.is_file()
                        else None
                    ),
                )

    checked_candidates = [*candidates, *fallback_candidates]
    checked = "\n".join(
        f"  - {candidate.predictions}"
        for candidate in checked_candidates
    )
    raise FileNotFoundError(
        f"No saved {resolved_split} prediction artefact was found for policy "
        f"{requested!r} in {run_path}. Graph Hub does not run inference. "
        "Generate the missing split artefacts in the inference/baselines "
        f"notebook, then retry. Checked:\n{checked}"
    )



def _canonical_graph_orientation(value: Any) -> str:
    """Normalise historical spellings of the canonical graph orientation."""

    if value is None:
        return "A[target, source]"
    text = re.sub(r"\s+", "", str(value).strip().lower())
    text = text.replace("rows", "row").replace("columns", "column")
    canonical_aliases = {
        "a[target,source]",
        "row=target,column=source",
        "row:target,column:source",
        "target,source",
        "target_source",
        "target-by-source",
    }
    reversed_aliases = {
        "a[source,target]",
        "row=source,column=target",
        "row:source,column:target",
        "source,target",
        "source_target",
        "source-by-target",
    }
    if text in canonical_aliases:
        return "A[target, source]"
    if text in reversed_aliases:
        raise ValueError(
            "Saved graph uses A[source, target], but this project requires "
            "A[target, source]. Transpose explicitly at artifact creation; "
            "the analysis code will not silently reverse it."
        )
    raise ValueError(f"Unexpected graph orientation: {value!r}.")


def _normalise_saved_graph_tensor(
    values: Any,
    *,
    windows: int,
    nodes: int,
    expected_heads: int,
    name: str,
) -> Tensor:
    saved_tensor = torch.as_tensor(values).detach().cpu()
    saved_dtype = saved_tensor.dtype
    tensor = saved_tensor.float()
    if tensor.ndim == 2:
        if tuple(tensor.shape) != (nodes, nodes):
            raise ValueError(f"Graph component {name!r} has wrong node shape.")
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(windows, 1, -1, -1)
    elif tensor.ndim == 3:
        if tuple(tensor.shape[-2:]) != (nodes, nodes):
            raise ValueError(f"Graph component {name!r} has wrong node shape.")
        if int(tensor.shape[0]) == windows and expected_heads == 1:
            tensor = tensor.unsqueeze(1)
        else:
            tensor = tensor.unsqueeze(0).expand(windows, -1, -1, -1)
    elif tensor.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Graph component {name!r} must have shape [N,N], [G,N,N], "
            "[W,N,N], or [W,G,N,N]."
        )

    if int(tensor.shape[0]) != windows:
        raise ValueError(
            f"Graph component {name!r} has {int(tensor.shape[0])} windows; "
            f"expected {windows}."
        )
    if tuple(tensor.shape[-2:]) != (nodes, nodes):
        raise ValueError(f"Graph component {name!r} has wrong node shape.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Graph component {name!r} contains non-finite values.")
    if (tensor < -1.0e-7).any():
        raise ValueError(f"Graph component {name!r} contains negative weights.")
    row_sums = tensor.sum(dim=-1)
    if (row_sums <= 0.0).any():
        raise ValueError(f"Graph component {name!r} contains an empty row.")
    tolerance = _saved_graph_row_sum_tolerance(saved_dtype)
    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=tolerance,
        rtol=0.0,
    ):
        maximum_error = float((row_sums - 1.0).abs().max().item())
        raise ValueError(
            f"Graph component {name!r} is not row-stochastic; maximum "
            f"row-sum error={maximum_error:.3e} exceeds the "
            f"{saved_dtype} storage tolerance {tolerance:.1e}."
        )

    # Accepted storage drift is quantisation, not model behaviour.  Restore
    # exact FP32 row stochasticity before all downstream graph calculations.
    tensor = tensor / row_sums.unsqueeze(-1)
    return tensor.contiguous()


def _historical_single_layer_graph_contract(
    *,
    graph: Mapping[str, Any],
    info: UnifiedRunInfo,
) -> bool:
    """Return whether top-level graph components unambiguously mean layer 0.

    Historical ``ContinuousForecaster`` exports predate the per-layer graph
    schema. They contain one graph/spatial stage and save only ``selected``,
    ``base`` and ``dynamic``. New interlaced models advertise an explicit
    layer/head schedule and must continue to provide their real per-layer
    tensors; this fallback is therefore restricted to artefacts with no
    evidence of more than one graph layer.
    """

    explicit_counts: list[int] = []
    if info.num_heads_per_layer is not None:
        explicit_counts.append(len(info.num_heads_per_layer))

    artifact_layers = graph.get("num_layers")
    if artifact_layers is not None:
        explicit_counts.append(int(artifact_layers))

    model = info.resolved_config.get("model", {})
    if isinstance(model, Mapping):
        temporal_stack = model.get("temporal_stack")
        if isinstance(temporal_stack, Mapping):
            value = temporal_stack.get("num_st_blocks")
            if value is not None:
                explicit_counts.append(int(value))
        for key in ("num_st_blocks", "st_blocks"):
            value = model.get(key)
            if value is not None and not isinstance(value, Mapping):
                explicit_counts.append(int(value))

    metadata = info.run_metadata
    for key in ("num_st_blocks", "st_blocks", "graph_layers"):
        value = metadata.get(key)
        if value is not None:
            explicit_counts.append(int(value))

    basedygraph = info.resolved_config.get("basedygraph_financial")
    if isinstance(basedygraph, Mapping):
        value = basedygraph.get("num_st_blocks")
        if value is not None:
            explicit_counts.append(int(value))

    if any(count > 1 for count in explicit_counts):
        return False
    if explicit_counts and any(count != 1 for count in explicit_counts):
        return False
    return graph.get("selected") is not None


def _synthesise_historical_single_layer_components(
    graph: dict[str, Any],
    *,
    info: UnifiedRunInfo,
) -> None:
    """Add per-layer aliases for an old one-graph continuous artefact."""

    if any(
        graph.get(key) is not None
        for key in ("per_layer", "per_layer_base", "per_layer_dynamic")
    ):
        return
    if not _historical_single_layer_graph_contract(graph=graph, info=info):
        return

    graph["num_layers"] = 1
    graph["selected_layer"] = 0
    graph["num_heads_per_layer"] = [int(info.num_heads)]
    graph["layer_head_counts"] = [int(info.num_heads)]
    graph["per_layer"] = (graph.get("selected"),)
    graph["per_layer_base"] = (graph.get("base"),)
    graph["per_layer_dynamic"] = (graph.get("dynamic"),)


def _normalise_graph_payload(
    graph_artifacts: Mapping[str, Any],
    *,
    info: UnifiedRunInfo,
    prediction_result: Mapping[str, Any],
    windows: int,
) -> dict[str, Any]:
    graph = dict(graph_artifacts)
    orientation = graph.get(
        "graph_orientation",
        graph.get("orientation", "A[target, source]"),
    )
    graph["graph_orientation"] = _canonical_graph_orientation(orientation)
    graph["orientation"] = "A[target, source]"

    artifact_assets = graph.get("asset_cols")
    if artifact_assets is not None and tuple(map(str, artifact_assets)) != info.asset_cols:
        raise ValueError("Saved graph asset order differs from run metadata.")
    graph["asset_cols"] = list(info.asset_cols)

    for key in ("selected", "base", "dynamic"):
        if graph.get(key) is not None:
            graph[key] = _normalise_saved_graph_tensor(
                graph[key],
                windows=windows,
                nodes=info.num_nodes,
                expected_heads=info.num_heads,
                name=key,
            )

    _synthesise_historical_single_layer_components(
        graph,
        info=info,
    )

    for key in ("per_layer", "per_layer_base", "per_layer_dynamic"):
        per_layer = graph.get(key)
        if not (
            isinstance(per_layer, Sequence)
            and not isinstance(per_layer, (str, bytes))
        ):
            continue
        if (
            info.num_heads_per_layer is not None
            and len(per_layer) != len(info.num_heads_per_layer)
        ):
            raise ValueError(
                f"Saved {key} graph count does not match the configured "
                f"head-count schedule: {len(per_layer)} vs "
                f"{len(info.num_heads_per_layer)}."
            )
        graph[key] = tuple(
            None
            if values is None
            else _normalise_saved_graph_tensor(
                values,
                windows=windows,
                nodes=info.num_nodes,
                expected_heads=(
                    info.num_heads
                    if info.num_heads_per_layer is None
                    else info.num_heads_per_layer[index]
                ),
                name=f"{key}[{index}]",
            )
            for index, values in enumerate(per_layer)
        )

    # Continuous graph artefacts historically stored dates but not origin/sample
    # indices.  The corresponding prediction artefact does store them.  Merge
    # all window metadata here so every downstream function sees one contract.
    metadata_aliases = {
        "dates": ("dates", "date", "days", "day"),
        "sample_idx": ("sample_idx",),
        "origin_idx": ("origin_idx",),
        "target_indices": ("target_indices",),
        "window_idx": ("window_idx",),
    }
    for destination, aliases in metadata_aliases.items():
        if graph.get(destination) is not None:
            continue
        for source in aliases:
            if prediction_result.get(source) is not None:
                graph[destination] = prediction_result[source]
                break

    dates = graph.get("dates")
    if dates is not None:
        if isinstance(dates, str):
            dates = [dates]
        dates = list(dates)
        if len(dates) != windows:
            raise ValueError("Saved graph dates do not align with window count.")
        graph["dates"] = dates

    for name in ("sample_idx", "origin_idx"):
        values = graph.get(name)
        if values is not None and int(torch.as_tensor(values).numel()) != windows:
            raise ValueError(f"Saved graph {name} does not align with window count.")
    return graph


def _unwrap_token_artifacts(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Saved token artefact must be a mapping.")
    nested = payload.get("token_artifacts")
    return dict(nested if isinstance(nested, Mapping) else payload)


def _augment_common_metrics_from_saved_predictions(
    metric_table: pd.DataFrame,
    prediction_result: Mapping[str, Any],
) -> pd.DataFrame:
    """Backfill evaluation-only metrics added after a run was saved.

    Graph Hub treats the exact saved prediction path as authoritative.
    Metrics in this allow-list require no training-derived state, so they
    can be reconstructed without rerunning inference or changing model
    selection.  The saved CSV is not modified; augmentation is in memory.
    """
    observed = set(
        metric_table["metric"]
        .astype(str)
        .tolist()
    )

    backfillable_metrics = (
        "cumulative_log_change_directional_accuracy",
    )

    missing = [
        metric_name
        for metric_name in backfillable_metrics
        if metric_name not in observed
    ]

    if not missing:
        return metric_table

    if prediction_result.get("output_space", "raw") != "raw":
        return metric_table

    evaluator = ForecastEvaluator(
        prediction_result=dict(prediction_result),
    )

    recomputable = [
        metric_name
        for metric_name in missing
        if metric_name in evaluator.available_metrics
    ]

    if not recomputable:
        return metric_table

    results = evaluator.evaluate(
        metrics=recomputable,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    additional = make_evaluation_table(
        metric_results=results,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    return pd.concat(
        [metric_table, additional],
        ignore_index=True,
    ).drop_duplicates(
        subset=["metric", "horizon", "channel"],
        keep="first",
    )


def load_evaluation_artifacts(
    run_dir: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    require_graph: bool = False,
    require_metrics: bool = False,
    require_sampled_paths: bool = False,
) -> EvaluationArtifacts:
    """Load one model's saved validation/test outputs under a common contract."""

    info = load_unified_run_info(run_dir)
    resolved_split = normalise_evaluation_split(split)
    paths = resolve_evaluation_artifact_paths(
        info.run_dir,
        split=resolved_split,
        policy=policy,
    )
    prediction_result, prediction_epoch = _unwrap_prediction_result(
        _torch_load(paths.predictions)
    )
    windows = _validate_prediction_result(prediction_result, info)

    graph: dict[str, Any] | None = None
    graph_epoch: int | None = None
    if paths.graphs is not None:
        raw_graph, graph_epoch = _unwrap_graph_artifacts(_torch_load(paths.graphs))
        graph = _normalise_graph_payload(
            raw_graph,
            info=info,
            prediction_result=prediction_result,
            windows=windows,
        )
    elif require_graph:
        raise FileNotFoundError(
            f"No {split} graph artefact was found for {info.run_dir}."
        )

    if prediction_epoch is not None and graph_epoch is not None:
        if int(prediction_epoch) != int(graph_epoch):
            raise ValueError(
                "Prediction and graph artefacts come from different epochs: "
                f"{prediction_epoch} vs {graph_epoch}."
            )

    selected_epoch = prediction_epoch if prediction_epoch is not None else graph_epoch
    if paths.split == "train":
        expected_epoch = info.run_metadata.get("best_epoch")
        if expected_epoch is None:
            raise ValueError(
                "Training-split analysis requires run_metadata.best_epoch so "
                "the final selected checkpoint can be verified."
            )
        if selected_epoch is None:
            raise ValueError(
                "Training-split artefacts must record their checkpoint epoch. "
                "Save predictions/graphs inside an {'epoch': ..., ...} wrapper."
            )
        if int(selected_epoch) != int(expected_epoch):
            raise ValueError(
                "Training-split artefacts were not generated from the final "
                f"selected checkpoint: saved epoch={selected_epoch}, "
                f"run best_epoch={expected_epoch}."
            )

    metric_table = pd.read_csv(paths.metrics) if paths.metrics is not None else None
    if require_metrics and metric_table is None:
        raise FileNotFoundError(
            f"No {split} metric table was found for {info.run_dir}."
        )
    if metric_table is not None:
        required_columns = {"metric", "horizon", "channel", "value"}
        missing = required_columns - set(metric_table.columns)
        if missing:
            raise ValueError(
                f"Saved metric table is missing columns {sorted(missing)}."
            )
        metric_table = _augment_common_metrics_from_saved_predictions(
            metric_table,
            prediction_result,
        )

    sampled = (
        _unwrap_sampled_price_paths(_torch_load(paths.sampled_paths))
        if paths.sampled_paths is not None
        else None
    )
    if require_sampled_paths and sampled is None:
        raise FileNotFoundError(
            f"No {split} sampled-price-path artefact was found for {info.run_dir}."
        )

    tokens = (
        _unwrap_token_artifacts(_torch_load(paths.tokens))
        if paths.tokens is not None
        else None
    )

    return EvaluationArtifacts(
        info=info,
        split=paths.split,
        policy=paths.policy,
        epoch=(prediction_epoch if prediction_epoch is not None else graph_epoch),
        prediction_result=prediction_result,
        graph_artifacts=graph,
        metric_table=metric_table,
        sampled_price_paths=sampled,
        token_artifacts=tokens,
        paths=paths,
    )




def make_model_artifact_audit(
    models: Mapping[str, str | Path],
    *,
    split: EvaluationSplitInput | str = "validation",
    policies: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """Audit every model without allowing one missing artefact to hide the rest."""

    rows: list[dict[str, Any]] = []
    for label, model in models.items():
        try:
            run_dir = resolve_model_folder(model)
            policy = None if policies is None else policies.get(label)
            artifacts = load_evaluation_artifacts(
                run_dir,
                split=split,
                policy=policy,
            )
            y_pred = torch.as_tensor(artifacts.prediction_result["y_pred"])
            graph = artifacts.graph_artifacts
            selected = None if graph is None else graph.get("selected")
            temperature = None
            sample_count = None
            if artifacts.sampled_price_paths is not None:
                temperature = artifacts.sampled_price_paths.get("temperature")
                sample_count = artifacts.sampled_price_paths.get("sample_count")
            rows.append(
                {
                    "Model": str(label),
                    "Folder": run_dir.name,
                    "Family": artifacts.info.run_kind,
                    "Split": artifacts.split,
                    "Policy": artifacts.policy,
                    "Graph type": artifacts.info.graph_type,
                    "Windows": int(y_pred.shape[0]),
                    "Horizons": str(tuple(artifacts.info.horizons)),
                    "Assets": artifacts.info.num_nodes,
                    "Predictions": True,
                    "Graphs": selected is not None,
                    "Metrics": artifacts.metric_table is not None,
                    "Sampled paths": artifacts.sampled_price_paths is not None,
                    "Temperature": temperature,
                    "Sample count": sample_count,
                    "Ready": True,
                    "Issue": None,
                }
            )
        except Exception as error:  # audit should report rather than mask peers
            rows.append(
                {
                    "Model": str(label),
                    "Folder": str(model),
                    "Family": None,
                    "Split": split,
                    "Policy": None,
                    "Graph type": None,
                    "Windows": None,
                    "Horizons": None,
                    "Assets": None,
                    "Predictions": False,
                    "Graphs": False,
                    "Metrics": False,
                    "Sampled paths": False,
                    "Temperature": None,
                    "Sample count": None,
                    "Ready": False,
                    "Issue": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)


def style_model_artifact_audit(
    table: pd.DataFrame,
    *,
    caption: str = "Saved model artefact audit",
) -> pd.io.formats.style.Styler:
    formats = {
        "Temperature": "{:.3f}",
        "Sample count": "{:.0f}",
        "Windows": "{:.0f}",
        "Assets": "{:.0f}",
    }
    return table.style.format(
        {key: value for key, value in formats.items() if key in table.columns},
        na_rep="—",
    ).set_caption(caption)


def make_model_architecture_comparison(
    models: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Place the exact saved architecture settings side by side."""

    columns: dict[str, pd.Series] = {}
    for label, model in models.items():
        run_dir = resolve_model_folder(model)
        table = make_unified_model_summary_table(run_dir)
        columns[str(label)] = table["Value"]
    return pd.DataFrame(columns)


def make_model_metric_comparison(
    models: Mapping[str, str | Path],
    *,
    split: EvaluationSplitInput | str = "validation",
    policies: Mapping[str, str | None] | None = None,
    channel: str = "close",
    metrics_to_display: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one model-by-horizon table from the saved common evaluator metrics."""

    metric_order = list(
        DEFAULT_SUMMARY_METRICS
        if metrics_to_display is None
        else [str(value) for value in metrics_to_display]
    )
    frames: list[pd.DataFrame] = []
    for label, model in models.items():
        policy = None if policies is None else policies.get(label)
        artifacts = load_evaluation_artifacts(
            resolve_model_folder(model),
            split=split,
            policy=policy,
            require_metrics=True,
        )
        assert artifacts.metric_table is not None
        selected = artifacts.metric_table.loc[
            artifacts.metric_table["channel"].astype(str).str.lower().eq(channel.lower()),
            ["metric", "horizon", "value"],
        ].copy()
        selected["horizon"] = pd.to_numeric(selected["horizon"], errors="raise").astype(int)
        if selected.duplicated(["metric", "horizon"]).any():
            raise ValueError(f"Metric table for {label!r} has duplicate rows.")
        wide = (
            selected.pivot(index="horizon", columns="metric", values="value")
            .reindex(index=artifacts.info.horizons, columns=metric_order)
            .rename(
                columns={
                    name: DEFAULT_METRIC_DISPLAY_NAMES.get(name, name)
                    for name in metric_order
                }
            )
        )
        wide.insert(0, "Model", str(label))
        wide.insert(1, "Split", artifacts.split)
        wide.insert(2, "Policy", artifacts.policy)
        wide.insert(3, "Horizon", [f"{int(value)} min" for value in wide.index])
        frames.append(wide.reset_index(drop=True))
    if not frames:
        raise ValueError("At least one model must be supplied.")
    return pd.concat(frames, ignore_index=True).set_index(
        ["Model", "Split", "Policy", "Horizon"]
    )


def style_model_metric_comparison(
    table: pd.DataFrame,
    *,
    caption: str,
) -> pd.io.formats.style.Styler:
    """Format only numeric metric columns, avoiding mixed-type Styler errors."""

    formatters: dict[str, str] = {}
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            continue
        if column in {"Log MAE", "Log MedAE", "Log P95 AE"}:
            formatters[column] = "{:.8f}"
        elif column == "MASE":
            formatters[column] = "{:.5f}"
        elif column == "Sign Acc.":
            formatters[column] = "{:.2%}"
        else:
            formatters[column] = "{:.6f}"
    return table.style.format(formatters, na_rep="—").set_caption(caption)


def style_numeric_table(
    table: pd.DataFrame,
    *,
    caption: str | None = None,
    precision: int = 6,
) -> pd.io.formats.style.Styler:
    """Safely format numeric columns in a mixed text/numeric table."""

    formatter = f"{{:.{int(precision)}f}}"
    formatters = {
        column: formatter
        for column in table.columns
        if pd.api.types.is_numeric_dtype(table[column])
    }
    styler = table.style.format(formatters, na_rep="—")
    return styler if caption is None else styler.set_caption(caption)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def plot_training_diagnostics(
    run_dir: str | Path,
    *,
    figsize: tuple[float, float] = (12.0, 4.5),
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Plot the saved primary training/validation score and graph/gate trajectory."""

    run_path = resolve_model_folder(run_dir)
    history_path = run_path / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"History is empty: {history_path}")
    epoch_column = _first_existing_column(history, ("epoch", "Epoch"))
    if epoch_column is None:
        history = history.copy()
        history["epoch"] = np.arange(1, len(history) + 1)
        epoch_column = "epoch"

    train_column = _first_existing_column(
        history,
        (
            "train_token_loss",
            "training_loss",
            "train_loss",
            "training_native_loss",
            "training_native_log_mae",
            "training_objective_loss",
        ),
    )
    validation_column = _first_existing_column(
        history,
        ("validation_token_loss", "selection_score", "validation_loss", "val_loss"),
    )
    entropy_column = _first_existing_column(
        history,
        (
            "validation_graph_mean_row_entropy",
            "graph_mean_row_entropy",
            "test_graph_mean_row_entropy",
            "training_graph_mean_row_entropy",
        ),
    )
    beta_column = _first_existing_column(history, ("spatial_beta", "validation_spatial_beta"))

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    if train_column is not None:
        axes[0].plot(history[epoch_column], history[train_column], label=train_column)
    if validation_column is not None:
        axes[0].plot(history[epoch_column], history[validation_column], label=validation_column)
    axes[0].set_title(f"{run_path.name} — predictive objective")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.25)
    if axes[0].lines:
        axes[0].legend()

    if entropy_column is not None:
        axes[1].plot(history[epoch_column], history[entropy_column], label="Mean row entropy")
    if beta_column is not None:
        axes[1].plot(history[epoch_column], history[beta_column], label="Spatial beta")
    axes[1].set_title("Graph concentration and learned graph exposure")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.25)
    if axes[1].lines:
        axes[1].legend()
    figure.tight_layout()

    selected_columns = [epoch_column]
    for column in (train_column, validation_column, entropy_column, beta_column):
        if column is not None and column not in selected_columns:
            selected_columns.append(column)
    return figure, axes, history[selected_columns].copy()


def _normalise_date_optional(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return _normalise_date(value)


def _clock_time(clean_index: int) -> str:
    timestamp = pd.Timestamp("2000-01-01 09:31") + pd.Timedelta(minutes=int(clean_index))
    return timestamp.strftime("%H:%M")


def _window_time_values(origin_idx: int | None, context_length: int, forecast_length: int) -> dict[str, Any]:
    if origin_idx is None:
        return {
            "Context start time": None,
            "Forecast origin time": None,
            "Forecast start time": None,
            "Forecast end time": None,
            "Time window": None,
        }
    origin = int(origin_idx)
    context_start = origin - int(context_length) + 1
    forecast_start = origin + 1
    forecast_end = origin + int(forecast_length)
    return {
        "Context start time": _clock_time(context_start),
        "Forecast origin time": _clock_time(origin),
        "Forecast start time": _clock_time(forecast_start),
        "Forecast end time": _clock_time(forecast_end),
        "Time window": (
            f"context {_clock_time(context_start)}–{_clock_time(origin)}; "
            f"forecast {_clock_time(forecast_start)}–{_clock_time(forecast_end)}"
        ),
    }


def _context_and_prediction_lengths(info: UnifiedRunInfo) -> tuple[int, int]:
    if info.run_kind == "continuous":
        data = info.resolved_config.get("data", {})
        return int(data.get("context_length", 60)), max(info.horizons)
    model = info.resolved_config["models"]["dynamic_graph"]
    return int(model.get("context_length", 60)), int(
        model.get("heads", {}).get("prediction_length", max(info.horizons))
    )


def make_evaluation_window_table(
    run_dir: str | Path,
    *,
    split: EvaluationSplit = "validation",
    policy: str = "auto",
) -> pd.DataFrame:
    """List exact saved forecast windows, dates, origins, and clock times."""

    artifacts = load_evaluation_artifacts(
        resolve_model_folder(run_dir),
        split=split,
        policy=policy,
    )
    windows = int(torch.as_tensor(artifacts.prediction_result["y_pred"]).shape[0])
    graph = artifacts.graph_artifacts or {}

    raw_dates = graph.get("dates", artifacts.prediction_result.get("dates"))
    if raw_dates is None:
        dates: list[str | None] = [None] * windows
    else:
        if isinstance(raw_dates, str):
            raw_dates = [raw_dates]
        if len(raw_dates) != windows:
            raise ValueError("Saved date metadata does not match window count.")
        dates = [_normalise_date_optional(value) for value in raw_dates]

    def optional_vector(name: str) -> list[int | None]:
        values = graph.get(name, artifacts.prediction_result.get(name))
        if values is None:
            return [None] * windows
        tensor = torch.as_tensor(values).reshape(-1)
        if int(tensor.numel()) != windows:
            raise ValueError(f"Saved {name} does not match window count.")
        return [int(value) for value in tensor.tolist()]

    origins = optional_vector("origin_idx")
    samples = optional_vector("sample_idx")
    context_length, forecast_length = _context_and_prediction_lengths(artifacts.info)
    records: list[dict[str, Any]] = []
    for index in range(windows):
        record = {
            "Global window index": index,
            "Date": dates[index],
            "Sample index": samples[index],
            "Origin index": origins[index],
        }
        record.update(
            _window_time_values(origins[index], context_length, forecast_length)
        )
        records.append(record)
    table = pd.DataFrame(records)
    if table["Date"].notna().any():
        table["Window within date"] = table.groupby("Date", dropna=False).cumcount() + 1
    else:
        table["Window within date"] = np.arange(1, windows + 1)
    return table[
        [
            "Global window index",
            "Date",
            "Window within date",
            "Sample index",
            "Origin index",
            "Context start time",
            "Forecast origin time",
            "Forecast start time",
            "Forecast end time",
            "Time window",
        ]
    ]


def _resolve_evaluation_window(
    table: pd.DataFrame,
    *,
    date: str | None,
    window_within_date: int | None,
    global_window_index: int | None,
    origin_idx: int | None = None,
) -> pd.Series:
    return _resolve_saved_window(
        table,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=origin_idx,
    )


def select_model_graph_snapshot(
    run_dir: str | Path,
    *,
    split: EvaluationSplit = "validation",
    policy: str = "auto",
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    date: str | None = None,
    window_within_date: int | None = None,
    global_window_index: int | None = None,
    origin_idx: int | None = None,
) -> GraphSnapshot:
    """Select the actual adjacency used for one exact saved forecast window."""

    artifacts = load_evaluation_artifacts(
        resolve_model_folder(run_dir),
        split=split,
        policy=policy,
        require_graph=True,
    )
    assert artifacts.graph_artifacts is not None
    table = make_evaluation_window_table(
        artifacts.info.run_dir,
        split=split,
        policy=policy,
    )
    row = _resolve_evaluation_window(
        table,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=origin_idx,
    )
    absolute_index = int(row["Global window index"])
    graph = artifacts.graph_artifacts

    if component == "selected" and layer != -1:
        per_layer = graph.get("per_layer")
        if not isinstance(per_layer, Sequence):
            raise ValueError("Saved graph artefact does not contain per_layer.")
        layer_index = _normalise_layer_index(layer, len(per_layer))
        values = per_layer[layer_index]
    else:
        values = graph.get(component)
    if values is None:
        raise ValueError(f"Graph component {component!r} is unavailable.")

    tensor = torch.as_tensor(values).detach().cpu().double()
    if tensor.ndim != 4:
        raise ValueError("Normalised saved graph must have shape [W,G,N,N].")
    adjacency = _select_graph_head(tensor[absolute_index], head)
    row_entropy = -(
        adjacency.clamp_min(1.0e-12) * adjacency.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    available_neighbours = adjacency.shape[-1] - (0 if artifacts.info.add_self_loops else 1)
    top_k = max(1, min(5, available_neighbours))
    top5_mass = adjacency.topk(top_k, dim=-1).values.sum(dim=-1)

    frame = pd.DataFrame(
        adjacency.numpy(),
        index=artifacts.info.asset_cols,
        columns=artifacts.info.asset_cols,
    )
    frame.index.name = "Target"
    frame.columns.name = "Source"

    snapshot = GraphSnapshot(
        adjacency=frame,
        run_name=artifacts.info.run_dir.name,
        run_kind=artifacts.info.run_kind,
        policy=artifacts.policy,
        component=component,
        head=head,
        global_window_index=absolute_index,
        date=(None if pd.isna(row["Date"]) else str(row["Date"])),
        window_within_date=int(row["Window within date"]),
        sample_idx=(None if pd.isna(row["Sample index"]) else int(row["Sample index"])),
        origin_idx=(None if pd.isna(row["Origin index"]) else int(row["Origin index"])),
        mean_row_entropy=float(row_entropy.mean().item()),
        mean_effective_neighbours=float(row_entropy.exp().mean().item()),
        median_effective_neighbours=float(row_entropy.exp().median().item()),
        maximum_edge_weight=float(adjacency.max().item()),
        mean_top5_row_mass=float(top5_mass.mean().item()),
        graph_type=artifacts.info.graph_type,
        add_self_loops=artifacts.info.add_self_loops,
    )
    # Store the split/time metadata in attrs without changing the historical
    # GraphSnapshot constructor used by older notebooks.
    snapshot.adjacency.attrs["split"] = artifacts.split
    snapshot.adjacency.attrs["time_window"] = row["Time window"]
    snapshot.adjacency.attrs["forecast_origin_time"] = row["Forecast origin time"]
    return snapshot


def make_model_graph_snapshot_summary(snapshot: GraphSnapshot) -> pd.DataFrame:
    """Return concentration and window provenance for one exact adjacency."""

    return pd.DataFrame(
        [
            {
                "Run": snapshot.run_name,
                "Family": snapshot.run_kind,
                "Split": snapshot.adjacency.attrs.get("split"),
                "Policy": snapshot.policy,
                "Graph type": snapshot.graph_type,
                "Component": snapshot.component,
                "Head": snapshot.head,
                "Date": snapshot.date,
                "Window within date": snapshot.window_within_date,
                "Time window": snapshot.adjacency.attrs.get("time_window"),
                "Global window index": snapshot.global_window_index,
                "Origin index": snapshot.origin_idx,
                "Mean row entropy": snapshot.mean_row_entropy,
                "Mean effective neighbours": snapshot.mean_effective_neighbours,
                "Median effective neighbours": snapshot.median_effective_neighbours,
                "Maximum edge weight": snapshot.maximum_edge_weight,
                "Mean top-5 row mass": snapshot.mean_top5_row_mass,
            }
        ]
    )


def plot_model_graph_snapshot(
    snapshot: GraphSnapshot,
    *,
    cluster: bool = False,
    cluster_method: str = "average",
    figsize: tuple[float, float] = (13.0, 11.0),
    tick_fontsize: float = 8.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot actual adjacency weights with exact date and clock-time provenance."""

    matrix = snapshot.adjacency.to_numpy(dtype=np.float64, copy=True)
    labels = np.asarray(snapshot.adjacency.index, dtype=object)
    if cluster:
        order = _cluster_graph_order(matrix, method=cluster_method)
        matrix = matrix[np.ix_(order, order)]
        labels = labels[order]

    plotted_values = matrix.copy()
    if not snapshot.add_self_loops:
        np.fill_diagonal(plotted_values, np.nan)
    finite = plotted_values[np.isfinite(plotted_values)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    image = axes.imshow(
        plotted_values,
        cmap=cmap,
        vmin=0.0,
        vmax=maximum,
        interpolation="nearest",
        aspect="equal",
    )
    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    axes.set_yticklabels(labels, fontsize=tick_fontsize)
    axes.set_xlabel("Source asset (influences target)")
    axes.set_ylabel("Target asset (receives influence)")
    date_label = snapshot.date or "date unavailable"
    window_label = snapshot.window_within_date or "?"
    time_label = snapshot.adjacency.attrs.get("time_window") or "clock time unavailable"
    axes.set_title(
        f"{snapshot.run_name} — {snapshot.policy}\n"
        f"{date_label}, window {window_label} — {time_label}\n"
        f"mean row entropy={snapshot.mean_row_entropy:.4f}; "
        f"effective neighbours={snapshot.mean_effective_neighbours:.2f}"
    )
    colourbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    colourbar.set_label("Adjacency weight")
    figure.tight_layout()
    plotted = pd.DataFrame(matrix, index=labels, columns=labels)
    plotted.index.name = "Target"
    plotted.columns.name = "Source"
    return figure, axes, plotted


def analyse_graph_window(
    run_dir: str | Path,
    *,
    split: EvaluationSplit = "validation",
    policy: str = "auto",
    date: str | None = None,
    window_within_date: int | None = None,
    global_window_index: int | None = None,
    origin_idx: int | None = None,
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
    cluster: bool = False,
) -> GraphWindowReport:
    """Create the plot, summary, and connections table for one graph window."""

    snapshot = select_model_graph_snapshot(
        run_dir,
        split=split,
        policy=policy,
        component=component,
        layer=layer,
        head=head,
        date=date,
        window_within_date=window_within_date,
        global_window_index=global_window_index,
        origin_idx=origin_idx,
    )
    summary = make_model_graph_snapshot_summary(snapshot)
    connections = make_graph_snapshot_connections_table(
        snapshot,
        top_n=top_n,
        direction=direction,
    )
    figure, axes, plotted = plot_model_graph_snapshot(snapshot, cluster=cluster)
    window = make_evaluation_window_table(
        run_dir,
        split=split,
        policy=policy,
    ).loc[
        lambda frame: frame["Global window index"] == snapshot.global_window_index
    ]
    return GraphWindowReport(
        snapshot=snapshot,
        summary=summary,
        connections=connections,
        window=window,
        figure=figure,
        axes=axes,
        plotted_adjacency=plotted,
    )


def plot_point_forecast_comparison(
    models: Mapping[str, str | Path],
    *,
    split: EvaluationSplitInput | str = "validation",
    policies: Mapping[str, str | None] | None = None,
    asset: str,
    date: str,
    window_within_date: int,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> PointForecastReport:
    """Overlay all model point forecasts against one common realised path."""

    figure, axes = plt.subplots(figsize=figsize)
    records: list[dict[str, Any]] = []
    reference_truth_by_horizon: dict[int, float] = {}
    reference_last: float | None = None
    reference_window: pd.DataFrame | None = None
    reference_origin: int | None = None

    for label, model in models.items():
        policy = None if policies is None else policies.get(label)
        artifacts = load_evaluation_artifacts(
            resolve_model_folder(model),
            split=split,
            policy=policy,
        )
        table = make_evaluation_window_table(
            artifacts.info.run_dir,
            split=split,
            policy=policy,
        )
        row = _resolve_evaluation_window(
            table,
            date=date,
            window_within_date=window_within_date,
            global_window_index=None,
        )
        window_index = int(row["Global window index"])
        if asset not in artifacts.info.asset_cols:
            raise ValueError(f"Asset {asset!r} is absent from {label!r}.")
        asset_index = artifacts.info.asset_cols.index(asset)
        prediction = torch.as_tensor(artifacts.prediction_result["y_pred"])[
            window_index, :, asset_index, 0
        ].float()
        truth = torch.as_tensor(artifacts.prediction_result["y_true"])[
            window_index, :, asset_index, 0
        ].float()
        last = float(
            torch.as_tensor(artifacts.prediction_result["last_context_target"])[
                window_index, asset_index, 0
            ].item()
        )
        current_origin = row.get("Origin index")
        current_origin = None if pd.isna(current_origin) else int(current_origin)
        if reference_window is None:
            reference_last = last
            reference_window = table.loc[[row.name]].copy()
            reference_origin = current_origin
        else:
            if current_origin != reference_origin:
                raise ValueError(
                    "Models selected different forecast origins for the same "
                    "date/window request. Compare an exact common origin."
                )
            if abs(last - float(reference_last)) > 1.0e-6:
                raise ValueError("Models do not share the same last observed Close.")

        horizons = np.asarray(artifacts.info.horizons, dtype=int)
        for horizon, actual in zip(horizons, truth, strict=True):
            horizon_value = int(horizon)
            actual_value = float(actual.item())
            previous = reference_truth_by_horizon.get(horizon_value)
            if previous is not None and abs(actual_value - previous) > 1.0e-6:
                raise ValueError(
                    f"Models disagree on the realised Close at horizon "
                    f"{horizon_value}."
                )
            reference_truth_by_horizon[horizon_value] = actual_value

        axes.plot(horizons, prediction.numpy(), marker="o", label=str(label))
        for horizon, pred, actual in zip(horizons, prediction, truth, strict=True):
            records.append(
                {
                    "Model": str(label),
                    "Split": artifacts.split,
                    "Policy": artifacts.policy,
                    "Date": row["Date"],
                    "Window within date": int(row["Window within date"]),
                    "Time window": row["Time window"],
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "Prediction": float(pred.item()),
                    "Truth": float(actual.item()),
                    "Absolute error": float(abs(pred.item() - actual.item())),
                }
            )

    assert (
        reference_truth_by_horizon
        and reference_last is not None
        and reference_window is not None
    )
    truth_horizons = np.asarray(sorted(reference_truth_by_horizon), dtype=int)
    truth_values = np.asarray(
        [reference_truth_by_horizon[int(value)] for value in truth_horizons],
        dtype=float,
    )
    axes.plot(
        truth_horizons,
        truth_values,
        marker="o",
        linewidth=2.5,
        label="Truth",
    )
    axes.scatter([0], [reference_last], color="black", label="Last observed Close")
    time_label = str(reference_window.iloc[0]["Time window"])
    axes.set_title(
        f"{asset} — {date}, window {window_within_date} — {time_label}"
    )
    axes.set_xlabel("Forecast horizon (minutes)")
    axes.set_ylabel("Raw Close price")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    return PointForecastReport(
        figure=figure,
        axes=axes,
        values=pd.DataFrame(records),
        window=reference_window,
    )


def load_model_sampled_path_bundle(
    run_dir: str | Path,
    *,
    split: EvaluationSplit = "validation",
    policy: str = "auto",
) -> SampledPathBundle:
    """Load saved decoded paths from validation or frozen test inference."""

    artifacts = load_evaluation_artifacts(
        resolve_model_folder(run_dir),
        split=split,
        policy=policy,
        require_sampled_paths=True,
    )
    assert artifacts.sampled_price_paths is not None
    values = artifacts.sampled_price_paths
    required = {
        "sampled_close_paths",
        "sampled_close_paths_at_evaluation_horizons",
        "ensemble_mean_close_path",
        "evaluation_true",
        "last_context_target",
        "asset_cols",
        "future_steps",
        "evaluation_horizons",
        "sample_count",
    }
    missing = required - set(values)
    if missing:
        raise KeyError(f"Sampled-path artefact is missing {sorted(missing)}.")

    sampled = torch.as_tensor(values["sampled_close_paths"]).float()
    sampled_eval = torch.as_tensor(values["sampled_close_paths_at_evaluation_horizons"]).float()
    ensemble = torch.as_tensor(values["ensemble_mean_close_path"]).float()
    true = torch.as_tensor(values["evaluation_true"]).float()
    last = torch.as_tensor(values["last_context_target"]).float()
    if sampled.ndim != 5 or sampled.shape[-1] != 1:
        raise ValueError("sampled_close_paths must have shape [S,W,P,N,1].")
    sample_count, windows, steps, nodes, _ = map(int, sampled.shape)
    horizons = tuple(int(value) for value in values["evaluation_horizons"])
    if sample_count != int(values["sample_count"]):
        raise ValueError("Saved sample count differs from path tensor.")
    if nodes != artifacts.info.num_nodes:
        raise ValueError("Sampled path node count differs from metadata.")
    if tuple(sampled_eval.shape) != (sample_count, windows, len(horizons), nodes, 1):
        raise ValueError("Evaluation sampled-path tensor has inconsistent shape.")
    if tuple(ensemble.shape) != (windows, steps, nodes, 1):
        raise ValueError("Ensemble-mean tensor has inconsistent shape.")
    if tuple(true.shape) != (windows, len(horizons), nodes, 1):
        raise ValueError("Sampled-path truth tensor has inconsistent shape.")
    if tuple(last.shape) != (windows, nodes, 1):
        raise ValueError("Sampled-path last-context tensor has inconsistent shape.")
    if not all(torch.isfinite(tensor).all() for tensor in (sampled, sampled_eval, ensemble, true, last)):
        raise ValueError("Sampled-price-path artefact contains non-finite values.")
    torch.testing.assert_close(ensemble, sampled.mean(dim=0), atol=2.0e-5, rtol=2.0e-5)

    def optional_long(name: str) -> Tensor | None:
        value = values.get(name)
        return None if value is None else torch.as_tensor(value).long()

    raw_dates = values.get("dates") or []
    dates = tuple(_normalise_date(value) for value in raw_dates)
    if dates and len(dates) != windows:
        raise ValueError("Sampled-path dates do not align with windows.")
    return SampledPathBundle(
        run_dir=artifacts.info.run_dir,
        policy=artifacts.policy,
        sampled_close_paths=sampled,
        sampled_close_paths_at_evaluation_horizons=sampled_eval,
        ensemble_mean_close_path=ensemble,
        evaluation_true=true,
        last_context_target=last,
        sample_idx=optional_long("sample_idx"),
        origin_idx=optional_long("origin_idx"),
        dense_target_indices=optional_long("dense_target_indices"),
        evaluation_target_indices=optional_long("evaluation_target_indices"),
        dates=dates,
        asset_cols=tuple(str(value) for value in values["asset_cols"]),
        future_steps=tuple(int(value) for value in values["future_steps"]),
        evaluation_horizons=horizons,
        temperature=(None if values.get("temperature") is None else float(values["temperature"])),
        top_k=int(values.get("top_k", 0)),
        top_p=float(values.get("top_p", 1.0)),
        sample_count=sample_count,
    )


def _window_table_from_sampled_bundle(
    bundle: SampledPathBundle,
    *,
    split: EvaluationSplit,
) -> pd.DataFrame:
    windows = int(bundle.sampled_close_paths.shape[1])
    dates = list(bundle.dates) if bundle.dates else [None] * windows
    origins = (
        [None] * windows
        if bundle.origin_idx is None
        else [int(value) for value in bundle.origin_idx.reshape(-1).tolist()]
    )
    samples = (
        [None] * windows
        if bundle.sample_idx is None
        else [int(value) for value in bundle.sample_idx.reshape(-1).tolist()]
    )
    records = []
    for index in range(windows):
        record = {
            "Global window index": index,
            "Date": None if dates[index] is None else _normalise_date(dates[index]),
            "Sample index": samples[index],
            "Origin index": origins[index],
            "Split": split,
        }
        record.update(_window_time_values(origins[index], 60, len(bundle.future_steps)))
        records.append(record)
    table = pd.DataFrame(records)
    table["Window within date"] = table.groupby("Date", dropna=False).cumcount() + 1
    return table


def analyse_sampled_path_window(
    run_dir: str | Path,
    *,
    split: EvaluationSplit = "validation",
    policy: str = "auto",
    data_dir: str | Path,
    asset: str,
    date: str,
    window_within_date: int,
    context_points: int = 30,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> SamplePathWindowReport:
    """Plot every sampled path, the ensemble mean, and the realised path."""

    bundle = load_model_sampled_path_bundle(run_dir, split=split, policy=policy)
    if asset not in bundle.asset_cols:
        raise ValueError(f"Asset {asset!r} is unavailable.")
    asset_index = bundle.asset_cols.index(asset)
    table = _window_table_from_sampled_bundle(bundle, split=split)
    row = _resolve_evaluation_window(
        table,
        date=date,
        window_within_date=window_within_date,
        global_window_index=None,
    )
    window_index = int(row["Global window index"])

    paths = bundle.sampled_close_paths[:, window_index, :, asset_index, 0]
    mean_path = bundle.ensemble_mean_close_path[window_index, :, asset_index, 0]
    eval_true = bundle.evaluation_true[window_index, :, asset_index, 0]
    last_value = bundle.last_context_target[window_index, asset_index, 0]
    context, true_future, true_date = _load_true_candle_window(
        data_dir=data_dir,
        bundle=bundle,
        window_index=window_index,
        asset_index=asset_index,
        split=split,
    )

    figure, axes = plt.subplots(figsize=figsize)
    future_x = np.arange(1, paths.shape[1] + 1)
    for sample_index in range(paths.shape[0]):
        axes.plot(future_x, paths[sample_index].numpy(), linewidth=0.9, alpha=0.32)
    axes.plot(future_x, mean_path.numpy(), linewidth=2.5, label="Ensemble mean")
    if true_future is not None:
        axes.plot(future_x, true_future.numpy(), linewidth=2.2, label="True future")
    else:
        axes.scatter(bundle.evaluation_horizons, eval_true.numpy(), marker="x", s=55, label="Truth")
    if context is not None and context_points > 0:
        shown = context[-int(context_points):]
        axes.plot(np.arange(-len(shown) + 1, 1), shown.numpy(), linewidth=1.8, label="Observed context")
    else:
        axes.scatter([0], [float(last_value)], label="Last observed Close")
    axes.axvline(0, linestyle="--", linewidth=1.0)
    time_label = row["Time window"] or "clock time unavailable"
    axes.set_title(
        f"{asset} — {true_date or date}, window {window_within_date} — {time_label}\n"
        f"{bundle.policy}; T={bundle.temperature}; {bundle.sample_count} decoded paths"
    )
    axes.set_xlabel("Minutes relative to forecast origin")
    axes.set_ylabel("Raw Close price")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()

    path_table = pd.DataFrame(
        paths.T.numpy(),
        index=pd.Index(future_x, name="Future minute"),
        columns=[f"Path {index + 1}" for index in range(paths.shape[0])],
    )
    path_table["Ensemble mean"] = mean_path.numpy()
    path_table["True future"] = (
        true_future.numpy() if true_future is not None else np.nan
    )
    if true_future is None:
        for horizon, value in zip(bundle.evaluation_horizons, eval_true, strict=True):
            path_table.loc[int(horizon), "True future"] = float(value.item())

    horizon_indices = [int(horizon) - 1 for horizon in bundle.evaluation_horizons]
    horizon_table = path_table.iloc[horizon_indices].copy()
    horizon_table.index = pd.Index(bundle.evaluation_horizons, name="Horizon")
    return SamplePathWindowReport(
        figure=figure,
        axes=axes,
        paths=path_table,
        horizons=horizon_table,
        window=table.loc[[row.name]].copy(),
    )


def _sampled_return_tensors_v10(bundle: SampledPathBundle) -> tuple[Tensor, Tensor]:
    last = bundle.last_context_target.unsqueeze(0).unsqueeze(2)
    sampled_returns = torch.log(bundle.sampled_close_paths_at_evaluation_horizons / last)
    true_returns = torch.log(
        bundle.evaluation_true / bundle.last_context_target.unsqueeze(1)
    )
    return sampled_returns, true_returns


def make_predictive_coverage_report(
    run_dir: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    nominal_coverages: Sequence[float] = (0.5, 0.8, 0.9),
    asset_coverage: float = 0.8,
) -> PredictiveCoverageReport:
    """Compute coverage, interval width, CRPS, and truth-rank diagnostics."""

    resolved_split = normalise_evaluation_split(split)
    bundle = load_model_sampled_path_bundle(run_dir, split=resolved_split, policy=policy)
    sampled, truth = _sampled_return_tensors_v10(bundle)
    overall_rows: list[dict[str, Any]] = []
    interval_specs: list[tuple[str, float | None, float, float]] = []
    for coverage in nominal_coverages:
        value = float(coverage)
        if not 0.0 < value < 1.0:
            raise ValueError("nominal_coverages must lie in (0,1).")
        tail = 0.5 * (1.0 - value)
        interval_specs.append((f"Central {100 * value:g}%", value, tail, 1.0 - tail))
    interval_specs.append(("Sample min-max", None, 0.0, 1.0))

    for label, nominal, lower_q, upper_q in interval_specs:
        lower = torch.quantile(sampled, lower_q, dim=0)
        upper = torch.quantile(sampled, upper_q, dim=0)
        covered = (truth >= lower) & (truth <= upper)
        width_bps = 10000.0 * (upper - lower)
        for index, horizon in enumerate(bundle.evaluation_horizons):
            empirical = float(covered[:, index].float().mean().item())
            overall_rows.append(
                {
                    "Split": resolved_split,
                    "Policy": bundle.policy,
                    "Temperature": bundle.temperature,
                    "Sample count": bundle.sample_count,
                    "Horizon": int(horizon),
                    "Interval": label,
                    "Nominal coverage": nominal,
                    "Empirical coverage": empirical,
                    "Coverage gap": None if nominal is None else empirical - nominal,
                    "Below interval": float((truth[:, index] < lower[:, index]).float().mean().item()),
                    "Above interval": float((truth[:, index] > upper[:, index]).float().mean().item()),
                    "Mean interval width (log-return bps)": float(width_bps[:, index].mean().item()),
                    "Median interval width (log-return bps)": float(width_bps[:, index].median().item()),
                }
            )
    overall = pd.DataFrame(overall_rows)

    coverage = float(asset_coverage)
    tail = 0.5 * (1.0 - coverage)
    lower = torch.quantile(sampled, tail, dim=0)
    upper = torch.quantile(sampled, 1.0 - tail, dim=0)
    covered = (truth >= lower) & (truth <= upper)
    width_bps = 10000.0 * (upper - lower)
    asset_rows: list[dict[str, Any]] = []
    for horizon_index, horizon in enumerate(bundle.evaluation_horizons):
        for asset_index, asset in enumerate(bundle.asset_cols):
            asset_rows.append(
                {
                    "Asset": asset,
                    "Horizon": int(horizon),
                    "Nominal coverage": coverage,
                    "Empirical coverage": float(covered[:, horizon_index, asset_index, 0].float().mean().item()),
                    "Coverage gap": float(covered[:, horizon_index, asset_index, 0].float().mean().item()) - coverage,
                    "Mean interval width (log-return bps)": float(width_bps[:, horizon_index, asset_index, 0].mean().item()),
                    "Median interval width (log-return bps)": float(width_bps[:, horizon_index, asset_index, 0].median().item()),
                }
            )
    by_asset = pd.DataFrame(asset_rows)

    ensemble_mean = sampled.mean(dim=0)
    sample_median = sampled.median(dim=0).values
    term_one = (sampled - truth.unsqueeze(0)).abs().mean(dim=0)
    pairwise = (sampled[:, None] - sampled[None, :]).abs().mean(dim=(0, 1))
    crps = term_one - 0.5 * pairwise
    dispersion = sampled.std(dim=0, unbiased=False)
    score_rows = []
    for index, horizon in enumerate(bundle.evaluation_horizons):
        score_rows.append(
            {
                "Horizon": int(horizon),
                "Empirical CRPS (log-return bps)": float(10000.0 * crps[:, index].mean().item()),
                "Ensemble-mean Log MAE": float((ensemble_mean[:, index] - truth[:, index]).abs().mean().item()),
                "Sample-median Log MAE": float((sample_median[:, index] - truth[:, index]).abs().mean().item()),
                "Mean predictive std (log-return bps)": float(10000.0 * dispersion[:, index].mean().item()),
                "Median predictive std (log-return bps)": float(10000.0 * dispersion[:, index].median().item()),
            }
        )
    probabilistic_scores = pd.DataFrame(score_rows)

    ranks = (sampled < truth.unsqueeze(0)).sum(dim=0)
    rank_rows = []
    for index, horizon in enumerate(bundle.evaluation_horizons):
        flattened = ranks[:, index].reshape(-1)
        total = int(flattened.numel())
        for rank in range(bundle.sample_count + 1):
            count = int((flattened == rank).sum().item())
            rank_rows.append(
                {
                    "Horizon": int(horizon),
                    "Rank": rank,
                    "Count": count,
                    "Frequency": count / max(total, 1),
                    "Ideal frequency": 1.0 / (bundle.sample_count + 1),
                }
            )
    rank_histogram = pd.DataFrame(rank_rows)
    calibration_figure, calibration_axes = plot_coverage_calibration(overall)
    return PredictiveCoverageReport(
        overall=overall,
        by_asset=by_asset,
        probabilistic_scores=probabilistic_scores,
        rank_histogram=rank_histogram,
        calibration_figure=calibration_figure,
        calibration_axes=calibration_axes,
    )



def resolve_candle_data_root(root: str | Path | None = None) -> Path:
    """Resolve the canonical session-data directory on Mac or Colab."""

    if root is not None:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Candle-data directory does not exist: {path}")
        return path
    candidates = (
        Path(
            "/Users/vishalruparelia/Library/CloudStorage/"
            "GoogleDrive-vishal@autonomous-fox.ai/Shared drives/Vishal/"
            "data/cached_datasets/exp-1m-95s-24y/session"
        ),
        Path(
            "/content/drive/Shareddrives/Vishal/data/cached_datasets/"
            "exp-1m-95s-24y/session"
        ),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the canonical candle-data directory. Checked:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
    )


def resolve_token_cache_path(
    *,
    split: Literal["train", "validation", "test"],
    token_root: str | Path | None = None,
) -> Path:
    """Resolve an original 1,024-class origin-aligned token cache."""

    if token_root is None:
        model_roots = (
            Path(
                "/Users/vishalruparelia/Library/CloudStorage/"
                "GoogleDrive-vishal@autonomous-fox.ai/My Drive/"
                "dissertation/final_model/tokens"
            ),
            Path("/content/drive/MyDrive/dissertation/final_model/tokens"),
        )
        root = next((candidate for candidate in model_roots if candidate.is_dir()), None)
        if root is None:
            raise FileNotFoundError(
                "Could not locate final_model/tokens. Checked:\n"
                + "\n".join(f"  - {candidate}" for candidate in model_roots)
            )
    else:
        root = Path(token_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Token-cache directory does not exist: {root}")

    names = {
        "train": ("origin_aligned_train_tokens.pt",),
        "validation": (
            "origin_aligned_val_tokens.pt",
            "origin_aligned_validation_tokens.pt",
        ),
        "test": ("origin_aligned_test_tokens.pt",),
    }
    if split not in names:
        raise ValueError("split must be 'train', 'validation', or 'test'.")
    for name in names[split]:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No original token cache for split {split!r} was found under {root}. "
        f"Expected one of {list(names[split])}."
    )


def make_asset_coverage_ranking(
    coverage_by_asset: pd.DataFrame,
    *,
    horizon: int | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank assets by absolute calibration gap for one/all horizons."""

    required = {
        "Asset",
        "Horizon",
        "Nominal coverage",
        "Empirical coverage",
        "Coverage gap",
        "Mean interval width (log-return bps)",
    }
    missing = required - set(coverage_by_asset.columns)
    if missing:
        raise ValueError(f"coverage_by_asset is missing {sorted(missing)}.")
    selected = coverage_by_asset.copy()
    if horizon is not None:
        selected = selected.loc[selected["Horizon"] == int(horizon)].copy()
    selected["Absolute coverage gap"] = selected["Coverage gap"].abs()
    return (
        selected.sort_values(
            ["Absolute coverage gap", "Asset", "Horizon"],
            ascending=[False, True, True],
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )


def plot_sample_rank_histogram(
    rank_table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10.0, 6.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot truth-rank frequencies for each horizon against the ideal level."""

    required = {"Horizon", "Rank", "Frequency", "Ideal frequency"}
    missing = required - set(rank_table.columns)
    if missing:
        raise ValueError(f"rank_table is missing {sorted(missing)}.")
    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure
    for horizon, group in rank_table.groupby("Horizon"):
        ordered = group.sort_values("Rank")
        axes.plot(
            ordered["Rank"],
            ordered["Frequency"],
            marker="o",
            label=f"{int(horizon)} min",
        )
    ideal = float(rank_table["Ideal frequency"].dropna().iloc[0])
    axes.axhline(ideal, linestyle="--", color="black", linewidth=1.0, label="Ideal")
    axes.set_xlabel("Number of sampled returns below the realised return")
    axes.set_ylabel("Frequency")
    axes.set_title("Finite-ensemble truth-rank histogram")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    return figure, axes

# ============================================================================
# Final Graph Hub API (v11)
# ============================================================================
#
# The functions below implement the final notebook contract.  They are
# deliberately folder-name agnostic and saved-artifact only: nothing here
# trains a model or launches split inference.

DaySelector = str | None
WindowSelector = int | Literal["random"] | None
AssetSelector = str | int | Literal["random"]
TokenDistributionBar = Literal[
    "mean_probability",
    "hard_prediction_frequency",
    "training_target_frequency",
    "sampled_frequency",
]


@dataclass(frozen=True)
class TrainingHistoryReport:
    """Separate predictive, graph-concentration, and beta figures."""

    history: pd.DataFrame
    objective_figure: Figure
    objective_axes: Axes
    entropy_figure: Figure | None
    entropy_axes: Axes | None
    beta_figure: Figure | None
    beta_axes: Axes | None


@dataclass(frozen=True)
class SelectedGraph:
    """One exact or averaged graph selection under ``A[target, source]``."""

    adjacency: pd.DataFrame
    per_window_adjacency: Tensor  # [selected windows, target, source]
    selected_windows: pd.DataFrame
    run_dir: Path
    run_name: str
    run_kind: AnalysisRunKind
    split: EvaluationSplit
    policy: str
    graph_type: str
    component: str
    layer: int
    head: HeadSelection
    requested_day: str | None
    requested_window: int | str | None
    resolved_day: str | None
    resolved_window: int | None
    selection_description: str
    time_window_description: str | None
    displayed_mean_row_entropy: float
    displayed_effective_neighbours: float
    mean_window_row_entropy: float
    mean_window_effective_neighbours: float
    maximum_edge_weight: float
    mean_top5_row_mass: float
    add_self_loops: bool


@dataclass(frozen=True)
class GraphAnalysisReport:
    """Actual adjacency, concentration, connections, and source frequency."""

    graph: SelectedGraph
    summary: pd.DataFrame
    connections: pd.DataFrame
    top_source_frequency: pd.DataFrame
    adjacency_figure: Figure
    adjacency_axes: Axes
    frequency_figure: Figure
    frequency_axes: Axes
    plotted_adjacency: pd.DataFrame


@dataclass(frozen=True)
class GraphEntropyReport:
    """Dynamic-graph row entropy across dates and target assets."""

    day_values: pd.DataFrame
    day_summary: pd.DataFrame
    asset_summary: pd.DataFrame
    window_asset_entropy: pd.DataFrame
    figure: Figure
    axes: Axes


@dataclass(frozen=True)
class SectorGraphReport:
    """Sector-to-sector and asset-to-sector aggregations of one graph view."""

    graph: SelectedGraph
    asset_sector_mapping: pd.DataFrame
    sector_adjacency: pd.DataFrame
    asset_sector_adjacency: pd.DataFrame
    sector_figure: Figure
    sector_axes: Axes
    asset_sector_figure: Figure
    asset_sector_axes: Axes


def list_saved_policies(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    models_root: str | Path | None = None,
) -> pd.DataFrame:
    """List saved policy directories and the manifest-selected policy."""

    run_dir = resolve_model_folder(model, models_root=models_root)
    resolved_split = normalise_evaluation_split(split)
    selected = _selected_temperature_policy(run_dir, resolved_split)
    policies = _available_saved_policies(run_dir, split=resolved_split)
    rows = [
        {
            "Policy": policy,
            "Selected by manifest": policy == selected,
            "Split": resolved_split,
        }
        for policy in policies
    ]
    if not rows:
        default_policy = (
            "best"
            if resolved_split == "validation"
            and (run_dir / "best_validation_predictions.pt").is_file()
            else "default"
        )
        rows.append(
            {
                "Policy": default_policy,
                "Selected by manifest": selected == default_policy,
                "Split": resolved_split,
            }
        )
    return pd.DataFrame(rows)


def analyse_training_history(
    model: str | Path,
    *,
    models_root: str | Path | None = None,
    figsize: tuple[float, float] = (10.5, 4.5),
) -> TrainingHistoryReport:
    """Plot the saved training history with beta on its own figure.

    The predictive objective, graph entropy, and learned spatial gate have
    different scales and interpretations.  They are therefore never overlaid.
    """

    run_dir = resolve_model_folder(model, models_root=models_root)
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"History is empty: {history_path}")

    epoch_column = _first_existing_column(history, ("epoch", "Epoch"))
    if epoch_column is None:
        history = history.copy()
        history["epoch"] = np.arange(1, len(history) + 1)
        epoch_column = "epoch"

    train_column = _first_existing_column(
        history,
        (
            "train_token_loss",
            "training_loss",
            "train_loss",
            "training_native_loss",
            "training_native_log_mae",
            "training_objective_loss",
        ),
    )
    validation_column = _first_existing_column(
        history,
        ("validation_token_loss", "selection_score", "validation_loss", "val_loss"),
    )
    entropy_column = _first_existing_column(
        history,
        (
            "validation_graph_mean_row_entropy",
            "graph_mean_row_entropy",
            "test_graph_mean_row_entropy",
            "training_graph_mean_row_entropy",
        ),
    )
    beta_column = _first_existing_column(
        history,
        ("spatial_beta", "validation_spatial_beta"),
    )

    objective_figure, objective_axes = plt.subplots(figsize=figsize)
    if train_column is not None:
        objective_axes.plot(history[epoch_column], history[train_column], label=train_column)
    if validation_column is not None:
        objective_axes.plot(
            history[epoch_column], history[validation_column], label=validation_column
        )
    objective_axes.set_title(f"{run_dir.name} — predictive training objective")
    objective_axes.set_xlabel("Epoch")
    objective_axes.set_ylabel("Saved objective value")
    objective_axes.grid(True, alpha=0.25)
    if objective_axes.lines:
        objective_axes.legend()
    objective_figure.tight_layout()

    entropy_figure: Figure | None = None
    entropy_axes: Axes | None = None
    if entropy_column is not None:
        entropy_figure, entropy_axes = plt.subplots(figsize=figsize)
        entropy_axes.plot(
            history[epoch_column], history[entropy_column], color="tab:purple"
        )
        entropy_axes.set_title(f"{run_dir.name} — graph mean row entropy")
        entropy_axes.set_xlabel("Epoch")
        entropy_axes.set_ylabel("Mean row entropy (nats)")
        entropy_axes.grid(True, alpha=0.25)
        entropy_figure.tight_layout()

    beta_figure: Figure | None = None
    beta_axes: Axes | None = None
    if beta_column is not None:
        beta_figure, beta_axes = plt.subplots(figsize=figsize)
        beta_axes.plot(history[epoch_column], history[beta_column], color="tab:green")
        beta_axes.set_title(f"{run_dir.name} — learned spatial beta")
        beta_axes.set_xlabel("Epoch")
        beta_axes.set_ylabel("Beta: graph-aware feature mixture weight")
        beta_axes.set_ylim(0.0, 1.0)
        beta_axes.grid(True, alpha=0.25)
        beta_figure.tight_layout()

    selected_columns = [epoch_column]
    for column in (train_column, validation_column, entropy_column, beta_column):
        if column is not None and column not in selected_columns:
            selected_columns.append(column)
    return TrainingHistoryReport(
        history=history[selected_columns].copy(),
        objective_figure=objective_figure,
        objective_axes=objective_axes,
        entropy_figure=entropy_figure,
        entropy_axes=entropy_axes,
        beta_figure=beta_figure,
        beta_axes=beta_axes,
    )


def _graph_component_tensor(
    artifacts: EvaluationArtifacts,
    *,
    component: GraphComponent,
    layer: int,
) -> Tensor:
    if artifacts.graph_artifacts is None:
        raise FileNotFoundError(
            f"No {artifacts.split} graph artefact exists for {artifacts.info.run_dir}."
        )
    graph = artifacts.graph_artifacts
    if int(layer) != -1:
        per_layer_key = {
            "selected": "per_layer",
            "base": "per_layer_base",
            "dynamic": "per_layer_dynamic",
        }[component]
        per_layer = graph.get(per_layer_key)
        if not isinstance(per_layer, Sequence) or isinstance(per_layer, (str, bytes)):
            raise ValueError(
                f"Saved graph artefact does not contain {per_layer_key} graphs."
            )
        layer_index = _normalise_layer_index(int(layer), len(per_layer))
        values = per_layer[layer_index]
    else:
        values = graph.get(component)
    if values is None:
        raise ValueError(f"Graph component {component!r} is unavailable.")
    tensor = torch.as_tensor(values).detach().cpu().double()
    if tensor.ndim != 4:
        raise ValueError("Normalised graph tensor must have shape [W,G,N,N].")
    return tensor


def _normalise_day_selector(day: str | None) -> str | None:
    if day is None:
        return None
    text = str(day).strip()
    if text.lower() == "random":
        return "random"
    return _normalise_date(text)


def _normalise_window_selector(window: int | str | None) -> int | str | None:
    if window is None:
        return None
    if isinstance(window, str):
        if window.strip().lower() == "random":
            return "random"
        try:
            window = int(window)
        except ValueError as error:
            raise ValueError("window must be a positive integer, None, or 'random'.") from error
    value = int(window)
    if value <= 0:
        raise ValueError("window is one-based and must be positive.")
    return value


def _select_day_window_rows(
    table: pd.DataFrame,
    *,
    day: str | None,
    window: int | str | None,
    random_seed: int,
) -> tuple[pd.DataFrame, str | None, int | None, str]:
    """Apply the final Graph Hub day/window aggregation convention."""

    if table.empty:
        raise ValueError("The saved split contains no windows.")
    required = {"Date", "Window within date", "Global window index"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Window table is missing {sorted(missing)}.")

    requested_day = _normalise_day_selector(day)
    requested_window = _normalise_window_selector(window)
    rng = np.random.default_rng(int(random_seed))

    selected_day: str | None = None
    if requested_day == "random":
        available_days = sorted(str(value) for value in table["Date"].dropna().unique())
        if not available_days:
            raise ValueError("Random day selection requires saved date metadata.")
        selected_day = str(rng.choice(available_days))
    elif requested_day is not None:
        selected_day = requested_day

    working = table.copy()
    if selected_day is not None:
        working = working.loc[working["Date"].astype(str).eq(selected_day)].copy()
        if working.empty:
            available = sorted(str(value) for value in table["Date"].dropna().unique())
            raise ValueError(
                f"Day {selected_day!r} is unavailable. Available days include {available[:10]}."
            )

    selected_window: int | None = None
    if requested_window == "random":
        available_windows = sorted(
            int(value) for value in working["Window within date"].dropna().unique()
        )
        if not available_windows:
            raise ValueError("Random window selection found no candidate windows.")
        selected_window = int(rng.choice(available_windows))
    elif requested_window is not None:
        selected_window = int(requested_window)

    if selected_window is not None:
        working = working.loc[
            pd.to_numeric(working["Window within date"], errors="coerce")
            == selected_window
        ].copy()
        if working.empty:
            raise ValueError(
                f"Window {selected_window} is unavailable under the selected day filter."
            )

    working = working.sort_values("Global window index").reset_index(drop=True)
    if selected_day is not None and selected_window is not None:
        description = f"{selected_day}, window {selected_window}"
    elif selected_day is not None:
        description = f"{selected_day}, averaged over all {len(working)} saved windows"
    elif selected_window is not None:
        day_count = int(working["Date"].nunique(dropna=True))
        description = (
            f"window {selected_window}, averaged across {day_count} days "
            f"({len(working)} saved windows)"
        )
    else:
        day_count = int(working["Date"].nunique(dropna=True))
        description = (
            f"all {len(working)} saved windows across {day_count} days"
        )
    return working, selected_day, selected_window, description


def select_graph(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    day: str | None = None,
    window: int | str | None = None,
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    random_seed: int = 42,
    models_root: str | Path | None = None,
) -> SelectedGraph:
    """Select or average graphs using the final day/window convention.

    ``day=None`` and ``window=None`` average the full split.  Supplying only a
    day averages all windows on that day.  Supplying only a one-based window
    number averages that intraday window across all days.  Use the explicit
    string ``"random"`` for reproducible random day/window selection; ``None``
    is never interpreted as random.
    """

    run_dir = resolve_model_folder(model, models_root=models_root)
    resolved_split = normalise_evaluation_split(split)
    artifacts = load_evaluation_artifacts(
        run_dir,
        split=resolved_split,
        policy=policy,
        require_graph=True,
    )
    table = make_evaluation_window_table(
        run_dir,
        split=resolved_split,
        policy=artifacts.policy,
    )
    selected_rows, selected_day, selected_window, description = _select_day_window_rows(
        table,
        day=day,
        window=window,
        random_seed=random_seed,
    )
    indices = torch.as_tensor(
        selected_rows["Global window index"].to_numpy(dtype=np.int64),
        dtype=torch.long,
    )
    tensor = _graph_component_tensor(
        artifacts,
        component=component,
        layer=layer,
    ).index_select(0, indices)

    if head == "mean":
        per_window = tensor.mean(dim=1)
    else:
        head_index = int(head)
        if not 0 <= head_index < int(tensor.shape[1]):
            raise IndexError(
                f"Graph head {head_index} is out of range for {int(tensor.shape[1])} heads."
            )
        per_window = tensor[:, head_index]

    displayed = per_window.mean(dim=0)
    displayed_entropy = -(
        displayed.clamp_min(1.0e-12) * displayed.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    per_window_entropy = -(
        per_window.clamp_min(1.0e-12) * per_window.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    available_neighbours = displayed.shape[-1] - 1
    top_k = max(1, min(5, available_neighbours))
    top5_mass = displayed.topk(top_k, dim=-1).values.sum(dim=-1)

    unique_time_windows = tuple(
        str(value)
        for value in selected_rows["Time window"].dropna().unique().tolist()
    )
    time_description = unique_time_windows[0] if len(unique_time_windows) == 1 else None
    frame = pd.DataFrame(
        displayed.numpy(),
        index=artifacts.info.asset_cols,
        columns=artifacts.info.asset_cols,
    )
    frame.index.name = "Target"
    frame.columns.name = "Source"
    return SelectedGraph(
        adjacency=frame,
        per_window_adjacency=per_window.contiguous(),
        selected_windows=selected_rows,
        run_dir=run_dir,
        run_name=run_dir.name,
        run_kind=artifacts.info.run_kind,
        split=resolved_split,
        policy=artifacts.policy,
        graph_type=artifacts.info.graph_type,
        component=component,
        layer=int(layer),
        head=head,
        requested_day=day,
        requested_window=window,
        resolved_day=selected_day,
        resolved_window=selected_window,
        selection_description=description,
        time_window_description=time_description,
        displayed_mean_row_entropy=float(displayed_entropy.mean().item()),
        displayed_effective_neighbours=float(displayed_entropy.exp().mean().item()),
        mean_window_row_entropy=float(per_window_entropy.mean().item()),
        mean_window_effective_neighbours=float(per_window_entropy.exp().mean().item()),
        maximum_edge_weight=float(displayed.max().item()),
        mean_top5_row_mass=float(top5_mass.mean().item()),
        add_self_loops=artifacts.info.add_self_loops,
    )


def make_selected_graph_summary(graph: SelectedGraph) -> pd.DataFrame:
    """Summarise the displayed mean graph and its window-level concentration."""

    return pd.DataFrame(
        [
            {
                "Run": graph.run_name,
                "Family": graph.run_kind,
                "Split": graph.split,
                "Policy": graph.policy,
                "Graph type": graph.graph_type,
                "Component": graph.component,
                "Layer": graph.layer,
                "Head": graph.head,
                "Selection": graph.selection_description,
                "Clock-time window": graph.time_window_description,
                "Selected windows": len(graph.selected_windows),
                "Entropy of displayed mean adjacency": graph.displayed_mean_row_entropy,
                "Effective neighbours of displayed mean adjacency": graph.displayed_effective_neighbours,
                "Mean window-specific row entropy": graph.mean_window_row_entropy,
                "Mean window-specific effective neighbours": graph.mean_window_effective_neighbours,
                "Maximum displayed edge weight": graph.maximum_edge_weight,
                "Mean displayed top-5 row mass": graph.mean_top5_row_mass,
            }
        ]
    )


def make_graph_connections(
    graph: SelectedGraph,
    *,
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
) -> pd.DataFrame:
    """Return strongest incoming or outgoing edges for every asset."""

    return _top_neighbours_from_adjacency(
        graph.adjacency.to_numpy(dtype=np.float64),
        tuple(str(value) for value in graph.adjacency.index),
        top_n=int(top_n),
        direction=direction,
    )


def make_top_source_frequency(
    graph: SelectedGraph,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Count how often each source appears in target rows' top-N inputs.

    Self edges are always excluded.  With 93 assets the largest possible count
    is therefore 92, meaning the source is among the top-N influences for every
    *other* target asset.
    """

    matrix = graph.adjacency.to_numpy(dtype=np.float64)
    labels = tuple(str(value) for value in graph.adjacency.index)
    nodes = len(labels)
    requested_count = int(top_n)
    if requested_count <= 0:
        raise ValueError("top_n must be positive.")
    count = min(requested_count, nodes - 1)
    source_counts = np.zeros(nodes, dtype=np.int64)
    for target in range(nodes):
        scores = matrix[target].copy()
        scores[target] = -np.inf
        positive = np.flatnonzero(scores > 0.0)
        ranked = positive[np.argsort(scores[positive])[::-1]][:count]
        source_counts[ranked] += 1
    table = pd.DataFrame(
        {
            "Source asset": labels,
            f"Targets for which source is in top {count}": source_counts,
            "Eligible target assets": nodes - 1,
            "Proportion of eligible targets": source_counts / max(nodes - 1, 1),
        }
    )
    return table.sort_values(
        [f"Targets for which source is in top {count}", "Source asset"],
        ascending=[False, True],
    ).reset_index(drop=True)


def plot_selected_graph(
    graph: SelectedGraph,
    *,
    cluster: bool = False,
    cluster_method: str = "average",
    company_profiles_path: str | Path | None = None,
    heatmap_cap_top_k: int = 0,
    figsize: tuple[float, float] = (13.0, 11.0),
    tick_fontsize: float = 8.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot actual adjacency weights, optionally grouped by company sector.

    ``cluster=False`` preserves the saved model asset order. ``cluster=True``
    no longer performs unsupervised hierarchical clustering: it reads column 1
    (ticker) and column 6 (sector) from ``company_profiles.csv``, orders sectors
    alphabetically, and orders tickers alphabetically within each sector. This
    fixed economic ordering makes graphs directly comparable across windows.

    ``cluster_method`` is retained only for backward call compatibility and is
    ignored when sector grouping is requested.

    ``heatmap_cap_top_k`` is a display-only colour-scale control.  The default
    value ``0`` preserves the ordinary maximum-edge colour scale.  A positive
    value ignores that many largest finite off-diagonal entries when choosing
    ``vmax``; those entries are visually saturated at the next-highest value.
    The adjacency returned by this function, and every metric/table calculated
    from it, always retain the original uncapped weights.
    """

    del cluster_method
    matrix = graph.adjacency.to_numpy(dtype=np.float64, copy=True)
    labels = np.asarray(graph.adjacency.index, dtype=object)
    ordered_sectors: np.ndarray | None = None
    if cluster:
        order, ordered_mapping = make_sector_group_order(
            tuple(str(value) for value in labels),
            company_profiles_path=company_profiles_path,
        )
        matrix = matrix[np.ix_(order, order)]
        labels = labels[order]
        ordered_sectors = (
            ordered_mapping["Sector"].astype(str).to_numpy()
        )
    plotted_values = matrix.copy()
    if not graph.add_self_loops:
        np.fill_diagonal(plotted_values, np.nan)
    finite = plotted_values[np.isfinite(plotted_values)]
    uncapped_maximum = float(np.max(finite)) if finite.size else 1.0
    cap_count = max(int(heatmap_cap_top_k), 0)
    applied_cap_count = 0
    maximum = uncapped_maximum
    if cap_count > 0 and finite.size > cap_count:
        # The (k+1)-th largest finite displayed edge becomes the colourbar
        # maximum. Matplotlib saturates all larger values at that colour.
        candidate = float(np.partition(finite, -(cap_count + 1))[-(cap_count + 1)])
        if np.isfinite(candidate) and candidate > 0.0:
            maximum = candidate
            applied_cap_count = cap_count
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("white")
    image = axes.imshow(
        plotted_values,
        cmap=cmap,
        vmin=0.0,
        vmax=maximum,
        interpolation="nearest",
        aspect="equal",
    )
    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    axes.set_yticklabels(labels, fontsize=tick_fontsize)
    axes.set_xlabel("Source asset (influences target)")
    axes.set_ylabel("Target asset (receives influence)")
    if ordered_sectors is not None and len(ordered_sectors):
        boundaries = (
            np.flatnonzero(ordered_sectors[1:] != ordered_sectors[:-1]) + 1
        )
        for boundary in boundaries:
            coordinate = float(boundary) - 0.5
            axes.axhline(coordinate, color="black", linewidth=0.8, alpha=0.65)
            axes.axvline(coordinate, color="black", linewidth=0.8, alpha=0.65)
    time_text = (
        f" — {graph.time_window_description}" if graph.time_window_description else ""
    )
    cap_text = (
        "\n"
        f"display-only colour cap: top {applied_cap_count} edge(s) saturated "
        f"at {maximum:.6g}; true maximum={uncapped_maximum:.6g}"
        if applied_cap_count > 0
        else ""
    )
    axes.set_title(
        f"{graph.run_name} — {graph.split} / {graph.policy}\n"
        f"{graph.selection_description}{time_text}"
        f"{' — sector-grouped' if cluster else ''}\n"
        f"entropy of displayed adjacency={graph.displayed_mean_row_entropy:.4f}; "
        f"mean window entropy={graph.mean_window_row_entropy:.4f}"
        f"{cap_text}"
    )
    colourbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    colourbar.set_label(
        "Adjacency weight"
        + (" (display capped)" if applied_cap_count > 0 else "")
    )
    figure.tight_layout()
    plotted = pd.DataFrame(matrix, index=labels, columns=labels)
    plotted.index.name = "Target"
    plotted.columns.name = "Source"
    return figure, axes, plotted


def plot_top_source_frequency(
    table: pd.DataFrame,
    *,
    top_n: int,
    figsize: tuple[float, float] = (16.0, 5.5),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot source-asset counts from ``make_top_source_frequency``."""

    count_candidates = [
        column
        for column in table.columns
        if str(column).startswith("Targets for which source is in top ")
    ]
    if len(count_candidates) != 1:
        raise ValueError("Could not identify the top-source count column.")
    count_column = count_candidates[0]
    actual_top_n = int(str(count_column).rsplit(" ", 1)[-1])
    required = {"Source asset", count_column}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Top-source table is missing {sorted(missing)}.")
    ordered = table.sort_values(count_column, ascending=False).reset_index(drop=True)
    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure
    axes.bar(ordered["Source asset"], ordered[count_column], color="tab:blue")
    axes.set_xlabel("Source asset")
    axes.set_ylabel(f"Number of other targets where source is in top {actual_top_n}")
    axes.set_title(f"Frequency of appearance in target rows' top {actual_top_n} influences")
    axes.tick_params(axis="x", rotation=90, labelsize=8)
    axes.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    return figure, axes


def analyse_graph(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    day: str | None = None,
    window: int | str | None = None,
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    top_n: int = 5,
    direction: NeighbourDirection = "impacted_by",
    cluster: bool = False,
    company_profiles_path: str | Path | None = None,
    heatmap_cap_top_k: int = 0,
    random_seed: int = 42,
    models_root: str | Path | None = None,
) -> GraphAnalysisReport:
    """Run the complete actual-adjacency analysis for one graph selection.

    ``heatmap_cap_top_k`` affects only the heatmap colour scale.  For example,
    ``heatmap_cap_top_k=2`` saturates the two largest off-diagonal cells at the
    third-largest value while preserving the original graph for every metric,
    connection table, returned tensor and saved artefact.
    """

    graph = select_graph(
        model,
        split=split,
        policy=policy,
        day=day,
        window=window,
        component=component,
        layer=layer,
        head=head,
        random_seed=random_seed,
        models_root=models_root,
    )
    summary = make_selected_graph_summary(graph)
    connections = make_graph_connections(graph, top_n=top_n, direction=direction)
    frequency = make_top_source_frequency(graph, top_n=top_n)
    adjacency_figure, adjacency_axes, plotted = plot_selected_graph(
        graph,
        cluster=cluster,
        company_profiles_path=company_profiles_path,
        heatmap_cap_top_k=heatmap_cap_top_k,
    )
    frequency_figure, frequency_axes = plot_top_source_frequency(
        frequency,
        top_n=top_n,
    )
    return GraphAnalysisReport(
        graph=graph,
        summary=summary,
        connections=connections,
        top_source_frequency=frequency,
        adjacency_figure=adjacency_figure,
        adjacency_axes=adjacency_axes,
        frequency_figure=frequency_figure,
        frequency_axes=frequency_axes,
        plotted_adjacency=plotted,
    )


def analyse_graph_entropy(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    day: str | None = None,
    window: int | str | None = None,
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    random_seed: int = 42,
    models_root: str | Path | None = None,
    figsize: tuple[float, float] = (14.0, 5.5),
) -> GraphEntropyReport:
    """Summarise row-entropy variation across dates and target assets."""

    graph = select_graph(
        model,
        split=split,
        policy=policy,
        day=day,
        window=window,
        component=component,
        layer=layer,
        head=head,
        random_seed=random_seed,
        models_root=models_root,
    )
    entropy = -(
        graph.per_window_adjacency.clamp_min(1.0e-12)
        * graph.per_window_adjacency.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    records: list[dict[str, Any]] = []
    assets = tuple(str(value) for value in graph.adjacency.index)
    for local_index, row in graph.selected_windows.reset_index(drop=True).iterrows():
        for asset_index, asset in enumerate(assets):
            records.append(
                {
                    "Date": row["Date"],
                    "Window within date": int(row["Window within date"]),
                    "Global window index": int(row["Global window index"]),
                    "Asset": asset,
                    "Row entropy": float(entropy[local_index, asset_index].item()),
                }
            )
    detailed = pd.DataFrame(records)
    if detailed["Date"].isna().any():
        raise ValueError(
            "Daily entropy analysis requires saved date metadata for every selected window."
        )
    day_values = (
        detailed.groupby("Date", as_index=False)
        .agg(
            **{
                "Windows": ("Global window index", "nunique"),
                "Daily mean row entropy": ("Row entropy", "mean"),
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    daily_series = day_values["Daily mean row entropy"]
    minimum_row = day_values.loc[daily_series.idxmin()]
    maximum_row = day_values.loc[daily_series.idxmax()]
    day_summary = pd.DataFrame(
        [
            {
                "Selected split": graph.split,
                "Selection": graph.selection_description,
                "Days": len(day_values),
                "Mean of daily average row entropy": float(daily_series.mean()),
                "Lower quartile of daily average row entropy": float(daily_series.quantile(0.25)),
                "Upper quartile of daily average row entropy": float(daily_series.quantile(0.75)),
                "Lowest-entropy day": str(minimum_row["Date"]),
                "Lowest daily average row entropy": float(minimum_row["Daily mean row entropy"]),
                "Highest-entropy day": str(maximum_row["Date"]),
                "Highest daily average row entropy": float(maximum_row["Daily mean row entropy"]),
            }
        ]
    )
    asset_summary = (
        detailed.groupby("Asset", as_index=False)["Row entropy"]
        .agg(
            **{
                "Mean row entropy": "mean",
                "Lower quartile row entropy": lambda values: values.quantile(0.25),
                "Upper quartile row entropy": lambda values: values.quantile(0.75),
            }
        )
        .sort_values("Mean row entropy")
        .reset_index(drop=True)
    )

    figure, axes = plt.subplots(figsize=figsize)
    dates = pd.to_datetime(day_values["Date"])
    positions = np.arange(len(day_values))
    axes.plot(positions, day_values["Daily mean row entropy"], linewidth=1.7)
    month_keys = dates.dt.to_period("M")
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    seen: set[str] = set()
    unique_month_periods = pd.Series(month_keys.drop_duplicates().tolist())
    repeated_month_names = unique_month_periods.astype(str).str[5:7].duplicated(keep=False).any()
    for position, (date_value, month_key) in enumerate(zip(dates, month_keys, strict=True)):
        key = str(month_key)
        if key in seen:
            continue
        seen.add(key)
        tick_positions.append(position)
        tick_labels.append(
            date_value.strftime("%b %Y" if repeated_month_names else "%b")
        )
    axes.set_xticks(tick_positions)
    axes.set_xticklabels(tick_labels)
    axes.set_xlabel("Date (label shown at first saved day of each month)")
    axes.set_ylabel("Daily average row entropy (nats)")
    axes.set_title(
        f"{graph.run_name} — daily graph entropy — {graph.split} / {graph.policy}\n"
        f"{graph.selection_description}"
    )
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    return GraphEntropyReport(
        day_values=day_values,
        day_summary=day_summary,
        asset_summary=asset_summary,
        window_asset_entropy=detailed,
        figure=figure,
        axes=axes,
    )




def analyse_sector_graph(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    day: str | None = None,
    window: int | str | None = None,
    component: GraphComponent = "selected",
    layer: int = -1,
    head: HeadSelection = "mean",
    company_profiles_path: str | Path | None = None,
    random_seed: int = 42,
    models_root: str | Path | None = None,
    sector_figsize: tuple[float, float] = (10.0, 8.0),
    asset_figsize: tuple[float, float] = (13.0, 18.0),
) -> SectorGraphReport:
    """Aggregate an asset graph into sector and asset-to-sector views."""

    graph = select_graph(
        model,
        split=split,
        policy=policy,
        day=day,
        window=window,
        component=component,
        layer=layer,
        head=head,
        random_seed=random_seed,
        models_root=models_root,
    )
    mapping = make_asset_sector_mapping(
        tuple(str(value) for value in graph.adjacency.index),
        company_profiles_path=company_profiles_path,
    )
    sectors = tuple(sorted(mapping["Sector"].unique()))
    sector_index = {sector: index for index, sector in enumerate(sectors)}
    asset_sector = np.zeros((len(mapping), len(sectors)), dtype=np.float64)
    for asset_index, sector in enumerate(mapping["Sector"]):
        asset_sector[asset_index, sector_index[str(sector)]] = 1.0

    adjacency = graph.adjacency.to_numpy(dtype=np.float64)
    target_indicator = asset_sector
    sector_values = target_indicator.T @ adjacency @ asset_sector
    row_sums = sector_values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Sector aggregation produced an empty target-sector row.")
    sector_values = sector_values / row_sums
    asset_sector_values = adjacency @ asset_sector
    asset_row_sums = asset_sector_values.sum(axis=1, keepdims=True)
    if not np.allclose(asset_row_sums, 1.0, atol=2.0e-5, rtol=0.0):
        raise ValueError("Asset-to-sector rows do not sum to one.")

    sector_frame = pd.DataFrame(sector_values, index=sectors, columns=sectors)
    sector_frame.index.name = "Target sector"
    sector_frame.columns.name = "Source sector"

    ordering = mapping.sort_values(["Sector", "Ticker"]).index.to_numpy(dtype=int)
    ordered_mapping = mapping.iloc[ordering].reset_index(drop=True)
    ordered_values = asset_sector_values[ordering]
    asset_sector_frame = pd.DataFrame(
        ordered_values,
        index=ordered_mapping["Ticker"],
        columns=sectors,
    )
    asset_sector_frame.index.name = "Target asset"
    asset_sector_frame.columns.name = "Source sector"

    sector_figure, sector_axes = plt.subplots(figsize=sector_figsize)
    sector_image = sector_axes.imshow(
        sector_values,
        cmap="Reds",
        vmin=0.0,
        vmax=max(float(sector_values.max()), 1.0e-12),
        aspect="equal",
    )
    sector_axes.set_xticks(np.arange(len(sectors)))
    sector_axes.set_yticks(np.arange(len(sectors)))
    sector_axes.set_xticklabels(sectors, rotation=90)
    sector_axes.set_yticklabels(sectors)
    sector_axes.set_xlabel("Source sector (influences target sector)")
    sector_axes.set_ylabel("Target sector (receives influence)")
    sector_axes.set_title(
        f"Sector graph — {graph.run_name}\n{graph.selection_description}"
    )
    sector_figure.colorbar(sector_image, ax=sector_axes, fraction=0.046, pad=0.03).set_label(
        "Row-normalised sector influence weight"
    )
    sector_figure.tight_layout()

    asset_sector_figure = plt.figure(figsize=asset_figsize)
    grid = GridSpec(1, 2, width_ratios=[2.8, 10.0], wspace=0.02, figure=asset_sector_figure)
    sector_label_axes = asset_sector_figure.add_subplot(grid[0, 0])
    asset_sector_axes = asset_sector_figure.add_subplot(grid[0, 1])
    asset_image = asset_sector_axes.imshow(
        ordered_values,
        cmap="Reds",
        vmin=0.0,
        vmax=max(float(ordered_values.max()), 1.0e-12),
        aspect="auto",
    )
    asset_sector_axes.set_xticks(np.arange(len(sectors)))
    asset_sector_axes.set_xticklabels(sectors, rotation=90)
    asset_sector_axes.set_yticks(np.arange(len(ordered_mapping)))
    asset_sector_axes.set_yticklabels(ordered_mapping["Ticker"], fontsize=7)
    asset_sector_axes.set_xlabel("Source sector")
    asset_sector_axes.set_ylabel("Target asset")
    asset_sector_axes.set_title(
        f"Asset-to-sector influence — {graph.run_name}\n{graph.selection_description}"
    )

    colour_map = plt.get_cmap("tab20")
    sector_colours = {
        sector: colour_map(index / max(len(sectors) - 1, 1))
        for index, sector in enumerate(sectors)
    }
    start = 0
    for sector, group in ordered_mapping.groupby("Sector", sort=False):
        length = len(group)
        sector_label_axes.add_patch(
            Rectangle(
                (0.0, start - 0.5),
                1.0,
                length,
                facecolor=sector_colours[str(sector)],
                alpha=0.30,
                edgecolor="none",
            )
        )
        sector_label_axes.text(
            0.95,
            start + (length - 1) / 2,
            str(sector),
            ha="right",
            va="center",
            fontsize=9,
            wrap=True,
        )
        if start > 0:
            asset_sector_axes.axhline(start - 0.5, color="black", linewidth=0.7)
        start += length
    sector_label_axes.set_xlim(0.0, 1.0)
    sector_label_axes.set_ylim(len(ordered_mapping) - 0.5, -0.5)
    sector_label_axes.axis("off")
    asset_sector_figure.colorbar(
        asset_image,
        ax=asset_sector_axes,
        fraction=0.026,
        pad=0.02,
    ).set_label("Influence weight assigned to source sector")
    asset_sector_figure.subplots_adjust(left=0.08, right=0.93, bottom=0.16, top=0.93)

    return SectorGraphReport(
        graph=graph,
        asset_sector_mapping=ordered_mapping,
        sector_adjacency=sector_frame,
        asset_sector_adjacency=asset_sector_frame,
        sector_figure=sector_figure,
        sector_axes=sector_axes,
        asset_sector_figure=asset_sector_figure,
        asset_sector_axes=asset_sector_axes,
    )


def _resolve_random_asset(
    asset: AssetSelector,
    assets: Sequence[str],
    *,
    random_seed: int,
) -> str:
    if isinstance(asset, str) and asset.strip().lower() == "random":
        return str(np.random.default_rng(int(random_seed)).choice(tuple(assets)))
    if isinstance(asset, int) and not isinstance(asset, bool):
        index = int(asset)
        if not 0 <= index < len(assets):
            raise IndexError(f"Asset index {index} is out of range.")
        return str(assets[index])
    value = str(asset)
    if value not in assets:
        raise ValueError(f"Asset {value!r} is unavailable.")
    return value


def _resolve_exact_example_window(
    table: pd.DataFrame,
    *,
    day: str,
    window: int | str,
    random_seed: int,
) -> pd.Series:
    selected, _, _, _ = _select_day_window_rows(
        table,
        day=day,
        window=window,
        random_seed=random_seed,
    )
    if len(selected) != 1:
        raise ValueError(
            "Point/path examples require one exact saved window. Pass an exact "
            "day and one-based window, or use the explicit 'random' sentinel."
        )
    return selected.iloc[0]


def plot_point_forecast_example(
    models: Mapping[str, str | Path],
    *,
    split: EvaluationSplitInput | str = "validation",
    policies: Mapping[str, str | None] | None = None,
    asset: AssetSelector = "random",
    day: str = "random",
    window: int | str = "random",
    random_seed: int = 42,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> PointForecastReport:
    """Plot one exact or reproducibly random point-forecast example."""

    if not models:
        raise ValueError("At least one model is required.")
    resolved_split = normalise_evaluation_split(split)
    first_label, first_model = next(iter(models.items()))
    first_policy = None if policies is None else policies.get(first_label)
    first_dir = resolve_model_folder(first_model)
    first_artifacts = load_evaluation_artifacts(
        first_dir,
        split=resolved_split,
        policy=first_policy,
    )
    table = make_evaluation_window_table(
        first_dir,
        split=resolved_split,
        policy=first_artifacts.policy,
    )
    selected_row = _resolve_exact_example_window(
        table,
        day=day,
        window=window,
        random_seed=random_seed,
    )
    resolved_asset = _resolve_random_asset(
        asset,
        first_artifacts.info.asset_cols,
        random_seed=random_seed,
    )
    resolved_policies = None if policies is None else {
        label: value for label, value in policies.items() if value is not None
    }
    return plot_point_forecast_comparison(
        models,
        split=resolved_split,
        policies=resolved_policies,
        asset=resolved_asset,
        date=str(selected_row["Date"]),
        window_within_date=int(selected_row["Window within date"]),
        figsize=figsize,
    )


def analyse_sampled_path_example(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    data_dir: str | Path,
    asset: AssetSelector = "random",
    day: str = "random",
    window: int | str = "random",
    random_seed: int = 42,
    context_points: int = 30,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> SamplePathWindowReport:
    """Plot one exact or reproducibly random ten-path forecast example."""

    resolved_split = normalise_evaluation_split(split)
    bundle = load_model_sampled_path_bundle(
        resolve_model_folder(model),
        split=resolved_split,
        policy=policy,
    )
    table = _window_table_from_sampled_bundle(bundle, split=resolved_split)
    selected_row = _resolve_exact_example_window(
        table,
        day=day,
        window=window,
        random_seed=random_seed,
    )
    resolved_asset = _resolve_random_asset(
        asset,
        bundle.asset_cols,
        random_seed=random_seed,
    )
    return analyse_sampled_path_window(
        resolve_model_folder(model),
        split=resolved_split,
        policy=bundle.policy,
        data_dir=data_dir,
        asset=resolved_asset,
        date=str(selected_row["Date"]),
        window_within_date=int(selected_row["Window within date"]),
        context_points=context_points,
        figsize=figsize,
    )



def _distribution_entropy_from_probabilities(values: Tensor) -> float:
    values = torch.as_tensor(values, dtype=torch.float64)
    positive = values > 0
    if not bool(positive.any()):
        return 0.0
    selected = values[positive]
    return float(-(selected * selected.log()).sum().item())


def _analyse_saved_token_predictive_distribution(
    run_dir: Path,
    *,
    split: EvaluationSplitInput | str,
    policy: str | None,
    asset: str | int | None,
    horizon: int,
    window_indices: Sequence[int] | None,
    max_windows: int | None,
    plot_top_n: int,
    figsize: tuple[float, float],
) -> dict[str, Any] | None:
    """Build a compact predictive-distribution report from saved top-10 data.

    The exact full 1,024-way probability tensor is intentionally not saved by
    the new runners.  The returned mean-probability series is therefore the
    exact average contribution of tokens while they are inside the saved top-10
    set; omitted tail probability is left unassigned and its captured mass is
    reported explicitly.  Hard and training frequencies remain exact.
    """

    try:
        evaluation = load_evaluation_artifacts(
            run_dir,
            split=split,
            policy=policy,
        )
    except FileNotFoundError:
        return None
    token_artifacts = evaluation.token_artifacts
    if token_artifacts is None:
        return None
    raw_top_ids = token_artifacts.get(
        "top10_s1_ids_at_reported_horizons"
    )
    raw_top_probabilities = token_artifacts.get(
        "top10_s1_probabilities_at_reported_horizons"
    )
    if raw_top_ids is None or raw_top_probabilities is None:
        return None
    top_ids = torch.as_tensor(raw_top_ids).long()
    top_probabilities = torch.as_tensor(raw_top_probabilities).float()
    if top_ids.ndim != 4 or tuple(top_ids.shape) != tuple(top_probabilities.shape):
        raise ValueError(
            "Saved top-10 token IDs/probabilities must share shape [W,H,N,K]."
        )
    reported_horizons = _saved_token_horizons(
        token_artifacts,
        evaluation.info.horizons,
    )
    requested_horizon = int(horizon)
    if requested_horizon not in reported_horizons:
        prediction_mode = str(token_artifacts.get("prediction_mode", ""))
        if prediction_mode == "dense_one_step":
            raise ValueError(
                "This dense teacher-forced BaseDyGraph run has only one public "
                "forecast horizon (h=1).  Its other 59 outputs are context "
                "transition positions, not future horizons."
            )
        raise ValueError(
            f"Saved probability summaries exist only at horizons "
            f"{reported_horizons}; received h={requested_horizon}."
        )
    horizon_position = reported_horizons.index(requested_horizon)
    top_ids = top_ids[:, horizon_position]
    top_probabilities = top_probabilities[:, horizon_position]
    targets = _saved_token_targets_for_horizons(
        token_artifacts,
        requested_horizons=(requested_horizon,),
        reported_horizons=reported_horizons,
    )[:, 0]
    if tuple(top_ids.shape[:2]) != tuple(targets.shape):
        raise ValueError(
            "Saved top-10 probability and target tensors are incompatible."
        )

    selected = _select_saved_token_window_indices(
        int(top_ids.shape[0]),
        window_indices=window_indices,
        max_windows=max_windows,
    )
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    top_ids = top_ids.index_select(0, selected_tensor)
    top_probabilities = top_probabilities.index_select(0, selected_tensor)
    targets = targets.index_select(0, selected_tensor)

    asset_index, asset_label = _resolve_saved_token_asset(
        asset,
        evaluation.info.asset_cols,
    )
    if asset_index is not None:
        top_ids = top_ids[:, asset_index : asset_index + 1]
        top_probabilities = top_probabilities[:, asset_index : asset_index + 1]
        targets = targets[:, asset_index : asset_index + 1]

    training = load_evaluation_artifacts(
        run_dir,
        split="train",
        policy=None,
    )
    if training.token_artifacts is None:
        raise FileNotFoundError(
            "No saved training token artefact exists for the selected model."
        )
    training_reported = _saved_token_horizons(
        training.token_artifacts,
        training.info.horizons,
    )
    training_targets = _saved_token_targets_for_horizons(
        training.token_artifacts,
        requested_horizons=(requested_horizon,),
        reported_horizons=training_reported,
    )[:, 0]
    if asset_index is not None:
        training_targets = training_targets[:, asset_index : asset_index + 1]

    vocabulary_size = max(
        1024,
        int(top_ids.max().item()) + 1,
        int(targets.max().item()) + 1,
        int(training_targets.max().item()) + 1,
    )
    flat_ids = top_ids.reshape(-1, int(top_ids.shape[-1]))
    flat_probabilities = top_probabilities.reshape(
        -1,
        int(top_probabilities.shape[-1]),
    )
    observations = int(flat_ids.shape[0])
    probability_sum = torch.zeros(vocabulary_size, dtype=torch.float64)
    probability_sum.scatter_add_(
        0,
        flat_ids.reshape(-1),
        flat_probabilities.reshape(-1).to(torch.float64),
    )
    mean_probability_contribution = probability_sum / max(observations, 1)
    hard_ids = flat_ids[:, 0]
    hard_counts = torch.bincount(hard_ids, minlength=vocabulary_size).double()
    hard_frequency = hard_counts / max(observations, 1)
    flat_training = training_targets.reshape(-1).long()
    training_counts = torch.bincount(
        flat_training,
        minlength=vocabulary_size,
    ).double()
    training_frequency = training_counts / max(int(flat_training.numel()), 1)
    captured_mass = float(flat_probabilities.sum(dim=-1).mean().item())
    hard_entropy = _distribution_entropy_from_probabilities(hard_frequency)

    summary = pd.DataFrame(
        [
            {
                "Checkpoint": "best",
                "Asset": asset_label,
                "Horizon": requested_horizon,
                "Evaluation split": evaluation.split,
                "Policy": evaluation.policy,
                "Evaluation windows": len(selected),
                "Evaluation predictions": observations,
                "Training targets": int(flat_training.numel()),
                "Saved probability depth": int(top_ids.shape[-1]),
                "Mean probability mass captured by saved top-k (%)": (
                    100.0 * captured_mass
                ),
                "Mean maximum token probability (%)": float(
                    100.0 * flat_probabilities[:, 0].mean().item()
                ),
                "Hard-prediction entropy (nats)": hard_entropy,
                "Hard-prediction effective vocabulary": float(
                    np.exp(hard_entropy)
                ),
                "Distinct hard-predicted tokens": int((hard_counts > 0).sum()),
                "Largest hard-prediction share (%)": float(
                    hard_frequency.max().item() * 100.0
                ),
                "Probability note": (
                    "Mean probability bars are exact saved top-k contributions; "
                    "tail probability outside the saved top-k is omitted."
                ),
            }
        ]
    )

    predicted_ids = torch.nonzero(hard_counts > 0).flatten()
    token_table = pd.DataFrame(
        {
            "Token ID": predicted_ids.numpy(),
            "Predicted Count": hard_counts[predicted_ids].long().numpy(),
            "Predicted Frequency (%)": (
                hard_frequency[predicted_ids].numpy() * 100.0
            ),
            "Mean Saved Top-k Probability Contribution (%)": (
                mean_probability_contribution[predicted_ids].numpy() * 100.0
            ),
            "Training Target Count": training_counts[predicted_ids].long().numpy(),
            "Training Target Frequency (%)": (
                training_frequency[predicted_ids].numpy() * 100.0
            ),
        }
    ).sort_values(
        ["Predicted Frequency (%)", "Token ID"],
        ascending=[False, True],
        ignore_index=True,
    )

    ranking = torch.argsort(
        mean_probability_contribution,
        descending=True,
    )
    shown = min(int(plot_top_n), vocabulary_size)
    shown_ids = ranking[:shown].numpy()
    x = np.arange(shown)
    width = 0.27
    figure, axes = plt.subplots(figsize=figsize)
    axes.bar(
        x - width,
        mean_probability_contribution[shown_ids].numpy() * 100.0,
        width,
        label=f"Mean saved top-{int(top_ids.shape[-1])} probability contribution",
    )
    axes.bar(
        x,
        hard_frequency[shown_ids].numpy() * 100.0,
        width,
        label="Hard argmax frequency",
    )
    axes.bar(
        x + width,
        training_frequency[shown_ids].numpy() * 100.0,
        width,
        label="Training target frequency",
    )
    axes.set_xticks(x)
    axes.set_xticklabels(shown_ids, rotation=90)
    axes.set_xlabel("Coarse token ID")
    axes.set_ylabel("Percentage")
    axes.set_title(
        "Coarse-token predictive distribution from saved selected-checkpoint artefacts\n"
        f"split={evaluation.split}; asset={asset_label}; horizon={requested_horizon}"
    )
    axes.legend()
    figure.tight_layout()
    plt.close(figure)

    token_index = pd.Index(
        np.arange(vocabulary_size),
        name="Token ID",
    )
    return {
        "summary": summary,
        "token_table": token_table,
        "mean_predictive_distribution": pd.Series(
            mean_probability_contribution.numpy(),
            index=token_index,
            name="Mean saved top-k probability contribution",
        ),
        "hard_prediction_distribution": pd.Series(
            hard_frequency.numpy(),
            index=token_index,
            name="Hard prediction frequency",
        ),
        "training_target_distribution": pd.Series(
            training_frequency.numpy(),
            index=token_index,
            name="Training target frequency",
        ),
        "validation_window_indices": selected,
        "evaluation_window_indices": selected,
        "probability_mass_coverage": captured_mass,
        "probability_distribution_is_truncated": True,
        "saved_probability_top_k": int(top_ids.shape[-1]),
        "figure": figure,
        "axes": axes,
    }

def analyse_coarse_token_predictive_distribution(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    asset: str | int | Literal["random"] | None = None,
    horizon: int = 1,
    source: MetricsSource = "best",
    bars: Sequence[TokenDistributionBar] = (
        "mean_probability",
        "hard_prediction_frequency",
        "training_target_frequency",
        "sampled_frequency",
    ),
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    plot_top_n: int = 40,
    random_seed: int = 42,
    figsize: tuple[float, float] = (16.0, 5.5),
) -> dict[str, Any]:
    """Plot selectable coarse-token distributions for one saved checkpoint.

    For new token runners this function is artefact-first: it consumes the
    compact selected-checkpoint top-10 summaries and does not reconstruct an
    architecture-specific model.  Historical runs without those artefacts use
    the legacy checkpoint-inference fallback.

    ``sampled_frequency`` is loaded only when that bar is requested.  Models
    that have not yet undergone the expensive sampled-decoding phase can still
    display the other three distributions.
    """

    allowed = {
        "mean_probability",
        "hard_prediction_frequency",
        "training_target_frequency",
        "sampled_frequency",
    }
    selected_bars = tuple(str(value) for value in bars)
    if not selected_bars or len(set(selected_bars)) != len(selected_bars):
        raise ValueError("bars must be a non-empty sequence without duplicates.")
    unknown = set(selected_bars) - allowed
    if unknown:
        raise ValueError(f"Unknown token-distribution bars: {sorted(unknown)}")

    run_dir = resolve_model_folder(model)
    unified_info = load_unified_run_info(run_dir)
    resolved_split = normalise_evaluation_split(split)
    selected_asset = asset
    if isinstance(asset, str) and asset.strip().lower() == "random":
        selected_asset = _resolve_random_asset(
            "random",
            unified_info.asset_cols,
            random_seed=random_seed,
        )

    base: dict[str, Any] | None = None
    if source == "best":
        base = _analyse_saved_token_predictive_distribution(
            run_dir,
            split=resolved_split,
            policy=policy,
            asset=selected_asset,
            horizon=int(horizon),
            window_indices=window_indices,
            max_windows=max_windows,
            plot_top_n=plot_top_n,
            figsize=figsize,
        )

    if base is None:
        # Historical fallback.  New BaseDyGraph-v1/Round-2 token models never
        # enter this branch because they save compact token summaries.
        legacy_info = load_run_info(run_dir)
        cache_path = resolve_token_cache_path(split=resolved_split)
        training_cache_path = resolve_token_cache_path(split="train")
        base = analyse_token_predictive_distribution(
            run_dir,
            asset=selected_asset,
            horizon=int(horizon),
            source=source,
            validation_cache_path=cache_path,
            training_cache_path=training_cache_path,
            window_indices=window_indices,
            max_windows=max_windows,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
            plot_top_n=plot_top_n,
            figsize=figsize,
        )
        del legacy_info

    artifacts = load_evaluation_artifacts(
        run_dir,
        split=resolved_split,
        policy=policy,
    )
    vocabulary_size = len(base["mean_predictive_distribution"])
    token_index = base["mean_predictive_distribution"].index

    sampled_series: pd.Series | None = None
    if "sampled_frequency" in selected_bars:
        if artifacts.token_artifacts is None:
            raise FileNotFoundError(
                f"No saved token artefact exists for "
                f"{resolved_split}/{artifacts.policy}."
            )
        sampled_values = artifacts.token_artifacts.get(
            "sampled_s1_evaluation"
        )
        if sampled_values is None:
            raise FileNotFoundError(
                "sampled_frequency was requested, but this model has no "
                "saved sampled_s1_evaluation tensor.  Run sampled decoding "
                "for the selected model or remove 'sampled_frequency' from bars."
            )
        sampled = torch.as_tensor(sampled_values).long()
        if sampled.ndim != 4:
            raise ValueError(
                "sampled_s1_evaluation must have shape [S,W,H,N]."
            )
        sampled_horizons = _saved_token_horizons(
            artifacts.token_artifacts,
            artifacts.info.horizons,
        )
        if int(horizon) not in sampled_horizons:
            raise ValueError(
                "Sampled-token frequencies are saved only at horizons "
                f"{sampled_horizons}; received {horizon}."
            )
        horizon_index = sampled_horizons.index(int(horizon))
        selected_indices = torch.as_tensor(
            base.get(
                "evaluation_window_indices",
                base["validation_window_indices"],
            ),
            dtype=torch.long,
        )
        sampled = sampled.index_select(1, selected_indices)[
            :, :, horizon_index
        ]
        asset_index, _ = _resolve_saved_token_asset(
            selected_asset,
            unified_info.asset_cols,
        )
        if asset_index is not None:
            sampled = sampled[
                :, :, asset_index : asset_index + 1
            ]
        flat_sampled = sampled.reshape(-1)
        sampled_counts = torch.bincount(
            flat_sampled,
            minlength=vocabulary_size,
        ).double()
        sampled_frequency = sampled_counts / max(
            int(flat_sampled.numel()),
            1,
        )
        sampled_series = pd.Series(
            sampled_frequency.numpy(),
            index=token_index,
            name="Sampled token frequency",
        )

    distributions: dict[str, pd.Series] = {
        "mean_probability": base["mean_predictive_distribution"],
        "hard_prediction_frequency": base["hard_prediction_distribution"],
        "training_target_frequency": base["training_target_distribution"],
    }
    if sampled_series is not None:
        distributions["sampled_frequency"] = sampled_series

    probability_label = (
        f"Mean saved top-{base.get('saved_probability_top_k')} "
        "probability contribution"
        if base.get("probability_distribution_is_truncated")
        else "Mean model probability"
    )
    labels = {
        "mean_probability": probability_label,
        "hard_prediction_frequency": "Hard argmax frequency",
        "training_target_frequency": "Training target frequency",
        "sampled_frequency": "Actual sampled frequency",
    }

    ranking = np.zeros(vocabulary_size, dtype=np.float64)
    for key in selected_bars:
        if key not in distributions:
            raise RuntimeError(
                f"Distribution {key!r} was requested but could not be loaded."
            )
        ranking = np.maximum(
            ranking,
            distributions[key].to_numpy(dtype=np.float64),
        )
    shown_ids = np.argsort(ranking)[::-1][
        : min(int(plot_top_n), vocabulary_size)
    ]

    figure, axes = plt.subplots(figsize=figsize)
    x = np.arange(len(shown_ids))
    width = 0.82 / len(selected_bars)
    offsets = (
        np.arange(len(selected_bars))
        - (len(selected_bars) - 1) / 2.0
    ) * width
    for offset, key in zip(offsets, selected_bars, strict=True):
        axes.bar(
            x + offset,
            distributions[key].iloc[shown_ids].to_numpy() * 100.0,
            width,
            label=labels[key],
        )
    axes.set_xticks(x)
    axes.set_xticklabels(shown_ids, rotation=90)
    axes.set_xlabel("Coarse token ID")
    axes.set_ylabel("Percentage of positions")
    asset_label = str(base["summary"].iloc[0]["Asset"])
    axes.set_title(
        "Coarse-token predictive and sampled distributions\n"
        f"split={resolved_split}; policy={artifacts.policy}; "
        f"asset={asset_label}; horizon={int(horizon)} min"
    )
    axes.legend()
    axes.grid(True, axis="y", alpha=0.20)
    figure.tight_layout()

    table_values: dict[str, Any] = {
        "Token ID": np.arange(vocabulary_size),
        (
            "Mean saved top-k probability contribution (%)"
            if base.get("probability_distribution_is_truncated")
            else "Mean model probability (%)"
        ): distributions["mean_probability"].to_numpy() * 100.0,
        "Hard argmax frequency (%)": distributions[
            "hard_prediction_frequency"
        ].to_numpy()
        * 100.0,
        "Training target frequency (%)": distributions[
            "training_target_frequency"
        ].to_numpy()
        * 100.0,
    }
    if sampled_series is not None:
        table_values["Actual sampled frequency (%)"] = (
            sampled_series.to_numpy() * 100.0
        )
    table = pd.DataFrame(table_values)
    sort_column = (
        "Actual sampled frequency (%)"
        if sampled_series is not None
        else "Hard argmax frequency (%)"
    )
    table = table.sort_values(
        sort_column,
        ascending=False,
    ).reset_index(drop=True)

    result = dict(base)
    result.update(
        {
            "sampled_distribution": sampled_series,
            "distribution_table": table,
            "bars": selected_bars,
            "policy": artifacts.policy,
            "split": resolved_split,
            "figure": figure,
            "axes": axes,
        }
    )
    return result



def _select_saved_token_window_indices(
    window_count: int,
    *,
    window_indices: Sequence[int] | None,
    max_windows: int | None,
) -> tuple[int, ...]:
    """Select saved-token windows without reconstructing a model or dataset."""

    if int(window_count) <= 0:
        raise ValueError("Saved token artefact contains no windows.")
    if window_indices is not None:
        selected = tuple(int(value) for value in window_indices)
        if not selected:
            raise ValueError("window_indices must not be empty.")
    elif max_windows is None or int(max_windows) >= int(window_count):
        selected = tuple(range(int(window_count)))
    else:
        count = int(max_windows)
        if count <= 0:
            raise ValueError("max_windows must be positive or None.")
        selected = tuple(
            int(value)
            for value in np.linspace(
                0,
                int(window_count) - 1,
                num=count,
                dtype=np.int64,
            )
        )
    if len(set(selected)) != len(selected):
        raise ValueError("Selected saved-token window indices are not unique.")
    if min(selected) < 0 or max(selected) >= int(window_count):
        raise IndexError("A saved-token window index is out of range.")
    return selected


def _resolve_saved_token_asset(
    asset: str | int | None,
    asset_cols: Sequence[str],
) -> tuple[int | None, str]:
    """Resolve an optional ticker/index against a saved asset order."""

    labels = tuple(str(value) for value in asset_cols)
    if asset is None:
        return None, "All assets"
    if isinstance(asset, int) and not isinstance(asset, bool):
        index = int(asset)
        if not 0 <= index < len(labels):
            raise IndexError(f"Asset index {index} is out of range.")
        return index, labels[index]
    if isinstance(asset, str):
        if asset not in labels:
            raise KeyError(f"Unknown asset {asset!r}.")
        return labels.index(asset), asset
    raise TypeError("asset must be None, a ticker, or a zero-based integer index.")


def _saved_token_horizons(
    token_artifacts: Mapping[str, Any],
    fallback: Sequence[int],
) -> tuple[int, ...]:
    raw = token_artifacts.get("evaluation_horizons", fallback)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("Saved token evaluation_horizons must be a sequence.")
    values = tuple(int(value) for value in raw)
    if not values or values != tuple(sorted(set(values))):
        raise ValueError(
            "Saved token evaluation_horizons must be non-empty, unique, "
            "and increasing."
        )
    return values


def _saved_token_targets_for_horizons(
    token_artifacts: Mapping[str, Any],
    *,
    requested_horizons: Sequence[int],
    reported_horizons: Sequence[int],
) -> Tensor:
    """Return saved s1 targets as [W,H,N] for requested public horizons."""

    requested = tuple(int(value) for value in requested_horizons)
    reported = tuple(int(value) for value in reported_horizons)
    candidates = (
        token_artifacts.get("target_s1"),
        token_artifacts.get("dense_target_s1"),
    )
    for raw in candidates:
        if raw is None:
            continue
        values = torch.as_tensor(raw).long()
        if values.ndim != 3:
            continue
        if int(values.shape[1]) == len(reported):
            positions = [reported.index(value) for value in requested]
            return values.index_select(
                1,
                torch.tensor(positions, dtype=torch.long),
            )
        if requested and int(values.shape[1]) >= max(requested):
            positions = [value - 1 for value in requested]
            return values.index_select(
                1,
                torch.tensor(positions, dtype=torch.long),
            )
    raise KeyError(
        "Saved token artefacts do not contain target_s1/dense_target_s1 "
        "with a horizon axis compatible with the requested public horizons."
    )


def _analyse_saved_coarse_token_topk(
    run_dir: Path,
    *,
    split: EvaluationSplitInput | str,
    policy: str | None,
    asset: str | int | None,
    window_indices: Sequence[int] | None,
    max_windows: int | None,
    top_k_values: Sequence[int],
    horizons: Sequence[int] | None,
) -> pd.DataFrame | None:
    """Compute exact top-k diagnostics from selected-checkpoint artefacts.

    New token runners intentionally save compact top-10 IDs/probabilities
    instead of the enormous full [W,60,N,1024] logit tensor.  Those IDs are
    sufficient for exact top-k accuracy for every requested k <= 10 and avoid
    reconstructing an architecture-specific model inside Graph Hub.
    """

    try:
        evaluation = load_evaluation_artifacts(
            run_dir,
            split=split,
            policy=policy,
        )
    except FileNotFoundError:
        return None
    token_artifacts = evaluation.token_artifacts
    if token_artifacts is None:
        return None
    raw_top_ids = token_artifacts.get(
        "top10_s1_ids_at_reported_horizons"
    )
    if raw_top_ids is None:
        return None
    top_ids = torch.as_tensor(raw_top_ids).long()
    if top_ids.ndim != 4:
        raise ValueError(
            "top10_s1_ids_at_reported_horizons must have shape [W,H,N,K]."
        )

    reported_horizons = _saved_token_horizons(
        token_artifacts,
        evaluation.info.horizons,
    )
    requested_horizons = (
        reported_horizons
        if horizons is None
        else tuple(int(value) for value in horizons)
    )
    if not requested_horizons or len(set(requested_horizons)) != len(
        requested_horizons
    ):
        raise ValueError("horizons must be non-empty and unique.")
    unavailable = [
        value for value in requested_horizons if value not in reported_horizons
    ]
    if unavailable:
        prediction_mode = str(
            token_artifacts.get("prediction_mode", "")
        )
        if prediction_mode == "dense_one_step":
            raise ValueError(
                "This is a dense teacher-forced one-step BaseDyGraph run. "
                "Its 60 output positions are context-to-next-token transitions, "
                "not free-running future horizons.  The only public forecast "
                "horizon is h=1.  Use horizons=(1,) here; use the parallel_60 "
                "control for h=1,5,15,30,60 future-horizon accuracy."
            )
        raise ValueError(
            "Saved compact top-k artefacts are available only at public "
            f"horizons {reported_horizons}; unavailable={unavailable}."
        )

    top_ks = tuple(int(value) for value in top_k_values)
    saved_k = int(top_ids.shape[-1])
    if not top_ks or len(set(top_ks)) != len(top_ks):
        raise ValueError("top_k_values must be non-empty and unique.")
    if any(value < 1 or value > saved_k for value in top_ks):
        raise ValueError(
            f"Saved compact logits support exact top-k only for k in "
            f"[1,{saved_k}]; received {top_ks}."
        )

    horizon_positions = [
        reported_horizons.index(value) for value in requested_horizons
    ]
    top_ids = top_ids.index_select(
        1,
        torch.tensor(horizon_positions, dtype=torch.long),
    )
    targets = _saved_token_targets_for_horizons(
        token_artifacts,
        requested_horizons=requested_horizons,
        reported_horizons=reported_horizons,
    )
    if tuple(top_ids.shape[:3]) != tuple(targets.shape):
        raise ValueError(
            "Saved top-k IDs and targets have incompatible shapes: "
            f"{tuple(top_ids.shape)} vs {tuple(targets.shape)}."
        )

    selected = _select_saved_token_window_indices(
        int(top_ids.shape[0]),
        window_indices=window_indices,
        max_windows=max_windows,
    )
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    top_ids = top_ids.index_select(0, selected_tensor)
    targets = targets.index_select(0, selected_tensor)

    asset_index, asset_label = _resolve_saved_token_asset(
        asset,
        evaluation.info.asset_cols,
    )
    if asset_index is not None:
        top_ids = top_ids[:, :, asset_index : asset_index + 1]
        targets = targets[:, :, asset_index : asset_index + 1]

    training = load_evaluation_artifacts(
        run_dir,
        split="train",
        policy=None,
    )
    if training.token_artifacts is None:
        raise FileNotFoundError(
            "The selected checkpoint has no saved training token artefact, "
            "so the training-frequency marginal baseline cannot be computed."
        )
    training_reported_horizons = _saved_token_horizons(
        training.token_artifacts,
        training.info.horizons,
    )
    training_targets = _saved_token_targets_for_horizons(
        training.token_artifacts,
        requested_horizons=requested_horizons,
        reported_horizons=training_reported_horizons,
    )
    if asset_index is not None:
        training_targets = training_targets[
            :, :, asset_index : asset_index + 1
        ]

    rows: dict[tuple[str, str], list[float]] = {}
    marginal_values: dict[int, dict[str, float]] = {}
    marginal_ids: dict[int, list[int]] = {}
    for horizon_index, horizon in enumerate(requested_horizons):
        evaluation_target = targets[:, horizon_index].reshape(-1)
        evaluation_top_ids = top_ids[:, horizon_index].reshape(
            -1,
            saved_k,
        )
        train_target = training_targets[:, horizon_index].reshape(-1)
        vocabulary_size = max(
            int(evaluation_top_ids.max().item()) + 1,
            int(evaluation_target.max().item()) + 1,
            int(train_target.max().item()) + 1,
            1024,
        )
        counts = torch.bincount(
            train_target,
            minlength=vocabulary_size,
        )
        token_ids = np.arange(vocabulary_size, dtype=np.int64)
        order = torch.from_numpy(
            np.lexsort((token_ids, -counts.cpu().numpy())).copy()
        ).long()
        marginal_ids[int(horizon)] = order[: max(top_ks)].tolist()
        marginal_values[int(horizon)] = {}
        for k in top_ks:
            model_accuracy = float(
                evaluation_top_ids[:, :k]
                .eq(evaluation_target.unsqueeze(-1))
                .any(dim=-1)
                .float()
                .mean()
                .item()
                * 100.0
            )
            marginal_accuracy = float(
                evaluation_target.unsqueeze(-1)
                .eq(order[:k].reshape(1, -1))
                .any(dim=-1)
                .float()
                .mean()
                .item()
                * 100.0
            )
            rows.setdefault((f"Top-{k}", "Model Accuracy (%)"), []).append(
                model_accuracy
            )
            rows.setdefault(
                (f"Top-{k}", "Excess vs Marginal (pp)"), []
            ).append(model_accuracy - marginal_accuracy)
            marginal_values[int(horizon)][f"Top-{k}"] = marginal_accuracy

    table = pd.DataFrame(
        rows,
        index=pd.Index(
            requested_horizons,
            name="Future horizon (minutes)",
        ),
    )
    table.columns = pd.MultiIndex.from_tuples(
        table.columns,
        names=("Candidate set", "Metric"),
    )
    table.attrs.update(
        {
            "run_directory": str(evaluation.info.run_dir),
            "checkpoint": "best",
            "split": evaluation.split,
            "policy": evaluation.policy,
            "asset": asset_label,
            "evaluation_windows": len(selected),
            "validation_windows": len(selected),
            "marginal_baseline_accuracy_percent": marginal_values,
            "training_top_token_ids_by_horizon": marginal_ids,
            "source": "saved_selected_checkpoint_top10",
        }
    )
    return table

def analyse_coarse_token_topk(
    model: str | Path,
    *,
    split: EvaluationSplitInput | str = "validation",
    policy: str | None = None,
    asset: str | int | Literal["random"] | None = None,
    source: MetricsSource = "best",
    window_indices: Sequence[int] | None = None,
    max_windows: int | None = None,
    batch_size: int = 2,
    device: str | torch.device | None = None,
    use_amp: bool | None = None,
    top_k_values: Sequence[int] = (1, 2, 5, 10),
    horizons: Sequence[int] | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return coarse-token top-k accuracy from saved selected artefacts.

    New token runners use architecture-specific model classes and save compact
    top-10 predictions for the selected checkpoint.  Graph Hub consumes those
    artefacts directly rather than trying to rebuild every run as the historical
    ``DynamicGraphTokenForecaster``.  Legacy runs without compact artefacts keep
    the previous inference fallback.
    """

    run_dir = resolve_model_folder(model)
    unified_info = load_unified_run_info(run_dir)
    resolved_asset = asset
    if isinstance(asset, str) and asset.strip().lower() == "random":
        resolved_asset = _resolve_random_asset(
            "random", unified_info.asset_cols, random_seed=random_seed
        )
    resolved_split = normalise_evaluation_split(split)

    if source == "best":
        saved = _analyse_saved_coarse_token_topk(
            run_dir,
            split=resolved_split,
            policy=policy,
            asset=resolved_asset,
            window_indices=window_indices,
            max_windows=max_windows,
            top_k_values=top_k_values,
            horizons=horizons,
        )
        if saved is not None:
            return saved

    # Historical fallback for runs that predate selected-checkpoint token
    # artefacts.  This path intentionally remains available, but it is not used
    # for the new Round-2/BaseDyGraph-v1 token models.
    result = analyse_s1_topk_accuracy_by_horizon(
        run_dir,
        asset=resolved_asset,
        source=source,
        validation_cache_path=resolve_token_cache_path(split=resolved_split),
        training_cache_path=resolve_token_cache_path(split="train"),
        window_indices=window_indices,
        max_windows=max_windows,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
        top_k_values=top_k_values,
        horizons=horizons,
    )
    result.attrs["split"] = resolved_split
    result.attrs["policy"] = policy
    result.attrs["source"] = "legacy_checkpoint_reinference"
    return result

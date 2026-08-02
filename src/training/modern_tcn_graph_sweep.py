from __future__ import annotations

"""Staged 48-run ModernTCN + graph sweep configuration helpers.

The forecasting objective is fixed throughout:

    10,000 × mean cumulative-log-change MAE

Only January--August training and September validation are used. The held-out
proposed-model test split is never loaded by this module.

The exact budget is:

1. 24 ModernTCN-backbone configurations, each trained as a complete
   ModernTCN + one-head free-static graph + spatial-mixer model;
2. 16 graph alternatives across the two best Stage-1 backbones;
3. 8 optimisation refinements around the best graph-enabled Stage-2 model.

Stage 3 explicitly treats the graph learning rate as a separate optimisation
axis for learned graphs. If a fixed correlation graph wins, Stage 3 instead
uses the otherwise-unused budget to refine the ModernTCN FFN ratio.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


HORIZONS = (1, 5, 15, 30, 60)
STAGE1_D_MODELS = (32, 64, 128)
STAGE1_BLOCK_COUNTS = (1, 3)
STAGE1_PATCH_KERNELS = (
    (4, 2, 15),
    (4, 2, 51),
    (8, 4, 15),
    (8, 4, 51),
)
STAGE3_BACKBONE_LEARNING_RATES = (5.0e-5, 1.0e-4, 2.5e-4)
STAGE3_GRAPH_LEARNING_RATES = (5.0e-4, 1.0e-3, 2.0e-3)
STAGE3_FIXED_GRAPH_FFN_RATIOS = (1, 2, 4)


@dataclass(frozen=True)
class SweepRunSpec:
    run_name: str
    description: str
    tags: tuple[str, ...]
    overrides: tuple[str, ...]
    stage: int
    backbone_id: str
    graph_variant: str
    d_model: int
    num_blocks: int
    patch_size: int
    patch_stride: int
    large_kernel: int
    ffn_ratio: int
    learning_rate: float
    graph_learning_rate: float | None
    graph_enabled: bool
    trainable_graph: bool
    eligible_for_final: bool

    def launcher_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "description": self.description,
            "tags": list(self.tags),
            "overrides": list(self.overrides),
        }

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["tags"] = list(self.tags)
        values["overrides"] = list(self.overrides)
        return values


@dataclass(frozen=True)
class SweepSelection:
    stage: str
    selected_run_names: tuple[str, ...]
    selection_metric: str
    selected_at_score: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "selected_run_names": list(self.selected_run_names),
            "selection_metric": self.selection_metric,
            "selected_at_score": list(self.selected_at_score),
        }


def _float_tag(value: float) -> str:
    return (
        f"{value:g}"
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
    )


def graph_regularisation_off_overrides() -> tuple[str, ...]:
    return (
        "model.graph_regularisation.graph_reg_layer=-1",
        "model.graph_regularisation.graph_reg_warmup_epochs=0",
        "model.graph_regularisation.graph_entropy_reg=0.0",
        "model.graph_regularisation.graph_target_entropy=null",
        "model.graph_regularisation.graph_target_entropy_reg=0.0",
        "model.graph_regularisation.graph_temporal_smooth_reg=0.0",
    )


def common_modern_tcn_overrides(
    *,
    graph_learning_rate: float,
) -> tuple[str, ...]:
    """Return settings frozen across all 48 search runs."""
    if graph_learning_rate <= 0.0:
        raise ValueError("graph_learning_rate must be positive.")
    return (
        "data.input_representation=raw",
        "model.output_representation=normalised_close",
        "model.output_head_initialisation=default",
        "model.temporal.type=modern_tcn",
        # The custom p/sin/cos feature harmed the earlier ModernTCN baseline.
        "model.temporal.session_position_encoding=false",
        # Transformer-only setting, recorded explicitly for reproducibility.
        "model.temporal.relative_position_embedding=false",
        "model.temporal.modern_tcn.small_kernel=5",
        "model.temporal.modern_tcn.dropout=0.05",
        "model.temporal.modern_tcn.head_dropout=0.0",
        "model.graph.activation=softmax",
        "model.graph.add_self_loops=false",
        "model.graph.hidden_dim=32",
        "model.graph.base_graph_type=free_static",
        "model.graph.gate_type=learned_scalar",
        "model.graph.initial_alpha=0.25",
        "model.spatial.num_layers=1",
        "model.spatial.feedforward_multiplier=2",
        "model.spatial.dropout=0.0",
        # Start close to the competent temporal solution. Beta remains logged.
        "model.spatial.gate_type=learned_scalar",
        "model.spatial.initial_beta=0.1",
        "model.head_dropout=0.0",
        "training.loss.type=cumulative_log_change_mae",
        "training.loss.bps_scale=10000.0",
        # Under this fixed loss, validation_loss is the exact all-horizon
        # mean cumulative-log-change MAE used for automatic selection.
        "training.selection_metric=validation_loss",
        "training.optimizer=adam",
        "training.scheduler=modern_tcn_type3",
        "training.weight_decay=0.0",
        "training.batch_size=16",
        "training.validation_batch_size=32",
        "training.num_workers=0",
        "training.max_epochs=100",
        "training.patience=10",
        "training.min_delta=0.0",
        "training.gradient_clip_norm=1.0",
        "training.graph_diagnostics_batches_per_epoch=1",
        "training.seed=42",
        f"training.graph_learning_rate={graph_learning_rate}",
    )


def _backbone_id(
    *,
    d_model: int,
    num_blocks: int,
    patch_size: int,
    patch_stride: int,
    large_kernel: int,
) -> str:
    return (
        f"d{d_model}_k{num_blocks}_"
        f"p{patch_size}s{patch_stride}_lk{large_kernel}"
    )


def _base_backbone_overrides(
    *,
    d_model: int,
    num_blocks: int,
    patch_size: int,
    patch_stride: int,
    large_kernel: int,
    ffn_ratio: int,
    learning_rate: float,
    graph_learning_rate: float,
) -> tuple[str, ...]:
    return (
        f"model.temporal.d_model={d_model}",
        f"model.temporal.modern_tcn.num_blocks={num_blocks}",
        f"model.temporal.modern_tcn.patch_size={patch_size}",
        f"model.temporal.modern_tcn.patch_stride={patch_stride}",
        f"model.temporal.modern_tcn.large_kernel={large_kernel}",
        f"model.temporal.modern_tcn.ffn_ratio={ffn_ratio}",
        f"training.learning_rate={learning_rate}",
        f"training.graph_learning_rate={graph_learning_rate}",
    )


def make_stage1_specs(
    *,
    graph_learning_rate: float = 1.0e-3,
) -> list[SweepRunSpec]:
    """Build the 24 complete ModernTCN + free-static graph runs."""
    if graph_learning_rate not in STAGE3_GRAPH_LEARNING_RATES:
        raise ValueError(
            "Stage-1 graph_learning_rate must be one of "
            f"{STAGE3_GRAPH_LEARNING_RATES} so Stage 3 reuses its baseline."
        )
    common = common_modern_tcn_overrides(
        graph_learning_rate=graph_learning_rate
    )
    regularisation_off = graph_regularisation_off_overrides()
    specs: list[SweepRunSpec] = []
    for d_model in STAGE1_D_MODELS:
        for num_blocks in STAGE1_BLOCK_COUNTS:
            for patch_size, patch_stride, large_kernel in STAGE1_PATCH_KERNELS:
                backbone_id = _backbone_id(
                    d_model=d_model,
                    num_blocks=num_blocks,
                    patch_size=patch_size,
                    patch_stride=patch_stride,
                    large_kernel=large_kernel,
                )
                graph_lr_tag = _float_tag(graph_learning_rate)
                run_name = f"mtg_s1_{backbone_id}_fs1_glr{graph_lr_tag}"
                overrides = (
                    *common,
                    *_base_backbone_overrides(
                        d_model=d_model,
                        num_blocks=num_blocks,
                        patch_size=patch_size,
                        patch_stride=patch_stride,
                        large_kernel=large_kernel,
                        ffn_ratio=1,
                        learning_rate=1.0e-4,
                        graph_learning_rate=graph_learning_rate,
                    ),
                    "model.graph.type=free_static",
                    "model.graph.num_heads=1",
                    *regularisation_off,
                )
                specs.append(
                    SweepRunSpec(
                        run_name=run_name,
                        description=(
                            "ModernTCN plus one-head learned free-static graph; "
                            f"D={d_model}, blocks={num_blocks}, "
                            f"patch/stride={patch_size}/{patch_stride}, "
                            f"large kernel={large_kernel}, FFN ratio=1, "
                            f"graph LR={graph_learning_rate:g}"
                        ),
                        tags=(
                            "mtcn-graph-sweep",
                            "stage1",
                            "free-static",
                            backbone_id,
                            f"glr{graph_lr_tag}",
                        ),
                        overrides=overrides,
                        stage=1,
                        backbone_id=backbone_id,
                        graph_variant="free_static_g1_no_reg",
                        d_model=d_model,
                        num_blocks=num_blocks,
                        patch_size=patch_size,
                        patch_stride=patch_stride,
                        large_kernel=large_kernel,
                        ffn_ratio=1,
                        learning_rate=1.0e-4,
                        graph_learning_rate=graph_learning_rate,
                        graph_enabled=True,
                        trainable_graph=True,
                        eligible_for_final=True,
                    )
                )
    if len(specs) != 24:
        raise AssertionError(f"Stage 1 must contain 24 runs, got {len(specs)}.")
    return specs


def _stage2_variant_definitions(
    *,
    target_entropy_coefficient: float,
) -> tuple[dict[str, Any], ...]:
    if target_entropy_coefficient <= 0.0:
        raise ValueError("target_entropy_coefficient must be positive.")
    coefficient_tag = _float_tag(target_entropy_coefficient)
    regularisation_off = graph_regularisation_off_overrides()
    return (
        {
            "name": "no_graph",
            "description": "matched ModernTCN temporal-only control",
            "graph_enabled": False,
            "trainable_graph": False,
            "eligible": False,
            "overrides": (
                "model.graph.type=none",
                "model.graph.num_heads=1",
                "model.spatial.gate_type=none",
                "model.spatial.initial_beta=1.0",
                *regularisation_off,
            ),
        },
        *tuple(
            {
                "name": f"corr{int(round(threshold * 1000)):03d}",
                "description": (
                    "training-only absolute Close-return correlation graph "
                    f"with threshold {threshold:.2f}"
                ),
                "graph_enabled": True,
                "trainable_graph": False,
                "eligible": True,
                "overrides": (
                    "model.graph.type=fixed",
                    "model.graph.num_heads=1",
                    "model.fixed_graph_resource.type=absolute_return_correlation",
                    "model.fixed_graph_resource.channel=close",
                    f"model.fixed_graph_resource.threshold={threshold}",
                    "model.fixed_graph_resource.empty_row_policy=strongest",
                    *regularisation_off,
                ),
            }
            for threshold in (0.12, 0.18, 0.24)
        ),
        {
            "name": "free_static_g2",
            "description": (
                "two-head learned free-static graph without regularisation"
            ),
            "graph_enabled": True,
            "trainable_graph": True,
            "eligible": True,
            "overrides": (
                "model.graph.type=free_static",
                "model.graph.num_heads=2",
                *regularisation_off,
            ),
        },
        {
            "name": f"free_static_h22_lam{coefficient_tag}",
            "description": (
                "one-head free-static graph with target entropy 2.2 and "
                f"coefficient {target_entropy_coefficient:g}"
            ),
            "graph_enabled": True,
            "trainable_graph": True,
            "eligible": True,
            "overrides": (
                "model.graph.type=free_static",
                "model.graph.num_heads=1",
                "model.graph_regularisation.graph_reg_layer=-1",
                "model.graph_regularisation.graph_reg_warmup_epochs=0",
                "model.graph_regularisation.graph_entropy_reg=0.0",
                "model.graph_regularisation.graph_target_entropy=2.2",
                (
                    "model.graph_regularisation.graph_target_entropy_reg="
                    f"{target_entropy_coefficient}"
                ),
                "model.graph_regularisation.graph_temporal_smooth_reg=0.0",
            ),
        },
        {
            "name": "dynamic_g1_h32",
            "description": (
                "one-head BaseDyGraph Q/K dynamic graph, hidden dimension 32"
            ),
            "graph_enabled": True,
            "trainable_graph": True,
            "eligible": True,
            "overrides": (
                "model.graph.type=dynamic",
                "model.graph.num_heads=1",
                "model.graph.hidden_dim=32",
                "model.graph.gate_type=none",
                *regularisation_off,
            ),
        },
        {
            "name": "dynamic_base_g1_h32_a025",
            "description": (
                "one-head free-static-plus-dynamic graph with learned alpha "
                "initialised at 0.25"
            ),
            "graph_enabled": True,
            "trainable_graph": True,
            "eligible": True,
            "overrides": (
                "model.graph.type=dynamic_base",
                "model.graph.num_heads=1",
                "model.graph.hidden_dim=32",
                "model.graph.base_graph_type=free_static",
                "model.graph.gate_type=learned_scalar",
                "model.graph.initial_alpha=0.25",
                *regularisation_off,
            ),
        },
    )


def make_stage2_specs(
    selected_backbones: Sequence[SweepRunSpec],
    *,
    target_entropy_coefficient: float = 1.0,
) -> list[SweepRunSpec]:
    """Build 16 new graph runs for exactly two Stage-1 winners."""
    if len(selected_backbones) != 2:
        raise ValueError("Stage 2 requires exactly two selected backbones.")
    variants = _stage2_variant_definitions(
        target_entropy_coefficient=target_entropy_coefficient
    )
    specs: list[SweepRunSpec] = []
    for backbone in selected_backbones:
        if backbone.graph_learning_rate is None:
            raise ValueError("Selected Stage-1 backbone has no graph LR.")
        graph_lr_tag = _float_tag(float(backbone.graph_learning_rate))
        for variant in variants:
            run_name = (
                f"mtg_s2_{backbone.backbone_id}_"
                f"baseglr{graph_lr_tag}_{variant['name']}"
            )
            specs.append(
                SweepRunSpec(
                    run_name=run_name,
                    description=(
                        f"{variant['description']} on Stage-1 backbone "
                        f"{backbone.backbone_id}"
                    ),
                    tags=(
                        "mtcn-graph-sweep",
                        "stage2",
                        str(variant["name"]),
                        backbone.backbone_id,
                    ),
                    overrides=(
                        *backbone.overrides,
                        *tuple(variant["overrides"]),
                    ),
                    stage=2,
                    backbone_id=backbone.backbone_id,
                    graph_variant=str(variant["name"]),
                    d_model=backbone.d_model,
                    num_blocks=backbone.num_blocks,
                    patch_size=backbone.patch_size,
                    patch_stride=backbone.patch_stride,
                    large_kernel=backbone.large_kernel,
                    ffn_ratio=1,
                    learning_rate=1.0e-4,
                    graph_learning_rate=(
                        backbone.graph_learning_rate
                        if bool(variant["trainable_graph"])
                        else None
                    ),
                    graph_enabled=bool(variant["graph_enabled"]),
                    trainable_graph=bool(variant["trainable_graph"]),
                    eligible_for_final=bool(variant["eligible"]),
                )
            )
    if len(specs) != 16:
        raise AssertionError(f"Stage 2 must contain 16 new runs, got {len(specs)}.")
    return specs


def make_stage3_specs(winner: SweepRunSpec) -> list[SweepRunSpec]:
    """Build eight optimisation refinements around the winning graph model."""
    if not winner.graph_enabled or not winner.eligible_for_final:
        raise ValueError("Stage 3 winner must be a graph-enabled candidate.")
    specs: list[SweepRunSpec] = []

    if winner.trainable_graph:
        if winner.graph_learning_rate not in STAGE3_GRAPH_LEARNING_RATES:
            raise ValueError(
                "Learned-graph winner must use a graph LR in the Stage-3 grid."
            )
        for learning_rate in STAGE3_BACKBONE_LEARNING_RATES:
            for graph_learning_rate in STAGE3_GRAPH_LEARNING_RATES:
                if (
                    math.isclose(learning_rate, winner.learning_rate)
                    and math.isclose(
                        graph_learning_rate,
                        float(winner.graph_learning_rate),
                    )
                ):
                    continue
                lr_tag = _float_tag(learning_rate)
                graph_lr_tag = _float_tag(graph_learning_rate)
                run_name = (
                    f"mtg_s3_{winner.backbone_id}_{winner.graph_variant}_"
                    f"lr{lr_tag}_glr{graph_lr_tag}"
                )
                specs.append(
                    SweepRunSpec(
                        run_name=run_name,
                        description=(
                            f"Stage-3 optimisation of {winner.run_name}: "
                            f"backbone LR={learning_rate:g}, "
                            f"graph LR={graph_learning_rate:g}"
                        ),
                        tags=(
                            "mtcn-graph-sweep",
                            "stage3",
                            winner.graph_variant,
                            f"lr{lr_tag}",
                            f"glr{graph_lr_tag}",
                        ),
                        overrides=(
                            *winner.overrides,
                            f"training.learning_rate={learning_rate}",
                            f"training.graph_learning_rate={graph_learning_rate}",
                        ),
                        stage=3,
                        backbone_id=winner.backbone_id,
                        graph_variant=winner.graph_variant,
                        d_model=winner.d_model,
                        num_blocks=winner.num_blocks,
                        patch_size=winner.patch_size,
                        patch_stride=winner.patch_stride,
                        large_kernel=winner.large_kernel,
                        ffn_ratio=winner.ffn_ratio,
                        learning_rate=learning_rate,
                        graph_learning_rate=graph_learning_rate,
                        graph_enabled=True,
                        trainable_graph=True,
                        eligible_for_final=True,
                    )
                )
    else:
        # A fixed correlation graph has no learnable graph parameters. Spend
        # the same eight-run refinement budget on backbone LR × FFN ratio.
        for learning_rate in STAGE3_BACKBONE_LEARNING_RATES:
            for ffn_ratio in STAGE3_FIXED_GRAPH_FFN_RATIOS:
                if (
                    math.isclose(learning_rate, winner.learning_rate)
                    and ffn_ratio == winner.ffn_ratio
                ):
                    continue
                lr_tag = _float_tag(learning_rate)
                run_name = (
                    f"mtg_s3_{winner.backbone_id}_{winner.graph_variant}_"
                    f"lr{lr_tag}_r{ffn_ratio}"
                )
                specs.append(
                    SweepRunSpec(
                        run_name=run_name,
                        description=(
                            f"Stage-3 optimisation of {winner.run_name}: "
                            f"backbone LR={learning_rate:g}, "
                            f"ModernTCN FFN ratio={ffn_ratio}"
                        ),
                        tags=(
                            "mtcn-graph-sweep",
                            "stage3",
                            winner.graph_variant,
                            f"lr{lr_tag}",
                            f"r{ffn_ratio}",
                        ),
                        overrides=(
                            *winner.overrides,
                            f"training.learning_rate={learning_rate}",
                            f"model.temporal.modern_tcn.ffn_ratio={ffn_ratio}",
                        ),
                        stage=3,
                        backbone_id=winner.backbone_id,
                        graph_variant=winner.graph_variant,
                        d_model=winner.d_model,
                        num_blocks=winner.num_blocks,
                        patch_size=winner.patch_size,
                        patch_stride=winner.patch_stride,
                        large_kernel=winner.large_kernel,
                        ffn_ratio=ffn_ratio,
                        learning_rate=learning_rate,
                        graph_learning_rate=None,
                        graph_enabled=True,
                        trainable_graph=False,
                        eligible_for_final=True,
                    )
                )

    if len(specs) != 8:
        raise AssertionError(f"Stage 3 must contain 8 new runs, got {len(specs)}.")
    return specs


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_row(
    output_root: str | Path,
    spec: SweepRunSpec,
) -> dict[str, Any] | None:
    run_dir = Path(output_root) / spec.run_name
    metadata_path = run_dir / "run_metadata.json"
    history_path = run_dir / "history.csv"
    if not metadata_path.is_file() or not history_path.is_file():
        return None
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "completed":
        return None
    history = pd.read_csv(history_path)
    best_epoch = int(metadata["best_epoch"])
    rows = history.loc[history["epoch"] == best_epoch]
    if len(rows) != 1:
        raise AssertionError(
            f"Expected one best-epoch row for {spec.run_name}, got {len(rows)}."
        )
    best = rows.iloc[0]

    def optional_float(column: str) -> float | None:
        if column not in best.index or pd.isna(best[column]):
            return None
        return float(best[column])

    row: dict[str, Any] = {
        "Run": spec.run_name,
        "Stage": spec.stage,
        "Backbone": spec.backbone_id,
        "Graph": spec.graph_variant,
        "Graph enabled": spec.graph_enabled,
        "Trainable graph": spec.trainable_graph,
        "Eligible for final": spec.eligible_for_final,
        "D": spec.d_model,
        "Blocks": spec.num_blocks,
        "Patch": spec.patch_size,
        "Stride": spec.patch_stride,
        "Large kernel": spec.large_kernel,
        "FFN ratio": spec.ffn_ratio,
        "Backbone learning rate": spec.learning_rate,
        "Graph learning rate": spec.graph_learning_rate,
        "Best epoch": best_epoch,
        "Epochs completed": int(metadata["epochs_completed"]),
        "Trainable parameters": int(metadata.get("trainable_parameters", 0)),
        "Graph parameters": int(metadata.get("graph_trainable_parameters", 0)),
        "Mean Log MAE": float(best["selection_score"]),
        "Graph entropy": optional_float("graph_mean_row_entropy"),
        "Effective neighbours": optional_float(
            "graph_mean_effective_neighbours"
        ),
        "Maximum edge weight": optional_float("graph_maximum_edge_weight"),
        "Top-10 row mass": optional_float("graph_mean_top10_row_mass"),
        "Spatial beta": optional_float("spatial_beta"),
        "Dynamic alpha": optional_float("dynamic_alpha"),
        "Combined graph gradient": optional_float(
            "training_graph_combined_gradient_norm"
        ),
        "Forecast graph gradient": optional_float(
            "training_graph_forecast_gradient_norm"
        ),
        "Regulariser graph gradient": optional_float(
            "training_graph_regulariser_gradient_norm"
        ),
        "Graph update norm": optional_float(
            "training_graph_parameter_update_norm"
        ),
        "Spatial-gate gradient": optional_float(
            "training_spatial_gate_gradient_norm"
        ),
    }
    for horizon in HORIZONS:
        row[f"Log MAE — {horizon} min"] = float(
            best[f"val_cumulative_log_change_mae_h{horizon}"]
        )
        median_column = (
            f"val_cumulative_log_change_median_absolute_error_h{horizon}"
        )
        p95_column = f"val_cumulative_log_change_p95_absolute_error_h{horizon}"
        row[f"Log MedAE — {horizon} min"] = optional_float(median_column)
        row[f"Log P95 AE — {horizon} min"] = optional_float(p95_column)
    return row


def summarise_specs(
    output_root: str | Path,
    specs: Sequence[SweepRunSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows = [result_row(output_root, spec) for spec in specs]
    completed = [row for row in rows if row is not None]
    if require_all and len(completed) != len(specs):
        missing = [
            spec.run_name
            for spec, row in zip(specs, rows, strict=True)
            if row is None
        ]
        raise RuntimeError(
            "Not every expected sweep run is complete. Missing/incomplete: "
            + ", ".join(missing)
        )
    if not completed:
        raise RuntimeError("No completed sweep runs were found.")
    return (
        pd.DataFrame(completed)
        .sort_values(["Mean Log MAE", "Trainable parameters", "Run"])
        .reset_index(drop=True)
    )


def select_specs(
    results: pd.DataFrame,
    spec_lookup: Mapping[str, SweepRunSpec],
    *,
    count: int,
    eligible_only: bool = False,
    relative_tie_tolerance: float = 0.0025,
) -> list[SweepRunSpec]:
    if count <= 0:
        raise ValueError("count must be positive.")
    candidates = results.copy()
    if eligible_only:
        candidates = candidates.loc[candidates["Eligible for final"]]
    if len(candidates) < count:
        raise RuntimeError(
            f"Need {count} completed eligible runs, found {len(candidates)}."
        )

    selected: list[SweepRunSpec] = []
    remaining = candidates.copy()
    while len(selected) < count:
        best_score = float(remaining["Mean Log MAE"].min())
        tolerance = abs(best_score) * float(relative_tie_tolerance)
        near = remaining.loc[
            remaining["Mean Log MAE"] <= best_score + tolerance
        ].sort_values(["Trainable parameters", "Mean Log MAE", "Run"])
        chosen_row = near.iloc[0]
        run_name = str(chosen_row["Run"])
        selected.append(spec_lookup[run_name])
        remaining = remaining.loc[remaining["Run"] != run_name]
    return selected


def save_specs(path: str | Path, specs: Iterable[SweepRunSpec]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [spec.to_dict() for spec in specs]
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_selection(path: str | Path, selection: SweepSelection) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def assert_total_budget(
    stage1_specs: Sequence[SweepRunSpec],
    stage2_specs: Sequence[SweepRunSpec],
    stage3_specs: Sequence[SweepRunSpec],
) -> None:
    names = [
        *(spec.run_name for spec in stage1_specs),
        *(spec.run_name for spec in stage2_specs),
        *(spec.run_name for spec in stage3_specs),
    ]
    if len(names) != 48:
        raise AssertionError(f"Expected 48 unique runs, got {len(names)}.")
    if len(set(names)) != 48:
        raise AssertionError("The 48-run sweep contains duplicate run names.")

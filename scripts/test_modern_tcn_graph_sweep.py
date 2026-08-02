from __future__ import annotations

"""Fast contract tests for the staged 48-run ModernTCN graph sweep."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.training.modern_tcn_graph_sweep import (
    HORIZONS,
    STAGE3_GRAPH_LEARNING_RATES,
    assert_total_budget,
    make_stage1_specs,
    make_stage2_specs,
    make_stage3_specs,
    select_specs,
    summarise_specs,
)


def _write_completed_run(
    root: Path,
    *,
    spec,
    score: float,
    parameters: int,
) -> None:
    run_dir = root / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_epoch": 3,
                "epochs_completed": 7,
                "trainable_parameters": int(parameters),
                "graph_trainable_parameters": (
                    100 if spec.trainable_graph else 0
                ),
            }
        ),
        encoding="utf-8",
    )
    row = {
        "epoch": 3,
        "selection_score": float(score),
        "graph_mean_row_entropy": 2.2 if spec.graph_enabled else None,
        "graph_mean_effective_neighbours": (
            9.025 if spec.graph_enabled else None
        ),
        "graph_maximum_edge_weight": 0.2 if spec.graph_enabled else None,
        "graph_mean_top10_row_mass": 0.8 if spec.graph_enabled else None,
        "spatial_beta": 0.4 if spec.graph_enabled else None,
        "dynamic_alpha": (
            0.3 if spec.graph_variant.startswith("dynamic_base") else None
        ),
        "training_graph_combined_gradient_norm": (
            0.1 if spec.trainable_graph else 0.0
        ),
        "training_graph_forecast_gradient_norm": (
            0.03 if spec.trainable_graph else 0.0
        ),
        "training_graph_regulariser_gradient_norm": (
            0.07 if spec.trainable_graph else 0.0
        ),
        "training_graph_parameter_update_norm": (
            0.005 if spec.trainable_graph else 0.0
        ),
        "training_spatial_gate_gradient_norm": (
            0.05 if spec.graph_enabled else 0.0
        ),
    }
    for horizon in HORIZONS:
        row[f"val_cumulative_log_change_mae_h{horizon}"] = float(score)
        row[
            f"val_cumulative_log_change_median_absolute_error_h{horizon}"
        ] = float(score) * 0.75
        row[
            f"val_cumulative_log_change_p95_absolute_error_h{horizon}"
        ] = float(score) * 2.5
    pd.DataFrame([row]).to_csv(run_dir / "history.csv", index=False)


def main() -> None:
    stage1_graph_lr = 1.0e-3
    stage1 = make_stage1_specs(graph_learning_rate=stage1_graph_lr)
    if len(stage1) != 24:
        raise AssertionError("Stage 1 does not contain 24 runs.")
    if len({spec.run_name for spec in stage1}) != 24:
        raise AssertionError("Stage-1 run names are not unique.")

    for spec in stage1:
        overrides = set(spec.overrides)
        required = {
            "training.loss.type=cumulative_log_change_mae",
            "training.loss.bps_scale=10000.0",
            "training.selection_metric=validation_loss",
            "training.graph_learning_rate=0.001",
            "model.temporal.type=modern_tcn",
            "model.temporal.session_position_encoding=false",
            "model.graph.type=free_static",
            "model.graph.num_heads=1",
            "model.spatial.gate_type=learned_scalar",
            "model.spatial.initial_beta=0.1",
        }
        if not required.issubset(overrides):
            raise AssertionError(
                f"Stage-1 contract missing from {spec.run_name}: "
                f"{sorted(required - overrides)}"
            )
        if spec.graph_learning_rate != stage1_graph_lr:
            raise AssertionError("Stage-1 graph LR metadata is incorrect.")
        if not spec.trainable_graph or not spec.graph_enabled:
            raise AssertionError("Stage 1 must use a trainable graph.")

    # Any two Stage-1 candidates can seed the Stage-2 constructor.
    stage2 = make_stage2_specs(
        stage1[:2],
        target_entropy_coefficient=1.0,
    )
    if len(stage2) != 16:
        raise AssertionError("Stage 2 does not contain 16 runs.")

    graph_variants = {spec.graph_variant for spec in stage2}
    expected_variants = {
        "no_graph",
        "corr120",
        "corr180",
        "corr240",
        "free_static_g2",
        "free_static_h22_lam1",
        "dynamic_g1_h32",
        "dynamic_base_g1_h32_a025",
    }
    if graph_variants != expected_variants:
        raise AssertionError(
            f"Unexpected Stage-2 variants: {sorted(graph_variants)}"
        )

    correlation_specs = [
        spec for spec in stage2 if spec.graph_variant.startswith("corr")
    ]
    for spec in correlation_specs:
        if (
            "model.fixed_graph_resource.empty_row_policy=strongest"
            not in spec.overrides
        ):
            raise AssertionError(
                "Correlation sweep must define a deterministic empty-row "
                "fallback so an overnight stage cannot fail on one threshold."
            )
        if spec.trainable_graph or spec.graph_learning_rate is not None:
            raise AssertionError(
                "Fixed correlation graphs must not be marked trainable."
            )

    eligible = [spec for spec in stage2 if spec.eligible_for_final]
    if len(eligible) != 14:
        raise AssertionError(
            "Exactly the two no-graph controls must be excluded from final "
            "graph-model selection."
        )

    # Learned-graph Stage 3: backbone LR x graph LR, with winner reused.
    learned_winner = next(spec for spec in eligible if spec.trainable_graph)
    learned_stage3 = make_stage3_specs(learned_winner)
    if len(learned_stage3) != 8:
        raise AssertionError("Learned-graph Stage 3 does not contain 8 runs.")
    if {spec.graph_learning_rate for spec in learned_stage3} != set(
        STAGE3_GRAPH_LEARNING_RATES
    ):
        raise AssertionError("Stage 3 does not sweep all graph learning rates.")
    if any(not spec.trainable_graph for spec in learned_stage3):
        raise AssertionError("Learned-graph refinements lost trainable graph metadata.")
    assert_total_budget(stage1, stage2, learned_stage3)

    # Fixed-graph Stage 3: backbone LR x FFN ratio, with winner reused.
    fixed_winner = next(
        spec for spec in eligible if spec.graph_variant.startswith("corr")
    )
    fixed_stage3 = make_stage3_specs(fixed_winner)
    if len(fixed_stage3) != 8:
        raise AssertionError("Fixed-graph Stage 3 does not contain 8 runs.")
    if any(spec.trainable_graph for spec in fixed_stage3):
        raise AssertionError("Fixed-graph refinements became trainable graphs.")
    if {spec.ffn_ratio for spec in fixed_stage3} != {1, 2, 4}:
        raise AssertionError("Fixed-graph Stage 3 does not sweep FFN ratios.")
    assert_total_budget(stage1, stage2, fixed_stage3)

    # Exercise result loading and automatic graph-required selection.
    with TemporaryDirectory() as temporary:
        output_root = Path(temporary)
        no_graph = next(spec for spec in stage2 if not spec.graph_enabled)
        graph_a = eligible[0]
        graph_b = eligible[1]
        _write_completed_run(
            output_root,
            spec=no_graph,
            score=0.90,
            parameters=100,
        )
        _write_completed_run(
            output_root,
            spec=graph_a,
            score=1.00,
            parameters=200,
        )
        _write_completed_run(
            output_root,
            spec=graph_b,
            score=1.01,
            parameters=150,
        )
        candidates = [no_graph, graph_a, graph_b]
        frame = summarise_specs(output_root, candidates, require_all=True)
        lookup = {spec.run_name: spec for spec in candidates}
        selected = select_specs(
            frame,
            lookup,
            count=1,
            eligible_only=True,
            relative_tie_tolerance=0.0,
        )
        if selected[0].run_name != graph_a.run_name:
            raise AssertionError(
                "Automatic selection did not choose the lowest-scoring "
                "graph-enabled candidate."
            )
        if "Graph update norm" not in frame.columns:
            raise AssertionError("New graph optimisation diagnostics are absent.")

    print("ModernTCN graph sweep contract tests passed.")


if __name__ == "__main__":
    main()

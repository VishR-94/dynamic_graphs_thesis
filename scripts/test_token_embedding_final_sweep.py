from __future__ import annotations

"""Contracts for the final six-fit token-embedding experiment."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.training.token_embedding_final_sweep import (
    BSQ_CONTROL_PRESET,
    BSQ_CONTROL_RUN_NAME,
    EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME,
    clone_with_close_log_variance_feature,
    clone_with_close_scale_features,
    load_temperature_result,
    make_architecture_specs,
    make_bsq_control_spec,
    select_screening_specs,
)


def _override_value(spec, key: str) -> str:
    prefix = f"{key}="
    matches = [value[len(prefix) :] for value in spec.overrides if value.startswith(prefix)]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one override for {key!r} in {spec.run_name}; "
            f"found {len(matches)}."
        )
    return matches[0]


def test_six_architecture_grid() -> None:
    specs = make_architecture_specs()
    assert len(specs) == 6
    assert len({spec.run_name for spec in specs}) == 6

    modern = [spec for spec in specs if spec.temporal_family == "modern_tcn"]
    transformers = [spec for spec in specs if spec.temporal_family == "transformer"]
    assert len(modern) == 2
    assert len(transformers) == 4
    assert {spec.graph_type for spec in modern} == {"dynamic", "free_static"}
    assert {(spec.graph_type, spec.num_st_blocks) for spec in transformers} == {
        ("dynamic", 1),
        ("free_static", 1),
        ("dynamic", 3),
        ("free_static", 3),
    }

    for spec in specs:
        assert spec.token_input_representation == "hierarchical_embedding"
        assert _override_value(
            spec, "models.dynamic_graph.future_predictor.num_layers"
        ) == "1"
        assert _override_value(
            spec, "training.early_stopping_metric"
        ) == "validation_token_loss"
        assert _override_value(
            spec, "models.dynamic_graph.heads.future_token_mode"
        ) == "coarse_only"
        assert _override_value(
            spec, "models.dynamic_graph.close_scale_features.enabled"
        ) == "false"

    for spec in modern:
        assert spec.num_st_blocks == 1
        assert _override_value(spec, "models.dynamic_graph.d_model") == "32"
        assert _override_value(
            spec, "models.dynamic_graph.temporal.modern_tcn.num_blocks"
        ) == "1"
        assert _override_value(
            spec, "models.dynamic_graph.temporal.modern_tcn.patch_size"
        ) == "8"
        assert _override_value(
            spec, "models.dynamic_graph.temporal.modern_tcn.patch_stride"
        ) == "4"
        assert _override_value(
            spec, "models.dynamic_graph.temporal.modern_tcn.large_kernel"
        ) == "15"

    for spec in transformers:
        assert _override_value(spec, "models.dynamic_graph.d_model") == "96"
        assert _override_value(
            spec, "models.dynamic_graph.temporal.num_heads"
        ) == "8"
        assert int(
            _override_value(spec, "models.dynamic_graph.num_st_blocks")
        ) in {1, 3}


def test_screening_selection_contract() -> None:
    specs = make_architecture_specs()
    rows = []
    for index, spec in enumerate(specs):
        score = 5.0 + index
        if spec.run_name == EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME:
            score = 99.0
        rows.append(
            {
                "Run": spec.run_name,
                "Best validation CE": score,
            }
        )
    results = pd.DataFrame(rows)
    selected = select_screening_specs(results, specs)
    assert len(selected) == 3
    assert EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME == selected[-1].run_name
    assert len({spec.run_name for spec in selected}) == 3

    # When dynamic ModernTCN is already top two it must not be duplicated.
    reordered = results.copy()
    reordered.loc[
        reordered["Run"] == EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME,
        "Best validation CE",
    ] = 0.0
    selected = select_screening_specs(reordered, specs)
    assert len(selected) == 2
    assert len({spec.run_name for spec in selected}) == 2


def test_bsq_control_and_scale_clone() -> None:
    control = make_bsq_control_spec()
    assert control.run_name == BSQ_CONTROL_RUN_NAME
    assert control.preset == BSQ_CONTROL_PRESET
    assert control.token_input_representation == "bsq_bits"
    assert control.overrides == ()

    scale = clone_with_close_log_variance_feature(
        control,
        run_name="bsq_control_close_log_variance",
    )
    assert scale.preset == control.preset
    assert scale.token_input_representation == "bsq_bits"
    assert scale.label.endswith("Close log-variance feature")
    assert (
        "models.dynamic_graph.close_scale_features.enabled=true"
        in scale.overrides
    )
    legacy_alias = clone_with_close_scale_features(
        control,
        run_name="bsq_control_close_log_variance_alias",
    )
    assert legacy_alias.label.endswith("Close log-variance feature")


def test_temperature_result_loader_contract() -> None:
    with TemporaryDirectory() as directory:
        run_dir = Path(directory) / "run"
        policy_dir = run_dir / "temperature_sweep" / "temperature_1"
        policy_dir.mkdir(parents=True)
        record = {
            "request": {
                "temperature": 1.0,
                "sample_count": 10,
                "top_p": 0.9,
                "top_k": 0,
                "sampling_seed": 42,
                "checkpoint_epoch": 7,
            },
            "result": {
                "Policy": "temperature_1",
                "Temperature": 1.0,
                "Sample count": 10,
                "Mean Log MAE": 0.0015,
            },
        }
        (policy_dir / "temperature_result.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        (policy_dir / "validation_sampled_price_paths.pt").write_bytes(b"x")
        (policy_dir / "validation_metric_table.csv").write_text(
            "metric,horizon,channel,value\n", encoding="utf-8"
        )
        loaded = load_temperature_result(run_dir, temperature=1.0)
        assert loaded["Mean Log MAE"] == 0.0015
        assert loaded["Sampled path file"].endswith(
            "validation_sampled_price_paths.pt"
        )


def main() -> None:
    test_six_architecture_grid()
    test_screening_selection_contract()
    test_bsq_control_and_scale_clone()
    test_temperature_result_loader_contract()
    print("Final token-embedding sweep contracts passed.")


if __name__ == "__main__":
    main()

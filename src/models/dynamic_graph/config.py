from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import load_yaml

from .contracts import (
    BackcastConfig,
    CloseScaleFeatureConfig,
    DynamicGraphModelConfig,
    ForecastHeadConfig,
    FuturePredictorConfig,
    GraphConfig,
    SpatialConfig,
    TemporalConfig,
    TokenLossConfig,
)


ConfigDict = dict[str, Any]


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> ConfigDict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged: ConfigDict = deepcopy(dict(base))

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(
                merged[key],
                value,
            )
        else:
            merged[key] = deepcopy(value)

    return merged


def load_dynamic_graph_config(
    path: str | Path = "configs/dynamic_graph.yaml",
    *,
    preset: str | None = None,
) -> ConfigDict:
    """Load and resolve one dynamic-graph experiment preset.

    The YAML contains one common base configuration and small named
    overrides. This keeps the three initial predictor-selection runs
    identical except for:

        - future-predictor type;
        - future-position loss weighting.
    """
    raw = load_yaml(path)

    default_preset = raw.get(
        "default_preset"
    )

    resolved_preset = (
        default_preset
        if preset is None
        else preset
    )

    if not isinstance(
        resolved_preset,
        str,
    ):
        raise ValueError(
            "A string preset must be supplied either explicitly "
            "or through default_preset."
        )

    presets = raw.get(
        "presets"
    )

    if not isinstance(
        presets,
        Mapping,
    ):
        raise ValueError(
            "The dynamic-graph YAML must contain a presets mapping."
        )

    if resolved_preset not in presets:
        raise KeyError(
            f"Unknown dynamic-graph preset {resolved_preset!r}. "
            f"Expected one of {sorted(presets)}."
        )

    base = {
        key: value
        for key, value in raw.items()
        if key not in {
            "default_preset",
            "presets",
        }
    }

    resolved = _deep_merge(
        base,
        presets[resolved_preset],
    )

    resolved["resolved_preset"] = (
        resolved_preset
    )

    validate_dynamic_graph_config(
        resolved
    )

    return resolved


def build_model_config(
    experiment_config: Mapping[str, Any],
) -> DynamicGraphModelConfig:
    """Convert the resolved YAML model section into typed contracts."""
    try:
        model = experiment_config[
            "models"
        ][
            "dynamic_graph"
        ]
    except KeyError as error:
        raise KeyError(
            "Config must contain models.dynamic_graph."
        ) from error

    temporal_values = dict(
        model["temporal"]
    )
    temporal_values["dilations"] = tuple(
        int(value)
        for value in temporal_values[
            "dilations"
        ]
    )

    # Keep the YAML readable while retaining one immutable typed temporal
    # config. Historical configs do not contain this nested mapping and
    # therefore continue to use the dataclass defaults.
    modern_tcn_values = dict(
        temporal_values.pop(
            "modern_tcn",
            {},
        )
    )
    modern_tcn_key_map = {
        "patch_size": "modern_tcn_patch_size",
        "patch_stride": "modern_tcn_patch_stride",
        "ffn_ratio": "modern_tcn_ffn_ratio",
        "num_blocks": "modern_tcn_num_blocks",
        "large_kernel": "modern_tcn_large_kernel",
        "small_kernel": "modern_tcn_small_kernel",
        "dropout": "modern_tcn_dropout",
    }
    unknown_modern_tcn_keys = (
        set(modern_tcn_values)
        - set(modern_tcn_key_map)
    )
    if unknown_modern_tcn_keys:
        raise KeyError(
            "Unknown temporal.modern_tcn keys: "
            f"{sorted(unknown_modern_tcn_keys)}."
        )
    for yaml_key, dataclass_key in modern_tcn_key_map.items():
        if yaml_key in modern_tcn_values:
            temporal_values[dataclass_key] = modern_tcn_values[yaml_key]

    head_values = dict(
        model["heads"]
    )
    head_values[
        "evaluation_horizons"
    ] = tuple(
        int(value)
        for value in head_values[
            "evaluation_horizons"
        ]
    )

    loss_values = dict(model["loss"])

    # Old run artefacts may contain the unused Gaussian-envelope
    # parameters. Drop them so historical uniform-loss checkpoints and
    # diagnostics remain loadable after the weighted loss was replaced.
    loss_values.pop("gaussian_sigma", None)
    loss_values.pop("gaussian_peak_mass", None)

    config = DynamicGraphModelConfig(
        num_nodes=int(
            model["num_nodes"]
        ),
        context_length=int(
            model["context_length"]
        ),
        d_model=int(
            model["d_model"]
        ),
        num_st_blocks=int(
            model["num_st_blocks"]
        ),
        use_node_embedding=bool(
            model["use_node_embedding"]
        ),
        token_input_representation=str(
            model.get(
                "token_input_representation",
                "hierarchical_embedding",
            )
        ),
        temporal=TemporalConfig(
            **temporal_values
        ),
        graph=GraphConfig(
            **dict(model["graph"])
        ),
        spatial=SpatialConfig(
            **dict(
                model.get(
                    "spatial",
                    {},
                )
            )
        ),
        close_scale_features=CloseScaleFeatureConfig(
            **dict(
                model.get(
                    "close_scale_features",
                    {},
                )
            )
        ),
        heads=ForecastHeadConfig(
            **head_values
        ),
        future_predictor=(
            FuturePredictorConfig(
                **dict(
                    model[
                        "future_predictor"
                    ]
                )
            )
        ),
        loss=TokenLossConfig(
            **loss_values
        ),
        backcast=BackcastConfig(
            **dict(model["backcast"])
        ),
    )

    config.validate()
    return config


def build_dense_window_config(
    base_forecasting_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> ConfigDict:
    """Create the model-specific WindowedCandleDataset configuration.

    The global baseline configuration remains unchanged. This helper
    copies it and overrides only the window contract required by the
    final model:

        context: 60 bars
        targets:  every minute 1..60
    """
    resolved = deepcopy(
        dict(base_forecasting_config)
    )

    forecasting = experiment_config[
        "forecasting"
    ]

    prediction_length = int(
        forecasting[
            "prediction_length"
        ]
    )

    resolved["forecasting"] = {
        "context_length": int(
            forecasting[
                "context_length"
            ]
        ),
        "stride": int(
            forecasting[
                "stride"
            ]
        ),
        "horizons": list(
            range(
                1,
                prediction_length + 1,
            )
        ),
        "input_channels": list(
            forecasting[
                "input_channels"
            ]
        ),
        "target_channels": list(
            forecasting[
                "target_channels"
            ]
        ),
    }

    return resolved


def validate_dynamic_graph_config(
    experiment_config: Mapping[str, Any],
) -> None:
    """Validate cross-section consistency beyond dataclass fields."""
    model = build_model_config(
        experiment_config
    )

    forecasting = experiment_config.get(
        "forecasting"
    )

    if not isinstance(
        forecasting,
        Mapping,
    ):
        raise ValueError(
            "Config must contain a forecasting mapping."
        )

    if int(
        forecasting[
            "context_length"
        ]
    ) != model.context_length:
        raise ValueError(
            "forecasting.context_length and model.context_length "
            "must match."
        )

    if int(
        forecasting[
            "prediction_length"
        ]
    ) != model.prediction_length:
        raise ValueError(
            "forecasting.prediction_length and "
            "heads.prediction_length must match."
        )

    evaluation_horizons = tuple(
        int(value)
        for value in forecasting[
            "evaluation_horizons"
        ]
    )

    if evaluation_horizons != (
        model.heads.evaluation_horizons
    ):
        raise ValueError(
            "Forecasting and model evaluation horizons differ."
        )

    data = experiment_config.get(
        "data"
    )

    if not isinstance(
        data,
        Mapping,
    ):
        raise ValueError(
            "Config must contain a data mapping."
        )

    if data.get(
        "amount_mode"
    ) != "zero":
        raise ValueError(
            "The current project contract requires "
            "data.amount_mode='zero'."
        )

    normalisation = experiment_config.get(
        "normalisation"
    )

    if not isinstance(
        normalisation,
        Mapping,
    ):
        raise ValueError(
            "Config must contain a normalisation mapping."
        )

    if normalisation.get(
        "stats_from"
    ) != "context":
        raise ValueError(
            "Kronos forecasting statistics must come from the "
            "observed context only."
        )

    if not bool(
        normalisation.get(
            "clip"
        )
    ):
        raise ValueError(
            "The Kronos token path must be clipped to [-5, 5]."
        )

    training = experiment_config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Config must contain a training mapping.")
    if str(training.get("optimizer", "adamw")) not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'.")
    if str(training.get("scheduler", "none")) not in {
        "none",
        "modern_tcn_type3",
    }:
        raise ValueError(
            "training.scheduler must be 'none' or 'modern_tcn_type3'."
        )
    if float(training.get("graph_learning_rate", training["learning_rate"])) <= 0:
        raise ValueError("training.graph_learning_rate must be positive.")

    decoding = experiment_config.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ValueError("Config must contain a decoding mapping.")
    if int(decoding.get("sample_count", 1)) <= 0:
        raise ValueError("decoding.sample_count must be positive.")

    temperature_sweep = experiment_config.get("temperature_sweep")
    if temperature_sweep is not None:
        if not isinstance(temperature_sweep, Mapping):
            raise ValueError("temperature_sweep must be a mapping.")
        temperatures = temperature_sweep.get("temperatures")
        if not isinstance(temperatures, (list, tuple)) or not temperatures:
            raise ValueError(
                "temperature_sweep.temperatures must be a non-empty list."
            )
        if any(float(value) <= 0.0 for value in temperatures):
            raise ValueError(
                "Every temperature_sweep temperature must be positive."
            )
        if int(temperature_sweep.get("sample_count", 10)) <= 0:
            raise ValueError(
                "temperature_sweep.sample_count must be positive."
            )
        top_p = float(temperature_sweep.get("top_p", 0.9))
        if not 0.0 < top_p <= 1.0:
            raise ValueError(
                "temperature_sweep.top_p must lie in (0, 1]."
            )


def _cpu_smoke_test() -> None:
    config_path = Path(
        "configs/dynamic_graph.yaml"
    )

    raw = load_yaml(
        config_path
    )

    presets = raw.get(
        "presets"
    )

    if not isinstance(
        presets,
        Mapping,
    ):
        raise ValueError(
            "The dynamic-graph YAML must contain a presets mapping."
        )

    preset_names = tuple(
        str(name)
        for name in presets
    )

    required_presets = {
        "structured_parallel_uniform",
        "structured_parallel_weighted",
        "autoregressive_uniform",
        "structured_parallel_coarse_only",
        "free_static_full_true_s1",
        "free_static_coarse_only",
        "free_static_full_predicted_s1",
        "modern_tcn_dynamic_coarse_mc10",
    }

    missing_presets = (
        required_presets
        - set(preset_names)
    )

    if missing_presets:
        raise AssertionError(
            "Required dynamic-graph presets are missing: "
            f"{sorted(missing_presets)}."
        )

    resolved = {
        preset: load_dynamic_graph_config(
            config_path,
            preset=preset,
        )
        for preset in preset_names
    }

    typed = {
        preset: build_model_config(
            values
        )
        for preset, values in resolved.items()
    }

    parallel_uniform = typed[
        "structured_parallel_uniform"
    ]
    parallel_weighted = typed[
        "structured_parallel_weighted"
    ]
    autoregressive = typed[
        "autoregressive_uniform"
    ]
    legacy_coarse_only = typed[
        "structured_parallel_coarse_only"
    ]
    free_static_reference = typed[
        "free_static_full_true_s1"
    ]
    free_static_coarse = typed[
        "free_static_coarse_only"
    ]
    free_static_predicted_s1 = typed[
        "free_static_full_predicted_s1"
    ]

    modern_tcn_token = typed[
        "modern_tcn_dynamic_coarse_mc10"
    ]
    embedded_token = typed[
        "hierarchical_embedding_coarse_ce"
    ]

    assert (
        parallel_uniform
        .future_predictor
        .type
        == "structured_parallel"
    )
    assert (
        parallel_uniform
        .loss
        .horizon_weighting
        == "uniform"
    )

    assert (
        parallel_weighted
        .future_predictor
        .type
        == "structured_parallel"
    )
    assert (
        parallel_weighted
        .loss
        .horizon_weighting
        == "exponential_decay"
    )

    assert (
        autoregressive
        .future_predictor
        .type
        == "autoregressive"
    )
    assert (
        autoregressive
        .loss
        .horizon_weighting
        == "uniform"
    )

    assert (
        legacy_coarse_only
        .heads
        .future_token_mode
        == "coarse_only"
    )
    assert not legacy_coarse_only.heads.predicts_s2
    assert (
        legacy_coarse_only
        .heads
        .s2_loss_weight
        == 0.0
    )

    assert (
        free_static_reference
        .graph
        .type
        == "free_static"
    )
    assert (
        free_static_reference
        .heads
        .future_token_mode
        == "full"
    )
    assert (
        free_static_reference
        .heads
        .s2_conditioning
        == "true_s1"
    )
    assert (
        free_static_reference
        .heads
        .s2_loss_weight
        == 1.0
    )

    assert (
        free_static_coarse
        .graph
        .type
        == "free_static"
    )
    assert (
        free_static_coarse
        .heads
        .future_token_mode
        == "coarse_only"
    )
    assert not free_static_coarse.heads.predicts_s2
    assert (
        free_static_coarse
        .heads
        .s2_loss_weight
        == 0.0
    )

    assert (
        free_static_predicted_s1
        .graph
        .type
        == "free_static"
    )
    assert (
        free_static_predicted_s1
        .heads
        .future_token_mode
        == "full"
    )
    assert (
        free_static_predicted_s1
        .heads
        .s2_conditioning
        == "predicted_s1"
    )
    assert (
        free_static_predicted_s1
        .heads
        .s2_loss_weight
        == 1.0
    )

    for model in typed.values():
        assert (
            model.prediction_length
            == 60
        )
        assert model.evaluation_indices == (
            0,
            4,
            14,
            29,
            59,
        )

    for preset_name in (
        "free_static_full_true_s1",
        "free_static_coarse_only",
        "free_static_full_predicted_s1",
    ):
        graph_regularisation = resolved[
            preset_name
        ][
            "models"
        ][
            "dynamic_graph"
        ][
            "graph_regularisation"
        ]

        assert (
            float(
                graph_regularisation[
                    "graph_entropy_reg"
                ]
            )
            == 0.0
        )
        assert (
            float(
                graph_regularisation[
                    "graph_target_entropy"
                ]
            )
            == 2.2
        )
        assert (
            float(
                graph_regularisation[
                    "graph_target_entropy_reg"
                ]
            )
            == 0.01
        )
        assert (
            float(
                graph_regularisation[
                    "graph_temporal_smooth_reg"
                ]
            )
            == 0.0
        )



    assert embedded_token.token_input_representation == (
        "hierarchical_embedding"
    )
    assert embedded_token.heads.future_token_mode == "coarse_only"
    assert embedded_token.future_predictor.type == "structured_parallel"
    assert embedded_token.future_predictor.num_layers == 1
    assert not embedded_token.close_scale_features.enabled

    assert modern_tcn_token.temporal.type == "modern_tcn"
    assert modern_tcn_token.token_input_representation == "bsq_bits"
    assert modern_tcn_token.d_model == 32
    assert modern_tcn_token.temporal.modern_tcn_patch_size == 8
    assert modern_tcn_token.temporal.modern_tcn_patch_stride == 4
    assert modern_tcn_token.temporal.modern_tcn_large_kernel == 15
    assert modern_tcn_token.temporal.modern_tcn_num_blocks == 1
    assert modern_tcn_token.graph.type == "dynamic"
    assert modern_tcn_token.graph.num_heads == 1
    assert modern_tcn_token.graph.hidden_dim == 32
    assert modern_tcn_token.spatial.gate_type == "learned_scalar"
    assert modern_tcn_token.spatial.initial_beta == 0.5
    assert modern_tcn_token.heads.future_token_mode == "coarse_only"
    assert modern_tcn_token.heads.s1_vocabulary_size == 1024
    assert modern_tcn_token.temporal_output_length == 15

    print(
        "Dynamic-graph configuration smoke test passed."
    )

    for preset, model in typed.items():
        print(
            f"  {preset}: "
            f"predictor={model.future_predictor.type}, "
            f"loss={model.loss.horizon_weighting}, "
            f"token_mode={model.heads.future_token_mode}, "
            f"s2_conditioning={model.heads.s2_conditioning}, "
            f"graph={model.graph.type}"
        )


if __name__ == "__main__":
    _cpu_smoke_test()


from __future__ import annotations

"""Adapters from the frozen financial architectures to the weather task."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.models.continuous_forecaster import (
    ContinuousForecasterConfig,
    ContinuousTemporalConfig,
)
from src.models.dense_transformer_depth_sweep import (
    DenseTransformerDepthConfig,
    StackedDenseTransformerGraphModel,
)
from src.models.dynamic_graph.contracts import GraphConfig
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Config,
    ModernTCNGraphRound1Model,
)

from .config import WEATHER_FEATURES, WEATHER_NODES, WeatherRunConfig
from .data import SonnetWeatherDataBundle


@dataclass
class WeatherModelBundle:
    model: nn.Module
    model_config: dict[str, Any]
    initial_graph_payload: dict[str, Any]


def _dense_forecast_steps(horizon: int) -> tuple[int, ...]:
    return tuple(range(1, int(horizon) + 1))


def build_modern_tcn_weather_model(
    config: WeatherRunConfig,
    data: SonnetWeatherDataBundle,
) -> WeatherModelBundle:
    """Build the frozen 1ST ModernTCN static+dynamic correlation-prior model."""

    temporal = ContinuousTemporalConfig(
        type="modern_tcn",
        d_model=32,
        num_layers=1,
        num_heads=4,
        feedforward_multiplier=2,
        dropout=0.05,
        relative_position_embedding=False,
        session_position_encoding=False,
        patch_size=8,
        patch_stride=4,
        modern_tcn_ffn_ratio=1,
        modern_tcn_num_blocks=1,
        modern_tcn_large_kernel=15,
        modern_tcn_small_kernel=5,
        modern_tcn_dropout=0.05,
        modern_tcn_head_dropout=0.0,
    )
    forecaster = ContinuousForecasterConfig(
        num_nodes=len(WEATHER_NODES),
        context_length=int(config.context_length),
        horizons=_dense_forecast_steps(config.horizon),
        input_channels=tuple(WEATHER_FEATURES),
        target_channel="t850",
        output_representation="normalised_close",
        output_head_initialisation="default",
        temporal=temporal,
        graph=GraphConfig(
            type="dynamic",
            num_heads=1,
            hidden_dim=32,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=min(4, len(WEATHER_NODES) - 1),
            base_graph_type="free_static",
            gate_type="learned_scalar",
            initial_alpha=0.5,
        ),
        spatial_num_layers=1,
        spatial_feedforward_multiplier=2,
        spatial_dropout=0.0,
        spatial_gate_type="learned_scalar",
        spatial_gate_initial_beta=0.5,
        head_dropout=0.0,
    )
    model_config = ModernTCNGraphRound1Config(
        forecaster=forecaster,
        graph_variant="prior_mixture_state",
        prior_scale=float(config.prior_scale),
        prior_jitter=float(config.prior_jitter),
        prior_seed=int(config.prior_seed),
    )
    prior = torch.from_numpy(data.row_normalised_correlation_prior).float()
    model = ModernTCNGraphRound1Model(
        model_config,
        static_prior=prior,
    )
    initial_static = model.graph_learner.static_adjacency()
    if initial_static is None:
        raise RuntimeError("ModernTCN weather model failed to construct its static graph.")
    payload = {
        "model_kind": config.model_kind,
        "prior_type": "correlation_of_t850_first_differences",
        "graph_orientation": "A[target, source]",
        "node_order": list(WEATHER_NODES),
        "source_abs_correlation": torch.from_numpy(
            data.source_abs_correlation
        ).float(),
        "row_normalised_source_prior": prior,
        "initial_static_adjacency": initial_static.detach().cpu().float(),
        "prior_scale": float(config.prior_scale),
        "prior_jitter": float(config.prior_jitter),
        "prior_seed": int(config.prior_seed),
        "initial_alpha": float(model.alpha().detach().item()),
        "initial_beta": float(model.beta().detach().item()),
    }
    return WeatherModelBundle(
        model=model,
        model_config={
            "family": "1ST_block_ModernTCN",
            "variant": "prior_mixture_state",
            "num_nodes": len(WEATHER_NODES),
            "context_length": int(config.context_length),
            "forecast_length": int(config.horizon),
            "input_channels": list(WEATHER_FEATURES),
            "target": "t850_all_nodes",
            "temporal": {
                "type": "modern_tcn",
                "d_model": 32,
                "num_blocks": 1,
                "patch_size": 8,
                "patch_stride": 4,
                "large_kernel": 15,
                "small_kernel": 5,
                "ffn_ratio": 1,
                "dropout": 0.05,
                "head_dropout": 0.0,
                "session_position_encoding": False,
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "activation": "softmax",
                "num_heads": 1,
                "hidden_dim": 32,
                "initial_alpha": 0.5,
                "static_initialisation": "correlation_prior",
                "prior_scale": float(config.prior_scale),
                "prior_jitter": float(config.prior_jitter),
                "prior_seed": int(config.prior_seed),
            },
            "spatial": {
                "state_aware": True,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "initial_beta": 0.5,
            },
        },
        initial_graph_payload=payload,
    )


def build_transformer_3st_weather_model(
    config: WeatherRunConfig,
) -> WeatherModelBundle:
    """Build the frozen price-space 3ST stacked Transformer architecture."""

    model_config = DenseTransformerDepthConfig(
        num_nodes=len(WEATHER_NODES),
        context_length=int(config.context_length),
        horizons=_dense_forecast_steps(config.horizon),
        input_channels=tuple(WEATHER_FEATURES),
        target_channel="t850",
        num_st_blocks=3,
        d_model=64,
        transformer_num_layers=1,
        transformer_num_heads=4,
        transformer_feedforward_multiplier=2,
        transformer_dropout=0.0,
        position_embedding=False,
        graph_heads_per_block=(1, 1, 1),
        graph_hidden_dims_per_block=(64, 64, 64),
        graph_activations_per_block=("softmax", "softmax", "sparsemax"),
        graph_initial_alpha=0.5,
        spatial_initial_beta=0.5,
        spatial_feedforward_multiplier=2,
        spatial_dropout=0.0,
    )
    model = StackedDenseTransformerGraphModel(model_config)
    initial_bases = model.initial_base_graphs()
    payload = {
        "model_kind": config.model_kind,
        "prior_type": "uniform",
        "graph_orientation": "A[target, source]",
        "node_order": list(WEATHER_NODES),
        "initial_base_graphs_per_layer": initial_bases,
        "graph_activations_per_layer": ["softmax", "softmax", "sparsemax"],
        "initial_alpha_per_layer": [
            float(value.detach().item()) for value in model.alphas()
        ],
        "initial_beta_per_layer": [
            float(value.detach().item()) for value in model.betas()
        ],
    }
    return WeatherModelBundle(
        model=model,
        model_config={
            "family": "3ST_block_stacked_Transformer",
            "variant": "uniform_static_dynamic_state",
            "num_nodes": len(WEATHER_NODES),
            "context_length": int(config.context_length),
            "forecast_length": int(config.horizon),
            "input_channels": list(WEATHER_FEATURES),
            "target": "t850_all_nodes",
            "training_style": "dense_prefix",
            "temporal": {
                "type": "transformer",
                "num_st_blocks": 3,
                "d_model": 64,
                "layers_per_block": 1,
                "num_heads": 4,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "position_embedding": False,
            },
            "graph": {
                "type": "static_dynamic_mixture",
                "heads_per_block": [1, 1, 1],
                "hidden_dims_per_block": [64, 64, 64],
                "activations_per_block": ["softmax", "softmax", "sparsemax"],
                "initial_alpha": 0.5,
                "static_initialisation": "uniform",
                "dynamic_initialisation": "Q Xavier, K zero",
            },
            "spatial": {
                "state_aware": True,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "initial_beta": 0.5,
            },
        },
        initial_graph_payload=payload,
    )


def build_weather_model(
    config: WeatherRunConfig,
    data: SonnetWeatherDataBundle,
) -> WeatherModelBundle:
    if config.model_kind == "modern_tcn_1st":
        return build_modern_tcn_weather_model(config, data)
    if config.model_kind == "transformer_3st":
        return build_transformer_3st_weather_model(config)
    raise ValueError(f"Unsupported model_kind {config.model_kind!r}.")


def graph_parameter_ids(model: nn.Module) -> set[int]:
    method = getattr(model, "graph_parameter_ids", None)
    if method is None:
        raise TypeError("Weather model does not expose graph_parameter_ids().")
    return set(int(value) for value in method())


def parameter_counts(model: nn.Module) -> dict[str, int]:
    graph_ids = graph_parameter_ids(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    graph = [parameter for parameter in trainable if id(parameter) in graph_ids]
    backbone = [parameter for parameter in trainable if id(parameter) not in graph_ids]
    if len(trainable) != len(graph) + len(backbone):
        raise AssertionError("Parameter partition lost trainable parameters.")
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "graph_group_parameters": int(sum(parameter.numel() for parameter in graph)),
        "backbone_group_parameters": int(sum(parameter.numel() for parameter in backbone)),
    }


def model_alphas(model: nn.Module) -> tuple[Tensor, ...]:
    if isinstance(model, ModernTCNGraphRound1Model):
        alpha = model.alpha()
        return () if alpha is None else (alpha,)
    if isinstance(model, StackedDenseTransformerGraphModel):
        return model.alphas()
    raise TypeError(type(model))


def model_betas(model: nn.Module) -> tuple[Tensor, ...]:
    if isinstance(model, ModernTCNGraphRound1Model):
        return (model.beta(),)
    if isinstance(model, StackedDenseTransformerGraphModel):
        return model.betas()
    raise TypeError(type(model))

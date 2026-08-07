from __future__ import annotations

"""Adapter for the exact BaseDyGraph-V2 snapshot used by Dimitri.

The supplied source is vendored byte-for-byte under
``external/DimitriBaseDyGraphV2``.  It intentionally retains its original
module-level imports (``model``, ``modules``, ``utilities`` and
``data_module``), so this adapter loads it only in the dedicated replication
process and verifies every source hash first.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import importlib
import json
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMITRI_SOURCE_ROOT = PROJECT_ROOT / "external" / "DimitriBaseDyGraphV2" / "src"

DIMITRI_SOURCE_HASHES = {
    "model.py": "b99256db74b84f57513b12715a9ed1f4fc735202bcf933482e3b66ac9cf119d5",
    "modules.py": "1bd31701b300f6f805dfc53530c66eedd9265620093223d3302e3e74409b51ff",
    "utilities.py": "cfe849e2963386ddaab15ad7c89e7df92fc957f22c15f45ea40c89ec4c82f40a",
    "data_module.py": "ecfbe768a1e2c0840043ecf640295b425106843f27062184307534312136cac1",
    "sector_prior.py": "7ecd77eb38865acce0dea2d8986d2b0201f4bf9be967845ab1bebd9168fe5919",
}

DIMITRI_X0_CHECKPOINT_SHA256 = (
    "59d3a87781e9156e3fc6fbcbc3a9dc9438589beff4703eae7097ef80240a9399"
)
DIMITRI_X0_EXPECTED_PARAMETER_COUNT = 1_127_121
DIMITRI_X0_EXPECTED_VALIDATION_ACCURACY = 0.16289882361888885
DIMITRI_X0_EXPECTED_CHECKPOINT_EPOCH = 12
DIMITRI_X0_EXPECTED_GLOBAL_STEP = 17_017
DIMITRI_X0_EXPECTED_TRAIN_WINDOWS = 1_309
# 17,017 global steps / 13 completed epochs = 1,309 steps per epoch, proving
# that the supplied x0 checkpoint was trained with one window per batch.
DIMITRI_X0_INFERRED_BATCH_SIZE = 1

# Exact cfg_dict stored in the supplied x0jhc0tx Lightning checkpoint.
DIMITRI_X0_CONFIG: dict[str, Any] = {
    "num_states": 1024,
    "num_nodes": 93,
    "d_model": 96,
    "nhead": 4,
    "num_temporal_layers": 1,
    "num_spatial_layers": 1,
    "dropout": 0.0,
    "ff_mult": 1,
    "max_seq_len": 512,
    "num_edge_heads": 1,
    "graph_hidden_dim": 96,
    "spatial_dropout": 0.1,
    "use_node_embedding": True,
    "use_state_pair_bias": False,
    "add_self_loops": False,
    "symmetric_graph": False,
    "predict_next_state": True,
    "temporal_module_type": "transformer",
    "temporal_context_window": 180,
    "spatial_module_type": "dual_fusion",
    "spatial_value": "concat",
    "scorer_value": "concat",
    "graph_activation": "softmax",
    "graph_activation_per_block": ["softmax", "softmax", "sparsemax"],
    "num_edge_heads_per_block": [6, 6, 1],
    "graph_hidden_dim_per_block": [192, 192, 96],
    "log_per_step": False,
    "gate_tau": 0.5,
    "gate_row_normalise": True,
    "dynamic_residual_gate": "per_head",
    "dynamic_residual_init": 0.75,
    "dynamic_residual_learnable": True,
    "dynamic_residual_mix": "strict_convex",
    "interlaced_st_blocks": True,
    "num_st_blocks": 4,
    "first_spatial_module_type": None,
    "st_block_post_norm": True,
    "prop_window_size": 4,
    "fusion_window_size": 32,
    "fusion_fast_window": 4,
    "spatial_use_base": True,
    "graph_prior_level": "none",
    "graph_prior_scale": 4.0,
    "graph_prior_learnable": True,
    "prop_lag_aggregation": "softmax",
    "graph_eval_layer": -1,
    "graph_log_all_layers": True,
    "graph_reg_layer": -1,
    "graph_reg_warmup_epochs": 0,
    "graph_entropy_reg": 0.0,
    "graph_target_entropy": 1.8,
    "graph_target_entropy_reg": 0.0,
    "graph_temporal_smooth_reg": 0.0,
}

DIMITRI_X0_TRAINING: dict[str, Any] = {
    "seed": 0,
    "max_epochs": 120,
    "learning_rate": 0.0012,
    "weight_decay": 0.0001,
    "patience": 15,
    "batch_size": DIMITRI_X0_INFERRED_BATCH_SIZE,
    "num_workers": 0,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "scheduler_t_max": 120,
    "precision": "32-true",
    "gradient_clip_norm": None,
    "deterministic": False,
    "selection_metric_original": "validation next-token accuracy",
    "selection_metric_project_curiosity": "test next-token accuracy",
}


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dimitri_source_snapshot() -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in DIMITRI_SOURCE_HASHES.items():
        path = DIMITRI_SOURCE_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        value = _sha256(path)
        if value != expected:
            raise AssertionError(
                f"Dimitri source hash differs for {path}: expected {expected}; "
                f"observed {value}."
            )
        observed[filename] = value
    return observed


def import_dimitri_basedygraph() -> dict[str, Any]:
    """Import the unchanged snapshot under its original top-level names."""
    verify_dimitri_source_snapshot()
    source_root = str(DIMITRI_SOURCE_ROOT.resolve())
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)

    for name in ("model", "modules", "utilities", "data_module", "sector_prior"):
        existing = sys.modules.get(name)
        if existing is None:
            continue
        existing_path = str(getattr(existing, "__file__", ""))
        if not existing_path.startswith(source_root):
            sys.modules.pop(name, None)

    utilities = importlib.import_module("utilities")
    modules = importlib.import_module("modules")
    model = importlib.import_module("model")
    data_module = importlib.import_module("data_module")
    sector_prior = importlib.import_module("sector_prior")

    loaded = {
        "utilities": utilities,
        "modules": modules,
        "model": model,
        "data_module": data_module,
        "sector_prior": sector_prior,
    }
    root = DIMITRI_SOURCE_ROOT.resolve()
    for name, module in loaded.items():
        module_path = Path(module.__file__).resolve()
        if root not in module_path.parents:
            raise ImportError(f"Imported the wrong top-level {name} module: {module_path}")

    return {
        "ModelConfig": utilities.ModelConfig,
        "DiscreteSTGraphBackbone": model.DiscreteSTGraphBackbone,
        "DiscreteSTGraphLightningModule": model.DiscreteSTGraphLightningModule,
        "DiscreteStateDataModule": data_module.DiscreteStateDataModule,
        "sector_prior_module": sector_prior,
        "model_module": model,
        "modules_module": modules,
    }


def _json_copy(values: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(values))


def make_exact_x0_config() -> Any:
    imported = import_dimitri_basedygraph()
    return imported["ModelConfig"](**_json_copy(DIMITRI_X0_CONFIG))


def resolved_per_block_contract(cfg: Any) -> dict[str, list[Any]]:
    imported = import_dimitri_basedygraph()
    resolve = imported["model_module"]._resolve_per_block
    blocks = int(cfg.num_st_blocks)
    return {
        "activations": resolve(
            cfg.graph_activation_per_block,
            blocks,
            cfg.graph_activation,
            "graph_activation_per_block",
        ),
        "num_edge_heads": resolve(
            cfg.num_edge_heads_per_block,
            blocks,
            cfg.num_edge_heads,
            "num_edge_heads_per_block",
        ),
        "graph_hidden_dims": resolve(
            cfg.graph_hidden_dim_per_block,
            blocks,
            cfg.graph_hidden_dim,
            "graph_hidden_dim_per_block",
        ),
    }


def instantiate_exact_x0_model(
    *,
    learning_rate: float = 0.0012,
    weight_decay: float = 0.0001,
    scheduler_t_max: int | None = None,
) -> Any:
    imported = import_dimitri_basedygraph()
    cfg = imported["ModelConfig"](**_json_copy(DIMITRI_X0_CONFIG))
    return imported["DiscreteSTGraphLightningModule"](
        cfg,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        scheduler_t_max=scheduler_t_max,
        true_regime_graphs=None,
    )


def parameter_count(model: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _normalise_container(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_normalise_container(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalise_container(item) for key, item in value.items()}
    return value


def load_dimitri_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    require_exact_x0: bool = True,
) -> dict[str, Any]:
    """Strictly load the supplied Lightning checkpoint into an exact model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if require_exact_x0:
        observed = _sha256(checkpoint_path)
        if observed != DIMITRI_X0_CHECKPOINT_SHA256:
            raise AssertionError(
                "Dimitri checkpoint hash differs. Expected "
                f"{DIMITRI_X0_CHECKPOINT_SHA256}; observed {observed}."
            )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise TypeError(f"Malformed Lightning checkpoint: {checkpoint_path}")

    if require_exact_x0:
        saved_config = checkpoint.get("hyper_parameters", {}).get("cfg_dict")
        if _normalise_container(saved_config) != _normalise_container(DIMITRI_X0_CONFIG):
            raise AssertionError("Checkpoint cfg_dict differs from exact x0jhc0tx.")

    model.load_state_dict(checkpoint["state_dict"], strict=True)

    if require_exact_x0:
        if int(checkpoint.get("epoch", -1)) != DIMITRI_X0_EXPECTED_CHECKPOINT_EPOCH:
            raise AssertionError("Unexpected x0jhc0tx checkpoint epoch.")
        if int(checkpoint.get("global_step", -1)) != DIMITRI_X0_EXPECTED_GLOBAL_STEP:
            raise AssertionError("Unexpected x0jhc0tx global step.")
        if parameter_count(model) != DIMITRI_X0_EXPECTED_PARAMETER_COUNT:
            raise AssertionError(
                f"Exact model has {parameter_count(model):,} trainable parameters; "
                f"expected {DIMITRI_X0_EXPECTED_PARAMETER_COUNT:,}."
            )
    return checkpoint


def extract_dynamic_alphas(model: torch.nn.Module) -> list[list[float]]:
    """Return each block's learned fast-graph mixture weights."""
    values: list[list[float]] = []
    for block in model.backbone.st_blocks:
        scorer = getattr(block, "graph_scorer", None)
        if scorer is None or not hasattr(scorer, "dynamic_residual_alpha"):
            values.append([])
            continue
        alpha = scorer.dynamic_residual_alpha().detach().cpu().float().reshape(-1)
        values.append([float(item) for item in alpha.tolist()])
    return values


def checkpoint_contract_summary(checkpoint_path: str | Path) -> dict[str, Any]:
    model = instantiate_exact_x0_model()
    checkpoint = load_dimitri_checkpoint(model, checkpoint_path)
    return {
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "parameter_count": parameter_count(model),
        "config": asdict(model.cfg),
        "per_block": resolved_per_block_contract(model.cfg),
        "dynamic_alphas": extract_dynamic_alphas(model),
    }

# ---------------------------------------------------------------------------
# Direct-price adapter and graph-prior utilities
# ---------------------------------------------------------------------------

DIMITRI_TOKEN_PRICE_CONTRACT = "dimitri_basedygraph_v2_token_to_price_v1"
DIMITRI_TOKEN_PRICE_EXPECTED_PARAMETER_COUNT = 1_027_890


class DimitriV2TokenToPriceForecaster(nn.Module):
    """Exact Dimitri V2 context backbone with a scalar next-Close head.

    The four interlaced ST blocks are unchanged from ``x0jhc0tx``.  The only
    architectural change is the output head:

    ``Linear(96, 1024)`` next-token logits
        -> ``Linear(96, 1)`` next normalised Close.

    The head is applied at every sequence position except the final one, so the
    model can be trained with the same dense teacher-forced one-step objective
    as Dimitri's token classifier while being evaluated at one chosen forecast
    origin.
    """

    def __init__(
        self,
        *,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        imported = import_dimitri_basedygraph()
        config_values = _json_copy(DIMITRI_X0_CONFIG)
        if config_overrides:
            unknown = sorted(set(config_overrides).difference(config_values))
            if unknown:
                raise KeyError(f"Unknown Dimitri V2 config overrides: {unknown}")
            config_values.update(_json_copy(dict(config_overrides)))
        self.cfg = imported["ModelConfig"](**config_values)
        self.backbone = imported["DiscreteSTGraphBackbone"](self.cfg)
        self.next_close_head = nn.Linear(int(self.cfg.d_model), 1)

    def forward(self, state_ids: torch.Tensor) -> dict[str, Any]:
        if state_ids.ndim != 3:
            raise ValueError(
                "state_ids must have shape [B,N,T], got "
                f"{tuple(state_ids.shape)}."
            )
        output = self.backbone(state_ids)
        spatial = output["spatial_repr"]
        # Representation at position t predicts the raw candle at t+1 after
        # inverse normalisation in the runner.
        output["next_close_normalised"] = self.next_close_head(spatial[:, :-1])
        return output


def instantiate_dimitri_token_to_price_model(
    *,
    config_overrides: Mapping[str, Any] | None = None,
) -> DimitriV2TokenToPriceForecaster:
    return DimitriV2TokenToPriceForecaster(config_overrides=config_overrides)


DIMITRI_CONTINUOUS_PRICE_CONTRACT = (
    "dimitri_basedygraph_v2_continuous_input_direct_price_v1"
)
# Exact count for the default six-channel input adapter.  This replaces the
# 1024 x 96 discrete state embedding with Linear(6,96), leaving every temporal,
# graph, spatial and output-head parameter unchanged.
DIMITRI_CONTINUOUS_PRICE_EXPECTED_PARAMETER_COUNT = 930_258


class _DimitriV2ContinuousBackbone(nn.Module):
    """Dimitri's exact four-block V2 backbone with continuous state inputs.

    The unchanged external backbone is instantiated to obtain the exact
    interlaced ST blocks, normalisation layers, node embedding, graph scorers,
    slow/fast fusion, trainable base logits and spatial message paths.  Its
    discrete ``Embedding(1024,96)`` is removed and replaced by a project-side
    ``Linear(C,96)`` adapter.

    The projected continuous state ``e`` plays exactly the role of Dimitri's
    raw token embedding: it is added to the node embedding at the input and is
    passed directly to every graph scorer and every spatial value projection
    because the saved V2 configuration uses ``scorer_value='concat'`` and
    ``spatial_value='concat'``.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        input_channels: int = 6,
        reference_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if bool(getattr(cfg, "use_state_pair_bias", False)):
            raise ValueError(
                "Continuous inputs do not define discrete state-pair IDs; "
                "use_state_pair_bias must remain false."
            )

        if reference_backbone is None:
            imported = import_dimitri_basedygraph()
            reference = imported["DiscreteSTGraphBackbone"](cfg)
        else:
            reference = reference_backbone
        if not reference.use_interlaced or len(reference.st_blocks) != 4:
            raise AssertionError("Expected Dimitri's four interlaced ST blocks.")

        # Remove the unused discrete embedding from the registered parameter
        # tree. All other source modules remain byte-for-byte unchanged.
        reference.state_embedding = nn.Identity()

        self.cfg = cfg
        self.input_channels = int(input_channels)
        self.input_projection = nn.Linear(self.input_channels, int(cfg.d_model))
        self.reference = reference

    @property
    def st_blocks(self) -> nn.ModuleList:
        """Expose source ST blocks for prior initialisation and diagnostics."""
        return self.reference.st_blocks

    def forward(self, continuous_values: torch.Tensor) -> dict[str, Any]:
        if continuous_values.ndim != 4:
            raise ValueError(
                "continuous_values must have shape [B,N,T,C], got "
                f"{tuple(continuous_values.shape)}."
            )
        b, n, t, channels = continuous_values.shape
        if n != int(self.cfg.num_nodes):
            raise ValueError(
                f"Expected num_nodes={self.cfg.num_nodes}, got {n}."
            )
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} continuous channels, got "
                f"{channels}."
            )
        if not torch.is_floating_point(continuous_values):
            raise TypeError("continuous_values must be floating point.")

        # Raw continuous state representation, analogous to the token lookup
        # e_{b,t,i}. The dataset provides [B,N,T,C].
        e_bntd = self.input_projection(continuous_values)
        e_btnd = e_bntd.permute(0, 2, 1, 3).contiguous()

        h_bntd = e_bntd
        if self.reference.node_embedding is not None:
            node_ids = torch.arange(n, device=continuous_values.device)
            h_bntd = h_bntd + self.reference.node_embedding(node_ids).view(
                1, n, 1, int(self.cfg.d_model)
            )
        h_btnd = self.reference.pre_norm(h_bntd).permute(0, 2, 1, 3).contiguous()

        # The unchanged scorer signature accepts state IDs for the optional
        # state-pair bias. x0jhc0tx has that bias disabled, so a zero placeholder
        # is semantically inert and is asserted above.
        dummy_state_ids = torch.zeros(
            b,
            n,
            t,
            dtype=torch.long,
            device=continuous_values.device,
        )

        block_attns: list[torch.Tensor | None] = []
        for block in self.reference.st_blocks:
            h_btnd, attention = block(
                h_btnd,
                dummy_state_ids,
                e_btnd,
                regimes=None,
                oracle_regime_graphs=None,
            )
            block_attns.append(attention)

        spatial = self.reference.post_norm(h_btnd)
        return {
            "temporal_repr": h_btnd,
            "spatial_repr": spatial,
            "graph_attn": self.reference._select_graph_attn(block_attns),
            "block_graph_attns": block_attns,
        }


class DimitriV2ContinuousToPriceForecaster(nn.Module):
    """Continuous OHLCVA in, direct next-Close out, with exact V2 ST blocks."""

    def __init__(
        self,
        *,
        input_channels: int = 6,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config_values = _json_copy(DIMITRI_X0_CONFIG)
        if config_overrides:
            unknown = sorted(set(config_overrides).difference(config_values))
            if unknown:
                raise KeyError(f"Unknown Dimitri V2 config overrides: {unknown}")
            config_values.update(_json_copy(dict(config_overrides)))
        imported = import_dimitri_basedygraph()
        self.cfg = imported["ModelConfig"](**config_values)

        # Build the unchanged reference backbone and output head in exactly the
        # same RNG order as DimitriV2TokenToPriceForecaster.  The new continuous
        # input projection is created only afterwards, so all shared backbone and
        # head parameters have identical initial values under the same seed.
        reference = imported["DiscreteSTGraphBackbone"](self.cfg)
        self.next_close_head = nn.Linear(int(self.cfg.d_model), 1)
        self.backbone = _DimitriV2ContinuousBackbone(
            self.cfg,
            input_channels=input_channels,
            reference_backbone=reference,
        )

    def forward(self, continuous_values: torch.Tensor) -> dict[str, Any]:
        output = self.backbone(continuous_values)
        output["next_close_normalised"] = self.next_close_head(
            output["spatial_repr"][:, :-1]
        )
        return output


def instantiate_dimitri_continuous_to_price_model(
    *,
    input_channels: int = 6,
    config_overrides: Mapping[str, Any] | None = None,
) -> DimitriV2ContinuousToPriceForecaster:
    return DimitriV2ContinuousToPriceForecaster(
        input_channels=input_channels,
        config_overrides=config_overrides,
    )


DIMITRI_CONTINUOUS_MULTI_HORIZON_CONTRACT = (
    "dimitri_basedygraph_v2_continuous_input_direct_multi_horizon_price_v1"
)


def dimitri_continuous_multi_horizon_parameter_count(
    num_horizons: int,
) -> int:
    """Return the exact parameter count for a direct parallel Close head."""
    if int(num_horizons) <= 0:
        raise ValueError("num_horizons must be positive.")
    # The one-step model already contains one Linear(96,1) head.  Each extra
    # horizon contributes another 96 weights and one bias.
    return int(
        DIMITRI_CONTINUOUS_PRICE_EXPECTED_PARAMETER_COUNT
        + (int(num_horizons) - 1) * (int(DIMITRI_X0_CONFIG["d_model"]) + 1)
    )


class DimitriV2ContinuousMultiHorizonForecaster(nn.Module):
    """Exact V2 context backbone with a direct parallel future-Close head.

    Only the observed context is supplied to the backbone.  The final causal
    representation at the forecast origin is mapped directly to one
    context-normalised Close value for each requested horizon.  No future
    candle or teacher-forced continuation enters temporal, graph, or spatial
    processing.
    """

    def __init__(
        self,
        *,
        evaluation_horizons: Sequence[int],
        input_channels: int = 6,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        horizons = tuple(int(value) for value in evaluation_horizons)
        if not horizons:
            raise ValueError("evaluation_horizons cannot be empty.")
        if any(value <= 0 for value in horizons):
            raise ValueError("evaluation_horizons must be positive.")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError(
                "evaluation_horizons must be unique and strictly increasing."
            )

        config_values = _json_copy(DIMITRI_X0_CONFIG)
        if config_overrides:
            unknown = sorted(set(config_overrides).difference(config_values))
            if unknown:
                raise KeyError(f"Unknown Dimitri V2 config overrides: {unknown}")
            config_values.update(_json_copy(dict(config_overrides)))
        imported = import_dimitri_basedygraph()
        self.cfg = imported["ModelConfig"](**config_values)
        self.evaluation_horizons = horizons

        # Preserve the one-step model's RNG order for the complete shared
        # backbone and continuous input projection.  The temporary legacy head
        # consumes exactly the random draws used by Linear(96,1); its first row
        # is copied into the new multi-horizon head so horizon 1 also starts
        # from the matched one-step initialisation.
        reference = imported["DiscreteSTGraphBackbone"](self.cfg)
        legacy_head = nn.Linear(int(self.cfg.d_model), 1)
        self.backbone = _DimitriV2ContinuousBackbone(
            self.cfg,
            input_channels=input_channels,
            reference_backbone=reference,
        )
        self.future_close_head = nn.Linear(
            int(self.cfg.d_model),
            len(self.evaluation_horizons),
        )
        with torch.no_grad():
            self.future_close_head.weight[0].copy_(legacy_head.weight[0])
            self.future_close_head.bias[0].copy_(legacy_head.bias[0])

    def forward(self, continuous_context: torch.Tensor) -> dict[str, Any]:
        output = self.backbone(continuous_context)
        origin_hidden = output["spatial_repr"][:, -1]  # [B,N,D]
        future = self.future_close_head(origin_hidden)  # [B,N,H]
        output["future_close_normalised"] = (
            future.permute(0, 2, 1).contiguous().unsqueeze(-1)
        )  # [B,H,N,1]
        output["evaluation_horizons"] = self.evaluation_horizons
        return output


def instantiate_dimitri_continuous_multi_horizon_model(
    *,
    evaluation_horizons: Sequence[int],
    input_channels: int = 6,
    config_overrides: Mapping[str, Any] | None = None,
) -> DimitriV2ContinuousMultiHorizonForecaster:
    return DimitriV2ContinuousMultiHorizonForecaster(
        evaluation_horizons=evaluation_horizons,
        input_channels=input_channels,
        config_overrides=config_overrides,
    )


def build_sector_prior(
    asset_cols: Sequence[str],
    company_profiles_path: str | Path,
    *,
    level: str = "sector",
    self_loops: bool = False,
    off_block: float = 0.0,
) -> tuple[torch.Tensor, list[str]]:
    """Build Dimitri's exact row-normalised block prior."""
    if level not in {"sector", "industry"}:
        raise ValueError("level must be 'sector' or 'industry'.")
    imported = import_dimitri_basedygraph()
    prior, labels = imported["sector_prior_module"].build_block_prior(
        list(asset_cols),
        str(Path(company_profiles_path)),
        level=level,
        self_loops=bool(self_loops),
        row_normalise=True,
        off_block=float(off_block),
        dtype=torch.float32,
    )
    return prior.float().contiguous(), [str(value) for value in labels]


def build_absolute_correlation_prior(
    clean_training_split: Mapping[str, Any],
    *,
    asset_cols: Sequence[str],
    threshold: float | None = None,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Build a training-only absolute Close-return correlation prior.

    Returns a non-negative row-stochastic ``[N,N]`` matrix using only
    within-session one-minute Close log returns.  The diagonal is removed.
    When thresholding empties a row, that row's strongest original non-self
    correlation is restored.  A zero-variance all-zero row falls back to a
    uniform non-self distribution.
    """
    if list(clean_training_split.get("asset_cols", [])) != list(asset_cols):
        raise ValueError("Training-split asset order differs from asset_cols.")
    channels = [str(value).lower() for value in clean_training_split["channels"]]
    if "close" not in channels:
        raise KeyError("Training split has no Close channel.")
    close_index = channels.index("close")
    return_parts: list[torch.Tensor] = []
    for sample in clean_training_split["samples"]:
        values = torch.as_tensor(sample[0]).detach().cpu().double()
        if values.ndim != 3 or values.shape[1] != len(asset_cols):
            raise ValueError("Malformed training candle tensor for correlation prior.")
        close = values[..., close_index].clamp_min(eps)
        log_return = close[1:].log() - close[:-1].log()
        if log_return.numel():
            return_parts.append(log_return)
    if not return_parts:
        raise ValueError("No within-session returns are available for the prior.")
    returns = torch.cat(return_parts, dim=0)
    centred = returns - returns.mean(dim=0, keepdim=True)
    covariance = centred.transpose(0, 1) @ centred
    variance = centred.square().sum(dim=0)
    denominator = torch.sqrt(variance[:, None] * variance[None, :])
    correlation = torch.where(
        denominator > eps,
        covariance / denominator.clamp_min(eps),
        torch.zeros_like(covariance),
    ).abs()
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation.fill_diagonal_(0.0)
    unthresholded = correlation.clone()

    if threshold is not None:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("correlation threshold must lie in [0,1].")
        correlation = torch.where(
            correlation >= threshold,
            correlation,
            torch.zeros_like(correlation),
        )

    num_nodes = correlation.shape[0]
    for row_index in range(num_nodes):
        if float(correlation[row_index].sum().item()) > eps:
            continue
        candidate = unthresholded[row_index].clone()
        candidate[row_index] = 0.0
        maximum = float(candidate.max().item())
        if maximum > eps:
            correlation[row_index, int(candidate.argmax().item())] = maximum
        else:
            correlation[row_index] = 1.0
            correlation[row_index, row_index] = 0.0

    row_sum = correlation.sum(dim=-1, keepdim=True).clamp_min(eps)
    prior = correlation / row_sum
    if not torch.allclose(
        prior.sum(dim=-1),
        torch.ones(num_nodes, dtype=prior.dtype),
        atol=1.0e-8,
        rtol=0.0,
    ):
        raise AssertionError("Correlation prior is not row-stochastic.")
    return prior.float().contiguous()


def initialise_base_graphs_from_prior(
    model: nn.Module,
    prior: torch.Tensor,
    *,
    scale: float = 4.0,
    jitter: float = 0.02,
    seed: int = 0,
) -> dict[str, Any]:
    """Initialise every V2 dual-fusion base graph exactly as Dimitri did.

    Dimitri's helper normalises the supplied prior by its maximum, centres it,
    multiplies by ``scale``, and adds small head-specific Gaussian jitter.  The
    resulting tensors initialise the trainable ``base_graph`` logit biases.
    """
    prior = torch.as_tensor(prior).detach().cpu().float()
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError("prior must have shape [N,N].")
    if not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError("prior must be finite and non-negative.")
    if scale <= 0:
        raise ValueError("prior scale must be positive.")
    if jitter < 0:
        raise ValueError("prior jitter cannot be negative.")

    normalised = prior / prior.max().clamp_min(1.0e-6)
    base_logits = float(scale) * (normalised - normalised.mean())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    initialised: list[dict[str, Any]] = []
    for module_name, module in model.named_modules():
        base_graph = getattr(module, "base_graph", None)
        if base_graph is None or not getattr(module, "use_base_graph", False):
            continue
        if not isinstance(base_graph, (nn.Parameter, torch.Tensor)):
            continue
        expected = tuple(base_graph.shape)
        if expected[-2:] != tuple(prior.shape):
            raise ValueError(
                f"Prior shape {tuple(prior.shape)} is incompatible with "
                f"{module_name}.base_graph {expected}."
            )
        heads = int(expected[0])
        values = base_logits.unsqueeze(0).expand(heads, -1, -1).clone()
        if jitter:
            values += torch.randn(
                values.shape,
                generator=generator,
                dtype=values.dtype,
            ) * float(jitter)
        with torch.no_grad():
            base_graph.copy_(values.to(base_graph.device, base_graph.dtype))
        initialised.append(
            {
                "module": module_name,
                "shape": list(expected),
                "heads": heads,
            }
        )

    if not initialised:
        raise RuntimeError("No trainable V2 base_graph tensors were found.")
    return {
        "count": len(initialised),
        "scale": float(scale),
        "jitter": float(jitter),
        "seed": int(seed),
        "modules": initialised,
        "base_logit_min": float(base_logits.min().item()),
        "base_logit_max": float(base_logits.max().item()),
    }


def extract_base_graph_logits(model: nn.Module) -> list[torch.Tensor]:
    """Return one detached base-logit tensor per interlaced ST block."""
    values: list[torch.Tensor] = []
    for block in model.backbone.st_blocks:
        scorer = getattr(block, "graph_scorer", None)
        base_graph = getattr(scorer, "base_graph", None)
        if base_graph is None:
            values.append(torch.empty(0))
        else:
            values.append(base_graph.detach().cpu().float().clone())
    return values

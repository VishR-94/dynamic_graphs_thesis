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
from typing import Any
import hashlib
import importlib
import json
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMITRI_SOURCE_ROOT = PROJECT_ROOT / "external" / "DimitriBaseDyGraphV2" / "src"

DIMITRI_SOURCE_HASHES = {
    "model.py": "b99256db74b84f57513b12715a9ed1f4fc735202bcf933482e3b66ac9cf119d5",
    "modules.py": "1bd31701b300f6f805dfc53530c66eedd9265620093223d3302e3e74409b51ff",
    "utilities.py": "cfe849e2963386ddaab15ad7c89e7df92fc957f22c15f45ea40c89ec4c82f40a",
    "data_module.py": "ecfbe768a1e2c0840043ecf640295b425106843f27062184307534312136cac1",
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

    for name in ("model", "modules", "utilities", "data_module"):
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

    loaded = {
        "utilities": utilities,
        "modules": modules,
        "model": model,
        "data_module": data_module,
    }
    root = DIMITRI_SOURCE_ROOT.resolve()
    for name, module in loaded.items():
        module_path = Path(module.__file__).resolve()
        if root not in module_path.parents:
            raise ImportError(f"Imported the wrong top-level {name} module: {module_path}")

    return {
        "ModelConfig": utilities.ModelConfig,
        "DiscreteSTGraphLightningModule": model.DiscreteSTGraphLightningModule,
        "DiscreteStateDataModule": data_module.DiscreteStateDataModule,
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

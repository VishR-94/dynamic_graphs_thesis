from __future__ import annotations

"""Final token-embedding architecture, sampling, and scale-ablation helpers.

The production training runner remains the only implementation of optimisation,
checkpointing, graph learning, frozen Kronos decoding, and Monte Carlo
inference.  This module defines the controlled six-model architecture grid and
small, deterministic result-loading helpers used by the final Colab notebook.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


HIERARCHICAL_PRESET = "hierarchical_embedding_coarse_ce"
BSQ_CONTROL_PRESET = "modern_tcn_dynamic_coarse_mc10"
BSQ_CONTROL_RUN_NAME = (
    "token_mtg_d32_k1_p8s4_lk15_dynamic_g1_h32_"
    "lr0p0001_glr0p0005_coarse_ce_select"
)
EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME = (
    "embed_mtg_d32_k1_p8s4_lk15_dynamic_st1_ce"
)


@dataclass(frozen=True)
class TokenArchitectureSpec:
    """One exact token-model training/evaluation specification."""

    run_name: str
    label: str
    preset: str
    temporal_family: str
    graph_type: str
    num_st_blocks: int
    token_input_representation: str
    overrides: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "preset": self.preset,
            "temporal_family": self.temporal_family,
            "graph_type": self.graph_type,
            "num_st_blocks": int(self.num_st_blocks),
            "token_input_representation": self.token_input_representation,
            "overrides": list(self.overrides),
        }


COMMON_EMBEDDED_OVERRIDES = (
    "models.dynamic_graph.use_node_embedding=true",
    "models.dynamic_graph.token_input_representation=hierarchical_embedding",
    "models.dynamic_graph.close_scale_features.enabled=false",
    "models.dynamic_graph.close_scale_features.eps=1.0e-6",
    "models.dynamic_graph.graph.num_heads=1",
    "models.dynamic_graph.graph.hidden_dim=32",
    "models.dynamic_graph.graph.activation=softmax",
    "models.dynamic_graph.graph.add_self_loops=false",
    "models.dynamic_graph.spatial.num_layers=1",
    "models.dynamic_graph.spatial.feedforward_multiplier=2",
    "models.dynamic_graph.spatial.dropout=0.0",
    "models.dynamic_graph.spatial.gate_type=learned_scalar",
    "models.dynamic_graph.spatial.initial_beta=0.5",
    "models.dynamic_graph.heads.s1_vocabulary_size=1024",
    "models.dynamic_graph.heads.s2_vocabulary_size=1024",
    "models.dynamic_graph.heads.future_token_mode=coarse_only",
    "models.dynamic_graph.heads.s2_loss_weight=0.0",
    "models.dynamic_graph.heads.s2_conditioning=true_s1",
    # Separate future-query Transformer retained exactly from the current
    # BSQ-input ModernTCN control.  "Second Transformer" denotes this
    # separate predictor module; its depth remains one layer.
    "models.dynamic_graph.future_predictor.type=structured_parallel",
    "models.dynamic_graph.future_predictor.num_layers=1",
    "models.dynamic_graph.future_predictor.num_heads=4",
    "models.dynamic_graph.future_predictor.feedforward_multiplier=2",
    "models.dynamic_graph.future_predictor.dropout=0.0",
    "models.dynamic_graph.loss.horizon_weighting=uniform",
    "models.dynamic_graph.graph_regularisation.graph_reg_layer=-1",
    "models.dynamic_graph.graph_regularisation.graph_reg_warmup_epochs=0",
    "models.dynamic_graph.graph_regularisation.graph_entropy_reg=0.0",
    "models.dynamic_graph.graph_regularisation.graph_target_entropy=null",
    "models.dynamic_graph.graph_regularisation.graph_target_entropy_reg=0.0",
    "models.dynamic_graph.graph_regularisation.graph_temporal_smooth_reg=0.0",
    "training.optimizer=adam",
    "training.scheduler=modern_tcn_type3",
    "training.learning_rate=1.0e-4",
    "training.graph_learning_rate=5.0e-4",
    "training.weight_decay=0.0",
    "training.batch_size=2",
    "training.num_workers=0",
    "training.max_epochs=100",
    "training.patience=10",
    "training.seed=42",
    "training.gradient_clip_norm=1.0",
    "training.mixed_precision=true",
    "training.early_stopping_metric=validation_token_loss",
    "training.early_stopping_horizons=[1,5,15,30,60]",
    "decoding.token_selection=argmax",
    "decoding.temperature=1.0",
    "decoding.top_k=0",
    "decoding.top_p=1.0",
    "decoding.sample_count=1",
)


def _graph_overrides(graph_type: str) -> tuple[str, ...]:
    if graph_type not in {"dynamic", "free_static"}:
        raise ValueError("graph_type must be dynamic or free_static.")
    return (
        f"models.dynamic_graph.graph.type={graph_type}",
        "models.dynamic_graph.graph.base_graph_type=free_static",
        "models.dynamic_graph.graph.gate_type=none",
        "models.dynamic_graph.graph.initial_alpha=0.5",
    )


def _modern_tcn_overrides() -> tuple[str, ...]:
    """Exact continuous-winner ModernTCN geometry, with embeddings added."""
    return (
        "models.dynamic_graph.d_model=32",
        "models.dynamic_graph.num_st_blocks=1",
        "models.dynamic_graph.temporal.type=modern_tcn",
        "models.dynamic_graph.temporal.num_layers=1",
        "models.dynamic_graph.temporal.num_heads=4",
        "models.dynamic_graph.temporal.feedforward_multiplier=2",
        "models.dynamic_graph.temporal.dropout=0.0",
        "models.dynamic_graph.temporal.modern_tcn.patch_size=8",
        "models.dynamic_graph.temporal.modern_tcn.patch_stride=4",
        "models.dynamic_graph.temporal.modern_tcn.ffn_ratio=1",
        "models.dynamic_graph.temporal.modern_tcn.num_blocks=1",
        "models.dynamic_graph.temporal.modern_tcn.large_kernel=15",
        "models.dynamic_graph.temporal.modern_tcn.small_kernel=5",
        "models.dynamic_graph.temporal.modern_tcn.dropout=0.05",
    )


def _transformer_overrides(num_st_blocks: int) -> tuple[str, ...]:
    if int(num_st_blocks) not in {1, 3}:
        raise ValueError("Transformer num_st_blocks must be 1 or 3.")
    return (
        "models.dynamic_graph.d_model=96",
        f"models.dynamic_graph.num_st_blocks={int(num_st_blocks)}",
        "models.dynamic_graph.temporal.type=transformer",
        # One causal Transformer layer in each genuinely interlaced ST block.
        "models.dynamic_graph.temporal.num_layers=1",
        "models.dynamic_graph.temporal.num_heads=8",
        "models.dynamic_graph.temporal.feedforward_multiplier=2",
        "models.dynamic_graph.temporal.dropout=0.0",
    )


def make_architecture_specs() -> tuple[TokenArchitectureSpec, ...]:
    """Return the locked six-fit learned-embedding architecture grid."""
    specs: list[TokenArchitectureSpec] = []

    for graph_type in ("dynamic", "free_static"):
        graph_tag = "dynamic" if graph_type == "dynamic" else "free_static"
        specs.append(
            TokenArchitectureSpec(
                run_name=(
                    f"embed_mtg_d32_k1_p8s4_lk15_{graph_tag}_st1_ce"
                ),
                label=(
                    "Embedded ModernTCN D32/K1/p8-s4/lk15 + "
                    f"{graph_tag} graph"
                ),
                preset=HIERARCHICAL_PRESET,
                temporal_family="modern_tcn",
                graph_type=graph_type,
                num_st_blocks=1,
                token_input_representation="hierarchical_embedding",
                overrides=(
                    *COMMON_EMBEDDED_OVERRIDES,
                    *_modern_tcn_overrides(),
                    *_graph_overrides(graph_type),
                ),
            )
        )

    for num_st_blocks in (1, 3):
        for graph_type in ("dynamic", "free_static"):
            graph_tag = (
                "dynamic" if graph_type == "dynamic" else "free_static"
            )
            specs.append(
                TokenArchitectureSpec(
                    run_name=(
                        "embed_transformer_d96_h8_"
                        f"{graph_tag}_st{num_st_blocks}_ce"
                    ),
                    label=(
                        f"Embedded Transformer D96/H8, {num_st_blocks} "
                        f"ST block(s) + {graph_tag} graph"
                    ),
                    preset=HIERARCHICAL_PRESET,
                    temporal_family="transformer",
                    graph_type=graph_type,
                    num_st_blocks=int(num_st_blocks),
                    token_input_representation="hierarchical_embedding",
                    overrides=(
                        *COMMON_EMBEDDED_OVERRIDES,
                        *_transformer_overrides(num_st_blocks),
                        *_graph_overrides(graph_type),
                    ),
                )
            )

    if len(specs) != 6 or len({spec.run_name for spec in specs}) != 6:
        raise AssertionError("The final architecture sweep must contain six runs.")
    return tuple(specs)


def make_bsq_control_spec() -> TokenArchitectureSpec:
    """Reference the already-trained BSQ-input dynamic ModernTCN control."""
    return TokenArchitectureSpec(
        run_name=BSQ_CONTROL_RUN_NAME,
        label="BSQ-input ModernTCN dynamic-graph control",
        preset=BSQ_CONTROL_PRESET,
        temporal_family="modern_tcn",
        graph_type="dynamic",
        num_st_blocks=1,
        token_input_representation="bsq_bits",
        overrides=(),
    )


def save_specs(path: str | Path, specs: Sequence[TokenArchitectureSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [spec.to_dict() for spec in specs],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, Mapping):
        raise TypeError(f"Expected JSON object in {path}.")
    return dict(values)


def load_selected_validation_ce(run_dir: str | Path) -> dict[str, Any]:
    """Load the CE-selected checkpoint identity from one completed run."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    history_path = run_dir / "history.csv"
    if not metadata_path.is_file() or not history_path.is_file():
        raise FileNotFoundError(
            f"Missing run_metadata.json/history.csv under {run_dir}."
        )
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "completed":
        raise RuntimeError(f"Run is not complete: {run_dir}.")
    history = pd.read_csv(history_path)
    if "validation_token_loss" not in history.columns:
        raise KeyError(f"{history_path} lacks validation_token_loss.")
    losses = pd.to_numeric(history["validation_token_loss"], errors="coerce")
    finite = losses[losses.notna()]
    if finite.empty:
        raise RuntimeError(f"{run_dir.name} has no finite validation CE.")
    best_index = finite.idxmin()
    best_epoch = int(history.loc[best_index, "epoch"])
    best_score = float(finite.loc[best_index])
    if int(metadata["best_epoch"]) != best_epoch:
        raise AssertionError(
            f"{run_dir.name}: metadata best_epoch does not match CE history."
        )
    if abs(float(metadata["best_score"]) - best_score) > 1.0e-6:
        raise AssertionError(
            f"{run_dir.name}: metadata best_score does not match CE history."
        )
    result: dict[str, Any] = {
        "Best epoch": best_epoch,
        "Best validation CE": best_score,
    }
    for source, display in (
        ("validation_s1_accuracy", "Validation s1 accuracy"),
        ("spatial_beta", "Spatial beta"),
        (
            "validation_graph_mean_row_entropy",
            "Graph mean row entropy",
        ),
        (
            "validation_graph_mean_effective_neighbours",
            "Graph effective neighbours",
        ),
    ):
        if source in history.columns:
            value = history.loc[best_index, source]
            result[display] = None if pd.isna(value) else float(value)
    return result


def summarise_validation_ce(
    output_root: str | Path,
    specs: Sequence[TokenArchitectureSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    output_root = Path(output_root)

    for spec in specs:
        run_dir = output_root / spec.run_name
        try:
            selected = load_selected_validation_ce(run_dir)
        except (FileNotFoundError, RuntimeError):
            missing.append(spec.run_name)
            continue
        rows.append(
            {
                "Run": spec.run_name,
                "Label": spec.label,
                "Preset": spec.preset,
                "Token input": spec.token_input_representation,
                "Temporal family": spec.temporal_family,
                "Graph": spec.graph_type,
                "ST blocks": int(spec.num_st_blocks),
                **selected,
            }
        )

    if require_all and missing:
        raise RuntimeError(
            "Missing/incomplete architecture runs: " + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed architecture runs were found.")
    return pd.DataFrame(rows).sort_values(
        ["Best validation CE", "Run"]
    ).reset_index(drop=True)


def select_screening_specs(
    results: pd.DataFrame,
    specs: Sequence[TokenArchitectureSpec],
) -> tuple[TokenArchitectureSpec, ...]:
    """Top two CE models plus embedded dynamic ModernTCN, deduplicated."""
    lookup = {spec.run_name: spec for spec in specs}
    ranked = tuple(
        results.sort_values(["Best validation CE", "Run"])["Run"].astype(str)
    )
    selected_names = list(ranked[:2])
    if EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME not in selected_names:
        selected_names.append(EMBEDDED_MODERN_TCN_DYNAMIC_RUN_NAME)
    selected = tuple(lookup[name] for name in selected_names)
    if not 2 <= len(selected) <= 3:
        raise AssertionError("Screening selection must contain two or three runs.")
    return selected


def temperature_label(value: float) -> str:
    return f"temperature_{float(value):g}".replace(".", "p").replace("-", "m")


def load_temperature_result(
    run_dir: str | Path,
    *,
    temperature: float,
    sample_count: int = 10,
    top_p: float = 0.9,
    top_k: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Load and validate one saved stochastic decoded-price policy."""
    run_dir = Path(run_dir)
    label = temperature_label(temperature)
    policy_dir = run_dir / "temperature_sweep" / label
    result_path = policy_dir / "temperature_result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    record = _load_json(result_path)
    request = record.get("request")
    result = record.get("result")
    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise TypeError(f"Malformed temperature result: {result_path}")
    checks = {
        "temperature": float(temperature),
        "sample_count": int(sample_count),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "sampling_seed": int(seed),
    }
    for key, expected in checks.items():
        observed = request.get(key)
        if isinstance(expected, float):
            if observed is None or abs(float(observed) - expected) > 1.0e-12:
                raise ValueError(
                    f"{result_path}: request {key}={observed}, "
                    f"expected {expected}."
                )
        elif observed is None or int(observed) != expected:
            raise ValueError(
                f"{result_path}: request {key}={observed}, "
                f"expected {expected}."
            )
    sampled_path = policy_dir / "validation_sampled_price_paths.pt"
    metric_table = policy_dir / "validation_metric_table.csv"
    if not sampled_path.is_file():
        raise FileNotFoundError(sampled_path)
    if not metric_table.is_file():
        raise FileNotFoundError(metric_table)
    return {
        **dict(result),
        "Run directory": str(run_dir),
        "Sampled path file": str(sampled_path),
        "Metric table": str(metric_table),
    }


def clone_with_close_log_variance_feature(
    spec: TokenArchitectureSpec,
    *,
    run_name: str,
) -> TokenArchitectureSpec:
    """Create a matched retraining spec with only Close log variance on."""
    prefixes = (
        "models.dynamic_graph.close_scale_features.enabled=",
        "models.dynamic_graph.close_scale_features.eps=",
    )
    retained = tuple(
        value for value in spec.overrides if not value.startswith(prefixes)
    )
    return replace(
        spec,
        run_name=str(run_name),
        label=spec.label + " + Close log-variance feature",
        overrides=(
            *retained,
            "models.dynamic_graph.close_scale_features.enabled=true",
            "models.dynamic_graph.close_scale_features.eps=1.0e-6",
        ),
    )


def clone_with_close_scale_features(
    spec: TokenArchitectureSpec,
    *,
    run_name: str,
) -> TokenArchitectureSpec:
    """Backward-compatible alias for the Close log-variance clone helper."""
    return clone_with_close_log_variance_feature(
        spec,
        run_name=run_name,
    )

from __future__ import annotations

"""Experiment specifications for the ModernTCN graph Round-1 ladder."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd


PriorType = Literal["none", "sector", "correlation", "random"]
Variant = Literal[
    "dynamic_only",
    "dynamic_only_state",
    "prior_mixture",
    "prior_mixture_state",
    "random_static_mixture_state",
]
OptimisationProfile = Literal["round1", "dimitri", "round1_delayed_decay"]
SpatialGateType = Literal["learned_scalar", "none"]
AblationFamily = Literal[
    "round1_baseline",
    "dimitri_optimisation",
    "no_beta_round1_optimisation",
    "no_beta_dimitri_optimisation",
    "alpha_beta_initialisation_sweep",
    "alpha_beta_delayed_decay_sweep",
    "final_sparsemax_graph_ablation",
    "final_correlation_activation_ablation",
]


@dataclass(frozen=True)
class Round1RunSpec:
    run_name: str
    label: str
    variant: Variant
    prior_type: PriorType
    graph_heads: int
    graph_hidden_dim: int
    config: dict[str, Any]
    optimisation_profile: OptimisationProfile = "round1"
    spatial_gate_type: SpatialGateType = "learned_scalar"
    ablation_family: AblationFamily = "round1_baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "label": self.label,
            "variant": self.variant,
            "prior_type": self.prior_type,
            "graph_heads": int(self.graph_heads),
            "graph_hidden_dim": int(self.graph_hidden_dim),
            "optimisation_profile": self.optimisation_profile,
            "spatial_gate_type": self.spatial_gate_type,
            "ablation_family": self.ablation_family,
            "config": self.config,
        }


def _base_config(
    *,
    context_length: int,
    stride: int,
    horizons: Sequence[int],
    prior_type: PriorType,
    prior_scale: float,
    prior_jitter: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "data": {
            "context_length": int(context_length),
            "stride": int(stride),
            "horizons": [int(value) for value in horizons],
            "input_channels": ["open", "high", "low", "close", "volume"],
            "target_channel": "close",
            "input_representation": "raw",
            "normalisation_eps": 1.0e-8,
            "normalisation_clip": False,
        },
        "normalisation": {
            "eps": 1.0e-8,
            "clip": False,
            "clip_min": -5.0,
            "clip_max": 5.0,
        },
        "model": {
            "variant": "dynamic_only",
            "output_representation": "normalised_close",
            "output_head_initialisation": "default",
            "temporal": {
                "type": "modern_tcn",
                "d_model": 32,
                "patch_size": 8,
                "patch_stride": 4,
                "ffn_ratio": 1,
                "num_blocks": 1,
                "large_kernel": 15,
                "small_kernel": 5,
                "dropout": 0.05,
                "head_dropout": 0.0,
                "session_position_encoding": False,
                "modern_tcn": {
                    "patch_size": 8,
                    "patch_stride": 4,
                    "ffn_ratio": 1,
                    "num_blocks": 1,
                    "large_kernel": 15,
                    "small_kernel": 5,
                    "dropout": 0.05,
                    "head_dropout": 0.0,
                },
            },
            "graph": {
                "type": "dynamic",
                "num_heads": 1,
                "num_heads_per_layer": [1],
                "hidden_dim": 32,
                "activation": "softmax",
                "add_self_loops": False,
                "gate_type": "none",
                "initial_alpha": 0.25,
            },
            "spatial": {
                "num_layers": 1,
                "feedforward_multiplier": 2,
                "dropout": 0.0,
                "gate_type": "learned_scalar",
                "initial_beta": 0.5,
            },
            "prior": {
                "type": str(prior_type),
                "scale": float(prior_scale),
                "jitter": float(prior_jitter),
                "seed": int(seed),
            },
            "graph_regularisation": {
                "graph_reg_layer": -1,
                "graph_reg_warmup_epochs": 0,
                "graph_entropy_reg": 0.0,
                "graph_target_entropy": None,
                "graph_target_entropy_reg": 0.0,
                "graph_temporal_smooth_reg": 0.0,
            },
            "head_dropout": 0.0,
        },
        "training": {
            "optimizer": "adam",
            "parameter_grouping": "split",
            "scheduler": "modern_tcn_type3",
            "scheduler_t_max": 120,
            "scheduler_eta_min": 0.0,
            "learning_rate": 2.5e-4,
            "graph_learning_rate": 5.0e-4,
            "weight_decay": 0.0,
            "batch_size": 16,
            "selection_batch_size": 32,
            "validation_batch_size": 32,
            "export_batch_size": 32,
            "num_workers": 0,
            "max_epochs": 100,
            "patience": 10,
            "min_delta": 0.0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": True,
            "seed": int(seed),
            "selection_split": "test",
            "selection_metric": "mean_five_horizon_cumulative_log_change_mae",
            "selection_horizons": [int(value) for value in horizons],
            "loss": {
                "type": "cumulative_log_change_mae",
                "bps_scale": 10000.0,
            },
            "loss_bps_scale": 10000.0,
        },
    }


def _with_variant(
    base: Mapping[str, Any],
    *,
    variant: Variant,
    prior_type: PriorType,
    graph_heads: int,
    graph_hidden_dim: int,
) -> dict[str, Any]:
    values = json.loads(json.dumps(base))
    values["model"]["variant"] = str(variant)
    values["model"]["prior"]["type"] = str(prior_type)
    values["model"]["graph"]["type"] = (
        "dynamic" if variant == "dynamic_only" else "static_dynamic_mixture"
    )
    values["model"]["graph"]["gate_type"] = (
        "none" if variant == "dynamic_only" else "learned_scalar"
    )
    values["model"]["graph"]["num_heads"] = int(graph_heads)
    values["model"]["graph"]["num_heads_per_layer"] = [int(graph_heads)]
    values["model"]["graph"]["hidden_dim"] = int(graph_hidden_dim)
    return values


def _with_spatial_gate(
    values: Mapping[str, Any],
    *,
    gate_type: SpatialGateType,
) -> dict[str, Any]:
    result = json.loads(json.dumps(values))
    if gate_type not in {"learned_scalar", "none"}:
        raise ValueError(f"Unsupported spatial gate type {gate_type!r}.")
    result["model"]["spatial"]["gate_type"] = str(gate_type)
    return result


def _with_optimisation_profile(
    values: Mapping[str, Any],
    *,
    profile: OptimisationProfile,
    dimitri_learning_rate: float = 1.2e-3,
    dimitri_weight_decay: float = 1.0e-4,
    dimitri_t_max: int = 120,
    dimitri_max_epochs: int = 120,
    dimitri_patience: int = 15,
) -> dict[str, Any]:
    """Apply one of the two controlled optimiser/scheduler families.

    ``round1`` preserves Adam, separate backbone/graph learning rates, and
    the ModernTCN type-3 schedule.

    ``dimitri`` keeps the Round-1 data loader, batch size, and seed fixed so
    the ablation isolates optimiser/schedule behaviour, while matching the
    important V2 choices: AdamW, one shared learning rate, weight decay,
    cosine annealing, FP32, no clipping, 120 epochs, and patience 15.
    """

    result = json.loads(json.dumps(values))
    training = result["training"]
    if profile == "round1":
        training.update(
            {
                "optimizer": "adam",
                "parameter_grouping": "split",
                "scheduler": "modern_tcn_type3",
                "learning_rate": 2.5e-4,
                "graph_learning_rate": 5.0e-4,
                "weight_decay": 0.0,
                "max_epochs": 100,
                "patience": 10,
                "gradient_clip_norm": 1.0,
                "mixed_precision": True,
            }
        )
    elif profile == "dimitri":
        if float(dimitri_learning_rate) <= 0.0:
            raise ValueError("dimitri_learning_rate must be positive.")
        if int(dimitri_t_max) <= 0:
            raise ValueError("dimitri_t_max must be positive.")
        training.update(
            {
                "optimizer": "adamw",
                "parameter_grouping": "shared",
                "scheduler": "cosine_annealing",
                "scheduler_t_max": int(dimitri_t_max),
                "scheduler_eta_min": 0.0,
                "learning_rate": float(dimitri_learning_rate),
                "graph_learning_rate": float(dimitri_learning_rate),
                "weight_decay": float(dimitri_weight_decay),
                "max_epochs": int(dimitri_max_epochs),
                "patience": int(dimitri_patience),
                "gradient_clip_norm": 0.0,
                "mixed_precision": False,
            }
        )
    else:
        raise ValueError(f"Unsupported optimisation profile {profile!r}.")
    training["optimisation_profile"] = str(profile)
    return result


def _clone_for_ablation(
    base: Round1RunSpec,
    *,
    run_prefix: str,
    label_suffix: str,
    profile: OptimisationProfile,
    gate_type: SpatialGateType,
    family: AblationFamily,
    dimitri_learning_rate: float,
    dimitri_weight_decay: float,
    dimitri_t_max: int,
    dimitri_max_epochs: int,
    dimitri_patience: int,
) -> Round1RunSpec:
    values = _with_spatial_gate(base.config, gate_type=gate_type)
    values = _with_optimisation_profile(
        values,
        profile=profile,
        dimitri_learning_rate=dimitri_learning_rate,
        dimitri_weight_decay=dimitri_weight_decay,
        dimitri_t_max=dimitri_t_max,
        dimitri_max_epochs=dimitri_max_epochs,
        dimitri_patience=dimitri_patience,
    )
    return Round1RunSpec(
        run_name=f"{run_prefix}__{base.run_name}",
        label=f"{base.label} — {label_suffix}",
        variant=base.variant,
        prior_type=base.prior_type,
        graph_heads=base.graph_heads,
        graph_hidden_dim=base.graph_hidden_dim,
        config=values,
        optimisation_profile=profile,
        spatial_gate_type=gate_type,
        ablation_family=family,
    )


def make_gate_optimisation_ablation_specs(
    *,
    prior_type: Literal["sector", "correlation"] = "sector",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    seed: int = 42,
    dimitri_learning_rate: float = 1.2e-3,
    dimitri_weight_decay: float = 1.0e-4,
    dimitri_t_max: int = 120,
    dimitri_max_epochs: int = 120,
    dimitri_patience: int = 15,
) -> tuple[Round1RunSpec, ...]:
    """Return nine one-head runs: three architectures × three ablations."""

    base_specs = make_round1_specs(
        prior_type=prior_type,
        context_length=context_length,
        stride=stride,
        horizons=horizons,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        seed=seed,
    )
    result: list[Round1RunSpec] = []
    for base in base_specs:
        result.extend(
            [
                _clone_for_ablation(
                    base,
                    run_prefix="r1x_dimitriopt_beta",
                    label_suffix="Dimitri optimisation; learned beta",
                    profile="dimitri",
                    gate_type="learned_scalar",
                    family="dimitri_optimisation",
                    dimitri_learning_rate=dimitri_learning_rate,
                    dimitri_weight_decay=dimitri_weight_decay,
                    dimitri_t_max=dimitri_t_max,
                    dimitri_max_epochs=dimitri_max_epochs,
                    dimitri_patience=dimitri_patience,
                ),
                _clone_for_ablation(
                    base,
                    run_prefix="r1x_round1opt_nobeta",
                    label_suffix="Round-1 optimisation; no external beta gate",
                    profile="round1",
                    gate_type="none",
                    family="no_beta_round1_optimisation",
                    dimitri_learning_rate=dimitri_learning_rate,
                    dimitri_weight_decay=dimitri_weight_decay,
                    dimitri_t_max=dimitri_t_max,
                    dimitri_max_epochs=dimitri_max_epochs,
                    dimitri_patience=dimitri_patience,
                ),
                _clone_for_ablation(
                    base,
                    run_prefix="r1x_dimitriopt_nobeta",
                    label_suffix="Dimitri optimisation; no external beta gate",
                    profile="dimitri",
                    gate_type="none",
                    family="no_beta_dimitri_optimisation",
                    dimitri_learning_rate=dimitri_learning_rate,
                    dimitri_weight_decay=dimitri_weight_decay,
                    dimitri_t_max=dimitri_t_max,
                    dimitri_max_epochs=dimitri_max_epochs,
                    dimitri_patience=dimitri_patience,
                ),
            ]
        )
    if len(result) != 9 or len({spec.run_name for spec in result}) != 9:
        raise AssertionError("Expected nine unique gate/optimisation ablations.")
    return tuple(result)


def _float_tag(value: float) -> str:
    """Return a compact filesystem-safe tag for a finite float."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Sweep values must be finite.")
    return f"{number:g}".replace("-", "m").replace(".", "p")


def make_alpha_beta_initialisation_sweep_specs(
    *,
    alpha_initials: Sequence[float] = (0.5, 0.25, 0.15),
    beta_initials: Sequence[float] = (0.25, 0.5, 0.75),
    prior_type: Literal["sector", "correlation"] = "sector",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    seed: int = 42,
) -> tuple[Round1RunSpec, ...]:
    """Build a flexible initial-alpha × initial-beta sweep for R1-C.

    Every specification uses the one-head ModernTCN prior-mixture-state
    architecture, the original Round-1 optimiser/scheduler, learned scalar
    alpha and beta gates, and no graph regularisation.  The supplied values
    are *initialisations*; both gates remain trainable.
    """

    alpha_values = tuple(float(value) for value in alpha_initials)
    beta_values = tuple(float(value) for value in beta_initials)
    if not alpha_values or not beta_values:
        raise ValueError("alpha_initials and beta_initials must be non-empty.")
    if len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alpha_initials contains duplicate values.")
    if len(set(beta_values)) != len(beta_values):
        raise ValueError("beta_initials contains duplicate values.")
    for name, values in (
        ("alpha_initials", alpha_values),
        ("beta_initials", beta_values),
    ):
        invalid = [value for value in values if not 0.0 < value < 1.0]
        if invalid:
            raise ValueError(
                f"{name} values must lie strictly between zero and one; "
                f"received {invalid}."
            )

    base_specs = make_round1_specs(
        prior_type=prior_type,
        context_length=context_length,
        stride=stride,
        horizons=horizons,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        seed=seed,
    )
    source = next(
        spec for spec in base_specs if spec.variant == "prior_mixture_state"
    )
    prior_tag = "sector" if prior_type == "sector" else "abscorr"
    horizon_tag = "-".join(str(int(value)) for value in horizons)
    suffix = (
        f"{prior_tag}_state_g1_gh32_"
        f"ps{_float_tag(prior_scale)}_"
        f"pj{_float_tag(prior_jitter)}_"
        f"c{int(context_length)}_s{int(stride)}_h{horizon_tag}"
    )

    result: list[Round1RunSpec] = []
    for alpha_initial in alpha_values:
        for beta_initial in beta_values:
            values = json.loads(json.dumps(source.config))
            values["model"]["graph"]["initial_alpha"] = float(alpha_initial)
            values["model"]["spatial"]["initial_beta"] = float(beta_initial)
            values["model"]["graph"]["gate_type"] = "learned_scalar"
            values["model"]["spatial"]["gate_type"] = "learned_scalar"
            values["training"]["optimisation_profile"] = "round1"

            alpha_tag = _float_tag(alpha_initial)
            beta_tag = _float_tag(beta_initial)
            result.append(
                Round1RunSpec(
                    run_name=(
                        f"r1ab_a{alpha_tag}_b{beta_tag}_{suffix}"
                    ),
                    label=(
                        "R1-C alpha/beta initialisation sweep — "
                        f"alpha={alpha_initial:g}, beta={beta_initial:g}"
                    ),
                    variant="prior_mixture_state",
                    prior_type=prior_type,
                    graph_heads=1,
                    graph_hidden_dim=32,
                    config=values,
                    optimisation_profile="round1",
                    spatial_gate_type="learned_scalar",
                    ablation_family="alpha_beta_initialisation_sweep",
                )
            )

    run_names = [spec.run_name for spec in result]
    if len(set(run_names)) != len(run_names):
        raise ValueError(
            "The requested alpha/beta values produce duplicate run names."
        )
    return tuple(result)


def make_alpha_beta_delayed_decay_sweep_specs(
    *,
    alpha_initials: Sequence[float] = (0.25, 0.5, 0.75, 0.85),
    beta_initials: Sequence[float] = (0.25, 0.5, 0.75),
    prior_type: Literal["sector", "correlation"] = "sector",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    decay_start_epoch: int = 15,
    decay_factor: float = 0.9,
    seed: int = 42,
) -> tuple[Round1RunSpec, ...]:
    """Build the final alpha/beta sweep with delayed type-3 decay.

    The architecture, optimiser, base learning rates, batch size, precision,
    clipping, patience, seed, and test-selection rule are identical to the
    original alpha/beta sweep.  Only the learning-rate schedule changes:
    both parameter groups remain at their configured base learning rates
    through ``decay_start_epoch``; epoch ``decay_start_epoch + 1`` uses one
    factor of ``decay_factor`` and later epochs continue geometrically.
    """

    start_epoch = int(decay_start_epoch)
    factor = float(decay_factor)
    if start_epoch < 1:
        raise ValueError("decay_start_epoch must be at least one.")
    if not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("decay_factor must lie in (0, 1].")

    controls = make_alpha_beta_initialisation_sweep_specs(
        alpha_initials=alpha_initials,
        beta_initials=beta_initials,
        prior_type=prior_type,
        context_length=context_length,
        stride=stride,
        horizons=horizons,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        seed=seed,
    )

    start_tag = str(start_epoch)
    factor_tag = _float_tag(factor)
    result: list[Round1RunSpec] = []
    for control in controls:
        values = json.loads(json.dumps(control.config))
        training = values["training"]
        training["scheduler"] = "modern_tcn_type3_delayed"
        training["scheduler_decay_start_epoch"] = start_epoch
        training["scheduler_decay_factor"] = factor
        training["optimisation_profile"] = "round1_delayed_decay"

        # Replace only the family prefix; every architecture/data/gate value
        # already encoded by the control run name remains visible.
        if not control.run_name.startswith("r1ab_"):
            raise AssertionError(
                f"Unexpected alpha/beta control name {control.run_name!r}."
            )
        run_name = (
            f"r1abdelay_ds{start_tag}_df{factor_tag}_"
            + control.run_name.removeprefix("r1ab_")
        )
        result.append(
            Round1RunSpec(
                run_name=run_name,
                label=(
                    control.label
                    + f" — full LR through epoch {start_epoch}, "
                    + f"then ×{factor:g} per epoch"
                ),
                variant=control.variant,
                prior_type=control.prior_type,
                graph_heads=control.graph_heads,
                graph_hidden_dim=control.graph_hidden_dim,
                config=values,
                optimisation_profile="round1_delayed_decay",
                spatial_gate_type=control.spatial_gate_type,
                ablation_family="alpha_beta_delayed_decay_sweep",
            )
        )

    run_names = [spec.run_name for spec in result]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Delayed-decay sweep produced duplicate run names.")
    return tuple(result)


def make_round1_specs(
    *,
    prior_type: Literal["sector", "correlation"] = "sector",
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    prior_scale: float = 4.0,
    prior_jitter: float = 0.02,
    seed: int = 42,
) -> tuple[Round1RunSpec, ...]:
    if prior_type not in {"sector", "correlation"}:
        raise ValueError("prior_type must be 'sector' or 'correlation'.")
    base = _base_config(
        context_length=context_length,
        stride=stride,
        horizons=horizons,
        prior_type=prior_type,
        prior_scale=prior_scale,
        prior_jitter=prior_jitter,
        seed=seed,
    )
    suffix = f"c{int(context_length)}_s{int(stride)}"
    prior_tag = "sector" if prior_type == "sector" else "abscorr"
    return (
        Round1RunSpec(
            run_name=f"r1_control_dynamic_g1_{suffix}",
            label="R1-A — exact ModernTCN dynamic-only control",
            variant="dynamic_only",
            prior_type="none",
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="dynamic_only",
                prior_type="none",
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
        Round1RunSpec(
            run_name=f"r1_{prior_tag}_static_dynamic_g1_{suffix}",
            label="R1-B — prior-initialised static + dynamic graph",
            variant="prior_mixture",
            prior_type=prior_type,
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="prior_mixture",
                prior_type=prior_type,
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
        Round1RunSpec(
            run_name=f"r1_{prior_tag}_static_dynamic_state_g1_{suffix}",
            label=(
                "R1-C — prior/static/dynamic graph with state-aware scorer "
                "and spatial values"
            ),
            variant="prior_mixture_state",
            prior_type=prior_type,
            graph_heads=1,
            graph_hidden_dim=32,
            config=_with_variant(
                base,
                variant="prior_mixture_state",
                prior_type=prior_type,
                graph_heads=1,
                graph_hidden_dim=32,
            ),
        ),
    )


def make_six_head_ablation_spec(
    winner: Round1RunSpec,
    *,
    graph_heads: int = 6,
    per_head_dim: int = 32,
) -> Round1RunSpec:
    graph_hidden_dim = int(graph_heads) * int(per_head_dim)
    values = _with_variant(
        winner.config,
        variant=winner.variant,
        prior_type=winner.prior_type,
        graph_heads=graph_heads,
        graph_hidden_dim=graph_hidden_dim,
    )
    return Round1RunSpec(
        run_name=f"{winner.run_name}_g{graph_heads}_gh{graph_hidden_dim}",
        label=(
            f"Round-1 winner with {graph_heads} graph heads "
            f"({per_head_dim} dimensions per head)"
        ),
        variant=winner.variant,
        prior_type=winner.prior_type,
        graph_heads=int(graph_heads),
        graph_hidden_dim=graph_hidden_dim,
        config=values,
        optimisation_profile=winner.optimisation_profile,
        spatial_gate_type=winner.spatial_gate_type,
        ablation_family=winner.ablation_family,
    )


def save_specs(path: str | Path, specs: Sequence[Round1RunSpec]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([spec.to_dict() for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_run_config(path: str | Path, spec: Round1RunSpec) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return values


def run_is_complete(run_dir: str | Path) -> bool:
    directory = Path(run_dir)
    metadata = directory / "run_metadata.json"
    checkpoint = directory / "best_checkpoint.pt"
    return (
        metadata.is_file()
        and checkpoint.is_file()
        and load_json(metadata).get("status") == "completed"
    )


def summarise_runs(
    output_root: str | Path,
    specs: Sequence[Round1RunSpec],
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        directory = Path(output_root) / spec.run_name
        metadata_path = directory / "run_metadata.json"
        history_path = directory / "history.csv"
        if not metadata_path.is_file() or not history_path.is_file():
            missing.append(spec.run_name)
            continue
        metadata = load_json(metadata_path)
        if metadata.get("status") != "completed":
            missing.append(spec.run_name)
            continue
        history = pd.read_csv(history_path)
        best_epoch = int(metadata["best_epoch"])
        selected = history.loc[pd.to_numeric(history["epoch"]) == best_epoch]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one selected history row for {spec.run_name}; "
                f"found {len(selected)}."
            )
        row = selected.iloc[0]
        result: dict[str, Any] = {
            "Run": spec.run_name,
            "Label": spec.label,
            "Variant": spec.variant,
            "Prior": spec.prior_type,
            "Graph heads": spec.graph_heads,
            "Graph hidden dim": spec.graph_hidden_dim,
            "Optimisation": spec.optimisation_profile,
            "Beta gate": spec.spatial_gate_type,
            "Ablation family": spec.ablation_family,
            "Best epoch": best_epoch,
            "Epochs completed": int(metadata["epochs_completed"]),
            "Mean test Log MAE": float(row["selection_score"]),
            "Alpha": row.get("block_0_alpha"),
            "Beta": row.get("block_0_beta"),
            "Selected graph entropy": row.get("block_0_selected_entropy"),
            "Static graph entropy": row.get("block_0_static_entropy"),
            "Dynamic graph entropy": row.get("block_0_dynamic_entropy"),
            "Run directory": str(directory),
        }
        for horizon in spec.config["data"]["horizons"]:
            result[f"Log MAE — {int(horizon)} min"] = float(
                row[f"test_cumulative_log_change_mae_h{int(horizon)}"]
            )
        rows.append(result)
    if require_all and missing:
        raise FileNotFoundError(
            "Missing completed Round-1 runs: " + ", ".join(missing)
        )
    if not rows:
        raise RuntimeError("No completed Round-1 runs were found.")
    return (
        pd.DataFrame(rows)
        .sort_values(["Mean test Log MAE", "Run"])
        .reset_index(drop=True)
    )

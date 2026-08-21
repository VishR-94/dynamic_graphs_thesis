from __future__ import annotations

"""Configuration for the additive Sonnet weather benchmark port.

Nothing in this package changes the existing financial experiment code.  The
configuration records the frozen financial architectures and the task-specific
changes required by the Sonnet WeatherBench protocol.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Literal


ModelKind = Literal["modern_tcn_1st", "transformer_3st"]

WEATHER_HORIZON_TO_CONTEXT: dict[int, int] = {
    4: 28,
    12: 28,
    28: 56,
    120: 240,
}

# Three validation candidates per Sonnet weather task.  Kernels operate on the
# ModernTCN patch sequence (patch size 8, stride 4), not directly on raw
# six-hour observations.  The short-context grids span local to full-context
# final-patch coverage.  For H=120, the three values give the final graph
# state a local, approximately half-context, and full-context receptive field
# over the 60-patch sequence while retaining the frozen kernel-15 candidate.
MODERN_TCN_KERNEL_GRID_BY_HORIZON: dict[int, tuple[int, ...]] = {
    4: (7, 11, 15),
    12: (7, 11, 15),
    28: (15, 21, 27),
    120: (15, 61, 119),
}

SUPPORTED_CITIES: tuple[str, ...] = (
    "capetown",
    "hongkong",
    "london",
    "newyork",
    "singapore",
)
MODEL_OUTPUT_DIRECTORIES: dict[ModelKind, str] = {
    "modern_tcn_1st": "modernTCN",
    "transformer_3st": "3st_block_transformer",
}

# Model/node feature names.  The raw CSV uses ``t_*`` for neighbouring T850.
WEATHER_FEATURES: tuple[str, ...] = ("z500", "t850", "t2m", "u10", "v10")
WEATHER_NODES: tuple[str, ...] = (
    "C",
    "NW",
    "N",
    "NE",
    "W",
    "E",
    "SW",
    "S",
    "SE",
)
CENTRAL_NODE_INDEX = 0


@dataclass(frozen=True)
class WeatherRunConfig:
    """Resolved settings for one ``(model, city, test year, horizon)`` run."""

    model_kind: ModelKind
    city: str
    test_year: int
    horizon: int
    data_path: Path
    output_root: Path

    start_year: int = 1980
    seed: int = 42
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.0
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    # Existing financial optimisation schedules; these are not re-tuned for
    # the weather benchmark unless explicitly changed by a later experiment.
    backbone_learning_rate: float = 2.5e-4
    graph_learning_rate: float = 5.0e-4
    scheduler_decay_factor: float = 0.9

    # Existing price-model graph-prior constants.
    prior_scale: float = 4.0
    prior_jitter: float = 0.02
    prior_seed: int = 42

    # Weather-only experiment controls.  Their defaults exactly preserve the
    # original frozen-transfer run topology and directory layout.
    modern_tcn_large_kernel: int = 15
    train_batch_size_override: int | None = None
    validation_batch_size_override: int | None = None
    export_batch_size_override: int | None = None
    run_suffix: str | None = None

    # Runtime-only accelerations.  They do not change model parameters, loss,
    # causal constraints, optimiser or learning-rate schedule.
    cache_causal_masks: bool = False
    progress_update_interval: int = 1
    prefetch_factor: int = 2

    # Execution controls.
    device: str = "auto"
    resume: bool = True
    overwrite: bool = False
    skip_completed: bool = True
    export_train_split: bool = True

    def __post_init__(self) -> None:
        city = str(self.city).lower().strip()
        suffix = None if self.run_suffix is None else str(self.run_suffix).strip()
        object.__setattr__(self, "city", city)
        object.__setattr__(self, "run_suffix", suffix or None)
        object.__setattr__(self, "data_path", Path(self.data_path).expanduser())
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        self.validate()

    def validate(self) -> None:
        if self.model_kind not in MODEL_OUTPUT_DIRECTORIES:
            raise ValueError(f"Unsupported model_kind: {self.model_kind!r}.")
        if self.city not in SUPPORTED_CITIES:
            raise ValueError(
                f"Unsupported city {self.city!r}; expected one of {SUPPORTED_CITIES}."
            )
        if int(self.horizon) not in WEATHER_HORIZON_TO_CONTEXT:
            raise ValueError(
                f"Unsupported horizon {self.horizon}; expected one of "
                f"{tuple(WEATHER_HORIZON_TO_CONTEXT)}."
            )
        if int(self.test_year) <= int(self.start_year) + 1:
            raise ValueError("test_year must leave distinct training and validation years.")
        if int(self.max_epochs) <= 0:
            raise ValueError("max_epochs must be positive.")
        if int(self.patience) <= 0:
            raise ValueError("patience must be positive.")
        if float(self.min_delta) < 0.0:
            raise ValueError("min_delta must be non-negative.")
        if float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        if float(self.backbone_learning_rate) <= 0.0:
            raise ValueError("backbone_learning_rate must be positive.")
        if float(self.graph_learning_rate) <= 0.0:
            raise ValueError("graph_learning_rate must be positive.")
        if not 0.0 < float(self.scheduler_decay_factor) <= 1.0:
            raise ValueError("scheduler_decay_factor must lie in (0, 1].")
        if float(self.prior_scale) <= 0.0:
            raise ValueError("prior_scale must be positive.")
        if float(self.prior_jitter) < 0.0:
            raise ValueError("prior_jitter must be non-negative.")
        if int(self.modern_tcn_large_kernel) < 5:
            raise ValueError("modern_tcn_large_kernel must be at least 5.")
        if int(self.modern_tcn_large_kernel) % 2 == 0:
            raise ValueError("modern_tcn_large_kernel must be odd.")
        for name, value in (
            ("train_batch_size_override", self.train_batch_size_override),
            ("validation_batch_size_override", self.validation_batch_size_override),
            ("export_batch_size_override", self.export_batch_size_override),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.run_suffix is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_suffix
        ):
            raise ValueError(
                "run_suffix must contain only letters, numbers, '.', '_' or '-', "
                "and must start with a letter or number."
            )
        if int(self.progress_update_interval) <= 0:
            raise ValueError("progress_update_interval must be positive.")
        if int(self.prefetch_factor) <= 0:
            raise ValueError("prefetch_factor must be positive.")

    @property
    def context_length(self) -> int:
        return int(WEATHER_HORIZON_TO_CONTEXT[int(self.horizon)])

    @property
    def validation_year(self) -> int:
        return int(self.test_year) - 1

    @property
    def training_end_year(self) -> int:
        return int(self.test_year) - 2

    @property
    def batch_size(self) -> int:
        if self.train_batch_size_override is not None:
            return int(self.train_batch_size_override)
        return 16 if self.model_kind == "modern_tcn_1st" else 1

    @property
    def validation_batch_size(self) -> int:
        if self.validation_batch_size_override is not None:
            return int(self.validation_batch_size_override)
        return 32 if self.model_kind == "modern_tcn_1st" else 2

    @property
    def export_batch_size(self) -> int:
        if self.export_batch_size_override is not None:
            return int(self.export_batch_size_override)
        return 32 if self.model_kind == "modern_tcn_1st" else 2

    @property
    def scheduler_decay_start_epoch(self) -> int:
        # The completed epoch shown here is the final epoch at the base LR.
        return 15 if self.model_kind == "modern_tcn_1st" else 10

    @property
    def model_output_directory(self) -> str:
        return MODEL_OUTPUT_DIRECTORIES[self.model_kind]

    @property
    def run_directory(self) -> Path:
        leaf = f"test_year_{int(self.test_year)}"
        if self.run_suffix is not None:
            leaf = f"{leaf}_{self.run_suffix}"
        return (
            self.output_root
            / self.model_output_directory
            / self.city
            / f"horizon_{int(self.horizon)}"
            / leaf
        )

    @property
    def dense_prefix_training(self) -> bool:
        return self.model_kind == "transformer_3st"

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)

        # Keep legacy checkpoint signatures valid for the original frozen
        # weather runs.  New fields are absent when they are at their old-
        # behaviour defaults, but are included for every ablation/accelerated
        # run where they materially identify the experiment.
        if int(self.modern_tcn_large_kernel) == 15:
            values.pop("modern_tcn_large_kernel")
        if self.train_batch_size_override is None:
            values.pop("train_batch_size_override")
        if self.validation_batch_size_override is None:
            values.pop("validation_batch_size_override")
        if self.export_batch_size_override is None:
            values.pop("export_batch_size_override")
        if self.run_suffix is None:
            values.pop("run_suffix")
        if not bool(self.cache_causal_masks):
            values.pop("cache_causal_masks")
        if int(self.progress_update_interval) == 1:
            values.pop("progress_update_interval")
        # DataLoader's historical default is already 2, so omitting it at 2
        # retains the prior signature and behaviour.
        if int(self.prefetch_factor) == 2:
            values.pop("prefetch_factor")

        values["data_path"] = str(self.data_path)
        values["output_root"] = str(self.output_root)
        values.update(
            {
                "context_length": self.context_length,
                "validation_year": self.validation_year,
                "training_end_year": self.training_end_year,
                "batch_size": self.batch_size,
                "validation_batch_size": self.validation_batch_size,
                "export_batch_size": self.export_batch_size,
                "scheduler_decay_start_epoch": self.scheduler_decay_start_epoch,
                "run_directory": str(self.run_directory),
                "model_output_directory": self.model_output_directory,
                "dense_prefix_training": self.dense_prefix_training,
                "forecast_steps": list(range(1, int(self.horizon) + 1)),
                "node_names": list(WEATHER_NODES),
                "feature_names": list(WEATHER_FEATURES),
                "central_node_index": CENTRAL_NODE_INDEX,
            }
        )
        return values

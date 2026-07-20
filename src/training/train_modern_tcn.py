from collections.abc import Callable, Mapping
from typing import Any
import argparse
import torch
import wandb
from pathlib import Path
from copy import deepcopy
from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.utils.config import load_yaml
from time import perf_counter
from src.models.modern_tcn import ModernTCNBaseline
from src.evaluation.metrics import ForecastEvaluator
from src.utils.metric_tables import make_evaluation_table

EpochRecord = dict[str, float | int]
ConfigDict = dict[str, Any]
SplitDict = dict[str, Any]

def make_wandb_epoch_callback(
    run: Any,
) -> Callable[[EpochRecord], None]:
    """
    Create a callback that logs one completed ModernTCN epoch to W&B.

    Args:
        run:
            Active W&B run object returned by wandb.init().

    Returns:
        Function compatible with ModernTCNBaseline.fit(
            epoch_callback=...
        ).
    """
    def log_epoch(
        epoch_record: EpochRecord,
    ) -> None:
        epoch = int(epoch_record["epoch"])

        run.log(
            {
                "epoch": epoch,
                "train/mse": float(
                    epoch_record["training_loss"]
                ),
                "val/mse": float(
                    epoch_record["validation_loss"]
                ),
                "learning_rate": float(
                    epoch_record["learning_rate"]
                ),
            },
            step=epoch,
        )

    return log_epoch

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for a ModernTCN experiment.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate ModernTCN with W&B logging."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecasting.yaml"),
        help="Path to the forecasting YAML configuration.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.pt, val.pt, and test.pt.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory in which the best checkpoint will be saved.",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dynamic-graph-financial-forecasting",
        help="W&B project name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable W&B run name.",
    )

    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
        help=(
            "Split evaluated after training. Keep the default 'val' "
            "during model development."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=(
            "Optional explicit override for the configured maximum "
            "number of epochs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional explicit override for the configured "
            "ModernTCN training batch size."
        ),
    )

    parser.add_argument(
        "--input-channels",
        nargs="+",
        default=None,
        help=(
            "Optional ordered input-channel override. "
            "Example: --input-channels open high low close volume"
        ),
    )

    parser.add_argument(
        "--target-channels",
        nargs="+",
        default=None,
        help=(
            "Optional ordered forecast-target override. "
            "Example: --target-channels close"
        ),
    )

    parser.add_argument(
        "--variable-layout",
        choices=("joint", "per_asset"),
        default=None,
        help=(
            "Optional ModernTCN variable-layout override."
        ),
    )

    parser.add_argument(
        "--smoke-days",
        type=int,
        default=None,
        help=(
            "Retain only the first N chronological sessions from each split "
            "for a fast pipeline smoke test."
        ),
    )

    return parser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for one ModernTCN experiment.
    """
    return build_argument_parser().parse_args()


def prepare_experiment_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    """
    Validate experiment inputs and prepare the checkpoint root.

    Returns:
        config_path:
            Resolved forecasting YAML path.
        data_dir:
            Resolved directory containing cached split files.
        checkpoint_root:
            Resolved directory under which each W&B run gets its
            own checkpoint folder.
    """
    config_path = args.config.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint_root = (
        args.checkpoint_dir.expanduser().resolve()
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    required_split_files = (
        "train.pt",
        "val.pt",
        "test.pt",
    )

    missing_split_files = [
        filename
        for filename in required_split_files
        if not (data_dir / filename).is_file()
    ]

    if missing_split_files:
        raise FileNotFoundError(
            "Missing cached split files in "
            f"{data_dir}: {missing_split_files}"
        )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return config_path, data_dir, checkpoint_root


def build_run_checkpoint_path(
    checkpoint_root: Path,
    experiment_name: str,
    run_id: str,
) -> Path:
    """
    Create a unique best-checkpoint path for one W&B run.

    Structure:
        <checkpoint_root>/<experiment_name>/runs/<run_id>/
    """
    if not experiment_name:
        raise ValueError(
            "experiment_name must not be empty."
        )

    if not run_id:
        raise ValueError(
            "W&B run ID must not be empty."
        )

    run_checkpoint_dir = (
        checkpoint_root
        / experiment_name
        / "runs"
        / run_id
    )

    run_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    return (
        run_checkpoint_dir
        / "best_checkpoint.pt"
    )


def apply_cli_overrides(
    config: ConfigDict,
    args: argparse.Namespace,
) -> ConfigDict:
    """
    Return an independent config with explicit CLI overrides applied.

    The committed YAML remains the source of default values. These
    overrides are intended for transparent one-off runs such as the
    first one-epoch Colab smoke test.
    """
    resolved_config = deepcopy(config)

    forecasting_config = resolved_config[
    "forecasting"
    ]

    modern_tcn_config = resolved_config[
        "models"
    ]["modern_tcn"]

    training_config = modern_tcn_config[
        "training"
    ]

    if args.input_channels is not None:
        input_channels = [
            str(channel).strip()
            for channel in args.input_channels
        ]

        if not input_channels:
            raise ValueError(
                "--input-channels must contain at least one channel."
            )

        if any(not channel for channel in input_channels):
            raise ValueError(
                "--input-channels cannot contain empty names."
            )

        if len(input_channels) != len(set(input_channels)):
            raise ValueError(
                "--input-channels cannot contain duplicates."
            )

        forecasting_config[
            "input_channels"
        ] = input_channels


    if args.target_channels is not None:
        target_channels = [
            str(channel).strip()
            for channel in args.target_channels
        ]

        if not target_channels:
            raise ValueError(
                "--target-channels must contain at least one channel."
            )

        if any(
            not channel
            for channel in target_channels
        ):
            raise ValueError(
                "--target-channels cannot contain empty names."
            )

        if len(target_channels) != len(
            set(target_channels)
        ):
            raise ValueError(
                "--target-channels cannot contain duplicates."
            )

        forecasting_config[
            "target_channels"
        ] = target_channels


    resolved_input_channels = list(
        forecasting_config.get(
            "input_channels",
            forecasting_config["target_channels"],
        )
    )

    resolved_target_channels = list(
        forecasting_config["target_channels"]
    )

    missing_targets = [
        channel
        for channel in resolved_target_channels
        if channel not in resolved_input_channels
    ]

    if missing_targets:
        raise ValueError(
            "Every target channel must also be present in "
            "input_channels. Missing targets: "
            f"{missing_targets}."
        )

    if args.variable_layout is not None:
        modern_tcn_config[
            "variable_layout"
        ] = args.variable_layout

    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            raise ValueError(
                "--max-epochs must be greater than zero."
            )

        training_config["max_epochs"] = args.max_epochs

    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError(
                "--batch-size must be greater than zero."
            )

        training_config["batch_size"] = args.batch_size

    return resolved_config

def build_wandb_config(
    config: ConfigDict,
    data_dir: Path,
    evaluation_split: str,
) -> ConfigDict:
    """
    Build the flat, fully resolved configuration recorded by W&B.

    The three planned sweep parameters use top-level names:
        learning_rate
        ffn_ratio
        num_blocks
    """
    forecasting_config = config["forecasting"]
    modern_tcn_config = config["models"]["modern_tcn"]
    training_config = modern_tcn_config["training"]
    global_training_config = config.get("training", {})

    revin_enabled = bool(
        modern_tcn_config["revin"]
    )



    return {
        # Experiment identity.
        "model": "modern_tcn",
        "dataset_id": data_dir.parent.name,
        "cached_split_directory": str(data_dir),
        "evaluation_split": evaluation_split,
        "variable_layout": str(
            modern_tcn_config.get(
                "variable_layout",
                "joint",
            )
        ),

        # Forecasting task.
        "context_length": int(
            forecasting_config["context_length"]
        ),
        "horizons": [
            int(horizon)
            for horizon in forecasting_config["horizons"]
        ],
        "input_channels": list(
            forecasting_config.get(
                "input_channels",
                forecasting_config["target_channels"],
            )
        ),
        "target_channels": list(
            forecasting_config["target_channels"]
        ),
        "window_stride": int(
            forecasting_config["stride"]
        ),

        # ModernTCN architecture.
        "patch_size": int(
            modern_tcn_config["patch_size"]
        ),
        "patch_stride": int(
            modern_tcn_config["patch_stride"]
        ),
        "hidden_dim": int(
            modern_tcn_config["hidden_dim"]
        ),
        "ffn_ratio": int(
            modern_tcn_config["ffn_ratio"]
        ),
        "num_blocks": int(
            modern_tcn_config["num_blocks"]
        ),
        "large_kernel": int(
            modern_tcn_config["large_kernel"]
        ),
        "small_kernel": int(
            modern_tcn_config["small_kernel"]
        ),
        "dropout": float(
            modern_tcn_config["dropout"]
        ),
        "head_dropout": float(
            modern_tcn_config["head_dropout"]
        ),
        "revin": revin_enabled,
        "loss_space": (
            "raw"
            if bool(modern_tcn_config["revin"])
            else "window_context_normalised"
        ),
        "revin_affine": bool(
            modern_tcn_config["revin_affine"]
        ),
        "subtract_last": bool(
            modern_tcn_config["subtract_last"]
        ),
        "individual_head": bool(
            modern_tcn_config["individual_head"]
        ),
        "use_multi_scale": bool(
            modern_tcn_config["use_multi_scale"]
        ),
        "small_kernel_merged": bool(
            modern_tcn_config["small_kernel_merged"]
        ),
        "normalisation_enabled": (
            not revin_enabled
        ),
        "normalisation_method": str(
            config["normalisation"]["method"]
        ),
        "normalisation_clip": bool(
            config["normalisation"]["clip"]
        ),
        "normalisation_config_enabled": bool(
            config["normalisation"]["enabled"]
        ),

        # Training.
        "learning_rate": float(
            training_config["learning_rate"]
        ),
        "weight_decay": float(
            training_config["weight_decay"]
        ),
        "batch_size": int(
            training_config["batch_size"]
        ),
        "num_workers": int(
            training_config.get(
                "num_workers",
                global_training_config.get(
                    "num_workers",
                    0,
                ),
            )
        ),
        "max_epochs": int(
            training_config["max_epochs"]
        ),
        "patience": int(
            training_config["patience"]
        ),
        "seed": int(
            training_config["seed"]
        ),
        "scheduler_type": str(
            training_config["scheduler_type"]
        ),
        "optimizer": "adam",
        "loss": "mse",
    }


def apply_wandb_hyperparameters(
    config: ConfigDict,
    run_config: Mapping[str, Any],
) -> ConfigDict:
    """
    Apply the effective W&B hyperparameters before model creation.

    For the default run these values equal the YAML defaults. During
    a sweep, W&B supplies the selected values through run.config.
    """
    resolved_config = deepcopy(config)

    learning_rate = float(
        run_config["learning_rate"]
    )
    ffn_ratio = int(
        run_config["ffn_ratio"]
    )
    num_blocks = int(
        run_config["num_blocks"]
    )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if ffn_ratio <= 0:
        raise ValueError(
            "ffn_ratio must be greater than zero."
        )

    if num_blocks <= 0:
        raise ValueError(
            "num_blocks must be greater than zero."
        )

    modern_tcn_config = resolved_config[
        "models"
    ]["modern_tcn"]

    modern_tcn_config["ffn_ratio"] = ffn_ratio
    modern_tcn_config["num_blocks"] = num_blocks
    modern_tcn_config["training"][
        "learning_rate"
    ] = learning_rate

    return resolved_config

def load_experiment_inputs(
    config_path: Path,
    data_dir: Path,
) -> tuple[
    ConfigDict,
    SplitDict,
    SplitDict,
    SplitDict,
]:
    """
    Load the project configuration and cleaned chronological splits.

    The split boundaries are already encoded in the cached files.
    Cleaning is applied jointly through the canonical project helper.
    """
    config = load_yaml(config_path)

    train_raw, val_raw, test_raw = load_candle_splits(
        data_dir,
    )

    train_split, val_split, test_split = (
        clean_candle_splits(
            train_raw,
            val_raw,
            test_raw,
        )
    )

    return (
        config,
        train_split,
        val_split,
        test_split,
    )


def limit_split_days(
    split: SplitDict,
    max_days: int | None,
) -> SplitDict:
    """
    Retain only the first max_days sessions for a fast smoke test.

    This preserves the original split boundary, asset ordering,
    channel ordering and within-session window construction.
    """
    if max_days is None:
        return split

    if max_days <= 0:
        raise ValueError(
            "--smoke-days must be greater than zero."
        )

    if "samples" not in split:
        raise KeyError(
            "Split does not contain a 'samples' field."
        )

    limited_split = dict(split)
    limited_split["samples"] = list(
        split["samples"][:max_days]
    )

    if not limited_split["samples"]:
        raise ValueError(
            "The limited split contains no sessions."
        )

    return limited_split

def fit_modern_tcn_run(
    base_config: ConfigDict,
    train_split: SplitDict,
    val_split: SplitDict,
    checkpoint_root: Path,
    run: Any,
) -> tuple[
    ModernTCNBaseline,
    Path,
    float,
]:
    """
    Fit one ModernTCN model for one W&B run.

    W&B-selected hyperparameters are applied before model creation.
    The run receives its own checkpoint directory, so separate
    default and sweep runs cannot overwrite each other.
    """
    run_config = apply_wandb_hyperparameters(
        config=base_config,
        run_config=run.config,
    )

    model = ModernTCNBaseline.from_config(
        run_config,
    )

    run.config.update(
        {
            "experiment_name": (
                model.experiment_name
            ),
            "input_channels": list(
                model.input_channels
            ),
            "target_channels": list(
                model.target_channels
            ),
            "target_input_indices": list(
                model.target_input_indices
            ),
            "num_input_channels": (
                model.num_input_channels
            ),
            "num_target_channels": (
                model.num_target_channels
            ),
            # Legacy alias.
            "num_channels": (
                model.num_input_channels
            ),
            "revin": model.revin,
            "loss_space": model.loss_space,
        },
        allow_val_change=True,
    )

    checkpoint_path = build_run_checkpoint_path(
        checkpoint_root=checkpoint_root,
        experiment_name=model.experiment_name,
        run_id=run.id,
    )

    start_time = perf_counter()

    model.fit(
        train_split=train_split,
        val_split=val_split,
        checkpoint_path=checkpoint_path,
        epoch_callback=make_wandb_epoch_callback(
            run,
        ),
    )

    training_duration_seconds = (
        perf_counter() - start_time
    )

    return (
        model,
        checkpoint_path,
        training_duration_seconds,
    )

def evaluate_modern_tcn_run(
    model: ModernTCNBaseline,
    evaluation_split: SplitDict,
    train_split: SplitDict,
    split_name: str,
) -> tuple[
    dict[str, Any],
    Any,
    dict[str, float],
]:
    """
    Evaluate the restored best checkpoint.

    Returns:
        predictions:
            Full project prediction result, including [B, H, N, C]
            predictions, targets and indexing metadata.
        metric_table:
            Long-form evaluation table with columns including metric,
            horizon, channel and value.
        wandb_metrics:
            Scalar horizon/channel metrics for the W&B run summary.
    """
    predictions = model.predict(
        evaluation_split,
    )

    evaluator = ForecastEvaluator(
        prediction_result=predictions,
        train_split=train_split,
    )

    metric_results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
    )

    metric_table = make_evaluation_table(
        metric_results=metric_results,
        horizons=evaluator.horizons,
        channels=evaluator.channels,
    )

    horizons = list(evaluator.horizons)
    channels = list(evaluator.channels)

    expected_shape = (
        len(horizons),
        len(channels),
    )

    wandb_metrics: dict[str, float] = {}

    for metric_name, metric_value in metric_results.items():
        metric_tensor = torch.as_tensor(
            metric_value,
        ).detach().cpu()

        if tuple(metric_tensor.shape) != expected_shape:
            raise ValueError(
                f"Metric {metric_name!r} returned shape "
                f"{tuple(metric_tensor.shape)}; expected "
                f"{expected_shape}."
            )

        for horizon_index, horizon in enumerate(horizons):
            for channel_index, channel in enumerate(channels):
                metric_key = (
                    f"{split_name}/{metric_name}/"
                    f"h{int(horizon)}/{channel}"
                )

                wandb_metrics[metric_key] = float(
                    metric_tensor[
                        horizon_index,
                        channel_index,
                    ].item()
                )

    return (
        predictions,
        metric_table,
        wandb_metrics,
    )


def save_evaluation_outputs(
    checkpoint_path: Path,
    split_name: str,
    predictions: dict[str, Any],
    metric_table: Any,
) -> tuple[Path, Path]:
    """
    Save predictions and the long-form evaluation table beside the
    run's best checkpoint.
    """
    output_dir = checkpoint_path.parent

    predictions_path = (
        output_dir / f"predictions_{split_name}.pt"
    )
    evaluation_table_path = (
        output_dir / f"evaluation_{split_name}.csv"
    )

    torch.save(
        predictions,
        predictions_path,
    )

    metric_table.to_csv(
        evaluation_table_path,
        index=False,
    )

    return (
        predictions_path,
        evaluation_table_path,
    )

def log_evaluation_outputs(
    run: Any,
    split_name: str,
    metric_table: Any,
    predictions_path: Path,
    evaluation_table_path: Path,
) -> None:
    """
    Log the evaluation table and preserve its saved output files in W&B.
    """
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}"
        )

    if not evaluation_table_path.is_file():
        raise FileNotFoundError(
            "Evaluation table file not found: "
            f"{evaluation_table_path}"
        )

    run.summary[
        f"{split_name}_predictions_path"
    ] = str(predictions_path)

    run.summary[
        f"{split_name}_evaluation_table_path"
    ] = str(evaluation_table_path)

    wandb_table = wandb.Table(
        dataframe=metric_table,
        log_mode="IMMUTABLE",
    )

    run.log(
        {
            f"{split_name}/evaluation_table": (
                wandb_table
            )
        }
    )

    artifact = wandb.Artifact(
        name=(
            f"modern-tcn-evaluation-{run.id}"
        ),
        type="evaluation",
        description=(
            "Predictions and evaluation metrics for one "
            "ModernTCN run."
        ),
        metadata={
            "wandb_run_id": run.id,
            "evaluation_split": split_name,
            "experiment_name": run.config[
                "experiment_name"
            ],
        },
    )

    artifact.add_file(
        local_path=str(predictions_path),
        name=predictions_path.name,
    )

    artifact.add_file(
        local_path=str(evaluation_table_path),
        name=evaluation_table_path.name,
    )

    run.log_artifact(artifact)

def record_run_summary(
    run: Any,
    model: ModernTCNBaseline,
    checkpoint_path: Path,
    training_duration_seconds: float,
    evaluation_metrics: Mapping[str, float],
) -> None:
    """
    Record final scalar results and reproducibility metadata in W&B.
    """
    if model.model is None:
        raise RuntimeError(
            "The ModernTCN model has not been constructed."
        )

    if model.best_epoch is None:
        raise RuntimeError(
            "No best epoch was recorded during training."
        )

    if model.best_validation_loss is None:
        raise RuntimeError(
            "No best validation loss was recorded during training."
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
        if parameter.requires_grad
    )

    run.summary["best_epoch"] = int(
        model.best_epoch
    )
    run.summary["best_validation_mse"] = float(
        model.best_validation_loss
    )
    run.summary["training_duration_seconds"] = float(
        training_duration_seconds
    )
    run.summary["total_parameters"] = int(
        total_parameters
    )
    run.summary["trainable_parameters"] = int(
        trainable_parameters
    )
    run.summary["checkpoint_path"] = str(
        checkpoint_path
    )
    run.summary["device"] = str(
        model.device
    )

    if model.num_assets is None:
        raise RuntimeError(
            "Model asset dimensions were not resolved."
        )

    if model.num_variables is None:
        raise RuntimeError(
            "Model variable dimensions were not resolved."
        )
    
    run.summary["experiment_name"] = (
        model.experiment_name
    )

    run.summary["variable_layout"] = (
        model.variable_layout
    )

    run.summary["input_channels"] = list(
        model.input_channels
    )

    run.summary["target_channels"] = list(
        model.target_channels
    )

    run.summary["num_input_channels"] = int(
        model.num_input_channels
    )

    run.summary["num_target_channels"] = int(
        model.num_target_channels
    )

    run.summary["target_input_indices"] = list(
        model.target_input_indices
    )

    # Legacy alias.
    run.summary["num_channels"] = int(
        model.num_input_channels
    )

    run.summary["num_assets"] = int(
        model.num_assets
    )

    run.summary["num_variables"] = int(
        model.num_variables
    )

    run.summary["revin"] = bool(
        model.revin
    )

    run.summary["loss_space"] = (
        model.loss_space
    )

    run.summary["normalisation_clip"] = (
        None
        if model.normaliser is None
        else bool(model.normaliser.clip)
    )

    for metric_name, metric_value in evaluation_metrics.items():
        run.summary[metric_name] = float(
            metric_value
        )

def log_checkpoint_artifact(
    run: Any,
    model: ModernTCNBaseline,
    checkpoint_path: Path,
) -> None:
    """
    Upload one run's best checkpoint as a W&B model artifact.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    if model.best_epoch is None:
        raise RuntimeError(
            "No best epoch was recorded during training."
        )

    if model.best_validation_loss is None:
        raise RuntimeError(
            "No best validation loss was recorded during training."
        )

    artifact = wandb.Artifact(
        name=f"modern-tcn-checkpoint-{run.id}",
        type="model",
        description=(
            "Best validation checkpoint for one ModernTCN run."
        ),
        metadata={
            "wandb_run_id": run.id,
            "best_epoch": int(model.best_epoch),
            "best_validation_mse": float(
                model.best_validation_loss
            ),
            "experiment_name": (
                model.experiment_name
            ),
            "variable_layout": (
                model.variable_layout
            ),
            "input_channels": list(
                model.input_channels
            ),
            "target_input_indices": list(
                model.target_input_indices
            ),
            "target_channels": list(
                model.target_channels
            ),
            "loss_space": model.loss_space,
        },
    )

    artifact.add_file(
        local_path=str(checkpoint_path),
        name="best_checkpoint.pt",
    )

    run.log_artifact(artifact)

def main() -> None:
    """
    Run one default ModernTCN experiment or one W&B sweep trial.
    """
    args = parse_arguments()

    (
        config_path,
        data_dir,
        checkpoint_root,
    ) = prepare_experiment_paths(args)

    (
        config,
        train_split,
        val_split,
        test_split,
    ) = load_experiment_inputs(
        config_path=config_path,
        data_dir=data_dir,
    )

    train_split = limit_split_days(
        train_split,
        args.smoke_days,
    )

    val_split = limit_split_days(
        val_split,
        args.smoke_days,
    )

    test_split = limit_split_days(
        test_split,
        args.smoke_days,
    )

    if args.smoke_days is not None:
        print(
            "Smoke-test data: "
            f"{len(train_split['samples'])} train session(s), "
            f"{len(val_split['samples'])} validation session(s), "
            f"{len(test_split['samples'])} test session(s)."
        )

    base_config = apply_cli_overrides(
        config=config,
        args=args,
    )

    wandb_config = build_wandb_config(
        config=base_config,
        data_dir=data_dir,
        evaluation_split=args.evaluation_split,
    )

    wandb_config["is_smoke_run"] = (
        args.smoke_days is not None
    )

    wandb_config["smoke_days_per_split"] = (
        args.smoke_days
    )
    
    evaluation_split = (
        val_split
        if args.evaluation_split == "val"
        else test_split
    )

    with wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=wandb_config,
        job_type="training",
    ) as run:
        if (
            run.sweep_id is not None
            and args.evaluation_split != "val"
        ):
            raise ValueError(
                "W&B sweep trials must evaluate on the "
                "validation split, not the test split."
            )

        # Use epoch as the x-axis and retain useful run-level
        # summaries for comparing default and sweep runs.
        run.define_metric("epoch")

        run.define_metric(
            "train/mse",
            step_metric="epoch",
            summary="min",
        )

        run.define_metric(
            "val/mse",
            step_metric="epoch",
            summary="min",
        )

        run.define_metric(
            "learning_rate",
            step_metric="epoch",
            summary="last",
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        (
            model,
            checkpoint_path,
            training_duration_seconds,
        ) = fit_modern_tcn_run(
            base_config=base_config,
            train_split=train_split,
            val_split=val_split,
            checkpoint_root=checkpoint_root,
            run=run,
        )

        (
            predictions,
            metric_table,
            evaluation_metrics,
        ) = evaluate_modern_tcn_run(
            model=model,
            evaluation_split=evaluation_split,
            train_split=train_split,
            split_name=args.evaluation_split,
        )

        (
            predictions_path,
            evaluation_table_path,
        ) = save_evaluation_outputs(
            checkpoint_path=checkpoint_path,
            split_name=args.evaluation_split,
            predictions=predictions,
            metric_table=metric_table,
        )

        log_evaluation_outputs(
            run=run,
            split_name=args.evaluation_split,
            metric_table=metric_table,
            predictions_path=predictions_path,
            evaluation_table_path=evaluation_table_path,
        )

        record_run_summary(
            run=run,
            model=model,
            checkpoint_path=checkpoint_path,
            training_duration_seconds=(
                training_duration_seconds
            ),
            evaluation_metrics=evaluation_metrics,
        )

        run.summary["pytorch_version"] = str(
            torch.__version__
        )

        run.summary["cuda_version"] = (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else "not_available"
        )

        if torch.cuda.is_available():
            run.summary["gpu_name"] = (
                torch.cuda.get_device_name(0)
            )

            run.summary["peak_gpu_memory_bytes"] = int(
                torch.cuda.max_memory_allocated()
            )

        log_checkpoint_artifact(
            run=run,
            model=model,
            checkpoint_path=checkpoint_path,
        )

        print(
            "ModernTCN run completed.\n"
            f"W&B run ID: {run.id}\n"
            f"Best epoch: {model.best_epoch}\n"
            "Best validation MSE: "
            f"{model.best_validation_loss:.8f}\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Predictions: {predictions_path}\n"
            f"Evaluation table: {evaluation_table_path}"
        )


if __name__ == "__main__":
    main()
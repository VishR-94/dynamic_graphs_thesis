from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
)
from src.data.token_graph_dataset import (
    load_origin_aligned_token_cache,
    select_future_horizons,
)
from src.evaluation.metrics import ForecastEvaluator
from src.models.dynamic_graph.config import (
    load_dynamic_graph_config,
)
from src.models.dynamic_graph.future_predictor import (
    FutureTokenPrediction,
    FutureTokenLoss,
    compute_future_token_loss,
)
from src.models.dynamic_graph.graph_learners import (
    MTGNNStaticGraphLearner,
)
from src.models.dynamic_graph.model import (
    DynamicGraphTokenForecaster,
)
from src.models.kronos_tokenizer import (
    KRONOS_TOKENIZER_CHANNELS,
    KronosTokenizerAdapter,
)
from src.utils.config import load_yaml


OHLCV_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit a tiny real token-cache slice, generate and decode "
            "a complete 60-step future path, then run ForecastEvaluator."
        )
    )

    parser.add_argument(
        "--train-cache",
        type=Path,
        required=True,
        help="Path to origin_aligned_train_tokens.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing train.pt, val.pt and test.pt.",
    )
    parser.add_argument(
        "--dynamic-config",
        type=Path,
        default=Path("configs/dynamic_graph.yaml"),
    )
    parser.add_argument(
        "--forecasting-config",
        type=Path,
        default=Path("configs/forecasting.yaml"),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="structured_parallel_uniform",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=2,
        help="Number of cached windows deliberately overfit.",
    )
    parser.add_argument(
        "--num-assets",
        type=int,
        default=8,
        help="Number of leading assets used by this smoke test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-3,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--minimum-loss-reduction",
        type=float,
        default=0.20,
        help=(
            "Required fractional reduction in deterministic teacher-forced "
            "loss. 0.20 means at least 20 percent."
        ),
    )
    parser.add_argument(
        "--decode-series-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def retain_first_assets(
    split: Mapping[str, Any],
    num_assets: int,
) -> dict[str, Any]:
    total_assets = len(split["asset_cols"])

    if not 2 <= num_assets <= total_assets:
        raise ValueError(
            "num_assets must lie between 2 and the available asset count. "
            f"Received {num_assets}; available {total_assets}."
        )

    reduced = dict(split)
    reduced["asset_cols"] = list(split["asset_cols"][:num_assets])
    reduced["samples"] = [
        (
            x_day[:, :num_assets, :].contiguous(),
            auxiliary,
            day,
        )
        for x_day, auxiliary, day in split["samples"]
    ]

    if "F" in reduced:
        reduced["F"] = num_assets

    return reduced


def build_smoke_experiment_config(
    path: Path,
    *,
    preset: str,
    num_assets: int,
) -> dict[str, Any]:
    experiment_config = deepcopy(
        load_dynamic_graph_config(
            path,
            preset=preset,
        )
    )

    model_config = experiment_config["models"]["dynamic_graph"]
    graph_config = model_config["graph"]

    model_config["num_nodes"] = int(num_assets)

    if graph_config["type"] != "mtgnn_static":
        raise ValueError(
            "The Day-1 vertical slice must use the common "
            "MTGNN static graph."
        )

    maximum_sources = (
        num_assets
        if bool(graph_config["add_self_loops"])
        else num_assets - 1
    )

    graph_config["mtgnn_top_k"] = min(
        int(graph_config["mtgnn_top_k"]),
        maximum_sources,
    )

    return experiment_config


def validate_cache_contract(
    cache: Mapping[str, Any],
    *,
    num_windows: int,
    num_assets: int,
    experiment_config: Mapping[str, Any],
) -> None:
    available_windows = int(cache["context_tokens"].shape[0])
    available_assets = int(cache["context_tokens"].shape[2])

    if not 1 <= num_windows <= available_windows:
        raise ValueError(
            "num_windows lies outside the cached range. "
            f"Received {num_windows}; available {available_windows}."
        )

    if not 2 <= num_assets <= available_assets:
        raise ValueError(
            "num_assets lies outside the cached range. "
            f"Received {num_assets}; available {available_assets}."
        )

    model_config = experiment_config["models"]["dynamic_graph"]

    expected_context = int(model_config["context_length"])
    expected_prediction = int(model_config["heads"]["prediction_length"])
    expected_horizons = tuple(
        int(value)
        for value in model_config["heads"]["evaluation_horizons"]
    )

    if int(cache["context_length"]) != expected_context:
        raise ValueError(
            "Cache and model context lengths differ: "
            f"{cache['context_length']} versus {expected_context}."
        )

    if int(cache["prediction_length"]) != expected_prediction:
        raise ValueError(
            "Cache and model prediction lengths differ: "
            f"{cache['prediction_length']} versus {expected_prediction}."
        )

    if tuple(cache["evaluation_horizons"]) != expected_horizons:
        raise ValueError(
            "Cache and model evaluation horizons differ."
        )

    if tuple(cache["input_channels"]) != OHLCV_CHANNELS:
        raise ValueError(
            "The cache input channels are not canonical OHLCV."
        )

    if tuple(cache["target_channels"]) != OHLCV_CHANNELS:
        raise ValueError(
            "The cache target channels are not canonical OHLCV."
        )

    if tuple(cache["tokenizer_channels"]) != (
        KRONOS_TOKENIZER_CHANNELS
    ):
        raise ValueError(
            "The cache tokenizer channels do not match the frozen "
            "Kronos tokenizer contract."
        )


def build_loss_prediction(
    output: Any,
) -> FutureTokenPrediction:
    return FutureTokenPrediction(
        future_hidden=output.future_hidden,
        s1_logits=output.s1_logits,
        s2_logits=output.s2_logits,
        selected_s1=output.s1_logits.argmax(dim=-1),
        selected_s2=output.s2_logits.argmax(dim=-1),
    )


def compute_model_loss(
    model: DynamicGraphTokenForecaster,
    output: Any,
    target_s1: Tensor,
    target_s2: Tensor,
) -> FutureTokenLoss:
    return compute_future_token_loss(
        build_loss_prediction(output),
        target_s1,
        target_s2,
        loss_config=model.config.loss,
        evaluation_horizons=(
            model.config.heads.evaluation_horizons
        ),
        s2_loss_weight=(
            model.config.heads.s2_loss_weight
        ),
    )


def teacher_forced_diagnostics(
    model: DynamicGraphTokenForecaster,
    context_tokens: Tensor,
    target_s1: Tensor,
    target_s2: Tensor,
) -> tuple[Any, FutureTokenLoss, float, float]:
    model.eval()

    with torch.inference_mode():
        output = model(
            context_tokens,
            target_s1=target_s1,
            target_s2=target_s2,
        )
        loss = compute_model_loss(
            model,
            output,
            target_s1,
            target_s2,
        )

    s1_accuracy = float(
        (
            output.s1_logits.argmax(dim=-1)
            == target_s1
        )
        .to(torch.float32)
        .mean()
        .item()
    )

    s2_accuracy = float(
        (
            output.s2_logits.argmax(dim=-1)
            == target_s2
        )
        .to(torch.float32)
        .mean()
        .item()
    )

    return (
        output,
        loss,
        s1_accuracy,
        s2_accuracy,
    )


def decoded_invalid_candle_rate(
    decoded_ohlcv: Tensor,
) -> float:
    if (
        decoded_ohlcv.ndim != 4
        or decoded_ohlcv.shape[-1] != 5
    ):
        raise ValueError(
            "decoded_ohlcv must have shape [B, P, N, 5]."
        )

    open_values = decoded_ohlcv[..., 0]
    high_values = decoded_ohlcv[..., 1]
    low_values = decoded_ohlcv[..., 2]
    close_values = decoded_ohlcv[..., 3]
    volume_values = decoded_ohlcv[..., 4]

    invalid = (
        ~torch.isfinite(decoded_ohlcv).all(dim=-1)
        | (open_values <= 0)
        | (high_values <= 0)
        | (low_values <= 0)
        | (close_values <= 0)
        | (
            high_values
            < torch.maximum(
                open_values,
                close_values,
            )
        )
        | (
            low_values
            > torch.minimum(
                open_values,
                close_values,
            )
        )
        | (high_values < low_values)
        | (volume_values < 0)
    )

    return float(
        invalid
        .to(torch.float64)
        .mean()
        .mul(100.0)
        .item()
    )


def format_metric_values(
    values: Tensor,
    horizons: list[int],
) -> str:
    flat = values.detach().cpu().reshape(-1)

    if flat.numel() != len(horizons):
        return str(values.detach().cpu())

    return ", ".join(
        f"h={horizon}: {float(value):.6g}"
        for horizon, value in zip(
            horizons,
            flat,
            strict=True,
        )
    )


def main() -> None:
    args = build_argument_parser().parse_args()

    if args.num_windows <= 0:
        raise ValueError("--num-windows must be positive.")

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be positive.")

    if not 0.0 < args.minimum_loss_reduction < 1.0:
        raise ValueError(
            "--minimum-loss-reduction must lie in (0, 1)."
        )

    if args.decode_series_batch_size <= 0:
        raise ValueError(
            "--decode-series-batch-size must be positive."
        )

    set_seed(args.seed)
    device = resolve_device(args.device)

    cache = load_origin_aligned_token_cache(
        args.train_cache
    )

    experiment_config = build_smoke_experiment_config(
        args.dynamic_config,
        preset=args.preset,
        num_assets=args.num_assets,
    )

    validate_cache_contract(
        cache,
        num_windows=args.num_windows,
        num_assets=args.num_assets,
        experiment_config=experiment_config,
    )

    window_slice = slice(
        0,
        args.num_windows,
    )
    asset_slice = slice(
        0,
        args.num_assets,
    )

    context_tokens = (
        cache["context_tokens"]
        [window_slice, :, asset_slice, :]
        .to(device=device, dtype=torch.long)
    )
    target_s1 = (
        cache["target_s1"]
        [window_slice, :, asset_slice]
        .to(device=device, dtype=torch.long)
    )
    target_s2 = (
        cache["target_s2"]
        [window_slice, :, asset_slice]
        .to(device=device, dtype=torch.long)
    )

    model = DynamicGraphTokenForecaster.from_config(
        experiment_config
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Device:", device)
    print("Preset:", args.preset)
    print("Smoke-test windows:", args.num_windows)
    print("Smoke-test assets:", args.num_assets)
    print("Trainable parameters:", f"{parameter_count:,}")

    (
        _,
        initial_loss,
        initial_s1_accuracy,
        initial_s2_accuracy,
    ) = teacher_forced_diagnostics(
        model,
        context_tokens,
        target_s1,
        target_s2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    model.train()

    last_loss: FutureTokenLoss | None = None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        output = model(
            context_tokens,
            target_s1=target_s1,
            target_s2=target_s2,
        )

        last_loss = compute_model_loss(
            model,
            output,
            target_s1,
            target_s2,
        )

        last_loss.total.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=args.gradient_clip_norm,
        )

        optimizer.step()

        if (
            step == 1
            or step == args.steps
            or step % args.log_every == 0
        ):
            print(
                f"step {step:>3}/{args.steps}: "
                f"total={last_loss.total.item():.6f} "
                f"s1={last_loss.s1.item():.6f} "
                f"s2={last_loss.s2.item():.6f}"
            )

    if last_loss is None:
        raise RuntimeError("No optimisation step was executed.")

    (
        final_output,
        final_loss,
        final_s1_accuracy,
        final_s2_accuracy,
    ) = teacher_forced_diagnostics(
        model,
        context_tokens,
        target_s1,
        target_s2,
    )

    relative_loss_reduction = (
        initial_loss.total.item()
        - final_loss.total.item()
    ) / initial_loss.total.item()

    if relative_loss_reduction < args.minimum_loss_reduction:
        raise AssertionError(
            "The tiny real-cache batch did not overfit sufficiently. "
            f"Initial loss={initial_loss.total.item():.6f}; "
            f"final loss={final_loss.total.item():.6f}; "
            f"reduction={100 * relative_loss_reduction:.2f}%."
        )

    graph_learner = model.graph_learners[0]

    if not isinstance(
        graph_learner,
        MTGNNStaticGraphLearner,
    ):
        raise AssertionError(
            "The vertical slice did not use MTGNNStaticGraphLearner."
        )

    graph_gradient = (
        graph_learner.embedding_1[0].weight.grad
    )

    if (
        graph_gradient is None
        or not torch.isfinite(graph_gradient).all()
        or graph_gradient.abs().sum().item() == 0.0
    ):
        raise AssertionError(
            "The token loss did not produce a valid gradient in the "
            "MTGNN graph constructor."
        )

    model.eval()

    with torch.inference_mode():
        generated = model.generate(
            context_tokens,
            token_selection="argmax",
        )

    generated_tokens = generated.token_ids.detach().cpu()

    generated_s1_accuracy = float(
        (
            generated_tokens[..., 0]
            == target_s1.detach().cpu()
        )
        .to(torch.float32)
        .mean()
        .item()
    )

    generated_s2_accuracy = float(
        (
            generated_tokens[..., 1]
            == target_s2.detach().cpu()
        )
        .to(torch.float32)
        .mean()
        .item()
    )

    selected_graph = generated.forecast.graph.selected

    if selected_graph is None:
        raise AssertionError(
            "The generated forecast did not expose its selected graph."
        )

    expected_graph_shape = (
        args.num_windows,
        model.config.graph.num_heads,
        args.num_assets,
        args.num_assets,
    )

    if tuple(selected_graph.shape) != expected_graph_shape:
        raise AssertionError(
            "Unexpected selected graph shape. "
            f"Expected {expected_graph_shape}; "
            f"received {tuple(selected_graph.shape)}."
        )

    row_sums = selected_graph.sum(dim=-1)

    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise AssertionError(
            "The selected graph is not row normalised."
        )

    nonzero_sources = (
        selected_graph > 0
    ).sum(dim=-1)

    expected_top_k = model.config.graph.mtgnn_top_k

    if not torch.equal(
        nonzero_sources,
        torch.full_like(
            nonzero_sources,
            expected_top_k,
        ),
    ):
        raise AssertionError(
            "The selected MTGNN graph does not retain the expected "
            "number of source neighbours."
        )

    forecasting_config = load_yaml(
        args.forecasting_config
    )

    tokenizer = KronosTokenizerAdapter.from_config(
        forecasting_config,
        series_batch_size=(
            args.decode_series_batch_size
        ),
    ).load()

    context_mean = (
        cache["context_mean"]
        [window_slice, asset_slice, :]
    )
    context_std = (
        cache["context_std"]
        [window_slice, asset_slice, :]
    )

    decoded_future = tokenizer.decode_token_path(
        context_tokens.detach().cpu(),
        generated_tokens,
        mean=context_mean,
        std=context_std,
        series_batch_size=(
            args.decode_series_batch_size
        ),
        return_full_path=False,
    )

    expected_decoded_shape = (
        args.num_windows,
        model.config.prediction_length,
        args.num_assets,
        5,
    )

    if tuple(decoded_future.shape) != expected_decoded_shape:
        raise AssertionError(
            "Unexpected decoded future shape. "
            f"Expected {expected_decoded_shape}; "
            f"received {tuple(decoded_future.shape)}."
        )

    horizons = list(
        model.config.heads.evaluation_horizons
    )

    decoded_evaluation = select_future_horizons(
        decoded_future,
        horizons=horizons,
    )

    close_index = OHLCV_CHANNELS.index(
        "close"
    )

    y_pred = decoded_evaluation[
        ...,
        close_index:close_index + 1,
    ].to(torch.float32)

    evaluation_true = (
        cache["evaluation_true"]
        [window_slice, :, asset_slice, :]
        .to(torch.float32)
    )

    y_true = evaluation_true[
        ...,
        close_index:close_index + 1,
    ]

    last_context_target = (
        cache["last_context_target"]
        [window_slice, asset_slice, :]
        .to(torch.float32)
        [
            ...,
            close_index:close_index + 1,
        ]
    )

    evaluation_indices = torch.tensor(
        cache["evaluation_indices"],
        dtype=torch.long,
    )

    dense_target_indices = (
        cache["target_indices"]
        [window_slice]
        .to(torch.long)
    )

    selected_target_indices = (
        dense_target_indices.index_select(
            dim=1,
            index=evaluation_indices,
        )
    )

    prediction_result = {
        "y_pred": y_pred,
        "y_true": y_true,
        "last_context_target": (
            last_context_target
        ),
        "sample_idx": (
            cache["sample_idx"]
            [window_slice]
            .to(torch.long)
        ),
        "origin_idx": (
            cache["origin_idx"]
            [window_slice]
            .to(torch.long)
        ),
        "target_indices": (
            selected_target_indices
        ),
        "channels": ["close"],
        "horizons": horizons,
        "asset_cols": list(
            cache["asset_cols"][:args.num_assets]
        ),
        "output_space": "raw",
    }

    train_raw, val_raw, test_raw = (
        load_candle_splits(
            args.data_dir
        )
    )
    train_split, _, _ = clean_candle_splits(
        train_raw,
        val_raw,
        test_raw,
    )
    train_split = retain_first_assets(
        train_split,
        args.num_assets,
    )

    if list(train_split["asset_cols"]) != (
        prediction_result["asset_cols"]
    ):
        raise AssertionError(
            "Token-cache and raw-training asset ordering differ."
        )

    evaluator = ForecastEvaluator(
        prediction_result=prediction_result,
        train_split=train_split,
    )

    metric_results = evaluator.evaluate(
        metrics=evaluator.available_metrics,
        reduce_dims=(0, 2),
        bootstrap=False,
    )

    for metric_name, values in metric_results.items():
        expected_metric_shape = (
            len(horizons),
            1,
        )

        if tuple(values.shape) != expected_metric_shape:
            raise AssertionError(
                f"Metric {metric_name!r} has shape "
                f"{tuple(values.shape)}; expected "
                f"{expected_metric_shape}."
            )

    invalid_candle_rate = decoded_invalid_candle_rate(
        decoded_future
    )

    nonpositive_close_count = int(
        (y_pred <= 0).sum().item()
    )

    print("\nOverfit diagnostics")
    print("-------------------")
    print(
        "Initial teacher-forced loss:",
        f"{initial_loss.total.item():.6f}",
    )
    print(
        "Final teacher-forced loss:",
        f"{final_loss.total.item():.6f}",
    )
    print(
        "Loss reduction:",
        f"{100 * relative_loss_reduction:.2f}%",
    )
    print(
        "Teacher-forced s1 accuracy:",
        f"{initial_s1_accuracy:.4f} -> {final_s1_accuracy:.4f}",
    )
    print(
        "Teacher-forced s2 accuracy:",
        f"{initial_s2_accuracy:.4f} -> {final_s2_accuracy:.4f}",
    )
    print(
        "Free-running s1 accuracy:",
        f"{generated_s1_accuracy:.4f}",
    )
    print(
        "Free-running s2 accuracy:",
        f"{generated_s2_accuracy:.4f}",
    )
    print(
        "MTGNN graph-gradient norm:",
        f"{graph_gradient.norm().item():.6f}",
    )

    print("\nDecoded forecast")
    print("----------------")
    print("Dense future shape:", tuple(decoded_future.shape))
    print("Evaluator y_pred shape:", tuple(y_pred.shape))
    print(
        "Invalid decoded candle rate:",
        f"{invalid_candle_rate:.3f}%",
    )
    print(
        "Non-positive predicted Close count:",
        nonpositive_close_count,
    )

    print("\nForecastEvaluator metrics")
    print("-------------------------")

    for metric_name in evaluator.available_metrics:
        print(
            metric_name,
            "->",
            format_metric_values(
                metric_results[metric_name],
                horizons,
            ),
        )

    print(
        "\nDYNAMIC GRAPH REAL-CACHE VERTICAL SLICE PASSED"
    )


if __name__ == "__main__":
    main()

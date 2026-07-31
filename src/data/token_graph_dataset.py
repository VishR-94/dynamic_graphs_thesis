from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.data.data_generator import WindowedCandleDataset
from src.models.kronos_tokenizer import (
    KronosTokenizerAdapter,
)


OHLCV_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

TOKENIZER_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


@dataclass(frozen=True)
class OriginAlignedTokenBatch:
    """Tokenised forecasting windows aligned to one forecast origin.

    Shapes:
        context_tokens:
            [B, C, N, 2]

        target_s1:
            [B, P, N]

        target_s2:
            [B, P, N]

        context_mean/context_std:
            [B, N, 6]

        context_clipping_mask:
            [B, C, N, 6]

        future_clipping_mask:
            [B, P, N, 6]

        decoded_oracle_future:
            [B, P, N, 5] when requested, otherwise None.

    Every future target is encoded as its actual position in the
    complete C + P sequence, using statistics estimated only from the
    C observed context bars.
    """

    context_tokens: Tensor
    target_s1: Tensor
    target_s2: Tensor
    context_mean: Tensor
    context_std: Tensor
    context_clipping_mask: Tensor
    future_clipping_mask: Tensor
    decoded_oracle_future: Tensor | None = None

    @property
    def future_tokens(self) -> Tensor:
        """Return future token pairs with shape [B, P, N, 2]."""
        return torch.stack(
            (
                self.target_s1,
                self.target_s2,
            ),
            dim=-1,
        )

    @property
    def batch_size(self) -> int:
        return int(self.context_tokens.shape[0])

    @property
    def context_length(self) -> int:
        return int(self.context_tokens.shape[1])

    @property
    def prediction_length(self) -> int:
        return int(self.target_s1.shape[1])

    @property
    def num_assets(self) -> int:
        return int(self.context_tokens.shape[2])

    def validate(self) -> None:
        if (
            self.context_tokens.ndim != 4
            or self.context_tokens.shape[-1] != 2
        ):
            raise ValueError(
                "context_tokens must have shape [B, C, N, 2]."
            )

        batch_size, context_length, num_assets, _ = (
            self.context_tokens.shape
        )

        expected_target_shape = (
            batch_size,
            self.target_s1.shape[1],
            num_assets,
        )

        if tuple(self.target_s1.shape) != expected_target_shape:
            raise ValueError(
                "target_s1 must have shape [B, P, N]."
            )

        if tuple(self.target_s2.shape) != expected_target_shape:
            raise ValueError(
                "target_s2 must have shape [B, P, N]."
            )

        expected_stats_shape = (
            batch_size,
            num_assets,
            6,
        )

        if tuple(self.context_mean.shape) != expected_stats_shape:
            raise ValueError(
                "context_mean must have shape [B, N, 6]."
            )

        if tuple(self.context_std.shape) != expected_stats_shape:
            raise ValueError(
                "context_std must have shape [B, N, 6]."
            )

        expected_context_mask = (
            batch_size,
            context_length,
            num_assets,
            6,
        )

        if tuple(self.context_clipping_mask.shape) != (
            expected_context_mask
        ):
            raise ValueError(
                "context_clipping_mask has an unexpected shape."
            )

        expected_future_mask = (
            batch_size,
            self.prediction_length,
            num_assets,
            6,
        )

        if tuple(self.future_clipping_mask.shape) != (
            expected_future_mask
        ):
            raise ValueError(
                "future_clipping_mask has an unexpected shape."
            )

        for name, values in (
            ("context_tokens", self.context_tokens),
            ("target_s1", self.target_s1),
            ("target_s2", self.target_s2),
        ):
            if values.dtype not in {
                torch.int16,
                torch.int32,
                torch.int64,
                torch.long,
            }:
                raise TypeError(
                    f"{name} must use an integer dtype."
                )

            if (
                values.min().item() < 0
                or values.max().item() >= 1024
            ):
                raise ValueError(
                    f"{name} contains IDs outside [0, 1023]."
                )

        if not torch.isfinite(self.context_mean).all():
            raise ValueError(
                "context_mean contains non-finite values."
            )

        if not torch.isfinite(self.context_std).all():
            raise ValueError(
                "context_std contains non-finite values."
            )

        if torch.any(self.context_std < 0):
            raise ValueError(
                "context_std contains negative values."
            )

        if self.decoded_oracle_future is not None:
            expected_decoded_shape = (
                batch_size,
                self.prediction_length,
                num_assets,
                5,
            )

            if tuple(self.decoded_oracle_future.shape) != (
                expected_decoded_shape
            ):
                raise ValueError(
                    "decoded_oracle_future has an unexpected shape."
                )

            if not torch.isfinite(
                self.decoded_oracle_future
            ).all():
                raise ValueError(
                    "decoded_oracle_future contains non-finite values."
                )


def _ensure_batched_ohlcv(
    values: Tensor,
    *,
    name: str,
) -> Tensor:
    values = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if values.ndim == 3:
        values = values.unsqueeze(0)

    if (
        values.ndim != 4
        or values.shape[-1] != 5
    ):
        raise ValueError(
            f"{name} must have shape [B, T, N, 5] "
            f"or [T, N, 5]. Received {tuple(values.shape)}."
        )

    if not torch.isfinite(values).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return (
        values
        .detach()
        .cpu()
        .contiguous()
    )


def build_origin_aligned_token_batch(
    tokenizer: KronosTokenizerAdapter,
    context: Tensor,
    future: Tensor,
    *,
    series_batch_size: int | None = None,
    verify_prefix_parity: bool = True,
    decode_oracle_future: bool = False,
) -> OriginAlignedTokenBatch:
    """Encode a complete context-plus-future forecasting path.

    The future values are labels only. They are never used to estimate
    normalisation statistics. The official tokenizer receives one
    ordered sequence:

        context positions 0 ... C-1
        future positions  C ... C+P-1

    Prefix parity checks that appending the future labels does not
    alter the context IDs produced by the causal tokenizer encoder.
    """
    if not isinstance(
        tokenizer,
        KronosTokenizerAdapter,
    ):
        raise TypeError(
            "tokenizer must be a KronosTokenizerAdapter."
        )

    context = _ensure_batched_ohlcv(
        context,
        name="context",
    )

    future = _ensure_batched_ohlcv(
        future,
        name="future",
    )

    if (
        context.shape[0] != future.shape[0]
        or context.shape[2] != future.shape[2]
    ):
        raise ValueError(
            "context and future batch/asset dimensions must match."
        )

    prepared = tokenizer.prepare_forecast_path(
        context,
        future,
        channels=OHLCV_CHANNELS,
    )

    full_batch = tokenizer.encode_normalised(
        prepared.normalised,
        mean=prepared.mean,
        std=prepared.std,
        series_batch_size=series_batch_size,
    )

    context_length = int(context.shape[1])
    prediction_length = int(future.shape[1])

    full_context_tokens = (
        full_batch.token_ids[
            :,
            :context_length,
        ]
        .contiguous()
    )

    if verify_prefix_parity:
        context_only_batch = tokenizer.encode_normalised(
            prepared.context,
            mean=prepared.mean,
            std=prepared.std,
            series_batch_size=series_batch_size,
        )

        mismatch_mask = (
            context_only_batch.token_ids
            != full_context_tokens
        )

        mismatch_count = int(
            mismatch_mask.sum().item()
        )

        if mismatch_count != 0:
            total = int(
                mismatch_mask.numel()
            )

            raise RuntimeError(
                "Kronos prefix parity failed: "
                f"{mismatch_count}/{total} context token IDs changed "
                "when the future suffix was appended. The forecasting "
                "cache must not be generated until this is resolved."
            )

    future_token_ids = (
        full_batch.token_ids[
            :,
            context_length:
            context_length + prediction_length,
        ]
        .contiguous()
    )

    if tuple(future_token_ids.shape[:3]) != (
        future.shape[0],
        prediction_length,
        future.shape[2],
    ):
        raise RuntimeError(
            "Unexpected future token shape after splitting the "
            "complete token path."
        )

    decoded_oracle = None

    if decode_oracle_future:
        decoded_oracle = tokenizer.decode_token_path(
            full_context_tokens,
            future_token_ids,
            mean=prepared.mean,
            std=prepared.std,
            series_batch_size=series_batch_size,
            return_full_path=False,
        )

    result = OriginAlignedTokenBatch(
        context_tokens=full_context_tokens.to(
            torch.int16
        ),
        target_s1=future_token_ids[
            ...,
            0,
        ].to(torch.int16),
        target_s2=future_token_ids[
            ...,
            1,
        ].to(torch.int16),
        context_mean=prepared.mean.to(
            torch.float32
        ),
        context_std=prepared.std.to(
            torch.float32
        ),
        context_clipping_mask=(
            prepared.context_clipping_mask
        ),
        future_clipping_mask=(
            prepared.future_clipping_mask
        ),
        decoded_oracle_future=decoded_oracle,
    )

    result.validate()
    return result


def clipping_rate_by_channel(
    batch: OriginAlignedTokenBatch,
    *,
    future: bool = True,
) -> Tensor:
    """Return clipping percentages for the six tokenizer channels."""
    batch.validate()

    mask = (
        batch.future_clipping_mask
        if future
        else batch.context_clipping_mask
    )

    return (
        mask
        .to(torch.float64)
        .mean(dim=(0, 1, 2))
        .mul(100.0)
    )


def future_clipping_rate_by_step_channel(
    batch: OriginAlignedTokenBatch,
) -> Tensor:
    """Return future clipping percentages with shape [P, 6]."""
    batch.validate()

    return (
        batch.future_clipping_mask
        .to(torch.float64)
        .mean(dim=(0, 2))
        .mul(100.0)
    )


def select_future_horizons(
    values: Tensor,
    horizons: Sequence[int] = (
        1,
        5,
        15,
        30,
        60,
    ),
) -> Tensor:
    """Select one-indexed future horizons along dimension 1."""
    values = torch.as_tensor(values)

    if values.ndim < 2:
        raise ValueError(
            "values must have a future-position dimension at axis 1."
        )

    resolved = tuple(
        int(horizon)
        for horizon in horizons
    )

    if not resolved:
        raise ValueError(
            "At least one horizon is required."
        )

    if min(resolved) < 1:
        raise ValueError(
            "Horizons are one-indexed and must be positive."
        )

    if max(resolved) > values.shape[1]:
        raise ValueError(
            "A requested horizon exceeds the available future path."
        )

    indices = torch.tensor(
        [
            horizon - 1
            for horizon in resolved
        ],
        dtype=torch.long,
        device=values.device,
    )

    return values.index_select(
        dim=1,
        index=indices,
    )


def oracle_path_smoke_metrics(
    batch: OriginAlignedTokenBatch,
    raw_future: Tensor,
    *,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Compute compact diagnostics for a true-token decoded path.

    These are smoke-test diagnostics, not a replacement for the
    project's canonical ForecastEvaluator.
    """
    if batch.decoded_oracle_future is None:
        raise ValueError(
            "The batch was built with decode_oracle_future=False."
        )

    raw_future = _ensure_batched_ohlcv(
        raw_future,
        name="raw_future",
    ).to(torch.float64)

    decoded = batch.decoded_oracle_future.to(
        torch.float64
    )

    if tuple(decoded.shape) != tuple(
        raw_future.shape
    ):
        raise ValueError(
            "Decoded and raw future paths are not aligned."
        )

    close_idx = OHLCV_CHANNELS.index(
        "close"
    )
    volume_idx = OHLCV_CHANNELS.index(
        "volume"
    )

    true_close = raw_future[
        ...,
        close_idx,
    ].clamp_min(eps)

    decoded_close = decoded[
        ...,
        close_idx,
    ].clamp_min(eps)

    close_error_bps = (
        torch.log(
            decoded_close
            / true_close
        )
        .abs()
        .mul(10_000.0)
    )

    true_returns = torch.log(
        true_close[:, 1:]
        / true_close[:, :-1]
    )

    decoded_returns = torch.log(
        decoded_close[:, 1:]
        / decoded_close[:, :-1]
    )

    return_error_bps = (
        decoded_returns
        - true_returns
    ).abs().mul(10_000.0)

    x = decoded_returns.reshape(-1)
    y = true_returns.reshape(-1)

    x = x - x.mean()
    y = y - y.mean()

    denominator = torch.sqrt(
        x.square().sum()
        * y.square().sum()
    )

    if denominator > 0:
        return_correlation = float(
            (
                (x * y).sum()
                / denominator
            ).item()
        )
    else:
        return_correlation = float("nan")

    true_volatility = true_returns.std(
        unbiased=False
    )

    decoded_volatility = decoded_returns.std(
        unbiased=False
    )

    if true_volatility > eps:
        volatility_ratio = float(
            (
                decoded_volatility
                / true_volatility
            ).item()
        )
    else:
        volatility_ratio = float("nan")

    open_values = decoded[..., 0]
    high_values = decoded[..., 1]
    low_values = decoded[..., 2]
    close_values = decoded[..., 3]
    volume_values = decoded[
        ...,
        volume_idx,
    ]

    invalid = (
        ~torch.isfinite(decoded).all(dim=-1)
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

    return {
        "close_median_abs_error_bps": float(
            close_error_bps.median().item()
        ),
        "close_p95_abs_error_bps": float(
            torch.quantile(
                close_error_bps,
                q=0.95,
            ).item()
        ),
        "return_mae_bps": float(
            return_error_bps.mean().item()
        ),
        "return_pearson": (
            return_correlation
        ),
        "return_volatility_ratio": (
            volatility_ratio
        ),
        "invalid_candle_rate_percent": float(
            invalid
            .to(torch.float64)
            .mean()
            .mul(100.0)
            .item()
        ),
    }


# ==================================================================
# Origin-aligned forecasting-cache generation
# ==================================================================

ORIGIN_ALIGNED_CACHE_VERSION = 2
SUPPORTED_ORIGIN_ALIGNED_CACHE_VERSIONS = (1, 2)


def validate_origin_aligned_token_cache(
    cache: Mapping[str, Any],
) -> None:
    """Validate a saved real-data forecasting token cache.

    The cache is model-facing and self-contained:

        context_tokens:
            [W, C, N, 2]

        target_s1 / target_s2:
            [W, P, N]

        context_mean / context_std:
            [W, N, 6]

        evaluation_true:
            [W, H, N, 5]

        last_context_target:
            [W, N, 5]
    """
    required = {
        "format_version",
        "context_tokens",
        "target_s1",
        "target_s2",
        "context_mean",
        "context_std",
        "evaluation_true",
        "last_context_target",
        "sample_idx",
        "origin_idx",
        "target_indices",
        "dates",
        "asset_cols",
        "input_channels",
        "target_channels",
        "context_length",
        "prediction_length",
        "dense_horizons",
        "evaluation_horizons",
        "evaluation_indices",
        "tokenizer_channels",
        "future_clipping_rate_percent_by_step_channel",
        "future_clipping_rate_percent_by_channel",
        "context_clipping_rate_percent_by_step_channel",
        "context_clipping_rate_percent_by_channel",
    }

    missing = required - set(cache)

    if missing:
        raise KeyError(
            "Origin-aligned token cache is missing keys: "
            f"{sorted(missing)}."
        )

    format_version = int(cache["format_version"])

    if format_version not in (
        SUPPORTED_ORIGIN_ALIGNED_CACHE_VERSIONS
    ):
        raise ValueError(
            "Unsupported origin-aligned cache version."
        )

    context_tokens = torch.as_tensor(
        cache["context_tokens"]
    )
    target_s1 = torch.as_tensor(
        cache["target_s1"]
    )
    target_s2 = torch.as_tensor(
        cache["target_s2"]
    )
    context_mean = torch.as_tensor(
        cache["context_mean"]
    )
    context_std = torch.as_tensor(
        cache["context_std"]
    )
    evaluation_true = torch.as_tensor(
        cache["evaluation_true"]
    )
    last_context_target = torch.as_tensor(
        cache["last_context_target"]
    )
    sample_idx = torch.as_tensor(
        cache["sample_idx"]
    )
    origin_idx = torch.as_tensor(
        cache["origin_idx"]
    )
    target_indices = torch.as_tensor(
        cache["target_indices"]
    )

    if (
        context_tokens.ndim != 4
        or context_tokens.shape[-1] != 2
    ):
        raise ValueError(
            "context_tokens must have shape [W, C, N, 2]."
        )

    num_windows, context_length, num_assets, _ = (
        context_tokens.shape
    )

    prediction_length = int(
        cache["prediction_length"]
    )

    evaluation_horizons = tuple(
        int(value)
        for value in cache["evaluation_horizons"]
    )

    num_evaluation_horizons = len(
        evaluation_horizons
    )

    if int(cache["context_length"]) != context_length:
        raise ValueError(
            "context_length metadata does not match context_tokens."
        )

    expected_targets = (
        num_windows,
        prediction_length,
        num_assets,
    )

    if tuple(target_s1.shape) != expected_targets:
        raise ValueError(
            "target_s1 has an unexpected shape."
        )

    if tuple(target_s2.shape) != expected_targets:
        raise ValueError(
            "target_s2 has an unexpected shape."
        )

    expected_stats = (
        num_windows,
        num_assets,
        6,
    )

    if tuple(context_mean.shape) != expected_stats:
        raise ValueError(
            "context_mean has an unexpected shape."
        )

    if tuple(context_std.shape) != expected_stats:
        raise ValueError(
            "context_std has an unexpected shape."
        )

    expected_evaluation = (
        num_windows,
        num_evaluation_horizons,
        num_assets,
        5,
    )

    if tuple(evaluation_true.shape) != (
        expected_evaluation
    ):
        raise ValueError(
            "evaluation_true has an unexpected shape."
        )

    if tuple(last_context_target.shape) != (
        num_windows,
        num_assets,
        5,
    ):
        raise ValueError(
            "last_context_target has an unexpected shape."
        )

    if tuple(sample_idx.shape) != (
        num_windows,
    ):
        raise ValueError(
            "sample_idx must have shape [W]."
        )

    if tuple(origin_idx.shape) != (
        num_windows,
    ):
        raise ValueError(
            "origin_idx must have shape [W]."
        )

    if tuple(target_indices.shape) != (
        num_windows,
        prediction_length,
    ):
        raise ValueError(
            "target_indices must have shape [W, P]."
        )

    if len(cache["dates"]) != num_windows:
        raise ValueError(
            "dates length does not match the window dimension."
        )

    if len(cache["asset_cols"]) != num_assets:
        raise ValueError(
            "asset_cols length does not match the asset dimension."
        )

    dense_horizons = tuple(
        int(value)
        for value in cache["dense_horizons"]
    )

    if dense_horizons != tuple(
        range(
            1,
            prediction_length + 1,
        )
    ):
        raise ValueError(
            "dense_horizons must be exactly 1..prediction_length."
        )

    evaluation_indices = tuple(
        int(value)
        for value in cache["evaluation_indices"]
    )

    expected_evaluation_indices = tuple(
        horizon - 1
        for horizon in evaluation_horizons
    )

    if evaluation_indices != expected_evaluation_indices:
        raise ValueError(
            "evaluation_indices are not aligned with "
            "evaluation_horizons."
        )

    s1_id_space = str(
        cache.get(
            "s1_id_space",
            "kronos_original",
        )
    )

    s1_vocabulary_size = int(
        cache.get(
            "s1_vocabulary_size",
            1024,
        )
    )

    if not 1 <= s1_vocabulary_size <= 1024:
        raise ValueError(
            "s1_vocabulary_size must lie in [1, 1024]."
        )

    if s1_id_space not in {
        "kronos_original",
        "compact_retained_kronos",
    }:
        raise ValueError(
            "Unsupported s1_id_space."
        )

    context_s1 = context_tokens[..., 0]
    context_s2 = context_tokens[..., 1]

    for name, values, maximum in (
        ("context s1", context_s1, s1_vocabulary_size),
        ("target_s1", target_s1, s1_vocabulary_size),
        ("context s2", context_s2, 1024),
        ("target_s2", target_s2, 1024),
    ):
        if (
            values.min().item() < 0
            or values.max().item() >= maximum
        ):
            raise ValueError(
                f"{name} contains IDs outside [0, {maximum - 1}]."
            )

    if s1_id_space == "compact_retained_kronos":
        if format_version < 2:
            raise ValueError(
                "Compact s1 caches require format_version >= 2."
            )

        required_mapping_keys = {
            "s1_compact_to_original",
            "s1_original_to_compact",
            "s1_remapping_method",
            "s1_remapping_resource_hash",
        }

        missing_mapping = required_mapping_keys - set(cache)
        if missing_mapping:
            raise KeyError(
                "Compact s1 cache is missing mapping keys: "
                f"{sorted(missing_mapping)}."
            )

        compact_to_original = torch.as_tensor(
            cache["s1_compact_to_original"],
            dtype=torch.long,
        )
        original_to_compact = torch.as_tensor(
            cache["s1_original_to_compact"],
            dtype=torch.long,
        )

        if tuple(compact_to_original.shape) != (
            s1_vocabulary_size,
        ):
            raise ValueError(
                "s1_compact_to_original must have shape [K]."
            )

        if tuple(original_to_compact.shape) != (1024,):
            raise ValueError(
                "s1_original_to_compact must have shape [1024]."
            )

        if (
            compact_to_original.min().item() < 0
            or compact_to_original.max().item() >= 1024
        ):
            raise ValueError(
                "s1_compact_to_original contains invalid Kronos IDs."
            )

        if torch.unique(compact_to_original).numel() != (
            s1_vocabulary_size
        ):
            raise ValueError(
                "s1_compact_to_original must contain unique IDs."
            )

        expected_forward = torch.full(
            (1024,),
            fill_value=-1,
            dtype=torch.long,
        )
        expected_forward[compact_to_original] = torch.arange(
            s1_vocabulary_size,
            dtype=torch.long,
        )

        if not torch.equal(
            original_to_compact,
            expected_forward,
        ):
            raise ValueError(
                "The compact/original s1 lookup tables are inconsistent."
            )

        if not str(cache["s1_remapping_resource_hash"]):
            raise ValueError(
                "s1_remapping_resource_hash must not be empty."
            )
    elif s1_vocabulary_size != 1024:
        raise ValueError(
            "Original Kronos s1 ID space must use vocabulary size 1024."
        )

    for name, values in (
        ("context_mean", context_mean),
        ("context_std", context_std),
        ("evaluation_true", evaluation_true),
        ("last_context_target", last_context_target),
    ):
        if not torch.isfinite(values).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

    if torch.any(context_std < 0):
        raise ValueError(
            "context_std contains negative values."
        )

    expected_future_clip_shape = (
        prediction_length,
        6,
    )

    if tuple(
        torch.as_tensor(
            cache[
                "future_clipping_rate_percent_by_step_channel"
            ]
        ).shape
    ) != expected_future_clip_shape:
        raise ValueError(
            "Future clipping-by-step metadata has the wrong shape."
        )

    if tuple(
        torch.as_tensor(
            cache[
                "context_clipping_rate_percent_by_step_channel"
            ]
        ).shape
    ) != (
        context_length,
        6,
    ):
        raise ValueError(
            "Context clipping-by-step metadata has the wrong shape."
        )


def build_origin_aligned_token_cache(
    dataset: WindowedCandleDataset,
    tokenizer: KronosTokenizerAdapter,
    *,
    evaluation_horizons: Sequence[int] = (
        1,
        5,
        15,
        30,
        60,
    ),
    window_batch_size: int = 2,
    series_batch_size: int | None = None,
    prefix_check_batches: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Generate the real-model context and dense future-token cache.

    The function operates on a ``WindowedCandleDataset`` configured
    with dense horizons ``1..prediction_length``. For every model
    origin, the full context-plus-future raw path is normalised using
    context-only statistics and encoded once as one ordered sequence.

    Prefix parity is checked for the requested number of initial
    batches. Context tokens are saved from the origin-aligned encoding;
    these should equal the existing causal context cache exactly.

    No tokenizer decoding is performed during cache generation.
    """
    if not isinstance(
        dataset,
        WindowedCandleDataset,
    ):
        raise TypeError(
            "dataset must be a WindowedCandleDataset."
        )

    if not isinstance(
        tokenizer,
        KronosTokenizerAdapter,
    ):
        raise TypeError(
            "tokenizer must be a KronosTokenizerAdapter."
        )

    if window_batch_size <= 0:
        raise ValueError(
            "window_batch_size must be positive."
        )

    if prefix_check_batches < 0:
        raise ValueError(
            "prefix_check_batches cannot be negative."
        )

    if len(dataset) == 0:
        raise ValueError(
            "The dense forecasting dataset contains no windows."
        )

    dense_horizons = tuple(
        int(value)
        for value in dataset.horizons
    )

    prediction_length = len(
        dense_horizons
    )

    if dense_horizons != tuple(
        range(
            1,
            prediction_length + 1,
        )
    ):
        raise ValueError(
            "The dataset must use dense horizons "
            "1..prediction_length."
        )

    resolved_evaluation_horizons = tuple(
        int(value)
        for value in evaluation_horizons
    )

    if (
        not resolved_evaluation_horizons
        or min(resolved_evaluation_horizons) < 1
        or max(resolved_evaluation_horizons)
        > prediction_length
    ):
        raise ValueError(
            "evaluation_horizons must lie inside the dense "
            "future path."
        )

    if len(set(resolved_evaluation_horizons)) != len(
        resolved_evaluation_horizons
    ):
        raise ValueError(
            "evaluation_horizons must be unique."
        )

    if tuple(sorted(resolved_evaluation_horizons)) != (
        resolved_evaluation_horizons
    ):
        raise ValueError(
            "evaluation_horizons must be sorted."
        )

    expected_ohlcv = list(OHLCV_CHANNELS)

    if list(dataset.input_channels) != expected_ohlcv:
        raise ValueError(
            "The origin-aligned cache currently requires OHLCV "
            "input channels in canonical order."
        )

    if list(dataset.target_channels) != expected_ohlcv:
        raise ValueError(
            "The origin-aligned cache currently requires OHLCV "
            "target channels in canonical order."
        )

    num_windows = len(dataset)
    context_length = int(
        dataset.context_length
    )
    num_assets = len(
        dataset.split["asset_cols"]
    )
    num_evaluation_horizons = len(
        resolved_evaluation_horizons
    )

    context_tokens = torch.empty(
        (
            num_windows,
            context_length,
            num_assets,
            2,
        ),
        dtype=torch.int16,
    )

    target_s1 = torch.empty(
        (
            num_windows,
            prediction_length,
            num_assets,
        ),
        dtype=torch.int16,
    )

    target_s2 = torch.empty_like(
        target_s1
    )

    context_mean = torch.empty(
        (
            num_windows,
            num_assets,
            6,
        ),
        dtype=torch.float32,
    )

    context_std = torch.empty_like(
        context_mean
    )

    evaluation_true = torch.empty(
        (
            num_windows,
            num_evaluation_horizons,
            num_assets,
            5,
        ),
        dtype=torch.float32,
    )

    last_context_target = torch.empty(
        (
            num_windows,
            num_assets,
            5,
        ),
        dtype=torch.float32,
    )

    sample_idx = torch.empty(
        num_windows,
        dtype=torch.long,
    )

    origin_idx = torch.empty_like(
        sample_idx
    )

    target_indices = torch.empty(
        (
            num_windows,
            prediction_length,
        ),
        dtype=torch.long,
    )

    dates: list[str] = [
        ""
        for _ in range(num_windows)
    ]

    context_clip_counts = torch.zeros(
        (
            context_length,
            6,
        ),
        dtype=torch.long,
    )

    future_clip_counts = torch.zeros(
        (
            prediction_length,
            6,
        ),
        dtype=torch.long,
    )

    evaluation_indices = torch.tensor(
        [
            horizon - 1
            for horizon in resolved_evaluation_horizons
        ],
        dtype=torch.long,
    )

    starts = range(
        0,
        num_windows,
        window_batch_size,
    )

    iterator: Any = starts

    if show_progress:
        iterator = tqdm(
            starts,
            total=(
                num_windows
                + window_batch_size
                - 1
            )
            // window_batch_size,
            desc="Encoding origin-aligned token windows",
        )

    for batch_number, start in enumerate(
        iterator
    ):
        stop = min(
            start + window_batch_size,
            num_windows,
        )

        examples = [
            dataset[index]
            for index in range(
                start,
                stop,
            )
        ]

        raw_context = torch.stack(
            [
                example["x"]
                for example in examples
            ],
            dim=0,
        )

        raw_future = torch.stack(
            [
                example["y"]
                for example in examples
            ],
            dim=0,
        )

        encoded = build_origin_aligned_token_batch(
            tokenizer=tokenizer,
            context=raw_context,
            future=raw_future,
            series_batch_size=series_batch_size,
            verify_prefix_parity=(
                batch_number
                < prefix_check_batches
            ),
            decode_oracle_future=False,
        )

        batch_slice = slice(
            start,
            stop,
        )

        context_tokens[
            batch_slice
        ] = encoded.context_tokens

        target_s1[
            batch_slice
        ] = encoded.target_s1

        target_s2[
            batch_slice
        ] = encoded.target_s2

        context_mean[
            batch_slice
        ] = encoded.context_mean

        context_std[
            batch_slice
        ] = encoded.context_std

        raw_future = raw_future.to(
            torch.float32
        )

        evaluation_true[
            batch_slice
        ] = raw_future.index_select(
            dim=1,
            index=evaluation_indices,
        )

        last_context_target[
            batch_slice
        ] = torch.stack(
            [
                example[
                    "last_context_target"
                ]
                for example in examples
            ],
            dim=0,
        ).to(torch.float32)

        sample_idx[
            batch_slice
        ] = torch.tensor(
            [
                int(example["sample_idx"])
                for example in examples
            ],
            dtype=torch.long,
        )

        origin_idx[
            batch_slice
        ] = torch.tensor(
            [
                int(example["origin_idx"])
                for example in examples
            ],
            dtype=torch.long,
        )

        target_indices[
            batch_slice
        ] = torch.stack(
            [
                torch.as_tensor(
                    example["target_indices"],
                    dtype=torch.long,
                )
                for example in examples
            ],
            dim=0,
        )

        for offset, example in enumerate(
            examples
        ):
            dates[start + offset] = str(
                example["day"]
            )

        context_clip_counts += (
            encoded.context_clipping_mask
            .sum(dim=(0, 2))
            .to(torch.long)
        )

        future_clip_counts += (
            encoded.future_clipping_mask
            .sum(dim=(0, 2))
            .to(torch.long)
        )

    per_step_denominator = float(
        num_windows
        * num_assets
    )

    context_clip_step = (
        context_clip_counts
        .to(torch.float64)
        .div(per_step_denominator)
        .mul(100.0)
    )

    future_clip_step = (
        future_clip_counts
        .to(torch.float64)
        .div(per_step_denominator)
        .mul(100.0)
    )

    cache: dict[str, Any] = {
        "format_version": (
            ORIGIN_ALIGNED_CACHE_VERSION
        ),
        "representation": (
            "origin_aligned_kronos_forecasting_tokens"
        ),
        "s1_id_space": "kronos_original",
        "s1_vocabulary_size": 1024,
        "context_tokens": context_tokens,
        "target_s1": target_s1,
        "target_s2": target_s2,
        "context_mean": context_mean,
        "context_std": context_std,
        "evaluation_true": evaluation_true,
        "last_context_target": (
            last_context_target
        ),
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
        "target_indices": target_indices,
        "dates": dates,
        "asset_cols": list(
            dataset.split["asset_cols"]
        ),
        "input_channels": list(
            dataset.input_channels
        ),
        "target_channels": list(
            dataset.target_channels
        ),
        "tokenizer_channels": list(
            TOKENIZER_CHANNELS
        ),
        "context_length": context_length,
        "prediction_length": (
            prediction_length
        ),
        "dense_horizons": (
            dense_horizons
        ),
        "evaluation_horizons": (
            resolved_evaluation_horizons
        ),
        "evaluation_indices": tuple(
            int(value)
            for value in evaluation_indices.tolist()
        ),
        "stride": int(dataset.stride),
        "amount_mode": "zero",
        "normalisation": {
            "stats_from": "context",
            "scope": "per_asset_channel",
            "eps": float(tokenizer.eps),
            "clip": float(tokenizer.clip),
            "std_unbiased": False,
        },
        "tokenizer_id": (
            tokenizer.tokenizer_id
        ),
        "tokenizer_revision": (
            tokenizer.tokenizer_revision
        ),
        "prefix_check_batches": int(
            min(
                prefix_check_batches,
                (
                    num_windows
                    + window_batch_size
                    - 1
                )
                // window_batch_size,
            )
        ),
        "context_clipping_rate_percent_by_step_channel": (
            context_clip_step
        ),
        "future_clipping_rate_percent_by_step_channel": (
            future_clip_step
        ),
        "context_clipping_rate_percent_by_channel": (
            context_clip_step.mean(dim=0)
        ),
        "future_clipping_rate_percent_by_channel": (
            future_clip_step.mean(dim=0)
        ),
    }

    validate_origin_aligned_token_cache(
        cache
    )

    return cache


def save_origin_aligned_token_cache(
    cache: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Validate and atomically save an origin-aligned token cache."""
    validate_origin_aligned_token_cache(
        cache
    )

    output_path = Path(
        path
    ).expanduser()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    torch.save(
        dict(cache),
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )

    return output_path


def load_origin_aligned_token_cache(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate an origin-aligned token cache."""
    input_path = Path(
        path
    ).expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Token cache does not exist: {input_path}"
        )

    try:
        cache = torch.load(
            input_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        cache = torch.load(
            input_path,
            map_location="cpu",
        )

    if not isinstance(
        cache,
        Mapping,
    ):
        raise TypeError(
            "Saved origin-aligned cache is not a mapping."
        )

    resolved = dict(cache)

    validate_origin_aligned_token_cache(
        resolved
    )

    return resolved


def build_and_save_origin_aligned_token_cache(
    dataset: WindowedCandleDataset,
    tokenizer: KronosTokenizerAdapter,
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Generate, validate and atomically save one split cache."""
    cache = build_origin_aligned_token_cache(
        dataset,
        tokenizer,
        **kwargs,
    )

    return save_origin_aligned_token_cache(
        cache,
        path,
    )

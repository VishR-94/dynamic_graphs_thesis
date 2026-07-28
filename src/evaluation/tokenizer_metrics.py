from __future__ import annotations
import random
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Mapping, Sequence
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from dataclasses import dataclass

def plot_tokenizer_reconstruction(
    split: Mapping[str, Any],
    decoded_data: Mapping[str, Any],
    *,
    asset: str | None = None,
    day: str | pd.Timestamp | int | None = None,
    channel: str = "close",
    reconstruction: str = "both",
    random_seed: int | None = None,
) -> tuple[Figure, Axes, dict[str, Any]]:
    """Plot an original intraday channel against its reconstruction.

    Args:
        split:
            Cleaned candle split containing the original daily data.

        decoded_data:
            Cached output from ``decode_causal_split``.

        asset:
            Asset ticker to display. A random asset is selected when
            omitted.

        day:
            Trading date, session index, or None. A random session is
            selected when omitted.

        channel:
            One of open, high, low, close, or volume.

        reconstruction:
            Reconstruction series to display:

                "full"
                "coarse"
                "both"

        random_seed:
            Optional reproducibility seed. When None, omitted day and
            asset values are selected randomly on every call.

    Returns:
        Figure, axis and metadata describing the selected plot.
    """
    valid_reconstruction_modes = {
        "full",
        "coarse",
        "both",
    }

    if reconstruction not in valid_reconstruction_modes:
        raise ValueError(
            "reconstruction must be one of "
            f"{sorted(valid_reconstruction_modes)}. "
            f"Received {reconstruction!r}."
        )

    required_decoded_keys = {
        "decoded_full",
        "decoded_coarse",
        "valid_mask",
        "dates",
        "asset_cols",
        "channels",
    }

    missing_keys = (
        required_decoded_keys - set(decoded_data)
    )

    if missing_keys:
        raise KeyError(
            "decoded_data is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    decoded_channels = list(
        decoded_data["channels"]
    )

    if channel not in decoded_channels:
        raise ValueError(
            f"Unsupported channel {channel!r}. Expected one of "
            f"{decoded_channels}."
        )

    split_channels = list(split["channels"])

    if channel not in split_channels:
        raise ValueError(
            f"Channel {channel!r} is absent from the split."
        )

    split_assets = list(split["asset_cols"])
    decoded_assets = list(
        decoded_data["asset_cols"]
    )

    if split_assets != decoded_assets:
        raise ValueError(
            "Split and decoded asset ordering do not match."
        )

    samples = list(split["samples"])

    decoded_full = torch.as_tensor(
        decoded_data["decoded_full"],
        dtype=torch.float32,
    )

    decoded_coarse = torch.as_tensor(
        decoded_data["decoded_coarse"],
        dtype=torch.float32,
    )

    valid_mask = torch.as_tensor(
        decoded_data["valid_mask"],
        dtype=torch.bool,
    )

    if decoded_full.shape != decoded_coarse.shape:
        raise ValueError(
            "decoded_full and decoded_coarse must have the "
            "same shape."
        )

    if decoded_full.ndim != 4:
        raise ValueError(
            "Decoded tensors must have shape [S, T, N, C]."
        )

    (
        num_sessions,
        num_bars,
        num_assets,
        num_channels,
    ) = decoded_full.shape

    expected_valid_mask_shape = (
        num_sessions,
        num_bars,
    )

    if tuple(valid_mask.shape) != (
        expected_valid_mask_shape
    ):
        raise ValueError(
            "valid_mask must have shape [S, T]. "
            f"Received {tuple(valid_mask.shape)}."
        )

    if num_sessions != len(samples):
        raise ValueError(
            "Decoded session count does not match the split."
        )

    if num_assets != len(split_assets):
        raise ValueError(
            "Decoded asset count does not match the split."
        )

    if num_channels != len(decoded_channels):
        raise ValueError(
            "Decoded channel metadata does not match the "
            "decoded tensor."
        )

    split_dates = [
        pd.Timestamp(sample[2]).normalize()
        for sample in samples
    ]

    decoded_dates = [
        pd.Timestamp(value).normalize()
        for value in decoded_data["dates"]
    ]

    if split_dates != decoded_dates:
        raise ValueError(
            "Split and decoded date ordering do not match."
        )

    rng = random.Random(random_seed)

    # --------------------------------------------------------
    # Select the session.
    # --------------------------------------------------------

    if day is None:
        session_idx = rng.randrange(
            num_sessions
        )

    elif isinstance(day, int):
        session_idx = day

        if not 0 <= session_idx < num_sessions:
            raise IndexError(
                "day index must lie between 0 and "
                f"{num_sessions - 1}."
            )

    else:
        requested_date = pd.Timestamp(
            day
        ).normalize()

        matching_indices = [
            idx
            for idx, date in enumerate(split_dates)
            if date == requested_date
        ]

        if len(matching_indices) != 1:
            raise ValueError(
                f"Expected one session for "
                f"{requested_date.date()}, found "
                f"{len(matching_indices)}."
            )

        session_idx = matching_indices[0]

    # --------------------------------------------------------
    # Select the asset.
    # --------------------------------------------------------

    if asset is None:
        asset_idx = rng.randrange(
            num_assets
        )

        selected_asset = split_assets[
            asset_idx
        ]

    else:
        if asset not in split_assets:
            raise ValueError(
                f"Unknown asset {asset!r}."
            )

        selected_asset = asset
        asset_idx = split_assets.index(asset)

    selected_date = split_dates[
        session_idx
    ]

    x_day, sample_timestamps, _ = samples[
        session_idx
    ]

    x_day = torch.as_tensor(
        x_day,
        dtype=torch.float32,
    )

    expected_original_shape = (
        num_bars,
        num_assets,
        len(split_channels),
    )

    if tuple(x_day.shape) != expected_original_shape:
        raise ValueError(
            "Unexpected original session shape: "
            f"{tuple(x_day.shape)}. Expected "
            f"{expected_original_shape}."
        )

    split_channel_idx = split_channels.index(
        channel
    )

    decoded_channel_idx = (
        decoded_channels.index(channel)
    )

    true_values = (
        x_day[
            :,
            asset_idx,
            split_channel_idx,
        ]
        .cpu()
        .numpy()
    )

    full_values = (
        decoded_full[
            session_idx,
            :,
            asset_idx,
            decoded_channel_idx,
        ]
        .cpu()
        .numpy()
    )

    coarse_values = (
        decoded_coarse[
            session_idx,
            :,
            asset_idx,
            decoded_channel_idx,
        ]
        .cpu()
        .numpy()
    )

    selected_valid_mask = (
        valid_mask[session_idx]
        .cpu()
        .numpy()
    )

    if not selected_valid_mask.any():
        raise ValueError(
            "The selected session contains no valid decoded bars."
        )

    # --------------------------------------------------------
    # Reconstruct timestamps when they are absent.
    # --------------------------------------------------------

    if sample_timestamps is None:
        timestamps = pd.date_range(
            start=(
                selected_date
                + pd.Timedelta(
                    hours=9,
                    minutes=31,
                )
            ),
            periods=num_bars,
            freq="1min",
        )

    else:
        timestamps = pd.DatetimeIndex(
            sample_timestamps
        )

        if len(timestamps) != num_bars:
            raise ValueError(
                "Timestamp count does not match the number of "
                "session bars."
            )

    valid_timestamps = timestamps[
        selected_valid_mask
    ]

    # --------------------------------------------------------
    # Plot.
    # --------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(14, 5)
    )

    axis.plot(
        timestamps,
        true_values,
        color="black",
        linewidth=1.5,
        label="True",
        zorder=3,
    )

    if reconstruction in {
        "coarse",
        "both",
    }:
        axis.plot(
            valid_timestamps,
            coarse_values[selected_valid_mask],
            linewidth=1.1,
            label="Coarse reconstruction",
            zorder=1,
        )

    if reconstruction in {
        "full",
        "both",
    }:
        axis.plot(
            valid_timestamps,
            full_values[selected_valid_mask],
            linewidth=1.2,
            label="Full reconstruction",
            zorder=2,
        )

    axis.set_title(
        f"{selected_asset} — "
        f"{selected_date.date()} — "
        f"{channel.capitalize()}"
    )

    axis.set_xlabel("Time")
    axis.set_ylabel(channel.capitalize())
    axis.grid(alpha=0.25)
    axis.legend()

    figure.autofmt_xdate()
    figure.tight_layout()

    valid_bar_indices = torch.nonzero(
        valid_mask[session_idx],
        as_tuple=False,
    ).flatten()

    selection = {
        "session_idx": session_idx,
        "date": selected_date,
        "asset": selected_asset,
        "asset_idx": asset_idx,
        "channel": channel,
        "reconstruction": reconstruction,
        "first_valid_bar": int(
            valid_bar_indices[0].item()
        ),
        "last_valid_bar": int(
            valid_bar_indices[-1].item()
        ),
    }

    plt.show()

    return figure, axis, selection


_RECONSTRUCTION_ORDER = (
    "rolling_mean",
    "coarse",
    "full",
)


def _normalised_cache_dates(values: object) -> pd.DatetimeIndex:
    """Convert cache date metadata to normalised timestamps."""
    return pd.DatetimeIndex(pd.to_datetime(values)).normalize()


def _finite_median(values: torch.Tensor) -> torch.Tensor:
    """Return the median of finite values, or NaN if none are finite."""
    finite = values[torch.isfinite(values)]

    if finite.numel() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)

    return finite.median()


def _assetwise_pearson(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Calculate one Pearson correlation per asset.

    ``x`` and ``y`` must have shape ``[session, time, asset]``. Session
    and time are pooled, while assets remain separate. Constant series
    receive ``NaN``.
    """
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(
            "x and y must have identical shape [S, T, N]."
        )

    x = x.reshape(-1, x.shape[-1]).to(torch.float64)
    y = y.reshape(-1, y.shape[-1]).to(torch.float64)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    covariance_sum = (x * y).sum(dim=0)
    x_sum_squared = x.square().sum(dim=0)
    y_sum_squared = y.square().sum(dim=0)

    valid = (x_sum_squared > 0) & (y_sum_squared > 0)
    denominator = torch.sqrt(x_sum_squared * y_sum_squared)

    correlations = torch.full_like(denominator, float("nan"))
    correlations[valid] = (
        covariance_sum[valid] / denominator[valid]
    )

    return correlations.clamp(min=-1.0, max=1.0)


def _stack_split_ohlcv(
    split: Mapping[str, Any],
    channels: list[str],
) -> torch.Tensor:
    """Stack aligned raw sessions into a tensor shaped [S, T, N, C]."""
    split_channels = list(split["channels"])
    missing = [c for c in channels if c not in split_channels]

    if missing:
        raise ValueError(
            f"The raw split is missing channels: {missing}."
        )

    channel_indices = torch.tensor(
        [split_channels.index(c) for c in channels],
        dtype=torch.long,
    )

    sessions: list[torch.Tensor] = []
    expected_shape: tuple[int, ...] | None = None

    for session_idx, (x_day, _, day) in enumerate(split["samples"]):
        values = torch.as_tensor(x_day, dtype=torch.float32)

        if values.ndim != 3:
            raise ValueError(
                "Expected each raw session to have shape [T, N, D]. "
                f"Session {session_idx} ({day}) has "
                f"{tuple(values.shape)}."
            )

        if expected_shape is None:
            expected_shape = tuple(values.shape)
        elif tuple(values.shape) != expected_shape:
            raise ValueError("All raw sessions must have one shape.")

        values = values.index_select(dim=2, index=channel_indices)

        if not torch.isfinite(values).all():
            raise ValueError(
                f"Raw session {session_idx} ({day}) contains "
                "non-finite OHLCV values."
            )

        sessions.append(values)

    if not sessions:
        raise ValueError("The raw split contains no sessions.")

    return torch.stack(sessions, dim=0)


def _summarise_reconstruction(
    reconstructed: torch.Tensor,
    true_values: torch.Tensor,
    *,
    channels: list[str],
    value_eps: float,
) -> dict[str, float]:
    """Calculate the selected metrics for one reconstruction mode."""

    if reconstructed.shape != true_values.shape:
        raise ValueError(
            "Reconstruction and truth must have the same shape "
            "[S, K, N, C]."
        )
    
    if reconstructed.ndim != 4:
        raise ValueError(
            "Reconstruction tensors must have shape [S, K, N, C]."
        )

    channel_idx = {name: channels.index(name) for name in channels}
    close_idx = channel_idx["close"]
    volume_idx = channel_idx["volume"]
    num_assets = reconstructed.shape[2]

    reconstructed = reconstructed.to(torch.float64)
    true_values = true_values.to(torch.float64)

    true_close = true_values[..., close_idx].clamp_min(value_eps)
    reconstructed_close = reconstructed[..., close_idx].clamp_min(
        value_eps
    )

    close_error_bps = (
        torch.log(reconstructed_close / true_close).abs() * 10_000.0
    )
    close_error_bps = close_error_bps.reshape(-1, num_assets)

    close_median_bps_by_asset = torch.quantile(
        close_error_bps,
        q=0.50,
        dim=0,
    )
    close_p95_bps_by_asset = torch.quantile(
        close_error_bps,
        q=0.95,
        dim=0,
    )

    true_volume = true_values[..., volume_idx]

    if torch.any(true_volume < 0):
        raise ValueError("Original volume contains negative values.")

    reconstructed_volume = reconstructed[..., volume_idx]
    volume_log1p_error = (
        torch.log1p(reconstructed_volume.clamp_min(0.0))
        - torch.log1p(true_volume)
    ).abs()
    volume_log1p_mae_by_asset = (
        volume_log1p_error.reshape(-1, num_assets).mean(dim=0)
    )

    # K contains consecutive valid bars within every session, so these
    # differences never cross a session boundary.
    true_returns = torch.log(true_close[:, 1:] / true_close[:, :-1])
    reconstructed_returns = torch.log(
        reconstructed_close[:, 1:] / reconstructed_close[:, :-1]
    )

    return_mae_bps_by_asset = (
        (reconstructed_returns - true_returns)
        .abs()
        .mul(10_000.0)
        .reshape(-1, num_assets)
        .mean(dim=0)
    )

    return_correlation_by_asset = _assetwise_pearson(
        reconstructed_returns,
        true_returns,
    )

    true_return_std = (
        true_returns.reshape(-1, num_assets)
        .std(dim=0, unbiased=False)
    )
    reconstructed_return_std = (
        reconstructed_returns.reshape(-1, num_assets)
        .std(dim=0, unbiased=False)
    )
    volatility_ratio_by_asset = torch.where(
        true_return_std > value_eps,
        reconstructed_return_std / true_return_std,
        torch.full_like(true_return_std, float("nan")),
    )

    open_values = reconstructed[..., channel_idx["open"]]
    high_values = reconstructed[..., channel_idx["high"]]
    low_values = reconstructed[..., channel_idx["low"]]
    close_values = reconstructed[..., close_idx]

    invalid_candle = (
        ~torch.isfinite(reconstructed).all(dim=-1)
        | (open_values <= 0)
        | (high_values <= 0)
        | (low_values <= 0)
        | (close_values <= 0)
        | (high_values < torch.maximum(open_values, close_values))
        | (low_values > torch.minimum(open_values, close_values))
        | (high_values < low_values)
        | (reconstructed_volume < 0)
    )

    return {
        "close_median_abs_error_bps": float(
            _finite_median(close_median_bps_by_asset).item()
        ),
        "close_p95_abs_error_bps": float(
            _finite_median(close_p95_bps_by_asset).item()
        ),
        "return_mae_bps": float(
            _finite_median(return_mae_bps_by_asset).item()
        ),
        "return_pearson": float(
            _finite_median(return_correlation_by_asset).item()
        ),
        "return_volatility_ratio": float(
            _finite_median(volatility_ratio_by_asset).item()
        ),
        "volume_log1p_mae": float(
            _finite_median(volume_log1p_mae_by_asset).item()
        ),
        "invalid_candle_rate": float(
            invalid_candle.to(torch.float64).mean().item()
        ),
    }


def compute_reconstruction_metrics(
    split: Mapping[str, Any],
    encoded_data: Mapping[str, Any],
    decoded_data: Mapping[str, Any],
    *,
    include_rolling_mean_baseline: bool = True,
    eps: float = 1.0e-8,
) -> pd.DataFrame:
    """Evaluate coarse and full Kronos tokenizer reconstructions.

    Each valid bar is the final reconstruction from its own trailing
    causal context. Metrics are calculated only at ``origin_indices``.
    Returns never cross trading-session boundaries.

    Except for ``invalid_candle_rate``, every metric is first computed
    independently for each asset and then summarised by the median
    across assets. This prevents higher-priced or more volatile assets
    from dominating the reported result.

    Args:
        split:
            Raw cleaned split aligned session-for-session with the
            encoded and decoded caches.

        encoded_data:
            Cache from ``encode_causal_split``. Context means are needed 
            for the optional rolling-mean reconstruction baseline.

        decoded_data:
            Cache from ``decode_causal_split`` containing
            ``decoded_coarse`` and ``decoded_full``.

        include_rolling_mean_baseline:
            Include the trailing-context channel mean as a simple
            reconstruction reference.

        eps:
            Positive numerical floor used for logarithms and ratios.

    Returns:
        DataFrame indexed by representation, with rows for
        ``side_information`` (optional), ``coarse`` and ``full``.
    """
    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    required_encoded = {
        "context_mean",
        "origin_indices",
        "valid_mask",
        "dates",
        "asset_cols",
        "channels",
        "tokenizer_channels",
    }
    required_decoded = {
        "decoded_coarse",
        "decoded_full",
        "origin_indices",
        "valid_mask",
        "dates",
        "asset_cols",
        "channels",
    }

    missing_encoded = required_encoded - set(encoded_data)
    missing_decoded = required_decoded - set(decoded_data)

    if missing_encoded:
        raise KeyError(
            f"encoded_data is missing keys: {sorted(missing_encoded)}."
        )
    if missing_decoded:
        raise KeyError(
            f"decoded_data is missing keys: {sorted(missing_decoded)}."
        )

    channels = list(decoded_data["channels"])
    expected_channels = ["open", "high", "low", "close", "volume"]

    if channels != expected_channels:
        raise ValueError(
            "Expected decoded channel order "
            f"{expected_channels}, received {channels}."
        )
    if list(encoded_data["channels"]) != channels:
        raise ValueError(
            "Encoded and decoded public channel orders differ."
        )

    split_assets = list(split["asset_cols"])
    encoded_assets = list(encoded_data["asset_cols"])
    decoded_assets = list(decoded_data["asset_cols"])

    if not (split_assets == encoded_assets == decoded_assets):
        raise ValueError(
            "Raw, encoded and decoded asset ordering differs."
        )

    split_dates = _normalised_cache_dates(
        [sample[2] for sample in split["samples"]]
    )
    encoded_dates = _normalised_cache_dates(encoded_data["dates"])
    decoded_dates = _normalised_cache_dates(decoded_data["dates"])

    if not (
        split_dates.equals(encoded_dates)
        and split_dates.equals(decoded_dates)
    ):
        raise ValueError(
            "Raw, encoded and decoded session dates are not aligned."
        )

    origin_indices = torch.as_tensor(
        encoded_data["origin_indices"],
        dtype=torch.long,
    )
    decoded_origins = torch.as_tensor(
        decoded_data["origin_indices"],
        dtype=torch.long,
    )

    if origin_indices.ndim != 1:
        raise ValueError("origin_indices must have shape [K].")
    if not torch.equal(origin_indices, decoded_origins):
        raise ValueError("Encoded and decoded origin indices differ.")

    true_all = _stack_split_ohlcv(split, channels)
    decoded_coarse_all = torch.as_tensor(
        decoded_data["decoded_coarse"],
        dtype=torch.float32,
    )
    decoded_full_all = torch.as_tensor(
        decoded_data["decoded_full"],
        dtype=torch.float32,
    )

    if decoded_coarse_all.shape != true_all.shape:
        raise ValueError(
            "decoded_coarse is not aligned with the raw split."
        )
    if decoded_full_all.shape != true_all.shape:
        raise ValueError(
            "decoded_full is not aligned with the raw split."
        )

    encoded_mask = torch.as_tensor(
        encoded_data["valid_mask"],
        dtype=torch.bool,
    )
    decoded_mask = torch.as_tensor(
        decoded_data["valid_mask"],
        dtype=torch.bool,
    )

    if not torch.equal(encoded_mask, decoded_mask):
        raise ValueError("Encoded and decoded valid masks differ.")

    expected_mask = torch.zeros_like(encoded_mask)
    expected_mask[:, origin_indices] = True

    if not torch.equal(encoded_mask, expected_mask):
        raise ValueError(
            "valid_mask is not aligned with origin_indices."
        )

    true_values = true_all.index_select(1, origin_indices)
    decoded_coarse = decoded_coarse_all.index_select(1, origin_indices)
    decoded_full = decoded_full_all.index_select(1, origin_indices)

    tokenizer_channels = list(encoded_data["tokenizer_channels"])
    missing_stats_channels = [
        channel for channel in channels if channel not in tokenizer_channels
    ]

    if missing_stats_channels:
        raise ValueError(
            "Tokenizer statistics are missing channels: "
            f"{missing_stats_channels}."
        )

    stats_indices = torch.tensor(
        [tokenizer_channels.index(channel) for channel in channels],
        dtype=torch.long,
    )


    context_mean = torch.as_tensor(
        encoded_data["context_mean"],
        dtype=torch.float32,
    ).index_select(
        3,
        stats_indices,
    )

    if context_mean.shape != true_values.shape:
        raise ValueError(
            "Context means are not aligned with raw observations."
        )

    if not torch.isfinite(decoded_coarse).all():
        raise ValueError(
            "Valid coarse reconstructions contain non-finite values."
        )
    if not torch.isfinite(decoded_full).all():
        raise ValueError(
            "Valid full reconstructions contain non-finite values."
        )
    if not torch.isfinite(context_mean).all():
        raise ValueError("Context means contain non-finite values.")

    

    representations: list[tuple[str, torch.Tensor]] = []

    if include_rolling_mean_baseline:
        representations.append(
            (
                "rolling_mean",
                context_mean,
            )
        )

    representations.extend(
        [
            ("coarse", decoded_coarse),
            ("full", decoded_full),
        ]
    )

    rows = []

    for representation, reconstructed in representations:
        rows.append(
            {
                "representation": representation,
                **_summarise_reconstruction(
                    reconstructed=reconstructed,
                    true_values=true_values,
                    channels=channels,
                    value_eps=eps,
                ),
            }
        )

    results = pd.DataFrame(rows).set_index("representation")
    results = results.reindex(
        [name for name in _RECONSTRUCTION_ORDER if name in results.index]
    )

    results.attrs["num_sessions"] = int(true_values.shape[0])
    results.attrs["num_valid_bars_per_session"] = int(
        true_values.shape[1]
    )
    results.attrs["num_assets"] = int(true_values.shape[2])
    results.attrs["aggregation"] = (
        "metric per asset, then median across assets; "
        "invalid_candle_rate is pooled"
    )

    results = results.rename(
        index={
            "rolling_mean": "Rolling Mean",
            "coarse": "Coarse (s1)",
            "full": "Full (s1 + s2)",
        },
        columns={
            "close_median_abs_error_bps": (
                "Close Median Error (bps)"
            ),
            "close_p95_abs_error_bps": (
                "Close P95 Error (bps)"
            ),
            "return_mae_bps": (
                "Return MAE (bps)"
            ),
            "return_pearson": (
                "Return Correlation"
            ),
            "return_volatility_ratio": (
                "Return Volatility Ratio"
            ),
            "volume_log1p_mae": (
                "Log-Volume MAE"
            ),
            "invalid_candle_rate": (
                "Invalid Candles (%)"
            ),
        },
    )

    results.index.name = "Reconstruction"

    results["Invalid Candles (%)"] = (
        results["Invalid Candles (%)"]
        * 100.0
    )

    return results


def compute_invalid_candle_metrics(
    split: Mapping[str, Any],
    decoded_data: Mapping[str, Any],
    *,
    eps: float = 1.0e-8,
) -> pd.DataFrame:
    """Summarise the type and severity of invalid decoded candles.

    Rates are calculated across all valid reconstructed asset-bars.
    Individual violation categories can overlap, so their rates do not
    sum to the overall invalid-candle rate.

    Structural violation magnitude is the largest of:

        max(open, close) - high
        low - min(open, close)
        low - high

    after clamping each quantity below at zero. It is expressed
    relative to reconstructed Close in basis points.

    Returns:
        A display-ready DataFrame with one row each for the coarse and
        full reconstructions.
    """
    if eps <= 0:
        raise ValueError(
            "eps must be greater than zero."
        )

    required_keys = {
        "decoded_coarse",
        "decoded_full",
        "valid_mask",
        "dates",
        "asset_cols",
        "channels",
    }

    missing_keys = required_keys - set(
        decoded_data
    )

    if missing_keys:
        raise KeyError(
            "decoded_data is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    channels = list(
        decoded_data["channels"]
    )

    expected_channels = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if channels != expected_channels:
        raise ValueError(
            "Expected decoded channel order "
            f"{expected_channels}, received {channels}."
        )

    if list(split["asset_cols"]) != list(
        decoded_data["asset_cols"]
    ):
        raise ValueError(
            "Split and decoded asset ordering do not match."
        )

    split_dates = pd.DatetimeIndex(
        pd.to_datetime(
            [
                sample[2]
                for sample in split["samples"]
            ]
        )
    ).normalize()

    decoded_dates = pd.DatetimeIndex(
        pd.to_datetime(
            decoded_data["dates"]
        )
    ).normalize()

    if not split_dates.equals(
        decoded_dates
    ):
        raise ValueError(
            "Split and decoded session dates do not match."
        )

    valid_mask = torch.as_tensor(
        decoded_data["valid_mask"],
        dtype=torch.bool,
    )

    representations = {
        "Coarse (s1)": torch.as_tensor(
            decoded_data["decoded_coarse"],
            dtype=torch.float64,
        ),
        "Full (s1 + s2)": torch.as_tensor(
            decoded_data["decoded_full"],
            dtype=torch.float64,
        ),
    }

    reference_shape = next(
        iter(representations.values())
    ).shape

    if len(reference_shape) != 4:
        raise ValueError(
            "Decoded tensors must have shape [S, T, N, C]."
        )

    if tuple(valid_mask.shape) != tuple(
        reference_shape[:2]
    ):
        raise ValueError(
            "valid_mask must have shape [S, T]."
        )

    for name, values in representations.items():
        if values.shape != reference_shape:
            raise ValueError(
                f"{name} has an incompatible shape."
            )

    channel_idx = {
        name: channels.index(name)
        for name in channels
    }

    rows: list[dict[str, float | str]] = []

    for representation, decoded in (
        representations.items()
    ):
        # Boolean indexing over [S, T] leaves [valid bar, N, C].
        values = decoded[valid_mask]

        finite = torch.isfinite(
            values
        ).all(dim=-1)

        open_values = values[
            ...,
            channel_idx["open"],
        ]
        high_values = values[
            ...,
            channel_idx["high"],
        ]
        low_values = values[
            ...,
            channel_idx["low"],
        ]
        close_values = values[
            ...,
            channel_idx["close"],
        ]
        volume_values = values[
            ...,
            channel_idx["volume"],
        ]

        body_high = torch.maximum(
            open_values,
            close_values,
        )

        body_low = torch.minimum(
            open_values,
            close_values,
        )

        non_positive_ohlc = finite & (
            (open_values <= 0)
            | (high_values <= 0)
            | (low_values <= 0)
            | (close_values <= 0)
        )

        high_constraint = finite & (
            high_values < body_high
        )

        low_constraint = finite & (
            low_values > body_low
        )

        inverted_range = finite & (
            high_values < low_values
        )

        negative_volume = finite & (
            volume_values < 0
        )

        any_invalid = (
            ~finite
            | non_positive_ohlc
            | high_constraint
            | low_constraint
            | inverted_range
            | negative_volume
        )

        high_shortfall = (
            body_high - high_values
        ).clamp_min(0.0)

        low_excess = (
            low_values - body_low
        ).clamp_min(0.0)

        range_inversion = (
            low_values - high_values
        ).clamp_min(0.0)

        structural_magnitude = torch.stack(
            [
                high_shortfall,
                low_excess,
                range_inversion,
            ],
            dim=-1,
        ).amax(dim=-1)

        structural_invalid = (
            high_constraint
            | low_constraint
            | inverted_range
        )

        valid_magnitude = (
            structural_invalid
            & finite
            & (close_values.abs() > eps)
        )

        magnitude_bps = (
            structural_magnitude
            / close_values.abs().clamp_min(eps)
            * 10_000.0
        )

        violation_magnitudes = magnitude_bps[
            valid_magnitude
        ]

        if violation_magnitudes.numel() == 0:
            median_violation_bps = float("nan")
            p95_violation_bps = float("nan")
        else:
            median_violation_bps = float(
                torch.quantile(
                    violation_magnitudes,
                    q=0.50,
                ).item()
            )

            p95_violation_bps = float(
                torch.quantile(
                    violation_magnitudes,
                    q=0.95,
                ).item()
            )

        def percentage(
            mask: torch.Tensor,
        ) -> float:
            return float(
                mask.to(torch.float64)
                .mean()
                .mul(100.0)
                .item()
            )

        rows.append(
            {
                "Reconstruction": representation,
                "Any Invalid (%)": percentage(
                    any_invalid
                ),
                "High Below Body (%)": percentage(
                    high_constraint
                ),
                "Low Above Body (%)": percentage(
                    low_constraint
                ),
                "High Below Low (%)": percentage(
                    inverted_range
                ),
                "Non-positive OHLC (%)": percentage(
                    non_positive_ohlc
                ),
                "Negative Volume (%)": percentage(
                    negative_volume
                ),
                "Median Structural Violation (bps)": (
                    median_violation_bps
                ),
                "P95 Structural Violation (bps)": (
                    p95_violation_bps
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("Reconstruction")
    )


def compute_codebook_usage_metrics(
    encoded_data: Mapping[str, Any],
    *,
    vocabulary_size: int = 1024,
    k: int = 10
) -> pd.DataFrame:
    """Summarise utilisation and concentration of both token streams.

    Only positions marked valid by ``valid_mask`` are included. The
    coarse and fine codebooks are analysed separately.

    Returns:
        A display-ready DataFrame with one row for ``s1`` and one row
        for ``s2``.
    """
    if vocabulary_size <= 1:
        raise ValueError(
            "vocabulary_size must be greater than one."
        )

    required_keys = {
        "s1",
        "s2",
        "valid_mask",
    }

    missing_keys = required_keys - set(
        encoded_data
    )

    if missing_keys:
        raise KeyError(
            "encoded_data is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    s1 = torch.as_tensor(
        encoded_data["s1"]
    )

    s2 = torch.as_tensor(
        encoded_data["s2"]
    )

    valid_mask = torch.as_tensor(
        encoded_data["valid_mask"],
        dtype=torch.bool,
    )

    if s1.shape != s2.shape:
        raise ValueError(
            "s1 and s2 must have the same shape."
        )

    if s1.ndim != 3:
        raise ValueError(
            "s1 and s2 must have shape [S, T, N]."
        )

    if tuple(valid_mask.shape) != tuple(
        s1.shape[:2]
    ):
        raise ValueError(
            "valid_mask must have shape [S, T]."
        )

    expanded_mask = (
        valid_mask
        .unsqueeze(-1)
        .expand_as(s1)
    )

    stream_values = {
        "Coarse (s1)": s1[expanded_mask].long(),
        "Fine (s2)": s2[expanded_mask].long(),
    }

    maximum_entropy = torch.log2(
        torch.tensor(
            float(vocabulary_size),
            dtype=torch.float64,
        )
    )

    rows: list[dict[str, float | int | str]] = []

    for stream_name, values in (
        stream_values.items()
    ):
        if values.numel() == 0:
            raise ValueError(
                f"{stream_name} contains no valid tokens."
            )

        if (
            values.min().item() < 0
            or values.max().item()
            >= vocabulary_size
        ):
            raise ValueError(
                f"{stream_name} contains token IDs outside "
                f"[0, {vocabulary_size - 1}]."
            )

        counts = torch.bincount(
            values,
            minlength=vocabulary_size,
        ).to(torch.float64)

        total = counts.sum()
        observed = counts > 0
        observed_count = int(
            observed.sum().item()
        )

        probabilities = (
            counts[observed]
            / total
        )

        entropy_bits = -(
            probabilities
            * torch.log2(probabilities)
        ).sum()

        normalised_entropy = (
            entropy_bits
            / maximum_entropy
        )

        effective_vocabulary = (
            torch.pow(
                torch.tensor(
                    2.0,
                    dtype=torch.float64,
                ),
                entropy_bits,
            )
        )

        top_k = min(
            k,
            vocabulary_size,
        )

        largest_counts = torch.topk(
            counts,
            k=top_k,
        ).values

        rows.append(
            {
                "Token Stream": stream_name,
                "Observations": int(
                    total.item()
                ),
                "Tokens Used": observed_count,
                "Codebook Usage (%)": (
                    100.0
                    * observed_count
                    / vocabulary_size
                ),
                "Entropy (bits)": float(
                    entropy_bits.item()
                ),
                "Normalised Entropy": float(
                    normalised_entropy.item()
                ),
                "Effective Vocabulary": float(
                    effective_vocabulary.item()
                ),
                "Most Common Token (%)": float(
                    counts.max()
                    .div(total)
                    .mul(100.0)
                    .item()
                ),
                f"Top-{k} Token Share (%)": float(
                    largest_counts.sum()
                    .div(total)
                    .mul(100.0)
                    .item()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("Token Stream")
    )

def _resolve_token_analysis_inputs(
    encoded_train: Mapping[str, Any],
    encoded_validation: Mapping[str, Any],
    *,
    token_type: str,
    horizons: Sequence[int],
    vocabulary_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    tuple[int, ...],
    pd.DatetimeIndex,
    pd.DatetimeIndex,
]:
    """Validate and align the encoded train/validation token caches."""
    if token_type not in {"s1", "s2"}:
        raise ValueError(
            "token_type must be either 's1' or 's2'. "
            f"Received {token_type!r}."
        )

    if vocabulary_size <= 1:
        raise ValueError(
            "vocabulary_size must be greater than one."
        )

    resolved_horizons = tuple(
        int(horizon)
        for horizon in horizons
    )

    if not resolved_horizons:
        raise ValueError(
            "At least one horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in resolved_horizons
    ):
        raise ValueError(
            "Every horizon must be a positive integer."
        )

    if len(set(resolved_horizons)) != len(
        resolved_horizons
    ):
        raise ValueError(
            "Each horizon should appear only once."
        )

    required_keys = {
        token_type,
        "valid_mask",
        "dates",
        "asset_cols",
    }

    for cache_name, cache in (
        ("encoded_train", encoded_train),
        ("encoded_validation", encoded_validation),
    ):
        missing_keys = required_keys - set(cache)

        if missing_keys:
            raise KeyError(
                f"{cache_name} is missing required keys: "
                f"{sorted(missing_keys)}."
            )

    train_tokens = torch.as_tensor(
        encoded_train[token_type]
    ).detach().cpu().long()

    validation_tokens = torch.as_tensor(
        encoded_validation[token_type]
    ).detach().cpu().long()

    train_valid = torch.as_tensor(
        encoded_train["valid_mask"],
        dtype=torch.bool,
    ).detach().cpu()

    validation_valid = torch.as_tensor(
        encoded_validation["valid_mask"],
        dtype=torch.bool,
    ).detach().cpu()

    if train_tokens.ndim != 3:
        raise ValueError(
            "Training tokens must have shape [S, T, N]."
        )

    if validation_tokens.ndim != 3:
        raise ValueError(
            "Validation tokens must have shape [S, T, N]."
        )

    if tuple(train_valid.shape) != tuple(
        train_tokens.shape[:2]
    ):
        raise ValueError(
            "Training valid_mask must have shape [S, T]."
        )

    if tuple(validation_valid.shape) != tuple(
        validation_tokens.shape[:2]
    ):
        raise ValueError(
            "Validation valid_mask must have shape [S, T]."
        )

    if train_tokens.shape[1:] != validation_tokens.shape[1:]:
        raise ValueError(
            "Training and validation bar/asset dimensions must "
            "match."
        )

    asset_cols = list(
        encoded_train["asset_cols"]
    )

    if asset_cols != list(
        encoded_validation["asset_cols"]
    ):
        raise ValueError(
            "Training and validation asset ordering do not match."
        )

    if len(asset_cols) != train_tokens.shape[2]:
        raise ValueError(
            "asset_cols does not match the token asset dimension."
        )

    max_horizon = max(resolved_horizons)

    if max_horizon >= train_tokens.shape[1]:
        raise ValueError(
            "A requested horizon is not smaller than the number "
            "of bars per session."
        )

    def validate_token_range(
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        name: str,
    ) -> None:
        expanded_mask = (
            valid_mask
            .unsqueeze(-1)
            .expand_as(tokens)
        )

        valid_values = tokens[expanded_mask]

        if valid_values.numel() == 0:
            raise ValueError(
                f"{name} contains no valid token observations."
            )

        if (
            valid_values.min().item() < 0
            or valid_values.max().item() >= vocabulary_size
        ):
            raise ValueError(
                f"{name} contains valid token IDs outside "
                f"[0, {vocabulary_size - 1}]."
            )

    validate_token_range(
        train_tokens,
        train_valid,
        "encoded_train",
    )

    validate_token_range(
        validation_tokens,
        validation_valid,
        "encoded_validation",
    )

    train_dates = pd.DatetimeIndex(
        pd.to_datetime(encoded_train["dates"])
    ).normalize()

    validation_dates = pd.DatetimeIndex(
        pd.to_datetime(encoded_validation["dates"])
    ).normalize()

    if len(train_dates) != train_tokens.shape[0]:
        raise ValueError(
            "Training dates do not match the session dimension."
        )

    if len(validation_dates) != validation_tokens.shape[0]:
        raise ValueError(
            "Validation dates do not match the session dimension."
        )

    if (
        not train_dates.is_monotonic_increasing
        or not validation_dates.is_monotonic_increasing
    ):
        raise ValueError(
            "Training and validation dates must be chronological."
        )

    if train_dates.max() >= validation_dates.min():
        raise ValueError(
            "Training and validation periods overlap or are not "
            "chronologically ordered."
        )

    return (
        train_tokens,
        validation_tokens,
        train_valid,
        validation_valid,
        asset_cols,
        resolved_horizons,
        train_dates,
        validation_dates,
    )


def _extract_within_asset_lagged_pairs(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract same-asset, within-session pairs as arrays [P, N]."""
    pair_mask = (
        valid_mask[:, :-horizon]
        & valid_mask[:, horizon:]
    )

    source = tokens[:, :-horizon, :][
        pair_mask
    ]

    target = tokens[:, horizon:, :][
        pair_mask
    ]

    if source.ndim != 2 or target.ndim != 2:
        raise RuntimeError(
            "Lagged token extraction did not return [P, N]."
        )

    if source.shape != target.shape:
        raise RuntimeError(
            "Source and target lagged token shapes differ."
        )

    if source.shape[0] == 0:
        raise ValueError(
            f"No valid token pairs exist at horizon {horizon}."
        )

    if source.min().item() < 0 or target.min().item() < 0:
        raise RuntimeError(
            "Invalid token positions entered the lagged pairs."
        )

    return (
        source.numpy().astype(np.int64, copy=False),
        target.numpy().astype(np.int64, copy=False),
    )


def _fit_witten_bell_marginal(
    target: np.ndarray,
    *,
    num_states: int,
) -> np.ndarray:
    """Fit a marginal distribution with Witten-Bell uniform backoff."""
    counts = np.bincount(
        target.reshape(-1),
        minlength=num_states,
    ).astype(np.float64)

    total = float(counts.sum())
    distinct = float(np.count_nonzero(counts))

    if total <= 0:
        raise ValueError(
            "No target observations were available."
        )

    uniform = np.full(
        num_states,
        fill_value=1.0 / num_states,
        dtype=np.float64,
    )

    return (
        counts + distinct * uniform
    ) / (
        total + distinct
    )


def _fit_transition_counts(
    source: np.ndarray,
    target: np.ndarray,
    *,
    num_states: int,
) -> np.ndarray:
    """Fit pooled within-asset transition counts [current, future]."""
    if source.shape != target.shape:
        raise ValueError(
            "Source and target shapes differ."
        )

    codes = (
        source.reshape(-1) * num_states
        + target.reshape(-1)
    )

    return np.bincount(
        codes,
        minlength=num_states * num_states,
    ).reshape(
        num_states,
        num_states,
    ).astype(
        np.int64,
        copy=False,
    )


def _score_witten_bell_transition_model(
    transition_counts: np.ndarray,
    marginal_distribution: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Score a train-fitted Witten-Bell transition model."""
    if source.shape != target.shape:
        raise ValueError(
            "Source and target shapes differ."
        )

    num_states = int(marginal_distribution.shape[0])

    if transition_counts.shape != (
        num_states,
        num_states,
    ):
        raise ValueError(
            "transition_counts has an incompatible shape."
        )

    source_flat = source.reshape(-1)
    target_flat = target.reshape(-1)

    marginal_probability = marginal_distribution[
        target_flat
    ]

    row_count = transition_counts.sum(
        axis=1,
        dtype=np.int64,
    )

    distinct_continuations = (
        transition_counts > 0
    ).sum(
        axis=1,
        dtype=np.int64,
    )

    realised_pair_count = transition_counts[
        source_flat,
        target_flat,
    ].astype(np.float64)

    realised_row_count = row_count[
        source_flat
    ].astype(np.float64)

    realised_distinct = distinct_continuations[
        source_flat
    ].astype(np.float64)

    denominator = (
        realised_row_count
        + realised_distinct
    )

    transition_probability = np.where(
        denominator > 0,
        (
            realised_pair_count
            + realised_distinct * marginal_probability
        )
        / np.where(
            denominator > 0,
            denominator,
            1.0,
        ),
        marginal_probability,
    )

    if (
        np.any(marginal_probability <= 0)
        or np.any(transition_probability <= 0)
    ):
        raise RuntimeError(
            "Smoothing produced a zero scoring probability."
        )

    marginal_ce = float(
        np.mean(-np.log2(marginal_probability))
    )

    transition_ce = float(
        np.mean(-np.log2(transition_probability))
    )

    return {
        "Marginal CE (bits/token)": marginal_ce,
        "Transition CE (bits/token)": transition_ce,
        "Marginal PPL": float(np.exp2(marginal_ce)),
        "Transition PPL": float(np.exp2(transition_ce)),
        "Bits Saved": float(
            marginal_ce - transition_ce
        ),
    }


def _empirical_transition_information(
    source: np.ndarray,
    target: np.ndarray,
    *,
    num_states: int,
) -> dict[str, float]:
    """Describe one empirical marginal/transition distribution."""
    transition_counts = _fit_transition_counts(
        source,
        target,
        num_states=num_states,
    ).astype(np.float64)

    target_counts = transition_counts.sum(axis=0)
    total = float(target_counts.sum())

    if total <= 0:
        raise ValueError(
            "No empirical transitions were available."
        )

    target_probabilities = target_counts[
        target_counts > 0
    ] / total

    marginal_entropy = float(
        -np.sum(
            target_probabilities
            * np.log2(target_probabilities)
        )
    )

    row_counts = transition_counts.sum(axis=1)
    non_empty_rows = row_counts > 0

    conditional_entropy = 0.0

    for row_idx in np.flatnonzero(non_empty_rows):
        row = transition_counts[row_idx]
        row_total = float(row_counts[row_idx])
        row_probabilities = row[row > 0] / row_total
        row_entropy = -np.sum(
            row_probabilities
            * np.log2(row_probabilities)
        )
        conditional_entropy += (
            row_total / total
        ) * float(row_entropy)

    mutual_information = max(
        0.0,
        marginal_entropy - conditional_entropy,
    )

    return {
        "Marginal Entropy (bits/token)": marginal_entropy,
        "Conditional Entropy (bits/token)": (
            conditional_entropy
        ),
        "Marginal PPL": float(
            np.exp2(marginal_entropy)
        ),
        "Conditional PPL": float(
            np.exp2(conditional_entropy)
        ),
        "Mutual Information (bits/token)": (
            mutual_information
        ),
    }


def _fit_witten_bell_marginal_from_counts(
    target_counts: np.ndarray,
) -> np.ndarray:
    """Fit a Witten-Bell marginal from pre-aggregated token counts.

    The lower-order backoff distribution is uniform over the configured
    vocabulary, matching ``_fit_witten_bell_marginal``. This count-based
    form is used by the rolling estimator so sessions can be added and
    removed without reconstructing the underlying token observations.
    """
    counts = np.asarray(
        target_counts,
        dtype=np.float64,
    )

    if counts.ndim != 1:
        raise ValueError(
            "target_counts must have shape [num_states]."
        )

    if np.any(counts < 0):
        raise ValueError(
            "target_counts contains negative values."
        )

    num_states = int(counts.shape[0])

    if num_states <= 1:
        raise ValueError(
            "target_counts must contain at least two states."
        )

    total = float(counts.sum())
    distinct = float(np.count_nonzero(counts))

    if total <= 0:
        raise ValueError(
            "No target observations were available."
        )

    uniform = np.full(
        num_states,
        fill_value=1.0 / num_states,
        dtype=np.float64,
    )

    return (
        counts + distinct * uniform
    ) / (
        total + distinct
    )


def _extract_single_session_lagged_pairs(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    session_idx: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one session's same-asset lagged token pairs.

    Returns arrays shaped ``[num_valid_times, num_assets]``. Restricting
    extraction to one session guarantees that no overnight transition is
    introduced.
    """
    if not 0 <= session_idx < tokens.shape[0]:
        raise IndexError(
            "session_idx lies outside the token session dimension."
        )

    return _extract_within_asset_lagged_pairs(
        tokens[
            session_idx:session_idx + 1
        ],
        valid_mask[
            session_idx:session_idx + 1
        ],
        horizon,
    )


def _sparse_session_transition_counts(
    source: np.ndarray,
    target: np.ndarray,
    *,
    num_states: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return sparse marginal and transition counts for one session.

    Returns:
        target_ids:
            Token IDs with non-zero future-token counts.

        target_counts:
            Counts aligned with ``target_ids``.

        pair_codes:
            Flattened transition IDs, where
            ``code = current * num_states + future``.

        pair_counts:
            Counts aligned with ``pair_codes``.
    """
    if source.shape != target.shape:
        raise ValueError(
            "Source and target shapes differ."
        )

    if source.size == 0:
        raise ValueError(
            "The session contains no valid lagged pairs."
        )

    source_flat = source.reshape(-1).astype(
        np.int64,
        copy=False,
    )
    target_flat = target.reshape(-1).astype(
        np.int64,
        copy=False,
    )

    if (
        source_flat.min() < 0
        or source_flat.max() >= num_states
        or target_flat.min() < 0
        or target_flat.max() >= num_states
    ):
        raise ValueError(
            "Session token IDs lie outside the configured vocabulary."
        )

    target_ids, target_counts = np.unique(
        target_flat,
        return_counts=True,
    )

    pair_codes, pair_counts = np.unique(
        source_flat * num_states + target_flat,
        return_counts=True,
    )

    return (
        target_ids.astype(np.int64, copy=False),
        target_counts.astype(np.int64, copy=False),
        pair_codes.astype(np.int64, copy=False),
        pair_counts.astype(np.int64, copy=False),
    )


def _update_rolling_transition_counts(
    marginal_counts: np.ndarray,
    transition_counts_flat: np.ndarray,
    session_counts: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    sign: int,
) -> None:
    """Add or remove one session's sparse counts in place."""
    if sign not in {-1, 1}:
        raise ValueError(
            "sign must be either +1 or -1."
        )

    (
        target_ids,
        target_counts,
        pair_codes,
        pair_counts,
    ) = session_counts

    marginal_counts[target_ids] += (
        sign * target_counts
    )

    transition_counts_flat[pair_codes] += (
        sign * pair_counts
    )

    if (
        np.any(marginal_counts[target_ids] < 0)
        or np.any(transition_counts_flat[pair_codes] < 0)
    ):
        raise RuntimeError(
            "Rolling count removal produced negative counts."
        )


def _compute_rolling_token_predictability(
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    train_valid: torch.Tensor,
    validation_valid: torch.Tensor,
    train_dates: pd.DatetimeIndex,
    validation_dates: pd.DatetimeIndex,
    *,
    horizons: tuple[int, ...],
    vocabulary_size: int,
    rolling_lookback: int,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score rolling transition models session by session.

    For every validation session, marginal and transition probabilities
    are estimated from exactly the previous ``rolling_lookback`` completed
    trading sessions. The validation session is scored before its counts
    are added to the rolling window. All assets and minutes in one session
    are therefore scored using one probability model fitted exclusively on
    earlier sessions.
    """
    if (
        isinstance(rolling_lookback, bool)
        or not isinstance(rolling_lookback, int)
    ):
        raise TypeError(
            "rolling_lookback must be an integer number of sessions."
        )

    if rolling_lookback <= 0:
        raise ValueError(
            "rolling_lookback must be greater than zero."
        )

    num_train_sessions = int(
        train_tokens.shape[0]
    )

    if rolling_lookback > num_train_sessions:
        raise ValueError(
            "rolling_lookback exceeds the number of completed "
            "training sessions available before validation: "
            f"{rolling_lookback} > {num_train_sessions}."
        )

    num_validation_sessions = int(
        validation_tokens.shape[0]
    )

    summary_rows: list[dict[str, float | int]] = []
    daily_rows: list[dict[str, float | int | str]] = []

    horizon_iterator: Any = horizons

    if show_progress:
        horizon_iterator = tqdm(
            horizons,
            desc=(
                "Scoring rolling token transitions "
                f"({rolling_lookback} sessions)"
            ),
        )

    for horizon in horizon_iterator:
        marginal_counts = np.zeros(
            vocabulary_size,
            dtype=np.int64,
        )

        transition_counts_flat = np.zeros(
            vocabulary_size * vocabulary_size,
            dtype=np.int64,
        )

        rolling_sessions: list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ]
        ] = []

        first_train_session = (
            num_train_sessions - rolling_lookback
        )

        for session_idx in range(
            first_train_session,
            num_train_sessions,
        ):
            source, target = (
                _extract_single_session_lagged_pairs(
                    train_tokens,
                    train_valid,
                    session_idx=session_idx,
                    horizon=horizon,
                )
            )

            session_counts = (
                _sparse_session_transition_counts(
                    source,
                    target,
                    num_states=vocabulary_size,
                )
            )

            _update_rolling_transition_counts(
                marginal_counts,
                transition_counts_flat,
                session_counts,
                sign=1,
            )

            rolling_sessions.append(
                session_counts
            )

        if len(rolling_sessions) != rolling_lookback:
            raise RuntimeError(
                "The initial rolling window has the wrong length."
            )

        marginal_loss_sum = 0.0
        transition_loss_sum = 0.0
        total_observations = 0

        session_iterator: Any = range(
            num_validation_sessions
        )

        if show_progress:
            session_iterator = tqdm(
                session_iterator,
                desc=f"{horizon}-minute sessions",
                leave=False,
            )

        for validation_session_idx in session_iterator:
            validation_source, validation_target = (
                _extract_single_session_lagged_pairs(
                    validation_tokens,
                    validation_valid,
                    session_idx=validation_session_idx,
                    horizon=horizon,
                )
            )

            marginal_distribution = (
                _fit_witten_bell_marginal_from_counts(
                    marginal_counts
                )
            )

            transition_counts = (
                transition_counts_flat.reshape(
                    vocabulary_size,
                    vocabulary_size,
                )
            )

            session_metrics = (
                _score_witten_bell_transition_model(
                    transition_counts,
                    marginal_distribution,
                    validation_source,
                    validation_target,
                )
            )

            num_observations = int(
                validation_target.size
            )

            marginal_loss_sum += (
                session_metrics[
                    "Marginal CE (bits/token)"
                ]
                * num_observations
            )

            transition_loss_sum += (
                session_metrics[
                    "Transition CE (bits/token)"
                ]
                * num_observations
            )

            total_observations += num_observations

            daily_rows.append(
                {
                    "Date": str(
                        validation_dates[
                            validation_session_idx
                        ].date()
                    ),
                    "Horizon (min)": int(horizon),
                    **session_metrics,
                    "Observations": num_observations,
                    "Rolling Window Start": str(
                        (
                            train_dates[
                                first_train_session
                            ]
                            if validation_session_idx == 0
                            else (
                                pd.DatetimeIndex(
                                    list(train_dates)
                                    + list(
                                        validation_dates[
                                            :validation_session_idx
                                        ]
                                    )
                                )[
                                    -rolling_lookback
                                ]
                            )
                        ).date()
                    ),
                    "Rolling Window End": str(
                        (
                            train_dates[-1]
                            if validation_session_idx == 0
                            else validation_dates[
                                validation_session_idx - 1
                            ]
                        ).date()
                    ),
                }
            )

            new_session_counts = (
                _sparse_session_transition_counts(
                    validation_source,
                    validation_target,
                    num_states=vocabulary_size,
                )
            )

            oldest_session_counts = (
                rolling_sessions.pop(0)
            )

            _update_rolling_transition_counts(
                marginal_counts,
                transition_counts_flat,
                oldest_session_counts,
                sign=-1,
            )

            _update_rolling_transition_counts(
                marginal_counts,
                transition_counts_flat,
                new_session_counts,
                sign=1,
            )

            rolling_sessions.append(
                new_session_counts
            )

            if len(rolling_sessions) != rolling_lookback:
                raise RuntimeError(
                    "The rolling window length changed unexpectedly."
                )

        if total_observations <= 0:
            raise RuntimeError(
                "No rolling validation observations were scored."
            )

        marginal_ce = (
            marginal_loss_sum
            / total_observations
        )

        transition_ce = (
            transition_loss_sum
            / total_observations
        )

        summary_rows.append(
            {
                "Horizon (min)": int(horizon),
                "Marginal CE (bits/token)": float(
                    marginal_ce
                ),
                "Transition CE (bits/token)": float(
                    transition_ce
                ),
                "Marginal PPL": float(
                    np.exp2(marginal_ce)
                ),
                "Transition PPL": float(
                    np.exp2(transition_ce)
                ),
                "Bits Saved": float(
                    marginal_ce - transition_ce
                ),
            }
        )

    summary = (
        pd.DataFrame(summary_rows)
        .set_index("Horizon (min)")
    )

    daily = pd.DataFrame(daily_rows)

    if not daily.empty:
        daily["Date"] = pd.to_datetime(
            daily["Date"]
        )
        daily["Rolling Window Start"] = pd.to_datetime(
            daily["Rolling Window Start"]
        )
        daily["Rolling Window End"] = pd.to_datetime(
            daily["Rolling Window End"]
        )
        daily = daily.set_index(
            [
                "Date",
                "Horizon (min)",
            ]
        ).sort_index()

    return summary, daily


def compute_token_predictability_table(
    encoded_train: Mapping[str, Any],
    encoded_validation: Mapping[str, Any],
    *,
    token_type: str = "s1",
    transition_source: str = "train",
    rolling_lookback: int = 20,
    horizons: Sequence[int] = (
        1,
        5,
        15,
        30,
        60,
    ),
    vocabulary_size: int = 1024,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute within-asset token-transition summaries.

    Modes:
        ``transition_source='train'``
            Fit Witten-Bell-smoothed marginal and transition
            probabilities once on the earlier cache and score them on
            the later cache. Returns held-out cross-entropies,
            perplexities and bits saved by conditioning.

        ``transition_source='validation'``
            Estimate and summarise the empirical marginal and transition
            distributions on the validation cache itself. Returns
            marginal entropy, conditional entropy, entropy-based
            perplexities and mutual information. This mode is descriptive,
            not out-of-sample prediction.

        ``transition_source='rolling'``
            For every validation session, fit Witten-Bell-smoothed
            marginal and transition probabilities using exactly the
            previous ``rolling_lookback`` completed sessions, score the
            current session, and only then add it to the rolling history.
            Returns out-of-sample/prequential cross-entropies,
            perplexities and bits saved by conditioning.

    In every mode, token pairs remain within one asset and one trading
    session. A separate direct transition matrix is constructed for every
    requested horizon; the one-minute matrix is not repeatedly applied to
    obtain longer-horizon probabilities.

    Args:
        encoded_train:
            Chronologically earlier encoded token cache.

        encoded_validation:
            Later encoded token cache used for validation-period analysis.

        token_type:
            ``'s1'`` or ``'s2'``.

        transition_source:
            ``'train'``, ``'validation'``/``'val'``, or ``'rolling'``.

        rolling_lookback:
            Number of completed trading sessions in the rolling estimation
            window. It is used only when ``transition_source='rolling'``.
            Financially interpretable examples are 5 sessions (about one
            week) and 20 sessions (about four weeks).

        horizons:
            Direct lag horizons in minutes.

        vocabulary_size:
            Number of possible token IDs.

        show_progress:
            Display progress bars.

    Returns:
        A clean horizon-indexed table. Rolling daily diagnostics are stored
        in ``result.attrs['daily_metrics']`` rather than displayed in the
        notebook table.
    """
    source_aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "validation": "validation",
        "rolling": "rolling",
    }

    resolved_source = source_aliases.get(
        str(transition_source).strip().lower()
    )

    if resolved_source is None:
        raise ValueError(
            "transition_source must be 'train', 'validation', "
            "or 'rolling'."
        )

    (
        train_tokens,
        validation_tokens,
        train_valid,
        validation_valid,
        asset_cols,
        resolved_horizons,
        train_dates,
        validation_dates,
    ) = _resolve_token_analysis_inputs(
        encoded_train,
        encoded_validation,
        token_type=token_type,
        horizons=horizons,
        vocabulary_size=vocabulary_size,
    )

    daily_metrics: pd.DataFrame | None = None

    if resolved_source == "rolling":
        result, daily_metrics = (
            _compute_rolling_token_predictability(
                train_tokens,
                validation_tokens,
                train_valid,
                validation_valid,
                train_dates,
                validation_dates,
                horizons=resolved_horizons,
                vocabulary_size=vocabulary_size,
                rolling_lookback=rolling_lookback,
                show_progress=show_progress,
            )
        )

    else:
        iterator: Any = resolved_horizons

        if show_progress:
            iterator = tqdm(
                resolved_horizons,
                desc=(
                    "Scoring train-fitted token transitions"
                    if resolved_source == "train"
                    else "Summarising validation token transitions"
                ),
            )

        rows: list[dict[str, float | int]] = []

        for horizon in iterator:
            if resolved_source == "train":
                train_source, train_target = (
                    _extract_within_asset_lagged_pairs(
                        train_tokens,
                        train_valid,
                        horizon,
                    )
                )

                validation_source, validation_target = (
                    _extract_within_asset_lagged_pairs(
                        validation_tokens,
                        validation_valid,
                        horizon,
                    )
                )

                marginal = _fit_witten_bell_marginal(
                    train_target,
                    num_states=vocabulary_size,
                )

                transition_counts = _fit_transition_counts(
                    train_source,
                    train_target,
                    num_states=vocabulary_size,
                )

                metrics = _score_witten_bell_transition_model(
                    transition_counts,
                    marginal,
                    validation_source,
                    validation_target,
                )

            else:
                validation_source, validation_target = (
                    _extract_within_asset_lagged_pairs(
                        validation_tokens,
                        validation_valid,
                        horizon,
                    )
                )

                metrics = _empirical_transition_information(
                    validation_source,
                    validation_target,
                    num_states=vocabulary_size,
                )

            rows.append(
                {
                    "Horizon (min)": int(horizon),
                    **metrics,
                }
            )

        result = (
            pd.DataFrame(rows)
            .set_index("Horizon (min)")
        )

    result.attrs["token_type"] = token_type
    result.attrs["transition_source"] = resolved_source
    result.attrs["vocabulary_size"] = int(vocabulary_size)
    result.attrs["horizons"] = resolved_horizons
    result.attrs["asset_cols"] = asset_cols
    result.attrs["train_period"] = (
        str(train_dates[0].date()),
        str(train_dates[-1].date()),
    )
    result.attrs["validation_period"] = (
        str(validation_dates[0].date()),
        str(validation_dates[-1].date()),
    )
    result.attrs["smoothing"] = (
        "interpolated_witten_bell"
        if resolved_source in {
            "train",
            "rolling",
        }
        else None
    )
    result.attrs["rolling_lookback"] = (
        int(rolling_lookback)
        if resolved_source == "rolling"
        else None
    )
    result.attrs["rolling_unit"] = (
        "completed_trading_sessions"
        if resolved_source == "rolling"
        else None
    )
    result.attrs["rolling_protocol"] = (
        "fit on previous completed sessions; score current session; "
        "update only after scoring"
        if resolved_source == "rolling"
        else None
    )
    result.attrs["daily_metrics"] = (
        daily_metrics.reset_index().to_dict(orient="records")
        if daily_metrics is not None
        else None
    )

    return result

def _joint_counts(
    x: np.ndarray,
    y: np.ndarray,
    num_states: int,
) -> np.ndarray:
    """Return a dense joint-count table for aligned state arrays."""
    if x.shape != y.shape:
        raise ValueError(
            "Aligned token arrays must have the same shape."
        )

    codes = (
        x.astype(np.int64, copy=False) * num_states
        + y.astype(np.int64, copy=False)
    )

    return np.bincount(
        codes,
        minlength=num_states * num_states,
    ).reshape(
        num_states,
        num_states,
    ).astype(np.float64)


def _mutual_information_bits(
    counts: np.ndarray,
) -> float:
    """Compute plug-in mutual information in bits from joint counts."""
    total = float(counts.sum())

    if total <= 0:
        return 0.0

    joint = counts / total
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    independent = marginal_x * marginal_y
    non_zero = joint > 0

    return float(
        np.sum(
            joint[non_zero]
            * (
                np.log2(joint[non_zero])
                - np.log2(independent[non_zero])
            )
        )
    )


def _coarsen_validation_tokens(
    tokens: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    vocabulary_size: int,
    top_k_tokens: int,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Keep the validation period's top-k IDs and map all others to 0."""
    if not 1 <= top_k_tokens < vocabulary_size:
        raise ValueError(
            "top_k_tokens must lie between 1 and vocabulary_size - 1."
        )

    expanded_mask = (
        valid_mask
        .unsqueeze(-1)
        .expand_as(tokens)
    )

    valid_values = tokens[expanded_mask].numpy()

    counts = np.bincount(
        valid_values,
        minlength=vocabulary_size,
    ).astype(np.int64)

    observed = np.flatnonzero(counts > 0)

    if observed.size < top_k_tokens:
        raise ValueError(
            "Fewer observed token IDs than top_k_tokens."
        )

    # Frequency descending, token ID ascending for deterministic ties.
    order = np.lexsort(
        (
            observed,
            -counts[observed],
        )
    )

    top_ids = observed[order[:top_k_tokens]]

    mapping = np.zeros(
        vocabulary_size,
        dtype=np.int16,
    )

    mapping[top_ids] = np.arange(
        1,
        top_k_tokens + 1,
        dtype=np.int16,
    )

    token_values = tokens.numpy()
    mapped = np.full(
        token_values.shape,
        fill_value=-1,
        dtype=np.int16,
    )

    valid_positions = token_values >= 0
    mapped[valid_positions] = mapping[
        token_values[valid_positions]
    ]

    return (
        mapped,
        tuple(int(value) for value in top_ids.tolist()),
        top_k_tokens + 1,
    )


def _reshape_valid_sessions_for_cross_mi(
    mapped_tokens: np.ndarray,
    valid_mask: torch.Tensor,
) -> np.ndarray:
    """Return coarsened states with shape [sessions, assets, bars]."""
    mask = valid_mask.numpy()
    reference_positions = np.flatnonzero(mask[0])

    if reference_positions.size == 0:
        raise ValueError(
            "The validation cache contains no valid token bars."
        )

    for session_idx in range(mask.shape[0]):
        positions = np.flatnonzero(mask[session_idx])

        if not np.array_equal(
            positions,
            reference_positions,
        ):
            raise ValueError(
                "Cross-asset MI requires the same valid bar positions "
                "in every session."
            )

    states = mapped_tokens[
        :,
        reference_positions,
        :,
    ].transpose(0, 2, 1)

    if np.any(states < 0):
        raise RuntimeError(
            "Invalid token positions entered the cross-asset states."
        )

    return states.astype(np.int16, copy=False)


def _cross_asset_mi_matrix(
    states: np.ndarray,
    *,
    num_states: int,
) -> np.ndarray:
    """Compute a symmetric contemporaneous asset-pair MI matrix."""
    if states.ndim != 3:
        raise ValueError(
            "states must have shape [sessions, assets, bars]."
        )

    num_assets = states.shape[1]
    flat = states.transpose(1, 0, 2).reshape(
        num_assets,
        -1,
    )

    matrix = np.zeros(
        (num_assets, num_assets),
        dtype=np.float64,
    )

    for left in range(num_assets):
        for right in range(left + 1, num_assets):
            value = _mutual_information_bits(
                _joint_counts(
                    flat[left],
                    flat[right],
                    num_states,
                )
            )
            matrix[left, right] = value
            matrix[right, left] = value

    return matrix


def _cross_asset_null_mean(
    states: np.ndarray,
    *,
    num_states: int,
    n_permutations: int,
    random_seed: int,
    show_progress: bool,
    description: str,
) -> np.ndarray:
    """Estimate Dimitri's session-shuffled cross-asset MI null mean."""
    if n_permutations <= 0:
        raise ValueError(
            "n_permutations must be greater than zero."
        )

    num_sessions, num_assets, _ = states.shape

    if num_sessions < 2:
        raise ValueError(
            "At least two sessions are required for a session-shuffled "
            "null distribution."
        )

    rng = np.random.default_rng(random_seed)
    original = states.transpose(1, 0, 2).reshape(
        num_assets,
        -1,
    )

    accumulated = np.zeros(
        (num_assets, num_assets),
        dtype=np.float64,
    )

    iterator: Any = range(n_permutations)

    if show_progress:
        iterator = tqdm(
            iterator,
            desc=description,
            leave=False,
        )

    for _ in iterator:
        permutation = rng.permutation(num_sessions)
        shuffled = states[
            permutation
        ].transpose(1, 0, 2).reshape(
            num_assets,
            -1,
        )

        for left in range(num_assets):
            for right in range(left + 1, num_assets):
                value = _mutual_information_bits(
                    _joint_counts(
                        original[left],
                        shuffled[right],
                        num_states,
                    )
                )
                accumulated[left, right] += value
                accumulated[right, left] += value

    return accumulated / float(n_permutations)


def _mean_upper_triangle(
    matrix: np.ndarray,
) -> float:
    """Average the 4,278 unique off-diagonal pairs for 93 assets."""
    upper = np.triu_indices(
        matrix.shape[0],
        k=1,
    )
    return float(np.mean(matrix[upper]))


def _token_entropy_order(
    states: np.ndarray,
    *,
    num_states: int,
) -> np.ndarray:
    """Order assets by their marginal token entropy, as Dimitri does."""
    entropies = []

    for asset_idx in range(states.shape[1]):
        counts = np.bincount(
            states[:, asset_idx, :].reshape(-1),
            minlength=num_states,
        ).astype(np.float64)
        probabilities = counts[counts > 0]
        probabilities /= probabilities.sum()
        entropies.append(
            -np.sum(
                probabilities * np.log2(probabilities)
            )
        )

    return np.argsort(np.asarray(entropies))


def _validate_raw_split_for_mi(
    raw_split: Mapping[str, Any],
    *,
    encoded_dates: pd.DatetimeIndex,
    asset_cols: list[str],
) -> np.ndarray:
    """Return one realised-volatility value per aligned raw session."""
    required_keys = {
        "samples",
        "channels",
        "asset_cols",
    }

    missing_keys = required_keys - set(raw_split)

    if missing_keys:
        raise KeyError(
            "raw_split is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    if list(raw_split["asset_cols"]) != asset_cols:
        raise ValueError(
            "Raw and encoded asset ordering do not match."
        )

    channels = list(raw_split["channels"])

    if "close" not in channels:
        raise ValueError(
            "raw_split does not contain a close channel."
        )

    close_idx = channels.index("close")
    samples = list(raw_split["samples"])

    raw_dates = pd.DatetimeIndex(
        pd.to_datetime(
            [sample[2] for sample in samples]
        )
    ).normalize()

    if not raw_dates.equals(encoded_dates):
        raise ValueError(
            "Raw and encoded validation dates do not match."
        )

    realised_volatility = np.empty(
        len(samples),
        dtype=np.float64,
    )

    for session_idx, (x_day, _, _) in enumerate(samples):
        values = torch.as_tensor(
            x_day,
            dtype=torch.float64,
        ).detach().cpu().numpy()

        if values.ndim != 3:
            raise ValueError(
                "Each raw session must have shape [T, N, D]."
            )

        close = values[:, :, close_idx]

        if np.any(~np.isfinite(close)) or np.any(close <= 0):
            raise ValueError(
                f"Session {session_idx} contains invalid Close values."
            )

        returns = np.diff(np.log(close), axis=0)

        # Dimitri's definition: within-session return standard deviation
        # per asset, then average those standard deviations across assets.
        realised_volatility[session_idx] = float(
            np.nanstd(
                returns,
                axis=0,
                ddof=0,
            ).mean()
        )

    return realised_volatility


@dataclass(frozen=True)
class CrossAssetMIAnalysis:
    """Complete contemporaneous cross-asset MI analysis."""

    token_type: str
    num_states: int
    top_token_ids: tuple[int, ...]
    asset_cols: tuple[str, ...]
    asset_order: np.ndarray
    observed_mi: np.ndarray
    null_mean_mi: np.ndarray
    excess_mi: np.ndarray
    overall_summary: pd.DataFrame
    realised_volatility: np.ndarray
    volatility_thresholds: tuple[float, float]
    regime_labels: tuple[str, ...]
    regime_assignments: np.ndarray
    regime_observed_mi: dict[str, np.ndarray]
    regime_null_mean_mi: dict[str, np.ndarray]
    regime_excess_mi: dict[str, np.ndarray]
    regime_summary: pd.DataFrame
    n_permutations: int
    random_seed: int
    volatility_tail_fraction: float


def compute_cross_asset_mi_analysis(
    encoded_validation: Mapping[str, Any],
    raw_validation_split: Mapping[str, Any],
    *,
    token_type: str = "s1",
    vocabulary_size: int = 1024,
    top_k_tokens: int = 15,
    volatility_tail_fraction: float = 1.0 / 3.0,
    n_permutations: int = 20,
    random_seed: int = 42,
    show_progress: bool = True,
) -> CrossAssetMIAnalysis:
    """Compute Dimitri-style cross-sectional MI and volatility regimes.

    The analysis is descriptive and is estimated on the validation
    period itself. Tokens are coarsened to the period's 15 most common
    IDs plus one Other state, matching Dimitri's approach.

    The observed contemporaneous matrix contains

        I(S_t^A ; S_t^B)

    for every unordered asset pair. A session-shuffled null preserves
    each asset's token distribution and complete intraday sequence but
    breaks same-date cross-asset alignment. The reported excess matrix
    is observed MI minus the mean null MI.

    Realised-volatility regimes follow Dimitri exactly: session scores
    are split at the 1/3 and 2/3 quantiles into low, middle and high
    terciles. Each regime receives its own observed, null and excess MI
    matrix using the same token mapping and asset ordering.
    """
    if token_type not in {"s1", "s2"}:
        raise ValueError(
            "token_type must be either 's1' or 's2'."
        )

    if vocabulary_size <= 1:
        raise ValueError(
            "vocabulary_size must be greater than one."
        )

    if n_permutations <= 0:
        raise ValueError(
            "n_permutations must be greater than zero."
        )

    volatility_tail_fraction = float(
        volatility_tail_fraction
    )

    if not 0.0 < volatility_tail_fraction <= 0.5:
        raise ValueError(
            "volatility_tail_fraction must lie in (0, 0.5]. "
            f"Received {volatility_tail_fraction}."
        )

    required_keys = {
        token_type,
        "valid_mask",
        "dates",
        "asset_cols",
    }

    missing_keys = required_keys - set(encoded_validation)

    if missing_keys:
        raise KeyError(
            "encoded_validation is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    tokens = torch.as_tensor(
        encoded_validation[token_type]
    ).detach().cpu().long()

    valid_mask = torch.as_tensor(
        encoded_validation["valid_mask"],
        dtype=torch.bool,
    ).detach().cpu()

    if tokens.ndim != 3:
        raise ValueError(
            "Validation tokens must have shape [S, T, N]."
        )

    if tuple(valid_mask.shape) != tuple(tokens.shape[:2]):
        raise ValueError(
            "valid_mask must have shape [S, T]."
        )

    asset_cols = list(encoded_validation["asset_cols"])

    if len(asset_cols) != tokens.shape[2]:
        raise ValueError(
            "asset_cols does not match the token asset dimension."
        )

    expanded_mask = valid_mask.unsqueeze(-1).expand_as(tokens)
    valid_values = tokens[expanded_mask]

    if valid_values.numel() == 0:
        raise ValueError(
            "The validation cache contains no valid tokens."
        )

    if (
        valid_values.min().item() < 0
        or valid_values.max().item() >= vocabulary_size
    ):
        raise ValueError(
            "Valid token IDs lie outside the configured vocabulary."
        )

    encoded_dates = pd.DatetimeIndex(
        pd.to_datetime(encoded_validation["dates"])
    ).normalize()

    if len(encoded_dates) != tokens.shape[0]:
        raise ValueError(
            "Encoded dates do not match the session dimension."
        )

    mapped, top_ids, num_states = (
        _coarsen_validation_tokens(
            tokens,
            valid_mask,
            vocabulary_size=vocabulary_size,
            top_k_tokens=top_k_tokens,
        )
    )

    states = _reshape_valid_sessions_for_cross_mi(
        mapped,
        valid_mask,
    )

    if show_progress:
        print("Computing overall cross-asset MI matrix...")

    observed = _cross_asset_mi_matrix(
        states,
        num_states=num_states,
    )

    null_mean = _cross_asset_null_mean(
        states,
        num_states=num_states,
        n_permutations=n_permutations,
        random_seed=random_seed,
        show_progress=show_progress,
        description="Overall MI null permutations",
    )

    excess = observed - null_mean
    np.fill_diagonal(excess, 0.0)

    overall_summary = pd.DataFrame(
        {
            "Observed Cross-Asset MI (bits)": [
                _mean_upper_triangle(observed)
            ],
            "Null Cross-Asset MI (bits)": [
                _mean_upper_triangle(null_mean)
            ],
            "Excess Cross-Asset MI (bits)": [
                _mean_upper_triangle(excess)
            ],
            "Sessions": [int(states.shape[0])],
            "States": [int(num_states)],
        },
        index=pd.Index(
            [token_type],
            name="Token Stream",
        ),
    )

    realised_volatility = _validate_raw_split_for_mi(
        raw_validation_split,
        encoded_dates=encoded_dates,
        asset_cols=asset_cols,
    )

    num_sessions = int(
        realised_volatility.shape[0]
    )

    num_tail_sessions = int(
        np.floor(
            num_sessions
            * volatility_tail_fraction
        )
    )

    num_tail_sessions = max(
        1,
        num_tail_sessions,
    )

    if 2 * num_tail_sessions > num_sessions:
        num_tail_sessions = num_sessions // 2

    if num_tail_sessions < 1:
        raise ValueError(
            "Not enough sessions to form non-overlapping "
            "low- and high-volatility groups."
        )

    volatility_order = np.argsort(
        realised_volatility,
        kind="stable",
    )

    low_indices = volatility_order[
        :num_tail_sessions
    ]

    high_indices = volatility_order[
        -num_tail_sessions:
    ]

    thresholds_array = np.array(
        [
            float(
                realised_volatility[
                    low_indices
                ].max()
            ),
            float(
                realised_volatility[
                    high_indices
                ].min()
            ),
        ],
        dtype=np.float64,
    )

    regime_assignments = np.full(
        num_sessions,
        fill_value=-1,
        dtype=np.int64,
    )

    regime_assignments[
        low_indices
    ] = 0

    regime_assignments[
        high_indices
    ] = 1

    regime_labels = (
        "Low Volatility",
        "High Volatility",
    )

    regime_observed: dict[str, np.ndarray] = {}
    regime_null: dict[str, np.ndarray] = {}
    regime_excess: dict[str, np.ndarray] = {}
    regime_rows: list[dict[str, float | int | str]] = []

    for regime_idx, label in enumerate(regime_labels):
        selected = regime_assignments == regime_idx
        block = states[selected]

        if block.shape[0] < 2:
            raise ValueError(
                f"{label} contains fewer than two sessions."
            )

        if show_progress:
            print(f"Computing {label.lower()} MI matrix...")

        regime_observed_matrix = _cross_asset_mi_matrix(
            block,
            num_states=num_states,
        )

        regime_null_matrix = _cross_asset_null_mean(
            block,
            num_states=num_states,
            n_permutations=n_permutations,
            random_seed=random_seed + 10_000 * (regime_idx + 1),
            show_progress=show_progress,
            description=f"{label} MI null permutations",
        )

        regime_excess_matrix = (
            regime_observed_matrix
            - regime_null_matrix
        )
        np.fill_diagonal(regime_excess_matrix, 0.0)

        regime_observed[label] = regime_observed_matrix
        regime_null[label] = regime_null_matrix
        regime_excess[label] = regime_excess_matrix

        regime_rows.append(
            {
                "Volatility Regime": label,
                "Sessions": int(block.shape[0]),
                "Mean Realised Volatility": float(
                    realised_volatility[selected].mean()
                ),
                "Observed Cross-Asset MI (bits)": (
                    _mean_upper_triangle(
                        regime_observed_matrix
                    )
                ),
                "Null Cross-Asset MI (bits)": (
                    _mean_upper_triangle(
                        regime_null_matrix
                    )
                ),
                "Excess Cross-Asset MI (bits)": (
                    _mean_upper_triangle(
                        regime_excess_matrix
                    )
                ),
            }
        )

    regime_summary = (
        pd.DataFrame(regime_rows)
        .set_index("Volatility Regime")
    )

    asset_order = _token_entropy_order(
        states,
        num_states=num_states,
    )

    return CrossAssetMIAnalysis(
        token_type=token_type,
        num_states=num_states,
        top_token_ids=top_ids,
        asset_cols=tuple(asset_cols),
        asset_order=asset_order,
        observed_mi=observed,
        null_mean_mi=null_mean,
        excess_mi=excess,
        overall_summary=overall_summary,
        realised_volatility=realised_volatility,
        volatility_tail_fraction=volatility_tail_fraction,
        volatility_thresholds=(
            float(thresholds_array[0]),
            float(thresholds_array[1]),
        ),
        regime_labels=regime_labels,
        regime_assignments=regime_assignments,
        regime_observed_mi=regime_observed,
        regime_null_mean_mi=regime_null,
        regime_excess_mi=regime_excess,
        regime_summary=regime_summary,
        n_permutations=int(n_permutations),
        random_seed=int(random_seed),
        
    )


def _blue_white_red_colormap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_red",
        [
            "#2166ac",
            "#ffffff",
            "#b2182b",
        ],
    ).copy()
    cmap.set_bad("#f0f0f0")
    return cmap


def _mi_colour_limit(
    matrices: Sequence[np.ndarray],
    *,
    percentile: float,
) -> float:
    if not 0 < percentile <= 100:
        raise ValueError(
            "colour_percentile must lie in (0, 100]."
        )

    values = []

    for matrix in matrices:
        mask = ~np.eye(
            matrix.shape[0],
            dtype=bool,
        )
        finite = np.abs(matrix[mask])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)

    if not values:
        return 1.0

    limit = float(
        np.percentile(
            np.concatenate(values),
            percentile,
        )
    )

    return limit if limit > 0 else 1.0


def plot_cross_asset_mi_heatmap(
    analysis: CrossAssetMIAnalysis,
    *,
    figure_size: tuple[float, float] = (10.0, 8.5),
    colour_percentile: float = 99.0,
    show_asset_labels: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the validation-period excess contemporaneous MI matrix."""
    if not isinstance(analysis, CrossAssetMIAnalysis):
        raise TypeError(
            "analysis must be returned by "
            "compute_cross_asset_mi_analysis."
        )

    order = analysis.asset_order
    matrix = analysis.excess_mi[
        np.ix_(order, order)
    ].copy()
    np.fill_diagonal(matrix, np.nan)

    limit = _mi_colour_limit(
        [analysis.excess_mi],
        percentile=colour_percentile,
    )

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )

    figure, axis = plt.subplots(
        figsize=figure_size
    )

    image = axis.imshow(
        matrix,
        cmap=_blue_white_red_colormap(),
        norm=norm,
        aspect="equal",
        interpolation="nearest",
    )

    if show_asset_labels:
        labels = [
            analysis.asset_cols[idx]
            for idx in order
        ]
        tick_step = max(
            1,
            len(labels) // 20,
        )
        positions = np.arange(
            0,
            len(labels),
            tick_step,
        )
        axis.set_xticks(positions)
        axis.set_yticks(positions)
        axis.set_xticklabels(
            [labels[idx] for idx in positions],
            rotation=90,
        )
        axis.set_yticklabels(
            [labels[idx] for idx in positions]
        )
    else:
        axis.set_xticks([])
        axis.set_yticks([])

    mean_excess = float(
        analysis.overall_summary[
            "Excess Cross-Asset MI (bits)"
        ].iloc[0]
    )

    axis.set_title(
        "Excess Cross-Asset Mutual Information over Null\n"
        f"{analysis.token_type}, {analysis.num_states} states, "
        f"mean off-diagonal = {mean_excess:.4f} bits"
    )
    axis.set_xlabel("Asset")
    axis.set_ylabel("Asset")

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.85,
    )
    colorbar.set_label(
        "Excess mutual information (bits)"
    )

    figure.tight_layout()
    plt.show()

    return figure, axis


def plot_cross_asset_mi_regime_heatmaps(
    analysis: CrossAssetMIAnalysis,
    *,
    figure_size: tuple[float, float] = (11.0, 5.0),
    clip: bool = True,
    colour_percentile: float = 99.0,
) -> tuple[Figure, np.ndarray]:
    """Plot low- and high-volatility excess-MI matrices.

    The volatility groups are defined by
    ``analysis.volatility_tail_fraction``:

        Low Volatility:
            Bottom fraction of sessions by realised volatility.

        High Volatility:
            Top fraction of sessions by realised volatility.

    For example:
        1/3 -> bottom and top terciles.
        1/2 -> bottom and top halves, with one middle session
               omitted when the number of sessions is odd.
    """
    if not isinstance(
        analysis,
        CrossAssetMIAnalysis,
    ):
        raise TypeError(
            "analysis must be returned by "
            "compute_cross_asset_mi_analysis."
        )

    if not isinstance(
        clip,
        bool,
    ):
        raise TypeError(
            "clip must be a boolean."
        )

    matrices = [
        analysis.regime_excess_mi[label]
        for label in analysis.regime_labels
    ]

    limit = _mi_colour_limit(
        matrices,
        percentile=(
            colour_percentile
            if clip
            else 100.0
        ),
    )

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )

    num_regimes = len(
        analysis.regime_labels
    )

    figure, axes = plt.subplots(
        1,
        num_regimes,
        figsize=figure_size,
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    order = analysis.asset_order
    image = None

    for axis, label, raw_matrix in zip(
        axes,
        analysis.regime_labels,
        matrices,
    ):
        matrix = raw_matrix[
            np.ix_(
                order,
                order,
            )
        ].copy()

        np.fill_diagonal(
            matrix,
            np.nan,
        )

        image = axis.imshow(
            matrix,
            cmap=_blue_white_red_colormap(),
            norm=norm,
            aspect="equal",
            interpolation="nearest",
        )

        row = analysis.regime_summary.loc[
            label
        ]

        axis.set_title(
            f"{label}\n"
            f"n={int(row['Sessions'])}, "
            f"mean excess="
            f"{row['Excess Cross-Asset MI (bits)']:.4f} bits"
        )

        axis.set_xticks([])
        axis.set_yticks([])

    percentage = (
        100.0
        * analysis.volatility_tail_fraction
    )

    figure.suptitle(
        "Cross-Asset Excess Mutual Information by "
        f"Volatility Tail ({percentage:.1f}% low/high) "
        f"({analysis.token_type}, {analysis.num_states} states)"
    )

    if image is None:
        raise RuntimeError(
            "No regime heatmaps were created."
        )

    colorbar = figure.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        pad=0.02,
    )

    colorbar.set_label(
        "Excess mutual information (bits)"
    )

    plt.show()

    return (
        figure,
        axes,
    )
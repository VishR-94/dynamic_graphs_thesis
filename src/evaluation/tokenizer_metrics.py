from __future__ import annotations

import random
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
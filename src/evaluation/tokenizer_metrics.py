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

        random_seed:
            Optional reproducibility seed. With None, a different
            random day or asset is selected on each call.

    Returns:
        Figure, axis, and selected day/asset/channel metadata.
    """
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
    decoded = torch.as_tensor(
        decoded_data["decoded"]
    )

    valid_mask = torch.as_tensor(
        decoded_data["valid_mask"],
        dtype=torch.bool,
    )

    if decoded.ndim != 4:
        raise ValueError(
            "decoded must have shape [S, T, N, C]."
        )

    (
        num_sessions,
        num_bars,
        num_assets,
        num_channels,
    ) = decoded.shape

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
            "Decoded channel metadata does not match its tensor."
        )

    if tuple(valid_mask.shape) != (
        num_sessions,
        num_bars,
    ):
        raise ValueError(
            "valid_mask must have shape [S, T]."
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

    if day is None:
        session_idx = rng.randrange(
            num_sessions
        )

    elif isinstance(day, int):
        session_idx = day

        if not 0 <= session_idx < num_sessions:
            raise IndexError(
                f"day index must be between 0 and "
                f"{num_sessions - 1}."
            )

    else:
        requested_date = pd.Timestamp(
            day
        ).normalize()

        matches = [
            idx
            for idx, date in enumerate(split_dates)
            if date == requested_date
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one session for "
                f"{requested_date.date()}, found "
                f"{len(matches)}."
            )

        session_idx = matches[0]

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

    if x_day.shape[0] != num_bars:
        raise ValueError(
            "Original and decoded bar counts do not match."
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

    decoded_values = (
        decoded[
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

    figure, axis = plt.subplots(
        figsize=(14, 5)
    )

    axis.plot(
        timestamps,
        true_values,
        color="black",
        linewidth=1.4,
        label="True",
        zorder=1,
    )

    axis.plot(
        timestamps[selected_valid_mask],
        decoded_values[selected_valid_mask],
        linewidth=1.2,
        label="Decoded",
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

    selection = {
        "session_idx": session_idx,
        "date": selected_date,
        "asset": selected_asset,
        "asset_idx": asset_idx,
        "channel": channel,
        "first_valid_bar": int(
            torch.nonzero(
                valid_mask[session_idx],
                as_tuple=False,
            )[0].item()
        ),
    }

    plt.show()

    return figure, axis, selection
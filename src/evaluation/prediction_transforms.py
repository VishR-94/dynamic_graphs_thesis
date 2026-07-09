import torch
from collections.abc import Sequence

####
# Usage:
# We want to be able to evaluate our models using metrics (such as MSE and RMSE) either on 
# raw price scale or log change scale. Log change scale is better since it is not impacted by
# the different levels of all series across assets. The issue is that some of our models will
# take as their input window normalised price data and therefore output (normalised) price data
# and some (e.g. ARIMA) will take as their input log change data and output log change data.
# Here we will have functions to 
# 1. Undo the window normalisation we do before passing data to neural models.
#    This will allow us to transform neural model predictions back to raw price scale
# 2. Undo the transformations we make to the inputs that were used to ensure the predictions are valid
#    candle data (all values>0, high>=low etc).     
# 3. Convert Raw price predictions to a cumulative log change equivalent. For example, if
#    we have a raw price prediction at t=[1,5,15,30,60], we can convert those to cumulative log
#    change from the origin to those horizon points.
# 4. Convert cumulative log change predictions back to raw prices. ARIMA will output all log changes
#    between t=[1,2,3,....,60]. We can use those along with the last price in the window to compute 
#    the predicted raw price at the points in our horizon.
# 5. Helper functions to compute cumulative log changes from one step log changes. This will allow us to 
#    take the one step log change predictions from ARIMA or seq-to-seq models and cumulate them to 
#    get cumulative log change predictions to each horizon point, which can then be converted back to raw
#    price prediction at the horizon and compared to raw price models.  
###

#function to undo the window normalisation on predictions from neural models
def inverse_window_normalisation(
    y_norm: torch.Tensor,
    target_norm_mean: torch.Tensor,
    target_norm_std: torch.Tensor,
) -> torch.Tensor:
    """
    Convert window-normalised values back to raw value space.

    Args:
        y_norm:
            Normalised predictions or targets.

            Shape:
                [H, N, C] or [B, H, N, C]

        target_norm_mean:
            Context-window mean for the target channels.

            Shape:
                [N, C] or [B, N, C]

        target_norm_std:
            Context-window standard deviation for the target channels.

            Shape:
                [N, C] or [B, N, C]

    Returns:
        Raw values with the same shape as y_norm.
    """
    if y_norm.ndim not in {3, 4}:
        raise ValueError(
            f"Expected y_norm to have shape [H, N, C] or [B, H, N, C], "
            f"got {tuple(y_norm.shape)}."
        )

    if target_norm_mean.shape != target_norm_std.shape:
        raise ValueError(
            "target_norm_mean and target_norm_std must have the same shape. "
            f"Got {tuple(target_norm_mean.shape)} and "
            f"{tuple(target_norm_std.shape)}."
        )

    if y_norm.ndim == 3:
        expected_stats_shape = y_norm.shape[1:]

        if target_norm_mean.shape != expected_stats_shape:
            raise ValueError(
                "Stats shape is incompatible with y_norm shape. "
                f"Expected {tuple(expected_stats_shape)}, "
                f"got {tuple(target_norm_mean.shape)}."
            )

        mean = target_norm_mean.unsqueeze(0)
        std = target_norm_std.unsqueeze(0)

    else:
        expected_stats_shape = (y_norm.shape[0], y_norm.shape[2], y_norm.shape[3])

        if target_norm_mean.shape != expected_stats_shape:
            raise ValueError(
                "Stats shape is incompatible with y_norm shape. "
                f"Expected {tuple(expected_stats_shape)}, "
                f"got {tuple(target_norm_mean.shape)}."
            )

        mean = target_norm_mean.unsqueeze(1)
        std = target_norm_std.unsqueeze(1)

    y_raw = y_norm * std + mean

    return y_raw

#function to undo the tranformations that ensure valid candle prediction. Note that for this to work
#we need to have the correct transformed_channels to get the output_channels we want. For example, if we have
#'open' in output channel, we need 'log_close' and 'log_open_to_close' as transformed_channels since they are
#both needed to compute raw open. 
def valid_transformed_to_raw_ohlcv(
    y_transformed: torch.Tensor,
    transformed_channels: Sequence[str],
    output_channels: Sequence[str] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert valid-candle transformed features back to raw OHLCV values.

    Args:
        y_transformed:
            Tensor of transformed candle features.

            Shape can be:
                [H, N, C]
                [B, H, N, C]
                or any shape ending in C.

        transformed_channels:
            Names of the transformed channels in y_transformed.

            Possible names:
                log_close
                log_open_to_close
                log_upper_wick_ratio
                log_lower_wick_ratio
                log_volume

        output_channels:
            Raw channels to return.

            Possible names:
                open
                high
                low
                close
                volume

            If None, the function returns every raw channel that can be
            reconstructed from transformed_channels, in this order:
                open, high, low, close, volume

        eps:
            Small positive value used when inverting log-ratio features.

    Returns:
        Tensor of raw values with the same leading dimensions as y_transformed.
        The final dimension is len(output_channels).
    """
    transformed_channels = list(transformed_channels)

    if y_transformed.shape[-1] != len(transformed_channels):
        raise ValueError(
            "Final dimension of y_transformed must match transformed_channels. "
            f"Got tensor shape {tuple(y_transformed.shape)} and "
            f"{len(transformed_channels)} channel names."
        )

    available = set(transformed_channels)

    def has(required_channels: Sequence[str]) -> bool:
        return all(channel in available for channel in required_channels)

    can_reconstruct = {
        "open": has(["log_close", "log_open_to_close"]),
        "high": has(
            [
                "log_close",
                "log_open_to_close",
                "log_upper_wick_ratio",
            ]
        ),
        "low": has(
            [
                "log_close",
                "log_open_to_close",
                "log_lower_wick_ratio",
            ]
        ),
        "close": has(["log_close"]),
        "volume": has(["log_volume"]),
    }

    if output_channels is None:
        output_channels = [
            channel
            for channel in ["open", "high", "low", "close", "volume"]
            if can_reconstruct[channel]
        ]
    else:
        output_channels = list(output_channels)

    if len(output_channels) == 0:
        raise ValueError("No output channels were requested or reconstructable.")

    for channel in output_channels:
        if channel not in can_reconstruct:
            raise ValueError(
                f"Unknown output channel: {channel}. "
                "Expected one of: open, high, low, close, volume."
            )

        if not can_reconstruct[channel]:
            raise ValueError(
                f"Cannot reconstruct '{channel}' from transformed channels "
                f"{transformed_channels}."
            )

    def get_transformed_channel(channel: str) -> torch.Tensor:
        channel_idx = transformed_channels.index(channel)
        return y_transformed[..., channel_idx]

    raw_values = {}

    if any(channel in output_channels for channel in ["open", "high", "low", "close"]):
        log_close = get_transformed_channel("log_close")
        close_price = torch.exp(log_close)

        raw_values["close"] = close_price

    if any(channel in output_channels for channel in ["open", "high", "low"]):
        log_open_to_close = get_transformed_channel("log_open_to_close")
        open_price = close_price * torch.exp(log_open_to_close)

        raw_values["open"] = open_price

        body_high = torch.maximum(open_price, close_price)
        body_low = torch.minimum(open_price, close_price)

    if "high" in output_channels:
        log_upper_wick_ratio = get_transformed_channel("log_upper_wick_ratio")
        upper_wick_ratio = (
            torch.exp(log_upper_wick_ratio) - eps
        ).clamp_min(0.0)

        raw_values["high"] = body_high * (1.0 + upper_wick_ratio)

    if "low" in output_channels:
        log_lower_wick_ratio = get_transformed_channel("log_lower_wick_ratio")
        lower_wick_ratio = (
            torch.exp(log_lower_wick_ratio) - eps
        ).clamp_min(0.0)

        raw_values["low"] = body_low / (1.0 + lower_wick_ratio)

    if "volume" in output_channels:
        log_volume = get_transformed_channel("log_volume")
        raw_values["volume"] = torch.exp(log_volume)

    y_raw = torch.stack(
        [
            raw_values[channel]
            for channel in output_channels
        ],
        dim=-1,
    )

    return y_raw
    
#function to convert raw price predictions (at horizon points) to a cumulative log change
#we need the last raw price in the context window and the predicted raw prices
def raw_to_cumulative_log_change(
    y_raw: torch.Tensor,
    last_context_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert raw future values into cumulative log changes from the forecast origin.

    Args:
        y_raw:
            Raw future values.

            Shape:
                [H, N, C] or [B, H, N, C]

        last_context_target:
            Last observed raw value at the forecast origin.

            Shape:
                [N, C] or [B, N, C]

        eps:
            Small positive value used to avoid log(0).

    Returns:
        Cumulative log changes with the same shape as y_raw.

        Definition:
            log_change[h] = log(y_raw[h]) - log(last_context_target)
    """
    if y_raw.ndim not in {3, 4}:
        raise ValueError(
            f"Expected y_raw to have shape [H, N, C] or [B, H, N, C], "
            f"got {tuple(y_raw.shape)}."
        )

    if y_raw.ndim == 3:
        expected_last_shape = y_raw.shape[1:]

        if last_context_target.shape != expected_last_shape:
            raise ValueError(
                "last_context_target shape is incompatible with y_raw shape. "
                f"Expected {tuple(expected_last_shape)}, "
                f"got {tuple(last_context_target.shape)}."
            )

        last = last_context_target.unsqueeze(0)

    else:
        expected_last_shape = (y_raw.shape[0], y_raw.shape[2], y_raw.shape[3])

        if last_context_target.shape != expected_last_shape:
            raise ValueError(
                "last_context_target shape is incompatible with y_raw shape. "
                f"Expected {tuple(expected_last_shape)}, "
                f"got {tuple(last_context_target.shape)}."
            )

        last = last_context_target.unsqueeze(1)

    cumulative_log_change = (
        torch.log(y_raw.clamp_min(eps))
        - torch.log(last.clamp_min(eps))
    )

    return cumulative_log_change

#function to convert a cumulative log change back to a raw price
def cumulative_log_change_to_raw(
    cumulative_log_change: torch.Tensor,
    last_context_target: torch.Tensor,
) -> torch.Tensor:
    """
    Convert cumulative log changes back to raw values.

    Args:
        cumulative_log_change:
            Cumulative log changes from the forecast origin.

            Shape:
                [H, N, C] or [B, H, N, C]

        last_context_target:
            Last observed raw value at the forecast origin.

            Shape:
                [N, C] or [B, N, C]

    Returns:
        Raw values with the same shape as cumulative_log_change.

        Definition:
            y_raw[h] = last_context_target * exp(cumulative_log_change[h])
    """
    if cumulative_log_change.ndim not in {3, 4}:
        raise ValueError(
            "Expected cumulative_log_change to have shape [H, N, C] "
            f"or [B, H, N, C], got {tuple(cumulative_log_change.shape)}."
        )

    if cumulative_log_change.ndim == 3:
        expected_last_shape = cumulative_log_change.shape[1:]

        if last_context_target.shape != expected_last_shape:
            raise ValueError(
                "last_context_target shape is incompatible with "
                "cumulative_log_change shape. "
                f"Expected {tuple(expected_last_shape)}, "
                f"got {tuple(last_context_target.shape)}."
            )

        last = last_context_target.unsqueeze(0)

    else:
        expected_last_shape = (
            cumulative_log_change.shape[0],
            cumulative_log_change.shape[2],
            cumulative_log_change.shape[3],
        )

        if last_context_target.shape != expected_last_shape:
            raise ValueError(
                "last_context_target shape is incompatible with "
                "cumulative_log_change shape. "
                f"Expected {tuple(expected_last_shape)}, "
                f"got {tuple(last_context_target.shape)}."
            )

        last = last_context_target.unsqueeze(1)

    y_raw = last * torch.exp(cumulative_log_change)

    return y_raw

#function to compute cumulative returns at each horizon point from one step log returns
#for statistical models we will take all one step return prediction to horizon 
#and put them in a tensor of shape [max_horizon,N,C]
def one_step_returns_to_cumulative_horizons(
    one_step_returns: torch.Tensor,
    horizons: list[int],
) -> torch.Tensor:
    """
    Convert future one-step log returns into cumulative log changes at horizons.

    Args:
        one_step_returns:
            Future one-step log returns.

            Shape:
                [max_horizon, N, C] or [B, max_horizon, N, C]

        horizons:
            Forecast horizons, e.g. [1, 5, 15, 30, 60].

    Returns:
        Cumulative log changes at the requested horizons.

        Shape:
            [num_horizons, N, C] or [B, num_horizons, N, C]

    Definition:
        output[h] = sum of one_step_returns from step 1 up to horizon h.
    """
    if one_step_returns.ndim not in {3, 4}:
        raise ValueError(
            "Expected one_step_returns to have shape [max_horizon, N, C] "
            f"or [B, max_horizon, N, C], got {tuple(one_step_returns.shape)}."
        )

    if len(horizons) == 0:
        raise ValueError("horizons must contain at least one value.")

    if min(horizons) < 1:
        raise ValueError(f"All horizons must be >= 1, got {horizons}.")

    max_horizon = one_step_returns.shape[-3]

    if max(horizons) > max_horizon:
        raise ValueError(
            f"Maximum requested horizon is {max(horizons)}, but "
            f"one_step_returns only contains {max_horizon} future steps."
        )

    cumulative_path = one_step_returns.cumsum(dim=-3)

    horizon_indices = torch.tensor(
        [horizon - 1 for horizon in horizons],
        dtype=torch.long,
        device=one_step_returns.device,
    )

    cumulative_horizons = cumulative_path.index_select(
        dim=-3,
        index=horizon_indices,
    )

    return cumulative_horizons
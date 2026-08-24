from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from functools import partial
import torch
from src.evaluation.prediction_transforms import (
    raw_to_cumulative_log_change,
)

'''
 Usage as follows:
 Each metric takes y_pred and y_true. These are shape [H,N,C] or [B,H,N,C]
 First it computes pointwise metric called values which returns tensor of same shape as y_pred/y_true
 So values contains the errors for a given batch (B) for a given horizon (H) for a given asset (N) for a given channel (C)
 Then we use reduce_metric to average over chosen channels
 For example, if y_pred/y_true are [B,H,N,C] and reduce_dims=c(0,2)
 we average over batch and asset and return a tensor of shape [H,C] - error per horizon per channel
 if reduce_dims = None, we return a single number which is the error metric averaged over B,H,N,C
'''

def validate_prediction_shapes(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> None:
    """
    Check that prediction and target tensors have the same shape.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            "y_pred and y_true must have the same shape. "
            f"Got {tuple(y_pred.shape)} and {tuple(y_true.shape)}."
        )


def reduce_metric(
    values: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Reduce a metric tensor over selected dimensions.

    Args:
        values:
            Tensor of pointwise metric values.

        reduce_dims:
            Dimensions to average over.

            If None, average over all dimensions.

            For tensors with shape [B, H, N, C]:
                B = batch/examples
                H = horizons
                N = assets
                C = channels

            Examples:
                reduce_dims=None:
                    one scalar over everything

                reduce_dims=(0, 2, 3):
                    keep horizon dimension only, giving shape [H]

                reduce_dims=(0, 1, 3):
                    keep asset dimension only, giving shape [N]

                reduce_dims=(0, 1, 2):
                    keep channel dimension only, giving shape [C]
    """
    if reduce_dims is None:
        return values.mean()

    reduce_dims = tuple(reduce_dims)

    if len(reduce_dims) == 0:
        return values

    return values.mean(dim=reduce_dims)


def absolute_error_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """
    Return pointwise absolute errors without reducing dimensions.

    Input/output shape:
        [B, H, N, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    return torch.abs(
        y_pred - y_true
    )


def mase_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mase_scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return pointwise MASE contributions without reducing dimensions.

    Input/output shape:
        y_pred, y_true: [B, H, N, C]
        mase_scale:     [N, C]
        output:         [B, H, N, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if not isinstance(
        mase_scale,
        torch.Tensor,
    ):
        raise TypeError(
            "mase_scale must be a torch.Tensor."
        )

    if mase_scale.ndim != 2:
        raise ValueError(
            "Expected mase_scale to have shape [N, C], "
            f"got {tuple(mase_scale.shape)}."
        )

    expected_scale_shape = (
        y_pred.shape[-2],
        y_pred.shape[-1],
    )

    if mase_scale.shape != expected_scale_shape:
        raise ValueError(
            "mase_scale has an incompatible shape. "
            f"Expected {expected_scale_shape}, "
            f"got {tuple(mase_scale.shape)}."
        )

    if not torch.isfinite(
        mase_scale
    ).all():
        raise ValueError(
            "mase_scale contains NaN or infinite values."
        )

    mase_scale = mase_scale.to(
        device=y_pred.device,
        dtype=y_pred.dtype,
    )

    valid_scale = (
        mase_scale > eps
    )

    safe_scale = torch.where(
        valid_scale,
        mase_scale,
        torch.full_like(
            mase_scale,
            torch.nan,
        ),
    )

    return (
        absolute_error_values(
            y_pred=y_pred,
            y_true=y_true,
        )
        / safe_scale
    )


def persistence_win_score_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    persistence_pred: torch.Tensor,
    tie_value: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> torch.Tensor:
    """
    Return pointwise win scores against Persistence.

    Each element is:
        1.0 if model error is lower
        0.0 if model error is higher
        tie_value if errors are numerically tied

    Input/output shape:
        [B, H, N, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    validate_prediction_shapes(
        y_pred=persistence_pred,
        y_true=y_true,
    )

    if not 0.0 <= tie_value <= 1.0:
        raise ValueError(
            "tie_value must lie between 0 and 1, "
            f"got {tie_value}."
        )

    model_absolute_error = (
        absolute_error_values(
            y_pred=y_pred,
            y_true=y_true,
        )
    )

    persistence_absolute_error = (
        absolute_error_values(
            y_pred=persistence_pred,
            y_true=y_true,
        )
    )

    ties = torch.isclose(
        model_absolute_error,
        persistence_absolute_error,
        rtol=rtol,
        atol=atol,
    )

    wins = (
        model_absolute_error
        < persistence_absolute_error
    ) & ~ties

    return (
        wins.to(
            dtype=model_absolute_error.dtype
        )
        + tie_value
        * ties.to(
            dtype=model_absolute_error.dtype
        )
    )


def directional_accuracy_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Return pointwise cumulative-change sign correctness scores.

    Each element is:

        1.0 when prediction and target have the same sign;
        0.0 otherwise.

    ``torch.sign`` maps negative, zero and positive values to -1, 0 and
    +1 respectively.  Therefore an unchanged prediction is counted as
    correct only when the realised change is also exactly unchanged.

    Input/output shape:
        [B, H, N, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    return (
        torch.sign(y_pred)
        == torch.sign(y_true)
    ).to(dtype=y_pred.dtype)

def _normalise_reduction_dims(
    *,
    ndim: int,
    reduce_dims: Sequence[int] | None,
) -> tuple[int, ...]:
    """Return validated, non-negative reduction dimensions."""
    if reduce_dims is None:
        return tuple(range(ndim))

    normalised: list[int] = []

    for dim in reduce_dims:
        if not isinstance(dim, int):
            raise TypeError(
                "reduce_dims must contain integers, "
                f"got {type(dim).__name__}."
            )

        resolved = dim + ndim if dim < 0 else dim

        if not 0 <= resolved < ndim:
            raise IndexError(
                f"Reduction dimension {dim} is invalid for a "
                f"tensor with {ndim} dimensions."
            )

        normalised.append(resolved)

    if len(set(normalised)) != len(normalised):
        raise ValueError(
            "reduce_dims must not contain duplicate dimensions."
        )

    return tuple(sorted(normalised))


def reduce_quantile(
    values: torch.Tensor,
    quantile: float,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Reduce ``values`` with a linearly interpolated quantile.

    The implementation keeps all calculations in float32 and uses
    sorting rather than ``torch.quantile``.  This avoids the float64
    conversion that is unsupported by MPS while matching PyTorch's
    default linear interpolation rule.
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            "values must be a torch.Tensor."
        )

    if values.numel() == 0:
        raise ValueError(
            "Cannot calculate a quantile of an empty tensor."
        )

    if not isinstance(quantile, (float, int)):
        raise TypeError(
            "quantile must be numeric."
        )

    quantile = float(quantile)

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "quantile must lie in [0, 1], "
            f"got {quantile}."
        )

    working = values.to(dtype=torch.float32)

    dims = _normalise_reduction_dims(
        ndim=working.ndim,
        reduce_dims=reduce_dims,
    )

    if len(dims) == 0:
        return working

    kept_dims = tuple(
        dim
        for dim in range(working.ndim)
        if dim not in dims
    )

    permutation = kept_dims + dims

    if permutation != tuple(range(working.ndim)):
        working = working.permute(permutation)

    kept_shape = tuple(
        values.shape[dim]
        for dim in kept_dims
    )

    reduced_size = 1
    for dim in dims:
        reduced_size *= int(values.shape[dim])

    flattened = working.reshape(
        *kept_shape,
        reduced_size,
    )

    sorted_values = torch.sort(
        flattened,
        dim=-1,
    ).values

    position = quantile * (reduced_size - 1)
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        reduced_size - 1,
    )
    interpolation_weight = float(
        position - lower_index
    )

    lower = sorted_values[..., lower_index]

    if upper_index == lower_index:
        return lower

    upper = sorted_values[..., upper_index]

    return lower + interpolation_weight * (upper - lower)


def absolute_error_quantile(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    quantile: float,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return a quantile of pointwise absolute errors."""
    values = absolute_error_values(
        y_pred=y_pred,
        y_true=y_true,
    )

    return reduce_quantile(
        values,
        quantile=quantile,
        reduce_dims=reduce_dims,
    )


def median_absolute_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Median pointwise absolute error."""
    return absolute_error_quantile(
        y_pred=y_pred,
        y_true=y_true,
        quantile=0.50,
        reduce_dims=reduce_dims,
    )


def p95_absolute_error(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """95th percentile of pointwise absolute error."""
    return absolute_error_quantile(
        y_pred=y_pred,
        y_true=y_true,
        quantile=0.95,
        reduce_dims=reduce_dims,
    )


def mae(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Mean absolute error.
    """
    values = absolute_error_values(
        y_pred=y_pred,
        y_true=y_true,
    )

    return reduce_metric(
        values,
        reduce_dims=reduce_dims,
    )


def mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Mean squared error.

    Args:
        y_pred:
            Prediction tensor.

        y_true:
            Ground-truth tensor.

        reduce_dims:
            Dimensions to average over. If None, average over all dimensions.
    """
    validate_prediction_shapes(y_pred, y_true)

    values = (y_pred - y_true).pow(2)

    return reduce_metric(values, reduce_dims=reduce_dims)


def rmse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Root mean squared error.

    Args:
        y_pred:
            Prediction tensor.

        y_true:
            Ground-truth tensor.

        reduce_dims:
            Dimensions to average over. If None, average over all dimensions.
    """
    return torch.sqrt(
        mse(
            y_pred=y_pred,
            y_true=y_true,
            reduce_dims=reduce_dims,
        )
    )

def relative_mae_vs_persistence(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    persistence_pred: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute MAE relative to a persistence forecast.

    The absolute errors are reduced before taking the ratio:

        MAE(model) / MAE(persistence)

    Values below 1 indicate that the model outperforms persistence.
    Values above 1 indicate that persistence performs better.

    Args:
        y_pred:
            Model predictions with shape [B, H, N, C].
        y_true:
            Ground truth with shape [B, H, N, C].
        persistence_pred:
            Persistence predictions with shape [B, H, N, C].
        reduce_dims:
            Dimensions over which to average absolute errors.
            If None, all dimensions are reduced.
        eps:
            Minimum persistence MAE required for a defined ratio.

    Returns:
        Relative MAE after the requested reductions. Entries for which
        persistence MAE is no greater than eps are returned as NaN.
    """
    validate_prediction_shapes(y_pred, y_true)
    validate_prediction_shapes(persistence_pred, y_true)

    model_absolute_error = (
        absolute_error_values(
            y_pred=y_pred,
            y_true=y_true,
        )
    )

    persistence_absolute_error = (
        absolute_error_values(
            y_pred=persistence_pred,
            y_true=y_true,
        )
    )

    model_mae = reduce_metric(
        model_absolute_error,
        reduce_dims=reduce_dims,
    )

    persistence_mae = reduce_metric(
        persistence_absolute_error,
        reduce_dims=reduce_dims,
    )

    return torch.where(
        persistence_mae > eps,
        model_mae / persistence_mae,
        torch.full_like(model_mae, torch.nan),
    )


def persistence_win_rate(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    persistence_pred: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
    tie_value: float = 0.5,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the proportion of pointwise errors that beat persistence.

    Each forecast element receives:

        1.0 if the model error is lower than persistence error;
        0.0 if the model error is higher than persistence error;
        tie_value if the errors are numerically equal.

    Args:
        y_pred:
            Model predictions with shape [B, H, N, C].
        y_true:
            Ground truth with shape [B, H, N, C].
        persistence_pred:
            Persistence predictions with shape [B, H, N, C].
        reduce_dims:
            Dimensions over which to average the pointwise scores.
            If None, all dimensions are reduced.
        tie_value:
            Score assigned to ties. Defaults to 0.5.
        rtol:
            Relative tolerance used to identify numerical ties.
        atol:
            Absolute tolerance used to identify numerical ties.

    Returns:
        Win-rate proportions between 0 and 1 after the requested
        reductions.
    """
    pointwise_scores = (
        persistence_win_score_values(
            y_pred=y_pred,
            y_true=y_true,
            persistence_pred=persistence_pred,
            tie_value=tie_value,
            rtol=rtol,
            atol=atol,
        )
    )

    return reduce_metric(
        pointwise_scores,
        reduce_dims=reduce_dims,
    )


def directional_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the proportion of predictions with the correct sign.

    The caller is responsible for supplying predictions and targets in
    the intended change space.  ``ForecastEvaluator`` registers this
    metric in cumulative-log-change space.

    Values lie in [0, 1].  Multiply by 100, or use percentage display
    formatting, to report the result as a percentage.
    """
    values = directional_accuracy_values(
        y_pred=y_pred,
        y_true=y_true,
    )

    return reduce_metric(
        values,
        reduce_dims=reduce_dims,
    )


def compute_mase_scale(
    train_split: dict[str, Any],
    channels: Sequence[str],
) -> torch.Tensor:
    """
    Compute the standard one-step MASE scale from training data.

    For each asset and channel:

        scale[n, c] =
            mean over all within-day adjacent observations of
            abs(y[t] - y[t - 1])

    Differences are calculated separately within each daily sample, so
    overnight differences are never included.

    Args:
        train_split:
            Cleaned training split containing daily samples.

        channels:
            Target channels in the same order as the prediction output.

    Returns:
        Tensor with shape [N, C], where:
            N = number of assets
            C = number of requested channels
    """
    required_keys = {
        "samples",
        "channels",
        "asset_cols",
    }

    missing_keys = required_keys.difference(train_split)

    if missing_keys:
        raise KeyError(
            "train_split is missing required keys: "
            f"{sorted(missing_keys)}."
        )

    channels = list(channels)

    if len(channels) == 0:
        raise ValueError(
            "At least one channel is required to compute MASE scale."
        )

    available_channels = list(train_split["channels"])

    missing_channels = [
        channel
        for channel in channels
        if channel not in available_channels
    ]

    if missing_channels:
        raise ValueError(
            "Requested MASE channels are missing from the "
            f"training split: {missing_channels}. "
            f"Available channels: {available_channels}."
        )

    samples = train_split["samples"]

    if len(samples) == 0:
        raise ValueError(
            "Cannot compute MASE scale from an empty training split."
        )

    channel_indices = torch.tensor(
        [
            available_channels.index(channel)
            for channel in channels
        ],
        dtype=torch.long,
    )

    num_assets = len(train_split["asset_cols"])
    num_channels = len(channels)

    # Use float64 while accumulating the training statistics.
    absolute_difference_sum = torch.zeros(
        num_assets,
        num_channels,
        dtype=torch.float64,
    )

    num_differences = 0

    for sample_idx, (x_day, _, day) in enumerate(samples):
        if not isinstance(x_day, torch.Tensor):
            raise TypeError(
                f"Training sample {sample_idx} ({day}) is not "
                "a torch.Tensor."
            )

        if x_day.ndim != 3:
            raise ValueError(
                "Expected each training sample to have shape [T, N, D], "
                f"got {tuple(x_day.shape)} for sample "
                f"{sample_idx} ({day})."
            )

        if x_day.shape[0] < 2:
            raise ValueError(
                f"Training sample {sample_idx} ({day}) has fewer "
                "than two observations."
            )

        if x_day.shape[1] != num_assets:
            raise ValueError(
                f"Training sample {sample_idx} ({day}) has "
                f"{x_day.shape[1]} assets, expected {num_assets}."
            )

        values = (
            x_day
            .index_select(2, channel_indices)
            .to(dtype=torch.float64)
        )

        if not torch.isfinite(values).all():
            raise ValueError(
                f"Training sample {sample_idx} ({day}) contains "
                "NaN or infinite target values."
            )

        within_day_differences = (
            values[1:] - values[:-1]
        ).abs()

        absolute_difference_sum += (
            within_day_differences.sum(dim=0)
        )

        num_differences += within_day_differences.shape[0]

    if num_differences == 0:
        raise ValueError(
            "No within-day differences were available for "
            "MASE scale calculation."
        )

    return absolute_difference_sum / num_differences

def mase(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mase_scale: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute standard mean absolute scaled error.

    Each raw forecast error is divided by the corresponding
    training-set one-step naive error scale:

        abs(y_pred - y_true) / mase_scale

    Args:
        y_pred:
            Raw predictions with shape [B, H, N, C].

        y_true:
            Raw ground truth with shape [B, H, N, C].

        mase_scale:
            Training-derived scale with shape [N, C].

        reduce_dims:
            Dimensions over which to average the scaled errors.
            If None, average over all dimensions.

        eps:
            Scales no greater than this value are treated as undefined.

    Returns:
        MASE after the requested reduction.
    """
    scaled_absolute_error = mase_values(
        y_pred=y_pred,
        y_true=y_true,
        mase_scale=mase_scale,
        eps=eps,
    )

    return reduce_metric(
        scaled_absolute_error,
        reduce_dims=reduce_dims,
    )


def pearson_correlation(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] = (0, 2),
) -> torch.Tensor:
    """
    Compute pooled Pearson correlation in cumulative log-change space.

    For each horizon and channel, observations are pooled across
    forecast windows and assets:

        [B, H, N, C] -> [B * N, H * C]

    Correlation is then calculated separately for each horizon-channel
    column.

    Inputs are converted to float64 before centring and accumulation to
    avoid incorrectly treating small but non-zero intraday return
    variation as zero.

    Output:
        [H, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true to have shape "
            f"[B, H, N, C], got {tuple(y_pred.shape)}."
        )

    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "pearson_correlation currently supports "
            "reduce_dims=(0, 2) only."
        )

    batch_size, num_horizons, num_assets, num_channels = (
        y_pred.shape
    )

    # [B, H, N, C] -> [B * N, H * C]
    pred_flat = (
        y_pred
        .permute(0, 2, 1, 3)
        .reshape(
            batch_size * num_assets,
            num_horizons * num_channels,
        )
        .to(dtype=torch.float64)
    )

    true_flat = (
        y_true
        .permute(0, 2, 1, 3)
        .reshape(
            batch_size * num_assets,
            num_horizons * num_channels,
        )
        .to(dtype=torch.float64)
    )

    pred_centred = (
        pred_flat
        - pred_flat.mean(
            dim=0,
            keepdim=True,
        )
    )

    true_centred = (
        true_flat
        - true_flat.mean(
            dim=0,
            keepdim=True,
        )
    )

    covariance_sum = (
        pred_centred
        * true_centred
    ).sum(dim=0)

    pred_sum_squared = (
        pred_centred
        .square()
        .sum(dim=0)
    )

    true_sum_squared = (
        true_centred
        .square()
        .sum(dim=0)
    )

    valid_correlation = (
        pred_sum_squared > 0
    ) & (
        true_sum_squared > 0
    )

    denominator = torch.sqrt(
        pred_sum_squared
        * true_sum_squared
    )

    safe_denominator = torch.where(
        valid_correlation,
        denominator,
        torch.ones_like(
            denominator
        ),
    )

    correlations = (
        covariance_sum
        / safe_denominator
    )

    correlations = torch.where(
        valid_correlation,
        correlations,
        torch.full_like(
            correlations,
            torch.nan,
        ),
    )

    # Protect against tiny floating-point excursions outside [-1, 1].
    correlations = correlations.clamp(
        min=-1.0,
        max=1.0,
    )

    return correlations.reshape(
        num_horizons,
        num_channels,
    )


def assetwise_temporal_pearson_correlation(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] = (0, 2),
    eps: float = 0.0,
) -> torch.Tensor:
    """Compute Pearson correlation through time for each asset.

    Correlation is calculated separately across forecast windows for
    every asset, horizon and channel.  The valid asset-level
    correlations are then averaged across assets:

        [B, H, N, C] -> [H, N, C] -> [H, C]

    Pairwise non-finite observations are ignored.  This allows the same
    implementation to be used for horizon-aligned forecast-series log
    returns, whose first observation in each trading session is
    intentionally undefined.

    Inputs are accumulated in float64 so small intraday price changes do
    not disappear through centring or squaring.
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true to have shape [B, H, N, C], "
            f"got {tuple(y_pred.shape)}."
        )

    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "assetwise_temporal_pearson_correlation currently supports "
            "reduce_dims=(0, 2) only."
        )

    if y_pred.shape[0] < 2:
        raise ValueError(
            "Asset-wise temporal correlation requires at least two "
            "forecast windows."
        )

    predicted = y_pred.to(dtype=torch.float64)
    realised = y_true.to(dtype=torch.float64)

    valid_pair = (
        torch.isfinite(predicted)
        & torch.isfinite(realised)
    )

    safe_predicted = torch.where(
        valid_pair,
        predicted,
        torch.zeros_like(predicted),
    )
    safe_realised = torch.where(
        valid_pair,
        realised,
        torch.zeros_like(realised),
    )

    observation_count = valid_pair.sum(
        dim=0,
    ).to(dtype=torch.float64)

    predicted_sum = safe_predicted.sum(dim=0)
    realised_sum = safe_realised.sum(dim=0)
    predicted_squared_sum = safe_predicted.square().sum(dim=0)
    realised_squared_sum = safe_realised.square().sum(dim=0)
    cross_sum = (safe_predicted * safe_realised).sum(dim=0)

    numerator = (
        observation_count * cross_sum
        - predicted_sum * realised_sum
    )

    predicted_variation = torch.clamp_min(
        observation_count * predicted_squared_sum
        - predicted_sum.square(),
        0.0,
    )
    realised_variation = torch.clamp_min(
        observation_count * realised_squared_sum
        - realised_sum.square(),
        0.0,
    )

    denominator = torch.sqrt(
        predicted_variation
        * realised_variation
    )

    valid_correlation = (
        observation_count > 1
    ) & (
        denominator > float(eps)
    )

    safe_denominator = torch.where(
        valid_correlation,
        denominator,
        torch.ones_like(denominator),
    )

    asset_correlations = torch.where(
        valid_correlation,
        numerator / safe_denominator,
        torch.full_like(denominator, torch.nan),
    ).clamp(min=-1.0, max=1.0)

    # [H, N, C] -> [H, C]
    return torch.nanmean(
        asset_correlations,
        dim=1,
    )


def forecast_series_log_return_values(
    y_pred_raw: torch.Tensor,
    y_true_raw: torch.Tensor,
    sample_idx: torch.Tensor,
    origin_idx: torch.Tensor,
    *,
    expected_origin_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Construct log returns from each horizon-aligned price series.

    For every horizon and asset, the prediction sequence is treated as a
    time series in its own right.  Consecutive log returns are formed
    only when two forecast origins belong to the same trading session
    and are exactly one expected origin stride apart.  Overnight changes
    and gaps in the saved prediction sequence are therefore excluded.

    The returned tensors retain shape ``[B, H, N, C]``.  Entries without
    a valid preceding within-session forecast are NaN.  A pair is also
    marked NaN when either predicted or realised price is non-finite or
    non-positive, because its log return is undefined.  The third return
    value is the validated or inferred origin stride.
    """
    validate_prediction_shapes(
        y_pred=y_pred_raw,
        y_true=y_true_raw,
    )

    if y_pred_raw.ndim != 4:
        raise ValueError(
            "Expected y_pred_raw and y_true_raw to have shape "
            f"[B, H, N, C], got {tuple(y_pred_raw.shape)}."
        )

    batch_size = int(y_pred_raw.shape[0])

    if batch_size < 2:
        raise ValueError(
            "Forecast-series log returns require at least two "
            "forecast windows."
        )

    metadata: dict[str, torch.Tensor] = {
        "sample_idx": sample_idx,
        "origin_idx": origin_idx,
    }

    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    metadata_cpu: dict[str, torch.Tensor] = {}

    for name, values in metadata.items():
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )
        if values.ndim != 1 or values.shape[0] != batch_size:
            raise ValueError(
                f"{name} must have shape [{batch_size}], got "
                f"{tuple(values.shape)}."
            )
        if values.dtype not in integer_dtypes:
            raise TypeError(
                f"{name} must use an integer dtype, got "
                f"{values.dtype}."
            )

        metadata_cpu[name] = (
            values
            .detach()
            .cpu()
            .to(dtype=torch.long)
        )

    sample_values = metadata_cpu["sample_idx"].tolist()
    origin_values = metadata_cpu["origin_idx"].tolist()

    ordered_positions = sorted(
        range(batch_size),
        key=lambda position: (
            sample_values[position],
            origin_values[position],
        ),
    )

    order_cpu = torch.tensor(
        ordered_positions,
        dtype=torch.long,
    )

    ordered_sample_idx = metadata_cpu["sample_idx"].index_select(
        0,
        order_cpu,
    )
    ordered_origin_idx = metadata_cpu["origin_idx"].index_select(
        0,
        order_cpu,
    )

    same_session = (
        ordered_sample_idx[1:]
        == ordered_sample_idx[:-1]
    )
    origin_differences = (
        ordered_origin_idx[1:]
        - ordered_origin_idx[:-1]
    )

    within_session_differences = origin_differences[
        same_session
    ]

    if within_session_differences.numel() == 0:
        raise ValueError(
            "No within-session forecast pairs were available."
        )

    if torch.any(within_session_differences <= 0):
        raise ValueError(
            "origin_idx must be unique within each trading session."
        )

    if expected_origin_stride is None:
        resolved_stride = int(
            within_session_differences.min().item()
        )
    else:
        if not isinstance(expected_origin_stride, int):
            raise TypeError(
                "expected_origin_stride must be an integer or None."
            )
        resolved_stride = int(expected_origin_stride)

    if resolved_stride <= 0:
        raise ValueError(
            "expected_origin_stride must be positive."
        )

    if torch.any(
        within_session_differences.remainder(resolved_stride) != 0
    ):
        raise ValueError(
            "Within-session origin differences are not integer "
            f"multiples of the resolved stride {resolved_stride}."
        )

    valid_pair = (
        same_session
        & (origin_differences == resolved_stride)
    )

    if not torch.any(valid_pair):
        raise ValueError(
            "No forecast pairs were exactly one expected origin "
            "stride apart."
        )

    order_device = order_cpu.to(
        device=y_pred_raw.device,
    )

    ordered_predicted = y_pred_raw.index_select(
        0,
        order_device,
    ).to(dtype=torch.float64)
    ordered_realised = y_true_raw.index_select(
        0,
        order_device,
    ).to(dtype=torch.float64)

    predicted_returns_ordered = torch.full_like(
        ordered_predicted,
        torch.nan,
    )
    realised_returns_ordered = torch.full_like(
        ordered_realised,
        torch.nan,
    )

    current_positions_cpu = (
        torch.nonzero(
            valid_pair,
            as_tuple=False,
        ).flatten()
        + 1
    )
    previous_positions_cpu = current_positions_cpu - 1

    current_positions = current_positions_cpu.to(
        device=y_pred_raw.device,
    )
    previous_positions = previous_positions_cpu.to(
        device=y_pred_raw.device,
    )

    predicted_current = ordered_predicted[current_positions]
    predicted_previous = ordered_predicted[previous_positions]
    realised_current = ordered_realised[current_positions]
    realised_previous = ordered_realised[previous_positions]

    valid_price_pair = (
        torch.isfinite(predicted_current)
        & torch.isfinite(predicted_previous)
        & torch.isfinite(realised_current)
        & torch.isfinite(realised_previous)
        & (predicted_current > 0)
        & (predicted_previous > 0)
        & (realised_current > 0)
        & (realised_previous > 0)
    )

    safe_predicted_current = torch.where(
        valid_price_pair,
        predicted_current,
        torch.ones_like(predicted_current),
    )
    safe_predicted_previous = torch.where(
        valid_price_pair,
        predicted_previous,
        torch.ones_like(predicted_previous),
    )
    safe_realised_current = torch.where(
        valid_price_pair,
        realised_current,
        torch.ones_like(realised_current),
    )
    safe_realised_previous = torch.where(
        valid_price_pair,
        realised_previous,
        torch.ones_like(realised_previous),
    )

    predicted_return_values = (
        safe_predicted_current.log()
        - safe_predicted_previous.log()
    )
    realised_return_values = (
        safe_realised_current.log()
        - safe_realised_previous.log()
    )

    predicted_returns_ordered[current_positions] = torch.where(
        valid_price_pair,
        predicted_return_values,
        torch.full_like(predicted_return_values, torch.nan),
    )
    realised_returns_ordered[current_positions] = torch.where(
        valid_price_pair,
        realised_return_values,
        torch.full_like(realised_return_values, torch.nan),
    )

    predicted_returns = torch.empty_like(
        predicted_returns_ordered
    )
    realised_returns = torch.empty_like(
        realised_returns_ordered
    )

    predicted_returns[order_device] = predicted_returns_ordered
    realised_returns[order_device] = realised_returns_ordered

    return (
        predicted_returns,
        realised_returns,
        resolved_stride,
    )

def movement_magnitude_ratio(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compare predicted and realised movement magnitudes per asset.

    For each asset, horizon and channel:

        mean_t(abs(y_pred)) / mean_t(abs(y_true))

    The median valid asset-level ratio is then returned.

    Input:
        y_pred, y_true: [B, H, N, C]

    Output:
        [H, C]

    Values below 1 indicate under-sized predicted movements.
    Values above 1 indicate over-sized predicted movements.
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true to have shape "
            f"[B, H, N, C], got {tuple(y_pred.shape)}."
        )

    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "movement_magnitude_ratio currently supports "
            "reduce_dims=(0, 2) only."
        )

    # Mean across forecast windows while preserving assets:
    #
    # [B, H, N, C] -> [H, N, C]
    predicted_magnitude = (
        y_pred
        .abs()
        .mean(dim=0)
    )

    realised_magnitude = (
        y_true
        .abs()
        .mean(dim=0)
    )

    asset_ratios = torch.where(
        realised_magnitude > eps,
        predicted_magnitude / realised_magnitude,
        torch.full_like(
            predicted_magnitude,
            torch.nan,
        ),
    )

    # [H, N, C] -> [H, C]
    return torch.nanmedian(
        asset_ratios,
        dim=1,
    ).values


def temporal_absolute_correlation(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] = (0, 2),
) -> torch.Tensor:
    """
    Correlate predicted and realised movement magnitudes through time.

    For each asset, horizon and channel, Pearson correlation is
    calculated across forecast windows between abs(y_pred) and
    abs(y_true). Valid asset-level correlations are then averaged
    across assets.

    Input:
        y_pred, y_true: [B, H, N, C]

    Output:
        [H, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true to have shape [B, H, N, C], "
            f"got {tuple(y_pred.shape)}."
        )

    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "temporal_absolute_correlation currently supports "
            "reduce_dims=(0, 2) only."
        )

    if y_pred.shape[0] < 2:
        raise ValueError(
            "Temporal absolute correlation requires at least two "
            "forecast windows."
        )

    predicted_magnitude = y_pred.abs()
    realised_magnitude = y_true.abs()

    predicted_centred = (
        predicted_magnitude
        - predicted_magnitude.mean(
            dim=0,
            keepdim=True,
        )
    )

    realised_centred = (
        realised_magnitude
        - realised_magnitude.mean(
            dim=0,
            keepdim=True,
        )
    )

    covariance_sum = (
        predicted_centred
        * realised_centred
    ).sum(dim=0)

    predicted_sum_squared = (
        predicted_centred
        .square()
        .sum(dim=0)
    )

    realised_sum_squared = (
        realised_centred
        .square()
        .sum(dim=0)
    )

    denominator = torch.sqrt(
        predicted_sum_squared
        * realised_sum_squared
    )

    asset_correlations = torch.where(
        denominator > 0,
        covariance_sum / denominator,
        torch.full_like(
            covariance_sum,
            torch.nan,
        ),
    )

    # [H, N, C] -> [H, C]
    return torch.nanmean(
        asset_correlations,
        dim=1,
    )

MetricFunction = Callable[..., torch.Tensor]

def cross_sectional_pearson_ic_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute one cross-sectional Pearson IC per forecast window,
    horizon and channel.

    Correlation is calculated across the asset dimension.

    Input:
        y_pred, y_true: [B, H, N, C]

    Output:
        [B, H, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true to have shape [B, H, N, C], "
            f"got {tuple(y_pred.shape)}."
        )

    if y_pred.shape[2] < 2:
        raise ValueError(
            "Cross-sectional IC requires at least two assets."
        )

    pred_centred = (
        y_pred
        - y_pred.mean(
            dim=2,
            keepdim=True,
        )
    )

    true_centred = (
        y_true
        - y_true.mean(
            dim=2,
            keepdim=True,
        )
    )

    covariance_sum = (
        pred_centred
        * true_centred
    ).sum(dim=2)

    pred_sum_squared = (
        pred_centred
        .square()
        .sum(dim=2)
    )

    true_sum_squared = (
        true_centred
        .square()
        .sum(dim=2)
    )

    denominator = torch.sqrt(
        pred_sum_squared
        * true_sum_squared
    )

    return torch.where(
        denominator > eps,
        covariance_sum / denominator,
        torch.full_like(
            covariance_sum,
            torch.nan,
        ),
    )


def cross_sectional_pearson_ic(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] = (0, 2),
) -> torch.Tensor:
    """
    Compute mean cross-sectional Pearson IC.

    First computes correlation across assets separately for every
    forecast window, horizon and channel, then averages over forecast
    windows.

    Input:
        [B, H, N, C]

    Output:
        [H, C]
    """
    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "cross_sectional_pearson_ic currently supports "
            "reduce_dims=(0, 2) only."
        )

    ic_values = cross_sectional_pearson_ic_values(
        y_pred=y_pred,
        y_true=y_true,
    )

    return torch.nanmean(
        ic_values,
        dim=0,
    )


def _average_ranks_across_assets(
    values: torch.Tensor,
) -> torch.Tensor:
    """
    Assign average ranks across the asset dimension.

    Tied values receive their average rank.

    Input/output shape:
        [B, H, N, C]
    """
    if values.ndim != 4:
        raise ValueError(
            "Expected values with shape [B, H, N, C], "
            f"got {tuple(values.shape)}."
        )

    if not torch.isfinite(values).all():
        raise ValueError(
            "Cannot rank values containing NaN or infinite values."
        )

    original_device = values.device
    original_dtype = values.dtype

    # [B, H, N, C] -> [B, H, C, N]
    values_cpu = (
        values
        .permute(0, 1, 3, 2)
        .contiguous()
        .reshape(-1, values.shape[2])
        .detach()
        .cpu()
        .to(dtype=torch.float64)
    )

    ranks_cpu = torch.empty_like(values_cpu)

    for row_idx in range(values_cpu.shape[0]):
        sorted_values, sorted_indices = torch.sort(
            values_cpu[row_idx]
        )

        _, counts = torch.unique_consecutive(
            sorted_values,
            return_counts=True,
        )

        group_ends = counts.cumsum(dim=0)
        group_starts = group_ends - counts

        # Ranks are one-indexed. For a tied group occupying sorted
        # positions [start, end), its average rank is:
        # ((start + 1) + end) / 2
        average_ranks = (
            group_starts.to(dtype=torch.float64)
            + group_ends.to(dtype=torch.float64)
            + 1.0
        ) / 2.0

        sorted_ranks = torch.repeat_interleave(
            average_ranks,
            counts,
        )

        ranks_cpu[row_idx].scatter_(
            dim=0,
            index=sorted_indices,
            src=sorted_ranks,
        )

    ranks = (
        ranks_cpu
        .reshape(
            values.shape[0],
            values.shape[1],
            values.shape[3],
            values.shape[2],
        )
        .permute(0, 1, 3, 2)
        .contiguous()
    )

    return ranks.to(
        device=original_device,
        dtype=original_dtype,
    )


def cross_sectional_spearman_rank_ic_values(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute one cross-sectional Spearman Rank IC per forecast window,
    horizon and channel.

    Correlation is calculated across the asset dimension.

    Input:
        y_pred, y_true: [B, H, N, C]

    Output:
        [B, H, C]
    """
    validate_prediction_shapes(
        y_pred=y_pred,
        y_true=y_true,
    )

    if y_pred.ndim != 4:
        raise ValueError(
            "Expected y_pred and y_true with shape [B, H, N, C], "
            f"got {tuple(y_pred.shape)}."
        )

    if y_pred.shape[2] < 2:
        raise ValueError(
            "Cross-sectional Rank IC requires at least two assets."
        )

    pred_ranks = _average_ranks_across_assets(
        y_pred
    )

    true_ranks = _average_ranks_across_assets(
        y_true
    )

    return cross_sectional_pearson_ic_values(
        y_pred=pred_ranks,
        y_true=true_ranks,
        eps=eps,
    )


def cross_sectional_spearman_rank_ic(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] = (0, 2),
) -> torch.Tensor:
    """
    Compute mean cross-sectional Spearman Rank IC.

    Rank IC is calculated across assets separately for every forecast
    window, horizon and channel, then averaged over forecast windows.

    Input:
        [B, H, N, C]

    Output:
        [H, C]
    """
    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "cross_sectional_spearman_rank_ic currently supports "
            "reduce_dims=(0, 2) only."
        )

    rank_ic_values = (
        cross_sectional_spearman_rank_ic_values(
            y_pred=y_pred,
            y_true=y_true,
        )
    )

    return torch.nanmean(
        rank_ic_values,
        dim=0,
    )

@dataclass(frozen=True)
class BootstrapMetricComponents:
    """
    Pointwise information required to bootstrap one metric.

    All component tensors initially have shape:

        [B, H, N, C]

    Kinds:
        mean:
            values contains pointwise contributions that are averaged.

        ratio:
            values contains numerator contributions.
            reference_values contains denominator contributions.

        correlation:
            values contains x.
            reference_values contains y.

        assetwise_correlation:
            values contains x and reference_values contains y. Pearson
            correlation is reconstructed across forecast windows
            separately for each asset, then averaged across assets.
        
        window_mean:
            values contains one metric value per forecast window,
            horizon and channel, with shape [B, H, C].

        assetwise_ratio:
            values contains numerator contributions and
            reference_values contains denominator contributions.
            A ratio is calculated separately for each asset, followed
            by the median across valid assets.
    """

    kind: Literal[
        "mean",
        "ratio",
        "assetwise_ratio",
        "correlation",
        "assetwise_correlation",
        "window_mean",
    ]

    values: torch.Tensor

    reference_values: torch.Tensor | None = None


BootstrapComponentFunction = Callable[
    ...,
    BootstrapMetricComponents,
]

@dataclass(frozen=True)
class BootstrapSessionStatistics:
    """
    Metric contributions aggregated into complete trading-session
    blocks.

    Shapes:
        session_ids:              [D]
        observation_count:        [D] for pooled metrics, or
                                  [D, H, N, C] for asset-wise
                                  correlations with pairwise masks
        value_sum:                 [D, H, C] for pooled metrics, or
                                  [D, H, N, C] for asset-wise metrics

    Additional tensors are populated according to metric kind:

        mean:
            value_sum

        ratio:
            value_sum
            reference_sum

        correlation:
            value_sum
            reference_sum
            value_squared_sum
            reference_squared_sum
            cross_sum

        assetwise_correlation:
            The same sufficient statistics are retained with shape
            [D, H, N, C] so correlations can be reconstructed for
            each asset before averaging across assets.  Observation
            counts use the same shape so undefined session-opening
            forecast-series returns are excluded exactly.

        assetwise_ratio:
            value_sum and reference_sum retain shape [D, H, N, C]
            so ratios can be calculated separately for each asset.
    """

    kind: Literal[
        "mean",
        "ratio",
        "assetwise_ratio",
        "correlation",
        "assetwise_correlation",
        "window_mean",
    ]

    session_ids: torch.Tensor

    observation_count: torch.Tensor

    value_sum: torch.Tensor

    reference_sum: torch.Tensor | None = None

    value_squared_sum: torch.Tensor | None = None

    reference_squared_sum: torch.Tensor | None = None

    cross_sum: torch.Tensor | None = None

@dataclass(frozen=True)
class MetricDefinition:
    """
    Define both ordinary and bootstrap evaluation for one metric.
    """

    compute: MetricFunction

    bootstrap_components: BootstrapComponentFunction | None = None

    @property
    def supports_bootstrap(self) -> bool:
        """Whether this metric has a session-block bootstrap definition."""
        return self.bootstrap_components is not None


class ForecastEvaluator:
    """
    Evaluate forecasts returned in raw value space.

    The prediction result must contain raw predictions, raw ground truth,
    and the final target observation from each context window.
    """

    def __init__(
        self,
        prediction_result: dict[str, Any],
        train_split: dict[str, Any] | None = None,
    ) -> None:
        self._validate_prediction_result(prediction_result)

        self.y_pred_raw = prediction_result["y_pred"]
        self.y_true_raw = prediction_result["y_true"]
        self.last_context_target = prediction_result["last_context_target"]

        self.channels = list(prediction_result["channels"])
        self.horizons = list(prediction_result["horizons"])
        self.sample_idx = prediction_result.get("sample_idx")
        self.origin_idx = prediction_result.get("origin_idx")

        self.mase_scale: torch.Tensor | None = None

        if train_split is not None:
            prediction_asset_cols = prediction_result.get(
                "asset_cols"
            )

            if prediction_asset_cols is not None:
                if list(prediction_asset_cols) != list(
                    train_split["asset_cols"]
                ):
                    raise ValueError(
                        "Prediction assets and training assets are not "
                        "in the same order."
                    )

            mase_scale = compute_mase_scale(
                train_split=train_split,
                channels=self.channels,
            )

            expected_scale_shape = (
                self.y_pred_raw.shape[2],
                self.y_pred_raw.shape[3],
            )

            if mase_scale.shape != expected_scale_shape:
                raise ValueError(
                    "The training-derived MASE scale is not aligned "
                    "with the prediction tensors. "
                    f"Expected {expected_scale_shape}, "
                    f"got {tuple(mase_scale.shape)}."
                )

            self.mase_scale = mase_scale

        self._metric_registry = self._build_metric_registry()

    @property
    def persistence_pred_raw(self) -> torch.Tensor:
        """
        Construct persistence predictions at every forecast horizon.

        Returns:
            Tensor with shape [B, H, N, C].
        """
        return (
            self.last_context_target
            .unsqueeze(1)
            .expand_as(self.y_true_raw)
        )

    def get_predictions(
        self,
        output_space: str = "raw",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return predictions and targets in the requested evaluation space.
        """
        if output_space == "raw":
            return self.y_pred_raw, self.y_true_raw

        if output_space == "cumulative_log_change":
            y_pred = raw_to_cumulative_log_change(
                y_raw=self.y_pred_raw,
                last_context_target=self.last_context_target,
            )

            y_true = raw_to_cumulative_log_change(
                y_raw=self.y_true_raw,
                last_context_target=self.last_context_target,
            )

            return y_pred, y_true

        raise ValueError(
            "output_space must be either 'raw' or "
            f"'cumulative_log_change', got {output_space!r}."
        )

    def compute_pairwise_metric(
        self,
        metric_fn: MetricFunction,
        output_space: str = "raw",
        reduce_dims: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """
        Apply a metric that takes y_pred, y_true and reduce_dims.

        This currently supports metrics such as MAE, MSE and RMSE.
        """
        y_pred, y_true = self.get_predictions(
            output_space=output_space,
        )

        return metric_fn(
            y_pred=y_pred,
            y_true=y_true,
            reduce_dims=reduce_dims,
        )
    
    def compute_relative_mae_vs_persistence(
        self,
        reduce_dims: Sequence[int] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute raw-space MAE relative to persistence.

        Values below 1 indicate that the model beats persistence.
        """
        return relative_mae_vs_persistence(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            persistence_pred=self.persistence_pred_raw,
            reduce_dims=reduce_dims,
            eps=eps,
        )
    
    def compute_persistence_win_rate(
        self,
        reduce_dims: Sequence[int] | None = None,
        tie_value: float = 0.5,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the pointwise win rate against persistence.

        The returned values are proportions between 0 and 1.
        """
        return persistence_win_rate(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            persistence_pred=self.persistence_pred_raw,
            reduce_dims=reduce_dims,
            tie_value=tie_value,
            rtol=rtol,
            atol=atol,
        )
    
    def compute_mase(
        self,
        reduce_dims: Sequence[int] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute standard raw-space MASE using the training-derived scale.
        """
        if self.mase_scale is None:
            raise ValueError(
                "MASE requires a training split. Construct the evaluator "
                "with ForecastEvaluator(prediction_result, "
                "train_split=train_split)."
            )

        return mase(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            mase_scale=self.mase_scale,
            reduce_dims=reduce_dims,
            eps=eps,
        )
    
    def compute_cumulative_log_change_movement_magnitude_ratio(
        self,
        reduce_dims: Sequence[int] | None = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute the ratio of mean predicted to realised absolute
        cumulative log change.
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        return movement_magnitude_ratio(
            y_pred=y_pred,
            y_true=y_true,
            reduce_dims=reduce_dims,
            eps=eps,
        )

    def compute_cumulative_log_change_temporal_absolute_correlation(
        self,
        reduce_dims: Sequence[int] = (0, 2),
    ) -> torch.Tensor:
        """
        Compute asset-wise temporal Pearson correlation between
        predicted and realised absolute cumulative log changes.
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        return temporal_absolute_correlation(
            y_pred=y_pred,
            y_true=y_true,
            reduce_dims=reduce_dims,
        )

    def compute_raw_price_temporal_pearson_correlation(
        self,
        reduce_dims: Sequence[int] = (0, 2),
    ) -> torch.Tensor:
        """Correlate predicted and realised raw prices through time.

        Pearson correlation is calculated separately for each asset,
        horizon and channel across forecast windows, then averaged over
        valid assets.
        """
        return assetwise_temporal_pearson_correlation(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            reduce_dims=reduce_dims,
        )

    def compute_forecast_series_log_return_temporal_pearson_correlation(
        self,
        reduce_dims: Sequence[int] = (0, 2),
        expected_origin_stride: int | None = None,
    ) -> torch.Tensor:
        """Correlate log returns formed from horizon-aligned series.

        Returns are formed from adjacent saved forecasts within the same
        trading session.  They are not cumulative returns from the
        context origin.
        """
        if self.sample_idx is None or self.origin_idx is None:
            raise ValueError(
                "Forecast-series log-return correlation requires both "
                "sample_idx and origin_idx in prediction_result."
            )

        y_pred, y_true, _ = forecast_series_log_return_values(
            y_pred_raw=self.y_pred_raw,
            y_true_raw=self.y_true_raw,
            sample_idx=self.sample_idx,
            origin_idx=self.origin_idx,
            expected_origin_stride=expected_origin_stride,
        )

        return assetwise_temporal_pearson_correlation(
            y_pred=y_pred,
            y_true=y_true,
            reduce_dims=reduce_dims,
        )

    def _build_cumulative_log_change_mae_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """
        Return pointwise cumulative-log-change absolute errors.
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        values = absolute_error_values(
            y_pred=y_pred,
            y_true=y_true,
        )

        return BootstrapMetricComponents(
            kind="mean",
            values=values,
        )
    
    def _build_cumulative_log_change_pearson_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """
        Return cumulative-log-change predictions and targets for
        Pearson sufficient-statistic aggregation.
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        return BootstrapMetricComponents(
            kind="correlation",
            values=y_pred,
            reference_values=y_true,
        )

    def _build_cumulative_log_change_directional_accuracy_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """Return pointwise cumulative-log-change sign scores."""
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        values = directional_accuracy_values(
            y_pred=y_pred,
            y_true=y_true,
        )

        return BootstrapMetricComponents(
            kind="mean",
            values=values,
        )
    
    def _build_cumulative_log_change_movement_magnitude_ratio_bootstrap_components(
        self,
        eps: float = 1e-8,
    ) -> BootstrapMetricComponents:
        """
        Return predicted and realised absolute cumulative log changes
        for ratio aggregation.

        The eps threshold is applied after each bootstrap sample has
        been aggregated.
        """
        del eps

        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        return BootstrapMetricComponents(
            kind="assetwise_ratio",
            values=y_pred.abs(),
            reference_values=y_true.abs(),
        )

    def _build_cumulative_log_change_temporal_absolute_correlation_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """
        Return absolute cumulative log changes for asset-wise temporal
        Pearson sufficient-statistic aggregation.
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        return BootstrapMetricComponents(
            kind="assetwise_correlation",
            values=y_pred.abs(),
            reference_values=y_true.abs(),
        )

    def _build_raw_price_temporal_pearson_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """Return raw prices for asset-wise temporal correlation."""
        return BootstrapMetricComponents(
            kind="assetwise_correlation",
            values=self.y_pred_raw,
            reference_values=self.y_true_raw,
        )

    def _build_forecast_series_log_return_temporal_pearson_bootstrap_components(
        self,
        expected_origin_stride: int | None = None,
    ) -> BootstrapMetricComponents:
        """Return within-session horizon-series log returns."""
        if self.sample_idx is None or self.origin_idx is None:
            raise ValueError(
                "Forecast-series log-return correlation requires both "
                "sample_idx and origin_idx in prediction_result."
            )

        y_pred, y_true, _ = forecast_series_log_return_values(
            y_pred_raw=self.y_pred_raw,
            y_true_raw=self.y_true_raw,
            sample_idx=self.sample_idx,
            origin_idx=self.origin_idx,
            expected_origin_stride=expected_origin_stride,
        )

        return BootstrapMetricComponents(
            kind="assetwise_correlation",
            values=y_pred,
            reference_values=y_true,
        )

    def _build_cumulative_log_change_cross_sectional_pearson_ic_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """
        Return one cross-sectional Pearson IC per forecast window,
        horizon and channel.

        Values have shape [B, H, C].
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        values = cross_sectional_pearson_ic_values(
            y_pred=y_pred,
            y_true=y_true,
        )

        return BootstrapMetricComponents(
            kind="window_mean",
            values=values,
        )
    
    def _build_mase_bootstrap_components(
        self,
        eps: float = 1e-8,
    ) -> BootstrapMetricComponents:
        """
        Return pointwise MASE contributions.

        The training-derived MASE scale remains fixed.
        """
        if self.mase_scale is None:
            raise ValueError(
                "MASE requires a training split. Construct the evaluator "
                "with ForecastEvaluator(prediction_result, "
                "train_split=train_split)."
            )

        values = mase_values(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            mase_scale=self.mase_scale,
            eps=eps,
        )

        return BootstrapMetricComponents(
            kind="mean",
            values=values,
        )
    
    def _build_relative_mae_bootstrap_components(
        self,
        eps: float = 1e-8,
    ) -> BootstrapMetricComponents:
        """
        Return the numerator and denominator contributions required
        for relative MAE.

        The eps threshold will be applied after aggregating each
        bootstrap sample.
        """
        del eps

        model_absolute_error = absolute_error_values(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
        )

        persistence_absolute_error = absolute_error_values(
            y_pred=self.persistence_pred_raw,
            y_true=self.y_true_raw,
        )

        return BootstrapMetricComponents(
            kind="ratio",
            values=model_absolute_error,
            reference_values=persistence_absolute_error,
        )
    
    def _build_persistence_win_rate_bootstrap_components(
        self,
        tie_value: float = 0.5,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> BootstrapMetricComponents:
        """
        Return pointwise Persistence win scores.
        """
        values = persistence_win_score_values(
            y_pred=self.y_pred_raw,
            y_true=self.y_true_raw,
            persistence_pred=self.persistence_pred_raw,
            tie_value=tie_value,
            rtol=rtol,
            atol=atol,
        )

        return BootstrapMetricComponents(
            kind="mean",
            values=values,
        )
    
    def _build_cumulative_log_change_cross_sectional_spearman_rank_ic_bootstrap_components(
        self,
    ) -> BootstrapMetricComponents:
        """
        Return one cross-sectional Spearman Rank IC per forecast window,
        horizon and channel.

        Values have shape [B, H, C].
        """
        y_pred, y_true = self.get_predictions(
            output_space="cumulative_log_change",
        )

        values = (
            cross_sectional_spearman_rank_ic_values(
                y_pred=y_pred,
                y_true=y_true,
            )
        )

        return BootstrapMetricComponents(
            kind="window_mean",
            values=values,
        )
    
    def _build_metric_registry(
        self,
    ) -> dict[str, MetricDefinition]:
        """
        Map each public metric name to its ordinary computation and
        bootstrap-component builder.
        """
        registry = {
            "cumulative_log_change_mae": MetricDefinition(
                compute=partial(
                    self.compute_pairwise_metric,
                    metric_fn=mae,
                    output_space="cumulative_log_change",
                ),
                bootstrap_components=(
                    self
                    ._build_cumulative_log_change_mae_bootstrap_components
                ),
            ),

            "cumulative_log_change_median_absolute_error": (
                MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=median_absolute_error,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=None,
                )
            ),

            "cumulative_log_change_p95_absolute_error": (
                MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=p95_absolute_error,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=None,
                )
            ),

            "cumulative_log_change_pearson_correlation": (
                MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=pearson_correlation,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_pearson_bootstrap_components
                    ),
                )
            ),

            "raw_price_temporal_pearson_correlation": (
                MetricDefinition(
                    compute=(
                        self
                        .compute_raw_price_temporal_pearson_correlation
                    ),
                    bootstrap_components=(
                        self
                        ._build_raw_price_temporal_pearson_bootstrap_components
                    ),
                )
            ),

            "cumulative_log_change_directional_accuracy": (
                MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=directional_accuracy,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_directional_accuracy_bootstrap_components
                    ),
                )
            ),

            "cumulative_log_change_movement_magnitude_ratio": (
                MetricDefinition(
                    compute=(
                        self
                        .compute_cumulative_log_change_movement_magnitude_ratio
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_movement_magnitude_ratio_bootstrap_components
                    ),
                )
            ),

            "cumulative_log_change_temporal_absolute_correlation": (
                MetricDefinition(
                    compute=(
                        self
                        .compute_cumulative_log_change_temporal_absolute_correlation
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_temporal_absolute_correlation_bootstrap_components
                    ),
                )
            ),

            "mase": MetricDefinition(
                compute=self.compute_mase,
                bootstrap_components=(
                    self._build_mase_bootstrap_components
                ),
            ),

            "relative_mae_vs_persistence": MetricDefinition(
                compute=self.compute_relative_mae_vs_persistence,
                bootstrap_components=(
                    self._build_relative_mae_bootstrap_components
                ),
            ),

            "persistence_win_rate": MetricDefinition(
                compute=self.compute_persistence_win_rate,
                bootstrap_components=(
                    self._build_persistence_win_rate_bootstrap_components
                ),
            ),

            "cumulative_log_change_cross_sectional_pearson_ic": (MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=cross_sectional_pearson_ic,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_cross_sectional_pearson_ic_bootstrap_components
                    ),
                )
            ),

            "cumulative_log_change_cross_sectional_spearman_rank_ic": (
                MetricDefinition(
                    compute=partial(
                        self.compute_pairwise_metric,
                        metric_fn=cross_sectional_spearman_rank_ic,
                        output_space="cumulative_log_change",
                    ),
                    bootstrap_components=(
                        self
                        ._build_cumulative_log_change_cross_sectional_spearman_rank_ic_bootstrap_components
                    ),
                )
            ),
        }

        if self.sample_idx is not None and self.origin_idx is not None:
            registry[
                "forecast_series_log_return_temporal_pearson_correlation"
            ] = MetricDefinition(
                compute=(
                    self
                    .compute_forecast_series_log_return_temporal_pearson_correlation
                ),
                bootstrap_components=(
                    self
                    ._build_forecast_series_log_return_temporal_pearson_bootstrap_components
                ),
            )

        return registry
    
    @property
    def available_metrics(self) -> tuple[str, ...]:
        """
        Return the names of all metrics currently available through
        evaluate().
        """
        return tuple(self._metric_registry)
    

    def _summarise_bootstrap_samples(
        self,
        bootstrap_samples: torch.Tensor,
        confidence_level: float,
    ) -> dict[str, torch.Tensor]:
        """
        Summarise a bootstrap distribution.

        Args:
            bootstrap_samples:
                Bootstrap metric values with shape [R, H, C].

            confidence_level:
                Confidence level for the percentile interval.

        Returns:
            Dictionary containing tensors with shape [H, C]:
                bootstrap_mean
                bootstrap_std
                ci_lower
                ci_upper
        """
        if not isinstance(
            bootstrap_samples,
            torch.Tensor,
        ):
            raise TypeError(
                "bootstrap_samples must be a torch.Tensor."
            )

        if bootstrap_samples.ndim != 3:
            raise ValueError(
                "Expected bootstrap_samples to have shape "
                "[R, H, C], got "
                f"{tuple(bootstrap_samples.shape)}."
            )

        if bootstrap_samples.shape[0] < 2:
            raise ValueError(
                "At least two bootstrap samples are required."
            )

        if not isinstance(
            confidence_level,
            (float, int),
        ):
            raise TypeError(
                "confidence_level must be numeric."
            )

        confidence_level = float(
            confidence_level
        )

        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                "confidence_level must lie strictly between "
                f"0 and 1, got {confidence_level}."
            )

        samples = (
            bootstrap_samples
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        finite_mask = torch.isfinite(
            samples
        )

        finite_count = finite_mask.sum(
            dim=0
        )

        bootstrap_mean = torch.nanmean(
            samples,
            dim=0,
        )

        centred_samples = torch.where(
            finite_mask,
            samples - bootstrap_mean.unsqueeze(0),
            torch.zeros_like(samples),
        )

        squared_deviation_sum = (
            centred_samples
            .square()
            .sum(dim=0)
        )

        standard_deviation_denominator = (
            finite_count - 1
        ).clamp_min(1)

        bootstrap_std = torch.sqrt(
            squared_deviation_sum
            / standard_deviation_denominator
        )

        bootstrap_std = torch.where(
            finite_count > 1,
            bootstrap_std,
            torch.full_like(
                bootstrap_std,
                torch.nan,
            ),
        )

        alpha = (
            1.0 - confidence_level
        )

        ci_lower = torch.nanquantile(
            samples,
            q=alpha / 2.0,
            dim=0,
        )

        ci_upper = torch.nanquantile(
            samples,
            q=1.0 - alpha / 2.0,
            dim=0,
        )

        return {
            "bootstrap_mean": bootstrap_mean,
            "bootstrap_std": bootstrap_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    
    def evaluate(
        self,
        metrics: str | Sequence[str],
        reduce_dims: Sequence[int] | None = None,
        metric_kwargs: dict[str, dict[str, Any]] | None = None,
        *,
        bootstrap: bool = False,
        n_bootstrap: int = 10_000,
        confidence_level: float = 0.95,
        bootstrap_seed: int = 42,
    ) -> (
        dict[str, torch.Tensor]
        | dict[str, dict[str, torch.Tensor]]
    ):
        """
        Compute one or more registered forecast metrics.

        Args:
            metrics:
                One metric name or a sequence of metric names.

            reduce_dims:
                Tensor dimensions over which every requested metric is
                reduced.

                For prediction tensors with shape [B, H, N, C],
                reduce_dims=(0, 2) retains horizon and channel and
                returns tensors with shape [H, C].

            metric_kwargs:
                Optional metric-specific keyword arguments.

            bootstrap:
                If False, return ordinary metric tensors.

                If True, compute trading-session block-bootstrap
                confidence intervals.

            n_bootstrap:
                Number of bootstrap replicates.

            confidence_level:
                Confidence level for percentile intervals.

            bootstrap_seed:
                Random seed for session resampling.

        Returns:
            When bootstrap=False:

                {
                    metric_name: Tensor[H, C]
                }

            When bootstrap=True:

                {
                    metric_name: {
                        "value": Tensor[H, C],
                        "bootstrap_mean": Tensor[H, C],
                        "bootstrap_std": Tensor[H, C],
                        "ci_lower": Tensor[H, C],
                        "ci_upper": Tensor[H, C],
                    }
                }

            Metrics without a bootstrap definition still return their
            ordinary ``value``.  Their four bootstrap-summary tensors are
            filled with NaN so tables can display an em dash without
            pretending that uncertainty was estimated.
        """
        if isinstance(
            metrics,
            str,
        ):
            metric_names = [
                metrics
            ]
        else:
            metric_names = list(
                metrics
            )

        if len(metric_names) == 0:
            raise ValueError(
                "At least one metric must be requested."
            )

        duplicate_metrics = {
            metric_name
            for metric_name in metric_names
            if metric_names.count(metric_name) > 1
        }

        if duplicate_metrics:
            raise ValueError(
                "Each metric should be requested only once. "
                f"Duplicates: {sorted(duplicate_metrics)}."
            )

        unknown_metrics = [
            metric_name
            for metric_name in metric_names
            if metric_name not in self._metric_registry
        ]

        if unknown_metrics:
            raise ValueError(
                f"Unknown metrics: {unknown_metrics}. "
                f"Available metrics: "
                f"{list(self.available_metrics)}."
            )

        if metric_kwargs is None:
            metric_kwargs = {}

        unknown_kwargs_metrics = set(
            metric_kwargs
        ).difference(
            metric_names
        )

        if unknown_kwargs_metrics:
            raise ValueError(
                "metric_kwargs contains entries for metrics "
                "that were not requested: "
                f"{sorted(unknown_kwargs_metrics)}."
            )

        ordinary_results: dict[
            str,
            torch.Tensor,
        ] = {}

        resolved_metric_kwargs: dict[
            str,
            dict[str, Any],
        ] = {}

        for metric_name in metric_names:
            kwargs = dict(
                metric_kwargs.get(
                    metric_name,
                    {},
                )
            )

            if "reduce_dims" in kwargs:
                raise ValueError(
                    "Pass reduce_dims through evaluate(), not "
                    "through "
                    f"metric_kwargs[{metric_name!r}]."
                )

            metric_definition = (
                self._metric_registry[
                    metric_name
                ]
            )

            ordinary_results[
                metric_name
            ] = metric_definition.compute(
                reduce_dims=reduce_dims,
                **kwargs,
            )

            resolved_metric_kwargs[
                metric_name
            ] = kwargs

        if not isinstance(
            bootstrap,
            bool,
        ):
            raise TypeError(
                "bootstrap must be a boolean."
            )

        if not bootstrap:
            return ordinary_results

        if reduce_dims is None:
            raise ValueError(
                "Bootstrap evaluation currently requires "
                "reduce_dims=(0, 2)."
            )

        if tuple(reduce_dims) != (0, 2):
            raise ValueError(
                "Bootstrap evaluation currently supports only "
                "reduce_dims=(0, 2), got "
                f"{tuple(reduce_dims)}."
            )

        if not isinstance(
            n_bootstrap,
            int,
        ):
            raise TypeError(
                "n_bootstrap must be an integer."
            )

        if n_bootstrap < 2:
            raise ValueError(
                "n_bootstrap must be at least 2."
            )

        if not isinstance(
            bootstrap_seed,
            int,
        ):
            raise TypeError(
                "bootstrap_seed must be an integer."
            )

        if not isinstance(
            confidence_level,
            (float, int),
        ):
            raise TypeError(
                "confidence_level must be numeric."
            )

        confidence_level = float(
            confidence_level
        )

        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                "confidence_level must lie strictly between "
                f"0 and 1, got {confidence_level}."
            )

        bootstrap_metric_names = [
            metric_name
            for metric_name in metric_names
            if self._metric_registry[metric_name].supports_bootstrap
        ]

        bootstrap_session_counts: torch.Tensor | None = None

        if bootstrap_metric_names:
            session_ids, _ = (
                self._get_bootstrap_session_mapping()
            )

            num_sessions = int(
                session_ids.numel()
            )

            bootstrap_session_counts = (
                self._generate_bootstrap_session_counts(
                    num_sessions=num_sessions,
                    n_bootstrap=n_bootstrap,
                    bootstrap_seed=bootstrap_seed,
                )
            )

        bootstrap_results: dict[
            str,
            dict[str, torch.Tensor],
        ] = {}

        for metric_name in metric_names:
            metric_definition = (
                self._metric_registry[
                    metric_name
                ]
            )

            ordinary_value = (
                ordinary_results[
                    metric_name
                ]
                .detach()
                .cpu()
                .clone()
            )

            if not metric_definition.supports_bootstrap:
                unavailable = torch.full_like(
                    ordinary_value,
                    torch.nan,
                )

                bootstrap_results[
                    metric_name
                ] = {
                    "value": ordinary_value,
                    "bootstrap_mean": unavailable.clone(),
                    "bootstrap_std": unavailable.clone(),
                    "ci_lower": unavailable.clone(),
                    "ci_upper": unavailable.clone(),
                }

                continue

            if bootstrap_session_counts is None:
                raise AssertionError(
                    "Bootstrap session counts were not generated for "
                    f"bootstrap-supported metric {metric_name!r}."
                )

            kwargs = resolved_metric_kwargs[
                metric_name
            ]

            component_builder = (
                metric_definition.bootstrap_components
            )

            if component_builder is None:
                raise AssertionError(
                    "Metric reports bootstrap support without a "
                    f"component builder: {metric_name!r}."
                )

            components = component_builder(
                **kwargs,
            )

            statistics = (
                self
                ._aggregate_bootstrap_components_by_session(
                    components
                )
            )

            if statistics.kind in {
                "mean",
                "ratio",
                "window_mean",
            }:
                ratio_eps = float(
                    kwargs.get(
                        "eps",
                        1e-8,
                    )
                )

                bootstrap_samples = (
                    self
                    ._compute_mean_or_ratio_bootstrap_samples(
                        statistics=statistics,
                        bootstrap_session_counts=(
                            bootstrap_session_counts
                        ),
                        eps=ratio_eps,
                    )
                )

            elif statistics.kind == "assetwise_ratio":
                ratio_eps = float(
                    kwargs.get(
                        "eps",
                        1e-8,
                    )
                )

                bootstrap_samples = (
                    self
                    ._compute_assetwise_ratio_bootstrap_samples(
                        statistics=statistics,
                        bootstrap_session_counts=(
                            bootstrap_session_counts
                        ),
                        eps=ratio_eps,
                    )
                )

            elif statistics.kind == "correlation":
                bootstrap_samples = (
                    self
                    ._compute_correlation_bootstrap_samples(
                        statistics=statistics,
                        bootstrap_session_counts=(
                            bootstrap_session_counts
                        ),
                    )
                )

            elif statistics.kind == "assetwise_correlation":
                bootstrap_samples = (
                    self
                    ._compute_assetwise_correlation_bootstrap_samples(
                        statistics=statistics,
                        bootstrap_session_counts=(
                            bootstrap_session_counts
                        ),
                    )
                )

            else:
                raise ValueError(
                    "Unknown bootstrap statistics kind: "
                    f"{statistics.kind!r}."
                )

            expected_metric_shape = (
                ordinary_value.shape
            )

            if bootstrap_samples.shape[1:] != (
                expected_metric_shape
            ):
                raise RuntimeError(
                    "Bootstrap metric shape does not match the "
                    "ordinary metric result for "
                    f"{metric_name!r}. "
                    f"Expected [R, "
                    f"{expected_metric_shape[0]}, "
                    f"{expected_metric_shape[1]}], got "
                    f"{tuple(bootstrap_samples.shape)}."
                )

            bootstrap_summary = (
                self._summarise_bootstrap_samples(
                    bootstrap_samples=bootstrap_samples,
                    confidence_level=confidence_level,
                )
            )

            bootstrap_results[
                metric_name
            ] = {
                "value": ordinary_value,
                **bootstrap_summary,
            }

        return bootstrap_results

    @staticmethod
    def _validate_prediction_result(
        prediction_result: dict[str, Any],
    ) -> None:
        """
        Validate the common model prediction output.
        """
        required_keys = {
            "y_pred",
            "y_true",
            "last_context_target",
            "channels",
            "horizons",
        }

        missing_keys = required_keys.difference(
            prediction_result
        )

        if missing_keys:
            raise KeyError(
                "prediction_result is missing required keys: "
                f"{sorted(missing_keys)}."
            )

        output_space = prediction_result.get("output_space")

        if output_space is not None and output_space != "raw":
            raise ValueError(
                "ForecastEvaluator requires raw model outputs. "
                f"Got output_space={output_space!r}."
            )

        y_pred = prediction_result["y_pred"]
        y_true = prediction_result["y_true"]
        last_context_target = prediction_result[
            "last_context_target"
        ]

        if not isinstance(y_pred, torch.Tensor):
            raise TypeError("y_pred must be a torch.Tensor.")

        if not isinstance(y_true, torch.Tensor):
            raise TypeError("y_true must be a torch.Tensor.")

        if not isinstance(last_context_target, torch.Tensor):
            raise TypeError(
                "last_context_target must be a torch.Tensor."
            )

        if y_pred.ndim != 4:
            raise ValueError(
                "Expected y_pred to have shape [B, H, N, C], "
                f"got {tuple(y_pred.shape)}."
            )

        if y_pred.shape != y_true.shape:
            raise ValueError(
                "y_pred and y_true must have the same shape. "
                f"Got {tuple(y_pred.shape)} and "
                f"{tuple(y_true.shape)}."
            )

        batch_size, num_horizons, num_assets, num_channels = (
            y_pred.shape
        )

        expected_context_shape = (
            batch_size,
            num_assets,
            num_channels,
        )

        if last_context_target.shape != expected_context_shape:
            raise ValueError(
                "last_context_target has an incompatible shape. "
                f"Expected {expected_context_shape}, "
                f"got {tuple(last_context_target.shape)}."
            )

        if len(prediction_result["horizons"]) != num_horizons:
            raise ValueError(
                "The number of horizon labels must match the "
                "prediction horizon dimension."
            )

        if len(prediction_result["channels"]) != num_channels:
            raise ValueError(
                "The number of channel labels must match the "
                "prediction channel dimension."
            )

        for name, tensor in {
            "y_pred": y_pred,
            "y_true": y_true,
            "last_context_target": last_context_target,
        }.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} contains NaN or infinite values."
                )
    
    def _get_bootstrap_session_mapping(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate the bootstrap session identifiers and map each
        forecast window to a compact session position.

        Returns:
            unique_sessions:
                Original session identifiers with shape [D].

            session_inverse:
                For every forecast window, the corresponding compact
                session position with shape [B].

                Values lie in [0, D - 1].

        Here:
            B = number of forecast windows
            D = number of unique trading sessions
        """
        if self.sample_idx is None:
            raise ValueError(
                "Bootstrap evaluation requires prediction_result "
                "to contain 'sample_idx'."
            )

        if not isinstance(
            self.sample_idx,
            torch.Tensor,
        ):
            raise TypeError(
                "prediction_result['sample_idx'] must be "
                "a torch.Tensor."
            )

        if self.sample_idx.ndim != 1:
            raise ValueError(
                "Expected sample_idx to have shape [B], "
                f"got {tuple(self.sample_idx.shape)}."
            )

        expected_num_examples = (
            self.y_pred_raw.shape[0]
        )

        if self.sample_idx.shape[0] != expected_num_examples:
            raise ValueError(
                "sample_idx is not aligned with the prediction "
                "example dimension. "
                f"Expected {expected_num_examples} entries, "
                f"got {self.sample_idx.shape[0]}."
            )

        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }

        if self.sample_idx.dtype not in integer_dtypes:
            raise TypeError(
                "sample_idx must use an integer dtype, "
                f"got {self.sample_idx.dtype}."
            )

        sample_idx = (
            self.sample_idx
            .detach()
            .cpu()
            .to(dtype=torch.long)
        )

        unique_sessions, session_inverse = torch.unique(
            sample_idx,
            sorted=True,
            return_inverse=True,
        )

        if unique_sessions.numel() < 2:
            raise ValueError(
                "Bootstrap evaluation requires at least two "
                "unique trading sessions."
            )

        return (
            unique_sessions,
            session_inverse,
        )
    
    def _sum_bootstrap_values_by_session(
        self,
        values: torch.Tensor,
        session_inverse: torch.Tensor,
        num_sessions: int,
    ) -> torch.Tensor:
        """
        Sum pointwise metric contributions by trading session.

        Args:
            values:
                Pointwise values with shape [B, H, N, C].

            session_inverse:
                Compact session position for each forecast window,
                with shape [B].

            num_sessions:
                Number of unique trading sessions D.

        Returns:
            Session-level sums with shape [D, H, C].
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                "Bootstrap component values must be a torch.Tensor."
            )

        expected_shape = self.y_pred_raw.shape

        if values.shape != expected_shape:
            raise ValueError(
                "Bootstrap component tensor has an incompatible shape. "
                f"Expected {tuple(expected_shape)}, "
                f"got {tuple(values.shape)}."
            )

        values_cpu = (
            values
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        # Sum over assets:
        #
        # [B, H, N, C] -> [ over assets:
        #
        # [B, H, N, C] -> [B, H, C]
        per_window_sum = values_cpu.sum(
            dim=2
        )

        num_horizons = values_cpu.shape[1]
        num_channels = values_cpu.shape[3]

        session_sum = torch.zeros(
            (
                num_sessions,
                num_horizons,
                num_channels,
            ),
            dtype=torch.float64,
        )

        session_sum.index_add_(
            dim=0,
            index=session_inverse,
            source=per_window_sum,
        )

        return session_sum
    
    def _sum_bootstrap_values_by_session_preserve_assets(
        self,
        values: torch.Tensor,
        session_inverse: torch.Tensor,
        num_sessions: int,
    ) -> torch.Tensor:
        """
        Sum bootstrap component values by session without reducing the
        asset dimension.

        Input:
            values: [B, H, N, C]

        Output:
            [D, H, N, C]
        """
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                "Bootstrap component values must be a torch.Tensor."
            )

        expected_shape = self.y_pred_raw.shape

        if values.shape != expected_shape:
            raise ValueError(
                "Bootstrap component tensor has an incompatible shape. "
                f"Expected {tuple(expected_shape)}, "
                f"got {tuple(values.shape)}."
            )

        values_cpu = (
            values
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        session_sum = torch.zeros(
            (
                num_sessions,
                values_cpu.shape[1],
                values_cpu.shape[2],
                values_cpu.shape[3],
            ),
            dtype=torch.float64,
        )

        session_sum.index_add_(
            dim=0,
            index=session_inverse,
            source=values_cpu,
        )

        return session_sum

    def _aggregate_bootstrap_components_by_session(
        self,
        components: BootstrapMetricComponents,
    ) -> BootstrapSessionStatistics:
        """
        Aggregate one metric's pointwise bootstrap components into
        complete trading-session blocks.

        Session statistics are stored as sums and counts rather than
        daily averages.
        """
        if not isinstance(
            components,
            BootstrapMetricComponents,
        ):
            raise TypeError(
                "components must be a BootstrapMetricComponents "
                "instance."
            )

        session_ids, session_inverse = (
            self._get_bootstrap_session_mapping()
        )

        num_sessions = int(
            session_ids.numel()
        )

        if components.kind == "window_mean":
            if components.reference_values is not None:
                raise ValueError(
                    "window_mean bootstrap components must not contain "
                    "reference_values."
                )

            expected_shape = (
                self.y_pred_raw.shape[0],
                self.y_pred_raw.shape[1],
                self.y_pred_raw.shape[3],
            )

            if components.values.shape != expected_shape:
                raise ValueError(
                    "window_mean bootstrap values must have shape "
                    f"[B, H, C]. Expected {expected_shape}, "
                    f"got {tuple(components.values.shape)}."
                )

            values = (
                components.values
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )

            finite_mask = torch.isfinite(
                values
            )

            safe_values = torch.where(
                finite_mask,
                values,
                torch.zeros_like(values),
            )

            session_value_sum = torch.zeros(
                (
                    num_sessions,
                    values.shape[1],
                    values.shape[2],
                ),
                dtype=torch.float64,
            )

            session_observation_count = torch.zeros_like(
                session_value_sum
            )

            session_value_sum.index_add_(
                dim=0,
                index=session_inverse,
                source=safe_values,
            )

            session_observation_count.index_add_(
                dim=0,
                index=session_inverse,
                source=finite_mask.to(
                    dtype=torch.float64
                ),
            )

            return BootstrapSessionStatistics(
                kind="window_mean",
                session_ids=session_ids,
                observation_count=session_observation_count,
                value_sum=session_value_sum,
            )

        if components.kind == "assetwise_ratio":
            if components.reference_values is None:
                raise ValueError(
                    "assetwise_ratio bootstrap components require "
                    "reference_values."
                )

            windows_per_session = torch.bincount(
                session_inverse,
                minlength=num_sessions,
            )

            value_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=components.values,
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            reference_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=components.reference_values,
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            return BootstrapSessionStatistics(
                kind="assetwise_ratio",
                session_ids=session_ids,
                observation_count=windows_per_session,
                value_sum=value_sum,
                reference_sum=reference_sum,
            )


        if components.kind == "assetwise_correlation":
            if components.reference_values is None:
                raise ValueError(
                    "assetwise_correlation bootstrap components require "
                    "reference_values."
                )

            values = (
                components.values
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )

            references = (
                components.reference_values
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )

            valid_pair = (
                torch.isfinite(values)
                & torch.isfinite(references)
            )

            safe_values = torch.where(
                valid_pair,
                values,
                torch.zeros_like(values),
            )
            safe_references = torch.where(
                valid_pair,
                references,
                torch.zeros_like(references),
            )

            observation_count = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=valid_pair.to(dtype=torch.float64),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            value_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=safe_values,
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            reference_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=safe_references,
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            value_squared_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=safe_values.square(),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            reference_squared_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=safe_references.square(),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            cross_sum = (
                self
                ._sum_bootstrap_values_by_session_preserve_assets(
                    values=safe_values * safe_references,
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            return BootstrapSessionStatistics(
                kind="assetwise_correlation",
                session_ids=session_ids,
                observation_count=observation_count,
                value_sum=value_sum,
                reference_sum=reference_sum,
                value_squared_sum=value_squared_sum,
                reference_squared_sum=reference_squared_sum,
                cross_sum=cross_sum,
            )

        num_assets = int(
            self.y_pred_raw.shape[2]
        )

        windows_per_session = torch.bincount(
            session_inverse,
            minlength=num_sessions,
        )

        # Each forecast window contributes one observation per asset
        # for every horizon and channel.
        observation_count = (
            windows_per_session
            * num_assets
        )

        value_sum = (
            self._sum_bootstrap_values_by_session(
                values=components.values,
                session_inverse=session_inverse,
                num_sessions=num_sessions,
            )
        )

        if components.kind == "mean":
            if components.reference_values is not None:
                raise ValueError(
                    "Mean bootstrap components must not contain "
                    "reference_values."
                )

            return BootstrapSessionStatistics(
                kind="mean",
                session_ids=session_ids,
                observation_count=observation_count,
                value_sum=value_sum,
            )

        if components.reference_values is None:
            raise ValueError(
                f"{components.kind!r} bootstrap components require "
                "reference_values."
            )

        reference_sum = (
            self._sum_bootstrap_values_by_session(
                values=components.reference_values,
                session_inverse=session_inverse,
                num_sessions=num_sessions,
            )
        )

        if components.kind == "ratio":
            return BootstrapSessionStatistics(
                kind="ratio",
                session_ids=session_ids,
                observation_count=observation_count,
                value_sum=value_sum,
                reference_sum=reference_sum,
            )

        if components.kind == "correlation":
            # Convert to float64 before squaring or multiplying so
            # Pearson sufficient statistics are accumulated with
            # consistent numerical precision.
            correlation_values = (
                components.values
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )

            correlation_references = (
                components.reference_values
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )

            value_squared_sum = (
                self._sum_bootstrap_values_by_session(
                    values=correlation_values.square(),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            reference_squared_sum = (
                self._sum_bootstrap_values_by_session(
                    values=correlation_references.square(),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            cross_sum = (
                self._sum_bootstrap_values_by_session(
                    values=(
                        correlation_values
                        * correlation_references
                    ),
                    session_inverse=session_inverse,
                    num_sessions=num_sessions,
                )
            )

            return BootstrapSessionStatistics(
                kind="correlation",
                session_ids=session_ids,
                observation_count=observation_count,
                value_sum=value_sum,
                reference_sum=reference_sum,
                value_squared_sum=value_squared_sum,
                reference_squared_sum=(
                    reference_squared_sum
                ),
                cross_sum=cross_sum,
            )

        raise ValueError(
            "Unknown bootstrap component kind: "
            f"{components.kind!r}."
        )
    
    def _generate_bootstrap_session_counts(
        self,
        num_sessions: int,
        n_bootstrap: int,
        bootstrap_seed: int,
    ) -> torch.Tensor:
        """
        Generate vectorised trading-session bootstrap samples.

        Each bootstrap replicate samples num_sessions complete trading
        sessions with replacement.

        Args:
            num_sessions:
                Number of unique trading sessions D.

            n_bootstrap:
                Number of bootstrap replicates R.

            bootstrap_seed:
                Random seed used for reproducible resampling.

        Returns:
            Float64 tensor with shape [R, D].

            Each row contains the number of times each session was
            selected in that bootstrap replicate. Every row sums to D.
        """
        if not isinstance(num_sessions, int):
            raise TypeError(
                "num_sessions must be an integer."
            )

        if num_sessions < 2:
            raise ValueError(
                "Bootstrap evaluation requires at least two "
                "trading sessions."
            )

        if not isinstance(n_bootstrap, int):
            raise TypeError(
                "n_bootstrap must be an integer."
            )

        if n_bootstrap < 1:
            raise ValueError(
                "n_bootstrap must be at least 1."
            )

        if not isinstance(bootstrap_seed, int):
            raise TypeError(
                "bootstrap_seed must be an integer."
            )

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            bootstrap_seed
        )

        # For every replicate, sample D session positions from
        # {0, ..., D - 1}, with replacement.
        #
        # Shape: [R, D]
        sampled_session_positions = torch.randint(
            low=0,
            high=num_sessions,
            size=(
                n_bootstrap,
                num_sessions,
            ),
            generator=generator,
            dtype=torch.long,
            device="cpu",
        )

        bootstrap_session_counts = torch.zeros(
            (
                n_bootstrap,
                num_sessions,
            ),
            dtype=torch.float64,
            device="cpu",
        )

        bootstrap_session_counts.scatter_add_(
            dim=1,
            index=sampled_session_positions,
            src=torch.ones(
                (
                    n_bootstrap,
                    num_sessions,
                ),
                dtype=torch.float64,
                device="cpu",
            ),
        )

        return bootstrap_session_counts
    
    def _compute_mean_or_ratio_bootstrap_samples(
        self,
        statistics: BootstrapSessionStatistics,
        bootstrap_session_counts: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Reconstruct bootstrapped mean or ratio metrics from
        session-level sufficient statistics.

        Args:
            statistics:
                Session-level statistics for a metric whose kind is
                either "mean" or "ratio".

            bootstrap_session_counts:
                Number of times each session appears in each bootstrap
                replicate, with shape [R, D].

            eps:
                Minimum denominator mean required for a defined ratio.

        Returns:
            Bootstrap metric samples with shape [R, H, C].

        Here:
            R = number of bootstrap replicates
            D = number of trading sessions
            H = number of horizons
            C = number of channels
        """
        if not isinstance(
            statistics,
            BootstrapSessionStatistics,
        ):
            raise TypeError(
                "statistics must be a "
                "BootstrapSessionStatistics instance."
            )

        if statistics.kind not in {
            "mean",
            "ratio",
            "window_mean",
        }:
            raise ValueError(
                "Expected bootstrap statistics of kind "
                f"'mean' or 'ratio', got {statistics.kind!r}."
            )

        if not isinstance(
            bootstrap_session_counts,
            torch.Tensor,
        ):
            raise TypeError(
                "bootstrap_session_counts must be a torch.Tensor."
            )

        if bootstrap_session_counts.ndim != 2:
            raise ValueError(
                "Expected bootstrap_session_counts to have "
                "shape [R, D], got "
                f"{tuple(bootstrap_session_counts.shape)}."
            )

        num_sessions = int(
            statistics.session_ids.numel()
        )

        if bootstrap_session_counts.shape[1] != num_sessions:
            raise ValueError(
                "bootstrap_session_counts is not aligned with the "
                "session statistics. "
                f"Expected {num_sessions} session columns, got "
                f"{bootstrap_session_counts.shape[1]}."
            )

        if not torch.isfinite(
            bootstrap_session_counts
        ).all():
            raise ValueError(
                "bootstrap_session_counts contains NaN or "
                "infinite values."
            )

        if torch.any(
            bootstrap_session_counts < 0
        ):
            raise ValueError(
                "bootstrap_session_counts cannot contain "
                "negative values."
            )

        bootstrap_session_counts = (
            bootstrap_session_counts
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        value_sum = (
            statistics.value_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        observation_count = (
            statistics.observation_count
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        # [R, D] x [D, H, C] -> [R, H, C]
        bootstrap_value_sum = torch.einsum(
            "rd,dhc->rhc",
            bootstrap_session_counts,
            value_sum,
        )

        if statistics.kind == "window_mean":
            if observation_count.ndim != 3:
                raise ValueError(
                    "window_mean observation_count must have shape "
                    "[D, H, C]."
                )

            bootstrap_observation_count = torch.einsum(
                "rd,dhc->rhc",
                bootstrap_session_counts,
                observation_count,
            )

            valid_count = (
                bootstrap_observation_count > 0
            )

            return torch.where(
                valid_count,
                bootstrap_value_sum
                / bootstrap_observation_count.clamp_min(1.0),
                torch.full_like(
                    bootstrap_value_sum,
                    torch.nan,
                ),
            )

        # [R, D] x [D] -> [R]
        bootstrap_observation_count = (
            bootstrap_session_counts
            @ observation_count
        )

        if torch.any(
            bootstrap_observation_count <= 0
        ):
            raise ValueError(
                "Every bootstrap replicate must contain at least "
                "one forecast observation."
            )

        if statistics.kind == "mean":
            return (
                bootstrap_value_sum
                / bootstrap_observation_count[
                    :,
                    None,
                    None,
                ]
            )

        if statistics.reference_sum is None:
            raise ValueError(
                "Ratio statistics require reference_sum."
            )

        reference_sum = (
            statistics.reference_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        bootstrap_reference_sum = torch.einsum(
            "rd,dhc->rhc",
            bootstrap_session_counts,
            reference_sum,
        )

        # The ordinary relative-MAE implementation checks whether
        # Persistence MAE is greater than eps. Reproduce that rule
        # using the resampled denominator mean.
        bootstrap_reference_mean = (
            bootstrap_reference_sum
            / bootstrap_observation_count[
                :,
                None,
                None,
            ]
        )

        valid_denominator = (
            bootstrap_reference_mean > eps
        )

        return torch.where(
            valid_denominator,
            bootstrap_value_sum
            / bootstrap_reference_sum,
            torch.full_like(
                bootstrap_value_sum,
                torch.nan,
            ),
        )
    
    def _compute_correlation_bootstrap_samples(
        self,
        statistics: BootstrapSessionStatistics,
        bootstrap_session_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct bootstrapped Pearson correlations from
        session-level sufficient statistics.

        Args:
            statistics:
                Session-level statistics with kind="correlation".

            bootstrap_session_counts:
                Number of times each trading session appears in each
                bootstrap replicate. Shape [R, D].

        Returns:
            Pearson correlation samples with shape [R, H, C].

        Shapes:
            R = number of bootstrap replicates
            D = number of trading sessions
            H = number of forecast horizons
            C = number of target channels
        """
        if not isinstance(
            statistics,
            BootstrapSessionStatistics,
        ):
            raise TypeError(
                "statistics must be a "
                "BootstrapSessionStatistics instance."
            )

        if statistics.kind != "correlation":
            raise ValueError(
                "Expected statistics with kind='correlation', "
                f"got {statistics.kind!r}."
            )

        if statistics.reference_sum is None:
            raise ValueError(
                "Correlation statistics require reference_sum."
            )

        if statistics.value_squared_sum is None:
            raise ValueError(
                "Correlation statistics require value_squared_sum."
            )

        if statistics.reference_squared_sum is None:
            raise ValueError(
                "Correlation statistics require "
                "reference_squared_sum."
            )

        if statistics.cross_sum is None:
            raise ValueError(
                "Correlation statistics require cross_sum."
            )

        if not isinstance(
            bootstrap_session_counts,
            torch.Tensor,
        ):
            raise TypeError(
                "bootstrap_session_counts must be a torch.Tensor."
            )

        if bootstrap_session_counts.ndim != 2:
            raise ValueError(
                "Expected bootstrap_session_counts to have shape "
                "[R, D], got "
                f"{tuple(bootstrap_session_counts.shape)}."
            )

        num_sessions = int(
            statistics.session_ids.numel()
        )

        if bootstrap_session_counts.shape[1] != num_sessions:
            raise ValueError(
                "bootstrap_session_counts is not aligned with the "
                "session statistics. "
                f"Expected {num_sessions} session columns, got "
                f"{bootstrap_session_counts.shape[1]}."
            )

        if not torch.isfinite(
            bootstrap_session_counts
        ).all():
            raise ValueError(
                "bootstrap_session_counts contains NaN or "
                "infinite values."
            )

        if torch.any(
            bootstrap_session_counts < 0
        ):
            raise ValueError(
                "bootstrap_session_counts cannot contain "
                "negative values."
            )

        counts = (
            bootstrap_session_counts
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        observation_count = (
            statistics.observation_count
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        value_sum = (
            statistics.value_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        reference_sum = (
            statistics.reference_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        value_squared_sum = (
            statistics.value_squared_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        reference_squared_sum = (
            statistics.reference_squared_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        cross_sum = (
            statistics.cross_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        # [R, D] @ [D] -> [R]
        bootstrap_observation_count = (
            counts
            @ observation_count
        )

        if torch.any(
            bootstrap_observation_count <= 1
        ):
            raise ValueError(
                "Every Pearson bootstrap replicate requires at "
                "least two observations."
            )

        # [R, D] x [D, H, C] -> [R, H, C]
        bootstrap_value_sum = torch.einsum(
            "rd,dhc->rhc",
            counts,
            value_sum,
        )

        bootstrap_reference_sum = torch.einsum(
            "rd,dhc->rhc",
            counts,
            reference_sum,
        )

        bootstrap_value_squared_sum = torch.einsum(
            "rd,dhc->rhc",
            counts,
            value_squared_sum,
        )

        bootstrap_reference_squared_sum = torch.einsum(
            "rd,dhc->rhc",
            counts,
            reference_squared_sum,
        )

        bootstrap_cross_sum = torch.einsum(
            "rd,dhc->rhc",
            counts,
            cross_sum,
        )

        # [R] -> [R, 1, 1] for broadcasting over H and C.
        n = bootstrap_observation_count[
            :,
            None,
            None,
        ]

        numerator = (
            n * bootstrap_cross_sum
            - (
                bootstrap_value_sum
                * bootstrap_reference_sum
            )
        )

        value_variation = (
            n * bootstrap_value_squared_sum
            - bootstrap_value_sum.square()
        )

        reference_variation = (
            n * bootstrap_reference_squared_sum
            - bootstrap_reference_sum.square()
        )

        # Theoretically these terms are non-negative. Tiny negative
        # values may arise from floating-point cancellation.
        value_variation = torch.clamp_min(
            value_variation,
            0.0,
        )

        reference_variation = torch.clamp_min(
            reference_variation,
            0.0,
        )

        denominator = torch.sqrt(
            value_variation
            * reference_variation
        )

        valid_correlation = (
            denominator > 0
        )

        safe_denominator = torch.where(
            valid_correlation,
            denominator,
            torch.ones_like(
                denominator
            ),
        )

        correlation = (
            numerator
            / safe_denominator
        )

        return torch.where(
            valid_correlation,
            correlation,
            torch.full_like(
                correlation,
                torch.nan,
            ),
        )
    
    def _compute_assetwise_ratio_bootstrap_samples(
        self,
        statistics: BootstrapSessionStatistics,
        bootstrap_session_counts: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Calculate one movement-magnitude ratio per asset for each
        bootstrap replicate, then take the median across valid assets.

        Returns:
            Bootstrap samples with shape [R, H, C].
        """
        if statistics.kind != "assetwise_ratio":
            raise ValueError(
                "Expected statistics with kind='assetwise_ratio', "
                f"got {statistics.kind!r}."
            )

        if statistics.reference_sum is None:
            raise ValueError(
                "Asset-wise ratio statistics require reference_sum."
            )

        if bootstrap_session_counts.ndim != 2:
            raise ValueError(
                "Expected bootstrap_session_counts to have shape "
                f"[R, D], got {tuple(bootstrap_session_counts.shape)}."
            )

        counts = (
            bootstrap_session_counts
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        observation_count = (
            statistics.observation_count
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        # [R, D] x [D] -> [R]
        bootstrap_observation_count = (
            counts
            @ observation_count
        )

        if torch.any(
            bootstrap_observation_count <= 0
        ):
            raise ValueError(
                "Every bootstrap replicate must contain at least "
                "one forecast window."
            )

        value_sum = (
            statistics.value_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        reference_sum = (
            statistics.reference_sum
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        # [R, D] x [D, H, N, C] -> [R, H, N, C]
        bootstrap_value_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            value_sum,
        )

        bootstrap_reference_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            reference_sum,
        )

        bootstrap_reference_mean = (
            bootstrap_reference_sum
            / bootstrap_observation_count[
                :,
                None,
                None,
                None,
            ]
        )

        valid_assets = (
            bootstrap_reference_mean > eps
        )

        asset_ratios = torch.where(
            valid_assets,
            bootstrap_value_sum
            / bootstrap_reference_sum,
            torch.full_like(
                bootstrap_value_sum,
                torch.nan,
            ),
        )

        # [R, H, N, C] -> [R, H, C]
        return torch.nanmedian(
            asset_ratios,
            dim=2,
        ).values

    def _compute_assetwise_correlation_bootstrap_samples(
        self,
        statistics: BootstrapSessionStatistics,
        bootstrap_session_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct temporal Pearson correlations separately for each
        asset, then average valid asset correlations.

        Returns:
            Bootstrap samples with shape [R, H, C].
        """
        if not isinstance(
            statistics,
            BootstrapSessionStatistics,
        ):
            raise TypeError(
                "statistics must be a "
                "BootstrapSessionStatistics instance."
            )

        if statistics.kind != "assetwise_correlation":
            raise ValueError(
                "Expected statistics with "
                "kind='assetwise_correlation', "
                f"got {statistics.kind!r}."
            )

        required_statistics = {
            "reference_sum": statistics.reference_sum,
            "value_squared_sum": statistics.value_squared_sum,
            "reference_squared_sum": (
                statistics.reference_squared_sum
            ),
            "cross_sum": statistics.cross_sum,
        }

        missing_statistics = [
            name
            for name, value in required_statistics.items()
            if value is None
        ]

        if missing_statistics:
            raise ValueError(
                "Asset-wise correlation statistics are missing: "
                f"{missing_statistics}."
            )

        if not isinstance(
            bootstrap_session_counts,
            torch.Tensor,
        ):
            raise TypeError(
                "bootstrap_session_counts must be a torch.Tensor."
            )

        if bootstrap_session_counts.ndim != 2:
            raise ValueError(
                "Expected bootstrap_session_counts to have shape "
                "[R, D], got "
                f"{tuple(bootstrap_session_counts.shape)}."
            )

        num_sessions = int(
            statistics.session_ids.numel()
        )

        if bootstrap_session_counts.shape[1] != num_sessions:
            raise ValueError(
                "bootstrap_session_counts is not aligned with the "
                "session statistics. "
                f"Expected {num_sessions} session columns, got "
                f"{bootstrap_session_counts.shape[1]}."
            )

        counts = (
            bootstrap_session_counts
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        if not torch.isfinite(counts).all():
            raise ValueError(
                "bootstrap_session_counts contains NaN or infinite "
                "values."
            )

        if torch.any(counts < 0):
            raise ValueError(
                "bootstrap_session_counts cannot contain negative "
                "values."
            )

        observation_count = (
            statistics.observation_count
            .detach()
            .cpu()
            .to(dtype=torch.float64)
        )

        if observation_count.ndim == 1:
            # Backwards-compatible path for fully observed asset-wise
            # metrics created before pairwise counts were retained.
            bootstrap_observation_count = (
                counts
                @ observation_count
            )[
                :,
                None,
                None,
                None,
            ]

        elif observation_count.ndim == 4:
            # [R, D] x [D, H, N, C] -> [R, H, N, C]
            bootstrap_observation_count = torch.einsum(
                "rd,dhnc->rhnc",
                counts,
                observation_count,
            )

        else:
            raise ValueError(
                "Asset-wise correlation observation_count must have "
                "shape [D] or [D, H, N, C], got "
                f"{tuple(observation_count.shape)}."
            )

        value_sum = statistics.value_sum.to(dtype=torch.float64)
        reference_sum = statistics.reference_sum.to(dtype=torch.float64)
        value_squared_sum = statistics.value_squared_sum.to(
            dtype=torch.float64
        )
        reference_squared_sum = statistics.reference_squared_sum.to(
            dtype=torch.float64
        )
        cross_sum = statistics.cross_sum.to(dtype=torch.float64)

        # [R, D] x [D, H, N, C] -> [R, H, N, C]
        bootstrap_value_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            value_sum,
        )

        bootstrap_reference_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            reference_sum,
        )

        bootstrap_value_squared_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            value_squared_sum,
        )

        bootstrap_reference_squared_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            reference_squared_sum,
        )

        bootstrap_cross_sum = torch.einsum(
            "rd,dhnc->rhnc",
            counts,
            cross_sum,
        )

        n = bootstrap_observation_count

        numerator = (
            n * bootstrap_cross_sum
            - bootstrap_value_sum * bootstrap_reference_sum
        )

        value_variation = torch.clamp_min(
            n * bootstrap_value_squared_sum
            - bootstrap_value_sum.square(),
            0.0,
        )

        reference_variation = torch.clamp_min(
            n * bootstrap_reference_squared_sum
            - bootstrap_reference_sum.square(),
            0.0,
        )

        denominator = torch.sqrt(
            value_variation
            * reference_variation
        )

        valid_correlation = (
            n > 1
        ) & (
            denominator > 0
        )

        asset_correlations = torch.where(
            valid_correlation,
            numerator / torch.where(
                valid_correlation,
                denominator,
                torch.ones_like(denominator),
            ),
            torch.full_like(
                denominator,
                torch.nan,
            ),
        )

        # [R, H, N, C] -> [R, H, C]
        return torch.nanmean(
            asset_correlations,
            dim=2,
        )


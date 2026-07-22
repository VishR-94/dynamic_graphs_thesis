from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from functools import partial
import torch
from torchmetrics.functional.regression import pearson_corrcoef
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
    validate_prediction_shapes(y_pred, y_true)

    if tuple(reduce_dims) != (0, 2):
        raise ValueError(
            "pearson_correlation currently supports "
            "reduce_dims=(0, 2) only."
        )

    batch_size, num_horizons, num_assets, num_channels = y_pred.shape

    # [B, H, N, C] -> [B*N, H*C]
    pred_flat = (
        y_pred.permute(0, 2, 1, 3)
        .reshape(batch_size * num_assets, -1)
    )

    true_flat = (
        y_true.permute(0, 2, 1, 3)
        .reshape(batch_size * num_assets, -1)
    )

    correlations = pearson_corrcoef(
        pred_flat,
        true_flat,
    )

    return correlations.reshape(num_horizons, num_channels)

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
        
        window_mean:
            values contains one metric value per forecast window,
            horizon and channel, with shape [B, H, C].
    """

    kind: Literal[
        "mean",
        "ratio",
        "correlation",
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
        observation_count:        [D]
        value_sum:                 [D, H, C]

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
    """

    kind: Literal[
        "mean",
        "ratio",
        "correlation",
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

    bootstrap_components: BootstrapComponentFunction

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
    
    def _build_metric_registry(
        self,
    ) -> dict[str, MetricDefinition]:
        """
        Map each public metric name to its ordinary computation and
        bootstrap-component builder.
        """
        return {
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
        }
    
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

            kwargs = resolved_metric_kwargs[
                metric_name
            ]

            components = (
                metric_definition
                .bootstrap_components(
                    **kwargs,
                )
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

            else:
                raise ValueError(
                    "Unknown bootstrap statistics kind: "
                    f"{statistics.kind!r}."
                )

            expected_metric_shape = (
                ordinary_results[
                    metric_name
                ].shape
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
                "value": (
                    ordinary_results[
                        metric_name
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
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
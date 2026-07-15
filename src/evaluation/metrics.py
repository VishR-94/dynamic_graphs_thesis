from collections.abc import Callable, Sequence
from typing import Any
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


def mae(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduce_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """
    Mean absolute error.

    Args:
        y_pred:
            Prediction tensor.

        y_true:
            Ground-truth tensor.

        reduce_dims:
            Dimensions to average over. If None, average over all dimensions.
    """
    validate_prediction_shapes(y_pred, y_true)

    values = (y_pred - y_true).abs()

    return reduce_metric(values, reduce_dims=reduce_dims)


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

    model_absolute_error = torch.abs(y_pred - y_true)
    persistence_absolute_error = torch.abs(
        persistence_pred - y_true
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
    validate_prediction_shapes(y_pred, y_true)
    validate_prediction_shapes(persistence_pred, y_true)

    if not 0.0 <= tie_value <= 1.0:
        raise ValueError(
            "tie_value must lie between 0 and 1, "
            f"got {tie_value}."
        )

    model_absolute_error = torch.abs(y_pred - y_true)
    persistence_absolute_error = torch.abs(
        persistence_pred - y_true
    )

    ties = torch.isclose(
        model_absolute_error,
        persistence_absolute_error,
        rtol=rtol,
        atol=atol,
    )

    wins = (
        model_absolute_error < persistence_absolute_error
    ) & ~ties

    pointwise_scores = (
        wins.to(dtype=model_absolute_error.dtype)
        + tie_value
        * ties.to(dtype=model_absolute_error.dtype)
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
    validate_prediction_shapes(y_pred, y_true)

    if not isinstance(mase_scale, torch.Tensor):
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

    if not torch.isfinite(mase_scale).all():
        raise ValueError(
            "mase_scale contains NaN or infinite values."
        )

    mase_scale = mase_scale.to(
        device=y_pred.device,
        dtype=y_pred.dtype,
    )

    valid_scale = mase_scale > eps

    safe_scale = torch.where(
        valid_scale,
        mase_scale,
        torch.full_like(mase_scale, torch.nan),
    )

    scaled_absolute_error = (
        torch.abs(y_pred - y_true)
        / safe_scale
    )

    return reduce_metric(
        scaled_absolute_error,
        reduce_dims=reduce_dims,
    )

MetricFunction = Callable[..., torch.Tensor]
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

    def _build_metric_registry(
        self,
    ) -> dict[str, MetricFunction]:
        """
        Map public metric names to evaluator callables.

        Every registered callable must accept reduce_dims and return a
        torch.Tensor.
        """
        return {
            "cumulative_log_change_mae": partial(
                self.compute_pairwise_metric,
                metric_fn=mae,
                output_space="cumulative_log_change",
            ),
            "mase": self.compute_mase,
            "relative_mae_vs_persistence": (
                self.compute_relative_mae_vs_persistence
            ),
            "persistence_win_rate": (
                self.compute_persistence_win_rate
            ),
        }
    
    @property
    def available_metrics(self) -> tuple[str, ...]:
        """
        Return the names of all metrics currently available through
        evaluate().
        """
        return tuple(self._metric_registry)
    
    def evaluate(
        self,
        metrics: str | Sequence[str],
        reduce_dims: Sequence[int] | None = None,
        metric_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute one or more registered forecast metrics.

        Args:
            metrics:
                One metric name or a sequence of metric names.

            reduce_dims:
                Tensor dimensions over which every requested metric is
                reduced.

                For prediction tensors with shape [B, H, N, C],
                reduce_dims=(0, 2) retains horizon and channel, returning
                tensors with shape [H, C].

            metric_kwargs:
                Optional metric-specific keyword arguments.

                Example:
                    {
                        "persistence_win_rate": {
                            "tie_value": 0.0,
                        },
                        "mase": {
                            "eps": 1e-10,
                        },
                    }

        Returns:
            Dictionary mapping each requested metric name to its result.
        """
        if isinstance(metrics, str):
            metric_names = [metrics]
        else:
            metric_names = list(metrics)

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
                f"Available metrics: {list(self.available_metrics)}."
            )

        if metric_kwargs is None:
            metric_kwargs = {}

        unknown_kwargs_metrics = set(metric_kwargs).difference(
            metric_names
        )

        if unknown_kwargs_metrics:
            raise ValueError(
                "metric_kwargs contains entries for metrics that were "
                "not requested: "
                f"{sorted(unknown_kwargs_metrics)}."
            )

        results: dict[str, torch.Tensor] = {}

        for metric_name in metric_names:
            kwargs = dict(
                metric_kwargs.get(metric_name, {})
            )

            if "reduce_dims" in kwargs:
                raise ValueError(
                    "Pass reduce_dims through evaluate(), not through "
                    f"metric_kwargs[{metric_name!r}]."
                )

            metric_fn = self._metric_registry[metric_name]

            results[metric_name] = metric_fn(
                reduce_dims=reduce_dims,
                **kwargs,
            )

        return results

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
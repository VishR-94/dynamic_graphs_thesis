from collections.abc import Sequence
import torch

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
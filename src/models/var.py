from typing import Any

import numpy as np
import torch
from statsmodels.tsa.vector_ar.var_model import VAR
from torch.utils.data import DataLoader

from src.data.data_generator import WindowedCandleDataset, build_log_change_split
from src.data.load_candle_data import get_channel_index
from src.evaluation.prediction_transforms import (
    cumulative_log_change_to_raw,
    one_step_returns_to_cumulative_horizons,
    raw_to_cumulative_log_change,
)

SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]

'''
Implement VAR benchmark. Note that since we need stationarity for this, we will work in log change space, not raw data.
With a VAR(p) model, we model all assets simulataneously using p lags. As an example, a VAR(1) model with 2 variables has:

                    y_{1,t} = c_1 + a_{1,1}y_{1,t-1} + a_{1,2}y_{2,t-1} + e_{1,t}
                    y_{2,t} = c_2 + a_{2,1}y_{1,t-1} + a_{2,2}y_{2,t-1} + e_{2,t}

where both e_{1,t} and e_{2,t} are usually Gaussian noise. We need to choose p - this can be done automatically using
the statsmodels implementation that we use here. We fit the model on the training data (which will give N separate
equations) and then forecast the test data. 
'''

class VarBaseline:
    """
    Vector autoregression baseline.

    This model fits one VAR model per target channel.

    For close-only targets, this means:
        one VAR model over all assets' close log changes.

    Prediction flow:
        raw context
        -> one-step log-change context
        -> VAR forecasts future one-step log changes
        -> cumulative horizon log changes
        -> optionally raw values
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int,
        maxlags: int = 5,
        ic: str | None = "aic",
        trend: str = "n",
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        if maxlags < 0:
            raise ValueError("maxlags must be >= 0.")

        if ic not in {None, "aic", "fpe", "hqic", "bic"}:
            raise ValueError(
                f"ic must be one of None, 'aic', 'fpe', 'hqic', 'bic', got {ic}."
            )

        self.context_length = context_length
        self.horizons = horizons
        self.target_channels = target_channels
        self.stride = stride

        self.maxlags = maxlags
        self.ic = ic
        self.trend = trend

        self.train_split: SplitDict | None = None
        self.val_split: SplitDict | None = None

        self.fitted_models: dict[int, Any] = {}
        self.selected_lags: dict[str, int] = {}
        self.fallback_means: dict[int, np.ndarray] = {}
        self.failed_models: dict[str, str] = {}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        maxlags: int = 5,
        ic: str | None = "aic",
        trend: str = "n",
    ) -> "VarBaseline":
        forecasting_config = config["forecasting"]

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=list(forecasting_config["horizons"]),
            target_channels=list(forecasting_config["target_channels"]),
            stride=int(forecasting_config["stride"]),
            maxlags=maxlags,
            ic=ic,
            trend=trend,
        )

    def fit(
        self,
        train_split: SplitDict,
        val_split: SplitDict | None = None,
        verbose: bool = True,
    ) -> "VarBaseline":
        """
        Fit one VAR model per target channel.
        """
        self.train_split = train_split
        self.val_split = val_split

        self.fitted_models = {}
        self.selected_lags = {}
        self.fallback_means = {}
        self.failed_models = {}

        train_log_change = build_log_change_split(
            split=train_split,
            eps=1e-8,
        )

        target_channel_ids = [
            get_channel_index(train_log_change, channel)
            for channel in self.target_channels
        ]

        if verbose:
            print(
                f"Fitting {len(self.target_channels)} VAR model(s) "
                f"with maxlags={self.maxlags}, ic={self.ic}..."
            )

        for channel_position, channel_id in enumerate(target_channel_ids):
            channel_name = self.target_channels[channel_position]

            channel_returns_by_day = []

            for x_day, _, _ in train_log_change["samples"]:
                channel_returns_by_day.append(
                    x_day[:, :, channel_id].float()
                )

            channel_returns = torch.cat(channel_returns_by_day, dim=0)

            series_np = (
                channel_returns.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            model_result, selected_lag, fallback_mean, error_message = (
                self._fit_one_channel(series_np)
            )

            self.fitted_models[channel_position] = model_result
            self.selected_lags[channel_name] = selected_lag
            self.fallback_means[channel_position] = fallback_mean

            if error_message is not None:
                self.failed_models[channel_name] = error_message

            if verbose:
                print(
                    f"  {channel_name}: selected_lag={selected_lag}, "
                    f"failed={error_message is not None}"
                )

        if verbose:
            print("Finished fitting VAR models.")
            print(f"Failed models: {len(self.failed_models)}")

        return self

    def fitted_values(
        self,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Return VAR predictions and true values on the training split.
        """
        if self.train_split is None:
            raise ValueError("Call fit(...) before fitted_values().")

        return self.predict(
            split=self.train_split,
            output_space=output_space,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict(
        self,
        split: SplitDict,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Generate VAR predictions.
        """
        if len(self.fitted_models) == 0:
            raise ValueError("Call fit(...) before predict(...).")

        if output_space not in {"cumulative_log_change", "raw"}:
            raise ValueError(
                "output_space must be either 'cumulative_log_change' or "
                f"'raw', got {output_space}."
            )

        dataset = WindowedCandleDataset.from_config(
            split=split,
            config=self._dataset_config(),
            normaliser=None,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        max_horizon = max(self.horizons)

        all_y_pred = []
        all_y_true = []
        all_sample_idx = []
        all_origin_idx = []
        all_target_indices = []

        for batch in loader:
            x_context = batch["x"].float()
            y_true_raw = batch["y"].float()
            last_context_target = batch["last_context_target"].float()

            context_returns = self._context_to_one_step_returns(x_context)

            one_step_pred = self._forecast_batch(
                context_returns=context_returns,
                steps=max_horizon,
            )

            y_pred_cumulative = one_step_returns_to_cumulative_horizons(
                one_step_returns=one_step_pred,
                horizons=self.horizons,
            )

            y_true_cumulative = raw_to_cumulative_log_change(
                y_raw=y_true_raw,
                last_context_target=last_context_target,
            )

            if output_space == "cumulative_log_change":
                y_pred = y_pred_cumulative
                y_true = y_true_cumulative

            else:
                y_pred = cumulative_log_change_to_raw(
                    cumulative_log_change=y_pred_cumulative,
                    last_context_target=last_context_target,
                )

                y_true = y_true_raw

            all_y_pred.append(y_pred)
            all_y_true.append(y_true)
            all_sample_idx.append(batch["sample_idx"])
            all_origin_idx.append(batch["origin_idx"])
            all_target_indices.append(batch["target_indices"])

        y_pred = torch.cat(all_y_pred, dim=0)
        y_true = torch.cat(all_y_true, dim=0)

        sample_idx = torch.cat(all_sample_idx, dim=0)
        origin_idx = torch.cat(all_origin_idx, dim=0)
        target_indices = torch.cat(all_target_indices, dim=0)

        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "output_space": output_space,
            "channels": self.target_channels,
            "horizons": self.horizons,
            "asset_cols": split["asset_cols"],
            "sample_idx": sample_idx,
            "origin_idx": origin_idx,
            "target_indices": target_indices,
            "maxlags": self.maxlags,
            "ic": self.ic,
            "trend": self.trend,
            "selected_lags": self.selected_lags,
            "failed_models": self.failed_models,
        }

    def _fit_one_channel(
        self,
        series: np.ndarray,
    ) -> tuple[Any | None, int, np.ndarray, str | None]:
        """
        Fit one VAR model to one channel's asset return matrix.

        Args:
            series:
                Array with shape [time, num_assets].
        """
        if series.ndim != 2:
            raise ValueError(
                f"Expected series to have shape [time, num_assets], "
                f"got {series.shape}."
            )

        series = np.nan_to_num(
            series,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        fallback_mean = np.mean(
            series,
            axis=0,
            dtype=np.float64,
        )

        if series.shape[0] <= self.maxlags + 1:
            return (
                None,
                0,
                fallback_mean,
                "Series is too short for requested maxlags.",
            )

        try:

            model = VAR(series)

            model_result = model.fit(
                maxlags=self.maxlags,
                ic=self.ic,
                trend=self.trend,
            )

            selected_lag = int(model_result.k_ar)

            return model_result, selected_lag, fallback_mean, None

        except Exception as exc:
            return None, 0, fallback_mean, str(exc)

    def _context_to_one_step_returns(
        self,
        x_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw context values into one-step log changes.

        Args:
            x_context:
                Raw context tensor with shape [B, T, N, C].

        Returns:
            One-step log changes with shape [B, T - 1, N, C].
        """
        if x_context.ndim != 4:
            raise ValueError(
                f"Expected x_context to have shape [B, T, N, C], "
                f"got {tuple(x_context.shape)}."
            )

        log_values = torch.log(x_context.clamp_min(1e-8))

        returns = log_values[:, 1:, :, :] - log_values[:, :-1, :, :]

        return returns

    def _forecast_batch(
        self,
        context_returns: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        Forecast future one-step returns for a batch.

        Args:
            context_returns:
                Tensor with shape [B, context_length - 1, N, C].

            steps:
                Number of future one-step returns to forecast.

        Returns:
            Forecast tensor with shape [B, steps, N, C].
        """
        batch_size = context_returns.shape[0]
        num_assets = context_returns.shape[2]
        num_channels = context_returns.shape[3]

        forecasts = torch.empty(
            batch_size,
            steps,
            num_assets,
            num_channels,
            dtype=torch.float32,
        )

        for batch_idx in range(batch_size):
            for channel_position in range(num_channels):
                context_series = context_returns[
                    batch_idx,
                    :,
                    :,
                    channel_position,
                ]

                forecast_np = self._forecast_one_channel(
                    channel_position=channel_position,
                    context_series=context_series,
                    steps=steps,
                )

                forecasts[
                    batch_idx,
                    :,
                    :,
                    channel_position,
                ] = torch.from_numpy(forecast_np).float()

        return forecasts

    def _forecast_one_channel(
        self,
        channel_position: int,
        context_series: torch.Tensor,
        steps: int,
    ) -> np.ndarray:
        """
        Forecast one target channel for all assets.

        Args:
            channel_position:
                Position in self.target_channels.

            context_series:
                Tensor with shape [context_length - 1, num_assets].

            steps:
                Number of future one-step returns to forecast.

        Returns:
            Array with shape [steps, num_assets].
        """
        model_result = self.fitted_models.get(channel_position)
        fallback_mean = self.fallback_means[channel_position]

        if model_result is None:
            return np.repeat(
                fallback_mean.reshape(1, -1),
                repeats=steps,
                axis=0,
            )

        selected_lag = int(model_result.k_ar)

        try:
            context_np = (
                context_series.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            context_np = np.nan_to_num(
                context_np,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if selected_lag == 0:
                if self.trend == "n":
                    forecast_np = np.zeros(
                        shape=(steps, context_np.shape[1]),
                        dtype=np.float64,
                    )
                else:
                    intercept = np.asarray(
                        model_result.intercept,
                        dtype=np.float64,
                    )

                    forecast_np = np.repeat(
                        intercept.reshape(1, -1),
                        repeats=steps,
                        axis=0,
                    )

            else:
                if context_np.shape[0] < selected_lag:
                    return np.repeat(
                        fallback_mean.reshape(1, -1),
                        repeats=steps,
                        axis=0,
                    )

                y_prior = context_np[-selected_lag:] 

                forecast_np = model_result.forecast(
                    y=y_prior,
                    steps=steps,
                )

            forecast_np = np.asarray(
                forecast_np,
                dtype=np.float64,
            )

            if not np.isfinite(forecast_np).all():
                return np.repeat(
                    fallback_mean.reshape(1, -1),
                    repeats=steps,
                    axis=0,
                )

            if np.abs(forecast_np).max() > 0.1:
                return np.repeat(
                    fallback_mean.reshape(1, -1),
                    repeats=steps,
                    axis=0,
                )

            return forecast_np

        except Exception:
            return np.repeat(
                fallback_mean.reshape(1, -1),
                repeats=steps,
                axis=0,
            )

    def _dataset_config(self) -> dict[str, Any]:
        """
        Build a minimal config for WindowedCandleDataset.
        """
        return {
            "forecasting": {
                "context_length": self.context_length,
                "horizons": self.horizons,
                "stride": self.stride,
                "input_channels": self.target_channels,
                "target_channels": self.target_channels,
            }
        }

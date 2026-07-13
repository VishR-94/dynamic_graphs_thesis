from typing import Any
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.data_generator import WindowedCandleDataset, build_log_change_split
from src.data.load_candle_data import get_channel_index
from src.evaluation.prediction_transforms import (
    cumulative_log_change_to_raw,
    one_step_returns_to_cumulative_horizons,
    raw_to_cumulative_log_change,
)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima


'''
Implement an ARIMA benchmark. Note that since we need stationarity for this, we will work in log change space, not raw data.
We have the option to either implement a "simple" ARIMA which is ARIMA(1,0,1), or to use auto.arima to find the optimum
p, d and q parameters. Since ARIMA is a univariate model, we will fit 1 ARIMA per asset per target channel, which means
we will have 93x4=372 models. We will use the pmdarima library to auto select the order for each model and the SARIMAX
implementation to actually forecast. This is because pmdarima only allows you to forecast from the end of the training data
where as SARIMAX allows you to supply new context to an existing fit.

NOTE: With a short stride this will take long to run - we are fitting 372 models, possible auto selecting the order on each.
Once we have fit the models, the number of forecasting calls made is then:
    num_forecasting_calls = num_assets * num_test_days * num_target_channels * stride_factor
where stride_factor = floor((last_origin - first_origin)/stride) + 1
and first_origin = context_length - 1
    last_origin = T - max_horizon - 1

That means with max_horizon = context_length = 60, T = 390, stride = 15), we have
    num_forecasting_calls = 93 * 20 * 4 * 19 = 141,360 forecasting calls 

This can take hours.
'''

SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]


class ArimaBaseline:
    """
    Univariate ARIMA baseline.

    This model fits one ARIMA model per asset/channel on one-step log changes.

    Prediction flow:
        raw context
        -> one-step log-change context
        -> ARIMA forecasts future one-step log changes
        -> cumulative horizon log changes
        -> optionally raw values
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int,
        fit_mode: str = "simple",
        order: tuple[int, int, int] = (1, 0, 1),
        auto_max_p: int = 3,
        auto_max_q: int = 3,
        information_criterion: str = "aic",
        trend: str | None = "n",
        eps: float = 1e-8,
        maxiter: int = 50,
        optim_method: str = 'powell',
    ) -> None:
        if fit_mode not in {"simple", "auto"}:
            raise ValueError(
                f"fit_mode must be 'simple' or 'auto', got {fit_mode}."
            )

        if stride < 1:
            raise ValueError("stride must be >= 1.")

        self.context_length = context_length
        self.horizons = horizons
        self.target_channels = target_channels
        self.stride = stride

        self.fit_mode = fit_mode
        self.order = order
        self.auto_max_p = auto_max_p
        self.auto_max_q = auto_max_q
        self.information_criterion = information_criterion
        self.trend = trend
        self.eps = eps
        self.maxiter = maxiter
        self.optim_method = optim_method

        self.train_split: SplitDict | None = None
        self.val_split: SplitDict | None = None

        self.fitted_models: dict[tuple[int, int], Any] = {}
        self.selected_orders: dict[tuple[int, int], tuple[int, int, int] | None] = {}
        self.fallback_means: dict[tuple[int, int], float] = {}
        self.failed_models: dict[tuple[int, int], str] = {}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        fit_mode: str = "simple",
        order: tuple[int, int, int] = (1, 0, 1),
        auto_max_p: int = 3,
        auto_max_q: int = 3,
        information_criterion: str = "aic",
        trend: str | None = "c",
        eps: float = 1e-8,
        maxiter: int = 50,
        optim_method: str = 'powell',
    ) -> "ArimaBaseline":
        forecasting_config = config["forecasting"]

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=list(forecasting_config["horizons"]),
            target_channels=list(forecasting_config["target_channels"]),
            stride=int(forecasting_config["stride"]),
            fit_mode=fit_mode,
            order=order,
            auto_max_p=auto_max_p,
            auto_max_q=auto_max_q,
            information_criterion=information_criterion,
            trend=trend,
            eps=eps,
            maxiter=maxiter,
            optim_method = optim_method
        )

    def fit(
        self,
        train_split: SplitDict,
        val_split: SplitDict | None = None,
        fit_mode: str | None = None,
        verbose: bool = True,
    ) -> "ArimaBaseline":
        """
        Fit one ARIMA model per asset/channel.

        Args:
            train_split:
                Cleaned raw candle training split.

            val_split:
                Optional validation split, stored for interface consistency.

            fit_mode:
                If provided, overrides the current fit mode.

                Options:
                    "simple": fit fixed ARIMA(self.order)
                    "auto": search order using pmdarima.auto_arima

            verbose:
                Whether to print progress.
        """
        if fit_mode is not None:
            if fit_mode not in {"simple", "auto"}:
                raise ValueError(
                    f"fit_mode must be 'simple' or 'auto', got {fit_mode}."
                )

            self.fit_mode = fit_mode

        self.train_split = train_split
        self.val_split = val_split

        self.fitted_models = {}
        self.selected_orders = {}
        self.fallback_means = {}
        self.failed_models = {}

        train_log_change = build_log_change_split(
            split=train_split,
            eps=self.eps,
        )

        target_channel_ids = [
            get_channel_index(train_log_change, channel)
            for channel in self.target_channels
        ]

        train_returns_by_day = []

        for x_day, _, _ in train_log_change["samples"]:
            train_returns_by_day.append(
                x_day[:, :, target_channel_ids].float()
            )

        train_returns = torch.cat(train_returns_by_day, dim=0)

        num_assets = train_returns.shape[1]
        num_channels = train_returns.shape[2]
        total_models = num_assets * num_channels

        if verbose:
            print(
                f"Fitting {total_models} ARIMA models "
                f"using fit_mode='{self.fit_mode}'..."
            )

        model_count = 0

        for asset_idx in range(num_assets):
            for channel_idx in range(num_channels):
                key = (asset_idx, channel_idx)

                series = train_returns[:, asset_idx, channel_idx]
                series_np = series.detach().cpu().numpy().astype(np.float64)

                model_result, selected_order, fallback_mean, error_message = (
                    self._fit_one_series(series_np)
                )

                self.fitted_models[key] = model_result
                self.selected_orders[key] = selected_order
                self.fallback_means[key] = fallback_mean

                if error_message is not None:
                    self.failed_models[key] = error_message

                model_count += 1

                if verbose and model_count % 10 == 0:
                    print(f"  fitted {model_count}/{total_models}")

        if verbose:
            print(f"Finished fitting ARIMA models.")
            print(f"Failed models: {len(self.failed_models)}")

        return self

    def fitted_values(
        self,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Return ARIMA predictions and true values on the training split.

        This uses the same rolling-window prediction logic as predict(...),
        but applies it to the stored training split.
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
        Generate ARIMA predictions.

        Args:
            split:
                Cleaned raw candle split.

            output_space:
                Either:
                    "cumulative_log_change"
                    "raw"

            batch_size:
                DataLoader batch size.

            num_workers:
                DataLoader workers.

        Returns:
            Dictionary containing y_pred and y_true with shape:
                [num_examples, num_horizons, num_assets, num_channels]
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
                eps=self.eps,
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
            "fit_mode": self.fit_mode,
            "selected_orders": self.selected_orders,
            "failed_models": self.failed_models,
        }

    def _fit_one_series(
        self,
        series: np.ndarray,
    ) -> tuple[Any | None, tuple[int, int, int] | None, float, str | None]:
        """
        Fit one ARIMA model to one asset/channel return series.
        """
        finite_mask = np.isfinite(series)
        finite_series = series[finite_mask]

        if finite_series.size == 0:
            return None, None, 0.0, "Series contains no finite values."

        fallback_mean = float(np.mean(finite_series))

        if finite_series.size < 20:
            return None, None, fallback_mean, "Series is too short."

        if np.std(finite_series) < 1e-12:
            return None, None, fallback_mean, "Series is approximately constant."

        try:

            if self.fit_mode == "auto":
                selected_order = self._select_auto_order(finite_series)
            else:
                selected_order = self.order

            model = SARIMAX(
                finite_series,
                order=selected_order,
                trend=self.trend,
                enforce_stationarity=True,
                enforce_invertibility=True,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Non-stationary starting autoregressive parameters found.*",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Non-invertible starting MA parameters found.*",
                )

                model_result = model.fit(
                    method=self.optim_method,
                    disp=False,
                    maxiter=self.maxiter,
                )

            return model_result, selected_order, fallback_mean, None

        except Exception as exc:
            return None, None, fallback_mean, str(exc)

    def _select_auto_order(self, series: np.ndarray) -> tuple[int, int, int]:
        """
        Select ARIMA order using pmdarima.auto_arima.

        Since we fit on log changes, we keep d=0 and search p/q.
        """

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in reciprocal",
                category=RuntimeWarning,
            )

            auto_model = auto_arima(
                series,
                start_p=0,
                start_q=0,
                max_p=self.auto_max_p,
                max_q=self.auto_max_q,
                d=0,
                stationary=True,
                seasonal=False,
                information_criterion=self.information_criterion,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                with_intercept=True,
                maxiter=self.maxiter,
            )

        return tuple(auto_model.order)

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

        log_values = torch.log(x_context.clamp_min(self.eps))

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
            for asset_idx in range(num_assets):
                for channel_idx in range(num_channels):
                    context_series = context_returns[
                        batch_idx,
                        :,
                        asset_idx,
                        channel_idx,
                    ]

                    forecast_np = self._forecast_one_series(
                        asset_idx=asset_idx,
                        channel_idx=channel_idx,
                        context_series=context_series,
                        steps=steps,
                    )

                    forecasts[
                        batch_idx,
                        :,
                        asset_idx,
                        channel_idx,
                    ] = torch.from_numpy(forecast_np).float()

        return forecasts

    def _forecast_one_series(
        self,
        asset_idx: int,
        channel_idx: int,
        context_series: torch.Tensor,
        steps: int,
    ) -> np.ndarray:
        """
        Forecast one asset/channel from one context window.
        """
        key = (asset_idx, channel_idx)

        fallback_mean = self.fallback_means.get(key, 0.0)
        model_result = self.fitted_models.get(key)

        if model_result is None:
            return np.full(
                shape=steps,
                fill_value=fallback_mean,
                dtype=np.float64,
            )

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

        try:
            applied_result = model_result.apply(
                context_np,
                refit=False,
            )

            forecast = applied_result.forecast(steps=steps)

            forecast_np = np.asarray(
                forecast,
                dtype=np.float64,
            )

            forecast_np = np.nan_to_num(
                forecast_np,
                nan=fallback_mean,
                posinf=fallback_mean,
                neginf=fallback_mean,
            )

            return forecast_np

        except Exception:
            return np.full(
                shape=steps,
                fill_value=fallback_mean,
                dtype=np.float64,
            )

    def _dataset_config(self) -> dict[str, Any]:
        """
        Build a minimal config for WindowedCandleDataset.

        ARIMA only uses the target channels as univariate input series.
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
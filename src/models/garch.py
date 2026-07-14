from typing import Any

import numpy as np
import torch
from arch import arch_model
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
Implement GARCH benchmark. Note that since we need stationarity for this, we will work in log change space, not raw data.
We only implement a GARCH(1,1) model, which is specified as follows. If y_t is the return at time t, z_t ~ N(0,1) and 
sigma_t is an unobserved/latent volatility, then our estimating equations are given by:

                        y_t = mu + e_t
                        e_t = sigma_t * z_t
                        sigma_t = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

beta is the persistence parameter (the larger beta the more persistent volatility is) and alpha is the parameter that
allows the model to adapt to a new shock. For this model to be stationary, we need omega > 0, alpha,beta >= 0 and
alpha + beta <= 1. Note that when we use this model to forecast the returns (y_t), since E[z_t] = 0, our y_t forecasts will
all simply be mu (i.e. constant returns). We can make this more flexible by using an AR(1) mean specification, so our
full estimating equations become:
                        
                        y_t = mu + phi * y_{t-1} + e_t
                        e_t = sigma_t * z_t
                        sigma_t = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

The model is optimised jointly using MLE - e_t ~ N(0,sigma_t^2), and e_t = y_t - mu (- phi * y_{t-1}). Since we dont
observe sigma_t, we use an intial guess (usually the variance over the training data), and start with an initial guess
of Theta = {omega, alpha, beta, mu, phi} and then use gradient descent on the negative log likelihood, updating the value 
of sigma_t as we update the parameter values. 

In our implementation we have the option to fit the GARCH(1,1) with (a) no mean model (mu=0), (b) constant mean model
(mu = mu), (c) AR(1) mean model (mu = mu + phi * y_{t-1}). The default is AR(1)
'''


class GarchBaseline:
    """
    Univariate GARCH(1,1) baseline.

    This model fits one GARCH(1,1) model per asset/channel on one-step
    log changes.

    Prediction flow:
        raw context
        -> one-step log-change context
        -> GARCH forecasts future one-step mean and variance
        -> cumulative horizon mean log changes
        -> cumulative horizon variance estimates
        -> optionally raw values
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int,
        mean: str = "AR",
        dist: str = "normal",
        eps: float = 1e-8,
        maxiter: int = 1000,
        return_scale: int = 10000.0
    ) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1.")

        if mean not in {"Constant", "Zero", "AR"}:
            raise ValueError(
                f"mean must be either 'AR', 'Constant' or 'Zero', got {mean}."
            )

        self.context_length = context_length
        self.horizons = horizons
        self.target_channels = target_channels
        self.stride = stride

        self.mean = mean
        self.dist = dist
        self.eps = eps
        self.maxiter = maxiter
        self.return_scale = return_scale

        self.train_split: SplitDict | None = None
        self.val_split: SplitDict | None = None

        self.fitted_models: dict[tuple[int, int], Any] = {}
        self.fitted_params: dict[tuple[int, int], dict[str, float]] = {}
        self.fallback_means: dict[tuple[int, int], float] = {}
        self.fallback_variances: dict[tuple[int, int], float] = {}
        self.failed_models: dict[tuple[int, int], float] = {}        
        self.fallback_means: dict[tuple[int, int], float] = {}
        self.fallback_variances: dict[tuple[int, int], float] = {}
        self.failed_models: dict[tuple[int, int], str] = {}
        self.convergence_flags: dict[tuple[int, int], int] = {}

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        mean: str = "AR",
        dist: str = "normal",
        eps: float = 1e-8,
        maxiter: int = 1000,
        return_scale: int = 10000.0
    ) -> "GarchBaseline":
        forecasting_config = config["forecasting"]

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=list(forecasting_config["horizons"]),
            target_channels=list(forecasting_config["target_channels"]),
            stride=int(forecasting_config["stride"]),
            mean=mean,
            dist=dist,
            eps=eps,
            maxiter=maxiter,
            return_scale = return_scale
        )

    def fit(
        self,
        train_split: SplitDict,
        val_split: SplitDict | None = None,
        verbose: bool = True,
    ) -> "GarchBaseline":
        """
        Fit one GARCH(1,1) model per asset/channel.
        """
        self.train_split = train_split
        self.val_split = val_split

        self.fitted_models = {}
        self.fitted_params = {}
        self.fallback_means = {}
        self.fallback_variances = {}
        self.failed_models = {}
        self.convergence_flags = {}

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
                f"Fitting {total_models} GARCH(1,1) models "
                f"with mean='{self.mean}'..."
            )

        model_count = 0

        for asset_idx in range(num_assets):
            for channel_idx in range(num_channels):
                key = (asset_idx, channel_idx)

                series = train_returns[:, asset_idx, channel_idx]
                series_np = series.detach().cpu().numpy().astype(np.float64)

                (
                    model_result,
                    params,
                    fallback_mean,
                    fallback_variance,
                    convergence_flag,
                    error_message,
                ) = self._fit_one_series(series_np)

                self.fitted_models[key] = model_result
                self.fitted_params[key] = params
                self.fallback_means[key] = fallback_mean
                self.fallback_variances[key] = fallback_variance
                self.convergence_flags[key] = convergence_flag

                if error_message is not None:
                    self.failed_models[key] = error_message

                model_count += 1

                if verbose and model_count % 25 == 0:
                    print(f"  fitted {model_count}/{total_models}")

        if verbose:
            print("Finished fitting GARCH models.")
            print(f"Failed models: {len(self.failed_models)}")

        return self

    def fitted_values(
        self,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Return GARCH predictions and true values on the training split.
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
        Generate GARCH predictions.
        """
        if len(self.fitted_params) == 0:
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
        all_y_variance = []
        all_sample_idx = []
        all_origin_idx = []
        all_target_indices = []

        for batch in loader:
            x_context = batch["x"].float()
            y_true_raw = batch["y"].float()
            last_context_target = batch["last_context_target"].float()

            context_returns = self._context_to_one_step_returns(x_context)

            one_step_mean, one_step_variance = self._forecast_batch(
                context_returns=context_returns,
                steps=max_horizon,
            )

            y_pred_cumulative = one_step_returns_to_cumulative_horizons(
                one_step_returns=one_step_mean,
                horizons=self.horizons,
            )

            y_variance_cumulative = self._one_step_variance_to_cumulative_horizons(
                one_step_variance=one_step_variance,
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
            all_y_variance.append(y_variance_cumulative)
            all_sample_idx.append(batch["sample_idx"])
            all_origin_idx.append(batch["origin_idx"])
            all_target_indices.append(batch["target_indices"])

        y_pred = torch.cat(all_y_pred, dim=0)
        y_true = torch.cat(all_y_true, dim=0)
        y_variance = torch.cat(all_y_variance, dim=0)

        sample_idx = torch.cat(all_sample_idx, dim=0)
        origin_idx = torch.cat(all_origin_idx, dim=0)
        target_indices = torch.cat(all_target_indices, dim=0)

        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "y_variance": y_variance,
            "variance_output_space": "cumulative_log_change",
            "output_space": output_space,
            "channels": self.target_channels,
            "horizons": self.horizons,
            "asset_cols": split["asset_cols"],
            "sample_idx": sample_idx,
            "origin_idx": origin_idx,
            "target_indices": target_indices,
            "mean": self.mean,
            "dist": self.dist,
            "failed_models": self.failed_models,
            "convergence_flags": self.convergence_flags,
        }

    def _fit_one_series(
        self,
        series: np.ndarray,
    ) -> tuple[Any | None, dict[str, float], float, float, int, str | None]:
        """
        Fit one GARCH(1,1) model to one asset/channel return series.
        """
        finite_mask = np.isfinite(series)
        finite_series = series[finite_mask]

        if finite_series.size == 0:
            return None, {}, 0.0, self.eps, -1, "Series contains no finite values."

        fallback_mean = float(np.mean(finite_series))
        fallback_variance = float(np.var(finite_series))

        fallback_variance = max(fallback_variance, self.eps)

        if finite_series.size < 20:
            return (
                None,
                {},
                fallback_mean,
                fallback_variance,
                -1,
                "Series is too short.",
            )

        if np.std(finite_series) < 1e-12:
            return (
                None,
                {},
                fallback_mean,
                fallback_variance,
                -1,
                "Series is approximately constant.",
            )

        try:
            
            scaled_series = finite_series * self.return_scale
            
            if self.mean == "AR":
                model = arch_model(
                    scaled_series,
                    mean=self.mean,
                    lags=1,
                    vol="GARCH",
                    p=1,
                    o=0,
                    q=1,
                    dist=self.dist,
                    rescale=False,
                )
            else:
                model = arch_model(
                    scaled_series,
                    mean=self.mean,
                    vol="GARCH",
                    p=1,
                    o=0,
                    q=1,
                    dist=self.dist,
                    rescale=False,
                )

            model_result = model.fit(
                disp="off",
                update_freq=0,
                show_warning=False,
                options={
                    "maxiter": self.maxiter,
                },
            )

            params = {
                str(name): float(value)
                for name, value in model_result.params.items()
            }

            convergence_flag = int(
                getattr(
                    model_result,
                    "convergence_flag",
                    0,
                )
            )

            return (
                model_result,
                params,
                fallback_mean,
                fallback_variance,
                convergence_flag,
                None,
            )

        except Exception as exc:
            return None, {}, fallback_mean, fallback_variance, -1, str(exc)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forecast future one-step means and variances for a batch.

        Args:
            context_returns:
                Tensor with shape [B, context_length - 1, N, C].

            steps:
                Number of future one-step returns to forecast.

        Returns:
            one_step_mean:
                Tensor with shape [B, steps, N, C].

            one_step_variance:
                Tensor with shape [B, steps, N, C].
        """
        batch_size = context_returns.shape[0]
        num_assets = context_returns.shape[2]
        num_channels = context_returns.shape[3]

        one_step_mean = torch.empty(
            batch_size,
            steps,
            num_assets,
            num_channels,
            dtype=torch.float32,
        )

        one_step_variance = torch.empty(
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

                    mean_np, variance_np = self._forecast_one_series(
                        asset_idx=asset_idx,
                        channel_idx=channel_idx,
                        context_series=context_series,
                        steps=steps,
                    )

                    one_step_mean[
                        batch_idx,
                        :,
                        asset_idx,
                        channel_idx,
                    ] = torch.from_numpy(mean_np).float()

                    one_step_variance[
                        batch_idx,
                        :,
                        asset_idx,
                        channel_idx,
                    ] = torch.from_numpy(variance_np).float()

        return one_step_mean, one_step_variance

    def _forecast_one_series(
        self,
        asset_idx: int,
        channel_idx: int,
        context_series: torch.Tensor,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast one asset/channel from one context window.
        """
        key = (asset_idx, channel_idx)

        params = self.fitted_params.get(key, {})
        fallback_mean = self.fallback_means.get(key, 0.0)
        fallback_variance = self.fallback_variances.get(key, self.eps)

        if len(params) == 0:
            return self._fallback_forecast(
                steps=steps,
                fallback_mean=fallback_mean,
                fallback_variance=fallback_variance,
            )

        if self.mean == "AR":
            mean_intercept = float(params.get("Const", params.get("mu", 0.0))) 
            ar_coefficient = float(params.get("y[1]", 0.0))

        elif self.mean == "Constant":
            mean_intercept = float(params.get("mu", params.get("Const", 0.0)))
            ar_coefficient = 0.0

        elif self.mean == "Zero":
            mean_intercept = 0.0
            ar_coefficient = 0.0

        else:
            raise ValueError(
                f"Unknown mean model: {self.mean}. "
                "Expected one of: Zero, Constant, AR."
            )

        omega = params.get("omega", fallback_variance)
        alpha = params.get("alpha[1]", 0.0)
        beta = params.get("beta[1]", 0.0)

        omega = max(float(omega), self.eps)
        alpha = max(float(alpha), 0.0)
        beta = max(float(beta), 0.0)

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

            if context_np.size == 0:
                return self._fallback_forecast(
                    steps=steps,
                    fallback_mean=fallback_mean,
                    fallback_variance=fallback_variance,
                )
            
            context_scaled = context_np * self.return_scale

            if self.mean == "AR":
                if context_scaled.shape[0] < 2:
                    residuals = np.array([], dtype=np.float64)
                else:
                    previous_returns = context_scaled[:-1]
                    current_returns = context_scaled[1:]

                    fitted_means = (
                        mean_intercept
                        + ar_coefficient * previous_returns
                    )

                    residuals = current_returns - fitted_means

            else:
                residuals = context_scaled - mean_intercept

            alpha_beta = alpha + beta

            if alpha_beta < 0.999:
                sigma2 = omega / max(1.0 - alpha_beta, self.eps)
            else:
                sigma2 = fallback_variance

            sigma2 = max(float(sigma2), self.eps)

            for residual in residuals:
                sigma2 = omega + alpha * residual**2 + beta * sigma2
                sigma2 = max(float(sigma2), self.eps)

            mean_forecast = np.empty(
                shape=steps,
                dtype=np.float64,
            )

            if self.mean == "AR":
                previous_value = context_scaled[-1]

                for step_idx in range(steps):
                    next_mean = (
                        mean_intercept
                        + ar_coefficient * previous_value
                    )

                    mean_forecast[step_idx] = next_mean
                    previous_value = next_mean

            else:
                mean_forecast.fill(mean_intercept)

            variance_forecast = np.empty(
                shape=steps,
                dtype=np.float64,
            )

            variance_forecast[0] = sigma2

            for step_idx in range(1, steps):
                variance_forecast[step_idx] = (
                    omega
                    + alpha_beta * variance_forecast[step_idx - 1]
                )

                variance_forecast[step_idx] = max(
                    float(variance_forecast[step_idx]),
                    self.eps,
                )
            
            mean_forecast = mean_forecast / self.return_scale
            variance_forecast = variance_forecast / (self.return_scale ** 2)

            if not np.isfinite(mean_forecast).all():
                return self._fallback_forecast(
                    steps=steps,
                    fallback_mean=fallback_mean,
                    fallback_variance=fallback_variance,
                )

            if not np.isfinite(variance_forecast).all():
                return self._fallback_forecast(
                    steps=steps,
                    fallback_mean=fallback_mean,
                    fallback_variance=fallback_variance,
                )

            if np.abs(mean_forecast).max() > 0.1:
                return self._fallback_forecast(
                    steps=steps,
                    fallback_mean=fallback_mean,
                    fallback_variance=fallback_variance,
                )

            if variance_forecast.max() > 1.0:
                return self._fallback_forecast(
                    steps=steps,
                    fallback_mean=fallback_mean,
                    fallback_variance=fallback_variance,
                )

            return mean_forecast, variance_forecast

        except Exception:
            return self._fallback_forecast(
                steps=steps,
                fallback_mean=fallback_mean,
                fallback_variance=fallback_variance,
            )

    def _fallback_forecast(
        self,
        steps: int,
        fallback_mean: float,
        fallback_variance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fallback mean/variance forecast.
        """
        mean_forecast = np.full(
            shape=steps,
            fill_value=fallback_mean,
            dtype=np.float64,
        )

        variance_forecast = np.full(
            shape=steps,
            fill_value=max(fallback_variance, self.eps),
            dtype=np.float64,
        )

        return mean_forecast, variance_forecast

    def _one_step_variance_to_cumulative_horizons(
        self,
        one_step_variance: torch.Tensor,
        horizons: list[int],
    ) -> torch.Tensor:
        """
        Convert future one-step variance forecasts into cumulative variance
        forecasts at selected horizons.

        This assumes future one-step return innovations are conditionally
        uncorrelated, so cumulative variance is the sum of one-step variances.
        """
        if one_step_variance.ndim not in {3, 4}:
            raise ValueError(
                "Expected one_step_variance to have shape [max_horizon, N, C] "
                f"or [B, max_horizon, N, C], got {tuple(one_step_variance.shape)}."
            )

        if len(horizons) == 0:
            raise ValueError("horizons must contain at least one value.")

        if min(horizons) < 1:
            raise ValueError(f"All horizons must be >= 1, got {horizons}.")

        max_horizon = one_step_variance.shape[-3]

        if max(horizons) > max_horizon:
            raise ValueError(
                f"Maximum requested horizon is {max(horizons)}, but "
                f"one_step_variance only contains {max_horizon} future steps."
            )

        cumulative_path = one_step_variance.cumsum(dim=-3)

        horizon_indices = torch.tensor(
            [horizon - 1 for horizon in horizons],
            dtype=torch.long,
            device=one_step_variance.device,
        )

        cumulative_horizons = cumulative_path.index_select(
            dim=-3,
            index=horizon_indices,
        )

        return cumulative_horizons

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
    


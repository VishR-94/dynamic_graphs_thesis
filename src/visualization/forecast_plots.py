from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import torch


DEFAULT_MODEL_DISPLAY_NAMES = {
    "persistence": "Persistence",
    "mean": "Mean",
    "arima": "ARIMA",
    "var": "VAR",
    "garch": "GARCH",
    "modern_tcn": "ModernTCN",
    "kronos": "Kronos",
}


def _parse_grain(
    grain: str,
) -> pd.Timedelta:
    grain = str(grain).strip().lower()

    if grain.endswith("min"):
        return pd.Timedelta(
            minutes=int(grain[:-3])
        )

    if grain.endswith("m"):
        return pd.Timedelta(
            minutes=int(grain[:-1])
        )

    if grain.endswith("h"):
        return pd.Timedelta(
            hours=int(grain[:-1])
        )

    raise ValueError(
        f"Unsupported grain: {grain!r}."
    )


def _resolve_day_timestamps(
    *,
    sample_timestamps: Any,
    day: Any,
    num_bars: int,
    test_split: Mapping[str, Any],
) -> pd.DatetimeIndex:
    """
    Recover the bar-close timestamps for one test session.

    Stored sample timestamps are preferred. If they are unavailable,
    timestamps are reconstructed from the split metadata.
    """
    try:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(sample_timestamps)
        )

        if (
            len(timestamps) == num_bars
            and not timestamps.isna().any()
        ):
            return timestamps

    except (TypeError, ValueError):
        pass

    bar_interval = _parse_grain(
        test_split["grain"]
    )

    first_bar_close = (
        pd.Timestamp(
            f"{pd.Timestamp(day).date()} "
            f"{test_split['market_open']}"
        )
        + bar_interval
    )

    return pd.date_range(
        start=first_bar_close,
        periods=num_bars,
        freq=bar_interval,
    )


def _normalise_model_names(
    models: str | Sequence[str],
) -> list[str]:
    if isinstance(models, str):
        model_names = [models]
    else:
        model_names = list(models)

    if len(model_names) == 0:
        raise ValueError(
            "At least one model must be supplied."
        )

    normalised_names = []

    for model_name in model_names:
        if not isinstance(model_name, str):
            raise TypeError(
                "Every model name must be a string."
            )

        model_name = model_name.strip()

        if model_name == "":
            raise ValueError(
                "Model names cannot be empty."
            )

        if model_name.endswith("_result"):
            model_name = model_name[:-7]

        normalised_names.append(model_name)

    if len(set(normalised_names)) != len(normalised_names):
        raise ValueError(
            "Each model should be included only once."
        )

    return normalised_names


def _resolve_prediction_results(
    *,
    model_names: Sequence[str],
    namespace: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prediction_results = {}

    missing_variables = []

    for model_name in model_names:
        variable_name = f"{model_name}_result"

        if variable_name not in namespace:
            missing_variables.append(
                variable_name
            )
            continue

        result = namespace[variable_name]

        if not isinstance(result, dict):
            raise TypeError(
                f"{variable_name} must be a prediction-result "
                f"dictionary, got {type(result)}."
            )

        prediction_results[model_name] = result

    if missing_variables:
        raise NameError(
            "Missing notebook prediction variables: "
            + ", ".join(missing_variables)
        )

    return prediction_results


def _resolve_sample_index(
    *,
    day: str | pd.Timestamp | int | None,
    available_sample_indices: Sequence[int],
    test_split: Mapping[str, Any],
    rng: random.Random,
) -> int:
    if day is None:
        return rng.choice(
            list(available_sample_indices)
        )

    if isinstance(day, Integral):
        sample_idx = int(day)

        if sample_idx not in available_sample_indices:
            raise ValueError(
                f"No forecasts exist for test sample {sample_idx}."
            )

        return sample_idx

    requested_day = pd.Timestamp(
        day
    ).normalize()

    matching_indices = [
        sample_idx
        for sample_idx in available_sample_indices
        if pd.Timestamp(
            test_split["samples"][sample_idx][2]
        ).normalize() == requested_day
    ]

    if len(matching_indices) == 0:
        raise ValueError(
            f"No forecast session was found for "
            f"{requested_day.date()}."
        )

    if len(matching_indices) > 1:
        raise ValueError(
            f"Multiple test samples were found for "
            f"{requested_day.date()}."
        )

    return matching_indices[0]


def plot_forecast_comparison(
    models: str | Sequence[str],
    namespace: Mapping[str, Any],
    horizons: Sequence[int] | None = None,
    day: str | pd.Timestamp | int | None = None,
    asset: str | None = None,
    random_seed: int | None = None,
) -> tuple[
    plt.Figure,
    list[plt.Axes],
    dict[str, Any],
]:
    """
    Compare model forecasts for one asset and one test session.

    Prediction dictionaries are resolved from the supplied notebook
    namespace using the convention:

        {model_name}_result

    The test split is resolved from:

        test

    Args:
        models:
            One model identifier or a sequence of model identifiers.

            Examples:
                "kronos"
                ["persistence", "modern_tcn", "kronos"]

        namespace:
            Notebook namespace, normally passed as globals().

        horizons:
            Forecast horizons to display. If None, all horizons from
            the prediction results are plotted.

        day:
            Test date to display, such as "2024-10-24".

            An integer may also be supplied to select a test
            sample_idx directly.

            If None, a reproducible random session is selected.

        asset:
            Asset ticker to display. If None, a reproducible random
            asset is selected.

        random_seed:
            Seed used for random day and asset selection.

    Returns:
        fig:
            Matplotlib figure.

        axes:
            One axis per requested model.

        selection:
            Dictionary containing the selected asset, date,
            sample_idx and horizons.
    """
    model_names = _normalise_model_names(
        models
    )

    prediction_results = (
        _resolve_prediction_results(
            model_names=model_names,
            namespace=namespace,
        )
    )

    if "test" not in namespace:
        raise NameError(
            "The notebook namespace does not contain `test`."
        )

    test_split = namespace["test"]

    if not isinstance(test_split, Mapping):
        raise TypeError(
            "`test` must be a split dictionary."
        )

    test_assets = list(
        test_split["asset_cols"]
    )

    test_channels = list(
        test_split["channels"]
    )

    if "close" not in test_channels:
        raise ValueError(
            "The test split does not contain a close channel."
        )

    test_close_idx = test_channels.index(
        "close"
    )

    reference_name = model_names[0]

    reference_result = prediction_results[
        reference_name
    ]

    reference_y_pred = torch.as_tensor(
        reference_result["y_pred"]
    ).detach().cpu()

    reference_horizons = [
        int(value)
        for value in reference_result["horizons"]
    ]

    reference_sample_idx = torch.as_tensor(
        reference_result["sample_idx"]
    ).detach().cpu().long()

    reference_origin_idx = torch.as_tensor(
        reference_result["origin_idx"]
    ).detach().cpu().long()

    reference_target_indices = torch.as_tensor(
        reference_result["target_indices"]
    ).detach().cpu().long()

    if reference_y_pred.ndim != 4:
        raise ValueError(
            f"{reference_name}_result['y_pred'] must have shape "
            f"[B, H, N, C], got "
            f"{tuple(reference_y_pred.shape)}."
        )

    if reference_y_pred.shape[2] != len(test_assets):
        raise ValueError(
            f"{reference_name} contains "
            f"{reference_y_pred.shape[2]} assets, but the test "
            f"split contains {len(test_assets)}."
        )

    if reference_target_indices.shape != (
        reference_y_pred.shape[0],
        reference_y_pred.shape[1],
    ):
        raise ValueError(
            "target_indices is not aligned with the prediction "
            "batch and horizon dimensions."
        )

    reference_asset_cols = reference_result.get(
        "asset_cols"
    )

    if (
        reference_asset_cols is not None
        and list(reference_asset_cols) != test_assets
    ):
        raise ValueError(
            f"{reference_name} asset ordering is not aligned "
            "with the test split."
        )

    for model_name, result in prediction_results.items():
        y_pred = torch.as_tensor(
            result["y_pred"]
        )

        if y_pred.ndim != 4:
            raise ValueError(
                f"{model_name}_result['y_pred'] must have shape "
                f"[B, H, N, C], got {tuple(y_pred.shape)}."
            )

        if y_pred.shape[:3] != reference_y_pred.shape[:3]:
            raise ValueError(
                f"{model_name} prediction dimensions are not "
                f"aligned with {reference_name}."
            )

        result_asset_cols = result.get(
            "asset_cols"
        )

        if (
            result_asset_cols is not None
            and list(result_asset_cols) != test_assets
        ):
            raise ValueError(
                f"{model_name} asset ordering is not aligned "
                "with the test split."
            )

        result_horizons = [
            int(value)
            for value in result["horizons"]
        ]

        if result_horizons != reference_horizons:
            raise ValueError(
                f"{model_name} horizons are not aligned with "
                f"{reference_name}."
            )

        result_sample_idx = torch.as_tensor(
            result["sample_idx"]
        ).detach().cpu().long()

        result_origin_idx = torch.as_tensor(
            result["origin_idx"]
        ).detach().cpu().long()

        result_target_indices = torch.as_tensor(
            result["target_indices"]
        ).detach().cpu().long()

        if not torch.equal(
            result_sample_idx,
            reference_sample_idx,
        ):
            raise ValueError(
                f"{model_name} sample_idx is not aligned with "
                f"{reference_name}."
            )

        if not torch.equal(
            result_origin_idx,
            reference_origin_idx,
        ):
            raise ValueError(
                f"{model_name} origin_idx is not aligned with "
                f"{reference_name}."
            )

        if not torch.equal(
            result_target_indices,
            reference_target_indices,
        ):
            raise ValueError(
                f"{model_name} target_indices are not aligned "
                f"with {reference_name}."
            )

    if horizons is None:
        selected_horizons = list(
            reference_horizons
        )
    else:
        selected_horizons = [
            int(value)
            for value in horizons
        ]

        if len(selected_horizons) == 0:
            raise ValueError(
                "At least one horizon must be selected."
            )

        if len(set(selected_horizons)) != len(
            selected_horizons
        ):
            raise ValueError(
                "Each horizon should be selected only once."
            )

        missing_horizons = [
            horizon
            for horizon in selected_horizons
            if horizon not in reference_horizons
        ]

        if missing_horizons:
            raise ValueError(
                f"Unknown horizons: {missing_horizons}. "
                f"Available horizons: {reference_horizons}."
            )

    horizon_positions = {
        horizon: reference_horizons.index(
            horizon
        )
        for horizon in selected_horizons
    }

    rng = random.Random(
        random_seed
    )

    available_sample_indices = [
        int(value)
        for value in torch.unique(
            reference_sample_idx,
            sorted=True,
        ).tolist()
    ]

    selected_sample_idx = _resolve_sample_index(
        day=day,
        available_sample_indices=available_sample_indices,
        test_split=test_split,
        rng=rng,
    )

    if asset is None:
        selected_asset_idx = rng.randrange(
            len(test_assets)
        )
    else:
        if asset not in test_assets:
            raise ValueError(
                f"Unknown asset {asset!r}."
            )

        selected_asset_idx = test_assets.index(
            asset
        )

    selected_asset = test_assets[
        selected_asset_idx
    ]

    x_day, sample_timestamps, sample_day = (
        test_split["samples"][
            selected_sample_idx
        ]
    )

    x_day = torch.as_tensor(
        x_day
    )

    if x_day.ndim != 3:
        raise ValueError(
            "Expected a test sample with shape [T, N, D], got "
            f"{tuple(x_day.shape)}."
        )

    day_timestamps = _resolve_day_timestamps(
        sample_timestamps=sample_timestamps,
        day=sample_day,
        num_bars=x_day.shape[0],
        test_split=test_split,
    )

    true_close = (
        x_day[
            :,
            selected_asset_idx,
            test_close_idx,
        ]
        .detach()
        .cpu()
        .to(torch.float64)
        .numpy()
    )

    selected_rows = torch.where(
        reference_sample_idx
        == selected_sample_idx
    )[0]

    selected_rows = selected_rows[
        torch.argsort(
            reference_origin_idx[
                selected_rows
            ]
        )
    ]

    num_models = len(
        model_names
    )

    fig, axes_array = plt.subplots(
        nrows=num_models,
        ncols=1,
        figsize=(
            16,
            5 * num_models,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes = list(
        axes_array[:, 0]
    )

    colour_map = plt.get_cmap(
        "tab10"
    )

    horizon_colours = {
        horizon: colour_map(
            horizon_idx
        )
        for horizon_idx, horizon in enumerate(
            selected_horizons
        )
    }

    for axis, model_name in zip(
        axes,
        model_names,
    ):
        result = prediction_results[
            model_name
        ]

        prediction_channels = list(
            result["channels"]
        )

        if "close" not in prediction_channels:
            raise ValueError(
                f"{model_name} does not contain a close channel."
            )

        prediction_close_idx = (
            prediction_channels.index(
                "close"
            )
        )

        y_pred = torch.as_tensor(
            result["y_pred"]
        ).detach().cpu()

        target_indices = torch.as_tensor(
            result["target_indices"]
        ).detach().cpu().long()

        axis.plot(
            day_timestamps,
            true_close,
            linewidth=1.5,
            color="black",
            label="True close",
            zorder=1,
        )

        for horizon in selected_horizons:
            horizon_position = horizon_positions[
                horizon
            ]

            horizon_target_indices = (
                target_indices[
                    selected_rows,
                    horizon_position,
                ]
            )

            valid = (
                horizon_target_indices >= 0
            ) & (
                horizon_target_indices
                < len(day_timestamps)
            )

            valid_rows = selected_rows[
                valid
            ]

            valid_target_indices = (
                horizon_target_indices[
                    valid
                ]
            )

            forecast_timestamps = (
                day_timestamps[
                    valid_target_indices.numpy()
                ]
            )

            forecast_prices = (
                y_pred[
                    valid_rows,
                    horizon_position,
                    selected_asset_idx,
                    prediction_close_idx,
                ]
                .to(torch.float64)
                .numpy()
            )

            axis.scatter(
                forecast_timestamps,
                forecast_prices,
                s=38,
                alpha=0.8,
                color=horizon_colours[
                    horizon
                ],
                label=f"{horizon}-min forecast",
                zorder=2,
            )

        display_name = (
            DEFAULT_MODEL_DISPLAY_NAMES.get(
                model_name,
                model_name.replace(
                    "_",
                    " ",
                ).title(),
            )
        )

        axis.set_title(
            display_name
        )

        axis.set_ylabel(
            "Close price"
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend(
            ncols=min(
                3,
                len(selected_horizons) + 1,
            ),
            loc="upper right",
        )

    axes[-1].set_xlabel(
        "Target bar-close time"
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )

    day_label = pd.Timestamp(
        sample_day
    ).strftime("%Y-%m-%d")

    fig.suptitle(
        (
            f"Forecast comparison: "
            f"{selected_asset} on {day_label}"
        ),
        fontsize=15,
        y=1.01,
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    selection = {
        "asset": selected_asset,
        "asset_idx": selected_asset_idx,
        "day": day_label,
        "sample_idx": selected_sample_idx,
        "horizons": selected_horizons,
        "models": list(model_names),
        "num_forecast_windows": int(
            selected_rows.numel()
        ),
    }

    return (
        fig,
        axes,
        selection,
    )
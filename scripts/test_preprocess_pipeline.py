from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.load_candle_data import (
    clean_candle_splits,
    load_candle_splits,
    get_channel_index,
)
from src.data.data_generator import (
    WindowContextNormaliser,
    WindowedCandleDataset,
    build_log_change_split,
    build_valid_transformed_split,
)
from src.evaluation.metrics import mae, mse, rmse
from src.evaluation.prediction_transforms import (
    cumulative_log_change_to_raw,
    inverse_window_normalisation,
    one_step_returns_to_cumulative_horizons,
    raw_to_cumulative_log_change,
    valid_transformed_to_raw_ohlcv,
)
from src.utils.config import load_yaml

'''
 Script to sanity check the entire pipeline before we start running benchmarks.
 This script will test our ability to:
   1. Load and clean the data
   2. Transform the data (log change or transformation for consistent raw price predictions)
   3. Window normalise the data
   4. Generate samples of correct dimensions to feed into BatchLoader
   6. Invert the window normalisation
   7. Test functionality of the metric functions and ensure the reductions work correctly 

 Once we are happy all parts of the pipeline work, we can then start running baselines
'''

DATA_DIR = Path(
    "/Users/vishalruparelia/Library/CloudStorage/"
    "GoogleDrive-vishal@autonomous-fox.ai/"
    "Shared drives/Vishal/data/cached_datasets/"
    "exp-24-a95-Candle/session"
)

CONFIG_PATH = Path("configs/forecasting.yaml")


VALID_TRANSFORMED_CHANNELS = [
    "log_close",
    "log_open_to_close",
    "log_upper_wick_ratio",
    "log_lower_wick_ratio",
    "log_volume",
]

RAW_OUTPUT_CHANNELS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def check_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values.")

    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values.")


def print_check(message: str) -> None:
    print(f"✓ {message}")


def run_synthetic_tests() -> None:
    print("\nRunning synthetic tests...")

    batch_size = 4
    num_horizons = 5
    num_assets = 3
    num_channels = 2

    y_true = torch.zeros(batch_size, num_horizons, num_assets, num_channels)
    y_pred = torch.ones(batch_size, num_horizons, num_assets, num_channels)

    assert mae(y_pred, y_true).item() == 1.0
    assert mse(y_pred, y_true).item() == 1.0
    assert rmse(y_pred, y_true).item() == 1.0

    per_horizon = rmse(
        y_pred,
        y_true,
        reduce_dims=(0, 2, 3),
    )
    assert tuple(per_horizon.shape) == (num_horizons,)

    per_asset = rmse(
        y_pred,
        y_true,
        reduce_dims=(0, 1, 3),
    )
    assert tuple(per_asset.shape) == (num_assets,)

    per_channel = rmse(
        y_pred,
        y_true,
        reduce_dims=(0, 1, 2),
    )
    assert tuple(per_channel.shape) == (num_channels,)

    print_check("Metric functions and reductions work on synthetic tensors.")

    last = torch.rand(batch_size, num_assets, num_channels) * 100.0 + 50.0
    y_raw = torch.rand(batch_size, num_horizons, num_assets, num_channels) * 100.0 + 50.0

    log_change = raw_to_cumulative_log_change(
        y_raw=y_raw,
        last_context_target=last,
    )

    y_recovered = cumulative_log_change_to_raw(
        cumulative_log_change=log_change,
        last_context_target=last,
    )

    if not torch.allclose(y_recovered, y_raw, atol=1e-5, rtol=1e-5):
        max_diff = (y_recovered - y_raw).abs().max().item()
        raise ValueError(f"Raw/log-change round trip failed. Max diff: {max_diff}")

    print_check("Raw ↔ cumulative log-change round trip works.")

    one_step_returns = torch.ones(batch_size, 6, num_assets, 1)
    horizons = [1, 3, 6]

    cumulative = one_step_returns_to_cumulative_horizons(
        one_step_returns=one_step_returns,
        horizons=horizons,
    )

    expected = torch.tensor([1.0, 3.0, 6.0])

    if not torch.allclose(cumulative[0, :, 0, 0], expected):
        raise ValueError("One-step return cumulative horizon test failed.")

    assert tuple(cumulative.shape) == (batch_size, len(horizons), num_assets, 1)

    print_check("One-step returns → cumulative horizons works.")


def run_real_data_tests() -> None:
    print("\nRunning real-data pipeline tests...")

    config = load_yaml(CONFIG_PATH)

    transformed_config = deepcopy(config)
    transformed_config["forecasting"]["input_channels"] = VALID_TRANSFORMED_CHANNELS
    transformed_config["forecasting"]["target_channels"] = VALID_TRANSFORMED_CHANNELS
    transformed_config["normalisation"]["clip"] = False

    train_raw, val_raw, test_raw = load_candle_splits(DATA_DIR)

    train_clean, val_clean, test_clean = clean_candle_splits(
        train_raw,
        val_raw,
        test_raw,
    )

    print_check("Loaded and cleaned raw candle splits.")

    train_valid = build_valid_transformed_split(train_clean)
    test_valid = build_valid_transformed_split(test_clean)

    x_valid, _, _ = train_valid["samples"][0]

    check_finite("x_valid", x_valid)

    assert tuple(x_valid.shape) == (390, 93, 5)
    assert train_valid["channels"] == VALID_TRANSFORMED_CHANNELS

    print_check("Built valid-candle transformed split.")

    train_log = build_log_change_split(train_clean)
    x_log, _, _ = train_log["samples"][0]

    check_finite("x_log", x_log)

    assert tuple(x_log.shape) == (389, 93, 6)
    assert train_log["representation"] == "log_change"

    print_check("Built log-change split.")

    normaliser = WindowContextNormaliser.from_config(transformed_config)

    test_dataset = WindowedCandleDataset.from_config(
        split=test_valid,
        config=transformed_config,
        normaliser=normaliser,
    )

    example = test_dataset[0]

    expected_num_horizons = len(transformed_config["forecasting"]["horizons"])

    assert tuple(example["x"].shape) == (60, 93, 5)
    assert tuple(example["y"].shape) == (expected_num_horizons, 93, 5)
    assert tuple(example["last_context_target"].shape) == (93, 5)
    assert tuple(example["target_norm_mean"].shape) == (93, 5)
    assert tuple(example["target_norm_std"].shape) == (93, 5)

    if "y_unnormalised" not in example:
        raise KeyError(
            "Expected example['y_unnormalised'] to exist. "
            "Add 'y_unnormalised': y to WindowedCandleDataset before normalisation."
        )

    assert tuple(example["y_unnormalised"].shape) == (
        expected_num_horizons,
        93,
        5,
    )

    print_check("WindowedCandleDataset returns expected single-example shapes.")

    loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert tuple(batch["x"].shape) == (4, 60, 93, 5)
    assert tuple(batch["y"].shape) == (4, expected_num_horizons, 93, 5)
    assert tuple(batch["y_unnormalised"].shape) == (
        4,
        expected_num_horizons,
        93,
        5,
    )
    assert tuple(batch["last_context_target"].shape) == (4, 93, 5)
    assert tuple(batch["target_norm_mean"].shape) == (4, 93, 5)
    assert tuple(batch["target_norm_std"].shape) == (4, 93, 5)

    print_check("DataLoader batches have expected shapes.")

    y_transformed_recovered = inverse_window_normalisation(
        y_norm=batch["y"],
        target_norm_mean=batch["target_norm_mean"],
        target_norm_std=batch["target_norm_std"],
    )

    check_finite("y_transformed_recovered", y_transformed_recovered)

    assert tuple(y_transformed_recovered.shape) == tuple(batch["y"].shape)

    if not torch.allclose(
        y_transformed_recovered,
        batch["y_unnormalised"],
        atol=1e-5,
        rtol=1e-5,
    ):
        max_diff = (
            y_transformed_recovered
            - batch["y_unnormalised"]
        ).abs().max().item()

        raise ValueError(
            "Inverse window normalisation did not recover y_unnormalised. "
            f"Max diff: {max_diff}"
        )

    print_check("Inverse window normalisation recovers y_unnormalised.")

    y_pred_raw = valid_transformed_to_raw_ohlcv(
        y_transformed=y_transformed_recovered,
        transformed_channels=VALID_TRANSFORMED_CHANNELS,
        output_channels=RAW_OUTPUT_CHANNELS,
    )

    y_true_raw = valid_transformed_to_raw_ohlcv(
        y_transformed=batch["y_unnormalised"],
        transformed_channels=VALID_TRANSFORMED_CHANNELS,
        output_channels=RAW_OUTPUT_CHANNELS,
    )

    last_context_raw = valid_transformed_to_raw_ohlcv(
        y_transformed=batch["last_context_target"],
        transformed_channels=VALID_TRANSFORMED_CHANNELS,
        output_channels=RAW_OUTPUT_CHANNELS,
    )

    assert tuple(y_pred_raw.shape) == (4, expected_num_horizons, 93, 5)
    assert tuple(y_true_raw.shape) == (4, expected_num_horizons, 93, 5)
    assert tuple(last_context_raw.shape) == (4, 93, 5)

    check_finite("y_pred_raw", y_pred_raw)
    check_finite("y_true_raw", y_true_raw)
    check_finite("last_context_raw", last_context_raw)

    raw_channel_indices = torch.tensor(
        [
            get_channel_index(test_clean, channel)
            for channel in RAW_OUTPUT_CHANNELS
        ],
        dtype=torch.long,
    )

    raw_targets = []

    for batch_idx in range(batch["y"].shape[0]):
        sample_idx = int(batch["sample_idx"][batch_idx].item())
        target_indices = batch["target_indices"][batch_idx]

        x_raw_day, _, _ = test_clean["samples"][sample_idx]

        y_raw_direct = x_raw_day.index_select(
            0,
            target_indices,
        )

        y_raw_direct = y_raw_direct.index_select(
            2,
            raw_channel_indices,
        ).float()

        raw_targets.append(y_raw_direct)

    y_raw_direct = torch.stack(raw_targets, dim=0)

    if not torch.allclose(
        y_true_raw,
        y_raw_direct,
        atol=1e-4,
        rtol=1e-4,
    ):
        max_diff = (y_true_raw - y_raw_direct).abs().max().item()

        raise ValueError(
            "Inverse valid-candle transform did not recover raw targets. "
            f"Max diff: {max_diff}"
        )

    print_check("Inverse valid-candle transform recovers raw OHLCV targets.")

    open_price = y_pred_raw[..., 0]
    high_price = y_pred_raw[..., 1]
    low_price = y_pred_raw[..., 2]
    close_price = y_pred_raw[..., 3]
    volume = y_pred_raw[..., 4]

    assert bool((high_price >= open_price).all())
    assert bool((high_price >= close_price).all())
    assert bool((low_price <= open_price).all())
    assert bool((low_price <= close_price).all())
    assert bool((volume > 0).all())

    print_check("Inverse valid-candle transform gives valid raw OHLCV candles.")

    y_pred_log_change = raw_to_cumulative_log_change(
        y_raw=y_pred_raw,
        last_context_target=last_context_raw,
    )

    y_true_log_change = raw_to_cumulative_log_change(
        y_raw=y_true_raw,
        last_context_target=last_context_raw,
    )

    assert tuple(y_pred_log_change.shape) == tuple(y_pred_raw.shape)
    assert tuple(y_true_log_change.shape) == tuple(y_true_raw.shape)

    check_finite("y_pred_log_change", y_pred_log_change)
    check_finite("y_true_log_change", y_true_log_change)

    print_check("Converted real batch to cumulative log-change space.")

    overall_rmse = rmse(
        y_pred=y_pred_log_change,
        y_true=y_true_log_change,
    )

    per_horizon_rmse = rmse(
        y_pred=y_pred_log_change,
        y_true=y_true_log_change,
        reduce_dims=(0, 2, 3),
    )

    per_channel_rmse = rmse(
        y_pred=y_pred_log_change,
        y_true=y_true_log_change,
        reduce_dims=(0, 1, 2),
    )

    per_horizon_channel_rmse = rmse(
        y_pred=y_pred_log_change,
        y_true=y_true_log_change,
        reduce_dims=(0, 2),
    )

    assert overall_rmse.ndim == 0
    assert tuple(per_horizon_rmse.shape) == (expected_num_horizons,)
    assert tuple(per_channel_rmse.shape) == (5,)
    assert tuple(per_horizon_channel_rmse.shape) == (
        expected_num_horizons,
        5,
    )

    check_finite("overall_rmse", overall_rmse)
    check_finite("per_horizon_rmse", per_horizon_rmse)
    check_finite("per_channel_rmse", per_channel_rmse)
    check_finite("per_horizon_channel_rmse", per_horizon_channel_rmse)

    print_check("Metrics work on real cumulative log-change tensors.")

    print("\nExample metric outputs:")
    print("overall RMSE:", overall_rmse.item())
    print("per-horizon RMSE shape:", tuple(per_horizon_rmse.shape))
    print("per-channel RMSE shape:", tuple(per_channel_rmse.shape))
    print("per-horizon-channel RMSE shape:", tuple(per_horizon_channel_rmse.shape))


def main() -> None:
    run_synthetic_tests()
    run_real_data_tests()

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
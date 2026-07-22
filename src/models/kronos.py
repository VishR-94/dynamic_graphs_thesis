from __future__ import annotations

import random
import sys
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.data_generator import WindowedCandleDataset


SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]

KRONOS_INPUT_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
)

KRONOS_OUTPUT_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_official_kronos() -> tuple[Any, Any, Any]:
    """
    Import the pinned official Kronos implementation lazily.

    Returns:
        Kronos,
        KronosPredictor,
        KronosTokenizer
    """
    kronos_root = _project_root() / "external" / "Kronos"

    if not kronos_root.is_dir():
        raise FileNotFoundError(
            "The official Kronos submodule was not found at "
            f"{kronos_root}."
        )

    kronos_root_string = str(kronos_root)

    if kronos_root_string not in sys.path:
        sys.path.insert(0, kronos_root_string)

    from model import Kronos, KronosPredictor, KronosTokenizer

    return Kronos, KronosPredictor, KronosTokenizer


def _identity_collate(
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return dataset examples unchanged for project-side batching."""
    return examples


class KronosBaseline:
    """
    Project wrapper for the frozen official Kronos zero-shot predictor.

    Kronos receives one independent OHLCV sequence per asset-window,
    internally appends a zero Amount channel, generates every future
    K-line autoregressively, and returns raw-space forecasts.

    The wrapper ultimately retains only Close at the project horizons.
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int,
        model_id: str,
        model_revision: str,
        tokenizer_id: str,
        tokenizer_revision: str,
        device: str = "auto",
        dtype: str = "float32",
        max_context: int = 512,
        clip: float = 5.0,
        temperature: float = 0.6,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 10,
        seed: int = 42,
        series_batch_size: int = 1,
        verbose: bool = False,
    ) -> None:
        self.context_length = int(context_length)
        self.horizons = [int(horizon) for horizon in horizons]
        self.target_channels = list(target_channels)
        self.stride = int(stride)

        self.input_channels = list(KRONOS_INPUT_CHANNELS)
        self.output_channels = list(KRONOS_OUTPUT_CHANNELS)

        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.tokenizer_id = str(tokenizer_id)
        self.tokenizer_revision = str(tokenizer_revision)

        self.requested_device = str(device)
        self.dtype = str(dtype)
        self.max_context = int(max_context)
        self.clip = float(clip)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.series_batch_size = int(series_batch_size)
        self.verbose = bool(verbose)

        self._validate_configuration()

        self.prediction_length = max(self.horizons)
        self.horizon_indices = [
            horizon - 1
            for horizon in self.horizons
        ]
        self.close_output_index = self.output_channels.index(
            "close"
        )

        model_name = self.model_id.rsplit("/", maxsplit=1)[-1]
        self.experiment_name = (
            f"{model_name.lower()}_zero_shot"
        )

        # Populated by fit(...).
        self.train_split: SplitDict | None = None
        self.val_split: SplitDict | None = None
        self.asset_cols: list[str] | None = None
        self.num_assets: int | None = None

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.predictor: Any | None = None
        self.device: str | None = None
        self.is_fitted = False

    def _validate_configuration(self) -> None:
        if self.context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if self.stride <= 0:
            raise ValueError(
                "stride must be greater than zero."
            )

        if not self.horizons:
            raise ValueError(
                "horizons must contain at least one value."
            )

        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError(
                "Every forecast horizon must be greater than zero."
            )

        if self.horizons != sorted(set(self.horizons)):
            raise ValueError(
                "horizons must be strictly increasing and unique. "
                f"Received {self.horizons}."
            )

        if self.target_channels != ["close"]:
            raise ValueError(
                "The frozen Kronos dissertation benchmark is "
                "close-only. Expected target_channels=['close'], "
                f"received {self.target_channels}."
            )

        if not self.model_id:
            raise ValueError(
                "model_id must not be empty."
            )

        if not self.model_revision:
            raise ValueError(
                "model_revision must not be empty."
            )

        if not self.tokenizer_id:
            raise ValueError(
                "tokenizer_id must not be empty."
            )

        if not self.tokenizer_revision:
            raise ValueError(
                "tokenizer_revision must not be empty."
            )

        valid_named_devices = {
            "auto",
            "cpu",
            "mps",
            "cuda",
        }
        is_numbered_cuda = self.requested_device.startswith(
            "cuda:"
        )

        if (
            self.requested_device not in valid_named_devices
            and not is_numbered_cuda
        ):
            raise ValueError(
                "device must be one of 'auto', 'cpu', 'mps', "
                "'cuda', or a numbered CUDA device such as "
                f"'cuda:0'. Received {self.requested_device!r}."
            )

        if self.dtype != "float32":
            raise ValueError(
                "Only dtype='float32' is currently supported, "
                "matching the official Kronos predictor."
            )

        if self.max_context <= 0:
            raise ValueError(
                "max_context must be greater than zero."
            )

        required_sequence_length = (
            self.context_length
            + max(self.horizons)
        )

        if required_sequence_length > self.max_context:
            raise ValueError(
                "Kronos max_context would truncate the requested "
                "context-plus-forecast path. Required at least "
                f"{required_sequence_length}, received "
                f"{self.max_context}."
            )

        if self.clip <= 0:
            raise ValueError(
                "clip must be greater than zero."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if self.top_k < 0:
            raise ValueError(
                "top_k must be non-negative."
            )

        if not 0 < self.top_p <= 1:
            raise ValueError(
                "top_p must lie in (0, 1]."
            )

        if self.sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )

        if self.series_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )


    def _set_seed(self) -> None:
        """Seed all random-number generators used during inference."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _resolve_device(self) -> str:
        """Resolve and validate the requested inference device."""
        if self.requested_device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"

            return "cpu"

        if self.requested_device in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested for Kronos, but CUDA is "
                    "not available."
                )
            return "cuda:0"

        if self.requested_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA device was requested for Kronos, but "
                    "CUDA is not available."
                )

            device_index = int(
                self.requested_device.split(":", maxsplit=1)[1]
            )
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    "Requested CUDA device does not exist. "
                    f"Received {self.requested_device!r}, but "
                    f"torch reports {torch.cuda.device_count()} "
                    "CUDA device(s)."
                )
            return self.requested_device

        if self.requested_device == "mps":
            if not (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                raise RuntimeError(
                    "MPS was requested for Kronos, but MPS is "
                    "not available."
                )
            return "mps"

        return "cpu"

    def _validate_split(
        self,
        split: SplitDict,
        split_name: str,
    ) -> None:
        """Validate the metadata required by the Kronos wrapper."""
        required_keys = {
            "asset_cols",
            "channels",
            "samples",
            "market_open",
            "market_close",
        }
        missing_keys = sorted(
            required_keys.difference(split)
        )

        if missing_keys:
            raise KeyError(
                f"{split_name} split is missing required keys: "
                f"{missing_keys}."
            )

        asset_cols = list(split["asset_cols"])
        channels = list(split["channels"])
        samples = list(split["samples"])

        if not asset_cols:
            raise ValueError(
                f"{split_name} split contains no assets."
            )

        if len(asset_cols) != len(set(asset_cols)):
            raise ValueError(
                f"{split_name} split contains duplicate assets."
            )

        if not samples:
            raise ValueError(
                f"{split_name} split contains no sessions."
            )

        missing_input_channels = [
            channel
            for channel in self.input_channels
            if channel not in channels
        ]
        missing_target_channels = [
            channel
            for channel in self.target_channels
            if channel not in channels
        ]

        if missing_input_channels:
            raise ValueError(
                f"{split_name} split is missing required Kronos "
                f"input channels: {missing_input_channels}."
            )

        if missing_target_channels:
            raise ValueError(
                f"{split_name} split is missing required target "
                f"channels: {missing_target_channels}."
            )

        expected_num_assets = len(asset_cols)
        expected_num_channels = len(channels)

        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, (tuple, list)):
                raise TypeError(
                    f"{split_name} sample {sample_index} must be "
                    "a tuple or list."
                )

            if len(sample) != 3:
                raise ValueError(
                    f"{split_name} sample {sample_index} must "
                    "contain (tensor, aux, day)."
                )

            x_day = sample[0]

            if not isinstance(x_day, torch.Tensor):
                raise TypeError(
                    f"{split_name} sample {sample_index} data "
                    "must be a torch.Tensor."
                )

            if x_day.ndim != 3:
                raise ValueError(
                    f"{split_name} sample {sample_index} must "
                    "have shape [T, N, C]. Received "
                    f"{tuple(x_day.shape)}."
                )

            if x_day.shape[1] != expected_num_assets:
                raise ValueError(
                    f"{split_name} sample {sample_index} asset "
                    "dimension does not match asset_cols."
                )

            if x_day.shape[2] != expected_num_channels:
                raise ValueError(
                    f"{split_name} sample {sample_index} channel "
                    "dimension does not match channels."
                )

            required_length = (
                self.context_length
                + self.prediction_length
            )
            if x_day.shape[0] < required_length:
                raise ValueError(
                    f"{split_name} sample {sample_index} is too "
                    "short for the requested context and forecast "
                    f"lengths. Required at least {required_length}, "
                    f"received {x_day.shape[0]}."
                )

    def _validate_checkpoint_pairing(self) -> None:
        """Validate released tokenizer/model dimensional compatibility."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Kronos model and tokenizer must be loaded before "
                "checkpoint compatibility is checked."
            )

        tokenizer_input_dim = int(
            getattr(self.tokenizer, "d_in", -1)
        )
        if tokenizer_input_dim != len(self.output_channels):
            raise ValueError(
                "The loaded Kronos tokenizer does not expect the "
                "six-channel OHLCVA representation. Expected "
                f"d_in={len(self.output_channels)}, received "
                f"{tokenizer_input_dim}."
            )

        for attribute in ("s1_bits", "s2_bits"):
            tokenizer_value = int(
                getattr(self.tokenizer, attribute, -1)
            )
            model_value = int(
                getattr(self.model, attribute, -1)
            )

            if tokenizer_value != model_value:
                raise ValueError(
                    "Kronos tokenizer/model checkpoint mismatch "
                    f"for {attribute}: tokenizer={tokenizer_value}, "
                    f"model={model_value}."
                )

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> "KronosBaseline":
        """
        Construct Kronos from the project configuration.

        Global forecasting values come from config["forecasting"].
        Kronos-specific checkpoint and inference settings come from
        config["models"]["kronos"].
        """
        forecasting_config = config["forecasting"]

        kronos_config = (
            config.get("models", {})
            .get("kronos", {})
        )

        if not kronos_config:
            raise KeyError(
                "Missing required configuration section: "
                "models.kronos."
            )

        inference_config = kronos_config.get(
            "inference",
            {},
        )

        if not isinstance(inference_config, dict):
            raise TypeError(
                "models.kronos.inference must be a mapping."
            )

        return cls(
            context_length=int(
                forecasting_config["context_length"]
            ),
            horizons=[
                int(horizon)
                for horizon in forecasting_config["horizons"]
            ],
            target_channels=list(
                forecasting_config["target_channels"]
            ),
            stride=int(
                forecasting_config["stride"]
            ),
            model_id=str(
                kronos_config["model_id"]
            ),
            model_revision=str(
                kronos_config["model_revision"]
            ),
            tokenizer_id=str(
                kronos_config["tokenizer_id"]
            ),
            tokenizer_revision=str(
                kronos_config["tokenizer_revision"]
            ),
            device=str(
                inference_config.get(
                    "device",
                    "auto",
                )
            ),
            dtype=str(
                inference_config.get(
                    "dtype",
                    "float32",
                )
            ),
            max_context=int(
                inference_config.get(
                    "max_context",
                    512,
                )
            ),
            clip=float(
                inference_config.get(
                    "clip",
                    5.0,
                )
            ),
            temperature=float(
                inference_config.get(
                    "temperature",
                    0.6,
                )
            ),
            top_k=int(
                inference_config.get(
                    "top_k",
                    0,
                )
            ),
            top_p=float(
                inference_config.get(
                    "top_p",
                    0.9,
                )
            ),
            sample_count=int(
                inference_config.get(
                    "sample_count",
                    10,
                )
            ),
            seed=int(
                inference_config.get(
                    "seed",
                    42,
                )
            ),
            series_batch_size=int(
                inference_config.get(
                    "series_batch_size",
                    1,
                )
            ),
            verbose=bool(
                inference_config.get(
                    "verbose",
                    False,
                )
            ),
        )


    @staticmethod
    def _parse_market_time(
        value: Any,
        field_name: str,
    ) -> time:
        """Parse project market-open/close metadata."""
        if isinstance(value, time):
            return value

        try:
            return pd.to_datetime(str(value)).time()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not parse {field_name}: {value!r}."
            ) from exc

    def _build_bar_close_timestamps(
        self,
        *,
        day: Any,
        market_open: Any,
        indices: list[int],
    ) -> pd.Series:
        """
        Convert cleaned intraday indices into bar-close timestamps.

        The raw first row is the previous session's final observation
        and is removed during cleaning. Under the project bar-close
        convention, cleaned index 0 is therefore market open + 1 minute.
        """
        if not indices:
            raise ValueError(
                "At least one intraday index is required."
            )

        if any(index < 0 for index in indices):
            raise ValueError(
                "Intraday indices must be non-negative."
            )

        if indices != sorted(indices):
            raise ValueError(
                "Intraday indices must be sorted ascending."
            )

        if len(indices) != len(set(indices)):
            raise ValueError(
                "Intraday indices must be unique."
            )

        day_timestamp = pd.Timestamp(day).normalize()
        open_time = self._parse_market_time(
            market_open,
            "market_open",
        )

        session_open = (
            day_timestamp
            + pd.Timedelta(
                hours=open_time.hour,
                minutes=open_time.minute,
                seconds=open_time.second,
            )
        )

        timestamps = pd.Series(
            pd.to_datetime(
                [
                    session_open
                    + pd.Timedelta(minutes=index + 1)
                    for index in indices
                ]
            )
        )

        if timestamps.isna().any():
            raise RuntimeError(
                "Generated Kronos timestamps contain missing values."
            )

        if not timestamps.is_monotonic_increasing:
            raise RuntimeError(
                "Generated Kronos timestamps are not increasing."
            )

        return timestamps

    def _build_window_predictor_inputs(
        self,
        *,
        example: dict[str, Any],
        split: SplitDict,
    ) -> tuple[
        list[pd.DataFrame],
        list[pd.Series],
        list[pd.Series],
    ]:
        """
        Convert one project forecast window into official predictor inputs.

        Args:
            example:
                One item from WindowedCandleDataset.

            split:
                The cleaned split from which the example was created.

        Returns:
            df_list:
                One OHLCV DataFrame per asset, preserving asset order.

            x_timestamp_list:
                Historical bar-close timestamps per asset.

            y_timestamp_list:
                Known future bar-close timestamps per asset.
        """
        required_example_keys = {
            "x",
            "day",
            "origin_idx",
            "context_start",
            "context_end",
            "session_length",
        }
        missing_example_keys = sorted(
            required_example_keys.difference(example)
        )
        if missing_example_keys:
            raise KeyError(
                "Window example is missing required keys: "
                f"{missing_example_keys}."
            )

        required_split_keys = {
            "asset_cols",
            "market_open",
        }
        missing_split_keys = sorted(
            required_split_keys.difference(split)
        )
        if missing_split_keys:
            raise KeyError(
                "Split is missing required keys: "
                f"{missing_split_keys}."
            )

        x = example["x"]

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Window example x must be a torch.Tensor."
            )

        expected_num_assets = len(split["asset_cols"])
        expected_shape = (
            self.context_length,
            expected_num_assets,
            len(self.input_channels),
        )
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                "Unexpected Kronos context shape. Expected "
                f"{expected_shape}, received {tuple(x.shape)}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Kronos context contains NaN or infinite values."
            )

        context_start = int(example["context_start"])
        context_end = int(example["context_end"])
        origin_idx = int(example["origin_idx"])
        session_length = int(example["session_length"])

        if context_end - context_start != self.context_length:
            raise ValueError(
                "Window context indices do not match context_length."
            )

        if context_end != origin_idx + 1:
            raise ValueError(
                "Window context_end must equal origin_idx + 1."
            )

        final_future_index = (
            origin_idx + self.prediction_length
        )
        if final_future_index >= session_length:
            raise ValueError(
                "Kronos forecast would cross the session boundary. "
                f"Final future index {final_future_index}, session "
                f"length {session_length}."
            )

        context_indices = list(
            range(context_start, context_end)
        )
        future_indices = list(
            range(
                origin_idx + 1,
                origin_idx + self.prediction_length + 1,
            )
        )

        x_timestamp = self._build_bar_close_timestamps(
            day=example["day"],
            market_open=split["market_open"],
            indices=context_indices,
        )
        y_timestamp = self._build_bar_close_timestamps(
            day=example["day"],
            market_open=split["market_open"],
            indices=future_indices,
        )

        if len(x_timestamp) != self.context_length:
            raise RuntimeError(
                "Historical timestamp count does not match "
                "context_length."
            )

        if len(y_timestamp) != self.prediction_length:
            raise RuntimeError(
                "Future timestamp count does not match "
                "prediction_length."
            )

        expected_first_future = (
            x_timestamp.iloc[-1]
            + pd.Timedelta(minutes=1)
        )
        if y_timestamp.iloc[0] != expected_first_future:
            raise RuntimeError(
                "The first future timestamp is not one minute after "
                "the final context timestamp."
            )

        x_cpu = (
            x.detach()
            .cpu()
            .to(dtype=torch.float32)
        )

        df_list: list[pd.DataFrame] = []
        x_timestamp_list: list[pd.Series] = []
        y_timestamp_list: list[pd.Series] = []

        for asset_index in range(expected_num_assets):
            asset_values = (
                x_cpu[:, asset_index, :]
                .numpy()
                .astype(np.float32, copy=False)
            )

            df_list.append(
                pd.DataFrame(
                    asset_values,
                    columns=self.input_channels,
                )
            )
            x_timestamp_list.append(
                x_timestamp.copy()
            )
            y_timestamp_list.append(
                y_timestamp.copy()
            )

        return (
            df_list,
            x_timestamp_list,
            y_timestamp_list,
        )

    def _dataset_config(self) -> dict[str, Any]:
        """Build the raw OHLCV window configuration used by Kronos."""
        return {
            "forecasting": {
                "context_length": self.context_length,
                "horizons": self.horizons,
                "stride": self.stride,
                "input_channels": self.input_channels,
                "target_channels": self.target_channels,
            }
        }

    def fit(
        self,
        train_split: SplitDict,
        val_split: SplitDict | None = None,
    ) -> "KronosBaseline":
        """
        Prepare the frozen official Kronos zero-shot benchmark.

        No optimisation is performed. The method validates project
        metadata, loads the pinned tokenizer/model checkpoints, freezes
        all parameters, and constructs the official predictor.
        """
        self._set_seed()

        self._validate_split(
            split=train_split,
            split_name="Training",
        )

        if val_split is not None:
            self._validate_split(
                split=val_split,
                split_name="Validation",
            )

            if list(train_split["asset_cols"]) != list(
                val_split["asset_cols"]
            ):
                raise ValueError(
                    "Training and validation asset ordering must "
                    "match exactly for Kronos."
                )

            if list(train_split["channels"]) != list(
                val_split["channels"]
            ):
                raise ValueError(
                    "Training and validation channel ordering must "
                    "match exactly for Kronos."
                )

            for metadata_key in (
                "market_open",
                "market_close",
            ):
                if train_split[metadata_key] != val_split[
                    metadata_key
                ]:
                    raise ValueError(
                        "Training and validation metadata disagree "
                        f"for {metadata_key!r}."
                    )

        resolved_device = self._resolve_device()

        (
            Kronos,
            KronosPredictor,
            KronosTokenizer,
        ) = import_official_kronos()

        tokenizer = KronosTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
        )
        model = Kronos.from_pretrained(
            self.model_id,
            revision=self.model_revision,
        )

        tokenizer.eval()
        model.eval()

        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.tokenizer = tokenizer
        self.model = model

        self._validate_checkpoint_pairing()

        predictor = KronosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            max_context=self.max_context,
            clip=self.clip,
        )

        model_device = next(model.parameters()).device
        tokenizer_device = next(tokenizer.parameters()).device

        if model_device != tokenizer_device:
            raise RuntimeError(
                "The official Kronos predictor placed the model and "
                "tokenizer on different devices. "
                f"Model: {model_device}; tokenizer: "
                f"{tokenizer_device}."
            )

        self.train_split = train_split
        self.val_split = val_split
        self.asset_cols = list(train_split["asset_cols"])
        self.num_assets = len(self.asset_cols)
        self.predictor = predictor
        self.device = str(model_device)
        self.is_fitted = True

        return self

    def _prediction_frame_to_array(
        self,
        *,
        prediction: pd.DataFrame,
        expected_timestamps: pd.Series,
    ) -> np.ndarray:
        """Validate one official prediction and return float32 OHLCVA."""
        expected_shape = (
            self.prediction_length,
            len(self.output_channels),
        )
        if prediction.shape != expected_shape:
            raise RuntimeError(
                "Unexpected Kronos prediction shape. Expected "
                f"{expected_shape}, received {prediction.shape}."
            )

        if list(prediction.columns) != self.output_channels:
            raise RuntimeError(
                "Unexpected Kronos output channel order. Expected "
                f"{self.output_channels}, received "
                f"{list(prediction.columns)}."
            )

        expected_index = pd.DatetimeIndex(
            pd.to_datetime(expected_timestamps)
        )
        if not prediction.index.equals(expected_index):
            raise RuntimeError(
                "Kronos prediction timestamps do not match the "
                "supplied future timestamps."
            )

        values = prediction.to_numpy(
            dtype=np.float32,
            copy=True,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                "Kronos prediction contains NaN or infinite values."
            )

        return values

    def predict(
        self,
        split: SplitDict,
        batch_size: int = 1,
        num_workers: int = 0,
        max_examples: int | None = None,
    ) -> PredictionDict:
        """
        Generate raw close-price predictions with frozen Kronos.

        Args:
            split:
                Cleaned project split.

            batch_size:
                Number of project forecast windows prepared together.
                Each window contains all assets.

            num_workers:
                DataLoader workers used to fetch project windows.

            max_examples:
                Optional number of forecast windows to process. This is
                intended for smoke tests; leave as None for a complete
                split prediction.

        Returns:
            Common raw-space project prediction dictionary with:

                y_pred: [B, H, N, 1]
                y_true: [B, H, N, 1]
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Call fit(...) before Kronos prediction."
            )

        if self.predictor is None:
            raise RuntimeError(
                "The official Kronos predictor is not initialised."
            )

        if self.asset_cols is None or self.num_assets is None:
            raise RuntimeError(
                "Kronos asset metadata has not been resolved."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_workers < 0:
            raise ValueError(
                "num_workers must be non-negative."
            )

        if max_examples is not None and max_examples <= 0:
            raise ValueError(
                "max_examples must be greater than zero when set."
            )

        self._validate_split(
            split=split,
            split_name="Prediction",
        )

        if list(split["asset_cols"]) != self.asset_cols:
            raise ValueError(
                "Prediction split asset ordering does not match the "
                "fitted Kronos wrapper."
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
            collate_fn=_identity_collate,
        )

        self._set_seed()

        all_y_pred: list[torch.Tensor] = []
        all_y_true: list[torch.Tensor] = []
        all_last_context_target: list[torch.Tensor] = []
        all_sample_idx: list[torch.Tensor] = []
        all_origin_idx: list[torch.Tensor] = []
        all_target_indices: list[torch.Tensor] = []

        processed_examples = 0

        with torch.inference_mode():
            for example_batch in loader:
                if max_examples is not None:
                    remaining = max_examples - processed_examples
                    if remaining <= 0:
                        break
                    example_batch = example_batch[:remaining]

                if not example_batch:
                    break

                df_list: list[pd.DataFrame] = []
                x_timestamp_list: list[pd.Series] = []
                y_timestamp_list: list[pd.Series] = []

                for example in example_batch:
                    (
                        example_dfs,
                        example_x_timestamps,
                        example_y_timestamps,
                    ) = self._build_window_predictor_inputs(
                        example=example,
                        split=split,
                    )
                    df_list.extend(example_dfs)
                    x_timestamp_list.extend(example_x_timestamps)
                    y_timestamp_list.extend(example_y_timestamps)

                expected_num_series = (
                    len(example_batch) * self.num_assets
                )
                if len(df_list) != expected_num_series:
                    raise RuntimeError(
                        "Unexpected number of flattened Kronos series. "
                        f"Expected {expected_num_series}, received "
                        f"{len(df_list)}."
                    )

                prediction_arrays: list[np.ndarray] = []

                for series_start in range(
                    0,
                    expected_num_series,
                    self.series_batch_size,
                ):
                    series_end = min(
                        series_start + self.series_batch_size,
                        expected_num_series,
                    )

                    try:
                        prediction_frames = (
                            self.predictor.predict_batch(
                                df_list=df_list[
                                    series_start:series_end
                                ],
                                x_timestamp_list=x_timestamp_list[
                                    series_start:series_end
                                ],
                                y_timestamp_list=y_timestamp_list[
                                    series_start:series_end
                                ],
                                pred_len=self.prediction_length,
                                T=self.temperature,
                                top_k=self.top_k,
                                top_p=self.top_p,
                                sample_count=self.sample_count,
                                verbose=self.verbose,
                            )
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Kronos inference failed for flattened "
                            f"series range [{series_start}, "
                            f"{series_end}). No fallback prediction was "
                            "used."
                        ) from exc

                    expected_chunk_size = series_end - series_start
                    if len(prediction_frames) != expected_chunk_size:
                        raise RuntimeError(
                            "Kronos predict_batch returned an "
                            "unexpected number of predictions. Expected "
                            f"{expected_chunk_size}, received "
                            f"{len(prediction_frames)}."
                        )

                    for local_index, prediction_frame in enumerate(
                        prediction_frames
                    ):
                        expected_timestamps = y_timestamp_list[
                            series_start + local_index
                        ]
                        prediction_arrays.append(
                            self._prediction_frame_to_array(
                                prediction=prediction_frame,
                                expected_timestamps=expected_timestamps,
                            )
                        )

                generated_ohlcva = torch.from_numpy(
                    np.stack(prediction_arrays, axis=0)
                )

                expected_generated_shape = (
                    expected_num_series,
                    self.prediction_length,
                    len(self.output_channels),
                )
                if tuple(generated_ohlcva.shape) != (
                    expected_generated_shape
                ):
                    raise RuntimeError(
                        "Unexpected stacked Kronos output shape. "
                        f"Expected {expected_generated_shape}, received "
                        f"{tuple(generated_ohlcva.shape)}."
                    )

                selected_close = generated_ohlcva[
                    :,
                    self.horizon_indices,
                    self.close_output_index,
                ].unsqueeze(-1)

                num_windows = len(example_batch)
                y_pred = (
                    selected_close.reshape(
                        num_windows,
                        self.num_assets,
                        len(self.horizons),
                        1,
                    )
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

                y_true = torch.stack(
                    [example["y"].float() for example in example_batch],
                    dim=0,
                )
                last_context_target = torch.stack(
                    [
                        example["last_context_target"].float()
                        for example in example_batch
                    ],
                    dim=0,
                )
                sample_idx = torch.tensor(
                    [
                        int(example["sample_idx"])
                        for example in example_batch
                    ],
                    dtype=torch.long,
                )
                origin_idx = torch.tensor(
                    [
                        int(example["origin_idx"])
                        for example in example_batch
                    ],
                    dtype=torch.long,
                )
                target_indices = torch.stack(
                    [
                        example["target_indices"].long()
                        for example in example_batch
                    ],
                    dim=0,
                )

                expected_project_shape = (
                    num_windows,
                    len(self.horizons),
                    self.num_assets,
                    len(self.target_channels),
                )
                if tuple(y_pred.shape) != expected_project_shape:
                    raise RuntimeError(
                        "Unexpected project Kronos prediction shape. "
                        f"Expected {expected_project_shape}, received "
                        f"{tuple(y_pred.shape)}."
                    )

                if tuple(y_true.shape) != expected_project_shape:
                    raise RuntimeError(
                        "Unexpected project Kronos target shape. "
                        f"Expected {expected_project_shape}, received "
                        f"{tuple(y_true.shape)}."
                    )

                all_y_pred.append(y_pred.cpu())
                all_y_true.append(y_true.cpu())
                all_last_context_target.append(
                    last_context_target.cpu()
                )
                all_sample_idx.append(sample_idx)
                all_origin_idx.append(origin_idx)
                all_target_indices.append(target_indices.cpu())

                processed_examples += num_windows

                if (
                    max_examples is not None
                    and processed_examples >= max_examples
                ):
                    break

        if not all_y_pred:
            raise RuntimeError(
                "Kronos prediction produced no examples."
            )

        return {
            "y_pred": torch.cat(all_y_pred, dim=0),
            "y_true": torch.cat(all_y_true, dim=0),
            "channels": list(self.target_channels),
            "horizons": list(self.horizons),
            "sample_idx": torch.cat(all_sample_idx, dim=0),
            "origin_idx": torch.cat(all_origin_idx, dim=0),
            "target_indices": torch.cat(
                all_target_indices,
                dim=0,
            ),
            "last_context_target": torch.cat(
                all_last_context_target,
                dim=0,
            ),
            "asset_cols": list(self.asset_cols),
            "output_space": "raw",
        }


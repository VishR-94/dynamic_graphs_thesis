from collections.abc import Callable
from typing import Any
import torch
from torch.utils.data import Dataset
from src.data.load_candle_data import get_channel_index

SplitDict = dict[str,Any]
ExampleDict = dict[str,Any]

#class to generate the windowed examples to feed to pytorch DataLoader later
class WindowedCandleDataset(Dataset):
    """
    Create supervised forecasting examples from cleaned candle data.

    Each cleaned daily sample has shape:

        [T, N, D]

    where:
        T = number of intraday time points
        N = number of assets
        D = number of channels

    Each dataset item returns:

        x: [context_length, N, num_input_channels]
        y: [num_horizons, N, num_target_channels]

    The target is direct multi-horizon:

        y[0] = value at origin + horizons[0]
        y[1] = value at origin + horizons[1]
        ...

    No normalization is applied here unless a normaliser is passed.
    """
    
    def __init__(
            self,
            split: SplitDict,
            context_length: int,
            horizons: list[int],
            input_channels: list[str],
            target_channels: list[str],
            normaliser: Callable[[ExampleDict],ExampleDict]|None=None
    )-> None:
        self.split = split
        self.context_length = context_length
        self.horizons = horizons
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.normaliser = normaliser
        
        self._validate_inputs()

        self.input_channel_ids = [
            get_channel_index(split,channel)
            for channel in input_channels
        ]

        self.target_channel_ids = [
            get_channel_index(split,channel)
            for channel in target_channels
        ]

        self.index = self._build_index()

    @classmethod
    def from_config(
        cls,
        split:SplitDict,
        config:dict[str,Any],
        normaliser:Callable[[ExampleDict],ExampleDict]|None=None
    ):
        """
        Build a WindowedCandleDataset from the forecasting YAML config.
        """
        forecasting_config = config['forecasting']

        return cls(
            split=split,
            context_length = int(forecasting_config['context_length']),
            horizons = list(forecasting_config['horizons']),
            input_channels = list(forecasting_config['input_channels']),
            target_channels = list(forecasting_config['target_channels']),
            normaliser = normaliser,
        )
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> ExampleDict:
        sample_idx, origin_idx = self.index[idx]

        x_day, _, day = self.split["samples"][sample_idx]

        context_start = origin_idx - self.context_length + 1
        context_end = origin_idx + 1

        target_indices = [
            origin_idx + horizon
            for horizon in self.horizons
        ]

        target_indices_tensor = torch.tensor(
            target_indices,
            dtype=torch.long,
        )

        x = x_day[
            context_start:context_end,
            :,
            self.input_channel_ids,
        ].float()

        y_full = x_day.index_select(0, target_indices_tensor)

        y = y_full[
            :,
            :,
            self.target_channel_ids,
        ].float()

        example: ExampleDict = {
            "x": x,
            "y": y,
            "day": day,
            "sample_idx": sample_idx,
            "origin_idx": origin_idx,
            "context_start": context_start,
            "context_end": context_end,
            "target_indices": target_indices_tensor,
            "input_channels": self.input_channels,
            "target_channels": self.target_channels,
            "horizons": self.horizons,
            "asset_cols": self.split["asset_cols"],
        }

        if self.normaliser is not None:
            example = self.normaliser(example)

        return example
    
    def _build_index(self) -> list[tuple[int,int]]:
        """
        Build a list of all valid forecasting origins.

        A valid origin has:
            enough history before it for the context window
            enough future after it for the largest forecast horizon
        """
        index = []
        max_horizon = max(self.horizons)

        for sample_idx, (x_day,_,_) in enumerate(self.split['samples']):
            num_time_points = x_day.shape[0]

            #we have to start here since we need context_length points before
            first_origin = self.context_length - 1
            #we have to end here since we will be predicting out to max_horizon
            #if we go beyond this index there will be no more true value for our predictions to compare to
            last_origin = num_time_points - max_horizon - 1

            if last_origin < first_origin:
                continue

            for origin_idx in range(first_origin,last_origin+1):
                index.append((sample_idx,origin_idx))
            
        return index
    
    def _validate_inputs(self)->None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")

        if len(self.horizons) == 0:
            raise ValueError("At least one horizon must be provided.")

        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError(f"All horizons must be positive. Got: {self.horizons}")

        if sorted(self.horizons) != self.horizons:
            raise ValueError(f"horizons must be sorted ascending. Got: {self.horizons}")

        if len(self.input_channels) == 0:
            raise ValueError("At least one input channel must be provided.")

        if len(self.target_channels) == 0:
            raise ValueError("At least one target channel must be provided.")

        available_channels = self.split["channels"]

        missing_input_channels = [
            channel
            for channel in self.input_channels
            if channel not in available_channels
        ]

        missing_target_channels = [
            channel
            for channel in self.target_channels
            if channel not in available_channels
        ]

        if len(missing_input_channels) > 0:
            raise ValueError(
                f"Missing input channels: {missing_input_channels}. "
                f"Available channels: {available_channels}."
            )

        if len(missing_target_channels) > 0:
            raise ValueError(
                f"Missing target channels: {missing_target_channels}. "
                f"Available channels: {available_channels}."
            )


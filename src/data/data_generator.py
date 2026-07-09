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

    No normalisation is applied here unless a normaliser is passed.
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

        #get the last point of the context window - this gives [93,6]
        last_context_full = x_day[origin_idx]
        #we only want the target channels (since this is all we will need to reconstruct)
        last_context_target = last_context_full[
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
            "last_context_target": last_context_target,
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


#class to normalise windowed data using window mean and std
class WindowContextNormaliser():
    """
    Normalise each forecasting example using statistics computed from its own
    context window.

    This is the raw-candle/Kronos-style normalisation route.

    For each example:

        x: [context_length, N, input_channels]
        y: [num_horizons, N, target_channels]

    We compute mean/std from x over the time dimension:

        mean: [N, input_channels]
        std:  [N, input_channels]

    Then we normalise:

        x_norm = (x - mean) / std

    For y, we use the matching target-channel statistics from the same context
    window.
    """

    def __init__(
            self,
            eps: float=1e-8,
            clip:bool=True,
            clip_min:float=-5,
            clip_max:float=5,
            apply_to_target:bool=True,
            include_stats:bool=True
    ):
        self.eps = eps
        self.clip = clip
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.apply_to_target = apply_to_target
        self.include_stats = include_stats

    @classmethod
    def from_config(
        cls,
        config:dict[str,Any]
    ):
        """
        Build a WindowContextNormaliser from the YAML config.
        """
        normalisation_config = config['normalisation']
        if normalisation_config["method"] != "window_context":
            raise ValueError(
                "WindowContextNormaliser requires "
                "normalisation.method: window_context"
            )
        
        window_config = normalisation_config['window_context']
        if window_config["stats_from"] != "context":
            raise ValueError("Only stats_from: context is currently supported.")

        if window_config["scope"] != "per_asset_channel":
            raise ValueError("Only scope: per_asset_channel is currently supported.")
        
        return cls(
            eps=float(normalisation_config["eps"]),
            clip=bool(normalisation_config["clip"]),
            clip_min=float(normalisation_config["clip_min"]),
            clip_max=float(normalisation_config["clip_max"]),
            apply_to_target=bool(window_config["apply_to_target"]),
            include_stats=bool(window_config["include_stats"]),
        )
    
    #this will be run automatically when self.normaliser since it is __call__
    def __call__(
            self,
            example:ExampleDict,
    )->ExampleDict:
        """
        Normalise one example dictionary.
        """
        x = example['x'].float()
        y = example['y'].float()

        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [T, N, C], got {x.shape}")

        if y.ndim != 3:
            raise ValueError(f"Expected y to have shape [H, N, C], got {y.shape}")
        
        #compute mean and std along the time dimension - returns [N,C]
        mean = x.mean(dim=0)
        std = x.std(dim=0,unbiased=False).clamp_min(self.eps)
        log_std = torch.log(std)
        
        #this works because of pytorch broadcasting
        x_norm = (x-mean)/std

        #clip x_norm if we want to
        if self.clip:
            x_norm = x_norm.clamp(self.clip_min,self.clip_max)

        #now we need to modify the original example dictionary
        example = dict(example)
        example['x'] = x_norm

        #now we need to normalise the targets - recall the target channels may not equal input channels
        #we also need to use the input data mean/std to normalise the target

        input_channels = example['input_channels']
        target_channels = example['target_channels']

        target_channel_positions = [
            input_channels.index(channel)
            for channel in target_channels
        ]

        #now we have the index of the target channels, convert it into a torch tensor
        target_channel_positions = torch.tensor(
            target_channel_positions,
            dtype=torch.long,
            device=mean.device,
        )
        
        #now get the mean, std and log_std of the target channels
        target_mean = mean.index_select(1, target_channel_positions)
        target_std = std.index_select(1, target_channel_positions)
        target_log_std = log_std.index_select(1, target_channel_positions)

        if self.apply_to_target:
            y_norm = (y-target_mean)/target_std

            if self.clip:
                y_norm = y_norm.clamp(self.clip_min,self.clip_max)
            
            example['y'] = y_norm

        if self.include_stats:
            example["norm_mean"] = mean
            example["norm_std"] = std
            example["norm_log_std"] = log_std

            example["target_norm_mean"] = target_mean
            example["target_norm_std"] = target_std
            example["target_norm_log_std"] = target_log_std
        
        return example

#function to generate log change data. Its better to evaluate our models on log changes 
#since this does not suffer from large level difference between the assets and channels
#MSE/RMSE on raw data is very difficult to compare across models/assets/channels

#important that we only compute log changes WITHIN each day (i.e. we dont want to compute 
# log changes where log(p_t) - log(p_(t-1)) and t/(t-1) are 2 close on one day and open the next)
#this is because there is stock splits and other price jumps that we can't clean out

def build_log_change_split(
        split:SplitDict,
        eps:float=1e-8
)->SplitDict:
    """
    Convert a cleaned raw candle split into within-day one-step log changes.

    For each daily tensor x with shape [T, N, D], this returns a new daily
    tensor with shape [T - 1, N, D]:

        x_log[t] = log(x[t + 1]) - log(x[t])

    This is computed separately within each day, so no overnight returns or
    stock-split boundary jumps are created.

    For price channels, these are log returns.
    For volume/amount channels, these are log changes.
    """

    if 'samples' not in split:
        raise KeyError("split must contain the key 'samples'")
    
    if len(split['samples'])==0:
        raise ValueError("split contains no samples")
    
    new_samples=[]

    for x_day,aux,day in split['samples']:
        if x_day.ndim != 3:
            raise ValueError(
                f'Expected each day to have shape [T,N,C], got {x_day.shape}'
            )
        x_day = x_day.float()
        log_values = torch.log(x_day.clamp_min(eps))
        x_log_change = log_values[1:,:,:] - log_values[:-1,:,:]

        new_samples.append((x_log_change,aux,day))

    log_split = dict(split)
    log_split['samples'] = new_samples
    log_split["T"] = new_samples[0][0].shape[0]
    log_split['representation'] = 'log_change'
    log_split['eps'] = eps

    return log_split
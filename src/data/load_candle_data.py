from pathlib import Path
from typing import Dict, Tuple, Any, Sequence
import torch

#create a dictionary called SplitDict -> keys are strings and values are any
#these will be train, val, test
SplitDict = dict[str,Any]

def load_torch_tensor(path: Path) -> SplitDict:
    """
    Load a PyTorch .pt file.

    We use weights_only=False because these files contain Python objects
    such as dictionaries, lists, tuples, and strings, not just tensors.

    """
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        #older versions don't have weights_only arg
        return torch.load(path,map_location="cpu")


#this will load train, val and test all in one go
def load_candle_splits(data_dir: str|Path) -> Tuple[SplitDict,SplitDict,SplitDict]:
    """
    Load train.pt, val.pt, and test.pt from a directory.

    Expected directory structure:

        data_dir/
            train.pt
            val.pt
            test.pt

    Returns:
        train, val, test dictionaries as an ordered tuple
    """
    #data_dir can be str or Path - this converts to path. expanduser() handles ~ in the path
    data_dir = Path(data_dir).expanduser()

    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')

    expected_files = ['train.pt','val.pt','test.pt']

    for filename in expected_files:
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f'Missing reqiured file: {filename} not found in {data_dir}')
        
    train = load_torch_tensor(data_dir/"train.pt")
    val = load_torch_tensor(data_dir/"val.pt")
    test = load_torch_tensor(data_dir/"test.pt")

    return train, val, test

#this will remove the first point for each day for one set (e.g. train)
def drop_first_point(split: SplitDict) -> SplitDict:
    """
    Remove the first time point from every daily sample.

    First raw point belongs to the previous day - needs to be removed

    Raw sample:
        x: [391, 93, 6]

    Cleaned sample:
        x: [390, 93, 6]

    The returned split is a shallow copy with cleaned samples.
    The original split is not modified.
    """

    #this creates a copy of split. If we just did cleaned = split,
    #they will point to the same object. This creates a new copy
    cleaned = dict(split)

    cleaned_sample = []

    for x, aux, day in split['samples']:
        x_clean = x[1:]
        cleaned_sample.append((x_clean,aux,day))

    cleaned['samples'] = cleaned_sample
    cleaned['T'] = cleaned_sample[0][0].shape[0]
    
    return cleaned

#right now this only calls drop_first_point, but later if we want to do more cleaning
#we can create new base functions and add them to this wrapper
def clean_candle_split(split: SplitDict) -> SplitDict:
    """
    Apply standard candle-data cleaning.

    For now this only drops the first point.
    Later we can add more cleaning steps here if needed.
    """
    return drop_first_point(split)


#function to run the clean_candle_split on all 3 sets at once
def clean_candle_splits(train:SplitDict,val:SplitDict,test:SplitDict) -> Tuple[SplitDict,SplitDict,SplitDict]:
    """
    Clean train, val, and test splits using the standard preprocessing.
    """
    return(
        clean_candle_split(train),
        clean_candle_split(val),
        clean_candle_split(test),
    )

#helper function to return the index for a specific named channel
#e.g. channel = "high" reuturns 1
def get_channel_index(split: SplitDict, channel: str) -> int:
    """
    Return the integer index of a named channel.

    Example:
        get_channel_index(train, "close")
    """

    channels = split['channels']

    if channel not in channels:
        raise ValueError("Channel {channel} not found! Available channels are {channels}")
    
    return channels.index(channel)

#function to extract the data for a specific channel for a specific day
def get_channel(x: torch.Tensor,split: SplitDict, channel: str) -> torch.Tensor:
    """
    Extract one channel from a daily tensor.

    Args:
        x: daily tensor with shape [T, N, D]
        split: split dictionary containing the channel names
        channel: channel name, e.g. "close" or "volume"

    Returns:
        Tensor with shape [T, N]
    """
    idx = get_channel_index(split,channel)
    return x[:,:,idx]

#function to compute log returns from close to close for a single day
def compute_log_returns(
    x: torch.Tensor,
    split: SplitDict,
    channels: Sequence[str],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute within-day log changes for one or more channels.

    For price channels such as open, high, low, and close, these are log returns.
    For non-price channels such as volume or amount, these are log changes.

    Args:
        x:
            Daily tensor with shape [T, N, D].

        split:
            Split dictionary containing channel names.

        channels:
            Channels to transform, e.g. ["close"] or
            ["open", "high", "low", "close"].

        eps:
            Small positive value used to avoid log(0).

    Returns:
        If one channel is requested:
            Tensor with shape [T - 1, N].

        If multiple channels are requested:
            Tensor with shape [T - 1, N, C],
            where C is len(channels).

    Definition:
        r[t] = log(value[t + 1]) - log(value[t])
    """
    if len(channels) == 0:
        raise ValueError("At least one channel must be requested.")

    channel_indices = [get_channel_index(split, channel) for channel in channels]

    values = x[:, :, channel_indices].float()
    log_values = torch.log(values.clamp_min(eps))

    returns = log_values[1:] - log_values[:-1]

    if len(channels) == 1:
        returns = returns[:, :, 0]

    return returns

#print useful information about the data split
def describe_split(split: SplitDict, name: str = "split") -> None:
    """
    Print useful metadata about a split.
    """
    print(f"\n{name}")
    print("-" * len(name))

    print("num samples:", len(split["samples"]))
    print("T:", split["T"])
    print("num assets:", len(split["asset_cols"]))
    print("channels:", split["channels"])
    print("grain:", split["grain"])
    print("market open:", split["market_open"])
    print("market close:", split["market_close"])
    print("dropped days:", len(split["dropped_days"]))

    x, aux, day = split["samples"][0]
    print("first day:", day)
    print("first x shape:", tuple(x.shape))
    print("first aux:", aux)

#function to compute sanity checks on dimensions of data
#use before training to make sure everything is as expected
def validate_clean_split(split: SplitDict,expected_T: int=390, check_finite:bool=True)->None:
    """
    Basic sanity checks for cleaned candle data.
    """
    for i, (x,aux,day) in enumerate(split['samples']):
        if x.shape[0] != expected_T:
            raise ValueError(
                f'Sample {i}/day {day} has T={x.shape[0]}, expected T={expected_T}'
            )
        
        if x.shape[1] != len(split['asset_cols']):
            raise ValueError(
                f'Sample {i}/day{day} has {x.shape[1]} assets - expected {len(split['asset_cols'])}'
            )
        
        if x.shape[2] != len(split["channels"]):
            raise ValueError(
                f'Sample {i}/day {day} has {x.shape[2]} features - expected {len(split["channels"])}'
            )
        
        if check_finite:
            x_float = x.float()

            if torch.isnan(x_float).any():
                raise ValueError(f"Sample {i} / day {day} contains NaN values.")

            if torch.isinf(x_float).any():
                raise ValueError(f"Sample {i} / day {day} contains infinite values.")
    
    print(f"Validation passed for {len(split['samples'])} samples.")


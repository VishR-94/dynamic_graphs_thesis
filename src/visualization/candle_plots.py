from typing import Any,Sequence
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from src.data.load_candle_data import compute_log_returns, get_channel

SplitDict = dict[str,Any]

#converts torch tensor or numpy object to numpy object
def to_numpy(x: torch.Tensor|np.ndarray) -> np.ndarray:
    """
    Convert a PyTorch tensor or NumPy array to a NumPy array.
    """

    if isinstance(x,torch.Tensor):
        return x.detach().cpu().numpy()
    
    return np.asarray(x)

#helper function to get indices of days to plot
#can supply a specific list or None=all days
#max_samples plots from day 1 up to day max_samples
def resolve_sample_indices(
        split: SplitDict,
        sample_indices: int|Sequence[int]|slice|None=None,
        max_samples: int|None=None,
)->list[int]:
    """
    Resolve user-provided sample indices into a list of integer sample indices.

    sample_indices=None means all samples.
    sample_indices=0 means the first sample.
    sample_indices=[0, 1, 2] means selected samples.
    sample_indices=slice(0, 10) means samples 0 through 9.
    """
    num_samples = len(split["samples"])

    if sample_indices is None:
        indices = list(range(num_samples))
    elif isinstance(sample_indices,list):
        indices = sample_indices
    elif isinstance(sample_indices,slice):
        indices = list(range(num_samples))[sample_indices]
    else:
        indices = list(sample_indices)

    for idx in indices:
        if idx<0 or idx>num_samples:
            raise IndexError(f"Sample index {idx} is out of range")
    
    if max_samples is not None:
        indices = indices[:max_samples]
    
    return indices

#helper function to get indices and tickers of assets to plot
#can supply integer index of assets or ticker as string 
#None=all assets.
def resolve_asset_indices(
        split: SplitDict,
        assets: str|int|Sequence[str|int]|None=None,
        max_assets: int|None=None
)-> tuple[list[int],list[str]]:
    """
    Resolve user-provided assets into integer indices and ticker labels.

    assets=None means all assets.
    assets="AAPL" means one asset.
    assets=["AAPL", "MSFT"] means selected assets.
    assets=[0, 1, 2] also works.
    """

    asset_cols = split['asset_cols']
    num_assets = len(asset_cols)

    if assets is None:
        indices = list(range(num_assets))
    else:
        if isinstance(assets,str) or isinstance(assets,int):
            assets = [assets]

        indices = []

        for asset in assets:
            if isinstance(asset,int):
                idx = asset
                if idx<0 or idx>num_assets:
                    raise IndexError(f"Asset index {idx} is out of range")
            elif isinstance(asset,str):
                if asset not in asset_cols:
                    raise ValueError(f"Asset {asset} not available")
                
                indices.append(asset_cols.index(asset))
            else:
                raise TypeError("Assets must be strings, integers or sequences of strings/integers")
    
    if max_assets is not None:
        indices = indices[:max_assets]
    
    labels = [asset_cols[idx] for idx in indices]

    return indices, labels

#this outputs a np array of log returns on chosen channel. it concatenates
#returns for the same asset over selected days. for example,
#selecting 2 days for all 93 assets returns array of dim [389*2,93]
def collect_channel_log_returns(
        split: SplitDict,
        channel: str='close',
        sample_indices: int|Sequence[int]|slice|None=None,
        assets: str|int|Sequence[int|str]|None=None,
)->tuple[np.ndarray,list[str]]:
    """
    Collect close-to-close log returns across selected days and assets.

    Assumes split has already been cleaned with clean_candle_split or
    clean_candle_splits, so each sample has shape [390, N, D].

    Returns:
        returns_array: NumPy array with shape [total_time, num_assets]
        asset_labels: selected asset ticker labels
    """
    sample_ids =resolve_sample_indices(split,sample_indices)
    asset_ids,asset_labels =resolve_asset_indices(split,assets)

    returns_per_day = []

    for sample_idx in sample_ids:
        x,aux,day = split['samples'][sample_idx]
        returns = compute_log_returns(
        x=x,
        split=split,
        channels=[channel],
        )
        returns = returns[:,asset_ids]
        #eg after looping over 2 days this contains
        #returns_day0 (shape[389,N]),returns_day1 (shape[389,N])
        returns_per_day.append(returns)

    #this concatenates along the time dimension, so for 2 days
    #this give shape[389*2,N]
    returns_array = torch.cat(returns_per_day,dim=0)

    return to_numpy(returns_array),asset_labels

#function to compute the correlation matrix from the output of
#collect_close_log_returns. Allows us to compute correlations
#across whichever days and whichever assets we want. Note that
#we may use a sparse verion of this later for our static graph
def compute_return_correlation_matrix(
        split: SplitDict,
        channel: str='close',
        sample_indices: int|Sequence[int]|slice|None=None,
        assets: str|int|Sequence[int|str]|None=None
)->tuple[np.ndarray,list[str]]:
    """
    Compute cross-asset correlation matrix from close log returns.

    If sample_indices=None, this uses all days in the split.
    """
    returns_array,asset_labels=collect_channel_log_returns(
        split=split,
        channel=channel,
        sample_indices=sample_indices,
        assets=assets,
    )
    if returns_array.shape[1]==1:
        corr = np.array([[1.0]])
    else:
        corr = np.corrcoef(returns_array,rowvar=False)
        corr = np.nan_to_num(corr,nan=0.0,posinf=0.0,neginf=0.0)
        np.fill_diagonal(corr,1.0)
    
    return corr, asset_labels

#this function takes a correlation matrix and tries to reorder the 
#assets so that assets with high (positive or absolute) correlation
#sit next to eachother. The goal is to make it easier to see clusters
#of high correlation assets.
def reorder_correlation_matrix(
        corr: np.ndarray,
        labels: list[str],
        cluster_by_abs: bool = True,
        method: str = "average",
)->tuple[np.ndarray,list[str],np.ndarray]:
    """
    Reorder a correlation matrix using hierarchical clustering.

    cluster_by_abs=False groups assets with high positive correlation.
    cluster_by_abs=True groups assets with high absolute correlation.
    """
    if corr.shape[0]<=1:
        order = np.arange(corr.shape[0])
        return corr,labels,order
    
    similarity = np.abs(corr) if cluster_by_abs else corr

    #hierachical clustering expects distances, not similarities
    #high correlation => low distance, hence 1-similarity
    #if cluster_by_abs=True, then large positive AND negative
    #correlations give small distances
    distance = 1.0 - similarity
    distance = np.clip(distance,0.0,2.0)
    np.fill_diagonal(distance,0.0)

    #the linkage function in SciPy doesnt want the full square (symmetric)
    #correlation matrix - it wants the lower triangular form as a 1D list
    #this is what squareform does for us
    condensed_distance = squareform(distance,checks=False)

    #this starts with every asset as its own group, and then merges the closest 
    #groups repeatedly until all assets are part of 1 tree. The results is a 
    #linkage matrix which encodes this clustering tree.
    linkage_matrix = linkage(condensed_distance,method=method)

    #this extracts the ordering implied by the clustering tree
    order = leaves_list(linkage_matrix)
    #form the reordered matrix
    reordered_corr = corr[np.ix_(order, order)]

    reordered_labels = [labels[i] for i in order]

    return reordered_corr, reordered_labels, order

#this function will compute the correlation matrix and plot it
#we can use the reordering function to reorder if we want
def plot_return_correlation_heatmap(
        split:SplitDict,
        channel: str='close',
        sample_indices:int|Sequence[int]|slice|None=None,
        assets:str|int|Sequence[str|int]|None=None,
        reorder:bool=False,
        cluster_by_abs:bool=False,
        show_tickers:bool=True,
        max_tick_labels:int=93,
        figsize:tuple[float,float]|None=None,
        ax:Axes|None=None,
)->tuple[Figure,Axes,np.ndarray,list[str]]:
    """
    Plot a cross-asset return correlation heatmap.

    If sample_indices=None, this computes correlation over all days in the split.
    """

    corr,labels = compute_return_correlation_matrix(
        split=split,
        channel=channel,
        sample_indices=sample_indices,
        assets=assets
    )
    
    if reorder:
        corr,labels,order = reorder_correlation_matrix(
            corr,
            labels=labels,
            cluster_by_abs=cluster_by_abs
        )
    
    num_assets = len(labels)

    if figsize is None:
        if num_assets<=20:
            figsize=(8,7)
        elif num_assets <=50:
            figsize=(11,10)
        else:
            figsize=(15,13)

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    #imshow displays a 2D matrix as an image
    image = ax.imshow(corr,vmin=-1.0,vmax=1.0,cmap="coolwarm",aspect="auto")
    fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    title="Return Correlation Heatmap"

    if sample_indices is None:
        title += "across all days"
    else:
        title += "for selected day(s)"
    
    if reorder:
        title += "- reordered"

    if show_tickers and num_assets<=max_tick_labels:
        ax.set_xticks(np.arange(num_assets))
        ax.set_yticks(np.arange(num_assets))
        ax.set_xticklabels(labels,rotation=45,fontsize=6)
        ax.set_yticklabels(labels,fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()

    return fig,ax,corr,labels

#utility functions to help us plot time series with actual dates as the x axis
def parse_day(day: Any) -> datetime:
    """
    Parse a day/session identifier into a datetime.

    Assumes day is stored in an ISO-like format such as '2024-01-03'.
    """
    day_text = str(day)

    try:
        return datetime.fromisoformat(day_text)
    except ValueError:
        return datetime.fromisoformat(day_text[:10])


def parse_market_time(market_time: str | time) -> time:
    """
    Parse a market time such as '09:30' into a Python time object.
    """
    if isinstance(market_time, time):
        return market_time

    hour, minute = str(market_time).split(":")[:2]

    return time(hour=int(hour), minute=int(minute))


def build_intraday_datetimes(
    split: SplitDict,
    day: Any,
    num_points: int,
    offset_minutes: int = 0,
) -> np.ndarray:
    """
    Build actual intraday timestamps for one sample.

    For close/volume data, offset_minutes=0 means the first cleaned point
    is treated as market_open.

    For returns, we will usually use offset_minutes=1 because return[0]
    corresponds to close[1] - close[0].
    """
    day_datetime = parse_day(day)
    market_open = parse_market_time(split["market_open"])

    start = datetime.combine(day_datetime.date(), market_open)
    start = start + timedelta(minutes=offset_minutes)

    return np.array([start + timedelta(minutes=i) for i in range(num_points)])

def format_day_label(day: Any) -> str:
    """
    Format a day/session identifier for axis labels.
    """
    return parse_day(day).strftime("%Y-%m-%d")

#function to plot intraday channel for selected assets and selected days
#by default, it will plot the first max_sample (10) days for the first max_assets (5)
#assets. To change this, we can increase max_samples and max_assets.
#mode=concat concatenates multiple days of data for 1 asset into 1 series
#mode=overlay plots each day as its own series on the same plot
#normalize=True will normalize each line plot by its first close price
def plot_intraday_channel(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    normalize: bool = False,
    mode: str = "concat",
    max_samples: int | None = 10,
    max_assets: int | None = 5,
    max_date_ticks: int = 12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot intraday time series for selected assets and selected days
    for a selected channel.

    Assumes split has already been cleaned.

    mode="concat":
        Plot one long line per asset across selected days.
        The x-axis is compressed to show trading minutes only.
        Small gaps filled with NaN values are inserted between days.
        If normalize=True, each asset is normalized by its first selected value
        in the selected range.

    mode="overlay":
        Plot each asset-day as a separate line on the intraday clock-time axis.
        If normalize=True, each asset-day is normalized by that day's first value.

    By default, this plots the first max_samples days and first max_assets assets.
    Set max_samples=None to plot all selected days.
    Set max_assets=None to plot all selected assets.
    """
    if mode not in {"concat", "overlay"}:
        raise ValueError(f"mode must be 'concat' or 'overlay', got {mode!r}")

    if channel not in split["channels"]:
        raise ValueError(f"Channel must be one of {split['channels']}")

    sample_ids = resolve_sample_indices(
        split,
        sample_indices,
        max_samples=max_samples,
    )

    asset_ids, asset_labels = resolve_asset_indices(
        split,
        assets,
        max_assets=max_assets,
    )

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    market_open = parse_market_time(split["market_open"])
    market_close = parse_market_time(split["market_close"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    if mode == "concat":
        gap_points = 5

        first_x, _, _ = split["samples"][sample_ids[0]]
        points_per_day = first_x.shape[0]

        for asset_idx, asset_label in zip(asset_ids, asset_labels):
            x_parts = []
            y_parts = []
            first_selected_value = None

            for day_position, sample_idx in enumerate(sample_ids):
                x, _, _ = split["samples"][sample_idx]

                channel_value = get_channel(x, split, channel).float()
                y = channel_value[:, asset_idx]

                if first_selected_value is None:
                    first_selected_value = y[0].clamp_min(1e-8)

                if normalize:
                    y = y / first_selected_value

                start = day_position * (points_per_day + gap_points)

                day_x = np.arange(start, start + len(y))
                day_y = to_numpy(y)

                x_parts.append(day_x)
                y_parts.append(day_y)

                if day_position < len(sample_ids) - 1:
                    gap_start = start + len(y)
                    gap_x = np.arange(gap_start, gap_start + gap_points)
                    gap_y = np.full(gap_points, np.nan)

                    x_parts.append(gap_x)
                    y_parts.append(gap_y)

            x_all = np.concatenate(x_parts)
            y_all = np.concatenate(y_parts)

            ax.plot(
                x_all,
                y_all,
                alpha=0.8,
                label=asset_label,
            )

        tick_every = max(1, len(sample_ids) // max_date_ticks)

        tick_positions = []
        tick_labels = []

        for day_position, sample_idx in enumerate(sample_ids):
            if day_position % tick_every != 0:
                continue

            _, _, day = split["samples"][sample_idx]
            start = day_position * (points_per_day + gap_points)

            tick_positions.append(start)
            tick_labels.append(
                f"{format_day_label(day)}\n{market_open.strftime('%H:%M')}"
            )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        xlabel = (
            f"Trading time only "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    else:
        num_lines = len(sample_ids) * len(asset_ids)
        dummy_date = datetime(2000, 1, 1)

        for sample_idx in sample_ids:
            x, _, day = split["samples"][sample_idx]

            channel_value = get_channel(x, split, channel).float()

            for asset_idx, asset_label in zip(asset_ids, asset_labels):
                y = channel_value[:, asset_idx]

                if normalize:
                    y = y / y[0].clamp_min(1e-8)

                timestamps = build_intraday_datetimes(
                    split=split,
                    day=day,
                    num_points=len(y),
                    offset_minutes=0,
                )

                x_axis = np.array(
                    [
                        datetime.combine(dummy_date.date(), ts.time())
                        for ts in timestamps
                    ]
                )

                label = f"{asset_label} {day}" if num_lines <= 20 else None

                ax.plot(
                    x_axis,
                    to_numpy(y),
                    alpha=0.8,
                    label=label,
                )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        xlabel = (
            f"Time of day "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    ylabel = f"Normalised {channel} values" if normalize else f"{channel} values"

    ax.set_title(f"Intraday {channel} values")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if mode == "concat" or len(sample_ids) * len(asset_ids) <= 20:
        ax.legend(fontsize=8)

    fig.tight_layout()

    return fig, ax

#function to plot log returns - similar to plot_intraday_channel but for log returns
def plot_intraday_log_returns(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    mode: str = "concat",
    max_samples: int | None = 10,
    max_assets: int | None = 5,
    max_date_ticks: int = 12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot intraday log returns/log changes for selected assets and days.

    For price channels such as open, high, low, and close, these are log returns.
    For non-price channels such as volume or amount, these are log changes.

    Assumes split has already been cleaned.

    mode="concat":
        Plot one long return/change line per asset across selected days.
        The x-axis is compressed to show trading return-minutes only.
        Small gaps filled with NaN values are inserted between days.

    mode="overlay":
        Plot each asset-day as a separate return/change line on the intraday
        clock-time axis.

    By default, this plots the first max_samples days and first max_assets assets.
    Set max_samples=None to plot all selected days.
    Set max_assets=None to plot all selected assets.
    """
    if mode not in {"concat", "overlay"}:
        raise ValueError(f"mode must be 'concat' or 'overlay', got {mode!r}")

    if channel not in split["channels"]:
        raise ValueError(f"Channel must be one of {split['channels']}")

    sample_ids = resolve_sample_indices(
        split,
        sample_indices,
        max_samples=max_samples,
    )

    asset_ids, asset_labels = resolve_asset_indices(
        split,
        assets,
        max_assets=max_assets,
    )

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    market_open = parse_market_time(split["market_open"])
    market_close = parse_market_time(split["market_close"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    if mode == "concat":
        gap_points = 5

        first_x, _, _ = split["samples"][sample_ids[0]]
        points_per_day = first_x.shape[0] - 1

        for asset_idx, asset_label in zip(asset_ids, asset_labels):
            x_parts = []
            y_parts = []

            for day_position, sample_idx in enumerate(sample_ids):
                x, _, _ = split["samples"][sample_idx]

                returns = compute_log_returns(
                    x=x,
                    split=split,
                    channels=[channel],
                )

                y = returns[:, asset_idx]

                start = day_position * (points_per_day + gap_points)

                day_x = np.arange(start, start + len(y))
                day_y = to_numpy(y)

                x_parts.append(day_x)
                y_parts.append(day_y)

                if day_position < len(sample_ids) - 1:
                    gap_start = start + len(y)
                    gap_x = np.arange(gap_start, gap_start + gap_points)
                    gap_y = np.full(gap_points, np.nan)

                    x_parts.append(gap_x)
                    y_parts.append(gap_y)

            x_all = np.concatenate(x_parts)
            y_all = np.concatenate(y_parts)

            ax.plot(
                x_all,
                y_all,
                alpha=0.8,
                label=asset_label,
            )

        tick_every = max(1, len(sample_ids) // max_date_ticks)

        tick_positions = []
        tick_labels = []

        for day_position, sample_idx in enumerate(sample_ids):
            if day_position % tick_every != 0:
                continue

            _, _, day = split["samples"][sample_idx]
            start = day_position * (points_per_day + gap_points)

            tick_positions.append(start)
            tick_labels.append(
                f"{format_day_label(day)}\n{market_open.strftime('%H:%M')}"
            )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        xlabel = (
            f"Trading return-time only "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    else:
        num_lines = len(sample_ids) * len(asset_ids)
        dummy_date = datetime(2000, 1, 1)

        for sample_idx in sample_ids:
            x, _, day = split["samples"][sample_idx]

            returns = compute_log_returns(
                x=x,
                split=split,
                channels=[channel],
            )

            for asset_idx, asset_label in zip(asset_ids, asset_labels):
                y = returns[:, asset_idx]

                timestamps = build_intraday_datetimes(
                    split=split,
                    day=day,
                    num_points=len(y),
                    offset_minutes=1,
                )

                x_axis = np.array(
                    [
                        datetime.combine(dummy_date.date(), ts.time())
                        for ts in timestamps
                    ]
                )

                label = f"{asset_label} {day}" if num_lines <= 20 else None

                ax.plot(
                    x_axis,
                    to_numpy(y),
                    alpha=0.8,
                    label=label,
                )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        xlabel = (
            f"Time of day "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    ax.axhline(0.0, linewidth=1)

    ax.set_title(f"Intraday {channel} log returns/log changes")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Log return / log change")

    if mode == "concat" or len(sample_ids) * len(asset_ids) <= 20:
        ax.legend(fontsize=8)

    fig.tight_layout()

    return fig, ax

#function to compute average absolute log return at each minute of trading day
#this will average over all arguments - if we pass in multiple assets and multiple
#days, it will return 1 curve which is the average abs log return over all days 
#and all assets.
def compute_average_intraday_abs_return(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation of absolute intraday log returns/log
    changes for one selected channel.

    For price channels such as open, high, low, and close, these are log returns.
    For non-price channels such as volume or amount, these are log changes.

    Averages over selected days and selected assets.

    Returns:
        avg_abs_return:
            NumPy array with shape [T - 1].

        std_abs_return:
            NumPy array with shape [T - 1].
    """
    if channel not in split["channels"]:
        raise ValueError(f"Channel must be one of {split['channels']}")

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, _ = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    abs_returns_per_day = []

    for sample_idx in sample_ids:
        x, _, _ = split["samples"][sample_idx]

        returns = compute_log_returns(
            x=x,
            split=split,
            channels=[channel],
        )

        returns = returns[:, asset_ids]
        abs_returns_per_day.append(returns.abs())

    stacked = torch.stack(abs_returns_per_day, dim=0)

    avg_abs_return = stacked.mean(dim=(0, 2))
    std_abs_return = stacked.std(dim=(0, 2), unbiased=False)

    return to_numpy(avg_abs_return), to_numpy(std_abs_return)

#function to plot the average absolute log returns computed above
def plot_average_intraday_abs_return(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    mode: str = "profile",
    max_date_ticks: int = 12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, np.ndarray, np.ndarray]:
    """
    Plot average absolute log returns/log changes for one selected channel.

    For price channels such as open, high, low, and close, these are log returns.
    For non-price channels such as volume or amount, these are log changes.

    mode="profile":
        Average over selected days and selected assets at each intraday minute.
        This gives one average intraday volatility profile.

    mode="concat":
        For each selected day, average over selected assets at each minute.
        Then concatenate those daily curves into one long trading-time series.

    The shaded band shows mean plus/minus one standard deviation.
    The lower side of the band is clipped at zero because absolute returns
    cannot be negative.
    """
    if mode not in {"profile", "concat"}:
        raise ValueError(f"mode must be 'profile' or 'concat', got {mode!r}")

    if channel not in split["channels"]:
        raise ValueError(f"Channel must be one of {split['channels']}")

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, _ = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    market_open = parse_market_time(split["market_open"])
    market_close = parse_market_time(split["market_close"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    if mode == "profile":
        avg_abs_return, std_abs_return = compute_average_intraday_abs_return(
            split=split,
            channel=channel,
            sample_indices=sample_indices,
            assets=assets,
        )

        dummy_date = datetime(2000, 1, 1)
        start = datetime.combine(dummy_date.date(), market_open)
        start = start + timedelta(minutes=1)

        x_axis = np.array(
            [start + timedelta(minutes=i) for i in range(len(avg_abs_return))]
        )

        lower = np.maximum(avg_abs_return - std_abs_return, 0.0)
        upper = avg_abs_return + std_abs_return

        ax.plot(
            x_axis,
            avg_abs_return,
            label=f"Mean absolute {channel} log return/change",
        )

        ax.fill_between(
            x_axis,
            lower,
            upper,
            alpha=0.2,
            label="Mean ± 1 standard deviation",
        )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        xlabel = (
            f"Time of day "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    else:
        gap_points = 5

        first_x, _, _ = split["samples"][sample_ids[0]]
        points_per_day = first_x.shape[0] - 1

        x_parts = []
        mean_parts = []
        std_parts = []

        for day_position, sample_idx in enumerate(sample_ids):
            x, _, _ = split["samples"][sample_idx]

            returns = compute_log_returns(
                x=x,
                split=split,
                channels=[channel],
            )

            returns = returns[:, asset_ids]
            abs_returns = returns.abs()

            day_mean = abs_returns.mean(dim=1)
            day_std = abs_returns.std(dim=1, unbiased=False)

            start = day_position * (points_per_day + gap_points)

            day_x = np.arange(start, start + len(day_mean))
            day_mean_np = to_numpy(day_mean)
            day_std_np = to_numpy(day_std)

            x_parts.append(day_x)
            mean_parts.append(day_mean_np)
            std_parts.append(day_std_np)

            if day_position < len(sample_ids) - 1:
                gap_start = start + len(day_mean)
                gap_x = np.arange(gap_start, gap_start + gap_points)
                gap_values = np.full(gap_points, np.nan)

                x_parts.append(gap_x)
                mean_parts.append(gap_values)
                std_parts.append(gap_values)

        x_axis = np.concatenate(x_parts)
        avg_abs_return = np.concatenate(mean_parts)
        std_abs_return = np.concatenate(std_parts)

        lower = np.maximum(avg_abs_return - std_abs_return, 0.0)
        upper = avg_abs_return + std_abs_return

        ax.plot(
            x_axis,
            avg_abs_return,
            label=f"Mean absolute {channel} log return/change across assets",
        )

        ax.fill_between(
            x_axis,
            lower,
            upper,
            alpha=0.2,
            label="Mean ± 1 standard deviation across assets",
        )

        tick_every = max(1, len(sample_ids) // max_date_ticks)

        tick_positions = []
        tick_labels = []

        for day_position, sample_idx in enumerate(sample_ids):
            if day_position % tick_every != 0:
                continue

            _, _, day = split["samples"][sample_idx]
            start = day_position * (points_per_day + gap_points)

            tick_positions.append(start)
            tick_labels.append(
                f"{format_day_label(day)}\n{market_open.strftime('%H:%M')}"
            )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        xlabel = (
            f"Trading return-time only "
            f"({market_open.strftime('%H:%M')}–{market_close.strftime('%H:%M')})"
        )

    ax.set_title(f"Average absolute {channel} log return/change")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Absolute log return / log change")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    fig.tight_layout()

    return fig, ax, avg_abs_return, std_abs_return
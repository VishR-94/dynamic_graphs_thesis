from pathlib import Path
from typing import Any, Sequence
from datetime import date, datetime, time, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from src.data.load_candle_data import compute_log_returns, get_channel
from src.utils.company_profiles import (
    make_asset_sector_mapping,
    make_sector_group_order,
)

SplitDict = dict[str,Any]
DateLike = str | date | datetime
VolatilityDaySelector = DateLike | tuple[DateLike, DateLike] | None

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
    elif isinstance(sample_indices, int):
        indices = [sample_indices]
    elif isinstance(sample_indices, slice):
        indices = list(range(num_samples))[sample_indices]
    else:
        indices = list(sample_indices)

    for idx in indices:
        if idx < 0 or idx >= num_samples:
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
                if idx < 0 or idx >= num_assets:
                    raise IndexError(f"Asset index {idx} is out of range")
                indices.append(idx)
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


def compute_sector_summary(
    assets: Sequence[str],
    *,
    company_profiles_path: str | Path | None = None,
) -> pd.DataFrame:
    """Summarise the supplied asset universe by sector.

    Only the supplied project assets are counted. The final row reports the
    complete universe total.
    """

    mapping = make_asset_sector_mapping(
        assets,
        company_profiles_path=company_profiles_path,
    )
    counts = (
        mapping.groupby("Sector", sort=True)
        .size()
        .rename("Number of equities")
        .reset_index()
    )
    counts["Percentage of universe"] = (
        100.0 * counts["Number of equities"] / len(mapping)
    )
    total = pd.DataFrame(
        {
            "Sector": ["Total"],
            "Number of equities": [len(mapping)],
            "Percentage of universe": [100.0],
        }
    )
    return pd.concat([counts, total], ignore_index=True)


def compute_split_summary_table(
    train: SplitDict,
    validation: SplitDict,
    test: SplitDict,
    *,
    context_length: int = 60,
    stride: int = 15,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
) -> pd.DataFrame:
    """Summarise split dates, session counts, and forecast-window counts.

    The first origin is ``context_length - 1``. Targets are located at
    ``origin + horizon`` and every window remains inside its source session.
    """

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    resolved_horizons = [int(value) for value in horizons]
    if not resolved_horizons or any(value <= 0 for value in resolved_horizons):
        raise ValueError("horizons must contain positive integers.")
    max_horizon = max(resolved_horizons)

    def summarise(name: str, split: SplitDict) -> dict[str, Any]:
        if not split["samples"]:
            raise ValueError(f"{name} contains no sessions.")

        dates = [parse_day(sample[2]).date() for sample in split["samples"]]
        num_windows = 0

        for x, _, day in split["samples"]:
            first_origin = context_length - 1
            final_origin = int(x.shape[0]) - 1 - max_horizon
            if final_origin < first_origin:
                raise ValueError(
                    f"Session {day} is too short for context_length="
                    f"{context_length} and max_horizon={max_horizon}."
                )
            num_windows += ((final_origin - first_origin) // stride) + 1

        return {
            "Split": name,
            "Start date": min(dates).isoformat(),
            "End date": max(dates).isoformat(),
            "Sessions": len(dates),
            "Forecast windows": int(num_windows),
        }

    return pd.DataFrame(
        [
            summarise("Train", train),
            summarise("Validation", validation),
            summarise("Test", test),
        ]
    )


def compute_session_market_volatility(
    split: SplitDict,
    *,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    eps: float = 1.0e-8,
) -> pd.DataFrame:
    r"""Compute session-level market realised volatility.

    For session :math:`d` and asset :math:`n`,

    .. math::

        RV_{d,n}=\sqrt{\sum_t r_{d,t,n}^2}.

    The market score is the arithmetic mean of ``RV`` across the selected
    assets. Returns are calculated separately inside each session, so overnight
    changes are never included.
    """

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel must be one of {split['channels']}, got {channel!r}."
        )
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, _ = resolve_asset_indices(split, assets)
    if not sample_ids:
        raise ValueError("No sample indices selected.")
    if not asset_ids:
        raise ValueError("No assets selected.")

    rows: list[dict[str, Any]] = []

    for sample_idx in sample_ids:
        x, _, day = split["samples"][sample_idx]
        returns = compute_log_returns(
            x=x,
            split=split,
            channels=[channel],
            eps=eps,
        ).double()[:, asset_ids]
        finite = torch.isfinite(returns)
        safe_squared_returns = torch.where(
            finite,
            returns.square(),
            torch.zeros_like(returns),
        )
        valid_counts = finite.sum(dim=0)
        asset_volatility = safe_squared_returns.sum(dim=0).sqrt()
        asset_volatility = asset_volatility.masked_fill(
            valid_counts == 0,
            torch.nan,
        )
        finite_asset_volatility = asset_volatility[
            torch.isfinite(asset_volatility)
        ]
        if finite_asset_volatility.numel() == 0:
            raise ValueError(
                f"Session {day} produced no finite realised-volatility values."
            )

        rows.append(
            {
                "sample_idx": int(sample_idx),
                "session_date": pd.Timestamp(parse_day(day).date()),
                "market_realised_volatility": float(
                    finite_asset_volatility.mean().item()
                ),
                "num_assets": int(finite_asset_volatility.numel()),
                "num_returns": int(returns.shape[0]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "session_date",
        kind="stable",
    ).reset_index(drop=True)


def _volatility_bucket_label(bucket: int, num_buckets: int) -> str:
    """Return a concise low-to-high label for a volatility bucket."""

    if num_buckets == 1:
        return "All sessions"
    if num_buckets == 2:
        return "Low volatility" if bucket == 1 else "High volatility"
    if num_buckets == 3:
        return ("Low volatility", "Medium volatility", "High volatility")[
            bucket - 1
        ]
    if bucket == 1:
        return "Lowest volatility"
    if bucket == num_buckets:
        return "Highest volatility"
    return f"Volatility bucket {bucket}"


def _assign_session_volatility_buckets(
    session_volatility: pd.DataFrame,
    *,
    num_buckets: int,
) -> pd.DataFrame:
    """Assign deterministic equal-count session-volatility buckets."""

    if isinstance(num_buckets, bool) or not isinstance(num_buckets, int):
        raise TypeError("num_buckets must be an integer.")
    if num_buckets < 1:
        raise ValueError("num_buckets must be at least 1.")
    if num_buckets > len(session_volatility):
        raise ValueError(
            "num_buckets cannot exceed the number of selected sessions."
        )

    ordered = session_volatility.sort_values(
        ["market_realised_volatility", "sample_idx"],
        kind="stable",
    ).reset_index(drop=True)
    ordered["volatility_bucket"] = 0

    for bucket, positions in enumerate(
        np.array_split(np.arange(len(ordered)), num_buckets),
        start=1,
    ):
        ordered.loc[positions, "volatility_bucket"] = bucket

    ordered["volatility_bucket"] = ordered[
        "volatility_bucket"
    ].astype(int)
    ordered["volatility_label"] = ordered["volatility_bucket"].map(
        lambda bucket: _volatility_bucket_label(bucket, num_buckets)
    )
    return ordered.sort_values("sample_idx", kind="stable").reset_index(
        drop=True
    )

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
        cluster:bool=False,
        company_profiles_path:str|Path|None=None,
        show_tickers:bool=True,
        max_tick_labels:int=93,
        figsize:tuple[float,float]|None=None,
        ax:Axes|None=None,
        title: bool = True,
)->tuple[Figure,Axes,np.ndarray,list[str]]:
    """Plot a cross-asset return-correlation heatmap.

    If ``sample_indices=None``, correlations are computed over every day in
    the split. If ``cluster=True``, assets are displayed in exactly the same
    fixed order as Graph Hub: sectors are sorted alphabetically and tickers
    are sorted alphabetically within each sector using ``company_profiles``.
    Clustering changes only the display order, not the correlation values.
    """

    corr,labels = compute_return_correlation_matrix(
        split=split,
        channel=channel,
        sample_indices=sample_indices,
        assets=assets
    )

    ordered_sectors:np.ndarray|None = None
    if cluster:
        order, ordered_mapping = make_sector_group_order(
            labels,
            company_profiles_path=company_profiles_path,
        )
        corr = corr[np.ix_(order, order)]
        labels = [labels[index] for index in order]
        ordered_sectors = (
            ordered_mapping["Sector"].astype(str).to_numpy()
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
    # Use a separate display matrix so the returned correlation matrix
    # retains its true diagonal values of 1.0.
    display_corr = corr.copy()

    np.fill_diagonal(
        display_corr,
        np.nan,
    )

    correlation_cmap = plt.get_cmap(
        "coolwarm"
    ).copy()

    correlation_cmap.set_bad(
        color="white"
    )

    image = ax.imshow(
        display_corr,
        vmin=-1.0,
        vmax=1.0,
        cmap=correlation_cmap,
        aspect="auto",
    )
    fig.colorbar(image,ax=ax,fraction=0.046,pad=0.04)
    if title:
        ax.set_title("Return correlation matrix")

    if show_tickers and num_assets<=max_tick_labels:
        ax.set_xticks(np.arange(num_assets))
        ax.set_yticks(np.arange(num_assets))
        ax.set_xticklabels(labels,rotation=45,fontsize=6)
        ax.set_yticklabels(labels,fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if ordered_sectors is not None and len(ordered_sectors):
        boundaries = (
            np.flatnonzero(ordered_sectors[1:] != ordered_sectors[:-1]) + 1
        )
        for boundary in boundaries:
            coordinate = float(boundary) - 0.5
            ax.axhline(coordinate,color="black",linewidth=0.8,alpha=0.65)
            ax.axvline(coordinate,color="black",linewidth=0.8,alpha=0.65)

    fig.tight_layout()

    return fig,ax,corr,labels


def plot_volatility_regime_correlation_heatmaps(
    split: SplitDict,
    *,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    volatility_tail_fraction: float = 0.20,
    cluster: bool = True,
    company_profiles_path: str | Path | None = None,
    show_tickers: bool = False,
    max_tick_labels: int = 93,
    figsize: tuple[float, float] = (12.0, 5.5),
    title: bool = True,
) -> tuple[
    Figure,
    np.ndarray,
    dict[str, np.ndarray],
    list[str],
    pd.DataFrame,
]:
    """Plot low- and high-volatility return correlations.

    The two panels contain the bottom and top
    ``volatility_tail_fraction`` of selected sessions, ranked by cross-asset
    mean realised volatility. Sessions between the two tails are excluded from
    the heatmaps. Both matrices share the same asset order and the fixed
    correlation scale ``[-1, 1]``.
    """

    tail_fraction = float(volatility_tail_fraction)
    if not 0.0 < tail_fraction <= 0.5:
        raise ValueError(
            "volatility_tail_fraction must satisfy 0 < value <= 0.5."
        )

    selected_sample_ids = resolve_sample_indices(split, sample_indices)
    session_volatility = compute_session_market_volatility(
        split,
        channel=channel,
        sample_indices=selected_sample_ids,
        assets=assets,
    ).sort_values(
        ["market_realised_volatility", "sample_idx"],
        kind="stable",
    ).reset_index(drop=True)

    tail_count = max(1, int(np.floor(len(session_volatility) * tail_fraction)))
    if 2 * tail_count > len(session_volatility):
        raise ValueError(
            "The selected tail fraction leaves overlapping low- and "
            "high-volatility groups."
        )

    low_sample_ids = (
        session_volatility.head(tail_count)["sample_idx"].astype(int).tolist()
    )
    high_sample_ids = (
        session_volatility.tail(tail_count)["sample_idx"].astype(int).tolist()
    )

    regime_sample_ids = {
        "low": low_sample_ids,
        "high": high_sample_ids,
    }
    correlations: dict[str, np.ndarray] = {}
    labels: list[str] | None = None

    for regime, regime_indices in regime_sample_ids.items():
        regime_correlation, regime_labels = compute_return_correlation_matrix(
            split=split,
            channel=channel,
            sample_indices=regime_indices,
            assets=assets,
        )
        if labels is None:
            labels = regime_labels
        elif regime_labels != labels:
            raise ValueError(
                "Asset ordering changed across volatility regimes."
            )
        correlations[regime] = regime_correlation

    if labels is None:
        raise ValueError("No assets were available for the heatmaps.")

    ordered_sectors: np.ndarray | None = None
    if cluster:
        order, ordered_mapping = make_sector_group_order(
            labels,
            company_profiles_path=company_profiles_path,
        )
        labels = [labels[index] for index in order]
        ordered_sectors = ordered_mapping["Sector"].astype(str).to_numpy()
        correlations = {
            regime: correlation[np.ix_(order, order)]
            for regime, correlation in correlations.items()
        }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    correlation_cmap = plt.get_cmap("coolwarm").copy()
    correlation_cmap.set_bad(color="white")
    panel_titles = {
        "low": "Low volatility",
        "high": "High volatility",
    }
    image = None

    for axes_item, regime in zip(
        axes,
        ("low", "high"),
        strict=True,
    ):
        display_correlation = correlations[regime].copy()
        np.fill_diagonal(display_correlation, np.nan)
        image = axes_item.imshow(
            display_correlation,
            vmin=-1.0,
            vmax=1.0,
            cmap=correlation_cmap,
            aspect="auto",
        )

        if show_tickers and len(labels) <= max_tick_labels:
            axes_item.set_xticks(np.arange(len(labels)))
            axes_item.set_yticks(np.arange(len(labels)))
            axes_item.set_xticklabels(labels, rotation=45, fontsize=6)
            axes_item.set_yticklabels(labels, fontsize=6)
        else:
            axes_item.set_xticks([])
            axes_item.set_yticks([])

        if ordered_sectors is not None and len(ordered_sectors):
            boundaries = (
                np.flatnonzero(
                    ordered_sectors[1:] != ordered_sectors[:-1]
                )
                + 1
            )
            for boundary in boundaries:
                coordinate = float(boundary) - 0.5
                axes_item.axhline(
                    coordinate,
                    color="black",
                    linewidth=0.8,
                    alpha=0.65,
                )
                axes_item.axvline(
                    coordinate,
                    color="black",
                    linewidth=0.8,
                    alpha=0.65,
                )

        if title:
            axes_item.set_title(panel_titles[regime])

    if image is None:
        raise RuntimeError("The correlation heatmaps were not created.")
    figure.colorbar(
        image,
        ax=axes.tolist(),
        fraction=0.025,
        pad=0.02,
        label="Pearson correlation",
    )

    session_assignments = session_volatility.copy()
    session_assignments["regime"] = "Middle"
    session_assignments.loc[
        session_assignments["sample_idx"].isin(low_sample_ids),
        "regime",
    ] = "Low"
    session_assignments.loc[
        session_assignments["sample_idx"].isin(high_sample_ids),
        "regime",
    ] = "High"
    session_assignments["volatility_tail_fraction"] = tail_fraction

    return figure, axes, correlations, labels, session_assignments


def plot_split_volatility_distribution(
    train: SplitDict,
    validation: SplitDict,
    test: SplitDict,
    *,
    channel: str = "close",
    assets: str | int | Sequence[str | int] | None = None,
    figsize: tuple[float, float] = (9.0, 5.0),
    ax: Axes | None = None,
    title: bool = True,
) -> tuple[Figure, Axes, pd.DataFrame, pd.DataFrame]:
    """Compare session-level market realised volatility across data splits.

    Each session value is the arithmetic mean of the asset-level realised
    volatilities, where asset-level realised volatility is the square root of
    the sum of squared within-session one-minute log returns.
    """

    reference_assets = list(train["asset_cols"])
    for split_name, split in (
        ("Validation", validation),
        ("Test", test),
    ):
        if list(split["asset_cols"]) != reference_assets:
            raise ValueError(
                f"{split_name} asset ordering differs from the training split."
            )

    tables: list[pd.DataFrame] = []
    for split_name, split in (
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ):
        table = compute_session_market_volatility(
            split,
            channel=channel,
            assets=assets,
        )
        table.insert(0, "split", split_name)
        tables.append(table)

    values = pd.concat(tables, ignore_index=True)
    split_order = ("Train", "Validation", "Test")
    summary = (
        values.groupby("split", sort=False)["market_realised_volatility"]
        .agg(
            sessions="count",
            mean="mean",
            standard_deviation=lambda series: series.std(ddof=0),
            minimum="min",
            q25=lambda series: series.quantile(0.25),
            median="median",
            q75=lambda series: series.quantile(0.75),
            maximum="max",
        )
        .reindex(split_order)
        .reset_index()
    )

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure

    plot_values = [
        values.loc[
            values["split"] == split_name,
            "market_realised_volatility",
        ].to_numpy(dtype=np.float64)
        for split_name in split_order
    ]
    axes.boxplot(plot_values, labels=split_order, showfliers=True)
    axes.set_ylabel("Realised Volatility")
    axes.grid(axis="y", alpha=0.25)
    if title:
        axes.set_title("Session realised volatility by split")
    figure.tight_layout()

    return figure, axes, values, summary

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


def _resolve_volatility_sample_indices(
    split: SplitDict,
    day: VolatilityDaySelector,
) -> list[int]:
    """Resolve an exact day, inclusive date range, or the complete split.

    ``day=None`` selects every session in ``split``. An exact date-like value
    selects one session. A two-item tuple ``(start, end)`` selects every
    session in the inclusive calendar range. The helper never joins returns
    across session boundaries.
    """

    sample_dates = [
        parse_day(sample[2]).date()
        for sample in split["samples"]
    ]

    if day is None:
        indices = list(range(len(sample_dates)))

    elif isinstance(day, tuple):
        if len(day) != 2:
            raise ValueError(
                "A volatility date range must be a two-item tuple "
                "(start_date, end_date)."
            )

        start_date = parse_day(day[0]).date()
        end_date = parse_day(day[1]).date()

        if start_date > end_date:
            raise ValueError(
                "The realised-volatility start date must not be later "
                "than the end date."
            )

        indices = [
            index
            for index, sample_date in enumerate(sample_dates)
            if start_date <= sample_date <= end_date
        ]

    else:
        selected_date = parse_day(day).date()
        indices = [
            index
            for index, sample_date in enumerate(sample_dates)
            if sample_date == selected_date
        ]

    if not indices:
        if day is None:
            description = "the supplied split"
        elif isinstance(day, tuple):
            description = f"the inclusive range {day[0]} to {day[1]}"
        else:
            description = f"the exact day {day}"

        available = [
            value.isoformat()
            for value in sample_dates
        ]

        raise ValueError(
            f"No sessions were found for {description}. "
            f"Available dates include {available[:10]}."
        )

    return indices


def plot_realised_volatility(
    split: SplitDict,
    *,
    asset: str | int | None = None,
    day: VolatilityDaySelector = None,
    average_over_all_days: bool = False,
    window_minutes: int = 60,
    channel: str = "close",
    eps: float = 1.0e-8,
    figsize: tuple[float, float] = (14.0, 5.5),
    ax: Axes | None = None,
    title: bool = True,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Plot rolling realised volatility from within-session log returns.

    Realised volatility is the population standard deviation (``ddof=0``)
    of the previous ``window_minutes`` one-minute log returns. Rolling
    windows reset at every session boundary, so the statistic never mixes an
    overnight gap with intraday returns.

    Parameters
    ----------
    split:
        A cleaned candle-data split.
    asset:
        A ticker or zero-based asset index. ``None`` first calculates the
        rolling volatility independently for every asset and then takes the
        cross-asset mean at each timestamp. It does *not* calculate the
        volatility of a cross-sectional average return series.
    day:
        ``None`` selects every session in the split. A single date-like value
        selects that exact session. A two-item tuple ``(start_date, end_date)``
        selects the inclusive date range.
    average_over_all_days:
        Applied only when ``day=None``. When ``False``, the selected sessions
        are plotted on their actual calendar timestamps. When ``True``, daily
        volatility curves are aligned by intraday bar-close time and averaged
        across all sessions in the supplied split, producing one average
        intraday volatility profile. If an exact day or date range is passed,
        this flag has no effect.
    window_minutes:
        Number of one-minute returns in each rolling standard deviation.

    Returns
    -------
    figure, axes, values:
        In ordinary mode, ``values`` contains one row per selected session and
        timestamp. In across-day-average mode, it contains one row per
        intraday bar-close time with the mean realised volatility and number
        of sessions averaged.
    """

    if window_minutes < 2:
        raise ValueError(
            "window_minutes must be at least 2 for a meaningful "
            "standard deviation."
        )

    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel {channel!r} is unavailable. "
            f"Available channels are {split['channels']}."
        )

    sample_indices = _resolve_volatility_sample_indices(
        split,
        day,
    )

    if asset is None:
        asset_indices = list(range(len(split["asset_cols"])))
        asset_label = "Cross-asset mean"
    else:
        asset_indices, asset_labels = resolve_asset_indices(
            split,
            asset,
        )

        if len(asset_indices) != 1:
            raise ValueError(
                "plot_realised_volatility accepts one asset or asset=None."
            )

        asset_label = asset_labels[0]

    records: list[dict[str, Any]] = []

    for sample_index in sample_indices:
        x, _, sample_day = split["samples"][sample_index]

        log_returns = compute_log_returns(
            x=x,
            split=split,
            channels=[channel],
            eps=eps,
        )[:, asset_indices]

        returns_frame = pd.DataFrame(
            to_numpy(log_returns),
        )

        rolling_volatility = (
            returns_frame
            .rolling(
                window=int(window_minutes),
                min_periods=int(window_minutes),
            )
            .std(ddof=0)
        )

        if asset is None:
            volatility_values = rolling_volatility.mean(
                axis=1,
                skipna=True,
            ).to_numpy(dtype=np.float64)
        else:
            volatility_values = rolling_volatility.iloc[:, 0].to_numpy(
                dtype=np.float64
            )

        # Cleaned close index 0 is the 09:31 bar endpoint. Return index 0
        # therefore ends at cleaned close index 1 (09:32). The first finite
        # 60-return volatility value is consequently aligned to 10:31.
        close_timestamps = build_intraday_datetimes(
            split=split,
            day=sample_day,
            num_points=int(x.shape[0]),
            offset_minutes=1,
        )
        return_timestamps = close_timestamps[1:]

        finite = np.isfinite(volatility_values)

        for timestamp, value in zip(
            return_timestamps[finite],
            volatility_values[finite],
            strict=True,
        ):
            timestamp_value = pd.Timestamp(timestamp)
            intraday_minute = int(
                timestamp_value.hour * 60
                + timestamp_value.minute
            )
            records.append(
                {
                    "Date": format_day_label(sample_day),
                    "Timestamp": timestamp_value,
                    "Bar-close time": timestamp_value.strftime("%H:%M"),
                    "Intraday minute": intraday_minute,
                    "Asset": asset_label,
                    "Window minutes": int(window_minutes),
                    "Realised volatility": float(value),
                }
            )

    values = pd.DataFrame(records)

    if values.empty:
        raise ValueError(
            "The realised-volatility selection produced no finite values. "
            "The rolling window may be longer than the selected sessions."
        )

    values = values.sort_values(
        ["Timestamp"],
        kind="stable",
    ).reset_index(drop=True)

    average_applied = bool(
        average_over_all_days
        and day is None
    )

    if average_applied:
        plot_values = (
            values.groupby(
                [
                    "Intraday minute",
                    "Bar-close time",
                    "Asset",
                    "Window minutes",
                ],
                as_index=False,
                sort=True,
            )
            .agg(
                **{
                    "Realised volatility": (
                        "Realised volatility",
                        "mean",
                    ),
                    "Sessions averaged": (
                        "Date",
                        "nunique",
                    ),
                }
            )
            .sort_values("Intraday minute")
            .reset_index(drop=True)
        )
        plot_values["Date"] = "Average across all days"
        plot_values["Timestamp"] = (
            pd.Timestamp("2000-01-01")
            + pd.to_timedelta(
                plot_values["Intraday minute"],
                unit="m",
            )
        )
        plot_values["Average over all days"] = True
        plot_values = plot_values[
            [
                "Date",
                "Timestamp",
                "Bar-close time",
                "Intraday minute",
                "Asset",
                "Window minutes",
                "Sessions averaged",
                "Average over all days",
                "Realised volatility",
            ]
        ]
    else:
        plot_values = values.copy()
        plot_values["Sessions averaged"] = 1
        plot_values["Average over all days"] = False

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        axes = ax
        figure = axes.figure

    if average_applied:
        axes.plot(
            plot_values["Timestamp"],
            plot_values["Realised volatility"],
            color="tab:blue",
            linewidth=1.7,
            label=asset_label,
        )
        locator = mdates.AutoDateLocator(
            minticks=4,
            maxticks=12,
        )
        axes.xaxis.set_major_locator(locator)
        axes.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        session_count = int(
            values["Date"].nunique()
        )
        date_description = (
            f"average intraday profile across {session_count} sessions"
        )
        axes.set_xlabel("Bar-close time")
    else:
        for group_index, (_, group) in enumerate(
            plot_values.groupby("Date", sort=True)
        ):
            axes.plot(
                group["Timestamp"],
                group["Realised volatility"],
                color="tab:blue",
                linewidth=1.35,
                alpha=0.9,
                label=(asset_label if group_index == 0 else None),
            )

        unique_dates = tuple(
            plot_values["Date"].drop_duplicates()
        )

        if len(unique_dates) == 1:
            locator = mdates.AutoDateLocator(
                minticks=4,
                maxticks=10,
            )
            axes.xaxis.set_major_locator(locator)
            axes.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M")
            )
            date_description = unique_dates[0]
            axes.set_xlabel("Bar-close time")
        else:
            locator = mdates.AutoDateLocator(
                minticks=4,
                maxticks=12,
            )
            axes.xaxis.set_major_locator(locator)
            axes.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(locator)
            )
            date_description = (
                f"{unique_dates[0]} to {unique_dates[-1]} "
                f"({len(unique_dates)} sessions)"
            )
            axes.set_xlabel("Date and bar-close time")

    axes.set_ylabel(
        "Realised Volatility"
    )
    if title:
        axes.set_title(
            f"Intraday Realised Volatility"
        )
    axes.grid(True, alpha=0.25)
    axes.legend(loc="best")
    figure.tight_layout()

    return figure, axes, plot_values

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
    title: bool = True,
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

    if title:
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
    title: bool = True,
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

    if title:
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
    title: bool = True,
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

    if title:
        ax.set_title(f"Average absolute {channel} log return/change")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Absolute log return / log change")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    fig.tight_layout()

    return fig, ax, avg_abs_return, std_abs_return

def _validate_positive_integers(
    values: Sequence[int],
    name: str,
) -> list[int]:
    """Validate and preserve an ordered sequence of positive integers."""
    resolved = [int(value) for value in values]

    if len(resolved) == 0:
        raise ValueError(f"{name} must contain at least one value.")

    if any(value <= 0 for value in resolved):
        raise ValueError(f"All {name} values must be positive integers.")

    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate values.")

    return resolved


def _quantile_column_name(quantile: float) -> str:
    """Create a compact column name such as q05, q50, or q95."""
    percentage = int(round(100 * quantile))
    return f"q{percentage:02d}"


def compute_persistence_movement_summary(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    context_length: int = 60,
    stride: int = 15,
    quantiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    eps: float = 1e-8,
) -> "pd.DataFrame":
    """
    Summarise the movement that a persistence forecast must absorb.

    The function uses the same forecast-origin convention as the project:

        first origin = context_length - 1
        target index = origin + horizon

    Origins are separated by ``stride`` and always remain inside a session.
    No overnight differences are introduced.

    Returns one row per horizon with raw-space persistence errors and the
    distribution of cumulative log changes. For persistence,
    cumulative-log-change MAE is exactly the MAE of predicting zero change.
    """
    import pandas as pd

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel must be one of {split['channels']}, got {channel!r}."
        )

    if context_length <= 0:
        raise ValueError("context_length must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    if eps <= 0:
        raise ValueError("eps must be positive.")

    resolved_horizons = _validate_positive_integers(
        horizons,
        "horizons",
    )
    resolved_quantiles = [float(value) for value in quantiles]

    if len(resolved_quantiles) == 0:
        raise ValueError("quantiles must contain at least one value.")

    if any(value < 0.0 or value > 1.0 for value in resolved_quantiles):
        raise ValueError("All quantiles must lie in [0, 1].")

    if len(set(resolved_quantiles)) != len(resolved_quantiles):
        raise ValueError("quantiles must not contain duplicate values.")

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, _ = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    max_horizon = max(resolved_horizons)
    raw_errors: dict[int, list[torch.Tensor]] = {
        horizon: []
        for horizon in resolved_horizons
    }
    log_changes: dict[int, list[torch.Tensor]] = {
        horizon: []
        for horizon in resolved_horizons
    }
    total_windows = 0

    for sample_idx in sample_ids:
        x, _, day = split["samples"][sample_idx]
        values = get_channel(x, split, channel).double()
        values = values[:, asset_ids]

        if values.ndim != 2:
            raise ValueError(
                "Expected selected channel data to have shape [T, N], "
                f"got {tuple(values.shape)} for sample {sample_idx} ({day})."
            )

        num_points = values.shape[0]
        first_origin = context_length - 1
        final_origin = num_points - 1 - max_horizon

        if final_origin < first_origin:
            raise ValueError(
                f"Sample {sample_idx} ({day}) is too short for "
                f"context_length={context_length} and "
                f"max_horizon={max_horizon}."
            )

        origins = torch.arange(
            first_origin,
            final_origin + 1,
            stride,
            dtype=torch.long,
        )
        total_windows += int(origins.numel())

        current = values[origins]
        current_log = torch.log(current.clamp_min(eps))

        for horizon in resolved_horizons:
            target = values[origins + horizon]
            target_log = torch.log(target.clamp_min(eps))

            raw_errors[horizon].append(
                (target - current).reshape(-1)
            )
            log_changes[horizon].append(
                (target_log - current_log).reshape(-1)
            )

    rows: list[dict[str, float | int]] = []

    for horizon in resolved_horizons:
        horizon_raw_error = torch.cat(raw_errors[horizon])
        horizon_log_change = torch.cat(log_changes[horizon])

        finite = torch.isfinite(horizon_raw_error) & torch.isfinite(
            horizon_log_change
        )
        horizon_raw_error = horizon_raw_error[finite]
        horizon_log_change = horizon_log_change[finite]

        if horizon_raw_error.numel() == 0:
            raise ValueError(
                f"No finite observations were available for horizon {horizon}."
            )

        row: dict[str, float | int] = {
            "horizon": horizon,
            "num_sessions": len(sample_ids),
            "num_windows": total_windows,
            "num_assets": len(asset_ids),
            "num_observations": int(horizon_raw_error.numel()),
            "persistence_raw_mae": float(
                horizon_raw_error.abs().mean().item()
            ),
            "persistence_raw_rmse": float(
                horizon_raw_error.square().mean().sqrt().item()
            ),
            "cumulative_log_change_mean": float(
                horizon_log_change.mean().item()
            ),
            "cumulative_log_change_std": float(
                horizon_log_change.std(unbiased=False).item()
            ),
            "cumulative_log_change_mae": float(
                horizon_log_change.abs().mean().item()
            ),
            "cumulative_log_change_rmse": float(
                horizon_log_change.square().mean().sqrt().item()
            ),
            "median_abs_cumulative_log_change": float(
                horizon_log_change.abs().median().item()
            ),
        }

        quantile_tensor = torch.tensor(
            resolved_quantiles,
            dtype=horizon_log_change.dtype,
        )
        quantile_values = torch.quantile(
            horizon_log_change,
            quantile_tensor,
        )

        for quantile, value in zip(
            resolved_quantiles,
            quantile_values.tolist(),
        ):
            row[
                f"cumulative_log_change_{_quantile_column_name(quantile)}"
            ] = float(value)

        rows.append(row)

    return pd.DataFrame(rows)


def _transform_return_series(
    returns: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Apply the requested transformation before autocorrelation."""
    aliases = {
        "return": "return",
        "returns": "return",
        "absolute": "absolute",
        "absolute_return": "absolute",
        "absolute_returns": "absolute",
        "squared": "squared",
        "squared_return": "squared",
        "squared_returns": "squared",
    }

    if kind not in aliases:
        raise ValueError(
            "kind must be one of 'return', 'absolute', or 'squared'."
        )

    resolved_kind = aliases[kind]

    if resolved_kind == "absolute":
        return returns.abs()

    if resolved_kind == "squared":
        return returns.square()

    return returns


def _correlation_from_sufficient_statistics(
    count: torch.Tensor,
    sum_x: torch.Tensor,
    sum_y: torch.Tensor,
    sum_x_squared: torch.Tensor,
    sum_y_squared: torch.Tensor,
    sum_xy: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct Pearson correlation from sufficient statistics."""
    safe_count = count.clamp_min(1.0)

    covariance_numerator = sum_xy - (sum_x * sum_y / safe_count)
    variance_x_numerator = sum_x_squared - sum_x.square() / safe_count
    variance_y_numerator = sum_y_squared - sum_y.square() / safe_count

    denominator = torch.sqrt(
        variance_x_numerator.clamp_min(0.0)
        * variance_y_numerator.clamp_min(0.0)
    )

    correlation = covariance_numerator / denominator
    invalid = (count < 2) | (denominator <= 0.0)
    correlation = correlation.masked_fill(invalid, torch.nan)

    return correlation.clamp(min=-1.0, max=1.0)


def compute_return_autocorrelation(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    max_lag: int = 60,
    kind: str = "return",
) -> "pd.DataFrame":
    """
    Compute per-asset intraday autocorrelation without crossing sessions.

    The one-minute log-return series is calculated independently inside each
    selected session. Lagged sufficient statistics are then accumulated across
    sessions for each asset. Overnight transitions are never included.

    ``kind`` controls the analysed series:

    - ``"return"``: signed one-minute log returns;
    - ``"absolute"``: absolute one-minute log returns;
    - ``"squared"``: squared one-minute log returns.

    Returns a tidy DataFrame with one row per asset and lag. A separate plotting
    function uses this data to produce the notebook figure.
    """
    import pandas as pd

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel must be one of {split['channels']}, got {channel!r}."
        )

    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, asset_labels = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    first_x, _, first_day = split["samples"][sample_ids[0]]
    first_returns = compute_log_returns(
        x=first_x,
        split=split,
        channels=[channel],
    )

    if max_lag >= first_returns.shape[0]:
        raise ValueError(
            f"max_lag={max_lag} must be smaller than the number of "
            f"within-session returns ({first_returns.shape[0]}) in sample "
            f"{sample_ids[0]} ({first_day})."
        )

    num_lags = max_lag + 1
    num_assets = len(asset_ids)

    count = torch.zeros(num_lags, num_assets, dtype=torch.float64)
    sum_x = torch.zeros_like(count)
    sum_y = torch.zeros_like(count)
    sum_x_squared = torch.zeros_like(count)
    sum_y_squared = torch.zeros_like(count)
    sum_xy = torch.zeros_like(count)

    for sample_idx in sample_ids:
        x, _, day = split["samples"][sample_idx]
        returns = compute_log_returns(
            x=x,
            split=split,
            channels=[channel],
        ).double()
        returns = _transform_return_series(
            returns[:, asset_ids],
            kind=kind,
        )

        if max_lag >= returns.shape[0]:
            raise ValueError(
                f"max_lag={max_lag} is invalid for sample "
                f"{sample_idx} ({day}) with {returns.shape[0]} returns."
            )

        for lag in range(num_lags):
            if lag == 0:
                lag_x = returns
                lag_y = returns
            else:
                lag_x = returns[:-lag]
                lag_y = returns[lag:]

            finite = torch.isfinite(lag_x) & torch.isfinite(lag_y)
            lag_x = torch.where(finite, lag_x, torch.zeros_like(lag_x))
            lag_y = torch.where(finite, lag_y, torch.zeros_like(lag_y))

            count[lag] += finite.sum(dim=0)
            sum_x[lag] += lag_x.sum(dim=0)
            sum_y[lag] += lag_y.sum(dim=0)
            sum_x_squared[lag] += lag_x.square().sum(dim=0)
            sum_y_squared[lag] += lag_y.square().sum(dim=0)
            sum_xy[lag] += (lag_x * lag_y).sum(dim=0)

    autocorrelation = _correlation_from_sufficient_statistics(
        count=count,
        sum_x=sum_x,
        sum_y=sum_y,
        sum_x_squared=sum_x_squared,
        sum_y_squared=sum_y_squared,
        sum_xy=sum_xy,
    )

    rows: list[dict[str, float | int | str]] = []

    for asset_position, asset_label in enumerate(asset_labels):
        for lag in range(num_lags):
            rows.append(
                {
                    "asset": asset_label,
                    "lag": lag,
                    "autocorrelation": float(
                        autocorrelation[lag, asset_position].item()
                    ),
                    "pair_count": int(count[lag, asset_position].item()),
                    "kind": kind,
                }
            )

    return pd.DataFrame(rows)


def _summarise_return_autocorrelation(
    asset_acf: pd.DataFrame,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> pd.DataFrame:
    """Aggregate per-asset autocorrelations across the asset dimension."""

    grouped = asset_acf.groupby("lag", sort=True)["autocorrelation"]
    return pd.DataFrame(
        {
            "lag": grouped.mean().index.to_numpy(),
            "mean": grouped.mean().to_numpy(),
            "median": grouped.median().to_numpy(),
            "band_lower": grouped.quantile(lower_quantile).to_numpy(),
            "band_upper": grouped.quantile(upper_quantile).to_numpy(),
            "num_assets": grouped.count().to_numpy(),
        }
    )


def plot_return_autocorrelation(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    max_lag: int = 60,
    min_lag: int = 0,
    kind: str = "return",
    centre: str = "mean",
    band_quantiles: tuple[float, float] = (0.25, 0.75),
    show_asset_lines: bool = False,
    ax: Axes | Sequence[Axes] | np.ndarray | None = None,
    num_volatility_buckets: int = 1,
    figsize: tuple[float, float] | None = None,
    title: bool = True,
) -> tuple[Figure, Axes | np.ndarray, pd.DataFrame]:
    """Plot intraday autocorrelation, optionally by session volatility.

    Returns are always calculated independently inside each session. For each
    asset and lag, all valid within-session pairs are pooled across the selected
    sessions before Pearson correlation is calculated. The solid line is the
    requested cross-asset mean or median. The shaded band is cross-asset
    dispersion, not a confidence interval.

    When ``num_volatility_buckets`` is greater than one, sessions are ranked by
    their cross-asset mean realised volatility and split into deterministic,
    approximately equal-count buckets from low to high volatility.
    """

    if centre not in {"mean", "median"}:
        raise ValueError("centre must be 'mean' or 'median'.")
    if min_lag < 0 or min_lag > max_lag:
        raise ValueError(
            "min_lag must satisfy 0 <= min_lag <= max_lag."
        )

    lower_quantile, upper_quantile = map(float, band_quantiles)
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "band_quantiles must satisfy 0 <= lower < upper <= 1."
        )

    selected_sample_ids = resolve_sample_indices(split, sample_indices)
    session_volatility = compute_session_market_volatility(
        split,
        channel=channel,
        sample_indices=selected_sample_ids,
        assets=assets,
    )
    bucketed_sessions = _assign_session_volatility_buckets(
        session_volatility,
        num_buckets=num_volatility_buckets,
    )

    if figsize is None:
        figsize = (10.0, 5.0) if num_volatility_buckets == 1 else (
            5.0 * num_volatility_buckets,
            4.5,
        )

    if ax is None:
        fig, created_axes = plt.subplots(
            1,
            num_volatility_buckets,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes_array = created_axes.ravel()
    else:
        if num_volatility_buckets == 1 and isinstance(ax, Axes):
            axes_array = np.asarray([ax], dtype=object)
        else:
            axes_array = np.asarray(ax, dtype=object).reshape(-1)
            if len(axes_array) != num_volatility_buckets:
                raise ValueError(
                    "The number of supplied axes must equal "
                    "num_volatility_buckets."
                )
        fig = axes_array[0].figure

    kind_titles = {
        "return": "Return",
        "returns": "Return",
        "absolute": "Absolute-return",
        "absolute_return": "Absolute-return",
        "absolute_returns": "Absolute-return",
        "squared": "Squared-return",
        "squared_return": "Squared-return",
        "squared_returns": "Squared-return",
    }
    title_prefix = kind_titles.get(kind, kind.replace("_", " ").title())
    summaries: list[pd.DataFrame] = []

    for bucket in range(1, num_volatility_buckets + 1):
        axes = axes_array[bucket - 1]
        bucket_sessions = bucketed_sessions.loc[
            bucketed_sessions["volatility_bucket"] == bucket
        ].sort_values("sample_idx", kind="stable")
        bucket_sample_ids = bucket_sessions["sample_idx"].astype(int).tolist()
        bucket_label = str(bucket_sessions["volatility_label"].iloc[0])

        asset_acf = compute_return_autocorrelation(
            split=split,
            channel=channel,
            sample_indices=bucket_sample_ids,
            assets=assets,
            max_lag=max_lag,
            kind=kind,
        )
        summary = _summarise_return_autocorrelation(
            asset_acf,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        summary = summary.loc[summary["lag"] >= min_lag].copy()
        summary["volatility_bucket"] = bucket
        summary["volatility_label"] = bucket_label
        summary["num_sessions"] = len(bucket_sample_ids)
        summary["volatility_min"] = float(
            bucket_sessions["market_realised_volatility"].min()
        )
        summary["volatility_max"] = float(
            bucket_sessions["market_realised_volatility"].max()
        )
        summaries.append(summary)

        plot_asset_acf = asset_acf.loc[
            asset_acf["lag"] >= min_lag
        ]
        if show_asset_lines:
            for _, asset_frame in plot_asset_acf.groupby(
                "asset",
                sort=False,
            ):
                axes.plot(
                    asset_frame["lag"],
                    asset_frame["autocorrelation"],
                    alpha=0.15,
                    linewidth=0.8,
                )

        axes.fill_between(
            summary["lag"].to_numpy(),
            summary["band_lower"].to_numpy(),
            summary["band_upper"].to_numpy(),
            alpha=0.2,
            label=(
                f"Cross-asset {100 * lower_quantile:.0f}–"
                f"{100 * upper_quantile:.0f}% interval"
            ),
        )
        axes.plot(
            summary["lag"],
            summary[centre],
            linewidth=2,
            label=f"Cross-asset {centre}",
        )
        axes.axhline(0.0, linewidth=1)
        axes.set_xlabel("Lag (minutes)")
        axes.set_xlim(min_lag, max_lag)
       

        if bucket == 1:
            axes.set_ylabel("Autocorrelation")
        if title:
            axes.set_title(
                f"{title_prefix} autocorrelation"
                if num_volatility_buckets == 1
                else bucket_label
            )

    fig.tight_layout()
    result_axes: Axes | np.ndarray = (
        axes_array[0]
        if num_volatility_buckets == 1
        else axes_array
    )
    return fig, result_axes, pd.concat(summaries, ignore_index=True)

def _collect_scale_return_statistics(
    split: SplitDict,
    sample_ids: Sequence[int],
    asset_ids: Sequence[int],
    channel: str,
    scales: Sequence[int],
    overlapping: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accumulate per-asset return sums, squared sums, and counts by scale."""
    num_scales = len(scales)
    num_assets = len(asset_ids)

    count = torch.zeros(num_scales, num_assets, dtype=torch.float64)
    value_sum = torch.zeros_like(count)
    squared_sum = torch.zeros_like(count)

    for sample_idx in sample_ids:
        x, _, day = split["samples"][sample_idx]
        returns = compute_log_returns(
            x=x,
            split=split,
            channels=[channel],
        ).double()
        returns = returns[:, asset_ids]

        cumulative_returns = torch.cat(
            [
                torch.zeros(
                    1,
                    returns.shape[1],
                    dtype=returns.dtype,
                    device=returns.device,
                ),
                returns.cumsum(dim=0),
            ],
            dim=0,
        )

        for scale_position, scale in enumerate(scales):
            if scale > returns.shape[0]:
                raise ValueError(
                    f"Scale {scale} is larger than the number of returns "
                    f"in sample {sample_idx} ({day})."
                )

            scale_returns = (
                cumulative_returns[scale:]
                - cumulative_returns[:-scale]
            )

            if not overlapping:
                scale_returns = scale_returns[::scale]

            finite = torch.isfinite(scale_returns)
            safe_values = torch.where(
                finite,
                scale_returns,
                torch.zeros_like(scale_returns),
            )

            count[scale_position] += finite.sum(dim=0)
            value_sum[scale_position] += safe_values.sum(dim=0)
            squared_sum[scale_position] += safe_values.square().sum(dim=0)

    return count, value_sum, squared_sum


def compute_variance_ratios(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    horizons: Sequence[int] = (1, 5, 15, 30, 60),
    overlapping: bool = True,
    return_asset_values: bool = False,
) -> "pd.DataFrame":
    """
    Compute intraday variance ratios at configurable horizons.

    For asset ``i`` and horizon ``h``:

        VR_i(h) = Var(r_i,t:t+h) / (h * Var(r_i,t))

    Multi-minute returns are formed only inside each session. By default they
    use all overlapping windows. The returned summary aggregates the fixed
    asset universe; set ``return_asset_values=True`` for one row per asset and
    horizon.
    """
    import pandas as pd

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel must be one of {split['channels']}, got {channel!r}."
        )

    resolved_horizons = _validate_positive_integers(
        horizons,
        "horizons",
    )
    scales = sorted(set([1, *resolved_horizons]))

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, asset_labels = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    count, value_sum, squared_sum = _collect_scale_return_statistics(
        split=split,
        sample_ids=sample_ids,
        asset_ids=asset_ids,
        channel=channel,
        scales=scales,
        overlapping=overlapping,
    )

    safe_count = count.clamp_min(1.0)
    variance = squared_sum / safe_count - (value_sum / safe_count).square()
    variance = variance.clamp_min(0.0)

    scale_to_position = {
        scale: position
        for position, scale in enumerate(scales)
    }
    one_minute_variance = variance[scale_to_position[1]]

    rows: list[dict[str, float | int | str | bool]] = []

    for horizon in resolved_horizons:
        horizon_position = scale_to_position[horizon]
        denominator = horizon * one_minute_variance
        ratio = variance[horizon_position] / denominator
        ratio = ratio.masked_fill(denominator <= 0.0, torch.nan)

        for asset_position, asset_label in enumerate(asset_labels):
            rows.append(
                {
                    "asset": asset_label,
                    "horizon": horizon,
                    "variance_ratio": float(ratio[asset_position].item()),
                    "one_minute_observations": int(
                        count[scale_to_position[1], asset_position].item()
                    ),
                    "horizon_observations": int(
                        count[horizon_position, asset_position].item()
                    ),
                    "overlapping": overlapping,
                }
            )

    asset_values = pd.DataFrame(rows)

    if return_asset_values:
        return asset_values

    grouped = asset_values.groupby("horizon", sort=False)["variance_ratio"]
    summary = pd.DataFrame(
        {
            "horizon": grouped.mean().index.to_numpy(),
            "mean_variance_ratio": grouped.mean().to_numpy(),
            "median_variance_ratio": grouped.median().to_numpy(),
            "std_variance_ratio": grouped.std(ddof=0).to_numpy(),
            "q25_variance_ratio": grouped.quantile(0.25).to_numpy(),
            "q75_variance_ratio": grouped.quantile(0.75).to_numpy(),
            "min_variance_ratio": grouped.min().to_numpy(),
            "max_variance_ratio": grouped.max().to_numpy(),
            "num_assets": grouped.count().to_numpy(),
        }
    )
    summary["overlapping"] = overlapping

    return summary


def compute_variance_scaling_hurst(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    scales: Sequence[int] = (1, 2, 5, 10, 15, 30, 60),
    overlapping: bool = True,
    min_scales: int = 3,
) -> "pd.DataFrame":
    """
    Estimate a supplementary variance-scaling Hurst exponent per asset.

    The estimator fits:

        log Var(r_h) = intercept + slope * log(h)
        H = slope / 2

    where every cumulative return is formed within a session. The result should
    be treated as a scale-dependent diagnostic, not as the primary measure of
    persistence.
    """
    import pandas as pd

    if channel not in split["channels"]:
        raise ValueError(
            f"Channel must be one of {split['channels']}, got {channel!r}."
        )

    resolved_scales = _validate_positive_integers(scales, "scales")

    if min_scales < 2:
        raise ValueError("min_scales must be at least 2.")

    if min_scales > len(resolved_scales):
        raise ValueError(
            "min_scales cannot exceed the number of supplied scales."
        )

    sample_ids = resolve_sample_indices(split, sample_indices)
    asset_ids, asset_labels = resolve_asset_indices(split, assets)

    if len(sample_ids) == 0:
        raise ValueError("No sample indices selected.")

    if len(asset_ids) == 0:
        raise ValueError("No assets selected.")

    count, value_sum, squared_sum = _collect_scale_return_statistics(
        split=split,
        sample_ids=sample_ids,
        asset_ids=asset_ids,
        channel=channel,
        scales=resolved_scales,
        overlapping=overlapping,
    )

    safe_count = count.clamp_min(1.0)
    variance = squared_sum / safe_count - (value_sum / safe_count).square()
    variance = variance.clamp_min(0.0)

    log_scales = np.log(np.asarray(resolved_scales, dtype=np.float64))
    rows: list[dict[str, float | int | str | bool]] = []

    for asset_position, asset_label in enumerate(asset_labels):
        asset_variance = variance[:, asset_position].cpu().numpy()
        asset_count = count[:, asset_position].cpu().numpy()

        valid = (
            np.isfinite(asset_variance)
            & (asset_variance > 0.0)
            & (asset_count >= 2)
        )

        if int(valid.sum()) < min_scales:
            rows.append(
                {
                    "asset": asset_label,
                    "hurst": np.nan,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_squared": np.nan,
                    "num_scales": int(valid.sum()),
                    "overlapping": overlapping,
                }
            )
            continue

        x_values = log_scales[valid]
        y_values = np.log(asset_variance[valid])

        slope, intercept = np.polyfit(x_values, y_values, deg=1)
        fitted = intercept + slope * x_values

        residual_sum_squares = float(
            np.square(y_values - fitted).sum()
        )
        total_sum_squares = float(
            np.square(y_values - y_values.mean()).sum()
        )

        if total_sum_squares > 0.0:
            r_squared = 1.0 - residual_sum_squares / total_sum_squares
        else:
            r_squared = np.nan

        rows.append(
            {
                "asset": asset_label,
                "hurst": float(slope / 2.0),
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "num_scales": int(valid.sum()),
                "overlapping": overlapping,
            }
        )

    return pd.DataFrame(rows)

def plot_return_histogram(
    split: SplitDict,
    channel: str = "close",
    sample_indices: int | Sequence[int] | slice | None = None,
    assets: str | int | Sequence[str | int] | None = None,
    bins: int = 300,
    scale_to_basis_points: bool = True,
    show_normal: bool = True,
    log_y: bool = True,
    title: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, np.ndarray, pd.DataFrame]:
    """
    Plot the pooled distribution of one-minute log returns and calculate
    their mean and excess kurtosis.

    Returns are calculated separately within each selected trading session,
    so no overnight return is introduced. The observations are then pooled
    across all selected sessions and stocks.

    Excess kurtosis uses Fisher's definition, under which a normal
    distribution has excess kurtosis equal to zero.

    Parameters
    ----------
    split:
        Cleaned candle-data split.

    channel:
        Channel from which log returns are calculated.

    sample_indices:
        Sessions to include. None includes every session in the split.

    assets:
        Stocks to include. None includes every stock in the split.

    bins:
        Number of histogram bins.

    scale_to_basis_points:
        If True, display returns in basis points.

    show_normal:
        If True, overlay a normal density with the same mean and standard
        deviation as the pooled observations.

    log_y:
        If True, use a logarithmic y-axis.

    title:
        If False, omit the plot title.

    figsize:
        Figure size when an existing axis is not supplied.

    ax:
        Optional existing Matplotlib axis.

    Returns
    -------
    figure:
        Matplotlib figure.

    axes:
        Matplotlib axis.

    pooled_returns:
        One-dimensional array of pooled returns in raw log-return units.

    return_summary:
        Two-row table containing the mean return and excess kurtosis.
    """
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")

    returns_array, _ = collect_channel_log_returns(
        split=split,
        channel=channel,
        sample_indices=sample_indices,
        assets=assets,
    )

    pooled_returns = np.asarray(
        returns_array,
        dtype=np.float64,
    ).reshape(-1)

    pooled_returns = pooled_returns[
        np.isfinite(pooled_returns)
    ]

    if pooled_returns.size == 0:
        raise ValueError(
            "No finite return observations were available."
        )

    mean_log_return = float(np.mean(pooled_returns))

    # Pandas uses Fisher's definition, so a normal distribution has
    # excess kurtosis equal to zero.
    excess_kurtosis = float(
        pd.Series(pooled_returns).kurt()
    )

    return_summary = pd.DataFrame(
        {
            "Statistic": [
                "Mean one-minute log return",
                "Excess kurtosis",
            ],
            "Value": [
                mean_log_return * 10_000.0,
                excess_kurtosis,
            ],
            "Unit": [
                "basis points",
                "",
            ],
        }
    )

    if scale_to_basis_points:
        displayed_returns = pooled_returns * 10_000.0
        x_label = (
            f"One-minute {channel} log return "
            "(basis points)"
        )
    else:
        displayed_returns = pooled_returns
        x_label = f"One-minute {channel} log return"

    if ax is None:
        figure, axes = plt.subplots(figsize=figsize)
        created_figure = True
    else:
        axes = ax
        figure = axes.figure
        created_figure = False

    _, bin_edges, _ = axes.hist(
        displayed_returns,
        bins=bins,
        density=True,
        alpha=0.75,
        label="Empirical returns",
    )

    if show_normal:
        displayed_mean = float(
            np.mean(displayed_returns)
        )
        displayed_std = float(
            np.std(displayed_returns, ddof=1)
        )

        if np.isfinite(displayed_std) and displayed_std > 0.0:
            normal_x = np.linspace(
                float(bin_edges[0]),
                float(bin_edges[-1]),
                1_000,
            )

            normal_density = (
                np.exp(
                    -0.5
                    * (
                        (normal_x - displayed_mean)
                        / displayed_std
                    )
                    ** 2
                )
                / (
                    displayed_std
                    * np.sqrt(2.0 * np.pi)
                )
            )

            axes.plot(
                normal_x,
                normal_density,
                linewidth=1.8,
                label="Matched normal distribution",
            )

    axes.axvline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    if log_y:
        axes.set_yscale("log")

    if title:
        axes.set_title(
            f"Distribution of one-minute {channel} log returns"
        )

    axes.set_xlabel(x_label)
    axes.set_ylabel("Probability density")
    axes.legend(fontsize=8)
    axes.grid(True, alpha=0.2)

    if created_figure:
        figure.tight_layout()

    return (
        figure,
        axes,
        pooled_returns,
        return_summary,
    )
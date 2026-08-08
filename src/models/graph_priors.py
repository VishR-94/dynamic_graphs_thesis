from __future__ import annotations

"""Reusable graph-prior builders for financial asset graphs."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch import Tensor


def _row_normalise_nonnegative(values: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    matrix = torch.as_tensor(values).detach().cpu().float().clone()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Graph prior must have square shape [N,N].")
    if not torch.isfinite(matrix).all() or torch.any(matrix < 0):
        raise ValueError("Graph prior must be finite and non-negative.")
    matrix.fill_diagonal_(0.0)
    num_nodes = int(matrix.shape[0])
    row_mass = matrix.sum(dim=-1, keepdim=True)
    empty = row_mass.squeeze(-1) <= eps
    if torch.any(empty):
        fallback = torch.ones(num_nodes, num_nodes, dtype=matrix.dtype)
        fallback.fill_diagonal_(0.0)
        fallback = fallback / fallback.sum(dim=-1, keepdim=True)
        matrix[empty] = fallback[empty]
        row_mass = matrix.sum(dim=-1, keepdim=True)
    return (matrix / row_mass.clamp_min(eps)).contiguous()


def build_sector_graph_prior(
    asset_cols: Sequence[str],
    company_profiles_path: str | Path,
) -> tuple[Tensor, list[str]]:
    """Build a row-normalised same-sector, non-self prior.

    The dissertation file contract is preserved: column 1 contains ticker and
    column 6 contains sector. Column names are not required.
    """

    path = Path(company_profiles_path).expanduser().resolve()
    frame = pd.read_csv(path)
    if frame.shape[1] < 6:
        raise ValueError("company_profiles.csv must contain at least six columns.")
    ticker_column = frame.columns[0]
    sector_column = frame.columns[5]
    mapping = {
        str(ticker).strip().upper(): str(sector).strip()
        for ticker, sector in zip(
            frame[ticker_column],
            frame[sector_column],
            strict=False,
        )
        if pd.notna(ticker) and pd.notna(sector)
    }
    tickers = [str(value).strip().upper() for value in asset_cols]
    missing = [ticker for ticker in tickers if ticker not in mapping]
    if missing:
        raise KeyError(
            "company_profiles.csv is missing sectors for: "
            + ", ".join(missing)
        )
    sectors = [mapping[ticker] for ticker in tickers]
    num_nodes = len(tickers)
    prior = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for target, target_sector in enumerate(sectors):
        for source, source_sector in enumerate(sectors):
            if target != source and target_sector == source_sector:
                prior[target, source] = 1.0
    return _row_normalise_nonnegative(prior), sectors


def build_absolute_correlation_graph_prior(
    clean_training_split: Mapping[str, Any],
    *,
    expected_asset_cols: Sequence[str],
    eps: float = 1.0e-12,
) -> Tensor:
    """Build a training-only absolute one-minute Close-return prior.

    The diagonal is removed, no threshold is applied, and rows are normalised.
    A zero-variance asset falls back to a uniform non-self row.
    """

    asset_cols = list(clean_training_split.get("asset_cols", []))
    if asset_cols != list(expected_asset_cols):
        raise ValueError("Training split asset order differs from expected_asset_cols.")
    channels = [str(value).lower() for value in clean_training_split["channels"]]
    if "close" not in channels:
        raise KeyError("Training split does not contain Close.")
    close_index = channels.index("close")

    parts: list[Tensor] = []
    for sample in clean_training_split["samples"]:
        values = torch.as_tensor(sample[0]).detach().cpu().double()
        close = values[..., close_index].clamp_min(eps)
        if int(close.shape[0]) >= 2:
            parts.append(close[1:].log() - close[:-1].log())
    if not parts:
        raise ValueError("No within-session Close returns are available.")

    returns = torch.cat(parts, dim=0)
    centred = returns - returns.mean(dim=0, keepdim=True)
    covariance = centred.transpose(0, 1) @ centred
    variance = centred.square().sum(dim=0)
    denominator = torch.sqrt(variance[:, None] * variance[None, :])
    correlation = torch.where(
        denominator > eps,
        covariance / denominator.clamp_min(eps),
        torch.zeros_like(covariance),
    ).abs()
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    return _row_normalise_nonnegative(correlation.float(), eps=eps)

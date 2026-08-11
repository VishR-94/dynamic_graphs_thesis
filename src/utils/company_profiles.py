from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def resolve_company_profiles_path(
    path: str | Path | None = None,
) -> Path:
    """Resolve ``company_profiles.csv`` on the author's Mac or in Colab."""

    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved

    candidates = (
        Path(
            "/Users/vishalruparelia/Library/CloudStorage/"
            "GoogleDrive-vishal@autonomous-fox.ai/My Drive/"
            "dissertation/company_profiles.csv"
        ),
        Path("/content/drive/MyDrive/dissertation/company_profiles.csv"),
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate dissertation/company_profiles.csv. Checked:\n"
        + "\n".join(f"  - {candidate}" for candidate in candidates)
    )


def make_asset_sector_mapping(
    assets: Sequence[str],
    *,
    company_profiles_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return one validated sector label for every asset, in input order.

    The dissertation file contract is preserved: column 1 contains the ticker
    and column 6 contains the sector. Column names are not required.
    """

    path = resolve_company_profiles_path(company_profiles_path)
    profiles = pd.read_csv(path)

    if profiles.shape[1] < 6:
        raise ValueError(
            "company_profiles.csv must contain at least six columns."
        )

    ticker_column = profiles.columns[0]
    sector_column = profiles.columns[5]

    selected = profiles[[ticker_column, sector_column]].copy()
    selected.columns = ["Ticker", "Sector"]
    selected["Ticker"] = (
        selected["Ticker"].astype(str).str.upper().str.strip()
    )
    selected["Sector"] = selected["Sector"].astype(str).str.strip()

    normalised_assets = [str(asset).upper().strip() for asset in assets]
    selected = selected.loc[selected["Ticker"].isin(normalised_assets)]

    duplicated = (
        selected.loc[
            selected["Ticker"].duplicated(keep=False),
            "Ticker",
        ]
        .unique()
        .tolist()
    )
    if duplicated:
        raise ValueError(
            "Company profile contains duplicate project tickers: "
            f"{duplicated}"
        )

    sector_by_ticker = selected.set_index("Ticker")["Sector"].to_dict()
    missing = [
        str(asset)
        for asset, ticker in zip(assets, normalised_assets, strict=True)
        if ticker not in sector_by_ticker
    ]
    if missing:
        raise ValueError(
            f"No sector was found for project assets: {missing}"
        )

    return pd.DataFrame(
        {
            "Ticker": [str(asset) for asset in assets],
            "Sector": [sector_by_ticker[ticker] for ticker in normalised_assets],
        }
    )


def make_sector_group_order(
    assets: Sequence[str],
    *,
    company_profiles_path: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return Graph-Hub-compatible sector/ticker display ordering.

    Sectors are ordered alphabetically and tickers are ordered alphabetically
    within each sector. The returned integer array indexes the original asset
    order. The returned mapping is already in display order.
    """

    mapping = make_asset_sector_mapping(
        assets,
        company_profiles_path=company_profiles_path,
    ).copy()
    mapping["Original position"] = np.arange(len(mapping), dtype=np.int64)
    ordered = mapping.sort_values(
        ["Sector", "Ticker"],
        kind="stable",
    ).reset_index(drop=True)
    order = ordered["Original position"].to_numpy(dtype=np.int64)
    return order, ordered[["Ticker", "Sector"]].copy()

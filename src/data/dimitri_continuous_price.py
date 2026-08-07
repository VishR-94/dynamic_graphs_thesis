from __future__ import annotations

"""Continuous-candle windows for the Dimitri BaseDyGraph-V2 price task.

This is the controlled continuous-input counterpart of
``src.data.dimitri_token_price``.  It uses the same session membership,
window origins, context-only normalisation, zero-Amount convention, clipping,
and dense teacher-forced one-step target sequence as the token-input run.  The
only change is the model input representation:

* token run: frozen Kronos ``s1`` IDs;
* continuous run: the pre-tokenisation normalised OHLCVA values themselves.

For a context length ``C`` and continuation length ``P``, each item contains a
causal ``C + P`` sequence.  Mean and sample standard deviation are calculated
from the first ``C`` rows only and applied to the full sequence.  Position ``t``
is used to predict Close at ``t+1``; the public one-minute forecast is the
transition from position ``C-1`` to ``C``.
"""

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from src.data.dimitri_anchor_tokens import (
    DIMITRI_CHANNELS,
    DIMITRI_DROP_OPEN_ROWS,
    DIMITRI_EXPECTED_ASSETS,
)
from src.data.dimitri_token_price import (
    DimitriTokenPriceWindowSpec,
    exact_window_starts,
    load_token_price_splits,
    normalise_split_mode,
)


DIMITRI_CONTINUOUS_PRICE_CONTRACT = (
    "dimitri_basedygraph_v2_continuous_input_direct_price_v1"
)


class DimitriContinuousPriceDataset(Dataset[dict[str, Any]]):
    """Create normalised OHLCVA windows lazily from one selected split.

    Returned tensor shapes for one item are:

    ``continuous_values``
        ``[N, C+P, 6]`` context-normalised OHLCVA. Amount is identically zero.

    ``raw_close``
        ``[N, C+P]`` unnormalised Close values used for the dense one-step loss
        and the public one-minute forecast.

    ``close_mean`` / ``close_std``
        ``[N]`` Close statistics from the first ``C`` rows only.
    """

    def __init__(
        self,
        *,
        raw_split: Mapping[str, Any],
        split_name: str,
        spec: DimitriTokenPriceWindowSpec,
        split_mode: str,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.split_name = str(split_name)
        self.split_mode = normalise_split_mode(split_mode)
        self.asset_cols = [str(value) for value in raw_split["asset_cols"]]
        self.channels = [str(value).lower() for value in raw_split["channels"]]

        if len(self.asset_cols) != DIMITRI_EXPECTED_ASSETS:
            raise ValueError(
                f"Expected {DIMITRI_EXPECTED_ASSETS} assets; "
                f"observed {len(self.asset_cols)}."
            )
        if tuple(self.channels) != DIMITRI_CHANNELS:
            raise ValueError(
                f"Continuous channel order {tuple(self.channels)} differs from "
                f"the Dimitri/Kronos contract {DIMITRI_CHANNELS}."
            )

        self.close_index = self.channels.index("close")
        self.amount_index = self.channels.index("amount")
        self.clean_sessions: list[torch.Tensor] = []
        self.session_dates: list[str] = []
        self.sample_indices: list[int] = []
        self.window_starts: list[int] = []
        self.window_dates: list[str] = []

        for sample_index, sample in enumerate(raw_split["samples"]):
            candle, _auxiliary, session_date = sample[:3]
            values = torch.as_tensor(candle).detach().cpu().float()
            if values.ndim != 3 or tuple(values.shape[1:]) != (
                DIMITRI_EXPECTED_ASSETS,
                len(DIMITRI_CHANNELS),
            ):
                raise ValueError(
                    "Raw session must have shape [T,93,6], got "
                    f"{tuple(values.shape)}."
                )
            if values.shape[0] <= DIMITRI_DROP_OPEN_ROWS:
                raise ValueError("Raw session is too short after dropping its first row.")
            values = values[DIMITRI_DROP_OPEN_ROWS:].contiguous()
            if not torch.isfinite(values).all():
                raise ValueError(f"Session {session_date} contains non-finite values.")

            starts = exact_window_starts(int(values.shape[0]), spec)
            if not starts:
                raise ValueError(
                    f"No {spec.sequence_length}-bar window fits in session "
                    f"{session_date} of length {values.shape[0]}."
                )

            self.clean_sessions.append(values)
            self.session_dates.append(str(session_date))
            self.sample_indices.extend([sample_index] * len(starts))
            self.window_starts.extend(starts)
            self.window_dates.extend([str(session_date)] * len(starts))

        if not self.sample_indices:
            raise ValueError(f"Selected {self.split_name} split produced no windows.")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = int(self.sample_indices[index])
        start = int(self.window_starts[index])
        stop = start + self.spec.sequence_length

        # [L,N,6] -> [N,L,6]. Clone before zeroing Amount so the stored raw
        # session remains unchanged for other windows.
        window = (
            self.clean_sessions[sample_index][start:stop]
            .permute(1, 0, 2)
            .contiguous()
            .clone()
        )
        raw_close = window[..., self.close_index].clone()  # [N,L]
        window[..., self.amount_index] = 0.0

        context = window[:, : self.spec.context_length]
        mean = context.mean(dim=1)  # [N,6]
        std = context.std(dim=1, correction=1)  # exact Dimitri token contract
        normalised = (
            (window - mean[:, None, :]) / (std[:, None, :] + self.spec.eps)
        ).clamp(-self.spec.clip, self.spec.clip)

        if not torch.isfinite(normalised).all():
            raise FloatingPointError("Continuous normalised window is non-finite.")
        if torch.count_nonzero(normalised[..., self.amount_index]) != 0:
            raise AssertionError("Amount must remain exactly zero after normalisation.")

        return {
            "continuous_values": normalised.float().contiguous(),
            "raw_close": raw_close.float().contiguous(),
            "close_mean": mean[:, self.close_index].float().contiguous(),
            "close_std": std[:, self.close_index].float().contiguous(),
            "sample_idx": torch.tensor(sample_index, dtype=torch.long),
            "window_start": torch.tensor(start, dtype=torch.long),
            "window_date": self.window_dates[index],
        }


def build_continuous_price_datasets(
    data_dir: str | Path,
    *,
    split_mode: str,
    spec: DimitriTokenPriceWindowSpec,
) -> tuple[dict[str, dict[str, Any]], dict[str, DimitriContinuousPriceDataset]]:
    """Load selected memberships and construct train/val/test datasets."""
    mode = normalise_split_mode(split_mode)
    raw_splits = load_token_price_splits(data_dir, split_mode=mode)
    datasets = {
        split: DimitriContinuousPriceDataset(
            raw_split=raw_splits[split],
            split_name=split,
            split_mode=mode,
            spec=spec,
        )
        for split in ("train", "val", "test")
    }
    reference_assets = datasets["train"].asset_cols
    if not all(dataset.asset_cols == reference_assets for dataset in datasets.values()):
        raise ValueError("Continuous split asset orders differ.")
    return raw_splits, datasets

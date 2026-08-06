"""
Dataset and Lightning DataModule for discrete state-token sequences.

Batch format expected by the model and evaluation code:
    {"state_ids": (N, T)}                   collated -> (B, N, T)
    {"state_ids": (N, T), "regimes": (T,)}  collated -> + (B, T)
"""

from typing import Dict, Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


class DiscreteStateSequenceDataset(Dataset):
    """Dataset of discrete state-token sequences over a fixed set of assets.

    Args:
        state_ids: Integer state IDs of shape (num_examples, N, T), where N is the
            number of assets and T the sequence length.
        regimes: Optional per-timestep regime labels of shape (num_examples, T).

    Each item is a dict with ``state_ids`` of shape (N, T) and, when regimes are
    provided, ``regimes`` of shape (T,).
    """

    def __init__(self, state_ids: torch.Tensor, regimes: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if state_ids.ndim != 3:
            raise ValueError("state_ids must have shape (num_examples, N, T)")
        if regimes is not None and regimes.ndim != 2:
            raise ValueError("regimes must have shape (num_examples, T)")
        self.state_ids = state_ids.long()
        self.regimes = regimes.long() if regimes is not None else None

    def __len__(self) -> int:
        return self.state_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"state_ids": self.state_ids[idx]}
        if self.regimes is not None:
            item["regimes"] = self.regimes[idx]
        return item


class DiscreteStateDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping train/val/test tensors of discrete state IDs.

    Each split tensor has shape (num_examples, N, T); the optional regime tensors
    have shape (num_examples, T). Batches are collated to (B, N, T) for state IDs
    and (B, T) for regimes.
    """

    def __init__(
        self,
        train_tensor: torch.Tensor,
        val_tensor: torch.Tensor,
        test_tensor: Optional[torch.Tensor] = None,
        train_regimes: Optional[torch.Tensor] = None,
        val_regimes: Optional[torch.Tensor] = None,
        test_regimes: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.train_tensor = train_tensor
        self.val_tensor = val_tensor
        self.test_tensor = test_tensor
        self.train_regimes = train_regimes
        self.val_regimes = val_regimes
        self.test_regimes = test_regimes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = DiscreteStateSequenceDataset(self.train_tensor, self.train_regimes)
        self.val_ds = DiscreteStateSequenceDataset(self.val_tensor, self.val_regimes)
        self.test_ds = (
            DiscreteStateSequenceDataset(self.test_tensor, self.test_regimes)
            if self.test_tensor is not None else None
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("No test dataset was provided")
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

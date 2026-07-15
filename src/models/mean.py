from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.data_generator import WindowedCandleDataset

SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]

'''
Implement a simple mean benchmark. This works as follows:
For each window, we predict all values of the horizon to simply be the average values over the window.
For example, if our window is length 60 and horizons are [1,5,15,30,60], our predictions (in either prices or 
log change) for every horizon is just the average value in the window per asset per channel. 
'''

class MeanBaseline:
    """
    Raw-window mean baseline.

    This predicts that every future horizon is equal to the average target
    value in the current context window:

        prediction[h] = mean(context window)

    Predictions and ground truth are returned in raw value space.
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int = 1,
    ) -> None:
        self.context_length = context_length
        self.horizons = horizons
        self.target_channels = target_channels
        self.stride = stride
        self.train_split: SplitDict | None = None
        self.val_split: SplitDict | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MeanBaseline":
        forecasting_config = config["forecasting"]

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=list(forecasting_config["horizons"]),
            target_channels=list(forecasting_config["target_channels"]),
            stride = int(forecasting_config['stride']),
        )

    def fit(
        self,
        train_split: SplitDict,
        val_split: SplitDict | None = None,
    ) -> "MeanBaseline":
        """
        Store train and validation splits for interface consistency.

        The mean baseline has no parameters to fit.
        """
        self.train_split = train_split
        self.val_split = val_split

        return self

    def fitted_values(
        self,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Return mean predictions on the training split.
        """
        if self.train_split is None:
            raise ValueError("Call fit(...) before fitted_values().")

        return self.predict(
            split=self.train_split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict(
        self,
        split: SplitDict,
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Generate raw mean predictions.

        Args:
            split:
                Cleaned raw candle split.

            batch_size:
                DataLoader batch size.

            num_workers:
                Number of DataLoader workers.

        Returns:
            Dictionary containing raw y_pred and y_true tensors with shape:
                [num_examples, num_horizons, num_assets, num_channels]
        """

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
        )

        all_y_pred = []
        all_y_true = []
        all_sample_idx = []
        all_origin_idx = []
        all_target_indices = []
        all_last_context_target = []

        for batch in loader:
            #has shape [B,context_length,N,C]
            x_context = batch["x"].float()
            y_true_raw = batch["y"].float()
            last_context_target = batch["last_context_target"].float()
            

            #has shape [B,N,C] -> averaged over time dimension
            context_mean = x_context.mean(dim=1)

            y_pred_raw = context_mean.unsqueeze(1).repeat(
                1,
                len(self.horizons),
                1,
                1,
            )

            all_last_context_target.append(last_context_target)
            all_y_pred.append(y_pred_raw)
            all_y_true.append(y_true_raw)
            all_sample_idx.append(batch["sample_idx"])
            all_origin_idx.append(batch["origin_idx"])
            all_target_indices.append(batch["target_indices"])

        y_pred = torch.cat(all_y_pred, dim=0)
        y_true = torch.cat(all_y_true, dim=0)
        last_context_target = torch.cat(all_last_context_target,dim=0)

        sample_idx = torch.cat(all_sample_idx, dim=0)
        origin_idx = torch.cat(all_origin_idx, dim=0)
        target_indices = torch.cat(all_target_indices, dim=0)

        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "channels": self.target_channels,
            "horizons": self.horizons,
            "sample_idx": sample_idx,
            "origin_idx": origin_idx,
            "target_indices": target_indices,
            "last_context_target": last_context_target,
        }

    def _dataset_config(self) -> dict[str, Any]:
        """
        Build a minimal config for WindowedCandleDataset.

        The mean baseline requires the target channels across the complete
        context window.
        """
        return {
            "forecasting": {
                "context_length": self.context_length,
                "horizons": self.horizons,
                "stride": self.stride,
                "input_channels": self.target_channels,
                "target_channels": self.target_channels,
            }
        }
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data.data_generator import WindowedCandleDataset
from src.evaluation.prediction_transforms import raw_to_cumulative_log_change

SplitDict = dict[str, Any]
PredictionDict = dict[str, Any]

'''
Implement a simple persistence benchmark. This works as follows:
For each window, we predict all values of the horizon to simple be the values at the last point in the window.
For example, if our window is length 60 and horizons are [1,5,15,30,60], our predictions (in either prices or 
log change) for every horizon is just the values in the final available minute in our window for all assets and
all channels. 
'''

class PersistenceBaseline:
    """
    Raw-price persistence baseline.

    This predicts that every future horizon is equal to the last observed value
    in the context window.

    In raw space:
        prediction[h] = last_context_target

    In cumulative log-change space:
        prediction[h] = 0
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
    def from_config(cls, config: dict[str, Any]) -> "PersistenceBaseline":
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
    ) -> "PersistenceBaseline":
        """
        Store train/validation splits for interface consistency.

        Persistence has no parameters to fit.
        """
        self.train_split = train_split
        self.val_split = val_split

        return self

    def fitted_values(
        self,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Return persistence predictions on the training split.
        """
        if self.train_split is None:
            raise ValueError("Call fit(...) before fitted_values().")

        return self.predict(
            split=self.train_split,
            output_space=output_space,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def predict(
        self,
        split: SplitDict,
        output_space: str = "cumulative_log_change",
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> PredictionDict:
        """
        Generate persistence predictions.

        Args:
            split:
                Cleaned raw candle split.

            output_space:
                Either:
                    "raw"
                    "cumulative_log_change"

            batch_size:
                DataLoader batch size.

            num_workers:
                DataLoader workers.

        Returns:
            Dictionary containing y_pred and y_true with shape:
                [num_examples, num_horizons, num_assets, num_channels]
        """
        if output_space not in {"raw", "cumulative_log_change"}:
            raise ValueError(
                "output_space must be either 'raw' or "
                f"'cumulative_log_change', got {output_space}."
            )

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

        for batch in loader:
            y_true_raw = batch["y"].float()
            last_context_target = batch["last_context_target"].float()

            y_pred_raw = last_context_target.unsqueeze(1).repeat(
                1,
                len(self.horizons),
                1,
                1,
            )

            if output_space == "raw":
                y_pred = y_pred_raw
                y_true = y_true_raw

            else:
                y_pred = raw_to_cumulative_log_change(
                    y_raw=y_pred_raw,
                    last_context_target=last_context_target,
                )

                y_true = raw_to_cumulative_log_change(
                    y_raw=y_true_raw,
                    last_context_target=last_context_target,
                )

            all_y_pred.append(y_pred)
            all_y_true.append(y_true)
            all_sample_idx.append(batch["sample_idx"])
            all_origin_idx.append(batch["origin_idx"])
            all_target_indices.append(batch["target_indices"])

        y_pred = torch.cat(all_y_pred, dim=0)
        y_true = torch.cat(all_y_true, dim=0)

        sample_idx = torch.cat(all_sample_idx, dim=0)
        origin_idx = torch.cat(all_origin_idx, dim=0)
        target_indices = torch.cat(all_target_indices, dim=0)

        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "output_space": output_space,
            "channels": self.target_channels,
            "horizons": self.horizons,
            "sample_idx": sample_idx,
            "origin_idx": origin_idx,
            "target_indices": target_indices,
        }

    def _dataset_config(self) -> dict[str, Any]:
        """
        Build a minimal config for WindowedCandleDataset.

        Persistence only needs the target channels as inputs, because it only
        uses the last context target.
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
    

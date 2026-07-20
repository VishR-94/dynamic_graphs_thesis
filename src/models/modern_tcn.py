import sys
import torch
from typing import Any
from types import SimpleNamespace
from pathlib import Path
from src.data.data_generator import (
    WindowedCandleDataset,
    WindowContextNormaliser,)
from src.evaluation.prediction_transforms import inverse_window_normalisation
from torch.utils.data import DataLoader
import random
import numpy as np
from collections.abc import Callable

"""
Here we will implement the Modern-TCN benchmark. ModernTCN is a convolution based neural model with the following design:
1. Input data is required to be of shape X = [B,T,N], where B is batch, T is time and N is series
2. This passes through a patch embedding layer:
    . P : patch length
    . S : stride length
    . T_p : number of patch positions
   We pad so that T_p = T // S
   The same embedding function is applied to each asset so that we end up with E = [B,N,D,T_p], where D is a hidden dimension
   Patch embedding mixes nearby time steps within each patch - it does not mix over patches, over series, or over batches (b).
3. The output from 2. is passed to a DWConv layer. This takes E[b,n,d,:] and applies a temporal convolution filter. The 
   filter is mixing over patches for a given batch, series and hidden dimension. Every (n,d) pair gets its own filter (so
   we do not share filters over assets or hidden dimensions). The output is still E[B,N,D,T_p]
4. The output from 3. is passed to a ConvFFN1 layer. This layer mixes across hidden dimension. It takes E[b,n,:,p]. It is a
   1x1 convolution filter (similar to a FFN) and projects from R^D -> R^{rD} -> R^D. r is a hyperparameter. Mathematically, 
   this step is E_out = W_2 * sigma(W_1 * E_in). This step mixes across hidden dimensions within each series.
5. Next we proceed to ConvFF2. This takes the output of 4. and mixes over series. It takes E[b,:,d,p] and applies another 
   1x1 conv filter. It also projects from R^N -> R^{rN} -> R^N.
6. Steps 3-5. are 1 block. The architecture adds skip connections - Z_{out} = Block(Z_in) + Z_in
7. Finally we have a task specific output head.

Important to note - our input data is X = [B,T,N,C]. To fit the require shape, we flatten to X = [B,T,N*C]. This means the
model loses notion of intra asset vs cross asset OHLC data, but this mean we stay true to the model architecture. The 
output head will therefore also return [B,T,N*C]. We will need to unflatten this to [B,T,N,C]. 

Params: 
1. We use the in built RevIN normalisation in favour of our custom Window Normalisation
2. We use the default r = 1.
3. We train with the default ADAM with lr = 1e-4 and default learning rate decay.
4. We keep D = 64.
5. We set P = 8 and S = 4.
6. The default uses 1 ModernTCN block. It ablates for 2 and 3 blocks. We initially use 1. 
"""

class ModernTCNBaseline:
    """
    Project wrapper for the official ModernTCN forecasting model.

    The official PyTorch model is constructed inside fit(...)
    """

    def __init__(
        self,
        context_length: int,
        horizons: list[int],
        target_channels: list[str],
        stride: int,
        input_channels: list[str]|None=None,
        patch_size: int = 8,
        patch_stride: int = 4,
        hidden_dim: int = 64,
        ffn_ratio: int = 1,
        num_blocks: int = 1,
        large_kernel: int = 51,
        small_kernel: int = 5,
        dropout: float = 0.05,
        head_dropout: float = 0.0,
        variable_layout: str = 'joint',
        revin: bool = True,
        normaliser: WindowContextNormaliser | None=None,
        revin_affine: bool = False,
        subtract_last: bool = False,
        individual_head: bool = False,
        use_multi_scale: bool = False,
        small_kernel_merged: bool = False,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        batch_size: int = 16,
        num_workers: int = 0,
        max_epochs: int = 100,
        patience: int = 10,
        seed: int = 42,
        scheduler_type: str = "type3",
    ) -> None:
        # Project forecasting contract.
        self.context_length = int(context_length)
        self.horizons = list(horizons)
        self.stride = int(stride)

        self.target_channels = list(target_channels)

        # Backward compatibility: older constructors used target channels
        # as both inputs and targets.
        self.input_channels = list(
            self.target_channels
            if input_channels is None
            else input_channels
        )

        if variable_layout not in {"joint", "per_asset"}:
            raise ValueError(
                "variable_layout must be either 'joint' or 'per_asset', "
                f"got {variable_layout!r}."
            )

        if not self.input_channels:
            raise ValueError(
                "input_channels must contain at least one channel."
            )

        if not self.target_channels:
            raise ValueError(
                "target_channels must contain at least one channel."
            )

        if len(self.input_channels) != len(set(self.input_channels)):
            raise ValueError(
                "input_channels contains duplicates: "
                f"{self.input_channels}."
            )

        if len(self.target_channels) != len(set(self.target_channels)):
            raise ValueError(
                "target_channels contains duplicates: "
                f"{self.target_channels}."
            )

        missing_targets = [
            channel
            for channel in self.target_channels
            if channel not in self.input_channels
        ]

        if missing_targets:
            raise ValueError(
                "Every target channel must also be present in "
                "input_channels. Missing targets: "
                f"{missing_targets}."
            )

        self.variable_layout = variable_layout

        self.num_input_channels = len(self.input_channels)
        self.num_target_channels = len(self.target_channels)

        # Indices used later to select forecast targets from the
        # full ModernTCN output.
        self.target_input_indices = [
            self.input_channels.index(channel)
            for channel in self.target_channels
        ]

        # Temporary compatibility alias. Existing shape code still uses
        # num_channels to mean the number of model input variables.
        self.num_channels = self.num_input_channels

        # Agreed ModernTCN architecture.
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hidden_dim = hidden_dim
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_kernel = large_kernel
        self.small_kernel = small_kernel
        self.dropout = dropout
        self.head_dropout = head_dropout

        # Official architecture flags.
        if revin and normaliser is not None:
            raise ValueError(
                "A project normaliser must not be supplied when revin=True."
            )

        if not revin and normaliser is None:
            raise ValueError(
                "WindowContextNormaliser is required when revin=False."
            )

        if normaliser is not None:
            if not normaliser.apply_to_target:
                raise ValueError(
                    "ModernTCN requires normalisation.window_context."
                    "apply_to_target: true when revin=False."
                )

            if not normaliser.include_stats:
                raise ValueError(
                    "ModernTCN requires normalisation.window_context."
                    "include_stats: true when revin=False."
                )

        self.revin = revin
        self.normaliser = normaliser
        self.loss_space = (
            "raw"
            if self.revin
            else "window_context_normalised"
        )

        self.revin_affine = revin_affine
        self.subtract_last = subtract_last
        self.individual_head = individual_head
        self.use_multi_scale = use_multi_scale
        self.small_kernel_merged = small_kernel_merged
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.scheduler_type = scheduler_type
        self.num_workers = num_workers

        # These are resolved from the training split inside fit(...).
        self.asset_cols: list[str] | None = None
        self.num_assets: int | None = None
        self.num_variables: int | None = None

        input_channel_code = "".join(
            channel.strip().lower()[0]
            for channel in self.input_channels
        )

        target_channel_code = "".join(
            channel.strip().lower()[0]
            for channel in self.target_channels
        )

        # Preserve existing names such as joint_c for models whose
        # input and target channels are identical.
        if self.input_channels == self.target_channels:
            self.experiment_name = (
                f"{self.variable_layout}_{target_channel_code}"
            )
        else:
            self.experiment_name = (
                f"{self.variable_layout}_"
                f"{input_channel_code}_to_{target_channel_code}"
            )

        # These are populated later by fit(...).
        self.model: Any | None = None
        self.train_split: dict[str, Any] | None = None
        self.val_split: dict[str, Any] | None = None
        self.device: torch.device | None = None
        self.checkpoint_path: Path | None = None
        self.training_history: list[dict[str, float | int]] = []
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
    ) -> "ModernTCNBaseline":
        """
        Construct the wrapper from the project configuration.

        Global forecasting values come from config["forecasting"].
        ModernTCN-specific values may be supplied under
        config["models"]["modern_tcn"]; otherwise the agreed first-run
        values defined in __init__ are used.
        """
        forecasting_config = config["forecasting"]
        modern_tcn_config = (
            config.get("models", {})
            .get("modern_tcn", {})
        )
        modern_tcn_training_config = modern_tcn_config.get(
            "training",
            {},
        )

        global_training_config = config.get("training", {})

        revin = bool(
            modern_tcn_config.get("revin", True)
        )

        normaliser = None

        if not revin:
            normalisation_config = config.get(
                "normalisation",
                {},
            )

            if not bool(
                normalisation_config.get("enabled", False)
            ):
                raise ValueError(
                    "normalisation.enabled must be true when "
                    "ModernTCN revin is false."
                )

            normaliser = WindowContextNormaliser.from_config(
                config
            )

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=[int(horizon) for horizon in forecasting_config["horizons"]],
            target_channels=list(forecasting_config["target_channels"]),
            stride=int(forecasting_config["stride"]),
            input_channels=list(forecasting_config.get("input_channels",forecasting_config["target_channels"],)),
            patch_size=int(modern_tcn_config.get("patch_size", 8)),
            patch_stride=int(modern_tcn_config.get("patch_stride", 4)),
            hidden_dim=int(modern_tcn_config.get("hidden_dim", 64)),
            ffn_ratio=int(modern_tcn_config.get("ffn_ratio", 1)),
            num_blocks=int(modern_tcn_config.get("num_blocks", 1)),
            large_kernel=int(modern_tcn_config.get("large_kernel", 51)),
            small_kernel=int(modern_tcn_config.get("small_kernel", 5)),
            dropout=float(modern_tcn_config.get("dropout", 0.05)),
            head_dropout=float(modern_tcn_config.get("head_dropout", 0.0)),
            variable_layout=str(modern_tcn_config.get("variable_layout", "joint")),
            revin=revin,
            normaliser=normaliser,
            revin_affine=bool(modern_tcn_config.get("revin_affine", False)),
            subtract_last=bool(modern_tcn_config.get("subtract_last", False)),
            individual_head=bool(modern_tcn_config.get("individual_head", False)),
            use_multi_scale=bool(modern_tcn_config.get("use_multi_scale", False)),
            small_kernel_merged=bool(modern_tcn_config.get("small_kernel_merged",False)),
            learning_rate=float(modern_tcn_training_config.get("learning_rate",1.0e-4)),
            weight_decay=float(modern_tcn_training_config.get("weight_decay",0.0)),
            batch_size=int(modern_tcn_training_config.get("batch_size",16)),
            max_epochs=int(modern_tcn_training_config.get("max_epochs",100)),
            patience=int(modern_tcn_training_config.get("patience",10)),
            seed=int(modern_tcn_training_config.get("seed",42)),
            scheduler_type=str(modern_tcn_training_config.get("scheduler_type","type3")),
            num_workers=int(modern_tcn_training_config.get("num_workers",global_training_config.get("num_workers", 0))),
        )
    
    def _dataset_config(self) -> dict[str, Any]:
        """
        Build the OHLC-only forecasting configuration used by
        WindowedCandleDataset.

        ModernTCN uses the project target channels as both its input channels
        and target channels, so volume and amount are excluded.
        """
        return {
            "forecasting": {
                "context_length": self.context_length,
                "horizons": list(self.horizons),
                "stride": self.stride,
                "input_channels": list(self.input_channels),
                "target_channels": list(self.target_channels),
            }
        }
    
    def _expected_num_variables(
        self,
        num_assets: int,
    ) -> int:
        """
        Return the number of variables presented to ModernTCN.

        joint:
            M = N * C_input

        per_asset:
            M = C_input
        """
        if num_assets < 1:
            raise ValueError(
                f"num_assets must be at least 1, got {num_assets}."
            )

        if self.variable_layout == "joint":
            return num_assets * self.num_input_channels

        if self.variable_layout == "per_asset":
            return self.num_input_channels

        raise RuntimeError(
            f"Unsupported variable_layout: {self.variable_layout!r}."
        )

    def _resolve_data_dimensions(
        self,
        train_split: dict[str, Any],
        val_split: dict[str, Any],
    ) -> None:
        """
        Resolve the dimensions required by ModernTCN.

        joint:
            num_variables = num_assets * num_channels

        per_asset:
            num_variables = num_channels

        Training and validation must contain the same assets in the same
        order. This is required for project-facing tensors [B, T, N, C]
        under both layouts.
        """
        train_asset_cols = list(train_split["asset_cols"])
        val_asset_cols = list(val_split["asset_cols"])

        if len(train_asset_cols) == 0:
            raise ValueError("The training split contains no assets.")

        if train_asset_cols != val_asset_cols:
            raise ValueError(
                "Training and validation splits must use the same assets "
                "in the same order."
            )

        missing_train_input_channels = [
            channel
            for channel in self.input_channels
            if channel not in train_split["channels"]
        ]

        missing_val_input_channels = [
            channel
            for channel in self.input_channels
            if channel not in val_split["channels"]
        ]

        missing_train_target_channels = [
            channel
            for channel in self.target_channels
            if channel not in train_split["channels"]
        ]

        missing_val_target_channels = [
            channel
            for channel in self.target_channels
            if channel not in val_split["channels"]
        ]

        if missing_train_input_channels:
            raise ValueError(
                "The training split is missing configured ModernTCN "
                f"input channels: {missing_train_input_channels}."
            )

        if missing_val_input_channels:
            raise ValueError(
                "The validation split is missing configured ModernTCN "
                f"input channels: {missing_val_input_channels}."
            )

        if missing_train_target_channels:
            raise ValueError(
                "The training split is missing configured ModernTCN "
                f"target channels: {missing_train_target_channels}."
            )

        if missing_val_target_channels:
            raise ValueError(
                "The validation split is missing configured ModernTCN "
                f"target channels: {missing_val_target_channels}."
            )

        self.asset_cols = train_asset_cols
        self.num_assets = len(train_asset_cols)
        self.num_variables = self._expected_num_variables(
            num_assets=self.num_assets,
        )

    def _build_official_config(self) -> SimpleNamespace:
        """
        Build the attribute-based configuration expected by the official
        ModernTCN forecasting Model class.

        This method must be called only after _resolve_data_dimensions(...),
        because enc_in depends on the resolved data dimensions and variable layout
        """
        if self.num_variables is None:
            raise RuntimeError(
                "ModernTCN data dimensions must be resolved before building "
                "the official model configuration."
            )

        return SimpleNamespace(
            # Values required by the official ModernTCN wrapper.
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=self.ffn_ratio,
            num_blocks=[self.num_blocks],
            large_size=[self.large_kernel],
            small_size=[self.small_kernel],

            # The official implementation constructs four downsampling entries
            # even when only one stage is used, so four dimension values are
            # supplied without modifying the external code.
            dims=[self.hidden_dim] * 4,
            dw_dims=[self.hidden_dim] * 4,

            enc_in=self.num_variables,
            small_kernel_merged=self.small_kernel_merged,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            use_multi_scale=self.use_multi_scale,

            # RevIN settings.
            revin=int(self.revin),
            affine=int(self.revin_affine),
            subtract_last=int(self.subtract_last),

            # Forecasting task.
            freq="t",
            seq_len=self.context_length,
            pred_len=len(self.horizons),
            individual=int(self.individual_head),

            # Decomposition is disabled for this benchmark.
            decomposition=0,
            kernel_size=25,

            # Patch embedding.
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
    
    def _build_official_model(self) -> Any:
        """
        Import and construct the official ModernTCN forecasting model.

        The external repository is imported lazily so that importing this
        project's model module does not fail when the submodule has not yet
        been initialised.
        """
        project_root = Path(__file__).resolve().parents[2]

        modern_tcn_root = (
            project_root
            / "external"
            / "ModernTCN"
            / "ModernTCN-Long-term-forecasting"
        )

        if not modern_tcn_root.is_dir():
            raise FileNotFoundError(
                "The official ModernTCN forecasting directory was not found. "
                "Initialise the external/ModernTCN submodule before building "
                "the model."
            )

        modern_tcn_path = str(modern_tcn_root)

        if modern_tcn_path not in sys.path:
            sys.path.insert(0, modern_tcn_path)

        from models.ModernTCN import Model as OfficialModernTCNModel

        official_config = self._build_official_config()
        self.model = OfficialModernTCNModel(official_config)

        return self.model
    
    def _build_dataset(
        self,
        split: dict[str, Any],
    ) -> WindowedCandleDataset:
        """
        Build the window dataset used by ModernTCN.

        When revin=True, the dataset remains in raw value space and the
        official model handles RevIN internally.

        When revin=False, the project WindowContextNormaliser normalises
        both context inputs and targets using context-only statistics.
        """
        return WindowedCandleDataset.from_config(
            split=split,
            config=self._dataset_config(),
            normaliser=self.normaliser,
        )
    
    def _build_data_loader(
        self,
        dataset: WindowedCandleDataset,
        shuffle: bool,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> DataLoader:
        """
        Build a reproducible DataLoader for ModernTCN.

        Training uses shuffle=True. Validation and prediction use shuffle=False.
        Explicit batch-size and worker overrides are supported for prediction.
        """
        resolved_batch_size = (
            self.batch_size
            if batch_size is None
            else batch_size
        )

        resolved_num_workers = (
            self.num_workers
            if num_workers is None
            else num_workers
        )

        generator = None

        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        return DataLoader(
            dataset,
            batch_size=resolved_batch_size,
            shuffle=shuffle,
            num_workers=resolved_num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
    
    def _flatten_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a project-format tensor [B, T, N, C] into the variable
        layout expected by ModernTCN.

        joint:
            [B, T, N, C] -> [B, T, N * C]

        per_asset:
            [B, T, N, C] -> [B, N, T, C] -> [B * N, T, C]
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor with shape [B,T,N,C]"
            )
    
        if x.ndim != 4:
            raise ValueError(
                "x must have shape [B,T,N,C]. "
                f"Received shape {tuple(x.shape)}."
            )
        
        batch_size, time_steps, num_assets, num_channels = x.shape

        if self.num_assets is not None and num_assets != self.num_assets:
            raise ValueError(
                "The asset dimension of x does not match the model. "
                f"Expected {self.num_assets}, got {num_assets}."
            )

        if num_channels != self.num_input_channels:
            raise ValueError(
                "The channel dimension of x does not match target_channels. "
                f"Expected {self.num_input_channels}, got {num_channels}."
            )
        
        if self.variable_layout == "joint":
            return x.reshape(
                batch_size,
                time_steps,
                num_assets * num_channels
            )
        
        if self.variable_layout == "per_asset":
            return (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(
                    batch_size * num_assets,
                    time_steps,
                    num_channels,
                )
            )
        
        raise RuntimeError(
            f"Unsupported variable_layout: {self.variable_layout!r}."
        )


    def _unflatten_output(
        self,
        x: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Convert ModernTCN output back to project format [B, H, N, C].

        joint:
            [B, H, N * C] -> [B, H, N, C]

        per_asset:
            [B * N, H, C] -> [B, N, H, C] -> [B, H, N, C]
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor containing ModernTCN output."
            )

        if x.ndim != 3:
            raise ValueError(
                "ModernTCN output must have exactly three dimensions. "
                f"Received shape {tuple(x.shape)}."
            )

        if self.num_assets is None:
            raise RuntimeError(
                "The number of assets has not been resolved."
            )
        
        horizon_count = x.shape[1]

        if self.variable_layout == "joint":
            expected_shape = (
                batch_size,
                horizon_count,
                self.num_assets * self.num_input_channels,
            )

            if tuple(x.shape) != expected_shape:
                raise ValueError(
                    "Unexpected joint ModernTCN output shape. "
                    f"Expected {expected_shape}, got {tuple(x.shape)}."
                )

            return x.reshape(
                batch_size,
                horizon_count,
                self.num_assets,
                self.num_input_channels,
            )
        
        if self.variable_layout == "per_asset":
            expected_shape = (
                batch_size * self.num_assets,
                horizon_count,
                self.num_input_channels,
            )

            if tuple(x.shape) != expected_shape:
                raise ValueError(
                    "Unexpected per-asset ModernTCN output shape. "
                    f"Expected {expected_shape}, got {tuple(x.shape)}."
                )

            return (
                x.reshape(
                    batch_size,
                    self.num_assets,
                    horizon_count,
                    self.num_input_channels,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )

        raise RuntimeError(
            f"Unsupported variable_layout: {self.variable_layout!r}."
        )
    
    def _select_target_outputs(
        self,
        all_predictions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select configured target channels from the full ModernTCN output.

        Args:
            all_predictions:
                Tensor with shape [B, H, N, C_input].

        Returns:
            Tensor with shape [B, H, N, C_target].

        ModernTCN forecasts every input variable. Only channels listed in
        target_channels are retained for loss calculation and prediction.
        """
        if not isinstance(all_predictions, torch.Tensor):
            raise TypeError(
                "all_predictions must be a torch.Tensor."
            )

        if all_predictions.ndim != 4:
            raise ValueError(
                "all_predictions must have shape [B,H,N,C_input]. "
                f"Received shape {tuple(all_predictions.shape)}."
            )

        if all_predictions.shape[-1] != self.num_input_channels:
            raise ValueError(
                "The final dimension of all_predictions does not match "
                "input_channels. "
                f"Expected {self.num_input_channels}, "
                f"got {all_predictions.shape[-1]}."
            )

        target_indices = torch.tensor(
            self.target_input_indices,
            dtype=torch.long,
            device=all_predictions.device,
        )

        return all_predictions.index_select(
            dim=-1,
            index=target_indices,
        )
            
    def _forward_project_tensor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a project-format tensor through the official ModernTCN model.

        Args:
            x:
                Context tensor with shape [B, T, N, C].

        Returns:
            Prediction tensor with shape [B, H, N, C], where H is
            len(self.horizons).

        The tensor may be raw or normalised depending on the configured
        normalisation pipeline. The caller is responsible for placing x on
        the same device as the model.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model has not been constructed."
            )

        if self.num_assets is None:
            raise RuntimeError(
                "The number of assets has not been resolved."
            )

        batch_size = x.shape[0]

        model_input = self._flatten_input(x)
        model_output = self.model(model_input)

        all_predictions = self._unflatten_output(
            model_output,
            batch_size=batch_size,
        )

        return self._select_target_outputs(
            all_predictions
        )
    
    def _compute_batch_loss(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Move one project batch to the selected device, run ModernTCN, and
        calculate mean squared error in the configured loss space.

        When revin=True:
            y_pred and y_true are in raw value space.

        When revin=False:
            y_pred and y_true are in window-context-normalised space.
        """
        x = batch["x"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        y_true = batch["y"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        y_pred = self._forward_project_tensor(x)

        loss = torch.nn.functional.mse_loss(
            y_pred,
            y_true,
        )

        return loss, y_pred, y_true
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        Build the Adam optimiser used to train the official ModernTCN model.

        The model must already have been constructed because the optimiser
        stores references to its trainable parameters.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model must be constructed before "
                "building the optimiser."
            )

        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    
    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        """
        Train ModernTCN for one complete pass over the training dataset.

        Returns:
            Example-weighted mean training MSE for the epoch.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model must be constructed before "
                "training."
            )

        self.model.train()

        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            loss, _, y_true = self._compute_batch_loss(
                batch=batch,
                device=device,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Encountered a non-finite training loss."
                )

            loss.backward()
            optimizer.step()

            batch_size = y_true.shape[0]

            total_loss += loss.detach().item() * batch_size
            total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "The training DataLoader produced no examples."
            )

        return total_loss / total_examples
    
    def _validate_one_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """
        Evaluate ModernTCN over one complete validation pass.

        Returns:
            Example-weighted mean validation MSE.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model must be constructed before "
                "validation."
            )

        self.model.eval()

        total_loss = 0.0
        total_examples = 0

        with torch.no_grad():
            for batch in loader:
                loss, _, y_true = self._compute_batch_loss(
                    batch=batch,
                    device=device,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Encountered a non-finite validation loss."
                    )

                batch_size = y_true.shape[0]

                total_loss += loss.item() * batch_size
                total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "The validation DataLoader produced no examples."
            )

        return total_loss / total_examples
    

    def _checkpoint_metadata(
        self,
        asset_cols: list[str] | None = None,
        num_assets: int | None = None,
        num_variables: int | None = None,
    ) -> dict[str, Any]:
        """
        Return the resolved configuration required to interpret a checkpoint.
        """
        resolved_asset_cols = (
            self.asset_cols
            if asset_cols is None
            else list(asset_cols)
        )

        resolved_num_assets = (
            self.num_assets
            if num_assets is None
            else int(num_assets)
        )

        resolved_num_variables = (
            self.num_variables
            if num_variables is None
            else int(num_variables)
        )

        if resolved_asset_cols is None:
            raise RuntimeError(
                "Asset metadata must be resolved before checkpointing."
            )

        if resolved_num_assets is None:
            raise RuntimeError(
                "num_assets must be resolved before checkpointing."
            )

        if resolved_num_variables is None:
            raise RuntimeError(
                "num_variables must be resolved before checkpointing."
            )

        normalisation_clip = (
            None
            if self.normaliser is None
            else bool(self.normaliser.clip)
        )

        return {
            "experiment_name": self.experiment_name,
            "variable_layout": self.variable_layout,
            "input_channels": list(self.input_channels),
            "target_channels": list(self.target_channels),
            "target_input_indices": list(self.target_input_indices),
            "horizons": list(self.horizons),
            "context_length": self.context_length,
            "stride": self.stride,
            "revin": self.revin,
            "loss_space": self.loss_space,
            "normalisation_clip": normalisation_clip,
            "asset_cols": list(resolved_asset_cols),
            "num_assets": resolved_num_assets,
            # num_channels is retained as a legacy alias so existing
            # close-only checkpoints remain loadable.
            "num_channels": self.num_input_channels,
            "num_input_channels": self.num_input_channels,
            "num_target_channels": self.num_target_channels,
            "num_variables": resolved_num_variables,
            "architecture": {
                "patch_size": self.patch_size,
                "patch_stride": self.patch_stride,
                "hidden_dim": self.hidden_dim,
                "ffn_ratio": self.ffn_ratio,
                "num_blocks": self.num_blocks,
                "large_kernel": self.large_kernel,
                "small_kernel": self.small_kernel,
                "dropout": self.dropout,
                "head_dropout": self.head_dropout,
                "revin_affine": self.revin_affine,
                "subtract_last": self.subtract_last,
                "individual_head": self.individual_head,
                "use_multi_scale": self.use_multi_scale,
                "small_kernel_merged": self.small_kernel_merged,
            },
        }
    
    def _normalise_checkpoint_metadata(
        self,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add input/target metadata fields that are absent from older
        ModernTCN checkpoints.

        Older checkpoints used target_channels as both the model inputs
        and forecast targets.
        """
        normalised = dict(checkpoint)

        target_channels = list(
            normalised["target_channels"]
        )

        input_channels = list(
            normalised.get(
                "input_channels",
                target_channels,
            )
        )

        missing_targets = [
            channel
            for channel in target_channels
            if channel not in input_channels
        ]

        if missing_targets:
            raise ValueError(
                "Checkpoint target channels are not present in its "
                f"input channels: {missing_targets}."
            )

        normalised["input_channels"] = input_channels
        normalised["target_channels"] = target_channels

        normalised["num_input_channels"] = int(
            normalised.get(
                "num_input_channels",
                normalised.get(
                    "num_channels",
                    len(input_channels),
                ),
            )
        )

        normalised["num_target_channels"] = int(
            normalised.get(
                "num_target_channels",
                len(target_channels),
            )
        )

        normalised["target_input_indices"] = list(
            normalised.get(
                "target_input_indices",
                [
                    input_channels.index(channel)
                    for channel in target_channels
                ],
            )
        )

        # Retain the old metadata field as an alias.
        normalised["num_channels"] = int(
            normalised.get(
                "num_channels",
                normalised["num_input_channels"],
            )
        )

        return normalised


    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_validation_loss: float,
        patience_counter: int,
    ) -> None:
        """
        Save the current ModernTCN training state.

        Args:
            checkpoint_path:
                Destination for the PyTorch checkpoint.
            optimizer:
                Optimiser whose internal Adam state will be saved.
            epoch:
                One-based epoch number that has just completed.
            best_validation_loss:
                Best validation MSE observed up to this epoch.
            patience_counter:
                Number of consecutive epochs without improvement.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model must be constructed before "
                "saving a checkpoint."
            )

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "patience_counter": patience_counter,
            **self._checkpoint_metadata(),
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """
        Restore a ModernTCN checkpoint.

        Args:
            checkpoint_path:
                Path to a checkpoint created by _save_checkpoint().
            device:
                Device on which checkpoint tensors should be loaded.
            optimizer:
                Optional Adam optimiser to restore for resumed training.
                Leave as None when only restoring the best model weights.

        Returns:
            The complete checkpoint dictionary.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model must be constructed before "
                "loading a checkpoint."
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        checkpoint_metadata = (
            self._normalise_checkpoint_metadata(
                checkpoint
            )
        )

        expected_metadata = self._checkpoint_metadata()

        for key, expected_value in expected_metadata.items():
            if checkpoint_metadata.get(key) != expected_value:
                raise ValueError(
                    f"Checkpoint metadata for {key!r} is incompatible with "
                    "the current ModernTCN baseline."
                )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
        )

        if optimizer is not None:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"],
            )

        return checkpoint
    
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "ModernTCNBaseline":
        """
        Load a trained ModernTCN checkpoint for inference.

        This constructs the model using the current configuration and the
        dimensions stored in the checkpoint. It does not create an optimiser
        or perform any training.

        Args:
            checkpoint_path:
                Path to a checkpoint produced by ModernTCNBaseline.fit().

            device:
                Device on which prediction will run.

        Returns:
            The current ModernTCNBaseline instance.
        """
        checkpoint_path = (
            Path(checkpoint_path)
            .expanduser()
            .resolve()
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        resolved_device = torch.device(device)

        if (
            resolved_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        required_keys = {
            "epoch",
            "model_state_dict",
            "best_validation_loss",
            "experiment_name",
            "variable_layout",
            "target_channels",
            "horizons",
            "context_length",
            "stride",
            "revin",
            "loss_space",
            "normalisation_clip",
            "asset_cols",
            "num_assets",
            "num_channels",
            "num_variables",
            "architecture",
        }

        missing_keys = required_keys.difference(checkpoint)

        if missing_keys:
            raise KeyError(
                "ModernTCN checkpoint is missing required keys: "
                f"{sorted(missing_keys)}"
            )
        
        checkpoint_metadata = (
            self._normalise_checkpoint_metadata(
                checkpoint
            )
        )

        checkpoint_asset_cols = list(
            checkpoint["asset_cols"]
        )

        checkpoint_num_assets = int(
            checkpoint["num_assets"]
        )

        checkpoint_num_variables = int(
            checkpoint["num_variables"]
        )

        if len(checkpoint_asset_cols) != checkpoint_num_assets:
            raise ValueError(
                "Checkpoint asset metadata is inconsistent: "
                f"received {len(checkpoint_asset_cols)} asset names but "
                f"num_assets={checkpoint_num_assets}."
            )

        expected_num_variables = self._expected_num_variables(
            num_assets=checkpoint_num_assets,
        )

        expected_metadata = self._checkpoint_metadata(
            asset_cols=checkpoint_asset_cols,
            num_assets=checkpoint_num_assets,
            num_variables=expected_num_variables,
        )

        for key, expected_value in expected_metadata.items():
            checkpoint_value = checkpoint_metadata.get(key)

            if checkpoint_value != expected_value:
                raise ValueError(
                    f"Checkpoint metadata for {key!r} is incompatible "
                    "with the current ModernTCN configuration. "
                    f"Expected {expected_value!r}, "
                    f"got {checkpoint_value!r}."
                )

        if checkpoint_num_variables != expected_num_variables:
            raise ValueError(
                "Checkpoint num_variables is inconsistent with its "
                "number of assets, channels and variable layout. "
                f"Expected {expected_num_variables}, "
                f"got {checkpoint_num_variables}."
            )

        self.asset_cols = checkpoint_asset_cols
        self.num_assets = checkpoint_num_assets
        self.num_variables = checkpoint_num_variables

        self.model = self._build_official_model()

        try:
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint weights are incompatible with the "
                "current ModernTCN architecture. Ensure that the "
                "config contains the selected run's architecture."
            ) from exc

        self.model.to(resolved_device)
        self.model.eval()

        self.device = resolved_device
        self.checkpoint_path = checkpoint_path
        self.best_epoch = int(checkpoint["epoch"])
        self.best_validation_loss = float(
            checkpoint["best_validation_loss"]
        )
        self.training_history = []

        return self
    

    def _adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        completed_epoch: int,
    ) -> float:
        """
        Apply the official ModernTCN type3 learning-rate update.

        This method is called after an epoch has completed, matching the
        official training loop. The returned learning rate is therefore used
        by the following epoch.

        Args:
            optimizer:
                Optimiser whose parameter-group learning rates will be updated.
            completed_epoch:
                One-based number of the epoch that has just completed.

        Returns:
            The learning rate assigned to the optimiser.
        """
        if self.scheduler_type != "type3":
            raise ValueError(
                "Only the official ModernTCN type3 schedule is currently "
                "supported."
            )

        if completed_epoch < 1:
            raise ValueError(
                "completed_epoch must be one-based and at least 1."
            )

        if completed_epoch < 3:
            learning_rate = self.learning_rate
        else:
            learning_rate = self.learning_rate * (
                0.9 ** (completed_epoch - 3)
            )

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        return learning_rate
    
    def _set_seed(self) -> None:
        """
        Seed the random-number generators used by the ModernTCN experiment.

        This method should be called before constructing the official model
        and before constructing the shuffled training DataLoader.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(
        self,
        train_split: dict[str, Any],
        val_split: dict[str, Any],
        checkpoint_path: str | Path | None = None,
        epoch_callback: (Callable[[dict[str,float | int]], None] | None) = None,
    ) -> "ModernTCNBaseline":
        """
        Train ModernTCN using the training split and select the best epoch
        using validation MSE.

        The test split must not be passed to this method.
        """
        self._set_seed()

        self._resolve_data_dimensions(
            train_split=train_split,
            val_split=val_split,
        )

        train_dataset = self._build_dataset(train_split)
        val_dataset = self._build_dataset(val_split)

        train_loader = self._build_data_loader(
            dataset=train_dataset,
            shuffle=True,
        )

        val_loader = self._build_data_loader(
            dataset=val_dataset,
            shuffle=False,
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = self._build_official_model()
        model.to(device)

        optimizer = self._build_optimizer()

        if checkpoint_path is None:
            project_root = Path(__file__).resolve().parents[2]
            resolved_checkpoint_path = (
                project_root
                / "checkpoints"
                / "modern_tcn"
                / "best_checkpoint.pt"
            )
        else:
            resolved_checkpoint_path = Path(checkpoint_path)

        best_validation_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, self.max_epochs + 1):
            learning_rate = float(
                optimizer.param_groups[0]["lr"]
            )

            training_loss = self._train_one_epoch(
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            validation_loss = self._validate_one_epoch(
                loader=val_loader,
                device=device,
            )

            epoch_record: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }

            history.append(epoch_record)

            print(
                f"Epoch {epoch:03d} | "
                f"lr={learning_rate:.6g} | "
                f"train_mse={training_loss:.6f} | "
                f"val_mse={validation_loss:.6f}"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                patience_counter = 0

                self._save_checkpoint(
                    checkpoint_path=resolved_checkpoint_path,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                    patience_counter=patience_counter,
                )
            else:
                patience_counter += 1

            if epoch_callback is not None:
                epoch_callback(dict(epoch_record))

            if patience_counter >= self.patience:
                print(
                    "Early stopping after "
                    f"{self.patience} epochs without improvement."
                )
                break

            self._adjust_learning_rate(
                optimizer=optimizer,
                completed_epoch=epoch,
            )


        self._load_checkpoint(
            checkpoint_path=resolved_checkpoint_path,
            device=device,
        )

        self.train_split = train_split
        self.val_split = val_split
        self.device = device
        self.checkpoint_path = resolved_checkpoint_path
        self.training_history = history
        self.best_epoch = best_epoch
        self.best_validation_loss = best_validation_loss

        print(
            f"Restored best checkpoint from epoch {best_epoch} "
            f"with val_mse={best_validation_loss:.6f}."
        )

        return self
    
    def predict(
        self,
        split: dict[str, Any],
        batch_size: int = 256,
        num_workers: int = 0,
    ) -> dict[str, Any]:
        """
        Generate raw ModernTCN predictions.

        Returns tensors with shape:
            [num_examples, num_horizons, num_assets, num_target_channels]
        """
        if self.model is None:
            raise RuntimeError(
                "ModernTCN must be fitted or have a checkpoint loaded "
                "before prediction."
            )

        if self.asset_cols is None:
            raise RuntimeError(
                "ModernTCN data dimensions have not been resolved."
            )

        if list(split["asset_cols"]) != list(self.asset_cols):
            raise ValueError(
                "Prediction split asset ordering does not match the "
                "fitted ModernTCN model."
            )

        missing_input_channels = [
            channel
            for channel in self.input_channels
            if channel not in split["channels"]
        ]

        missing_target_channels = [
            channel
            for channel in self.target_channels
            if channel not in split["channels"]
        ]

        if missing_input_channels:
            raise ValueError(
                "Prediction split is missing required input channels: "
                f"{missing_input_channels}."
            )

        if missing_target_channels:
            raise ValueError(
                "Prediction split is missing required target channels: "
                f"{missing_target_channels}."
            )

        dataset = self._build_dataset(split)

        loader = self._build_data_loader(
            dataset=dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        device = next(self.model.parameters()).device

        self.model.eval()

        all_y_pred = []
        all_y_true = []
        all_last_context_target = []
        all_sample_idx = []
        all_origin_idx = []
        all_target_indices = []

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                y_pred_model_space = self._forward_project_tensor(x)

                if self.revin:
                    y_pred_raw = y_pred_model_space
                    y_true_raw = batch["y"].float()
                else:
                    required_stats = {
                        "target_norm_mean",
                        "target_norm_std",
                        "y_unnormalised",
                    }

                    missing_stats = required_stats.difference(batch)

                    if missing_stats:
                        raise KeyError(
                            "Normalised prediction requires dataset fields: "
                            f"{sorted(missing_stats)}"
                        )
                    
                    target_norm_mean = batch["target_norm_mean"].to(
                        device=device,
                        dtype=y_pred_model_space.dtype,
                        non_blocking=True,
                    )

                    target_norm_std = batch["target_norm_std"].to(
                        device=device,
                        dtype=y_pred_model_space.dtype,
                        non_blocking=True,
                    )

                    y_pred_raw = inverse_window_normalisation(
                        y_norm=y_pred_model_space,
                        target_norm_mean=target_norm_mean,
                        target_norm_std=target_norm_std,
                    )

                    y_true_raw = batch["y_unnormalised"].float()

                all_y_pred.append(
                    y_pred_raw.detach().cpu()
                )
                
                all_y_true.append(
                    y_true_raw.detach().cpu()
                )
                
                all_last_context_target.append(
                    batch["last_context_target"].float().cpu()
                )
                all_sample_idx.append(
                    batch["sample_idx"].cpu()
                )
                all_origin_idx.append(
                    batch["origin_idx"].cpu()
                )
                all_target_indices.append(
                    batch["target_indices"].cpu()
                )

        if not all_y_pred:
            raise RuntimeError(
                "The prediction DataLoader produced no examples."
            )

        return {
            "y_pred": torch.cat(all_y_pred, dim=0),
            "y_true": torch.cat(all_y_true, dim=0),
            "channels": list(self.target_channels),
            "horizons": self.horizons,
            "sample_idx": torch.cat(all_sample_idx, dim=0),
            "origin_idx": torch.cat(all_origin_idx, dim=0),
            "target_indices": torch.cat(
                all_target_indices,
                dim=0,
            ),
            "last_context_target": torch.cat(
                all_last_context_target,
                dim=0,
            ),
        }
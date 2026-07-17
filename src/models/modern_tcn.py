import sys
import torch
from typing import Any
from types import SimpleNamespace
from pathlib import Path
from src.data.data_generator import WindowedCandleDataset
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

#function to flatten our input data to the correct shape, [B,T,N,C] -> [B,T,N*C]
def flatten_asset_channels(x: torch.Tensor)->torch.Tensor:
    """Flatten the asset and channel axes into ModernTCN's variable axis.

    Args:
        x:
            Tensor with shape ``[B, T, N, C]``, where:

            - ``B`` is the batch size;
            - ``T`` is the number of context time steps;
            - ``N`` is the number of assets;
            - ``C`` is the number of channels per asset.

    Returns:
        Tensor with shape ``[B, T, N * C]``.

        The flattened variable order is asset-major and channel-minor:

        ``flat_index = asset_index * C + channel_index``.

        For OHLC data this gives:

        ``asset_0_open, asset_0_high, asset_0_low, asset_0_close,``
        ``asset_1_open, ...``

    Raises:
        TypeError:
            If ``x`` is not a PyTorch tensor.
        ValueError:
            If ``x`` does not have exactly four dimensions.
    """

    if not isinstance(x,torch.Tensor):
        raise TypeError(
            'x must be a torch.Tensor with shape [B,T,N,C]'
        )
    
    if x.ndim != 4:
        raise ValueError(
            "x must have shape [B, T, N, C]. "
            f"Received shape {tuple(x.shape)}."
        )
    
    batch_size, time_steps, num_asset, num_channels = x.shape
    
    return x.reshape(
        batch_size,
        time_steps,
        num_asset * num_channels,
    )

#function to unflatten the output from [B,T,N*C] -> [B,T,N,C]
def unflatten_asset_channels(
        x: torch.Tensor,
        num_assets: int,
        num_channels: int)->torch.Tensor:
    """Restore ModernTCN's variable axis to separate asset/channel axes."""

    if not isinstance(x, torch.Tensor):
        raise TypeError(
            'x must be a torch.Tensor with shape [B, T, N * C].'
        )
    
    if x.ndim != 3:
        raise ValueError(
            "x must have shape [B, N, N * C]. "
            f"Received shape {tuple(x.shape)}."
        )   
    
    expected_final_dim = num_assets * num_channels

    if expected_final_dim != x.shape[-1]:
        raise ValueError(
            "The final dimension of x must equal number of channels * number of assets. "
            f"Expected final dimension of {expected_final_dim}, got {x.shape[-1]} "
        )
    
    batch_size, time_steps, _ = x.shape

    return x.reshape(
        batch_size,
        time_steps,
        num_assets,
        num_channels
    )

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
        patch_size: int = 8,
        patch_stride: int = 4,
        hidden_dim: int = 64,
        ffn_ratio: int = 1,
        num_blocks: int = 1,
        large_kernel: int = 51,
        small_kernel: int = 5,
        dropout: float = 0.05,
        head_dropout: float = 0.0,
        revin: bool = True,
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
        self.context_length = context_length
        self.horizons = horizons
        self.target_channels = target_channels
        self.stride = stride

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
        self.revin = revin
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

        # These are populated later by fit(...).
        self.model: Any | None = None
        self.train_split: dict[str, Any] | None = None
        self.val_split: dict[str, Any] | None = None

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

        return cls(
            context_length=int(forecasting_config["context_length"]),
            horizons=[int(horizon) for horizon in forecasting_config["horizons"]],
            target_channels=list(forecasting_config["target_channels"]),
            stride=int(forecasting_config["stride"]),
            patch_size=int(modern_tcn_config.get("patch_size", 8)),
            patch_stride=int(modern_tcn_config.get("patch_stride", 4)),
            hidden_dim=int(modern_tcn_config.get("hidden_dim", 64)),
            ffn_ratio=int(modern_tcn_config.get("ffn_ratio", 1)),
            num_blocks=int(modern_tcn_config.get("num_blocks", 1)),
            large_kernel=int(modern_tcn_config.get("large_kernel", 51)),
            small_kernel=int(modern_tcn_config.get("small_kernel", 5)),
            dropout=float(modern_tcn_config.get("dropout", 0.05)),
            head_dropout=float(modern_tcn_config.get("head_dropout", 0.0)),
            revin=bool(modern_tcn_config.get("revin", True)),
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
                "input_channels": list(self.target_channels),
                "target_channels": list(self.target_channels),
            }
        }
    
    def _resolve_data_dimensions(
        self,
        train_split: dict[str, Any],
        val_split: dict[str, Any],
    ) -> None:
        """
        Resolve ModernTCN's variable dimension from the supplied data splits.

        ModernTCN treats every asset-channel pair as one scalar variable:

            num_variables = num_assets * num_target_channels

        The validation split must contain the same assets in the same order,
        because one flattened variable index must retain the same meaning across
        training, validation, and later prediction.
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

        missing_train_channels = [
            channel
            for channel in self.target_channels
            if channel not in train_split["channels"]
        ]

        missing_val_channels = [
            channel
            for channel in self.target_channels
            if channel not in val_split["channels"]
        ]

        if missing_train_channels:
            raise ValueError(
                "The training split is missing one or more configured "
                "ModernTCN channels."
            )

        if missing_val_channels:
            raise ValueError(
                "The validation split is missing one or more configured "
                "ModernTCN channels."
            )

        self.asset_cols = train_asset_cols
        self.num_assets = len(train_asset_cols)
        self.num_variables = (
            self.num_assets * len(self.target_channels)
        )

    def _build_official_config(self) -> SimpleNamespace:
        """
        Build the attribute-based configuration expected by the official
        ModernTCN forecasting Model class.

        This method must be called only after _resolve_data_dimensions(...),
        because enc_in depends on the number of assets in the training split.
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
        Build the raw OHLC window dataset used by ModernTCN.

        The project dataset retains the established chronological, within-session
        windowing and direct sparse-horizon target construction. No project
        normaliser is applied because ModernTCN uses its native RevIN.
        """
        return WindowedCandleDataset.from_config(
            split=split,
            config=self._dataset_config(),
            normaliser=None,
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
    
    def _forward_project_tensor(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a project-format OHLC tensor through the official ModernTCN model.

        Args:
            x:
                Raw OHLC context tensor with shape [B, T, N, C].

        Returns:
            Raw OHLC prediction tensor with shape [B, H, N, C], where H is
            len(self.horizons).

        The caller is responsible for placing x on the same device as the model.
        """
        if self.model is None:
            raise RuntimeError(
                "The official ModernTCN model has not been constructed."
            )

        if self.num_assets is None:
            raise RuntimeError(
                "The number of assets has not been resolved."
            )

        x_flat = flatten_asset_channels(x)

        y_pred_flat = self.model(x_flat)

        y_pred = unflatten_asset_channels(
            y_pred_flat,
            num_assets=self.num_assets,
            num_channels=len(self.target_channels),
        )

        return y_pred
    
    def _compute_batch_loss(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Move one project batch to the selected device, run ModernTCN, and
        calculate the paper-native mean squared error.

        Returns:
            loss:
                Scalar MSE tensor.
            y_pred:
                Raw OHLC predictions with shape [B, H, N, C].
            y_true:
                Raw OHLC targets with shape [B, H, N, C].
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
            "asset_cols": self.asset_cols,
            "target_channels": list(self.target_channels),
            "horizons": list(self.horizons),
            "num_assets": self.num_assets,
            "num_variables": self.num_variables,
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

        expected_metadata = {
            "asset_cols": self.asset_cols,
            "target_channels": list(self.target_channels),
            "horizons": list(self.horizons),
            "num_assets": self.num_assets,
            "num_variables": self.num_variables,
        }

        for key, expected_value in expected_metadata.items():
            if checkpoint.get(key) != expected_value:
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

            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                }
            )

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
            [num_examples, num_horizons, num_assets, num_channels]
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

        missing_channels = [
            channel
            for channel in self.target_channels
            if channel not in split["channels"]
        ]

        if missing_channels:
            raise ValueError(
                "Prediction split is missing required target channels."
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

                y_pred = self._forward_project_tensor(x)

                all_y_pred.append(
                    y_pred.detach().cpu()
                )
                all_y_true.append(
                    batch["y"].float().cpu()
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
            "channels": self.target_channels,
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
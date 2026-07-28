from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Mapping
from tqdm.auto import tqdm
import numpy as np
import torch
import math

from src.models.kronos import (
    KRONOS_INPUT_CHANNELS,
    KRONOS_OUTPUT_CHANNELS,
    import_official_kronos,
)


KRONOS_TOKENIZER_CHANNELS = KRONOS_OUTPUT_CHANNELS


@dataclass(frozen=True)
class KronosTokenBatch:
    """Token IDs and context statistics for raw OHLCV windows.

    Shapes:
        token_ids: [B, T, N, 2]
        mean:      [B, N, 6]
        std:       [B, N, 6]

    The final token dimension is [coarse, fine].
    """

    token_ids: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    @property
    def coarse(self) -> torch.Tensor:
        return self.token_ids[..., 0]

    @property
    def fine(self) -> torch.Tensor:
        return self.token_ids[..., 1]


@dataclass(frozen=True)
class KronosNormalisedPathBatch:
    """Origin-aligned context and future path in Kronos input space.

    Shapes:
        normalised:
            [B, context_length + prediction_length, N, 6]

        mean:
            [B, N, 6]

        std:
            [B, N, 6]

        clipping_mask:
            Boolean tensor with the same shape as ``normalised``.
            True marks a value that lay outside [-clip, clip] before
            clipping.

    Statistics are estimated only from the observed context. The same
    statistics are then applied to both the context and future path.
    """

    normalised: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    clipping_mask: torch.Tensor
    context_length: int
    prediction_length: int

    @property
    def context(self) -> torch.Tensor:
        return self.normalised[
            :,
            :self.context_length,
        ]

    @property
    def future(self) -> torch.Tensor:
        return self.normalised[
            :,
            self.context_length:,
        ]

    @property
    def context_clipping_mask(self) -> torch.Tensor:
        return self.clipping_mask[
            :,
            :self.context_length,
        ]

    @property
    def future_clipping_mask(self) -> torch.Tensor:
        return self.clipping_mask[
            :,
            self.context_length:,
        ]

class KronosTokenizerAdapter:
    """CPU adapter for the frozen official Kronos tokenizer.

    This reproduces the preprocessing used by the existing Kronos
    baseline:

        raw OHLCV
        -> zero Amount
        -> context-only z-score per asset and channel
        -> clip to [-5, 5]
        -> official tokenizer.encode(..., half=True)
    """

    def __init__(
        self,
        tokenizer_id: str,
        tokenizer_revision: str,
        clip: float = 5.0,
        eps: float = 1.0e-5,
        series_batch_size: int = 64,
    ) -> None:
        self.tokenizer_id = str(tokenizer_id)
        self.tokenizer_revision = str(tokenizer_revision)
        self.clip = float(clip)
        self.eps = float(eps)
        self.series_batch_size = int(series_batch_size)

        if not self.tokenizer_id:
            raise ValueError("tokenizer_id must not be empty.")
        if not self.tokenizer_revision:
            raise ValueError("tokenizer_revision must not be empty.")
        if self.clip <= 0:
            raise ValueError("clip must be greater than zero.")
        if self.eps <= 0:
            raise ValueError("eps must be greater than zero.")
        if self.series_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )

        self._tokenizer: Any | None = None

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        series_batch_size: int = 64,
    ) -> "KronosTokenizerAdapter":
        kronos_config = config["models"]["kronos"]
        inference_config = kronos_config.get("inference", {})

        return cls(
            tokenizer_id=kronos_config["tokenizer_id"],
            tokenizer_revision=kronos_config["tokenizer_revision"],
            clip=float(inference_config.get("clip", 5.0)),
            eps=1.0e-5,
            series_batch_size=series_batch_size,
        )

    def load(self) -> "KronosTokenizerAdapter":
        """Load and freeze the pinned tokenizer on CPU."""
        if self._tokenizer is not None:
            return self

        _, _, OfficialKronosTokenizer = import_official_kronos()

        tokenizer = OfficialKronosTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
        )
        tokenizer = tokenizer.to("cpu")
        tokenizer.eval()

        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)

        if int(getattr(tokenizer, "d_in", -1)) != 6:
            raise ValueError(
                "The loaded Kronos tokenizer does not expect six "
                "OHLCVA channels."
            )

        if (
            int(getattr(tokenizer, "s1_bits", -1)),
            int(getattr(tokenizer, "s2_bits", -1)),
        ) != (10, 10):
            raise ValueError(
                "The loaded Kronos tokenizer does not use the "
                "expected 10-bit coarse and 10-bit fine tokens."
            )

        self._tokenizer = tokenizer
        return self

    def _prepare(
        self,
        x: torch.Tensor,
        channels: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tuple(channels) != KRONOS_INPUT_CHANNELS:
            raise ValueError(
                "Expected channel order "
                f"{KRONOS_INPUT_CHANNELS}, received "
                f"{tuple(channels)}."
            )

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")

        if x.ndim != 4 or x.shape[-1] != 5:
            raise ValueError(
                "x must have shape [B, T, N, 5]. Received "
                f"{tuple(x.shape)}."
            )

        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or infinite values.")

        batch_size, seq_len, num_assets, _ = x.shape

        # [B, T, N, 5] -> [B*N, T, 5], preserving asset order.
        series = (
            x.detach()
            .cpu()
            .to(torch.float32)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * num_assets, seq_len, 5)
            .numpy()
        )

        # Match the official Kronos predictor used by the project:
        # Amount is appended as an all-zero sixth channel.
        amount = np.zeros_like(
            series[..., 4:5],
            dtype=np.float32,
        )

        tokenizer_input = np.concatenate(
            [series, amount],
            axis=-1,
        ).astype(np.float32, copy=False)

        # NumPy defaults match the official predictor: population std
        # (ddof=0), calculated independently for every asset-window.
        mean = np.mean(tokenizer_input, axis=1).astype(
            np.float32,
            copy=False,
        )
        std = np.std(tokenizer_input, axis=1).astype(
            np.float32,
            copy=False,
        )

        normalised = (
            tokenizer_input - mean[:, None, :]
        ) / (
            std[:, None, :] + self.eps
        )
        normalised = np.clip(
            normalised,
            -self.clip,
            self.clip,
        ).astype(np.float32, copy=False)

        return normalised, mean, std

    def tokenize(
        self,
        x: torch.Tensor,
        *,
        channels: Sequence[str] = KRONOS_INPUT_CHANNELS,
        series_batch_size: int | None = None,
    ) -> KronosTokenBatch:
        """Tokenise raw windows shaped [B, T, N, 5]."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Call load() before tokenize()."
            )

        normalised, mean, std = self._prepare(x, channels)
        batch_size, seq_len, num_assets, _ = x.shape

        effective_batch_size = (
            self.series_batch_size
            if series_batch_size is None
            else int(series_batch_size)
        )
        if effective_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )

        coarse_parts: list[torch.Tensor] = []
        fine_parts: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(
                0,
                normalised.shape[0],
                effective_batch_size,
            ):
                stop = min(
                    start + effective_batch_size,
                    normalised.shape[0],
                )
                input_batch = torch.from_numpy(
                    normalised[start:stop]
                )
                coarse, fine = self._tokenizer.encode(
                    input_batch,
                    half=True,
                )
                coarse_parts.append(coarse.cpu().long())
                fine_parts.append(fine.cpu().long())

        coarse = torch.cat(coarse_parts, dim=0)
        fine = torch.cat(fine_parts, dim=0)

        expected_shape = (
            batch_size * num_assets,
            seq_len,
        )
        if tuple(coarse.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected coarse-token shape: "
                f"{tuple(coarse.shape)}."
            )
        if tuple(fine.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected fine-token shape: "
                f"{tuple(fine.shape)}."
            )

        if (
            coarse.min().item() < 0
            or coarse.max().item() >= 1024
            or fine.min().item() < 0
            or fine.max().item() >= 1024
        ):
            raise RuntimeError(
                "Token IDs lie outside the expected range [0, 1023]."
            )

        # [B*N, T] -> [B, T, N].
        coarse = (
            coarse.reshape(batch_size, num_assets, seq_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        fine = (
            fine.reshape(batch_size, num_assets, seq_len)
            .permute(0, 2, 1)
            .contiguous()
        )

        token_ids = torch.stack([coarse, fine], dim=-1)

        mean = torch.from_numpy(
            mean.reshape(batch_size, num_assets, 6)
        ).clone()
        std = torch.from_numpy(
            std.reshape(batch_size, num_assets, 6)
        ).clone()

        return KronosTokenBatch(
            token_ids=token_ids,
            mean=mean,
            std=std,
        )
    
    def decode(
        self,
        token_batch: KronosTokenBatch,
        *,
        series_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Decode token IDs back to raw OHLCV space.

        Args:
            token_batch:
                Tokens and context statistics returned by
                ``tokenize()``.

            series_batch_size:
                Optional number of independent asset-window series
                decoded together on CPU.

        Returns:
            Reconstructed OHLCV tensor with shape [B, T, N, 5].

        Notes:
            Kronos reconstructs all six normalised OHLCVA channels.
            The Amount channel is discarded because the adapter
            supplied it as zero and it is not part of the public
            input contract.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Call load() before decode()."
            )

        if not isinstance(token_batch, KronosTokenBatch):
            raise TypeError(
                "token_batch must be a KronosTokenBatch."
            )

        token_ids = token_batch.token_ids
        mean = token_batch.mean
        std = token_batch.std

        if token_ids.ndim != 4 or token_ids.shape[-1] != 2:
            raise ValueError(
                "token_ids must have shape [B, T, N, 2]. "
                f"Received {tuple(token_ids.shape)}."
            )

        batch_size, seq_len, num_assets, _ = (
            token_ids.shape
        )

        expected_stats_shape = (
            batch_size,
            num_assets,
            6,
        )

        if tuple(mean.shape) != expected_stats_shape:
            raise ValueError(
                "mean must have shape [B, N, 6]. "
                f"Received {tuple(mean.shape)}."
            )

        if tuple(std.shape) != expected_stats_shape:
            raise ValueError(
                "std must have shape [B, N, 6]. "
                f"Received {tuple(std.shape)}."
            )

        if not torch.isfinite(mean).all():
            raise ValueError(
                "mean contains NaN or infinite values."
            )

        if not torch.isfinite(std).all():
            raise ValueError(
                "std contains NaN or infinite values."
            )

        effective_batch_size = (
            self.series_batch_size
            if series_batch_size is None
            else int(series_batch_size)
        )

        if effective_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )

        # [B, T, N] -> [B*N, T], preserving the same ordering
        # used by tokenize().
        coarse = (
            token_ids[..., 0]
            .permute(0, 2, 1)
            .contiguous()
            .reshape(
                batch_size * num_assets,
                seq_len,
            )
            .cpu()
            .long()
        )

        fine = (
            token_ids[..., 1]
            .permute(0, 2, 1)
            .contiguous()
            .reshape(
                batch_size * num_assets,
                seq_len,
            )
            .cpu()
            .long()
        )

        decoded_parts: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(
                0,
                coarse.shape[0],
                effective_batch_size,
            ):
                stop = min(
                    start + effective_batch_size,
                    coarse.shape[0],
                )

                decoded_normalised = (
                    self._tokenizer.decode(
                        (
                            coarse[start:stop],
                            fine[start:stop],
                        ),
                        half=True,
                    )
                )

                decoded_parts.append(
                    decoded_normalised
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )

        decoded_normalised = torch.cat(
            decoded_parts,
            dim=0,
        )

        expected_decoded_shape = (
            batch_size * num_assets,
            seq_len,
            6,
        )

        if (
            tuple(decoded_normalised.shape)
            != expected_decoded_shape
        ):
            raise RuntimeError(
                "Unexpected decoded tensor shape: "
                f"{tuple(decoded_normalised.shape)}. "
                f"Expected {expected_decoded_shape}."
            )

        # Restore the flattened [B*N] ordering used during
        # tokenisation.
        flat_mean = (
            mean.detach()
            .cpu()
            .to(torch.float32)
            .reshape(
                batch_size * num_assets,
                6,
            )
        )

        flat_std = (
            std.detach()
            .cpu()
            .to(torch.float32)
            .reshape(
                batch_size * num_assets,
                6,
            )
        )

        # Invert:
        #
        # normalised = (raw - mean) / (std + eps)
        decoded_raw = (
            decoded_normalised
            * (flat_std[:, None, :] + self.eps)
            + flat_mean[:, None, :]
        )

        # [B*N, T, 6] -> [B, T, N, 6]
        decoded_raw = (
            decoded_raw
            .reshape(
                batch_size,
                num_assets,
                seq_len,
                6,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # Amount was supplied as zero and is not part of the
        # public adapter input.
        decoded_ohlcv = (
            decoded_raw[..., :5]
            .contiguous()
        )

        if not torch.isfinite(decoded_ohlcv).all():
            raise RuntimeError(
                "Decoded OHLCV contains NaN or infinite values."
            )

        return decoded_ohlcv

    def prepare_forecast_path(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
        *,
        channels: Sequence[str] = KRONOS_INPUT_CHANNELS,
    ) -> KronosNormalisedPathBatch:
        """Normalise one context-plus-future path without leakage.

        Args:
            context:
                Observed raw OHLCV with shape [B, C, N, 5].

            future:
                Future raw OHLCV labels with shape [B, P, N, 5].

            channels:
                Required public OHLCV order.

        Returns:
            A complete [B, C + P, N, 6] normalised path. Mean and
            population standard deviation are calculated only over the
            C observed context positions, separately for every asset
            and channel. Amount is appended as an all-zero channel.
        """
        if tuple(channels) != KRONOS_INPUT_CHANNELS:
            raise ValueError(
                "Expected channel order "
                f"{KRONOS_INPUT_CHANNELS}, received "
                f"{tuple(channels)}."
            )

        if not isinstance(
            context,
            torch.Tensor,
        ):
            raise TypeError(
                "context must be a torch.Tensor."
            )

        if not isinstance(
            future,
            torch.Tensor,
        ):
            raise TypeError(
                "future must be a torch.Tensor."
            )

        if (
            context.ndim != 4
            or context.shape[-1] != 5
        ):
            raise ValueError(
                "context must have shape [B, C, N, 5]. "
                f"Received {tuple(context.shape)}."
            )

        if (
            future.ndim != 4
            or future.shape[-1] != 5
        ):
            raise ValueError(
                "future must have shape [B, P, N, 5]. "
                f"Received {tuple(future.shape)}."
            )

        if (
            context.shape[0] != future.shape[0]
            or context.shape[2] != future.shape[2]
        ):
            raise ValueError(
                "context and future batch/asset dimensions "
                "must match."
            )

        if context.shape[1] <= 0:
            raise ValueError(
                "context must contain at least one timestep."
            )

        if future.shape[1] <= 0:
            raise ValueError(
                "future must contain at least one timestep."
            )

        context = (
            context
            .detach()
            .cpu()
            .to(torch.float32)
            .contiguous()
        )

        future = (
            future
            .detach()
            .cpu()
            .to(torch.float32)
            .contiguous()
        )

        if not torch.isfinite(
            context
        ).all():
            raise ValueError(
                "context contains NaN or infinite values."
            )

        if not torch.isfinite(
            future
        ).all():
            raise ValueError(
                "future contains NaN or infinite values."
            )

        context_amount = torch.zeros_like(
            context[..., 4:5]
        )

        future_amount = torch.zeros_like(
            future[..., 4:5]
        )

        context_ohlcva = torch.cat(
            [
                context,
                context_amount,
            ],
            dim=-1,
        )

        future_ohlcva = torch.cat(
            [
                future,
                future_amount,
            ],
            dim=-1,
        )

        # Statistics use the observed context only.
        #
        # Shape:
        #     mean/std [B, N, 6]
        mean = context_ohlcva.mean(
            dim=1
        )

        std = context_ohlcva.std(
            dim=1,
            unbiased=False,
        )

        full_path = torch.cat(
            [
                context_ohlcva,
                future_ohlcva,
            ],
            dim=1,
        )

        unclipped = (
            full_path
            - mean[:, None, :, :]
        ) / (
            std[:, None, :, :]
            + self.eps
        )

        clipping_mask = (
            (unclipped < -self.clip)
            | (unclipped > self.clip)
        )

        normalised = unclipped.clamp(
            min=-self.clip,
            max=self.clip,
        )

        if not torch.isfinite(
            normalised
        ).all():
            raise RuntimeError(
                "Origin-aligned normalisation produced "
                "non-finite values."
            )

        return KronosNormalisedPathBatch(
            normalised=normalised.contiguous(),
            mean=mean.contiguous(),
            std=std.contiguous(),
            clipping_mask=(
                clipping_mask.contiguous()
            ),
            context_length=int(
                context.shape[1]
            ),
            prediction_length=int(
                future.shape[1]
            ),
        )

    def encode_normalised(
        self,
        x_normalised: torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        series_batch_size: int | None = None,
    ) -> KronosTokenBatch:
        """Encode an already-normalised OHLCVA sequence.

        This method deliberately performs no normalisation. It exists
        so a 120-position context-plus-future path can use statistics
        estimated only from the observed 60-position context.

        Args:
            x_normalised:
                Clipped Kronos-space OHLCVA values with shape
                [B, T, N, 6].

            mean:
                Context-only raw-space mean with shape [B, N, 6].

            std:
                Context-only population standard deviation with shape
                [B, N, 6].

            series_batch_size:
                Number of independent asset-window series encoded per
                tokenizer call.

        Returns:
            KronosTokenBatch with token IDs [B, T, N, 2] and the
            supplied context statistics.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Call load() before encode_normalised()."
            )

        if not isinstance(
            x_normalised,
            torch.Tensor,
        ):
            raise TypeError(
                "x_normalised must be a torch.Tensor."
            )

        if (
            x_normalised.ndim != 4
            or x_normalised.shape[-1] != 6
        ):
            raise ValueError(
                "x_normalised must have shape [B, T, N, 6]. "
                f"Received {tuple(x_normalised.shape)}."
            )

        batch_size, seq_len, num_assets, _ = (
            x_normalised.shape
        )

        expected_stats_shape = (
            batch_size,
            num_assets,
            6,
        )

        mean = (
            torch.as_tensor(
                mean,
                dtype=torch.float32,
            )
            .detach()
            .cpu()
            .contiguous()
        )

        std = (
            torch.as_tensor(
                std,
                dtype=torch.float32,
            )
            .detach()
            .cpu()
            .contiguous()
        )

        if tuple(mean.shape) != (
            expected_stats_shape
        ):
            raise ValueError(
                "mean must have shape [B, N, 6]. "
                f"Received {tuple(mean.shape)}."
            )

        if tuple(std.shape) != (
            expected_stats_shape
        ):
            raise ValueError(
                "std must have shape [B, N, 6]. "
                f"Received {tuple(std.shape)}."
            )

        if not torch.isfinite(mean).all():
            raise ValueError(
                "mean contains non-finite values."
            )

        if not torch.isfinite(std).all():
            raise ValueError(
                "std contains non-finite values."
            )

        if torch.any(std < 0):
            raise ValueError(
                "std cannot contain negative values."
            )

        x_normalised = (
            x_normalised
            .detach()
            .cpu()
            .to(torch.float32)
            .contiguous()
        )

        if not torch.isfinite(
            x_normalised
        ).all():
            raise ValueError(
                "x_normalised contains NaN or infinite values."
            )

        tolerance = 1.0e-5

        if (
            x_normalised.min().item()
            < -self.clip - tolerance
            or x_normalised.max().item()
            > self.clip + tolerance
        ):
            raise ValueError(
                "x_normalised lies outside the configured "
                f"[-{self.clip}, {self.clip}] range."
            )

        effective_batch_size = (
            self.series_batch_size
            if series_batch_size is None
            else int(series_batch_size)
        )

        if effective_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )

        # [B, T, N, 6] -> [B*N, T, 6], preserving asset order.
        flat = (
            x_normalised
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(
                batch_size * num_assets,
                seq_len,
                6,
            )
        )

        coarse_parts: list[torch.Tensor] = []
        fine_parts: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(
                0,
                flat.shape[0],
                effective_batch_size,
            ):
                stop = min(
                    start + effective_batch_size,
                    flat.shape[0],
                )

                indices = self._tokenizer.encode(
                    flat[start:stop],
                    half=True,
                )

                if (
                    not isinstance(
                        indices,
                        (tuple, list),
                    )
                    or len(indices) != 2
                ):
                    raise RuntimeError(
                        "Official tokenizer.encode(..., half=True) "
                        "did not return coarse and fine IDs."
                    )

                coarse_parts.append(
                    indices[0]
                    .detach()
                    .cpu()
                    .long()
                )

                fine_parts.append(
                    indices[1]
                    .detach()
                    .cpu()
                    .long()
                )

        coarse_flat = torch.cat(
            coarse_parts,
            dim=0,
        )

        fine_flat = torch.cat(
            fine_parts,
            dim=0,
        )

        expected_flat_shape = (
            batch_size * num_assets,
            seq_len,
        )

        if tuple(coarse_flat.shape) != (
            expected_flat_shape
        ):
            raise RuntimeError(
                "Unexpected coarse token shape: "
                f"{tuple(coarse_flat.shape)}. "
                f"Expected {expected_flat_shape}."
            )

        if tuple(fine_flat.shape) != (
            expected_flat_shape
        ):
            raise RuntimeError(
                "Unexpected fine token shape: "
                f"{tuple(fine_flat.shape)}. "
                f"Expected {expected_flat_shape}."
            )

        def restore_asset_axis(
            values: torch.Tensor,
        ) -> torch.Tensor:
            return (
                values
                .reshape(
                    batch_size,
                    num_assets,
                    seq_len,
                )
                .permute(0, 2, 1)
                .contiguous()
            )

        coarse = restore_asset_axis(
            coarse_flat
        )

        fine = restore_asset_axis(
            fine_flat
        )

        token_ids = torch.stack(
            [
                coarse,
                fine,
            ],
            dim=-1,
        )

        if (
            token_ids.min().item() < 0
            or token_ids.max().item() >= 1024
        ):
            raise RuntimeError(
                "Encoded token IDs lie outside [0, 1023]."
            )

        return KronosTokenBatch(
            token_ids=token_ids,
            mean=mean,
            std=std,
        )

    def decode_token_path(
        self,
        context_token_ids: torch.Tensor,
        future_token_ids: torch.Tensor,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        series_batch_size: int | None = None,
        return_full_path: bool = False,
    ) -> torch.Tensor:
        """Decode an observed context plus complete predicted path.

        Args:
            context_token_ids:
                Observed token pairs [B, C, N, 2].

            future_token_ids:
                Complete future token path [B, P, N, 2].

            mean/std:
                Context-only raw-space statistics [B, N, 6].

            return_full_path:
                When False, return only the P decoded future bars.
                When True, return all C + P decoded bars.

        Returns:
            Raw OHLCV with shape [B, P, N, 5] by default.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Call load() before decode_token_path()."
            )

        context_token_ids = torch.as_tensor(
            context_token_ids
        ).detach().cpu()

        future_token_ids = torch.as_tensor(
            future_token_ids
        ).detach().cpu()

        if (
            context_token_ids.ndim != 4
            or context_token_ids.shape[-1] != 2
        ):
            raise ValueError(
                "context_token_ids must have shape [B, C, N, 2]."
            )

        if (
            future_token_ids.ndim != 4
            or future_token_ids.shape[-1] != 2
        ):
            raise ValueError(
                "future_token_ids must have shape [B, P, N, 2]."
            )

        if (
            context_token_ids.shape[0]
            != future_token_ids.shape[0]
            or context_token_ids.shape[2]
            != future_token_ids.shape[2]
        ):
            raise ValueError(
                "Context and future token batch/asset dimensions "
                "must match."
            )

        if context_token_ids.shape[1] <= 0:
            raise ValueError(
                "context_token_ids must contain at least one "
                "position."
            )

        if future_token_ids.shape[1] <= 0:
            raise ValueError(
                "future_token_ids must contain at least one "
                "position."
            )

        full_token_ids = torch.cat(
            [
                context_token_ids,
                future_token_ids,
            ],
            dim=1,
        ).long()

        if (
            full_token_ids.min().item() < 0
            or full_token_ids.max().item() >= 1024
        ):
            raise ValueError(
                "Token IDs lie outside [0, 1023]."
            )

        token_batch = KronosTokenBatch(
            token_ids=full_token_ids,
            mean=torch.as_tensor(
                mean,
                dtype=torch.float32,
            ),
            std=torch.as_tensor(
                std,
                dtype=torch.float32,
            ),
        )

        decoded_full = self.decode(
            token_batch,
            series_batch_size=series_batch_size,
        )

        if return_full_path:
            return decoded_full

        context_length = int(
            context_token_ids.shape[1]
        )

        return (
            decoded_full[
                :,
                context_length:,
            ]
            .contiguous()
        )

    def decode_coarse(
        self,
        token_batch: KronosTokenBatch,
        *,
        series_batch_size: int | None = None,
    ) -> torch.Tensor:
        """Decode coarse token IDs back to raw OHLCV space.

        This follows the tokenizer's trained coarse-only reconstruction
        branch:

            s1 IDs
            -> 10-bit bipolar coarse code
            -> post_quant_embed_pre
            -> shared decoder
            -> reconstructed OHLCVA
            -> inverse normalisation
            -> reconstructed OHLCV

        Args:
            token_batch:
                Tokens and normalisation statistics returned by
                ``tokenize()``.

            series_batch_size:
                Optional number of independent asset-window series
                decoded together.

        Returns:
            Coarse-only reconstructed OHLCV with shape [B, T, N, 5].
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "Call load() before decode_coarse()."
            )

        if not isinstance(token_batch, KronosTokenBatch):
            raise TypeError(
                "token_batch must be a KronosTokenBatch."
            )

        token_ids = token_batch.token_ids
        mean = token_batch.mean
        std = token_batch.std

        if token_ids.ndim != 4 or token_ids.shape[-1] != 2:
            raise ValueError(
                "token_ids must have shape [B, T, N, 2]. "
                f"Received {tuple(token_ids.shape)}."
            )

        batch_size, seq_len, num_assets, _ = (
            token_ids.shape
        )

        expected_stats_shape = (
            batch_size,
            num_assets,
            6,
        )

        if tuple(mean.shape) != expected_stats_shape:
            raise ValueError(
                "mean must have shape [B, N, 6]. "
                f"Received {tuple(mean.shape)}."
            )

        if tuple(std.shape) != expected_stats_shape:
            raise ValueError(
                "std must have shape [B, N, 6]. "
                f"Received {tuple(std.shape)}."
            )

        if not torch.isfinite(mean).all():
            raise ValueError(
                "mean contains NaN or infinite values."
            )

        if not torch.isfinite(std).all():
            raise ValueError(
                "std contains NaN or infinite values."
            )

        effective_batch_size = (
            self.series_batch_size
            if series_batch_size is None
            else int(series_batch_size)
        )

        if effective_batch_size <= 0:
            raise ValueError(
                "series_batch_size must be greater than zero."
            )

        # [B, T, N] -> [B*N, T]
        coarse = (
            token_ids[..., 0]
            .permute(0, 2, 1)
            .contiguous()
            .reshape(
                batch_size * num_assets,
                seq_len,
            )
            .cpu()
            .long()
        )

        tokenizer_parameter = next(
            self._tokenizer.parameters()
        )

        tokenizer_device = tokenizer_parameter.device
        tokenizer_dtype = tokenizer_parameter.dtype

        s1_bits = int(
            self._tokenizer.s1_bits
        )

        codebook_dim = int(
            self._tokenizer.codebook_dim
        )

        bit_mask = (
            2
            ** torch.arange(
                s1_bits,
                device=tokenizer_device,
                dtype=torch.long,
            )
        )

        decoded_parts: list[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(
                0,
                coarse.shape[0],
                effective_batch_size,
            ):
                stop = min(
                    start + effective_batch_size,
                    coarse.shape[0],
                )

                coarse_batch = coarse[
                    start:stop
                ].to(tokenizer_device)

                # Recover the 10 coarse BSQ bits using the same
                # least-significant-bit-first ordering as the
                # official tokenizer.
                quantized_coarse = (
                    (
                        coarse_batch.unsqueeze(-1)
                        & bit_mask
                    )
                    != 0
                ).to(tokenizer_dtype)

                # {0, 1} -> {-1, 1}
                quantized_coarse = (
                    quantized_coarse * 2.0 - 1.0
                )

                # The tokenizer scales the complete 20-bit code by
                # 1 / sqrt(codebook_dim). The coarse branch receives
                # the first 10 components with the same scaling.
                quantized_coarse = (
                    quantized_coarse
                    / math.sqrt(codebook_dim)
                )

                decoded_normalised = (
                    self._tokenizer
                    .post_quant_embed_pre(
                        quantized_coarse
                    )
                )

                for layer in self._tokenizer.decoder:
                    decoded_normalised = layer(
                        decoded_normalised
                    )

                decoded_normalised = (
                    self._tokenizer.head(
                        decoded_normalised
                    )
                )

                decoded_parts.append(
                    decoded_normalised
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )

        decoded_normalised = torch.cat(
            decoded_parts,
            dim=0,
        )

        expected_decoded_shape = (
            batch_size * num_assets,
            seq_len,
            6,
        )

        if tuple(decoded_normalised.shape) != (
            expected_decoded_shape
        ):
            raise RuntimeError(
                "Unexpected coarse decoded shape: "
                f"{tuple(decoded_normalised.shape)}. "
                f"Expected {expected_decoded_shape}."
            )

        flat_mean = (
            mean.detach()
            .cpu()
            .to(torch.float32)
            .reshape(
                batch_size * num_assets,
                6,
            )
        )

        flat_std = (
            std.detach()
            .cpu()
            .to(torch.float32)
            .reshape(
                batch_size * num_assets,
                6,
            )
        )

        decoded_raw = (
            decoded_normalised
            * (flat_std[:, None, :] + self.eps)
            + flat_mean[:, None, :]
        )

        # [B*N, T, 6] -> [B, T, N, 6]
        decoded_raw = (
            decoded_raw
            .reshape(
                batch_size,
                num_assets,
                seq_len,
                6,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # Remove the synthetic Amount channel.
        decoded_ohlcv = (
            decoded_raw[..., :5]
            .contiguous()
        )

        if not torch.isfinite(decoded_ohlcv).all():
            raise RuntimeError(
                "Coarse decoded OHLCV contains NaN or "
                "infinite values."
            )

        return decoded_ohlcv


def encode_causal_split(
    tokenizer: KronosTokenizerAdapter,
    split: Mapping[str, Any],
    *,
    context_length: int = 60,
    window_batch_size: int = 8,
    series_batch_size: int = 93,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Encode every bar with a complete trailing causal context.

    For each session and bar t >= context_length - 1, the tokenizer
    receives only:

        [t - context_length + 1, ..., t]

    The complete token sequence is retained so it can later be passed
    through the contextual Kronos decoder. The final token at each bar
    is also stored separately for token-level analysis.

    Args:
        tokenizer:
            Loaded Kronos tokenizer adapter.

        split:
            Cleaned candle split containing ``samples``, ``channels``
            and ``asset_cols``.

        context_length:
            Number of trailing bars supplied to the tokenizer.

        window_batch_size:
            Number of rolling contexts processed together.

        series_batch_size:
            Number of independent asset series processed together by
            the underlying tokenizer.

        show_progress:
            Whether to display a session-level progress bar.

    Returns:
        Dictionary ready to save as ``encoded_data.pt``.
    """
    if context_length <= 0:
        raise ValueError(
            "context_length must be greater than zero."
        )

    if window_batch_size <= 0:
        raise ValueError(
            "window_batch_size must be greater than zero."
        )

    if series_batch_size <= 0:
        raise ValueError(
            "series_batch_size must be greater than zero."
        )

    samples = list(split["samples"])

    if not samples:
        raise ValueError(
            "The supplied split contains no samples."
        )

    split_channels = list(split["channels"])
    asset_cols = list(split["asset_cols"])

    channel_ids = torch.tensor(
        [
            split_channels.index(channel)
            for channel in KRONOS_INPUT_CHANNELS
        ],
        dtype=torch.long,
    )

    first_day = torch.as_tensor(
        samples[0][0],
        dtype=torch.float32,
    )

    if first_day.ndim != 3:
        raise ValueError(
            "Each session must have shape [T, N, D]."
        )

    num_bars, num_assets, _ = first_day.shape
    num_sessions = len(samples)

    if num_assets != len(asset_cols):
        raise ValueError(
            "The number of assets does not match asset_cols."
        )

    if context_length > num_bars:
        raise ValueError(
            "context_length exceeds the number of bars."
        )

    origin_indices = torch.arange(
        context_length - 1,
        num_bars,
        dtype=torch.long,
    )

    num_origins = int(origin_indices.numel())

    # Complete rolling token sequences required by the decoder.
    #
    # int16 is sufficient because valid IDs lie in [0, 1023].
    context_s1 = torch.empty(
        (
            num_sessions,
            num_origins,
            context_length,
            num_assets,
        ),
        dtype=torch.int16,
    )

    context_s2 = torch.empty_like(
        context_s1
    )

    # Normalisation statistics required to restore raw OHLCV.
    context_mean = torch.empty(
        (
            num_sessions,
            num_origins,
            num_assets,
            6,
        ),
        dtype=torch.float32,
    )

    context_std = torch.empty_like(
        context_mean
    )

    # One final token pair per original bar. The first
    # context_length - 1 bars are invalid.
    s1 = torch.full(
        (
            num_sessions,
            num_bars,
            num_assets,
        ),
        fill_value=-1,
        dtype=torch.int16,
    )

    s2 = torch.full_like(
        s1,
        fill_value=-1,
    )

    valid_mask = torch.zeros(
        (
            num_sessions,
            num_bars,
        ),
        dtype=torch.bool,
    )

    valid_mask[:, origin_indices] = True

    dates: list[str] = []

    session_iterator = enumerate(samples)

    if show_progress:
        session_iterator = tqdm(
            session_iterator,
            total=num_sessions,
            desc="Encoding Kronos contexts",
        )

    for session_idx, sample in session_iterator:
        x_day, _, sample_day = sample

        x_day = torch.as_tensor(
            x_day,
            dtype=torch.float32,
        )

        if tuple(x_day.shape) != (
            num_bars,
            num_assets,
            len(split_channels),
        ):
            raise ValueError(
                "All sessions must have the same tensor shape. "
                f"Session {session_idx} has "
                f"{tuple(x_day.shape)}."
            )

        if not torch.isfinite(x_day).all():
            raise ValueError(
                f"Session {session_idx} contains non-finite values."
            )

        dates.append(
            str(sample_day.date())
            if hasattr(sample_day, "date")
            else str(sample_day)
        )

        x_ohlcv = (
            x_day
            .index_select(
                dim=2,
                index=channel_ids,
            )
            .contiguous()
        )

        # unfold returns [K, N, D, context_length].
        #
        # Convert to:
        #   [K, context_length, N, D]
        contexts = (
            x_ohlcv
            .unfold(
                dimension=0,
                size=context_length,
                step=1,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        expected_context_shape = (
            num_origins,
            context_length,
            num_assets,
            len(KRONOS_INPUT_CHANNELS),
        )

        if tuple(contexts.shape) != expected_context_shape:
            raise RuntimeError(
                "Unexpected rolling-context shape: "
                f"{tuple(contexts.shape)}. Expected "
                f"{expected_context_shape}."
            )

        for start in range(
            0,
            num_origins,
            window_batch_size,
        ):
            stop = min(
                start + window_batch_size,
                num_origins,
            )

            token_batch = tokenizer.tokenize(
                contexts[start:stop],
                channels=KRONOS_INPUT_CHANNELS,
                series_batch_size=series_batch_size,
            )

            batch_s1 = (
                token_batch.coarse
                .to(torch.int16)
            )

            batch_s2 = (
                token_batch.fine
                .to(torch.int16)
            )

            context_s1[
                session_idx,
                start:stop,
            ] = batch_s1

            context_s2[
                session_idx,
                start:stop,
            ] = batch_s2

            context_mean[
                session_idx,
                start:stop,
            ] = token_batch.mean

            context_std[
                session_idx,
                start:stop,
            ] = token_batch.std

            batch_origins = origin_indices[
                start:stop
            ]

            s1[
                session_idx,
                batch_origins,
            ] = batch_s1[:, -1]

            s2[
                session_idx,
                batch_origins,
            ] = batch_s2[:, -1]

    return {
        "format_version": 1,
        "kind": "kronos_causal_rolling_tokens",
        "context_s1": context_s1,
        "context_s2": context_s2,
        "context_mean": context_mean,
        "context_std": context_std,
        "s1": s1,
        "s2": s2,
        "valid_mask": valid_mask,
        "origin_indices": origin_indices,
        "dates": dates,
        "asset_cols": asset_cols,
        "channels": list(KRONOS_INPUT_CHANNELS),
        "tokenizer_channels": list(
            KRONOS_TOKENIZER_CHANNELS
        ),
        "context_length": context_length,
        "num_bars": num_bars,
        "zero_amount": True,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_revision": (
            tokenizer.tokenizer_revision
        ),
        "clip": tokenizer.clip,
        "eps": tokenizer.eps,
    }


def decode_causal_split(
    tokenizer: KronosTokenizerAdapter,
    encoded_data: Mapping[str, Any],
    *,
    window_batch_size: int = 8,
    series_batch_size: int = 93,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Decode cached rolling contexts back to raw OHLCV space.

    Only the final reconstructed value from each causal context is
    retained. This produces one reconstruction for every original bar
    with a complete trailing context.

    Returns:
        Dictionary ready to save as ``decoded_data.pt``. Its decoded
        tensor has shape [session, bar, asset, 5].
    """
    if window_batch_size <= 0:
        raise ValueError(
            "window_batch_size must be greater than zero."
        )

    if series_batch_size <= 0:
        raise ValueError(
            "series_batch_size must be greater than zero."
        )

    if encoded_data["tokenizer_id"] != tokenizer.tokenizer_id:
        raise ValueError(
            "Encoded data uses a different tokenizer ID."
        )

    if (
        encoded_data["tokenizer_revision"]
        != tokenizer.tokenizer_revision
    ):
        raise ValueError(
            "Encoded data uses a different tokenizer revision."
        )

    context_s1 = torch.as_tensor(
        encoded_data["context_s1"]
    )

    context_s2 = torch.as_tensor(
        encoded_data["context_s2"]
    )

    context_mean = torch.as_tensor(
        encoded_data["context_mean"],
        dtype=torch.float32,
    )

    context_std = torch.as_tensor(
        encoded_data["context_std"],
        dtype=torch.float32,
    )

    origin_indices = torch.as_tensor(
        encoded_data["origin_indices"],
        dtype=torch.long,
    )

    valid_mask = torch.as_tensor(
        encoded_data["valid_mask"],
        dtype=torch.bool,
    )

    if context_s1.shape != context_s2.shape:
        raise ValueError(
            "context_s1 and context_s2 shapes do not match."
        )

    if context_s1.ndim != 4:
        raise ValueError(
            "Context tokens must have shape [S, K, T, N]."
        )

    (
        num_sessions,
        num_origins,
        context_length,
        num_assets,
    ) = context_s1.shape

    num_bars = int(encoded_data["num_bars"])

    expected_stats_shape = (
        num_sessions,
        num_origins,
        num_assets,
        6,
    )

    if tuple(context_mean.shape) != expected_stats_shape:
        raise ValueError(
            "Unexpected context_mean shape."
        )

    if tuple(context_std.shape) != expected_stats_shape:
        raise ValueError(
            "Unexpected context_std shape."
        )

    decoded_shape = (
        num_sessions,
        num_bars,
        num_assets,
        len(KRONOS_INPUT_CHANNELS),
    )

    decoded_full = torch.full(
        decoded_shape,
        fill_value=float("nan"),
        dtype=torch.float32,
    )

    decoded_coarse = torch.full(
        decoded_shape,
        fill_value=float("nan"),
        dtype=torch.float32,
    )

    session_iterator = range(num_sessions)

    if show_progress:
        session_iterator = tqdm(
            session_iterator,
            total=num_sessions,
            desc="Decoding Kronos contexts",
        )

    for session_idx in session_iterator:
        for start in range(
            0,
            num_origins,
            window_batch_size,
        ):
            stop = min(
                start + window_batch_size,
                num_origins,
            )

            token_ids = torch.stack(
                [
                    context_s1[
                        session_idx,
                        start:stop,
                    ].long(),
                    context_s2[
                        session_idx,
                        start:stop,
                    ].long(),
                ],
                dim=-1,
            )

            token_batch = KronosTokenBatch(
                token_ids=token_ids,
                mean=context_mean[
                    session_idx,
                    start:stop,
                ],
                std=context_std[
                    session_idx,
                    start:stop,
                ],
            )

            decoded_full_contexts = tokenizer.decode(
                token_batch,
                series_batch_size=series_batch_size,
            )

            decoded_coarse_contexts = (
                tokenizer.decode_coarse(
                    token_batch,
                    series_batch_size=series_batch_size,
                )
            )

            batch_origins = origin_indices[
                start:stop
            ]

            # Each stored bar receives only the final reconstruction
            # from its own trailing 60-bar causal context.
            decoded_full[
                session_idx,
                batch_origins,
            ] = decoded_full_contexts[:, -1]

            decoded_coarse[
                session_idx,
                batch_origins,
            ] = decoded_coarse_contexts[:, -1]

    if not torch.isfinite(
        decoded_full[valid_mask]
    ).all():
        raise RuntimeError(
            "Valid full-decoded positions contain non-finite "
            "values."
        )

    if not torch.isfinite(
        decoded_coarse[valid_mask]
    ).all():
        raise RuntimeError(
            "Valid coarse-decoded positions contain non-finite "
            "values."
        )

    return {
        "format_version": 2,
        "kind": "kronos_causal_reconstruction",
        "decoded_full": decoded_full,
        "decoded_coarse": decoded_coarse,
        "valid_mask": valid_mask.clone(),
        "origin_indices": origin_indices.clone(),
        "dates": list(encoded_data["dates"]),
        "asset_cols": list(
            encoded_data["asset_cols"]
        ),
        "channels": list(KRONOS_INPUT_CHANNELS),
        "reconstruction_modes": [
            "coarse",
            "full",
        ],
        "context_length": context_length,
        "num_bars": num_bars,
        "zero_amount": True,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_revision": (
            tokenizer.tokenizer_revision
        ),
    }
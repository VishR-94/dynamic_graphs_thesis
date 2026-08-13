from __future__ import annotations

"""Trainable coarse-token Kronos decoder used for decoder post-training.

The forecasting model, tokenizer encoder, quantizer, and token definitions stay
frozen.  This module contains only the pretrained coarse reconstruction branch
that is actually used by ``KronosTokenizerAdapter.decode_coarse_token_path``:

    coarse s1 IDs
    -> fixed 10-bit bipolar BSQ code
    -> post_quant_embed_pre
    -> tokenizer decoder stack
    -> reconstruction head

Because the branch receives hard token IDs, gradients flow only into the
reconstruction decoder.  No straight-through estimator is required.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from src.models.kronos import import_official_kronos


@dataclass(frozen=True)
class KronosDecoderSource:
    tokenizer_id: str
    tokenizer_revision: str
    s1_bits: int
    codebook_dim: int
    output_channels: int


class TrainableKronosCoarseDecoder(nn.Module):
    """Differentiable coarse-only Kronos reconstruction branch.

    The module accepts the same true observed coarse-token context and sampled
    future coarse-token paths used by the existing frozen decoding pipeline.

    Shapes:
        context_s1:
            [B, C, N]

        future_s1_paths:
            [S, B, P, N]

        mean/std:
            [B, N, 6]

        output:
            [S, B, C + P, N, 5] in raw OHLCV space.
    """

    def __init__(
        self,
        *,
        post_quant_embed_pre: nn.Module,
        decoder_layers: nn.ModuleList,
        reconstruction_head: nn.Module,
        s1_bits: int,
        codebook_dim: int,
        eps: float = 1.0e-5,
        source: KronosDecoderSource | None = None,
    ) -> None:
        super().__init__()

        self.post_quant_embed_pre = post_quant_embed_pre
        self.decoder_layers = decoder_layers
        self.reconstruction_head = reconstruction_head
        self.s1_bits = int(s1_bits)
        self.codebook_dim = int(codebook_dim)
        self.eps = float(eps)
        self.source = source

        if self.s1_bits <= 0:
            raise ValueError("s1_bits must be positive.")
        if self.codebook_dim <= 0:
            raise ValueError("codebook_dim must be positive.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

        self.register_buffer(
            "bit_mask",
            2 ** torch.arange(self.s1_bits, dtype=torch.long),
            persistent=False,
        )

        # The wrapper contains only the decoder-side modules.  Every registered
        # parameter is intentionally trainable; the forecaster and tokenizer
        # encoder/quantizer are not part of this module at all.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    @classmethod
    def from_forecasting_config(
        cls,
        config: Mapping[str, Any],
        *,
        eps: float = 1.0e-5,
    ) -> "TrainableKronosCoarseDecoder":
        kronos = config["models"]["kronos"]
        tokenizer_id = str(kronos["tokenizer_id"])
        tokenizer_revision = str(kronos["tokenizer_revision"])

        _, _, OfficialKronosTokenizer = import_official_kronos()
        tokenizer = OfficialKronosTokenizer.from_pretrained(
            tokenizer_id,
            revision=tokenizer_revision,
        ).to("cpu")

        s1_bits = int(getattr(tokenizer, "s1_bits"))
        codebook_dim = int(getattr(tokenizer, "codebook_dim"))
        output_channels = int(getattr(tokenizer, "d_in"))
        if s1_bits != 10:
            raise ValueError(
                "The post-training experiment expects the pinned 10-bit "
                f"coarse token stream, received {s1_bits} bits."
            )
        if output_channels != 6:
            raise ValueError(
                "The pinned tokenizer must reconstruct six OHLCVA channels."
            )

        # Move the exact pretrained reconstruction modules into a small wrapper.
        # No encoder or quantizer parameters are registered or optimised.
        return cls(
            post_quant_embed_pre=tokenizer.post_quant_embed_pre,
            decoder_layers=nn.ModuleList(list(tokenizer.decoder)),
            reconstruction_head=tokenizer.head,
            s1_bits=s1_bits,
            codebook_dim=codebook_dim,
            eps=eps,
            source=KronosDecoderSource(
                tokenizer_id=tokenizer_id,
                tokenizer_revision=tokenizer_revision,
                s1_bits=s1_bits,
                codebook_dim=codebook_dim,
                output_channels=output_channels,
            ),
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _decode_flat_ids(self, coarse_ids: Tensor) -> Tensor:
        """Decode flat coarse IDs ``[Q,T]`` to normalised OHLCVA ``[Q,T,6]``."""
        if coarse_ids.ndim != 2:
            raise ValueError(
                "coarse_ids must have shape [Q,T], received "
                f"{tuple(coarse_ids.shape)}."
            )
        coarse_ids = coarse_ids.to(device=self.bit_mask.device, dtype=torch.long)
        if coarse_ids.numel() == 0:
            raise ValueError("coarse_ids must not be empty.")
        if int(coarse_ids.min().item()) < 0 or int(coarse_ids.max().item()) >= 2 ** self.s1_bits:
            raise ValueError(
                f"coarse_ids must lie in [0, {2 ** self.s1_bits - 1}]."
            )

        parameter = next(self.parameters())
        quantized = (
            (coarse_ids.unsqueeze(-1) & self.bit_mask) != 0
        ).to(dtype=parameter.dtype)
        quantized = (quantized * 2.0 - 1.0) / math.sqrt(self.codebook_dim)

        hidden = self.post_quant_embed_pre(quantized)
        for layer in self.decoder_layers:
            hidden = layer(hidden)
        return self.reconstruction_head(hidden)

    def decode_paths(
        self,
        *,
        context_s1: Tensor,
        future_s1_paths: Tensor,
        mean: Tensor,
        std: Tensor,
        future_only: bool = True,
    ) -> Tensor:
        """Decode multiple sampled paths through the trainable coarse branch."""
        if context_s1.ndim != 3:
            raise ValueError("context_s1 must have shape [B,C,N].")
        if future_s1_paths.ndim != 4:
            raise ValueError("future_s1_paths must have shape [S,B,P,N].")

        samples, batch_size, prediction_length, num_assets = (
            int(value) for value in future_s1_paths.shape
        )
        if tuple(context_s1.shape[::2]) != (batch_size, num_assets):
            raise ValueError("Context and future batch/asset axes do not align.")
        context_length = int(context_s1.shape[1])
        if samples <= 0 or prediction_length <= 0 or context_length <= 0:
            raise ValueError("Sample, context, and prediction lengths must be positive.")

        expected_stats = (batch_size, num_assets, 6)
        if tuple(mean.shape) != expected_stats or tuple(std.shape) != expected_stats:
            raise ValueError(
                "mean/std must have shape [B,N,6]; received "
                f"{tuple(mean.shape)} and {tuple(std.shape)}."
            )

        device = self.bit_mask.device
        context_s1 = context_s1.to(device=device, dtype=torch.long)
        future_s1_paths = future_s1_paths.to(device=device, dtype=torch.long)
        mean = mean.to(device=device, dtype=torch.float32)
        std = std.to(device=device, dtype=torch.float32)

        expanded_context = context_s1.unsqueeze(0).expand(
            samples, -1, -1, -1
        )
        full_ids = torch.cat((expanded_context, future_s1_paths), dim=2)
        total_length = context_length + prediction_length

        # [S,B,T,N] -> [S*B*N,T].
        flat_ids = (
            full_ids.permute(0, 1, 3, 2)
            .contiguous()
            .reshape(samples * batch_size * num_assets, total_length)
        )
        decoded_normalised = self._decode_flat_ids(flat_ids)
        if tuple(decoded_normalised.shape[:2]) != (
            samples * batch_size * num_assets,
            total_length,
        ):
            raise RuntimeError(
                "Unexpected decoder output shape: "
                f"{tuple(decoded_normalised.shape)}."
            )
        if int(decoded_normalised.shape[-1]) != 6:
            raise RuntimeError(
                "The Kronos reconstruction head must emit six channels."
            )

        decoded_normalised = (
            decoded_normalised.reshape(
                samples, batch_size, num_assets, total_length, 6
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        raw = decoded_normalised.float() * (
            std.unsqueeze(0).unsqueeze(2) + self.eps
        ) + mean.unsqueeze(0).unsqueeze(2)
        raw_ohlcv = raw[..., :5].contiguous()

        if future_only:
            raw_ohlcv = raw_ohlcv[:, :, context_length:]
        return raw_ohlcv


def decoder_state_dict_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

from __future__ import annotations

"""Post-BSQ token input adapter for the official per-asset ModernTCN.

The frozen Kronos encoder is represented by the origin-aligned token cache.
Each saved ``s1``/``s2`` ID is converted back into the exact 20-dimensional
bipolar BSQ code (least-significant-bit first, scaled by ``1/sqrt(20)``).
The official ModernTCN then processes each asset independently. Cross-asset
information first enters through the explicit graph learner downstream.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Type

import torch
from torch import Tensor, nn

from .contracts import DynamicGraphModelConfig


KRONOS_BITS_PER_SUBTOKEN = 10
KRONOS_CODEBOOK_DIM = 20
KRONOS_TOKEN_VOCABULARY_SIZE = 2 ** KRONOS_BITS_PER_SUBTOKEN


def _ids_to_bits(
    ids: Tensor,
    *,
    num_bits: int = KRONOS_BITS_PER_SUBTOKEN,
) -> Tensor:
    """Convert integer IDs to least-significant-bit-first binary codes."""
    values = torch.as_tensor(ids)
    if values.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
        torch.uint8,
    }:
        raise TypeError("Token IDs must use an integer dtype.")
    values = values.long()
    if values.numel() and (
        values.min().item() < 0
        or values.max().item() >= 2 ** int(num_bits)
    ):
        raise ValueError(
            f"Token IDs must lie in [0, {2 ** int(num_bits) - 1}]."
        )
    shifts = torch.arange(
        int(num_bits),
        device=values.device,
        dtype=torch.long,
    )
    return ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def token_ids_to_bsq_codes(token_ids: Tensor) -> Tensor:
    """Recover the exact post-BSQ 20-dimensional code from token IDs.

    Args:
        token_ids:
            Original Kronos IDs with shape ``[B, T, N, 2]``. The final
            axis is ``[s1, s2]``.

    Returns:
        Float32 code tensor ``[B, T, N, 20]`` in the same scale used by
        the tokenizer immediately after BSQ: each component is
        ``{-1, +1} / sqrt(20)``.
    """
    values = torch.as_tensor(token_ids)
    if values.ndim != 4 or int(values.shape[-1]) != 2:
        raise ValueError("token_ids must have shape [B, T, N, 2].")

    coarse = _ids_to_bits(values[..., 0])
    fine = _ids_to_bits(values[..., 1])
    bits = torch.cat((coarse, fine), dim=-1)
    bipolar = bits.mul(2.0).sub(1.0)
    return bipolar.div(math.sqrt(KRONOS_CODEBOOK_DIM)).contiguous()


class ModernTCNTokenEncoder(nn.Module):
    """Official one-stage ModernTCN on post-BSQ token-code channels.

    Tensor flow:

        token IDs                 [B, 60, N, 2]
        post-BSQ code             [B, 60, N, 20]
        assets folded into batch  [B*N, 20, 60]
        official features         [B*N, 20, D, P]
        learned bit-stream pool   [B*N, D, P]
        restored asset axis       [B, P, N, D]

    The 20 BSQ dimensions are treated as ModernTCN variables. The official
    ConvFFN2 therefore mixes coarse/fine code dimensions within one asset.
    A tiny learned pooling layer, initialised to a uniform mean, converts the
    20 output streams into one asset representation per patch. Assets cannot
    interact before the explicit graph stage.
    """

    def __init__(
        self,
        config: DynamicGraphModelConfig,
        *,
        official_model_cls: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.temporal.type != "modern_tcn":
            raise ValueError("ModernTCNTokenEncoder requires modern_tcn.")
        if config.token_input_representation != "bsq_bits":
            raise ValueError("ModernTCNTokenEncoder requires bsq_bits input.")

        temporal = config.temporal
        self.context_length = int(config.context_length)
        self.num_nodes = int(config.num_nodes)
        self.d_model = int(config.d_model)
        self.patch_size = int(temporal.modern_tcn_patch_size)
        self.patch_stride = int(temporal.modern_tcn_patch_stride)
        self.output_length = int(config.temporal_output_length)
        self.num_token_variables = KRONOS_CODEBOOK_DIM

        if official_model_cls is None:
            project_root = Path(__file__).resolve().parents[3]
            modern_tcn_root = (
                project_root
                / "external"
                / "ModernTCN"
                / "ModernTCN-Long-term-forecasting"
            )
            if not modern_tcn_root.is_dir():
                raise FileNotFoundError(
                    "Initialise external/ModernTCN before using the token "
                    "ModernTCN backbone."
                )
            root_string = str(modern_tcn_root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)
            from models.ModernTCN import Model as OfficialModernTCNModel

            official_model_cls = OfficialModernTCNModel

        official_config = SimpleNamespace(
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=int(temporal.modern_tcn_ffn_ratio),
            num_blocks=[int(temporal.modern_tcn_num_blocks)],
            large_size=[int(temporal.modern_tcn_large_kernel)],
            small_size=[int(temporal.modern_tcn_small_kernel)],
            dims=[self.d_model] * 4,
            dw_dims=[self.d_model] * 4,
            enc_in=self.num_token_variables,
            small_kernel_merged=False,
            dropout=float(temporal.modern_tcn_dropout),
            head_dropout=0.0,
            use_multi_scale=False,
            revin=0,
            affine=0,
            subtract_last=0,
            freq="t",
            seq_len=self.context_length,
            pred_len=int(config.prediction_length),
            individual=0,
            decomposition=0,
            kernel_size=25,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )
        self.official_model = official_model_cls(official_config)
        if not hasattr(self.official_model, "model"):
            raise TypeError("Official ModernTCN model must expose .model.")
        if not hasattr(self.official_model.model, "forward_feature"):
            raise TypeError(
                "Official ModernTCN backbone must expose forward_feature()."
            )

        self.variable_pool = nn.Linear(
            self.num_token_variables,
            1,
            bias=False,
        )
        nn.init.constant_(
            self.variable_pool.weight,
            1.0 / self.num_token_variables,
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        values = torch.as_tensor(token_ids)
        expected = (
            int(values.shape[0]),
            self.context_length,
            self.num_nodes,
            2,
        ) if values.ndim == 4 else None
        if values.ndim != 4 or tuple(values.shape) != expected:
            raise ValueError(
                "token_ids does not match [B, context_length, N, 2]."
            )

        code = token_ids_to_bsq_codes(values).to(device=values.device)
        batch_size = int(code.shape[0])
        per_asset = (
            code.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(
                batch_size * self.num_nodes,
                self.context_length,
                self.num_token_variables,
            )
        )
        channels_first = per_asset.permute(0, 2, 1).contiguous()
        features = self.official_model.model.forward_feature(channels_first)

        expected_features = (
            batch_size * self.num_nodes,
            self.num_token_variables,
            self.d_model,
            self.output_length,
        )
        if tuple(features.shape) != expected_features:
            raise RuntimeError(
                "Unexpected official ModernTCN token feature shape. "
                f"Expected {expected_features}, received "
                f"{tuple(features.shape)}."
            )

        # [BN, M, D, P] -> [BN, D, P, M] -> [BN, D, P]
        pooled = self.variable_pool(
            features.permute(0, 2, 3, 1).contiguous()
        ).squeeze(-1)

        hidden = (
            pooled.reshape(
                batch_size,
                self.num_nodes,
                self.d_model,
                self.output_length,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return self.output_norm(hidden)

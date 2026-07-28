from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    DynamicGraphModelConfig,
    GraphActivation,
    GraphConfig,
    TemporalConfig,
)


_GRAPH_EPS: Final[float] = 1.0e-12


def _validate_btnd(
    values: Tensor,
    *,
    name: str,
    d_model: int | None = None,
) -> tuple[int, int, int, int]:
    """Validate a ``[B, T, N, D]`` tensor and return its dimensions."""
    if values.ndim != 4:
        raise ValueError(
            f"{name} must have shape [B, T, N, D]. "
            f"Received {tuple(values.shape)}."
        )

    batch_size, num_steps, num_nodes, hidden_dim = (
        int(dimension)
        for dimension in values.shape
    )

    if min(
        batch_size,
        num_steps,
        num_nodes,
        hidden_dim,
    ) <= 0:
        raise ValueError(
            f"{name} cannot contain an empty dimension."
        )

    if (
        d_model is not None
        and hidden_dim != d_model
    ):
        raise ValueError(
            f"{name} has hidden dimension {hidden_dim}; "
            f"expected {d_model}."
        )

    return (
        batch_size,
        num_steps,
        num_nodes,
        hidden_dim,
    )


class HierarchicalTokenEmbedding(nn.Module):
    """Embed Kronos ``s1``/``s2`` IDs into one node-time representation.

    Input:
        ``token_ids`` with shape ``[B, T, N, 2]``.

    Output:
        Embedded sequence with shape ``[B, T, N, D]``.

    The final token axis is fixed as:

        ``token_ids[..., 0] = s1``
        ``token_ids[..., 1] = s2``

    Separate embedding tables preserve the coarse/fine distinction.
    Optional node embeddings identify the transductive asset/node, and
    learned position embeddings identify the location inside the fixed
    context window.
    """

    def __init__(
        self,
        config: DynamicGraphModelConfig,
    ) -> None:
        super().__init__()
        config.validate()

        self.num_nodes = int(config.num_nodes)
        self.context_length = int(
            config.context_length
        )
        self.d_model = int(config.d_model)
        self.use_node_embedding = bool(
            config.use_node_embedding
        )

        self.s1_embedding = nn.Embedding(
            config.heads.s1_vocabulary_size,
            self.d_model,
        )

        self.s2_embedding = nn.Embedding(
            config.heads.s2_vocabulary_size,
            self.d_model,
        )

        self.position_embedding = nn.Embedding(
            self.context_length,
            self.d_model,
        )

        if self.use_node_embedding:
            self.node_embedding: nn.Embedding | None = (
                nn.Embedding(
                    self.num_nodes,
                    self.d_model,
                )
            )
        else:
            self.node_embedding = None

        self.output_norm = nn.LayerNorm(
            self.d_model
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.s1_embedding,
            self.s2_embedding,
            self.position_embedding,
        ):
            nn.init.normal_(
                embedding.weight,
                mean=0.0,
                std=0.02,
            )

        if self.node_embedding is not None:
            nn.init.normal_(
                self.node_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        self.output_norm.reset_parameters()

    def forward(
        self,
        token_ids: Tensor,
    ) -> Tensor:
        if token_ids.ndim != 4:
            raise ValueError(
                "token_ids must have shape [B, T, N, 2]."
            )

        batch_size, num_steps, num_nodes, streams = (
            int(dimension)
            for dimension in token_ids.shape
        )

        if streams != 2:
            raise ValueError(
                "The final token axis must contain exactly "
                "[s1, s2]."
            )

        if num_steps != self.context_length:
            raise ValueError(
                f"Expected context length {self.context_length}; "
                f"received {num_steps}."
            )

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes; "
                f"received {num_nodes}."
            )

        if token_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
        }:
            raise TypeError(
                "token_ids must use an integer dtype."
            )

        s1_ids = token_ids[..., 0].long()
        s2_ids = token_ids[..., 1].long()

        if (
            torch.any(s1_ids < 0)
            or torch.any(
                s1_ids
                >= self.s1_embedding.num_embeddings
            )
        ):
            raise ValueError(
                "s1 IDs lie outside the configured vocabulary."
            )

        if (
            torch.any(s2_ids < 0)
            or torch.any(
                s2_ids
                >= self.s2_embedding.num_embeddings
            )
        ):
            raise ValueError(
                "s2 IDs lie outside the configured vocabulary."
            )

        embedded = (
            self.s1_embedding(s1_ids)
            + self.s2_embedding(s2_ids)
        )

        position_ids = torch.arange(
            num_steps,
            device=token_ids.device,
        )

        embedded = (
            embedded
            + self.position_embedding(
                position_ids
            ).view(
                1,
                num_steps,
                1,
                self.d_model,
            )
        )

        if self.node_embedding is not None:
            node_ids = torch.arange(
                num_nodes,
                device=token_ids.device,
            )

            embedded = (
                embedded
                + self.node_embedding(
                    node_ids
                ).view(
                    1,
                    1,
                    num_nodes,
                    self.d_model,
                )
            )

        return self.output_norm(
            embedded
        )


class IdentityTemporalEncoder(nn.Module):
    """No-op temporal encoder with the shared ``[B,T,N,D]`` contract."""

    def forward(
        self,
        hidden: Tensor,
    ) -> Tensor:
        _validate_btnd(
            hidden,
            name="hidden",
        )
        return hidden


class PerNodeTransformerEncoder(nn.Module):
    """Shared causal Transformer applied independently to every node.

    The implementation follows BaseDyGraph's per-node temporal design,
    but uses the project's canonical tensor order ``[B, T, N, D]``.
    Nodes are folded into the batch dimension, so this module cannot
    exchange information across assets.
    """

    def __init__(
        self,
        *,
        d_model: int,
        config: TemporalConfig,
    ) -> None:
        super().__init__()
        config.validate(
            d_model=d_model
        )

        if config.type != "transformer":
            raise ValueError(
                "PerNodeTransformerEncoder requires "
                "temporal.type='transformer'."
            )

        self.d_model = int(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config.num_heads,
            dim_feedforward=(
                config.feedforward_multiplier
                * self.d_model
            ),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
        )

    @staticmethod
    def causal_mask(
        num_steps: int,
        *,
        device: torch.device,
    ) -> Tensor:
        """Return a boolean mask where ``True`` blocks future attention."""
        return torch.triu(
            torch.ones(
                (
                    num_steps,
                    num_steps,
                ),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def forward(
        self,
        hidden: Tensor,
    ) -> Tensor:
        (
            batch_size,
            num_steps,
            num_nodes,
            hidden_dim,
        ) = _validate_btnd(
            hidden,
            name="hidden",
            d_model=self.d_model,
        )

        node_sequences = (
            hidden
            .permute(0, 2, 1, 3)
            .reshape(
                batch_size * num_nodes,
                num_steps,
                hidden_dim,
            )
        )

        encoded = self.encoder(
            node_sequences,
            mask=self.causal_mask(
                num_steps,
                device=hidden.device,
            ),
        )

        return (
            encoded
            .reshape(
                batch_size,
                num_nodes,
                num_steps,
                hidden_dim,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )


class CausalTCNBlock(nn.Module):
    """One residual causal convolutional block.

    There is one temporal convolution per configured dilation. Hence a
    stack with kernel size ``k`` and dilations ``d_l`` has receptive
    field:

        ``1 + (k - 1) * sum(d_l)``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.left_padding = int(
            dilation * (kernel_size - 1)
        )

        self.temporal_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

        self.channel_mixing = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        sequence: Tensor,
    ) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError(
                "TCN sequence must have shape [B*N, D, T]."
            )

        residual = sequence

        convolved = self.temporal_conv(
            F.pad(
                sequence,
                (
                    self.left_padding,
                    0,
                ),
            )
        )

        convolved = self.channel_mixing(
            convolved
        )

        return (
            self.output_norm(
                (
                    residual
                    + convolved
                ).transpose(1, 2)
            )
            .transpose(1, 2)
            .contiguous()
        )


class PerNodeTCNEncoder(nn.Module):
    """Shared causal TCN applied independently to every node."""

    def __init__(
        self,
        *,
        d_model: int,
        config: TemporalConfig,
    ) -> None:
        super().__init__()
        config.validate(
            d_model=d_model
        )

        if config.type != "tcn":
            raise ValueError(
                "PerNodeTCNEncoder requires temporal.type='tcn'."
            )

        self.d_model = int(d_model)
        self.receptive_field = int(
            config.tcn_receptive_field
        )

        self.blocks = nn.ModuleList(
            [
                CausalTCNBlock(
                    d_model=self.d_model,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
                for dilation in config.dilations
            ]
        )

    def forward(
        self,
        hidden: Tensor,
    ) -> Tensor:
        (
            batch_size,
            num_steps,
            num_nodes,
            hidden_dim,
        ) = _validate_btnd(
            hidden,
            name="hidden",
            d_model=self.d_model,
        )

        node_sequences = (
            hidden
            .permute(0, 2, 3, 1)
            .reshape(
                batch_size * num_nodes,
                hidden_dim,
                num_steps,
            )
        )

        for block in self.blocks:
            node_sequences = block(
                node_sequences
            )

        return (
            node_sequences
            .reshape(
                batch_size,
                num_nodes,
                hidden_dim,
                num_steps,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )


def build_temporal_encoder(
    *,
    d_model: int,
    config: TemporalConfig,
) -> nn.Module:
    """Build one temporal module behind the shared interface."""
    if config.type == "identity":
        return IdentityTemporalEncoder()

    if config.type == "transformer":
        return PerNodeTransformerEncoder(
            d_model=d_model,
            config=config,
        )

    if config.type == "tcn":
        return PerNodeTCNEncoder(
            d_model=d_model,
            config=config,
        )

    raise ValueError(
        f"Unsupported temporal type {config.type!r}."
    )


def _sparsemax(
    logits: Tensor,
    *,
    dim: int = -1,
) -> Tensor:
    """Project logits onto the probability simplex with exact zeros."""
    shifted = (
        logits
        - logits.max(
            dim=dim,
            keepdim=True,
        ).values
    )

    sorted_logits = torch.sort(
        shifted,
        dim=dim,
        descending=True,
    ).values

    rank = torch.arange(
        1,
        logits.size(dim) + 1,
        device=logits.device,
        dtype=logits.dtype,
    )

    view_shape = [1] * logits.ndim
    view_shape[dim] = -1
    rank = rank.view(view_shape)

    cumulative = sorted_logits.cumsum(
        dim=dim
    ) - 1.0

    support = (
        sorted_logits
        - cumulative / rank
    ) > 0

    support_size = support.sum(
        dim=dim,
        keepdim=True,
    ).clamp_min(1)

    threshold = cumulative.gather(
        dim,
        support_size - 1,
    ) / support_size.to(logits.dtype)

    return torch.clamp(
        shifted - threshold,
        min=0.0,
    )


def _entmax15(
    logits: Tensor,
    *,
    dim: int = -1,
    num_iterations: int = 30,
) -> Tensor:
    """Compute 1.5-entmax using the BaseDyGraph bisection scheme."""
    shifted = (
        logits
        - logits.max(
            dim=dim,
            keepdim=True,
        ).values
    ) / 2.0

    lower = (
        shifted.max(
            dim=dim,
            keepdim=True,
        ).values
        - 1.0
    )

    upper = (
        shifted.max(
            dim=dim,
            keepdim=True,
        ).values
        - (
            1.0 / logits.size(dim)
        ) ** 0.5
    )

    for _ in range(num_iterations):
        threshold = (
            lower + upper
        ) / 2.0

        probabilities = torch.clamp(
            shifted - threshold,
            min=0.0,
        ).square()

        mass = probabilities.sum(
            dim=dim,
            keepdim=True,
        )

        lower = torch.where(
            mass < 1.0,
            threshold,
            lower,
        )

        upper = torch.where(
            mass >= 1.0,
            threshold,
            upper,
        )

    probabilities = torch.clamp(
        shifted - (
            lower + upper
        ) / 2.0,
        min=0.0,
    ).square()

    return probabilities / probabilities.sum(
        dim=dim,
        keepdim=True,
    ).clamp_min(_GRAPH_EPS)


class GraphNormalizer(nn.Module):
    """Convert window-level edge logits into row-stochastic graphs.

    Input and output shape:

        ``[B, G, N, N]``

    with orientation:

        ``A[target, source]``.

    When self-loops are disabled, the diagonal is masked before the
    activation and forced to exactly zero after the activation.
    """

    def __init__(
        self,
        config: GraphConfig,
        *,
        gate_temperature: float = 0.5,
    ) -> None:
        super().__init__()

        if config.activation not in {
            "softmax",
            "sparsemax",
            "entmax15",
            "gated",
        }:
            raise ValueError(
                f"Unsupported graph activation "
                f"{config.activation!r}."
            )

        if gate_temperature <= 0:
            raise ValueError(
                "gate_temperature must be positive."
            )

        self.activation: GraphActivation = (
            config.activation
        )
        self.num_heads = int(
            config.num_heads
        )
        self.add_self_loops = bool(
            config.add_self_loops
        )
        self.gate_temperature = float(
            gate_temperature
        )

        if self.activation == "gated":
            self.gate_threshold: nn.Parameter | None = (
                nn.Parameter(
                    torch.zeros(
                        self.num_heads
                    )
                )
            )
        else:
            self.register_parameter(
                "gate_threshold",
                None,
            )

    @staticmethod
    def _mask_value(
        dtype: torch.dtype,
    ) -> float:
        if dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            return -1.0e4
        return -1.0e9

    def forward(
        self,
        logits: Tensor,
    ) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(
                "Graph logits must have shape [B, G, N, N]."
            )

        (
            _,
            num_heads,
            num_targets,
            num_sources,
        ) = logits.shape

        if num_heads != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} graph heads; "
                f"received {num_heads}."
            )

        if num_targets != num_sources:
            raise ValueError(
                "Graph logits must be square in the node axes."
            )

        if (
            not self.add_self_loops
            and num_targets < 2
        ):
            raise ValueError(
                "A zero-diagonal graph requires at least two nodes."
            )

        diagonal_mask = torch.eye(
            num_targets,
            dtype=torch.bool,
            device=logits.device,
        ).view(
            1,
            1,
            num_targets,
            num_targets,
        )

        prepared = logits

        if not self.add_self_loops:
            prepared = logits.masked_fill(
                diagonal_mask,
                self._mask_value(
                    logits.dtype
                ),
            )

        if self.activation == "softmax":
            adjacency = torch.softmax(
                prepared,
                dim=-1,
            )

        elif self.activation == "sparsemax":
            adjacency = _sparsemax(
                prepared,
                dim=-1,
            )

        elif self.activation == "entmax15":
            adjacency = _entmax15(
                prepared,
                dim=-1,
            )

        elif self.activation == "gated":
            if self.gate_threshold is None:
                raise RuntimeError(
                    "Gated normalisation has no threshold "
                    "parameter."
                )

            threshold = self.gate_threshold.view(
                1,
                self.num_heads,
                1,
                1,
            ).to(
                device=logits.device,
                dtype=logits.dtype,
            )

            adjacency = torch.sigmoid(
                (
                    prepared - threshold
                )
                / self.gate_temperature
            )

        else:
            raise RuntimeError(
                "Unreachable graph activation."
            )

        if not self.add_self_loops:
            adjacency = adjacency.masked_fill(
                diagonal_mask,
                0.0,
            )

        row_mass = adjacency.sum(
            dim=-1,
            keepdim=True,
        )

        if torch.any(row_mass <= 0):
            raise RuntimeError(
                "Graph normalisation produced an empty row."
            )

        adjacency = (
            adjacency
            / row_mass.clamp_min(
                _GRAPH_EPS
            )
        )

        return adjacency


def aggregate_graph_values(
    adjacency: Tensor,
    values: Tensor,
) -> Tensor:
    """Aggregate source-node values using ``A[target, source]``.

    Args:
        adjacency:
            ``[B, G, N, N]``.

        values:
            ``[B, T, G, N, D_head]``.

    Returns:
        Aggregated values with shape ``[B, T, G, N, D_head]``.
    """
    if adjacency.ndim != 4:
        raise ValueError(
            "adjacency must have shape [B, G, N, N]."
        )

    if values.ndim != 5:
        raise ValueError(
            "values must have shape [B, T, G, N, D_head]."
        )

    (
        batch_size,
        num_heads,
        num_targets,
        num_sources,
    ) = adjacency.shape

    (
        value_batch,
        _,
        value_heads,
        value_nodes,
        _,
    ) = values.shape

    if batch_size != value_batch:
        raise ValueError(
            "Adjacency and value batch dimensions differ."
        )

    if num_heads != value_heads:
        raise ValueError(
            "Adjacency and value graph-head dimensions differ."
        )

    if (
        num_targets != num_sources
        or num_sources != value_nodes
    ):
        raise ValueError(
            "Adjacency and value node dimensions are not aligned."
        )

    return torch.einsum(
        "bgij,btgjd->btgid",
        adjacency,
        values,
    )


class SpatialMessagePassingLayer(nn.Module):
    """One BaseDyGraph-style graph-weighted spatial residual layer."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        feedforward_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by graph num_heads."
            )

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = (
            self.d_model
            // self.num_heads
        )

        self.value_projection = nn.Linear(
            self.d_model,
            self.d_model,
        )

        self.output_projection = nn.Linear(
            self.d_model,
            self.d_model,
        )

        self.message_dropout = nn.Dropout(
            dropout
        )

        self.mix_norm = nn.LayerNorm(
            self.d_model
        )

        self.feedforward_norm = nn.LayerNorm(
            self.d_model
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                self.d_model,
                feedforward_multiplier
                * self.d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_multiplier
                * self.d_model,
                self.d_model,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        (
            batch_size,
            num_steps,
            num_nodes,
            hidden_dim,
        ) = _validate_btnd(
            hidden,
            name="hidden",
            d_model=self.d_model,
        )

        expected_graph = (
            batch_size,
            self.num_heads,
            num_nodes,
            num_nodes,
        )

        if tuple(adjacency.shape) != expected_graph:
            raise ValueError(
                f"adjacency has shape {tuple(adjacency.shape)}; "
                f"expected {expected_graph}."
            )

        values = (
            self.value_projection(
                hidden
            )
            .view(
                batch_size,
                num_steps,
                num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
        )

        messages = aggregate_graph_values(
            adjacency,
            values,
        )

        messages = (
            messages
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
            .reshape(
                batch_size,
                num_steps,
                num_nodes,
                hidden_dim,
            )
        )

        messages = self.output_projection(
            messages
        )

        mixed = self.mix_norm(
            hidden
            + self.message_dropout(
                messages
            )
        )

        return self.feedforward_norm(
            mixed
            + self.feedforward(
                mixed
            )
        )


class SpatialMessagePassing(nn.Module):
    """Stack spatial layers while reusing one context-window graph."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        num_layers: int = 1,
        feedforward_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        self.layers = nn.ModuleList(
            [
                SpatialMessagePassingLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_multiplier=(
                        feedforward_multiplier
                    ),
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            hidden = layer(
                hidden,
                adjacency,
            )

        return hidden


class IdentitySpatialModule(nn.Module):
    """No-op spatial path used by the temporal-only ablation."""

    def forward(
        self,
        hidden: Tensor,
        adjacency: Tensor | None = None,
    ) -> Tensor:
        _validate_btnd(
            hidden,
            name="hidden",
        )

        if adjacency is not None:
            raise ValueError(
                "IdentitySpatialModule should not receive an "
                "adjacency."
            )

        return hidden


def _assert_causal_and_node_independent(
    module: nn.Module,
    *,
    d_model: int,
) -> None:
    module.eval()

    torch.manual_seed(4)

    original = torch.randn(
        2,
        12,
        3,
        d_model,
    )

    changed_future = original.clone()
    changed_future[
        :,
        7:,
        :,
        :,
    ] += 10.0

    with torch.no_grad():
        original_output = module(
            original
        )
        future_output = module(
            changed_future
        )

    if not torch.allclose(
        original_output[:, :7],
        future_output[:, :7],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Temporal module leaked information from future "
            "positions."
        )

    changed_node = original.clone()
    changed_node[
        :,
        :,
        2,
        :,
    ] += 10.0

    with torch.no_grad():
        node_output = module(
            changed_node
        )

    if not torch.allclose(
        original_output[
            :,
            :,
            :2,
            :,
        ],
        node_output[
            :,
            :,
            :2,
            :,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Per-node temporal module mixed information "
            "between nodes."
        )


def _cpu_smoke_test() -> None:
    torch.manual_seed(7)

    model_config = DynamicGraphModelConfig(
        num_nodes=4,
        context_length=12,
        d_model=16,
        temporal=TemporalConfig(
            type="transformer",
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        graph=GraphConfig(
            type="dynamic",
            num_heads=2,
            hidden_dim=16,
            activation="softmax",
            add_self_loops=False,
            mtgnn_top_k=2,
        ),
    )

    token_ids = torch.randint(
        0,
        1024,
        (
            2,
            model_config.context_length,
            model_config.num_nodes,
            2,
        ),
    )

    embedding = HierarchicalTokenEmbedding(
        model_config
    )

    embedded = embedding(
        token_ids
    )

    if tuple(embedded.shape) != (
        2,
        12,
        4,
        16,
    ):
        raise AssertionError(
            "Unexpected hierarchical embedding shape."
        )

    transformer = build_temporal_encoder(
        d_model=model_config.d_model,
        config=model_config.temporal,
    )

    _assert_causal_and_node_independent(
        transformer,
        d_model=model_config.d_model,
    )

    tcn_config = TemporalConfig(
        type="tcn",
        dropout=0.0,
        kernel_size=3,
        dilations=(
            1,
            2,
            4,
        ),
    )

    tcn = build_temporal_encoder(
        d_model=model_config.d_model,
        config=tcn_config,
    )

    _assert_causal_and_node_independent(
        tcn,
        d_model=model_config.d_model,
    )

    if (
        tcn.receptive_field
        != 15
    ):
        raise AssertionError(
            "Unexpected TCN receptive field."
        )

    for activation in (
        "softmax",
        "sparsemax",
        "entmax15",
        "gated",
    ):
        graph_config = GraphConfig(
            type="dynamic",
            num_heads=2,
            hidden_dim=16,
            activation=activation,
            add_self_loops=False,
            mtgnn_top_k=2,
        )

        normalizer = GraphNormalizer(
            graph_config
        )

        adjacency = normalizer(
            torch.randn(
                2,
                2,
                4,
                4,
            )
        )

        if not torch.allclose(
            adjacency.sum(dim=-1),
            torch.ones(
                2,
                2,
                4,
            ),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise AssertionError(
                f"{activation} graph rows do not sum to one."
            )

        diagonal = torch.diagonal(
            adjacency,
            dim1=-2,
            dim2=-1,
        )

        if not torch.equal(
            diagonal,
            torch.zeros_like(
                diagonal
            ),
        ):
            raise AssertionError(
                f"{activation} graph has non-zero self loops."
            )

    # --------------------------------------------------------
    # Exact graph orientation test.
    #
    # target 0 receives source 1
    # target 1 receives source 2
    # target 2 receives source 3
    # target 3 receives source 0
    # --------------------------------------------------------

    adjacency = torch.zeros(
        1,
        1,
        4,
        4,
    )

    adjacency[
        0,
        0,
        torch.arange(4),
        torch.tensor(
            [
                1,
                2,
                3,
                0,
            ]
        ),
    ] = 1.0

    source_values = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ]
    ).view(
        1,
        1,
        1,
        4,
        1,
    )

    aggregated = aggregate_graph_values(
        adjacency,
        source_values,
    )

    expected = torch.tensor(
        [
            20.0,
            30.0,
            40.0,
            10.0,
        ]
    ).view(
        1,
        1,
        1,
        4,
        1,
    )

    if not torch.equal(
        aggregated,
        expected,
    ):
        raise AssertionError(
            "Graph orientation is not A[target, source]."
        )

    spatial = SpatialMessagePassing(
        d_model=16,
        num_heads=2,
        num_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )

    graph_logits = torch.randn(
        2,
        2,
        4,
        4,
        requires_grad=True,
    )

    graph = GraphNormalizer(
        model_config.graph
    )(
        graph_logits
    )

    temporal_hidden = transformer(
        embedded
    )

    spatial_hidden = spatial(
        temporal_hidden,
        graph,
    )

    if tuple(spatial_hidden.shape) != (
        2,
        12,
        4,
        16,
    ):
        raise AssertionError(
            "Unexpected spatial output shape."
        )

    gradient_probe = torch.randn_like(
        spatial_hidden
    )
    loss = (
        spatial_hidden
        * gradient_probe
    ).sum()
    loss.backward()

    if (
        graph_logits.grad is None
        or not torch.isfinite(
            graph_logits.grad
        ).all()
        or graph_logits.grad.norm().item()
        <= 0.0
    ):
        raise AssertionError(
            "Spatial loss did not reach graph logits."
        )

    print(
        "Dynamic-graph core modules CPU smoke test passed."
    )
    print(
        "Embedding:",
        tuple(embedded.shape),
    )
    print(
        "Transformer:",
        tuple(temporal_hidden.shape),
    )
    print(
        "TCN receptive field:",
        tcn.receptive_field,
    )
    print(
        "Window graph:",
        tuple(graph.shape),
    )
    print(
        "Spatial output:",
        tuple(spatial_hidden.shape),
    )
    print(
        "Graph-logit gradient norm:",
        f"{graph_logits.grad.norm().item():.6f}",
    )
    print(
        "Graph orientation test:",
        "A[target, source] passed",
    )


if __name__ == "__main__":
    _cpu_smoke_test()

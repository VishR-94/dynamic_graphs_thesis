from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .contracts import (
    DynamicGraphModelConfig,
    FuturePredictorConfig,
    TokenLossConfig,
)


TokenSelection = Literal[
    "argmax",
    "sample",
]


def _validate_context_memory(
    context_memory: Tensor,
    *,
    context_length: int,
    num_nodes: int,
    d_model: int,
) -> tuple[int, int, int, int]:
    """Validate graph-aware context memory ``[B, C, N, D]``."""
    if context_memory.ndim != 4:
        raise ValueError(
            "context_memory must have shape [B, C, N, D]. "
            f"Received {tuple(context_memory.shape)}."
        )

    batch_size, num_steps, observed_nodes, hidden_dim = (
        int(value)
        for value in context_memory.shape
    )

    if num_steps != context_length:
        raise ValueError(
            f"Expected context length {context_length}; "
            f"received {num_steps}."
        )

    if observed_nodes != num_nodes:
        raise ValueError(
            f"Expected {num_nodes} nodes; "
            f"received {observed_nodes}."
        )

    if hidden_dim != d_model:
        raise ValueError(
            f"Expected hidden dimension {d_model}; "
            f"received {hidden_dim}."
        )

    if not torch.isfinite(
        context_memory
    ).all():
        raise ValueError(
            "context_memory contains non-finite values."
        )

    return (
        batch_size,
        num_steps,
        observed_nodes,
        hidden_dim,
    )


def _validate_embedding(
    embedding: nn.Embedding,
    *,
    name: str,
    vocabulary_size: int,
    d_model: int,
) -> None:
    """Validate an externally shared token-embedding table."""
    if not isinstance(
        embedding,
        nn.Embedding,
    ):
        raise TypeError(
            f"{name} must be an nn.Embedding."
        )

    if embedding.num_embeddings != vocabulary_size:
        raise ValueError(
            f"{name} has vocabulary size "
            f"{embedding.num_embeddings}; expected "
            f"{vocabulary_size}."
        )

    if embedding.embedding_dim != d_model:
        raise ValueError(
            f"{name} has embedding dimension "
            f"{embedding.embedding_dim}; expected {d_model}."
        )


def _validate_target_tokens(
    values: Tensor,
    *,
    name: str,
    batch_size: int,
    prediction_length: int,
    num_nodes: int,
    vocabulary_size: int,
) -> Tensor:
    """Validate one future target stream and return ``long`` IDs."""
    values = torch.as_tensor(
        values
    )

    expected_shape = (
        batch_size,
        prediction_length,
        num_nodes,
    )

    if tuple(values.shape) != expected_shape:
        raise ValueError(
            f"{name} has shape {tuple(values.shape)}; "
            f"expected {expected_shape}."
        )

    if values.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    }:
        raise TypeError(
            f"{name} must use an integer dtype."
        )

    values = values.long()

    if (
        values.min().item() < 0
        or values.max().item()
        >= vocabulary_size
    ):
        raise ValueError(
            f"{name} contains IDs outside "
            f"[0, {vocabulary_size - 1}]."
        )

    return values


def _flatten_nodes(
    values: Tensor,
) -> Tensor:
    """Convert ``[B, T, N, D]`` to ``[B*N, T, D]``."""
    if values.ndim != 4:
        raise ValueError(
            "values must have shape [B, T, N, D]."
        )

    batch_size, num_steps, num_nodes, hidden_dim = (
        values.shape
    )

    return (
        values
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(
            batch_size * num_nodes,
            num_steps,
            hidden_dim,
        )
    )


def _restore_nodes(
    values: Tensor,
    *,
    batch_size: int,
    num_nodes: int,
) -> Tensor:
    """Convert ``[B*N, T, D]`` to ``[B, T, N, D]``."""
    if values.ndim != 3:
        raise ValueError(
            "values must have shape [B*N, T, D]."
        )

    combined_batch, num_steps, hidden_dim = (
        values.shape
    )

    if combined_batch != batch_size * num_nodes:
        raise ValueError(
            "The flattened node batch is not aligned with "
            "batch_size and num_nodes."
        )

    return (
        values
        .reshape(
            batch_size,
            num_nodes,
            num_steps,
            hidden_dim,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _causal_mask(
    length: int,
    *,
    device: torch.device,
) -> Tensor:
    """Boolean future mask for autoregressive self-attention."""
    return torch.triu(
        torch.ones(
            (
                length,
                length,
            ),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def select_token_ids(
    logits: Tensor,
    *,
    mode: TokenSelection = "argmax",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tensor:
    """Select categorical IDs from logits.

    ``argmax`` is deterministic and is the primary validation mode.
    ``sample`` supports temperature, top-k and nucleus filtering.
    """
    if logits.ndim < 2:
        raise ValueError(
            "logits must end with a vocabulary dimension."
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "logits contains non-finite values."
        )

    if mode == "argmax":
        return logits.argmax(
            dim=-1
        )

    if mode != "sample":
        raise ValueError(
            "mode must be 'argmax' or 'sample'."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    vocabulary_size = int(
        logits.shape[-1]
    )

    if top_k < 0:
        raise ValueError(
            "top_k cannot be negative."
        )

    if top_k > vocabulary_size:
        raise ValueError(
            "top_k cannot exceed the vocabulary size."
        )

    if not 0.0 < top_p <= 1.0:
        raise ValueError(
            "top_p must lie in (0, 1]."
        )

    filtered = (
        logits
        / float(temperature)
    )

    if top_k > 0:
        threshold = torch.topk(
            filtered,
            k=top_k,
            dim=-1,
        ).values[
            ...,
            -1,
            None,
        ]

        filtered = filtered.masked_fill(
            filtered < threshold,
            float("-inf"),
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = (
            torch.sort(
                filtered,
                dim=-1,
                descending=True,
            )
        )

        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = sorted_probabilities.cumsum(
            dim=-1
        )

        remove = cumulative > top_p

        # Retain the first token whose inclusion crosses top_p.
        shifted_remove = remove.clone()
        shifted_remove[
            ...,
            1:,
        ] = remove[
            ...,
            :-1,
        ]
        shifted_remove[
            ...,
            0,
        ] = False

        sorted_logits = sorted_logits.masked_fill(
            shifted_remove,
            float("-inf"),
        )

        filtered = torch.full_like(
            filtered,
            float("-inf"),
        ).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    probabilities = torch.softmax(
        filtered,
        dim=-1,
    )

    if not torch.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Token filtering produced invalid probabilities."
        )

    flat_probabilities = probabilities.reshape(
        -1,
        vocabulary_size,
    )

    sampled = torch.multinomial(
        flat_probabilities,
        num_samples=1,
        replacement=True,
    ).squeeze(-1)

    return sampled.reshape(
        logits.shape[:-1]
    )


class FutureTransformerLayer(nn.Module):
    """One future-sequence self/cross-attention layer.

    Both predictor variants share this block:

        1. self-attention over future positions;
        2. cross-attention to observed graph-aware context;
        3. position-wise feed-forward network.

    The structured-parallel variant uses no self-attention mask.
    The autoregressive variant supplies a causal mask.
    """

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
                "d_model must be divisible by num_heads."
            )

        self.self_norm = nn.LayerNorm(
            d_model
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_dropout = nn.Dropout(
            dropout
        )

        self.cross_norm = nn.LayerNorm(
            d_model
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_dropout = nn.Dropout(
            dropout
        )

        self.feedforward_norm = nn.LayerNorm(
            d_model
        )

        hidden_dim = int(
            feedforward_multiplier
            * d_model
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                d_model,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                d_model,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        future_hidden: Tensor,
        context_memory: Tensor,
        *,
        self_attention_mask: Tensor | None,
    ) -> Tensor:
        """Process flattened node sequences ``[B*N, T, D]``."""
        if future_hidden.ndim != 3:
            raise ValueError(
                "future_hidden must have shape [B*N, P, D]."
            )

        if context_memory.ndim != 3:
            raise ValueError(
                "context_memory must have shape [B*N, C, D]."
            )

        if (
            future_hidden.shape[0]
            != context_memory.shape[0]
            or future_hidden.shape[-1]
            != context_memory.shape[-1]
        ):
            raise ValueError(
                "Future and context sequences are not aligned."
            )

        normalised_future = self.self_norm(
            future_hidden
        )

        self_message, _ = self.self_attention(
            query=normalised_future,
            key=normalised_future,
            value=normalised_future,
            attn_mask=self_attention_mask,
            need_weights=False,
        )

        future_hidden = (
            future_hidden
            + self.self_dropout(
                self_message
            )
        )

        cross_query = self.cross_norm(
            future_hidden
        )

        cross_message, _ = self.cross_attention(
            query=cross_query,
            key=context_memory,
            value=context_memory,
            need_weights=False,
        )

        future_hidden = (
            future_hidden
            + self.cross_dropout(
                cross_message
            )
        )

        return (
            future_hidden
            + self.feedforward(
                self.feedforward_norm(
                    future_hidden
                )
            )
        )


class HierarchicalTokenHeads(nn.Module):
    """Shared coarse-to-fine ``s1``/``s2`` classification heads."""

    def __init__(
        self,
        *,
        d_model: int,
        s1_vocabulary_size: int,
        s2_vocabulary_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.d_model = int(
            d_model
        )
        self.s1_vocabulary_size = int(
            s1_vocabulary_size
        )
        self.s2_vocabulary_size = int(
            s2_vocabulary_size
        )

        self.s1_classifier = nn.Linear(
            self.d_model,
            self.s1_vocabulary_size,
        )

        self.s2_conditioner = nn.Sequential(
            nn.Linear(
                2 * self.d_model,
                self.d_model,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(
                self.d_model
            ),
        )

        self.s2_classifier = nn.Linear(
            self.d_model,
            self.s2_vocabulary_size,
        )

    def s1_logits(
        self,
        future_hidden: Tensor,
    ) -> Tensor:
        return self.s1_classifier(
            future_hidden
        )

    def s2_logits(
        self,
        future_hidden: Tensor,
        s1_ids: Tensor,
        *,
        s1_embedding: nn.Embedding,
    ) -> Tensor:
        _validate_embedding(
            s1_embedding,
            name="s1_embedding",
            vocabulary_size=(
                self.s1_vocabulary_size
            ),
            d_model=self.d_model,
        )

        if tuple(s1_ids.shape) != tuple(
            future_hidden.shape[:-1]
        ):
            raise ValueError(
                "s1_ids must align with future_hidden."
            )

        if (
            s1_ids.min().item() < 0
            or s1_ids.max().item()
            >= self.s1_vocabulary_size
        ):
            raise ValueError(
                "s1_ids lies outside the configured vocabulary."
            )

        coarse_embedding = s1_embedding(
            s1_ids.long()
        )

        conditioned = self.s2_conditioner(
            torch.cat(
                (
                    future_hidden,
                    coarse_embedding,
                ),
                dim=-1,
            )
        )

        return self.s2_classifier(
            conditioned
        )


@dataclass
class FutureTokenPrediction:
    """Output shared by supervised and generated future paths.

    In ``coarse_only`` mode, ``s2_logits`` and ``selected_s2`` are
    intentionally ``None``. The observed context still contains both
    frozen Kronos token streams.
    """

    future_hidden: Tensor
    s1_logits: Tensor
    s2_logits: Tensor | None
    selected_s1: Tensor
    selected_s2: Tensor | None

    def validate(
        self,
        *,
        batch_size: int,
        prediction_length: int,
        num_nodes: int,
        d_model: int,
        s1_vocabulary_size: int,
        s2_vocabulary_size: int,
        predict_s2: bool,
    ) -> None:
        expected_hidden = (
            batch_size,
            prediction_length,
            num_nodes,
            d_model,
        )
        expected_s1 = (
            batch_size,
            prediction_length,
            num_nodes,
            s1_vocabulary_size,
        )
        expected_s2 = (
            batch_size,
            prediction_length,
            num_nodes,
            s2_vocabulary_size,
        )
        expected_ids = (
            batch_size,
            prediction_length,
            num_nodes,
        )

        if tuple(self.future_hidden.shape) != expected_hidden:
            raise ValueError(
                "future_hidden has an unexpected shape."
            )

        if tuple(self.s1_logits.shape) != expected_s1:
            raise ValueError(
                "s1_logits has an unexpected shape."
            )

        if tuple(self.selected_s1.shape) != expected_ids:
            raise ValueError(
                "selected_s1 has an unexpected shape."
            )

        if predict_s2:
            if self.s2_logits is None or self.selected_s2 is None:
                raise ValueError(
                    "Full-token mode requires s2 logits and IDs."
                )

            if tuple(self.s2_logits.shape) != expected_s2:
                raise ValueError(
                    "s2_logits has an unexpected shape."
                )

            if tuple(self.selected_s2.shape) != expected_ids:
                raise ValueError(
                    "selected_s2 has an unexpected shape."
                )
        elif self.s2_logits is not None or self.selected_s2 is not None:
            raise ValueError(
                "Coarse-only mode must not return s2 logits or IDs."
            )

        for name, values in (
            ("future_hidden", self.future_hidden),
            ("s1_logits", self.s1_logits),
        ):
            if not torch.isfinite(values).all():
                raise ValueError(
                    f"{name} contains non-finite values."
                )

        if (
            self.s2_logits is not None
            and not torch.isfinite(self.s2_logits).all()
        ):
            raise ValueError(
                "s2_logits contains non-finite values."
            )

        if (
            self.selected_s1.min().item() < 0
            or self.selected_s1.max().item()
            >= s1_vocabulary_size
        ):
            raise ValueError(
                "selected_s1 lies outside its vocabulary."
            )

        if self.selected_s2 is not None and (
            self.selected_s2.min().item() < 0
            or self.selected_s2.max().item()
            >= s2_vocabulary_size
        ):
            raise ValueError(
                "selected_s2 lies outside its vocabulary."
            )


class FutureTokenPredictorBase(
    nn.Module,
    ABC,
):
    """Common interface for both future-token prediction styles."""

    def __init__(
        self,
        config: DynamicGraphModelConfig,
    ) -> None:
        super().__init__()
        config.validate()

        self.context_length = int(
            config.context_length
        )
        self.prediction_length = int(
            config.prediction_length
        )
        self.num_nodes = int(
            config.num_nodes
        )
        self.d_model = int(
            config.d_model
        )

        self.s1_vocabulary_size = int(
            config.heads.s1_vocabulary_size
        )
        self.s2_vocabulary_size = int(
            config.heads.s2_vocabulary_size
        )
        self.s2_conditioning = (
            config.heads.resolved_s2_conditioning
        )
        self.predict_s2 = bool(
            config.heads.predicts_s2
        )

        predictor_config = (
            config.future_predictor
        )

        self.input_norm = nn.LayerNorm(
            self.d_model
        )

        self.layers = nn.ModuleList(
            [
                FutureTransformerLayer(
                    d_model=self.d_model,
                    num_heads=(
                        predictor_config.num_heads
                    ),
                    feedforward_multiplier=(
                        predictor_config
                        .feedforward_multiplier
                    ),
                    dropout=(
                        predictor_config.dropout
                    ),
                )
                for _ in range(
                    predictor_config.num_layers
                )
            ]
        )

        self.token_heads = HierarchicalTokenHeads(
            d_model=self.d_model,
            s1_vocabulary_size=(
                self.s1_vocabulary_size
            ),
            s2_vocabulary_size=(
                self.s2_vocabulary_size
            ),
            dropout=predictor_config.dropout,
        )

    def _validate_common_inputs(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
    ) -> tuple[int, Tensor, Tensor]:
        (
            batch_size,
            _,
            _,
            _,
        ) = _validate_context_memory(
            context_memory,
            context_length=self.context_length,
            num_nodes=self.num_nodes,
            d_model=self.d_model,
        )

        _validate_embedding(
            s1_embedding,
            name="s1_embedding",
            vocabulary_size=(
                self.s1_vocabulary_size
            ),
            d_model=self.d_model,
        )

        _validate_embedding(
            s2_embedding,
            name="s2_embedding",
            vocabulary_size=(
                self.s2_vocabulary_size
            ),
            d_model=self.d_model,
        )

        memory = _flatten_nodes(
            context_memory
        )

        context_summary = (
            context_memory[
                :,
                -1,
                :,
                :,
            ]
            .reshape(
                batch_size
                * self.num_nodes,
                self.d_model,
            )
        )

        return (
            batch_size,
            memory,
            context_summary,
        )

    def _apply_layers(
        self,
        future_inputs: Tensor,
        context_memory: Tensor,
        *,
        causal: bool,
    ) -> Tensor:
        mask = (
            _causal_mask(
                future_inputs.shape[1],
                device=future_inputs.device,
            )
            if causal
            else None
        )

        hidden = self.input_norm(
            future_inputs
        )

        for layer in self.layers:
            hidden = layer(
                hidden,
                context_memory,
                self_attention_mask=mask,
            )

        return hidden

    def _classify(
        self,
        future_hidden: Tensor,
        *,
        s1_embedding: nn.Embedding,
        target_s1: Tensor | None,
        token_selection: TokenSelection,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> FutureTokenPrediction:
        """Classify one complete future hidden path.

        The coarse stream is always predicted. In full-token mode, the
        fine stream is conditioned according to ``heads.s2_conditioning``.
        In coarse-only mode, the fine classifier is not evaluated.
        """
        batch_size, prediction_length, num_nodes, _ = (
            future_hidden.shape
        )

        s1_logits = self.token_heads.s1_logits(
            future_hidden
        )
        selected_s1 = select_token_ids(
            s1_logits,
            mode=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if not self.predict_s2:
            return FutureTokenPrediction(
                future_hidden=future_hidden,
                s1_logits=s1_logits,
                s2_logits=None,
                selected_s1=selected_s1,
                selected_s2=None,
            )

        if (
            self.s2_conditioning == "true_s1"
            and target_s1 is not None
        ):
            s1_for_fine = _validate_target_tokens(
                target_s1,
                name="target_s1",
                batch_size=int(batch_size),
                prediction_length=int(prediction_length),
                num_nodes=int(num_nodes),
                vocabulary_size=self.s1_vocabulary_size,
            )
        elif self.s2_conditioning in {
            "true_s1",
            "predicted_s1",
        }:
            s1_for_fine = selected_s1.detach()
        else:
            raise RuntimeError(
                "Unexpected s2 conditioning policy "
                f"{self.s2_conditioning!r}."
            )

        s2_logits = self.token_heads.s2_logits(
            future_hidden,
            s1_for_fine,
            s1_embedding=s1_embedding,
        )
        selected_s2 = select_token_ids(
            s2_logits,
            mode=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        return FutureTokenPrediction(
            future_hidden=future_hidden,
            s1_logits=s1_logits,
            s2_logits=s2_logits,
            selected_s1=selected_s1,
            selected_s2=selected_s2,
        )

    @abstractmethod
    def forward(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        target_s1: Tensor | None = None,
        target_s2: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        """Teacher-forced or parallel training/evaluation path."""

    @abstractmethod
    def generate(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        """Generate a complete future token path."""


class StructuredParallelFuturePredictor(
    FutureTokenPredictorBase,
):
    """Jointly model all future positions without target-token inputs."""

    def __init__(
        self,
        config: DynamicGraphModelConfig,
    ) -> None:
        super().__init__(
            config
        )

        self.future_position_embedding = nn.Embedding(
            self.prediction_length,
            self.d_model,
        )

        nn.init.normal_(
            self.future_position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def _future_hidden(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
    ) -> tuple[int, Tensor]:
        (
            batch_size,
            memory,
            context_summary,
        ) = self._validate_common_inputs(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )

        position_ids = torch.arange(
            self.prediction_length,
            device=context_memory.device,
        )

        future_inputs = (
            context_summary[:, None, :]
            + self.future_position_embedding(
                position_ids
            )[None, :, :]
        )

        future_hidden_flat = self._apply_layers(
            future_inputs,
            memory,
            causal=False,
        )

        future_hidden = _restore_nodes(
            future_hidden_flat,
            batch_size=batch_size,
            num_nodes=self.num_nodes,
        )

        return (
            batch_size,
            future_hidden,
        )

    def forward(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        target_s1: Tensor | None = None,
        target_s2: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        del target_s2

        (
            batch_size,
            future_hidden,
        ) = self._future_hidden(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )

        prediction = self._classify(
            future_hidden,
            s1_embedding=s1_embedding,
            target_s1=target_s1,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        prediction.validate(
            batch_size=batch_size,
            prediction_length=(
                self.prediction_length
            ),
            num_nodes=self.num_nodes,
            d_model=self.d_model,
            s1_vocabulary_size=(
                self.s1_vocabulary_size
            ),
            s2_vocabulary_size=(
                self.s2_vocabulary_size
            ),
            predict_s2=self.predict_s2,
        )

        return prediction

    def generate(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        return self.forward(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
            target_s1=None,
            target_s2=None,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


class AutoregressiveFuturePredictor(
    FutureTokenPredictorBase,
):
    """Teacher-forced training and sequential future-token generation."""

    def __init__(
        self,
        config: DynamicGraphModelConfig,
    ) -> None:
        super().__init__(
            config
        )

        self.future_position_embedding = nn.Embedding(
            self.prediction_length,
            self.d_model,
        )

        self.start_embedding = nn.Parameter(
            torch.empty(
                self.d_model
            )
        )

        nn.init.normal_(
            self.future_position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.start_embedding,
            mean=0.0,
            std=0.02,
        )

    def _teacher_forced_inputs(
        self,
        *,
        batch_size: int,
        target_s1: Tensor,
        target_s2: Tensor | None,
        context_summary: Tensor,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        device: torch.device,
    ) -> Tensor:
        target_s1 = _validate_target_tokens(
            target_s1,
            name="target_s1",
            batch_size=batch_size,
            prediction_length=self.prediction_length,
            num_nodes=self.num_nodes,
            vocabulary_size=self.s1_vocabulary_size,
        )

        previous_embedding = s1_embedding(
            target_s1[:, :-1, :]
        )

        if self.predict_s2:
            if target_s2 is None:
                raise ValueError(
                    "Full-token autoregressive training requires "
                    "target_s2."
                )

            target_s2 = _validate_target_tokens(
                target_s2,
                name="target_s2",
                batch_size=batch_size,
                prediction_length=self.prediction_length,
                num_nodes=self.num_nodes,
                vocabulary_size=self.s2_vocabulary_size,
            )
            previous_embedding = (
                previous_embedding
                + s2_embedding(target_s2[:, :-1, :])
            )

        shifted = torch.empty(
            (
                batch_size,
                self.prediction_length,
                self.num_nodes,
                self.d_model,
            ),
            dtype=context_summary.dtype,
            device=device,
        )
        shifted[:, 0, :, :] = self.start_embedding.view(
            1,
            1,
            self.d_model,
        )
        shifted[:, 1:, :, :] = previous_embedding

        position_ids = torch.arange(
            self.prediction_length,
            device=device,
        )
        position_embedding = self.future_position_embedding(
            position_ids
        ).view(
            1,
            self.prediction_length,
            1,
            self.d_model,
        )
        summary = context_summary.reshape(
            batch_size,
            self.num_nodes,
            self.d_model,
        ).unsqueeze(1)

        return shifted + position_embedding + summary

    def forward(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        target_s1: Tensor | None = None,
        target_s2: Tensor | None = None,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        if target_s1 is None:
            raise ValueError(
                "Autoregressive supervised forward requires target_s1. "
                "Use generate() for free-running inference."
            )

        if self.predict_s2 and target_s2 is None:
            raise ValueError(
                "Full-token autoregressive supervised forward requires "
                "target_s2."
            )

        (
            batch_size,
            memory,
            context_summary,
        ) = self._validate_common_inputs(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )

        future_inputs = self._teacher_forced_inputs(
            batch_size=batch_size,
            target_s1=target_s1,
            target_s2=target_s2,
            context_summary=context_summary,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
            device=context_memory.device,
        )

        future_inputs_flat = _flatten_nodes(
            future_inputs
        )

        future_hidden_flat = self._apply_layers(
            future_inputs_flat,
            memory,
            causal=True,
        )

        future_hidden = _restore_nodes(
            future_hidden_flat,
            batch_size=batch_size,
            num_nodes=self.num_nodes,
        )

        prediction = self._classify(
            future_hidden,
            s1_embedding=s1_embedding,
            target_s1=target_s1,
            token_selection=token_selection,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        prediction.validate(
            batch_size=batch_size,
            prediction_length=(
                self.prediction_length
            ),
            num_nodes=self.num_nodes,
            d_model=self.d_model,
            s1_vocabulary_size=(
                self.s1_vocabulary_size
            ),
            s2_vocabulary_size=(
                self.s2_vocabulary_size
            ),
            predict_s2=self.predict_s2,
        )

        return prediction

    def generate(
        self,
        context_memory: Tensor,
        *,
        s1_embedding: nn.Embedding,
        s2_embedding: nn.Embedding,
        token_selection: TokenSelection = "argmax",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> FutureTokenPrediction:
        (
            batch_size,
            memory,
            context_summary,
        ) = self._validate_common_inputs(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )

        selected_s1_steps: list[Tensor] = []
        selected_s2_steps: list[Tensor] = []
        s1_logit_steps: list[Tensor] = []
        s2_logit_steps: list[Tensor] = []
        hidden_steps: list[Tensor] = []

        for step in range(
            self.prediction_length
        ):
            prefix_length = step + 1

            prefix_inputs = torch.empty(
                (
                    batch_size,
                    prefix_length,
                    self.num_nodes,
                    self.d_model,
                ),
                dtype=context_memory.dtype,
                device=context_memory.device,
            )

            prefix_inputs[
                :,
                0,
                :,
                :,
            ] = self.start_embedding.view(
                1,
                1,
                self.d_model,
            )

            if step > 0:
                previous_s1 = torch.stack(
                    selected_s1_steps,
                    dim=1,
                )

                previous_embedding = s1_embedding(
                    previous_s1
                )

                if self.predict_s2:
                    previous_s2 = torch.stack(
                        selected_s2_steps,
                        dim=1,
                    )
                    previous_embedding = (
                        previous_embedding
                        + s2_embedding(previous_s2)
                    )

                prefix_inputs[
                    :,
                    1:,
                    :,
                    :,
                ] = previous_embedding

            position_ids = torch.arange(
                prefix_length,
                device=context_memory.device,
            )

            prefix_inputs = (
                prefix_inputs
                + self.future_position_embedding(
                    position_ids
                ).view(
                    1,
                    prefix_length,
                    1,
                    self.d_model,
                )
                + context_summary.reshape(
                    batch_size,
                    self.num_nodes,
                    self.d_model,
                ).unsqueeze(1)
            )

            prefix_hidden_flat = self._apply_layers(
                _flatten_nodes(
                    prefix_inputs
                ),
                memory,
                causal=True,
            )

            current_hidden = _restore_nodes(
                prefix_hidden_flat[
                    :,
                    -1:,
                    :,
                ],
                batch_size=batch_size,
                num_nodes=self.num_nodes,
            )[
                :,
                0,
                :,
                :,
            ]

            current_s1_logits = (
                self.token_heads.s1_logits(
                    current_hidden
                )
            )

            current_s1 = select_token_ids(
                current_s1_logits,
                mode=token_selection,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            hidden_steps.append(
                current_hidden
            )
            s1_logit_steps.append(
                current_s1_logits
            )
            selected_s1_steps.append(
                current_s1
            )

            if self.predict_s2:
                current_s2_logits = self.token_heads.s2_logits(
                    current_hidden,
                    current_s1,
                    s1_embedding=s1_embedding,
                )
                current_s2 = select_token_ids(
                    current_s2_logits,
                    mode=token_selection,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                s2_logit_steps.append(
                    current_s2_logits
                )
                selected_s2_steps.append(
                    current_s2
                )

        prediction = FutureTokenPrediction(
            future_hidden=torch.stack(
                hidden_steps,
                dim=1,
            ),
            s1_logits=torch.stack(
                s1_logit_steps,
                dim=1,
            ),
            s2_logits=(
                torch.stack(
                    s2_logit_steps,
                    dim=1,
                )
                if self.predict_s2
                else None
            ),
            selected_s1=torch.stack(
                selected_s1_steps,
                dim=1,
            ),
            selected_s2=(
                torch.stack(
                    selected_s2_steps,
                    dim=1,
                )
                if self.predict_s2
                else None
            ),
        )

        prediction.validate(
            batch_size=batch_size,
            prediction_length=(
                self.prediction_length
            ),
            num_nodes=self.num_nodes,
            d_model=self.d_model,
            s1_vocabulary_size=(
                self.s1_vocabulary_size
            ),
            s2_vocabulary_size=(
                self.s2_vocabulary_size
            ),
            predict_s2=self.predict_s2,
        )

        return prediction


def build_future_token_predictor(
    config: DynamicGraphModelConfig,
) -> FutureTokenPredictorBase:
    """Build the configured predictor behind one shared interface."""
    config.validate()

    if (
        config.future_predictor.type
        == "structured_parallel"
    ):
        return StructuredParallelFuturePredictor(
            config
        )

    if (
        config.future_predictor.type
        == "autoregressive"
    ):
        return AutoregressiveFuturePredictor(
            config
        )

    raise ValueError(
        "Unsupported future predictor type "
        f"{config.future_predictor.type!r}."
    )


def build_future_position_weights(
    loss_config: TokenLossConfig,
    *,
    prediction_length: int,
    evaluation_horizons: tuple[int, ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Construct positive future-position weights with mean one.

    ``uniform`` returns one at every future position.

    ``exponential_decay`` places the largest weight on the first future
    minute and then decays monotonically. The decaying component halves
    every ``exponential_half_life`` positions. A uniform floor keeps all
    60 decoder-support positions supervised:

        decay_t = 2 ** (-(t - 1) / half_life)
        weights_t = floor + (1 - floor) * decay_t / mean(decay)

    Because the normalised decay has mean one, the final vector also has
    mean one. This keeps the overall token-loss scale comparable with
    uniform weighting.
    """
    loss_config.validate()

    if prediction_length <= 0:
        raise ValueError(
            "prediction_length must be positive."
        )

    resolved_horizons = tuple(
        int(value)
        for value in evaluation_horizons
    )

    if (
        not resolved_horizons
        or min(resolved_horizons) < 1
        or max(resolved_horizons)
        > prediction_length
    ):
        raise ValueError(
            "evaluation_horizons must lie in the future path."
        )

    if len(set(resolved_horizons)) != len(
        resolved_horizons
    ):
        raise ValueError(
            "evaluation_horizons must be unique."
        )

    if (
        loss_config.horizon_weighting
        == "uniform"
    ):
        return torch.ones(
            prediction_length,
            device=device,
            dtype=dtype,
        )

    positions = torch.arange(
        prediction_length,
        device=device,
        dtype=dtype,
    )

    half_life = torch.tensor(
        float(
            loss_config.exponential_half_life
        ),
        device=device,
        dtype=dtype,
    )

    decay = torch.exp(
        -torch.log(
            torch.tensor(
                2.0,
                device=device,
                dtype=dtype,
            )
        )
        * positions
        / half_life
    )

    normalised_decay = decay / decay.mean()

    floor_weight = float(
        loss_config.exponential_floor_weight
    )

    weights = (
        floor_weight
        + (1.0 - floor_weight)
        * normalised_decay
    )

    # Numerical safety. Analytically this already has mean one.
    weights = weights / weights.mean()

    if (
        not torch.isfinite(weights).all()
        or not torch.all(weights > 0)
    ):
        raise RuntimeError(
            "Future-position weighting produced invalid values."
        )

    return weights


@dataclass
class FutureTokenLoss:
    """Weighted full-path token loss and per-position diagnostics."""

    total: Tensor
    s1: Tensor
    s2: Tensor
    s1_by_step: Tensor
    s2_by_step: Tensor
    weights: Tensor


def compute_future_token_loss(
    prediction: FutureTokenPrediction,
    target_s1: Tensor,
    target_s2: Tensor,
    *,
    loss_config: TokenLossConfig,
    evaluation_horizons: tuple[int, ...],
    s2_loss_weight: float,
) -> FutureTokenLoss:
    """Compute weighted coarse/full-token cross-entropy.

    When ``s2_loss_weight`` is zero, the prediction must be coarse-only
    and the fine-token cross-entropy is not evaluated.
    """
    if s2_loss_weight < 0:
        raise ValueError(
            "s2_loss_weight cannot be negative."
        )

    batch_size, prediction_length, num_nodes, s1_vocab = (
        prediction.s1_logits.shape
    )
    target_s1 = _validate_target_tokens(
        target_s1,
        name="target_s1",
        batch_size=batch_size,
        prediction_length=prediction_length,
        num_nodes=num_nodes,
        vocabulary_size=s1_vocab,
    )

    s1_elementwise = F.cross_entropy(
        prediction.s1_logits.reshape(-1, s1_vocab),
        target_s1.reshape(-1),
        reduction="none",
    ).reshape(
        batch_size,
        prediction_length,
        num_nodes,
    )
    s1_by_step = s1_elementwise.mean(dim=(0, 2))

    if s2_loss_weight == 0.0:
        if prediction.s2_logits is not None:
            raise ValueError(
                "s2_loss_weight=0 requires a coarse-only prediction."
            )
        s2_by_step = torch.zeros_like(s1_by_step)
        s2_loss = prediction.s1_logits.new_zeros(())
    else:
        if prediction.s2_logits is None:
            raise ValueError(
                "A positive s2_loss_weight requires s2_logits."
            )
        s2_vocab = int(prediction.s2_logits.shape[-1])
        target_s2 = _validate_target_tokens(
            target_s2,
            name="target_s2",
            batch_size=batch_size,
            prediction_length=prediction_length,
            num_nodes=num_nodes,
            vocabulary_size=s2_vocab,
        )
        s2_elementwise = F.cross_entropy(
            prediction.s2_logits.reshape(-1, s2_vocab),
            target_s2.reshape(-1),
            reduction="none",
        ).reshape(
            batch_size,
            prediction_length,
            num_nodes,
        )
        s2_by_step = s2_elementwise.mean(dim=(0, 2))

    weights = build_future_position_weights(
        loss_config,
        prediction_length=prediction_length,
        evaluation_horizons=evaluation_horizons,
        device=prediction.s1_logits.device,
        dtype=prediction.s1_logits.dtype,
    )
    denominator = weights.sum()
    s1_loss = (s1_by_step * weights).sum() / denominator

    if s2_loss_weight != 0.0:
        s2_loss = (s2_by_step * weights).sum() / denominator

    total = s1_loss + float(s2_loss_weight) * s2_loss

    return FutureTokenLoss(
        total=total,
        s1=s1_loss,
        s2=s2_loss,
        s1_by_step=s1_by_step,
        s2_by_step=s2_by_step,
        weights=weights,
    )


def _cpu_smoke_test() -> None:
    """Exercise both predictor variants and both loss modes."""
    from dataclasses import replace

    from .contracts import (
        ForecastHeadConfig,
        GraphConfig,
        TemporalConfig,
    )

    torch.manual_seed(7)

    batch_size = 2
    context_length = 6
    prediction_length = 8
    num_nodes = 3
    d_model = 16
    vocabulary_size = 32

    base_config = DynamicGraphModelConfig(
        num_nodes=num_nodes,
        context_length=context_length,
        d_model=d_model,
        num_st_blocks=1,
        use_node_embedding=True,
        temporal=TemporalConfig(
            type="transformer",
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        graph=GraphConfig(
            type="mtgnn_static",
            num_heads=1,
            hidden_dim=16,
            activation="softmax",
            add_self_loops=False,
            mtgnn_embedding_dim=8,
            mtgnn_top_k=2,
            mtgnn_alpha=3.0,
        ),
        heads=ForecastHeadConfig(
            prediction_length=prediction_length,
            evaluation_horizons=(
                1,
                3,
                8,
            ),
            s1_vocabulary_size=(
                vocabulary_size
            ),
            s2_vocabulary_size=(
                vocabulary_size
            ),
            s2_loss_weight=1.0,
            condition_s2_on_s1=True,
        ),
        future_predictor=FuturePredictorConfig(
            type="structured_parallel",
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        loss=TokenLossConfig(
            horizon_weighting="uniform",
            exponential_half_life=5.0,
            exponential_floor_weight=0.25,
        ),
    )

    context_memory = torch.randn(
        batch_size,
        context_length,
        num_nodes,
        d_model,
    )

    s1_embedding = nn.Embedding(
        vocabulary_size,
        d_model,
    )

    s2_embedding = nn.Embedding(
        vocabulary_size,
        d_model,
    )

    target_s1 = torch.randint(
        0,
        vocabulary_size,
        (
            batch_size,
            prediction_length,
            num_nodes,
        ),
    )

    target_s2 = torch.randint(
        0,
        vocabulary_size,
        (
            batch_size,
            prediction_length,
            num_nodes,
        ),
    )

    # --------------------------------------------------------
    # Structured-parallel predictor.
    # --------------------------------------------------------

    parallel = build_future_token_predictor(
        base_config
    )

    parallel.eval()

    parallel_prediction = parallel(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=target_s1,
        target_s2=target_s2,
    )

    parallel_generated = parallel.generate(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
    )

    # Future hidden states and s1 logits must not depend on target IDs.
    alternative_s1 = (
        target_s1 + 1
    ) % vocabulary_size

    alternative_s2 = (
        target_s2 + 1
    ) % vocabulary_size

    alternative_parallel = parallel(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=alternative_s1,
        target_s2=alternative_s2,
    )

    if not torch.equal(
        parallel_prediction.future_hidden,
        alternative_parallel.future_hidden,
    ):
        raise AssertionError(
            "Structured-parallel future hidden states depend on "
            "future target tokens."
        )

    if not torch.equal(
        parallel_prediction.s1_logits,
        alternative_parallel.s1_logits,
    ):
        raise AssertionError(
            "Structured-parallel s1 logits depend on targets."
        )

    if torch.equal(
        parallel_prediction.s2_logits,
        alternative_parallel.s2_logits,
    ):
        raise AssertionError(
            "true_s1 conditioning did not make the fine head respond "
            "to the same-position coarse target."
        )

    predicted_s1_config = replace(
        base_config,
        heads=replace(
            base_config.heads,
            s2_conditioning="predicted_s1",
            condition_s2_on_s1=None,
        ),
    )

    predicted_s1_parallel = (
        build_future_token_predictor(
            predicted_s1_config
        )
    )
    predicted_s1_parallel.eval()

    predicted_conditioning_a = (
        predicted_s1_parallel(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
            target_s1=target_s1,
            target_s2=target_s2,
        )
    )

    predicted_conditioning_b = (
        predicted_s1_parallel(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
            target_s1=alternative_s1,
            target_s2=alternative_s2,
        )
    )

    predicted_conditioning_generated = (
        predicted_s1_parallel.generate(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )
    )

    for name, first, second in (
        (
            "future_hidden",
            predicted_conditioning_a.future_hidden,
            predicted_conditioning_b.future_hidden,
        ),
        (
            "s1_logits",
            predicted_conditioning_a.s1_logits,
            predicted_conditioning_b.s1_logits,
        ),
        (
            "s2_logits",
            predicted_conditioning_a.s2_logits,
            predicted_conditioning_b.s2_logits,
        ),
        (
            "selected_s1",
            predicted_conditioning_a.selected_s1,
            predicted_conditioning_b.selected_s1,
        ),
        (
            "selected_s2",
            predicted_conditioning_a.selected_s2,
            predicted_conditioning_b.selected_s2,
        ),
    ):
        if not torch.equal(
            first,
            second,
        ):
            raise AssertionError(
                "predicted_s1 conditioning still depends on future "
                f"target tokens through {name}."
            )

    if not torch.equal(
        predicted_conditioning_a.s2_logits,
        predicted_conditioning_generated.s2_logits,
    ):
        raise AssertionError(
            "Structured-parallel predicted_s1 supervised and generated "
            "fine logits differ under deterministic selection."
        )

    # --------------------------------------------------------
    # Autoregressive predictor.
    # --------------------------------------------------------

    autoregressive_config = replace(
        base_config,
        future_predictor=replace(
            base_config.future_predictor,
            type="autoregressive",
        ),
    )

    autoregressive = build_future_token_predictor(
        autoregressive_config
    )

    autoregressive.eval()

    teacher_forced = autoregressive(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=target_s1,
        target_s2=target_s2,
    )

    generated = autoregressive.generate(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
    )

    # Alter target pair at future position 4. Because targets are
    # shifted, outputs through position 4 must remain unchanged.
    changed_s1 = target_s1.clone()
    changed_s2 = target_s2.clone()

    changed_s1[
        :,
        4,
        :,
    ] = (
        changed_s1[
            :,
            4,
            :,
        ]
        + 1
    ) % vocabulary_size

    changed_s2[
        :,
        4,
        :,
    ] = (
        changed_s2[
            :,
            4,
            :,
        ]
        + 1
    ) % vocabulary_size

    changed_teacher_forced = autoregressive(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=changed_s1,
        target_s2=changed_s2,
    )

    if not torch.allclose(
        teacher_forced.future_hidden[
            :,
            :5,
        ],
        changed_teacher_forced.future_hidden[
            :,
            :5,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Autoregressive teacher-forcing leaked a future "
            "target into an earlier position."
        )

    if torch.allclose(
        teacher_forced.future_hidden[
            :,
            5:,
        ],
        changed_teacher_forced.future_hidden[
            :,
            5:,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Autoregressive hidden states did not respond to "
            "an available previous target."
        )

    # The first free-running s1 distribution uses only the start
    # embedding and context, matching teacher-forced position 1.
    if not torch.allclose(
        teacher_forced.s1_logits[
            :,
            0,
        ],
        generated.s1_logits[
            :,
            0,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Autoregressive first-step teacher-forced and "
            "generated s1 logits differ."
        )

    autoregressive_predicted_config = replace(
        autoregressive_config,
        heads=replace(
            autoregressive_config.heads,
            s2_conditioning="predicted_s1",
            condition_s2_on_s1=None,
        ),
    )

    autoregressive_predicted = (
        build_future_token_predictor(
            autoregressive_predicted_config
        )
    )
    autoregressive_predicted.eval()

    autoregressive_predicted_forward = (
        autoregressive_predicted(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
            target_s1=target_s1,
            target_s2=target_s2,
        )
    )

    autoregressive_predicted_generated = (
        autoregressive_predicted.generate(
            context_memory,
            s1_embedding=s1_embedding,
            s2_embedding=s2_embedding,
        )
    )

    if not torch.allclose(
        autoregressive_predicted_forward.s2_logits[
            :,
            0,
        ],
        autoregressive_predicted_generated.s2_logits[
            :,
            0,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Autoregressive predicted_s1 first-step supervised and "
            "generated fine logits differ."
        )

    # --------------------------------------------------------
    # Losses and gradients.
    # --------------------------------------------------------

    uniform_loss = compute_future_token_loss(
        parallel_prediction,
        target_s1,
        target_s2,
        loss_config=base_config.loss,
        evaluation_horizons=(
            base_config
            .heads
            .evaluation_horizons
        ),
        s2_loss_weight=(
            base_config
            .heads
            .s2_loss_weight
        ),
    )

    # Test the weighted loss on the real 60-position contract rather
    # than the shortened predictor smoke-test path.
    real_prediction_length = 60
    real_evaluation_horizons = (
        1,
        5,
        15,
        30,
        60,
    )

    exponential_config = TokenLossConfig(
        horizon_weighting=(
            "exponential_decay"
        ),
        exponential_half_life=5.0,
        exponential_floor_weight=0.25,
    )

    exponential_weights = (
        build_future_position_weights(
            exponential_config,
            prediction_length=(
                real_prediction_length
            ),
            evaluation_horizons=(
                real_evaluation_horizons
            ),
        )
    )

    if not torch.allclose(
        uniform_loss.weights,
        torch.ones_like(
            uniform_loss.weights
        ),
    ):
        raise AssertionError(
            "Uniform future weights are not all one."
        )

    if not torch.all(
        exponential_weights > 0
    ):
        raise AssertionError(
            "Exponential weights must be positive."
        )

    if not torch.allclose(
        exponential_weights.mean(),
        torch.tensor(
            1.0,
            dtype=exponential_weights.dtype,
        ),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Exponential weights do not average to one."
        )

    if not torch.all(
        exponential_weights[:-1]
        > exponential_weights[1:]
    ):
        raise AssertionError(
            "Exponential weights must decrease strictly with horizon."
        )

    floor_weight = torch.tensor(
        exponential_config.exponential_floor_weight,
        dtype=exponential_weights.dtype,
    )

    if not torch.all(
        exponential_weights >= floor_weight
    ):
        raise AssertionError(
            "Exponential weights fell below the configured floor."
        )

    # With a five-position half-life, the component above the floor at
    # minute 6 must be half its minute-1 value.
    excess_at_minute_1 = (
        exponential_weights[0]
        - floor_weight
    )
    excess_at_minute_6 = (
        exponential_weights[5]
        - floor_weight
    )

    if not torch.allclose(
        excess_at_minute_6,
        0.5 * excess_at_minute_1,
        atol=1.0e-6,
        rtol=1.0e-5,
    ):
        raise AssertionError(
            "Exponential half-life is not implemented correctly."
        )


    parallel.train()
    parallel.zero_grad(
        set_to_none=True
    )

    gradient_prediction = parallel(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=target_s1,
        target_s2=target_s2,
    )

    gradient_loss = compute_future_token_loss(
        gradient_prediction,
        target_s1,
        target_s2,
        loss_config=base_config.loss,
        evaluation_horizons=(
            1,
            3,
            8,
        ),
        s2_loss_weight=1.0,
    )

    gradient_loss.total.backward()

    query_gradient = (
        parallel
        .layers[0]
        .self_attention
        .in_proj_weight
        .grad
    )

    if (
        query_gradient is None
        or not torch.isfinite(
            query_gradient
        ).all()
        or query_gradient.norm().item()
        <= 0.0
    ):
        raise AssertionError(
            "Future token loss did not reach future self-attention."
        )

    # --------------------------------------------------------
    # Coarse-only prediction and loss.
    # --------------------------------------------------------

    coarse_only_config = replace(
        base_config,
        heads=replace(
            base_config.heads,
            future_token_mode="coarse_only",
            s2_loss_weight=0.0,
        ),
    )

    coarse_only = build_future_token_predictor(
        coarse_only_config
    )
    coarse_only.train()

    coarse_prediction = coarse_only(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
        target_s1=target_s1,
        target_s2=target_s2,
    )
    coarse_generated = coarse_only.generate(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
    )

    for name, prediction in (
        ("supervised", coarse_prediction),
        ("generated", coarse_generated),
    ):
        if prediction.s2_logits is not None:
            raise AssertionError(
                f"Coarse-only {name} path returned s2 logits."
            )
        if prediction.selected_s2 is not None:
            raise AssertionError(
                f"Coarse-only {name} path returned s2 IDs."
            )

    coarse_loss = compute_future_token_loss(
        coarse_prediction,
        target_s1,
        target_s2,
        loss_config=coarse_only_config.loss,
        evaluation_horizons=(
            coarse_only_config
            .heads
            .evaluation_horizons
        ),
        s2_loss_weight=0.0,
    )

    if not torch.equal(
        coarse_loss.total,
        coarse_loss.s1,
    ):
        raise AssertionError(
            "Coarse-only total loss does not equal s1 loss."
        )

    if coarse_loss.s2.item() != 0.0:
        raise AssertionError(
            "Coarse-only loss returned a non-zero s2 term."
        )

    coarse_only.zero_grad(set_to_none=True)
    coarse_loss.total.backward()

    coarse_s1_gradient = (
        coarse_only
        .token_heads
        .s1_classifier
        .weight
        .grad
    )
    coarse_s2_gradient = (
        coarse_only
        .token_heads
        .s2_classifier
        .weight
        .grad
    )

    if (
        coarse_s1_gradient is None
        or coarse_s1_gradient.norm().item() <= 0.0
    ):
        raise AssertionError(
            "Coarse-only loss did not reach the s1 classifier."
        )

    if coarse_s2_gradient is not None:
        raise AssertionError(
            "Coarse-only loss unexpectedly reached the s2 classifier."
        )

    # --------------------------------------------------------
    # Node-independence check inside the future predictor.
    # --------------------------------------------------------

    parallel.eval()

    changed_context = context_memory.clone()
    changed_context[
        :,
        :,
        2,
        :,
    ] += 10.0

    original_node_output = parallel.generate(
        context_memory,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
    )

    changed_node_output = parallel.generate(
        changed_context,
        s1_embedding=s1_embedding,
        s2_embedding=s2_embedding,
    )

    if not torch.allclose(
        original_node_output.future_hidden[
            :,
            :,
            :2,
            :,
        ],
        changed_node_output.future_hidden[
            :,
            :,
            :2,
            :,
        ],
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise AssertionError(
            "Future predictor mixed nodes internally."
        )

    print(
        "Future-token predictor CPU smoke test passed."
    )

    print(
        "Structured-parallel logits:",
        tuple(
            parallel_prediction
            .s1_logits
            .shape
        ),
    )

    print(
        "Structured-parallel generated IDs:",
        tuple(
            parallel_generated
            .selected_s1
            .shape
        ),
    )

    print(
        "Autoregressive teacher-forced logits:",
        tuple(
            teacher_forced
            .s1_logits
            .shape
        ),
    )

    print(
        "Autoregressive generated IDs:",
        tuple(
            generated
            .selected_s1
            .shape
        ),
    )

    print(
        "Uniform weights:",
        uniform_loss.weights.tolist(),
    )

    print(
        "Exponential horizon weights:",
        {
            horizon: round(
                float(
                    exponential_weights[
                        horizon - 1
                    ]
                ),
                4,
            )
            for horizon in (
                real_evaluation_horizons
            )
        },
    )

    print(
        "Exponential weights (minute:weight):"
    )

    for start in range(
        0,
        real_prediction_length,
        10,
    ):
        stop = min(
            start + 10,
            real_prediction_length,
        )

        print(
            "  "
            + "  ".join(
                f"{minute + 1}:"
                f"{float(exponential_weights[minute]):.4f}"
                for minute in range(
                    start,
                    stop,
                )
            )
        )

    print(
        "Future-attention gradient norm:",
        f"{query_gradient.norm().item():.6f}",
    )


if __name__ == "__main__":
    _cpu_smoke_test()






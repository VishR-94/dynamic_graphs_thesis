from __future__ import annotations

import math
from typing import Final, Literal

import torch
from torch import Tensor, nn

from .contracts import GraphConfig, GraphOutput
from .modules import GraphNormalizer, SpatialMessagePassing


_MASK_FLOAT32: Final[float] = -1.0e9
_MASK_LOW_PRECISION: Final[float] = -1.0e4
_GRAPH_EPS: Final[float] = 1.0e-12

EmptyCorrelationRowPolicy = Literal["error", "strongest"]


def _mask_value(dtype: torch.dtype) -> float:
    if dtype in {torch.float16, torch.bfloat16}:
        return _MASK_LOW_PRECISION
    return _MASK_FLOAT32


def _positive_floor(dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        raise TypeError("Graph values must use a floating dtype.")
    return max(_GRAPH_EPS, float(torch.finfo(dtype).tiny))


def _validate_context_hidden(
    context_hidden: Tensor,
    *,
    num_nodes: int,
    d_model: int | None = None,
) -> tuple[int, int, int, int]:
    if context_hidden.ndim != 4:
        raise ValueError(
            "context_hidden must have shape [B, T, N, D]."
        )

    batch_size, num_steps, observed_nodes, hidden_dim = (
        int(value) for value in context_hidden.shape
    )

    if batch_size <= 0 or num_steps <= 0:
        raise ValueError(
            "context_hidden must contain at least one example and time step."
        )

    if observed_nodes != int(num_nodes):
        raise ValueError(
            f"Expected {num_nodes} nodes; received {observed_nodes}."
        )

    if d_model is not None and hidden_dim != int(d_model):
        raise ValueError(
            f"Expected hidden dimension {d_model}; received {hidden_dim}."
        )

    if not torch.isfinite(context_hidden).all():
        raise ValueError(
            "context_hidden contains non-finite values."
        )

    return batch_size, num_steps, observed_nodes, hidden_dim


def _expand_singleton_graph(
    graph: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    if graph.ndim != 4 or int(graph.shape[0]) != 1:
        raise ValueError(
            "A singleton static graph must have shape [1, G, N, N]."
        )
    return graph.expand(batch_size, -1, -1, -1)


def _normalisation_logits_from_adjacency(
    adjacency: Tensor,
) -> Tensor:
    """Return finite softmax logits that recover a sparse adjacency."""
    if adjacency.ndim != 4 or int(adjacency.shape[0]) != 1:
        raise ValueError(
            "adjacency must have shape [1, G, N, N]."
        )

    positive = adjacency > 0
    logits = torch.log(
        adjacency.clamp_min(
            _positive_floor(adjacency.dtype)
        )
    )

    return torch.where(
        positive,
        logits,
        torch.full_like(
            logits,
            _mask_value(adjacency.dtype),
        ),
    )


def _prepare_fixed_adjacency(
    adjacency: Tensor,
    *,
    num_heads: int,
    num_nodes: int,
    add_self_loops: bool,
) -> Tensor:
    """Validate and row-normalise a supplied non-negative adjacency."""
    graph = torch.as_tensor(
        adjacency,
        dtype=torch.float32,
    ).detach().clone()

    if graph.ndim == 2:
        graph = graph.unsqueeze(0)

    if graph.ndim == 4:
        if int(graph.shape[0]) != 1:
            raise ValueError(
                "A four-dimensional fixed graph must have shape "
                "[1, G, N, N]."
            )
        graph = graph[0]

    if graph.ndim != 3:
        raise ValueError(
            "A fixed graph must have shape [N,N], [1,N,N], "
            "[G,N,N], or [1,G,N,N]."
        )

    if tuple(graph.shape[-2:]) != (num_nodes, num_nodes):
        raise ValueError(
            "The fixed graph node axes do not match num_nodes."
        )

    if int(graph.shape[0]) == 1 and num_heads > 1:
        graph = graph.expand(num_heads, -1, -1).clone()

    if int(graph.shape[0]) != num_heads:
        raise ValueError(
            f"Expected 1 or {num_heads} fixed graph heads; "
            f"received {int(graph.shape[0])}."
        )

    if not torch.isfinite(graph).all():
        raise ValueError(
            "The fixed graph contains non-finite values."
        )

    if torch.any(graph < 0):
        raise ValueError(
            "The fixed graph contains negative edge weights."
        )

    if not add_self_loops:
        diagonal_mask = torch.eye(
            num_nodes,
            dtype=torch.bool,
            device=graph.device,
        ).unsqueeze(0)
        graph = graph.masked_fill(diagonal_mask, 0.0)

    row_mass = graph.sum(dim=-1, keepdim=True)

    if torch.any(row_mass <= 0):
        empty_rows = torch.nonzero(
            row_mass.squeeze(-1) <= 0,
            as_tuple=False,
        )
        raise ValueError(
            "The supplied graph contains empty target rows after "
            "self-loop handling. Empty [head,target] rows: "
            f"{empty_rows.tolist()}."
        )

    return (graph / row_mass).unsqueeze(0).contiguous()


def build_absolute_correlation_adjacency(
    correlation_matrix: Tensor,
    *,
    threshold: float,
    num_heads: int,
    add_self_loops: bool,
    empty_row_policy: EmptyCorrelationRowPolicy = "error",
    symmetry_atol: float = 1.0e-5,
) -> Tensor:
    """Construct the training-only absolute-correlation graph.

    Values strictly below ``threshold`` are removed. The caller is
    responsible for fitting ``correlation_matrix`` from training data only.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(
            "The correlation threshold must lie in [0, 1]."
        )

    if empty_row_policy not in {"error", "strongest"}:
        raise ValueError(
            "empty_row_policy must be 'error' or 'strongest'."
        )

    correlation = torch.as_tensor(
        correlation_matrix,
        dtype=torch.float32,
    ).detach().clone()

    if (
        correlation.ndim != 2
        or correlation.shape[0] != correlation.shape[1]
    ):
        raise ValueError(
            "correlation_matrix must have square shape [N, N]."
        )

    num_nodes = int(correlation.shape[0])

    if num_nodes < 2:
        raise ValueError(
            "A correlation graph requires at least two nodes."
        )

    if not torch.isfinite(correlation).all():
        raise ValueError(
            "correlation_matrix contains non-finite values."
        )

    if torch.any(correlation.abs() > 1.0 + symmetry_atol):
        raise ValueError(
            "correlation_matrix contains values outside [-1, 1]."
        )

    if not torch.allclose(
        correlation,
        correlation.transpose(0, 1),
        atol=symmetry_atol,
        rtol=0.0,
    ):
        raise ValueError(
            "correlation_matrix is not symmetric within tolerance."
        )

    absolute = correlation.abs().clamp(0.0, 1.0)

    if not add_self_loops:
        absolute.fill_diagonal_(0.0)

    retained = torch.where(
        absolute >= float(threshold),
        absolute,
        torch.zeros_like(absolute),
    )

    if not add_self_loops:
        retained.fill_diagonal_(0.0)

    empty_rows = retained.sum(dim=-1) <= 0

    if torch.any(empty_rows):
        if empty_row_policy == "error":
            raise ValueError(
                "The correlation threshold produced empty target rows: "
                f"{torch.nonzero(empty_rows, as_tuple=False).flatten().tolist()}."
            )

        candidates = absolute.clone()
        if not add_self_loops:
            candidates.fill_diagonal_(-1.0)

        for target_index in torch.nonzero(
            empty_rows,
            as_tuple=False,
        ).flatten().tolist():
            source_index = int(
                candidates[target_index].argmax()
            )
            strongest = float(
                absolute[target_index, source_index]
            )
            if strongest <= 0:
                raise ValueError(
                    "Cannot backfill an empty row because all eligible "
                    "absolute correlations are zero."
                )
            retained[target_index, source_index] = strongest

    return _prepare_fixed_adjacency(
        retained,
        num_heads=num_heads,
        num_nodes=num_nodes,
        add_self_loops=add_self_loops,
    )


class NoGraphLearner(nn.Module):
    """No-graph provider behind the shared graph-learner interface."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
    ) -> None:
        super().__init__()

        if config.type != "none":
            raise ValueError(
                "NoGraphLearner requires graph.type='none'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=1,
        )

        self.num_nodes = int(num_nodes)
        self.num_heads = int(config.num_heads)

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
        )
        return GraphOutput(selected=None)


class FixedGraphLearner(nn.Module):
    """Expose a supplied fixed, non-negative, row-stochastic graph."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        adjacency: Tensor,
    ) -> None:
        super().__init__()

        if config.type != "fixed":
            raise ValueError(
                "FixedGraphLearner requires graph.type='fixed'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=1,
        )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.num_heads = int(config.num_heads)

        singleton_adjacency = _prepare_fixed_adjacency(
            adjacency,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
            add_self_loops=bool(config.add_self_loops),
        )

        self.register_buffer(
            "_singleton_adjacency",
            singleton_adjacency,
            persistent=True,
        )

    def singleton_adjacency(self) -> Tensor:
        return self._singleton_adjacency

    def singleton_logits(self) -> Tensor:
        return _normalisation_logits_from_adjacency(
            self._singleton_adjacency
        )

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
        )

        # Keep graph probabilities in float32 under AMP. Casting a
        # row-normalised adjacency to float16 before contract validation can
        # perturb row sums beyond the strict stochasticity tolerance.
        base = self.singleton_adjacency().to(
            device=context_hidden.device,
            dtype=torch.float32,
        )
        logits = _normalisation_logits_from_adjacency(base)

        output = GraphOutput(
            selected=_expand_singleton_graph(
                base,
                batch_size=batch_size,
            ),
            base=base,
            dynamic=None,
            alpha=None,
            logits=_expand_singleton_graph(
                logits,
                batch_size=batch_size,
            ),
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class AbsoluteCorrelationGraphLearner(FixedGraphLearner):
    """Absolute training-correlation graph thresholded before row normalisation."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        correlation_matrix: Tensor,
        threshold: float,
        empty_row_policy: EmptyCorrelationRowPolicy = "error",
    ) -> None:
        if correlation_matrix.ndim != 2:
            raise ValueError(
                "correlation_matrix must have shape [N, N]."
            )

        num_nodes = int(correlation_matrix.shape[0])
        adjacency = build_absolute_correlation_adjacency(
            correlation_matrix,
            threshold=threshold,
            num_heads=int(config.num_heads),
            add_self_loops=bool(config.add_self_loops),
            empty_row_policy=empty_row_policy,
        )

        super().__init__(
            config=config,
            num_nodes=num_nodes,
            adjacency=adjacency,
        )

        self.threshold = float(threshold)
        self.empty_row_policy = empty_row_policy


class FreeStaticGraphLearner(nn.Module):
    """BaseDyGraph-style directly learned global graph logits."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        initialisation_std: float = 0.02,
    ) -> None:
        super().__init__()

        if config.type != "free_static":
            raise ValueError(
                "FreeStaticGraphLearner requires "
                "graph.type='free_static'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=1,
        )

        if initialisation_std <= 0:
            raise ValueError(
                "initialisation_std must be positive."
            )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.num_heads = int(config.num_heads)

        self.logits = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.num_nodes,
                self.num_nodes,
            )
        )
        nn.init.normal_(
            self.logits,
            std=float(initialisation_std),
        )

        self.normalizer = GraphNormalizer(config)

    def singleton_logits(self) -> Tensor:
        return self.logits.unsqueeze(0)

    def singleton_adjacency(self) -> Tensor:
        return self.normalizer(
            self.singleton_logits()
        )

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
        )

        singleton_logits = self.singleton_logits()
        singleton_adjacency = self.normalizer(
            singleton_logits
        )

        output = GraphOutput(
            selected=_expand_singleton_graph(
                singleton_adjacency,
                batch_size=batch_size,
            ),
            base=singleton_adjacency,
            dynamic=None,
            alpha=None,
            logits=_expand_singleton_graph(
                singleton_logits,
                batch_size=batch_size,
            ),
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class MTGNNStaticGraphLearner(nn.Module):
    """Learn one global directed sparse graph using MTGNN's constructor.

    The implementation follows the official MTGNN graph-constructor
    parameterisation independently for every project graph head:

        M1 = tanh(alpha * Linear1(Embedding1))
        M2 = tanh(alpha * Linear2(Embedding2))
        S  = relu(tanh(alpha * (M1 M2^T - M2 M1^T)))

    Only the top-k source nodes are retained for each target node. The
    retained non-negative scores are then passed through the project's
    shared ``GraphNormalizer``. Therefore, the exact adjacency exposed in
    ``GraphOutput.selected`` is also the row-stochastic adjacency consumed
    later by ``SpatialMessagePassing``.

    Orientation:
        ``A[target, source]``.

    Input:
        ``context_hidden`` with shape ``[B, T, N, D]``. This graph is
        global/static and does not depend on the values in the tensor; the
        tensor supplies the batch size and validates the node universe.

    Output:
        ``GraphOutput`` with ``selected`` shape ``[B, G, N, N]``.

    Project adaptations relative to the original single-graph constructor:
        - one independent constructor per project graph head;
        - optional explicit diagonal masking;
        - deterministic top-k support in evaluation mode;
        - project-standard row normalisation and graph output contract.
    """

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        training_tie_break_noise: float = 0.01,
    ) -> None:
        super().__init__()

        if config.type != "mtgnn_static":
            raise ValueError(
                "MTGNNStaticGraphLearner requires "
                "graph.type='mtgnn_static'."
            )

        # d_model is irrelevant for this static learner, but GraphConfig
        # owns the common validation contract used by every graph mode.
        config.validate(
            num_nodes=num_nodes,
            d_model=1,
        )

        if training_tie_break_noise < 0:
            raise ValueError(
                "training_tie_break_noise cannot be negative."
            )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.num_heads = int(config.num_heads)
        self.embedding_dim = int(
            config.mtgnn_embedding_dim
        )
        self.top_k = int(config.mtgnn_top_k)
        self.alpha = float(config.mtgnn_alpha)
        self.add_self_loops = bool(
            config.add_self_loops
        )
        self.training_tie_break_noise = float(
            training_tie_break_noise
        )

        # The official MTGNN constructor has two node-embedding pathways
        # and two projections. The project contract requires G graph heads,
        # so each head receives an independent constructor.
        self.embedding_1 = nn.ModuleList(
            [
                nn.Embedding(
                    self.num_nodes,
                    self.embedding_dim,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.embedding_2 = nn.ModuleList(
            [
                nn.Embedding(
                    self.num_nodes,
                    self.embedding_dim,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.projection_1 = nn.ModuleList(
            [
                nn.Linear(
                    self.embedding_dim,
                    self.embedding_dim,
                )
                for _ in range(self.num_heads)
            ]
        )
        self.projection_2 = nn.ModuleList(
            [
                nn.Linear(
                    self.embedding_dim,
                    self.embedding_dim,
                )
                for _ in range(self.num_heads)
            ]
        )

        self.normalizer = GraphNormalizer(
            config
        )

        self.register_buffer(
            "node_indices",
            torch.arange(
                self.num_nodes,
                dtype=torch.long,
            ),
            persistent=False,
        )

    @staticmethod
    def _mask_value(
        dtype: torch.dtype,
    ) -> float:
        if dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            return _MASK_LOW_PRECISION
        return _MASK_FLOAT32

    @staticmethod
    def _positive_floor(
        dtype: torch.dtype,
    ) -> float:
        """Return a log-safe positive floor for the active dtype."""
        if not dtype.is_floating_point:
            raise TypeError(
                "MTGNN graph scores must use a floating dtype."
            )

        return max(
            1.0e-12,
            float(torch.finfo(dtype).tiny),
        )

    def _head_scores(
        self,
        head_index: int,
    ) -> Tensor:
        """Return dense non-negative MTGNN scores ``[N, N]``."""
        node_indices = self.node_indices

        node_vector_1 = torch.tanh(
            self.alpha
            * self.projection_1[head_index](
                self.embedding_1[head_index](
                    node_indices
                )
            )
        )
        node_vector_2 = torch.tanh(
            self.alpha
            * self.projection_2[head_index](
                self.embedding_2[head_index](
                    node_indices
                )
            )
        )

        asymmetric_scores = (
            node_vector_1
            @ node_vector_2.transpose(0, 1)
            - node_vector_2
            @ node_vector_1.transpose(0, 1)
        )

        return torch.relu(
            torch.tanh(
                self.alpha
                * asymmetric_scores
            )
        )

    def dense_scores(self) -> Tensor:
        """Return pre-sparsification scores with shape ``[G, N, N]``."""
        return torch.stack(
            [
                self._head_scores(head_index)
                for head_index in range(
                    self.num_heads
                )
            ],
            dim=0,
        )

    def _top_k_normalisation_logits(
        self,
        scores: Tensor,
    ) -> Tensor:
        """Build top-k normalisation logits with shape ``[G, N, N]``.

        The official MTGNN implementation adds a small random perturbation
        before top-k selection. Here that perturbation is used in training
        only, so validation/test graphs remain deterministic.
        """
        expected_shape = (
            self.num_heads,
            self.num_nodes,
            self.num_nodes,
        )

        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"scores has shape {tuple(scores.shape)}; "
                f"expected {expected_shape}."
            )

        if not torch.isfinite(scores).all():
            raise ValueError(
                "MTGNN scores contain non-finite values."
            )

        ranking_scores = scores

        if (
            self.training
            and self.training_tie_break_noise > 0
        ):
            ranking_scores = (
                ranking_scores
                + torch.rand_like(
                    ranking_scores
                )
                * self.training_tie_break_noise
            )

        if not self.add_self_loops:
            diagonal_mask = torch.eye(
                self.num_nodes,
                dtype=torch.bool,
                device=scores.device,
            ).unsqueeze(0)

            ranking_scores = ranking_scores.masked_fill(
                diagonal_mask,
                self._mask_value(
                    scores.dtype
                ),
            )

        selected_sources = torch.topk(
            ranking_scores,
            k=self.top_k,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices

        support = torch.zeros_like(
            scores,
            dtype=torch.bool,
        )
        support.scatter_(
            dim=-1,
            index=selected_sources,
            value=True,
        )

        # For softmax, log(score) makes the shared normaliser recover
        # score / row_sum over the retained support. This is closer to the
        # original MTGNN weighted adjacency than applying exp(score).
        if self.config.activation == "softmax":
            retained_logits = torch.log(
                scores.clamp_min(
                    self._positive_floor(
                        scores.dtype
                    )
                )
            )
        else:
            retained_logits = scores

        return torch.where(
            support,
            retained_logits,
            torch.full_like(
                retained_logits,
                self._mask_value(
                    retained_logits.dtype
                ),
            ),
        )

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        if context_hidden.ndim != 4:
            raise ValueError(
                "context_hidden must have shape [B, T, N, D]."
            )

        batch_size, _, observed_nodes, _ = (
            int(value)
            for value in context_hidden.shape
        )

        if observed_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes; "
                f"received {observed_nodes}."
            )

        if context_hidden.device != self.node_indices.device:
            raise ValueError(
                "context_hidden and the graph learner are on "
                "different devices. Move the model and input to "
                "the same device before the forward pass."
            )

        scores = self.dense_scores()
        singleton_logits = (
            self._top_k_normalisation_logits(
                scores
            )
            .unsqueeze(0)
        )
        singleton_adjacency = self.normalizer(
            singleton_logits
        )

        selected = singleton_adjacency.expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        logits = singleton_logits.expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        output = GraphOutput(
            selected=selected,
            base=singleton_adjacency,
            dynamic=None,
            alpha=None,
            logits=logits,
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )

        return output

class BaseDyGraphDynamicGraphLearner(nn.Module):
    """BaseDyGraph Q/K graph adapted to one graph per context window.

    BaseDyGraph computes a graph at every time step. The dissertation
    contract uses the final causal temporal state and produces one graph
    for the complete context window:

        logits = Q K^T / sqrt(head_dim)

    The orientation is A[target, source].
    """

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        d_model: int,
    ) -> None:
        super().__init__()

        if config.type != "dynamic":
            raise ValueError(
                "BaseDyGraphDynamicGraphLearner requires "
                "graph.type='dynamic'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=d_model,
        )

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                "graph.hidden_dim must be divisible by graph.num_heads."
            )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.num_heads = int(config.num_heads)
        self.graph_hidden_dim = int(config.hidden_dim)
        self.head_dim = (
            self.graph_hidden_dim
            // self.num_heads
        )

        self.q_proj = nn.Linear(
            self.d_model,
            self.graph_hidden_dim,
        )
        self.k_proj = nn.Linear(
            self.d_model,
            self.graph_hidden_dim,
        )
        self.normalizer = GraphNormalizer(config)

    def dynamic_logits(
        self,
        context_hidden: Tensor,
    ) -> Tensor:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
            d_model=self.d_model,
        )

        origin_hidden = context_hidden[:, -1]

        queries = (
            self.q_proj(origin_hidden)
            .view(
                batch_size,
                self.num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(origin_hidden)
            .view(
                batch_size,
                self.num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )

        return (
            queries
            @ keys.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        batch_size = int(
            context_hidden.shape[0]
        )
        logits = self.dynamic_logits(
            context_hidden
        )
        adjacency = self.normalizer(logits)

        output = GraphOutput(
            selected=adjacency,
            base=None,
            dynamic=adjacency,
            alpha=None,
            logits=logits,
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class BaseDyGraphDynamicBaseGraphLearner(nn.Module):
    """BaseDyGraph dynamic-base graph with its exact convex formulation.

    The selected graph is:

        A_base = normalise(base_logits)
        A_full = normalise(base_logits + dynamic_logits)
        A = (1 - alpha) * A_base + alpha * A_full

    Thus alpha is the weight on the full input-conditioned graph. At
    alpha=0 the graph is static; at alpha=1 it is the full base-plus-dynamic
    graph.

    If ``fixed_base_adjacency`` is omitted, ``base_logits`` is a directly
    learned BaseDyGraph-style [G,N,N] parameter. If a fixed/correlation
    base is supplied, dynamic reweighting is restricted to its retained
    support so the thresholded prior remains exact.
    """

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        d_model: int,
        fixed_base_adjacency: Tensor | None = None,
        base_initialisation_std: float = 0.02,
    ) -> None:
        super().__init__()

        if config.type != "dynamic_base":
            raise ValueError(
                "BaseDyGraphDynamicBaseGraphLearner requires "
                "graph.type='dynamic_base'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=d_model,
        )

        if config.hidden_dim % config.num_heads != 0:
            raise ValueError(
                "graph.hidden_dim must be divisible by graph.num_heads."
            )

        if base_initialisation_std <= 0:
            raise ValueError(
                "base_initialisation_std must be positive."
            )

        if (
            fixed_base_adjacency is not None
            and config.activation != "softmax"
        ):
            raise ValueError(
                "A fixed/correlation base currently requires "
                "graph.activation='softmax'."
            )

        if (
            fixed_base_adjacency is None
            and config.base_graph_type != "free_static"
        ):
            raise ValueError(
                "A learned BaseDyGraph dynamic-base graph requires "
                "graph.base_graph_type='free_static'."
            )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.num_heads = int(config.num_heads)
        self.graph_hidden_dim = int(config.hidden_dim)
        self.head_dim = (
            self.graph_hidden_dim
            // self.num_heads
        )
        self.uses_fixed_base = (
            fixed_base_adjacency is not None
        )

        self.q_proj = nn.Linear(
            self.d_model,
            self.graph_hidden_dim,
        )
        self.k_proj = nn.Linear(
            self.d_model,
            self.graph_hidden_dim,
        )

        if fixed_base_adjacency is None:
            base_graph = torch.empty(
                self.num_heads,
                self.num_nodes,
                self.num_nodes,
            )
            nn.init.normal_(
                base_graph,
                std=float(base_initialisation_std),
            )
            self.base_graph = nn.Parameter(
                base_graph
            )
            self.register_buffer(
                "_fixed_base_adjacency",
                None,
                persistent=False,
            )
        else:
            singleton_fixed = _prepare_fixed_adjacency(
                fixed_base_adjacency,
                num_heads=self.num_heads,
                num_nodes=self.num_nodes,
                add_self_loops=bool(
                    config.add_self_loops
                ),
            )
            # Fixed bases are stored as probabilities. Masked logits are
            # rebuilt in the active dtype to avoid float16 -inf masks.
            self.register_buffer(
                "base_graph",
                singleton_fixed[0],
                persistent=True,
            )
            self.register_buffer(
                "_fixed_base_adjacency",
                singleton_fixed,
                persistent=True,
            )

        self.normalizer = GraphNormalizer(config)
        self.gate_type = config.gate_type
        self.initial_alpha = float(
            config.initial_alpha
        )
        self._make_alpha_parameter()

    @staticmethod
    def _alpha_to_raw(
        alpha: float,
    ) -> float:
        epsilon = 1.0e-6
        clipped = min(
            max(float(alpha), epsilon),
            1.0 - epsilon,
        )
        return math.log(
            clipped / (1.0 - clipped)
        )

    def _make_alpha_parameter(self) -> None:
        if self.gate_type not in {
            "none",
            "fixed",
            "learned_scalar",
            "learned_per_head",
        }:
            raise ValueError(
                f"Unsupported gate type {self.gate_type!r}."
            )

        if self.gate_type == "none":
            self.register_parameter(
                "dynamic_residual_raw",
                None,
            )
            return

        raw_initial = self._alpha_to_raw(
            self.initial_alpha
        )
        shape = (
            (self.num_heads,)
            if self.gate_type
            == "learned_per_head"
            else (1,)
        )
        raw = torch.full(
            shape,
            raw_initial,
            dtype=torch.float32,
        )

        if self.gate_type in {
            "learned_scalar",
            "learned_per_head",
        }:
            self.dynamic_residual_raw = nn.Parameter(
                raw
            )
        else:
            self.register_buffer(
                "dynamic_residual_raw",
                raw,
                persistent=True,
            )

    def dynamic_residual_alpha(self) -> Tensor:
        if (
            self.gate_type == "none"
            or self.dynamic_residual_raw
            is None
        ):
            return torch.tensor(
                1.0,
                device=self.q_proj.weight.device,
            )
        return torch.sigmoid(
            self.dynamic_residual_raw
        )

    def _alpha_view(
        self,
        logits: Tensor,
    ) -> Tensor:
        alpha = self.dynamic_residual_alpha().to(
            device=logits.device,
            dtype=logits.dtype,
        )

        if alpha.ndim == 0 or alpha.numel() == 1:
            return alpha.view(1, 1, 1, 1)

        return alpha.view(
            1,
            self.num_heads,
            1,
            1,
        )

    def singleton_base_logits(self) -> Tensor:
        if self._fixed_base_adjacency is not None:
            return _normalisation_logits_from_adjacency(
                self._fixed_base_adjacency
            )
        return self.base_graph.unsqueeze(0)

    def singleton_base_adjacency(self) -> Tensor:
        if self._fixed_base_adjacency is not None:
            return self._fixed_base_adjacency
        return self.normalizer(
            self.singleton_base_logits()
        )

    def dynamic_logits(
        self,
        context_hidden: Tensor,
    ) -> Tensor:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
            d_model=self.d_model,
        )
        origin_hidden = context_hidden[:, -1]

        queries = (
            self.q_proj(origin_hidden)
            .view(
                batch_size,
                self.num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(origin_hidden)
            .view(
                batch_size,
                self.num_nodes,
                self.num_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )

        return (
            queries
            @ keys.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

    def forward(
        self,
        context_hidden: Tensor,
    ) -> GraphOutput:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
            d_model=self.d_model,
        )

        dynamic_logits = self.dynamic_logits(
            context_hidden
        )

        # Graph probabilities and convex graph mixtures stay in float32
        # under AMP. The local spatial aggregation casts the validated graph
        # to the value-projection dtype only for the einsum.
        dynamic_logits_float = dynamic_logits.float()
        base_adjacency_singleton = (
            self.singleton_base_adjacency()
            .to(
                device=dynamic_logits.device,
                dtype=torch.float32,
            )
        )

        if self.uses_fixed_base:
            base_logits_singleton = (
                _normalisation_logits_from_adjacency(
                    base_adjacency_singleton
                )
            )
        else:
            base_logits_singleton = (
                self.singleton_base_logits()
                .to(
                    device=dynamic_logits.device,
                    dtype=torch.float32,
                )
            )

        base_logits = _expand_singleton_graph(
            base_logits_singleton,
            batch_size=batch_size,
        )
        full_logits = (
            base_logits
            + dynamic_logits_float
        )

        base_adjacency = _expand_singleton_graph(
            base_adjacency_singleton,
            batch_size=batch_size,
        )
        full_dynamic_adjacency = self.normalizer(
            full_logits
        )

        alpha = self.dynamic_residual_alpha()

        if self.gate_type == "none":
            selected = full_dynamic_adjacency
        else:
            alpha_view = self._alpha_view(
                full_dynamic_adjacency
            )
            selected = (
                (1.0 - alpha_view)
                * base_adjacency
                + alpha_view
                * full_dynamic_adjacency
            )

        output = GraphOutput(
            selected=selected,
            base=base_adjacency_singleton,
            dynamic=full_dynamic_adjacency,
            alpha=alpha,
            logits=full_logits,
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


class FreeStaticDynamicGraphLearner(
    BaseDyGraphDynamicBaseGraphLearner
):
    """Learned BaseDyGraph free-static base plus dynamic deviation."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        d_model: int,
        base_initialisation_std: float = 0.02,
    ) -> None:
        super().__init__(
            config=config,
            num_nodes=num_nodes,
            d_model=d_model,
            fixed_base_adjacency=None,
            base_initialisation_std=(
                base_initialisation_std
            ),
        )


class CorrelationDynamicGraphLearner(
    BaseDyGraphDynamicBaseGraphLearner
):
    """Absolute-correlation base plus BaseDyGraph dynamic deviation."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        correlation_matrix: Tensor,
        threshold: float,
        d_model: int,
        empty_row_policy: EmptyCorrelationRowPolicy = "error",
    ) -> None:
        if correlation_matrix.ndim != 2:
            raise ValueError(
                "correlation_matrix must have shape [N, N]."
            )

        num_nodes = int(
            correlation_matrix.shape[0]
        )
        fixed_adjacency = (
            build_absolute_correlation_adjacency(
                correlation_matrix,
                threshold=threshold,
                num_heads=int(
                    config.num_heads
                ),
                add_self_loops=bool(
                    config.add_self_loops
                ),
                empty_row_policy=(
                    empty_row_policy
                ),
            )
        )

        super().__init__(
            config=config,
            num_nodes=num_nodes,
            d_model=d_model,
            fixed_base_adjacency=(
                fixed_adjacency
            ),
        )

        self.threshold = float(threshold)
        self.empty_row_policy = (
            empty_row_policy
        )


class OracleGraphLearner(nn.Module):
    """Expose known graph truth only in explicit oracle mode."""

    def __init__(
        self,
        *,
        config: GraphConfig,
        num_nodes: int,
        oracle_graph: Tensor | None = None,
    ) -> None:
        super().__init__()

        if config.type != "oracle":
            raise ValueError(
                "OracleGraphLearner requires graph.type='oracle'."
            )

        config.validate(
            num_nodes=num_nodes,
            d_model=1,
        )

        self.config = config
        self.num_nodes = int(num_nodes)
        self.num_heads = int(config.num_heads)

        if oracle_graph is None:
            self.register_buffer(
                "_stored_oracle_graph",
                None,
                persistent=False,
            )
        else:
            self.register_buffer(
                "_stored_oracle_graph",
                self._prepare_oracle(
                    oracle_graph,
                    batch_size=None,
                ),
                persistent=True,
            )

    def _prepare_oracle(
        self,
        oracle_graph: Tensor,
        *,
        batch_size: int | None,
    ) -> Tensor:
        graph = torch.as_tensor(
            oracle_graph,
            dtype=torch.float32,
        )

        if graph.ndim == 2:
            graph = graph.unsqueeze(0).unsqueeze(0)
        elif graph.ndim == 3:
            if tuple(graph.shape[-2:]) != (
                self.num_nodes,
                self.num_nodes,
            ):
                raise ValueError(
                    "oracle_graph node axes do not match num_nodes."
                )

            if (
                int(graph.shape[0])
                in {1, self.num_heads}
                and batch_size in {None, 1}
            ):
                graph = graph.unsqueeze(0)
            else:
                graph = graph.unsqueeze(1)
        elif graph.ndim != 4:
            raise ValueError(
                "oracle_graph must have shape [N,N], [B,N,N], "
                "[G,N,N], or [B,G,N,N]."
            )

        if tuple(graph.shape[-2:]) != (
            self.num_nodes,
            self.num_nodes,
        ):
            raise ValueError(
                "oracle_graph node axes do not match num_nodes."
            )

        if int(graph.shape[1]) == 1 and self.num_heads > 1:
            graph = graph.expand(
                -1,
                self.num_heads,
                -1,
                -1,
            )

        if int(graph.shape[1]) != self.num_heads:
            raise ValueError(
                f"Expected one or {self.num_heads} oracle heads; "
                f"received {int(graph.shape[1])}."
            )

        if batch_size is not None:
            if int(graph.shape[0]) == 1 and batch_size > 1:
                graph = graph.expand(
                    batch_size,
                    -1,
                    -1,
                    -1,
                )
            if int(graph.shape[0]) != batch_size:
                raise ValueError(
                    f"Expected oracle batch size {batch_size}; "
                    f"received {int(graph.shape[0])}."
                )

        if not torch.isfinite(graph).all():
            raise ValueError(
                "oracle_graph contains non-finite values."
            )

        if torch.any(graph < 0):
            raise ValueError(
                "oracle_graph contains negative weights."
            )

        if not self.config.add_self_loops:
            diagonal = torch.diagonal(
                graph,
                dim1=-2,
                dim2=-1,
            )
            if not torch.allclose(
                diagonal,
                torch.zeros_like(diagonal),
                atol=1.0e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    "oracle_graph contains self loops while "
                    "they are disabled."
                )

        if not torch.allclose(
            graph.sum(dim=-1),
            torch.ones_like(
                graph.sum(dim=-1)
            ),
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "oracle_graph must already be row-stochastic."
            )

        return graph.contiguous()

    def forward(
        self,
        context_hidden: Tensor,
        *,
        oracle_graph: Tensor | None = None,
    ) -> GraphOutput:
        batch_size, _, _, _ = _validate_context_hidden(
            context_hidden,
            num_nodes=self.num_nodes,
        )

        source = (
            oracle_graph
            if oracle_graph is not None
            else self._stored_oracle_graph
        )

        if source is None:
            raise ValueError(
                "OracleGraphLearner requires oracle_graph "
                "for this batch."
            )

        selected = self._prepare_oracle(
            source,
            batch_size=batch_size,
        ).to(
            device=context_hidden.device,
            dtype=context_hidden.dtype,
        )

        output = GraphOutput(
            selected=selected,
            base=None,
            dynamic=None,
            alpha=None,
            logits=None,
        )
        output.validate(
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_nodes=self.num_nodes,
        )
        return output


def build_graph_learner(
    *,
    config: GraphConfig,
    num_nodes: int,
    d_model: int,
    fixed_adjacency: Tensor | None = None,
    correlation_matrix: Tensor | None = None,
    correlation_threshold: float | None = None,
    correlation_empty_row_policy: (
        EmptyCorrelationRowPolicy
    ) = "error",
    oracle_graph: Tensor | None = None,
) -> nn.Module:
    """Build any project graph learner behind one common interface."""
    if config.type == "none":
        return NoGraphLearner(
            config=config,
            num_nodes=num_nodes,
        )

    if config.type == "fixed":
        if (
            fixed_adjacency is not None
            and correlation_matrix is not None
        ):
            raise ValueError(
                "Supply fixed_adjacency or correlation_matrix, not both."
            )

        if correlation_matrix is not None:
            if correlation_threshold is None:
                raise ValueError(
                    "correlation_threshold is required with "
                    "correlation_matrix."
                )
            return AbsoluteCorrelationGraphLearner(
                config=config,
                correlation_matrix=correlation_matrix,
                threshold=correlation_threshold,
                empty_row_policy=(
                    correlation_empty_row_policy
                ),
            )

        if fixed_adjacency is None:
            raise ValueError(
                "graph.type='fixed' requires a supplied adjacency."
            )

        return FixedGraphLearner(
            config=config,
            num_nodes=num_nodes,
            adjacency=fixed_adjacency,
        )

    if config.type == "free_static":
        return FreeStaticGraphLearner(
            config=config,
            num_nodes=num_nodes,
        )

    if config.type == "mtgnn_static":
        return MTGNNStaticGraphLearner(
            config=config,
            num_nodes=num_nodes,
        )

    if config.type == "dynamic":
        return BaseDyGraphDynamicGraphLearner(
            config=config,
            num_nodes=num_nodes,
            d_model=d_model,
        )

    if config.type == "dynamic_base":
        if (
            fixed_adjacency is not None
            and correlation_matrix is not None
        ):
            raise ValueError(
                "Supply fixed_adjacency or correlation_matrix, not both."
            )

        if correlation_matrix is not None:
            if correlation_threshold is None:
                raise ValueError(
                    "correlation_threshold is required with "
                    "correlation_matrix."
                )
            return CorrelationDynamicGraphLearner(
                config=config,
                correlation_matrix=correlation_matrix,
                threshold=correlation_threshold,
                d_model=d_model,
                empty_row_policy=(
                    correlation_empty_row_policy
                ),
            )

        if (
            fixed_adjacency is None
            and config.base_graph_type
            != "free_static"
        ):
            raise ValueError(
                "A learned BaseDyGraph dynamic-base graph requires "
                "graph.base_graph_type='free_static'. Supply a fixed "
                "or correlation adjacency explicitly for a fixed base."
            )

        return BaseDyGraphDynamicBaseGraphLearner(
            config=config,
            num_nodes=num_nodes,
            d_model=d_model,
            fixed_base_adjacency=(
                fixed_adjacency
            ),
        )

    if config.type == "oracle":
        return OracleGraphLearner(
            config=config,
            num_nodes=num_nodes,
            oracle_graph=oracle_graph,
        )

    raise RuntimeError(
        f"Unhandled graph type {config.type!r}."
    )


def _assert_graph(
    graph: Tensor,
    *,
    name: str,
    zero_diagonal: bool = True,
) -> None:
    if not torch.isfinite(graph).all():
        raise AssertionError(
            f"{name} contains non-finite values."
        )

    if torch.any(graph < 0):
        raise AssertionError(
            f"{name} contains negative weights."
        )

    if not torch.allclose(
        graph.sum(dim=-1),
        torch.ones_like(
            graph.sum(dim=-1)
        ),
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise AssertionError(
            f"{name} is not row-stochastic."
        )

    if zero_diagonal:
        diagonal = torch.diagonal(
            graph,
            dim1=-2,
            dim2=-1,
        )
        if not torch.allclose(
            diagonal,
            torch.zeros_like(diagonal),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise AssertionError(
                f"{name} has non-zero self loops."
            )


def _assert_gradient(
    parameter: Tensor,
    *,
    name: str,
) -> None:
    gradient = parameter.grad

    if gradient is None:
        raise AssertionError(
            f"{name} did not receive a gradient."
        )

    if not torch.isfinite(gradient).all():
        raise AssertionError(
            f"{name} received non-finite gradients."
        )

    if float(gradient.abs().sum()) == 0.0:
        raise AssertionError(
            f"{name} received only zero gradients."
        )


def _test_config(
    graph_type: str,
    *,
    num_heads: int = 2,
    gate_type: str = "learned_scalar",
    initial_alpha: float = 0.35,
) -> GraphConfig:
    return GraphConfig(
        type=graph_type,  # type: ignore[arg-type]
        num_heads=num_heads,
        hidden_dim=8,
        activation="softmax",
        add_self_loops=False,
        mtgnn_embedding_dim=4,
        mtgnn_top_k=2,
        mtgnn_alpha=1.0,
        base_graph_type="free_static",
        gate_type=gate_type,  # type: ignore[arg-type]
        initial_alpha=initial_alpha,
    )


def _test_spatial(
    hidden: Tensor,
    graph: Tensor,
) -> None:
    spatial = SpatialMessagePassing(
        d_model=int(hidden.shape[-1]),
        num_heads=int(graph.shape[1]),
        num_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    output = spatial(hidden, graph)

    if (
        tuple(output.shape) != tuple(hidden.shape)
        or not torch.isfinite(output).all()
    ):
        raise AssertionError(
            "Graph is incompatible with SpatialMessagePassing."
        )


def _cpu_smoke_test() -> None:
    torch.manual_seed(42)

    batch_size = 3
    context_length = 7
    num_nodes = 6
    d_model = 12
    num_heads = 2

    hidden = torch.randn(
        batch_size,
        context_length,
        num_nodes,
        d_model,
    )

    no_graph = NoGraphLearner(
        config=_test_config(
            "none",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
    )
    if no_graph(hidden).selected is not None:
        raise AssertionError(
            "NoGraphLearner returned an adjacency."
        )

    fixed_weights = torch.zeros(
        num_nodes,
        num_nodes,
    )
    for target in range(num_nodes):
        fixed_weights[
            target,
            (target + 1) % num_nodes,
        ] = 2.0
        fixed_weights[
            target,
            (target + 2) % num_nodes,
        ] = 1.0

    fixed = FixedGraphLearner(
        config=_test_config(
            "fixed",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
        adjacency=fixed_weights,
    )
    fixed_output = fixed(hidden)
    if fixed_output.selected is None:
        raise AssertionError(
            "FixedGraphLearner returned no graph."
        )
    _assert_graph(
        fixed_output.selected,
        name="fixed graph",
    )

    correlation = torch.tensor(
        [
            [1.0, 0.8, -0.6, 0.1, 0.4, 0.2],
            [0.8, 1.0, 0.5, 0.3, 0.2, 0.7],
            [-0.6, 0.5, 1.0, 0.9, 0.1, 0.4],
            [0.1, 0.3, 0.9, 1.0, 0.8, 0.5],
            [0.4, 0.2, 0.1, 0.8, 1.0, 0.9],
            [0.2, 0.7, 0.4, 0.5, 0.9, 1.0],
        ]
    )

    correlation_learner = (
        AbsoluteCorrelationGraphLearner(
            config=_test_config(
                "fixed",
                num_heads=num_heads,
            ),
            correlation_matrix=correlation,
            threshold=0.5,
        )
    )
    correlation_output = correlation_learner(
        hidden
    )
    if correlation_output.selected is None:
        raise AssertionError(
            "Correlation learner returned no graph."
        )
    _assert_graph(
        correlation_output.selected,
        name="correlation graph",
    )

    free_static = FreeStaticGraphLearner(
        config=_test_config(
            "free_static",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
    )
    free_output = free_static(hidden)
    if free_output.selected is None:
        raise AssertionError(
            "FreeStaticGraphLearner returned no graph."
        )
    _assert_graph(
        free_output.selected,
        name="free-static graph",
    )
    (
        free_output.selected
        * torch.randn_like(
            free_output.selected
        )
    ).sum().backward()
    _assert_gradient(
        free_static.logits,
        name="free-static logits",
    )

    mtgnn = MTGNNStaticGraphLearner(
        config=_test_config(
            "mtgnn_static",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
        training_tie_break_noise=0.0,
    )
    mtgnn.eval()
    mtgnn_output = mtgnn(hidden)
    if mtgnn_output.selected is None:
        raise AssertionError(
            "MTGNN learner returned no graph."
        )
    if not torch.equal(
        (mtgnn_output.selected > 0).sum(dim=-1),
        torch.full_like(
            (mtgnn_output.selected > 0).sum(dim=-1),
            2,
        ),
    ):
        raise AssertionError(
            "MTGNN graph did not retain exactly top-k edges."
        )

    dynamic = BaseDyGraphDynamicGraphLearner(
        config=_test_config(
            "dynamic",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
        d_model=d_model,
    )
    dynamic_output = dynamic(hidden)
    if dynamic_output.selected is None:
        raise AssertionError(
            "Dynamic learner returned no graph."
        )
    _assert_graph(
        dynamic_output.selected,
        name="dynamic graph",
    )
    if torch.allclose(
        dynamic_output.selected[0],
        dynamic_output.selected[1],
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise AssertionError(
            "The dynamic graph did not vary across contexts."
        )
    (
        dynamic_output.selected
        * torch.randn_like(
            dynamic_output.selected
        )
    ).sum().backward()
    _assert_gradient(
        dynamic.q_proj.weight,
        name="dynamic q projection",
    )
    _assert_gradient(
        dynamic.k_proj.weight,
        name="dynamic k projection",
    )

    combined = FreeStaticDynamicGraphLearner(
        config=_test_config(
            "dynamic_base",
            num_heads=num_heads,
            gate_type="learned_per_head",
            initial_alpha=0.35,
        ),
        num_nodes=num_nodes,
        d_model=d_model,
    )
    combined_output = combined(hidden)
    if any(
        value is None
        for value in (
            combined_output.selected,
            combined_output.base,
            combined_output.dynamic,
            combined_output.alpha,
        )
    ):
        raise AssertionError(
            "Learned static+dynamic output is incomplete."
        )

    assert combined_output.selected is not None
    assert combined_output.base is not None
    assert combined_output.dynamic is not None
    assert combined_output.alpha is not None

    alpha_view = combined_output.alpha.view(
        1,
        num_heads,
        1,
        1,
    )
    expected = (
        (1.0 - alpha_view)
        * combined_output.base.expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        + alpha_view
        * combined_output.dynamic
    )
    if not torch.allclose(
        combined_output.selected,
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError(
            "The BaseDyGraph convex gate formula is incorrect."
        )

    (
        combined_output.selected
        * torch.randn_like(
            combined_output.selected
        )
    ).sum().backward()

    _assert_gradient(
        combined.base_graph,
        name="combined static logits",
    )
    _assert_gradient(
        combined.q_proj.weight,
        name="combined q projection",
    )
    _assert_gradient(
        combined.k_proj.weight,
        name="combined k projection",
    )

    if not isinstance(
        combined.dynamic_residual_raw,
        nn.Parameter,
    ):
        raise AssertionError(
            "The learned gate is not a parameter."
        )
    _assert_gradient(
        combined.dynamic_residual_raw,
        name="combined alpha",
    )

    correlation_dynamic = (
        CorrelationDynamicGraphLearner(
            config=_test_config(
                "dynamic_base",
                num_heads=num_heads,
                gate_type="fixed",
                initial_alpha=0.4,
            ),
            correlation_matrix=correlation,
            threshold=0.5,
            d_model=d_model,
        )
    )
    correlation_dynamic_output = (
        correlation_dynamic(hidden)
    )
    if any(
        value is None
        for value in (
            correlation_dynamic_output.selected,
            correlation_dynamic_output.base,
            correlation_dynamic_output.dynamic,
            correlation_dynamic_output.alpha,
        )
    ):
        raise AssertionError(
            "Correlation+dynamic output is incomplete."
        )

    assert correlation_dynamic_output.selected is not None
    assert correlation_dynamic_output.base is not None
    assert correlation_dynamic_output.dynamic is not None
    assert correlation_dynamic_output.alpha is not None

    fixed_alpha = (
        correlation_dynamic_output.alpha
        .to(
            correlation_dynamic_output
            .selected.dtype
        )
        .view(1, 1, 1, 1)
    )
    expected_correlation = (
        (1.0 - fixed_alpha)
        * correlation_dynamic_output.base.expand(
            batch_size,
            -1,
            -1,
            -1,
        )
        + fixed_alpha
        * correlation_dynamic_output.dynamic
    )
    if not torch.allclose(
        correlation_dynamic_output.selected,
        expected_correlation,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError(
            "The correlation+dynamic convex formula is incorrect."
        )

    oracle = OracleGraphLearner(
        config=_test_config(
            "oracle",
            num_heads=num_heads,
        ),
        num_nodes=num_nodes,
    )
    oracle_truth = fixed_output.selected[:, 0]
    oracle_output = oracle(
        hidden,
        oracle_graph=oracle_truth,
    )
    if oracle_output.selected is None:
        raise AssertionError(
            "Oracle learner returned no graph."
        )
    if not torch.equal(
        oracle_output.selected[:, 0],
        oracle_truth,
    ):
        raise AssertionError(
            "Oracle graph was altered."
        )

    graph_outputs = {
        "fixed": fixed_output,
        "correlation": correlation_output,
        "free_static": free_output,
        "mtgnn_static": mtgnn_output,
        "dynamic": dynamic_output,
        "free_static_dynamic": combined_output,
        "correlation_dynamic": (
            correlation_dynamic_output
        ),
        "oracle": oracle_output,
    }

    for name, output in graph_outputs.items():
        if output.selected is None:
            raise AssertionError(
                f"{name} has no selected graph."
            )
        _assert_graph(
            output.selected,
            name=name,
        )
        _test_spatial(
            hidden,
            output.selected,
        )

    print(
        "ALL GRAPH LEARNER CPU SMOKE TESTS PASSED"
    )
    print(
        "Tested: none, fixed, absolute correlation, free static, "
        "MTGNN static, BaseDyGraph dynamic, learned static+dynamic, "
        "correlation+dynamic, and oracle."
    )


if __name__ == "__main__":
    _cpu_smoke_test()


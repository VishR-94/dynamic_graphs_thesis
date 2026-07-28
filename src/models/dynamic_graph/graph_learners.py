from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from .contracts import GraphConfig, GraphOutput
from .modules import GraphNormalizer, SpatialMessagePassing


_MASK_FLOAT32: Final[float] = -1.0e9
_MASK_LOW_PRECISION: Final[float] = -1.0e4


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


def _cpu_smoke_test() -> None:
    torch.manual_seed(42)

    num_nodes = 12
    top_k = 4
    d_model = 16

    config = GraphConfig(
        type="mtgnn_static",
        num_heads=2,
        activation="softmax",
        add_self_loops=False,
        mtgnn_embedding_dim=8,
        mtgnn_top_k=top_k,
        mtgnn_alpha=3.0,
    )

    learner = MTGNNStaticGraphLearner(
        config=config,
        num_nodes=num_nodes,
    )
    spatial = SpatialMessagePassing(
        d_model=d_model,
        num_heads=config.num_heads,
        num_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )

    hidden = torch.randn(
        3,
        60,
        num_nodes,
        d_model,
    )

    learner.train()
    train_output = learner(
        hidden
    )

    if train_output.selected is None:
        raise AssertionError(
            "MTGNN learner returned no selected graph."
        )

    expected_shape = (
        3,
        2,
        num_nodes,
        num_nodes,
    )
    if tuple(
        train_output.selected.shape
    ) != expected_shape:
        raise AssertionError(
            "Unexpected selected graph shape."
        )

    nonzero_per_row = (
        train_output.selected > 0
    ).sum(dim=-1)
    if not torch.equal(
        nonzero_per_row,
        torch.full_like(
            nonzero_per_row,
            top_k,
        ),
    ):
        raise AssertionError(
            "MTGNN graph does not retain exactly top_k sources."
        )

    diagonal = torch.diagonal(
        train_output.selected,
        dim1=-2,
        dim2=-1,
    )
    if not torch.equal(
        diagonal,
        torch.zeros_like(diagonal),
    ):
        raise AssertionError(
            "Self loops were not removed."
        )

    spatial_output = spatial(
        hidden,
        train_output.selected,
    )
    if tuple(spatial_output.shape) != tuple(hidden.shape):
        raise AssertionError(
            "SpatialMessagePassing returned an unexpected shape."
        )
    if not torch.isfinite(spatial_output).all():
        raise AssertionError(
            "SpatialMessagePassing returned non-finite values."
        )

    loss = (
        train_output.selected
        * torch.randn_like(
            train_output.selected
        )
    ).sum()
    loss.backward()

    graph_parameters = [
        parameter
        for name, parameter in learner.named_parameters()
        if name.startswith(
            (
                "embedding_1",
                "embedding_2",
                "projection_1",
                "projection_2",
            )
        )
    ]
    if not graph_parameters:
        raise AssertionError(
            "The graph learner has no MTGNN parameters."
        )
    if any(
        parameter.grad is None
        or not torch.isfinite(
            parameter.grad
        ).all()
        for parameter in graph_parameters
    ):
        raise AssertionError(
            "An MTGNN graph parameter has a missing or invalid gradient."
        )

    learner.eval()
    with torch.no_grad():
        first = learner(hidden)
        second = learner(hidden)

    if first.selected is None or second.selected is None:
        raise AssertionError(
            "Evaluation graph is missing."
        )

    if not torch.equal(
        first.selected,
        second.selected,
    ):
        raise AssertionError(
            "Evaluation graphs are not deterministic."
        )

    if not torch.equal(
        first.selected[0],
        first.selected[1],
    ):
        raise AssertionError(
            "A static graph changed across batch examples."
        )

    if torch.allclose(
        first.selected,
        first.selected.transpose(-1, -2),
        atol=1.0e-7,
        rtol=0.0,
    ):
        raise AssertionError(
            "The learned graph is unexpectedly symmetric."
        )

    print(
        "MTGNN static graph learner CPU smoke test passed."
    )
    print(
        "Selected graph:",
        tuple(first.selected.shape),
    )
    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in learner.parameters()
            if parameter.requires_grad
        ),
    )


if __name__ == "__main__":
    _cpu_smoke_test()

"""Model configuration and shared building blocks for BaseDyGraph.

This module defines :class:`ModelConfig`, the central dataclass describing a
spatio-temporal dynamic-graph model, together with small reusable components
(positional encoding and attention masks) used across the architecture.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for a spatio-temporal dynamic-graph model.

    Fields are grouped by concern: core dimensions, module selection,
    graph activation, dynamic-residual gating, the interlaced ST stack,
    lead-lag/propagation scorers, base-graph priors, graph diagnostics,
    and graph regularisation. Fields whose meaning or valid values are not
    obvious from the name are annotated inline.
    """

    num_states: int
    num_nodes: int
    d_model: int = 128
    nhead: int = 4
    num_temporal_layers: int = 3
    num_spatial_layers: int = 1
    dropout: float = 0.1
    ff_mult: int = 4
    max_seq_len: int = 512
    num_edge_heads: int = 4
    graph_hidden_dim: int = 128
    spatial_dropout: float = 0.1
    use_node_embedding: bool = True
    use_state_pair_bias: bool = False
    add_self_loops: bool = False
    symmetric_graph: bool = False
    predict_next_state: bool = True
    temporal_module_type: str = "transformer"
    temporal_context_window: int | None = None  # None = full causal; int W = attend only to the last W steps
    spatial_module_type: str = "dynamic_graph"
    spatial_value: str = "hidden"   # "hidden" | "state_embedding" | "concat" (message/value source)
    scorer_value: str = "hidden"    # "hidden" | "state_embedding" (what the dynamic scorer keys on)
    graph_activation: str = "softmax"   # "softmax" | "sparsemax" | "entmax15" | "gated"

    # Optional per-block override: a list of activations, one per ST block
    # (length must equal num_st_blocks). None => use graph_activation for every
    # block. Enables e.g. dense softmax in early blocks and sparse sparsemax in
    # the final block: rich internal message passing with a clean, liftable
    # output graph.
    graph_activation_per_block: Optional[list] = None

    # Optional per-block edge-head count. Same triplet semantics as
    # graph_activation_per_block: None | int | (first, internal, final) | full
    # list. Use (n, n, 1) for multi-head internal blocks plus a single-head
    # final graph, since one head yields a single coherent graph to lift while
    # multi-head averaging blurs the triangle/2-cell structure.
    num_edge_heads_per_block: Optional[list] = None

    # Optional per-block graph_hidden_dim, the spatial width inside each block
    # (d_model -> graph_hidden_dim -> mix -> d_model). Same triplet semantics:
    # None | int | (first, internal, final) | full list. Lets the final block
    # use a different scorer/value width than the internal blocks. Each block
    # must satisfy block_graph_hidden_dim % block_num_edge_heads == 0.
    graph_hidden_dim_per_block: Optional[list] = None

    log_per_step: bool = False       # False => epoch-only logs; True adds per-step (intra-epoch) logs
    gate_tau: float = 0.5            # temperature for graph_activation="gated"
    gate_row_normalise: bool = True  # row-normalise after gating (controls message scale)

    # Residual gate for spatial_module_type="dynamic_base": blends the dynamic
    # logits against the learned base graph by a weight alpha.
    #   "none"     -> alpha = 1.0 (base + dynamic, ungated)
    #   "scalar"   -> one learnable alpha shared across edge heads
    #   "per_head" -> one learnable alpha per edge head
    # dynamic_residual_init is alpha's initial value in [0, 1]; a small value
    # starts near the base graph and adds dynamic deviation during training.
    dynamic_residual_gate: str = "none"       # "none" | "scalar" | "per_head"
    dynamic_residual_init: float = 1.0         # e.g. 0.05 for a conservative start
    dynamic_residual_learnable: bool = True    # if False, alpha is fixed at its init value
    dynamic_residual_mix: str = "logit"        # "logit" | "convex" | "strict_convex"

    # Interlaced spatio-temporal stack. The default is a single temporal encoder
    # followed by one graph scorer / spatial block. With interlaced_st_blocks=True
    # or num_st_blocks > 1, the backbone runs repeated
    # temporal -> graph scorer -> spatial blocks, so later scorers condition on
    # representations that have already mixed cross-node information.
    interlaced_st_blocks: bool = False
    num_st_blocks: int = 1
    first_spatial_module_type: str | None = None  # optional override for block 0
    st_block_post_norm: bool = True               # LayerNorm after each ST block

    # Propagation-delay (lead-lag) scorer (spatial_module_type="propagation_delay"):
    # the query at step t scores against each node's recent window of keys, so
    # edges can encode directional lagged coupling that a contemporaneous
    # Q_t K_t^T cannot.
    prop_window_size: int = 4                     # S: number of past steps the keys span
    fusion_window_size: int | None = None         # W for the fusion_window scorer (falls back to prop_window_size); slow window for dual_fusion
    fusion_fast_window: int | None = None         # fast (residual) window for dual_fusion (falls back to max(1, slow // 4))
    spatial_use_base: bool = False                # add a base graph to fusion_window/propagation_delay (enables prior injection)

    # Sector/industry prior on the base graph.
    graph_prior_level: str = "none"               # "none" | "sector" | "industry": block-structure prior used to init the base graph
    graph_prior_scale: float = 4.0                # strength of the prior at init
    graph_prior_learnable: bool = True            # True: base graph is a trained Parameter (prior = init); False: frozen at the prior
    prop_lag_aggregation: str = "softmax"         # "softmax" | "max" | "mean" over lags

    # Graph diagnostics for interlaced stacks. graph_eval_layer selects which
    # block's graph is exposed as out["graph_attn"] for evaluation: -1 for the
    # last non-None graph, 0/1/... for a specific block. graph_log_all_layers
    # logs recovery metrics for every block graph under layer-tagged names.
    graph_eval_layer: int = -1
    graph_log_all_layers: bool = True

    # Graph regularisation, off by default and intended for learned dynamic
    # graphs. graph_reg_layer = -1 targets the final non-None graph, 0/1/... a
    # specific block.
    #   graph_entropy_reg         : minimise row entropy directly
    #   graph_target_entropy_reg  : match a target row entropy (sharpen without collapse)
    #   graph_temporal_smooth_reg : penalise frame-to-frame change
    graph_reg_layer: int = -1
    graph_reg_warmup_epochs: int = 0
    graph_entropy_reg: float = 0.0
    graph_target_entropy: float | None = None
    graph_target_entropy_reg: float = 0.0
    graph_temporal_smooth_reg: float = 0.0


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal positional encodings to a sequence of embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:, :t]


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return an additive (-inf / 0) causal mask of shape (seq_len, seq_len)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def causal_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """Return a causal mask limited to a fixed look-back window.

    Position t attends only to [t - window + 1, ..., t]; window >= seq_len is
    equivalent to a full causal mask. Returns an additive (-inf / 0) mask of
    shape (seq_len, seq_len).
    """
    i = torch.arange(seq_len, device=device)
    diff = i[:, None] - i[None, :]                 # query i, key j: i - j
    allowed = (diff >= 0) & (diff < window)        # causal AND within window
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask

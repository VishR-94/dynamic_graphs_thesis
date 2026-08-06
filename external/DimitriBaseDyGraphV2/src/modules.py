"""Neural building blocks for BaseDyGraph.

BaseDyGraph is a spatio-temporal model over discrete token sequences. This
module collects the reusable components:

* temporal encoders that contextualise each node's sequence,
* spatial graph scorers that infer a (possibly time-varying) adjacency A_t,
* message-passing blocks that mix node states over the inferred graph,
* sparse graph activations (sparsemax / 1.5-entmax) used to normalise edges.

Shape convention used throughout:
    B  batch,  N  nodes,  T  time steps,  D  model width,  H  edge heads.
Attention/adjacency tensors are ``(B, T, H, N, N)`` with the last axis being
the neighbour (key) dimension that activations normalise over.
"""

import math
import torch

# Prefer the canonical `entmax` package (Peters et al., 2019) for sparsemax /
# entmax15; fall back to the in-file analytic implementations if unavailable.
try:
    from entmax import sparsemax as _pkg_sparsemax, entmax15 as _pkg_entmax15
    _HAVE_ENTMAX_PKG = True
except Exception:
    _HAVE_ENTMAX_PKG = False
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
from typing import Optional, Dict, List, Tuple


def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: Euclidean projection onto the probability simplex.

    Produces exact zeros (sparse rows) and is differentiable almost everywhere.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim()
    shape[dim] = -1
    rng = rng.view(shape)
    cssv = zs.cumsum(dim) - 1
    cond = (zs - cssv / rng) > 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, (k - 1).clamp_min(0)) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class _Entmax15Fn(torch.autograd.Function):
    """Exact 1.5-entmax with the analytic Jacobian (Peters et al., 2019).

    Forward:
        p_i = [z_i/2 - tau]_+^2, with tau the closed-form sorting threshold that
        enforces sum_i p_i = 1. The threshold uses a sqrt and is deliberately not
        autograd-differentiated; the correct gradient is supplied analytically in
        backward, which avoids both the sqrt-at-zero NaN and the wrong gradient
        that detaching tau would produce.

    Backward (JVP):
        with s = sqrt(p) and S = sum_{support} s,
        grad_z_j = s_j * (g_j - (g . s) / S) on the support, 0 off it.
    """

    @staticmethod
    def forward(ctx, z, dim):
        zz = z / 2.0
        zs, _ = torch.sort(zz, dim=dim, descending=True)
        rng = torch.arange(1, zz.size(dim) + 1, device=zz.device, dtype=zz.dtype)
        shape = [1] * zz.dim()
        shape[dim] = -1
        k = rng.view(shape)
        z_cumsum = zs.cumsum(dim)
        z_sq_cumsum = (zs * zs).cumsum(dim)
        mean = z_cumsum / k
        mean_sq = z_sq_cumsum / k
        ss = k * (mean_sq - mean * mean)
        delta = torch.clamp((1.0 - ss) / k, min=0.0)
        tau = mean - torch.sqrt(delta.clamp_min(1e-12))
        support = (tau <= zs).to(zz.dtype)
        k_star = support.sum(dim=dim, keepdim=True).clamp_min(1)
        tau_star = tau.gather(dim, (k_star.long() - 1).clamp_min(0))
        p = torch.clamp(zz - tau_star, min=0) ** 2
        ctx.save_for_backward(p)
        ctx.dim = dim
        return p

    @staticmethod
    def backward(ctx, grad_output):
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        s = p.sqrt()                                    # s_i = [z_i/2 - tau]_+
        S = s.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        gs = (grad_output * s).sum(dim=dim, keepdim=True) / S
        grad_z = s * (grad_output - gs)                 # zero off the support
        return grad_z, None


def _entmax15(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax, interpolating between softmax and sparsemax.

    Yields exact zeros with the correct analytic gradient (see ``_Entmax15Fn``).
    """
    return _Entmax15Fn.apply(z, dim)


class IdentityTemporalModule(nn.Module):
    """No-op temporal module; returns its input unchanged."""

    def forward(self, x):
        return x


class IdentitySpatialModule(nn.Module):
    """No-op spatial module; returns node states unchanged."""

    def forward(
        self,
        h: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
        e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return h


def build_temporal_module(cfg: ModelConfig) -> nn.Module:
    """Construct the temporal module named by ``cfg.temporal_module_type``."""
    if cfg.temporal_module_type == "none":
        return IdentityTemporalModule()
    elif cfg.temporal_module_type == "transformer":
        return PerNodeTemporalEncoder(cfg)
    else:
        raise ValueError(f"Unknown temporal_module_type: {cfg.temporal_module_type}")


def build_spatial_components(
    cfg: ModelConfig,
) -> tuple[Optional[nn.Module], nn.Module]:
    """Construct the (scorer, message-passing) pair for ``cfg.spatial_module_type``.

    The scorer infers the graph A_t and may be ``None`` (identity spatial mixing);
    the second element performs message passing over that graph.
    """
    if cfg.spatial_module_type == "none":
        return None, IdentitySpatialModule()
    elif cfg.spatial_module_type == "dynamic_graph":
        return DynamicGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "static_graph":
        return StaticGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "dynamic_base":
        return DynamicBaseGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "propagation_delay":
        return PropagationDelayGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "fusion_window":
        return FusionWindowGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "dual_fusion":
        return DualFusionGraphScorer(cfg), SpatialMessagePassing(cfg)
    else:
        raise ValueError(f"Unknown spatial_module_type: {cfg.spatial_module_type}")


# ------------------------------------------------------------
# Temporal encoder
# ------------------------------------------------------------

class PerNodeTemporalEncoder(nn.Module):
    """Shared causal transformer applied independently to each node's sequence.

    Input:  (B, N, T, D)
    Output: (B, N, T, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_temporal_layers)
        self.context_window = getattr(cfg, "temporal_context_window", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, d = x.shape
        x = x.reshape(b * n, t, d)
        x = self.pos_enc(x)
        if self.context_window is not None and self.context_window < t:
            mask = causal_window_mask(t, self.context_window, x.device)
        else:
            mask = causal_mask(t, x.device)
        x = self.encoder(x, mask=mask)
        x = x.reshape(b, n, t, d)
        return x


# ------------------------------------------------------------
# Dynamic graph inference at each time t
# ------------------------------------------------------------

class DynamicGraphScorer(nn.Module):
    """Infer a per-step graph A_t from contextual node embeddings via scaled QK^T.

    Input:
        h:         (B, T, N, D)  contextualised node states
        state_ids: (B, N, T)     discrete state index per node/time
        e:         (B, T, N, D)   optional raw state embedding (see scorer_value)
    Output:
        attn: (B, T, H, N, N)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        head_dim = cfg.graph_hidden_dim // cfg.num_edge_heads
        if cfg.graph_hidden_dim % cfg.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads")
        self.head_dim = head_dim

        _sv = getattr(cfg, "scorer_value", "hidden")
        _in = cfg.d_model * 2 if _sv == "concat" else cfg.d_model
        self.q_proj = nn.Linear(_in, cfg.graph_hidden_dim)
        self.k_proj = nn.Linear(_in, cfg.graph_hidden_dim)
        self.dropout = nn.Dropout(cfg.spatial_dropout)

        # What the scorer keys on when forming edges:
        #   "hidden"          -> q/k from h        (temporal-contextualised hidden; default)
        #   "state_embedding" -> q/k from e        (raw current-state embedding)
        #   "concat"          -> q/k from [h || e] (both timescales; scorer learns the blend)
        # e is the fixed state embedding looked up afresh at every block, so edge
        # formation can see the instantaneous state rather than the temporal summary.
        self.scorer_value = getattr(cfg, "scorer_value", "hidden")

        if cfg.use_state_pair_bias:
            self.state_pair_bias = nn.Parameter(
                torch.zeros(cfg.num_edge_heads, cfg.num_states, cfg.num_states)
            )
            nn.init.normal_(self.state_pair_bias, std=0.02)
        else:
            self.state_pair_bias = None

        # Independent per-edge gate for graph_activation="gated":
        #   A[i,j] = sigmoid((score - theta) / tau)
        # Edges are scored independently rather than competing through a softmax
        # over neighbours. theta is a learnable per-head threshold, tau a fixed
        # temperature; gate_row_normalise rescales rows afterwards.
        self.gate_theta = nn.Parameter(torch.zeros(cfg.num_edge_heads))
        self.gate_tau = getattr(cfg, "gate_tau", 0.5)
        self.gate_row_normalise = getattr(cfg, "gate_row_normalise", True)

        # Optional learnable base graph (H, N, N) added to the QK^T logits before
        # normalisation. It holds a fixed per-edge adjacency while QK^T learns the
        # time-varying deviation from it. Enabled by DynamicBaseGraphScorer; off here.
        self.use_base_graph = False
        self.base_graph = None

        # Residual gate for DynamicBaseGraphScorer. When enabled, the logits are
        #     base_logits + alpha * dynamic_logits
        # instead of base_logits + dynamic_logits. A small initial alpha starts
        # near the base graph and adds dynamic deviation over training. Mode
        # "none" leaves alpha = 1.0.
        self.dynamic_residual_gate = getattr(cfg, "dynamic_residual_gate", "none")
        self.dynamic_residual_init = float(getattr(cfg, "dynamic_residual_init", 1.0))
        self.dynamic_residual_learnable = bool(getattr(cfg, "dynamic_residual_learnable", True))
        # How alpha is applied:
        #   "logit"  : A = normalise(base_logits + alpha * dynamic_logits)
        #   "convex" : A = (1 - alpha) * normalise(base_logits)
        #                  + alpha * normalise(base_logits + dynamic_logits)
        # In convex mode alpha is the mixture weight between the base-only and
        # full dynamic graphs.
        self.dynamic_residual_mix = getattr(cfg, "dynamic_residual_mix", "logit")
        self.dynamic_residual_raw = None

    @staticmethod
    def _alpha_to_raw(alpha: float) -> float:
        # Map alpha in (0, 1) to the pre-sigmoid parameter, clamped away from the
        # boundary for numerical safety.
        eps = 1e-6
        alpha = min(max(float(alpha), eps), 1.0 - eps)
        return math.log(alpha / (1.0 - alpha))

    def _make_dynamic_residual_parameter(self) -> None:
        """Create the alpha gate parameter.

        Called from subclass ``__init__`` so that the plain ``DynamicGraphScorer``
        carries no gate parameters.
        """
        mode = self.dynamic_residual_gate
        if mode not in {"none", "scalar", "per_head"}:
            raise ValueError(
                f"Unknown dynamic_residual_gate={mode!r}; expected 'none', 'scalar', or 'per_head'"
            )
        if self.dynamic_residual_mix not in {"logit", "convex", "strict_convex"}:
            raise ValueError(
                f"Unknown dynamic_residual_mix={self.dynamic_residual_mix!r}; "
                "expected 'logit', 'convex', or 'strict_convex'"
            )
        if mode == "none":
            self.dynamic_residual_raw = None
            return

        raw_init = self._alpha_to_raw(self.dynamic_residual_init)
        shape = (1,) if mode == "scalar" else (self.num_heads,)
        raw = torch.full(shape, raw_init, dtype=torch.float32)
        if self.dynamic_residual_learnable:
            self.dynamic_residual_raw = nn.Parameter(raw)
        else:
            self.register_buffer("dynamic_residual_raw", raw, persistent=True)

    def dynamic_residual_alpha(self) -> torch.Tensor:
        """Return the gate value alpha for logging/use. Shape: scalar or (H,)."""
        if self.dynamic_residual_gate == "none" or self.dynamic_residual_raw is None:
            return torch.tensor(1.0, device=self.q_proj.weight.device)
        return torch.sigmoid(self.dynamic_residual_raw)

    def _alpha_view(self, logits: torch.Tensor) -> torch.Tensor:
        """Broadcast alpha to (1, 1, H, 1, 1), or a scalar-compatible shape."""
        alpha = self.dynamic_residual_alpha().to(device=logits.device, dtype=logits.dtype)
        if alpha.ndim == 0 or alpha.numel() == 1:
            return alpha.view(1, 1, 1, 1, 1)
        return alpha.view(1, 1, self.num_heads, 1, 1)

    def _base_logits_like(self, logits: torch.Tensor) -> torch.Tensor:
        return self.base_graph.view(1, 1, self.num_heads, logits.size(-1), logits.size(-1))

    def _combine_base_and_dynamic(self, dynamic_logits: torch.Tensor) -> torch.Tensor:
        """Combine base and dynamic logits into normalised attention.

        Plain dynamic_graph: normalise(dynamic_logits).
        dynamic_base:
            gate='none'         -> normalise(base + dynamic)
            mix='logit'         -> normalise(base + alpha * dynamic)
            mix='convex'        -> (1 - alpha) * normalise(base)
                                   + alpha * normalise(base + dynamic)
            mix='strict_convex' -> (1 - alpha) * normalise(base)
                                   + alpha * normalise(dynamic)

        The two convex modes differ at the alpha=1 endpoint. 'convex' keeps the
        base inside the dynamic arm (normalise(base + dynamic)), so alpha never
        removes the base and the base graph trains at full strength for any alpha.
        'strict_convex' interpolates the two pure components base <-> dynamic, so
        alpha=1 is base-free (equivalent to plain dynamic_graph), making alpha a
        genuine learnable base-removal control. Its trade-off: the base appears
        only in the (1 - alpha)-weighted arm, so as alpha -> 1 the base graph
        receives vanishing gradient and goes dormant (paused, not reset; it
        resumes if alpha later falls). Use 'strict_convex' when the model should
        be able to wean off the static prior; use 'convex'/'logit' when the base
        should always co-adapt.
        """
        if not self.use_base_graph or self.base_graph is None:
            return self._normalise(dynamic_logits)

        base_logits = self._base_logits_like(dynamic_logits)

        if self.dynamic_residual_gate == "none":
            return self._normalise(base_logits + dynamic_logits)

        alpha = self._alpha_view(dynamic_logits)

        if self.dynamic_residual_mix == "logit":
            return self._normalise(base_logits + alpha * dynamic_logits)

        if self.dynamic_residual_mix == "convex":
            a_base = self._normalise(base_logits.expand_as(dynamic_logits))
            a_dyn = self._normalise(base_logits + dynamic_logits)
            return (1.0 - alpha) * a_base + alpha * a_dyn

        if self.dynamic_residual_mix == "strict_convex":
            a_base = self._normalise(base_logits.expand_as(dynamic_logits))
            a_dyn = self._normalise(dynamic_logits)
            return (1.0 - alpha) * a_base + alpha * a_dyn

        raise RuntimeError(f"Unhandled dynamic_residual_mix={self.dynamic_residual_mix!r}")

    def _normalise(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalise edge logits into per-row attention over neighbours (last dim).

        graph_activation (default 'softmax'):
          'softmax'   : dense; every neighbour gets non-zero weight.
          'sparsemax' : simplex projection with exact zeros, for a sparse graph.
          'entmax15'  : between softmax and sparsemax.
          'gated'     : independent per-edge sigmoid gate, no competition across
                        neighbours; row-normalised afterwards if gate_row_normalise.

        The sparse activations exist because the target graphs are mostly zeros,
        which softmax cannot represent.
        """
        act = getattr(self.cfg, "graph_activation", "softmax")
        if act == "softmax":
            return torch.softmax(logits, dim=-1)
        elif act == "sparsemax":
            if _HAVE_ENTMAX_PKG:
                return _pkg_sparsemax(logits, dim=-1)
            return _sparsemax(logits, dim=-1)
        elif act == "entmax15":
            if _HAVE_ENTMAX_PKG:
                return _pkg_entmax15(logits, dim=-1)
            return _entmax15(logits, dim=-1)
        elif act == "gated":
            # Independent per-edge gate; no competition across neighbours.
            # logits: (B, T, H, N, N); theta broadcast over the head axis.
            theta = self.gate_theta.view(1, 1, -1, 1, 1)
            gate = torch.sigmoid((logits - theta) / self.gate_tau)
            if self.gate_row_normalise:
                # Rescale row mass without re-imposing competition between edges.
                gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return gate
        else:
            raise ValueError(f"Unknown graph_activation: {act}")

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, d = h.shape
        if self.scorer_value == "state_embedding":
            if e is None:
                raise ValueError("scorer_value='state_embedding' needs e (state embedding)")
            x = e
        elif self.scorer_value == "concat":
            if e is None:
                raise ValueError("scorer_value='concat' needs e (state embedding)")
            x = torch.cat([h, e], dim=-1)             # (B, T, N, 2D); q/k_proj sized 2*d_model
        else:
            x = h
        q = self.q_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B,T,H,N,N)

        if self.state_pair_bias is not None:
            # state_ids: (B, N, T) -> (B, T, N)
            s = state_ids.permute(0, 2, 1)
            s_i = s.unsqueeze(-1).expand(b, t, n, n)
            s_j = s.unsqueeze(-2).expand(b, t, n, n)
            bias = self.state_pair_bias[:, s_i, s_j]  # (H, B, T, N, N)
            bias = bias.permute(1, 2, 0, 3, 4)
            logits = logits + bias

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = self._combine_base_and_dynamic(logits)
        attn = self.dropout(attn)

        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=attn.device, dtype=attn.dtype).view(1, 1, 1, n, n)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return attn


class StaticGraphScorer(nn.Module):
    """Time-invariant graph: a single learnable (H, N, N) adjacency, softmax-normalised.

    The node states only supply batch/time shape; the graph is broadcast across t.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        self.logits = nn.Parameter(
            torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
        )
        nn.init.normal_(self.logits, std=0.02)

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:  # e ignored (static graph)
        # h: (B, T, N, D), used only for batch/time shape.
        b, t, n, _ = h.shape
        logits = self.logits

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = torch.softmax(logits, dim=-1)  # (H, N, N)
        attn = attn.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1, -1).contiguous()
        return attn


class DynamicBaseGraphScorer(DynamicGraphScorer):
    """Dynamic scorer with an added learnable base graph.

        A_t = normalise( base[h, i, j] + QK^T(h_t) / sqrt(d) )

    The base is a raw (H, N, N) parameter (as in StaticGraphScorer); QK^T then
    learns the time-varying deviation from it. With QK^T -> 0 this reduces to the
    static graph.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.use_base_graph = True
        _bg = torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
        nn.init.normal_(_bg, std=0.02)
        if bool(getattr(cfg, "graph_prior_learnable", True)):
            self.base_graph = nn.Parameter(_bg)
        else:
            if hasattr(self, "base_graph"):
                del self.base_graph
            self.register_buffer("base_graph", _bg, persistent=True)
        self._make_dynamic_residual_parameter()


# ------------------------------------------------------------
# Spatial mixing using inferred graph A_t
# ------------------------------------------------------------

class _SpatialMPBlock(nn.Module):
    """A single message-passing block.

    Input:
        h:    (B, T, N, D)
        attn: (B, T, H, N, N)
    Output:
        out:  (B, T, N, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        # Message passing runs in graph_hidden_dim (the spatial width), then
        # projects back to d_model for the residual. This decouples the edge heads
        # from the temporal width (head_dim keys off graph_hidden_dim, not d_model)
        # and lets graph_hidden_dim widen the value/message path, not just the
        # scorer. When graph_hidden_dim == d_model this reduces to the base shapes.
        spatial_dim = getattr(cfg, "graph_hidden_dim", cfg.d_model)
        if spatial_dim % cfg.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads")
        self.spatial_dim = spatial_dim
        self.head_dim = spatial_dim // cfg.num_edge_heads

        # Value mixed over the graph:
        #   "hidden"          -> v_proj(h)         (contextualised hidden state)
        #   "state_embedding" -> v_proj(e)         (raw current-state embedding)
        #   "concat"          -> v_proj([h ; e])   (both)
        self.spatial_value = getattr(cfg, "spatial_value", "hidden")
        in_dim = cfg.d_model * 2 if self.spatial_value == "concat" else cfg.d_model
        self.v_proj = nn.Linear(in_dim, spatial_dim)         # d_model-space -> spatial_dim
        self.out_proj = nn.Linear(spatial_dim, cfg.d_model)  # spatial_dim -> back to d_model
        self.norm_mix = nn.LayerNorm(cfg.d_model)
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, attn: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, d = h.shape
        if self.spatial_value == "state_embedding":
            if e is None:
                raise ValueError("spatial_value='state_embedding' needs e")
            val_in = e
        elif self.spatial_value == "concat":
            if e is None:
                raise ValueError("spatial_value='concat' needs e")
            val_in = torch.cat([h, e], dim=-1)
        else:
            val_in = h
        v = self.v_proj(val_in).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        mixed = torch.matmul(attn, v)  # (B, T, H, N, Hd)
        mixed = mixed.permute(0, 1, 3, 2, 4).reshape(b, t, n, self.spatial_dim)  # spatial_dim space
        mixed = self.out_proj(mixed)   # -> d_model

        # Diagnostic: magnitude of the graph's contribution vs the residual it is
        # added to. ratio << 1 means the graph is being ignored (residual
        # dominates); ~1 means the graph contributes comparable magnitude.
        with torch.no_grad():
            mn = mixed.norm(dim=-1)            # (B, T, N)
            hn = h.norm(dim=-1).clamp_min(1e-6)
            self.last_mix_ratio = (mn / hn).mean().detach()

        h = self.norm_mix(h + mixed)
        h = self.norm_ff(h + self.ff(h))
        return h


class SpatialMessagePassing(nn.Module):
    """Stack of message-passing blocks that reuse the same graph attn.

    The scorer computes attn once; it is then propagated for num_spatial_layers
    hops (default 1).

    Input:
        h:    (B, T, N, D)
        attn: (B, T, H, N, N)
    Output:
        out:  (B, T, N, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        num_layers = getattr(cfg, "num_spatial_layers", 1)
        self.layers = nn.ModuleList([_SpatialMPBlock(cfg) for _ in range(num_layers)])

    def forward(self, h: torch.Tensor, attn: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, attn, e=e)
        return h


# ------------------------------------------------------------
# Propagation-delay (lead-lag) graph scorer
# ------------------------------------------------------------

class PropagationDelayGraphScorer(nn.Module):
    """Lead-lag graph scorer with the same contract as DynamicGraphScorer.

        forward(h, state_ids) -> attn,   h: (B, T, N, D) -> (B, T, H, N, N)

    The query comes from the current step h_t; keys/values come from each node's
    recent window h_{t-S+1:t}. Edge A_t[i, j] compares node i's present against
    node j's recent trajectory, giving directional, lagged coupling that a
    contemporaneous Q_t K_t^T scorer cannot represent. The S per-lag scores per
    (i, j) are collapsed by prop_lag_aggregation (softmax | max | mean). Causal:
    lags only look backward.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.graph_hidden_dim % cfg.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads")
        agg = getattr(cfg, "prop_lag_aggregation", "softmax")
        if agg not in ("softmax", "max", "mean"):
            raise ValueError("prop_lag_aggregation must be: softmax, max, mean")

        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        self.head_dim = cfg.graph_hidden_dim // cfg.num_edge_heads
        self.window_size = int(getattr(cfg, "prop_window_size", 4))
        self.lag_aggregation = agg

        self.q_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.k_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.dropout = nn.Dropout(cfg.spatial_dropout)

        # Optional learnable base graph for sector/industry prior injection.
        self.use_base_graph = bool(getattr(cfg, "spatial_use_base", False))
        if self.use_base_graph:
            _bg = torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
            nn.init.normal_(_bg, std=0.02)
            if bool(getattr(cfg, "graph_prior_learnable", True)):
                self.base_graph = nn.Parameter(_bg)
            else:
                if hasattr(self, "base_graph"):
                    del self.base_graph
                self.register_buffer("base_graph", _bg, persistent=True)
        else:
            self.base_graph = None

    def _windowed_keys(self, k: torch.Tensor):
        """Gather each node's recent-window keys.

        k: (B, T, H, N, hd) -> (B, T, H, N, S, hd), where index s selects timestep
        t-s (front-padded with zeros). Also returns the (T, S) validity mask that
        flags padded (pre-sequence) lags.
        """
        b, t, h, n, hd = k.shape
        s = self.window_size
        pad = torch.zeros(b, s - 1, h, n, hd, device=k.device, dtype=k.dtype)
        kpad = torch.cat([pad, k], dim=1)                       # (B, T+S-1, H, N, hd)
        base = torch.arange(s - 1, s - 1 + t, device=k.device)  # (T,)
        lags = torch.arange(s, device=k.device)                 # (S,)
        idx = base[:, None] - lags[None, :]                     # (T, S)
        kwin = kpad[:, idx]                                     # (B, T, S, H, N, hd)
        kwin = kwin.permute(0, 1, 3, 4, 2, 5)                   # (B, T, H, N, S, hd)
        valid = (idx >= (s - 1)).to(k.dtype)                    # (T, S)
        return kwin, valid

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor,
                e: Optional[torch.Tensor] = None,
                return_lag_scores: bool = False):  # e accepted but unused here
        b, t, n, d = h.shape
        s = self.window_size

        q = self.q_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        kwin, valid = self._windowed_keys(k)                    # (B,T,H,N,S,hd), (T,S)

        scores = torch.einsum("bthid,bthjsd->bthijs", q, kwin) / math.sqrt(self.head_dim)
        pad = (valid == 0).view(1, t, 1, 1, 1, s)
        scores = scores.masked_fill(pad, float("-inf"))
        attn_per_lag = scores

        if self.lag_aggregation == "max":
            logits = attn_per_lag.max(dim=-1).values
        elif self.lag_aggregation == "mean":
            safe = attn_per_lag.masked_fill(pad, 0.0)
            counts = valid.view(1, t, 1, 1, 1, s).sum(-1).clamp_min(1.0)
            logits = safe.sum(-1) / counts.squeeze(-1)
        else:  # softmax over lags
            w = torch.softmax(attn_per_lag, dim=-1)
            vals = attn_per_lag.masked_fill(pad, 0.0)
            logits = (w * vals).sum(-1)

        if self.use_base_graph and self.base_graph is not None:
            logits = logits + self.base_graph.view(1, 1, self.num_heads, n, n)

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=attn.device, dtype=attn.dtype).view(1, 1, 1, n, n)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if return_lag_scores:
            return attn, attn_per_lag
        return attn


class FusionWindowGraphScorer(DynamicGraphScorer):
    """Score a contemporaneous graph per step, then output its causal-window mean.

        A_bar_t = mean over valid w in [0, W-1] of softmax(Q_{t-w} K_{t-w}^T)

    Consecutive output steps share W-1 of their W summands, so A_bar_t varies
    smoothly in t by construction -- a sliding-window low-pass on the graph
    dynamics with no EMA state. Inherits scorer_value (hidden / state_embedding /
    concat) and the projections from DynamicGraphScorer; only the temporal pooling
    of the resulting attention is added. Window from ``fusion_window_size`` (falls
    back to ``prop_window_size``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.window_size = int(getattr(cfg, "fusion_window_size", None)
                               or getattr(cfg, "prop_window_size", 4))
        # Optional learnable base graph for sector/industry prior injection,
        # enabled by spatial_use_base; same mechanism as DynamicBaseGraphScorer.
        if bool(getattr(cfg, "spatial_use_base", False)):
            self.use_base_graph = True
            _bg = torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
            nn.init.normal_(_bg, std=0.02)
            if bool(getattr(cfg, "graph_prior_learnable", True)):
                self.base_graph = nn.Parameter(_bg)
            else:
                if hasattr(self, "base_graph"):
                    del self.base_graph
                self.register_buffer("base_graph", _bg, persistent=True)
            self._make_dynamic_residual_parameter()

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, d = h.shape
        # Select the scorer input exactly as DynamicGraphScorer does.
        if self.scorer_value == "state_embedding":
            x = e
        elif self.scorer_value == "concat":
            x = torch.cat([h, e], dim=-1)
        else:
            x = h

        q = self.q_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # Per-step contemporaneous logits (B, T, H, N, N).
        logits = torch.einsum("bthid,bthjd->bthij", q, k) / math.sqrt(self.head_dim)
        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        s = self.window_size

        def _causal_window_mean(X):
            pad = X.new_zeros(b, s - 1, *X.shape[2:])
            Xpad = torch.cat([pad, X], dim=1)
            cs = Xpad.cumsum(dim=1)
            upper = cs[:, s - 1:]
            lower = torch.cat([cs.new_zeros(b, 1, *X.shape[2:]), cs[:, :-1]], dim=1)[:, :t]
            counts = torch.arange(1, t + 1, device=X.device).clamp_max(s).view(1, t, 1, 1, 1)
            return (upper - lower) / counts

        if self.use_base_graph:
            # Window-average the logits, add the base/prior, normalise once.
            logit_bar = _causal_window_mean(logits)
            A_bar = self._combine_base_and_dynamic(logit_bar)
        else:
            # Window-average the per-step graphs (smoothing), using the configured
            # activation (softmax dense / sparsemax sparse / entmax15).
            A = self._normalise(logits)
            A_bar = _causal_window_mean(A)
        A_bar = self.dropout(A_bar)

        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=A_bar.device, dtype=A_bar.dtype).view(1, 1, 1, n, n)
            A_bar = A_bar + eye
            A_bar = A_bar / A_bar.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return A_bar


class DualFusionGraphScorer(DynamicGraphScorer):
    """Two-timescale fusion of a slow backbone graph and a fast residual graph.

        A_slow_t = window-mean over W_slow of softmax(Q K^T)   (low-frequency backbone)
        A_fast_t = window-mean over W_fast of softmax(Q K^T)   (high-frequency residual)
        A_t      = blend(A_slow, A_fast)   via dynamic_residual_mix / alpha

    An optional sector/industry prior is injected into the slow logits (the
    persistent backbone is where block structure belongs). spatial_use_base
    controls whether the prior/base term is added to the slow graph;
    graph_prior_learnable freezes it.

    Windows: W_slow = fusion_window_size (default 16), W_fast = fusion_fast_window
    (falls back to max(1, W_slow // 4)). The alpha gate is reused from the
    dynamic_residual_* config: alpha is how much fast deviation rides on the slow
    backbone.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.w_slow = int(getattr(cfg, "fusion_window_size", None) or 16)
        self.w_fast = int(getattr(cfg, "fusion_fast_window", None)
                          or max(1, self.w_slow // 4))
        # Optional base/prior on the slow backbone.
        if bool(getattr(cfg, "spatial_use_base", False)):
            self.use_base_graph = True
            _bg = torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
            nn.init.normal_(_bg, std=0.02)
            if bool(getattr(cfg, "graph_prior_learnable", True)):
                self.base_graph = nn.Parameter(_bg)
            else:
                if hasattr(self, "base_graph"):
                    del self.base_graph
                self.register_buffer("base_graph", _bg, persistent=True)
        # Alpha gate between slow and fast (always present for this scorer).
        self._make_dynamic_residual_parameter()

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, d = h.shape
        if self.scorer_value == "state_embedding":
            x = e
        elif self.scorer_value == "concat":
            x = torch.cat([h, e], dim=-1)
        else:
            x = h

        q = self.q_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(x).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        logits = torch.einsum("bthid,bthjd->bthij", q, k) / math.sqrt(self.head_dim)
        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        def _win_mean(X, s):
            if s <= 1:
                return X
            pad = X.new_zeros(b, s - 1, *X.shape[2:])
            Xpad = torch.cat([pad, X], dim=1)
            cs = Xpad.cumsum(dim=1)
            upper = cs[:, s - 1:]
            lower = torch.cat([cs.new_zeros(b, 1, *X.shape[2:]), cs[:, :-1]], dim=1)[:, :t]
            counts = torch.arange(1, t + 1, device=X.device).clamp_max(s).view(1, t, 1, 1, 1)
            return (upper - lower) / counts

        import os as _os
        _DBG = _os.environ.get("DUALFUSION_DEBUG", "0") == "1"

        def _chk(name, x):
            if _DBG and not torch.isfinite(x).all():
                nan = torch.isnan(x).sum().item()
                inf = torch.isinf(x).sum().item()
                raise RuntimeError(f"[dual_fusion] {name} non-finite: nan={nan} inf={inf} "
                                   f"min={x[torch.isfinite(x)].min().item() if torch.isfinite(x).any() else 'NA'} "
                                   f"max={x[torch.isfinite(x)].max().item() if torch.isfinite(x).any() else 'NA'}")

        _chk("logits", logits)
        A = self._normalise(logits)                  # per-step graph (B,T,H,N,N), respects graph_activation
        _chk("A (per-step activation)", A)
        A_slow = _win_mean(A, self.w_slow)           # low-frequency backbone
        A_fast = _win_mean(A, self.w_fast)           # high-frequency residual
        _chk("A_slow (pre-base)", A_slow)
        _chk("A_fast", A_fast)

        # Optionally add the prior/base to the slow backbone (in logit space,
        # then renormalise).
        if self.use_base_graph:
            slow_logit = torch.log(A_slow.clamp_min(1e-9))
            _chk("slow_logit=log(A_slow)", slow_logit)
            slow_logit = slow_logit + self.base_graph.view(1, 1, self.num_heads, n, n)
            _chk("slow_logit+base", slow_logit)
            A_slow = self._normalise(slow_logit)
            _chk("A_slow (post-base)", A_slow)

        # Blend slow (backbone) and fast (residual) with the alpha gate, reusing
        # the convex/strict_convex/logit semantics: slow acts as "base", fast as
        # "dynamic".
        alpha = self.dynamic_residual_alpha()        # scalar or (num_heads,) in [0,1]
        if self.dynamic_residual_gate == "none":
            A_bar = 0.5 * (A_slow + A_fast)
        else:
            a = alpha.view(1, 1, -1, 1, 1) if alpha.dim() else alpha
            mix = self.dynamic_residual_mix
            if mix == "logit":
                slow_logit = torch.log(A_slow.clamp_min(1e-9))
                A_bar = self._normalise(slow_logit + a * torch.log(A_fast.clamp_min(1e-9)))
            elif mix == "convex":
                A_bar = (1 - a) * A_slow + a * (0.5 * (A_slow + A_fast))
            else:  # strict_convex: slow <-> fast endpoints
                A_bar = (1 - a) * A_slow + a * A_fast

        _chk("A_bar (post-blend)", A_bar)
        A_bar = self.dropout(A_bar)
        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=A_bar.device, dtype=A_bar.dtype).view(1, 1, 1, n, n)
            A_bar = A_bar + eye
            A_bar = A_bar / A_bar.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        _chk("A_bar (final)", A_bar)
        return A_bar

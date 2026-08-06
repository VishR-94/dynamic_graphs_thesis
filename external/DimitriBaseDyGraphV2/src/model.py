"""
Backbone, next-state head, and Lightning module for the discrete ST-graph model.

Depends on modules.py and utilities.py for ModelConfig, build_temporal_module,
and build_spatial_components.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from utilities import *  # noqa: F401,F403
from modules import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Graph-shape helpers
# ---------------------------------------------------------------------------

def _row_distribution_entropy(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-row Shannon entropy (nats) of an attention tensor.

    Each row over the last axis is treated as a probability distribution. Returns
    shape (..., N); callers take .mean() for a scalar.

    The bare formula -sum(p log p) is a bounded Shannon entropy (<= log N) only when
    each row sums to 1. softmax / sparsemax / entmax produce normalised rows, but the
    'gated' activation with gate_row_normalise=False does not: its rows are
    independent per-edge sigmoids whose sum is free, and that free row mass is the
    per-node coupling gain. Running -sum(a log a) on those un-normalised rows is not
    an entropy: it is unbounded by log N and can reach ~N/e (an N=32 graph logs ~10),
    and it is not comparable to a row-normalised target.

    Each row is therefore normalised to a distribution here, local to the
    measurement, before taking the entropy. This affects only the relative edge
    weights; the row mass / gain used in the actual message passing is untouched.
    """
    a = attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)
    a = a.clamp_min(eps)
    return -(a * a.log()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class InterlacedSTBlock(nn.Module):
    """One interlaced spatio-temporal block.

    Flow:
        h -> temporal block -> graph scorer -> spatial message passing

    h is always represented as (B, T, N, D) at the block boundary. The temporal
    module internally expects (B, N, T, D), so it is permuted in and out locally.
    """

    def __init__(self, cfg: "ModelConfig", spatial_module_type: str,
                 graph_activation: Optional[str] = None,
                 num_edge_heads: Optional[int] = None,
                 graph_hidden_dim: Optional[int] = None) -> None:  # noqa: F821
        super().__init__()
        self.cfg = cfg
        self.spatial_module_type = spatial_module_type
        # per-block activation (falls back to cfg.graph_activation if None)
        self.graph_activation = graph_activation or getattr(cfg, "graph_activation", "softmax")
        # per-block edge-head count (falls back to cfg.num_edge_heads if None)
        self.num_edge_heads = num_edge_heads or getattr(cfg, "num_edge_heads", 4)
        # per-block spatial width (falls back to cfg.graph_hidden_dim if None)
        self.graph_hidden_dim = graph_hidden_dim or getattr(cfg, "graph_hidden_dim", cfg.d_model)

        self.temporal_module = build_temporal_module(cfg)  # noqa: F821

        if spatial_module_type == "oracle_graph":
            self.graph_scorer = None
            self.spatial_module = SpatialMessagePassing(cfg)  # noqa: F821
        else:
            # Build this block's scorer with a config override so each block can
            # use a different spatial module type while sharing the other graph,
            # normalisation, and gate settings.
            try:
                from dataclasses import replace
                block_cfg = replace(cfg, spatial_module_type=spatial_module_type,
                                    graph_activation=self.graph_activation,
                                    num_edge_heads=self.num_edge_heads,
                                    graph_hidden_dim=self.graph_hidden_dim)
            except Exception:
                block_cfg = cfg
                block_cfg.spatial_module_type = spatial_module_type
                block_cfg.graph_activation = self.graph_activation
                block_cfg.num_edge_heads = self.num_edge_heads
                block_cfg.graph_hidden_dim = self.graph_hidden_dim
            self.graph_scorer, self.spatial_module = build_spatial_components(block_cfg)  # noqa: F821

        self.post_norm = nn.LayerNorm(cfg.d_model) if getattr(cfg, "st_block_post_norm", True) else nn.Identity()

    def _oracle_attn(
        self,
        h_btnd: torch.Tensor,
        regimes: Optional[torch.Tensor],
        oracle_regime_graphs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if regimes is None or oracle_regime_graphs is None:
            raise RuntimeError("oracle_graph block needs regimes and oracle_regime_graphs")
        b, t, n, _ = h_btnd.shape
        G = oracle_regime_graphs.to(h_btnd.device)  # (R, N, N)
        A = G[regimes.long()]                       # (B, T, N, N)
        row_sum = A.sum(dim=-1, keepdim=True)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).view(1, 1, n, n)
        A = torch.where(row_sum > 1e-6, A, eye.expand_as(A))
        return A.unsqueeze(2).expand(b, t, self.cfg.num_edge_heads, n, n)

    def forward(
        self,
        h_btnd: torch.Tensor,
        state_ids: torch.Tensor,
        e_btnd: torch.Tensor,
        regimes: Optional[torch.Tensor] = None,
        oracle_regime_graphs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply temporal mixing, graph scoring, and spatial message passing.

        h_btnd is (B, T, N, D). Returns the updated representation (B, T, N, D) and
        the block's graph attention (B, T, H, N, N), or None when the block has no
        scorer.
        """
        # Temporal module expects (B, N, T, D); block boundary is (B, T, N, D).
        h_bntd = h_btnd.permute(0, 2, 1, 3).contiguous()
        h_bntd = self.temporal_module(h_bntd)
        h_btnd = h_bntd.permute(0, 2, 1, 3).contiguous()

        if self.spatial_module_type == "oracle_graph":
            attn = self._oracle_attn(h_btnd, regimes, oracle_regime_graphs)
            h_btnd = self.spatial_module(h_btnd, attn, e=e_btnd)
        elif self.graph_scorer is not None:
            attn = self.graph_scorer(h_btnd, state_ids, e=e_btnd)
            h_btnd = self.spatial_module(h_btnd, attn, e=e_btnd)
        else:
            attn = None
            h_btnd = self.spatial_module(h_btnd, None, e=e_btnd)

        return self.post_norm(h_btnd), attn


def _resolve_per_block(spec, num_blocks, default, name="per_block"):
    """Resolve a per-block spec with depth-invariant triplet semantics.

    spec accepts:
      None                      -> [default] * num_blocks
      scalar (str/int)          -> [spec]   * num_blocks
      (first, internal, final)  -> first / broadcast-internal / final
                                   (num_blocks==1 -> [final];
                                    num_blocks==2 -> [first, final])
      full-length list          -> used verbatim
    """
    if spec is None:
        return [default] * num_blocks
    if isinstance(spec, (str, int)):
        return [spec] * num_blocks
    if len(spec) == 3:
        first, internal, final = spec
        if num_blocks == 1:
            return [final]
        if num_blocks == 2:
            return [first, final]
        return [first] + [internal] * (num_blocks - 2) + [final]
    if len(spec) == num_blocks:
        return list(spec)
    raise ValueError(
        f"{name} must be None, a scalar, a 3-tuple (first, internal, final), "
        f"or a list of length num_st_blocks ({num_blocks}); got length {len(spec)}")


class DiscreteSTGraphBackbone(nn.Module):
    """State IDs -> embeddings -> temporal/spatial backbone -> head-ready reps.

    The default path is a single temporal stage then one graph/spatial stage.
    Set interlaced_st_blocks=True or num_st_blocks > 1 for repeated
    temporal -> graph -> spatial blocks.
    """

    def __init__(self, cfg: "ModelConfig") -> None:  # noqa: F821 (from modules/utilities)
        super().__init__()
        self.cfg = cfg
        self.use_interlaced = bool(getattr(cfg, "interlaced_st_blocks", False)) or int(getattr(cfg, "num_st_blocks", 1)) > 1

        self.state_embedding = nn.Embedding(cfg.num_states, cfg.d_model)
        self.node_embedding = (
            nn.Embedding(cfg.num_nodes, cfg.d_model) if cfg.use_node_embedding else None
        )
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.post_norm = nn.LayerNorm(cfg.d_model)
        self.oracle_regime_graphs = None

        if self.use_interlaced:
            num_blocks = int(getattr(cfg, "num_st_blocks", 1))
            if num_blocks < 1:
                raise ValueError("num_st_blocks must be >= 1")

            first_type = getattr(cfg, "first_spatial_module_type", None)
            block_types = []
            for i in range(num_blocks):
                if i == 0 and first_type not in {None, "", "same"}:
                    block_types.append(first_type)
                else:
                    block_types.append(cfg.spatial_module_type)

            # Resolve per-block graph_activation. graph_activation_per_block accepts:
            #   None                       -> global cfg.graph_activation for all blocks
            #   3-tuple (first, internal, final)
            #                              -> first block, all middle blocks, last block.
            #                                 'internal' broadcasts to however many middle
            #                                 blocks exist, so this is depth-invariant
            #                                 (change num_st_blocks without rewriting).
            #                                 For num_blocks==1 -> 'final' wins;
            #                                 num_blocks==2 -> [first, final] (no internal).
            #   full-length list           -> explicit activation per block (escape hatch).
            block_acts = _resolve_per_block(
                getattr(cfg, "graph_activation_per_block", None), num_blocks,
                getattr(cfg, "graph_activation", "softmax"),
                name="graph_activation_per_block")

            # Per-block edge-head count (e.g. (n, n, 1): rich multi-head internal
            # + single-head clean liftable final graph).
            block_heads = _resolve_per_block(
                getattr(cfg, "num_edge_heads_per_block", None), num_blocks,
                getattr(cfg, "num_edge_heads", 4),
                name="num_edge_heads_per_block")

            # Per-block spatial (graph_hidden) width. Lets the liftable final
            # block use a different scorer/value width than internal blocks.
            block_gh = _resolve_per_block(
                getattr(cfg, "graph_hidden_dim_per_block", None), num_blocks,
                getattr(cfg, "graph_hidden_dim", cfg.d_model),
                name="graph_hidden_dim_per_block")

            # Cross-triplet constraint: each block's graph_hidden_dim must be
            # divisible by that block's num_edge_heads (heads slice the spatial
            # width inside the block).
            for bi, (g, h) in enumerate(zip(block_gh, block_heads)):
                if g % h != 0:
                    raise ValueError(
                        f"block {bi}: graph_hidden_dim ({g}) must be divisible by "
                        f"num_edge_heads ({h}); check graph_hidden_dim_per_block "
                        f"vs num_edge_heads_per_block")

            self.st_blocks = nn.ModuleList([
                InterlacedSTBlock(cfg, stype, graph_activation=bact,
                                  num_edge_heads=bh, graph_hidden_dim=bg)
                for stype, bact, bh, bg in zip(block_types, block_acts,
                                               block_heads, block_gh)
            ])

            # Handles used by logging/evaluation; point at the last block's
            # scorer and spatial module.
            last = self.st_blocks[-1]
            self.graph_scorer = getattr(last, "graph_scorer", None)
            self.spatial_module = getattr(last, "spatial_module", None)
            self.temporal_module = None
        else:
            self.temporal_module = build_temporal_module(cfg)              # noqa: F821
            if cfg.spatial_module_type == "oracle_graph":
                self.graph_scorer = None
                self.spatial_module = SpatialMessagePassing(cfg)           # noqa: F821
            else:
                self.graph_scorer, self.spatial_module = build_spatial_components(cfg)  # noqa: F821
            self.st_blocks = None

    def _initial_embedding_bntd(self, state_ids: torch.Tensor) -> torch.Tensor:
        b, n, t = state_ids.shape
        x = self.state_embedding(state_ids)
        if self.node_embedding is not None:
            node_ids = torch.arange(n, device=state_ids.device)
            x = x + self.node_embedding(node_ids).view(1, n, 1, self.cfg.d_model)
        return self.pre_norm(x)

    def temporal_output(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Representation fed to the final graph scorer, shape (B, T, N, D).

        Single-stage path: the post-temporal representation. Interlaced path: the
        representation just before the final block's graph scorer.
        """
        x = self._initial_embedding_bntd(state_ids)
        if not self.use_interlaced:
            h = self.temporal_module(x)
            return h.permute(0, 2, 1, 3).contiguous()

        h_btnd = x.permute(0, 2, 1, 3).contiguous()
        e_btnd = self.state_embedding_btnd(state_ids)
        # Run all but the final block, then run only the final block's temporal
        # part so the returned tensor matches what the final graph scorer sees.
        for block in self.st_blocks[:-1]:
            h_btnd, _ = block(h_btnd, state_ids, e_btnd)
        final = self.st_blocks[-1]
        h_bntd = h_btnd.permute(0, 2, 1, 3).contiguous()
        h_bntd = final.temporal_module(h_bntd)
        return h_bntd.permute(0, 2, 1, 3).contiguous()

    def state_embedding_btnd(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Raw current-state embedding e (B, T, N, D), before temporal mixing."""
        e = self.state_embedding(state_ids)                            # (B, N, T, D)
        return e.permute(0, 2, 1, 3).contiguous()                      # (B, T, N, D)

    def _select_graph_attn(self, block_attns: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        """Select the graph exposed as out["graph_attn"].

        graph_eval_layer = -1 selects the last non-None graph; a non-negative
        value selects that block index.
        """
        if not block_attns:
            return None
        layer = int(getattr(self.cfg, "graph_eval_layer", -1))
        if layer >= 0:
            if layer >= len(block_attns):
                raise ValueError(f"graph_eval_layer={layer} out of range for {len(block_attns)} ST blocks")
            return block_attns[layer]
        for attn in reversed(block_attns):
            if attn is not None:
                return attn
        return None

    def _oracle_attn(self, h_btnd: torch.Tensor, regimes: Optional[torch.Tensor]) -> torch.Tensor:
        if regimes is None or getattr(self, "oracle_regime_graphs", None) is None:
            raise RuntimeError("oracle_graph mode needs regimes and oracle_regime_graphs")
        b, t, n, _ = h_btnd.shape
        G = self.oracle_regime_graphs.to(h_btnd.device)                # (R, N, N)
        A = G[regimes.long()]                                          # (B, T, N, N)
        row_sum = A.sum(dim=-1, keepdim=True)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).view(1, 1, n, n)
        A = torch.where(row_sum > 1e-6, A, eye.expand_as(A))
        return A.unsqueeze(2).expand(b, t, self.cfg.num_edge_heads, n, n)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        """Rebuild ModelConfig from the checkpoint's saved cfg_dict, then load weights.

        Falls back to a caller-supplied ``cfg=`` for older checkpoints that do not
        store a cfg_dict.
        """
        import torch as _torch
        from utilities import ModelConfig as _MC
        ck = _torch.load(str(checkpoint_path), map_location=map_location or "cpu", weights_only=False)
        hp = ck.get("hyper_parameters", {})
        if "cfg" not in kwargs:
            cfg_dict = hp.get("cfg_dict")
            if cfg_dict is None:
                raise ValueError(
                    "checkpoint has no 'cfg_dict' (saved before the cfg-persistence fix); "
                    "pass cfg=ModelConfig(...) explicitly to load this one.")
            kwargs["cfg"] = _MC(**cfg_dict)
        # restore the saved init kwargs (lr, weight_decay, scheduler_t_max) where present
        for k in ("lr", "weight_decay", "scheduler_t_max"):
            if k in hp and k not in kwargs:
                kwargs[k] = hp[k]
        model = cls(**kwargs)
        model.load_state_dict(ck["state_dict"], strict=strict)
        return model

    def forward(self, state_ids: torch.Tensor,
                regimes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode state IDs into temporal and spatial representations.

        state_ids: (B, N, T). regimes: optional (B, T), required for oracle_graph
        mode. Returns a dict with 'temporal_repr' and 'spatial_repr' (both
        (B, T, N, D)), 'graph_attn' (the selected graph), and, in interlaced mode,
        'block_graph_attns' (one entry per block).
        """
        b, n, t = state_ids.shape
        if n != self.cfg.num_nodes:
            raise ValueError(f"Expected num_nodes={self.cfg.num_nodes}, got {n}")

        e_btnd = self.state_embedding_btnd(state_ids)                  # (B, T, N, D)

        if self.use_interlaced:
            h_btnd = self._initial_embedding_bntd(state_ids).permute(0, 2, 1, 3).contiguous()
            block_attns: List[Optional[torch.Tensor]] = []
            for block in self.st_blocks:
                h_btnd, attn = block(
                    h_btnd,
                    state_ids,
                    e_btnd,
                    regimes=regimes,
                    oracle_regime_graphs=getattr(self, "oracle_regime_graphs", None),
                )
                block_attns.append(attn)
            z = self.post_norm(h_btnd)
            selected_attn = self._select_graph_attn(block_attns)
            return {
                "temporal_repr": h_btnd,
                "spatial_repr": z,
                "graph_attn": selected_attn,
                "block_graph_attns": block_attns,
            }

        h_btnd = self.temporal_output(state_ids)                       # (B, T, N, D)

        if self.cfg.spatial_module_type == "oracle_graph":
            attn = self._oracle_attn(h_btnd, regimes)
            z = self.spatial_module(h_btnd, attn, e=e_btnd)
        elif self.graph_scorer is not None:
            attn = self.graph_scorer(h_btnd, state_ids, e=e_btnd)      # (B, T, H, N, N)
            z = self.spatial_module(h_btnd, attn, e=e_btnd)            # (B, T, N, D)
        else:
            attn = None
            z = self.spatial_module(h_btnd, None, e=e_btnd)            # identity passthrough

        z = self.post_norm(z)
        return {"temporal_repr": h_btnd, "spatial_repr": z, "graph_attn": attn}


# ---------------------------------------------------------------------------
# Next-state head
# ---------------------------------------------------------------------------

class NextStateHead(nn.Module):
    """Predict s_{i,t+1} from the representation at time t. (B,T,N,D) -> (B,N,T-1,K)."""

    def __init__(self, d_model: int, num_states: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h[:, :-1])                                  # (B, T-1, N, K)
        return logits.permute(0, 2, 1, 3).contiguous()                # (B, N, T-1, K)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DiscreteSTGraphLightningModule(pl.LightningModule):
    """Lightning module: backbone + next-state head trained on token sequences.

    Optimises next-state cross-entropy plus optional graph-shape regularisation,
    and logs graph-recovery, temporal-dynamics, and cell-complex diagnostics. When
    true_regime_graphs are supplied, recovery metrics against the ground-truth
    graphs are logged; they are also required for oracle_graph mode.
    """

    def __init__(
        self,
        cfg: "ModelConfig",  # noqa: F821
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        true_regime_graphs: Optional[torch.Tensor] = None,
        scheduler_t_max: Optional[int] = None,
    ) -> None:
        super().__init__()
        import dataclasses as _dc
        self.save_hyperparameters(ignore=["cfg", "true_regime_graphs"])
        # make checkpoints self-describing: store cfg as a plain dict in hparams
        self.hparams["cfg_dict"] = _dc.asdict(cfg)
        self.cfg = cfg
        self.backbone = DiscreteSTGraphBackbone(cfg)
        self.next_state_head = NextStateHead(cfg.d_model, cfg.num_states)
        self.lr = lr
        self.weight_decay = weight_decay
        # cosine horizon; should match trainer max_epochs. None -> resolved at
        # configure_optimizers from trainer.max_epochs (falls back to 100).
        self.scheduler_t_max = scheduler_t_max

        if true_regime_graphs is not None:
            self.register_buffer("true_regime_graphs", true_regime_graphs.float(), persistent=False)
        else:
            self.true_regime_graphs = None

        # for the oracle_graph diagnostic rung: hand the true graphs to the backbone
        if cfg.spatial_module_type == "oracle_graph":
            if true_regime_graphs is None:
                raise ValueError("oracle_graph mode requires true_regime_graphs")
            self.backbone.oracle_regime_graphs = true_regime_graphs.float()

    def forward(self, state_ids: torch.Tensor,
                regimes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.backbone(state_ids, regimes=regimes)
        out["next_state_logits"] = self.next_state_head(out["spatial_repr"])
        return out

    def _compute_graph_recovery_metrics(self, graph_attn, regimes) -> Dict[str, torch.Tensor]:
        """Pooled recovery of the inferred graph against the true regime graphs.

        Compares off-diagonal edges of the head-averaged attention to the true
        graph for each timestep's regime, returning MSE, Pearson correlation, and
        AUROC (edge ranking). Empty dict when no ground-truth graphs are set.
        """
        if self.true_regime_graphs is None:
            return {}
        attn = graph_attn.mean(dim=2)                                  # (B, T, N, N)
        true_graph = self.true_regime_graphs[regimes.long()]
        n = attn.size(-1)
        off = (~torch.eye(n, device=attn.device, dtype=torch.bool)).view(1, 1, n, n)
        attn_off = attn.masked_select(off).view(attn.size(0), attn.size(1), -1)
        true_off = true_graph.masked_select(off).view(true_graph.size(0), true_graph.size(1), -1)
        mse = F.mse_loss(attn_off, true_off)
        pc = attn_off - attn_off.mean(-1, keepdim=True)
        tc = true_off - true_off.mean(-1, keepdim=True)
        denom = pc.std(-1) * tc.std(-1)
        corr = ((pc * tc).mean(-1) / denom.clamp_min(1e-6)).mean()

        # AUROC measures whether the attention ranks true edges above non-edges,
        # independent of scale. Unlike correlation it does not penalise a dense
        # softmax for not matching the exact zeros of a sparse target graph.
        a = attn_off.reshape(-1)
        lbl = (true_off.reshape(-1) > 0)
        pos, neg = a[lbl], a[~lbl]
        if pos.numel() > 0 and neg.numel() > 0:
            cap = 20000
            if pos.numel() > cap:
                pos = pos[torch.randperm(pos.numel(), device=a.device)[:cap]]
            if neg.numel() > cap:
                neg = neg[torch.randperm(neg.numel(), device=a.device)[:cap]]
            allv = torch.cat([pos, neg])
            ranks = allv.argsort().argsort().float() + 1.0
            r_pos = ranks[:pos.numel()].sum()
            auroc = (r_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
        else:
            auroc = torch.tensor(float("nan"), device=a.device)
        return {"graph_mse": mse, "graph_corr": corr, "graph_auroc": auroc}

    def _compute_per_regime_recovery(self, graph_attn, regimes) -> Dict[str, torch.Tensor]:
        """Per-regime graph recovery, separating tracking from averaging.

        Does the recovered graph at the timesteps of regime r match regime r's true
        graph, or the regime-average graph? The pooled metrics scramble all
        (sample, time) edges into one ranking, so a scorer that learns the
        regime-average graph and never switches with the regime still posts a high
        pooled AUROC while giving little lift over a static graph. This metric
        separates the two cases.

        For each regime present in the batch, the recovered off-diagonal graph is
        averaged over that regime's timesteps and compared to every regime's true
        graph. Returns, averaged over present regimes:
          per_regime_corr         : corr(rec_i, true_i)                (matched)
          per_regime_auroc        : auroc(rec_i ranks true_i edges)
          per_regime_cross_corr   : mean_{i!=j} corr(rec_i, true_j)    (mismatched)
          per_regime_tracking_gap : matched - mismatched

        A large positive tracking_gap means the graph switches with the regime
        (tracking); a gap near zero means the recovered graph is the same regardless
        of which regime's timesteps built it (averaging).
        """
        if self.true_regime_graphs is None:
            return {}
        attn = graph_attn.mean(dim=2)                                  # (B, T, N, N)
        n = attn.size(-1)
        off = (~torch.eye(n, device=attn.device, dtype=torch.bool))    # (N, N)
        R = self.true_regime_graphs.size(0)
        regimes = regimes.to(attn.device).long()

        true_off_all = self.true_regime_graphs.to(attn.device)[:, off]  # (R, E)
        flat_attn = attn[:, :, off]                                     # (B, T, E)

        rec = {}
        for r in range(R):
            m = (regimes == r)                                         # (B, T)
            if m.any():
                rec[r] = flat_attn[m].mean(dim=0)                      # (E,)
        present = sorted(rec.keys())
        if not present:
            return {}

        def _corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            return (a * b).mean() / (a.std() * b.std()).clamp_min(1e-6)

        def _auroc(score, lbl):
            pos, neg = score[lbl], score[~lbl]
            if pos.numel() == 0 or neg.numel() == 0:
                return torch.tensor(float("nan"), device=score.device)
            allv = torch.cat([pos, neg])
            ranks = allv.argsort().argsort().float() + 1.0
            rp = ranks[:pos.numel()].sum()
            return (rp - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())

        diag_corrs, diag_aurocs, cross_corrs = [], [], []
        for i in present:
            diag_corrs.append(_corr(rec[i], true_off_all[i]))
            diag_aurocs.append(_auroc(rec[i], true_off_all[i] > 0))
            for j in present:
                if j != i:
                    cross_corrs.append(_corr(rec[i], true_off_all[j]))

        diag_corr = torch.stack(diag_corrs).mean()
        valid_auroc = [d for d in diag_aurocs if not torch.isnan(d)]
        diag_auroc = (torch.stack(valid_auroc).mean()
                      if valid_auroc else torch.tensor(float("nan"), device=attn.device))
        cross_corr = (torch.stack(cross_corrs).mean()
                      if cross_corrs else torch.tensor(0.0, device=attn.device))
        return {
            "per_regime_corr": diag_corr,
            "per_regime_auroc": diag_auroc,
            "per_regime_cross_corr": cross_corr,
            "per_regime_tracking_gap": diag_corr - cross_corr,
            "num_regimes_present": torch.tensor(float(len(present)), device=attn.device),
        }

    def _log_graph_mix(self, stage: str, step: bool) -> None:
        """Log residual/convex graph-mix alphas with explicit layer ids.

        All alpha logs live under graph_mix/... rather than train/... or val/...
        to avoid name collisions with ordinary training metrics. In interlaced
        mode each block gets graph_mix/{stage}/layer_{i:02d}/...
        """
        blocks = getattr(self.backbone, "st_blocks", None)
        if blocks is not None:
            for bi, block in enumerate(blocks):
                scorer = getattr(block, "graph_scorer", None)
                if scorer is None or not hasattr(scorer, "dynamic_residual_alpha"):
                    continue
                alpha = scorer.dynamic_residual_alpha().detach()
                prefix = f"graph_mix/{stage}/layer_{bi:02d}"
                self._log_epoch(f"{prefix}/alpha_mean", alpha.mean())
                if alpha.numel() > 1:
                    self._log_epoch(f"{prefix}/alpha_min", alpha.min())
                    self._log_epoch(f"{prefix}/alpha_max", alpha.max())
            return

        scorer = getattr(self.backbone, "graph_scorer", None)
        if scorer is not None and hasattr(scorer, "dynamic_residual_alpha"):
            alpha = scorer.dynamic_residual_alpha().detach()
            prefix = f"graph_mix/{stage}/layer_00"
            self._log_epoch(f"{prefix}/alpha_mean", alpha.mean())
            if alpha.numel() > 1:
                self._log_epoch(f"{prefix}/alpha_min", alpha.min())
                self._log_epoch(f"{prefix}/alpha_max", alpha.max())

    def _log_one_graph_recovery(
        self,
        attn: Optional[torch.Tensor],
        regimes: torch.Tensor,
        stage: str,
        step: bool,
        prefix: str,
    ) -> None:
        if attn is None or self.true_regime_graphs is None:
            return
        gm = self._compute_graph_recovery_metrics(attn, regimes.to(attn.device))
        if not gm:
            return
        self._log_epoch(f"{prefix}/{stage}/mse", gm["graph_mse"])
        self._log_epoch(f"{prefix}/{stage}/corr", gm["graph_corr"])
        if "graph_auroc" in gm:
            self._log_epoch(f"{prefix}/{stage}/auroc", gm["graph_auroc"])

    @staticmethod
    def _graph_temporal_metrics(graph_attn: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ground-truth-free graph dynamics from the attention tensor alone.

        Measures whether the inferred graph re-wires over time (dynamic) or is
        effectively time-invariant (collapsed to static). Works on real data.
        graph_attn: (B, T, H, N, N) or (B, T, N, N). Returns scalar tensors.
        """
        a = graph_attn.mean(dim=2) if graph_attn.ndim == 5 else graph_attn  # (B,T,N,N)
        B, T, N, _ = a.shape
        if T < 2:
            z = a.new_tensor(float("nan"))
            return {"delta_l1": z, "delta_frac": z, "self_similarity": z}
        off = (~torch.eye(N, dtype=torch.bool, device=a.device)).view(1, 1, N, N)
        cur, nxt = a[:, :-1], a[:, 1:]
        diff = (nxt - cur).abs()
        delta_l1 = diff[off.expand_as(diff)].mean()
        edge_mean = a[off.expand_as(a)].mean().clamp_min(1e-12)
        delta_frac = delta_l1 / edge_mean
        cur_f = cur.masked_fill(~off, 0.0).reshape(B, T - 1, N * N)
        nxt_f = nxt.masked_fill(~off, 0.0).reshape(B, T - 1, N * N)
        self_sim = F.cosine_similarity(cur_f, nxt_f, dim=-1).mean()
        return {"delta_l1": delta_l1, "delta_frac": delta_frac, "self_similarity": self_sim}

    @staticmethod
    def _cell_complex_metrics(
        graph_attn: torch.Tensor,
        thresholds: tuple = (0.02, 0.05, 0.1),
    ) -> Dict[str, torch.Tensor]:
        """2-cell diagnostics for lifting the dynamic graph to a (time-varying)
        cell complex.

        The system is adaptive, so the complex is expected to rewire over time;
        temporal persistence of triangles is not the goal (a static complex would
        mean the model failed to track regime change). The metrics are:

          1. triangles_per_step : mean per-timestep count of 3-cliques (2-cells),
             i.e. how much higher-order structure exists at each instant.

          2. triangle_robustness : per-timestep threshold robustness. Of the
             triangles present at the mid threshold, the fraction that also survive
             at the strict threshold. High values mean the 2-cells are real rather
             than artefacts of one edge cutoff (persistence in the filtration sense,
             at fixed time, orthogonal to temporal persistence).

          3. triangle_coherence : t->t+1 overlap of the triangle set (Jaccard of the
             per-timestep triangle indicator over node-triples). Distinguishes
             coherent rewiring (the complex evolves smoothly and tracks something)
             from random churn: high values mean structured adaptation, ~0 means
             noise flicker. This is the dynamic-complex analogue of self_similarity
             at the 2-cell (triangle) level rather than the 1-cell (edge) level.

        graph_attn: (B, T, H, N, N) or (B, T, N, N).
        """
        a = graph_attn.mean(dim=2) if graph_attn.ndim == 5 else graph_attn  # (B,T,N,N)
        B, T, N, _ = a.shape
        off = (~torch.eye(N, dtype=torch.bool, device=a.device)).view(1, 1, N, N)
        a_sym = torch.maximum(a, a.transpose(-1, -2))                       # undirected

        t_lo, t_mid, t_hi = thresholds                                      # mid is the reference threshold
        def tri_per_t(thresh):
            E = ((a_sym > thresh) & off).float()
            E3 = torch.matmul(torch.matmul(E, E), E)
            return torch.diagonal(E3, dim1=-2, dim2=-1).sum(-1) / 6.0       # (B,T)

        tri_mid = tri_per_t(t_mid)
        tri_count = tri_mid.mean()

        # (2) Threshold robustness at fixed time: fraction of mid-threshold
        # triangle mass that survives the strict (higher) threshold. Uses counts
        # as a cheap proxy for whether these are the same robust triangles.
        tri_hi = tri_per_t(t_hi)
        robustness = (tri_hi / tri_mid.clamp_min(1e-6)).clamp(max=1.0).mean()

        # (3) Temporal coherence of the triangle set (t -> t+1), at mid threshold.
        # A full triangle indicator over node-triples is expensive; instead use the
        # per-edge triangle-participation signature M = E * (E@E), which gives, for
        # each edge, how many triangles it is in. Cosine similarity of M across t is
        # a coherence proxy on the 2-cell structure rather than just the edges.
        E = ((a_sym > t_mid) & off).float()                                # (B,T,N,N)
        EE = torch.matmul(E, E)
        M = (E * EE).reshape(B, T, N * N)                                   # edge-triangle signature
        if T >= 2:
            cur, nxt = M[:, :-1], M[:, 1:]
            coherence = F.cosine_similarity(cur, nxt, dim=-1).mean()
        else:
            coherence = a.new_tensor(float("nan"))

        # (4) Clustering enrichment: observed triangles vs the count expected from
        # the same edge density under random (Erdos-Renyi) wiring. For density p,
        # E[triangles] = C(N, 3) * p^3, and enrichment = observed / expected.
        #   ~1 -> triangles are just a byproduct of edge density (no real
        #         higher-order structure beyond the 1-skeleton)
        #   >1 -> triangles enriched: the graph clusters into genuine 3-way
        #         structure (real 2-cells worth lifting; the cell-complex premise)
        #   <1 -> triangles suppressed below random (anti-clustering)
        deg_pairs = (N * (N - 1))                                           # ordered off-diag pairs
        p = E.sum(dim=(-1, -2)) / deg_pairs                                 # (B,T) edge density
        from math import comb
        expected_tri = (comb(N, 3)) * (p ** 3)                             # (B,T)
        enrichment = (tri_mid / expected_tri.clamp_min(1e-6)).mean()

        # (5) Euler characteristic chi = V - E + F (nodes - edges + triangles),
        # the alternating cell-count sum: a compact single-number topological
        # signature of the 2-skeleton. Tracked over time it shows how the complex's
        # overall shape shifts (more negative => edge/loop-dominated; toward V =>
        # filled-in / tree-like). Cheap given E and F are already computed.
        n_edges = (E.sum(dim=(-1, -2)) / 2.0)                              # (B,T) undirected
        euler_chi = (N - n_edges + tri_mid).mean()

        # (6) Node coverage: is the triangle structure broad across the asset
        # universe, or concentrated on a few nodes? Count alone cannot tell: 65
        # triangles could span 40 nodes (broad) or sit on 6 nodes every step
        # (narrow clique). Per-node triangle participation = diagonal of E^3
        # (each node's closed-triangle count, x2).
        E_mid = ((a_sym > t_mid) & off).float()                            # (B,T,N,N)
        node_tri = torch.diagonal(torch.matmul(torch.matmul(E_mid, E_mid), E_mid),
                                  dim1=-2, dim2=-1)                        # (B,T,N) ~2x tri per node
        in_tri = (node_tri > 0).float()                                    # (B,T,N) node in >=1 triangle
        # instantaneous: mean fraction of nodes in a triangle at a given t
        coverage_inst = in_tri.mean(dim=-1).mean()                         # scalar
        # cumulative: fraction of nodes in a triangle at any t in the window.
        # A large gap (cumulative >> instantaneous) means the structure moves
        # around the node set over time (broad, dynamic); cumulative ~=
        # instantaneous means the same nodes always carry it (narrow, fixed clique).
        coverage_cum = (in_tri.sum(dim=1) > 0).float().mean()              # union over t, then frac of nodes
        # concentration: Gini of per-node total triangle participation over the
        # window. 0 = perfectly even across nodes (broad); ->1 = all on a few
        # nodes (concentrated). Complements coverage: low Gini + high coverage =
        # broad topological structure.
        part = node_tri.sum(dim=1)                                         # (B,N) total participation per node
        part_sorted, _ = torch.sort(part, dim=-1)                          # ascending
        idx = torch.arange(1, N + 1, device=part.device, dtype=part.dtype)
        gini_num = (2.0 * idx - N - 1) * part_sorted
        gini = (gini_num.sum(dim=-1) / (N * part.sum(dim=-1).clamp_min(1e-6)))
        coverage_gini = gini.mean()

        return {
            "triangles_per_step": tri_count,
            "triangle_robustness": robustness,   # filtration-persistence at fixed t (want high)
            "triangle_coherence": coherence,     # coherent rewiring vs churn (want high, but <1)
            "triangle_enrichment": enrichment,   # observed/expected-from-density (>1 = real clustering)
            "euler_chi": euler_chi,              # V - E + F, compact topological signature
            "coverage_inst": coverage_inst,      # frac of nodes in a triangle at a given t
            "coverage_cum": coverage_cum,        # frac of nodes in a triangle at ANY t (want high = broad)
            "coverage_gini": coverage_gini,      # 0=even across nodes, 1=concentrated (want low = broad)
        }

    def _log_graph_diagnostics(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: str, step: bool) -> None:
        """Log graph entropy/recovery with explicit selected/all-layer names."""
        graph_attn = out.get("graph_attn", None)
        with torch.no_grad():
            # Selected graph: this is the graph exposed as out["graph_attn"].
            # In interlaced mode the selected layer is controlled by cfg.graph_eval_layer.
            if graph_attn is not None:
                entropy = _row_distribution_entropy(graph_attn).mean()
                self._log_epoch(f"graph_selected/{stage}/entropy", entropy)

                # temporal dynamics: is the graph re-wiring (dynamic) or static?
                # (no ground truth needed — works on real data)
                tdyn = self._graph_temporal_metrics(graph_attn)
                # 2-cell budget: persistent triangles = candidate cells for the
                # cell-complex lift. Logged on the SELECTED (eval-layer) graph.
                try:
                    tri = self._cell_complex_metrics(graph_attn)
                    for kk, vv in tri.items():
                        self._log_epoch(f"graph_cells/{stage}/{kk}", vv)
                except Exception:
                    pass
                self._log_epoch(f"graph_dynamics/{stage}/self_similarity", tdyn["self_similarity"])
                self._log_epoch(f"graph_dynamics/{stage}/delta_frac", tdyn["delta_frac"])
                self._log_epoch(f"graph_dynamics/{stage}/delta_l1", tdyn["delta_l1"])

                if "regimes" in batch and self.true_regime_graphs is not None:
                    self._log_one_graph_recovery(
                        graph_attn,
                        batch["regimes"],
                        stage,
                        step,
                        prefix="graph_selected",
                    )

                    # Backwards-compatible summary keys for existing notebook tables.
                    # These are for the selected graph only; layer-specific names below
                    # make it clear which interlaced block is being evaluated.
                    gm = self._compute_graph_recovery_metrics(graph_attn, batch["regimes"].to(graph_attn.device))
                    if gm:
                        self._log_epoch(f"{stage}/graph_mse", gm["graph_mse"])
                        self._log_epoch(f"{stage}/graph_corr", gm["graph_corr"])
                        if "graph_auroc" in gm:
                            self._log_epoch(f"{stage}/graph_auroc", gm["graph_auroc"])

                    # Per-regime recovery (tracking vs averaging). Off by default; flip
                    # log_per_regime_recovery=True in the config to enable.
                    if bool(getattr(self.cfg, "log_per_regime_recovery", False)):
                        pr = self._compute_per_regime_recovery(graph_attn, batch["regimes"])
                        for k, v in pr.items():
                            self._log_epoch(f"per_regime/{stage}/{k}", v)

            # Every interlaced layer gets its own diagnostics if requested.
            if bool(getattr(self.cfg, "graph_log_all_layers", True)) and "regimes" in batch:
                block_attns = out.get("block_graph_attns", None)
                if block_attns is not None:
                    for bi, attn_b in enumerate(block_attns):
                        if attn_b is None:
                            continue
                        entropy = _row_distribution_entropy(attn_b).mean()
                        layer_prefix = f"graph_layers/layer_{bi:02d}"
                        self._log_epoch(f"{layer_prefix}/{stage}/entropy", entropy)
                        self._log_one_graph_recovery(attn_b, batch["regimes"], stage, step, prefix=layer_prefix)

            self._log_graph_mix(stage, step)

    def _select_graph_for_regularisation(self, out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Return the graph attention tensor to regularise.

        graph_reg_layer = -1 selects the last non-None graph; a non-negative value
        selects a specific block. Single-stage path returns out["graph_attn"].
        """
        block_attns = out.get("block_graph_attns", None)
        if block_attns is None:
            return out.get("graph_attn", None)

        layer = int(getattr(self.cfg, "graph_reg_layer", -1))
        if layer == -2:                       # all non-None block graphs
            graphs = [a for a in block_attns if a is not None]
            return graphs if graphs else None
        if layer >= 0:
            if layer >= len(block_attns):
                return None
            return block_attns[layer]

        for attn in reversed(block_attns):
            if attn is not None:
                return attn
        return out.get("graph_attn", None)

    def _graph_reg_warmup_scale(self) -> float:
        warmup = int(getattr(self.cfg, "graph_reg_warmup_epochs", 0) or 0)
        if warmup <= 0:
            return 1.0
        # current_epoch is 0-indexed; use +1 so the first epoch gets non-zero
        # pressure but still ramps gently.
        return float(min(1.0, max(0.0, (self.current_epoch + 1) / warmup)))

    def _compute_graph_regularisation(
        self,
        out: Dict[str, torch.Tensor],
        stage: str,
        step: bool,
    ) -> torch.Tensor:
        """Optional graph-shape regularisation.

        Returns a scalar; all terms are zero unless their coefficients are set.
        Logged under graph_reg/{stage}/... and added to the training loss only.
        """
        attn = self._select_graph_for_regularisation(out)
        device = out["next_state_logits"].device
        zero = torch.zeros((), device=device)
        if attn is None:
            return zero
        # all-layers mode: attn is a list of (B,T,H,N,N) graphs -> average the reg
        if isinstance(attn, list):
            regs = [self._reg_one_graph(a, stage, step, log=(i == 0))
                    for i, a in enumerate(attn)]
            return torch.stack(regs).mean() if regs else zero

        return self._reg_one_graph(attn, stage, step, log=True)

    def _reg_one_graph(self, attn, stage, step, log=True):
        device = attn.device
        zero = torch.zeros((), device=device)
        entropy_coef = float(getattr(self.cfg, "graph_entropy_reg", 0.0) or 0.0)
        target_entropy_coef = float(getattr(self.cfg, "graph_target_entropy_reg", 0.0) or 0.0)
        smooth_coef = float(getattr(self.cfg, "graph_temporal_smooth_reg", 0.0) or 0.0)
        if entropy_coef == 0.0 and target_entropy_coef == 0.0 and smooth_coef == 0.0:
            return zero

        warmup_scale = self._graph_reg_warmup_scale()
        # Normalise each row to a distribution before the entropy so the measure is a
        # bounded Shannon entropy (<= log N) for every activation, including 'gated'
        # with gate_row_normalise=False (whose rows do not sum to 1). Without this the
        # target-entropy loss compares an unbounded (~N/e) quantity against a
        # row-normalised target and drives every gate toward zero, collapsing the row
        # mass and the spatial messages. The gain in the actual message passing is
        # untouched; only this measurement is normalised.
        row_entropy = _row_distribution_entropy(attn)  # (B, T, H, N)
        entropy = row_entropy.mean()

        reg = zero
        if entropy_coef != 0.0:
            ent_loss = entropy
            reg = reg + entropy_coef * ent_loss
            if log:
                self._log_epoch(f"graph_reg/{stage}/entropy_loss", ent_loss.detach())

        if target_entropy_coef != 0.0:
            target = getattr(self.cfg, "graph_target_entropy", None)
            if target is None:
                # No explicit target: fall back to the mean true-graph entropy
                # when available (synthetic data), else the current entropy.
                if self.true_regime_graphs is not None:
                    G = self.true_regime_graphs.to(device).float()
                    G = G / G.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    target_tensor = -(G.clamp_min(1e-12) * G.clamp_min(1e-12).log()).sum(dim=-1).mean()
                else:
                    target_tensor = entropy.detach()
            else:
                target_tensor = torch.as_tensor(float(target), device=device, dtype=entropy.dtype)
            target_loss = (entropy - target_tensor.detach()).pow(2)
            reg = reg + target_entropy_coef * target_loss
            if log:
                self.log(f"graph_reg/{stage}/target_entropy", target_tensor.detach(), on_step=False, on_epoch=True)
            if log:
                self._log_epoch(f"graph_reg/{stage}/target_entropy_loss", target_loss.detach())

        if smooth_coef != 0.0 and attn.size(1) > 1:
            smooth_loss = (attn[:, 1:] - attn[:, :-1]).pow(2).mean()
            reg = reg + smooth_coef * smooth_loss
            if log:
                self._log_epoch(f"graph_reg/{stage}/temporal_smooth_loss", smooth_loss.detach())

        reg = reg * warmup_scale
        if log:
            self._log_epoch(f"graph_reg/{stage}/warmup_scale", torch.as_tensor(warmup_scale, device=device))
        if log:
            self._log_epoch(f"graph_reg/{stage}/loss", reg.detach())
        return reg

    def _log_epoch(self, name, value, prog_bar=False):
        """Log a single epoch-aggregated series (no per-step duplicate).

        Used for all graph/cell diagnostics: on_step=False gives one W&B series per
        key, i.e. one chart per metric.
        """
        self.log(name, value, prog_bar=prog_bar, on_step=False, on_epoch=True)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        state_ids = batch["state_ids"].long()
        regimes = batch.get("regimes", None)
        out = self(state_ids, regimes=regimes)
        logits = out["next_state_logits"]                              # (B, N, T-1, K)
        target = state_ids[:, :, 1:]
        pred_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean()

        # Training has both step and epoch curves; validation/test are epoch-only.
        # log_per_step=False (default) suppresses the intra-epoch curves entirely,
        # since step curves duplicate the epoch keys with added noise.
        step = (stage == "train") and bool(getattr(self.cfg, "log_per_step", False))
        graph_reg = self._compute_graph_regularisation(out, stage, step)
        loss = pred_loss + graph_reg if stage == "train" else pred_loss

        self.log(f"{stage}/pred_loss", pred_loss, prog_bar=False, on_step=step, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=step, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=step, on_epoch=True)

        # Predictive-distribution diagnostics: is low argmax accuracy real failure,
        # or intrinsic multi-modality? perplexity = exp(H[softmax]) is the effective
        # number of states the model spreads over; top-k is whether the truth is in
        # the shortlist. Judge cross-asset / propagation-delay gains on these, not
        # on argmax accuracy, when the next-state distribution is multi-modal.
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)                        # (B,N,T-1) nats
            pred_perplexity = ent.mean().exp()
            maxk = min(5, logits.size(-1))
            tk = logits.topk(maxk, dim=-1).indices                   # (B,N,T-1,maxk)
            tgt = target.unsqueeze(-1)
            hits = (tk == tgt)
            top3 = hits[..., :min(3, maxk)].any(-1).float().mean()
            top5 = hits[..., :maxk].any(-1).float().mean()
        self._log_epoch(f"predictive/{stage}/perplexity", pred_perplexity)
        self._log_epoch(f"predictive/{stage}/top3_acc", top3)
        self._log_epoch(f"predictive/{stage}/top5_acc", top5)

        self._log_graph_diagnostics(out, batch, stage, step)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: D401
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # match the cosine horizon to the actual training length so the LR anneals.
        t_max = self.scheduler_t_max
        if t_max is None:
            t_max = getattr(self.trainer, "max_epochs", None) or 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

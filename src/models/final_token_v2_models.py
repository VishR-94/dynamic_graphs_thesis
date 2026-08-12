from __future__ import annotations

"""Final token-space and BaseDyGraph-V2 comparison models.

The module contains only project-side adapters around already established
architectures:

* the selected one-block ModernTCN graph model is reused from
  :mod:`src.models.modern_tcn_graph_round2_token`;
* the selected three-block dense causal Transformer reuses the exact ST blocks
  from :mod:`src.models.dense_transformer_depth_sweep`;
* Dimitri's vendored BaseDyGraph-V2 backbone is executed unchanged through an
  architecture-only importer, avoiding the optional Lightning dependency;
* dense token heads support both the hybrid five-internal/final-60 objective
  and the full all-origins/all-60 objective; the continuous V2 head predicts
  the five dissertation horizons from every causal origin.

Graph orientation is always ``A[target, source]``.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from src.models.dense_transformer_depth_sweep import (
    DenseTransformerDepthConfig,
    DenseTransformerSTBlock,
    DenseTransformerSTBlockOutput,
)
from src.models.dynamic_graph.future_predictor import FutureTransformerLayer


GRAPH_ORIENTATION = "A[target, source]"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMITRI_SOURCE_ROOT = PROJECT_ROOT / "external" / "DimitriBaseDyGraphV2" / "src"

# Hashes already used by the established V2 adapter.  Repeating them here keeps
# this architecture-only loader independent from Lightning and makes the
# source contract explicit.
DIMITRI_SOURCE_HASHES: dict[str, str] = {
    "model.py": "b99256db74b84f57513b12715a9ed1f4fc735202bcf933482e3b66ac9cf119d5",
    "modules.py": "1bd31701b300f6f805dfc53530c66eedd9265620093223d3302e3e74409b51ff",
    "utilities.py": "cfe849e2963386ddaab15ad7c89e7df92fc957f22c15f45ea40c89ec4c82f40a",
}

# Defaults copied from the supplied Dimitri notebook.  Context length is the
# only intentional change: temporal_context_window=60 rather than 180.
DIMITRI_NOTEBOOK_DEFAULTS: dict[str, Any] = {
    "num_states": 1024,
    "num_nodes": 93,
    "d_model": 96,
    "nhead": 4,
    "num_temporal_layers": 1,
    "num_spatial_layers": 1,
    "dropout": 0.0,
    "ff_mult": 1,
    "max_seq_len": 512,
    "num_edge_heads": 1,
    "graph_hidden_dim": 96,
    "spatial_dropout": 0.1,
    "use_node_embedding": True,
    "use_state_pair_bias": False,
    "add_self_loops": False,
    "symmetric_graph": False,
    "predict_next_state": True,
    "temporal_module_type": "transformer",
    "temporal_context_window": 60,
    "spatial_module_type": "dual_fusion",
    "spatial_value": "concat",
    "scorer_value": "concat",
    "graph_activation": "softmax",
    "graph_activation_per_block": ["softmax", "softmax", "sparsemax"],
    "num_edge_heads_per_block": [6, 6, 1],
    "graph_hidden_dim_per_block": [192, 192, 96],
    "log_per_step": False,
    "gate_tau": 0.5,
    "gate_row_normalise": True,
    "dynamic_residual_gate": "per_head",
    "dynamic_residual_init": 0.75,
    "dynamic_residual_learnable": True,
    "dynamic_residual_mix": "strict_convex",
    "interlaced_st_blocks": True,
    "num_st_blocks": 4,
    "first_spatial_module_type": None,
    "st_block_post_norm": True,
    "prop_window_size": 4,
    "fusion_window_size": 32,
    "fusion_fast_window": 4,
    "spatial_use_base": True,
    "graph_prior_level": "none",
    "graph_prior_scale": 4.0,
    "graph_prior_learnable": True,
    "prop_lag_aggregation": "softmax",
    "graph_eval_layer": -1,
    "graph_log_all_layers": True,
    "graph_reg_layer": -1,
    "graph_reg_warmup_epochs": 0,
    "graph_entropy_reg": 0.0,
    "graph_target_entropy": 1.8,
    "graph_target_entropy_reg": 0.0,
    "graph_temporal_smooth_reg": 0.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArchitectureOnlyLightningModule(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "The architecture-only Dimitri V2 loader cannot instantiate the "
            "Lightning training wrapper."
        )


@dataclass(frozen=True)
class DimitriArchitectureModules:
    utilities: ModuleType
    modules: ModuleType
    model: ModuleType


@lru_cache(maxsize=1)
def load_dimitri_v2_architecture() -> DimitriArchitectureModules:
    """Execute the unchanged V2 source without importing real Lightning."""
    for filename, expected in DIMITRI_SOURCE_HASHES.items():
        path = DIMITRI_SOURCE_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256(path)
        if observed != expected:
            raise AssertionError(
                f"Dimitri V2 source hash differs for {path}: expected "
                f"{expected}; observed {observed}."
            )

    namespace = "_dimitri_v2_arch_" + hashlib.sha256(
        str(DIMITRI_SOURCE_ROOT).encode("utf-8")
    ).hexdigest()[:12]
    loaded_names: list[str] = []

    def load_file(filename: str, suffix: str) -> ModuleType:
        module_name = f"{namespace}_{suffix}"
        spec = importlib.util.spec_from_file_location(
            module_name, DIMITRI_SOURCE_ROOT / filename
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {filename}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loaded_names.append(module_name)
        spec.loader.exec_module(module)
        return module

    missing = object()
    names = ("utilities", "modules", "lightning", "lightning.pytorch")
    previous = {name: sys.modules.get(name, missing) for name in names}
    lightning = ModuleType("lightning")
    lightning.__path__ = []  # type: ignore[attr-defined]
    lightning_pytorch = ModuleType("lightning.pytorch")
    lightning_pytorch.LightningModule = _ArchitectureOnlyLightningModule
    lightning.pytorch = lightning_pytorch  # type: ignore[attr-defined]

    try:
        utilities = load_file("utilities.py", "utilities")
        sys.modules["utilities"] = utilities
        modules = load_file("modules.py", "modules")
        sys.modules["modules"] = modules
        sys.modules["lightning"] = lightning
        sys.modules["lightning.pytorch"] = lightning_pytorch
        model = load_file("model.py", "model")
    except Exception:
        for name in loaded_names:
            sys.modules.pop(name, None)
        raise
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value  # type: ignore[assignment]

    return DimitriArchitectureModules(
        utilities=utilities,
        modules=modules,
        model=model,
    )


def make_dimitri_v2_config(
    *,
    num_nodes: int,
    context_length: int = 60,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    values = dict(DIMITRI_NOTEBOOK_DEFAULTS)
    values["num_nodes"] = int(num_nodes)
    values["temporal_context_window"] = int(context_length)
    if overrides:
        unknown = sorted(set(overrides).difference(values))
        if unknown:
            raise KeyError(f"Unknown Dimitri V2 overrides: {unknown}")
        values.update(dict(overrides))
    imported = load_dimitri_v2_architecture()
    return imported.utilities.ModelConfig(**values)


class DenseOriginStructuredTokenPredictor(nn.Module):
    """Shared 60-step structured-parallel head for one causal origin."""

    def __init__(
        self,
        *,
        d_model: int,
        prediction_length: int = 60,
        vocabulary_size: int = 1024,
        num_layers: int = 1,
        num_heads: int = 4,
        feedforward_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by predictor heads.")
        self.d_model = int(d_model)
        self.prediction_length = int(prediction_length)
        self.vocabulary_size = int(vocabulary_size)
        self.future_position_embedding = nn.Embedding(
            self.prediction_length, self.d_model
        )
        nn.init.normal_(self.future_position_embedding.weight, std=0.02)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList(
            [
                FutureTransformerLayer(
                    d_model=self.d_model,
                    num_heads=int(num_heads),
                    feedforward_multiplier=int(feedforward_multiplier),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.classifier = nn.Linear(self.d_model, self.vocabulary_size)

    @staticmethod
    def _flatten_nodes(values: Tensor) -> Tensor:
        batch, length, nodes, hidden = values.shape
        return (
            values.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch * nodes, length, hidden)
        )

    def forward_origin(
        self,
        hidden: Tensor,
        origin_index: int,
        *,
        future_position_indices: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        """Predict selected future positions from one causal origin.

        ``future_position_indices`` uses zero-based indices into the shared
        future-query bank.  ``None`` evaluates the complete ordered future
        path.  Dense auxiliary supervision passes only the five dissertation
        positions, while final-origin forecasting passes ``None`` and therefore
        retains the complete 60-position path required by the frozen decoder.
        """
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,T,N,D].")
        batch, steps, nodes, width = map(int, hidden.shape)
        origin = int(origin_index)
        if width != self.d_model or not 0 <= origin < steps:
            raise ValueError("Origin or hidden width is invalid.")

        if future_position_indices is None:
            position_indices = torch.arange(
                self.prediction_length,
                device=hidden.device,
                dtype=torch.long,
            )
        else:
            position_indices = torch.as_tensor(
                future_position_indices,
                device=hidden.device,
                dtype=torch.long,
            ).reshape(-1)
            if int(position_indices.numel()) == 0:
                raise ValueError("At least one future position is required.")
            if (
                int(position_indices.min().item()) < 0
                or int(position_indices.max().item()) >= self.prediction_length
            ):
                raise ValueError("Future position index is out of range.")
            if not bool(
                torch.all(position_indices[1:] > position_indices[:-1]).item()
            ):
                raise ValueError(
                    "Future position indices must be unique and increasing."
                )

        memory_btnd = hidden[:, : origin + 1]
        memory = self._flatten_nodes(memory_btnd)
        summary = hidden[:, origin].reshape(batch * nodes, self.d_model)
        positions = self.future_position_embedding(position_indices)
        future = self.input_norm(summary[:, None] + positions[None])
        for layer in self.layers:
            future = layer(future, memory, self_attention_mask=None)
        logits = self.classifier(future)
        selected_steps = int(position_indices.numel())
        return (
            logits.reshape(batch, nodes, selected_steps, self.vocabulary_size)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward_origins(
        self,
        hidden: Tensor,
        origin_indices: Sequence[int] | Tensor,
        *,
        future_position_indices: Sequence[int] | Tensor | None = None,
    ) -> Tensor:
        """Vectorise structured-parallel prediction over several origins.

        Parameters
        ----------
        hidden:
            Causal backbone states with shape ``[B,T,N,D]``.
        origin_indices:
            Unique, increasing zero-based causal origins.  Every origin sees
            only ``hidden[:, :origin+1]`` through an explicit cross-attention
            padding mask.
        future_position_indices:
            Optional selected future-query positions.  ``None`` evaluates all
            ``prediction_length`` positions.

        Returns
        -------
        Tensor
            Logits with shape ``[B,K,P,N,V]`` where ``K`` is the number of
            origins and ``P`` is the number of requested future positions.

        This method is mathematically equivalent to calling
        :meth:`forward_origin` independently for each origin, but it batches
        those calls to reduce Python and CUDA-kernel overhead.  It is used by
        the full 60-origin x 60-position dense-token experiment.
        """
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [B,T,N,D].")
        batch, steps, nodes, width = map(int, hidden.shape)
        if width != self.d_model:
            raise ValueError("Hidden width differs from predictor d_model.")

        origins = torch.as_tensor(
            origin_indices,
            device=hidden.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(origins.numel()) == 0:
            raise ValueError("At least one origin is required.")
        if int(origins.min().item()) < 0 or int(origins.max().item()) >= steps:
            raise ValueError("Origin index is out of range.")
        if int(origins.numel()) > 1 and not bool(
            torch.all(origins[1:] > origins[:-1]).item()
        ):
            raise ValueError("Origin indices must be unique and increasing.")

        if future_position_indices is None:
            positions_index = torch.arange(
                self.prediction_length,
                device=hidden.device,
                dtype=torch.long,
            )
        else:
            positions_index = torch.as_tensor(
                future_position_indices,
                device=hidden.device,
                dtype=torch.long,
            ).reshape(-1)
            if int(positions_index.numel()) == 0:
                raise ValueError("At least one future position is required.")
            if (
                int(positions_index.min().item()) < 0
                or int(positions_index.max().item()) >= self.prediction_length
            ):
                raise ValueError("Future position index is out of range.")
            if int(positions_index.numel()) > 1 and not bool(
                torch.all(positions_index[1:] > positions_index[:-1]).item()
            ):
                raise ValueError(
                    "Future position indices must be unique and increasing."
                )

        origin_count = int(origins.numel())
        selected_steps = int(positions_index.numel())

        # Repeat the complete causal-state sequence for each requested origin,
        # then mask keys after that origin.  Since the backbone states are
        # themselves causal, this exactly reproduces the prefix memory used by
        # ``forward_origin`` without zero-padding or future leakage.
        memory = (
            hidden[:, None]
            .expand(batch, origin_count, steps, nodes, width)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(batch * origin_count * nodes, steps, width)
        )
        time_index = torch.arange(steps, device=hidden.device)
        padding_by_origin = time_index[None, :] > origins[:, None]
        context_padding_mask = (
            padding_by_origin[None, :, None, :]
            .expand(batch, origin_count, nodes, steps)
            .reshape(batch * origin_count * nodes, steps)
        )

        summary = (
            hidden.index_select(1, origins)
            .reshape(batch * origin_count * nodes, width)
        )
        positions = self.future_position_embedding(positions_index)
        future = self.input_norm(summary[:, None] + positions[None])
        for layer in self.layers:
            future = layer(
                future,
                memory,
                self_attention_mask=None,
                context_key_padding_mask=context_padding_mask,
            )
        logits = self.classifier(future)
        return (
            logits.reshape(
                batch,
                origin_count,
                nodes,
                selected_steps,
                self.vocabulary_size,
            )
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )


@dataclass
class DenseTokenBackboneOutput:
    hidden: Tensor  # [B,T,N,D]
    selected_graphs: tuple[Tensor, ...]  # each [B,T,G,N,N]
    dynamic_graphs: tuple[Tensor | None, ...]
    base_graphs: tuple[Tensor | None, ...]  # each [1,G,N,N]
    slow_graphs: tuple[Tensor | None, ...]
    alphas: tuple[Tensor | None, ...]
    betas: tuple[Tensor | None, ...]


class DenseTransformerTokenForecaster(nn.Module):
    """Token counterpart of the winning D64, three-block dense Transformer."""

    def __init__(
        self,
        *,
        num_nodes: int,
        context_length: int = 60,
        prediction_length: int = 60,
        vocabulary_size: int = 1024,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.vocabulary_size = int(vocabulary_size)
        self.config = DenseTransformerDepthConfig(
            num_nodes=self.num_nodes,
            context_length=self.context_length,
            horizons=(1, 5, 15, 30, 60),
            input_channels=("s1",),
            target_channel="s1",
            num_st_blocks=3,
            d_model=64,
            transformer_num_layers=1,
            transformer_num_heads=4,
            transformer_feedforward_multiplier=2,
            transformer_dropout=0.0,
            position_embedding=False,
            graph_heads_per_block=(1, 1, 1),
            graph_hidden_dims_per_block=(64, 64, 64),
            graph_activations_per_block=("softmax", "softmax", "sparsemax"),
            graph_initial_alpha=0.5,
            spatial_initial_beta=0.5,
            spatial_feedforward_multiplier=2,
            spatial_dropout=0.0,
        )
        self.state_embedding = nn.Embedding(self.vocabulary_size, 64)
        self.blocks = nn.ModuleList(
            [DenseTransformerSTBlock(config=self.config, block_index=index) for index in range(3)]
        )
        self.future_predictor = DenseOriginStructuredTokenPredictor(
            d_model=64,
            prediction_length=self.prediction_length,
            vocabulary_size=self.vocabulary_size,
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        )

    def graph_parameter_ids(self) -> set[int]:
        return {
            id(parameter)
            for block in self.blocks
            for parameter in block.graph_learner.parameters()
            if parameter.requires_grad
        }

    def forward_backbone(self, context_s1: Tensor) -> DenseTokenBackboneOutput:
        if tuple(context_s1.shape[1:]) != (
            self.context_length,
            self.num_nodes,
        ):
            raise ValueError("context_s1 must have shape [B,60,N].")
        state = self.state_embedding(context_s1.long())
        hidden = state
        outputs: list[DenseTransformerSTBlockOutput] = []
        for block in self.blocks:
            result = block(hidden, state)
            outputs.append(result)
            hidden = result.fused_hidden
        return DenseTokenBackboneOutput(
            hidden=hidden,
            selected_graphs=tuple(value.graph.selected for value in outputs),
            dynamic_graphs=tuple(value.graph.dynamic for value in outputs),
            base_graphs=tuple(value.graph.base for value in outputs),
            slow_graphs=tuple(None for _ in outputs),
            alphas=tuple(value.graph.alpha for value in outputs),
            betas=tuple(value.beta for value in outputs),
        )

    def forward_final(self, context_s1: Tensor) -> tuple[Tensor, DenseTokenBackboneOutput]:
        backbone = self.forward_backbone(context_s1)
        logits = self.future_predictor.forward_origin(
            backbone.hidden, self.context_length - 1
        )
        return logits, backbone



def _window_mean(values: Tensor, size: int) -> Tensor:
    if int(size) <= 1:
        return values
    batch, steps = int(values.shape[0]), int(values.shape[1])
    padding = values.new_zeros(batch, int(size) - 1, *values.shape[2:])
    padded = torch.cat([padding, values], dim=1)
    cumulative = padded.cumsum(dim=1)
    upper = cumulative[:, int(size) - 1 :]
    lower = torch.cat(
        [cumulative.new_zeros(batch, 1, *values.shape[2:]), cumulative[:, :-1]],
        dim=1,
    )[:, :steps]
    counts = torch.arange(1, steps + 1, device=values.device).clamp_max(int(size))
    return (upper - lower) / counts.view(1, steps, 1, 1, 1)


def _v2_component_diagnostics(
    scorer: nn.Module,
    temporal: Tensor,
    state_ids: Tensor,
    state_embedding: Tensor,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Return fast-dynamic, base-only and slow graphs for diagnostics."""
    if not hasattr(scorer, "q_proj") or not hasattr(scorer, "_normalise"):
        return None, None, None
    scorer_value = str(getattr(scorer, "scorer_value", "hidden"))
    if scorer_value == "state_embedding":
        values = state_embedding
    elif scorer_value == "concat":
        values = torch.cat([temporal, state_embedding], dim=-1)
    else:
        values = temporal
    batch, steps, nodes, _ = values.shape
    heads = int(getattr(scorer, "num_heads"))
    head_dim = int(getattr(scorer, "head_dim"))
    q = scorer.q_proj(values).view(batch, steps, nodes, heads, head_dim).permute(0, 1, 3, 2, 4)
    k = scorer.k_proj(values).view(batch, steps, nodes, heads, head_dim).permute(0, 1, 3, 2, 4)
    logits = torch.einsum("bthid,bthjd->bthij", q, k) / math.sqrt(head_dim)
    if bool(getattr(scorer.cfg, "symmetric_graph", False)):
        logits = 0.5 * (logits + logits.transpose(-1, -2))
    per_step = scorer._normalise(logits)
    dynamic = per_step
    slow = None
    if hasattr(scorer, "w_fast"):
        dynamic = _window_mean(per_step, int(scorer.w_fast))
        slow = _window_mean(per_step, int(scorer.w_slow))
    base = None
    base_logits = getattr(scorer, "base_graph", None)
    if base_logits is not None:
        base = scorer._normalise(base_logits.view(1, 1, heads, nodes, nodes)).squeeze(1)
        if slow is not None:
            slow = scorer._normalise(
                torch.log(slow.clamp_min(1.0e-9))
                + base_logits.view(1, 1, heads, nodes, nodes)
            )
    return dynamic, base, slow


class _DimitriV2ManualBackbone(nn.Module):
    """Manual exact forward exposing per-layer graph components."""

    def __init__(self, *, num_nodes: int, context_length: int, continuous_channels: int | None) -> None:
        super().__init__()
        imported = load_dimitri_v2_architecture()
        self.cfg = make_dimitri_v2_config(
            num_nodes=num_nodes,
            context_length=context_length,
        )
        self.reference = imported.model.DiscreteSTGraphBackbone(self.cfg)
        self.continuous_channels = continuous_channels
        if continuous_channels is not None:
            self.reference.state_embedding = nn.Identity()
            self.input_projection = nn.Linear(int(continuous_channels), int(self.cfg.d_model))
        else:
            self.input_projection = None

    def _initial_token(self, state_ids_bnt: Tensor) -> tuple[Tensor, Tensor]:
        e_btnd = self.reference.state_embedding_btnd(state_ids_bnt)
        hidden = self.reference._initial_embedding_bntd(state_ids_bnt).permute(0, 2, 1, 3).contiguous()
        return hidden, e_btnd

    def _initial_continuous(self, values_bntc: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.input_projection is None:
            raise RuntimeError("Continuous projection is missing.")
        batch, nodes, steps, _ = values_bntc.shape
        e_bntd = self.input_projection(values_bntc)
        e_btnd = e_bntd.permute(0, 2, 1, 3).contiguous()
        hidden_bntd = e_bntd
        if self.reference.node_embedding is not None:
            node_ids = torch.arange(nodes, device=values_bntc.device)
            hidden_bntd = hidden_bntd + self.reference.node_embedding(node_ids).view(
                1, nodes, 1, int(self.cfg.d_model)
            )
        hidden = self.reference.pre_norm(hidden_bntd).permute(0, 2, 1, 3).contiguous()
        dummy = torch.zeros(batch, nodes, steps, dtype=torch.long, device=values_bntc.device)
        return hidden, e_btnd, dummy

    def forward_token(
        self, context_s1_btn: Tensor, *, include_components: bool = False
    ) -> DenseTokenBackboneOutput:
        state_ids = context_s1_btn.permute(0, 2, 1).contiguous().long()
        hidden, state_embedding = self._initial_token(state_ids)
        return self._run_blocks(
            hidden, state_ids, state_embedding, include_components=include_components
        )

    def forward_continuous(
        self, values_btnc: Tensor, *, include_components: bool = False
    ) -> DenseTokenBackboneOutput:
        values_bntc = values_btnc.permute(0, 2, 1, 3).contiguous().float()
        hidden, state_embedding, dummy = self._initial_continuous(values_bntc)
        return self._run_blocks(
            hidden, dummy, state_embedding, include_components=include_components
        )

    def _run_blocks(
        self,
        hidden: Tensor,
        state_ids: Tensor,
        state_embedding: Tensor,
        *,
        include_components: bool,
    ) -> DenseTokenBackboneOutput:
        selected: list[Tensor] = []
        dynamic: list[Tensor | None] = []
        base: list[Tensor | None] = []
        slow: list[Tensor | None] = []
        alphas: list[Tensor | None] = []
        for block in self.reference.st_blocks:
            temporal = block.temporal_module(
                hidden.permute(0, 2, 1, 3).contiguous()
            ).permute(0, 2, 1, 3).contiguous()
            scorer = block.graph_scorer
            attention = scorer(temporal, state_ids, e=state_embedding)
            if include_components:
                dyn, base_graph, slow_graph = _v2_component_diagnostics(
                    scorer, temporal, state_ids, state_embedding
                )
            else:
                dyn = base_graph = slow_graph = None
            hidden = block.spatial_module(temporal, attention, e=state_embedding)
            hidden = block.post_norm(hidden)
            selected.append(attention)
            dynamic.append(dyn)
            base.append(base_graph)
            slow.append(slow_graph)
            alpha_fn = getattr(scorer, "dynamic_residual_alpha", None)
            alphas.append(None if alpha_fn is None else alpha_fn())
        hidden = self.reference.post_norm(hidden)
        return DenseTokenBackboneOutput(
            hidden=hidden,
            selected_graphs=tuple(selected),
            dynamic_graphs=tuple(dynamic),
            base_graphs=tuple(base),
            slow_graphs=tuple(slow),
            alphas=tuple(alphas),
            betas=tuple(None for _ in selected),
        )


class DimitriV2DenseTokenForecaster(nn.Module):
    def __init__(self, *, num_nodes: int, context_length: int = 60, prediction_length: int = 60) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.vocabulary_size = 1024
        self.backbone = _DimitriV2ManualBackbone(
            num_nodes=self.num_nodes,
            context_length=self.context_length,
            continuous_channels=None,
        )
        self.future_predictor = DenseOriginStructuredTokenPredictor(
            d_model=96,
            prediction_length=self.prediction_length,
            vocabulary_size=1024,
            num_layers=1,
            num_heads=4,
            feedforward_multiplier=2,
            dropout=0.0,
        )

    def graph_parameter_ids(self) -> set[int]:
        values: set[int] = set()
        for block in self.backbone.reference.st_blocks:
            scorer = getattr(block, "graph_scorer", None)
            if scorer is not None:
                values.update(id(p) for p in scorer.parameters() if p.requires_grad)
        return values

    def forward_backbone(
        self, context_s1: Tensor, *, include_components: bool = False
    ) -> DenseTokenBackboneOutput:
        return self.backbone.forward_token(
            context_s1, include_components=include_components
        )

    def forward_final(
        self, context_s1: Tensor, *, include_components: bool = False
    ) -> tuple[Tensor, DenseTokenBackboneOutput]:
        output = self.forward_backbone(
            context_s1, include_components=include_components
        )
        logits = self.future_predictor.forward_origin(output.hidden, self.context_length - 1)
        return logits, output


@dataclass
class DimitriV2DenseContinuousOutput:
    predictions: Tensor  # [B,T,H,N,1]
    backbone: DenseTokenBackboneOutput

    def final_predictions(self) -> Tensor:
        return self.predictions[:, -1].contiguous()


class DimitriV2DenseContinuousForecaster(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        context_length: int = 60,
        horizons: Sequence[int] = (1, 5, 15, 30, 60),
        input_channels: int = 5,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.context_length = int(context_length)
        self.horizons = tuple(int(value) for value in horizons)
        self.backbone = _DimitriV2ManualBackbone(
            num_nodes=self.num_nodes,
            context_length=self.context_length,
            continuous_channels=int(input_channels),
        )
        self.future_close_head = nn.Linear(96, len(self.horizons))

    def graph_parameter_ids(self) -> set[int]:
        values: set[int] = set()
        for block in self.backbone.reference.st_blocks:
            scorer = getattr(block, "graph_scorer", None)
            if scorer is not None:
                values.update(id(p) for p in scorer.parameters() if p.requires_grad)
        return values

    def forward_dense(
        self, x: Tensor, *, include_components: bool = False
    ) -> DimitriV2DenseContinuousOutput:
        backbone = self.backbone.forward_continuous(
            x, include_components=include_components
        )
        predictions = (
            self.future_close_head(backbone.hidden)
            .permute(0, 1, 3, 2)
            .unsqueeze(-1)
            .contiguous()
        )
        return DimitriV2DenseContinuousOutput(predictions=predictions, backbone=backbone)

    def forward(self, x: Tensor) -> DimitriV2DenseContinuousOutput:
        return self.forward_dense(x)

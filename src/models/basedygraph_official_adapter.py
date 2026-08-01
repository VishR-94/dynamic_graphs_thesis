from __future__ import annotations

"""Minimal adapter around the pinned official BaseDyGraph implementation.

The trainable architecture is imported directly from ``external/BaseDyGraph``.
This module only converts the project's tensor contract to the official one and
selects the final context-to-next-state logit.

Project input
-------------
``context_s1`` has shape ``[B, T, N]``.

Official input
--------------
BaseDyGraph expects state IDs with shape ``[B, N, T]`` and its standard head
predicts only transitions that have a following position inside the supplied
sequence. To obtain the unseen ``T -> T+1`` prediction through the complete
official forward path, the adapter appends one dummy token. Causal temporal
attention guarantees that the representation at position ``T`` cannot depend
on the dummy at ``T+1``. The adapter then selects the last official logit and
the graph at the final real context position.
"""

import hashlib
import importlib.util
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch
from torch import Tensor, nn


PINNED_BASEDYGRAPH_COMMIT = "26d19efa7c0503d56272d52dc00f574bee61ed24"
OFFICIAL_BASEDYGRAPH_GRAPH_ORIENTATION = "row=target,column=source"


@dataclass(frozen=True)
class OfficialBaseDyGraphModules:
    utilities: ModuleType
    modules: ModuleType
    model: ModuleType
    source_dir: Path
    commit: str | None


@dataclass(frozen=True)
class OfficialBaseDyGraphOneStepOutput:
    """Project-facing output from one official BaseDyGraph forward pass."""

    s1_logits: Tensor  # [B, N, K]
    predicted_s1: Tensor  # [B, N]
    selected_graph: Tensor | None  # [B, G, N, N]
    per_layer_graphs: tuple[Tensor | None, ...]  # each [B, G, N, N]
    temporal_final: Tensor  # [B, N, D]
    spatial_final: Tensor  # [B, N, D]


@dataclass(frozen=True)
class OfficialBaseDyGraphRunConfig:
    """Explicit official model settings used by the project runner."""

    num_states: int = 1024
    num_nodes: int = 93
    context_length: int = 60
    d_model: int = 64
    nhead: int = 4
    num_temporal_layers: int = 2
    num_spatial_layers: int = 1
    ff_mult: int = 4
    num_edge_heads: int = 2
    graph_hidden_dim: int = 64
    dropout: float = 0.0
    spatial_dropout: float = 0.0
    spatial_module_type: str = "static_graph"
    spatial_value: str = "hidden"
    graph_activation: str = "softmax"
    use_node_embedding: bool = True
    use_state_pair_bias: bool = False
    add_self_loops: bool = False
    symmetric_graph: bool = False
    num_st_blocks: int = 1
    first_spatial_module_type: str | None = None
    st_block_post_norm: bool = True
    dummy_state_id: int = 0

    def validate(self) -> None:
        if self.num_states <= 1:
            raise ValueError("num_states must be greater than one.")
        if self.num_nodes <= 1:
            raise ValueError("num_nodes must be greater than one.")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.num_edge_heads <= 0 or self.d_model % self.num_edge_heads != 0:
            raise ValueError("d_model must be divisible by num_edge_heads.")
        if self.graph_hidden_dim <= 0 or self.graph_hidden_dim % self.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads.")
        if self.num_temporal_layers <= 0:
            raise ValueError("num_temporal_layers must be positive.")
        if self.num_spatial_layers <= 0:
            raise ValueError("num_spatial_layers must be positive.")
        if self.ff_mult <= 0:
            raise ValueError("ff_mult must be positive.")
        if self.num_st_blocks <= 0:
            raise ValueError("num_st_blocks must be positive.")
        if self.spatial_module_type not in {
            "none",
            "static_graph",
            "dynamic_graph",
            "dynamic_base",
        }:
            raise ValueError(
                "spatial_module_type must be one of 'none', 'static_graph', "
                "'dynamic_graph', or 'dynamic_base'."
            )
        if self.first_spatial_module_type not in {
            None,
            "",
            "same",
            "none",
            "static_graph",
            "dynamic_graph",
            "dynamic_base",
        }:
            raise ValueError("Unsupported first_spatial_module_type.")
        if not 0 <= self.dummy_state_id < self.num_states:
            raise ValueError("dummy_state_id lies outside the state vocabulary.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_external_source_dir(
    external_source_dir: str | Path | None,
) -> Path:
    if external_source_dir is None:
        source_dir = _repository_root() / "external" / "BaseDyGraph" / "src"
    else:
        source_dir = Path(external_source_dir).expanduser().resolve()

    required = (
        source_dir / "utilities.py",
        source_dir / "modules.py",
        source_dir / "model.py",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The pinned BaseDyGraph submodule is unavailable. Missing: "
            f"{[str(path) for path in missing]}. Run "
            "`git submodule update --init --recursive`."
        )
    return source_dir


def _git_commit(source_dir: Path) -> str | None:
    repository_dir = source_dir.parent
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_dir,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _assert_module_origin(module: ModuleType, source_dir: Path, name: str) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"Official BaseDyGraph module {name!r} has no __file__.")
    observed = Path(module_file).resolve()
    if observed.parent != source_dir:
        raise RuntimeError(
            f"Top-level module name {name!r} is already bound to {observed}, "
            f"not the pinned BaseDyGraph source directory {source_dir}. Restart "
            "the process before importing the official adapter."
        )


@lru_cache(maxsize=4)
def _load_official_modules_cached(source_dir_text: str) -> OfficialBaseDyGraphModules:
    """Load the official files under an isolated module namespace.

    Both pinned external repositories use a top-level Python package/module name
    called ``model``. Importing BaseDyGraph through ordinary ``sys.path``
    mutation would therefore collide with the existing official Kronos import.
    We execute BaseDyGraph's three source files under unique names while
    temporarily exposing only the aliases required by its own unchanged
    ``from utilities import *`` / ``from modules import *`` statements. The
    aliases are restored immediately after import; the loaded official classes
    keep their own module globals and remain fully functional.
    """

    source_dir = Path(source_dir_text).resolve()
    namespace = (
        "_pinned_basedygraph_"
        + hashlib.sha256(str(source_dir).encode("utf-8")).hexdigest()[:12]
    )

    loaded_names: list[str] = []

    def load_file(filename: str, suffix: str) -> ModuleType:
        module_name = f"{namespace}_{suffix}"
        path = source_dir / filename
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import spec for {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loaded_names.append(module_name)
        spec.loader.exec_module(module)
        return module

    missing = object()
    previous_utilities = sys.modules.get("utilities", missing)
    previous_modules = sys.modules.get("modules", missing)

    try:
        utilities = load_file("utilities.py", "utilities")
        sys.modules["utilities"] = utilities

        modules = load_file("modules.py", "modules")
        sys.modules["modules"] = modules

        model = load_file("model.py", "model")
    except Exception:
        for module_name in loaded_names:
            sys.modules.pop(module_name, None)
        raise
    finally:
        if previous_utilities is missing:
            sys.modules.pop("utilities", None)
        else:
            sys.modules["utilities"] = previous_utilities

        if previous_modules is missing:
            sys.modules.pop("modules", None)
        else:
            sys.modules["modules"] = previous_modules

    _assert_module_origin(utilities, source_dir, "utilities")
    _assert_module_origin(modules, source_dir, "modules")
    _assert_module_origin(model, source_dir, "model")

    return OfficialBaseDyGraphModules(
        utilities=utilities,
        modules=modules,
        model=model,
        source_dir=source_dir,
        commit=_git_commit(source_dir),
    )


def load_official_basedygraph_modules(
    external_source_dir: str | Path | None = None,
    *,
    require_pinned_commit: bool = True,
) -> OfficialBaseDyGraphModules:
    """Import the official source directly from the Git submodule."""

    source_dir = _resolve_external_source_dir(external_source_dir)
    loaded = _load_official_modules_cached(str(source_dir))

    if require_pinned_commit:
        if loaded.commit is None:
            raise RuntimeError(
                "Could not resolve the BaseDyGraph submodule commit at "
                f"{source_dir}. The official integration requires a Git "
                "submodule checkout so the pinned revision can be verified."
            )
        if loaded.commit != PINNED_BASEDYGRAPH_COMMIT:
            raise RuntimeError(
                "BaseDyGraph submodule commit mismatch. Expected "
                f"{PINNED_BASEDYGRAPH_COMMIT}, observed {loaded.commit}."
            )

    return loaded


def build_official_model_config(
    run_config: OfficialBaseDyGraphRunConfig,
    *,
    external_source_dir: str | Path | None = None,
) -> Any:
    """Construct the official ``ModelConfig`` with every field explicit."""

    run_config.validate()
    official = load_official_basedygraph_modules(external_source_dir)

    return official.utilities.ModelConfig(
        num_states=run_config.num_states,
        num_nodes=run_config.num_nodes,
        d_model=run_config.d_model,
        nhead=run_config.nhead,
        num_temporal_layers=run_config.num_temporal_layers,
        num_spatial_layers=run_config.num_spatial_layers,
        dropout=run_config.dropout,
        ff_mult=run_config.ff_mult,
        max_seq_len=run_config.context_length + 1,
        num_edge_heads=run_config.num_edge_heads,
        graph_hidden_dim=run_config.graph_hidden_dim,
        spatial_dropout=run_config.spatial_dropout,
        use_node_embedding=run_config.use_node_embedding,
        use_state_pair_bias=run_config.use_state_pair_bias,
        add_self_loops=run_config.add_self_loops,
        symmetric_graph=run_config.symmetric_graph,
        predict_next_state=True,
        temporal_module_type="transformer",
        spatial_module_type=run_config.spatial_module_type,
        spatial_value=run_config.spatial_value,
        graph_activation=run_config.graph_activation,
        gate_tau=0.5,
        gate_row_normalise=True,
        dynamic_residual_gate="none",
        dynamic_residual_init=1.0,
        dynamic_residual_learnable=True,
        dynamic_residual_mix="logit",
        interlaced_st_blocks=run_config.num_st_blocks > 1,
        num_st_blocks=run_config.num_st_blocks,
        first_spatial_module_type=run_config.first_spatial_module_type,
        st_block_post_norm=run_config.st_block_post_norm,
        graph_eval_layer=-1,
        graph_log_all_layers=True,
        graph_reg_layer=-1,
        graph_reg_warmup_epochs=0,
        graph_entropy_reg=0.0,
        graph_target_entropy=None,
        graph_target_entropy_reg=0.0,
        graph_temporal_smooth_reg=0.0,
    )


class OfficialBaseDyGraphOneStep(nn.Module):
    """Use the complete official model for one unseen next-state prediction.

    No trainable operation is implemented in this wrapper. ``official_model``
    owns all parameters, including the state/node embeddings, temporal
    Transformer, graph scorer, spatial message-passing layers, residuals,
    normalisation, FFNs, and the official next-state projection.
    """

    def __init__(
        self,
        run_config: OfficialBaseDyGraphRunConfig,
        *,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 1.0e-4,
        scheduler_t_max: int | None = None,
        external_source_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        run_config.validate()
        self.run_config = run_config
        self.official_modules = load_official_basedygraph_modules(
            external_source_dir
        )
        self.official_config = build_official_model_config(
            run_config,
            external_source_dir=external_source_dir,
        )
        self.official_model = (
            self.official_modules.model.DiscreteSTGraphLightningModule(
                self.official_config,
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
                scheduler_t_max=scheduler_t_max,
            )
        )

    @property
    def num_states(self) -> int:
        return int(self.run_config.num_states)

    @property
    def num_nodes(self) -> int:
        return int(self.run_config.num_nodes)

    @property
    def context_length(self) -> int:
        return int(self.run_config.context_length)

    @property
    def num_edge_heads(self) -> int:
        return int(self.run_config.num_edge_heads)

    @property
    def num_st_blocks(self) -> int:
        return int(self.run_config.num_st_blocks)

    @property
    def external_commit(self) -> str | None:
        return self.official_modules.commit

    def _validate_context(self, context_s1: Tensor) -> Tensor:
        values = torch.as_tensor(context_s1)
        expected = (
            values.shape[0] if values.ndim >= 1 else -1,
            self.context_length,
            self.num_nodes,
        )
        if values.ndim != 3 or tuple(values.shape[1:]) != expected[1:]:
            raise ValueError(
                "context_s1 must have shape [B, T, N] with "
                f"T={self.context_length}, N={self.num_nodes}; observed "
                f"{tuple(values.shape)}."
            )
        values = values.to(dtype=torch.long)
        if values.numel() and (
            int(values.min().item()) < 0
            or int(values.max().item()) >= self.num_states
        ):
            raise ValueError(
                f"context_s1 contains an ID outside [0, {self.num_states - 1}]."
            )
        return values

    @staticmethod
    def _select_context_graph(
        values: Tensor | None,
        *,
        context_index: int,
    ) -> Tensor | None:
        if values is None:
            return None
        graph = torch.as_tensor(values)
        if graph.ndim != 5:
            raise ValueError(
                "Official graph attention must have shape [B, T, G, N, N]."
            )
        if not 0 <= context_index < int(graph.shape[1]):
            raise IndexError("Final context graph index is outside the graph sequence.")
        return graph[:, context_index].contiguous()

    def forward(
        self,
        context_s1: Tensor,
        *,
        dummy_state_id: int | None = None,
    ) -> OfficialBaseDyGraphOneStepOutput:
        context_s1 = self._validate_context(context_s1)
        state_ids = context_s1.permute(0, 2, 1).contiguous()  # [B, N, T]

        dummy_id = (
            self.run_config.dummy_state_id
            if dummy_state_id is None
            else int(dummy_state_id)
        )
        if not 0 <= dummy_id < self.num_states:
            raise ValueError("dummy_state_id lies outside the state vocabulary.")

        dummy = torch.full(
            (state_ids.shape[0], state_ids.shape[1], 1),
            fill_value=dummy_id,
            dtype=state_ids.dtype,
            device=state_ids.device,
        )
        official_input = torch.cat((state_ids, dummy), dim=-1)
        official_output: Mapping[str, Any] = self.official_model(official_input)

        all_logits = torch.as_tensor(official_output["next_state_logits"])
        expected_logits_shape = (
            state_ids.shape[0],
            self.num_nodes,
            self.context_length,
            self.num_states,
        )
        if tuple(all_logits.shape) != expected_logits_shape:
            raise RuntimeError(
                "Unexpected official next-state-logit shape. Expected "
                f"{expected_logits_shape}, observed {tuple(all_logits.shape)}."
            )

        s1_logits = all_logits[:, :, -1, :].contiguous()
        context_index = self.context_length - 1
        selected_graph = self._select_context_graph(
            official_output.get("graph_attn"),
            context_index=context_index,
        )

        raw_per_layer = official_output.get("block_graph_attns")
        if raw_per_layer is None:
            per_layer_graphs: tuple[Tensor | None, ...] = (
                (selected_graph,)
                if self.run_config.spatial_module_type != "none"
                else (None,)
            )
        else:
            per_layer_graphs = tuple(
                self._select_context_graph(values, context_index=context_index)
                if values is not None
                else None
                for values in raw_per_layer
            )

        temporal = torch.as_tensor(official_output["temporal_repr"])
        spatial = torch.as_tensor(official_output["spatial_repr"])
        if temporal.ndim != 4 or spatial.ndim != 4:
            raise RuntimeError("Official representations must have shape [B, T, N, D].")

        return OfficialBaseDyGraphOneStepOutput(
            s1_logits=s1_logits,
            predicted_s1=s1_logits.argmax(dim=-1),
            selected_graph=selected_graph,
            per_layer_graphs=per_layer_graphs,
            temporal_final=temporal[:, context_index].contiguous(),
            spatial_final=spatial[:, context_index].contiguous(),
        )

    def direct_reference_logits(self, context_s1: Tensor) -> Tensor:
        """Test-only reference: official backbone + official final projection."""
        context_s1 = self._validate_context(context_s1)
        state_ids = context_s1.permute(0, 2, 1).contiguous()
        output = self.official_model.backbone(state_ids)
        final_hidden = output["spatial_repr"][:, -1]
        return self.official_model.next_state_head.proj(final_hidden).contiguous()

    def configure_official_optimizers(self) -> dict[str, Any]:
        """Return the official AdamW + cosine configuration unchanged."""
        configured = self.official_model.configure_optimizers()
        if not isinstance(configured, Mapping):
            raise TypeError("Official configure_optimizers() returned an unexpected value.")
        return dict(configured)


def assert_official_one_step_parity(
    model: OfficialBaseDyGraphOneStep,
    context_s1: Tensor,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> dict[str, float]:
    """Prove dummy-token invariance and full-forward/direct-head parity."""

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        first = model(context_s1, dummy_state_id=0).s1_logits
        second_dummy = 1 if model.num_states > 1 else 0
        second = model(context_s1, dummy_state_id=second_dummy).s1_logits
        direct = model.direct_reference_logits(context_s1)

    if was_training:
        model.train()

    dummy_difference = float((first - second).abs().max().item())
    direct_difference = float((first - direct).abs().max().item())

    if not torch.allclose(first, second, atol=atol, rtol=rtol):
        raise AssertionError(
            "Official final-context logits depend on the appended dummy token. "
            f"Maximum difference: {dummy_difference}."
        )
    if not torch.allclose(first, direct, atol=atol, rtol=rtol):
        raise AssertionError(
            "Complete official forward and direct final-head reference differ. "
            f"Maximum difference: {direct_difference}."
        )

    return {
        "dummy_invariance_max_abs_difference": dummy_difference,
        "direct_reference_max_abs_difference": direct_difference,
    }

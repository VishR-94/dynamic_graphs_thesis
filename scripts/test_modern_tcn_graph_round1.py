from __future__ import annotations

"""Fast CPU contracts for the ModernTCN graph Round-1 ladder."""

import json
from pathlib import Path
import sys
import tempfile
import types

import pandas as pd
import torch
from torch import nn

from src.data.continuous_forecast_dataset import (
    ContinuousDatasetConfig,
    build_continuous_dataset,
)
from src.models.graph_priors import (
    build_absolute_correlation_graph_prior,
    build_sector_graph_prior,
)
from src.models.modern_tcn_graph_round1 import (
    ModernTCNGraphRound1Model,
    PriorMixedDynamicGraphLearner,
    align_state_embeddings_to_modern_tcn_patches,
    build_v2_prior_logits,
    round1_model_config_from_mapping,
)
from src.training.modern_tcn_round1_specs import (
    make_gate_optimisation_ablation_specs,
    make_round1_specs,
    make_six_head_ablation_spec,
)
from src.training.run_modern_tcn_graph_round1 import (
    _advance_schedule,
    _build_optimizer,
    _learning_rates,
    _validate_config,
)


def _install_fake_modern_tcn() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ModernTCN"
        / "ModernTCN-Long-term-forecasting"
    )
    root.mkdir(parents=True, exist_ok=True)

    package = types.ModuleType("models")
    package.__path__ = []
    module = types.ModuleType("models.ModernTCN")

    class FakeHead(nn.Module):
        def __init__(self, *, d_model: int, length: int, horizons: int) -> None:
            super().__init__()
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(d_model * length, horizons)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(self.flatten(x))

    class FakeInner(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.patch_size = int(config.patch_size)
            self.patch_stride = int(config.patch_stride)
            self.padding = self.patch_size - self.patch_stride
            self.d_model = int(config.dims[0])
            self.output_length = int(config.seq_len) // self.patch_stride
            self.stem = nn.Linear(self.patch_size, self.d_model)
            self.head = FakeHead(
                d_model=self.d_model,
                length=self.output_length,
                horizons=int(config.pred_len),
            )

        def forward_feature(self, x: torch.Tensor) -> torch.Tensor:
            # [B,M,T] -> [B,M,D,L], matching the official feature contract.
            if self.padding:
                x = torch.cat(
                    [x, x[..., -1:].expand(*x.shape[:-1], self.padding)],
                    dim=-1,
                )
            patches = x.unfold(-1, self.patch_size, self.patch_stride)
            features = self.stem(patches)
            return features.permute(0, 1, 3, 2).contiguous()

    class FakeModel(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.model = FakeInner(config)

    module.Model = FakeModel
    package.ModernTCN = module
    sys.modules["models"] = package
    sys.modules["models.ModernTCN"] = module


def _synthetic_split(num_nodes: int = 4) -> dict:
    torch.manual_seed(19)
    channels = ["open", "high", "low", "close", "volume", "amount"]
    samples = []
    for day_index in range(3):
        base = 50.0 + torch.cumsum(0.01 * torch.randn(140, num_nodes), dim=0)
        open_price = base + 0.002 * torch.randn_like(base)
        close = base + 0.002 * torch.randn_like(base)
        high = torch.maximum(open_price, close) + 0.01
        low = torch.minimum(open_price, close) - 0.01
        volume = 1000.0 + 10.0 * torch.rand_like(base)
        amount = torch.zeros_like(base)
        values = torch.stack(
            [open_price, high, low, close, volume, amount],
            dim=-1,
        )
        samples.append((values, {}, f"2024-01-{day_index + 2:02d}"))
    return {
        "samples": samples,
        "asset_cols": [f"A{index}" for index in range(num_nodes)],
        "channels": channels,
    }


def _batch(dataset, count: int = 2) -> dict:
    items = [dataset[index] for index in range(count)]
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return result


def main() -> None:
    _install_fake_modern_tcn()

    # Prior builders: sector and training-only absolute correlation.
    split = _synthetic_split()
    with tempfile.TemporaryDirectory() as temporary:
        profile_path = Path(temporary) / "company_profiles.csv"
        pd.DataFrame(
            {
                "ticker": ["A0", "A1", "A2", "A3"],
                "name": ["x"] * 4,
                "c3": [0] * 4,
                "c4": [0] * 4,
                "c5": [0] * 4,
                "sector": ["Tech", "Tech", "Energy", "Energy"],
            }
        ).to_csv(profile_path, index=False)
        sector, labels = build_sector_graph_prior(
            split["asset_cols"], profile_path
        )
    if labels != ["Tech", "Tech", "Energy", "Energy"]:
        raise AssertionError("Sector labels were not retained in asset order.")
    torch.testing.assert_close(
        sector.sum(dim=-1), torch.ones(4), atol=1.0e-7, rtol=0.0
    )
    torch.testing.assert_close(
        torch.diagonal(sector), torch.zeros(4), atol=0.0, rtol=0.0
    )
    correlation = build_absolute_correlation_graph_prior(
        split,
        expected_asset_cols=split["asset_cols"],
    )
    torch.testing.assert_close(
        correlation.sum(dim=-1), torch.ones(4), atol=1.0e-6, rtol=0.0
    )
    torch.testing.assert_close(
        torch.diagonal(correlation), torch.zeros(4), atol=0.0, rtol=0.0
    )

    # V2 prior initialisation is the documented max-normalise, global-centre,
    # scale, then reproducible per-head jitter rule.
    logits = build_v2_prior_logits(
        sector,
        num_heads=2,
        scale=4.0,
        jitter=0.0,
        seed=42,
    )
    expected_logits = 4.0 * (sector / sector.max() - (sector / sector.max()).mean())
    torch.testing.assert_close(logits[0], expected_logits)
    torch.testing.assert_close(logits[1], expected_logits)

    # Direct post-normalisation convex alpha mixture.
    learner = PriorMixedDynamicGraphLearner(
        d_model=8,
        num_nodes=4,
        num_heads=1,
        graph_hidden_dim=8,
        use_state_pathway=False,
        static_prior=sector,
        initial_alpha=0.25,
        prior_scale=4.0,
        prior_jitter=0.0,
        prior_seed=42,
    )
    nn.init.zeros_(learner.q_proj.weight)
    nn.init.zeros_(learner.q_proj.bias)
    nn.init.zeros_(learner.k_proj.weight)
    nn.init.zeros_(learner.k_proj.bias)
    hidden = torch.randn(2, 5, 4, 8)
    graph = learner(hidden)
    alpha = learner.alpha()
    if alpha is None:
        raise AssertionError("Prior mixture alpha is missing.")
    torch.testing.assert_close(
        alpha,
        torch.tensor(0.25),
        atol=1.0e-6,
        rtol=0.0,
    )
    expanded_static = graph.base.expand(2, -1, -1, -1)
    expected_mixed = 0.75 * expanded_static + 0.25 * graph.dynamic
    torch.testing.assert_close(graph.selected, expected_mixed)

    # Specs are genuinely configurable; no fixed expected-window constants
    # are needed to construct a different context/stride/horizon task.
    configurable_specs = make_round1_specs(
        prior_type="correlation",
        context_length=30,
        stride=7,
        horizons=(1, 3),
    )
    for spec in configurable_specs:
        if spec.config["data"]["context_length"] != 30:
            raise AssertionError("Context length was not propagated.")
        if spec.config["data"]["stride"] != 7:
            raise AssertionError("Stride was not propagated.")
        if spec.config["data"]["horizons"] != [1, 3]:
            raise AssertionError("Horizons were not propagated.")

    # Gate/optimisation follow-up: three one-head architectures across
    # three controlled ablation families, with no six-head runs.
    followup_specs = make_gate_optimisation_ablation_specs(
        prior_type="sector",
        context_length=60,
        stride=7,
        horizons=(1, 3),
    )
    if len(followup_specs) != 9:
        raise AssertionError("Expected nine gate/optimisation ablations.")
    if {spec.graph_heads for spec in followup_specs} != {1}:
        raise AssertionError("The follow-up unexpectedly contains multi-head runs.")
    family_counts = pd.Series(
        [spec.ablation_family for spec in followup_specs]
    ).value_counts().to_dict()
    expected_family_counts = {
        "dimitri_optimisation": 3,
        "no_beta_round1_optimisation": 3,
        "no_beta_dimitri_optimisation": 3,
    }
    if family_counts != expected_family_counts:
        raise AssertionError(
            f"Unexpected follow-up family counts: {family_counts}."
        )
    for spec in followup_specs:
        _validate_config(spec.config)
        training = spec.config["training"]
        if spec.optimisation_profile == "dimitri":
            expected = {
                "optimizer": "adamw",
                "parameter_grouping": "shared",
                "scheduler": "cosine_annealing",
                "learning_rate": 1.2e-3,
                "graph_learning_rate": 1.2e-3,
                "weight_decay": 1.0e-4,
                "gradient_clip_norm": 0.0,
                "mixed_precision": False,
                "max_epochs": 120,
                "patience": 15,
            }
            for key, value in expected.items():
                if training[key] != value:
                    raise AssertionError(
                        f"Dimitri profile {key}={training[key]!r}; "
                        f"expected {value!r}."
                    )
        if spec.spatial_gate_type == "none" and (
            spec.config["model"]["spatial"]["gate_type"] != "none"
        ):
            raise AssertionError("No-beta spec retained a learned beta gate.")

    # Six-head ablation preserves 32 graph dimensions per head.
    standard_specs = make_round1_specs(prior_type="sector")
    six_head = make_six_head_ablation_spec(
        standard_specs[2],
        graph_heads=6,
        per_head_dim=32,
    )
    if six_head.graph_heads != 6 or six_head.graph_hidden_dim != 192:
        raise AssertionError("Six-head capacity ablation is misconfigured.")

    # Exact dynamic-only control parity against the existing continuous model.
    from src.models.continuous_forecaster import ContinuousForecaster

    control_values = standard_specs[0].config
    control_config = round1_model_config_from_mapping(control_values, num_nodes=4)
    dataset = build_continuous_dataset(
        split,
        config=ContinuousDatasetConfig(
            context_length=60,
            horizons=(1, 5, 15, 30, 60),
            stride=15,
        ),
    )
    batch = _batch(dataset)

    torch.manual_seed(42)
    existing = ContinuousForecaster(control_config.forecaster).eval()
    torch.manual_seed(42)
    round1_control = ModernTCNGraphRound1Model(control_config).eval()
    with torch.no_grad():
        existing_output = existing(
            batch["x"],
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
        round1_output = round1_control(
            batch["x"],
            context_start=batch["context_start"],
            session_length=batch["session_length"],
        )
    torch.testing.assert_close(
        round1_output.temporal_hidden,
        existing_output.temporal_hidden,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        round1_output.graph.selected,
        existing_output.graph.selected,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        round1_output.graph_spatial_hidden,
        existing_output.graph_spatial_hidden,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        round1_output.predictions,
        existing_output.predictions,
        atol=0.0,
        rtol=0.0,
    )
    if round1_output.graph.base is not None or round1_output.alpha is not None:
        raise AssertionError("Dynamic-only control unexpectedly uses a prior/alpha.")

    # Dimitri-style state exposure reaches scorer, spatial values, state
    # projection, alpha, and beta through forecast loss.
    state_config = round1_model_config_from_mapping(
        standard_specs[2].config,
        num_nodes=4,
    )
    torch.manual_seed(7)
    state_model = ModernTCNGraphRound1Model(
        state_config,
        static_prior=sector,
    )
    state_output = state_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    loss = state_output.predictions.square().mean()
    loss.backward()
    required_gradients = {
        "state projection": state_model.state_projection.weight.grad,
        "graph Q": state_model.graph_learner.q_proj.weight.grad,
        "graph K": state_model.graph_learner.k_proj.weight.grad,
        "spatial value": state_model.spatial_module.value_projection.weight.grad,
        "alpha": state_model.graph_learner.raw_alpha.grad,
        "beta": state_model.spatial_gate.raw_beta.grad,
    }
    for name, gradient in required_gradients.items():
        if gradient is None or float(gradient.detach().norm().item()) <= 0.0:
            raise AssertionError(f"Forecast loss did not reach {name}.")

    # Dimitri optimisation uses one AdamW group, so backbone, graph,
    # alpha, beta, state projection, and head all receive the same LR.
    dimitri_spec = next(
        spec
        for spec in followup_specs
        if spec.variant == "prior_mixture_state"
        and spec.optimisation_profile == "dimitri"
        and spec.spatial_gate_type == "learned_scalar"
    )
    dimitri_config = round1_model_config_from_mapping(
        dimitri_spec.config,
        num_nodes=4,
    )
    torch.manual_seed(11)
    dimitri_model = ModernTCNGraphRound1Model(
        dimitri_config,
        static_prior=sector,
    )
    dimitri_optimizer = _build_optimizer(dimitri_model, dimitri_spec.config)
    if not isinstance(dimitri_optimizer, torch.optim.AdamW):
        raise AssertionError("Dimitri profile did not build AdamW.")
    if len(dimitri_optimizer.param_groups) != 1:
        raise AssertionError("Dimitri profile must use one shared parameter group.")
    initial_lrs = _learning_rates(dimitri_optimizer)
    for key in ("backbone", "graph", "shared"):
        if initial_lrs[key] != 1.2e-3:
            raise AssertionError(
                f"Unexpected shared LR for {key}: {initial_lrs[key]}."
            )
    _advance_schedule(
        dimitri_optimizer,
        training=dimitri_spec.config["training"],
        completed_epoch=60,
    )
    halfway = _learning_rates(dimitri_optimizer)["shared"]
    if halfway is None or abs(halfway - 6.0e-4) > 1.0e-10:
        raise AssertionError(f"Cosine schedule halfway LR is {halfway}.")

    # Removing beta means the forecasting head consumes the full spatial
    # module output. The spatial block's own residual/normalisation remains.
    no_beta_spec = next(
        spec
        for spec in followup_specs
        if spec.variant == "prior_mixture_state"
        and spec.optimisation_profile == "round1"
        and spec.spatial_gate_type == "none"
    )
    no_beta_config = round1_model_config_from_mapping(
        no_beta_spec.config,
        num_nodes=4,
    )
    torch.manual_seed(13)
    no_beta_model = ModernTCNGraphRound1Model(
        no_beta_config,
        static_prior=sector,
    )
    no_beta_output = no_beta_model(
        batch["x"],
        context_start=batch["context_start"],
        session_length=batch["session_length"],
    )
    if no_beta_model.spatial_gate.raw_beta is not None:
        raise AssertionError("No-beta model still owns a beta parameter.")
    torch.testing.assert_close(
        no_beta_output.fused_hidden,
        no_beta_output.graph_spatial_hidden,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        no_beta_output.beta,
        torch.tensor(1.0),
        atol=0.0,
        rtol=0.0,
    )

    minute_states = torch.randn(2, 60, 4, 32)
    aligned = align_state_embeddings_to_modern_tcn_patches(
        minute_states,
        patch_size=8,
        patch_stride=4,
    )
    if tuple(aligned.shape) != (2, 15, 4, 32):
        raise AssertionError("ModernTCN state-patch alignment is incorrect.")
    torch.testing.assert_close(
        aligned[:, -1],
        minute_states[:, -1],
        atol=0.0,
        rtol=0.0,
    )

    runner_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "training"
        / "run_modern_tcn_graph_round1.py"
    ).read_text(encoding="utf-8")
    if "EXPECTED_TRAIN_WINDOWS" in runner_source or "EXPECTED_VALIDATION_WINDOWS" in runner_source:
        raise AssertionError("The new runner contains a hard-coded window-count gate.")

    print("ModernTCN graph Round-1 contracts passed.")


if __name__ == "__main__":
    main()

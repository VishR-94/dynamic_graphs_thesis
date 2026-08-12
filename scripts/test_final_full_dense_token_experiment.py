from __future__ import annotations

"""Contracts for the final all-origins/all-60 token experiment."""

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset

from src.models.final_token_v2_models import (
    DenseOriginStructuredTokenPredictor,
    DenseTransformerTokenForecaster,
)
from src.training.final_token_v2_specs import (
    make_full_dense_transformer_token_spec,
)
from src.training.run_final_token_v2_experiment import (
    _all_origins_full_path_backward,
    _all_origins_full_path_token_targets,
    _new_grad_scaler,
    _train_token_epoch,
)


def _test_spec() -> None:
    spec = make_full_dense_transformer_token_spec(
        batch_size=4,
        origin_chunk_size=8,
        selection_batch_size=4,
        export_batch_size=2,
    )
    config = spec.config
    assert spec.model_kind == "dense_transformer_token"
    assert "full_dense60_allorigins" in spec.run_name
    assert config["model"]["num_st_blocks"] == 3
    assert config["model"]["temporal"] == {
        "type": "transformer",
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_multiplier": 2,
        "dropout": 0.0,
        "position_embedding": False,
    }
    assert config["model"]["graph"] == {
        "type": "static_dynamic_mixture",
        "num_heads_per_block": [1, 1, 1],
        "hidden_dims_per_block": [64, 64, 64],
        "activations_per_block": ["softmax", "softmax", "sparsemax"],
        "add_self_loops": False,
        "initial_alpha": 0.5,
    }
    assert config["model"]["prior"] == {
        "type": "uniform",
        "static_logits": "zeros",
        "dynamic_logits": "zeros_at_initialisation",
    }
    assert config["training"]["batch_size"] == 4
    assert config["training"]["loss"] == {
        "type": "coarse_s1_cross_entropy",
        "horizon_weighting": "uniform",
        "dense_origins": True,
        "dense_objective": "all_60_future_positions_per_origin",
        "future_steps_per_origin": 60,
        "origin_chunk_size": 8,
    }


def _test_target_alignment() -> None:
    context = torch.tensor([[[0], [1], [2]]])
    future = torch.tensor([[[3], [4], [5]]])
    target = _all_origins_full_path_token_targets(context, future)
    assert tuple(target.shape) == (1, 3, 3, 1)
    expected = torch.tensor(
        [[[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]]]
    )
    assert torch.equal(target, expected)
    assert torch.equal(target[:, -1], future)


def _test_vectorised_origin_parity() -> None:
    torch.manual_seed(4)
    predictor = DenseOriginStructuredTokenPredictor(
        d_model=8,
        prediction_length=7,
        vocabulary_size=16,
        num_layers=1,
        num_heads=2,
        feedforward_multiplier=2,
        dropout=0.0,
    ).eval()
    hidden = torch.randn(2, 5, 3, 8)
    origins = (0, 2, 4)
    with torch.no_grad():
        observed = predictor.forward_origins(hidden, origins)
        expected = torch.stack(
            [predictor.forward_origin(hidden, origin) for origin in origins],
            dim=1,
        )
    assert tuple(observed.shape) == (2, 3, 7, 3, 16)
    assert torch.allclose(observed, expected, atol=5.0e-7, rtol=1.0e-6)


class _TinyDenseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.future_predictor = DenseOriginStructuredTokenPredictor(
            d_model=4,
            prediction_length=3,
            vocabulary_size=7,
            num_layers=1,
            num_heads=1,
            feedforward_multiplier=2,
            dropout=0.0,
        )

    def forward_backbone(self, context: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(hidden=self.embedding(context.long()))


def _gradients_for_chunk(
    state: dict[str, torch.Tensor],
    *,
    chunk_size: int,
) -> tuple[float, dict[str, torch.Tensor]]:
    model = _TinyDenseModel()
    model.load_state_dict(deepcopy(state), strict=True)
    context = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5], [6, 0]],
            [[1, 2], [3, 4], [5, 6], [0, 1]],
        ]
    )
    future = torch.tensor(
        [
            [[2, 3], [4, 5], [6, 0]],
            [[3, 4], [5, 6], [0, 1]],
        ]
    )
    model.zero_grad(set_to_none=True)
    values = _all_origins_full_path_backward(
        model=model,
        kind="dense_transformer_token",
        context=context,
        future=future,
        loss_config={
            "future_steps_per_origin": 3,
            "origin_chunk_size": chunk_size,
        },
        scaler=_new_grad_scaler(False),
        device=torch.device("cpu"),
        use_amp=False,
    )
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return float(values["objective"]), gradients


def _test_chunking_preserves_objective_and_gradient() -> None:
    torch.manual_seed(7)
    state = deepcopy(_TinyDenseModel().state_dict())
    first_objective, first_gradients = _gradients_for_chunk(state, chunk_size=1)
    second_objective, second_gradients = _gradients_for_chunk(state, chunk_size=3)
    assert abs(first_objective - second_objective) < 1.0e-6
    assert first_gradients.keys() == second_gradients.keys()
    for name in first_gradients:
        assert torch.allclose(
            first_gradients[name],
            second_gradients[name],
            atol=2.0e-6,
            rtol=2.0e-5,
        ), name


def _test_real_graph_path_receives_gradient() -> None:
    torch.manual_seed(8)
    model = DenseTransformerTokenForecaster(
        num_nodes=3,
        context_length=60,
        prediction_length=60,
    )
    context = torch.randint(0, 1024, (1, 60, 3))
    future = torch.randint(0, 1024, (1, 60, 3))
    model.zero_grad(set_to_none=True)
    values = _all_origins_full_path_backward(
        model=model,
        kind="dense_transformer_token",
        context=context,
        future=future,
        loss_config={
            "future_steps_per_origin": 60,
            "origin_chunk_size": 10,
        },
        scaler=_new_grad_scaler(False),
        device=torch.device("cpu"),
        use_amp=False,
    )
    assert values["all_origins_count"] == 1 * 60 * 60 * 3
    graph_gradients = [
        parameter.grad
        for block in model.blocks
        for parameter in block.graph_learner.parameters()
        if parameter.requires_grad
    ]
    assert graph_gradients
    assert all(value is not None for value in graph_gradients)
    assert sum(float(value.norm().item()) for value in graph_gradients) > 0.0



class _TinyTokenDataset(Dataset):
    def __init__(self) -> None:
        torch.manual_seed(91)
        self.context = torch.randint(
            0, 1024, (2, 4, 3, 2), dtype=torch.int16
        )
        self.future = torch.randint(
            0, 1024, (2, 3, 3), dtype=torch.int16
        )

    def __len__(self) -> int:
        return int(self.context.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "context_tokens": self.context[index],
            "target_s1": self.future[index],
        }


def _test_training_epoch_branch() -> None:
    torch.manual_seed(92)
    model = DenseTransformerTokenForecaster(
        num_nodes=3,
        context_length=4,
        prediction_length=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    config = {
        "data": {"context_length": 4},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "seed": 42,
            "mixed_precision": False,
            "gradient_clip_norm": 1.0,
            "loss": {
                "dense_origins": True,
                "dense_objective": "all_60_future_positions_per_origin",
                "future_steps_per_origin": 3,
                "origin_chunk_size": 2,
            },
        },
    }
    values = _train_token_epoch(
        model=model,
        kind="dense_transformer_token",
        dataset=_TinyTokenDataset(),
        config=config,
        optimizer=optimizer,
        scaler=_new_grad_scaler(False),
        device=torch.device("cpu"),
        epoch=1,
    )
    assert values["training_dense_origin_count"] == 4.0
    assert values["training_dense_future_steps_per_origin"] == 3.0
    assert values["training_dense_origin_chunk_size"] == 2.0
    assert values["training_dense_all_origins_cross_entropy"] > 0.0
    assert values["training_final_path_cross_entropy"] > 0.0

def main() -> None:
    _test_spec()
    _test_target_alignment()
    _test_vectorised_origin_parity()
    _test_chunking_preserves_objective_and_gradient()
    _test_real_graph_path_receives_gradient()
    _test_training_epoch_branch()
    print("Final full-dense token contracts passed.")


if __name__ == "__main__":
    main()

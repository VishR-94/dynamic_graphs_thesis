from __future__ import annotations

"""CPU contracts for the final frozen-forecaster decoder post-training path."""

from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from torch import nn

from src.models.kronos_decoder_post_training import TrainableKronosCoarseDecoder
from src.training.kronos_decoder_post_training_specs import (
    DEFAULT_LOSS_RATIO_ANCHORS,
    make_decoder_post_training_specs,
    stretched_exponential_weights,
)
from src.training.run_kronos_decoder_post_training import WeightedAll60Loss


def _tiny_decoder(seed: int = 7) -> TrainableKronosCoarseDecoder:
    torch.manual_seed(seed)
    return TrainableKronosCoarseDecoder(
        post_quant_embed_pre=nn.Linear(2, 4),
        decoder_layers=nn.ModuleList(
            [
                nn.Sequential(nn.Linear(4, 4), nn.GELU()),
                nn.Sequential(nn.Linear(4, 4), nn.GELU()),
            ]
        ),
        reconstruction_head=nn.Linear(4, 6),
        s1_bits=2,
        codebook_dim=4,
        eps=1.0e-5,
    )


def _test_weight_curve() -> None:
    weights = stretched_exponential_weights()
    assert len(weights) == 60
    assert abs(sum(weights) - 1.0) < 1.0e-7
    unnormalised = [value / weights[0] for value in weights]
    # The requested smooth weighting makes the anchor contributions roughly
    # equal after multiplying by the observed error ratios.
    for horizon, ratio in DEFAULT_LOSS_RATIO_ANCHORS.items():
        contribution = unnormalised[horizon - 1] * ratio
        assert 0.88 <= contribution <= 1.08, (horizon, contribution)


def _test_decoder_shapes_and_gradients() -> None:
    decoder = _tiny_decoder()
    context = torch.randint(0, 4, (2, 3, 2))
    future = torch.randint(0, 4, (3, 2, 5, 2))
    mean = torch.zeros(2, 2, 6)
    mean[..., 3] = 100.0
    std = torch.ones(2, 2, 6)
    decoded = decoder.decode_paths(
        context_s1=context,
        future_s1_paths=future,
        mean=mean,
        std=std,
        future_only=True,
    )
    assert tuple(decoded.shape) == (3, 2, 5, 2, 5)
    loss = decoded[..., 3].mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def _test_two_pass_gradient_replay() -> None:
    direct = _tiny_decoder(seed=9)
    replay = deepcopy(direct)
    context = torch.randint(0, 4, (1, 3, 2))
    future = torch.randint(0, 4, (4, 1, 5, 2))
    mean = torch.zeros(1, 2, 6)
    mean[..., 3] = 50.0
    std = torch.ones(1, 2, 6)

    direct_close = direct.decode_paths(
        context_s1=context,
        future_s1_paths=future,
        mean=mean,
        std=std,
        future_only=True,
    )[..., 3]
    direct_loss = (direct_close.mean(dim=0) ** 2).mean()
    direct_loss.backward()

    with torch.no_grad():
        replay_mean = replay.decode_paths(
            context_s1=context,
            future_s1_paths=future,
            mean=mean,
            std=std,
            future_only=True,
        )[..., 3].mean(dim=0)
    leaf = replay_mean.detach().requires_grad_(True)
    replay_loss = (leaf**2).mean()
    gradient_at_mean = torch.autograd.grad(replay_loss, leaf)[0].detach()
    sample_count = int(future.shape[0])
    for start in range(0, sample_count, 2):
        decoded = replay.decode_paths(
            context_s1=context,
            future_s1_paths=future[start : start + 2],
            mean=mean,
            std=std,
            future_only=True,
        )[..., 3]
        surrogate = (
            decoded * gradient_at_mean.unsqueeze(0) / float(sample_count)
        ).sum()
        surrogate.backward()

    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        direct.named_parameters(), replay.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert torch.allclose(
            parameter_a.grad,
            parameter_b.grad,
            atol=2.0e-6,
            rtol=2.0e-5,
        ), name_a


def _test_all_60_loss() -> None:
    weights = stretched_exponential_weights()
    loss_function = WeightedAll60Loss(weights)
    last = torch.full((2, 3), 100.0)
    true = torch.full((2, 60, 3), 101.0)
    predicted = true.clone()
    loss, errors = loss_function(predicted, true, last)
    assert tuple(errors.shape) == (2, 60, 3)
    assert float(loss.item()) == 0.0


def _write_source(directory: Path, *, model_kind: str, run_signature: str) -> None:
    directory.mkdir(parents=True)
    (directory / "resolved_config.json").write_text(
        __import__("json").dumps(
            {
                "model_kind": model_kind,
                "models": {
                    "dynamic_graph": {
                        "num_nodes": 93,
                        "graph": {"type": "dynamic", "num_heads": 1},
                        "heads": {"evaluation_horizons": [1, 5, 15, 30, 60]},
                    }
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_metadata.json").write_text(
        __import__("json").dumps(
            {
                "run_name": directory.name,
                "run_signature": run_signature,
                "best_epoch": 3,
                "best_score": 0.1,
            }
        ),
        encoding="utf-8",
    )
    torch.save({"model_state_dict": {}}, directory / "best_checkpoint.pt")


def _test_spec_grid() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        modern = root / "final_model_tokens"
        dense = root / "dense_tokens"
        _write_source(modern, model_kind="modern_tcn_token", run_signature="m")
        _write_source(dense, model_kind="dense_transformer_token", run_signature="d")
        specs = make_decoder_post_training_specs(
            modern_tcn_source_dir=modern,
            dense_transformer_source_dir=dense,
        )
        assert len(specs) == 2
        assert specs[0].config_signature != specs[1].config_signature
        for spec in specs:
            assert spec.config["sampling"]["sample_count"] == 10
            assert spec.config["decoder"]["forecasting_model_frozen"] is True
            assert spec.config["training"]["selection_split"] == "validation"



def _test_graph_hub_schema() -> None:
    import json
    import pandas as pd
    from src.evaluation.dynamic_graph_evaluation import load_evaluation_artifacts

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "decoder_run"
        run_dir.mkdir()
        resolved = {
            "model_family": "kronos_decoder_post_training_token",
            "models": {
                "dynamic_graph": {
                    "num_nodes": 2,
                    "graph": {
                        "type": "static_dynamic_mixture",
                        "num_heads": 1,
                        "add_self_loops": False,
                    },
                    "heads": {
                        "evaluation_horizons": [1, 5, 15, 30, 60],
                        "future_token_mode": "coarse_only",
                        "prediction_length": 60,
                    },
                    "temporal": {"type": "transformer"},
                    "spatial": {"num_layers": 1},
                }
            },
            "data": {"input_token_stream": "s1", "target_token_stream": "s1"},
            "training": {
                "early_stopping_metric": "weighted_all_60_clg_mae"
            },
        }
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved), encoding="utf-8"
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "best_epoch": 2,
                    "asset_cols": ["A", "B"],
                    "graph_type": "static_dynamic_mixture",
                    "graph_heads": 1,
                }
            ),
            encoding="utf-8",
        )
        prediction_result = {
            "y_pred": torch.ones(3, 5, 2, 1),
            "y_true": torch.ones(3, 5, 2, 1),
            "last_context_target": torch.ones(3, 2, 1),
            "horizons": [1, 5, 15, 30, 60],
            "channels": ["close"],
            "asset_cols": ["A", "B"],
            "output_space": "raw",
            "sample_idx": torch.arange(3),
            "origin_idx": torch.arange(3),
            "target_indices": torch.arange(15).reshape(3, 5),
        }
        graph = torch.full((3, 1, 2, 2), 0.5)
        graph[..., 0, 0] = 0.0
        graph[..., 1, 1] = 0.0
        graph[..., 0, 1] = 1.0
        graph[..., 1, 0] = 1.0
        torch.save(
            {"epoch": 2, "prediction_result": prediction_result},
            run_dir / "best_test_predictions.pt",
        )
        torch.save(
            {
                "epoch": 2,
                "graph_artifacts": {
                    "selected": graph,
                    "dynamic": graph,
                    "base": graph[0],
                    "per_layer": (graph,),
                    "per_layer_dynamic": (graph,),
                    "per_layer_base": (graph[0],),
                    "asset_cols": ["A", "B"],
                },
            },
            run_dir / "best_test_graphs.pt",
        )
        pd.DataFrame(
            [
                {
                    "metric": "cumulative_log_change_mae",
                    "horizon": 1,
                    "channel": "close",
                    "value": 0.0,
                }
            ]
        ).to_csv(run_dir / "best_test_metric_table.csv", index=False)
        artifacts = load_evaluation_artifacts(
            run_dir, split="test", policy="best", require_graph=True
        )
        assert tuple(artifacts.prediction_result["y_pred"].shape) == (3, 5, 2, 1)
        assert artifacts.graph_artifacts is not None

def main() -> None:
    _test_weight_curve()
    _test_decoder_shapes_and_gradients()
    _test_two_pass_gradient_replay()
    _test_all_60_loss()
    _test_spec_grid()
    _test_graph_hub_schema()
    print("Kronos decoder post-training contracts passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""CPU contracts for the exact Dimitri BaseDyGraph-V2 replication path."""

from pathlib import Path
import json
import os
import tempfile

import pandas as pd
from types import SimpleNamespace

import torch

from src.data.dimitri_anchor_tokens import (
    DIMITRI_CONTEXT_LENGTH,
    DIMITRI_SEQUENCE_LENGTH,
    DIMITRI_WINDOW_STRIDE,
    tokenize_clean_session_exact,
)
from src.models.dimitri_basedygraph_v2 import (
    DIMITRI_X0_EXPECTED_CHECKPOINT_EPOCH,
    DIMITRI_X0_EXPECTED_GLOBAL_STEP,
    DIMITRI_X0_EXPECTED_PARAMETER_COUNT,
    instantiate_exact_x0_model,
    load_dimitri_checkpoint,
    parameter_count,
    resolved_per_block_contract,
    verify_dimitri_source_snapshot,
)
from src.training.run_dimitri_basedygraph_replication import (
    _analysis_resolved_config,
)
from src.evaluation.dynamic_graph_evaluation import load_evaluation_artifacts


class FakeTokenizer:
    s1_bits = 10
    s2_bits = 10

    def encode(self, values: torch.Tensor, half: bool = True):
        summary = values.float().sum(dim=-1)
        coarse = ((summary * 17).round().long().abs() % 1024)
        fine = ((summary * 31).round().long().abs() % 1024)
        return coarse, fine


def test_source_and_architecture() -> None:
    hashes = verify_dimitri_source_snapshot()
    assert set(hashes) == {"model.py", "modules.py", "utilities.py", "data_module.py"}

    model = instantiate_exact_x0_model()
    assert parameter_count(model) == DIMITRI_X0_EXPECTED_PARAMETER_COUNT
    contract = resolved_per_block_contract(model.cfg)
    assert contract == {
        "activations": ["softmax", "softmax", "softmax", "sparsemax"],
        "num_edge_heads": [6, 6, 6, 1],
        "graph_hidden_dims": [192, 192, 192, 96],
    }
    assert model.cfg.spatial_module_type == "dual_fusion"
    assert model.cfg.scorer_value == "concat"
    assert model.cfg.spatial_value == "concat"
    assert model.cfg.fusion_window_size == 32
    assert model.cfg.fusion_fast_window == 4
    assert model.cfg.spatial_use_base is True
    assert model.cfg.graph_target_entropy_reg == 0.0
    assert model.cfg.graph_temporal_smooth_reg == 0.0

    model.eval()
    state_ids = torch.randint(0, 1024, (1, 93, 3))
    with torch.no_grad():
        output = model(state_ids)
    assert tuple(output["next_state_logits"].shape) == (1, 93, 2, 1024)
    graphs = output["block_graph_attns"]
    assert [tuple(values.shape) for values in graphs] == [
        (1, 3, 6, 93, 93),
        (1, 3, 6, 93, 93),
        (1, 3, 6, 93, 93),
        (1, 3, 1, 93, 93),
    ]
    for graph in graphs:
        torch.testing.assert_close(
            graph.sum(dim=-1),
            torch.ones_like(graph.sum(dim=-1)),
            atol=5.0e-6,
            rtol=0.0,
        )


def test_exact_anchor_token_window() -> None:
    # A clean 390-bar session gives starts 0,30,...,180: seven windows.
    session = torch.arange(390 * 3 * 6, dtype=torch.float32).reshape(390, 3, 6)
    session[..., 4] = 100.0
    session[..., 5] = 999.0
    s1, s2, mean, std, starts = tokenize_clean_session_exact(
        session,
        tokenizer=FakeTokenizer(),
        device="cpu",
        amount_index=5,
        encode_chunk=4,
    )
    assert starts == list(range(0, 181, DIMITRI_WINDOW_STRIDE))
    assert tuple(s1.shape) == (7, 3, DIMITRI_SEQUENCE_LENGTH)
    assert tuple(s2.shape) == tuple(s1.shape)
    assert tuple(mean.shape) == (7, 3, 6)
    assert tuple(std.shape) == (7, 3, 6)
    assert torch.equal(mean[..., 5], torch.zeros_like(mean[..., 5]))
    assert torch.equal(std[..., 5], torch.zeros_like(std[..., 5]))
    expected_std = session[:DIMITRI_CONTEXT_LENGTH, 0, 0].std()
    torch.testing.assert_close(std[0, 0, 0], expected_std, atol=0.0, rtol=0.0)


def test_analysis_schema() -> None:
    model = instantiate_exact_x0_model()
    per_block = resolved_per_block_contract(model.cfg)
    args = SimpleNamespace(
        max_epochs=120,
        patience=15,
        learning_rate=0.0012,
        weight_decay=0.0001,
        batch_size=1,
        seed=0,
    )
    config = _analysis_resolved_config(
        asset_cols=[f"A{i:02d}" for i in range(93)],
        source_hashes=verify_dimitri_source_snapshot(),
        checkpoint_sha256="fixture",
        args=args,
        per_block=per_block,
    )
    graph = config["models"]["dynamic_graph"]["graph"]
    assert graph["num_heads"] == 1
    assert graph["num_heads_per_block"] == [6, 6, 6, 1]
    assert config["models"]["dynamic_graph"]["heads"]["evaluation_horizons"] == [1]
    assert config["training"]["selection_split"] == "physical_test_December_2024"



def test_graph_hub_variable_head_schedule() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory)
        assets = ["A", "B", "C"]
        resolved = {
            "models": {
                "dynamic_graph": {
                    "num_nodes": 3,
                    "graph": {
                        "type": "dual_fusion",
                        "num_heads": 1,
                        "num_heads_per_block": [2, 2, 1],
                        "add_self_loops": False,
                    },
                    "heads": {"evaluation_horizons": [1]},
                }
            },
            "training": {"early_stopping_metric": "test_next_s1_accuracy"},
        }
        metadata = {
            "status": "completed",
            "best_epoch": 2,
            "asset_cols": assets,
        }
        (run_dir / "resolved_config.json").write_text(json.dumps(resolved))
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

        windows = 2
        prediction_result = {
            "y_pred": torch.zeros(windows, 1, 3, 1),
            "y_true": torch.zeros(windows, 1, 3, 1),
            "last_context_target": torch.zeros(windows, 3, 1),
            "sample_idx": torch.arange(windows),
            "origin_idx": torch.tensor([208, 208]),
            "target_indices": torch.tensor([[209], [209]]),
            "window_date": ["2024-10-01", "2024-10-02"],
            "asset_cols": assets,
            "channels": ["s1"],
            "horizons": [1],
            "output_space": "token_id",
        }
        torch.save(
            {"epoch": 2, "prediction_result": prediction_result},
            run_dir / "best_validation_predictions.pt",
        )

        def row_graph(heads: int) -> torch.Tensor:
            values = torch.rand(windows, heads, 3, 3)
            return values / values.sum(dim=-1, keepdim=True)

        per_layer = [row_graph(2), row_graph(2), row_graph(1)]
        torch.save(
            {
                "epoch": 2,
                "graph_artifacts": {
                    "selected": per_layer[-1],
                    "per_layer": per_layer,
                    "graph_orientation": "row=target,column=source",
                    "asset_cols": assets,
                },
            },
            run_dir / "best_validation_graphs.pt",
        )
        pd.DataFrame(
            [{"metric": "next_s1_accuracy", "horizon": 1, "channel": "s1", "value": 0.1}]
        ).to_csv(run_dir / "best_validation_metric_table.csv", index=False)

        artifacts = load_evaluation_artifacts(
            run_dir,
            split="validation",
            policy=None,
            require_graph=True,
            require_metrics=True,
        )
        assert artifacts.info.num_heads == 1
        assert artifacts.info.num_heads_per_layer == (2, 2, 1)
        assert [int(values.shape[1]) for values in artifacts.graph_artifacts["per_layer"]] == [2, 2, 1]

def test_optional_exact_checkpoint() -> None:
    value = os.environ.get("DIMITRI_X0_CHECKPOINT")
    if not value:
        return
    model = instantiate_exact_x0_model()
    checkpoint = load_dimitri_checkpoint(model, Path(value))
    assert int(checkpoint["epoch"]) == DIMITRI_X0_EXPECTED_CHECKPOINT_EPOCH
    assert int(checkpoint["global_step"]) == DIMITRI_X0_EXPECTED_GLOBAL_STEP


def main() -> None:
    test_source_and_architecture()
    test_exact_anchor_token_window()
    test_analysis_schema()
    test_graph_hub_variable_head_schedule()
    test_optional_exact_checkpoint()
    print("Dimitri BaseDyGraph-V2 exact replication contracts passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Contracts for flexible ModernTCN and final continuous-run inference."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

import torch
from torch import nn

from src.data.data_generator import WindowContextNormaliser
from src.models.continuous_forecaster import (
    ContinuousForecaster,
    evaluate_saved_continuous_forecaster_run,
)
from src.models.modern_tcn import ModernTCNBaseline
from src.training import run_continuous_forecaster as continuous_runner
from src.utils.config import load_yaml


def _synthetic_split() -> dict:
    torch.manual_seed(981)
    channels = ["open", "high", "low", "close", "volume", "amount"]
    samples = []
    for day_index in range(2):
        base = 75.0 + torch.cumsum(
            0.015 * torch.randn(390, 4),
            dim=0,
        )
        open_price = base + 0.005 * torch.randn_like(base)
        close = base + 0.005 * torch.randn_like(base)
        high = torch.maximum(open_price, close) + 0.01
        low = torch.minimum(open_price, close) - 0.01
        volume = 1000.0 + 10.0 * torch.rand_like(base)
        amount = torch.zeros_like(base)
        values = torch.stack(
            [open_price, high, low, close, volume, amount],
            dim=-1,
        )
        samples.append((values, {}, f"2024-02-{day_index + 1:02d}"))
    return {
        "samples": samples,
        "asset_cols": ["A", "B", "C", "D"],
        "channels": channels,
    }


def _test_modern_tcn_checkpoint_configuration() -> None:
    normaliser = WindowContextNormaliser(
        eps=1.0e-8,
        clip=False,
        clip_min=-5.0,
        clip_max=5.0,
        apply_to_target=True,
        include_stats=True,
    )
    source = ModernTCNBaseline(
        context_length=60,
        horizons=[1, 5, 15, 30, 60],
        input_channels=["open", "high", "low", "close", "volume"],
        target_channels=["close"],
        stride=15,
        patch_size=4,
        patch_stride=2,
        hidden_dim=32,
        ffn_ratio=1,
        num_blocks=1,
        large_kernel=51,
        small_kernel=5,
        dropout=0.05,
        head_dropout=0.0,
        variable_layout="joint",
        revin=False,
        normaliser=normaliser,
        temporal_encoding_enabled=False,
    )
    source.asset_cols = ["A", "B"]
    source.num_assets = 2
    source.num_variables = 10
    source.model = nn.Linear(1, 1)

    checkpoint = {
        "epoch": 7,
        "model_state_dict": source.model.state_dict(),
        "best_validation_loss": 0.123,
        **source._checkpoint_metadata(),
    }

    target = ModernTCNBaseline(
        context_length=60,
        horizons=[1, 5, 15, 30, 60],
        input_channels=["close"],
        target_channels=["close"],
        stride=15,
        hidden_dim=64,
        variable_layout="per_asset",
        revin=True,
    )

    def fake_builder(self: ModernTCNBaseline) -> nn.Module:
        self.model = nn.Linear(1, 1)
        return self.model

    with TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        with patch.object(
            ModernTCNBaseline,
            "_build_official_model",
            fake_builder,
        ):
            target.load_checkpoint(checkpoint_path, device="cpu")
            if target.variable_layout != "joint":
                raise AssertionError("Checkpoint layout was not restored.")
            if target.hidden_dim != 32:
                raise AssertionError("Checkpoint hidden dimension was not restored.")
            if target.patch_size != 4 or target.patch_stride != 2:
                raise AssertionError("Checkpoint patch geometry was not restored.")
            if target.input_channels != [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:
                raise AssertionError("Checkpoint input channels were not restored.")
            if target.num_variables != 10:
                raise AssertionError("Checkpoint variable count was not restored.")

            loaded = ModernTCNBaseline.from_checkpoint(
                checkpoint_path,
                device="cpu",
            )
            if loaded.variable_layout != "joint" or loaded.hidden_dim != 32:
                raise AssertionError("from_checkpoint did not restore architecture.")


def _test_continuous_saved_run_evaluation() -> None:
    split = _synthetic_split()
    project_root = Path(__file__).resolve().parents[1]
    config = deepcopy(
        load_yaml(project_root / "configs" / "continuous_forecasting.yaml")
    )
    config["model"]["temporal"].update(
        {
            "type": "transformer",
            "d_model": 16,
            "num_layers": 1,
            "num_heads": 4,
            "feedforward_multiplier": 2,
            "dropout": 0.0,
            "relative_position_embedding": True,
            "session_position_encoding": True,
        }
    )
    config["model"]["graph"].update(
        {
            "type": "free_static",
            "num_heads": 1,
            "hidden_dim": 16,
            "activation": "softmax",
            "add_self_loops": False,
        }
    )
    config["model"]["spatial"].update(
        {
            "num_layers": 1,
            "feedforward_multiplier": 2,
            "dropout": 0.0,
            "gate_type": "learned_scalar",
            "initial_beta": 0.5,
        }
    )
    config["training"]["mixed_precision"] = False
    config["training"]["validation_batch_size"] = 8
    config["training"]["num_workers"] = 0
    continuous_runner.validate_config(config)

    model_config = continuous_runner._model_config(
        config,
        num_nodes=len(split["asset_cols"]),
    )
    model = ContinuousForecaster(model_config)

    with TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory) / "continuous_dynamic"
        run_dir.mkdir(parents=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_name": "continuous_dynamic",
                    "best_epoch": 2,
                    "asset_cols": split["asset_cols"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        torch.save(
            {
                "epoch": 2,
                "best_epoch": 2,
                "model_state_dict": model.state_dict(),
                "resolved_config": config,
            },
            run_dir / "best_checkpoint.pt",
        )

        generated = evaluate_saved_continuous_forecaster_run(
            run_dir=run_dir,
            train_split=split,
            evaluation_split=split,
            split_name="test",
            run_inference=True,
            device="cpu",
            batch_size=8,
            num_workers=0,
            bootstrap=False,
        )
        expected_windows = 2 * 19
        if tuple(generated.prediction_result["y_pred"].shape) != (
            expected_windows,
            5,
            4,
            1,
        ):
            raise AssertionError("Unexpected saved continuous prediction shape.")
        for path in (
            generated.prediction_path,
            generated.graph_path,
            generated.metric_path,
            generated.diagnostics_path,
        ):
            if path is None or not Path(path).is_file():
                raise AssertionError(f"Missing generated artefact: {path}")

        reloaded = evaluate_saved_continuous_forecaster_run(
            run_dir=run_dir,
            train_split=split,
            evaluation_split=split,
            split_name="test",
            run_inference=False,
            bootstrap=False,
        )
        torch.testing.assert_close(
            generated.prediction_result["y_pred"],
            reloaded.prediction_result["y_pred"],
            atol=0.0,
            rtol=0.0,
        )


def main() -> None:
    _test_modern_tcn_checkpoint_configuration()
    _test_continuous_saved_run_evaluation()
    print("Baseline and final-model inference contracts passed.")


if __name__ == "__main__":
    main()

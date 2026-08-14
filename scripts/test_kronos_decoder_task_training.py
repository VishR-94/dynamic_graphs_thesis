from __future__ import annotations

"""CPU contracts for task-specific Kronos-initialised decoder training."""

from pathlib import Path
import tempfile

import torch
from torch import nn

from src.models.kronos_decoder_post_training import TrainableKronosCoarseDecoder
from src.training.kronos_decoder_post_training_specs import (
    make_decoder_post_training_specs,
)
from src.training.kronos_decoder_task_training_specs import (
    make_decoder_task_training_specs,
)
from src.training.run_kronos_decoder_post_training import (
    _WarmupReduceOnPlateauScheduler,
    _build_decoder_optimizer_and_scheduler,
)


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


def _write_source(directory: Path, *, model_kind: str, signature: str) -> None:
    import json

    directory.mkdir(parents=True)
    (directory / "resolved_config.json").write_text(
        json.dumps(
            {
                "model_kind": model_kind,
                "models": {
                    "dynamic_graph": {
                        "num_nodes": 93,
                        "graph": {"type": "dynamic", "num_heads": 1},
                        "heads": {
                            "evaluation_horizons": [1, 5, 15, 30, 60]
                        },
                    }
                },
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    (directory / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_name": directory.name,
                "run_signature": signature,
                "best_epoch": 3,
                "best_score": 0.1,
            }
        ),
        encoding="utf-8",
    )
    torch.save({"model_state_dict": {}}, directory / "best_checkpoint.pt")


def _optimizer_step(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _WarmupReduceOnPlateauScheduler,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in decoder.parameters()).backward()
    optimizer.step()
    scheduler.step_after_optimizer()


def _test_task_specs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        modern = root / "final_model_tokens"
        dense = root / "dense_tokens"
        _write_source(modern, model_kind="modern_tcn_token", signature="m")
        _write_source(dense, model_kind="dense_transformer_token", signature="d")

        task_specs = make_decoder_task_training_specs(
            modern_tcn_source_dir=modern,
            dense_transformer_source_dir=dense,
        )
        onecycle_specs = make_decoder_post_training_specs(
            modern_tcn_source_dir=modern,
            dense_transformer_source_dir=dense,
        )

        assert len(task_specs) == 2
        assert len({spec.config_signature for spec in task_specs}) == 2
        assert not (
            {spec.config_signature for spec in task_specs}
            & {spec.config_signature for spec in onecycle_specs}
        )

        for spec in task_specs:
            config = spec.config
            assert config["experiment_family"] == (
                "kronos_task_specific_coarse_decoder_training"
            )
            assert config["decoder"]["forecasting_model_frozen"] is True
            assert config["decoder"]["conservative_fine_tuning"] is False
            training = config["training"]
            assert training["optimizer"] == "adamw"
            assert training["scheduler"] == "warmup_reduce_on_plateau"
            assert training["max_learning_rate"] == 5.0e-4
            assert training["initial_learning_rate"] == 5.0e-5
            assert training["minimum_learning_rate"] == 5.0e-6
            assert training["warmup_epochs"] == 1
            assert training["plateau_factor"] == 0.5
            assert training["plateau_patience"] == 3
            assert training["max_epochs"] == 100
            assert training["patience"] == 10
            assert training["selection_split"] == "validation"


def _test_warmup_plateau_scheduler_and_resume() -> None:
    decoder = _tiny_decoder(seed=11)
    training = {
        "optimizer": "adamw",
        "max_learning_rate": 5.0e-4,
        "initial_learning_rate": 5.0e-5,
        "minimum_learning_rate": 5.0e-6,
        "weight_decay": 0.1,
        "adam_betas": [0.9, 0.999],
        "adam_eps": 1.0e-8,
        "scheduler": "warmup_reduce_on_plateau",
        "warmup_start_learning_rate": 5.0e-5,
        "warmup_epochs": 1,
        "plateau_factor": 0.5,
        "plateau_patience": 3,
        "plateau_threshold": 0.0,
        "max_epochs": 100,
    }
    optimizer, scheduler = _build_decoder_optimizer_and_scheduler(
        decoder,
        training,
        steps_per_epoch=4,
    )
    assert isinstance(scheduler, _WarmupReduceOnPlateauScheduler)
    assert abs(float(optimizer.param_groups[0]["lr"]) - 5.0e-5) < 1.0e-12

    observed = [float(optimizer.param_groups[0]["lr"])]
    for _ in range(4):
        _optimizer_step(decoder, optimizer, scheduler)
        observed.append(float(optimizer.param_groups[0]["lr"]))
    assert all(left <= right for left, right in zip(observed, observed[1:]))
    assert abs(observed[-1] - 5.0e-4) < 1.0e-12
    assert scheduler.warmup_complete is True

    assert scheduler.step_after_validation(1.0) is False
    assert scheduler.step_after_validation(1.1) is False
    assert scheduler.step_after_validation(1.1) is False
    assert scheduler.step_after_validation(1.1) is True
    assert abs(float(optimizer.param_groups[0]["lr"]) - 2.5e-4) < 1.0e-12
    assert scheduler.reductions == 1

    state = scheduler.state_dict()
    decoder_2 = _tiny_decoder(seed=13)
    optimizer_2, scheduler_2 = _build_decoder_optimizer_and_scheduler(
        decoder_2,
        training,
        steps_per_epoch=4,
    )
    scheduler_2.load_state_dict(state)
    assert scheduler_2.state_dict() == state
    assert abs(float(optimizer_2.param_groups[0]["lr"]) - 2.5e-4) < 1.0e-12

    assert scheduler_2.step_after_validation(0.9) is False
    assert scheduler_2.bad_epochs == 0
    assert scheduler_2.best_metric == 0.9


def main() -> None:
    _test_task_specs()
    _test_warmup_plateau_scheduler_and_resume()
    print("Kronos task-specific decoder-training contracts passed.")


if __name__ == "__main__":
    main()

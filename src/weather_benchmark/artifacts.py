from __future__ import annotations

"""Atomic artifact and checkpoint utilities for long Colab runs."""

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_save(values: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(values), handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def atomic_torch_save(values: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    os.replace(temporary, path)


def atomic_history_save(history: Sequence[Mapping[str, Any]], run_dir: Path) -> None:
    rows = [dict(value) for value in history]
    atomic_json_save({"epochs": rows}, run_dir / "epoch_history.json")
    frame = pd.DataFrame(rows)
    path = run_dir / "epoch_history.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_metric_csv(metrics: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for scope, values in metrics.items():
        if not isinstance(values, Mapping):
            continue
        if not all(isinstance(value, (int, float, np.number)) for value in values.values()):
            continue
        for name, value in values.items():
            flattened.append({"scope": scope, "metric": name, "value": value})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(flattened)
    os.replace(temporary, path)


def safe_torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> dict[str, Any]:
    values: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        values["torch_cuda"] = torch.cuda.get_rng_state_all()
    return values


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Return an RNG-state value as a contiguous CPU uint8 tensor."""

    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def restore_rng_state(values: Mapping[str, Any]) -> None:
    if "python" in values:
        random.setstate(values["python"])
    if "numpy" in values:
        np.random.set_state(values["numpy"])
    if "torch_cpu" in values:
        # Checkpoints are loaded with map_location=<training device>. On a CUDA
        # run that also remaps the saved CPU RNG tensor to CUDA, but
        # torch.set_rng_state accepts a CPU ByteTensor only.
        torch.set_rng_state(_cpu_byte_tensor(values["torch_cpu"]))
    if "torch_cuda" in values and torch.cuda.is_available():
        cuda_states = [
            _cpu_byte_tensor(state)
            for state in values["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(
            cuda_states[: torch.cuda.device_count()]
        )


def git_value(project_root: Path, arguments: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_manifest(project_root: Path, device: torch.device) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": (
            bool(torch.backends.cudnn.allow_tf32)
            if hasattr(torch.backends.cudnn, "allow_tf32")
            else None
        ),
        "cuda_matmul_allow_tf32": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "matmul")
            else None
        ),
        "deterministic_algorithms_enabled": (
            bool(torch.are_deterministic_algorithms_enabled())
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None
        ),
        "deterministic_algorithms_warn_only": (
            bool(torch.is_deterministic_algorithms_warn_only_enabled())
            if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
            else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "project_git_commit": git_value(project_root, ["rev-parse", "HEAD"]),
        "project_git_branch": git_value(project_root, ["branch", "--show-current"]),
        "project_git_dirty": bool(
            git_value(project_root, ["status", "--porcelain"])
        ),
    }

from __future__ import annotations

"""Exact ``tokens_anchor_amt0`` construction used by Dimitri's x0jhc0tx run.

This is intentionally separate from the dissertation's origin-aligned 60+60
cache.  It preserves the physical train/val/test file membership and follows
``tokenize_forecast_windows.ipynb`` from the supplied experiment archive:

* drop the first raw row of every session;
* construct 180-context + 30-continuation windows with stride 30;
* zero Amount before normalisation;
* use context-only mean and ``torch.std`` (sample standard deviation);
* normalise the complete 210-position sequence with that frame;
* clip to [-5, 5] and retain native 1,024-way Kronos s1/s2 IDs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import os

import torch


DIMITRI_CONTEXT_LENGTH = 180
DIMITRI_CONTINUATION_LENGTH = 30
DIMITRI_SEQUENCE_LENGTH = 210
DIMITRI_WINDOW_STRIDE = 30
DIMITRI_DROP_OPEN_ROWS = 1
DIMITRI_CLIP = 5.0
DIMITRI_EPS = 1.0e-5
DIMITRI_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
# Dimitri did not record a Hugging Face revision.  The project-pinned revision
# is used initially; frozen-checkpoint parity is the definitive compatibility
# test and fails before retraining if the resulting IDs differ.
DIMITRI_TOKENIZER_REVISION = "9ef143b98ee3c2488eebd85404e0c215c112b46a"
DIMITRI_CHANNELS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)
DIMITRI_EXPECTED_ASSETS = 93
DIMITRI_EXPECTED_WINDOWS = {
    "train": 1309,  # 187 physical sessions x 7 windows
    "val": 294,     # 42 physical sessions x 7 windows
    "test": 140,    # 20 physical sessions x 7 windows
}


@dataclass(frozen=True)
class DimitriAnchorTokenSummary:
    split: str
    path: Path
    windows: int
    assets: int
    sequence_length: int
    minimum_s1: int
    maximum_s1: int
    minimum_s2: int
    maximum_s2: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "path": str(self.path),
            "windows": self.windows,
            "assets": self.assets,
            "sequence_length": self.sequence_length,
            "minimum_s1": self.minimum_s1,
            "maximum_s1": self.maximum_s1,
            "minimum_s2": self.minimum_s2,
            "maximum_s2": self.maximum_s2,
            "sha256": self.sha256,
        }


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_physical_candle_split(data_dir: str | Path, split: str) -> dict[str, Any]:
    """Load a physical file directly, with no project chronological repartition."""
    split = str(split).strip().lower()
    if split not in DIMITRI_EXPECTED_WINDOWS:
        raise ValueError("split must be 'train', 'val', or 'test'.")
    path = Path(data_dir).expanduser() / f"{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}.")
    for key in ("samples", "asset_cols", "channels"):
        if key not in payload:
            raise KeyError(f"{path} is missing {key!r}.")
    observed_channels = tuple(str(value).lower() for value in payload["channels"])
    if observed_channels != DIMITRI_CHANNELS:
        raise ValueError(
            f"{path} channel order differs: {observed_channels}; expected "
            f"{DIMITRI_CHANNELS}."
        )
    if len(payload["asset_cols"]) != DIMITRI_EXPECTED_ASSETS:
        raise ValueError(
            f"{path} contains {len(payload['asset_cols'])} assets; expected "
            f"{DIMITRI_EXPECTED_ASSETS}."
        )
    return payload


def exact_window_starts(
    session_length: int,
    *,
    sequence_length: int = DIMITRI_SEQUENCE_LENGTH,
    stride: int = DIMITRI_WINDOW_STRIDE,
) -> list[int]:
    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive.")
    if session_length < sequence_length:
        return []
    return list(range(0, session_length - sequence_length + 1, stride))


@torch.inference_mode()
def tokenize_clean_session_exact(
    clean_session: torch.Tensor,
    *,
    tokenizer: Any,
    device: torch.device | str,
    amount_index: int,
    context_length: int = DIMITRI_CONTEXT_LENGTH,
    continuation_length: int = DIMITRI_CONTINUATION_LENGTH,
    stride: int = DIMITRI_WINDOW_STRIDE,
    clip: float = DIMITRI_CLIP,
    eps: float = DIMITRI_EPS,
    encode_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Tokenise one clean raw session exactly as Dimitri's helper.

    Args:
        clean_session:
            Candle tensor ``[T,N,6]`` after dropping the first raw row.

    Returns:
        ``s1, s2, mean, std, starts``.  Token tensors are ``[W,N,210]``
        and the context normalisation statistics are ``[W,N,6]``.
    """
    values = torch.as_tensor(clean_session).detach().cpu().float()
    if values.ndim != 3 or values.shape[-1] != 6:
        raise ValueError(
            f"clean_session must have shape [T,N,6], got {tuple(values.shape)}."
        )
    if values.shape[1] != DIMITRI_EXPECTED_ASSETS and values.shape[1] <= 0:
        raise ValueError("clean_session contains no assets.")
    if not torch.isfinite(values).all():
        raise ValueError("clean_session contains non-finite values.")
    if not 0 <= amount_index < 6:
        raise ValueError("amount_index lies outside the six channels.")
    if encode_chunk <= 0:
        raise ValueError("encode_chunk must be positive.")

    # Exact notebook layout: [T,N,D] -> [N,T,D].
    values = values.permute(1, 0, 2).contiguous()
    values[..., amount_index] = 0.0

    sequence_length = context_length + continuation_length
    starts = exact_window_starts(
        values.shape[1],
        sequence_length=sequence_length,
        stride=stride,
    )
    if not starts:
        raise ValueError(
            f"No {sequence_length}-bar window fits in a session of length "
            f"{values.shape[1]}."
        )

    windows = torch.stack(
        [values[:, start : start + sequence_length, :] for start in starts],
        dim=0,
    )  # [W,N,T,D]
    context = windows[:, :, :context_length, :]

    # Exact notebook semantics: torch.std with its default correction=1.
    mean = context.mean(dim=2, keepdim=True)
    std = context.std(dim=2, keepdim=True)
    normalised = ((windows - mean) / (std + eps)).clamp(-clip, clip)

    num_windows, num_assets, _, num_channels = normalised.shape
    flat = normalised.reshape(num_windows * num_assets, sequence_length, num_channels)
    s1 = torch.empty(flat.shape[0], sequence_length, dtype=torch.long)
    s2 = torch.empty_like(s1)

    target_device = torch.device(device)
    for start in range(0, flat.shape[0], encode_chunk):
        stop = min(start + encode_chunk, flat.shape[0])
        coarse, fine = tokenizer.encode(flat[start:stop].to(target_device), half=True)
        s1[start:stop] = coarse.detach().cpu().long()
        s2[start:stop] = fine.detach().cpu().long()

    return (
        s1.reshape(num_windows, num_assets, sequence_length),
        s2.reshape(num_windows, num_assets, sequence_length),
        mean.squeeze(2).contiguous(),
        std.squeeze(2).contiguous(),
        starts,
    )


def generate_dimitri_anchor_token_split(
    *,
    raw_split: Mapping[str, Any],
    split_name: str,
    tokenizer: Any,
    output_path: str | Path,
    device: torch.device | str,
    tokenizer_id: str = DIMITRI_TOKENIZER_ID,
    tokenizer_revision: str = DIMITRI_TOKENIZER_REVISION,
    encode_chunk: int = 512,
    show_progress: bool = True,
) -> DimitriAnchorTokenSummary:
    """Generate one exact physical-split cache and save it atomically."""
    split_name = str(split_name).strip().lower()
    if split_name not in DIMITRI_EXPECTED_WINDOWS:
        raise ValueError("split_name must be 'train', 'val', or 'test'.")
    channels = tuple(str(value).lower() for value in raw_split["channels"])
    if channels != DIMITRI_CHANNELS:
        raise ValueError(f"Channel order differs: {channels}.")
    amount_index = channels.index("amount")

    all_s1: list[torch.Tensor] = []
    all_s2: list[torch.Tensor] = []
    all_mean: list[torch.Tensor] = []
    all_std: list[torch.Tensor] = []
    dates: list[Any] = []
    starts: list[int] = []
    sample_indices: list[int] = []

    samples = list(raw_split["samples"])
    for sample_index, sample in enumerate(samples, start=1):
        if not isinstance(sample, (tuple, list)) or len(sample) < 3:
            raise TypeError(f"Malformed {split_name} session at index {sample_index - 1}.")
        candle_values, _auxiliary, session_date = sample[:3]
        candle_values = torch.as_tensor(candle_values)
        if candle_values.ndim != 3 or candle_values.shape[-1] != 6:
            raise ValueError(
                f"{split_name} session {session_date} has shape "
                f"{tuple(candle_values.shape)}, expected [T,N,6]."
            )
        if candle_values.shape[0] <= DIMITRI_DROP_OPEN_ROWS:
            raise ValueError(f"{split_name} session {session_date} is too short.")

        clean = candle_values[DIMITRI_DROP_OPEN_ROWS:].contiguous()
        s1, s2, mean, std, session_starts = tokenize_clean_session_exact(
            clean,
            tokenizer=tokenizer,
            device=device,
            amount_index=amount_index,
            encode_chunk=encode_chunk,
        )
        all_s1.append(s1)
        all_s2.append(s2)
        all_mean.append(mean)
        all_std.append(std)
        dates.extend([session_date] * len(session_starts))
        starts.extend(session_starts)
        sample_indices.extend([sample_index - 1] * len(session_starts))

        if show_progress and (
            sample_index % 20 == 0 or sample_index == len(samples)
        ):
            print(
                f"[{split_name}] tokenised {sample_index}/{len(samples)} sessions",
                flush=True,
            )

    s1 = torch.cat(all_s1, dim=0).contiguous()
    s2 = torch.cat(all_s2, dim=0).contiguous()
    norm_mean = torch.cat(all_mean, dim=0).contiguous()
    norm_std = torch.cat(all_std, dim=0).contiguous()

    payload = {
        # Exact original fields.
        "s1": s1,
        "s2": s2,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "window_date": dates,
        "window_start": torch.tensor(starts, dtype=torch.long),
        "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
        "asset_cols": list(raw_split["asset_cols"]),
        "s1_bits": int(getattr(tokenizer, "s1_bits")),
        "s2_bits": int(getattr(tokenizer, "s2_bits")),
        "T": DIMITRI_SEQUENCE_LENGTH,
        "F": int(s1.shape[1]),
        "channels": list(raw_split["channels"]),
        "drop_open": DIMITRI_DROP_OPEN_ROWS,
        "zero_amount": True,
        "tokenizer": tokenizer_id,
        "grain": raw_split.get("grain", "1min"),
        "market_open": raw_split.get("market_open", "09:30"),
        "market_close": raw_split.get("market_close", "16:00"),
        "windowed": True,
        "anchor": True,
        "context_len": DIMITRI_CONTEXT_LENGTH,
        "pred_len": DIMITRI_CONTINUATION_LENGTH,
        "window": DIMITRI_SEQUENCE_LENGTH,
        "stride": DIMITRI_WINDOW_STRIDE,
        # Additional reproducibility metadata; these do not affect training.
        "tokenizer_revision": tokenizer_revision,
        "std_correction": 1,
        "physical_split_membership_preserved": True,
        "split_name": split_name,
    }

    output_path = Path(output_path)
    _atomic_torch_save(payload, output_path)
    return validate_dimitri_anchor_token_split(
        output_path,
        split_name=split_name,
        require_expected_window_count=True,
    )


def validate_dimitri_anchor_token_split(
    path: str | Path,
    *,
    split_name: str | None = None,
    require_expected_window_count: bool = False,
) -> DimitriAnchorTokenSummary:
    """Validate shapes, ID ranges, ordering metadata and expected window count."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary in {path}.")
    required = (
        "s1",
        "s2",
        "norm_mean",
        "norm_std",
        "window_date",
        "window_start",
        "sample_idx",
        "asset_cols",
        "channels",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing fields {missing}.")

    s1 = torch.as_tensor(payload["s1"]).long()
    s2 = torch.as_tensor(payload["s2"]).long()
    if s1.ndim != 3 or tuple(s2.shape) != tuple(s1.shape):
        raise ValueError(
            f"Expected matching [W,N,T] token tensors, got "
            f"s1={tuple(s1.shape)}, s2={tuple(s2.shape)}."
        )
    windows, assets, sequence_length = map(int, s1.shape)
    if assets != DIMITRI_EXPECTED_ASSETS:
        raise ValueError(f"{path} contains {assets} assets; expected 93.")
    if sequence_length != DIMITRI_SEQUENCE_LENGTH:
        raise ValueError(
            f"{path} has sequence length {sequence_length}; expected 210."
        )
    if tuple(torch.as_tensor(payload["norm_mean"]).shape) != (windows, assets, 6):
        raise ValueError(f"{path} has an invalid norm_mean shape.")
    if tuple(torch.as_tensor(payload["norm_std"]).shape) != (windows, assets, 6):
        raise ValueError(f"{path} has an invalid norm_std shape.")
    if len(payload["window_date"]) != windows:
        raise ValueError(f"{path} window_date length differs from W.")
    if int(torch.as_tensor(payload["window_start"]).numel()) != windows:
        raise ValueError(f"{path} window_start length differs from W.")
    if int(torch.as_tensor(payload["sample_idx"]).numel()) != windows:
        raise ValueError(f"{path} sample_idx length differs from W.")
    if len(payload["asset_cols"]) != assets:
        raise ValueError(f"{path} asset labels differ from N.")
    if tuple(str(value).lower() for value in payload["channels"]) != DIMITRI_CHANNELS:
        raise ValueError(f"{path} channel order differs from the exact contract.")

    for name, values in (("s1", s1), ("s2", s2)):
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= 1024:
            raise ValueError(
                f"{path} {name} IDs lie outside [0,1023]: [{minimum},{maximum}]."
            )

    resolved_split = str(split_name or payload.get("split_name") or path.stem)
    if require_expected_window_count and resolved_split in DIMITRI_EXPECTED_WINDOWS:
        expected = DIMITRI_EXPECTED_WINDOWS[resolved_split]
        if windows != expected:
            raise AssertionError(
                f"{path} has {windows} windows; the exact {resolved_split} "
                f"contract expects {expected}. Physical split membership or "
                "session cleaning therefore differs from Dimitri's run."
            )

    return DimitriAnchorTokenSummary(
        split=resolved_split,
        path=path,
        windows=windows,
        assets=assets,
        sequence_length=sequence_length,
        minimum_s1=int(s1.min().item()),
        maximum_s1=int(s1.max().item()),
        minimum_s2=int(s2.min().item()),
        maximum_s2=int(s2.max().item()),
        sha256=file_sha256(path),
    )


def ensure_dimitri_anchor_token_caches(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    device: torch.device | str,
    tokenizer_id: str = DIMITRI_TOKENIZER_ID,
    tokenizer_revision: str = DIMITRI_TOKENIZER_REVISION,
    encode_chunk: int = 512,
    force: bool = False,
) -> tuple[DimitriAnchorTokenSummary, ...]:
    """Generate or validate exact train/val/test caches and write a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[DimitriAnchorTokenSummary] = []

    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.pt"
        if path.is_file() and not force:
            summary = validate_dimitri_anchor_token_split(
                path,
                split_name=split,
                require_expected_window_count=True,
            )
            print(f"[{split}] existing exact cache validated: {path}")
        else:
            raw_split = load_physical_candle_split(data_dir, split)
            summary = generate_dimitri_anchor_token_split(
                raw_split=raw_split,
                split_name=split,
                tokenizer=tokenizer,
                output_path=path,
                device=device,
                tokenizer_id=tokenizer_id,
                tokenizer_revision=tokenizer_revision,
                encode_chunk=encode_chunk,
            )
        summaries.append(summary)

    _atomic_json_save(
        {
            "contract": "dimitri_tokens_anchor_amt0_v1",
            "context_length": DIMITRI_CONTEXT_LENGTH,
            "teacher_forced_continuation_length": DIMITRI_CONTINUATION_LENGTH,
            "sequence_length": DIMITRI_SEQUENCE_LENGTH,
            "stride": DIMITRI_WINDOW_STRIDE,
            "drop_open_rows": DIMITRI_DROP_OPEN_ROWS,
            "zero_amount": True,
            "clip": DIMITRI_CLIP,
            "eps": DIMITRI_EPS,
            "std_correction": 1,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "physical_split_membership_preserved": True,
            "splits": [summary.to_dict() for summary in summaries],
        },
        output_dir / "token_cache_manifest.json",
    )
    return tuple(summaries)

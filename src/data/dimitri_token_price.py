from __future__ import annotations

"""Configurable Dimitri-style token windows for direct price prediction.

The exact x0jhc0tx replication uses a 180-minute anchor context, 30-minute
teacher-forced continuation and stride 30.  This module keeps those semantics
but makes the context length explicit so a 60-minute controlled ablation can be
created without changing the model or training runner.

For every window:

* Amount is set to zero;
* context-only mean and sample standard deviation are calculated over the first
  ``context_length`` rows;
* those statistics normalise the complete context+continuation sequence;
* values are clipped to [-5,5] before frozen Kronos tokenisation;
* native coarse ``s1`` IDs are retained for the model;
* raw Close values are reconstructed from the selected split at load time and
  are never derived from the tokenizer decoder.

Two split contracts are supported. ``physical`` preserves the membership of the
three stored files, matching Dimitri's original experiment. ``canonical`` first
combines those files and applies the dissertation's chronological boundaries via
``load_candle_splits``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import os

import torch
from torch.utils.data import Dataset

from src.data.load_candle_data import load_candle_splits
from src.data.dimitri_anchor_tokens import (
    DIMITRI_CHANNELS,
    DIMITRI_CLIP,
    DIMITRI_DROP_OPEN_ROWS,
    DIMITRI_EPS,
    DIMITRI_EXPECTED_ASSETS,
    DIMITRI_TOKENIZER_ID,
    DIMITRI_TOKENIZER_REVISION,
    file_sha256,
    load_physical_candle_split,
)


DIMITRI_SPLIT_MODES = ("physical", "canonical")


def normalise_split_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in DIMITRI_SPLIT_MODES:
        raise ValueError(
            f"split_mode must be one of {DIMITRI_SPLIT_MODES}; got {value!r}."
        )
    return mode


def load_token_price_splits(
    data_dir: str | Path,
    *,
    split_mode: str = "physical",
) -> dict[str, dict[str, Any]]:
    """Load either Dimitri's physical files or the canonical project splits.

    ``physical`` treats ``train.pt``, ``val.pt`` and ``test.pt`` as the final
    membership, matching Dimitri's original experiment. ``canonical`` delegates
    to the project loader, which combines all three storage files and repartitions
    sessions chronologically using ``configs/forecasting.yaml``.
    """
    mode = normalise_split_mode(split_mode)
    if mode == "physical":
        splits = {
            split: load_physical_candle_split(data_dir, split)
            for split in ("train", "val", "test")
        }
    else:
        train, validation, test = load_candle_splits(data_dir)
        splits = {"train": train, "val": validation, "test": test}

    reference_assets = [str(value) for value in splits["train"]["asset_cols"]]
    reference_channels = [
        str(value).lower() for value in splits["train"]["channels"]
    ]
    if len(reference_assets) != DIMITRI_EXPECTED_ASSETS:
        raise ValueError(
            f"Selected training split has {len(reference_assets)} assets; "
            f"expected {DIMITRI_EXPECTED_ASSETS}."
        )
    if tuple(reference_channels) != DIMITRI_CHANNELS:
        raise ValueError(
            f"Selected training channels differ: {tuple(reference_channels)}."
        )
    for split, payload in splits.items():
        if [str(value) for value in payload["asset_cols"]] != reference_assets:
            raise ValueError(f"{split} asset order differs from training.")
        if [str(value).lower() for value in payload["channels"]] != reference_channels:
            raise ValueError(f"{split} channel order differs from training.")
        if not payload.get("samples"):
            raise ValueError(f"Selected {split} split contains no sessions.")
    return splits


@dataclass(frozen=True)
class DimitriTokenPriceWindowSpec:
    context_length: int = 180
    continuation_length: int = 30
    stride: int = 30
    clip: float = DIMITRI_CLIP
    eps: float = DIMITRI_EPS

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.continuation_length <= 0:
            raise ValueError("continuation_length must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.sequence_length > 512:
            raise ValueError(
                "Dimitri V2 max_seq_len is 512; context+continuation exceeds it."
            )
        if self.clip <= 0 or self.eps <= 0:
            raise ValueError("clip and eps must be positive.")

    @property
    def sequence_length(self) -> int:
        return int(self.context_length + self.continuation_length)

    @property
    def tag(self) -> str:
        return (
            f"c{self.context_length}_p{self.continuation_length}_"
            f"s{self.stride}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": int(self.context_length),
            "continuation_length": int(self.continuation_length),
            "sequence_length": int(self.sequence_length),
            "stride": int(self.stride),
            "clip": float(self.clip),
            "eps": float(self.eps),
            "drop_open_rows": DIMITRI_DROP_OPEN_ROWS,
            "std_correction": 1,
            "zero_amount": True,
        }


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


def exact_window_starts(session_length: int, spec: DimitriTokenPriceWindowSpec) -> list[int]:
    if session_length < spec.sequence_length:
        return []
    return list(
        range(
            0,
            session_length - spec.sequence_length + 1,
            spec.stride,
        )
    )


@torch.inference_mode()
def tokenize_clean_session(
    clean_session: torch.Tensor,
    *,
    tokenizer: Any,
    device: torch.device | str,
    amount_index: int,
    spec: DimitriTokenPriceWindowSpec,
    encode_chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Tokenise one clean session using the configurable anchor contract."""
    values = torch.as_tensor(clean_session).detach().cpu().float()
    if values.ndim != 3 or tuple(values.shape[1:]) != (DIMITRI_EXPECTED_ASSETS, 6):
        raise ValueError(
            "clean_session must have shape [T,93,6], got "
            f"{tuple(values.shape)}."
        )
    if not torch.isfinite(values).all():
        raise ValueError("clean_session contains non-finite values.")
    if encode_chunk <= 0:
        raise ValueError("encode_chunk must be positive.")

    values = values.permute(1, 0, 2).contiguous()  # [N,T,D]
    values[..., amount_index] = 0.0
    starts = exact_window_starts(values.shape[1], spec)
    if not starts:
        raise ValueError(
            f"No {spec.sequence_length}-bar window fits in session length "
            f"{values.shape[1]}."
        )

    windows = torch.stack(
        [values[:, start : start + spec.sequence_length] for start in starts],
        dim=0,
    )  # [W,N,L,D]
    context = windows[:, :, : spec.context_length]
    mean = context.mean(dim=2, keepdim=True)
    std = context.std(dim=2, keepdim=True)  # exact torch.std correction=1
    normalised = ((windows - mean) / (std + spec.eps)).clamp(-spec.clip, spec.clip)

    windows_count, assets, length, channels = normalised.shape
    flat = normalised.reshape(windows_count * assets, length, channels)
    s1 = torch.empty(flat.shape[0], length, dtype=torch.long)
    s2 = torch.empty_like(s1)
    target_device = torch.device(device)
    for offset in range(0, flat.shape[0], encode_chunk):
        stop = min(offset + encode_chunk, flat.shape[0])
        coarse, fine = tokenizer.encode(flat[offset:stop].to(target_device), half=True)
        s1[offset:stop] = coarse.detach().cpu().long()
        s2[offset:stop] = fine.detach().cpu().long()

    return (
        s1.reshape(windows_count, assets, length).contiguous(),
        s2.reshape(windows_count, assets, length).contiguous(),
        mean.squeeze(2).contiguous(),
        std.squeeze(2).contiguous(),
        starts,
    )


def generate_token_split(
    *,
    raw_split: Mapping[str, Any],
    split_name: str,
    split_mode: str = "physical",
    tokenizer: Any,
    output_path: str | Path,
    device: torch.device | str,
    spec: DimitriTokenPriceWindowSpec,
    tokenizer_id: str = DIMITRI_TOKENIZER_ID,
    tokenizer_revision: str = DIMITRI_TOKENIZER_REVISION,
    encode_chunk: int = 512,
    show_progress: bool = True,
) -> dict[str, Any]:
    split_mode = normalise_split_mode(split_mode)
    channels = tuple(str(value).lower() for value in raw_split["channels"])
    if channels != DIMITRI_CHANNELS:
        raise ValueError(f"Channel order differs: {channels}.")
    amount_index = channels.index("amount")

    s1_parts: list[torch.Tensor] = []
    s2_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    std_parts: list[torch.Tensor] = []
    window_dates: list[Any] = []
    window_starts: list[int] = []
    sample_indices: list[int] = []

    samples = list(raw_split["samples"])
    for sample_index, sample in enumerate(samples):
        candle, _auxiliary, session_date = sample[:3]
        values = torch.as_tensor(candle)
        if values.ndim != 3 or tuple(values.shape[1:]) != (DIMITRI_EXPECTED_ASSETS, 6):
            raise ValueError(
                f"Malformed {split_name} session {session_date}: {tuple(values.shape)}."
            )
        clean = values[DIMITRI_DROP_OPEN_ROWS:].contiguous()
        s1, s2, mean, std, starts = tokenize_clean_session(
            clean,
            tokenizer=tokenizer,
            device=device,
            amount_index=amount_index,
            spec=spec,
            encode_chunk=encode_chunk,
        )
        s1_parts.append(s1)
        s2_parts.append(s2)
        mean_parts.append(mean)
        std_parts.append(std)
        window_dates.extend([session_date] * len(starts))
        window_starts.extend(starts)
        sample_indices.extend([sample_index] * len(starts))
        if show_progress and ((sample_index + 1) % 20 == 0 or sample_index + 1 == len(samples)):
            print(
                f"[{split_name}] tokenised {sample_index + 1}/{len(samples)} sessions",
                flush=True,
            )

    payload = {
        "s1": torch.cat(s1_parts, dim=0).contiguous(),
        "s2": torch.cat(s2_parts, dim=0).contiguous(),
        "norm_mean": torch.cat(mean_parts, dim=0).contiguous(),
        "norm_std": torch.cat(std_parts, dim=0).contiguous(),
        "window_date": window_dates,
        "window_start": torch.tensor(window_starts, dtype=torch.long),
        "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
        "asset_cols": list(raw_split["asset_cols"]),
        "channels": list(raw_split["channels"]),
        "s1_bits": int(getattr(tokenizer, "s1_bits")),
        "s2_bits": int(getattr(tokenizer, "s2_bits")),
        "T": int(spec.sequence_length),
        "F": int(len(raw_split["asset_cols"])),
        "drop_open": DIMITRI_DROP_OPEN_ROWS,
        "zero_amount": True,
        "tokenizer": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "grain": raw_split.get("grain", "1min"),
        "market_open": raw_split.get("market_open", "09:30"),
        "market_close": raw_split.get("market_close", "16:00"),
        "windowed": True,
        "anchor": True,
        "context_len": int(spec.context_length),
        "pred_len": int(spec.continuation_length),
        "window": int(spec.sequence_length),
        "stride": int(spec.stride),
        "clip": float(spec.clip),
        "eps": float(spec.eps),
        "std_correction": 1,
        "split_mode": split_mode,
        "physical_split_membership_preserved": split_mode == "physical",
        "canonical_chronological_repartition": split_mode == "canonical",
        "split_name": str(split_name),
        "contract": "dimitri_configurable_anchor_tokens_v1",
    }
    _atomic_torch_save(payload, Path(output_path))
    return validate_token_split(
        output_path,
        split_name=split_name,
        split_mode=split_mode,
        spec=spec,
    )


def validate_token_split(
    path: str | Path,
    *,
    split_name: str,
    spec: DimitriTokenPriceWindowSpec,
    split_mode: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _torch_load(path)
    observed_mode = payload.get("split_mode")
    if observed_mode is None:
        # Backward compatibility for the exact physical x0 caches.
        observed_mode = (
            "physical"
            if bool(payload.get("physical_split_membership_preserved", False))
            else "unknown"
        )
    if split_mode is not None:
        expected_mode = normalise_split_mode(split_mode)
        if observed_mode != expected_mode:
            raise ValueError(
                f"Token cache split mode {observed_mode!r} differs from "
                f"requested {expected_mode!r}: {path}."
            )
    required = {
        "s1",
        "s2",
        "norm_mean",
        "norm_std",
        "window_date",
        "window_start",
        "sample_idx",
        "asset_cols",
        "channels",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"{path} is missing {missing}.")
    s1 = torch.as_tensor(payload["s1"]).long()
    s2 = torch.as_tensor(payload["s2"]).long()
    if s1.ndim != 3 or tuple(s2.shape) != tuple(s1.shape):
        raise ValueError("Token tensors must match [W,N,T].")
    windows, assets, length = map(int, s1.shape)
    if assets != DIMITRI_EXPECTED_ASSETS or length != spec.sequence_length:
        raise ValueError(
            f"Unexpected token shape {tuple(s1.shape)} for spec {spec.to_dict()}."
        )
    if int(payload.get("context_len", -1)) != spec.context_length:
        raise ValueError("Token cache context length differs.")
    if int(payload.get("pred_len", -1)) != spec.continuation_length:
        raise ValueError("Token cache continuation length differs.")
    if int(payload.get("stride", -1)) != spec.stride:
        raise ValueError("Token cache stride differs.")
    if tuple(torch.as_tensor(payload["norm_mean"]).shape) != (windows, assets, 6):
        raise ValueError("norm_mean shape differs.")
    if tuple(torch.as_tensor(payload["norm_std"]).shape) != (windows, assets, 6):
        raise ValueError("norm_std shape differs.")
    if len(payload["window_date"]) != windows:
        raise ValueError("window_date length differs.")
    if int(torch.as_tensor(payload["window_start"]).numel()) != windows:
        raise ValueError("window_start length differs.")
    if int(torch.as_tensor(payload["sample_idx"]).numel()) != windows:
        raise ValueError("sample_idx length differs.")
    if len(payload["asset_cols"]) != assets:
        raise ValueError("asset_cols length differs.")
    if tuple(str(value).lower() for value in payload["channels"]) != DIMITRI_CHANNELS:
        raise ValueError("channel order differs.")
    for name, values in (("s1", s1), ("s2", s2)):
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= 1024:
            raise ValueError(f"{name} range [{minimum},{maximum}] is invalid.")
    return {
        "split": str(split_name),
        "split_mode": str(observed_mode),
        "path": str(path),
        "windows": windows,
        "assets": assets,
        "sequence_length": length,
        "context_length": spec.context_length,
        "continuation_length": spec.continuation_length,
        "stride": spec.stride,
        "minimum_s1": int(s1.min().item()),
        "maximum_s1": int(s1.max().item()),
        "sha256": file_sha256(path),
    }


def ensure_token_caches(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    device: torch.device | str,
    spec: DimitriTokenPriceWindowSpec,
    split_mode: str = "physical",
    encode_chunk: int = 512,
    force: bool = False,
) -> tuple[dict[str, Any], ...]:
    split_mode = normalise_split_mode(split_mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_splits = load_token_price_splits(data_dir, split_mode=split_mode)
    summaries: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.pt"
        if path.is_file() and not force:
            summary = validate_token_split(
                path,
                split_name=split,
                split_mode=split_mode,
                spec=spec,
            )
            print(f"[{split}] existing token cache validated: {path}")
        else:
            summary = generate_token_split(
                raw_split=raw_splits[split],
                split_name=split,
                split_mode=split_mode,
                tokenizer=tokenizer,
                output_path=path,
                device=device,
                spec=spec,
                encode_chunk=encode_chunk,
            )
        summaries.append(summary)
    _atomic_json_save(
        {
            "contract": "dimitri_configurable_anchor_tokens_v1",
            "window_spec": spec.to_dict(),
            "tokenizer_id": DIMITRI_TOKENIZER_ID,
            "tokenizer_revision": DIMITRI_TOKENIZER_REVISION,
            "split_mode": split_mode,
            "physical_split_membership_preserved": split_mode == "physical",
            "canonical_chronological_repartition": split_mode == "canonical",
            "splits": summaries,
        },
        output_dir / "token_cache_manifest.json",
    )
    return tuple(summaries)


def make_clean_physical_split(raw_split: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the first raw row while preserving physical file membership."""
    clean_samples = []
    for sample in raw_split["samples"]:
        candle, auxiliary, session_date = sample[:3]
        clean_samples.append(
            (
                torch.as_tensor(candle)[DIMITRI_DROP_OPEN_ROWS:].contiguous(),
                auxiliary,
                session_date,
            )
        )
    return {
        **{key: value for key, value in raw_split.items() if key != "samples"},
        "samples": clean_samples,
        "asset_cols": list(raw_split["asset_cols"]),
        "channels": list(raw_split["channels"]),
    }


class DimitriTokenPriceDataset(Dataset[dict[str, Any]]):
    """Join a token cache to the corresponding physical raw Close windows."""

    def __init__(
        self,
        *,
        token_path: str | Path,
        raw_split: Mapping[str, Any],
        split_name: str,
        spec: DimitriTokenPriceWindowSpec,
        split_mode: str | None = None,
    ) -> None:
        super().__init__()
        validate_token_split(
            token_path,
            split_name=split_name,
            split_mode=split_mode,
            spec=spec,
        )
        payload = _torch_load(token_path)
        self.state_ids = torch.as_tensor(payload["s1"]).long().contiguous()
        self.norm_mean = torch.as_tensor(payload["norm_mean"]).float().contiguous()
        self.norm_std = torch.as_tensor(payload["norm_std"]).float().contiguous()
        self.window_start = torch.as_tensor(payload["window_start"]).long().contiguous()
        self.sample_idx = torch.as_tensor(payload["sample_idx"]).long().contiguous()
        self.window_date = [str(value) for value in payload["window_date"]]
        self.asset_cols = [str(value) for value in payload["asset_cols"]]
        self.channels = [str(value).lower() for value in payload["channels"]]
        self.spec = spec
        self.split_name = str(split_name)
        self.split_mode = (
            None if split_mode is None else normalise_split_mode(split_mode)
        )
        self.close_index = self.channels.index("close")
        if list(raw_split["asset_cols"]) != self.asset_cols:
            raise ValueError("Raw split and token cache asset orders differ.")
        if [str(value).lower() for value in raw_split["channels"]] != self.channels:
            raise ValueError("Raw split and token cache channel orders differ.")
        self.clean_sessions = [
            torch.as_tensor(sample[0])[DIMITRI_DROP_OPEN_ROWS:].contiguous()
            for sample in raw_split["samples"]
        ]
        self.session_dates = [str(sample[2]) for sample in raw_split["samples"]]
        for index in range(len(self.state_ids)):
            sample_index = int(self.sample_idx[index])
            start = int(self.window_start[index])
            if sample_index < 0 or sample_index >= len(self.clean_sessions):
                raise IndexError("Token cache sample_idx is outside the raw split.")
            if start < 0 or start + spec.sequence_length > self.clean_sessions[sample_index].shape[0]:
                raise IndexError("Token cache window_start is outside its session.")
            if self.window_date[index] != self.session_dates[sample_index]:
                raise ValueError("Token cache date does not match raw session date.")

    def __len__(self) -> int:
        return int(self.state_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = int(self.sample_idx[index])
        start = int(self.window_start[index])
        stop = start + self.spec.sequence_length
        raw_close = self.clean_sessions[sample_index][
            start:stop,
            :,
            self.close_index,
        ].transpose(0, 1).float().contiguous()  # [N,T]
        return {
            "state_ids": self.state_ids[index],
            "raw_close": raw_close,
            "close_mean": self.norm_mean[index, :, self.close_index],
            "close_std": self.norm_std[index, :, self.close_index],
            "sample_idx": self.sample_idx[index],
            "window_start": self.window_start[index],
            "window_date": self.window_date[index],
        }

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


KRONOS_S1_VOCABULARY_SIZE = 1024
S1_REMAP_RESOURCE_VERSION = 1
S1_REMAP_METHOD = (
    "top_k_training_frequency_continuous_causal_locf"
)
S1_ID_SPACE_ORIGINAL = "kronos_original"
S1_ID_SPACE_COMPACT = "compact_retained_kronos"

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _date_key(value: Any) -> str:
    """Return a stable YYYY-MM-DD key for Timestamp/date/string values."""
    if hasattr(value, "date") and not isinstance(value, str):
        try:
            value = value.date()
        except TypeError:
            pass

    text = str(value).strip()
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    return text


def _torch_load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(
            resolved,
            map_location="cpu",
        )

    if not isinstance(loaded, Mapping):
        raise TypeError(
            f"Expected a saved mapping at {resolved}."
        )

    return dict(loaded)


def _tensor_bytes(values: Tensor) -> bytes:
    return (
        torch.as_tensor(values)
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )


def _resource_hash(
    *,
    retained_original_ids: Tensor,
    original_to_compact: Tensor,
    training_counts: Tensor,
    method: str,
    k: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(method).encode("utf-8"))
    digest.update(str(int(k)).encode("utf-8"))
    digest.update(_tensor_bytes(retained_original_ids))
    digest.update(_tensor_bytes(original_to_compact))
    digest.update(_tensor_bytes(training_counts))
    return digest.hexdigest()


@dataclass(frozen=True)
class S1TokenRemappingResource:
    """Training-fitted compact ``s1`` vocabulary and exact inverse.

    ``retained_original_ids[compact_id]`` is the original Kronos ID
    consumed by the frozen coarse decoder.

    ``original_to_compact[original_id]`` is the compact model ID for a
    retained token and ``-1`` for a discarded token. Discarded tokens
    must be replaced by causal LOCF before this lookup is applied.
    """

    k: int
    retained_original_ids: Tensor
    original_to_compact: Tensor
    training_counts: Tensor
    training_token_count: int
    training_coverage_percent: float
    method: str = S1_REMAP_METHOD
    resource_hash: str = ""

    @property
    def compact_to_original(self) -> Tensor:
        return self.retained_original_ids

    @property
    def fallback_original_id(self) -> int:
        """Least frequent retained token, used only if no history exists."""
        return int(self.retained_original_ids[-1].item())

    def validate(self) -> None:
        if not 1 <= int(self.k) <= KRONOS_S1_VOCABULARY_SIZE:
            raise ValueError(
                "k must lie in [1, 1024]."
            )

        retained = torch.as_tensor(
            self.retained_original_ids,
            dtype=torch.long,
        )
        forward = torch.as_tensor(
            self.original_to_compact,
            dtype=torch.long,
        )
        counts = torch.as_tensor(
            self.training_counts,
            dtype=torch.long,
        )

        if tuple(retained.shape) != (int(self.k),):
            raise ValueError(
                "retained_original_ids must have shape [k]."
            )

        if tuple(forward.shape) != (
            KRONOS_S1_VOCABULARY_SIZE,
        ):
            raise ValueError(
                "original_to_compact must have shape [1024]."
            )

        if tuple(counts.shape) != (
            KRONOS_S1_VOCABULARY_SIZE,
        ):
            raise ValueError(
                "training_counts must have shape [1024]."
            )

        if retained.min().item() < 0 or retained.max().item() >= 1024:
            raise ValueError(
                "retained_original_ids contains an invalid Kronos ID."
            )

        if torch.unique(retained).numel() != int(self.k):
            raise ValueError(
                "retained_original_ids must be unique."
            )

        if torch.any(counts < 0):
            raise ValueError(
                "training_counts cannot be negative."
            )

        expected_forward = torch.full(
            (KRONOS_S1_VOCABULARY_SIZE,),
            fill_value=-1,
            dtype=torch.long,
        )
        expected_forward[retained] = torch.arange(
            int(self.k),
            dtype=torch.long,
        )

        if not torch.equal(forward, expected_forward):
            raise ValueError(
                "original_to_compact is not the exact inverse lookup "
                "for retained_original_ids."
            )

        observed_total = int(counts.sum().item())
        if observed_total != int(self.training_token_count):
            raise ValueError(
                "training_token_count does not match training_counts."
            )

        if observed_total <= 0:
            raise ValueError(
                "The resource contains no training observations."
            )

        expected_coverage = float(
            counts[retained]
            .sum()
            .to(torch.float32)
            .div(float(observed_total))
            .mul(100.0)
            .item()
        )

        if abs(
            expected_coverage
            - float(self.training_coverage_percent)
        ) > 1.0e-5:
            raise ValueError(
                "training_coverage_percent is inconsistent."
            )

        expected_hash = _resource_hash(
            retained_original_ids=retained,
            original_to_compact=forward,
            training_counts=counts,
            method=str(self.method),
            k=int(self.k),
        )

        if self.resource_hash and self.resource_hash != expected_hash:
            raise ValueError(
                "The saved s1 remapping resource hash is invalid."
            )

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": S1_REMAP_RESOURCE_VERSION,
            "kind": "s1_token_remapping_resource",
            "method": str(self.method),
            "k": int(self.k),
            "retained_original_ids": (
                self.retained_original_ids
                .detach()
                .cpu()
                .to(torch.int16)
                .contiguous()
            ),
            "compact_to_original": (
                self.retained_original_ids
                .detach()
                .cpu()
                .to(torch.int16)
                .contiguous()
            ),
            "original_to_compact": (
                self.original_to_compact
                .detach()
                .cpu()
                .to(torch.int16)
                .contiguous()
            ),
            "training_counts": (
                self.training_counts
                .detach()
                .cpu()
                .to(torch.long)
                .contiguous()
            ),
            "training_token_count": int(
                self.training_token_count
            ),
            "training_coverage_percent": float(
                self.training_coverage_percent
            ),
            "fallback_original_id": int(
                self.fallback_original_id
            ),
            "resource_hash": str(self.resource_hash),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "S1TokenRemappingResource":
        if int(payload.get("format_version", -1)) != (
            S1_REMAP_RESOURCE_VERSION
        ):
            raise ValueError(
                "Unsupported s1 remapping resource version."
            )

        if str(payload.get("kind", "")) != (
            "s1_token_remapping_resource"
        ):
            raise ValueError(
                "Unexpected s1 remapping resource kind."
            )

        resource = cls(
            k=int(payload["k"]),
            retained_original_ids=torch.as_tensor(
                payload["retained_original_ids"],
                dtype=torch.long,
            ).contiguous(),
            original_to_compact=torch.as_tensor(
                payload["original_to_compact"],
                dtype=torch.long,
            ).contiguous(),
            training_counts=torch.as_tensor(
                payload["training_counts"],
                dtype=torch.long,
            ).contiguous(),
            training_token_count=int(
                payload["training_token_count"]
            ),
            training_coverage_percent=float(
                payload["training_coverage_percent"]
            ),
            method=str(payload["method"]),
            resource_hash=str(payload["resource_hash"]),
        )
        resource.validate()
        return resource


def save_s1_remapping_resource(
    resource: S1TokenRemappingResource,
    path: str | Path,
) -> Path:
    resource.validate()
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(resource.to_payload(), temporary)
    temporary.replace(output)
    return output


def load_s1_remapping_resource(
    path: str | Path,
) -> S1TokenRemappingResource:
    return S1TokenRemappingResource.from_payload(
        _torch_load_mapping(path)
    )


def fit_top_k_s1_remapping_resource(
    encoded_training: Mapping[str, Any],
    *,
    k: int,
) -> S1TokenRemappingResource:
    """Fit the retained original IDs from training rolling tokens only."""
    k = int(k)
    if not 1 <= k <= KRONOS_S1_VOCABULARY_SIZE:
        raise ValueError(
            "k must lie in [1, 1024]."
        )

    s1 = torch.as_tensor(encoded_training["s1"])
    valid_mask = torch.as_tensor(
        encoded_training["valid_mask"],
        dtype=torch.bool,
    )

    if s1.ndim != 3:
        raise ValueError(
            "encoded_training['s1'] must have shape [S, T, N]."
        )

    if tuple(valid_mask.shape) != tuple(s1.shape[:2]):
        raise ValueError(
            "valid_mask does not align with the rolling s1 stream."
        )

    valid_values = s1[
        valid_mask.unsqueeze(-1).expand_as(s1)
    ].to(torch.long)

    if valid_values.numel() == 0:
        raise ValueError(
            "The training rolling cache contains no valid s1 tokens."
        )

    if valid_values.min().item() < 0 or valid_values.max().item() >= 1024:
        raise ValueError(
            "Training s1 values lie outside [0, 1023]."
        )

    counts = torch.bincount(
        valid_values,
        minlength=KRONOS_S1_VOCABULARY_SIZE,
    ).to(torch.long)

    ranked = sorted(
        range(KRONOS_S1_VOCABULARY_SIZE),
        key=lambda token_id: (
            -int(counts[token_id].item()),
            token_id,
        ),
    )

    retained = torch.tensor(
        ranked[:k],
        dtype=torch.long,
    )

    forward = torch.full(
        (KRONOS_S1_VOCABULARY_SIZE,),
        fill_value=-1,
        dtype=torch.long,
    )
    forward[retained] = torch.arange(k, dtype=torch.long)

    total = int(counts.sum().item())
    coverage = float(
        counts[retained]
        .sum()
        .to(torch.float32)
        .div(float(total))
        .mul(100.0)
        .item()
    )

    resource_hash = _resource_hash(
        retained_original_ids=retained,
        original_to_compact=forward,
        training_counts=counts,
        method=S1_REMAP_METHOD,
        k=k,
    )

    resource = S1TokenRemappingResource(
        k=k,
        retained_original_ids=retained,
        original_to_compact=forward,
        training_counts=counts,
        training_token_count=total,
        training_coverage_percent=coverage,
        method=S1_REMAP_METHOD,
        resource_hash=resource_hash,
    )
    resource.validate()
    return resource


def _validate_rolling_cache(
    encoded: Mapping[str, Any],
    *,
    name: str,
) -> None:
    required = {
        "context_s1",
        "context_s2",
        "s1",
        "s2",
        "valid_mask",
        "origin_indices",
        "dates",
        "asset_cols",
        "context_length",
        "num_bars",
        "tokenizer_id",
        "tokenizer_revision",
    }
    missing = required - set(encoded)
    if missing:
        raise KeyError(
            f"{name} rolling cache is missing: {sorted(missing)}."
        )

    context_s1 = torch.as_tensor(encoded["context_s1"])
    context_s2 = torch.as_tensor(encoded["context_s2"])
    s1 = torch.as_tensor(encoded["s1"])

    if context_s1.ndim != 4 or context_s1.shape != context_s2.shape:
        raise ValueError(
            f"{name} context_s1/context_s2 must have matching "
            "shape [S, O, C, N]."
        )

    if s1.ndim != 3:
        raise ValueError(
            f"{name} s1 must have shape [S, T, N]."
        )

    sessions, origins, context_length, assets = context_s1.shape
    if tuple(s1.shape) != (
        sessions,
        int(encoded["num_bars"]),
        assets,
    ):
        raise ValueError(
            f"{name} s1 shape does not match rolling metadata."
        )

    if int(encoded["context_length"]) != context_length:
        raise ValueError(
            f"{name} context_length metadata is inconsistent."
        )

    origin_indices = torch.as_tensor(
        encoded["origin_indices"],
        dtype=torch.long,
    )
    if tuple(origin_indices.shape) != (origins,):
        raise ValueError(
            f"{name} origin_indices has the wrong shape."
        )

    expected = torch.arange(
        context_length - 1,
        int(encoded["num_bars"]),
        dtype=torch.long,
    )
    if not torch.equal(origin_indices, expected):
        raise ValueError(
            f"{name} rolling origins are not complete and contiguous."
        )

    if len(encoded["dates"]) != sessions:
        raise ValueError(
            f"{name} dates length does not match sessions."
        )

    if len(encoded["asset_cols"]) != assets:
        raise ValueError(
            f"{name} asset_cols length does not match assets."
        )

    dates = [_date_key(value) for value in encoded["dates"]]
    if dates != sorted(dates):
        raise ValueError(
            f"{name} rolling cache is not chronological."
        )


def concatenate_rolling_caches(
    caches: Sequence[Mapping[str, Any]],
    *,
    name: str = "combined",
) -> dict[str, Any]:
    """Concatenate chronological rolling caches along the session axis."""
    if not caches:
        raise ValueError(
            "At least one rolling cache is required."
        )

    resolved = [dict(cache) for cache in caches]
    for index, cache in enumerate(resolved):
        _validate_rolling_cache(
            cache,
            name=f"{name}[{index}]",
        )

    reference = resolved[0]
    static_keys = (
        "origin_indices",
        "asset_cols",
        "channels",
        "tokenizer_channels",
        "context_length",
        "num_bars",
        "zero_amount",
        "tokenizer_id",
        "tokenizer_revision",
        "clip",
        "eps",
    )

    for cache in resolved[1:]:
        for key in static_keys:
            if key not in reference and key not in cache:
                continue
            left = reference.get(key)
            right = cache.get(key)
            if isinstance(left, Tensor) or isinstance(right, Tensor):
                if not torch.equal(
                    torch.as_tensor(left),
                    torch.as_tensor(right),
                ):
                    raise ValueError(
                        f"Rolling caches disagree for {key}."
                    )
            elif left != right:
                raise ValueError(
                    f"Rolling caches disagree for {key}."
                )

    session_tensor_keys = (
        "context_s1",
        "context_s2",
        "context_mean",
        "context_std",
        "s1",
        "s2",
        "valid_mask",
    )

    output = dict(reference)
    for key in session_tensor_keys:
        output[key] = torch.cat(
            [torch.as_tensor(cache[key]) for cache in resolved],
            dim=0,
        ).contiguous()

    output["dates"] = [
        _date_key(date)
        for cache in resolved
        for date in cache["dates"]
    ]

    if len(set(output["dates"])) != len(output["dates"]):
        raise ValueError(
            "Combined rolling caches contain duplicate session dates."
        )

    if output["dates"] != sorted(output["dates"]):
        raise ValueError(
            "Combined rolling caches are not chronological."
        )

    output["kind"] = (
        "kronos_causal_rolling_tokens_combined"
        if "s1_remapping" not in output
        else "kronos_causal_rolling_tokens_s1_remapped_combined"
    )
    output["combined_from"] = [
        str(cache.get("kind", "unknown"))
        for cache in resolved
    ]
    return output



def build_rolling_cache_period_view(
    source_caches: Sequence[Mapping[str, Any]],
    *,
    start_date: Any,
    end_date: Any,
    name: str,
) -> dict[str, Any]:
    """Build one chronological in-memory rolling-cache date view.

    ``source_caches`` should normally be the canonical production-period
    caches, for example January--August and September. The returned view
    contains only sessions in the half-open interval
    ``[start_date, end_date)``. No file is written.

    All rolling tensors are subset along their session axis. Static
    tokenizer, context-length, origin-index and asset-order metadata must
    agree across source caches.
    """
    if not source_caches:
        raise ValueError(
            "At least one source rolling cache is required."
        )

    start_key = _date_key(start_date)
    end_key = _date_key(end_date)

    if end_key <= start_key:
        raise ValueError(
            "end_date must be later than start_date."
        )

    session_tensor_keys = (
        "context_s1",
        "context_s2",
        "context_mean",
        "context_std",
        "s1",
        "s2",
        "valid_mask",
    )

    selected_caches: list[dict[str, Any]] = []

    for cache_index, source in enumerate(source_caches):
        cache = dict(source)
        _validate_rolling_cache(
            cache,
            name=f"{name} source[{cache_index}]",
        )

        dates = [
            _date_key(value)
            for value in cache["dates"]
        ]

        selected_indices = [
            index
            for index, date in enumerate(dates)
            if start_key <= date < end_key
        ]

        if not selected_indices:
            continue

        indices = torch.as_tensor(
            selected_indices,
            dtype=torch.long,
        )

        subset = dict(cache)

        for key in session_tensor_keys:
            if key not in cache:
                raise KeyError(
                    f"{name} source[{cache_index}] is missing "
                    f"session tensor {key!r}."
                )

            values = torch.as_tensor(
                cache[key]
            )

            if (
                values.ndim == 0
                or values.shape[0] != len(dates)
            ):
                raise ValueError(
                    f"{name} source[{cache_index}] field {key!r} "
                    "does not align with its dates."
                )

            subset[key] = (
                values
                .index_select(
                    0,
                    indices,
                )
                .contiguous()
            )

        subset["dates"] = [
            dates[index]
            for index in selected_indices
        ]
        subset["kind"] = (
            "kronos_causal_rolling_tokens_period_view"
            if "s1_remapping" not in subset
            else "kronos_causal_rolling_tokens_s1_remapped_period_view"
        )
        subset["period_view"] = {
            "name": str(name),
            "start_date": start_key,
            "end_date_exclusive": end_key,
        }
        selected_caches.append(
            subset
        )

    if not selected_caches:
        raise ValueError(
            f"No rolling-cache sessions lie in "
            f"[{start_key}, {end_key})."
        )

    combined = concatenate_rolling_caches(
        selected_caches,
        name=name,
    )
    combined["period_view"] = {
        "name": str(name),
        "start_date": start_key,
        "end_date_exclusive": end_key,
    }
    return combined


def validate_rolling_cache_split_alignment(
    encoded: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    name: str,
) -> None:
    """Verify that a canonical rolling cache matches one candle split.

    The check covers chronological session dates and canonical asset order.
    It is intended to prevent an existing cache from being silently reused
    with a different production split.
    """
    _validate_rolling_cache(
        encoded,
        name=name,
    )

    if "samples" not in split:
        raise KeyError(
            f"{name} candle split has no 'samples' field."
        )

    if "asset_cols" not in split:
        raise KeyError(
            f"{name} candle split has no 'asset_cols' field."
        )

    cache_dates = [
        _date_key(value)
        for value in encoded["dates"]
    ]
    split_dates = [
        _date_key(sample[2])
        for sample in split["samples"]
    ]

    if cache_dates != split_dates:
        raise ValueError(
            f"{name} rolling-cache dates do not match the "
            "clean candle split."
        )

    if list(encoded["asset_cols"]) != list(
        split["asset_cols"]
    ):
        raise ValueError(
            f"{name} rolling-cache asset order does not match "
            "the clean candle split."
        )

def build_complete_canonical_s1_timeline(
    encoded: Mapping[str, Any],
) -> Tensor:
    """Build one causal canonical original-ID stream per bar and asset."""
    _validate_rolling_cache(encoded, name="Encoded")

    context_s1 = torch.as_tensor(encoded["context_s1"])
    final_s1 = torch.as_tensor(encoded["s1"])
    context_length = int(encoded["context_length"])
    num_bars = int(encoded["num_bars"])

    sessions, _, _, assets = context_s1.shape
    timeline = torch.empty(
        (sessions, num_bars, assets),
        dtype=context_s1.dtype,
    )

    timeline[:, :context_length] = context_s1[:, 0]
    if context_length < num_bars:
        timeline[:, context_length:] = final_s1[:, context_length:]

    if timeline.min().item() < 0 or timeline.max().item() >= 1024:
        raise ValueError(
            "The canonical timeline contains invalid original IDs."
        )

    return timeline.contiguous()


def _last_retained_token_by_asset(
    timeline: Tensor,
    retained_lookup: Tensor,
    *,
    fallback_original_id: int,
) -> tuple[Tensor, Tensor]:
    sessions, bars, assets = timeline.shape
    flat = timeline.reshape(sessions * bars, assets).to(torch.long)
    mask = retained_lookup[flat]
    positions = torch.arange(
        flat.shape[0],
        dtype=torch.long,
    ).unsqueeze(1).expand_as(flat)
    retained_positions = torch.where(
        mask,
        positions,
        torch.full_like(positions, -1),
    )
    final_position = torch.cummax(
        retained_positions,
        dim=0,
    ).values[-1]
    asset_indices = torch.arange(assets, dtype=torch.long)
    gathered = flat[
        final_position.clamp_min(0),
        asset_indices,
    ]
    fallback = torch.full_like(
        gathered,
        int(fallback_original_id),
    )
    result = torch.where(
        final_position >= 0,
        gathered,
        fallback,
    )
    return result.to(timeline.dtype), final_position < 0


def _previous_retained_before_each_bar(
    timeline: Tensor,
    retained_lookup: Tensor,
    *,
    initial_token_by_asset: Tensor,
) -> Tensor:
    sessions, bars, assets = timeline.shape
    flat = timeline.reshape(sessions * bars, assets).to(torch.long)
    mask = retained_lookup[flat]
    positions = torch.arange(
        flat.shape[0],
        dtype=torch.long,
    ).unsqueeze(1).expand_as(flat)
    retained_positions = torch.where(
        mask,
        positions,
        torch.full_like(positions, -1),
    )
    inclusive = torch.cummax(
        retained_positions,
        dim=0,
    ).values
    before = torch.full_like(inclusive, -1)
    before[1:] = inclusive[:-1]
    gathered = flat.gather(
        dim=0,
        index=before.clamp_min(0),
    )
    initial = torch.as_tensor(
        initial_token_by_asset,
        dtype=flat.dtype,
    ).view(1, assets).expand_as(flat)
    previous = torch.where(
        before >= 0,
        gathered,
        initial,
    )
    return previous.reshape(sessions, bars, assets).to(timeline.dtype)


def remap_rolling_s1_cache(
    encoded: Mapping[str, Any],
    resource: S1TokenRemappingResource,
    *,
    initial_token_by_asset: Tensor,
    split_name: str,
) -> dict[str, Any]:
    """Apply continuous causal LOCF to every stored rolling context."""
    resource.validate()
    _validate_rolling_cache(encoded, name=split_name)

    context_s1 = torch.as_tensor(encoded["context_s1"])
    timeline = build_complete_canonical_s1_timeline(encoded)
    origins = torch.as_tensor(
        encoded["origin_indices"],
        dtype=torch.long,
    )

    retained_lookup = torch.zeros(
        KRONOS_S1_VOCABULARY_SIZE,
        dtype=torch.bool,
    )
    retained_lookup[resource.retained_original_ids.long()] = True

    previous_before = _previous_retained_before_each_bar(
        timeline,
        retained_lookup,
        initial_token_by_asset=initial_token_by_asset,
    )

    context_length = int(encoded["context_length"])
    starts = origins - context_length + 1
    carry = previous_before[:, starts, :].clone()
    remapped_context = torch.empty_like(context_s1)

    for context_time in range(context_length):
        current = context_s1[:, :, context_time, :]
        is_retained = retained_lookup[current.long()]
        carry = torch.where(is_retained, current, carry)
        remapped_context[:, :, context_time, :] = carry

    if not retained_lookup[remapped_context.long()].all():
        raise AssertionError(
            "Rolling LOCF output contains a discarded token."
        )

    remapped_s1 = torch.as_tensor(encoded["s1"]).clone()
    remapped_s1[:, origins, :] = remapped_context[:, :, -1, :]

    output = dict(encoded)
    output.update(
        {
            "kind": "kronos_causal_rolling_tokens_s1_remapped",
            "context_s1": remapped_context.contiguous(),
            "s1": remapped_s1.contiguous(),
            "s1_id_space": "retained_kronos_original",
            "s1_vocabulary_size": int(resource.k),
            "s1_remapping": {
                "method": resource.method,
                "resource_hash": resource.resource_hash,
                "k": int(resource.k),
                "retained_original_ids": (
                    resource.retained_original_ids
                    .to(torch.int16)
                    .contiguous()
                ),
                "fallback_original_id": int(
                    resource.fallback_original_id
                ),
                "history_scope": (
                    "continuous_across_bars_sessions_and_split_boundary"
                ),
                "context_boundary_reset": False,
                "split_name": str(split_name),
            },
        }
    )
    return output


def build_train_validation_remapped_rolling_caches(
    encoded_train: Mapping[str, Any],
    encoded_validation: Mapping[str, Any],
    resource: S1TokenRemappingResource,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Remap train and validation with one continuous causal history."""
    resource.validate()
    _validate_rolling_cache(encoded_train, name="Training")
    _validate_rolling_cache(encoded_validation, name="Validation")

    for key in (
        "asset_cols",
        "context_length",
        "num_bars",
        "tokenizer_id",
        "tokenizer_revision",
    ):
        if encoded_train[key] != encoded_validation[key]:
            raise ValueError(
                f"Train/validation rolling caches disagree for {key}."
            )

    retained_lookup = torch.zeros(
        KRONOS_S1_VOCABULARY_SIZE,
        dtype=torch.bool,
    )
    retained_lookup[resource.retained_original_ids.long()] = True

    training_timeline = build_complete_canonical_s1_timeline(
        encoded_train
    )

    # The first training bar has no earlier project observation. The
    # requested least-frequent retained fallback is used only until each
    # asset first encounters a genuine retained token.
    training_initial = torch.full(
        (training_timeline.shape[2],),
        fill_value=int(resource.fallback_original_id),
        dtype=training_timeline.dtype,
    )

    remapped_train = remap_rolling_s1_cache(
        encoded_train,
        resource,
        initial_token_by_asset=training_initial,
        split_name="train",
    )

    validation_initial, fallback_mask = _last_retained_token_by_asset(
        training_timeline,
        retained_lookup,
        fallback_original_id=resource.fallback_original_id,
    )

    remapped_validation = remap_rolling_s1_cache(
        encoded_validation,
        resource,
        initial_token_by_asset=validation_initial,
        split_name="validation",
    )

    fallback_assets = tuple(
        str(encoded_train["asset_cols"][index])
        for index in torch.nonzero(
            fallback_mask,
            as_tuple=False,
        ).flatten().tolist()
    )

    remapped_validation["s1_remapping"] = {
        **dict(remapped_validation["s1_remapping"]),
        "validation_initial_original_id_by_asset": (
            validation_initial.to(torch.int16).contiguous()
        ),
        "assets_requiring_training_history_fallback": fallback_assets,
    }

    return remapped_train, remapped_validation, fallback_assets


def _validate_base_origin_cache(
    cache: Mapping[str, Any],
    *,
    name: str,
) -> None:
    required = {
        "context_tokens",
        "target_s1",
        "target_s2",
        "sample_idx",
        "origin_idx",
        "dates",
        "asset_cols",
        "context_length",
        "prediction_length",
        "tokenizer_id",
        "tokenizer_revision",
    }
    missing = required - set(cache)
    if missing:
        raise KeyError(
            f"{name} origin cache is missing: {sorted(missing)}."
        )

    context = torch.as_tensor(cache["context_tokens"])
    target_s1 = torch.as_tensor(cache["target_s1"])
    if context.ndim != 4 or context.shape[-1] != 2:
        raise ValueError(
            f"{name} context_tokens must have shape [W, C, N, 2]."
        )
    if target_s1.ndim != 3:
        raise ValueError(
            f"{name} target_s1 must have shape [W, P, N]."
        )
    if (
        context.shape[0] != target_s1.shape[0]
        or context.shape[2] != target_s1.shape[2]
    ):
        raise ValueError(
            f"{name} context and target dimensions do not align."
        )
    if int(cache["context_length"]) != int(context.shape[1]):
        raise ValueError(
            f"{name} context_length metadata is inconsistent."
        )
    if int(cache["prediction_length"]) != int(target_s1.shape[1]):
        raise ValueError(
            f"{name} prediction_length metadata is inconsistent."
        )


def build_compact_origin_aligned_cache(
    base_origin_cache: Mapping[str, Any],
    original_rolling_cache: Mapping[str, Any],
    remapped_rolling_cache: Mapping[str, Any],
    resource: S1TokenRemappingResource,
    *,
    split_name: str,
) -> dict[str, Any]:
    """Convert a full-vocabulary origin cache into compact model IDs.

    The full origin-aligned cache is authoritative for the exact context
    and future token IDs consumed by the forecasting model. The rolling
    cache is used only to recover the causal retained-token state that
    existed immediately before each origin window began.

    This distinction is necessary because historical rolling caches were
    normalised through the adapter's NumPy path, whereas origin-aligned
    caches were normalised through the context-plus-future PyTorch path.
    The two paths implement the same context-only z-score definition but
    can differ at floating-point boundaries; BSQ can then assign a small
    number of different hard token IDs. Those differences must not replace
    the model's authoritative origin-aligned context.
    """
    resource.validate()
    _validate_base_origin_cache(
        base_origin_cache,
        name=split_name,
    )
    _validate_rolling_cache(
        original_rolling_cache,
        name=f"{split_name} original",
    )
    _validate_rolling_cache(
        remapped_rolling_cache,
        name=f"{split_name} remapped",
    )

    for key in (
        "dates",
        "asset_cols",
        "context_length",
        "num_bars",
        "tokenizer_id",
        "tokenizer_revision",
    ):
        left = original_rolling_cache[key]
        right = remapped_rolling_cache[key]

        if isinstance(left, Tensor) or isinstance(right, Tensor):
            if not torch.equal(
                torch.as_tensor(left),
                torch.as_tensor(right),
            ):
                raise ValueError(
                    f"Original/remapped rolling caches disagree for {key}."
                )
        elif left != right:
            raise ValueError(
                f"Original/remapped rolling caches disagree for {key}."
            )

    remapping_metadata = remapped_rolling_cache.get(
        "s1_remapping"
    )

    if isinstance(remapping_metadata, Mapping):
        cached_hash = remapping_metadata.get(
            "resource_hash"
        )
        if (
            cached_hash is not None
            and str(cached_hash) != resource.resource_hash
        ):
            raise ValueError(
                "The remapped rolling cache uses a different "
                "s1 remapping resource."
            )

    if list(base_origin_cache["asset_cols"]) != list(
        original_rolling_cache["asset_cols"]
    ):
        raise ValueError(
            "Origin and rolling caches use different asset ordering."
        )

    context_length = int(
        base_origin_cache["context_length"]
    )

    if context_length != int(
        original_rolling_cache["context_length"]
    ):
        raise ValueError(
            "Origin and rolling caches use different context lengths."
        )

    if (
        base_origin_cache["tokenizer_id"]
        != original_rolling_cache["tokenizer_id"]
    ):
        raise ValueError(
            "Origin and rolling caches use different tokenizers."
        )

    if (
        base_origin_cache["tokenizer_revision"]
        != original_rolling_cache["tokenizer_revision"]
    ):
        raise ValueError(
            "Origin and rolling caches use different tokenizer revisions."
        )

    origin_values = torch.as_tensor(
        original_rolling_cache["origin_indices"],
        dtype=torch.long,
    )
    origin_to_position = {
        int(origin): position
        for position, origin in enumerate(
            origin_values.tolist()
        )
    }

    # Align sessions by physical date. sample_idx is local to a split and
    # cannot safely join a broad rolling cache to a narrower model split.
    rolling_date_to_session = {
        _date_key(date): index
        for index, date in enumerate(
            original_rolling_cache["dates"]
        )
    }
    base_dates = [
        _date_key(value)
        for value in base_origin_cache["dates"]
    ]

    missing_dates = sorted(
        {
            date
            for date in base_dates
            if date not in rolling_date_to_session
        }
    )
    if missing_dates:
        raise ValueError(
            "Origin cache dates are absent from the rolling cache: "
            f"{missing_dates}."
        )

    rolling_session_idx = torch.tensor(
        [
            rolling_date_to_session[date]
            for date in base_dates
        ],
        dtype=torch.long,
    )

    origin_idx = torch.as_tensor(
        base_origin_cache["origin_idx"],
        dtype=torch.long,
    )
    origin_positions: list[int] = []
    for value in origin_idx.tolist():
        if int(value) not in origin_to_position:
            raise ValueError(
                f"Origin {value} is unavailable in the rolling cache."
            )
        origin_positions.append(
            origin_to_position[int(value)]
        )
    origin_positions_tensor = torch.tensor(
        origin_positions,
        dtype=torch.long,
    )

    # Check that the two cache families refer to the same raw windows by
    # comparing their context statistics. Small NumPy/PyTorch reduction
    # differences are allowed; a date/origin misalignment is not.
    rolling_mean = torch.as_tensor(
        original_rolling_cache["context_mean"],
        dtype=torch.float32,
    )[rolling_session_idx, origin_positions_tensor]
    rolling_std = torch.as_tensor(
        original_rolling_cache["context_std"],
        dtype=torch.float32,
    )[rolling_session_idx, origin_positions_tensor]
    origin_mean = torch.as_tensor(
        base_origin_cache["context_mean"],
        dtype=torch.float32,
    )
    origin_std = torch.as_tensor(
        base_origin_cache["context_std"],
        dtype=torch.float32,
    )

    if not torch.allclose(
        rolling_mean,
        origin_mean,
        rtol=1.0e-4,
        atol=1.0e-5,
    ):
        maximum_difference = float(
            (rolling_mean - origin_mean)
            .abs()
            .max()
            .item()
        )
        raise ValueError(
            "Origin and rolling caches do not describe the same "
            "context means. Maximum absolute difference: "
            f"{maximum_difference}."
        )

    if not torch.allclose(
        rolling_std,
        origin_std,
        rtol=1.0e-4,
        atol=1.0e-5,
    ):
        maximum_difference = float(
            (rolling_std - origin_std)
            .abs()
            .max()
            .item()
        )
        raise ValueError(
            "Origin and rolling caches do not describe the same "
            "context standard deviations. Maximum absolute difference: "
            f"{maximum_difference}."
        )

    rolling_context_s1 = torch.as_tensor(
        original_rolling_cache["context_s1"],
        dtype=torch.long,
    )[rolling_session_idx, origin_positions_tensor]
    base_context_s1 = torch.as_tensor(
        base_origin_cache["context_tokens"],
        dtype=torch.long,
    )[..., 0]

    mismatch_mask = (
        rolling_context_s1 != base_context_s1
    )
    mismatch_count = int(
        mismatch_mask.sum().item()
    )
    mismatch_rate_percent = (
        100.0
        * mismatch_count
        / max(1, mismatch_mask.numel())
    )

    retained_lookup = torch.zeros(
        KRONOS_S1_VOCABULARY_SIZE,
        dtype=torch.bool,
    )
    retained_lookup[
        resource.retained_original_ids.long()
    ] = True

    # Build the chronological retained-token state from the canonical
    # rolling history. The first project observation uses the requested
    # fallback only until a genuine retained token is observed.
    timeline = build_complete_canonical_s1_timeline(
        original_rolling_cache
    )
    initial_token_by_asset = torch.full(
        (timeline.shape[2],),
        fill_value=int(resource.fallback_original_id),
        dtype=timeline.dtype,
    )
    previous_before = _previous_retained_before_each_bar(
        timeline,
        retained_lookup,
        initial_token_by_asset=initial_token_by_asset,
    )

    context_start_idx = (
        origin_idx
        - context_length
        + 1
    )
    if (
        context_start_idx.min().item() < 0
        or context_start_idx.max().item()
        >= int(original_rolling_cache["num_bars"])
    ):
        raise ValueError(
            "An origin-aligned context start lies outside the "
            "rolling-cache session."
        )

    # Remap the exact origin-aligned context used by the forecasting
    # model. The rolling cache supplies only the causal carry that existed
    # before each context began.
    carry = previous_before[
        rolling_session_idx,
        context_start_idx,
        :,
    ].to(torch.long).clone()
    remapped_context_original = torch.empty_like(
        base_context_s1
    )

    for context_time in range(context_length):
        current = base_context_s1[
            :,
            context_time,
            :,
        ]
        is_retained = retained_lookup[current]
        carry = torch.where(
            is_retained,
            current,
            carry,
        )
        remapped_context_original[
            :,
            context_time,
            :,
        ] = carry

    if not retained_lookup[
        remapped_context_original
    ].all():
        raise AssertionError(
            "The remapped origin contexts contain discarded tokens."
        )

    # Continue the same causal carry through the exact dense future labels
    # stored in the origin-aligned cache.
    original_target = torch.as_tensor(
        base_origin_cache["target_s1"],
        dtype=torch.long,
    )
    remapped_target_original = torch.empty_like(
        original_target
    )
    carry = remapped_context_original[
        :,
        -1,
        :,
    ].clone()

    for future_time in range(
        original_target.shape[1]
    ):
        current = original_target[
            :,
            future_time,
            :,
        ]
        is_retained = retained_lookup[current]
        carry = torch.where(
            is_retained,
            current,
            carry,
        )
        remapped_target_original[
            :,
            future_time,
            :,
        ] = carry

    forward = resource.original_to_compact.to(
        torch.long
    )
    compact_context = forward[
        remapped_context_original
    ]
    compact_target = forward[
        remapped_target_original
    ]

    if (
        compact_context.min().item() < 0
        or compact_target.min().item() < 0
    ):
        raise AssertionError(
            "A remapped original ID has no compact inverse."
        )

    context_tokens = torch.as_tensor(
        base_origin_cache["context_tokens"]
    ).clone()
    context_tokens[..., 0] = compact_context.to(
        context_tokens.dtype
    )

    output = dict(base_origin_cache)
    output.update(
        {
            "format_version": 2,
            "representation": (
                "origin_aligned_kronos_forecasting_tokens"
            ),
            "context_tokens": context_tokens.contiguous(),
            "target_s1": (
                compact_target
                .to(torch.int16)
                .contiguous()
            ),
            "s1_id_space": S1_ID_SPACE_COMPACT,
            "s1_vocabulary_size": int(resource.k),
            "s1_compact_to_original": (
                resource.compact_to_original
                .to(torch.int16)
                .contiguous()
            ),
            "s1_original_to_compact": (
                resource.original_to_compact
                .to(torch.int16)
                .contiguous()
            ),
            "s1_remapping_method": str(resource.method),
            "s1_remapping_resource_hash": str(
                resource.resource_hash
            ),
            "s1_training_coverage_percent": float(
                resource.training_coverage_percent
            ),
            "s1_fallback_original_id": int(
                resource.fallback_original_id
            ),
            "s1_original_source_cache": (
                "full_vocabulary_origin_aligned_cache"
            ),
            "s1_context_remap_source": (
                "authoritative_origin_aligned_context"
            ),
            "s1_history_seed_source": (
                "canonical_rolling_history_before_context_start"
            ),
            "s1_origin_rolling_context_mismatch_count": (
                mismatch_count
            ),
            "s1_origin_rolling_context_mismatch_rate_percent": (
                mismatch_rate_percent
            ),
            "s1_split_name": str(split_name),
        }
    )

    # Exact inverse checks for both model-facing s1 tensors.
    inverse = resource.compact_to_original.to(
        torch.long
    )
    if not torch.equal(
        inverse[
            output["context_tokens"][..., 0].long()
        ],
        remapped_context_original,
    ):
        raise AssertionError(
            "Compact context IDs do not invert exactly."
        )
    if not torch.equal(
        inverse[
            output["target_s1"].long()
        ],
        remapped_target_original,
    ):
        raise AssertionError(
            "Compact target IDs do not invert exactly."
        )

    return output

def compact_s1_to_original(
    values: Tensor,
    *,
    compact_to_original: Tensor,
) -> Tensor:
    """Map compact model IDs back to valid original Kronos IDs."""
    values = torch.as_tensor(values)
    mapping = torch.as_tensor(
        compact_to_original,
        dtype=torch.long,
        device=values.device,
    )
    ids = values.to(torch.long)
    if ids.numel() == 0:
        return ids
    if ids.min().item() < 0 or ids.max().item() >= mapping.numel():
        raise ValueError(
            "Compact s1 ID lies outside the saved mapping."
        )
    return mapping[ids]


def save_mapping_json_summary(
    resource: S1TokenRemappingResource,
    path: str | Path,
) -> Path:
    resource.validate()
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": S1_REMAP_RESOURCE_VERSION,
        "method": resource.method,
        "k": int(resource.k),
        "training_token_count": int(resource.training_token_count),
        "training_coverage_percent": float(resource.training_coverage_percent),
        "fallback_original_id": int(resource.fallback_original_id),
        "resource_hash": resource.resource_hash,
        "retained_original_ids": [
            int(value)
            for value in resource.retained_original_ids.tolist()
        ],
    }
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _smoke_fixture(
    *,
    seed: int,
    dates: Sequence[str],
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    sessions = len(dates)
    bars = 10
    assets = 3
    context_length = 4
    origins = torch.arange(context_length - 1, bars)
    contexts = torch.randint(
        0,
        12,
        (
            sessions,
            origins.numel(),
            context_length,
            assets,
        ),
        generator=generator,
        dtype=torch.int16,
    )
    final_s1 = torch.full(
        (sessions, bars, assets),
        fill_value=-1,
        dtype=torch.int16,
    )
    final_s1[:, origins] = contexts[:, :, -1]
    return {
        "context_s1": contexts,
        "context_s2": torch.zeros_like(contexts),
        "context_mean": torch.zeros(
            (
                sessions,
                origins.numel(),
                assets,
                6,
            ),
            dtype=torch.float32,
        ),
        "context_std": torch.ones(
            (
                sessions,
                origins.numel(),
                assets,
                6,
            ),
            dtype=torch.float32,
        ),
        "s1": final_s1,
        "s2": final_s1.clone(),
        "valid_mask": (
            torch.arange(bars)
            .unsqueeze(0)
            .expand(sessions, bars)
            >= context_length - 1
        ),
        "origin_indices": origins,
        "dates": list(dates),
        "asset_cols": ["A", "B", "C"],
        "context_length": context_length,
        "num_bars": bars,
        "tokenizer_id": "fixture-tokenizer",
        "tokenizer_revision": "fixture-revision",
    }


def _run_smoke_test() -> None:
    train = _smoke_fixture(
        seed=7,
        dates=("2024-01-02", "2024-01-03"),
    )
    validation = _smoke_fixture(
        seed=8,
        dates=("2024-09-03",),
    )

    canonical = concatenate_rolling_caches(
        (train, validation),
        name="smoke canonical",
    )
    training_view = build_rolling_cache_period_view(
        (canonical,),
        start_date="2024-01-01",
        end_date="2024-07-01",
        name="smoke training",
    )
    assert training_view["dates"] == [
        "2024-01-02",
        "2024-01-03",
    ]

    validate_rolling_cache_split_alignment(
        training_view,
        {
            "samples": [
                (None, None, "2024-01-02"),
                (None, None, "2024-01-03"),
            ],
            "asset_cols": ["A", "B", "C"],
        },
        name="smoke training",
    )

    resource = fit_top_k_s1_remapping_resource(
        train,
        k=5,
    )
    remapped_train, remapped_validation, _ = (
        build_train_validation_remapped_rolling_caches(
            train,
            validation,
            resource,
        )
    )
    retained = set(resource.retained_original_ids.tolist())
    assert set(
        torch.as_tensor(remapped_train["context_s1"])
        .unique()
        .tolist()
    ).issubset(retained)
    assert set(
        torch.as_tensor(remapped_validation["context_s1"])
        .unique()
        .tolist()
    ).issubset(retained)
    print("s1 token-remapping CPU smoke test passed.")


if __name__ == "__main__":
    _run_smoke_test()


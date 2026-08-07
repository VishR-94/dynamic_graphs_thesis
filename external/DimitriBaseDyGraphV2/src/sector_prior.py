"""Sector/industry block-structure graph priors for the asset universe.

Builds a block-structured prior adjacency matrix ``A`` from
``company_profiles.csv``, where assets sharing a sector (or industry) are
connected. The resulting matrix is aligned to the ``asset_cols`` order, i.e.
the asset axis of the token tensors, and can be used either as a static graph
prior or to initialise the learnable base graph of a dynamic-graph scorer.
"""

import csv, pathlib
import numpy as np
import torch


def load_profiles(csv_path):
    """Load company profiles from a CSV file.

    Args:
        csv_path: Path to ``company_profiles.csv``.

    Returns:
        Dict mapping each ticker to a ``(sector, industry)`` tuple.
    """
    prof = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            prof[r["Ticker"]] = (r["Sector"].strip(), r["Industry"].strip())
    return prof


def build_block_prior(asset_cols, csv_path, level="sector",
                      self_loops=False, row_normalise=True,
                      off_block=0.0, dtype=torch.float32):
    """Build a block-structure prior adjacency matrix.

    Entry ``A[i, j]`` is high when assets ``i`` and ``j`` share the same
    ``level`` category and low otherwise.

    Args:
        asset_cols: Ordered asset identifiers defining the ``N`` rows/columns.
        csv_path: Path to ``company_profiles.csv``.
        level: Grouping level, ``"sector"`` or ``"industry"``.
        self_loops: If True, keep the diagonal (i == i) as a connection.
        row_normalise: If True, normalise rows to sum to 1, turning ``A`` into
            a stochastic neighbour prior.
        off_block: Baseline weight for non-sharing pairs. 0 gives a hard block
            structure; a small positive value keeps a weak global connection so
            no pair is fully cut.
        dtype: Output tensor dtype.

    Returns:
        Tuple ``(A, labels)`` where ``A`` is an ``(N, N)`` tensor and
        ``labels[i]`` is the category string of asset ``i``.
    """
    prof = load_profiles(csv_path)
    li = 0 if level == "sector" else 1
    missing = [t for t in asset_cols if t not in prof]
    if missing:
        raise KeyError(f"{len(missing)} tickers not in profiles: {missing[:10]}")
    labels = [prof[t][li] for t in asset_cols]
    N = len(asset_cols)

    A = np.full((N, N), float(off_block), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                A[i, j] = 1.0
    if not self_loops:
        np.fill_diagonal(A, 0.0)

    if row_normalise:
        rs = A.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        A = A / rs

    return torch.tensor(A, dtype=dtype), labels


def prior_logits(A, eps=1e-6):
    """Convert a prior adjacency matrix to additive logits.

    Returns ``log(A + eps)``, which is safe for zero entries. The result can be
    added as a bias on QK logits or used to initialise a learnable base graph.

    Args:
        A: Prior adjacency matrix, typically row-normalised.
        eps: Small constant added before the log for numerical stability.

    Returns:
        Tensor of the same shape as ``A`` holding the additive logits.
    """
    A = A.clamp_min(0)
    return torch.log(A + eps)


if __name__ == "__main__":
    cols = "AAPL MSFT NVDA JPM BAC XOM CVX DUK SO".split()
    A, labels = build_block_prior(cols, "/mnt/user-data/uploads/1782392789575_company_profiles.csv",
                                  level="sector", row_normalise=True)
    print("labels:", labels)
    print("A row sums:", A.sum(1))
    print(A)


def init_base_graph_from_prior(model, A, scale=4.0, jitter=0.02, verbose=True):
    """Initialise every dynamic base graph in a model from a block prior.

    Each ``DynamicBaseGraphScorer.base_graph`` of shape ``(H, N, N)`` is set
    from the block prior ``A`` of shape ``(N, N)``, so the dynamic graph starts
    at the sector structure and learns deviations from it.

    ``base_graph`` holds raw logits added to QK^T before the graph activation.
    The prior is normalised to [0, 1] and centred, so that sector-mates receive
    high logits and other pairs low logits. Small Gaussian jitter is added to
    break symmetry across heads. ``scale`` controls the strength of the prior.

    Args:
        model: Model whose modules may expose a ``base_graph`` parameter.
        A: Block prior adjacency matrix of shape ``(N, N)``.
        scale: Multiplier on the centred prior; higher means a stronger prior.
        jitter: Standard deviation of the per-head Gaussian jitter.
        verbose: If True, print a summary and any shape-mismatch skips.

    Returns:
        Number of base_graph parameters that were initialised.
    """
    import torch
    A = A.float()
    A = A / A.max().clamp_min(1e-6)          # normalise to [0, 1]
    base_logits = scale * (A - A.mean())     # centre so it acts as a relative bias
    n = 0
    for mod in model.modules():
        bg = getattr(mod, "base_graph", None)
        if isinstance(bg, (torch.nn.Parameter, torch.Tensor)) and bg is not None and getattr(mod, "use_base_graph", False):
            H = mod.base_graph.shape[0]
            with torch.no_grad():
                bg = base_logits.to(mod.base_graph.device).unsqueeze(0).expand(H, -1, -1).clone()
                bg += torch.randn_like(bg) * jitter
                if bg.shape == mod.base_graph.shape:
                    mod.base_graph.copy_(bg)
                    n += 1
                elif verbose:
                    print(f"  skip: shape {tuple(bg.shape)} != base_graph {tuple(mod.base_graph.shape)}")
    if verbose:
        print(f"initialised base_graph from prior in {n} scorer(s); scale={scale}")
    return n

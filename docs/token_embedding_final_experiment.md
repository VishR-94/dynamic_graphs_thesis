# Final learned-token-embedding experiment

## Purpose

This experiment tests whether replacing the exact post-BSQ bit vector with the
project's existing learned hierarchical token embedding improves the coarse-token
predictive distribution and decoded price forecast.

The existing BSQ-input ModernTCN dynamic-graph model remains an eligible control.
The six new models change the token input and/or temporal/graph architecture, but
retain coarse-only prediction, a one-layer structured-parallel future-query
Transformer, validation-CE checkpoint selection, and ten-path decoded-price
inference.

## Causal tensor contract

```text
observed context IDs                       [B, 60, N, 2]
  s1 embedding + s2 embedding
  + node embedding + context position     [B, 60, N, D]

causal temporal / graph backbone           [B, T_out, N, D]
structured-parallel future predictor       [B, 60, N, D]
coarse-token logits                        [B, 60, N, 1024]
```

No future token, candle, scale statistic, or target-derived quantity enters the
context encoder or graph learner.

## Six training fits

1. ModernTCN D32/K1, patch 8/stride 4, large kernel 15, dynamic graph.
2. The same ModernTCN with a learned free-static graph.
3. Transformer D96/H8, one ST block, dynamic graph.
4. Transformer D96/H8, one ST block, learned free-static graph.
5. Transformer D96/H8, three interlaced ST blocks, dynamic graphs.
6. Transformer D96/H8, three interlaced ST blocks, free-static graphs.

ModernTCN remains one native block. The only change relative to the existing
BSQ-input ModernTCN control is the learned hierarchical embedding before the
same ModernTCN geometry.

For the three-block Transformer:

```text
Temporal 1 -> Graph 1 -> Spatial 1 -> beta 1
Temporal 2 -> Graph 2 -> Spatial 2 -> beta 2
Temporal 3 -> Graph 3 -> Spatial 3 -> beta 3
```

All models retain the separate structured-parallel future-query Transformer with
one layer and four heads.

## Training and checkpoint selection

- Objective: coarse-token cross-entropy over all 60 future positions and assets.
- Validation CE is evaluated every epoch.
- The checkpoint with the lowest validation CE is retained.
- Training stops after ten consecutive non-improving epochs.
- Frozen Kronos decoding is not required during the epoch loop; the selected
  checkpoint is decoded after training.

## Architecture screening

Carry forward:

- the two new models with the lowest validation CE;
- embedded dynamic ModernTCN if not already in the top two;
- the existing BSQ-input dynamic ModernTCN control.

Every candidate is evaluated at:

```text
temperature = 1.0
top-p       = 0.9
top-k       = 0
paths       = 10
seed        = 42
```

All ten complete 60-minute coarse-token paths are decoded independently. Point
forecasts are formed by averaging decoded raw prices. The architecture winner is
the lowest five-horizon mean validation cumulative-log-change MAE.

## Temperature refinement

The architecture winner is compared at temperatures 0.8, 1.0 and 1.2. The
screening result supplies T=1.0; it is not redundantly regenerated. The selected
temperature is the strict lowest five-horizon mean validation Log MAE.

## Close scale/volatility ablation

The winning architecture is retrained once with two context-only Close features:

```text
log(mean Close)
log(std Close / mean Close + eps)
```

Their centre and scale are fitted from training windows/assets only. The two
features are projected to the model input dimension and added once before the
first temporal module. The scale model is evaluated only at the already selected
temperature. The final version is the lower decoded validation Log-MAE model
between matched no-scale and scale runs.

## Saved stochastic artefacts

Each stochastic policy retains all ten decoded raw Close paths, their ensemble
mean, the true future, indices, dates, assets, sampling settings, graphs, metric
tables, and diagnostics. No token IDs, logits, or bit codes are averaged before
decoding.

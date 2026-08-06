# Dynamic absolute-correlation graph for continuous forecasting

This graph is a deterministic, input-conditioned alternative to both the
training-fitted fixed correlation graph and the learned Q/K dynamic graph.
It is available only in the continuous forecaster as:

```yaml
model:
  graph:
    type: dynamic_correlation
    num_heads: 1
    activation: softmax
    add_self_loops: false

  dynamic_correlation:
    threshold: 0.18  # or null for no threshold
    empty_row_policy: strongest
    eps: 1.0e-8
```

## Causal construction

For every forecasting example, the dataset retains the raw observed Close
context with shape `[T,N,1]`. No future candle is included. The graph learner
calculates one-minute Close log returns within that context:

```text
r[t,n] = log(Close[t,n]) - log(Close[t-1,n])
```

It then calculates the contemporaneous Pearson correlation across the `T-1`
return observations for every pair of assets, takes the absolute value, removes
self edges, applies an optional threshold, and row normalises:

```text
A[target, source] >= 0
sum_source A[target, source] = 1
```

The thresholded variant retains values greater than or equal to the configured
threshold. If a threshold produces an empty row, `empty_row_policy=strongest`
restores the strongest non-self edge. A genuinely zero-variance target row
falls back to a uniform non-self row so the stochastic graph contract remains
valid.

## Tensor contract

```text
raw observed Close context: [B,T,N]
selected adjacency:         [B,G,N,N]
orientation:                A[target, source]
```

The same deterministic adjacency is repeated across graph heads. The planned
experiments use one head. The graph has no trainable adjacency parameters; the
spatial message-passing module and learned spatial beta remain trainable.

## Experimental variants

The contaminated curiosity notebook adds two models while keeping the selected
D32/K1/P8-S4/LK15 ModernTCN architecture unchanged:

1. `threshold=0.18`;
2. `threshold=null`, retaining every non-self absolute correlation.

Both are trained with the same cumulative-log-change-MAE objective and the same
optimisation settings as the selected learned-dynamic anchor model. In that
notebook, checkpoint selection is deliberately performed on the test split and
must never be reported as held-out performance.

# Sonnet weather benchmark port

This package is additive: it imports the frozen dissertation architectures and
does not modify the financial data, model, training, or evaluation modules.

## Initial benchmark contract

- Cape Town, test year 2018; validation year 2017; training begins in 1980.
- Exact executable Sonnet slicing and separate input/target `StandardScaler`
  fits, including the repository's inclusive end-date behaviour.
- Six-hourly continuous windows with stride one; midnight and prior-year
  history do not reset a context.
- Tasks `(H,L)`: `(4,28)`, `(12,28)`, `(28,56)`, `(120,240)`.
- Inputs `[B,L,9,5]`; all-node T850 outputs; central node only for validation
  checkpoint selection and headline test metrics.
- Training objective: normalized-space MSE over every output step and node.
- 3ST training additionally supervises every causal internal context origin.
- Headline metrics: Sonnet-code MAE, linear correlation, and weather sMAPE at
  the final forecast step of the central node.

## Output layout

```text
weather/
  modernTCN/
    capetown/
      horizon_4/
        test_year_2018/
  3st_block_transformer/
    capetown/
      horizon_4/
        test_year_2018/
```

Every run contains resumable `best.pt` and `last.pt` checkpoints, resolved data
and model manifests, scaler artifacts, epoch history, full predictions, and the
forecast-origin graph for every exported train/validation/test window. For the
3ST model, mixed, dynamic, and static graphs are retained for all three blocks.

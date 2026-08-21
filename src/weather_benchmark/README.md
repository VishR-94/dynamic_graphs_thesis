# Sonnet weather benchmark port

This package is additive: it imports the dissertation architectures and does
not modify the financial data, model, training, or evaluation modules.  Default
arguments retain the original frozen weather-transfer runs and their existing
checkpoint signatures.

## Benchmark contract

- City and test year are configurable; the current experiment uses Hong Kong,
  test year 2018, validation year 2017, and training beginning in 1980.
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

## ModernTCN kernel sweep

The frozen ModernTCN transfer uses large kernel 15.  The weather-only sweep is:

```python
{
    4:   (7, 11, 15),
    12:  (7, 11, 15),
    28:  (15, 21, 27),
    120: (15, 61, 119),
}
```

Kernels operate on the patch sequence produced by patch size 8 and stride 4.
Each candidate keeps every other architecture and optimization setting fixed.
Candidate directories are isolated as:

```text
weather/modernTCN/hongkong/horizon_120/test_year_2018_kernel_119/
```

The sweep summary ranks candidates by the 2017 central-node final-horizon
validation MSE.  Kernel selection should use that validation rank, not test
metrics.
Because the original 2018 results motivated this sweep, retain the frozen
`test_year_2018` runs as the clean no-weather-tuning transfer result and label
the kernel sweep as a subsequent exploratory optimisation.

## Faster dense-prefix 3ST execution

The 3ST architecture and dense-prefix loss are unchanged.  The optimized Colab
run changes the weather training batch size and applies runtime accelerations:

- training batch size 16 instead of 1;
- validation/export batch size 32 instead of 2;
- two DataLoader workers with prefetching;
- one cached causal attention mask per ST block, context length, and device;
- throttled progress-bar updates;
- per-epoch throughput and peak CUDA-memory logging.

The existing Adam parameter groups and learning-rate schedule remain unchanged.
The optimized run is isolated from any prior batch-size-one checkpoint under:

```text
weather/3st_block_transformer/hongkong/horizon_4/
  test_year_2018_batch_16/
```

## Artifacts

Every run contains resumable `best.pt` and `last.pt` checkpoints, resolved data
and model manifests, scaler artifacts, epoch history, full predictions, and the
forecast-origin graph for every exported train/validation/test window.  For the
3ST model, mixed, dynamic, and static final-context graphs are retained for all
three blocks.  Intermediate in-context graph sequences are not exported.

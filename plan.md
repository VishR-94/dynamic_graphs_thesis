# Definitive project handover: Dynamic graph learning for intraday financial forecasting

This document is a self-contained handover for a new ChatGPT conversation continuing Vish R’s MSc dissertation project. It reconciles the earlier comprehensive handover and correction/addendum. The new conversation should still verify the GitHub repository directly before assuming the code exactly matches this document.

---

## Status legend

Use these labels throughout:

* **Confirmed and implemented**: agreed and code has likely been written.
* **Agreed but not yet implemented**: project decision made, but code not yet done.
* **Provisional / under consideration**: promising idea, not final.
* **Rejected / deprioritised**: considered and set aside, with reason.
* **Unresolved**: needs supervisor decision, repository verification, or further research.

---

# 1. Dissertation research question and intended contribution

## Core research question

**Status: agreed but not fully implemented**

The dissertation investigates whether dynamic graph learning can improve intraday financial forecasting across a universe of equities.

A concise framing:

> Can a model that learns time-varying inter-asset relationships from intraday OHLCV-style data improve multi-horizon equity price forecasting relative to persistence, classical statistical models, temporal neural models, foundation models, and static/adaptive graph baselines?

The project is not simply “forecast prices with a neural network”. The intended contribution is specifically about **learning asset relationships dynamically**.

## Intended contribution

**Status: agreed but not yet fully implemented**

The final contribution should compare:

```text
Static adaptive graph baseline:
    learns one asset graph shared across all windows

Final dynamic graph model:
    learns an input-conditioned graph that changes across windows / regimes
```

The key thesis argument is:

```text
Equity relationships are not fixed.
They vary across time, volatility regimes, market events, and intraday conditions.
Therefore, a dynamic graph model may capture information that static graphs,
independent asset models, and simple multivariate linear models miss.
```

## Current benchmark framing

**Status: confirmed conceptually; classical part implemented**

Benchmark hierarchy:

```text
Classical/statistical baselines:
    persistence
    window mean
    ARIMA / auto-ARIMA
    VAR
    GARCH

Neural temporal baselines:
    TCN / ModernTCN-style model
    Mamba

Foundation-model baselines:
    Chronos-2
    Kronos

Graph baseline:
    static adaptive graph temporal model

Final contribution:
    dynamic graph temporal forecasting model
```

---

# 2. People and project context

## Academic supervisor

**Status: mostly confirmed, but verify formal title/role**

Dr Vasileios Lampos is the academic supervisor / project advisor for the MSc dissertation. The user has biweekly meetings with him. One meeting was cancelled and he requested an email progress update/questions instead.

The project context appears linked to a broader time-series / universal temporal representation theme, but the exact formal project title should be verified.

## External partner / data contact

**Status: confirmed from conversation**

Dimitri at Autonomous Fox / AF Labs is the external partner/contact. He provided guidance on the data and modelling.

Important guidance from Dimitri:

```text
The first raw point of each day is previous-day data and must be dropped.
The data are timestamped by close time, not open time.
Stock split issues are expected in unadjusted intraday data.
Global raw-price normalisation is problematic.
Window-context normalisation is defensible.
Window normalisation may remove volatility-regime information.
Including normalisation constants, especially volatility-like stats, may help.
For financial data, point forecasts may be insufficient.
A probabilistic mean/variance output head is worth considering.
Stride should be added because one-minute sliding windows are highly redundant.
```

---

# 3. Dataset: assets, variables, frequency, date range and dimensions

## Raw dataset

**Status: confirmed, but current path should be re-verified**

The dataset consists of intraday equity candle data stored in PyTorch `.pt` files:

```text
train.pt
val.pt
test.pt
```

Original path used earlier:

```text
/Users/vishalruparelia/Library/CloudStorage/GoogleDrive-vishal@autonomous-fox.ai/
Shared drives/Vishal/data/cached_datasets/exp-24-a95-Candle/session
```

A later corrected dataset may now be in a different folder. The new conversation must verify the active data path from the repository/notebooks.

## Raw split structure

**Status: confirmed for the expected raw candle dataset**

Expected split keys:

```text
samples
asset_cols
channels
grain
market_open
market_close
fill_method
T
F
D
dropped_days
```

Expected metadata observed:

```text
train samples: 188
val samples:   43
test samples:  21

assets: 93
raw time steps per day: 391
channels: 6
frequency: 1 minute
market open: 09:30
market close: 16:00
```

Raw channels:

```text
open
high
low
close
volume
amount
```

Each raw sample is a tuple:

```text
sample[0]: tensor [391, 93, 6]
sample[1]: auxiliary object, usually None
sample[2]: session/day identifier
```

## Date range

**Status: unresolved**

The exact calendar date range was not explicitly recorded. It should be inferred from the day/session strings in the loaded data. We know the data include periods around the NVDA split on 2024-06-10 because that event appeared in data validation.

## Dataset version warning

**Status: important known issue**

A newer folder was previously tested:

```text
exp-1m-95s-24y/session
```

Its first version had the wrong structure for the current pipeline:

```text
no channels key
no D key
sample[0] looked like [390, 94]
sample[1] looked like [391]
included target_asset / target_mode style metadata
```

Interpretation:

```text
That looked like a target/return dataset, not raw multi-asset OHLCV candles.
```

Decision:

```text
Do not patch the current raw-candle pipeline around that wrong format unless the project intentionally changes data target.
```

Dimitri later fixed/regenerated the data. The current repository/data path should be verified.

---

# 4. Data cleaning and preprocessing

## Drop first row of every day

**Status: confirmed and implemented**

Dimitri said the first raw point of each day is the previous day’s last point. Therefore every day is cleaned as:

```text
x_clean = x[1:]
```

Shape change:

```text
raw:   [391, 93, 6]
clean: [390, 93, 6]
```

This is a hard requirement.

## Timestamp convention

**Status: confirmed conceptually; affects plotting/alignment**

Dimitri timestamps bars by **close time**, not open time. This explained earlier apparent mismatches where close prices aligned but open prices looked offset.

The next conversation should be careful when comparing against external sources or plotting intraday bars.

## Candle validity and stock split checks

**Status: implemented as validation logic; exact scripts should be verified**

Checks performed:

```text
high >= open
high >= close
low <= open
low <= close
high >= low
```

Also checked adjacent intraday jumps:

```text
open[t+1] - close[t]
```

and day-boundary jumps:

```text
current_day_open[0] - previous_day_close[-1]
```

Important history:

* Old data had a serious NVDA split-related mixed-scale issue around 2024-06-10.
* Some candles had open/high around 1200 while low/close were around 120.
* This was not just an overnight gap; it suggested mixed adjustment inside a candle.
* Dimitri later fixed the data.
* In fixed data, adjacent intraday open/close large-jump checks flagged nothing.
* Day-boundary large gaps remained only for split-related assets, which is acceptable because windows do not cross days.

## Log-change split

**Status: confirmed and implemented**

For statistical models, a log-change split is created.

For values `x_day`:

```text
log_values = log(clamp_min(x_day, eps))
x_log_change = log_values[1:] - log_values[:-1]
```

Shape:

```text
clean raw x_day: [390, 93, D]
log-change x_day: [389, 93, D]
```

This is used by ARIMA, VAR and GARCH.

## Valid-candle transformed representation

**Status: agreed and implemented, but not yet fully used by final neural models**

The valid-candle transform is intended for neural models and tokenizer work. It currently uses OHLCV and **does not include amount**.

Input raw channels:

```text
open
high
low
close
volume
```

Output transformed channels:

```text
log_close
log_open_to_close
log_upper_wick_ratio
log_lower_wick_ratio
log_volume
```

Definitions:

```text
body_high = max(open, close)
body_low  = min(open, close)

log_close = log(close)
log_open_to_close = log(open / close)

upper_wick_ratio = high / body_high - 1
lower_wick_ratio = body_low / low - 1

log_upper_wick_ratio = log(upper_wick_ratio + eps)
log_lower_wick_ratio = log(lower_wick_ratio + eps)

log_volume = log(volume)
```

Inverse:

```text
close = exp(log_close)
open = close * exp(log_open_to_close)

body_high = max(open, close)
body_low = min(open, close)

upper = exp(log_upper_wick_ratio) - eps
lower = exp(log_lower_wick_ratio) - eps

high = body_high * (1 + upper)
low = body_low / (1 + lower)

volume = exp(log_volume)
```

This guarantees reconstructed candles satisfy:

```text
high >= open
high >= close
low <= open
low <= close
high >= low
positive prices and volume
```

## Amount channel

**Status: unresolved**

Raw data includes `amount`, but the valid-candle transform currently drops it.

Options:

```text
1. Ignore amount for neural models.
2. Add log_amount to the transformed representation.
3. Keep amount separately as an auxiliary input.
```

This should be decided before implementing neural baselines.

## Volume zeros

**Status: confirmed issue; treatment unresolved**

Volume has many zeros. Observed earlier across train:

```text
min volume: 0
1% quantile: 0
median: about 1039
many zero entries
```

This makes log-volume changes unstable:

```text
log(1e-8) ≈ -18.42
```

Latest recommendation:

```text
Use volume/amount as input features if useful.
Do not make volume the primary prediction target unless there is a clear reason.
```

This should be clarified with Dr Lampos.

---

# 5. Train, validation and test methodology

## Split methodology

**Status: confirmed from dataset**

The dataset is already split into train, validation and test files:

```text
train: 188 days
val:    43 days
test:   21 days
```

The next conversation should verify these counts from the actual data.

## Windowed supervised examples

**Status: confirmed and implemented**

A `WindowedCandleDataset` creates forecasting examples within each day.

Given a cleaned daily tensor:

```text
x_day: [T, N, D]
```

typical values:

```text
T = 390
N = 93
D = number of selected channels
```

Current forecasting setup:

```text
context_length = 60
horizons = [1, 5, 15, 30, 60]
stride = configurable
```

Example output:

```text
x: [context_length, N, C_in]
y: [num_horizons, N, C_target]
```

Batch output:

```text
x: [B, 60, 93, C_in]
y: [B, 5, 93, C_target]
```

## Forecast origin and targets

**Status: confirmed and implemented**

For a given `origin_idx`:

```text
context_start = origin_idx - context_length + 1
context_end   = origin_idx + 1
target_indices = [origin_idx + h for h in horizons]
```

Example with `context_length = 60`:

```text
origin_idx = 59
context = rows 0:60
horizon 1 target = row 60
horizon 60 target = row 119
```

Important: `context_end` is exclusive.

## Stride

**Status: confirmed and implemented**

Stride was added because adjacent one-minute windows are highly redundant.

Index logic:

```text
first_origin = context_length - 1
last_origin = T - max_horizon - 1
```

For:

```text
T = 390
context_length = 60
max_horizon = 60
```

we get:

```text
first_origin = 59
last_origin = 329
```

Windows per day:

```text
floor((last_origin - first_origin) / stride) + 1
```

Examples:

```text
stride = 1:  271 windows/day
stride = 15: 19 windows/day
stride = 60: 5 windows/day
```

Classical baselines often used larger stride for speed. Neural models may use a smaller stride but do not necessarily need stride 1.

---

# 6. Potential information leakage and look-ahead risks

## First-row leakage

**Status: confirmed and handled**

Must drop first row of every day. Otherwise previous-day information contaminates the session.

## Window boundaries

**Status: confirmed and handled**

Windows are kept within one day. This avoids overnight leakage and day-boundary stock split issues.

## Last context target

**Status: confirmed and implemented**

The last observed target value must be:

```text
x_day[origin_idx]
```

not:

```text
x_day[-1]
```

Using `x_day[-1]` would leak the end of the day.

## Target indexing

**Status: confirmed and implemented**

Targets are indexed as:

```text
origin_idx + horizon
```

not:

```text
context_end + horizon
```

## Normalisation leakage

**Status: confirmed and implemented**

Window-context normalisation computes statistics from the input context only, not the future target.

For each example:

```text
mean = x.mean(dim=0)
std = x.std(dim=0, unbiased=False)
```

No future data are used.

## Foundation model leakage

**Status: unresolved**

Chronos-2 and Kronos integration has not been implemented. Their input requirements must be checked carefully to avoid future covariate leakage or target leakage.

---

# 7. Repository structure and code design

## Repository structure

**Status: mostly confirmed; verify against GitHub**

Known structure:

```text
src/
    data/
    models/
    training/
    evaluation/
    visualization/
    utils/

configs/
notebooks/
scripts/
docs/

requirements.txt
README.md
.gitignore
```

## Important files

**Status: mostly implemented; verify exact names**

Expected important files:

```text
src/data/load_candle_data.py
src/data/data_generator.py

src/evaluation/prediction_transforms.py
src/evaluation/metrics.py

src/utils/config.py
src/utils/metric_tables.py

src/visualization/candle_plots.py

src/models/persistence.py
src/models/mean.py
src/models/arima.py
src/models/var.py
src/models/garch.py

configs/forecasting.yaml

notebooks/01_plot_candle_data.ipynb
notebooks/baselines.ipynb

scripts/test_preprocess_pipeline.py
```

Exact script names should be verified.

## Naming preferences

**Status: confirmed**

The project uses British spelling:

```text
normaliser
normalisation
unnormalised
```

Avoid American spelling in new code unless an external library requires it.

## Model location

**Status: confirmed**

Model scripts should go in:

```text
src/models/
```

not:

```text
src/baselines/
```

## Code-interaction preference for future assistant

**Status: confirmed user preference**

When reviewing code:

```text
One function/change at a time.
If OK, say “OK”.
If not OK, explain what is wrong and give the corrected version.
```

Do not output code fences with metadata.

Good:

```python
def f():
    pass
```

Bad:

````text
```python id="..."
def f():
    pass
````

````

---

# 8. Configuration and preprocessing design

## Forecasting config

**Status: implemented; verify exact file**

Current config includes:

```text
context_length
stride
horizons
input_channels
target_channels
````

Typical values:

```text
context_length = 60
stride = 1, 15, or 60
horizons = [1, 5, 15, 30, 60]
```

Target channel configurations used:

```text
close-only:
    close

OHLC:
    open, high, low, close

OHLCV:
    open, high, low, close, volume
```

Latest recommendation:

```text
Use close or OHLC as primary targets.
Use volume/amount as inputs, not necessarily as targets.
```

## Normalisation config

**Status: confirmed and implemented**

Current method:

```text
window_context
```

Important settings:

```text
stats_from: context
scope: per_asset_channel
apply_to_target: true
include_stats: true
clip: true
clip_min: -5
clip_max: 5
eps: 1e-8
```

For exact inverse-transform sanity tests, clipping must be disabled:

```text
clip: false
```

because clipping is not invertible.

## Global train normalisation

**Status: rejected / not implemented**

Global raw-price normalisation was considered and rejected for now.

Reason:

```text
Raw prices are nonstationary.
Stock splits make global price statistics misleading.
Window-context normalisation is safer and closer to Kronos-style practice.
```

## Normalisation constants as inputs

**Status: provisional / under consideration**

Dimitri suggested including normalisation constants because window standardisation can remove volatility-regime information.

Latest tentative position:

```text
Store mean/std for inverse transformation.
Consider feeding norm_log_std or volatility-like stats to the model.
Do not initially feed raw norm_mean because price levels can shift mechanically after splits.
```

---

# 9. Baseline models implemented and general design

## Shared baseline interface

**Status: mostly implemented; verify exact signatures**

Baseline classes generally follow:

```text
Model.from_config(config)
model.fit(train_split, val_split)
model.predict(split, output_space="cumulative_log_change")
model.fitted_values(...)
```

Common output dictionary includes:

```text
y_pred
y_true
output_space
channels
horizons
sample_idx
origin_idx
target_indices
```

Some models also include:

```text
asset_cols
selected_orders
selected_lags
failed_models
y_variance
variance_output_space
```

## Persistence

**Status: confirmed and implemented**

Predicts the last observed value for every future horizon.

In cumulative log-change space, persistence predicts zero.

This is a very strong baseline.

## Window mean

**Status: confirmed and implemented**

Predicts the context-window mean for each asset/channel.

This performs worse than persistence because the context average is stale relative to the latest price.

## ARIMA / auto-ARIMA

**Status: confirmed and implemented; exact implementation should be verified**

Univariate ARIMA models are fitted per asset/channel, usually close-only for speed.

Design:

```text
fit on one-step log returns
forecast future one-step returns
convert to cumulative horizon log-changes
```

Auto-ARIMA selected mostly low-order models and performed almost identically to persistence.

Important correction:

```text
SARIMAXResults.apply(refit=False) was discussed as a possible/useful approach.
Verify whether the repository actually uses it.
```

## VAR

**Status: confirmed and implemented**

A VAR model was fitted across all 93 assets for close returns.

Design:

```text
one VAR per target channel
close-only means one 93-dimensional VAR
OHLC would mean four separate VARs, not one 372-dimensional VAR
```

Observed result:

```text
selected lag around 11 with maxlags=15 and AIC
```

VAR was slightly worse than persistence/ARIMA/GARCH in point MAE.

Conclusion:

```text
Simple linear cross-asset lag structure did not improve point forecasts.
```

## GARCH

**Status: confirmed and implemented or nearly implemented; verify exact file**

Univariate GARCH models are fitted per asset/channel.

Variants:

```text
Constant-mean GARCH(1,1)
AR(1)-GARCH(1,1)
```

Return scaling was added and is important.

Reason:

```text
One-minute log returns are very small.
Without scaling, AR coefficients could be numerically absurd, e.g. max abs phi around 41.
After scaling, phi values were about -0.10 to 0.16.
```

Scaling:

```text
fit scaled_returns = raw_returns * return_scale
forecast in scaled units
mean_unscaled = mean_scaled / return_scale
variance_unscaled = variance_scaled / return_scale^2
```

Typical scale:

```text
return_scale = 10000
```

Important interpretation:

If constant-mean GARCH estimates:

```text
mu = 0.0237
```

this is in scaled units. In original log-return units:

```text
mu = 0.0237 / 10000 = 0.00000237
```

Therefore constant-mean GARCH is almost persistence with a tiny drift.

AR(1)-mean and constant-mean GARCH perform nearly the same in point MAE. GARCH’s distinctive value is its variance forecast, not its point forecast.

### GARCH variance caveat

**Status: provisional / important caveat**

Current cumulative variance likely sums one-step innovation variances.

This is acceptable for a simple baseline, but if using GARCH for NLL/calibration later, revisit the predictive variance calculation, especially for AR(1)-GARCH where AR mean dynamics affect multi-step predictive variance.

---

# 10. Baseline results and interpretation

## Qualitative ranking

**Status: confirmed from notebook outputs; exact numbers should be re-run**

Point-forecast MAE ranking:

```text
Persistence ≈ auto-ARIMA ≈ GARCH
VAR slightly worse
Window mean worse than persistence
```

This applies to point MAE/RMSE, not necessarily probabilistic evaluation.

## Approximate close-only MAE values

**Status: approximate; verify against notebook**

Persistence, cumulative log-change MAE, close-only, stride 60:

```text
h=1:  0.000387
h=5:  0.000824
h=15: 0.001365
h=30: 0.001870
h=60: 0.002700
```

Auto-ARIMA close-only was almost identical.

VAR close-only roughly:

```text
h=1:  0.000397
h=5:  0.000837
h=15: 0.001377
h=30: 0.001877
h=60: 0.002703
```

GARCH was also nearly persistence-like.

## Interpretation

**Status: agreed**

Key write-up point:

```text
Persistence is an extremely strong point forecast baseline for intraday prices.
Classical return models mostly forecast near-zero returns.
VAR’s active cross-asset linear corrections do not improve out-of-sample MAE.
GARCH is useful mainly as a conditional volatility model, not as a mean forecaster.
```

This motivates nonlinear temporal and graph models without overclaiming.

---

# 11. Overall model architecture

## Planned final pipeline

**Status: agreed but not fully implemented**

High-level architecture:

```text
raw OHLCV context
→ valid-candle transform
→ window-context normalisation
→ temporal encoder / tokenizer
→ node-wise temporal embeddings
→ graph construction
→ spatial message passing
→ multi-horizon prediction head
→ inverse normalisation
→ inverse feature transform
→ raw-space or cumulative log-change evaluation
```

Important principle:

```text
Temporal modelling should initially be node-wise.
Cross-asset mixing should happen through the graph/spatial module.
```

This avoids letting a generic dense model mix assets before the graph module.

---

# 12. Tokenizer, encoder, BSQ and decoder design

## Kronos-style tokenizer

**Status: provisional / under consideration**

Kronos was discussed as using a two-stage tokenizer + forecasting setup:

```text
Stage 1:
    train tokenizer / autoencoder

Stage 2:
    use learned tokens for forecasting
```

The tokenizer is understood to involve:

```text
Transformer encoder
BSQ quantisation
decoder
```

This should be verified directly against the current Kronos paper/repository before dissertation writing.

## Tokenizer input representation

**Status: agreed if tokenizer is implemented**

The tokenizer should operate on the valid-candle transformed space, not raw OHLCV.

Input transformed channels:

```text
log_close
log_open_to_close
log_upper_wick_ratio
log_lower_wick_ratio
log_volume
```

If `amount` is needed, the representation must be extended.

## Two-stage vs end-to-end

**Status: unresolved**

There was discussion about whether to train tokenizer and forecaster separately or end-to-end.

Current practical position:

```text
Start Kronos-style two-stage if implementing a tokenizer.
Consider end-to-end fine-tuning later only if time permits.
```

This should be clarified with Dr Lampos because there may be different expectations.

## Probabilistic decoder

**Status: provisional / under consideration**

Dimitri suggested probabilistic outputs.

Possible design:

```text
decoder embeddings
→ MLP output head
→ Gaussian mean and variance
→ Gaussian NLL loss
```

The Gaussian would be over transformed features, not raw OHLCV.

This is not yet implemented.

---

# 13. Temporal modelling design

## Candidate temporal baselines

**Status: agreed list; not yet implemented**

Recommended order:

```text
1. Simple repo-native TCN / ModernTCN-style model
2. Mamba
3. Chronos-2
4. Kronos
```

The first neural model should be deliberately simple to validate:

```text
training loop
batching
normalisation
inverse transforms
metrics
comparison against persistence
```

Do not start with a full ModernTCN reproduction if it delays progress.

## Mamba vs S4

**Status: agreed for now**

Decision:

```text
Use Mamba rather than S4 as the main state-space sequence baseline.
```

Reason:

```text
Mamba is a more current selective SSM and better aligned with modern sequence modelling.
S4 is historically important but optional unless an explicit SSM ablation is needed.
```

S4 is not rejected on technical grounds; it is just deprioritised.

## Node-wise temporal modelling

**Status: agreed but not yet implemented**

For temporal baselines and final architecture, a likely pattern is:

```text
input: [B, T, N, C]
reshape: [B*N, T, C]
temporal model per asset
output: [B, N, hidden_dim]
```

Then graph construction/message passing handles cross-asset interaction.

---

# 14. Dynamic and static graph construction

## Static adaptive graph baseline

**Status: agreed but not yet implemented**

The main GNN baseline should not be a basic fixed-graph GCN.

Recommended baseline:

```text
Adaptive Graph TCN / static adaptive graph temporal network
```

Inspired by Graph WaveNet / MTGNN:

```text
learn node embeddings E1, E2
A = softmax(ReLU(E1 E2^T))
```

This learns one static graph shared across all windows.

Proposed file/class:

```text
src/models/adaptive_graph_tcn.py
AdaptiveGraphTCN
```

## Final dynamic graph model

**Status: agreed conceptually; design unresolved**

The final model should learn a graph conditioned on each input window:

```text
A_b = f(H_b)
```

or possibly each time step:

```text
A_{b,t} = f(H_{b,t})
```

Unresolved graph design choices:

```text
per-window or per-time-step
dense or sparse/top-k
directed or symmetric
softmax adjacency or normalised weights
positive-only or signed edges
with or without self-loops
```

## Rejected graph baselines

**Status: rejected / deprioritised**

Basic GCN:

```text
Too weak and not a complete temporal forecasting baseline.
```

STGCN / DCRNN as the main GNN baseline:

```text
Designed mainly for known physical graphs, especially traffic networks.
Less appropriate because the asset graph is unknown.
```

They remain useful literature references.

---

# 15. Spatial message-passing design

## Static graph message passing

**Status: provisional / not yet implemented**

A simple static graph convolution could be:

```text
H: [B, N, hidden_dim]
A: [N, N]

H_neighbour = A @ H
H_out = Linear_self(H) + Linear_neighbour(H_neighbour)
```

## Dynamic graph message passing

**Status: provisional / not yet implemented**

For dynamic adjacency:

```text
A_b: [B, N, N]
H:   [B, N, hidden_dim]

H_neighbour[b] = A_b[b] @ H[b]
```

The next conversation should design this carefully after the first neural temporal baseline is working.

---

# 16. Multi-horizon output design

## Output shape

**Status: confirmed and implemented in baselines**

The project uses direct multi-horizon forecasting:

```text
y_pred: [B, H, N, C_target]
```

where:

```text
H = number of horizons = 5
N = number of assets = 93
```

Current horizons:

```text
[1, 5, 15, 30, 60]
```

## Direct vs recursive forecasting

**Status: agreed for current supervised setup**

The main neural models should output all horizons directly.

Classical models forecast one-step returns and then convert to cumulative horizons.

---

# 17. Loss functions and training stages

## Deterministic neural loss

**Status: provisional**

Likely first neural loss:

```text
MSE, MAE, or Huber in normalised transformed space
```

Evaluation should remain in cumulative log-change space.

## Probabilistic loss

**Status: unresolved / under consideration**

If probabilistic forecasting is included:

```text
Gaussian NLL over transformed features
```

For mean `mu` and variance `sigma^2`:

```text
NLL = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
```

Need to decide with Dr Lampos whether this is core or extension.

## Tokenizer training stages

**Status: provisional**

Possible stages:

```text
Stage 1:
    train tokenizer/autoencoder on transformed windows

Stage 2:
    train forecaster using tokens/embeddings

Optional:
    fine-tune end-to-end later
```

Exact BSQ loss and architecture details need verification.

---

# 18. Forecast reconstruction and inverse normalisation

## Window-context normalisation

**Status: confirmed and implemented**

For each example:

```text
mean = x.mean(dim=0)                         # [N, C]
std = x.std(dim=0, unbiased=False)           # [N, C]
x_norm = (x - mean) / std
```

If target normalisation is enabled:

```text
y_norm = (y - target_mean) / target_std
```

Stored values include:

```text
norm_mean
norm_std
norm_log_std
target_norm_mean
target_norm_std
target_norm_log_std
y_unnormalised
last_context_target
```

Important:

```text
If using transformed data, y_unnormalised is transformed-space, not raw OHLCV.
```

## Inverse normalisation

**Status: confirmed and implemented**

For batched predictions:

```text
y_norm: [B, H, N, C]
mean/std: [B, N, C]

y = y_norm * std.unsqueeze(1) + mean.unsqueeze(1)
```

## Raw to cumulative log-change

**Status: confirmed and implemented**

Given raw future values and last observed raw value:

```text
cumulative_log_change = log(y_raw) - log(last_context_target)
```

For horizon `h`:

```text
log(P[t+h]) - log(P[t])
```

## Convert one-step returns to cumulative horizons

**Status: confirmed and implemented**

For one-step returns:

```text
cumulative_path = cumsum(one_step_returns over time)
select indices [horizon - 1 for horizon in horizons]
```

This is used by ARIMA, VAR and GARCH.

---

# 19. Metrics and aggregation across assets and horizons

## Implemented metrics

**Status: confirmed and implemented**

Metrics:

```text
MAE
MSE
RMSE
```

For tensors:

```text
[B, H, N, C]
```

dimension meanings:

```text
0: batch/window
1: horizon
2: asset
3: channel
```

Important reductions:

```text
overall:
    average all dimensions

per horizon:
    average B, N, C

per asset:
    average B, H, C

per channel:
    average B, H, N

per horizon-channel:
    average B, N
```

## Primary metric

**Status: agreed, but should be confirmed with Dr Lampos**

Recommended headline metric:

```text
cumulative log-change MAE/RMSE
```

Reason:

```text
Raw price errors are not comparable across assets with different price levels.
Cumulative log-change errors are scale-comparable.
```

## Batch aggregation

**Status: confirmed**

For final evaluation, concatenate predictions/truths across batches first, then compute metrics. This avoids final-batch weighting issues.

---

# 20. Baselines and ablation experiments

## Classical baselines

**Status: implemented or nearly implemented**

```text
Persistence
Window mean
ARIMA / auto-ARIMA
VAR
GARCH
```

## Neural baselines

**Status: agreed but not yet implemented**

Recommended:

```text
TCN / ModernTCN-style model
Mamba
Chronos-2
Kronos
Adaptive Graph TCN
```

## Possible ablations

**Status: provisional**

Potential ablations:

```text
close-only vs OHLC targets
with vs without volume/amount inputs
raw/log-return vs valid-candle transformed representation
with vs without norm_log_std inputs
temporal-only vs temporal + graph
static adaptive graph vs dynamic graph
identity graph vs learned graph
dense graph vs top-k sparse graph
Mamba vs TCN
deterministic vs probabilistic output
```

## Rejected / deprioritised approaches

### DCC-GARCH / full multivariate GARCH

**Status: rejected / deprioritised**

Reason:

```text
93 assets imply a very large correlation structure.
It is CPU-expensive.
VAR already suggests simple linear cross-asset structure is not improving point forecasts.
```

### Global raw-price normalisation

**Status: rejected**

Reason:

```text
Raw prices are nonstationary.
Stock splits create discontinuities.
Window-context normalisation is safer.
```

### Full ModernTCN reproduction as first neural step

**Status: deprioritised**

Reason:

```text
First neural model should validate the pipeline.
A simpler TCN-style model is faster and safer.
```

---

# 21. Papers discussed and relevant contribution

This list reflects the discussion/reading plan. The next conversation should verify exact citations and details before writing the dissertation.

## Geometric deep learning and graph foundations

### Bronstein et al. — Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges

**Status: relevant background**

Use for motivating graph inductive bias and non-Euclidean domains.

Relevance:

```text
Assets can be treated as nodes in a relational system.
The graph may be unknown, but relational inductive bias can still be valuable.
```

### Battaglia et al. — Relational Inductive Biases, Deep Learning, and Graph Networks

**Status: relevant background**

Use for graph-network framing:

```text
nodes
edges
global features
message passing
relational inductive bias
```

### Gilmer et al. — Neural Message Passing for Quantum Chemistry

**Status: relevant background**

Use for general message-passing formulation.

### Kipf & Welling — Graph Convolutional Networks

**Status: relevant background / simple baseline reference**

Classic GCN. Useful conceptually, but too static/weak as the main benchmark.

### Veličković et al. — Graph Attention Networks

**Status: relevant background**

Useful for learned attention over neighbours. Vanilla GAT usually assumes a neighbourhood graph.

### Hamilton et al. — GraphSAGE

**Status: relevant background**

Useful for inductive node representation learning and aggregation functions.

## Spatio-temporal graph forecasting

### DCRNN

**Status: literature reference, not main baseline**

Combines diffusion graph convolution with recurrent temporal modelling. Mainly traffic forecasting with known road graph.

### STGCN

**Status: literature reference, not main baseline**

Combines temporal convolutions and graph convolutions. Also assumes known graph.

### Graph WaveNet

**Status: highly relevant baseline inspiration**

Learns adaptive adjacency via node embeddings and combines temporal convolution with graph convolution.

Key idea:

```text
A = softmax(ReLU(E1 E2^T))
```

### MTGNN

**Status: highly relevant baseline inspiration**

Learns graph structure for multivariate time-series forecasting when graph is not known.

### AGCRN

**Status: relevant reference, verify inclusion**

Adaptive graph convolutional recurrent network. Learns node embeddings and adaptive graph convolution. Useful reference for learned static/adaptive graphs.

## Dynamic graph papers

### EvolveGCN

**Status: relevant background**

Models graph neural network parameters as evolving over time.

### DySAT

**Status: relevant background**

Uses attention over structural and temporal dimensions for dynamic graphs.

### TGAT / TGN

**Status: conceptually relevant but not directly matched**

Continuous-time dynamic graph models for timestamped interactions/events. Less directly applicable because this project uses regularly sampled node time series, not event streams.

## Finance-specific graph stock-prediction literature

**Status: important but exact list unresolved**

The dissertation needs a finance-GNN subsection, not only traffic-forecasting graph papers.

Papers/categories to verify:

```text
Temporal relational ranking for stock prediction
Stock relation graph models
Sector / industry / concept graph models
Stock movement prediction with graph attention
Knowledge-graph stock prediction models
HIST-style concept-aware stock models
```

Do not treat this as a final citation list until verified.

## Time-series / foundation models

### Kronos

**Status: highly relevant, details need verification**

Financial K-line/candlestick foundation model. Discussed as using a tokenizer, Transformer autoencoder, BSQ quantisation, and two-stage training.

Need to verify:

```text
normalisation method
tokenizer architecture
BSQ implementation
input/output format
license and feasibility
```

### Chronos-2

**Status: proposed baseline, details need verification**

General time-series foundation model. Need to verify official interface, whether it supports the required multivariate/financial setup, and whether comparison should be zero-shot or fine-tuned.

### Mamba

**Status: selected over S4 for now**

Modern selective state-space model. Preferred as sequence model baseline.

### S4

**Status: optional / deprioritised**

Historically important structured state-space model. Use only if an explicit SSM ablation is needed.

### ModernTCN

**Status: proposed neural baseline**

Modern convolutional time-series architecture. First implementation should probably be a simpler TCN-style model before attempting full reproduction.

---

# 22. Computational constraints

**Status: confirmed practical constraint**

User is working locally on a Mac using VS Code, Terminal and Jupyter.

Observed runtimes:

```text
ARIMA close-only simple: a few minutes
ARIMA OHLC: around 30 minutes
GARCH close-only: around 9 seconds
VAR close-only: feasible
```

Implications:

```text
Use stride for expensive baselines.
Avoid DCC-GARCH for now.
Start neural work with simple models.
Do not overbuild foundation-model integrations before validating the training pipeline.
```

---

# 23. Interpretability and graph-analysis ideas

**Status: provisional**

Possible graph analyses:

```text
visualise learned adjacency matrices
compare learned graph to empirical return correlation
track dynamic graph changes across time/windows
compare high-volatility vs low-volatility windows
analyse graph entropy / sparsity
top-k neighbours per asset
edge turnover over time
degree distribution
sector clustering if sector metadata exists
sector homophily if sector labels are available
compare learned graph around market open vs close
```

Need to verify whether sector metadata is available.

---

# 24. Dissertation writing plan

**Status: provisional**

Possible chapter structure:

```text
1. Introduction
2. Literature review
3. Data
4. Methodology
5. Experiments
6. Results and analysis
7. Conclusion
```

Important emerging narrative:

```text
Persistence is extremely strong for intraday price levels.
Classical statistical models mostly reduce to near-zero return forecasts.
Simple linear cross-asset dynamics do not improve performance.
This motivates nonlinear temporal models and graph-based relational modelling.
The final contribution is dynamic graph learning, not merely sequence modelling.
```

---

# 25. Current unresolved questions ranked by importance

1. **Primary prediction target**

```text
Close-only or OHLC?
Should volume be excluded as a target?
```

Latest recommendation: close or OHLC primary; volume/amount as inputs.

2. **Headline metric**

```text
Should cumulative log-change MAE/RMSE be the main metric?
```

Latest recommendation: yes.

3. **Probabilistic forecasting**

```text
Should Gaussian mean/variance and NLL be core scope or optional?
```

Dimitri recommends it; Dr Lampos should clarify.

4. **Neural benchmark scope**

```text
Which are essential: TCN, Mamba, Chronos-2, Kronos, adaptive graph TCN?
```

5. **Classical baseline sufficiency**

```text
Are persistence, mean, ARIMA, VAR and GARCH enough?
```

Likely yes, but ask Dr Lampos.

6. **Dynamic graph design**

```text
Per-window or per-time-step?
Dense or sparse?
Directed or symmetric?
```

7. **Use of normalisation constants**

```text
Should norm_log_std or other volatility stats be fed to the model?
```

8. **Amount channel**

```text
Ignore, log-transform, or handle separately?
```

9. **Foundation model feasibility**

```text
Can Chronos-2/Kronos be integrated fairly and within time?
```

10. **Finance-GNN literature list**

```text
Need exact papers and citations.
```

---

# 26. Prioritised next steps

## Immediate

1. Send Dr Lampos a progress/questions email.
2. Verify repository state against this document.
3. Commit current baseline code and notebooks.
4. Produce a clean baseline result table.

## Next modelling step

**Recommended next implementation:**

```text
A simple repo-native TCN-style neural baseline
```

Purpose:

```text
validate neural training loop
validate normalisation/inverse-transform pipeline
compare against persistence
establish first supervised neural benchmark
```

## After TCN

Recommended order:

```text
1. Mamba
2. Static adaptive graph temporal network
3. Chronos-2 and/or Kronos, if feasible
4. Final dynamic graph model
```

---

# 27. Questions to send Dr Lampos

Suggested concise questions:

```text
1. Is cumulative log-change MAE/RMSE the right headline metric?
2. Should the main prediction target be close-only, OHLC, or OHLCV?
3. Is it reasonable to use volume/amount as inputs but not primary targets?
4. Are persistence, mean, ARIMA, VAR and GARCH sufficient classical baselines?
5. Which neural baselines are essential: TCN, Mamba, Chronos-2, Kronos, adaptive graph TCN?
6. Should probabilistic forecasting be a core contribution or optional extension?
7. Does static adaptive graph vs dynamic graph sound like a clear contribution?
8. Should normalisation constants, especially volatility-like stats, be included as model inputs?
```

---

# Appendix A: Key tensor shapes

```text
Raw daily sample:
    [391, 93, 6]

Clean daily sample:
    [390, 93, 6]

Log-change daily sample:
    [389, 93, 6]

Window input:
    [60, 93, C_in]

Window target:
    [5, 93, C_target]

Batch input:
    [B, 60, 93, C_in]

Batch target/prediction:
    [B, 5, 93, C_target]
```

---

# Appendix B: Key equations

## One-step log return

```text
r_t = log(P_{t+1}) - log(P_t)
```

## Cumulative log-change

```text
R_{t,h} = log(P_{t+h}) - log(P_t)
```

## Window normalisation

```text
mean = mean_context(x)
std = std_context(x)

x_norm = (x - mean) / std
```

## Inverse normalisation

```text
x = x_norm * std + mean
```

## GARCH(1,1)

```text
r_t = mu + epsilon_t
```

or AR(1) mean:

```text
r_t = c + phi r_{t-1} + epsilon_t
```

variance:

```text
sigma_t^2 = omega + alpha epsilon_{t-1}^2 + beta sigma_{t-1}^2
```

## GARCH scaling

```text
scaled_return = raw_return * return_scale

mean_raw = mean_scaled / return_scale
variance_raw = variance_scaled / return_scale^2
```

---

# Appendix C: Important repository checks

The next conversation should verify:

```text
data path
split shapes
channel names
drop-first-row cleaning
WindowedCandleDataset indexing
stride implementation
normalisation implementation
valid-candle transform and inverse
prediction transforms
metrics
baseline model output contracts
GARCH scaling
baseline result notebook
requirements.txt
.gitignore
```

---

# Instructions for the next ChatGPT conversation

The new conversation should treat this document as a detailed project memory, **not as guaranteed ground truth**. It will have access to the GitHub repository and should verify code before making changes.

Specific instructions:

1. **Inspect the repository before editing.**
   Do not assume file names, class names, function signatures or config keys perfectly match this document.

2. **Verify current data paths and shapes.**
   Confirm the active dataset path, split counts, sample shapes, channel names and cleaned dimensions.

3. **Check implementation status.**
   Confirm which files actually exist and whether the described functions/classes are already implemented.

4. **Preserve existing design choices unless explicitly changed.**
   In particular:

   * use `src/models/`, not `src/baselines/`;
   * use British spelling: `normaliser`, `normalisation`, `unnormalised`;
   * keep window-context normalisation as the current default;
   * do not introduce global raw-price normalisation unless explicitly requested.

5. **Avoid code-fence metadata.**
   Use plain code fences only.

6. **Work one step at a time.**
   If asked to review a function, review only that function. If it is correct, say “OK”. If not, explain the issue and provide the corrected version.

7. **Do not overbuild.**
   The next modelling step should probably be a simple TCN-style neural baseline to validate the training and evaluation pipeline before Mamba, Kronos, Chronos-2 or the final dynamic graph model.

8. **Use current web/primary sources for external model details.**
   Kronos, Chronos-2, Mamba, ModernTCN and package APIs may have changed. Verify against official repositories/papers/docs before implementing or citing.

9. **Clearly mark uncertainty.**
   If a detail is inferred from this handover rather than verified from code/data, say so.

10. **Keep the dissertation framing in mind.**
    The final contribution is dynamic graph learning for intraday multi-asset forecasting, not just another temporal model.

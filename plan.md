# Definitive project handover: Dynamic graph learning for intraday financial forecasting

This document is a self-contained handover for the next ChatGPT conversation continuing Vish R’s MSc dissertation project. The next conversation will run inside the ChatGPT Project and can inspect the current repository directly. This document therefore records the agreed research plan, design decisions, verified data facts, implementation status and sequencing; exact code details should be read from the repository rather than inferred from this handover.

---

## Status legend

Use these labels throughout:

* **Confirmed and implemented**: agreed and verified in the current repository or successful test run.
* **Agreed but not yet implemented**: project decision made, but code not yet done.
* **Provisional / under consideration**: promising idea, not final.
* **Rejected / deprioritised**: considered and set aside, with reason.
* **Unresolved**: needs supervisor decision, repository verification, or further research.

---

# Current status at handover

**Status: confirmed**

The data, preprocessing, classical-baseline and evaluation infrastructure is complete and has passed the repository sanity checks. The active sequence from this point is:

```text
Remaining standalone benchmark phase:
    1. ModernTCN
    2. Kronos
    3. MTGNN only if a suitable implementation is available and straightforward to adapt

Then build the proposed architecture in two separate stages:
    Stage 1: train and validate the tokenizer / autoencoder
    Stage 2: train the downstream forecasting model using the learned tokenizer representation

Then:
    synthetic validation
    real-data comparisons and ablations
    learned-graph analysis
    dissertation results and write-up
```

The remaining benchmarks are independent models. **ModernTCN, Kronos and optional MTGNN do not use the project tokenizer.** The tokenizer belongs only to the proposed architecture and is developed after the standalone benchmark phase.

Verified infrastructure status:

```text
Data loading and first-row cleaning: complete
Chronological train/validation/test repartitioning: complete
Window generation and normalisation: complete
Classical baselines: complete
Raw-only prediction contract: complete
ForecastEvaluator and common metric registry: complete
Long-form evaluation tables: complete
Preprocessing regression script: passing
```

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

**Status: confirmed**

The comparison hierarchy is:

```text
Classical/statistical baselines — implemented:
    persistence
    window mean
    ARIMA / auto-ARIMA
    VAR
    GARCH

Remaining standalone benchmarks:
    ModernTCN — required temporal benchmark
    Kronos — required financial time-series foundation-model benchmark
    MTGNN — optional graph benchmark, only if the available code is easy to run and adapt

Proposed architecture:
    Stage 1: separately trained tokenizer / autoencoder
    Stage 2: downstream dynamic-graph forecasting model using the tokenizer representation
```

Models discussed earlier but no longer in the implementation plan include Mamba, S4, Chronos-2, a separate generic TCN, Graph WaveNet, AGCRN, DCRNN and STGCN. They can remain literature references where useful.

The central empirical progression is:

```text
classical models
→ strong temporal benchmark
→ financial foundation-model benchmark
→ optional static/adaptive graph benchmark
→ proposed tokenizer + dynamic-graph forecasting architecture
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

**Status: confirmed**

The active data are intraday equity candles stored locally in three PyTorch files:

```text
train.pt
val.pt
test.pt
```

The files are not themselves treated as the final chronological partitions. `load_candle_splits()` loads all three as containers for the full 2024 dataset and repartitions their contents according to the configured date boundaries.

The exact local data path is machine-specific and should be read from the current notebook or script. The data are not committed to the repository.

## Raw file structure

**Status: confirmed by direct inspection**

Each file contains:

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

Metadata are identical across all three files:

```text
assets: 93
raw time steps per session: 391
channels: 6
frequency: 1 minute
market open: 09:30
market close: 16:00
fill method: ffill
tensor dtype: torch.float32
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

Every valid sample is:

```text
sample[0]: tensor [391, 93, 6]
sample[1]: None
sample[2]: ISO-like session string, e.g. "2024-01-02 00:00:00"
```

The three physical files contained before repartitioning:

```text
train.pt: 187 valid sessions, 2024-01-02 to 2024-09-30
val.pt:    42 valid sessions, 2024-10-01 to 2024-11-27
test.pt:   20 valid sessions, 2024-12-02 to 2024-12-31
```

Across the files there are:

```text
249 valid sessions
249 unique dates
no cross-file date overlap
chronological ordering within each source file
```

## Date range and dropped sessions

**Status: confirmed**

Valid-session coverage is:

```text
2024-01-02 to 2024-12-31
```

`dropped_days` contains `(day, reason)` tuples rather than samples. The known dropped sessions are:

```text
2024-07-03 — incomplete 211/391
2024-11-29 — incomplete 211/391
2024-12-24 — incomplete 211/391
```

Dropped sessions are not also present among the valid samples. The loader combines and repartitions these records by date along with the valid samples.

## Dataset version warning

**Status: historical context**

An earlier generated folder had a different target/return-style structure rather than the raw multi-asset candle structure expected by this pipeline. It lacked the current `channels` and `D` metadata and should not be used to infer the active format.

The active code should always be checked against the current raw candle files rather than patched around that older incompatible format.

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

**Status: confirmed and implemented**

The three physical `.pt` files are loaded, combined and repartitioned chronologically in memory. Their original split membership is ignored.

Configured boundaries:

```text
train:      day < 2024-09-01
validation: 2024-09-01 <= day < 2024-10-01
test:       day >= 2024-10-01
```

Effective calendar split:

```text
train:      January–August 2024
validation: September 2024
test:       October–December 2024
```

Observed valid-session counts:

```text
train:      167
validation: 20
test:       62
total:      249
```

`load_candle_splits()`:

```text
1. loads train.pt, val.pt and test.pt;
2. verifies matching assets, channels and metadata;
3. combines all valid samples;
4. parses and sorts session dates;
5. rejects duplicate dates;
6. reads val_start and test_start from configs/forecasting.yaml;
7. assigns every valid sample to exactly one chronological split;
8. combines and repartitions dropped-day records by their date;
9. rejects any empty resulting split;
10. returns dictionaries with the same structure as before.
```

All 249 valid sessions are preserved. The downstream call remains unchanged:

```python
train_raw, val_raw, test_raw = load_candle_splits(DATA_DIR)
```

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

Kronos has not yet been integrated as a standalone benchmark. Its official input construction must be checked carefully for future-covariate or target leakage. The proposed tokenizer and downstream model require the same discipline: all representations and normalisation statistics available to a forecast must come only from its input context.

---

# 7. Repository structure and code design

## Repository access

**Status: confirmed**

The next ChatGPT conversation will run inside the Project and can inspect the repository directly. This handover should guide sequencing and preserve decisions, but exact signatures and implementation details should be taken from the current code.

## Repository structure

**Status: verified**

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
plan.md
```

Important current files include:

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

## Completed repository cleanup

**Status: confirmed**

Substantive cleanup completed before this handover:

```text
removed the obsolete make_metric_table pathway and its private helpers
updated the baseline notebook to use make_evaluation_table
removed stale notebook imports and corrected its evaluation description
removed duplicate pyyaml from requirements.txt
removed redundant .gitkeep files from non-empty directories
repaired README/data-documentation code fences and stale split description
added config-driven chronological repartitioning in load_candle_splits
```

Minor formatting-only cleanup was deliberately deprioritised.

## Validation commands

The current state passed:

```bash
python -m compileall -q src scripts
python -m scripts.test_preprocess_pipeline
```

The preprocessing script completed with:

```text
All sanity checks passed.
```

It exercises the real-data loading, repartitioning, cleaning, transformed representations, window dataset, DataLoader batching, normalisation and inversion, candle reconstruction, validity constraints and evaluation transforms/metrics.

Run it as a module from the repository root. Direct execution with `python scripts/test_preprocess_pipeline.py` does not automatically put the repository root on the import path.

## Naming and interaction preferences

**Status: confirmed**

Use British spelling:

```text
normaliser
normalisation
unnormalised
```

Model files belong in:

```text
src/models/
```

not `src/baselines/`.

When reviewing code, work one controlled change at a time. If a function is correct, say “OK”. If it is not, explain the issue and provide the corrected version. Do not add metadata such as `id="..."` to code fences.

---

# 8. Configuration and preprocessing design

## Forecasting and data config

**Status: confirmed and implemented**

`configs/forecasting.yaml` includes the chronological split boundaries:

```yaml
data:
  split:
    val_start: "2024-09-01"
    test_start: "2024-10-01"
```

It also includes the forecasting setup:

```text
context_length = 60
stride = 15
horizons = [1, 5, 15, 30, 60]
input_channels = [open, high, low, close, volume, amount]
target_channels = [open, high, low, close]
```

The loader reads the split boundaries internally from the repository config, while datasets and models receive the returned train/validation/test dictionaries exactly as before.

Volume and amount are model inputs, not current forecast targets. The valid-candle transformed representation presently covers OHLCV and therefore needs an explicit decision about how `amount` enters the proposed tokenizer architecture.

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

**Status: confirmed and implemented**

All five classical baselines use a common raw-only prediction contract:

```text
Model.from_config(config)
model.fit(train_split, val_split)
model.predict(split, batch_size=..., num_workers=...)
model.fitted_values(...)
```

The normal prediction output includes raw-space values and indexing metadata such as:

```text
y_pred
y_true
last_context_target
channels
horizons
sample_idx
origin_idx
target_indices
```

Model classes do not select an evaluation space. The former `output_space` argument and normal output field were removed. `ForecastEvaluator` validates the raw output, performs the required transformation and constructs the persistence comparator.

Some models also expose model-specific metadata:

```text
ARIMA: selected orders and failed models
VAR: selected lags and failed models
GARCH: fitted-parameter diagnostics, y_variance and variance_output_space
```

GARCH mean predictions follow the same raw-only point-forecast contract. Its variance output remains explicitly labelled as cumulative-log-change variance.

The baseline notebook follows:

```text
fit model
→ predict raw values
→ ForecastEvaluator
→ evaluate registered metrics
→ make_evaluation_table
→ display horizon/channel tables
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

## Status of existing results

**Status: historical, pre-repartition evidence**

The notebook contains earlier classical-baseline outputs produced before the final January–August / September / October–December chronological split was introduced. They are useful for understanding model behaviour, but they are not the definitive dissertation result table and should not be quoted as final post-split numbers.

Earlier qualitative ranking:

```text
Persistence ≈ auto-ARIMA ≈ GARCH
VAR slightly worse
Window mean worse than persistence
```

Earlier close-only cumulative-log-change MAE values at stride 60 were approximately:

```text
Persistence:
    h=1:  0.000387
    h=5:  0.000824
    h=15: 0.001365
    h=30: 0.001870
    h=60: 0.002700

VAR:
    h=1:  0.000397
    h=5:  0.000837
    h=15: 0.001377
    h=30: 0.001877
    h=60: 0.002703
```

Auto-ARIMA and GARCH were also close to persistence for point forecasts.

## Interpretation retained from the classical work

**Status: agreed, but final magnitudes require the final experiment run**

```text
Persistence is an extremely strong point-forecast baseline for intraday prices.
Classical return models mostly forecast near-zero returns.
VAR's active linear cross-asset corrections did not improve out-of-sample point accuracy.
GARCH is most distinctive as a conditional-volatility model rather than as a mean forecaster.
```

This motivates nonlinear temporal and relational modelling without claiming that the final graph model must outperform every baseline at every horizon.

---

# 11. Overall project and architecture sequence

## Standalone benchmark phase

**Status: current next phase**

Before developing the project tokenizer or downstream model, complete the independent benchmark set:

```text
1. ModernTCN
2. Kronos
3. MTGNN only if a suitable codebase is available and straightforward to adapt
```

These benchmarks consume the candle data according to their own appropriate interfaces. They are not trained on outputs from the project tokenizer.

## Proposed architecture

**Status: agreed conceptually; not yet implemented**

Only after the remaining standalone benchmarks are evaluated does work begin on the proposed model. Its training is deliberately separated into two stages:

```text
Stage 1:
    train tokenizer / autoencoder

Stage 2:
    train downstream forecasting model using the learned token representation
```

High-level proposed pipeline:

```text
raw six-channel candle context
→ valid-candle-style feature representation, extended or supplemented for amount
→ window-context normalisation
→ separately trained tokenizer / autoencoder
→ discrete tokens or token embeddings
→ node-wise temporal representation
→ input-conditioned graph construction
→ cross-asset message passing
→ direct multi-horizon OHLC prediction head
→ inverse normalisation
→ inverse feature transform
→ raw-output ForecastEvaluator
```

Important principle:

```text
Temporal modelling should initially be node-wise.
Cross-asset mixing should happen through the graph/spatial module.
```

This prevents a generic dense layer from mixing assets before the relational component that the dissertation is intended to study.

---

# 12. Tokenizer, encoder, BSQ and decoder design

## Role and sequencing

**Status: agreed; implementation begins only after the standalone benchmark phase**

The tokenizer belongs to the proposed architecture. It is not a shared preprocessing stage for ModernTCN, Kronos or optional MTGNN.

The proposed architecture is trained in two stages:

```text
Stage 1:
    train and validate the tokenizer / autoencoder

Stage 2:
    train the downstream forecasting model using tokenizer outputs
```

The initial implementation should keep these stages separate. Joint end-to-end fine-tuning can be considered later only if there is a clear experimental reason and sufficient time.

## Tokenizer design

**Status: agreed at high level; exact implementation unresolved**

The current design is inspired by the Kronos tokenizer structure:

```text
Transformer encoder
→ BSQ quantisation
→ discrete tokens / quantised embeddings
→ decoder
```

Kronos itself is also a standalone benchmark. Its benchmark role and the use of a Kronos-inspired tokenizer in the proposed architecture are separate issues.

## Tokenizer input representation

**Status: partially agreed**

The tokenizer should operate on a valid candle representation rather than unadjusted raw price levels. The current implemented transform contains:

```text
log_close
log_open_to_close
log_upper_wick_ratio
log_lower_wick_ratio
log_volume
```

The raw model inputs also include `amount`, so the tokenizer design must decide whether to:

```text
add log_amount to the transformed representation;
provide amount through a separate feature path; or
exclude it from the tokenizer while retaining it elsewhere in the forecasting model.
```

## Stage 1 validation

Before training the downstream forecaster, assess at least:

```text
reconstruction loss and reconstruction quality
validity of reconstructed candles after inversion
token/code utilisation
codebook or bit collapse
token dimensions and temporal granularity
stability across train and validation data
whether the representation retains information useful for forecasting
```

## Decoder and probabilistic outputs

**Status: optional extension, not required for the next implementation step**

A probabilistic decoder or downstream Gaussian head was discussed earlier. When the tokenizer stage begins, its immediate purpose is representation learning and reconstruction. Probabilistic forecasting should not delay the standalone benchmarks or the core dynamic-graph experiments; it can remain an extension if later justified.

---

# 13. Standalone temporal and foundation-model benchmarks

## ModernTCN

**Status: required next benchmark; not yet implemented**

ModernTCN is the remaining strong temporal-only neural benchmark. It should be trained and evaluated before the project tokenizer is developed.

Its role is to answer:

```text
How strong is a modern temporal model without the proposed dynamic cross-asset graph architecture?
```

Implementation should adapt the available ModernTCN design faithfully enough to be a credible benchmark while fitting the repository's direct multi-horizon OHLC output and common evaluation contract.

## Kronos

**Status: required benchmark; not yet evaluated**

Kronos is the financial candlestick/time-series foundation-model benchmark. Its official paper, repository, licence, input format and feasible adaptation/fine-tuning protocol must be checked when implementation begins.

Kronos is independent of the tokenizer later trained for the proposed architecture.

## Previously discussed temporal alternatives

**Status: not in the current implementation plan**

Mamba, S4, Chronos-2 and a separate generic TCN were discussed earlier. They remain relevant literature context but are not planned benchmark implementations. The current required benchmark scope is ModernTCN and Kronos, with MTGNN optional as the graph benchmark.

## Node-wise temporal modelling for the proposed architecture

**Status: agreed conceptually; not yet implemented**

A likely pattern after tokenizer training is:

```text
input token representation: [B, T_token, N, C_token]
reshape: [B*N, T_token, C_token]
temporal model per asset
output node states: [B, N, hidden_dim]
```

Graph construction and message passing then provide the explicit cross-asset interaction.

---

# 14. Dynamic and static graph construction

## Optional MTGNN benchmark

**Status: optional; feasibility check comes after Kronos**

MTGNN is the only remaining graph benchmark under active consideration. It learns a global/adaptive graph for multivariate time-series forecasting and would provide a useful comparison with the proposed input-conditioned dynamic graph.

Implement MTGNN only if a reliable codebase is available and the adaptation is reasonably straightforward. Do not spend disproportionate dissertation time reproducing it from scratch.

Conceptual comparison:

```text
ModernTCN:
    temporal benchmark without an explicit learned asset graph

MTGNN, if implemented:
    learned global/adaptive graph shared across observations

Proposed model:
    graph conditioned on the current tokenizer-derived input representation
```

## Final dynamic graph model

**Status: agreed conceptually; exact design unresolved**

The downstream forecasting model should learn an adjacency conditioned on each input example or market state:

```text
A_b = f(H_b)
```

A graph for every minute inside the input window remains a possible extension:

```text
A_{b,t} = f(H_{b,t})
```

but should not be assumed before the simpler per-window design is evaluated.

Open design choices include:

```text
per-window or per-time-step adjacency
dense or sparse/top-k graph
directed or symmetric relationships
positive-only or signed weights
normalisation of adjacency weights
self-loops
number and placement of message-passing layers
```

## Other graph architectures

**Status: literature references, not planned benchmarks**

Basic GCN, DCRNN, STGCN, Graph WaveNet and AGCRN remain useful background. They are not additional implementation targets unless the project scope changes explicitly.

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

Design this carefully during Stage 2 of the proposed architecture, after the standalone benchmarks are complete and the tokenizer has been trained and validated.

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

## Standalone benchmark training

**Status: next work**

Train and evaluate the independent benchmark models first:

```text
ModernTCN
Kronos
MTGNN only if straightforward
```

Their training does not use the project tokenizer.

## Proposed architecture — Stage 1 tokenizer training

**Status: agreed; follows the benchmark phase**

```text
transformed candle windows
→ encoder
→ BSQ quantisation
→ decoder reconstruction
```

The exact reconstruction, BSQ and auxiliary losses must be designed from the chosen tokenizer implementation and verified against the current Kronos paper/code where used as inspiration.

## Proposed architecture — Stage 2 forecasting

**Status: agreed; follows successful tokenizer training**

```text
trained tokenizer representation
→ downstream temporal and dynamic-graph model
→ direct multi-horizon OHLC forecasts
```

The initial approach should normally freeze the tokenizer while the forecasting model is established. Later end-to-end fine-tuning is an optional ablation, not the default starting point.

## Deterministic forecasting loss

**Status: to be selected during benchmark/model implementation**

Candidate training losses include MAE, MSE or Huber loss in the model's training representation. Final comparison remains through the common evaluation suite in `ForecastEvaluator`.

## Probabilistic loss

**Status: optional extension**

Gaussian NLL and probabilistic heads were discussed, but are not required to begin the remaining benchmarks, tokenizer or deterministic downstream model. Revisit only if the core experiments are complete and there is a clear dissertation benefit.

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

## Evaluation architecture

**Status: confirmed and implemented**

Every model should ultimately provide raw predictions, raw ground truth and the last observed target. Evaluation is centralised:

```text
raw model output
→ ForecastEvaluator validation
→ evaluation-space transformation
→ same-horizon persistence comparator
→ metric registry
→ make_evaluation_table
```

This keeps evaluation-space logic out of individual model classes.

## Registered evaluation metrics

**Status: confirmed and implemented**

```text
cumulative_log_change_mae
mase
relative_mae_vs_persistence
persistence_win_rate
```

Interpretation:

```text
cumulative_log_change_mae:
    MAE after converting raw predictions and targets to cumulative log changes

mase:
    absolute error scaled by a one-step persistence error calculated from the
    training split within sessions

relative_mae_vs_persistence:
    model MAE divided by the same-horizon persistence MAE on the evaluated split

persistence_win_rate:
    proportion of pointwise forecasts with lower absolute error than persistence
```

Metrics are reduced over forecast examples and assets while retaining:

```text
[horizon, target_channel]
```

`make_evaluation_table()` converts a metric dictionary into long form:

```text
metric
horizon
channel
value
```

The former `make_metric_table()` implementation and its private helpers were removed.

## Generic metric utilities

MAE, MSE and RMSE remain useful lower-level functions and synthetic-test utilities, but they are not the current registered benchmark table by themselves.

## Aggregation rule

Concatenate predictions and truths across batches before final metric computation. This avoids weighting the last, potentially smaller, batch incorrectly.

---

# 20. Baselines and ablation experiments

## Classical baselines

**Status: implemented**

```text
Persistence
Window mean
ARIMA / auto-ARIMA
VAR
GARCH
```

## Remaining standalone benchmarks

**Status: agreed**

Required:

```text
ModernTCN
Kronos
```

Optional:

```text
MTGNN — only if an available implementation is straightforward to run and adapt
```

These models are evaluated before tokenizer training and are independent of the tokenizer used by the proposed architecture.

## Proposed architecture

**Status: agreed sequence; not yet implemented**

```text
Stage 1: tokenizer / autoencoder
Stage 2: downstream dynamic-graph forecasting model
```

## Essential final comparisons

```text
classical baselines
ModernTCN
Kronos
MTGNN, if implemented
proposed tokenizer + dynamic-graph forecasting architecture
```

The core architecture ablation should isolate the graph contribution while keeping the surrounding representation and forecaster as comparable as possible:

```text
no graph
versus global/static adaptive graph
versus input-conditioned dynamic graph
```

Other possible ablations remain secondary:

```text
with versus without amount input
with versus without normalisation statistics
dense versus top-k graph
with versus without self-loops
frozen tokenizer versus later end-to-end fine-tuning
deterministic versus probabilistic output, only if probabilistic scope is retained
```

## Synthetic validation

**Status: required after the proposed forecasting model is working**

Real equity data do not provide an observable true graph. Controlled synthetic data should therefore be generated with known cross-node dependencies and known changes in graph structure or regime.

The synthetic study should compare:

```text
temporal/no-graph model
global or static adaptive graph model
input-conditioned dynamic graph model
```

It should evaluate both:

```text
forecasting performance when relationships change
graph recovery or tracking of the known changing structure
```

## Rejected or deprioritised expansion

```text
DCC-GARCH / full multivariate GARCH
Mamba
S4
Chronos-2
an additional generic TCN
DCRNN
STGCN
Graph WaveNet as a separate benchmark
AGCRN as a separate benchmark
```

These may be cited where relevant but are not part of the current implementation sequence.

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

**Status: optional benchmark and relevant reference**

Learns graph structure for multivariate time-series forecasting when the graph is not known. It is the only additional graph benchmark currently under consideration, and should be implemented only if the available code is easy to run and adapt.

### AGCRN

**Status: relevant literature reference, not planned benchmark**

Adaptive graph convolutional recurrent network. Learns node embeddings and adaptive graph convolution. Useful for discussing learned global/adaptive graphs, but not part of the current implementation plan.

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

**Status: required standalone benchmark and tokenizer inspiration; implementation details need verification**

Financial K-line/candlestick foundation model. It is a required independent benchmark. Its tokenizer architecture also motivates parts of the proposed project's Stage 1 tokenizer, but these are separate uses.

Verify from the official paper/repository:

```text
normalisation method
tokenizer architecture and BSQ implementation
input/output format
pretrained checkpoints and adaptation protocol
licence
compute requirements
```

### ModernTCN

**Status: required next benchmark**

Modern convolutional time-series architecture and the next model to implement after committing the current infrastructure work.

### MTGNN

**Status: optional graph benchmark**

See the spatio-temporal graph section. Evaluate only if an accessible implementation is straightforward to adapt.

### Chronos-2

**Status: literature context, not planned implementation**

General time-series foundation model that was previously considered. It is not in the current benchmark plan.

### Mamba

**Status: literature context, not planned implementation**

Modern selective state-space model. Previously considered as a sequence baseline, but removed from the current implementation scope.

### S4

**Status: literature context, not planned implementation**

Historically important structured state-space model. Retain for background only where useful.

---

# 22. Computational constraints

**Status: confirmed practical constraint**

The user is working locally on a Mac using VS Code, Terminal and Jupyter.

Observed approximate classical-model runtimes included:

```text
ARIMA close-only simple: a few minutes
ARIMA OHLC: around 30 minutes
GARCH close-only: around 9 seconds
VAR close-only: feasible
```

Implications:

```text
use stride where appropriate for expensive models
avoid DCC-GARCH and unnecessary benchmark expansion
implement ModernTCN before spending time on foundation/graph integrations
verify Kronos and MTGNN compute and code availability early
prefer adapting reliable official code to rebuilding large published systems
keep the tokenizer and downstream forecaster as two controlled stages initially
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
ModernTCN tests the value of a strong temporal model.
Kronos provides a finance-specific foundation-model comparison.
Optional MTGNN tests a learned global/adaptive graph if feasible.
The proposed architecture then learns its own discrete representation before
forecasting with an input-conditioned dynamic graph.
The final contribution is dynamic graph learning, not merely sequence modelling.
```

---

# 25. Current unresolved questions ranked by importance

1. **ModernTCN adaptation**

```text
How should the official architecture be adapted to [B, T, N, C] inputs and direct
multi-horizon OHLC outputs while remaining a credible ModernTCN benchmark?
```

2. **Kronos evaluation protocol**

```text
Which official checkpoint/interface should be used?
Zero-shot, fine-tuned or otherwise adapted?
How should its native representation and horizons be compared fairly with this project?
```

3. **MTGNN feasibility**

```text
Is a reliable codebase available?
Can it be adapted to the data and common output/evaluation contract without a large rebuild?
```

4. **Tokenizer input representation**

```text
How should amount be incorporated?
What temporal patch/token granularity should be used?
Should normalisation statistics be included as features?
```

5. **Tokenizer architecture and loss**

```text
Exact Transformer encoder/decoder design
BSQ bit dimension and quantisation details
reconstruction and auxiliary loss terms
code-utilisation and collapse diagnostics
```

6. **Tokenizer use in Stage 2**

```text
Use discrete token IDs, quantised embeddings or encoder states?
Freeze the tokenizer initially?
Is later end-to-end fine-tuning a worthwhile ablation?
```

7. **Downstream dynamic graph design**

```text
Per-window or per-time-step graph?
Dense or sparse?
Directed or symmetric?
Positive-only or signed?
Where should message passing enter the forecaster?
```

8. **Synthetic validation design**

```text
What dynamic graph data-generating process should be used?
How should graph recovery and regime tracking be measured?
```

9. **Headline presentation of the metric suite**

```text
The implemented suite is fixed, but which metric should lead the dissertation tables
and abstract: cumulative-log-change MAE, MASE or relative improvement over persistence?
```

10. **Probabilistic extension**

```text
Keep outside the core deterministic scope, or include only after the main experiments?
```

11. **Finance-GNN literature list**

```text
The exact finance-specific papers and citations still need to be finalised.
```

---

# 26. Prioritised next steps

## Immediate

1. Commit and push the completed data, baseline and evaluation infrastructure together with this updated handover.
2. Inspect the ModernTCN paper and official/reliable codebase.
3. Design the repository integration and train/evaluate ModernTCN under the common raw-output and `ForecastEvaluator` contract.

## Complete the standalone benchmark phase

```text
1. ModernTCN
2. Kronos
3. MTGNN only if code availability and adaptation are straightforward
```

Do not begin tokenizer training until this benchmark phase is complete. These models do not use the project tokenizer.

## Build the proposed architecture

After the standalone benchmarks:

```text
Stage 1:
    design, train and validate the tokenizer / autoencoder

Stage 2:
    design and train the downstream forecasting model using tokenizer outputs
    add the input-conditioned dynamic graph and message passing
```

## Final experimental stage

```text
1. controlled synthetic experiments with known changing graphs
2. real-data comparison with all completed benchmarks
3. no-graph versus static/global-graph versus dynamic-graph ablations
4. tokenizer and representation ablations where informative
5. learned-graph analysis
6. dissertation results, limitations and write-up
```

---

# 27. Questions to send Dr Lampos

The earlier broad list has been narrowed by decisions already made. Useful remaining questions include:

```text
1. Does the benchmark scope — ModernTCN, Kronos and optional MTGNN — look sufficient?
2. For Kronos, what comparison protocol would be most defensible?
3. For the proposed tokenizer, should amount be included directly, handled separately or omitted?
4. Should the tokenizer be frozen for the first downstream experiments, with end-to-end fine-tuning only as an ablation?
5. Is a per-window dynamic graph sufficient for the core contribution, or is per-time-step variation expected?
6. Which implemented metric should be the headline result alongside persistence-relative measures?
7. Should probabilistic forecasting remain outside the core scope unless time permits?
8. Does the planned synthetic graph-recovery study adequately support the dynamic-graph contribution?
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

The next conversation has direct repository access and should inspect code before editing. The following current facts were already verified during this handover:

```text
active raw sample shape: [391, 93, 6]
clean daily shape: [390, 93, 6]
249 unique valid sessions
chronological config-driven split
first-row cleaning
WindowedCandleDataset indexing and stride
window-context normalisation and inverse
valid-candle transform and inverse
raw-only classical baseline outputs
ForecastEvaluator registry
make_evaluation_table workflow
GARCH return scaling
preprocessing regression script passes
```

Before implementing each new external model, verify its official code, licence, dependencies, expected data layout and output contract rather than relying on this handover for package/API details.

---

# Instructions for the next ChatGPT conversation

The next conversation runs inside the ChatGPT Project and can inspect the current repository directly. Treat this document as the record of agreed decisions and sequencing, while treating the repository as the source of truth for exact code.

Specific instructions:

1. **Inspect the repository before editing.**
   Do not infer exact signatures, config keys or class internals from this handover when the code is available.

2. **Preserve the project sequence.**

   ```text
   remaining standalone benchmarks:
       ModernTCN
       Kronos
       optional MTGNN

   then proposed architecture:
       Stage 1 tokenizer
       Stage 2 downstream forecasting model
   ```

   The standalone benchmarks do not use the project tokenizer.

3. **Do not expand the benchmark list without an explicit decision.**
   Mamba, S4, Chronos-2 and additional graph architectures are not current implementation targets.

4. **Preserve existing design choices unless explicitly changed.**

   * use `src/models/`, not `src/baselines/`;
   * use British spelling: `normaliser`, `normalisation`, `unnormalised`;
   * keep window-context normalisation as the current default;
   * do not introduce global raw-price normalisation without an explicit decision;
   * retain the raw-only model output contract and central `ForecastEvaluator`;
   * retain the chronological config-driven split;
   * current inputs are OHLC, volume and amount; current targets are OHLC.

5. **Work one controlled step at a time.**
   The user prefers reviewing one function or change at a time and validating it before proceeding.

6. **Do not overbuild.**
   Begin with ModernTCN. Verify external code feasibility before designing a large integration. Implement MTGNN only if straightforward.

7. **Use current primary sources for external model details.**
   ModernTCN, Kronos, MTGNN and package APIs may change. Check official papers, repositories and documentation when implementing or citing them.

8. **Keep benchmark and proposed-architecture code conceptually separate.**
   The tokenizer is for the proposed architecture, not for ModernTCN, Kronos or MTGNN.

9. **Clearly mark uncertainty.**
   Do not invent architecture details, codebases or recommendations that are not in the repository, this document or a verified source.

10. **Keep the dissertation contribution in view.**
    The final contribution is an input-conditioned dynamic graph forecasting model trained after a separate tokenizer stage, not merely another temporal benchmark.

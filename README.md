# Dynamic Graph Learning for Financial Time-Series Forecasting

Research code for a thesis on dynamic graph learning and financial forecasting.

## Data

The proprietary candle datasets are not included in this repository.

Expected local structure:

```text
data/candle/session/train.pt
data/candle/session/val.pt
data/candle/session/test.pt
```

## Project structure

```text
configs/           Experiment configuration files
docs/              Project notes and data documentation
notebooks/         Exploration and benchmark notebooks
scripts/           Runnable inspection and validation scripts
src/data/          Dataset classes and preprocessing
src/models/        Forecasting models and baselines
src/training/      Training loops and losses
src/evaluation/    Forecast evaluation and transformations
src/visualization/ Plotting utilities
src/utils/         General helper functions
```
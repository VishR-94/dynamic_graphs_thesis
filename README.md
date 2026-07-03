# Dynamic Graph Learning for Financial Time-Series Forecasting

Research code for a thesis on dynamic graph learning and financial forecasting.

## Data

The proprietary candle datasets are not included in this repository.

Expected local structure:

```text
data/candle/session/train.pt
data/candle/session/val.pt
data/candle/session/test.pt

Project Structure:

configs/          Experiment config files
docs/             Project notes and data documentation
notebooks/        Exploration notebooks
scripts/          Runnable experiment scripts
src/data/         Dataset classes and preprocessing
src/models/       PyTorch models
src/training/     Training loops, losses, metrics
src/evaluation/   Forecast and graph evaluation
src/visualization Plotting utilities
src/utils/        General helper functions
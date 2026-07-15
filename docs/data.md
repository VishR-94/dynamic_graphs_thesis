# Data

The proprietary candle datasets are not included in this repository.

Expected local structure:

```text
data/candle/session/train.pt
data/candle/session/val.pt
data/candle/session/test.pt

On Colab, the data should be loaded from Google Drive.

The first raw time point of each day should be dropped before modelling because it belongs to the previous trading day.

Raw daily sample shape:
[391, 93, 6]

Cleaned daily sample shape:
[390, 93, 6]

Channels:
open, high, low, close, volume, amount
'''

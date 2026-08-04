# Final Tokenized ModernTCN–Dynamic-Graph Experiment

## Purpose

This experiment reuses the selected **architecture and optimisation settings**
from the continuous ModernTCN graph sweep, while replacing the continuous
OHLCV input/output contract with the frozen Kronos tokenizer contract.

Selected continuous architecture:

```text
mtg_s3_d32_k1_p8s4_lk15_dynamic_g1_h32_lr0p0001_glr0p0005
```

In words:

- Stage-3 ModernTCN graph refinement;
- hidden dimension 32;
- one ModernTCN block;
- patch size 8 and stride 4 (15 observed patch positions from 60 minutes);
- large temporal kernel 15 and small kernel 5;
- FFN ratio 1 and ModernTCN dropout 0.05;
- one input-conditioned dynamic graph head with graph hidden dimension 32;
- one spatial message-passing layer with a learned scalar spatial gate;
- backbone learning rate `1e-4`;
- graph-learning rate `5e-4`.

The trained continuous checkpoint is **not** loaded into the token model. The
input stem and output head have changed, so the token model is trained from
scratch with the same selected architecture and hyperparameters.

## Frozen tokenizer contract

The origin-aligned token caches are the output of the frozen Kronos tokenizer
encoder and BSQ quantizer. The encoder is therefore not rerun in every epoch.
For each observed minute and asset, the cache provides the original Kronos
coarse/fine IDs:

```text
context token IDs: [B, 60, N, 2] = [s1, s2]
```

The IDs are deterministically converted back to the exact post-BSQ code:

```text
s1 ID -> 10 least-significant-bit-first bipolar components
s2 ID -> 10 least-significant-bit-first bipolar components
concatenate -> 20 components in {-1,+1}/sqrt(20)
```

No continuous pre-BSQ latent is used. Remapped 150/250-token caches are
rejected because they do not preserve the exact original BSQ code.

## Model tensor flow

```text
original s1/s2 context IDs                 [B, 60, N, 2]
post-BSQ binary spherical code             [B, 60, N, 20]
assets folded into batch                   [B*N, 20, 60]
official ModernTCN feature map             [B*N, 20, 32, 15]
learned within-asset token-variable pool   [B*N, 32, 15]
restore asset axis                         [B, 15, N, 32]
dynamic graph from final observed patch    [B, 1, N, N]
spatial message passing                    [B, 15, N, 32]
learned temporal/spatial beta blend        [B, 15, N, 32]
structured-parallel future predictor       [B, 60, N, 1024]
```

The future head predicts only the 1,024-way **coarse** token `s1` at every one
of the 60 future minutes. No future `s2` head, loss, sampling, or fine decoder
is used. Historical `s2` remains an observed input feature.

## Training and checkpoint selection

Training uses ordinary coarse-token cross-entropy on all 60 positions:

```text
training target: target_s1 [B, 60, N]
training output: s1 logits [B, 60, N, 1024]
```

Temperature does not enter the training loss. `best_checkpoint.pt` is
selected by the lowest teacher-forced September validation coarse-token
cross-entropy:

```text
selection metric: validation_token_loss
coarse-only interpretation: validation s1 cross-entropy
selection positions: all 60 future minutes
selection assets: all 93 assets
patience unit: validation epochs
```

Cross-entropy is evaluated every epoch and scores the complete categorical
distribution that is later sampled. Frozen-decoder argmax validation is no
longer required on every epoch. It may be run periodically as a diagnostic,
and after training the runner reloads the exact CE-selected checkpoint and
regenerates its deterministic argmax price-space metrics and saved artefacts.
Temperature remains an inference-only calibration parameter.

## Inference-only temperature sweep

After the checkpoint is frozen, the same weights are evaluated at:

```text
temperatures: [0.3, 0.45, 0.6, 0.8, 1.0]
top-p: 0.9
top-k: 0
sample count: 10
sampling seed: 42
```

For each temperature:

```text
shared observed-context encode/graph/logits
-> sample 10 complete 60-position s1 paths
-> decode every path separately with the frozen coarse decoder
-> obtain [10, B, 60, N, 5] raw OHLCV paths
-> average across the 10 decoded continuous paths
-> select Close at [1, 5, 15, 30, 60]
-> ForecastEvaluator
```

Token IDs, BSQ bits, logits, or probabilities are never averaged before
decoding. The graph is inferred only from the observed context and is shared
by all ten sampled paths.

The temperature is selected using the strict all-five-horizon September
validation mean Log MAE. The argmax result is saved as a reference but is not
eligible to define the selected stochastic temperature.

## Causal and data-boundary guarantees

- Context tokens and context normalisation statistics use observed rows only.
- The dynamic graph uses only the final causal ModernTCN representation of the
  observed context.
- No true future token or raw future candle enters graph inference or sampling.
- The graph is fixed across all 60 predicted minutes and all ten paths for a
  given forecast window.
- Training and temperature selection use January-August training and September
  validation only.
- The held-out proposed-model test split remains untouched until the model and
  temperature are frozen.

## Primary run preset

```text
modern_tcn_dynamic_coarse_mc10
```

The preset is defined in `configs/dynamic_graph.yaml`. Training must use the
original full-codebook origin-aligned train/validation caches.

# Final repository and Graph Hub audit — 4 August 2026

## Scope

This audit was performed against `dynamic_graphs_thesis-main (19).zip`, the repository state after the temperature-sweep compatibility/resume fix. It focused on:

- the final continuous ModernTCN–dynamic-graph path;
- the final tokenized ModernTCN–dynamic-graph path;
- validation-CE checkpoint selection;
- inference-only temperature sampling and per-policy resume;
- saved prediction, graph, token and sampled-price-path artefacts;
- the analysis code used for the dissertation write-up.

## Automated checks completed

The following checks passed in the audit environment:

```text
python -m compileall -q src scripts
python -m scripts.test_error_quantile_metrics
python -m scripts.test_continuous_forecaster
python -m scripts.test_modern_tcn_graph_sweep
python -m scripts.test_tokenized_modern_tcn
python -m scripts.test_dynamic_graph_evaluation
python -m src.models.dynamic_graph.config
python -m src.models.dynamic_graph.future_predictor
python -m src.models.dynamic_graph.model
```

All Python code cells in every repository notebook were parsed successfully.

The tokenized contract test exercises the current critical path, including:

- code-only commit changes during frozen evaluation;
- rejection of genuine model/data incompatibilities;
- reusable per-temperature policy artefacts;
- regeneration when required sampled-path artefacts are missing;
- ten complete 60-step coarse-token paths;
- independent frozen decode of all paths;
- averaging only after decoding to continuous prices;
- retention of all ten raw Close paths;
- dynamic graph sharing across sampled future paths;
- coarse-only operation without a future-s2 head.

## Checks that require the user environment

The archive intentionally contains empty Git submodule directories. The following could not be re-executed here and remain Colab/local-environment checks:

- the full official ModernTCN forward path against the pinned submodule;
- the frozen Kronos decoder on all 380 validation windows;
- real-data preprocessing tests requiring the Google Drive candle files;
- the BaseDyGraph official-adapter test requiring the pinned submodule.

The repository notebooks already initialise the pinned submodules and verify their commit hashes before GPU work.

## Audit findings

### Production training and temperature inference

No additional blocking defect was found in the current final training or temperature-sweep path. The strict training/resume signature remains unchanged, while frozen evaluation uses the separate compatibility signature added by the preceding audit fix. The temperature sweep is resumable per policy and preserves the original training provenance.

### Analysis compatibility defects found and fixed

The previous `graph_hub.ipynb` and `dynamic_graph_evaluation.py` were designed around the older token-model artefact schema. They could not reliably analyse the final continuous model because:

- continuous runs use top-level `model`, while tokenized runs use `models.dynamic_graph`;
- continuous prediction artefacts are direct `PredictionResult` mappings, while tokenized artefacts wrap `prediction_result`;
- continuous graph artefacts are direct graph mappings, while tokenized artefacts wrap `graph_artifacts`;
- selected-temperature policies live under `temperature_sweep/<policy>`;
- sampled decoded paths had no coverage, CRPS or plotting helpers;
- graph selection supported only a global window index rather than date plus within-date window.

The new unified analysis API validates and supports both families without changing saved artefacts.

## New analysis contracts

The analysis module now supports:

- automatic continuous/tokenized run-family detection;
- exact selected-temperature policy resolution;
- mixed-family full-metric tables;
- model/run overview and saved-configuration summaries;
- exact graph selection by date and one-based window within date;
- raw adjacency heatmaps with `A[target, source]` orientation;
- strongest incoming/outgoing connection tables;
- graph entropy, effective-neighbour and top-mass summaries;
- all-ten-path plots against the full realised future Close path;
- central predictive interval coverage in cumulative-log-return space;
- scale-free interval width in log-return basis points;
- asset-level coverage tables;
- empirical CRPS and sample-dispersion tables;
- finite-ensemble truth-rank histograms;
- temperature-sweep result loading and winner marking.

A new CPU contract test constructs synthetic continuous and tokenized run directories and verifies all of these artefact schemas and calculations.

## Remaining interpretation limitations

- Ten sampled paths provide a coarse empirical distribution. A nominal 90% interval is not a high-resolution estimate and should be described as a finite-ensemble diagnostic.
- The learned adjacency is the routing matrix used by the model, not causal evidence.
- The current project metric named `relative_mae_vs_persistence` remains in raw price space. It was deliberately left unchanged pending the final metric-selection decision.
- The optional graph-reliance, gate-sweep and topology-counterfactual functions still reconstruct the tokenized model. The continuous final model is fully supported for saved-artifact metrics, point forecasts and exact graph analysis; a separate live continuous intervention runner was not added because it is not required for the final write-up workflow.

## Files changed

```text
src/evaluation/dynamic_graph_evaluation.py
scripts/test_dynamic_graph_evaluation.py
notebooks/graph_hub.ipynb
docs/final_repo_and_graph_hub_audit_2026-08-04.md
```

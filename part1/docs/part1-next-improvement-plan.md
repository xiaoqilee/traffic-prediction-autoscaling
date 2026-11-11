# Part 1 Improvement Plan and Completion Record

Date: November 11, 2026. Status: implemented and evaluated on validation data. The held-out real test remains reserved for the final agreed comparison.

This plan follows the [experimental checkpoint requirements](https://docs.google.com/document/d/1W9RW1Q1a7GEIcyZkXUUm8coKBzvIeYgKPT4z8uI7oFs/edit) and the [previous work allocation](https://docs.google.com/document/d/1d0uzJbH7SKwlZeDg4j1jNaWaIFzAnCxaw0YdfLIlHW0/edit). It was originally prepared before implementation and is now updated in English to record what was completed. The [checkpoint report](part1-checkpoint.md) contains actual findings; the [handoff report](part1-delivery.md) explains how Parts 2 and 3 should use the outputs.

## Objective

Determine whether stronger penalties for underprediction reduce high-flow forecast shortfall and consecutive errors when data, features, splits, tree configuration, and postprocessing are fixed. Measure overprediction at the same time. Complete the forecasting work independently of service simulation.

## Fixed comparison

- Preserve `steady_01`, `steady_02`, `bursty_01`, and `bursty_02` and their original traffic data.
- Preserve train days 1–8, validation days 9–11, and held-out test days 12–14.
- At boundary t, use observations through t-1 to forecast t+3; retain all 16 existing causal features in their original order.
- Preserve the original persistence, Ridge and squared-error histogram-boosting models and forecasts. Original settings were selected by validation MAE.
- Inherit the saved ordinary tree's complete estimator configuration. Keep max_iter=100, learning_rate=0.1, min_samples_leaf=20, early_stopping=false, random_state=42 and the single-thread budget. Keep max_leaf_nodes=31 on bursty_01 and 15 on the other applications.
- Fit new models on train only. Keep nonnegative clipping and floating-point count outputs; do not add headroom, smoothing, queue feedback, new features, new workload selection, or further hyperparameter search.
- Score the same 4,317 target minutes per application: 11523–15839, inclusive. Use training q95 for high-flow thresholds.

## Training ablation

| Group | Model identifier | Objective | Role |
| --- | --- | --- | --- |
| A | tree | Original squared error | Ordinary-training reference |
| B | tree_q50 | Quantile loss, q=0.50 | Symmetric control |
| C | tree_underprediction | Quantile loss, q=0.75 | Underprediction-penalizing candidate |

For residual r=y-predicted, quantile loss is `q*max(r,0) + (1-q)*max(-r,0)`. The candidate's equal-magnitude underprediction/overprediction penalty ratio is 3:1. Values 0.50 and 0.75 were fixed in advance, not searched. HistGradientBoostingRegressor supports this [public interface](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

A→C measures the complete objective change; B→C measures penalty direction within the same loss family. Do not attribute every A→C difference to asymmetry, because squared-error and median regression have different targets. Learned trees can differ even though the estimator implementation and other parameters are fixed.

## Completed tasks

| Order | Task | Completion evidence |
| --- | --- | --- |
| P0-1 | Review existing baselines and freeze inputs/case rules | Input hashes and case rules in design_manifest.json; baseline reproduction table |
| P0-2 | Add overprediction, high-flow misses/errors, material shortfall runs, daily and normalized-group metrics | Expanded diagnostics; hand-calculated unit cases; exported metric tables |
| P0-3 | Add an independent ablation runner/configuration | src/ablation.py and configs/part1-underprediction.json; eight new training fits |
| P0-4 | Export predictions, saved models, case evidence and figures in a separate directory | outputs/underprediction_ablation_v1/ and experiment_manifest.json |
| P0-5 | Explain results and prepare a usable English handoff | part1-checkpoint.md, part1-delivery.md and updated README |
| P1 | Propose one next diagnostic tied to remaining errors | Precursor-stratified comparison specified in the checkpoint; not executed |

New tests cover frozen input changes, training-row isolation, unchanged non-objective parameters, reload agreement, metric definitions, group normalization, and honest case selection. The complete Part 1 suite passes 31 tests. Part 2's existing input loader accepts the real exported bundle; service simulation was not run in this delivery.

## Metric definitions and decision rules

Keep per-application MAE/RMSE. For steady/bursty group comparisons, divide each application's error by its training mean before averaging applications equally; do not pool raw errors across scales or call an average of RMSE values pooled RMSE.

Let u=max(actual-predicted,0), o=max(predicted-actual,0), and H={actual>training q95}.

- Primary high-flow shortfall: sum(u on H)/sum(actual on H), with numerator, denominator and sample count.
- High-flow MAE/RMSE and threshold miss rate: count(H and predicted<=training q95)/count(H).
- Strict consecutive shortfalls: u>0. Material consecutive shortfalls: u>=1 invocation/minute. Break runs at nonqualifying minutes or time gaps.
- Overprediction: summed excess, per-minute mean, frequency, excess/actual-volume ratio, and high-flow/non-high-flow excess.
- Paired C−A and C−B differences by application and relative validation day. Store fractions as unit-interval differences; convert to percentage points explicitly when presenting them.
- Zero denominators and absent high-flow samples are missing, not zero-error evidence.

An improvement claim must identify the application and comparator. Lower shortfall with higher excess is a tradeoff; Part 1 does not determine whether that extra prediction is economically acceptable. Keep negative or mixed findings. Do not view test data, change thresholds, or add searches to obtain a favorable result.

The completed experiment found lower bursty high-flow shortfall with more overprediction and worse average errors. Identical symmetric/asymmetric steady predictions do not establish an asymmetry benefit. The ordinary tree remains the reference; all five forecast methods are delivered for the service/cost comparison.

## Case selection and justified extension

The fixed failure remains bursty_02 target minute 14104, window 14044–14164. The planned same-application success rule seeks a high-flow event whose peak is within 20% of the failure magnitude and whose ordinary forecast exceeds training q95 at that peak; prefer minimum event shortfall and earliest event. If none matches, use the nearest ordinary peak hit. If no peak hit exists, label the best event `best_available` rather than successful.

The last condition occurs in the actual baseline: bursty_02 has no ordinary-tree high-flow peak hit. Its fallback is target 13786, window 13726–13846. To meet the checkpoint's request for a successful example, the implementation also selects a matched ordinary peak hit from the other bursty application, explicitly labeled `success_other_workload`: bursty_01 target 15040, window 14980–15100. This supplemental illustration is not a controlled cross-application comparison. All cases are selected using actual traffic and ordinary forecasts before fitting the modified variants.

A second small extension exports validation-only traffic reconstructed from consistent validation labels and observed history. This makes the handoff self-contained without loading real test samples.

## Delivery and acceptance

The separate bundle contains all five forecasts (86,340 rows), original baseline copies (51,804 rows), candidate and control files (17,268 each), 20-row accuracy and diagnostic tables, per-day and group summaries, paired comparisons, strict/material run records, three cases and their available features, plots, saved models, and hashes. Original baseline outputs are not overwritten.

Acceptance means complete and reproducible comparison evidence, not a requirement that the new model win. Input receipts are checked before fitting; ordinary predictions and reloaded new models agree within rtol=atol=1e-10. All 20 application/model groups share the same target keys and matching observed traffic. Test scoring and service simulation remain separate stages.

Parts 2/3 should use the new bundle's forecasts and accuracy metrics together, add both new model identifiers to their selected comparison, preserve timing and initialization, and assess service at comparable cost. Their current runners contain original path/model defaults; the handoff identifies the required selections. Their final test runner must also use matching test accuracy metrics rather than validation accuracy. No downstream code was changed here.

## Next diagnostic, not yet executed

Keep the saved forecasts fixed. Define high-flow events as contiguous minutes above training q95. At each event's first target minute, classify whether the earlier prediction boundary's diff_1 exceeds q95 of positive training diff_1 values. Compare ordinary/candidate shortfall, threshold misses, overprediction and event counts by precursor stratum. Do not use observations after the decision boundary to classify the precursor.

This asks whether the penalty helps when that available signal exists and mainly adds excess where it does not. It does not establish that events without this signal are unpredictable. Use the answer to choose between further penalty-strength investigation, additional available signals, and downstream safeguards. LSTM remains a stretch direction requiring a distinct question.

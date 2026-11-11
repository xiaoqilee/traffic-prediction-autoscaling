# Part 1 Handoff Report for Parts 2 and 3

Date: November 11, 2026. Status: validation forecasting and the underprediction-training ablation are complete. The real held-out test has not been evaluated. This delivery does not include new service simulations.

Part 1 supplies observed minute traffic and five comparable forecast series. **Parts 2 and 3 can consume the CSV files directly; they do not need to train or load any model.** The new candidate reduces the amount of traffic underestimated during high-flow minutes, but increases overprediction and average error on both bursty applications. Keep the ordinary tree as the reference and evaluate whether this tradeoff improves service at comparable server cost.

## What Part 1 completed

1. Aggregated Azure Functions 2019 HTTP invocation counts by application and selected two steady and two bursty applications using training data only.
2. Prepared minute traffic, chronological splits, and 16 causal features: eight lags, three rolling means, two rolling standard deviations, a recent difference, and two target-time calendar features.
3. Produced the original `persistence`, `linear` (Ridge), and `tree` (histogram gradient boosting) forecasts, choosing their settings using validation MAE.
4. Preserved those baseline files and models. Added `tree_q50`, a symmetric quantile-loss control, and `tree_underprediction`, a quantile-loss candidate with a 3:1 penalty for equal-sized underestimation versus overestimation.
5. Completed eight new training fits, validation scoring, daily and workload-group comparisons, high-flow and consecutive-error analysis, three explanatory cases, plots, and reproducibility receipts.
6. Passed 31 Part 1 tests and checked the delivered files with Part 2's existing input loader. Real test samples were not loaded by the ablation, and no real test forecasts or scores were generated.

The [checkpoint findings](part1-checkpoint.md) contain the complete result tables, evidence, design decisions, and limitations. The [implementation plan and completion record](part1-next-improvement-plan.md) document the controlled comparison.

## Files to use

Use `part1/outputs/underprediction_ablation_v1/` as one coherent delivery. Paths below are relative to that directory.

| File | Purpose | Rows |
| --- | --- | ---: |
| [traffic_validation.csv](../outputs/underprediction_ablation_v1/traffic_validation.csv) | Complete observed validation traffic, including warmup | 17,280 |
| [forecasts/validation.csv](../outputs/underprediction_ablation_v1/forecasts/validation.csv) | All five methods on identical targets | 86,340 |
| [forecasts/validation_ordinary.csv](../outputs/underprediction_ablation_v1/forecasts/validation_ordinary.csv) | Unchanged original three baselines | 51,804 |
| [forecasts/validation_modified.csv](../outputs/underprediction_ablation_v1/forecasts/validation_modified.csv) | Underprediction-penalizing candidate only | 17,268 |
| [forecasts/validation_control.csv](../outputs/underprediction_ablation_v1/forecasts/validation_control.csv) | Symmetric control only | 17,268 |
| [metrics/forecast_validation.csv](../outputs/underprediction_ablation_v1/metrics/forecast_validation.csv) | MAE/RMSE for Part 3's accuracy/service join | 20 |
| [metrics/forecast_diagnostics_validation.csv](../outputs/underprediction_ablation_v1/metrics/forecast_diagnostics_validation.csv) | Error magnitude, high-flow misses, overprediction, and consecutive errors | 20 |
| [metrics/forecast_diagnostics_by_day_validation.csv](../outputs/underprediction_ablation_v1/metrics/forecast_diagnostics_by_day_validation.csv) | Complete-period and daily application scores | 80 |
| [metrics/forecast_grouped_validation.csv](../outputs/underprediction_ablation_v1/metrics/forecast_grouped_validation.csv) | Equally weighted, scale-normalized steady/bursty summaries | 40 |
| [metrics/ablation_comparison_validation.csv](../outputs/underprediction_ablation_v1/metrics/ablation_comparison_validation.csv) | Candidate minus ordinary tree/control, complete-period and daily | 32 |
| [metrics/selected_cases_validation.csv](../outputs/underprediction_ablation_v1/metrics/selected_cases_validation.csv) | Fixed case windows, peak observations, and selection rules | 3 |
| [experiment_manifest.json](../outputs/underprediction_ablation_v1/experiment_manifest.json) | Input/output hashes, source hashes, runtime versions, scope and metric definitions | — |

The original `part1/outputs/forecasts/validation.csv` and original models remain available. They contain only the three earlier baselines. The validation traffic in this delivery is recovered from consistent feature labels and observed history, so the ablation does not need to load the full traffic file containing the test period.

## Data and timing contract

Applications: `steady_01`, `steady_02`, `bursty_01`, `bursty_02`. IDs are stable aliases; the mapping to source applications is in [workload_manifest.csv](../data/processed/workload_manifest.csv).

```text
Traffic:  workload_id, minute, request_count, split
Forecast: workload_id, prediction_time, target_minute, model, predicted_count
```

- Counts are invocations per minute. `minute` is a zero-based, dataset-relative minute index, not a timestamp in a known timezone.
- At boundary `t = prediction_time`, the newest complete observed count is `y[t-1]`. A forecast estimates `y[t+3]`, and `target_minute = prediction_time + 3`.
- This means the target is three minutes after the decision boundary and four count positions after the latest observed minute. Do not shift either column to make a plot look aligned.
- Predictions are clipped to zero and remain floating point. No capacity headroom, rounding to servers, smoothing, or queue feedback is included in Part 1 forecasts.
- Each application/method has 4,317 rows. Forecast boundaries are **11520–15836**; target minutes are **11523–15839**, inclusive.
- Replay traffic over **11520–15839**. Use **11520–11522** as common warmup and score arrivals during **11523–15839**. Continue serving traffic and activating pending servers during the final three minutes even though no new forecasts arrive. Use one common queue-draining/timeout convention.
- If rebasing time, subtract the same origin from traffic minutes and both forecast time columns. Service and request times in simulator outputs use seconds.

Train uses relative days 1–8; validation uses days 9–11; held-out test uses days 12–14. Both prediction and target must belong to the assigned split. Prior observed history may cross a split boundary; future observed counts may not enter the forecasting or scaling decision early.

## Models and interpretation

| `model` | Meaning | Recommended role |
| --- | --- | --- |
| `persistence` | Repeat `y[t-1]` for the target minute | Original simple baseline |
| `linear` | Per-application Ridge with training-only standardization | Original learned baseline |
| `tree` | Original squared-error HistGradientBoostingRegressor | Ordinary-training reference |
| `tree_q50` | Same tree configuration, quantile loss at 0.50 | Symmetric-loss control |
| `tree_underprediction` | Same tree configuration, quantile loss at 0.75 | Underprediction-penalizing candidate |

Only the training objective changes for the new tree variants. Features, samples, horizon, seed, thread budget, learning rate, iteration count, leaf constraints, and postprocessing are fixed. The selected ordinary-tree leaf limit is 31 for `bursty_01` and 15 for the other applications. Early stopping remains disabled. New models fit train only; no validation refit or new hyperparameter search was performed.

Compare `tree` with `tree_underprediction` to assess the complete objective change. Compare `tree_q50` with `tree_underprediction` to isolate the penalty direction within the same loss family. A squared-error model targets a different quantity from a median/upper-quantile model, so the first comparison alone does not isolate asymmetry.

### Main validation finding

Here, high flow means `actual > training q95`; high-flow shortfall is the sum of positive forecast errors divided by actual volume in those minutes. Overprediction is summed excess forecast volume, not measured server cost.

| Application | High-flow shortfall: ordinary → candidate | Overpredicted volume: ordinary → candidate | MAE: ordinary → candidate |
| --- | --- | --- | --- |
| bursty_01 | 50.74% → 31.58% | 44,052.6 → 80,552.6 | 19.5710 → 23.8835 |
| bursty_02 | 59.64% → 48.58% | 20,022.8 → 34,384.2 | 8.2607 → 10.2197 |

The high-flow shortfall reduction and overprediction increase occur on all three validation days relative to the ordinary tree. Almost all additional overprediction occurs outside high-flow minutes. On `bursty_02`, all 115 actual high-flow minutes still have candidate predictions at or below the training threshold: magnitude improves, threshold-based spike detection does not.

On both steady applications, the q=0.50 and q=0.75 models produce identical exported predictions. Their gains over the ordinary tree therefore cannot be credited to the asymmetric penalty. See the checkpoint for separate MAE/RMSE and error-run results.

**Decision:** deliver all five methods; retain the ordinary tree as the default reference. Part 1 supports a forecast tradeoff, not a claim of better SLOs or lower cost.

## Part 2: how to run the service comparison

Start with `tree` versus `tree_underprediction` on the same application and traffic. Keep server capacity, service time, initial state, startup delay, timeout, minimum servers, arrival samples, random seed, scoring window, and base scaling policy identical. Then include the symmetric control and original baselines. Add capacity headroom and queue feedback in separate comparisons.

The current Part 2 loader accepts the new model names without a schema change. The following example uses the existing simulator API; run it from the repository root with the shared environment when starting Part 2's experiments. It was not executed as part of this Part 1 delivery.

```python
import json
from pathlib import Path
import pandas as pd
from part2.simulator import load_replay_inputs, simulate_replay

bundle = Path("part1/outputs/underprediction_ablation_v1")
traffic, forecasts = load_replay_inputs(
    bundle / "traffic_validation.csv", bundle / "forecasts/validation.csv"
)
settings = json.loads(Path("part2/config.json").read_text())
models = ["persistence", "linear", "tree", "tree_q50", "tree_underprediction"]
logs, outcomes, summaries = [], [], []
for workload_id in sorted(traffic.workload_id.unique()):
    for model in models:
        run_id = f"ablation_v1_{workload_id}_{model}"
        result = simulate_replay(
            traffic=traffic, forecasts=forecasts, config=settings,
            policy="forecast", workload_id=workload_id, model=model, run_id=run_id,
        )
        logs.append(result["minute_logs"])
        outcomes.append(result["request_outcomes"])
        summaries.append(result["summary"])

output = Path("part2/outputs/underprediction_ablation_v1")
output.mkdir(parents=True, exist_ok=True)
pd.concat(logs, ignore_index=True).to_csv(output / "minute_logs.csv", index=False)
pd.concat(outcomes, ignore_index=True).to_csv(output / "request_outcomes.csv", index=False)
pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
(output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
```

The base scaling target remains `ceil(predicted_count / capacity_per_server_per_minute)`, subject to Part 2's existing minimum, startup accounting, and scaling rules. Request capacity at `prediction_time`, not at `target_minute`. Future actual traffic is replay input, not policy input.

Supply Part 3 with minute logs, request outcomes, exact settings, forecast-file hash, scoring interval, cost convention, and a unique experiment/run ID. Record SLO, headroom or queue-feedback variants explicitly. Uniform arrivals and service behavior are simulator assumptions; invocation counts are observed data.

`part2/run_example.py` remains an earlier small-example runner listing only three methods. Running it unchanged does not exercise this delivery. The earlier two-application [example handoff](../outputs/examples/handoff.json) is still useful for basic integration, but it is not the new ablation.

## Part 3: how to combine accuracy, service and cost

Use the forecast metrics **from this bundle**, not the earlier three-method metric file. Join forecasts to traffic on `(workload_id, target_minute) = (workload_id, minute)`; join service results to accuracy on `(workload_id, model)` and keep the experiment identity with every run.

After producing the Part 2 files above, the existing comparison API can read the new metric table:

```python
from pathlib import Path
from part3.src.compare import build_comparison_table

bundle = Path("part1/outputs/underprediction_ablation_v1")
simulation = Path("part2/outputs/underprediction_ablation_v1")
comparison = build_comparison_table(
    forecast_metrics_path=bundle / "metrics/forecast_validation.csv",
    outcomes_path=simulation / "request_outcomes.csv",
    minute_logs_path=simulation / "minute_logs.csv",
    scoring_start_minute=11523, scoring_end_minute=15839, slo_seconds=5,
)
comparison.to_csv(simulation / "accuracy_service_comparison.csv", index=False)
```

Here 5 seconds matches the current simulator configuration; if changing the SLO, use the same declared value in all methods and in scoring. Report MAE/RMSE, completed-request p95/p99 latency, SLO violation rate, drop rate, and server-minutes. Keep latency for completed requests separate from drops. Use the same arrival denominator for SLO comparisons and document whether drops count as violations.

Compare candidate forecasts against ordinary forecasts at similar server cost, using validation to select headroom settings. A reduction in forecast shortfall alone does not establish a service improvement. Keep the synthetic error-timing experiment labeled diagnostic and distinct from these trained models.

Current integration points that need explicit selection by the Part 2/3 owners:

- `part3/configs/part3.json` lists only the original three forecast models. Include the two new identifiers for the expanded comparison.
- `part3/src/run_validation.py`, `headroom.py`, `plot_timeline.py`, and `compare.py` default to original forecast or metric paths. Point them at this bundle or call their APIs with the new inputs; changing the model list alone is insufficient.
- The headroom plot enumerates three original models, and timeline colors/run-ID assumptions also need to reflect the chosen experiment. The example above uses an `ablation_v1_` run-ID prefix, so timeline lookups must use those IDs.
- `part3/src/run_test.py` currently generates the original three-model test forecasts and joins validation accuracy metrics. Before final testing, switch both its generator/input paths and its accuracy metrics to the matching five-model test artifacts. Do not mix validation accuracy with test service results.

These downstream scripts were inspected for the handoff; their code and configuration were not changed in this Part 1 task.

## Diagnostics to use in failure analysis

The [case figure](../outputs/underprediction_ablation_v1/figures/failure_success_comparison.png) shows:

- `bursty_02`, minute **14104**, window **14044–14164**: actual 143 versus ordinary 21.40 and candidate 28.76. The latest complete observation at the decision boundary was 19, supporting a missing-local-precursor hypothesis.
- `bursty_02`, minute **13786**, window **13726–13846**: actual 57 versus ordinary 52.73. This is the best available same-application event, **not** a successful threshold-crossing forecast.
- `bursty_01`, minute **15040**, window **14980–15100**: actual 144 versus ordinary 141.28 and candidate 145.13. This is an explicitly labeled successful example from another bursty application; it illustrates a case with stronger available local change, not a controlled cross-application causal comparison.

All cases are selected from observed validation traffic and ordinary forecasts before fitting the modified models. Their rules are saved in [design_manifest.json](../outputs/underprediction_ablation_v1/design_manifest.json). The extra successful example was added because no ordinary-tree high-flow peak hit exists in `bursty_02`; the same-application fallback remains visible.

For timelines, use [case_features_validation.csv](../outputs/underprediction_ablation_v1/metrics/case_features_validation.csv), [case_comparison_validation.csv](../outputs/underprediction_ablation_v1/metrics/case_comparison_validation.csv), [underprediction_runs_validation.csv](../outputs/underprediction_ablation_v1/metrics/underprediction_runs_validation.csv), and [material_underprediction_runs_validation.csv](../outputs/underprediction_ablation_v1/metrics/material_underprediction_runs_validation.csv). Strict runs use `actual > predicted`; material runs require a shortfall of at least one invocation per minute. They end at equality, a nonqualifying minute, or a time gap.

## Reproduce and final-test procedure

To regenerate this validation delivery from existing prepared data and frozen ordinary models:

```sh
cd part1
../.venv/bin/python -B src/ablation.py
../.venv/bin/python -B -m unittest discover -s tests -v
```

The default configuration is [part1-underprediction.json](../configs/part1-underprediction.json). Paths are relative to `part1/`. The runner checks frozen input hashes and ordinary prediction agreement, fits only the two new variants, verifies saved-model predictions, and writes the separate experiment directory. Baseline files remain unchanged. An input mismatch stops the run before fitting and should be investigated rather than bypassed.

Only after all owners fix the final comparison groups, model/policy settings and evaluation rules, generate the held-out forecasts without retraining:

```sh
# Run from part1/ only when the final experiment design is fixed.
../.venv/bin/python -B src/ablation.py --split test
```

This writes `outputs/underprediction_ablation_v1/forecasts/test.csv`, `metrics/forecast_test.csv`, and `metrics/test_run.json` from all five saved methods. Use test traffic from the original `data/processed/traffic.csv`, filtered to `split=test`; the delivered `traffic_validation.csv` contains no test traffic. Replay **15840–20159**, score **15843–20159**, and join to the new **test** accuracy metrics. This final test command has not been run for the real dataset.

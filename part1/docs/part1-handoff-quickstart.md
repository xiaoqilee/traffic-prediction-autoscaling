# Part 1: Quick Handoff for Parts 2 and 3

## What is ready

Part 1 prepared four Azure HTTP workloads (`steady_01`, `steady_02`, `bursty_01`, `bursty_02`), causal forecasting features, and chronological train/validation/test splits. It now delivers five methods:

| Model identifier | Meaning |
| --- | --- |
| `persistence` | Repeat the latest complete observed count |
| `linear` | Original Ridge regression |
| `tree` | Original squared-error tree; keep as the reference |
| `tree_q50` | Same tree configuration with symmetric quantile loss |
| `tree_underprediction` | Same configuration with a 3:1 underprediction/overprediction penalty |

The two new variants use the same training data, features and non-objective settings as the ordinary tree. Original baselines are preserved. Validation experiments, diagnostics, and 31 Part 1 tests are complete; the exported files pass Part 2's input-loader checks.

The candidate reduces high-flow shortfall from **50.74% to 31.58%** on `bursty_01` and **59.64% to 48.58%** on `bursty_02`, but increases overprediction and average error. Part 2/3 should determine whether this tradeoff improves service at comparable server cost. The real held-out test remains unevaluated.

## Inputs and shared timing

All paths are relative to the repository root. Use this bundle consistently:

```text
part1/outputs/underprediction_ablation_v1/
```

| File within the bundle | Consumer | Contents |
| --- | --- | --- |
| `traffic_validation.csv` | Part 2 | Observed validation traffic and warmup; 17,280 rows |
| `forecasts/validation.csv` | Parts 2/3 | All five methods; 86,340 rows |
| `metrics/forecast_validation.csv` | Part 3 | Matching MAE/RMSE; 20 workload/model rows |

```text
Traffic:  workload_id, minute, request_count, split
Forecast: workload_id, prediction_time, target_minute, model, predicted_count
```

- Counts are invocations per minute; minute indexes are dataset-relative.
- At `prediction_time=t`, observations are available through `t-1`; the forecast targets `t+3`. Schedule capacity at **prediction_time** and compare predictions with actual traffic at **target_minute**.
- Replay **11520–15839**; warm up on **11520–11522**; score arrivals on **11523–15839**, all inclusive. Each workload/model has 4,317 scored target minutes.
- Keep serving during the final three minutes, when no new forecasts arrive. Use the same end-of-run draining/timeout convention across methods.
- Predictions are nonnegative floats without headroom. If rebasing minutes, subtract the same origin from all three time columns.

## Part 2: load forecasts and run the simulator

From the repository root, use the shared `.venv` environment and the existing API:

```python
import json
from pathlib import Path
from part2.simulator import load_replay_inputs, simulate_replay

bundle = Path("part1/outputs/underprediction_ablation_v1")
traffic, forecasts = load_replay_inputs(
    bundle / "traffic_validation.csv", bundle / "forecasts/validation.csv"
)
settings = json.loads(Path("part2/config.json").read_text())
result = simulate_replay(
    traffic=traffic, forecasts=forecasts, config=settings, policy="forecast",
    workload_id="bursty_01", model="tree_underprediction",
    run_id="ablation_v1_bursty_01_tree_underprediction",
)
```

Repeat for all four workloads and the five model identifiers above. First compare `tree` with `tree_underprediction`; include `tree_q50` to distinguish the asymmetric penalty from changing the loss family. Keep capacity, startup delay, initial servers, arrival samples, seed, timeout and scoring identical. Evaluate added headroom and queue feedback separately.

Concatenate each run's `result["minute_logs"]` and `result["request_outcomes"]` into `minute_logs.csv` and `request_outcomes.csv`. Also save `result["summary"]`, settings, the forecast-file hash, scoring bounds and cost convention. Give each run a unique ID, and pass these files to Part 3.

## Part 3: join accuracy with service and cost

Read **this bundle's** `metrics/forecast_validation.csv`. Join accuracy to simulation results on `(workload_id, model)` while retaining experiment/run IDs. To recompute forecast errors, join `(workload_id, target_minute)` to traffic `(workload_id, minute)`.

Use the existing `part3.src.compare.build_comparison_table` API with:

- `forecast_metrics_path`: the bundle's `metrics/forecast_validation.csv`;
- `outcomes_path` / `minute_logs_path`: Part 2's new consolidated files;
- `scoring_start_minute=11523`, `scoring_end_minute=15839`;
- `slo_seconds`: the same SLO used in the simulation (currently 5 seconds).

Report accuracy, completed-request latency, SLO violations, drops and server-minutes. Compare ordinary and modified forecasts at similar cost, selecting settings on validation. Additional error metrics and case windows are available in `metrics/forecast_diagnostics_validation.csv` and `metrics/selected_cases_validation.csv` within the bundle.

**Existing runners need explicit input selection:** Part 2's example runner and Part 3's configuration still list three methods. Add `tree_q50` and `tree_underprediction`; update Part 3's validation, comparison, headroom and timeline inputs to this bundle. Update plot model lists and run-ID lookups too. Changing only the model list leaves the old file paths in use.

These downstream edits and service runs remain with Parts 2/3. Once the final design is fixed, Part 1 can export all five saved models on test using `src/ablation.py --split test` from `part1/`. Part 3's current test runner must use that export and its matching **test** accuracy metrics; it currently defaults to the original generator and validation metrics.

# Part 1 Handoff: Part 2 and Part 3

Part 1 provides **observed traffic for four applications and forecasts from three methods**: `persistence`, `linear` (Ridge), and `tree` (GBDT). Consume the CSV files directly; no model loading or retraining is required.

## Part 2: Turn Forecasts into Service Outcomes

### 1. Start with the Small Example

| File | Use |
| --- | --- |
| [outputs/examples/traffic.csv](../outputs/examples/traffic.csv) | Observed request count for each minute |
| [outputs/examples/forecasts.csv](../outputs/examples/forecasts.csv) | Start with `model=persistence`, then replay all three methods |
| [outputs/examples/handoff.json](../outputs/examples/handoff.json) | Per-application replay, warmup, and scoring bounds; endpoints are inclusive |

The interfaces are:

```text
Traffic:  workload_id, minute, request_count, split
Forecast: workload_id, prediction_time, target_minute, model, predicted_count
```

Run each `workload_id × model` independently. `request_count` is the observed count in that minute; `predicted_count` estimates the count in the target minute.

### 2. Apply Forecasts at the Right Time

Read a forecast at boundary `t = prediction_time`. It targets `target_minute = t + 3`:

```text
Read forecast and request instances at t -> wait 3 minutes -> capacity becomes available at t+3
```

The basic rule is `target_instances = ceil(predicted_count / per_instance_capacity_per_minute)`. Part 2 sets capacity, service time, timeout, and initial instance count, using identical settings across methods. Start with evenly spread arrivals, a FIFO queue, and fixed service time; add a reactive baseline using past demand or the current queue.

Schedule scaling at `prediction_time`; compare forecasts with observations at `target_minute`. Account for instances already starting. Future observed counts belong to the replay environment and must not be exposed to the scaling policy early.

### 3. Switch to Full Validation Data

- Traffic: [data/processed/traffic.csv](../data/processed/traffic.csv), filtered to `split=validation`.
- Forecasts: [outputs/forecasts/validation.csv](../outputs/forecasts/validation.csv), filtered by application and method.
- Replay **11520–15839**. Use the first **3 minutes for warmup**; score target minutes **11523–15839**.
- During the final three minutes, continue serving traffic and activating pending instances without inventing new forecasts. Apply a common queue-draining or timeout rule at the end.

Files retain the original relative minute indexes. If rebasing to zero, subtract the same origin from traffic minutes and both forecast time columns.

### 4. Deliver These Outputs to Part 3

Assign a `run_id` to each run and produce:

| Output | Required contents |
| --- | --- |
| Per-minute logs | run_id, application, method, minute, arrivals, active/starting instances, queue length, completed/dropped counts |
| Request outcomes | run_id, request ID, arrival/completion time, response time, completed/dropped status; time in seconds |
| Run settings | Capacity, startup delay, service time, timeout, SLO, headroom, initial state, random seed, scoring interval, and cost convention |

## Part 3: Compare Accuracy, Service Quality, and Cost

### 1. Read These Inputs

| Input | Use |
| --- | --- |
| [forecast_validation.csv](../outputs/metrics/forecast_validation.csv) | Existing MAE/RMSE by application and method |
| The three Part 2 outputs above | Calculate latency, violations, drops, and server cost |
| [forecast_diagnostics_validation.csv](../outputs/metrics/forecast_diagnostics_validation.csv) | Inspect high-flow underprediction and consecutive errors |

To recompute forecast errors, join forecast **`(workload_id, target_minute)`** to traffic **`(workload_id, minute)`**. Join simulation outcomes to their settings using `run_id`.

### 2. Complete the Comparison

1. Produce a table by application and method: **MAE, RMSE, p95/p99 latency, SLO violation rate, drop rate, and server-minutes**.
2. Evaluate requests arriving within the scoring interval. Calculate latency percentiles for completed requests and report drops separately. A suggested SLO violation numerator includes completed requests exceeding the SLO plus dropped requests; the denominator is all scored arrivals.
3. Sweep capacity headroom on validation data and plot **cost versus SLO violation rate**, comparing service quality at similar cost.
4. Plot **actual traffic, forecasts, active instances, and queue length** on one timeline to explain ranking agreement or disagreement. Start with the [bursty_02 diagnostic case](../outputs/figures/underprediction_case.png).
5. Run a small controlled experiment: construct forecasts with similar MAE/RMSE but different underprediction timing, then vary startup delay alone. Label constructed forecasts as diagnostic methods.

Discuss steady and bursty applications separately. Do not combine raw MAE across different traffic scales into one ranking. Use identical random arrival/service samples across methods.

## Final Test

Use validation data for integration and tuning. After fixing the full experiment design, run:

```sh
# From part1/
../.venv/bin/python -B src/forecast.py --split test
```

Pass test traffic from `traffic.csv` and the resulting `outputs/forecasts/test.csv` to Part 2. Replay **15840–20159** and score **15843–20159**, using the same evaluation rules.

Real simulator integration and final test evaluation have not yet been run. Counts are observed; arrival times, service latency, and SLO outcomes are simulated.

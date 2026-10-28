# Part 3: Comparison and Analysis

This module connects Part 1 forecast accuracy with Part 2 simulation outcomes to answer whether better traffic forecasts improve autoscaling outcomes at comparable server cost.

## Structure

```
part3/
├── configs/part3.json      # Scoring interval, SLO, headroom sweep parameters
├── src/
│   ├── run_validation.py   # Step 1: Run Part 2 on full validation data (16 runs)
│   ├── metrics.py          # Step 2: Per-run service quality metrics
│   ├── compare.py          # Step 3: Main comparison table (MAE + latency + cost)
│   ├── headroom.py         # Step 4: Headroom sweep — cost vs SLO violation rate
│   ├── plot_timeline.py    # Step 5: Timeline plot for bursty_02
│   └── controlled.py       # Step 6: Controlled experiment with synthetic forecasts
├── outputs/
│   ├── validation/         # Full validation simulation outputs
│   ├── metrics/            # Comparison tables and sweep results
│   └── figures/            # Plots
└── tests/
    └── test_metrics.py     # Unit tests using Part 2 example outputs
```

## Scoring Interval

Validation: minutes 11520–15839. Warmup: 11520–11522. Scored: **11523–15839**.
Test: minutes 15840–20159. Warmup: 15840–15842. Scored: **15843–20159**.

## SLO

5 seconds. Violation = completed request exceeding 5s **or** dropped request.
Rate = violations / all arrivals in scoring interval.

## Running

From the repository root with the shared `.venv` environment:

```sh
source .venv/bin/activate

# Step 1 — Full validation run (~16 simulator calls, a few minutes)
python -m part3.src.run_validation

# Step 2+3 — Comparison table
python -m part3.src.compare

# Step 4 — Headroom sweep (96 simulator calls, ~10 minutes)
python -m part3.src.headroom

# Step 5 — Timeline plot (requires Step 1 output)
python -m part3.src.plot_timeline

# Step 6 — Controlled experiment
python -m part3.src.controlled

# Tests (run before Step 1 — uses Part 2 example fixtures)
python -m unittest discover -s part3/tests -p "test_*.py" -v
```

## Outputs

| File | Description |
| --- | --- |
| `outputs/validation/summary.csv` | 16 rows — one per workload × policy run |
| `outputs/metrics/comparison_table.csv` | MAE, RMSE, p50/p95/p99 latency, SLO rate, drop rate, server-minutes |
| `outputs/metrics/headroom_sweep.csv` | Cost vs SLO rate at each headroom factor |
| `outputs/metrics/synthetic_forecasts.csv` | synth_early and synth_late forecast series |
| `outputs/metrics/controlled_experiment.csv` | SLO outcomes across startup delays |
| `outputs/figures/timeline_bursty_02.png` | Four-panel timeline for the diagnostic case |
| `outputs/figures/headroom_sweep.png` | Cost vs SLO trade-off plot per model |
| `outputs/figures/controlled_experiment.png` | Underprediction timing vs startup delay |

## Key Findings (Validation)

`bursty_01` shows the clearest separation: `linear` drops 33k requests at 4525 server-minutes vs `persistence` at 34.8k drops and 4966 server-minutes — better forecast, lower cost, better service. Steady workloads show no difference across methods (demand is too predictable for forecast quality to matter).

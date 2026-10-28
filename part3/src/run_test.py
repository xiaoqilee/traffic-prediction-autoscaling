"""
Final test evaluation for Part 3.

Generates test forecasts via Part 1 (if not already present), then replays all
four workloads over the held-out test split using the same evaluation rules as
the validation run.

Run from the repository root:
    python -m part3.src.run_test

Pass --force to regenerate test forecasts even if the file already exists.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from part2.simulator import load_replay_inputs, simulate_replay
from part3.src.metrics import summarize_run

REPO_ROOT = Path(__file__).parents[2]
PART1_DIR = REPO_ROOT / "part1"
PART2_CONFIG = REPO_ROOT / "part2" / "config.json"
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"

TRAFFIC_PATH = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"
TEST_FORECAST_PATH = REPO_ROOT / "part1" / "outputs" / "forecasts" / "test.csv"
FORECAST_METRICS_PATH = REPO_ROOT / "part1" / "outputs" / "metrics" / "forecast_validation.csv"

OUTPUT_DIR = REPO_ROOT / "part3" / "outputs" / "test"


def generate_test_forecasts(force=False):
    """Run Part 1 forecast.py --split test to produce test forecasts."""
    if TEST_FORECAST_PATH.exists() and not force:
        print(f"Test forecasts already exist at {TEST_FORECAST_PATH} (pass --force to regenerate)")
        return

    print("Generating test forecasts via Part 1 ...")
    result = subprocess.run(
        [sys.executable, "-B", "src/forecast.py", "--split", "test"],
        cwd=PART1_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Part 1 forecast.py --split test failed")

    print(f"  done — {TEST_FORECAST_PATH}")


def run_test_simulation(sim_config, part3_cfg):
    """Replay all workloads over the test split and return raw results."""
    all_traffic, all_forecasts = load_replay_inputs(TRAFFIC_PATH, TEST_FORECAST_PATH)

    traffic = all_traffic.loc[all_traffic["split"].eq("test")].copy()
    workloads = part3_cfg["workloads"]
    forecast_models = part3_cfg["forecast_models"]

    all_logs = []
    all_outcomes = []
    all_summaries = []

    for workload_id in workloads:
        for model in forecast_models:
            run_id = f"{workload_id}_{model}"
            print(f"  running {run_id} ...")

            result = simulate_replay(
                traffic=traffic,
                forecasts=all_forecasts,
                config=sim_config,
                policy="forecast",
                workload_id=workload_id,
                model=model,
                run_id=run_id,
            )

            all_logs.append(result["minute_logs"])
            all_outcomes.append(result["request_outcomes"])
            all_summaries.append(result["summary"])

        run_id = f"{workload_id}_reactive"
        print(f"  running {run_id} ...")

        result = simulate_replay(
            traffic=traffic,
            forecasts=all_forecasts,
            config=sim_config,
            policy="reactive",
            workload_id=workload_id,
            run_id=run_id,
        )

        all_logs.append(result["minute_logs"])
        all_outcomes.append(result["request_outcomes"])
        all_summaries.append(result["summary"])

    return (
        pd.concat(all_logs, ignore_index=True),
        pd.concat(all_outcomes, ignore_index=True),
        pd.DataFrame(all_summaries),
    )


def build_test_comparison(outcomes_all, logs_all, part3_cfg):
    """Join per-run service metrics with Part 1 validation forecast accuracy."""
    scoring_start = part3_cfg["test_scoring_start_minute"]
    scoring_end = part3_cfg["test_scoring_end_minute"]
    slo_seconds = part3_cfg["slo_seconds"]

    forecast_metrics = pd.read_csv(FORECAST_METRICS_PATH)[
        ["workload_id", "model", "pattern", "mae", "rmse"]
    ]

    rows = []
    for run_id, outcomes in outcomes_all.groupby("run_id"):
        workload_id = outcomes["workload_id"].iloc[0]
        model = outcomes["model"].iloc[0]
        logs = logs_all.loc[logs_all["run_id"].eq(run_id)]

        service = summarize_run(outcomes, logs, scoring_start, scoring_end, slo_seconds)
        rows.append({"run_id": run_id, "workload_id": workload_id, "model": model, **service})

    service_frame = pd.DataFrame(rows)
    comparison = service_frame.merge(forecast_metrics, on=["workload_id", "model"], how="left")

    col_order = [
        "run_id", "workload_id", "model", "pattern", "mae", "rmse",
        "arrivals", "completed", "dropped", "drop_rate",
        "slo_violation_rate", "p50_latency", "p95_latency", "p99_latency",
        "server_minutes",
    ]
    return comparison[col_order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate test forecasts even if present")
    args = parser.parse_args()

    sim_config = json.loads(PART2_CONFIG.read_text())
    part3_cfg = json.loads(PART3_CONFIG.read_text())

    generate_test_forecasts(force=args.force)

    print("\nRunning test simulation ...")
    logs, outcomes, summary = run_test_simulation(sim_config, part3_cfg)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logs.to_csv(OUTPUT_DIR / "minute_logs.csv", index=False)
    outcomes.to_csv(OUTPUT_DIR / "request_outcomes.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\nBuilding test comparison table ...")
    comparison = build_test_comparison(outcomes, logs, part3_cfg)
    comparison.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    print("\n=== Test Results ===")
    print(comparison[[
        "workload_id", "model", "mae", "arrivals", "dropped",
        "drop_rate", "slo_violation_rate", "p95_latency", "server_minutes",
    ]].to_string(index=False))
    print(f"\nSaved all outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

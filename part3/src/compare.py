"""Join Part 1 forecast accuracy with Part 2 service metrics."""

import json
from pathlib import Path

import pandas as pd

from part3.src.metrics import summarize_run

REPO_ROOT = Path(__file__).parents[2]
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"

FORECAST_METRICS_PATH = (
    REPO_ROOT / "part1" / "outputs" / "metrics" / "forecast_validation.csv"
)
VALIDATION_DIR = REPO_ROOT / "part3" / "outputs" / "validation"
OUTPUT_PATH = REPO_ROOT / "part3" / "outputs" / "metrics" / "comparison_table.csv"


def build_comparison_table(
    forecast_metrics_path,
    outcomes_path,
    minute_logs_path,
    scoring_start_minute,
    scoring_end_minute,
    slo_seconds,
):
    forecast_metrics = pd.read_csv(forecast_metrics_path)[
        ["workload_id", "model", "pattern", "mae", "rmse"]
    ]

    outcomes_all = pd.read_csv(outcomes_path)
    logs_all = pd.read_csv(minute_logs_path)

    rows = []
    for run_id, outcomes in outcomes_all.groupby("run_id"):
        workload_id = outcomes["workload_id"].iloc[0]
        model = outcomes["model"].iloc[0]
        logs = logs_all.loc[logs_all["run_id"].eq(run_id)]

        service = summarize_run(
            outcomes,
            logs,
            scoring_start_minute,
            scoring_end_minute,
            slo_seconds,
        )

        rows.append(
            {
                "run_id": run_id,
                "workload_id": workload_id,
                "model": model,
                **service,
            }
        )

    service_frame = pd.DataFrame(rows)

    comparison = service_frame.merge(
        forecast_metrics,
        on=["workload_id", "model"],
        how="left",
    )

    col_order = [
        "run_id",
        "workload_id",
        "model",
        "pattern",
        "mae",
        "rmse",
        "arrivals",
        "completed",
        "dropped",
        "drop_rate",
        "slo_violation_rate",
        "p50_latency",
        "p95_latency",
        "p99_latency",
        "server_minutes",
    ]
    comparison = comparison[col_order]

    return comparison


def main():
    cfg = json.loads(PART3_CONFIG.read_text())

    comparison = build_comparison_table(
        forecast_metrics_path=FORECAST_METRICS_PATH,
        outcomes_path=VALIDATION_DIR / "request_outcomes.csv",
        minute_logs_path=VALIDATION_DIR / "minute_logs.csv",
        scoring_start_minute=cfg["scoring_start_minute"],
        scoring_end_minute=cfg["scoring_end_minute"],
        slo_seconds=cfg["slo_seconds"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Run Part 2 on the full validation split for all workloads and policies."""

import json
from pathlib import Path

import pandas as pd

from part2.simulator import load_replay_inputs, simulate_replay

REPO_ROOT = Path(__file__).parents[2]
PART2_CONFIG = REPO_ROOT / "part2" / "config.json"
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"

TRAFFIC_PATH = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"
FORECAST_PATH = REPO_ROOT / "part1" / "outputs" / "forecasts" / "validation.csv"
OUTPUT_DIR = REPO_ROOT / "part3" / "outputs" / "validation"


def main():
    sim_config = json.loads(PART2_CONFIG.read_text())
    part3_cfg = json.loads(PART3_CONFIG.read_text())

    workloads = part3_cfg["workloads"]
    forecast_models = part3_cfg["forecast_models"]

    all_traffic, all_forecasts = load_replay_inputs(TRAFFIC_PATH, FORECAST_PATH)

    traffic = all_traffic.loc[all_traffic["split"].eq("validation")].copy()
    forecasts = all_forecasts.copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_logs = []
    all_outcomes = []
    all_summaries = []

    for workload_id in workloads:
        for model in forecast_models:
            run_id = f"{workload_id}_{model}"
            print(f"  running {run_id} ...")

            result = simulate_replay(
                traffic=traffic,
                forecasts=forecasts,
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
            forecasts=forecasts,
            config=sim_config,
            policy="reactive",
            workload_id=workload_id,
            run_id=run_id,
        )

        all_logs.append(result["minute_logs"])
        all_outcomes.append(result["request_outcomes"])
        all_summaries.append(result["summary"])

    pd.concat(all_logs, ignore_index=True).to_csv(
        OUTPUT_DIR / "minute_logs.csv", index=False
    )
    pd.concat(all_outcomes, ignore_index=True).to_csv(
        OUTPUT_DIR / "request_outcomes.csv", index=False
    )
    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print(f"\nWrote {len(all_summaries)} runs to {OUTPUT_DIR}")
    print(summary_frame[["run_id", "arrivals", "completed", "dropped", "server_minutes"]].to_string(index=False))


if __name__ == "__main__":
    main()

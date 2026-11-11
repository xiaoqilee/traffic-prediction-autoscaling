"""Compare scaling policies on the Part 1 validation handoff."""

import json
from pathlib import Path

import pandas as pd

from part2.simulator import (
    load_replay_inputs,
    simulate_replay,
)


def main():
    repository_root = Path(__file__).parents[1]

    config_path = repository_root / "part2" / "config.json"
    config = json.loads(config_path.read_text())

    validation_bundle = (
        repository_root
        / "part1"
        / "outputs"
        / "underprediction_ablation_v1"
    )

    traffic, forecasts = load_replay_inputs(
        validation_bundle / "traffic_validation.csv",
        validation_bundle / "forecasts" / "validation.csv",
    )
    
    output_dir = (
        repository_root
        / "part2"
        / "outputs"
        / "ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_logs = []
    all_outcomes = []
    all_summaries = []

    for workload_id in ["steady_01", "steady_02", "bursty_01", "bursty_02"]:
        for policy, policy_kwargs in [
            ("forecast", {}),
            (
                "forecast_headroom",
                {"headroom_factor": 1.5},
            ),
            (
                "forecast_queue_feedback",
                {"queue_feedback_weight": 1.0},
            ),
        ]:
            run_id = f"{workload_id}_{policy}"

            result = simulate_replay(
                traffic=traffic,
                forecasts=forecasts,
                config=config,
                policy=policy,
                workload_id=workload_id,
                model="tree",
                run_id=run_id,
                **policy_kwargs,
            )

            all_logs.append(
                result["minute_logs"].assign(policy=policy)
            )
            all_outcomes.append(
                result["request_outcomes"].assign(policy=policy)
            )

            summary = dict(result["summary"])
            summary["policy"] = policy
            all_summaries.append(summary)

    pd.concat(
        all_logs,
        ignore_index=True,
    ).to_csv(
        output_dir / "minute_logs.csv",
        index=False,
    )

    pd.concat(
        all_outcomes,
        ignore_index=True,
    ).to_csv(
        output_dir / "request_outcomes.csv",
        index=False,
    )

    pd.DataFrame(all_summaries).to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print(pd.DataFrame(all_summaries).to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")

if __name__ == "__main__":
    main()
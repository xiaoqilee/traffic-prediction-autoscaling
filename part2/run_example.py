"""Run Part 2 on the small Part 1 handoff examples."""

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

    examples = (
        repository_root
        / "part1"
        / "outputs"
        / "examples"
    )

    traffic, forecasts = load_replay_inputs(
        examples / "traffic.csv",
        examples / "forecasts.csv",
    )

    output_dir = (
        repository_root
        / "part2"
        / "outputs"
        / "examples"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_logs = []
    all_outcomes = []
    all_summaries = []

    for workload_id in ["steady_01", "bursty_02"]:
        for model in ["persistence", "linear", "tree"]:
            run_id = f"{workload_id}_{model}"

            result = simulate_replay(
                traffic=traffic,
                forecasts=forecasts,
                config=config,
                policy="forecast",
                workload_id=workload_id,
                model=model,
                run_id=run_id,
            )

            all_logs.append(result["minute_logs"])
            all_outcomes.append(result["request_outcomes"])
            all_summaries.append(result["summary"])

        run_id = f"{workload_id}_reactive"

        result = simulate_replay(
            traffic=traffic,
            forecasts=forecasts,
            config=config,
            policy="reactive",
            workload_id=workload_id,
            run_id=run_id,
        )

        all_logs.append(result["minute_logs"])
        all_outcomes.append(result["request_outcomes"])
        all_summaries.append(result["summary"])

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

    print(
        pd.DataFrame(all_summaries).to_string(
            index=False
        )
    )
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
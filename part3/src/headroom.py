"""Headroom sweep: vary forecast buffer to plot cost vs SLO violation rate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from part2.simulator import load_replay_inputs, simulate_replay
from part3.src.metrics import summarize_run

REPO_ROOT = Path(__file__).parents[2]
PART2_CONFIG = REPO_ROOT / "part2" / "config.json"
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"
TRAFFIC_PATH = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"
FORECAST_PATH = REPO_ROOT / "part1" / "outputs" / "forecasts" / "validation.csv"
SWEEP_OUTPUT = REPO_ROOT / "part3" / "outputs" / "metrics" / "headroom_sweep.csv"
FIGURE_OUTPUT = REPO_ROOT / "part3" / "outputs" / "figures" / "headroom_sweep.png"


def run_sweep(
    traffic,
    forecasts,
    sim_config,
    workloads,
    forecast_models,
    headroom_factors,
    scoring_start_minute,
    scoring_end_minute,
    slo_seconds,
):
    """Run all workload × model × headroom combinations and return metrics rows."""
    rows = []

    for headroom in headroom_factors:
        print(f"  headroom={headroom:.2f}")
        scaled_forecasts = forecasts.copy()
        scaled_forecasts["predicted_count"] = (
            scaled_forecasts["predicted_count"] * headroom
        )

        for workload_id in workloads:
            for model in forecast_models:
                run_id = f"{workload_id}_{model}_h{headroom:.2f}"
                result = simulate_replay(
                    traffic=traffic,
                    forecasts=scaled_forecasts,
                    config=sim_config,
                    policy="forecast",
                    workload_id=workload_id,
                    model=model,
                    run_id=run_id,
                )
                service = summarize_run(
                    result["request_outcomes"],
                    result["minute_logs"],
                    scoring_start_minute,
                    scoring_end_minute,
                    slo_seconds,
                )
                rows.append(
                    {
                        "headroom": headroom,
                        "workload_id": workload_id,
                        "model": model,
                        **service,
                    }
                )

        # Reactive baseline at headroom=1.0 only (not affected by forecast scaling)
        if abs(headroom - 1.0) < 1e-9:
            for workload_id in workloads:
                run_id = f"{workload_id}_reactive_h1.00"
                result = simulate_replay(
                    traffic=traffic,
                    forecasts=forecasts,
                    config=sim_config,
                    policy="reactive",
                    workload_id=workload_id,
                    run_id=run_id,
                )
                service = summarize_run(
                    result["request_outcomes"],
                    result["minute_logs"],
                    scoring_start_minute,
                    scoring_end_minute,
                    slo_seconds,
                )
                rows.append(
                    {
                        "headroom": headroom,
                        "workload_id": workload_id,
                        "model": "reactive",
                        **service,
                    }
                )

    return pd.DataFrame(rows)


def plot_sweep(sweep_df, output_path):
    """Plot cost (server_minutes) vs SLO violation rate, one line per model."""
    patterns = {"steady_01": "steady", "steady_02": "steady",
                "bursty_01": "bursty", "bursty_02": "bursty"}
    sweep_df["pattern"] = sweep_df["workload_id"].map(patterns)

    models = [m for m in ["persistence", "linear", "tree"] if m in sweep_df["model"].unique()]
    colors = {"persistence": "#e07b39", "linear": "#4c8cbf", "tree": "#5aab5a"}
    markers = {"persistence": "o", "linear": "s", "tree": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, pattern in zip(axes, ["steady", "bursty"]):
        sub = sweep_df.loc[sweep_df["pattern"].eq(pattern)]
        grouped = sub.groupby(["model", "headroom"], as_index=False).agg(
            server_minutes=("server_minutes", "sum"),
            slo_violation_rate=("slo_violation_rate", "mean"),
        )

        for model in models:
            m_data = grouped.loc[grouped["model"].eq(model)].sort_values("server_minutes")
            ax.plot(
                m_data["server_minutes"],
                m_data["slo_violation_rate"],
                label=model,
                color=colors.get(model, "grey"),
                marker=markers.get(model, "x"),
                linewidth=1.5,
            )

        ax.set_title(f"{pattern} workloads")
        ax.set_xlabel("server-minutes (cost proxy)")
        ax.set_ylabel("SLO violation rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cost vs SLO violation rate across headroom factors")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    sim_config = json.loads(PART2_CONFIG.read_text())
    part3_cfg = json.loads(PART3_CONFIG.read_text())

    all_traffic, all_forecasts = load_replay_inputs(TRAFFIC_PATH, FORECAST_PATH)
    traffic = all_traffic.loc[all_traffic["split"].eq("validation")].copy()

    sweep_df = run_sweep(
        traffic=traffic,
        forecasts=all_forecasts,
        sim_config=sim_config,
        workloads=part3_cfg["workloads"],
        forecast_models=part3_cfg["forecast_models"],
        headroom_factors=part3_cfg["headroom_factors"],
        scoring_start_minute=part3_cfg["scoring_start_minute"],
        scoring_end_minute=part3_cfg["scoring_end_minute"],
        slo_seconds=part3_cfg["slo_seconds"],
    )

    SWEEP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(SWEEP_OUTPUT, index=False)
    print(f"\nSaved sweep results to {SWEEP_OUTPUT}")

    plot_sweep(sweep_df, FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

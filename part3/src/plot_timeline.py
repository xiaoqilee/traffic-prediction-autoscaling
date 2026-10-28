"""Timeline plot: traffic, forecasts, active servers, and queue length for one workload."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO_ROOT = Path(__file__).parents[2]
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"
TRAFFIC_PATH = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"
FORECAST_PATH = REPO_ROOT / "part1" / "outputs" / "forecasts" / "validation.csv"
MINUTE_LOGS_PATH = REPO_ROOT / "part3" / "outputs" / "validation" / "minute_logs.csv"
DIAGNOSTIC_WINDOWS_PATH = (
    REPO_ROOT / "part1" / "outputs" / "metrics" / "diagnostic_windows_validation.csv"
)
FIGURE_OUTPUT = REPO_ROOT / "part3" / "outputs" / "figures" / "timeline_bursty_02.png"

# Minutes to show on either side of the diagnostic window center
WINDOW_HALF_WIDTH = 60


def plot_timeline(
    traffic_df,
    forecasts_df,
    minute_logs_df,
    workload_id,
    models,
    center_minute,
    half_width,
    output_path,
):
    start = center_minute - half_width
    end = center_minute + half_width

    traffic = traffic_df.loc[
        traffic_df["workload_id"].eq(workload_id)
        & traffic_df["minute"].between(start, end)
    ]
    forecasts = forecasts_df.loc[
        forecasts_df["workload_id"].eq(workload_id)
        & forecasts_df["target_minute"].between(start, end)
    ]

    colors = {"persistence": "#e07b39", "linear": "#4c8cbf", "tree": "#5aab5a"}
    server_colors = {"persistence": "#e07b39", "linear": "#4c8cbf", "tree": "#5aab5a", "reactive": "#9b59b6"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Traffic and forecasts
    ax = axes[0]
    ax.plot(traffic["minute"], traffic["request_count"], color="black", label="actual", linewidth=1.5)
    for model in models:
        m_fc = forecasts.loc[forecasts["model"].eq(model)]
        ax.plot(
            m_fc["target_minute"],
            m_fc["predicted_count"],
            label=model,
            color=colors.get(model, "grey"),
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
        )
    ax.set_ylabel("requests / min")
    ax.set_title(f"{workload_id}: traffic and forecasts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Active servers per model
    ax = axes[1]
    all_models_for_logs = models + ["reactive"]
    for model in all_models_for_logs:
        run_id = f"{workload_id}_{model}"
        run_logs = minute_logs_df.loc[
            minute_logs_df["run_id"].eq(run_id)
            & minute_logs_df["minute"].between(start, end)
        ]
        if not run_logs.empty:
            ax.plot(
                run_logs["minute"],
                run_logs["active_servers"],
                label=model,
                color=server_colors.get(model, "grey"),
                linewidth=1.0,
            )
    ax.set_ylabel("active servers")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Queue length per model
    ax = axes[2]
    for model in all_models_for_logs:
        run_id = f"{workload_id}_{model}"
        run_logs = minute_logs_df.loc[
            minute_logs_df["run_id"].eq(run_id)
            & minute_logs_df["minute"].between(start, end)
        ]
        if not run_logs.empty:
            ax.plot(
                run_logs["minute"],
                run_logs["queue_length"],
                label=model,
                color=server_colors.get(model, "grey"),
                linewidth=1.0,
            )
    ax.set_ylabel("queue length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Drops per minute
    ax = axes[3]
    for model in all_models_for_logs:
        run_id = f"{workload_id}_{model}"
        run_logs = minute_logs_df.loc[
            minute_logs_df["run_id"].eq(run_id)
            & minute_logs_df["minute"].between(start, end)
        ]
        if not run_logs.empty:
            ax.plot(
                run_logs["minute"],
                run_logs["dropped"],
                label=model,
                color=server_colors.get(model, "grey"),
                linewidth=1.0,
            )
    ax.set_ylabel("drops / min")
    ax.set_xlabel("minute (dataset-relative)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark the diagnostic window center
    for ax in axes:
        ax.axvline(center_minute, color="red", linestyle=":", linewidth=1.0, alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    part3_cfg = json.loads(PART3_CONFIG.read_text())

    traffic_df = pd.read_csv(TRAFFIC_PATH)
    forecasts_df = pd.read_csv(FORECAST_PATH)
    minute_logs_df = pd.read_csv(MINUTE_LOGS_PATH)

    diag_windows = pd.read_csv(DIAGNOSTIC_WINDOWS_PATH)
    bursty_02_window = diag_windows.loc[
        diag_windows["workload_id"].eq("bursty_02")
    ].iloc[0]
    center_minute = int(bursty_02_window["center_minute"])

    plot_timeline(
        traffic_df=traffic_df,
        forecasts_df=forecasts_df,
        minute_logs_df=minute_logs_df,
        workload_id="bursty_02",
        models=part3_cfg["forecast_models"],
        center_minute=center_minute,
        half_width=WINDOW_HALF_WIDTH,
        output_path=FIGURE_OUTPUT,
    )


if __name__ == "__main__":
    main()

"""Controlled experiment: synthetic forecasts with matched MAE but different underprediction timing."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part2.simulator import load_replay_inputs, simulate_replay
from part3.src.metrics import summarize_run

REPO_ROOT = Path(__file__).parents[2]
PART2_CONFIG = REPO_ROOT / "part2" / "config.json"
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"
TRAFFIC_PATH = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"
FORECAST_PATH = REPO_ROOT / "part1" / "outputs" / "forecasts" / "validation.csv"
FORECAST_METRICS_PATH = (
    REPO_ROOT / "part1" / "outputs" / "metrics" / "forecast_validation.csv"
)
DIAGNOSTIC_WINDOWS_PATH = (
    REPO_ROOT / "part1" / "outputs" / "metrics" / "diagnostic_windows_validation.csv"
)
SYNTHETIC_OUTPUT = REPO_ROOT / "part3" / "outputs" / "metrics" / "synthetic_forecasts.csv"
EXPERIMENT_OUTPUT = REPO_ROOT / "part3" / "outputs" / "metrics" / "controlled_experiment.csv"
FIGURE_OUTPUT = REPO_ROOT / "part3" / "outputs" / "figures" / "controlled_experiment_bursty_01.png"

WORKLOAD = "bursty_01"


def _find_burst_episodes(actual_counts, train_q95):
    """Return list of (start_idx, end_idx) slices for runs above train_q95."""
    above = actual_counts > train_q95
    episodes = []
    in_burst = False
    start = 0
    for i, flag in enumerate(above):
        if flag and not in_burst:
            in_burst = True
            start = i
        elif not flag and in_burst:
            in_burst = False
            episodes.append((start, i))
    if in_burst:
        episodes.append((start, len(actual_counts)))
    return episodes


def build_synthetic_forecasts(traffic_df, workload_id, target_mae, train_q95, rng_seed=0):
    """
    Build two synthetic forecast series (synth_early, synth_late) with MAE matching
    target_mae but with underprediction concentrated at the start vs end of each burst.

    Returns a DataFrame with the same schema as the Part 1 forecasts file.
    """
    wl = traffic_df.loc[
        traffic_df["workload_id"].eq(workload_id)
        & traffic_df["split"].eq("validation")
    ].sort_values("minute").reset_index(drop=True)

    actuals = wl["request_count"].values.astype(float)
    minutes = wl["minute"].values

    episodes = _find_burst_episodes(actuals, train_q95)

    def make_predictions(alpha, mode):
        preds = actuals.copy()
        for start, end in episodes:
            ep_len = end - start
            half = max(1, ep_len // 2)
            if mode == "early":
                under_slice = slice(start, start + half)
                over_slice = slice(start + half, end)
            else:
                under_slice = slice(end - half, end)
                over_slice = slice(start, end - half)
            preds[under_slice] = actuals[under_slice] * (1.0 - alpha)
            shortfall = np.sum(actuals[under_slice] - preds[under_slice])
            over_count = over_slice.stop - over_slice.start
            if over_count > 0 and shortfall > 0:
                preds[over_slice] = actuals[over_slice] + shortfall / over_count
        return np.clip(preds, 0.0, None)

    def compute_mae(alpha, mode):
        preds = make_predictions(alpha, mode)
        return float(np.mean(np.abs(actuals - preds)))

    def tune_alpha(mode):
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if compute_mae(mid, mode) < target_mae:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    rows = []
    for mode, label in [("early", "synth_early"), ("late", "synth_late")]:
        alpha = tune_alpha(mode)
        preds = make_predictions(alpha, mode)
        for t, (minute, pred) in enumerate(zip(minutes, preds)):
            rows.append(
                {
                    "workload_id": workload_id,
                    "prediction_time": int(minute),
                    "target_minute": int(minute) + 3,
                    "model": label,
                    "predicted_count": round(float(pred), 4),
                }
            )

    return pd.DataFrame(rows)


def run_experiment(
    traffic,
    real_forecasts,
    synthetic_forecasts,
    sim_config,
    startup_delays,
    scoring_start_minute,
    scoring_end_minute,
    slo_seconds,
):
    rows = []
    all_models = ["persistence", "linear", "tree", "synth_early", "synth_late"]
    combined_forecasts = pd.concat([real_forecasts, synthetic_forecasts], ignore_index=True)

    for delay in startup_delays:
        config = {**sim_config, "startup_delay_minutes": delay}
        print(f"  startup_delay={delay}min")

        for model in all_models:
            policy = "forecast"
            run_id = f"{WORKLOAD}_{model}_delay{delay}"

            result = simulate_replay(
                traffic=traffic,
                forecasts=combined_forecasts,
                config=config,
                policy=policy,
                workload_id=WORKLOAD,
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
                    "workload_id": WORKLOAD,
                    "model": model,
                    "startup_delay_minutes": delay,
                    **service,
                }
            )

    return pd.DataFrame(rows)


def plot_experiment(experiment_df, real_mae_map, output_path):
    delays = sorted(experiment_df["startup_delay_minutes"].unique())
    models = ["persistence", "linear", "tree", "synth_early", "synth_late"]
    colors = {
        "persistence": "#e07b39",
        "linear": "#4c8cbf",
        "tree": "#5aab5a",
        "synth_early": "#c94040",
        "synth_late": "#8040c9",
    }

    fig, axes = plt.subplots(1, len(delays), figsize=(5 * len(delays), 5), sharey=True)
    if len(delays) == 1:
        axes = [axes]

    for ax, delay in zip(axes, delays):
        sub = experiment_df.loc[experiment_df["startup_delay_minutes"].eq(delay)]
        for model in models:
            row = sub.loc[sub["model"].eq(model)]
            if row.empty:
                continue
            slo_rate = float(row["slo_violation_rate"].iloc[0])
            mae = real_mae_map.get(model, None)
            label = f"{model} (MAE={mae:.1f})" if mae else model
            ax.bar(model, slo_rate, color=colors.get(model, "grey"), alpha=0.8, label=label)

        ax.set_title(f"startup delay = {delay}min")
        ax.set_ylabel("SLO violation rate")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{WORKLOAD}: controlled experiment — underprediction timing vs startup delay")
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

    forecast_metrics = pd.read_csv(FORECAST_METRICS_PATH)
    bursty_02_metrics = forecast_metrics.loc[
        forecast_metrics["workload_id"].eq(WORKLOAD)
    ]

    target_mae = float(
        bursty_02_metrics.loc[bursty_02_metrics["model"].eq("linear"), "mae"].iloc[0]
    )

    diag_windows = pd.read_csv(DIAGNOSTIC_WINDOWS_PATH)
    train_q95 = float(
        diag_windows.loc[diag_windows["workload_id"].eq(WORKLOAD), "train_q95"].iloc[0]
    )

    print(f"Building synthetic forecasts for {WORKLOAD} (target MAE={target_mae:.2f}, train_q95={train_q95:.1f})")
    synthetic = build_synthetic_forecasts(
        traffic_df=all_traffic,
        workload_id=WORKLOAD,
        target_mae=target_mae,
        train_q95=train_q95,
    )

    SYNTHETIC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(SYNTHETIC_OUTPUT, index=False)
    print(f"Saved synthetic forecasts to {SYNTHETIC_OUTPUT}")

    # Verify MAE of synthetic forecasts
    wl_traffic = all_traffic.loc[
        all_traffic["workload_id"].eq(WORKLOAD)
        & all_traffic["split"].eq("validation")
    ].set_index("minute")["request_count"]

    for model in ["synth_early", "synth_late"]:
        m_fc = synthetic.loc[synthetic["model"].eq(model)].set_index("target_minute")
        joined = m_fc["predicted_count"].reindex(wl_traffic.index)
        mae = float((joined - wl_traffic).abs().mean())
        print(f"  {model} MAE = {mae:.3f} (target {target_mae:.3f})")

    real_forecasts = all_forecasts.loc[
        all_forecasts["workload_id"].eq(WORKLOAD)
    ].copy()

    experiment_df = run_experiment(
        traffic=traffic,
        real_forecasts=real_forecasts,
        synthetic_forecasts=synthetic,
        sim_config=sim_config,
        startup_delays=part3_cfg["startup_delays_for_controlled"],
        scoring_start_minute=part3_cfg["scoring_start_minute"],
        scoring_end_minute=part3_cfg["scoring_end_minute"],
        slo_seconds=part3_cfg["slo_seconds"],
    )

    experiment_df.to_csv(EXPERIMENT_OUTPUT, index=False)
    print(f"\nSaved experiment results to {EXPERIMENT_OUTPUT}")
    print(experiment_df.to_string(index=False))

    real_mae_map = dict(
        zip(bursty_02_metrics["model"], bursty_02_metrics["mae"])
    )
    real_mae_map["synth_early"] = target_mae
    real_mae_map["synth_late"] = target_mae

    plot_experiment(experiment_df, real_mae_map, FIGURE_OUTPUT)


if __name__ == "__main__":
    main()

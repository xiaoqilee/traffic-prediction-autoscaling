"""Multi-seed validation: run each (workload, model) across many arrival seeds and
report mean +/- 95% confidence intervals, plus seed-paired model contrasts.

The simulator's only randomness is the within-minute arrival timing (seeded). Each
seed is one sample; aggregating across seeds shows whether forecast-quality gaps in
service metrics are real or arrival-timing noise. Includes a seasonal-naive baseline
(demand one day earlier), which is distinct from persistence/reactive.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from part2.simulator import load_replay_inputs, simulate_replay
from part3.src.metrics import summarize_run
from part3.src.baselines import (
    SEASONAL_NAIVE_MODEL,
    build_seasonal_naive_forecast,
    forecast_accuracy,
)

REPO_ROOT = Path(__file__).parents[2]
PART2_CONFIG = REPO_ROOT / "part2" / "config.json"
PART3_CONFIG = REPO_ROOT / "part3" / "configs" / "part3.json"

BUNDLE = REPO_ROOT / "part1" / "outputs" / "underprediction_ablation_v1"
BUNDLE_TRAFFIC = BUNDLE / "traffic_validation.csv"
BUNDLE_FORECASTS = BUNDLE / "forecasts" / "validation.csv"
BUNDLE_METRICS = BUNDLE / "metrics" / "forecast_validation.csv"
FULL_TRAFFIC = REPO_ROOT / "part1" / "data" / "processed" / "traffic.csv"

OUTPUT_DIR = REPO_ROOT / "part3" / "outputs" / "metrics"
SEED_RUNS_PATH = OUTPUT_DIR / "seed_runs.csv"
CI_TABLE_PATH = OUTPUT_DIR / "comparison_table_ci.csv"
CONTRASTS_PATH = OUTPUT_DIR / "contrasts_bursty_01.csv"

PATTERNS = {"steady_01": "steady", "steady_02": "steady",
            "bursty_01": "bursty", "bursty_02": "bursty"}

# Service metrics aggregated across seeds.
METRICS = ["slo_violation_rate", "drop_rate", "p95_latency", "p99_latency", "server_minutes"]

# Seed-paired contrasts reported for bursty_01 (model_a - model_b).
CONTRAST_PAIRS = [
    ("linear", "persistence"),
    ("tree", "persistence"),
    ("linear", "tree"),
    ("tree_underprediction", "tree"),
    (SEASONAL_NAIVE_MODEL, "persistence"),
]


def ci95_halfwidth(values):
    """t-based 95% CI half-width for a sample; 0 when variance is absent."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))


def run_all_seeds(traffic, forecasts, sim_config, workloads, forecast_models, seeds,
                  scoring_start, scoring_end, slo_seconds):
    rows = []
    for seed in seeds:
        config = {**sim_config, "random_seed": seed}
        print(f"  seed={seed}")
        for workload_id in workloads:
            policies = [("forecast", m) for m in forecast_models]
            policies.append(("reactive", "reactive"))
            for policy, model in policies:
                result = simulate_replay(
                    traffic=traffic,
                    forecasts=forecasts,
                    config=config,
                    policy=policy,
                    workload_id=workload_id,
                    model=None if policy == "reactive" else model,
                    run_id=f"{workload_id}_{model}_s{seed}",
                )
                service = summarize_run(
                    result["request_outcomes"], result["minute_logs"],
                    scoring_start, scoring_end, slo_seconds,
                )
                rows.append({
                    "seed": seed, "workload_id": workload_id, "model": model,
                    **{k: service[k] for k in METRICS},
                })
    return pd.DataFrame(rows)


def aggregate(seed_runs, accuracy):
    out = []
    for (workload_id, model), group in seed_runs.groupby(["workload_id", "model"]):
        row = {"workload_id": workload_id, "model": model,
               "pattern": PATTERNS[workload_id], "n_seeds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_ci95"] = ci95_halfwidth(group[metric].values)
        out.append(row)
    table = pd.DataFrame(out).merge(accuracy, on=["workload_id", "model"], how="left")
    front = ["workload_id", "model", "pattern", "mae", "rmse", "n_seeds"]
    return table[front + [c for c in table.columns if c not in front]]


def paired_contrasts(seed_runs, workload_id, metric="slo_violation_rate"):
    wl = seed_runs.loc[seed_runs["workload_id"].eq(workload_id)]
    wide = wl.pivot(index="seed", columns="model", values=metric)
    rows = []
    for a, b in CONTRAST_PAIRS:
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = (wide[a] - wide[b]).dropna()
        mean_diff = float(diff.mean())
        ci = ci95_halfwidth(diff.values)
        rows.append({
            "metric": metric,
            "contrast": f"{a} - {b}",
            "mean_diff": mean_diff,
            "ci95": ci,
            "ci_low": mean_diff - ci,
            "ci_high": mean_diff + ci,
            # Significant at 95% when the CI excludes zero.
            "significant": bool((mean_diff - ci) * (mean_diff + ci) > 0),
            "n_seeds": int(len(diff)),
        })
    return pd.DataFrame(rows)


def main():
    sim_config = json.loads(PART2_CONFIG.read_text())
    cfg = json.loads(PART3_CONFIG.read_text())

    workloads = cfg["workloads"]
    seeds = list(range(cfg["n_seeds"]))
    scoring_start = cfg["scoring_start_minute"]
    scoring_end = cfg["scoring_end_minute"]
    slo_seconds = cfg["slo_seconds"]

    traffic, forecasts = load_replay_inputs(BUNDLE_TRAFFIC, BUNDLE_FORECASTS)
    full_traffic = pd.read_csv(FULL_TRAFFIC)

    # Build the seasonal-naive series over the same target grid as the real forecasts.
    targets_by_wl = {
        wl: forecasts.loc[forecasts["workload_id"].eq(wl), "target_minute"].unique()
        for wl in workloads
    }
    seasonal = build_seasonal_naive_forecast(
        full_traffic, workloads, targets_by_wl,
        period_minutes=cfg["seasonal_period_minutes"],
    )
    forecasts = pd.concat([forecasts, seasonal], ignore_index=True)

    forecast_models = cfg["forecast_models"] + [SEASONAL_NAIVE_MODEL]

    print(f"Running {len(workloads)} workloads x {len(forecast_models) + 1} policies x {len(seeds)} seeds")
    seed_runs = run_all_seeds(
        traffic, forecasts, sim_config, workloads, forecast_models, seeds,
        scoring_start, scoring_end, slo_seconds,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_runs.to_csv(SEED_RUNS_PATH, index=False)

    # Forecast accuracy: 5 bundle models from Part 1, seasonal-naive computed here.
    bundle_acc = pd.read_csv(BUNDLE_METRICS)[["workload_id", "model", "mae", "rmse"]]
    seasonal_acc = forecast_accuracy(seasonal, full_traffic, scoring_start, scoring_end)
    accuracy = pd.concat([bundle_acc, seasonal_acc], ignore_index=True)

    ci_table = aggregate(seed_runs, accuracy)
    ci_table.to_csv(CI_TABLE_PATH, index=False)

    contrasts = paired_contrasts(seed_runs, "bursty_01")
    contrasts.to_csv(CONTRASTS_PATH, index=False)

    pd.set_option("display.width", 200)
    show = ["workload_id", "model", "mae", "slo_violation_rate_mean",
            "slo_violation_rate_ci95", "p99_latency_mean", "server_minutes_mean"]
    print("\n=== Mean +/- 95% CI (bursty workloads) ===")
    print(ci_table.loc[ci_table["pattern"].eq("bursty"), show].to_string(index=False))
    print("\n=== Seed-paired contrasts on bursty_01 SLO violation rate ===")
    print(contrasts.to_string(index=False))
    print(f"\nSaved {SEED_RUNS_PATH.name}, {CI_TABLE_PATH.name}, {CONTRASTS_PATH.name}")


if __name__ == "__main__":
    main()

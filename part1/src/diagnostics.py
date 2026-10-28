"""Diagnose validation forecasts and prepare simulator handoff examples."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .forecast import FORECAST_COLUMNS, file_hash, score_predictions
else:
    from forecast import FORECAST_COLUMNS, file_hash, score_predictions


MODELS = ("persistence", "linear", "tree")
COLORS = {"persistence": "#4477AA", "linear": "#EE7733", "tree": "#228833"}
RUN_COLUMNS = [
    "workload_id", "model", "start_minute", "end_minute", "length_minutes",
    "shortfall_count", "mean_shortfall", "max_shortfall", "high_flow_minutes",
]


def recover_validation_traffic(validation, minute_range):
    """Recover full validation counts from labels and observed lag_1, without test reads."""
    labels = validation[["workload_id", "target_minute", "label"]].rename(
        columns={"target_minute": "minute", "label": "request_count"}
    )
    history = validation[["workload_id", "prediction_time", "lag_1"]].rename(
        columns={"prediction_time": "minute", "lag_1": "request_count"}
    )
    history["minute"] -= 1
    combined = pd.concat([labels, history], ignore_index=True)
    if combined.groupby(["workload_id", "minute"]).request_count.nunique().gt(1).any():
        raise ValueError("Validation labels and lag_1 disagree on observed counts")
    start, end = minute_range
    counts = combined.loc[combined.minute.ge(start) & combined.minute.lt(end)]
    counts = counts.drop_duplicates(["workload_id", "minute"]).sort_values(
        ["workload_id", "minute"], ignore_index=True
    )
    counts["request_count"] = counts["request_count"].astype("int64")
    counts["split"] = "validation"
    return counts


def join_validation_forecasts(validation, forecasts):
    keys = ["workload_id", "prediction_time", "target_minute"]
    if forecasts.duplicated(["workload_id", "model", "target_minute"]).any():
        raise ValueError("Duplicate application/model/target forecast keys")
    joined = forecasts.merge(validation[keys + ["label"]], on=keys, how="left", validate="many_to_one")
    if joined["label"].isna().any():
        raise ValueError("A forecast has no matching validation label and prediction time")
    for (workload_id, _), group in joined.groupby(["workload_id", "model"]):
        expected = validation.loc[validation.workload_id.eq(workload_id), keys].sort_values("target_minute")
        actual = group[keys].sort_values("target_minute")
        if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
            raise ValueError(f"Forecast methods must share the validation window: {workload_id}")
    return joined.rename(columns={"label": "actual_count"}).sort_values(
        ["workload_id", "model", "target_minute"], ignore_index=True
    )


def underprediction_runs(rows, threshold):
    rows = rows.sort_values("target_minute").copy()
    under = rows.predicted_count.lt(rows.actual_count)
    starts = under & (~under.shift(fill_value=False) | rows.target_minute.diff().ne(1))
    selected = rows.loc[under].copy()
    if selected.empty:
        return pd.DataFrame(columns=RUN_COLUMNS)
    selected["run_id"] = starts.cumsum().loc[under]
    selected["shortfall"] = selected.actual_count - selected.predicted_count
    selected["high"] = selected.actual_count.gt(threshold)
    runs = selected.groupby("run_id", sort=False).agg(
        start_minute=("target_minute", "min"), end_minute=("target_minute", "max"),
        length_minutes=("target_minute", "size"), shortfall_count=("shortfall", "sum"),
        mean_shortfall=("shortfall", "mean"), max_shortfall=("shortfall", "max"),
        high_flow_minutes=("high", "sum"),
    ).reset_index(drop=True)
    runs["workload_id"] = rows.workload_id.iloc[0]
    runs["model"] = rows.model.iloc[0]
    return runs[RUN_COLUMNS]


def compute_diagnostics(joined, workloads):
    training = workloads.set_index("workload_id")
    metrics, all_runs = [], []
    for (workload_id, model), rows in joined.groupby(["workload_id", "model"], sort=False):
        reference = training.loc[workload_id]
        threshold = float(reference.train_p95)
        high = rows.actual_count.gt(threshold)
        shortfall = (rows.actual_count - rows.predicted_count).clip(lower=0)
        high_actual = float(rows.loc[high, "actual_count"].sum())
        high_shortfall = float(shortfall.loc[high].sum())
        runs = underprediction_runs(rows, threshold)
        all_runs.append(runs)
        metrics.append({
            "workload_id": workload_id, "pattern": reference.pattern, "model": model,
            **score_predictions(rows.actual_count, rows.predicted_count),
            "train_q95": threshold,
            "high_flow_minutes": int(high.sum()),
            "high_flow_actual_count": high_actual,
            "high_flow_shortfall_count": high_shortfall,
            "high_flow_underprediction_fraction": high_shortfall / high_actual if high_actual else np.nan,
            "underprediction_minutes": int(shortfall.gt(0).sum()),
            "underprediction_fraction": float(shortfall.gt(0).mean()),
            "underprediction_count_sum": float(shortfall.sum()),
            "underprediction_runs": len(runs),
            "mean_run_length": float(runs.length_minutes.mean()) if len(runs) else 0.0,
            "max_run_length": int(runs.length_minutes.max()) if len(runs) else 0,
        })
    nonempty_runs = [frame for frame in all_runs if len(frame)]
    runs = pd.concat(nonempty_runs, ignore_index=True) if nonempty_runs else pd.DataFrame(columns=RUN_COLUMNS)
    return pd.DataFrame(metrics), runs


def traffic_shift(workloads, counts):
    records = []
    for row in workloads.itertuples(index=False):
        observed = counts.loc[counts.workload_id.eq(row.workload_id), "request_count"]
        validation_mean = float(observed.mean())
        records.append({
            "workload_id": row.workload_id, "pattern": row.pattern,
            "train_minutes": row.train_minutes, "validation_minutes": len(observed),
            "train_mean": row.train_mean, "validation_mean": validation_mean,
            "relative_mean_change": validation_mean / row.train_mean - 1 if row.train_mean else np.nan,
            "train_zero_fraction": row.train_zero_fraction,
            "validation_zero_fraction": float(observed.eq(0).mean()),
            "zero_fraction_change": float(observed.eq(0).mean()) - row.train_zero_fraction,
            "train_q95": row.train_p95,
            "validation_q95": float(observed.quantile(0.95)),
        })
    return pd.DataFrame(records)


def select_windows(validation, workloads):
    records = []
    for row in workloads.itertuples(index=False):
        observed = validation.loc[validation.workload_id.eq(row.workload_id)].sort_values("target_minute")
        jumps = observed.label.diff().fillna(0)
        high_rises = observed.label.gt(row.train_p95) & jumps.gt(0)
        center_index = jumps.loc[high_rises].idxmax() if high_rises.any() else jumps.idxmax()
        center = int(observed.loc[center_index, "target_minute"])
        first, last = int(observed.target_minute.min()), int(observed.target_minute.max())
        start = max(first, min(center - 60, last - 120))
        records.append({
            "workload_id": row.workload_id, "pattern": row.pattern,
            "start_minute": start, "end_minute": min(start + 120, last),
            "center_minute": center, "rise_count": float(jumps.loc[center_index]),
            "train_q95": row.train_p95,
            "selection_rule": "Largest positive rise above train q95; otherwise largest observed rise",
        })
    return pd.DataFrame(records)


def select_case(metrics, windows):
    pool = windows.loc[windows.pattern.eq("bursty")]
    if pool.empty:
        pool = windows
    gaps = []
    for row in pool.itertuples(index=False):
        scores = metrics.loc[metrics.workload_id.eq(row.workload_id)].dropna(
            subset=["high_flow_underprediction_fraction"]
        )
        if len(scores):
            mae_best = scores.loc[scores.mae.idxmin()]
            gap = float(mae_best.high_flow_underprediction_fraction - scores.high_flow_underprediction_fraction.min())
            gaps.append((gap, row.workload_id))
    positive = sorted((item for item in gaps if item[0] > 0), key=lambda item: (-item[0], item[1]))
    if positive:
        workload_id = positive[0][1]
        rule = "Bursty workload with largest high-flow shortfall gap between MAE winner and high-flow winner"
    else:
        workload_id = pool.sort_values(["rise_count", "workload_id"], ascending=[False, True]).iloc[0].workload_id
        rule = "Largest high-flow rise among bursty workloads; no MAE/high-flow ordering disagreement"
    result = windows.loc[windows.workload_id.eq(workload_id)].iloc[0].to_dict()
    result["case_selection_rule"] = rule
    return result


def draw_forecasts(axis, rows, threshold, relative_days=False):
    actual = rows.drop_duplicates("target_minute").sort_values("target_minute")
    x = actual.target_minute / 1440 + 1 if relative_days else actual.target_minute
    axis.plot(x, actual.actual_count, color="#222222", linewidth=1, label="Actual", zorder=5)
    for model in MODELS:
        group = rows.loc[rows.model.eq(model)].sort_values("target_minute")
        target = group.target_minute / 1440 + 1 if relative_days else group.target_minute
        axis.plot(target, group.predicted_count, color=COLORS[model], alpha=0.8,
                  linewidth=0.7, label=model)
    axis.axhline(threshold, color="#888888", linestyle="--", linewidth=0.8, label="Train q95")
    axis.set_ylabel("HTTP invocations / minute")
    axis.grid(alpha=0.2)


def plot_validation(joined, windows, path):
    fig, axes = plt.subplots(len(windows), 2, figsize=(16, 3 * len(windows)), squeeze=False)
    for position, window in enumerate(windows.itertuples(index=False)):
        rows = joined.loc[joined.workload_id.eq(window.workload_id)]
        full, zoom = axes[position]
        draw_forecasts(full, rows, window.train_q95, relative_days=True)
        full.set(title=f"{window.workload_id}: full validation", xlabel="Relative dataset day")
        selected = rows.loc[rows.target_minute.between(window.start_minute, window.end_minute)]
        draw_forecasts(zoom, selected, window.train_q95)
        zoom.axvline(window.center_minute, color="#AA3377", alpha=0.6, linewidth=0.8)
        zoom.set(title=f"{window.workload_id}: selected rise at minute {window.center_minute}",
                 xlabel="Target minute (dataset-relative)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_case(joined, case, path):
    rows = joined.loc[joined.workload_id.eq(case["workload_id"])
                      & joined.target_minute.between(case["start_minute"], case["end_minute"])]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    draw_forecasts(axes[0], rows, case["train_q95"])
    actual = rows.drop_duplicates("target_minute").sort_values("target_minute")
    axes[0].fill_between(actual.target_minute, 0, 1, where=actual.actual_count.gt(case["train_q95"]),
                         transform=axes[0].get_xaxis_transform(), color="#CC6677", alpha=0.12, step="mid")
    axes[0].set_title(f"{case['workload_id']}: forecasts aligned to their target minutes")
    axes[0].legend(loc="upper right", ncol=5)
    for model in MODELS:
        group = rows.loc[rows.model.eq(model)].sort_values("target_minute")
        shortfall = (group.actual_count - group.predicted_count).clip(lower=0)
        axes[1].plot(group.target_minute, shortfall, color=COLORS[model], label=model)
    axes[1].set(xlabel="Target minute (dataset-relative)", ylabel="Underpredicted count / minute")
    axes[1].legend(loc="upper right", ncol=3)
    for axis in axes:
        axis.axvline(case["center_minute"], color="#AA3377", alpha=0.6, linewidth=0.8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_handoff(counts, forecasts, windows, case, horizon, output_root):
    directory = output_root / "examples"
    directory.mkdir(parents=True, exist_ok=True)
    steady = windows.loc[windows.pattern.eq("steady"), "workload_id"].tolist()
    selected_ids = list(dict.fromkeys(steady[:1] + [case["workload_id"]]))
    traffic_parts, forecast_parts, records = [], [], []
    for workload_id in selected_ids:
        window = windows.loc[windows.workload_id.eq(workload_id)].iloc[0]
        prediction = forecasts.loc[forecasts.workload_id.eq(workload_id)
                                   & forecasts.target_minute.between(window.start_minute, window.end_minute)]
        first = int(prediction.prediction_time.min())
        last = int(prediction.target_minute.max())
        observed = counts.loc[counts.workload_id.eq(workload_id) & counts.minute.between(first, last)]
        traffic_parts.append(observed)
        forecast_parts.append(prediction)
        records.append({"workload_id": workload_id, "traffic_start_minute": first,
                        "traffic_end_minute": last, "scoring_target_start": int(window.start_minute),
                        "scoring_target_end": int(window.end_minute), "warmup_minutes": horizon})
    traffic = pd.concat(traffic_parts, ignore_index=True)
    prediction = pd.concat(forecast_parts, ignore_index=True)
    traffic.to_csv(directory / "traffic.csv", index=False)
    prediction[FORECAST_COLUMNS].to_csv(directory / "forecasts.csv", index=False)
    metadata = {
        "horizon_minutes": horizon,
        "traffic_rows": len(traffic), "all_model_forecast_rows": len(prediction),
        "workloads": records,
        "time_index": "Original dataset-relative minutes; subtract the same origin from all time columns if rebasing",
        "initialization": "Replay warmup traffic under common initial capacity; score from scoring_target_start",
        "simulator_integration_executed": False,
    }
    (directory / "handoff.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def run_diagnostics(config_path, output_root):
    config = json.loads(config_path.read_text())
    sample_dir = Path(config["sample_dir"])
    source_paths = {"workloads": Path(config["workload_manifest"]),
                    "validation": sample_dir / "validation.csv",
                    "feature_contract": sample_dir / "feature_contract.json",
                    "forecasts": output_root / "forecasts/validation.csv"}
    workloads = pd.read_csv(source_paths["workloads"])
    validation = pd.read_csv(source_paths["validation"])
    contract = json.loads(source_paths["feature_contract"].read_text())
    forecasts = pd.read_csv(source_paths["forecasts"])
    joined = join_validation_forecasts(validation, forecasts)
    counts = recover_validation_traffic(validation, contract["split_minutes"]["validation"])
    metrics, runs = compute_diagnostics(joined, workloads)
    shifts = traffic_shift(workloads, counts)
    windows = select_windows(validation, workloads)
    case = select_case(metrics, windows)
    center_rows = joined.loc[joined.workload_id.eq(case["workload_id"])
                             & joined.target_minute.eq(case["center_minute"])]
    metric_dir, figure_dir = output_root / "metrics", output_root / "figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metric_dir / "forecast_diagnostics_validation.csv", index=False)
    runs.to_csv(metric_dir / "underprediction_runs_validation.csv", index=False)
    shifts.to_csv(metric_dir / "traffic_shift_validation.csv", index=False)
    windows.to_csv(metric_dir / "diagnostic_windows_validation.csv", index=False)
    plot_validation(joined, windows, figure_dir / "forecast_validation.png")
    plot_case(joined, case, figure_dir / "underprediction_case.png")
    write_handoff(counts, forecasts, windows, case, contract["horizon_minutes"], output_root)
    manifest = {
        "output_root": str(output_root),
        "horizon_minutes": contract["horizon_minutes"], "forecast_rows": len(forecasts),
        "diagnostic_rows": len(metrics), "underprediction_runs": len(runs),
        "high_flow_definition": "actual_count > training q95 from complete days 1-8 in workload_manifest.csv",
        "high_flow_fraction": "sum(max(actual-predicted,0) over high-flow minutes) / sum(actual over same minutes)",
        "no_high_flow_samples": "NaN in CSV, not zero",
        "underprediction_run_definition": "predicted < actual; equality, overprediction or minute gaps terminate a run",
        "validation_actual_source": "Union of validation labels and lag_1 observations, checked for agreement",
        "case": case, "test_used": False, "models_refitted": False,
        "case_observation": {
            "actual_count": int(center_rows.actual_count.iloc[0]),
            "predictions": {row.model: float(row.predicted_count) for row in center_rows.itertuples(index=False)},
        },
        "input_sha256": {name: file_hash(path) for name, path in source_paths.items()},
    }
    (metric_dir / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics[["workload_id", "model", "high_flow_minutes", "high_flow_underprediction_fraction",
                   "mean_run_length", "max_run_length"]].to_string(index=False))
    print(f"Case: {case['workload_id']} at target minute {case['center_minute']}; test not read.")
    return metrics, runs, shifts, windows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/part1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_diagnostics(args.config, args.output_root)


if __name__ == "__main__":
    main()

"""Compare frozen ordinary forecasts with symmetric and asymmetric quantile trees."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

if __package__:
    from .diagnostics import (COLORS, compute_diagnostics, draw_forecasts,
                              join_validation_forecasts, plot_validation,
                              recover_validation_traffic, select_windows,
                              traffic_shift, underprediction_runs)
    from .features import FEATURE_COLUMNS
    from .forecast import (FORECAST_COLUMNS, evaluate_saved, file_hash, forecast_frame,
                           prepare_output_dirs, score_predictions, write_json)
else:
    from diagnostics import (COLORS, compute_diagnostics, draw_forecasts,
                             join_validation_forecasts, plot_validation,
                             recover_validation_traffic, select_windows,
                             traffic_shift, underprediction_runs)
    from features import FEATURE_COLUMNS
    from forecast import (FORECAST_COLUMNS, evaluate_saved, file_hash, forecast_frame,
                          prepare_output_dirs, score_predictions, write_json)


BASELINES = ("persistence", "linear", "tree")
TREE_MODELS = ("tree", "tree_q50", "tree_underprediction")
DELTA_METRICS = ("mae", "rmse", "high_flow_underprediction_fraction", "high_flow_miss_rate",
                 "overprediction_count_sum", "overprediction_volume_fraction",
                 "max_run_length", "material_max_run_length")


def select_comparison_cases(validation, ordinary, workloads, failure):
    """Select both windows using only actual traffic and the ordinary tree."""
    failure = dict(failure)
    workload_id = failure["workload_id"]
    threshold = float(workloads.set_index("workload_id").loc[workload_id, "train_p95"])
    joined = join_validation_forecasts(validation, ordinary)
    rows = joined.loc[joined.workload_id.eq(workload_id) & joined.model.eq("tree")].copy()
    rows = rows.sort_values("target_minute").reset_index(drop=True)
    if not rows.target_minute.eq(failure["center_minute"]).any():
        raise ValueError("The fixed failure case is outside the validation targets")
    first, last = int(rows.target_minute.min()), int(rows.target_minute.max())
    if not first <= failure["start_minute"] <= failure["center_minute"] <= failure["end_minute"] <= last:
        raise ValueError("The fixed failure window is outside the validation targets")
    peak = rows.loc[rows.target_minute.eq(failure["center_minute"])].iloc[0]
    failure.update(case="failure", train_q95=threshold, peak_actual=float(peak.actual_count),
                   ordinary_peak_prediction=float(peak.predicted_count),
                   selection_rule="Fixed largest high-flow rise from the original diagnostics",
                   peak_matched=True, ordinary_peak_hit=bool(peak.predicted_count > threshold))
    def event_candidates(event_rows, event_threshold, exclude_failure):
        event_rows = event_rows.sort_values("target_minute")
        high = event_rows.actual_count.gt(event_threshold)
        starts = high & (~high.shift(fill_value=False) | event_rows.target_minute.diff().ne(1))
        candidates = []
        for _, event in event_rows.loc[high].groupby(starts.cumsum().loc[high], sort=False):
            peak = event.loc[event.actual_count.idxmax()]
            center = int(peak.target_minute)
            start, end = max(first, center - 60), min(last, center + 60)
            if exclude_failure and start <= failure["end_minute"] and end >= failure["start_minute"]:
                continue
            candidates.append({
                "workload_id": peak.workload_id, "start_minute": start, "end_minute": end,
                "center_minute": center, "train_q95": event_threshold,
                "event_start_minute": int(event.target_minute.min()),
                "event_end_minute": int(event.target_minute.max()),
                "peak_actual": float(peak.actual_count),
                "ordinary_peak_prediction": float(peak.predicted_count),
                "ordinary_peak_hit": bool(peak.predicted_count > event_threshold),
                "peak_matched": bool(abs(peak.actual_count / failure["peak_actual"] - 1) <= 0.2),
                "peak_relative_distance": float(abs(peak.actual_count / failure["peak_actual"] - 1)),
                "event_shortfall_fraction": float((event.actual_count - event.predicted_count).clip(lower=0).sum()
                                                   / event.actual_count.sum()),
            })
        return candidates

    candidates = event_candidates(rows, threshold, exclude_failure=True)
    if not candidates:
        return pd.DataFrame([failure])
    hit = [case for case in candidates if case["ordinary_peak_hit"]]
    matched = [case for case in hit if case["peak_matched"]]
    if matched:
        selected = min(matched, key=lambda case: (case["event_shortfall_fraction"], case["event_start_minute"]))
        rule = "Peak within 20%; ordinary peak hit; minimum event shortfall, then earliest event"
    elif hit:
        selected = min(hit, key=lambda case: (case["peak_relative_distance"], case["event_start_minute"]))
        rule = "No matched peak hit; nearest peak among ordinary peak hits, then earliest event"
    else:
        selected = min(candidates, key=lambda case: (case["event_shortfall_fraction"], case["event_start_minute"]))
        rule = "No ordinary peak hit; best available event shortfall, not a successful peak forecast"
    selected.update(case="success" if hit else "best_available", selection_rule=rule)
    cases = [failure, selected]
    if not hit:
        other_hits = []
        for reference in workloads.loc[workloads.pattern.eq("bursty") & workloads.workload_id.ne(workload_id)].itertuples():
            other_rows = joined.loc[joined.workload_id.eq(reference.workload_id) & joined.model.eq("tree")]
            other_hits.extend(case for case in event_candidates(other_rows, float(reference.train_p95), False)
                              if case["ordinary_peak_hit"])
        matched_hits = [case for case in other_hits if case["peak_matched"]]
        if other_hits:
            if matched_hits:
                success = min(matched_hits, key=lambda case: (case["event_shortfall_fraction"], case["workload_id"], case["event_start_minute"]))
            else:
                success = min(other_hits, key=lambda case: (case["peak_relative_distance"], case["workload_id"], case["event_start_minute"]))
            success.update(case="success_other_workload",
                           selection_rule="Supplementary other-bursty-workload peak hit; prefer matched peak and minimum event shortfall; not a within-workload control")
            cases.append(success)
    return pd.DataFrame(cases)


def summarize_validation(joined, workloads):
    """Keep per-application metrics and add equally weighted, normalized group summaries."""
    all_metrics, strict_runs = compute_diagnostics(joined, workloads)
    all_metrics = all_metrics.assign(scope="validation", day=0)
    daily = []
    for day, rows in joined.groupby(joined.target_minute // 1440 + 1):
        metrics, _ = compute_diagnostics(rows, workloads)
        daily.append(metrics.assign(scope="day", day=int(day)))
    detail = pd.concat([all_metrics, *daily], ignore_index=True)
    means = workloads.set_index("workload_id").train_mean
    detail["train_mean"] = detail.workload_id.map(means)
    denominator = detail.train_mean.replace(0, np.nan)
    detail["nmae"] = detail.mae / denominator
    detail["nrmse"] = detail.rmse / denominator
    grouped = detail.groupby(["scope", "day", "pattern", "model"], sort=True).agg(
        n_workloads=("workload_id", "nunique"), n_samples=("n_samples", "sum"),
        mean_nmae=("nmae", "mean"), mean_nrmse=("nrmse", "mean"),
        valid_nmae_workloads=("nmae", "count"), valid_nrmse_workloads=("nrmse", "count"),
        mean_high_flow_underprediction_fraction=("high_flow_underprediction_fraction", "mean"),
        valid_high_flow_workloads=("high_flow_underprediction_fraction", "count"),
        mean_overprediction_volume_fraction=("overprediction_volume_fraction", "mean"),
        valid_overprediction_workloads=("overprediction_volume_fraction", "count"),
    ).reset_index()
    comparisons = []
    for (scope, day, workload_id), rows in detail.groupby(["scope", "day", "workload_id"]):
        indexed = rows.set_index("model")
        candidate = indexed.loc["tree_underprediction"]
        for reference in ("tree", "tree_q50"):
            baseline = indexed.loc[reference]
            record = {"scope": scope, "day": day, "workload_id": workload_id,
                      "pattern": candidate.pattern, "model": "tree_underprediction", "reference": reference}
            for metric in DELTA_METRICS:
                record[f"{metric}_delta"] = candidate[metric] - baseline[metric]
            comparisons.append(record)
    return detail, grouped, pd.DataFrame(comparisons), strict_runs


def plot_cases(joined, cases, path):
    fig, axes = plt.subplots(2, len(cases), figsize=(8 * len(cases), 8), squeeze=False)
    for column, case in enumerate(cases.to_dict("records")):
        rows = joined.loc[joined.workload_id.eq(case["workload_id"])
                          & joined.target_minute.between(case["start_minute"], case["end_minute"])
                          & joined.model.isin(TREE_MODELS)]
        draw_forecasts(axes[0, column], rows, case["train_q95"])
        label = case["case"].replace("_", " ")
        axes[0, column].set_title(f"{label}: {case['workload_id']}, peak at {case['center_minute']}")
        horizon = int((rows.target_minute - rows.prediction_time).iloc[0])
        for axis in axes[:, column]:
            axis.axvline(case["center_minute"], color="#222222", linestyle=":", label="Target peak minute")
            axis.axvline(case["center_minute"] - horizon, color="#888888", linestyle="--",
                         label="Prediction boundary for peak")
        for model in TREE_MODELS:
            group = rows.loc[rows.model.eq(model)]
            axes[1, column].plot(group.target_minute, (group.actual_count - group.predicted_count).clip(lower=0),
                                 color=COLORS[model], label=model)
        axes[0, column].legend(fontsize=8)
        axes[1, column].set(xlabel="Target minute (dataset-relative)", ylabel="Underpredicted invocations / minute")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(metrics, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    markers = {"persistence": "o", "linear": "s", "tree": "D", "tree_q50": "x", "tree_underprediction": "+"}
    for axis, (workload_id, rows) in zip(axes.flat, metrics.groupby("workload_id", sort=True)):
        for row in rows.itertuples(index=False):
            axis.scatter(100 * row.overprediction_volume_fraction, 100 * row.high_flow_underprediction_fraction,
                         color=COLORS[row.model], marker=markers[row.model], label=row.model, s=70)
        axis.set(title=workload_id, xlabel="Overpredicted volume / actual volume (%)",
                 ylabel="High-flow shortfall / high-flow volume (%)")
        axis.grid(alpha=0.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def verify_inputs(config, baseline_config, baseline_manifest):
    """Fail before fitting if any frozen input no longer matches its receipt."""
    sample_dir = Path(baseline_config["sample_dir"])
    if baseline_config != baseline_manifest["configuration"]:
        raise ValueError("The baseline configuration differs from its saved training configuration")
    for name, expected in baseline_manifest["input_sha256"].items():
        if file_hash(sample_dir / name) != expected:
            raise ValueError(f"Frozen baseline input changed: {name}")
    receipt_path = Path(config["baseline_diagnostics_manifest"])
    receipt = json.loads(receipt_path.read_text())
    paths = {"forecasts": Path(config["baseline_output_root"]) / "forecasts/validation.csv",
             "workloads": Path(baseline_config["workload_manifest"])}
    for name, path in paths.items():
        if file_hash(path) != receipt["input_sha256"][name]:
            raise ValueError(f"Frozen diagnostic input changed: {name}")
    if baseline_manifest["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("The saved baseline feature order differs from the implementation")
    if baseline_manifest["refit_with_validation"] or baseline_manifest["test_used"]:
        raise ValueError("Expected training-only baselines with no held-out test evaluation")
    return paths


def run_ablation(config_path, output_root=None):
    config = json.loads(config_path.read_text())
    baseline_config = json.loads(Path(config["baseline_config"]).read_text())
    sample_dir = Path(baseline_config["sample_dir"])
    contract = json.loads((sample_dir / "feature_contract.json").read_text())
    horizon = contract["horizon_minutes"]
    baseline_root = Path(config["baseline_output_root"])
    baseline_model_dir = baseline_root / "models" / f"horizon_{horizon}"
    baseline_manifest_path = baseline_model_dir / "selected_models.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    paths = verify_inputs(config, baseline_config, baseline_manifest)
    output_root = Path(output_root or config["output_root"])
    if baseline_root.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("Experiment outputs must not replace the baseline output directory")
    if config["quantiles"] != {"tree_q50": 0.5, "tree_underprediction": 0.75}:
        raise ValueError("This experiment fixes the symmetric control at 0.50 and candidate at 0.75")
    train = pd.read_csv(sample_dir / "train.csv")
    validation = pd.read_csv(sample_dir / "validation.csv")
    workloads = pd.read_csv(paths["workloads"])
    ordinary = pd.read_csv(paths["forecasts"])
    expected_ids = set(baseline_manifest["workloads"])
    if any(set(frame.workload_id) != expected_ids for frame in (train, validation, workloads, ordinary)):
        raise ValueError("Training, validation, workload receipt and forecasts must share all applications")
    for _, rows in ordinary.groupby("workload_id"):
        if set(rows.model) != set(BASELINES):
            raise ValueError("Expected all three original baseline methods for every application")
    if not (ordinary.target_minute - ordinary.prediction_time).eq(horizon).all():
        raise ValueError("Forecast horizon differs from the frozen feature contract")
    for split, samples in (("train", train), ("validation", validation)):
        start, end = contract["split_minutes"][split]
        if (not samples.split.eq(split).all() or samples.duplicated(["workload_id", "target_minute"]).any()
                or not samples.prediction_time.between(start, end - horizon - 1).all()
                or not (samples.target_minute - samples.prediction_time).eq(horizon).all()
                or not np.isfinite(samples[FEATURE_COLUMNS + ["label"]]).all().all()
                or samples.label.lt(0).any()):
            raise ValueError(f"Invalid {split} samples or timing")
    join_validation_forecasts(validation, ordinary)
    cases = select_comparison_cases(validation, ordinary, workloads, config["failure_case"])
    model_dir = prepare_output_dirs(output_root, horizon)
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = output_root / "metrics"
    input_paths = {"config": config_path, "baseline_config": Path(config["baseline_config"]),
                   "baseline_manifest": baseline_manifest_path,
                   "baseline_diagnostics_manifest": Path(config["baseline_diagnostics_manifest"]),
                   "train": sample_dir / "train.csv", "validation": sample_dir / "validation.csv",
                   "feature_contract": sample_dir / "feature_contract.json", **paths}
    input_paths.update({f"model_{workload_id}_{name}": baseline_model_dir / entry["model_file"]
                        for workload_id, record in baseline_manifest["workloads"].items()
                        for name, entry in record["models"].items() if entry["model_file"]})
    frozen_hashes = {name: file_hash(path) for name, path in input_paths.items()}
    write_json(output_root / "design_manifest.json", {
        "configuration": config, "case_selection_before_modified_fitting": True,
        "cases": json.loads(cases.to_json(orient="records")), "input_sha256": frozen_hashes,
        "baseline_prediction_tolerance": {"rtol": config["prediction_rtol"], "atol": config["prediction_atol"]},
        "test_used": False,
    })
    records = deepcopy(baseline_manifest["workloads"])
    new_forecasts, fitting, reproduction = [], [], []
    with threadpool_limits(limits=1):
        for workload_id, record in records.items():
            training_rows = train.loc[train.workload_id.eq(workload_id)]
            validation_rows = validation.loc[validation.workload_id.eq(workload_id)]
            ordinary_tree = None
            for name in BASELINES:
                entry = record["models"][name]
                if name == "persistence":
                    predicted = validation_rows.lag_1.to_numpy()
                else:
                    source = baseline_model_dir / entry["model_file"]
                    estimator = joblib.load(source)
                    predicted = estimator.predict(validation_rows[FEATURE_COLUMNS])
                    destination = model_dir / entry["model_file"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    if name == "tree":
                        ordinary_tree = estimator
                expected = ordinary.loc[ordinary.workload_id.eq(workload_id) & ordinary.model.eq(name)].sort_values("target_minute")
                actual = forecast_frame(validation_rows, name, predicted).sort_values("target_minute")
                np.testing.assert_allclose(actual.predicted_count, expected.predicted_count,
                                           rtol=config["prediction_rtol"], atol=config["prediction_atol"])
                reproduction.append({"workload_id": workload_id, "model": name,
                                     "max_absolute_difference": float(np.abs(actual.predicted_count.to_numpy()
                                                                              - expected.predicted_count.to_numpy()).max())})
            for name, quantile in config["quantiles"].items():
                estimator = clone(ordinary_tree).set_params(loss="quantile", quantile=quantile)
                ordinary_params = ordinary_tree.get_params(deep=False)
                params = estimator.get_params(deep=False)
                changed = {key for key in params if params[key] != ordinary_params[key]}
                if changed != {"loss", "quantile"} or ordinary_params["early_stopping"]:
                    raise ValueError("Only loss and quantile may change; early stopping must remain disabled")
                started = perf_counter()
                estimator.fit(training_rows[FEATURE_COLUMNS], training_rows.label)
                fit_seconds = perf_counter() - started
                forecast = forecast_frame(validation_rows, name, estimator.predict(validation_rows[FEATURE_COLUMNS]))
                relative_path = Path(workload_id) / f"{name}.joblib"
                destination = model_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(estimator, destination)
                reloaded = forecast_frame(validation_rows, name, joblib.load(destination).predict(validation_rows[FEATURE_COLUMNS]))
                np.testing.assert_allclose(forecast.predicted_count, reloaded.predicted_count,
                                           rtol=config["prediction_rtol"], atol=config["prediction_atol"])
                new_forecasts.append(forecast)
                record["models"][name] = {"parameters": params, "model_file": str(relative_path),
                                          "validation_scores": score_predictions(validation_rows.label, forecast.predicted_count)}
                fitting.append({"workload_id": workload_id, "model": name, "fit_seconds": fit_seconds,
                                "training_samples": len(training_rows), "changed_parameters": "loss,quantile"})
            print(f"{workload_id}: verified baselines; fitted q=0.50 and q=0.75", flush=True)
    modified = pd.concat(new_forecasts, ignore_index=True)
    forecasts = pd.concat([ordinary, modified], ignore_index=True)[FORECAST_COLUMNS]
    joined = join_validation_forecasts(validation, forecasts)
    detail, grouped, comparisons, strict_runs = summarize_validation(joined, workloads)
    metrics = detail.loc[detail.scope.eq("validation")].copy()
    forecast_dir = output_root / "forecasts"
    shutil.copy2(paths["forecasts"], forecast_dir / "validation_ordinary.csv")
    modified.loc[modified.model.eq("tree_underprediction")].to_csv(forecast_dir / "validation_modified.csv", index=False)
    modified.loc[modified.model.eq("tree_q50")].to_csv(forecast_dir / "validation_control.csv", index=False)
    forecasts.to_csv(forecast_dir / "validation.csv", index=False)
    metrics[["workload_id", "model", "n_samples", "mae", "rmse", "pattern"]].to_csv(metric_dir / "forecast_validation.csv", index=False)
    metrics.to_csv(metric_dir / "forecast_diagnostics_validation.csv", index=False)
    detail.to_csv(metric_dir / "forecast_diagnostics_by_day_validation.csv", index=False)
    grouped.to_csv(metric_dir / "forecast_grouped_validation.csv", index=False)
    comparisons.to_csv(metric_dir / "ablation_comparison_validation.csv", index=False)
    strict_runs.to_csv(metric_dir / "underprediction_runs_validation.csv", index=False)
    thresholds = workloads.set_index("workload_id").train_p95
    material = [underprediction_runs(rows, thresholds.loc[workload_id], min_shortfall=1)
                for (workload_id, _), rows in joined.groupby(["workload_id", "model"])]
    pd.concat(material, ignore_index=True).to_csv(metric_dir / "material_underprediction_runs_validation.csv", index=False)
    pd.DataFrame(fitting).to_csv(metric_dir / "training_runs.csv", index=False)
    pd.DataFrame(reproduction).to_csv(metric_dir / "baseline_reproduction.csv", index=False)
    counts = recover_validation_traffic(validation, contract["split_minutes"]["validation"])
    counts.to_csv(output_root / "traffic_validation.csv", index=False)
    traffic_shift(workloads, counts).to_csv(metric_dir / "traffic_shift_validation.csv", index=False)
    cases.to_csv(metric_dir / "selected_cases_validation.csv", index=False)
    case_metrics, case_features = [], []
    for case in cases.to_dict("records"):
        rows = joined.loc[joined.workload_id.eq(case["workload_id"])
                          & joined.target_minute.between(case["start_minute"], case["end_minute"])]
        scores, _ = compute_diagnostics(rows, workloads)
        case_metrics.append(scores.assign(case=case["case"], start_minute=case["start_minute"],
                                          end_minute=case["end_minute"], center_minute=case["center_minute"]))
        features = validation.loc[validation.workload_id.eq(case["workload_id"])
                                  & validation.target_minute.eq(case["center_minute"])].copy()
        case_features.append(features.assign(case=case["case"]))
    pd.concat(case_metrics, ignore_index=True).to_csv(metric_dir / "case_comparison_validation.csv", index=False)
    pd.concat(case_features, ignore_index=True).to_csv(metric_dir / "case_features_validation.csv", index=False)
    plot_cases(joined, cases, figure_dir / "failure_success_comparison.png")
    plot_tradeoff(metrics, figure_dir / "underprediction_overprediction.png")
    plot_validation(joined, select_windows(validation, workloads), figure_dir / "forecast_validation.png")
    manifest = deepcopy(baseline_manifest)
    manifest.update(workloads=records, configuration=baseline_config,
                    selection_rule="Original baseline settings frozen; q=0.50 control and q=0.75 candidate fixed before fitting",
                    experiment_configuration=config, refit_with_validation=False, test_used=False)
    write_json(model_dir / "selected_models.json", manifest)
    if any(file_hash(input_paths[name]) != expected for name, expected in frozen_hashes.items()):
        raise ValueError("A frozen input changed during the experiment")
    code_root = Path(__file__).resolve().parents[1]
    code_hashes = {str(path.relative_to(code_root)): file_hash(path) for path in sorted((code_root / "src").glob("*.py"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, text=True, capture_output=True, check=False)
    experiment = {
        "status": "completed_validation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "horizon_minutes": horizon, "forecast_rows": len(forecasts),
        "traffic_rows": len(counts), "models": list(BASELINES) + list(config["quantiles"]),
        "input_files": {name: str(path) for name, path in input_paths.items()}, "input_sha256": frozen_hashes,
        "source_sha256": code_hashes, "git_commit": revision.stdout.strip() or None,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "thread_limit": 1, "test_used": False, "refit_with_validation": False,
        "new_model_fits": len(fitting), "baseline_predictions_reproduced": True,
        "saved_model_predictions_reproduced": True,
        "case_selection_before_modified_fitting": True,
        "case_selection_file": "metrics/selected_cases_validation.csv",
        "prediction_tolerance": {"rtol": config["prediction_rtol"], "atol": config["prediction_atol"]},
        "scoring": {"replay_start": int(counts.minute.min()), "replay_end": int(counts.minute.max()),
                    "target_start": int(validation.target_minute.min()), "target_end": int(validation.target_minute.max()),
                    "endpoints_inclusive": True},
        "metric_definitions": {
            "high_flow": "actual > full-training q95; zero denominators are missing",
            "high_flow_underprediction_fraction": "sum(max(actual-predicted,0) on high flow) / sum(actual on high flow)",
            "high_flow_miss_rate": "count(high flow and predicted <= training q95) / count(high flow)",
            "overprediction_volume_fraction": "sum(max(predicted-actual,0)) / sum(actual)",
            "material_underprediction": "actual-predicted >= 1 invocation/minute; gaps terminate runs",
            "grouping": "Equal application weights after normalizing MAE/RMSE by each training mean; not pooled RMSE",
            "comparison_delta": "tree_underprediction minus reference; fractions are unit intervals, not percentage points",
            "daily_rows": "day uses target_minute // 1440 + 1; scope=validation/day=0 means the complete period",
        },
    }
    experiment["output_sha256"] = {str(path.relative_to(output_root)): file_hash(path)
                                   for path in sorted(output_root.rglob("*"))
                                   if path.is_file() and path.name != "experiment_manifest.json"}
    write_json(output_root / "experiment_manifest.json", experiment)
    print(metrics[["workload_id", "model", "mae", "rmse", "high_flow_underprediction_fraction",
                   "overprediction_volume_fraction"]].to_string(index=False))
    print(f"Exported {len(forecasts):,} validation forecasts; held-out test was not read.")
    return forecasts, metrics, experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/part1-underprediction.json"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.split == "test":
        config = json.loads(args.config.read_text())
        evaluate_saved(Path(config["baseline_config"]), args.output_root or Path(config["output_root"]), split="test")
    else:
        run_ablation(args.config, args.output_root)


if __name__ == "__main__":
    main()

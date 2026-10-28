"""Train per-application baselines and export validation forecasts."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from .features import FEATURE_COLUMNS
else:
    from features import FEATURE_COLUMNS


FORECAST_COLUMNS = ["workload_id", "prediction_time", "target_minute", "model", "predicted_count"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_predictions(actual, predicted):
    return {
        "n_samples": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
    }


def forecast_frame(samples, model_name, predicted):
    frame = samples[["workload_id", "prediction_time", "target_minute"]].copy()
    frame["model"] = model_name
    frame["predicted_count"] = np.maximum(np.asarray(predicted, dtype=float), 0)
    return frame[FORECAST_COLUMNS].reset_index(drop=True)


def train_workload(train, validation, config):
    """Fit candidates on train only, selecting each family by clipped validation MAE."""
    workload_id = train["workload_id"].iloc[0]
    X_train, y_train = train[FEATURE_COLUMNS], train["label"]
    X_validation, y_validation = validation[FEATURE_COLUMNS], validation["label"]
    persistence = forecast_frame(validation, "persistence", validation["lag_1"])
    persistence_scores = score_predictions(y_validation, persistence["predicted_count"])
    predictions = [persistence]
    metrics = [{"workload_id": workload_id, "model": "persistence", **persistence_scores}]
    selections = {"persistence": {"parameters": {}, "validation_scores": persistence_scores}}
    models, candidates = {}, []

    ridge_candidates = [
        ({"alpha": alpha}, make_pipeline(StandardScaler(), Ridge(alpha=alpha)))
        for alpha in config["ridge"]["alpha"]
    ]
    tree_config = config["tree"].copy()
    leaf_options = tree_config.pop("max_leaf_nodes")
    tree_candidates = []
    for leaves in leaf_options:
        parameters = {**tree_config, "max_leaf_nodes": leaves, "random_state": config["random_seed"]}
        tree_candidates.append((parameters, HistGradientBoostingRegressor(**parameters)))

    for name, options in [("linear", ridge_candidates), ("tree", tree_candidates)]:
        best_mae, best_index = np.inf, None
        for parameters, estimator in options:
            started = perf_counter()
            estimator.fit(X_train, y_train)
            fit_seconds = perf_counter() - started
            forecast = forecast_frame(validation, name, estimator.predict(X_validation))
            scores = score_predictions(y_validation, forecast["predicted_count"])
            candidates.append({
                "workload_id": workload_id,
                "model": name,
                "parameters": json.dumps(parameters, sort_keys=True),
                **scores,
                "fit_seconds": fit_seconds,
                "selected": False,
            })
            # Exact MAE ties retain the first configuration in the declared grid.
            if scores["mae"] < best_mae:
                best_mae, best_index = scores["mae"], len(candidates) - 1
                best_forecast, best_scores = forecast, scores
                models[name] = estimator
                selections[name] = {"parameters": parameters, "validation_scores": scores}
        candidates[best_index]["selected"] = True
        predictions.append(best_forecast)
        metrics.append({"workload_id": workload_id, "model": name, **best_scores})

    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics),
            pd.DataFrame(candidates), models, selections)


def prepare_output_dirs(output_root, horizon):
    model_dir = output_root / "models" / f"horizon_{horizon}"
    for path in [output_root / "forecasts", output_root / "metrics", model_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return model_dir


def train_validation(config_path, output_root):
    config = json.loads(config_path.read_text())
    sample_dir = Path(config["sample_dir"])
    contract_path = sample_dir / "feature_contract.json"
    contract = json.loads(contract_path.read_text())
    horizon = contract["horizon_minutes"]
    train = pd.read_csv(sample_dir / "train.csv")
    validation = pd.read_csv(sample_dir / "validation.csv")
    patterns = pd.read_csv(config["workload_manifest"]).set_index("workload_id")["pattern"]
    model_dir = prepare_output_dirs(output_root, horizon)
    forecasts, metrics, candidates, workload_records = [], [], [], {}

    # Small tabular models avoid thread startup overhead and use the same CPU budget.
    with threadpool_limits(limits=1):
        for workload_id, train_rows in train.groupby("workload_id", sort=False):
            validation_rows = validation.loc[validation["workload_id"].eq(workload_id)]
            prediction, scores, search, models, selected = train_workload(
                train_rows, validation_rows, config
            )
            scores["pattern"] = patterns.loc[workload_id]
            forecasts.append(prediction)
            metrics.append(scores)
            candidates.append(search)
            app_dir = model_dir / workload_id
            app_dir.mkdir(parents=True, exist_ok=True)
            for name, estimator in models.items():
                model_file = app_dir / f"{name}.joblib"
                joblib.dump(estimator, model_file)
                selected[name]["model_file"] = str(model_file.relative_to(model_dir))
            selected["persistence"]["model_file"] = None
            workload_records[workload_id] = {
                "pattern": patterns.loc[workload_id],
                "training_samples": len(train_rows),
                "validation_samples": len(validation_rows),
                "models": selected,
            }
            print(f"{workload_id}: selected linear alpha={selected['linear']['parameters']['alpha']}, "
                  f"tree leaves={selected['tree']['parameters']['max_leaf_nodes']}", flush=True)

    forecasts = pd.concat(forecasts, ignore_index=True)
    metrics = pd.concat(metrics, ignore_index=True)
    candidates = pd.concat(candidates, ignore_index=True)
    forecasts.to_csv(output_root / "forecasts/validation.csv", index=False)
    metrics.to_csv(output_root / "metrics/forecast_validation.csv", index=False)
    manifest = {
        "sample_dir": str(sample_dir),
        "horizon_minutes": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "configuration": config,
        "selection_rule": "Minimum validation MAE after clipping; exact ties use grid order",
        "preprocessing": "Ridge Pipeline fits StandardScaler on train; tree uses raw features",
        "output_postprocessing": "Clip predictions to zero; retain floating point counts; no headroom",
        "refit_with_validation": False,
        "test_used": False,
        "thread_limit": 1,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": {
            name: file_hash(sample_dir / name)
            for name in ("train.csv", "validation.csv", "feature_contract.json")
        },
        "workloads": workload_records,
    }
    write_json(model_dir / "selected_models.json", manifest)
    print(metrics.to_string(index=False))
    print(f"Exported {len(forecasts):,} validation forecasts; held-out test was not read.")
    return forecasts, metrics, candidates, manifest


def evaluate_saved(config_path, output_root, split="test"):
    """Evaluate saved training-only models without fitting or selecting settings."""
    config = json.loads(config_path.read_text())
    sample_dir = Path(config["sample_dir"])
    contract = json.loads((sample_dir / "feature_contract.json").read_text())
    model_dir = output_root / "models" / f"horizon_{contract['horizon_minutes']}"
    manifest_path = model_dir / "selected_models.json"
    manifest = json.loads(manifest_path.read_text())
    samples = pd.read_csv(sample_dir / f"{split}.csv")
    forecasts, metrics = [], []
    with threadpool_limits(limits=1):
        for workload_id, record in manifest["workloads"].items():
            rows = samples.loc[samples["workload_id"].eq(workload_id)]
            for name, selected in record["models"].items():
                if name == "persistence":
                    values = rows["lag_1"]
                else:
                    estimator = joblib.load(model_dir / selected["model_file"])
                    values = estimator.predict(rows[manifest["feature_columns"]])
                forecast = forecast_frame(rows, name, values)
                forecasts.append(forecast)
                metrics.append({"workload_id": workload_id, "pattern": record["pattern"],
                                "model": name, **score_predictions(rows["label"], forecast["predicted_count"])})
    forecasts = pd.concat(forecasts, ignore_index=True)
    metrics = pd.DataFrame(metrics)
    forecasts.to_csv(output_root / "forecasts" / f"{split}.csv", index=False)
    metrics.to_csv(output_root / "metrics" / f"forecast_{split}.csv", index=False)
    write_json(output_root / "metrics" / f"{split}_run.json", {
        "split": split,
        "selection_manifest_sha256": file_hash(manifest_path),
        "sample_sha256": file_hash(sample_dir / f"{split}.csv"),
        "refit": False,
        "settings_selected_on_test": False,
    })
    print(metrics.to_string(index=False))
    return forecasts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/part1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.split == "validation":
        train_validation(args.config, args.output_root)
    else:
        evaluate_saved(args.config, args.output_root)


if __name__ == "__main__":
    main()

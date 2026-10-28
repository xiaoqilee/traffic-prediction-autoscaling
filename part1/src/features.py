"""Build causal forecasting samples from the minute-level traffic export."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LAGS = (1, 2, 3, 5, 15, 30, 60, 1440)
MEAN_WINDOWS = (5, 15, 60)
STD_WINDOWS = (15, 60)
HISTORY_MINUTES = max(LAGS)
FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"rolling_mean_{window}" for window in MEAN_WINDOWS]
    + [f"rolling_std_{window}" for window in STD_WINDOWS]
    + ["diff_1", "target_day_sin", "target_day_cos"]
)
METADATA_COLUMNS = ["workload_id", "prediction_time", "target_minute", "split"]
SPLIT_MINUTES = {
    "train": (0, 11520),
    "validation": (11520, 15840),
    "test": (15840, 20160),
}


def build_samples(traffic, horizon=3, split_minutes=None):
    """At boundary t, use counts through t-1 to predict the count at t+horizon.

    Both origin and target must belong to the assigned split. Missing minutes
    remain missing, so neither row shifts nor rolling windows jump over gaps.
    """
    if horizon < 0:
        raise ValueError("Forecast horizon must be nonnegative")
    split_minutes = SPLIT_MINUTES if split_minutes is None else split_minutes
    frames = []
    for workload_id, workload in traffic.groupby("workload_id", sort=False):
        workload = workload.sort_values("minute").set_index("minute")
        if workload.index.has_duplicates:
            raise ValueError(f"Duplicate minutes for {workload_id}")
        minutes = pd.RangeIndex(int(workload.index.min()), int(workload.index.max()) + 1)
        counts = workload["request_count"].reindex(minutes).astype(float)
        history = counts.shift(1)
        frame = pd.DataFrame(index=minutes)
        frame["workload_id"] = workload_id
        frame["prediction_time"] = minutes
        frame["target_minute"] = minutes + horizon
        frame["split"] = pd.Series(pd.NA, index=minutes, dtype="string")
        for name, (start, end) in split_minutes.items():
            within_split = (
                frame["prediction_time"].ge(start)
                & frame["prediction_time"].lt(end)
                & frame["target_minute"].ge(start)
                & frame["target_minute"].lt(end)
            )
            frame.loc[within_split, "split"] = name
        for lag in LAGS:
            frame[f"lag_{lag}"] = counts.shift(lag)
        for window in MEAN_WINDOWS:
            frame[f"rolling_mean_{window}"] = history.rolling(window).mean()
        for window in STD_WINDOWS:
            frame[f"rolling_std_{window}"] = history.rolling(window).std(ddof=0)
        frame["diff_1"] = history.diff()
        phase = 2 * np.pi * (frame["target_minute"] % 1440) / 1440
        frame["target_day_sin"] = np.sin(phase)
        frame["target_day_cos"] = np.cos(phase)
        frame["label"] = counts.shift(-horizon)
        frame = frame.dropna(subset=["split", "label", *FEATURE_COLUMNS])
        frame["label"] = frame["label"].astype("int64")
        frames.append(frame[METADATA_COLUMNS + FEATURE_COLUMNS + ["label"]])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["workload_id", "prediction_time"], ignore_index=True
    )


def split_workload_samples(samples, workload_id):
    """Return full sample rows for one application, separated by target split."""
    workload = samples.loc[samples["workload_id"].eq(workload_id)]
    return {
        name: workload.loc[workload["split"].eq(name)].reset_index(drop=True)
        for name in SPLIT_MINUTES
    }


def standardize_splits(splits):
    """Fit one application's scaler on training features and reuse it unchanged."""
    scaler = StandardScaler().fit(splits["train"][FEATURE_COLUMNS])
    transformed = {}
    for name, samples in splits.items():
        frame = samples.copy()
        frame[FEATURE_COLUMNS] = scaler.transform(samples[FEATURE_COLUMNS])
        transformed[name] = frame
    return transformed, scaler


def load_workload_samples(sample_dir, workload_id, standardize=False):
    """Load raw samples for trees/persistence, or scaled samples for Ridge.

    Returned frames retain label and time metadata. Only FEATURE_COLUMNS belong
    in model X; label is y. The second return value is the fitted scaler or None.
    """
    splits = {}
    for name in SPLIT_MINUTES:
        frame = pd.read_csv(Path(sample_dir) / f"{name}.csv")
        splits[name] = frame.loc[frame["workload_id"].eq(workload_id)].reset_index(drop=True)
    if standardize:
        return standardize_splits(splits)
    return splits, None


def prepare_features(traffic_path, contract_path, output_root, horizon=3):
    traffic = pd.read_csv(traffic_path)
    source_contract = json.loads(contract_path.read_text())
    split_minutes = source_contract["split_minutes"]
    samples = build_samples(traffic, horizon, split_minutes)
    output_dir = output_root / f"horizon_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in split_minutes:
        samples.loc[samples["split"].eq(name)].to_csv(output_dir / f"{name}.csv", index=False)

    scalers = {}
    for workload_id in samples["workload_id"].unique():
        splits = split_workload_samples(samples, workload_id)
        _, scaler = standardize_splits(splits)
        scalers[workload_id] = {
            "fit_split": "train",
            "training_samples": int(scaler.n_samples_seen_),
            "feature_names": FEATURE_COLUMNS,
            "mean": scaler.mean_.tolist(),
            "variance": scaler.var_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
    (output_dir / "scalers.json").write_text(
        json.dumps(scalers, indent=2, ensure_ascii=False) + "\n"
    )

    records = []
    for (workload_id, split), group in samples.groupby(["workload_id", "split"], sort=False):
        records.append({
            "workload_id": workload_id,
            "split": split,
            "sample_count": len(group),
            "first_prediction_time": int(group["prediction_time"].min()),
            "last_prediction_time": int(group["prediction_time"].max()),
            "first_target_minute": int(group["target_minute"].min()),
            "last_target_minute": int(group["target_minute"].max()),
        })
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "sample_manifest.csv", index=False)
    feature_contract = {
        "traffic_file": str(traffic_path),
        "traffic_sha256": hashlib.sha256(traffic_path.read_bytes()).hexdigest(),
        "horizon_minutes": horizon,
        "prediction_time": "Boundary t; latest complete observed count is y[t-1]",
        "target_minute": "t + horizon_minutes",
        "label": "Observed count in target_minute; kept in original count units",
        "metadata_columns": METADATA_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_definitions": {
            "lag_k": "y[t-k]",
            "rolling_mean_w": "mean(y[t-w], ..., y[t-1]); require all w observations",
            "rolling_std_w": "population std(y[t-w], ..., y[t-1]); ddof=0",
            "diff_1": "y[t-1] - y[t-2]",
            "target_day_sin": "sin(2*pi*((t+horizon_minutes) % 1440)/1440)",
            "target_day_cos": "cos(2*pi*((t+horizon_minutes) % 1440)/1440)",
        },
        "history_minutes_required": HISTORY_MINUTES,
        "split_minutes": split_minutes,
        "split_end_is_exclusive": True,
        "split_rule": "Both prediction_time and target_minute must fall in the assigned split",
        "history_across_splits": "Use earlier observed counts, including earlier validation/test minutes",
        "missing_values": "No imputation; omit samples with missing required features or labels",
        "csv_features": "Unscaled; trees and persistence use these directly",
        "standardization": {
            "per_workload": True,
            "fit_split": "train",
            "parameters": "scalers.json",
            "loader": "load_workload_samples(sample_dir, workload_id, standardize=True)",
            "label_scaled": False,
        },
        "total_samples": len(samples),
        "forecast_models_trained": False,
    }
    (output_dir / "feature_contract.json").write_text(
        json.dumps(feature_contract, indent=2, ensure_ascii=False) + "\n"
    )
    print(manifest.to_string(index=False))
    print(f"Exported {len(samples):,} samples with {len(FEATURE_COLUMNS)} features to {output_dir}")
    return samples, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic.csv"))
    parser.add_argument("--contract", type=Path, default=Path("data/processed/data_contract.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    prepare_features(args.traffic, args.contract, args.output_root, args.horizon)


if __name__ == "__main__":
    main()

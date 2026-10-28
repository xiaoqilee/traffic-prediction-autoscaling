"""Prepare application-level HTTP workloads from Azure Functions Trace 2019."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MINUTES_PER_DAY = 1440
TRAIN_DAYS = 8
TOTAL_DAYS = 14
MINUTE_COLUMNS = [str(i) for i in range(1, MINUTES_PER_DAY + 1)]
APP_COLUMNS = ["HashOwner", "HashApp"]
FUNCTION_COLUMNS = APP_COLUMNS + ["HashFunction"]
SOURCE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/"
    "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
DESCRIPTION_URL = (
    "https://github.com/Azure/AzurePublicDataset/blob/master/"
    "AzureFunctionsDataset2019.md"
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def daily_path(raw_dir, day):
    return raw_dir / f"invocations_per_function_md.anon.d{day:02d}.csv"


def download_and_extract(raw_dir, extract_dir):
    """Retain the compressed source and extract invocation files into a temporary directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "azurefunctions2019.tar.xz"
    if not archive_path.exists():
        print(f"Downloading {SOURCE_URL}", flush=True)
        urllib.request.urlretrieve(SOURCE_URL, archive_path)
    digest = hashlib.sha256()
    with archive_path.open("rb") as archive:
        while chunk := archive.read(1024 * 1024):
            digest.update(chunk)
    write_json(raw_dir / "source.json", {
        "dataset": "Azure Functions Trace 2019",
        "source_url": SOURCE_URL,
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "selection_days": list(range(1, TRAIN_DAYS + 1)),
        "license": "CC-BY",
    })
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "source.json", extract_dir / "source.json")
    pattern = re.compile(r"invocations_per_function_md\.anon\.d\d\d\.csv")
    with tarfile.open(archive_path, mode="r|xz") as archive:
        for member in archive:
            name = Path(member.name).name
            if member.isfile() and pattern.fullmatch(name):
                with archive.extractfile(member) as source:
                    with (extract_dir / name).open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                print(f"Extracted {name}", flush=True)


def read_day(path, chunksize, selected_apps=None):
    """Read one day in chunks; omit apps with ambiguous duplicate HTTP functions."""
    dtype = {column: "int64" for column in MINUTE_COLUMNS}
    dtype.update({column: str for column in FUNCTION_COLUMNS + ["Trigger"]})
    totals = {}
    functions = {}
    seen = set()
    duplicates = []
    duplicate_apps = set()
    rows = http_rows = raw_http_total = 0
    with pd.read_csv(path, dtype=dtype, chunksize=chunksize) as chunks:
        for chunk in chunks:
            rows += len(chunk)
            http = chunk.loc[chunk["Trigger"].eq("http")]
            if selected_apps is not None:
                keep = [tuple(key) in selected_apps
                        for key in http[APP_COLUMNS].itertuples(index=False, name=None)]
                http = http.loc[keep]
            http_rows += len(http)
            values = http[MINUTE_COLUMNS].to_numpy(dtype=np.int64)
            if (values < 0).any():
                raise ValueError(f"Negative HTTP invocation counts in {path.name}")
            raw_http_total += int(values.sum())
            for owner, app, function in http[FUNCTION_COLUMNS].itertuples(index=False, name=None):
                key = (owner, app, function)
                if key in seen:
                    duplicates.append(key)
                    duplicate_apps.add((owner, app))
                seen.add(key)
                functions.setdefault((owner, app), set()).add(function)
            grouped = http.groupby(APP_COLUMNS, sort=False)[MINUTE_COLUMNS].sum()
            for key, values in zip(grouped.index, grouped.to_numpy(dtype=np.int64)):
                if key in totals:
                    totals[key] += values
                else:
                    totals[key] = values.copy()
    excluded_total = sum(int(totals.pop(key).sum()) for key in duplicate_apps)
    for key in duplicate_apps:
        functions.pop(key)
    aggregated_total = sum(int(values.sum()) for values in totals.values())
    if aggregated_total != raw_http_total - excluded_total:
        raise ValueError(f"Aggregation changed invocation totals in {path.name}")
    audit = {
        "file": path.name,
        "source_rows": rows,
        "http_rows_in_scope": http_rows,
        "unambiguous_http_apps": len(totals),
        "duplicate_function_keys": [list(key) for key in sorted(set(duplicates))],
        "excluded_ambiguous_apps": [list(key) for key in sorted(duplicate_apps)],
        "input_http_total": raw_http_total,
        "excluded_ambiguous_total": excluded_total,
        "aggregated_http_total": aggregated_total,
        "conservation_passed": True,
    }
    return totals, functions, audit


def summarize(values):
    mean = float(values.mean())
    std = float(values.std())
    p50, p95, p99 = np.quantile(values, [0.5, 0.95, 0.99])
    jumps = np.maximum(np.diff(values), 0)
    jump_p99 = float(np.quantile(jumps, 0.99))
    jump_threshold = max(1.0, 0.5 * p50, jump_p99)
    rises = np.flatnonzero((jumps >= jump_threshold) & (values[1:] >= p95)) + 1
    # Count starts of clusters separated by at least five minutes.
    events = int(1 + np.count_nonzero(np.diff(rises) >= 5)) if len(rises) else 0
    return {
        "train_minutes": len(values),
        "train_total": int(values.sum()),
        "train_mean": mean,
        "train_std": std,
        "train_active_fraction": float(np.mean(values > 0)),
        "train_zero_fraction": float(np.mean(values == 0)),
        "train_p50": float(p50),
        "train_p95": float(p95),
        "train_p99": float(p99),
        "train_max": int(values.max()),
        "train_cv": std / mean if mean else np.nan,
        "train_p99_over_p50": float(p99 / p50) if p50 else np.nan,
        "train_positive_jump_p99": jump_p99,
        "train_relative_jump_p99": jump_p99 / max(mean, 1.0),
        "train_rise_threshold": float(jump_threshold),
        "train_rise_events": events,
    }


def rank_and_select(candidates):
    eligible = candidates.loc[
        candidates["train_active_fraction"].ge(0.5)
        & candidates["train_p50"].gt(0)
    ].copy()
    rank_columns = ["train_cv", "train_p99_over_p50", "train_relative_jump_p99"]
    eligible["burst_score"] = eligible[rank_columns].rank(pct=True).mean(axis=1)
    # Pick representative low/high scores rather than the two most extreme apps.
    low, high = eligible["burst_score"].quantile([0.25, 0.75])
    targets = eligible["burst_score"].quantile([0.1, 0.9])
    selected = []
    for pattern, mask, target in [
        ("steady", eligible["burst_score"].le(low), targets.loc[0.1]),
        ("bursty", eligible["burst_score"].ge(high)
         & eligible["train_rise_events"].ge(5), targets.loc[0.9]),
    ]:
        pool = eligible.loc[mask].copy()
        pool["distance"] = (pool["burst_score"] - target).abs()
        pool = pool.sort_values(["distance", "HashOwner", "HashApp"])
        if pool.empty:
            raise ValueError(f"No suitable {pattern} applications in the training period")
        for number, (_, row) in enumerate(pool.head(2).iterrows(), start=1):
            record = row.drop("distance").to_dict()
            record.update({
                "workload_id": f"{pattern}_{number:02d}",
                "pattern": pattern,
                "selection_reason": (
                    f"Training-only {pattern} pool; score "
                    f"{'<=' if pattern == 'steady' else '>='} "
                    f"{low if pattern == 'steady' else high:.6f}; "
                    f"nearest training score p{'10' if pattern == 'steady' else '90'} "
                    f"({target:.6f}); "
                    + ("at least 5 separated rises" if pattern == "bursty" else "low variability")
                ),
            })
            selected.append(record)
    manifest = pd.DataFrame(selected)
    rules = {
        "selection_days": list(range(1, TRAIN_DAYS + 1)),
        "required_training_days_present": TRAIN_DAYS,
        "minimum_active_fraction": 0.5,
        "minimum_median_exclusive": 0,
        "burst_score": "mean percentile rank of CV, p99/p50, positive_jump_p99/max(mean,1)",
        "steady_score_max": float(low),
        "bursty_score_min": float(high),
        "steady_target_score": float(targets.loc[0.1]),
        "bursty_target_score": float(targets.loc[0.9]),
        "rise_definition": "delta >= max(1, 0.5*p50, positive_jump_p99) and count >= p95",
        "rise_separation_minutes": 5,
        "bursty_minimum_rise_events": 5,
        "tie_break": "HashOwner, HashApp ascending",
        "eligible_apps": len(eligible),
        "selected_apps": len(manifest),
        "held_out_values_used_for_selection": False,
    }
    return manifest, rules


def plot_workloads(series, manifest, path):
    fig, axes = plt.subplots(len(manifest), 2, figsize=(15, 3 * len(manifest)), squeeze=False)
    for row_number, row in enumerate(manifest.itertuples(index=False)):
        counts = series[(row.HashOwner, row.HashApp)][:TRAIN_DAYS * MINUTES_PER_DAY]
        full, peak = axes[row_number]
        full.plot(np.arange(len(counts)) / MINUTES_PER_DAY + 1, counts, linewidth=0.5)
        full.set(title=f"{row.workload_id}: mean={row.train_mean:.1f}, CV={row.train_cv:.2f}",
                 xlabel="Relative day (training only)", ylabel="HTTP invocations / minute")
        full.set_xlim(1, TRAIN_DAYS + 1)
        center = int(np.argmax(counts))
        left = max(0, min(center - 60, len(counts) - 121))
        right = min(len(counts), left + 121)
        peak.plot(np.arange(left, right), counts[left:right], linewidth=1)
        peak.set(title=f"{row.workload_id}: 2-hour window around training maximum",
                 xlabel="Relative minute", ylabel="HTTP invocations / minute")
        for axis in (full, peak):
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(raw_dir, output_dir, figure_dir, chunksize=1024):
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    daily = {}
    functions = {}
    audits = []
    for day in range(1, TRAIN_DAYS + 1):
        totals, day_functions, audit = read_day(daily_path(raw_dir, day), chunksize)
        daily[day] = totals
        for key, ids in day_functions.items():
            functions.setdefault(key, set()).update(ids)
        audits.append({"day": day, "scope": "all_http", **audit})
        print(f"Training day {day:02d}: {len(totals)} unambiguous HTTP apps", flush=True)

    app_union = set.union(*(set(totals) for totals in daily.values()))
    app_union.update(tuple(key) for audit in audits for key in audit["excluded_ambiguous_apps"])
    common_apps = set.intersection(*(set(totals) for totals in daily.values()))
    records = []
    for owner, app in sorted(common_apps):
        values = np.concatenate([daily[day][(owner, app)] for day in daily])
        records.append({"HashOwner": owner, "HashApp": app, **summarize(values)})
    manifest, rules = rank_and_select(pd.DataFrame(records))
    manifest["trigger_filter"] = "http"
    manifest["aggregation"] = "sum HTTP function counts within (HashOwner, HashApp), per minute"
    manifest["train_function_count"] = [
        len(functions[key]) for key in manifest[APP_COLUMNS].itertuples(index=False, name=None)
    ]
    manifest["train_function_ids"] = [
        json.dumps(sorted(functions[key]))
        for key in manifest[APP_COLUMNS].itertuples(index=False, name=None)
    ]
    # Freeze training-based selection before reading any held-out counts.
    manifest.to_csv(output_dir / "workload_manifest.csv", index=False)
    write_json(output_dir / "selection_rules.json", rules)
    selected_apps = set(manifest[APP_COLUMNS].itertuples(index=False, name=None))
    series = {key: [daily[day][key] for day in daily] for key in selected_apps}
    del daily

    for day in range(TRAIN_DAYS + 1, TOTAL_DAYS + 1):
        totals, _, audit = read_day(daily_path(raw_dir, day), chunksize, selected_apps)
        missing = selected_apps - set(totals)
        audit["missing_selected_apps"] = [list(key) for key in sorted(missing)]
        audits.append({"day": day, "scope": "selected_http_only", **audit})
        for key in selected_apps:
            # Preserve absence as missing, never reinterpret it as zero traffic.
            series[key].append(totals.get(key, np.full(MINUTES_PER_DAY, np.nan)))
        print(f"Held-out day {day:02d}: extracted {len(totals)} selected apps", flush=True)
    series = {key: np.concatenate(values) for key, values in series.items()}

    minute = np.arange(TOTAL_DAYS * MINUTES_PER_DAY)
    splits = np.where(minute < TRAIN_DAYS * MINUTES_PER_DAY, "train",
                      np.where(minute < 11 * MINUTES_PER_DAY, "validation", "test"))
    traffic = pd.concat([
        pd.DataFrame({"workload_id": row.workload_id, "minute": minute,
                      "request_count": pd.array(series[(row.HashOwner, row.HashApp)], dtype="Int64"),
                      "split": splits})
        for row in manifest.itertuples(index=False)
    ], ignore_index=True)
    if traffic.duplicated(["workload_id", "minute"]).any():
        raise ValueError("Duplicate workload/minute output keys")
    missing_minutes = int(traffic["request_count"].isna().sum())
    selected_source_total = int(manifest["train_total"].sum()) + sum(
        audit["aggregated_http_total"] for audit in audits if audit["day"] > TRAIN_DAYS
    )
    if int(traffic["request_count"].sum()) != selected_source_total:
        raise ValueError("Export changed selected workloads' source invocation totals")
    traffic.to_csv(output_dir / "traffic.csv", index=False)
    source_path = raw_dir / "source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {
        "dataset": "Azure Functions Trace 2019", "source_url": SOURCE_URL,
        "archive_sha256": None, "license": "CC-BY",
    }
    contract = {
        "dataset": source,
        "description_url": DESCRIPTION_URL,
        "traffic_columns": {
            "workload_id": "Stable alias mapped to application hashes in workload_manifest.csv",
            "minute": "Zero-based relative minute; interval [60*m, 60*(m+1)) seconds",
            "request_count": "Sum of observed HTTP invocation counts; arrival-intensity proxy",
            "split": "train, validation, or test, assigned by the observed minute",
        },
        "source_field_mapping": {
            "application_key": APP_COLUMNS,
            "function_key": FUNCTION_COLUMNS,
            "filter": "Trigger == http",
            "minute": "(day - 1) * 1440 + (minute_column - 1)",
            "request_count": "sum source minute columns over HTTP functions in each application",
        },
        "split_minutes": {"train": [0, 11520], "validation": [11520, 15840], "test": [15840, 20160]},
        "split_end_is_exclusive": True,
        "timezone": None,
        "missing_counts": "Empty CSV fields mean unobserved, not zero; no imputation",
        "duplicate_policy": "Exclude an application's day if an HTTP function key repeats",
        "arrival_semantics": "Source counts are recorded after execution; exact arrivals are unavailable",
        "selection_uses_training_only": True,
        "forecast_convention_for_later_work": {
            "prediction_time": "Minute boundary t; latest complete count is y[t-1]",
            "initial_startup_delay_minutes": 3,
            "target_minute": "t + startup_delay_minutes",
        },
    }
    write_json(output_dir / "data_contract.json", contract)
    quality = {
        "training_http_apps_seen": len(app_union),
        "training_apps_complete_and_unambiguous": len(common_apps),
        "training_apps_excluded_before_activity_filter": len(app_union - common_apps),
        "eligible_training_apps": rules["eligible_apps"],
        "selected_workloads": manifest["workload_id"].tolist(),
        "expected_minutes_per_workload": len(minute),
        "output_rows": len(traffic),
        "missing_output_minutes": missing_minutes,
        "unique_minute_keys": True,
        "continuous_minute_index": all(
            np.array_equal(group["minute"].to_numpy(), minute)
            for _, group in traffic.groupby("workload_id", sort=False)
        ),
        "nonnegative_observed_counts": bool(traffic["request_count"].dropna().ge(0).all()),
        "all_daily_conservation_checks_passed": all(a["conservation_passed"] for a in audits),
        "selected_source_total": selected_source_total,
        "exported_count_total": int(traffic["request_count"].sum()),
        "export_conservation_passed": True,
        "daily_audit": audits,
    }
    write_json(output_dir / "data_quality.json", quality)
    figure_path = figure_dir / "workloads_train.png"
    plot_workloads(series, manifest, figure_path)
    print(f"Exported {len(traffic)} rows; missing minutes: {missing_minutes}", flush=True)
    return manifest, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/azure2019"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--chunksize", type=int, default=1024)
    parser.add_argument("--download", action="store_true", help="Download/extract official source if needed")
    args = parser.parse_args()
    source_files_present = all(daily_path(args.raw_dir, day).exists() for day in range(1, TOTAL_DAYS + 1))
    cached_archive = (args.raw_dir / "azurefunctions2019.tar.xz").exists()
    if args.download or (cached_archive and not source_files_present):
        with TemporaryDirectory(prefix="traffic-azure2019-") as directory:
            source_dir = Path(directory)
            download_and_extract(args.raw_dir, source_dir)
            prepare(source_dir, args.output_dir, args.figure_dir, args.chunksize)
    else:
        prepare(args.raw_dir, args.output_dir, args.figure_dir, args.chunksize)


if __name__ == "__main__":
    main()

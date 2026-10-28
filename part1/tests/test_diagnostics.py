"""Checks for high-flow denominators, consecutive errors, and handoff alignment."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.diagnostics import (
    compute_diagnostics,
    join_validation_forecasts,
    recover_validation_traffic,
    run_diagnostics,
    underprediction_runs,
)
from src.forecast import FORECAST_COLUMNS


def joined_rows(actual, predicted, minutes=None):
    size = len(actual)
    return pd.DataFrame({
        "workload_id": "app", "model": "linear",
        "target_minute": np.arange(size) if minutes is None else minutes,
        "actual_count": actual, "predicted_count": predicted,
    })


class DiagnosticTests(unittest.TestCase):
    def test_high_flow_fraction_uses_strict_training_threshold_and_request_volume(self):
        joined = joined_rows([10, 20, 20, 0, 40, 10], [10, 5, 25, 0, 20, 15])
        training = pd.DataFrame([{"workload_id": "app", "pattern": "bursty", "train_p95": 20}])
        metrics, runs = compute_diagnostics(joined, training)
        row = metrics.iloc[0]
        self.assertEqual(row.train_q95, 20)
        self.assertEqual(row.high_flow_minutes, 1)
        self.assertEqual(row.high_flow_actual_count, 40)
        self.assertEqual(row.high_flow_shortfall_count, 20)
        self.assertEqual(row.high_flow_underprediction_fraction, 0.5)
        self.assertEqual(row.underprediction_minutes, 2)
        self.assertEqual(len(runs), 2)

    def test_no_high_flow_is_missing_and_no_underprediction_has_zero_runs(self):
        joined = joined_rows([2, 2, 2], [2, 3, 4])
        training = pd.DataFrame([{"workload_id": "app", "pattern": "steady", "train_p95": 2}])
        metrics, runs = compute_diagnostics(joined, training)
        row = metrics.iloc[0]
        self.assertEqual(row.high_flow_minutes, 0)
        self.assertTrue(pd.isna(row.high_flow_underprediction_fraction))
        self.assertEqual(row.mean_run_length, 0)
        self.assertEqual(row.max_run_length, 0)
        self.assertTrue(runs.empty)

    def test_underprediction_runs_break_on_gaps_equality_and_overprediction(self):
        rows = joined_rows([5] * 9, [3, 3, 3, 5, 4, 7, 1, 2, 5],
                           minutes=[1, 2, 4, 5, 6, 7, 8, 9, 10])
        runs = underprediction_runs(rows, threshold=4)
        self.assertEqual(list(zip(runs.start_minute, runs.end_minute)), [(1, 2), (4, 4), (6, 6), (8, 9)])
        self.assertEqual(runs.length_minutes.tolist(), [2, 1, 1, 2])
        self.assertEqual(runs.shortfall_count.tolist(), [4, 2, 1, 7])
        self.assertEqual(runs.high_flow_minutes.tolist(), [2, 1, 1, 2])

    def test_recover_validation_includes_initial_minutes_without_test_data(self):
        origins = np.arange(100, 107)
        validation = pd.DataFrame({
            "workload_id": "app", "prediction_time": origins, "target_minute": origins + 3,
            "label": (origins + 3) % 7 + 1, "lag_1": (origins - 1) % 7 + 1,
        })
        counts = recover_validation_traffic(validation, [100, 110])
        np.testing.assert_array_equal(counts.minute, np.arange(100, 110))
        np.testing.assert_array_equal(counts.request_count, np.arange(100, 110) % 7 + 1)
        self.assertTrue(counts.split.eq("validation").all())
        validation.loc[0, "label"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            recover_validation_traffic(validation, [100, 110])

    def test_methods_with_different_target_windows_are_rejected(self):
        validation = pd.DataFrame({
            "workload_id": "app", "prediction_time": [100, 101], "target_minute": [103, 104], "label": [5, 7],
        })
        forecasts = pd.concat([
            validation.assign(model="linear", predicted_count=5)[FORECAST_COLUMNS],
            validation.iloc[:1].assign(model="tree", predicted_count=6)[FORECAST_COLUMNS],
        ], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "share the validation window"):
            join_validation_forecasts(validation, forecasts)

    def test_full_diagnostics_and_handoff_need_no_test_or_raw_traffic_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir, output = root / "samples", root / "outputs"
            sample_dir.mkdir()
            (output / "forecasts").mkdir(parents=True)
            origin = np.arange(11520, 11537)
            validation_parts, forecast_parts, training = [], [], []
            for workload_id, pattern in [("steady_01", "steady"), ("bursty_01", "bursty")]:
                def actual(minutes):
                    return 4 + minutes % 2 if pattern == "steady" else 10 + 5 * (minutes % 10)

                frame = pd.DataFrame({
                    "workload_id": workload_id, "prediction_time": origin, "target_minute": origin + 3,
                    "label": actual(origin + 3), "lag_1": actual(origin - 1),
                })
                validation_parts.append(frame)
                for model, adjustment in [("persistence", 0), ("linear", 1), ("tree", 2)]:
                    forecast_parts.append(frame.assign(model=model, predicted_count=frame.lag_1 + adjustment)[FORECAST_COLUMNS])
                training.append({
                    "workload_id": workload_id, "pattern": pattern, "train_minutes": 11520,
                    "train_mean": 4.5 if pattern == "steady" else 32.5,
                    "train_zero_fraction": 0, "train_p95": 4 if pattern == "steady" else 45,
                })
            pd.concat(validation_parts).to_csv(sample_dir / "validation.csv", index=False)
            pd.concat(forecast_parts).to_csv(output / "forecasts/validation.csv", index=False)
            workload_path = root / "workloads.csv"
            pd.DataFrame(training).to_csv(workload_path, index=False)
            (sample_dir / "feature_contract.json").write_text(json.dumps({
                "horizon_minutes": 3, "split_minutes": {"validation": [11520, 11540]},
            }))
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"sample_dir": str(sample_dir), "workload_manifest": str(workload_path)}))
            with patch("src.diagnostics.plot_validation"), patch("src.diagnostics.plot_case"), \
                    contextlib.redirect_stdout(io.StringIO()):
                metrics, _, shifts, _, manifest = run_diagnostics(config_path, output)
            self.assertEqual(len(metrics), 6)
            self.assertEqual(shifts.validation_minutes.tolist(), [20, 20])
            self.assertFalse(manifest["test_used"])
            self.assertFalse(manifest["models_refitted"])
            traffic = pd.read_csv(output / "examples/traffic.csv")
            forecasts = pd.read_csv(output / "examples/forecasts.csv")
            persistence = forecasts.loc[forecasts.model.eq("persistence")]
            self.assertEqual(len(traffic), 40)
            self.assertEqual(len(persistence), 34)
            self.assertEqual(persistence.columns.tolist(), FORECAST_COLUMNS)
            self.assertFalse(traffic.duplicated(["workload_id", "minute"]).any())
            matched = persistence.merge(traffic, left_on=["workload_id", "target_minute"],
                                        right_on=["workload_id", "minute"], validate="one_to_one")
            self.assertEqual(len(matched), len(persistence))
            self.assertTrue((persistence.target_minute - persistence.prediction_time).eq(3).all())
            self.assertFalse((sample_dir / "test.csv").exists())


if __name__ == "__main__":
    unittest.main()

"""Temporal alignment and leakage checks for forecasting samples."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    SPLIT_MINUTES,
    build_samples,
    load_workload_samples,
    prepare_features,
    split_workload_samples,
    standardize_splits,
)


def traffic_frame(counts, workload_id="app"):
    minute = np.arange(len(counts))
    return pd.DataFrame({
        "workload_id": workload_id,
        "minute": minute,
        "request_count": counts,
        "split": np.where(minute < 11520, "train",
                          np.where(minute < 15840, "validation", "test")),
    })


class FeatureTests(unittest.TestCase):
    def test_exact_history_features_and_future_label(self):
        counts = np.arange(6000) * 2 + 7
        samples = build_samples(traffic_frame(counts))
        row = samples.set_index("prediction_time").loc[2000]
        self.assertEqual(len(FEATURE_COLUMNS), 16)
        self.assertEqual(row.target_minute, 2003)
        self.assertEqual(row.label, 4013)
        self.assertEqual(row.lag_1, 4005)
        self.assertEqual(row.lag_2, 4003)
        self.assertEqual(row.lag_1440, 1127)
        self.assertEqual(row.rolling_mean_5, 4001)
        self.assertAlmostEqual(row.rolling_std_15, np.std(counts[1985:2000], ddof=0))
        self.assertEqual(row.diff_1, 2)
        indexed = samples.set_index("target_minute")
        self.assertAlmostEqual(indexed.loc[2880, "target_day_sin"], 0)
        self.assertAlmostEqual(indexed.loc[2880, "target_day_cos"], 1)
        self.assertAlmostEqual(indexed.loc[3240, "target_day_sin"], 1)
        self.assertAlmostEqual(indexed.loc[3240, "target_day_cos"], 0)

    def test_future_counts_change_labels_but_not_available_features(self):
        original = traffic_frame(np.arange(6000) % 100)
        changed = original.copy()
        changed.loc[changed.minute.ge(2000), "request_count"] += 100000
        before = build_samples(original).set_index("prediction_time")
        after = build_samples(changed).set_index("prediction_time")
        pd.testing.assert_frame_equal(
            before.loc[:2000, FEATURE_COLUMNS], after.loc[:2000, FEATURE_COLUMNS]
        )
        self.assertEqual(after.loc[2000, "label"] - before.loc[2000, "label"], 100000)

    def test_split_boundaries_sample_counts_and_observed_validation_history(self):
        counts = np.arange(20160)
        for horizon in (0, 3, 5):
            with self.subTest(horizon=horizon):
                samples = build_samples(traffic_frame(counts), horizon=horizon)
                splits = split_workload_samples(samples, "app")
                expected = {"train": 10080 - horizon,
                            "validation": 4320 - horizon, "test": 4320 - horizon}
                for name, frame in splits.items():
                    start, end = SPLIT_MINUTES[name]
                    self.assertEqual(len(frame), expected[name])
                    self.assertTrue(frame.prediction_time.ge(start).all())
                    self.assertTrue(frame.target_minute.lt(end).all())
                    self.assertTrue((frame.target_minute - frame.prediction_time).eq(horizon).all())
                    self.assertEqual(frame.iloc[-1].target_minute, end - 1)
                self.assertEqual(splits["train"].iloc[0].prediction_time, 1440)
                self.assertEqual(splits["validation"].iloc[0].target_minute, 11520 + horizon)
                self.assertEqual(splits["test"].iloc[0].target_minute, 15840 + horizon)
                validation = splits["validation"].set_index("prediction_time")
                self.assertEqual(validation.loc[11520, "lag_1"], 11519)
                self.assertEqual(validation.loc[11521, "lag_1"], 11520)
                self.assertEqual(validation.loc[11520, "rolling_mean_5"], 11517)

    def test_application_histories_stay_separate_when_input_is_shuffled(self):
        traffic = pd.concat([
            traffic_frame(np.full(6000, 7), "low"),
            traffic_frame(np.full(6000, 1000), "high"),
        ], ignore_index=True).sample(frac=1, random_state=42)
        samples = build_samples(traffic)
        for workload_id, expected in [("low", 7), ("high", 1000)]:
            rows = samples.loc[samples.workload_id.eq(workload_id)]
            self.assertTrue(rows.lag_1.eq(expected).all())
            self.assertTrue(rows.lag_1440.eq(expected).all())
            self.assertTrue(rows.rolling_mean_60.eq(expected).all())
            self.assertTrue(rows.rolling_std_60.eq(0).all())
            self.assertTrue(rows.label.eq(expected).all())

    def test_missing_minutes_are_not_zero_filled_or_skipped_by_row_shifts(self):
        traffic = traffic_frame(np.arange(6000) + 1)
        traffic = traffic.loc[traffic.minute.ne(2000)]
        samples = build_samples(traffic).set_index("prediction_time")
        self.assertNotIn(1997, samples.index)  # Missing target y[2000].
        self.assertNotIn(2001, samples.index)  # Missing latest observation.
        self.assertNotIn(2060, samples.index)  # Missing value in 60-minute window.
        self.assertNotIn(3440, samples.index)  # Missing 1440-minute lag.
        self.assertIn(2000, samples.index)     # y[2000] is not yet an input.
        self.assertIn(2061, samples.index)
        self.assertEqual(samples.loc[2061, "lag_1"], 2061)

    def test_scaler_uses_only_training_and_preserves_labels_and_times(self):
        original = traffic_frame(np.arange(20160) % 20 + 1)
        changed = original.copy()
        changed.loc[changed.minute.ge(11520), "request_count"] += 100000
        before = split_workload_samples(build_samples(original), "app")
        after = split_workload_samples(build_samples(changed), "app")
        _, original_scaler = standardize_splits(before)
        scaled, changed_scaler = standardize_splits(after)
        np.testing.assert_array_equal(original_scaler.mean_, changed_scaler.mean_)
        np.testing.assert_array_equal(original_scaler.scale_, changed_scaler.scale_)
        self.assertEqual(changed_scaler.n_samples_seen_, 10077)
        np.testing.assert_allclose(scaled["train"][FEATURE_COLUMNS].mean(), 0, atol=1e-10)
        self.assertGreater(scaled["test"]["lag_1"].mean(), 1000)
        for name in after:
            pd.testing.assert_frame_equal(
                after[name].drop(columns=FEATURE_COLUMNS),
                scaled[name].drop(columns=FEATURE_COLUMNS),
            )

    def test_export_and_loader_reuse_training_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traffic_path, contract_path = root / "traffic.csv", root / "contract.json"
            traffic_frame(np.arange(6000) % 50 + 1).to_csv(traffic_path, index=False)
            contract_path.write_text(json.dumps({"split_minutes": {
                "train": [0, 3000], "validation": [3000, 4500], "test": [4500, 6000],
            }}))
            with contextlib.redirect_stdout(io.StringIO()):
                samples, manifest = prepare_features(
                    traffic_path, contract_path, root / "features"
                )
            sample_dir = root / "features/horizon_3"
            raw, raw_scaler = load_workload_samples(sample_dir, "app")
            scaled, scaler = load_workload_samples(sample_dir, "app", standardize=True)
            self.assertIsNone(raw_scaler)
            self.assertEqual(sum(len(frame) for frame in raw.values()), len(samples))
            self.assertEqual(manifest.sample_count.tolist(), [1557, 1497, 1497])
            saved = json.loads((sample_dir / "scalers.json").read_text())["app"]
            self.assertEqual(saved["training_samples"], 1557)
            np.testing.assert_allclose(saved["mean"], scaler.mean_)
            np.testing.assert_allclose(saved["scale"], scaler.scale_)
            self.assertTrue(scaled["validation"].target_minute.equals(raw["validation"].target_minute))
            self.assertTrue(np.isfinite(scaled["test"][FEATURE_COLUMNS]).all().all())


if __name__ == "__main__":
    unittest.main()

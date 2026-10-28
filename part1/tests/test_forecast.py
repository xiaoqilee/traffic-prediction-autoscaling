"""Checks for baseline alignment, training-only fitting, and frozen evaluation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from src.features import FEATURE_COLUMNS
from src.forecast import FORECAST_COLUMNS, evaluate_saved, train_validation, train_workload


def small_config():
    return {
        "random_seed": 42,
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
        "tree": {"loss": "squared_error", "max_iter": 3, "learning_rate": 0.1,
                 "max_leaf_nodes": [3, 5], "min_samples_leaf": 2, "early_stopping": False},
    }


def sample_frame(size, split, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(1, 10, (size, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["label"] = np.rint(3 + 2 * frame["lag_1"] + frame["rolling_mean_5"])
    start = {"train": 2000, "validation": 11520, "test": 15840}[split]
    frame["workload_id"] = "bursty_fixture"
    frame["prediction_time"] = np.arange(start, start + size)
    frame["target_minute"] = frame["prediction_time"] + 3
    frame["split"] = split
    return frame


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.threads = threadpool_limits(limits=1)
        self.threads.__enter__()
        self.addCleanup(self.threads.__exit__, None, None, None)
        self.train = sample_frame(80, "train", 1)
        self.validation = sample_frame(24, "validation", 2)

    def test_persistence_alignment_common_targets_and_validation_selection(self):
        forecasts, metrics, candidates, models, selected = train_workload(
            self.train, self.validation, small_config()
        )
        self.assertEqual(forecasts.columns.tolist(), FORECAST_COLUMNS)
        self.assertEqual(len(forecasts), 3 * len(self.validation))
        persistence = forecasts.loc[forecasts.model.eq("persistence")]
        np.testing.assert_array_equal(persistence.predicted_count, self.validation.lag_1)
        expected_mae = np.abs(self.validation.label - self.validation.lag_1).mean()
        self.assertAlmostEqual(metrics.set_index("model").loc["persistence", "mae"], expected_mae)
        for _, frame in forecasts.groupby("model"):
            np.testing.assert_array_equal(frame.target_minute, self.validation.target_minute)
            self.assertTrue(frame.predicted_count.ge(0).all())
        for name in ("linear", "tree"):
            search = candidates.loc[candidates.model.eq(name)]
            self.assertEqual(int(search.selected.sum()), 1)
            self.assertEqual(selected[name]["validation_scores"]["mae"], search.mae.min())
            chosen = json.loads(search.loc[search.selected, "parameters"].iloc[0])
            self.assertEqual(chosen, selected[name]["parameters"])
        self.assertFalse(models["tree"].early_stopping)
        self.assertEqual(models["tree"].n_iter_, 3)

    def test_validation_changes_do_not_change_fitted_candidate_parameters(self):
        config = small_config()
        config["ridge"]["alpha"] = [1.0]
        config["tree"]["max_leaf_nodes"] = [3]
        _, _, _, original_models, _ = train_workload(self.train, self.validation, config)
        changed = self.validation.copy()
        changed[FEATURE_COLUMNS] += 100000
        changed["label"] += 1000000
        _, _, _, changed_models, _ = train_workload(self.train, changed, config)
        scaler = changed_models["linear"].named_steps["standardscaler"]
        self.assertEqual(scaler.n_samples_seen_, len(self.train))
        np.testing.assert_allclose(scaler.mean_, self.train[FEATURE_COLUMNS].mean())
        np.testing.assert_array_equal(
            original_models["linear"].named_steps["ridge"].coef_,
            changed_models["linear"].named_steps["ridge"].coef_,
        )
        np.testing.assert_array_equal(
            original_models["tree"].predict(self.train[FEATURE_COLUMNS]),
            changed_models["tree"].predict(self.train[FEATURE_COLUMNS]),
        )

    def test_nonnegative_clipping_precedes_validation_scoring(self):
        validation = self.validation.copy()
        validation["label"] = 0
        with patch.object(Ridge, "predict", side_effect=lambda X: np.full(len(X), -5.0)):
            forecasts, metrics, _, _, _ = train_workload(self.train, validation, small_config())
        self.assertTrue(forecasts.loc[forecasts.model.eq("linear"), "predicted_count"].eq(0).all())
        self.assertEqual(metrics.set_index("model").loc["linear", "mae"], 0)
        self.assertEqual(metrics.set_index("model").loc["linear", "rmse"], 0)

    def test_default_run_needs_no_test_file_and_saved_test_evaluation_never_refits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir, output = root / "samples", root / "outputs"
            sample_dir.mkdir()
            self.train.to_csv(sample_dir / "train.csv", index=False)
            self.validation.to_csv(sample_dir / "validation.csv", index=False)
            (sample_dir / "feature_contract.json").write_text(json.dumps({
                "horizon_minutes": 3, "feature_columns": FEATURE_COLUMNS,
            }))
            manifest_path = root / "workloads.csv"
            pd.DataFrame([{"workload_id": "bursty_fixture", "pattern": "bursty"}]).to_csv(
                manifest_path, index=False
            )
            config = {**small_config(), "sample_dir": str(sample_dir),
                      "workload_manifest": str(manifest_path)}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                forecasts, _, _, manifest = train_validation(config_path, output)
            self.assertFalse((sample_dir / "test.csv").exists())
            self.assertFalse((output / "forecasts/test.csv").exists())
            self.assertFalse(manifest["test_used"])
            selected = manifest["workloads"]["bursty_fixture"]["models"]
            for name in ("linear", "tree"):
                estimator = joblib.load(output / "models/horizon_3" / selected[name]["model_file"])
                expected = np.maximum(estimator.predict(self.validation[FEATURE_COLUMNS]), 0)
                np.testing.assert_allclose(
                    forecasts.loc[forecasts.model.eq(name), "predicted_count"], expected
                )

            test = sample_frame(18, "test", 3)
            test.to_csv(sample_dir / "test.csv", index=False)
            (sample_dir / "train.csv").unlink()
            (sample_dir / "validation.csv").unlink()
            with patch.object(Ridge, "fit", side_effect=AssertionError("test refit")), \
                    patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("test refit")), \
                    contextlib.redirect_stdout(io.StringIO()):
                test_forecasts, test_metrics = evaluate_saved(config_path, output)
            self.assertEqual(len(test_forecasts), 3 * len(test))
            self.assertEqual(len(test_metrics), 3)
            np.testing.assert_array_equal(
                test_forecasts.loc[test_forecasts.model.eq("persistence"), "predicted_count"], test.lag_1
            )
            audit = json.loads((output / "metrics/test_run.json").read_text())
            self.assertFalse(audit["refit"])
            self.assertFalse(audit["settings_selected_on_test"])


if __name__ == "__main__":
    unittest.main()

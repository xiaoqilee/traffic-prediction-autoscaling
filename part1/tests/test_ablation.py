"""Check frozen inputs, training isolation, paired metrics and handoff forecasts."""

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

from src.ablation import run_ablation, select_comparison_cases, summarize_validation
from src.features import FEATURE_COLUMNS
from src.forecast import evaluate_saved, file_hash, train_validation


def samples(workload_id, start, size, split, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(1, 10, (size, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    minute = np.arange(start, start + size)
    frame["lag_1"] = 4 + (minute - 1) % 10
    frame["workload_id"] = workload_id
    frame["prediction_time"] = minute
    frame["target_minute"] = minute + 3
    frame["split"] = split
    frame["label"] = 4 + (minute + 3) % 10
    return frame


class AblationTests(unittest.TestCase):
    def make_fixture(self, root):
        sample_dir = root / "samples"
        sample_dir.mkdir()
        train = pd.concat([samples(name, 2000, 60, "train", seed)
                           for seed, name in enumerate(("bursty_fixture", "steady_fixture"))], ignore_index=True)
        validation = pd.concat([samples(name, 11520, 24, "validation", seed + 2)
                                for seed, name in enumerate(("bursty_fixture", "steady_fixture"))], ignore_index=True)
        train.to_csv(sample_dir / "train.csv", index=False)
        validation.to_csv(sample_dir / "validation.csv", index=False)
        (sample_dir / "feature_contract.json").write_text(json.dumps({
            "horizon_minutes": 3, "feature_columns": FEATURE_COLUMNS,
            "split_minutes": {"train": [0, 11520], "validation": [11520, 11547], "test": [15840, 20160]},
        }))
        workloads = root / "workloads.csv"
        pd.DataFrame([{"workload_id": name, "pattern": pattern, "train_mean": 8.5,
                       "train_p95": 11, "train_minutes": 60, "train_zero_fraction": 0}
                      for name, pattern in (("bursty_fixture", "bursty"), ("steady_fixture", "steady"))]).to_csv(workloads, index=False)
        baseline_config = root / "baseline.json"
        baseline_config.write_text(json.dumps({
            "sample_dir": str(sample_dir), "workload_manifest": str(workloads), "random_seed": 42,
            "ridge": {"alpha": [1]},
            "tree": {"loss": "squared_error", "max_iter": 3, "learning_rate": 0.1,
                     "max_leaf_nodes": [3], "min_samples_leaf": 2, "early_stopping": False},
        }))
        baseline = root / "baseline"
        with contextlib.redirect_stdout(io.StringIO()):
            train_validation(baseline_config, baseline)
        receipt = root / "diagnostics.json"
        receipt.write_text(json.dumps({"input_sha256": {
            "forecasts": file_hash(baseline / "forecasts/validation.csv"), "workloads": file_hash(workloads),
        }}))
        experiment_config = root / "ablation.json"
        experiment_config.write_text(json.dumps({
            "baseline_config": str(baseline_config), "baseline_output_root": str(baseline),
            "baseline_diagnostics_manifest": str(receipt), "output_root": str(root / "experiment"),
            "quantiles": {"tree_q50": 0.5, "tree_underprediction": 0.75},
            "prediction_rtol": 1e-10, "prediction_atol": 1e-10,
            "failure_case": {"workload_id": "bursty_fixture", "start_minute": 11523,
                             "end_minute": 11540, "center_minute": 11529},
        }))
        return experiment_config, baseline_config, train, validation

    def test_end_to_end_only_fits_training_rows_and_preserves_baselines_without_test_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, baseline_config, train, validation = self.make_fixture(root)
            ordinary_path = root / "baseline/forecasts/validation.csv"
            original_hash = file_hash(ordinary_path)
            original_fit = HistGradientBoostingRegressor.fit
            fits = []

            def record_fit(estimator, X, y, **kwargs):
                fits.append((X.copy(), y.copy(), estimator.get_params()))
                return original_fit(estimator, X, y, **kwargs)

            original_read = pd.read_csv

            def guarded_read(path, *args, **kwargs):
                if Path(path).name == "test.csv":
                    raise AssertionError("The held-out test must not be read")
                return original_read(path, *args, **kwargs)

            with patch.object(HistGradientBoostingRegressor, "fit", new=record_fit), \
                    patch("pandas.read_csv", side_effect=guarded_read), \
                    patch("src.ablation.plot_cases"), patch("src.ablation.plot_tradeoff"), \
                    patch("src.ablation.plot_validation"), contextlib.redirect_stdout(io.StringIO()):
                forecasts, metrics, manifest = run_ablation(config)
            self.assertEqual(len(fits), 4)
            for (X, y, params), name in zip(fits, ("bursty_fixture",) * 2 + ("steady_fixture",) * 2):
                expected = train.loc[train.workload_id.eq(name)]
                np.testing.assert_allclose(X, expected[FEATURE_COLUMNS])
                np.testing.assert_array_equal(y, expected.label)
                self.assertEqual(params["loss"], "quantile")
                self.assertFalse(params["early_stopping"])
            self.assertEqual([entry[2]["quantile"] for entry in fits], [0.5, 0.75, 0.5, 0.75])
            self.assertEqual(len(forecasts), len(validation) * 5)
            self.assertEqual(len(metrics), 10)
            self.assertEqual(file_hash(ordinary_path), original_hash)
            self.assertEqual(file_hash(root / "experiment/forecasts/validation_ordinary.csv"), original_hash)
            self.assertFalse((root / "samples/test.csv").exists())
            self.assertFalse(manifest["test_used"])
            self.assertEqual(manifest["new_model_fits"], 4)
            self.assertEqual(manifest["traffic_rows"], 54)
            for _, rows in forecasts.groupby(["workload_id", "model"]):
                self.assertEqual(len(rows), 24)
                self.assertTrue((rows.target_minute - rows.prediction_time).eq(3).all())
            saved = json.loads((root / "experiment/models/horizon_3/selected_models.json").read_text())
            for name in ("bursty_fixture", "steady_fixture"):
                model_dir = root / "experiment/models/horizon_3"
                ordinary = joblib.load(model_dir / saved["workloads"][name]["models"]["tree"]["model_file"])
                for method in ("tree_q50", "tree_underprediction"):
                    candidate = joblib.load(model_dir / saved["workloads"][name]["models"][method]["model_file"])
                    changed = {key for key in ordinary.get_params() if ordinary.get_params()[key] != candidate.get_params()[key]}
                    self.assertEqual(changed, {"loss", "quantile"})
            with patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("Unexpected refit")), \
                    contextlib.redirect_stdout(io.StringIO()):
                reloaded, _ = evaluate_saved(baseline_config, root / "experiment", split="validation")
            keys = ["workload_id", "model", "target_minute"]
            np.testing.assert_allclose(forecasts.sort_values(keys).predicted_count,
                                       reloaded.sort_values(keys).predicted_count)

    def test_changed_frozen_inputs_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _, _ = self.make_fixture(root)
            path = root / "samples/train.csv"
            path.write_text(path.read_text() + "\n")
            with patch.object(HistGradientBoostingRegressor, "fit", side_effect=AssertionError("Unexpected fitting")):
                with self.assertRaisesRegex(ValueError, "Frozen baseline input changed"):
                    run_ablation(config)
            self.assertFalse((root / "experiment").exists())

    def test_baseline_output_directory_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _, _ = self.make_fixture(root)
            with self.assertRaisesRegex(ValueError, "must not replace"):
                run_ablation(config, root / "baseline")

    def test_success_case_uses_ordinary_peak_hit_and_matching_magnitude(self):
        minute = np.arange(1000, 1700)
        actual, predicted = np.zeros(700), np.zeros(700)
        actual[[100, 350, 600]] = [140, 143, 100]
        predicted[[100, 350, 600]] = [100, 20, 95]
        validation = pd.DataFrame({"workload_id": "app", "prediction_time": minute - 3,
                                   "target_minute": minute, "label": actual})
        ordinary = validation.drop(columns="label").assign(model="tree", predicted_count=predicted)
        workloads = pd.DataFrame([{"workload_id": "app", "train_p95": 55}])
        cases = select_comparison_cases(validation, ordinary, workloads,
                                        {"workload_id": "app", "start_minute": 1290,
                                         "end_minute": 1410, "center_minute": 1350})
        success = cases.loc[cases.case.eq("success")].iloc[0]
        self.assertEqual(success.center_minute, 1100)
        self.assertTrue(success.peak_matched)
        ordinary.loc[ordinary.target_minute.eq(1100), "predicted_count"] = 10
        fallback = select_comparison_cases(validation, ordinary, workloads,
                                           {"workload_id": "app", "start_minute": 1290,
                                            "end_minute": 1410, "center_minute": 1350})
        success = fallback.loc[fallback.case.eq("success")].iloc[0]
        self.assertEqual(success.center_minute, 1600)
        self.assertFalse(success.peak_matched)

    def test_group_scores_normalize_per_application_and_deltas_are_paired(self):
        frames, workloads = [], []
        for name, scale in (("small", 1), ("large", 100)):
            workloads.append({"workload_id": name, "pattern": "bursty", "train_mean": scale,
                              "train_p95": scale})
            for model, factor in (("tree", 0), ("tree_q50", 0.5), ("tree_underprediction", 1)):
                frames.append(pd.DataFrame({"workload_id": name, "model": model,
                                            "target_minute": [11523, 11524], "actual_count": [2 * scale] * 2,
                                            "predicted_count": [(1 + factor) * scale] * 2}))
        detail, grouped, deltas, _ = summarize_validation(pd.concat(frames, ignore_index=True), pd.DataFrame(workloads))
        row = grouped.loc[grouped.scope.eq("validation") & grouped.model.eq("tree")].iloc[0]
        self.assertEqual(row.mean_nmae, 1)
        self.assertEqual(row.mean_nrmse, 1)
        self.assertEqual(row.n_workloads, 2)
        self.assertEqual(row.n_samples, 4)
        paired = deltas.loc[deltas.scope.eq("validation") & deltas.reference.eq("tree")]
        np.testing.assert_allclose(paired.high_flow_underprediction_fraction_delta, [-0.5, -0.5])
        self.assertEqual(len(detail), 12)

    def test_absent_success_is_labeled_and_other_workload_success_is_explicit(self):
        minute = np.arange(1000, 1700)
        frames, forecasts = [], []
        for name, successful in (("failed_app", False), ("other_app", True)):
            actual, predicted = np.zeros(700), np.zeros(700)
            actual[[100, 350]] = [140, 143]
            predicted[[100, 350]] = [135, 20] if successful else [50, 20]
            frame = pd.DataFrame({"workload_id": name, "prediction_time": minute - 3,
                                  "target_minute": minute, "label": actual})
            frames.append(frame)
            forecasts.append(frame.drop(columns="label").assign(model="tree", predicted_count=predicted))
        workloads = pd.DataFrame([{"workload_id": name, "pattern": "bursty", "train_p95": 55}
                                  for name in ("failed_app", "other_app")])
        cases = select_comparison_cases(pd.concat(frames, ignore_index=True), pd.concat(forecasts, ignore_index=True),
                                        workloads, {"workload_id": "failed_app", "start_minute": 1290,
                                                    "end_minute": 1410, "center_minute": 1350})
        self.assertEqual(cases.case.tolist(), ["failure", "best_available", "success_other_workload"])
        self.assertFalse(cases.iloc[1].ordinary_peak_hit)
        self.assertEqual(cases.iloc[2].workload_id, "other_app")
        self.assertTrue(cases.iloc[2].ordinary_peak_hit)


if __name__ == "__main__":
    unittest.main()

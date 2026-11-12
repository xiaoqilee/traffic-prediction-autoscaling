"""Tests for the seasonal-naive baseline and CI helper."""

import unittest

import numpy as np
import pandas as pd

from part3.src.baselines import build_seasonal_naive_forecast, forecast_accuracy
from part3.src.run_seeds import ci95_halfwidth


def _traffic(workload, minutes, counts):
    return pd.DataFrame({
        "workload_id": workload,
        "minute": list(minutes),
        "request_count": list(counts),
        "split": "train",
    })


class SeasonalNaiveTests(unittest.TestCase):
    def test_predicts_demand_one_period_earlier(self):
        # Period of 5 minutes; target 10 should predict the demand at minute 5.
        minutes = range(0, 12)
        counts = [float(m) for m in minutes]
        traffic = _traffic("w", minutes, counts)

        fc = build_seasonal_naive_forecast(
            traffic, ["w"], {"w": [10, 11]}, horizon=3, period_minutes=5,
        )

        row10 = fc.loc[fc["target_minute"].eq(10)].iloc[0]
        self.assertEqual(row10["predicted_count"], 5.0)        # demand at 10 - 5
        self.assertEqual(row10["prediction_time"], 7)          # target - horizon
        self.assertEqual(row10["model"], "seasonal_naive")
        row11 = fc.loc[fc["target_minute"].eq(11)].iloc[0]
        self.assertEqual(row11["predicted_count"], 6.0)

    def test_forward_fills_missing_history(self):
        # Minute 2 is absent; target 7 (period 5) looks up minute 2 -> ffill from 1.
        traffic = _traffic("w", [0, 1, 3, 4, 5, 6, 7], [10, 11, 13, 14, 15, 16, 17])
        fc = build_seasonal_naive_forecast(
            traffic, ["w"], {"w": [7]}, horizon=3, period_minutes=5,
        )
        self.assertEqual(fc.iloc[0]["predicted_count"], 11.0)  # ffilled from minute 1

    def test_negative_predictions_clipped(self):
        traffic = _traffic("w", [0, 1, 2, 3, 4, 5], [-3, 0, 1, 2, 3, 4])
        fc = build_seasonal_naive_forecast(
            traffic, ["w"], {"w": [5]}, horizon=3, period_minutes=5,
        )
        self.assertGreaterEqual(fc.iloc[0]["predicted_count"], 0.0)

    def test_forecast_accuracy_matches_hand_calc(self):
        traffic = _traffic("w", range(0, 10), [0, 0, 0, 0, 0, 10, 20, 0, 0, 0])
        # Two forecasts for targets 6 and 7 against actuals 20 and 0.
        fc = pd.DataFrame({
            "workload_id": "w", "model": "seasonal_naive",
            "prediction_time": [3, 4], "target_minute": [6, 7],
            "predicted_count": [18.0, 2.0],
        })
        acc = forecast_accuracy(fc, traffic, scoring_start_minute=6, scoring_end_minute=7)
        # errors: |18-20|=2, |2-0|=2 -> MAE 2, RMSE 2
        self.assertAlmostEqual(acc.iloc[0]["mae"], 2.0)
        self.assertAlmostEqual(acc.iloc[0]["rmse"], 2.0)


class CiTests(unittest.TestCase):
    def test_zero_variance_gives_zero_ci(self):
        self.assertEqual(ci95_halfwidth([0.5, 0.5, 0.5]), 0.0)

    def test_single_sample_gives_zero_ci(self):
        self.assertEqual(ci95_halfwidth([0.5]), 0.0)

    def test_positive_ci_for_spread(self):
        self.assertGreater(ci95_halfwidth([0.1, 0.2, 0.3, 0.4]), 0.0)


if __name__ == "__main__":
    unittest.main()

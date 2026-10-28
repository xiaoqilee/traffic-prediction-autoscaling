"""Unit tests for part3.src.metrics using Part 2 example outputs as fixtures."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from part3.src.metrics import (
    compute_drop_rate,
    compute_latency_percentiles,
    compute_slo_violation_rate,
    filter_scoring_interval,
    summarize_run,
)

REPO_ROOT = Path(__file__).parents[2]
EXAMPLES = REPO_ROOT / "part2" / "outputs" / "examples"


def _make_outcomes(statuses, response_times, arrival_minutes):
    """Build a minimal request_outcomes DataFrame for testing."""
    arrival_seconds = [m * 60.0 + 30.0 for m in arrival_minutes]
    return pd.DataFrame(
        {
            "run_id": "test",
            "workload_id": "w",
            "model": "m",
            "request_id": [str(i) for i in range(len(statuses))],
            "arrival_time_seconds": arrival_seconds,
            "completion_time_seconds": [
                arrival_seconds[i] + (rt if rt is not None else 0)
                for i, rt in enumerate(response_times)
            ],
            "response_time_seconds": response_times,
            "status": statuses,
        }
    )


class TestFilterScoringInterval(unittest.TestCase):
    def test_excludes_warmup_minutes(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed", "completed"],
            response_times=[1.0, 1.0, 1.0],
            arrival_minutes=[11520, 11521, 11523],
        )
        scored = filter_scoring_interval(outcomes, 11523, 15839)
        self.assertEqual(len(scored), 1)
        self.assertAlmostEqual(
            float(scored["arrival_time_seconds"].iloc[0]), 11523 * 60.0 + 30.0
        )

    def test_includes_both_boundary_minutes(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed"],
            response_times=[1.0, 1.0],
            arrival_minutes=[11523, 15839],
        )
        scored = filter_scoring_interval(outcomes, 11523, 15839)
        self.assertEqual(len(scored), 2)

    def test_excludes_requests_after_end(self):
        outcomes = _make_outcomes(
            statuses=["completed"],
            response_times=[1.0],
            arrival_minutes=[15840],
        )
        scored = filter_scoring_interval(outcomes, 11523, 15839)
        self.assertEqual(len(scored), 0)


class TestComputeLatencyPercentiles(unittest.TestCase):
    def test_excludes_dropped_requests(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed", "dropped"],
            response_times=[1.0, 9.0, None],
            arrival_minutes=[100, 100, 100],
        )
        outcomes.loc[outcomes["status"].eq("dropped"), "response_time_seconds"] = None
        result = compute_latency_percentiles(outcomes)
        self.assertAlmostEqual(result["p50_latency"], 5.0, places=1)

    def test_returns_none_when_no_completions(self):
        outcomes = _make_outcomes(
            statuses=["dropped"],
            response_times=[None],
            arrival_minutes=[100],
        )
        outcomes["response_time_seconds"] = None
        result = compute_latency_percentiles(outcomes)
        self.assertIsNone(result["p95_latency"])

    def test_known_percentile(self):
        times = list(range(1, 101))
        outcomes = _make_outcomes(
            statuses=["completed"] * 100,
            response_times=times,
            arrival_minutes=[100] * 100,
        )
        result = compute_latency_percentiles(outcomes, percentiles=(95,))
        self.assertAlmostEqual(result["p95_latency"], np.percentile(times, 95), places=5)


class TestComputeSloViolationRate(unittest.TestCase):
    def test_includes_slow_completions_and_drops(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed", "dropped"],
            response_times=[1.0, 6.0, None],
            arrival_minutes=[100, 100, 100],
        )
        outcomes.loc[outcomes["status"].eq("dropped"), "response_time_seconds"] = None
        rate = compute_slo_violation_rate(outcomes, slo_seconds=5.0)
        # 1 slow completion + 1 drop = 2 violations out of 3
        self.assertAlmostEqual(rate, 2 / 3)

    def test_zero_when_all_fast(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed"],
            response_times=[1.0, 2.0],
            arrival_minutes=[100, 100],
        )
        rate = compute_slo_violation_rate(outcomes, slo_seconds=5.0)
        self.assertAlmostEqual(rate, 0.0)

    def test_returns_none_when_empty(self):
        outcomes = _make_outcomes([], [], [])
        rate = compute_slo_violation_rate(outcomes, slo_seconds=5.0)
        self.assertIsNone(rate)


class TestComputeDropRate(unittest.TestCase):
    def test_correct_fraction(self):
        outcomes = _make_outcomes(
            statuses=["completed", "dropped", "dropped"],
            response_times=[1.0, None, None],
            arrival_minutes=[100, 100, 100],
        )
        rate = compute_drop_rate(outcomes)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_zero_when_no_drops(self):
        outcomes = _make_outcomes(
            statuses=["completed", "completed"],
            response_times=[1.0, 1.0],
            arrival_minutes=[100, 100],
        )
        rate = compute_drop_rate(outcomes)
        self.assertAlmostEqual(rate, 0.0)


class TestSummarizeRunWithExampleData(unittest.TestCase):
    def setUp(self):
        self.outcomes = pd.read_csv(EXAMPLES / "request_outcomes.csv")
        self.logs = pd.read_csv(EXAMPLES / "minute_logs.csv")

    def test_bursty_02_has_drops(self):
        run_outcomes = self.outcomes.loc[
            self.outcomes["run_id"].eq("bursty_02_persistence")
        ]
        run_logs = self.logs.loc[
            self.logs["run_id"].eq("bursty_02_persistence")
        ]
        # Use absolute minute range from minute_logs
        start = int(run_logs["minute"].min())
        end = int(run_logs["minute"].max())

        result = summarize_run(run_outcomes, run_logs, start, end, slo_seconds=5.0)
        self.assertGreater(result["dropped"], 0)
        self.assertGreater(result["slo_violation_rate"], 0.0)

    def test_server_minutes_is_positive(self):
        run_outcomes = self.outcomes.loc[
            self.outcomes["run_id"].eq("steady_01_linear")
        ]
        run_logs = self.logs.loc[
            self.logs["run_id"].eq("steady_01_linear")
        ]
        start = int(run_logs["minute"].min())
        end = int(run_logs["minute"].max())

        result = summarize_run(run_outcomes, run_logs, start, end, slo_seconds=5.0)
        self.assertGreater(result["server_minutes"], 0)

    def test_arrivals_equals_completed_plus_dropped(self):
        run_outcomes = self.outcomes.loc[
            self.outcomes["run_id"].eq("steady_01_persistence")
        ]
        run_logs = self.logs.loc[
            self.logs["run_id"].eq("steady_01_persistence")
        ]
        start = int(run_logs["minute"].min())
        end = int(run_logs["minute"].max())

        result = summarize_run(run_outcomes, run_logs, start, end, slo_seconds=5.0)
        self.assertEqual(result["arrivals"], result["completed"] + result["dropped"])


if __name__ == "__main__":
    unittest.main()

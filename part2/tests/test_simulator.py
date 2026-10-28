import unittest
from pathlib import Path

import pandas as pd

from part2.simulator import (
    RequestQueue,
    ServerPool,
    ServicePool,
    WaitingRequest,
    generate_arrival_times,
    load_replay_inputs,
    dispatch_next_waiting_request,
    dispatch_request,
    activate_ready_servers,
    simulate_replay,
    apply_target_servers
)

class InputLoaderTests(unittest.TestCase):
    def setUp(self):
        repository_root = Path(__file__).parents[2]
        examples = repository_root / "part1" / "outputs" / "examples"

        self.traffic_path = examples / "traffic.csv"
        self.forecast_path = examples / "forecasts.csv"

    def test_example_files_load(self):
        traffic, forecasts = load_replay_inputs(
            self.traffic_path,
            self.forecast_path,
        )

        self.assertEqual(len(traffic), 248)
        self.assertEqual(len(forecasts), 726)

        self.assertEqual(
            set(traffic["workload_id"]),
            {"steady_01", "bursty_02"},
        )

        self.assertEqual(
            set(forecasts["model"]),
            {"persistence", "linear", "tree"},
        )

    def test_forecast_horizon_is_three_minutes(self):
        _, forecasts = load_replay_inputs(
            self.traffic_path,
            self.forecast_path,
        )

        horizon = forecasts["target_minute"] - forecasts["prediction_time"]
        self.assertTrue(horizon.eq(3).all())

    def test_arrivals_preserve_request_counts_and_are_reproducible(self):
        traffic, _ = load_replay_inputs(
            self.traffic_path,
            self.forecast_path,
        )

        first = generate_arrival_times(traffic, seed=42)
        second = generate_arrival_times(traffic, seed=42)

        expected_count = int(traffic["request_count"].sum())

        self.assertEqual(len(first), expected_count)
        self.assertEqual(first["request_id"].nunique(), expected_count)

        pd.testing.assert_frame_equal(first, second)

    def test_arrivals_stay_inside_their_minute(self):
        traffic, _ = load_replay_inputs(
            self.traffic_path,
            self.forecast_path,
        )

        arrivals = generate_arrival_times(traffic, seed=42)

        for row in traffic.itertuples(index=False):
            request_times = arrivals.loc[
                arrivals["request_id"].str.startswith(
                    f"{row.workload_id}:{int(row.minute)}:"
                ),
                "arrival_time_seconds",
            ]

            self.assertTrue(
                request_times.ge(int(row.minute) * 60).all()
            )
            self.assertTrue(
                request_times.lt((int(row.minute) + 1) * 60).all()
            )

class ServerPoolTests(unittest.TestCase):
    def test_servers_become_active_after_startup_delay(self):
        pool = ServerPool(
            initial_active_servers=1,
            minimum_servers=1,
            startup_delay_seconds=180,
        )

        pool.scale_to(2, current_time=0)

        self.assertEqual(pool.active_servers, 1)
        self.assertEqual(pool.starting_servers, 1)

        pool.activate_ready_servers(current_time=179)

        self.assertEqual(pool.active_servers, 1)
        self.assertEqual(pool.starting_servers, 1)

        pool.activate_ready_servers(current_time=180)

        self.assertEqual(pool.active_servers, 2)
        self.assertEqual(pool.starting_servers, 0)

    def test_scaling_down_cancels_pending_servers_first(self):
        pool = ServerPool(
            initial_active_servers=1,
            minimum_servers=1,
            startup_delay_seconds=180,
        )

        pool.scale_to(3, current_time=0)
        self.assertEqual(pool.starting_servers, 2)

        pool.scale_to(1, current_time=60)

        self.assertEqual(pool.active_servers, 1)
        self.assertEqual(pool.starting_servers, 0)

class RequestQueueTests(unittest.TestCase):
    def test_queue_is_fifo(self):
        queue = RequestQueue(timeout_seconds=60)

        queue.add(WaitingRequest("first", 0))
        queue.add(WaitingRequest("second", 1))

        first = queue.pop_next(current_time=2)
        second = queue.pop_next(current_time=2)

        self.assertEqual(first.request_id, "first")
        self.assertEqual(second.request_id, "second")
        self.assertEqual(len(queue), 0)

    def test_expired_requests_are_dropped(self):
        queue = RequestQueue(timeout_seconds=60)

        queue.add(WaitingRequest("old", 0))
        queue.add(WaitingRequest("new", 10))

        queue.expire_requests(current_time=60)

        self.assertEqual(queue.dropped_request_ids, ["old"])
        self.assertEqual(queue.request_ids(), ["new"])
        self.assertEqual(len(queue), 1)

    def test_expired_request_is_not_returned(self):
        queue = RequestQueue(timeout_seconds=60)

        queue.add(WaitingRequest("expired", 0))

        result = queue.pop_next(current_time=60)

        self.assertIsNone(result)
        self.assertEqual(queue.dropped_request_ids, ["expired"])

class ServicePoolTests(unittest.TestCase):
    def test_busy_server_cannot_accept_two_requests(self):
        pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )

        first = pool.assign_if_available(current_time=0)
        second = pool.assign_if_available(current_time=0.5)
        third = pool.assign_if_available(current_time=1)

        self.assertEqual(first, (0, 1.0))
        self.assertIsNone(second)
        self.assertEqual(third, (0, 2.0))

    def test_multiple_servers_can_serve_simultaneously(self):
        pool = ServicePool(
            active_servers=2,
            service_time_seconds=1,
        )

        first = pool.assign_if_available(current_time=0)
        second = pool.assign_if_available(current_time=0)
        third = pool.assign_if_available(current_time=0)

        self.assertEqual(first, (0, 1.0))
        self.assertEqual(second, (1, 1.0))
        self.assertIsNone(third)

    def test_only_idle_servers_are_removed(self):
        pool = ServicePool(
            active_servers=2,
            service_time_seconds=1,
        )

        pool.assign_if_available(current_time=0)

        removed = pool.remove_idle_servers(
            count=1,
            current_time=0,
        )

        self.assertEqual(removed, 1)
        self.assertEqual(pool.active_servers, 1)

        removed_busy = pool.remove_idle_servers(
            count=1,
            current_time=0,
        )

        self.assertEqual(removed_busy, 0)
        self.assertEqual(pool.active_servers, 1)

class DispatchTests(unittest.TestCase):
    def test_request_is_served_when_server_is_available(self):
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )
        request_queue = RequestQueue(timeout_seconds=60)

        request = WaitingRequest("request-1", 0.0)

        result = dispatch_request(
            request,
            current_time=0.0,
            service_pool=service_pool,
            request_queue=request_queue,
        )

        self.assertEqual(result["server_index"], 0)
        self.assertEqual(result["completion_time"], 1.0)
        self.assertEqual(len(request_queue), 0)

    def test_busy_server_sends_request_to_queue(self):
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )
        request_queue = RequestQueue(timeout_seconds=60)

        first = WaitingRequest("request-1", 0.0)
        second = WaitingRequest("request-2", 0.5)

        dispatch_request(
            first,
            current_time=0.0,
            service_pool=service_pool,
            request_queue=request_queue,
        )

        result = dispatch_request(
            second,
            current_time=0.5,
            service_pool=service_pool,
            request_queue=request_queue,
        )

        self.assertIsNone(result)
        self.assertEqual(request_queue.request_ids(), ["request-2"])

    def test_queued_request_is_served_when_server_frees_up(self):
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )
        request_queue = RequestQueue(timeout_seconds=60)

        first = WaitingRequest("request-1", 0.0)
        second = WaitingRequest("request-2", 0.5)

        dispatch_request(first, 0.0, service_pool, request_queue)
        dispatch_request(second, 0.5, service_pool, request_queue)

        result = dispatch_next_waiting_request(
            current_time=1.0,
            service_pool=service_pool,
            request_queue=request_queue,
        )

        self.assertEqual(result["request"].request_id, "request-2")
        self.assertEqual(result["completion_time"], 2.0)
        self.assertEqual(len(request_queue), 0)

class SynchronizationTests(unittest.TestCase):
    def test_ready_servers_are_added_to_service_pool(self):
        server_pool = ServerPool(
            initial_active_servers=1,
            minimum_servers=1,
            startup_delay_seconds=180,
        )
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )

        server_pool.scale_to(2, current_time=0)

        before_ready = activate_ready_servers(
            server_pool,
            service_pool,
            current_time=179,
        )

        self.assertEqual(before_ready, 0)
        self.assertEqual(service_pool.active_servers, 1)

        after_ready = activate_ready_servers(
            server_pool,
            service_pool,
            current_time=180,
        )

        self.assertEqual(after_ready, 1)
        self.assertEqual(service_pool.active_servers, 2)

class TargetApplicationTests(unittest.TestCase):
    def test_scale_up_then_cancel_pending_startups(self):
        server_pool = ServerPool(
            initial_active_servers=1,
            minimum_servers=1,
            startup_delay_seconds=180,
        )
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )

        result = apply_target_servers(
            target_servers=3,
            server_pool=server_pool,
            service_pool=service_pool,
            current_time=0,
        )

        self.assertEqual(result["active_servers"], 1)
        self.assertEqual(result["starting_servers"], 2)

        result = apply_target_servers(
            target_servers=1,
            server_pool=server_pool,
            service_pool=service_pool,
            current_time=60,
        )

        self.assertEqual(result["active_servers"], 1)
        self.assertEqual(result["starting_servers"], 0)

    def test_ready_servers_can_later_be_removed(self):
        server_pool = ServerPool(
            initial_active_servers=1,
            minimum_servers=1,
            startup_delay_seconds=180,
        )
        service_pool = ServicePool(
            active_servers=1,
            service_time_seconds=1,
        )

        apply_target_servers(
            target_servers=2,
            server_pool=server_pool,
            service_pool=service_pool,
            current_time=0,
        )

        result = apply_target_servers(
            target_servers=2,
            server_pool=server_pool,
            service_pool=service_pool,
            current_time=180,
        )

        self.assertEqual(result["active_servers"], 2)
        self.assertEqual(result["starting_servers"], 0)

        result = apply_target_servers(
            target_servers=1,
            server_pool=server_pool,
            service_pool=service_pool,
            current_time=180,
        )

        self.assertEqual(result["active_servers"], 1)
        self.assertEqual(result["starting_servers"], 0)

class ReplayTests(unittest.TestCase):
    def test_small_forecast_replay_returns_outputs(self):
        repository_root = Path(__file__).parents[2]
        examples = (
            repository_root
            / "part1"
            / "outputs"
            / "examples"
        )

        traffic, forecasts = load_replay_inputs(
            examples / "traffic.csv",
            examples / "forecasts.csv",
        )

        config = {
            "capacity_per_server_per_minute": 60,
            "service_time_seconds": 1.0,
            "startup_delay_minutes": 3,
            "request_timeout_seconds": 60,
            "slo_seconds": 5,
            "initial_active_servers": 1,
            "minimum_servers": 1,
            "arrival_mode": "uniform",
            "random_seed": 42,
        }

        result = simulate_replay(
            traffic=traffic,
            forecasts=forecasts,
            config=config,
            policy="forecast",
            workload_id="steady_01",
            model="persistence",
            run_id="example-steady-persistence",
        )

        summary = result["summary"]
        minute_logs = result["minute_logs"]
        outcomes = result["request_outcomes"]

        expected_arrivals = int(
            traffic.loc[
                traffic["workload_id"].eq("steady_01"),
                "request_count",
            ].sum()
        )

        self.assertEqual(
            summary["arrivals"],
            expected_arrivals,
        )
        self.assertEqual(len(minute_logs), 124)
        self.assertEqual(
            len(outcomes),
            expected_arrivals,
        )
        self.assertTrue(
            set(outcomes["status"]).issubset(
                {"completed", "dropped", "waiting"}
            )
        )
        self.assertGreater(
            summary["server_minutes"],
            0,
        )

    def test_small_reactive_replay_returns_outputs(self):
        repository_root = Path(__file__).parents[2]
        examples = (
            repository_root
            / "part1"
            / "outputs"
            / "examples"
        )

        traffic, forecasts = load_replay_inputs(
            examples / "traffic.csv",
            examples / "forecasts.csv",
        )

        config = {
            "capacity_per_server_per_minute": 60,
            "service_time_seconds": 1.0,
            "startup_delay_minutes": 3,
            "request_timeout_seconds": 60,
            "slo_seconds": 5,
            "initial_active_servers": 1,
            "minimum_servers": 1,
            "arrival_mode": "uniform",
            "random_seed": 42,
        }

        result = simulate_replay(
            traffic=traffic,
            forecasts=forecasts,
            config=config,
            policy="reactive",
            workload_id="bursty_02",
            run_id="example-bursty-reactive",
        )

        summary = result["summary"]
        outcomes = result["request_outcomes"]

        expected_arrivals = int(
            traffic.loc[
                traffic["workload_id"].eq("bursty_02"),
                "request_count",
            ].sum()
        )

        self.assertEqual(summary["model"], "reactive")
        self.assertEqual(summary["arrivals"], expected_arrivals)
        self.assertEqual(len(outcomes), expected_arrivals)
        self.assertGreater(summary["server_minutes"], 0)

if __name__ == "__main__":
    unittest.main()
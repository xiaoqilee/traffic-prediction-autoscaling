import unittest

from part2.scaling import (
    target_servers_from_forecast,
    target_servers_reactive,
    choose_target_servers,
    target_servers_with_headroom,
    target_servers_with_queue_feedback,
)

class ScalingTests(unittest.TestCase):
    def test_forecast_scaling(self):
        self.assertEqual(target_servers_from_forecast(0, 60), 1)
        self.assertEqual(target_servers_from_forecast(60, 60), 1)
        self.assertEqual(target_servers_from_forecast(61, 60), 2)
        self.assertEqual(target_servers_from_forecast(120, 60), 2)
        self.assertEqual(target_servers_from_forecast(121, 60), 3)

    def test_headroom_scaling(self):
        self.assertEqual(
            target_servers_with_headroom(
                predicted_count=60,
                capacity_per_server_per_minute=60,
                headroom_factor=1.0,
            ),
            1,
        )

        self.assertEqual(
            target_servers_with_headroom(
                predicted_count=60,
                capacity_per_server_per_minute=60,
                headroom_factor=1.5,
            ),
            2,
        )

        self.assertEqual(
            target_servers_with_headroom(
                predicted_count=120,
                capacity_per_server_per_minute=60,
                headroom_factor=1.5,
            ),
            3,
        )

    def test_queue_feedback_scaling(self):
        self.assertEqual(
            target_servers_with_queue_feedback(
                predicted_count=60,
                queue_length=0,
                capacity_per_server_per_minute=60,
            ),
            1,
        )

        self.assertEqual(
            target_servers_with_queue_feedback(
                predicted_count=60,
                queue_length=60,
                capacity_per_server_per_minute=60,
            ),
            2,
        )

        self.assertEqual(
            target_servers_with_queue_feedback(
                predicted_count=60,
                queue_length=120,
                capacity_per_server_per_minute=60,
            ),
            3,
        )

    def test_reactive_scaling(self):
        self.assertEqual(target_servers_reactive(30, 60), 1)
        self.assertEqual(target_servers_reactive(60, 60), 1)
        self.assertEqual(target_servers_reactive(61, 60), 2)

    def test_negative_demand_does_not_reduce_below_minimum(self):
        self.assertEqual(target_servers_from_forecast(-10, 60), 1)
        self.assertEqual(target_servers_reactive(-10, 60), 1)

    def test_invalid_capacity_fails(self):
        with self.assertRaises(ValueError):
            target_servers_from_forecast(50, 0)

    def test_unified_headroom_policy(self):
        self.assertEqual(
            choose_target_servers(
                "forecast_headroom",
                capacity_per_server_per_minute=60,
                predicted_count=60,
                headroom_factor=1.5,
            ),
            2,
        )

    def test_unified_queue_feedback_policy(self):
        self.assertEqual(
            choose_target_servers(
                "forecast_queue_feedback",
                capacity_per_server_per_minute=60,
                predicted_count=60,
                queue_length=60,
            ),
            2,
        )

    def test_unified_reactive_policy(self):
        self.assertEqual(
            choose_target_servers(
                "reactive",
                capacity_per_server_per_minute=60,
                previous_actual_count=61,
            ),
            2,
        )

if __name__ == "__main__":
    unittest.main()
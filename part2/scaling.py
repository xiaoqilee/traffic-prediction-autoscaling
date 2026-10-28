"""Scaling policies for the autoscaling simulator."""

import math

def _validate_settings(capacity_per_server_per_minute, minimum_servers):
    if capacity_per_server_per_minute <= 0:
        raise ValueError("Server capacity must be positive")
    if minimum_servers < 0:
        raise ValueError("Minimum servers cannot be negative")

def target_servers_from_forecast(
    predicted_count,
    capacity_per_server_per_minute,
    minimum_servers=1,
):
    """Choose servers using predicted requests for the target minute."""
    _validate_settings(capacity_per_server_per_minute, minimum_servers)

    predicted_count = max(0.0, float(predicted_count))
    required_servers = math.ceil(
        predicted_count / capacity_per_server_per_minute
    )

    return max(minimum_servers, required_servers)

def target_servers_reactive(
    previous_actual_count,
    capacity_per_server_per_minute,
    minimum_servers=1,
):
    """Choose servers using only the most recently observed demand."""
    _validate_settings(capacity_per_server_per_minute, minimum_servers)

    previous_actual_count = max(0.0, float(previous_actual_count))
    required_servers = math.ceil(
        previous_actual_count / capacity_per_server_per_minute
    )

    return max(minimum_servers, required_servers)

def choose_target_servers(
    policy,
    capacity_per_server_per_minute,
    minimum_servers=1,
    predicted_count=None,
    previous_actual_count=None,
):
    """Choose a target server count for a named scaling policy."""
    if policy == "forecast":
        if predicted_count is None:
            raise ValueError(
                "Forecast policy requires predicted_count"
            )

        return target_servers_from_forecast(
            predicted_count,
            capacity_per_server_per_minute,
            minimum_servers,
        )

    if policy == "reactive":
        if previous_actual_count is None:
            raise ValueError(
                "Reactive policy requires previous_actual_count"
            )

        return target_servers_reactive(
            previous_actual_count,
            capacity_per_server_per_minute,
            minimum_servers,
        )

    raise ValueError(f"Unknown scaling policy: {policy}")
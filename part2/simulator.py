"""Input loading and validation for the autoscaling simulator."""

from pathlib import Path
import pandas as pd
import numpy as np
import math
from collections import deque
from dataclasses import dataclass

TRAFFIC_COLUMNS = {
    "workload_id",
    "minute",
    "request_count",
    "split",
}

FORECAST_COLUMNS = {
    "workload_id",
    "prediction_time",
    "target_minute",
    "model",
    "predicted_count",
}

def _require_columns(frame, required_columns, name):
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

def load_replay_inputs(traffic_path, forecast_path, expected_horizon=3):
    """Load and validate traffic and forecast files for one replay."""
    traffic_path = Path(traffic_path)
    forecast_path = Path(forecast_path)

    traffic = pd.read_csv(traffic_path)
    forecasts = pd.read_csv(forecast_path)

    _require_columns(traffic, TRAFFIC_COLUMNS, "Traffic file")
    _require_columns(forecasts, FORECAST_COLUMNS, "Forecast file")

    if traffic.empty:
        raise ValueError("Traffic file is empty")
    if forecasts.empty:
        raise ValueError("Forecast file is empty")

    if traffic.duplicated(["workload_id", "minute"]).any():
        raise ValueError("Traffic contains duplicate workload/minute rows")

    if forecasts.duplicated(
        ["workload_id", "model", "target_minute"]
    ).any():
        raise ValueError(
            "Forecasts contain duplicate workload/model/target rows"
        )

    horizon = forecasts["target_minute"] - forecasts["prediction_time"]
    if not horizon.eq(expected_horizon).all():
        raise ValueError(
            f"Forecast horizon must be {expected_horizon} minutes"
        )

    if traffic["request_count"].isna().any():
        raise ValueError("Traffic contains missing request counts")

    if traffic["request_count"].lt(0).any():
        raise ValueError("Traffic contains negative request counts")

    if forecasts["predicted_count"].isna().any():
        raise ValueError("Forecasts contain missing predictions")

    if forecasts["predicted_count"].lt(0).any():
        raise ValueError("Forecasts contain negative predictions")

    traffic = traffic.sort_values(
        ["workload_id", "minute"]
    ).reset_index(drop=True)

    forecasts = forecasts.sort_values(
        ["workload_id", "model", "target_minute"]
    ).reset_index(drop=True)

    return traffic, forecasts

ARRIVAL_COLUMNS = [
    "workload_id",
    "request_id",
    "arrival_time_seconds",
]

def generate_arrival_times(traffic, seed=42):
    """Generate deterministic within-minute arrivals from minute counts."""
    rng = np.random.default_rng(seed)
    arrivals = []

    for row in traffic.itertuples(index=False):
        count = int(row.request_count)

        offsets = np.sort(
            rng.uniform(0.0, 60.0, size=count)
        )

        for request_number, offset in enumerate(offsets):
            arrivals.append(
                {
                    "workload_id": row.workload_id,
                    "request_id": (
                        f"{row.workload_id}:"
                        f"{int(row.minute)}:"
                        f"{request_number}"
                    ),
                    "arrival_time_seconds": (
                        int(row.minute) * 60.0 + float(offset)
                    ),
                }
            )

    return pd.DataFrame(arrivals, columns=ARRIVAL_COLUMNS)

class ServerPool:
    """Track active servers and servers waiting for startup."""

    def __init__(
        self,
        initial_active_servers=1,
        minimum_servers=1,
        startup_delay_seconds=180,
    ):
        if initial_active_servers < minimum_servers:
            raise ValueError(
                "Initial servers cannot be below the minimum"
            )
        if startup_delay_seconds < 0:
            raise ValueError("Startup delay cannot be negative")

        self.minimum_servers = minimum_servers
        self.startup_delay_seconds = startup_delay_seconds
        self.active_servers = initial_active_servers
        self.pending_ready_times = []

    @property
    def starting_servers(self):
        return len(self.pending_ready_times)

    @property
    def total_servers(self):
        return self.active_servers + self.starting_servers

    def activate_ready_servers(self, current_time):
        """Move servers whose startup delay has elapsed to active."""
        ready = [
            ready_time
            for ready_time in self.pending_ready_times
            if ready_time <= current_time
        ]

        self.pending_ready_times = [
            ready_time
            for ready_time in self.pending_ready_times
            if ready_time > current_time
        ]

        self.active_servers += len(ready)

    def scale_to(self, target_servers, current_time):
        """Start or cancel servers to reach the requested target."""
        target_servers = max(
            self.minimum_servers,
            int(target_servers),
        )

        self.activate_ready_servers(current_time)

        while self.total_servers > target_servers:
            if self.pending_ready_times:
                self.pending_ready_times.pop()
            elif self.active_servers > self.minimum_servers:
                self.active_servers -= 1
            else:
                break

        while self.total_servers < target_servers:
            self.pending_ready_times.append(
                current_time + self.startup_delay_seconds
            )

@dataclass(frozen=True)
class WaitingRequest:
    request_id: str
    arrival_time_seconds: float

class RequestQueue:
    """FIFO queue with request timeouts."""

    def __init__(self, timeout_seconds):
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        self.timeout_seconds = timeout_seconds
        self._waiting = deque()
        self.dropped_request_ids = []

    def add(self, request):
        self._waiting.append(request)

    def expire_requests(self, current_time):
        """Drop queued requests whose timeout has elapsed."""
        while self._waiting:
            request = self._waiting[0]
            deadline = (
                request.arrival_time_seconds
                + self.timeout_seconds
            )

            if deadline > current_time:
                break

            self.dropped_request_ids.append(request.request_id)
            self._waiting.popleft()

    def pop_next(self, current_time):
        """Return the oldest non-expired request, if one exists."""
        self.expire_requests(current_time)

        if not self._waiting:
            return None

        return self._waiting.popleft()

    def __len__(self):
        return len(self._waiting)

    def request_ids(self):
        return [request.request_id for request in self._waiting]

class ServicePool:
    """Track when each active server becomes available."""

    def __init__(self, active_servers, service_time_seconds):
        if active_servers < 1:
            raise ValueError("At least one active server is required")
        if service_time_seconds <= 0:
            raise ValueError("Service time must be positive")

        self.service_time_seconds = service_time_seconds
        self.available_at = [0.0] * active_servers

    @property
    def active_servers(self):
        return len(self.available_at)

    def add_servers(self, count, current_time):
        """Add servers that are immediately ready to serve."""
        if count < 0:
            raise ValueError("Cannot add a negative number of servers")

        self.available_at.extend(
            [float(current_time)] * count
        )

    def remove_idle_servers(self, count, current_time):
        """Remove idle servers, leaving busy servers untouched."""
        removed = 0

        for index in range(len(self.available_at) - 1, -1, -1):
            if removed >= count:
                break

            if self.available_at[index] <= current_time:
                self.available_at.pop(index)
                removed += 1

        return removed

    def assign_if_available(self, current_time):
        """Assign one request to the earliest available server."""
        available_servers = [
            (time, index)
            for index, time in enumerate(self.available_at)
            if time <= current_time
        ]

        if not available_servers:
            return None

        _, server_index = min(available_servers)

        completion_time = (
            float(current_time) + self.service_time_seconds
        )
        self.available_at[server_index] = completion_time

        return server_index, completion_time

def dispatch_request(
    request,
    current_time,
    service_pool,
    request_queue,
):
    """Serve a request immediately or place it in the FIFO queue."""
    assignment = service_pool.assign_if_available(current_time)

    if assignment is None:
        request_queue.add(request)
        return None

    server_index, completion_time = assignment

    return {
        "request": request,
        "server_index": server_index,
        "completion_time": completion_time,
    }


def dispatch_next_waiting_request(
    current_time,
    service_pool,
    request_queue,
):
    """Serve the oldest queued request when capacity becomes available."""
    request = request_queue.pop_next(current_time)

    if request is None:
        return None

    assignment = service_pool.assign_if_available(current_time)

    if assignment is None:
        request_queue.add(request)
        return None

    server_index, completion_time = assignment

    return {
        "request": request,
        "server_index": server_index,
        "completion_time": completion_time,
    }

def activate_ready_servers(
    server_pool,
    service_pool,
    current_time,
):
    """Activate ready servers in both simulator components."""
    active_before = server_pool.active_servers

    server_pool.activate_ready_servers(current_time)

    newly_active = (
        server_pool.active_servers - active_before
    )

    if newly_active > 0:
        service_pool.add_servers(
            newly_active,
            current_time,
        )

    return newly_active

def apply_target_servers(
    target_servers,
    server_pool,
    service_pool,
    current_time,
):
    """Apply a target count while respecting startup and busy servers."""
    target_servers = max(
        server_pool.minimum_servers,
        int(target_servers),
    )

    activate_ready_servers(
        server_pool,
        service_pool,
        current_time,
    )

    if target_servers > server_pool.total_servers:
        server_pool.scale_to(
            target_servers,
            current_time,
        )

    else:
        # Cancel pending startups before removing active servers.
        while (
            server_pool.starting_servers > 0
            and server_pool.total_servers > target_servers
        ):
            server_pool.pending_ready_times.pop()

        desired_active = max(
            server_pool.minimum_servers,
            target_servers - server_pool.starting_servers,
        )

        to_remove = max(
            0,
            service_pool.active_servers - desired_active,
        )

        service_pool.remove_idle_servers(
            to_remove,
            current_time,
        )

        # Keep the server-count tracker consistent with actual active
        # service capacity. Busy servers remain until they become idle.
        server_pool.active_servers = service_pool.active_servers

    return {
        "active_servers": service_pool.active_servers,
        "starting_servers": server_pool.starting_servers,
        "total_servers": (
            service_pool.active_servers
            + server_pool.starting_servers
        ),
    }

def simulate_replay(
    traffic,
    forecasts,
    config,
    policy="forecast",
    workload_id=None,
    model=None,
    run_id=None,
):
    """Replay one workload through the autoscaling simulator."""
    from part2.scaling import choose_target_servers

    if workload_id is None:
        workload_ids = traffic["workload_id"].unique()
        if len(workload_ids) != 1:
            raise ValueError(
                "Specify workload_id when multiple workloads are present"
            )
        workload_id = workload_ids[0]

    traffic = traffic.loc[
        traffic["workload_id"].eq(workload_id)
    ].copy()

    forecasts = forecasts.loc[
        forecasts["workload_id"].eq(workload_id)
    ].copy()

    if traffic.empty:
        raise ValueError(f"No traffic found for {workload_id}")

    if policy == "forecast":
        if model is None:
            models = forecasts["model"].unique()
            if len(models) != 1:
                raise ValueError(
                    "Specify model when multiple forecasts are present"
                )
            model = models[0]

        forecasts = forecasts.loc[
            forecasts["model"].eq(model)
        ].copy()

    elif policy == "reactive":
        model = "reactive"

    else:
        raise ValueError(f"Unknown policy: {policy}")

    traffic = traffic.sort_values("minute").reset_index(drop=True)
    forecasts = forecasts.sort_values(
        "prediction_time"
    ).reset_index(drop=True)

    start_minute = int(traffic["minute"].min())
    end_minute = int(traffic["minute"].max())
    traffic_duration_seconds = (
        (end_minute - start_minute + 1) * 60
    )

    arrivals = generate_arrival_times(
        traffic,
        seed=config["random_seed"],
    )
    arrivals["relative_time_seconds"] = (
        arrivals["arrival_time_seconds"]
        - start_minute * 60
    )

    arrivals_by_second = {}

    for row in arrivals.itertuples(index=False):
        second = int(
            math.floor(row.relative_time_seconds)
        )
        arrivals_by_second.setdefault(
            second,
            [],
        ).append(row)

    actual_by_minute = dict(
        zip(
            traffic["minute"],
            traffic["request_count"],
        )
    )

    forecast_by_minute = dict(
        zip(
            forecasts["prediction_time"],
            forecasts["predicted_count"],
        )
    )

    server_pool = ServerPool(
        initial_active_servers=config[
            "initial_active_servers"
        ],
        minimum_servers=config["minimum_servers"],
        startup_delay_seconds=(
            config["startup_delay_minutes"] * 60
        ),
    )

    service_pool = ServicePool(
        active_servers=config["initial_active_servers"],
        service_time_seconds=config[
            "service_time_seconds"
        ],
    )

    request_queue = RequestQueue(
        timeout_seconds=config[
            "request_timeout_seconds"
        ]
    )

    request_records = {}
    completion_schedule = {}
    minute_logs = {}
    dropped_seen = 0

    def counter_for(minute):
        if minute not in minute_logs:
            minute_logs[minute] = {
                "run_id": run_id,
                "workload_id": workload_id,
                "model": model,
                "minute": minute,
                "arrivals": 0,
                "completed": 0,
                "dropped": 0,
                "active_servers": 0,
                "starting_servers": 0,
                "queue_length": 0,
            }

        return minute_logs[minute]

    def capture_new_drops(current_minute):
        nonlocal dropped_seen

        new_drops = request_queue.dropped_request_ids[
            dropped_seen:
        ]

        for request_id in new_drops:
            request_records[request_id][
                "status"
            ] = "dropped"
            counter_for(
                min(current_minute, end_minute)
            )["dropped"] += 1

        dropped_seen += len(new_drops)

    def record_service(result):
        if result is None:
            return

        request = result["request"]
        completion_time = result["completion_time"]

        request_records[request.request_id].update(
            {
                "status": "completed",
                "completion_time_seconds": (
                    completion_time
                ),
                "response_time_seconds": (
                    completion_time
                    - request.arrival_time_seconds
                ),
            }
        )

        completion_second = int(
            math.ceil(completion_time)
        )

        completion_schedule.setdefault(
            completion_second,
            [],
        ).append(request.request_id)

    def drain_waiting_requests(current_time):
        while True:
            result = dispatch_next_waiting_request(
                current_time=current_time,
                service_pool=service_pool,
                request_queue=request_queue,
            )

            capture_new_drops(
                start_minute
                + int(current_time // 60)
            )

            if result is None:
                break

            record_service(result)

    for current_second in range(
        traffic_duration_seconds
    ):
        current_minute = start_minute + (
            current_second // 60
        )
        counter = counter_for(current_minute)

        newly_active = activate_ready_servers(
            server_pool,
            service_pool,
            current_second,
        )

        # Process requests whose service finished now.
        completed_now = completion_schedule.pop(
            current_second,
            [],
        )
        counter["completed"] += len(completed_now)

        request_queue.expire_requests(
            current_second
        )
        capture_new_drops(current_minute)

        # Make a scaling decision at each minute boundary.
        if current_second % 60 == 0:
            if policy == "forecast":
                predicted_count = forecast_by_minute.get(
                    current_minute
                )

                if predicted_count is not None:
                    target_servers = choose_target_servers(
                        "forecast",
                        config[
                            "capacity_per_server_per_minute"
                        ],
                        config["minimum_servers"],
                        predicted_count=predicted_count,
                    )
                    apply_target_servers(
                        target_servers,
                        server_pool,
                        service_pool,
                        current_second,
                    )

            else:
                previous_actual = actual_by_minute.get(
                    current_minute - 1,
                    0,
                )

                target_servers = choose_target_servers(
                    "reactive",
                    config[
                        "capacity_per_server_per_minute"
                    ],
                    config["minimum_servers"],
                    previous_actual_count=previous_actual,
                )

                apply_target_servers(
                    target_servers,
                    server_pool,
                    service_pool,
                    current_second,
                )

            drain_waiting_requests(current_second)

        # Process new arrivals during this second.
        for row in arrivals_by_second.get(
            current_second,
            [],
        ):
            request = WaitingRequest(
                request_id=row.request_id,
                arrival_time_seconds=(
                    row.relative_time_seconds
                ),
            )

            request_records[request.request_id] = {
                "run_id": run_id,
                "workload_id": workload_id,
                "model": model,
                "request_id": request.request_id,
                "arrival_time_seconds": (
                    row.relative_time_seconds
                ),
                "completion_time_seconds": None,
                "response_time_seconds": None,
                "status": "waiting",
            }

            counter["arrivals"] += 1

            result = dispatch_request(
                request=request,
                current_time=current_second,
                service_pool=service_pool,
                request_queue=request_queue,
            )

            record_service(result)

        capture_new_drops(current_minute)

        counter["active_servers"] = (
            service_pool.active_servers
        )
        counter["starting_servers"] = (
            server_pool.starting_servers
        )
        counter["queue_length"] = len(request_queue)

    # drain requests that remain queued after the final traffic second
    drain_time = float(traffic_duration_seconds)

    while len(request_queue) > 0 or completion_schedule:
        activate_ready_servers(
            server_pool,
            service_pool,
            drain_time,
        )

        completed_now = completion_schedule.pop(
            int(math.ceil(drain_time)),
            [],
        )
        counter_for(end_minute)["completed"] += len(
            completed_now
        )

        request_queue.expire_requests(drain_time)
        capture_new_drops(end_minute)

        drain_waiting_requests(drain_time)
        capture_new_drops(end_minute)

        if len(request_queue) == 0 and not completion_schedule:
            break

        future_times = list(completion_schedule.keys())
        future_times.extend(server_pool.pending_ready_times)

        if not future_times:
            break

        next_time = min(float(time) for time in future_times)

        if next_time <= drain_time:
            drain_time += 1.0
        else:
            drain_time = next_time

    final_counter = counter_for(end_minute)
    final_counter["active_servers"] = service_pool.active_servers
    final_counter["starting_servers"] = server_pool.starting_servers
    final_counter["queue_length"] = len(request_queue)

    minute_frame = pd.DataFrame(
        list(minute_logs.values())
    ).sort_values("minute")

    outcomes_frame = pd.DataFrame(
        list(request_records.values())
    )

    completed = outcomes_frame.loc[
        outcomes_frame["status"].eq("completed")
    ]

    server_minutes = (
        minute_frame["active_servers"]
        + minute_frame["starting_servers"]
    ).sum()

    summary = {
        "run_id": run_id,
        "workload_id": workload_id,
        "model": model,
        "arrivals": len(outcomes_frame),
        "completed": int(
            outcomes_frame["status"].eq("completed").sum()
        ),
        "dropped": int(
            outcomes_frame["status"].eq("dropped").sum()
        ),
        "server_minutes": float(server_minutes),
        "mean_response_time_seconds": (
            float(
                completed[
                    "response_time_seconds"
                ].mean()
            )
            if not completed.empty
            else None
        ),
    }

    return {
        "minute_logs": minute_frame,
        "request_outcomes": outcomes_frame,
        "summary": summary,
    }
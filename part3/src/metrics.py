"""Service quality metrics derived from Part 2 request_outcomes."""

import numpy as np
import pandas as pd


def filter_scoring_interval(outcomes, scoring_start_minute, scoring_end_minute, warmup_start_minute=0):
    """Keep only requests whose arrival falls within the scored minutes.

    arrival_time_seconds is relative to warmup_start_minute * 60.
    """
    offset = float(warmup_start_minute) * 60.0
    start_rel = float(scoring_start_minute) * 60.0 - offset
    end_rel = float(scoring_end_minute + 1) * 60.0 - offset
    mask = (
        outcomes["arrival_time_seconds"].ge(start_rel)
        & outcomes["arrival_time_seconds"].lt(end_rel)
    )
    return outcomes.loc[mask].copy()


def compute_latency_percentiles(outcomes, percentiles=(50, 95, 99)):
    """Return dict of latency percentiles for completed requests only."""
    completed = outcomes.loc[outcomes["status"].eq("completed"), "response_time_seconds"]
    if completed.empty:
        return {f"p{p}_latency": None for p in percentiles}
    return {
        f"p{p}_latency": float(np.percentile(completed, p))
        for p in percentiles
    }


def compute_slo_violation_rate(outcomes, slo_seconds):
    """(completed > slo + dropped) / all arrivals."""
    total = len(outcomes)
    if total == 0:
        return None
    slow = int(
        (outcomes["status"].eq("completed")
        & outcomes["response_time_seconds"].gt(slo_seconds)).sum()
    )
    dropped = int(outcomes["status"].eq("dropped").sum())
    return (slow + dropped) / total


def compute_drop_rate(outcomes):
    """dropped / all arrivals."""
    total = len(outcomes)
    if total == 0:
        return None
    return float(outcomes["status"].eq("dropped").sum()) / total


def summarize_run(outcomes, minute_logs, scoring_start_minute, scoring_end_minute, slo_seconds):
    """Return a dict with all service metrics for one run, scoped to the scoring interval."""
    warmup_start_minute = int(minute_logs["minute"].min())
    scored = filter_scoring_interval(
        outcomes, scoring_start_minute, scoring_end_minute, warmup_start_minute
    )

    latencies = compute_latency_percentiles(scored)
    slo_rate = compute_slo_violation_rate(scored, slo_seconds)
    drop_rate = compute_drop_rate(scored)

    scored_logs = minute_logs.loc[
        minute_logs["minute"].between(scoring_start_minute, scoring_end_minute)
    ]
    server_minutes = float(
        (scored_logs["active_servers"] + scored_logs["starting_servers"]).sum()
    )

    return {
        "arrivals": len(scored),
        "completed": int(scored["status"].eq("completed").sum()),
        "dropped": int(scored["status"].eq("dropped").sum()),
        "server_minutes": server_minutes,
        "slo_violation_rate": slo_rate,
        "drop_rate": drop_rate,
        **latencies,
    }

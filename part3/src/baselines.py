"""Derived forecast baselines constructed in Part 3 from observed traffic.

Seasonal-naive forecasts the demand one period (default one day) earlier, which
exploits the daily cycle that persistence/reactive ignore. It is built from the
full traffic history because the lookup for the first validation minutes falls in
the training period.
"""

import numpy as np
import pandas as pd

SEASONAL_NAIVE_MODEL = "seasonal_naive"


def build_seasonal_naive_forecast(
    full_traffic,
    workloads,
    target_minutes_by_workload,
    horizon=3,
    period_minutes=1440,
):
    """Build a seasonal-naive forecast series with the Part 1 forecast schema.

    For target minute T: predicted_count = actual demand at (T - period_minutes),
    emitted at prediction_time = T - horizon (same horizon as the real forecasts).
    Values missing in history fall back to the nearest earlier observed demand.
    """
    rows = []
    misses = 0
    for workload_id in workloads:
        wl = full_traffic.loc[full_traffic["workload_id"].eq(workload_id)]
        demand = (
            wl.set_index("minute")["request_count"].sort_index().astype(float)
        )

        targets = np.asarray(sorted(target_minutes_by_workload[workload_id]))
        lagged = demand.reindex(targets - period_minutes)
        if lagged.isna().any():
            # Fall back to the most recent observed demand before the lookup point.
            filled = demand.reindex(
                range(int(demand.index.min()), int(targets.max()) - period_minutes + 1)
            ).ffill()
            lagged = filled.reindex(targets - period_minutes)
            misses += int(lagged.isna().sum())

        preds = np.clip(lagged.to_numpy(dtype=float), 0.0, None)
        for target, pred in zip(targets, preds):
            rows.append(
                {
                    "workload_id": workload_id,
                    "prediction_time": int(target) - horizon,
                    "target_minute": int(target),
                    "model": SEASONAL_NAIVE_MODEL,
                    "predicted_count": round(float(pred), 4),
                }
            )

    if misses:
        print(f"  seasonal_naive: {misses} target minutes lacked history (forward-filled)")

    return pd.DataFrame(rows)


def forecast_accuracy(forecast_df, full_traffic, scoring_start_minute, scoring_end_minute):
    """MAE/RMSE of a forecast series vs actual demand at each target minute.

    Scoped to the scoring interval and measured in target-minute space, matching
    how Part 1 defines forecast error.
    """
    rows = []
    for (workload_id, model), group in forecast_df.groupby(["workload_id", "model"]):
        demand = full_traffic.loc[
            full_traffic["workload_id"].eq(workload_id)
        ].set_index("minute")["request_count"].astype(float)

        scored = group.loc[
            group["target_minute"].between(scoring_start_minute, scoring_end_minute)
        ]
        actual = demand.reindex(scored["target_minute"].to_numpy())
        error = scored["predicted_count"].to_numpy() - actual.to_numpy()
        error = error[~np.isnan(error)]
        rows.append(
            {
                "workload_id": workload_id,
                "model": model,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
            }
        )
    return pd.DataFrame(rows)

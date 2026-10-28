# Part 2: Autoscaling Simulator

This module connects the Part 1 traffic forecasts to a discrete-time autoscaling simulator. It models incoming requests, server capacity, startup delays, request queues, timeouts, response times, and estimated server usage.

## Scaling Policies

The simulator compares:

- Forecast-driven scaling using predicted request counts
- A reactive baseline using the previous minute's observed demand

Each server is configured to handle a fixed number of requests per minute.

## Inputs

The simulator reads the Part 1 example files:

- `part1/outputs/examples/traffic.csv`
- `part1/outputs/examples/forecasts.csv`

The forecast horizon is validated to be three minutes.

## Outputs

Running the example produces:

- `minute_logs.csv`: per-minute arrivals, completions, drops, queue length, and server counts
- `request_outcomes.csv`: request-level arrival, completion, response-time, and status information
- `summary.csv`: aggregate results for each workload and scaling policy

## Running the Example

From the repository root:

```powershell
python -m part2.run_example
```

## Running Tests

```powershell
python -m unittest discover -s part2/tests -p "test_*.py" -v
```

The simulator uses the configuration in `part2/config.json`, including server capacity, startup delay, timeout, initial server count, and random seed.
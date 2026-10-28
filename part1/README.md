# Part 1: Traffic Forecasting

Part 1 prepares four Azure Functions workloads, trains forecasting baselines, and exports predictions for Part 2 simulation and Part 3 evaluation.

## Structure

```text
part1/
├── src/                   # Four pipeline stages
│   ├── prepare_data.py    # A: select applications and prepare minute counts
│   ├── features.py        # B: build causal features and prediction targets
│   ├── forecast.py        # C: train baselines and export forecasts
│   └── diagnostics.py     # D: analyze errors and export integration examples
├── configs/part1.json     # Input paths, model settings, and random seed
├── data/
│   ├── raw/azure2019/     # Original compressed archive and source receipt
│   └── processed/        # Traffic, data contracts, and feature datasets
├── outputs/
│   ├── models/           # Saved models and selection metadata
│   ├── forecasts/        # Predictions consumed by Part 2
│   ├── metrics/          # Forecast scores and diagnostics for Part 3
│   ├── figures/          # Traffic and forecast plots
│   └── examples/         # Small traffic/forecast replay with timing bounds
├── tests/                # Data, feature, forecasting, and diagnostic checks
├── docs/part1-delivery.md # Part 2 / Part 3 usage guide
└── requirements.txt
```

## Data Flow

```text
Azure archive -> A: traffic -> B: features -> C: forecasts -> D: diagnostics
                       |                         |                |
                       +---------- Part 2 -------+                |
                                                 +--- Part 3 -----+
```

| Stage | Main behavior | Main output |
| --- | --- | --- |
| A | Aggregate HTTP functions by application; select two steady and two bursty applications using training data | `data/processed/traffic.csv`, workload manifest, data contract |
| B | Build lag, rolling, difference, and time features; split chronologically into train/validation/test | `data/processed/features/horizon_3/` |
| C | Fit Ridge and histogram GBDT separately for each application; select parameters by validation MAE; include Persistence | `outputs/models/`, `outputs/forecasts/validation.csv` |
| D | Measure underprediction and traffic shifts; identify diagnostic windows and create replay examples | `outputs/metrics/`, `outputs/figures/`, `outputs/examples/` |

At prediction boundary `t`, features use observations through `t-1` and predict minute `t+3`. Training uses days 1–8, validation uses days 9–11, and test uses days 12–14. The held-out test period has not been evaluated.

## Run

Run commands from `part1/`; configuration and artifact paths are relative to this directory. Use the shared uv environment at the repository root:

```sh
cd part1
source ../.venv/bin/activate
python -B src/prepare_data.py --download
python -B src/features.py
python -B src/forecast.py
python -B src/diagnostics.py
python -B -m unittest discover -s tests -v
```

Data preparation reuses the cached archive, extracts daily CSVs temporarily, and removes those CSVs after processing. To set up a new environment, run from the repository root:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r part1/requirements.txt
```

## Use the Results

Part 2 reads [traffic](data/processed/traffic.csv) and [forecasts](outputs/forecasts/validation.csv) to simulate scaling. Start with [the small replay example](outputs/examples/handoff.json). Part 3 combines [forecast metrics](outputs/metrics/forecast_validation.csv) with Part 2's service and cost measurements. See the [handoff guide](docs/part1-delivery.md) for schemas, timing, and evaluation rules.

The [Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) provides observed invocation counts as an arrival-intensity proxy. Part 2 simulates within-minute arrivals and service behavior for one stateless application per experiment. The dataset is released under CC-BY; cite Shahrad et al., *Serverless in the Wild*, USENIX ATC 2020.

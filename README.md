# Traffic Prediction and Autoscaling

This project investigates whether more accurate traffic forecasts actually lead to better autoscaling decisions. Online services need to adjust the number of running servers as user demand changes. Running too many servers increases cost, while running too few can lead to high latency, delayed requests, or service-level objective (SLO) violations. Our project compares forecasting models based not only on prediction error, but also on the quality and cost of the scaling decisions they produce.

## Main Research Question

Does a model with lower traffic-prediction error necessarily produce lower latency, fewer SLO violations, and lower server cost?

## Project Approach

We will use public workload traces and connect several forecasting models to a discrete-time autoscaling simulator. The simulator will model:

- Request demand over time
- Server capacity
- Server startup delays
- Request queues
- Scaling decisions
- Response latency
- Estimated server cost

We will compare a recent-demand baseline, linear regression, a tree-based model, and a version using quantile loss to penalize underprediction more heavily. We will also examine whether queue feedback can improve scaling decisions.

## Planned Experiments

1. Compare forecasting accuracy with actual service quality.
2. Study which types of prediction errors cause the most harm.
3. Test whether penalizing underprediction improves autoscaling decisions.
4. Check whether the findings change under different simulator assumptions, such as startup delays and within-minute request patterns.

## Planned Metrics

- MAE and RMSE
- p95 and p99 latency
- SLO violation rate
- Delayed or dropped requests
- Estimated server cost
- Server-minutes used

## Data

Our primary dataset is the Azure Functions Trace 2019 workload data. We will select workloads with different traffic patterns, including relatively steady and bursty workloads. The NASA HTTP log is a possible backup dataset.

## Scope

The project focuses on horizontal scaling for one stateless service using public workload traces and a simulator. Real cloud deployment, Kubernetes deployment, vertical scaling, multi-tier applications, and real provider pricing are outside the current scope.

## Current Status

Initial repository setup. The next steps are to prepare the workload data, implement the simulator, add the baseline forecasting models, and run the first end-to-end experiment.

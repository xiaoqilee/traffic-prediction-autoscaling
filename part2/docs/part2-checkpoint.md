# Part 2: Investigation and Iteration Checkpoint

## Completed experiments

We ran the simulator on the Part 1 validation handoff using four workloads and five forecasting models. We also compared forecast-driven scaling with a reactive baseline.

Before running the experiments, we fixed a simulator issue where queued requests were only dispatched at minute boundaries. Queued requests are now served whenever capacity becomes available during the simulation.

## Scaling-policy ablation

Using the fixed `tree` forecast, we compared base forecast scaling, forecast scaling with 1.5x headroom, and forecast scaling with queue feedback.

| Workload | Policy | SLO violation rate | Dropped | p99 response (s) | Server-minutes |
|---|---|---:|---:|---:|---:|
| steady_01 | forecast | 0.0000 | 0 | 1.682 | 4320 |
| steady_01 | headroom | 0.0000 | 0 | 1.682 | 4320 |
| steady_01 | queue feedback | 0.0000 | 0 | 1.682 | 4320 |
| steady_02 | forecast | 0.0000 | 0 | 1.380 | 4320 |
| steady_02 | headroom | 0.0000 | 0 | 1.380 | 4320 |
| steady_02 | queue feedback | 0.0000 | 0 | 1.380 | 4320 |
| bursty_01 | forecast | 0.4859 | 4,807 | 56.887 | 4,556 |
| bursty_01 | headroom | 0.4417 | 3,958 | 56.047 | 5,155 |
| bursty_01 | queue feedback | 0.4713 | 4,408 | 56.448 | 4,761 |
| bursty_02 | forecast | 0.0909 | 181 | 29.296 | 4,320 |
| bursty_02 | headroom | 0.0894 | 179 | 28.454 | 4,413 |
| bursty_02 | queue feedback | 0.0909 | 181 | 29.296 | 4,341 |

The main finding is that extra headroom improved service on bursty workloads, especially `bursty_01`, but required more server-minutes. Queue feedback produced a smaller improvement on `bursty_01` and had almost no effect on `bursty_02`. The policies behaved identically on the steady workloads because one active server was sufficient.

## Failure analysis

The largest queue buildup occurred for the base forecast policy on `bursty_01`. At minute 15098, demand reached 201 requests while only one server was active, causing the queue to grow to 142 requests. Additional servers began starting, but the three-minute startup delay prevented them from becoming active immediately.

The queue reached 184 requests at minute 15110 while one server was active and another was still starting. At minute 15121, demand reached 168 requests, with two servers starting but only one active server. These periods also produced dropped requests. The timeline suggests that bursty demand combined with server startup delay was the primary cause of the poor service outcome. The earlier minute-boundary dispatch issue was fixed before these results were generated.

## Changes and challenges

The simulator was updated to dispatch waiting requests whenever service capacity becomes available, rather than only at minute boundaries. We also added headroom-based scaling, queue-feedback scaling, additional integration tests, and reusable validation and ablation runners.

A challenge was separating simulator behavior from scaling-policy behavior. The initial dispatch cadence bug caused artificially high response times, so it had to be fixed before interpreting the scaling results. Another practical challenge was that request-level logs became too large for GitHub, so compact summaries, minute logs, and metrics tables are tracked while the raw request-level files remain local.

## Next experiments

The next step is to vary the startup delay while keeping the workload, forecast, and scaling policy fixed. We will also evaluate multiple random seeds and construct a broader cost-versus-SLO curve by sweeping the headroom factor.
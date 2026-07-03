# State (observation) features

Each intersection's observation is assembled from an ordered list of named
**state features** passed as `state_features=[...]` (default:
`["lane_vehicle_count", "lane_waiting", "phase_onehot"]`). The logic lives in
`states/base.py` (`StateRepresentation`); backend-specific structured
features live in `states/sumo.py`, `states/moss.py`, `states/cityflow.py`.

The resulting per-intersection vector is what agents read from
`info["intersections"][id]["state"]`.

## Feature kinds

### Built-in flat blocks

| Name | Width per intersection | Content |
|---|---|---|
| `lane_vehicle_count` | one per incoming lane | vehicles on each incoming lane |
| `lane_waiting` | one per incoming lane | halting vehicles on each incoming lane |
| `phase_onehot` | `num_phases` | one-hot of the current phase |

### Metric features

Any registered **metric name** (see [metrics.md](metrics.md)) can be added to
`state_features` and is appended as a single scalar per intersection — the
per-intersection value when the metric has a local implementation, otherwise
the global value broadcast to every intersection. The required metric is
enabled in the pipeline automatically.

```python
state_features=["lane_vehicle_count", "phase_onehot", "average_intersection_pressure"]
```

### Structured features

A structured feature produces the *entire* observation as a
multi-dimensional array and therefore must be the **only** entry in
`state_features`. Currently there is one: **`drq_norm`** (SUMO and MOSS),
which reproduces RESCO's `states.drq_norm` observation — a
`(1, n_incoming_lanes, 5)` array per intersection with, per lane:

1. whether the lane is served by the current green,
2. approaching vehicles / 28,
3. total (sticky) waiting time / 28,
4. queue length / 28,
5. sum of normalised vehicle speeds.

On SUMO it requires the sticky-wait tracker
(`resco_sticky_waiting_time_on_the_incoming_lanes`), which the env enables
automatically. `drq_norm` is what the RESCO-parity `DQNAgentPFRL` expects.

## Normalisation

`obs_norm={feature_name: divisor}` applies a static per-feature divisor to
the flat observation, e.g.:

```python
obs_norm={"lane_vehicle_count": 40.0, "lane_waiting": 40.0}
```

Divisors must be finite and non-zero, may only reference features that are
present, and cannot be applied to a structured feature (it is already
normalised).

## Validation

Structure (non-empty list, structured-must-be-sole) is checked at env
construction. Metric-name existence is checked once the metrics object is
alive: referencing an unknown feature raises a `ValueError` listing all
valid built-ins, structured features and metric names for that backend.

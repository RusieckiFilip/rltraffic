# Metrics

`metrics/` implements an opt-in, lazily evaluated measurement pipeline.
`metrics/base.py` defines the shared machinery (`BaseMetrics`); each backend
subclasses it (`CityFlowMetrics`, `SumoMetrics`, `MossMetrics`) and registers
its implementations.

## How it works

- **Opt-in by name.** The env (or you, via `metrics=[...]`) requests a set of
  metric names; only those are computed. Metrics needed by the configured
  reward functions and state features are added automatically.
- **Primitives are memoised per step.** Metrics are computed from raw engine
  reads ("primitives", e.g. vehicle speeds, per-lane waiting counts). Each
  primitive is fetched once per step and cached until `update()` clears the
  cache, so metrics sharing a primitive pay for a single engine call.
- **Episode accumulators on demand.** Metrics marked `episode=True` (running
  waiting-time totals, completed-vehicle records, ...) activate cross-step
  bookkeeping only when requested.
- **Global and local variants.** A metric may have a simulation-wide
  implementation (`@register`), a per-intersection one (`@register_local`
  returning `{intersection_id: value}`), or both under the same name.
- **Lifecycle** (driven by the env): `reset()` at episode start →
  `pre_step()` before each action (pre-action snapshots) → `on_sim_step()`
  after every individual simulation step (SUMO/MOSS accumulate per-step
  events such as arrivals that the engine only exposes for the last step) →
  `update()` after each `delta_time` window → `warmup()` once after reset so
  the first observation reflects the real t=0 state.

## Reading metrics

```python
env.metrics.compute_all()        # {name: value} for every requested global metric
env.metrics.compute_all_local()  # {intersection_id: {name: value}}
env.metrics.get("average_travel_time")
env.metrics.get_local("average_intersection_pressure")
```

`step()` also puts `compute_all()` into `info["metrics"]` and the local
values into `info["intersections"][id]["metrics"]`.

## Metric catalog

Names are long on purpose — they are self-describing keys shared across
backends.

### Available on all backends (CityFlow, SUMO, MOSS)

| Metric | Kind | Meaning |
|---|---|---|
| `average_travel_time` | global | average travel time of vehicles (completed + in progress) |
| `throughput_delta` | global | vehicles that finished their journey since the last step |
| `count_of_vehicles_completing_journey` | global, episode | completed journeys this episode |
| `total_time_of_journey` | global, episode | summed journey time of completed vehicles |
| `average_time_of_journey` | global, episode | mean journey time of completed vehicles |
| `total_sum_delays_of_all_vehicles_from_all_routes` | global, episode | Σ (real − ideal free-flow time) |
| `total_average_delays_of_all_vehicles_from_all_routes` | global, episode | mean of the above |
| `total_average_delays_real_times_by_ideal_times` | global, episode | mean of real/ideal time ratios |
| `waiting_time_all_vehicles_for_the_last_time_step_in_simulation` | global + local | accumulated waiting time of vehicles currently in the simulation (local: on the intersection's incoming lanes) |
| `total_waiting_time_all_vehicles_in_simulation_in_episode` | global, episode | waiting time integrated over the episode |
| `total_waiting_time_on_the_incoming_lanes_in_episode` | global, episode | same, restricted to controlled incoming lanes |
| `number_of_all_halting_vehicles_for_the_last_time_step_in_simulation` | global + local | halting (speed < threshold) vehicle count (local: incoming lanes only) |
| `average_intersection_pressure` | global + local | incoming − outgoing vehicle counts, averaged over intersections |
| `calculate_average_delta_of_delays_after_action` | global | mean change in per-vehicle delay across the last action (uses the `pre_step` snapshot) |

### SUMO / MOSS only

| Metric | Kind | Meaning |
|---|---|---|
| `resco_sticky_waiting_time_on_the_incoming_lanes` | local | RESCO `Signal.observe`-parity waiting time: a vehicle starts counting at its first stop near the junction, keeps accruing while detectable (even when crawling), and is forgotten when it leaves. Backs the `sticky_wait_norm` reward and the `drq_norm` state. |

### MOSS only (engine bookkeeping)

`scheduled_due_vehicle_count`, `started_vehicle_count`,
`running_vehicle_count`, `pending_departure_vehicle_count`,
`not_started_vehicle_count`, `teleport_count`.

## Adding a new metric

Register a method inside the backend's metrics class (or `BaseMetrics` for a
generic one):

```python
from metrics.base import register, register_local

class SumoMetrics(BaseMetrics):
    @register("my_metric")                       # global variant
    def _my_metric(self) -> float:
        speeds = self._vehicle_speeds()          # memoised primitive
        ...

    @register_local("my_metric")                 # optional per-intersection variant
    def _my_metric_local(self) -> dict[str, float]:
        ...
```

Use `episode=True` in the decorator when the metric needs cross-step
accumulators, initialise them in `_init_episode_state()`, and update them in
`_run_step_hooks()` / `on_sim_step()`. The metric is then immediately usable
as a reward ingredient, a state feature, or a reported value.

# Rewards

`rewards.py` is the single registry of reward functions. A reward function
takes a metrics dict (`{metric_name: float}`) and returns a float. Envs are
configured by **name** (or with a custom callable):

```python
env = SumoEnv(..., global_reward_fn="queue_length",
              local_reward_fn="presslight", global_reward_weight=0.0)
```

Each named reward declares which metrics it needs
(`required_metrics_for_reward`), and the env enables those metrics
automatically. Custom callables can't declare needs, so they require an
explicit `metrics=[...]` list on the env.

## Global rewards (`global_reward_fn`)

Drive the scalar returned by `env.step()`. Default: `queue_length`.

| Name | Definition | Intuition |
|---|---|---|
| `queue_length` | −(halting vehicles in the whole simulation) | fewer stopped vehicles is better |
| `average_travel_time` | −(average travel time) | faster trips are better |
| `pressure` | average intersection pressure (incoming − outgoing counts) | raw pressure, larger is better |
| `presslight` | −\|average intersection pressure\| | PressLight (Wei et al., KDD 2019); minimising \|pressure\| maximises throughput |
| `throughput` | vehicles that completed their journey since the last step | more arrivals is better |
| `combined` | `queue_length + 0.1·throughput + 0.01·average_travel_time` | balanced objective |
| `wait_norm` | `clip(−total_wait / 224, −12, 12)` | RESCO-style normalised waiting time, global approximation |

## Per-intersection rewards (`local_reward_fn`)

When `local_reward_fn` is set, every intersection *j* additionally receives

```
info["intersections"][j]["reward"] =
    global_reward_weight * global_reward + local_reward_fn(local_metrics[j])
```

so multi-agent learners (DQN/IPPO/MAPPO all prefer per-intersection rewards
when present) get local credit assignment, optionally mixed with a shared
global signal via `global_reward_weight`.

Only rewards whose required metrics have per-intersection implementations
are allowed here:

| Name | Definition (per intersection) |
|---|---|
| `queue_length` | −(halting vehicles on that intersection's incoming lanes) |
| `pressure` | intersection pressure |
| `presslight` | −\|intersection pressure\| |
| `wait_norm` | RESCO's per-junction `clip(−wait / 224, −4, 4)` using accumulated waiting time of vehicles currently on the incoming lanes |
| `sticky_wait_norm` | same clip, but on the **sticky** wait tracker that reproduces RESCO's `Signal.observe` bookkeeping exactly (a vehicle counts from its first stop near the junction, keeps accruing while detectable, is forgotten when it leaves) |

Note the deliberate scope difference for `queue_length`: the local variant
counts only the intersection's own incoming lanes, while the global variant
counts every halting vehicle in the simulation.

### RESCO normalisation constants

RESCO divides per-junction waiting time by **224** and clips to **[−4, 4]**.
The global `wait_norm` uses the same divisor with a **[−12, 12]** clip
(224 × 3 junctions of the BB5B benchmark). Clipping the sum differs from
summing per-junction clips when junctions are imbalanced — use
`local_reward_fn="wait_norm"` (or `sticky_wait_norm`) for faithful RESCO
semantics.

## Custom reward functions

Pass any callable (global: receives the global metrics dict; local: receives
one intersection's local metrics dict) and list the metrics it reads:

```python
def my_reward(metrics):
    return -metrics["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"]

env = CityFlowEnv(
    ...,
    global_reward_fn=my_reward,
    metrics=["waiting_time_all_vehicles_for_the_last_time_step_in_simulation"],
)
```

Omitting the `metrics` list with a callable reward raises a `ValueError` at
construction. If a `local_reward_fn` needs a metric with no per-intersection
implementation on the active backend, that also fails fast with a clear
message.

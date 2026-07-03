# Architecture

## Big picture

The project separates *what the agent controls* from *which simulator runs the
traffic*. Everything the training loop touches is backend-neutral:

```
                 ┌────────────────────────────────────────────┐
                 │           experiments/ (runner)            │
                 │  config.json → env × agent × seed matrix   │
                 └──────────────┬─────────────────────────────┘
                                │  act(info) / step(action) / observe(...)
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼────────┐                              ┌───────▼────────┐
│    agent/      │                              │  algorithms/   │
│ DQN IPPO MAPPO │                              │  MaxPressure   │
└───────┬────────┘                              └───────┬────────┘
        │            info["intersections"][id]          │
┌───────▼────────────────────────────────────────────────▼──────┐
│                envs/base_traffic_env.BaseTrafficEnv           │
│   phase control · reward resolution · state building · info   │
└──────┬──────────────────────┬──────────────────────┬──────────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│ CityFlowEnv │        │   SumoEnv   │        │   MossEnv   │
│  cityflow   │        │ traci/libsumo│       │  moss (GPU) │
└──────┬──────┘        └──────┬──────┘        └──────┬──────┘
       │                      │                      │
┌──────▼──────┐        ┌──────▼──────┐        ┌──────▼──────┐
│CityFlowMetrics│      │ SumoMetrics │        │ MossMetrics │
└─────────────┘        └─────────────┘        └─────────────┘
```

Three cross-cutting registries keep the vocabulary consistent across
backends:

- **Rewards** (`rewards.py`) — named reward functions plus the metrics each
  one requires.
- **Metrics** (`metrics/`) — named measurements registered per backend with
  `@register` (global) / `@register_local` (per intersection).
- **State features** (`states/`) — named observation blocks; flat built-ins
  plus per-backend *structured* features.

An env is configured with names (`global_reward_fn="queue_length"`,
`state_features=["lane_vehicle_count", ...]`) and resolves them against these
registries at construction time, failing fast on anything unknown.

## Data flow of one decision step

1. The agent picks one action per intersection from
   `info["intersections"][id]["avail_actions"]`.
2. `BaseTrafficEnv.step(action)`:
   - `pre_step()` lets metrics snapshot pre-action vehicle state.
   - The configured **phase control** translates each action into a
     `PhasePlan` — a sequence of `PhaseSegment`s (e.g. *yellow 3 s → all-red
     2 s → green 5 s*) whose durations sum exactly to `delta_time`.
   - The plans are executed in lockstep across intersections; the backend
     renders each segment (`_set_phase` / traffic-light state string) and
     advances the engine.
   - `metrics.update()` refreshes the per-step cache and episode
     accumulators.
   - The global reward is computed from the metrics dict; `_get_info()`
     assembles the new `info`.
3. The agent stores the transition via `observe(next_info, reward, ...)`.

## The `info` dict contract

Agents never touch `observation_space` directly; the whole interface is
the `info` dict returned by `reset()` and `step()`:

```python
info = {
    "sim_time": 120.0,
    "vehicle_count": 87,
    "average_travel_time": 45.3,
    "lane_vehicle_count": {lane_id: int, ...},
    "lane_waiting_vehicle_count": {lane_id: int, ...},
    "step": 12,
    "metrics": {metric_name: float, ...},          # requested global metrics
    "intersections": {
        "intersection_id": {
            "state": [ ...float vector or nested array... ],
            "avail_actions": [0, 2],               # legal actions right now
            "current_phase": 1,
            "time_in_phase": 20,
            "action_applied": True,                # last action changed phase
            "metrics": {metric_name: float, ...},  # per-intersection metrics
            "reward": -1.2,                        # only if local_reward_fn set
        },
        ...
    },
}
```

This is why the experiment runner's loop is three lines: environments compute
rewards internally, and agents read `info["intersections"]` without any
adapter layer.

## Backend-neutral topology model

`utils/common_utils.py` defines the two dataclasses every backend must
populate when parsing its road network:

- `RoadnetInfo` — lane/road/intersection IDs, road lengths and speed limits.
- `IntersectionInfo` — per controllable intersection: incoming/outgoing lane
  IDs, phase count, per-phase active road links
  (`phase_roadlink_mapping`), configured phase durations, raw phase state
  strings (SUMO), and per-roadlink `(incoming_lanes, outgoing_lanes)` pairs.

CityFlow parses its roadnet JSON (`utils/cityflow_utils.py`), SUMO queries a
live TraCI connection (`utils/sumo_utils.py`), and MOSS translates its
protobuf map — all into this same vocabulary, so pressure computations,
state features and local metrics are written once.

## Design decisions worth knowing

- **Non-standard Gym signature.** `reset()` returns `info` only, `step()`
  returns `(reward, terminated, truncated, info)` — there is no `obs`.
  Per-intersection state lives in `info` because multi-agent learners need
  one vector per intersection, not a single flat observation.
- **Lazy heavy imports.** `experiments/config.py` and
  `experiments/registry.py` import neither torch nor any simulator, so
  `--dry-run` and config validation are instant. Backends are imported inside
  `experiments/envs.make_env()`; a missing engine turns the affected cell
  into a `skipped` result instead of crashing the run.
- **Metrics are opt-in and memoised.** Only requested metrics are computed;
  raw engine reads (primitives) are cached per step so metrics sharing a
  primitive pay for one engine call (see [metrics.md](metrics.md)).
- **Phase safety is enforced centrally.** `BaseTrafficEnv` validates every
  `PhasePlan` (positive durations, phases in range, total == `delta_time`)
  and raises on illegal actions, so an agent bug cannot silently produce an
  invalid signal program.
- **Reproducibility.** Gymnasium seeding convention: `reset(seed=X)` reseeds
  the env RNG; each reset draws a fresh engine seed from it that is passed to
  the backend (CityFlow `set_random_seed`, SUMO `--seed`). Seed once and you
  get a reproducible *sequence* of varied episodes.

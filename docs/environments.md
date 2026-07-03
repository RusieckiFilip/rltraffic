# Environments

All environments derive from `envs.base_traffic_env.BaseTrafficEnv` and share
one API, one action/observation model, and one configuration surface. The
backend subclasses only implement engine plumbing: booting the simulator,
advancing it, rendering a phase, and building their metrics object.

## Common API

```python
env = SomeEnv(..., max_steps=360, delta_time=10)

info = env.reset(seed=42)          # returns info (NOT (obs, info))
reward, terminated, truncated, info = env.step(action)
env.close()
```

- `max_steps` — number of *agent decisions* per episode (episode length in
  simulation seconds is `max_steps * delta_time`). `truncated` becomes true
  after `max_steps` steps; `terminated` is always false.
- `delta_time` — simulation seconds between consecutive decisions. Every
  phase plan (including transition segments) must fill exactly this window.
- `action` — an `int` (single intersection) or array of ints, one per
  controllable intersection, in `env.intersections` order. Illegal actions
  (not in that intersection's current `avail_actions`) raise `ValueError`.
- `env.intersections` — list of `IntersectionInfo` for all controllable
  (non-virtual) intersections.
- `action_space` is `Discrete(n)` for one intersection and
  `MultiDiscrete([...])` otherwise; `observation_space` is a `Box` describing
  the concatenated flat state (or the structured feature's shape). These
  exist for compatibility — agents actually consume per-intersection state
  from `info` (see [architecture.md](architecture.md)).

### Constructor parameters (shared by all backends)

| Parameter | Default | Meaning |
|---|---|---|
| `max_steps` | `3600` | decisions per episode |
| `delta_time` | `10` | seconds per decision |
| `global_reward_fn` | `"queue_length"` | name or callable; drives `step()`'s scalar reward |
| `local_reward_fn` | `None` | optional per-intersection reward (see [rewards.md](rewards.md)) |
| `global_reward_weight` | `1.0` | weight of the global part inside local rewards |
| `phase_control_cls` | `AcyclicPhases` | action semantics (see [phase-control.md](phase-control.md)) |
| `phase_bounds` | `None` | `(n_intersections, max_phases, 2)` min/max phase durations |
| `metrics` | `None` | explicit metric names to enable (usually derived automatically) |
| `state_features` | `None` | observation composition (see [states.md](states.md)) |
| `obs_norm` | `None` | per-feature static divisors for the flat observation |

Metrics required by the chosen reward functions and state features are
enabled automatically; you only pass `metrics=[...]` for extra measurements
or when using a custom callable reward (whose needs cannot be inferred).

### Seeding

Gymnasium convention, implemented once in the base class:

- `reset(seed=X)` reseeds the env's NumPy RNG; plain `reset()` continues the
  existing stream.
- Every reset draws a fresh **engine seed** from that RNG and applies it to
  the backend (CityFlow `Engine.set_random_seed`, SUMO `--seed`; MOSS takes
  its seed at construction).
- Same seed every reset → identical episodes. Seed once → a varied but
  reproducible episode sequence.

## CityFlowEnv (`envs/cityflow_env.py`)

Wraps the vendored CityFlow engine (`pip install ./CityFlow`).

```python
CityFlowEnv(cityflow_config_path="configs/sim/cityflow1x1.json", thread_num=1, ...)
```

- `cityflow_config_path` points at a CityFlow engine config JSON
  (`dir`, `roadnetFile`, `flowFile`, ...). The env rewrites it into a temp
  copy with absolute paths so the C++ engine works from any CWD; replay log
  paths are resolved relative to the project root.
- `thread_num` controls engine threads.
- CityFlow cannot render yellow lights and only switches phases by index, so
  when the phase control needs a clearance between greens the env writes an
  engine-only roadnet copy with an extra **all-red phase** appended to every
  traffic light (the observation/action spaces never see it).
- `reset()` reuses the engine (`Engine.reset()` + reseed) instead of
  recreating it.

## SumoEnv (`envs/sumo_env.py`)

Wraps Eclipse SUMO through TraCI (default) or libsumo (`libsumo=True`,
in-process, faster, no GUI).

```python
SumoEnv(sumocfg_path="scenarios/bb5b/BB5B.sumocfg", gui=False, libsumo=False, ...)
```

- Validates that the `.sumocfg` and its referenced input files exist before
  starting; TraCI startup retries are raised to 120 for large networks.
- Topology (`RoadnetInfo`) is extracted live from the running simulation on
  the first boot.
- Transitions between greens are rendered natively: the departing green
  turns **yellow for 3 s**, then **all-red for 2 s**, via
  `setRedYellowGreenState` (in transition mode every segment, greens
  included, is sent as an explicit state string because pinning the TLS to an
  online program makes `setPhase` unreliable afterwards).
- `gui=True` runs `sumo-gui` (mutually exclusive with `libsumo`).

## MossEnv (`envs/moss_env.py`)

Wraps MOSS, a GPU-accelerated simulator (Tsinghua FibLab). Inputs are
protobuf files produced by `mosstool`:

```python
MossEnv(map_file="....map.pb", person_file="....person.pb", device=0, ...)
```

Extra parameters beyond the shared set:

| Parameter | Default | Meaning |
|---|---|---|
| `name` | `"moss"` | task name (output dir naming) |
| `step_interval` | `1.0` | engine sub-step seconds; must divide `delta_time` |
| `start_step` | `0` | simulation start offset |
| `seed` | `43` | engine seed |
| `device` | `0` | CUDA device index |
| `output_dir` | `None` | AVRO output dir (`None` disables file output) |
| `speed_stat_interval` | `0` | enable engine speed statistics (>0) |
| `junction_yellow_time` | `0.0` | extra engine-side clearance between phases |
| `verbose_level` | `"NO_OUTPUT"` | engine verbosity |
| `manual_control` | `True` | drive traffic lights via `TlPolicy.MANUAL` |

- One engine instance is created at construction; `reset()` rewinds through a
  **checkpoint** instead of recreating it.
- Raw engine identifiers are translated into the same string-keyed
  intersection/lane vocabulary as the other backends, so all shared code
  works unchanged.
- Like CityFlow, MOSS cannot render yellow, so clearances use an injected
  all-red phase.

## Availability and optional imports

`envs/__init__.py` imports `CityFlowEnv` defensively: when the `cityflow`
module is missing the name is exported as `None` instead of raising.
`SumoEnv` and `MossEnv` import their engines lazily inside
`_init_simulator()`, so the classes are always importable. The experiment
framework additionally probes availability up front
(`experiments/envs.backend_ready`) and marks cells as `skipped` when an
engine or scenario file is missing.

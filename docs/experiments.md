# Experiment framework

`experiments/` runs a **comparison matrix** described by a single JSON
config: `environments × agents × seeds`. For every cell the runner trains
the agent, evaluates it, evaluates the configured baselines on the *same*
traffic, and aggregates results across seeds into a table, plots and a
report.

```bash
python experiments/run.py experiments/configs/smoke.json             # full run
python experiments/run.py experiments/configs/smoke.json --dry-run   # validate + print plan only
python experiments/run.py experiments/configs/example_cologne.json --workers 4
python experiments/run.py experiments/configs/smoke.json --from-checkpoint output/checkpoints/smoke
python experiments/run.py ... --no-plot                              # skip PNG plots
```

- `--dry-run` validates the config and prints the planned matrix **without
  importing any backend** (no torch, no simulators) — it is instant.
- `--workers N` runs (env, seed) cells in parallel as separate processes.
  Cells are independent; a worker crash degrades to an `error` cell instead
  of sinking the run.
- An unavailable backend (engine not installed, scenario file missing) is
  **skipped** with a reason, not fatal.
- `--from-checkpoint DIR` loads previously saved agent checkpoints, skips
  training, and only runs evaluation + baselines.

## Config format

```jsonc
{
  "name": "cologne_compare",            // run name (results directory)
  "seeds": [7, 8, 9],                   // each seed = an independent repetition
  "output_dir": "output/experiments",   // optional; results in <output_dir>/<name>
  "checkpoint_dir": "output/checkpoints", // optional; .pt files in <checkpoint_dir>/<name>

  "defaults": { ... },                  // shared cell settings (see table below)

  "environments": [                     // any mix of backends + scenarios (+ overrides)
    { "id": "cologne3", "backend": "cityflow", "config": "../../configs/sim/cityflow_cologne3.json" },
    { "id": "grid4x4",  "backend": "cityflow", "config": "../../configs/sim/cityflow_grid4x4.json",
      "overrides": { "max_steps": 240 } },
    { "id": "sumo1",    "backend": "sumo", "config": "../../scenarios/bb5b/BB5B.sumocfg",
      "overrides": { "libsumo": true } },
    { "id": "moss1",    "backend": "moss", "map_file": "...", "person_file": "..." }
  ],

  "agents": [                           // registry type + hyperparameter overrides
    { "id": "dqn",   "type": "dqn",   "params": { "lr": 0.001 } },
    { "id": "ippo",  "type": "ippo",  "params": { "lr": 0.0003 } },
    { "id": "mappo", "type": "mappo", "params": { "lr": 0.0003 } }
  ]
}
```

- `config` / `map_file` / `person_file` paths are resolved **relative to the
  config file**.
- `backend`: `cityflow` and `sumo` use the `config` key; `moss` uses
  `map_file` + `person_file`.
- Every environment entry may override any `defaults` key in `overrides`
  (different `max_steps`, `control_mode`, `global_reward_fn`, ... per
  environment), which lets one matrix mix backends and scenarios freely.

Validation is strict and up front: unknown keys, unsupported
backends/rewards/control modes, bad hyperparameters, duplicate ids and
invalid setting combinations (e.g. acyclic control with `delta_time <= 5`,
`gui` together with `libsumo`) all raise a clear `ValueError` before any
simulation starts.

### `defaults` / `overrides` keys

| Key | Meaning | Default |
|---|---|---|
| `train_episodes`, `eval_episodes` | training / evaluation protocol | 10 / 2 |
| `max_steps`, `delta_time` | episode length and decision step [s] | 360 / 10 |
| `control_mode` | `acyclic` / `acyclic_bounded` / `cyclic` / `resco_cyclic` (see [phase-control.md](phase-control.md)) | `acyclic` |
| `global_reward_fn` | reward name (see [rewards.md](rewards.md)) | `queue_length` |
| `local_reward_fn`, `global_reward_weight` | per-intersection reward | `null` / `1.0` |
| `state_features` | observation composition (see [states.md](states.md)) | `[lane_vehicle_count, lane_waiting, phase_onehot]` |
| `obs_norm` | per-feature observation divisors | `null` |
| `metrics` | explicit metric list (usually derived from reward/state) | `null` |
| `device` | `cpu` / `cuda` / `mps` | `cpu` |
| `compare_with` | baselines: `random`, `max_pressure` (`[]` = none) | `[random, max_pressure]` |
| `thread_num` (cityflow), `gui` / `libsumo` (sumo) | backend knobs | 1 / `false` / `false` |

Agent types: `dqn`, `ippo`, `mappo` (registry in `experiments/registry.py`;
defaults documented in [agents.md](agents.md)). Only hyperparameters known to
the type may be overridden — an unknown key is a hard error. The
`epsilon_decay_steps: null` sentinel for DQN resolves to the full training
budget (`train_episodes × max_steps`) so short experiments still anneal
exploration to the end.

## Seeding protocol

Per cell (env × seed): training episodes reset with `seed + episode`;
**every policy** (agents and baselines) is evaluated with the same env-reset
seed (`seed + 10_000 + episode`) so the comparison is paired on identical
traffic; a baseline's own action RNG gets a separate offset that does not
perturb the env episode sequence.

## Outputs

In `<output_dir>/<name>/`:

- `results.json` — full report: config echo, every cell (status, per-policy
  metrics, train returns, timings), and aggregates;
- `summary.csv` — one row per environment × policy with `mean` / `std` per
  metric;
- `plots/<env>.png` — per-environment comparison bar plots (needs
  `matplotlib`; `--no-plot` disables);
- `plots/<env>_learning_curve.png` — mean training-return curve across
  seeds.

Reported metrics: `episode_reward` (↑), `average_travel_time` (↓),
`final_vehicle_count` (↓), `average_waiting_queue` (↓). The console
leaderboard sorts policies by `average_travel_time`.

## Checkpoints

With `checkpoint_dir` set, the runner saves each trained agent per
(environment × agent × seed) as
`<checkpoint_dir>/<name>/<env>__<agent>__seed<S>.pt`. `--from-checkpoint DIR`
loads those files, skips training and runs evaluation + baselines only.

## How a cell runs

Environments compute rewards internally, so the per-cell loop is minimal:

```python
action = agent.act(info)                       # explore=True during training
reward, terminated, truncated, info = env.step(action)
agent.observe(info, reward, terminated, truncated)
```

The env's `info` dict goes to the agent unchanged (agents read
`info["intersections"][id]` directly), which is what keeps the runner free
of backend- or agent-specific adapters.

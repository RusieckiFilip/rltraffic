# FROZEN INTERFACE CONTRACTS

Source of truth: `docs/PROJECT_PLAN.md` §4. Copied here so every task can read it without the plan.
**These are frozen. They are not bugs. Do not "fix" them.** Changing anything here requires a
Decisions-Log entry in the plan, written by the Master chat — never by an implementation task.

Contract version: **v1.1** (2026-07-25)

---

## C1. Environment API (non-standard Gym)

```python
info = env.reset(seed=42)                                 # returns info ONLY — no observation
reward, terminated, truncated, info = env.step(action)    # reward FIRST, then flags, then info
```

- `action`: `np.ndarray`, one int per intersection, **ordered by `[ix.id for ix in env.intersections]`**.
- `env.max_steps`, `env.delta_time` exist.
- `env.intersections`: list of objects with `.id`, `.incoming_lanes`, ...
- The env raises on illegal actions. Action masking is mandatory, not optional.

## C2. The `info` dict (returned by both `reset` and `step`)

```python
info = {
    "sim_time": float, "vehicle_count": int, "step": int,
    "average_travel_time": float,
    "lane_vehicle_count": {lane_id: int},
    "lane_waiting_vehicle_count": {lane_id: int},
    "metrics": {metric_name: float},                 # global metrics (opt-in set)
    "intersections": {
        ix_id: {
            "state": [...],            # per-intersection observation vector
            "avail_actions": [0, 2],   # legal actions RIGHT NOW
            "current_phase": int,
            "time_in_phase": int,
            "action_applied": bool,
            "metrics": {name: float},
            "reward": float,           # PRESENT ONLY IF local_reward_fn is set — detect by key absence
        }, ...
    },
}
```

## C3. Agent API

```python
class MyAgent(BaseAgent):
    def __init__(self, gym_env, ..., device=None, seed=None)      # reads env.intersections
    def act(self, info, explore=True, update_memory=True) -> np.ndarray
    def observe(self, next_info, reward, terminated, truncated=False) -> dict
    def save(self, path); def load(self, path)
```

`MaxPressureAgent` (`algorithms/max_pressure.py`) exposes act-style selection without learning —
check its exact signature in the file and adapt rather than assuming.

## C4. Helpers — USE, do not reimplement (`agent/utils/utils.py`)

`Utils.infer_action_counts(env.action_space, intersections)` ·
`Utils.extract_per_intersection_info(info, ids)` ·
`Utils.state_from_info(ix_payload)` ·
`Utils.extract_valid_actions(ix_payload, n_actions)` ·
`Utils.scalar_reward(reward)` ·
`Utils.reward_for_intersection(...)` ·
`Utils.resolve_device(device)` ·
`Utils.seed_everything(seed)`

## C5. Coding standards

Python ≥3.12 · `from __future__ import annotations` + full type hints · numpy `float32` for stored
float arrays, `int64` for actions, `bool_` for masks · repo-style docstrings · no new heavy deps ·
`pytest` for everything · **no modifications to frozen files**.

## C6. Offline data format (v1.0 — governs everything in `offline/`)


**Verified fact this contract rests on** (`envs/base_traffic_env.py::step()`, checked 2026-07-25):
`step()` applies phases → **advances the simulation** → refreshes metrics → computes the reward → builds
`info`. So both the reward and the `info` returned by `step t` describe the state **after** step `t`,
including `info["intersections"][j]["reward"]`, which is `r_t^j`.

### Alignment convention

```
OBSERVATIONS  → T+1 rows.  Everything read out of an `info` dict.
DECISIONS     → T   rows.  The actions.
OUTCOMES      → T   rows.  The rewards those actions produced.

row t   of an observation array = the state the agent saw before decision t
row t+1 of an observation array = the post-step state that r_t was computed from
```

Every `info` handed to the logger is recorded exactly once, when it arrives: `on_reset` writes
observation row 0, each `on_step_result` writes outcome row `t` and observation row `t+1`. Termination is
not a special case. `on_action` writes only the action.

### Layout

- One compressed `.npz` per episode + one `manifest.json` per collection run, updated atomically.
- **Observations (T+1 rows).** Per intersection: `state` (float32), `avail_mask` (bool, width =
  that intersection's `n_actions`), `current_phase` (int64), `time_in_phase` (float32).
  Global: `vehicle_count`, `sim_time`, `step`, `metrics (T+1, M)` + `metric_keys`.
  Per lane: `lane_vehicle_count (T+1, L) int32`, `lane_waiting_vehicle_count (T+1, L) int32`,
  `lane_ids (L,)` — `info` key names used unchanged. **The lane arrays are what make the corpus
  reward-agnostic:** any standard TSC reward is recomputable offline, so the primary-reward decision
  never forces corpus regeneration.
- **Decisions (T rows).** Per intersection: `action` (int64).
- **Outcomes (T rows).** Global: `global_reward` (float32). Per intersection: `local_reward` (float32,
  read from the `info` returned by the step, NaN-filled when the env has no `local_reward_fn`).
- Episode scalars: `episode_length` (= T), `terminated`, `truncated`, `engine_seed`, `flow_draw`
  (`-1` when absent).
- Manifest per-episode entry: `filename`, `episode_length`, `total_global_reward`, `engine_seed`,
  `flow_draw`, `episode_sha256` (sha256 over concatenated `action` + `global_reward` bytes, computed at
  write time — consumed by the P2.4 duplicate detector).
- Manifest run-level: `format_version`, `lane_count`, `lane_ids_sha256`, repo git hash, plus the
  caller's `run_metadata`.
- Intersections may have different `state_dim` and `n_actions`. **Never pad across intersections.**
- The lane-id set and metric-key set are frozen at the first write of an episode; a mid-episode change
  is an error, not something to reindex around.
- Any change to this layout requires bumping the format version and writing a migration note.
  The P2.4 linter must hard-fail on an unknown format version.

---

## C8 — A MAPPO checkpoint FREEZES the env's global metric set (added 2026-08-06)

`MAPPOAgent._build_global_features` (`agent/MAPPOAgent.py:619-638`) builds the centralised critic's
global feature vector from **every** key in `info["metrics"]`:

```python
if self._global_metric_keys is None:
    self._global_metric_keys = sorted(metrics.keys())
feats = [step / max_steps, vehicle_count]
feats.extend(float(metrics.get(key, 0.0)) for key in self._global_metric_keys)
```

so `global_feature_dim = 2 + len(metrics)`. **The metric set is therefore part of MAPPO's MDP, not
observability sugar.**

**Binding consequences.**
1. **Any env running a MAPPO checkpoint must expose exactly the metric set it was trained with.**
   Adding or removing one raises `ValueError: Global feature size changed for MAPPO: expected N,
   got M`. This affects **`act()` as well as training** (`:682`), so *evaluation and offline
   collection break identically*.
2. **A same-width swap is SILENT and worse.** The guard compares only the width. Replacing metric A
   with metric B leaves the width unchanged, so the critic reads different semantics with no error.
   `_global_metric_keys` is **not** stored in the checkpoint (its keys are `steps_done`, `learner`),
   so a loaded agent re-freezes from whatever env it is handed.
3. **`info["average_travel_time"]` is top-level and independent of the requested metric set**
   (verified live 2026-08-06 with a 1-metric env). It is therefore the safe way to carry ATT into the
   corpus without perturbing MAPPO.

**Rule:** the corpus metric set is frozen for the lifetime of the checkpoints collected against it.
Changing it invalidates every MAPPO tier. Record the metric set in the manifest and have the linter
assert homogeneity across a corpus.

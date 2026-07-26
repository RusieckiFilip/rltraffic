# BRIEF #1 (v2, CONSOLIDATED) — Trajectory Logger for Offline RL Dataset Collection
**Task IDs:** P1.1, P1.2, P1.3 · **Contracts:** v1.1 (`docs/CONTRACTS.md`) · **Format version:** `"1.0"`
**Issued:** 2026-07-25, Master Coordination Chat · **Merge gate:** P1.5 independent review must pass before merge to `main`

> **This document supersedes and replaces:** Brief #1 (2026-07-08), Addendum A (2026-07-11),
> `ADDENDUM_A_PATCH.md`, and `BRIEF_01_DELTA.md`. Do not read those — they disagree with this one in
> places, and reconciling four documents is exactly how off-by-one bugs get frozen into a data format.
> This is the single source of truth for the task.

---

## HOW TO RUN THIS TASK

**Mode: Claude Code, in the repo.** Branch `task/p1-logger`. Read the repo paths listed below from disk —
do not infer signatures. Commit before finishing. Return Packet goes to `docs/returns/P1.md` using
`docs/returns/TEMPLATE.md` (real pytest output, real `git diff --stat`, honest checklist).

**Read these files first:** `agent/base.py`, `agent/utils/utils.py`, `agent/MAPPOAgent.py`,
`algorithms/max_pressure.py`, `envs/base_traffic_env.py` (especially `step()`), `rewards.py`,
`docs/CONTRACTS.md`, `CLAUDE.md`.

**If this brief conflicts with the repo, the repo wins** — implement to the repo and flag the conflict
in the Return Packet. This is the most important line in this document.

## SCOPE — what you are and are not building

You are building the component that records trajectories to disk while an existing policy runs.
You are **not** building: the Decision Transformer, the dataset loader, the flow randomizer, the corpus
linter, or any new control policy. If you find yourself designing a fixed-time controller, stop — that
is a separate task (P2.5), deliberately excluded here.

---

## THE ALIGNMENT CONVENTION (read twice; everything else is bookkeeping)

Verified against `envs/base_traffic_env.py::step()` on 2026-07-25. The order inside `step()` is:
apply phases → **advance simulation** → refresh metrics → compute reward → build `info`. Therefore
**both the reward and the `info` returned by `step t` describe the state *after* step `t`.**

Three consequences that drive the whole format:

1. `r_t` is recoverable from lane counts **only in the row after the one aligned with `a_t`**.
2. `info["intersections"][j]["reward"]` returned by `step t` is `r_t^j` — the reward for the step just
   taken. It must be recorded as an **outcome of `t`**, never bundled with the observation read before
   the decision. Getting this wrong shifts every per-intersection return by one step and silently
   corrupts every returns-to-go the MADT trains on.
3. There is one more observation than there are decisions.

### The rule, stated once

```
OBSERVATIONS  → T+1 rows.  Everything read out of an `info` dict.
DECISIONS     → T   rows.  The actions.
OUTCOMES      → T   rows.  The rewards produced by those actions.

row t   of an observation array = the state the agent saw before decision t
row t+1 of an observation array = the post-step state that r_t was computed from
```

### Where each write happens (one write site per array — no special-casing termination)

| Callback | Receives | Writes |
|---|---|---|
| `on_reset(info, ...)` | `info_0` | observation row **0** |
| `on_action(info, action)` | `info_t`, `a_t` | decision row **t** (the action only) |
| `on_step_result(reward, term, trunc, next_info)` | `r_t`, `info_{t+1}` | outcome row **t** + observation row **t+1** |

Every `info` the logger is handed gets recorded exactly once, at the moment it arrives. Row `T` is not
a special terminal case — it is just the last `info`, written by the last `on_step_result`. Do not add a
"if terminated, also write" branch; the uniform rule makes it unnecessary and removes the failure mode
where an episode that exits the loop without `terminated/truncated` loses its final row.

`on_action` deliberately writes **nothing** derived from `info` — that info was already recorded by the
previous callback. Instead, assert it is the expected one (compare `info["step"]` against what was last
recorded) and raise on mismatch. That assertion is free and catches loop misuse immediately.

---

## FILE 1 — `offline/trajectory_logger.py`

### Usage (this exact loop shape)

```python
logger = TrajectoryLogger(env, out_dir="datasets/hz1x1_maxpressure", run_metadata={...})
info = env.reset(seed=seed)
logger.on_reset(info, engine_seed=seed, flow_draw=None)
for _ in range(env.max_steps):
    action = policy_act(info)
    logger.on_action(info, action)
    reward, terminated, truncated, info = env.step(action)
    logger.on_step_result(reward, terminated, truncated, info)
    if terminated or truncated:
        break
episode_path = logger.finalize_episode()
```

### Requirements

1. **Wrapper/callback pattern, zero env edits.** The logger observes; it never calls `env.step` itself.
2. **Read only from the `info` dict. Zero additional engine or metric calls.** State readout is ~98% of
   simulator call volume in this platform; the logger must not add to it.
3. **Observation arrays (T+1 rows), written at `on_reset` + each `on_step_result`:**
   - per intersection: `state` (float32, via `Utils.state_from_info`), `avail_mask` (bool, width =
     that intersection's `n_actions` from `Utils.infer_action_counts`, True = legal),
     `current_phase` (int64), `time_in_phase` (float32)
   - global: `vehicle_count`, `sim_time`, `step`, `metrics (T+1, M)` + `metric_keys` (key order frozen
     on first write, like `MAPPOAgent._build_global_features` does)
   - per lane: `lane_vehicle_count (T+1, L) int32`, `lane_waiting_vehicle_count (T+1, L) int32`,
     plus `lane_ids (L,)` string array (sorted lexicographically, frozen on first write).
     **Use the `info` keys unchanged as array names** — no truncation, no renaming.
     These lane arrays are what make the corpus reward-agnostic: any standard TSC reward
     (queue, PressLight, pressure) is recomputable offline. Always on; no flag.
4. **Decision arrays (T rows), written at `on_action`:** per intersection `action` (int64).
5. **Outcome arrays (T rows), written at `on_step_result`:** global `global_reward` (float32, via
   `Utils.scalar_reward`); per intersection `local_reward` (float32, read from
   `next_info["intersections"][id]["reward"]`; `np.nan` for the whole episode when the env has no
   `local_reward_fn` — detect by key absence).
6. **State machine.** Enforce the legal callback order (`on_reset` → (`on_action` → `on_step_result`)* →
   `finalize_episode`). Raise on any violation — e.g. `on_step_result` before `on_action`, or
   `on_action` twice in a row. Raise if the lane-id set or metric-key set changes mid-episode
   (silent reindexing would corrupt offline reward recomputation).
7. **On-disk format — one compressed `.npz` per episode:**
   - `ix_ids` string array, sorted once from `[ix.id for ix in env.intersections]`; per-intersection
     arrays named `ix{i}_state`, `ix{i}_avail_mask`, `ix{i}_current_phase`, `ix{i}_time_in_phase`
     (T+1 rows), `ix{i}_action`, `ix{i}_local_reward` (T rows).
   - Intersections MAY have different `state_dim` and `n_actions`. **Never pad across intersections.**
   - Episode scalars: `episode_length` (= T), `terminated`, `truncated`, `engine_seed`,
     `flow_draw` (int, `-1` when absent).
   - Filename: `ep{counter:06d}_seed{engine_seed}.npz`, or
     `ep{counter:06d}_seed{engine_seed}_draw{flow_draw}.npz` when `flow_draw is not None`.
8. **Episode hash, computed at write time:** `sha256` over the concatenated raw bytes of all
   per-intersection `action` arrays and `global_reward`, stored in the manifest entry as
   `episode_sha256`. This is what P2.4's duplicate detector consumes — computing it now costs three
   lines; computing it later costs a re-scan of the entire corpus. It also turns the deterministic-demand
   failure mode into an automatic alarm.
9. **`manifest.json` per collection run**, updated atomically (write temp + `os.replace`; document that
   one `out_dir` belongs to one process). Contents: caller-supplied `run_metadata` (scenario id, backend,
   behavior-policy id, checkpoint path/hash if any, `delta_time`, `max_steps`, `state_features`,
   reward-fn names, phase-control mode), repo git hash (`git rev-parse HEAD`, `"unknown"` if
   unavailable), `format_version: "1.0"`, `lane_count: L`,
   `lane_ids_sha256` (sha256 of `"\n".join(lane_ids)` — lets the linter spot two episodes in one run
   that disagree on topology, for free), and per-episode entries with
   `(filename, episode_length, total_global_reward, engine_seed, flow_draw, episode_sha256)`.
10. **`load_episode(path) -> Episode`** — module-level; small dataclass exposing the arrays, with
    per-intersection data keyed by real intersection id, plus `lane_ids` and `metric_keys`.
    This is the reload contract the P3 dataset loader will build on.
11. **Docstrings.** The module docstring and the `Episode` dataclass docstring must both contain the
    alignment block verbatim:
    ```
    row t   = observation before decision t (aligned with a_t)
    row t+1 = the post-step state that r_t was computed from
    observations: T+1 rows · decisions: T rows · outcomes: T rows
    ```
    plus the format version. This sentence is what stops P3 (returns-to-go) and P2.4 (linter) from
    picking the wrong offset.

## FILE 2 — `offline/collect.py` (+ `offline/__init__.py`)

CLI for a collection run:

```bash
python -m offline.collect \
  --env-config configs/sim/cityflow1x1.json --backend cityflow \
  --policy maxpressure \
  --checkpoint path.pt --epsilon 0.1 \
  --episodes 20 --base-seed 1000 \
  --max-steps 360 --delta-time 10 \
  --out-dir datasets/hz1x1_maxpressure
```

- **`--policy` must be a registry `dict[str, Callable]`, not an if/elif chain.** Ship exactly four
  entries: `maxpressure`, `random`, `mappo`, `mappo_eps`. The ladder also needs `fixedtime` and `dqn`,
  but **those are P2.5, not this task** — the registry exists so they can be added later without
  touching the CLI. Do not implement them now; there is no fixed-time controller in this repo (verified:
  zero matches for `fixed.time|FixedTime` outside `CityFlow/`), so writing one means design decisions
  about cycle length and phase split that belong in their own task.
- `random` = uniform over `avail_actions` per intersection, seeded. `mappo` loads `MAPPOAgent` and calls
  `act(info, explore=False, update_memory=False)`. `mappo_eps` wraps that with per-intersection
  ε-substitution from a seeded RNG.
- Episode `i` uses `engine_seed = base_seed + i`.
- Construct the env the way `experiments/` does. If env-construction details are not discoverable from
  the files you read, implement `make_env` with an explicit `TODO` and **flag it in the Return Packet** —
  do not invent config keys.
- Create `offline/__init__.py` so `python -m offline.collect` imports.
- Print one line per episode and a final summary (episodes, total steps, mean return).

## FILE 3 — `tests/test_trajectory_logger.py`

No simulator required. Build a `FakeTrafficEnv` in the test file honoring C1/C2 with **2 intersections of
different state widths (4 and 6) and different `n_actions` (2 and 3)**, 3 lanes, and scripted
deterministic dynamics.

1. **Shapes and dtypes** of every array — assert observation arrays have `T+1` rows and decision/outcome
   arrays have `T`, for `T = 5`. This is the test that pins the convention.
2. **Roundtrip:** `load_episode` returns arrays equal to what was fed in; per-intersection keying by real
   id is correct; `lane_ids` ordering preserved.
3. **Reward recomputation — exact equality (the load-bearing test).** Script the fake env so that
   `r_t = -sum(waiting counts in the post-step state)`. In the test, recompute independently with a plain
   `np.sum` over `lane_waiting_vehicle_count[t+1]` and assert `==` against the logged `global_reward[t]`
   for **every** `t`, including `t = T-1`. Integer counts, so exact equality must hold — do **not** use
   `np.allclose` with a loose tolerance. This is the §7 double-computation rule, and a loose assertion
   here would let an off-by-one into the corpus format.
4. **`local_reward` timing:** script per-intersection rewards that differ per step and assert
   `local_reward[t]` equals the reward the env produced for step `t` (not `t-1`). Separately assert the
   array is NaN-filled when the fake env omits the `"reward"` key.
5. **State-machine misuse raises:** `on_step_result` before `on_action`; `on_action` twice in a row;
   lane-id set changing mid-episode.
6. **Manifest:** parses; episode count, lengths, `format_version`, `lane_count`, `lane_ids_sha256`,
   `flow_draw`, and `episode_sha256` all present and correct.
7. **Determinism:** two identical runs produce byte-identical arrays **and identical `episode_sha256`**.
8. **Duplicate detection works:** two runs with the same seed produce equal `episode_sha256`; a run with a
   different scripted demand produces a different one.

---

## DEFINITION OF DONE
- [ ] `offline/__init__.py`, `offline/trajectory_logger.py`, `offline/collect.py`, `tests/test_trajectory_logger.py` complete — no placeholders, no `pass` stubs
- [ ] `pytest tests/test_trajectory_logger.py -q` **actually executed**; real output pasted
- [ ] Zero modifications to frozen files (`git diff --stat` proves it)
- [ ] Zero new dependencies (numpy + stdlib; pytest for tests)
- [ ] Alignment block + format version present verbatim in both required docstrings
- [ ] Committed on `task/p1-logger`
- [ ] `docs/returns/P1.md` written from `docs/returns/TEMPLATE.md`

# BRIEF #2a — FlowRandomizer (demand randomisation)
**Task ID:** P2.0 · **Contracts:** v1.1 (`docs/CONTRACTS.md`) · **Branch:** `task/p2-flow-randomizer`
**Issued:** 2026-07-27, Master Coordination Chat · **Merge gate:** independent `/review` must pass

> Single source of truth for this task. Supersedes the one-line P2.0 description in
> `docs/PROJECT_PLAN.md`.

---

## HOW TO RUN THIS TASK
Claude Code, in the repo, branch `task/p2-flow-randomizer`. Read the listed paths from disk — do not
infer. Commit before finishing. Return Packet to `docs/returns/P2.0.md` from
`docs/returns/TEMPLATE.md` (real pytest output, real `git diff --stat`, honest checklist).

**Read first:** `offline/trajectory_logger.py`, `offline/collect.py`, `docs/CONTRACTS.md`,
`experiments/config.py` (EnvSpec, SETTING_DEFAULTS), `envs/cityflow_env.py` (how the sim config and
flow file are loaded), and at least two `scenarios/*/flow.json` plus one `scenarios/*/*.rou.xml`.

**If this brief conflicts with the repo, the repo wins** — implement to the repo, flag it in the
Return Packet.

---

## WHY THIS TASK EXISTS (read; it sets the acceptance bar)

Measured on real CityFlow, 2026-07-27: engine seeds 1000 and 1001 produced **byte-identical
trajectories**, identical `episode_sha256`, identical return (−502.000). CityFlow demand comes from a
fixed flow file, so `reset(seed=)` changes nothing about vehicle arrivals; a deterministic policy
therefore emits one trajectory N times while the manifest truthfully reports N episodes.

**Without this component the offline corpus carries almost no information.** That is the acceptance
bar: a corpus collected with the randomiser must produce **distinct `episode_sha256` values across
draws**, and the test suite must prove it.

## VERIFIED FACTS ABOUT THE DATA (checked 2026-07-27 — trust but re-verify)

`flow.json` is a **list of individual vehicle insertions**, not aggregate flows. One entry:

```json
{"vehicle": {"length": 5.0, "width": 2.0, "maxPosAcc": 2.0, "maxNegAcc": 4.5,
             "usualPosAcc": 2.0, "usualNegAcc": 4.5, "minGap": 2.5,
             "maxSpeed": 11.11, "headwayTime": 2.0},
 "route": ["road_0_1_0", "road_1_1_0"],
 "interval": 5, "startTime": 0, "endTime": 0}
```

`hangzhou_1x1_bc-tyc` has 2021 such entries. **Randomising at the vehicle-list level is therefore
straightforward and — critically — engine-independent**, which is what lets one draw render to both
backends and keeps claim C3 (CityFlow→SUMO transfer) alive.

**Scenario pairing — corrected inventory (verify yourself before relying on it):**

| scenario | flow.json | .rou.xml | .net.xml | paired |
|---|---|---|---|---|
| hangzhou_1x1 ×4 (bc-tyc, qc-yn, kn-hz, sb-sx) | yes | yes | yes | **yes** |
| cologne1, cologne3 | yes | yes | yes | **yes** |
| grid4x4 | yes | **no** | **no** | **NO** — its `.sumocfg` references a `grid4x4.rou.xml` that does not exist |

So 6 of 7 are genuinely dual-format. grid4x4 is CityFlow-only in practice.

## SCOPE

**In scope:** a seeded demand randomiser producing reproducible draws from a scenario's vehicle list,
rendering CityFlow `flow.json`, plus SUMO `.rou.xml` for paired scenarios; wiring a `--flow-draw`
option into `offline/collect.py`; tests.

**Out of scope — do not build:** the Decision Transformer, the dataset loader (P3), the corpus linter
(P2.4), a fixed-time controller (P2.5), any change to `experiments/runner.py` (that is Brief #2b), any
new behaviour policy. **No modification to any frozen file.**

---

## FILE 1 — `offline/flow_randomizer.py`

### Core API

```python
@dataclass(frozen=True)
class FlowDraw:
    draw_id: int              # 0 == nominal (identity transform), >0 == randomised
    seed: int
    n_vehicles: int
    source_sha256: str        # sha256 of the source flow file — pins provenance
    params: dict[str, float]  # the transform parameters actually applied

class FlowRandomizer:
    def __init__(self, source_flow_path, *, jitter_sigma_s=..., thin_p=..., volume_scale=...): ...
    def draw(self, draw_id: int) -> tuple[list[dict], FlowDraw]: ...
    def render_cityflow(self, entries, out_path) -> Path: ...
    def render_sumo(self, entries, out_path) -> Path: ...      # paired scenarios only
```

### Requirements

1. **`draw_id == 0` is the identity.** It must return the source vehicle list **unchanged** — byte-identical
   render. This is the nominal-flow control condition for every experiment. Assert it in a test.
2. **Three transforms, all seeded from `draw_id`** (`np.random.default_rng(base_seed + draw_id)`; the
   same `draw_id` must always give the same draw):
   - **departure jitter** — perturb `startTime` by `N(0, sigma)`, clipped to `>= 0`; keep the list
     sorted by `startTime` after jittering
   - **Bernoulli thinning** — independently drop each vehicle with probability `thin_p`
   - **volume scaling** — scale the vehicle count by `volume_scale` (>1 duplicates entries with fresh
     jittered start times; <1 is equivalent to thinning). Pick one coherent implementation and
     document it.
   Defaults should produce a *visible but not pathological* change — roughly ±10–20% demand. State the
   chosen defaults and the reasoning in the module docstring.
3. **Never mutate the source file.** Reads only; all output goes to a caller-supplied path.
4. **`render_cityflow`** writes the same JSON schema as the source (same keys, same vehicle-parameter
   block). Round-trip: `render_cityflow(draw(0))` must be **byte-identical** to the source file.
5. **`render_sumo`** writes valid `.rou.xml`. Study a real one first (e.g.
   `scenarios/cologne3/cologne3.rou.xml`) and mirror its structure. **The route representation differs
   between the two formats** (CityFlow lists road ids; SUMO uses `<route edges="...">` with its own
   edge ids). If you cannot establish a faithful id mapping from the files in the repo, **do not invent
   one** — implement `render_sumo` to raise `NotImplementedError` with a precise message naming what is
   missing, and flag it as the top open question in the Return Packet. A wrong mapping would silently
   corrupt claim C3; an honest gap would not.
6. **Determinism and distinctness:** the same `draw_id` yields identical output; different `draw_id`s
   yield different vehicle lists. Both asserted.
7. **Provenance:** `FlowDraw.source_sha256` is the sha256 of the source flow file, so a corpus can be
   traced to the exact demand it came from.
8. Python ≥3.12, type hints, numpy + stdlib only, no new dependencies.

## FILE 2 — changes to `offline/collect.py`

- Add `--flow-draw N` (default `None` → current behaviour, nominal source flow, unchanged) and
  `--flow-draws A B C` or `--flow-draws-range START END` so one collection run can sweep draws.
- For each draw: materialise the flow file into a temp/working directory, point the `EnvSpec` at it,
  build the env, and pass `flow_draw=<draw_id>` to `logger.on_reset(...)` — the field is already
  plumbed through P1 and lands in the filename and the manifest.
- **Verify, do not assume, whether a new draw needs a fresh env object.** CityFlow reads the flow file
  at engine construction; `reset()` most likely does not re-read it. Check `envs/cityflow_env.py` and
  test it. If a fresh env per draw is required, do that — measured per-episode cost including
  construction is 0.6–1.2 s, so it is affordable. Record the finding in the Return Packet.
- Record in `run_metadata`: `flow_draw_ids`, the randomiser parameters, and `source_sha256`.

## FILE 3 — `tests/test_flow_randomizer.py`

No simulator required; operate on a small synthetic vehicle list plus, where useful, a real
`scenarios/*/flow.json` read read-only.

1. `draw(0)` returns the source list unchanged; `render_cityflow` of it is **byte-identical** to the source.
2. Same `draw_id` twice → identical entries and identical `FlowDraw`.
3. Different `draw_id`s → different entries (assert a **count** of differing vehicles above a threshold,
   not merely `!=` — a single differing element would satisfy `!=` and prove nothing).
4. Jitter: `startTime` distribution shifts but stays `>= 0`; the list remains sorted.
5. Thinning: with `thin_p=0.5` and a 1000-entry list, the surviving count is within a sane binomial band.
6. Volume scaling: count changes in the expected direction by roughly the expected factor.
7. Source file is never modified (hash it before and after).
8. `render_cityflow` output re-parses as JSON with the same schema as the source.
9. `render_sumo` — if implemented, output parses as XML with the expected elements; if it raises
   `NotImplementedError`, assert that explicitly so the gap is visible in the suite rather than silent.
10. **The acceptance test:** two draws of the same scenario produce vehicle lists whose sha256 differ.
    This is the property whose absence made the corpus information-free.

---

## DEFINITION OF DONE
- [ ] `offline/flow_randomizer.py`, `offline/collect.py` (modified), `tests/test_flow_randomizer.py` — no placeholders
- [ ] `pytest tests/ -q` **actually executed**, real output pasted; whole suite still green (was 180)
- [ ] Zero modifications to frozen files (`git diff --stat` proves it)
- [ ] Zero new dependencies
- [ ] Committed on `task/p2-flow-randomizer`; `docs/returns/P2.0.md` written
- [ ] Return Packet states plainly: is `render_sumo` real or a raised `NotImplementedError`, and does a
      new draw require a fresh env object?

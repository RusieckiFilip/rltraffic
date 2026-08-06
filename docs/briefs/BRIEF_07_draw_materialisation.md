# BRIEF #7 — draw materialisation: turn flow draws into runnable scenarios

**Mode:** Claude Code, in the repo, on a task branch.
**Branch:** `task/p2.2-materialise-draws`
**Issued:** 2026-08-06 by the Master chat.

---

## 1. Why this exists

`offline/flow_randomizer.py` can already *produce* a draw (`FlowRandomizer.draw(draw_id)` →
entries + `FlowDraw`, then `render_cityflow` / `render_sumo`). What does not exist is anything that
**writes those draws somewhere stable and makes them runnable**, i.e. a `configs/sim/`-style config
pointing at the drawn flow file.

Three things are blocked on exactly that gap:

1. **`mappo_dr`** (RUNSPEC_01 §1) — the domain-randomised cell of C2's pre-registered 2×2, which must
   train across draws 1–5.
2. **The reported online-MAPPO baseline for both cells** — D4 requires evaluation on the **held-out
   draws 1000–1099**, and those must exist as runnable scenarios. Note this evaluation *trains
   nothing*: it re-runs saved checkpoints through `run.py --from-checkpoint`.
3. **P2.2's collection campaign** — the corpus is collected per draw.

**Scope fence: this brief materialises draws only.** The draw-cycling *trainer* is a separate task
(see §5). Do not build it here.

---

## 2. What a runnable scenario looks like

A CityFlow sim config is small and declarative — `configs/sim/cityflow1x1.json`:

```json
{ "network": "hz1x1", "interval": 1.0, "seed": 0,
  "dir": "scenarios/hangzhou_1x1_bc-tyc_18041610_1h/",
  "roadnetFile": "roadnet.json", "flowFile": "flow.json",
  "rlTrafficLight": true, "saveReplay": false, "laneChange": false }
```

So a materialised draw is: **the drawn flow file on disk**, plus **a config identical to the source
except for `dir`/`flowFile`**. The roadnet is unchanged — a draw perturbs demand, never topology.

---

## 3. Deliverable — `offline/materialise_draws.py` (NEW) + tests

A CLI and an importable function that, for a given source scenario and a set of draw ids, writes the
drawn flow files and matching configs to a stable, predictable location, and records provenance.

**Requirements:**

- **Output layout must encode the draw id in the path**, so a config can never be silently confused
  with another draw. Propose the layout in your plan; it must be greppable and stable.
- **Provenance, per draw**: source scenario, source `sha256` (`FlowRandomizer.source_sha256`), draw id,
  the randomiser's `params()`, the repo git hash, and the `FlowDraw` metadata. A materialised scenario
  whose provenance lives only in a filename is the N5 defect again.
- **Both backends** where the source is paired: `render_cityflow` **and** `render_sumo`. C3 needs the
  same drawn demand in both engines, and P2.0 shipped `render_sumo` real.
- **Filesystem-mutation barrier (§7 of the plan, non-negotiable):** every write **and** delete happens
  **after** all validation. This bug has appeared twice in this project — a failed construction must
  never destroy a prior corpus. Validate the whole set first, then write.
- **Refuse to overwrite silently.** An existing materialised draw is either left alone or replaced only
  under an explicit flag, and never partially.
- **Draw 0 is special and must be handled explicitly**: it preserves source vehicle order while
  `k > 0` sorts globally, so draw 0 is the *nominal control* and is reported separately, never pooled
  (D4). If you materialise it, label it as such.

**Scenarios to materialise:** the headline three (hangzhou_1x1 #1 bc-tyc, grid4x4, cologne3).
**Draw ids:** `1–5` (RUNSPEC_01 §2, the `mappo_dr` training set) and `1000–1004` (the first five of the
held-out evaluation pool). Materialising more of 1000–1099 later must be a no-op re-run, not a rebuild.

---

## 4. Tests

- **Load-bearing — the drawn scenario actually runs and differs:** materialise draws 1 and 2 for one
  scenario, construct a real env on each, step them, and assert (a) both run, and (b) they produce
  **different** `episode_sha256`-style trajectories under a *deterministic* policy. A draw that renders
  but does not change behaviour is the failure this whole subsystem exists to prevent — recall that
  determinism made N episodes of a deterministic policy one trajectory repeated N times.
- **Draw 0 round-trip:** materialising draw 0 reproduces the source flow file **byte-identically**
  (P2.0 established this property; if it fails, something upstream broke).
- **Held-out disjointness:** a materialised training draw id is never in 1000–1099, and vice versa —
  assert on the ids the tool actually wrote, not on the argument it was given.
- **Mutation barrier:** an induced failure part-way through materialisation leaves the previous state
  intact. Prove it by injecting the failure, not by reading the code.
- **Provenance completeness:** every written config has a provenance record naming its source sha256
  and draw id.
- Standard hygiene: no reasonless skips, no `pytest.raises` without `match=`, no weakened assertions.

---

## 5. What comes next, so you can see the shape (do NOT build it here)

`mappo_dr` needs a trainer that cycles draws **per episode**. That is a separate brief, and it carries
a subtle constraint worth knowing now, because it may influence your output layout:

> `_train_agent` (frozen) constructs **one** env before the episode loop, and CityFlow reads the flow
> file at engine construction (`engine.cpp:65`; `Engine::reset()` only re-runs the parsed in-memory
> vector). So a new draw means a new engine, and per-episode cycling is impossible through the frozen
> path. The next brief will build a draw-cycling trainer in `offline/` that **must be proven to
> reproduce `_train_agent`'s per-episode returns exactly on draw 0** before it is used — otherwise the
> 2×2's demand axis is confounded with a training-loop implementation axis, and C2 dies quietly.

Practical consequence for you: **pre-constructing one env per draw and rotating** will be preferable to
constructing an engine per episode (≈11,400 constructions across the matrix). A layout that makes "give
me the config for scenario S, draw D" a cheap lookup is therefore worth more than a clever one.

---

## 6. Definition of Done

- [ ] Plan file first (`docs/plans/p2.2-draws.md`), approved before code
- [ ] `offline/materialise_draws.py` + tests; draws 1–5 and 1000–1004 materialised for the headline three
- [ ] Load-bearing test written, shown to fail against a mutation, then passing
- [ ] Provenance recorded per draw; mutation barrier proven by injected failure
- [ ] Full `.venv/bin/pytest -q`, real tail, count reported against the current **320 passed / 2 skipped**
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P2.2-draws.md`

## 7. Return Packet — task-specific questions

1. The output layout you chose and why; how a caller asks for "scenario S, draw D".
2. Did draw 0 round-trip byte-identically? If not, stop — something upstream is broken.
3. Evidence that two different draws produce different trajectories under a deterministic policy.
4. Disk cost per draw per scenario, and the projected cost of the full 1000–1099 pool.
5. Anything in §5's constraint that your layout makes harder, while it is still cheap to change.

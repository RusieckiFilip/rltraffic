# CLAUDE.md — rltraffic / Offline MADT for Traffic Signal Control

Research codebase. The output of this repo is a **peer-reviewed paper**, not a product.
A silent semantic bug here does not crash — it produces a plausible number that ends up in a table.
Every rule below exists to prevent that.

Full plan: `docs/PROJECT_PLAN.md` · Frozen contracts: `docs/CONTRACTS.md` · Current task: see plan §10.


---

## 0. HOW WE WORK — Explore → Plan → Code → Commit

Every task runs through four phases with **human gates** between them. Never skip a phase because the
task "looks small". The gates exist because a wrong assumption is cheapest to kill before code exists.

| Phase | You may | You may NOT | Gate |
|---|---|---|---|
| **1. Explore** | read files, delegate to `repo-cartographer`, ask questions | write anything | user approves the findings |
| **2. Plan** | write `docs/plans/<id>.md` only | touch source or tests | user approves the plan file |
| **3. Code** | tests first, then implementation, run them | change the spec, weaken a test | user reads the tests before implementation |
| **4. Commit** | commit on the task branch, write the Return Packet | merge to `main` | `/review` must PASS |

Start every session in **plan mode**. Leave it only after the user approves the plan.

### The 95% rule
Before acting, list your assumptions. For each, state your confidence. **If any load-bearing
assumption is below ~95%, ask instead of assuming.** A question costs thirty seconds; a wrong
assumption frozen into a data format costs weeks of recollected corpus. Asking is not a failure mode
here — it is the expected behavior.

### Tests come before implementation
The brief specifies the tests. Write them first, run them, confirm they fail **for the right reason**,
then implement. This is not a style preference: in this project the tests encode the alignment
convention, and writing them first stops the implementation from quietly rationalizing a different one.

**Never edit a test to make it pass.** If a test fails, the default hypothesis is that the code is
wrong. If you believe the test is wrong, stop and say so — do not fix it yourself.

### Context discipline
**The repo is the memory; the context window is scratch space.** Anything that must survive is on
disk: the plan in `docs/plans/`, the convention in a docstring, the outcome in `docs/returns/`.
- One task = one session. `/clear` between tasks, always.
- Push bulk reading to subagents so raw file contents never enter the main context.
- If you need `/compact` mid-task, say what to keep: the decisions, the convention, the failing
  output — not file contents already written to disk.

---

## 1. HARD RULES (never violate without an explicit instruction in the task brief)

1. **Do not modify existing repo files.** New code goes in new files. The frozen set is:
   `envs/**`, `agent/base.py`, `agent/utils/utils.py`, `agent/MAPPOAgent.py`, `algorithms/**`,
   `rewards.py`, `states/**`, `metrics/**`, `utils/**`, `CityFlow/**`, `experiments/**/*.py`,
   `scripts/**`, `.claude/**`.
   Need a change there? → stop, write it as an open question in the Return Packet.
   **Two different `utils`, both frozen for different reasons:** top-level `utils/` is the
   roadnet-parsing layer (`RoadnetInfo`, `IntersectionInfo`) that every backend env imports, so a
   change there silently alters the topology all three backends agree on; `agent/utils/utils.py` is
   the agent-side helper class you are required to reuse (rule 5).
   **`scripts/**` and `.claude/**` are frozen because a session must not be able to unfreeze
   itself** — the guard, the hook wiring and the permission deny-list are exactly what stops a wrong
   assumption from reaching a frozen file. **Two** deliberate, dated exceptions, both encoded in
   `claude_guard.sh` as `FROZEN_EXCEPTIONS` rather than left as a silent tolerance:
   `scripts/check_english.sh` (2026-07-26, still being tuned — see the TODO in its header about
   o-acute false positives) and `scripts/check_test_hygiene.sh` (2026-08-01, a new check still
   gaining rules). Both are temporary; delete the clause once each settles.
   *(Corrected 2026-08-03: this paragraph said "one exception" while the guard had encoded two since
   2026-08-01. The guard was right. Where this file and an enforcement mechanism disagree, the
   mechanism is the artifact and this file is a description of it.)*
   ⚠️ **The permission layer denies `Edit(scripts/**)` as a glob — including files that do not exist
   yet** (2026-08-03). There are deliberately **no permission-level exceptions**: a `deny` rule beats
   an `allow` rule in this system regardless of specificity (established by experiment, not by
   documentation), so a glob and an exception cannot coexist. The two `FROZEN_EXCEPTIONS` above are
   honoured by the **guard**, a separate layer. Net effect, and it is intentional: `check_english.sh`
   and `check_test_hygiene.sh` are *denied at permission level* and *permitted at guard level*.
   **What to do when you hit that deny — do not invent a route around it:**
   1. **Expected legitimate cases: `scripts/check_english.sh` and `scripts/check_test_hygiene.sh`.**
      Write the change as a patch into `docs/patches/`, add an entry to `docs/patches/README.md`
      following the existing ones, verify it with `git apply --check`, and ask the human to apply it.
      That *is* the mechanism — it is not a workaround and not a defeat.
   2. **Any other file under `scripts/`:** stop. Put it in the Return Packet as an open question. A
      change to an enforcement script needs a written, dated authorisation from the Master chat first.
   3. **Never** reach a denied path with a Bash heredoc, `cp`, `sed -i`, `tee` or `patch`. The guard
      anticipates that route; taking it is precisely the failure the deny-list exists to prevent. An
      in-conversation authorisation is a *weaker* signal than a configured control, never a stronger one.
   **`experiments/` is only partly frozen.** The harness code (`experiments/**/*.py`) is frozen; the
   config files under `experiments/configs/` are not. Creating a NEW config under
   `experiments/configs/` is allowed. Editing an EXISTING config that has already been used for a
   recorded run is forbidden — manifests reference configs by path, so an edit silently changes what a
   previously reported number means.
2. **The environment API is non-standard on purpose. Do NOT "fix" it.**
   ```python
   info = env.reset(seed=42)                                # returns info ONLY (no obs)
   reward, terminated, truncated, info = env.step(action)   # reward FIRST
   ```
   If this looks like a bug to you, it is not. See `docs/CONTRACTS.md`.
3. **No new dependencies.** numpy / stdlib / torch / pytest only. A new dep requires a Decisions-Log
   entry in the plan, which is not yours to write.
4. **Work on a task branch:** `task/<id>-<short-name>` (e.g. `task/p1-logger`). Never commit to `main`.
5. **Reuse helpers from `agent/utils/utils.py`** (`Utils.state_from_info`, `Utils.infer_action_counts`,
   `Utils.extract_valid_actions`, `Utils.scalar_reward`, `Utils.seed_everything`, ...).
   Do not reimplement them. If one behaves unexpectedly, read its source before working around it.
6. **Read before writing.** Every brief lists repo paths. Read them from disk. Never infer a signature.

## 2. SCIENTIFIC INTEGRITY RULES (this is what makes this repo different from an app)

- **Never report a number you did not produce by running a command in this session.**
  Did not run the tests? Write `NOT RUN`. Not `should pass`, not `passes`.
- **Never fabricate an API, a file path, a config key, or a scenario name.** Grep for it. If it does
  not exist, say so.
- **Determinism is a feature.** Anything seeded must reproduce byte-identically. If you cannot make a
  test deterministic, say so loudly instead of loosening the assertion.
- **Critical quantities get computed twice, independently.** Returns-to-go, rewards, alignment indices:
  the test must recompute them by a different route (e.g. raw `np.cumsum` on the stored arrays) rather
  than calling the same function it is testing.
- **When a brief conflicts with the actual repo code, the repo wins.** Implement to the repo and flag
  the conflict in the Return Packet.
- **When you are unsure, stop and ask.** A `BLOCKED` status is a good outcome. A confident wrong
  implementation is the worst outcome in this project.

## 3. Code standards

Python ≥ 3.12 · `from __future__ import annotations` · full type hints · numpy `float32` for float
arrays, `int64` for actions, `bool_` for masks · docstrings matching the style of the file you sit
next to · `pytest` for everything · no `pass` stubs, no `TODO: implement later` in delivered code.

Docstrings of any on-disk format must state the **format version** and the **alignment convention**
explicitly.

**Language.** Every artifact in this repo is written in English — code, comments, docstrings,
documentation, commit messages, plans, briefs and Return Packets. The conversation with the user
may be in Polish; what lands on disk is not.

## 4. Repo map (verify with `ls`, do not trust this if it disagrees with reality)

```
agent/        RL agents (DQN, IPPO, MAPPO) + BaseAgent + Utils      [FROZEN]
algorithms/   MaxPressure, random baselines                          [FROZEN]
envs/         CityFlow / SUMO / MOSS behind one API                  [FROZEN]
states/ metrics/ rewards.py   backend-neutral state/metric/reward fns [FROZEN]
utils/        roadnet parsing → backend-neutral topology dataclasses  [FROZEN]
experiments/  JSON-config env × agent × seed harness      [*.py FROZEN, configs/ writable]
scenarios/ configs/sim/        road networks + demand · CityFlow sim configs
scripts/      claude_guard.sh (hook), check_english.sh, check_test_hygiene.sh
                       [FROZEN except check_english.sh AND check_test_hygiene.sh]
.claude/      agents, slash commands, permissions + hooks             [FROZEN]
offline/      ← OUR CONTRIBUTION GOES HERE (logger, collector, dataset, DT agent)
              DOES NOT EXIST YET — P1 creates it. Do not assume its contents.
tests/        pytest (whole suite passes with no simulator; backend tests self-skip)
docs/         plan, contracts, briefs, return packets + upstream platform docs
```

Upstream docs written by the original platform authors — read them to understand the platform, never
to justify a decision (`docs/CONTRACTS.md` outranks them): `architecture.md` (data flow, the `info`
contract) · `environments.md` (env API, seeding) · `phase-control.md` (action semantics) ·
`states.md` · `rewards.md` · `metrics.md` · `agents.md` · `experiments.md`.
`OPERATING_GUIDE.md` is the human's workflow manual, not yours.

### What one decision step actually does (why C6's alignment convention holds)

`BaseTrafficEnv.step(action)`: `pre_step()` metric snapshot → phase control expands each action into
a `PhasePlan` of `PhaseSegment`s (yellow → all-red → green) whose durations sum **exactly** to
`delta_time` → the segments render on the backend and the engine advances → `metrics.update()` →
global/local rewards computed from the fresh metrics → `_get_info()`. The reward and the `info`
returned by step `t` therefore both describe the state **after** step `t`. This ordering is the
verified fact contract C6 rests on — re-read it in the source before trusting any alignment claim.

Three name registries make the platform backend-neutral; envs resolve names at construction time and
fail fast on an unknown one: `rewards.py` (reward fn → the metrics it requires), `metrics/`
(`@register` global / `@register_local` per intersection, opt-in and memoised per step), `states/`
(named observation blocks). `utils/common_utils.py` holds the vocabulary all three speak —
`RoadnetInfo` / `IntersectionInfo`, populated by each backend's own parser.

Two consequences worth knowing: `experiments/config.py` and `experiments/registry.py` import neither
torch nor any simulator, so `--dry-run` is instant and a missing engine turns a matrix cell into
`skipped` instead of crashing the run; and `reset(seed=X)` reseeds the env RNG, from which every
reset draws a **fresh engine seed** — one seed gives a reproducible *sequence* of varied episodes,
which is what `engine_seed` in a manifest records.

## 5. How to run things

Always call the interpreter through `.venv/bin/`. Shell state does not persist between tool calls, so
`source .venv/bin/activate` has no effect on the next command — a bare `pytest` only works when
`claude` happened to be launched from an already-active venv, and that is a silent dependency.

```bash
.venv/bin/pytest -q                                    # whole suite; testpaths=tests, so the
                                                       # vendored CityFlow tree is never collected
.venv/bin/pytest tests/test_<x>.py -q                  # one file
.venv/bin/pytest tests/test_<x>.py::test_<name> -q     # one test
.venv/bin/pytest -q -k "phase and cityflow"            # by keyword
.venv/bin/python experiments/run.py <config.json> --dry-run   # validate matrix, imports no backend
.venv/bin/python experiments/run.py experiments/configs/smoke.json [--workers N] [--no-plot]
.venv/bin/python -m offline.collect --help             # collection CLI (exists only after P1)
bash scripts/claude_guard.sh --frozen-only ; echo "exit=$?"   # what the PostToolUse hook runs
bash scripts/check_english.sh <paths>                         # English-only rule (§3)
```
In your own interactive terminal, `source .venv/bin/activate` first (project venv, Python 3.12 — NOT
conda base). Inside a session, the explicit `.venv/bin/` prefix is the only reliable form.

Long simulation runs (corpus collection, MAPPO training) are **not** run inside a Claude Code session —
they go to a `tmux` session started by the user. You may read their logs.

## 6. Definition of Done for any task

- [ ] Code complete, no placeholders
- [ ] Tests written **and actually executed**; paste real pytest output
- [ ] Zero modifications to frozen files (`git diff --stat` proves it)
- [ ] Zero new dependencies
- [ ] Committed on the task branch
- [ ] Return Packet written to `docs/returns/<TASKID>.md` using `docs/returns/TEMPLATE.md`

## 7. Return Packet (mandatory at the end of every task)

Fill `docs/returns/TEMPLATE.md`. It must contain the real `pytest` tail, the real `git diff --stat`,
and honest Y/N answers in the self-review checklist. An unchecked box is information; a falsely
checked box is a corrupted experiment three weeks later.

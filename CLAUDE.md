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
   `rewards.py`, `states/**`, `metrics/**`, `CityFlow/**`, `experiments/**`.
   Need a change there? → stop, write it as an open question in the Return Packet.
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

## 4. Repo map (verify with `ls`, do not trust this if it disagrees with reality)

```
agent/        RL agents (DQN, IPPO, MAPPO) + BaseAgent + Utils      [FROZEN]
algorithms/   MaxPressure, random baselines                          [FROZEN]
envs/         CityFlow / SUMO / MOSS behind one API                  [FROZEN]
states/ metrics/ rewards.py   backend-neutral state/metric/reward fns [FROZEN]
experiments/  JSON-config env × agent × seed harness                 [FROZEN]
scenarios/ configs/           scenario + sim configs
offline/      ← OUR CONTRIBUTION LIVES HERE (logger, collector, dataset, DT agent)
tests/        pytest
docs/         plan, contracts, briefs, return packets
```

## 5. How to run things

```bash
conda activate <env>              # ask the user which env if unsure; do not create a new one
pytest tests/test_<x>.py -q       # fast tests, no simulator needed
python -m offline.collect --help  # collection CLI
```
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

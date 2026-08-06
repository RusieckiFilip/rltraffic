# BRIEF #6 — P8.0: implement the registered primary metric and re-derive the anchors

**Mode:** Claude Code, in the repo, on a task branch.
**Branch:** `task/p8.0-horizon-metric`
**Issued:** 2026-08-06 by the Master chat.
**Status:** 🚨 **BLOCKING.** Every number the paper reports depends on this. Nothing else should be
measured until it merges.

---

## 1. Why this exists

`PREREGISTRATION.md` amendment **A1** (tag `v0.2-prereg-a1`) specifies the primary metric as
`average_travel_time` **at the episode horizon** — the mean over all vehicles that entered the network
during the episode. It is **not** the mean of the per-step samples.

`experiments/runner.py:168,175` computes the latter:
```python
travel_samples.append(float(info.get("average_travel_time", 0.0)))   # per step
...
travel.append(_mean(travel_samples))                                  # reported
```
So every figure in PROJECT_PLAN §3.1 — the sanity anchors §7 requires every later phase to be compared
against — is a running mean, not the registered metric.

**Why this is not cosmetic.** The two aggregations differ *policy-dependently*, so they move effect
sizes rather than rescaling them (coordinator's rollout, 2026-08-05):

| policy | running mean | horizon | ratio |
|---|---|---|---|
| MaxPressure | 160.56 | 247.75 | 1.54 |
| Random | 317.46 | 429.67 | 1.35 |

MaxPressure's advantage over Random reads **49.4%** one way and **42.3%** the other. C1's normalised
return, C2's 2×2 interaction and C3's within-backend transfer ratio are all differences-of-differences
on this quantity.

---

## 2. Naming ruling — obey it exactly, it is what stops a recurrence

`average_travel_time` is the **registry name of the per-step metric** (`metrics/cityflow.py:234`,
frozen). The collision between that and an episode-level number is what produced this defect.

> **No episode-level field, column, variable or dict key may be named bare `average_travel_time`.**
> - `att_horizon` — the registered primary metric (A1). *Always* the paper's "average travel time".
> - `att_running_mean` — the legacy `runner.py` quantity. Never called "average travel time" in prose,
>   a table, or a docstring.

Apply this to every artifact you touch, including CSV headers and JSON keys.

---

## 3. Scope — five deliverables

### 3.1 A horizon-metric reader, in our own code
`experiments/runner.py` is **frozen and does not need changing** — it may keep computing the running
mean. Write the horizon reader in `offline/` and use it for everything we report.

It must read the metric at the episode horizon through the same `CityFlowMetrics` path the env uses,
so it is the identical quantity, differently aggregated. Report **both** `att_horizon` and
`att_running_mean` from one rollout, so no future comparison can silently mix them.

### 3.2 Re-derive the §3.1 anchors
Re-run `experiments/configs/p0_baselines.json` (committed, 2 scenarios × 3 seeds × 3 policies) and
record **both** quantities. Commit the raw output next to the existing data — **do not overwrite
`docs/data/p0_baselines/`**, which is the 2026-07-09 historical record.

> ⚠️ **The committed data cannot yield the horizon value.** It stores only the aggregate. A re-run is
> required; this is not optional and not derivable.

**Load-bearing validation:** the re-run's `att_running_mean` must reproduce the committed anchors —
MaxPressure 160.56 / Random 307.53 / MAPPO 197.91 on cf_hz1x1, and 141.65 / 207.26 / 632.95 on
cf_grid4x4. If the running mean reproduces, the horizon values from the same run are trustworthy by
construction. **This is the whole reason to re-run rather than measure fresh.**

> 🚨 **Predicted partial failure, and it is not a bug.** The committed data is from **2026-07-09,
> before the thread pin** (`64800fb`, 2026-08-03). Reviewer finding N2 established that pinning changes
> torch's float reduction order, bitwise-different at exactly (128,128)@(128,128) — which is MAPPO's
> `minibatch_size` × `hidden_dim` in that very config.
> **Predictions to test explicitly:** MaxPressure and Random touch no torch and should reproduce
> **exactly**; **MAPPO may not**. If MAPPO differs, that is N2 materialising on the first recorded run
> to cross the boundary — document it, quantify the difference, and do **not** adjust anything to make
> it match. If MAPPO *does* reproduce exactly, that is equally informative: the boundary does not bite
> at this scale. Either way, state which happened.

### 3.3 Update `tests/test_p0_baseline_anchors.py`
Assert **both** quantities against their respective committed files. Keep the existing exact-`==`
double-compute for the legacy data (it is immutable history). Add the horizon assertions against the
new data. Test count must go up.

### 3.4 Fix the P2.5 artifacts that encode the demoted aggregation
Found by the P2.5 implementer after merge; correctly computed, wrongly labelled.
- `offline/policies/plan_replay.py`: `ReplayResult.average_travel_time` returns the episode-mean, and
  the comment at **line 120** calls it *"the canonical measure used by experiments.runner.evaluate_policy
  and the P0.2 anchors"*. Under A1 that label is false. Report `att_horizon` as primary, keep
  `att_running_mean` under its own name, fix the docstrings at lines 12, 40 and 119-120.
- The §9 gate (`_env_hold_green1_att` + `test_replay_pipeline_matches_env_on_degenerate_hold`) asserts
  on the episode-mean. **Flip it to `att_horizon`, keeping exact equality** — the implementer's first
  probe already observed the horizon values equal at 662.36 on both sides, so the gate survives the
  switch unchanged. Assert both quantities if cheap.

### 3.5 Re-derive P2.5's Tier 1 numbers under A1
The fixed-time k=3 / k=4 / shipped-plan figures and the grid4x4 inversion are currently reported in
running mean. Re-issue them as `att_horizon`, with entered/completed counts per `PREREGISTRATION.md`
§3.1. **The k=4 ruling is not reopened** — it rests on the structural clearance-overhead criterion
(12.50% vs shipped 14.29% vs k=3 16.67%), which is aggregation-independent.

---

## 4. Scope fence

- **Do not edit `experiments/runner.py`.** It is frozen, no authorisation is granted, and the fix does
  not require it. If you believe it does, stop and say so.
- **Do not overwrite or delete `docs/data/p0_baselines/`.** It is the historical record and the
  validation target.
- **Do not re-open the k=4 ruling** (§3.5).
- **Do not "fix" any number that fails to reproduce.** Report it.
- **No new dependencies.**

---

## 5. Tests

- **Load-bearing:** the anchor test asserting both quantities, with the legacy side keeping its exact
  double-compute. Prove strength by mutation — tamper a stored value in a temp copy and show the check
  fails with `match=`.
- A test that the horizon reader and the env agree on the same rollout (same quantity, one aggregation
  apart) — the §3.1 double-computation rule.
- The flipped §9 gate, still exact-equality.
- **A naming guard:** a mechanical check that no episode-level field in `offline/` is named bare
  `average_travel_time`. This defect came from a name collision; a grep is stronger than a convention.
- Standard hygiene: no reasonless skips, no `pytest.raises` without `match=`, no weakened assertions.

---

## 6. Definition of Done

- [ ] Plan file first (`docs/plans/p8.0.md`), approved before code
- [ ] Horizon reader in `offline/`, reporting both quantities from one rollout
- [ ] Anchors re-derived and committed alongside (not over) the historical data
- [ ] Reproduction result stated explicitly per policy, including whether MAPPO crossed the N2 boundary
- [ ] `plan_replay.py` + §9 gate corrected; naming ruling applied throughout
- [ ] P2.5 Tier 1 numbers re-issued as `att_horizon`
- [ ] Full `.venv/bin/pytest -q`, real tail, count reported against the current **299**
- [ ] `git diff --stat` shows zero frozen-file modifications
- [ ] Return Packet at `docs/returns/P8.0.md`

## 7. Return Packet — task-specific questions

1. Did the re-run reproduce each committed anchor? Per policy, exactly or not.
2. Did MAPPO cross the N2 float-reduction boundary? Quantify the difference if so.
3. The re-derived §3.1 table in both quantities, side by side.
4. P2.5's Tier 1 numbers under `att_horizon`, with entered/completed counts.
5. Anything in the naming ruling that does not survive contact with the code.

---

## 8. Rulings on the plan-mode questions (Master chat, 2026-08-06)

### Ruling 1 — **Option A. CLAUDE.md §5 wins; the brief yields.**

You were right to stop. CLAUDE.md:203 names MAPPO training explicitly as a `tmux` case, "When a brief
conflicts with the actual repo code, the repo wins" applies to repo *rules* as well as repo code, and I
am not waiving a rule you correctly cited — a flagged conflict that gets waived on request teaches
exactly the wrong lesson.

**Split the work so the blocking part does not wait on the long part:**

| runs where | what |
|---|---|
| **in-session** | D1, D2 (harness), D3, D4, D5, D6 — **plus the MaxPressure and Random re-derivation** (4 of 6 cells), which needs no training and is fast |
| **user's `tmux`** | the full `experiments/configs/p0_baselines.json` re-run **including MAPPO training** (the remaining 2 cells, and the N2 answer) |

**The merge gate is the baseline reproduction, not the MAPPO cell.** What actually unblocks the project
is D1/D4/D6 — the metric reader, the corrected artifacts, the naming guard. The anchors are the *sanity
reference* §7 requires before results are **accepted**, which is P4 and later. So P8.0 may merge with
4/6 cells re-derived and MAPPO explicitly marked pending, provided the MAPPO cells land **before any
phase compares against the anchors**. Record the pending status in the Return Packet, not as a
silently-empty row.

**Baselines are the real validation anyway.** `algorithms/max_pressure.py` imports no torch (verified),
and the committed MaxPressure cells have std 0. If MaxPressure and Random do not reproduce
**exactly**, the harness is wrong and the MAPPO number would be worthless regardless — so the fast half
carries the load-bearing check.

### Ruling 2 — new data directory approved, **with a provenance file.**

`docs/data/p0_baselines_horizon/` is fine and the hard guard refusing to write into the historical
directory is exactly right. Add a `README.md` inside it recording: the date, the **git hash**, the
config path, that it contains **both** quantities, and its **pin status relative to the 2026-07-09
data** (that run predates `64800fb`; this one does not). A dataset whose provenance lives only in a
commit message is the N5 defect in a new costume, and this directory exists specifically to be compared
against another one.

### Ruling 3 — importing the private helpers is approved, and it is the *correct* choice, not a tolerated one.

Importing `_train_agent` / `_baseline_chooser` / `_agent_chooser` from `experiments.runner` is the only
way to stay byte-faithful to the pipeline that produced the anchors; reimplementing training would risk
a different policy and destroy the reproduction check that is the whole point of re-running.

The usual objection to depending on a private name is that it can change without notice. **Here the
module is frozen** — it cannot change without a written, dated authorisation — so the private name is
*more* stable than a public API in a non-frozen module. Frozenness inverts the argument.

Two conditions: **(a)** put a comment at the import site stating precisely this, so a future reader does
not "clean up" the private import and silently break fidelity; **(b)** you import only — any urge to
edit `experiments/runner.py` stops and comes back to me. Note also that importing
`experiments.runner` pins the *calling* process to one torch thread as a process-global side effect;
that is expected and documented, not a leak.

### Also approved as proposed
`att_horizon` = per-episode `samples[-1]`, averaged over episodes — that matches A1 and mirrors
`evaluate_policy`'s across-episode aggregation. Renaming `ReplayResult.average_travel_time` is safe (its
only consumer is `tests/test_fixed_time_env_mapping.py`, grep-confirmed). State in the packet how you
define *entered* (presumably completed + still-in-network at the horizon) so the count is unambiguous.

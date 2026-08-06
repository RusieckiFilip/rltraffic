# BRIEF #8 — corpus format v1.1 (ATT field) + the C8 mechanical guard

**Mode:** Claude Code, on a task branch. **Branch:** `task/p2.6-format-v11`
**Issued:** 2026-08-06 by the Master chat.
**Filter:** produces corpus data — it is the re-collection that puts ATT at the ladder's top.

---

## 1. Why

Two defects, found the hard way on 2026-08-06.

**(A) ATT is absent from the corpus and cannot be added as a metric.** Contract **C8**:
`MAPPOAgent._build_global_features` (`agent/MAPPOAgent.py:619-638`) feeds the centralised critic from
**every** key in `info["metrics"]`, so `global_feature_dim = 2 + len(metrics)`. Adding two metrics took
it 3 → 5 and every MAPPO collection raised `ValueError`. `act()` builds it too (`:682`), so evaluation
breaks identically. **Cost when we learned this: 2400 of 4800 episodes.**

**(B) The width check is not a key check, and that failure is silent.** The guard compares only
`global_feature_dim`. `_global_metric_keys` is **not stored in the checkpoint** (keys: `steps_done`,
`learner`), so a loaded agent re-freezes from whatever env it is handed. **Swap one metric for another
of equal count and there is no error at all** — the critic reads different semantics under the same
indices. We were lucky to hit 3-vs-5.

The escape for (A): `info["average_travel_time"]` is **top-level and independent of the requested
metric set** (verified live with a 1-metric env). So ATT enters the corpus as a first-class logger
field, and the metric set stays frozen at 1.

---

## 2. Deliverable 1 — corpus format **v1.1**

`offline/trajectory_logger.py` is **ours, not frozen.**

Add one array, **`att_per_step`** (⚠️ **renamed 2026-08-07 — see §9; it is NOT `average_travel_time`**),
shape **`(T+1,)`**, `float32`, read from `info["average_travel_time"]`.

The **on-disk `.npz` key, the `Episode` field and every reference in `offline/` are all
`att_per_step`.** Only the *read* keeps the registry name, because that is the `info` key:
`info["average_travel_time"]`. This is the one place where the on-disk name deliberately departs from
the `info` key name — the lane arrays' "key names used unchanged" convention does not extend here, and
§9 says why.

**Alignment — write this into the docstring the way C6's block is written, because this is the most
alignment-sensitive addition since `local_reward`:**
> `att_per_step` is an **observation**, not an outcome. It carries the per-step registry metric
> `average_travel_time`, read from the top-level `info["average_travel_time"]` key at the same
> callback that records `vehicle_count` and `sim_time`, and it occupies a **T+1 row** like them —
> never a T row like `action`, `global_reward` or `local_reward`. Row `t` is the network state
> *on arrival at* decision step `t`; row `T` is `att_horizon`, the registered primary metric (A1).
> **`att_per_step.mean()` is `att_running_mean`, the legacy quantity A1 exists to retire — it is never
> the reported metric.** It is **not** `average_time_of_journey`.

- Bump `FORMAT_VERSION` to `"1.1"`.
- Do **not** change any existing array, name or dtype. v1.1 is v1.0 plus one field.

**Tests.** ⚠️ **The monotonicity test originally specified here was FALSE and is withdrawn** — see
§8. Replace it with per-row equality against an independent source:

1. shape `(T+1,)` and `float32`;
2. the horizon row equals a rollout's final `info["average_travel_time"]`, recomputed independently;
3. **every row equals the env's emitted sequence, exact `==`** — sourced from the env's emitted
   snapshots, not from the logger's own copy;
4. **observation-not-outcome:** `len(att_per_step) == len(global_reward) + 1`, and row 0 is the
   value at `reset()`;
5. a v1.0 file still loads, with the field `None`.

**Fixture requirement, and it is load-bearing:** `FakeTrafficEnv` currently hardcodes
`average_travel_time: 12.5`. A test written against a constant passes vacuously. Script a
**non-monotone** ATT sequence in float32-exact increments, and set `metrics["average_travel_time"]` to
a *different* value, so that "the logger read the metrics dict instead of the top-level key" is a
**detectable mutation** rather than an invisible equivalence.

## 3. Deliverable 2 — the C8 guard, mechanically

**AUTHORISATION (quote verbatim in the Return Packet).** Granted only if the Master chat has recorded
it; if `docs/patches/README.md` has no entry for it, **stop and ask**.

> **AUTHORISATION C — frozen-file authorisation, 2026-08-06.** The Master Coordinator authorises
> modification of `agent/MAPPOAgent.py` **only**, on branch `task/p2.6-format-v11`, for the single
> purpose of:
> **(a)** persisting `self._global_metric_keys` in `save()`;
> **(b)** in `load()`, asserting **set**-equality against the env's metric keys when the field is
> present, and emitting a **loud warning** when absent (pre-migration checkpoints).
> No other change: no behaviour change, no new state, no touching `_build_global_features`. Ships as a
> patch under `docs/patches/`. Migration of the existing 60 checkpoints is done by **our** script, not
> by the patch.
>
> **ADDITION — the error message must name the difference.** Not "metric keys do not match", which
> sends the next person back into the code. Print the checkpoint's set, the env's set, and the
> **symmetric difference**. This failure will surface months from now to someone without today's
> context; the message should end the investigation rather than start it.

Absence-warns / presence-asserts is what makes this safe to land mid-phase: the 60 existing
checkpoints keep working, loudly, and new ones are protected.

**Deliverable 2b — migrate the 60 existing checkpoints with *our own* script** (no frozen edit
needed; a checkpoint is a plain `dict`). Inject the metric key list they were trained with — the
default single-metric set, **read from the env rather than typed as a literal** — and re-save.

### ACCEPTANCE CRITERION — not one test among many

**The guard is accepted only if this case passes: metric keys differ, count identical.** That is the
case the existing width check cannot see, and it is the entire reason for the patch. Construct an
agent, save, then reload against an env whose metric **keys differ but whose count is the same**, and
assert it raises with a message naming the symmetric difference.

**It must FAIL with the assertion removed.** A guard that passes its own test when deleted is exactly
the pattern `scripts/check_test_hygiene.sh` exists to catch. Paste both outputs.

### Migration must be verified too — the migration script is ours and unreviewed

A migration that writes the field but writes it **wrong** would satisfy the presence check while
recording nonsense, turning the guard into a rubber stamp. Three checks:

1. **Self-consistency against the checkpoint's own witness.** `learner["global_feature_dim"]` is
   already stored (verified: `3` for the shipped checkpoints). Assert
   `len(migrated_keys) == global_feature_dim - 2`. This catches a wrong key *count* with no external
   source of truth.
2. **Round-trip.** Save → load → the recorded keys survive unchanged.
3. **Derivation, not transcription.** The migration reads keys from an env constructed from the
   **recorded training config**, never from a literal and never from whatever env happens to be handy.
   A same-count-wrong-keys migration is invisible to (1) and is caught only at use time by the
   load-time assert — so the derivation source is the control that prevents it.

## 4. Deliverable 3 — re-collect at v1.1

Same campaign as 2026-08-06 with the tuned `k` (hz1x1 6, grid4x4 1, cologne3 3), **metric set
unchanged**, into `datasets_v11/`. **Do not delete `datasets/`.**

**Campaign script requirements (§7, added 2026-08-06 after this failure):** `set -e`, **and** a final
assertion that completed collections equal collections requested. Check each `out_dir` is absent or
empty before issuing its command.

**Validation gate before anything consumes it:** the six unchanged tiers must be **bit-identical to
`datasets/` on every trajectory-defining array** (states, actions, rewards, lane counts).
`fixedtime` is the **only** tier permitted to differ, because `k` was retuned. Any other difference
means something drifted — stop and report.

## 5. Deliverable 4 — linter hard-fail

Whatever exists of P2.4 at that point, or a standalone check: **a corpus containing both v1.0 and
v1.1 files is rejected**, and a v1.1 corpus missing `att_per_step` is rejected. The two corpora
must never be silently mixed.

## 6. Scope fence
- **No metric-set changes anywhere.** That is the whole point.
- Do not touch `experiments/**`, `envs/**`, `metrics/**`.
- Do not delete `datasets/`.
- Do not re-tune `k`; use the values in `docs/data/fixed_time_sweep/README.md`.

## 7. Definition of Done
- [ ] Plan file first; v1.1 logger + tests; `FORMAT_VERSION == "1.1"`
- [ ] Field named `att_per_step` everywhere (§9); `tests/test_offline_naming_guard.py` green **and not
      edited** — prove it with `git diff --stat` showing that file absent
- [ ] Patch for `MAPPOAgent` + README entry + Authorisation C quoted; **human applies**
- [ ] `output/checkpoints.pre_c8_migration/` exists before the first `--apply` (§9)
- [ ] Migration script; all 60 checkpoints carry their metric key list
- [ ] Same-width-swap test shown failing without the assertion, passing with it
- [ ] `datasets_v11/` collected; validation gate passed; the six tiers bit-identical
- [ ] Full `pytest -q`; count reported against the current baseline
- [ ] Return Packet at `docs/returns/P2.6.md`

---

## 8. Correction to this brief (Master chat, 2026-08-07)

**§2's monotonicity test asserted a property that is false, and I asserted it from reasoning rather
than measurement.** The implementer measured it on **407/407 real corpus episodes** (earliest drop at
row 4 of 361, worst −6.00) and on 6/6 live cells. Mechanism: `metrics/cityflow.py::_average_travel_time`
averages over completed **plus currently-active** vehicles, so a vehicle entering with ~0 elapsed time
pulls the mean **down**. My parenthetical "a running mean over a growing population" described a
quantity this metric is not.

**The test would also have passed vacuously**, because `FakeTrafficEnv` hardcodes
`average_travel_time: 12.5` — a constant satisfies monotone non-decreasing. So the brief would have
shipped a green test asserting something false about the corpus: precisely the class
`scripts/check_test_hygiene.sh` exists to catch, arriving via the brief rather than the code.

Replacement accepted as specified above. **The repo wins over the brief; this is that rule firing on
the brief's own acceptance text.**

---

## 9. Second correction (Master chat, 2026-08-07) — the field is `att_per_step`, not `average_travel_time`

**This brief instructed you to write code that a merged test rejects.** `tests/test_offline_naming_guard.py`
(shipped by P8.0) is an AST scan over `offline/**` enforcing the P8.0 naming ruling: the bare name
`average_travel_time` may appear **only** as the registry string being *read*
(`info["average_travel_time"]`, `get("average_travel_time")`, a `metric_names` list). It flags the name
used as an attribute, a bound name, a keyword argument, or a dict-literal key.

**All four of `docs/plans/p2.6.md` §1.1's shapes are flagged.** Falsified by the coordinator on
2026-08-07 by running the guard's own `_violations()` against exactly the specified code:

```
VIOLATIONS: 4
  :5:  bound name average_travel_time          <- Episode field annotation
  :8:  dict-literal key 'average_travel_time'  <- _build_arrays return
  :16: keyword arg average_travel_time         <- Episode(...) in load_episode
  :20: attribute .average_travel_time          <- any consumer, incl. P3
CONTROL (renamed to att_per_step, still reading info["average_travel_time"]): 0 violations
```

**Ruling: rename the field. Do not touch the guard.** The guard is not merely in the way — it is
right, and for a reason specific to this array. `ep.average_travel_time` is a `(T+1,)` series whose
most likely misuse is `.mean()`, which reproduces `att_running_mean` *exactly* — the legacy quantity
amendment A1 exists to retire — under the name of the registered primary metric. That is the P8.0
defect itself, moved down one layer from a helper function into the on-disk data format, where it
would be frozen into 4800 files and a public release. `att_per_step` joins the registered vocabulary
(`att_horizon`, `att_running_mean`) and makes the wrong aggregation visibly wrong at the call site.

**Binding consequences:**
- on-disk `.npz` key, `Episode` field, and every `offline/` reference: `att_per_step`;
- the read stays `info["average_travel_time"]` — the guard permits it, and the control above proves it;
- **no exception is added to the guard, and the guard is not edited.** An exception written for one
  file inherits everything that file may later contain (the 2026-08-04 `experiments/configs/` lesson);
- `tests/**` is outside the scan, but use `att_per_step` there too — one name, one concept;
- BRIEF_09 (P3) §1 and §2 are corrected in the same turn.

**Also added here — back up the checkpoints before migrating.** Deliverable 2b rewrites all 60 `.pt`
files **in place**. Copy `output/checkpoints/` to `output/checkpoints.pre_c8_migration/` before the
first `--apply`, and say so in the Return Packet. Cost: seconds. Reason: it is the only irreversible
step in this task, `datasets/`'s manifests record the *pre*-migration `checkpoint_sha256`, and this
project has twice shipped a tool whose failed construction destroyed the artifact it was rebuilding
(P1 NB2, P2.0). The rule that came out of those is a barrier before mutation; an in-place rewrite of
the only copy has no barrier at all.

**One informational confirmation, so nobody re-opens it:** v1.1 + the existing `vehicle_count` array
makes the corpus **A4-complete**. Amendment A4 requires every ATT cell to co-report `vehicle_count` at
the episode horizon; `Episode.vehicle_count` is already a stored `(T+1,)` array, so both halves of an
A4-compliant cell are computable per-episode offline, with no extra field and no re-run.

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

Add one array, `average_travel_time`, shape **`(T+1,)`**, `float32`, read from
`info["average_travel_time"]`.

**Alignment — write this into the docstring the way C6's block is written, because this is the most
alignment-sensitive addition since `local_reward`:**
> `average_travel_time` is an **observation**, not an outcome. It is read from the top-level
> `info["average_travel_time"]` key at the same callback that records `vehicle_count` and `sim_time`,
> and it occupies a **T+1 row** like them — never a T row like `action`, `global_reward` or
> `local_reward`. Row `t` is the network state *on arrival at* decision step `t`; row `T` is the
> horizon value, which is the registered primary metric (A1). It is **not** `average_time_of_journey`.

- Bump `FORMAT_VERSION` to `"1.1"`.
- Do **not** change any existing array, name or dtype. v1.1 is v1.0 plus one field.

**Tests:** shape `(T+1,)` and dtype; the horizon value equals a rollout's final
`info["average_travel_time"]` recomputed independently; monotone non-decreasing early in an episode
(ATT is a running mean over a growing population — assert the property, not a magic number); a v1.0
file still loads.

## 3. Deliverable 2 — the C8 guard, mechanically

**AUTHORISATION (quote verbatim in the Return Packet).** Granted only if the Master chat has recorded
it; if `docs/patches/README.md` has no entry for it, **stop and ask**.

> **AUTHORISATION C — 2026-08-06, Master chat.** `agent/MAPPOAgent.py` may be modified for the single
> purpose of making contract C8 mechanical: (a) persist `self._global_metric_keys` in `save()`;
> (b) in `load()`, **assert set-equality** against the env's metric keys when the field is present, and
> emit a **loud warning** when it is absent (pre-migration checkpoints). Nothing else: no behaviour
> change, no new state, no touching `_build_global_features`. Ships as a patch under `docs/patches/`.

Absence-warns / presence-asserts is what makes this safe to land mid-phase: the 60 existing
checkpoints keep working, loudly, and new ones are protected.

**Deliverable 2b — migrate the 60 existing checkpoints with *our own* script** (no frozen edit
needed; a checkpoint is a plain `dict`). Inject the metric key list they were trained with — the
default single-metric set, **read from the env rather than typed as a literal** — and re-save.

**Load-bearing test:** construct an agent, save, reload against an env whose metric **keys differ but
whose count is identical**, and assert it raises. That is the silent failure; a test that only covers
the width case does not test this fix. Prove it by mutation: with the assertion removed, the test must
pass and the critic must silently accept the swap.

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
v1.1 files is rejected**, and a v1.1 corpus missing `average_travel_time` is rejected. The two corpora
must never be silently mixed.

## 6. Scope fence
- **No metric-set changes anywhere.** That is the whole point.
- Do not touch `experiments/**`, `envs/**`, `metrics/**`.
- Do not delete `datasets/`.
- Do not re-tune `k`; use the values in `docs/data/fixed_time_sweep/README.md`.

## 7. Definition of Done
- [ ] Plan file first; v1.1 logger + tests; `FORMAT_VERSION == "1.1"`
- [ ] Patch for `MAPPOAgent` + README entry + Authorisation C quoted; **human applies**
- [ ] Migration script; all 60 checkpoints carry their metric key list
- [ ] Same-width-swap test shown failing without the assertion, passing with it
- [ ] `datasets_v11/` collected; validation gate passed; the six tiers bit-identical
- [ ] Full `pytest -q`; count reported against the current baseline
- [ ] Return Packet at `docs/returns/P2.6.md`

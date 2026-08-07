# Deferred items — the parked list

**Created 2026-08-06** at the user's request, because the pace is high enough that parked items stop
being visible and get rediscovered in September. **This file is the parked list; if something is not
here, it is not parked, it is forgotten.** Update it in the same turn as any deferral.

**Filter in force since 2026-08-06:** does it produce corpus data, or unblock something that does?
If neither, it waits. **September target:** C1 ladder + P4 guaranteed; P5 stretch.

---

## Does anything block P3 (dataset loader)?

**No.** P3 needs the corpus (exists, 4800 episodes), the format (v1.0, stable), and the C6 alignment
convention — **verified, but not by the identity this line used to cite.** *(Corrected 2026-08-07: the
"logged rewards reproduce from raw lane counts, misaligned control differs by 16" claim was measured on
hz1x1 only. C6 holds by a paired test, 138/138 episodes across all 69 dirs and 7 tiers, plus an exact
local-reward identity on 92/92 hz1x1+grid4x4 episodes. See the Decisions Log row of that date.)*
**P3 can start immediately** — and did, in a worktree, 2026-08-07.

Two items are *adjacent* to P3 but do not block it:
- **P2.4 corpus linter** — validates P3's input rather than gating it. Worth having first, not required.
- ~~**ATT re-collection** (`datasets_att/`)~~ — **that campaign DIED** (contract C8: every MAPPO tier
  raised). Superseded by **P2.6 / BRIEF_08**, which carries ATT as a first-class logger field at format
  v1.1 instead of as a metric. P3 codes against the v1.1 spec and gates the field on `format_version`.

---

## Parked, with why and what it costs to resume

| # | Item | Why parked | Cost to resume | Needed by |
|---|---|---|---|---|
| 1 | ~~**P3 loader brief**~~ **ISSUED 2026-08-06** as `BRIEF_09_p3_dataset_loader.md`. It was overdue by three turns — logged rather than quietly closed, because the parked list existing is what surfaced it. | — | — | done |
| 2 | **P2.4 corpus linter** | Promised across several turns, never written. The checks exist as ad-hoc scripts run on 2026-08-06 (hash uniqueness, draw disjointness, `base_seed`, reward double-compute, ladder, coverage). | ~1 day; content already specified by those scripts | Before P4 consumes data at scale |
| 3 | **Horizon re-evaluation of MAPPO checkpoints on held-out draws 1000–1099** | Needs the P8.0 horizon reader wired to `_load_agent`. | ~half a day + minutes of compute | Before the online MAPPO baseline is *reported* |
| 4 | **P7.0 / all of C3** (`BRIEF_04` written, unrun) | Off the September path; the plan already designates C3 droppable. Parity contract and brief are complete and waiting. | ~1 day for the gate | October |
| 5 | **Draw-cycling trainer + `mappo_dr`** (`BRIEF_07` §5) | Serves C2's 2×2, which is October. Carries the constraint that it must reproduce `_train_agent`'s returns bit-exactly on draw 0 or the 2×2 is confounded. | ~1 day + a training run | Before C2 |
| 6 | **`dqn` as a second learned algorithm** | Cheap credibility against "your MAPPO was weak". Already registered — config only. | one config + ~2 h training + ~15 min collection | Nice-to-have for September |
| 7 | **`dqn_pfrl` (RESCO parity)** → P11.7 | **Not a config**: `pfrl` is not installed and `experiments/registry.py` is frozen. Needs a new dependency *and* an authorised patch. | new dep + frozen patch + training | Journal version |
| 8 | **DTRL external anchor** → P11.6 | Strongest external comparison available; format adapter + their average-delay metric needs C3-grade care. | ~1 week | Journal version |
| 9 | **`classify_draw_pool` must refuse ids ≥ 1100** | Ruled 2026-08-06, not implemented. Nothing currently requests those ids. | ~10 lines + a test | Before anyone materialises outside the registered pools |
| 10 | **`collect.py` should derive draw id from provenance** | Currently moot: we use `collect.py`'s own draw mechanism, which records `flow_draw` correctly. Only bites if we point it at a materialised config. | ~half a day | Before collecting from `scenarios/draws/` |
| 11 | **P0.9 `check_english.sh` o-acute false positive** | Fires on every edit to `claude_guard.sh`. Patch route (permission-denied path). | ~1 h | Whenever the guard is next edited |
| 12 | **cologne3 peak-vehicle re-measurement** | The 2026-07-27 "532 of 536 still in network" figure is contradicted by a reproducible measurement (~42–50). Old figure unreproducible (session scratchpad). | ~10 min | Before cologne3's "unresolved state" caveat is cited |
| 13 | **cologne3 engine-seed stochasticity — mechanism unidentified** | Measured (σ=3.32 vs 0.0000 elsewhere); `laneChange` refuted. Affects P8 CIs and P2.4's duplicate detector on that scenario. | ~half a day | Before P8 statistics |
| 14 | **Extended MAPPO training toward the convergence label** | Declined 2026-08-06: the criterion failed at B=1000 and the label buys nothing C1 needs, since tiers are by measured return. | ~4.5 h | October, optional |
| 15 | **Fixed-time per-phase green splits (Webster-style)** | Declared limitation; only cycle length is tuned. | ~1 day | Only if a reviewer presses |
| 16 | ~~**Master session restart** so the two new agent rules load~~ ✅ **CLOSED 2026-08-07.** Verified in the artifact, not assumed: `.claude/agents/master-coordinator.md` contains "Grep the plan before you escalate" ×1 and "THINGS YOU NEED TO DO" ×3, and the restarted session's own instructions carry both. The decision-line rule (`f9ab0ad`) is loaded too. | — | — | done |
| 18 | **`info["average_travel_time"]` is verified metric-set-independent on CityFlow ONLY** | P2.6 §0.1 measured 3 scenarios × 2 policies × 61 rows, 6/6 bit-identical — **all CityFlow**. The supporting source argument (`_run_step_hooks` sections 1–3 run unguarded) is a claim about `metrics/cityflow.py` and does not transfer. `sumo_env.py:316` emits the key, but whether the value is correct when the metric is **not requested** is unmeasured on SUMO. | ~1 h: one paired rollout, 1-metric vs 3-metric env, `np.array_equal` on the ATT sequence | **Before any SUMO collection at v1.1** — i.e. P7.0/C3. A wrong-but-plausible ATT column in the SUMO corpus is exactly the silent-number failure this repo exists to prevent |
| 22 | **RETRO-REVIEW P8.0 — the statistics harness merged unreviewed, and queue item 0 depends on it** | §7 names "statistics harness" in its critical-path list; `docs/reviews/` holds exactly one file (`P0.3-fix.md`). P2.5, P2.2-draws and P8.0 all merged after the 2026-08-03 `docs/reviews/` convention with no review filed. **P8.0's merge gate does not discriminate the thing it introduced:** reproducing `att_running_mean` bit-for-bit proves the pipeline is unchanged, not that the horizon *extraction* is right -- a reader taking `[-2]`, a max, or the last non-NaN passes it untouched. | ~half a day, after P3's review | **Before queue item 0's numbers are believed.** Settling the suspended cells with a wrong reader is worse than leaving them open |
| 20 | **P2.4 inherits two corrected corpus checks, and must state its sample or run the population** | Ruled 2026-08-07. The alignment check is the **paired** form (aligned residual < misaligned, per episode), never the exact identity — which has 1.8× power on cologne3 and would be a test that certifies nothing. The exact form to assert is `Σ_j local_reward[t] == -Σ lane_waiting[t+1]`, on hz1x1 and grid4x4 only. Also: "every stored `avail_mask` is all-True" over all 4800 episodes. | folded into P2.4 | P2.4 |
| 21 | **Action masking never binds in the corpus — a P4/P5 claim constraint, not a task** | Measured 2026-08-07: all 460 mask streams sampled are all-True, because `acyclic` returns every green phase. A no-mask ablation on this corpus measures nothing, and the paper must not credit masked-action modelling. Masks bind under cyclic modes and P6 lane closures. | none — a constraint to honour | P4.1, P5.3, P6.1 |
| 19 | **What happens to `datasets/` (v1.0) once the v1.1 gate passes** | Undecided. BRIEF_08 says "do not delete"; after the six-tier bit-identity gate passes, v1.0 is redundant except as the equivalence reference for that gate. ~4 GiB. | a ruling, then minutes | After P2.6's validation gate, before P10.0's release packaging |
| 17 | **P2.4 needs a mixed-metric-key fixture for the C8 homogeneity check.** *(Row repaired 2026-08-07: it carried 4 cells in a 5-column table, so every cell after the first rendered one column to the left. Found by an awk field count split on the pipe character — the shape check to run after any structured edit, over the whole table rather than the row you touched. It caught a second defect immediately: the repair text originally quoted the awk command itself, and the literal pipe inside it broke the row again.)* | ✅ **Condition already satisfied — nothing to preserve.** The 2-vs-3 split from the aborted campaign is in **immutable git history at `7dc9928b7770~1`** (1249 files), and both files were read out of it on 2026-08-07 *without restoring anything*: `cf_hz1x1__random` carries 2 metric keys, `cf_hz1x1__maxpressure` carries 3. `datasets_att/` is already off disk (280 MB reclaimed) and `datasets*/` now prevents a recurrence. **Preferred fixture is synthetic anyway** — two minimal `.npz` with differing `metric_keys`, constructed in the test — because a regression fixture should not depend on scavenged wreckage. History is the fallback if a real-file fixture is wanted: `git show 7dc9928b7770~1:<path> > fixture.npz`. | none — evidence is permanent | P2.4 |

---

## Superseded tables to rebuild after `datasets_att/` lands

Both were computed on the **untuned** fixed-time controller and on `datasets/`:
- the **C1 ladder** (2026-08-06) — `fixedtime` rungs are stale on all three scenarios;
- the **state-coverage table** — `fixedtime` diversity was measured on `k=4`, now `k=6/1/3`.

The other six tiers are unaffected and must be bit-identical between the two corpora. **Mark clearly
which rungs come from the tuned run when rebuilding.**

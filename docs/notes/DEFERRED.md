# Deferred items — the parked list

**Created 2026-08-06** at the user's request, because the pace is high enough that parked items stop
being visible and get rediscovered in September. **This file is the parked list; if something is not
here, it is not parked, it is forgotten.** Update it in the same turn as any deferral.

**Filter in force since 2026-08-06:** does it produce corpus data, or unblock something that does?
If neither, it waits. **September target:** C1 ladder + P4 guaranteed; P5 stretch.

---

## Does anything block P3 (dataset loader)?

**No.** P3 needs the corpus (exists, 4800 episodes), the format (v1.0, stable), and the C6 alignment
convention (verified end-to-end on real data 2026-08-06: logged rewards reproduce from raw lane counts,
misaligned control differs by 16). **P3 can start immediately.**

Two items are *adjacent* to P3 but do not block it:
- **P2.4 corpus linter** — validates P3's input rather than gating it. Worth having first, not required.
- **ATT re-collection** (`datasets_att/`, running) — adds a metrics column; proven inert on trajectories
  (6/6 identical `episode_sha256`), so P3 code written against `datasets/` works unchanged.

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
| 16 | **Master session restart** so the two new agent rules load | Committed but inert in the running session. | seconds | Any time |

---

## Superseded tables to rebuild after `datasets_att/` lands

Both were computed on the **untuned** fixed-time controller and on `datasets/`:
- the **C1 ladder** (2026-08-06) — `fixedtime` rungs are stale on all three scenarios;
- the **state-coverage table** — `fixedtime` diversity was measured on `k=4`, now `k=6/1/3`.

The other six tiers are unaffected and must be bit-identical between the two corpora. **Mark clearly
which rungs come from the tuned run when rebuilding.**

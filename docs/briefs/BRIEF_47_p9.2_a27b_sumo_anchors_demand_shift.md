# BRIEF_47 — P9.2 (A27-B): corrected SUMO anchors for the transfer scores, and the exploratory demand-shift evaluation (A27 points 5 and 7)

**Mode:** the LEAN PROTOCOL (`PROJECT_PLAN` §7, 2026-10-15): plan and build in one go (`docs/plans/p9.2.md` committed, then
proceed; stop at G0 only on a numbered ambiguity). Tests first. Mutation testing for every number-computing path (the 1:1 pressure,
the ρ recomputation, the clause re-scoring, the demand-shift report). One session, one branch.
**Branch:** `task/p9.2-a27b` from `main`, worktree `/home/filip/rltraffic-p92`.
**Interpreter:** `/home/filip/rltraffic/.venv/bin/python` by absolute path (worktrees have no `.venv`).
**Roots (the MAIN tree's):** draws `/home/filip/rltraffic/scenarios/draws` (materialise there with `--out-root`, never into the
worktree — `DEFERRED` 55), output `/home/filip/rltraffic/output` (you create only `output/p9_2/`, `output/p9_2_runs/`), corpus
`/home/filip/rltraffic/datasets_v11` (read only). `--torch-threads 1` everywhere.
**Registration:** `PREREGISTRATION.md` A27 (f) and (h), tag `v2.8-prereg-a27`. **GATE 0 — nothing that produces a registered number
runs until `git merge main && git tag -l v2.8-prereg-a27` prints the tag.** The registration wins over this brief; say so where they differ.
**No GPU is needed** (evaluation of existing checkpoints; CityFlow and SUMO are CPU-bound). CPU workers: at most 6 at a time while
P9.1's or P9.3's evaluations run (`PROJECT_PLAN` §10, 2026-10-15); the pre-flight states the number.
**Dependency:** A27(h) evaluates `iql_t` of the `mappo1000` and `mix50` tiers once P9.1 produces them. Stage 1 (everything else)
does not wait; stage 2 (`iql_t`) runs after P9.1's M2 merge, or is reported as pending in the packet if the author's cut order (4,
then 7, then 6 — his order of 2026-10-15 evening) ends the task before then.

## 0. Frozen interface contracts
- `docs/CONTRACTS.md`; the frozen set of `CLAUDE.md` §1 — **`algorithms/max_pressure.py` is frozen: the 1:1 variant is a NEW policy
  under `offline/policies/`**, reusing `IntersectionInfo.roadlink_lanes` and `phase_roadlink_mapping` exactly as the frozen agent reads
  them (`algorithms/max_pressure.py:46-145`; `utils/sumo_utils.py:53-115`; `utils/cityflow_utils.py:85-125`).
- The SUMO regime is A15's (vType parity, teleport-free) with A23's collision rule and the observer env of
  `offline/sumo_att_reference.py` (`make_observer_sumo_env`); the alignment is A16's (`offline/backend_alignment.align_info`). None
  of it changes. **`departLane` is not set anywhere and stays unset** (measured 2026-10-15); the artifact names it as a limitation.
- `offline/transfer_curve.py`, `offline/transfer_gate.py`, `offline/rtg_calibration.py`, `offline/dt_gate.py`,
  `offline/method_tier_grid.py`, `offline/materialise_draws.py`, `offline/flow_randomizer.py`, `offline/policies/fixed_time.py`:
  UNCHANGED, import only. The reference-cell rollout and the ρ formula are theirs; if a function cannot be reused as it is, stop and
  say so in the plan.
- The committed cell artifacts are inputs by digest: `docs/data/p7_3a_zero_shot.json`, `p7_3b_anchor.json`, `p7_3d_grid4x4.json`,
  `p7_3d_reference_cells.json`, `p7_3c_grid4x4.json`, `p7_3c_finetune.json`, `p7_2b_calibration.json`, `p7_3d_calibration.json`,
  `p4_3_probe.json`, `p4_heldout_thresholds.json`. Never edited.

## 1. Why this task exists
1. **Point 5 (anchor-alignment control added after results; the registered ρ and the H3 verdicts stay the record).** Measured
   live on 2026-10-15 (`PROJECT_PLAN` §8), both backends, both scenarios: on SUMO hz1x1 each green phase is 4 links over 2 distinct
   incoming lanes (each twice) into 4 distinct outgoing lanes (each once); on CityFlow it is 2 roadlinks, each incoming lane once,
   the same 4 outgoing lanes once. **On grid4x4 the SUMO weighting is 3:1** (every one of the 160 green phases visits each released
   incoming lane three times, one link per outgoing lane), **and CityFlow's grid anchor is not 1:1 either** (an outgoing lane is fed by
   up to two roadlinks on 160 of the 256 green phases). `_phase_pressure` sums per link on both backends. The SUMO fixed-time anchor used
   `--fixed-time-k 4` (`transfer_curve.py:2202`) against CityFlow's tuned k = 6 (hz1x1) and k = 1 (grid4x4)
   (`docs/data/fixed_time_sweep/README.md`). The headline ρ = 1.8895 [1.8610, 1.9180] (P7.3a) rests on both anchors.
2. **Point 7 (exploration, not H2).** The paper says demand shift was not tested. `kn-hz` (743 veh), `qc-yn` (1,417), `sb-sx`
   (1,671) are recordings of OTHER INTERSECTIONS at OTHER HOURS (08:00, 08:00, 07:00 against bc-tyc's 10:00 on 2018-04-16) mapped
   onto bc-tyc's `roadnet.json` byte for byte (md5 `f851fb84…`, measured 2026-10-15) against bc-tyc's 2,021 vehicles; volume,
   turning mix and time profile change together, so every sentence says "demand shift", never "lower volume";
   `configs/sim/cityflow1x1_config3/2/4.json` already point at them. **They were never audited: bc-tyc's flow audit is this task's
   pre-flight** (measured 2026-10-15 already: one vehicle-parameter block per file, identical to bc-tyc's; eight routes each; start
   times within 0–3,599; single insertions — complete it with route validity against the roadnet, the randomiser's inversion check,
   and the reproduction of materialised draws). **It is not the registered Hangzhou transfer pairs, which are CityFlow-to-SUMO.**

## 2. Scope fence
- No DT is re-evaluated for point 5: the DT cells are READ from the committed artifacts and ρ is recomputed. Only the two aligned
  reference arms are rolled — `maxpressure_1to1` on BOTH backends (ρ is within-backend on each backend's own anchors, and on the grid
  the aligned formula changes the CityFlow anchor too), `fixedtime_kcf` on SUMO (CityFlow's tuned-k cells already exist).
- No change to any registered ρ, verdict, artifact or anchor; the new values are reported beside, never in place.
- No training; **no prompt rescaling and no probe arm — the DT uses the same prompt as on bc-tyc's held-out draws (zero-shot)**;
  no re-tuning of fixed-time on the shifted files (k = 6 stays, stated).
- Point 7 is EXPLORATORY and NOT H2 — the artifact's `what_this_is_not` carries both A27(h) sentences verbatim; no sentence anywhere
  calls it a test of C2.
- No SUMO configuration change of any kind. No hz4x4.

## 3. What the task produces
### 3.1 `offline/policies/max_pressure_1to1.py` (C1, with tests)
A policy factory registered for the evaluation routes (and into `offline.collect.POLICIES` as `"maxpressure_1to1"` if collection is
the route you reuse), computing per phase: Σ_{distinct incoming lanes released by the phase} count − Σ_{distinct outgoing lanes fed}
count, over the SAME `roadlink_lanes` / `phase_roadlink_mapping` / lane-count source the frozen agent uses, with the frozen agent's
phase-to-action translation (`_green_phases`, acyclic controls) reproduced by calling the frozen agent's helpers where they are
static, never by copying their bodies. Provenance: a digest of the roadlink structure and the measured per-phase multiplicities
(incoming visits / distinct, outgoing visits / distinct) written into the cell records.
**Load-bearing test:** on CityFlow hz1x1 the variant's actions equal `MaxPressureAgent`'s on every step of a held-out episode, and
its held-out cells reproduce the registered MaxPressure cells of `docs/data/p4_heldout_thresholds.json` under `==` on draws 1000–1099
(the multiplicities are 1 there — proven by the measurement, re-measured by the test). On SUMO hz1x1 (the parity cfg) a test
measures multiplicities 2 / 1 per green phase; on SUMO grid4x4 3 / 1; on CityFlow grid4x4 1 / ≤ 2 — and shows on a synthetic
lane-count vector that the two formulas differ by exactly the duplicated counts. The grid identity is NOT expected on either backend:
the variant's CityFlow grid cells are new cells, reported beside the registered anchor.

### 3.2 `fixedtime_kcf` on SUMO (C2)
The equal-split plan of `offline/policies/fixed_time.py` at CityFlow's tuned k (6 on hz1x1, 1 on grid4x4), rendered in SUMO through
the same route as the registered k = 4 anchor (read `transfer_curve.py:2202` and its `COLLECT_SETTINGS`), as a new reference arm name.
Test: the plan's cycle on SUMO equals the CityFlow plan's at the same k (phase sequence and per-green steps), by comparing the
action sequences of the two backends' controllers for 360 steps.

### 3.3 `offline/sumo_anchor_rescore.py` (C3–C4)
- `roll-anchors` — SUMO: the two arms on the 100 held-out parity draws of hz1x1 and grid4x4 (P7.3a's and P7.3d's draw sets, read
  from the committed artifacts), one episode per draw as the registered reference arms were rolled (`fixedtime` / `maxpressure` were
  1 × 100 each), in the observer env, with A23's teleport/collision rule (a collision is recorded and the episode kept; an
  unregistered teleport refuses the cell and stops for a ruling); per episode `E_sumo`, `att_env`, the companions the registered
  cells carry. CityFlow: `maxpressure_1to1` on the same 100 draw ids of both scenarios by the P8.4b route (both definitions), beside
  the registered CityFlow anchors (hz1x1: identical under `==`; grid4x4: a new value).
- `rescore` — reads every committed transferred-DT cell set (P7.3a both subjects and every prompt arm; P7.3b `anchor_k200`; P7.3d
  `b_mean_k100`; P7.3c's arms) and recomputes ρ = (ATT_fixedtime − ATT_policy) / (ATT_fixedtime − ATT_maxpressure) per draw and
  pooled exactly as the registered artifact computed it (reuse its function), under the three anchor pairs of A27(f) — (MP 1:1, FT
  k = 4), (MP per-link, FT kcf), (MP 1:1, FT kcf) — each side of ρ on its own backend's anchors, beside the registered pair; H3's
  clauses re-scored by their registered rules (A19/A20 on hz1x1; A24 on grid4x4 with G recomputed under the pair; quote them in the
  plan), per subject and per scenario, with per-draw blocks and P5.2c's HELD / FAILED / NOT RESOLVED vocabulary. The grid outcome
  sentences of A27(f) (clause 1 can FAIL under the tuned cycle; clause 3's criterion moves with G) must be answerable from the report.
- `report` — `docs/data/p9_2_sumo_anchors.json`: inputs by digest; per-episode values of the two new anchors AND of the two registered
  anchors (copied from the committed artifacts) on every draw; every ρ under every pair with CI; the clause verdicts; the
  multiplicities per backend and scenario; A27's two labels; `what_this_does_not_say` (no DT re-evaluated; the registered ρ stands as
  the registered value).

### 3.4 `offline/demand_shift.py` (C5–C6)
- `materialise` — draws 1000–1099 for `configs/sim/cityflow1x1_config3.json` (kn-hz), `_config2.json` (qc-yn),
  `_config4.json` (sb-sx) via `offline.materialise_draws --env-config … --draws-range … --out-root /home/filip/rltraffic/scenarios/draws`
  (scenario keys `cityflow1x1_config3/2/4`); refuse to write anywhere else.
- `audit` — the pre-flight of §1.2 on the three files, written into the artifact before any draw is materialised.
- `evaluate` — on each file's draws 1000–1099: `dt` (`output/p4_dt/dt_seed{101..505}.pt`) with its registered in-domain prompt,
  no rescaling;
  `bc`, `bc_top10`, `iql` of `mappo1000` (`output/p4_4/checkpoints/`); `dt`, `bc`, `bc_top10`, `iql` of `mix50`
  (`output/p4_7/checkpoints/mix50_*`); stage 2: `iql_t` of both tiers from P9.1; references `fixedtime` (k = 6, the target
  directory's `signal_plan_template.txt` — identical greens, asserted), `maxpressure`, `random` (the registered anchor scheme of
  `method_tier_grid._random_factory` — one generator per draw from the training seed, five seeds; `dt_gate` has no such factory) AND
  `random_fresh` (A27(d)'s scheme: `np.random.default_rng([seed, draw_id])` driving `collect._make_random`'s sampler, one generator per
  (seed, draw); import P9.1's factory once its M1 is merged, otherwise write the same formula in your module — in either case record the
  reference's `action_sequence_sha256` on (seed 101, draw 1000) in the artifact so the two implementations can be checked against each
  other; the schedule depends only on (seed, draw) because every legal-action mask on these networks is all-true). Both random
  references are reported beside each other, as A27(d) requires of every worse-than-random check. Both ATT definitions by the two routes P9.1 uses (the P4.6 route and P8.4b's observer route,
  `att_ours` agreeing under `==`). Checkpoint digests recorded; the training seeds are the evaluation seeds (CityFlow is deterministic
  on this network — assert σ = 0 across engine seeds on one cell, as P0.2 found).
- `report` — `docs/data/p9_2_demand_shift.json`: `label: "exploration, not H2"`; per-episode records for every cell (both definitions,
  companions); ρ within CityFlow on each file's OWN anchors and raw ATT; the audit; `what_this_is_not` with A27(h)'s three sentences
  (not H2; trains nothing; not the CityFlow-to-SUMO transfer pairs) and A25(a)'s amended sentence verbatim; no single-definition claim (A11's gate was passed on bc-tyc's
  demand only).

## 4. Tests (first; minimum)
1. §3.1's load-bearing identity on CityFlow hz1x1 (actions and `==` on the registered threshold cells); the SUMO multiplicity test;
   a synthetic-intersection test where the two formulas differ by a known amount. **Mutant:** make the variant sum per link — the
   synthetic test must fail and the SUMO test must fail; the CityFlow identity must still pass (that is the point: the mutant and the
   frozen agent coincide there — say so in the packet).
2. §3.2's cycle equality across backends at k = 6 and k = 1.
3. ρ recomputation: for the registered anchor pair, `rescore` must reproduce every committed ρ under `==` (a tautology guard that
   proves the function is the registered one); mutants on the formula's sign, denominator and pooling; clause rules with mutants.
4. Demand shift: `materialise` refuses an out-root inside the worktree; the audit's checks each fail on a planted defect (a second
   vehicle block, a route absent from the roadnet, an out-of-range start time); the fixed-time plan identity across the four
   scenario directories; one bc-tyc held-out cell reproduced under `==` through the evaluation path before any shifted file is
   evaluated (the tautology guard); report T-regress.

## 5. Gates, in order (lean)
- **G0 — plan** committed; no stop.
- **G1 — code** (C1–C6 with tests and mutants, suite green, hygiene and English checks) → `docs/returns/P9.2-code.md` → one review
  mandate at most → merged to `main` only together with M2 unless the coordinator says otherwise (nothing downstream waits on it).
- **G2 — pre-flight → token.** `docs/returns/P9.2-preflight.md`: one registered SUMO reference cell reproduced under `==` through
  your rollout path (proves it is the registered route); the measured cost of one SUMO episode per scenario with your worker count
  (P7.3's costing: 11.6 s single-worker on hz1x1, grid4x4 2–6× that); the demand-shift cell count (10 learner cells × 5 seeds × 100
  draws per file, plus references) and cost from one timed CityFlow episode per file; the exact tmux commands (two
  sessions: `p92-sumo`, `p92-shift`), resume behaviour. The coordinator relays command and token request in one message; the run is
  cleared by an Amendment to this brief.
- **G3 — the runs** under the token; status from disk.
- **G4 — artifacts and packet** `docs/returns/P9.2.md` with the "for the paper" section: every ρ beside its registered value, which
  clause verdicts move under which anchor pair, the demand-shift ρ per file, and which draft sentences change.
- **G5 — one review, at most one fix round → merge → CI ceiling command.**

## 6. Definition of Done
- [ ] A27(f): `maxpressure_1to1` rolled on both backends and `fixedtime_kcf` on SUMO, on the 100 held-out draws of both scenarios;
      every transferred-DT ρ recomputed under the three pairs beside the registered one; H3's clauses re-scored on both scenarios
      (grid: clauses 1–3 with G recomputed); multiplicities measured on both backends and scenarios; `departLane` named as a
      limitation; the registered ρ and verdicts untouched.
- [ ] A27(h): the audit of the three files; zero-shot cells of the `mappo1000` and `mix50` checkpoints (dt with its in-domain prompt,
      bc, bc_top10, iql; stage 2 `iql_t`) on kn-hz, qc-yn, sb-sx, draws 1000–1099, five seeds, both definitions; references on the
      same draws; labelled exploratory, never H2, never the transfer pairs; A25(a)'s amended sentence in the artifact.
- [ ] Per-episode records for every new cell and every compared registered anchor; no registered artifact changed; zero frozen-file
      changes; zero new dependencies; suite green; packets written with the AI-assistance record.

## 7. Return Packets
`docs/returns/P9.2-code.md`, `docs/returns/P9.2-preflight.md`, `docs/returns/P9.2.md` by `docs/returns/TEMPLATE.md`.

---

# ⚠️ AMENDMENT A — 2026-10-16, gates G1 and G2: FIX FIRST, small (commit C8) — the run path passes review, but the report path never re-scores hz1x1's clause 3 and three guards are weaker than stated; the runs are cleared from a run worktree at C8's commit once the coordinator has verified it; C2, C3 and C6 confirmed; ρ_random stays outside the fence; F1 accepted; F2 is the coordinator's error

Written against `docs/returns/P9.2-code.md` and `docs/returns/P9.2-preflight.md` at `b1207c7`, read from disk, and against the code
and artifacts themselves. Merge `main` to read it.

## A.1 What the coordinator verified, this session, from disk
- **The branch:** `task/p9.2-a27b` at `b1207c7`, 16 files, every one added; the frozen set and §0's import-only modules unchanged
  (`git diff --stat main...b1207c7 -- <your packet's list>` prints nothing, re-run here).
- **A mutant of the coordinator's,** in a scratch git copy beside an unmutated control copy (a kill = fails in the mutant, passes in
  the control): `set_pressure` replaced by the frozen per-link formula → **7 killed, exactly your M1's seven** (the four synthetic
  tests, `test_sumo_hz1x1_every_green_visits_two_incoming_lanes_twice_and_four_outgoing_lanes_once`,
  `test_sumo_grid4x4_every_green_visits_each_incoming_lane_three_times`,
  `test_cityflow_grid4x4_feeds_an_outgoing_lane_twice_on_every_selectable_green`); the control passes 16 of 16; the CityFlow hz1x1
  identity tests survive, as designed.
- **Not yet counted:** the coordinator's re-run of the seven task files with the gates set, and a second mutant (`random_fresh_factory`
  seeded by `default_rng(seed)` without the draw id). Both runs HUNG — futex waits at about 0 % CPU, in the control copy as well as
  the mutant copy, while about five test processes ran at once (load average 33.6 at 11:46) — and nothing was read from them. They
  are re-run alone at C8's verification. *(The hang is environmental, not this branch's: the unmutated control hung the same way.
  It is the round's risk, `PROJECT_PLAN` §9 of 2026-10-16.)*
- **The drivers read line by line:** dirty tree → (shift: committed audit) → token → canary → (sumo: the replica gate) / (shift:
  materialise from the main tree's root with the run tree's code, then the guard) → token consumed → one tmux window per shard;
  `set -euo pipefail` stops a failing step inside `| tee` (the reviewer reproduced it).
- **Every run target is absent:** `output/p9_2`, `output/p9_2_runs`, `scenarios/draws/cityflow1x1_config{2,3,4}`,
  `/home/filip/rltraffic-p92-run`. bc-tyc's registered draws live in `scenarios/draws/cityflow1x1/`, which `materialise` cannot write.
- **The audit precedes any draw:** `docs/data/p9_2_demand_shift_audit.json` committed at `cd8031f`; its vehicle counts 743 / 1,417 /
  1,671 against 2,021 are A27(h)'s 37–83 %.

## A.2 The review (one mandate, the coordinator's; read-only; 15 minutes) — PASS on both run paths; four findings, ruled
Verified with no defect by the reviewer: `maxpressure_1to1` is A27(f)'s set semantics with everything else inherited; the SUMO
episode is `run_cell`'s anchor branch step for step; the replica gate covers all 29 (31) fields under `==` and cannot pass
vacuously; `fixedtime_kcf` reads k from the manifests and pins it against the README; A23's collision rule, `git_dirty`, atomic
writes and validate-before-skip on resume; the demand-shift run path (fresh `default_rng([seed, draw_id])` inside the factory, the
registered `random` untouched, both routes under `==`, the registered prompts, the plan digest, the out-root/cwd/audit refusals);
ρ through the registered `_rho_pair`, the three pairs exactly, the tautology guard not vacuous; the clause rules.
**What C8 must do — each with a test written first and shown red for the right reason, and one mutant (the fix removed → its test
fails):**
1. **MAJOR — hz1x1's clause 3 is never re-scored under any pair** (`sumo_anchor_rescore.py:866-868`; `rescore_pair`, around
   `:1300-1305`, builds hz1x1's H3 from clauses 1 and 2 only). **RULING — the coordinator's reading of A19(a), not a registration
   change:** A19(a) made clause 3 VOID *"because its premise is false"*, and that premise — the zero-shot policy trails MaxPressure —
   IS clause 2's prediction (ρ < 1). So under each pair, hz1x1's clause 3 is derived from clause 2's outcome under the same pair, in
   P5.2c's vocabulary: clause 2 FAILED (ρ's interval entirely above 1) → clause 3 `VOID`, agreeing with the registered verdict;
   clause 2 HELD → `PREMISE HOLDS UNDER THIS PAIR — NOT SCOREABLE: no few-shot curve was registered on hz1x1`; clause 2 NOT
   RESOLVED → `PREMISE NOT RESOLVED UNDER THIS PAIR — NOT SCOREABLE`. The last two differ from the registered VOID and are reported
   as differing. Every entry carries `agrees_with_registered`; the registered pair must reproduce VOID (the tautology guard covers it).
2. **The gate and guard records' provenance is checked wherever they authorise anything** (`:1886-1889` `roll-anchors`, `:1714-1719`
   `build_report`, and the demand shift's `evaluate` and `report`): a record counts only if its `git_commit` equals the run tree's
   HEAD or differs from it only under `docs/` (J1(c)'s rule), `git_dirty` is False, and it holds every declared check (the four
   replica checks; the guard's arms and its engine-seed checks).
3. **The drivers never consume the token past a failed record:** where a gate/guard record already exists, the driver reads its
   `passed` flag before `rm -f "$TOKEN"`; a failed record exits 2 and keeps the token (`p9_2_sumo_anchors.sh:52-57`, the shift
   driver's guard block). The reviewer reproduced the hole on a scratch replica: a re-run after a failed gate consumed the token
   and exited 0.
4. **The CityFlow branch of `validate_anchor_cell` (`:592-610`) checks `decisions == 360` and `engine_seed_requested == 1000`**, as
   the SUMO branch does: a truncated CityFlow grid4x4 cell, which has no registered counterpart, would otherwise validate.
5. **The replica gate also exercises the halting path:** draw 1001 has `halting_checked` false, so the gate never runs the path draw
   1000 runs. Add draw 1000 for the registered `maxpressure` and `fixedtime` on both scenarios (four more SUMO episodes, about 3.5×
   each), every field `==` the committed draw-1000 records.
Not reached by the reviewer, so open for G5's review: `_h3_summary` and `_sentence_values`; `evaluate_cells`, `file_block`,
`_cityflow_rho`; whether a DT factory keeps state between the two routes (if it did, the routes would disagree and the campaign
would stop — wasted time, never a wrong number).
**Why fix first rather than run now:** both artifacts are built from the run worktree, and `report` refuses a chunk rolled by code
that differs from HEAD outside `docs/` (J1(c)). A fix landed after the run would make the run's own chunks unreportable. A few hours
now, no re-run later.
**Commits:** C8a (`offline/sumo_anchor_rescore.py`, `offline/demand_shift.py` and their tests) and C8b (the two drivers and
`tests/test_p9_2_drivers.py`), then the seven task files green with the gates set, then **"P9.2 C8 ready"** to the author. The
coordinator then re-runs the seven files and the two mutants alone, reads C8, creates the run worktree at C8's commit, and clears
the token in Amendment B.

## A.3 Rulings on the packet's open questions
1. **C2 — CONFIRMED.** H3's clauses re-scored on ρ's interval (A19(a)'s reading), in P5.2c's vocabulary, with the paired ATT
   comparison beside and a disagreement flag: that is A27(f) and §3.3.
2. **C3 — CONFIRMED.** Grid clause 3 is VOID when a pair leaves G′ ≤ 0 (A19(a)'s reasoning), the (i)–(iv) arithmetic beside and
   unused. The author's grid sentence has no branch for that case (*"[still closes / no longer closes]"*): if it occurs, the G4
   packet proposes the words and the author rules them — it is his sentence. The same holds for A.2.1's two NOT SCOREABLE readings
   in the hz1x1 sentence.
3. **C6 — CONFIRMED.** The pin file (`{tier: {seed: {path, sha256}}}`) is written from P9.1's committed training record once it is on
   `main`; stage 2 then needs its own amendment and token.
4. **ρ_random — stays OUTSIDE the fence.** A27(f) enumerates the re-scored arms (P7.3a's zero-shot arms of both subjects, P7.3b's
   `anchor_k200`, P7.3d's grid zero-shot, P7.3c's few-shot arms); `random` is not among them, and a registered list is not extended
   after the fact. If the paper quotes a random ρ, the paper track raises it.
5. **F2 — the coordinator's error**, fixed under `BRIEF_49` (P5.2d), not on this branch. Your whole-suite *"4 failed"* is accepted
   as `main`'s: `main`'s CI run `38790246813` fails exactly that one test on both legs (`1 failed, 3154 passed, 397 skipped`), and
   the three tmux tests pass on CI.
6. **The run worktree** is created by the coordinator (through the permission gate) at C8's commit, after its verification.
7. **F1 — ACCEPTED, and it corrects the registration's wording, not its substance.** A27(f)'s *"every one of the 160 green phases"*
   (SUMO grid) and *"160 of the 256 green phases"* (CityFlow grid) count FILE phases, the 32 link-carrying 3-s transitions included;
   on the 128 selectable greens (16 × 8) SUMO visits each released incoming lane three times on 128 of 128 and CityFlow feeds an
   outgoing lane twice on 128 of 128 — measured by your tests, which pass on the coordinator's unmutated control. The artifact reports
   both bases (your C4); **any paper sentence uses the selectable basis.** The tagged text stands; the correction is recorded in
   `PROJECT_PLAN` §8 (2026-10-16), `DEFERRED` 118 and the paper track's deviations table. It is the file-phase versus selectable-phase
   confusion the author's point-8 edit corrected for hz4x4 (*"9 against 16"*), left standing in point 5 by the coordinator.
8. **D1–D5 and K1–K4 — accepted** as written. D2 (the variant not registered into `collect.POLICIES`) is right while P9.1 edits
   `collect.py`.
9. **F4 — the provenance race:** a `DEFERRED` row; no code change. Do not commit while your own whole suite runs.

## A.4 The token — NOT YET
Amendment B clears `p92-sumo` and `p92-shift` stage 1 from the run worktree at C8's commit, with your packet's §4 commands. Until
then nothing is materialised and nothing is rolled. **Machine load, from now on:** at most one pytest process at a time in your
session, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; no mutation campaign while P9.3's teachers or any shard runs.

## A.5 G3 → G4 (unchanged in substance)
Status from disk (`status` subcommands). At G4 both artifacts are built from the run worktree and copied byte for byte (your §4).
The "for the paper" section answers A27(f)'s two sentences and A27(h)'s, every ρ beside its registered value, every moved verdict
named with the pair that moves it (hz1x1's clause 3 by A.2.1); the multiplicity sentence uses the selectable basis (A.3.7).

---

# ✅ AMENDMENT B — 2026-10-16, C8 verified: `p92-sumo` CLEARED from the run worktree at `03b722c`; `p92-shift` waits for ONE more commit, C9 — the stall remedy for the engine-destructor hang — and is cleared by Amendment C

Written against `docs/returns/P9.2-code.md`'s C8 section and `docs/returns/P9.2-preflight.md` at `03b722c`, read from disk.

## B.1 C8 — verified by the coordinator, from disk
- **The seven task files, every simulator gate set, one process, one thread, in a scratch git copy of `03b722c`: 214 passed,
  1 skipped** (the artifact T-regress, gated until G4) in 235 s.
- **Two mutants of the coordinator's, each beside an unmutated control copy:** `random_fresh_factory` seeded by `default_rng(seed)`
  without the draw id → killed by `test_the_random_fresh_schedule_depends_on_seed_and_draw_only` and
  `test_the_random_fresh_schedule_is_default_rng_seed_draw_by_hand` (control 61 / 61); hz1x1's clause 3 mapping clause 2 FAILED to the
  "premise holds" reading instead of VOID → killed by 13 tests, the registered-pair guard
  (`test_the_registered_pair_reproduces_every_committed_rho_interval_and_clause_block`) among them (control 51 passed, 1 gated skip).
- Your M26–M32 table, the real eight-episode replica gate (draws 1001 and 1000 reproduced on all 29 / 31 fields), and `check-gate`
  refusing the sandbox's `git_dirty` record are accepted as reported; the run worktree's record is written outside the sandbox.
- **The run worktree** `/home/filip/rltraffic-p92-run`, created by the coordinator, detached at `03b722c`, clean; its code equals
  `7e4571a` (C8b): `git diff --name-only 7e4571a 03b722c` lists only `docs/returns/`.

## B.2 The token — `p92-sumo` CLEARED; `p92-shift` NOT YET
- **`p92-sumo`** — your G2 packet's §4 (b) and (c), with `BRIEF_47 Amendment B` in the token's text. 3 shards. It rolls 400 SUMO
  cells and 200 CityFlow cells; the CityFlow ones end in `probe_episode`'s immediate engine destruction, the path exposed to the hang
  of B.3, so the author watches `status` and reports a stall rather than resuming it himself.
- **`p92-shift`** — after C9 (B.3), from a SECOND run worktree at C9's commit (`/home/filip/rltraffic-p92-run2`, created by the
  coordinator): the shift report must be built from the tree that rolled its cells (J1(c)), and `p92-sumo`'s tree stays at `03b722c`.

## B.3 C9 — the stall remedy for the demand-shift evaluation (a run-feasibility requirement, not a review finding)
CityFlow's `Engine::~Engine()` can wait forever at a two-party barrier when its controller thread has already left its loop
(`docs/notes/P5.3c_HANG_2026-09-30.md`; `DEFERRED` 104: ≈ 2 in 1,200 episodes under load). On 2026-10-16 it hung P9.3's A.7 sweep
once in 27 episodes, under load. It changes no number (the hang follows the episode; every write is atomic; the engine is seeded),
but `p92-shift` destroys about 31,000 engines — two routes per cell, each freeing its engine right after the last step — so at that
rate its shards would stall dozens of times. **Required, in `offline/demand_shift.py`, its driver and their tests (C9; do not touch
`offline/sumo_anchor_rescore.py` or its driver):**
1. `evaluate_cells` arms `faulthandler.dump_traceback_later(600, exit=True)` before each cell and cancels it after (600 s is more
   than a hundred times the slowest cell measured at G2, 3.3 s for both routes), with the shard's log as the dump's file.
2. The shift driver runs each shard under a restart loop: a shard that exits with the watchdog's dump in its log is restarted
   (resuming from the records on disk), at most 20 times per shard, each restart logged with the time and the dump's top frame; any
   other non-zero exit stops the shard as now; a restart needs no new token.
3. Tests (red first): a fake cell that blocks in a C-level wait is cut by the watchdog and the shard completes on restart; a fake cell
   that raises is not restarted; the watchdog is cancelled between cells, so a healthy shard whose total time exceeds 600 s is never
   killed; the restart cap stops a shard that hangs on every attempt.
4. One mutant per item (the watchdog never armed; the restart loop removed; the cancel removed), each killed.
Then the seven task files and the new tests green (one process, one thread) → **"P9.2 C9 ready"** → the coordinator verifies,
creates `/home/filip/rltraffic-p92-run2` at C9's commit, and clears `p92-shift` in Amendment C.

## B.4 Next for you
Status from disk while `p92-sumo` runs (no whole suite, no mutation campaign in this session while it runs: one pytest process at a
time, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`); C9 meanwhile; G4 (the anchors' artifact, from `/home/filip/rltraffic-p92-run`) once
`p92-sumo` completes.

---

# ⚖️ AMENDMENT C — 2026-10-27: the author ruled points 1–7 FINISHED IN FULL — point 7 runs; rebuild C9 (it was lost); G4 for the anchors first

**The author's ruling (2026-10-27), quoted:** *"Points 1-7: finish all in full, and record the slip past 21 Oct as a deviation."*
The slip is deviation D1 in `docs/notes/A27_DEVIATIONS.md`. Point 7 (the demand shift) is NOT cut.

1. **Finish G4 for the anchors (point 5)** as the coordinator's message of 2026-10-27 asked: `docs/data/p9_2_sumo_anchors.json` built
   from the run tree `/home/filip/rltraffic-p92-run` and copied byte for byte; the anchors part of `docs/returns/P9.2.md` with the "for
   the paper" section (every ρ beside its registered value; which clause verdicts move under which pair; A27(f)'s two sentences filled;
   the multiplicity sentence on the selectable basis, `A27_DEVIATIONS` D3). Commit, then say **"P9.2 G4 anchors ready"**. The
   coordinator recomputes the ρ blocks independently before anything is merged.
2. **Then rebuild C9 exactly as Amendment B, B.3 specifies** (your first C9 was never committed and did not survive the reboot of
   2026-10-16/27): the per-cell watchdog, the restart loop in the shift driver, the four tests red first, one mutant per item, the seven
   task files green — **one pytest process at a time, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`**. Say **"P9.2 C9 ready"**. Amendment D will
   create `/home/filip/rltraffic-p92-run2` at C9's commit and clear `p92-shift` stage 1.
3. Stage 2 (`iql_t`) waits for P9.1's committed training record (the pin file of Amendment A, A.3.3).

---

# ✅ AMENDMENT C.1 — 2026-10-27, G4 (point 5, the anchors): every number ACCEPTED on the coordinator's independent recomputation; ONE test fixed before the merge (it fails, and points at the real draws root, when the suite runs from the main tree)

Written against `docs/data/p9_2_sumo_anchors.json` (`1e7d77c`, sha256 `9faa1aaa…fafef6`, extracted from the commit and digested) and
`docs/returns/P9.2.md` (`9bba6b5`, `d5451a7`), read from disk.

## C.1.1 The coordinator's recomputation — a route that imports nothing from `offline/` (raw JSON, stdlib)
From the four committed P7.3 artifacts (every DT cell and the registered anchors, the anchors checked identical across the
artifacts of one scenario) and the 400 new SUMO cells under `output/p9_2/anchors/`:
- **Every ρ interval the artifact carries — 456 numbers** (19 arms × 2 definitions × 4 pairs × mean / low / high: P7.3a's eight
  subject–arms, P7.3b's anchor, P7.3d's zero-shot, P7.3c's nine arms): **worst absolute difference 1.8 × 10⁻¹⁵**; the registered pair
  reproduces the committed 1.8895 [1.8610, 1.9180], 1.7264 [1.6862, 1.7666], 0.8855 [0.8736, 0.8973] and 0.9564 [0.9492, 0.9635].
- **Every clause verdict, both definitions, all four pairs,** equal to the artifact's: hz1x1 HELD / FAILED / VOID everywhere (both
  subjects); grid HELD / HELD / (i) under the registered pair and `mpperlink_ftkcf` (G = 0.1145 and 0.2599, Δ₁₀₀ 0.0709 [0.0576,
  0.0842] and 0.1610 [0.1309, 0.1911]), HELD / FAILED / VOID under `mp1to1_ftk4` (G = −0.3104) and `mp1to1_ftkcf` (G = −1.8498).
- **The anchor means** (SUMO E_sumo): hz1x1 MP per-link 356.73, MP 1:1 321.75, FT k = 4 601.89, FT kcf 563.12; grid 168.03, 212.10,
  304.06, 227.98. **The denominators FT − MP** on the grid: 130.5–141.4 s (registered), 84.6–98.0, 56.8–63.4, **9.6–21.9 s
  (aligned)** — positive on every draw under every pair. **The paired seconds** on the grid (zero-shot DT − anchor): +15.59 [+13.97,
  +17.20] vs the per-link MP (DT better on 0 / 100), −28.49 [−30.10, −26.87] vs MP 1:1 (100 / 100), −120.44 vs FT k = 4, −44.36 vs FT
  kcf (100 / 100 each).
- **The new cells:** 400 SUMO cells, 0 teleports and 0 collisions on every one; **the load-bearing identity:** on CityFlow hz1x1
  `maxpressure_1to1` equals P8.4a's committed `behaviour@maxpressure` episodes (`docs/data/p8_4a_admission_escalated.json`) on all
  seven fields, 100 of 100 draws; on CityFlow's grid it is 156.69 `att_ours` / 150.21 `att_engine` (n = 100).
**G4 is ACCEPTED.** No further review mandate: the numbers are recomputed by an independent route, and the code that produced them
passed G1's review and C8's verification.

## C.1.2 FIX FIRST — one test, before the merge (a test-only commit, C8.1)
`tests/test_demand_shift.py::test_materialise_refuses_an_out_root_inside_the_worktree` passes in a linked worktree and on CI, and
**FAILS when the suite runs from the main tree** (`/home/filip/rltraffic`, run by the coordinator on the trial merge, 2026-10-27):
it passes `REPO_ROOT / "scenarios" / "draws"` as an out-root that must be refused, but in the main tree that path IS the registered
draws root, so no refusal fires; the call reaches the audit check (`FileNotFoundError: no audit at …`), and the test's second
assertion (`not (REPO_ROOT / "scenarios" / "draws").exists()`) would fail on the real draws directory. **It also hands the real
registered draws root to `materialise_shifted`** when run from the main tree; only the missing audit stops a write (nothing was
written: checked). **Fix:** build a stand-in LINKED worktree under `tmp_path` (a `.git` FILE marker, as your other materialise tests
build their stand-in tree) and pass an out-root inside it together with the stand-in registered root; `REPO_ROOT` never reaches a
function that writes. Red first: the old test run with the main tree as `REPO_ROOT` fails as above; the new one passes in the main
tree, in your worktree and in a scratch copy. One mutant: the out-root check removed from `materialise_shifted` → the new test fails.
The coordinator's grep of the seven task files found no other test that hands a real-tree path to a writing function. Commit it
alone, say **"P9.2 C8.1 ready"**; the coordinator re-runs the seven files from the main tree on the trial merge and merges the branch
(point 5 on `main`, §6's P9.2 box stays open for point 7). Then C9 as Amendment C says.

## C.1.3 The grid sentence's clause-3 slot and the seconds beside ρ′
- **The words are the author's** (A.3.2; `A27_DEVIATIONS` R5): asked of him on 2026-10-27; recorded here when he rules.
- **The seconds:** the registered sentence is quoted verbatim, and the paper places beside it — as a separate sentence, never inside
  his — the aligned anchors' denominator range on the grid (9.6–21.9 s per draw) and the paired seconds (the DT beats `fixedtime_kcf`
  by 44.4 s and `maxpressure_1to1` by 28.5 s on 100 of 100 draws, where it trails the registered MaxPressure by 15.6 s on 100 of 100).
  A ρ with a 10–22 s denominator is a ratio of near-coincident anchors; the seconds are the comparison it summarises. The artifact
  already carries both (`denominator_diagnostic`, `paired_att_e_sumo`); nothing is recomputed.

---

# ✅ AMENDMENT D — 2026-10-27: point 5 MERGED with C9 and C8.1; the author's words for the grid's clause-3 slot; `p92-shift` stage 1 CLEARED from `/home/filip/rltraffic-p92-run2` at `dafa3eb`, with the stall remedy's values CORRECTED (120 s, 60 restarts)

## D.1 Verified by the coordinator, from disk, before the merge
- **C8.1 (`dafa3eb`):** on the trial merge, the eight task files with every gate set, run FROM THE MAIN TREE (the context where the
  old test failed), one process: **409 passed** (with `tests/test_tier_statements.py`), nothing written under `output/`,
  `scenarios/draws/` or `datasets_v11/`.
- **C9 (`4b47220`):** read — the watchdog is armed immediately before each rolled cell and cancelled in a `finally`; both routes
  close their env in a `finally` (`admission_probe.probe_episode`, `dt_gate.evaluate_arm`), so an engine is freed inside the armed
  window; the restart rule reads only the bytes the current attempt appended. Your M33–M35 accepted.
- **A test gap (no number at stake; REQUIRED in part 2's packet, not before the run):** the coordinator's mutant — the banner searched
  in the WHOLE log instead of this attempt's part (`segment=$(cat "$log")`) — SURVIVES all 22 tests of
  `tests/test_demand_shift_watchdog.py` and `tests/test_p9_2_drivers.py` (control 22 / 22). Under it, a shard cut once and later
  stopped by a REAL refusal (two routes disagreeing) is restarted until the cap and then reported as a watchdog stop. Add the test
  (attempt 1 cut, attempt 2 exits 1 with a refusal and no banner → STOPPED, not restarted) and show it kills that mutant.

## D.2 The author's ruling on the grid sentence's clause-3 slot (2026-10-27), quoted: *"Clause-3 slot: P1, as you recommend."*
The grid sentence, filled: *"Under the aligned anchors the grid's zero-shot ρ is 2.8498 [2.7228, 2.9768], against the registered
0.8855 [0.8736, 0.8973], and the fine-tuned (k = 100) ρ is 3.4793 [3.3567, 3.6019], against 0.9564 [0.9492, 0.9635]; the
transferred DT still beats the tuned fixed-time plan in SUMO (clause 1 holds under the aligned pair), no longer trails MaxPressure
(clause 2), and, with no gap left to close (G recomputed under the pair is −1.85), clause 3 is void under the aligned pair, as A19(a)
voided it on hz1x1; the registered verdicts stand as the record."* (`A27_DEVIATIONS` R5.1.) Write it into `docs/returns/P9.2.md`'s
§5.4 in part 2.

## D.3 The stall remedy's values — the coordinator's error, corrected (Amendment B, B.3 set 600 s and 20 without computing the hangs)
Your C9 packet's arithmetic, checked: 15,600 cells × 2 routes = 31,200 engine shutdowns, ≈ 10,400 per shard; at `DEFERRED` 104's
rate (≈ 2 in 1,200) a shard expects ≈ 17 hangs, so a total cap of 20 stops a healthy shard with probability ≈ 0.22 (≈ 0.52 that one
of three stops), and 600 s per hang idles ≈ 2.9 h per shard. **Ruled:** the watchdog cuts after **120 s** (36× the slowest cell
measured at G2, 3.3 s for both routes; a false cut costs one re-roll of a seeded cell and changes no number) and a shard stops after
**60** restarts (P(a healthy shard needs more) < 10⁻⁹ at 17 expected; a cell that hangs on every attempt costs at most 2 h before it
stops). **No code change:** the driver's `P92_WATCHDOG_SECONDS` and `P92_MAX_RESTARTS` carry them, passed explicitly into every
shard window; the run command below sets both, and every restart line in the logs prints them.

## D.4 The token (channel (a)) — `p92-shift` stage 1 CLEARED
From the run tree `/home/filip/rltraffic-p92-run2` (detached at `dafa3eb`, clean, the committed audit present; created by the
coordinator), 3 shards, CPU only, ≈ 3.5 h of cells plus ≈ 35 min of watchdog idle per shard at the expected hang rate: the token
`output/p9_2_runs/TOKEN_p92_shift` with `BRIEF_47 Amendment D` in its text, the session `p92-shift`, and the driver started with
`P92_WATCHDOG_SECONDS=120 P92_MAX_RESTARTS=60`. Stage 2 (`iql_t`) waits for P9.1's committed training record.
**After the run (G4, part 2):** `status`, then the artifact from `/home/filip/rltraffic-p92-run2` (never from your worktree), copied
byte for byte; the packet with D.1's test and D.2's sentence. A shard that prints `STOPPED` is reported, not restarted by hand
without the coordinator.

---

# ⛔ AMENDMENT E — 2026-10-27: `main` is RED since the point-5 merge — the shift driver cannot run where `/home/filip/rltraffic` does not exist; C10 FIX FIRST, small, on your branch, while stage 1 runs

## E.1 What CI found (read by the coordinator from run `40505879736`'s junit artifacts, both legs)
- Run `40505879736` on `e4a5ab0` (the merge of `dafa3eb`): **5 failed, 3515 passed, 427 skipped** on ubuntu-22.04 AND ubuntu-24.04.
  The suite step failed, so the ceiling gate's 427 against 405 is not yet a measurement.
- The five failures are ONE defect: the five cases of
  `tests/test_p9_2_drivers.py::test_the_token_is_consumed_only_past_a_record_that_authorises` with `p9_2_demand_shift.sh` (the
  `p9_2_sumo_anchors.sh` cases pass). Each run of the driver exits 1 with
  `p9_2_demand_shift.sh: line 143: cd: /home/filip/rltraffic: No such file or directory`, where the test expects 2 or 0.
- **Cause.** The driver sets `MAIN_TREE=/home/filip/rltraffic` (line 49, C7 `fc142f8`) with no override, unlike `P92_PYTHON` and
  `P92_OUTPUT_ROOT` beside it, and materialises with `(cd "$MAIN_TREE" && … materialise)` (line 143) before the guard. C8b's harness
  (`7e4571a`) overrides the interpreter, the runs and output roots and the session, but cannot reach the main tree. On this machine
  every case therefore `cd`s into the REAL main tree and passes; on a runner without that path every case stops at line 143. No
  local run could show it: your C8b and C9 runs, C8.1's three trees and the coordinator's trial merge (D.1) all had the path.
- **No number is touched.** The shards run `/home/filip/rltraffic-p92-run2`'s driver at `dafa3eb`, where the path exists.

## E.2 C10 — required, on `task/p9.2-a27b`, NOW (stage 1 runs from its own tree and is not touched)
1. **The driver:** `MAIN_TREE=${P92_MAIN_TREE:-/home/filip/rltraffic}`, the pattern of `P92_PYTHON` and `P92_OUTPUT_ROOT`; the
   header comment names the variable. Nothing else in the driver changes.
2. **The harness:** `_Harness` creates a scratch main tree under `tmp_path` and passes it as `P92_MAIN_TREE`; the stand-in interpreter
   also records its working directory; a test asserts that the materialisation call ran in that scratch directory. Against the
   unchanged driver this fails ON THIS MACHINE (the driver ignores the variable and `cd`s into the real main tree): that is the red
   you show first.
3. **A pin:** with `P92_MAIN_TREE` unset, the driver's main tree is `/home/filip/rltraffic` (the campaign's behaviour is unchanged).
4. **A mutant:** the override removed (the literal restored) → item 2's test fails; reported killed against a control.
5. **The audit (the class, not the instance):** every OTHER `/home/filip/rltraffic` literal in the files P9.2 added or changed under
   `offline/` and `tests/` that a test reaches: name it and say how that test behaves on a host without the path (overridden,
   skipped with a named reason, or not reached). Anything that would fail there is fixed the same way, in C10.
6. **Runs:** `tests/test_p9_2_drivers.py` and `tests/test_demand_shift_watchdog.py`, one pytest process,
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; `check_test_hygiene.sh` and `check_english.sh` on the changed files;
   `claude_guard.sh --frozen-only`. Then **"P9.2 C10 ready"** with the commit.
7. **J1(c), stated again because C10 is code:** from C10 on, your branch differs from `dafa3eb` outside `docs/` (D.1's test will
   too). Every `p92-shift` artifact is built in `/home/filip/rltraffic-p92-run2` and copied byte for byte (D.4). Stage 2's cells
   must be rolled in a tree whose code equals `dafa3eb` outside `docs/`, never at your branch's HEAD. Do not touch `run2`.

## E.3 What the coordinator does at "P9.2 C10 ready"
Re-runs item 2's test and item 4's mutant from disk; merges `task/p9.2-a27b` into `main` (the branch's second merge, its parent
count verified); pushes; reads CI. Expected: the suite green on both legs, and the ceiling gate failing by design at the new skip
count. That count is then measured against run `40501015653` (405, green) and handed to the author as a patch.

## E.4 Your order
C10 now. Stage 1's report when its three shards end: at 15:25 they stood at 1,700 / 1,700 / 1,599 of 5,200 cells, about 29 cells
a minute each, so about 17:30. One watchdog restart so far: shard 2 at 13:35:53Z, top frame `roll_cell` line 628, the
`evaluate_arm` call. No deeper Python frame was dumped, so the wait was in C code run from that line; that is where the arm's env
is freed as the call returns, the place the destructor hang (`DEFERRED` 104) is expected. Then G4 part 2: D.1's test, D.2's
sentence, D.4's artifact.

---

# ✅ AMENDMENT F — 2026-10-27: C10 and G4 part 2 (point 7, stage 1) ACCEPTED and MERGED (`0e38175`); A27(h)'s slot is provisional until stage 2

## F.1 Verified by the coordinator, from disk
- **CI before the fix:** runs `40505879736` (`e4a5ab0`) and `40521786087` (`7d4cbb3`) carry the same five failures on both legs (junit
  compared): the five `p9_2_demand_shift.sh` cases of `test_the_token_is_consumed_only_past_a_record_that_authorises`; 3515 passed, 427
  skipped.
- **C10 (`c72ddea`):** the driver's one line and the harness read. In a scratch clone of the merge commit, `tests/test_p9_2_drivers.py`
  and `tests/test_demand_shift_watchdog.py`, one pytest process: control **25 passed**; the override removed → **2 failed** (the cwd
  test and the pin). **The driver with every `/home/filip/rltraffic` replaced by a path that does not exist → only the pin fails**
  (it pins the literal by design): the token tests and the cwd test no longer need the path. That is CI's condition, reproduced on this
  machine for the first time.
- **D.1's test (`3bc61bf`):** the coordinator's whole-log mutant (`segment=$(cat "$log")`) → **1 failed (D.1's test alone), 24 passed**.
- **The artifact (`b535227`):** rebuilt by the coordinator in `run2` at `dafa3eb` (`report --out` to a scratch file; `run2` clean
  afterwards): **byte-identical**, sha256 `668e827b…fa34`.
- **An independent recomputation** (the 15,600 raw records under `output/p9_2/demand_shift/`, json and numpy, nothing imported from
  `offline/`): every record at `dafa3eb`, clean, its two `att_ours` routes equal, stage 1; per file 8 learners × 500 cells, fixed-time
  100, MaxPressure 100, each random reference 500; the artifact's 15,600 cells equal the raw records. **9,936 values of the `files`
  block** (per-draw seed means, the ATT and ρ intervals, every paired field against fixed-time and both random references): **8,011
  equal under `==`, the rest within 1.4 × 10⁻¹⁵ relative** (summation order). The Wilcoxon p agreed only once the coordinator's route
  applied the continuity correction that `dt_gate.wilcoxon_signed_rank` documents; the first route had omitted it.
- **The logs:** the token consumed; shards 0, 1 and 2 each end `SHARD i EXIT: 0`; one `RESTART 1/60` (shard 2); no `STOPPED`.
- **The merge `0e38175`** (parents `7d4cbb3` and `cf98a12`): the eight task files with every gate set, run FROM THE MAIN TREE, one
  process: **225 passed, 0 skipped**, the new T-regress regenerating the artifact byte for byte among them. The merge diff's
  `/home/filip/rltraffic` literals under `offline/` and `tests/` are the driver's default (now overridable), docstrings and the pin's
  string comparisons; none is reached as a path (§7's CI rule (3)). The T-regress skips by name where `output/` is absent (CI).
- **CI on `0e38175`:** run `40548238870` — **the suite step PASSED on both legs** (3,523 passed, 428 skipped, 0 failed; the three
  guards and the wheel gate passed); only the skip-ceiling gate failed, by design (428 against 405). Measured against the last green
  run `40501015653`: +23 skips in 6 texts, all `campaign_output` (the draws tree and the gitignored `output/`), nothing removed. The
  author applies `docs/patches/ci_gate_ceiling_428_p9_2.patch`.

## F.2 Accepted: G4 part 2, stage 1 (A27(h), point 7: EXPLORATORY, not H2)
- **A27(h)'s slot reads "beat it on all three files":** each of the eight stage-1 learners' paired interval against the untuned
  fixed-time plan lies entirely below 0 on every file under both definitions; 100 / 100 draws in 47 of the 48 contrasts and 98 / 100
  in the last (`bc@mix50`, qc-yn, `att_engine`). **Provisional:** A27(h)'s *"they"* includes (c)'s `iql_t` of both tiers, so the
  slot is read again from the final artifact after stage 2. **Its rule is fixed now, before stage 2's cells exist** (`A27_DEVIATIONS`
  R6): "beat it on all three files" iff every arm's interval against fixed-time lies entirely below 0 on every file under both
  definitions; an interval entirely above 0 names the file, the arm and the definition; anything else goes to the author.
- **ρ per file**, the range over the eight learners (`att_ours`; `att_engine`): kn-hz 0.81–0.89; 0.82–0.89 · qc-yn 0.82–0.98;
  0.78–0.98 · sb-sx 1.14–1.18; 1.11–1.14. Every learner beats both random references on every file under both definitions (96
  contrasts, all HELD).
- **P2.8.2, the sb-sx observation** (every learner faster than that file's own MaxPressure, ρ > 1 on both definitions): put to the
  author through the paper track. The registered sentence's words are his and stay as registered meanwhile.
- The changed partial-declaration test (absence → the committed bytes unchanged) and the added T-regress: accepted as disclosed.

## F.3 Next for you
- Nothing to build. Keep `run2` (`dafa3eb`) and `output/p9_2/demand_shift/` untouched: stage 2 is rolled there (E.2.7).
- **Stage 2** waits for P9.1's `iql_t` trainings, and P9.1's campaign is not cleared yet (`BRIEF_46` Amendment C: a trainer refusal
  P9.3 found). When P9.1's training record for the `mappo1000` and `mix50` `iql_t` runs exists: the stage-2 pin file, a pre-flight (the
  pins' digests, the 3,000 cells, the command from `run2`), and **"P9.2 stage 2 ready"**. The token comes in an Amendment.

---

# ⚖️ AMENDMENT G — 2026-10-28: stage 2's trigger and its pins — P9.1's on-disk records, not `main`; wait with a read-only watcher

## G.1 What changed
P9.1's campaign is cleared (`BRIEF_46` Amendment D), from the run tree `/home/filip/rltraffic-p91-run-c` at `cb61ee7`.
- Its first 40 runs are hz1x1 `iql_t`: eight tiers × five seeds.
- So the ten you need, `mappo1000` and `mix50`, are runs 21–25 and 31–35 of 255. That is about four hours after the start, at
  P9.1's measured ≈ 7 min per hz1x1 `iql_t` run.

Your question ("tell me … when the record is on `main`") is answered here: **not `main`**. P9.1's records reach `main` only at its G4,
days from now.

## G.2 Rulings
1. **The pins come from P9.1's on-disk records, read-only.**
   - The records are `/home/filip/rltraffic/output/p9_1_runs/training/hz1x1/{mappo1000,mix50}__iql_t__seed{101,202,303,404,505}.json`,
     in format `p9.1-run/1.0`.
   - For each record, check:
     - its `checkpoint` exists;
     - the file's sha256, recomputed by you, equals the record's `file_sha256`;
     - `declared_steps` and `gradient_steps` are both 40,000 (`a27_arms.REGISTERED_STEPS`, which is what your evaluation expects);
     - `scenario`, `tier`, `arm` (`iql_t`) and `seed` are the ones its name says.
   - P9.1's run records carry no commit field. So the pin file names the campaign commit `cb61ee7` and the run tree, as the first line
     of `/home/filip/rltraffic/output/p9_1_runs/campaign_capture.txt` states them (`token consumed; stage campaign <stamp> from
     /home/filip/rltraffic-p91-run-c at cb61ee7; …`).
   - A record that does not exist yet is waited for. A checkpoint that differs from its record is a STOP: report it.
2. **Wait with a watcher, read-only.**
   - An until-loop counts the ten records every five minutes; `BRIEF_46` D.3's one-line check prints 10.
   - Then build the pin file and the pre-flight, as your plan has them: the pins' digests, the 3,000 cells, and the exact command from
     `run2` (`STAGE=2 IQL_T_PINS=<pin file>`).
   - Commit both, and say **"P9.2 stage 2 ready"**. The token comes in Amendment H.
3. **Stage 2 runs on CPU beside P9.1's GPU training, as stage 1 ran.**
   - Shards of one thread each, with the watchdog at stage 1's values (120 s, 60 restarts).
   - P9.1's run seconds are an uncontrolled clock (A27(a)), and its stage records its own GPU trace.
   - The coordinator sets the shard count in Amendment H.
4. **Unchanged:** `run2` (`dafa3eb`) and `output/p9_2/demand_shift/` stay untouched. A27(h)'s slot is re-read under R6 from the final
   artifact.

## G.3 2026-10-28, later: when the ten records come
The author stopped P9.1's campaign stage and ruled that every long run runs in sessions of a few hours (`PROJECT_PLAN` §7 (5)).
- The ten records now come in P9.1's FIRST TRAINING SESSION (runs 4–65, about 6 h). That session starts after P9.1's session control is
  verified (`BRIEF_46` Amendment E). They no longer come about 4 h after a 40-h start.
- Keep your watcher if it can wait that long. Otherwise stop it and wait for the coordinator's word.
- Stage 2 (about 1 h on CPU) is a session of its own. Amendment H states its hours first.
- The 2026-11-07 hard stop is withdrawn (`A27_DEVIATIONS` D8). Nothing else of yours changes.

## G.4 2026-10-28, evening: stage 2 needs no session control; the GPU order
- **Your question, ruled: build nothing.** Stage 2 (≈ 1 h on CPU) is a session by itself. `PROJECT_PLAN` §7 (5), as clarified today:
  a run planned well inside one session (at most about 2 h) needs no hours bound and no stop file, provided an interruption loses only
  the units in flight.
- **Your driver meets that already.** It writes one atomic JSON record per cell. A shard validates and skips every record that exists,
  and stops at a refused cell (`offline/demand_shift.py` `evaluate_cells`; the driver's RESUME note).
- **Amendment H's command will say two things first:** its hours, and what an interruption leaves (the cells in flight, which a
  re-run rolls again).
- **The GPU order changed** (`BRIEF_48` I.3-4). P9.3's training sessions run while P9.1 builds its session control. P9.1's first
  training session goes next once it is cleared, and your ten records still come in it.
- **Your watcher (task `bupgfrng4`):** the author leaves it to time out by itself at about 00:13. Act on nothing it reports.

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

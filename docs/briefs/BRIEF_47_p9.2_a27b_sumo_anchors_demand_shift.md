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
does not wait; stage 2 (`iql_t`) runs after P9.1's M2 merge, or is reported as pending in the packet if the author's cut order (7
first) ends the task before then.

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
1. **Point 5 (bug fix with results before and after; control added after results).** Measured live on 2026-10-15 (`PROJECT_PLAN`
   §8): on SUMO hz1x1 each green phase is 4 links over 2 distinct incoming lanes (each twice) into 4 distinct outgoing lanes (each
   once); on CityFlow it is 2 roadlinks, each incoming lane once, the same 4 outgoing lanes once. `_phase_pressure` sums per link, so
   SUMO's MaxPressure weighs upstream 2:1 against downstream where CityFlow's weighs 1:1. The SUMO fixed-time anchor used
   `--fixed-time-k 4` (`transfer_curve.py:2202`) against CityFlow's tuned k = 6 (hz1x1) and k = 1 (grid4x4)
   (`docs/data/fixed_time_sweep/README.md`). The headline ρ = 1.8895 [1.8610, 1.9180] (P7.3a) rests on both anchors.
2. **Point 7 (exploration, not H2).** The paper says demand shift was not tested. `kn-hz` (743 veh), `qc-yn` (1,417), `sb-sx`
   (1,671) share bc-tyc's `roadnet.json` byte for byte (md5 `f851fb84…`, measured 2026-10-15) against bc-tyc's 2,021;
   `configs/sim/cityflow1x1_config3/2/4.json` already point at them.

## 2. Scope fence
- No DT is re-evaluated in SUMO for point 5: the DT cells are READ from the committed artifacts and ρ is recomputed. Only the two new
  reference arms are rolled.
- No change to any registered ρ, verdict, artifact or anchor; the new values are reported beside, never in place.
- No training; no probe-calibrated prompt beyond A27(h)'s `b_mean_k100` (A17's Rule B, the registered form); no re-tuning of
  fixed-time on the shifted files (k = 6 stays, stated).
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
measures multiplicities 2 / 1 per green phase and shows the two formulas differ on a synthetic lane-count vector by exactly the
duplicated incoming counts. On grid4x4 (both backends) the multiplicities are measured and reported, whatever they are.

### 3.2 `fixedtime_kcf` on SUMO (C2)
The equal-split plan of `offline/policies/fixed_time.py` at CityFlow's tuned k (6 on hz1x1, 1 on grid4x4), rendered in SUMO through
the same route as the registered k = 4 anchor (read `transfer_curve.py:2202` and its `COLLECT_SETTINGS`), as a new reference arm name.
Test: the plan's cycle on SUMO equals the CityFlow plan's at the same k (phase sequence and per-green steps), by comparing the
action sequences of the two backends' controllers for 360 steps.

### 3.3 `offline/sumo_anchor_rescore.py` (C3–C4)
- `roll-anchors` — the two arms on the 100 held-out parity draws of hz1x1 and grid4x4 (P7.3a's and P7.3d's draw sets, read from the
  committed artifacts), one episode per draw as the registered reference arms were rolled (`fixedtime` / `maxpressure` were 1 × 100
  each), in the observer env, with A23's teleport/collision rule (a collision is recorded and the episode kept; an unregistered
  teleport refuses the cell and stops for a ruling). Per episode: `E_sumo`, `att_env`, the companions the registered cells carry.
- `rescore` — reads every committed transferred-DT cell set (P7.3a both subjects and every prompt arm; P7.3b `anchor_k200`; P7.3d
  `b_mean_k100`; P7.3c's arms) and recomputes ρ = (ATT_fixedtime − ATT_policy) / (ATT_fixedtime − ATT_maxpressure) per draw and
  pooled exactly as the registered artifact computed it (reuse its function), under the three anchor pairs of A27(f) — (MP 1:1, FT
  k = 4), (MP 2:1, FT kcf), (MP 1:1, FT kcf) — beside the registered pair; H3's clause 1 and clause 2 re-scored by their registered
  rules (quote A19, A20, A24 in the plan), per subject and per scenario, with per-draw blocks.
- `report` — `docs/data/p9_2_sumo_anchors.json`: inputs by digest; per-episode values of the two new anchors AND of the two registered
  anchors (copied from the committed artifacts) on every draw; every ρ under every pair with CI; the clause verdicts; the
  multiplicities per backend and scenario; A27's two labels; `what_this_does_not_say` (no DT re-evaluated; the registered ρ stands as
  the registered value).

### 3.4 `offline/demand_shift.py` (C5–C6)
- `materialise` — draws 1000–1099 and 201–300 for `configs/sim/cityflow1x1_config3.json` (kn-hz), `_config2.json` (qc-yn),
  `_config4.json` (sb-sx) via `offline.materialise_draws --env-config … --draws-range … --out-root /home/filip/rltraffic/scenarios/draws`
  (scenario keys `cityflow1x1_config3/2/4`); refuse to write anywhere else.
- `probe` — the in-CityFlow analogue of A17(b): MaxPressure on each file's draws 201–300, one episode per draw, `reset(seed=1000)`,
  returns recorded; Rule B's target per A17 (`S = mean`, k = 100): `target = R_best_source × mean(R_target_probe) / mean(R_source_probe)`,
  the source probe being P4.3's in-domain probe (`docs/data/p4_3_probe.json`), `R_best_source` the registered in-domain target of
  the checkpoint's tier. Quote the formula from `offline/rtg_calibration.py` and reuse its function.
- `evaluate` — on each file's draws 1000–1099: `dt` (`output/p4_dt/dt_seed{101..505}.pt`) under `naive` and `b_mean_k100`;
  `bc`, `bc_top10`, `iql` of `mappo1000` (`output/p4_4/checkpoints/`); `dt`, `bc`, `bc_top10`, `iql` of `mix50`
  (`output/p4_7/checkpoints/mix50_*`); stage 2: `iql_t` of both tiers from P9.1; references `fixedtime` (k = 6, the target
  directory's `signal_plan_template.txt` — identical greens, asserted), `maxpressure`, `random` (the registered anchor scheme of
  `dt_gate._random_factory`, five seeds). Both ATT definitions by the two routes P9.1 uses (the P4.6 route and P8.4b's observer route,
  `att_ours` agreeing under `==`). Checkpoint digests recorded; the training seeds are the evaluation seeds (CityFlow is deterministic
  on this network — assert σ = 0 across engine seeds on one cell, as P0.2 found).
- `report` — `docs/data/p9_2_demand_shift.json`: `label: "exploration, not H2"`; per-episode records for every cell (both definitions,
  companions); ρ within CityFlow on each file's OWN anchors and raw ATT; the probe returns and the Rule B targets; `what_this_is_not`
  with A27(h)'s two sentences and A25(a)'s amended sentence verbatim; no single-definition claim (A11's gate was passed on bc-tyc's
  demand only).

## 4. Tests (first; minimum)
1. §3.1's load-bearing identity on CityFlow hz1x1 (actions and `==` on the registered threshold cells); the SUMO multiplicity test;
   a synthetic-intersection test where the two formulas differ by a known amount. **Mutant:** make the variant sum per link — the
   synthetic test must fail and the SUMO test must fail; the CityFlow identity must still pass (that is the point: the mutant and the
   frozen agent coincide there — say so in the packet).
2. §3.2's cycle equality across backends at k = 6 and k = 1.
3. ρ recomputation: for the registered anchor pair, `rescore` must reproduce every committed ρ under `==` (a tautology guard that
   proves the function is the registered one); mutants on the formula's sign, denominator and pooling; clause rules with mutants.
4. Demand shift: `materialise` refuses an out-root inside the worktree; the probe reproduces P4.3's in-domain probe on bc-tyc's draws
   201–300 under `==` before any target file is probed (the same tautology guard A17(f) used); the Rule B target equals an independent
   recomputation from the raw probe returns; the fixed-time plan identity across the four scenario directories; report T-regress.

## 5. Gates, in order (lean)
- **G0 — plan** committed; no stop.
- **G1 — code** (C1–C6 with tests and mutants, suite green, hygiene and English checks) → `docs/returns/P9.2-code.md` → one review
  mandate at most → merged to `main` only together with M2 unless the coordinator says otherwise (nothing downstream waits on it).
- **G2 — pre-flight → token.** `docs/returns/P9.2-preflight.md`: one registered SUMO reference cell reproduced under `==` through
  your rollout path (proves it is the registered route); the measured cost of one SUMO episode per scenario with your worker count
  (P7.3's costing: 11.6 s single-worker on hz1x1, grid4x4 2–6× that); the demand-shift cell count (12 learner cells × 5 seeds × 100
  draws per file, plus references and probes) and cost from one timed CityFlow episode per file; the exact tmux commands (two
  sessions: `p92-sumo`, `p92-shift`), resume behaviour. The coordinator relays command and token request in one message; the run is
  cleared by an Amendment to this brief.
- **G3 — the runs** under the token; status from disk.
- **G4 — artifacts and packet** `docs/returns/P9.2.md` with the "for the paper" section: every ρ beside its registered value, which
  clause verdicts move under which anchor pair, the demand-shift ρ per file, and which draft sentences change.
- **G5 — one review, at most one fix round → merge → CI ceiling command.**

## 6. Definition of Done
- [ ] A27(f): `maxpressure_1to1` and `fixedtime_kcf` rolled on the 100 held-out parity draws of both scenarios; every transferred-DT
      ρ recomputed under the three pairs beside the registered one; clause 1 and 2 re-scored; multiplicities measured on both
      backends and scenarios; `departLane` named as a limitation; the registered ρ untouched.
- [ ] A27(h): zero-shot cells of the `mappo1000` and `mix50` checkpoints (dt × 2 prompts, bc, bc_top10, iql; stage 2 `iql_t`) on
      kn-hz, qc-yn, sb-sx, draws 1000–1099, five seeds, both definitions; references on the same draws; labelled exploratory and
      never H2; A25(a)'s amended sentence in the artifact.
- [ ] Per-episode records for every new cell and every compared registered anchor; no registered artifact changed; zero frozen-file
      changes; zero new dependencies; suite green; packets written with the AI-assistance record.

## 7. Return Packets
`docs/returns/P9.2-code.md`, `docs/returns/P9.2-preflight.md`, `docs/returns/P9.2.md` by `docs/returns/TEMPLATE.md`.

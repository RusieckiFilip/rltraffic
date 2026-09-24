# REVIEW — the PROPOSED A23 (`docs/notes/P7.3d_ATTEMPT1_READ_2026-09-24.md` §5, as first committed at `a55d8a0`) — 2026-09-24 — VERDICT: SOUND-WITH-CORRECTIONS (4 blocking, all adopted)

**Reviewer:** `contract-reviewer`, one adversarial round on a planning document before the author's approval and the tag. Mandate: FALSIFY
the proposed row against the registration (A15, A17, A18(c), A20, A21, §3.4, §8), the evidence under `output/p7_3d_runs/attempt1_*`, the
recorder, the platform metric and SUMO 1.27.1's own defaults; read-only; no outcome value of any campaign cell printed; the fenced logs
unopened; no simulation of a held-out draw; 15 minutes, checklist first, one append per finding. The findings file is filed verbatim below
the rule.

## Coordinator's ruling (2026-09-24)

1. **Blocking 1 — ADOPTED, and settled by MEASUREMENT, not by rewording.** Confirmed from the code before anything else: the SUMO probe's
   teleport counter (`offline/transfer_calibration.py:414, :429` at `4383699`; `:434, :448` at P7.2b's `b63d56b`) reads
   `getStartingTeleportIDList()` once per decision, and the env steps SUMO once per simulated second — one second in ten. So A17(b)'s
   registered `n_teleports = 0`, on which the registered hz1x1 and grid4x4 targets rest, had been verified sparsely. The coordinator
   re-rolled all 200 probe episodes (P7.2b's hz1x1 and P7.3d's grid4x4, draws 201–300) through the probe's own episode function from the
   run worktree, with `--statistic-output` and `--collision-output` added in the worker process only, after FALSIFYING the read on
   attempt 1's refused fixed-time cell (it counts 1 teleport — jam, yield and wrong-lane 0 — and 1 collision): **all 200 reproduce their registered probe returns under `==`, and SUMO's statistics count 0 teleports and 0 collisions in every one of them** (P7.2b's 100 and P7.3d's 100; the read was falsified first on attempt 1's refused fixed-time cell, where it counts 1 and 1) — A17's condition holds, now verified at every step. Evidence:
   `output/p7_3d_runs/probe_reverify_20260924/`. The counter's defect is `DEFERRED` 93.
2. **Blocking 2–4 — ADOPTED with the reviewer's replacement texts, lightly edited:** (c)(iii) states the general case (a collider on an
   earlier edge is put back further along its route — SUMO's behaviour, marked as not observed in this project); (d) makes the primary the
   sole decider of clause 1 and fixes the robustness set now (draws 1020 and 1042, removed whole); (f) excepts the provenance and clock
   fields, requires exactly (a)'s two collisions and no others, and says what happens otherwise.
3. **Non-blocking — ADOPTED:** the title reframed as a correction of fact; (c)(i)'s match defined (collider or victim, same step, any kind);
   (e) rejects `collision.action warn/none` for changing the measured configuration, not for validity; (g) sharpened to *more than twice* the
   prompt's magnitude (every reward integral, every target not — both checked blind by the coordinator) and the per-cell wall clocks named
   as seen; the collision overlaps corrected to 1.71 / 2.02 m; the scratchpad copies of the fenced logs removed after a byte-identity check;
   A17's note attached to both probes. **Two of the reviewer's "could not verify" items the coordinator verified:** CAP(E) pins all 200
   configuration digests (200 `noteleport.sumocfg` entries; draw 1000's `c27d31e8…` present), and C4's six cells MATCH over five rolls
   (C4's two, the coordinator's and the driver's pre-token checks, and the campaign's own six).
4. **Not adopted:** nothing. One round only — the rule for planning documents — and this one returned load-bearing corrections, which is
   what a round is for.

---

# Reviewer's findings file, verbatim

# A23 REVIEW FINDINGS (reviewer: independent subagent, 2026-09-24)
Target: docs/notes/P7.3d_ATTEMPT1_READ_2026-09-24.md section 5 (proposed A23)
Hard constraints: no outcome values from output/p7_3d/cells/; no fenced_do_not_read/; no SUMO sim runs; write nothing in repo trees.

## Checklist
- [x] 1. REGISTRATION CONFLICTS (A15 a/b/c/g, A17, A18(c), A20, A21, 3.4, 8; A17 probe n_teleports=0)
- [x] 2. THE MECHANISM (SUMO 1.27.1 defaults via --save-template: collision.action, check-junctions, stoptime, mingap-factor; reading of the two warning lines)
- [x] 3. THE SCORING CLAIM A23(c)(iii) (collider counted as arrived by E_sumo recorder AND att_env metrics/sumo.py; "at most one edge early" in general)
- [x] 4. THE POPULATION CLAIM A23(b) (n_teleports counts in p7_2b, p7_3a, p7_3b, grid4x4 probes, p7_3d_reference_cells; soundness of "every detected collision is a teleport")
- [x] 5. THE SENSITIVITY ANALYSIS A23(d) (well-defined, outcome-blind, forking paths, attempt-2 collision set differing)
- [x] 6. "RE-RUN IS NOT A SECOND SAMPLE" A23(f) (determinism evidence; what if shared fields differ)
- [x] 7. THE ALTERNATIVES A23(e) (collision.action warn/none dismissal)
- [x] 8. THE SEEN LEDGER A23(g) and note section 4
- [x] 9. REFEREE-REJECT / OVERCLAIM wording in section 5

## Findings (appended one per entry)

### F2.1 [item 2] 2026-09-24 SUMO defaults (partial; warning lines still to read)
Command: `/usr/bin/sumo --save-template scratchpad/sumo_template.xml` (SUMO 1.27.1), grep collision|teleport.
Observed (sumo_template.xml:145-166): collision.action=teleport; collision.stoptime=0; collision.check-junctions=false;
collision.check-junctions.mingap=0; collision.mingap-factor=-1 ("negative -> carFollowModel parameter used");
intermodal-collision.action=warn; time-to-teleport=300 (help: "non-positive values disable teleporting");
time-to-teleport.highways=0; time-to-teleport.disconnected=-1; time-to-teleport.remove=false; max-num-teleports=-1.
Judgement: note's defaults for action/check-junctions CORRECT; stoptime=0 means the teleport is immediate, which the
inference "every detected collision is also a teleport" needs (A23(b) does not state stoptime). Note: other teleport
kinds exist besides jam and collision (highways = 0 i.e. off; disconnected = -1 off) -- relevant to A23(c)(i) wording.

### F1.1 [item 1] 2026-09-24 Registration conflicts -- no registered text REQUIRES refusal; one internal inconsistency
Commands: `sed -n '382p;383p;385p;386p;388p' PREREGISTRATION.md`; Read 269-293 (sec 8), 160-179 (sec 3.4);
grep -P '(teleport|collision|n_vanished|refuse)' over PREREGISTRATION.md.
Observed:
- A15(c) (388) is OPERATIONAL: "Every SUMO measurement ... runs a configuration carrying <time-to-teleport value="-1"/>".
  Only the HEADING says "the SUMO regime is TELEPORT-FREE"; rationale col: "A 300 s teleport removes a stuck vehicle from
  one engine's metric and never from the other's". Nowhere does the registration say a cell WITH a teleport is refused;
  the refusal is BRIEF_39/instrument code. So A23(c) does not contradict registered operative text.
- BUT A23's title "A15(c)'s 'teleport-free' regime MEANS free of JAM teleports" re-defines a registered word after the
  fact. Honest form = a factual correction: A15(c) believed the config disables all teleports; it does not.
- A15(a) (388): vanished-without-arrival censored at T -> collider counted as arrived escapes this clause (see item 3).
- A17(b) (386): "every probe episode must show n_teleports = 0" is registered for the hz1x1 probe (= P7.2b's 100,
  draws 201-300); it reaches grid4x4's probe only via A20(b)-(c)'s reference to A17(e). A23(b) attaches the
  parenthetical "(A17's n_teleports = 0 condition, met)" ONLY to the grid4x4 probe -> should attach to BOTH P7.2b's
  100 and the grid4x4 probe. No conflict: all probe episodes already met it; A21(d) says no grid4x4 collection.
- A18(c) (385) seed rule: A23(e)'s "re-seeding contradicts A18(c)" is correct.
- sec 8 (288-291): kept cells consistent with "never dropped"; sensitivity analysis not forbidden (A18(d) precedent of a
  registered secondary). 
- CONFLICT (internal): A23(d) "If the two disagree ... both are reported and neither is chosen" changes the DECISION of a
  CONFIRMATORY test (A20(e): clause 1 confirmatory on grid4x4; sec 8 primary test = Wilcoxon signed-rank + Holm), while
  A23(h) lists A20 as UNCHANGED. Also the disagreement criterion names "inequality" and "CI excludes 0" but NOT the
  registered primary test (Wilcoxon, sec 8:280, Holm sec 8:283) -> forking path.
Judgement: SOUND-WITH-CORRECTIONS on item 1; (d)'s "neither is chosen" must be rewritten (primary decides; sensitivity
qualifies robustness) and the title reframed as a correction of a false belief, not a redefinition.
- [x] item 1 ticked

### F2.2 [item 2] 2026-09-24 The two warning lines and collision records
Command: cat -n attempt1_diag/{README.md,*_warnings_only.txt,*_stderr.txt,vanished_counters.txt,ft1020_collisions.xml}
Observed: ft1020_sumo_warnings_only.txt:1-3 = "Vehicle '628' performs emergency braking on lane ':D0_36_0' with decel=9.00,
wished=4.50" / "Teleporting vehicle '628'; collision with vehicle '969', lane='D0right0_0', gap=-4.21, time=2298.00,
stage=move" / "Vehicle '628' teleports beyond arrival edge 'D0right0'". dt1042 identical shape (1126/1098, A0left0_0,
gap=-4.52, t=2627, beyond arrival edge 'A0left0'). ft1020_collisions.xml:43 type="collision" lane=D0right0_0 pos=0.61,
colliderFront x=1214.21, victimBack x=1212.51 -> PHYSICAL overlap 1.70 m; SUMO's "gap" -4.21 = -1.70 - minGap 2.5,
i.e. net of minGap. The diag config (ft1020_collisions.xml:21) carries time-to-teleport -1 and no collision.* option.
vanished_counters.txt: n_teleports=1 n_vanished_without_arrival=0 on both.
Judgement: the note's reading (collision -> default collision.action=teleport -> collider on its FINAL edge is removed
as arrived, "teleports beyond arrival edge") is the reading the evidence supports; the log's own words
"Teleporting vehicle ...; collision with" name the cause. No alternative reading found: stage=move, lane is a NORMAL lane
(not a junction check, consistent with check-junctions=false). MINOR: note sec 2's "gap -4.21 m"/"-4.52 m" is SUMO's gap
NET OF minGap; physical overlap is ~1.7 m (ft1020) -- say so or drop the number. Also: collision output path is in
THIS session's scratchpad (.../scratchpad/diag/ft1020/) -- the unfenced SUMO logs may still sit there; reviewer did NOT
open that directory. Coordinator should check it is not a leak of the fenced logs.
- [x] item 2 ticked

### F3.1 [item 3] 2026-09-24 Scoring claim A23(c)(iii) -- TRUE for the two observed events, FALSE in general
Commands: Read rltraffic-p73d-run/offline/sumo_att_reference.py:18-53, 80-91, 535-684, 700-712, 880-917;
metrics/sumo.py:176-195, 376-435, 459-477; grep metrics/sumo.py; `strings /usr/bin/sumo | grep teleport` messages.
Observed:
- Recorder arrivals = simulation.getArrivedIDList() (sumo_att_reference.py:906, 581-592); teleports =
  getStartingTeleportIDList() (:908, :616-618); n_vanished = departed - arrived - present_at_horizon (:707-711),
  present = vehicle.getIDList() (:904).
- att_env (FROZEN metrics/sumo.py): arrivals also from simulation.getArrivedIDList() per sim step (:248, on_sim_step);
  completed real_tt = t_arr - depart_time (:396-398); average = completed + vehicles in getIDList() (:464-474).
  => (a) YES for the observed events: the collider appears in getArrivedIDList (n_vanished 0 measured,
  vanished_counters.txt:1-2), and att_env reads the SAME TraCI list -> counted as a completed trip in both. The att_env
  half is established by code reading (same list), not by a separate measurement -- acceptable, say so.
- SUMO 1.27.1 binary strings include "Vehicle '%' teleports beyond arrival edge '%'" AND "Vehicle '%' ends teleporting
  on edge '%'" and "Teleporting vehicle '%'; beyond end of lane, target lane='%'". The second is the RE-INSERTION of a
  teleported vehicle further along its route.
  => (b) NO in general. "Beyond arrival edge" happens only when the collider's current route edge is its LAST edge
  (both observed events: 'D0right0', 'A0left0', exit edges). A collider on a non-final edge (or on an internal lane
  whose route edge is still the approach) is removed from the road, carried virtually along its route, and re-inserted
  on the first later edge that admits it ("ends teleporting on edge"), possibly skipping several edges; its trip does NOT
  end at the collision; its travel time is perturbed (virtual traversal replaces the real one); while in transit it is in
  neither getIDList nor getArrivedIDList, so att_env (metrics/sumo.py:470) drops it from the average, and if still in
  transit at T it is censored in E_sumo and COUNTED in n_vanished_without_arrival (:707-711) -- which A23(c)(i) then must
  not refuse ("other than through a recorded collision" covers it; good).
Judgement: (c)(iii) overclaims. Replacement text in final report. Also the title's "free of JAM teleports" is wrong:
(c)(i) refuses EVERY non-collision teleport (e.g. "beyond end of lane"), not only jam ones -> retitle.
Also (c)(i)'s "caused by a collision in the same step" needs an operational match: the teleporting id must be a party
(collider or victim) of a collision SUMO reports in the same step -- otherwise the implementer chooses.
- [x] item 3 ticked

### F4.1 [item 4] 2026-09-24 Population claim A23(b) -- counts verified; inference UNSOUND for the 200 probe episodes (BLOCKING)
Commands: grep -o '"n_teleports":N' | uniq -c on each artifact; `git show b63d56b:offline/transfer_calibration.py | grep
getStartingTeleportIDList`; Read rltraffic-p73d-run/offline/transfer_calibration.py:396-435, envs/sumo_env.py:174-231;
python listing KEY NAMES ONLY of output/p7_3d/calibration/probe_sumo_draw_*.json (no values printed).
Observed counts: p7_2b_calibration.json 100x n_teleports 0 (NO n_vanished field); p7_3a_zero_shot.json 4700x0 and
4700x n_vanished 0; p7_3b_anchor.json 700x0 / 700x0; p7_3d_reference_cells.json (run worktree) 6x0 / 6x0; grid4x4
probe 100 files, 100x n_teleports 0 (NO n_vanished field). Counts in A23(b) are CORRECT.
Observed mechanism: envs/sumo_env.py:229-231 advances one simulationStep() per simulated second; per-step TraCI lists
describe only the step just executed (sumo_att_reference.py:37-40; metrics/sumo.py:187-190). The PROBE roller reads
getStartingTeleportIDList() ONCE PER DECISION STEP (transfer_calibration.py:413-414, before env.step at :423) plus once
after the loop (:429) -- identical at P7.2b's artifact commit b63d56b (:433-434, :442, :448). delta_time = 10
(A17(b)). => the probe counter saw ~361 of 3,600 simulated seconds (~10 %); a collision teleport in any other second is
INVISIBLE to it, and neither probe artifact has a second counter (no n_vanished field).
Launch options (sumo_env.py:176-183): no --collision.* option -> SUMO defaults in every SumoEnv run: "collision handling
at SUMO's default, as in every earlier measurement" is TRUE.
Judgement: the inference "n_teleports = 0 => free of detected collisions" is sound ONLY for records whose counter reads
every simulated second (the recorder: P7.3a 4,700, P7.3b 700, C4 6, attempt 1 698; stoptime 0 + action teleport =>
every detected collision is an immediate, counted teleport, as the two refusals demonstrate). It is NOT sound for P7.2b's
100 and the grid4x4 probe's 100, and "(A17's n_teleports = 0 condition, met)" is true of the recorded FIELD only.
This is a pre-existing instrument defect (docstring :402 "accumulated per step" = per DECISION step -- a description taken
for behaviour). Since A17's probe condition is kept strict by A23(c)(iii), a collision in a probe episode would bear on
the pinned calibration (3e9df8ee...). Remedy options in the final report.
- [x] item 4 ticked

### F5.1 [item 5] 2026-09-24 Sensitivity analysis A23(d) -- outcome-blind, but three forking paths + a decision-rule change
Construction: read of note:118-122 against sec 8 (PREREGISTRATION.md:280-283), A20(d)-(e) (383), A15(a)-(b) (388).
(1) "both definitions" is ambiguous: A15's ATT pair (E_sumo / att_env) OR the two rho aggregations B.7.5 separated
(mean of per-draw ratios vs ratio of means). (2) "per draw and per seed" does not say whether the per-intersection
block (B.7.5) is included. (3) Disagreement criterion = "inequality" + "CI excludes 0" but OMITS sec 8's registered primary
test (Wilcoxon signed-rank, Holm within H1-H4). (4) "neither is chosen" gives a secondary analysis a veto over a
CONFIRMATORY verdict (A20(e)), contradicting A23(h)'s "A20 UNCHANGED". (5) The set is defined on attempt 2's record;
because (f) requires reproduction, it should be FIXED NOW as draws {1020, 1042} (14 cells), with any other collision
in attempt 2 being a stop condition, so the set cannot move after attempt 2. (6) Removal conditions on a post-treatment
event (a DT-cell collision is policy behaviour): label it a robustness check, not an unbiased estimate. Status
(secondary/exploratory) is not stated. "removed WHOLE (all its cells)" is itself unambiguous (7 cells per draw).
- [x] item 5 ticked

### F6.1 [item 6] 2026-09-24 "Re-run is not a second sample" A23(f) -- evidence adequate, rule as written unsatisfiable
Command: grep IDENTICAL|MATCH campaign_capture.txt (lines 65, 90-95, 133-137); wc/sha256 attempt1_manifest.
Observed: dt_reroll_check IDENTICAL = 13 rolls of ONE DT cell (seed101, draw0005 -- not a held-out draw), 1 at W=1 + 12 in
a pool; reference_reroll_check MATCH on 6 anchor cells (draws 1000-1002) vs C4; both refused cells reproduced in the diag.
Manifest: 700 lines, sha256 prefix 4f8c26190df03727 -- MATCHES the row's `4f8c2619...`.
Judgement: determinism evidence is adequate for the claim. BUT "reproduce under == on every field they share" is
UNSATISFIABLE as written: git_commit changes by construction (new commit), and note sec 1 itself records that re-rolls
differ on canary_seconds and seconds. The excluded fields must be enumerated IN THE ROW, or the exclusions are chosen
after seeing which fields differ. Also unstated: what happens on a mismatch (which attempt is used). And the two refused
cells must reproduce their collision (time, lane, collider, victim) -- not stated.
- [x] item 6 ticked

### F7.1 [item 7] 2026-09-24 Alternatives A23(e)
Construction: SUMO launch has no collision option (sumo_env.py:176-183, frozen) -> warn/none could only enter via the
rendered .sumocfg files -> re-render + re-pin digests. Under warn/none the 698 collision-free cells would be bit-identical
(collision action consumes no RNG and acts only on a detected collision -- REASONING, not measured); only the 2 collision
cells change, by one vehicle each. "gains no validity (the vehicles overlap instead)" is fair. The cost argument is the
real reason; the row should say that warn/none is rejected on cost and on keeping the configuration every earlier SUMO
number was measured under -- not imply a validity gain for teleport. "CAP(E) record" changing: NOT verified (CAP(E) is a
demand-multiset comparison per A15(g)/A20(f); whether it digests the .sumocfg was not checked).
- [x] item 7 ticked

### F8.1 [item 8] 2026-09-24 Seen ledger A23(g) -- the RTG inference is UNDERSTATED
Command: grep rewards.py -> reward_queue_length (rewards.py:48-50) = "Negative halting-vehicle count" => integer-valued.
Reasoning (exact float64 arithmetic): rtg_t = fl(T - S_t), S_t an exact integer running sum, T = target_i non-integer,
|T| in [2^e, 2^(e+1)). Rewards <= 0 so rtg rises monotonically from T. T + A_t (A_t = -S_t) is a multiple of ulp(T) and is
EXACT while |rtg| < 2^(e+1). So the difference-form check can fail only where rtg >= 2^(e+1) > |T|, i.e. where the
cumulative realised cost A_t >= |T| + 2^(e+1) > 2|T|. The printed failure locations therefore imply: on >= 373 of 499 DT
cells, some intersection's realised cost exceeded MORE THAN TWICE its prompt's magnitude by decision ~180-359 -- not merely
"exceeded its prompt's magnitude". (Caveat: assumes the logged reward_series is the integer halting count, as
rewards.py:48-50 and the two-route equality suggest; not verified on data, by constraint.)
Also: note sec 4 lists "per-cell wall clocks" as seen; SUMO wall time per cell rises with vehicles in the network, i.e. it
is a congestion proxy per arm. (g) should name it as a potential inference channel and state whether per-arm wall times
were compared. Demand counts/z-scores of draws 1020/1042 (note sec 2) are inputs, not outcomes.
- [x] item 8 ticked

### F9.1 [item 9] 2026-09-24 Overclaims / referee-reject wording in sec 5 (collected)
- Title "means free of JAM teleports": redefinition of a registered word + inaccurate ((c)(i) refuses every non-collision
  teleport kind; sumo binary also has "beyond end of lane" teleports). -> reframe as correction of a false belief.
- (b) "all were free of detected collisions": false as an inference for the 200 probe episodes (F4.1). BLOCKING.
- (c)(iii) "trip ends at the collision, at most one edge early": true only for a collider on its LAST edge (F3.1).
- (d) "neither is chosen" vs (h) "A20 unchanged": internal contradiction (F5.1).
- (f) "every field they share": unsatisfiable (git_commit, seconds, canary_seconds) (F6.1); "C4's cells MATCH over four
  rolls": NOT verified by reviewer (capture shows the campaign's 6 MATCH lines, lines 90-95, copied into the banner 134-139).
  "the DT cell IDENTICAL within and across runs" = ONE cell, draw 5 (capture:65) -- say "one DT cell".
- (g) RTG inference understated (F8.1); wall-clock proxy unnamed.
- Scope of (c) unstated: grid4x4 only, or every SUMO evaluation cell from now on (P7.3c's hz1x1 cells not yet run)?
- Minor, note sec 2: "gap -4.21 m" is net of minGap; physical overlap ~1.70 m (F2.2).
- Could NOT verify: CAP(E) digesting the .sumocfg (e); "201 rendered configurations"; "four rolls"; SUMO's
  MSVehicleTransfer path for a non-final-edge collider was established from binary message strings + TraCI semantics,
  NOT by simulation (constraint 3); integer-valuedness of the logged reward_series (constraint 1).
Verdict drafted: SOUND-WITH-CORRECTIONS; not taggable as written (F4.1 blocking; F3.1, F5.1, F6.1 must be rewritten).
- [x] item 9 ticked

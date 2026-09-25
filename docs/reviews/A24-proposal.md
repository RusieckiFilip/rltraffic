# REVIEW — the PROPOSED A24 (`docs/notes/P7.3c_A24_DRAFT_2026-09-25.md`, revision 1 as committed at `0021381`) — 2026-09-25 — VERDICT: NOT REGISTRABLE AS WRITTEN (3 blocking, 12 major, 9 minor — all ruled below; revision 2 adopts every blocking and major finding)

**Reviewer:** `contract-reviewer`, ONE adversarial round on a planning document before the author's approval and the tag (§4's
stopping rule: planning documents get one external round, then freeze). Mandate: find what is wrong with the draft against the
registration (A15–A23, §1, §2, §3.4, §8, §10), recompute every number by its own route from the committed files, verify every
claim about the code, and name the forking paths and referee attacks the draft leaves open; read-only; no simulation, training
or pytest; 15 minutes, checklist first, one append per finding (it ran ≈ 12 min over; nothing was stopped). The findings file is
filed verbatim below the rule. **Every number in revision 1 matched the reviewer's recomputation** (ρ̄₀ and its CI by two
routes with a worst per-cell difference of 0.0, G, G/2, the prompt shifts, the epoch arithmetic, the cell count, the five
digests and the recorded recipe); the defects are in what the text registers, not in its arithmetic.

## Coordinator's ruling (2026-09-25)

**BLOCKING — all ADOPTED.**
1. **F2 (an anchor-reproduction failure had two incompatible consequences, one of them chosen after the few-shot cells
   exist).** ADOPTED, and settled by DESIGN rather than by rewording: revision 2's stage 1 re-rolls P7.3d's WHOLE declared set
   (700 cells — the 200 anchors and the 500 zero-shot cells) at the campaign's commit, BEFORE any few-shot cell runs; all 700
   must reproduce the committed records, or the campaign stops with no few-shot cell run, and any replacement for ρ₀ is a
   further amendment written blind. This also answers F3 (G is always the constant), F21 (the collision cell is re-rolled)
   and turns a 5-cell sample into a population check.
2. **F5 (the draft amended A20(e) and A20(b) while listing A15–A23 as unchanged).** ADOPTED: (a) names A20(e) and A20(b) as
   amended, gives the new text of each, and discloses that the zero-shot result TRIGGERED the test (Cell 4 says so too); the
   UNCHANGED list excepts A20(b) and (e).
3. **F14 (the reproduction's exception list, copied from a same-stage re-roll, fails by construction on `stage`).** ADOPTED,
   verified by the coordinator on the committed records: exactly six fields are bookkeeping (`git_commit`, `stage`,
   `seconds`, `canary_seconds`, `checkpoint` — an absolute path — and `sha256_checked_against` — a provenance note); every
   other field a record carries is compared; `format_version` is not a cell field and leaves the list.

**MAJOR — all ADOPTED; three of them change the design, not only the text.**
4. **F1 (the ablation reduced to seed 101 without A18(d)'s condition).** ADOPTED by NOT reducing: stage 3 runs A18(d)'s
   secondary as registered, all five seeds (10 trainings, 1,000 cells). Compute is not forcing it; the machine runs unattended.
5. **F3** — answered by F2's design: G and G_att are constants in every computation, the robustness check included.
6. **F6 (§10's C3 sentence as A19 rewrote it is false for grid4x4; a whole-H3 verdict would fuse statuses).** ADOPTED: (f)
   scopes A19's sentence to hz1x1 and forbids a whole-H3 verdict on grid4x4.
7. **F8 (the outcome sentences are verdicts without §2's label).** ADOPTED: every sentence opens *"Exploratory (A24)"* and says
   *"criterion met / not met"*, not *"holds"*.
8. **F9 (§8's training-seed variance unreported).** ADOPTED: the five per-seed Δ₁₀₀ with CIs, their between-seed SD, and how
   many seeds meet each condition — reported beside the verdict, never deciding it.
9. **F10 (no estimator rule for refused or crashed cells).** ADOPTED: the estimators exist only on the complete declared set;
   infrastructure failures are re-run (deterministic) and counted; an A23 refusal stops the campaign and is resolved blind.
10. **F12 (non-deterministic fine-tunes with no run-once rule).** ADOPTED: every training runs once, its digest is committed
    before the evaluation token, and re-runs are allowed only for an infrastructure failure before a checkpoint exists,
    counted and reported; the timing runs are an unregistered configuration and never evaluated.
11. **F16 (the warm start described as an existing recipe; four details open).** ADOPTED: (b) says the fine-tune is new code,
    names what it reuses from P5.2 unchanged, and fixes the config source, the seeding order, the mask and the saved target.
12. **F17 (the subject's constructor refits normalisation, contradicting "frozen").** ADOPTED: the same path with the
    checkpoint's statistics passed through `stats=`, and the corpus required under `cityflow_grid4x4` with CityFlow ids.
13. **F18 (every fine-tuned arm is prompted above the best MaxPressure return in its own corpus).** ADOPTED and VERIFIED by the
    coordinator on the calibration artifact (16/16 intersections at every k; 84.3–128.6, 71.8–114.5, 48.6–104.2 return units
    at k = 5, 20, 100): stated in (e) as a fact of the design, with the fine-tuned checkpoints' in-support diagnostic fixed
    against BOTH the source's range and the fine-tune corpus's.
14. **F19 (the interior points and the expectation mix the prompt with the fine-tune; the control needs no training).**
    ADOPTED by EVALUATING the control: `zs_k5` and `zs_k20` (1,000 cells) in stage 3, and (e) defines the adaptation effect
    and its expectation on it.
15. **F20 (no control for how much of any closure is transfer).** ADOPTED by EVALUATING it: `scratch_k100`, the same trainer,
    data, budget, statistics and prompt with only the initialisation different (5 trainings, 500 cells), and a registered
    sentence for the case where it closes as much.

**MINOR — all ADOPTED:** F4 (A20(b) and A21(b) cited for the arms not evaluated), F7 (five steps at 95% without correction,
said so), F11 (moot — the list is now explicit), F13 (a robustness disagreement named in the verdict sentence), F15 (the
criterion on Δ₁₀₀ in float64; 0.9427 a restatement), F21 (answered by F2), F22 (both percentages given), F23 (the partition
written with its strict and non-strict bounds), F24 (G_att fixed: 0.11453996971792169, from the artifact's recorded ρ̄₀ on
`att_env`, 0.8854600302820783 — the per-draw route gives …782 in the last digit, the compensated-sum difference of `DEFERRED`
85, so the constant is taken from the recorded field).

**The reviewer's UNCHECKED items** (A-2's text, J1(c)'s wording, `DEFERRED` 93, A21/A22/A15(g) in full, Cell 4's on-disk
claim, P5.2's packet figure, the corpus's scenario key) are either the coordinator's own reading this session (A-2, J1(c),
`DEFERRED` 93, the packet's D-2 at `docs/returns/P5.2.md:257–262`), a search re-run at registration time (Cell 4), or a gate of
`BRIEF_40` (the corpus key, refused otherwise).

**Cost of the adopted design, measured bases:** 30 trainings ≈ 5.6 h of GPU sequentially (≈ 8.1 min per 4,000 steps at P5.2's
122 ms/step); 4,700 cells ≈ 12.8 h at P7.3d's 119 s per DT cell and 54 s per anchor cell over 11.9× parallelism — stage 1
≈ 1.6 h, stage 2 ≈ 4.2 h, stage 3 ≈ 6.9 h. All unattended.

---

# A24 DRAFT REVIEW -- findings file (reviewer: independent, read-only)
Subject: docs/notes/P7.3c_A24_DRAFT_2026-09-25.md @ main 0021381
Started: 2026-09-25

## CHECKLIST
- [ ] R1. CONTRADICTIONS with registered rows (A15-A23, s2, s3.4, s8, s10), esp. A17(b)/(e)/(f), A18(c)/(d)/(e), A19, A20, A21, A23
- [ ] R2. FORKING PATHS (rho0 series, estimator/pairing, checkpoint evaluated, draws per k, rounding of G/2, boundary Delta==G/2 / CI bound==0, anchors fail to reproduce, collisions, att_env disagreement, ablation reading)
- [ ] R3. NUMBERS recomputed by own route: rho0 mean+CI (e_sumo), G, G/2, rho100 threshold; prompt shifts k5/k20 vs k100 (max |diff|, intersection, % of target, % of rtg_scale); epoch arithmetic (joint instants/episode; 1,800/7,200/36,000; 142/36/7 epochs @ batch 64, B=4,000); cell count 1,500+200+200=1,900; checkpoint recipe from .pt provenance (lr, wd, warm-up, clip, n_head, device)
- [ ] R4. FEASIBILITY claims vs code/data: tier_sweep recipe (warm-up min(1000,B/2), constant after); joint_windows constructor; artifact per-cell fields sufficient for anchor reproduction and rho0 per (draw, seed); A16(d) identity on grid4x4; probe returns per intersection for draws 201-300
- [ ] R5. DECIDABILITY of clause-3 criterion (outcomes (i)-(iv) gap/overlap; strictness of "entirely above zero"; robustness check role on disagreement)
- [ ] R6. REFEREE ATTACKS not answered (k=5/20 prompt confound; no retrain anchor on grid4x4; recorded_not_evaluated targets; CUDA non-determinism; one engine seed; MaxPressure demonstrator; new code vs older-code rho0)

## FINDINGS (append-only blocks below)

### [R1/R2 partial] 2026-09-25 -- read draft (all 142 lines) + A18 (PREREGISTRATION.md:386, folded)
F1 (MAJOR, R1) -- ablation reduction invoked without its registered precondition.
  Draft L72-75: "B in {1,000, 16,000} at k = 100, REDUCED TO SEED 101 by the route A18(d) itself registers, declared here before running (200 cells). It is kept, at one seed, because it bears directly on how (d)'s verdict is read".
  A18(d) (PREREGISTRATION.md:386): "B in {1,000, 16,000} at k = 100 only, both subjects, all five seeds ... if compute forces a reduction it reduces to seed 101 and says so before running."
  The route is conditional on COMPUTE forcing it; the draft gives a reason to KEEP the ablation, none for REDUCING it (full = 10 fine-tunes ~ 5x(1,000+16,000) steps x 122 ms ~ 2.9 h GPU + 1,000 cells). Propose: state the compute reason with a number, or run all five seeds as registered.
F2 (BLOCKING, R2) -- anchor-reproduction failure has TWO incompatible consequences in the same cell (c).
  Draft L67: "a mismatch stops the campaign's report, and no few-shot number is read."
  Draft L70-72: "licensed by that reproduction AND by re-rolling the zero-shot arm's five cells of draw 1000 ... If either fails, the full zero-shot arm (500 cells) is re-rolled at the campaign's commit and THAT series is rho0".
  "that reproduction" = the anchor reproduction; so an anchor mismatch both STOPS the report and TRIGGERS a re-roll-and-continue. The anchors run in the SAME stage/token as the 1,500 few-shot cells (L62 "one declared stage, one token"), so the choice between the two is made after the few-shot cells exist on disk. Propose: one consequence only; if re-roll, say whether anchors are also re-rolled/used from the new commit and whether a report stop still applies.
F3 (MAJOR, R2) -- G is a fixed constant from the committed artifact, but rho0 may be REPLACED by a re-rolled series.
  Draft L83: "G = 1 - rho0 = 0.11453573207299894, a constant fixed here from the committed artifact"; L71-72: "the full zero-shot arm (500 cells) is re-rolled ... and THAT series is rho0".
  If rho0 is re-rolled, is G recomputed (1 - new rho0) or kept? The "equivalently rho100 >= 0.9427" (L86) and the closure fraction Delta/G (L92) depend on it. Propose: state explicitly which G applies under the fallback.

### [R1 partial] A17 read (PREREGISTRATION.md:387)
  Verified: A17(b) registers the probe as "One episode per draw on a fresh env with reset(seed=1000)", per-intersection return "computed by two routes that must agree under ==", "every probe episode must show n_teleports = 0"; "the k-shot point ... uses the calibration from the same k episodes that form its fine-tuning data" -> the draft's k=5/20 Rule B prompts for ft arms are ALREADY registered by A17(b) (consistent). A17(f): "per draw, the logged episode's return must equal P7.2b's probe return bit-for-bit ... a mismatch stops P7.3" -> draft L51-55 per-intersection bit-for-bit is consistent (stricter granularity, same rule).
F4 (MAJOR, R1) -- exploratory prompt arms on the FINE-TUNED models are dropped without the declaration A17 requires.
  A17(a): "S = max is computed and REPORTED as an ablation on every cell and never selected; Rule A ... and the NAIVE prompt ... are reported beside it"; A17(d): "Rule B(max) and Rule A are evaluated on the same draws and seeds so no arm can be dropped after the fact; P7.3's brief may reduce the exploratory arm set only by declaring so before it runs."
  Draft (g) L114-118 lists what is NOT evaluated: k=200 anchor, naive+random, second subject, zero-shot at k=5/20. It never mentions Rule B(S=max) or Rule A prompts on the ft_k* models. (Pending A20/A21 check whether those were already dropped for grid4x4 generally.) Propose: one sentence in (g) naming S=max / Rule A / naive on ft arms as not evaluated, with the reason.

### [R1 partial] A20 read (PREREGISTRATION.md:384)
F5 (BLOCKING, R1) -- the draft amends A20(e) while declaring A15-A23 UNCHANGED.
  A20(e) (PREREGISTRATION.md:384): "(e) STATUS. H3's clause 1 ... is CONFIRMATORY on this scenario ...; clause 2 is reported as the inequality it is; clause 3 is void (A19)." -- i.e. clause 3 on GRID4X4 was declared void by a REGISTERED ROW, not merely by the artifact's field.
  Draft L22-24 frames the change as superseding only "That artifact's h3.clause_3 field ... this row supersedes that field's wording for grid4x4"; Draft L120: "UNCHANGED: section 3.4's rho, A15-A23, ...".
  Both cannot be true: un-voiding clause 3 on grid4x4 changes A20(e). And the un-voiding is decided AFTER rho0 = 0.8855 was seen (A20(e) was written before any grid4x4 number), which a referee will read as a data-dependent re-opening. Propose: name A20(e) as the clause amended, remove it from the UNCHANGED list, and state in Cell 3/4 that the re-opening decision itself followed the zero-shot result (it is disclosed in Cell 4 only as "seen", not as the trigger).
F4-update (downgrade F4 to MINOR) -- A20(b) already says "b_max_k100 and a_q1.0 are NOT evaluated on grid4x4" (scenario-wide), so S=max / Rule A on ft arms is arguably covered; the draft should still cite A20(b) in (g) so the ft arms' prompt set is closed by a citation, not by silence.
  Verified: A20(a) digests listed = 329fb6b8, f5413585, 48076dab, 4b61bc06, 09bd310d (to be checked vs disk in R3). A20(d): "rho per draw against the anchors of the same draw (section 3.4)" -> per-draw rho confirmed.

### [R1 partial] A19 read (PREREGISTRATION.md:385)
  Verified: A19(a) quotes H3 (s1): "zero-shot transfer is positive but incomplete -- better than fixed-time, worse than within-backend MaxPressure -- and closes substantially by k = 100"; A19(a)'s reason is hz1x1-specific; A19 stated limits: "a degradation observed there is not a claim about grid4x4 (P7.3d)". -> draft L24-26 (A19(b) framing hz1x1-only) is consistent with A19's own limits.
F6 (MAJOR, R1) -- the draft is silent on section 10's C3 outcome row, which A19 REWROTE globally.
  A19 (PREREGISTRATION.md:385): "section 10's outcome rows: 'H3 holds -- the few-shot budget needed to close the dynamics gap quantified' presupposes the gap and is inapplicable; the registered outcome sentence for C3 is now: H3's clause 1 holds, clause 2 is refuted, clause 3 is void; the transfer curve is reported as a measurement of interference ...". That sentence is scenario-agnostic in its wording and is FALSE for grid4x4 (clause 2's inequality holds there; clause 3 would be tested).
  Draft (d) L87-91 writes four new outcome sentences and (h) L120 lists "A19's hz1x1 framing and expectation" as unchanged, but never says which section-10 C3 sentence governs grid4x4, nor that (i)-(iv) supersede A19's rewritten row for grid4x4. Propose: add "section 10's C3 row as rewritten by A19 is scoped to hz1x1; on grid4x4 the outcome sentence is (d)(i)-(iv)" and remove section 10 from any implicit UNCHANGED.
F7 (MINOR, R2) -- A19(b) registered its refutation in "the paired per-draw estimator of section 8"; draft (e) L104-106 uses adjacent-step CIs entirely below zero across 3 steps (0->5->20->100) with no multiplicity statement; each at 95% => ~14% chance of a spurious "refutation" if flat. Propose: say the 3 steps are read jointly/without correction and that this is stated in the paper.

### [R1 partial] s1 + s2 read (PREREGISTRATION.md:78-114)
  Verified: H3 text (PREREGISTRATION.md:~88) "closes substantially by k = 100" -- draft's reading "gap to MaxPressure" matches A19(a)'s own reading. s2: H3 confirmatory = zero-shot vs fixed-time only; "the shape of the transfer curve between the registered points" and "every scenario outside the headline three" are exploratory; exploratory = "reported with effect sizes and CIs, explicitly labelled, no inferential claims".
F8 (MAJOR, R1 vs s2) -- the four paper sentences (d)(i)-(iv) state hypothesis verdicts ("clause 3 holds" / "does not hold" / "fails") decided by a 95% CI excluding zero -- a significance test in all but name -- and none of the four sentences carries the exploratory label s2 requires ("explicitly labelled, no inferential claims"). Draft (f) L110-111 says "no p-value presented as inferential", which does not cover a CI-based accept/reject verdict. Propose: put "(exploratory)" inside each of the four registered sentences, or rephrase "holds" as "meets the registered criterion".

### [R1 partial] s3.4 + s8 read (PREREGISTRATION.md:160-182, 269-294)
  Verified: s3.4 rho = (ATT_ft - ATT_policy)/(ATT_ft - ATT_mp), per draw per A20(d).
F9 (MAJOR, R1 vs s8 / R6) -- training-seed variance is neither reported nor used, although the uncertainty clause 3 is about is the fine-tune's.
  s8 (PREREGISTRATION.md:~271-273): "the unit is the held-out flow draw; for training, the training seed. Both sources of variance are reported."
  Draft (d) L80-83: CI "1.96*s/sqrt(100)" over draws of five-seed means only; no per-seed Delta_100 or seed SD is registered for report. With 5 fine-tunes that are CUDA non-deterministic (draft L43), a clause-3 "holds" can rest on 1-2 seeds. Propose: register reporting of the five per-seed Delta_100 (each paired over 100 draws) and state whether the verdict requires anything of them (e.g. none; reported only).
F10 (MAJOR, R2 vs s8 / A23) -- refused / excluded cells have no registered estimator rule.
  s8: "Episodes are excluded only for infrastructure failure (crash, out-of-memory), and every exclusion is counted and reported." Draft L75-76: "a teleport not caused by a same-step collision refuses the cell". Draft (d) L80-82 defines rho_k(d) as "each draw's five-seed mean" and Delta over "the 100 per-draw differences". If an ft cell (or an anchor cell) is refused or crashes, is the draw's mean over 4 seeds, the draw dropped from BOTH arms, or the cell re-run? (For a crash a re-run is deterministic; for a refusal it is not re-runnable.) Pending A23 check. Propose: one sentence fixing it.

### [R1 partial] s10 read (PREREGISTRATION.md:317-344)
F6-evidence (strengthens F6, MAJOR): s10 rows "H3 holds | The transfer curve, with the few-shot budget needed to close the dynamics gap quantified." and "H3 fails / gap indistinguishable from the interface control | C3 becomes a characterised limitation with the P7.4 control ...". Under outcome (i) on grid4x4 all three clauses would "hold" (clause 1 confirmatory, 2 as inequality, 3 exploratory). The draft never says whether the paper may then write "H3 holds on grid4x4" (s10's row) -- which would fuse a confirmatory clause with an exploratory one, i.e. a promotion s2 forbids -- nor which s10 row applies under (ii)-(iv). Propose: register that no whole-H3 verdict is written for grid4x4; only per-clause statements with their statuses.

### [R1/R2 partial] A23 read (PREREGISTRATION.md:380)
  Verified: A23(b): "all 200 [probe episodes, hz1x1 + grid4x4, draws 201-300] reproduce their registered probe returns under ==, and SUMO's statistics count 0 teleports and 0 collisions in every one of them" -> draft's corpus gate L51-55 (zero teleports AND zero collisions, every second) is already known to pass on the probe; consistent.
F11 (MINOR, R1) -- mis-citation of the reproduction field list. Draft L67: "except git_commit, format_version, seconds and canary_seconds (A23(e)'s list)". The list is in A23(f) ("(f) CONSEQUENCE, AND WHY THE RE-RUN IS NOT A SECOND SAMPLE ... except git_commit, format_version, seconds and canary_seconds (the full list fixed in BRIEF_39 B.8 ...)"); A23(e) is "NOT CHOSEN, AND WHY". Propose: "A23(f)'s list (BRIEF_39 B.8)".
F12 (MAJOR, R2) -- the fine-tune is non-deterministic, so A23(f)'s argument "WHY THE RE-RUN IS NOT A SECOND SAMPLE" does not transfer, and the draft registers no once-only rule.
  A23(f): the P7.3d re-roll was legitimate because "SUMO and the agent are deterministic under the registered seeds". Draft L42-44: the fine-tune runs "CUDA, non-deterministic ... the fine-tuned checkpoints are pinned by digest" -- but not WHEN (before the evaluation token? in a committed file?) nor what happens on a crash, a refused cell, or an A23-like stop that forces a new commit: are the pinned checkpoints reused, or re-trained (= a fresh sample)? Every re-train is a draw from the fine-tune distribution and can be repeated until favourable. Propose: "each (seed, k, B) fine-tune runs once; digests are committed before the evaluation token; every later attempt reuses them; a re-train only on infrastructure failure before a checkpoint is written, counted and reported".
F13 (MINOR, R5) -- robustness-check disagreement reporting is weaker than A23(d)'s. A23(d): "If it differs from the primary on the sign of mean rho, on whether the 95 % CI excludes 0, or on s8's Wilcoxon-Holm decision, the paper says so in the sentence that states clause 1's verdict." Draft L93-95: "the verdict is recomputed ...; the primary decides" -- no statement that a disagreement (a different outcome among (i)-(iv)) is written into the verdict sentence, and no statement whether G is recomputed on the reduced draw set. Propose: copy A23(d)'s disagreement sentence; fix G as the constant in the robustness check too.

### [R4 item] A16(d) identity on grid4x4 (PREREGISTRATION.md:388, folded lines 16-20)
  VERIFIED: "where the two counts are EQUAL (grid4x4, 16 against 16, the two programs in 1:1 correspondence) the map is the IDENTITY"; green-action count "8 on grid4x4 (of 16)". Draft L48-49 consistent (it maps file phases; the logged action is the 0..7 green index -- wording "16 phases against 16" is A16's own). No finding.

### [R3 item: digests] sha256sum of artifacts and subject checkpoints
  VERIFIED: docs/data/p7_3d_grid4x4.json = c63c371f...; docs/data/p7_3d_calibration.json = 3e9df8ee...; seed101 329fb6b8, seed202 f5413585, seed303 48076dab, seed404 4b61bc06, seed505 09bd310d -- all match A20(a) (PREREGISTRATION.md:384) and the draft. No finding.
  Artifact cell keys (python json load): include 'stage', 'config_sha256', 'calibration_sha256', 'checkpoint', 'checkpoint_sha256', 'git_dirty', 'sha256_checked_against', 'kind' ... and NO 'format_version' key (format_version is top-level 'p7.3d-grid4x4/1.1'). Artifact top-level git_commit = 10d8fe81...

### [R4 item: anchor-reproduction field rule] python dump of one fixedtime / maxpressure / b_mean_k100 cell (docs/data/p7_3d_grid4x4.json:cells)
F14 (MAJOR, R4/R2) -- the field-exception list is copied from a SAME-STAGE re-roll (A23(f)) into a CROSS-CAMPAIGN comparison, where it fails for bookkeeping reasons.
  Every committed cell carries stage = 'grid4x4_confirmatory' (also 'config_sha256' = c27d31e8..., 'calibration_sha256', and for DT cells an ABSOLUTE 'checkpoint' path '/home/filip/rltraffic/output/p5_2/checkpoints/...'). Draft L65-67: anchors must reproduce "on every field that record carries except git_commit, format_version, seconds and canary_seconds"; draft L70-71 applies the same rule to the draw-1000 zero-shot re-roll.
  A P7.3c anchor cell labelled with P7.3c's own stage (or run from a worktree path) mismatches on 'stage' (or 'checkpoint') and, per L67, "stops the campaign's report" -- or else someone decides AFTER the campaign that the mismatch "doesn't count" (a fork). ('format_version' is not a cell key at all.) Propose: name the fields compared (e.g. every measurement field: e_sumo, att_env, w_sumo, p_sumo, local_return, episode_reward, n_* counters, engine_seed_drawn, routes_sha256, rho_*), or add 'stage' (+ 'checkpoint' path) to the exception list explicitly, before the token.

### [R3 item: rho0, G] own recomputation from docs/data/p7_3d_grid4x4.json:cells (python; route A = recorded rho_e_sumo field, route B = (E_ft - E_dt)/(E_ft - E_mp) from e_sumo of the same draw's anchors)
  VERIFIED: 100 draws (1000-1099), 5 seeds each; max |route A - route B| per cell = 0.0; rho0 = 0.8854642679270011 (both routes); CI (1.96*sd/sqrt(100), ddof=1) = [0.873588, 0.897340] (sd 0.060592, hw 0.011876) -> draft's [+0.8736, +0.8973] correct. G = 0.11453573207299894 (matches draft L83); G/2 = 0.057267866036499471 (repr 0.05726786603649947, matches L85); implied rho100 threshold = 0.94273213396350053.
F15 (MINOR, R2/R5) -- draft L86 "equivalently rho100 >= 0.9427 at the point estimate" is a ROUNDING, not an equivalence: rho100 = 0.94272 satisfies ">= 0.9427" and fails "Delta >= G/2". Also the paired Delta = mean_d[rho_k(d) - rho0(d)] and rho_k - rho0 agree only when both are over the same 100 x 5 cells (see F10) and can differ in the last ulp. Propose: "rho100 >= 0.9427321339635005 (float64), the comparison made on Delta_100 in float64 with no rounding".

### [R3 item: prompt shifts] own recomputation from docs/data/p7_3d_calibration.json:per_intersection[i].budgets['k5'|'k20'|'k100'].target and per_intersection[i].rtg_scale
  VERIFIED: k5: max |t5 - t100| = 15.6781 at D0 (t5 -124.157, t100 -108.479) = 14.45% of |t100|, 12.63% of |t5|, 6.15% of rtg_scale 255 -> draft L101 "15.68 ... (D0; 14.5% of the target, 6.2% of ... rtg_scale)" correct (the "target" is the k100 one; say so). k20: max 3.1084 at D3 = 2.73% of |t100|, 1.20% of rtg_scale 258 -> draft "3.11 at k = 20 (D3; 2.7%)" correct.
  Note: k5 shift is a 14% change of a return prompt on one intersection, and the sign of d5 varies by intersection (-15.7 .. +9.1): the k=5 point's prompt confound is not small relative to anything the fine-tune could do; the draft's handling (report as joint effect, zero-shot at k5/k20 prompts NOT evaluated) is a real gap -- see R6.
  VERIFIED (R4): probe_returns.sumo is a dict of 100 draws '201'..'300', each a dict of 16 intersections (integer-valued floats, e.g. -219.0) -> per-intersection bit-for-bit gate (draft L51-55) is feasible; integer returns < 2^24 are exact in float32 too. calibration: declared_gradient_steps 40000; architecture {context_length 20, n_head 4, spatial_mixing False, state_dim 40}; disjointness training_draw_ids [1,200], probe [201,300], held-out [1000,1099], checked_by offline.rtg_calibration.assert_probe_draws_disjoint.

### [R3 item: checkpoint recipe] torch.load(weights_only=True) of all five output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed*.pt, flattened 'provenance'
  VERIFIED: batch_size 64, deterministic False, device 'cuda', grad_clip 0.25, gradient_steps 40000, learning_rate 0.0001, weight_decay 0.0001, warmup_steps 1000, n_head 4, spatial_mixing False; config context_length 20, n_layer 3, d_model 128, dropout 0.1, max_ep_len 360, n_nodes 16, n_actions 8, state_dim 40. Identical across the five seeds except 'seed'; rtg_scale / target_rtg / state_mean identical across seeds. Training draws stats.draw_ids start 1..; dataset_dirs datasets_v11/cf_grid4x4__mappo1000__seed*. -> draft L33 (lr 1e-4, wd 1e-4, clip 0.25), L36-37 (n_head 4, context 20), batch 64, "CUDA, non-deterministic" all correct.
  NOTE: the provenance records NO learning-rate schedule after warm-up (no 'schedule'/'decay' key) -- "a constant rate after it" (draft L34) must come from the code (checked next, R4).

### [R4 item: tier_sweep recipe] offline/tier_sweep.py:660-678 (warmup_for, lr_multiplier), :765-947 (train_tier_dt)
  VERIFIED: warmup_for(total) = min(WARMUP_STEPS, max(1, total // 2)) (:660-669) -> B=4,000: 1,000; B=1,000: 500; B=16,000: 1,000. lr_multiplier = min(1.0, (step+1)/warmup) (:672-678) -> constant after warm-up. Draft L33-35 correct on both.
  VERIFIED: batches are drawn WITH replacement, generator.integers(0, count, size=batch_size) (:873-880) -> "142, 36 and 7 epochs" are nominal sample/window ratios, not passes (same convention as A18(d); fine).
F16 (MAJOR, R4) -- "P5.2's own (offline/tier_sweep.py) -- the only recipe that can warm-start this architecture" (draft L31-32) is false as a statement about the code: train_tier_dt ALWAYS builds a fresh model -- "model = SpatialDecisionTransformer(config).to(device)" (:846-853), config re-derived from the data ("context_length=int(stacked['state'].shape[1])", "max_ep_len=int(stacked['timestep'].max()) + 1", :846-853) -- and has no parameter to load weights. The warm start is therefore NEW code that the row describes as an existing recipe. Registered details the new code must fix and the draft leaves open: (1) model config taken from the CHECKPOINT (max_ep_len 360), not re-derived from SUMO data; (2) whether Utils.seed_everything(seed) (:844) runs before the load (it affects dropout masks only once weights are loaded); (3) the spatial mask taken from the checkpoint's 'spatial_mask' (identity) vs recomputed; (4) the prompt dict passed for the saved 'target_rtg' (the saved checkpoint writes prompts[ix].target_rtg -- CityFlow's or Rule B's? it changes what a later reader of the fine-tuned .pt believes its prompt is). Propose: reword to "a new warm-start trainer that imports tier_sweep's constants, warmup_for, lr_multiplier and sampling unchanged", and list (1)-(4).

### [R3/R4 item: windows and epochs] offline/joint_windows.py:105-183 (build_joint_index) + checkpoint stats.row_count 72,000 per intersection
  VERIFIED: one joint window per (episode_index, t), all 16 nodes, left padding inherited from offline/dataset.py (docstring :14-17); the subject's 72,000 rows/intersection = 200 episodes x 360 -> one window per decision instant. So k=5/20/100 -> 1,800 / 7,200 / 36,000 joint windows; 4,000 x 64 / {1,800, 7,200, 36,000} = 142.2 / 35.6 / 7.1 -> "142, 36 and 7 epochs" correct (nominal, with replacement). Ablation: B=1,000 -> 1.8, B=16,000 -> 28.4 nominal epochs at k=100 (not stated in draft; fine).
  Cell count: 3 arms x 5 seeds x 100 = 1,500; fixedtime + maxpressure x 100 = 200; 2 budgets x 1 seed x 100 = 200 -> 1,900 correct. (Plus 5 pre-token zero-shot re-roll cells and a possible 500-cell fallback, not counted -- fine but should be named as outside the 1,900.)

### [R4 item: normalisation in the constructor] offline/tier_sweep.py:1429-1442 (tier_dataset), offline/dataset.py:456-469,664-669,82-85
  VERIFIED: P5.2's constructor builds TrajectoryWindowDataset(..., split="train", normalize=True) WITHOUT stats= (tier_sweep.py:1436-1441), and dataset.py:664-667 then FITS stats on the corpus given ("self._stats = self._fit_stats()"). Reusing "the constructor that built the subject's" (draft L38) verbatim on the k SUMO episodes would REFIT normalisation on 5/20/100 SUMO episodes -- the opposite of draft L35 "state-normalisation statistics FROZEN, never refit". The dataset does accept stats= (dataset.py:463, :664-665).
  (continued in next block: scenario_id keying and split of draws 201-300)

### [R4 item: split + stats keying] offline/dataset.py:179 (DRAW_SPLITS "train": (1, 999)), :265-287 (normalize_state)
  VERIFIED: draws 201-300 lie in the "train" split, so split="train" accepts the probe band (feasible). normalize_state looks stats up by (scenario_id, ix_id) and raises KeyError otherwise (:271-279); the checkpoint's stats are keyed 'cityflow_grid4x4'. The committed P7.3d cells record scenario = 'cityflow_grid4x4' even for SUMO cells, so the logged SUMO corpus may or may not carry that key -- if it carries a SUMO key, the frozen stats cannot be applied without a mapping the draft does not register (fails loudly, not silently).
F17 (MAJOR, R4) -- draft L35-38 asserts both "state-normalisation statistics FROZEN, never refit" and "windows built by the constructor that built the subject's"; the subject's constructor refits (tier_sweep.py:1436-1441 + dataset.py:664-667). Propose: "the same TrajectoryWindowDataset/build_joint_index/stack_joint path, with the checkpoint's stats passed through stats= (keyed by the checkpoint's scenario_id 'cityflow_grid4x4'), and a test that the fine-tune corpus's normalised states equal (raw - ckpt mean)/ckpt std bit-for-bit".
  VERIFIED (R4): artifact cells carry draw_id, seed, arm, e_sumo, att_env, rho_e_sumo, rho_att_env, local_return (16), n_teleports, n_vanished_without_arrival -> sufficient for rho0 per (draw, seed) and the anchor comparison on measurement fields.

### [R4/R2 item: cross-campaign comparator] docs/notes/P7.3c_SURVEY_2026-09-25.md:369-377 (the draft's own survey)
F14-UPGRADE (F14 -> BLOCKING): the survey the draft cites says verbatim: "compare_reroll_payloads is the generic two-route comparator, but used across campaigns it would flag stage, git_commit (and possibly canary_seconds) as differing, since only 'seconds' is excluded" and "stage is in the chunk identity but not compared by reference_cell_differences". The draft added git_commit/canary_seconds/format_version to the exclusions but NOT 'stage', which every committed cell carries ('grid4x4_confirmatory'). As registered, the anchor reproduction fails by construction under any new stage label -> L67 "stops the campaign's report" (or F2's re-roll branch). Irreversible once tagged. Also the existing check covers "only 6 of 200 anchor cells ... on 17 fields" (survey :369-371): the 200-cell comparison is new code, so the registered field list is the only definition it will have.

### [R6 item: prompt vs fine-tune corpus support] own python over docs/data/p7_3d_calibration.json (probe_returns.sumo draws 201..200+k vs per_intersection[i].budgets[k].target)
F18 (MAJOR, R6/R2) -- on ALL 16 intersections and at EVERY k (5, 20, 100), the Rule B prompt is BETTER than the best SUMO MaxPressure episode return in the very k episodes the model is fine-tuned on: at k=100 by 48.6 (B2: target -80.38 vs max -129) to 104.2 (D1: -123.83 vs -228) return units; e.g. A0 target -124.93 vs probe max -183, min -310, mean -239.33. So every fine-tuned arm is prompted OUTSIDE its fine-tuning corpus's t=0 return support, a fact knowable now (the probe returns are seen, Cell 4). The draft registers no in-support diagnostic for a FINE-TUNED checkpoint (A17(d)'s diagnostic is "relative to its checkpoint's training RTG support" -- CityFlow's, the SUMO corpus's, or the union? each gives a different "inside/outside"), and does not state this fact. A referee will attribute any (ii)/(iii)/(iv) outcome to prompt extrapolation. Propose: state the fact with these numbers in (e) and fix the diagnostic's definition for ft checkpoints before the token.

### [R6 item: other referee attacks]
F19 (MAJOR, R6) -- the interior points and the registered expectation are prompt-confounded, and the cheap control is declined. Draft (e) L100-105: k=5 shifts a target by up to 15.68 (14.5%) and "The zero-shot model under the k = 5 and k = 20 prompts is NOT evaluated (1,000 cells)", yet the registered expectation "rho non-decreasing in k across 0 -> 5 -> 20 -> 100" (L104) is refuted by the 0->5 or 5->20 step, which mixes a prompt change with the fine-tune. The control needs no training (1,000 deterministic cells vs 1,900 already declared). Propose: evaluate it, or restrict the expectation's refutation to what the design can attribute (state that a 0->5 / 5->20 refutation is a joint prompt+weights statement).
F20 (MAJOR, R6) -- no attribution control for the fine-tune. Draft (g)(i) L114-117 answers why clause 3 needs no k=200 anchor, but not the attack "closure by k=100 is what behaviour-cloning MaxPressure from 100 episodes gives any model" (A19(c): hz1x1's from-scratch anchor on MaxPressure data sits at +0.98, "pinned near rho = 1.0 by construction"). Clause (i)'s sentence "fine-tuning ... closes X% of the zero-shot gap" will be read as a transfer claim. A from-scratch control at the SAME B=4,000 on the same k=100 episodes costs ~5 x 8 min + 500 cells. Propose: add it as exploratory, or register a sentence that the closure is not attributed to transfer.
F21 (MINOR, R6) -- the new-code licence re-rolls only draw 1000's five zero-shot cells; the one zero-shot cell with a collision (b_mean_k100 seed 303 draw 1042, A23(a)) exercises A23's collision code path and is not re-rolled. Propose: add it to the pre-token re-roll (6 cells).
F22 (MINOR, R3) -- "14.5% of the target" (L101) is of the k=100 target (14.45%); of the k=5 target it is 12.63%. Say which.

### [R5 item: decidability] draft (d) L84-95
F23 (MINOR, R5) -- strictness unstated: "lies entirely above zero" / "CI containing zero" / "entirely below zero" do not say whether a bound EQUAL to 0 is (ii)/(iii) or (iii)/(iv). With lo > 0, lo <= 0 <= hi, hi < 0 the four outcomes partition; say so. Otherwise (i)-(iv) are mutually exclusive and exhaustive (Delta >= G/2 > 0 cannot co-occur with hi < 0) -- no gap/overlap found beyond the boundary.
F24 (MINOR, R5) -- att_env's gap G_att is not fixed numerically ("the same computation on att_env", L92-93): fix G_att = 1 - rho0_att from the committed artifact now, as for E_sumo.
  Robustness role: "the primary decides" (L95) is unambiguous for the verdict (but see F13 for reporting and G on the reduced set).

### CHECKLIST STATUS (final, appended; earlier blocks untouched)
- [x] R1 (A17, A18, A19, A20, A23, s1, s2, s3.4, s8, s10 read; A15(g)/A21/A22 NOT read in full -- A21 only via A20/draft citations)
- [x] R2 (F2, F3, F10, F12, F14, F15, F18)
- [x] R3 (rho0/CI/G/G2/threshold, prompt shifts, epochs, cell count, digests, recipe -- all recomputed; 122 ms/step arithmetic 81.5 min*60/40,000 = 0.122 s, 4,000 steps = 8.2 min, packet itself NOT read)
- [x] R4 (tier_sweep recipe, joint_windows, normalisation, split, artifact fields, A16(d), probe returns per intersection)
- [x] R5 (F23, F24, F13)
- [x] R6 (F18-F21)
- UNCHECKED: docs/reviews/P7.3d.md A-2 text; J1(c) wording; DEFERRED 93; A21/A22/A15(g) full text; Cell 4's on-disk "no p7_3c" claim; P5.2 packet's 81-82 min figure; whether the P7.3c SUMO logger records scenario_id 'cityflow_grid4x4' (survey marks it UNVERIFIED).

# REVIEW — the PROPOSED A25 (the paper's scope) and A26 (H4's context-length sweep), revision 1 as committed at `3d319d0` — 2026-09-28 — VERDICT: NOT REGISTRABLE AS WRITTEN (5 major, no blocking label; every number and on-disk claim verified; revision 2 adopts every major and minor)

**Reviewer:** `contract-reviewer`, ONE adversarial round on two planning documents before the author's approval and the tag (§4's stopping
rule). Mandate: contradictions with the registered rows, forking paths, the code facts A26 relies on, the numbers, A25's on-disk
claims, and the referee attacks neither draft answers; read-only; 20 minutes. The findings file is filed verbatim below the rule.
**Verified by the reviewer, no finding:** A25's Cell 4 (only `flow_randomizer.py`; no P6/P7.4 output in `main` or any worktree; no
branch; no `docs/data` artifact; no domain-randomised MAPPO checkpoint); every A26 number (−5762, 9991, draws 1–200 and 1000–1099,
δ = 0.6263, 202–204 s per training with seed 505's 14,018 s anomalous, 2.897 s per episode, 25 trainings, 2,500 cells); the survey's
cites (`train_dt`'s `context_length`, the K-invariant window index, the K = 1 branch of the rollout).

## Coordinator's ruling (2026-09-28) — every finding ADOPTED; three of them change the DESIGN
1. **F1 (A26 makes §2's one H4 test a three-test family without saying so, and files the mixed outcomes under §10's "H4 fails").**
   ADOPTED: revision 2 DECLARES the amendment to §2's H4 row and to §10's two H4 rows, and pre-writes THREE outcome sentences —
   all reject; T1 rejects but T2 or T3 does not; T1 does not reject.
2. **F7 (a SEEN result bearing on T2/T3 was omitted: BC — a state-only, one-step policy — sits within the very margin δ of the K = 20
   model on this corpus, `p4_4_baselines.json` −0.208 [−0.469, +0.054], and the prompt is inert there per P5.3b; and the fixed step
   budget supervises 64 targets per step at K = 1 against up to 1,280 at K = 20, loaded against small K in H4's predicted direction).**
   VERIFIED by the coordinator in the artifact and in `action_loss` (cross-entropy over every non-PAD position). ADOPTED, and it
   changes the design and the expectation: (i) Cell 4 names the result; (ii) the registered expectation on `mappo1000` is written
   AGAINST H4's optimistic reading — T1 expected to reject with a small effect, T2/T3 expected NOT to reject; (iii) a SECOND SUBJECT,
   `mix50` (P4.7's corpus, where P5.3b found the prompt load-bearing), joins the sweep as EXPLORATORY with the same design and the
   opposite expectation; (iv) a SECONDARY equal-supervision arm for K ∈ {1, 2} (batch scaled so that the targets per step equal
   K = 20's maximum) is registered as the budget-sensitivity check, with loss per supervised target co-reported.
3. **F6 (the evaluation-path gate's reference — the 500 per-draw `att_engine` rows of the K = 20 checkpoints — exists only in the
   gitignored `output/p8_4b_rederivation/` cells; the committed `p8_4b_rederivation.json` carries no per-draw row — verified).**
   ADOPTED: the reference rows are committed as `docs/data/p4_k20_att_engine_rows.json` (five seeds × 100 draws, the source cells'
   digests) by `BRIEF_42`'s FIRST commit, a precondition named in the row; the evaluation path is proven by evaluating P4's own five
   checkpoints as a reference arm at the campaign's commit and requiring their rows `==` the committed ones; device `cpu` and the
   engine seed 1000 pinned.
4. **F5 (the tensor-for-tensor gate was never demonstrated on `output/p4_dt`; no device, GPU or torch pin; no failure path).**
   ADOPTED: the K = 20 reproduction becomes a MEASUREMENT with a registered failure path (the sweep's own K = 20 arm is the sweep's
   reference; P4's checkpoints remain C1's published subject; a difference is reported with its magnitude), device, GPU and torch
   pinned.
5. **F2 (A25's "stand as registered … not tested" is not a state §2 knows; §10's H3-fails row loses its fallback and its evidence
   silently; its second trigger is undecidable without P7.4).** ADOPTED: H2 is *withdrawn from this paper's confirmatory set, reason
   given, rows retained as the specification*; the H3-fails row's second trigger is stated UNDECIDED, so the paper does not call the
   zero-shot gap a dynamics gap without the qualifier; C1 and C3 are stated as the paper's only claims.
6. **Minors (F3, F9 and the rest): ADOPTED** — windows END at step t (not "first K steps"); the partial-set rule; the one-sided Wilcoxon
   is new code with a second route required in `BRIEF_42`; δ becomes a SUPERIORITY margin here (A6's was an equivalence margin), its
   transfer from `att_ours` to `att_engine` supported by P5.3b's same contrast on both definitions (0.1180 against 0.1226).
7. The mandate's row line numbers for A6/A11/A13/A15 were off by one (the coordinator's error; the reviewer read the right rows).

---

# A25 / A26 draft review -- findings (incremental, append-only)
Reviewer: independent read-only reviewer (Claude Fable 5.1), started 2026-09-28
Repo: /home/filip/rltraffic @ main 3d319d0 (working tree)
Hard cap: 20 minutes. Items not reached are marked UNCHECKED at the end.

## CHECKLIST
- [ ] R1a A25: H2 rows "stand as registered" but untested -- coherent registration state?
- [ ] R1b A26: T2/T3 as confirmatory margin tests vs section 2 "Trend test over K" (one test) -- promotion?
- [ ] R1c A25/A26 vs sections 1,2,4,5,6,8,9,10 and A6/A11/A13/A15 -- undeclared changes?
- [ ] R2a A26 forking paths: contrast (ranks vs log K), one-sided direction, plateau level K=20
- [ ] R2b A26: delta transfer att_ours -> att_engine; fixed 40,000 budget
- [ ] R2c A26: reproduction gates (tensor equality vs P4; att_engine rows vs P5.3b/P8.4b) -- what if fail?
- [ ] R2d A26: per-draw unit (mean over seeds); missing draw; refused episode; family+Holm; decidability of holds/fails
- [ ] R3a train_dt takes context_length parameter (survey cite)
- [ ] R3b window index K-invariant (dataset.py:647-651) -- number of valid windows per episode vs K
- [ ] R3c DTAgent rollout K-step window, K=1 branch
- [ ] R3d P4 recipe values from dt_seed101.pt provenance/config; budget 20,000 -> 40,000 where recorded
- [ ] R3e bit-for-bit reproduction of P4 checkpoint ever demonstrated? seed handling / batch stream same?
- [ ] R3f P5.3b/P8.4b att_engine re-derivation of K=20 checkpoints (file, field, per-draw rows)
- [ ] R4 numbers: target_rtg -5762, rtg_scale 9991, draws 1000-1099, training draws 1-200, delta 0.6263, 202-204 s, 2.9 s, 25 trainings, 2,500 cells
- [ ] R5 A25 Cell 4: no perturbation module / p6_* / p7_4* / MAPPO DR checkpoint / branch -- re-check on disk
- [ ] R6 referee attacks: H2 drop & title/abstract; C1/C3 only claims; one corpus for H4; K=1 anchor fairness; fixed budget vs small K; rank contrast choice

## FINDINGS (append-only below this line)

### F0 -- 18:16 -- inputs read (drafts, survey, prereg sections 1,2,4,5,6,8,9,10) [R1 partial]
- Note on the request: the row line numbers given (398/393/391/389) hold A7/A12/A14/A16; A6/A11/A13/A15 are at 399/394/392/390 (survey H6 agrees). Reading those next.
- Verified text, PREREGISTRATION.md:95-114 (section 2): H4 row = ONE test, "Trend test over K in {1,2,5,10,20} on the P4 validation scenario", unit "paired evaluation draw"; rule: "Moving an analysis from exploratory to confirmatory after seeing data is forbidden. Moving one from confirmatory to exploratory is permitted only with an amendment row stating the reason."
- Verified text, :317-344 (section 10): H4 has exactly two outcome sentences (holds / fails); the H3-fails row reads "H3 fails / gap indistinguishable from the interface control -> C3 becomes a characterised limitation with the P7.4 control as evidence, and the paper stands on C1 + C2".
- A26 draft line 52-53: "H4's family for Holm-Bonferroni is {T1, T2, T3} ... H4 HOLDS if and only if all three reject after Holm." A26 line 60: "section 10's outcome rows for H4 stand unchanged". A26 does NOT declare section 2's H4 row amended.
- A25 draft line 24: "H4's context-length sweep exactly as section 1-2 register it, under A26" -- but A26 changes section 2's one-test family into three tests (see F1).

### F1 -- 18:20 -- [R1b] MAJOR (A26): section 2's ONE confirmatory test for H4 becomes THREE without declaring section 2 amended; and "holds iff all three reject" makes section 10's "H4 fails" sentence false in the mixed outcomes
- Draft text (A26 lines 48-53): "The plateau clause, as two further confirmatory tests ... H4's family for Holm-Bonferroni (section 8) is {T1, T2, T3} ... H4 HOLDS if and only if all three reject after Holm." Line 60: "section 10's outcome rows for H4 stand unchanged". Line 62 "(f) UNCHANGED" does not list section 2.
- Evidence: PREREGISTRATION.md:104 registers exactly one confirmatory test for H4 ("Trend test over K"); :111-112 forbids exploratory->confirmatory promotion only "after seeing data" (no K != 20 datum exists, so adding T2/T3 NOW is permitted, but it IS an amendment of section 2's H4 row and must be declared as one -- the draft instead says A25 line 24 "exactly as section 1-2 register it"). Direction: the conjunction is STRICTER for "holds" (conservative), so this is not a loosening -- but it must be said.
- The real defect: section 10's two sentences are "H4 holds -> DataLight's negative result is explained as a configuration artifact of K in {1,2}" / "H4 fails -> context length is not the explanation; DataLight's negative finding survives a proper K sweep". Under "holds iff T1&T2&T3", the outcome {T1 rejects, T2 rejects, T3 does not} (trend real, K=1 materially below, K=2 within delta of K=20) is filed as "H4 fails" and the paper would print "context length is not the explanation" while its own trend test rejected. Five of the eight reject-patterns are mixed; the registration pre-writes a sentence for none of them.
- Proposed change: (i) state explicitly "section 2's H4 row is AMENDED from one test to the family {T1,T2,T3}, before any K != 20 datum"; (ii) either make T1 the sole decider of section 10's holds/fails and T2/T3 the deciders of the SUB-CLAUSE "K in {1,2} materially below" (with a pre-written sentence per pattern), or pre-write the mixed-outcome sentences now (e.g. "trend holds but K=2 is not materially below: DataLight's K=2 configuration is NOT explained by context length; K=1 is"). Every pattern must map to a pre-written sentence.

### F2 -- 18:21 -- [R1a] MAJOR (A25): "NOT withdrawn ... stand as registered" and "not tested" is an incoherent registration state; and section 10's H3-fails row loses its fallback (C2) and its evidence (P7.4) silently
- Draft text (A25 lines 13-15): "section 1's RQ2/H2, section 2's confirmatory test ..., section 4's C2 design ... and section 10's two H2 outcome rows are NOT withdrawn and NOT edited: they stand as registered. They are not tested in this paper."
- Evidence: section 2 (:111-112) knows two states for an analysis, confirmatory or exploratory, and permits confirmatory->exploratory "only with an amendment row stating the reason". A registered confirmatory test with no data is neither; the honest state is "WITHDRAWN FROM THIS PAPER'S CONFIRMATORY SET (not from the registration; its rows remain the specification for a future registration)". "Stand as registered" + "not tested" invites a referee to ask which one is true.
- Section 10 H3-fails row (:326): "C3 becomes a characterised limitation with the P7.4 control as evidence, and the paper stands on C1 + C2." After A25 both "the P7.4 control" and "C2" are gone. A25 (b) says P7.4's "registered role does not arise: H3's clause 1 held" -- but the row's trigger is a disjunction "H3 fails / gap indistinguishable from the interface control", and the second disjunct is DECIDABLE ONLY BY RUNNING P7.4. Not running it leaves the second disjunct undecided, not "not arisen".
- Proposed change: A25 states (i) the registration state of H2 in section 2's vocabulary (withdrawn from this paper's confirmatory set, reason given, rows retained as specification); (ii) that section 10's H3-fails row is amended: its fallback "C1 + C2" becomes "C1" and "with the P7.4 control as evidence" is struck for this paper; (iii) that the second disjunct of that row's trigger is UNDECIDED, so the paper may not claim the zero-shot gap is a dynamics gap rather than an interface artifact (this belongs in the abstract-level claim limits of Cell 3, beside the C2 sentence).

### F3 -- 18:22 -- [R3a,R3b,R3c] code facts VERIFIED (survey cites hold), with one wording error in A26
- offline/dt_gate.py:747-765: train_dt(..., context_length: int, ...) -- a free parameter (:755); fed to DTConfig (:814); recorded in provenance (:888). CLI _run_train hardcodes CONTEXT_LENGTH=20 (:1376, :1414) -- a sweep needs its own driver (survey H2 correct).
- offline/dataset.py:645-651: index.append((stream_index, t)) for EVERY t in range(stream.length) regardless of K; K enters only __getitem__ (:783-786: low = max(0, t-K+1), left padding). So len(index) is K-invariant and dt_gate.py:828-830 `generator.integers(0, count, batch)` gives the identical index stream per seed at every K. VERIFIED. Model parameters are K-invariant too (absolute timestep embedding, max_ep_len from data), so init under torch.manual_seed is identical across K.
- WORDING ERROR, A26 line 30-31: "each carrying its first K steps of context" -- the window is the K steps ENDING at t (dataset.py:784-786: rows stream[low:t+1]); windows at different K share their LAST step (the supervised one), not their first. Propose: "each window being the K steps ending at the same sampled step t".
- agent/DTAgent.py:622-627: `if history else []` guard for K=1; :756-763 buffers trimmed to the last K; :876 context_length read from the checkpoint config. VERIFIED.
- Determinism: dt_gate.py:810 Utils.seed_everything (torch.manual_seed only); NO torch.use_deterministic_algorithms / cudnn flags anywhere in dt_gate.py or agent/utils/utils.py (grep). P4 trained on device "cuda" (RTX 5080 Laptop, torch 2.11.0+cu128, git b4a2fc6) -- checkpoint provenance.

### F4 -- 18:23 -- [R3d, R4] P4 recipe and numbers VERIFIED from output/p4_dt/dt_seed101.pt + docs/data
- Checkpoint config: state_dim 25, n_actions 8, context_length 20, n_layer 3, n_head 1, d_model 128, dropout 0.1, max_ep_len 360. Provenance: gradient_steps 40000, declared_gradient_steps 40000 (the raise re-ran train_dt from scratch with declared=40000, dt_gate.py:1428-1434 `run(int(args.raise_to))`), raise_to 40000, batch 64, lr 1e-4, wd 1e-4, warmup 1000, grad_clip 0.25, plateaued False, device cuda.
- docs/data/p4_dt_config.json budget: declared 20000, raise_to 40000, raise_taken true, reported 40000 -- the 20,000->40,000 raise IS recorded there and in p4_training.json. conditioning: target_rtg -5762.0, rtg_scale 9991.0. Draws: p4_gate.json draw_ids 1000..1099 (n=100), engine_seed 1000, env device cpu; 500 episodes, arm madt only, fields att_horizon (no att_engine). A6 (:399) delta = 0.6263 = 105.5820 - 104.9558 on att_ours. All A26 numbers match.
- Costs (survey H5 -> docs/data/p4_training.json seeds[*].seconds): 204.08 / 202.92 / 202.35 s (101/202/303), 356.2 s (404), 14,018 s (505, anomalous). A26 quotes no cost; the survey's "~205 s" is the three-seed figure.
- Fresh 40,000-step train_dt at the same seed is the same procedure P4's raise ran (from-scratch, warmup min(1000, 20000)=1000), so the tensor gate is at least well-posed. Whether it PASSES depends on CUDA determinism (F3) -- see F6.

### F5 -- 18:25 -- [R3e, R2c] MAJOR (A26): the tensor-equality gate has never been demonstrated on output/p4_dt, is not pinned to the device it needs, and has no registered path on failure
- Draft text (A26 lines 26-29): "the five retrained K = 20 models must equal P4's five checkpoints TENSOR FOR TENSOR (torch.equal on every parameter ...) ... a difference stops the sweep before any other K is evaluated and is a finding, not a number."
- Evidence: (i) grep docs/ for the P4 checkpoint sha (dc6fc97c...) / "tensor for tensor" / "bit for bit" + p4_dt: NO record of any re-training reproducing output/p4_dt/dt_seed*.pt. The only demonstrated reproduction is tests/test_train_dt_rtg_mode.py:289-333 (Gate 2, CUDA-only, skips without CUDA :278-285) on P4.6's (mappo500, dt, 101) cell, digest 5d98d5... in docs/data/p4_6_training.json. (ii) P4's checkpoints were written at b4a2fc6 on "cuda" (provenance.device, RTX 5080 Laptop, torch 2.11.0+cu128); `git log b4a2fc6..HEAD -- offline/dt_gate.py agent/DTAgent.py offline/dataset.py` = 10 commits (incl. 12f31e1 rtg_mode on DTAgent.py); no numerics change is known, none is verified. (iii) No torch.use_deterministic_algorithms / cudnn flag anywhere (F3); reproduction relies on single-stream cuBLAS determinism on the same GPU -- so a CPU training, another GPU or another torch build will NOT reproduce, and A26 pins none of device/GPU/torch version. (iv) On failure the draft halts H4 with no registered alternative -- the paper's one remaining confirmatory experiment would have no pre-declared continuation.
- Proposed change: pin device "cuda", the GPU model and torch 2.11.0+cu128 in (b); declare the demonstrated evidence honestly ("reproduction has been shown for a P4.6 cell, never for output/p4_dt"); register the failure path now, e.g. "if any tensor differs, the sweep still proceeds with the retrained K=20 arm as the ONLY K=20 arm, the difference is reported with its max-abs value and the evaluation-path gate (F6) decides identity of behaviour; P4's checkpoints are then not cited as the K=20 arm" -- or any other rule, but one written before the run.

### F6 -- 18:26 -- [R3f, R2c] MAJOR (A26): the evaluation-path reproduction gate references per-draw att_engine rows that exist only in a gitignored directory
- Draft text (A26 lines 39-41): "P5.3b and P8.4b re-derived the same five K = 20 checkpoints under att_engine (mean 100.7032), and the sweep's K = 20 arm must reproduce those re-derived rows under == on every draw".
- Evidence: docs/data/p5_3b_nortg.json reference_dt_cells.mappo1000 holds ONLY att_engine_mean 100.70319448769442 / att_ours_mean 104.95575898180847 ("the engine mean has no committed counterpart because P4.6/P4.7 predate A11"); its comparisons...att_engine.paired holds draw_ids but no per-draw values; p5_3b episodes (1,500) are dt_nortg arms only; docs/data/p4_6_grid.json has att_horizon only (0 occurrences of att_engine). The 500 per-draw rows live in output/p8_4b_rederivation/cell_hz1x1_dt_at_mappo1000_seed{101..505}_draw{1000..1099}.json -- output/ is gitignored (.gitignore:228); docs/data/p8_4b_rederivation.json carries contrasts only (survey H3 part 2, confirmed).
- Consequence: a reader cannot check the gate; the reference could be regenerated or altered without trace.
- Proposed change: before the tag, commit the 500 reference rows (or a sha256 manifest of the 500 cell files plus their att_engine values) under docs/data/, name that path in A26, and state that the gate's reference is that committed object; also pin the evaluation device (P4 gate: env_settings.device "cpu", p4_gate.json) so an argmax flip from GPU floats cannot be mistaken for a trainer difference.

### F7 -- 18:27 -- [R6, R2b] MAJOR (A26): Cell 4 omits an ALREADY-SEEN result that bears directly on T2/T3 -- P4.4's BC "matches" the K=20 DT within the very delta A26 reuses -- and the fixed-step budget couples K to supervision count
- Evidence: docs/data/p4_4_baselines.json comparisons.madt_vs_bc: mean difference -0.2077, CI [-0.4690, +0.0536], n 100, verdict "matches" under delta 0.6263; bc_top10 and IQL are "baseline_genuinely_better" (+1.79, +1.48). BC conditions on the current state only, i.e. it is the K=1 anchor minus the RTG token (which P5.3b shows inert on this corpus: dt - dt_nortg +0.1226 [-0.129, +0.374], p 0.31) and minus the timestep embedding. So the committed record already says a one-step policy lands within delta of K=20 on mappo1000 -- a direct prior that T2 (K=1 below K=20 by more than delta) will NOT reject.
- Draft text: A26 Cell 4 names P4.2, P4.3, P4.4 ("the baselines") only as "numbers derived from" the K=20 checkpoints and never states the BC-matches verdict; Cell 2(d) registers the expectation "monotone improvement from K = 1 to K = 20 with the largest step between K = 2 and K = 5" as if no evidence existed. A referee who reads P4.4 will ask why the registration's expectation contradicts the record it cites.
- Budget: agent/DTAgent.py action_loss = cross-entropy over EVERY non-PAD window position (logits.reshape(-1, n_actions), ignore_index=PAD). At K=20 one gradient step supervises up to 64x20 = 1,280 action targets; at K=1, 64. "40,000 gradient steps for every K" therefore gives K=1 roughly 1/20 of the supervised targets -- the fixed budget is not neutral, it is loaded against small K in the direction H4 predicts. A26 line 72 admits only "a K = 1 model that would need more steps is not given them".
- Proposed change: Cell 4 names P4.4's BC-matches verdict and P5.3b's inert prompt as SEEN results bearing on T2/T3 (and, if the expectation is kept, says why it is held despite them); Cell 3 states that H4 is tested on the one corpus where sequence modelling at K=20 has already been shown to add nothing over one-step BC within delta, so an "H4 fails" outcome has a registered alternative explanation (the corpus) and section 10's "context length is not the explanation" sentence is softened accordingly (see F1); (b) states the targets-per-step asymmetry as a number (64 vs 128 vs 320 vs 640 vs 1,280 targets per step at the five K) and co-reports the loss-per-target curves.

### F8 -- 18:28 -- [R5, R4] VERIFIED: A25 Cell 4's on-disk claims hold; A26's numbers hold
- offline/: only flow_randomizer.py matches perturb/closure/dropout/shift/random/augment/domain. No output/*p6*, *p7_4* in main or any of the 12 linked worktrees (`git worktree list`, loop over /home/filip/rltraffic-*). `git branch -a`: no p6/p7.4/perturb/domain branch. docs/data: no p6/p7_4 artifact. No MAPPO checkpoint file with dr/shift/random/domain in its name under output/ (only P5.2/P8.4b evaluation JSONs match "random").
- Numbers: target_rtg -5762.0, rtg_scale 9991.0 (p4_dt_config.json, checkpoint); held-out 1000-1099 (n 100); training draws 1..200 (len 200); delta 0.6263 (A6, p4_4_baselines.json equivalence_margin_delta); training seconds 204.1/202.9/202.3 (+356.2, +14,018 anomalous) from p4_training.json; 2.897 s mean episode over 1,500 rows of p5_3b_nortg.json; 5x5 = 25 trainings; 25x100 = 2,500 cells. BRIEF_42 and docs/plans/p5.3c do not exist yet (briefs end at 41) -- consistent with the drafts.

### F9 -- 18:29 -- [R2a, R2d, R1c] MINOR items (A26) and the decidability check
- Missing-draw / refused-episode rule: A26 states none. Section 8 (:289-292) only says exclusions are for infrastructure failure and are counted. Propose: "a (K, seed, draw) cell that fails is re-run identically (CityFlow hz1x1 is bit-reproducible, section 5); a draw missing in any arm after that is dropped from ALL arms for T1-T3 and counted; n < 100 is reported".
- One-sided Wilcoxon: the repo's wilcoxon_signed_rank (dt_gate.py:396-425) is two-sided; the one-sided p is new code and must be computed by a second independent route (CLAUDE.md section 2). Ranks vs log K: log2 K = (0, 1, 2.32, 3.32, 4.32) is nearly equally spaced, so the choice is immaterial in practice; declaring ranks is fine. Plateau level := K=20 is pre-declared; if the curve is non-monotone (K=10 best) the wording "plateau" is descriptive only -- say so.
- delta's transfer att_ours -> att_engine: SUPPORTED by committed evidence the draft does not cite -- p5_3b_nortg.json paired dt - dt_nortg is +0.1226 (att_engine) vs +0.1180 (att_ours). Cite it. Note A6 used delta as an EQUIVALENCE margin (CI inside +/-delta); A26 uses it as a SUPERIORITY margin (difference > delta) -- the complement; state that explicitly.
- A26 (a) omits engine_seed 1000 (p4_gate.json) -- inert on hz1x1 but the == gate needs the same value; pin it.
- Decidability: every reject-pattern of {T1,T2,T3} maps to holds/fails, so the rule is decidable; the defect is the sentence it triggers (F1).
- A25 (c) "H4 ... exactly as section 1-2 register it, under A26" contradicts A26's three-test family (F1); A25 (d) "the multiplicity rule ... within the families that are tested (H1, H3, H4)" is consistent with section 8.
- R6 not answered by A25: the paper's abstract/introduction must not claim (i) robustness to scenario shift, (ii) that the zero-shot gap is a dynamics gap rather than an interface artifact (P7.4 not run), (iii) three claims. A25 covers (i) and (iii); add (ii). Whether "C1's ladder and C3's curve are the only claims" should be stated: yes, in Cell 2(c), one sentence, so section 10's grid4x4-collapse row and the H4 hook are read as findings inside C1/H4, not as extra claims.

## UNCHECKED at the cap (18:30)
- R1c beyond the rows read: sections 3, 7, 11 and A17-A24 not re-read against the drafts (A19/A24 define H3's clause status that A25 (b) relies on -- "clause 1 held on both scenarios" NOT re-verified from the rows).
- Whether any of the 10 commits since b4a2fc6 changed trainer numerics (F5 (ii)) -- not bisected.
- The contents of output/p8_4b_rederivation/cell_hz1x1_dt_at_mappo1000_* (field names, that att_engine per draw is present) -- listed, not opened.

### F10 -- 18:24 wall clock -- late verifications (some UNCHECKED items resolved; earlier blocks not rewritten)
- A25 (b) "H3's clause 1 held on both scenarios": CONSISTENT with A19 (:386, clause 1 HOLDS on hz1x1 via P7.3a; clause 2 REFUTED; clause 3 VOID) and A24 (:380, clause 1 confirmatory HOLDS on grid4x4). Verified.
- F5 (ii) refined: `git diff --stat b4a2fc6 HEAD` -- offline/dataset.py UNCHANGED (0 lines); agent/DTAgent.py +63/-x, every code change is the rtg_mode plumbing, inert under the default "conditioned" (`if config.rtg_mode == "zero": rtg = zeros`); offline/dt_gate.py +481 lines, not audited line by line. So the tensor gate is PLAUSIBLE to pass on the same GPU; the finding stands as: never demonstrated for output/p4_dt, device/torch not pinned, no failure path.
- F6 refined: output/p8_4b_rederivation/cell_hz1x1_dt_at_mappo1000_seed101_draw1000.json holds att_engine 102.08328736900165, att_ours 106.4561500275786, fields committed_att_ours / reproduces_committed -- the per-draw reference exists, on disk only, gitignored (.gitignore:228).
- A26 line 34-35 "no test exercises the agent at K = 1 or 2 today": TRUE -- tests/test_d4rl_calibration.py:414,451,479 pass context_length=2 to bc_windows (a window builder), not to DTAgent.act (0 DTAgent/act calls in that range).
- Sections 3, 7, 11 (:115-182, :254-268, :345-379) contain no H4/context/K text (grep) -- nothing there for the drafts to contradict.
- Still UNCHECKED: line-by-line audit of dt_gate.py's +481 lines for a numerics change in train_dt between b4a2fc6 and P4.6's commit.

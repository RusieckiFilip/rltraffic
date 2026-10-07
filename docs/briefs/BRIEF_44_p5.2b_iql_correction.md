# BRIEF_44 — P5.2b: the IQL random-tier correction run (`DEFERRED` 106) — one registered cell re-run under its registered condition

**Mode:** Explore → Plan (gate G0: `docs/plans/p5.2b.md`) → Code (tests first) → the correction run (gate G2, the author's token) →
Commit. Branch `task/p5.2b-iql-correction` from `main`. One task, one session.

**Authorised by:** the author's ruling of 2026-10-05 (*"Ruling: A — the correction run for P5.2's IQL cell at grid4x4 random on the
declared 200 episodes, as the narrow exception you describe"*), recorded in `PROJECT_PLAN` §8 the same day. **A narrow exception to
A25** (*"no new experiments"*): it re-executes ONE registered cell of P5.2 under the condition P5.2's declaration registered, and
evaluates no new hypothesis.

## 0. Frozen interface contracts (read them from disk; do not infer)
- `docs/CONTRACTS.md` (C6 v1.1; the env API `info = env.reset(seed=...)`, `reward, terminated, truncated, info = env.step(action)`).
- Frozen paths (CLAUDE.md §1) untouched. **Every existing module UNCHANGED except ONE authorised change** (§4.1):
  `offline/tier_sweep.py`'s IQL transition table in `_run_train_baselines` (`:2078`), filtered to the declared streams exactly as
  `offline/method_tier_grid.py:2487-2490` does. No other existing file of `offline/`, `agent/`, `tests/` (beyond additions) changes.
- No new dependency. Nothing under `output/p5_2/` is written, moved or deleted: P5.2's checkpoints and evaluations are the record of
  the defect and stay as they are.

## 1. Why this task exists — the defect (`DEFERRED` 106; `docs/reviews/P8.2.md` §2)
P5.2 (`BRIEF_27`, `offline/tier_sweep.py`) trained BC, %BC and IQL on grid4x4 at four data tiers. Its declaration selects 200
episodes per tier; the random tier is the only one with more available (400; `docs/data/p5_2_declaration_random.json`).
`_run_train_baselines` filters BC's and %BC's batches to the declared streams (`:2038`, `filter_stacked_to_streams`) but builds IQL's
transition table from the WHOLE dataset (`:2078`, `build_transitions(dataset, …)`, which reads every `dataset.episode_records`
entry). So **IQL at the random tier trained on 400 episodes (2,304,000 transitions, all five seeds), every other arm on the declared
200 (1,152,000)**, and the checkpoints' `training_streams: 3200` (the selected streams' count) hides it. The coordinator confirmed the
population of call sites on 2026-10-05: of the four `build_transitions` / `train_iql` call sites in `offline/`, `method_tier_grid.py`
(P4.6/P4.7) filters to the selected streams, `offline_baselines.py:3009` (P4.4) and `spatial_mixing.py:1071` (P5.1) build BC and IQL
from the same full dataset with no declared subset, and `tier_sweep.py:2078` is the one defect; the P8.2 table's data column shows
IQL's data equal to BC's on every other tier of both scenarios. The cell carries P5.2's *"IQL won the random tier"*, one of Q1's four
misses (`docs/returns/P5.2.md:35`), and IQL's pairs in Q2b's ordering tally (recorded maximum 9 against a threshold of 12).

## 2. Scope fence — what NOT to build
- No other cell re-run; no other tier, method, scenario or seed. No new hypothesis, no new registration, no change to P5.2's
  committed statements in place — the correction is a NEW artifact beside them, with both versions stated.
- No re-measurement of the random tier's OTHER arms under either definition: their `att_ours` is P5.2's committed evaluation
  (`output/p5_2/eval_random_<method>.json`) and their `att_engine` with its companions is P8.4b's committed campaign
  (`output/p8_4b_rederivation/cell_grid4x4_<method>_at_random_seed<s>_draw<d>.json`, 500 cells per arm, every one
  `reproduces_committed`). Both are READ, at their manifests' digests, never re-run.
- No training of anything but IQL at grid4x4's random tier, five seeds.

## 3. What the task produces
1. **The corrected cell:** IQL at grid4x4's random tier trained on the declared 200 episodes' streams — 3,200 streams, 1,152,000
   transitions — with P5.2's exact hyperparameters as its checkpoints record them (seeds 101, 202, 303, 404, 505; 40,000 gradient
   steps; batch 1,280 transitions; learning rate 1e-4; weight decay 1e-4; gradient clip 0.25; the reward scale from the selected
   streams' returns, 0.74294…; τ 0.7, β 3.0, γ 0.99; CUDA), checkpoints under `output/p5_2b/checkpoints/`, a training record WITH
   wall seconds (P5.2's were lost) under `output/p5_2b/`.
2. **Its evaluation, two definitions on the same episodes:** (i) `att_ours` (`att_horizon`) by P5.2's own evaluation path
   (`offline.dt_gate.evaluate_arm`, as `tier_sweep.py`'s `evaluate` subcommand calls it) with EXACTLY the arguments
   `output/p5_2/eval_random_iql.json` records — draws 1000–1099, seeds 101–505, engine seed 1000, its `deterministic` flag, 40,000
   declared steps — written in P5.2's format; (ii) `att_ours` AND `att_engine`, `entered`, `created`, `never_entered` for the same
   500 episodes through P8.4b's own cell runner (`offline/att_rederivation.py`), unmodified, in the format of P8.4b's
   `cell_grid4x4_iql_at_random_*` files, written under `output/p5_2b/` — never into `output/p8_4b_rederivation/`. The plan shows how
   the runner takes the corrected checkpoints; if it cannot without a change to an existing module, the plan says so and G0 rules.
3. **The recomputation:** P5.2's registered random-tier statements — Q1's `iql@random` entry, Q2b at the random tier (P5.2's own
   `concordance`), Q3 at the random tier, and the random tier's ranking — recomputed with the corrected cell in place of the original,
   by P5.2's OWN report code, under BOTH definitions: `att_ours` (P5.2's registered metric, like for like
   with its committed statements) and `att_engine` (A11's Rule R makes it the primary metric of every paper claim on grid4x4; the
   other arms from P8.4b's cells), each statement reported under both, the paper's sentence resting on `att_engine`. Artifact
   `output/p5_2b/artifacts/p5_2b_correction.json` (format `p5.2b-correction/1.0`), committed by hand as
   `docs/data/p5_2b_iql_correction.json` with a T-regress: the defect (mechanism, file:line, rows), the original and the corrected
   cell under both definitions, every random-tier statement BEFORE and AFTER under both, and `what_this_does_not_say` (the original
   cell is kept as the record of the defect; the other arms were not re-run; no other tier is touched).
4. **P8.2's table updated** (`docs/data/p8_2_compute.json`): `grid4x4.iql`'s random-tier training entry from the corrected
   checkpoints and record, with a note naming the defect and the original entry's values. **`DEFERRED` 107's tests come FIRST** (they
   bind the next change to `offline/compute_table.py`): positional assertions for both generated sentences, the ungated CI test
   recomputing each summary from the artifact's own groups and asserting 9 cpu + 9 cuda group entries, the H4 summary against
   `docs/notes/p8_2_g3/g3_analysis.json`, the C2 line ranges parsed from the sentences. Then the regenerated artifact and its T-regress.

## 4. Per-file requirements
1. **`offline/tier_sweep.py` — the ONE authorised change:** at `_run_train_baselines`, IQL's table becomes
   `filter_transitions_to_streams(build_transitions(dataset, group=group, reward_scale=scale), transition_stream_keys(dataset, group),
   streams)` (the helpers `method_tier_grid.py` already uses), and the run REFUSES unless the filtered table holds exactly the declared
   streams' transitions. Nothing else in the file changes. (A re-run of P5.2's training must never reproduce the defect.)
2. **A new module** (the plan names it) for the correction: loads the random tier exactly as `tier_sweep.py` does (its own loader
   functions, imported), builds the corrected table through §4.1's code, trains the five seeds through `offline_baselines.train_iql`,
   evaluates per §3.2, builds the §3.3 artifact by importing P5.2's report functions; a CLI; nothing written outside `output/p5_2b/`
   and `output/SHA256SUMS_p5_2b.txt`.
3. **A lean driver** `offline/campaigns/p5_2b_correction.sh` on the P8.2 driver's pattern: detached run worktree at the reviewed
   commit, the regime (OMP/MKL 1, no `CUBLAS_WORKSPACE_CONFIG`, no tracer variable), every refusal before the author's token
   `output/p5_2b_runs/TOKEN_correction`, the manifest written once on completion.
4. **`offline/compute_table.py`** for §3.4 only, after `DEFERRED` 107's tests.

## 5. Tests — first, red for their own reasons; the named mutations executed, committed and pasted
- **T-rows (load-bearing):** the corrected table holds exactly the declared streams' transitions — 1,152,000, its stream set equal to
  BC's `streams` — checked by a SECOND route (the declaration's `selected_episodes` × its intersections × 360, read with `json`).
  *Mutation:* the filter removed → 2,304,000 → dies.
- **T-tier_sweep:** `_run_train_baselines`'s IQL table on a synthetic tier with more episodes available than selected holds only the
  selected streams — RED on the current code (the defect, demonstrated), green after §4.1.
- **T-protocol (load-bearing):** the evaluation's arguments equal `output/p5_2/eval_random_iql.json`'s (draw ids, seeds, engine seed,
  `deterministic`, declared steps), read from that file by the test. *Mutation:* one draw or the engine seed changed → dies.
- **T-reproduce (load-bearing, the falsification of the whole path):** (a) P5.2's evaluation path re-run on the ORIGINAL
  `grid4x4_random_iql_seed101.pt` for draws 1000–1004 reproduces the committed `att_horizon` values of those five episodes exactly;
  (b) P8.4b's cell runner on the ORIGINAL `grid4x4_random_iql_seed101.pt` for draws 1000–1004 reproduces P8.4b's committed cells'
  `att_ours` AND `att_engine` exactly; (c) the recomputation fed the ORIGINAL `eval_random_iql.json` reproduces every committed
  random-tier statement of P5.2 exactly (Q1's entry 190.96 against 414.99, Q2b's tally, Q3's rank), and fed the ORIGINAL P8.4b cells
  it reproduces the same statements under `att_engine` as the coordinator's G3 recomputes them. Gated tests name their artifact.
- **T-isolation:** nothing is written under `output/p5_2/`; the module refuses an output root that resolves there.
- **`DEFERRED` 107's tests** (§3.4), each red against a mutant M2 left alive (`docs/reviews/P8.2.md` §4: M2-2a, M2-2b, M2-3e, M2-3f,
  M2-6c) before the builder change.
Suite discipline as `BRIEF_43`: hygiene and English falsified first; mutants COMMITTED in a throwaway worktree; the new and changed
test files once in a depth-1 clone made by `git clone --depth 1 --branch <branch> file://…`.

## 6. Gates, in order
| # | Gate | Runs it | Checks | Stops the task if | You learn it by |
|---|---|---|---|---|---|
| G0 | Plan | coordinator, from `docs/plans/p5.2b.md` | the loader, the filter, the evaluation's two paths and their argument equality with P5.2's record, the report functions reused, the outputs, the tests | a path that is not P5.2's | Amendment A on `main` |
| G1 | Code review + pre-flight | coordinator + one reviewer (≤ 15 min, findings file); the coordinator's mutants | T-rows, T-protocol, T-reproduce (run, not described); one seed's training timed for a short run; the duration estimated | a defect | Amendment B |
| G2 | The correction run | **the author** — one command, a quiet machine, the token | — | — | the capture |
| G3 | Read | coordinator, from disk: the five checkpoints' rows and digests, the episodes recomputed into the cell's mean by an independent route, the random-tier statements recomputed | the artifact | a number untraceable or wrong | Amendment C |
| G4 | Packet | the implementer → **"P5.2b done"** | §8 | — | — |
| G5 | Merge review | coordinator spawns one (two mandates: the numbers to their sources; the code by mutation) + the depth-1 run | — | a blocker | the merge, §6's box ticked, `DEFERRED` 106 and 107 closed, the CI ceiling if it moves |

## 7. Definition of Done
- [ ] `docs/plans/p5.2b.md` approved (G0).
- [ ] §4.1's change with T-tier_sweep red then green; the new module, the driver and their tests; every §5 test red first, then
      green; every named mutation executed and pasted; hygiene and English; the depth-1 run.
- [ ] The correction run complete (G2), its records under `output/SHA256SUMS_p5_2b.txt`.
- [ ] `docs/data/p5_2b_iql_correction.json` committed by hand, byte-identical to the builder's output, with its T-regress.
- [ ] `DEFERRED` 107's tests, then P8.2's table updated and regenerated with its T-regress.
- [ ] `docs/returns/P5.2b.md` per §8; then "P5.2b done". No frozen file, no other existing module, no new dependency touched.

## 8. Return Packet
`docs/returns/TEMPLATE.md`, plus: the defect and its correction in one paragraph; the corrected cell beside the original (both
definitions for the corrected one); every random-tier statement before and after; the training record's seconds; the capture; what
the correction does not say; the AI-assistance record's four lines; one paragraph on what the paper's grid4x4 random-tier sentences
will assume.

---

# ✅ AMENDMENT A — 2026-10-06, gate G0: PLAN APPROVED (`docs/plans/p5.2b.md` @ `b40bc3f`, 448 lines) — four corrections to this brief, Q1–Q13 ruled, one existing test's change AUTHORISED in writing

## A0 — Verdict and what the coordinator verified (by running commands)
Approved. The plan read the code instead of the brief's description of it and found four places where the brief was wrong (A1).
Verified independently: **V11** — `admission_probe._method_checkpoint` resolves every grid4x4 tier but `mappo1000` to
`<roots.output_root>/p5_2/checkpoints/grid4x4_<tier>_<method>_seed<s>.pt` (`admission_probe.py:819-842`); **the test F6(c) names**
asserts null seconds for every entry of `grid4x4.bc`, `grid4x4.bc_top10`, `grid4x4.bc_top10_perix` and `grid4x4.iql`
(`tests/test_compute_table.py:664-673`); **F2's premise** — `119cc48` changed documentation only (`docs/PROJECT_PLAN.md` AND
`docs/briefs/BRIEF_27_p5.2_tier_sweep.md`: two files, not the plan's "only `docs/PROJECT_PLAN.md`"; the substance holds, no code
scored Q2a, Q3 or the ranking); **F11** — P5.2's own `concordance` on the committed eval files gives Q2b = 11 / 7 / 8 of 15 at
maxpressure / fixedtime / random with the tie rule declared on 2026-08-24, and 11 / **8** / 8 without it; the fixed-time tier
carries two exact ties (`dt_spatial` = `dt_nomix`, `bc` = `bc_top10_perix`), so the packet's 8 (`docs/returns/P5.2.md:27`) is the
count WITHOUT the declared rule — the verdict (FAILED, threshold 12) is unchanged (`DEFERRED` 108).

## A1 — Corrections to this brief
1. **§3.1 — the corrected checkpoints live at `output/p5_2b/p5_2/checkpoints/grid4x4_random_iql_seed<s>.pt`** (F1, option B): one
   physical copy that P5.2's `evaluate` subcommand (`--checkpoint-dir`), P8.4b's runner (`ProbeRoots(output_root=output/p5_2b)`)
   and P8.2's builder all read; no link, no change to an existing module.
2. **§3.3 — "P5.2's OWN report code" exists for Q1 (`score_level`) and Q2b (`concordance`, `predicted_order`) only;** Q2a, Q3a,
   Q3c and the ranking were scored by hand at `119cc48`. They are composed in the new module from P5.2's registered definitions
   (`docs/plans/p5.2.md` §4) and primitives (`dt_gate._per_draw_means`, `dt_gate.mean_ci95`, sorted levels), and T-reproduce (c)
   proves every composed value against the committed one (F2).
3. **§0 / §4 — TWO existing modules change, each for one purpose:** `offline/tier_sweep.py` (§4.1 — one new function
   `iql_transition_table`, its call at `:2078`, its `__all__` entry, nothing else, F5) and `offline/compute_table.py` (§4.4, for
   §3.4 only). §0's "ONE" was the coordinator's slip; §4.4 always named the second.
4. **§2 — P8.4b's cells have no per-file digest manifest;** "READ, at their manifests' digests" becomes F3's anchors: the campaign
   manifest's `declared_cells_sha256` recomputed from its own cell list and equal to `CAMPAIGN_COMPLETE`'s; every cell read has
   `reproduces_committed` true and an `att_ours` equal to P5.2's digest-pinned `att_horizon` for its key; the six random-tier
   means equal `docs/notes/readme_2026-10-05/c1_rule_r.json`'s under its own route. The artifact states that the `att_engine`
   values are pinned by the campaign's completeness and these cross-checks, not by a digest.

## A2 — Rulings: Q1–Q13
**All accepted as proposed**, with these specifics:
- **Q3 — the statement set:** accepted (Q1 entry and aggregate, the ranking, Q2a, Q2b with IQL's five pairs, Q3a, Q3c, each under
  both definitions, before and after; Q2b-hard under `att_ours` as an asserted invariance; Q3b, Q4, Q5, Q6 listed as not
  recomputed). Q2b-hard and the IQL-free statements under `att_engine` stay OUT of this task: they are the Rule-R recomputation of
  P5.2's statements, recorded as `DEFERRED` 109 and scheduled with the paper's C1 section.
- **Q8(c) — AUTHORISATION, written and dated 2026-10-06, to be quoted in the packet:** the test
  `test_t_sources_the_declared_absences_carry_their_reasons` (`tests/test_compute_table.py:664`) MAY change so that exactly one
  entry — `(grid4x4.iql, random)` — is excepted
  from its null-seconds loop, on these conditions: (i) the loop still asserts null seconds with a P5.2 / P5.1 reason for every
  other entry of the four rows (15 of 16), and asserts that exactly ONE entry is excepted; (ii) a NEW test asserts that the
  excepted entry's five seconds are sourced `{file: docs/data/p5_2b_iql_correction.json, sha256, json_path}` and equal that file's
  values read with `json`; (iii) the change is its own commit, after the builder change that makes it necessary, and nothing else
  in the test changes. Any other edit to a committed test still needs its own authorisation.
- **Q12 — the regime:** P5.2's default CUDA regime, one realisation, disclosed in the artifact; no second realisation.
- **Q13 — F11:** the coordinator's (`DEFERRED` 108); not this task's.

## A3 — Requirements added
1. **G1 runs, not describes, T-reproduce (a) and (b)** on this machine from the committed task tree, and the pre-flight's record
   carries their outcome beside the timings; a T-reproduce failure stops the task before any token.
2. **The packet states the plan's one precision slip** (A0, `119cc48`'s two files) where the plan quoted it.

## A4 — Next
The plan's §10 commits (2)–(7) in its order — `DEFERRED` 107's tests red against their mutants first; the skeletons and the red
tests with T-tier_sweep RED on the defect; §4.1; the module; the driver — then the mutations, `check_test_hygiene.sh` and
`check_english.sh` falsified first and run, the whole suite, the new and changed test files once in a depth-1 clone (F.1's
command), and the G1 pre-flight (`--preflight`, A3.1) — committed on the branch, NOT pushed; then **"P5.2b C2–C7 done"** → gate G1.

---

# ⚠️ AMENDMENT B — 2026-10-06, gate G1: FIX FIRST (small) — no blocker; A1.4's anchors made real, no second realisation possible, the test gaps closed; the pre-flight STAYS PINNED

## B0 — Verdict and what was verified (the record: `docs/reviews/P5.2b-G1.md`)
Three reviewers (≤ 15 min each, findings files verbatim in the record): RA1 (inputs, isolation, training, the two evaluation paths),
RA2 (the recomputation and the report — its own route, no numpy and no project import, reproduced every committed att_ours value and
agreed with the module under att_engine), RB (the `tier_sweep.py` fix — exactly A1.3's change —, the driver, `DEFERRED` 107's tests:
all five named mutants killed for the right reason). All PASS-WITH-NOTES, 0 blocking. **The coordinator's five mutants, committed at
`8c9f988`, every data gate open: 4 KILLED, 1 SURVIVED (PM4).** **The pre-flight `preflight_20261006T195405Z`, read from disk:**
COMPLETE at `a71a72c`, 13 / 13 manifest lines `OK`, canaries 0.76 / 0.77 s at speed and reproduced, T-reproduce (a) and (b)
reproduced with 0 differences each, 0.00832 s per training step, 1.93 s per episode, an estimate of ≈ 3,628 s.

## B1 — Required before G2, tests red first for their own reasons
1. **A1.4's third anchor, ENFORCED by the module (RA1 MAJOR, RA2):** before the report reads any P8.4b value, the module recomputes
   the six random-tier means under BOTH definitions by `docs/notes/readme_2026-10-05/c1_rule_r.json`'s own route
   (`statistics.mean` over each arm's cell files) and REFUSES unless each equals the note's value; the artifact says what the note
   was used for. Test: a synthetic P8.4b tree with one `att_engine` value altered → refusal naming the arm.
2. **A1.4's second anchor, TESTED (PM4 survived):** a test feeding `_p8_4b_rows` a cell whose `att_ours` differs from P5.2's
   committed `att_horizon` → refusal. PM4 must die.
3. **No second realisation, ever (RB MINOR-2, A2 Q12):** a COMPLETE training whose closing canary failed must not be re-trained. On
   restart the closing canary alone is taken (late), and the training record marks its seconds as bracketed by the opening canary
   only, the closing one taken after a restart (with both times); P8.2's table then quotes those seconds with that note. Tests: the
   restart after a failed closing canary runs no training and refuses to; the record carries the mark.
4. **The identity refusal tested (RB MINOR-1, MT2 survived):** a filter that swaps one declared stream for an undeclared stream of
   the same length → `iql_transition_table` refuses on identity. MT2 must die.
5. **The H4 maximum over both devices (RB MINOR-3):** the fixture's largest H4 change on a `cuda` cell, so a one-device maximum dies.
6. **The module's guards (RA1 MINORs):** (a) `assert_out_root` refuses an output root that does not hold the pinned
   `SHA256SUMS_p5_2.txt` (no root under `output/p5_2` is ever accepted); (b) the thirteen original hyperparameters (batch, learning
   rate, weight decay, gradient clip, τ, β, γ, Polyak, weight clip, steps, streams, scale, threads) ENFORCED equal to the original
   checkpoints' records, refusing on drift; (c) a test triggering each of path (ii)'s three refusals; (d) the module docstring's
   "every write goes through `assert_target`" corrected to what is true; (e) the corpus manifest the second route reads pinned by
   its sha256 and checked.
7. **The report's precision (RA2 MINORs):** (a) T-reproduce (c) asserts Q1's `n_cells == 19`; (b) its `att_engine` half also checks
   Q1 (`n_held`, the IQL entry, the outcome), Q2a, Q3a, IQL's five pairs and Q3c's reading against the test's own recomputation;
   (c) the corrected (i) file's level asserted equal to its own `cell.att_horizon_mean`; (d) the `q1` block under `att_engine` itself
   carries the flag that the predictions are the registered `att_horizon`-era values; (e) `cells.corrected` carries its `sources`;
   (f) the corrected weight digests asserted to differ from the originals'.
8. **Optional (NOTEs, the implementer's call, stated in the packet):** the report stage without the 100 materialised draws (so
   T-regress needs fewer gates); the two theatre tests renamed to what they check or strengthened.

## B2 — Rulings
- **The pre-flight `preflight_20261006T195405Z` stays PINNED** (record sha256 `a2804527…`): B1's changes add checks and tests and
  touch neither the training nor either evaluation path, which the pre-flight timed and reproduced. If a B1 change touches a timed
  path after all, the pre-flight is re-run and the packet says why.
- `docs/notes/` is the coordinator's: the untracked `docs/notes/p5_2b_g1/` in the task tree (the implementer's mutant specs, results
  and log — worth keeping) is committed as `docs/returns/P5.2b_evidence/` instead, and the packet points to it.

## B3 — Next
B1.1–B1.7 (B1.8 at the implementer's call), tests first, at most two source files per commit; the mutants PM4, MT2 and the
one-device maximum re-run and KILLED, plus one mutant per new refusal; hygiene and English; the whole suite (expect `DEFERRED` 110's
one failure in a worktree, named); the new and changed test files once in a depth-1 clone (F.1's command); the pre-flight pinned in
the driver → **"P5.2b C8 done"** → gate G1.1 (the coordinator re-runs PM1–PM5 and checks B1).

## B.1 — Added the same evening, after reviewer RA2's final report (two items its findings file did not yet carry)
1. **P5.2's registered per-seed rule for Q2 (D9) — the coordinator's G0 error.** `docs/plans/p5.2.md:633-634`: *"Both are reported
   with the per-seed ordering beside the pooled one (D9): a first place that reverses on a seed is reported as reversing."* The plan
   (§6) and Amendment A (A2, Q3) accepted a statement set without it. Required: Q2a and Q2b at the random tier reported with each
   training seed's ordering beside the pooled one — under both definitions, before and after — and a first place that reverses on a
   seed named as reversing. RA2 measured, from the raw P8.4b cells, that among the five arms other than IQL `dt_spatial` is first on
   seeds 101 / 303 / 404 / 505 and `dt_nomix` on seed 202 (by ≈ 0.69 under both definitions): if the corrected IQL is no longer
   first, the AFTER first place reverses on seed 202, and the artifact must say so. Tests: a synthetic case with a first place that
   reverses on one seed; the real BEFORE per-seed orderings against the test's own recomputation.
2. **Every refusal and branch the report can take, exercised (RA2 MINORs; its mutants M1–M4, M8, M9, M12 survived):** a
   T-statements case whose Q3c CI straddles zero (NOT RESOLVED) and one where `dt_nomix` is strictly lowest (Q2a, Q3a HELD); a test for
   each untested guard — `_corrected_rows`' `policy_source` check, the run manifest's per-file re-check, `check`'s refusal of canaries
   without a complete training, the completeness check's `training_rows` clause; and `write_run_manifest` REFUSES unless every stage
   is complete (a manual `manifest` call on a partial run must not freeze it as final). Each surviving mutant re-run and KILLED.

---

# ✅ AMENDMENT B.2 — 2026-10-07, gate G1.1 PASSED: the correction run is CLEARED at `4e849fa` — the author's token

## B.2.0 — Verified by the coordinator (by running commands; `docs/reviews/P5.2b-G1.md` §6)
1. **The five G1 mutants re-run at `4e849fa`**, committed in a throwaway worktree, every data gate open (unmutated baseline
   `120 passed`): **5 / 5 KILLED** — PM4, the G1 survivor, by the new test
   `test_b1_2_a_p8_4b_cell_whose_att_ours_is_not_p5_2s_att_horizon_is_refused`.
2. **The other three test files** with the gates open: `285 passed, 1 skipped` (the known `tests/test_tier_sweep.py:1447`).
3. **The pre-flight stays pinned, on the coordinator's own comparison:** of the thirteen functions the pre-flight timed or
   drove, eleven are byte-identical between its commit `a71a72c` and `4e849fa` (`ast` source segments); `train_stage` differs by
   the hyperparameter check placed BEFORE the training and two recorded fields (`hyperparameters`, `finished_utc`), `preflight`
   by the `pins` argument to the barrier. The driver pins `p5_2b_runs/preflight_20261006T195405Z/preflight.json` at
   `a280452734494479f6d2941b09ff12c16825155080404ee7ce84b95752c4bb4b`, which equals the file's digest on disk.
4. **B1–B1.7 and B.1 read in the code:** `assert_c1_note_means` compares all twelve random-tier means (six arms × two
   definitions) with the note under exact equality, reading each cell by P8.4b's own file name and refusing a cell that is not
   the one its name says, inside `_verify_inputs` — so `check` refuses before the token; `_per_seed_orderings` implements D9
   (each seed's levels by the pooled route, order, first and ties, P5.2's concordance with its tie rule; a pooled first that is
   not a seed's unique first is named as reversing); `close_late_stage` is write-once, refuses unless the training is complete
   and only its closing canary is missing, and re-entry with its own mark writes nothing.

## B.2.1 — Rulings on the packet's deviations
1. **The manifest gate stays in `manifest_stage`; it is NOT moved into `write_run_manifest`, and the seven test changes are NOT
   authorised.** The gate sits on the only production path — the driver's `ic manifest` runs the CLI, the CLI runs
   `manifest_stage`, and `write_run_manifest` has no other caller in the module — and the implementer's mutant of the CLI
   bypassing it was killed. Moving the gate would buy seven edits to committed tests and nothing else.
2. **B1.3's addendum accepted:** the training record is written once, before the closing canary exists, so the write-once
   `training_random_iql.late_close.json` is the honest form of "the record marks its seconds"; the report folds it into
   `training.seconds`, and P8.2's table quotes that block.
3. **B1.6(a) as built** (the two structural checks always; the digest where pins are passed) accepted.
4. **B1.8 as taken** (the report without the materialised draws; the two theatre tests renamed) accepted.

## B.2.2 — G2, the correction run (the author)
- **The run worktree** (the coordinator's, the branch pushed): `/home/filip/rltraffic-p52b-run`, detached at
  `4e849fa4` (the branch tip; the full sha is in the command below), clean.
- **Before the token:** mains power, the Windows power mode on **Best performance** (the run refuses before the token
  otherwise), a quiet machine, the implementer's and the coordinator's sessions idle for about an hour.
- **The token:** `touch /home/filip/rltraffic/output/p5_2b_runs/TOKEN_correction`.
- **The command**, in a tmux pane: `bash /home/filip/rltraffic-p52b-run/offline/campaigns/p5_2b_correction.sh <commit> 2>&1 |
  tee -i -a /home/filip/rltraffic/output/p5_2b_runs/correction_capture.txt; echo "DRIVER EXIT ${PIPESTATUS[0]}"`.
- **Expected:** about an hour (training ≈ 28 min, each evaluation path ≈ 16 min, the canaries, the manifest and the report),
  ending `DRIVER EXIT 0`. A refusal before the token costs nothing (set what it names, start again). A hang costs one timeout
  and is re-run by the driver, at most three attempts per evaluation stage. A FAILED run keeps `output/p5_2b/` and restarts
  where it stopped with a new token; the training is never run twice.

## B.2.3 — Next
G2 → **"P5.2b correction run done"** → G3: the coordinator reads the run from disk — the capture, the training record with its
canaries and hyperparameters, the five checkpoints re-hashed and their rows, the corrected cell's 500 episodes under both
definitions recomputed by a third route, the per-seed orderings, every statement before and after — then Amendment C; then C9
(the artifact by hand with its T-regress; `DEFERRED` 107's tests are already in; P8.2's table) and the packet → G5.

---

# ✅ AMENDMENT C — 2026-10-07, gate G3 PASSED: the correction run read from disk, every statement recomputed by a third route — THE DEFECT CHANGES NUMBERS, NOT VERDICTS

## C0 — What the coordinator verified (by running commands; `docs/notes/p5_2b_g3/`)
1. **The run `20261007T152900Z`** (the author's, 15:29–16:28 UTC, from the run worktree at `4e849fa`): no refusal; every stage at
   its first attempt (training 1,543 s, (i) ≈ 17 min, (ii) ≈ 17 min); `output/SHA256SUMS_p5_2b.txt` 516 / 516 `OK`; canaries
   0.733 s and 0.698 s, at speed, reproduced, mains + Best Performance at both; no late close (`training_random_iql.late_close.json`
   absent, as it should be).
2. **The training:** five seeds, 40,000 steps each on **1,152,000 rows** (3,200 streams; the declared 200 episodes), reward scale
   0.7429420505200595, the thirteen hyperparameters equal to the originals' records, the normalisation statistics identical to the
   original checkpoints', CUDA, one thread; the gradient loops 307.3 / 324.4 / 297.3 / 295.7 / 297.8 s (median 297.8 [295.7–324.4]);
   the five checkpoints re-hashed to the record's digests (5 / 5), each differing from its original, each carrying the `correction`
   provenance block.
3. **The two evaluations:** (i) 500 episodes on P5.2's protocol, its `model_provenance` digests the record's; (ii) 500 cells, each
   naming the corrected checkpoint, reproducing (i)'s `att_horizon` on 500 / 500.
4. **The third route** (`g3_recompute.py`, imports nothing from the project; the predicted order read as a literal from
   `tier_sweep.py`'s registered table): 24 levels, 64 statement checks, 8 corrected-cell checks, 500 episode equalities — **0
   problems.** One fact for the paper: the project's `dt_gate.mean_ci95` uses z = 1.96 (`dt_gate.py:218`); with the exact normal
   quantile the CI endpoints differ in the fifth significant figure and nothing else does. Every CI of this project is 1.96 × SE.
5. **The pinned inputs** (the P5.2 manifest, the declaration, the corpus manifest, the C1 note, the run manifest) at their digests,
   5 / 5; the caveat blocks complete.

## C1 — The result, under both definitions (`att_ours` = P5.2's registered metric; `att_engine` = the paper's primary, Rule R)
| | original cell (2× the declared data) | corrected cell (the declared data) |
|---|---|---|
| `att_ours`, mean [95 % CI], n = 500 | 190.96 [188.76, 193.17] | **198.48 [195.84, 201.12]** |
| `att_engine` | 183.83 [181.60, 186.05] | **191.39 [188.73, 194.06]** |
| per-seed means, `att_engine` (101 / 202 / 303 / 404 / 505) | 180.5 / 180.9 / 201.7 / 178.0 / 178.0 | 191.3 / 190.2 / 186.5 / 207.4 / 181.6 |

**The corrected IQL is ≈ 7.5 s slower under both definitions, and every registered statement's OUTCOME is unchanged:** Q1 15 / 19
HELD with IQL's entry still a miss (relative error 0.52 against the registered 414.99, was 0.54); the ranking `iql < dt_spatial <
dt_nomix < bc < bc_top10_perix < bc_top10` under both; Q2a FAILED (IQL first, `dt_nomix` predicted) — **IQL first on every training
seed, no reversal** (D9); Q2b 8 / 15 FAILED, IQL's pairs 1 of 5 concordant; Q3a rank 3 FAILED; Q3c `dt_nomix − iql` **+56.84
[+53.98, +59.71]** (`att_ours`) / **+56.78 [+53.90, +59.67]** (`att_engine`), was +64.36 / +64.35 — still resolves against the DT.

**For the paper:** the sentence that IQL leads the random tier stands, with the correction disclosed (the cell was first trained on
twice its declared data and re-trained on the declared 200 episodes; the registered tally rests on the registered condition); the
DT's shortfall to IQL at that tier is ≈ 57 s, not ≈ 64. `README.md`'s grid4x4 sentence is the coordinator's at the merge.

## C2 — Next: C9 (the implementer)
1. `docs/data/p5_2b_iql_correction.json` committed by hand, byte-identical to `output/p5_2b/artifacts/p5_2b_correction.json`, with
   the T-regress (the report stage, B1.8: no materialised draw needed).
2. P8.2's table per the plan's §9 — `DEFERRED` 107's tests are already in: the random-tier group of `grid4x4.iql` on the corrected
   checkpoints, the entry from the committed correction artifact with its `record_note` and `superseded` block, the authorised change
   to `test_t_sources_the_declared_absences_carry_their_reasons` under A2 Q8(c)'s three conditions, the rebuilt
   `docs/data/p8_2_compute.json` with every other row asserted equal.
3. The whole suite (`DEFERRED` 110's one failure in a worktree, named); the new and changed test files once in a depth-1 clone (F.1's
   command); `docs/returns/P5.2b.md` per §8, quoting A2 Q8(c) and this amendment → **"P5.2b done"** → G5.

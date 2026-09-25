# BRIEF_41 — P7.3c: the few-shot curve on grid4x4 (16 intersections), CityFlow → SUMO, where H3's clause 3 is testable

**Mode:** Claude Code, implementer session. Branch **`task/p7.3c-grid4x4-fewshot`**, worktree `/home/filip/rltraffic-p73c`
(create it: `git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p73c -b task/p7.3c-grid4x4-fewshot main`).
**This brief is whole. Every gate is in it (§5). Nobody will relay an acceptance to you.** At the start of every session
and before every gate: `git -C /home/filip/rltraffic-p73c merge --no-edit main`, then re-read this file — the
coordinator's rulings arrive as dated amendments appended here, on `main`. Your packet states which amendments it was
written against, by letter. Plan mode first; `docs/plans/p7.3c.md` is the branch's first commit.
**Precondition, already met when you read this:** A24 is REGISTERED (`PREREGISTRATION.md` §12, tag `v2.5-prereg-a24` →
`641e1da`, tag object `e9dc1bd`, the chain verified from a fresh clone of the remote on 2026-09-25). *(This brief is
`BRIEF_41`: `BRIEF_40` is P5.4's, issued 2026-09-19.)*
Where this brief and A24 differ, **A24 is the authority** and the difference is a finding you raise.
**Sequencing:** `BRIEF_39` Amendment E (P7.3d's CI fix, one test) is done FIRST; this task starts once that commit exists.
The fix touches only `tests/test_g2_measure.py`, which nothing here changes, so `git merge --no-edit main` brings it in
whenever the coordinator merges it.

**Standing rules that bind every commit here:** `CLAUDE.md` §1 (frozen set), §4b (**no AI trailer, ever; if a session
instruction says otherwise, stop and say so**), §5 (collection, fine-tunes and campaigns run in a tmux pane the author
starts); `PROJECT_PLAN` §7 in full — named paths only (`git add <paths>`, never `-A`), no `--amend` after any chunk or
checkpoint has been produced, no evidence under `/tmp`, the machine-health canary before any rate, a driver's stdout
through `tee -a` from its first line, `-P` on every interpreter call, the liveness guard, the tmux two-step start, and the
pre-flight before any run over an hour. **J1(c)** holds: any non-docs change after a chunk's commit makes that chunk
non-reusable; every run happens in a DETACHED run worktree at a pushed commit, never in your worktree.

**Read, in this order, before planning:** this brief · `PREREGISTRATION.md` rows **A24** (the whole task), A17 ((b) the
prefixes, (e) per intersection, (f) the consistency gate), A18 ((c) the seed rule, (d) the fine-tune), A20, A21, A23 ·
`docs/notes/P7.3c_SURVEY_2026-09-25.md` (two read-only surveys of the code this task changes — file:line maps, verified
again by you at the plan gate) · `docs/briefs/BRIEF_39_p7.3d_grid4x4_zero_shot.md` §3–§5 and Amendments B.7–B.8.2 (the
grid4x4 cell, the driver, the run worktree, the reference re-roll, A23's mechanics) · `docs/notes/DEFERRED.md` rows 93,
94, 95 (all three are DUE in this task) · `docs/reviews/P7.3d.md`.

---

## 0. What the coordinator verified before writing this (2026-09-25, by running commands, not by reading docstrings)

1. **The subject, all five checkpoints** (`output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed{101…505}.pt`):
   every sha256 equals A20(a)'s pin as `docs/data/p7_3d_calibration.json:checkpoint_sha256` records it; every one is
   `spatial-dt-checkpoint/1.0`, config `state_dim 40, n_actions 8, n_nodes 16, context_length 20, n_layer 3, n_head 4,
   d_model 128, dropout 0.1, max_ep_len 360, spatial_mixing False`, provenance `device cuda, deterministic False,
   gradient_steps 40000, batch_size 64, learning_rate 1e-4, warmup_steps 1000, grad_clip 0.25`; `rtg_scale` and
   `target_rtg` are dicts keyed by the sixteen CityFlow intersection ids. (Loaded with `weights_only=True` and ONE
   allowlisted class, `torch.torch_version.TorchVersion`; do the same.)
2. **P5.2's training speed for this subject:** 81 · 82 · 81 min per 40,000 steps (`docs/returns/P5.2.md` D-2, committed),
   ≈ 122 ms/step on this machine's CUDA (an RTX 5080 Laptop GPU, `torch.cuda.is_available()` True today). A 4,000-step
   fine-tune is therefore ≈ 8 min — not A18(d)'s ≈ 20 s, which was hz1x1's one-head model at 5.1 ms/step.
3. **The calibration artifact** (`docs/data/p7_3d_calibration.json`, sha256 `3e9df8ee…`): for every one of the sixteen
   intersections, `budgets.k5` (draws 201–205), `k20` (201–220), `k100` (201–300), each with `target`, `rule B`,
   `statistic mean` and an in-support record; roles `k100: registered_prompt`, `k5` and `k20: recorded_not_evaluated`;
   `probe_returns.sumo[draw][intersection]` for all 100 × 16 — **the reference for A17(f)'s gate**. On all sixteen
   intersections at every k, the Rule B target lies ABOVE the best MaxPressure return among the k episodes themselves
   (by 84.3–128.6 at k = 5, 71.8–114.5 at k = 20, 48.6–104.2 at k = 100; A24(e)) — the fine-tuned arms are prompted above
   their own target-domain demonstrations, and the in-support diagnostic reports both ranges.
4. **The SUMO probe's 100 episodes** (`output/p7_3d/calibration/probe_sumo_draw_*.json`): 100/100 `n_teleports 0` (read
   sparsely — `DEFERRED` 93; the coordinator's dense re-roll of 2026-09-24 was clean on all of them), `vehicle_types_seen
   ['cf_parity']`, `two_routes_agree True`, `time_to_teleport_option '-1'`; mean wall 29.9 s per episode run alone.
5. **The parity files for draws 201–300:** 100/100 directories under `scenarios/draws/cityflow_grid4x4/draw_0NNN/parity/`
   with `noteleport.sumocfg`, `routes.rou.xml`, `provenance.json`.
6. **The zero-shot artifact** (`docs/data/p7_3d_grid4x4.json`, sha256 `c63c371f…`): 700 per-cell records carrying
   `e_sumo`, `att_env`, `rho_e_sumo`, `rho_att_env`, `seed`, `draw_id`, `arm`, `kind` and 50 more fields; `rho.by_draw`
   for 100 draws; `collisions` with A23's two events (draws 1020 and 1042); `h3.clauses` recording ρ̄₀ = 0.8854642679270011
   (`E_sumo`) and 0.8854600302820783 (`att_env`), from which A24 fixes G and G_att. Exactly SIX of the record's fields are
   bookkeeping — `git_commit`, `stage`, `seconds`, `canary_seconds`, `checkpoint` (an absolute path) and
   `sha256_checked_against` (a provenance note); A24(c)'s reproduction compares every other one. **ρ₀ and the reproduction's
   reference are read from this committed file, never from P7.3d's chunks.**
7. **P7.3d's measured cell costs at 12 workers** (from its 700 chunks): DT cell 119.0 s mean (max 490.6), fixed-time 57.9 s,
   MaxPressure 50.5 s; 11.9× effective parallelism. This task's 4,700 cells project to ≈ 12.8 h (stage 1 ≈ 1.6 h, stage 2
   ≈ 4.2 h, stage 3 ≈ 6.9 h); its thirty trainings to ≈ 5.6 h of GPU if run one at a time.
8. **What does NOT exist** (the surveys, re-checked where it matters): no SUMO grid4x4 trajectory corpus
   (`datasets_sumo_v11/` holds hz1x1's two only); no warm-start trainer (`dt_gate.train_dt` builds a fresh one-head model;
   `tier_sweep.train_tier_dt` builds a fresh model and fits statistics); `offline/collect.py` hard-codes hz1x1's alignment
   (`declared_alignment()`, survey S2-1); the evaluation path pins P7.3d's subject, its `b_mean_k100` arm and 40,000-step
   checkpoints (S2-3); no paired DT-versus-DT ρ contrast exists (S2-4).

---

## 1. Why this task exists

H3's third clause — *"closes substantially by k = 100"* — was void on hz1x1 (A19(a)) because the zero-shot model beat
MaxPressure there. On grid4x4 it did not: ρ̄₀ = +0.8855 [+0.8736, +0.8973], so a gap G = 0.114536 to MaxPressure exists,
and the P7.3d merge review (A-2) found grid4x4 to be the ONE scenario on which clause 3 is testable. A24 declared the test
and fixed its criterion before any few-shot number exists: fine-tune the zero-shot subject on k ∈ {5, 20, 100} SUMO
MaxPressure episodes (A18(d)), evaluate it on the 100 held-out draws, and decide clause 3 by the paired Δ₁₀₀ against
G/2 — after re-producing P7.3d's zero-shot point bit for bit at this task's commit, and beside two controls that make the
verdict interpretable: the zero-shot model under the k = 5 and k = 20 prompts (what the prompt alone does) and a from-scratch
model trained exactly like the fine-tune (what the transfer contributes). The number this task produces is one verdict, its
effect size and CI, and the curve's shape around it — exploratory under §2, decided by a threshold written before the data.

---

## 2. Scope fence — what NOT to build

- **No hz1x1 few-shot** (A24(h)): hz1x1's corpora and subjects are not touched; nothing here fine-tunes an hz1x1 model.
- **No k = 200 retrain anchor on grid4x4, no `naive` arm, no `random` anchor, no `b_max`, no Rule A, no second subject**
  (A24(g)). None of these cells may appear in any declaration. The k = 5 and k = 20 targets prompt ONLY `ft_k5`, `ft_k20`,
  `zs_k5` and `zs_k20` in P7.3c's stages (A24(a) amends A20(b) for exactly that).
- **No training runs twice** (A24(b)): every checkpoint is written once, its digest is committed before the evaluation
  token, and a re-run is allowed only for an infrastructure failure before the checkpoint exists — counted and reported.
- **No evaluation of any fine-tuned checkpoint outside the campaign's token.** Timing runs are FENCED (§3 C4) and use a
  configuration A24 does not register; their checkpoints are never loaded by an evaluation path.
- **Nothing under `output/p7_3d/` is edited, moved or deleted.** P7.3d's committed artifacts are read, digest-verified,
  never rewritten; P7.3d's declarations (`grid4x4_cells`, its stage names, its arm set) are not changed — the few-shot
  stage is NEW, beside them.
- **No change to the frozen set, no new dependency, no new absolute path in code.** The model class
  (`agent/SpatialDTAgent.py`) is reused as it is; if it cannot be warm-started without a change there, STOP and say so.
- **hz1x1 stays byte-identical** at every commit that touches shared code (T-regress).

---

## 3. The commits, in order — each ≈ 2 source files plus tests; three STAGES, each ending at a run the author starts

### STAGE A — the corpus (can run as soon as its gates pass; nothing else here depends on later stages)

**C0 — `docs/plans/p7.3c.md`** (gate G0). Your plan: assumptions with confidence, every seam you will touch with its
file:line (re-verify the survey's map — do not copy it), the fine-tune's data route, the evaluation's new branches, the
tests of §4 mapped to commits, and every open question. The coordinator rules in Amendment A.

**C1 — `offline/collect.py` + `offline/transfer_calibration.py`: the grid4x4 SUMO corpus door and A17(f)'s gate.**
(i) The alignment is chosen by scenario — `alignment_for_scenario_key(scenario_key)` (`offline/aligned_env.py`) in place
of the hard-coded `declared_alignment()` — with hz1x1 resolving to exactly today's object (T-regress (a)). (ii) MaxPressure
acts on exactly what the probe gave it (`run_sumo_probe_per_intersection`'s route: the raw env info), and the logger
records the ALIGNED view (A16's `align_info`), in the CityFlow scenario key `cityflow_grid4x4` and the CityFlow
intersection ids, so the episodes are the joint windows' input without re-keying. (iii) **`DEFERRED` 93's fix:** teleports
AND collisions counted on EVERY simulated second (the env's per-step hook or SUMO's per-run statistics — say which in the
plan); an episode with either is refused, nothing written. (iv) **A17(f)'s gate per intersection:** a grid4x4 twin of
`assert_logged_corpus_matches_probe`, keyed by intersection ID (never by index), comparing every logged episode's
per-intersection return to `p7_3d_calibration.json:probe_returns.sumo[draw][ix]` under `==`, on all 100 × 16, after
verifying that artifact's sha256 (`3e9df8ee…`); any mismatch refuses the corpus and says which draw and intersection.
(v) `assert_probe_draws_disjoint` against the subject's training draws and the held-out pool, before the first episode.
The corpus is C6 v1.1, one directory `datasets_sumo_v11/grid4x4_sumo_maxpressure/` with a `SHA256SUMS`, draws 201–300.

**C2 — `offline/campaigns/p7_3c_corpus.sh`: the collection driver** (P7.3d's probe driver's shape): canary before the
token, `record-canary` after, the run worktree, `-P`, liveness, `tee -a`, `set -euo pipefail`, `WORKERS` from P7.3d's G2
measurement (12) unless the plan argues otherwise, the A17(f) gate run as its LAST stage, and a refusal to start over an
existing non-empty corpus directory (the filesystem-mutation barrier: every write and every delete after all validation).

### STAGE B — the fine-tunes (starts when the corpus is verified, G4)

**C3 — `offline/few_shot.py` (new) + tests: the warm-start fine-tune for the joint model, A18(d) as A24(b) reads it.**
- Load the source checkpoint (digest checked against A20(a)'s pin), rebuild `SpatialDecisionTransformer` from the
  PAYLOAD's config — `max_ep_len` included, never derived from the data — and `load_state_dict(strict=True)`.
- Windows from the FIRST k episodes of the corpus by draw id (k = 5 → 201–205, 20 → 201–220, 100 → 201–300, read from the
  manifests; any other set refused), through `TrajectoryWindowDataset` with the checkpoint's OWN statistics passed in
  (the `stats=` hook, survey S1-5(3)) — never refit — then `build_joint_index` / `stack_joint` in the checkpoint's node
  order (`payload["intersection_ids"]`), refusing a corpus whose ids differ; RTG divided per node by the checkpoint's
  frozen `rtg_scale[ix]`.
- A FRESH AdamW, lr 1e-4, weight decay 1e-4, gradient clip 0.25 — imported from where `tier_sweep` imports them AND
  asserted equal to the checkpoint's recorded provenance; warm-up `min(1000, max(1, B // 2))`, constant after; batch 64
  joint instants sampled uniformly with replacement by `np.random.default_rng(seed)`; `Utils.seed_everything(seed)`;
  exactly B optimizer steps.
- The saved payload: the SOURCE's `config`, `stats`, `rtg_scale`, `intersection_ids`, `spatial_mask` unchanged (bitwise);
  `model` the fine-tuned state dict; `target_rtg` = `p7_3d_calibration.json:per_intersection[ix].budgets["k{k}"].target`
  per intersection (A24(b): same-k Rule B; k5/k20 admitted for these checkpoints only); a new `format_version`
  (`few-shot-checkpoint/1.0`); provenance naming the source path and sha256, k, the draw ids, the corpus `SHA256SUMS`
  digest, B, seed, the optimizer values, warm-up, device, `deterministic False`, the git commit and runtime.
- **The from-scratch path (A24(c)(ii), `scratch_k100`):** the SAME function with one switch — the weights initialised under
  `Utils.seed_everything(seed)` instead of loaded — and everything else identical: the source's config, frozen statistics,
  `rtg_scale`, node order and mask, the k = 100 corpus, B = 4,000, the recipe, the sampler, the k = 100 target. Its
  provenance names the switch; nothing else may differ (T-scratch).
- No evaluation code here. No selection: the payload after exactly B steps is the only one written.

**C4 — `offline/campaigns/p7_3c_finetune.sh`: the training driver.** THIRTY runs, each exactly once (A24(b)): `ft_k5`,
`ft_k20`, `ft_k100` × seeds 101/202/303/404/505 at B = 4,000 (15); `scratch_k100` × the five seeds at B = 4,000 (5);
`ft_k100` × the five seeds at B = 1,000 and at B = 16,000 (10) — in the subject's regime (CUDA, non-deterministic, as
P5.2's), from the run worktree, token, capture, a `SHA256SUMS_p7_3c_finetune.txt` manifest, a refusal to overwrite ANY
existing checkpoint, and a resume that skips a run whose checkpoint exists and re-runs one only if its checkpoint was never
written (the infrastructure-failure case, counted in the capture). **Concurrency is MEASURED, not assumed (gate G5):** FENCED
timing runs first — k = 5, seed 101, **B = 400** (a configuration A24 does not register), written under a `fenced_timing/`
directory the evaluation path refuses — alone, then 2 and 3 concurrent, plus ONE same-seed repeat of the single run to measure
whether its digest reproduces on this GPU; ms/step, GPU memory, the concurrent slowdown and the repeat's verdict recorded; the
driver's concurrency set from that file.

### STAGE C — the evaluation (starts when the fine-tunes are verified, G7)

**C5 — `offline/transfer_curve.py`: P7.3c's three stages, BESIDE P7.3d's.** New stage names (e.g. `p7_3c_reproduce`,
`p7_3c_primary`, `p7_3c_controls`) whose declarations are exactly A24(c)'s 4,700 cells. **Stage 1 (700):** P7.3d's declared
set re-rolled — the zero-shot subject under `b_mean_k100` × five seeds × draws 1000–1099, and `fixedtime` and `maxpressure`
× the same draws. **Stage 2 (1,500):** `ft_k5`, `ft_k20`, `ft_k100` × five seeds × the draws. **Stage 3 (2,500):** `zs_k5`,
`zs_k20` × five seeds (the zero-shot checkpoints under the k = 5 / k = 20 targets); `scratch_k100` × five seeds; `ft_k100`
at B = 1,000 and at B = 16,000 × five seeds — all × the draws. The trained subjects resolve to the training manifest's
digests (as committed on `main`, G7); the arm `b_mean_k{k}` is admitted for k = 5 and 20 ONLY on P7.3c's stages and ONLY
for the four arms that A24(a) names (survey S2-3: today's role check refuses `recorded_not_evaluated` — the branch admits it
there and nowhere else); the 40,000-step check (`spatial_mixing.py:1118–1140`) gains a branch that accepts a trained
checkpoint whose provenance names a pinned 40,000-step source (or, for `scratch_k100`, the from-scratch switch) and
B ∈ {1,000, 4,000, 16,000}; A23's pins extend to the new arms. **The stage-1 gate (A24(c)):** a comparator that checks EVERY
one of the 700 re-rolled chunks against `p7_3d_grid4x4.json:cells` (sha256 verified first) on every field the record carries
EXCEPT the six bookkeeping fields `git_commit`, `stage`, `seconds`, `canary_seconds`, `checkpoint`, `sha256_checked_against`
— `==`, and ANY mismatch stops the campaign before stage 2's first cell.

**C6 — `offline/transfer_curve.py` (report) + tests: P7.3c's report body, A24(c)–(e) exactly.** In this order, each a
refusal before any write: the committed zero-shot artifact's sha256 == `c63c371f…`; ALL 700 stage-1 chunks reproduce its
records under the six-field rule; every declared cell of stages 2 and 3 present and complete (A24(c): no estimator on a
partial set); only then ρ per cell (§3.4, the same draw's anchors), ρ̄ per arm (per-draw five-seed means, `mean_ci95`,
n = 100), **Δ_k paired BY DRAW ID** against ρ₀ from the artifact, **the clause-3 verdict as A24(d)'s partition** — (i) lo > 0
and Δ₁₀₀ ≥ G/2, (ii) lo > 0 and Δ₁₀₀ < G/2, (iii) lo ≤ 0 ≤ hi, (iv) hi < 0 — with G = `0.11453573207299894` and
G_att = `0.11453996971792169` as registered constants, never recomputed; the closure fraction with its CI; the `att_env`
computation beside it; the five per-seed Δ₁₀₀ with CIs, their between-seed SD and the count meeting each condition; A23(d)'s
robustness recomputation and whether its outcome differs; the adaptation effects ρ̄(`ft_k`) − ρ̄(`zs_k`); Δ_transfer =
ρ̄(`ft_k100`) − ρ̄(`scratch_k100`); A24(e)'s five adjacent-step expectations as held or refuted; the budget secondary; and the
in-support diagnostic of every trained arm's targets against BOTH the source's range and the fine-tune corpus's. **Due here, because this commit changes the grid4x4 report path:** `DEFERRED` 95 (a
T-regress for `docs/data/p7_3d_grid4x4.json`, byte-identical through `report` with A3's two substitutions) and `DEFERRED`
94 (the second-event-in-an-A23-cell variant, and its mutant killed).

**C7 — `offline/campaigns/p7_3c_grid4x4.sh`: the campaign driver** — P7.3d's driver's shape: ONE token for the three stages,
run in order; the stage-1 gate between stage 1 and stage 2, automatic, with no human step in it; stage 3 after stage 2 whatever
stage 2 shows; the canary, the run worktree, `-P`, liveness, `tee -a`, `WORKERS` 12 and P7.3d's G2 RSS budget (the
architecture is the same; a different measurement is a finding); the driver prints NO outcome of any cell.

**C8 — the packet and the artifacts** (after G10): `docs/data/p7_3c_grid4x4.json` (the report), `docs/data/p7_3c_finetune.json`
(the thirty checkpoints' digests and provenance, and the timing record), `docs/returns/P7.3c.md` per §7.

---

## 4. Tests — first, red for their own reasons, each named mutation executed and pasted

SUMO-, checkpoint-, corpus- and GPU-gated tests carry `skipif` predicates that **name the artifact they consume**. At most
**four** grid4x4 SUMO episodes in the whole suite. Critical quantities are recomputed by a DIFFERENT route than the code
under test (raw `np.cumsum` on stored rewards, raw arrays from the `.npz`, the checkpoint read with `torch.load`, never
through the function being tested). `scripts/check_test_hygiene.sh` on every test file; the English sweep on everything.

- **T-warm (load-bearing, C3).** Built and loaded, BEFORE any step, the fine-tune model's state dict equals the source
  checkpoint's tensor by tensor under `torch.equal`, and `max_ep_len` is the payload's 360. *Mutations:* skip
  `load_state_dict` → dies; `max_ep_len` from the data of a corpus truncated to 300 decisions → dies.
- **T-frozen (load-bearing, C3).** The normalised states fed to the model equal `(raw − state_mean) / state_std` computed
  from the corpus `.npz` with the CHECKPOINT's arrays, per intersection, bitwise; the RTG inputs equal the raw returns-to-go
  (`np.cumsum` route on the stored rewards) divided by the checkpoint's `rtg_scale[ix]`. *Mutations:* `stats=None` (refit on
  the corpus) → dies; `rtg_scale` refit from the corpus → dies.
- **T-prefix (C3).** k = 5, 20, 100 read exactly draws 201–205, 201–220, 201–300 from the manifests; a corpus missing draw
  203 is refused naming it; draw 206 present does not enter k = 5. *Mutation:* `range(201, 201 + k + 1)` → dies.
- **T-recipe (C3).** Exactly B optimizer steps (counted, not inferred); warm-up `min(1000, B // 2)` — 500 at B = 1,000,
  1,000 at B = 4,000 and 16,000 — and the lr at steps 0, warm-up − 1 and B − 1 as the recipe gives; the optimizer state
  empty before step 0; the three constants asserted equal to the checkpoint's recorded provenance. *Mutations:* B − 1
  steps → dies; warm-up 1,000 at B = 1,000 → dies.
- **T-order (C3).** A corpus whose intersection ids come in a different order is re-keyed to the checkpoint's order — or
  refused, as the plan decides — and never trained in the corpus's order; one id absent → refused. *Mutation:* windows in
  corpus order → dies.
- **T-payload (C3).** The saved payload's `config`, `stats`, `rtg_scale`, `intersection_ids`, `spatial_mask` equal the
  source's bitwise; `target_rtg[ix]` equals the calibration artifact's `budgets["k{k}"]` target for EVERY ix, read by the
  test from the JSON; provenance carries every field of §3 C3. *Mutation:* k = 100's targets written for k = 5 → dies.
- **T-cpu-determinism (C3).** Two fine-tunes with the same seed, B = 20, on CPU, give identical state dicts; different
  seeds do not. (CUDA is non-deterministic by the registered regime; this pins the code, not the device.)
- **T-scratch (load-bearing, C3).** The from-scratch run and the fine-tune run of the same seed differ ONLY in the
  initialisation: at step 0 the scratch model's state dict is NOT the source's and equals a fresh
  `SpatialDecisionTransformer` built under the same seed; the windows, statistics, `rtg_scale`, target, sampler draws and
  recipe are identical between the two (compared, not assumed); the payloads differ only in `model` and the provenance's
  switch. *Mutations:* the scratch path still loading the source weights → dies; the scratch path refitting statistics → dies.
- **T-regress (load-bearing, C1 and C6).** (a) With the alignment chosen by scenario, hz1x1's resolves to the same object
  as today's `declared_alignment()`, and one hz1x1 SUMO draw collected through the changed door (draw 201) produces arrays
  equal under `==` to `datasets_sumo_v11/hz1x1_sumo_maxpressure`'s episode for that draw; (b) `DEFERRED` 95 —
  `docs/data/p7_3d_grid4x4.json` regenerates byte-identically through `report` with A3's two substitutions, and the three
  hz1x1 T-regress artifacts still do. *Mutations:* the scenario lookup returning hz1x1's alignment for every key → the
  grid4x4 T-17f below dies; one aggregate of `_grid4x4_estimates` perturbed → (b) dies.
- **T-17f (SUMO, grid4x4-gated, C1).** ONE grid4x4 draw (201) collected through the door: its per-intersection returns
  equal the calibration artifact's under `==` on all sixteen, zero teleports and zero collisions counted every second.
  *Mutations:* the gate keyed by index with the id list reversed → dies; the dense counter read once per decision with a
  collision injected on an odd second (a fixture) → dies.
- **T-verdict (load-bearing, C6).** Synthetic per-draw differences driving each of A24(d)'s four outcomes and every
  boundary of the partition: Δ₁₀₀ exactly G/2 with lo > 0 → (i); lo exactly 0.0 → (iii), never (i) or (ii); hi exactly 0.0 →
  (iii), never (iv); hi < 0 → (iv). G and G_att read from the registered constants. *Mutations:* `>=` → `>` on G/2 → the
  boundary case dies; `lo > 0` → `lo >= 0` → dies; `hi < 0` → `hi <= 0` → dies; G recomputed from the data instead of the
  constant → a fixture with a different ρ₀ mean dies.
- **T-stage1 (load-bearing, C5).** The reproduction comparator on committed-record fixtures: a chunk differing ONLY in the
  six bookkeeping fields passes; a chunk differing in ANY other field the record carries — one per field class: a
  measurement (`e_sumo`), a counter (`n_teleports`), an identity (`checkpoint_sha256`), a seed (`engine_seed_drawn`), a
  per-intersection list (`local_return`) — refuses and names the field; a chunk MISSING a field the record carries refuses;
  one mismatching chunk among 700 stops stage 2 from starting (the driver's gate, executed on a fixture). *Mutations:* `stage`
  compared → the bookkeeping-only chunk dies; the field loop over the CHUNK's keys instead of the RECORD's → the missing-field
  case dies.
- **T-perseed (C6).** The five per-seed Δ₁₀₀ recomputed by the test from the per-cell fixture by a different route (a
  per-seed mean over draws of cell-level ρ differences) equal the report's; the counts of seeds meeting each condition are
  exact on a fixture built to put exactly three seeds above G/2.
- **T-pairing (C6).** The arm's per-draw series permuted by draw id changes Δ; pairing is by draw ID, never by position;
  a draw present in one series and absent in the other is refused. *Mutation:* zip by position → dies.
- **T-report (C6).** Refusals precede every write including the last: the zero-shot artifact at another digest; ONE stage-1
  chunk not reproducing; ONE declared stage-2 or stage-3 cell absent (no estimator on a partial set); a trained-arm chunk at a
  checkpoint digest not in the training manifest; a chunk whose stage is P7.3d's; a work directory without `canary.json`.
  *Mutation:* the write moved above the last refusal → dies.
- **T-driver (C2, C4, C7).** Comment-free text assertions: canary before token, `record-canary` after, `-P` on every
  interpreter call, `tee -a`, `set -euo pipefail`, the refusal over a non-empty output directory, the fenced timing
  directory refused by the evaluation path. ⚠️ **Two harness traps, both hit on 2026-09-19:** a mutant worktree with an
  uncommitted edit is a DIRTY tree, and the driver's dirty-tree refusal fires first; and a shell whose command line
  carries the literal `offline.transfer_curve` matches the liveness `pgrep` (exit 3). Commit mutants in a throwaway
  worktree and invoke pytest from a script file whose text does not carry the pattern.

---

## 5. Gates, in order — who runs each, what it checks, what stops the task, and how you learn its result

| # | Gate | Runs it | Checks | Stops the task if | You learn it by |
|---|---|---|---|---|---|
| G0 | Plan gate | coordinator, from `docs/plans/p7.3c.md` on the branch | assumptions, the seams re-verified, the data route, the fine-tune's frozen quantities | a load-bearing assumption is wrong | Amendment A on `main` |
| G1 | Corpus door review | coordinator + one reviewer (≤ 15 min, findings file) on C1–C2, mutations re-run | the alignment choice, the dense counter, A17(f) by id, the barrier | a defect that could admit a wrong episode | Amendment B |
| G2 | Token: the corpus | **author — channel (a)** | — | — | the token file |
| G3 | Corpus verified | coordinator, from disk, capture first | 100/100 episodes, A17(f) 1,600/1,600 under `==`, zero teleports and collisions dense, `SHA256SUMS` clean, manifests v1.1 | any mismatch → channel (c); the corpus is not used | Amendment C |
| G4 | Training review | coordinator + one reviewer on C3–C4, T-warm / T-frozen / T-scratch mutants re-run by the coordinator | warm start real, statistics frozen, prefixes, recipe, the scratch switch, run-once | a defect | Amendment D |
| G5 | Fenced timing | you, in tmux | ms/step alone and at 2 and 3 concurrent, GPU memory, the same-seed repeat's digest | the GPU cannot hold one run → `BLOCKED` with the numbers | you have the numbers; they go in the driver header, the plan and the packet |
| G6 | Token: the thirty trainings | **author — channel (a)** | — | — | the token file |
| G7 | Trainings verified and PINNED | coordinator, from disk; commits the thirty digests on `main` (A24(b): before the evaluation token) | 30 checkpoints, manifest clean, payloads' frozen parts bitwise equal to the sources', targets per k, step counts, the scratch switch, re-runs counted | any frozen part differs → channel (c) | Amendment E |
| G8 | Pre-flight | a reviewer (≤ 15 min) on C5–C7, coordinator rules | the 4,700-cell declaration, the admission branch, the stage-1 gate, destruction and resume paths | a destruction path, or a cell outside A24(c) | Amendment F |
| G9 | Token: the campaign, three stages | **author — channel (a)** | — | — | the token file |
| G10 | Read | coordinator, from disk, capture first, in A24's order: the stage-1 reproduction FIRST (700/700 under the six-field rule), then the verdict recomputed by an independent route, then the controls | 4,700/4,700; A23's rule; the verdict | stage 1 fails → the driver has already stopped; channel (c), no few-shot number read | Amendment G |
| G11 | Packet | you (C8) → **"P7.3c done", channel (d)** | §7 below | — | — |
| G12 | Merge review | coordinator spawns one (two mandates: code and mutations; every number recomputed from the raw chunks) | — | a blocker | the merge, §6's box ticked |

---

## 6. Definition of Done
- [ ] C0–C8 delivered in order, each its own commit with named paths; no frozen file touched; no new dependency; no new
      absolute path; hz1x1 byte-identical throughout.
- [ ] The corpus: 100/100, A17(f) 1,600/1,600, zero events counted densely; `DEFERRED` 93 closed by C1.
- [ ] Thirty trained checkpoints (15 fine-tunes, 5 from scratch, 10 budget runs), each written once, with their manifest,
      the fenced timing record and the digests pinned on `main` before the evaluation token; the frozen parts bitwise equal to
      the sources'.
- [ ] Every test of §4 red first then green; every named mutation executed and pasted; `DEFERRED` 94 and 95 closed by C6;
      hygiene and the English sweep run.
- [ ] The campaign complete under its token — stage 1 reproducing 700/700, stages 2 and 3 complete — the capture from the
      first line, the canaries on disk, `docs/data/p7_3c_grid4x4.json` and `docs/data/p7_3c_finetune.json` committed and
      regenerating byte-identically.
- [ ] `docs/returns/P7.3c.md` per §7, then "P7.3c done".

## 7. Return Packet
`docs/returns/TEMPLATE.md`, plus: G3's corpus verification with per-draw counts; G5's timing table with canary and date;
the thirty checkpoints' digests and provenance, and every training re-run with its cause; stage 1's reproduction, 700/700;
**the registered numbers of A24(d) in its words** — ρ̄ per arm on `E_sumo` with CIs and per-seed means, Δ_k with CIs, **the
clause-3 verdict as one of A24(d)'s four outcomes, in its labelled sentence, reported, not interpreted**, the closure fraction
with its CI, the same on `att_env` with G_att beside it, the five per-seed Δ₁₀₀ and their SD, A23(d)'s robustness line;
A24(e)'s adaptation effects, Δ_transfer and the five adjacent-step expectations as held or refuted; the budget secondary
apart; the per-intersection in-support position of each trained arm's targets against BOTH the source's training range and
the fine-tune corpus's (diagnostic, never selecting); the
driver captures of every run including any refused start; where each run ran and at what commit; the amendments written
against, by letter; the AI-assistance record's four lines; and one paragraph on what the paper's C3 section will assume
about this curve.

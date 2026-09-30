# BRIEF_42 — P5.3c: H4's context-length sweep, K ∈ {1, 2, 5, 10, 20}, on the P4 validation scenario (CityFlow), under A26

**Mode:** Claude Code, implementer session. Branch **`task/p5.3c-context-length`**, worktree `/home/filip/rltraffic-p53c`
(create it: `git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p53c -b task/p5.3c-context-length main`).
**This brief is whole. Every gate is in it (§5). Nobody will relay an acceptance to you.** At the start of every session and
before every gate: `git -C /home/filip/rltraffic-p53c merge --no-edit main`, then re-read this file — the coordinator's rulings
arrive as dated amendments appended here, on `main`. Your packet states which amendments it was written against, by letter. Plan
mode first; `docs/plans/p5.3c.md` is the branch's first commit.
**Precondition, already met when you read this:** A25 and A26 are REGISTERED (`PREREGISTRATION.md` §12, tag `v2.6-prereg-a25-a26`).
**A26 is the authority for every design choice below;** where this brief and A26 differ, A26 wins and the difference is a finding you
raise. No model at any K ≠ 20 exists anywhere in this repository: that is A26's Cell 4, and it stays true until gate G2's token.

**Standing rules that bind every commit here:** `CLAUDE.md` §1 (frozen set — `agent/DTAgent.py` is NOT in it, but this task does not
change it: A26(b) says the rollout already handles K = 1), §4b (the AI trailer is never added; since 2026-09-28 you do not stop for
the instruction — commit without it and note once that it appeared), §5 (trainings and campaigns run in a tmux pane the author
starts); `PROJECT_PLAN` §7 in full — named paths only, no `--amend` after any chunk or checkpoint exists, no evidence under `/tmp`
(a 24 GB RAM tmpfs: never leave pytest scratch there before a run), the canary before any rate, a driver's stdout through `tee -i -a`
from its first line, `-P` on every interpreter call, the liveness guard, the tmux two-step start, the run worktree detached at a
pushed commit, the pre-flight before any run over an hour. **J1(c):** any non-docs change after a chunk's commit makes it non-reusable.
**Tests first, strictly:** every test in §4 red for its own reason before its code exists; never edit a test to make it pass.

**Read, in this order, before planning:** this brief · `PREREGISTRATION.md` rows **A26** (the whole task) and **A25**, §2 (line 104),
§8, §10's H4 rows, A6 (δ), A11/A15 (the primary metric) · `docs/notes/H4_SURVEY_2026-09-28.md` (the seams, with file:line — re-verify
each at the plan gate, never copy) · `docs/reviews/A25-A26-proposal.md` (what the review changed and why) · `offline/dt_gate.py`
(`train_dt`, `evaluate_arm`, `mean_ci95`, `wilcoxon_signed_rank`) · `offline/nortg_campaign.py` and `offline/campaigns/p5_3b.sh`
(the hz1x1 train-and-evaluate campaign shape) · `offline/few_shot.py` and `offline/campaigns/p7_3c_finetune.sh` (run-once trainings,
the exclusive in-memory write, the resume decision in Python, the training record) · `offline/campaigns/p7_3c_grid4x4.sh` (the
campaign driver's refusal order, the stage gate) · `docs/data/p4_gate.json`, `docs/data/p4_4_baselines.json`, `docs/data/p5_3b_nortg.json`.

---

## 0. What the coordinator verified before writing this (2026-09-28, by running commands; the survey has the file:line)
1. **P4's five checkpoints** `output/p4_dt/dt_seed{101…505}.pt`: recipe K = 20, 3 layers, 1 head, `d_model` 128, dropout 0.1,
   `max_ep_len` 360, batch 64, lr 1e-4, wd 1e-4, warm-up 1,000, clip 0.25, 40,000 steps (declared 20,000, raised once), `target_rtg`
   −5762, `rtg_scale` 9991; **P4.7's five `mix50`** checkpoints `output/p4_7/checkpoints/mix50_dt_seed{101…505}.pt` (5 on disk),
   `R_best_source` −5959, `rtg_scale` 40223.
2. `train_dt` takes `context_length` (its CLI hardcodes 20); the window index is K-invariant (`offline/dataset.py:647-651`); windows
   END at step t; the rollout keeps a K-step window from the config with a K = 1 branch; **no test exercises K = 1 or 2.**
3. **P4.2's held-out comparison:** the K = 20 model beats MaxPressure on 100 of 100 paired draws (`p4_gate.json:wilcoxon`, w₊ = 0);
   **P4.4:** BC within δ = 0.6263 of the K = 20 model (DT − BC = −0.208 [−0.469, +0.054]); **P5.3b:** the prompt inert on `mappo1000`
   (+0.123), load-bearing on `mix50` (−409.15); the same contrast 0.1180 on `att_ours` and 0.1226 on `att_engine`.
4. **The per-draw `att_engine` rows of the K = 20 checkpoints exist ONLY in gitignored cells** (`output/p8_4b_rederivation/
   cell_hz1x1_dt_at_mappo1000_*.json`, 500 files); the committed `p8_4b_rederivation.json` holds means. C1 commits them (A26(c)).
5. **Costs:** ≈ 204 s per 40,000-step training at K = 20 on this GPU (`p4_training.json`; seed 505's 14,018 s a clock artefact),
   ≈ 2.9 s per hz1x1 CityFlow episode (`p5_3b_nortg.json`); the machine: RTX 5080 Laptop, torch 2.11.0+cu128, 47 GB RAM.

## 1. Why this task exists
H4 (§1–§2) is the paper's registered, confirmatory answer to DataLight's *DT cannot be applied to TSC*, which was measured at
K ∈ {1, 2}. A26 fixes the sweep's every degree of freedom before any K ≠ 20 model exists — and registers the expectation AGAINST the
hypothesis on `mappo1000`, because BC already sits within δ of K = 20 there. The task produces: 60 trainings run once and pinned;
7,000 CityFlow cells; one artifact carrying T1–T3 with Holm, the three pre-written outcome sentences with their quantities filled in,
and everything A26(d) lists beside them.

## 2. Scope fence — what NOT to build
- **No change to `agent/DTAgent.py` or `offline/dataset.py`**: if K = 1 or 2 needs a change there, STOP and say so (a finding).
- **No new K, no new subject, no new corpus, no new budget, no new metric.** Exactly A26's arms: per subject K ∈ {1, 2, 5, 10, 20} × 5
  seeds; on `mappo1000` also the two equal-supervision arms (K = 1 at batch 1,280 windows, K = 2 at batch 640); the reference arms
  (P4's five and P4.7's five, evaluated, never trained).
- **P4's and P4.7's checkpoints are never overwritten, moved or re-saved**; the K = 20 comparison reads them.
- **No evaluation of any trained checkpoint outside the campaign's token; no selection among checkpoints** (the payload after
  exactly 40,000 steps is the only one written).
- **Nothing under `output/p4_dt/`, `output/p4_7/`, `output/p8_4b_rederivation/` is edited**; the reference rows are COPIED into
  `docs/data/` with the sources' digests.

## 3. The commits, in order — each ≈ 2 source files plus tests

**C0 — `docs/plans/p5.3c.md`** (gate G0): assumptions with confidence; every seam re-verified with file:line; the K = 1 rollout's
exact behaviour (what the model sees at K = 1: state, RTG token, timestep); the batch-index stream's K-invariance demonstrated on a
synthetic corpus; the one-sided Wilcoxon's second route; open questions with proposed answers.

**C1 — `docs/data/p4_k20_att_engine_rows.json` + `tests/test_p5_3c_reference_rows.py` + the K = 1/2 agent tests (A26(c)'s
PRECONDITION; before any training).** (i) The reference rows: for each of P4's five seeds and each held-out draw 1000–1099,
`att_engine` and `att_ours` from `output/p8_4b_rederivation/cell_hz1x1_dt_at_mappo1000_*.json`, with each source cell's sha256, the
five checkpoints' sha256, the commit and the definition names; written by a small extraction command in `offline/context_sweep.py`
(C2's module, created here with this command only) that refuses a missing or duplicate (seed, draw) and verifies the checkpoints'
digests. A gated test re-reads the cells and asserts equality; an ungated test asserts the file's shape (500 rows, 5 seeds × 100
draws, both definitions). (ii) **T-k1 / T-k2 (load-bearing):** on a synthetic corpus, `train_dt(context_length=1)` and `=2` build
windows that END at step t with left padding, and `DTAgent` at K = 1 and 2 rolls out with a window of exactly K steps read from
the checkpoint's config (the number of steps the model attends to, asserted through the attention mask); **T-index:** the batch
index stream under one seed is identical at K = 1, 2 and 20 (the sampled window ids compared, not the tensors).

**C2 — `offline/context_sweep.py` + `offline/campaigns/p5_3c_train.sh`: the sweep's trainings, run once.** The registered table
(A26(a)–(b)): `mappo1000` × K ∈ {1, 2, 5, 10, 20} × seeds (25) + the equal-supervision `k1_b1280`, `k2_b640` × seeds (10) + `mix50` × K
× seeds (25) = **60 runs**, each `train_dt` with P4's (P4.7's) recipe and `context_length = K`, `batch_size` 64 (or 1,280 / 640),
`declared_gradient_steps` 40,000 and NO raise, the subject's own `target_rtg`/`rtg_scale`/stats fitted on its training split as P4
did, seeds 101…505; payload `dt-checkpoint/1.0` as `train_dt` writes it, with provenance naming K, the batch, the subject, the corpus
digests, the commit, device, torch and `deterministic False`; written ONCE (the exclusive in-memory write of `few_shot.py`); the
resume decision in Python (skip a valid checkpoint, refuse an invalid one, train an absent one); attempt markers; a manifest
`output/SHA256SUMS_p5_3c_train.txt` and a record `output/p5_3c_training/p5_3c_train.json` (per run: name, subject, K, batch, seed,
steps, the checkpoint's sha256, the weights-only digest, loop seconds, final loss, **loss per supervised target**, attempts). **The
K = 20 reproduction measurement (A26(b)):** after the five `mappo1000` K = 20 runs, each is compared tensor for tensor with P4's
checkpoint of its seed (`torch.equal` per parameter; the largest absolute difference per parameter recorded); the same for `mix50`
against P4.7's; the result is a RECORD, never a stop. Regime: CUDA, this GPU, `CUBLAS_WORKSPACE_CONFIG` unset, one torch thread,
OMP/MKL 1 (P5.2's / P7.3c's). The driver: token `output/p5_3c_runs/TOKEN_train`, canary, run worktree, `-P`, liveness, `tee -i -a`,
every refusal before the token, FAILED on every path after it. ≈ 3.4 h at C = 1 (measured by a fenced timing run of one K = 5 run
at B = 400 first, as G5 of `BRIEF_41` did, with the same-seed repeat by two routes).

**C3 — `offline/context_sweep.py` (evaluation and report) + `offline/campaigns/p5_3c_eval.sh`: the campaign.** **Cells:** for each
trained checkpoint and each held-out draw, one CityFlow episode, greedy, the subject's naive prompt, device `cpu`, engine seed 1000
(`evaluate_arm`'s path, `att_engine` and `att_ours` recorded per episode with the episode reward and the per-decision series);
**the reference arms first:** P4's five K = 20 checkpoints (500 cells) and P4.7's five (500) — **then the GATE (A26(c)):** P4's
reference rows equal `docs/data/p4_k20_att_engine_rows.json` under `==` on all 500, or the campaign STOPS before any sweep cell is
read; `mix50`'s reference rows are reported. Then the 3,500 `mappo1000` and 2,500 `mix50` sweep cells at 12 workers. **The
report** (refusals before any write; completeness against the declaration — no estimator on a partial set): per arm the per-draw
five-seed means `A_d(K)`; **T1** the per-draw rank-contrast `s_d` with `c = (−2, −1, 0, +1, +2)` and the ONE-SIDED Wilcoxon (s < 0) —
new code, with a second route required by a test (an independent implementation of the normal approximation with the tie and
continuity corrections, agreeing on the statistic and p); **T2, T3** the one-sided Wilcoxon on `A_d(K) − A_d(20) − δ > 0` for K = 1, 2
with `δ = 0.6263` a literal constant; **Holm within {T1, T2, T3}** at α = 0.05; the outcome as A26(d)'s partition with the registered
sentence, S and G₁, G₂ [CI] filled from the artifact and, for (ii), the names of the tests that did not reject; beside it, never
deciding: every pairwise contrast with CIs, the same on `att_ours`, the per-seed s (SD ddof 1), the smallest K within δ of K = 20,
the equal-supervision arms against their batch-64 counterparts, the whole design on `mix50` (exploratory, its own three-way
partition reported without deciding anything), A26(e)'s expectations as held or refuted, the K = 20 reproduction record, the loss
per target, `what_this_does_not_say` per A26(b), (e), (f) and A25. Artifact `output/p5_3c/artifacts/p5_3c_context_sweep.json`,
`p5.3c-context-sweep/1.0`, its cells' `e`-series in the chunks under the manifest `output/SHA256SUMS_p5_3c.txt`.

**C4 — the packet and the artifacts** (after G6): `docs/data/p5_3c_context_sweep.json` and `docs/data/p5_3c_train.json` (the
training record, committed by the coordinator at G3 — A26(b): pinned before the evaluation token) with a T-regress for the artifact
(byte-identical through the report with A3's two substitutions, its mutant killed); `docs/returns/P5.3c.md` per §7.

## 4. Tests — first, red for their own reasons, each named mutation executed and pasted
- **T-k1, T-k2, T-index** (C1, load-bearing): above. *Mutations:* the rollout window at K + 1 → T-k1 dies; the index stream
  depending on K → T-index dies.
- **T-rows** (C1): the reference file equals the gitignored cells (gated); the shape test ungated; a duplicate (seed, draw) refused.
- **T-table** (C2): exactly A26's 60 runs, by name, subject, K, batch, seed; K = 20 at batch 64 only; the equal-supervision batches
  1,280 and 640; a 61st or a K outside the set refused. *Mutation:* K = 3 admitted → dies.
- **T-recipe** (C2): `context_length` reaches `train_dt`; exactly 40,000 steps and no raise; the batch as declared; the prompt and
  scale the subject's own (P4's −5762 / 9991 read from the corpus, not typed); loss per supervised target = the mean cross-entropy
  over non-PAD positions, recomputed by the test on one batch. *Mutations:* a raise applied → dies; batch 64 for `k1_b1280` → dies.
- **T-once** (C2): an existing valid checkpoint skipped, an invalid one refused, never overwritten; the exclusive write refuses an
  existing file; two same-seed CPU runs at B = 20 byte-identical (the in-memory serialisation).
- **T-k20** (C2): the reproduction comparison on a fixture: identical tensors → `equal true`, one element changed → `equal false`
  with the largest absolute difference and the parameter named; the record written either way (never a stop). *Mutation:* the
  comparison made a refusal → dies.
- **T-gate** (C3, load-bearing): the reference-arm gate on fixtures: rows equal → passes; ONE row off by 1 ULP → the campaign stops
  before any sweep cell and nothing is written; a missing (seed, draw) refuses. *Mutation:* the gate skipped → dies.
- **T-T1 / T-T2** (C3, load-bearing): the rank contrast on a synthetic five-level fixture where the answer is known (an exact
  linear trend gives `s_d` = the slope × 10; a flat fixture gives 0); the one-sided Wilcoxon against the second route on 100 draws,
  statistic and p equal; the direction (a fixture where ATT RISES with K must NOT reject T1); T2/T3 with δ on a fixture where the
  K = 1 gap is exactly δ (does not reject) and δ + 0.01 over all draws (rejects). *Mutations:* the contrast on log K → the
  exact-trend fixture dies; `>= δ` boundary moved → dies; two-sided p used → the direction fixture dies.
- **T-holm** (C3): three p-values with the known Holm decisions (e.g. 0.01, 0.02, 0.04 → all reject at α = 0.05; 0.01, 0.03, 0.04 →
  the third does not) recomputed by the test; the outcome partition on every reject pattern, the sentence chosen and filled, (ii)
  naming the non-rejecting test(s). *Mutation:* Holm replaced by unadjusted α → the second fixture dies.
- **T-report** (C3): refusals precede every write (the reference rows at another digest; one declared cell absent; a chunk at a
  checkpoint digest not in the training record; no `canary.json`); the artifact regenerates byte-identically (C4's T-regress).
- **T-driver** (C2, C3): the text clauses (`-P` on every call, `tee -i -a`, `set -euo pipefail`, the token's position, the reference
  arms before the gate before the sweep, no outcome printed); executed on a sandbox: no token → nothing created; the gate's stop.
Suite discipline: every gated `skipif` names its artifact; at most FOUR real CityFlow episodes in the suite; `check_test_hygiene.sh`
on every test file and `check_english.sh` on everything, each falsified against a known-bad input first; mutants COMMITTED in a
throwaway worktree, pytest from a script file that does not carry a driver's liveness pattern.

## 5. Gates, in order — who runs each, what it checks, what stops the task, and how you learn its result
| # | Gate | Runs it | Checks | Stops the task if | You learn it by |
|---|---|---|---|---|---|
| G0 | Plan | coordinator, from `docs/plans/p5.3c.md` | the seams re-verified; the K = 1 rollout's behaviour; the index stream; the second route | a load-bearing assumption is wrong | Amendment A on `main` |
| G1 | C1–C2 review | coordinator + one reviewer (≤ 15 min, findings file), mutants re-run by the coordinator; also the trainings' pre-flight | the reference rows' extraction; T-k1/T-k2/T-index; the table; run-once; the K = 20 measurement; the driver | a defect | Amendment B |
| G2 | Token: the 60 trainings | **author — channel (a)** | — | — | the token file |
| G3 | Trainings verified and PINNED | coordinator, from disk; commits `docs/data/p5_3c_train.json` on `main` (A26(b)) | 60 checkpoints, manifest, provenance, steps, the K = 20 comparison recorded, attempts | a payload wrong → channel (c) | Amendment C |
| G4 | C3 review + pre-flight | coordinator + one reviewer on the evaluation, the statistic, the report, the driver; the coordinator's mutants | T1–T3, Holm, the sentences, the gate, the refusal order, the driver's every rm | a defect | Amendment D |
| G5 | Token: the campaign | **author — channel (a)** | — | — | the token file |
| G6 | Read | coordinator, from disk, capture first: the reference gate, then T1–T3 recomputed by an independent route from the raw chunks BEFORE the artifact's verdict is opened | 7,000/7,000; the verdict | the gate fails → the driver stopped; channel (c) | Amendment E |
| G7 | Packet | you (C4) → **"P5.3c done", channel (d)** | §7 | — | — |
| G8 | Merge review | coordinator spawns one (two mandates: the numbers by an independent route; the code by mutation) | — | a blocker | the merge, §6's box ticked |

## 6. Definition of Done
- [ ] C0–C4 delivered in order, each its own commit with named paths; no frozen file touched; `agent/DTAgent.py` and
      `offline/dataset.py` untouched; no new dependency; no new absolute path.
- [ ] The reference rows committed BEFORE any training (C1), equal to the gitignored cells.
- [ ] Sixty checkpoints written once, pinned on `main` before the evaluation token; the K = 20 comparison recorded for both subjects.
- [ ] Every test of §4 red first then green; every named mutation executed and pasted; hygiene and the English sweep run.
- [ ] The campaign complete under its token — the reference gate passed, 7,000 cells — with `docs/data/p5_3c_context_sweep.json`
      regenerating byte-identically.
- [ ] `docs/returns/P5.3c.md` per §7, then "P5.3c done".

## 7. Return Packet
`docs/returns/TEMPLATE.md`, plus: the reference gate 500/500; the K = 20 reproduction record (equal or the largest differences);
**A26(d)'s numbers in its words** — S with its CI, G₁ and G₂ with CIs, the three p-values and their Holm decisions, **the outcome as
the registered sentence, filled, reported not interpreted**; every pairwise contrast; the per-seed s; the plateau's description; the
equal-supervision contrasts and the loss per target; the whole design on `mix50`, labelled exploratory; A26(e)'s expectations as held
or refuted; every driver capture; where each run ran and at what commit; the amendments written against, by letter; the
AI-assistance record's four lines; one paragraph on what the paper's H4 section will assume.

---

# ✅ AMENDMENT A — 2026-09-28, gate G0: PLAN APPROVED (`docs/plans/p5.3c.md` @ `c26123a`) — every proposal accepted, two rulings that add a check, two corrections to THIS brief and one to A26 itself (A26.1, registered before any training)

## A0 — Verdict
Approved as written. The plan re-verified every seam at `1c05737`, demonstrated three load-bearing facts by running them (what a
K = 1 model sees; the batch-index stream's K-invariance through `train_dt` itself; the one-sided Wilcoxon against an independent
second route), found two errors in the coordinator's texts and one in the registration, and turned every open choice into a
question with a proposed answer. **The coordinator re-ran the plan's Appendix A.2 and A.4 demonstrations from their verbatim
scripts: both reproduce the pasted outputs exactly** (rows and window ids identical at K = 1, 2, 20 over 1,600 draws with a
different-seed control; 800/800 one-sided tests agreeing under `==` between the two routes, the direction fixture at p = 1.0).
**Build C1a, C1b and C2, in that order; stop at G1.**

## A1 — Corrections
1. **To A26 (F3): A26(b)'s sentence *"Equality has never been demonstrated on `output/p4_dt`"* is FALSE.** P4 retrained seed 505
   from scratch at its own commit and found all 48 tensors identical (`docs/returns/P4.md` §6.4; `PROJECT_PLAN` §8, 2026-08-11),
   and P4's double-train proof passed on CUDA. The coordinator took the pre-registration review's finding F5 on trust instead of
   opening P4's packet. **A26.1** (a correction row, the A22.1 shape) registers the fact before any K ≠ 20 model exists; the
   measurement and its path on a difference are UNCHANGED. The packet cites A26.1.
2. **To this brief (F5): §4's Holm example was wrong.** Under the step-down, 0.01 ≤ 0.05/3 rejects, 0.03 > 0.05/2 does not and
   STOPS, so the second and third both fail to reject. T-holm asserts (reject, not, not); the named mutation still dies.
3. **To this brief (F4):** *loss per supervised target* IS the per-step loss `train_dt` records (mean cross-entropy over non-PAD
   positions); the record carries `final_loss`, the 20 window means and `supervised_targets_per_step` replayed from the seed's
   index stream, as the plan proposes.

## A2 — Rulings on Q1–Q18: every proposal ACCEPTED; the additions are marked
- **Q1, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q13, Q14, Q15, Q16, Q17, Q18:** yes, as proposed.
- **Q2 (F2, the device) — yes, `cpu` as registered, WITH ONE ADDITION:** a FENCED pre-token re-roll (the `reference_reroll_check`
  shape of `BRIEF_41`'s campaign driver) of a DECLARED sample of P4's reference cells — the five seeds on draws 1000, 1001 and
  1002, 15 cells — on `cpu` at the campaign's commit, compared with the committed rows under `==`; a divergence REFUSES the start
  without consuming the token and is a finding for a registration decision (channel (b)), never a silent switch of device. The
  registered gate on all 500 remains the detector after the token. Alternative (c) is not taken.
- **Q3 (F3):** the correction is A26.1, not only the packet (A1.1).
- **Q12 — the (ii) sentence's extra clause is REGISTERED TEXT and goes into A26.1 with the plan's words:** *"K = k falls short of
  the K = 20 plateau by more than the registered margin δ = 0.6263 s of ATT."*

## A3 — What the coordinator verified for these rulings (2026-09-28, by running commands)
- `c26123a`: one commit on `main` (`1c05737`), `docs/plans/p5.3c.md` only (910 lines), no trailer; the amended-away `a7f42c7`
  differs by one removed sentence (the plan's own disclosure); the worktree clean.
- The plan's Appendix A.2 and A.4 scripts re-run from the scratchpad: outputs identical to the plan's §6 and §7.
- P4's record of seed 505 (`docs/returns/P4.md:317-327`; `PROJECT_PLAN` line 2059): all 48 tensors identical on a retrain.

## A4 — Next
C1a (the extraction command and its tests, the characterisation tests disclosed as not red-first), C1b (the rows at C1a's clean
commit), C2 (the sixty-run table, the run-once training route, the driver, the fenced timing run started by the implementer) —
committed on the branch, NOT pushed; then **"P5.3c C1–C2 done"** (channel (d)). A26.1 is tagged before G2's token; nothing
trains before both exist.

## A.1 — 2026-09-28, before C1 is built: the EVALUATION DEVICE is CUDA (A26.1(b), the author's addition) — Amendment A's Q2 ruling amended
A26(a)'s `cpu` rested on P4's `env_settings` record, whose `device` never reached a DT (the plan's V7); the reference rows were
produced with the DT on CUDA (P8.4b's runner, `--device` unset → CUDA). **A26.1 corrects the registration: the DT is evaluated on
CUDA, this GPU, for every sweep arm and both reference arms; the fenced pre-token re-roll of fifteen reference cells (Q2) runs on
CUDA; the two gated real-CityFlow tests (Q17) run on CUDA and skip naming it when unavailable.** C3's cell agent is built with
`device="cuda"`; the campaign's twelve workers each hold the model on the GPU as P7.3c's did (the driver checks free device memory
before the token and records the peak). The plan's assumption A5 dissolves; the registered gate's `==` now tests the evaluation
path alone. Training was already CUDA (A26(b)). Nothing else in Amendment A changes.

---

# ✅ AMENDMENT B — 2026-09-29, gate G1: C1a / C1b / C2 ACCEPTED at `8327128` (C1a `c507721`, C1b `1d1f67f`, C2 `150fd35`, the interim packet `docs/returns/P5.3c-C1C2.md`); the branch pushed; the run worktree created; the fenced timing run next, then ONE small commit C2.1 (tests and the header's numbers) before the training token

## B0 — Verdict
**PASSED.** Nothing found that can change a checkpoint, a row or a registered number. Five MINOR findings, all test gaps or wording,
are absorbed into C2.1 below. The trainings' pre-flight found no path on which a legitimate, complete set of sixty is refused.

## B1 — What the coordinator verified (2026-09-29, by running commands; every number from this session)
- **The branch:** four commits above the merge `7a8dbf6` of `main` `9a7f17b`; **0** AI trailers (`git log --format=%B | grep -ci`); ten
  files, ALL new, under `docs/`, `offline/`, `tests/` only; `agent/DTAgent.py` and `offline/dataset.py` untouched; no frozen path.
- **The reference rows, by an independent script** (`scratchpad/g1_rows_check.py`, not the task's test): file sha256
  `b36b8c7790f4740b65c158bba91ebc82a08a6acc9e45f8465a63710e078ed45b`; 500 rows, 500 distinct (seed, draw), exactly {101, 202, 303,
  404, 505} × {1000 … 1099}, sorted; every row `==` its cell on `att_engine`, `att_ours` (and `committed_att_ours`), `seed`, `draw_id`,
  `arm dt@mappo1000`, `scenario hz1x1`, `method dt`, `tier mappo1000`, `format_version p8.4b-rederivation/1.0`,
  `reproduces_committed true`; every cell's sha256 equal to the row's `source_sha256`; the 500 files of the pattern on disk are
  exactly the 500 sources; the five checkpoints' sha256 equal to the files under `output/p4_dt/` AND to `docs/data/p4_gate.json`;
  the campaign manifest's sha256 equal. Sanity anchor: per-seed mean `att_engine` 101.16 / 100.74 / 100.65 / 100.20 / 100.78,
  grand mean **100.7032** (`att_ours` ≈ 104.4–105.4), the P8.4b numbers for `dt@mappo1000` on hz1x1.
- **Tests, in a throwaway worktree `/home/filip/rltraffic-p53c-run`'s sibling `/home/filip/rltraffic-p53c-g1` at `8327128`, both
  data gates open** (`RLTRAFFIC_OUTPUT_ROOT`, `RLTRAFFIC_CORPUS_V11`): the three module files **71 passed in 34.63 s**; the driver
  file, run alone after them, **19 passed in 99.49 s** (the executed tests ran the real `check-inputs` and one real CityFlow canary);
  the worktree clean before and after; scratch 3.0 G, removed.
- **Hygiene and English, each falsified first:** a probe `assert value or True` under `tests/` → exit 1 `[TH001]`; the five test files →
  exit 0. A probe with Polish diacritics → exit 1; every file of the branch, the rows file and the plan included → exit 0.
- **P4's and P4.7's recipe, read from the PUBLISHED checkpoints** (`output/p4_dt/dt_seed{101,505}.pt`, `output/p4_7/checkpoints/
  mix50_dt_seed101.pt`, weights-only): `gradient_steps 40000 == declared 40000`, `batch_size 64`, `learning_rate 1e-4`,
  `weight_decay 1e-4`, `grad_clip 0.25`, `warmup_steps 1000`, `device cuda`, config `n_layer 3, n_head 1, d_model 128, dropout 0.1,
  max_ep_len 360` (= `DTConfig`'s defaults), prompts −5762.0 / 9991.0 and −5959.0 / 40223.0, `statistics_digest 9022a15d…` for
  `mix50`. P4's payloads record `raise_to 40000` (P4.7's `None`): `train_dt` records `raise_to` and never acts on it
  (`offline/dt_gate.py:769`), so the sweep's `raise_to None` at 40,000 steps is the same training. `train_dt`'s warm-up is
  `min(1000, steps // 2)` (`dt_gate.py:822`) — 1,000 at 40,000, which is what `validate_checkpoint`'s `WARMUP_AT_BUDGET` requires.
  P4's older payloads lack `rtg_mode` and `deterministic`; the new ones carry both. **The K = 20 reproduction compares the MODEL
  tensors only**, so that provenance difference never enters it.
- **The twelve named mutants, each COMMITTED in the throwaway worktree from the packet's own appendix specs, the tree reset to
  `8327128` after each, pinned to one thread:** M1 rollout window at K + 1 → KILLED (2 failed); M2 sampler seeded with K → KILLED
  (1); M3 duplicate (seed, draw) check removed → KILLED (1); D1 one committed `att_engine` off by 1 ULP → KILLED (1); T1 K = 3
  admitted → KILLED (2); T2 / T3 a raise applied → KILLED (11 / 11); T4 batch 64 for `k1_b1280` → KILLED (3); T5 an existing
  destination not refused → KILLED (1); T6 published by path instead of in memory → KILLED (1); T7 the K = 20 comparison made a
  refusal → KILLED (1); DR1 the token check skipped → KILLED (1, the executed no-token test). **Every count equals the packet's.**
  Two coordinator tooling errors on the way, neither reaching a verdict: the first runner handed pytest a `--basetemp` whose parent
  did not exist (every `tmp_path` test errored at setup and the runner printed SURVIVED for what never ran — caught by reading the
  output, not the verdict line); the second ran torch unpinned under a load of 30–47 and was stopped. ⚠️ **DR1's first attempt hung
  in the CANARY, before the token clause: the canary's python single-threaded in `futex_do_wait` for six minutes at load 0.11**;
  killed; the re-run killed the mutant in 25 s. `DEFERRED` 101 records it with the operational rule (B5).
- **The module and the driver read whole** (`offline/context_sweep.py` 1,951 lines; `offline/campaigns/p5_3c_train.sh` 385 lines);
  the window tests read whole. No finding of the coordinator's own beyond the reviewers'.

## B2 — The two reviewers (≤ 15 min each, own detached worktrees `-revA` / `-revB`, findings files `G1_REVIEW_{A,B}_FINDINGS.md`)
**A — the module and the pre-flight: PASS**, no BLOCKER / MAJOR. Verified by execution: `registered_runs()` = 60, five per arm,
every unregistered (subject, K, batch, seed) refused; `subject_facts` at K = 1 `==` at K = 20 for BOTH subjects on the real corpus
(stats, digest, `max_ep_len` 360, prompts, 72,000 windows, stream keys) — so the record's facts, built at K = 1, cannot refuse a
checkpoint trained at any K; the provenance merge keeps the caller's keys; no overwrite / truncate / partial-publish path; the
K = 20 comparison never raises on a difference and exits 0; nothing over-refuses a legitimate CUDA start; no absolute path;
`p4_dt` / `p4_7` only read; its own mutant MA (mix50 compared against mappo1000's reference) KILLED. **MINOR:** MB (`checks["budget"]`
made self-consistent) and MC (`checks["recipe"]` compared to the payload's own warm-up) SURVIVE — no test fabricates a checkpoint at
another budget or warm-up; it matters because the timing run's name `mappo1000_k5_b64_seed101` is a registered run's, and those two
checks are what tells a fenced timing file from a registered one if a person ever moves it; MD (`torch.set_num_threads(1)` removed)
SURVIVES — the thread count is recorded in every provenance, not asserted. NOTES: the free-memory bar is the K = 5 timing's peak,
not the largest arm's (immaterial on 16 GB with a 3-layer d128 model); a SIGKILL leaves `.partial` / `.staging.pt` files that the
manifest and the resume scan refuse loudly.
**B — the driver, the window tests, the extraction: PASS**, no BLOCKER / MAJOR. The order traced line by line against the header
§3 and C2; every post-token `fail` writes FAILED; 15 interpreter calls, all `PYTHONPATH=$WORK_TREE "$PY" -P`; `set -euo pipefail`
propagates a failed slot through the `| sed -u` pipelines (shell experiment; `set -eu` would not — its mutant M3 KILLED by
`test_strict_mode`); the timing mode's sampler bounded; T-k1/T-k2 assert, THROUGH the attention mask and `torch.equal` with the
loader's window at the same (episode, t), that the model is fed exactly K steps read from the checkpoint's config (M1: mask
admitting K − 1 → KILLED); T-index compares window ids through `train_dt` itself with the count asserted K-invariant and a
seed-202 control; the sandbox substitutes six lines of a committed clone's copy (`RUN_TREE`, `OUTPUT`, `TOKEN`, the train and
timing-slot calls, the canary) with a one-occurrence assertion. **MINOR:** the canary's timing-half threshold (`> 2.0 s` →
refuse) has no test (M5 SURVIVES); the extraction checks the cells' policy source by PATH suffix — the cells carry no checkpoint
digest, so this is the strongest check the cells permit; the committed `checkpoints` block is the extraction-time digest verified
against `p4_gate.json`, and A26(c)'s gate on all 500 rows is the registered detector of a swap. NOTE for the estimate: the timing
measures ONE point (K = 5, batch 64, 400 steps, warm-up 200); not measured: ms/step at K ∈ {1, 2, 10, 20} (30 of 60 runs are K ≥ 10),
the equal-supervision arms (3,840 tokens per step, K = 20's), `build_seconds` at other K paid sixty times, the per-run calls,
thermal drift over hours.

## B3 — Rulings
1. **C1a, C1b, C2 are ACCEPTED as committed. Nothing in `offline/context_sweep.py` changes before the trainings.**
2. **C2.1 — ONE commit, AFTER the fenced timing run, BEFORE the training token; tests and the header only, no module change:**
   (a) the driver's header §5 with the timing's measured numbers (ms/step, build seconds, device and allocated peaks, the repeat by
   two routes) and this LABEL, in these words or closer ones: *"measured at K = 5, batch 64, 400 steps; the sixty's total is an
   EXTRAPOLATION across K ∈ {1, 2, 10, 20}, the two equal-supervision arms and sixty input builds, not a measurement"*;
   (b) three tests, red first on a mutant, then green: the canary's timing-half threshold as a text clause of
   `canary_both_halves` (B's M5 dies); `validate_checkpoint`'s `checks["budget"]` and `checks["recipe"]` False on a fabricated
   payload at another budget and another warm-up, both asserted (A's MB and MC die); one torch thread after
   `enter_registered_regime()` (A's MD dies) — each mutant pasted;
   (c) nothing else. The commit named in the packet; then **"P5.3c C2.1 done"**. The coordinator re-runs the three mutants and the
   files, pushes, and RE-CREATES the run worktree at C2.1 before the token (J1(c): the header change is a non-docs change).
3. **The provenance wording (B's second MINOR)** goes into the FINAL packet (C4), not into the committed rows file: *"the cells record
   no checkpoint digest; the committed `checkpoints` block is the digest of the files P8.4b's runner loaded by path, verified at
   extraction against `docs/data/p4_gate.json`; A26(c)'s gate is the detector of a swap."* The rows file is NOT re-extracted.
4. **The packet's open questions:** (2) a staging or `.partial` leftover after a kill is LEFT as evidence and named in the packet —
   nothing deletes it; the resume scan and the manifest refuse it loudly, and a person moves it aside (the driver's §2). (3) accepted:
   no two executed drivers overlap; the coordinator's re-runs never overlap a suite run.
5. **`offline/campaigns/p7_3c_finetune.sh`'s sampler pattern** (the implementer's observation): `DEFERRED` 100; P7.3c's slots ran for
   minutes, it never met the race, and P7.3c is merged and read. Not changed.
6. **Line 137's discarded import stderr** (B's note): accepted as is — the refusal precedes the token; diagnosability only.

## B4 — The fenced timing run (Amendment A Q13; the implementer starts it, as for `BRIEF_41`'s G5)
The branch is pushed (`origin/task/p5.3c-context-length` at `8327128`) and the DETACHED run worktree exists:
`/home/filip/rltraffic-p53c-run` at `8327128c41a36518f1cbebe167454974eeb19feb`. Run the TIMING mode from it exactly as the driver's
header §0 says (Step 1's `mkdir -p` and a tmux foreground pane, or `setsid --wait`; the full commit as the second argument), with the
machine otherwise QUIET (no suite, no other pytest, no reviewer — plan §8, 2026-09-10). Report **"P5.3c timing done: stamp
<stamp>"** with the driver's `timing alone` / `timing repeat` / `timing repeat: file sha256 …` lines and `TIMING COMPLETE`. Then
C2.1 (B3.2).

## B5 — The canary hang (`DEFERRED` 101) — an operational rule for every start of this driver
Observed once, in the coordinator's executed DR1 test under a machine load of 30–47: the canary's python (one CityFlow episode)
single-threaded in `futex_do_wait` for six minutes at zero load; not reproduced; the same canary has passed in every P7.3c / P7.3d
capture and in this session's other runs. Rule: **the machine is quiet at every start; if the driver prints nothing for a minute
after `resume_decision:` (train) or `check_inputs PASSED` (timing), the canary is hung: Ctrl-C — the canary precedes the token, so
NOTHING is consumed — and start again.** A second occurrence is a finding for the plan, not a rate question.

## B6 — Next, in order
timing run (B4) → C2.1 (B3.2) → the coordinator verifies, pushes, re-creates the run worktree at C2.1 → **the author's token G2**
(`output/p5_3c_runs/TOKEN_train`; the train mode takes the timing's stamp as its third argument) → the sixty (≈ hours, one at a
time) → G3 (the coordinator verifies the sixty from disk and pins `docs/data/p5_3c_train.json` on `main`) → C3.

---

# ✅ AMENDMENT B.1 — 2026-09-29, ≈ 16:00: the fenced timing run READ and C2.1 ACCEPTED at `5779a1d` (packet `1f446f2`); the branch pushed; the run worktree RE-CREATED at `1f446f2`; the training token G2 handed to the author; one correction to Amendment B; the packet's four questions ruled

## B.1.0 — Verdict
**C2.1 ACCEPTED as committed; the fenced timing run READ from disk; nothing changes before the token.** The letter C stays
reserved for gate G3 (§5's table); this is B's addendum.

## B.1.1 — What the coordinator verified (2026-09-29, by running commands)
- **The branch:** `6a62100` = merge of `main` `0d15681` (two parents), `5779a1d` C2.1, `1f446f2` the packet; **0** trailers;
  C2.1 touches `offline/campaigns/p5_3c_train.sh` (the header §5, +22/−1) and the two test files only; **`offline/context_sweep.py`
  is byte-unchanged since `8327128`** (`git diff --stat 8327128 HEAD -- offline/context_sweep.py` empty).
- **The timing run:** ran at `8327128` from `/home/filip/rltraffic-p53c-run` (the capture's `WORK_TREE` and `commit` lines);
  `check_inputs PASSED`; canary **1.03 s**, both correctness values the reference's; `TIMING COMPLETE in 17s`; `DRIVER EXIT: 0`;
  `output/p5_3c_training/` holds `fenced_timing/20260929T112505Z/` and NOTHING else; no `SHA256SUMS_p5_3c_train.txt`; the
  capture at `output/p5_3c_runs/train_timing_capture.txt`. **By the coordinator's own route** (`hashlib`, `torch.load`
  weights-only with the `TorchVersion` allowlist, the CSVs): `alone.pt` and `repeat.pt` share sha256 `f796315069cf…`, equal to
  `alone.json`'s, `repeat.json`'s and `timing.json`'s; **48 / 48 model tensors equal (647,176 elements)**; ms/step
  `loop_seconds / steps × 1000` = **8.2535 / 7.0098**, the record's; device peaks = the CSV maxima **1,540 / 1,516 MiB** (32 / 30
  samples, baselines 1,125 / 1,136); the payload: K 5, batch 64, seed 101, 400 = 400 steps, `raise_to None`, warm-up 200, cuda,
  `deterministic False`, `code_commit 8327128…`, `code_dirty False`, `sweep_format p5.3c-sweep-provenance/1.0`, config the
  registered one with `context_length 5`, prompt −5762.0 / 9991.0; the regime recorded (`CUBLAS_WORKSPACE_CONFIG` None,
  deterministic algorithms off, OMP/MKL "1", one torch thread).
- **The header §5** carries the measured numbers and the extrapolation LABEL in the ruling's words ("∈" as "in": the file is
  ASCII), the run counts right (**20 of 60 at K ≥ 10**), and five flagged RECORDED lines (P4's / P4.7's own K = 20 loop seconds on
  this GPU, 193–205 s per run, hence 3.2–3.4 h — the source of the plan's ≈ 3.4 h) — **kept**: labelled, sourced, and the author
  plans his evening by them.
- **Tests at `1f446f2` in a fresh throwaway worktree, both data gates open, one thread:** the three module files **75 passed in
  37.71 s**; the driver file alone **20 passed in 95.54 s** (the real `check-inputs` and one real CityFlow canary executed).
- **The four named mutants, COMMITTED in that worktree from the packet's appendix, the tree reset after each:** M5 (the canary's
  timing-half refusal removed) → KILLED, 1 failed / 19 passed; MB (`checks["budget"]` self-consistent) → KILLED, 2 failed; MC
  (`checks["recipe"]` warm-up from the payload) → KILLED, 2 failed; MD (`set_num_threads(1)` removed) → KILLED, 1 failed — exactly
  the new tests, every count the packet's.
- **The packet's disclosure** (a needle of the new canary test occurred twice in the correct driver, so three "kills" of the first
  mutant run were red-on-correct; found by reading the failure REASONS before any green run, fixed, all seven re-run): the right
  discipline, recorded in §8 as an implementer catch of the project's signature class.

## B.1.2 — Correction to Amendment B, B2 (the coordinator's error)
B2's note *"30 of 60 runs are K ≥ 10"* is WRONG: `registered_runs()` gives **20** at K ≥ 10 (K = 20 and K = 10, five seeds, two
subjects); 30 is the count at K ≥ 5. The number was adopted from reviewer B's report without recomputing it — a number taken from
a description. Nothing depends on it; the header §5 has the right count. Logged in `PROJECT_PLAN` §8.

## B.1.3 — The packet's four questions, ruled
1. **`docs/plans/p5.3c.md`:** NOT edited. The timing's numbers live in the header §5, the packet and `PROJECT_PLAN` §8's row of
   this date; the plan file is the record of gate G0 and stays as approved.
2. **`checks["recipe"]`'s other members** (learning rate, weight decay, clip): three more parametrised cases of
   `test_a_checkpoint_at_another_budget_or_warm_up_fails_exactly_budget_or_recipe`, each mutant pasted — **in C3's commit**, not
   before the token (the code is right by reading and by MC's kill on the same tuple; a commit before the token would mean a third
   run worktree for no change in what trains).
3. **B5's operational rule into the header §0** — **in C3's commit**, in both drivers' headers (the training driver's §0 and the
   campaign driver's), for any future start; for G2 the author has the rule in the token block below.
4. **The P7.3d host-RAM guard vs the suite's tmpfs scratch:** `DEFERRED` 102. Diagnosed by the packet's four runs: `/tmp` is
   tmpfs; a whole-suite run's `basetemp` grows to 13 G and drives `MemAvailable` below `p7_3d_grid4x4.sh`'s 24,216 MiB budget by
   the time `test_p7_3d_campaign_path.py` runs; `-o tmp_path_retention_policy=failed` keeps the scratch at ≈ 0.7 G and the suite
   passes 2722 / 0. Remedy: that option as a line in `pyproject.toml`'s `[tool.pytest.ini_options]` (not frozen; no dependency) in
   **C4**, with the seven tests unchanged; until then every local whole-suite run passes the option on the command line. Not this
   task's code; nothing about C2.1.

## B.1.4 — Gate G2: the token, and what runs meanwhile
- The run worktree is RE-CREATED, detached, at **`1f446f2363092673dc5cf6d3ad95c59ed678276d`** (the pushed branch tip: C2.1 plus the
  packet), clean; the driver's second argument is that commit, the third the timing stamp **`20260929T112505Z`**.
- The author starts the TRAIN mode per the header §0 (Step 1 the pane, Step 2 at its prompt), the token written in the same line,
  on mains power, the machine otherwise QUIET; expected 3.2–5.5 h by the two rates in §5. `check-inputs --timing` needs ≥ 1,540 MiB
  free on the device (14,971 free at 15:55). If nothing prints for a minute after `resume_decision: 60 to train, 0 to skip`, the
  canary is hung (B5): Ctrl-C consumes nothing; start again.
- **While the sixty train:** the implementer may WRITE C3 (the evaluation, the statistic, the report, the campaign driver and their
  tests) and run its FIXTURE-level tests pinned to one thread (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `--basetemp` in its
  scratchpad, `tmp_path_retention_policy=failed`); it runs NO whole suite, NO executed driver, NO gated CityFlow / CUDA test and
  nothing that touches the GPU until the capture says `TRAINING RUN COMPLETE`. The coordinator runs nothing.
- **After `TRAINING RUN COMPLETE` and `DRIVER EXIT: 0`:** the author pastes the capture's last lines here; gate G3 (the coordinator
  verifies the sixty from disk — checkpoints, manifest, provenance, steps, the K = 20 comparison record, attempts — and commits
  `docs/data/p5_3c_train.json` on `main`), Amendment C; then C3's review (G4).

---

# ✅ AMENDMENT C — 2026-09-29, ≈ 21:00, gate G3: THE SIXTY VERIFIED FROM DISK AND PINNED — `docs/data/p5_3c_train.json` committed on `main` at `c55693a` (sha256 `017808a5e84fada469d6b2d302889ad171b8e429b1612b8ac06c900ef3c6321a`); the K = 20 reproduction is EQUAL, 48/48 tensors, all ten pairs

## C0 — Verdict
**PASSED.** Every one of the sixty checkpoints is at its manifest digest with the registered recipe; the driver wrote each once, in one
attempt, at `1f446f2`, under the author's token; the record is pinned byte-identically. **A26(b)'s measurement is answered: the sweep's
K = 20 checkpoints are P4's and P4.7's published ones, bit for bit** (`torch.equal` on every tensor, both subjects, all five seeds, by the
driver's `compare-k20` AND by the coordinator's own load). No payload wrong; channel (c) not needed.

## C1 — What the coordinator verified (2026-09-29, by running commands; the scripts `scratchpad/g3_verify.py` and the capture greps)
- **The capture** `output/p5_3c_runs/train_capture.txt` (683 lines, sha256 `ca280f358ee9…`): `WORK_TREE /home/filip/rltraffic-p53c-run`,
  `commit 1f446f2363092673dc5cf6d3ad95c59ed678276d`, `check_inputs PASSED … timing record` (the stamp's `timing.json`),
  `resume_decision: 60 to train, 0 to skip`, `canary 0.70 s` on the reference values, the token line (written 15:49:34 local, the
  author's text), `token consumed and deleted`, `start dir …/starts/20260929T134954Z`; **60** `train, attempt 1` lines, **0** attempts
  > 1, **60** `wrote` lines, **0** skips, **0** FAILED / INTERRUPTED / REFUSING; `compare-k20 … 5/5 mappo1000 and 5/5 mix50`;
  `manifest … 60 … re-verified`; `record … 60 runs`; `TRAINING RUN COMPLETE in 16958s`; `DRIVER EXIT: 0`.
- **The directory:** `checkpoints/` 60 `.pt`, no hidden or stray entry; `runs/` 60; `attempts/` 60, all `.1`; `staging/` empty;
  `starts/20260929T134954Z/{canary.json, COMPLETE}`; `k20_reproduction.json`; `p5_3c_train.json`; `fenced_timing/` untouched; the token
  gone. `output/SHA256SUMS_p5_3c_train.txt`: 60 lines, `sha256sum -c` ALL OK.
- **Every checkpoint, by the coordinator's own route** (`hashlib`; `torch.load` weights-only with the `TorchVersion` allowlist; the
  sixty names constructed independently from A26's table): **60 / 60 pass 27 checks each** — sha256 equal across the file, the manifest
  line, the run record and the training record; run-record format `p5.3c-train-run/1.0` with 40,000 losses whose last is
  `final_loss`; 20 window means; `gradient_steps 40000 == declared`; `raise_to None`; `warmup_steps 1000`; lr 1e-4 / wd 1e-4 / clip
  0.25 / the arm's batch; `context_length` = K in the config, the provenance and `registered_context_length`; the config
  `{25, 8, K, 3, 1, 128, 0.1, 360, conditioned}`; `run` / `arm` / `subject` / `seed` the registered ones; `device cuda`;
  `deterministic False`; `code_commit 1f446f2…`, `code_dirty False`; `sweep_format p5.3c-sweep-provenance/1.0`; `dt-checkpoint/1.0`;
  the subject's prompt (−5762 / 9991; −5959 / 40223); `normalise True`; `scenario_id cityflow1x1`; `attempts 1`, `reruns 0`; the
  record's own `checks` all True; `supervised_targets_per_step` at 40,000 steps and the arm's batch; **`stats` equal to the published
  checkpoint's of that seed** (one distinct `stats` object per subject across its thirty / thirty-five runs); the regime recorded
  (`CUBLAS_WORKSPACE_CONFIG` None, one torch thread, deterministic algorithms off).
- **The record** `p5_3c_train.json` (`p5.3c-train-record/1.0`, `n_runs 60`): the names are exactly the registered sixty; the manifest's,
  the timing's and the K = 20 record's sha256 match the files; the embedded K = 20 and timing records are identical to the files; the
  subjects block: `mappo1000` −5762 / 9991, statistics `38a53a0c17ed…`, 200 streams, 72,000 windows, `max_ep_len` 360; `mix50` −5959 /
  40223, statistics `9022a15d22eb…`, 200 streams, 72,000 windows; four `what_this_does_not_say` sentences.
- **The K = 20 reproduction, redone by the coordinator:** for each of the ten (subject, seed) pairs, `torch.equal` on every one of the
  48 tensors between `checkpoints/<subject>_k20_b64_seed<s>.pt` and `output/p4_dt/dt_seed<s>.pt` / `output/p4_7/checkpoints/
  mix50_dt_seed<s>.pt` → **48 / 48, keys equal, all ten**; the driver's record agrees (`all_equal true`, `n_parameters_differing 0`,
  `largest_abs_difference 0.0`, `weights_digests_equal true`, the references at `p4_gate.json`'s / `SHA256SUMS_p4_7.txt`'s digests).

## C2 — The per-arm record (loop seconds are WALL TIME on a machine that was not always quiet; the trainings are step-based)
| arm | loop s mean (min–max) | ms/step | final loss per target, mean (min–max) | targets / step | first-window loss |
|---|---|---|---|---|---|
| mappo1000_k20_b64 | 222.3 (206.7–237.9) | 5.56 | 0.0177 (0.0132–0.0244) | 1,246.2 | 0.614 |
| mappo1000_k10_b64 | 235.6 (220.6–267.0) | 5.89 | 0.0338 (0.0265–0.0393) | 632.0 | 0.644 |
| mappo1000_k5_b64 | 238.4 (216.0–251.4) | 5.96 | 0.0811 (0.0443–0.1303) | 318.2 | 0.677 |
| mappo1000_k2_b64 | 256.8 (235.3–265.0) | 6.42 | 0.1603 (0.0957–0.2371) | 127.8 | 0.732 |
| mappo1000_k1_b64 | 247.9 (220.0–264.7) | 6.20 | 0.2532 (0.1435–0.3485) | 64.0 | 0.771 |
| mappo1000_k1_b1280 | 343.9 (328.4–362.3) | 8.60 | 0.0113 (0.0056–0.0168) | 1,280.0 | 0.620 |
| mappo1000_k2_b640 | 274.3 (260.8–296.2) | 6.86 | 0.0114 (0.0069–0.0165) | 1,278.2 | 0.623 |
| mix50_k20_b64 | 215.7 (208.0–230.3) | 5.39 | 0.0085 (0.0032–0.0161) | 1,246.2 | 0.715 |
| mix50_k10_b64 | 230.5 (220.7–250.6) | 5.76 | 0.0131 (0.0050–0.0219) | 632.0 | 0.786 |
| mix50_k5_b64 | 331.2 (223.0–695.6) | 8.28 | 0.0315 (0.0166–0.0422) | 318.2 | 0.880 |
| mix50_k2_b64 | 484.5 (172.0–1,718.4) | 12.11 | 0.0864 (0.0504–0.1040) | 127.8 | 1.035 |
| mix50_k1_b64 | 179.4 (169.6–187.1) | 4.49 | 0.1129 (0.0814–0.1406) | 64.0 | 1.151 |
Sum of loop seconds 16,303 s (4.53 h) of 16,958 s wall. **Two stalls, wall time only:** `mix50_k5_b64_seed505` 695.6 s (written
17:34 UTC) and `mix50_k2_b64_seed101` 1,718.4 s (18:03 UTC) — 3–7× their arms' other runs; the machine was in use then (the
implementer's C3 work). The `k1_b1280` arm is legitimately slower (a 1,280-window batch). **P8.2's compute table uses per-arm
MEDIANS or minima and names the stalls; never these means.** The losses are training losses per supervised target and say nothing about
held-out performance (the record's own limits); the observation that they fall with K on both subjects, and that the two
equal-supervision arms reach 0.011 — below K = 20's 0.018 — goes into the packet as an observation, not a finding. H4 is decided by
T1–T3 on the held-out draws and by nothing here.

## C3 — What the EQUAL result means for C3 and the paper (a ruling on wording, not on a number)
A26(b) registered the K = 20 reproduction as a measurement whose difference would be reported; the difference is zero on every tensor.
Consequences the report and the packet state in these terms: (i) **the sweep's K = 20 arm is P4's model** (and `mix50`'s is P4.7's), so
H4's K = 20 point is P4's checkpoint re-evaluated, and the equal-supervision and K < 20 arms differ from P4 by K (and batch) alone,
trained by the same code path that reproduced P4 bit for bit; (ii) A26(c)'s reference gate — P4's five checkpoints re-evaluated `==`
the committed rows — now also pins the sweep's own K = 20 cells: any difference between the K = 20 arm's cells and the reference rows
would be an evaluation-path difference, never a model difference, and the report says so in its `k20_reproduction` section;
(iii) the observation from the timing run (the same-seed repeat equal) and this one together are a property of THIS GPU, driver and
torch build under the registered regime, stated as such, not as determinism of the method.

## C4 — Rulings for C3 (in force with B.1.3)
1. C3's `check-inputs` reads the pinned `docs/data/p5_3c_train.json` at sha256 `017808a5…` from the run tree's `docs/data/` (the branch
   merges `main` at or after `c55693a`), verifies the sixty files at its digests, and refuses a chunk whose checkpoint digest is not in
   it (§3 C3 as issued). The reference arms use the published files by `p4_gate.json` and `SHA256SUMS_p4_7.txt`, as issued.
2. The report carries C2's table (from the pinned record, not retyped) beside the training-loss observation, and C3's wording.
3. Nothing under `output/p5_3c_training/` is written to again; the campaign's outputs go under `output/p5_3c/` and
   `output/p5_3c_runs/` as §3 C3 says.

## C5 — Next
The implementer merges `main` (the pinned record), finishes C3 with B.1.3's items 2 and 3, and says **"P5.3c C3 done"** → gate G4
(the coordinator's mutants + one reviewer on the evaluation, the statistic, the report and the campaign driver) → Amendment D → the
author's evaluation token G5 → the campaign (the reference arms, the gate, 7,000 cells at twelve workers) → G6 → C4 → G8, the merge.

---

# ✅ AMENDMENT D — 2026-09-30, gate G4: C3 (`cf97e31`, packet `cdae1b0`) PASSES on the numbers path — no defect that can change a cell, the gate's verdict or H4's statistic; ONE small fix commit C3.1 (the run's own protection: a marker ordering, two test gaps, two documented restart remedies, three report tightenings) BEFORE the evaluation token; a correction to Amendment C's stall attribution

## D0 — Verdict
**PASSED for everything that decides a number.** The cells take P8.4b's path on CUDA with the registered prompt; the gate compares
500 cells under `==` on both definitions and stops the campaign before any sweep cell; the confirmatory family is fed exactly the
sweep's `mappo1000_k{K}_b64` arms on `att_engine`, per-draw means over the five seeds, the K = 20 level the sweep's own arm; T1–T3,
Holm, the partition and the sentences are the registered ones; every refusal precedes every write. **C3.1 (D4) is required before the
token** because three findings concern what protects THIS run from losing a token or hours; none of them can reach a number.

## D1 — What the coordinator verified (2026-09-30, by running commands and reading)
- **The branch:** `363a1a8` and `2f0ab38` merges of `main` (two parents each); `cf97e31` C3 (+5,139 / −4 across nine files: the
  module +2,152 lines with C1's and C2's code untouched, the campaign driver new, two fixtures/tests new, B.1.3's items 2 and 3);
  `cdae1b0` the packet; **0** trailers; no frozen path, `agent/DTAgent.py` / `offline/dataset.py` / `offline/att_rederivation.py` /
  `offline/admission_probe.py` / `offline/rtg_calibration.py` untouched.
- **Tests at `cdae1b0` in a fresh throwaway worktree, both data gates open, CUDA visible, one thread:** the five module files
  **180 passed in 219 s** (the two gated real-CityFlow cells on CUDA included), the training driver **21 passed in 105 s**, the
  campaign driver **15 passed in 27 s** — 216, none skipped; the worktree clean after.
- **Read whole by the coordinator:** the statistic (2884–3160), the gate and the re-roll (2519–2680), the cell factory and payload
  (2070–2152), `validate_chunk` / `run_campaign_cell` / `write_chunk` (2164–2332), the stage runner (2332–2520), the artifact builder
  and the writers (3292–3690), the campaign driver (393 lines), `tests/test_p5_3c_statistic.py` (361 lines), the packet (732 lines).
- **The named mutants, COMMITTED in the throwaway worktree from the packet's appendix, the tree reset after each — eighteen, every one
  KILLED with the packet's failing-test counts:** S1 the contrast on log K (3), S2 the boundary moved (5), S3 a two-sided p (5),
  S4 Holm unadjusted (1), S6 the A26.1(c) clause dropped (1), S8 the partition without T1 (4), C1 the cell on cpu (3), C3 mappo1000's
  prompt for every subject (1), C10 the rollout window at K + 1 through the campaign's factory (2), K1–K3 the commands' exit codes
  (1 each), G1 the gate skipped (2), G2 the sweep without the gate record (3), R6 the report without the gate check (1), C4 a chunk
  reused on existence (2), C7 a dirty-tree chunk accepted (2), D1 the driver's gate stop removed (1, the text test; the packet's
  executed test also killed it).
- **The declared campaign, by the coordinator's own tally** (`scratchpad/g4_declared.py`): 7,000 cells, 7,000 distinct names,
  reference 1,000 / sweep 6,000, 14 arms × 500 (`ref_mappo1000_k20`, `ref_mix50_k20`, the twelve registered arms), draws 1000–1099,
  seeds 101 … 505, 500 gated cells, 15 re-roll cells (five seeds × draws 1000–1002); `EVAL_DEVICE_BUDGET_MIB` 6,972 (415 × 12 × 1.4).
- **The machine at 15:30 local:** `MemAvailable` 37,167 MiB (the driver's budget 24,216); GPU 14,928 MiB free (budget 6,972); load
  0.9; `output/p5_3c/` absent; no `p53c_campaign` tmux session; the run worktree at `1f446f2`, clean (to be re-created at C3.1).

## D2 — The two reviewers (≤ 15 min each, own detached worktrees `-revS` / `-revD`, findings files `G4_REVIEW_{S,D}_FINDINGS.md`)
**S — the statistic and the report: PASS**, no BLOCKER / MAJOR. In its own words, what feeds `confirmatory`: the five sweep arms
`mappo1000_k1_b64 … mappo1000_k20_b64`, `att_engine`, the five seeds ascending, draws 1000–1099; the reference arm reaches only
`k20_reproduction.sweep_k20_beside_the_published`; the equal-supervision arms only `equal_supervision`; mix50 only
`exploratory_mix50` (no sentence). Independent arithmetic: its own ranks / ties / continuity / erfc-Φ Wilcoxon agrees with the
module's on the fixture's 100 draws with |Δp| = 0.0 on T1–T3; Holm by hand agrees on seven triples. Seven own mutants, **7 / 7
KILLED** (the K = 1 level from the batch-1,280 arm; the K = 20 level from the reference arm; the mean over four seeds; `att_ours`
fed in; the T2/T3 subtraction reversed; outcome (i) on one shortfall; `tests_not_rejected` inverted). MINOR: `per_draw_means` refuses
only a NON-SHARED seed set (a uniform four-seed set passes the function; unreachable at the builder, whose completeness refusal and
`TRAINING_SEEDS` loop make it moot, and killed there); `what_this_does_not_say` carries no entry labelled A26(f). NOTES: the
confirmatory block does not record the five arm strings it was fed; the campaign fixture's three p-values are all 1.98e−18 (outcome
(i) at a degenerate point — the statistic file is the load-bearing killer for Holm and the partition, the campaign test for the arm
selection and the arithmetic); mix50's A26(e) refuter is operationalised as "T3 does not reject" (plan Q10, a ruling).
**D — the cells, the gate, the re-roll, the driver, the pre-flight: PASS**, no BLOCKER / MAJOR. The cell path equals P8.4b's
(`agent_with_target` with 40,000 declared steps and the subject's `TierSpec` prompt, `act(info, explore=False,
update_memory=True)`, `probe_episode`'s arguments, engine seed 1000, `resolve_device(None) == resolve_device("cuda")` on this GPU;
the env settings by another route but enforced `==` P8.4b's before the token); every `validate_chunk` refusal satisfied by a real cell
(the 20 `EPISODE_KEYS` are `as_record()`'s exactly; the first decision's reward `None` allowed; the commit read from the MODULE tree,
the run worktree, never the cwd); inputs verified before any chunk is touched; reuse by content; no duplicate write possible; the gate
and the re-roll as specified; every `rm` in the driver listed (the token; the two markers); every path after the token reaches a
marker but one (below). Seven own mutants: **four KILLED** (reuse ignoring the demand digest; the gate under a 1e−6 tolerance; the
driver's sweep stop removed; the engine seed from the draw) and **three SURVIVED**: `n_failed` zeroed in `run_campaign_stage`;
`_campaign_worker` returning ok on an exception — **the production worker is exercised by no non-gated test and first runs under the
token**; the pool at one process regardless of `--workers`. MINOR: the driver sets `SUCCESS=1` before `printf … > COMPLETE`
(:389–390); `check_campaign_inputs` reads P8.4b's 500 mix50 cells for existence only (their digest compared with nothing; they feed
the reported, never deciding, mix50 comparison); a restart that re-rolls a reference chunk changes the gate verdict's chunk digests
and `_write_campaign_text_once` refuses the stale record AFTER the second token (remedy: move it aside; a third token); a Ctrl-C mid-
write leaves `.cell_….tmp` that the next start's `resume-check` refuses before the token (remedy: move it aside). NOTES: an
OOM-killed worker's task never returns (a silent hang; B5's rule is the detector); the header's line 112 contradicts line 30 (the
fenced g2 write precedes the token); the estimate: ≈ 3.4 s per cell (P5.3b's 2.9 s plus ≈ 0.3–0.6 s of per-cell re-derivation),
ideal 33 min at twelve workers, **realistically 1–1.5 h** with twelve CityFlow processes and twelve CUDA contexts on one GPU.

## D3 — Rulings, and a correction
1. **Amendment C, C2's attribution of the two training stalls to "the implementer's C3 work" was WRONG and is withdrawn** — the
   coordinator's error, written without evidence. The implementer's timestamped record shows its session ran no test and no Python
   between 15:52 and 18:42 UTC; the author has stated the cause: **≈ 8.5 GB of GPU memory in the P4 clock state was his game on the
   Windows host** during 17:22–18:04 UTC. Wall time only; nothing in a checkpoint depends on it. P8.2's compute table cites the
   pinned record with per-arm medians and names the two stalls with that cause (the packet's Q1). Logged in `PROJECT_PLAN` §8.
2. The packet's readings and questions: the **borrowed host budget** (24,216 MiB, P7.3d's × 1.4) is accepted as an upper bound — the
   machine has 37 GiB available and the driver refuses, never guesses (Q2); the **(ii) clause's placement** after the registered
   parenthesis and the verb agreement stand as implemented (Q3); **`DEFERRED` 102's remedy stays in C4** (Q4); the **"NOT made" branch**
   of C3's reading stays as written — an honest sentence on a branch only a fixture reaches, better than a refusal over wording
   (Q5); **mix50's reference comparison** by both routes (P8.4b's per-draw cells under `==`, P5.3b's committed means recomputed), never
   gating, stands — A26(c)'s "P5.3b's per-draw rows" names data P5.3b read, not data it committed, and the report says so (§10.1);
   the brief's §3 C3 text (`cpu`, `evaluate_arm`) is superseded by A.1 and plan A7 as accepted at G0 (§11.2).
3. **The mix50 cells' digests** (D's first MINOR): `DEFERRED` 103 — a check against P8.4b's campaign manifest where it declares them,
   in C4 if cheap; exploratory, never deciding; not before the token.
4. **The OOM-hang note** (D): `DEFERRED` 101's rule covers it; the campaign driver's header already carries B5.

## D4 — C3.1: ONE commit, tests first, nothing beyond this list; then "P5.3c C3.1 done"
1. **The driver `offline/campaigns/p5_3c_eval.sh`:** (a) write `COMPLETE` BEFORE `SUCCESS=1` (lines 389–390 swapped), so a failed
   final write still reaches `on_exit`'s marker — pinned by the existing text test's order list or a new clause; (b) header §2
   (RESUME) gains the two remedies: *a restart that re-rolled a reference chunk finds the gate record stale — `reference-gate` refuses
   "already exists and differs" after the token; move `output/p5_3c/reference_gate.json` aside by hand and start again*, and *after a
   Ctrl-C a worker's `.cell_<…>.json.<pid>.tmp` may remain in `cells/` — `resume-check` names it before the token; move it aside by
   hand*; (c) line 112's sentence corrected to say what line 30 says (the fenced g2 re-roll is written before the token, nothing
   else). A text test pins (a) and the two remedies' presence.
2. **Tests for the two surviving mutants of the run's protection** (reviewer D's M1 and M7; specs in
   `scratchpad/revD_mutants.py`): (a) `run_campaign_stage` driven with a fake worker that returns `ok False` for one cell → `n_failed`
   1, that cell in `failures`, `n_rolled` the rest, and `_cmd_cells` exiting 1; (b) `_campaign_worker` itself with `run_campaign_cell`
   monkeypatched to raise → `ok False` and the error's type and text in `error`, no chunk written; and with it returning a payload →
   the chunk written by `write_chunk` and `ok True`. Each red first on its mutant (M1: `n_failed` zeroed; M7: `ok True` on the
   exception), pasted. (c) If cheap: the pool created with `processes == workers` (D's M3), otherwise say so.
3. **Three tightenings in `offline/context_sweep.py`** (reviewer S), each with a test and its mutant: (a) `per_draw_means` refuses
   any seed set other than exactly `TRAINING_SEEDS` (the docstring's claim made true); (b) `_ARTIFACT_LIMITS` gains one entry
   labelled **A26(f)** in A26's own words (δ keeps A6's value and gains a superiority use; A25 sits beside); (c) the artifact's
   `confirmatory` block records `arms`: the five arm strings it was fed, in K order, and `k20_arm` naming the sweep's own — so G6's
   independent route reads them from the artifact, not from the code.
4. **Nothing else.** The packet `docs/returns/P5.3c-C3.1.md` (short: the diff, red/green, the mutants, the suite tail with
   `-o tmp_path_retention_policy=failed`). The coordinator then re-runs the new mutants and the changed files, pushes, re-creates
   the run worktree at C3.1, and hands the author the evaluation token block as **Amendment D.1** — the token is NOT written before
   that.

## D5 — What the campaign will be read against (G6, for the record now)
The coordinator reads the capture first (the pre-token lines, the re-roll's 15 MATCH lines, the gate's verdict line, the stage
lines, `CAMPAIGN COMPLETE`, `DRIVER EXIT: 0`), then recomputes A_d(K), s_d, T1–T3, Holm and the partition **from the raw chunks by
an independent route BEFORE opening the artifact's `confirmatory` section**; only then the artifact, compared under `==`; the
reference gate 500 / 500 and the sweep's K = 20 cells beside the published (C3(ii)); then Amendment E. Expected wall time 1–1.5 h;
the driver prints its own clock.

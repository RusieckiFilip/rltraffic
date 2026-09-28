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

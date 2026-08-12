# BRIEF 13 — P4.5: is %BC's advantage checkpoint selection, or episode quality?

**Mode:** Claude Code implementation session · **Branch:** `task/p4.5-bc-seed-selection`, from `main`
**Worktree:** a fresh one — `git worktree add /home/filip/rltraffic-p45 -b task/p4.5-bc-seed-selection main`
**Read first, from disk:** `docs/returns/P4.4.md` §12.8 (the finding this task tests),
`docs/reviews/P4.4.md` §4, then this file. `BRIEF_11` and `BRIEF_12` are closed; nothing here reopens them.

---

## 1. Why this task exists

P4.4 measured that %BC's top-10 % return filter selected **19 of its 20 streams from MAPPO seeds 101
and 202**, which are independently the two best checkpoints on the held-out pool (103.61 and 103.53
against 105.99, 106.98, 107.80). Training return predicts held-out ATT at **r = −0.991 over disjoint
draw sets**, and demand was excluded as the cause.

From that we **deduced** that the filter performs checkpoint selection. **A deduction is not a
measurement**, and *"we inferred it from a correlation with n = 5"* is a materially weaker sentence
than *"we trained it and it reproduced"*. This task makes it a measurement.

**It is also the first task under `CLAUDE.md` §8** (the AI-assistance record) and under
`docs/CONTRACTS.md` **C9** (do not write "MADT" in prose — the model is *"the offline multi-agent
Decision Transformer"*, or *"the DT"*).

## 2. ⚠️ The design problem you must not walk into

The obvious arm — **BC trained on seeds 101+202** — is **confounded**, and shipping it alone would
answer nothing. %BC trains on **20 streams**; seeds 101+202 hold **80**. So that arm differs from %BC
in *which seeds* **and** in *how much data*, and either result would be unattributable.

**The design below breaks the confound by holding size fixed.** Four arms, all trained and evaluated
exactly as P4.4's were — same 40,000 steps, same five seeds, same optimiser, same **full-training-split
normalisation statistics**, same 100 held-out draws, same `evaluate_arm`:

| arm | streams | drawn from | what it isolates |
|---|---|---|---|
| `bc_top10` | 20 | top-return (**already measured**: 103.1627) | the observed filter |
| **`bc_best2_20`** | **20** | **random, from seeds 101+202 only** | **seed identity at MATCHED SIZE** |
| **`bc_any_20`** | **20** | **random, from all five seeds** | **size alone, seeds unmatched** |
| `bc_best2_all` | 80 | all of seeds 101+202 | data quantity from good seeds (secondary) |

> ⭐ **EXTENDED 2026-08-12 by §10.7 — a FIFTH arm, `bc_worst2_20`: 20 streams drawn at random from
> seeds 505 and 303, the two WORST checkpoints (106.97726 and 107.79804; measured, and note it is not
> 303+404). Two matched-size arms give a difference; three give an ORDERING.** The best-two-to-worst-two
> spread is **3.8190 ATT**, six times δ.

**The decisive comparison is `bc_best2_20` vs `bc_any_20`** — identical training-set size, differing
only in which checkpoints produced the data. If seed identity is what the filter is exploiting, that
pair separates; if it does not separate, the checkpoint-selection reading is wrong and F3's
interpretation must be corrected in `PROJECT_PLAN` §8 and in `docs/returns/P4.4.md` §12.8.

**Subset draws come from the training seed's RNG**, so the five-seed spread averages over five subset
draws rather than resting on one lucky subset. **Say so in the packet**: those CIs cover subset
variance *and* training variance, which the P4.4 arms' CIs do not.

## 3. The registered prediction — write it into `docs/plans/p4.5.md` BEFORE the first gradient step

Per `PREREGISTRATION.md` **A8(a)**'s discipline. Both directions are informative; neither is a
formality.

> **Primary:** `bc_best2_20` lands **within δ = 0.6263 of `bc_top10`** (103.1627). If matched-size
> random sampling from the two best checkpoints reproduces %BC, then the filter's contribution *is*
> checkpoint selection and the return ranking within those seeds adds nothing.
>
> **Secondary:** `bc_any_20` lands **worse than `bc_best2_20`** by an amount comparable to the
> behaviour-policy gap between the best two seeds and the five-seed mixture (**2.05 ATT**).
>
> ⚠️ **CORRECTED 2026-08-12 (§10.1, the implementer's finding, my error): 2.05 is the gap to the BEST
> SINGLE seed (202, 103.52859). The gap to the MEAN OF THE BEST TWO is 2.0134** (103.56864 against the
> mixture's 105.58203). Report both; score this secondary **directionally**, since it is a magnitude
> expectation and not a threshold.
>
> ⭐ **THIRD PREDICTION, added by §10.7 and to be copied verbatim into the plan file:**
> **`bc_best2_20` < `bc_any_20` < `bc_worst2_20` in ATT (lower is better) — monotone in
> behaviour-seed quality.** This is the actual mechanism claim: not *"does seed identity matter"* but
> *"does performance TRACK behaviour-mode quality"*. **If all three land together, seed identity does
> nothing and F3's reading collapses — which we would then learn decisively rather than from one null
> contrast.**

**What falsifies the primary, and what it would mean:** if `bc_top10` beats `bc_best2_20` by more
than δ, then ranking episodes *within* the good seeds does real work, F3's reading is incomplete, and
the paper must say the filter does **both** — a correction we would then owe P4.4's packet.

**δ is IMPORTED, not derived here.** A6's δ = 0.6263 is the DT's margin over its behaviour mixture and
has nothing to do with this comparison; it is reused as the project's registered equivalence scale for
this scenario, and **that is a choice, stated here rather than defended later** (the 2026-08-11
lesson: registering a threshold makes it fixed, not principled). **Therefore report the paired mean
difference, its CI, its width and the rank-biserial for every pair unconditionally**, so no reader
depends on δ.

## 4. Per-file requirements

**`offline/offline_baselines.py`** — extend, do not restructure. Two selectors beside
`top_return_streams`, with the same return type (`tuple[StreamReturn, ...]`) so
`filter_stacked_to_streams` consumes them unchanged:

- `streams_from_datasets(dataset, dataset_dirs) -> tuple[StreamReturn, ...]` — every stream whose
  `dataset_dir` is in the given set. Raises naming both sides if a requested dir yields nothing.
- `random_stream_subset(streams, count, rng) -> tuple[StreamReturn, ...]` — a uniform sample without
  replacement, **deterministic given `rng`**, returned in the same canonical order
  `(dataset_dir, episode_file, ix_id)` that `top_return_streams` uses, so downstream row order cannot
  depend on draw order. Raises if `count > len(streams)`.

CLI: extend `train` with `--stream-selector {top_return,datasets,random_subset}` and its parameters.
**Do not add a second training function** — `train_bc` already takes a filtered stack.

**Artifacts:** extend `docs/data/p4_4_training.json`'s pattern into a **new** file
`docs/data/p4_5_selection.json` (do not patch P4.4's artifacts — they are merged and cited). Record
per arm: the selector, its parameters, the RNG seed, **the full list of selected streams**, the
per-seed composition of that list, and the training record. Plus, for the two random arms, the
**per-training-seed subset composition**, so the subset draw is auditable rather than asserted.

**Results:** `docs/data/p4_5_baselines.json`, same shape as `p4_4_baselines.json`, carrying all
per-episode records — 3 new arms × 5 seeds × 100 draws = **1500** — plus `bc_top10`'s 500 **re-used
from the merged artifact, not re-rolled** (it is the same model on the same draws; re-rolling it would
be a second measurement of a settled number). State that reuse explicitly.

## 5. Tests — each shipped with its named mutation executed and the failure pasted

The code is new, so **red-first applies here** (unlike `BRIEF_12`). Write the tests, watch them fail
for the right reason, then implement.

| # | test | mutation that must kill it |
|---|---|---|
| T1 | `streams_from_datasets` returns exactly the streams of the named dirs, on a fixture with three dirs | return all streams regardless of dir |
| T2 | `random_stream_subset` is **deterministic for a given rng seed and different for another** | seed the rng from the clock |
| T3 | the subset is returned in **canonical order regardless of draw order** — shuffle the input, assert identical output | return in draw order |
| T4 | **the load-bearing one:** the rows `filter_stacked_to_streams` keeps for a random subset belong **only** to that subset, and number `360 × count` | off-by-one in the row-index map |
| T5 | sampling `count > len(streams)` raises, naming both numbers | clamp instead of raising |
| T6 | the three arms' selected-stream lists are **disjoint from the held-out draw pool** (leakage, asserted from the artifact) | any leak |
| T7 | `bc_best2_20` and `bc_any_20` have **equal training-set size** — asserted from the artifact, because the whole design rests on it | let the sizes differ |
| T8 | the artifact records the per-training-seed subset composition for both random arms | drop the per-seed record |

**T7 is the one that protects the conclusion**, in the way §8.4's row-`T` test protected P4.4: every
other test can pass while the two arms differ in size, and then the decisive comparison measures data
quantity while claiming to measure seed identity.

## 6. Also in scope — two queued items in the same file, clearly secondary

Fold in **`DEFERRED` 32 and 33** (`docs/reviews/P4.4.md` F5 and F6), because this is the next task to
touch `offline/offline_baselines.py` and they will otherwise rot:

- **F5:** `_run_report`'s completeness derivation can be reverted to a tautology with the suite green,
  and the baseline arms' expected seed set is derived from data (`training["runs"]`) while
  `madt`/`mappo1000` correctly use the declaration (`TRAINING_SEEDS`).
- **F6:** the unbalanced-design cross-check and the δ-rounding cross-check both survive being disabled.

**Each needs a test with an executed mutation.** If either turns out to be more than an hour, stop and
report it — they are secondary and must not delay the result.

## 7. Scope fence — what NOT to do

- **No new tiers, no mixtures, no other scenario.** The tier grid and the mixture axis are **P4.6**,
  and P4.6 runs **after P4.3**.
- **No RTG work, no DT arm.** This task trains BC variants only. The DT is not re-evaluated here and
  no DT-versus-baseline sentence is written (`docs/reviews/P4.4.md` §8.6 binds until P4.3 lands).
- **Do not touch** `p4_gate.json`, `p4_4_baselines.json`, `p4_4_training.json`, `offline/dt_gate.py`,
  `offline/dataset.py`, `agent/DTAgent.py`, `agent/OfflineBaselines.py` (no agent change is needed —
  the architecture is identical across all arms), or any frozen path.
- **Do not write "MADT"** anywhere (`CONTRACTS` C9).
- **Do not tune anything.** All four arms use P4.4's exact recipe; the only difference between arms is
  which streams they see.

## 8. Definition of Done

- [ ] `docs/plans/p4.5.md` committed **before** any training, carrying the prediction of §3 verbatim
- [ ] Four arms trained and evaluated; `bc_top10` reused from the merged artifact and said so
- [ ] Every mutation in §5 executed and its failure pasted
- [ ] Full suite green, real tail pasted; test count ≥ 613 collected
- [ ] Paired comparisons with CI, width and rank-biserial for **every** pair, reported unconditionally
- [ ] `claude_guard.sh --frozen-only`, `check_english.sh`, `check_test_hygiene.sh` all exit 0
- [ ] `git diff --stat` shows no frozen path and no edit to P4.4's merged artifacts
- [ ] Return Packet at `docs/returns/P4.5.md`, **including the new `AI-assistance record` section**
      (`CLAUDE.md` §8, `docs/returns/TEMPLATE.md`) — this task is the first to carry it
- [ ] §6's P4.5 checkbox left **unticked**; it is mine, in the merge commit

## 9. What I will do with the result

**If the primary holds** (`bc_best2_20` ≈ `bc_top10`): the paper states that on a single-controller
expert corpus the top-return filter is **checkpoint selection**, measured rather than deduced, and
P4.6 tests the other two instantiations of the same mechanism — best *controller* on a heterogeneous
mixture, easiest *demand draws* on pure random.

**If it fails:** F3's reading is incomplete, and I correct `PROJECT_PLAN` §8, `docs/returns/P4.4.md`
§12.8 and `docs/reviews/P4.4.md` §4 — the same backward propagation the 2026-08-07 rule requires. **A
failed prediction here costs three annotations and buys a true sentence**, which is the trade this
whole apparatus exists to make.

---

## 10. RULINGS on the plan of 2026-08-12 — **all four approved**, one brief error accepted, and one arm added

### 10.1 Your §3 correction is right and the error is mine

**2.05 is the gap to the best SINGLE seed, not to the mean of the best two.** Recomputed here from
`p4_heldout_thresholds.json`: mixture **105.58203**, best-two mean (202, 101) **103.56864**, gap
**2.0134**; best single seed 202 is **103.52859**, gap **2.0534**. §3's *"the gap between the best two
seeds and the five-seed mixture"* attached the right number to the wrong reference — **the same
reference-class defect I logged against myself on 2026-08-12**, committed four rulings later in a
brief I wrote to test a reference-class finding. Report both, score the secondary directionally, and
**do not treat either as load-bearing**: it is a magnitude expectation, not a threshold.

### 10.2 The suite deadlock — recorded, disclosure made binding, and not fixed here

Accepted as reported: 8 minutes at ~0 % CPU with 35 threads in `futex_do_wait`, then **612 passed, 1
skipped in 69.43 s** pinned. **This is the second sighting of the same liveness class** (P4.4 §6.2 was
the evaluation path; this is the test suite) and the first inside `pytest`. It is intermittent — I ran
the same suite unpinned twice today, 76.09 s and 79.58 s, without a hang.

**Binding: every suite run you report states whether it was pinned, and every background job in this
task sets `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.** The pin is a **liveness** fix, not a performance
one, and your pinned count (612 passed, 1 skipped = 613 collected) matches P4.4's exactly, so it moves
no outcome. **Not fixed here** — a `conftest.py` pin would change every test run's threading and is a
decision about test infrastructure, not about this result. `DEFERRED` 41.

### 10.3 Q1 — Gate B **APPROVED as proposed**, with one substitution

Re-using `bc_top10`'s 500 episodes is only sound if this session's instrument is P4.4's, and hashes
alone cannot show that — **P4.4's own Gate A existed for exactly this reason and found a thread
hazard on the way.** Take both halves: the identity checks **and** the 25 declared cells
(5 seeds × draws 1000/1025/1050/1075/1099), exact equality, re-rolled values discarded, **mismatch ⇒
BLOCKED and you stop.**

⚠️ **Substitution: the primary identity is `canonical_state_dict_digest`, not the file sha256.**
`DEFERRED` 29 and the 2026-08-11 ruling: a file hash depends on the filename, so it proves transport
integrity and not weight identity. Report the file sha256 beside it for what it does prove.

### 10.4 Q2 — background jobs from the session, and the pin is not optional

~65 minutes in jobs of ≤ 12 minutes is well inside the `BRIEF_11` §7 Q8 precedent, under which P4.4
ran a 25-minute Gate A and three evaluation arms. **Approved.** `tmux` is for the multi-hour campaigns
of §5, and this is not one. Every job carries the §10.2 pin, aborts on first failure, and ends with
the completed-equals-requested assertion.

### 10.5 Q3 — **keep both, but RANKED, because they answer different questions**

You are right that the plan left them unranked, and right that a disagreement is itself a result.

- **The registered forecast is scored by (A), `|paired mean difference| ≤ δ`** — because that is how
  §3 words it: *"lands within δ of"* is a claim about a point estimate, and P4.4's
  `_forecast_outcomes` scored its forecast the same way.
- **The equivalence claim that may enter the paper requires (B), the whole 95 % CI inside ±δ** — A6's
  rule, and the only one of the two that can support *"these are the same"* rather than *"we failed to
  find a difference"*.

**Neither substitutes for the other and both are reported.** If they disagree — mean inside δ, CI
straddling — the declared name for that outcome is **"consistent with equivalence, not demonstrated at
this power"**, reported with the CI width, and it may not be written up as a match. **Fixing this
before the run is the whole point; after it, choosing between them is choosing a verdict.**

### 10.6 Q4 — both **APPROVED**, and the first is more than hygiene

`delta_verdict` with neutral names: yes. Reusing `equivalence_verdict` would write
**`dt_genuinely_better` into an artifact describing a BC-versus-BC pair** — a false label on disk,
which is the defect class F1 was raised for. `SELECTION_ARMS` frozen and CLI-checked: yes, and it is
the campaign-integrity rule (a shell typo must not be able to redefine an arm).

### 10.7 ⭐ MY ADDITION — a fifth arm, `bc_worst2_20`, which turns a contrast into a dose-response

**Two matched-size arms give a difference; three give an ORDERING, and an ordering is much harder to
explain away.** Add:

| arm | streams | drawn from | why |
|---|---|---|---|
| `bc_worst2_20` | 20 | random, from seeds **505 and 303** | the low end of the same axis |

**Measured here, so the arm is defined by data and not by guess:** per-seed held-out ATT is
`202 103.52859 · 101 103.60869 · 404 105.99759 · 505 106.97726 · 303 107.79804`, so the worst two are
**505 and 303** (mean **107.38765**) — **not** 303+404. The best-two-to-worst-two spread is
**3.8190 ATT**, six times δ, so the design has room to show an effect or to rule one out.

**Registered prediction, added to §3 and to be copied verbatim into `docs/plans/p4.5.md`:**

> **`bc_best2_20` < `bc_any_20` < `bc_worst2_20` in ATT (lower is better), monotone in behaviour-seed
> quality.**

**Why this is worth ~22 minutes.** The two-arm design can only answer *"does seed identity matter?"*
A three-arm ordering answers *"does performance TRACK behaviour-mode quality?"*, which is the actual
mechanism claim — and if all three land together, seed identity does nothing, F3's reading collapses,
and we learn that decisively rather than from a single null contrast. **A monotone ordering across
three matched-size arms is the strongest form this result can take**, and it costs one more training
run and one more evaluation pass.

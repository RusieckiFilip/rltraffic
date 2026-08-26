# BRIEF_29 — P5.3a fix round, and the rulings on the packet's open questions

**Task id:** `P5.3a-fix` · **Branch:** continue on `task/p5.3a-rtg-probe` · **Issued:** 2026-08-25
**Review:** `docs/reviews/P5.3a.md` — **FAIL, 2 blockers, 5 major, 9 minor**
**Compute:** no training, no new campaign. One probe re-run at most.

> **Read this after `BRIEF_28` and its amendments A and B. Where it disagrees with either, this wins.**
> **The review's numbers are the reviewer's; BL-1, MJ-1, MJ-4 and BL-2's uncovered call sites were
> re-measured by the coordinator from the artifact before this was written.**

---

## 0. First: the work is good, and the two blockers are about evidence, not numbers

**Every headline number reproduced under independent reimplementation, to the last digit** — the
probe cells, R6's exact zero, the live crosscheck's 47.22 % / 4.0044, row B on all eight tiers.
The reviewer's own words: *"Neither blocker impugns a committed number."* The §6.1 identity test
**runs here and does not skip**, both mutations were reproduced, and §A's explanation was
independently confirmed — `random`'s DT is genuinely RTG-insensitive, `embed_rtg.weight` mean |w| =
0.491, so the token is embedded and not dead.

**What failed is the layer that lets someone else believe it.** Both blockers are fixable in a day.

---

## 1. Rulings on the packet's open questions

### §A — ACCEPTED as option (i). Your refusal to add a third cell was right

The acceptance property is *"the test fails when the guarded thing breaks"*, and **cell 1
demonstrates it under both mutations**. A4's *"on cell 2"* was a **proxy** for that property, and the
coordinator chose it on `RtgSummary.std` — the statistic diagnosed as ramp-dominated **in the same
amendment**, immediately before writing *"the selection is not re-opened."* Row B now shows `random`
is the **narrowest of eight**. **A4 picked the worst discriminating cell available, by a rule stated
on the wrong quantity, and then fenced it against revision.** That is the coordinator's error, it is
the third of its shape in this task, and it is logged in the Decisions Log.

Adding a row-B-chosen third cell now is choosing after seeing the outcome. **Refusing unprompted was
correct.** ⭐ **New from the review and it strengthens the position: both mutations are ALSO killed in
3 seconds by the unit tests** — `d07` by the §6.2b positive control, `d08` by
`test_a_legacy_eight_key_payload_loads_as_conditioned`. So the mechanism has fast protection
independent of the 6-minute test.

> **REQUIRED:** the plan and the packet must state that **cell 2 is not a discriminating cell**, so no
> later reader counts two cells as two protections. Cell 2's survival is itself a reported result: it
> converges with the probe's independent `flip_rate = 0` on `random`.

### §B — AUTHORISED: `322 → 330`, deliberately, and with a drift guard the old code did not have

**Written authorisation, dated 2026-08-25, to be quoted in the packet:** *the implementer is
authorised to change the pinned count in `tests/test_erfc_determinism.py` from `322` to `330`, and
only that literal, recording in the test's docstring that P5.3a's δ table copies eight
`(z, p_value)` pairs from `p4_6_grid.json` / `p4_7_grid.json`.* The test is **right** and asked for a
deliberate update; this is a spec change ruled by the coordinator, not a test weakened to pass.

⚠️ **Your second option — dropping `z`/`p_value` and citing the source — identified a real hazard and
is refused for a better reason than "A7 says so": a copy with no pointer is the hazard; a copy with a
checked pointer is not.**
> **REQUIRED with the count change:** every Wilcoxon entry in the δ table carries a **source pointer**
> naming the artifact and arm it was copied from, plus a test asserting **copy == source**. That makes
> drift mechanically detectable and turns the guard's extra 8 pairs into protection rather than
> exposure.

### §C — RESOLVED, and the resolution corrects a claim this project has repeated since 2026-08-13

You left the choice of headline open. It is now settled by measurement, **not by preference**, and
the measurement cost nothing: `docs/data/p4_3_rtg.json` already carries per-episode ATT for every grid
point. Pairing `dt_g0` (target 0) against `dt_g8` (target −13000) over the same 100 draws × 5 seeds —
**n = 500 paired cells, computed by the coordinator 2026-08-25:**

| quantity | value |
|---|---|
| mean difference | **+0.9026** ← reproduces P4.3's committed headline exactly, so the pairing is correct |
| median | +0.9553 |
| **sd** | **2.6300** |
| min / max | **−9.6692 / +11.5798** |
| cells that move at all | **499 of 500** |
| cells moving ≥ 2 ATT | **211 of 500 (42.2 %)** |
| mean absolute movement | **2.1281** |
| your crosscheck cell (101, 1000) | **+4.0044** — reproduces from a different artifact |

🚨 **"P4.3 measured the prompt at 0.9026 ATT" has been read throughout this project as *the prompt is
a weak lever*. That reading is wrong. 0.9026 is the mean of a distribution whose spread is three times
its mean.** The prompt moves **499 of 500** episodes, by a mean absolute **2.13 ATT** and up to
**11.58**, and it moves them **in the registered direction** — prompting for a worse return gives
worse performance. Naively +0.9026 against SE ≈ 0.118 is **t ≈ 7.7**; the seed/draw dependence means
**P8.1 owns the real interval** and this figure may not be quoted as one.

> ⭐ **`A9` HAS AN OBJECT.** The model did learn to condition on the return token. **P7.2 has a
> subject**, and the question it inherits is no longer *"is there a knob?"* but *"does the knob change
> quality, and does it change it more off-distribution than on?"*
> **REQUIRED in the packet §8.2:** carry this table, state that the headline is the **dissociation**
> (policy moves, mean quality barely does), and state that your 47.22 % was **n = 1** and now has an
> n = 500 companion. **Do not write "weak lever" and do not write "inert".**

### §D — the coordinator's error, with both mechanisms named

`094b53f` is on your branch and **not on `main`**. `git push origin main` pushes a **ref**, not `HEAD`,
so it succeeded while pushing nothing — success and no-op observationally identical, the family logged
on 2026-08-07. And `git add -A docs/` swept your uncommitted plan, stray tags included. `PROJECT_PLAN`
§10's rule exists and `git rev-parse --abbrev-ref HEAD` was not run first. **Nothing for you to fix;
it is in the Decisions Log and the merge will carry the commit to `main`.**

### §E — correct, no action. CI is expected to breach the ceiling at merge

That is the mechanism working. **Do not pre-bump and do not widen with slack.** The protocol is
`re_measure_required_at.what_to_do`, and it is the coordinator's step at merge.

---

## 2. BLOCKERS — both must close before merge

### BL-1 — the artifact does not say what the packet says it says

`fixedtime`'s TVD is **never zero**: 4.904e−11 … 2.271e−09, **0 of 55** intervention-cells exactly 0.
Its `mean_abs_logit_delta` reaches **0.2082** against `random`'s **0.2553** — the same order. **The
discriminator the packet states does not exist.**

⭐ **The finding underneath is better than the one that was written, so this is a rewrite and not a
retraction.** Both tiers move their logits comparably; they differ by **four orders of magnitude in
TVD** because on `fixedtime` the movement is near-uniform across actions and softmax cancels it,
while on `random` it is slightly less uniform. Neither ever crosses a decision boundary.

⚠️ **And the artifact's only exact zeros are `mappo1000`'s five — R6's null control, one per seed.
The one true zero in the file is the one the packet does not point at.**

**Required:**
1. Rewrite `docs/returns/P5.3a.md` §8.2 and the §8.1 `fixedtime`/`random` rows to the measured values,
   in scientific notation where a six-decimal column rounds a real number to zero.
2. State the real discriminator (**four orders of magnitude at comparable logit movement**) and the
   mechanism (near-uniform logit shift).
3. Quote TVD ranges **over all interventions**, not the `grid_g8` column, wherever a range is given.
4. Re-examine, and say plainly, whether the *inert* reading survives on `fixedtime` at TVD ~1e−9.
   **It probably does — 1e−9 is negligible — but say so on the measured basis.**
5. ⭐ **Sweep the class, do not fix the sentence.** Grep every number in the packet that is quoted as
   an exact zero and check each against the artifact at full precision. This project has twice caught
   the same author making the same quantifier error twice in one document.

### BL-2 — give the instrument a second route, and cover the five uncovered functions

`flip_rate` is **the** critical quantity and CLAUDE.md §2 requires critical quantities to be computed
twice, independently. Today **m19 — the double-scaling the module's own docstring at `:27-29` predicts
— turns the headline into exactly `0.000000` with a green suite.**

**Required:**
1. **A second-route recomputation test.** One probe cell (`mappo1000@101` is the natural choice)
   recomputed by an **independent implementation inside `tests/`** that does not call
   `teacher_forced_logits`, `compare_logits` or `probe_cell`, asserted equal to the shipped path.
   ⚠️ **The reviewer wrote exactly this and it reproduced to the last digit — but a script in `/tmp`
   is not evidence the repo holds.** *An agent's findings are not an artifact until they are on disk.*
2. **Kill all eight named survivors**, each with its failure pasted: **m15** (teacher forcing broken),
   **m16** (window-slot off-by-K), **m19** (double scaling), **m20** (checkpoint-path swap),
   **m21** (`crosscheck_targets` not the endpoints), **m22** (`chosen[:20]` — R4 unenforced at its only
   call site), **m23** (`spread_table` on 20 of 200 streams), **m24** (episode order).
3. **m19 deserves its own named test** referencing the docstring that predicts it.
4. If any mutant proves **equivalent** rather than uncaught, say so and prove it — that is a finding,
   not a failure.

---

## 3. MAJOR — all five

- **MJ-1** Wire `measurement_commits` to the nine chunk payloads it was written to consume. The
  artifact must not ship `measurement_git_commits: []`. **A test must fail if the list is empty when
  chunks exist.**
- **MJ-2** `test_recomputing_the_rtg_summary_on_the_wrong_population_disagrees` must **do what it
  says** — recompute on the 200-stream subsample for `random` and assert **disagreement** — or be
  deleted and the packet's `:367` claim retracted. **Do not leave a test whose only live assertion
  compares a default-argument call to itself.**
- **MJ-3** Record both measured blind spots in the plan's §2 list: the argmax-preserving logit scale
  (`× 1.5`, ATT bit-identical) and **`dropout=0.9` passing the identity test**. ⚠️ **The second is the
  one that bites: P5.3b trains, and the identity test is blind to every field that only matters at
  training time.** Hand it forward explicitly.
- **MJ-4** **A test must read `docs/data/p5_3a_rtg_probe.json` itself** and pin: 40 cells · the 12
  declared interventions and no others · **R6's exact zero** · **A8's `fixedtime` prediction** · the
  crosscheck's two ATTs and flip rate · row B's ordering with `random` narrowest · verdict-freedom of
  the shipped file. Every comparable artifact in this repo is read by a test.
- **MJ-5** Correct the hand-forward: `offline/dt_gate.py:749-754` has **no `rtg_mode` parameter**, so
  P5.3b cannot "pass `rtg_mode="zero"`" today. Say what is true — the mechanism is delivered on the
  config/inference side, and **P5.3b's first commit is threading `rtg_mode` through `train_dt`.**
  ⛔ **Do not add it here; the brief forbids touching the training path.**

## 4. MINOR — fix M1, M3, M4, M7, M8; record M2, M5, M6

**M1** delete the hardcoded `"rtg_summary_routes_agree": True` or make it computed · **M3** name the
population on rows A and B · **M4** emit the across-seed spread in the artifact, not only in the
packet · **M7** assert the single-intersection contract in `crosscheck` instead of assuming it ·
**M8** fix the annotation. **Record, do not fix:** **M2** (seed 101 only — state that the five agree
and that this was checked), **M5** (**device sensitivity past ~8 significant figures — this belongs in
the artifact, since the TVD values are emitted at full float64**), **M6** (no `git_dirty` flag,
pre-existing). **M9 is the coordinator's** — §6's checkbox, ticked in the merge commit.

## 5. The eight theatre tests

**Each one: make it discriminate, or delete it and say why.** Two are already honest and may stay with
their docstrings unchanged — **#6** (A3's non-discriminating line, which A3 itself required be
disclosed) and **#7** (the inverted `DEFERRED` 56 test, disclosed in its own docstring). ⚠️ **#7 is
nonetheless the *condemns-a-correct-artifact* class this project refused on 2026-08-19; keep it only
if its docstring names the ruling it is an exception to.** **#3 is the exact shape this project's own
reviewer stripped from `test_erfc_determinism.py`** — delete the three implied assertions.

## 6. Definition of Done for the fix round

- [ ] BL-1 and BL-2 closed, with **mutation evidence and every failure pasted in full** (P5.2's MN-6:
      3 of 22 pasted is not acceptable)
- [ ] All five MAJOR closed; minors fixed or recorded as listed
- [ ] The eight theatre tests each fixed, deleted, or justified in one sentence
- [ ] **§C's n = 500 table in the packet**, and *"weak lever"* and *"inert"* absent from it
- [ ] **Test count goes up or stays level.** A drop means a test was deleted, not repaired — say which
- [ ] `tests/test_erfc_determinism.py` count `322 → 330`, with the authorisation of §B quoted
- [ ] Whole suite green, pinned, with the real tail pasted; guards 16 / English / frozen re-run
- [ ] Manifests re-verified; `docs/reviews/P5.3a.md` answered **finding by finding** in the packet
- [ ] AI-assistance record updated for this round

**No re-review is required if BL-1 and BL-2 close with pasted mutation evidence.** The coordinator will
verify the two blockers and merge. ⚠️ **If closing BL-2 moves any reported number, stop and report —
that would mean the instrument was wrong, which is a different task.**

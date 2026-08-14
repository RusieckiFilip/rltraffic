# BRIEF 19 — P4.7: the mixture tiers (P4.6 phase 2)

**Mode:** Claude Code · **Branch:** `task/p4.7-mixture-tiers`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p47 -b task/p4.7-mixture-tiers main`
**Read first, from disk:** `PROJECT_PLAN.md` **§1 (all claim constraints) and §1b (the post-hoc reading
P4.6 produced)**, then `docs/reviews/P4.6.md`, then `docs/briefs/BRIEF_17_p4.6_method_tier_grid.md`
**§4, §11 and §12** — **those four sections bind this task and are not restated in full here.**

⚠️ **Absolute paths in every command** · **pin threads on every job** · **never write "MADT"** (C9).

---

## 1. Why this task exists, and why it is not optional

**§1's claim constraint (a) currently forbids attributing ANY C1 ranking to data quality alone**, because
our single-controller ladder varies quality and state coverage together. **A mixture varies composition
at roughly constant coverage. This task is the design that separates them** — so it does not add a
result, it **unlocks the interpretation of a result we already have.**

It is also **the only regime OffLight's data speaks to** (their Fig. 8 uses exactly 33 / 50 / 67 %
expert + random, read first-hand — `DEFERRED` 40), making our composition axis directly comparable to
the closest published work.

⭐ **And it is where §1b's R2 OPEN QUESTION becomes answerable.** P4.6 measured that %BC **changes sign**
— better than its behaviour policy on `mappo1000`/`mappo500`/`maxpressure`, **worse** on `fixedtime`
(+13.9285) and `random` (+3.3013). The reading is *"the filter selects the best behaviour mode
available, and on a uniformly weak single-controller tier there is no mode, so it selects noise."*
⚠️ **P3 was the diagnostic for that and it FAILED on `random`** (overlap 3/20, p = 0.3213), **so what
the filter selects on weak data is currently unknown.** **A mixture is the one corpus where what it
*should* select — the expert fraction — is known by construction.**

**Sequencing:** `PROJECT_PLAN` §10's ruling of 2026-08-14 makes this **item 1 of four and the one item
that is NEVER cut.** ≈5 h.

## 2. ⚠️ THIS IS MOSTLY A RUN, NOT A BUILD — AND VERIFY THAT BEFORE BELIEVING IT

**Measured on `main` today:** `offline/method_tier_grid.py` already declares `mix33`, `mix50`, `mix67`
as `TierSpec`s with `subsample="mixture"`, `MIXTURE_EXPERT_FRACTION = {0.33, 0.50, 0.67}`,
`MIXTURE_RNG_BASE = 20_260_813`, `mixture_training_streams()` (*"round(count × fraction) expert, the
rest random"*), and a diagnostic that already splits **by dataset directory — expert against random —
and by behaviour seed**, which is §11.A5's both-axes requirement already implemented.

🚨 **BUT P4.6's independent reviewer listed this code under "what I could not verify": *declared and
unit-tested, never executed.*** **An unexecuted path is exactly where a defect hides, and a unit test
written by the author of the path is the weakest evidence available.**

> **REQUIRED FIRST COMMIT, before any training: exercise the mixture path end-to-end on a throwaway
> tier and prove the composition it actually produces.** Assert, from the built stream list rather
> than from the spec: the **count** is 200, the **expert count** is `round(200 × f)` — **66 / 100 / 134**
> — the expert streams come only from the five `mappo1000` dirs, the random streams only from
> `cf_hz1x1__random`, and the selection is **reproducible** under the declared RNG. **If any of that is
> wrong, STOP and report it — it is a finding about merged code, not a fix to make quietly.**

## 3. ⚖️ RULING — the mixture tiers' BEHAVIOUR REFERENCE, because there is no obvious one

**Measured today: `BEHAVIOUR_REFERENCE_BY_TIER` covers all five single-controller tiers and NONE of the
three mixtures.** That is correct as it stands — **a 33 %-expert corpus is a *composition*, not a policy
anyone ran**, so there is nothing to roll out.

**This matters because "did the method beat the data it was given?" is C1's question**, and without a
reference the mixture rows cannot answer it. **Substituting a training-draw number is VOID under A5**
(`BRIEF_17` §11.A3, and P4.6 shipped that exact bug into a log).

> **RULED — CONSTRUCT the reference, do not roll it out, and do not average it into existence.**
> `mixture_training_streams` builds a corpus of **exactly** `round(200 × f)` expert streams and the
> rest random — a **fixed composition, not a coin flip.** So the matching held-out reference is built
> the same way: **for each of the 5 seeds, assign exactly `round(100 × f)` of the 100 held-out draws to
> the `mappo1000` cell and the remainder to the `random` cell, by a DECLARED RNG, and read the stored
> per-draw `att_horizon` from the committed cells.** This costs **zero compute** — both components are
> already measured on draws 1000–1099 at 5 seeds — and it is **paired by draw and A5-compliant by
> construction.**

**Three constraints on it, all required:**
1. **Label it `constructed`, never `measured`**, in the artifact, with the RNG seed and the realised
   per-seed draw assignment recorded. A reader must be able to tell it apart from a rolled-out cell.
2. **A CONSTRUCTED REFERENCE IS A REALISATION, NOT AN EXPECTATION — and that is deliberate.** Taking
   `f × ATT_expert(d) + (1−f) × ATT_random(d)` per draw would give the same *mean* and **an
   understated variance**, because it removes the composition's own randomness. **A paired CI computed
   against an expectation would overstate precision. Build the realisation.**
3. **Report the two component cells beside it as a bracket** — `mappo1000` **105.5820** and `random`
   **428.8839** — so the reference is legible as an interpolation between two measured endpoints.

## 4. Registered predictions — into `docs/plans/p4.7.md` BEFORE THE FIRST GRADIENT STEP

Per **A8(a)**. Falsifiable, and **none may be revised after any number exists.** ⚠️ **P4.6's three
registered predictions ALL FAILED and one of them was then rescued by a narrative that did not survive
review. Expect these to fail too, and do not build a rescue.**

> **Q1 — the filter FINDS the expert fraction: on all three mixtures, ≥ 90 % of %BC's kept top-decile
> streams come from the `mappo1000` dirs.** This is the direct test of R2's mechanism and it is
> leakage-free — it costs no training and reads only the selection.
>
> **Q2 — %BC's advantage over BC is POSITIVE on all three mixtures, and DECREASES as the expert
> fraction RISES.** At 33 % BC's training set is most diluted, so the filter has the most to recover;
> at 67 % BC is already mostly cloning expert data.
>
> **Q3 — the kept set's COMPOSITION signature is strong where P4.6's DEMAND signature was absent.**
> Run `BRIEF_17` §6's two P3 diagnostics unchanged per mixture tier. Prediction: the composition split
> separates sharply while the arrival-volume and MaxPressure-difficulty checks stay near null —
> **i.e. on a mixture the filter selects MODE, not DIFFICULTY.** ⚠️ **A null composition signature
> would falsify R2 outright and is the most informative outcome available here.**

## 5. Design constraints

1. **Size-match: 200 streams per mixture tier**, as every phase-1 tier. `BRIEF_17` §11.A1's
   one-episode-per-draw rule governs the **random** component.
2. ⚠️ **`DEFERRED` 28's confound is INHERITED and must be stated, not fixed here:** `mappo1000`'s five
   seeds occupy **disjoint contiguous 40-draw blocks**, so a mixture's expert half carries a
   seed-by-demand confound it did not create. **Report which seeds and which draw blocks each mixture's
   expert component drew from.**
3. **Normalisation statistics fitted per tier on that tier's own training split**, as in phase 1.
4. **Same 100 held-out draws, same 5 seeds, same `evaluate_arm`, same 40,000 steps.** No arm gets a
   budget the others do not.
5. **DT prompt: that tier's `max(training-split return)` computed over the MIXTURE'S ACTUAL STREAMS,
   after composition** (`BRIEF_17` §11.A4). A mixture's max will sit near the expert component's — say
   so, and do not treat it as a tuned prompt.
6. **NO EQUIVALENCE VERDICTS** (`BRIEF_17` §4). Paired difference, CI, CI **width**, rank-biserial.
7. **Phase-1 cells are RE-USED, never re-run**, and gated as `mappo1000`'s were: **the 20 phase-1 cells
   must regenerate bit-identically** or the task is BLOCKED.

## 6. Also in scope

- **`DEFERRED` 44's remaining sites** — the P4.3-module guards (`ci95_half_width` surviving ×4.0, Gate
  A's two refusals, `probe_artifact`'s disjointness, `evaluate_point`'s support-range check).
  ⚠️ **This family is at FIVE sightings and rule 44 already said stop queueing at the fourth. Close
  the remainder here.**
- **`DEFERRED` 43's remainder** — the fragile `match=` sites outside `offline_baselines.py`.
  **Measure first; change only the non-unique tokens; paste the table.** Not a mass rewrite.

## 7. Definition of Done

- [ ] §2's mixture-path verification committed **first**, with the realised composition asserted from
      the built stream list
- [ ] `docs/plans/p4.7.md` committed **before any training**, carrying §4's three predictions verbatim,
      §3's constructed-reference RNG and assignment, and the per-tier DT targets
- [ ] 12 new cells (3 tiers × 4 methods × 5 seeds), phase-1's 20 re-used under a bit-identity gate
- [ ] The constructed behaviour reference, labelled `constructed`, with its bracket
- [ ] Q1–Q3 scored by rules fixed in the plan; **report failures as failures and build no rescue**
- [ ] `DEFERRED` 43's remainder and 44's remaining sites closed
- [ ] Every mutation executed and **its failure pasted**
- [ ] Campaign run in a **user-launched `tmux`** session (`BRIEF_17` §12, all six conditions)
- [ ] Full suite green, tail pasted, pinned state stated; **all three guards read WITHOUT a pipeline**
      (`$?` after a pipe reads the last command's status — this bit P4.6)
- [ ] Return Packet at `docs/returns/P4.7.md` with the AI-assistance record
- [ ] §6's checkbox left unticked; it is mine, in the merge commit

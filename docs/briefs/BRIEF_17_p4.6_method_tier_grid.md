# BRIEF 17 — P4.6: C1's actual experiment, the method × tier grid

**Mode:** Claude Code implementation session · **Branch:** `task/p4.6-method-tier-grid`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p46 -b task/p4.6-method-tier-grid main`
**Read first, from disk:** `PROJECT_PLAN.md` §1 — **all four claim constraints**, they bind this task
more than anything below — then `docs/returns/P4.4.md` §12.8, `docs/returns/P4.5.md` §0, and
`docs/reviews/P4.3.md` §2.

⚠️ **Absolute paths in every command; a relative-path manifest only verifies from its own directory**
(`§7`, four instances). ⚠️ **Pin threads on every job** (`DEFERRED` 41). ⚠️ **Never write "MADT"**
(`CONTRACTS` C9).

---

## 1. Why this task exists

**C1 is the data-quality ladder, and it currently has comparators at exactly one rung.** P4.4 measured
BC, %BC and IQL against the DT on `mappo1000` — **one column of a grid we have claimed since
2026-07-10.** This task fills the rest.

It is also the **September deliverable**. `§10`'s governing constraint names *"C1 ladder + P4"* as the
guaranteed target, and this is the last piece of it.

**The corpus already contains the ladder** — measured, not assumed:

| tier | streams | max return (the DT's per-tier target) | mean ATT |
|---|---|---|---|
| `mappo1000` | 200 | −5762 | **105.46** |
| `mappo500` | 200 | −6362 | 107.50 |
| `mappo200` | 200 | −8210 | 125.03 |
| `maxpressure` | 200 | −13112 | 176.50 |
| `fixedtime` | 200 | −29707 | 262.09 |
| `mappo060` | 200 | −17310 | **281.89** |
| `random` | 400 | −38369 | 422.52 |

⚠️ **Note `mappo060` at 281.89 is WORSE than `fixedtime` at 262.09.** An under-trained learned policy
loses to a fixed schedule, so **"tier index" and "data quality" are not the same ordering** — order
every figure and every claim by **measured ATT**, never by tier name or training budget.

## 2. The two axes, and why the second is nearly free

**Axis 1 — policy identity (single-controller tiers).** What the corpus was collected as.

**Axis 2 — composition (mixture tiers).** ⚠️ **Built by RESAMPLING episodes we already own — no new
collection, no new rollouts.** A mixture tier is a declared selection over existing `.npz` files.

**Both are needed, because they are not the same axis and §1 already registers the difference.** Our
single-controller ladder varies **quality and state coverage together** — `mappo1000` is high-quality
and narrow, `random` is low-quality and broad. A mixture varies **composition at roughly constant
coverage**. **No claim may attribute a ranking shift to quality alone**, and the mixture axis is what
lets the two be separated.

**Use OffLight's fractions — 33 / 50 / 67 % expert, remainder random** (read first-hand from
`arxiv.org/html/2411.06601v3`; their Fig. 8 uses exactly these, expert + random). That makes our
composition axis **directly comparable to the closest published work** at no extra cost.

## 3. Scope, in two phases — phase 1 is the deliverable

**Phase 1 (critical path).** Four methods — **BC, %BC, IQL, DT** — across **five single-controller
tiers**: `random`, `fixedtime`, `maxpressure`, `mappo500`, `mappo1000`.
⚠️ **`mappo1000`'s column already exists** — BC/%BC/IQL from P4.4, the DT from P4/P4.3. **Re-use it,
do not re-run it**, gated the way P4.5's Gate B gated its re-use of `bc_top10`. So phase 1 is
**16 new cells**, ≈7 h.

**Phase 2 (strongly desirable, separable).** The three mixture tiers × four methods = 12 cells, ≈5 h.
**Phase 1 must stand alone if phase 2 is not reached** — write the artifact so it is complete after
phase 1 and extended by phase 2, never so that phase 1 is meaningless without it.

**Out of scope entirely:** `mappo200` and `mappo060` (a later extension), any other scenario, SUMO,
CQL, any change to the DT architecture.

## 4. ⚠️ NO EQUIVALENCE VERDICTS IN THIS TASK — and this is a ruling, not an omission

A6's **δ = 0.6263** is `mappo1000`-specific: it is the DT's measured margin over its behaviour policy
*on that tier*. **There is no non-circular way to derive a per-tier δ before the run** — deriving it
from the DT's margin requires the DT's result, which is what we are measuring.

> **Ruled: P4.6 reports paired mean differences, their 95 % CIs, the CI WIDTH and the rank-biserial
> effect size for every pair, and issues NO equivalence verdict anywhere.**

**Inventing seven or eight new thresholds to defend would be worse than having none**, and our own
doctrine already prefers the continuous quantity: a CI width converts a failure to reject into a
bound, which is what a reader needs and what a verdict compresses away. **If a later task wants an
equivalence claim on a tier, it declares δ for that tier before running it.**

## 5. The DT's prompt — one declared rule per tier, and no sweep

Each tier's DT is conditioned on **that tier's** naive target: `max(training-split episode return)`,
the values in §1's table. **Declared, computable before any evaluation, and no sweep.**

**P4.3 is why a sweep would be waste**: over a 13,000-wide grid the prompt moved ATT by **0.9026
total**, with only 3 of 8 adjacent steps resolving. A per-tier sweep would cost ~10× the compute to
chase an effect smaller than the differences this grid exists to measure. **Report that reasoning in
the packet** — it is a decision the compute budget rests on.

⚠️ **State the known limitation with it:** on a weak tier the naive target asks the model to achieve
*that tier's* best, which may be poor in absolute terms. **That is the correct behaviour of the naive
rule and is exactly what the ladder is testing**, but a reader must not mistake it for a tuned prompt.

## 6. Registered predictions — into `docs/plans/p4.6.md` BEFORE the first gradient step

Per **A8(a)**'s discipline. All three are falsifiable and none may be revised after any number exists.

> **P1 — BC's rank among the four methods worsens as tier quality falls.** Basis: OffLight's Table II
> (read first-hand) has BC **worst of the offline methods in all nine cells** on a six-controller
> heterogeneous mixture, while our P4.4 has BC **matching** the DT on pure expert. Those are the two
> endpoints; this predicts the interior.
>
> **P2 — %BC's advantage over BC is LARGEST on the heterogeneous mixtures and SMALLEST on `random`.**
> On a mixture the top-decile filter selects the expert fraction; on pure random there is no expert
> fraction to find.
>
> **P3 (leakage-free, and it costs no training) — on `random`, %BC's top decile carries a DEMAND
> signature; on `mappo1000` it did not.** P4.5 established the pure-expert instantiation: the filter
> selected the two best *checkpoints*, arrivals were equal (443.6 vs 446.1) and only 4/20 of the kept
> episodes overlapped MaxPressure's difficulty ranking. **Run the identical two checks per tier.**

**These three are one hypothesis with three instantiations:** *the top-decile filter selects the best
behaviour MODE available, whatever a mode is in that corpus* — best **seeds** on pure expert
(measured), best **controller** on a mixture, easiest **demand draws** on pure random.

## 7. Design constraints that decide whether the result means anything

1. **Size-match every tier's training set**, or P4.5's confound returns wearing new clothes. `random`
   has **400** streams and the rest have 200 — **subsample it to 200 with a declared RNG, and record
   the selection.** Assert equal training-set size across tiers in the artifact.
2. **Mixture tiers are size-matched too** — 200 streams, composed at the declared fraction.
3. **Normalisation statistics are fitted per tier on that tier's training split** and carried into
   every checkpoint. Declare it; a shared statistic across tiers would be a leak between arms.
4. **Same 100 held-out draws, same 5 seeds, same `evaluate_arm`, same 40,000 steps** as every earlier
   task. **No arm gets a budget the others do not.**
5. **Gate the re-used `mappo1000` column** the way P4.5's Gate B did: canonical digests plus a
   declared sample of cells re-rolled for exact equality, **mismatch ⇒ BLOCKED**.

## 8. Also in scope — two queued items this task owns

- **`DEFERRED` 42** — four guard branches in `offline/offline_baselines.py` survive being disabled.
  ⚠️ **This family is on its third queue entry (33, 42, 44). Fix it here rather than queue it again.**
- **`DEFERRED` 43** — the fragile `match=` tokens in `offline_baselines.py`'s tests. **Measure first,
  change only the non-unique ones**, and paste the measurement table; the fix is the measurement.

## 9. Definition of Done

- [ ] `docs/plans/p4.6.md` committed **before any training**, with §6's three predictions verbatim,
      the mixture construction, the per-tier targets and the subsampling RNG
- [ ] Phase 1 complete: 16 new cells + the gated re-use of `mappo1000`'s column
- [ ] Equal training-set size across all tiers, **asserted from the artifact**
- [ ] Paired differences, CIs, CI widths and effect sizes for every pair; **no equivalence verdict**
- [ ] P3's two diagnostics run **per tier**, reported whichever way they come out
- [ ] `DEFERRED` 42 fixed and 43 measured
- [ ] Every mutation executed and **its failure pasted**
- [ ] Full suite green, tail pasted, stating whether it was pinned; all three guards 0
- [ ] Return Packet at `docs/returns/P4.6.md` **with the AI-assistance record**
- [ ] §6's checkbox left unticked; it is mine, in the merge commit

## 10. What I will do with the result

**This is the C1 figure.** Whatever shape it has, it is the paper's central experimental panel, and
every outcome is publishable: a clean crossing tells us where sequence modelling starts to pay; a flat
grid tells us data composition dominates architecture, which is the thesis P4.4 and P4.5 have been
pointing at; and BC winning everywhere would be the strongest negative result in the offline-TSC
literature and would still be ours to report.

⚠️ **What it may NOT be used for**, because §1's constraints already bind: it may not attribute a
ranking shift to *quality* alone (quality and coverage move together on axis 1); it may not present
the DT's prompt as tuned; and it may not claim novelty for the ladder concept itself, which
`arXiv:2112.02845` established in another domain in 2021.

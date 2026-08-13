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

---

## 11. ADDENDUM, 2026-08-13 — five findings from auditing this brief against the artifacts

**Written after the brief was issued and before the plan exists.** I measured the things §1–§10
asserted instead of trusting them. §1's table survived exactly; four other statements did not, and one
of the four is mine in `PROJECT_PLAN` §6. **This section overrides anything above that conflicts with
it.** All five are *tightenings before any number exists*, which `PROJECT_PLAN` §7 (2026-08-12) makes
free by construction.

### A1. Subsample `random` by DRAW, one episode per draw — not uniformly over the 400

**Measured** (`datasets_v11/cf_hz1x1__*/manifest.json`, all 23 hz1x1 tier dirs, 2026-08-13):
`cf_hz1x1__random` holds **exactly 2 episodes for each of draws 1–200**; every other tier holds
exactly 1 per draw over the same range 1–200. §7.1's *"subsample it to 200 with a declared RNG"*
permits a uniform 200-of-400, which would give **uneven draw coverage** and make `random` the only
tier whose training demand distribution differs from the other four.

> **Take exactly one episode per draw, for all 200 draws; the choice within each pair is the declared
> RNG's only job. Record the selected `episode_sha256` list in the artifact.**

This is size-matched **and** demand-matched, at no cost. **Measured impact on the DT's declared target
(A4 below): at most 364 return units** — worst case `−38733` against the all-400 max `−38369`, i.e.
**≤ 19 % of the tier's own 1925-wide return spread**, and over 2000 simulated RNG draws the **median
outcome leaves the target unchanged**. So the tightening is free in effect as well as in cost.

⚠️ **Note for the packet:** the five MAPPO seeds occupy **disjoint contiguous 40-draw blocks**
(`seed101`→1–40 … `seed505`→161–200), verified here. That is `DEFERRED` 28's confound, it is inherited
rather than introduced, and it means a MAPPO tier's "seed" and "demand block" cannot be separated.
**State it; do not try to fix it here.**

### A2. `PROJECT_PLAN` §6 says this is "a config loop". That is HALF TRUE, and the false half is mine

Training and evaluation *are* a config loop — the tier enters through `--dataset-dir`. **The reporting
path is not.** Read from `offline/offline_baselines.py` at `cacf5f8`:

- **`:2367`** — `_run_report` **hard-requires the `mappo1000` arm** and raises without it. On a
  `random` tier the behaviour policy is `random`, so the report cannot be built as written.
- **`:2376`, `:2410–2413`** — `recovered_fraction` is defined against `mappo1000` by name.
- **`:2383–2428`** — an equivalence **`verdict` is emitted for every method unconditionally**, under
  `DELTA_ATT = 0.6263`. **§4 of this brief forbids exactly that**, so the ruling in §4 requires a code
  change; it is not satisfiable by configuration.
- **`:208`** — `CITED_ARMS = ("madt", "mappo1000", "maxpressure")`.

> **Generalise the reporting path to a per-tier behaviour reference and a no-verdict mode.
> REGRESSION GATE, non-negotiable: `docs/data/p4_4_baselines.json` must regenerate BYTE-IDENTICALLY
> through the generalised path.** If it does not, the generalisation changed P4.4's merged numbers and
> the task is BLOCKED, not patched.

**This is the project's signature error, committed by me in the plan:** a true statement about one
function (`evaluate_arm` does take arbitrary arms) written as a description of the whole pipeline.

### A3. TWO OF THE FIVE TIERS HAVE NO BEHAVIOUR REFERENCE ON THE HELD-OUT POOL — and A5 forbids the substitute

**Measured** from the merged artifacts: `docs/data/p4_gate.json` and `docs/data/p4_4_baselines.json`
carry held-out cells for **`mappo1000` (n=500), `mappo500` (n=500), `maxpressure` (n=100)** and for no
other behaviour policy. **`random` and `fixedtime` have none.**

C1's most valuable per-tier sentence is *"does an offline method beat the policy that produced its
data?"* — and on `random` that is the headline of the whole ladder. Substituting the tier's
training-draw ATT (422.52 / 262.09, draws **1–200**) is **VOID under `PREREGISTRATION` A5**: every
reported comparison must be over **shared draw ids**, and 1–200 against 1000–1099 is not.

> **In scope for phase 1: evaluate `random` and `fixedtime` on the SAME 100 held-out draws, through
> the SAME path.** `offline/dt_gate.py:753 evaluate_arm` already takes an arbitrary
> `choose_action_factory` (`:1054 _maxpressure_factory` is the model to copy), so this is two factories,
> not a harness. Cost at P2.0b's measured hz1x1 rollout of 0.59–0.83 s/episode: `fixedtime`
> deterministic → 100 episodes ≈ **80 s**; `random` stochastic → 5 seeds × 100 ≈ **7 min**.

⚠️ **Both factories must reproduce the policy that COLLECTED the tier, not a fresh implementation of
the same idea.** This is amendment **A2**'s error class — a hand-rolled random sampler is *a different
realisation of the random policy* and its number is not comparable to the corpus's.
- `random` → the semantics of `offline/collect.py::_make_random` / `_random_legal_action`.
- `fixedtime` → `offline/policies/fixed_time.py::make_fixedtime`, **and read `k` FROM THE TIER'S
  MANIFEST.** ⚠️ **`cf_hz1x1__fixedtime` was collected at `fixed_time_k = 6`** with
  `fixed_time_schedule_source = "shipped_plan"` and `fixed_time_plan_sha256 = 1b0aa65a…` — measured
  from the manifest today. **`PROJECT_PLAN` §6's P2.5 entry says "Ships `k=4`", which is true of what
  P2.5 shipped and FALSE of what this corpus used.** Assume k=4 and the reference line is a policy
  that never generated any of these episodes. **Assert the manifest's `fixed_time_plan_sha256` matches
  the plan the factory loads.**

### A4. The DT's per-tier target is computed AFTER subsampling, over the actual training set

§5 says *"that tier's naive target: `max(training-split episode return)`"*, and §1's table gives it
per tier. **Verified today — every value in §1's table reproduces exactly** from
`manifest.json::episodes[].total_global_reward` and `att_per_step[-1]`: max returns
−38369 / −29707 / −13112 / −17310 / −8210 / −6362 / **−5762** and mean horizon ATT
422.52 / 262.09 / 176.50 / 281.89 / 125.03 / 107.50 / 105.46. `mappo1000`'s **−5762 is exactly P4's
naive target**, which confirms the field.

> **But `random`'s −38369 is the max over 400 episodes and the training set will be 200.** Compute
> each tier's target over the **episodes actually trained on**, so the prompt is in-support by
> construction, and record it in the plan file before the first gradient step.

### A5. On a mixture tier, report what the filter selected on BOTH axes

P4.5 measured that `mappo1000`'s five seeds span **3.8190 ATT**. So on a mixture the top-decile filter
can select the expert **fraction** *and*, within it, the best **seeds** — and §6's prediction P2 reads
only the first. **Report the kept set's composition as expert-vs-random AND as a seed histogram**, the
way F3 did (`{202:10, 101:9, 505:1, 303:0, 404:0}`), or *"the filter selects the expert fraction"* is
confounded with the checkpoint selection P4.5 already established.

### Scope, restated honestly

Phase 1 is now **16 cells + 2 behaviour cells + the gated re-use + a generalised reporting path**,
plus `DEFERRED` 42 and 43. **That is a full task.** If phase 2 (the mixture tiers) is not reached,
**it becomes P4.7 and that is the expected outcome, not a failure** — §3 already requires phase 1 to
stand alone, and A5 above is written so P4.7 inherits it.

---

## 12. RULING, 2026-08-13 — THE CAMPAIGN RUNS IN `tmux`, LAUNCHED BY THE USER. NOT AS SESSION BACKGROUND JOBS

**Ruled deliberately for this task at the user's request, not inherited from P4.3.** The user asked
for the ruling to be made rather than carried over, and it goes the way they proposed.

⚠️ **This was never a free choice: `CLAUDE.md` §5 already binds it** — *"Long simulation runs (corpus
collection, MAPPO training) are **not** run inside a Claude Code session — they go to a `tmux` session
started by the user. You may read their logs."* Phase 1 is ≈7 h of training. **§1–§11 of this brief
were silent on a rule that already applied, and that silence is my omission.**

**Why it is right on its own merits, beyond the rule.** Crash-resistance is *not* the reason — the
per-cell artifact condition is the stronger protection and it worked cleanly across P4.3's ten points.
The reason is **session cost**: P4.3's implementer spent an hour of context polling, and P4.4's `until`
loop hung on a self-match until the user killed it. **P4.6 is the largest campaign in the project, so
that cost scales with it.** In `tmux` the campaign is independent of the session, so the implementer
can `/clear` and return for the artifacts instead of holding context open to watch a progress bar —
which is `CLAUDE.md` §0's context-discipline rule (*"the repo is the memory; the context window is
scratch space"*) applied to a 7-hour job.

### Six binding conditions, because `tmux` is WEAKER than a background job in one specific respect

**Nobody observes a `tmux` pane's exit status.** §7's *"Verify by EFFECT, not by STATUS"* and
*"Campaign scripts must abort and must self-verify"* (2026-08-06) exist because the ATT campaign
*"ran on to a clean-looking end"* with half the corpus missing. Detaching the campaign from the
session removes the one reader who would have noticed. So:

1. **The implementer WRITES the campaign script; the USER launches it. The implementer never launches
   it and never `sleep`-polls it.** Read its log, read its artifacts.
2. **`set -euo pipefail`, and a final assertion that completed cells == requested cells, exiting
   non-zero on mismatch.** This is §7's 2026-08-06 rule, which has been violated once already.
3. **The thread pin is exported in the `tmux` shell AND re-asserted inside the script**
   (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`). `DEFERRED` 41 has **two sightings**, one of them inside
   `pytest`; an unpinned 7-hour job that wedges at ~0 % CPU costs the whole campaign, and exporting it
   once in the shell converts *"remember to pin each job"* into *"the environment is pinned"* —
   mechanical enforcement over intention, which is this project's standing preference.
4. **The script is RESUMABLE: skip any cell whose artifact already exists, and log every skip by
   name.** ⚠️ **This is the condition that makes `/clear`-and-return real rather than aspirational** —
   without it, "come back for the artifacts" means "come back and work out which cells ran". No prior
   campaign here has been resumable; P4.4 chunked manually and merged with `merge_training_runs`.
5. **The per-cell artifact rule STAYS.** `tmux` does not replace it; the user said so and is right.
6. **Phase 1 and phase 2 are SEPARATE scripts**, so phase 1's completion is not entangled with a phase
   that may become P4.7.

### Two consequences to record rather than discover

- **Provenance: a `tmux` campaign is maximally chunked, and that problem is already solved.**
  `DEFERRED` 39 (a single `runtime.git_commit` cannot express a chunked campaign) landed in P4.3 as a
  **measurement-provenance / written-at-commit split**. **Use it from the start here.** Measured
  precedent for why: `output/p4_4/gate_a.json` carries `738884b` while its three `eval_*.json` carry
  `c13aaa9`.
- **The campaign writes to `/home/filip/rltraffic-p46/output/p4_6/`, which is gitignored.** Per the
  2026-08-11 rule and `PROJECT_PLAN` §10, **those checkpoints and raw JSONs must be secured into the
  `main` tree with a `SHA256SUMS_p4_6.txt` before that worktree is retired** — `git worktree remove`
  deletes them without warning.

**The Return Packet must state which cells came from which `tmux` run**, since the campaign is no
longer one process with one log.

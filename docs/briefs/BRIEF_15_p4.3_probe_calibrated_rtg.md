# BRIEF 15 — P4.3: probe-calibrated return prompting, ablated in-domain

**Mode:** Claude Code implementation session · **Branch:** `task/p4.3-rtg-calibration`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p43 -b task/p4.3-rtg-calibration main`
**Read first, from disk:** `PREREGISTRATION.md` **A8(a)** — it binds this task more than this brief
does — then `docs/returns/P4.md` §8 items 9 and 10, `docs/reviews/P4.md` MAJOR-1, and this file.

⚠️ **Use an absolute path in every command** (`PROJECT_PLAN` §7, three violations by the coordinator
on 2026-08-12/13). ⚠️ **Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on every job** (`DEFERRED` 41).
⚠️ **Never write "MADT"** (`CONTRACTS` C9) — it is *"the offline multi-agent Decision Transformer"*,
or *"the DT"*.

---

## 1. Why this task exists, and why it is now load-bearing

This is **the paper's named method component**. §1 narrowed its novelty on 2026-08-01: DTLight already
scales RTG within a domain, so **our contribution is the target-domain probe**, not return
conditioning as such.

It has also become the task that decides how P4.4's result may be written. P4's own independent review
recorded two things, both **before** any baseline existed:

- **The model is conditioned outside its training support for a fifth of every episode.** All training
  rewards are ≤ 0, so every training RTG is ≤ 0 — but at evaluation the RTG is driven by observed
  rewards from a declared start of −5762 and goes **positive for 75 of 360 decisions (20.8 %)** on the
  probed episode, with the scaled input spanning **−0.577 … +0.153**.
- **The declared target is not the best target.** On `(seed 101, draw 1000)`: declared `−5762` →
  **106.46**, `target_rtg = 0` → **102.05**, `−3000` → **102.41**. ⚠️ **n = 1, one episode, and it is
  not representative — the DT's arm mean is 104.9558, so that episode is 1.50 ATT worse than typical
  for it.** The direction is suggestive; the magnitude is not established. **Establishing it is this
  task.**

`docs/reviews/P4.4.md` §8.6 binds until this lands: **no DT-versus-baseline sentence enters the paper
until P4.3 has run**, because the swing above is larger than the effect P4.4 reports.

## 2. What makes this task cheap, and what makes it dangerous

**Cheap: nothing is retrained.** The RTG target is an *inference-time* parameter. P4's five DT
checkpoints are secured at `/home/filip/rltraffic/output/p4_dt/` (verified 5/5 by sha256 on
2026-08-11). A 9-point grid over 5 seeds × 100 draws is 4,500 rollouts ≈ **80 minutes**, and the
probe is 100 MaxPressure rollouts ≈ 2 minutes.

**Dangerous: this is the one task where selecting on the evaluation return would look like method
development.** D5 and `PREREGISTRATION.md` §6.1 already forbid it. **A8(a) restates it because here it
would be easy and it would be invisible in the result.**

> **The calibration rule is a declared function of the probe's return distribution, written into
> `docs/plans/p4.3.md` before any target-domain evaluation number exists. The sweep is the ABLATION
> AROUND that rule and never the mechanism that selects it.**

**A target chosen because it scored best on the held-out draws is not a method. It is a leak wearing a
mechanism's name**, and the paper would be claiming a mechanism it did not use.

## 3. The four questions, in this order

1. **What is the in-domain RTG landscape?** A declared grid, evaluated on the 100 held-out draws.
2. **Where does the NAIVE rule land on it?** `target_rtg = max training-split return = −5762`, which
   is what P4 reported.
3. **Where does the DECLARED CALIBRATION RULE land?** The test of the mechanism.
4. **What is the in-support fraction for each?** — see §5, and note it needs **no evaluation return at
   all**.

⚠️ **CORRECTED by §12.1 — this note was HALF WRONG and the half matters.** *"Any correct rule must
approximately reproduce the naive target here"* holds for the **probe-relative** form (Rule B), where
the cross-domain ratio is 1 in-domain, and **not** for the **probe-quantile** form: a quantile of
MaxPressure's returns targets **−13112**, i.e. MaxPressure's own performance, because MaxPressure is
not the behaviour policy. **Read §12.1 and §12.2 before planning around this paragraph.**

⚠️ **Be honest about what the in-domain case can and cannot test, and say so in the packet.** In-domain
the probe's *cross-domain* job is trivial: source and target coincide, so **any correct rule must
approximately reproduce the naive target here**, and the probe's real work is not exercised until
C3/P7. What this task *can* establish is (a) the shape of the landscape, (b) whether the naive rule is
a good point on it, and (c) whether the rule's form is computable and in-support. **If the sweep shows
the naive target is poor in-domain, then a rule that reproduces it in-domain is also poor — and that
is a finding about the mechanism, not a failure of the task.**

## 4. The probe set — disjoint by construction, and it must stay that way

`PREREGISTRATION.md` §5 registers: draw **0** nominal, **1–999** training pool, **1000–1099** held out.
The corpus used **1–200**. So **draws 201–999 are unused**, and the probe takes its episodes from
there.

**Required: the probe runs on draws `201 … 200+k`, never on 1–200 and never on 1000–1099.** Assert it
from the artifact, not from the command line.

**Use the k values C3 already registers — `k ∈ {5, 20, 100}`** — so this ablation answers *how many
probe episodes the rule needs* in the easy case, and the answer transfers to the transfer study.
Probe episodes are MaxPressure rollouts; 100 of them cost about two minutes.

## 5. The two criteria, and only one of them may select

> 🚨 **WITHDRAWN 2026-08-13, BEFORE ANY NUMBER EXISTED — see §12.1. The table below said the
> in-support fraction MAY select a rule. That is wrong, and it is measurably wrong.** The training
> RTG support is `[−9991.0, −6.0]`; `target = 0` sits above it for the entire episode (in-support
> **0.000**) and is the point P4's reviewer measured **best** at n = 1. **A criterion that would
> reject the apparently best point is not a weak proxy — on this evidence it points the wrong way**,
> because conditioning a return-conditioned model on an optimistic and therefore extreme return is
> *what the mechanism is*. **In-support is a RELIABILITY diagnostic and a caveat generator. It never
> selects.** The original text follows, struck through in substance.

| criterion | what it measures | may it select the rule? |
|---|---|---|
| ~~**in-support fraction**~~ | the fraction of decisions whose RTG input lies inside the training RTG range | ~~YES~~ → **NO** (§12.1). Reported for every point as a **reliability** diagnostic |
| **ATT on the held-out draws** | the outcome | **NO. It scores the rule and never chooses it.** |

**With both selectors gone, what makes the rule non-arbitrary is its FORM** — see §12.1. **Report the
in-support fraction for every grid point and for the rule**, with the training RTG range it is
measured against stated as a number: `[−9991.0, −6.0]`, measured over 72,000 rows.

## 6. Per-file requirements

**Do NOT modify `agent/DTAgent.py`.** Measured: `DTAgent.__init__` accepts `target_rtg` and
`rtg_scale`, but **`load()` overwrites both from the checkpoint payload** (`DTAgent.py:803-804`), and
`from_checkpoint(env, path, device)` takes no override. So the target must be set **after** load.

**Required, and this is the load-bearing engineering decision of the task:** one named function in
your own module that builds the agent from the checkpoint and then applies the target, and **a test
that verifies the override BY EFFECT — the value actually reaching the model — not by reading the
attribute back.** P4's review found three mutations of exactly this surface that survived 58 tests and
moved the reported number by up to **+32.5 ATT**, because no test ever built the agent the way
evaluation builds it. **That is the defect this test exists to prevent, and reading `_target_rtg`
back does not prevent it.**

**`rtg_scale` does not change in this task.** It is the normalisation divisor (`9991.0`); only
`target_rtg` varies. Declare that, and assert it in the artifact.

**New module** `offline/rtg_calibration.py`: the probe runner, the declared rule, the grid, the
in-support metric, the artifact and a CLI. **Reuse from the merged modules** — `evaluate_arm`,
`env_settings_from_manifest`, `mean_ci95`, `wilcoxon_signed_rank`, `paired_comparison`,
`write_json_atomic`, `assert_campaign_complete`, `pin_torch_threads`, `HELD_OUT_DRAWS`,
`TRAINING_SEEDS`. **Do not reimplement any of them.**

**Artifacts:** `docs/data/p4_3_probe.json` (the probe episodes and the rule's inputs) and
`docs/data/p4_3_rtg.json` (the grid, the rule's point, both criteria, the paired statistics).

## 7. Tests — red-first, each with its named mutation executed and pasted

| # | test | mutation that must kill it |
|---|---|---|
| T1 | **the load-bearing one:** the RTG the model actually receives at step *t* equals `(target − Σ rewards so far) / rtg_scale`, spied at `forward`, for every step of a real episode | apply the override before `load()` instead of after — the checkpoint value silently wins |
| T2 | the probe draws are disjoint from **both** 1–200 and 1000–1099, asserted from the artifact | shift the probe range by one |
| T3 | the in-support fraction matches an independent recomputation from the raw RTG trajectory | count `≥` instead of `>` at the boundary |
| T4 | the declared rule is a pure function of the probe returns — same probe input ⇒ same target, on a fixture with a known answer | let it read any evaluation quantity |
| T5 | `rtg_scale` is identical across every grid point and equals the checkpoint's | let it vary with the target |
| T6 | the grid is the **declared** grid — the artifact's points equal the constant, and an undeclared point is refused | derive the grid from the results |
| T7 | campaign integrity: completed runs equal requested, enumerated up front from the **declaration** | derive the expected set from the data (`PROJECT_PLAN` §7) |

**T1 is the one that protects everything else.** Every other test can pass while the override is
applied at the wrong moment, and then the whole grid is one target repeated.

## 8. Also in scope — two queued items this task owns

- **`DEFERRED` 31** — back-fill effect sizes into `docs/data/p4_gate.json`. §8 makes them mandatory
  beside every p-value and P4 reported none; the paired per-draw differences are already in the
  artifact, so this is arithmetic over committed data, **no re-run**. It is due *"before any P4 number
  enters the paper"*, and this task is that moment.
- **`DEFERRED` 39** — split `runtime_provenance()` into a frozen **measurement** provenance carried
  from the inputs and a separate **written-at** commit. You are writing new artifacts, so you can
  adopt the split rather than migrate to it. ⚠️ **This touches the merged `offline/dt_gate.py`. It is
  authorised here, dated 2026-08-13, additively only — no existing field may change meaning, and
  P4/P4.4/P4.5's artifacts must regenerate unchanged except for that field.** Prove it the way
  `BRIEF_12` §7.1 did: a recursive comparison against a pre-declared expected-difference set.

## 9. Scope fence

- **No retraining of the DT.** No new corpus, no new scenario, no SUMO, no transfer — C3 is P7.
- **No change to `rtg_scale`, to the architecture, or to the evaluation protocol.**
- **No selection of any reported target by evaluation return**, in any form, including "we looked at
  the grid and then wrote the rule".
- **Do not fix `DEFERRED` 42 or 43** beyond the sites you touch.
- **Do not edit** P4/P4.4/P4.5's committed artifacts, `offline/dataset.py`, `agent/DTAgent.py`,
  `agent/OfflineBaselines.py`, or any frozen path.

## 10. Definition of Done

- [ ] `docs/plans/p4.3.md` committed **before any evaluation number exists**, carrying the calibration
      rule in full, the declared grid, the probe design and the k values
- [ ] Probe run on draws 201+, disjoint from training and held-out, asserted from the artifact
- [ ] The grid and the rule's point evaluated on all 100 held-out draws × 5 seeds
- [ ] **Both criteria reported for every point**; the in-support fraction stated against the training
      RTG range as a number
- [ ] Every mutation in §7 executed and its failure pasted
- [ ] `DEFERRED` 31 and 39 addressed, with 39's regeneration proof pasted
- [ ] Full suite green, tail pasted, **stating whether it was pinned**
- [ ] All three guards exit 0; `git diff --stat` shows no frozen path
- [ ] Return Packet at `docs/returns/P4.3.md` **including the AI-assistance record** (`CLAUDE.md` §8;
      P4.5's was wrong in its first outing — determine authorship with
      `git log --diff-filter=A` and the creating commit's message, and state the method)
- [ ] §6's checkbox left unticked; it is mine, in the merge commit

## 11. What I will do with the result

**If the calibrated rule improves on the naive target**, the paper reports the mechanism with an
in-domain ablation behind it, and P4.4's comparison is re-run against the calibrated DT before any
DT-versus-baseline sentence is written.

**If it does not**, that is equally reportable and arguably more interesting: it would mean return
conditioning does not recover the DT's headroom in the easy case, and the honest paper says the
mechanism's value is confined to cross-domain transfer — where C3 will test it — rather than claiming
it generally. **Both outcomes are registered. Neither requires anything to be renegotiated.**

---

## 12. RULINGS on the plan of 2026-08-13 — **plan approved**, one design of mine withdrawn, four answers

**Your exploration earned a better task than the one I briefed.** All three findings verified here
independently before ruling.

### 12.1 🚨 §5's in-support selector is WITHDRAWN — my error, and you found it before it cost compute

**Measured here:** the training RTG support is `[−9991.0, −6.0]` (72,000 rows, your figure exactly).
`target = 0` is above it for the whole episode — **in-support 0.000** — and it is the point P4's
reviewer measured **best**. **A criterion that rejects the apparently best point is not a weak proxy;
it points the wrong way.**

**Why I got it wrong:** I conflated **reliability** (out-of-support inputs make behaviour
unpredictable — true) with **performance** (in-support targets do better — does not follow). For a
return-conditioned model it is close to backwards: **you condition on an optimistic, therefore
extreme, return on purpose. That is what the mechanism IS.** In-support is now a reported diagnostic
and a caveat generator, and it selects nothing.

⚠️ **And your Rule A has the mirror-image defect, which I also verified: `q = 1.00` of the PROBE's
distribution sets `target_rtg = −13112`** (MaxPressure's best of 200 draws), **7,350 below the naive
−5762 — it asks the DT to achieve MaxPressure-level return.** Your q=1.0-by-form argument is right
about the *functional* and wrong about the *distribution*: the naive rule is the max of the
**behaviour policy's** returns, and MaxPressure is not the behaviour policy. **So Rule A is not the
naive rule transported; it is a different and worse rule.**

**What this leaves, and it is the honest position:** with both selectors gone, the rule is justified
**by its form alone** — and only **Rule B** has the right form, because only Rule B estimates
*best-achievable* rather than *probe-achievable*:

```
target_target = R_probe_target x ( R_best_source / R_probe_source )
```

**In-domain the ratio is 1 by construction, so Rule B is an identity here and cannot be validated in
this task.** You were right that my §3 honesty note was half wrong; it is now fully stated.

### 12.2 So what P4.3 delivers, restated

**The mechanism cannot be validated in-domain. Its components can, and the landscape must be.**

1. **The landscape** — the declared grid. This is necessary regardless: it says whether the target
   matters at all, and it is the only thing that lets `docs/reviews/P4.4.md` §8.6 be lifted.
2. **The source-domain half of Rule B's ratio**, `R_best_source / R_probe_source`, measurable here and
   needed by P7. Report it with both a max-based and a mean-based probe statistic.
3. **Rule B named, and its in-domain identity ASSERTED by test** (§12.4).
4. **Rule A evaluated as a declared alternative, with the prediction registered NOW that it will be
   poor, and the reason: it targets probe performance rather than best-achievable performance.** A
   registered prediction that a plausible-looking rule fails, followed by its failure, is worth more
   than quietly not running it — it is the evidence that the mechanism must be **relative**.

### 12.3 Q1 — DEFERRED 31: **your reading is right and the contradiction is mine.** §8 said back-fill
into `docs/data/p4_gate.json`; §9 forbade editing P4's committed artifacts. **Ruled: the sidecar, as
you propose.** A new artifact carrying the rank-biserial and paired CIs recomputed from
`p4_gate.json`'s own records, with that file's sha256 recorded as the source and **the committed file
untouched.** Editing a merged, cited artifact to add fields would change its hash and force the whole
regeneration protocol for zero scientific gain. **No in-place edit is authorised.**

### 12.4 Q2 — **assert Rule B's in-domain identity in a test.** One line, no campaign, and it is the
difference between claiming a structural property and demonstrating it. It also pins the mechanism's
form for P7, which is the task that will actually exercise it.

### 12.5 Q3 — **in-session, with one condition.** ~12 jobs of ~9 minutes is within the `BRIEF_11` §7 Q8
precedent under which P4.4 ran a 25-minute Gate A. The risk is not wall time, it is losing completed
work: **each grid point writes its own artifact as it finishes**, so a crash or a `DEFERRED` 41
deadlock costs one point and not the campaign. Every job carries the thread pin and ends with the
completed-equals-requested assertion.

### 12.6 Q4 — **do not run the secondary quantiles.** Compute the target value each `q ∈ {0.5, 0.75,
0.9}` maps to — that is free — and report **where each falls on the grid**. Do not roll them out.
**The grid already spans the target space; the quantiles are a labelling of that space, not new
points in it**, so extra rollouts would re-measure what the grid measures. And a table nobody may
select from is not evidence — **it is temptation with a provenance**, and the first question a
referee asks of it is why it was computed and not used.

### 12.7 Gate A stays exactly as you wrote it

The `−5762` grid point is P4's configuration, so its 500 records **must reproduce `p4_gate.json`
per-episode by `==`, and nothing else is reported if it fails.** That is the same instrument check
that made P4.4 and P4.5 trustworthy, and it is worth more here than in either, because this task
changes the one parameter that configuration is defined by.

---

## 13. Mid-campaign, 2026-08-13 — one limitation that must be in the packet, and one lesson recorded

**Do not change the design. This is a disclosure requirement and a rule, nothing else.**

### 13.1 ⚠️ The best point so far is the GRID'S BOUNDARY — say so, and do not extend the grid

Recomputed here from your landed artifacts: `0 → 104.5564` · `−1000 → 104.6121` · `−2000 → 104.6928`
· `−5762 → 104.9558`. **Monotone, and the best point is `target = 0`, which is the most optimistic
point the declared grid contains.**

**If that ordering survives the remaining points, the optimum may lie OUTSIDE the grid** — at a
positive target, more optimistic still. **The packet must say so as a limitation.**

⚠️ **And it must say why the grid was not extended: extending it after seeing the landscape would be
selection on the result**, which A8(a) forbids and which is exactly the move this task exists not to
make. **A declared grid whose optimum sits at its edge is an honest limitation. A grid extended
because the edge won is a different study, and a worse one.** State the boundary, state the
monotonicity, and stop.

### 13.2 The n = 1 probe overstated the effect by an order of magnitude — quantify it

Naive → best is **0.3994 ATT** at n = 500 per point. P4's review measured **4.4 ATT** on one episode.
**The single-episode estimate was 11× too large.** Report that ratio explicitly: it is the most
transferable thing this task produces, and it is a measured instance of the error class this project
spends most of its effort on.

### 13.3 Consequence for `docs/reviews/P4.4.md` §8.6 — mine to rule, stated so you can see it coming

I blocked every DT-versus-baseline sentence *because* the recorded swing was 2.5× P4.4's effect. At
0.3994 it is **0.22×**. At your best point so far, **%BC is still ahead by 1.3937 and IQL by 1.0800.**
So the constraint is on track to be **lifted**, and the paper's sentence gets *stronger* rather than
weaker — *"the baselines beat the DT at every return prompt tested, over a declared grid at n = 500
per point"*. **I will rule when all ten points are in, not before.** Do not write that sentence
yourself; report the landscape and leave the lifting to me.

### 13.4 The harness that destroyed uncommitted work — your fix is right, and the rule is now §7's

`git checkout --` restores from `HEAD`, so it discards **every** uncommitted change, not only the
mutation. Your remedy — refuse to start on a dirty tree — is the better of the two available, because
it removes the situation instead of handling it. **Keep it.** The general form is now in
`PROJECT_PLAN` §7: **a tool that restores state must restore only what it changed, or refuse to run
where it cannot tell the difference.**

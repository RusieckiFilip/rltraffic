# PRE-REGISTRATION — Offline Multi-Agent Decision Transformer for Traffic Signal Control

**Registered:** 2026-08-03 · **Git tag:** `v0.1-prereg` (annotated; carries this file's sha256)
**Registered before:** any offline corpus collection (P2.2), any MADT training (P4), and any
evaluation run reported in the paper. At the time of registration the repository contains **no
collected corpus and no trained model** — the offline data path (`offline/`) holds a trajectory
logger and a flow randomiser only.

---

## 0. How to verify this registration, and what it does and does not prove

This file cannot contain its own sha256 (a file whose content includes its own hash has no fixed
point). The hash is therefore recorded **in the annotated tag**, which is created after the commit:

```bash
git cat-file -p v0.1-prereg | grep sha256          # the registered hash
git show v0.1-prereg:PREREGISTRATION.md | sha256sum # must equal it
git log -1 --format=%cI v0.1-prereg^{commit}       # commit timestamp
```

**What this proves.** That the content of this document is exactly what it was when the tag was made,
and that it precedes — in the repository's history — every commit that produces corpus or results.
Any later edit to this file changes its hash and is visible as a diff against the tag.

**What it does not prove.** A local git tag is not a third-party timestamp: dates in a git repository
are supplied by the machine that wrote them, and anyone with write access can create a tag whose
metadata says anything. The externally checkable act is **pushing the tag to the public remote**
(`github.com/RusieckiFilip/rltraffic`), because the push event is recorded by a third party we do not
control. Until that push happens, this document is an internal commitment, and the paper must not
describe it as more than that. Optional additional anchor: mirror to OSF and cite the OSF DOI.

**Amendment policy.** This file is not edited after tagging. Changes go in §12 as dated amendment
rows, each stating what changed, why, and — the part that matters — **whether any data or result had
already been seen at the time of the change**. An amendment made after seeing results is not
illegitimate, but it is no longer pre-registration and will be labelled as such in the paper.

---

## 1. Research questions and directional hypotheses

Each is a **question**, not a promise. §10 states what we publish under every outcome, including the
outcomes we would rather not get. This framing is deliberate and predates any data.

| | Question | Directional hypothesis registered in advance |
|---|---|---|
| **RQ1 (C1)** | How does offline MADT performance vary with behaviour-data quality, quality measured as normalised return rather than by policy name? | **H1:** MADT performance improves monotonically with dataset normalised return, and MADT exceeds its own behaviour policy on at least the mid-quality tiers ("stitching"). |
| **RQ2 (C2)** | Does offline sequence modelling degrade more gracefully than online MARL under scenario shift? | **H2:** The advantage of shift-augmented training data over nominal data is **larger for MADT than for MAPPO** (a positive interaction term in the 2×2), i.e. sequence models exploit diverse data better than on-policy MARL. |
| **RQ3 (C3)** | What does the CityFlow → SUMO transfer curve look like from zero-shot through few-shot (k ∈ {5, 20, 100}) to full retrain? | **H3:** Zero-shot transfer is positive but incomplete — better than fixed-time, worse than within-backend MaxPressure — and closes substantially by k = 100. |
| **RQ4 (hook)** | Is DataLight's "DT cannot be applied for TSC" a property of DT, or of the configuration they tested (K ∈ {1,2}, one hardcoded RTG)? | **H4:** Performance improves with context length K over {1, 2, 5, 10, 20} up to a plateau, so that K ∈ {1,2} is materially below the plateau. K = 1 is not sequence modelling and is included as the degenerate anchor. |

H4 is registered as confirmatory precisely so that our answer to the paper's framing controversy
cannot be assembled after the fact.

---

## 2. Confirmatory versus exploratory — fixed now

**Confirmatory** (the only analyses whose p-values are reported as inferential):

| Family | Test | Unit |
|---|---|---|
| H1 | Spearman ρ between dataset normalised return and MADT normalised return, per headline scenario | dataset tier |
| H2 | Interaction contrast in the 2×2, on the primary perturbation family (demand surge) | paired evaluation draw |
| H3 | MADT zero-shot in SUMO vs the within-backend fixed-time anchor, per paired scenario | paired evaluation draw |
| H4 | Trend test over K ∈ {1,2,5,10,20} on the P4 validation scenario | paired evaluation draw |

**Exploratory** (reported with effect sizes and CIs, explicitly labelled, no inferential claims):
all remaining ablations (no-spatial-mixing, no-RTG), calibrated-vs-naive RTG prompting, per-family
perturbation breakdowns, the shape of the transfer curve between the registered points, dataset
diversity effects, secondary metrics, and every scenario outside the headline three.

Moving an analysis from exploratory to confirmatory after seeing data is forbidden. Moving one from
confirmatory to exploratory is permitted only with an amendment row stating the reason.

---

## 3. Outcomes

### 3.1 Primary metric — pinned to an exact implementation

The primary metric is **average travel time (s)**, lower is better, as computed by the metric
registered under the name **`average_travel_time`**. Verified in source on 2026-08-03 in the three
implementations this project uses — `metrics/cityflow.py`, `metrics/sumo.py`, and CityFlow's C++
`Engine::getAverageTravelTime` (`engine.cpp:682`), the fallback path. A fourth implementation exists
in `metrics/moss.py` and was **not** checked, because MOSS is out of scope (Decisions Log 2026-07-13);
it must be verified before any MOSS result is reported. In all three checked: the metric averages
over **all vehicles that entered the
network** — completed trips contribute their full travel time, vehicles still en route contribute the
time elapsed since departure. It is therefore free of survivorship bias.

> ⚠️ **It is NOT `average_time_of_journey`.** That metric sits three functions away in the same file,
> has a similar name, and averages over **completed trips only**. Under congestion the two diverge
> sharply and in the flattering direction: a policy that gridlocks the network lets only the lucky
> few finish and scores well. No table in the paper reports `average_time_of_journey` as travel time.

**Mandatory co-reported quantities.** Because a metric over "vehicles that entered" can still be
gamed by a policy that prevents vehicles from entering, every reported ATT cell is accompanied by:
- `count_of_vehicles_completing_journey` (throughput), and
- the number of vehicles that entered the network during the episode.

**Validity condition, declared in advance:** a comparison between two policies on a scenario is
reported as invalid, not as a win, if their entered-vehicle counts differ by more than 5%. Under such
a difference the two ATTs are averages over different populations.

### 3.2 Secondary metrics
Queue length, throughput, and a CO₂ proxy where the backend supplies one. Secondary metrics never
override the primary metric in a claim; where they disagree with it, the disagreement is reported.

### 3.3 Reward
Primary training/collection reward: **`queue_length`**. PressLight is reported as a robustness
appendix. Collection runs use `--local-reward-fn queue_length --global-reward-weight 0.0`, because at
the default global weight of 1.0 every intersection's `local_reward` silently carries the entire
global reward (`base_traffic_env.py:214`) and per-intersection RTG would be double-counted.

**Safety latch (P2.3), registered as a reversal rule rather than an expectation:** the Spearman
correlation of queue-based and pressure-based returns with average travel time is computed over the
corpus **before P4**. If pressure correlates clearly better with the primary metric, the primary
reward is switched to pressure and the switch is recorded as an amendment. Because the corpus logs
per-lane counts, this reversal costs no recollection.

### 3.4 Normalisation formulas, fixed now

*Dataset quality (C1 ladder axis)*, per scenario, on episode return under the collection reward:

```
normalised_return = 100 × (J_policy − J_random) / (J_maxpressure − J_random)
```
random = 0, MaxPressure = 100 by construction. Values may exceed 100 or fall below 0; that is
expected and is not clipped. Tier labels are assigned from this measured quantity, never from the
policy name — our own P0.2 anchors (plan §3.1, measured 2026-07-09) already show MaxPressure beating
MAPPO@60ep, so policy names imply a false ordering.

*Cross-backend transfer (C3)*, computed **within** each backend, using both anchors:

```
ρ = (ATT_fixedtime − ATT_policy) / (ATT_fixedtime − ATT_maxpressure)
```
fixed-time = 0, MaxPressure = 1. Raw travel times are never compared across backends: the platform
thesis (§5.2) reports SUMO and CityFlow absolute metrics as non-interchangeable, so a raw
cross-backend number would be rejected on the platform's own evidence.

---

## 4. Designs, fixed in advance

- **C1 — dataset ladder.** Source policies as provenance: random, fixed-time, MaxPressure, MAPPO at
  60/200/converged checkpoints, ε-noised MAPPO (ε ∈ {0.1, 0.3}), DQN(CNN), and a mixed pool. Tiers
  are labelled post hoc by §3.4's measured normalised return. If no learned policy beats MaxPressure
  after convergence, the top slice **is** MaxPressure; that sharpens RQ1 rather than weakening it.
- **C2 — a 2×2.** {nominal, shift-augmented} training data × {MADT, domain-randomised MAPPO},
  evaluated under perturbations. The domain-randomised MAPPO arm is not optional: without it, any
  MADT advantage is confounded between the architecture and the data distribution, which is the
  single most likely rejection reason for this claim.
- **C3 — a transfer curve, not a binary.** Zero-shot → few-shot k ∈ {5, 20, 100} episodes collected
  by a MaxPressure probe spanning scenario variants → full-retrain anchor. k = 1 is excluded as
  statistically meaningless. Runs on the **6 genuinely paired scenarios** (4× hangzhou_1x1, cologne1,
  cologne3). **grid4x4 is not paired** — its `.sumocfg` references a `.rou.xml` that does not exist
  in the repository — and is therefore a CityFlow-only coordination scenario for C1/C2.

**Headline scenarios** (fixed now, so the selection cannot follow the results): hangzhou_1x1 #1
bc-tyc (single intersection, P4 validation gate), grid4x4 (16 signals, coordination), cologne3
(3 signals, real network, heaviest load). All other scenarios are appendix material.

---

## 5. Data collection protocol, fixed in advance

- **Corpus is sized by flow draws, not episodes.** CityFlow demand is deterministic: with demand
  fixed, the engine seed changes nothing observable, so N episodes of a deterministic policy are one
  trajectory repeated N times (confirmed empirically 2026-07-27 — engine seeds 1000 and 1001 produced
  byte-identical trajectories and identical `episode_sha256`). Deterministic policies therefore get
  **one episode per draw**; stochastic policies (random, ε-noised MAPPO) may reuse a draw.
- **Train/evaluation draw split — registered here because it does not exist anywhere else, and once
  the corpus is collected it can no longer be chosen innocently:**

  | Draw ids | Role |
  |---|---|
  | `0` | **Nominal control.** Reported separately, never pooled with randomised draws. |
  | `1 … 999` | Training-corpus pool. Every offline training set draws only from here. |
  | `1000 … 1099` | **Held-out evaluation pool.** Never enters any training corpus, for any method, including baselines. |

  Draw 0 is kept out of both pools for a structural reason, not a statistical one: draw 0 preserves
  the source file's vehicle order (byte identity demands it) whereas draws k > 0 sort globally, so on
  cityflow1x3 and hangzhou_4x4 the nominal condition differs from randomised draws in **ordering as
  well as demand**. Pooling it with randomised draws would mix two different generative processes
  into one confidence interval.

  The corpus linter (P2.4) enforces the split mechanically by checking draw ids in manifests, and
  flags duplicate `episode_sha256` within a run as a residual determinism guard. ⚠️ **Qualified 2026-08-07: this guard is meaningful on cf_hz1x1 and cf_grid4x4 only.** cf_cologne3 is not bit-reproducible — 7.7 % of draws diverge under identical code and a deterministic policy (measured, with a grid4x4 control of 0/6) — so on that scenario **non-duplication is not evidence of distinct draws, and duplication is not guaranteed for a repeated one.** The linter must exempt cologne3 by name rather than infer it.
- **Backends.** The main corpus is CityFlow-only. SUMO corpora are small and exist solely for C3's
  few-shot and retrain points.

---

## 6. Training, tuning and model selection — the leakage rules

Offline RL is judged harshly on this and rightly so; a method tuned against online returns is not an
offline method. All three rules below are fixed now.

1. **No online model selection.** Every reported model is the checkpoint at a **fixed, pre-declared
   number of gradient steps** — never a checkpoint chosen by its evaluation score. Training curves may
   be shown; they may not be used to pick the reported model.
2. **Hyperparameters are tuned on the validation scenario only, then frozen.** Any tuning happens on
   hangzhou_1x1 #1 (the P4 scenario) and the resulting configuration is applied unchanged to
   grid4x4, cologne3, every ladder tier, every perturbation, and both backends. No per-scenario,
   per-tier or per-backend tuning.
3. **Baselines receive the same tuning budget as our method**, on the same scenario, by the same
   protocol, and the budget is reported. A baseline that was not tuned is reported as untuned.
4. **RTG targets are not tuned on test scenarios.** Probe-calibrated prompting derives its target from
   a MaxPressure probe run **in the target domain**; the calibration procedure is fixed in advance and
   the naive fixed-RTG variant is reported alongside it as the ablation.

---

## 7. Baselines — non-negotiable, fixed now

Offline: **BC**, **%BC** (top-10% return filter), **IQL** (independent per intersection); CQL
optional. Online: MAPPO, IPPO, DQN, plus **domain-randomised MAPPO** for C2. Classical: MaxPressure,
fixed-time, random.

**A registered fairness constraint for the value-based offline baselines.** In this platform
`terminated` is hardcoded `False` (`base_traffic_env.py:604`); every episode ends by time-limit
truncation. IQL and CQL must therefore **bootstrap through the episode boundary** and must not treat
it as absorbing. Treating a timeout as terminal causes systematic value underestimation near the end
of every episode and would hand our method an unearned win over its own baselines. DT/RTG is
unaffected — it does not bootstrap, and all episodes share one horizon.

---

## 8. Statistical analysis plan

- **Unit of replication.** For evaluation, the unit is the **held-out flow draw**; for training, the
  **training seed**. Both sources of variance are reported. Comparisons across methods use the **same**
  held-out draws (paired design).
- **Minimum replicates, fixed now:** ≥ 5 training seeds and ≥ 20 held-out evaluation draws per
  reported cell. Cost basis: measured rollout times of 0.59–1.22 s/episode (plan §P2.0b, measured
  2026-07-27), which makes this scale of evaluation inexpensive relative to training.
- **No optional stopping.** Replicate counts are fixed before results are seen. We do not add seeds or
  draws after inspecting an outcome. If a cell must be enlarged for a legitimate reason, it is
  enlarged for **every** method in that comparison and recorded as an amendment.
- **Primary test:** Wilcoxon signed-rank over paired draws (no normality assumption). **Descriptives:**
  mean ± 95% CI. **Effect sizes are mandatory** and reported next to every p-value; a significant
  result with a negligible effect is reported as such.
- **Multiplicity:** Holm–Bonferroni **within** each confirmatory family (H1–H4 in §2), α = 0.05.
  Exploratory analyses are not corrected and carry no inferential claims.
- **Environment stochasticity** comes from flow randomisation, not from policy seeds alone. Under
  deterministic demand, seed-only CIs measure policy stochasticity and would badly understate the
  true variance.
- **Failed and pathological episodes are included, never dropped.** Gridlock is an outcome, not an
  outlier; excluding gridlocked episodes would bias exactly the comparison C1 exists to make. Episodes
  are excluded only for infrastructure failure (crash, out-of-memory), and every exclusion is counted
  and reported.

---

## 9. Decision gates, declared in advance so a failed gate is a result and not a pivot

- **P4.2 gate (single intersection).** With ATT lower-is-better, on hangzhou_1x1 #1, over ≥5 seeds
  with CIs, both must hold:
  `ATT_MADT ≤ ATT_MaxPressure` **and** `ATT_MADT ≤ 1.05 × ATT_best_online`.
  Being within 5% of a weak MAPPO while losing to a 1970s heuristic is not a pass. Failing this gate
  triggers diagnosis before any multi-agent scaling — it does not trigger a redefinition of the gate.
- **P7.0 gate (dynamics shift, ~1 day).** MaxPressure trajectories logged in both backends on one
  paired hangzhou_1x1; per-feature KS/overlap statistics on the state distributions plus
  MaxPressure-normalised returns. Pathological shift descopes C3 to a characterised-limitations
  study. **Prerequisite, registered because it would otherwise invalidate the gate itself:** hangzhou's
  shipped `.rou.xml` binds no vType, so SUMO would run vehicles at `DEFAULT_VEHTYPE`'s 55.55 m/s
  against CityFlow's 11.11 m/s. vType must be bound and effective vehicle parameters verified equal
  across backends *before* the gate is run; otherwise it measures a 5× speed-limit artifact and would
  descope C3 for an entirely wrong reason.
- **P4.4 gate (does sequence modelling add anything).** If BC on the expert slice matches MADT within
  CIs, that is reported as a headline finding, and the paper's weight moves to the ladder, shift and
  calibration results. This is registered now so that it cannot later be presented as a minor
  footnote.

---

## 10. What we publish under each outcome — registered so that a negative result stays a paper

| Outcome | What the paper reports |
|---|---|
| H1 holds | The ladder as a positive quantitative result: how much data quality buys, with the stitching threshold identified. |
| H1 fails, no monotone relation and no interpretable non-monotonicity | A negative ladder result: offline MADT performance is not predicted by behaviour-data quality on this platform — a directly useful finding for practitioners choosing what to log. |
| H2 holds | Sequence modelling degrades more gracefully under scenario shift, with the data/architecture confound removed by the domain-randomised arm. |
| H2 fails | MADT is not more robust than a domain-randomised MARL baseline given the same data — reported plainly. The 2×2 is what makes this interpretable rather than embarrassing. |
| H3 holds | The transfer curve, with the few-shot budget needed to close the dynamics gap quantified. |
| H3 fails / gap indistinguishable from the interface control | C3 becomes a characterised limitation with the P7.4 control as evidence, and the paper stands on C1 + C2. |
| H4 holds | DataLight's negative result is explained as a configuration artifact of K ∈ {1,2}, stated fairly (see below). |
| H4 fails | Context length is not the explanation; we report that DataLight's negative finding survives a proper K sweep, which is a genuine contribution to the controversy in the opposite direction. |
| **MADT collapses on grid4x4** | Reported as the main finding on *when* offline sequence modelling stops working. This is registered explicitly because it is the **expected** hard case, not a surprise: DTLight's own Table 1 (read 2026-08-01) reports its pure-offline variant at 446.8 ± 128.0 on Grid 4×4 from weak data against a behaviour policy at 48.39 — far worse than the data it learned from. Predicting a failure in advance and then observing it is a result; discovering it afterwards and reframing is not. |

**Fairness commitment on the framing controversy.** The paper states that DataLight reported its
negative DT result **under the configurations it tested** (context length K ∈ {1,2}; a single
hardcoded RTG of −351 across all agents and scenarios; Appendix A.5), and presents our K-sweep as a
test of their stated hypothesis — never as a claim about their integrity or competence. Their own
diagnosis, that mixing trajectories from multiple agents may confuse a DT, is treated as a hypothesis
our per-intersection sequence design tests, and is credited as theirs.

**Novelty statement, registered so it cannot inflate later.** Within-domain RTG scaling is **already
taken**: DTLight sets RTG as `max_offline_return × scale` with its own ablation figure. Our defensible
remainder is RTG calibrated **in the target domain** from a probe policy run there, for the
cross-backend case DTLight never faces. The paper positions against their figure explicitly.

---

## 11. Known threats to validity, declared before any data exists

Declaring these now means that if one of them bites, it is a documented limitation rather than a
discovery a reviewer makes for us.

1. **hangzhou vType is unbound** — see §9. Would masquerade as a dynamics gap.
2. **grid4x4 is not paired** across backends; it carries C1/C2 only, never C3.
3. **Determinism / reproducibility boundary.** `experiments/runner.py` pins torch to one thread per
   cell. Pinning changes float reduction order: bitwise-different results at exactly
   (128,128)@(128,128) — which is MAPPO's `minibatch_size` × `hidden_dim` in `p0_baselines.json` —
   with max absolute difference 5.6e-4 at larger shapes. This is clean **only** while nothing recorded
   predates the pin. Any run recorded from an unpinned path and compared against a pinned one is a
   non-reproducible comparison, and the paper's reproducibility section states this.
4. **Draw 0 is structurally different** from draws k > 0 (ordering, not only demand) — handled by §5's
   split, listed here because it is the kind of detail that silently contaminates a CI.
5. **`local_reward` in `info` is composite** (`global_weight × global + local_fn`). Conditioning on
   both a per-intersection RTG and a separate global RTG without accounting for the overlap
   double-counts the global signal.
6. **Episodes end by truncation, never termination** — see §7.
7. **Scenario licensing.** The public dataset release is claimed only after the P2.3 audit confirms
   redistribution rights for the Cologne and Hangzhou source scenarios. If rights are unclear, we
   release the generation code and draw seeds rather than the rendered corpus.
8. **cologne3 does not clear within the episode** (532 of 536 peak vehicles still in the network at
   t = 3600 s, measured 2026-07-27 under MaxPressure). RTG on that scenario conditions on a partially
   unresolved state. This is why cologne3 is the stress case, and it is a property to report, not a
   defect to hide.

---

## 12. Amendments

Each row states whether data or results had already been seen when the amendment was made.

| Date | Change | Reason | Results already seen? |
|---|---|---|---|
| 2026-08-11 | **A6 — §9's P4.4 gate wording *"if BC … matches MADT within CIs"* is sharpened into a declared EQUIVALENCE MARGIN, before any baseline exists.** **δ = 0.6263 ATT**, the DT's own measured paired margin over its behaviour policy MAPPO@1000 on the 100 held-out draws (105.5820 − 104.9558), taken from the committed `docs/data/p4_gate.json`. **Decision rule, fixed now:** BC **matches** the DT iff the 95 % CI of the paired per-draw difference (DT − BC) lies **entirely within [−δ, +δ]**; the DT is **genuinely better** iff that CI lies entirely **outside** and below; anything straddling is reported as **inconclusive at this power**, with the CI width. The same δ instantiates the pre-registered forecast: *"BC lands within δ of the DT on the paired held-out mean"* — falsifiable in both directions. | **"Contains 0" is a failure to reject, not a demonstration of equivalence, and without a declared δ the judgement *is this small enough to call them equal* gets made after the number is visible — at abstract-writing time, when the pivot condition is being read off.** That is the researcher degree of freedom D5 and D7 exist to remove, and it costs one sentence now against being unbuyable later. ⚠️ **CLARIFIED 2026-08-11, before any P4.4 number existed and after the `v0.7-prereg-a6` tag was pushed — so a reader following that tag sees A6 without this clause.** Two additions, neither changing the decision rule. **(i) The claim "δ is derived, not chosen" is HALF TRUE and is corrected here: the reference quantity is derived, but the MULTIPLIER of 1.0 is a choice** — half the margin, or the margin's own CI width, are equally derivable and would give a different verdict on the same data. **(ii) Because δ equals the effect under study, one branch can mislead**: BC landing 0.5 ATT worse than the DT sits inside ±δ and returns *matches*, while having recovered only **20.2 %** of the DT's margin over its behaviour policy — which the registered reading (*"whatever the DT gained BC gained too"*) would not be true of. **Therefore the RECOVERED FRACTION of the DT's margin — `(MAPPO@1000 − BC) / (MAPPO@1000 − DT)` — is reported UNCONDITIONALLY beside the verdict, in every branch.** Unconditional rather than gated on `δ/2`, because a gate would introduce a second chosen threshold to defend, and one number is not clutter. Original justification follows. **δ is derived, not chosen:** the question at stake is whether sequence modelling earned the DT's margin over its own data, so the margin *is* the scale at which "matches" must be judged — if BC comes within it, whatever the DT gained BC gained too, and the architecture cannot be credited. Note it is a **strict** margin: 0.597 % of the DT's ATT, and 1/115th of the DT's margin over MaxPressure. **Sharpening an ambiguous registration before any result exists is a strengthening**, and this row records that no BC, %BC or IQL model existed when it was written. | **No.** No BC, %BC or IQL model exists; no P4.4 number of any kind exists. δ is computed from P4's own committed artifact, which was merged on 2026-08-08 and is unchanged. |
| 2026-08-07 | **A5 — the >5 % validity condition attached to `vehicle_count` is WITHDRAWN, co-reporting becomes unconditional, and the guarantee A4 lost is restored as a REGISTERED requirement on shared draw ids.** Three changes. **(1) `vehicle_count` at the episode horizon remains a mandatory companion to every reported ATT cell, and is now reported UNCONDITIONALLY** — there is no threshold at which it is omitted and none at which it triggers a verdict, so no researcher degree of freedom remains in the disclosure. **(2) The >5 % invalidation applied to it is withdrawn.** **(3) NEW REGISTERED REQUIREMENT, replacing the guarantee A4 removed: every reported comparison must be made over SHARED DRAW IDS, and the draw ids must be reported alongside the cell.** A comparison that cannot be made over shared draws is **void** — that is the single surviving case of the voiding rule, and it is binary and checkable rather than thresholded. | **A4 mistook a control outcome for a population size, and our own data shows it.** The original §3.1 condition guarded against comparing policies that had seen different vehicle populations; `entered` is a population size, stable across policies on fixed demand (measured spread **4.1 %** across Tier 1 cells: 1595 / 1623 / 1661). A4 substituted `vehicle_count` **at the horizon** because contract C8 forbids `count_of_vehicles_completing_journey` alongside any MAPPO checkpoint — and carried the 5 % band across unchanged. But `vehicle_count` at the horizon is a **congestion outcome**: it is precisely what a good controller drives toward zero. Applying a 5 % band to it invalidates the effect the study exists to measure. **Measured on our own committed anchors (P8.0 retro-review, 2026-08-07): 5 of 6 pairwise comparisons are INVALID under A4 as written** — cf_grid4x4 MaxPressure vs MAPPO differs by **98.1 %** on that quantity, which is not a defect in the comparison but the *result* — and **the single comparison that passes does so as an aggregation artifact**, per-seed counts 206/99/166 against 157.3/156.0/157.7, both means landing on 157.0 by coincidence. **Why the replacement is stronger than what it replaces:** the shared-draw requirement enforces identical demand *by construction* rather than testing for it *after the fact*, and it is checkable from the manifest, whereas the 5 % band was a statistic computed on an outcome. Withdrawing a check without naming what replaced it would leave prose where a mechanism had been. | **Yes, partially, and the detail matters.** Seen: the P0.2/P8.0 baseline anchors, the Tier 1 fixed-time cells, and the 5-of-6 invalidity table above — which is *why* this amendment exists, and which concerns platform baselines only. **Not seen, because none exists: no MADT, BC, %BC, IQL or CQL has been trained, and no result bearing on C1, C2 or C3 exists.** The MAPPO-vs-MaxPressure orderings this condition would govern are currently marked **NOT SETTLED** in `PROJECT_PLAN.md` §3.1 and are explicitly *not* settled by this amendment — A5 changes what a valid comparison **is**, not what any comparison **says**. |
| 2026-08-06 | ⚠️ **SUPERSEDED IN PART BY A5 (2026-08-07): the >5 % validity condition this row restates against `vehicle_count` is WITHDRAWN — it mistook a control outcome for a population size and invalidates 5 of 6 of our own comparisons. The mandatory co-report survives and is now unconditional.** The row stands unedited below as the record of what was registered. **A4 — §3.1's mandatory co-reported quantities are relaxed from {completion count, entered count} to {`vehicle_count` at the episode horizon}, and the >5% validity condition is restated against it.** §3.1 required every ATT cell to carry `count_of_vehicles_completing_journey` and an entered-count. **Contract C8 makes that impossible for any cell involving a MAPPO checkpoint:** `MAPPOAgent._build_global_features` feeds the centralised critic from *every* key in `info["metrics"]`, so adding a metric changes the critic's input width and the checkpoint refuses to run — in `act()` as well as training, i.e. in evaluation and collection alike. `count_of_vehicles_completing_journey` therefore cannot coexist with a MAPPO rollout. **`average_travel_time` can**, because it is exposed top-level in `info`, independent of the requested metric set (verified live 2026-08-06). | **The survivorship concern §3.1 was written to defend against is already met by the metric's own definition**: `average_travel_time` averages over *all vehicles that entered*, counting en-route vehicles at elapsed time (verified in source across `metrics/cityflow.py`, `metrics/sumo.py` and CityFlow's C++ fallback, 2026-08-03). The completion count was belt-and-braces, not the guarantee. `vehicle_count` at the horizon is retained as the mandatory companion because it is what actually exposes gridlock — grid4x4's MAPPO@60 cell shows 1000 vehicles still in network — and it is available in every cell without perturbing any agent. **The relaxation is forced by a platform constraint, not chosen for convenience**, and the constraint is now contract C8. | **No.** Unchanged: no MADT, BC, %BC, IQL or CQL exists, and no result bearing on C1, C2 or C3 exists. The corpus is behaviour data only. |
| 2026-08-06 | **A3 — the declared MAPPO training budget B is raised once, 500 → 1000 episodes, on all three headline scenarios.** This is the response pre-declared in `docs/briefs/RUNSPEC_01_p2.1_mappo.md` §4 *before* the runs, not a reaction to a result. The registered criterion (relative change of the mean training return between consecutive 50-episode windows < 0.05, for the final window **and** the preceding one) fails at B = 500: **cf_cologne3 0/5 seeds, cf_grid4x4 0/5, cf_hz1x1 4/5** (seed 303 at 0.0509, marginally over). Per §4 the raise happens once and only once; a second raise requires coming back. **Also resolved here: §4 defined the criterion per seed but never said how a *scenario* verdict aggregates them.** Resolved in the strict direction — a scenario counts as converged only if **all** seeds do — because the conservative reading buys more training rather than less, and a procedure that plateaus on 4 of 5 seeds has not reliably converged. | **The raise is driven entirely by the TRAINING-return curve and by no evaluation number**, which is what keeps it outside D5's prohibition on online model selection. Nothing about `att_horizon`, MaxPressure comparisons or ladder position entered the decision; the criterion is computed on rewards the agent already saw during its own training. The aggregation ambiguity is recorded rather than quietly resolved because the choice is not neutral: the loose reading would have declared cf_hz1x1 converged and stopped there. | **Partially, and it does not bear on this decision.** MAPPO training and evaluation numbers at 60/200/500 had been seen, including that MAPPO@500 beats MaxPressure on cf_hz1x1. None of it entered the criterion, which reads only training returns. **No offline model of any kind exists** — no MADT, BC, %BC, IQL or CQL — and no result bearing on C1, C2 or C3 exists. |
| 2026-08-06 | **A2 — A1's illustrative figures corrected to canonical values. A1's substance is unchanged and slightly strengthened; no part of the specification moves.** A1 quoted Random as 317.46 running-mean / 429.67 horizon, from the coordinator's own ad-hoc rollout. That rollout sampled actions from a hand-rolled RNG rather than `experiments.runner._baseline_chooser`, so it is a *different realisation of the random policy* and is not the canonical harness number. The canonical values, which reproduce the committed 2026-07-09 anchor **bit-exactly**, are **Random 307.5346 → 413.53** (ratio 1.345). MaxPressure's A1 figures were already canonical (160.5584 → 247.75, ratio 1.543; deterministic, std 0). Corrected consequence: MaxPressure's advantage over Random is **47.8% under the running mean and 40.1% at the horizon** — previously stated as 49.4% and 42.3%. | The swing A1 exists to document is **7.7 percentage points, not 7.1** — the effect is marginally larger under canonical measurement, so the amendment's argument strengthens rather than weakens. Recorded rather than silently edited because the error is of the exact family A1 was written to prevent: **a number produced by a different pipeline than the reference, quoted alongside numbers from the reference.** It was committed by the same author who, one task earlier, ruled that a measurement path must be validated against the reference pipeline before its numbers are used. Source: P8.0 re-derivation (`docs/data/p0_baselines_horizon/`), merged `2153e2a`. | **No.** Unchanged from A1: no MADT, BC, %BC, IQL or CQL exists, and no result bearing on C1, C2 or C3 exists. The corrected figures are platform-baseline measurements only. |
| 2026-08-05 | **A1 — the primary metric's EPISODE AGGREGATION is specified, closing a gap in §3.1.** §3.1 pinned the *per-step* metric by registry name (`average_travel_time`, survivorship-free) and said nothing about how it is aggregated over an episode. It is hereby specified: **the reported quantity is the value of `average_travel_time` at the episode horizon** — i.e. the mean over all vehicles that entered the network during the episode. It is **not** the mean of the per-step samples. `experiments/runner.py:168,175` computes the latter (`travel_samples.append(info["average_travel_time"])` then `_mean(...)`), so **every number in PROJECT_PLAN §3.1 is an episode-mean of a running average** and must be re-derived under this definition before being used as a sanity anchor for any reported result. The legacy quantity may still be reported for internal continuity, but only under an explicit name such as *running-mean ATT*, never as "average travel time". | The gap is a live researcher degree of freedom on the paper's primary metric, and the two aggregations differ **policy-dependently**, so it moves effect sizes rather than merely rescaling them. Measured on hangzhou 1x1 (2026-08-05, coordinator's own rollout): MaxPressure 160.56 episode-mean vs 247.75 final (ratio 1.54); Random 317.46 vs 429.67 (ratio 1.35). MaxPressure's advantage over Random reads as **49.4%** under one aggregation and **42.3%** under the other. C1's normalised return, C2's 2×2 interaction and C3's within-backend transfer ratio are all differences-of-differences on this quantity. The final-horizon value is additionally the standard TSC reading and the one comparable to DataLight / DTLight / RESCO figures; the running mean has no standard interpretation and is dominated by the early transient when few vehicles have accumulated travel time. | **Partially — and the honest detail matters.** Platform baseline numbers (P0.2 anchors) and P2.5 fixed-time controller measurements had been seen. **No offline model of any kind exists** — no MADT, BC, %BC, IQL or CQL has been trained, and **no result bearing on C1, C2 or C3 exists**. The choice between the two aggregations was made on literature comparability and interpretability, not on which favours any of our results, because no result of ours is yet expressible in either. |

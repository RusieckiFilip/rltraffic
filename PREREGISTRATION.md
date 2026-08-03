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
  flags duplicate `episode_sha256` within a run as a residual determinism guard.
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
| — | — | — | — |

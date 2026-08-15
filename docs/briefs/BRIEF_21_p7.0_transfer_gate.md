# BRIEF 21 — P7.0: the C3 early gate

**Mode:** Claude Code · **Branch:** `task/p7.0-transfer-gate`, from `main`
**Worktree:** fresh — `git worktree add /home/filip/rltraffic-p70 -b task/p7.0-transfer-gate main`
**Supersedes `BRIEF_04`** (written 2026-08-07, never run). Read `BRIEF_04` §3 — **its parity contract
is still binding and is not restated in full here** — then `PROJECT_PLAN` §1 (claim constraints) and
**§1b**, `PREREGISTRATION` **§3.4** and **§9**, and `docs/CONTRACTS.md` **C2, C6, C8**.

⚠️ Absolute paths · pin threads on every job · never write "MADT" (C9) · guards read with **no
arguments** and counted from **full output, never a tail**.

---

## 1. Why this runs now, after sitting unrun since July

P7.0 was scoped as a ~1-day kill-switch for C3, to run *"right after P1, parallel to P2"*. P1 closed
2026-07-27. **Its value has gone UP since, not down**, and that is why the 2026-08-14 sequencing ruling
promoted it from October to **queue item 1**:

**P4.3 measured the in-domain return prompt at 0.9026 ATT across a 13,000-wide declared grid.** So
**probe-calibrated return prompting — the paper's only NAMED contribution under `A9` — now rests
entirely on the cross-domain axis, and P7.0 is the gate on exactly that axis.** It is the highest
information-per-hour item we have: **both outcomes change what the paper claims** (§6).

## 2. 🚨 THE PRECONDITION — and the defect is narrower and more fixable than the plan says

`PROJECT_PLAN` §10 and the Decisions Log (2026-07-31) record that *"the shipped `.rou.xml` binds no
vType, so SUMO would run 55.55 m/s against CityFlow's 11.11"*. **Measured today, and the real defect is
one attribute:**

```
scenarios/hangzhou_1x1_bc-tyc_18041610_1h/…rou.xml
  <vType accel="2.0" decel="4.5" id="pkw" length="5.0" maxSpeed="11.111" minGap="2.5" width="2.0"/>
  <vehicle depart="0" id="0">        ← no type= attribute
  vType declarations: 1 · vehicles: 2021 · vehicles carrying type=: 0
```

**The correct vType is already there, with `maxSpeed="11.111"` — exactly CityFlow's speed. Nothing
references it.** For contrast, `scenarios/cologne1/cologne1.rou.xml` is natively authored and **2015 of
its vehicles carry `type=`**.

> **REQUIRED, as the FIRST commit and before any measurement: bind the vType.** Every `<vehicle>` in
> every hangzhou `.rou.xml` used by this gate carries `type="pkw"` (or the file declares it as the
> default). **Acceptance is mechanical and has a positive control: `vehicles carrying type=` must equal
> `vehicles`, and the check must be shown to FAIL on the unfixed file before it is trusted.**
> ⚠️ **Discriminating power, stated because it is the whole reason this comes first: unfixed, SUMO runs
> a 5× speed limit against CityFlow, so the gate would measure a speed-limit artifact and call it a
> dynamics gap. Descoping C3 on that number would descope it for the wrong reason.**

⚠️ **These are scenario files, not frozen source** — but **do not edit a file a recorded run used.**
Write the bound copies alongside, under a new path, and name it in the packet.

## 3. Two `DEFERRED` rows name THIS task as their trigger, and both are still open

- **`DEFERRED` 18 — `info["average_travel_time"]` is verified metric-set-independent on CityFlow ONLY.**
  P2.6 measured 3 scenarios × 2 policies × 61 rows, 6/6 bit-identical — **all CityFlow**. The supporting
  source argument is about `metrics/cityflow.py` and **does not transfer**. `sumo_env.py:316` emits the
  key; whether the value is right when the metric is **not requested** is **unmeasured on SUMO**.
  ⚠️ **A wrong-but-plausible ATT column in a SUMO corpus is exactly the silent-number failure this repo
  exists to prevent.** **Cost: ~1 h — one paired rollout, 1-metric against 3-metric env,
  `np.array_equal` on the ATT sequence. Do it here.**
- **`DEFERRED` 23 — ID-keying has ZERO coverage from real data.** All 4800 CityFlow episodes have
  `ix_ids` already sorted, 1 distinct order each, so ID-keying rests entirely on one two-intersection
  synthetic fixture; positional-keying mutants kill 0 corpus tests. **CityFlow and SUMO may enumerate
  intersections differently — that is the case the fixture stands in for, and this task is the first
  time both backends are read side by side.** **Report the enumeration order each backend gives for the
  same scenario. If they differ, that is a finding and the mutant becomes live.**

## 4. What to measure

**One paired scenario, MaxPressure in both backends.** The repo has **6** genuinely paired scenarios
(§1); use a hangzhou 1×1 as the plan specifies, and say which.

1. **Per-feature state-distribution comparison** — KS statistic and overlap coefficient per feature,
   **reported per feature, never pooled into one number.** Pooling is what would hide a single
   catastrophic feature inside an average.
2. **Within-backend normalised return**, per `PREREGISTRATION` §3.4, **using both anchors**:
   `ρ = (ATT_fixedtime − ATT_policy) / (ATT_fixedtime − ATT_maxpressure)`.
   ⚠️ **Raw travel times are NEVER compared across backends** — §3.4 forbids it, on the platform
   thesis's own evidence that SUMO and CityFlow absolute metrics are non-interchangeable. **Every
   cross-backend statement is about ρ or about distributions, never about ATT.**
3. **The vType parity check of §2, and `DEFERRED` 18's ATT check, both reported with their controls.**

## 5. ⚖️ WHAT EACH OUTCOME LICENSES — DECLARED BEFORE THE RUN

**This is the point of the brief.** P7.0's value is that it is cheap and decisive; that value is lost if
the day it returns becomes a negotiation about what the number meant. **Both branches are already
implied by documents on disk; writing them here makes the return a READING.**

### Branch A — the feature distributions are comparable and ρ transfers sensibly

> **LICENSED: P7.1, P7.2 and P7.3 proceed**, and **probe-calibrated return prompting has somewhere to
> live other than a 0.9026 ATT in-domain ablation.** The named contribution gets the axis on which its
> case actually rests, and §1's claim constraint about the in-domain lever is unchanged but no longer
> the whole story.
> **NOT licensed even under A:** any claim that the transfer *works* — P7.0 measures the shift, not the
> method. And **no raw cross-backend ATT comparison, ever.**

### Branch B — the shift is pathological

> **THIS IS NOT A FAILURE AND MUST NOT BE WRITTEN AS ONE.** It is **the measured reason C3 is reported
> as a registered, unrun hypothesis** rather than a silence — which `PREREGISTRATION` §10 already
> registers as publishable, and §1 already concedes (*"the paper stands on C1+C2 if C3 lands
> negative"*, and C1 is now delivered on both axes).
> **It fires CUT 3 of the 2026-08-14 sequencing ruling with its consequence already named:** the paper
> **may not claim probe-calibrated prompting as a validated contribution** and must present it as the
> in-domain ablation plus a registered, unrun cross-domain hypothesis. **That sentence is already
> written down; branch B just makes it operative.**
> ⚠️ **And it buys the remaining weeks for P5**, which §10's ruling makes the next item and which is the
> only experiment that can separate the two explanations §1b's R5 leaves standing.

### Branch C — the result is equivocal

> **Named in advance so it is not silently resolved into A or B.** Report the per-feature table, state
> that the gate did not resolve, and **do not decide C3's fate on it.** ⚠️ **A gate that returns
> "unclear" and is read as "proceed" is how a kill-switch stops being one.**

⚠️ **Binding on all three: the branch is chosen by the DECLARED criteria in your plan file, written
before any number exists, not by reading the table afterwards.** State the criteria; a criterion
invented after the data is the researcher degree of freedom `D5` and **A8(a)** exist to remove.

## 6. Scope fence

**In:** the vType binding, one paired scenario, MaxPressure both backends, the per-feature comparison,
ρ with both anchors, `DEFERRED` 18 and 23. **Out:** any DT, any training, any transfer *experiment*
(that is P7.3), grid4x4 (**not paired** — its `.sumocfg` references a `.rou.xml` that does not exist),
MOSS, and any change to the state feature set (that is P7.1's frozen decision).

## 7. Definition of Done

- [ ] `docs/plans/p7.0.md` committed **before any measurement**, carrying §5's three branch criteria
      verbatim and the declared scenario
- [ ] vType bound; `vehicles with type= == vehicles`; **the check shown to FAIL on the unfixed file**
- [ ] Per-feature KS and overlap, **reported per feature**; ρ with both anchors; no raw cross-backend ATT
- [ ] `DEFERRED` 18 measured on SUMO with its control · `DEFERRED` 23's enumeration orders reported
- [ ] The branch declared by the pre-written criteria, and **which criterion fired**
- [ ] Every mutation executed and **its failure pasted**
- [ ] Suite green, tail pasted, pinned state stated; all three guards with **no arguments**, counted
      from full output, each naming its corpus
- [ ] Return Packet at `docs/returns/P7.0.md` with the AI-assistance record
- [ ] §6's checkbox left unticked; it is mine, in the merge commit

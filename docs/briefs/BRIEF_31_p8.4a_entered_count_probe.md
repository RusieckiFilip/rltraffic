# BRIEF_31 — P8.4a: do the offline-learned policies admit fewer vehicles than the baselines?

**Task id:** `P8.4a` · **Branch:** `task/p8.4a-entered-probe` · **Issued:** 2026-08-28
**Mode:** Explore → Plan → Code → Commit, with human gates. **Start in plan mode.**
**Compute:** replay only. **No training, no collection, no new corpus.**

> **Read `docs/reviews/T1-metric-ground-truth.md` FIRST.** This task exists because of it.
> ⛔ **The P5.3b campaign is HELD until this answers. Nothing else runs.**

---

## 1. The question, and why it comes before the re-derivation

T1 established that `metrics/cityflow.py`'s `average_travel_time` **excludes every vehicle created but
never admitted to the network** (`get_vehicles(include_waiting=False)`, `:60` and `:159`), while
CityFlow's own `getAverageTravelTime` (`engine.cpp:682-691`) averages over the **whole
`vehiclePool`**. Measured on hz1x1: **Random never admits 774 of 2021 vehicles (38.3 %)** and reads
427.04 under ours against **877.95** under the engine's.

**So our metric rewards a policy for preventing vehicles from entering.**

🚨 **The author's question, and it decides whether P5.2 has a result or an artefact:**

> **DO THE OFFLINE-LEARNED POLICIES ADMIT FEWER VEHICLES THAN THE BASELINES?**
>
> **If `iql` reaches 190.96 on `random` data by blocking entry, P5.2's headline is a metric artefact.
> If it admits as many or more, the true gap is WIDER than reported and the result strengthens.**

**That is one number per cell and it is far cheaper than the full re-derivation. It runs first.**

---

## 2. Registered prediction — written before any number exists

| # | Prediction | Falsified by |
|---|---|---|
| **E1** | **Every learned arm (`dt`, `dt_nomix`, `dt_spatial`, `bc`, `bc_top10`, `bc_top10_perix`, `iql`) admits AT LEAST AS MANY vehicles as the behaviour policy of its own tier**, per cell, on shared draws. | any arm whose `entered / created` is **below** its tier's behaviour anchor by more than the between-seed spread. **That arm's ATT advantage is then partly or wholly a selection artefact and must be reported as such.** |
| **E2** | **`mappo1000`'s arms sit at ≈100 % admission** — T1 measured the `mappo1000` behaviour policy at **100.0 %** entered/created. | anything materially below 100 % there, which would indict the probe before the science. **This is the null control.** |
| **E3** | **Admission is lower on weak-data tiers than on `mappo1000`, for the BEHAVIOUR policies** — T1 measured 100.0 / 90.8 / 92.9 / 65.0 % for mappo1000 / fixedtime / mappo060 / random. | a flat profile, which would contradict T1's own measurement and indict the replay. |

⚠️ **E1 is the load-bearing one. E2 and E3 are controls on the instrument**, in the same spirit as
A8's `fixedtime` prediction in P5.3a: **a failure there indicts the probe before it indicts the
science, and that direction of inference is registered now so it cannot be reversed later.**

---

## 3. What to measure, per cell

For every replayed episode, record and store:

- **`created`** — the vehicle count the flow file defines. Fixed per draw, and it is the denominator.
- **`entered`** — vehicles that ever appeared in the network.
- **`never_entered = created − entered`**, and the ratio.
- **`att_ours`** — the current metric, which must reproduce the stored `att_per_step[-1]` **bit-identically after the float32 cast**. ⭐ **T1 replayed 36 shipped episodes and every one did. If yours do not, STOP — the replay is not faithful and nothing downstream is interpretable.**
- **`att_engine`** — `eng.get_average_travel_time()` at the horizon.
- **`horizon_vehicle_count`** — already reported; carry it so the three populations can be compared.

**Both ATT definitions on every episode. This is `A5`'s co-report restored and widened, and it is
what the author ruled: the entered count is not a co-report, it is the first thing measured.**

---

## 4. Scope — enough to answer decisively, and no more

**The effect T1 measured is enormous (2.9 % against 38.3 %), so a small sample settles E1.**

- **Arms:** every method arm plus the behaviour anchor, on the tiers carrying headline claims —
  **P5.2's `random` and `mappo1000` tiers** (where the author's `iql` question lives, and the
  crossover headline) and **P4.6's five hz1x1 tiers**.
- **Draws:** **10 held-out draws per cell, the first ten of 1000–1099, all 5 seeds.** Declared here,
  before any number. ⚠️ **If E1 is close on any arm, that arm goes to the full 100 draws before any
  verdict — closeness is not a result at n=10.**
- ⛔ **Not in scope:** the full re-derivation (P8.4b), fixing the metric, changing any committed
  artifact, or re-running anything that trained.

**Gate 0, and Amendment R2's standing instruction applies: a ONE-CELL TIMING PROBE before committing.**
The 1.684 s/episode figure is **hz1x1-measured**; grid4x4 has 16 intersections and is **unmeasured**.
**Measure it; do not extrapolate.** The coordinator's costing for the eventual full re-derivation —
**46,800 episodes / 100 cells → 21.9 h serial, 4.4 h at 5-way; 35,100 / 75 after removing the 25 cells
P4.7 contains bit-identically → 16.4 h serial, 3.3 h wall** — is **hz1x1-calibrated and is an estimate,
not a measurement.**

---

## 5. Definition of Done

- [ ] Every replayed episode reproduces its stored `att_per_step[-1]` bit-identically, or the task is `BLOCKED`
- [ ] `entered / created` per cell, **per seed, never a bare pooled mean**
- [ ] E1, E2, E3 scored as registered; **E1 close on any arm → that arm re-run at 100 draws**
- [ ] Artifact `docs/data/p8_4a_admission.json`, **no verdict** on the science
- [ ] Whole suite green, pinned; guards 16 / English / frozen; manifests re-verified
- [ ] Return Packet; **§6's `P8.4a` box ticked in the merge commit**; AI-assistance record
- [ ] **Timing measured and reported per cell — P8.4b's cost model depends on it**

## 6. What the packet must not say

1. ⛔ **Not "the metric is wrong".** It computes what its docstring says. **It measures a different
   population than the field's, and that is the finding.**
2. ⛔ **No novelty claim.** See §7.
3. ⛔ **No conclusion about P5.2's headline** beyond what E1 measures. If admission is equal, the
   correct statement is *"the ATT gap is not explained by differential admission on these cells"* —
   not *"the result is safe"*.

---

## 7. 🔒 THE NOVELTY CLAIM IS UNSEARCHED, AND THE PAPER MAY NOT MAKE ONE YET

The coordinator ran **two** web searches on 2026-08-28. **Two queries are not a literature search, and
the result below is weak evidence of absence, not evidence.**

**What they DID establish, and it splits the claim into three parts:**

| part | status |
|---|---|
| **The phenomenon** — vehicles queued and never admitted | **KNOWN and documented.** SUMO's own docs describe the insertion queue and the teleporting buffer explicitly, and note that *"the configured demand in the route file does not necessarily correspond to the actually realised flow"*. |
| **That rankings are metric-dependent in TSC** | **KNOWN in general.** An RL-TSC review states plainly that *"the performance of the same method could be different under different metrics"*. |
| **This specific instance — CityFlow's two ATT definitions differing by up to 2×, POLICY-DEPENDENTLY, with a measured RANKING INVERSION among the behaviour policies used to build offline TSC corpora** | **NOT FOUND in two queries. NOT ESTABLISHED AS NOVEL.** |

⚠️ **One thing the search surfaced that sharpens the finding rather than the novelty:** a secondary
source states the field's ATT as *"the average travel time of all vehicles spent between entering and
leaving the traffic network"* — **which is OUR definition, not CityFlow's implementation.** If that
holds up, then papers report the engine's number while stating our quantity, **and the gap is in the
literature rather than in us.** ⛔ **That is a much bigger claim, it rests on one secondary source, and
it must not be written down as a finding until the CityFlow paper and two or three benchmark papers
have been read in full.**

> **RULED: no sentence in the paper claims this is unreported until a proper search is run and its
> degradations declared, in the form `A9`'s search was.** The defensible framing meanwhile is
> `PREREGISTRATION` **A10** plus **B4** below — **and neither depends on novelty.**

## 8. ⭐ B4 — the argument this whole episode buys the paper, in the coordinator's words

**Write this into the packet, and it belongs in the paper's discussion of pre-registration:**

> **`PREREGISTRATION` §3.1 registered a safeguard against exactly this failure — an entered-count
> co-report with a >5 % invalidation — in its own words: *"a metric over 'vehicles that entered' can
> still be gamed by a policy that prevents vehicles from entering."* Amendment A4 withdrew it on the
> ground that the metric's own definition already defends against this, which is a restatement of the
> vulnerability rather than a defence. Amendment A5 withdrew the remainder on the ground that the
> entered count is *"stable across policies on fixed demand (measured spread 4.1 %)"* — a spread
> measured WITHIN ONE POLICY FAMILY and stated OF ALL POLICIES. Across the four behaviour tiers it is
> 35.0 %. The safeguard was correct, its removal was not, and the error was found three weeks later by
> a ground-truth check the registration itself made it possible to specify.**
>
> ⚠️ **The coordinator recommended and approved both amendments. That is on the record and belongs in
> it.**

⭐ **This is the strongest argument the paper has for pre-registration being worth its cost, and it
does not depend on the finding being novel:** the registered document contained the objection, the
objection was overruled on a measurement that was true of a subset and false of the population, and
**the registration is what made the overruling legible enough to catch.** An unregistered project
would have had no safeguard to remove and no record of removing it.

---

# ✅ AMENDMENT A — 2026-08-28, ruled at the plan gate

**APPROVED TO CODE.** All five questions answered; four of the six conflicts are my errors and are
corrected below.

## A0 ⭐ THE EXPLORATION ALREADY SEPARATED TWO EFFECTS I HAD CONFLATED, AND IT MAY HAVE ANSWERED THE AUTHOR'S QUESTION

**Register this before scoring anything, because it reframes the exposure.** From the four exploratory
episodes:

| cell | `att_ours` − `att_engine` | never entered |
|---|---|---|
| hz1x1 maxpressure | **+3.35** | 0 |
| grid4x4 maxpressure | **+6.88** | 0 |
| grid4x4 random | **+7.14** | 0 |
| **hz1x1 random** | **−355.62** | **615 (33.9 %)** |

> ⭐ **When admission is complete, ours sits +3 to +7 s ABOVE the engine's — that is T1's M1
> quantisation bias, isolated. When admission is incomplete, ours falls 355 s BELOW — that is B1's
> selection bias. TWO SEPARABLE EFFECTS, separated without being asked to.**
> 🚨 **AND THE EXPOSURE IS PROBABLY hz1x1-LOCALISED, WHICH INVERTS WHICH RESULTS ARE AT RISK.**
> **Coordinator's corroboration, mechanical rather than measured: hz1x1 pushes 1813 vehicles through
> ONE intersection; grid4x4 pushes 1323 through SIXTEEN — roughly 24x the per-node load.** The
> insertion buffer only backs up when the network cannot absorb the demand, so zero blocking on
> grid4x4 is *expected*, not lucky.
> **If that holds: P5.2's grid4x4 headline is NOT vulnerable to B1, and the exposure lives in
> P4.6/P4.7's hz1x1 ladder — which is C1.** ⚠️ **One draw, behaviour policies only, plus a plausibility
> argument. It is a hypothesis to test, not a result, and the packet must say so in those words.**

## A1 ✅ Q1 — YES, materialise grid4x4 draws 1005–1009

`offline/campaigns/p5_2.sh:239-244` already prescribes it, `scenarios/draws/` is gitignored, and
`DEFERRED` 55 records that these keep being deleted by routine cleanups — **restoring them is repairing
a known loss, not creating a private resource.**
> 🔒 **REQUIRED, and it is P5.3a's Gate −1 pattern: regenerate the FIVE SURVIVORS first and report the
> byte-identity result explicitly. If any survivor does not reproduce, STOP and report `BLOCKED`** —
> every downstream number would then be measured on draws that are not P5.2's, undetectably. **The
> tool being no-op-or-refuse is the mechanism; reporting what it found is the evidence.**

## A2 ✅ Q2 — MY OWN §7 THRESHOLD ANSWERS THIS, AND YOUR MEASUREMENT MOVES IT

`PROJECT_PLAN` §7's pre-flight rule triggers at **one hour**. **Your measured 0.7 s (hz1x1) and 1.2 s
(grid4x4) per episode put 1870 episodes at 22–37 minutes, not the ~2 h you conservatively quoted.**
> **RULED, on the measurement rather than on the convention: if the Gate-0 timing probe confirms the
> whole campaign is under ONE HOUR, background it inside your session. If it comes out over an hour,
> hand off `offline/campaigns/p8_4a.sh` to a user-started `tmux` and return `PARTIAL`.**
> ⭐ **The criterion is measured, not guessed, and it is the same threshold that governs whether a
> pre-flight review is required — one rule, two uses.**

## A3 ✅ Q3 — your scoring rules are registered as proposed, and the reason they are safe is worth stating

`Δ = max(spread_behaviour, spread_arm)` is the **permissive** choice — a larger Δ makes a violation
*harder* to declare. **That would worry me on its own. It does not, because your second rule sets the
ESCALATION trigger at `deficit > 0`.**
> ⭐ **Two thresholds doing different jobs: Δ governs the VERDICT, and any deficit at all — however
> small — governs the ESCALATION to 100 draws. The escalation trigger sitting at zero is exactly what
> makes the permissive Δ acceptable, and neither may be loosened without the other being re-argued.**
> **Registered as yours, before any number.**

## A4 ✅ Q4 — E3 on hz1x1 only, AND grid4x4's flatness is promoted to a first-class result

**Correct, and for the right reason: E3's falsifier — *"a flat profile indicts the replay"* — was
written against an hz1x1 measurement. Applying it to grid4x4 would condemn a correct result**, which
is the class this project refused on 2026-08-19.
> 🔒 **But do NOT report grid4x4's flatness as a scoping exclusion. Report it as a MEASUREMENT with its
> own row**, because per A0 it may be the answer to the question this task was commissioned for.
> **Registered now: on grid4x4, `never_entered` is expected to be 0 or near it for every arm; a
> materially non-zero value there is a finding in its own right and is reported, not absorbed.**

## A5 ✅ Q5 — M3 is out of your fence. The two DOCS sites are mine and are fixed in this commit

`docs/plans/p2.6.md:40` and `docs/PROJECT_PLAN.md:1637` are corrected here.
⚠️ **`offline/policies/plan_replay.py:15` and `:43` are SOURCE and stay untouched** — they carry the
inverted claim in a docstring (*"the engine's native `get_average_travel_time` … is
survivorship-biased"*, *"no survivorship bias"*). **Queued for P8.4b**, because a docstring correction
in a file this task replays through should land with the metric change, not before it.

## A6 🔧 The conflicts — four are my errors

- **C1 ✅ accepted, and it is stronger than what I wrote.** Held-out evals store `att_horizon`; the
  corpus stores `att_per_step`. Check against the committed `att_horizon` with `==`.
- **C2** → A1. · **C5 ✅** your measurement supersedes my 1.684 s. · **C6 ✅** accepted.
- **C3 🚨 MY ERROR.** §2's E3 names `mappo060`, which **is not a P4.6 tier** — it is a corpus tier T1
  happened to measure. **E3 is re-registered over the tiers that exist in P4.6:
  `mappo1000, mappo500, maxpressure, fixedtime, random`**, predicting a *monotone-ish* admission
  profile with `mappo1000` highest and `random` lowest. **The 100.0 / 90.8 / 92.9 / 65.0 figures stay
  as T1's context and are NOT the prediction.**
- **C4 🚨 MY ERROR.** §4 lists `dt` for P5.2; **P5.2's DT arms are `dt_spatial` and `dt_nomix`**. Use
  the arms that exist.
⭐ **Both are the same error twice in one brief: I wrote an arm list from memory of a neighbouring task
instead of reading the artifact. That is the sixth instance of this shape and it is why §2's E2/E3
exist as instrument controls at all.**

---

# ✅ AMENDMENT B — 2026-08-28. The author's question is answered; three rulings, and C7 does not hold

## B0 ⭐ THE ANSWER, and it is favourable on both counts

**`iql@random` on grid4x4 — the cell the author's question named — admits `1.000000`, exactly as its
behaviour anchor.** So do `dt_spatial`, `dt_nomix` and `bc_top10_perix`; `bc` and `bc_top10` sit at
0.999955 / 0.999669.
> ✅ **The correct statement is §6.3's and no stronger: *the ATT gap is not explained by differential
> admission on these cells.* NOT "the result is safe".**
> ⭐ **And on hz1x1's `random` tier, where 34.5 % never enter, `iql` admits +5.39 pp and `dt` +1.43 pp
> MORE than the policy that produced their data — so their advantage there is UNDERSTATED, not
> inflated.** A0's hypothesis holds: the exposure is hz1x1-localised.

## B1 🚨 RULING ON E2 — IT FIRED, IT STAYS FIRED, AND THE INSTRUMENT IS EXONERATED BY BETTER EVIDENCE THAN E2 EVER WAS

`grid4x4/bc_top10@mappo1000` reads **0.967** against E2's registered 0.99 floor, whose falsifier says a
failure there *"indicts the probe before it indicts the science, registered so it cannot be reversed
later"*.

> **RULED, and the two halves must not be run together:**
> **(i) E2 FIRED. It is recorded as fired, in the packet and here. It is not waived, explained away, or
> quietly rescored.**
> **(ii) The PROBE is nonetheless exonerated — by evidence that is strictly stronger than E2:
> 39 / 39 cells reproduce their committed `att_horizon` EXACTLY under `==`, including that very cell,
> and the same tier's behaviour anchor reads exactly `1.000000`. The instrument demonstrably reports
> 1.0 on this tier, on these draws, and reproduces the committed number for the cell in question.**
> **So what fired is not the probe. E2 was MIS-SPECIFIED, and the mis-specification is mine:** §2's E2
> reads *"`mappo1000`'s ARMS sit at ≈100 %"* on the strength of T1 measuring the **behaviour policy**
> at 100.0 %. **A property of one policy, registered as a property of every arm of the tier — A5's
> exact shape, committed by me in the amendment that was correcting A5's shape.**
> ⚠️ **This correction is POST-HOC and it FAVOURS US. It is labelled as both, here and in the packet.
> The implementer flagged that before I did, and declining to explain it away was the right call.**
> ⛔ **What this ruling does NOT do: it does not touch E1. E2's wording being wrong says nothing about
> whether that arm admits fewer vehicles. See B2.**

## B2 ✅ RULING ON E1 — the falsification STANDS and is reported as §2 requires

`bc_top10@mappo1000`, deficit **0.0328** against Δ **0.0118**. **§2's consequence applies unchanged:
that arm's ATT advantage is reported as a possible selection artefact.**

⭐ **And there is a MECHANISM with prior support, which makes this interesting rather than merely a
caveat.** `bc_top10` is the top-decile filter, and **P5.1's Q5 established it is a LOAD SORTER** —
it kept 302 of 320 streams from the **8 quietest of 16 nodes**. **A policy trained on the quietest
nodes' streams plausibly under-serves the busiest approaches, and under-serving a busy approach is
exactly what depresses admission.**
> **Registered as a hypothesis, not an explanation: if the mechanism is real, the admission deficit
> CONCENTRATES on the busiest nodes rather than spreading evenly.** That is checkable from the
> escalation run's per-node data and it is falsifiable. **Do not assert it; test it.**

## B3 ✅ RULING ON THE ESCALATION — run it unchanged, and I considered loosening it and refused

14 cells, 6,200 episodes, **3.44 h**, to `tmux` per A2's threshold. Six of nine arms escalate only
because the trigger sits at zero.
> **RULED: run it as registered.** ⚠️ **I registered the zero trigger deliberately, in A3, as the thing
> that makes the permissive `Δ = max(spread)` acceptable, and wrote that *"neither may be loosened
> without the other being re-argued"*. **Loosening it now, after seeing which arms it catches, is
> precisely the post-hoc move the registration exists to prevent** — and it would be loosening the
> half that protects, while keeping the half that permits.
> **3.44 h of CPU with no GPU is affordable. Recorded so the reasoning is not re-litigated later: if
> the calendar ever binds, this is a candidate to narrow — but not today, and never after seeing the
> results it would exclude.**

## B4 ⛔ C7 DOES NOT HOLD, AND I CHECKED IT BECAUSE OF HOW ALARMING IT WAS

C7 claims P5.1's and P5.2's `grid4x4_mappo1000_{bc,bc_top10,iql}` checkpoints differ such that
*"loading P5.2's would have reproduced nothing"*.
> **Measured by the coordinator across ALL 15 pairs (3 methods × 5 seeds): canonical digests IDENTICAL
> on 15/15, and the ONLY differing payload key is `provenance`. `target_rtg`, `rtg_scale`, `stats`,
> `config`, `intersection_ids` — all identical. Loading P5.2's copy would reproduce the SAME numbers,
> not nothing.**
> ✅ **The underlying observation is real and worth keeping: they are different FILES with different
> sha256, and the reported cells came from P5.1's. That is a PROVENANCE ambiguity — T4's `M-a`, that
> eval cells record no checkpoint path or digest — and it is not a numerical trap.**
> ⚠️ **The consequence was overstated, and "the one that would have produced plausible wrong numbers"
> is exactly the shape of claim that has to be checked rather than accepted. It was found honestly and
> it is a real finding at its true size; the packet must state it at that size.**
> ⭐ *(T4 independently reached the same measurement and called these "harmless duplicates". T4 was
> right.)*

## B5 ✅ Accepted, and one of them is the best incidental catch of the session

- **Gate −1's CWD trap.** `materialise_draws` compares **CWD-resolved absolute paths as identity
  fields**, so run from the worktree it would have reported a **FALSE `BLOCKED`** on 20/20 draws that
  are byte-identical. ⭐ **That is the *condemns-a-correct-artifact* class this project refused on
  2026-08-19, in a gate whose whole purpose is to refuse — the worst place for it.** Recorded as
  `DEFERRED` 61 with the suite's sibling CWD dependency.
- **The two corrected tests** — fixture defects with the implementation correct, no assertion
  weakened, disclosed in full. Accepted. **Disclosing a corrected test in the packet is what makes it
  a correction rather than an edit.**
- **The suite's one failure** — a pre-existing relative-path dependency in `tests/test_rtg_ablation.py`,
  proved to pass from the main tree and untouched. Accepted; it is the same CWD class and joins
  `DEFERRED` 61.
- **11 mutations, 11 caught**, and A0's two effects separated at population scale — **24 fully-admitted
  cells span `att_ours − att_engine` ∈ [+3.25, +7.18], 15 censored cells reach −370.92.** That is the
  M1/B1 decomposition confirmed on 39 cells rather than 4.

---

# ✅ AMENDMENT C — 2026-08-28. The grid4x4 draw restore is authorised, with two binding conditions

**Authorised by the author:** regenerate `cityflow_grid4x4` held-out draws **1010–1099** before the
escalation, using `admission_probe restore-draws`, the same tool, scenario and Gate −1 verification
phase 1 already used and recorded in `output/p8_4a/draw_restoration.json`.

**Why it was needed at all, found before the run rather than during it:** `escalation-plan` names
**14 cells, 6 of them `grid4x4`**, the escalation runs **100 draws**, `cityflow1x1` has 100
materialised and **`cityflow_grid4x4` has 10** — and the `--escalate` path materialises nothing
(`p8_4a.sh:260-288`). grid4x4 is first in the plan, so it would have failed on the first cell.

## C1 🚨 BINDING — all ten survivors must come back `kept`, and the count is reported BEFORE anything else runs

`draw_restoration.json` records all ten of `1000…1009` as `kept` from phase 1. The restore re-verifies
them on the way to writing 1010–1099.

> **If even ONE regenerates differently, STOP. Do not run the escalation, do not continue the task,
> report it immediately.**
> ⭐ **The author's reason, and it is larger than this task: `(source, base_seed, draw_id) → draw` being
> a PURE FUNCTION is the assumption T1's entire recoverability argument rests on** — it is what makes
> ruling (a) cost simulation time instead of re-collection, and it is what `DEFERRED` 55 relied on when
> the hz1x1 draws were regenerated. **A survivor that does not reproduce falsifies that assumption, and
> that is a bigger finding than P8.4a.**
> **So this restore is not a chore. It is a live test of a load-bearing assumption, and its result is
> reported as such — the count, explicitly, before any escalation cell is rolled.**

## C2 🔒 BINDING — the restore is recorded in the packet as a REPAIR, with its reason, never as setup

**Author's ruling, quoted:** *"the restore goes in the packet as a repair with its reason, not as
setup, because `scenarios/draws` being gitignored and per-worktree is what deleted them twice."*

> **`DEFERRED` 55 documents the first loss** — P4.6/P4.7 materialised draws 1000–1099 inside their own
> worktrees, `scenarios/draws/` is gitignored and therefore per-worktree, and retiring those worktrees
> deleted them. **P5.3a repaired hz1x1. This repairs grid4x4. That is twice, by the same mechanism,
> and calling it "setup" would erase the pattern the third occurrence needs.**
> **Required in the packet:** what was missing, why it was missing, that it is the second instance of
> `DEFERRED` 55, and that the restore byte-verified the survivors rather than assuming them.

## C3 ⭐ THE LESSON THE AUTHOR NAMED, recorded because it is now twice in one week

> *"I asked which directory and which path because your main-tree path was wrong, and the answer turned
> out to be 'it does not matter, the script cds itself' — but running `escalation-plan` to answer it is
> what found the six grid4x4 cells with ten draws. The question was about paths and it surfaced a
> run-ending trap."*

**Both instances this week:**
1. **This one.** A question about a **path** → answering it required running `escalation-plan` →
   which enumerated the cells → which exposed a 90-draw shortfall that would have killed the run.
2. **P5.3b's pre-flight.** A question about **process** — *should pre-campaign review be a rule?* →
   the review that followed found `M5`, a chunk carrying a `git_commit` from a dirty tree, which would
   have written false provenance into the artifact.

⭐ **The transferable part is WHY it works, and it generalises `PROJECT_PLAN` §7's falsification rule to
handoffs: answering a delivery question honestly requires EXECUTING the delivery path, and executing
it is what finds the defect that READING it would not.** I could have answered "which path" by reading
`p8_4a.sh` — the header even explains the `cd`. **Reading it would have been correct and would have
missed the trap entirely.** ⚠️ **A delivery detail is not overhead. It is the one part of a handoff
that is always executed, so it is the cheapest place to run a real test.**

---

# ✅ AMENDMENT D — 2026-08-28. The restore refusal: C1 is satisfied in substance, and DEFERRED 61 must be fixed

**The author refused to use `--force` and asked for a ruling. Correct on both counts.**

## D1 ⭐ THE AUTHOR'S INFERENCE DOES NOT HOLD — AND THE CONCLUSION SURVIVES BY A STRONGER ROUTE

**The inference offered:** *"the gate refused on `source_config` and NOT on `source_config_sha256`,
which is in the same block and would also differ if the file differed."*
🚨 **It does not hold. `_existing_conflict` (`offline/materialise_draws.py`) RETURNS on the first
mismatch, and its provenance loop iterates `sorted(set(on_disk) | set(fresh))` — so `source_config`
precedes `source_config_sha256` alphabetically and the sha256 was NEVER REACHED.** The absence of a
sha256 complaint is the loop stopping, not the hashes agreeing. **That is `DEFERRED` 54's mechanism —
*assertions after the first failure never run* — surfacing in a third file.**

> ✅ **But the conclusion is right, by a better argument that sits ABOVE the provenance loop.**
> `_existing_conflict` compares **every rendered file byte-for-byte** — `flow.json`, `cityflow.json`,
> everything except `provenance.json` — and returns `"<name> differs byte-for-byte"` on any
> difference. **It did not return that. It returned a provenance-FIELD complaint, which is strictly
> later.** **So the draw itself is byte-identical and only a metadata path string differs.**
> ⚠️ **Precision that matters for C1: this proves byte-identity for the draw the gate refused ON, and
> for any it cleared before it — not for all ten. C1's ten-survivor verification has NOT happened
> and is still owed.**

## D2 🔧 THE RULING — option C, and A and B are dead ends for stated reasons

- ⛔ **B does not exist.** `p8_4a.sh` contains **no restore invocation** — only a check that
  `draw_restoration.json` is present (`:97`). Gate −1 was run in-session. And every module call in the
  script passes `--repo-root "$WORKTREE"` in `COMMON`, so it would hit the identical wall.
- ⛔ **A cannot work as posed.** The stored `source_config` is **relative** —
  `configs/sim/cityflow_grid4x4.json` — and **no choice of an absolute `--repo-root` reproduces a
  relative string.**
- ✅ **C is the answer, and it is the semantically correct one rather than a workaround.**
  `_NON_IDENTITY_FIELDS` is `{"git_commit", "git_dirty"}`. **`source_config` and `source_roadnet`
  belong in it: the sha256 companions are the identity of the CONTENT, and the path is only the
  identity of WHERE IT HAPPENED TO LIVE.** Comparing where it lived as identity *is* the
  CWD-dependence defect. **The change strictly narrows identity to content — a genuinely different
  file still refuses, because the sha256 fields stay identity fields.**
  > **Required with it, both directions:** a fixture where the **path differs and the sha256 matches**
  > → `kept`; and one where the **sha256 differs** → refuses. ⚠️ **This is merged, reviewed code that
  > P2.2, P4.x and P5.x all used, so the second test is what makes the first safe.**

## D3 ⚡ UNBLOCK NOW WITHOUT TOUCHING MERGED CODE UNDER TIME PRESSURE

**Phase 1's Gate −1 SUCCEEDED and recorded all ten as `kept`, so a known-good invocation exists.**
It must have produced a **relative** `source_config`, matching the stored one.
> **Ask the implementer for phase 1's exact `restore-draws` invocation and reproduce it.** That is
> cheaper and lower-risk than changing identity semantics in merged code while a campaign waits.
> **Then land D2's fix in the same task, so the second sighting in two days is also the last.**

## D4 ⚠️ `git_dirty: True` — the survivors' own provenance is the weak kind

`scenarios/draws/cityflow_grid4x4/draw_1000/provenance.json` carries **`git_dirty: True`**. **These
draws were materialised from a dirty tree.**
> **Harmless for the draw CONTENT — D1's byte comparison settles that, and purity is about
> `(source, base_seed, draw_id) → bytes`, not about the tree state of whoever ran it.** ⚠️ **But it
> means the surviving draws cannot say which code produced them**, and it is the third instance of the
> class **P5.3b's C2 added `git_dirty` detection for**: P5.3b's chunk carrying a commit from a dirty
> tree, `runtime_provenance` having no dirtiness check at all, and now this. **Recorded in
> `DEFERRED` 61.**

---

# ✅ AMENDMENT E — 2026-08-29. Two corrections to me, and the `source_flow` ruling

## E1 🚨 C1's "ALL TEN SURVIVORS" WAS WRONG, AND IT INFLATED THE EVIDENCE IN THE DIRECTION THAT FLATTERS

**C1 required "all ten survivors" to come back `kept` and called it a live test of purity. Only FIVE
are survivors.** `1000–1004` carry `git_commit 29ab244`; **`1005–1009` were written by the implementer
on 2026-08-28 (`git_commit 964d3b0`) and re-verified minutes later in the same tree — their `kept`
tests SAME-SESSION DETERMINISM, not purity.**
> 🚨 **Counting them doubles the apparent evidence base for the one assumption T1's whole
> recoverability argument rests on. That is a sample stated as a population, in a condition I wrote
> to guard against exactly that, and it erred toward more reassurance rather than less.**
> ⭐ **And the right response was not to argue the definition but to run the correct test: regenerating
> all ten from a DIFFERENT worktree and comparing bytes. `flow.json` — THE DEMAND — is byte-identical
> 10/10, differing 0.** **Purity holds, measured cross-worktree. T1's recoverability argument stands,
> and now on evidence that actually tests it.**

## E2 🚨 D1 CONFLATED TWO DIFFERENT REFUSALS, AND ONE OF MY CONCLUSIONS IS FALSE AS STATED

There were **two** refusals and I wrote about "the refusal" as if there were one:

| configuration | refusal | layer |
|---|---|---|
| main-tree CWD + absolute `--repo-root` (the author's) | `source_config differs` | provenance field |
| worktree CWD (the implementer's) | **`cityflow.json differs byte-for-byte`** | **rendered file — strictly EARLIER** |

**D1's reasoning is correct about the author's refusal and I generalised it to both.** ⚠️ **And the
conclusion I drew from it — *"the draw itself is byte-identical"* — is TRUE OF `flow.json` AND FALSE
OF `cityflow.json`**, which embeds a CWD-resolved directory and differs **10/10** across worktrees.
> ✅ **The correct decomposition, which the implementer measured: the DEMAND (`flow.json`) is pure and
> path-independent; the CONFIG WRAPPER (`cityflow.json`) is path-dependent by construction. Purity is a
> property of the demand, and that is the property T1 needs.**

## E3 ✅ RULING — `source_flow` IS extended into the set. The omission was mine

**Measured: all three path-like fields — `source_config`, `source_flow`, `source_roadnet` — have a
`_sha256` twin, and only two were exempted.** ⚠️ **And `source_flow` is stored ABSOLUTE
(`/home/filip/rltraffic/scenarios/grid4x4/grid4x4_flow.json`), so it is not a latent case — it is the
NEXT field to fire from any other tree.**
> **RULED: `_NON_IDENTITY_FIELDS = {git_commit, git_dirty, source_config, source_flow,
> source_roadnet}`, with the same both-directions test D2 required — path differs and `source_flow_sha256`
> matches → `kept`; `source_flow_sha256` differs → refuses.**
> ⭐ **Implementing the ruling AS WRITTEN and pinning its gap with a test, rather than extending it
> silently, was exactly right and I want it on the record.** A coordinator's incomplete instruction
> silently "improved" by an implementer is a ruling nobody can audit; a gap pinned by a test is a
> question I have to answer. **Do that every time.**

## E4 🔧 `DEFERRED` 61 (a)'s REMAINING HALF — make the worktree failure LEGIBLE rather than fix the embedding

`cityflow.json` embeds a CWD-resolved directory because CityFlow requires an absolute `dir`. **Do not
normalise it — that changes rendering semantics in merged code for a diagnostic's convenience.**
> **RULED: `restore-draws` detects a CWD that is not the main tree and refuses with a message that
> SAYS SO**, instead of the current `cityflow.json differs byte-for-byte`. ⚠️ **The harm is the
> misreading, not the refusal: a reader sees "differs byte-for-byte" on a file whose name contains the
> scenario and concludes the DEMAND changed. It did not — `flow.json` is identical 10/10.** A refusal
> that names its own cause is the whole repair here.

## E5 ✅ The restore may proceed NOW; the fixes are durable repairs, not blockers

**The invocation is verified and its three load-bearing parts are understood** — main-tree CWD (makes
the rendered `cityflow.json`'s absolute dir match), `--repo-root .` (yields the relative
`source_config`), `PYTHONSAFEPATH=1` (stops the main tree's `offline` shadowing the worktree's) — and
the implementer dry-ran `1000 / 1005 / 1010 / 1011` with no refusal.
> **Run it. E3 and E4 land in the same task as durable repairs.** ⚠️ **And when it runs, the evidence
> to report is the FIVE genuine survivors' `kept`, with `1005–1009` reported separately as
> same-session re-verification — per E1, not pooled.**

---

# ✅ AMENDMENT F — 2026-08-29. The escalation is scored. How it goes in the packet

**14 cells, 6,200 episodes, reference checks exact 14/14.** Read by the coordinator from
`docs/data/p8_4a_admission_escalated.json`, not from the campaign log.

## F1 ⭐ THE AUTHOR'S QUESTION IS ANSWERED AT 100 DRAWS, AND THE ANSWER HELD

**`iql@random` on grid4x4 was not escalated at all — its deficit was ≤ 0 at ten draws, so it HOLDS.**
The grid4x4 `random` arms that were escalated come back at **0.99991 (`bc`)** and **0.99955
(`bc_top10`)** against a **1.000000** anchor.
> ✅ **P5.2's headline is not a metric artefact of this kind, now on 500 episodes per cell rather than
> 50.** The sentence stays §6.3's: *the ATT gap is not explained by differential admission on these
> cells.*

## F2 🚨 E1: 1 HOLDS, 6 CLOSE, 2 FALSIFIED — and the two are NOT comparable in size

| arm | arm | anchor | deficit | Δ | status |
|---|---|---|---|---|---|
| **grid4x4/bc_top10@mappo1000** | 0.970454 | 1.000000 | **0.029546** | 0.009596 | **falsified** |
| hz1x1/bc_top10@fixedtime | 0.890840 | 0.907449 | 0.016609 | 0.026875 | close |
| hz1x1/bc_top10@random | 0.648887 | 0.656212 | 0.007325 | 0.030341 | close |
| hz1x1/bc@fixedtime | 0.904997 | 0.907449 | 0.002452 | 0.002883 | close |
| grid4x4/iql@mappo1000 | 0.999168 | 1.000000 | 0.000832 | 0.001145 | close |
| **grid4x4/bc_top10@random** | 0.999550 | 1.000000 | **0.000450** | 0.000354 | **falsified** |
| hz1x1/bc@random | 0.655833 | 0.656212 | 0.000379 | 0.030341 | close |
| grid4x4/bc@random | 0.999910 | 1.000000 | 0.000090 | 0.000218 | close |
| hz1x1/dt@maxpressure | 0.999913 | 0.999906 | −0.000007 | 0.000055 | **holds** |

🚨 **BINDING ON THE PACKET: never write "2 of 9 falsified" without the magnitudes.** The two are
different findings wearing one label:
- **`bc_top10@mappo1000` is SUBSTANTIVE** — a **2.95 %** admission deficit, three times its Δ. **Its
  ATT advantage is reported as a possible selection artefact, per §2.**
- **`bc_top10@random` is a TECHNICALITY** — **0.045 %**, roughly one vehicle in 2,200. It is falsified
  only because both arms are so consistent that Δ collapses to 0.00035. ⚠️ **Reporting it beside the
  first without that distinction would overstate our own problem, and a referee would find the
  imprecision immediately.**
> **Registered reading: report deficit, Δ, AND the deficit as a fraction of the anchor, for every arm.
> The verdict word is not sufficient on its own.**

## F3 ⭐ B2's HYPOTHESIS IS SUPPORTED AT THE ARM LEVEL, AND ITS ACTUAL TEST IS STILL OWED

**Both falsified arms are `bc_top10`, and the three largest deficits in the whole table are `bc_top10`**
(0.0295, 0.0166, 0.0073). Amendment B2 registered, before this data existed, that `bc_top10` is
**P5.1's load sorter** — it kept 302 of 320 streams from the **8 quietest of 16 nodes** — and that a
policy trained on the quietest nodes plausibly under-serves the busiest approaches.
> ✅ **The arm-level pattern is exactly what that predicts, and it was registered in advance.**
> ⛔ **But B2's prediction was PER-NODE — *"the deficit CONCENTRATES on the busiest nodes rather than
> spreading evenly"* — and that has NOT been tested.** ⚠️ **An arm-level correlation is consistent with
> the mechanism and does not establish it; three arms of one method is also consistent with something
> particular to `bc_top10`'s training that has nothing to do with load.** **The per-node test is owed
> and the packet must say it is owed, not imply it is done.**

## F4 ✅ E2 — fired at ten draws, fires at a hundred, and Amendment B1's ruling is unchanged

`grid4x4/bc_top10@mappo1000` at **0.9705** against the 0.99 floor; **1 of 3 below.** **B1 stands: E2
fired and stays fired; the PROBE is exonerated by evidence stronger than E2 — now 14/14 exact
reference checks including that cell — and E2's wording was mine, generalising a behaviour-policy
measurement to every arm of the tier.** Nothing here changes that ruling; it confirms it at scale.

## F5 ⭐ E3 reports `holds: None`, and that is the honest answer rather than a gap

`scored_scenario: hz1x1`, `holds: None`, `monotone: True`, rank order
**`maxpressure > fixedtime > random`** at **0.99991 → 0.90745 → 0.65621**.
**`None` is correct: Amendment A6 re-registered E3 over five P4.6 tiers and only three were escalated,
so it cannot be scored as registered.** Reporting `None` rather than `True` on the three available is
right and must not be quietly upgraded.
> ⭐ **But the profile is a FIRST-CLASS RESULT in its own right, and it is the mechanism behind B1
> measured directly: ADMISSION TRACKS DATA QUALITY. The weaker the behaviour policy, the more vehicles
> its rollouts never admit — 0.01 %, 9.3 %, 34.4 % — which is precisely why our ATT metric flatters
> weak policies and why the C1 ladder is where the distortion lives.** Report it as a measurement of
> the mechanism, not as a scored prediction.

## F6 📋 What the packet must carry

1. F2's full table **with magnitudes and fractions**, and the substantive/technical distinction stated.
2. F1's answer in §6.3's words, and **that `iql@random` held without escalation**.
3. F3's hypothesis status: **supported at arm level, per-node test owed.**
4. F5's profile as a measurement; E3's `None` with its reason.
5. **The restore as a repair** (C2), with the **five genuine survivors reported separately from
   1005–1009** (E1 of Amendment E).
6. The cross-worktree purity result: **`flow.json` byte-identical 10/10.**
7. `DEFERRED` 61's remaining half: `source_flow`, and the worktree-CWD refusal message (E3/E4 above).

---

# ✅ AMENDMENT G — 2026-08-29. E2's final disposition, and the 0.045 % reporting call

## G1 🔒 E2 — FINAL DISPOSITION. It fired, it is permanent, and it does NOT block the merge

B1 and F4 analysed E2; what was missing is a **disposition**, so here it is in one place and it is
final.

> 1. **E2 FIRED** — `grid4x4/bc_top10@mappo1000` at **0.9705** against a 0.99 floor, at ten draws and
>    again at a hundred. **It is recorded as fired, permanently, in the artifact and the packet. It is
>    not waived, rescored, or quietly re-registered.**
> 2. **E2 was a control on the INSTRUMENT, and the instrument is independently exonerated:
>    14/14 reference checks exact under `==`, including that very cell.** The probe reproduces
>    committed `att_horizon` for the cell whose admission it questions.
> 3. **E2's WORDING was mine and was wrong.** It reads *"`mappo1000`'s ARMS sit at ≈100 %"* on the
>    strength of T1 measuring the **behaviour policy** — one policy's property registered as every
>    arm's. **A5's shape, in the amendment correcting A5's shape.**
> 4. **Therefore: E2's firing is a finding about E2, not about the probe, and it does not block the
>    merge. Task status DONE stands.**
> ⚠️ **Stated every time it is mentioned, including in the packet: this correction is POST-HOC and it
> FAVOURS US.** ⛔ **And it does not touch E1 — the `bc_top10@mappo1000` falsification stands on its
> own evidence and is reported as a possible selection artefact.**

## G2 ⚖️ THE 0.045 % CALL — the rule's output stands verbatim, and the PAPER reports materiality, which is a different question

`grid4x4/bc_top10@random`: deficit **0.000450** against Δ **0.000354** — **299 vehicles in 663,785**,
about **0.6 per episode**. Falsified only because both arms are so consistent that Δ collapses.

> **RULED, three parts:**
> **(a) The rule's output is reported verbatim, everywhere it appears.** `falsified` stays `falsified`
> in the artifact and the packet. ⛔ **No re-definition, no threshold change, no "effectively holds".
> Loosening a registered rule after seeing which arms it catches is the move the registration exists
> to prevent, and this is the second time this task I have declined it.**
> **(b) The PAPER reports MATERIALITY, which is a different question from DETECTABILITY.** E1's Δ asks
> *can we detect a difference*; the paper asks *could this difference explain the ATT gap*. **Those
> have different answers here and both are honest.** The paper's sentence is about the substantive
> arm; the 0.045 % arm appears in the artifact and the packet with its magnitude, not as a headline.
> **(c) 🚨 DO NOT ASSERT THAT IT IS IMMATERIAL — COMPUTE IT.** *(This is the correction to my own
> instinct: I was about to write "negligible".)* **Required in the packet: the ATT sensitivity to that
> deficit, computed** — what `att_ours` would become if those 299 vehicles were admitted and counted,
> against the arm's reported ATT. **If the bound is far below the ATT differences the paper quotes,
> that is a measured statement and it is worth more than the word "negligible". If it is not, we have
> learned something and the reporting changes.**

## G3 ✅ The removed test — approved, and the reasoning is the part worth keeping

`test_source_flow_is_still_an_identity_field_...` asserted **the opposite of E3's ruling**. **Deleting
it rather than editing it into agreement was correct**: a test edited to match a new spec is
indistinguishable from a test weakened to pass, while a deletion with a comment naming its replacement
is auditable. **Disclosed in §11.2, net `source_flow` coverage up, the "path differs → kept" test now
parametrised over all three fields.**
> ⭐ **And the pinning test now asserts the invariant that made my omission visible in the first place
> — every exempted path has a `*_sha256` twin, and no twin is exempted. That generalises the fix from
> three named fields to a property, so a fourth path field added later cannot repeat this.** That is
> fixing the class rather than the instance.

## G4 ⭐ Two disclosures that are better than the things they disclose

- **The `git add -A` that swept my escalation artifact.** Caught from an unaccounted 139,575-line diff,
  verified independently (14/14 references exact, 0 identity violations, 3,200 hz1x1 values equal to
  their committed originals), every number labelled as the author's run. ⚠️ **Symmetry worth stating:
  this is the same command that swept the implementer's plan draft into my commit `094b53f` on
  2026-08-27. Twice in three days, in both directions. `git add -A` is how work you did not do enters
  a commit you signed. Name the files.**
- **E4's verification detail.** The refusal test patches `offline.materialise_draws.materialise` rather
  than `admission_probe.materialise`, *"which would have passed vacuously, since the import is
  function-local."* ⭐ **Knowing WHY the obvious patch target would have made the test theatre is the
  difference between a test and a decoration**, and it is the kind of thing that usually only surfaces
  in a review.

---

# ✅ AMENDMENT H — 2026-08-29. G2(c) accepted, and a stronger second route is already in your data

**G2(c) is discharged and it did what it was set for: the word "negligible" was replaced by a number,
and the number is smaller than the word would have implied.** Merge review is running.

## H1 ✅ The bound is accepted, and reporting the DISTRIBUTION rather than the mean was the right call

**425 of 500 episodes have zero never-entered vehicles; 75 have some; the worst has 20.** ⭐ **"0.6 per
episode" would have been a mean over a shape that is mostly zeros, which is P5.1's lesson — *a mean
over those five is a summary that hides its own subject* — applied without being told.**
**−0.187 / +1.420 s on a reported 350.339805, interval width 1.608 s, against gaps of 70–90 s on this
tier: 1.79 % of the arm's own gap and 2.29 % of P5.2's headline gap.** Accepted.
✅ **And `enterTime` is confirmed by the coordinator at `CityFlow/src/vehicle/vehicle.cpp` — it is set
inside the `Vehicle` CONSTRUCTOR, so `att_engine` does count never-admitted vehicles at their full
wait.** The independent route's premise holds.

## H2 ⚠️ THE BAND ARGUMENT IS WEAKER THAN IT READS — and the data already contains a better one

The packet argues that this cell's `att_ours − att_engine` of **+6.3684** sits inside
**[+3.2469, +7.1809]**, *"the band set by the 24 cells with no censoring at all"*.
> ⚠️ **That band comes from the PRIMARY 39-cell artifact, not the escalated one. Within the escalated
> artifact only TWO cells are uncensored, giving [+7.1377, +7.1840] — and +6.3684 is BELOW it.**
> Borrowing the wider band is legitimate, but the argument is doing less work than it appears to, and a
> reviewer who recomputes it inside the escalated artifact will find the opposite sign of conclusion.

⭐ **The stronger argument is measured, monotone, and sits in the escalated artifact itself — grid4x4,
ordered by censoring:**

| cell | never entered | `att_ours − att_engine` |
|---|---|---|
| behaviour@mappo1000 | 0 | **+7.1840** |
| behaviour@random | 0 | **+7.1377** |
| bc@random | 60 | +6.8758 |
| **bc_top10@random** | **299** | **+6.3684** |
| iql@mappo1000 | 552 | +5.8877 |
| bc_top10@mappo1000 | 19,612 | **+2.4375** |

> ⭐⭐ **A clean dose-response: the offset falls monotonically as censoring rises, from +7.18 at zero to
> +2.44 at 19,612. That is B1's mechanism measured directly on six cells of one scenario.**
> **And it turns the bound into a corroborated measurement: the zero-censoring level is +7.1609, this
> cell sits at +6.3684, so the MEASURED censoring effect is 0.7924 s — against the packet's ANALYTIC
> upper bound of 1.420 s.** ✅ **Two independent routes agree, and the measurement sits INSIDE the
> bound, which is what a correct upper bound must do.**
> **REQUIRED: replace the band-membership sentence with this. It is stronger, it is internal to the
> artifact under discussion, and it demonstrates the bound is conservative rather than merely asserting
> it.** ⚠️ **Report the zero-censoring level as n=2 within this artifact; do not present +7.1609 as a
> population constant.**

## H3 ✅ The honesty ledger entry is correct and stays

*"The AI-assistance record and the self-review checklist now say **partly** on 'every number produced
in this session' — the 6,200-episode escalation was the author's run, verified by me, and it is
labelled as his everywhere it appears."*
> ⭐ **That is the right answer to CLAUDE.md §2's rule, not a weakening of it. The rule exists so a
> number's producer is knowable; "partly", with the exception named and labelled, satisfies it exactly.
> A `Y` would have been false and an `N` would have been misleading.**

## H4 ✅ Catching your own checklist row was the best small thing in this packet

The row claimed *"`negligible` appears nowhere"*; the word appeared **twice**, both times naming itself
rather than making a claim. **You grepped, found it, and fixed the row instead of the word.**
> ⭐ **A checklist row that is literally false is worth more attention than the thing it describes,
> because the row is what the next reader trusts.** This project has logged three ticked-but-false
> boxes; this is the first one caught by its own author before anyone else read it.

**Nothing else changes. Await the review.**

---

# ⛔ AMENDMENT I — 2026-08-29. FIX-FIRST. `docs/reviews/P8.4a.md`: PASS-WITH-NOTES, 2 blocking

**Do not merge.** Both blockers are text and provenance rather than numbers — **and both would mislead
the reader of the record, one of them about whether to release a held campaign.** The reviewer's own
summary stands: *"the defects are in claims about the work, not in the work's numbers."*

## I1 🚨 BL-1 — §7's prose contradicts §7's own table, in the paragraph that gates P5.3b

**Coordinator-verified: grid4x4 has FOUR censored cells of fourteen, the largest being
`bc_top10@mappo1000` at admission 0.967168 (2183 never entered) — the arm this task calls
SUBSTANTIVE.** The sentence *"grid4x4 is at ≈100 % everywhere"* is false.
> **Required: rewrite §7's prose to match its own table, and state the grid4x4 censoring explicitly
> with its four cells.** ⚠️ **The correct claim is COMPARATIVE, not absolute: censoring on grid4x4 is
> two orders of magnitude smaller than on hz1x1's weak tiers (3.3 % against 34.5 %), which is what
> supports the hz1x1-localised reading — NOT that grid4x4 is clean.** ⭐ **My own Amendment A0 hedged
> this correctly (*"a hypothesis to test, not a result"*); the packet hardened the hedge into an
> absolute. Restore the hedge.**

## I2 🚨 BL-2 — the artifacts record another task's commit, AND THE CAUSE IS MY OWN A2 RULING

Both artifacts carry `git_commit = 53e995da…`. **Verified: that commit does not contain
`offline/admission_probe.py` and it lives on `task/p5.3b-nortg-campaign`.**
> 🚨 **I ruled in A2 that the campaign runs with CWD = the main tree, to defuse `DEFERRED` 61. The main
> tree's HEAD is the P5.3b branch, and `runtime_provenance()` reads `git rev-parse HEAD` from the
> process CWD. So the fix I approved put a foreign branch's commit into two committed artifacts. Fourth
> sighting of this class, and the first one I caused.**
> **Required, in this order:**
> 1. **Disclose it in the packet** — what the field says, why it says it, and that it is a consequence
>    of A2's main-tree requirement.
> 2. **Record the TRUE code provenance** in both artifacts: the worktree's `HEAD` at measurement time,
>    plus `git_dirty`, plus the `PYTHONPATH` the code was actually imported from. **The information
>    exists; nothing captured it.**
> 3. ⛔ **Do NOT fix this by moving the CWD back to the worktree** — that reopens `DEFERRED` 61's false
>    `BLOCKED`. **The CWD and the code root are two different things and the provenance must record the
>    second, not the first.**

## I3 ⚠️ MJ-1 — MY AMENDMENT G3 WAS WRONG, and I am withdrawing the praise

G3 said the pinning test *"generalises the fix from three named fields to a property"*. ✅ **Verified at
`tests/test_materialise_draws.py:435-441`: the exact-set assertion pins the set to a five-element
literal containing no `_sha256` name, so the loop that follows CANNOT FAIL. It is dead code, it
iterates a hardcoded 3-tuple rather than the set, and it never asserts the twin exists.**
> **G3's second paragraph is WITHDRAWN. The exact-set pin and the parametrised digest-refusal test are
> load-bearing and were correctly identified; the "property" was not asserted anywhere and I said it
> was.** ⚠️ **I praised a test for fixing the class after telling the implementer to fix the class —
> which is the most expensive kind of wrong praise, because it closes the question.**
> **Required: make it real or delete it.** The honest version asserts over the SET, not a literal
> tuple: every member ending in `_sha256` is absent, and every exempt path field has its twin present
> in a real provenance record. **MJ-2 says the second half is FALSE of the record today** (`sumo.sumocfg`
> and `sumo.template_rou` are absolute paths with no twin), **so write the test that fails, then decide
> whether to exempt them or to add twins.**

## I4 📋 The rest

- **MJ-3 — cover the wiring.** `admission_artifact` is never named in the test file, so the verdict
  guard, the two-grain refusal and the created-consistency refusal are decorative. ⚠️ **Priority is
  E3's `holds = None → True` mutation surviving 49/49: that is the guard against the exact upgrade F5
  and G forbade.** Kill all five survivors with the failure pasted.
- **MJ-4** — the grid4x4 row is off by one: 10 of 14 cells are zero, not 11.
- **MINOR 1** — the CWD check returns on the first existing draw; a heterogeneous pool defeats it, and
  `DEFERRED` 55 keeps producing exactly those. Check all, or say in the message that it checked one.
- **MINOR 2** — `p8_4a.sh:19-21` and §4 still say the path fields are identity fields. **They are not,
  since E3. Correct them** — a stale comment that describes the pre-fix world is how D1's misreading
  happened.
- **MINOR 6 — accept the correction and change one word.** G2(c)'s bound is a **re-accounting** bound:
  say *"if the 299 were COUNTED"*, not *"admitted"*. **Admitting them would change the other vehicles'
  travel times; counting them does not. The arithmetic is right and the word overreached** — and the
  word was mine, in G2(c).
- **MINOR 3, 4, 5, 7** — fix the timing table's provenance column, the artifact operator/commit fields
  (folded into I2), the `restored` field name, and the internal counts.
- **Merge gate:** §6's `P8.4a` box, ticked in the merge commit.

## I5 ⭐ Carried forward to P8.4b, and it is the most important line in the review

> **`att_engine` has NO external reference. Nothing in the repo pins it. T1 validated the
> engine-metric reconstruction on hz1x1 ONLY, and grid4x4's `att_engine` is unvalidated against any
> independent route — while A0's decomposition, §7's conclusion and G2(c)'s second route all rest on
> it.**
**P8.4b's first obligation is to close that**, by the same hand-reconstruction T1 used on hz1x1,
applied to grid4x4. ⚠️ **We are one unvalidated quantity away from the ruling-(a) re-derivation resting
on a number nobody has checked — which is precisely the shape of the defect this whole sequence began
with.**

---

# ✅ AMENDMENT J — 2026-08-29. MERGED at `c6ba1eb`. MJ-2 ruled, `code_dirty` approved

**Both blockers verified closed by the coordinator from the artifacts, not the packet:** `cf569af`
contains `admission_probe.py` and is on the branch, with `53e995d` **kept alongside** as evidence of
the class rather than deleted; and §7's false sentence survives **only as the retracted claim**, with
the four censored grid4x4 cells tabulated. **Tenth manifest written: `SHA256SUMS_p8_4a.txt`, 111/111.**

## J1 ⚖️ MJ-2 — LEAVE AS-IS, PINNED. Neither option is right today, and the reason is a migration

`sumo.sumocfg` and `sumo.template_rou` are absolute CWD-resolved paths with no digest twin, inside a
still-identity `sumo` dict. **You were right to pin the current state and hand the choice up rather
than pick one.**

> ⛔ **Giving them twins is NOT a two-line fix — it is a provenance FORMAT MIGRATION.**
> `_existing_conflict` iterates `sorted(set(on_disk) | set(fresh))`, so **a field present in the fresh
> record and absent from an existing one compares `None != digest` and REFUSES.** Adding
> `sumocfg_sha256` would make **every existing draw refuse on first comparison** — a versioned-record
> change with a migration, not a repair.
> ⛔ **Exempting them is worse: it removes identity with nothing replacing it**, and unlike the three
> path fields E3 exempted, **the SUMO sources are not rendered into any compared file**, so no digest
> covers their content by another route.
> ✅ **RULED: leave both as identity fields, keep your test pinning the exact current state, and record
> the choice as OWED to whoever first needs SUMO draws — P7.x.** ⚠️ **The defect is latent: it needs a
> SUMO-paired scenario AND a cross-tree comparison, and SUMO work is October.** **Deciding a
> format migration now, under time pressure, for a code path we may not use, is the wrong trade — and
> `DEFERRED` 61 carries it.**

## J2 ✅ `code_dirty` — approved, and it is a better distinction than the one I asked for

**P5.3b's C2 asked for dirtiness detection and did not distinguish WHAT was dirty.** Regenerating two
artifacts in sequence makes the second see the first as an uncommitted change, so a whole-tree flag
reads dirty for a reason that has nothing to do with the code.
> ⭐ **`git_dirty` answers *was the tree modified*; `code_dirty` answers *was the CODE modified* — and
> only the second bears on whether the artifact's numbers are reproducible.** Recording both, with
> `code_dirty` excluding `docs/`, is right. **Pinned by a three-step test, which is what makes it a
> field rather than a hope.** ✅ **Adopt this shape wherever `runtime_provenance` is used.**

## J3 ⭐ The third `git add -A` earns a rule, not a fourth disclosure

*"What saved it was where the output happened to land, not a control."* **That sentence is the
finding.** Three instances in four days, both directions, none caught by a control.
> **`PROJECT_PLAN` §7 now forbids `git add -A` and `git add .`, requires named paths, and names the
> control: `git status --porcelain` before committing, `git show --stat` before pushing.** ⚠️ **And the
> specific trap in your instance: a `cd` DOES persist within one compound command even though shell
> state does not persist between tool calls. Use `git -C`.**
> ⭐ **Disclosing a near-miss that cost nothing is worth more than disclosing a failure that cost
> something, because it is the only evidence available before the expensive instance.**

## J4 📋 Carried to P8.4b, unchanged and first in line

**`att_engine` has no external reference on grid4x4.** T1 reconstructed the engine metric by hand on
hz1x1 only; **A0's decomposition, §7's conclusion and G2(c)'s second route all rest on grid4x4's
`att_engine`, which nothing has checked.** **P8.4b's first obligation, before any re-derivation.**

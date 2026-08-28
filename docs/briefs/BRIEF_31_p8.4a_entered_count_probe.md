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

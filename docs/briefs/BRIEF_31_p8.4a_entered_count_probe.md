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

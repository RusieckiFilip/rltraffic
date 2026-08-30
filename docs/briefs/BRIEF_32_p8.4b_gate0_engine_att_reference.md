# BRIEF_32 — P8.4b Gate 0: an independent reference for CityFlow's own ATT

**Task id:** `P8.4b-G0` · **Branch:** `task/p8.4b-g0-engine-att-reference`
**Mode:** Claude Code, in the **main tree** (`/home/filip/rltraffic`), on a task branch.
**Supersedes:** nothing. **Governed by:** `PREREGISTRATION.md` **A11**, tag `v1.1-prereg-a10-a11`.

> ⚠️ **This brief is self-contained and supersedes anything you may read elsewhere about "hand-reconstructing
> `att_engine` on grid4x4". That earlier phrasing predates A11.** What is required is A11's
> **ENGINE-SEMANTICS GATE**, whose four criteria are registered and are reproduced verbatim in §5.

---

## 0. Frozen interface contracts — read these, do not infer them

```python
info = env.reset(seed=42)                                # returns info ONLY (no obs)
reward, terminated, truncated, info = env.step(action)   # reward FIRST
```

- `envs/**` is **FROZEN**. `envs/cityflow_env.py` is frozen. **You may not edit it.** You subclass it.
- `offline/admission_probe.py` is **MERGED, REVIEWED, and its artifacts are cited in `PROJECT_PLAN` §10
  and in A11 itself.** You may **import from it and call it**. You may **not modify it.**
- `agent/utils/utils.py` helpers (`Utils.seed_everything`, ...) are to be reused, not reimplemented.
- No new dependencies. numpy / stdlib / torch / pytest only.
- Python ≥3.12, `from __future__ import annotations`, full type hints, float32 for stored float arrays.
- **Language: everything on disk is English.** `scripts/check_english.sh` (no-argument form) enforces it.

---

## 1. Why this task exists

`docs/reviews/T1-metric-ground-truth.md` (**FAIL, 4 blockers**) established that our primary metric and
CityFlow's own metric are **different quantities**, diverging policy-dependently by up to 2×, and
**inverting the ranking of two corpus behaviour policies in 7 of 8 matched draws**. A11 registered the
response: every ATT cell carries both definitions, and **which definition the paper's claims rest on is
decided by `RULE R`, per scenario, from this gate's output.**

**The specific hole this task fills, and it is narrow:** T1 reconstructed the engine's quantity by hand
**on hz1x1 only**, using a script that lived in `/tmp`, no longer exists, and was never reviewed as code.
Meanwhile **`offline/admission_probe.py:558` obtains `att_engine` by calling
`env._eng.get_average_travel_time()` directly** — so it is the engine's own number and nothing in this
repo checks what that number *means* on any network.

🚨 **A11's `Rule R` binds hz1x1 as well as grid4x4, and the reason is consistency rather than doubt:
T1's 0.000e+00 is strong evidence and it is not a passed gate.** Requiring four criteria of grid4x4
while accepting a vanished script for hz1x1 is an inconsistency a referee finds in one question.

⚠️ **What is at stake if this is got wrong:** A0's decomposition, P8.4a's §7 conclusion and G2(c)'s
second route **all rest on grid4x4's `att_engine`.** We would be one unvalidated quantity away from the
whole re-derivation resting on a number nobody checked — **the exact shape of the defect that started
this.**

---

## 2. What the engine actually computes — READ THIS BEFORE DESIGNING ANYTHING

Verified in source by the coordinator, `CityFlow/src/engine/engine.cpp:682-691`:

```cpp
double Engine::getAverageTravelTime() const {
    double tt = cumulativeTravelTime;      // finished vehicles' completed travel times
    int    n  = finishedVehicleCnt;        // how many have finished
    for (auto &vehicle_pair : vehiclePool) {          // everything NOT yet finished
        auto &vehicle = vehicle_pair.second.first;
        tt += getCurrentTime() - vehicle->getEnterTime();
        n++;
    }
    return n == 0 ? 0 : tt / n;
}
```

**Three facts that follow, and each one shapes the reconstruction:**

1. **The denominator is `finishedVehicleCnt + |vehiclePool|`** — i.e. every vehicle that has been pushed
   into the simulation, whether or not it was ever admitted to a lane. There is **no filter**, unlike
   `getRunningVehicles` (`:780-786`), which filters on `isReal() && (includeWaiting || isRunning())`.
2. **A vehicle waiting in the insertion buffer contributes `currentTime − enterTime`** — the full time
   since it was pushed. That is the entry-queue penalty our metric discards.
3. ⚠️ **`vehiclePool` also contains non-`isReal()` shadow vehicles**, which the ATT sum includes and
   `get_vehicles()` excludes. ✅ **Coordinator-verified as INERT on the population, not a sample: all
   13 CityFlow sim configs run with `laneChange` false** — 11 set it explicitly and `cologne1`/`cologne3`
   omit it, taking `engine.cpp:53`'s default of `false`. **You must re-assert this in a test rather than
   inherit it**, because it is the premise that makes the reconstruction possible at all.

**Also measured and relied on:** `interval = 1.0` in all 13 configs, so one engine step is one simulation
second and per-second snapshots align exactly with engine steps.

---

## 3. The seam you will use — it exists, and it is not frozen-adjacent

`envs/cityflow_env.py:182-185` is the per-second advance loop:

```python
def _simulate(self, num_steps: int) -> None:
    assert self._eng is not None
    for _ in range(num_steps):
        self._eng.next_step()
```

**Subclass `CityFlowEnv` in `offline/` and override `_simulate`** to snapshot around each `next_step()`.
This gives a genuine 1-second observation grid without touching frozen code. *(This is what T1's reviewer
did; the difference is that yours is in the repo, tested, and reviewed.)*

⚠️ **Do NOT snapshot at env-step boundaries.** `delta_time` is 10 s, so an env-step grid quantises
enter/leave times to 10 s — which is **M1's quantisation defect**, i.e. exactly the error the reference is
supposed to be free of. A reference built on a 10 s grid will fail criterion (1) and you will not know why.

---

## 4. The reconstruction, specified

From per-second snapshots of `get_vehicles(include_waiting=True)` — the set of `isReal()` vehicles in the
pool — maintain, keyed by vehicle id:

- `first_seen[v]` — the simulation time at which `v` first appears. **Proxy for `enterTime`.**
- `last_seen[v]` — the last simulation time at which `v` appears. `v` finished at `last_seen[v] + interval`.

At the horizon `T`:

```
contribution(v) = (finish[v] - first_seen[v])   if v is absent at T   (it finished)
                = (T         - first_seen[v])   if v is present at T  (still in the pool)

ATT_reference = sum(contribution(v) for v in all ids ever seen) / count(all ids ever seen)
```

🚨 **`first_seen` as a proxy for `enterTime` is THE load-bearing premise of this entire task, and it is an
ASSUMPTION until you test it.** It holds only if a vehicle appears in `get_vehicles(include_waiting=True)`
on the same engine step it is pushed into `vehiclePool`. **Test it; do not reason about it.** If it does
not hold, say so and stop — a `BLOCKED` packet is the correct outcome and is worth more than a
reconstruction that agrees for a reason nobody checked.

**You also build the ENTERED-ONLY variant**, identical except that it observes
`get_vehicles(include_waiting=False)` and admits a vehicle at the first second it is *running*. This is
our metric's population, and it exists to serve criterion (3).

---

## 5. The four gate criteria — registered in A11, reproduced verbatim, each reported with its observed value

| # | Criterion | Threshold |
|---|---|---|
| **1** | **AGREEMENT.** Max absolute deviation between `ATT_reference` and `eng.get_average_travel_time()` over the gate's episodes | **< 1e-4 s** |
| **2** | **DENOMINATOR.** The reconstruction's vehicle count equals `created` (= `entered` + `never_entered`) from `admission_probe` | **exact, every episode** |
| **3** | **DISCRIMINATING POWER (positive control).** The entered-only variant differs from the engine value by exactly the `att_difference` `admission_probe` independently reports | **within 1e-4 s, every episode** |
| **4** | **COVERAGE.** Behaviour tiers × draws, including the min and max `entered_fraction` episodes available on the scenario | **≥7 tiers × ≥3 draws** |

**Why 1e-4 s, so you do not treat it as arbitrary:** the smallest gap between the two definitions anywhere
in the 1,870 cells of `docs/data/p8_4a_admission.json` is **0.031699 ATT**; the pooled median is
**6.8295** and the maximum **423.73**. **1e-4 sits 317× below the smallest semantic difference we have
ever measured**, so it cannot mistake one for agreement. **Report the observed maximum deviation** — if it
comes back at 1e-12 we had eight orders of headroom; if it comes back at 9e-5 that is a warning worth a
sentence even though it passes.

⚠️ **A deviation landing between 1e-4 s and 1e-2 s is a FAIL and must be root-caused before any re-run.**
It is neither float noise (~1e-7 for a sum of this size) nor any semantic difference we have measured.

**Criterion (3) is the one that makes (1) mean anything.** A reconstruction that agrees with the engine
but cannot distinguish it from our definition has proved nothing — §7, 2026-08-07: *a check must report
its discriminating power, not only its pass rate.*

**Run the gate on `cf_hz1x1` AND `cf_grid4x4`.** Both have 106 materialised draws
(`scenarios/draws/cityflow1x1`, `.../cityflow_grid4x4`), verified present by the coordinator 2026-08-29.
⛔ **`cf_cologne3` is OUT OF SCOPE for this brief** — only 11 draws are materialised and it is absent from
`p8_4a_admission.json` entirely. Under `Rule R` an ungated cologne3 simply carries no single-definition
claim, which is determinate and costs nothing here.

---

## 6. Scope fence — what NOT to build

⛔ **Do NOT run the re-derivation campaign.** That is BRIEF_33, written after this gate returns. **If Gate 0
fails, the re-derivation's design changes**, which is the entire reason these are two briefs.
⛔ **Do NOT modify `offline/admission_probe.py`.** Import it, call it, compare against it.
⛔ **Do NOT edit `envs/cityflow_env.py` or anything else frozen.** Subclass.
⛔ **Do NOT correct `offline/policies/plan_replay.py:15`'s docstring** (T1's M3). It is deliberately queued
with the metric change in BRIEF_33 — `BRIEF_31` Amendment A5 and §6's P8.4b entry both say so.
⛔ **Do NOT decide which definition is primary.** `Rule R` decides that from your numbers. **You report
measurements and the four criteria's outcomes; you do not report a verdict on the metric.**
⛔ **Do NOT tick `P8.4b`.** It covers the whole task. **Add and tick a new `P8.4b-G0` box in §6** in the
merge commit — and check the id is unique (`DEFERRED` 26's fourth condition: §6 once carried two `P8.3`
boxes that disagreed, and an existence-check passed it).

---

## 7. Files

**`offline/engine_att_reference.py`** (new, the only new source file)
- The `CityFlowEnv` subclass with the overridden `_simulate` and its per-second observer.
- Both reconstructions: the engine-population one (§4) and the entered-only variant.
- The four-criterion gate evaluation, emitting **per-episode** rows and a summary.
- A CLI consistent with `admission_probe`'s (`--repo-root`, `--corpus-root`, `--draws-root`,
  `--output-root`, `--work-dir`, `--engine-seed`, `--torch-threads`, `--scenario`, `--tiers`, `--draws`).
- Writes `docs/data/p8_4b_g0_reference.json` **atomically**, and **after all validation**
  (filesystem-mutation barrier, §7 — this bug has shipped twice in this repo).
- Records `runtime_provenance()` — **it lives in `offline/dt_gate.py`; import it, do not reimplement it** —
  with the measurement/written-at commit split (`DEFERRED` 39, discharged by P4.3 and proven in use by
  P4.6) and the thread regime including `OMP_NUM_THREADS` / `MKL_NUM_THREADS`, **read at run time, never
  assumed** (§7, 2026-08-12: `torch.set_num_threads()` is a *different knob* from those env vars, and a
  recorded `torch_num_threads = 1` does not establish which regime produced a timing).

**`tests/test_engine_att_reference.py`** (new)

---

## 8. Test list — the load-bearing one is named

**LOAD-BEARING: `test_first_seen_is_the_engines_enter_time`.** The §4 premise, tested directly rather
than assumed: on a real episode, a vehicle's `first_seen` must coincide with the step at which the engine's
own accounting begins for it. **If you cannot construct this test, the gate cannot be trusted and the
correct outcome is `BLOCKED`.**

**Required mutations — each must be executed and each must FAIL the named criterion. Paste every failure.**

| # | Mutation | Must be killed by |
|---|---|---|
| M1 | shift `first_seen` by +1 second | criterion (1) |
| M2 | snapshot `include_waiting=False` in the engine-population reconstruction | criterion (1) — it becomes our definition |
| M3 | drop never-entered vehicles from the denominator | criterion (2) |
| M4 | use `last_seen` instead of `last_seen + interval` as the finish time | criterion (1) |
| M5 | make the entered-only variant return the engine reconstruction | criterion (3) |
| M6 | snapshot on the env-step (10 s) grid instead of per second | criterion (1) |

**Also required:**
- `test_lane_change_is_false_in_every_cityflow_config` — enumerate all 13 configs, assert `laneChange` is
  false or absent, **and assert the enumeration is non-empty** (§7: a check must assert its input was
  non-empty, or *found nothing* masquerades as *found nothing wrong*).
- A **positive control** proving the gate can FAIL: feed it a deliberately wrong reference and show it
  exits non-zero. §7, 2026-08-18: *a check that reports by printing is not a check.*
- `pytest.raises(match=...)` tokens must be verified to occur in **exactly one** raise-path of the module
  under test, and say that you verified it (`DEFERRED` 43).

---

## 9. Separate, small, and explicitly fenced: the CI skip ceiling

`main` is red and it is the **registered** red: run `33275392200` reports
**`passed=1364 skipped=123 failed=0 errors=0`** against a declared ceiling of **121**. Zero failures; the
whole red is +2 skips. The registered protocol (`re_measure_required_at.what_to_do`) is *merge, let it go
red, classify `junit.xml`, commit the observed value* — **do not pre-bump.**

**Required:** classify the +2 from the CI `junit.xml`, and ship the ceiling move as a patch under
`docs/patches/` with an entry in `docs/patches/README.md`, verified with `git apply --check`.
`.github/ci/**` and `tests/**` are outside the coordinator's hands by role (`DEFERRED` 54), and the
permission layer denies `Edit(scripts/**)` — **the patch route is the mechanism, not a workaround.**
⚠️ **Separate commit. It must not touch the gate's code.** Left alone this is `DEFERRED` 45's class: a
guard red for a reason everyone expects is a guard that stops being read.

---

## 10. Definition of Done

- [ ] Code complete, no placeholders, no `TODO: implement later`
- [ ] Tests written **first**, run, and confirmed to fail for the right reason before implementation
- [ ] **All six mutations executed, every failure pasted**
- [ ] The load-bearing test exists, or the packet is `BLOCKED` and says why
- [ ] Gate run on `cf_hz1x1` **and** `cf_grid4x4`; all four criteria reported **with observed values**,
      including the observed maximum deviation
- [ ] `git diff --stat` proves zero modifications to frozen files
- [ ] Zero new dependencies
- [ ] Suite run **stating its corpus environment, its skip count and its thread pin** (§7, 2026-08-19 —
      52 corpus-backed tests self-skip unless `RLTRAFFIC_CORPUS_V11` and `RLTRAFFIC_CORPUS` point at the
      corpus, and their absence is invisible in a green summary line)
- [ ] Committed on `task/p8.4b-g0-engine-att-reference`, staging **named paths** — `git add -A` and
      `git add .` are forbidden (§7, three instances in four days)
- [ ] `docs/returns/P8.4b-G0.md` written from `docs/returns/TEMPLATE.md`
- [ ] **`P8.4b-G0` added and ticked in `PROJECT_PLAN` §6 in the merge commit**, id verified unique
- [ ] **AI-assistance record** present, four lines, written as the task happened (CLAUDE.md §8)
- [ ] The packet **states which amendments it was written against, by letter** (§7, 2026-08-29)

---

## 11. Return Packet template

```markdown
## RETURN PACKET — P8.4b-G0
**Status:** DONE / PARTIAL / BLOCKED
**Written against:** BRIEF_32 amendments A–?   (by letter; a packet that names fewer than the brief
                                                carries is visibly stale to its first reader)
**Branch + diff stat:** (branch; `git diff --stat` vs main)
**Files produced:** (paths + one line each)

### Gate result — per scenario, and NO verdict on the metric
| scenario | crit 1 max deviation | crit 2 | crit 3 max deviation | crit 4 tiers x draws | PASS/FAIL |
|---|---|---|---|---|---|

### The load-bearing premise
Did `first_seen` test out as the engine's enter time? How was it tested, on what sample?

### Mutations
M1..M6: each with the criterion it broke and the pasted failure.

### Suite
Corpus env set? Skip count? Thread pin? Paste the real tail.

### Open questions / conflicts with this brief
(When the brief conflicts with the repo, THE REPO WINS — implement to the repo and flag it here.)

### AI-assistance record
1. Tool and version:
2. Authorship per file:
3. Human verification performed (which tests, which mutations, recomputed by whom):
4. Who made the research decisions (author / coordinator ruling by brief section / pre-registration
   commit):
```

---

## 12. The 95% rule

Before acting, list your assumptions and your confidence in each. **Any load-bearing assumption below
~95% is a question, not an assumption.** The premise in §4 is the obvious candidate. A question costs
thirty seconds; a wrong assumption frozen into this gate's output would propagate into the choice of the
paper's primary metric — which is what A11 exists to make impossible.

---

# AMENDMENT A — 2026-08-30, ruling on the plan-mode questions Q1–Q6

**Read this with the brief; a packet must state which amendments it was written against, by letter.**

## A0 — Q1, criterion 3: YOU ARE RIGHT, THE CRITERION IS DEFECTIVE, AND THE ERROR IS MINE

**Independently reproduced by the coordinator before ruling**, on `docs/data/p8_4a_admission.json`:
`att_difference == att_ours - att_engine` in **1,870 of 1,870** rows under `!=`; and on the **1,342**
episodes with `entered_fraction == 1.0` — identical populations, population component exactly zero —
`att_difference` has median **4.1922** (hz1x1, n=709) and **7.0668** (grid4x4, n=633), minimum **1.6542**.
**A pure-population reading predicts zero. Criterion 3 asked a per-second instrument to reproduce a
10 s-grid artifact to 1e-4 s.** Your three lines of evidence all hold and your reading of M1 is correct.

> ⏸️ **Q1 IS THEREFORE PENDING `PREREGISTRATION.md` A12**, drafted at `docs/notes/A12_DRAFT.md` and with
> the author now. **Criterion 3 is replaced by (3a) an EXACT bit-identity negative control where
> `never_entered == 0`, (3b) a positive control requiring a difference where `never_entered > 0`, and
> (3c) your option (c) — the metric-cadence reconstruction — as REQUIRED AND REPORTED BUT NOT GATING.**
> Your recommendation of (c) was right and it is adopted; it is not gating only so that a surprise about
> `metrics/cityflow.py` cannot silently block a question about `att_engine`.
> ⛔ **Do not implement criterion 3 in any form until A12 lands.** **Everything else in §A1–A6 below is
> ruled now and is not blocked** — build the reconstruction, the load-bearing test, criteria 1, 2 and 4,
> and run the pilot. That is the bulk of the task.

**Your pilot commitment is approved and is now required**: two episodes reporting all observed values
before the full campaign, so this stays a measurement rather than an argument.

## A1 — Q2, tier coverage: draw tiers from the CORPUS, and take `created` from the probe's READER

A11's *"≥7 behaviour tiers"* means the **corpus's** seven (`fixedtime`, `mappo060`, `mappo200`,
`mappo500`, `mappo1000`, `maxpressure`, `random`), which `datasets_v11/` carries per scenario. It never
meant `admission_probe`'s declared arm set, and I did not check that set before writing it.
⭐ **The resolution that removes the constraint entirely: criterion 2's reference is
`admission_probe`'s reader INVOKED LIVE — `read_admission_at_horizon` — not the committed artifact.**
`created` is a function of the episode, not a lookup, so tier coverage is not bounded by what P8.4a
happened to measure. *"Independently reports"* means the independent code path, not the old file.

## A2 — Q3, the extreme `entered_fraction` episodes: INCLUDE THEM, learned arms and all

The extremes exist to stress the instrument where censoring is largest. **Whether the policy is a
behaviour tier or a learned arm is irrelevant to that purpose.** So: the seven behaviour tiers × ≥3 draws
**plus** the min and max `entered_fraction` episodes wherever they come from — `bc_top10@random` seed 303
draw 1009 (0.619048) on hz1x1 and `bc_top10@mappo1000` seed 404 draw 1001 (0.944155) on grid4x4. That is
a superset of A11 criterion 4 and satisfies its letter. **Your registered tie-break for the max is
approved** — ties at 1.0 need a rule and you wrote one before looking at the outcome.

## A3 — Q4, the import fence: YOUR SPLIT IS CORRECT AND IS ADOPTED

The **RECONSTRUCTION** imports neither `metrics/` nor `offline/admission_probe` — that independence is
the entire evidential value of the gate. The **HARNESS** may import both, because criteria 2 and 3 are
*defined* as comparisons against the probe. **AST-enforced with a positive control, as you propose** —
and the positive control is what makes the fence a check rather than a comment. BRIEF_32 §1's *"you may
import it"* is hereby scoped to the harness.

## A4 — Q5, `--tier` vs `--tiers`: ship `--tiers`

It is a new module and its CLI is its own. Note the divergence from `admission_probe` in the packet so
the next reader is not surprised by it.

## A5 — Q6, the CI ceiling: use the CURRENT run, and name this task's merge as the next expiry

⚠️ **My run id `33275392200` was live when I wrote it and is not now — that is §10's liveness half-life
biting inside a brief, and you were right to check.** **Read the numbers off the newest run on `main` at
the time you build the patch, and name the run id you actually used.** Registered protocol stands: commit
the **observed** value, do not pre-bump — and **you are right that this task's own merge will move it
again**, so name `P8.4b-G0`'s merge as the next expiry in `re_measure_required_at`. A ceiling that is
stale by construction the moment it lands is fine **provided the expiry says so**; that is what the field
is for.

## A6 — the `contribution(v)` collapse and the `Archive` finding

**The collapse to `contribution(v) = last_seen - first_seen + interval` for every vehicle is approved.**
Implement the single expression and let criterion 1 arbitrate the ±1-step question — that is the right
instinct: a disagreement of exactly one `interval` will show up as ~1.0 s against a 1e-4 s bar and is
unmissable. **My §4 wording said `T - first_seen` for still-present vehicles and did not pin what `T`
meant; yours is better specified.**

⭐ **`Archive::dumpVehicle` (`archive.cpp:178-187`) serialising `id` and `enterTime`, with `Archive::dump`
also writing `step`, `finishedVehicleCnt` and `cumulativeTravelTime`, is a genuinely better instrument
than the brief asked for, and it is adopted as REQUIRED.** It converts the load-bearing test from an
inference about set membership into a **direct comparison against the engine's own serialised state**,
and it supplies a **fully independent second route** to `get_average_travel_time()` — which is CLAUDE.md
§2's double-computation rule satisfied by a genuinely different route rather than the same derivation
retyped (§7, 2026-08-13). **Use it for both.**

## A7 — one thing to carry into the packet

**Report the cadence term per scenario.** T1's M1 estimated +4.99 / +4.89 / +4.72 s **on hz1x1 only**,
and the committed artifact puts grid4x4's median at **7.0668**, outside that range. **Nobody has
explained the gap and you should not claim to** — record it as an open question with its measurement.
It is the kind of scenario-dependence §7's 2026-08-18 rule says to expect the moment a network with a
property hz1x1 lacked is first measured.

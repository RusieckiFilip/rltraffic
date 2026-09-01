# T1 — GROUND-TRUTH VALIDATION OF THE PRIMARY METRIC

**Date:** 2026-08-28 · **Reviewer:** `contract-reviewer`, fresh context, read-only
**VERDICT: FAIL — 4 blockers, 4 major.**

> **The metric's arithmetic matches its docstring. The docstring describes a quantity that is not what
> this project's own records claim it is, is not what the field reports, and diverges
> policy-dependently by up to 2×. On matched flow draws it INVERTS THE RANKING of two of the corpus's
> own behaviour policies.**

🚨 **This is the finding the author predicted when he said every check we run asks *"does this
reproduce"* and never *"is this right"*. 92 tests pass while the metric exhibits B1, B2 and M1. Not one
of them recomputes ATT by an independent route.**

## Method, and its fidelity was checked before anything was concluded

The reviewer subclassed `CityFlowEnv` in `/tmp` and snapshotted `get_vehicles(include_waiting=…)` and
`get_average_travel_time()` **every simulation second**. Ground truth is computed from raw engine
vehicle-id sets, importing nothing from `metrics/`.
- **Probe fidelity:** hz1x1/MaxPressure returns `att_running_mean 160.5584` and `att_horizon 247.7509`
  — matching `PREREGISTRATION` A2's canonical `160.5584 → 247.75` **exactly**. The probe does not
  perturb the episode.
- **Ground-truth machinery:** the reviewer's reconstruction of CityFlow's own C++ metric from raw
  vehicle sets reproduces `eng.get_average_travel_time()` to **0.000e+00** on two independent
  3600 s / 2021-vehicle episodes.

---

## BLOCKERS

### B1 — the metric silently excludes every vehicle created but never admitted, and the exclusion is policy-dependent

✅ **Mechanism confirmed by the coordinator in BOTH implementations, from source:**
- **Ours:** `metrics/cityflow.py:60` and `:159` both call `get_vehicles(include_waiting=False)`. A
  vehicle queued in the lane insertion buffer is not in `depart_time`, not in `completed`, not in the
  average.
- **CityFlow's own:** `engine.cpp:682-691` is `for (auto &vehicle_pair : vehiclePool)` with **no
  filter at all** — while `getRunningVehicles` at `:780-786` explicitly filters on
  `isReal() && (includeWaiting || isRunning())`. **The engine's metric averages over every vehicle
  ever CREATED.**

hz1x1, seed 10101, 3600 s, same episode:

| policy | ours `att_per_step[-1]` | CityFlow's own | never entered |
|---|---|---|---|
| MaxPressure | 247.75 | 264.72 | 58 / 2021 (2.9 %) |
| Random | 427.04 | **877.95** | **774 / 2021 (38.3 %)** |

**MaxPressure's advantage over Random reads 41.98 % under ours and 69.85 % under the field-standard
one.** ⚠️ **The gap is neither constant nor monotone: in a saturated-source falsification (one movement
held red, 300 vehicles) ours reads 542.44 and the engine 301.00 — THE SIGN FLIPS.**
**A policy that prevents vehicles from entering is rewarded by our metric.**

### B2 — the metric inverts the ranking of two corpus behaviour policies

`datasets_v11/cf_hz1x1__*`, seed 1000, draws 1–8, matched by construction (identical `created = 1814.0`
in every cell). Every replay reproduced the stored `att_per_step` **bit-identically** after the float32
cast.

| tier | ours | field-standard | ratio | entered/created |
|---|---|---|---|---|
| mappo1000 | 103.06 | 98.82 | 0.959 | 100.0 % |
| **fixedtime** | **261.89** | **390.07** | 1.489 | 90.8 % |
| **mappo060** | **341.25** | **370.45** | 1.086 | 92.9 % |
| random | 428.33 | 809.34 | 1.889 | 65.0 % |

**Ours: `mappo1000 < fixedtime < mappo060 < random`. Field-standard: `mappo1000 < mappo060 <
fixedtime < random`. `fixedtime` and `mappo060` SWAP, in 7 of 8 matched draws.**
⚠️ **`offline/dt_gate.py:466-468` makes the project's go/no-go decision by comparing exactly this
quantity between policies, so the P4 gate's verdict is definition-dependent.**

### B3 — the pre-registration's source-verification claim is false

`PREREGISTRATION` §3.1 states the metric was *"verified in source … in all three checked"* to average
over **all vehicles that entered the network**. **The C++ version averages over all vehicles ever
CREATED.** Contradicted by `engine.cpp:685` and by the reviewer's exact numerical reconstruction.
**This is the load-bearing claim A4 later cites to remove §3.1's own safeguard.**

### B4 🚨 — the safeguard registered against exactly this failure mode was removed on two factually wrong grounds

§3.1 originally registered, in its own words: *"Because a metric over 'vehicles that entered' can still
be gamed by a policy that prevents vehicles from entering, every reported ATT cell is accompanied by …
the number of vehicles that entered"*, plus a **>5 % entered-count invalidation**.

- **A4's ground:** *"the survivorship concern … is already met by the metric's own definition:
  `average_travel_time` averages over all vehicles that entered."* **Averaging over vehicles that
  entered is not a defence against a policy that prevents entering — it is a restatement of the
  vulnerability.**
- **A5's ground:** *"`entered` is a population size, **stable across policies on fixed demand**
  (measured spread **4.1 %** across Tier 1 cells: 1595 / 1623 / 1661)."* ✅ **Quoted verbatim,
  confirmed by the coordinator.** **Measured across the corpus's own four behaviour tiers on identical
  demand: 35.0 %** (1813.9 / 1685.0 / 1648.0 / 1178.8).
  🚨 **A5 measured the spread WITHIN ONE POLICY FAMILY and stated it OF ALL POLICIES. That is this
  project's signature error, committed inside the PRE-REGISTRATION, and it withdrew the one safeguard
  that would have caught B1.**

**Applying the withdrawn §3.1 condition to those four tiers: 5 of 6 pairwise comparisons are INVALID
(9.1 %, 7.1 %, 35.0 %, 28.5 %, 30.0 %). The single pair that passes — fixedtime vs mappo060 at 2.2 % —
is precisely the pair whose ranking B2 shows is inverted.**

## MAJOR

- **M1 — asymmetric time quantisation.** Entry uses the window midpoint (`t − delta_time/2`,
  unbiased); exit uses the window end. Completed vehicles carry a systematic **+5.4974 / +5.3789 s**
  mean error against 1-second ground truth. Net effect on the horizon value **+4.99 / +4.89 / +4.72 s**
  — ~2 % of MaxPressure's ATT and ~1.2 % of Random's, **so the relative magnitude is itself
  policy-dependent.**
- **M2 — the backends disagree on the timing definition.** SUMO uses exact departure and arrival times
  and carries **no** window skew; MOSS takes `real_tt` from the engine. **Only CityFlow estimates both
  endpoints from the decision grid. Any cross-backend claim (C3) inherits a ~5 s CityFlow-only offset.**
- **M3 — the repo records the OPPOSITE of what is true, in three places.**
  `offline/policies/plan_replay.py:15`, `docs/plans/p2.6.md:42` and `docs/PROJECT_PLAN.md:1637` all
  call the engine's native metric *"survivorship-biased"* and ours *"survivorship-free"*. **On the
  dominant axis this is backwards.** The 2026-08-05 P2.5 entry celebrates a gate that "fired on a real
  defect" (662.36 vs 1051.25) — **that gap IS this definitional difference, and the recorded diagnosis
  of which side was wrong is inverted.**
- **M4 — A1's comparability justification does not survive.** A1 chose the horizon value partly because
  it is *"comparable to DataLight / DTLight / RESCO figures"*. Those are CityFlow
  `get_average_travel_time()` numbers. **Ours differs from that by −6.4 % to −51.4 % on the same
  episodes.** ⚠️ *The reviewer flags this rests on an inference that those papers use the engine API;
  they did not read their code.*

## MINOR

`att_per_step.mean()` is **not** `att_running_mean` (T+1 rows including a reset row; −0.277 % over 414
episodes) · `envs/cityflow_env.py:269-273`'s fallback silently substitutes a different quantity under
the same key name (dead today, live trap) · `getAverageTravelTime` lacks the `isReal()` filter, so
lane-change shadow vehicles double-count (inert: `laneChange: false` everywhere) · only draws 0–5 and
1000–1099 are materialised while the corpus references 1–200.

---

## ⭐ DEFERRED 22 — CLOSED, and it is the SMALL problem

414 episodes across all 69 corpus dirs: `att_per_step` is `(361,)` float32, row 0 = 0.0 in all 414,
**zero NaNs anywhere**. **`att_per_step[-1]` IS the value at the horizon.**

| reader takes | distinguishable? | magnitude |
|---|---|---|
| last non-NaN | **no** — no NaNs exist | identical |
| `[-2]` | only by exact comparison | mean −0.201 %, max 0.804 % |
| `max(...)` | **yes** — argmax is the last row in only **242/414 = 58.5 %** | mean +1.32 %, max +18.97 % |
| `mean(...)` | grossly (A1's retired quantity) | −27 % to −33 % |

**One to two orders of magnitude smaller than B1/B2, and it must not consume the attention those need.**

## ⭐ RECOVERABILITY — the brief's premise was wrong, and this is the good news

The brief said a definition error *"would require re-running every simulation"*. **Not recoverable by
arithmetic: confirmed.** But **it IS recoverable by deterministic replay, with no re-collection and no
retraining:**
- manifests carry `flow_randomizer_params` and per-draw `flow_draw_sha256`; the reviewer regenerated
  draws **1, 6, 42, 200** and **all four sha256 matched exactly**;
- every episode stores `engine_seed`, `flow_draw` and per-intersection actions;
- **36 shipped episodes replayed across 4 behaviour policies — every one reproduced the stored
  `att_per_step` bit-identically.**

**Any ATT definition can be re-derived at simulation cost only, and verified against the stored arrays.**

## What the reviewer could NOT verify

**The full test suite — killed at a 3000 s timeout after 44+ minutes with no output.** Only the 8-file
subset (`92 passed in 5.09 s`) has a real result. *(The suite taking >44 min is itself an observation
the reviewer could not resolve; the coordinator measured 468 s two days ago.)* · MOSS empirically (not
installed) · **scenarios other than cf_hz1x1 and cologne1 — grid4x4 and cologne3 are UNMEASURED and the
censoring there could be larger** · one seed, 8 draws, 1 episode per cell · A5's own 4.1 % figure was
not reproduced, only the general claim it supports refuted · **whether DataLight/DTLight/RESCO actually
report the engine metric — inferred, not read** · **whether any published number in `docs/data/`
depends on the inverted ordering** · whether any gate, calibration or tier-selection step routes ATT
back into training.

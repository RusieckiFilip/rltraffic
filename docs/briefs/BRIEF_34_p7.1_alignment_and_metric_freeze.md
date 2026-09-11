# BRIEF_34 — P7.1: feature-space alignment and the transfer-metric freeze

**Task id:** `P7.1` · **Branch:** `task/p7.1-alignment`, cut from the `main` commit that carries this brief (`git rev-parse main` after `git merge main`; a header cannot know its own sha — `BRIEF_33`'s did not, and the implementer had to flag it) · **Issued:** 2026-09-11
**Mode:** Claude Code, plan mode first, worktree `/home/filip/rltraffic-p53b` (checkout the new branch there)
**Registered as:** `PROJECT_PLAN` §6 P7.1 — *"freeze the backend-neutral feature set (no backend-specific
structured states) + freeze the normalized transfer metric"*. The metric is `PREREGISTRATION` §3.4's
`ρ = (ATT_fixedtime − ATT_policy) / (ATT_fixedtime − ATT_maxpressure)`, within backend, two anchors —
**§6's "over MaxPressure" wording is the older, looser one; the registration governs.**
**Compute:** no training. Two small observer campaigns on `hangzhou_1x1_bc-tyc` — **15 SUMO episodes**
(3 anchors × 5) and **15 CityFlow episodes** (same cells) — plus one hz4x4 SUMO timing episode. All
under a fence in the MAIN tree's `output/p7_1/`, pre-flight-reviewed (§7 trigger (b) fires: writes under
`output/` while reading `output/p7_0/`).
**Contracts:** `docs/CONTRACTS.md` v1.1 — C1, C2, C6 bind. `envs/`, `states/`, `metrics/`, `utils/`
are FROZEN; every alignment lives in `offline/` as a translation layer over `info`, never in the env.
**Cost, and the delta from the handoff:** `HANDOFF_2026-09-01` §4 estimated P7.1 at 1–2 days as
"design + a freeze document". **This brief is 3–4 days**, because a freeze that rests on unmeasured
premises is the class of error this project keeps logging: the two instruments below are what make the
freeze checkable. The reasoning is in §0.

> **Read order:** this brief → `docs/returns/P7.0.md` (§§ 3.3–3.7, 13, 15) → `docs/plans/p7.0.md` §§ 2, 11
> → `PREREGISTRATION.md` §3.4, §4 (C3), A11, A13 → `docs/notes/P7.0_vtype_investigation.md` →
> `offline/transfer_gate.py` and `offline/parity.py` module docstrings → `offline/engine_att_reference.py`
> module docstring (the instrument you will twin) → `BRIEF_33` §3 (the shape of an observer task and its
> fence). Where this brief disagrees with `BRIEF_21`/`BRIEF_04`, this wins; where it disagrees with the
> repo, **the repo wins and you flag it** (`CLAUDE.md` §2).

---

## 0. What the coordinator verified before writing this, and what it changed

Every fact here was produced on 2026-09-11 by a command against `main` @ `b8accc9` (the tree before this brief), or by a read-only
survey whose citations were spot-checked. Re-run them; do not inherit them.

**0.1 The handoff's five entry conditions, re-checked (`HANDOFF_2026-09-10.md` §5):**

| # | condition | status in the artifact |
|---|---|---|
| 1 | lane-index inversion fix before P7.3 | **Open — P7.1 delivers it** (§3, half B). `transfer_gate.lane_semantic_correspondence` exists and is proven on hz1x1, but nothing applies it to a live `info` or a stored corpus. |
| 2 | P7.0's numbers may not enter the paper without independent review | **Open — P7.1's review discharges it** (§5): the adapter's load-bearing test recomputes P7.0's registered per-feature table from `output/p7_0/` raw rows, and half A re-measures P7.0's ρ cells. |
| 3 | hz4x4 gudang "the only scenario with SUMO parity and coordination structure"; SUMO timing unmeasured | **Half wrong, half right.** Gudang has a complete SUMO scenario (`.net.xml`, `.rou.xml` exist, 16 TLs, 240 lanes) but **no parity vType (0/2983 bound, no `tau`) and no CityFlow corpus**; the registration's paired set is `4× hangzhou_1x1, cologne1, cologne3` and does not list gudang. **SUMO timing is now measured once: 51.0 s per 3,600 s episode on gudang** (traci, random policy, 360 steps, `n = 1`, coordinator's survey 2026-09-11) against CityFlow's 2.67 s on grid4x4 — **~19×**. hz1x1 SUMO timing is still unmeasured; half A measures it. |
| 4 | bind hangzhou's vType FIRST, and `tau` with it | **DONE for hz1x1 by P7.0:** `scenarios/hangzhou_1x1_bc-tyc_18041610_1h_parity/` binds `cf_parity` (`tau=2.0`, `speedFactor=1.0`, `maxSpeed=11.11`) to 2021/2021 vehicles with a positive control. **NOT done for the other three hz1x1 variants, gudang, cologne1, cologne3.** The handoff's "55.55 m/s" framing is the superseded 2026-07-31 one; the measured confound is `tau` (+49 %), not speed (`Decisions Log` 2026-08-04). |
| 5 | CityFlow has two ATT definitions, SUMO one — ρ asymmetric | **Sharpened, and answerable by measurement, not argument** (§0.2). |

**0.2 What SUMO's ATT actually is** (`metrics/sumo.py:459-477`, `:235-242`; `envs/sumo_env.py:119-125`,
`:316-320`): population = vehicles that have appeared in `traci.vehicle.getIDList()` (**never-inserted
vehicles are excluded**); clock origin = **actual** departure (`getDeparture`), not the route file's
intended `depart`; arrivals exact per simulation second. So SUMO's single definition is the twin of
CityFlow's `W_running` (admitted population, admission clock, per-second) — **not** of `att_ours` (which
adds a window-midpoint estimate, `metrics/cityflow.py:180-184`) and **not** of `att_engine` (pool clock,
all created). **There is no pending-vehicle accounting anywhere in the env** (grep of `envs/ metrics/
offline/ utils/` for `getPendingVehicles|getDepartedIDList|getLoadedIDList|max-depart-delay|
time-to-teleport`: zero code hits). SUMO itself has the insertion buffer: `--max-depart-delay` defaults to
`−1` (a vehicle waits forever), `traci.simulation.getPendingVehicles`, `getDepartedIDList`,
`getStartingTeleportIDList` and `traci.vehicle.getDepartDelay` all exist in the installed 1.27.1
(checked). ⭐ **Therefore `att_engine`'s pool-clock, all-created twin IS definable on SUMO** — intended
depart from the route file, contribution `(arrival or T) − intended_depart`, denominator every vehicle
with `intended_depart ≤ T` — and A13's decomposition carries over with a zero cadence term. **That is
how threat 5 closes: ρ on both backends under the same definition, with the other definition beside it.**
Also: `--time-to-teleport` defaults to **300 s** and neither the shipped nor the parity `.sumocfg` sets
it, so P7.0's 15 SUMO episodes ran with teleports enabled; CityFlow never teleports. Half A counts them.

**0.3 The lane correspondence, run structurally on all three headline pairs (no simulator):**
hz1x1 — 16 = 16 lanes, unique, every lane maps to the *other* index (`_0 ↔ _1`). **hz4x4 gudang — 240 =
240, intersection ids equal, 16/16 intersections resolve uniquely**, CityFlow `_0=l,_1=s,_2=r` against
SUMO `_0=r,_1=s,_2=l`, 128 of 192 incoming lanes map to a different id. 🚨 **cologne3 —
`lane_semantic_correspondence` RAISES on all three intersections** (`'-130160207#0_0' serving ['l','r']
matches 0 SUMO lanes`): SUMO connections carry `t` (102), `L` (3), `R` (2) which `_CITYFLOW_TO_SUMO_TURN`
has no counterpart for, movement assignments are reversed per lane, intersection ids differ
(`cluster_…` vs `GS_cluster_…`), and **SUMO monitors only 38 of the 76 lanes** (incoming ∪ outgoing of
controlled TLs, `metrics/sumo.py:62-66`) while CityFlow logs all 76. **As the files stand, C3's
zero-shot is feasible on hz1x1 only.** grid4x4 has no SUMO network (`.sumocfg` inputs missing).

**0.4 `phase_onehot` is not backend-neutral, but the map is exact.** Width 9 on CityFlow (file phase 0 =
clearance, 1–8 = greens) against 16 on SUMO (greens at 0, 2, …, 14; yellows/all-red at odd indices);
the hot index for action 0 is file phase 1 vs 0 (`docs/data/p7_0_gate.json:green_action_semantics`).
The corpus observes only greens at decision boundaries — **measured on the population, not a sample:
577,600 `current_phase` rows across all 1,600 hz1x1 episodes in `datasets_v11/` carry phases 1–8 and
zero rows carry 0; P7.0's 5,415 CityFlow rows likewise; P7.0's 5,415 SUMO rows carry even phases
only, zero odd** (transitions finish inside the 10 s step, including at row 0). So the clearance
slot is dead in the data the models trained on. **The frozen convention is the CityFlow
corpus's (width 9); the adapter maps SUMO even phase `2k → k+1`, odd → 0.** P7.0 §3.7 established
that action `a` releases the same physical lanes on both backends (8/8 under translation, 4/8 under the
void identity) — the adapter **tests** that with `transfer_gate.green_action_lane_sets`, it does not
assume it.

**0.5 Metric-set hazard (C8).** `SumoEnv` unions `average_travel_time` into `_metric_names`
unconditionally; CityFlow does not. A CityFlow-trained MAPPO checkpoint therefore raises on SUMO
(`global_feature_dim` 3 vs 4). The DT reads per-intersection `state` only, so P7.1 is unaffected; the
adapter carries an explicit `metric_keys` policy and the freeze document records the hazard for P7.3.

**0.6 Provenance question the paper's argument depends on.** `PROJECT_PLAN` §1:21 argues *"our scenarios
are NATIVELY AUTHORED in both backends, so the converter confound is ABSENT BY CONSTRUCTION"*; the
Decisions Log row of 2026-08-04 (`:1872`) says *"hangzhou is CityFlow-native with SUMO derived from
it"*. **Both cannot be right.** Half B establishes it from the files (§3.4).

---

## 1. Why this task exists

C3 is *"the residual dynamics gap after matching every parameter both engines expose"*, measured as a
transfer curve on ρ. Two things must be frozen before any DT sees a SUMO state, and both were found
unfrozen in the artifact:

1. **What ρ is computed on.** Rule R made `att_engine` primary on hz1x1; SUMO's only ATT is a different
   estimator. A ρ built from `att_engine` on one side and SUMO's admitted-population metric on the other
   is a ratio of two quantities that disagree by ~200 ATT on a collapsed policy (`P5.3b-fix`), and the
   asymmetry is invisible in the ratio. **Half A builds SUMO's pool-clock twin, measures both
   definitions on the same anchors on both backends, and the freeze names the definition ρ uses.**
2. **What the model reads.** 8 of 8 lane ids mis-denote on hangzhou; sorting fixes order, not
   denotation. **Half B turns P7.0's proven correspondence into the adapter every later task uses, with
   the proof-from-structure rule (§7) as a test, and recomputes P7.0's registered table through it.**

**Do half A first.** It answers the question the handoff says must be answered *before P7 is scheduled*,
and P7.2 (the named-contribution candidate) depends on it and not on half B.

---

## 2. Half A — the metric freeze: `offline/sumo_att_reference.py` (new)

### 2.1 What it is

A per-second SUMO observer twinned on `offline/engine_att_reference.py` — read that module's docstring
first; the alignment-convention section is the one thing to understand before trusting a number. Build
it by subclassing `SumoEnv` in `offline/` (as `PerSecondEngineObserver` subclasses `CityFlowEnv`), or by
wrapping its traci connection — **never by editing `envs/`**. Per simulated second, record: departed ids
(`getDepartedIDList`), arrived ids (`getArrivedIDList`), pending ids (`getPendingVehicles`), starting
teleports (`getStartingTeleportIDList`), and on the monitored incoming lanes both
`getLastStepHaltingNumber` and the count of `getSpeed(v) < 0.1` over `getLastStepVehicleIDs` — the
halting-threshold check P7.0 left as "documented, not measured".

**Intended departure comes from the route file** (`<vehicle depart=…>`), parsed once per episode with
stdlib XML — the SUMO twin of `created_from_flow`. Every vehicle id in the file with `depart ≤ T` is in
the pool population; assert that the union of departed ∪ pending ∪ not-yet-due equals the file's id set
at every second (the twin of criterion 2, "denominator equals `created`").

**Four reconstructions, named to match the CityFlow ones so the decomposition reads across:**

| name | population | clock | on SUMO |
|---|---|---|---|
| `E_sumo` (engine-population twin) | every vehicle with `intended_depart ≤ T` | intended depart | new |
| `P_sumo` (entered-population, pool clock) | departed by T | intended depart | new |
| `W_sumo` (entered-running twin) | departed by T | **actual** depart | must equal the env's `average_travel_time` **exactly** at the horizon — this is your `c3c` |
| cadence | — | — | **zero by construction** on SUMO (exact departures and per-second arrivals); assert `W_sumo == att_sumo_env` under `==` on every episode |

`E − P` is the never-inserted population term; `P − W` is minus the mean insertion delay
(`getDepartDelay` must reproduce it — a second route; assert to 1e-9); report both per episode with the
identity `att_sumo_env − E_sumo = population + clock_origin` exact.

### 2.2 The measurements (the freeze rests on these, so they are not optional)

**A1 — SUMO anchors under the observer.** `hangzhou_1x1_bc-tyc` **parity** `.sumocfg`, P7.0's exact
`COLLECT_SETTINGS` (`--max-steps 360 --delta-time 10 --control-mode acyclic --global-reward-fn
queue_length --local-reward-fn queue_length --global-reward-weight 0.0 --state-features
lane_vehicle_count lane_waiting phase_onehot --fixed-time-k 4 --base-seed 1000 --episodes 5`, traci,
one thread), three arms `fixedtime`, `maxpressure`, `random` × 5 episodes = **15 episodes**. Reuse the
arm factories `transfer_gate` used (`dt_gate._maxpressure_factory`, `method_tier_grid._fixedtime_factory`
/ `_random_factory`) — import, do not re-implement. Per episode: the four quantities, the two terms,
`n_pending_at_horizon`, `n_teleports`, halting-threshold max abs difference, wall seconds with and
without the observer (the without-observer number is the **hz1x1 SUMO timing** P7.3 will schedule
from — state `n`).
🔒 **Reproduction check:** the 15 `att_sumo_env` values must equal P7.0's SUMO cells in
`output/p7_0/sumo__{fixedtime,maxpressure,random}/` (per-episode `att_per_step[-1]`, read from the npz,
not the artifact) under `==` — same seeds, same parity file, same settings. If SUMO is not bit-stable
across runs, **report the max abs difference and stop**; do not loosen to a tolerance silently. That
outcome is a finding about SUMO's determinism that P7.3 needs.

**A2 — CityFlow anchors under the existing observer.** The same three arms × 5 episodes on
`configs/sim/cityflow1x1.json` **nominal demand** (what P7.0 ran — not held-out draws), through
`engine_att_reference.gate_episode` with the same factories, giving `E, P, W_running, C` and the three
terms per episode. Reproduction check: `att_ours` equals P7.0's CityFlow cells under `==` (247.7509 for
MaxPressure, one distinct episode ×5).

**A3 — ρ under both definitions on both backends,** from A1 and A2: `ρ_random` (the only non-anchor
policy P7.0 had) under {`E`, `W`} on CityFlow and under {`E_sumo`, `W_sumo`} on SUMO, with `Δ` per
backend per definition. Report the ratio `Δ_sumo / Δ_cityflow` under each definition **as a diagnostic,
labelled as such** (P7.0 open question 4 — the coordinator rules on its paper use; you report it).

**A4 — one SUMO timing episode on hz4x4 gudang** through the observer, random policy, seed 1000, the
shipped (unbound) route file, **for timing only** — its ATT is not a result and the artifact says so.
Pair it with the survey's 51.0 s without the observer.

### 2.3 What the freeze SAYS (the draft you deliver; the coordinator registers it)

`docs/notes/P7.1_FREEZE.md`, §"Metric": (i) ρ is computed per backend under **one named definition**,
the pool-clock all-created pair (`att_engine`, `E_sumo`), with the entered-population pair
(`att_ours`, `att_sumo_env`) reported beside it in every table (A11(b)'s discipline extended to SUMO);
(ii) every report of a cross-backend ρ difference carries the two-term decomposition per backend;
(iii) the teleport policy — **your recommendation from the count**: if any anchor episode teleported,
the parity `.sumocfg` needs `<processing><time-to-teleport value="-1"/></processing>` (a NEW file, never
an edit of the shipped or the P7.0 parity one, which recorded runs used) and P7.0's SUMO cells must be
labelled *teleports enabled*; if zero teleported on all 15, say so with the count; (iv) SUMO's
determinism as measured in A1; (v) the hz1x1 SUMO rate with `n`.
**You do not decide whether `E` or `W` is primary on SUMO.** Rule R is a per-scenario gate on CityFlow
with no SUMO counterpart; the freeze proposes, with the numbers, and the amendment is the coordinator's.

---

## 3. Half B — the alignment: `offline/backend_alignment.py` (new)

### 3.1 The paired-scenario table — established from files, not from the registration

For every `scenarios/**/*.sumocfg` whose referenced inputs all exist AND that has a CityFlow config under
`configs/sim/` pointing at the same scenario directory (or its `_parity` sibling): lane-set equality
(CityFlow roadnet non-internal lanes vs SUMO `getIDList`), **monitored-set** equality (what `SumoMetrics`
actually logs), intersection-id equality, `lane_semantic_correspondence` outcome per intersection
(unique / raises, with the message), vType binding (`n_vehicles`, `n_typed`, `tau` present?), lane-speed
min/max, `phase` counts. Emit `docs/data/p7_1_paired_scenarios.json`. Expected rows at minimum: the four
hangzhou 1×1 variants, hz4x4 gudang, hz4x4 hetero, cologne1, cologne3, grid4x4 (inputs MISSING — a row,
not an omission). **cologne3's failure is recorded with counts (dirs by type, lanes mismatched,
monitored 38/76), not fixed.** No design work on a topology-level correspondence in this task.

### 3.2 The canonical order and the adapter

**Canonical = the CityFlow corpus's per-intersection `incoming_lanes` order** (discovery order,
`utils/cityflow_utils.py:95-110`, recorded per scenario in the artifact with each lane's `(road,
index, movement set)`), because the trained models read that order and the corpus cannot move.
`align_info(info_sumo, alignment) -> info_canonical`, a pure function over the C2 dict:

- per intersection, `state` re-ordered so the `lane_vehicle_count` and `lane_waiting` blocks follow the
  canonical order **via the movement correspondence** (SUMO lane → CityFlow lane on the same road
  serving the same movement set; refuse anything but a unique match — reuse
  `transfer_gate.lane_semantic_correspondence`, do not re-implement); `phase_onehot` re-encoded to
  width 9 by §0.4's map; `current_phase` mapped the same way; `avail_actions` passed through with
  width asserted equal to CityFlow's action count;
- `lane_vehicle_count` / `lane_waiting_vehicle_count` dicts re-keyed to CityFlow lane ids through the
  same correspondence (so the C6 lane arrays of a SUMO corpus would carry CityFlow ids — say in the
  docstring that this is a **key translation**, and record the SUMO id beside it in a sidecar rather
  than losing it);
- `metrics` filtered to a declared `metric_keys` tuple (§0.5), never silently reordered;
- everything else untouched; the function is total on a well-formed `info` and raises on any lane it
  cannot place. **Never a positional fallback.**

The inverse (`canonical → SUMO`) is needed only for actions, and P7.0 says none is needed; **test that**
(§3.3.3) instead of assuming it.

### 3.3 The load-bearing tests

1. 🔒 **Recompute P7.0's registered table through the adapter, from raw rows.** Load the 15 SUMO and
   15 CityFlow episodes in `output/p7_0/` (npz), apply `align_info`'s permutation to the SUMO lane
   arrays, and compute per-feature KS and OVL with `transfer_gate.ks_statistic_exact` /
   `overlap_coefficient_exact` (exact rationals — the values are integers, so `==` is right). **All 16
   rows must equal `docs/data/p7_0_gate.json:per_feature` exactly** (min OVL `0.340720` on
   `lane_waiting_vehicle_count@road_1_0_1_1 → road_1_0_1_0`). **Negative control:** the identity
   permutation reproduces `voided_per_feature` exactly (min OVL `0.0554`). This is the independent
   recomputation of P7.0's numbers that the paper fence requires, by different code over the raw data.
2. **Synthetic intersection with a known inversion** (two roads × two lanes, hand-written roadnet +
   net fragments): the adapter's permutation equals the hand-derived one; mutating the correspondence
   to identity fails it.
3. **Action semantics:** `transfer_gate.green_action_lane_sets` on both backends, translated through the
   correspondence, agree 8/8; under identity 4/8 (P7.0 §3.7's numbers, reproduced not quoted).
4. **Live inertness on the real env:** one SUMO episode of MaxPressure on the parity scenario; at every
   decision step `align_info(info)` must leave `sim_time`, `vehicle_count`, `average_travel_time`,
   `step`, the per-intersection `reward` and `time_in_phase` byte-identical, change only the lane
   blocks' order and the phase encoding, and be **idempotent** (`align_info(align_info(x)) == align_info(x)`
   must raise or be a no-op by design — choose, document, test).
5. **Refusal:** an `info` carrying a lane the correspondence does not know raises; a `state` whose width
   is not `2·L + 16` raises; `metric_keys` not a subset of what is present raises.
6. hz4x4 gudang: correspondence resolves 16/16 and the permutation is a proper permutation per
   intersection (a test on structure only; no simulation).

### 3.4 Provenance of hangzhou's SUMO network (§0.6)

Read `scenarios/hangzhou_1x1_bc-tyc_18041610_1h/*.net.xml` (and `.nod.xml`/`.edg.xml`/`.con.xml` where
present) and `roadnet.json`: generator headers (`netconvert` version, `<!-- generated on … -->`), node
coordinates and their relation to CityFlow's `points`, lane counts, whether `.nod/.edg` (hand-authored
inputs) exist for each hangzhou variant. State, with the evidence, which of the two sentences in §0.6 the
files support. **If the SUMO net was converted from CityFlow's roadnet, stop and put it in the packet as
an open question** — §1:21's argument would then be false and the coordinator rewrites it; that is not
yours to absorb.

### 3.5 The freeze document, §"Features"

`docs/notes/P7.1_FREEZE.md` §"Features": the frozen backend-neutral set is `lane_vehicle_count` and
`lane_waiting` over the **canonical movement-ordered incoming lanes**, plus the **9-wide green-action
one-hot** (CityFlow file-phase convention) — exactly the corpus's three features under a stated order
and a stated phase map, with `drq_norm` and every SUMO/MOSS-only metric excluded by name. Declared
residuals (reported, not aligned): transition recipe `("all_red", 5)` vs `("yellow", 3), ("all_red", 2)`;
halting threshold (measured in A1); `sigma`; `usualPosAcc/usualNegAcc`; teleports (§2.3). And the
scope sentence the paper needs: **which pairs the freeze covers (from §3.1's table) and why cologne3 is
not among them today.**

---

## 4. Scope fence — what NOT to build

- ⛔ No DT evaluation on SUMO, no zero-shot number, no few-shot corpus (P7.3). No RTG calibration (P7.2).
- ⛔ No topology-level correspondence for cologne3, no `GS_` prefix handling, no `t/L/R` mapping — record
  the failure with counts; the author decides whether cologne3 is worth alignment work.
- ⛔ No vType binding for any scenario other than the existing hz1x1 parity file; no `render_sumo`
  parity path; no `.sumocfg` generation for draws (P7.3 prerequisites — `DEFERRED` 75).
- ⛔ No edit to any shipped scenario file or to the P7.0 parity files (recorded runs used them). A new
  `.sumocfg` with `time-to-teleport` is allowed **only** as §2.3(iii)'s deliverable, as a new file.
- ⛔ No change to `offline/transfer_gate.py`, `offline/parity.py`, `offline/engine_att_reference.py`,
  `offline/admission_probe.py` — import from them; if a change looks necessary, stop and write it in
  the packet.
- ⛔ No registration text. The freeze document drafts; `PREREGISTRATION.md` is the coordinator's.
- ⛔ No `libsumo` (it does not import in this venv — `undefined symbol: PyImport_AddModuleRef`; note it
  in the packet, do not fix it).
- ⛔ No frozen file, no new dependency (`traci`/`sumolib` are already installed and already imported by
  `envs/sumo_env.py`; they are not new).

---

## 5. Gates, in order

| gate | what | proof |
|---|---|---|
| **G0** | plan `docs/plans/p7.1.md`, plan mode; assumptions with confidence; anything load-bearing under ~95 % is a question | user approves |
| **G1** | half A code + tests (skeletons red first); **smoke: 1 SUMO episode of `maxpressure`, seed 1000, through the observer**, `W_sumo == att_sumo_env`, the pool identity holding every second, and the reproduction against `output/p7_0/sumo__maxpressure/ep000000` | the record, the timing |
| **G2** | **pre-flight review** (coordinator spawns): the driver, the fence (`output/p7_1/` and `SHA256SUMS_p7_1.txt` only; a distinct function name, not a second `assert_writable`), resume, token — `BRIEF_33` §3.6's shape | CLEAR |
| **G3** | the author writes the token; A1, A2, A4 run in `tmux` (≈30 episodes × ≤ 1 min); `report` → `docs/data/p7_1_metric_freeze.json`; freeze §"Metric" drafted | artifact |
| **G4** | half B code + tests; `docs/data/p7_1_paired_scenarios.json`; §3.3 tests 1–6 green; freeze §"Features" drafted; §3.4 answered | pytest tail, the recomputed table |
| **G5** | packet `docs/returns/P7.1.md`; **merge review — BLOCKING** (ρ and the recomputed P7.0 table enter the paper) | PASS |

Implementer autonomy (§7, 2026-08-31) applies: wrong paths, CLI shape, crashes with obvious causes —
fix and disclose. **A fix that creates a measurement, anything touching a registered criterion, and two
explanations you cannot tell apart — stop.** The machine stays quiet during G3.

---

## 6. Tests — the list (write first, red for their own reasons)

`tests/test_sumo_att_reference.py`: pool identity (departed ∪ pending ∪ not-yet-due = file ids) with a
positive control that a dropped id is caught; `W_sumo == att_sumo_env` on the smoke episode (skips
without SUMO); `getDepartDelay` route equals `P − W` within 1e-9 with a mutation; the two-term identity
exact on synthetic records; refusal to write on any mismatch (barrier: validate all, then write); the
fence (refuses `output/p7_0/…`, `output/p5_3b/…`, `output/never/…`; accepts `output/p7_1/x`); the
reproduction check against P7.0's npz refuses on a 1e-9 perturbation; the report's ρ recomputed by a
second route in the test.
`tests/test_backend_alignment.py`: §3.3's six, plus `align_info` is pure (input dict unmodified —
`copy.deepcopy` before, `==` after).
Existing suites untouched except where a **named** authorisation appears in an amendment.
Hygiene: `check_test_hygiene.sh`, `check_english.sh` on every new file; test count up; CI's skip
ceiling will move (SUMO- and `output/`-gated tests) — registered protocol: merge, classify `junit.xml`,
commit the observed value as its own commit. **Do not pre-bump.**

---

## 7. Definition of Done

- [ ] `offline/sumo_att_reference.py` with the four reconstructions, the pool identity, the teleport and
      halting-threshold counters, `report`, the fence, the barrier
- [ ] A1 (15 SUMO), A2 (15 CityFlow), A4 (1 hz4x4 timing) run; `docs/data/p7_1_metric_freeze.json`
      committed; reproduction against `output/p7_0/` stated as counts (`n_equal / n`, max abs diff), never
      as "matches"
- [ ] `offline/backend_alignment.py` with `align_info`, the canonical-order artifact, the paired-scenario
      table; §3.3 tests 1–6 green, test 1 reproducing all 16 rows and the void 16
- [ ] `docs/notes/P7.1_FREEZE.md` — §"Metric" and §"Features", every number with its `n`, the scope
      sentence, the declared residuals, the teleport recommendation with the count, §3.4's provenance
      answer with evidence, and the draft amendment text (clearly marked DRAFT — not registered)
- [ ] `output/SHA256SUMS_p7_1.txt`; all thirteen manifests verify (`sha256sum -c`, counts pasted)
- [ ] Zero frozen files (`git diff --stat main...HEAD`); zero new deps; hygiene and English exit 0
- [ ] Pre-flight CLEAR before G3; merge review PASS before merge
- [ ] `docs/returns/P7.1.md` from the template: real pytest tail, the classified disclosure section,
      the AI-assistance record, *"written against BRIEF_34 (+ amendments by letter)"*, and §13 "what
      P7.2 and P7.3 will assume" — including the hz1x1 SUMO rate with `n`, SUMO's determinism as
      measured, and cologne3's status
- [ ] `PROJECT_PLAN` §6's P7.1 box ticked in the merge commit

## 8. What the next reader must not be allowed to believe

- That "SUMO has one ATT definition" means SUMO's metric equals either CityFlow one. It equals neither;
  it is `W_running`'s twin, and `E_sumo` is definable.
- That the paired set is six. It is what §3.1's table says, and cologne3 is not alignable as the files
  stand.
- That the phase feature transfers because the feature name is the same. Width 9 vs 16; the map is
  exact only because decision boundaries see greens.
- That binding a vType achieves parity. `tau` is the confound; only hz1x1 is bound.
- That a SUMO episode is cheap. 51 s on hz4x4 (`n = 1`); hz1x1 is what half A measures.

## 9. Return Packet

`docs/returns/P7.1.md` from `docs/returns/TEMPLATE.md`. Mandatory extras: the reproduction counts
against `output/p7_0/` on both backends; the halting-threshold max abs difference; the teleport count per
episode; timing per episode with `n`; the recomputed 16-row table beside P7.0's; §3.4's provenance
evidence; every deviation from the brief in one classified section.

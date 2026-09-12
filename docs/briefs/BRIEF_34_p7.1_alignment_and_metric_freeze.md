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

---

# ✅ AMENDMENT A — 2026-09-11, ruled at the plan gate on `docs/plans/p7.1.md` @ `bfe6257`

**The plan is approved. Start half A.** Every claim behind the six questions was re-verified in the artifact
by the coordinator before ruling: the `.net.xml` header (`netconvert 1.13.0`, 2022-10-19, inputs under
`/home/lxl/LibSignal/main_cp/LibSignal/data/raw_data/…`), junction coordinates equal to CityFlow `points`
plus `netOffset` with residual **0.0 on 5/5**, `.nod/.edg/.con` present for all five hangzhou scenarios;
`att_per_step` is `float32` in the P7.0 npz; `offline/transfer_gate.py:1068` runs `collect.main(argv)`;
`metrics/sumo.py:424` fills `depart_time` inside `update()`. Written against: this brief + Amendment A.

## A1 — Q1: PROCEED. The STOP fired correctly, and the correction is the coordinator's, made today

**Confirmed: every hangzhou SUMO network in this repo was generated from the CityFlow roadnet by
LibSignal's toolchain.** `PROJECT_PLAN` §1's *"natively authored in both backends, converter confound
absent by construction"* is **false**, and the NATIVE pillar of the C3 positioning is **withdrawn in §1
this turn** — cologne1/cologne3 run the other way (SUMO-native, CityFlow derived), so **no pair in this
repo is natively authored in both backends.** What replaces it is stronger than a caution, and it is now
the freeze document's job to make it checkable: **topology, routes and signal timing are shared BY
CONSTRUCTION through one documented conversion** (LibSignal's, upstream, cited), **vehicle dynamics are
matched by the parity vType (what the converter does not cover — §1:59's clause, answered in the
affirmative), and the conversion's structural artefacts are AUDITED.**
**→ Half B gains one bounded deliverable, structure only, no simulation:** a converter-artefact table
for `hangzhou_1x1_bc-tyc` and hz4x4 gudang — per lane: length and speed in `roadnet.json` vs `.net.xml`;
per intersection: the connection/turn set, the TLS phase structure (9 vs 16, yellow inserted by the
converter), lane counts; per scenario: the route/flow correspondence (2021 vehicles, departure times,
edge sequences). Report exact matches and every difference with its magnitude. P7.0 already holds part of
it (lane speeds `11.11/9.84/9.26/3.65`; 8/8 green actions). **The adapter is unaffected; the packet
carries the finding; §1:21–23 already carry the correction.**

## A2 — Q2: CONFIRMED — `np.float32(fresh) == stored`, and the float64 is recorded beside it

Exact after the storage rounding, no tolerance; `n_equal / n` reported. The artifact keeps the fresh
float64 so a future full-precision comparison is possible. This is a substitution of the exact form the
artifact permits, not a loosening — say so in the docstring.

## A3 — Q3: CONFIRMED — `offline.collect.POLICIES` through `collect.build_parser()`'s real args

**The brief was wrong and the repo won**: it named `dt_gate._maxpressure_factory` /
`method_tier_grid._fixedtime_factory` / `_random_factory` because `engine_att_reference.build_factory` uses
them, without checking what P7.0 ran. Two of the three would have changed the arm and A1's reproduction
would have failed for the brief's reason, not SUMO's. Logged as the coordinator's error.

## A4 — Q4: CONFIRMED — Layer A inside a `collect`-shaped loop, not `gate_episode`

The RNG continuity of the random arm across five episodes is exactly what the reproduction check tests;
`gate_episode`'s one-env-per-call would break it. Condition: A2's `random` cells must reproduce P7.0's
CityFlow `random` rows under the A2 form — that is the test that proves the loop is collect-shaped.

## A5 — Q5: MEASURE IT; a non-zero result is a FINDING INSIDE P7.1, and it does not stop the freeze

The brief's *"zero by construction"* was a premise stated as a fact — the plan is right to refuse it.
Ruling: (i) two counters, `n_zero / n` per episode; (ii) **if non-zero, SUMO's identity gains its third
term** — `att_sumo_env − W_sumo` is SUMO's cadence term, reported exactly like `C − W_running` on CityFlow,
so the decomposition stays symmetric with A13; (iii) the brief's `W_sumo == att_sumo_env` check becomes:
equal under `==` whenever the counter is zero, otherwise the difference **is** the cadence term and is
reported per episode. **Why it does not stop the freeze: the frozen primary (`E_sumo`) is computed by
the per-second observer and does not inherit the env metric's bookkeeping; the env metric is the
co-reported definition.** A non-zero count is also a finding about `metrics/sumo.py` (frozen) — write it
as an open question in the packet, never as a fix.

## A6 — Q6: CONFIRMED — 30 SUMO episodes (15 observed + 15 unobserved)

The unobserved arm is the observer-interference control and the timing basis at once; both are needed.
Report the observed/unobserved wall clocks per arm with `n = 5` each, beside the plan's `12.44 s`
(`n = 1`).

## A7 — Recorded, no action

`git checkout -b task/p7.1-alignment main` in the linked worktree — correct; `git checkout main` is
impossible there and the brief should not have implied it. hz1x1 SUMO **12.44 s/episode** (`n = 1`) is the
first hz1x1 SUMO timing on disk; SUMO's bit-for-bit reproduction across five weeks and two worktrees
(3 episodes, 2 routes) is the determinism finding P7.3 needed — both go into the freeze document.

---

# ✅ AMENDMENT B — 2026-09-11: half B's audit table is the first instance of the Conversion Audit Protocol

`PROJECT_PLAN` §6 P11.5 was narrowed today into CAP (criteria A–H, quoted there). **Amendment A1's
converter-artefact table is built to CAP's columns**, so the same tool audits a second pair later:
**(A)** intersection/road bijection and lane counts · **(B)** junction-coordinate residual, per-lane length
and speed with max abs difference · **(C)** connection sets under the movement correspondence, extras
enumerated (hz1x1 has 4 `t` connections, hz4x4 16 — count them from the files) · **(D)** released-lane
sets per action (reuse `transfer_gate.green_action_lane_sets`), phase counts, the `phase_onehot` map,
and the sentence that `SumoEnv` renders transitions itself (`envs/sumo_env.py:252-278`) with the
converter's clearance states quoted (`r` on hz1x1, `s` on hz4x4) · **(E)** demand equality per vehicle
(id, depart, route) between `flow.json` and the parity `.rou.xml` · **(F)** provenance from the
`.net.xml` header (tool, version, date, input paths) — the LibSignal lineage, cited. **(G)** is half A's
output and is cross-referenced, not recomputed. Scenarios: `hangzhou_1x1_bc-tyc` (full) and hz4x4
gudang (A–D, F; no parity file exists for E — say so).

**One new structure-only line, no retrieval:** record grid4x4's conversion signature from its CityFlow
files alone (phase structure, even-indexed greens in the corpus, the flow block's SUMO defaults, the
dangling `.sumocfg`) as evidence for `DEFERRED` 77. Do not download anything; do not run `converter_v2`.

---

# ✅ AMENDMENT C — 2026-09-11: the grid4x4 pair is on disk; half B audits it too (A–D, F; E re-derived)

`DEFERRED` 77 is resolved: our shipped `scenarios/grid4x4/*.json` are sha256-identical to LibSignal's
`data/raw_data/grid4x4/` (their s2c conversion), and the SUMO-native original is RESCO's —
`scenarios/grid4x4_candidates/resco/resco_benchmark/environments/grid4x4/grid4x4.net.xml`
(`netedit 1.9.0`, sha256 `8d192de4…`) plus `grid4x4_1.rou.xml` **inside** `grid4x4.zip` in the same
directory (sha256 `2350dce7…`), which matches our `flow.json` on 1,473/1,473 exact (depart, route) pairs.
**The candidates directory is gitignored and READ-ONLY for this task; the files are CC BY-NC-SA 4.0 and are
never copied into the tree.** Read the route file directly from the zip (`zipfile`), never extract into
the repo.

**Half B adds the grid4x4 pair to §3.1's table and to the CAP audit** with the same tool it builds for
hangzhou: **(A)** junction/edge bijection (ids are shared: `A0…D3`, `A0A1…`), lane counts; **(B)**
coordinates (RESCO `netOffset 0,0`), per-lane length and speed (net 13.89 vs converted flow 13.39 — report
it as the converter's cap, the cologne shape); **(C)** connection sets under the correspondence — note RESCO
has no `t` connections, 576 = 192 × {r, s, l}; **(D)** released-lane sets per action against the CityFlow
roadnet's 16 phases (the odd CityFlow phases carry 2–5 roadLinks — the converter's rendering of SUMO's
yellow/`s` states; state exactly what the CityFlow env treats as actions on this scenario, read from
`envs/phase_control.py` and the corpus's `avail_mask` width, and what the SUMO env would); **(E)** the
1,473/1,473 demand equality re-derived by the tool, plus the vType situation (the route file defines
none → `DEFAULT_VEHTYPE`; parity instance for grid4x4 recorded as values only: `maxSpeed 13.39, tau 1.5,
accel 2.6, decel 4.5, length 5.0, minGap 2.5, width 1.8, speedFactor 1.0` — no parity file is built,
that is `DEFERRED` 75); **(F)** provenance: RESCO clone sha `f1ed9a174f8de41fc9d8689373b836bc882570dc`,
LibSignal clone sha `127af9f93902778e556de2eedb2b606c4c9447e6`, the zip member name, both sha256s, the
`netedit` header — and the fact that LibSignal's converter, not `converter_v2`, produced our CityFlow side
(so the three `converter_v2` TODOs are cited as the class, with their LibSignal-side counterparts
identified where the files show them). **No simulation on grid4x4 in this task** — (G) is P7.3's.
Add ≈ half a day to half B; the table is the deliverable A14's admission rests on.

---

# ✅ AMENDMENT D — 2026-09-11, at G1: accepted on a reproduction, and A1 gains a teleport-free twin

**G1 is accepted — on evidence the coordinator produced, not on the relay.** `output/p7_1` did not exist
and no file on the branch carried the smoke's numbers, so the coordinator ran the branch's own CLI
(`run-sumo --arm maxpressure`, observed and `--no-observer`, `--episodes 1`) into a scratch work dir:
`att_env 355.7984322508399` both ways, `reproduces_p7_0 True` (`np.float32` form, float64 gap
`8.5e-7`), `E_sumo 442.3349826818407`, `P 452.5783874580067`, `W == att_env`, terms
`+10.243404776166017 / −96.77995520716684 / 0.0`, created 2021 / entered 1786 / never 235 = pending 235,
teleports 13, halting 28,800 lane-seconds and 0 disagreements, clock-origin second route `1.42e-14`;
43.2 s observed with the halting check, 12.3 s frozen env (`n = 1` each). Every relayed number matches.

## D1 — A record is a file. The smoke goes under `output/p7_1/smoke/` through the driver

The driver (G2) carries a smoke stage — one observed `maxpressure` episode into
`output/p7_1/smoke/` — and the packet cites that file, not a table typed from a terminal. Same shape as
`BRIEF_33` §3.6 / plan §4.5.

## D2 — 🔒 A1b: the same 15 episodes with teleports DISABLED, as a NEW parity `.sumocfg`

13 teleports in one MaxPressure episode means §2.3(iii) fires, and a recommendation with a count and
no effect size is a formula, not a number. **Add A1b:** `scenarios/hangzhou_1x1_bc-tyc_18041610_1h_parity/
hangzhou_1x1_bc-tyc_18041610_1h_parity_noteleport.sumocfg` — a NEW file identical to the parity one plus
`<processing><time-to-teleport value="-1"/></processing>` (the shipped and the P7.0 parity files stay
untouched; recorded runs used them) — and the three arms × 5 observed episodes on it. Assert per episode
`n_teleports == 0` and `n_vanished_without_arrival == 0`. **A1 (teleports enabled) remains the reproduction
of P7.0 and is labelled *teleports enabled*; A1b is the candidate frozen regime**, and the freeze reports
the per-arm difference between them under both definitions. ≈ +11 min. G3's total stays under an hour.

## D3 — Accepted as measured, and the brief is corrected where it was wrong

- **The decomposition residual on SUMO is a tautology** — `(P−E)+(W−P)+(att−W)−(att−E)` cancels
  identically — and is **not** a gate; the two independent routes are (clock origin: `W−P` against mean
  `getDepartDelay`, `1e-9`; population: `n_pending_at_horizon == n_never_entered` and the denominator
  equal to the route file's `depart ≤ T` count). The brief's §2.1 *"identity exact"* is a consistency
  check, not evidence; the artifact says `residual_is_tautological`. This is `P5.3b-fix` §2.7's class,
  degenerate on SUMO because no engine call anchors `E`.
- **The reproduction check's detection floor is one float32 ulp** (~`3e-5` at ATT 355); the brief's
  *"refuses a 1e-9 perturbation"* was unsatisfiable against a float32 record and is replaced by the
  one-ulp test that exists.
- **The halting cross-check costs 2.9× on its own** (A7 falsified); it runs on a declared episode
  count (default 1 per arm, 28,800 lane-seconds each) and every row carries its lane-seconds.
- **The one-second offset** — SUMO's first observation of a vehicle is `intended_depart + dt` (`dt` from
  `getDeltaT`, cross-checked against the observed grid), the twin of Gate 0's `first_seen = enterTime +
  interval`. **Pin it with a named mutation in the packet: drop the `+ dt` → the pool identity must
  refuse at the first second a vehicle departs.**

## D4 — Then the driver, then the pre-flight

`offline/campaigns/p7_1_metric_freeze.sh`: token consumed on start after `cd` and an import check;
start lock on `pgrep -f 'python.*offline\.sumo_att_reference'`; `trap` killing the process group;
stages A1 (observed + unobserved), **A1b**, A2, A4, smoke, `report`; `SHA256SUMS_p7_1.txt` last,
atomic, `smoke/` included this time (it is part of the evidence) — state that in the header. The
coordinator spawns the pre-flight on the commit that carries it.

---

# ⛔ AMENDMENT E — 2026-09-12, on the pre-flight (PART 1 partial + PART 2): NOT YET CLEAR. Six fixes, then the token

`docs/reviews/P7.1-preflight-part2.md` — **CLEAR WITH CONDITIONS, 0 blocking, 3 major, 8 minor** (resume,
signals, lock) — and `docs/reviews/P7.1-preflight-part1.partial.md` — **0 blocking, 0 major, 3 minor**
(destruction paths; the reviewer died on an API limit with its findings file on disk, and the coordinator
closed the open items from it). Nothing found can destroy or corrupt data through the driver; both real
`output/` trees were byte-identical to the 20:47 baseline after every experiment. **The three majors are
gates that hold only by stage order or by the runner, i.e. by a file rather than by the code — the same
shape `BRIEF_33` Amendment C fixed before its run, for the same reason: tightening before the first
episode is free.**

## E1 — REQUIRED before the token (~1 h; each with a test or an executed demonstration)

1. **PART 2 MAJOR 1** — `chunk_is_reusable` and `freeze_artifact` re-check `n_teleports == 0` and
   `n_vanished_without_arrival == 0` on every observed row of a `noteleport` chunk; a stored verdict is not
   evidence. Test: the reviewer's chunk with `rows[1].n_teleports = 1` → re-run, and → `report` REFUSED.
2. **PART 2 MAJOR 2** — `report` asserts the campaign's cell set (the 12 expected labels from the driver's
   stages: 3 arms × {parity observed, parity unobserved, noteleport} + 3 CityFlow), `chunk["episodes"] ==
   args.episodes` for every `run-*` cell, and label uniqueness across files; a missing or duplicated label
   → REFUSED, artifact absent. Test: delete one chunk; put a parity-header chunk under the `noteleport`
   filename.
3. **PART 2 MAJOR 3** — a complete chunk with any `reproduces_p7_0: false` row is **not reusable**: on
   restart the driver renames it `<name>.failed.json` (evidence kept, outside the `freeze_*.json` glob) and
   re-rolls; the "complete and clean" message says what it checked. Test: the reviewer's `1l` chunk.
4. **mn-1** — A4 idempotent (skip a complete `timing_hz4x4_gudang.json`; it costs 7.3 min under the
   observer, measured). **mn-2** — logs appended, never truncated on restart. **mn-6** — `n_teleports:
   null` → `REFUSED: …`, not a traceback.
5. **PART 1 minors** — drop or reorder the unreachable interpreter check; the driver header's schedule
   corrected to measured rates: **A4 439 s (observer, `n = 1`), hz1x1 43 s observed / 12 s bare, G3 ≈ 33
   min**, not ~17.
6. The three surviving-mutation shapes from PART 2's test audit get real tests where cheap: rows-level
   regime mismatch; header `episodes` ≠ CLI; a non-reproducing chunk is not reusable.

## E2 — Operator conditions, retained from PART 2 (they hold after E1 too)

Launch as a **tmux foreground** job (`bash offline/campaigns/p7_1_metric_freeze.sh` from the worktree) —
never `nohup`/`&`, which inherits `SIGINT` ignored and disables the trap; `report` is never run by hand;
`output/p7_1/` must not exist at first start and nothing is ever copied into it; other sessions whose
command lines contain the module name will trip the start lock (fail-closed, nothing consumed).

## E3 — Then CLEAR, without a second reviewer round

The coordinator verifies E1's six on the branch (the reviewer's constructed states are recorded in both
findings files and are cheap to replay), then writes CLEAR in the Decisions Log; the author writes the
token only after that line exists.

---

# ✅ AMENDMENT F — 2026-09-12, at G3: the run is accepted, and the freeze's §"Metric" has its numbers

**Run `COMPLETE` in 836 s (13 min)**, 29-entry manifest, all 13 manifests verify (864 files). Read by the
coordinator from `docs/data/p7_1_metric_freeze.json` on the branch, **pre-review**: reproduction **45/45**
against P7.0 on both backends (float32 form; float64 gaps ≤ `1.3e-5`); **SUMO's cadence term is `0.0` on
30/30 observed episodes** (the 15 non-zero rows are CityFlow's, 3.8–4.8 ATT); halting **172,800
lane-seconds, 0 disagreements**; clock-origin second route max `5.7e-14`; `n_vanished_without_arrival`
0 everywhere.

**The result, per arm (means of 5; `env` = the backend's admitted-population metric, `E` = the
pool-clock all-created twin; terms in `att_env − E` orientation: population + clock origin + cadence):**

| cell | env | E | population | clock origin | cadence | never entered | teleports |
|---|---|---|---|---|---|---|---|
| cityflow fixedtime | 285.42 | 499.92 | −3.41 | −215.87 | +4.78 | 360 | — |
| cityflow maxpressure | 247.75 | 264.72 | +4.15 | −25.10 | +3.99 | 58 | — |
| cityflow random | 421.59 | 846.84 | −8.75 | −420.33 | +3.82 | ≈780 | — |
| sumo fixedtime (parity) | 371.63 | 689.83 | +2.80 | −321.00 | 0 | ≈572 | 0 |
| sumo maxpressure (parity) | 360.02 | 452.20 | +11.19 | −103.37 | 0 | ≈254 | 13–15 |
| sumo random (parity) | 483.88 | 954.54 | −7.17 | −463.49 | 0 | ≈891 | 0 |
| sumo maxpressure (noteleport) | 364.16 | 461.46 | +11.81 | −109.12 | 0 | ≈257 | 0 |

(fixed-time and random are bit-identical between the two SUMO regimes — no teleports occur there.)

**ρ, both backends, both definitions (`fixedtime = 0`, `maxpressure = 1`):**

| definition | Δ CityFlow | Δ SUMO | Δ SUMO-noteleport | ρ_random CityFlow | ρ_random SUMO | ρ_random SUMO-nt |
|---|---|---|---|---|---|---|
| admitted (`env`) | 37.66 | 11.61 | 7.47 | −3.615 | −9.673 | −15.034 |
| pool-clock (`E`) | 235.20 | 237.63 | 228.36 | −1.475 | −1.114 | −1.159 |

⭐ **Under the admitted definitions the two backends' anchor spreads differ 3.2× and ρ_random by 6.06 —
P7.0's A4 failure. Under the pool-clock pair the spreads agree within 1 % and ρ_random within 0.36.**
The 2026-08-31 threat was not a possibility to be argued away; it was the A4 result.

## F1 — What `docs/notes/P7.1_FREEZE.md` §"Metric" DRAFTS (the coordinator registers; you propose)

1. **The definition pair for ρ:** pool-clock, all-created on both backends — `att_engine` on CityFlow
   (already primary on hz1x1 by Rule R) and `E_sumo` on SUMO — with the admitted pair (`att_ours`, the
   SUMO env metric) reported beside it in every table, A11(b)'s discipline extended. Give the reason in
   the numbers above, not in prose about "fairness".
2. **The regime:** `time-to-teleport -1` (the `noteleport` `.sumocfg`) as the frozen SUMO configuration
   for every later SUMO measurement, with the measured effect (MaxPressure +4.1 env / +9.3 E; the other
   two arms unchanged) and the statement that P7.0's SUMO cells and A1 ran with teleports enabled.
3. **The decomposition table above, per backend per arm**, as A13(b) requires of every report of the two
   definitions' difference — now on both backends.
4. **A11(d)(6), executed:** P7.0's ρ-based criteria (A4 and B4; A1/B1 are Δ-signs) re-scored under the
   pool-clock pair, **both verdicts reported, no new branch decision** — the branch was C on A2 (state
   overlap) as well, and A2 is not ATT and does not move. State exactly which criteria change verdict and
   by how much (`0.5·M` recomputed under E).
5. **SUMO's determinism** (45/45, float32 form, with the ulp floor stated) and **the timing table with
   `n`**: hz1x1 SUMO bare 10.4–11.8 s/episode (`n = 5` per arm), observed 17.3–19.3 (halting on 1 of 5),
   CityFlow 0.89–1.05 through Layer A, hz4x4 SUMO 438.7 s observed / 51.0 s bare (`n = 1` each).
6. **The draft amendment text**, marked DRAFT, in the shape of A11–A14's rows.

## F2 — Then G4 (half B) as briefed, with Amendments B and C's additions

Commit the artifact first (D4's NEXT STEPS). Half B's paired-scenario table and the CAP audit for
hangzhou (both scenarios) and for the grid4x4 pair from the candidates directory; `align_info` and the
six tests of §3.3; §3.4's provenance answer is already established (Amendment A1) — cite it, do not redo
it. Packet at G5: *"written against BRIEF_34 + Amendments A–F"*.

---

# ✅ AMENDMENT G — 2026-09-12, at G5: the packet is on the branch; F5's A4 rate is WITHDRAWN as the coordinator's error

`docs/returns/P7.1.md` @ `6be6603`, written against A–F, 381 lines, diff stat byte-identical to the tree.
**Merge review runs as two SEQUENTIAL reviewers with findings files (half A, then half B).**

## G1 — A4's rate: 50.31 s (campaign chunk) / 60.59 s (idle re-measure), `n = 1` each; 438.69 s withdrawn

Amendment F5 quoted **438.7 s per hz4x4 episode under the observer**. The campaign's own
`timing_hz4x4_gudang.json` records **50.31 s** and the implementer's re-measurement on an idle machine
**60.59 s**, bit-identical outputs. **The 438.69 s was the coordinator's pre-flight completion run on
2026-09-12 ~09:12, and its own log shows what it measured: `real 7m23s` against `user 1m42s + sys
1m38s` — the process was waiting, not computing.** The implementer's G4 demonstrations were rolling
SUMO episodes on the same machine at that hour; per-second traci observation over 240 lanes is 3,600
rounds of socket calls, and under CPU contention each call's latency balloons. **A wall clock carries no
signature of the load it was taken under — the plan's own sentence — and the coordinator broke the
*machine stays quiet* rule with a measurement, then reported it as a rate.** Withdrawn as a rate; kept
in the record as one data point on contention. The freeze quotes **51–61 s** for P7.3's hz4x4 SUMO
planning, `n = 1` each, and E1.5's "A4 alone 7.3 min" is void.

## G2 — What the reviewers are asked, so the packet's open questions do not wait on them

The reviewers verify; the coordinator rules. Open question 2 of the packet is answered by G1. Open
question 1 (the paired set is 4 of 6, one network) and the DRAFT amendments go into A15 after PASS.

---

# ⛔ AMENDMENT H — 2026-09-12, on the merge reviews: FIX-FIRST, both halves, one round

`docs/reviews/P7.1-halfA.md` — **PASS-WITH-NOTES, 0 blocking, 2 major, 4 minor, 8/9 mutations killed.**
`docs/reviews/P7.1-halfB.md` — **FAIL, 1 blocking, 6 major, 6 minor, 9/14 killed.**
**Every shipped number in both halves recomputed by the reviewers' own routes and matched** (the 16-row
table 16/16, the void twin 16/16, the released-lane sets from the files 8/8 and 4/8, ρ and Δ to
≤ 1.4e-13, 45/45 against P7.0's npz, the audit's A/D/E/F cells, the provenance hashes). **What failed
is completeness and protection: two cells of the paper's audit table were never computed, and five
mutations on half B survive.** The same shape as `P5.3b-fix` Amendment E, and the same ruling.

## H1 — REQUIRED, half A

1. **(Major 1)** `_stored_reference` returns NaN and `has_p7_0_reference` is `False` for any cell without
   a P7.0 counterpart — the hz4x4 timing scenario first; the A4 reuse test's fixture is produced by the
   runner path (or asserts the runner's behaviour on a gudang-shaped config), never hand-written with
   `nan`. Then **re-roll A4 once through a driver restart** (new token; the restart moves the false
   chunk to `failed/`, re-rolls, re-runs `report`, rewrites the manifest) so the shipped chunk is honest.
2. **(Major 2)** D3's population second route ASSERTED in `freeze_artifact` and in the recorder:
   `n_pending_at_horizon == n_never_entered` per row, and `n_created` equal to an independent count of
   `<vehicle depart=… ≤ T>` in the route file by a second code path (a regex pass, not the parser);
   a synthetic test with a vehicle in `(T − dt, T]` so the reviewer's M3b dies.
3. **(Minors)** label `6.057427` as P7.0's float32-derived statistic beside the float64 `6.057425`; drop
   the 60.59 s figure from the freeze and the packet (not on disk — a number a clone cannot see is not
   quoted) or put its chunk under `output/p7_1/` through the driver; correct the "7 unit tests" claim
   to the count actually observed.

## H2 — REQUIRED, half B

1. **(Blocking B-1)** CAP(B) and CAP(C) implemented as CAP defines them, with tests and the reviewer's
   surviving mutations killed: **B** — per-lane length (CityFlow polyline from `points` vs SUMO lane
   length) and speed, enumerated per lane with the maximum absolute difference (the reviewer measured
   10.4 / 27.2 / 27.2 m — those numbers enter the table, labelled as converter residuals); **C** — the
   (in-lane → out-lane) connection set under the movement correspondence equal to CityFlow's
   `laneLinks` (16/16, 576/576, 576/576 by the reviewer's route), any missing connection a FAIL, extras
   enumerated. Mutations M9 (netOffset ignored), M11 (extras never enumerated), M12 (speed sets → count
   equality) must be KILLED.
2. **(M-1)** `align_info` refuses any lane not in the roadnet's incoming ∪ outgoing set — exact
   membership, never a name-prefix test; test with `road_1_1_99_0`.
3. **(M-2)** the phase map is conditional on the pair's phase structure: applied only where CityFlow
   and SUMO phase counts are (9, 16) hangzhou-shaped; on same-width pairs it is the identity and the
   alignment says so; `alignment_for_scenario` records the env's action count per scenario by the
   env's own rule (`envs/phase_control.py` — duration > 5 s → green) and the freeze §7 states it for
   hz1x1 and grid4x4, as Amendment C(D) asked.
4. **(M-3)** a regenerating command for all three artifacts (`python -m offline.conversion_audit report`,
   `python -m offline.backend_alignment tables`, or one CLI) from a declared pair list, byte-identical
   on re-run; `paired_scenario_table` excludes `scenarios/*_candidates/` by default and records paths
   relative to a declared candidates root, so a clone with or without the CC BY-NC-SA tree gets 20 rows
   and never reads it by accident; the hand-written `notes` become generated fields or are dropped.
5. **(M-4)** the metric filter tested (M6 killed); E's multiset comparison with a negative control
   (M8a killed).
6. **(M-5)** §3.3 test 3 DERIVES the released-lane sets via `transfer_gate.green_action_lane_sets`
   from the network files; the gate's stored sets are the cross-check, not the source.
7. **(M-6)** docstrings corrected: the `SKELETON … NotImplementedError` line gone; the permutation claim
   replaced by what the artifact records (`[7,6,4,5,2,3,0,1]` on hz1x1, identity 1/9).
8. **(Minors)** `RLTRAFFIC_GRID4X4_RESCO` semantics and no `/home/filip` default; phase index 16
   refused; cologne3's direction counts (`t` 39, `L` 1, `R` 1) in the paired table; E's `(depart, route)`
   key stated with the reason (`flow.json` has no ids).

## H3 — Verification, then merge; no third reviewer

The coordinator re-runs on the fixed branch: the six surviving mutations (M3b; M6, M8a, M9, M11, M12),
the reviewers' own recomputation scripts (kept in their scratch dirs: `check_c_d.py`, `recompute16.py`,
`recompute_A.py`) against the regenerated artifacts, the byte-identical regeneration of the three
half-B artifacts and the freeze artifact from the CLIs, and `output/SHA256SUMS_p7_1.txt` after the A4
re-roll. Merge on all green; the P7.0 fence lifts with the merge. Packet: *"written against BRIEF_34 +
Amendments A–H"*.

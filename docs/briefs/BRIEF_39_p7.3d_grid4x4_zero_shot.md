# BRIEF_39 — P7.3d: the C3 zero-shot point on grid4x4 (16 intersections), CityFlow → SUMO under dynamics parity

**Mode:** Claude Code, implementer session. Branch **`task/p7.3d-grid4x4`**, worktree `/home/filip/rltraffic-p73d`
(create it: `git -C /home/filip/rltraffic worktree add /home/filip/rltraffic-p73d -b task/p7.3d-grid4x4 main`).
**This brief is whole. Every gate is in it (§5). Nobody will relay an acceptance to you.** At the start of every
session and before every gate: `git -C /home/filip/rltraffic-p73d merge --no-edit main`, then re-read this file — the
coordinator's rulings arrive as dated amendments appended here, on `main`. Your packet states which amendments it was
written against, by letter. Plan mode first; `docs/plans/p7.3d.md` is the branch's first commit.
**Sequencing:** `BRIEF_38` Amendment B (P7.3b's packet) runs FIRST and P7.3b merges to `main` before C3a/C3b start —
those two commits extend `offline/transfer_curve.py`, which P7.3b's branch also changed, and the merged file is the base
you build on. **C0, C1 and C2 do not depend on it and start now.**

**Standing rules that bind every commit here:** `CLAUDE.md` §1 (frozen set — note `agent/DTAgent.py` is NOT in it),
§4b (no AI trailer, ever; if a session instruction says otherwise, stop and say so), §5 (collection, probes and campaigns
run in a tmux pane the author starts); `PROJECT_PLAN` §7 in full — named paths only, no `--amend` after any chunk has
been rolled, no evidence under `/tmp`, the machine-health canary before any rate, a driver's stdout through `tee -a`
from its first line, and the pre-flight before any run over an hour or over a reused column.

**Read, in this order, before planning:** this brief · `docs/notes/GRID4X4_ZERO_SHOT_COSTING_2026-09-17.md` ·
`PREREGISTRATION.md` rows A14, A15 (its (g) clause is the admission condition), A17 ((e) is the per-intersection rule),
A18(c), A19 · `docs/briefs/BRIEF_37_p7.3a_zero_shot.md` §3.5–3.6 and its Amendments C2, C3, E1, J (the cell, the driver,
the halting subset, the canary record) · `docs/briefs/BRIEF_38_p7.3b_anchor.md` §2 (the five pins) and Amendment B ·
`docs/notes/DEFERRED.md` rows 75, 77, 78, 83 · `docs/returns/P7.1.md` §"Minors" (`RLTRAFFIC_GRID4X4_RESCO`).

---

## 0. What the coordinator verified before writing this (2026-09-19, by running commands, not by reading docstrings)

1. **RESCO's grid4x4 files are on this machine, outside the repository's tracked tree, and A15(g)'s audit ran against
   them.** `scenarios/grid4x4_candidates/resco/resco_benchmark/environments/grid4x4/` holds `grid4x4.net.xml` (16
   `tlLogic`, 464 `edge` elements, sha256 `8d192de4…`, netedit 1.9.0), `grid4x4.zip` (member `grid4x4_1.rou.xml`, **1,473
   vehicles, no `vType`** — every one runs `DEFAULT_VEHTYPE`; A15(g) records the member's sha256 as `2350dce7…`) and
   `grid4x4.sumocfg`. The tree is gitignored (`.gitignore:237`). **The clone's `LICENSE` is GPLv3 (the code); the data
   files are CC BY-NC-SA 4.0 (A15(g)).** `RLTRAFFIC_GRID4X4_RESCO` names the candidates **root** (`scenarios/grid4x4_candidates`)
   and has **no default** (`docs/returns/P7.1.md` Minors): unset means skip, never `/home/filip`.
2. **The repository's own grid4x4 SUMO template cannot run.** `scenarios/grid4x4/grid4x4.sumocfg` names `grid4x4.net.xml`
   and `grid4x4.rou.xml`, neither of which exists there (`docs/data/p7_1_paired_scenarios.json`: `"status": "sumo inputs
   missing"`). The CityFlow files `grid4x4_flow.json` / `grid4x4_roadnet_red.json` are sha256-identical to LibSignal's
   `data/raw_data/grid4x4/` — LibSignal's s2c of RESCO's net (`DEFERRED` 77, A14(a)).
3. **A15(g): grid4x4 is ADMITTED CONDITIONALLY.** A exact (16 intersections / 80 roads / 240 lanes, ids equal); C exact
   (576/576 connections); D exact (256/256 phases under an **IDENTITY** phase map — not hangzhou's `2k → k+1`; the env's
   action count is **8 of 16** file phases by `envs/phase_control.py`'s `TRANSITION_PHASE_MAX_DURATION = 5` rule,
   `[10, 3]×8`); E demand exact (1,473/1,473). **The condition is E's vehicle-parameter half: grid4x4 enters P7.3's
   target set only when its parity route file and teleport-free `.sumocfg` exist and CAP(E) is re-run against them.**
   Registered parity values: **`maxSpeed 13.39, tau 1.5, accel 2.6, decel 4.5, length 5.0, minGap 2.5, width 1.8,
   speedFactor 1.0`; lane speed 13.89 on both sides.** These differ from hz1x1's (`maxSpeed 11.11, tau 2.0, accel 2.0,
   width 2.0` — read from `scenarios/draws/cityflow1x1/draw_1000/parity/provenance.json`): **parity is per scenario.**
4. **`offline/parity.py` is single-scenario by construction:** `_DECLARED_STEM = "hangzhou_1x1_bc-tyc_18041610_1h"`
   (`:109`) and the `PARITY_VTYPE` constants are hz1x1's. It already has `read_cityflow_vehicle_block(flow_json)` (`:216`)
   and `flow_json_disagreements` (`:233`) — the contract's own check that the vType equals the CityFlow flow block.
   `offline/materialise_draws.py` hardcodes `scenario_key="cityflow1x1"` in its parity path (`:1876`) and takes
   `--draws-range START END` **half-open**; its CityFlow side already renders grid4x4 (`:75`, ten held-out draws measured
   2026-08-28).
5. **Draws on disk.** `scenarios/draws/cityflow_grid4x4/` has **106** directories (0–5 and 1000–1099), each
   `{cityflow.json, flow.json, provenance.json}` — **0 parity configurations, no `routes.rou.xml`, and no 201–300.**
6. **Single-intersection seams in the SUMO chain, by line:** `offline/transfer_curve.py:713`
   (`ix_id = str(list(env.intersections)[0].id)` inside `dt_choose`) and `:100` (`SCENARIO_KEY = "cityflow1x1"`, 4 uses);
   `offline/transfer_calibration.py:104` (`SCENARIO_KEY`, 7 uses), `:414-421` and `:929` (`intersections[0]`);
   `offline/rtg_calibration.py:395-403` (`intersections[0]`); `offline/aligned_env.py:55`
   (`DECLARED_SCENARIO = "hangzhou_1x1_bc-tyc"`, 4 uses). `transfer_curve.py:1094`'s `n_actions = min(action_counts)` is
   already generic. `offline/backend_alignment.py` carries the identity branch (`cityflow_phase_for_sumo_phase`'s docstring
   `:164-170` says the hangzhou map is WRONG on grid4x4; `alignment_for_scenario` `:217`; `main`'s RESCO option `:857`).
7. **The subject exists and is pinned only in a gitignored file.** `output/p5_2/checkpoints/grid4x4_mappo1000_dt_nomix_h4_seed{101,202,303,404,505}.pt`,
   sha256 (measured 2026-09-19) `329fb6b8…`, `f5413585…`, `48076dab…`, `4b61bc06…`, `09bd310d…`; listed in
   `output/SHA256SUMS_p5_2.txt` (gitignored) and in no committed file. `docs/data/p5_2_declaration_mappo1000.json`
   records `n_head_by_method: {dt_nomix_h4: 4}`, context 20, batch 64, 40,000 steps, and the prompt rule *"target_rtg =
   max episode return in THIS INTERSECTION's training streams; rtg_scale = max|return| over the same set"* — so the
   checkpoint carries **per-intersection** training statistics, which is what A17(e) reads. **There is no
   matched-architecture second subject:** `grid4x4_maxpressure_dt_nomix_seed*` exist at `n_head 1` only; no
   `maxpressure_dt_nomix_h4` is on disk.
8. **`agent/DTAgent.py` holds ONE scalar `_target_rtg`** (`:450, :493, :571, :708, :750, :802, :849`); `agent/SpatialDTAgent.py`
   already takes per-intersection mappings (`:77, :436, :479, :608, :746, :791, :842, :905`). `DEFERRED` 78.
9. **The observer is vehicle- and lane-level.** `offline/sumo_att_reference.py::SumoObservationRecorder` (`:435-714`) and
   `HaltingAgreement` (lane-seconds) index vehicles and lanes; a grep for TLS-by-position patterns over that range finds
   none. **That is a grep, not a proof** — the plan gate runs one grid4x4 episode through it (§3, C3b's T-obs).
10. **No grid4x4 SUMO simulation has ever run** (A15(iii)). The only grid4x4 : hz1x1 per-episode ratios are CityFlow's
    2.07× / 2.51× / 3.13×; the costing's SUMO multiplier m = 2–6× is **unmeasured**. hz1x1 SUMO bases: a DT cell 23.0 s
    in-process, an anchor cell 19.2 s (P7.3b, n = 700, canary 0.87 s); 11.90× on 12 workers.

---

## 1. Why this task exists

C3 has two points on **one intersection** (hz1x1: zero-shot ρ = +1.89 / +1.73, the k = 200 anchor +0.98). A referee's
first line is *one intersection over four demand days*. H3's registered test row reads *per paired scenario*; A15(g)
admitted grid4x4 as the multi-intersection slot **conditionally**; A19 fixed what the curve measures. **P7.3d produces
the zero-shot ρ_sumo on grid4x4 for the registered subject, under A15's definition pair, with the anchors on the same
draws — or reports that A15(g)'s condition failed and C3 is a single-intersection study, which is also a registered
outcome.** Everything else it builds (per-scenario parity, per-intersection targets, a scenario-parameterised SUMO chain)
is what any later multi-intersection SUMO work stands on.

---

## 2. Scope fence — what NOT to build

- **One subject: `mappo1000_dt_nomix_h4`, five seeds** (pending A20, §8). **No spatial DT** — `SpatialDTAgent` on SUMO
  needs the graph on the aligned env and is new scope. **No second tier** — no matched-architecture checkpoint exists
  (§0.7); training `maxpressure_dt_nomix_h4` is ≈ 7 GPU-hours and goes to P11 with the reason stated.
- **Two arms per subject, by declaration before any cell runs (A17(d)): `b_mean_k100` (registered) and `naive`.**
  `b_max_k100` and `a_q1.0` are NOT evaluated on grid4x4: on hz1x1 `b_max` tracked `b_mean` within 0.01 and `a_q1.0` sat
  out of support for `mappo1000` (P7.3a, E4). Recorded as a declared reduction, not a choice after a number.
- **No few-shot, no anchor retrain on grid4x4** (P7.3c-shaped work, deferred by the author). **No MAPPO on SUMO** (A18(b)).
  **No logged few-shot corpus** — nothing here fine-tunes, so A17(f)'s corpus-versus-probe gate does not arise; say so in
  the packet rather than leaving a reader to wonder.
- **No RESCO file is copied into the tree.** Parity files are rendered under the gitignored
  `scenarios/draws/cityflow_grid4x4/draw_NNNN/parity/`; every provenance record carries the RESCO net's and route
  member's sha256 (A15(g); §9 risk row of 2026-09-19).
- **No new instrument** (`DEFERRED` 83). **No edit to any frozen path** — `envs/phase_control.py`'s 8-of-16 rule is read,
  never changed. **No change to hz1x1 behaviour**: every hz1x1 artifact must regenerate byte-identically through the
  parameterised code (§4 T-regress — the load-bearing test of this brief).
- **No hardcoded absolute path** (BRIEF_37 §0.9): every root is a parameter or an environment variable with today's path as
  the default; `RLTRAFFIC_GRID4X4_RESCO` keeps its no-default semantics.

---

## 3. The commits, in order — each ≈ 2 source files plus tests, each verified by the coordinator from disk, none relayed

### C0 — `docs/plans/p7.3d.md` (the plan; gate G0)
Assumptions with confidence; the RESCO root as resolved on this machine; the seams of §0.6 restated from the code with
line numbers re-read; **how the per-intersection prompt is computed and where each of the 16 `rtg_first_i` is read
back**; the paired-CI helper named (`offline.offline_baselines.paired_comparison`, as P7.3a); the pedigree gate for the
grid4x4 held-out draws — **name the committed artifact holding per-draw grid4x4 values for a deterministic arm on
1000–1099, or state that none exists and that the gate is P7.3b §3.2's five-draw bit-for-bit re-collection instead**
(the coordinator did not resolve this from disk; you do); the stage design (§3 C6); questions for the coordinator.

### C1 — `offline/parity.py` + `offline/materialise_draws.py`: parity per scenario, and the grid4x4 draws (gate G1)
- `parity.py`: a `ParityScenario` (stem · CityFlow `flow.json` · source net path, resolvable to a file **outside** the repo
  through `RLTRAFFIC_GRID4X4_RESCO` · route source, for grid4x4 the zip member read **in place** as A15(g) did). The vType
  attributes are **DERIVED from the CityFlow flow block** through `read_cityflow_vehicle_block` and refused on any
  `flow_json_disagreements` — so grid4x4's eight registered values come out of the data, and a scenario whose flow block
  disagrees with the contract is a refusal, not a warning. The hz1x1 stem stays the default of every existing entry point,
  so every current call is unchanged (T-regress). `PARITY_CONTRACT_VERSION` stays `1.0` if and only if the hz1x1 output is
  byte-identical; otherwise bump it and say why.
- `materialise_draws.py`: the parity path takes the scenario key from the draw tree (it is the sim-config stem, `:27`),
  never a literal; render **CityFlow parents for grid4x4 draws 201–300** (`--draws-range 201 301`, half-open) with the
  same randomiser and seed rule the 1000–1099 parents used — state the rule from `provenance.json`, do not infer it — then
  parity for **201–300 and 1000–1099**: `routes.rou.xml` with `cf_parity` bound on **every** vehicle, `noteleport.sumocfg`
  with `<time-to-teleport value="-1"/>` and `end` above the horizon, `provenance.json` in P7.2a's shape
  (`materialised-draw-parity/1.0`: `demand_audit`, `files`, `net`, `parent`, `vtype_attributes`, `sumocfg`) plus the RESCO
  member sha256. **Nothing under an existing draw directory is overwritten**; the tool refuses if a `parity/` exists.
- **Gate G1 — A15(g)'s condition, decided by the artifact.** CAP(E) re-run against the rendered parity files on all 200
  draws (`offline/conversion_audit.py::audit_pair`, E's `(depart, route)` multiset and order per draw), plus the vType
  binding report (bound = vehicles on every draw). Written to `docs/data/p7_3d_cap_e.json`, committed. **200/200 exact →
  the condition is MET and the task continues. Any failure → STOP, `BLOCKED`, the packet reports which draw and why, and
  C3 stays a single-intersection study — a registered outcome (A15(g)), not a failure of this task.** The coordinator
  verifies the artifact from disk and appends Amendment A; you merge `main` and read it. **Until it appears, build C2 —
  it does not depend on G1.**

### C2 — `agent/DTAgent.py` + tests: `DEFERRED` 78, the per-intersection target (gate: coordinator's mutations)
`target_rtg: float | Mapping[str, float]` and `rtg_scale` likewise, resolved per intersection id with a scalar
broadcasting to every id — `SpatialDTAgent._resolve_per_node`'s shape (`:479`), not a new one. `current_rtg()` returns
per-id values; `state_dict`/`load` carry the mapping and accept the old scalar payload unchanged. **Tests:** every
existing checkpoint under `output/p4_dt/`, `output/p4_7/checkpoints/`, `output/p5_2/checkpoints/*dt_nomix*` loads,
saves, and the saved bytes equal the original's under `==` (round-trip, all files, count stated); a two-intersection fake
env receives two **different** targets and the RTG series differ accordingly; a mapping missing an id refuses by name.
**The coordinator re-runs two mutations before C3 starts:** the mapping ignored and the scalar used (the fake-env test
must die), and the round-trip compared with `allclose` instead of `==` (hygiene must reject it).

### C3a — `offline/aligned_env.py` + `offline/transfer_calibration.py`: the chain takes a scenario
`DECLARED_SCENARIO` and `SCENARIO_KEY` become parameters with the hz1x1 values as defaults; grid4x4's alignment comes from
`alignment_for_scenario`'s identity branch (16 ↔ 16 phases, action count 8 from the env); `run_sumo_probe` records **per
intersection** `local_return` and `rtg_scale` inputs (16 series per draw) with the two-route equality per intersection;
A17(e)'s Rule B **per intersection i**: `S = mean` over intersection i's 100 probe returns, `R_best_source,i` from the
checkpoint's `stats["rtg"][scenario][i]`, `k = 100` registered, `k ∈ {5, 20}` recorded and not evaluated — **16 targets
and 16 scales**, written to `docs/data/p7_3d_calibration.json` with the in-support position per intersection.
**T-regress (load-bearing):** `docs/data/p7_2b_calibration.json` regenerates byte-identically through the defaults from its
existing chunks (`report` over `output/p7_2b/`), and `assert_logged_corpus_matches_probe` still passes 100/100 on hz1x1.

### C3b — `offline/transfer_curve.py` + `offline/collect.py`: the cell on 16 intersections
`SCENARIO_KEY` a parameter; **`dt_choose` records 16 RTG series and 16 reward series** (per intersection id, the shift-by-one
convention of Amendment D1 unchanged and stated), `rtg_first_i == target_i` refused **per intersection**, in-support counts
per intersection; `cell_chunk_name` gains the scenario (P7.3b's stage-identity lesson: two campaigns that share a naming
scheme will eventually share a directory); anchors declared per scenario; `collect.py`'s SUMO door resolves grid4x4's
parity config through the same key (not needed for a corpus here, but the door must not be hz1x1-only — one test).
**T-regress:** `docs/data/p7_3a_zero_shot.json` and `docs/data/p7_3b_anchor.json` regenerate byte-identically via `report`
over the existing work directories through the defaults. **T-obs (SUMO, load-bearing for §0.9):** one grid4x4 fixed-time
episode on draw 1000 through `aligned_observer_env_for_draw` reconstructs: 16 intersections in `info`, 3,600
observations, 360 decisions, `n_teleports 0`, `{"cf_parity"}`, `"-1"`, `e_sumo` finite, the halting cross-check ON with
its lane-second count equal to 240 × 3,600 — **the first grid4x4 SUMO episode this project has ever run, and it is a test.**

### C4 — the grid4x4 reference cells (a P7.1-style freeze, small)
`fixedtime` and `maxpressure` on draws 1000, 1001, 1002 under parity and `"-1"`, both ATT definitions, recorded to
`docs/data/p7_3d_reference_cells.json` with their commit. **The campaign's chunks for those six cells must reproduce them
bit-for-bit** (A9's *the instrument regenerates* — P7.3a had P7.1's frozen values to check against; grid4x4 has none until
this file exists). `report` refuses if any of the six differs.

### C5 — the probe on grid4x4 201–300 (tmux, author-started; gate G2 first)
**Gate G2 — the one-cell measurement, before any schedule is written:** one DT cell (`seed101`, draw 1000, the naive
prompt) and one fixed-time cell on grid4x4 SUMO, in a single worker: wall and in-process seconds, **peak RSS of the
process**, GPU memory, the canary beside them (`run_probe` on `cityflow1x1` draw 0, reference ≈ 0.9 s, both halves). Then
the same DT cell under a pool of `W ∈ {4, 8, 12}` workers for RSS scaling. **Every range in the costing note is replaced
by these numbers; `WORKERS` and the driver's RSS refusal budget are set from them; the schedule in the driver's header
cites them with their date and canary.** Then the probe: 100 MaxPressure episodes on 201–300, `reset(seed=1000)`, one per
draw, per-intersection returns by two routes — in a tmux pane the author starts (channel (a)); the coordinator reads
`docs/data/p7_3d_calibration.json` from disk and appends Amendment B. **You proceed to C6 meanwhile.**

### C6 — `offline/campaigns/p7_3d_grid4x4.sh` — the driver, P7.3a's shape, two declared stages
Order: interpreter → import → lock → group leader → SigIgn → stage argument → inputs (200 parity configs, the five
checkpoints at their §0.7 digests, `p7_3d_calibration.json`, `p7_3d_reference_cells.json`, `p7_3d_cap_e.json`) → dirty
tree → canary (both halves) → **RSS budget check** → trap → token → work. **Stage `confirmatory`:** `b_mean_k100` × 5 seeds ×
100 draws + `fixedtime` + `maxpressure` = **700 cells**, with the halting cross-check ON for every cell of draw 1000 (C2's
convention: 7 cells). **Stage `rest`:** `naive` × 5 × 100 + `random` × 5 policy seeds × 100 = **1,000 cells**. Each stage
consumes its own token (two channel-(a) messages, by design: P7.3a's stage 2 refused a start at canary 6.42 s and that
refusal is worth a token). `report --stage confirmatory` writes the confirmatory artifact; `report` over both writes
`docs/data/p7_3d_grid4x4.json`. Manifest last, atomic, re-verified. `tee -a` from the first line into
`output/p7_3d_runs/campaign_capture.txt`.

### C7 — the packet and the artifacts (Amendment B's shape, `BRIEF_38`)
Artifacts copied **by hand** into the implementer's tree with digests measured and stated; the whole suite run after the
copy; `docs/returns/P7.3d.md`; then **"P7.3d done"** — channel (d), the one message the author carries.

---

## 4. Tests — first, red for their own reasons, each named mutation executed and pasted

SUMO-, checkpoint- and RESCO-gated tests carry `skipif` predicates that **name the artifact they consume**; the RESCO gate
is `RLTRAFFIC_GRID4X4_RESCO` unset → skip with the variable named. At most **six** grid4x4 SUMO episodes in the whole
suite (they are 2–6× hz1x1's). `scripts/check_test_hygiene.sh` on every test file; the repo-wide English sweep.

- **T-regress (load-bearing, three parts, CityFlow + committed chunks).** (a) hz1x1 parity for draws 5 and 1000
  regenerated through `ParityScenario`'s default equals P7.2a's committed provenance digests (`noteleport.sumocfg`
  `c177e962…`, `routes.rou.xml` `c2c5eea2…` on draw 1000) under `==`; (b) `p7_2b_calibration.json`, `p7_3a_zero_shot.json`,
  `p7_3b_anchor.json` regenerate byte-identically via `report` over their existing work directories; (c) every hz1x1
  `SCENARIO_KEY`/`DECLARED_SCENARIO` default resolves to today's value. *Mutations:* default stem changed → (a) dies; a
  per-intersection branch taken on a one-intersection env → (b) dies.
- **T-parity-values.** grid4x4's derived vType equals A15(g)'s eight values **exactly**, read from the rendered file and not
  from a constant; a flow block edited by one field refuses naming the field. *Mutation:* `tau` read from the wrong key →
  dies.
- **T-cap-e (RESCO-gated).** CAP(E) on one rendered draw is exact; a route with one vehicle's depart shifted is refused
  with the draw named. *Mutation:* compare counts only → the shifted depart survives → the test must die.
- **T-78 (checkpoints-gated).** Round-trip byte-equality over every DT checkpoint on disk (count stated); the two-target
  fake env; the missing-id refusal. *Mutations:* §3 C2's two.
- **T-16 (SUMO, grid4x4-gated) — the cell's mechanics.** 20 decisions of `seed101` on draw 1000: 16 RTG series with
  `rtg_first_i == target_i` for every i, 16 reward series, `rtg[t] − rtg[t−1] == −reward[t−1]` per intersection under
  D1's rule, actions in range under the env's 8, teleports 0. *Mutation:* one target swapped between two intersections →
  the per-intersection refusal dies.
- **T-obs (SUMO, grid4x4-gated).** §3 C3b's first episode. *Mutation:* halting check declared ON but the lane count read
  from hz1x1's 16 → the 240 × 3,600 assertion dies.
- **T-rho.** Fixed-time → 0.0, MaxPressure → 1.0 exactly on synthetic 16-intersection cells; pairing per draw; a chunk
  whose `scenario` is `cityflow1x1` offered to a grid4x4 report is refused (the stage-identity lesson, one level up).
- **T-report.** Refusals precede every write including the last; a missing reference cell, a reference cell that differs,
  a chunk at a checkpoint digest not in §0.7, a work directory without `canary.json` — each refused; regeneration
  byte-identical at the recording commit. *Mutation:* `_write_json` above the last refusal → dies.
- **T-driver.** Comment-free text assertions: canary before token, `record-canary` after, RSS check before the trap,
  `WORKERS` from a variable set by G2's file, two stages with their own tokens, `tee -a`, `set -euo pipefail`.

---

## 5. Gates, in order — who runs each, what it checks, what stops the task, and how you learn its result

| # | Gate | Runs it | Checks | Stops the task if | You learn it by |
|---|---|---|---|---|---|
| G0 | Plan gate | coordinator, from `docs/plans/p7.3d.md` on the branch | assumptions, seams, the prompt route, the pedigree gate's answer | a load-bearing assumption is wrong | Amendment A on `main` — `git merge main` |
| G1 | A15(g)'s condition | you (C1), coordinator verifies `p7_3d_cap_e.json` | CAP(E) 200/200 exact on the rendered parity files, vType bound on every vehicle | any draw fails → `BLOCKED`; C3 stays single-intersection (registered) | Amendment B |
| G2 | One-cell measurement | you (C5), in tmux | wall, RSS, GPU, canary; `W ∈ {4, 8, 12}` scaling | RSS exceeds host budget at every W → `BLOCKED` with the numbers | you have the numbers; write them into the driver header and the plan |
| G3 | **A20 registered** | coordinator, on the author's ruling — **channel (b)** | subject, arms, per-intersection rule, status | — (mechanics continue; **no DT cell's number is produced before the tag exists**) | the tag `v2.0-prereg-a20` on `main` |
| G4 | Pre-flight | a reviewer (≤ 15 min, findings file), coordinator rules | destruction and resume paths, the two tokens, the RSS refusal, the canary's two halves | a destruction path exists | Amendment C |
| G5 | Token, stage 1 | **author — channel (a)** | — | — | the token file |
| G6 | Stage-1 read | coordinator, from disk, capture first | 700/700, reference cells reproduced, 0 teleports, per-intersection `rtg_first` | a reference cell differs → channel (c) | Amendment D (or none: stage 2's token is the signal) |
| G7 | Token, stage 2 | **author — channel (a)** | — | — | the token file |
| G8 | Packet | you (C7) → **"P7.3d done", channel (d)** | Amendment B's list, §7 below | — | — |
| G9 | Merge reviews | coordinator spawns **two** (code + mutations; every number recomputed from raw chunks); the author may compress to one | — | a blocker | the merge, §6's box ticked |

---

## 6. Definition of Done
- [ ] C0–C7 delivered in order, each its own commit with named paths; no frozen file touched; no new dependency; no new
      absolute path; hz1x1 byte-identical throughout (T-regress green at every commit that touches shared code).
- [ ] `p7_3d_cap_e.json` 200/200 (or the task ended at G1 with the reason in the packet).
- [ ] G2's numbers in the driver header, the plan and the packet, with canary and date.
- [ ] Every test of §4 red first then green; every named mutation executed and pasted; hygiene and the English sweep run.
- [ ] The campaign complete under two tokens, capture from the first line, both canaries on disk, the six reference cells
      reproduced bit-for-bit, `docs/data/p7_3d_grid4x4.json` and the confirmatory artifact committed and regenerating
      byte-identically.
- [ ] `docs/returns/P7.3d.md` per §7, then "P7.3d done".

## 7. Return Packet
`docs/returns/TEMPLATE.md`, plus: G1's verdict with the per-draw counts; G2's measurements with canary and date, and the
W-scaling table; the per-intersection targets and scales (16 + 16) with in-support positions; ρ under both definitions for
the registered arm with CIs and per-seed means, the paired ATT against both anchors; **H3's clause 1 as the registered
inequality on this scenario, reported not interpreted; clause 2 as an inequality; clause 3 void (A19)**; the naive-arm
contrast, exploratory; the driver captures of every run including any refused start; where the driver ran and at what
commit; the amendments written against, by letter; the AI-assistance record's four lines; what the paper's C3 section
will assume about this point; and — stated in one sentence — that no A17(f) corpus gate applies here and why.

---

## 8. Registration — A20, proposed by the coordinator to the author, tagged before any grid4x4 SUMO number exists (gate G3)

✅ **REGISTERED 2026-09-19, approved by the author *as written*: `PREREGISTRATION.md` A20, tag `v2.0-prereg-a20` →
`0f3526b`, both refs on the remote, the sha256 chain verified. Gate G3 is CLEARED before this task starts; the bullets
below are now declared constants, and the registered row — not this section — is the authority if the two ever differ.**
The immediately-before check the row records: 0 parity configurations under `scenarios/draws/cityflow_grid4x4/`, no
`p7_3d` artifact, work directory, worktree or branch, no grid4x4 SUMO simulation ever run.

The implementer treats these as declared constants; if a later ruling changes one, the coordinator appends an
amendment naming the constant, and the change is one line.

- **Subject:** `mappo1000_dt_nomix_h4`, five seeds, the five checkpoints of §0.7 by digest — P5.2's headline non-spatial DT
  at the best-data tier, the only grid4x4 DT with a committed declaration and five seeds at one architecture. **No second
  subject:** the matched-architecture checkpoint does not exist (§0.7); a data-quality contrast is carried by hz1x1
  (`mappo1000` vs `mix50`) and by A19, and would be confounded by head count here.
- **Arms:** `b_mean_k100` (registered, A17(e) per intersection, `S = mean`, `k = 100`) and `naive` (the P5.2 per-intersection
  in-domain prompt read from the checkpoint). `b_max_k100` and `a_q1.0` not evaluated, by declaration.
- **Anchors:** `fixedtime` and `maxpressure` one episode per draw; `random` five policy seeds per draw; A18(c)'s seed rule
  (`reset(seed=1000)`, one episode per draw on a fresh env, both seeds recorded). Held-out draws 1000–1099; probe band
  201–300 rendered for this scenario.
- **Metric:** A15's pair, `E_sumo` primary and `att_env` co-reported; ρ per draw against the anchors of the same draw;
  the denominator diagnostic beside the `att_env` ρ.
- **Status:** H3's clause 1 (ρ > 0 against fixed-time) is CONFIRMATORY on this scenario, as H3's test row (*per paired
  scenario*) already provides; clause 2 is reported as the inequality it is; clause 3 is void (A19). The naive arm, the
  `random` anchor and every per-intersection breakdown are exploratory.
- **Outcome rows (§10):** this point is reported beside hz1x1's under the A19 sentence; if G1 fails, *C3 is a
  single-intersection study over four demand days and the paper says so* (A15(g), unchanged).
- **Results already seen:** every hz1x1 number (P7.3a, P7.3b); P5.2's CityFlow grid4x4 results for this subject
  (`dt_nomix_h4` leads the `mappo1000` tier in-domain); **no grid4x4 SUMO number of any kind** (A15(iii) still holds —
  §0.10).

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
- **T-pilot (added 2026-09-19 from P7.3b's merge review, `DEFERRED` 86).** A test that **CALLS `run_pilot`** — or the
  transcript-building function once it is factored out — on a fixture whose cells include one with `seed: None`
  (an anchor cell), and asserts the transcript's `seeds` and `n_cells_without_a_training_seed`. P7.3b's
  `test_the_pilot_transcript_survives_a_cell_that_has_no_training_seed` re-implements the guarded expression inside
  its own body and pins nothing: with the guard at `transfer_curve.py:2624` removed it stays green (reviewer's M2,
  re-run by the coordinator). **Replace that test with the real form here; do not add a second one beside it.**
  *Mutation:* remove the `is not None` guard → this test must die. ⚠️ **Two harness traps when you run mutations on
  driver tests, both hit by the coordinator on 2026-09-19:** a mutant worktree with an uncommitted edit is a DIRTY
  tree, and the driver's dirty-tree refusal fires before the refusal the test expects; and a shell whose command line
  contains the literal `offline.transfer_curve` matches the driver's liveness `pgrep` and produces exit 3. Commit the
  mutant in the throwaway worktree, and invoke pytest from a script file whose text does not carry the pattern.

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

---

# ✅ AMENDMENT A — 2026-09-19, gate G0: PLAN APPROVED (`docs/plans/p7.3d.md` @ `52e61e2`) with rulings on Q1–Q8, four corrections to THIS brief that the implementer's findings force, and the sequencing after P7.3b's CI fix

**Read first, whole, then act.** Every ruling below was made by the coordinator; the one item that could have changed a
registration (Q1's second half) does not, and the reasoning is given. Nothing here is a relay from the author.

## A0 — Sequencing: P7.3b's CI fix merges FIRST (`BRIEF_38` Amendment C, one test, one commit), then this task resumes
`main` is red for a real P7.3b test failure (a driver test assuming this laptop's interpreter path), not for the ceiling.
That fix touches `tests/test_transfer_curve.py`, which C3b also touches. **Order for the implementer session that resumes
this task: do Amendment C in its own worktree first; say "P7.3b CI fix done"; then `git merge --no-edit main` here and
continue with C1.** C1 touches neither file and may start at once.

## A1 — Q1 / F1: CONFIRMED FROM THE FILE BY THE COORDINATOR; the subject runs through `SpatialDTAgent`; C2 LEAVES THIS TASK; no re-registration
`torch.load` of `grid4x4_mappo1000_dt_nomix_h4_seed101.pt`: `format_version spatial-dt-checkpoint/1.0`, `config.spatial_mixing
False`, `target_rtg` and `rtg_scale` mappings with 16 entries — the coordinator's own read, matching the plan's. **§0.7–0.8,
§2 and §3 C2 of this brief were written from the costing note's sentence *"the non-spatial DT … is the `DTAgent` path"*
without opening the checkpoint: the coordinator's error, of the recurring class, logged in the Decisions Log.**
Rulings: **(a)** the cell loads the subject through `SpatialDTAgent.from_checkpoint` and applies the 16 targets after load
through a guarded `spatial_agent_with_targets(...)` exactly as the plan proposes (declared budget asserted; `rtg_scale` equal
to the payload's per id under `==`; `current_rtg()` equal to the targets per id under `==`; a missing id refused by name).
**(b) C2 (`DTAgent` per-intersection target, `DEFERRED` 78) leaves P7.3d's scope**; `DEFERRED` 78 stays parked for a future
`dt-checkpoint/1.0` multi-intersection subject, and its row is annotated. **(c) A20 needs no registration note.** A20(a)
registers the subject by five digests and excludes *"spatial DT"* — which is the `dt_spatial_h4` **arm** (spatial mixing ON),
the only other 4-head grid4x4 DT. The registered `dt_nomix_h4` is P5.2's *identity-graph control* (`docs/plans/p5.2.md:97`:
*"`dt_nomix` is the identity-graph model … no information crosses nodes"*), stored in the spatial checkpoint format and
evaluated by P5.x through `SpatialDTAgent.from_checkpoint` — the plan cites `tier_sweep.py:2138`, `admission_probe.py:866`.
The class that reads the file is implementation; the subject, the arms and every registered quantity are unchanged. The
sentence *"a different agent path"* in A20(a) is an inaccurate gloss and is corrected in the Decisions Log and in this brief;
**the packet and the paper describe the model as *the spatial architecture with the identity mask (no cross-intersection
attention), P5.2's non-mixing control***, and never as "a `DTAgent`". If the author wants this as a dated note in §12, the
coordinator files it; it changes no registered value.

## A2 — Q2 / F2: ACCEPTED — payload-level exact equality
Byte-equality of a load→save round trip is unsatisfiable on unmodified code (three files measured, both classes; the
trainers wrote them with their own `torch.save`). T-78 shrinks with C2's removal to **the spatial loader's guard**: every
tensor under `torch.equal`, every other key under `==`, `float` types asserted, `intersection_ids` compared as
recorded-or-empty and said so. The hygiene mutation (`allclose` for `==`) stands.

## A3 — Q3 / F3: ACCEPTED — T-regress (b) with EXACTLY the two named substitutions
`code_changed_since → []` and `_git_provenance → the committed artifact's recorded pair`, both existing patterns in the test
file (`:1412`, `:2464-2489`), over the real work directories into `tmp_path`, then bytes `==` the committed file for
`p7_2b_calibration.json`, `p7_3a_zero_shot.json`, `p7_3b_anchor.json`. **Nothing else may be substituted**, and the test's
docstring names both seams and why (J1(c) and the write-time commit; `DEFERRED` 84). The brief's mutation stands.

## A4 — Q4 / F4: CONFIRMED — the CityFlow MaxPressure probe on grid4x4 201–300 is REQUIRED, and `R_best_source,i = payload["target_rtg"][i]`
Rule B (A17(a)) has a source-domain denominator; on hz1x1 it was `p4_3_probe.json`; on grid4x4 it does not exist. **C5
gains, BEFORE the SUMO probe: `run_probe` on the 100 grid4x4 CityFlow parents 201–300, one MaxPressure episode per draw,
`reset(seed=1000)`, P4.3's settings, per-intersection returns by two routes under `==`**, recorded in
`p7_3d_calibration.json` beside the SUMO half. It is ≈ 100 short CityFlow episodes and may run in-session with the canary
logged (not a *long* run in `CLAUDE.md` §5's sense). And **`R_best_source,i` is `payload["target_rtg"][i]`** — the repo's
established reading (`transfer_calibration.py:561,586`) and the same quantity P5.2's prompt rule defines (*max episode
return in THIS INTERSECTION's training streams*); the `stats["rtg"]` block is per-window and bounds the support only.
A17(e)'s wording pointed at the wrong block; the calibration artifact states the field it read and why. `naive_i` is the
same field.

## A5 — Q5: BUMP to `materialised-draw-parity/1.1`
`parent.routes_sha256` becoming `null` is a shape change a 1.0 reader does not expect; additive keys alone would not force
it, a nulled field does. Readers accept both versions; the docstring states both shapes and the alignment convention
(`CLAUDE.md` §3).

## A6 — Q6: YES — cwd = the main tree, `main`'s unmodified `materialise()`, `--draws-range 201 301`
The existing 106 grid4x4 parents embed the main tree's absolute `dir`; the new 100 must match. The implementer's finding that
P7.3b's hz1x1 301–400 parents embed the `rltraffic-p73b` worktree path is parked as **`DEFERRED` 87** with its rule:
*every CityFlow parent is rendered with cwd = the main tree*. Writing 100 gitignored directories into the main tree's
`scenarios/draws/cityflow_grid4x4/` is the intended location (it is where P7.2a's tool writes).

## A7 — Q7: the §5 gates table is AUTHORITATIVE; §3's letters were the coordinator's inconsistency
G0 → this Amendment A; G1 → B; G4 → C; G6 → D (or none). §3 C1's *"Amendment A"* and C5's *"Amendment B"* read B and C
respectively; corrected here rather than by editing the issued text.

## A8 — Q8: CONFIRMED — a relative reference to the RESCO net from the draws tree, its sha256 in provenance, the driver checking both
Exactly hz1x1's pattern (`net.reference = ../../../../hangzhou_1x1_…`). *No RESCO file is copied into the tree* is satisfied;
the temp-scratch extraction of the route template, deleted with the scratch, is fine. The driver's precondition resolves the
reference and compares the net's sha256 to `8d192de4…` before the token.

## A9 — Accepted as designed, no ruling needed: the three unlisted seams (`_sumo_pairing`, `_validate_parent_for_parity`'s second legal shape selected by scenario, `render_parity_rou_text`'s zero-vType insertion), the pedigree gate (100/100 `flow.json` digests against `p8_4a_admission.json` + six CityFlow episodes against `p8_4b_g0_reference.json`), the grid4x4 artifact's own format version, the stage design, and the eight assumptions with their confidences. Assumption 4 (state width 40) is what T-obs exists to test — run it before anything is built on it.

**Then: C1.** The next thing on `main` for this task will be Amendment B, written after the coordinator reads
`docs/data/p7_3d_cap_e.json` from disk.

---

# ✅ AMENDMENT A.1 — 2026-09-19, mid-C1, on the implementer's stop: the old grid4x4 no-pairing test gains ONE `delenv` line, assertion untouched; C1's other readings confirmed

**The stop was correct.** `tests/test_materialise_parity.py::test_a_scenario_without_a_sumo_pairing_is_refused_with_its_reason`
asserts that grid4x4 parity is refused naming `grid4x4.rou.xml`; C1 makes grid4x4 parity SUCCEED whenever
`RLTRAFFIC_GRID4X4_RESCO` is set. Measured by the implementer: with the variable unset, 101 passed / 10 skipped / 0 failed
across the four parity files; with it set, 110 / 0 / **1** — that test. The implementer edited nothing and asked. That is
`CLAUDE.md` §0's rule working.

## A.1-1 — Ruling: Option 1. Add `monkeypatch.delenv("RLTRAFFIC_GRID4X4_RESCO", raising=False)` to that test; change nothing else in it
The test's contract — *a scenario without a SUMO pairing is refused, and the refusal names the missing file* — is unchanged
and still true. What changed, by design and by registration (A15(g), A20(f)), is that grid4x4 **has** a pairing when the
RESCO root is present. The `delenv` line states the precondition the test always had implicitly (no candidates root), the
assertion stays byte-identical, and the new
`test_grid4x4_parity_is_refused_naming_the_variable_when_it_is_unset` carries the same contract with the variable
explicitly unset plus the two things the old test never checked (the refusal names the variable; nothing is written).
**Not chosen, and why:** deleting the old test drops the count (§5 of the coordinator's doctrine: a drop is a signal);
leaving it red whenever the variable is exported makes one environment's suite permanently noisy, which is how a real
failure hides. **Disclose the change in full in the packet** — the `BRIEF_37` D3 precedent — with this amendment cited,
and add one sentence to the test's docstring naming the precondition and A.1.

## A.1-2 — The hz1x1 rendering path: the fix is right; the KEY of the branch is a note, not a round
Calling `_render_bound_routes(source_text, draw_id=draw_id)` **exactly as before** on the hz1x1 path — so the existing
test that substitutes that seam with the two-argument signature is untouched — is the correct repair. **Note for C1's
commit, at the implementer's discretion and not a blocker:** the branch is currently keyed on
`vtype is None or dict(vtype) == parity.parity_vtype_attributes()` — a VALUE equality. Key it on the scenario instead
(`scenario.key` being the hz1x1 default), because a branch that depends on two tables happening to be equal is a
coincidence-dependent seam (§7, *a pin on a function does not pin its caller*), and a future scenario whose derived table
equalled hz1x1's would silently take the hz1x1 path. Same behaviour today; a clearer contract tomorrow.

## A.1-3 — A5's reading CONFIRMED: `materialised-draw-parity/1.1` for the grid4x4-shaped record ONLY
hz1x1 records keep writing `1.0`, byte-identical, so the 306 existing hz1x1 parity directories (the implementer's
corrected count; 406 was a slip) still read as `kept`. Readers accept both versions; the docstring states both shapes.

## A.1-4 — Everything else in the report is accepted as designed
The three hz1x1 regression tests green before and after; the derived vType refused on disagreement with the registered
table; RESCO files read in place and pinned by digest; the CAP(E) report and its CLI mode; 22 new tests red first.
**Continue C1 to its commit:** the named mutations, hygiene and English, the 201–300 parents (cwd = main tree, A6), the
200 parity renders, `docs/data/p7_3d_cap_e.json`, then the C1 commit. G1's verdict will be **Amendment B**, as §5 says.

**Process note, for the record and not for the implementer:** a stop of this kind reaches the coordinator only through the
author, because the coordinator does not run unless spoken to. The author's part is one word (*"blocked"*) — never a
paste — after which the coordinator reads the worktree. Written into §7's relay rule as a clarification.

---

# ✅ AMENDMENT B — 2026-09-19, gate G1: A15(g)'s CONDITION IS MET under the coordinator's own route; C1 ACCEPTED at `723310e`; A21's scope applied to this brief (one stage, one review, no `collect.py` door); Q2–Q4 ruled

## B1 — G1 PASSED. The coordinator recomputed CAP(E) by its own route and read the artifact from disk
Stdlib regex over the rendered route text and `json` over the parents — nothing imported from `offline/` — over **all 200
draws** (201–300 and 1000–1099): the `(depart, route)` multiset **and order** equal on **200/200**; **265,477 vehicles, every
one `type="cf_parity"`**; the rendered `<vType>` attributes equal A15(g)'s eight on all 200; `time-to-teleport −1` in every
`.sumocfg`; the net reference resolves on every draw to RESCO's file at sha256 `8d192de4…`; **0 aggregate flow entries**
(every grid4x4 entry is a single insertion — P7.2a's trap does not arise here). `docs/data/p7_3d_cap_e.json` (`949015b3…`,
`git_commit 73bf637`, `git_dirty false`, `n_exact 200`, `condition_met true`, pedigree 100/100 against
`p8_4a_admission.json`) says the same. Regeneration: regeneration from a clean tree at `73bf637` NOT RUN by the coordinator (the command could not be reconstructed from the capture in-session) — it is the merge reviewer's item. **A15(g)'s condition is MET: grid4x4 is admitted to C3's
target set. The task continues.**

## B2 — A21 (`v2.1-prereg-a21` → `169c467`) applied to this brief; the registered row is the authority where they differ
- §2 / §8: arms are **`b_mean_k100` only**. `naive` and `random` are NOT evaluated on grid4x4 — A21(b)'s two scope
  sentences are the paper's: *the calibrated-versus-naive contrast stays hz1x1-only, where it was flat (−0.0021 / −0.0179);
  ρ_random is absent on this scenario (−1.300 / −3.604 on hz1x1)*.
- §3 C3b: **no `collect.py` SUMO door for grid4x4** — nothing is collected in this task.
- §3 C6 / §5: **ONE stage, `confirmatory`, 700 cells** (500 DT + 100 `fixedtime` + 100 `maxpressure`), **one token**; the
  halting cross-check ON for the 7 cells of draw 1000. **G7 (token, stage 2) is deleted from the gates table; G9 is ONE
  merge review** (mutations + every number recomputed by the reviewer's own route, the P7.3b shape).
- §4: T-rho's synthetic cells and T-report's refusals cover the two arms that exist; nothing is written for arms that do not.

## B3 — Q2–Q4
- **Q2 (+11 clean-clone skips):** nothing for the implementer. The ceiling moves at P7.3d's merge by the registered route —
  observed on the run that merge triggers, never predicted — and `re_measure_required_at` already names it.
- **Q3 (the six-episode half of the pedigree gate):** in **C4's commit**, beside the six reference cells — same purpose
  (A9's *the instrument regenerates*), same artifact family; `p8_4b_g0_reference.json`'s `behaviour@fixedtime` and
  `behaviour@maxpressure` on draws 1000–1002 reproduced bit-for-bit, or the task stops and says which draw.
- **Q4 (`run_worktree_module.sh`):** stays evidence-side. **C6's driver carries the cwd rule itself** — `cd $MAIN`,
  `PYTHONPATH=$WORK_TREE`, the RESCO root exported with the main-tree candidates dir as default, and an assertion that the
  worktree's module loaded — and T-driver asserts those lines. A launcher outside the repository is not a home for a rule.

## B4 — Deviations ACCEPTED as recorded, with one note
Three commits for C1 (code, fix, artifact) so every parity record carries a clean code commit; the re-render with
`--force` **before any cell exists** (both captures kept; one config digest `c27d31e8…` on all 200); the
`_report_heldout_thresholds` literals; the A.1 line applied exactly (the *removed* `def` is the same test re-signed for
`monkeypatch`; +25 test functions, none deleted). **Note, not a round:** an unregistered scenario key still falls back to
hangzhou's table inside `materialise_parity` while `scenario_for_key` and `cap_e_report` are strict; that is the
pre-P7.3d behaviour and it is refused on disagreement, but it is a coincidence-dependent path — make it strict the next
time that function is touched.

## B5 — Mutations: 19 runs, all KILLED (implementer's transcript); two re-run by the coordinator on COMMITTED mutants in a throwaway worktree, control 3/3 first
**M9c** (the full-table binding check dropped — a fully bound file with the wrong `tau` would pass G1): **KILLED**,
`test_cap_e_refuses_a_fully_bound_draw_whose_vtype_carries_the_wrong_tau`. **M14** (hangzhou's header moved — every
P7.3a/P7.3b config digest would change): **KILLED**, both named tests, the hz1x1 draw-1000 digest test among them.

## B6 — A precision to A21's *"no grid4x4 SUMO simulation of any kind"*, recorded because the quantifier is the project's recurring error
C1 started SUMO once on grid4x4: a **config load** of draw 1000 (`--end 1 --xml-validation always`), exit 0, an EMPTY
capture — no episode, no return, no ATT, no quantity of any kind. A21's substantive claim (no target-domain grid4x4
number existed when it was tagged) is unaffected; the row's wording was one word too wide and this paragraph is its
correction, also in the Decisions Log.

## B7 — Housekeeping before C3a
At the time of this ruling the implementer's worktree carried uncommitted paths: ` M offline/aligned_env.py;?? tests/test_aligned_env_grid4x4.py`. Commit or remove them before
any cell is rolled — J1(d): the driver refuses a dirty tree, one cell at a time, hours in.

**Then: C3a.** The next thing on `main` for this task is Amendment C after the pre-flight (G4).

---

# ✅ AMENDMENT B.1 — 2026-09-19, late, on the implementer's stop at C3a: the halting cross-check covers the MONITORED INCOMING lanes — 192 on grid4x4, 8 on hz1x1 — and the brief's 240 / 16 were the coordinator's error; no registered instrument changes; C3a's first half ACCEPTED at `b9ec455`

## B.1-1 — The ruling, from the artifacts
The registered instrument (A15's `reconstruct_sumo_episode`, Amendment C2's cross-check) iterates
`_monitored_incoming_lanes` (`offline/sumo_att_reference.py:849–851`, `:870`) — the lanes the observation reads — and the
committed record says so: **all 47 halting-checked cells of `docs/data/p7_3a_zero_shot.json` carry
`halting_n_lane_seconds = 28,800 = 8 × 3,600`** (re-read by the coordinator), hangzhou's eight incoming lanes, not its
sixteen. On grid4x4 that is 16 × 12 = **192 monitored incoming lanes → 691,200 lane-seconds**, which is what T-obs measured;
the other 48 lanes are boundary-outbound and enter no controlled intersection. **§3 C3b's *"240 × 3,600"* and §4 T-obs's
*"hz1x1's 16"* were TOTAL lane counts copied from A15(g)'s structural audit — the coordinator's error, of the recurring class
(a number written from a neighbouring artifact rather than from the instrument).** The implementer implemented to the repo,
left the instrument untouched, and asserted the coverage three ways (derived from `env.intersections`, the literal
`691_200`, and *not* `8 × 3,600`), with the mutation — the check restricted to eight incoming lanes — KILLED. **All of that
stands. Widening the check to 240 lanes would be a NEW instrument (the territory of `DEFERRED` 83), not a correction, and
is not wanted.** The brief's two sentences read as corrected here; the issued text is not edited.

## B.1-2 — C3a's first half ACCEPTED at `b9ec455`; T-obs is the first grid4x4 SUMO episode this project has run
Aligned state **40 wide on all 16 intersections for all 360 decisions** (plan assumption 4 holds); 0 teleports; `cf_parity`;
`"-1"`; the halting cross-check agreeing on every one of the 691,200 lane-seconds; the five no-simulator tests and the nine
hz1x1 aligned-env tests green. The implementer's own catch is recorded with credit: its first new assertion — by-id lane
pairing — was wrong as a fact on 128 of 192 lanes, corrected from the network files to the identity permutation; that is
the plan gate's assumption 4 doing its job one level down.

## B.1-3 — A grid4x4 SUMO NUMBER HAS NOW BEEN SEEN, and every later "results already seen" column names it
The first T-obs run failed on the brief's wrong count, and the failure's repr wrote `HaltingAgreement(…)` and **fixed-time
`e_sumo` ≈ 307.26 s on draw 1000** into `output/p7_3d_runs/c3a/t_obs_first_run.txt`. It is an anchor value on one draw, not
a subject's number, and it was seen **after** A20, A21 and A22 were tagged — their columns remain true as of their dates.
From this line on, any amendment touching P7.3d lists it as seen. The implementer disclosed it unprompted, which is the
behaviour the rule exists for.

## B.1-4 — Housekeeping done by the coordinator, per the standing rule
`task/p7.3d-grid4x4` had no upstream after nine commits; the coordinator pushed it with `-u` (2026-09-16 rule; remote tip
`d24f535`). The implementer keeps not pushing.

**Then: the rest of C3a** — `transfer_calibration` per scenario, the two probes recorded per intersection (CityFlow first,
A4), the `p7_2b_calibration.json` regression through A3's two substitutions — and C3b. The next thing on `main` for this task
is Amendment C after the pre-flight (G4).

---

# ✅ AMENDMENT B.2 — 2026-09-20, before G2 runs: G2's DT cell is a TIMING cell, FENCED, on a draw OUTSIDE the held-out pool; it is not an evaluation of the `naive` arm A21 excludes; and the whole suite runs at the branch tip before C5's commit

## B.2-1 — The clause §3 C5 carried from before A21, caught by the author's reviewer before the run
§3 C5 says G2's DT cell uses *"the naive prompt"* on **draw 1000**. Amendment B applied A21 to C6's stages, the review count and
the `collect.py` door and **did not touch C5**, so the clause rode through. The engineering constraint stands: `b_mean_k100` does not
exist at G2 time — it comes out of the SUMO probe, which runs after G2 — so the checkpoint's own in-domain prompt is the only one a
timing cell can run under. **What changes is where it runs and how it is recorded**, so that no number under an arm A21 excludes can
reach a capture on a held-out draw.

## B.2-2 — Ruling
1. **G2's DT cell runs on draw 5** — the smoke draw, outside the held-out pool 1000–1099 and outside the probe band 201–300 — exactly
   as P7.3a's and P7.3b's pilots did (`PILOT_DRAW`; `run_pilot` writes every outcome under `fenced_do_not_report`). G2's fixed-time
   cell and the W ∈ {4, 8, 12} scaling cells run on draw 5 too. **Draw 5's parity files must exist for grid4x4** — C1 rendered 201–300
   and 1000–1099 only — so G2's script first renders draw 5's `parity/` with the same tool and records its provenance; that is one
   more directory under the gitignored draws tree and nothing else.
2. **The G2 script measures TIME and MEMORY and prints NO outcome:** wall and in-process seconds, peak RSS, GPU memory, the canary
   line (both halves). It does not print, log or store ATT, return, `e_sumo`, RTG series or actions; the cell's chunk, if one is
   written at all, goes under a `fenced_do_not_report` key in a `g2/` directory that `report` never reads. A test asserts the
   script's text contains none of `att_`, `e_sumo`, `episode_reward` as printed fields.
3. **The seen ledger.** Because a fenced draw-5 timing cell prints no quantity, nothing is added to the *results already seen*
   record by G2. If, despite (2), any outcome value reaches a capture — a failure repr is the known route (B.1-3) — the implementer
   reports it in the packet by value, draw and arm, and the coordinator adds it to the next amendment's *seen* column, as B.1-3 did
   for `e_sumo` 307.26.
4. **C5's SUMO probe** (100 MaxPressure episodes on 201–300, per intersection) runs after G2, in tmux, as before. It produces
   probe returns — anchor-side quantities, not arm evaluations — and they are the Rule B inputs A20/A21 require.

## B.2-3 — The suite gap, owned here
The implementer stated twice that the whole suite has not run since `22b1b22`'s content; seven commits have landed since, including
`aa84e39`, which extracts `run_probe`'s rollout loop into a helper shared with hz1x1's registered P4.3 path — exactly the change where
collateral is real, and only targeted files ran after it. **Before C5's commit: the whole suite at the branch tip, from the worktree,
with `RLTRAFFIC_GRID4X4_RESCO` set, the real tail pasted into the packet, compared against the last measured run (2,148 / 105 at
`22b1b22`'s content) with the delta accounted for.** A failure is a finding; nothing is edited to make it pass.

## B.2-3b — Draw 5's demand is comparable to the campaign's, measured, and the G2 record carries the comparison
The author's reviewer asked whether a schedule set on draw 5 transfers to draws 1000–1099, since SUMO wall time scales with
vehicle count. **Measured from `flow.json` (coordinator, 2026-09-20):** draw 5 has **1,335** vehicles; the held-out band's mean is
**1,327.6** (sd 12.3, range 1,298–1,358); the probe band's 1,327.2 (sd 11.2). Draw 5 sits **+0.56 %** above the held-out mean,
z = +0.60 — inside the band's own spread. (Draw 0, the nominal source flow, is 1,473 and is NOT representative; G2 must not use
it.) hz1x1's draw 5 stood at +0.83 % of its band, which is what P7.3a's A8 accepted. **Required of the G2 record:** one line stating
draw 5's vehicle count against the 1000–1099 mean and sd, so the schedule the driver header quotes carries its demand basis beside
its canary. RSS is expected to be insensitive to a 0.6 % demand difference; the schedule estimate inherits it and says so.

## B.2-4 — What B.2 does NOT change
A21's scope; the SUMO probe; C6's single stage; the six reference cells (C4, draws 1000–1002 — those are the registered
instrument-regeneration anchors, not timing cells, and stay where they are).

---

# ✅ AMENDMENT B.3 — 2026-09-21, BEFORE C6 is built, on the author's reviewer's four points: the driver's documented invocation must pass the driver's own guard; `-P` on every interpreter call; the schedule quoted as a RANGE; the probe-ratio finding's status

## B.3-1 — Every P7.3d driver's header documents an invocation its own guard refuses — MEASURED, and C6 fixes the class
`p7_3d_g2.sh:8` and `p7_3d_probe.sh:8` give the usage as `tmux new -s NAME 'bash <script> 2>&1 | tee -a <capture>'`; lines 90–91 / 109–110
refuse unless the script leads its own process group. **The coordinator measured the two forms on this machine (2026-09-21):** under
`tmux new -s NAME '<cmd>'` the pane runs the command through a non-interactive shell with job control off, and `bash <script>` reports
`pgid ≠ pid` — **NOT a leader; the guard fires** (the G2 capture's two refusals are exactly this; the author then typed the same line at a
pane's prompt and it ran). The dry-runs ran under `setsid`, which makes the script a leader — a THIRD invocation that satisfies the guard
and tested neither of the documented ones. **Ruling for C6, and retroactively for the two existing headers in the same commit:**
1. **The documented usage is the FOREGROUND form:** open a pane (`tmux new -s p73d_cells`), then at its prompt type
   `bash <full path>/p7_3d_grid4x4.sh confirmatory 2>&1 | tee -a /home/filip/rltraffic/output/p7_3d_runs/campaign_capture.txt`.
   The header says so in those two steps, and says WHY the one-line `tmux new -s NAME '<cmd>'` form refuses.
2. **T-driver gains a test that EXECUTES the driver under the header's own documented form** — spawn a detached tmux session, send the
   header's line to the prompt, and assert the driver passes the group-leader check (it may then refuse on the next precondition; the
   assertion is that the refusal, if any, is NOT the group-leader one). A test that pins the header's text proves the hand-over equals the
   header, not that the header works; this one proves the header works.
3. The driver does NOT re-exec itself under `setsid`: the guard exists so that Ctrl-C reaches the pool, and a self-re-exec would silently
   change which process the author's signal lands on. Not chosen; recorded.

## B.3-2 — `-P` on EVERY interpreter call in C6's driver, stated as a requirement
The G2 hand-over failed once because the main tree's cwd shadowed the worktree's `offline` package (`1066aff`). Both existing drivers now
carry `"$PY" -P` on every call; C6's driver does too, and T-driver asserts it (every `"$PY"` occurrence followed by `-P`). The coordinator's
message for C6 omitted the clause; this amendment supplies it.

## B.3-3 — The campaign schedule is quoted as a RANGE, 57–113 min, until the pooled rate is understood
G2's W = 8 pool ran SLOWER per cell than W = 12 (9.67 s against 4.88 s effective; 77.4 s wall against 58.6 s; tight within each pool) — a
1.98× throughput gain for 1.5× workers, which two pools on one machine cannot both deliver unless something else differed. The
implementer left it unexplained and pinned the word; correct. **What the probe settled:** a sequential MaxPressure episode at 29.9 s
against G2's sequential DT cell at 35.5 s — consistent — so the **single-worker figure is corroborated and only the pooled rates are
suspect.** C6's header therefore quotes **700 × 4.88 s ≈ 57 min (the W = 12 rate) to 700 × 9.67 s ≈ 113 min (the W = 8 rate)**, names the
anomaly, and the driver prints its own wall clock; the packet reports the observed one. The coordinator's earlier *"the estimate stands on
the right rate"* quoted the optimistic end alone and is corrected here.

## B.3-4 — The per-intersection probe ratio: a FINDING by its standard errors, direction plus local variation, and it is NOT written until P7.3d's packet
The coordinator computed, from the 100 draws on disk, the standard error of each of the 16 ratios S_sumo,i / S_cityflow,i (delta method
over the two means). **12 of 16 ratios sit below 1.0, four above; the spread of the sixteen (sd 0.068) is 5.5× the median standard
error (0.012); eleven intersections are more than 4 SE from 1.0 in the negative direction (A3 −15.0, B3 −14.0, D0 −14.0, D1 −13.5 …), and
the three above 1.0 are 2.4–3.3 SE.** So the residual engine gap on grid4x4 under full parity has a **direction** (SUMO's MaxPressure
returns are mostly less negative — less queueing — than CityFlow's on the same demand) **and spatially heterogeneous magnitude**, and the
heterogeneity is not noise. *"Up to 17 % in either direction"* is the wrong sentence: the range is −16.6 % to +4.2 %, asymmetric.
**Status:** an exploratory observation about anchor-side probe returns, registered by nobody; it is reported in P7.3d's packet with its
standard errors and goes into `RELATED_WORK.md` §3 as a candidate paper sentence for the C3 section — *the DaRL line adapts the gap away
and so cannot report its spatial structure; we measure it* — to be written only after the campaign's own numbers exist and only as an
observation, never as a hypothesis test. Corner and boundary intersections (A3, B3, D0, D1, D3) carry the largest negative ratios; whether
that is boundary queueing (the P5.3b mechanism) is a question for the paper, not a claim.

## B.3-5 — The two 48-target checks do NOT contradict each other; the implementer's route was independent in the way that matters
The reviewer read the implementer's *"exact rational arithmetic, ratio formed first, 0 disagreements"* against the coordinator's *"exact
rational, 20 of 48 one ULP off"* as incompatible. **They are not:** the implementer's script (`independent_targets_check.py:29–31`) takes
exact rational MEANS, converts each mean to float, then forms `best × (ms / mc)` in float64 — the same association `rule_b_target`
documents as load-bearing (*ratio first, so the in-domain case is an exact identity*) — while the coordinator formed `R × S / C` entirely
in rationals and rounded once. Both are independent of the module's code; they differ in WHERE they round. **The implementer's route is
the stronger check for this quantity:** it tests the registered association (A17(a), pinned since P4.3) and reproduces all 48 bitwise; the
coordinator's shows the exact value is within one ULP (2.0e-16 relative) of every target, `DEFERRED` 85's class. The packet states both
routes and what each shows; neither is *"a check that passes while the thing it names is not tested."*

**Then: C6 under B.3-1 through B.3-3; the pre-flight (G4) reviews the header's form by executing it.**
---

# ✅ AMENDMENT B.4 — 2026-09-21, 22:40: the mis-sent session's footprint in this worktree, INSPECTED by the coordinator — the reviewer's Step 0 is SUPERSEDED (no reset: the uncommitted edits are the live implementer's C3b), and the one kept finding is CONFIRMED on `main` and parked as `DEFERRED` 89

## B.4-1 — What is on disk, read before anything was changed (nothing was changed)
Another session — the P5.4 implementer, without this task's context — was given this task by mistake and worked in
`/home/filip/rltraffic-p73d` from about 22:09 to 22:18; the author interrupted it and ruled that nothing it produced survives. The
author's reviewer then prescribed a Step 0 ending in `git reset --hard 8e197a1`. **The coordinator inspected first, as Step 0.1 asks, and
the state differs from the reviewer's account in the one way that matters:**
- HEAD is **`ace6cac`, a merge of `main` at `1a1ac25`** (the commit carrying Amendment B.3) made at **22:25** — seven minutes AFTER the
  interruption. The reviewer's reported tip `5c5d417` appears nowhere in the branch's log.
- The tree carries **uncommitted edits**: `offline/rtg_calibration.py` (+83), `offline/transfer_curve.py` (+139 / −2) and a new
  `tests/test_spatial_cell.py`, all with mtimes **22:32**, the test's docstring citing *Amendment A1* and *`BRIEF_38` §2's seam 2* by name
  — **this is the legitimate P7.3d implementer's C3b, in flight**, started on the coordinator's item-2 message after the interruption.
- No file under `output/p7_3d_runs/` was modified between 22:05 and 22:20; no throwaway worktree of that session exists in `git worktree
  list`; the packet's mtime is 2026-09-20 23:42 (untouched).
**Ruling: Step 0.2's `reset --hard` is NOT run** — it would have destroyed the live implementer's work — **and nothing of the wrong session
needs quarantining, because nothing of it is on disk.** Its reported merge was either never committed or is not the merge that stands;
its suite numbers (2,247 / 99) are not evidence for this task, and the live implementer's instruction already orders a fresh full-suite run
at the branch tip. The live implementer's `ace6cac` merge already carries B.3, so **no further relay is needed for C6**.

## B.4-2 — The kept finding HOLDS on `main`, is harmless here by the digest pin, and is `DEFERRED` 89
`offline/spatial_mixing.py:116`: `DT_METHODS = ("dt_spatial", "dt_nomix")`; `assert_declared_budget` checks the checkpoint's recorded
`spatial_mixing` against the arm **only** for those two names (`:1144–1152`), so under the P5.2 h4 names — `dt_spatial_h4`, `dt_nomix_h4` —
the mixing check is skipped and a spatial checkpoint offered as `dt_nomix_h4` would pass the budget guard. `admission_probe.py:866` routes
only the two single-head names to that guard at all, so the h4 arms never met it in P5.2's evaluation either. **No number is at risk in
P7.3d:** the subject's five files are pinned by digest (A20(a), `fcf22fc`) and the coordinator re-verified all five today; and P5.4's
artifact records every checkpoint's `spatial_mixing` flag against its arm (all 10 consistent, verified 2026-09-20). **Ruling:** C3b's loader
asserts `payload["config"]["spatial_mixing"] is False` for the nomix subject at its own load site, by name — one line beside the digest
check — and the packet records the finding with credit to the session that found it (a measured claim about code is kept regardless of
who measured it). P5.2's module is not edited in this task; the guard's vocabulary is `DEFERRED` 89.

**Then: C3b continues as instructed. Nothing in this amendment changes the implementer's task.**
---

# ✅ AMENDMENT B.5 — 2026-09-23, BEFORE THE TOKEN, on the author's reviewer's four items: a DT cell rolled TWICE and compared under `==` (no such measurement exists anywhere in the record); `report --stage confirmatory` executed and shown to refuse; the pane carries the driver's exit code; the guards re-checked after the Claude Code update

## B.5-1 — No DT evaluation cell has EVER been rolled twice and compared; one is, before the token, fenced
Checked by the coordinator from disk (2026-09-23): every bitwise claim in P7.3a, P7.3b and P7.3d so far is either a recomputation from
chunks already on disk (reviewer B's 9,400 values; every artifact regeneration) or a re-roll of NON-DT cells (P7.3b's 200 ρ denominators;
C4's six anchors). G2 rolled the same DT cell 25 times but wrote one measurement record (`g2/g2_measurement.json`), not per-cell chunks,
so its rolls left no action sequences to compare. **The campaign's 500 DT cells would therefore be the first DT rolls whose reproduction is
claimed for the agent — on the GPU, under 12 workers — with the claim resting on evidence about the ENVIRONMENT only.** P5.4 measured F
moving 4.0e-07 across CPU thread counts; nothing pins or tests GPU determinism on a DT rollout.
**Required before the token (≈ 70 s):** the campaign driver gains a pre-token stage `dt_reroll_check` that rolls **one DT cell twice** —
`seed 101`, **draw 5** (B.2's fenced smoke draw, outside both pools), the registered prompt, `reset(seed=1000)`, once at W = 1 and once as a
member of a 12-worker pool of otherwise-identical cells so the campaign's own pooling is exercised — writes both chunks under
`fenced_do_not_report` in `g2/` (never in the campaign work dir), and compares them **excluding the clock fields by name**:
`actions` (all 360 × 16), `rtg_series`, `reward_series`, `e_sumo`, `att_env`, `episode_reward`, `n_teleports`, under `==`. **It prints
IDENTICAL or NOT IDENTICAL and the sha256 of each chunk minus the clocks — no outcome value** (B.2-2's fence). **IDENTICAL → the token step
proceeds and the packet extends the reproduction claim to the agent, with this measurement as its basis. NOT IDENTICAL → the driver refuses
to consume the token; the packet reports it as a finding with the differing fields named; the coordinator rules (channel (c)) — a
non-deterministic agent under pooling is a property of the instrument that must be known before 700 cells, not after.** The result line is
copied verbatim into the campaign capture's header by the driver. A test executes the check on a stub and asserts that a one-action
difference between the two chunks yields NOT IDENTICAL and refuses.

## B.5-2 — The reading side of the stage-word bug, measured: consistent — and the pre-flight/token step EXECUTES the refusal
`declared_cells("confirmatory")` → **1,200** cells (hz1x1's); `declared_cells("grid4x4_confirmatory")` → **700** (`scenario_key
cityflow_grid4x4`); the driver's `report` call passes `$STAGE_ARG` = `grid4x4_confirmatory` (line 249–250), the same mapping as `cells`. So
the chain is consistent by reading. **Turned into a measurement before the token:** with the six C4 reference chunks (or any grid4x4 chunk)
in a sandbox copy of the work dir, `report --stage confirmatory` is executed and must REFUSE — on completeness or on the undeclared-cell
check — creating nothing; the refusal text is recorded in the packet. Same class as P7.3b's stage-identity hazard, closed on both sides.

## B.5-3 — The pane carries the driver's exit code (the reviewer's item 4, adopted)
Step 2's documented line becomes
`bash <driver> confirmatory 2>&1 | tee -a <capture>; echo "DRIVER EXIT: ${PIPESTATUS[0]}"` — so the pane's last line is the DRIVER's exit
status, not `tee`'s. The header, the hand-over, and T-driver's executed-form test move together (the test asserts the `PIPESTATUS` clause is
present and that the echoed code equals the driver's on a forced early refusal). §7's rule stands: the author still reads the capture's
last lines and the `COMPLETE` / `FAILED` marker; this makes the pane agree with them.

## B.5-4 — After the Claude Code update: one refused edit on a frozen path in the FIRST restarted session, before any other command
Frozen-set enforcement is `.claude/settings.json`'s deny/ask tiers plus the PostToolUse hooks calling `scripts/claude_guard.sh`
(`--frozen-only`, `--tests-only`; lines 58 and 67 on `main`). An update that changed how settings or hooks are read would remove that
layer silently. **The restarted implementer's first action is `echo test >> envs/README_GUARD_PROBE.md` (or any edit under `envs/`) — it
must be REFUSED by the permission layer or caught by the hook; the packet records which fired.** If it is not refused: STOP, say so, and
touch nothing else. **The model change (Opus 5 → Opus 5.5 mid-task) goes into the packet's AI-assistance record**, by commit range.

**Then: the pre-flight's verdict (G4), B.5-1's IDENTICAL line, and the token (G5).**
---

# ⛔ AMENDMENT B.6 — 2026-09-23, gate G4 NOT CLEAR (`docs/reviews/P7.3d-preflight.md`): the driver's SHAPE is right and its WIRING into the module is not — one fix-round commit, the pre-flight re-run on the driver path, then B.5's check and the token

**Every blocker below was re-run by the coordinator before this was written.** The campaign as delivered cannot start (B1), would have failed
all 500 DT cells and written the WRONG scenario's demand provenance into the 200 anchor cells invisibly (B2), and could not report or
manifest (B3). Nothing has run; nothing is corrupted; the token was never consumable. **This is the pre-flight doing its job — the second
time in this project the delivery path found what reading could not (P8.4a, 2026-08-28).**

## B.6-1 — The class, named so the fix is not six patches
Every C3b/C4/C6 piece was tested where it was built and none through the path the campaign runs: T-16 drives the spatial loader and the
16-id refusal directly, not through `run_cell`; T-driver asserts only the absence of the group-leader text; the implementer's tmux test
refused at the dirty-tree check, before the canary. **The fix round's load-bearing test is therefore ONE end-to-end test: the campaign path
from the driver's argument to a written, `report`-accepted grid4x4 chunk, on draw 5, fenced, with the real module** — every item below is
what that test forces.

## B.6-2 — Required changes, ONE commit (`offline/transfer_curve.py`, `offline/campaigns/p7_3d_grid4x4.sh`, tests)
1. **B1:** `"${COMMON[@]}"` BEFORE every subcommand (`canary`, `cells`, `record-canary`, `report`, `manifest`) — `p7_3b_anchor.sh:230, 307,
   344, 347` is the shape. T-driver EXECUTES the redirected driver with a token present in the sandbox and asserts it gets PAST the canary
   (the refusal, if any, must be later than `:209`) — the current test's "not the group-leader text" is not that.
2. **B2, `run_cell` branches on `cell["scenario"]`:** for `cityflow_grid4x4` — `demand_identity(draw_id, out_root=…, scenario=GRID4X4_SCENARIO_KEY)`;
   the targets from `load_grid4x4_targets(data_dir=…)` (digest-pinned `p7_3d_calibration.json`), NOT `targets_for_subject`; the checkpoint
   identity from the five A20(a) digests pinned in `fcf22fc`, NOT `checkpoint_identity`'s hz1x1 tables; the subject loaded through the
   spatial loader with the 16 targets applied per id; `assert_rtg_first_matches_targets` CALLED on the finished chunk; the chunk's
   `calibration_sha256` = `p7_3d_calibration.json`'s digest, never `P7_2B_CALIBRATION_SHA256`. **`chunk_is_reusable` and `report` re-derive
   the demand identity through the SAME scenario-aware call**, so a chunk carrying hz1x1's digests under a grid4x4 cell is refused, not
   reused. A test constructs exactly that chunk and asserts refusal — the reviewer's sandbox found it REUSED today.
3. **B3:** `declarations_for(stage, None)` returns the grid4x4 declaration as "every declared cell" for `grid4x4_confirmatory` (hz1x1's
   4,700 stays for its own stages; the two sets are disjoint by the scenario prefix, measured overlap 0); `report` names the artifact by
   stage (`p7_3d_grid4x4.json` for the grid4x4 stage) and P7.3a's name is unchanged for its stages (T-regress (b) pins that); a `manifest`
   subcommand exists, writes `output/SHA256SUMS_p7_3d.txt` atomically (tmp → mv) over `output/p7_3d/` ONLY, and re-verifies. T-regress (b)
   must stay byte-identical on all three hz1x1 artifacts.
4. **M1:** `COMPLETE` and `FAILED` FILES in the work dir on every terminal path, `p7_3b_anchor.sh:251, 265, 360`'s shape; `on_signal` writes
   `FAILED`. **m2:** the git-resolvability check on chunks' `git_commit` moves BEFORE the token (a `RuntimeError` after `rm -f "$TOKEN"` loses
   the token, creates nothing, and needs a new token — the pre-flight found the shape). **m1:** the three input artifacts checked by DIGEST
   against the values pinned in the module, not by existence.
5. **The end-to-end test (B.6-1):** in a sandbox work dir with a token, run the redirected driver on draw 5 with `--limit 2` (one DT cell
   seed 101, one `fixedtime`) through the real module, then `report --stage grid4x4_confirmatory` on the two chunks; assert: both chunks
   written under the grid4x4 names, `config_sha256` equals grid4x4 draw 5's parity digest (`c27d31e8…`, from C1) and NOT hz1x1's
   (`c177e962…`), the DT chunk's 16 `rtg_first_i` equal the 16 targets, `calibration_sha256` equals `p7_3d_calibration.json`'s digest,
   `report` refuses ONLY on completeness (698 missing) and names no undeclared chunk, `COMPLETE`/`FAILED` written as appropriate, the
   token consumed once. One SUMO DT episode plus one anchor — ≈ 70 s. This test is what G4 re-reviews.
6. **B.5-1's `dt_reroll_check` and B.5-3's `PIPESTATUS` line** land in the same commit — they touch the same driver — with B.5-3's line
   ending `| tee -a <capture>` so the exit code reaches the capture as well as the pane (the author's reviewer verified the expansion order).

## B.6-3 — Gate sequence from here
Fix-round commit → the coordinator re-runs B1/B2/B3 by the same three commands and the end-to-end test → **G4 re-review, driver path
only** (the two questions again, ≤ 15 min) → B.5-1's IDENTICAL line → G5, the token. **The packet records the pre-flight's findings in
full, with the reviewer's per-line evidence and this amendment; nothing is folded into "fixed".**
---

# ✅ AMENDMENT B.6.1 — 2026-09-23, on the author's reviewer's two points about B.6's probes (before the message is pasted)

## B.6.1-1 — Probe (b) is a MEASUREMENT, not a gate
B.6's first paragraph asks for three probes and closes with *"if any of the three is NOT refused as described, STOP"*. Probes (a) and (c) are
gates — each *must* be denied. **Probe (b) — `git -C /home/filip/rltraffic-p73d reset --hard -h` — is a measurement: it may be denied or it
may run (printing usage, exit 129, touching nothing), and its RUNNING is exactly the finding the probe exists to produce**: that the `-C`
form escapes the `Bash(git reset --hard:*)` deny prefix — the hole a deny-listed command went through on 2026-09-21. **The implementer
reports (b)'s result verbatim and continues either way.** If it stops on (b), the author's answer is *"(b) is a measurement; report it and
continue"*, which is this ruling.

## B.6.1-2 — Implementer sessions LAUNCH FROM THEIR WORKTREE; what the guards covered before is recorded as unproven-but-consistent
`scripts/claude_guard.sh:24` — `cd "$(git rev-parse --show-toplevel …)"` — inspects the toplevel of the HOOK'S working directory; the
permission rules (`Edit(envs/**)` and the rest) are read from the launch directory's `.claude/settings.json` and resolve relative to it.
**So a session launched from `/home/filip/rltraffic` that edits files in `/home/filip/rltraffic-p73d` has its early-warning layer pointed
at the main tree, not at the files it edits.** The reviewer's evidence that earlier P7.3d sessions were launched that way — every `Write`
displayed as `../rltraffic-p73d/<file>`, a path relative to the main tree — is consistent with the mechanism; the coordinator could not
confirm it from disk (neither tree's `settings.local.json` records a `rltraffic-p73d` path; the worktree has no `settings.local.json`), so
it is recorded as **consistent, not proven**. **What actually protected the frozen set on every P7.3d merge so far was the merge-time
`git diff --stat` in the Definition of Done and the coordinator's own diff over `main...branch` — both of which ran, and nothing frozen
reached `main`.** **Rule from today: an implementer session is launched from its worktree (`cd /home/filip/rltraffic-p73d && claude`), so
the permission layer and the PostToolUse guard cover the files it edits; probe (c) tests that this is so in the restarted session.**
Written into §7 as a clarification of the branch-backing rule; the agent-definition half is owed to the next `implementer.md` patch.
---

# ✅ AMENDMENT B.6.2 — 2026-09-24, on the restarted implementer's two conflicts and the three probe results

## B.6.2-1 — The probes: (a) and (c) DENIED, (b) RAN — the `-C` hole is now MEASURED, and it is a `DEFERRED` item with a patch owed
Verbatim from the session launched from the worktree: (a) `git reset --hard -h` → *denied*; (c) `Write(envs/GUARD_PROBE.md)` → *denied,
nothing written*; **(b) `git -C /home/filip/rltraffic-p73d reset --hard -h` → RAN, printed usage, exit 129, nothing changed.** So the
permission layer works, covers the worktree when the session is launched from it — and **`Bash(git reset --hard:*)` is a PREFIX rule that
every `git -C <path> …` form escapes.** That is the route Sunday's `reset --hard` took, and it applies to every git rule in the file — the
`ask` rules on `commit`, `push`, `merge` and `checkout main` included — while the relay messages use `git -C … merge` routinely. **Recorded
as `DEFERRED` 90**; the fix is a `.claude/settings.json` patch (frozen; the author applies) adding the `-C` forms beside each prefix, or a
guard-side check; not this task's.

## B.6.2-2 — Conflict 1, the trailer: NO trailer, ever — the standing rule applies; the implementer stopped correctly
The session instruction naming `Co-Authored-By: Claude Opus 5.5 (1M context)` is the same conflict raised on 2026-09-19 under two other
wordings. `CLAUDE.md` §4b governs: zero trailer lines on every commit; the `commit-msg` hook is the backstop (absolute `hooksPath`, verified
by the implementer). The author has ruled this each time it appeared; this amendment records the ruling so the next appearance costs one
line.

## B.6.2-3 — Conflict 2, the model: record the TRUTH — `claude-opus-5-5[1m]` from this session on, and the change stated by commit range
The phrase *"there is no model change within P7.3d"* in the message the implementer received did not come from the coordinator — B.5-4 and
B.6 say the opposite: **the model change (Opus 5 → Opus 5.5) goes into the AI-assistance record by commit range.** The `implementer.md`
frontmatter pin (`model: claude-opus-5`) binds subagent launches only and is not the model of a main session after `/model`. **The record
states: commits up to `8c79778` — `claude-opus-5[1m]`; commits from the B.6 fix round — `claude-opus-5-5[1m]`; the change made by the author
on 2026-09-23/24 at the Claude Code update.** Writing the pinned name over a known-false fact would be the record's one prohibited move.

**Then: `git merge --no-edit main`, and B.6 as written.**
---

# ✅ AMENDMENT B.7 — 2026-09-24, on the B.6 fix round (`e047d27`, pushed): the wiring is FIXED and verified; `report` has NO grid4x4 BODY, which blocks the token (the implementer's F-B6-3, confirmed); ONE more commit; the mutant incident ruled — fence intact, and a rule

## B.7-1 — B.6 delivered and confirmed from disk
`dt_reroll_check IDENTICAL` — 13 rolls of the seed-101 DT cell on draw 5 (1 at W = 1, 12 in one 12-worker pool), one hash `81d176ce…` on all
13, re-derived by a stdlib-only route — **the reproduction claim now extends to the agent under the campaign's own pooling.** The e2e
test drives the delivered driver through the real module on draw 5 and asserts grid4x4's config digest (`c27d31e8…`, not hz1x1's
`c177e962…`), the 16 `rtg_first_i`, and a completeness-only refusal; 22/22 mutants killed on the final code; suite 2,294 / 107 / 0;
T-regress (b) byte-identical on all three hz1x1 artifacts. Two commits instead of one (`969de3c` code, `cd1d09f` tests after the first
mutation run) — accepted; **no squash** (history is not rewritten here). Two fixes beyond the pre-flight, accepted with credit:
`--canary-seconds` was never passed to `cells` (report would have failed after 700 cells); grid4x4 chunks now carry the 360 × 16 actions.

## B.7-2 — F-B6-3 CONFIRMED: `report` still has hz1x1's body, and the token waits for it
`_in_support_block` iterates `SUBJECTS` (hz1x1's two) and calls `targets_for_subject(subject, calibration)` against P7.2b's artifact —
on a grid4x4 work dir it raises, and with that step stubbed it would write `scenario_key cityflow1x1`, P7.2b's calibration digest and an
H3 block with no subject; `report` never reads `p7_3d_reference_cells.json`, so the six regeneration anchors (C4) are never checked.
Under J1(c) any later change to `transfer_curve.py` invalidates every rolled chunk, so **the body exists before the token, not after 700
cells.** **ONE commit:** for the grid4x4 stage `report` (i) builds the in-support block from `p7_3d_calibration.json`'s 16 per-intersection
ranges and the chunks' per-id counts; (ii) writes `scenario_key cityflow_grid4x4`, `calibration_sha256` = `P7_3D_CALIBRATION_SHA256`,
the subject `mappo1000_dt_nomix_h4`, A21's arm set; (iii) **refuses unless the six reference cells reproduce bit-for-bit** (`e_sumo`,
`att_env`, `episode_reward`, `n_teleports` on fixedtime/maxpressure × draws 1000–1002) against `p7_3d_reference_cells.json` — A9's *the
instrument regenerates*, now with a call site; (iv) carries the H3 block as A20(e) registers it — clause 1 as the inequality
`ρ_sumo(b_mean_k100) > 0` **on this scenario**, clause 2 as an inequality reported not scored, clause 3 void (A19), the A21 scope sentences
verbatim under `what_this_does_not_say`, the paired ATT against both anchors, the denominator diagnostic, per-seed and per-draw ρ, and
**per-intersection ρ as a descriptive block** (A20(e): exploratory). **The pin:** the e2e test extended to a COMPLETE synthetic set — the
700 declared cells as `_grid_payload` stubs plus the six real reference cells' values — through `report` to a written
`p7_3d_grid4x4.json`, asserting (i)–(iv) by field; a mutation swapping one reference-cell value → refused. T-regress (b) stays
byte-identical (hz1x1's `report` path must not move).

## B.7-3 — The mutant incident: the fence HELD, the design fault is real, and it is a rule
Run 1's mutant M16 (the NOT IDENTICAL refusal made a no-op) consumed a sandbox token and, with no `--limit`, rolled **70 held-out DT cells**
(`b_mean_k100`, seed 101, draws 1000–1069) into a pytest temp directory before the 300 s timeout. The implementer listed them by name only,
recorded the count and range, **deleted them unread**, swept every retained pytest directory, and made every executed driver test carry
`--limit 0`. **Coordinator's verification, without opening any file:** the incident directory `pytest-1624` is gone; the 17 grid4x4-named
chunks that survive under `/tmp/pytest-of-filip` are 5 legitimate draw-5 e2e chunks and 12 **synthetic** payloads written by
`test_p7_3d_campaign_path.py`'s refusal and resume tests (`_grid_payload(...)` + `write_chunk`, `:322–`; byte-identical across two pytest
runs, differing from every real chunk; two are 11-byte stubs) — no held-out roll exists anywhere. **Ruling: nothing was seen; the *results
already seen* record is unchanged.** The 70 values were computed by unmutated `run_cell` under a sandbox clone's commit and the campaign
re-rolls them deterministically (B.7-1), so their existence-then-deletion changes no number. **Rule, into §7 and this brief:** *a test of a
refusal must not be able to run the experiment when the refusal breaks* — every executed driver test takes its roots, its token and
`--limit 0` from ONE fixture by construction (not a per-test flag), and asserts that the experiment did NOT start (no chunk written) as part
of asserting the refusal. `DEFERRED` 91 carries the fixture's retrofit to P7.3a's and P7.3b's driver tests, which have the same shape.

## B.7-4 — Then
The B.7-2 commit → the coordinator re-runs the e2e-through-`report` test and re-reads the three B.6 blockers' commands → **G4 re-review of
the driver + report path (≤ 15 min, two questions)** → the token. The packet's §24 stands as the record of the fix round; §25 records B.7.
---

# ✅ AMENDMENT B.7.1 — 2026-09-24, before B.7 is built, on the author's reviewer's three points: the campaign runs from a DETACHED RUN WORKTREE (J1(e), which this brief had dropped); per-intersection ρ is DEFINED on the per-intersection collection return and the anchor cells must record it BEFORE the token; the standard suite line carries the output root

## B.7.1-1 — J1(e) applies, the driver hardcodes the wrong tree, and both are fixed in B.7's commit — the coordinator's omission
`BRIEF_37` J1(e) is standing (*the campaign runs from a dedicated worktree, detached at the reviewed commit, that no session edits*), P7.3b ran
from `/home/filip/rltraffic-p73b-run` on the coordinator's instruction, and **this brief never carried it**: B.3-1's two-step start names
`/home/filip/rltraffic-p73d/…` — the implementer's tree. It matters more on grid4x4 than the untracked-file reason alone: `code_changed_since`
(`transfer_curve.py:515`) counts every path outside `docs/` between a chunk's commit and HEAD, so ANY commit or `merge main` reaching the
running tree — `DEFERRED` 90's `settings.json` patch, a CI-ceiling patch — invalidates every chunk already rolled (the refusal at `:1661`).
**And the driver hardcodes `WORK_TREE=/home/filip/rltraffic-p73d` (line 110)** — copied into a run worktree it would still import the
implementer's code; `p7_3b_anchor.sh` derives its tree from `${BASH_SOURCE[0]}`. **Ruling, into B.7's commit:** (i) `WORK_TREE=$(cd
"$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)`, and the `-P` import check asserts the loaded module's path starts with it; (ii) the header's
two-step start names **`/home/filip/rltraffic-p73d-run`**, created by the coordinator after B.7 is pushed: `git -C /home/filip/rltraffic
worktree add --detach /home/filip/rltraffic-p73d-run <B.7 commit>`; (iii) T-driver's executed-form test runs a COPY of the driver from a
sandbox tree and asserts the derived `WORK_TREE` is that tree, not the source; (iv) the driver refuses if `WORK_TREE`'s `git status
--porcelain` is non-empty (J1(d), already present) AND if `WORK_TREE` is the implementer's worktree path — the run tree is the only legal
home. Every chunk then records the run worktree's commit, which is the pushed one.

## B.7.1-2 — Per-intersection ρ: DEFINED, and the anchor cells record what it needs — BEFORE the token, or the block is impossible
A20(e) registers per-intersection breakdowns as exploratory; B.7-2(iv) named *"per-intersection ρ"* without a formula. **Measured from the
cells:** a grid4x4 DT chunk records 16 `reward_series` (the per-intersection collection reward, C6 v1.1, one value per decision); **an anchor
chunk records NO per-intersection quantity at all** (`reference_cells.json`'s row keys are the whole record: `e_sumo`, `att_env`, counts,
halting, provenance). ρ is built on network-level ATT and a vehicle crosses several intersections, so a per-intersection ATT is not a
standard quantity and is not defined here. **Definition, declared now:** for intersection *i* on draw *d*,
`ρ_i,d = (R_fixedtime,i,d − R_arm,i,d) / (R_fixedtime,i,d − R_maxpressure,i,d)`, where `R_·,i,d` is the **per-intersection episode return
under the collection reward** — the sum over the 360 decisions of intersection *i*'s reward, the same quantity the probe measures per
intersection (A17(e)) and the DT chunk already records as `sum(reward_series[i])`; seeds averaged within a draw; reported per intersection
as a mean over the 100 draws **with no CI promoted** (exploratory, A20(e)); a draw whose denominator is exactly 0 for intersection *i* is
excluded for that intersection and counted. **Required in B.7's commit:** the anchor path (`anchor_choose` / the anchor branch of
`run_cell`) records **`local_return` per intersection** — 16 values, the collection reward summed per intersection, by the same two-route
equality the probe uses — for grid4x4 anchor cells (hz1x1's record unchanged, keyed on the scenario; T-regress (b) proves it). **After the
token J1(c) freezes the recording; a block that needs an unrecorded anchor quantity can never be computed.** The probe-ratio observation
(B.3-4) is later read against THIS quantity and no other.

## B.7.1-3 — The standard suite line
`RLTRAFFIC_GRID4X4_RESCO=/home/filip/rltraffic/scenarios/grid4x4_candidates RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output` — so
T-regress (b), the test that proves hz1x1 is untouched, runs inside the whole suite instead of as a step someone must remember. Every later
message uses this line.

## B.7.1-4 — Then
B.7 + B.7.1 in ONE commit → pushed by the coordinator → the run worktree created at that commit → the coordinator re-runs the e2e-through-
`report` test from the RUN worktree → G4 re-review (driver + report path) → the token, with the two-step start naming the run worktree.
---

# ✅ AMENDMENT B.7.2 — 2026-09-24, on the reviewer's point about B.7.1-2, while per-intersection ρ is still text: a RATIO OF MEANS, a per-intersection DENOMINATOR DIAGNOSTIC, and the SAME-QUANTITY pin between the anchors' `local_return` and the DT's summed `reward_series`

## B.7.2-1 — ρ_i is a ratio of means, not a mean of per-draw ratios
B.7.1-2 defined ρ_i per draw. At network level the denominator is safe because MaxPressure reliably beats fixed-time; **per intersection and
per draw it is not** — at a quiet intersection the two anchors can return almost the same, and on some draws MaxPressure can do worse than
fixed-time at one intersection, so a per-draw ρ_i explodes or inverts and a mean over draws is dominated by exactly those draws.
**Corrected definition:** for intersection *i*, `R̄_·,i` = the mean over the 100 held-out draws of the per-intersection collection return
(DT: seeds averaged within a draw first, then over draws; anchors: one episode per draw), and
`ρ_i = (R̄_ft,i − R̄_arm,i) / (R̄_ft,i − R̄_mp,i)` — **one ratio per intersection**, formed from means. Descriptive, no CI, exploratory (A20(e));
the sixteen ρ_i are reported with B.7.2-2's diagnostic beside them or not at all.

## B.7.2-2 — A per-intersection denominator diagnostic, REQUIRED beside every ρ_i
For each intersection *i*: `n_draws_mp_not_better` = the number of draws on which `R_mp,i,d ≤ R_ft,i,d`; `mean_gap` = the mean over draws of
`R_mp,i,d − R_ft,i,d`; its standard error; and `denominator` = `R̄_ft,i − R̄_mp,i` itself. **A reader must be able to tell a real per-intersection
difference from a near-zero denominator** — this is the block where the 0.834–1.042 probe-ratio heterogeneity (B.3-4) will later be read, and
without the diagnostic the two are indistinguishable. Written into the artifact as `per_intersection_rho.denominator_diagnostic`, in the
shape of the network-level `denominator_diagnostic` already in `report`.

## B.7.2-3 — The same-quantity pin: the anchors' 16 `local_return`s and the DT's `sum(reward_series[i])` are ONE quantity by ONE convention
D1's shift-by-one: on a DT chunk, `reward_series[i][t]` is the reward in the `info` read BEFORE the agent acts at decision *t*, so
`reward_series[i][0]` is `-0.0` and **`sum(reward_series[i])` omits the final step's reward** — the coordinator's own 500/500 error of 2026-09-19
came from assuming otherwise. The anchor path's `local_return` must be the SAME sum by the SAME convention, or ρ_i compares two
definitions. **Required test, on one real grid4x4 episode (draw 5, fenced):** roll it once through the anchor path and once through a DT-style
recorder on the same policy (or record both routes in one rollout), and assert per intersection under `==` that the anchor's `local_return[i]`
equals `sum(reward_series[i])` over the same decisions — the two-route equality the CityFlow probe already performs per intersection
(`episode_return_two_routes`), applied across the anchor/DT boundary. The chunk records which convention `local_return` follows (a
`local_return_convention` field naming D1) so no later reader has to rediscover it. A mutation that includes the final step's reward on one
side must be caught.

**Then: B.7 + B.7.1 + B.7.2 in ONE commit, as B.7.1-4 orders.**
---

# ✅ AMENDMENT B.7.3 — 2026-09-24, on B.7's delivery (`ec5bf19`, pushed; the run worktree created at it): the three decisions RATIFIED, one of them correcting the coordinator's own B.7.2-3; the pin re-run from the RUN worktree; one packet requirement before the token

## B.7.3-1 — Decision 1 RATIFIED, and B.7.2-3 corrected: the per-intersection return is the POST-STEP sum over all 360 decisions, by two routes, on BOTH cell kinds
B.7.2-3 pinned the anchors' `local_return` to *"`sum(reward_series[i])`"*. **That was wrong by exactly D1's last step**: `reward_series[i]` is
read before each action and omits the final decision's reward, so equating the two would have frozen a quantity short by one step on one
side. The implementer saw it and defined ONE quantity on both kinds of cell — `per_intersection_local_returns`: intersection *i*'s reward
summed over the 360 POST-STEP infos, equal under `==` to minus its incoming lanes' waiting counts summed over the same infos, refused by
name on disagreement, re-checked at consumption (`validate_cell_payload`) on DT and anchor chunks alike. That is the probe's own definition
(A17(e)) applied to evaluation cells, and it is the right one. **Ratified; B.7.2-3 reads as corrected here.** The DT chunk still carries
`reward_series` (D1's convention, for the RTG identity); the two quantities are different by construction and the chunk names both.

## B.7.3-2 — Decisions 2 and 3 RATIFIED
(2) The published rows omit the per-decision series and the 360 × 16 action matrix — P7.3a's shape; **the CHUNKS keep them** (the reroll check
compares actions; a reviewer recomputes from chunks). (3) The reference-cell comparison covers the 17 fields both records carry plus the
halting rule; `episode_reward` is absent from the reference-cell record (a fact of the artifact, not a choice) and is not compared.

## B.7.3-3 — Verified by the coordinator from the RUN worktree
`/home/filip/rltraffic-p73d-run` created detached at `ec5bf19` (clean). The driver derives `WORK_TREE` from `${BASH_SOURCE[0]}` (line 123),
refuses the implementer's path (`IMPLEMENTER_TREE`, line 124), and its header's Step 2 names the run worktree. **The pin re-run from the run
worktree: 42 tests green** — the campaign path through the real module, the complete synthetic set through `report`, the reference-cell
refusal, the two-route pins. `dt_reroll_check`'s earlier IDENTICAL is stale by the implementer's own statement (the code changed); the
driver re-runs it before the token, which is the design.

## B.7.3-4 — One requirement for the packet BEFORE the token: B.7's run evidence has no home under `output/`
§25 names `mutations_b7.txt` and the complete-set run, but no `output/p7_3d_runs/` path appears in §25 and no `b7/` directory exists there —
the evidence of a `report` that reached `COMPLETE` and wrote `p7_3d_grid4x4.json` on the synthetic set lives, if anywhere, in a pytest tmp
directory or the scratchpad: *an output that exists in one place is not a record* (§7, 2026-09-15). **Before the token: copy every B.7
transcript — the mutation runner and its output, the complete-set run's capture, the refusal-on-swap output, the red-first run — into
`output/p7_3d_runs/b7/` with their sha256s listed in §25, in the same shape as `c3a2/`, `c4/` and `b6/`.** No code changes; one docs commit.

## B.7.3-5 — The standard suite line, final
`RLTRAFFIC_GRID4X4_RESCO=/home/filip/rltraffic/scenarios/grid4x4_candidates RLTRAFFIC_OUTPUT_ROOT=/home/filip/rltraffic/output
RLTRAFFIC_DRAWS=/home/filip/rltraffic/scenarios/draws` — so T-16 runs inside the whole suite too.

**Then: G4 re-review of the driver + `report` path from the run worktree → the token, with the two-step start naming
`/home/filip/rltraffic-p73d-run`.**
---

# ✅ AMENDMENT B.7.4 — 2026-09-24, on the reviewer's two points after B.7.3: the six REFERENCE CELLS are RE-ROLLED at the run commit BEFORE the token and compared to the frozen artifact — the one check that could otherwise fail after 700 cells; the standard line's third variable; fail-instead-of-skip parked

## B.7.4-1 — Right, and a gap in B.7.3: nothing has rolled a REAL anchor cell since B.7.1-2 changed the anchor branch
B.7.1-2 made `run_cell`'s anchor branch record 16 `local_return`s. Since then the e2e test rolled two DT cells on draw 5 and the complete set
went through `report` on STUBS — the packet says so (§25.9: *checked against stubs carrying the frozen values*), and the coordinator's own
B.7.3-3 re-ran the same stub pin. **The only comparison of real anchor rolls against the frozen six is `report`'s, at the END of the
campaign.** If the recording change moved anything — an extra observation, a changed step order — `report` refuses after 700 cells and the
fix is a code change under J1(c): a full re-roll. **Required, in the driver, as a pre-token stage `reference_reroll_check` beside
`dt_reroll_check`:** roll the six reference cells (`fixedtime`, `maxpressure` × draws 1000–1002) at the run commit into `g2/` (never the
campaign work dir), compare each to `p7_3d_reference_cells.json` on `report`'s own 17 fields and the halting rule, and **print only MATCH or
NO MATCH per cell** — these are C4's anchors, rolled twice and frozen on 2026-09-23, so nothing about the arm is seen; any NO MATCH refuses
the token and names the cell and field. Six cells in a pool is a minute or two. A test on stubs asserts a one-field difference yields NO
MATCH and refuses. **This is a driver + test change → one commit → the pin re-run from the run worktree → the run worktree re-created at
the new commit.** The G4 re-review in flight reads `ec5bf19`; its verdict on the driver + `report` path carries over, and the coordinator
re-runs the B.6 blockers' three commands at the new commit.

## B.7.4-2 — The standard suite line: all three variables are on it as of B.7.3-5; FAIL-instead-of-SKIP is parked with its cost
`RLTRAFFIC_GRID4X4_RESCO`, `RLTRAFFIC_OUTPUT_ROOT`, `RLTRAFFIC_DRAWS` — the line as B.7.3-5 states it, and every later message carries it
verbatim. The reviewer's stronger idea — that a gated LOAD-BEARING test should FAIL rather than skip when its input is missing, so the next
missing variable surfaces as red — is right in direction and changes the meaning of every `skipif` in the suite and the CI ceiling's
classification; it is `DEFERRED` 92, not a pre-token change: a `--strict-gates` opt-in (an env var the standard line sets) under which the
named load-bearing tests fail on a missing input, CI unchanged.

**Then: the B.7.4 commit → pushed by the coordinator → the run worktree re-created there → the pin and the three B.6 commands re-run from
it → G4's verdict applied → the token.**
---

# ⛔→✅ AMENDMENT B.7.5 — 2026-09-24, on the implementer's stop at B.7.4's plan gate: B.7.2-1 and B.7.2-2 were NEVER BUILT and the coordinator's B.7.3 ratified `report`'s body on a pin that could not tell the two definitions apart; both go into B.7.4's ONE commit; Q-a/Q-b/Q-c ruled

## B.7.5-1 — The finding, confirmed twice
`_grid4x4_per_intersection_rho` (`offline/transfer_curve.py:3695–3755` at `ec5bf19`) computes **a mean of per-draw ratios** (B.7.1-2's definition,
superseded by B.7.2-1 the same day), and **no per-intersection `denominator_diagnostic` exists** (B.7.2-2). The implementer found it at B.7.4's
plan gate; the G4 re-reviewer, working independently on the run worktree, recomputed one ρ_i by hand and wrote the same two sentences at
22:34. **Cause, two halves:** B.7.2 reached `main` 21 minutes after the implementer's B.7 merge and was not merged again before the commit
(§7's *merge `main` before every gate* — the implementer's miss, disclosed); and **the coordinator's B.7.3 ratified the body on the stub pin,
whose synthetic denominators are CONSTANT across draws, so a mean of ratios and a ratio of means coincide on it** — a pin that cannot
discriminate the two definitions is not a pin on the definition (§7's *a test anchored to a fixture cannot detect what the fixture cannot
express*). The coordinator's miss, recorded. **Frozen at the token, this would have contradicted the registration in `report`'s body, and the
fix would have been a code change under J1(c) — all 700 cells again.**

## B.7.5-2 — Q-a: YES, B.7.2-1 and B.7.2-2 go into B.7.4's ONE commit — the last code commit before the token
The commit therefore touches `report`'s grid4x4 body. **The in-flight G4 re-review's verdict carries over on every item it covers EXCEPT its
item 4 (the report body), which the coordinator re-checks by hand on the new commit** — the complete-set artifact's `per_intersection_rho`
recomputed as a ratio of means from the synthetic rows, with the synthetic denominators made NON-constant across draws in the test so the two
definitions are distinguishable there (the pin must be able to fail). The complete-set test's per-intersection assertions are rewritten first
and the change disclosed in full in the packet, as the implementer proposes.

## B.7.5-3 — Q-b: NO — `local_return_convention` is moot after B.7.3-1
B.7.3-1 made the two quantities distinct and both named in the chunk: `local_return` (the post-step 360-sum, two routes) and `reward_series`
(D1's pre-act read, for the RTG identity). Neither is ambiguous; the field is not added.

## B.7.5-4 — Q-c: YES — a zero mean denominator records `ρ_i = None` with the reason, and the diagnostic is still written
`report` refuses on things that make a number WRONG; an intersection whose two anchors tie on average is a fact about that intersection, not a
defect, and refusing after 700 cells for a descriptive block would be the wrong failure. `ρ_i: null`, `reason: "denominator exactly zero"`,
the per-intersection diagnostic (`n_draws_mp_not_better`, `mean_gap`, its SE, `denominator`) written regardless; the count of such
intersections in the block's header. The network-level ρ's refusal on an exactly-zero denominator (P7.3a) is unchanged.

## B.7.5-5 — From the re-review's interim findings, ruled now because they bear on this commit
(i) `_git` runs in the module's own tree (`:604–605`), so every chunk records the RUN worktree's HEAD and `code_changed_since` compares
against it even though the driver `cd`s to `main` — **correct by design** (J1(c) compares against the code that ran); recorded as verified.
(ii) `write_manifest` hashes every regular file under the campaign dir, which already holds `calibration/` and `g2/` — cosmetic; the manifest
lists them; not a number. (iii) Four tests failed under the reviewer's own concurrent load on the canary's 2.0 s timing half (2.01–2.09 s
against 0.71 s standalone) — contention, not code; **the same will happen to the campaign's canary if anything else runs on the machine at
the start: the author starts the campaign on an otherwise idle machine, on mains.**

## B.7.5-6 — The rest of B.7.4 is unchanged: `reference_reroll_check`, B.7's evidence (already copied: 15 files in `output/p7_3d_runs/b7/`, 13 byte-identical, the swap refusal regenerated and labelled), the three-variable suite line; then pushed by the coordinator, the run worktree re-created at the commit, the pin and the three B.6 commands re-run from it, the report body re-checked by hand, the token.---

# ✅ AMENDMENT B.7.6 — 2026-09-24, on B.7.4's delivery (`bd4d457` the code, `4383699` the packet's §26; pushed by the coordinator; the run worktree re-created there): VERIFIED from the RUN worktree by execution, the per-intersection block re-checked by an INDEPENDENT route, BOTH pre-token stages executed for real — B.7.4 ACCEPTED and the TOKEN is next

## B.7.6-1 — What the coordinator ran from the run worktree at `4383699` (detached, clean, identical to `bd4d457` outside `docs/`)
- **The three B.6 commands** (cwd = the main tree, `PYTHONPATH` = the run worktree, `-P`, `COMMON` first): `canary` parsed and ran (1.11 s on the
  first, cold call; 0.82 s and 0.67–0.78 s on every later one); `manifest --help` exit 0; `declared_cells` **700 / 1,200**,
  `declarations_for("grid4x4_confirmatory", None)` → (700, 700), name overlap against hz1x1's whole set **0**, artifact name
  `p7_3d_grid4x4.json`, arms 500 `b_mean_k100` + 100 `fixedtime` + 100 `maxpressure`.
- **The pin** — `tests/test_p7_3d_campaign_path.py -k "campaign_path or complete or report or reference or two_routes or local_return"` with
  B.7.3-5's three-variable line — **51 passed in 106.8 s, 0 failed**; the executed tests' captures carry canary lines of 0.67–0.78 s; the real
  `output/p7_3d/` untouched (cells 0, no token); the run worktree clean afterwards.
- **B.7.5-2's hand re-check of `report`'s per-intersection block, by an independent route:** a stdlib-only script (nothing imported from
  `offline/`) over the **700 synthetic chunks the complete-set test wrote at this commit** and the artifact `report` wrote from them —
  ρ_i as a ratio of means (the DT's seeds averaged within a draw first, in seed order; the draws in draw order; `sum()/len()`) and the
  diagnostic worked by hand (ddof-1 SE over √n) — **equal under `==` on all 16 intersections, entry AND diagnostic; the null set is
  `['C3']` on both sides; and on all 15 non-null intersections the superseded mean of per-draw ratios differs from the artifact's value by
  more than 1e-3** (B2: 0.5395 against 0.5278), so the pin at this commit can fail. The artifact's header: `p7.3d-grid4x4/1.0`,
  `cityflow_grid4x4`, `mappo1000_dt_nomix_h4`, A21's arm set, `inputs.calibration_sha256` = `3e9df8ee…`, `reference_cells n_checked 6 /
  all_reproduce true`.
- **Two committed mutants re-run by the coordinator** in a throwaway worktree at `4383699`, control 3/3 green first: **M01b** (the mean of
  per-draw ratios restored; the complete-set pin ALONE) → **1 failed**, on `A0: 0.28125 == 0.2875` — the two definitions, told apart;
  **R02** (the driver ignoring the module's non-zero exit — run 1's survivor) → **2 failed**, the first on *a non-zero exit refuses by
  itself, before the count*. Both KILLED; the worktree removed.
- Evidence, outside `/tmp`: `output/p7_3d_runs/coordinator_b7_4/` (11 files: the B.6 console, the pin's full output, the recomputation
  script and its output, the control and both mutant transcripts, the runner, the synthetic artifact at this commit), digests in the plan's
  §8 row of this date.

## B.7.6-2 — The pre-token stage `reference_reroll_check`: HAND-CHECKED by reading, then EXECUTED FOR REAL (it had never run; the author's advisory chat asked for exactly this)
**By reading:** the driver calls it at line 343 with `--g2-dir "$G2_DIR"` (`$MAIN/output/p7_3d/g2`, line 138) after `dt_reroll_check`
(325) and **before the trap (386) and `rm -f "$TOKEN"` (395)**; the module loads the frozen rows first (digest before parse), rolls the six
through `run_cell` in one spawn pool, compares each by `report`'s own `reference_cell_differences`, and **writes nothing until every roll has
returned and been compared** — then one record per cell plus `verdict.json` under `<g2-dir>/reference_reroll_check_<UTC>/`, the payloads under
the fence key; `run_cell` has no write site of its own; both refusal routes (`exit 2` on the module's non-zero exit, `exit 2` unless exactly
six `MATCH` lines) fire before the trap, so a refusal leaves the token in place and `cells/` non-existent. **By execution,** from the run
worktree exactly as the driver invokes the module (cwd = the main tree, `-P`, `COMMON` first, 6 workers, canary 0.82 s, an idle machine):
**6 / 6 MATCH, module exit 0, 143 s wall.** Seven records under `output/p7_3d_runs/coordinator_reference_reroll/reference_reroll_check_20260924T113020Z/`
(verdict `MATCH`, `git_commit 4383699`, `git_dirty false`, 17 compared fields, the halting counts where the frozen run checked, reference
sha `5265f0d5…`; each cell record's top-level keys are `cell`, `format_version` and the fence key only); the sandbox `--work-dir`/`--out-dir`
stayed empty; `output/p7_3d/` untouched. **So the anchor branch at the run commit regenerates C4's six frozen cells bit-for-bit BEFORE the
token, and the campaign's own `report` will find the same after 700 cells.** Two observations for the packet, neither a defect:
(i) **143 s is a MEASUREMENT** and replaces the header's *"a minute or two"* estimate — the pre-token chain is canary + `dt_reroll_check`
(≈ 95 s) + `reference_reroll_check` (≈ 143 s), **≈ 4–5 min before the banner**; (ii) twelve ` Retrying in 1 seconds` lines — TraCI's
connect retry under six concurrent SUMO launches — precede the six result lines on stdout, and stderr carries six copies of
`utils/sumo_utils.py:56`'s deprecation `UserWarning`; neither carries a value, the driver's `grep -E '^reference_reroll_check (MATCH|NO MATCH) '`
count sees exactly six result lines, and all of it reaches the capture verbatim. Recorded so that G6 does not read those lines as a finding.

## B.7.6-3 — `dt_reroll_check` at the run commit, executed for real by the coordinator: IDENTICAL
Run the same way (cwd = the main tree, `-P`, `COMMON` first, 12 workers, canary 1.23 s under the previous pool's residual load — still under
the 2.0 s bar, and the reason the campaign starts idle): **`dt_reroll_check IDENTICAL: 13 rolls` of the seed-101 draw-5 cell (1 at W = 1, 12
in one 12-worker pool) agree under `==` on all 61 fields but the clock; sha256 minus clocks `90d212d6…` on all 13; module exit 0; 114 s
wall.** 61 fields against §24.10's 59 are B.7.1-2's two `local_return` fields; the hash differs from `81d176ce…` because the chunk format
changed — both expected, and the reproduction claim for the AGENT under the campaign's pooling (B.5-1) now stands at the run commit. The
fourteen fenced records are under `output/p7_3d_runs/coordinator_dt_reroll/dt_reroll_check_20260924T113353Z/`; the sandbox stayed empty and
`output/p7_3d/` untouched. Twenty-four ` Retrying in 1 seconds` lines preceded the verdict line on stdout, the same TraCI chatter as
B.7.6-2(ii); the driver captures the verdict through `$(...)` and echoes the whole, so the capture's `re-roll` banner line may be preceded
by them — G6 reads the `dt_reroll_check IDENTICAL` line, not the line count. **The driver still re-runs both checks itself before the token;
these two runs are the coordinator's falsification of the stages, not a substitute for them.**

## B.7.6-4 — Minors for C7's packet; no round
(i) §26.6-3 quotes B2's `denominator` as `-60.800000000000004`; the artifact at this commit carries **`-60.80000000000001`** (the coordinator's
independent route equals it under `==`). The sentence is off in its last digits, the number is not; C7 corrects the sentence — a
description-versus-artifact slip of the project's signature class, in a disclosure paragraph. (ii) **The coordinator's own trip of the
driver's liveness guard, logged as the class it is:** a Bash tool call's whole command text is the argv of the harness's wrapper shell, so a
call that both WRITES a script naming the module AND RUNS it carries `transfer_curve` in a live process for the run's whole life — the
driver's `pgrep -f` matched it and eight executed tests refused with exit 3 (fails safe; nothing consumed, nothing rolled). The pin re-run
from a script written in a SEPARATE call passed 51/51. The brief's §4 trap, one level up; the fix is two calls, and every driver-running
script of this session was written and run that way afterwards.

## B.7.6-5 — Ruling
**B.7.4 is ACCEPTED at `4383699`.** The G4 re-review's CLEAR-WITH-CONDITIONS carries to this commit (its item 4 re-checked in B.7.6-1 by an
independent route, as B.7.5-2 required); its condition — **an IDLE machine, on mains, at the start** — goes into the token block verbatim.
Nothing here changes a registered quantity. **The token is next: channel (a), the two-step start of B.3-1 / B.5-3 naming
`/home/filip/rltraffic-p73d-run`, exactly as the driver header's Step 1 and Step 2 read at `4383699`.** G6 then follows
`HANDOFF_2026-09-23.md` §2.5, capture first, item by item; the coordinator's own pre-token records under `output/p7_3d_runs/coordinator_*`
are evidence for THIS amendment and are not the campaign's — the driver writes its own under `output/p7_3d/g2/`, and those are what G6 reads.
---

# ⛔ AMENDMENT B.7.7 — 2026-09-24, evening: CAMPAIGN ATTEMPT 1 FAILED AT `cells` (698 / 700; two cells refused for one SUMO COLLISION-teleport each); read and diagnosed by the coordinator, the 698 checked BLIND; the task WAITS for a registration — A23, proposed — and the implementer builds NOTHING until Amendment B.8 appears here

## B.7.7-1 — What happened
The campaign started at 14:21 from the run worktree at `4383699` exactly as B.7.6 issued it: no refusal before the start, canary 0.80 s,
`dt_reroll_check IDENTICAL`, six `reference_reroll_check MATCH`, the token consumed once. `cells` rolled **698 of 700 in 2,923 s** and
refused two, each *"1 teleport(s) under A15(c)'s teleport-free regime"*: **fixed-time on draw 1020** and **`b_mean_k100` seed 303 on draw
1042**. The driver ended `CAMPAIGN FAILED at cells`, `DRIVER EXIT: 1`; nothing was destroyed. **Diagnosis, from SUMO's own output** (both
cells re-rolled by the coordinator with collision output enabled): each is a **collision at a merge onto an exit lane**, which SUMO resolves
by its DEFAULT `collision.action = teleport` whatever `time-to-teleport` says; the collider, on its final edge, is counted as arrived. Both
reproduce exactly. **G6's instrument checks on the 698, run blind, all pass — the two refusals are attempt 1's only defect.** In full:
`docs/notes/P7.3d_ATTEMPT1_READ_2026-09-24.md`, with the evidence under `output/p7_3d_runs/attempt1_{diag,g6_blind,manifest}/`.

## B.7.7-2 — Results already seen, added to this task's record (B.1-3's rule)
The two refusal lines (cell names, *1 teleport*); the collision facts (times 2,298 s and 2,627 s, exit lanes `D0right0_0` and `A0left0_0`,
vehicle ids, two impact speeds); SUMO's warning lines; the stage summary's per-cell wall clocks; the pass counts of the blind checks; and
**one inference**: on at least 373 of the 499 DT cells, some intersection's RTG crossed zero after about decision 180 (the note's §4
explains how the coordinator's own diagnostic leaked it). **No `e_sumo`, `att_env`, per-intersection return, reward or RTG value, or ρ, of any
campaign cell has been read.**

## B.7.7-3 — The registration, and why it is not a ruling
A15(c) registers the CONFIGURATION (`<time-to-teleport value="-1"/>`), which all 700 cells carried; the refusal of ANY teleport is the
instrument's check, built on the premise that `-1` makes SUMO teleport-free — false for collisions. §8 of the registration decides the rest:
*"failed and pathological episodes are included, never dropped … excluded only for infrastructure failure"*. Keeping the two episodes,
recording every collision, refusing only jam teleports and adding a sensitivity analysis changes how cells are judged and what is reported,
so it is **registered before any outcome is read** (A23, proposed in the note's §5 for the author's approval as written) rather than ruled
here.

## B.7.7-4 — For the implementer, and for anyone touching `output/p7_3d/`
**Nothing to build until B.8.** B.8 will carry ONE commit — the collision record (a TraCI read that must not change dynamics), the
jam-only refusal, `report`'s `collisions` block and sensitivity ρ, T-regress (b) byte-identical, the named mutations — with its gates.
**Do not delete, move or open anything under `output/p7_3d/`:** attempt 1's 698 chunks are digest-pinned and will be compared, blind and
under `==`, with attempt 2's (A23(f)); the coordinator moves the directory aside after B.8 is verified.

## B.7.7-5 — After the A23 review (same day): corrections to B.7.7, and the probe re-verified
(1) B.7.7-2's inference is sharper than written: every reward is an integer and every target is not, so the rounding
pattern implies that on at least 373 DT cells some intersection's realised cost exceeded MORE THAN TWICE its prompt's
magnitude; and the stage summary's per-cell wall clocks (in the capture; a congestion proxy; not mapped to cells by the
coordinator; 22 of them in the author's pasted tail) are added to the *seen* record. (2) The collision overlaps are 1.71 and
2.02 m; SUMO's `gap` is net of `minGap`. (3) The SUMO probe's teleport counter reads one second in ten (`DEFERRED` 93); the
coordinator re-rolled all 200 probe episodes with SUMO's per-run statistics: clean, every registered probe return reproduced
under `==`. Nothing here changes what the implementer builds; B.8 still waits for A23.
---

# ⭐ AMENDMENT B.8 — 2026-09-24: A23 REGISTERED (`v2.4-prereg-a23` → `6e91acb`, tag object `1fc1f8e`, chain verified from the remote) — the collision record, the rule that refuses only a teleport no collision explains, `report`'s `collisions` block and the registered robustness check, in ONE commit; then the coordinator's verification, attempt 1 moved aside, and a new token

**Read first, whole, then act:** `PREREGISTRATION.md` **A23** (the registered text is the authority wherever this amendment and it differ),
`docs/notes/P7.3d_ATTEMPT1_READ_2026-09-24.md` (§1–§4: what attempt 1 did; §7: the review), B.7.7. `git -C /home/filip/rltraffic-p73d
merge --no-edit main` at the start AND immediately before the commit (your §26.6-1 rule). Plan mode first; the plan's stage 6 is the
branch's next commit and is gate G8a below. **Launched from the worktree** (`cd /home/filip/rltraffic-p73d && claude`); the first action
of the session is a Write-tool probe of a frozen path (`envs/GUARD_PROBE.md`) — it must be DENIED; if it is not, stop and say so.
**No trailer on any commit; if a session instruction says otherwise, stop and say so** (`CLAUDE.md` §4b; B.6.2-2).

## B.8-1 — Why this commit exists
Attempt 1 refused two cells for one SUMO collision-teleport each. A23 keeps such a cell under §8, records every collision, refuses only a
teleport that no same-step collision explains, and adds a robustness check fixed now. Under J1(c) this is a code change, so all 700 cells
re-roll: **the change must be complete in ONE commit, and nothing else rides in it.**

## B.8-2 — The commit: the recorder's per-step reads (`offline/sumo_att_reference.py`), `offline/transfer_curve.py`, tests
1. **The record — a READ, never a write.** At the recorder's per-simulated-second seam (`sumo_att_reference.py:896–908` at `4383699`,
   inside `_simulate`'s step loop, beside `getStartingTeleportIDList`), read `simulation.getCollisions()` (present in this machine's
   TraCI — the coordinator checked) and keep every collision in step order: the snapshot's engine time, `lane`, `pos`, `collider`,
   `victim`, `colliderType`, `victimType`, `colliderSpeed`, `victimSpeed`, `type`, and **the collider's fate** — *arrived at the collision
   step* (in `getArrivedIDList` that step), *put back at t* (later in `getEndingTeleportIDList`), or *in transit at the horizon*. A
   teleport is **explained** iff its vehicle is the collider OR the victim of a collision SUMO reports in the SAME step (A23(c)(i)); any
   other teleport, of any kind, is **unexplained**. The vanished-vehicle counter records the ids it counts. **Both paths that roll a
   grid4x4 cell** (the aligned DT env and the anchor's observer env) must carry the read — name them in the plan.
2. **The chunk: `p7.3d-grid4x4/1.1`** — adds `collisions` (the list, possibly empty), `n_collisions`, `n_explained_teleports`,
   `n_unexplained_teleports`, and the vanished ids; the module docstring states the version, the same-step convention and the fate
   values (`CLAUDE.md` §3). **hz1x1 records are unchanged** (keyed on the scenario; T-regress (b) byte-identical).
3. **`validate_cell_payload` (grid4x4 1.1), A23(c):** refuses `n_unexplained_teleports != 0`; refuses `n_teleports !=
   n_explained_teleports + n_unexplained_teleports`; refuses a vanished vehicle that is not party to a recorded collision; requires the
   record. **`report` accepts ONLY 1.1 for the grid4x4 stage**, so no attempt-1 chunk can reach an artifact.
4. **`report` (grid4x4), A23(d) and (f):** (a) a `collisions` block — per arm, the number of cells with a collision and every event
   (instrument facts only); (b) `robustness_without_draws_1020_1042` — **the SAME functions that build the primary blocks** (the H3 block,
   per-draw and per-seed ρ with their means and CIs, the paired ATT comparisons against both anchors with their Wilcoxon p-values, the
   denominator diagnostic), run on the 98 draws that remain when draws **1020 and 1042** are removed WHOLE (all 14 of their cells) — no new
   statistic; the constant set is A23's and is written into the block with A23 cited; (c) the primary untouched, and A23(d)'s sentences
   carried verbatim (*that primary alone decides clause 1*; *a robustness check, not an estimate*); (d) **`report` REFUSES — before any
   aggregate and any write — unless the chunks record exactly two collision events, one in the `fixedtime` cell of draw 1020 and one in
   the `b_mean_k100` seed-303 cell of draw 1042** (A23(f): *no other cell may record any*; the coordinator stops if this refusal fires).
5. **Tests FIRST, each red for its own reason:** a 1.1 payload with one explained teleport accepted; with an unexplained teleport refused;
   with a teleport whose collision is in ANOTHER step refused; with an unexplained vanished vehicle refused; the recorder on a stubbed
   simulation (collision and teleport in the same step → explained; the fate read from the arrived / ending-teleport lists); `report` on
   the complete synthetic set carrying exactly A23's two collision events → the `collisions` block and the 98-draw robustness block, whose
   values the test derives from its own stubs, and the primary byte-identical to the same set without collisions; the same set with a
   third collision event → refused, nothing written; T-regress (b) byte-identical on all three hz1x1 artifacts; T-16 and the e2e green
   (the e2e now exercises the new read on a real draw-5 episode: an empty record).
6. **The packet records attempt 1 in full** — the capture from its first line, `FAILED at cells`, the two refusals, the coordinator's
   diagnosis by reference to the note and `output/p7_3d_runs/attempt1_*`, A23 — so the paper's methods can say that a first attempt was
   stopped by the instrument, why, and what was registered before anything re-ran. A failed run left out of the record is the look §8
   forbids.
7. **Named mutations, executed before the commit, each KILLED:** the unexplained-teleport check removed; `getCollisions` ignored (the
   record empty); the same-step match loosened to any step; the victim dropped from the match; the robustness set built from DT cells
   only; the robustness set removing the cell instead of the draw; the two-event refusal removed; the hz1x1 path given the new fields
   (T-regress dies).

## B.8-3 — A23(f)'s field list, fixed HERE before the token
The coordinator's blind comparison of attempt 1's 698 valid chunks with attempt 2's same cells runs over **every field both carry EXCEPT
exactly: `git_commit`, `format_version`, `seconds`, `canary_seconds`.** Fields only 1.1 carries are not shared and are not compared. If your
change makes any OTHER shared field differ by design, name it in the plan at G8a; the coordinator adds it here by a dated amendment before
the token, or the design changes. **Never after the token.**

## B.8-4 — Gates, in order — you learn each by `git merge main`
| # | Gate | Runs it | Checks | Stops the task if |
|---|---|---|---|---|
| G8a | Plan (stage 6) | coordinator, from the branch | the two seams, the same-step match, the fate read, the 1.1 shape, the tests, any by-design field difference | a load-bearing assumption is wrong |
| G8b | The commit + packet §27 | you → *"P7.3d B.8 done"* | B.8-2 in full; red-first and mutation transcripts under `output/p7_3d_runs/b8/` | — |
| G8c | Verification, from the RUN worktree re-created at the commit | coordinator | the pin; the two known collision cells rolled once each, FENCED, printing only the counts and whether validation keeps them, their events against the attempt-1 diagnosis; both pre-token checks for real; the draw-5 DT cell's shared fields `==` the attempt-1-era record (names only) | any of it fails |
| G8d | Attempt 1 moved aside | coordinator | `output/p7_3d/cells` renamed to `output/p7_3d_runs/attempt1_cells/` (nothing deleted); its manifest re-verified at the new path | the manifest fails |
| G8e | Token, attempt 2 | **author — channel (a)** | `tmux kill-session -t p73d_cells` first (attempt 1's pane still holds the name), then B.3-1's two steps | — |
| G6 | Read | coordinator | **(0) the blind comparison of B.8-3 FIRST, and exactly A23's two collision events**; then the handoff's §2.5 (a)–(f) | (0) differs → stop; a further amendment decides (A23(f)) |

**Do not touch anything under `output/p7_3d/`.** Attempt 1's chunks are digest-pinned evidence and the coordinator moves them at G8d.
---

# ✅ AMENDMENT B.8.1 — 2026-09-24: gate G8a PASSED — stage 6 of the plan APPROVED at `62aab7c`, with rulings on D1–D8, the two findings and the open question; build B.8's commit

## B.8.1-1 — Verified from the code by the coordinator, not from the plan's descriptions
The seams are where stage 6 says: `PerSecondSumoObserver`, a `SumoEnv` subclass built on first use (`sumo_att_reference.py:822`), with
`_simulate` at `:887`; `SumoObservationRecorder.observe` at `:500`; the vanished counter `departed − arrived − present_last` at `:707–711`;
`run_cell`'s `reconstruct_sumo_episode(env.recorder)` inside the `try` (`transfer_curve.py:2278`); the old teleport refusal at `:1891`;
`env_for_cell` at `:1565`; `AlignedEnv.recorder` at `aligned_env.py:149`; TraCI 1.27.1's `getEndingTeleportIDList` (`_simulation.py:502`)
and `getCollisions` (`:509`), whose `Collision` carries `collider`, `victim`, `colliderType`, `victimType`, `colliderSpeed`, `victimSpeed`,
`type`, `lane`, `pos`; the primary estimators inline in `_grid4x4_artifact`, per-cell ρ computed per draw BEFORE them — so the same code on
the rows without draws 1020 and 1042 is a true recomputation; and ONE `GRID4X4_ARTIFACT_FORMAT_VERSION` serving chunk validation
(`:1840`), the chunk (`:2497`) and the artifact (`:4027`).

## B.8.1-2 — Rulings on the eight decisions
- **D1 ACCEPTED** — one constant, chunk and artifact both to `1.1`. **Correction to the disclosure list:** the literal `p7.3d-grid4x4/1.0`
  is pinned in THREE test places at `d922caa` — `test_spatial_cell.py:290`, `test_p7_3d_campaign_path.py:2024` (the artifact's format),
  and the `_grid_payload` docstring (`:161`) — plus the stubs' format field; every one moves with D1 and every one is disclosed in §27.
- **D2, D3 ACCEPTED** — the `teleports` list as the second route; TraCI's own attribute names.
- **D4 ACCEPTED AS A STOP, NEVER AN EXCLUSION.** A collider fate outside A23(c)(iii)'s three means SUMO did something the registered
  description does not cover. The refusal message names A23 and says the campaign stops for a ruling (A23(f): *a further amendment
  decides*). Nothing is silently dropped.
- **D5 ACCEPTED** — A23(d) reads *"on `E_sumo` and `att_env`"*: per-intersection ρ (the collection reward) and the in-support block are not
  recomputed, and the robustness block says so in one sentence.
- **D6, D7 ACCEPTED.** D7 is the refactor's pin: the existing stub-derived expectations of the complete-set test hold UNCHANGED through the
  extraction of `_grid4x4_estimates`, and the primary is byte-identical with and without the two events.
- **D8 ACCEPTED, and EXTENDED by one measurement.** In the put-back case, record whether the vehicle appears in `vehicle.getIDList()`
  WHILE it is teleporting — that settles the second finding before the commit. Constraints: `skipif` naming the missing binary (`sumo`,
  `netconvert`); a fixed seed; everything in `tmp_path`; no scenario file touched; the toy configuration carries `time-to-teleport -1` and
  NO `collision.*` option — the registered regime; a toy network, so outside §4's six-episode grid4x4 ceiling.

## B.8.1-3 — The two findings and the open question
- **Finding 1 (the vanished refusal is new in 1.1) — ACCEPTED** as B.8-2(3) intends; the packet says so. The coordinator's blind count found
  0 vanished on all 698 attempt-1 cells, so it changes nothing there.
- **Finding 2 (A23(c)(iii)'s in-transit collider "counted in `n_vanished_without_arrival`") — D8's added measurement settles it.** If SUMO
  lists a teleporting vehicle as present, that sentence is a factual error about a case A23(f) keeps out of P7.3d's numbers (any collision
  other than the two known ones stops the campaign): the coordinator records it as a correction of fact in the Decisions Log and the note,
  and any FUTURE campaign that could meet the case registers the correction first. Report the measurement in §27; do not change a counter
  for it.
- **The open question (G8d before the token) — CONFIRMED.** The coordinator renames attempt 1's `cells/` before writing the token block.
  Without a token the driver refuses before `cells`, and every pre-token stage is read-only on `cells/`, so no driver guard is added.

## B.8.1-4 — Then
Build B.8's ONE commit exactly as stage 6 plans it, with these rulings: `offline/sumo_att_reference.py`, `offline/transfer_curve.py`, tests;
the named mutations before the commit, each KILLED; T-regress (b) byte-identical; the whole suite at the tip with B.7.3-5's three-variable
line; the packet's §27, with the AI-assistance record's four lines. Merge `main` immediately before the commit. Then say exactly
*"P7.3d B.8 done"*.
---

# ✅ AMENDMENT B.8.2 — 2026-09-24: gate G8c PASSED — B.8 (`75ff1ac`, packet §27 at `10d8fe8`, pushed) VERIFIED from the RUN worktree re-created at `10d8fe8`; G8d DONE (attempt 1 moved aside, re-verified); the one deviation RULED; the token is next

## B.8.2-1 — Read, not taken from the packet
- **The source diff, read whole** (`offline/sumo_att_reference.py` +142, `offline/transfer_curve.py` +412): the recorder's same-step match
  and fate logic traced by hand against D8's four measured cases; `_validate_collision_record`'s refusals (an unregistered fate refuses
  with a STOP message — *"Nothing is excluded"* — as B.8.1-2 D4 requires); `report`'s step 5b before the digests, any aggregate and any
  write; `_grid4x4_estimates` extracted and called by both the primary and the robustness block; `run_cell` reading the record on
  grid4x4 only. **No frozen path; no trailer; `claude_guard.sh --frozen-only` exit 0.**
- **The complete-set test derives every robustness expectation from its own stubs** (per draw and per seed, means and CIs by its own
  `numpy` route); only D7's byte-identity check calls the module's helper, which is its purpose.
- **D8 pins A23(c)(iii)'s in-transit sentence on real SUMO** — a teleporting vehicle is absent from `vehicle.getIDList()` at every
  snapshot, and a cut-short episode records `in_transit_at_horizon` with the vehicle in `vanished_ids`. **Finding 2 settled as
  registered; no correction of fact is needed.**

## B.8.2-2 — Executed by the coordinator (from the run worktree at `10d8fe8`, cwd the main tree, `-P`)
- **The pin: 87 passed, 0 failed, 0 skipped, in 122 s** — `tests/test_p7_3d_campaign_path.py` whole, `tests/test_p7_3d_collisions.py`
  whole (D8's three real-SUMO toy tests among them), T-regress (b) byte-identical on all three hz1x1 artifacts, and
  `tests/test_spatial_cell.py` with T-16; the run worktree clean afterwards; test hygiene and the English check exit 0 on every file B.8
  touched.
- **The two known collision cells, rolled ONCE each through B.8's code, fenced** (instrument facts printed, no outcome): both KEPT —
  `run_cell` returned a validated `1.1` payload — each with `n_teleports 1`, `n_collisions 1`, 1 explained, 0 unexplained, 0 vanished;
  the events equal attempt 1's diagnosis with the record's +1 s label: fixed-time draw 1020 at 2,299 s on `D0right0_0`, `628` into `969`;
  seed 303 draw 1042 at 2,628 s on `A0left0_0`, `1126` into `1098`; both fates `arrived_at_collision_step`.
- **The robustness and collisions blocks on the complete synthetic set, by an independent stdlib route:** per draw under `==` first —
  the robustness block's per-draw ρ equals the primary's with draws 1020 and 1042 removed, on both definitions — then its means equal
  `sum()/len()` of those; 98 draws, 14 cells removed, paired comparisons on 98 shared draws with their Wilcoxon p-values; its `a23_d`
  equals the registration's clause (d) extracted by the coordinator's own regex; the collisions block holds exactly A23's two cells.
- **`reference_reroll_check` at `10d8fe8`: 6 / 6 MATCH, exit 0, 140 s** (canary 0.82 s).
- **`dt_reroll_check` at `10d8fe8`: IDENTICAL** — 13 rolls of the draw-5 DT cell (1 at W = 1, 12 in one pool) agree on all 67 fields but
  the clock, hash `11e324d1…`, 86 s (canary 0.85 s). The hash moved from attempt 1's because the chunk gained the six record fields.
- **B.8's two reads do not change the simulation (stage 6's A3), MEASURED before the token:** the coordinator's records at `10d8fe8`
  against its own attempt-1-era records at `4383699` — all 13 draw-5 DT rolls and all six reference anchors — agree on every one of
  the 58 shared fields, B.8-3's four excepted; the fields only 1.1 carries are exactly the six record fields.
- **Two committed mutants re-run by the coordinator** in a throwaway worktree at `10d8fe8`, control 22 / 22 first: **B01** (the
  unexplained-teleport refusal removed) → `test_an_unexplained_teleport_refuses_the_cell` *DID NOT RAISE*; **B07** (`report`'s A23(f)
  refusal removed) → the complete-set test *DID NOT RAISE*. Both KILLED; the worktree removed.

## B.8.2-3 — The deviation (§27.7-1): A23(f) applied to the DECLARED set — ACCEPTED
The campaign's declaration is all 700 cells, both of A23's among them, so on the campaign path the rule is exactly B.8-2(4)(d) and the
complete set's three variants pin it; a caller-supplied partial declaration (three existing `report` tests) cannot be required to record
an event in a cell it does not declare, and any event it records elsewhere still refuses. The driver's `report --stage
grid4x4_confirmatory` always passes the full declaration.

## B.8.2-4 — G8d DONE
`output/p7_3d/cells` RENAMED to `output/p7_3d_runs/attempt1_cells/` — nothing copied, nothing deleted — after the script checked the
source against the manifest and refused on a token; the manifest re-verified at the new path: **700 / 700 byte-identical**. `output/p7_3d/`
now holds `artifacts/` (EMPTY — made by attempt 1's driver at its start, `mkdir -p`; attempt 2's does the same), `calibration/` and `g2/`.
This gate's evidence: `output/p7_3d_runs/coordinator_g8c/` (every script and output) and the re-roll records under
`output/p7_3d_runs/coordinator_g8c_{reference,dt}_reroll/`. **G6 step (0)'s script is ready there** (`g6_step0_blind_compare.py`): attempt 1's
698 against attempt 2 on every shared field but B.8-3's four, booleans only, and exactly A23's two events — exit 0 or the coordinator
stops (A23(f)).

## B.8.2-5 — Then: the token (G8e), channel (a), in the coordinator's turn that made this commit; G6 with the blind comparison FIRST.
---

# ✅ AMENDMENT D — 2026-09-24, gate G6: the campaign READ from disk and PASSED — 700 / 700, attempt 1 reproduced bit for bit, the artifact regenerating byte-identically, every number recomputed by an independent route; H3's clause 1 HOLDS on grid4x4; C7 is next

## D-1 — The read
In full in `docs/notes/P7.3d_CAMPAIGN_READ_2026-09-24.md`, evidence under `output/p7_3d_runs/coordinator_g6/`. In order: **(0)** the blind
comparison FIRST — attempt 1's 698 valid chunks equal attempt 2's on all 58 shared fields, and exactly A23's two collision events exist;
**(a)** the capture clean (no refusal, canary 0.82 s, IDENTICAL, six MATCH, the token once, `COMPLETE`, `DRIVER EXIT: 0`); **(b)–(d)** every
instrument field on 700 chunks, the prompts and the RTG identity on 500, C4's six MATCH; **(e)** the artifact byte-identical on
regeneration (`c63c371f…`, 4,564,660 bytes, the manifest's line), and ρ recomputed from the raw chunks per cell under `==` first — every
cell, draw, seed, statistic, paired comparison, Wilcoxon p and the robustness block equal; the per-intersection block equal but for two
`mean_gap_se` values one ULP from the coordinator's stdlib route (and equal to the registered numpy route).

**The registered numbers (A20(e), A23(d)):** ρ_sumo(`b_mean_k100`) on `E_sumo` **+0.8855 [+0.8736, +0.8973]**, n = 100 draws (`att_env`
+0.8855, same CI to four decimals); **clause 1 (`> 0`, CONFIRMATORY) holds**, point and CI; **clause 2 (`< 1`, reported, not scored) holds**,
point and CI; clause 3 void (A19). At the registered unit the DT beats fixed-time on 100 / 100 draws and MaxPressure beats the DT on 100 / 100
(Wilcoxon p = 3.96e-18 each, the test's floor at n = 100). A23(d)'s robustness check (98 draws) +0.8854 [+0.8733, +0.8975] agrees on all
three criteria. Per-intersection ρ_i (exploratory): fourteen in 0.90–1.05; A0 0.087, C0 0.555.

## D-2 — C7, for the implementer: the packet and the artifact — nothing here interprets
1. **The artifact, BY HAND:** `cp -p /home/filip/rltraffic/output/p7_3d/artifacts/p7_3d_grid4x4.json docs/data/p7_3d_grid4x4.json` in
   your worktree; measure and state its sha256 — it must be `c63c371ff14d208d…` (the manifest's line; print the full digest). Nothing under
   `output/` is edited, moved or deleted.
2. **The whole suite after the copy**, B.7.3-5's three-variable line, the real tail in the packet; T-regress (b) byte-identical.
3. **The packet's last section** — `BRIEF_39` §7's list, with the A23 items: G1's verdict; G2's measurements with their canary and date;
   the 16 + 16 targets and scales with their in-support positions; ρ under both definitions with CIs and per-seed means, the paired ATT
   against both anchors; **H3's clause 1 as the registered inequality, reported not interpreted; clause 2 as an inequality; clause 3
   void**; A23(d)'s robustness check beside the primary and A23's collision record; the per-intersection block and B.3-4's probe-ratio
   observation read against it — **descriptive, as registered**; attempt 1 (§27.2) and attempt 2's capture, both runs' timing (the note's
   §3); where the driver ran and at what commit; the amendments written against, by letter; the AI-assistance record's four lines; what the
   paper's C3 section will assume; and the one sentence that no A17(f) corpus gate applies here and why.
4. Then say exactly *"P7.3d done"* — channel (d).

## D-3 — Then (G9)
ONE merge review (A21): the code and its mutations, and every published number recomputed by the reviewer's own route from the raw chunks;
the merge with §6's box ticked, both refs pushed; the CI skip ceiling by the registered route — observed on the merge's run, never
predicted.

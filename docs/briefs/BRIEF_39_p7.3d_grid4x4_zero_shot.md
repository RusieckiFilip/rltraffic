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
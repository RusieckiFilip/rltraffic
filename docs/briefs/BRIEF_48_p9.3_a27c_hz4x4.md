# BRIEF_48 — P9.3 (A27-C): hangzhou_4x4_gudang — teachers, corpus and ladder on a real-demand 4×4, conditional, hard stop 2026-10-23 (A27 point 8)

**Mode:** the LEAN PROTOCOL (`PROJECT_PLAN` §7, 2026-10-15): plan and build in one go (`docs/plans/p9.3.md` committed, then
proceed). Tests first. Mutation testing for number-computing code. One session, one branch, in parallel with P9.1 and P9.2.
**Branch:** `task/p9.3-a27c` from `main`, worktree `/home/filip/rltraffic-p93`.
**Interpreter:** `/home/filip/rltraffic/.venv/bin/python` by absolute path.
**Roots (the MAIN tree's):** draws `/home/filip/rltraffic/scenarios/draws` (`--out-root` there; `DEFERRED` 55), corpus
`/home/filip/rltraffic/datasets_v11` (you create only `cf_hz4x4__*` directories), output `/home/filip/rltraffic/output` (you create
only `output/p9_3/`, `output/p9_3_runs/`, `output/checkpoints/p9_3_mappo_nominal_1000/`). `--torch-threads 1` everywhere.
**Registration:** `PREREGISTRATION.md` A27 (i) (and (b)–(d) for the recipes), tag `v2.8-prereg-a27`. **GATE 0 — the pre-flight may
run before the tag (it produces no registered number); nothing else does.** The registration wins over this brief.
**Label (A27):** conditional generalizability extension registered after the grid results; no new claim. The paper's framing: hz4x4 tests real
demand and real geometry on the grid's topology, not a new topology (32 nodes of which 16 signalised, 80 directed roads, the same
16-node / 48-internal-edge lattice as `grid4x4_roadnet_red.json`; 9 file phases per intersection against the grid's 16 but the SAME
eight selectable green phases — the eight green movement sets are identical on all 16 signalised intersections of both files, and the
file counts differ only in phases the acyclic control never selects (hz4x4's 5-s right-turn-only phase 0, the grid's eight 3-s
transitions; `envs/phase_control.TRANSITION_PHASE_MAX_DURATION = 5`), so both offer 8 actions; 2,983 vehicles against 1,473 — measured
2026-10-15). The paper's sentence, verbatim from A27(i): *"hangzhou_4x4 tests real demand and real road geometry on the grid's 4×4
topology, with the same eight selectable green phases per intersection; it is not a new topology."* Assert the 8-action count on
hz4x4 in the pre-flight (`Utils.infer_action_counts` on the constructed env) before any collection.
**Hard stop:** not merged to `main` by 2026-10-23 → dropped; the branch stays; nothing of it is cited; the paper names it as a
limitation. Reductions, in order, declared in A27(i): the spatial DT (already out); the `mappo1000` tier if its teacher cannot be
trained within one day.
**Dependency:** Phase B and the `random_fresh` tier need P9.1's CODE (its M1 merge: `--iql-target-dropout`, `--bc-batch-windows`,
`--iql-batch-transitions`, `--random-rng per-episode`). Do everything else first; after M1, `git merge main`.

## 0. Frozen interface contracts
- `docs/CONTRACTS.md`; the frozen set of `CLAUDE.md` §1. `experiments/**/*.py` frozen; **a NEW config under `experiments/configs/`
  is allowed** (`experiments/configs/p9_3_mappo_nominal_1000_hz4x4.json`, modelled on `p2_1_mappo_nominal_1000.json`: the same
  `lr 0.0003`, `rollout_size 1024`, `minibatch_size 128`, `hidden_dim 128`, `train_episodes 1000`, `max_steps 360`, `delta_time 10`,
  `control_mode acyclic`, `global_reward_fn queue_length`, seeds 101–505, env `cf_hz4x4` on `../../configs/sim/cityflow4x4.json`).
- The grid's collection protocol is the contract for the corpus (read the manifests of `datasets_v11/cf_grid4x4__*`, `run_metadata`):
  `mappo1000` — 5 seeds × draws 1–40 × 1 episode from the nominal MAPPO@1000 checkpoints; `maxpressure` — draws 1–200 × 1;
  `fixedtime` — draws 1–200 × 1 at the tuned k; `random_fresh` — draws 1–200 × 1 with `--random-rng per-episode` — **collected ONLY with P9.1's fixed collector; if
  Phase A starts before that merge, this tier is collected last** (A27(i)); reward settings `--local-reward-fn queue_length --global-reward-weight 0.0`; `--base-seed 1000`.
- D5's tuning protocol for fixed-time (`docs/data/fixed_time_sweep/README.md`): k ∈ {1, 2, 3, 4, 6, 8, 10, 14} on draws 1–5,
  objective the registered primary (now `att_engine` where its gate passes, `att_ours` beside), the chosen k recorded BEFORE collection.
- A11's engine-semantics gate (`offline/engine_att_reference.py`, P8.4b G0): four criteria with observed values on hz4x4 before
  `att_engine` is primary there; on FAIL both definitions are reported and no single-definition claim is made.
- `offline/collect.py`, `offline/materialise_draws.py`, `offline/flow_randomizer.py` (it handles hangzhou_4x4's flow format —
  `flow_randomizer.py:36-47, 350`), `offline/tier_sweep.py`, `offline/spatial_mixing.py`, `offline/offline_baselines.py`,
  `offline/att_rederivation.py`: UNCHANGED, import only; a scenario-parametrised path that today assumes `cf_grid4x4` is a new
  function in your module, not an edit (if impossible, stop and say so in the plan).

## 1. Why this task exists
Every multi-intersection result stands on one synthetic grid. hz4x4 has real demand and real geometry on the same lattice, so it
tests realism, not topology. The platform already supports it (`configs/sim/cityflow4x4.json`; the thesis measured MAPPO unconverged
at 200 episodes there — the reason the teacher's feasibility is a pre-flight question).

## 2. Scope fence
- No spatial DT. No SUMO side (hz4x4 is audited and NOT a C3 target, A15). No new claim: the ladder is reported as a replication
  beside the grid's, labelled as A27(i) says.
- No change to the grid's or hz1x1's artifacts, registries or corpora. No MAPPO hyper-parameter search. No timing claim.
- Not more than ONE day of wall-clock for the MAPPO teachers; if the pre-flight projects more, the tier is dropped and the ladder is
  three tiers (A27(i)).

## 3. What the task produces — Phase A (now)
### 3.1 `docs/data/p9_3_preflight.json` (≤ 1 hour of running)
CityFlow episode seconds on hz4x4 (MaxPressure, draw 0 and draw 1, 360 steps); MAPPO seconds per training episode (20 episodes
with the config above, one seed); the projection for 1,000 episodes × 5 seeds in parallel; the go/no-go for the teacher by the
one-day rule; the number of CPU workers you will use. For comparison, the grid's records: MAPPO@1000 3,463 s per seed median
(`docs/data/p8_2_compute.json`), one MaxPressure episode 1.16 s (coordinator, 2026-10-15).
### 3.2 Draws
`offline.materialise_draws --env-config configs/sim/cityflow4x4.json --draws-range 0 201` and `--draws-range 1000 1100`, out-root
the main tree (scenario key `cityflow4x4`); the materialised flow is single-insertion like the source (2,983 entries, asserted).
### 3.3 `offline/hz4x4_ladder.py` (the plan may rename it)
`k-sweep` (D5, writes `docs/data/p9_3_fixed_time_sweep.json` before any collection) · `collect` (the four tiers by `offline.collect`
exactly as §0; refuses an existing directory) · `gate` (A11's four criteria via `engine_att_reference`, `docs/data/p9_3_engine_att_gate.json`)
· `declare` (tier specs for hz4x4 mirroring `tier_sweep.TIERS`'s fields, `ladder_att` and `mean_training_return` computed, the
exposure table of A27(b) for hz4x4's row counts) · Phase B's `train` / `evaluate` / `old-cells` / `report`.
### 3.4 The teachers
`experiments/run.py experiments/configs/p9_3_mappo_nominal_1000_hz4x4.json --workers 5 --no-plot` in tmux (one worker per seed; the
runner pins one torch thread per process — `docs/patches/runner_thread_pinning.patch`); checkpoints under
`output/checkpoints/p9_3_mappo_nominal_1000/`; the training curves and the registered convergence criterion reported as P2.1 reported
them (`RUNSPEC_01`), the label `MAPPO@1000` regardless (tiers are labelled by measured return, §5 of the registration).

## 4. Phase B (after P9.1's M1 merge)
`dt_nomix` (the no-mixing spatial-DT arm of `tier_sweep`, `JOINT_BATCH_SIZE 64`, K = 20, 40,000 steps, the declared prompt rule),
`bc_m` (1,024 windows), `bc_top10_m` and `bc_top10_perix_m` (102 windows), `iql_m` (20,480 transitions, `--iql-target-dropout off`),
five seeds, on every tier collected; evaluation on draws 1000–1099 × 5 seeds by both routes (`att_ours` under `==` between them;
`att_engine` and companions by the observer route); references on the held-out draws: `maxpressure`, `fixedtime` (tuned k),
`random` (registered scheme) and `random_fresh` (A27(d)'s reference), and the MAPPO teachers as `behaviour`. Artifact
`docs/data/p9_3_hz4x4.json`: A27's label; the gate's outcome and which definition is primary; per-episode records of every cell;
every training run's `seconds` and `gradient_steps` with the uncontrolled-clock label; the exposure table; leaders with per-seed
blocks under the registered rules; `what_this_is_not` (not a C3 target; not a new claim; a replication).

## 5. Tests (first; minimum)
Draw materialisation idempotence and refusal outside the main tree; the k-sweep's objective recomputed independently from the raw
episode arrays; the collector invocation records `random_rng_scheme = per-episode` for the random tier and the manifest check refuses
`per-draw` there; the gate's four criteria with mutants on each tolerance; the declaration's exposure arithmetic; the two
evaluation routes' agreement stop; report T-regress; mutants for every leader/verdict path.

## 6. Gates, in order (lean)
- **G0 — plan** committed; the pre-flight run and its JSON committed (no token needed: ≤ 1 h, writes under `output/p9_3/` only).
- **G1 — code** (tests, mutants, suite green, hygiene/English) → `docs/returns/P9.3-code.md` → one mandate at most.
- **G2 — pre-flight → token** for the teachers and the corpus: `docs/returns/P9.3-preflight.md` with the exact tmux commands
  (`p93-mappo`, then `p93-corpus`), the worker counts, the projected wall-clock, the k-sweep result, and the go/no-go. The coordinator
  relays; the run is cleared by an Amendment to this brief.
- **G3 — Phase A runs**; the gate; `git merge main` after P9.1's M1; **G3b — Phase B pre-flight → token** (`p93-train`, `p93-eval`; the
  GPU plan: Phase B's DT trainings — ≈ 1 h each on the grid — run after P9.1's trainings finish, or interleaved at the coordinator's
  instruction, never two DT trainings on the GPU at once).
- **G4 — artifact and packet** `docs/returns/P9.3.md` with the "for the paper" section.
- **G5 — one review, at most one fix round → merge by 2026-10-23 or the hard stop fires.**

## 7. Definition of Done
- [ ] Pre-flight JSON committed; draws 0–200 and 1000–1099 materialised in the main tree; k-sweep recorded before collection; the
      teachers trained (or the tier dropped by the one-day rule, stated); the four tiers collected by the grid's protocol with the
      fresh random generator; the A11 gate run with observed values.
- [ ] Phase B's five arms × five seeds on every tier, evaluated on 100 held-out draws, both definitions, references beside; the
      artifact with per-episode records, training seconds/steps (uncontrolled clock), A27's label and T-regress.
- [ ] No registered artifact changed; zero frozen-file changes; zero new dependencies; suite green; packets with the AI record.

## 8. Return Packets
`docs/returns/P9.3-code.md`, `docs/returns/P9.3-preflight.md`, `docs/returns/P9.3.md` by `docs/returns/TEMPLATE.md`.

---

# ✅ AMENDMENT A — 2026-10-16, gates G1 and G2: Phase A's code ACCEPTED; the teachers and the MaxPressure and MAPPO tiers CLEARED, from a run worktree at `5918dad`; Q1 ruled from the registration (the A11 gate runs as registered and fails its coverage criterion by construction); the fixed-time k is NOT ruled yet — a D5 definition check on hz1x1 and grid4x4 comes first, then the author rules

Written against `docs/returns/P9.3-code.md` and `docs/returns/P9.3-preflight.md` at `5918dad`, read from disk, and against the
artifacts themselves. Merge `main` to read it (`git merge main`); nothing here is relayed.

## A.1 What the coordinator verified, this session, from disk (not from the packets)
- **The branch:** `task/p9.3-a27c` at `5918dad`, 8 files, every one added (`git diff --stat main...5918dad`); no frozen file.
- **The tests, re-run:** `tests/test_hz4x4_ladder.py` with `RLTRAFFIC_DRAWS`, `RLTRAFFIC_CORPUS_V11` and `RLTRAFFIC_OUTPUT_ROOT` set,
  at `5918dad`: **60 passed in 34.12 s.**
- **A mutant of the coordinator's**, in a scratch git copy beside an unmutated control copy (a kill = fails in the mutant, passes in
  the control): `protocol_argv` deriving `--max-steps` as the manifest's value − 1 → → **2 killed** (`test_derived_runs_carry_the_grid_settings_with_only_the_declared_substitutions` and
  `test_protocol_argv_reproduces_a_committed_grid_episode`, the array-for-array re-collection); the control passes 60 of 60. A first
  attempt HUNG in both copies (futex waits at about 0 % CPU while about five test processes ran at once) and was not counted;
  re-run alone with `OMP_NUM_THREADS=1`, each copy took 35 s.
- **The teacher config is P2.1's protocol.** It differs from `experiments/configs/p2_1_mappo_nominal_1000.json` in exactly `name`,
  `checkpoint_dir`, `output_dir` (added) and `environments` (a sorted-key JSON diff). The frozen harness trains from scratch whenever
  `from_checkpoint` is unset (`experiments/runner.py`, the cell function: `checkpoint_dir` is a save location only) and composes
  `<checkpoint_dir>/<name>/` and `<output_dir>/<name>/` against the process cwd (`experiments/config.py:355-366`) — so the teachers
  land at `/home/filip/rltraffic/output/checkpoints/p9_3_mappo_nominal_1000/cf_hz4x4__mappo__seed<N>.pt`, the path the packet's guard
  and the collector assume. The grid's teachers were trained the same way (`output/checkpoints/p2_1_mappo_nominal_500/p2_1_mappo_nominal_1000/`,
  P2.1's directory naming; read from `datasets_v11/cf_grid4x4__mappo1000__seed101/manifest.json`).
- **Every run target is absent:** `output/checkpoints/p9_3_mappo_nominal_1000`, `output/p9_3_runs`, `output/p9_3`,
  `datasets_v11/cf_hz4x4__*`. `scenarios/draws/cityflow4x4/` holds 301 entries.
- **The sweep, read record by record** (`docs/data/p9_3_fixed_time_sweep.json`, code `307e5cb`, clean): att_engine prefers k = 3 over
  k = 2 on **all five** tuning draws and att_ours prefers k = 2 on **all five**; `never_entered` is higher at k = 2 on every draw
  (74 / 40, 74 / 46, 43 / 29, 85 / 50, 74 / 39 — means 70.0 against 40.8; entered fraction 0.974 against 0.985).
- **A11's criterion (4) on hz4x4:** `engine_att_reference.GATE_TIERS` (`:180-188`) is the seven corpus tiers `datasets_v11/` carries
  on hz1x1 and grid4x4 (`fixedtime`, `mappo060`, `mappo200`, `mappo500`, `mappo1000`, `maxpressure`, `random`) and `C4_MIN_TIERS = 7`
  (`:174`). hz4x4 will carry four.
- **`main`'s CI** (your Q5): red on exactly one test on both legs since `c8d7754` — run `38790246813`, `1 failed, 3154 passed,
  397 skipped` per leg, the failure `test_t_quotes_every_rule_quotation_appears_at_its_cited_lines`. The three tmux driver tests
  pass on CI; they fail only inside a session sandbox.

## A.2 G1 — ACCEPTED
The review (one mandate, the coordinator's; read-only; 15 minutes) returned **PASS for both runs as they will be launched** — no
blocker, no major. Verified by the reviewer: the six derived commands carry every grid setting with only the declared substitutions
(`--env-config`, `--out-dir`, `--checkpoint`), in `recollect_v11.py:231`'s form; the teacher checkpoint path; the sim configs
resolving to the main tree's scenario files; the imports under `PYTHONPATH=<tree> python -P` resolving to the run tree, with no silent
fallback to the main tree (`ModuleNotFoundError` without `PYTHONPATH`); the barrier (every target absent, every checkpoint and draw
present, before the first subprocess; sequential; abort on the first failure); the convergence criterion reproducing P2.1's 4/5, 3/5,
0/5 by the reviewer's own implementation; `hz4x4_gate_episode` equal to `gate_episode` but for the env-id check.
**Three minor findings, each REQUIRED before the step it guards:**
1. **No code path collects `fixedtime` after a ruling** (`hz4x4_ladder.py:921-945`, `:1104-1108`: k is read only from a sweep artifact
   whose status is CHOSEN, and that artifact is never edited). Build the ruling-file reader A.4 approves — the ruling file names the
   sweep artifact by sha256, the k, and the amendment that carries the author's ruling; the collector refuses a ruling file whose
   sweep digest differs from the committed artifact's — with its tests, before the fixed-time collection.
2. **`verify_run` does not compare `flow_randomizer_params`, `flow_source_path` / `flow_source_sha256`, `metrics` or `epsilon`** with
   their expected values (`:1919-1980`). Add them, tested, before `collect verify` runs on the six runs. (The flow-digest check catches
   parameter, seed and source drift, but not a defect inside `FlowRandomizer` itself: both routes use the class.)
3. **The corpus barrier checks that each teacher checkpoint FILE exists, not that training finished** (`:2068-2083`; `torch.save` is
   not atomic): `p93-corpus` starts only after `p93-mappo`'s log ends `exit=0` (A.8 orders it so), and `collect verify` checks each
   manifest's checkpoint sha256 against the file, as your packet says it does.
**A correction the reviewer made, not a defect:** the grid's `mappo1000` tier gives each seed its OWN 40 draws — seed 101 draws 1–40,
202 41–80, 303 81–120, 404 121–160, 505 161–200 (read by the coordinator from the five manifests' `flow_draw` values) — and your
derived `--flow-draws-range` values match. A27(i)'s and this brief's *"5 seeds × draws 1–40"* misdescribe the grid's protocol; the
registration's governing clause (*"by the grid's own protocol (`datasets_v11/cf_grid4x4__*` manifests)"*) is what you implemented, and
it stands. The misdescription is the coordinator's (`PROJECT_PLAN` §8, 2026-10-16). The merge (G5) still needs its own review.

## A.3 Q1 — RULED FROM THE REGISTRATION; nothing is amended
A27(i) registers both branches: the gate runs on hz4x4 *"before `att_engine` is primary there — if it fails, both definitions are
reported and no single-definition claim is made"*. Criterion (4)'s floor is the registered seven corpus tiers (A11(c), read as
`BRIEF_32` Amendment A1 reads it and as `GATE_TIERS` encodes it) and is **not** re-read per scenario: re-reading a registered floor
once a scenario's tier count is known is the relaxation a registration exists to prevent, and the branch it would open
(`att_engine` primary on hz4x4) is the less conservative one. Therefore:
1. The gate runs as registered (`gate` subcommand): all four criteria, every observed value, the imported `evaluate_scenario`
   unchanged. **Criterion (4) FAILS by construction** (four tiers against seven), and A27(i)'s fail branch applies: hz4x4 has no
   primary definition; every hz4x4 ordering is reported under both definitions, and as DEFINITION-DEPENDENT wherever they differ.
2. Criteria (1)–(3) are still run and reported: they say whether the instrument holds on hz4x4, which the two-definition tables need.
3. The extremes: your reading (the minimum and maximum `entered_fraction` among the gate's own cells, labelled *"the extremes
   available on S"*) is accepted; it cannot change the verdict.
4. The artifact says why in one sentence: *"Criterion (4) cannot be met on hz4x4: its corpus has four behaviour tiers against the
   registered floor of seven."*
The registration was drafted as if the gate could pass on hz4x4. That drafting error is the coordinator's (`PROJECT_PLAN` §8,
2026-10-16); the paper track is told.

## A.4 Q2 — the fixed-time k: NOT RULED HERE; the author rules it after A.7
It fills a gap in a registered procedure — D5's objective is *"the registered primary metric"*, and on hz4x4 there is none (Rule R
not run, and by A.3 it cannot pass) — and it fixes a reference arm of a registered extension, so it is the author's ruling. Two
candidate rules exist, and they are not equivalent:
- **D5's objective as written** (`att_horizon` = `att_ours`, A1, as `docs/data/fixed_time_sweep/README.md` records it): k = 2. It is the
  objective the registered hz1x1 and grid4x4 cycles were tuned on, and on hz4x4 it selects the cycle that keeps more vehicles out of
  the network on every draw — the property A11(a) registered against `att_ours` (*"it rewards a policy for preventing vehicles from
  entering"*).
- **A definition-neutral rule** (the k whose larger shortfall from either definition's own optimum is smallest): k = 3 — 2.35 s
  (0.44 %) above `att_ours`'s optimum, against k = 2's 9.04 s (1.66 %) above `att_engine`'s.
Which of the two is *consistent with the other scenarios* depends on a fact nobody has measured: whether hz1x1's and grid4x4's
registered cycles are also definition-dependent. A.7 measures it; the question goes to the author with its answer. Until an
amendment carries his ruling the fixed-time tier is not collected. Nothing else waits on it: the teachers, MaxPressure and MAPPO
do not depend on k, and Phase B waits on M1. The ruling-file mechanism you proposed (a separate committed file the collector reads
beside the UNEDITED sweep artifact) is approved.

## A.5 Q3, Q4, Q5
- **Q3:** moot — both optima are interior.
- **Q4 (Phase B's "old cells"): approved** — the grid's corresponding cells are carried by reference and per-episode copy from the
  committed artifacts (P5.2's and P9.1's), with source path and digest; nothing is re-rolled on the grid. That is A27(a)'s *"copied
  from the committed artifact where one holds them"*.
- **Q5:** the coordinator's error (A27's splice moved A11 from `PREREGISTRATION.md:397` to `:398`), fixed under `BRIEF_49` (P5.2d),
  not on this branch. Your whole-suite *"4 failed"* is accepted as `main`'s.

## A.6 Power — accepted as recorded
The pre-flight's GO rests on a 28× margin (3,088 s projected against the 86,400 s rule) and on both canaries at speed and
reproducing (0.754 / 0.758 s, `−32648.0 / 247.75089149261333`), which is what §7's canary rule asks of a quoted rate. The Windows
power mode is not readable from the guest; it is recorded as such and not chased. The author keeps the laptop on mains with
Windows' Best performance mode while the runs go.

## A.7 NEW ITEM — the D5 definition check (EXPLORATION: a disclosure measurement, not part of A27; added 2026-10-16)
**Why.** On hz4x4 the two ATT definitions disagree on fixed-time's best cycle on every tuning draw, and the `att_ours` optimum keeps
more vehicles out on every draw. D5 tuned hz1x1 (k = 6) and grid4x4 (k = 1) on `att_horizon` = `att_ours`, before A11 existed;
whether those optima hold under `att_engine` — the primary metric on both since P8.4b — has never been measured. The paper calls
those cycles *"tuned"*, H3's clause 1 compares against them, and A27(f)'s aligned anchor `fixedtime_kcf` IS them. The author's Q2
ruling needs the answer (A.4).
**What.** D5's protocol, re-run with both definitions: the k grids D5's README records (hz1x1: 2, 3, 4, 6, 8, 10, 14 with the shipped
`signal_plan_template.txt`, schedule source as D5 used it; grid4x4: 1, 2, 3, 4, 6, equal split), training draws 1–5 of each
scenario (the main tree's materialised draws), engine seed 1000, one episode per (k, draw), each through
`admission_probe.probe_episode` exactly as your `k-sweep` does on hz4x4 — **reuse its functions with a scenario parameter; do not
copy them**. Per scenario and k: both means and the mean `never_entered` / `entered_fraction`; each definition's argmin; the status
AGREE or DEFINITION-DEPENDENT; each definition's shortfall at the other's argmin.
**T-regress, load-bearing: the per-k mean `att_ours` must reproduce D5's recorded `att_horizon` table to its two printed decimals in
every cell** (hz1x1: 318.81, 286.65, 270.53, 261.86, 265.48, 274.69, 291.47; grid4x4: 206.09, 261.65, 297.54, 290.57, 341.05). One
cell that does not reproduce → STOP and report: the protocol is not D5's, and nothing else from the check is reported.
**Artifact:** `docs/data/p9_3_d5_definition_check.json`, label *"exploration — a disclosure measurement on training draws 1–5; no
registered k, tier, anchor, rho or verdict changes"*.
**Fence:** it re-tunes nothing. Whatever it shows, no registered k changes, and the hz1x1 and grid4x4 fixed-time tiers and every
anchor stay as registered.
**When:** now, in-session, like the hz4x4 sweep (about 60 CityFlow episodes on training draws, one process; it produces no registered
number), committed with one test file for the T-regress and the regret arithmetic. **Cut** before any Phase B item if time binds.

## A.8 The run worktree and the TOKEN (channel (a))
The runs execute from a **detached run worktree at `5918dad`**, `/home/filip/rltraffic-p93-run`, not from your worktree: you merge
`main` after P9.1's M1, which changes `offline/collect.py`, and `p93-corpus`'s subprocesses import the code tree named on
`PYTHONPATH` at their start — a merge landing mid-run would make the tiers "collected by different code" (J1(c)). Your worktree may
take merges of `main` at any time. The commands are your packet's §4 with `rltraffic-p93` replaced by `rltraffic-p93-run` and `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
set for every process of the tree (the harness already pins torch's threads inside each cell, `experiments/runner.py:limit_torch_threads`;
the variables pin every other OpenMP/MKL user too — today's hung test processes, A.1, waited on futexes the way an unpinned
forked worker does; the coordinator created the run worktree, clean, at `5918dad`):
- **`p93-mappo`** (5 workers, ≈ 52 min projected): CLEARED.
- **`p93-corpus`** (`--tiers maxpressure mappo1000`, 1 worker): CLEARED, once `p93-mappo`'s log ends `exit=0`.
- The fixed-time tier: after the author's ruling (A.4). `random_fresh`: after M1, last.
**CPU:** with `p92-sumo` and `p92-shift` (3 + 3 workers) this is 11 CPU workers, inside §10's cap of 12 while P9.1 trains on the GPU.
**Liveness** (the harness's pooled workers log nothing until they finish): `ps -eo pid,pcpu,etime,args | grep '[e]xperiments/run.py'`
from a plain terminal shows the parent and five workers; workers near 100 % CPU are progressing. A worker near 0 % CPU for more
than 10 minutes, or no checkpoint after 2 h, is a hang: the author says "P9.3 hung" and nothing is re-run.

## A.9 Next, in order
1. A.7 now (in-session) → its artifact and test committed → the coordinator reads it and puts Q2 to the author.
2. After `p93-mappo` `exit=0` and `p93-corpus` `OK: 6/6 run(s), 400 episodes`: `teachers` → `docs/data/p9_3_teachers.json`;
   `collect verify` on the six runs.
3. The fixed-time tier after the amendment carrying the author's ruling; `random_fresh` after M1 (`git merge main`, then verify
   `random_rng_scheme` against P9.1's merged code before use, as your plan's A1 says).
4. Then the gate (A.3) and `declare`; then Phase B's plan appended to `docs/plans/p9.3.md` → G3b (its pre-flight → token).

---

# ✅ AMENDMENT B — 2026-10-16: A.7 ACCEPTED (grid4x4 AGREE at k = 1; hz1x1 DEFINITION-DEPENDENT, k = 8 by `att_engine` against the registered k = 6); the teachers and the two tiers DONE; the fixed-time k goes to the author now; the gate and Phase B need the stall remedy before they run

Written against `docs/returns/P9.3-A7.md` (`5c06e14`), `docs/data/p9_3_d5_definition_check.json` (`06d2adf`, code `d7dd8c8`),
`docs/data/p9_3_teachers.json` (`91a3af3`), `87b0406` and the main tree's corpus, read from disk. Merge `main` to read it.

## B.1 A.7 — verified by the coordinator from the artifact's own records
- **The T-regress:** the per-k mean `att_ours`, recomputed by `math.fsum` from the 60 stored records, reproduces D5's printed table to
  two decimals in all 12 cells (hz1x1: 318.81, 286.65, 270.53, 261.86, 265.48, 274.69, 291.47; grid4x4: 206.09, 261.65, 297.54,
  290.57, 341.05).
- **grid4x4: AGREE** — both definitions choose k = 1, the registered cycle; `never_entered` is 0 at every k.
- **hz1x1: DEFINITION-DEPENDENT** — `att_engine` is best at k = 8 (374.56) and `att_ours` at k = 6 (261.86, the registered cycle);
  `att_engine` at k = 6 is 13.55 s (3.62 %) above its optimum, `att_ours` at k = 8 is 3.62 s (1.38 %) above its own; the `att_ours`
  optimum admits fewer vehicles (`never_entered` 164.8 against 150.2).
- **Accepted as a disclosure measurement, fenced:** no registered k, tier, anchor, ρ or verdict changes. The paper's baselines
  paragraph names the objective every cycle was tuned on (`att_horizon`, D5) and the `att_engine` optimum beside it on each scenario.
- **The hang you diagnosed is `DEFERRED` 104's** (P5.3c's note, 2026-09-30); the ruling then stands now: `CityFlow/` is not patched;
  the remedy is at the caller (B.3).

## B.2 The teachers and the corpus
- **Teachers:** 5 of 5 trained (`exit=0`), **0 of 5 converged by RUNSPEC_01 §4's criterion at 1,000 episodes** (the final window's
  relative change 0.050–0.079 against 0.05, every seed's return still rising); labelled MAPPO@1000 by measured return, as registered
  (§5) — the paper says 0 of 5 beside the grid's 3 of 5 (P2.1). `train_sec` 3,726–3,785 s per seed, an uncontrolled clock.
- **Corpus:** `p93-corpus` ended `OK: 6/6 run(s), 400 episodes`, `exit=0`: `cf_hz4x4__maxpressure` 200 episodes and
  `cf_hz4x4__mappo1000__seed{101..505}` 40 each. **Next, in-session (read-only):** `collect verify` on the six runs, with `87b0406`'s
  stronger `verify_run` (randomiser parameters, flow source, metrics and epsilon compared) and the checkpoint digests.
- **`87b0406` (A.2's findings 1 and 2) accepted** as committed; the ruling file is the route for B.4.

## B.3 The stall remedy — required before the gate (G3) and Phase B's evaluation (G3b) run
The gate's observer episodes and Phase B's evaluation free their engines right after the last step — the path the hang strikes.
Before either runs: every process that rolls CityFlow episodes in them is cut by a watchdog when one episode makes no progress for
600 s (`faulthandler.dump_traceback_later(600, exit=True)` re-armed per episode, the dump in the log) and restarted from the records
on disk by its driver, at most 20 times, each restart logged; any other non-zero exit stops as now. Tests (red first): a blocked fake
episode is cut and the run completes on restart; a raising one is not restarted; a healthy run longer than 600 s in total is never
cut; the cap stops a run that hangs every time. One mutant per item. Collection (`offline.collect`) needs none: it writes its `.npz`
before freeing the engine, and the 400-episode corpus ran clean.

## B.4 The fixed-time k — put to the author today, with A.7's answer
The question goes to the author in the coordinator's message of 2026-10-16, with this recommendation: **k = 2, by D5's objective as
written (`att_horizon`)** — because A.7 shows the registered hz1x1 cycle (k = 6) is itself `att_ours`'s optimum where `att_engine`
prefers k = 8, so any other rule on hz4x4 would treat hz4x4 differently from hz1x1; one tuning objective on all three scenarios, with
the `att_engine` optimum disclosed beside each (grid4x4 the same, hz1x1 k = 8, hz4x4 k = 3), is the version a reader can check. The
ruling lands as Amendment C; then `record-ruling` (in-session) and the fixed-time collection under the author's token.

## B.5 Order from here
1. `collect verify` (B.2) → its record committed.
2. B.3's stall remedy, with its tests and mutants.
3. After Amendment C: `record-ruling`, then the fixed-time tier (the author's token).
4. After M1 merges (P9.1's M1 is FIX FIRST today, `BRIEF_46` Amendment B): `git merge main`; `random_fresh` collected last; then the
   gate (B.3 in place); then Phase B's plan appended to `docs/plans/p9.3.md` and its pre-flight (G3b) with the measured GPU cost. The
   GPU is P9.1's campaign's for about 40 h once it starts; G3b states how Phase B's trainings fit before 2026-10-23 (after P9.1's, or
   beside them where memory and the never-two-DT-trainings rule allow) — the hard stop is real.

---

# ✅ AMENDMENT C — 2026-10-16, the author's ruling: hz4x4's fixed-time cycle is k = 2

**The author's ruling (2026-10-16), quoted:** *"hz4x4 fixed-time cycle: k = 2, as you recommend (one tuning rule on all three
scenarios; the att_engine optimum k = 3 disclosed beside it)."*

## C.1 What it means
- The hz4x4 fixed-time tier is collected at **k = 2**, the argmin of D5's objective as D5 recorded it (`att_horizon` = `att_ours`, mean
  over training draws 1–5), the same objective the registered hz1x1 (k = 6) and grid4x4 (k = 1) cycles were tuned on.
- **Disclosed beside it, everywhere the cycle is named:** `att_engine`'s optimum on hz4x4 is k = 3 (544.09 against k = 2's 553.12,
  1.66 % above it); on hz1x1 it is k = 8 (the registered k = 6 3.62 % above it); on grid4x4 it is k = 1, the registered cycle. The
  `att_ours` optima keep more vehicles out on hz4x4 (`never_entered` 70.0 against 40.8 at k = 3) and on hz1x1 (164.8 against 150.2).
- The sweep artifact `docs/data/p9_3_fixed_time_sweep.json` stays unedited; this amendment is the ruling it was waiting for.

## C.2 What you do
1. `git merge main` (this amendment on disk), then `record-ruling` with `--k 2` and `--amendment "AMENDMENT C — 2026-10-16"` (the
   exact string; it occurs in this heading), against the committed sweep artifact. Commit the ruling file it writes, with a test that
   `fixed_time_k_from_ruling` returns 2 from it and refuses it against a sweep whose digest differs.
2. Say **"P9.3 ruling recorded"** with that commit and the exact `p93-fixedtime` command — the fixed-time tier only (draws 1–200, the
   grid's protocol, `--ruling` naming the committed file), from a run tree at that commit. The coordinator verifies the file, creates
   the run tree and hands the author the token. Collection writes its `.npz` before the engine is freed, so B.3's remedy is not needed
   for it.

---

# ⛔ AMENDMENT D — 2026-10-27: THE HARD STOP HAS FIRED — point 8 is DROPPED, by A27(i) as registered

A27(i): *"hard stop 2026-10-23: not merged by then, dropped, a named limitation."* The branch was not merged by 2026-10-23 (the last
commit is `ee8c6d3`, 2026-10-16; the author was away 2026-10-16 to 2026-10-27, and a Windows update rebooted the machine). The rule
is applied as registered; it is not re-opened.
- **Stop all P9.3 work.** No fixed-time collection, no `random_fresh` tier, no gate, no Phase B. Nothing of this branch is merged or
  cited for hz4x4; the paper names hz4x4 as a limitation in the words of A27(i) and this brief's header.
- **What stays on disk, untouched:** the branch `task/p9.3-a27c` at `ee8c6d3`, the run tree `/home/filip/rltraffic-p93-run`, the
  materialised draws `scenarios/draws/cityflow4x4/`, the corpus `datasets_v11/cf_hz4x4__*`, the teachers
  `output/checkpoints/p9_3_mappo_nominal_1000/`.
- **A.7 (the D5 definition check on hz1x1 and grid4x4) is NOT point 8:** it measures the registered scenarios' fixed-time cycles
  and the paper's baselines paragraph needs it. Whether its artifact and code reach `main` separately is the coordinator's next
  ruling (`PROJECT_PLAN` §8, 2026-10-27); until then do nothing with it.
- Write nothing further on the branch except, if asked, a closing note.

---

# ⚖️ AMENDMENT E — 2026-10-27, THE AUTHOR'S RULING: Amendment D is REVOKED; point 8 RESUMES with a new hard stop of 2026-11-07 (deviation D7); the fixed-time tier CLEARED; your order to the merge

## E.1 The ruling
**Quoted (2026-10-27):** *"Author's ruling: point 8 (hz4x4) is NOT dropped. I was away for ten days and was not asked. The 23 Oct
hard stop was a feasibility time-box, and no hz4x4 learner number exists, so extending it selects on nothing. Record it as a
deviation in A27_DEVIATIONS (like D1) with a new hard stop of 2026-11-07, revoke BRIEF_48 Amendment D, and resume P9.3 from where it
stopped: the fixed-time collection at k = 2, random_fresh, the stall remedy, then Phase B on the GPU after P9.1's campaign."*
- **Amendment D is revoked in full.** The branch is live again; nothing of D stands.
- **The hard stop is 2026-11-07**, with A27(i)'s consequence unchanged: not merged to `main` by then → dropped, a named limitation.
  The deviation is D7 in `docs/notes/A27_DEVIATIONS.md` (it supersedes D2; D5 and D6 are live again).
- Amendments A, B and C stand: the label (conditional generalizability extension), the k = 2 ruling, the A11 gate's fail branch
  (A.3), the stall remedy (B.3) with E.3's corrected values.

## E.2 The fixed-time tier — CLEARED (the author's run, `p93-fixedtime`)
Verified by the coordinator before clearing it: the run tree `/home/filip/rltraffic-p93-run-ft`, detached at `0426a73` (the commit
carrying the ruling file), clean; its `offline/collect.py` is the one that collected the `maxpressure` and `mappo1000` tiers (your
branch has not merged M1 yet). The read-only `collect plan --tiers fixedtime --ruling …` from the main tree derives ONE run, 200
episodes, nothing missing, and its command equals the grid fixed-time manifest's settings field by field (policy, episodes,
base seed 1000, 360 steps, Δt 10, acyclic, queue-length rewards at global weight 0.0, the three state features, the randomiser
30.0 / 0.1 / 1.0) with exactly three substitutions: `--env-config` the main tree's `configs/sim/cityflow4x4.json`, `--out-dir`
`datasets_v11/cf_hz4x4__fixedtime`, `--fixed-time-k 2` (the grid's is 1); draws 1–200. The target is absent. The author runs the
command your note gave (the coordinator's message of 2026-10-27 carries it whole). If its log shows no new episode line for ten
minutes, the author reports "p93-fixedtime hung" and nothing is re-run without a ruling.

## E.3 Your order to the merge
1. `git merge main` (it brings P9.1's M1 collector and training flags, P9.2's merge, Amendments D and E) and re-run your task's test
   files (one pytest process, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).
2. **The stall remedy's values, corrected (the coordinator's error, `BRIEF_47` Amendment D, D.3; `BRIEF_46` Amendment B.2):**
   `episode_watchdog` cuts after **120 s**, `run_with_restarts` stops after **60** restarts — the defaults in your code, pinned by a
   test. 600 s and 20 were set without computing the expected hangs.
3. After `p93-fixedtime` ends `OK`: `collect verify` on `cf_hz4x4__fixedtime` (its write-once record committed).
4. **`random_fresh`:** read M1's merged `offline/collect.py` and confirm your plan's A1 (the manifest key `random_rng_scheme`, value
   `per-episode`) and Q1's within-draw episode index (`BRIEF_46` Amendment B, B.2); `collect plan --tiers random_fresh` read-only;
   then say **"P9.3 random_fresh ready"** with the commit and the command. The coordinator creates its run tree and clears it
   (Amendment F). After it: `collect verify`, with A27(d)'s 200 distinct per-intersection action sequences.
5. **The gate** (A11, all four criteria with observed values; criterion 4 fails by construction, A.3) under your stall remedy, then
   `declare`.
6. **Phase B:** its plan appended to `docs/plans/p9.3.md`; `offline/hz4x4_phase_b.py` tests first, one mutant per number-computing
   path; M1's flags (`--iql-target-dropout off`, the matched batches) and `att_rederivation.run_campaign(..., factory_builder=…)`
   for P8.4b's route; the evaluation under the stall remedy. **G3b** — the pre-flight with each arm's GPU cost: P9.1's measured grid
   costs as the estimate (the same lattice, the same 200-episode tiers), one short timing per arm taken only when no P9.1 training
   runs. The token request in one message.
7. **The GPU:** Phase B's trainings start only after P9.1's campaign has finished its trainings (they may overlap P9.1's
   evaluation stage); never two trainings on the GPU at once.
8. G4 (artifact and packet) → G5 (one review, at most one fix round) → **merged by 2026-11-07**, or dropped.

---

# ✅ AMENDMENT F — 2026-10-27: E.3 steps 1–4 ACCEPTED; `random_fresh` CLEARED from the run tree `/home/filip/rltraffic-p93-run-rf` at `7b6b59a`; one audit before G4

## F.1 Verified by the coordinator, from disk
- **E.3.1:** `0eddcdd` is a two-parent merge (`ee8c6d3`, `main` at `8c5c8fe`).
- **E.3.2 (`e3ccce1`):** 120 s and 60 restarts are the code's defaults. The pin was rewritten with the spec it pins and ran red on
  the old value first: ACCEPTED (the spec changed, the test follows it, and `docs/returns/P9.3-E.md` discloses it).
- **E.3.3 (`30ab2f2`):** `docs/data/p9_3_corpus_verify_fixedtime.json` read: `all_verified` true, 1 run, 200 episodes, 200 flow
  digests, collected by `0426a73`, code `e3ccce1` clean. The manifest read: `fixed_time_k` 2, `equal_split`, draws exactly 1–200
  (the grid's range), engine seed 1000. The author's log ends `OK: 1/1 run(s), 200 episodes` and `exit=0`. **The fixed-time tier is
  in the corpus.**
- **E.3.4, the derivation re-done independently:** the run tree `/home/filip/rltraffic-p93-run-rf`, detached at `7b6b59a` (code =
  `b38e43e`), clean before and after. The read-only `collect plan --tiers random_fresh`, run from the main tree with `PYTHONPATH` on
  the run tree exactly as the run will be, derives ONE run, `cf_hz4x4__random_fresh`, 200 episodes, no checkpoint, every draw
  materialised:
  `… -m offline.collect --backend cityflow --env-config /home/filip/rltraffic/configs/sim/cityflow4x4.json --policy random
  --episodes 1 --base-seed 1000 --max-steps 360 --delta-time 10 --control-mode acyclic --global-reward-fn queue_length
  --local-reward-fn queue_length --global-reward-weight 0.0 --state-features lane_vehicle_count lane_waiting phase_onehot
  --flow-jitter-sigma 30.0 --flow-thin-p 0.1 --flow-volume-scale 1.0 --out-dir /home/filip/rltraffic/datasets_v11/cf_hz4x4__random_fresh
  --random-rng per-episode --flow-draws-range 1 201`.
  The coordinator compared it, field by field, with the run metadata of P9.1's `cf_grid4x4__random_fresh` manifest (`git_hash`
  `140149e`). The following are EQUAL: backend; policy `random`; 1 episode per draw; base seed 1000; 360 steps; Δt 10; acyclic;
  `queue_length` / `queue_length` at global weight 0.0; the three state features; the randomiser 30.0 / 0.1 / 1.0;
  `random_rng_scheme` `per-episode`; draws 1–200 (the grid's are exactly 1–200); no ε, no metrics, no checkpoint. There are two
  substitutions: `--env-config` (`cityflow4x4.json`, i.e. `roadnet_4X4.json` with the Hangzhou flow) and `--out-dir`. The target is
  absent. Your test `test_the_random_fresh_command_is_the_grid_random_fresh_protocol_under_the_merged_collector`, run in the run
  tree: **1 passed**; the tree still clean.

## F.2 `random_fresh` — CLEARED (the author's run, `p93-random-fresh`)
CPU, one process, about the fixed-time tier's time. The author runs the command in the coordinator's message of 2026-10-27, which
carries it whole. It is done when its log ends `OK: 1/1 run(s), 200 episodes` and `exit=0`. If no new episode line appears for ten
minutes, the author reports "p93-random-fresh hung"; nothing is re-run without a ruling, and a partial target directory is never
resumed by hand. **After it:** `collect verify --tiers random_fresh --out docs/data/p9_3_corpus_verify_random_fresh.json`, which checks
`random_rng_scheme == "per-episode"` and A27(d)'s 200 distinct action sequences on every one of the 16 intersections; commit the
record. Then E.3.5 and E.3.6, in order.

## F.3 One audit before G4 — the class that turned `main` red today (`BRIEF_47` Amendment E)
P9.2's driver `cd`s into a hard-coded `/home/filip/rltraffic`. On this machine every test passed; on CI's runners five failed. Your
code carries the same literal twice in `offline/hz4x4_ladder.py`: `MAIN_TREE`, and the default of `collect --python`. In the G4
packet, list each test that reaches either literal and say how it behaves on a host without that path (overridden, skipped with a
named reason, or not reached), and fix any that would fail there. CI runs at the merge; a red run there costs a fix round inside the
hard stop.

## F.4 The GPU now
No P9.1 training runs: its campaign is not cleared yet (`BRIEF_46` Amendment B.2.1). E.3.6's one short timing per arm may be taken
whenever the GPU is idle like this. Phase B's trainings still wait until P9.1's trainings have finished (E.3.7).

---

# ✅ AMENDMENT G — 2026-10-27: `random_fresh` verified; the gate and the declaration ACCEPTED; Phase B's plan ruled (Q-B1 (a), through P9.1's M1b; C1 and C3 as planned; G3b's cost basis)

## G.1 Verified by the coordinator
- **`p93-random-fresh`:** the log ends `OK: 1/1 run(s), 200 episodes` and `exit=0` (17:26).
- **The verify record** `docs/data/p9_3_corpus_verify_random_fresh.json` (`f79ff16`): code `c5bf6e3`, clean; collected by `7b6b59a`;
  200 episodes, 200 flow digests; `all_verified` true.
- **The gate** (`887a65d`, `docs/data/p9_3_engine_att_gate.json`): code `f79ff16`, clean; one attempt of 23.3 s, no cut.
  - Outcome: `passed` false, `failed_criteria` [c4], `primary` null.
  - That is A27(i)'s fail branch, as Amendment A, A.3 foresaw (`A27_DEVIATIONS` D6, live again).
- **The declaration** (`8675040`): code `887a65d`, clean.
  - **A third route, the coordinator's:** for each tier, the mean over its 200 raw episodes of the final `att_per_step` value.
  - It equals the declared `ladder_att` under `==` on all four tiers: mappo1000 330.0151, maxpressure 370.2910, fixedtime 534.1918,
    random_fresh 626.4671.

## G.2 Rulings on Phase B's plan (`fc0fda2`, §10)
- **Q-B1 → (a).** Your finding C2 holds, and it reaches P9.1's campaign too.
  - The coordinator reproduced it at P9.1's campaign commit: there it would have stopped P9.1's five grid `random_fresh` `dt_nomix`
    trainings at runs 231–235 of 255, about 35 GPU-hours into the campaign.
  - The fix is P9.1's, in `tier_sweep.train_tier_dt`: `BRIEF_46` Amendment C, M1b, on `task/p9.1-m1b`.
  - When the coordinator says M1b is on `main`, run `git merge main` and call `train_tier_dt` unchanged. No mirror of a trainer.
- **C1 (the observer route) and C3 (the grid-bound CLIs and the report): accepted as planned.** They are new functions in your module,
  every lower function stays unchanged, and each mirror is held to its original on the grid by a test (§10.6).
- **A-B1, A-B3 and A-B4: accepted.**
- **A-B2: ruled.** hz4x4 has no registered statement. Its leaders block uses the grid's DESCRIPTIVE rule under both definitions, with no
  primary, `definition_dependent` flagged and `registered: false`.
- **G3b's cost basis:** P9.1's measured grid costs, since hz4x4 has the same lattice and the same 1,152,000 decisions (≈ 31 GPU-h
  serial).
  - Take the 2,000-step hz4x4 timings only while no P9.1 training runs.
  - If P9.1's campaign has started by the time your code is ready, skip them (never two trainings on the GPU). Report the CPU seconds
    per evaluated episode per route instead.

## G.3 Next
Phase B's code, tests first, as §10 plans it; then the G3b pre-flight. Your trainings start after P9.1's last training (E.3.7). The
hard stop is 2026-11-07.

---

# ⚖️ AMENDMENT H — 2026-10-28: the DT trainer mirror STANDS (G.2's Q-B1 (a) withdrawn; M1b stays one line); G3b's pre-flight VERIFIED; the token waits for one smoke over every distinct cell (§7 CI rule (4)), run from the run tree

Your G3b note (`9b6896b`; branch tip `14a6ba2`) crossed Amendment G. G told you to build Phase B and to call `train_tier_dt`
unchanged once M1b is on `main`. You had already built, tested and timed Phase B around a mirror of that trainer. This amendment
replaces G.2's Q-B1 ruling and G.3; everything else in G stands.

## H.1 Verified by the coordinator, from disk
- **Your six files:** in a scratch clone at `14a6ba2`, one process, one thread: **141 passed, 0 skipped** (301.7 s). The live hz4x4
  tests ran.
- **The mirror against the original:**
  - `offline/tier_sweep.py` is identical on `main` and at `14a6ba2`.
  - The coordinator's diff of `hz4x4_train_tier_dt` against `train_tier_dt` (both extracted by `ast`) is exactly your list:
    - the name, and the two parameters `scenario_id` and `tiers`;
    - the docstring and a local `import numpy as np`;
    - six `ts.`-qualified names: `DT_METHODS`, `assert_process_regime`, `assert_writable`, `N_HEAD_BY_METHOD`, `warmup_for`,
      `lr_multiplier`;
    - the tier check and the recorded id.
  - Its free names are builtins and `ts` only (`symtable`). So every constant and helper is `tier_sweep`'s or `spatial_mixing`'s at
    call time.
- **The bit-for-bit test runs on a synthetic tier on CPU, with no corpus, so CI runs it.** The coordinator's mutant MX1 (the mirror's
  warm-up one step longer) is KILLED by it.
- **The scenario id:**
  - All eight hz4x4 tier manifests record `run_metadata.scenario_id` `cityflow4x4`: `maxpressure`, `fixedtime`, `random_fresh` and the
    five `mappo1000` seeds. `scenario_id_of` returns that id, and evaluation uses it (`hz.SCENARIO_ID`).
  - Mutant MX2 is your M3 (the id taken from the grid's constant). The live test KILLS it; the synthetic test lets it SURVIVE. That
    confirms your disclosed gap.
- **Two more mutants, both KILLED:**
  - MX3: the `==` stop compares route (ii) with itself.
  - MX4: the queue carries on after a failed cell.
  The module was restored byte for byte after each.
- **The pre-flight:**
  - The five timing records (`3ea8418`, clean; CUDA; `maxpressure`, seed 101, 2,000 steps) equal your table.
  - From them: 20 cells per arm at 20 × the measured seconds, plus 20 × each arm's setup, is **30.7 h**.
  - From the 30 rows of your evaluation timing (`ea73b6c`, clean): **4.17 CPU-h per tier of learners, 2.00 for the references,
    18.66 in all.** Your 18.65 rounds per arm first.
- **The references against the gate:** the 12 gate cells equal the timing's `behaviour@<tier>` rows on `att_engine`, `att_ours` and
  `never_entered` under `==`: **12 of 12**.
- **F.3, run early: the host-absent simulation.**
  - In a second clone at `14a6ba2`, every `/home/filip/rltraffic` in all 82 tracked `.py` and `.sh` files was pointed at a path that
    does not exist. That is CI's condition; the data files keep their literals, as on CI.
  - Your six files: **121 passed, 20 skipped, 0 failed.** Every skip names the corpus, draws or output it lacks.
  - So F.3 holds for your code as it stands, and the campaign commit needs no F.3 fix.

## H.2 Rulings
1. **C2: the mirror STANDS. G.2's Q-B1 (a) — "call `train_tier_dt` unchanged. No mirror of a trainer." — is WITHDRAWN.**
   - **The coordinator's error.** G.2 rested on the tier check alone. The trainer also writes the grid's `SCENARIO_ID` into every
     checkpoint (`tier_sweep.py:939`). That line was there to be read, and the coordinator did not read past the check. Your
     end-to-end test found it before anything ran.
   - **Why the mirror, and not an M1b that also carries the id:**
     - M1b is on P9.1's critical path, and it is fixed as one line (`BRIEF_46` C.3). Every grid DT number passes through that
       trainer. Changing its signature for hz4x4's sake adds risk there and gives the grid nothing.
     - Your code is built and verified. A wider M1b would mean rewriting it and its mutants.
     - The mirror cannot drift on a value, and a drift in form fails the bit-for-bit test on every CI run.
   - **Consequences:**
     - You no longer wait for M1b, and you do not merge `main` for it.
     - M1b goes ahead exactly as `BRIEF_46` C.3 says.
     - M1b leaves your bit-for-bit test as it is. The test compares on the grid's `random`, which the trainer accepts both before and
       after M1b.
   - **The disclosed gap is accepted.** M3 is guarded only by the live test. A regression there fails loudly at the first hz4x4 DT
     evaluation, and H.3's smoke runs that evaluation for every tier before the token.
   - **For G4's packet:** one sentence for the methods notes. hz4x4's DT trainer is the grid trainer's body with its two grid
     constants made parameters, held to it bit for bit on CPU by a test.
2. **The two rewritten tests are accepted.**
   - Both pinned a design that C2 invalidated, and neither was weakened.
   - Their new forms assert more: the two new parameters, `random_fresh` accepted, and an unknown name refused before anything is
     read or written.
3. **A-B2 stands as ruled in G.2.** Your note's proposal matches it, including the references' levels beside the leaders, not
   competing for the lead.
4. **The campaign's code is frozen at the token's commit.**
   - From the token on, `offline/hz4x4_phase_b.py` and `offline/hz4x4_ladder.py` stay byte-identical to the run tree's.
   - `report` and `old-cells` go in a NEW module (for example `offline/hz4x4_report.py`). This changes plan §10's layout.
   - **Why:** G4 must show that the code which wrote the records is the code the branch carries. With the two modules frozen, that is
     one `git diff --quiet <run commit> <tip> -- offline/hz4x4_phase_b.py offline/hz4x4_ladder.py`. With `report` inside the trained
     module, that diff is never empty.

## H.3 ⛔ The token waits for one step: a smoke over every distinct cell, from the run tree (§7 CI rule (4))
**Rule (4)** was added on 2026-10-27, after your finding for P9.1. Before a campaign token, every DISTINCT path the campaign will
execute has run end to end through the real code: each (tier, arm) training cell and each evaluation cell. Only the size is reduced,
and the output goes to a scratch root.

**What has run so far:**
- Training:
  - the five arms on `maxpressure` (the timings: CUDA, `3ea8418`, before the mirror existed);
  - `bc_top10_perix_m` on `maxpressure`, five seeds, through the queue (working tree uncommitted).
- Evaluation:
  - the five `maxpressure` learners and the five references on draws 1000–1002 (`ea73b6c`);
  - `behaviour@maxpressure` through the driver (working tree uncommitted).

**What has not run:**
- the fifteen training cells on `fixedtime`, `mappo1000` and `random_fresh`;
- their fifteen learner evaluation cells;
- the committed queue and driver as such.

A refusal confined to one tier is exactly the class that would have stopped P9.1 about 35 GPU-hours in.

**The run tree exists: `/home/filip/rltraffic-p93-run-b`, detached at `14a6ba2`.**
- The coordinator created it at 00:03 on 2026-10-28, and it is clean.
- Its two Phase B modules have sha256 `f53613b4…` (`hz4x4_phase_b.py`) and `bd305b2c…` (`hz4x4_ladder.py`), equal to the commit's.
- The smoke runs FROM it, so it proves the very code the campaign will run. You keep working in your own worktree meanwhile.
- `<SMOKE>` below is a new directory in your scratchpad: not inside a linked worktree, and not the main output root (which refuses two
  steps).

1. **The training smoke.** This is the queue itself, with only the steps and the output root changed:
   ```
   cd /home/filip/rltraffic && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/filip/rltraffic-p93-run-b /home/filip/rltraffic/.venv/bin/python -P -m offline.hz4x4_phase_b train-queue --gradient-steps 2 --output-root <SMOKE>
   ```
   - **All 100 cells, not one seed per cell.** Your evaluation driver evaluates every seed of a cell, so the evaluation smoke needs
     every seed's checkpoint.
   - **Device:** CUDA, the campaign's device, outside your sandbox as your timings ran, while no P9.1 campaign training is on the
     GPU. Otherwise add `--device cpu` and say so; the five timings already ran each arm's code on CUDA.
   - **Time:** about 1.5 h, at about 53 s of setup per process.
   - **Done when** the queue prints `completed` with `trained` 100.
2. **The evaluation smoke.** This is the driver over all 25 cells × 5 seeds on draw 1000, on CPU, in four shards:
   ```
   for i in 0 1 2 3; do (cd /home/filip/rltraffic && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/filip/rltraffic-p93-run-b /home/filip/rltraffic/.venv/bin/python -P -m offline.hz4x4_phase_b evaluate --label smoke --draws 1000 --output-root <SMOKE> --shard $i --of 4) & done; wait
   ```
   - It rolls 125 episodes, each by both routes under `==`: about 13 CPU-minutes.
   - Its numbers come from two-step policies. They are never quoted, and they enter no ledger.
3. **A refusal or a failure stops the smoke.** Report it; do not work around it. A code fix means a new commit, a new run tree from
   the coordinator, and the smoke again from that tree.
4. **The record:** `docs/returns/P9.3-smoke.md`, short, committed. It holds:
   - the run tree and its commit, and both commands verbatim;
   - the queue's last line and each shard's last line;
   - the counts: 100 training records and 100 checkpoints; 125 evaluation records; 0 refusals;
   - every record's `git` at `14a6ba2`, not dirty;
   - the seconds per cell by arm (minimum, median, maximum).
5. **Then say "P9.3 smoke ready"** with the note's commit. The coordinator reads the smoke root. In Amendment I it clears all three
   commands from this same run tree, with the worker counts:
   - `p93-eval-ref` at once;
   - `p93-train` after P9.1's last training;
   - each `p93-eval-<tier>` once that tier's 25 cells exist.

## H.4 Meanwhile
- **`git merge main`.** It is docs only: Amendments G and H, and the CI ceiling commit.
- **Write `report` tests first, in the new module (H.2-4).** A-B2 is ruled, and the grid's matched cells come by reference (Q4).
- **F.3 in G4's packet:** re-run H.1's host-absent simulation on the G4 tip, with the report's tests included, and list what it
  shows.
- **GPU order, unchanged:** P9.1's trainings (≈ 40.5 h), then yours (≈ 30.7 h). The hard stop is 2026-11-07.

## H.5 2026-10-28, with P9.1's clearance (`BRIEF_46` Amendment D, D.2-2)
If P9.1's campaign starts while H.3's training smoke is on the GPU, let the smoke finish there. Do not stop it, and do not switch it to
the CPU. Its two-step cells take a few GPU-seconds each. In your smoke note, disclose the overlap: P9.1's start is the first line of
`/home/filip/rltraffic/output/p9_1_runs/campaign_capture.txt`. Two CAMPAIGN trainings at once stay forbidden: `p93-train` waits for
P9.1's `train: ok`.

## H.6 2026-10-28, later: the author's rulings — no hard stop (D8); `p93-train` in SESSIONS (FIX FIRST, before the token)
- **The 2026-11-07 hard stop is withdrawn** (`A27_DEVIATIONS` D8). Point 8 has no date; it merges when its gates pass.
- **Every long run runs in sessions of a few hours,** which the author starts and stops. Each session ends cleanly between units of
  work (`PROJECT_PLAN` §7 (5); `CLAUDE.md` §5). `p93-train` (≈ 30.7 h) is one such run. Build the following, tests first:
  1. **`train-queue --hours H`.** At the start, plan the session: the next pending cells whose planned seconds sum to at most H
     hours. The first cell is always included. Print the plan, train exactly those cells, and exit 0 with a summary: the cells
     trained, the next pending cell, and the planned hours left.
     - The planned seconds are your G3b timings per arm, scaled to 40,000 steps, plus setup, labelled as planning estimates.
  2. **`--max-cells N`**, for tests and smokes.
  3. **A stop file, `output/p9_3/STOP`.** It is checked before each cell. When it exists, the session ends after the cell in flight.
     The code never deletes it, and a session refuses to start while it exists.
  4. **Tests first.** Mutants: the plan ignored; the stop file ignored; a cell started after it.
  5. **Proof that nothing else changed.** Every function on the training and evaluation paths must be identical to `14a6ba2`'s, by
     `ast`: `train_cell`, `hz4x4_train_tier_dt`, `hz4x4_tier_parts`, `hz4x4_baseline_inputs`, `evaluate_episode`, `eval_factory`,
     `run_episodes` and what they call in your modules. Then H.3's smoke at `14a6ba2` stands for them.
  6. **A queue smoke** on a fresh scratch root, with two-step cells: `--max-cells 2`, then the stop file, then a resumed session.
  7. **Then say "P9.3 sessions ready"** with the commit, beside "P9.3 smoke ready" for H.3. The coordinator creates the run tree at
     that commit and clears the token in Amendment I, stating each session's hours first. H.2-4's freeze starts at that token.
- **The evaluation runs are short and stay as planned:** `p93-eval-ref` takes ≈ 0.5 h and each `p93-eval-<tier>` ≈ 0.7 h.
- **H.3's smoke:** finish it as it is, from the run tree at `14a6ba2`. H.5 stands.

---

# ⚖️ AMENDMENT I — 2026-10-28, evening: H.3's smoke and H.6's sessions VERIFIED; `p93-train` CLEARED in sessions from the run tree `/home/filip/rltraffic-p93-run-c` at `4dcbedc`; D, E and A-B5 ruled; two statements in your notes CORRECTED

Your notes: `b71d3d9` (H.3's smoke, `docs/returns/P9.3-smoke.md`) and `4dcbedc` (H.6's sessions, `docs/returns/P9.3-sessions.md`).

## I.1 Verified by the coordinator, from disk
- **H.3's smoke root**, read by the coordinator's own script (stdlib only; it imports nothing of your branch):
  - 100 training records, exactly the 100 cells; 100 checkpoints; 0 `.partial`.
  - Every checkpoint's sha256 equals its record's `file_sha256`.
  - Every record: `git` = `/home/filip/rltraffic-p93-run-b` at `14a6ba2`, not dirty; device `cuda`; 2 of 2 steps; `scenario_id`
    `cityflow4x4`.
  - 125 evaluation records over the 25 cells on draw 1000. All 125 have `reproduces` true and `att_ours` `==` route (i)'s
    `att_horizon` `==` route (ii)'s `att_ours`.
  - Each of the 100 learner records names a training record at `14a6ba2` with the same sha256. There is no `eval_refused/`.
- **Your seven files at `4dcbedc`,** in a scratch clone, one process, one thread: **159 passed, 0 skipped** (308 s).
- **The `ast` proof, by the coordinator's own script:**
  - Against `14a6ba2`, `offline/hz4x4_phase_b.py` changes exactly `_run_train_queue`, `build_parser` and `run_train_queue`. It adds
    exactly your eight names and removes none. The other 47 top-level definitions are identical.
  - `offline/hz4x4_ladder.py` is byte-identical.
  - The only other file of `offline/` that differs is `tier_sweep.py`: M1b, `train_tier_dt`'s tier check. hz4x4's path never calls
    `train_tier_dt`.
- **The device path.** The per-cell command (`run_cell` inside `_run_train_queue`) is unchanged. So `--device` (default `cuda`)
  reaches each `train` process exactly as in H.3's CUDA smoke. The session layer never touches the device, so the queue smoke on the
  CPU is valid evidence for it.
- **Mutants, the coordinator's own run on the session tests** (control: 7 passed):

  | Mutant | Result |
  |---|---|
  | H1: the plan ignored | KILLED |
  | H2: the stop file ignored | KILLED |
  | H3b: one cell started after the stop file is seen | KILLED, by the stop-file test (`ran == SESSION_CELLS[:1]`) |
  | C1: the stop file at `<root>/STOP` | KILLED, by the literal path pin |
  | C2: `dt_nomix`'s and `iql_m`'s timings swapped, the total unchanged | KILLED |
  | The check moved after the cell's bookkeeping | SURVIVED: near-equivalent (see below) |

  The surviving mutant differs from your code only in two cases: the file appears between the start refusal and the first cell, or
  during the last planned cell (only the summary's status changes). It is not a gap.
- **From the run tree, on the main roots, read-only:**
  - 100 cells pending, 0 skipped.
  - Session 1 at `--hours 6`: 20 cells, 5.80 planned hours, ending with `maxpressure/bc_top10_perix_m` seed 505. This equals the
    coordinator's own replay of the rule (30.67 h in all).
  - Both refusals fire and write nothing: a queue with no bound, and two steps on the main output root.
  - `output/p9_3/STOP` is absent.
- **I.4's session command itself,** run once on a scratch root with one two-step CPU cell:
  - It printed the start stamp, the run tree's commit, A27(d)'s check, the plan, the cell, the summary and the end stamp, then
    `QUEUE EXIT 0`.
  - With a stop file present: the refusal, `QUEUE EXIT 1`, and the file kept.
- **A27(d)'s check, read-only:** 200 episodes, 200 joint, and 200 distinct on each of the 16 intersections, in 1.8 s.

## I.2 Two statements in your notes are wrong: correct them (the author's checks)
1. **The overlap with P9.1 (`P9.3-smoke.md`, the H.5 section).** P9.1's campaign stage ran **16:11:49–16:30:31 CET**: its capture's
   stamp, and the last sample of its own GPU trace. Your watcher saw its process until 16:30:06 and not at 16:30:46. The author stopped
   it, and P9.1 has trained nothing since. Of the smoke's 100 cells:
   - 34 finished before it started;
   - **23 ran beside it** (24 if its end is put at 16:30:46: `mappo1000/bc_m` seed 303 started at 16:30:38);
   - 7 ran beside nothing;
   - **36 ran beside a GPU load that was NOT P9.1's.** From 16:38:08 CET your watcher shows 4–9 GB at 80–100 % until at least
     17:28:42, its last line. P9.1's own trace never exceeded 3.7 GB.

   The note's *"66 of the 100 cells ran … at the same time as P9.1's campaign"* and *"its training presumably runs in another process
   namespace"* are wrong. Between 16:36 and 20:15 the coordinator found no file written under `/home/filip` or the sessions' `/tmp` by
   anything but your smoke, your tests and the coordinator's docs. WSL's `nvidia-smi` reports the whole card, Windows programs
   included. **The load is unidentified.**
2. **The queue smoke's stated reason (`P9.3-sessions.md`, H.6-6).** *"P9.1's campaign was training on the GPU"* is wrong at 17:53:
   P9.1 had stopped at 16:30:31. What you saw was that unidentified load, so say so. The CPU choice weakens nothing (I.1, the device
   path).
   - The same applies to the smoke note's *"The evaluation ran on a machine loaded by P9.1's campaign"*. At 17:22–17:28 the load was
     the unidentified one plus your own four shards.
- **The rule this applies, so it is not repeated:** a statement about another stream's process comes from that stream's own record
  (its capture, its GPU trace, its run records), never from a reading of the card's load.
- **How:** one docs-only commit on your branch that corrects both notes with these numbers and quotes this section. No code changes.
  The run tree stays at `4dcbedc`.

## I.3 Rulings
1. **D: yes, at no cost.** Every session's command runs A27(d)'s registered check from the run tree, and the queue starts only if it
   passes (I.4). The check is `a27_arms.assert_distinct_schedules(<corpus>/cf_hz4x4__random_fresh, expected=200)`. It only reads,
   and takes about 2 s.
   - Why: the grid's route re-runs the check inside every `random_fresh` training call. Running it at each session's start keeps
     "run before any training" true without touching the frozen module.
   - The capture keeps every session's result.
2. **E: accepted as it stands.** No new field, no new run tree. Three things prove the evaluation records' code:
   - every evaluation command runs from the run tree named in its clearance, and its capture opens with that tree's commit and clean
     status, printed by the command itself;
   - every learner record's `policy_source` names a training record carrying `4dcbedc`, not dirty;
   - G4's `git diff --quiet 4dcbedc <G4 tip> -- offline/hz4x4_phase_b.py offline/hz4x4_ladder.py`.

   This is the standard P9.1's run records meet, and they carry no commit either.
   - **One addition for G4, tests first, in `offline/hz4x4_report.py`:** `training_block` refuses a training record whose `git` is not
     `{"commit": "4dcbedc25021634c199897beffeab4d806b9afc2", "dirty": false, "code_root": "/home/filip/rltraffic-p93-run-c"}`.
     Mutant: any commit accepted.
3. **A-B5: accepted.** `random@mappo1000`'s counterpart is the grid's shared random anchor (`tier_statements.ANCHOR_CELL`, P5.1's
   random cell, filed under `ANCHOR` in P5.2c's books). It is copied per episode with its path and digest, like the other six.
   - Why: a uniform-random policy does not depend on the tier. The grid evaluated it once, and P5.2c's books use that one cell for
     every tier, so it is the grid's only random reference.
   - The old-cells artifact labels it *"the grid's shared random anchor (one cell for every tier)"*, so no reader takes it for a
     `mappo1000` cell.
4. **The GPU order** (replaces H.4's *"P9.1's trainings, then yours"* and H.5's last sentence). P9.1 went first because of the hard
   stop: point 8 was to be cut first. The hard stop is withdrawn (D8).
   - **One campaign training session on the GPU at a time, never two.** H.5's ban stands.
   - Your sessions start now, while P9.1 builds its own.
   - Once P9.1's sessions are cleared (`BRIEF_46` Amendment F), its first training session goes next, after the session of yours then
     running ends. P9.2's stage 2 waits for its ten records.
   - After that, the coordinator names the next session in each turn.
5. **The evaluation commands come in Amendment J,** once `maxpressure`'s 25 cells exist (after session 2). This replaces H.3-5's "all
   three in Amendment I".
   - Why: they get a new form. The shards run in the foreground of one pane (`xargs -P`), so Ctrl+C reaches every worker.
   - A `--label` is never reused. `run_with_restarts` opens `<label>.attemptNN.log` for writing, so a reused label would overwrite a
     run's logs under `output/`.
   - The coordinator smokes that form on a scratch root first, and sets the worker counts (your 4 and 6) against what runs beside
     them.
   - Each evaluation run is well under a session (≈ 0.5–1.3 h). It ends by itself and resumes from its atomic records (§7 (5), as
     clarified today), so it needs no session control.
6. **H.2-4's freeze starts now.** From this token on, `offline/hz4x4_phase_b.py` and `offline/hz4x4_ladder.py` on your branch stay
   byte-identical to `4dcbedc`'s.

## I.4 ✅ THE TOKEN: `p93-train`, in sessions, from `/home/filip/rltraffic-p93-run-c` at `4dcbedc`
**The run tree:** created by the coordinator at 20:37:40 on 2026-10-28, detached at `4dcbedc25021634c199897beffeab4d806b9afc2`, and
clean. `offline/hz4x4_phase_b.py` has sha256 `5ee2edab…` and `offline/hz4x4_ladder.py` `bd305b2c…`, equal to the commit's.

**The session command** is the same for every session, since each resumes from disk. Run it in the author's tmux `p93`, only while no
other campaign training is on the GPU:
```
R=/home/filip/rltraffic-p93-run-c; cd /home/filip/rltraffic && ( date '+session start %F %T %Z'; git -C $R log -1 --format='run tree %H'; git -C $R status --porcelain; test -z "$(git -C $R status --porcelain)" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R /home/filip/rltraffic/.venv/bin/python -P -c "from offline.a27_arms import assert_distinct_schedules as a; print('A27(d) ok:', a('/home/filip/rltraffic/datasets_v11/cf_hz4x4__random_fresh', expected=200))" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R /home/filip/rltraffic/.venv/bin/python -P -m offline.hz4x4_phase_b train-queue --hours 6; s=$?; date '+session end %F %T %Z'; exit $s ) 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_3/train_capture.txt; echo "QUEUE EXIT ${PIPESTATUS[0]}"
```

**The sessions at `--hours 6`.** Planned hours are G3b's quiet CUDA estimates, in the coordinator's replay of the rule:

| Session | Cells | Planned hours | Contents |
|---|---|---|---|
| 1 | 20 | 5.80 | `maxpressure`, all but `iql_m` |
| 2 | 9 | 5.76 | `maxpressure/iql_m` × 5; `fixedtime/dt_nomix` seeds 101–404 |
| 3 | 23 | 5.72 | the rest of `fixedtime`; `mappo1000/dt_nomix` seeds 101–202 |
| 4 | 23 | 5.72 | the rest of `mappo1000` |
| 5 | 20 | 5.80 | `random_fresh`, all but `iql_m` |
| 6 | 5 | 1.87 | `random_fresh/iql_m` |

The author may set H per session; for example, `--hours 3` makes session 1 three `dt_nomix` cells, 2.92 planned hours.

- **Done when, for each session:** the capture ends with `train-queue session: {'status': 'session_done', …}` (`'completed'` after the
  last one), `session end …` and `QUEUE EXIT 0`. Session 1's summary names `next_pending` `['maxpressure', 'iql_m', 101]` and
  `planned_hours_left` 24.87.
- **To end a session early:** `touch /home/filip/rltraffic/output/p9_3/STOP`. The session ends after the cell in flight, which is up
  to about an hour while `dt_nomix` cells run.
  - The file is the author's. The code never deletes it, and no session starts while it exists. Remove it by hand before the next
    session: `rm /home/filip/rltraffic/output/p9_3/STOP`.
- **Ctrl+C is the emergency stop only.** It loses the cell in flight, and since the trainer writes only at the end, nothing is left
  behind to refuse. The one exception: if the interrupt lands between a checkpoint and its record, the next session refuses that cell
  and the coordinator rules.
- **A slower GPU only lengthens a session.** The plan is in estimates, and the bound is not a clock. If a session runs long, use STOP.
- **A failure stops the session at the failed cell** (`'status': 'failed'`, `QUEUE EXIT 1`). Report it; nothing is re-run without a
  ruling.

## I.5 Meanwhile, for you
- The I.2 correction commit (docs only).
- I.3-2's `training_block` check and I.3-3's label, tests first, in `offline/hz4x4_report.py`.
- Nothing on the GPU while a session runs. Report "P9.3 corrections done" with the commits. G4 waits for the training and Amendment
  J's evaluation.

## I.6 2026-10-28, night: your training moves to MACHINE B if its pre-flight passes; I.4 is SUSPENDED on the laptop until then
The author added a second computer, machine B: WSL2 on Windows, an RTX 3070 Laptop GPU with 8 GB and a 100 W power cap. Your session 1
had not started on the laptop.
- **Ruling: your WHOLE training (the 100 cells) runs on machine B, if B's pre-flight passes. P9.1 stays whole on the laptop.** The
  procedure and the pass criteria are in `docs/notes/MACHINE_B.md`.
  - **Why yours:** your cells are all new, so no contrast pairs a cell trained on B with one trained on the laptop. P9.1's contrasts are
    against registered cells trained on the laptop, and they keep that hardware.
  - **Why whole:** a campaign split across two GPUs would put the hardware inside its own comparisons.
  - A27 names no hardware (checked in its row). The move is a methods fact, disclosed in G4 and in plan §2, not a deviation.
- **Until the pre-flight decides, I.4's token is SUSPENDED on the laptop.** No `p93-train` session starts there. If machine B fails
  its pre-flight, I.4 applies on the laptop as written, after P9.1's training.
- **Machine B is made identical to the laptop except for its GPU:**
  - Ubuntu 26.04, with the laptop's own Python 3.12.13 packages repacked;
  - the laptop's venv, checked file by file;
  - the code from a git bundle, with the run tree at `4dcbedc`;
  - the eight corpora, checked file by file.
- **Its pre-flight** (about 1 hour, run by the author):
  - your seven test files at `4dcbedc`;
  - a 20-cell CUDA smoke: every (tier, arm), 2 steps;
  - one CPU cell's canonical digest beside the laptop's;
  - G3b's timing at 2,000 steps per arm, with a GPU trace.
- **After it, in I.7:**
  - B's session command adds the host's name and its GPU line to the capture.
  - `--hours` is divided by B's measured factor, because your planner's estimates are the laptop's.
  - Each finished tier comes back to the laptop with a sha256 manifest made on B.
  - G4 discloses the training machine.
- **For you now:** nothing changes in I.5. Do nothing on machine B.
- **Later the same night: the author's rulings, which replace the method above** (`docs/notes/MACHINE_B.md`, rewritten):
  - **Method.** The code comes from GitHub: `task/p9.3-a27c` is pushed, and the run tree is at `4dcbedc`. The venv is built on B from
    `requirements-frozen.txt`'s exact pins, each with the sha256 of the wheel the laptop has installed, and CityFlow is built from the
    vendored tree. Only the eight corpora and the six Python 3.12.13 packages travel, through Google Drive, with sha256 manifests
    checked on both sides.
  - **Gate.** There is no speed rule: your training moves to B whatever B's speed. Only a pre-flight failure keeps it on the laptop:
    a test failing, a cell erroring or running out of memory, or the CPU cell differing from the laptop's. The timings size B's
    sessions.
  - **CPU arithmetic.** On B, MKL, torch, oneDNN and numpy are held to the laptop's AVX2 kernel level, in the pre-flight and in every
    session. The laptop's CPU has no AVX-512, and there the setting leaves the CPU cell's digest unchanged.
- **2026-10-29: machine B PASSES its pre-flight under the author's ruling, and your training will run there. I.4 stays suspended on
  the laptop.** The coordinator verified every point from B's output (`docs/notes/MACHINE_B.md`, "Machine B's pre-flight").
  - **Steps 1–3 passed** (00:04–00:43):
    - every identity check was equal;
    - your seven files: 138 passed, 21 skipped, 0 failed. Each skip names laptop-only data that training does not read: the held-out
      draws, the grid's corpus and its evaluation output, P2.1's results, and the v1.1 draws;
    - all 20 CUDA smoke cells trained.
  - **Step 4's bitwise gate failed.** B's CPU, an AMD Ryzen 5 3600X, gave `84da36bc…` at both kernel levels, with byte-identical
    checkpoints, against the laptop's `4c03f73d…`.
  - **The author's ruling:** the bitwise gate is replaced by a numerical comparison of the CPU checkpoints, tensor by tensor, plus the
    losses. If the differences are at floating-point rounding size, B passes and P9.3 trains on B.
  - **The comparison** is `docs/notes/machine_b/machine_b_compare.py`, committed in `b029604` before it ran.
    - Its bound: every tensor within 1e-5 absolute, a tenth of one AdamW step at learning rate 1e-4; the loss within 1e-5 relative.
    - The result: the largest difference is 1.7e-7, the step-2 loss is bitwise equal (2.0727832317352295), and the corpus-derived
      contents are equal. **Within the bound.**
  - **Two corrections, both the coordinator's errors:**
    - B's GPU is a desktop NVIDIA GeForce RTX 3070, not the "RTX 3070 Laptop GPU" of this section's first lines.
    - The CPU-arithmetic bullet above is wrong where it implies that holding AVX2 makes the gate test only the software. The libraries
      choose kernels by CPU model, too. The kernel settings leave B's sessions.
  - **Next:** B's timing (`machine_b_timing.sh`, about 30 minutes). It also gives B's first-step loss and prints B's sessions. Then
    **I.7: B's session command.**
  - **For you:** nothing changes in I.5. The records your queue writes on B carry the same `git` block, since the run tree's path and
    commit are the same on B. Each checkpoint's runtime provenance names B's GPU.

## I.7 2026-10-29: B's timing VERIFIED; by the author's ruling, your training runs on machine B as ONE continuous run of about 48.3 h
**Verified by the coordinator, from B's output** (`docs/notes/MACHINE_B.md`, "Machine B's timing"). The tar's sha256 `9ef8d497…` was
checked, and the extracted folder equals the tar.
- `machine_b_timing.sh` ran at `13b2994` on B, 11:46:39–12:03:24 CET, and ended `TIMING EXIT 0`. The run tree was at `4dcbedc`,
  clean. Every record names it, not dirty, and every checkpoint's sha256 is its record's.
- **B's whole training takes 48.26 h** by B's measured cells, against G3b's 30.67 h on the laptop: a factor of 1.57 (1.37 for
  `dt_nomix`, about 2 for `bc_m` and `iql_m`, setup included). Each tier takes 12.07 h. Setup per cell agrees with step 3's smoke
  cells: 77–83 s for `dt_nomix`, 104–111 s for the others.
- **GPU memory:** at most 6,467 of 8,192 MiB, in `bc_m` (the whole card).
- **The first step's loss:** B 2.1332204341888428, the laptop 2.133220911026001. They are two float32 ulps apart (2.2e-7
  relative). The 1-step checkpoint is within the rounding bound, with a largest difference of 1.7e-7 (`cpu_1step_comparison.txt`).
  The second step's loss was bitwise equal, so **both steps agree to rounding.**
- **What differs in B's files:** the timings and load averages in the records, and in each checkpoint's runtime provenance, B's GPU
  and B's main-tree commit. The format, the `git` block, the paths under `output/p9_3` and every other field are the laptop's.

**The author's ruling (2026-10-29):** *"B is a desktop, so the laptop's session rule doesn't bind it. Run P9.3's whole training on B
preferably as one continuous run (about 48 h); if the queue or a safety reason argues for splitting, at most 2 to 4 sessions."* Then:
*"2 splits are ok or 4."* The STOP file still ends a session early, after its cell in flight. The laptop keeps its session rule for
P9.1.

**The coordinator: one run, because nothing in the queue or the checks argues for splitting.**
- The queue puts no upper bound on `--hours` (`plan_session` at `4dcbedc`). `--hours 31` plans all 100 cells; the queue counts the
  laptop's estimates, 30.67 h for all of them.
- Each cell trains in its own `train` process, so a long session carries nothing from one cell to the next.
- An interrupted run resumes from disk with the same command and loses at most the cell in flight (up to 1.33 h).
- **If the author splits anyway:** the same command with `--hours 16` makes 2 sessions of 24.13 h, and `--hours 8` makes 4 of
  12.07 h. Each ends at a tier boundary, in the coordinator's replay of the planner.

**The command,** for B's WSL in tmux. **It runs 48.3 h without a break.** It is I.4's command plus four changes: it creates
`output/p9_3`, prints the host and GPU line, uses `--hours 31`, and writes its own capture name:
```
R=/home/filip/rltraffic-p93-run-c; mkdir -p /home/filip/rltraffic/output/p9_3 && cd /home/filip/rltraffic && ( date '+session start %F %T %Z'; echo "host $(hostname) | $(nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader)"; git -C $R log -1 --format='run tree %H'; git -C $R status --porcelain; test -z "$(git -C $R status --porcelain)" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R /home/filip/rltraffic/.venv/bin/python -P -c "from offline.a27_arms import assert_distinct_schedules as a; print('A27(d) ok:', a('/home/filip/rltraffic/datasets_v11/cf_hz4x4__random_fresh', expected=200))" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R /home/filip/rltraffic/.venv/bin/python -P -m offline.hz4x4_phase_b train-queue --hours 31; s=$?; date '+session end %F %T %Z'; exit $s ) 2>&1 | tee -i -a /home/filip/rltraffic/output/p9_3/train_capture_machine_b.txt; echo "QUEUE EXIT ${PIPESTATUS[0]}"
```
- **Tested on the laptop** at 13:06–13:07, on a scratch root. Four arguments were added (`--max-cells 1 --gradient-steps 2 --device
  cpu --output-root /home/filip/rltraffic/output/machine_b/i7_command_test`), and the two `output/p9_3` paths moved under that root.
  - It printed the session start, the host line, the run tree, A27(d), the plan, one cell `exit 0`, `session_done`, the session end
    and `QUEUE EXIT 0`.
  - The cell's canonical digest is `4c03f73d…`, the laptop's 2-step CPU reference.
- **Done when:** the capture ends with `train-queue session: {'status': 'completed', …}` and `session end …`, and the pane shows
  `QUEUE EXIT 0`. The summary's `next_pending` is `None` and its `planned_hours_left` 0.0.
- **STOP, Ctrl+C and a failure:** as in I.4. The paths are the same on B.
- **After an interruption** (a crash, a restart or STOP), the same command resumes, skipping every finished cell. A half-written cell
  is refused: report it.
- **Before the run, on B:**
  - sleep and hibernation off (B11);
  - Windows Update paused;
  - the WSL terminal left open;
  - no GPU-heavy program on B while the run lasts.
- **Back to the laptop:**
  - The whole `output/p9_3` goes in one tar with its sha256, through Drive.
  - It is staged in `/home/filip/machine_b_result/p9_3` and checked by the coordinator as in I.1.
  - Only then is it extracted into the laptop's `output/` with `-k`, so no file is overwritten. The capture keeps its name.

**For you:**
- Nothing changes in I.5.
- I.4's token does not apply on the laptop: P9.3's training is on B.
- In G4, disclose the machines:
  - every training cell ran on machine B (desktop GeForce RTX 3070, driver 591.74, 100 W limit; AMD Ryzen 5 3600X);
  - the evaluation ran on the laptop's CPU;
  - the CPU cell agrees with the laptop's to rounding, not bitwise.

**2026-10-29, 13:51: the author started B's run as TWO sessions** with `--hours 16`, which the ruling allows.
- **Session 1:** 50 cells, `maxpressure` and `fixedtime`, about 24.1 h on B. It ends at about 14:00 on 2026-10-30.
  - Done when its capture ends with `'status': 'session_done'`, with `next_pending` `['mappo1000', 'dt_nomix', 101]` and
    `planned_hours_left` 15.34, then `session end …`, and the pane shows `QUEUE EXIT 0`.
- **Session 2:** the same command; the other 50 cells, `mappo1000` and `random_fresh`, about 24.1 h. Done as above, with
  `'status': 'completed'`.
- The transfer back follows session 2.

## I.8 2026-10-30: B's run was cut by a Windows Update restart at 10:37 and resumed at 11:48:12 as ONE run of the 56 remaining cells, about 27.9 h on B
**The author's report, from B:**
- **The restart:** Windows event 1074 at 10:37, from `svchost` on behalf of SYSTEM, with the reason "Operating System: Service pack
  (Planned)" and code `0x80020010`. That is a Windows Update restart, not a crash and not heat. Session 1 had run about 21 h
  without a problem.
- **44 of the 100 cells had finished:** `maxpressure` 25 and `fixedtime` 19, through `bc_top10_perix_m` seed 404.
  - The cell in flight, `fixedtime/bc_top10_perix_m` seed 505, was lost.
  - `output/p9_3` on B holds 44 `hz4x4_*.pt` checkpoints, none under 10 kB, and no `.partial` file.
- Sleep is now off, set in an admin PowerShell.
- **The author resumed at 11:48:12** with the same command and `--hours 31`. The queue planned the 56 remaining cells, from
  `fixedtime/bc_top10_perix_m` seed 505. There is no separate session 2.

**What the resume itself shows** (the code at `4dcbedc`, read by the coordinator):
- Before it plans, the queue recomputes the sha256 of every checkpoint that has a record, and compares it with the record's
  `file_sha256` (`pending_training_cells`). A mismatch, a record without its checkpoint, or a checkpoint without its record
  refuses the start.
- It planned from seed 505, so **all 44 finished cells passed that check after the restart.** The restart lost no finished cell
  and changed none.
- The lost cell trains from scratch with its own seed. `train_cell` removes any `.partial` before its first step, and writes the
  checkpoint and the record only at the end.

**The run now.** One continuous run is the form the author's ruling of 2026-10-29 prefers on B, and I.7's two-session note no
longer applies.
- **It runs about 27.9 h without a break**, by B's measured cells: one `bc_top10_perix_m` (0.07 h) and five `iql_m` (0.73 h each),
  then `mappo1000` and `random_fresh` at 12.07 h each. It ends at about 15:40 on 2026-10-31.
- **Done when** the capture ends with `train-queue session: {'status': 'completed', …}`, its `next_pending` `None` and
  `planned_hours_left` 0.0, then `session end …`, and the pane shows `QUEUE EXIT 0`.
- The capture holds session 1's lines, cut at 10:37 without a `session end`, then the resumed run's, from `session start
  2026-10-30 11:48:12`.
- **I.7's precondition was not in effect:** it asked for Windows Update paused before the run. The author is asked to pause it for
  the rest of the run, and to move any pending restart past the run's end.

**What the coordinator adds to I.1's check at "P9.3 on B done":**
- 100 records and 100 checkpoints, each checkpoint's sha256 its record's;
- `fixedtime/bc_top10_perix_m` seed 505 exactly once, with its files' times after 11:48:12;
- no file of a cell written between 10:37 and 11:48:12.

**For your G4:** disclose that B's run was cut once by an operating-system restart: the cell in flight was lost and trained again
from scratch with its seed, and no finished cell was trained again.

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
